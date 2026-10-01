"""One-pass decision student: LM-head letter readout, KL to soft targets, sub-question supervision.

Every question (the final decision and each sub-question) is its own sequence that repeats the
problem text; hybrid linear-attention models (Qwen3.5) ignore attention masks, so we do not pack
questions with a block-causal mask. Sequences are right-padded; the readout is the next-token
distribution at the last real token, restricted to the option-letter tokens.
"""
from __future__ import annotations

import dataclasses
import math
import random
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from .metrics import calibration  # noqa: F401  (re-exported for scripts)

SUBQ_TEMPLATE = "{problem}\n\nIntermediate question: {question}\nOptions:\nA) {a}\nB) {b}\nAnswer:"
FINAL_TEMPLATE = "{problem}\n\nAnswer:"
RATIONALE_PREFIX = "{problem}\n\nReasoning:"          # rationale-LM baseline: CoT text, then the answer


@dataclass
class Example:
    text: str
    labels: list[str]            # option letters, e.g. ["A", "B", "C"]
    target: list[float]          # soft target over labels (sums to 1)
    weight: float
    kind: str                    # "final" | "subq"
    item_id: str
    gold: int | None = None      # index of the gold label (final questions, if known)
    continuation: str | None = None   # kind "lm": text scored token by token after `text`


def letter_token_ids(tokenizer, letters: list[str]) -> list[int]:
    """Token ids of ' A', ' B', ... (the token that follows 'Answer:'); each must be a single token."""
    ids = []
    for L in letters:
        t = tokenizer.encode(" " + L, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"label ' {L}' is not a single token: {t}")
        ids.append(t[0])
    return ids


def _soft(p_yes: float, yes_first: bool, eps: float = 1e-4) -> list[float]:
    p = min(max(p_yes, eps), 1 - eps)
    return [p, 1 - p] if yes_first else [1 - p, p]


_OPT_LINE = re.compile(r"^([A-Z])\) (.*)$")


def permute_options(prompt: str, labels: list[str], target: list[float], gold: int | None,
                    rng: random.Random) -> tuple[str, list[float], int | None] | None:
    """Shuffle the lettered options of a prompt that ends in an 'Options:' block; the target mass and the gold
    index follow the option texts. "None of the above" stays last and ordered scales (options starting with a
    digit) are left alone. Returns None when the prompt does not have exactly the expected option lines."""
    head, sep, tail = prompt.rpartition("\nOptions:\n")
    if not sep:
        return None
    ms = [_OPT_LINE.match(line) for line in tail.split("\n")]
    if len(ms) != len(labels) or not all(ms) or [m.group(1) for m in ms] != labels:
        return None
    texts = [m.group(2) for m in ms]
    if all(t[:1].isdigit() for t in texts):
        return None
    free = [i for i, t in enumerate(texts) if not t.lower().startswith("none of the above")]
    pinned = [i for i in range(len(texts)) if i not in free]
    perm = rng.sample(free, len(free)) + pinned          # new position i shows old option perm[i]
    body = "\n".join(f"{L}) {texts[j]}" for L, j in zip(labels, perm))
    return head + sep + body, [target[j] for j in perm], (perm.index(gold) if gold is not None else None)


def build_examples(item: dict, *, final: str, subq: str, subq_target: str, lambda_sub: float,
                   rng: random.Random, rationale_lm: bool = False, lambda_lm: float = 1.0,
                   permute_final: float = 0.0, subq_k: int = 0, subq_weight: str = "split",
                   subq_frac: float = 0.0) -> list[Example]:
    """Turn one student-data record into training examples for a given arm.

    final: "none" | "teacher" (teacher answer distribution) | "gold" (one-hot gold)
    subq: "none" | "cot" (teacher-extracted sub-questions) | "random" (matched generic questions) |
          "mix" (as many sub-questions as "cot", half CoT nodes and half matched controls)
    subq_target: "fresh" (teacher answer without CoT) | "cot" (with CoT) | "truth" (program truth,
                 falling back to the CoT answer, then the fresh one, when no truth is available) |
                 "commit" (the teacher's value-commitment confidence where the CoT settled the node)
    rationale_lm: add a token-level LM example on the teacher's reasoning and answer (DHRD-style baseline)
    permute_final: probability of showing the final question with its options in a new random order (the
                   target follows the option texts), so the student cannot lean on letter positions
    subq_k: if > 0, use at most this many (randomly drawn) sub-questions per item and epoch
    subq_weight: "split" (each of the n sub-questions weighs lambda_sub / n) | "each" (each weighs lambda_sub)
    subq_frac: if > 0, use a random ceil(frac * n) of the item's sub-questions per epoch
    Every example's weight is multiplied by item["weight"] (default 1; e.g. per-category balancing weights).
    """
    wi = item.get("weight", 1.0)
    ex: list[Example] = []
    labels = item["labels"]
    gold = labels.index(item["gold_label"]) if item.get("gold_label") in labels else None
    target = None
    if final == "teacher" and item.get("teacher"):
        t = [item["teacher"].get(L, 0.0) for L in labels]
        s = sum(t)
        if s > 0:
            target = [x / s for x in t]
    elif final == "gold" and gold is not None:
        target = [1.0 if i == gold else 0.0 for i in range(len(labels))]
    if target is not None:
        prompt, fgold = item["prompt"], gold
        if permute_final > 0 and rng.random() < permute_final:
            shuffled = permute_options(prompt, labels, target, gold, rng)
            if shuffled:
                prompt, target, fgold = shuffled
        ex.append(Example(FINAL_TEMPLATE.format(problem=prompt), labels, target, wi, "final", item["item_id"], fgold))
    if subq == "mix":            # same number of sub-questions as "cot": half CoT nodes, half matched controls
        cot = [dict(sq, _kind="cot") for sq in item.get("subqs", [])]
        ctl = [dict(sq, _kind="random") for sq in item.get("random_subqs", [])]
        n = len(cot)
        k = (n + 1) // 2
        pool = rng.sample(cot, k) + rng.sample(ctl, min(n - k, len(ctl)))
    else:
        pool = {"cot": item.get("subqs", []), "random": item.get("random_subqs", [])}.get(subq, [])
    usable = []
    for sq in pool:
        p = None
        if sq.get("_kind") == "random":      # controls in a mix keep their own (fresh) target
            p = sq.get("p_fresh")
        elif subq_target == "truth" and sq.get("truth") is not None:
            p = 1.0 if sq["truth"] else 0.0
        elif subq_target == "commit" and sq.get("p_commit") is not None:
            p = sq["p_commit"]
        elif subq_target in ("cot", "truth", "commit"):
            p = sq.get("p_cot")
        if p is None:
            p = sq.get("p_fresh")
        if p is not None:
            usable.append((sq["question"], p))
    if subq_k > 0 and len(usable) > subq_k:
        usable = rng.sample(usable, subq_k)
    if subq_frac > 0 and usable:
        usable = rng.sample(usable, max(1, math.ceil(subq_frac * len(usable))))
    w = wi * (lambda_sub if subq_weight == "each" else lambda_sub / max(1, len(usable)))
    for q, p in usable:
        yes_first = rng.random() < 0.5
        a, b = ("Yes", "No") if yes_first else ("No", "Yes")
        ex.append(Example(SUBQ_TEMPLATE.format(problem=item["prompt"], question=q, a=a, b=b), ["A", "B"],
                          _soft(p, yes_first), w, "subq", item["item_id"]))
    if rationale_lm and item.get("rationale") and item.get("teacher"):
        answer = max(item["teacher"], key=item["teacher"].get)
        ex.append(Example(RATIONALE_PREFIX.format(problem=item["prompt"]), labels, [], lambda_lm, "lm",
                          item["item_id"], continuation=f" {item['rationale'].strip()}\n\nAnswer: {answer}"))
    return ex


def node_depths(nodes: list[dict]) -> dict[str, int]:
    """Depth of each tree node: 0 for nodes that depend on no other node (facts read off the problem), else one
    more than the deepest node it depends on. Unknown ids and cycles count as depth 0."""
    by = {n["id"]: n for n in nodes if "id" in n}
    memo: dict[str, int] = {}

    def depth(i: str, seen: frozenset) -> int:
        if i in memo:
            return memo[i]
        if i in seen or i not in by:
            return 0
        deps = [x for x in (by[i].get("depends_on") or []) if x in by]
        memo[i] = 1 + max(depth(x, seen | {i}) for x in deps) if deps else 0
        return memo[i]
    return {i: depth(i, frozenset()) for i in by}


def build_stage_examples(item: dict, *, stage: int, n_stages: int, k: int, rng: random.Random,
                         subq_target: str = "cot", p_new: float = 0.5, weight: float = 1.0,
                         level_balanced: bool = False) -> list[Example]:
    """Depth-curriculum sub-questions for one item. Stage s (0-based, of n_stages) covers the item's tree up to
    level ceil((s+1)(M+1)/n_stages) - 1, where M is the item's deepest level, so every item reaches its own top in
    the last stage. Each of the k drawn nodes comes from the levels this stage adds with probability p_new, else
    from all levels covered so far (replay); stages that add no level replay only. level_balanced draws a covered
    level uniformly first (so the many depth-0 facts do not dominate replay); stage=n_stages-1 with p_new=0 is
    plain replay over the whole tree (used for node replay during the final stage)."""
    nodes = [sq for sq in item.get("subqs", []) if "id" in sq]
    if not nodes:
        return []
    d = node_depths(nodes)
    top = max(d.values())
    hi = math.ceil((stage + 1) * (top + 1) / n_stages) - 1
    lo = math.ceil(stage * (top + 1) / n_stages) - 1 if stage > 0 else -1      # levels <= lo were covered before
    covered = [sq for sq in nodes if d[sq["id"]] <= hi]
    new = [sq for sq in covered if d[sq["id"]] > lo]
    ex = []
    for _ in range(k):
        pool = new if new and rng.random() < p_new else covered
        if level_balanced:
            lv = rng.choice(sorted({d[x["id"]] for x in pool}))
            pool = [x for x in pool if d[x["id"]] == lv]
        sq = rng.choice(pool)
        p = sq.get("truth") if subq_target == "truth" and sq.get("truth") is not None else None
        p = (1.0 if p else 0.0) if p is not None else sq.get("p_cot" if subq_target != "fresh" else "p_fresh")
        if p is None:
            p = sq.get("p_fresh")
        if p is None:
            continue
        yes_first = rng.random() < 0.5
        a, b = ("Yes", "No") if yes_first else ("No", "Yes")
        ex.append(Example(SUBQ_TEMPLATE.format(problem=item["prompt"], question=sq["question"], a=a, b=b), ["A", "B"],
                          _soft(p, yes_first), weight * item.get("weight", 1.0), "subq", item["item_id"]))
    return ex


def _node_p(sq: dict, subq_target: str = "cot") -> float | None:
    if subq_target == "truth" and sq.get("truth") is not None:
        return 1.0 if sq["truth"] else 0.0
    p = sq.get("p_cot") if subq_target != "fresh" else sq.get("p_fresh")
    return sq.get("p_fresh") if p is None else p


def _fact_line(sq: dict) -> str | None:
    p = (1.0 if sq["truth"] else 0.0) if sq.get("truth") is not None else sq.get("p_cot")
    if p is None or 0.3 < p < 0.7:
        return None
    return f"Q: {sq['question']} A: {'Yes' if p >= 0.5 else 'No'}"


def build_transition_views(item: dict, *, k: int, rng: random.Random, shuffled: bool = False, keep_prob: float = 1.0,
                           weight: float = 0.5, subq_target: str = "cot") -> list[Example]:
    """Tree-transition auxiliary views: k nodes, each asked with the results of its parent nodes stated in the prompt
    (the local step parents -> child of the teacher's dependency tree). Nodes with parents are drawn first.
    shuffled=True is the edge-rewiring control: the same node, label and number of stated results, but the stated
    results come from nodes of the same item that are neither the node's parents nor its descendants. keep_prob drops
    each stated result independently (scaffold withdrawal; 0 gives plain node questions)."""
    nodes = [sq for sq in item.get("subqs", []) if "id" in sq]
    if not nodes:
        return []
    by = {sq["id"]: sq for sq in nodes}
    parents = {sq["id"]: [x for x in (sq.get("depends_on") or []) if x in by] for sq in nodes}
    children = {i: [j for j in by if i in parents[j]] for i in by}
    def descendants(i: str) -> set:
        out, todo = set(), list(children[i])
        while todo:
            j = todo.pop()
            if j not in out:
                out.add(j)
                todo += children[j]
        return out
    with_p = [sq for sq in nodes if parents[sq["id"]] and _node_p(sq, subq_target) is not None]
    without = [sq for sq in nodes if not parents[sq["id"]] and _node_p(sq, subq_target) is not None]
    pick = rng.sample(with_p, min(k, len(with_p)))
    pick += rng.sample(without, min(k - len(pick), len(without)))
    ex = []
    for sq in pick:
        given = [by[x] for x in parents[sq["id"]]]
        if shuffled and given:
            banned = {sq["id"], *parents[sq["id"]], *descendants(sq["id"])}
            pool = [x for x in nodes if x["id"] not in banned]
            given = rng.sample(pool, min(len(given), len(pool)))
        lines = [ln for ln in (_fact_line(x) for x in given) if ln and rng.random() < keep_prob]
        problem = item["prompt"] + (f"\n\n{FACTS_HEADER}\n" + "\n".join(lines) if lines else "")
        yes_first = rng.random() < 0.5
        a, b = ("Yes", "No") if yes_first else ("No", "Yes")
        ex.append(Example(SUBQ_TEMPLATE.format(problem=problem, question=sq["question"], a=a, b=b), ["A", "B"],
                          _soft(_node_p(sq, subq_target), yes_first), weight * item.get("weight", 1.0), "subq",
                          item["item_id"]))
    return ex


def _descendants(nodes: list[dict]) -> dict[str, set]:
    by = {sq["id"] for sq in nodes}
    children = {i: [sq["id"] for sq in nodes if i in (sq.get("depends_on") or [])] for i in by}
    out = {}
    for i in by:
        seen, todo = set(), list(children[i])
        while todo:
            j = todo.pop()
            if j not in seen:
                seen.add(j)
                todo += children[j]
        out[i] = seen
    return out


def full_tree_plan(item: dict, *, cap: int, rng: random.Random, subq_target: str = "cot") -> list[dict]:
    """The nodes of a full-tree group (at most `cap`, sampled if the tree is larger) with both fact sets:
    "parents" (true parent results) and "matched" (as many results from other nodes of the same item, depth-matched
    where possible, excluding the node, its parents, its descendants and the near-answer nodes: sinks at the maximum
    depth). A node gets facts in both modes only if it is eligible: it has parents, every parent has a confident
    answer, and enough matched replacements exist. Otherwise it is asked plainly in every mode, so the two tree
    modes differ only in which results are stated, never in how many."""
    nodes = [sq for sq in item.get("subqs", []) if "id" in sq]
    asked = [sq for sq in nodes if _node_p(sq, subq_target) is not None]
    if len(asked) > cap:
        asked = rng.sample(asked, cap)
    if not nodes:
        return []
    by = {sq["id"]: sq for sq in nodes}
    depth = node_depths(nodes)
    desc = _descendants(nodes)
    maxd = max(depth.values())
    sinks = {i for i in by if not any(i in (sq.get("depends_on") or []) for sq in nodes)}
    confident = {i for i in by if _fact_line(by[i])}
    plan = []
    for sq in asked:
        i = sq["id"]
        par = [x for x in (sq.get("depends_on") or []) if x in by]
        entry = {"node": sq, "parents": [], "matched": []}
        if par and all(x in confident for x in par):
            pool = [j for j in by if j in confident and j != i and j not in par and j not in desc[i]
                    and not (j in sinks and depth[j] == maxd)]
            if len(pool) >= len(par):
                rest, matched = pool[:], []
                for x in par:
                    same = [j for j in rest if depth[j] == depth[x]]
                    cand = same or sorted(rest, key=lambda j: abs(depth[j] - depth[x]))[:max(1, len(rest))]
                    j = rng.choice(same) if same else cand[0]
                    matched.append(j)
                    rest.remove(j)
                entry = {"node": sq, "parents": par, "matched": matched}
        plan.append(entry)
    return plan


def build_full_tree(item: dict, *, mode: str, rng: random.Random, cap: int = 10, aux_total: float = 1.0,
                    final: str = "teacher", subq_target: str = "cot", permute_final: float = 0.0) -> list[Example]:
    """One problem with ALL its tree nodes (up to `cap`) instead of k sampled ones: [final] + one view per node.
    mode "true": eligible nodes are asked with their parents' results stated; "matched": with the same number of
    depth-matched non-parent results (see full_tree_plan); "plain": every node asked without facts; "placebo": the
    plain views at zero weight; "control": the item's control questions (fresh targets), plainly. The auxiliary
    weight is normalised per problem (aux_total / number of views), so larger trees do not weigh more."""
    fin = [e for e in build_examples(item, final=final, subq="none", subq_target=subq_target, lambda_sub=0.0, rng=rng,
                                     permute_final=permute_final) if e.kind == "final"]
    if not fin:
        return []
    wi = item.get("weight", 1.0)
    views = []
    if mode == "control":
        pool = [sq for sq in item.get("random_subqs", []) if sq.get("p_fresh") is not None]
        if len(pool) > cap:
            pool = rng.sample(pool, cap)
        views = [(sq, [], "fresh") for sq in pool]
    else:
        plan = full_tree_plan(item, cap=cap, rng=rng, subq_target=subq_target)
        by = {sq["id"]: sq for sq in item.get("subqs", []) if "id" in sq}
        for e in plan:
            facts = {"true": e["parents"], "matched": e["matched"]}.get(mode, [])
            views.append((e["node"], [by[j] for j in facts], subq_target))
    out = list(fin)
    w = 0.0 if mode == "placebo" else aux_total / max(1, len(views))
    for sq, facts, tgt in views:
        lines = [_fact_line(x) for x in facts]
        rng.shuffle(lines)
        problem = item["prompt"] + (f"\n\n{FACTS_HEADER}\n" + "\n".join(lines) if lines else "")
        yes_first = rng.random() < 0.5
        a, b = ("Yes", "No") if yes_first else ("No", "Yes")
        out.append(Example(SUBQ_TEMPLATE.format(problem=problem, question=sq["question"], a=a, b=b), ["A", "B"],
                           _soft(_node_p(sq, tgt), yes_first), w * wi, "subq", item["item_id"]))
    return out


SUBQ_MC_TEMPLATE = "{problem}\n\nIntermediate question: Which of these questions has the answer Yes?\nOptions:\n{opts}\nAnswer:"
MC_NONE = "None of them"


def _mc_capacity(n_yes: int, n_no: int, m: int, k: int) -> int:
    """How many disjoint m-question views (at most one Yes each) the confident questions allow, up to k."""
    c = 0
    while c < k:
        if n_yes >= 1 and n_no >= m - 1:
            n_yes, n_no = n_yes - 1, n_no - (m - 1)
        elif n_no >= m:
            n_no -= m
        else:
            break
        c += 1
    return c


def mc_pool(item: dict, kind: str) -> tuple[list[dict], list[dict], str]:
    pool = item.get("subqs", []) if kind == "cot" else item.get("random_subqs", [])
    key = "p_cot" if kind == "cot" else "p_fresh"
    return ([sq for sq in pool if sq.get(key) is not None and sq[key] >= 0.7],
            [sq for sq in pool if sq.get(key) is not None and sq[key] <= 0.3], key)


def mc_limit(item: dict, k: int, m: int = 3) -> int:
    """Views per item for the matched multiple-choice arms: the smaller of the two pools' capacities, so the CoT and
    the control arm give every item the same number of weighted views."""
    caps = []
    for kind in ("cot", "random"):
        y, n, _ = mc_pool(item, kind)
        caps.append(_mc_capacity(len(y), len(n), m, k))
    return min(caps)


def build_mc_views(item: dict, *, k: int, rng: random.Random, kind: str = "cot", m: int = 3, weight: float = 0.5,
                   p_one: float = 0.75) -> list[Example]:
    """Multiple-choice auxiliary views: up to k views, each packing m of the item's yes/no questions (CoT nodes, or
    the control questions with their fresh targets) as "Which of these questions has the answer Yes?" with the
    questions as options plus a final "None of them". Only confident questions are used (p <= 0.3 or >= 0.7), at most
    one confident Yes per view: with probability p_one a view gets one Yes and m-1 No questions, otherwise m No
    questions (answer "None of them"). Views use disjoint questions. The soft target puts P(i) proportional to
    p_i * prod_{j != i} (1 - p_j) and P(none) to prod_j (1 - p_j), renormalised over these m + 1 events."""
    yes, no, key = mc_pool(item, kind)
    k = min(k, _mc_capacity(len(yes), len(no), m, k))
    rng.shuffle(yes)
    rng.shuffle(no)
    out = []
    for made in range(k):
        n_yes = 1 if (yes and rng.random() < p_one) else 0
        left = k - made - 1                          # keep the remaining views feasible
        if n_yes == 0 and (len(no) < m or _mc_capacity(len(yes), len(no) - m, m, left) < left):
            n_yes = 1
        if n_yes == 1 and (not yes or len(no) < m - 1 or _mc_capacity(len(yes) - 1, len(no) - m + 1, m, left) < left):
            n_yes = 0
        if len(no) < m - n_yes:
            break
        chosen = [yes.pop() for _ in range(n_yes)] + [no.pop() for _ in range(m - n_yes)]
        rng.shuffle(chosen)
        ps = [min(max(sq[key], 1e-4), 1 - 1e-4) for sq in chosen]
        none = math.prod(1 - p for p in ps)
        scores = [p * none / (1 - p) for p in ps] + [none]
        z = sum(scores)
        letters = [chr(65 + i) for i in range(m + 1)]
        opts = "\n".join(f"{L}) {t}" for L, t in zip(letters, [sq["question"] for sq in chosen] + [MC_NONE]))
        out.append(Example(SUBQ_MC_TEMPLATE.format(problem=item["prompt"], opts=opts), letters, [x / z for x in scores],
                           weight * item.get("weight", 1.0), "subq", item["item_id"]))
    return out


def build_group(item: dict, *, aux_kind: str, k: int, aux_weight: float, rng: random.Random, final: str = "teacher",
                subq_target: str = "cot", permute_final: float = 0.0, keep_prob: float = 1.0,
                aux_active: bool = True) -> list[Example]:
    """One problem for exposure-matched grouped training: [final] + exactly k auxiliary views. aux_kind "cot" uses
    CoT nodes, "random" the matched control questions (fresh targets), "placebo" the same CoT views with zero loss
    weight (same forward/backward work, no auxiliary signal), "tree" CoT nodes asked with their parents' results
    stated, "tree_shuf" the same with unrelated nodes' results stated (edge-rewiring control). Missing views are
    zero-weight copies of the final. aux_active=False keeps the views but gives them zero weight (placebo for this
    item), so per-family auxiliary supervision stays exposure-matched."""
    if aux_kind in ("cot_mc", "random_mc"):
        fin = [e for e in build_examples(item, final=final, subq="none", subq_target=subq_target, lambda_sub=0.0,
                                         rng=rng, permute_final=permute_final) if e.kind == "final"]
        if not fin:
            return []
        aux = build_mc_views(item, k=mc_limit(item, k), rng=rng, kind="cot" if aux_kind == "cot_mc" else "random",
                             weight=aux_weight if aux_active else 0.0)
        return fin + aux + [dataclasses.replace(fin[0], weight=0.0) for _ in range(k - len(aux))]
    if aux_kind in ("tree", "tree_shuf"):
        fin = [e for e in build_examples(item, final=final, subq="none", subq_target=subq_target, lambda_sub=0.0,
                                         rng=rng, permute_final=permute_final) if e.kind == "final"]
        if not fin:
            return []
        aux = build_transition_views(item, k=k, rng=rng, shuffled=aux_kind == "tree_shuf", keep_prob=keep_prob,
                                     weight=aux_weight if aux_active else 0.0, subq_target=subq_target)
        return fin + aux + [dataclasses.replace(fin[0], weight=0.0) for _ in range(k - len(aux))]
    sub, tgt = ("random", "fresh") if aux_kind == "random" else ("cot", subq_target)
    views = build_examples(item, final=final, subq=sub, subq_target=tgt, lambda_sub=aux_weight, rng=rng,
                           permute_final=permute_final, subq_k=k, subq_weight="each")
    fin = [e for e in views if e.kind == "final"]
    if not fin:
        return []
    aux = [e for e in views if e.kind == "subq"][:k]
    if aux_kind == "placebo" or not aux_active:
        aux = [dataclasses.replace(e, weight=0.0) for e in aux]
    return fin + aux + [dataclasses.replace(fin[0], weight=0.0) for _ in range(k - len(aux))]


FACTS_HEADER = "Known intermediate results:"


def fact_block(item: dict, min_depth: int = 0) -> str:
    """The item's tree nodes as stated results ("Q: ... A: Yes/No"), shallow to deep, keeping nodes of depth >=
    min_depth. The answer is the program truth when known, else the teacher's CoT-conditioned answer; nodes the
    teacher was unsure about (0.3 < p < 0.7) are left out. Returns "" when nothing is kept."""
    nodes = [sq for sq in item.get("subqs", []) if "id" in sq]
    if not nodes:
        return ""
    d = node_depths(nodes)
    lines = []
    for sq in sorted(nodes, key=lambda x: (d[x["id"]], x["id"])):
        if d[sq["id"]] < min_depth:
            continue
        p = (1.0 if sq["truth"] else 0.0) if sq.get("truth") is not None else sq.get("p_cot")
        if p is None or 0.3 < p < 0.7:
            continue
        lines.append(f"Q: {sq['question']} A: {'Yes' if p >= 0.5 else 'No'}")
    return f"\n\n{FACTS_HEADER}\n" + "\n".join(lines) if lines else ""


def build_fact_examples(item: dict, *, stage: int, n_stages: int, final: str = "teacher") -> list[Example]:
    """Fact-internalization curriculum: the final question with the item's tree nodes stated in the prompt. Stage s
    of n_stages drops the nodes shallower than ceil(s (M+1) / n_stages) (shallow facts are internalized first);
    stage n_stages (or any stage that drops everything) gives the plain final question. n_stages=0 keeps all."""
    nodes = [sq for sq in item.get("subqs", []) if "id" in sq]
    top = max(node_depths(nodes).values()) if nodes else 0
    cut = 0 if n_stages == 0 else math.ceil(stage * (top + 1) / n_stages)
    ex = build_examples(dict(item, prompt=item["prompt"] + fact_block(item, cut)), final=final, subq="none",
                        subq_target="cot", lambda_sub=0.0, rng=random.Random(0))
    return [e for e in ex if e.kind == "final"]


def text_parts(model):
    """(backbone returning last_hidden_state, output embedding) for causal or multimodal wrappers."""
    head = model.get_output_embeddings()
    inner = getattr(model, "model", model)
    body = getattr(inner, "language_model", None) or getattr(model, "language_model", None)
    if body is None:
        body = model.get_decoder() if hasattr(model, "get_decoder") else inner
    return body, head


def collate(tokenizer, batch: list[Example], max_len: int, device) -> dict:
    enc = [tokenizer.encode(e.text, add_special_tokens=False)[-max_len:] for e in batch]  # keep the question end
    L = max(len(x) for x in enc)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = torch.full((len(enc), L), pad, dtype=torch.long)
    mask = torch.zeros((len(enc), L), dtype=torch.long)
    for i, x in enumerate(enc):
        ids[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device),
            "last": torch.tensor([len(x) - 1 for x in enc], device=device)}


def label_logits(model, tokenizer, batch: list[Example], max_len: int, label_ids_cache: dict,
                 hidden_out: list | None = None, layers: list[int] | None = None) -> list[torch.Tensor]:
    """Logits over each example's option letters at the readout position (optionally also appends each
    example's readout hidden state, detached, to hidden_out; with layers, a [len(layers) + 1, hidden] stack of the
    readout position's states after those decoder layers plus the final readout state)."""
    body, head = text_parts(model)
    dev = head.weight.device
    b = collate(tokenizer, batch, max_len, dev)
    want_layers = bool(layers) and hidden_out is not None
    res = body(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=want_layers)
    hidden = res.last_hidden_state
    rows = torch.arange(hidden.shape[0], device=dev)
    h = hidden[rows, b["last"]]
    if hidden_out is not None:
        if want_layers:
            per = [res.hidden_states[l][rows, b["last"]] for l in layers] + [h]
            hidden_out.extend(torch.stack(per, 1).detach().float().cpu())
        else:
            hidden_out.extend(h.detach().float().cpu())
    out = []
    for i, e in enumerate(batch):
        key = tuple(e.labels)
        if key not in label_ids_cache:
            label_ids_cache[key] = torch.tensor(letter_token_ids(tokenizer, e.labels), device=dev)
        w = head.weight[label_ids_cache[key]]
        z = h[i].to(w.dtype) @ w.T
        if getattr(head, "bias", None) is not None:
            z = z + head.bias[label_ids_cache[key]]
        out.append(z.float())
    return out


def _ce_sum(h: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy((h.to(w.dtype) @ w.T).float(), y, reduction="sum")


def lm_loss(model, tokenizer, batch: list[Example], max_len: int, chunk: int = 512) -> torch.Tensor:
    """Sum over examples of weight x mean token cross-entropy on each continuation.

    The vocabulary projection runs in checkpointed chunks so full-vocabulary logits (248k for Qwen3.5)
    are never held for the whole sequence. Continuations longer than the budget keep their start and
    their last 64 tokens (where the answer is)."""
    body, head = text_parts(model)
    dev = head.weight.device
    seqs, starts = [], []
    for e in batch:
        pre = tokenizer.encode(e.text, add_special_tokens=False)[-(max_len // 2):]
        cont = tokenizer.encode(e.continuation, add_special_tokens=False)
        room = max_len - len(pre)
        if len(cont) > room:
            cont = cont[:room - 64] + cont[-64:]
        seqs.append(pre + cont)
        starts.append(len(pre))
    L = max(len(x) for x in seqs)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, x in enumerate(seqs):
        ids[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    ids, mask = ids.to(dev), mask.to(dev)
    hidden = body(input_ids=ids, attention_mask=mask).last_hidden_state
    total = hidden.new_zeros((), dtype=torch.float32)
    for i, (x, st) in enumerate(zip(seqs, starts)):
        h, y = hidden[i, st - 1:len(x) - 1], ids[i, st:len(x)]
        ce = sum(torch.utils.checkpoint.checkpoint(_ce_sum, h[j:j + chunk], head.weight, y[j:j + chunk],
                                                   use_reentrant=False) for j in range(0, len(y), chunk))
        total = total + batch[i].weight * ce / max(1, len(y))
    return total


def kl_loss(logits: list[torch.Tensor], batch: list[Example], smoothing: float = 0.0) -> torch.Tensor:
    """Weighted soft cross-entropy (= KL(target || p) + const). smoothing mixes each target with the uniform
    distribution over its options: t' = (1 - smoothing) t + smoothing / K (label smoothing; the teacher's
    targets are near one-hot)."""
    tot = 0.0
    for z, e in zip(logits, batch):
        t = torch.tensor(e.target, device=z.device)
        if smoothing:
            t = (1 - smoothing) * t + smoothing / t.numel()
        tot = tot + e.weight * -(t * F.log_softmax(z, -1)).sum()
    return tot / len(batch)


def brier_loss(logits: list[torch.Tensor], batch: list[Example]) -> torch.Tensor:
    """Weighted Brier score of the readout against the soft target (a proper scoring rule, minimised at p = t)."""
    tot = 0.0
    for z, e in zip(logits, batch):
        t = torch.tensor(e.target, device=z.device)
        tot = tot + e.weight * ((F.softmax(z, -1) - t) ** 2).sum()
    return tot / len(batch)


def load_model(path: str, dtype):
    import transformers
    try:
        m = transformers.AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    except Exception:
        m = transformers.AutoModelForImageTextToText.from_pretrained(path, torch_dtype=dtype)
    for n, p in m.named_parameters():
        if "visual" in n or "vision" in n:
            p.requires_grad = False
    return m


@torch.no_grad()
def evaluate(model, tok, items: list[dict], max_len: int, bs: int, cache: dict,
             hidden: list | None = None, layers: list[int] | None = None) -> tuple[dict, list[dict]]:
    """Final-question predictions for eval records (prompt, labels, gold_label; optional gold_probs, group).

    Items are batched by length to limit padding; predictions come back in input order. If `hidden` is a list,
    it receives each item's readout hidden state (float32, CPU), also in input order."""
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    order = sorted(range(len(items)), key=lambda i: len(items[i]["prompt"]))
    probs_by, hid_by = {}, {}
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        batch = [Example(FINAL_TEMPLATE.format(problem=items[i]["prompt"]), items[i]["labels"],
                         [0.0] * len(items[i]["labels"]), 1.0, "final", items[i]["item_id"]) for i in idx]
        hs = [] if hidden is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            zs = label_logits(model, tok, batch, max_len, cache, hidden_out=hs, layers=layers)
        for j, (i, z) in enumerate(zip(idx, zs)):
            probs_by[i] = torch.softmax(z, -1).tolist()
            if hs is not None:
                hid_by[i] = hs[j]
    if hidden is not None:
        hidden.extend(hid_by[i] for i in range(len(items)))
    probs, gold, preds = [], [], []
    for i, it in enumerate(items):
        p = probs_by[i]
        preds.append({"item_id": it["item_id"], "group": it.get("group"), "labels": it["labels"], "probs": p,
                      "gold_label": it.get("gold_label"), "gold_probs": it.get("gold_probs")})
        if it.get("gold_label") in it["labels"]:
            probs.append(p)
            gold.append(it["labels"].index(it["gold_label"]))
    if was_training:
        model.train()
    return (calibration(probs, gold) if gold else {"n": 0}), preds


def subq_sets(spec: str) -> list[tuple[str, str]]:
    """--eval-subq value -> [(result name, path)]: a bare path is scored as "subq"; "name=path" entries (comma
    separated) as "subq_<name>"."""
    out = []
    for part in (x.strip() for x in spec.split(",")):
        if part:
            name, _, path = part.rpartition("=")
            out.append((f"subq_{name}" if name else "subq", path))
    return out


@torch.no_grad()
def evaluate_subq(model, tok, items: list[dict], max_len: int, bs: int, cache: dict,
                  seed: int = 0) -> tuple[dict, list[dict]]:
    """Student accuracy on the sub-questions of held-out items (the mechanism check: does the one-pass
    student answer the intermediate judgements?). Target: program truth when known, else the teacher's
    CoT-conditioned answer. Also scores the matched control questions against the teacher's fresh answer.
    Broken down by node type and source; yes/no order randomised per question as in training."""
    rng = random.Random(seed)
    exs, meta = [], []
    for it in items:
        for kind, pool in (("cot", it.get("subqs", [])), ("control", it.get("random_subqs", []))):
            for sq in pool:
                if kind == "cot":
                    tgt = sq["truth"] if sq.get("truth") is not None else (
                        None if sq.get("p_cot") is None else sq["p_cot"] > 0.5)
                else:
                    tgt = None if sq.get("p_fresh") is None else sq["p_fresh"] > 0.5
                if tgt is None:
                    continue
                yes_first = rng.random() < 0.5
                a, b = ("Yes", "No") if yes_first else ("No", "Yes")
                exs.append(Example(SUBQ_TEMPLATE.format(problem=it["prompt"], question=sq["question"], a=a, b=b),
                                   ["A", "B"], [0.5, 0.5], 1.0, "subq", it["item_id"]))
                meta.append({"item_id": it["item_id"], "source": it.get("source"), "kind": kind,
                             "type": sq.get("type") or kind, "truth_known": sq.get("truth") is not None,
                             "target": bool(tgt), "yes_first": yes_first, "question": sq["question"],
                             "var": sq.get("var")})
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    order = sorted(range(len(exs)), key=lambda i: len(exs[i].text))
    p_yes = [0.0] * len(exs)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            zs = label_logits(model, tok, [exs[i] for i in idx], max_len, cache)
        for i, z in zip(idx, zs):
            pa = torch.softmax(z, -1)[0].item()
            p_yes[i] = pa if meta[i]["yes_first"] else 1 - pa
    if was_training:
        model.train()
    groups: dict[str, list[int]] = {}
    preds = []
    for m, py in zip(meta, p_yes):
        ok = int((py > 0.5) == m["target"])
        preds.append(dict(m, p_yes=py, correct=ok))
        for g in ("all_" + m["kind"], f"{m['kind']}/{m['source']}", f"{m['kind']}/type/{m['type']}"):
            groups.setdefault(g, []).append(ok)
        if m["kind"] == "cot" and m["truth_known"]:
            groups.setdefault("cot/program_truth", []).append(ok)
    metrics = {g: {"n": len(v), "acc": sum(v) / len(v)} for g, v in sorted(groups.items())}
    return metrics, preds