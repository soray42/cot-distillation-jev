"""One-pass decision student: LM-head letter readout, KL to soft targets, sub-question supervision.

Every question (the final decision and each sub-question) is its own sequence that repeats the
problem text; hybrid linear-attention models (Qwen3.5) ignore attention masks, so we do not pack
questions with a block-causal mask. Sequences are right-padded; the readout is the next-token
distribution at the last real token, restricted to the option-letter tokens. The opt-in --tree-pack (Pack) reads
several node questions of one problem in one plain causal pass instead: later questions see the earlier questions,
never an answer.
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
from .treeplan import (CTX_VIEWS, LOCAL_HEADER, MC_SYMBOLS, _descendants, _fact_line, _node_p,  # noqa: F401
                       ancestors, answer_component, depth_split, full_tree_plan, is_local_leaf, is_mc_node,
                       local_context, mc_check, mc_labels, mc_options_block, mix_statements, node_depths,
                       problem_no_options, scaffold_fade, scaffold_rng, statement_block, subq_keep, view_leak)

SUBQ_TEMPLATE = "{problem}\n\nIntermediate question: {question}\nOptions:\nA) {a}\nB) {b}\nAnswer:"
# a multiple-choice tree node (treeplan.is_mc_node): its own options, the layout of SUBQ_TEMPLATE
SUBQ_OPTS_TEMPLATE = "{problem}\n\nIntermediate question: {question}\nOptions:\n{opts}\nAnswer:"
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
    """Token ids of ' A', ' B', ... (the token that follows 'Answer:'; under --chat-template the bare 'A', 'B', ...
    that opens the assistant turn); each must be a single token. A label whose spaced form is not one token but whose
    bare form is (Qwen splits ' 0' into ' ' + '0': the digit labels 0-9 of option lists beyond A-Z, a-z) is read as the
    bare token."""
    ids = []
    sp = "" if _chat(tokenizer) else " "      # --chat-template: the assistant turn opens with the bare letter
    for L in letters:
        t = tokenizer.encode(sp + L, add_special_tokens=False)
        if len(t) != 1 and sp:
            bare = tokenizer.encode(L, add_special_tokens=False)
            t = bare if len(bare) == 1 else t
        if len(t) != 1:
            raise ValueError(f"label '{sp}{L}' is not a single token: {t}")
        ids.append(t[0])
    return ids


# ---- --chat-template (opt-in; nothing below runs on the default path) ---------------------------------------------
CHAT_INSTRUCTION = "\n\nAnswer with the letter of the correct option only."     # as scripts/eval_llm_baseline.py
_NO_THINK_CLOSE = "\n\n</think>\n\n"
_CHAT: dict[int, tuple] = {}


def chat_render(proc, typed: bool = False, bos: str | None = None):
    """The --chat-template renderer of an Example text: the text without its closing "Answer:" cue, plus
    CHAT_INSTRUCTION, as one user turn of the model's chat template with thinking off, ending where the assistant's
    answer starts. For a final question this is exactly scripts/eval_llm_baseline.py's rendering of the prompt. A
    template that ignores enable_thinking but opens a think block gets it closed empty; a leading BOS string is cut
    (special_prefix puts the BOS token back). typed: content as typed parts (multimodal processors)."""
    def render(text: str) -> str:
        body = text
        for cue in ("\n\nAnswer:", "\nAnswer:"):
            if text.endswith(cue):
                body = text[:-len(cue)]
                break
        content = body + CHAT_INSTRUCTION
        out = proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": content}] if typed
                                         else content}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        if out.rstrip().endswith("<think>"):
            out = out.rstrip() + _NO_THINK_CLOSE
        return out[len(bos):] if bos and out.startswith(bos) else out
    return render


def set_chat_render(tokenizer, render) -> None:
    """--chat-template: every Example text this tokenizer encodes (text_ids: collate, micro-batch lengths, FSDP split)
    is rendered by render first, and option letters are read as bare tokens ("A") instead of " A". Keyed by the
    tokenizer object (kept, so its id cannot be reused), like special_prefix."""
    _CHAT[id(tokenizer)] = (tokenizer, render)


def _chat(tokenizer):
    hit = _CHAT.get(id(tokenizer))
    return hit[1] if hit is not None and hit[0] is tokenizer else None


def use_chat_template(tokenizer, model_path: str) -> str:
    """Set the chat renderer for --chat-template, the template loaded as scripts/eval_llm_baseline.py loads it (the
    model's processor, else the tokenizer). Returns a rendered example for the log."""
    import transformers
    try:
        proc = transformers.AutoProcessor.from_pretrained(model_path)
    except Exception:
        proc = tokenizer
    typed = isinstance(proc, transformers.ProcessorMixin)
    set_chat_render(tokenizer, chat_render(proc, typed, tokenizer.bos_token if special_prefix(tokenizer) else None))
    letter_token_ids(tokenizer, ["A", "B", "C", "D", "E"])          # bare letters must be single tokens
    return _chat(tokenizer)(FINAL_TEMPLATE.format(problem="Question: Is it?\nOptions:\nA) Yes\nB) No"))


def text_ids(tokenizer, text: str) -> list[int]:
    """Token ids of an Example text without special tokens (rendered as a chat turn under --chat-template)."""
    r = _chat(tokenizer)
    return tokenizer.encode(r(text) if r else text, add_special_tokens=False)


_PREFIX: dict[int, tuple] = {}


def special_prefix(tokenizer) -> list[int]:
    """Special tokens the tokenizer itself puts before every text: [<bos>] for Gemma, [] for Qwen.

    Our encodes use add_special_tokens=False so that left truncation keeps the question end; callers re-attach this
    prefix after truncation (Gemma degrades badly without its BOS). The tokenizer object is kept in the cache, so
    its id cannot be reused by another tokenizer."""
    hit = _PREFIX.get(id(tokenizer))
    if hit is None or hit[0] is not tokenizer:
        probe = "Answer:"
        with_sp = list(tokenizer.encode(probe, add_special_tokens=True))
        without = list(tokenizer.encode(probe, add_special_tokens=False))
        pre = with_sp[:len(with_sp) - len(without)] if with_sp[len(with_sp) - len(without):] == without else []
        hit = _PREFIX[id(tokenizer)] = (tokenizer, pre)
    return hit[1]


def encode_tail(tokenizer, text: str, max_len: int) -> list[int]:
    """Token ids of text, cut to its last tokens so that the special prefix plus the text fit max_len."""
    pre = special_prefix(tokenizer)
    ids = text_ids(tokenizer, text)
    return pre + ids[-max(1, max_len - len(pre)):]


def logit_softcap(model) -> float | None:
    """final_logit_softcapping of the (text) config: 30.0 for Gemma 4, None for Qwen. The HF forward applies
    s * tanh(z / s) to every logit, so a readout from h @ W.T must do the same to match the model's distribution."""
    cfg = model.config
    cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    return getattr(cfg, "final_logit_softcapping", None) or None


def head_logits(h: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None,
                softcap: float | None = None) -> torch.Tensor:
    """h @ w.T (+ bias) for rows w of the output embedding. Without a soft-cap: exactly the previous readout (in w's
    dtype, so Qwen numbers are unchanged). With one: in float32 outside autocast (bf16 spacing is 0.125-0.25 at
    |z| = 16-64, coarse for calibrated letter probabilities), then s * tanh(z / s)."""
    if not softcap:
        z = h.to(w.dtype) @ w.T
        return z + bias if bias is not None else z
    with torch.autocast(device_type=h.device.type, enabled=False):
        z = h.float() @ w.float().T
        if bias is not None:
            z = z + bias.float()
        return softcap * torch.tanh(z / softcap)


def _soft(p_yes: float, yes_first: bool, eps: float = 1e-4) -> list[float]:
    p = min(max(p_yes, eps), 1 - eps)
    return [p, 1 - p] if yes_first else [1 - p, p]


def mc_node_example(problem: str, sq: dict, weight: float, item_id) -> Example:
    """A multiple-choice tree node (treeplan.is_mc_node) asked like a final question: the context, the node question
    with its own options under the letters A, B, ... (SUBQ_OPTS_TEMPLATE), readout at "Answer:" over those letters, a
    one-hot target on its answer index (label smoothing, if any, is applied by kl_loss as for the final question).
    Draws nothing from any generator (the option order is the data's)."""
    labels = mc_labels(sq)
    return Example(SUBQ_OPTS_TEMPLATE.format(problem=problem, question=sq["question"], opts=mc_options_block(sq)),
                   labels, [float(i == sq["answer"]) for i in range(len(labels))], weight, "subq", item_id)


_OPT_LINE = re.compile(r"^([A-Za-z0-9])\) (.*)$")     # option symbols A-Z, a-z, 0-9 (lists of up to 62 options)


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
                   subq_frac: float = 0.0, subq_filter: str = "all") -> list[Example]:
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
    subq_filter: "all" | "path" | "mirror": ask only the tree nodes subq_keep admits (decision chains; the matched
                 controls of "random" and "mix" are not filtered)
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
    if subq_filter != "all":
        item = dict(item, subqs=[sq for sq in item.get("subqs", []) if subq_keep(sq, subq_filter)])
    if subq == "mix":            # same number of sub-questions as "cot": half CoT nodes, half matched controls
        cot = [dict(sq, _kind="cot") for sq in item.get("subqs", [])]
        ctl = [dict(sq, _kind="random") for sq in item.get("random_subqs", [])]
        n = len(cot)
        k = (n + 1) // 2
        pool = rng.sample(cot, k) + rng.sample(ctl, min(n - k, len(ctl)))
    else:
        pool = {"cot": item.get("subqs", []), "random": item.get("random_subqs", [])}.get(subq, [])
    if any(is_mc_node(sq) for sq in pool):              # opt-in MC nodes: the full-tree arms read them
        raise ValueError(f"{item.get('item_id')}: multiple-choice tree nodes are read by the full-tree arms "
                         "(--tree-full) and evaluate_subq, not by --subq")
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


def build_transition_views(item: dict, *, k: int, rng: random.Random, shuffled: bool = False, keep_prob: float = 1.0,
                           weight: float = 0.5, subq_target: str = "cot", subq_filter: str = "all") -> list[Example]:
    """Tree-transition auxiliary views: k nodes, each asked with the results of its parent nodes stated in the prompt
    (the local step parents -> child of the teacher's dependency tree). Nodes with parents are drawn first.
    shuffled=True is the edge-rewiring control: the same node, label and number of stated results, but the stated
    results come from nodes of the same item that are neither the node's parents nor its descendants. keep_prob drops
    each stated result independently (scaffold withdrawal; 0 gives plain node questions). subq_filter restricts the
    asked nodes only (parents and the rewiring pool come from every node)."""
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
    if subq_filter != "all":
        with_p = [sq for sq in with_p if subq_keep(sq, subq_filter)]
        without = [sq for sq in without if subq_keep(sq, subq_filter)]
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


def _mc_node_view(item: dict, sq: dict, lines: list[str], tgt: str, rng: random.Random, weight: float,
                  m: int = 3, p_pair: float = 0.85) -> Example | None:
    """A tree node asked in multiple-choice form, with the same stated results as its yes/no view: the node, other
    confident nodes of the item and "None of them" as options ("Which of these questions has the answer Yes?"). A Yes
    node goes with m-1 confident No nodes; a No node with one confident Yes node (when the item has one) and m-2 No
    nodes (with probability p_pair; otherwise "None of them" is the answer, about 7% of views). The soft target is the
    build_mc_views one, P(i) proportional to p_i * prod_{j != i} (1 - p_j) and P(none)
    to prod_j (1 - p_j). None when the node has no target or the item has too few confident nodes."""
    p = _node_p(sq, tgt)
    rest = [x for x in item.get("subqs", []) if x is not sq and x.get("question") != sq.get("question")
            and _node_p(x, tgt) is not None]
    no = [x for x in rest if _node_p(x, tgt) <= 0.3]
    yes = [x for x in rest if _node_p(x, tgt) >= 0.7]
    if p is None:
        return None
    # a No node is asked next to a confident Yes node of the item when there is one, so "None of them" is not the
    # answer half of the time (a hedge-option prior); the node itself is still an option to judge
    extra = [rng.choice(yes)] if p < 0.5 and yes and rng.random() < p_pair else []
    if len(no) < m - 1 - len(extra):
        return None
    chosen = [sq] + extra + rng.sample(no, m - 1 - len(extra))
    rng.shuffle(chosen)
    ps = [min(max(_node_p(x, tgt), 1e-4), 1 - 1e-4) for x in chosen]
    none = math.prod(1 - q for q in ps)
    scores = [q * none / (1 - q) for q in ps] + [none]
    z = sum(scores)
    letters = [chr(65 + i) for i in range(m + 1)]
    opts = "\n".join(f"{L}) {t}" for L, t in zip(letters, [x["question"] for x in chosen] + [MC_NONE]))
    problem = item["prompt"] + (f"\n\n{FACTS_HEADER}\n" + "\n".join(lines) if lines else "")
    return Example(SUBQ_MC_TEMPLATE.format(problem=problem, opts=opts), letters, [x / z for x in scores], weight,
                   "subq", item["item_id"])


def build_full_tree(item: dict, *, mode: str, rng: random.Random, cap: int = 10, aux_total: float = 1.0,
                    final: str = "teacher", subq_target: str = "cot", permute_final: float = 0.0,
                    node_format: str = "yesno", eligible_only: bool = False, subq_filter: str = "all",
                    view: str = "problem", sources: set | None = None, node_weight: float = 0.0,
                    stats: dict | None = None, mix_p: float = 0.5, mix_seed: int = 0,
                    mix_epoch: int = 0, scaffold_p: float | None = None, depth_weight: float = 0.0,
                    node_types: set | frozenset | None = None, target_seed: int | None = None,
                    target_kind: str = "shuffled", placebo_bank: PlaceboBank | None = None) -> list[Example]:
    """One problem with ALL its tree nodes (up to `cap`) instead of k sampled ones: [final] + one view per node.
    mode "true": eligible nodes are asked with their parents' results stated; "matched": with the same number of
    depth-matched non-parent results (see full_tree_plan); "plain": every node asked without facts; "placebo": the
    plain views at zero weight; "control": the item's control questions (fresh targets), plainly. The auxiliary
    weight is normalised per problem (aux_total / number of views), so larger trees do not weigh more.
    node_format "mc" asks each node in multiple-choice form (_mc_node_view) where the item allows it.
    eligible_only keeps only the eligible views of the plan (TFE/TFME/GFE: every asked view differs between true and
    matched); the plan, cap included, is drawn first as before, so the asked nodes are the same in every mode. An item
    with no eligible view trains its final question only. With eligible_only, plain views draw from rng as if they
    stated the parents' results, so GFE stays in lockstep with TFE/TFME (same node subsample, same option order).
    subq_filter restricts the asked nodes (full_tree_plan); the control mode is not filtered.
    cap 0 asks every node. sources (a set of item sources, None = all): items of other sources train their final
    question only. node_weight > 0 gives every node view that weight (times the item weight) instead of the
    per-problem split aux_total / views. view "local" / "ancestors" (Tree Forcing): see _local_views; "problem" (the
    default) is the behaviour above. view "statements" (GFP-S, mode "plain" only): the problem view with the problem
    replaced by its statement block (treeplan.statement_block: no final question, no options); rng draws, nodes,
    targets and weights are exactly those of mode "plain" with the problem view (GFP). views "local+ctxparse" (TFLP)
    and "mix" (MIX), mode "true" only: TFL with some nodes asked in the statements view instead (see _local_views;
    mix_p, mix_seed and mix_epoch key MIX's per-node coin).
    GTF (grounded Tree Forcing) = GFP-S plus two options, both off by default:
    - scaffold_p (None = off; view "statements" only): this problem's probability p(t) (treeplan.scaffold_fade) that a
      derived node is asked with its parents' results stated under the statement block (_scaffold_lines); the coin
      and the line order come from the node's own generator (treeplan.scaffold_rng, keyed by mix_seed, mix_epoch,
      item and node), never from rng, so nodes, order, targets (yes/no order included) and weights stay GFP-S's;
    - depth_weight G (0 = off; every view and mode except control): the problem's node-view weight (aux_total, or
      node_weight x views) split in proportion to (1 + node depth)^G instead of evenly (_depths).
    node_types (None = every node; --tree-node-types, e.g. GFPSNC = GFP-S without the K&K "case" nodes; every view and
    mode except control): only the nodes whose "type" is listed are asked (full_tree_plan, before the cap; items of
    other sources are untouched). The weight convention is unchanged, so the kept views share the same per-problem
    total (aux_total / kept views each); depths, parents, matched pools and scaffold results still use every node.
    Not in rng lockstep with the unfiltered arm (fewer views draw fewer yes/no orders).
    Multiple-choice nodes (treeplan.is_mc_node: the node carries "options" and "answer"; opt-in data such as
    scripts/build_fa_sf.py --trees) are asked in the same context as a yes/no node of the view (stated results and
    GTF scaffold included) with their own options (mc_node_example: SUBQ_OPTS_TEMPLATE, labels A, B, ..., one-hot
    target on the answer, the per-problem weight) and draw nothing from rng; node_format "mc" leaves them as they are.
    The local views (_local_views) refuse them.
    target_seed (None = off, the data's targets; --tree-target shuffled = GFPSR / GFPQR, the random-label control):
    mode "plain" with views "problem" / "statements" / "statements+question" only (yes/no node format, no eligible_only,
    no scaffold: no view states or selects by a real result). Every asked node keeps its view, question, option order
    and weight, and the final question is untouched; only a node's target is replaced by its shuffled_targets
    one (the item's P(Yes) values permuted among its nodes, keyed by target_seed and the item; an MC node: a uniformly
    drawn option, one-hot). Nothing is drawn from rng, so the run is in rng lockstep with the real-target arm (same
    nodes, order, option orders and weights). stats counts shuffled_views, shuffled_agree (the random target's answer
    is the real one), shuffled_yes and real_yes (yes/no views with P(Yes) >= .5).
    Why not --aux-kind cot_randlabel (G3R) or a --subq-target value: cot_randlabel is the grouped loop's (build_group),
    draws from the training rng (breaking the lockstep) and redraws a uniform 50/50 label per view and epoch;
    --subq-target picks which real field is the target and also feeds the non-tree loops.
    target_kind "placebo" (with target_seed; --tree-target placebo = GFPSX / GFPQX, the placebo control): the same
    scope, but every asked node keeps its view, option order, weight and answer while its question is replaced by its
    placebo_questions one (a surface question about the problem text with a correct answer, keyed by target_seed and
    the item; placebo_bank supplies the "no" sentences of other problems); the target is the hard answer (with hard node
    targets, every view's target vector is GFPS's). Nothing is drawn from rng (rng lockstep with GFPS / GFPQ). An asked
    multiple-choice node raises ValueError. stats counts placebo_views, placebo_yes, real_yes, placebo_words /
    real_words (whitespace words of the placebo / node questions) and placebo_<template> per template. (Unrelated to
    mode "placebo", the zero-weight plain views.)"""
    fin = [e for e in build_examples(item, final=final, subq="none", subq_target=subq_target, lambda_sub=0.0, rng=rng,
                                     permute_final=permute_final) if e.kind == "final"]
    if not fin:
        return []
    if sources is not None and item.get("source") not in sources:
        return fin
    if scaffold_p is not None and (view != "statements" or mode != "plain" or not 0.0 <= scaffold_p <= 1.0):
        raise ValueError(f"scaffold_p {scaffold_p}: a probability, for --tree-full plain --tree-view statements (GTF)")
    if depth_weight and mode == "control":
        raise ValueError("--tree-depth-weight needs tree nodes: control questions have no depth")
    if node_types is not None and mode == "control":
        raise ValueError("--tree-node-types needs tree nodes: control questions have no type")
    if target_seed is not None and target_kind not in ("shuffled", "placebo"):
        raise ValueError(f"target_kind {target_kind}: shuffled or placebo")
    if target_seed is not None and (mode != "plain" or view not in ("problem", "statements", "statements+question")
                                    or node_format != "yesno" or eligible_only or scaffold_p is not None):
        raise ValueError(f"--tree-target {target_kind} is implemented for --tree-full plain, views problem / statements "
                         "/ statements+question, yes/no node format, without --tree-eligible-only or --tree-scaffold")
    if view not in ("problem", "statements", "statements+question"):
        return fin + _local_views(item, mode=mode, view=view, rng=rng, cap=cap, aux_total=aux_total,
                                  subq_target=subq_target, node_format=node_format, eligible_only=eligible_only,
                                  subq_filter=subq_filter, node_weight=node_weight, stats=stats, mix_p=mix_p,
                                  mix_seed=mix_seed, mix_epoch=mix_epoch, depth_weight=depth_weight,
                                  node_types=node_types)
    if view in ("statements", "statements+question") and (mode != "plain" or eligible_only or node_format != "yesno"):
        raise ValueError("--tree-view statements is implemented for --tree-full plain (yes/no nodes, all nodes)")
    context = (statement_block(item) if view == "statements" else problem_no_options(item)
               if view == "statements+question" else item["prompt"])
    wi = item.get("weight", 1.0)
    views = []
    if eligible_only and mode == "control":
        raise ValueError("eligible_only needs tree nodes: control questions have no parents")
    by = {sq["id"]: sq for sq in item.get("subqs", []) if "id" in sq}
    if mode == "control":
        pool = [sq for sq in item.get("random_subqs", []) if sq.get("p_fresh") is not None]
        if len(pool) > cap:
            pool = rng.sample(pool, cap)
        views = [(sq, [], "fresh", 0) for sq in pool]
    else:
        plan = full_tree_plan(item, cap=cap, rng=rng, subq_target=subq_target, subq_filter=subq_filter,
                              node_types=node_types, stats=stats)
        if eligible_only:
            plan = [e for e in plan if e["parents"]]
        for e in plan:
            facts = {"true": e["parents"], "matched": e["matched"]}.get(mode, [])
            views.append((e["node"], [by[j] for j in facts], subq_target, len(e["parents"])))
    out = list(fin)
    w = 0.0 if mode == "placebo" else (node_weight or aux_total / max(1, len(views)))
    ws = [w] * len(views)
    if depth_weight and views:                              # GTF: same total per problem, split by depth
        total = 0.0 if mode == "placebo" else (node_weight * len(views) if node_weight else aux_total)
        depth = _depths(item)
        ws = depth_split(total, [depth[sq["id"]] for sq, *_ in views], depth_weight)
    if stats is not None:
        stats["items_with_views"] = stats.get("items_with_views", 0) + bool(views)
        stats["views"] = stats.get("views", 0) + len(views)
        if scaffold_p is not None:
            for key in ("scaffold_views", "scaffold_lines", "scaffold_eligible"):
                stats[key] = stats.get(key, 0)
    shuf = plc = None
    if target_seed is not None and target_kind == "placebo":    # GFPSX / GFPQX: the placebo questions, drawn off rng
        plc = placebo_questions(item, target_seed, placebo_bank, subq_target) if views else {}
        st = stats if stats is not None else {}
        for key in ("placebo_views", "placebo_yes", "real_yes", "placebo_words", "real_words",
                    *(f"placebo_{t}" for t in PLACEBO_TEMPLATES)):
            st[key] = st.get(key, 0)
    elif target_seed is not None:                           # GFPSR / GFPQR: the random targets, drawn off rng
        shuf = shuffled_targets(item, target_seed, subq_target)
        st = stats if stats is not None else {}
        for key in ("shuffled_views", "shuffled_agree", "shuffled_yes", "real_yes"):
            st[key] = st.get(key, 0)
    for (sq, facts, tgt, k), wk in zip(views, ws):
        lines = [_fact_line(x) for x in facts]
        rng.shuffle(lines)
        if eligible_only and not lines:                     # GFE: the draws of the k stated lines of TFE/TFME
            rng.shuffle([None] * k)
        mc = mode != "control" and is_mc_node(sq)         # a node with its own options (opt-in data)
        if node_format == "mc" and mode != "control" and not mc:   # multiple-choice node view, same stated results
            ex = _mc_node_view(item, sq, lines, tgt, rng, wk * wi)
            if ex is not None:
                out.append(ex)
                continue
        problem = context + (f"\n\n{FACTS_HEADER}\n" + "\n".join(lines) if lines else "")
        if scaffold_p is not None:                          # GTF (lines is empty in mode plain)
            sc = _scaffold_lines(item, sq, by, scaffold_p, mix_seed, mix_epoch, stats)
            if sc:
                problem = context + f"\n\n{LOCAL_HEADER}\n" + "\n".join(sc)
        if mc:                                              # its options and answer, as a final question; no draw
            if plc is not None:
                raise ValueError(f"--tree-target placebo: {item['item_id']} node {sq['id']} is multiple-choice")
            if shuf is not None:                            # GFPSR / GFPQR: a random option
                st["shuffled_views"] += 1
                st["shuffled_agree"] += shuf[sq["id"]] == sq["answer"]
                sq = dict(sq, answer=shuf[sq["id"]])
            out.append(mc_node_example(problem, sq, wk * wi, item["item_id"]))
            continue
        yes_first = rng.random() < 0.5
        a, b = ("Yes", "No") if yes_first else ("No", "Yes")
        p = _node_p(sq, tgt)
        question = sq["question"]
        if shuf is not None:                                # GFPSR / GFPQR: the permuted target, same view and draws
            st["shuffled_views"] += 1
            st["shuffled_agree"] += (shuf[sq["id"]] >= 0.5) == (p >= 0.5)
            st["shuffled_yes"] += shuf[sq["id"]] >= 0.5
            st["real_yes"] += p >= 0.5
            p = shuf[sq["id"]]
        elif plc is not None:                               # GFPSX / GFPQX: the placebo question, its correct answer
            q, yes, kind = plc[sq["id"]]
            st["placebo_views"] += 1
            st["placebo_yes"] += yes
            st["real_yes"] += p >= 0.5
            st["placebo_words"] += len(q.split())
            st["real_words"] += len(question.split())
            st[f"placebo_{kind}"] += 1
            question, p = q, float(yes)
        out.append(Example(SUBQ_TEMPLATE.format(problem=problem, question=question, a=a, b=b), ["A", "B"],
                           _soft(p, yes_first), wk * wi, "subq", item["item_id"]))
    return out


def shuffled_targets(item: dict, seed: int, subq_target: str = "cot") -> dict:
    """--tree-target shuffled (GFPSR / GFPQR: the random-label control of GFPS / GFPQ): a random target for every tree
    node of the item that has one, by node id. Yes/no nodes (a target under subq_target, _node_p): the item's P(Yes)
    values permuted among them, so the item keeps exactly its multiset of targets (its yes rate, and the soft values if
    any) while a node's target no longer depends on its question (a node keeps its own value by chance, about half the
    time on balanced trees). Multiple-choice nodes (is_mc_node): an option index drawn uniformly. Drawn over every
    node with a target (before the cap, --tree-node-types or --subq-filter select the asked ones), from generators of
    their own seeded with str keys (sha512, as treeplan.scaffold_rng): (seed, item) for the permutation, (seed, item,
    node) for an MC draw. The epoch is not in the key: every node keeps one random target over the whole run (a fixed
    random labelling, as consistent across epochs as the real targets, not noise redrawn each epoch), and the draw
    depends on nothing else (not the training rng, PYTHONHASHSEED, the cap or the order of the views)."""
    nodes = [sq for sq in item.get("subqs", []) if "id" in sq]
    yn = [sq for sq in nodes if not is_mc_node(sq) and _node_p(sq, subq_target) is not None]
    ps = [_node_p(sq, subq_target) for sq in yn]
    random.Random(f"tree-target|{seed}|{item['item_id']}").shuffle(ps)
    out = {sq["id"]: p for sq, p in zip(yn, ps)}
    for sq in nodes:
        if is_mc_node(sq):
            out[sq["id"]] = random.Random(f"tree-target|{seed}|{item['item_id']}|{sq['id']}").randrange(
                len(sq["options"]))
    return out


# ---- --tree-target placebo (GFPSX / GFPQX): surface questions about the problem text instead of the node questions ----
PLACEBO_TEMPLATES = ("name", "number", "word", "sentence")    # this order breaks ties and orders the fallbacks
PLACEBO_MIN_K = 3              # a sentence question quotes at least this many words (or the whole, shorter sentence)
PLACEBO_OVERLAP = 0.5          # a foreign sentence may share at most this share of its content tokens with the prompt
PLACEBO_FOREIGN = 24           # other problems drawn per problem for its "no" sentences and words
_PL_FIXED = {"name": 6, "number": 7, "word": 8}      # words of the fixed templates (a sentence: 7 + k)
_PL_STOP = frozenset("""a about above after again against all also am an and any are as at be because been before
being below between both but by can could did do does doing down during each either else every few for from further
had has have having he her here hers herself him himself his how i if in into is it its itself just let me more most
much must my myself neither no nor not now of off on only or other our ours ourselves out over own per same she should
so some such than that the their theirs them themselves then there these they this those through to too under until
up upon us very was we were what when where whether which while who whom whose why will with within without would yet
you your yours yourself""".split())
_PL_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
            "seventeen eighteen nineteen").split()
_PL_TENS = dict(zip(range(20, 100, 10), "twenty thirty forty fifty sixty seventy eighty ninety".split()))
_PL_ORD = ("zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth "
           "fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth").split()
_PL_NUMX = {1: "once single", 2: "twice double doubled half halves pair pairs", 3: "thrice triple tripled thirds",
            4: "quarter quarters", 12: "dozen dozens", 100: "hundred", 1000: "thousand", 10 ** 6: "million"}
_PL_NUMWORDS = frozenset(_PL_ONES + _PL_ORD + list(_PL_TENS.values()) + [w[:-1] + "ieth" for w in _PL_TENS.values()]
                         + " ".join(_PL_NUMX.values()).split() + ["billion", "percent", "hundredth", "thousandth"])
# Names with no other English reading (Will, Max, Grace, Rose, Mark, Bill, ... left out): numgen, K&K (sources.NAMES and
# scripts/build_kk_deep.py's EXTRA_NAMES), PrOntoQA's entities and frequent GSM8K names.
_PL_NAME_X = ("Quinn Rosa Sam Tina Umar Vera Yara John Mary Mike Peter Sarah Jerry Martha Harry Lisa Steve Janet Joe "
              "Kevin Carl Jill Ryan Michael Jim Jason Carla Jenny Ben Andrew Anna Jane Jimmy Billy Tony Julia Adam Emily "
              "Jake Susan Lucy Jacob Jeff Dave Betty Monica Kim Gerald Amanda Kyle Rick Rachel Nancy Andy Jose").split()
_PL_NOT_NAMES = frozenset("Will Max Kit Ivy Opal Hope Ray Pip Wren Sally Grace Rose Mark Bill May June Bo Mo Cy Ty "
                          "Val".split())
# "No" words when no other problem of the source has one (e.g. the K&K and Lean vocabularies are shared by all items).
_PL_WORD_POOL = ("anchor basket blanket bridge candle carpet castle cherry compass copper cotton desert engine feather "
                 "forest garden glacier hammer harbor helmet island jacket kettle ladder lantern lemon market meadow "
                 "mirror needle orchard paddle parade pencil pepper pillow planet pocket puzzle rabbit ribbon river "
                 "saddle saucer shovel silver spider stable tablet ticket timber tunnel valley violin wagon walnut "
                 "window winter").split()
_PL_NUM = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_PL_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_PL_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_PL_NAMES: list = []


def _pl_names() -> list[str]:
    """The name pool of the name questions, sorted (built once)."""
    if not _PL_NAMES:
        from . import numgen, pronto, sources
        pool = set(sources.NAMES) | {n for n, _ in numgen.NAMES} | set(pronto.ENTITIES) | set(_PL_NAME_X)
        _PL_NAMES.extend(sorted(pool - _PL_NOT_NAMES))
    return _PL_NAMES


def _pl_val(tok: str):
    """A number token's value ("1,200" -> 1200, "2.50" -> 2.5, 24.0 -> 24): what number questions compare."""
    v = float(str(tok).replace(",", ""))
    return int(v) if v == int(v) else v


def _pl_nums(text: str) -> set:
    return {_pl_val(x) for x in _PL_NUM.findall(text)}


def _pl_words(text: str) -> list[str]:
    """The word tokens of a text: letters only, a possessive 's dropped, other contractions skipped."""
    out = []
    for m in _PL_WORD.finditer(text.replace("’", "'")):
        w = m.group()
        if "'" in w:
            if not w.lower().endswith("'s"):
                continue
            w = w[:-2]
        out.append(w)
    return out


def _pl_variants(w: str) -> set:
    """w lower-cased with its plural / singular look-alikes (candle ~ candles, snarpus ~ snarpuses, city ~ cities)."""
    w = w.lower()
    v = {w, w + "s", w + "es"}
    if w.endswith("s"):
        v.add(w[:-1])
    if w.endswith("es"):
        v.add(w[:-2])
    if w.endswith("ies"):
        v.add(w[:-3] + "y")
    if w.endswith("y"):
        v.add(w[:-1] + "ies")
    return v


def _pl_spelled(n: int) -> list[str]:
    """Word forms of a whole number that a text may use instead of digits (four, fourth, quarter, sixty-one, ...)."""
    out = list(_PL_NUMX.get(n, "").split())
    if 0 <= n < 20:
        out += [_PL_ONES[n], _PL_ORD[n]]
    elif 20 <= n < 100:
        t = _PL_TENS[n - n % 10]
        out += [t, t[:-1] + "ieth"] if n % 10 == 0 else [f"{t}-{_PL_ONES[n % 10]}", f"{t} {_PL_ONES[n % 10]}",
                                                         f"{t}-{_PL_ORD[n % 10]}", f"{t} {_PL_ORD[n % 10]}"]
    return out


def _pl_sentences(text: str) -> list[str]:
    """The sentences of a text, each an exact substring of it: per line, a K&K "- " bullet dropped, split after . ! ?"""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("- "):
            line = line[2:]
        out += [s for s in _PL_SENT.split(line) if s.strip()]
    return out


def _pl_prefix(s: str, k: int) -> tuple[str, int]:
    """The first k words of sentence s, cut at the k-th word's last character (an exact substring of s), with "..."
    when words are left out; and the number of words quoted."""
    ws = list(re.finditer(r"\S+", s))
    k = min(k, len(ws))
    return s[:ws[k - 1].end()] + ("..." if k < len(ws) else ""), k


def placebo_text(item: dict) -> str:
    """The text a placebo question is about: the statement block (treeplan.statement_block, GFP-S's context), or for a
    gsm8k_tf item without a clean statement / question cut (about 60% of GSM8K) the whole problem text (GFPQ's context
    minus the "Question: " label). Inside every view's context (problem / statements / statements+question) wherever
    that view exists. Other sources without a statement block raise ValueError."""
    from .treeplan import GSM8K_TF, _gsm8k_tf_problem
    try:
        return statement_block(item)
    except ValueError:
        if item.get("source") != GSM8K_TF:
            raise
        return _gsm8k_tf_problem(item)


def placebo_tree_numbers(item: dict) -> set:
    """Every number of the item's reasoning tree and answer: in the node questions, the nodes' value / probe fields, the
    numgen program's quantities and the answer options. A number question never asks one of them; a tree number that
    the statement block does not state (a derived value, a wrong probe, an option) occurs in no placebo question."""
    out = set()
    for sq in item.get("subqs", []):
        if "id" not in sq:
            continue
        out |= _pl_nums(sq.get("question", ""))
        for key in ("value", "probe"):
            if isinstance(sq.get(key), (int, float)) and not isinstance(sq.get(key), bool):
                out.add(_pl_val(sq[key]))
    prog = item.get("program")
    for q in (prog.get("quantities") or []) if isinstance(prog, dict) else []:
        if isinstance(q, dict) and isinstance(q.get("value"), (int, float)) and not isinstance(q.get("value"), bool):
            out.add(_pl_val(q["value"]))
    out |= _pl_nums(item.get("prompt", "").partition("\nOptions:\n")[2])
    return out


class PlaceboBank:
    """--tree-target placebo: the placebo texts (placebo_text) of a set of problems by source, sorted by item id (not by
    file order): another problem of the same source is where a placebo "no" sentence (and a "no" word) comes from.
    train_student builds it from the training items of --tree-sources; items without a placebo text are left out."""

    def __init__(self, items):
        by: dict = {}
        for it in items:
            try:
                t = placebo_text(it)
            except ValueError:
                continue
            by.setdefault(it.get("source"), []).append((str(it["item_id"]), t))
        self.texts = {s: sorted(v) for s, v in by.items()}

    def draw(self, item: dict, text: str, rng: random.Random, n: int) -> list[str]:
        """Up to n texts of other problems of item's source (not the item, not the same text), drawn by rng."""
        pool = self.texts.get(item.get("source"), [])
        out = []
        for _ in range(n if len(pool) > 1 else 0):
            iid, t = pool[rng.randrange(len(pool))]
            if iid != str(item["item_id"]) and t != text:
                out.append(t)
        return out


def placebo_questions(item: dict, seed: int, bank: PlaceboBank | None = None, subq_target: str = "cot") -> dict:
    """--tree-target placebo (GFPSX / GFPQX: the placebo control of GFPS / GFPQ): for every yes/no tree node of the item
    with a target, by node id, (question, answer, template): a surface question about the problem's text (placebo_text)
    with a correct answer, in place of the node question. The answer is the node's own (P(Yes) >= .5), so a problem's
    yes rate, and with hard targets every view's target, is GFPS's.
    Templates:
    - name: "Is <Name> mentioned in the problem?": a name of the pool that the text has (case-sensitive whole word), or
      one the whole prompt does not have (any case, plural look-alikes too);
    - number: "Does the problem mention the number N?": a number token of the text, or a nearby whole number the whole
      prompt has neither as a number nor in words (four / fourth / quarter / sixty-one ...);
    - word: "Does the word '<w>' appear in the problem?": a lower-case content word of the text (not a stop word, number
      word or name; at least 3 letters), or one of another problem of the source (else of a fixed pool) that the whole
      prompt does not have in any case or plural form;
    - sentence: "Is the sentence '<first k words...>' part of the problem?": the first k words of a sentence of the text
      (an exact substring), or of a sentence of another problem of the source (bank) that is not a substring of the
      prompt, shares no capitalised word (names) and no single letter (variables) with the prompt, at most
      PLACEBO_OVERLAP of its content tokens with the prompt (no paraphrase of a premise: "Every X is Y" vs "Each X is
      Y"), and contains no tree number.
    Never about the tree: a name, number or word question never uses a token of any node question (any node, asked or
    not; words by plural look-alikes) nor a tree number (placebo_tree_numbers); a sentence question quotes this
    problem's own text or holds no tree number. Absent tokens are checked against the whole prompt (final question and
    options included), so the answer is right in every view (GFPQ shows the final question).
    Length: a node of L words gets a template whose length (name 6, number 7, word 8, sentence 7 + k words, k >= 3) is
    within max(2, L / 4) of L, at random among those that fit (else the nearest); a sentence question quotes
    k = L - 7 words where its sentence allows. Templates are picked among those the problem can ask with both answers,
    so the template does not tell the answer; a problem's questions are distinct where its supply allows.
    Drawn over every yes/no node with a target (before the cap or filters select the asked ones), in node order, from a
    generator of its own keyed by (seed, item) (sha512 of a str key, as shuffled_targets): nothing from the training
    rng, no epoch (each node keeps its placebo question over the run). With the same bank the questions depend on
    nothing else; without a bank (None) no "no" sentence exists and "no" words come from the fixed pool.
    Multiple-choice nodes get none (build_full_tree refuses them under placebo). ValueError if the item has no
    placebo_text."""
    text = placebo_text(item)
    prompt = item.get("prompt", "")
    nodes = [sq for sq in item.get("subqs", []) if "id" in sq]
    yn = [sq for sq in nodes if not is_mc_node(sq) and _node_p(sq, subq_target) is not None]
    if not yn:
        return {}
    rng = random.Random(f"tree-placebo|{seed}|{item['item_id']}")
    tree_nums = placebo_tree_numbers(item)
    node_words = {w.lower() for sq in nodes for w in _pl_words(sq.get("question", ""))}
    p_words = {w.lower() for w in _pl_words(prompt)}
    p_nums = _pl_nums(prompt)
    names = _pl_names()
    name_set = set(names)
    lower_names = {n.lower() for n in names}

    def in_tree(w):
        return bool(_pl_variants(w) & node_words)

    def in_prompt(w):
        return bool(_pl_variants(w) & p_words)

    def content(w):
        lw = w.lower()
        return len(w) >= 3 and w.isalpha() and lw not in _PL_STOP and lw not in _PL_NUMWORDS and lw not in lower_names

    def uniq(xs):
        return list(dict.fromkeys(xs))

    tw = _pl_words(text)
    yes_num, seen = [], set()
    for x in _PL_NUM.findall(text):                     # surface form as in the text, one per value, text order
        if _pl_val(x) not in tree_nums and _pl_val(x) not in seen:
            seen.add(_pl_val(x))
            yes_num.append(x)
    supply = {True: {"name": uniq(w for w in tw if w in name_set and not in_tree(w)),
                     "word": uniq(w for w in tw if w == w.lower() and content(w) and not in_tree(w))},
              False: {"name": [n for n in names if not in_prompt(n) and not in_tree(n)]}}
    supply[True]["number"] = yes_num
    own = _pl_sentences(text)

    def foreign_ok(q: str) -> bool:
        """A "no" sentence quote: absent, about other entities, not a paraphrase, no tree number."""
        if q in prompt or q in text:
            return False
        nums = _pl_nums(q)
        if nums & tree_nums:
            return False
        ws = _pl_words(q)
        for w in ws:
            if len(w) == 1 and w not in ("a", "A", "I") and w.lower() in p_words:
                return False                             # a variable of this problem (Lean)
            if w[0].isupper() and w.lower() not in _PL_STOP and in_prompt(w):
                return False                             # a name / proper noun of this problem
        toks = [w for w in ws if w.lower() not in _PL_STOP]
        if not toks and not nums:
            return False
        shared = sum(in_prompt(w) for w in toks) + len(nums & p_nums)
        return shared <= PLACEBO_OVERLAP * (len(toks) + len(nums))

    # Other problems of the source: PLACEBO_FOREIGN of them, more (up to 4x) until there is a "no" sentence per "no"
    # node (sources with few passing sentences, e.g. Lean, stop at the cap).
    need = sum(_node_p(sq, subq_target) < 0.5 for sq in yn)
    foreign_texts, foreign = [], []
    for j in range(4 * PLACEBO_FOREIGN if bank is not None else 0):
        if j >= PLACEBO_FOREIGN and len(foreign) >= need:
            break
        for t in bank.draw(item, text, rng, 1):
            foreign_texts.append(t)
            ss = _pl_sentences(t)
            for x in rng.sample(ss, min(4, len(ss))):
                if foreign_ok(x) and x not in foreign:
                    foreign.append(x)
    no_words = uniq(w for t in foreign_texts for w in _pl_words(t)
                    if w == w.lower() and content(w) and not in_tree(w) and not in_prompt(w))
    supply[False]["word"] = no_words + [w for w in _PL_WORD_POOL if w not in no_words and not in_tree(w)
                                        and not in_prompt(w)]
    supply[True]["sentence"], supply[False]["sentence"] = own, foreign
    ok = {y: {t: bool(supply[y].get(t)) or (t == "number" and not y) for t in PLACEBO_TEMPLATES} for y in (True, False)}
    both = [t for t in PLACEBO_TEMPLATES if ok[True][t] and ok[False][t]]

    def nwords(x):
        return len(x.split())

    smax = {y: max((nwords(x) for x in supply[y]["sentence"]), default=0) for y in (True, False)}
    used: set = set()                                   # questions asked so far
    taken: dict = {}                                    # (template, answer) -> the supply items used so far

    uses: dict = {}                                     # template -> nodes given it so far

    def room(t, y_ok):
        """Whether the template has room for one more node without looking at its answer: its uses stay under
        len(y_ok) x the smaller supply of the answers in y_ok (an absent number is always new)."""
        cap = min(float("inf") if (t == "number" and not y) else len(supply[y].get(t, ())) for y in y_ok)
        return uses.get(t, 0) < len(y_ok) * cap

    def pick(t, y, render):
        xs = supply[y][t]
        new = [x for x in xs if x not in taken.get((t, y), ())]
        x = (new or xs)[rng.randrange(len(new or xs))]
        taken.setdefault((t, y), set()).add(x)
        return render(x)

    def absent_number() -> str:
        low = prompt.lower()
        bases = sorted({int(abs(v)) for v in _pl_nums(text)} - {0}) or list(range(2, 13))

        def fits(c):
            return (c >= 1 and c not in p_nums and c not in tree_nums and f"Does the problem mention the number {c}?"
                    not in used and not any(re.search(rf"\b{re.escape(w)}\b", low) for w in _pl_spelled(c)))
        for _ in range(60):
            b = bases[rng.randrange(len(bases))]
            c = b + rng.choice((-1, 1)) * rng.randint(1, max(3, b // 10))
            if fits(c):
                return str(c)
        c = max(bases) + 1
        while not fits(c):
            c += 1
        return str(c)

    def make(t, yes, L):
        if t == "name":
            return pick(t, yes, lambda x: f"Is {x} mentioned in the problem?")
        if t == "number":
            if yes:
                return pick(t, yes, lambda x: f"Does the problem mention the number {x}?")
            return f"Does the problem mention the number {absent_number()}?"
        if t == "word":
            return pick(t, yes, lambda x: f"Does the word '{x}' appear in the problem?")
        k = max(PLACEBO_MIN_K, L - 7)                   # sentence: 7 + k words
        ss = supply[yes]["sentence"]
        tk = taken.setdefault((t, yes), set())
        fallback = None                                 # random order: unused sentences of k words or more first
        for x in sorted(rng.sample(ss, len(ss)), key=lambda x: (x in tk, nwords(x) < k)):
            q, _ = _pl_prefix(x, k)
            if not yes and not foreign_ok(q.removesuffix("...")):
                continue
            qq = f"Is the sentence '{q}' part of the problem?"
            if qq not in used:
                tk.add(x)
                return qq
            fallback = fallback or qq
        return fallback

    def span(t, y_ok):
        if t != "sentence":
            return _PL_FIXED[t], _PL_FIXED[t]
        hi = min(smax[y] for y in y_ok)
        return 7 + min(PLACEBO_MIN_K, hi), 7 + hi

    out = {}
    for sq in rng.sample(yn, len(yn)):                  # random order: which nodes meet a used-up supply is by chance,
        yes = _node_p(sq, subq_target) >= 0.5           # not by the node order (e.g. pronto's first node is always Yes)
        L = nwords(sq["question"])
        avail, y_ok = (both, (True, False)) if both else ([t for t in PLACEBO_TEMPLATES if ok[yes][t]], (yes,))

        def dist(t):
            lo, hi = span(t, y_ok)
            return max(lo - L, 0, L - hi)
        pool = ([t for t in avail if room(t, y_ok)]                      # templates with unused items first (by
                or [t for t in avail if t == "sentence"] or avail)       # counts, not this node's answer); then a
        tol = max(2.0, L / 4)                                            # sentence quoted at another length
        cands = [t for t in pool if dist(t) <= tol] or [t for t in pool if dist(t) == min(map(dist, pool))]
        first = cands[rng.randrange(len(cands))]
        q = None
        for t in [first] + sorted((t for t in PLACEBO_TEMPLATES if ok[yes][t] and t != first), key=dist):
            q = make(t, yes, L)
            if q is not None:
                break
        if q is None:
            raise ValueError(f"{item['item_id']}: no placebo question for node {sq['id']}")
        uses[t] = uses.get(t, 0) + 1
        used.add(q)
        out[sq["id"]] = (q, yes, t)
    return out


def _depths(item: dict) -> dict:
    """Node depth by id: the node_depth field when every node has one (v3s/v3sl/numgen), else node_depths()."""
    nodes = [sq for sq in item.get("subqs", []) if "id" in sq]
    return ({sq["id"]: sq["node_depth"] for sq in nodes} if all("node_depth" in sq for sq in nodes)
            else node_depths(nodes))


def _scaffold_lines(item: dict, sq: dict, by: dict, p: float, seed: int, epoch: int, stats: dict | None) -> list[str]:
    """GTF's stated results for one node view (empty: the plain GFP-S view): the "Q: ... A: Yes/No" lines of the
    node's parents (depends_on within the item; _fact_line: program truth, else a confident CoT answer), in an order
    drawn from the node's own generator, with probability p (its first draw). Always plain: leaves (parse / read
    nodes with local_lines, or no parent), nodes with an unconfident parent, and nodes whose stated results would give
    the final answer away (view_leak, the TF views' filter). Every case is counted in stats."""
    st = stats if stats is not None else {}
    bump = lambda key, n=1: st.__setitem__(key, st.get(key, 0) + n)
    par = [x for x in (sq.get("depends_on") or []) if x in by and x != sq["id"]]
    if is_local_leaf(sq) or not par:
        bump("scaffold_leaf")
        return []
    stated = [by[j] for j in par]
    if not all(_fact_line(x) for x in stated):
        bump("scaffold_unconfident")
        return []
    leak = view_leak(item, sq, stated)
    if leak:
        bump(f"scaffold_leak_{leak}")
        return []
    bump("scaffold_eligible")
    r = scaffold_rng(seed, epoch, item["item_id"], sq["id"])
    if not r.random() < p:
        return []
    lines = [_fact_line(x) for x in stated]
    r.shuffle(lines)
    bump("scaffold_views")
    bump("scaffold_lines", len(lines))
    return lines


def _local_views(item: dict, *, mode: str, view: str, rng: random.Random, cap: int, aux_total: float,
                 subq_target: str, node_format: str, eligible_only: bool, subq_filter: str, node_weight: float,
                 stats: dict | None, mix_p: float = 0.5, mix_seed: int = 0, mix_epoch: int = 0,
                 depth_weight: float = 0.0, node_types: set | frozenset | None = None) -> list[Example]:
    """Tree Forcing node views WITHOUT the original problem (local_context): the item's preamble, then
    - parse nodes (local_lines): the statement line(s) they read, in every mode;
    - other nodes, view "local": mode "true" = the results of the node's parents (its depends_on; every parent of a
      solver-tree node is sufficient, scripts/build_local_kk.py), mode "matched" = as many matched non-parent results
      (full_tree_plan's selection, post-audit rules) where the plan finds them, else the true parents (the view is
      then the same in both modes; counted as matched_frozen), so the two modes state the same number of results in
      every view and draw rng in lockstep;
    - other nodes, view "ancestors" (mode "true" only): the results of every transitive ancestor, shallow to deep;
    - other nodes, view "local" with mode "plain" (TFLN): nothing stated, the preamble only. The node set, the drops
      and the rng draws are those of mode "true" (the parents' results are selected and checked, and their line order
      drawn, but not shown), so TFLN is in lockstep with TFL: same nodes, order, weights and targets.
    - views "local+ctxparse" (TFLP) and "mix" (MIX), mode "true" only: TFL, except that some nodes are asked in the
      GFP-S statements view (treeplan.statement_block: the problem without its final question and options, nothing
      stated) instead of their local view. TFLP: the parse nodes. MIX: each node with probability mix_p, by its own
      coin (treeplan.mix_statements, keyed by mix_seed, mix_epoch, item and node), so every node has one view per
      epoch. Both keep the non-leaf nodes without parents that TFL drops, always in the statements view (counted as
      statements_no_parents; none in v3sl or numgen, where such a node would break the lockstep with TFL).
      A statements view still draws TFL's line-order shuffle (the TFLN trick), so both arms are in rng lockstep with
      TFL (their local views are TFL's, byte for byte) and have its nodes, order, weights and soft targets; nodes,
      order, weights and P(Yes) are GFP-S's too as long as TFL drops nothing (0 drops on v3sl and numgen).
      stated_lines counts the results shown, withheld_lines those of TFL's view not shown (MIX's derived statements
      views).
    Views that would state the final answer (view_leak) are dropped and counted; so are non-leaf nodes without
    parents or with a parent whose result is not confident (nothing sufficient to state). The node is then asked as
    in the problem-view arms (yes/no order drawn from rng, the same soft target). Weight: node_weight, else
    aux_total / views (the full-tree convention); depth_weight G splits the same total in proportion to
    (1 + node depth)^G. node_types restricts the asked nodes (full_tree_plan; see build_full_tree)."""
    if (mode not in ("true", "matched", "plain") or (view != "local" and mode != "true")
            or view not in ("local", "ancestors") + CTX_VIEWS):
        raise ValueError(f"--tree-view {view} is implemented for --tree-full true (and matched / plain with the local "
                         "view)")
    if eligible_only or node_format != "yesno":
        raise ValueError(f"--tree-view {view} does not combine with --tree-eligible-only / --node-format mc")
    if any(is_mc_node(sq) for sq in item.get("subqs", [])):
        raise ValueError(f"{item.get('item_id')}: multiple-choice tree nodes are read in the problem / statements "
                         f"views only, not in --tree-view {view}")
    ctx_view = view in CTX_VIEWS
    if ctx_view and not 0.0 <= mix_p <= 1.0:
        raise ValueError(f"mix_p {mix_p} is not a probability")
    block = statement_block(item) if ctx_view else None      # TFLP/MIX: raises for sources without a statement block
    if ctx_view and item.get("local_preamble") is None:
        raise ValueError(f"{item.get('item_id')}: no local_preamble (local views need e.g. data/student_v3sl)")
    st = stats if stats is not None else {}
    bump = lambda k, n=1: st.__setitem__(k, st.get(k, 0) + n)
    nodes = [sq for sq in item.get("subqs", []) if "id" in sq]
    by = {sq["id"]: sq for sq in nodes}
    depth = ({i: by[i]["node_depth"] for i in by} if all("node_depth" in sq for sq in nodes) else node_depths(nodes))
    anc = ancestors(nodes) if view == "ancestors" else None
    views, no_par = [], set()
    for e in full_tree_plan(item, cap=cap, rng=rng, subq_target=subq_target, subq_filter=subq_filter,
                            node_types=node_types, stats=st):
        sq = e["node"]
        if is_local_leaf(sq):
            stated = []
        else:
            par = [x for x in (sq.get("depends_on") or []) if x in by and x != sq["id"]]
            if not par and not ctx_view:
                bump("dropped_no_parents")
                continue
            if not par:                             # TFLP / MIX: no local view, asked in the statements view
                ids = []
                no_par.add(sq["id"])
            elif view == "ancestors":
                ids = sorted(anc[sq["id"]] - {sq["id"]})
            elif mode == "matched" and e["parents"]:
                ids = e["matched"]
            else:
                ids = par
                if mode == "matched":
                    bump("matched_frozen")
            stated = [by[j] for j in ids]
            if not all(_fact_line(x) for x in stated):
                bump("dropped_unconfident_parent")
                continue
        leak = view_leak(item, sq, stated)
        if leak:
            bump(f"dropped_leak_{leak}")
            continue
        views.append((sq, stated))
    bump("items_with_views", bool(views))
    bump("views", len(views))
    bump("leaf_views", sum(is_local_leaf(sq) for sq, _ in views))
    if mode == "plain":
        bump("withheld_lines", sum(len(x) for _, x in views))
    elif not ctx_view:
        bump("stated_lines", sum(len(x) for _, x in views))
    else:                                           # TFLP / MIX: counted per view below
        for k in ("stated_lines", "withheld_lines", "local_views", "statements_views", "statements_leaf_views"):
            bump(k, 0)
        bump("statements_no_parents", len(no_par))
    w = (node_weight or aux_total / max(1, len(views))) * item.get("weight", 1.0)
    ws = [w] * len(views)
    if depth_weight and views:                      # same total per problem, split by depth
        ws = [x * item.get("weight", 1.0) for x in depth_split(node_weight * len(views) if node_weight else aux_total,
                                                               [depth[sq["id"]] for sq, _ in views], depth_weight)]
    out = []
    for (sq, stated), w in zip(views, ws):
        if mode == "plain" and stated:              # TFLN: TFL's line-order draw, nothing shown
            rng.shuffle([None] * len(stated))
            stated = []
        if ctx_view and (sq["id"] in no_par or (is_local_leaf(sq) if view == "local+ctxparse" else
                                                mix_statements(mix_seed, mix_epoch, item["item_id"], sq["id"], mix_p))):
            rng.shuffle([None] * len(stated))       # TFLP / MIX statements view: TFL's line-order draw, nothing shown
            ctx = block
            bump("statements_views")
            bump("statements_leaf_views", is_local_leaf(sq))
            bump("withheld_lines", len(stated))
        else:
            ctx = local_context(item, sq, stated, depth=depth, rng=rng,
                                order="depth" if view == "ancestors" else "shuffle")
            if ctx_view:
                bump("local_views")
                bump("stated_lines", len(stated))
        yes_first = rng.random() < 0.5
        a, b = ("Yes", "No") if yes_first else ("No", "Yes")
        out.append(Example(SUBQ_TEMPLATE.format(problem=ctx, question=sq["question"], a=a, b=b), ["A", "B"],
                           _soft(_node_p(sq, subq_target), yes_first), w, "subq", item["item_id"]))
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
                aux_active: bool = True, aux_answer: str = "all", subq_filter: str = "all") -> list[Example]:
    """One problem for exposure-matched grouped training: [final] + exactly k auxiliary views. aux_kind "cot" uses
    CoT nodes, "random" the matched control questions (fresh targets), "placebo" the same CoT views with zero loss
    weight (same forward/backward work, no auxiliary signal), "tree" CoT nodes asked with their parents' results
    stated, "tree_shuf" the same with unrelated nodes' results stated (edge-rewiring control). Missing views are
    zero-weight copies of the final. aux_active=False keeps the views but gives them zero weight (placebo for this
    item), so per-family auxiliary supervision stays exposure-matched. "cot_randlabel": the "cot" views with each
    target redrawn uniformly (yes/no) per view and epoch (format-only control). aux_answer "exclude"/"only" keeps only
    the CoT nodes that are not / are answer components (K&K "Is X a knight?"; treeplan.answer_component). subq_filter
    "path"/"mirror" asks only that chain's nodes (decision chains; for tree/tree_shuf the stated results may still
    come from any node)."""
    if aux_answer != "all":
        item = dict(item, subqs=[sq for sq in item.get("subqs", []) if answer_component(sq, item) == (aux_answer == "only")])
    if subq_filter != "all" and aux_kind == "cot_mc":
        item = dict(item, subqs=[sq for sq in item.get("subqs", []) if subq_keep(sq, subq_filter)])
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
                                     weight=aux_weight if aux_active else 0.0, subq_target=subq_target,
                                     subq_filter=subq_filter)
        return fin + aux + [dataclasses.replace(fin[0], weight=0.0) for _ in range(k - len(aux))]
    sub, tgt = ("random", "fresh") if aux_kind == "random" else ("cot", subq_target)
    views = build_examples(item, final=final, subq=sub, subq_target=tgt, lambda_sub=aux_weight, rng=rng,
                           permute_final=permute_final, subq_k=k, subq_weight="each", subq_filter=subq_filter)
    fin = [e for e in views if e.kind == "final"]
    if not fin:
        return []
    aux = [e for e in views if e.kind == "subq"][:k]
    if aux_kind == "cot_randlabel":
        aux = [dataclasses.replace(e, target=[1.0, 0.0] if rng.random() < 0.5 else [0.0, 1.0]) for e in aux]
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


# ---- --tree-pack (opt-in; nothing below runs on the default path) ------------------------------------------------------
PACK_CUE = "\n\nIntermediate question: "


@dataclass
class Pack:
    """--tree-pack: several GFP-S node views of one problem read in ONE causal pass. parts[0] is the first view's own
    text (statement block + its question, options and "Answer:" cue, exactly its unpacked text); each further part is
    a further view's question chunk (PACK_CUE + question + options + "Answer:"), so a later question sees the block and
    the earlier QUESTIONS, never an answer. nodes[k] is view k as built for GFP-S (labels, target, weight, kind,
    item_id; its own text is not read): the readout of part k is the next-token distribution at the last token of
    part k (its "Answer:" cue), restricted to the option letters, as for the unpacked view. Parts are tokenized one by
    one and concatenated (pack_ids), so a cue's last token is the unpacked one (no merge with the next chunk)."""
    parts: list[str]
    nodes: list[Example]
    item_id: str
    kind: str = "pack"

    @property
    def text(self) -> str:
        return "".join(self.parts)


def readouts(batch: list) -> list:
    """The examples whose letter logits label_logits returns, in order: each Pack's node views in its place (batch
    itself when it has no Pack)."""
    if not any(isinstance(e, Pack) for e in batch):
        return batch
    return [n for e in batch for n in (e.nodes if isinstance(e, Pack) else [e])]


def pack_ids(tokenizer, p: Pack) -> tuple[list[int], list[int]]:
    """A Pack's token ids (the special prefix, then each part tokenized on its own) and its readout positions (the last
    token of every part)."""
    ids, pos = list(special_prefix(tokenizer)), []
    for part in p.parts:
        ids += text_ids(tokenizer, part)
        pos.append(len(ids) - 1)
    return ids, pos


def pack_rng(seed: int, epoch: int, item_id) -> random.Random:
    """--tree-pack: the node order of one problem's packs in one epoch, from its own generator (str key through sha512,
    as treeplan.scaffold_rng), never the training rng, so a packed run draws GFP-S's nodes, option orders and targets."""
    return random.Random(f"pack|{seed}|{epoch}|{item_id}")


def pack_views(group: list[Example], tokenizer, block: str, *, max_len: int, max_nodes: int, rng: random.Random,
               stats: dict | None = None) -> list:
    """--tree-pack: a GFP-S full-tree group ([final] + node views, each view = block + PACK_CUE + its question) with its
    node views packed. The views are shuffled by rng (pack_rng) and filled greedily into Packs of at most max_nodes
    views and max_len tokens (special prefix included); every Pack re-reads the block. A Pack of one view stays the
    plain view, and so does a view whose own text exceeds max_len (collate cuts it as in GFP-S). Weights and targets
    are the views' own, so the update loss sums the same per-node terms as GFP-S; the other examples (the final
    question) come first, unchanged. stats counts pack_seqs, pack_nodes, pack_single and pack_long."""
    head = block + PACK_CUE
    nodes = [e for e in group if e.kind == "subq"]
    out = [e for e in group if e.kind != "subq"]
    for e in nodes:
        if not e.text.startswith(head):
            raise ValueError(f"{e.item_id}: --tree-pack needs GFP-S views (statement block + question)")
    order = nodes[:]
    rng.shuffle(order)
    st = stats if stats is not None else {}
    bump = lambda k, n=1: st.__setitem__(k, st.get(k, 0) + n)
    for k in ("pack_seqs", "pack_nodes", "pack_single", "pack_long"):
        bump(k, 0)
    n_pre = len(special_prefix(tokenizer))
    cur: list[Example] = []
    size = 0

    def flush():
        if len(cur) == 1:
            out.append(cur[0])
            bump("pack_single")
        elif cur:
            out.append(Pack([cur[0].text] + [e.text[len(block):] for e in cur[1:]], list(cur), cur[0].item_id))
            bump("pack_seqs")
            bump("pack_nodes", len(cur))
    for e in order:
        if cur and len(cur) < max_nodes:
            n = len(text_ids(tokenizer, e.text[len(block):]))
            if size + n <= max_len:
                cur.append(e)
                size += n
                continue
        flush()
        cur, size = [e], n_pre + len(text_ids(tokenizer, e.text))
        if size > max_len:                  # too long even alone: the plain view, cut by collate as in GFP-S
            out.append(e)
            bump("pack_long")
            cur, size = [], 0
    flush()
    return out


def seq_len(tokenizer, e, max_len: int) -> int:
    """Token length of an example as collate builds it (a Pack is never cut: pack_views keeps it within max_len)."""
    if isinstance(e, Pack):
        return len(pack_ids(tokenizer, e)[0])
    return min(max_len, len(special_prefix(tokenizer)) + len(text_ids(tokenizer, e.text)))


def collate(tokenizer, batch: list[Example], max_len: int, device) -> dict:
    if any(isinstance(e, Pack) for e in batch):     # --tree-pack: rows/pos index every readout (readouts(batch))
        return _collate_packs(tokenizer, batch, max_len, device)
    enc = [encode_tail(tokenizer, e.text, max_len) for e in batch]  # keep the question end (and a BOS if any)
    L = max(len(x) for x in enc)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = torch.full((len(enc), L), pad, dtype=torch.long)
    mask = torch.zeros((len(enc), L), dtype=torch.long)
    for i, x in enumerate(enc):
        ids[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device),
            "last": torch.tensor([len(x) - 1 for x in enc], device=device)}


def _collate_packs(tokenizer, batch: list, max_len: int, device) -> dict:
    """collate for a batch with Packs: the same right-padded rows (plain examples cut as encode_tail cuts them), plus
    rows/pos, the row and position of every readout in readouts(batch) order."""
    enc, rows, pos = [], [], []
    for i, e in enumerate(batch):
        if isinstance(e, Pack):
            x, ps = pack_ids(tokenizer, e)
            if len(x) > max_len:
                raise ValueError(f"{e.item_id}: a pack of {len(x)} tokens exceeds max_len {max_len}")
        else:
            x = encode_tail(tokenizer, e.text, max_len)
            ps = [len(x) - 1]
        enc.append(x)
        rows += [i] * len(ps)
        pos += ps
    L = max(len(x) for x in enc)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = torch.full((len(enc), L), pad, dtype=torch.long)
    mask = torch.zeros((len(enc), L), dtype=torch.long)
    for i, x in enumerate(enc):
        ids[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device),
            "last": torch.tensor([len(x) - 1 for x in enc], device=device),
            "rows": torch.tensor(rows, device=device), "pos": torch.tensor(pos, device=device)}


def token_budget_batches(lengths: list[int], budget: int, max_seqs: int = 16) -> list[list[int]]:
    """Split one update's sequences (token lengths as collate sees them) into micro-batches of indices, longest first:
    a micro-batch grows while (n + 1) x its longest length <= budget padded tokens and n < max_seqs; a sequence
    longer than the budget goes alone. Every index appears once. The update's gradient does not depend on the split
    (sum over sequences of w x CE / norm), so this only changes the number of forward passes and the padding."""
    out: list[list[int]] = []
    for i in sorted(range(len(lengths)), key=lambda j: -lengths[j]):
        if out and len(out[-1]) < max_seqs and (len(out[-1]) + 1) * lengths[out[-1][0]] <= budget:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def micro_batches(tokenizer, batch: list[Example], max_len: int, micro_bs: int, token_budget: int = 0,
                  max_seqs: int = 16, padded: list | None = None) -> list[list[Example]]:
    """One update's micro-batches: consecutive chunks of micro_bs, or with token_budget the length-sorted split of
    token_budget_batches (lengths counted as collate truncates them; padded, if a list, receives each micro-batch's
    padded token count)."""
    if not token_budget:
        return [batch[i:i + micro_bs] for i in range(0, len(batch), micro_bs)]
    n_pre = len(special_prefix(tokenizer))
    lens = [seq_len(tokenizer, e, max_len) if isinstance(e, Pack) else
            min(max_len, n_pre + len(text_ids(tokenizer, e.text))) for e in batch]
    groups = token_budget_batches(lens, token_budget, max_seqs)
    if padded is not None:
        padded.extend(len(b) * max(lens[j] for j in b) for b in groups)
    return [[batch[j] for j in b] for b in groups]


def label_logits(model, tokenizer, batch: list[Example], max_len: int, label_ids_cache: dict,
                 hidden_out: list | None = None, layers: list[int] | None = None) -> list[torch.Tensor]:
    """Logits over each example's option letters at the readout position (optionally also appends each
    example's readout hidden state, detached, to hidden_out; with layers, a [len(layers) + 1, hidden] stack of the
    readout position's states after those decoder layers plus the final readout state). A Pack (--tree-pack) gives one
    entry per node view, in readouts(batch) order."""
    if getattr(model, "cotd_readout", False):    # FSDP2 root (cotdistill.fsdp.Readout): this function runs in its forward
        return model.readout(tokenizer, batch, max_len, label_ids_cache, hidden_out, layers)
    body, head = text_parts(model)
    dev = head.weight.device
    b = collate(tokenizer, batch, max_len, dev)
    want_layers = bool(layers) and hidden_out is not None
    res = body(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=want_layers,
               use_cache=False)
    hidden = res.last_hidden_state
    rows, last = torch.arange(hidden.shape[0], device=dev), b["last"]
    if "pos" in b:                               # --tree-pack: every node view's "Answer:" cue of each pack
        rows, last, batch = b["rows"], b["pos"], readouts(batch)
    h = hidden[rows, last]
    if hidden_out is not None:
        if want_layers:
            per = [res.hidden_states[l][rows, last] for l in layers] + [h]
            hidden_out.extend(torch.stack(per, 1).detach().float().cpu())
        else:
            hidden_out.extend(h.detach().float().cpu())
    out, cap = [], logit_softcap(model)
    for i, e in enumerate(batch):
        key = tuple(e.labels)
        if key not in label_ids_cache:
            label_ids_cache[key] = torch.tensor(letter_token_ids(tokenizer, e.labels), device=dev)
        rows = label_ids_cache[key]
        bias = head.bias[rows] if getattr(head, "bias", None) is not None else None
        out.append(head_logits(h[i], head.weight[rows], bias, cap).float())
    return out


def _ce_sum(h: torch.Tensor, w: torch.Tensor, y: torch.Tensor, softcap: float | None = None) -> torch.Tensor:
    z = (h.to(w.dtype) @ w.T).float()
    if softcap:                      # Gemma: the model's own (capped) next-token distribution
        z = softcap * torch.tanh(z / softcap)
    return F.cross_entropy(z, y, reduction="sum")


def lm_loss(model, tokenizer, batch: list[Example], max_len: int, chunk: int = 512) -> torch.Tensor:
    """Sum over examples of weight x mean token cross-entropy on each continuation.

    The vocabulary projection runs in checkpointed chunks so full-vocabulary logits (248k for Qwen3.5)
    are never held for the whole sequence. Continuations longer than the budget keep their start and
    their last 64 tokens (where the answer is)."""
    body, head = text_parts(model)
    dev = head.weight.device
    seqs, starts = [], []
    for e in batch:
        pre = encode_tail(tokenizer, e.text, max_len // 2)
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
    hidden = body(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    total = hidden.new_zeros((), dtype=torch.float32)
    cap = logit_softcap(model)
    for i, (x, st) in enumerate(zip(seqs, starts)):
        h, y = hidden[i, st - 1:len(x) - 1], ids[i, st:len(x)]
        ce = sum(torch.utils.checkpoint.checkpoint(_ce_sum, h[j:j + chunk], head.weight, y[j:j + chunk], cap,
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


def load_model(path: str, dtype, attn_implementation: str | None = None):
    import transformers
    kw = {"attn_implementation": attn_implementation} if attn_implementation else {}
    cfg = transformers.AutoConfig.from_pretrained(path)
    if getattr(cfg, "model_type", None) == "gemma4_unified" and (cfg.vision_config is not None or
                                                                   cfg.audio_config is not None):
        # Gemma 4 12B "Unified" is encoder-free: images and audio enter through small linear embedders
        # (model.embed_vision / model.embed_audio, 52.4M params) beside the text decoder. With both sub-configs
        # removed, Gemma4UnifiedForConditionalGeneration builds the text decoder + tied head only (11.91B params),
        # loads it with the same keys and logits, skips the embedder weights as unexpected keys, and saves a
        # text-only checkpoint (vision_config/audio_config null) that loads back through this same path.
        # (Gemma4UnifiedForCausalLM.from_pretrained on this checkpoint finds no weights under its key prefix
        # model.* and silently returns a randomly initialised model; do not use it.)
        cfg.vision_config = None
        cfg.audio_config = None
        kw["config"] = cfg
        print("gemma4_unified: text-only load (vision/audio embedders dropped)", flush=True)
    try:
        m = transformers.AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, **kw)
    except Exception:
        m = transformers.AutoModelForImageTextToText.from_pretrained(path, torch_dtype=dtype, **kw)
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
    if getattr(model, "cotd_readout", False):    # FSDP2 root: the same batches, split across the ranks and gathered
        exs = [Example(FINAL_TEMPLATE.format(problem=it["prompt"]), it["labels"], [0.0] * len(it["labels"]), 1.0,
                       "final", it["item_id"]) for it in items]
        zs, hs = model.batched(tok, exs, order, bs, max_len, cache, hidden is not None, layers)
        for i, z in enumerate(zs):
            probs_by[i] = torch.softmax(z, -1).tolist()
            if hidden is not None:
                hid_by[i] = hs[i]
    else:
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
    Broken down by node type and source; yes/no order randomised per question as in training.
    Multiple-choice nodes (treeplan.is_mc_node) are asked with their own options (mc_node_example, no rng draw) and
    scored by argmax == answer; their preds carry probs, pred, answer and rank instead of p_yes (p_yes is P(answer)),
    and they also count in the group cot/mc."""
    rng = random.Random(seed)
    exs, meta = [], []
    for it in items:
        for kind, pool in (("cot", it.get("subqs", [])), ("control", it.get("random_subqs", []))):
            for sq in pool:
                if kind == "cot" and is_mc_node(sq):           # own options, as a final question; no rng draw
                    exs.append(mc_node_example(it["prompt"], sq, 1.0, it["item_id"]))
                    meta.append({"item_id": it["item_id"], "source": it.get("source"), "kind": kind,
                                 "type": sq.get("type") or kind, "truth_known": True, "target": sq["answer"],
                                 "mc": len(sq["options"]), "question": sq["question"], "var": sq.get("var")})
                    continue
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
                if kind == "cot" and "on_path" in sq and "node_depth" in sq:      # decision chains: chain x level,
                    tag = "path" if sq["on_path"] else "mirror"                  # chain x levels below the top,
                    meta[-1]["chain"] = f"{tag}/d{sq['node_depth']}"             # path nodes x item depth
                    if "depth" in it:
                        meta[-1]["chain_top"] = f"{tag}/top-{it['depth'] - sq['node_depth']}"
                        meta[-1]["chain_item"] = f"{tag}/item_d{it['depth']}"
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    order = sorted(range(len(exs)), key=lambda i: len(exs[i].text))
    p_yes = [0.0] * len(exs)
    mc_probs: dict[int, list[float]] = {}

    def keep(i: int, z) -> None:
        if "mc" in meta[i]:
            mc_probs[i] = torch.softmax(z.float(), -1).tolist()
            p_yes[i] = mc_probs[i][meta[i]["target"]]
            return
        pa = torch.softmax(z, -1)[0].item()
        p_yes[i] = pa if meta[i]["yes_first"] else 1 - pa
    if getattr(model, "cotd_readout", False):    # FSDP2 root: the same batches, split across the ranks and gathered
        for i, z in enumerate(model.batched(tok, exs, order, bs, max_len, cache)[0]):
            keep(i, z)
    else:
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                zs = label_logits(model, tok, [exs[i] for i in idx], max_len, cache)
            for i, z in zip(idx, zs):
                keep(i, z)
    if was_training:
        model.train()
    groups: dict[str, list[int]] = {}
    preds = []
    for i, (m, py) in enumerate(zip(meta, p_yes)):
        if "mc" in m:
            pr = mc_probs[i]
            pred = max(range(len(pr)), key=pr.__getitem__)           # the first maximum (the trainer's argmax)
            ok = int(pred == m["target"])
            rank = 1 + sum(1 for j, x in enumerate(pr) if x > pr[m["target"]] or (x == pr[m["target"]]
                                                                                 and j < m["target"]))
            preds.append(dict(m, p_yes=py, probs=pr, pred=pred, answer=m["target"], rank=rank, correct=ok))
            groups.setdefault("cot/mc", []).append(ok)
        else:
            ok = int((py > 0.5) == m["target"])
            preds.append(dict(m, p_yes=py, correct=ok))
        for g in ("all_" + m["kind"], f"{m['kind']}/{m['source']}", f"{m['kind']}/type/{m['type']}"):
            groups.setdefault(g, []).append(ok)
        if m["kind"] == "cot" and m["truth_known"] and "mc" not in m:
            groups.setdefault("cot/program_truth", []).append(ok)
        for g in ("chain", "chain_top", "chain_item"):
            if g in m:
                groups.setdefault(f"cot/{m[g]}", []).append(ok)
    metrics = {g: {"n": len(v), "acc": sum(v) / len(v)} for g, v in sorted(groups.items())}
    return metrics, preds

def local_eval_examples(items: list[dict], *, view: str = "local", seed: int = 0) -> tuple[list[Example], list[dict]]:
    """The held-out local-step probe: every node of the items that carry local fields (K&K in data/student_v3sl),
    asked in its local view with its TRUE parents' results (view "local") or all ancestors' results ("ancestors"),
    no problem text; target = program truth. Leaf (parse) nodes show their statement line(s). Views that would state
    the final answer (view_leak) are skipped, as in training."""
    rng = random.Random(seed)
    exs, meta = [], []
    for it in items:
        if it.get("local_preamble") is None:
            continue
        nodes = [sq for sq in it.get("subqs", []) if "id" in sq]
        by = {sq["id"]: sq for sq in nodes}
        depth = ({i: by[i]["node_depth"] for i in by} if all("node_depth" in sq for sq in nodes) else node_depths(nodes))
        anc = ancestors(nodes) if view == "ancestors" else None
        for sq in nodes:
            tgt = sq["truth"] if sq.get("truth") is not None else (None if sq.get("p_cot") is None else sq["p_cot"] > 0.5)
            if tgt is None:
                continue
            if is_local_leaf(sq):
                stated = []
            else:
                par = [x for x in (sq.get("depends_on") or []) if x in by and x != sq["id"]]
                ids = sorted(anc[sq["id"]] - {sq["id"]}) if view == "ancestors" else par
                stated = [by[j] for j in ids]
                if not par or not all(_fact_line(x) for x in stated) or view_leak(it, sq, stated):
                    continue
            ctx = local_context(it, sq, stated, depth=depth, rng=rng, order="depth" if view == "ancestors" else "shuffle")
            yes_first = rng.random() < 0.5
            a, b = ("Yes", "No") if yes_first else ("No", "Yes")
            exs.append(Example(SUBQ_TEMPLATE.format(problem=ctx, question=sq["question"], a=a, b=b), ["A", "B"],
                               [0.5, 0.5], 1.0, "subq", it["item_id"]))
            meta.append({"item_id": it["item_id"], "id": sq["id"], "type": sq.get("type") or "node",
                         "depth": depth[sq["id"]], "n_stated": len(stated), "target": bool(tgt),
                         "trivial": None if is_local_leaf(sq) else sq.get("local_trivial"),
                         "yes_first": yes_first, "question": sq["question"]})
    return exs, meta


def _depth_bin(d: int) -> str:
    for lo, hi in ((0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16)):
        if lo <= d <= hi:
            return f"{lo}" if lo == hi else f"{lo}-{hi}"
    return "17+"


@torch.no_grad()
def evaluate_local(model, tok, items: list[dict], max_len: int, bs: int, cache: dict, *, view: str = "local",
                   seed: int = 0) -> tuple[dict, list[dict]]:
    """Accuracy on local_eval_examples (does the student apply each local operator from its parents' results alone?),
    overall and by node type, exact depth and depth bin, and for derived nodes split into those their own hypothesis
    decides (derived_trivial, data field local_trivial) and those that need the stated results (derived_nontrivial,
    the operator test proper). Yes/no order randomised per question."""
    exs, meta = local_eval_examples(items, view=view, seed=seed)
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    order = sorted(range(len(exs)), key=lambda i: len(exs[i].text))
    p_yes = [0.0] * len(exs)
    if getattr(model, "cotd_readout", False):    # FSDP2 root: the same batches, split across the ranks and gathered
        for i, z in enumerate(model.batched(tok, exs, order, bs, max_len, cache)[0]):
            pa = torch.softmax(z, -1)[0].item()
            p_yes[i] = pa if meta[i]["yes_first"] else 1 - pa
    else:
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
        gs = ["all", f"type/{m['type']}", f"depth/{m['depth']}", f"depth_bin/{_depth_bin(m['depth'])}",
              f"type_depth_bin/{m['type']}/{_depth_bin(m['depth'])}"]
        if m["trivial"] is not None:     # derived nodes: decided by their own hypothesis, or needing stated results
            gs.append("derived_trivial" if m["trivial"] else "derived_nontrivial")
            if not m["trivial"]:
                gs.append(f"nontrivial_depth_bin/{_depth_bin(m['depth'])}")
        for g in gs:
            groups.setdefault(g, []).append(ok)
    metrics = {g: {"n": len(v), "acc": sum(v) / len(v)} for g, v in sorted(groups.items())}
    return metrics, preds
