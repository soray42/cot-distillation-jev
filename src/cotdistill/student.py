"""One-pass decision student: LM-head letter readout, KL to soft targets, sub-question supervision.

Every question (the final decision and each sub-question) is its own sequence that repeats the
problem text; hybrid linear-attention models (Qwen3.5) ignore attention masks, so we do not pack
questions with a block-causal mask. Sequences are right-padded; the readout is the next-token
distribution at the last real token, restricted to the option-letter tokens.
"""
from __future__ import annotations

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
                 hidden_out: list | None = None) -> list[torch.Tensor]:
    """Logits over each example's option letters at the readout position (optionally also appends each
    example's readout hidden state, detached, to hidden_out)."""
    body, head = text_parts(model)
    dev = head.weight.device
    b = collate(tokenizer, batch, max_len, dev)
    hidden = body(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
    h = hidden[torch.arange(hidden.shape[0], device=dev), b["last"]]
    if hidden_out is not None:
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


def kl_loss(logits: list[torch.Tensor], batch: list[Example]) -> torch.Tensor:
    """Weighted soft cross-entropy (= KL(target || p) + const)."""
    tot = 0.0
    for z, e in zip(logits, batch):
        t = torch.tensor(e.target, device=z.device)
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
             hidden: list | None = None) -> tuple[dict, list[dict]]:
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
            zs = label_logits(model, tok, batch, max_len, cache, hidden_out=hs)
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
                             "target": bool(tgt), "yes_first": yes_first, "question": sq["question"]})
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