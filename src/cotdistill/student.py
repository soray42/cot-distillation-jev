"""One-pass decision student: LM-head letter readout, KL to soft targets, sub-question supervision.

Every question (the final decision and each sub-question) is its own sequence that repeats the
problem text; hybrid linear-attention models (Qwen3.5) ignore attention masks, so we do not pack
questions with a block-causal mask. Sequences are right-padded; the readout is the next-token
distribution at the last real token, restricted to the option-letter tokens.
"""
from __future__ import annotations

import random
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


def build_examples(item: dict, *, final: str, subq: str, subq_target: str, lambda_sub: float,
                   rng: random.Random, rationale_lm: bool = False, lambda_lm: float = 1.0) -> list[Example]:
    """Turn one student-data record into training examples for a given arm.

    final: "none" | "teacher" (teacher answer distribution) | "gold" (one-hot gold)
    subq: "none" | "cot" (teacher-extracted sub-questions) | "random" (matched generic questions)
    subq_target: "fresh" (teacher answer without CoT) | "cot" (with CoT) | "truth" (program truth,
                 falling back to the CoT answer, then the fresh one, when no truth is available) |
                 "commit" (the teacher's value-commitment confidence where the CoT settled the node)
    rationale_lm: add a token-level LM example on the teacher's reasoning and answer (DHRD-style baseline)
    """
    ex: list[Example] = []
    labels = item["labels"]
    gold = labels.index(item["gold_label"]) if item.get("gold_label") in labels else None
    if final == "teacher" and item.get("teacher"):
        t = [item["teacher"].get(L, 0.0) for L in labels]
        s = sum(t)
        if s > 0:
            ex.append(Example(FINAL_TEMPLATE.format(problem=item["prompt"]), labels, [x / s for x in t], 1.0,
                              "final", item["item_id"], gold))
    elif final == "gold" and gold is not None:
        ex.append(Example(FINAL_TEMPLATE.format(problem=item["prompt"]), labels,
                          [1.0 if i == gold else 0.0 for i in range(len(labels))], 1.0, "final", item["item_id"], gold))
    pool = {"cot": item.get("subqs", []), "random": item.get("random_subqs", [])}.get(subq, [])
    usable = []
    for sq in pool:
        p = None
        if subq_target == "truth" and sq.get("truth") is not None:
            p = 1.0 if sq["truth"] else 0.0
        elif subq_target == "commit" and sq.get("p_commit") is not None:
            p = sq["p_commit"]
        elif subq_target in ("cot", "truth", "commit"):
            p = sq.get("p_cot")
        if p is None:
            p = sq.get("p_fresh")
        if p is not None:
            usable.append((sq["question"], p))
    for q, p in usable:
        yes_first = rng.random() < 0.5
        a, b = ("Yes", "No") if yes_first else ("No", "Yes")
        ex.append(Example(SUBQ_TEMPLATE.format(problem=item["prompt"], question=q, a=a, b=b), ["A", "B"],
                          _soft(p, yes_first), lambda_sub / max(1, len(usable)), "subq", item["item_id"]))
    if rationale_lm and item.get("rationale") and item.get("teacher"):
        answer = max(item["teacher"], key=item["teacher"].get)
        ex.append(Example(RATIONALE_PREFIX.format(problem=item["prompt"]), labels, [], lambda_lm, "lm",
                          item["item_id"], continuation=f" {item['rationale'].strip()}\n\nAnswer: {answer}"))
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


def label_logits(model, tokenizer, batch: list[Example], max_len: int, label_ids_cache: dict) -> list[torch.Tensor]:
    """Logits over each example's option letters at the readout position."""
    body, head = text_parts(model)
    dev = head.weight.device
    b = collate(tokenizer, batch, max_len, dev)
    hidden = body(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
    h = hidden[torch.arange(hidden.shape[0], device=dev), b["last"]]
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
def evaluate(model, tok, items: list[dict], max_len: int, bs: int, cache: dict) -> tuple[dict, list[dict]]:
    """Final-question predictions for eval records (prompt, labels, gold_label; optional gold_probs, group).

    Items are batched by length to limit padding; predictions come back in input order."""
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    order = sorted(range(len(items)), key=lambda i: len(items[i]["prompt"]))
    probs_by = {}
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        batch = [Example(FINAL_TEMPLATE.format(problem=items[i]["prompt"]), items[i]["labels"],
                         [0.0] * len(items[i]["labels"]), 1.0, "final", items[i]["item_id"]) for i in idx]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            zs = label_logits(model, tok, batch, max_len, cache)
        for i, z in zip(idx, zs):
            probs_by[i] = torch.softmax(z, -1).tolist()
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

