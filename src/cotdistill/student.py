"""One-pass decision student: LM-head letter readout, KL to soft targets, sub-question supervision.

Every question (the final decision and each sub-question) is its own sequence that repeats the
problem text; hybrid linear-attention models (Qwen3.5) ignore attention masks, so we do not pack
questions with a block-causal mask. Sequences are right-padded; the readout is the next-token
distribution at the last real token, restricted to the option-letter tokens.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

SUBQ_TEMPLATE = "{problem}\n\nIntermediate question: {question}\nOptions:\nA) {a}\nB) {b}\nAnswer:"
FINAL_TEMPLATE = "{problem}\n\nAnswer:"


@dataclass
class Example:
    text: str
    labels: list[str]            # option letters, e.g. ["A", "B", "C"]
    target: list[float]          # soft target over labels (sums to 1)
    weight: float
    kind: str                    # "final" | "subq"
    item_id: str
    gold: int | None = None      # index of the gold label (final questions, if known)


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
                   rng: random.Random) -> list[Example]:
    """Turn one student-data record into training examples for a given arm.

    final: "none" | "teacher" (teacher answer distribution) | "gold" (one-hot gold)
    subq: "none" | "cot" (teacher-extracted sub-questions) | "random" (matched generic questions)
    subq_target: "fresh" (teacher answer without CoT) | "cot" (with CoT) | "truth" (program truth,
                 falling back to fresh when no truth is available)
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
        elif subq_target == "cot":
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


def kl_loss(logits: list[torch.Tensor], batch: list[Example]) -> torch.Tensor:
    """Weighted soft cross-entropy (= KL(target || p) + const)."""
    tot = 0.0
    for z, e in zip(logits, batch):
        t = torch.tensor(e.target, device=z.device)
        tot = tot + e.weight * -(t * F.log_softmax(z, -1)).sum()
    return tot / len(batch)


def calibration(probs: list[list[float]], gold: list[int], bins: int = 15) -> dict:
    """Accuracy, NLL, multiclass Brier and top-1 ECE (equal-width bins) against gold labels."""
    n = len(gold)
    conf = [max(p) for p in probs]
    acc = [int(max(range(len(p)), key=p.__getitem__) == g) for p, g in zip(probs, gold)]
    nll = -sum(math.log(max(p[g], 1e-12)) for p, g in zip(probs, gold)) / n
    brier = sum(sum((pi - (i == g)) ** 2 for i, pi in enumerate(p)) for p, g in zip(probs, gold)) / n
    ece = 0.0
    for k in range(bins):
        lo, hi = k / bins, (k + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (k == 0 and c == 0)]
        if idx:
            ece += len(idx) / n * abs(sum(acc[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    return {"n": n, "acc": sum(acc) / n, "nll": nll, "brier": brier, "ece": ece}
