"""Torch-free metrics shared by training, evaluation and analysis scripts."""
from __future__ import annotations

import math


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
