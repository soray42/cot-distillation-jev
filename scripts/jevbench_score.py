"""JevBench scores from our prediction files, by the item rules of the official methods, on the public items only.

- v1.4.2 item rule: argmax for every type (ties go to the lexicographically smallest label). The "Intelligence" proxy is
  the tier accuracy chance-corrected against the mean 1/K of the tier, clipped at 0, weighted easy .14 / standard .28 /
  hard .30 and renormalised (the judge tier, weight .28, is not public).
- v1.5 item rules: Choice by argmax; Noul by a band on P(yes) (<= .20 is No, >= .80 is Yes, otherwise an abstention,
  scored wrong); Score by the expected level, nMAE = |EV - gold| / (K - 1). Chance correction per type and tier,
  CC = 100 (s - c) / (1 - c), not clipped: c is the mean 1/K for Choice, .5 for Noul, and for Score the mean of
  1 - nMAE of a uniformly random level. Tier weights 10 / 20 / 40 (judge 30 not public), renormalised over the tiers
  a type has; the three types weigh 1/3 each (the equal-types amendment).
These are proxies on the 231 public items (data/raw/jevbench/METHOD-v1.5.md; notes/next_round/jevbench_errors.md §3),
not board numbers. An optional temperature divides the letter logits first (fit it on non-JevBench data only).

  python scripts/jevbench_score.py name=dir [name=dir ...] [--temperature T]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIER_W14 = {"easy": .14, "original": .28, "hard": .30}
TIER_W15 = {"easy": 10, "original": 20, "hard": 40}


def items() -> dict[str, dict]:
    return {json.loads(l)["item_id"]: json.loads(l) for l in open(ROOT / "data/eval/jevbench_public.jsonl")}


def soften(probs: list[float], T: float) -> list[float]:
    if T == 1.0:
        return probs
    z = [math.log(max(p, 1e-12)) / T for p in probs]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    return [x / sum(e) for x in e]


def score(pred_dir: str, T: float = 1.0) -> dict:
    its = items()
    by14, by15 = {}, {}
    for line in open(Path(pred_dir) / "preds_jevbench.jsonl"):
        p = json.loads(line)
        it = its[p["item_id"]]
        tier = it["group"].split("/")[0]
        labels, probs = p["labels"], soften(p["probs"], T)
        K = len(labels)
        top = sorted(range(K), key=lambda i: (-probs[i], labels[i]))[0]
        ok = labels[top] == p["gold_label"]
        by14.setdefault(tier, []).append((ok, 1 / K))
        t = it["type"]
        if t == "noul":
            names = it.get("label_names") or {}
            yes = next(L for L, n in names.items() if str(n).lower() == "yes")
            py = probs[labels.index(yes)]
            ans = "yes" if py >= .8 else "no" if py <= .2 else None
            s, c = float(ans is not None and ans == str(names[p["gold_label"]]).lower()), .5
        elif t == "score":
            order = it.get("label_order") or labels
            lv = {L: order.index(L) for L in labels}
            ev = sum(pr * lv[L] for L, pr in zip(labels, probs))
            g = lv[p["gold_label"]]
            s = 1 - abs(ev - g) / (K - 1)
            c = sum(1 - abs(k - g) / (K - 1) for k in range(K)) / K
        else:
            s, c = float(ok), 1 / K
        by15.setdefault(t, {}).setdefault(tier, []).append((s, c))

    def cc(rows: list[tuple[float, float]]) -> float:
        a = sum(r[0] for r in rows) / len(rows)
        ch = sum(r[1] for r in rows) / len(rows)
        return 100 * (a - ch) / (1 - ch)
    acc = sum(ok for rows in by14.values() for ok, _ in rows) / sum(len(r) for r in by14.values())
    w = {k: v for k, v in TIER_W14.items() if k in by14}
    i14 = sum(w[k] * max(0.0, cc(by14[k])) for k in w) / sum(w.values())
    per_type = {}
    for t, tiers in by15.items():
        w = {k: TIER_W15[k] for k in tiers}
        per_type[t] = sum(w[k] * cc(tiers[k]) for k in w) / sum(w.values())
    i15 = sum(per_type.values()) / len(per_type)
    return {"acc": acc, "v14": i14, "v15": i15, "v15_types": per_type}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=prediction directory")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()
    print(f"{'model':18s} {'argmax acc':>10s} {'v1.4 proxy':>10s} {'v1.5 proxy':>10s}   choice / noul / score (v1.5 CC)")
    for r in args.runs:
        name, d = r.split("=", 1)
        s = score(d, args.temperature)
        t = s["v15_types"]
        print(f"{name:18s} {s['acc']:10.3f} {s['v14']:10.1f} {s['v15']:10.1f}   "
              f"{t.get('choice', float('nan')):.1f} / {t.get('noul', float('nan')):.1f} / {t.get('score', float('nan')):.1f}")


if __name__ == "__main__":
    main()
