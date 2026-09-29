"""Compare answer diversity across solve-only runs (e.g. temperature, option permutation, effort).

For each run: per-trace accuracy, share of items where the K traces disagree, mean entropy of the
vote distribution over outcomes, majority-vote accuracy, and Brier / NLL of the (smoothed) vote
distribution against the gold outcome, plus the share of saturated per-trace answers.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize(run: str, alpha: float = 0.5) -> dict:
    d = ROOT / "teacher_cache" / run
    res = [json.loads(p.read_text()) for p in sorted(d.glob("*-*.json"))]
    calls = [json.loads(l) for l in open(d / "calls.jsonl")] if (d / "calls.jsonl").exists() else []
    acc, sat, disagree, ent, maj, brier, nll, toks = [], [], [], [], [], [], [], []
    for r in res:
        gold, outcomes = r["item"]["gold"], list(r["item"]["options"])
        votes = Counter()
        for tr in r["traces"]:
            do = tr.get("dist_outcome") or {}
            if not do:
                continue
            a = max(do, key=do.get)
            votes[a] += 1
            acc.append(a == gold)
            sat.append(max(do.values()) >= 0.999)
            toks.append(tr["conf"]["n"])
        k = sum(votes.values())
        if not k:
            continue
        disagree.append(len(votes) > 1)
        p = {o: votes[o] / k for o in outcomes}
        ent.append(-sum(v * math.log(v) for v in p.values() if v > 0))
        maj.append(votes.most_common(1)[0][0] == gold)
        ps = {o: (votes[o] + alpha) / (k + alpha * len(outcomes)) for o in outcomes}   # Dirichlet-smoothed vote
        brier.append(sum((ps[o] - (o == gold)) ** 2 for o in outcomes))
        nll.append(-math.log(ps[gold]))
    n = len(disagree)
    return {"run": run, "items": n, "traces": len(acc), "trace_acc": sum(acc) / len(acc),
            "majority_acc": sum(maj) / n, "items_with_disagreement": sum(disagree) / n,
            "mean_vote_entropy": sum(ent) / n, "smoothed_vote_brier": sum(brier) / n,
            "smoothed_vote_nll": sum(nll) / n, "saturated_trace_share": sum(sat) / len(sat),
            "mean_reasoning_tokens": sum(toks) / len(toks), "usd": sum(c["usd"] for c in calls)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    rows = [summarize(r) for r in args.runs]
    keys = list(rows[0])
    print(" | ".join(f"{k:>10.10s}" for k in keys))
    for row in rows:
        print(" | ".join(f"{v:>10.3f}" if isinstance(v, float) else f"{str(v):>10.10s}" for v in row.values()))


if __name__ == "__main__":
    main()
