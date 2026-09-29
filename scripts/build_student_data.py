"""Convert teacher_cache/<run>/ results into student training data.

Each output record: prompt (problem with lettered options), labels, gold_label, teacher (answer
distribution over letters, averaged over traces after mapping permuted orders back), subqs (question,
p_cot = mean P(yes) with CoT in context, p_fresh = P(yes) without CoT, truth, span_minp, status),
random_subqs (matched-count control questions with p_fresh). Split into train/val by item-id hash.

  python3 scripts/build_student_data.py --runs hard1 hard2 --out data/student --val-frac 0.1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.teacher import resolve_p_yes  # noqa: E402


def teacher_dist(res: dict) -> dict[str, float]:
    order = res["item"]["label_order"]
    acc = {k: 0.0 for k in order}
    n = 0
    for tr in res["traces"]:
        do = tr.get("dist_outcome")
        if do is None:                      # older runs: letters in the canonical order
            do = {order[ord(l) - 65]: p for l, p in tr["dist"]["probs"].items()}
        if do:
            n += 1
            for k, p in do.items():
                acc[k] += p
    return {chr(65 + i): acc[k] / n for i, k in enumerate(order)} if n else {}


def convert(res: dict) -> dict:
    it = res["item"]
    labels = [chr(65 + i) for i in range(len(it["label_order"]))]
    subqs = []
    for sq in res.get("subquestions", []):
        ps = [p for p in (resolve_p_yes(a) for a in sq.get("answers", [])) if p is not None]
        subqs.append({"question": sq["question"], "p_cot": sum(ps) / len(ps) if ps else None,
                      "p_fresh": resolve_p_yes(sq.get("answer_nocot") or {}), "truth": sq.get("truth"),
                      "span_minp": (sq.get("span_stats") or {}).get("min_p"), "status": sq.get("status")})
    randoms = [{"question": r["question"], "p_fresh": resolve_p_yes(r.get("answer_nocot") or {})}
               for r in res.get("random_subquestions", [])]
    return {"item_id": it["item_id"], "source": it.get("domain") or it.get("source"), "prompt": res["prompt"],
            "labels": labels, "gold_label": it.get("gold_label"), "teacher": teacher_dist(res),
            "depth": it.get("depth"), "subqs": subqs, "random_subqs": randoms,
            "rationale": (res["traces"][0].get("reasoning") or None) if res.get("traces") else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="data/student")
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "val": []}
    for run in args.runs:
        for p in sorted((ROOT / "teacher_cache" / run).glob("*.json")):
            if p.name in ("summary.json",):
                continue
            res = json.loads(p.read_text())
            if "item" not in res:
                continue
            rec = convert(res)
            h = int(hashlib.sha1(rec["item_id"].encode()).hexdigest(), 16) % 1000 / 1000
            splits["val" if h < args.val_frac else "train"].append(rec)
    for name, recs in splits.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        n_sq = sum(len(r["subqs"]) for r in recs)
        print(f"{name}: {len(recs)} items, {n_sq} sub-questions, "
              f"{sum(len(r['random_subqs']) for r in recs)} random sub-questions")


if __name__ == "__main__":
    main()
