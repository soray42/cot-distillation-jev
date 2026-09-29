"""Per-group scores for prediction files (student preds_<set>.jsonl or teacher runs).

Accuracy, NLL, Brier and ECE against gold labels; KL(gold || p) where the set ships a soft gold
(Typed Decisions); and the JevBench v1.5 rule that a Noul answer with P(yes) in [0.2, 0.8] counts as
an abstention scored wrong.

  python3 scripts/score_preds.py runs/A3-*/preds_jevbench.jsonl --eval data/eval/jevbench_public.jsonl
  python3 scripts/score_preds.py --teacher teval_jevbench --eval data/eval/jevbench_public.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.metrics import calibration  # noqa: E402


def teacher_preds(run: str) -> list[dict]:
    out = []
    for p in sorted((ROOT / "teacher_cache" / run).glob("*.json")):
        res = json.loads(p.read_text())
        if "item" not in res or not res.get("traces"):
            continue
        it, tr = res["item"], res["traces"][0]
        labels = it.get("labels") or [chr(65 + i) for i in range(len(it["label_order"]))]
        d = tr.get("dist_outcome") or {}
        probs = [float(d.get(L, 0.0)) for L in labels]
        out.append({"item_id": it["item_id"], "labels": labels, "probs": probs if sum(probs) > 0 else None})
    return out


def kl(gold: list[float], p: list[float]) -> float:
    return sum(g * (math.log(max(g, 1e-12)) - math.log(max(q, 1e-12))) for g, q in zip(gold, p) if g > 0)


def score(preds: list[dict], evalset: dict[str, dict]) -> dict:
    groups = collections.defaultdict(list)
    for pr in preds:
        it = evalset.get(pr["item_id"])
        if it is None:
            continue
        groups["all"].append((pr, it))
        groups[it.get("group") or "?"].append((pr, it))
        if it.get("type"):
            groups[f"type/{it['type']}"].append((pr, it))
    out = {}
    for g, rows in sorted(groups.items()):
        ok = [(pr, it) for pr, it in rows if pr.get("probs") and it.get("gold_label") in it["labels"]]
        m = calibration([pr["probs"] for pr, _ in ok], [it["labels"].index(it["gold_label"]) for _, it in ok]) \
            if ok else {"n": 0}
        m["n_missing_pred"] = len(rows) - len(ok)
        if m.get("n"):                                    # missing predictions count as wrong
            m["acc_all"] = m["acc"] * m["n"] / len(rows)
        soft = [(pr, it) for pr, it in ok if it.get("gold_probs")]
        if soft:
            m["kl_gold"] = sum(kl([it["gold_probs"][L] for L in it["labels"]], pr["probs"]) for pr, it in soft) / len(soft)
        noul = [(pr, it) for pr, it in ok if it.get("type") == "noul"]
        if noul:
            right = 0
            for pr, it in noul:
                yes = next((L for L, n in it["label_names"].items() if n in ("yes", "true")), None)
                p_yes = pr["probs"][it["labels"].index(yes)] if yes else None
                if p_yes is not None and not (0.2 <= p_yes <= 0.8):
                    right += (p_yes > 0.5) == (it["label_names"][it["gold_label"]] in ("yes", "true"))
            m["noul_acc_abstain_wrong"] = right / len(noul)
        out[g] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="*")
    ap.add_argument("--teacher", default=None, help="teacher_cache run to score instead of preds files")
    ap.add_argument("--eval", required=True)
    ap.add_argument("--groups", action="store_true", help="print every group, not just all/type")
    args = ap.parse_args()
    evalset = {r["item_id"]: r for r in (json.loads(l) for l in open(args.eval) if l.strip())}
    sources = [("teacher:" + args.teacher, teacher_preds(args.teacher))] if args.teacher else \
        [(p, [json.loads(l) for l in open(p) if l.strip()]) for p in args.preds]
    for name, preds in sources:
        res = score(preds, evalset)
        print(f"== {name}")
        for g, m in res.items():
            if args.groups or g == "all" or g.startswith("type/"):
                print(f"  {g:40s} " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                               for k, v in m.items()))


if __name__ == "__main__":
    main()
