"""Dual-process cascade frontier: the student (System 1) answers when its top probability is at least tau,
otherwise the decision goes to the reasoning teacher (System 2).

For each tau: share deferred, accuracy, teacher USD per 1,000 decisions and mean teacher seconds per
decision, from the student's preds file and a teacher solve-only run on the same eval set. Items the
teacher has not answered count as teacher-wrong at the teacher's mean cost.

  python3 scripts/cascade.py --student runs/A3/preds_kk.jsonl --teacher teval_kk --eval data/eval/kk_heldout.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def teacher_table(run: str) -> dict[str, dict]:
    """item_id -> {correct-free prediction, usd, secs} from a solve-only run and its call log."""
    cost: dict[str, list[float]] = {}
    for line in open(ROOT / "teacher_cache" / run / "calls.jsonl"):
        c = json.loads(line)
        iid = c["tag"].split("/")[0]
        v = cost.setdefault(iid, [0.0, 0.0])
        v[0] += c.get("usd") or 0.0
        v[1] += c.get("secs") or 0.0
    out = {}
    for p in (ROOT / "teacher_cache" / run).glob("*.json"):
        res = json.loads(p.read_text())
        if "item" not in res or not res.get("traces"):
            continue
        iid = res["item"]["item_id"]
        d = res["traces"][0].get("dist_outcome") or {}
        usd, secs = cost.get(iid, [0.0, 0.0])
        out[iid] = {"pred": max(d, key=d.get) if d else None, "usd": usd, "secs": secs}
    return out


def frontier(student: list[dict], teacher: dict[str, dict], evalset: dict[str, dict], taus: list[float]) -> list[dict]:
    rows = [(s, evalset[s["item_id"]]) for s in student if s["item_id"] in evalset]
    known = [t for t in teacher.values()]
    mean_usd = sum(t["usd"] for t in known) / max(1, len(known))
    mean_secs = sum(t["secs"] for t in known) / max(1, len(known))
    out = []
    for tau in taus:
        n = right = defer = 0
        usd = secs = 0.0
        for s, it in rows:
            n += 1
            gold = it["gold_label"]
            conf = max(s["probs"])
            if conf >= tau:
                right += s["labels"][s["probs"].index(conf)] == gold
                continue
            defer += 1
            t = teacher.get(s["item_id"])
            usd += t["usd"] if t else mean_usd
            secs += t["secs"] if t else mean_secs
            right += bool(t) and t["pred"] == gold
        out.append({"tau": tau, "n": n, "defer": defer / n, "acc": right / n,
                    "usd_per_1k": 1000 * usd / n, "teacher_secs_per_decision": secs / n})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", nargs="+", required=True, help="preds_<set>.jsonl files (one curve each)")
    ap.add_argument("--teacher", required=True, help="teacher_cache solve-only run on the same eval set")
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", default=None, help="write all curves as JSON")
    args = ap.parse_args()
    evalset = {r["item_id"]: r for r in (json.loads(l) for l in open(args.eval) if l.strip())}
    teacher = teacher_table(args.teacher)
    taus = [0.0] + [round(0.3 + 0.05 * i, 2) for i in range(14)] + [0.99, 1.01]
    curves = {}
    for path in args.student:
        student = [json.loads(l) for l in open(path) if l.strip()]
        curves[path] = frontier(student, teacher, evalset, taus)
        print(f"== {path}")
        for r in curves[path]:
            print(f"  tau={r['tau']:.2f} defer={r['defer']:.3f} acc={r['acc']:.3f} "
                  f"usd/1k={r['usd_per_1k']:.3f} teacher_s={r['teacher_secs_per_decision']:.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps(curves, indent=1))


if __name__ == "__main__":
    main()
