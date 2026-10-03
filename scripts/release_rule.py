"""The release-candidate rule of notes/prereg_release.md §1, computed from prediction files.

A candidate passes when
- its mean of BBH, MuSR and JevBench public is above that of A2-v4 and above that of TF-ep2, and
- none of BBH, MuSR, JevBench public, TD and K&K diag A drops by more than 2 points against the better of the two.
K&K diag A is the accuracy on the R1, R2 and C items of the locked diagnostic (data/eval/diag_kk.jsonl groups diag/R1,
diag/R2, diag/C). Each model is a comma-separated list of directories; a set is read from the first that has it.

  python scripts/release_rule.py --ref A2-v4=results/preds_2026-10-03/A2-v4-... TF-ep2=results/preds_2026-10-03/TF-ep2-... \\
      --cand M1=runs/wise-M1-test/a0.8 M2=runs/wise-M2-test/a1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SETS = ["bbh", "musr", "jevbench", "td", "diagA"]
MEAN_SETS = ["bbh", "musr", "jevbench"]
DIAG_A = ("diag/R1", "diag/R2", "diag/C")


def acc(dirs: list[Path], name: str) -> float | None:
    f = next((d / f"preds_{'diag' if name == 'diagA' else name}.jsonl" for d in dirs
              if (d / f"preds_{'diag' if name == 'diagA' else name}.jsonl").exists()), None)
    if f is None:
        return None
    rows = [json.loads(l) for l in open(f)]
    if name == "diagA":
        rows = [r for r in rows if r.get("group") in DIAG_A]
    ok = [r["labels"][max(range(len(r["probs"])), key=r["probs"].__getitem__)] == r["gold_label"] for r in rows]
    return sum(ok) / len(ok)


def scores(spec: str) -> tuple[str, dict]:
    name, paths = spec.split("=", 1)
    dirs = [Path(p) for p in paths.split(",")]
    return name, {s: acc(dirs, s) for s in SETS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", nargs=2, required=True, help="A2-v4=dirs TF-ep2=dirs")
    ap.add_argument("--cand", nargs="+", required=True, help="name=dirs")
    args = ap.parse_args()
    refs = dict(scores(s) for s in args.ref)
    cands = dict(scores(s) for s in args.cand)
    mean = lambda d: sum(d[s] for s in MEAN_SETS) / len(MEAN_SETS)
    print("model".ljust(10) + "".join(s.rjust(10) for s in SETS) + "  mean(bbh,musr,jev)")
    for n, d in {**refs, **cands}.items():
        print(n.ljust(10) + "".join(f"{d[s]:10.3f}" if d[s] is not None else "         -" for s in SETS)
              + (f"  {mean(d):.4f}" if all(d[s] is not None for s in MEAN_SETS) else "  -"))
    best = {s: max(r[s] for r in refs.values()) for s in SETS}
    for n, d in cands.items():
        missing = [s for s in SETS if d[s] is None]
        if missing:
            print(f"{n}: cannot judge, missing {missing}")
            continue
        above = {r: mean(d) > mean(refs[r]) for r in refs}
        drops = {s: round(d[s] - best[s], 4) for s in SETS if d[s] < best[s] - 0.02}
        ok = all(above.values()) and not drops
        print(f"{n}: {'PASS' if ok else 'FAIL'} | mean {mean(d):.4f} above "
              + ", ".join(f"{r} ({mean(refs[r]):.4f}): {'yes' if v else 'no'}" for r, v in above.items())
              + f" | drops > 2 points vs the better reference: {drops or 'none'}")


if __name__ == "__main__":
    main()
