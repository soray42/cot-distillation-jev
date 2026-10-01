"""Continuous learning-curve metrics from mid-training evaluations (metrics.json "trajectory").

The K&K diagnostic is bimodal across runs (learned .43-.67 vs not .17-.34), so a single end-of-training number is a
coin flip. Instead, per run:
  click step   the first evaluation step at which the diagnostic subset (diagA = R1/R2/C items of 150 programs, whose
               accuracy equals the primary endpoint A on that subset) reaches --threshold (.40, inside the gap);
               runs that never reach it are censored at their last step
  plateau      the mean diagA over the evaluations after the click
The val K&K subset (valkk) is reported alongside: checkpoint choices use it, never the diagnostic.

  python scripts/curve_metrics.py results/hpc/TF-c4-...-s11 results/hpc/TFM-c4-...-s11 ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--set", default="diagA")
    ap.add_argument("--threshold", type=float, default=0.40)
    args = ap.parse_args()
    print(f"{'run':52}{'click step':>11}{'plateau':>9}{'last':>7}  curve ({args.set} at each evaluation)")
    for run in args.runs:
        p = Path(run)
        traj = json.loads((p / "trajectory.json").read_text()) if (p / "trajectory.json").exists() else \
            json.loads((p / "metrics.json").read_text()).get("trajectory", [])
        pts = [(r["step"], r[args.set]["acc"]) for r in traj if args.set in r]
        if (p / "metrics.json").exists():                      # the end-of-training evaluation is the last point
            m = json.loads((p / "metrics.json").read_text())
            fin = m.get("eval", {}).get(args.set)
            if fin and m.get("history"):
                pts.append((m["history"][-1]["step"], fin["acc"]))
        if not pts:
            print(f"{p.name[:52]:52}  no {args.set} evaluations")
            continue
        click = next((s for s, a in pts if a >= args.threshold), None)
        after = [a for s, a in pts if click is not None and s > click]
        plateau = sum(after) / len(after) if after else None
        curve = " ".join(f"{a:.2f}" for _, a in pts)
        print(f"{p.name[:52]:52}{(str(click) if click else f'>{pts[-1][0]} (censored)'):>11}"
              f"{(f'{plateau:.3f}' if plateau is not None else '-'):>9}{pts[-1][1]:7.3f}  {curve}")


if __name__ == "__main__":
    main()
