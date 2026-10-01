"""Preregistered E2 / tree-transition analysis (notes/prereg_E2.md, notes/prereg_T.md) on the locked diag_kk set.

Per run: accuracy on R0, R1, R2 and C; the primary endpoint A = mean over the 400 programs of the mean correctness over
{R1, R2, C}; both R0 and C correct; all of R0, R1, R2 correct; NLL and Brier on diag; accuracy on the secondary sets.
Per contrast (e.g. G3-G4): the paired difference in A for every seed (block), their mean and range, and a paired
program-cluster bootstrap (same resampled programs for both arms and every seed; conditional on the trained runs).

  python scripts/analyze_e2.py --run G3 0 results/cloud/G3-...-s0 --run G4 0 results/cloud/G4-...-s0 \
      --run G0 0 results/cloud/G0-...-s0 --contrast G3-G4 --contrast G3-G0
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

SECONDARY = ["val", "kk", "kkdeep", "jl", "jevbench", "td", "bbh", "musr", "bbeh", "policy", "sharc", "folio", "gsm8k",
             "val_facts"]
SHIFTED = ("R1", "R2", "C")


def read_preds(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def correct(p: dict) -> bool:
    return p["labels"][max(range(len(p["probs"])), key=p["probs"].__getitem__)] == p["gold_label"]


def diag_table(run: Path) -> dict[str, dict[str, dict]]:
    """program -> renderer -> {correct, p_gold}"""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for p in read_preds(run / "preds_diag.jsonl"):
        prog, tag = p["item_id"].rsplit("-", 1)
        out[prog][tag] = {"correct": correct(p), "p_gold": p["probs"][p["labels"].index(p["gold_label"])],
                          "probs": p["probs"], "gold": p["labels"].index(p["gold_label"])}
    return out


def primary(table: dict, progs: list[str]) -> float:
    return sum(sum(table[g][r]["correct"] for r in SHIFTED) / 3 for g in progs) / len(progs)


def run_summary(run: Path) -> dict:
    t = diag_table(run)
    progs = sorted(t)
    s = {"programs": len(progs)}
    for r in ("R0", "R1", "R2", "C"):
        s[r] = sum(t[g][r]["correct"] for g in progs) / len(progs)
    s["A"] = primary(t, progs)
    s["R0&C"] = sum(t[g]["R0"]["correct"] and t[g]["C"]["correct"] for g in progs) / len(progs)
    s["R0&R1&R2"] = sum(all(t[g][r]["correct"] for r in ("R0", "R1", "R2")) for g in progs) / len(progs)
    s["indep(R0)xC"] = s["R0"] * s["C"]               # expected R0&C if the two were unrelated
    cells = [t[g][r] for g in progs for r in ("R0", "R1", "R2", "C")]
    s["nll"] = -sum(math.log(max(c["p_gold"], 1e-12)) for c in cells) / len(cells)
    s["brier"] = sum(sum((q - (i == c["gold"])) ** 2 for i, q in enumerate(c["probs"])) for c in cells) / len(cells)
    for name in SECONDARY:
        f = run / f"preds_{name}.jsonl"
        if f.exists():
            ps = [p for p in read_preds(f) if p.get("gold_label") in p["labels"]]
            if name == "bbh":                         # macro-average over the BBH tasks
                by = defaultdict(list)
                for p in ps:
                    by[(p.get("group") or "").split("/")[0] + "/" + (p.get("group") or "").split("/")[-1]].append(correct(p))
                s[name] = sum(sum(v) / len(v) for v in by.values()) / len(by)
            else:
                s[name] = sum(correct(p) for p in ps) / len(ps)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=3, action="append", metavar=("ARM", "SEED", "DIR"), required=True)
    ap.add_argument("--contrast", action="append", default=[], help="ARM1-ARM2, paired within seeds")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    runs = {(a, s): Path(d) for a, s, d in args.run}
    summ = {k: run_summary(v) for k, v in runs.items()}
    cols = ["A", "R0", "R1", "R2", "C", "R0&C", "indep(R0)xC", "R0&R1&R2", "nll", "brier"]
    print("diag (400 programs)".ljust(14) + "".join(c.rjust(12) for c in cols))
    for (a, s), m in sorted(summ.items()):
        print(f"{a} s{s}".ljust(14) + "".join(f"{m[c]:12.3f}" for c in cols))
    sec = [c for c in SECONDARY if any(c in m for m in summ.values())]
    print("\nsecondary".ljust(14) + "".join(c[:9].rjust(10) for c in sec))
    for (a, s), m in sorted(summ.items()):
        print(f"{a} s{s}".ljust(14) + "".join((f"{m[c]:10.3f}" if c in m else "-".rjust(10)) for c in sec))
    tables = {k: diag_table(v) for k, v in runs.items()}
    for con in args.contrast:
        a1, a2 = con.split("-")
        seeds = sorted(s for (a, s) in runs if a == a1 and (a2, s) in runs)
        if not seeds:
            continue
        progs = sorted(tables[(a1, seeds[0])])
        d = {s: primary(tables[(a1, s)], progs) - primary(tables[(a2, s)], progs) for s in seeds}
        rng = random.Random(0)
        boots = []
        for _ in range(args.boot):
            sample = [progs[rng.randrange(len(progs))] for _ in progs]
            boots.append(sum(primary(tables[(a1, s)], sample) - primary(tables[(a2, s)], sample) for s in seeds) / len(seeds))
        boots.sort()
        lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
        mean = sum(d.values()) / len(d)
        print(f"\n{con}: per seed " + ", ".join(f"s{s} {v:+.3f}" for s, v in d.items()) +
              f" | mean {mean:+.3f} (range {min(d.values()):+.3f}..{max(d.values()):+.3f})"
              f" | program bootstrap 95% [{lo:+.3f}, {hi:+.3f}] (conditional on these runs)")
        gate = mean >= 0.03 and min(d.values()) >= 0
        print(f"  engineering gate (>= +3 pp, no negative seed): {'met' if gate else 'not met'} on {len(seeds)} seed(s)")


if __name__ == "__main__":
    main()
