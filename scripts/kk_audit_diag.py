"""Diagnostics of a K&K audit run from its saved outputs (CPU, no model, no torch): why IIA equals the random-subspace
value, what the inv retention measures, and whether the sites pass their full-exchange positive controls.

Per condition x layer x method, on the all-pairs test set (in_all) and on the jointly-correct test set (in_test):
  iia        patched answer == target (inv: target = base_gold)
  coinc      P(unpatched base_pred == target): the IIA of an intervention that changes nothing
  unchanged  patched answer == unpatched base_pred (for inv this is the retention of the model's own answer)
  chg / hit  pairs whose answer moved, and those that moved onto the target
A method whose `unchanged` is ~1 has iia == coinc whatever its subspace, so das == random exactly. `full` is the
positive control of the site: if the full-state exchange leaves the answer unchanged, no subspace of it can move it.

  python scripts/kk_audit_diag.py results/hpc/kkaudit/GFPS2B-s12 results/hpc/kkaudit/TFL2B-s12 [--layers 16,20]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(d: Path) -> dict:
    conds: dict = {}
    for line in open(d / "preds.jsonl"):
        r = json.loads(line)
        c = conds.setdefault(r["cond"], {"pairs": None, "preds": defaultdict(dict)})
        if r["kind"] == "pairs":
            c["pairs"] = r["pairs"]
        else:
            c["preds"][int(r["layer"])][r["method"]] = r["preds"]
    return conds


def stats(pairs: list[dict], preds: list[int], field: str, keep) -> dict:
    sel = [(p, k) for p, k in zip(pairs, preds) if keep(p)]
    n = max(1, len(sel))
    hit = lambda p, k: p.get(field) is not None and k == p[field]
    return {"n": len(sel), "iia": sum(hit(p, k) for p, k in sel) / n,
            "unchanged": sum(k == p["base_pred"] for p, k in sel) / n,
            "chg": sum(k != p["base_pred"] for p, k in sel),
            "hit": sum(k != p["base_pred"] and hit(p, k) for p, k in sel)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--layers", default="", help="comma list (default: all)")
    args = ap.parse_args()
    only = {int(x) for x in args.layers.split(",") if x}
    for d in map(Path, args.dirs):
        head = next((json.loads(l) for l in open(d / "summary.jsonl") if '"_head"' in l), {})
        print(f"===== {d.name}: base_acc {head.get('base_acc')} source_acc {head.get('source_acc')} "
              f"jointly_correct {head.get('jointly_correct')}")
        for cname, c in load(d).items():
            field = "base_gold" if cname.startswith("inv") else "target"
            P = c["pairs"]
            A = [p for p in P if p["in_all"]]
            coinc = sum(p["base_pred"] == p[field] for p in A) / max(1, len(A))
            acc = sum(p["base_ok"] for p in A) / max(1, len(A))
            joint = sum(p["base_ok"] and p["source_ok"] for p in A) / max(1, len(A))
            print(f"{cname:<22} n_all {len(A)}  base_acc {acc:.3f}  joint {joint:.3f}  coinc {coinc:.3f}")
            for L in sorted(c["preds"]):
                if only and L not in only:
                    continue
                cells = []
                for m, ks in c["preds"][L].items():
                    a = stats(P, ks, field, lambda p: p["in_all"])
                    t = stats(P, ks, field, lambda p: p["in_test"])
                    cells.append(f"{m} all {a['iia']:.3f} unch {a['unchanged']:.3f} chg {a['chg']} hit {a['hit']}"
                                 f" / joint {t['iia']:.3f}")
                print(f"   L{L:<2} " + " | ".join(cells))


if __name__ == "__main__":
    main()
