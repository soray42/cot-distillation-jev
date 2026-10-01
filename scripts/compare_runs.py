"""Compare student runs (and the untrained base) on every eval set, with paired bootstrap against a reference.

Reads <runs_dir>/<run>/preds_<set>.jsonl and the eval files. Prints, per eval set, accuracy / NLL / Brier /
ECE per run, accuracy by depth for the K&K and JustLogic held-out sets, and for each run vs --ref the paired
accuracy difference with a 95% bootstrap interval (items resampled; the same draws for all runs).

  python3 scripts/compare_runs.py --runs-dir results/runs --ref A2 --out results/compare.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.metrics import calibration  # noqa: E402

EVAL_FILES = {"val": None, "kk": "data/eval/kk_heldout.jsonl", "jl": "data/eval/jl_heldout.jsonl",
              "jevbench": "data/eval/jevbench_public.jsonl", "td": "data/eval/typed_decisions_test.jsonl",
              "bbeh": "data/eval/bbeh.jsonl", "bbh": "data/eval/bbh.jsonl", "musr": "data/eval/musr.jsonl",
              "policy": "data/eval/policy_heldout.jsonl", "sharc": "data/eval/sharc_dev.jsonl",
              "folio": "data/eval/folio_val.jsonl", "kkdeep": "data/eval/kk_deep.jsonl"}


def short(run: str) -> str:
    return run.split("-Qwen")[0] if run.startswith("A") else run


def load_preds(path: Path) -> dict[str, dict]:
    return {p["item_id"]: p for p in (json.loads(l) for l in open(path) if l.strip())}


def correct_vec(preds: dict[str, dict], ids: list[str], gold: dict[str, str]) -> list[int]:
    out = []
    for i in ids:
        p = preds.get(i)
        if not p or not p.get("probs"):
            out.append(0)
            continue
        k = max(range(len(p["probs"])), key=p["probs"].__getitem__)
        out.append(int(p["labels"][k] == gold[i]))
    return out


def bootstrap(a: list[int], b: list[int], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    m = len(a)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        diffs.append(sum(a[i] - b[i] for i in idx) / m)
    diffs.sort()
    return sum(x - y for x, y in zip(a, b)) / m, diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="results/runs")
    ap.add_argument("--ref", default="A2", help="short run name to compare against (e.g. A2, base)")
    ap.add_argument("--val", default="data/student_tree/val.jsonl", help="records for the val set")
    ap.add_argument("--match", default=None, help="regex: only runs whose directory name matches")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rd = ROOT / args.runs_dir
    runs = sorted(p.name for p in rd.iterdir() if p.is_dir() and any(p.glob("preds_*.jsonl"))
                  and (not args.match or re.search(args.match, p.name)))
    report = {}
    for es, path in EVAL_FILES.items():
        path = args.val if es == "val" else path
        if not (ROOT / path).exists():
            continue
        items = {r["item_id"]: r for r in (json.loads(l) for l in open(ROOT / path) if l.strip())}
        have = {r: load_preds(rd / r / f"preds_{es}.jsonl") for r in runs if (rd / r / f"preds_{es}.jsonl").exists()}
        if not have:
            continue
        ids = sorted(i for i in items if items[i].get("gold_label") in items[i]["labels"])
        gold = {i: items[i]["gold_label"] for i in ids}
        print(f"\n=== {es} (n={len(ids)})")
        vecs, rows = {}, {}
        for r, preds in have.items():
            ok = [i for i in ids if i in preds and preds[i].get("probs")]
            m = calibration([preds[i]["probs"] for i in ok], [items[i]["labels"].index(gold[i]) for i in ok]) if ok else {}
            vecs[r] = correct_vec(preds, ids, gold)
            m["acc_all"] = sum(vecs[r]) / len(ids)
            rows[r] = m
            print(f"  {short(r):28s} acc={m['acc_all']:.3f} nll={m.get('nll', float('nan')):.3f} "
                  f"brier={m.get('brier', float('nan')):.3f} ece={m.get('ece', float('nan')):.3f}")
        exact = [r for r in have if short(r) == args.ref or r == args.ref]
        prefix = [r for r in have if r.startswith(args.ref + "-")]
        ref = exact[0] if exact else (prefix[0] if len(prefix) == 1 else None)   # never guess between several
        if ref:
            for r in have:
                if r != ref:
                    d, lo, hi = bootstrap(vecs[r], vecs[ref])
                    rows[r][f"vs_{args.ref}"] = [d, lo, hi]
                    print(f"    {short(r):26s} - {args.ref}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        if es in ("kk", "jl", "kkdeep"):
            depth = collections.defaultdict(list)
            for n, i in enumerate(ids):
                depth[items[i].get("depth")].append(n)
            print("  accuracy by depth: " + "  ".join(f"d{d}" for d in sorted(depth)))
            for r in have:
                accs = [sum(vecs[r][n] for n in depth[d]) / len(depth[d]) for d in sorted(depth)]
                rows[r]["by_depth"] = dict(zip(map(str, sorted(depth)), accs))
                print(f"  {short(r):28s} " + " ".join(f"{a:.2f}" for a in accs))
        report[es] = rows
    if args.out:
        (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
