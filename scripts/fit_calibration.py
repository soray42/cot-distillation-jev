"""Post-hoc calibration of a decision model's option probabilities, fitted on selection sets only.

Per question type (choice / noul / score) a temperature T, and for yes/no questions an additive bias b on the Yes
logit, are fitted by minimising NLL on the fitting sets (default: val and v4heldout; never a test set). They are then
applied to every evaluation set of the run and accuracy, NLL and ECE are reported before and after.

  python scripts/fit_calibration.py results/hpc/TF-ep2-...-s2 --fit val,v4heldout --apply jevbench,td,bbh,musr
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if (Path.cwd() / "data/eval").exists() and not (ROOT / "data/eval").exists():   # HPC: data next to the repo clone
    ROOT = Path.cwd()
FILES = {"val": None, "v4heldout": "data/eval/v4_heldout.jsonl", "jevbench": "data/eval/jevbench_public.jsonl",
         "td": "data/eval/typed_decisions_test.jsonl", "bbh": "data/eval/bbh.jsonl", "musr": "data/eval/musr.jsonl",
         "policy": "data/eval/policy_heldout.jsonl", "folio": "data/eval/folio_val.jsonl", "sharc": "data/eval/sharc_dev.jsonl",
         "claims_ho": "data/eval/claims_ho.jsonl", "tdq": "data/eval/td_noul_qform.jsonl", "tdn": "data/eval/td_noul_neg.jsonl"}


def item_types(name: str) -> dict[str, tuple[str, int | None]]:
    """item_id -> (type, index of the Yes option or None)."""
    path = FILES.get(name)
    if not path or not (ROOT / path).exists():
        return {}
    out = {}
    for line in open(ROOT / path):
        it = json.loads(line)
        names = it.get("label_names") or {}
        yes = next((i for i, L in enumerate(it["labels"]) if str(names.get(L, "")).lower() in ("yes", "true")), None)
        t = it.get("type") or "choice"
        if t == "noul" and yes is None:
            t = "choice"
        out[it["item_id"]] = (t, yes)
    return out


def load(run: Path, name: str) -> list[dict]:
    types = item_types(name)
    rows = []
    for line in open(run / f"preds_{name}.jsonl"):
        p = json.loads(line)
        if p.get("gold_label") not in p["labels"]:
            continue
        t, yes = types.get(p["item_id"], ("choice", None))
        z = [math.log(max(x, 1e-12)) for x in p["probs"]]
        rows.append({"z": z, "gold": p["labels"].index(p["gold_label"]), "type": t, "yes": yes})
    return rows


def probs(r: dict, T: float, b: float) -> list[float]:
    z = [x / T for x in r["z"]]
    if r["yes"] is not None:
        z[r["yes"]] += b
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def metrics(rows: list[dict], params: dict) -> dict:
    if not rows:
        return {}
    nll = acc = 0.0
    bins = [[0, 0.0, 0.0] for _ in range(10)]
    for r in rows:
        T, b = params.get(r["type"], (1.0, 0.0))
        p = probs(r, T, b)
        k = max(range(len(p)), key=p.__getitem__)
        acc += k == r["gold"]
        nll -= math.log(max(p[r["gold"]], 1e-12))
        i = min(9, int(p[k] * 10))
        bins[i][0] += 1; bins[i][1] += p[k]; bins[i][2] += k == r["gold"]
    n = len(rows)
    ece = sum(abs(c - a) for _, c, a in bins) / n
    return {"n": n, "acc": round(acc / n, 4), "nll": round(nll / n, 4), "ece": round(ece, 4)}


def fit(rows: list[dict], with_bias: bool) -> tuple[float, float]:
    best = (1.0, 0.0, float("inf"))
    for T in [0.5 + 0.05 * i for i in range(51)]:
        for b in ([x * 0.25 for x in range(-24, 25)] if with_bias else [0.0]):
            nll = sum(-math.log(max(probs(r, T, b)[r["gold"]], 1e-12)) for r in rows) / len(rows)
            if nll < best[2]:
                best = (T, b, nll)
    return best[0], best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--fit", default="val,v4heldout,claims_ho")
    ap.add_argument("--apply", default="jevbench,td,bbh,musr")
    args = ap.parse_args()
    for r in args.runs:
        run = Path(r)
        fit_rows = [x for n in args.fit.split(",") if (run / f"preds_{n}.jsonl").exists() for x in load(run, n)]
        params = {}
        for t in ("choice", "score", "noul"):
            sub = [x for x in fit_rows if x["type"] == t]
            if len(sub) >= 30:
                params[t] = fit(sub, with_bias=t == "noul")
        print(f"\n{run.name}: fitted on {args.fit} ({len(fit_rows)} items): "
              + ", ".join(f"{t} T={T:.2f}" + (f" b={b:+.2f}" if t == 'noul' else "") for t, (T, b) in params.items()))
        for n in args.apply.split(","):
            if not (run / f"preds_{n}.jsonl").exists():
                continue
            rows = load(run, n)
            a, c = metrics(rows, {}), metrics(rows, params)
            print(f"  {n:9s} n {a['n']:5d} | acc {a['acc']:.3f} -> {c['acc']:.3f} | nll {a['nll']:.3f} -> {c['nll']:.3f} "
                  f"| ece {a['ece']:.3f} -> {c['ece']:.3f}")


if __name__ == "__main__":
    main()
