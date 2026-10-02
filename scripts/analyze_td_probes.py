"""Readout of the TD yes/no format probes (scripts/build_td_probes.py) for one or more runs.

Per form (orig, plain, qform, neg, swap) and question type: accuracy, the share of Yes answers, and the AUC of P(Yes)
against the truth of the asked statement. Two derived checks:
  negation consistency  per item, P(Yes | claim) + P(Yes | negated claim) should be 1; reported as the mean of that
                        sum (1.0 = consistent, near 2 = Yes to both, i.e. acquiescence) and the share of items where
                        the two answers disagree as they should
  contextual calibration  P(Yes) on the content-free input (state "{}") per question type and form; subtracting its
                        logit from every item's Yes logit (Zhao et al. 2021) needs no labels; accuracy after it

  python scripts/analyze_td_probes.py results/hpc/debug/debug-tfep2 results/hpc/debug/debug-a2v4
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMS = {"orig": "tdo", "plain": "tdp", "qform": "tdq", "neg": "tdn", "swap": "tds"}


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def load_items(form: str) -> dict:
    return {json.loads(l)["item_id"]: json.loads(l) for l in open(ROOT / f"data/eval/td_noul_{form}.jsonl")}


def p_yes(pred: dict, item: dict) -> float:
    yes = next(L for L, n in item["label_names"].items() if n == "true")
    return pred["probs"][pred["labels"].index(yes)]


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    items = {f: load_items(f) for f in list(FORMS) + ["cf"]}
    for r in args.runs:
        run = Path(r)
        print(f"\n=== {run.name}")
        py = {}                                   # (form, base item id) -> (P(Yes), statement true?)
        for form, name in FORMS.items():
            path = run / f"preds_{name}.jsonl"
            if not path.exists():
                continue
            for line in open(path):
                p = json.loads(line)
                it = items[form][p["item_id"]]
                truth = it["label_names"][it["gold_label"]] == "true"
                py[(form, p["item_id"].rsplit("/", 1)[0])] = (p_yes(p, it), truth)
        cf = {}
        if (run / "preds_tdcf.jsonl").exists():
            for line in open(run / "preds_tdcf.jsonl"):
                p = json.loads(line)
                qt, form = p["item_id"].split("/")[1], p["item_id"].split("/")[2]
                cf[(qt, form)] = p_yes(p, items["cf"][p["item_id"]])
        qtypes = sorted({k[1].split("/")[-1] for k in py})
        print(f"{'form':6s} {'acc':>6s} {'P(Yes)':>7s} {'AUC':>5s} | per type acc/P(Yes): " + " ".join(q[:10] for q in qtypes))
        for form in FORMS:
            rows = [(k[1], v) for k, v in py.items() if k[0] == form]
            if not rows:
                continue
            acc = sum((v[0] > 0.5) == v[1] for _, v in rows) / len(rows)
            yes = sum(v[0] > 0.5 for _, v in rows) / len(rows)
            a = auc([v[0] for _, v in rows if v[1]], [v[0] for _, v in rows if not v[1]])
            per = []
            for q in qtypes:
                sub = [v for k, v in rows if k.endswith("/" + q)]
                per.append(f"{sum((v[0] > 0.5) == v[1] for v in sub) / len(sub):.2f}/{sum(v[0] > 0.5 for v in sub) / len(sub):.2f}")
            print(f"{form:6s} {acc:6.3f} {yes:7.3f} {a:5.2f} | " + " ".join(per))
        both = [(py[("orig", k)][0], py[("neg", k)][0]) for (f, k) in py if f == "plain" and ("neg", k) in py
                and ("orig", k) in py]
        if both:
            s = sum(a + b for a, b in both) / len(both)
            ok = sum((a > 0.5) != (b > 0.5) for a, b in both) / len(both)
            print(f"negation: mean P(Yes|claim)+P(Yes|negation) = {s:.3f} (1 = consistent); answers flip in {ok:.3f} of items")
        if cf:
            for form in ("orig", "qform", "neg"):
                rows = [(k[1], v) for k, v in py.items() if k[0] == form]
                adj = [((logit(v[0]) - logit(cf[(k.split("/")[-1], form)])) > 0) == v[1] for k, v in rows
                       if (k.split("/")[-1], form) in cf]
                if adj:
                    base = sum((v[0] > 0.5) == v[1] for _, v in rows) / len(rows)
                    print(f"contextual calibration ({form}): acc {base:.3f} -> {sum(adj) / len(adj):.3f}; content-free P(Yes) "
                          + " ".join(f"{q[:8]}={cf.get((q, form), float('nan')):.2f}" for q in qtypes))


if __name__ == "__main__":
    main()
