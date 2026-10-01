"""Gate G2 readout on the rule-decision evaluations (scripts/build_rulegen.py) for one or more runs.

Each run directory needs preds_<name>.jsonl for the rule sets (names below; set with --names) and, for the
self-consistency measures, preds_subq_ho.jsonl (the held-out pair bases' tree nodes, --eval-subq ho=...).

Per run:
  acc            accuracy on in-domain, held-out and held-out-prose cases, by the size of the deciding rule
                 (c0 = default, c1..c3 = conditions of the firing rule)
  edit/sens      one-fact edits that change the gold decision: both base and edit right; independence baseline
                 acc_base x acc_edit; follow = P(edit right | base right)
  invariance     edits that cannot or do not change the decision (masked, unread fact, rule twins that keep it):
                 P(prediction unchanged)
  rule/change    rule twins that change the decision: both right
  self           on the held-out pair bases: P(final = policy applied to the model's own predicate answers),
                 overall and where at least one own predicate answer is wrong (then that program output differs
                 from gold for some items); predicates the tree does not ask keep their true value
  third          interchange pairs where base and source are both right (how many pairs a causal audit could use)

Paired contrasts between arms (--contrast A-B, repeatable) are computed per seed and averaged, with a bootstrap over
held-out cases within seed.

  python scripts/analyze_rule.py results/hpc/rule/*-s3[1-3] --contrast G3-G0 --contrast G3-G4
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
from cotdistill import rulegen as rg  # noqa: E402

NAMES = {"id": "rule_id", "ho": "rule_ho", "prose": "rule_hop", "cf": "rule_hocf", "src": "rule_hosrc"}


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()] if path.exists() else []


def preds(run: Path, name: str) -> dict[str, str]:
    """item_id -> predicted outcome key."""
    out = {}
    for p in read(run / f"preds_{name}.jsonl"):
        out[p["item_id"]] = p["labels"][max(range(len(p["probs"])), key=p["probs"].__getitem__)]
    return out


def load_items(path: Path) -> dict[str, dict]:
    return {r["item_id"]: r for r in read(path)}


def outcome(item: dict, letter: str) -> str:
    return item["label_order"][ord(letter) - 65]


def run_metrics(run: Path, names: dict, items: dict, pairs: list[dict], subq_items: dict) -> dict:
    m: dict = {"run": run.name}
    pr = {k: preds(run, v) for k, v in names.items()}
    # accuracy by deciding-rule size
    for k in ("id", "ho", "prose"):
        if not pr[k]:
            continue
        by = collections.defaultdict(list)
        for i, letter in pr[k].items():
            it = items[k].get(i)
            if it:
                ok = letter == it["gold_label"]
                by["all"].append(ok)
                by[f"c{it['n_conds_fire']}"].append(ok)
        m[f"acc_{k}"] = {g: round(sum(v) / len(v), 4) for g, v in sorted(by.items())}
        m[f"n_{k}"] = len(by["all"])
    # counterfactual pairs
    base_ok = {i: pr["ho"][i] == items["ho"][i]["gold_label"] for i in pr["ho"] if i in items["ho"]}
    base_out = {i: outcome(items["ho"][i], pr["ho"][i]) for i in pr["ho"] if i in items["ho"]}
    groups = collections.defaultdict(list)
    for p in pairs:
        b, o = p["base_id"], p["other_id"]
        if b not in base_ok:
            continue
        if p["type"] == "third":
            if o in pr["src"]:
                s_ok = pr["src"][o] == items["src"][o]["gold_label"]
                groups[f"third/{p['var'][0]}{p['rule_size']}"].append((base_ok[b], s_ok))
            continue
        if o not in pr["cf"]:
            continue
        it = items["cf"][o]
        o_ok = pr["cf"][o] == it["gold_label"]
        same = outcome(it, pr["cf"][o]) == base_out[b]
        if p["type"] == "edit" and p["sub"] == "sensitive":
            groups["edit/sens"].append((base_ok[b], o_ok))
        elif p["type"] == "rule" and p["gold_change"]:
            groups["rule/change"].append((base_ok[b], o_ok))
        else:
            key = "inv/" + (p["sub"] if p["type"] != "rule" else "rule")
            groups[key].append(same)
            groups["inv/all"].append(same)
    for g, v in sorted(groups.items()):
        if g.startswith("inv/"):
            m[g] = {"unchanged": round(sum(v) / len(v), 4), "n": len(v)}
        elif g.startswith("third/"):
            m[g] = {"both_right": sum(a and b for a, b in v), "n": len(v)}
        else:
            ab = sum(a and b for a, b in v) / len(v)
            ind = (sum(a for a, _ in v) / len(v)) * (sum(b for _, b in v) / len(v))
            fol = sum(b for a, b in v if a) / max(1, sum(a for a, _ in v))
            m[g] = {"both": round(ab, 4), "indep": round(ind, 4), "follow": round(fol, 4), "n": len(v)}
    # self-consistency with the model's own predicate answers
    own = collections.defaultdict(dict)
    for p in read(run / "preds_subq_ho.jsonl"):
        if p["kind"] == "cot" and p.get("var", "") and p["var"].startswith("p:"):
            own[p["item_id"]][p["var"]] = p["p_yes"] > 0.5
    agree, agree_wrong, n_wrong = [], [], 0
    for i, ov in own.items():
        if i not in base_out or i not in subq_items:
            continue
        case = rg.case_from_record(subq_items[i])
        prog = case.evaluate(ov)["decision"]
        agree.append(prog == base_out[i])
        if prog != case.gold():
            n_wrong += 1
            agree_wrong.append(prog == base_out[i])
    if agree:
        m["self"] = {"agree": round(sum(agree) / len(agree), 4), "n": len(agree),
                     "agree_when_own_program_differs_from_gold": round(sum(agree_wrong) / max(1, len(agree_wrong)), 4),
                     "n_differs": n_wrong}
    return m


def contrast(runs: dict, a: str, b: str, items: dict, names: dict, n_boot: int = 2000) -> dict | None:
    """Held-out accuracy and edit-pair 'both right' differences A-B, paired by seed; bootstrap over base cases."""
    seeds = sorted(set(runs.get(a, {})) & set(runs.get(b, {})))
    if not seeds:
        return None
    diffs = []
    for s in seeds:
        pa, pb = preds(runs[a][s], names["ho"]), preds(runs[b][s], names["ho"])
        ids = [i for i in pa if i in pb and i in items["ho"]]
        da = [int(pa[i] == items["ho"][i]["gold_label"]) - int(pb[i] == items["ho"][i]["gold_label"]) for i in ids]
        diffs.append(da)
    rng = random.Random(0)
    means = [sum(d) / len(d) for d in diffs]
    boot = []
    for _ in range(n_boot):
        boot.append(sum(sum(rng.choice(d) for _ in d) / len(d) for d in diffs) / len(diffs))
    boot.sort()
    return {"seeds": seeds, "per_seed": [round(x, 4) for x in means], "mean": round(sum(means) / len(means), 4),
            "ci95_within_seed": [round(boot[int(0.025 * n_boot)], 4), round(boot[int(0.975 * n_boot)], 4)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--names", default=",".join(f"{k}={v}" for k, v in NAMES.items()))
    ap.add_argument("--contrast", action="append", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    names = dict(x.split("=", 1) for x in args.names.split(","))
    items = {"id": load_items(ROOT / "data/eval/rule_id.jsonl"), "ho": load_items(ROOT / "data/eval/rule_ho.jsonl"),
             "prose": load_items(ROOT / "data/eval/rule_ho_prose.jsonl"),
             "cf": load_items(ROOT / "data/eval/rule_ho_cf.jsonl"), "src": load_items(ROOT / "data/eval/rule_ho_src.jsonl")}
    pairs = read(ROOT / "data/rule/pairs_ho.jsonl")
    subq_items = load_items(ROOT / "data/rule/ho_subq.jsonl")
    rows, by_arm = [], collections.defaultdict(dict)
    for r in args.runs:
        run = Path(r)
        m = run_metrics(run, names, items, pairs, subq_items)
        rows.append(m)
        mm = re.match(r"^([A-Za-z0-9]+).*-s(\d+)$", run.name)
        if mm:                                   # predicates-only data makes G3 into G3P
            by_arm[mm.group(1) + ("P" if "student_rule_p" in run.name else "")][int(mm.group(2))] = run
        es, inv, sf = m.get("edit/sens", {}), m.get("inv/all", {}), m.get("self", {})
        print(f"{run.name[:40]:40s} ho {m.get('acc_ho', {}).get('all', float('nan')):.3f} "
              f"(c2 {m.get('acc_ho', {}).get('c2', float('nan')):.3f} c3 {m.get('acc_ho', {}).get('c3', float('nan')):.3f}) "
              f"prose {m.get('acc_prose', {}).get('all', float('nan')):.3f} id {m.get('acc_id', {}).get('all', float('nan')):.3f} | "
              f"edit both {es.get('both', float('nan')):.3f} vs indep {es.get('indep', float('nan')):.3f} "
              f"follow {es.get('follow', float('nan')):.3f} | invariant {inv.get('unchanged', float('nan')):.3f} | "
              f"self {sf.get('agree', float('nan')):.3f} (own-wrong {sf.get('agree_when_own_program_differs_from_gold', float('nan')):.3f}, "
              f"n {sf.get('n_differs', 0)})")
    out = {"runs": rows, "contrasts": {}}
    for c in args.contrast:
        a, b = c.split("-")
        out["contrasts"][c] = contrast(by_arm, a, b, items, names)
        print(c, json.dumps(out["contrasts"][c]))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
