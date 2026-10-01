"""Rule-decision data (src/cotdistill/rulegen.py): training items with program trees, in-domain and held-out
evaluation sets, and program-exact counterfactual pairs.

  data/student_rule/{train,val}.jsonl    training domains, renderers list/bullets; subqs = program tree nodes
                                         (truth-valued), random_subqs = matched questions about unread facts
  data/eval/rule_id.jsonl                training domains, new policies and cases (list renderer)
  data/eval/rule_ho.jsonl                held-out domains (list renderer)
  data/eval/rule_ho_prose.jsonl          the same held-out cases in the held-out prose renderer
  data/eval/rule_{id,ho}_cf.jsonl        counterfactual items for the first --cf-bases base cases: one-fact edits
                                         (sensitive/masked), unread-fact edits, rule twins
  data/eval/rule_{id,ho}_src.jsonl       third-outcome sources for every base case (same policy; interchange
                                         targets in the pair file)
  data/rule/pairs_{id,ho}.jsonl          pair metadata linking base and counterfactual/source items
  data/rule/{id,ho}_subq.jsonl           the pair bases with their tree nodes and controls (for --eval-subq)
  data/rule/summary.json                 counts by domain, firing-rule size, pair type and variable

  python scripts/build_rulegen.py
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cotdistill import rulegen as rg  # noqa: E402


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{path.relative_to(ROOT)}: {len(rows)}")


def rule_size(case: rg.Case, var: str | None) -> int | None:
    """Conditions of the rule a variable belongs to (a:k), or the largest rule reading a predicate (p:pid)."""
    if not var:
        return None
    kind, name = var.split(":", 1)
    if kind == "a":
        return len(case.rules[int(name) - 1].conds)
    return max(len(r.conds) for r in case.rules if any(p.pid == name for p, _ in r.conds))


def pairs_for(cases: list[rg.Case], n_bases: int, seed: int, tag: str):
    cf_items, src_items, pairs = [], [], []
    for case in cases[:n_bases]:
        rng = random.Random(f"{seed}-pairs-{case.item_id}")
        for meta, other in (rg.edit_pairs(case, rng) + rg.distractor_pairs(case, rng) + rg.rule_twins(case, rng)):
            cf_items.append(rg.eval_record(other))
            pairs.append({"pair_id": other.item_id, "split": tag, **meta, "base_id": case.item_id,
                          "other_id": other.item_id, "base_gold": case.gold(), "other_gold": other.gold(),
                          "rule_size": rule_size(case, meta.get("var")), "domain": case.domain.name})
    for case in cases:                               # interchange pairs for every base case
        rng = random.Random(f"{seed}-third-{case.item_id}")
        for meta, src in rg.third_outcome_pairs(case, rng):
            src_items.append(rg.eval_record(src))
            pairs.append({"pair_id": src.item_id, "split": tag, **meta, "base_id": case.item_id,
                          "other_id": src.item_id, "base_gold": meta["base_out"], "other_gold": meta["source_out"],
                          "rule_size": rule_size(case, meta["var"]), "domain": case.domain.name})
    return cf_items, src_items, pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--n-id", type=int, default=400)
    ap.add_argument("--n-ho", type=int, default=1000)
    ap.add_argument("--cf-bases", type=int, default=300, help="base cases per split that get counterfactual pairs")
    args = ap.parse_args()

    train = rg.generate(args.n_train, rg.TRAIN_DOMAINS, seed=11, renderers=rg.TRAIN_RENDERERS)
    val = rg.generate(args.n_val, rg.TRAIN_DOMAINS, seed=12, renderers=rg.TRAIN_RENDERERS)
    test_id = rg.generate(args.n_id, rg.TRAIN_DOMAINS, seed=13)
    test_ho = rg.generate(args.n_ho, rg.HELDOUT_DOMAINS, seed=14)
    for cs in (train, val, test_id, test_ho):
        random.Random(len(cs)).shuffle(cs)       # interleave domains, so any prefix (e.g. --cf-bases) covers all

    rng = random.Random(0)
    write(ROOT / "data/student_rule/train.jsonl", [rg.train_record(c, rng) for c in train])
    write(ROOT / "data/student_rule/val.jsonl", [rg.train_record(c, rng) for c in val])
    write(ROOT / "data/eval/rule_id.jsonl", [rg.eval_record(c) for c in test_id])
    write(ROOT / "data/eval/rule_ho.jsonl", [rg.eval_record(c) for c in test_ho])
    prose = []
    for c in test_ho:
        r = rg.eval_record(rg.Case(c.item_id + "-prose", c.domain, c.rules, c.facts, c.label_order, "prose"))
        prose.append(r)
    write(ROOT / "data/eval/rule_ho_prose.jsonl", prose)

    summary: dict = {}
    for tag, cases, seed in (("id", test_id, 13), ("ho", test_ho, 14)):
        cf, src, pairs = pairs_for(cases, args.cf_bases, seed, tag)
        write(ROOT / f"data/eval/rule_{tag}_cf.jsonl", cf)
        write(ROOT / f"data/eval/rule_{tag}_src.jsonl", src)
        write(ROOT / f"data/rule/pairs_{tag}.jsonl", pairs)
        sub_rng = random.Random(f"{seed}-subq")              # the pair bases' tree nodes and controls (--eval-subq)
        write(ROOT / f"data/rule/{tag}_subq.jsonl", [rg.train_record(c, sub_rng) for c in cases[:args.cf_bases]])
        c = collections.Counter()
        for p in pairs:
            sub = p["sub"] if p["type"] != "rule" else f"{p['sub']}/{'change' if p['gold_change'] else 'same'}"
            c[f"{p['type']}/{sub}"] += 1
            if p["type"] == "third":
                c[f"third/{p['var'][0]}/size{p['rule_size']}"] += 1
        summary[f"pairs_{tag}"] = dict(sorted(c.items()))

    train_prompts = {r["prompt"] for r in (rg.eval_record(c) for c in train)}
    for name in ("rule_id", "rule_ho", "rule_ho_prose", "rule_id_cf", "rule_ho_cf", "rule_id_src", "rule_ho_src"):
        rows = [json.loads(l) for l in open(ROOT / f"data/eval/{name}.jsonl")]
        overlap = sum(r["prompt"] in train_prompts for r in rows)
        summary.setdefault("train_prompt_overlap", {})[name] = overlap
    for tag, cases in (("train", train), ("val", val), ("id", test_id), ("ho", test_ho)):
        res = [c.evaluate() for c in cases]
        summary[tag] = {
            "n": len(cases),
            "default_share": round(sum(r["fire"] is None for r in res) / len(res), 3),
            "firing_rule_size": dict(collections.Counter(len(c.rules[r["fire"] - 1].conds) if r["fire"] else 0
                                                         for c, r in zip(cases, res))),
            "gold": dict(collections.Counter(r["decision"] for r in res).most_common()),
            "mean_nodes": round(sum(len(rg.tree_nodes(c)) for c in cases) / len(cases), 2),
        }
    (ROOT / "data/rule").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/rule/summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
