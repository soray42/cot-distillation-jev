"""v5 training data: the release-clean v4tf data with the format mix repaired (notes/forgetting_report.md).

The v4t mix lost general multiple-choice reasoning: exam-style multiple choice was 3.6% of rows, while 2-option
items (short classification, 11,866 yes/no claim twins and the yes/no tree nodes) took most of the loss weight.
v5 changes three things, all on the training split (val is copied unchanged):
1. Rule-claim twins: the yes/no claim/negation pairs about rule-decision cases are kept for a third of the cases
   (whole pairs, so a constant "Yes" still cannot fit). The other cases' claims come back as one 4-option item
   "Which of these statements about the case is true?": one true statement and three false ones about the same case
   (predicates, rule conditions, the decision), all program-exact.
2. tasksource claim twins: whole pairs kept for a third of the source items, the rest dropped.
3. Exam-style multiple-choice replay (data/v5/replay_mc.jsonl, scripts/build_v5_replay.py) is added.
Tree nodes are unchanged here; their weight and format are training flags (--aux-total, --node-format).

  python scripts/build_v5.py --base data/student_v4tf_rel --replay data/v5/replay_mc.jsonl --out data/student_v5
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from build_v4t import rule_state  # noqa: E402
from cotdistill import rulegen as rg  # noqa: E402


def keep_third(key: str) -> bool:
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % 3 == 0


def rule_statements(case: rg.Case) -> tuple[list[str], list[str]]:
    """(true statements, false statements) about one case: every predicate both ways, every rule's conditions both
    ways, and the decision against each other outcome."""
    dom, res = case.domain, case.evaluate()
    true, false = [], []
    for pred in rg._all_preds(case):
        v = res["p"][pred.pid]
        (true if v else false).append(rg._cap(rg.clause(dom, pred, True)) + ".")
        (false if v else true).append(rg._cap(rg.clause(dom, pred, False)) + ".")
    for k, a in enumerate(res["a"], 1):
        (true if a else false).append(f"All conditions of rule {k} hold in this case.")
        (false if a else true).append(f"Not all conditions of rule {k} hold in this case.")
    true.append(f"Under this policy the agent should {dom.imperative(res['decision'])}.")
    false += [f"Under this policy the agent should {dom.imperative(o[0])}." for o in dom.outcomes
              if o[0] != res["decision"]]
    return list(dict.fromkeys(true)), list(dict.fromkeys(x for x in false if x not in true))


def rule_claim_mc(cases: list[rg.Case]) -> list[dict]:
    out = []
    for case in cases:
        true, false = rule_statements(case)
        rng = random.Random(f"v5mc-{case.item_id}")
        if not true or len(false) < 3:
            continue
        opts = [rng.choice(true)] + rng.sample(false, 3)
        rng.shuffle(opts)
        letters = ["A", "B", "C", "D"]
        gold = letters[[i for i, o in enumerate(opts) if o in true][0]]
        out.append({"item_id": f"{case.item_id}-mc", "source": f"claims_mc/rule/{case.domain.name}",
                    "prompt": rule_state(case) + "\nQuestion: Which of these statements about the case is true?\n"
                              "Options:\n" + "\n".join(f"{L}) {o}" for L, o in zip(letters, opts)),
                    "labels": letters, "gold_label": gold, "teacher": {L: float(L == gold) for L in letters},
                    "subqs": [], "random_subqs": [], "augment": "claim_mc"})
    return out


def mass(rows: list[dict], aux_total: float) -> dict:
    """Loss-weight shares by format: final questions by option count, plus the tree nodes (aux_total per problem
    with nodes), as in student.build_full_tree."""
    m = collections.Counter()
    for r in rows:
        w = r.get("weight", 1.0)
        m["final_2opt" if len(r["labels"]) == 2 else "final_3plus"] += w
        if any("id" in s for s in r.get("subqs") or []):
            m["tree_nodes"] += aux_total * w
    tot = sum(m.values())
    return {k: round(v / tot, 3) for k, v in sorted(m.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/student_v4tf_rel")
    ap.add_argument("--replay", default="data/v5/replay_mc.jsonl")
    ap.add_argument("--out", default="data/student_v5")
    ap.add_argument("--aux-total", type=float, default=0.5, help="node weight per problem planned for training "
                                                                 "(only for the summary's loss-weight shares)")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(ROOT / args.base / "train.jsonl")]
    summary: dict = {"base": args.base, "replay": args.replay, "before": {"items": len(rows),
                                                                           "mass": mass(rows, args.aux_total)}}
    kept, dropped = [], collections.Counter()
    for r in rows:
        if r.get("augment") == "claim_twin":
            pair = re.sub(r"-(p|pn|a|an|d|dw|claim|negclaim)$", "", r["item_id"])
            if not keep_third(pair):
                dropped[r["source"].split("/")[0]] += 1
                continue
        kept.append(r)
    # the cases whose yes/no twins were dropped come back as 4-option statements; the same cases, regenerated
    # exactly as build_v4t.rule_twins made them (1200 cases, seed 21)
    cases = [c for c in rg.generate(1200, rg.TRAIN_DOMAINS, seed=21, prefix="clm") if not keep_third(c.item_id)]
    mc = rule_claim_mc(cases)
    replay = [json.loads(l) for l in open(ROOT / args.replay)] if (ROOT / args.replay).exists() else []
    train = kept + mc + replay
    random.Random("shuffle-v5").shuffle(train)
    with open(out / "train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    (out / "val.jsonl").write_bytes((ROOT / args.base / "val.jsonl").read_bytes())
    summary.update(twins_dropped=dict(dropped), rule_claim_mc=len(mc), replay=len(replay),
                   after={"items": len(train), "mass": mass(train, args.aux_total)},
                   md5={s: hashlib.md5((out / f"{s}.jsonl").read_bytes()).hexdigest() for s in ("train", "val")})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
