"""TF-v4 training data: student_v4 plus the v4 teacher trees and claim/negation twins (Yes-bias fix).

1. data/student_v4/{train,val}.jsonl are copied; items whose teacher tree exists in teacher_cache/label_v4_c*_tree get its
   nodes (`subqs`) and controls (`random_subqs`) through build_student_data.convert. Final targets stay as in
   student_v4, so the run differs from A2-v4 in the supervision added, not in the final labels.
2. Claim twins, new items, training split only: the declarative-claim yes/no format ("Question: <claim>." with
   "No"/"Yes" options, the format of the TD yes/no questions) is absent from training, and every trained model answers
   such claims Yes far too often while still ranking them correctly (notes, 2026-10-02). Every claim comes with its
   negation, so answering Yes by default cannot fit:
   - tasksource yes/no items in fixed templates ('Is "X" the correct label/answer ...?', 'Does at least one item have
     the label "X"?', 'Do all items have the same label?') rewritten as the claim and its negation;
   - rule-decision cases (src/cotdistill/rulegen.py, training domains only) rendered as a JSON state with a policy and
     a case, with program-exact claims about one predicate, one rule and the decision, each with its negation.
   Option order (No/Yes or Yes/No) is random per item.

  python scripts/build_v4t.py      # -> data/student_v4t/{train,val}.jsonl, data/student_v4t/summary.json,
                                   #    data/eval/claims_ho.jsonl (held-out-domain claim twins, evaluation only)
"""
from __future__ import annotations

import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from build_student_data import convert  # noqa: E402
from cotdistill import rulegen as rg  # noqa: E402

TS_TEMPLATES = [   # (pattern on the question line, claim, negation); \1 is the quoted label
    (re.compile(r'^(.*?)Is "([^"]+)" the correct label for this example\?$'),
     r'\1"\2" is the correct label for this example.', r'\1"\2" is not the correct label for this example.'),
    (re.compile(r'^(.*?)Is "([^"]+)" the correct answer to the question\?$'),
     r'\1"\2" is the correct answer to the question.', r'\1"\2" is not the correct answer to the question.'),
    (re.compile(r'^(.*?)Is "([^"]+)" the correct answer\?$'),
     r'\1"\2" is the correct answer.', r'\1"\2" is not the correct answer.'),
    (re.compile(r'^(.*?)Does at least one item have the label "([^"]+)"\?(.*)$'),
     r'\1At least one item has the label "\2".\3', r'\1No item has the label "\2".\3'),
    (re.compile(r'^(.*?)Do all items have the same label\?(.*)$'),
     r'\1All items have the same label.\2', r'\1Not all items have the same label.\2'),
]


def yes_no_item(item_id: str, source: str, body: str, claim: str, truth: bool, rng: random.Random) -> dict:
    opts = ["Yes", "No"] if rng.random() < 0.5 else ["No", "Yes"]
    gold = "A" if (opts[0] == "Yes") == truth else "B"
    return {"item_id": item_id, "source": source, "prompt": f"{body}\nQuestion: {claim}\nOptions:\nA) {opts[0]}\nB) {opts[1]}",
            "labels": ["A", "B"], "gold_label": gold, "teacher": {"A": float(gold == "A"), "B": float(gold == "B")},
            "subqs": [], "random_subqs": [], "augment": "claim_twin"}


def ts_twins(rows: list[dict], rng: random.Random) -> list[dict]:
    out = []
    for r in rows:
        if not r["source"].startswith("tasksource"):
            continue
        head, _, tail = r["prompt"].rpartition("\nOptions:\n")
        names = [o[3:].strip() for o in tail.splitlines() if len(o) > 3 and o[1] == ")"]
        if sorted(n.lower() for n in names) != ["no", "yes"]:
            continue
        body, _, qline = head.rpartition("\n")
        q = qline[len("Question: "):] if qline.startswith("Question: ") else qline
        truth = names[ord(r["gold_label"]) - 65].lower() == "yes"
        for pat, pos, neg in TS_TEMPLATES:
            if pat.match(q):
                out.append(yes_no_item(r["item_id"] + "-claim", r["source"], body, pat.sub(pos, q), truth, rng))
                out.append(yes_no_item(r["item_id"] + "-negclaim", r["source"], body, pat.sub(neg, q), not truth, rng))
                break
    return out


def rule_state(case: rg.Case) -> str:
    dom = case.domain
    rules = [f"If {rg.conds_text(dom, r)}, {dom.imperative(r.outcome)}." for r in case.rules]
    state = {"policy": {"title": dom.title, "rules_in_order": rules, "otherwise": dom.imperative(dom.default) + "."},
             "case": dict(rg.case_lines(dom, case.facts))}
    return "State:\n" + json.dumps(state, indent=1) + "\n"


def rule_twins(n: int, seed: int, rng: random.Random, domains: tuple = rg.TRAIN_DOMAINS, prefix: str = "clm") -> list[dict]:
    out = []
    for case in rg.generate(n, domains, seed=seed, prefix=prefix):
        dom, res, body = case.domain, case.evaluate(), rule_state(case)
        src = f"claims/rule/{dom.name}"
        pred = rng.choice(rg._all_preds(case))
        v = res["p"][pred.pid]
        out.append(yes_no_item(f"{case.item_id}-p", src, body, rg._cap(rg.clause(dom, pred, True)) + ".", v, rng))
        out.append(yes_no_item(f"{case.item_id}-pn", src, body, rg._cap(rg.clause(dom, pred, False)) + ".", not v, rng))
        k = rng.randrange(len(case.rules)) + 1
        a = res["a"][k - 1]
        out.append(yes_no_item(f"{case.item_id}-a", src, body, f"All conditions of rule {k} hold in this case.", a, rng))
        out.append(yes_no_item(f"{case.item_id}-an", src, body, f"Not all conditions of rule {k} hold in this case.",
                               not a, rng))
        wrong = rng.choice([o[0] for o in dom.outcomes if o[0] != res["decision"]])
        out.append(yes_no_item(f"{case.item_id}-d", src, body,
                               f"Under this policy the agent should {dom.imperative(res['decision'])}.", True, rng))
        out.append(yes_no_item(f"{case.item_id}-dw", src, body,
                               f"Under this policy the agent should {dom.imperative(wrong)}.", False, rng))
    return out


def main() -> None:
    rng = random.Random(0)
    trees = {}
    for d in sorted((ROOT / "teacher_cache").glob("label_v4_c*_tree")):
        for p in sorted(d.glob("*.json")):
            res = json.loads(p.read_text())
            if "item" in res:
                rec = convert(res)
                trees[rec["item_id"].removeprefix("v4-")] = rec
    summary: dict = {"trees_loaded": len(trees)}
    out_dir = ROOT / "data/student_v4t"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        rows = [json.loads(l) for l in open(ROOT / f"data/student_v4/{split}.jsonl")]
        hit = 0
        for r in rows:
            t = trees.get(r["item_id"])
            if t and t["subqs"]:
                r["subqs"], r["random_subqs"], r["has_tree"] = t["subqs"], t["random_subqs"], True
                hit += 1
        extra = []
        if split == "train":
            extra = ts_twins(rows, rng) + rule_twins(1200, seed=21, rng=rng)
        rows += extra
        random.Random(f"shuffle-{split}").shuffle(rows)
        with open(out_dir / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        gold_yes = collections.Counter()
        for r in extra:
            names = [o[3:] for o in r["prompt"].rpartition("\nOptions:\n")[2].splitlines()]
            gold_yes[(r["source"].split("/")[0], names[ord(r["gold_label"]) - 65])] += 1
        summary[split] = {"items": len(rows), "v4_items_given_trees": hit,
                          "items_with_nodes": sum(bool(r.get("subqs")) for r in rows),
                          "nodes": sum(len(r.get("subqs") or []) for r in rows), "claim_twins": len(extra),
                          "claim_gold": {f"{a}/{b}": v for (a, b), v in sorted(gold_yes.items())}}
    # held-out claim set (rule-decision held-out domains, never trained): fits the yes/no bias in the claim format and
    # measures negation consistency (each claim's negation follows it in the file, ids <case>-x / <case>-xn)
    ho = rule_twins(400, seed=41, rng=random.Random(41), domains=rg.HELDOUT_DOMAINS, prefix="clmho")
    for r in ho:
        r["type"], r["label_names"] = "noul", {L: ("true" if n == "Yes" else "false") for L, n in zip(
            r["labels"], [o[3:] for o in r["prompt"].rpartition("\nOptions:\n")[2].splitlines()])}
    with open(ROOT / "data/eval/claims_ho.jsonl", "w") as f:
        for r in ho:
            f.write(json.dumps(r) + "\n")
    summary["claims_ho"] = len(ho)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
