"""Student data with program proof trees for Knights & Knaves: data/student_v3f with every K&K item's tree nodes
(subqs) replaced by the nodes of a deterministic solver's proof (teacher_cache/solver_kk_v3f/reasoning_trees, built by
its build_reasoning_trees.py on the v3f prompts themselves, augmented "None of the above" items included: every non-parse node is checked to follow from its listed parents alone over all role
assignments). Finals, control questions and every other family are unchanged (DeepSeek), so arms on this data differ
from their v3f twins only in the K&K nodes.

Cleaning: the three generic rule nodes (one person one role, knights tell the truth, knaves lie) are dropped; "verify"
nodes (checks of the finished assignment, after every role is derived) are dropped; repeated questions are merged into
their first occurrence (dependencies remapped). About half of the nodes are asked in their opposite wording (answer
flipped), as tree_item does for teacher trees. Targets are the program's answers (truth, p_cot 1/0); node_depth is
recomputed on the cleaned graph; tree_fix is set so the full-tree planner uses the post-audit matching.

  python scripts/build_solver_kk.py --out data/student_v3s
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.sources import kk_subq_truth  # noqa: E402
from cotdistill.teacher import leaks_final  # noqa: E402
from cotdistill.treeplan import node_depths  # noqa: E402

GENERIC = ("Is each of the ", "Must a knight's statement", "Must a knave's statement")


def clean(nodes: list[dict], rec: dict, rng: random.Random) -> tuple[list[dict], collections.Counter]:
    c = collections.Counter()
    alias, out, first = {}, [], {}
    for n in nodes:
        if n["question"].startswith(GENERIC):
            c["generic"] += 1
            continue
        if n["type"] == "verify":
            c["verify"] += 1
            continue
        deps = [alias.get(d, d) for d in n.get("depends_on") or []]
        deps = [d for d in dict.fromkeys(deps) if d in first.values()]
        if n["question"] in first:                     # same judgement asked again: merge into the first one
            c["duplicate"] += 1
            alias[n["id"]] = first[n["question"]]
            continue
        if leaks_final(n["question"], rec):
            c["leak"] += 1
            continue
        first[n["question"]] = n["id"]
        out.append({"id": n["id"], "type": n["type"], "depends_on": deps, "question": n["question"],
                    "answer": n["answer"], "opposite": n.get("opposite")})
    kept = {n["id"] for n in out}
    for n in out:                                      # edges to dropped nodes (generic rules) go
        n["depends_on"] = [d for d in n["depends_on"] if d in kept]
    depth = node_depths(out)
    subqs = []
    for n in out:
        yes = n["answer"] == "yes"
        q, opp = n["question"], n["opposite"]
        if opp and rng.random() < 0.5:
            q, opp, yes = opp, q, not yes
            c["flipped"] += 1
        t = kk_subq_truth(q, rec)
        if t is not None and t != yes:                 # the solver's answers are exact; any disagreement is a bug
            raise AssertionError(f"{rec['item_id']}: {q} solver {yes} program {t}")
        subqs.append({"question": q, "opposite": opp, "p_cot": 1.0 if yes else 0.0, "p_fresh": None, "truth": yes,
                      "type": n["type"], "depends_on": n["depends_on"], "id": n["id"], "initial": None,
                      "status": "direct", "node_depth": depth[n["id"]], "source_tree": "solver"})
    c["kept"] += len(subqs)
    return subqs, c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/student_v3f")
    ap.add_argument("--trees", default="teacher_cache/solver_kk_v3f/reasoning_trees")
    ap.add_argument("--out", default="data/student_v3s")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    items = {json.loads(l)["item_id"]: json.loads(l) for l in open(ROOT / "teacher_cache/label_kk_v1/items.jsonl")}
    sol = {json.loads(l)["item_id"]: json.loads(l) for l in open(ROOT / args.trees / "solutions.jsonl")}
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    tot, sizes, depths, par = collections.Counter(), [], [], []
    summary = {}
    for split in ("train", "val"):
        n_kk = 0
        with open(out / f"{split}.jsonl", "w") as f:
            for line in open(ROOT / args.base / f"{split}.jsonl"):
                it = json.loads(line)
                if it.get("source") == "knights_knaves":
                    roles = sol[it["item_id"]]["roles"]         # this prompt's own solution (augmented items differ)
                    rec = dict(items[it["item_id"]], prompt=it["prompt"],
                               predicates=[{"question": f"Is {k} a knight?", "truth": v == "knight"} for k, v in roles.items()])
                    assert sol[it["item_id"]]["answer"] == it["gold_label"], it["item_id"]
                    nodes = json.load(open(ROOT / args.trees / f"trees/{it['item_id']}.json"))["nodes"]
                    subqs, c = clean(nodes, rec, rng)
                    tot.update(c)
                    it = dict(it, subqs=subqs, tree_fix=True, has_tree=True, tree_source="solver_kk_v3f")
                    sizes.append(len(subqs))
                    depths.append(max(s["node_depth"] for s in subqs))
                    par.append(sum(bool(s["depends_on"]) for s in subqs) / len(subqs))
                    n_kk += 1
                f.write(json.dumps(it) + "\n")
        summary[split] = {"kk_items": n_kk, "md5": hashlib.md5((out / f"{split}.jsonl").read_bytes()).hexdigest()}
    m = lambda x: round(statistics.mean(x), 2)
    summary.update({"nodes_per_kk_item": [m(sizes), min(sizes), max(sizes)], "max_depth": [m(depths), min(depths), max(depths)],
                    "share_with_parents": m(par), "cleaning": dict(tot), "base": args.base, "trees": args.trees})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
