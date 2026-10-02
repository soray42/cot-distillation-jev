"""Tree-graph and TF/TFM covariate balance report for student data (tree audit, notes/tree_audit_summary.md).

Per family, from the data and from the full-tree plan the TF and TFM arms train on (treeplan.full_tree_plan, a fixed
seed per item, as many plans as --plans per item):
  graph     nodes; edges; dropped-parent edges (data built after the audit: a parent that was filtered out, replaced by
            its nearest kept ancestors; older data: a depends_on entry naming a node that is not in the data, i.e. a
            dangling edge); nodes with at least one such parent; nodes whose only parents were lost (asked plainly)
  views     asked nodes; eligible views (facts stated in both arms); views where TF and TFM state different facts
  balance   over the facts of eligible views, TF vs TFM: mean node depth, Yes share, answer-component share; exact-depth
            match rate; share of TFM facts that are ancestors of the asked node (0 after the fix by construction)

  python scripts/tree_balance.py data/student_v3 data/student_v3f --out runs/report_tree_balance.json
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
from cotdistill import treeplan as S  # noqa: E402


def family(src: str | None) -> str:
    s = (src or "").removeprefix("v4/").split("/")[0]
    return {"returns": "policy", "expense": "policy", "subscription": "policy"}.get(s, "arc" if s.startswith("arc") else s)


def ancestors(i: str, by: dict) -> set[str]:
    out, todo = set(), list(by[i].get("depends_on") or [])
    while todo:
        j = todo.pop()
        if j in by and j not in out:
            out.add(j)
            todo += by[j].get("depends_on") or []
    return out


def fact_yes(sq: dict) -> bool:
    return S._fact_line(sq).endswith("A: Yes")


def report(path: Path, plans: int) -> dict:
    T = collections.defaultdict(collections.Counter)
    for split in ("train", "val"):
        for line in open(path / f"{split}.jsonl"):
            r = json.loads(line)
            nodes = [sq for sq in r.get("subqs") or [] if "id" in sq]
            if not nodes:
                continue
            c = T[family(r.get("source"))]
            by = {sq["id"]: sq for sq in nodes}
            fix = bool(r.get("tree_fix"))
            depth = {i: by[i]["node_depth"] for i in by} if fix else S.node_depths(nodes)
            c["items"] += 1
            c["nodes"] += len(nodes)
            for sq in nodes:
                orig = sq.get("parents_orig") if fix else sq.get("depends_on") or []
                lost = [d for d in orig if d not in by]
                c["edges"] += len(orig)
                c["edges_parent_lost"] += len(lost)
                c["nodes_parent_lost"] += bool(lost)
                c["nodes_all_parents_lost"] += bool(orig) and not [d for d in (sq.get("depends_on") or []) if d in by]
            for k in range(plans):
                plan = S.full_tree_plan(r, cap=10, rng=random.Random(f"balance-{k}-{r['item_id']}"))
                for e in plan:
                    c["views"] += 1
                    if not e["parents"]:
                        continue
                    c["eligible"] += 1
                    c["differ"] += set(e["parents"]) != set(e["matched"])
                    anc = ancestors(e["node"]["id"], by)
                    for x, j in zip(e["parents"], e["matched"]):
                        c["facts"] += 1
                        c["tf_depth"] += depth[x]
                        c["tfm_depth"] += depth[j]
                        c["tf_yes"] += fact_yes(by[x])
                        c["tfm_yes"] += fact_yes(by[j])
                        c["tf_comp"] += S.answer_component(by[x], r)
                        c["tfm_comp"] += S.answer_component(by[j], r)
                        c["same_depth"] += depth[x] == depth[j]
                        c["tfm_ancestor"] += j in anc
    out = {}
    for fam, c in sorted(T.items()):
        f = max(1, c["facts"])
        out[fam] = {"items": c["items"], "nodes": c["nodes"],
                    "parent_lost_edge_share": round(c["edges_parent_lost"] / max(1, c["edges"]), 4),
                    "nodes_with_lost_parent": round(c["nodes_parent_lost"] / c["nodes"], 4),
                    "nodes_all_parents_lost": c["nodes_all_parents_lost"],
                    "eligible_share": round(c["eligible"] / max(1, c["views"]), 4),
                    "tf_tfm_differ_share_of_views": round(c["differ"] / max(1, c["views"]), 4),
                    "depth_tf": round(c["tf_depth"] / f, 3), "depth_tfm": round(c["tfm_depth"] / f, 3),
                    "yes_tf": round(c["tf_yes"] / f, 3), "yes_tfm": round(c["tfm_yes"] / f, 3),
                    "comp_tf": round(c["tf_comp"] / f, 3), "comp_tfm": round(c["tfm_comp"] / f, 3),
                    "same_depth_match": round(c["same_depth"] / f, 3),
                    "tfm_ancestor_share": round(c["tfm_ancestor"] / f, 4), "facts": c["facts"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--plans", type=int, default=1, help="full-tree plans drawn per item (different seeds)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = {}
    cols = ["items", "nodes", "parent_lost_edge_share", "nodes_with_lost_parent", "eligible_share",
            "tf_tfm_differ_share_of_views", "depth_tf", "depth_tfm", "yes_tf", "yes_tfm", "comp_tf", "comp_tfm",
            "same_depth_match", "tfm_ancestor_share"]
    short = ["items", "nodes", "lost_edge", "lost_node", "elig", "differ", "d_tf", "d_tfm", "y_tf", "y_tfm",
             "c_tf", "c_tfm", "same_d", "anc"]
    for d in args.data:
        res[d] = report(ROOT / d, args.plans)
        print(f"\n== {d}")
        print(f"{'family':16s}" + "".join(f"{s:>10s}" for s in short))
        for fam, row in res[d].items():
            print(f"{fam:16s}" + "".join(f"{row[k]:>10}" for k in cols))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
