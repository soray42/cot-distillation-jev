"""Teacher reasoning effort vs tree shape: solve K&K items at several reasoning efforts (one trace each), extract the
reasoning tree from each trace, and compare tree size, depth, node types, parent structure and node correctness
against program truth (role nodes). Solve + tree only: no node answering, no control questions.

  python scripts/effort_tree_sweep.py --ids teacher_cache/fable_kk_v1/selected_ids.json --n 50 \
      --efforts low,medium,high --out teacher_cache/effort_sweep_kk --budget 0.4
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.deepseek import DeepSeek  # noqa: E402
from cotdistill.sources import kk_subq_truth  # noqa: E402
from cotdistill.teacher import extract_tree, leaks_final, solve  # noqa: E402
from cotdistill.treeplan import node_depths  # noqa: E402


def run_one(client, rec: dict, effort: str, out: Path) -> dict | None:
    f = out / effort / f"{rec['item_id']}.json"
    if f.exists():
        return json.loads(f.read_text())
    tr = solve(client, rec, 1, effort=effort)[0]
    nodes = extract_tree(client, rec, tr) if tr.get("reasoning") else []
    for nd in nodes:
        nd["leak"] = leaks_final(nd["question"], rec)
        nd["truth"] = kk_subq_truth(nd["question"], rec)
    res = {"item_id": rec["item_id"], "effort": effort, "gold": rec["gold_label"],
           "pred": max(tr["dist"]["probs"], key=tr["dist"]["probs"].get) if tr["dist"].get("probs") else None,
           "cot_chars": len(tr.get("reasoning") or ""), "finish": tr.get("finish_reason"),
           "usage": tr.get("usage", {}), "nodes": nodes}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(res))
    return res


def stats(rows: list[dict]) -> dict:
    n_nodes, depth, npar, elig, types, acc, cov, chars, right = [], [], [], [], collections.Counter(), [], [], [], []
    for r in rows:
        nodes = [nd for nd in r["nodes"] if not nd["leak"]]
        n_nodes.append(len(nodes))
        chars.append(r["cot_chars"])
        right.append(r["pred"] == r["gold"])
        if not nodes:
            continue
        d = node_depths(nodes)
        depth.append(max(d.values()))
        ids = {nd["id"] for nd in nodes}
        ps = [len([x for x in (nd.get("depends_on") or []) if x in ids]) for nd in nodes]
        npar.append(statistics.mean(ps))
        elig.append(sum(p > 0 for p in ps) / len(ps))
        types.update(nd.get("type") for nd in nodes)
        known = [nd for nd in nodes if nd["truth"] is not None]
        cov.append(len(known) / len(nodes))
        acc += [(nd["stated"] == "yes") == nd["truth"] for nd in known]
    m = lambda x: round(statistics.mean(x), 3) if x else None
    tot = sum(types.values()) or 1
    return {"items": len(rows), "answer_acc": m(right), "cot_chars": m(chars), "nodes": m(n_nodes),
            "max_depth": m(depth), "parents_per_node": m(npar), "share_with_parents": m(elig),
            "types": {k: round(v / tot, 3) for k, v in types.most_common()},
            "truth_coverage": m(cov), "node_acc_vs_truth": m(acc), "n_truth_nodes": len(acc),
            "completion_tokens": m([r["usage"].get("completion_tokens", 0) for r in rows])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="json list of K&K item ids (teacher_cache/label_kk_v1 items)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--efforts", default="low,medium,high")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget", type=float, default=0.4, help="USD: stop submitting new items above this spend")
    args = ap.parse_args()
    ids = json.load(open(args.ids))[:args.n]
    items = {json.loads(l)["item_id"]: json.loads(l) for l in open(ROOT / "teacher_cache/label_kk_v1/items.jsonl")}
    recs = [items[i] for i in ids]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = DeepSeek(api_key=open(Path.home() / ".cotd_ds_key").read().strip(), log_path=str(out / "calls.jsonl"))
    spent = lambda: sum(json.loads(l).get("usd") or 0 for l in open(out / "calls.jsonl")) if (out / "calls.jsonl").exists() else 0.0
    efforts = args.efforts.split(",")
    results = {e: [] for e in efforts}
    jobs = [(rec, e) for rec in recs for e in efforts]          # interleaved, so a budget stop keeps the efforts balanced
    with ThreadPoolExecutor(args.workers) as ex:
        futs = []
        for rec, e in jobs:
            if spent() > args.budget:
                print(f"budget reached at ${spent():.3f}; stopping submissions", flush=True)
                break
            futs.append((e, ex.submit(run_one, client, rec, e, out)))
        for e, fu in futs:
            try:
                r = fu.result()
                if r:
                    results[e].append(r)
            except Exception as err:              # one failed item must not lose the rest
                print("item failed:", e, err, flush=True)
    common = set.intersection(*[{r["item_id"] for r in v} for v in results.values()]) if results else set()
    summary = {e: stats([r for r in v if r["item_id"] in common]) for e, v in results.items()}
    summary["_spent_usd"] = round(spent(), 3)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
