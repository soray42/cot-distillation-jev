"""Fine-grained tree extraction pilot: re-extract trees from the teacher's saved raw reasoning (no new solve) with
TREE_PROMPT_FINE (one node per inference step, no node cap, complete depends_on), and compare with the registered
trees of the same items (teacher_cache/label_kk_v1_tree): node count, depth, parents and ancestors per node, node
types, and node correctness against program truth (role nodes).

  python scripts/fine_tree_pilot.py --ids teacher_cache/fable_kk_v1/selected_ids.json --n 50 \
      --out teacher_cache/fine_tree_kk --budget 0.1
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
from cotdistill.teacher import TREE_PROMPT_FINE, leaks_final, parse_tree  # noqa: E402
from cotdistill.treeplan import node_depths  # noqa: E402


def extract_fine(client, rec: dict, cot: str) -> list[dict]:
    msg = [{"role": "user", "content": TREE_PROMPT_FINE.format(problem=rec["prompt"], cot=cot)}]
    resp = client.chat(msg, thinking=False, logprobs=False, max_tokens=16000, json_mode=True, tag=f"{rec['item_id']}/fine")
    nodes = parse_tree(resp["choices"][0]["message"].get("content") or "", cap=None)
    if not nodes:
        resp = client.chat(msg, thinking=False, logprobs=False, max_tokens=16000, json_mode=False,
                           tag=f"{rec['item_id']}/fine_retry")
        nodes = parse_tree(resp["choices"][0]["message"].get("content") or "", cap=None)
    return nodes


def ancestors(nodes: list[dict]) -> dict[str, set]:
    by = {n["id"]: n for n in nodes}
    memo: dict[str, set] = {}

    def a(i: str, seen: frozenset) -> set:
        if i in memo:
            return memo[i]
        s: set = set()
        for p in by[i].get("depends_on") or []:
            if p in by and p not in seen and p != i:
                s |= {p} | a(p, seen | {i})
        memo[i] = s
        return s
    return {i: a(i, frozenset()) for i in by}


def stats(trees: list[list[dict]], rec_of: list[dict]) -> dict:
    n, depth, par, anc, elig, acc, cov = [], [], [], [], [], [], []
    types = collections.Counter()
    for nodes, rec in zip(trees, rec_of):
        nodes = [nd for nd in nodes if not leaks_final(nd["question"], rec)]
        n.append(len(nodes))
        if not nodes:
            continue
        d = node_depths(nodes)
        depth.append(max(d.values()))
        A = ancestors(nodes)
        ids = {nd["id"] for nd in nodes}
        ps = [len([x for x in (nd.get("depends_on") or []) if x in ids]) for nd in nodes]
        par.append(statistics.mean(ps))
        anc.append(statistics.mean(len(A[nd["id"]]) for nd in nodes))
        elig.append(sum(p > 0 for p in ps) / len(ps))
        types.update(nd["type"] for nd in nodes)
        known = [(nd, kk_subq_truth(nd["question"], rec)) for nd in nodes]
        known = [(nd, t) for nd, t in known if t is not None]
        cov.append(len(known) / len(nodes))
        acc += [(nd["stated"] == "yes") == t for nd, t in known]
    m = lambda x: round(statistics.mean(x), 3) if x else None
    tot = sum(types.values()) or 1
    return {"items": len(trees), "nodes": m(n), "nodes_max": max(n) if n else None, "max_depth": m(depth),
            "parents_per_node": m(par), "ancestors_per_node": m(anc), "share_with_parents": m(elig),
            "types": {k: round(v / tot, 3) for k, v in types.most_common()},
            "truth_coverage": m(cov), "node_acc_vs_truth": m(acc), "n_truth_nodes": len(acc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget", type=float, default=0.1)
    args = ap.parse_args()
    ids = json.load(open(args.ids))[:args.n]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = DeepSeek(api_key=open(Path.home() / ".cotd_ds_key").read().strip(), log_path=str(out / "calls.jsonl"))
    spent = lambda: sum(json.loads(l).get("usd") or 0 for l in open(out / "calls.jsonl")) if (out / "calls.jsonl").exists() else 0.0

    def one(i: str):
        f = out / f"{i}.json"
        if f.exists():
            return json.loads(f.read_text())
        res = json.load(open(ROOT / f"teacher_cache/label_kk_v1/{i}.json"))
        rec = dict(res["item"], prompt=res["prompt"])
        nodes = extract_fine(client, rec, res["traces"][0]["reasoning"])
        r = {"item_id": i, "nodes": nodes}
        f.write_text(json.dumps(r))
        return r
    done = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = []
        for i in ids:
            if spent() > args.budget:
                print(f"budget reached at ${spent():.3f}", flush=True)
                break
            futs.append(ex.submit(one, i))
        for fu in futs:
            try:
                done.append(fu.result())
            except Exception as err:
                print("item failed:", err, flush=True)
    recs, fine, old = [], [], []
    for r in done:
        res = json.load(open(ROOT / f"teacher_cache/label_kk_v1_tree/{r['item_id']}.json"))
        recs.append(dict(res["item"], prompt=res["prompt"]))
        fine.append(r["nodes"])
        old.append([{"id": nd.get("id"), "type": nd.get("type"), "depends_on": nd.get("depends_on") or [],
                     "question": nd["question"], "stated": nd.get("stated")} for nd in res["subquestions"] if "id" in nd])
    summary = {"fine": stats(fine, recs), "registered": stats(old, recs), "_spent_usd": round(spent(), 3)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
