"""Convert teacher_cache/<run>/ results into student training data.

Each output record: prompt (problem with lettered options), labels, gold_label, teacher (answer
distribution over letters, averaged over traces after mapping permuted orders back), subqs (question,
p_cot = mean P(yes) with CoT in context, p_fresh = P(yes) without CoT, truth, span_minp, status),
random_subqs (matched-count control questions with p_fresh). Split into train/val by item-id hash.

  python3 scripts/build_student_data.py --runs hard1 hard2 --out data/student --val-frac 0.1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.sources import kk_node_truth  # noqa: E402
from cotdistill.teacher import leaks_answer, node_text_ok, resolve_p_yes  # noqa: E402


def teacher_dist(res: dict) -> dict[str, float]:
    order = res["item"]["label_order"]
    acc = {k: 0.0 for k in order}
    n = 0
    for tr in res["traces"]:
        do = tr.get("dist_outcome")
        if do is None:                      # older runs: letters in the canonical order
            do = {order[ord(l) - 65]: p for l, p in tr["dist"]["probs"].items()}
        if do:
            n += 1
            for k, p in do.items():
                acc[k] += p
    return {chr(65 + i): acc[k] / n for i, k in enumerate(order)} if n else {}


def tree_graph(nodes: list[dict]) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Dependencies kept only to earlier known nodes other than the node itself, and each node's depth (0 without
    parents, else 1 + the deepest parent) on this full graph, before any node is filtered out."""
    deps, depth = {}, {}
    for sq in nodes:
        i = sq.get("id")
        if i is None or i in deps:
            continue
        deps[i] = list(dict.fromkeys(d for d in sq.get("depends_on") or [] if d in deps and d != i))
        depth[i] = 1 + max(depth[d] for d in deps[i]) if deps[i] else 0
    return deps, depth


def rewire(deps: dict[str, list[str]], kept: set[str]) -> dict[str, list[str]]:
    """Each kept node's parents with every dropped parent replaced by its own nearest kept ancestors."""
    memo: dict[str, list[str]] = {}

    def up(j: str) -> list[str]:
        if j in kept:
            return [j]
        if j not in memo:
            memo[j] = list(dict.fromkeys(x for d in deps.get(j, []) for x in up(d)))
        return memo[j]
    return {i: list(dict.fromkeys(x for d in deps[i] for x in up(d))) for i in deps if i in kept}


def convert(res: dict, keep_inconsistent: bool = False, tree_fix: bool = True) -> dict:
    """tree_fix=False reproduces the data built before the tree audit (student_v3, md5 188e2e95...). With it (audit
    F1, F2, F5, F12): K&K rule nodes get program truth under the rules (kk_rule_truth); nodes asking the final
    question in any family, in either wording, are dropped (leaks_answer); node_depth is the depth on the full graph
    before filtering; the children of a dropped node depend on its nearest kept ancestors instead (parents_orig and
    rewired record the change); self-loops are gone."""
    it = res["item"]
    labels = [chr(65 + i) for i in range(len(it["label_order"]))]
    subqs, dropped, leaked = [], 0, 0
    item = dict(it, prompt=res["prompt"])
    deps, ndepth = tree_graph(res.get("subquestions", [])) if tree_fix else ({}, {})
    for sq in res.get("subquestions", []):
        if not node_text_ok(sq["question"]):     # also for trees extracted before the notation filter existed
            dropped += 1
            continue
        if tree_fix and any(leaks_answer(q, item) for q in (sq["question"], sq.get("opposite")) if q):
            leaked += 1
            continue
        ps = [p for p in (resolve_p_yes(a) for a in sq.get("answers", [])) if p is not None]
        truth = sq.get("truth")
        if truth is None and it.get("domain") == "knights_knaves":
            truth = kk_node_truth(sq["question"], item, rule=tree_fix)  # rules, hypotheticals, same/opposite roles
        stated = sq.get("stated") if sq.get("stated") in ("yes", "no") else None
        p_cot = sum(ps) / len(ps) if ps else None
        # keep a node when its signals agree; program truth, when known, outranks the CoT-conditioned re-ask
        # (a no-thinking re-ask often misjudges hypotheticals such as dead-end branches)
        if truth is not None and stated is not None:
            consistent = (stated == "yes") == truth
        else:
            consistent = p_cot is None or stated is None or (p_cot > 0.5) == (stated == "yes")
        if not keep_inconsistent and not consistent:
            dropped += 1
            continue
        if stated is not None and (p_cot is None or (p_cot > 0.5) != (stated == "yes")):
            p_cot = 1.0 if stated == "yes" else 0.0              # the extracted (verified) answer as a hard label
        commit = sq.get("commit")
        p_commit = None
        if commit and sq.get("stated") in ("yes", "no"):       # teacher's confidence where it settled the node
            p_commit = commit["conf"] if sq["stated"] == "yes" else 1 - commit["conf"]
        subqs.append({"question": sq["question"], "p_cot": p_cot,
                      "p_fresh": resolve_p_yes(sq.get("answer_nocot") or {}), "truth": truth,
                      "type": sq.get("type"), "depends_on": sq.get("depends_on"), "id": sq.get("id"),
                      "initial": sq.get("initial"), "p_commit": p_commit,
                      "span_minp": (sq.get("span_stats") or {}).get("min_p"), "status": sq.get("status")})
    randoms = [{"question": r["question"], "p_fresh": resolve_p_yes(r.get("answer_nocot") or {})}
               for r in res.get("random_subquestions", [])][:len(subqs)]     # matched count after filtering
    out = {"item_id": it["item_id"], "source": it.get("domain") or it.get("source"), "prompt": res["prompt"],
           "labels": labels, "gold_label": it.get("gold_label"), "teacher": teacher_dist(res),
           "depth": it.get("depth"), "subqs": subqs, "random_subqs": randoms,
           "rationale": (res["traces"][0].get("reasoning") or None) if res.get("traces") else None,
           "n_inconsistent_dropped": dropped}
    if tree_fix:
        kept = {sq["id"] for sq in subqs if sq.get("id") in deps}
        new = rewire(deps, kept)
        for sq in subqs:
            i = sq.get("id")
            if i in new:
                sq.update(depends_on=new[i], parents_orig=deps[i], rewired=new[i] != deps[i], node_depth=ndepth[i])
        out.update(tree_fix=1, n_leak_dropped=leaked)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="data/student")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--keep-inconsistent", action="store_true",
                    help="keep sub-questions whose extracted answer disagrees with the CoT-conditioned answer")
    ap.add_argument("--filter-wrong-teacher", action="store_true",
                    help="drop items whose teacher answer disagrees with the gold label (rejection filtering)")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "val": []}
    for run in args.runs:
        for p in sorted((ROOT / "teacher_cache" / run).glob("*.json")):
            if p.name in ("summary.json",):
                continue
            res = json.loads(p.read_text())
            if "item" not in res:
                continue
            rec = convert(res, keep_inconsistent=args.keep_inconsistent)
            if args.filter_wrong_teacher and rec["teacher"] and rec["gold_label"] and \
                    max(rec["teacher"], key=rec["teacher"].get) != rec["gold_label"]:
                continue
            h = int(hashlib.sha1(rec["item_id"].encode()).hexdigest(), 16) % 1000 / 1000
            splits["val" if h < args.val_frac else "train"].append(rec)
    for name, recs in splits.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        n_sq = sum(len(r["subqs"]) for r in recs)
        n_drop = sum(r.get("n_inconsistent_dropped", 0) for r in recs)
        print(f"{name}: {len(recs)} items, {n_sq} sub-questions ({n_drop} inconsistent dropped), "
              f"{sum(len(r['random_subqs']) for r in recs)} random sub-questions")


if __name__ == "__main__":
    main()
