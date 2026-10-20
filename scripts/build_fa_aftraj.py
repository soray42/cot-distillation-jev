"""AFTraj-2K in-domain failure attribution as one-pass student data: train on AFTraj's own training pool, test on its
paper test split (test_unsafe, 163 failed trajectories), the protocol of ReCast (arXiv 2610.11334) Table 2, where
StepFinder 73.82, ASCon 83.03 and ReCast 84.05 Hit@1 are trained on the same dataset's training split.

Records: the fa1 builds' normalized AFTraj records (data/student_fa1/norm/aftraj.jsonl: split pool 951 / test 163,
steps = turns[1:], gold = the decisive step), clean v1 text (fa_public.clean_record), rendered exactly as the
StepFinder-protocol data (scripts/build_fa_sf.py: fit_caps, make_items with the fa1d per-item draws, the same budget
and caps). Every test record is kept (all 163, as the published numbers). Train / val drop the meta-leak records only:
the baselines of the published table train on the raw split, so clean v1's group drop (aftraj/agentic/injected, a
"Thought: I need" tell) is off unless --drop-groups.

  CUDA_VISIBLE_DEVICES= ~/.venvs/sev/bin/python -P -E scripts/build_fa_aftraj.py [--trees data/raw/aftraj_trees/aftraj.jsonl]

Writes
  data/student_fa_aftraj/train.jsonl, val.jsonl, summary.json   val = about --val-frac of the pool, whole task groups
  data/eval/fa_aftraj_test.jsonl                                 the test split (no reference line), trajectories of
                                                                 2-62 steps (the option alphabet); the summary lists
                                                                 the rest
  data/student_fa_aftraj_t2/train.jsonl, val.jsonl               with --trees: the MC reasoning-tree nodes (traj_id
                                                                 "aftraj/<uid>"), converted and checked as build_fa_sf
  data/student_fa_aftraj_t3/                                     v3 (opt-in): --trees-addon <add-on lines> and / or
                                                                 --pointer-render free, as build_fa_sf (summary.json)
With --export-sf (and nothing else): the pool (train + val) in StepFinder's file format for the tree teachers,
data/raw/aftraj_sf/AFTraj/train/<uid>.json (mistake_reason from the parquet), plus data/raw/aftraj_trees/all_ids.json
and --shards work lists with balanced length-rank targets (scripts/extract_fa_trees.py render / assemble --raw
data/raw/aftraj_sf).
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import random
import sys
import types
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cotdistill import fa_public as fp  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_for_aftraj", ROOT / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sf = _load("build_fa_sf")         # build_items, tree_items, statement_block, token_report, md5_all, write
bfp = sf.bfp
SUBSET = "aftraj"
PUBLISHED = {"StepFinder": 73.82, "ASCon": 83.03, "ReCast": 84.05}    # ReCast Table 2, Hit@1, 163 failed test trajectories


def read_trees(path: Path) -> dict:
    """uid -> tree line (traj_id "aftraj/<uid>"); a later line for the same trajectory replaces an earlier."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                x = json.loads(line)
                sub, _, uid = str(x.get("traj_id", "")).partition("/")
                if sub != SUBSET or not uid:
                    raise SystemExit(f"{path}: traj_id {x.get('traj_id')!r} is not {SUBSET}/<uid>")
                out[uid] = x
    return out


def val_groups(pool: list[dict], frac: float, seed: int) -> set:
    groups = collections.defaultdict(list)
    for r in pool:
        groups[fp.norm_task(r["task"])].append(r)
    want, chosen, n = round(frac * len(pool)), set(), 0
    for g in sorted(groups, key=lambda g: fp.stable_hash(seed, "afval", g)):
        if n >= want:
            break
        chosen.add(g)
        n += len(groups[g])
    return chosen


def export_sf(args) -> None:
    import pandas as pd                       # python3's user site, as fa_public.read_aftraj
    df = pd.read_parquet(ROOT / "data/raw/aftraj/aftraj_unsafe.parquet")
    reason = {str(r["conv_id"]): (r["mistake_reason"] if isinstance(r["mistake_reason"], str) else "")
              for _, r in df.iterrows()}
    out = ROOT / "data/raw/aftraj_sf/AFTraj/train"
    out.mkdir(parents=True, exist_ok=True)
    ids = []
    with open(ROOT / args.norm) as f:
        for r in map(json.loads, f):
            r2, why = fp.clean_record(r, "v1", drop_groups={})
            if r["split"] != "pool" or why or not 2 <= len(r2["steps"]) <= fp.MAX_STEPS:
                continue
            hist = [{"step_id": k, "name": st["agent"], "role": st["agent"], "content": st["text"]}
                    for k, st in enumerate(r2["steps"])]
            assert hist[r2["dec"]]["name"] == r2["gold_agent"], r["uid"]
            rs = reason.get(r["uid"], "").strip()
            d = {"question": r2["task"], "ground_truth": r2["answer"], "history": hist,
                 "mistake_agent": r2["gold_agent"], "mistake_step": r2["dec"],
                 "mistake_reason": rs or "(not given: an injected error; find it from the log and the ground truth)"}
            (out / f"{r['uid']}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1))
            ids.append({"traj_id": f"{SUBSET}/{r['uid']}", "subset": SUBSET, "n_steps": len(hist),
                        "mistake_step": r2["dec"], "tercile": None, "in_fa1_pool": None})
    random.Random("aftraj-trees|20261014").shuffle(ids)
    for i, x in enumerate(ids):           # balanced length-rank targets of the content and conflict nodes
        x["target_len_rank"] = {"content": i % 4, "conflict": (i // 4 + i) % 4}
    base = ROOT / "data/raw/aftraj_trees"
    (base / "shards").mkdir(parents=True, exist_ok=True)
    (base / "src").mkdir(exist_ok=True)
    (base / "all_ids.json").write_text(json.dumps(ids, indent=0))
    for k in range(args.shards):
        (base / "shards" / f"shard_{k:02d}.json").write_text(json.dumps(ids[k::args.shards], indent=0))
    print(f"exported {len(ids)} trajectories, {args.shards} shards")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--norm", default="data/student_fa1/norm/aftraj.jsonl")
    ap.add_argument("--out", default="data/student_fa_aftraj")
    ap.add_argument("--test-out", default="data/eval/fa_aftraj_test.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--gt-p", type=float, default=0.5)
    ap.add_argument("--budget", type=int, default=6144)
    ap.add_argument("--cap-max", type=int, default=3000)
    ap.add_argument("--cap-min", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default=bfp.TOKENIZER)
    ap.add_argument("--trees", default="")
    ap.add_argument("--trees-keep-failed", action="store_true")
    sf.add_v3_args(ap, pattern=False)
    ap.add_argument("--drop-groups", action="store_true", help="also drop clean v1's groups from train / val")
    ap.add_argument("--export-sf", action="store_true", help="only export the pool for the tree teachers")
    ap.add_argument("--shards", type=int, default=14)
    args = ap.parse_args(argv)
    if args.export_sf:
        return export_sf(args)

    counts: collections.Counter = collections.Counter()
    recs = []
    with open(ROOT / args.norm) as f:
        for r in map(json.loads, f):
            r2, why = fp.clean_record(r, "v1", drop_groups=None if args.drop_groups else {})
            counts[f"clean/{r['split']}/{why or 'kept'}"] += 1
            counts["clean/text_changed"] += r2["steps"] != r["steps"] or r2["task"] != r["task"]
            if why is None or r["split"] == "test":
                recs.append(r2)
    pool = [r for r in recs if r["split"] == "pool"]
    test_all = [r for r in recs if r["split"] == "test"]
    bad = [r["uid"] for r in pool if not 2 <= len(r["steps"]) <= fp.MAX_STEPS]
    pool = [r for r in pool if 2 <= len(r["steps"]) <= fp.MAX_STEPS]
    counts["pool/dropped_steps"] = len(bad)
    test = [r for r in test_all if 2 <= len(r["steps"]) <= fp.MAX_STEPS]
    excluded = [{"uid": r["uid"], "n_steps": len(r["steps"]), "decisive_step": r["dec"]} for r in test_all
                if not 2 <= len(r["steps"]) <= fp.MAX_STEPS]
    vg = val_groups(pool, args.val_frac, args.seed)
    val = [r for r in pool if fp.norm_task(r["task"]) in vg]
    train = [r for r in pool if fp.norm_task(r["task"]) not in vg]
    assert not {fp.norm_task(r["task"]) for r in val} & {fp.norm_task(r["task"]) for r in train}
    test_tasks = {fp.norm_task(r["task"]) for r in test_all}
    counts["train/tasks_in_test"] = sum(fp.norm_task(r["task"]) in test_tasks for r in train)
    print(f"aftraj: pool {len(pool)} (val {len(val)} in {len(vg)} tasks, train {len(train)}); test {len(test)} of "
          f"{len(test_all)}; {dict(counts)}", flush=True)

    tok = bfp.Tok(args.tokenizer)
    caps, cap_stats = {}, {}
    for name, rs in (("train", train), ("val", val), ("test", test)):
        st: dict = {}
        caps.update(bfp.fit_caps(rs, tok, args, st))
        cap_stats[name] = st | {"cut": sf.cut_stats(rs, caps)}
    fails: list = []
    files: dict = {}
    tr, _ = sf.build_items(train, caps, SUBSET, "train", args, fails)
    va, _ = sf.build_items(val, caps, SUBSET, "val", args, fails)
    te = []
    for k, r in enumerate(sorted(test, key=lambda r: r["uid"])):
        s, _ = fp.make_items(r, f"fasf-aftraj-test-{k:05d}", "fa_sf/aftraj_test", cap=caps[r["uid"]], with_gt=False,
                             rng=random.Random(f"fa1|{args.seed}|{r['uid']}|items"))
        b = fp.audit_item(s)
        if b:
            fails.append((s["item_id"], b[:3]))
        te.append(s)
    if fails:
        raise SystemExit(f"{len(fails)} items fail fa_public.audit_item, e.g. {fails[:3]}")
    out = ROOT / args.out
    sf.write(out / "train.jsonl", tr, files)
    sf.write(out / "val.jsonl", va, files)
    sf.write(ROOT / args.test_out, te, files)

    trees = None
    if args.trees:
        tl = read_trees(ROOT / args.trees)
        by_uid = {r["uid"]: r for r in pool}
        tstats = {"file": args.trees, "lines": len(tl), "items": 0, "items_with_tree": 0,
                  "refused": collections.Counter(), "types": collections.Counter(), "n_options": collections.Counter(),
                  "answer_pos": collections.Counter(), "depth": collections.Counter(),
                  "answer_step_is_decisive": collections.Counter(), "facts": collections.Counter()}
        addons, suffix = sf.v3_setup(args, tstats, SUBSET)
        t_tr = sf.tree_items(tr, by_uid, tl, args.trees_keep_failed, tstats, addons, args.pointer_render)
        t_va = sf.tree_items(va, by_uid, tl, args.trees_keep_failed, tstats, addons, args.pointer_render)
        for it in t_tr + t_va:
            sf.statement_block(it)
        sf.write(ROOT / f"{args.out}{suffix}" / "train.jsonl", t_tr, files)
        sf.write(ROOT / f"{args.out}{suffix}" / "val.jsonl", t_va, files)
        trees = {k: (dict(sorted(v.items(), key=lambda kv: str(kv[0]))) if isinstance(v, collections.Counter) else v)
                 for k, v in tstats.items()}
        trees["tokens_train"] = sf.token_report(t_tr, tok, args.budget)
        if "v3" in tstats:
            sf.v3_summary(ROOT / f"{args.out}{suffix}", args, trees, files)

    summary = {"args": vars(args), "counts": dict(sorted(counts.items())), "published_hit1": PUBLISHED,
               "train": len(tr), "val": len(va), "val_tasks": len(vg), "test": len(te), "test_all": len(test_all),
               "test_excluded": excluded, "caps": cap_stats,
               "baselines": {n: bfp.baselines(rs, train) for n, rs in (("val", val), ("test_fit", test),
                                                                        ("test_all", test_all))},
               "tokens_train": sf.token_report(tr, tok, args.budget), "trees": trees, "md5": sf.md5_all(files)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: summary[k] for k in ("train", "val", "test", "test_all", "md5")}, indent=1), flush=True)
    return summary


if __name__ == "__main__":
    main()
