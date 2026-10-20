"""Real-failure failure-attribution data (observed failures, human step labels) for the generalization study.

Sources (all observed, not injected; data/raw, downloaded 2026-10-14):
  teb   TrajErrBench (THU-KEG/TrajDebug, MIT), English tau2-bench part: 400 failed customer-service dialogues; label
        metadata.annotation.critical_error_step indexes messages (step 0 = the policy system message)
  apb   AgentProcessBench (HF LulaCola/AgentProcessBench @ cd81f326, MIT), subsets bfcl / hotpotqa / tau2 (gaia_dev is
        left out: GAIA overlaps Who&When): trajectories with final_label -1 and at least one -1 step label; the decisive
        step = the first -1 (labels index messages)
  rse   RootSE (HF dengdan1999/RootSE @ c3e54cf2, MIT): 102 repository-level coding-agent failures; failure_id indexes
        original_traj (thought / action / response / observation per step)

Every record is rendered as the StepFinder-protocol items (fa_public.make_items): the log, "which step contains the
decisive error", options "Step k [Agent]" (2-62 steps; longer logs are counted and left out), no reference answer.

Design (generalization): teb + apb form the REAL pool, split by task (fa_public.norm_task over both sources, so a
tau2 task never sits on both sides): --test-frac of the task groups -> in-domain test, --val-frac -> val, the rest ->
train. rse is never trained on (held-out domain: code agents), nor are Who&When, TraceElephant, AFTraj, LongRCA.

  CUDA_VISIBLE_DEVICES= ~/.venvs/sev/bin/python -P -E scripts/build_fa_real.py            # items
  python3 scripts/build_fa_real.py --export-sf                                          # tree-teacher inputs

Writes data/student_fa_real/{train,val}.jsonl (+ _t2 with --trees; _t3 with v3 --trees-addon / --pointer-render free,
as build_fa_sf), data/eval/fa_real_test.jsonl (in-domain),
data/eval/fa_rse.jsonl (held-out), data/student_fa_real/summary.json; --export-sf writes data/raw/real_sf/REAL/train/
<uid>.json (StepFinder file format, train + val) and data/raw/real_trees/{all_ids.json, shards/}.
"""
from __future__ import annotations

import argparse
import ast
import collections
import glob
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cotdistill import fa_public as fp  # noqa: E402

SUBSET = "real"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_for_real", ROOT / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tool_calls_text(tc) -> str:
    if not tc:
        return ""
    if isinstance(tc, str):
        try:
            tc = ast.literal_eval(tc)                      # a Python literal list (never eval)
        except (ValueError, SyntaxError):
            return f"Tool calls: {tc}"
    out = []
    for c in tc if isinstance(tc, list) else [tc]:
        f = c.get("function", c) if isinstance(c, dict) else {}
        out.append(f"{f.get('name', '?')}({f.get('arguments', '')})")
    return "Tool calls: " + "; ".join(out)


def read_teb(counts) -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(ROOT / "data/raw/trajdebug/data/trajerrbench/en/tau2bench/*.json"))):
        d = json.loads(Path(f).read_text())
        msgs, meta = d["messages"], d["metadata"]
        k = int(meta["annotation"]["critical_error_step"])
        steps = [(m.get("name") or m["role"], str(m.get("content", ""))) for m in msgs]
        out.append(fp._rec(f"teb-{Path(f).stem}", "teb", "tau2", "natural", str(meta.get("task_description", "")),
                           None, steps, k, fp.agent_name(steps[k][0]), split="pool",
                           mode=meta["annotation"].get("critical_error_type")))
        counts["teb/read"] += 1
    return out


def read_apb(counts) -> list[dict]:
    out = []
    for sub in ("bfcl", "hotpotqa", "tau2"):
        for line in open(ROOT / f"data/raw/agentprocessbench/{sub}/test.jsonl"):
            r = json.loads(line)
            neg = [int(i) for i, v in r["step_labels"].items() if v == -1]
            if str(r["final_label"]) != "-1" or not neg:
                counts[f"apb/{sub}/skip_not_failed_or_no_error_step"] += 1
                continue
            steps = []
            for m in r["messages"]:
                text = str(m.get("content") or "")
                tc = tool_calls_text(m.get("tool_calls"))
                steps.append((m.get("name") or m["role"], "\n".join(x for x in (text, tc) if x)))
            k = min(neg)
            uid = f"apb-{sub}-{r['total_index']}"
            out.append(fp._rec(uid, "apb", sub, "natural", str(r.get("question", "")), None, steps, k,
                               fp.agent_name(steps[k][0]), split="pool"))
            counts[f"apb/{sub}/read"] += 1
    return out


def read_rse(counts) -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(ROOT / "data/raw/rootse/data/**/*.json"), recursive=True)):
        d = json.loads(Path(f).read_text())
        traj = d["original_traj"]
        steps = []
        for s in traj:
            parts = [f"{lab}: {str(s.get(key) or '').strip()}" for key, lab in
                     (("thought", "Thought"), ("action", "Action"), ("response", "Response"),
                      ("observation", "Observation")) if str(s.get(key) or "").strip()]
            steps.append((str(d.get("agent") or "agent"), "\n".join(parts)))
        k = int(d["failure_id"])
        if not 0 <= k < len(steps):
            counts["rse/drop/label_out_of_range"] += 1
            continue
        out.append(fp._rec(f"rse-{d['instance_id']}-{d.get('agent', '')}-{d.get('model', '')}", "rse",
                           str(d.get("agent") or "agent"), "natural", str(d.get("original_mission_prompt", "")), None,
                           steps, k, fp.agent_name(steps[k][0]), split="test"))
        counts["rse/read"] += 1
    return out


def fits(r) -> bool:
    return 2 <= len(r["steps"]) <= fp.MAX_STEPS


def split(pool: list[dict], test_frac: float, val_frac: float, seed: int) -> dict:
    groups = collections.defaultdict(list)
    for r in pool:
        groups[fp.norm_task(r["task"])].append(r)
    order = sorted(groups, key=lambda g: fp.stable_hash(seed, "realsplit", g))
    n, want_test, want_val = len(pool), round(test_frac * len(pool)), round(val_frac * len(pool))
    out, cnt = {}, {"test": 0, "val": 0}
    for g in order:
        side = "test" if cnt["test"] < want_test else ("val" if cnt["val"] < want_val else "train")
        if side != "train":
            cnt[side] += len(groups[g])
        for r in groups[g]:
            out[r["uid"]] = side
    assert n == len(out)
    return out


def export_sf(train_val: list[dict], shards: int) -> None:
    base = ROOT / "data/raw/real_sf/REAL/train"
    base.mkdir(parents=True, exist_ok=True)
    ids = []
    for r in train_val:
        hist = [{"step_id": k, "name": s["agent"], "role": s["agent"], "content": s["text"]}
                for k, s in enumerate(r["steps"])]
        d = {"question": r["task"], "ground_truth": "", "history": hist, "mistake_agent": r["gold_agent"],
             "mistake_step": r["dec"], "mistake_reason": "(not given: human-labelled earliest decisive error; "
             "find it from the log)" if not r.get("mode") else f"(error type label: {r['mode']})"}
        (base / f"{r['uid']}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1))
        ids.append({"traj_id": f"{SUBSET}/{r['uid']}", "subset": SUBSET, "n_steps": len(hist), "mistake_step": r["dec"],
                    "tercile": None, "in_fa1_pool": None})
    random.Random("real-trees|20261014").shuffle(ids)
    for i, x in enumerate(ids):
        x["target_len_rank"] = {"content": i % 4, "conflict": (i // 4 + i) % 4}
    tb = ROOT / "data/raw/real_trees"
    (tb / "shards").mkdir(parents=True, exist_ok=True)
    (tb / "src").mkdir(exist_ok=True)
    (tb / "all_ids.json").write_text(json.dumps(ids, indent=0))
    for k in range(shards):
        (tb / "shards" / f"shard_{k:02d}.json").write_text(json.dumps(ids[k::shards], indent=0))
    print(f"exported {len(ids)} trajectories, {shards} shards")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/student_fa_real")
    ap.add_argument("--eval-dir", default="data/eval", help="where fa_real_test.jsonl and fa_rse.jsonl are written")
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--gt-p", type=float, default=0.5)
    ap.add_argument("--budget", type=int, default=6144)
    ap.add_argument("--test-budget", type=int, default=6144, help="token budget of the test files (as training)")
    ap.add_argument("--cap-max", type=int, default=3000)
    ap.add_argument("--cap-min", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trees", default="")
    ap.add_argument("--trees-keep-failed", action="store_true")
    ap.add_argument("--trees-addon", default="", help="v3 add-on lines (fa_trees.addon_record), merged by traj_id")
    ap.add_argument("--pointer-render", choices=("index", "free"), default="index")
    ap.add_argument("--trees-suffix", default="")
    ap.add_argument("--export-sf", action="store_true")
    ap.add_argument("--shards", type=int, default=12)
    args = ap.parse_args(argv)
    counts: collections.Counter = collections.Counter()
    pool_all = read_teb(counts) + read_apb(counts)
    rse_all = read_rse(counts)
    pool = [r for r in pool_all if fits(r)]
    rse = [r for r in rse_all if fits(r)]
    counts["pool/over_62_steps"] = len(pool_all) - len(pool)
    counts["rse/over_62_steps"] = len(rse_all) - len(rse)
    side = split(pool, args.test_frac, args.val_frac, args.seed)
    by = {s: [r for r in pool if side[r["uid"]] == s] for s in ("train", "val", "test")}
    print({s: len(v) for s, v in by.items()}, "rse", len(rse), dict(counts), flush=True)
    if args.export_sf:
        export_sf(by["train"] + by["val"], args.shards)
        return
    sf = _load("build_fa_sf")
    bfp = sf.bfp
    tok = bfp.Tok(bfp.TOKENIZER)
    caps, cap_stats = {}, {}
    for name, rs, budget in (("train", by["train"], args.budget), ("val", by["val"], args.budget),
                             ("test", by["test"], args.test_budget), ("rse", rse, args.test_budget)):
        st: dict = {}
        a2 = argparse.Namespace(budget=budget, cap_max=args.cap_max, cap_min=args.cap_min)
        caps.update(bfp.fit_caps(rs, tok, a2, st))
        cap_stats[name] = st
    fails: list = []
    files: dict = {}
    tr, _ = sf.build_items(by["train"], caps, SUBSET, "train", args, fails)
    va, _ = sf.build_items(by["val"], caps, SUBSET, "val", args, fails)

    def test_items(rs, tag):
        out = []
        for k, r in enumerate(sorted(rs, key=lambda r: r["uid"])):
            s, _ = fp.make_items(r, f"fa-{tag}-{k:05d}", f"fa_sf/{tag}/{r['origin']}_{r['subset']}", cap=caps[r["uid"]],
                                 with_gt=False, rng=random.Random(f"fa1|{args.seed}|{r['uid']}|items"))
            b = fp.audit_item(s)
            if b:
                fails.append((s["item_id"], b[:3]))
            out.append(s)
        return out
    te, rs_items = test_items(by["test"], "real"), test_items(rse, "rse")
    if fails:
        raise SystemExit(f"{len(fails)} items fail fa_public.audit_item, e.g. {fails[:3]}")
    out = ROOT / args.out
    sf.write(out / "train.jsonl", tr, files)
    sf.write(out / "val.jsonl", va, files)
    sf.write(ROOT / args.eval_dir / "fa_real_test.jsonl", te, files)
    sf.write(ROOT / args.eval_dir / "fa_rse.jsonl", rs_items, files)
    trees = None
    if args.trees:
        tl = {}
        for line in open(ROOT / args.trees):
            if line.strip():
                x = json.loads(line)
                tl[str(x["traj_id"]).partition("/")[2]] = x
        by_uid = {r["uid"]: r for r in by["train"] + by["val"]}
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
        trees = {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in tstats.items()}
        if "v3" in tstats:
            sf.v3_summary(ROOT / f"{args.out}{suffix}", args, trees, files)
    comp = lambda rs: dict(collections.Counter(f"{r['origin']}/{r['subset']}" for r in rs))  # noqa: E731
    summary = {"args": vars(args), "counts": dict(sorted(counts.items())),
               "sizes": {"train": len(tr), "val": len(va), "test": len(te), "rse": len(rs_items)},
               "composition": {s: comp(by[s]) for s in by} | {"rse": comp(rse)}, "caps": cap_stats,
               "trees": trees, "md5": sf.md5_all(files)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in ("sizes", "composition", "md5")}, indent=1))


if __name__ == "__main__":
    main()
