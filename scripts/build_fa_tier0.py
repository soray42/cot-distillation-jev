"""Tier-0 failure-attribution data (notes/next_round/failure_attribution/pilot_design.md §2.1; generator in
src/cotdistill/fa_tier0.py): simulated multi-agent group chats on numgen word problems, programmatic fault injection,
exact labels from re-simulation. CPU only, no API, no model.

  data/student_fa0/train.jsonl, val.jsonl   step-level items in the ww_alg_step layout (source fa_tier0): prompt,
                                            labels, label_order, label_names, gold_label, one-hot teacher, gold_probs,
                                            type, group, item_id, subqs (V/C/D/I/R nodes: id, question, opposite,
                                            truth, p_cot, type, depends_on, node_depth, status, ...), tree_fix, plus
                                            fa / trajectory / program (exact labels; the trainer reads none of them).
                                            "Correct final answer (for reference)" in --gt-p of the items (with_gt).
  data/student_fa0/train_agent.jsonl, val_agent.jsonl   the agent-level question of the same trajectories
  data/eval/fa0_test.jsonl                  test trajectories, step level, no ground truth line, with nodes
  data/eval/fa0_test_gt.jsonl               the same with the ground-truth line (no nodes; the ww_alg_step_gt twin)
  data/eval/fa0_test_agent.jsonl            agent level
  data/student_fa0/summary.json             md5s, counts, histograms, trivial baselines, surface probe, token lengths

Splits: one trajectory per problem; problems generated per split from their own seeded stream (test, then val, then
train), never repeating a numgen signature or a task text across splits, and never equal to a problem of
data/student_numgen or data/eval/numgen_* (training frames only: numgen.HELD_OUT stays out). Decisive positions:
the generator draws a uniform relative target position per problem; each split is then selected from a pool of
--pool x n trajectories with equal quotas over relative-position deciles.

  ~/.venvs/sev/bin/python scripts/build_fa_tier0.py        # tokenizers for the Qwen3.5 token counts (else chars/4)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import resource
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cotdistill import fa_tier0 as fa  # noqa: E402
from cotdistill import numgen as ng  # noqa: E402
from cotdistill.treeplan import problem_no_options, statement_block  # noqa: E402

TRAIN_FRAMES = [f for f in ng.FRAMES if f["id"] not in ng.HELD_OUT]
TOKENIZER = "results/weights/T1-Qwen3.5-2B-Base-student_v3-s2/model/tokenizer.json"


def rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prog_signature(prog: dict) -> str:
    """numgen.signature from an item's program field."""
    return prog["frame"] + "|" + ";".join(f"{q['op']}:{q['const']}:{','.join(q['inputs'])}:{q['value']}"
                                          for q in prog["quantities"])


def numgen_exclusions() -> tuple[set, set, list]:
    sigs, tasks, files = set(), set(), []
    paths = [ROOT / "data/student_numgen/train.jsonl", ROOT / "data/student_numgen/val.jsonl"] + \
        sorted((ROOT / "data/eval").glob("numgen_*.jsonl"))
    for path in paths:
        if not path.exists():
            continue
        files.append(str(path.relative_to(ROOT)))
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("program"):
                    sigs.add(prog_signature(r["program"]))
                first = r["prompt"].split("\nOptions:\n")[0]
                if first.startswith("Question: "):
                    tasks.add(first[len("Question: "):])
    return sigs, tasks, files


def rel_bin(dec: int, n: int) -> int:
    return min(9, int(10 * dec / max(1, n - 1)))


def select(bins: list[int], n: int) -> list[int]:
    """Indices (in generation order) of n pool items with equal quotas over the 10 bins; bins that run short leave
    their slots to the others, filled round-robin from the least filled bin."""
    by = collections.defaultdict(list)
    for i, b in enumerate(bins):
        by[b].append(i)
    quota = n // 10
    take = {b: by[b][:quota] for b in range(10)}
    left = {b: by[b][quota:] for b in range(10)}
    while sum(len(v) for v in take.values()) < n:
        open_ = [b for b in range(10) if left[b]]
        if not open_:
            raise SystemExit(f"pool too small for {n} items")
        b = min(open_, key=lambda x: (len(take[x]), x))
        take[b].append(left[b].pop(0))
    return sorted(i for v in take.values() for i in v)


# ------------------------------------------------------------------------------------------------- generation
def gen_pool(split: str, n_pool: int, args, seen_sig: set, seen_task: set, excl: tuple, counts: collections.Counter):
    """Pass 1: (problem, trajectory key, relative-position bin) for n_pool problems with a kept trajectory."""
    prng = random.Random(f"fa0|{args.seed}|{split}|problems")
    depths = [int(x) for x in args.depths.split(",")]
    pool = []
    i = 0
    while len(pool) < n_pool:
        frame = prng.choice(TRAIN_FRAMES)
        depth = prng.choice(depths)
        try:
            p = ng.generate(frame, depth, prng)
        except RuntimeError:
            counts[f"{split}/numgen_fail"] += 1
            continue
        sig, task = ng.signature(p), f"{p.body} {p.question}"
        if sig in excl[0] or task in excl[1]:
            counts[f"{split}/numgen_overlap_dropped"] += 1
            continue
        if sig in seen_sig or task in seen_task:
            counts[f"{split}/duplicate_dropped"] += 1
            continue
        seen_sig.add(sig)
        seen_task.add(task)
        key = f"fa0-{split}-p{i}"
        i += 1
        res = fa.build_trajectory(p, key, random.Random(f"fa0|{args.seed}|{key}"), p_multi=args.p_multi,
                                  min_steps=args.min_steps, max_steps=args.max_steps)
        if res is None:
            counts[f"{split}/problem_without_trajectory"] += 1
            continue
        for k, v in res["rejects"].items():
            counts[f"{split}/reject/{k}"] += v
        n = len(res["run"].steps)
        pool.append((p, key, rel_bin(res["labels"]["decisive"], n)))
    return pool


def write_split(split: str, chosen: list, args, out_paths: dict, stats: list, fails: list) -> None:
    """Pass 2: rebuild the selected trajectories (deterministic per key) and write their items."""
    files = {k: open(v, "w") for k, v in out_paths.items()}
    for k, (p, key, _) in enumerate(chosen):
        res = fa.build_trajectory(p, key, random.Random(f"fa0|{args.seed}|{key}"), p_multi=args.p_multi,
                                  min_steps=args.min_steps, max_steps=args.max_steps)
        irng = random.Random(f"fa0|{args.seed}|{key}|items")
        with_gt = split != "test" and irng.random() < args.gt_p
        iid = f"fa0-{split}-{k:05d}"
        step, agent = fa.make_items(res, iid, with_gt, irng)
        bad = fa.audit_item(step)
        # the contexts the trainer builds for GFP-S / GFPQ node views
        block = statement_block(step)
        if (step["prompt"] != block + "\n\n" + fa.Q_STEP + "\nOptions:\n" + fa.options_block(
                list(step["label_names"].values())) or problem_no_options(step) != block + "\n\n" + fa.Q_STEP
                or "Question:" in block or "Options:" in block):
            bad.append("statement block")
        if bad:
            fails.append((iid, bad[:3]))
        if split == "test":
            files["step"].write(json.dumps(step) + "\n")
            gt = {k2: v for k2, v in fa.make_items(res, iid + "-gt", True, random.Random(0))[0].items()
                  if k2 not in ("subqs", "trajectory", "program")}
            gt["has_tree"] = False
            files["gt"].write(json.dumps(gt) + "\n")
        else:
            files["step"].write(json.dumps(step) + "\n")
        files["agent"].write(json.dumps(agent) + "\n")
        stats.append(summarize(step, agent, res))
    for f in files.values():
        f.close()


def summarize(step: dict, agent: dict, res: dict) -> dict:
    tr = step["trajectory"]
    nodes = step["subqs"]
    return {"split_id": step["item_id"], "n": len(tr), "dec": step["fa"]["decisive_step"],
            "roles": [t["role"] for t in tr], "agents": [t["agent"] for t in tr],
            "agent_opts": agent["label_names"], "agent_gold": agent["gold_label"],
            "invalid": step["fa"]["invalid_steps"], "inherited": step["fa"]["inherited_steps"],
            "faults": {t["step"]: t["fault"] for t in tr if t["fault"]},
            "dec_fault": step["fa"]["decisive_fault"], "dec_role": step["fa"]["decisive_role"],
            "pattern": step["fa"]["pattern"], "candidates": step["fa"]["candidates"], "depth": step["depth"],
            "with_gt": step["with_gt"], "n_agents": step["fa"]["n_agents"], "verbose": step["fa"]["verbose"],
            "nodes": [(n["type"], n["truth"], n["polarity"], n["node_depth"]) for n in nodes],
            "texts": [s.text for s in res["run"].steps], "events": [t["event"] for t in tr],
            "prompt_chars": len(step["prompt"]),
            "agent_chars": len(agent["prompt"])}


# ------------------------------------------------------------------------------------------------- statistics
def pct(xs: list, ps=(5, 25, 50, 75, 95, 99, 100)) -> dict:
    a = np.array(xs, dtype=float)
    return {f"p{p}": round(float(np.percentile(a, p)), 1) for p in ps} | {"mean": round(float(a.mean()), 1)}


def split_stats(S: list) -> dict:
    n = len(S)
    hist = lambda xs: dict(sorted(collections.Counter(xs).items()))
    multi = [s for s in S if len(s["invalid"]) >= 2]
    nodes = [x for s in S for x in s["nodes"]]
    fam = collections.defaultdict(lambda: [0, 0])
    fam_pol = collections.defaultdict(lambda: [0, 0])
    for t, truth, pol, _ in nodes:
        fam[t][0] += 1
        fam[t][1] += truth
        fam_pol[f"{t}/{pol}"][0] += 1
        fam_pol[f"{t}/{pol}"][1] += truth
    all_faults = collections.Counter(f for s in S for f in s["faults"].values())
    nondec = collections.Counter(f for s in S for i, f in s["faults"].items() if i != s["dec"])
    return {
        "n": n, "steps": hist([s["n"] for s in S]), "steps_mean": round(sum(s["n"] for s in S) / n, 2),
        "agents": hist([s["n_agents"] for s in S]), "depth": hist([s["depth"] for s in S]),
        "with_gt": round(sum(s["with_gt"] for s in S) / n, 3), "verbose": round(sum(s["verbose"] for s in S) / n, 3),
        "n_invalid": hist([len(s["invalid"]) for s in S]),
        "multi_error_share": round(len(multi) / n, 3),
        "first_invalid_is_decisive": round(sum(s["invalid"][0] == s["dec"] for s in S) / n, 3),
        "first_invalid_is_decisive_in_multi": round(sum(s["invalid"][0] == s["dec"] for s in multi) / max(1, len(multi)), 3),
        "more_than_one_candidate": round(sum(len(s["candidates"]) > 1 for s in S) / n, 3),
        "inherited_steps_per_item": round(sum(len(s["inherited"]) for s in S) / n, 2),
        "decisive_index": hist([s["dec"] for s in S]),
        "decisive_rel_decile": hist([rel_bin(s["dec"], s["n"]) for s in S]),
        "decisive_is_last": round(sum(s["dec"] == s["n"] - 1 for s in S) / n, 3),
        "decisive_fault": dict(collections.Counter(s["dec_fault"] for s in S).most_common()),
        "decisive_role": dict(collections.Counter(s["dec_role"] for s in S).most_common()),
        "nondecisive_fault": dict(nondec.most_common()), "all_faults": dict(all_faults.most_common()),
        "pattern": dict(collections.Counter(s["pattern"] for s in S).most_common()),
        "nodes_per_item": round(len(nodes) / n, 1),
        "node_families": {k: {"n": v[0], "yes": round(v[1] / v[0], 3)} for k, v in sorted(fam.items())},
        "node_families_by_wording": {k: {"n": v[0], "yes": round(v[1] / v[0], 3)} for k, v in sorted(fam_pol.items())},
        "node_yes_overall": round(sum(t for _, t, _, _ in nodes) / len(nodes), 3),
        "node_depth": hist([d for *_, d in nodes]),
        "prompt_chars": pct([s["prompt_chars"] for s in S]),
    }


def baselines(S: list, train: list) -> dict:
    n = len(S)
    role_freq = collections.Counter(s["dec_role"] for s in train)
    top_role = role_freq.most_common(1)[0][0]
    const_idx = collections.Counter(s["dec"] for s in train).most_common(1)[0][0]
    agent_role_freq = collections.Counter(s["dec_role"] for s in train)

    def first_of(s, role):
        return next((i for i, r in enumerate(s["roles"]) if r == role), 0)
    can_err = lambda e: (e in ("plan", "given", "report") or e.startswith("check:")
                         or (e.startswith("op:") and not e.endswith("/exec")))
    out = {
        "chance_step": round(sum(1 / s["n"] for s in S) / n, 3),
        "chance_over_steps_that_can_err": round(sum(1 / sum(map(can_err, s["events"])) for s in S) / n, 3),
        "always_first": round(sum(s["dec"] == 0 for s in S) / n, 3),
        "always_last": round(sum(s["dec"] == s["n"] - 1 for s in S) / n, 3),
        "first_step_of_top_faulty_role": {"role (train)": top_role,
                                          "acc": round(sum(s["dec"] == first_of(s, top_role) for s in S) / n, 3)},
        "best_constant_index": {"index (train)": const_idx,
                                "acc": round(sum(s["dec"] == min(const_idx, s["n"] - 1) for s in S) / n, 3)},
        "oracle_first_invalid_step": round(sum(s["dec"] == s["invalid"][0] for s in S) / n, 3),
        "oracle_last_invalid_step": round(sum(s["dec"] == s["invalid"][-1] for s in S) / n, 3),
        "chance_agent": round(sum(1 / len(s["agent_opts"]) for s in S) / n, 3),
        "agent_always_A": round(sum(s["agent_gold"] == "A" for s in S) / n, 3),
    }
    # the most frequent decisive role on train, mapped to the agent playing it (else option A)
    acc = 0
    for s in S:
        name = {v: k for k, v in s["agent_opts"].items()}
        pick = "A"
        for role, _ in agent_role_freq.most_common():
            ag = next((a for a, r in zip(s["agents"], s["roles"]) if r == role), None)
            if ag is not None:
                pick = name[ag]
                break
        acc += pick == s["agent_gold"]
    out["agent_top_faulty_role"] = round(acc / n, 3)
    return out


ROLES = ["planner", "reader", "calculator", "coder", "executor", "verifier", "reporter"]
CUES = ["TERMINATE", "Correct", "Confirmed", "matches", "mistake", "not ", "Correction", "```", "Recomput",
        "Updated", "final answer", "Answer:", "For reference", "Side note", "keep track"]


EVENTS = ["plan", "given", "op", "check", "side", "chat", "report", "term", "re"]


def step_features(s: dict, level: str) -> np.ndarray:
    """level "position": index features only; "structure": + the step's role and action type (plan, given, operation,
    check, side remark, chatter, report, terminate, recomputation; the Executor's output); "surface": + length, digit
    count, cue words and first-turn-of-the-role."""
    n = s["n"]
    rows = []
    for i in range(n):
        rel = i / max(1, n - 1)
        f = [rel, rel * rel, i == 0, i == n - 1, i == n - 2] + [min(i, 15) == k for k in range(16)]
        if level in ("structure", "surface"):
            ev = s["events"][i]
            f += [s["roles"][i] == r for r in ROLES] + [ev.split(":")[0] == e for e in EVENTS] + [ev.endswith("/exec")]
        if level == "surface":
            t = s["texts"][i]
            digits = sum(c.isdigit() for c in t)
            f += [math.log1p(len(t)) / 6, math.log1p(digits) / 4] + [c in t for c in CUES] + \
                 [s["roles"][:i].count(s["roles"][i]) == 0]
        rows.append(np.array(f, dtype=float))
    return np.stack(rows)


def fit_clogit(data: list, steps: int = 400, lr: float = 0.5, lam: float = 1e-3) -> np.ndarray:
    d = data[0][0].shape[1]
    w = np.zeros(d)
    for _ in range(steps):
        g = lam * w
        for X, y in data:
            z = X @ w
            p = np.exp(z - z.max())
            p /= p.sum()
            g += (X.T @ p - X[y]) / len(data)
        w -= lr * g
    return w


def probe(train: list, test: list, rng: random.Random, n_fit: int = 3000) -> dict:
    """Conditional-logit stumps over the steps of a trajectory (fit on train, scored on test): position only; position +
    structure (role, action type: which kinds of steps can be wrong at all); + surface (length, digits, cue words). A
    gap between structure and surface would flag a template tell (pilot_design §2.5)."""
    fit = rng.sample(train, min(n_fit, len(train)))
    out = {}
    for level in ("position", "structure", "surface"):
        w = fit_clogit([(step_features(s, level), s["dec"]) for s in fit])
        acc = sum(int(np.argmax(step_features(s, level) @ w)) == s["dec"] for s in test) / len(test)
        out[level] = round(acc, 3)
    return out


def token_stats(paths: dict, tok_path: str) -> dict:
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(ROOT / tok_path))
        count = lambda xs: [len(e.ids) for e in tok.encode_batch(xs, add_special_tokens=False)]
        method = f"Qwen3.5 tokenizer ({tok_path})"
        for s in (" A", " Z", " Yes", " No"):
            assert len(tok.encode(s, add_special_tokens=False).ids) == 1, s
    except Exception as e:                      # noqa: BLE001
        count = lambda xs: [len(x) // 4 for x in xs]
        method = f"chars/4 (no tokenizer: {type(e).__name__})"
    out = {"method": method}
    for name, path in paths.items():
        finals, views, per_item = [], [], []
        with open(path) as f:
            for line in f:
                it = json.loads(line)
                finals.append(it["prompt"] + "\n\nAnswer:")
                if it.get("subqs"):
                    per_item.append(len(it["subqs"]))
                    views.append(statement_block(it) + "\n\nIntermediate question: " + it["subqs"][0]["question"]
                                 + "\nOptions:\nA) Yes\nB) No\nAnswer:")
        lf = count(finals)
        out[name] = {"final_tokens": pct(lf)}
        if views:
            lv = count(views)
            out[name]["gfps_view_tokens"] = pct(lv)
            out[name]["tokens_per_epoch_final_only"] = int(sum(lf))
            out[name]["tokens_per_epoch_gfps_cap10"] = int(sum(lf) + sum(v * min(10, k) for v, k in zip(lv, per_item)))
            out[name]["tokens_per_epoch_gfps_cap0"] = int(sum(lf) + sum(v * k for v, k in zip(lv, per_item)))
    return out


# ------------------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/student_fa0")
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--pool", type=float, default=1.5, help="pool size per split / n (relative-position quotas)")
    ap.add_argument("--p-multi", type=float, default=0.6, help="probability of a second (or third) mistake")
    ap.add_argument("--gt-p", type=float, default=0.5, help="train/val items with the ground-truth line")
    ap.add_argument("--min-steps", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--depths", default="2,3,4,5,6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--no-numgen-exclusion", action="store_true", help="tests: skip reading data/ numgen files")
    args = ap.parse_args()
    out, ev = ROOT / args.out, ROOT / args.eval_dir
    out.mkdir(parents=True, exist_ok=True)
    ev.mkdir(parents=True, exist_ok=True)
    excl = (set(), set(), []) if args.no_numgen_exclusion else numgen_exclusions()
    print(f"numgen exclusions: {len(excl[0])} signatures, {len(excl[1])} tasks from {len(excl[2])} files", flush=True)
    counts: collections.Counter = collections.Counter()
    seen_sig, seen_task = set(), set()
    paths = {"test": {"step": ev / "fa0_test.jsonl", "gt": ev / "fa0_test_gt.jsonl", "agent": ev / "fa0_test_agent.jsonl"},
             "val": {"step": out / "val.jsonl", "agent": out / "val_agent.jsonl"},
             "train": {"step": out / "train.jsonl", "agent": out / "train_agent.jsonl"}}
    sizes = {"test": args.n_test, "val": args.n_val, "train": args.n_train}
    stats: dict = {}
    fails: list = []
    sig_by_split: dict = {}
    for split in ("test", "val", "train"):
        n = sizes[split]
        before = set(seen_sig)
        pool = gen_pool(split, math.ceil(n * args.pool), args, seen_sig, seen_task, excl, counts)
        sig_by_split[split] = seen_sig - before
        chosen = [pool[i] for i in select([b for *_, b in pool], n)]
        stats[split] = []
        write_split(split, chosen, args, paths[split], stats[split], fails)
        print(split, len(pool), "->", len(chosen), f"rss {rss_mb()} MB", flush=True)
    summary: dict = {"args": vars(args), "generator": "src/cotdistill/fa_tier0.py", "source": fa.SOURCE,
                     "files": {}, "md5": {}, "counts": dict(sorted(counts.items()))}
    for split, ps in paths.items():
        for kind, path in ps.items():
            rel = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            summary["md5"][rel] = md5(path)
            summary["files"][rel] = sum(1 for _ in open(path))
    summary["stats"] = {s: split_stats(stats[s]) for s in stats}
    summary["baselines"] = {s: baselines(stats[s], stats["train"]) for s in stats}
    summary["surface_probe_test"] = probe(stats["train"], stats["test"], random.Random(args.seed))
    summary["disjoint"] = {
        "test_train_signatures": len(sig_by_split["test"] & sig_by_split["train"]),
        "test_val_signatures": len(sig_by_split["test"] & sig_by_split["val"]),
        "val_train_signatures": len(sig_by_split["val"] & sig_by_split["train"]),
        "numgen_files_excluded": excl[2]}
    summary["tokens"] = token_stats({"train": paths["train"]["step"], "val": paths["val"]["step"],
                                     "test": paths["test"]["step"], "test_agent": paths["test"]["agent"]},
                                    args.tokenizer)
    summary["audit"] = {"failures": len(fails), "examples": fails[:10]}
    summary["rss_mb"] = rss_mb()
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("baselines", "surface_probe_test", "disjoint", "audit")}, indent=1))
    print(json.dumps(summary["tokens"], indent=1))
    print(f"rss {rss_mb()} MB")
    if fails:
        print("AUDIT FAILED", fails[:5])
        sys.exit(1)


if __name__ == "__main__":
    main()
