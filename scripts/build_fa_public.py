"""Tier-1 public failure-attribution data (notes/next_round/failure_attribution/tier1_public.md; library
src/cotdistill/fa_public.py): Who&When Pro (text), AFTraj-2K (unsafe), StepFinder (regenerated train set) and AgenTracer
data-v1.0.0 as training data, TraceElephant as an extra eval set, in the Who&When eval layout (source fa_public).

Two stages (the readers need pandas, which only python3's user site has; the token counts need tokenizers, which only
the sev venv has). Run both from a neutral cwd (the raw data is untrusted; it is parsed, never executed):

  cd <scratch> && python3 -P -E ~/projects/jev/scripts/build_fa_public.py --stage normalize
  cd <scratch> && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/sev/bin/python -P -E \\
      ~/projects/jev/scripts/build_fa_public.py --stage build

  normalize  data/student_fa1/norm/{wwpro,aftraj,stepfinder,agentracer,traceelephant}.jsonl, whowhen.json,
             counts.json (one record per trajectory; drops counted by reason)
  build      data/student_fa1/train.jsonl, val.jsonl          step-level items with nodes (subqs), 50% with the
                                                              "Correct final answer (for reference)" line
             data/student_fa1/train_agent.jsonl, val_agent.jsonl   the agent-level question (multi-agent traces)
             data/eval/fa1_<src>_test.jsonl, _test_gt.jsonl, _test_agent.jsonl   for src in wwpro (10% of Who&When Pro
                                                              tasks), aftraj (the paper's test_unsafe), agentracer (its
                                                              test split), traceelephant (all, eval only)
             data/student_fa1/summary.json                    counts, dedup, mix, histograms, baselines, tokens, md5s

Splits: held-out sets first (above), then every pool trajectory whose task overlaps a held-out task or a Who&When
question (fa_public.OverlapIndex: >= 50% shared 8-grams after dropping benchmark boilerplate; or a shared GAIA /
AssistantBench id) is removed; val = about --n-val trajectories by task (no task in both val and train); per-source
caps with equal quotas over relative-position deciles where a source is subsampled. Every step of a trajectory is cut
to the same per-trajectory cap (at most --cap-max characters), the largest whose step prompt with the reference line
fits --budget Qwen3.5 tokens (prompt + "\\n\\nAnswer:").
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
import subprocess
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cotdistill import fa_public as fp  # noqa: E402

TOKENIZER = "results/weights/T1-Qwen3.5-2B-Base-student_v3-s2/model/tokenizer.json"
ORIGINS = ("wwpro", "aftraj", "stepfinder", "agentracer", "traceelephant")
TEST_ORIGINS = ("wwpro", "aftraj", "agentracer", "traceelephant")


def rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------- stage normalize
def stage_normalize(args) -> None:
    raw = ROOT / args.raw
    out = ROOT / args.out / "norm"
    out.mkdir(parents=True, exist_ok=True)
    counts: collections.Counter = collections.Counter()
    readers = {
        "wwpro": lambda: fp.read_wwpro(str(raw / "whowhen_pro/text.jsonl"), counts),
        "aftraj": lambda: fp.read_aftraj(str(raw / "aftraj/aftraj_unsafe.parquet"),
                                         str(raw / "aftraj/splits_test.json"), counts),
        "stepfinder": lambda: fp.read_stepfinder(str(raw / "stepfinder/repo/data"), counts),
        "agentracer": lambda: fp.read_agentracer(str(raw / "agentracer/agentracer-data-v1.0.0.zip"), counts),
        "traceelephant": lambda: fp.read_traceelephant(str(raw / "traceelephant/data.zip"), counts),
    }
    for origin in args.origins.split(","):
        n = 0
        with open(out / f"{origin}.jsonl", "w") as f:
            for rec in readers[origin]():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                counts[f"{origin}/kept_{rec['split']}"] += 1
        print(f"{origin}: {n} records, rss {rss_mb()} MB", flush=True)
    ww = fp.read_whowhen(str(raw / "who_and_when"))
    (out / "whowhen.json").write_text(json.dumps(ww, ensure_ascii=False))
    counts["whowhen/trajectories"] = len(ww)
    (out / "counts.json").write_text(json.dumps(dict(sorted(counts.items())), indent=1))
    print(json.dumps(dict(sorted(counts.items())), indent=1))


# ------------------------------------------------------------------------------------------------- stage build
def load(norm: Path, origin: str) -> list[dict]:
    with open(norm / f"{origin}.jsonl") as f:
        return [json.loads(line) for line in f]


def rel_bin(dec: int, n: int) -> int:
    return min(9, int(10 * dec / max(1, n - 1)))


def pick(recs: list[dict], n: int, key: str, balance: bool) -> list[dict]:
    """n records in a seeded order; with balance, equal quotas over relative-position deciles of the decisive step
    (deciles that run short leave their slots to the others, filled round-robin from the least filled decile)."""
    order = sorted(recs, key=lambda r: fp.stable_hash(key, r["uid"]))
    if n >= len(order):
        return order
    if not balance:
        return order[:n]
    by = collections.defaultdict(list)
    for r in order:
        by[rel_bin(r["dec"], len(r["steps"]))].append(r)
    take = {b: by[b][:n // 10] for b in range(10)}
    left = {b: by[b][n // 10:] for b in range(10)}
    while sum(len(v) for v in take.values()) < n:
        b = min((x for x in range(10) if left[x]), key=lambda x: (len(take[x]), x))
        take[b].append(left[b].pop(0))
    return [r for v in take.values() for r in v]


def wwpro_test(recs: list[dict], frac: float, seed: int) -> set:
    """Task groups (normalized query, across frameworks) held out so that about frac of every framework's traces is in
    the test set (stratified by framework; a chosen group takes its traces of every framework)."""
    groups = collections.defaultdict(list)
    for r in recs:
        groups[fp.norm_task(r["task"])].append(r)
    by_fw = collections.defaultdict(set)
    for g, rs in groups.items():
        for r in rs:
            by_fw[r["subset"]].add(g)
    chosen: set = set()
    for fw in sorted(by_fw):
        total = sum(r["subset"] == fw for r in recs)
        have = sum(r["subset"] == fw for g in chosen for r in groups[g])
        for g in sorted(by_fw[fw], key=lambda g: fp.stable_hash(seed, "wwtest", g)):
            if have >= round(frac * total):
                break
            if g not in chosen:
                chosen.add(g)
                have += sum(r["subset"] == fw for r in groups[g])
    return chosen


class Tok:
    def __init__(self, path: str):
        try:
            from tokenizers import Tokenizer
            self.tok = Tokenizer.from_file(str(ROOT / path))
            self.method = f"Qwen3.5 tokenizer ({path})"
            for s in (" A", " Z", " a", " z", "0", "9", " Yes", " No"):
                assert len(self.tok.encode(s, add_special_tokens=False).ids) == 1, s
        except Exception as e:                  # noqa: BLE001
            self.tok = None
            self.method = f"chars/3.6 (no tokenizer: {type(e).__name__})"

    def count(self, texts: list[str]) -> list[int]:
        if self.tok is None:
            return [math.ceil(len(t) / 3.6) for t in texts]
        out = []
        for i in range(0, len(texts), 64):            # small batches: long prompts, RSS < 1.5 GB
            out += [len(e.ids) for e in self.tok.encode_batch(texts[i:i + 64], add_special_tokens=False)]
        return out


def fit_caps(recs: list[dict], tok: Tok, args, stats: dict) -> dict:
    """Per-trajectory step cap: the largest cap (<= --cap-max) whose step prompt with the reference line fits --budget
    tokens; the character budget starts at --budget x 3.6 and shrinks by the measured token excess (6 rounds)."""
    caps, cb = {}, {}
    active = list(range(len(recs)))
    for r in recs:
        cb[r["uid"]] = args.budget * 3.6
    toks = {}
    for _ in range(8):
        for i in active:
            r = recs[i]
            caps[r["uid"]] = fp.fit_cap(r, int(cb[r["uid"]]), args.cap_max, args.cap_min)
        n = tok.count([fp.step_prompt(recs[i], caps[recs[i]["uid"]], True) + "\n\nAnswer:" for i in active])
        nxt = []
        for i, t in zip(active, n):
            r = recs[i]
            toks[r["uid"]] = t
            if t > args.budget and caps[r["uid"]] > args.cap_min:
                cb[r["uid"]] *= 0.985 * args.budget / t
                nxt.append(i)
        active = nxt
        if not active:
            break
    stats["over_budget"] = sum(t > args.budget for t in toks.values())
    stats["at_cap_max"] = sum(c == args.cap_max for c in caps.values())
    stats["at_cap_min"] = sum(c == args.cap_min for c in caps.values())
    return caps


def pct(xs: list, ps=(5, 25, 50, 75, 90, 95, 99, 100)) -> dict:
    import numpy as np
    if not xs:
        return {}
    a = np.array(xs, dtype=float)
    return {f"p{p}": round(float(np.percentile(a, p)), 1) for p in ps} | {"mean": round(float(a.mean()), 1)}


def hist(xs) -> dict:
    return dict(sorted(collections.Counter(xs).items()))


def describe(recs: list[dict]) -> dict:
    if not recs:
        return {"n": 0}
    n = len(recs)
    return {"n": n, "by_subset": dict(collections.Counter(f"{r['origin']}/{r['subset']}" for r in recs).most_common()),
            "steps": hist(min(len(r["steps"]), 62) for r in recs), "steps_pct": pct([len(r["steps"]) for r in recs]),
            "decisive_index": hist(r["dec"] for r in recs),
            "decisive_rel_decile": hist(rel_bin(r["dec"], len(r["steps"])) for r in recs),
            "decisive_first": round(sum(r["dec"] == 0 for r in recs) / n, 3),
            "decisive_last": round(sum(r["dec"] == len(r["steps"]) - 1 for r in recs) / n, 3),
            "over_26_steps": sum(len(r["steps"]) > 26 for r in recs),
            "over_52_steps": sum(len(r["steps"]) > 52 for r in recs),
            "multi_agent": round(sum(len(fp.agents_in_order(r)) >= 2 for r in recs) / n, 3),
            "with_answer": round(sum(bool(r.get("answer")) for r in recs) / n, 3),
            "mode": dict(collections.Counter(r.get("mode") for r in recs if r.get("mode")).most_common())}


def baselines(recs: list[dict], train: list[dict]) -> dict:
    """Trivial step / agent baselines on recs; constants fit on train (best constant index; most frequent gold agent
    name), plus the oracle best constant index on recs itself."""
    if not recs:
        return {}
    n = len(recs)
    const = collections.Counter(r["dec"] for r in train).most_common(1)[0][0]
    oracle_c = max(range(62), key=lambda c: sum(r["dec"] == min(c, len(r["steps"]) - 1) for r in recs))
    out = {"n": n, "chance_step": round(sum(1 / len(r["steps"]) for r in recs) / n, 3),
           "always_first_step": round(sum(r["dec"] == 0 for r in recs) / n, 3),
           "always_step_1": round(sum(r["dec"] == min(1, len(r["steps"]) - 1) for r in recs) / n, 3),
           "always_last_step": round(sum(r["dec"] == len(r["steps"]) - 1 for r in recs) / n, 3),
           "best_constant_index_train": {"index": const, "acc": round(
               sum(r["dec"] == min(const, len(r["steps"]) - 1) for r in recs) / n, 3)},
           "best_constant_index_oracle": {"index": oracle_c, "acc": round(
               sum(r["dec"] == min(oracle_c, len(r["steps"]) - 1) for r in recs) / n, 3)}}
    ag = [r for r in recs if not r.get("single_agent") and r.get("gold_agent") in fp.agents_in_order(r)
          and len(fp.agents_in_order(r)) >= 2]
    if ag:
        top = collections.Counter(r["gold_agent"] for r in train if r.get("gold_agent")).most_common()
        acc = 0
        for r in ag:
            opts = fp.agents_in_order(r)
            pick_ = next((a for a, _ in top if a in opts), opts[0])
            acc += pick_ == r["gold_agent"]
        own = collections.Counter(r["gold_agent"] for r in ag).most_common(1)[0]
        out["agent"] = {"n": len(ag), "chance": round(sum(1 / len(fp.agents_in_order(r)) for r in ag) / len(ag), 3),
                        "always_A_first_speaker": round(sum(fp.agents_in_order(r)[0] == r["gold_agent"] for r in ag)
                                                        / len(ag), 3),
                        "most_frequent_agent_train": round(acc / len(ag), 3),
                        "most_frequent_agent_oracle": {"agent": own[0], "acc": round(sum(
                            r["gold_agent"] == own[0] for r in ag) / len(ag), 3)},
                        "agent_of_last_step": round(sum(r["steps"][-1]["agent"] == r["gold_agent"] for r in ag)
                                                    / len(ag), 3)}
    return out


def step_features(r: dict, cap: int, level: str):
    import numpy as np
    n = len(r["steps"])
    lens = [min(len(s["text"].strip()), cap) for s in r["steps"]]
    rank = sorted(range(n), key=lambda i: lens[i])
    rk = {i: k / max(1, n - 1) for k, i in enumerate(rank)}
    rows = []
    for i in range(n):
        rel = i / max(1, n - 1)
        f = [rel, rel * rel, i == 0, i == n - 1, i == n - 2, i == 1] + [min(i, 9) == k for k in range(10)]
        if level == "surface":
            f += [math.log1p(lens[i]) / 8, rk[i], lens[i] == max(lens), lens[i] == min(lens),
                  r["steps"][i]["agent"] != (r["steps"][i - 1]["agent"] if i else None)]
        rows.append(np.array(f, dtype=float))
    return np.stack(rows)


def fit_clogit(data: list, steps: int = 300, lr: float = 0.5, lam: float = 1e-3):
    import numpy as np
    w = np.zeros(data[0][0].shape[1])
    for _ in range(steps):
        g = lam * w
        for X, y in data:
            z = X @ w
            p = np.exp(z - z.max())
            p /= p.sum()
            g += (X.T @ p - X[y]) / len(data)
        w -= lr * g
    return w


def probe(train: list[dict], evals: dict, caps: dict, seed: int, n_fit: int = 3000) -> dict:
    """Surface probe (conditional logit over a trajectory's steps, fit on a train sample): position only vs position
    plus surface (cut length, length rank, longest / shortest step, speaker change). A large gap flags a style tell."""
    import numpy as np
    fit = random.Random(seed).sample(train, min(n_fit, len(train)))
    out = {}
    for level in ("position", "surface"):
        w = fit_clogit([(step_features(r, caps[r["uid"]], level), r["dec"]) for r in fit])
        out[level] = {name: round(sum(int(np.argmax(step_features(r, caps[r["uid"]], level) @ w)) == r["dec"]
                                      for r in rs) / max(1, len(rs)), 3) for name, rs in evals.items() if rs}
    return out


def stage_build(args) -> None:
    norm = ROOT / args.out / "norm"
    out, ev = ROOT / args.out, ROOT / args.eval_dir
    ev.mkdir(parents=True, exist_ok=True)
    counts: collections.Counter = collections.Counter()
    tok = Tok(args.tokenizer)
    origins = [o for o in ORIGINS if (norm / f"{o}.jsonl").exists()]
    if not args.agentracer:
        origins = [o for o in origins if o != "agentracer"]
    recs = {o: load(norm, o) for o in origins}
    print("loaded", {o: len(v) for o, v in recs.items()}, f"rss {rss_mb()} MB", flush=True)
    for o in origins:                                       # > 62 steps or a single step: no item
        keep = []
        for r in recs[o]:
            n = len(r["steps"])
            if n > fp.MAX_STEPS or n < 2:
                counts[f"{o}/drop/{'over_62_steps' if n > 2 else 'one_step'}_{r['split']}"] += 1
            else:
                keep.append(r)
        recs[o] = keep
    # ---- held-out sets
    tests: dict = {o: [r for r in recs.get(o, []) if r["split"] == "test"] for o in TEST_ORIGINS if o in recs}
    if "wwpro" in recs:
        held = wwpro_test(recs["wwpro"], args.wwpro_test_frac, args.seed)
        tests["wwpro"] = [r for r in recs["wwpro"] if fp.norm_task(r["task"]) in held]
        for r in tests["wwpro"]:
            r["split"] = "test"
    pool = [r for o in origins for r in recs[o] if r["split"] == "pool"]
    counts["pool/before_filters"] = len(pool)
    keep = []
    for r in pool:                           # label consistency (pool only; eval files keep the labels as given)
        if "agent_mismatch" in r["flags"]:
            counts[f"{r['origin']}/drop/agent_mismatch"] += 1
        elif not r["single_agent"] and r.get("gold_agent") not in fp.agents_in_order(r):
            counts[f"{r['origin']}/drop/gold_agent_not_an_agent"] += 1
        else:
            keep.append(r)
    pool = keep
    # ---- dedup against Who&When and every held-out task
    ww = json.loads((norm / "whowhen.json").read_text())
    refs = [(f"whowhen/{x['subset']}", x["question"]) for x in ww]
    refs += [(f"test/{o}", r["task"]) for o, rs in tests.items() for r in rs]
    idx = fp.OverlapIndex(refs, df=args.boilerplate_df, thr=args.overlap)
    ww_ids = {x["question_ID"] for x in ww if x["question_ID"]}
    test_ids = {i for rs in tests.values() for r in rs for i in r["ids"]}
    dedup: collections.Counter = collections.Counter()
    sf_q = collections.defaultdict(set)
    keep = []
    for r in pool:
        ratio, who = idx.match(r["task"])
        uid_hit = set(r["ids"]) & ww_ids
        if ratio >= args.overlap:
            for k in sorted({w.split("/")[0] + "/" + w.split("/")[1] for w in who}):
                dedup[f"{r['origin']}/{k}"] += 1
            dedup[f"{r['origin']}/dropped"] += 1
            if r["origin"] == "stepfinder" and any(w.startswith("whowhen") for w in who):
                sf_q[r["subset"]].add(fp.norm_task(r["task"]))
        elif uid_hit or set(r["ids"]) & test_ids:
            dedup[f"{r['origin']}/{'whowhen' if uid_hit else 'test'}_id"] += 1
            dedup[f"{r['origin']}/dropped"] += 1
        else:
            keep.append(r)
    dedup["boilerplate_8grams"] = len(idx.boiler)
    for s, qs in sf_q.items():
        dedup[f"stepfinder/{s}/distinct_questions_overlapping_whowhen"] = len(qs)
    pool = keep
    counts["pool/after_dedup"] = len(pool)
    print("dedup", dict(sorted(dedup.items())), f"rss {rss_mb()} MB", flush=True)
    # ---- val (by task group), then per-source caps on what is left
    caps_cfg, share = json.loads(args.caps), json.loads(args.share)

    def targets(recs: list[dict]) -> dict:
        """Train size per (origin, subset): subset caps (counts), then origin share caps of the total (fixed point)."""
        n = collections.Counter((r["origin"], r["subset"]) for r in recs)
        want = {k: min(v, caps_cfg.get(f"{k[0]}/{k[1]}", 10 ** 9)) for k, v in n.items()}
        for _ in range(20):
            total = sum(want.values())
            for o, frac in share.items():
                mine = sum(v for k, v in want.items() if k[0] == o)
                if mine > frac * total + 1:
                    scale = frac / (1 - frac) * (total - mine) / mine
                    want = {k: (int(v * scale) if k[0] == o else v) for k, v in want.items()}
        return want

    per = collections.defaultdict(list)                 # (origin, subset) -> records
    for r in pool:
        per[(r["origin"], r["subset"])].append(r)
    pre = targets(pool)
    total = sum(pre.values())
    val: list = []
    val_groups: set = set()                             # task groups across every source: none in both val and train
    for k in sorted(per):
        rs, n_val = per[k], round(args.n_val * pre[k] / total)
        sub = pre[k] < len(rs)                          # subsampled for train: val drawn with the same decile quotas
        groups = collections.defaultdict(list)
        for r in rs:
            groups[fp.norm_task(r["task"])].append(r)
        cand: list = []
        for g in sorted(groups, key=lambda g: fp.stable_hash(args.seed, "val", g)):
            if len(cand) >= (math.ceil(1.5 * n_val) if sub else n_val):
                break                                   # <= 8 trajectories a group (StepFinder: ~17 per question)
            val_groups.add(g)
            cand += pick(groups[g], 8, f"{args.seed}|valg|{g}", False)
        val += pick(cand, n_val, f"{args.seed}|val|{k[0]}|{k[1]}", sub)
        counts[f"{k[0]}/{k[1]}/pool"] = len(rs)
        counts[f"{k[0]}/{k[1]}/val"] = min(n_val, len(cand))
    in_val = {r["uid"] for r in val}
    rest = [r for r in pool if fp.norm_task(r["task"]) not in val_groups]
    val_discard = collections.Counter(f"{r['origin']}/{r['subset']}" for r in pool
                                      if fp.norm_task(r["task"]) in val_groups and r["uid"] not in in_val)
    want = targets(rest)
    train: list = []
    by_k = collections.defaultdict(list)
    for r in rest:
        by_k[(r["origin"], r["subset"])].append(r)
    for k in sorted(by_k):
        train += pick(by_k[k], want[k], f"{args.seed}|train|{k[0]}|{k[1]}", want[k] < len(by_k[k]))
        counts[f"{k[0]}/{k[1]}/train"] = min(want[k], len(by_k[k]))
    print("train", len(train), "val", len(val), f"rss {rss_mb()} MB", flush=True)
    vt = {fp.norm_task(r["task"]) for r in val}
    assert not vt & {fp.norm_task(r["task"]) for r in train}, "val and train share a task"
    if args.dry:
        print(json.dumps({"counts": dict(sorted(counts.items())), "want": {"/".join(k): v for k, v in want.items()},
                          "train": describe(train), "val": describe(val),
                          "tests": {o: describe(rs) for o, rs in tests.items()}}, indent=1))
        return
    # ---- caps, items
    allrecs = {"train": train, "val": val} | {f"test_{o}": rs for o, rs in tests.items()}
    cap_stats: dict = {}
    caps = {}
    for name, rs in allrecs.items():
        st: dict = {}
        caps.update(fit_caps(rs, tok, args, st))
        cap_stats[name] = st | {"cap": pct([caps[r["uid"]] for r in rs])}
        print("caps", name, st, f"rss {rss_mb()} MB", flush=True)
    files: dict = {}
    fails = []
    node_stats = collections.defaultdict(collections.Counter)

    def write(path: Path, items: list[dict]) -> None:
        with open(path, "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        files[str(path.relative_to(ROOT))] = len(items)

    for split in ("train", "val"):
        rs = sorted(allrecs[split], key=lambda r: fp.stable_hash(args.seed, split, "order", r["uid"]))
        steps, agents = [], []
        for k, r in enumerate(rs):
            rng = random.Random(f"fa1|{args.seed}|{r['uid']}|items")
            with_gt = rng.random() < args.gt_p
            s, a = fp.make_items(r, f"fa1-{split}-{k:05d}", f"fa1/{r['origin']}", cap=caps[r["uid"]], with_gt=with_gt,
                                 rng=rng)
            bad = fp.audit_item(s)
            if bad:
                fails.append((s["item_id"], bad[:3]))
            for n in s["subqs"]:
                node_stats[split][(n["type"], fp.plain_truth(n), n["polarity"])] += 1
            steps.append(s)
            if a:
                agents.append(a)
        write(out / f"{split}.jsonl", steps)
        write(out / f"{split}_agent.jsonl", agents)
    for o, rs in tests.items():
        rs = sorted(rs, key=lambda r: r["uid"])
        steps, gts, agents = [], [], []
        for k, r in enumerate(rs):
            iid = f"fa1-{o}-test-{k:05d}"
            s, a = fp.make_items(r, iid, f"fa1/{o}", cap=caps[r["uid"]], with_gt=False,
                                 rng=random.Random(f"fa1|{args.seed}|{r['uid']}|items"))
            bad = fp.audit_item(s)
            if bad:
                fails.append((iid, bad[:3]))
            for n in s["subqs"]:
                node_stats[f"test_{o}"][(n["type"], fp.plain_truth(n), n["polarity"])] += 1
            steps.append(s)
            if r.get("answer"):
                g, _ = fp.make_items(r, iid + "-gt", f"fa1/{o}", cap=caps[r["uid"]], with_gt=True,
                                     rng=random.Random(0), nodes=False)
                g["has_tree"] = False
                gts.append(g)
            if a:
                agents.append(a)
        write(ev / f"fa1_{o}_test.jsonl", steps)
        write(ev / f"fa1_{o}_test_gt.jsonl", gts)
        write(ev / f"fa1_{o}_test_agent.jsonl", agents)
    print("written", files, f"rss {rss_mb()} MB", flush=True)
    # ---- statistics
    summary: dict = {"args": vars(args), "library": "src/cotdistill/fa_public.py", "source": fp.SOURCE,
                     "normalize_counts": json.loads((norm / "counts.json").read_text()), "build_counts":
                     dict(sorted(counts.items())), "dedup": dict(sorted(dedup.items())),
                     "val_groups_extra_trajectories_discarded": dict(val_discard), "files": files, "md5": {},
                     "caps": cap_stats}
    summary["mix_train"] = {o: round(sum(r["origin"] == o for r in train) / len(train), 3) for o in origins}
    summary["mix_train_subset"] = dict(collections.Counter(f"{r['origin']}/{r['subset']}" for r in train).most_common())
    summary["stats"] = {name: describe(rs) for name, rs in allrecs.items()}
    summary["stats_by_origin"] = {name: {o: describe([r for r in rs if r["origin"] == o]) for o in origins}
                                  for name, rs in (("train", train), ("val", val))}
    summary["baselines"] = {name: baselines(rs, train) for name, rs in allrecs.items()}
    summary["baselines_train_by_origin"] = {o: baselines([r for r in train if r["origin"] == o], train)
                                            for o in origins}
    summary["nodes"] = {}
    for name, c in node_stats.items():
        fam = collections.defaultdict(lambda: [0, 0])
        for (t, truth, pol), v in c.items():
            fam[t][0] += v
            fam[t][1] += v * truth
        n_items = len(allrecs[name])
        summary["nodes"][name] = {"per_item": round(sum(c.values()) / max(1, n_items), 2),
                                  "families": {t: {"n": v[0], "plain_yes": round(v[1] / v[0], 3)}
                                               for t, v in sorted(fam.items())},
                                  "stated_yes": round(sum(v for (t, tr, pol), v in c.items()
                                                          if tr == (pol == "plain")) / max(1, sum(c.values())), 3)}
    summary["surface_probe"] = probe(train, {k: v for k, v in allrecs.items() if k != "train"} | {
        f"val/{o}": [r for r in val if r["origin"] == o] for o in origins} | {
        f"test_wwpro/{fw}": [r for r in tests.get("wwpro", []) if r["subset"] == fw]
        for fw in sorted({r["subset"] for r in tests.get("wwpro", [])})}, caps, args.seed)
    summary["tokens"] = token_stats(files, tok)
    summary["audit"] = {"failures": len(fails), "examples": fails[:10]}
    for p in files:
        summary["md5"][p] = md5(ROOT / p)
    summary["rss_mb"] = rss_mb()
    summary["tokenizer"] = tok.method
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("mix_train", "dedup", "audit", "files", "caps")}, indent=1))
    print(f"rss {rss_mb()} MB")
    if fails:
        print("AUDIT FAILED", fails[:5])
        sys.exit(1)


def token_stats(files: dict, tok: Tok) -> dict:
    """Final-prompt token percentiles per file; for the step files with nodes also the GFP-S node view length
    (statement block + one node question) and tokens per epoch for finals only and GFP-S at several --tree-cap.
    Files are streamed in chunks (only the lengths are kept)."""
    from cotdistill.treeplan import statement_block
    out = {"method": tok.method}
    for p in files:
        lf, lv, k_nodes = [], [], []
        finals, views = [], []

        def flush():
            lf.extend(tok.count(finals))
            lv.extend(tok.count(views))
            finals.clear()
            views.clear()
        with open(ROOT / p) as f:
            for line in f:
                it = json.loads(line)
                finals.append(it["prompt"] + "\n\nAnswer:")
                if it.get("subqs"):
                    k_nodes.append(len(it["subqs"]))
                    views.append(statement_block(it) + "\n\nIntermediate question: " + it["subqs"][0]["question"]
                                 + "\nOptions:\nA) Yes\nB) No\nAnswer:")
                if len(finals) >= 128:
                    flush()
        flush()
        e = {"final_tokens": pct(lf), "over_6144": sum(x > 6144 for x in lf), "n": len(lf),
             "tokens_final_only": int(sum(lf))}
        if lv:
            e["node_view_tokens"] = pct(lv)
            e["nodes_per_item"] = pct(k_nodes)
            for cap in (2, 4, 6, 10, 0):
                e[f"tokens_per_epoch_gfps_cap{cap}"] = int(sum(lf) + sum(v * (min(cap, k) if cap else k)
                                                                         for v, k in zip(lv, k_nodes)))
        out[Path(p).name] = e
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("normalize", "build", "all"), default="all",
                    help="all: normalize in a python3 -P -E subprocess when this interpreter has no pandas, then build")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/student_fa1")
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--origins", default=",".join(ORIGINS), help="normalize: which sources to read")
    ap.add_argument("--no-agentracer", dest="agentracer", action="store_false",
                    help="build without AgenTracer (its data release has no licence)")
    ap.add_argument("--wwpro-test-frac", type=float, default=0.10)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--overlap", type=float, default=0.5, help="8-gram overlap ratio that removes a pool trajectory")
    ap.add_argument("--boilerplate-df", type=int, default=5, help="8-grams in this many held-out tasks are ignored")
    ap.add_argument("--caps", default=json.dumps({"wwpro/smolagents": 1200, "wwpro/mathchat": 500,
                                                  "agentracer/agentic": 800, "agentracer/coding": 800}),
                    help="JSON {origin/subset: max train trajectories}")
    ap.add_argument("--share", default=json.dumps({"wwpro": 0.45, "agentracer": 0.40, "stepfinder": 0.35}),
                    help="JSON {origin: max share of train}")
    ap.add_argument("--gt-p", type=float, default=0.5)
    ap.add_argument("--budget", type=int, default=6144, help="Qwen3.5 tokens of a step prompt + '\\n\\nAnswer:'")
    ap.add_argument("--cap-max", type=int, default=3000)
    ap.add_argument("--cap-min", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--dry", action="store_true", help="build: stop after the splits and print their statistics")
    args = ap.parse_args()
    if not fp.precut_ok(args.cap_max):
        raise SystemExit(f"--cap-max {args.cap_max} > PRECUT {fp.PRECUT}")
    if args.stage in ("normalize", "all"):
        try:                                     # only the AFTraj reader needs pandas (python3's user site)
            if "aftraj" in args.origins.split(","):
                import pandas  # noqa: F401
            stage_normalize(args)
        except ImportError:
            if args.stage == "normalize":
                raise
            cmd = ["python3", "-P", "-E", str(Path(__file__).resolve()), "--stage", "normalize", "--raw", args.raw,
                   "--out", args.out, "--origins", args.origins]
            print("normalize in a subprocess:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, cwd=os.environ.get("TMPDIR", "/tmp"))
    if args.stage in ("build", "all"):
        stage_build(args)


if __name__ == "__main__":
    main()
