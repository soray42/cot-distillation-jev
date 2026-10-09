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

Length debias (off by default; off = the e00ab58 outputs byte for byte, tests/test_fa_public.py TestDebias): injected
steps of Who&When Pro, AFTraj and AgenTracer differ in length from the agents' own steps (the injector's text is shorter
in mathchat / macnet / dylan / smolagents, longer in AFTraj), so step length predicts the gold step beyond position.
--debias-length wwpro,aftraj,agentracer removes that tell from the pool after dedup and before val / train are drawn
(debias_length: within-group probes, rejection rounds, groups that cannot be fixed dropped from train and val); test
sets are untouched, so the eval files stay the same (--compare-eval data/eval checks it):

  cd <scratch> && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/sev/bin/python -P -E \\
      ~/projects/jev/scripts/build_fa_public.py --stage build --norm data/student_fa1/norm --out data/student_fa1b \\
      --eval-dir data/student_fa1b/eval_check --compare-eval data/eval --debias-length wwpro,aftraj,agentracer
  -> summary.json debias (per group: probes before / after, rounds, kept, dropped) and length_probe (both modes:
     probes fit on train applied to val / tests, within-train CV, per-group fitted probes)

Cleaning (off by default; off = the e00ab58 / 4a79d36 outputs byte for byte, tests/test_fa_public.py TestClean):
--clean v1 (fa_public.clean_record; label_audit.md and the n-gram probe scripts/fa_ngram_probe.py) runs after dedup and
before the length debias, on the pool and on the test records: injector-produced strings rewritten in every step
(fa_public.TEXT_RULES: the "_attacked_solution" file names, "I am at step N ..." sentences, "system override",
smolagents' replay / live code-fence formats); agentracer / aftraj trajectories that narrate the injection
(fa_public.META) dropped from train / val and from the cleaned test files; --clean-drop-groups (default
fa_public.CLEAN_DROP_GROUPS: AgenTracer coding and math, WW-Pro debate / dylan, AgenTracer agentic Injection Error, AFTraj
agentic injected) dropped from train / val and kept in the test files (AgenTracer coding test records flagged
label_unreliable).
Test files are written with --eval-suffix (default _c under --clean) and kept only where they differ from the
unsuffixed file of --compare-eval (default --eval-dir), so the original eval files are never overwritten and an
unchanged test set keeps its original file (summary.json clean.eval_files says which file to use):

  cd <scratch> && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/sev/bin/python -P -E \\
      ~/projects/jev/scripts/build_fa_public.py --stage build --norm data/student_fa1/norm --out data/student_fa1c \\
      --debias-length wwpro,aftraj,agentracer --clean v1
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
    out = ROOT / args.norm if args.norm else ROOT / args.out / "norm"
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
        elif level == "structure":                      # position + speaker change: surface without the lengths
            f += [r["steps"][i]["agent"] != (r["steps"][i - 1]["agent"] if i else None)]
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


# ------------------------------------------------------------------------------------------------- length debias
LEVELS = ("position", "structure", "surface")
DEBIAS_FOLD_SEEDS = (1, 2)


def debias_group(r: dict) -> str:
    """Selection unit: Who&When Pro framework; AFTraj / AgenTracer domain and provenance (injected / diagnosed, kept in
    the record's benchmark field); else origin/subset."""
    key = f"{r['origin']}/{r['subset']}"
    return key + f"/{r['benchmark']}" if r["origin"] in ("aftraj", "agentracer") else key


def tensor(recs: list[dict], cap, level: str):
    """Padded step features (N, max steps, F) float32, mask, gold index; cap an int or a uid -> cap dict."""
    import numpy as np
    m = max(len(r["steps"]) for r in recs)
    rows = [step_features(r, cap[r["uid"]] if isinstance(cap, dict) else cap, level) for r in recs]
    X = np.zeros((len(recs), m, rows[0].shape[1]), dtype=np.float32)
    mask = np.zeros((len(recs), m), dtype=bool)
    for j, f in enumerate(rows):
        X[j, :len(f)] = f
        mask[j, :len(f)] = True
    return X, mask, np.array([r["dec"] for r in recs])


def clogit_probs(X, mask, w):
    import numpy as np
    z = np.where(mask, X @ w, -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def fit_clogit_v(X, mask, y, iters: int = 30, lam: float = 1e-3):
    """The objective of fit_clogit (mean conditional-logit NLL + lam/2 |w|^2) over padded trajectories, minimised
    by damped Newton steps (exact optimum, about 10x faster than its 300 gradient steps; padding has p = 0)."""
    import numpy as np
    n, f = len(y), X.shape[2]
    rows = np.arange(n)
    Xy = X[rows, y].astype(np.float64)

    def objective(w):
        P = clogit_probs(X, mask, w)
        return -np.log(np.maximum(P[rows, y], 1e-300)).mean() + lam / 2 * w @ w, P

    w = np.zeros(f)
    val, P = objective(w)
    for _ in range(iters):
        Xp = np.einsum("nm,nmf->nf", P, X, dtype=np.float64)
        g = (Xp - Xy).mean(0) + lam * w
        H = np.einsum("nm,nmf,nmg->fg", P, X, X, dtype=np.float64) / n - Xp.T @ Xp / n + lam * np.eye(f)
        step = np.linalg.solve(H, g)
        t = 1.0
        while True:
            w2 = w - t * step
            v2, P2 = objective(w2)
            if v2 <= val - 1e-4 * t * (g @ step) or t < 1e-6:
                break
            t /= 2
        done = val - v2 < 1e-9
        w, val, P = w2, v2, P2
        if done:
            break
    return w


def cv_probe(T: dict, idx, tasks: list[str], fold_seed, folds: int, levels=LEVELS) -> tuple[dict, dict]:
    """Within-group cross-validated probes on the records idx of a group (T: level -> tensor of the whole group);
    folds by task (fa_public.norm_task), so the trajectories of one task never sit in train and test of a fold.
    Returns accuracy per level and the held-out probability of the gold step per level (idx order)."""
    import numpy as np
    fold = np.array([fp.stable_hash("debias-fold", fold_seed, tasks[i]) % folds for i in idx])
    acc, pg = {}, {}
    for lev in levels:
        X, mask, y = (a[idx] for a in T[lev])
        p = np.zeros(len(idx))
        hit = 0
        for k in range(folds):
            te = fold == k
            if not te.any() or te.all():
                continue
            P = clogit_probs(X[te], mask[te], fit_clogit_v(X[~te], mask[~te], y[~te]))
            hit += int((P.argmax(1) == y[te]).sum())
            p[te] = P[np.arange(int(te.sum())), y[te]]
        acc[lev] = hit / max(1, len(idx))
        pg[lev] = p
    return acc, pg


def cv_mean(T, idx, tasks, folds, levels=LEVELS) -> dict:
    """cv_probe accuracy averaged over DEBIAS_FOLD_SEEDS (CV noise of one fold assignment is about +-2 points)."""
    runs = [cv_probe(T, idx, tasks, s, folds, levels)[0] for s in DEBIAS_FOLD_SEEDS]
    return {lev: round(sum(r[lev] for r in runs) / len(runs), 4) for lev in levels}


def length_stats(recs: list[dict], cap: int) -> dict:
    """Where the decisive step sits by length (cut at cap) and position: shortest / longest shares against their
    chance (mean 1/n), mean length-rank quantile of the decisive step (0 shortest .. 1 longest; null .5), last-step
    share, relative-position deciles."""
    if not recs:
        return {"n": 0}
    sh = lo = ch = q = last = 0.0
    for r in recs:
        lens = [min(len(s["text"].strip()), cap) for s in r["steps"]]
        d, n = lens[r["dec"]], len(lens)
        sh += d == min(lens)
        lo += d == max(lens)
        ch += 1 / n
        q += (sum(x < d for x in lens) + 0.5 * (sum(x == d for x in lens) - 1)) / max(1, n - 1)
        last += r["dec"] == n - 1
    k = len(recs)
    return {"n": k, "decisive_shortest": round(sh / k, 3), "decisive_longest": round(lo / k, 3),
            "chance_1_over_n": round(ch / k, 3), "decisive_length_rank_q": round(q / k, 3),
            "decisive_last": round(last / k, 3),
            "decisive_rel_decile": hist(rel_bin(r["dec"], len(r["steps"])) for r in recs)}


def debias_length(pool: list[dict], args) -> tuple[list[dict], dict]:
    """Remove the step-length tell of injected sources from the pool (before val / train are drawn; test sets are
    never touched). Per group (debias_group) of the origins in --debias-length, rendered lengths approximated by
    min(len, --cap-max):
      1. within-group CV (5 folds by task, averaged over 2 fold assignments) of three conditional-logit probes:
         position, structure (position + speaker change), surface (structure + length: log length, length rank,
         longest, shortest). gap = surface - structure is what the lengths add.
      2. gap <= --debias-tau: keep the group. Else rejection rounds towards "length carries no information": with
         cross-fitted held-out probabilities of the gold step under both probes (fit on the current kept set), keep a
         trajectory iff u < min(1, p_structure(gold) / p_surface(gold)) (u a stable hash of the uid and the round).
         One round turns P(decisive | position, speaker, length) towards P(decisive | position, speaker) only where
         a trajectory could have had its decisive step elsewhere; trajectories whose length pattern always marks the
         gold (an injected step much shorter than any other) keep a share of about p_structure each round, so the
         rounds repeat (AFLite-like) until the re-measured gap is <= tau (at most --debias-rounds).
      3. the kept set falls under max(--debias-min-n, --debias-min-share x n) first: the group cannot be fixed and is
         dropped from train and val (it stays in its test file; listed in debias.dropped).
    Groups under --debias-min-n trajectories are kept as they are (too small to measure)."""
    import numpy as np
    origins = {o for o in args.debias_length.split(",") if o}
    by = collections.defaultdict(list)
    for r in pool:
        if r["origin"] in origins:
            by[debias_group(r)].append(r)
    keep: set = set()
    report: dict = {"origins": sorted(origins), "tau": args.debias_tau, "min_share": args.debias_min_share,
                    "min_n": args.debias_min_n, "folds": args.debias_folds, "fold_seeds": list(DEBIAS_FOLD_SEEDS),
                    "rounds": args.debias_rounds, "length_cap": args.cap_max, "groups": {}, "dropped": []}
    for g in sorted(by):
        rs = by[g]
        n = len(rs)
        ent: dict = {"n": n, "before": length_stats(rs, args.cap_max)}
        if n < args.debias_min_n:
            ent.update(status="too_small", kept=n)
            keep |= {(r["origin"], r["uid"]) for r in rs}
            report["groups"][g] = ent
            continue
        T = {lev: tensor(rs, args.cap_max, lev) for lev in LEVELS}
        tasks = [fp.norm_task(r["task"]) for r in rs]
        every = np.arange(n)
        ent["probe_before"] = cv_mean(T, every, tasks, args.debias_folds)
        gap0 = ent["probe_before"]["surface"] - ent["probe_before"]["structure"]
        chosen = None
        if gap0 <= args.debias_tau:
            chosen = (None, every, ent["probe_before"])
        else:
            u_of = lambda r, k: fp.stable_hash(args.seed, "debias", k, r["origin"], r["uid"]) / 16 ** 12  # noqa: E731
            need = max(args.debias_min_n, math.ceil(args.debias_min_share * n))
            idx, trace = every, []
            for k in range(args.debias_rounds):
                _, pg = cv_probe(T, idx, tasks, 0, args.debias_folds, ("structure", "surface"))
                ratio = pg["structure"] / np.maximum(pg["surface"], 1e-9)
                u = np.array([u_of(rs[i], k) for i in idx])
                idx = idx[u < np.minimum(1.0, ratio)]
                if len(idx) < need:
                    trace.append({"round": k + 1, "kept": int(len(idx))})
                    break
                a = cv_mean(T, idx, tasks, args.debias_folds, ("structure", "surface"))
                trace.append({"round": k + 1, "kept": int(len(idx)), "gap": round(a["surface"] - a["structure"], 4)})
                if a["surface"] - a["structure"] <= args.debias_tau:
                    chosen = (k + 1, idx, None)
                    break
            ent["trace"] = trace
        if chosen is None:
            ent.update(status="dropped", kept=0)
            report["dropped"].append(g)
        else:
            rounds, idx, after = chosen
            sel = [rs[i] for i in idx]
            ent.update(status="kept_all" if rounds is None else "selected", rounds=rounds, kept=len(sel),
                       kept_share=round(len(sel) / n, 3),
                       probe_after=after or cv_mean(T, idx, tasks, args.debias_folds),
                       after=length_stats(sel, args.cap_max))
            keep |= {(r["origin"], r["uid"]) for r in sel}
        del T
        report["groups"][g] = ent
        print("debias", g, {k: v for k, v in ent.items() if k in ("n", "status", "rounds", "kept", "probe_before",
                                                                 "probe_after")}, f"rss {rss_mb()} MB", flush=True)
    out = [r for r in pool if r["origin"] not in origins or (r["origin"], r["uid"]) in keep]
    report["pool_before"], report["pool_after"] = len(pool), len(out)
    return out, report


def probe_report(train: list[dict], evals: dict, caps: dict, folds: int) -> dict:
    """Probes at the rendered lengths (the fitted per-trajectory caps), three levels (position, structure,
    surface): (a) fit on all of train (fit_clogit_v), accuracy on every eval set (val, tests and their groups);
    (b) within-train CV per debias_group (folds by task, one fold assignment)."""
    import numpy as np
    out: dict = {"fit_on_train": {}, "train_cv": {}}
    T = {lev: tensor(train, caps, lev) for lev in LEVELS}
    w = {lev: fit_clogit_v(*T[lev]) for lev in LEVELS}
    del T
    for name, rs in evals.items():
        if not rs:
            continue
        e = {"n": len(rs)}
        for lev in LEVELS:
            X, mask, y = tensor(rs, caps, lev)
            e[lev] = round(float((clogit_probs(X, mask, w[lev]).argmax(1) == y).mean()), 3)
        out["fit_on_train"][name] = e
    by = collections.defaultdict(list)
    for r in train:
        by[debias_group(r)].append(r)
    for g, rs in sorted(by.items()):
        if len(rs) < 2 * folds:
            continue
        T = {lev: tensor(rs, caps, lev) for lev in LEVELS}
        acc, _ = cv_probe(T, np.arange(len(rs)), [fp.norm_task(r["task"]) for r in rs], 0, folds)
        out["train_cv"][g] = {"n": len(rs)} | {k: round(v, 3) for k, v in acc.items()}
    # (c) per group: probes fit on the group's train trajectories, applied to the group's trajectories of every
    # top-level eval set (a model reads the framework off the log, so its fair position prior is per group); pooled
    # per origin over the groups that have train data (groups without train data, e.g. dropped ones, are listed)
    hits: dict = collections.defaultdict(lambda: collections.Counter())
    out["fit_by_group"] = {}
    for g, rs in sorted(by.items()):
        if len(rs) < 2 * folds:
            continue
        w = {lev: fit_clogit_v(*tensor(rs, caps, lev)) for lev in LEVELS}
        for name, es in evals.items():
            sel = [r for r in es if "/" not in name and debias_group(r) == g]
            if not sel:
                continue
            e = {"n": len(sel)}
            for lev in LEVELS:
                X, mask, y = tensor(sel, caps, lev)
                e[lev] = int((clogit_probs(X, mask, w[lev]).argmax(1) == y).sum())
                hits[f"{name}/{g.split('/')[0]}"][lev] += e[lev]
            hits[f"{name}/{g.split('/')[0]}"]["n"] += len(sel)
            out["fit_by_group"][f"{name}/{g}"] = {"n": len(sel)} | {lev: round(e[lev] / len(sel), 3) for lev in LEVELS}
    out["fit_by_group_pooled"] = {k: {"n": c["n"]} | {lev: round(c[lev] / c["n"], 3) for lev in LEVELS}
                                  for k, c in sorted(hits.items())}
    out["fit_by_group_uncovered"] = {name: dict(collections.Counter(
        debias_group(r) for r in es if debias_group(r) not in by or len(by[debias_group(r)]) < 2 * folds))
        for name, es in evals.items() if "/" not in name}
    return out


def clean_sets(pool: list[dict], tests: dict, args) -> tuple[list[dict], dict, dict]:
    """--clean: fa_public.clean_record on every pool and test record (after dedup, before the length debias). Pool
    records with a drop reason leave the pool; test records leave their test set only for "meta_leak" (group drops
    are train / val only). Returns the pool, the test sets and the report (rules, per-group counts, dropped groups for
    fa_readout's dagger, test uids dropped or kept although their group left train)."""
    if args.clean_drop_groups is None:
        drop = dict(fp.CLEAN_DROP_GROUPS)
    else:
        drop = {g: fp.CLEAN_DROP_GROUPS.get(g, "--clean-drop-groups") for g in args.clean_drop_groups.split(",") if g}
    rep: dict = {"version": args.clean, "text_rules": [
        {"name": n, "scope": list(sc), "pattern": pat.pattern, "replacement": rp if isinstance(rp, str) else
         "instruction(s)"} for n, sc, pat, rp in fp.TEXT_RULES],
        "meta_regex": fp.META.pattern, "meta_origins": list(fp.META_ORIGINS), "drop_groups": drop,
        "label_flags": fp.LABEL_FLAGS, "pool": {}, "test": {}, "substitutions": {}}
    subs: dict = collections.defaultdict(collections.Counter)
    per: dict = collections.defaultdict(collections.Counter)
    out = []
    for r in pool:
        c: collections.Counter = collections.Counter()
        r2, why = fp.clean_record(r, args.clean, drop, c)
        g = fp.clean_group(r)
        subs[f"pool/{g}"].update(c)
        per[g]["n"] += 1
        if why:
            per[g][why.split(":")[0]] += 1
        else:
            per[g]["kept"] += 1
            out.append(r2)
    rep["pool"] = {g: dict(c) for g, c in sorted(per.items())}
    rep["dropped"] = sorted(g for g, c in per.items() if not c["kept"])
    new_tests: dict = {}
    for o, rs in tests.items():
        keep, info = [], collections.defaultdict(list)
        for r in rs:
            c = collections.Counter()
            r2, why = fp.clean_record(r, args.clean, drop, c)
            g = fp.clean_group(r)
            subs[f"test_{o}/{g}"].update(c)
            if why == "meta_leak":
                info["dropped_meta_leak"].append(r["uid"])
                continue
            if why:
                info["kept_group_dropped_from_train"].append(r["uid"])
            if r2["flags"] != r["flags"]:
                info["flagged_" + "_".join(sorted(set(r2["flags"]) - set(r["flags"])))].append(r["uid"])
            keep.append(r2)
        new_tests[o] = keep
        rep["test"][o] = {"n_before": len(rs), "n_after": len(keep)} | {k: {"n": len(v), "uids": sorted(v)}
                                                                         for k, v in sorted(info.items())}
    rep["substitutions"] = {k: dict(v) for k, v in sorted(subs.items()) if v}
    print("clean", {g: dict(c) for g, c in sorted(per.items())}, f"rss {rss_mb()} MB", flush=True)
    return out, new_tests, rep


def stage_build(args) -> None:
    norm = ROOT / args.norm if args.norm else ROOT / args.out / "norm"
    out, ev = ROOT / args.out, ROOT / args.eval_dir
    ev.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
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
    test_order = {o: {u: k for k, u in enumerate(sorted(r["uid"] for r in rs))} for o, rs in tests.items()}
    clean = None
    if args.clean:                                      # off: nothing below runs and nothing is consumed
        pool, tests, clean = clean_sets(pool, tests, args)
        counts["pool/after_clean"] = len(pool)
    debias = None
    if args.debias_length:                              # off: nothing below runs and nothing is consumed
        pool, debias = debias_length(pool, args)
        counts["pool/after_debias"] = len(pool)
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
    del recs, pool, per, rest, by_k, keep, idx            # records no split uses (peak RSS); outputs unaffected
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
    sfx = args.eval_suffix
    for o, rs in tests.items():
        rs = sorted(rs, key=lambda r: r["uid"])
        steps, gts, agents = [], [], []
        for r in rs:
            iid = f"fa1-{o}-test-{test_order[o][r['uid']]:05d}"          # the id of the uncleaned set
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
        write(ev / f"fa1_{o}_test{sfx}.jsonl", steps)
        write(ev / f"fa1_{o}_test_gt{sfx}.jsonl", gts)
        write(ev / f"fa1_{o}_test_agent{sfx}.jsonl", agents)
    if sfx:                       # suffixed eval files are kept only where they differ from the unsuffixed reference
        ref_dir = ROOT / (args.compare_eval or args.eval_dir)
        decided = {}
        for p in [p for p in files if (ROOT / p).parent.resolve() == ev.resolve()]:
            ref = ref_dir / (Path(p).name[:-len(sfx + ".jsonl")] + ".jsonl")
            if ref.exists() and md5(ROOT / p) == md5(ref):
                (ROOT / p).unlink()
                del files[p]
                decided[Path(p).name] = f"unchanged: use {ref.relative_to(ROOT) if ref.is_relative_to(ROOT) else ref}"
            else:
                decided[Path(p).name] = "written" if ref.exists() else "written (no reference file)"
        if clean is not None:
            clean["eval_files"] = decided
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
    sets = {"val": val} | {name: rs for name, rs in allrecs.items() if name.startswith("test_")}
    summary["length_probe"] = probe_report(train, sets | {
        f"{name}/{g}": [r for r in rs if debias_group(r) == g] for name, rs in sets.items()
        for g in sorted({debias_group(r) for r in rs})}, caps, args.debias_folds)
    if debias is not None:
        summary["debias"] = debias
    if clean is not None:
        summary["clean"] = clean
    if args.compare_eval and not sfx:
        ref = ROOT / args.compare_eval
        summary["eval_vs_compare"] = {"dir": args.compare_eval, "same_md5": {
            Path(p).name: (md5(ROOT / p) == md5(ref / Path(p).name)) if (ref / Path(p).name).exists() else None
            for p in files if (ROOT / p).parent.resolve() == ev.resolve()}}
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
    ap.add_argument("--norm", default=None, help="normalized records: written by normalize, read by build "
                                                 "(default <out>/norm)")
    ap.add_argument("--debias-length", default="",
                    help="comma list of origins whose step-length tell is removed from the pool by within-group "
                         "rejection sampling, unfixable groups dropped from train / val (debias_length); "
                         "off by default, e.g. wwpro,aftraj,agentracer")
    ap.add_argument("--debias-tau", type=float, default=0.01, help="largest accepted CV gap surface - structure")
    ap.add_argument("--debias-min-share", type=float, default=1 / 3, help="smaller kept share: group dropped")
    ap.add_argument("--debias-min-n", type=int, default=40, help="smaller kept set: group dropped; smaller group: kept")
    ap.add_argument("--debias-folds", type=int, default=5)
    ap.add_argument("--debias-rounds", type=int, default=8, help="most rejection rounds before a group is dropped")
    ap.add_argument("--compare-eval", default=None,
                    help="record whether every eval file written equals the same-named file of this dir (md5); with "
                         "--eval-suffix the reference dir of the suffixed files")
    ap.add_argument("--clean", default="", choices=("", "v1"),
                    help="label / artefact cleaning (fa_public.clean_record; off by default): v1")
    ap.add_argument("--clean-drop-groups", default=None,
                    help="comma list of groups (origin/subset[/benchmark]) dropped from train / val under --clean "
                         "(default fa_public.CLEAN_DROP_GROUPS)")
    ap.add_argument("--eval-suffix", default=None,
                    help="suffix of the test files written (default '' and, under --clean, _c); suffixed files equal "
                         "to the unsuffixed reference (--compare-eval, else --eval-dir) are removed again")
    args = ap.parse_args()
    if args.eval_suffix is None:
        args.eval_suffix = "_c" if args.clean else ""
    if args.clean and not args.eval_suffix:
        raise SystemExit("--clean writes cleaned test files: give a non-empty --eval-suffix (the originals stay)")
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
                   "--out", args.out, "--origins", args.origins] + (["--norm", args.norm] if args.norm else [])
            print("normalize in a subprocess:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, cwd=os.environ.get("TMPDIR", "/tmp"))
    if args.stage in ("build", "all"):
        stage_build(args)


if __name__ == "__main__":
    main()
