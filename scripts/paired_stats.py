#!/usr/bin/env python3
"""Paired seed x item statistics for Tree Forcing comparisons: arm A minus arm B (the baseline).

Every run is a directory <root>/<run>/ holding preds_<set>.jsonl (item_id, labels, probs, gold_label per line)
and metrics.json (args.seed, args.epochs). An arm is one or more run-dir patterns with a {seed} placeholder
(optionally {ep} for an epoch variant and the glob wildcards * ?), or an explicit list of run dirs.

Design (all differences are A - B, in accuracy points):
  * Paired seeds only: a seed enters when both arms have it (minus --exclude-seeds, e.g. screening seeds).
  * Items: the test sets are concatenated (default gsm8kv2+gsm1k) and keyed by "<set>:<item_id>". Every run
    in the comparison must hold exactly the same keys with the same gold label (error otherwise).
  * Per-seed difference d_s = acc_A,s - acc_B,s, its mean, and the seed-level paired Student-t interval
    (df = k - 1, two-sided p; none when k = 1).
  * Item-cluster bootstrap: resample test items once per iteration with the SAME indices for every seed and
    both arms; statistic = mean over drawn items of the per-item seed-mean difference.
  * Two-level bootstrap (primary): resample seeds, then items (shared indices); statistic = mean difference.
  * "significant" only when the two-level interval AND the seed-t interval both exclude 0.
  * Extras: top-1 ECE (15 equal-width bins, cotdistill.metrics.calibration) per arm on the test sets, accuracy
    by depth group (gs2+gs3+gs4 vs gs5+gs6+gs7+gs8, when the files exist; gs5p is their union and is not
    used), and any other set group passed with --sets.
  * --select-by val: for an arm whose pattern has {ep}, pick the epoch variant with the highest seed-mean
    val accuracy (preds_val.jsonl only; ties -> fewer epochs), over the seeds that could enter the paired
    comparison. Test sets are never read during selection.

Pairwise:
  python scripts/paired_stats.py --a 'GFPQ-c{ep}lr3ch-Qwen3.5-4B-student_gsm8k_tf-s{seed}' --a-epochs 1,2 \
      --b 'A2-c1lr3ch-Qwen3.5-4B-student_gsm8k_tf-s{seed}' --select-by val
Table (one baseline per table, arms as rows; YAML or JSON spec, see TABLE_SPEC_HELP):
  python scripts/paired_stats.py --table spec.yaml --out notes/table.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.metrics import calibration  # noqa: E402

DEFAULT_ROOT = "results/chtc_repo/runs"
DEFAULT_TEST = "gsm8kv2+gsm1k"
DEPTH_GROUPS = (("gs2-4", "gs2+gs3+gs4"), ("gs5-8", "gs5+gs6+gs7+gs8"))
VAL_SET = "val"

TABLE_SPEC_HELP = """\
root: results/chtc_repo/runs        # optional
test: gsm8kv2+gsm1k                 # optional
sets: [bbh500]                      # optional extra set groups ("name=a+b" or "a+b")
depth: true                         # optional, depth groups when files exist
select_by: val                      # optional: none | val
exclude_seeds: []                   # optional, applies to every row
boot: 10000                         # optional
tables:
  - title: Qwen3.5-4B GSM8K vs A2 c1
    baseline: {name: A2 c1, pattern: "A2-c1lr3ch-Qwen3.5-4B-student_gsm8k_tf-s{seed}"}
    arms:
      - {name: GFPQ ch, pattern: "GFPQ-c{ep}lr3ch-Qwen3.5-4B-student_gsm8k_tf-s{seed}", epochs: [1, 2]}
      - {name: LS excl s12, pattern: "LS-sc2-c2lr3ch-Qwen3.5-4B-student_gsm8k_tf_si-s{seed}", exclude_seeds: [12]}
"""


def log(msg: str = "") -> None:
    print(msg, file=sys.stderr)


# ----------------------------------------------------------------------------------------------------------
# Student t without scipy
# ----------------------------------------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta (modified Lentz)."""
    tiny, eps = 1e-300, 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """P(|T| >= |t|) for Student t with df degrees of freedom."""
    if math.isinf(t):
        return 0.0
    return betainc_reg(df / 2.0, 0.5, df / (df + t * t))


def t_ppf(q: float, df: float) -> float:
    """Quantile of Student t (q in (0.5, 1)); bisection on the two-sided tail."""
    if not 0.5 < q < 1.0:
        raise ValueError("t_ppf needs 0.5 < q < 1")
    target = 2.0 * (1.0 - q)
    lo, hi = 0.0, 1.0
    while t_two_sided_p(hi, df) > target:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_two_sided_p(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def seed_t(d: list[float] | np.ndarray, alpha: float = 0.05) -> dict:
    """Paired Student-t on per-seed differences (df = k - 1). k = 1: no interval."""
    d = np.asarray(d, dtype=float)
    k = len(d)
    out = {"k": k, "mean": float(d.mean()) if k else float("nan"), "df": k - 1,
           "lo": None, "hi": None, "p": None, "sd": None, "t": None, "note": ""}
    if k < 2:
        out["note"] = "k=1: no seed-level interval (needs >= 2 paired seeds)"
        return out
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(k)
    tc = t_ppf(1.0 - alpha / 2.0, k - 1)
    out.update(sd=sd, lo=out["mean"] - tc * se, hi=out["mean"] + tc * se, tcrit=tc)
    if se == 0.0:
        out.update(t=float("inf") if out["mean"] else 0.0, p=0.0 if out["mean"] else 1.0,
                   note="zero spread across seeds")
    else:
        out["t"] = out["mean"] / se
        out["p"] = t_two_sided_p(out["t"], k - 1)
    return out


# ----------------------------------------------------------------------------------------------------------
# Bootstraps over a (seeds x items) difference matrix D = correct_A - correct_B
# ----------------------------------------------------------------------------------------------------------

def _pct(stats: np.ndarray, alpha: float) -> tuple[float, float]:
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def item_bootstrap(D: np.ndarray, n_boot: int, rng: np.random.Generator, alpha: float = 0.05,
                   independent: bool = False, chunk: int = 256) -> tuple[float, float, np.ndarray]:
    """Item-cluster bootstrap. Shared (default): one draw of item indices per iteration, used for every seed,
    so the statistic is the mean over drawn items of the per-item seed-mean difference. independent=True draws
    separate indices per seed (ignores the cross-seed correlation of items; kept for comparison/tests)."""
    k, n = D.shape
    stats = np.empty(n_boot)
    delta = D.mean(axis=0)
    rows = np.arange(k)[None, :, None]
    for s in range(0, n_boot, chunk):
        b = min(chunk, n_boot - s)
        if independent:
            idx = rng.integers(0, n, size=(b, k, n))
            stats[s:s + b] = D[rows, idx].mean(axis=(1, 2))
        else:
            idx = rng.integers(0, n, size=(b, n))
            stats[s:s + b] = delta[idx].mean(axis=1)
    lo, hi = _pct(stats, alpha)
    return lo, hi, stats


def two_level_bootstrap(D: np.ndarray, n_boot: int, rng: np.random.Generator, alpha: float = 0.05,
                        chunk: int = 256) -> tuple[float, float, np.ndarray]:
    """Resample seeds (with replacement), then items with indices shared by all drawn seeds."""
    k, n = D.shape
    stats = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        b = min(chunk, n_boot - s)
        iidx = rng.integers(0, n, size=(b, n))
        sidx = rng.integers(0, k, size=(b, k))
        per_seed = D[:, iidx].mean(axis=2).T          # (b, k): item-resampled mean difference per seed
        stats[s:s + b] = np.take_along_axis(per_seed, sidx, axis=1).mean(axis=1)
    lo, hi = _pct(stats, alpha)
    return lo, hi, stats


def excludes_zero(lo, hi) -> bool:
    return lo is not None and hi is not None and (lo > 0 or hi < 0)


def paired_stats(cA: np.ndarray, cB: np.ndarray, n_boot: int = 10000, boot_seed: int = 0,
                 alpha: float = 0.05) -> dict:
    """cA, cB: (k seeds, n items) 0/1 correctness, rows aligned by seed, columns by item."""
    cA = np.asarray(cA, dtype=np.int8)
    cB = np.asarray(cB, dtype=np.int8)
    if cA.shape != cB.shape or cA.ndim != 2 or cA.shape[0] < 1 or cA.shape[1] < 1:
        raise ValueError(f"need two equal (k, n) matrices, got {cA.shape} and {cB.shape}")
    D = (cA - cB).astype(np.int8)
    accA, accB = cA.mean(axis=1), cB.mean(axis=1)
    d = accA - accB
    t = seed_t(d, alpha)
    rng = np.random.default_rng(boot_seed)
    ilo, ihi, _ = item_bootstrap(D, n_boot, rng, alpha)
    rng = np.random.default_rng(boot_seed + 1)
    tlo, thi, _ = two_level_bootstrap(D, n_boot, rng, alpha)
    sig = excludes_zero(tlo, thi) and excludes_zero(t["lo"], t["hi"])
    return {"k": int(cA.shape[0]), "n_items": int(cA.shape[1]), "acc_a": accA.tolist(), "acc_b": accB.tolist(),
            "d": d.tolist(), "mean": float(d.mean()), "t": t, "item_boot": [ilo, ihi],
            "two_level": [tlo, thi], "significant": bool(sig), "n_boot": n_boot, "alpha": alpha}


# ----------------------------------------------------------------------------------------------------------
# Runs, patterns, predictions
# ----------------------------------------------------------------------------------------------------------

@dataclass
class Run:
    name: str
    path: Path
    seed: int
    epochs: float | None


@dataclass
class ArmSpec:
    name: str
    patterns: list[str]
    epochs: list[float] | None = None
    exclude_seeds: list[int] = field(default_factory=list)


def read_preds(path: Path) -> list[dict]:
    """The one reader for preds_<set>.jsonl files (tests hook it to audit which files a step opens)."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_args(run_dir: Path) -> dict | None:
    p = run_dir / "metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f).get("args", {})


def fmt_ep(ep) -> str:
    ep = float(ep)
    return str(int(ep)) if ep == int(ep) else f"{ep:g}"


def pattern_regex(pat: str, ep=None) -> re.Pattern:
    out = []
    seen = set()
    for tok in re.split(r"(\{seed\}|\{ep\}|\*|\?)", pat):
        if tok in ("{seed}", "{ep}"):
            if tok in seen:
                raise ValueError(f"{tok} appears twice in {pat!r}")
            seen.add(tok)
        if tok == "{seed}":
            out.append(r"(?P<seed>\d+)")
        elif tok == "{ep}":
            out.append(re.escape(fmt_ep(ep)) if ep is not None else r"(?P<ep>\d+(?:\.\d+)?)")
        elif tok == "*":
            out.append(r"[^/]*")
        elif tok == "?":
            out.append(r"[^/]")
        else:
            out.append(re.escape(tok))
    return re.compile("^" + "".join(out) + "$")


def _listdir(root: Path) -> list[str]:
    return sorted(e for e in os.listdir(root) if (root / e).is_dir())


def discover_epochs(arm: ArmSpec, root: Path) -> list[float]:
    eps = set()
    names = _listdir(root)
    for pat in arm.patterns:
        if "{ep}" not in pat:
            continue
        rx = pattern_regex(pat)
        for n in names:
            m = rx.match(n)
            if m:
                eps.add(float(m.group("ep")))
    return sorted(eps)


def resolve_variant(arm: ArmSpec, root: Path, ep=None) -> dict[int, Run]:
    """seed -> Run for one epoch variant (ep=None when the patterns have no {ep})."""
    runs: dict[int, Run] = {}
    names = None
    for pat in arm.patterns:
        if "{seed}" in pat or "*" in pat or "?" in pat:
            names = names if names is not None else _listdir(root)
            rx = pattern_regex(pat, ep)
            hits = [(n, rx.match(n)) for n in names]
            hits = [(n, m) for n, m in hits if m]
        else:                                                    # explicit run dir (name under root or path)
            p = Path(pat) if os.path.isabs(pat) else root / pat
            if not p.is_dir():
                raise FileNotFoundError(f"[{arm.name}] run dir not found: {p}")
            hits = [(str(p), None)]
        for n, m in hits:
            path = Path(n) if os.path.isabs(n) else root / n
            args = read_args(path)
            seed = int(m.group("seed")) if m is not None and "seed" in m.groupdict() else None
            if args is not None and args.get("seed") is not None:
                if seed is not None and int(args["seed"]) != seed:
                    raise ValueError(f"[{arm.name}] {path.name}: name says seed {seed}, metrics.json says {args['seed']}")
                seed = int(args["seed"])
            if seed is None:
                raise ValueError(f"[{arm.name}] {path.name}: no seed in the name pattern or metrics.json")
            epochs = float(args["epochs"]) if args and args.get("epochs") is not None else None
            if ep is not None and epochs is not None and abs(epochs - float(ep)) > 1e-9:
                raise ValueError(f"[{arm.name}] {path.name}: pattern says {fmt_ep(ep)} epochs, metrics.json says {epochs}")
            if args is None:
                log(f"  warning: [{arm.name}] {path.name} has no metrics.json (seed from the name)")
            if seed in runs and runs[seed].path != path:
                raise ValueError(f"[{arm.name}] seed {seed} matched twice: {runs[seed].path.name} and {path.name}")
            runs[seed] = Run(path.name, path, seed, epochs)
    return runs


def resolve_arm(arm: ArmSpec, root: Path) -> dict:
    """{ep or None: {seed: Run}} with arm-level excluded seeds removed."""
    has_ep = any("{ep}" in p for p in arm.patterns)
    if has_ep and not all("{ep}" in p for p in arm.patterns):
        raise ValueError(f"[{arm.name}] mix of patterns with and without {{ep}}")
    eps = (arm.epochs if arm.epochs else discover_epochs(arm, root)) if has_ep else [None]
    if has_ep and not eps:
        raise ValueError(f"[{arm.name}] no epoch variant found for {arm.patterns}")
    out = {}
    for ep in eps:
        runs = resolve_variant(arm, root, ep)
        runs = {s: r for s, r in runs.items() if s not in set(arm.exclude_seeds)}
        if runs:
            out[ep] = runs
    if not out:
        raise ValueError(f"[{arm.name}] no runs matched {arm.patterns} under {root}")
    return out


def top1(row: dict) -> tuple[int, float, int]:
    """(correct, top-option probability, gold index) for one preds row; missing probs count as wrong."""
    labels, probs, gold = row["labels"], row.get("probs"), row["gold_label"]
    gi = labels.index(gold) if gold in labels else -1
    if not probs:
        return 0, float("nan"), gi
    k = max(range(len(probs)), key=probs.__getitem__)          # first max, as cotdistill.metrics
    return int(labels[k] == gold), float(probs[k]), gi


def val_accuracy(run: Run) -> float:
    rows = read_preds(run.path / f"preds_{VAL_SET}.jsonl")
    return sum(top1(r)[0] for r in rows) / len(rows)


def select_epoch(arm_name: str, variants: dict, allowed_seeds: set[int] | None, exclude: set[int],
                 out: list[str] | None = None) -> tuple:
    """Pick the epoch variant with the highest seed-mean val accuracy (ties -> fewer epochs).

    The mean for each variant runs over its seeds that could enter the paired comparison
    (variant seeds & allowed_seeds - exclude). Reads preds_val.jsonl only."""
    say = (lambda m: out.append(m)) if out is not None else log
    if list(variants) == [None]:
        return None, {}
    info = {}
    for ep in sorted(variants):
        seeds = sorted(s for s in variants[ep] if s not in exclude and (allowed_seeds is None or s in allowed_seeds))
        if not seeds:
            continue
        accs = {s: val_accuracy(variants[ep][s]) for s in seeds}
        info[ep] = {"seeds": seeds, "val": accs, "mean": float(np.mean(list(accs.values())))}
    if not info:
        raise ValueError(f"[{arm_name}] no epoch variant has a usable seed")
    best = max(info.values(), key=lambda v: v["mean"])["mean"]
    chosen = min(ep for ep, v in info.items() if v["mean"] >= best - 1e-12)
    parts = [f"ep{fmt_ep(ep)}: val {100 * v['mean']:.2f} (s{','.join(map(str, v['seeds']))})" for ep, v in info.items()]
    say(f"[{arm_name}] select-by val -> ep{fmt_ep(chosen)}   " + "; ".join(parts))
    seed_sets = {tuple(v["seeds"]) for v in info.values()}
    if len(seed_sets) > 1:
        common = sorted(set.intersection(*(set(v["seeds"]) for v in info.values())))
        diag = "; ".join(f"ep{fmt_ep(ep)} {100 * np.mean([v['val'][s] for s in common]):.2f}" for ep, v in info.items()) \
            if common else "no common seed"
        say(f"[{arm_name}]   warning: candidates have different seed sets; on common seeds "
            f"s{','.join(map(str, common))}: {diag}")
    return chosen, info


def parse_group(spec: str) -> tuple[str, list[str]]:
    name, _, body = spec.partition("=") if "=" in spec else (spec, "", spec)
    sets = [s.strip() for s in body.split("+") if s.strip()]
    if not sets:
        raise ValueError(f"empty set group {spec!r}")
    return name.strip(), sets


def load_matrix(runs: list[Run], sets: list[str]) -> dict:
    """Aligned (runs x items) correctness / confidence / gold over the concatenated sets."""
    per_run = []
    for r in runs:
        rows = {}
        for s in sets:
            p = r.path / f"preds_{s}.jsonl"
            if not p.exists():
                raise FileNotFoundError(f"{r.name}: missing preds_{s}.jsonl")
            for row in read_preds(p):
                key = f"{s}:{row['item_id']}"
                if key in rows:
                    raise ValueError(f"{r.name}: duplicate item {key}")
                rows[key] = row
        per_run.append(rows)
    keys = sorted(per_run[0])
    ref = set(keys)
    for r, rows in zip(runs, per_run):
        if set(rows) != ref:
            miss, extra = len(ref - set(rows)), len(set(rows) - ref)
            raise ValueError(f"item sets differ for {'+'.join(sets)}: {r.name} lacks {miss} / has {extra} extra "
                             f"items vs {runs[0].name}")
    k, n = len(runs), len(keys)
    correct = np.zeros((k, n), dtype=np.int8)
    ece = []
    gold0 = [(per_run[0][key]["gold_label"], tuple(per_run[0][key]["labels"])) for key in keys]
    n_missing = 0
    for j, rows in enumerate(per_run):
        probs, gidx = [], []
        for i, key in enumerate(keys):
            row = rows[key]
            if (row["gold_label"], tuple(row["labels"])) != gold0[i]:
                raise ValueError(f"{runs[j].name}: gold/labels of {key} differ from {runs[0].name}")
            c, _, gi = top1(row)
            correct[j, i] = c
            if row.get("probs") and gi >= 0:
                probs.append(row["probs"])
                gidx.append(gi)
            else:
                n_missing += 1
        ece.append(calibration(probs, gidx, bins=15)["ece"] if probs else float("nan"))
    if n_missing:
        log(f"  warning: {n_missing} rows without probs or with gold outside labels (counted wrong, not in ECE)")
    return {"keys": keys, "correct": correct, "ece": ece}


# ----------------------------------------------------------------------------------------------------------
# One comparison
# ----------------------------------------------------------------------------------------------------------

@dataclass
class Options:
    root: Path
    test: str = DEFAULT_TEST
    sets: list[str] = field(default_factory=list)
    depth: bool = True
    select_by: str = "none"
    exclude_seeds: list[int] = field(default_factory=list)
    boot: int = 10000
    boot_seed: int = 0
    alpha: float = 0.05


def pick(arm: ArmSpec, variants: dict, other_seeds: set[int] | None, opt: Options, notes: list[str]):
    exclude = set(opt.exclude_seeds) | set(arm.exclude_seeds)
    if list(variants) == [None]:
        return None, variants[None], {}
    if len(variants) == 1:
        ep = next(iter(variants))
        notes.append(f"[{arm.name}] single epoch variant ep{fmt_ep(ep)}")
        return ep, variants[ep], {}
    if opt.select_by != "val":
        raise ValueError(f"[{arm.name}] {len(variants)} epoch variants ({', '.join(fmt_ep(e) for e in variants)}): "
                         f"pass one epoch or --select-by val")
    ep, info = select_epoch(arm.name, variants, other_seeds, exclude, notes)
    return ep, variants[ep], info


def compare(a: ArmSpec, b: ArmSpec, opt: Options, b_resolved: tuple | None = None) -> dict:
    notes: list[str] = []
    va = resolve_arm(a, opt.root)
    if b_resolved is None:
        vb = resolve_arm(b, opt.root)
        seeds_a_all = set().union(*(set(v) for v in va.values()))
        ep_b, runs_b, info_b = pick(b, vb, seeds_a_all, opt, notes)
    else:
        ep_b, runs_b, info_b = b_resolved
    ep_a, runs_a, info_a = pick(a, va, set(runs_b), opt, notes)
    exclude = set(opt.exclude_seeds) | set(a.exclude_seeds) | set(b.exclude_seeds)
    seeds = sorted((set(runs_a) & set(runs_b)) - exclude)
    dropped = {"a_only": sorted(set(runs_a) - set(runs_b) - exclude), "b_only": sorted(set(runs_b) - set(runs_a) - exclude),
               "excluded": sorted((set(runs_a) | set(runs_b)) & exclude)}
    res = {"a": a.name, "b": b.name, "ep_a": ep_a, "ep_b": ep_b, "select_a": info_a, "select_b": info_b,
           "seeds": seeds, "dropped": dropped, "notes": notes,
           "runs_a": [runs_a[s].name for s in seeds], "runs_b": [runs_b[s].name for s in seeds]}
    if not seeds:
        res["error"] = "no paired seed"
        return res
    ra, rb = [runs_a[s] for s in seeds], [runs_b[s] for s in seeds]
    _, test_sets = parse_group(opt.test)
    m = load_matrix(ra + rb, test_sets)
    k = len(seeds)
    st = paired_stats(m["correct"][:k], m["correct"][k:], opt.boot, opt.boot_seed, opt.alpha)
    st.update(test=opt.test, ece_a=m["ece"][:k], ece_b=m["ece"][k:])
    res["main"] = st
    groups = []
    if opt.depth:
        groups += [(n, s) for n, s in DEPTH_GROUPS]
    groups += [parse_group(g) for g in opt.sets]
    extras = {}
    for name, sets in groups:
        sets = sets if isinstance(sets, list) else parse_group(sets)[1]
        if not all((r.path / f"preds_{s}.jsonl").exists() for r in ra + rb for s in sets):
            extras[name] = {"missing": True, "sets": sets}
            continue
        mx = load_matrix(ra + rb, sets)
        sx = paired_stats(mx["correct"][:k], mx["correct"][k:], opt.boot, opt.boot_seed, opt.alpha)
        sx.update(sets=sets, ece_a=mx["ece"][:k], ece_b=mx["ece"][k:])
        extras[name] = sx
    res["extras"] = extras
    return res


# ----------------------------------------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------------------------------------

def pp(x, signed: bool = True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:+.2f}" if signed else f"{100 * x:.2f}"


def ci(lo, hi) -> str:
    return "n/a" if lo is None else f"[{pp(lo)}, {pp(hi)}]"


def fmt_p(p) -> str:
    return "n/a" if p is None else (f"{p:.3f}" if p >= 0.001 else f"{p:.1e}")


def text_report(res: dict) -> str:
    L = [f"== {res['a']}  minus  {res['b']}   (accuracy points; positive = A better)"]
    for n in res["notes"]:
        L.append("  " + n)
    if res.get("ep_a") is not None:
        L.append(f"  A epochs: {fmt_ep(res['ep_a'])}")
    if res.get("ep_b") is not None:
        L.append(f"  B epochs: {fmt_ep(res['ep_b'])}")
    dr = res["dropped"]
    L.append(f"  paired seeds: {res['seeds']}   (A only: {dr['a_only']}, B only: {dr['b_only']}, excluded: {dr['excluded']})")
    if "error" in res:
        L.append("  " + res["error"])
        return "\n".join(L)
    st = res["main"]
    t = st["t"]
    L.append(f"  test {st['test']}: n_items={st['n_items']}  k={st['k']}")
    for s, a, b, d in zip(res["seeds"], st["acc_a"], st["acc_b"], st["d"]):
        L.append(f"    s{s}: A {pp(a, False)}  B {pp(b, False)}  d {pp(d)}")
    L.append(f"  mean d = {pp(st['mean'])}   acc A {pp(np.mean(st['acc_a']), False)}  B {pp(np.mean(st['acc_b']), False)}")
    if t["lo"] is None:
        L.append(f"  seed-t: {t['note']}")
    else:
        L.append(f"  seed-t (df={t['df']}): {ci(t['lo'], t['hi'])}  p={fmt_p(t['p'])}" + (f"  ({t['note']})" if t["note"] else ""))
    L.append(f"  item-cluster bootstrap (shared indices): {ci(*st['item_boot'])}")
    L.append(f"  two-level bootstrap (PRIMARY):           {ci(*st['two_level'])}")
    verdict = "SIGNIFICANT" if st["significant"] else "not significant"
    why = "" if st["significant"] else (" (k=1: seed-t unavailable)" if st["k"] < 2 else "")
    L.append(f"  -> {verdict} (rule: two-level CI and seed-t CI both exclude 0){why}")
    L.append(f"  ECE (15 bins, top-1) A {pp(np.mean(st['ece_a']), False)}  B {pp(np.mean(st['ece_b']), False)}")
    for name, sx in res["extras"].items():
        if sx.get("missing"):
            L.append(f"  [{name}] files missing for some run ({'+'.join(sx['sets'])}); skipped")
            continue
        tx = sx["t"]
        L.append(f"  [{name}] n={sx['n_items']}: A {pp(np.mean(sx['acc_a']), False)}  B {pp(np.mean(sx['acc_b']), False)}  "
                 f"d {pp(sx['mean'])}  two-level {ci(*sx['two_level'])}  seed-t {ci(tx['lo'], tx['hi'])}")
    return "\n".join(L)


def md_row(label: str, res: dict, extra_names: list[str]) -> str:
    if "error" in res:
        return f"| {label} | – | 0 | {res['error']} |" + " |" * (7 + len(extra_names))
    st, t = res["main"], res["main"]["t"]
    ep = fmt_ep(res["ep_a"]) if res.get("ep_a") is not None else "–"
    if res.get("select_a"):
        ep += " (val)"
    ds = " / ".join(f"s{s} {pp(d)}" for s, d in zip(res["seeds"], st["d"]))
    tci = "n/a (k=1)" if t["lo"] is None else f"{ci(t['lo'], t['hi'])}, p={fmt_p(t['p'])}"
    sig = "**yes**" if st["significant"] else "no"
    cells = [label, ep, str(st["k"]), f"{pp(np.mean(st['acc_a']), False)} / {pp(np.mean(st['acc_b']), False)}", ds,
             f"**{pp(st['mean'])}**", tci, ci(*st["item_boot"]), ci(*st["two_level"]), sig,
             f"{pp(np.mean(st['ece_a']), False)} / {pp(np.mean(st['ece_b']), False)}"]
    for name in extra_names:
        sx = res["extras"].get(name)
        cells.append("n/a" if sx is None or sx.get("missing") else f"{pp(sx['mean'])} {ci(*sx['two_level'])}")
    return "| " + " | ".join(cells) + " |"


def md_header(extra_names: list[str]) -> str:
    cols = ["Arm", "ep", "k", "Acc arm / base", "d_s per seed", "mean Δ", "seed-t 95% CI, p", "item-boot 95% CI",
            "two-level 95% CI (primary)", "sig", "ECE arm / base"] + [f"Δ {n} [two-level]" for n in extra_names]
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols)


# ----------------------------------------------------------------------------------------------------------
# Table mode
# ----------------------------------------------------------------------------------------------------------

def load_spec(path: Path) -> dict:
    text = Path(path).read_text()
    if str(path).endswith(".json"):
        return json.loads(text)
    import yaml
    return yaml.safe_load(text)


def arm_from(d: dict) -> ArmSpec:
    pats = d.get("pattern", d.get("patterns"))
    pats = [pats] if isinstance(pats, str) else list(pats)
    eps = d.get("epochs")
    eps = [float(e) for e in (eps if isinstance(eps, list) else [eps])] if eps is not None else None
    return ArmSpec(d["name"], pats, eps, [int(s) for s in d.get("exclude_seeds", [])])


def run_table(spec: dict, root_override: str | None = None) -> str:
    root = Path(root_override or spec.get("root", DEFAULT_ROOT))
    root = root if root.is_absolute() else ROOT / root
    opt = Options(root=root, test=spec.get("test", DEFAULT_TEST), sets=list(spec.get("sets", [])),
                  depth=bool(spec.get("depth", True)), select_by=spec.get("select_by", "none"),
                  exclude_seeds=[int(s) for s in spec.get("exclude_seeds", [])], boot=int(spec.get("boot", 10000)),
                  boot_seed=int(spec.get("boot_seed", 0)), alpha=float(spec.get("alpha", 0.05)))
    extra_names = ([n for n, _ in DEPTH_GROUPS] if opt.depth else []) + [parse_group(g)[0] for g in opt.sets]
    out = []
    for tab in spec["tables"]:
        base = arm_from(tab["baseline"])
        vb = resolve_arm(base, opt.root)
        arms = [arm_from(x) for x in tab["arms"]]
        notes: list[str] = []
        seeds_all = set()
        for a in arms:
            seeds_all |= set().union(*(set(v) for v in resolve_arm(a, opt.root).values()))
        b_res = pick(base, vb, seeds_all, opt, notes)
        out.append(f"### {tab.get('title', base.name)}\n")
        out.append(f"Baseline: `{base.name}` = {', '.join('`' + p + '`' for p in base.patterns)}"
                   + (f", ep{fmt_ep(b_res[0])}" if b_res[0] is not None else "")
                   + f"; seeds available {sorted(b_res[1])}. Test = {opt.test} (items keyed set:item_id). "
                   f"Δ = arm − baseline in accuracy points; bootstrap B = {opt.boot}.\n")
        rows, sel, detail = [], [], []
        for a in arms:
            res = compare(a, base, opt, b_resolved=b_res)
            rows.append(md_row(a.name, res, extra_names))
            sel += res["notes"]
            dr = res["dropped"]
            detail.append(f"- `{a.name}`: runs {', '.join('`' + r + '`' for r in res['runs_a'][:1])}"
                          f"{' …' if len(res['runs_a']) > 1 else ''}; paired seeds {res['seeds']}"
                          + (f"; excluded {dr['excluded']}" if dr["excluded"] else "")
                          + (f"; arm-only seeds {dr['a_only']}" if dr["a_only"] else "")
                          + (f"; baseline-only seeds {dr['b_only']}" if dr["b_only"] else ""))
            log(text_report(res))
        out.append(md_header(extra_names))
        out += rows
        out.append("")
        if notes or sel:
            out.append("Epoch selection (val only, seed-mean over the seeds that enter the paired comparison):")
            out += [f"- {n}" for n in notes + sel]
            out.append("")
        out.append("Seeds:")
        out += detail
        out.append("")
    out.append("sig = two-level CI and seed-t CI both exclude 0 (k=1 never qualifies). ECE: top-1, 15 equal-width "
               "bins, seed mean, on the test sets.")
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------

def _eps(s: str | None):
    return [float(x) for x in s.split(",")] if s else None


def _seeds(s: str | None):
    return [int(x) for x in s.split(",")] if s else []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", nargs="+", help="arm A: run-dir pattern(s) with {seed} [and {ep}], or run dirs")
    ap.add_argument("--b", nargs="+", help="arm B (baseline): same forms as --a")
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--a-epochs", default=None, help="comma list of {ep} values for A (default: discover)")
    ap.add_argument("--b-epochs", default=None)
    ap.add_argument("--select-by", choices=["none", "val"], default="none")
    ap.add_argument("--exclude-seeds", default=None, help="comma list (e.g. screening seeds), applied to both arms")
    ap.add_argument("--test", default=DEFAULT_TEST, help="'+'-joined sets concatenated into the test pool")
    ap.add_argument("--sets", action="append", default=[], help="extra set group 'name=a+b' or 'a+b' (repeatable)")
    ap.add_argument("--no-depth", action="store_true", help="skip the gs2-4 / gs5-8 depth groups")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--json", default=None, help="write the full result as JSON")
    ap.add_argument("--table", default=None, help="YAML/JSON spec: arms vs one baseline per table -> markdown")
    ap.add_argument("--out", default=None, help="table mode: write the markdown here")
    ap.add_argument("--spec-help", action="store_true", help="print an example table spec")
    args = ap.parse_args(argv)
    if args.spec_help:
        print(TABLE_SPEC_HELP)
        return 0
    if args.table:
        spec = load_spec(Path(args.table))
        if args.select_by != "none":
            spec["select_by"] = args.select_by
        md = run_table(spec, args.root if args.root != DEFAULT_ROOT else None)
        print(md)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(md + "\n")
        return 0
    if not args.a or not args.b:
        ap.error("--a and --b are required (or --table)")
    root = Path(args.root)
    root = root if root.is_absolute() else (Path.cwd() / root if (Path.cwd() / root).exists() else ROOT / root)
    opt = Options(root=root, test=args.test, sets=args.sets, depth=not args.no_depth, select_by=args.select_by,
                  exclude_seeds=_seeds(args.exclude_seeds), boot=args.boot, boot_seed=args.boot_seed, alpha=args.alpha)
    a = ArmSpec(args.a_name, args.a, _eps(args.a_epochs))
    b = ArmSpec(args.b_name, args.b, _eps(args.b_epochs))
    res = compare(a, b, opt)
    print(text_report(res))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
