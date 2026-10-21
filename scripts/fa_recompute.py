#!/usr/bin/env python3
"""FA readout under the 2026-10-20 statistics rules (scripts/paired_stats.py): TF (GFPS on reasoning trees) minus A2,
per test file (framework), the equal-weight mean over the five new frameworks, the in-domain REAL / AFTraj pairs, the
position-prior split, the content-node ablation, and (once they exist) the LOFO folds. CPU only, numpy only.

  python scripts/fa_recompute.py                         # results/hpc/fasf -> notes/.../fa_readout_<date>.md + JSON
  python scripts/fa_recompute.py --root results/hpc/fasf --out-md notes/next_round/failure_attribution/x.md

Rules (what changed from fa_pool.py / the 10-19 tables):
  * Unit of replication = the seed. Per seed s, d_s = acc_TF,s - acc_A2,s on the same items; seed-level paired
    Student-t with df = k - 1 (none at k = 1). Two-level bootstrap (resample seeds, then items with the SAME
    indices for every drawn seed and both arms; paired_stats.two_level_bootstrap), and the item-cluster bootstrap
    (shared indices) for reference. "significant" = the two-level interval AND the seed-t interval exclude 0, so a
    k = 1 comparison is never significant (item bootstrap only, reported as such).
  * Checkpoints chosen on val only. Primary: ONE epoch per arm = the --eval-at point (or the end) with the highest
    seed-mean accuracy on the run's own val file over the seeds entering the comparison (ties: fewer epochs; points
    every seed has). Secondary column: the per-run first-best val point (StepFinder's main.py rule, the 10-19
    pre-registration sf_protocol.md §9). Test files are never read for selection.
  * Screening vs confirmation seeds kept apart: "screen" = seed 12 alone (k = 1); "confirm" = seeds 13 / 14 (s12
    excluded; k = 2 -> df = 1, t crit 12.7); "all" = every paired seed (the §9 registration, which includes the
    screening seed; flagged).
  * Equal-weight pools: the mean of the per-file differences; seed-t on the per-seed equal-weight means; two-level
    bootstrap resampling seeds once per draw (shared over files) and items within each file.
  * alg + hc pooled: per seed, the mean of the alg-trained and hc-trained differences (same test items), i.e. a
    (seed) unit holding both training subsets.
  * Position-prior split: per test file, the best constant position (oracle on the test file: "step k" k = 0..7,
    "last - k" k = 0..3; ties to the earlier candidate), items on-prior (gold = that position) / off-prior.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P = _load("fa_pool_for_recompute", "scripts/fa_pool.py")
R = P.R
S = _load("paired_stats_for_recompute", "scripts/paired_stats.py")

M = "Qwen3.5-0.8B-Base"
NEW = ("xagtr", "xaftraj", "xte", "xwwpro", "xce")          # the five new frameworks (_sfclean files)
NEW_LABEL = {"xagtr": "AgenTracer", "xaftraj": "AFTraj", "xte": "TraceElephant", "xwwpro": "WW-Pro",
             "xce": "CORRECT", "sfalg": "Who&When alg", "sfhc": "Who&When hc", "lrca": "LongRCA",
             "real": "REAL (in-domain test)", "rse": "RootSE", "aft": "AFTraj (in-domain test, 163)"}
FILES = ("sfalg", "sfhc") + NEW + ("lrca", "rse", "real", "aft")
SCREEN, CONFIRM = (12,), (13, 14)
LOFO_FOLDS = {"whowhen": ("sfalg", "sfhc"), "aftraj": ("xaftraj",), "real": ("real",), "agentracer": ("xagtr",),
              "wwpro": ("xwwpro",), "correct": ("xce",)}


def pairs_spec() -> list[dict]:
    """The comparisons: name, A (baseline) pattern, B (TF) pattern with {seed}, files, kind."""
    out = []
    for sub in ("alg", "hc"):
        out.append({"name": "sf_%s" % sub, "title": "StepFinder %s-trained: GFPS-t2 minus A2" % sub,
                    "a": "A2-sfx6e-%s-student_fa_sf_%s-s{seed}" % (M, sub),
                    "b": "GFPS-sfs3e-%s-student_fa_sf_%s_t2-s{seed}" % (M, sub), "files": FILES, "pool": True})
        out.append({"name": "sfn_%s" % sub, "title": "StepFinder %s-trained: GFPS-t2 content nodes only (sfs3en) "
                    "minus A2" % sub, "a": "A2-sfx6e-%s-student_fa_sf_%s-s{seed}" % (M, sub),
                    "b": "GFPS-sfs3en-%s-student_fa_sf_%s_t2-s{seed}" % (M, sub), "files": FILES, "pool": True})
        out.append({"name": "sfnvall_%s" % sub, "title": "StepFinder %s-trained: content nodes only minus all nodes "
                    "(sfs3en - sfs3e)" % sub, "a": "GFPS-sfs3e-%s-student_fa_sf_%s_t2-s{seed}" % (M, sub),
                    "b": "GFPS-sfs3en-%s-student_fa_sf_%s_t2-s{seed}" % (M, sub), "files": FILES, "pool": True})
    out.append({"name": "real", "title": "REAL-trained (in-domain): GFPS-t2 minus A2",
                "a": "A2-sfx6r-%s-student_fa_real-s{seed}" % M, "b": "GFPS-sfs3r-%s-student_fa_real_t2-s{seed}" % M,
                "files": FILES, "pool": False})
    out.append({"name": "aftraj", "title": "AFTraj-trained (in-domain): GFPS-t2 minus A2",
                "a": "A2-sfx6r-%s-student_fa_aftraj-s{seed}" % M, "b": "GFPS-sfs3r-%s-student_fa_aftraj_t2-s{seed}" % M,
                "files": FILES, "pool": False})
    for v, ta, tb in (("", "sfx6o", "sfs3o"), ("_cap", "sfx6oc", "sfs3oc")):
        for fold in LOFO_FOLDS:
            out.append({"name": "lofo%s_%s" % (v, fold), "title": "LOFO%s held out %s: GFPS minus A2" % (v, fold),
                        "a": "A2-%s-%s-student_fa_lofo_%s%s_a2-s{seed}" % (ta, M, fold, v),
                        "b": "GFPS-%s-%s-student_fa_lofo_%s%s_tf-s{seed}" % (tb, M, fold, v),
                        "files": FILES, "pool": False, "lofo": fold, "variant": v})
    return out


# ------------------------------------------------------------------------------------------------- runs
def runs_of(root: Path, pattern: str) -> dict:
    """seed -> run dict (fa_readout.load_runs) for the dirs matching the {seed} pattern; only finished runs."""
    rx = re.compile("^" + re.escape(pattern).replace(r"\{seed\}", r"(?P<seed>\d+)") + "$")
    out = {}
    for d in sorted(root.iterdir()) if root.exists() else []:
        m = rx.match(d.name)
        if m and (d / "metrics.json").exists():
            r = R.load_runs([str(d)])
            if r:
                out[int(m.group("seed"))] = r[0]
    return out


def acc_of(preds: dict) -> float | None:
    v = [float(p["labels"][int(np.argmax(p["probs"]))] == p["gold_label"]) for p in preds.values()
         if p.get("gold_label") in p["labels"]]
    return float(np.mean(v)) if v else None


def val_points(r: dict) -> dict:
    """epoch -> val accuracy at every checkpoint of the run (mid points + the end)."""
    out = {}
    for ep, by in R.mid_preds(Path(r["dir"])).items():
        if "val" in by:
            out[ep] = acc_of(by["val"])
    fe = P.final_epoch(r)
    pv = Path(r["dir"]) / "preds_val.jsonl"
    if fe is not None and pv.exists():
        out[fe] = acc_of(R.read_preds(pv))
    return {k: v for k, v in sorted(out.items()) if v is not None}


def select_arm_epoch(runs: dict, seeds: list) -> tuple:
    """One epoch for the arm: highest seed-mean val accuracy over the points every seed has (ties: fewer epochs)."""
    pts = {s: val_points(runs[s]) for s in seeds}
    common = sorted(set.intersection(*(set(p) for p in pts.values()))) if pts else []
    if not common:
        return None, {}
    means = {ep: float(np.mean([pts[s][ep] for s in seeds])) for ep in common}
    best = max(means.values())
    return min(ep for ep, m in means.items() if m >= best - 1e-12), means


def per_run_epoch(r: dict) -> float | None:
    vp = val_points(r)
    best, at = -1.0, None
    for ep, a in vp.items():         # first best (strictly better replaces), main.py
        if a > best:
            best, at = a, ep
    return at


def correctness(r: dict, ep: float | None) -> dict:
    """{eval name: {item_id: correct}} of a run at epoch ep (None or the final epoch = the final predictions)."""
    fe = P.final_epoch(r)
    e = None if ep is None or (fe is not None and ep == fe) else ep
    preds, _ = P.run_preds(r, e, False)
    emap = R.eval_map(Path(r["dir"]), r["metrics"])
    out = {}
    for name, pr in preds.items():
        if not pr:
            continue
        path = R.resolve(name, emap, next(iter(pr)), r.get("data") or "")
        items = P.step_items(path) if path is not None else None
        if items:
            out[name] = ({iid: v[0] for iid, v in P.score(pr, items).items()}, path)
    return out


# ------------------------------------------------------------------------------------------------- statistics
def stats_D(D: np.ndarray, boot: int, seed: int = 0) -> dict:
    """D: (k seeds, n items) per-item differences B - A (0 / +-1, or averages). Mean, seed-t, bootstraps."""
    k, n = D.shape
    d = D.mean(axis=1)
    t = S.seed_t(d)
    ilo, ihi, _ = S.item_bootstrap(D, boot, np.random.default_rng(seed))
    if k > 1:
        tlo, thi, _ = S.two_level_bootstrap(D, boot, np.random.default_rng(seed + 1))
    else:
        tlo = thi = None
    sig = k > 1 and bool(t["sd"]) and S.excludes_zero(tlo, thi) and S.excludes_zero(t["lo"], t["hi"])
    return {"k": k, "n": n, "zero_spread": k > 1 and not t["sd"], "mean": 100 * float(d.mean()), "per_seed": [100 * float(x) for x in d],
            "t_lo": None if t["lo"] is None else 100 * t["lo"], "t_hi": None if t["hi"] is None else 100 * t["hi"],
            "p": t["p"], "two_level": None if tlo is None else [100 * tlo, 100 * thi], "item_boot": [100 * ilo, 100 * ihi],
            "significant": bool(sig)}


def stats_equal_weight(Ds: list, boot: int, seed: int = 0) -> dict:
    """Equal-weight mean over files of the per-file mean difference; Ds = [(k, n_f) arrays] with the same seeds."""
    k = Ds[0].shape[0]
    d = np.mean([D.mean(axis=1) for D in Ds], axis=0)            # per-seed equal-weight difference
    t = S.seed_t(d)
    rng = np.random.default_rng(seed)
    ib, tb = np.empty(boot), np.empty(boot)
    for b in range(boot):
        sidx = rng.integers(0, k, size=k)
        per_i, per_t = [], []
        for D in Ds:
            idx = rng.integers(0, D.shape[1], size=D.shape[1])
            per_i.append(D[:, idx].mean())
            per_t.append(D[sidx][:, idx].mean())
        ib[b], tb[b] = np.mean(per_i), np.mean(per_t)
    ilo, ihi = np.percentile(ib, [2.5, 97.5])
    tlo, thi = np.percentile(tb, [2.5, 97.5]) if k > 1 else (None, None)
    sig = k > 1 and bool(t["sd"]) and S.excludes_zero(tlo, thi) and S.excludes_zero(t["lo"], t["hi"])
    return {"k": k, "n_files": len(Ds), "zero_spread": k > 1 and not t["sd"], "mean": 100 * float(d.mean()), "per_seed": [100 * float(x) for x in d],
            "t_lo": None if t["lo"] is None else 100 * t["lo"], "t_hi": None if t["hi"] is None else 100 * t["hi"],
            "p": t["p"], "two_level": None if tlo is None else [100 * float(tlo), 100 * float(thi)],
            "item_boot": [100 * float(ilo), 100 * float(ihi)], "significant": bool(sig)}


# ------------------------------------------------------------------------------------------------- position prior
CANDS = [("step %d" % k, (lambda n, k=k: k)) for k in range(8)] + \
        [("last" if k == 0 else "last-%d" % k, (lambda n, k=k: n - 1 - k)) for k in range(4)]


def positions(path: Path) -> dict:
    """item_id -> (gold step position, number of steps) of a step-level FA file."""
    out = {}
    for line in open(path):
        if not line.strip():
            continue
        it = json.loads(line)
        opts = R.options(it)
        ms = [R.STEP_OPT.match(o) for o in opts]
        if not all(ms) or it.get("gold_label") not in it["labels"]:
            continue
        g = it["labels"].index(it["gold_label"])
        out[it["item_id"]] = (int(ms[g].group(1)), len(opts))
    return out


def best_constant(pos: dict) -> tuple:
    best = None
    for name, f in CANDS:
        acc = float(np.mean([g == f(n) for g, n in pos.values()]))
        if best is None or acc > best[1] + 1e-12:
            best = (name, acc, f)
    return best


# ------------------------------------------------------------------------------------------------- analysis
def compare(spec: dict, root: Path, seeds_req, boot: int, sel: str) -> dict | None:
    A, Bm = runs_of(root, spec["a"]), runs_of(root, spec["b"])
    seeds = sorted(set(A) & set(Bm) & set(seeds_req)) if seeds_req else sorted(set(A) & set(Bm))
    if not seeds:
        return None
    if sel == "arm":
        ea, ma = select_arm_epoch(A, seeds)
        eb, mb = select_arm_epoch(Bm, seeds)
        eps_a, eps_b = {s: ea for s in seeds}, {s: eb for s in seeds}
        selinfo = {"a": ea, "b": eb, "a_val": ma, "b_val": mb}
    else:
        eps_a = {s: per_run_epoch(A[s]) for s in seeds}
        eps_b = {s: per_run_epoch(Bm[s]) for s in seeds}
        selinfo = {"a": eps_a, "b": eps_b}
    ca = {s: correctness(A[s], eps_a[s]) for s in seeds}
    cb = {s: correctness(Bm[s], eps_b[s]) for s in seeds}
    res = {"seeds": seeds, "runs_a": [A[s]["name"] for s in seeds], "runs_b": [Bm[s]["name"] for s in seeds],
           "selection": selinfo, "files": {}, "D": {}, "prior": {}}
    for f in spec["files"]:
        if not all(f in ca[s] and f in cb[s] for s in seeds):
            continue
        ids = None
        for s in seeds:
            x = set(ca[s][f][0]) & set(cb[s][f][0])
            ids = x if ids is None else ids & x
        ids = sorted(ids)
        if not ids:
            continue
        D = np.array([[cb[s][f][0][i] - ca[s][f][0][i] for i in ids] for s in seeds], dtype=float)
        accA = 100 * np.mean([[ca[s][f][0][i] for i in ids] for s in seeds])
        accB = 100 * np.mean([[cb[s][f][0][i] for i in ids] for s in seeds])
        res["files"][f] = dict(stats_D(D, boot), acc_a=float(accA), acc_b=float(accB), path=str(ca[seeds[0]][f][1]))
        res["D"][f] = (ids, D)
        pos = positions(ca[seeds[0]][f][1])
        name, cacc, fn = best_constant({i: pos[i] for i in ids if i in pos})
        on = np.array([i in pos and pos[i][0] == fn(pos[i][1]) for i in ids])
        res["prior"][f] = {"const": name, "const_acc": 100 * cacc, "n_on": int(on.sum()), "n_off": int((~on).sum())}
        for lab, mask in (("on", on), ("off", ~on)):
            if mask.sum():
                res["prior"][f][lab] = stats_D(D[:, mask], boot)
                res["D"]["%s:%s" % (f, lab)] = (list(np.array(ids)[mask]), D[:, mask])
    newf = [f for f in NEW if f in res["files"]]
    if len(newf) == len(NEW):
        res["eq_new"] = stats_equal_weight([res["D"][f][1] for f in NEW], boot)
        for lab in ("on", "off"):
            if all("%s:%s" % (f, lab) in res["D"] for f in NEW):
                res["eq_new_" + lab] = stats_equal_weight([res["D"]["%s:%s" % (f, lab)][1] for f in NEW], boot)
    return res


def pooled_alg_hc(ra: dict, rh: dict, boot: int) -> dict | None:
    """Per seed, the mean of the alg- and hc-trained differences on the same items."""
    if not ra or not rh:
        return None
    seeds = sorted(set(ra["seeds"]) & set(rh["seeds"]))
    if not seeds:
        return None
    out = {"seeds": seeds, "files": {}}
    Ds = {}
    for f in ra["files"]:
        if f not in rh["files"]:
            continue
        ia, Da = ra["D"][f]
        ih, Dh = rh["D"][f]
        common = sorted(set(ia) & set(ih))
        pa = {i: k for k, i in enumerate(ia)}
        ph = {i: k for k, i in enumerate(ih)}
        sa = [ra["seeds"].index(s) for s in seeds]
        sh = [rh["seeds"].index(s) for s in seeds]
        D = (Da[sa][:, [pa[i] for i in common]] + Dh[sh][:, [ph[i] for i in common]]) / 2
        Ds[f] = D
        out["files"][f] = stats_D(D, boot)
    if all(f in Ds for f in NEW):
        out["eq_new"] = stats_equal_weight([Ds[f] for f in NEW], boot)
    return out


# ------------------------------------------------------------------------------------------------- report
def fx(x, nd=1):
    return "–" if x is None else ("%+.*f" % (nd, x))


def ci(lo, hi):
    return "–" if lo is None else "[%+.1f, %+.1f]" % (lo, hi)


def cell(st: dict | None) -> str:
    if not st:
        return "–"
    s = "%s" % fx(st["mean"])
    if st["k"] > 1:
        s += " t%s 2L%s" % (ci(st["t_lo"], st["t_hi"]), ci(*(st["two_level"] or (None, None))))
        if st["significant"]:
            s += " **sig**"
        elif st.get("zero_spread"):
            s += " (equal seed differences: t undefined, not sig)"
    else:
        s += " (k=1, item %s)" % ci(*st["item_boot"])
    return s


def lofo_summary(allres: dict, variant: str, sset: str, boot: int, seed: int = 0) -> dict | None:
    """The pre-registered LOFO readout (lofo_prereg_2026-10-21.md): per fold the held-out TF - A2 (whowhen: equal
    weight of its two Who&When files), the equal-weight mean over the folds (per seed: mean over folds; two-level
    bootstrap: seeds once per draw, shared over folds, items within each file), its off-prior twin, and the number of
    folds with TF - A2 > 0. Only folds with a result for the seed set enter; seeds = those every entering fold has."""
    folds = {}
    for fold, files in LOFO_FOLDS.items():
        b = allres.get("lofo%s_%s" % (variant, fold))
        r = b and b["by"].get(sset)
        if r and all(f in r["D"] for f in files):
            folds[fold] = r
    if not folds:
        return None
    seeds = sorted(set.intersection(*(set(r["seeds"]) for r in folds.values())))
    if not seeds:
        return None
    def mats(r, files, lab=""):
        idx = [r["seeds"].index(x) for x in seeds]
        out = []
        for f in files:
            key = f if not lab else "%s:%s" % (f, lab)
            if key not in r["D"]:
                return None
            out.append(r["D"][key][1][idx])
        return out
    per, per_off, fold_stats = [], [], {}
    for fold, r in folds.items():
        Ms = mats(r, LOFO_FOLDS[fold])
        per.append(Ms)
        fold_stats[fold] = stats_equal_weight(Ms, boot) if len(Ms) > 1 else stats_D(Ms[0], boot)
        Mo = mats(r, LOFO_FOLDS[fold], "off")
        if Mo is not None:
            per_off.append(Mo)
    def eqfold(groups):
        k = len(seeds)
        d = np.mean([np.mean([M.mean(axis=1) for M in g], axis=0) for g in groups], axis=0)
        t = S.seed_t(d)
        rng = np.random.default_rng(seed)
        ib, tb = np.empty(boot), np.empty(boot)
        for b in range(boot):
            sidx = rng.integers(0, k, size=k)
            fi, ft = [], []
            for g in groups:
                gi, gt = [], []
                for M in g:
                    ii = rng.integers(0, M.shape[1], size=M.shape[1])
                    gi.append(M[:, ii].mean())
                    gt.append(M[sidx][:, ii].mean())
                fi.append(np.mean(gi))
                ft.append(np.mean(gt))
            ib[b], tb[b] = np.mean(fi), np.mean(ft)
        ilo, ihi = np.percentile(ib, [2.5, 97.5])
        tlo, thi = np.percentile(tb, [2.5, 97.5]) if k > 1 else (None, None)
        sig = k > 1 and bool(t["sd"]) and S.excludes_zero(tlo, thi) and S.excludes_zero(t["lo"], t["hi"])
        return {"k": k, "mean": 100 * float(d.mean()), "per_seed": [100 * float(x) for x in d],
                "t_lo": None if t["lo"] is None else 100 * t["lo"], "t_hi": None if t["hi"] is None else 100 * t["hi"],
                "p": t["p"], "two_level": None if tlo is None else [100 * float(tlo), 100 * float(thi)],
                "item_boot": [100 * float(ilo), 100 * float(ihi)], "significant": bool(sig),
                "zero_spread": k > 1 and not t["sd"]}
    out = {"variant": variant or "main", "seeds": seeds, "folds": fold_stats, "n_folds": len(folds),
           "eq_folds": eqfold(per), "eq_folds_off": eqfold(per_off) if len(per_off) == len(per) else None,
           "folds_positive": sum(1 for v in fold_stats.values() if v["mean"] > 0)}
    e, o = out["eq_folds"], out["eq_folds_off"]
    if sset == "screen" and len(folds) == len(LOFO_FOLDS if not variant else [f for f in LOFO_FOLDS if f != "whowhen"]):
        out["G1"] = {"a_eq_ge_2": e["mean"] >= 2.0, "b_4of6_pos": out["folds_positive"] >= 4,
                     "c_off_ge_0": o is not None and o["mean"] >= 0}
        out["G1"]["pass"] = all(out["G1"].values())
    if sset == "confirm" and len(seeds) >= 2:
        out["C"] = {"mean_ge_1.5": e["mean"] >= 1.5, "all_seeds_pos": all(x > 0 for x in e["per_seed"]),
                    "two_level_lo_gt_0": bool(e["two_level"] and e["two_level"][0] > 0)}
        out["C"]["pass"] = all(out["C"].values())
    return out


def report(allres: dict, root: Path, boot: int) -> str:
    L = ["# FA readout under the new statistics rules (%s)" % dt.date.today().isoformat(), "",
         "Source: `%s` (local runs at the time of the run). Script: `scripts/fa_recompute.py`." % root,
         "Rules: seed = unit; per-seed difference TF − A2 (points); seed-level paired t (df = k − 1; with k = 2 the "
         "t critical value is 12.7) = `t[...]`; two-level bootstrap (seeds, then items with shared indices; %d draws) = "
         "`2L[...]`; **sig** only when both exclude 0. k = 1: item bootstrap only, never significant. Checkpoint: one "
         "epoch per arm, highest seed-mean val accuracy (ties → fewer epochs); test files never used for selection. "
         "Seed sets: screen = s12 (k = 1), confirm = s13 + s14 (s12 excluded), all = every paired seed (includes the "
         "screening seed; the 10-19 §9 registration). Position constants: the 10-19 note listed AgenTracer 'step 4 "
         "(25.0)' and WW-Pro 'step 6 (44.3)'; on the files the best constant is 'last' (24.2 / 43.1) and step 4 / "
         "step 6 score 15.1 / 3.7, so those two labels look like a candidate-index mislabel in the lost 10-19 script; "
         "the other five constants reproduce exactly. The 10-19 per-framework s12 / s13 numbers (alg: +11.8 / +2.1 "
         "AgenTracer, +12.3 / +3.4 AFTraj, +6.3 / +2.5 TE, +4.1 / -2.5 WW-Pro, -6.7 / -32.6 CORRECT) reproduce "
         "exactly under the per-seed val choice." % boot, ""]
    for key, block in allres.items():
        if key.startswith("_lofo"):
            continue
        spec, by = block["spec"], block["by"]
        if not any(by.values()):
            continue
        L += ["## %s" % spec["title"], ""]
        for sset, res in by.items():
            if not res:
                continue
            sel = res["selection"]
            L.append("- **%s** seeds %s; epochs A2 %s / TF %s (val means A2 %s; TF %s)" % (
                sset, ",".join(map(str, res["seeds"])), sel["a"], sel["b"],
                {k: round(100 * v, 1) for k, v in (sel.get("a_val") or {}).items()},
                {k: round(100 * v, 1) for k, v in (sel.get("b_val") or {}).items()}))
        L += ["", "| test file | n | " + " | ".join("%s: TF − A2" % s for s in by if by[s]) + " | acc A2 → TF (all) |",
              "|---|---|" + "---|" * sum(1 for s in by if by[s]) + "---|"]
        files = [f for f in FILES if any(r and f in r["files"] for r in by.values())]
        for f in files:
            ref = next(r for r in by.values() if r and f in r["files"])["files"][f]
            row = "| %s | %d | " % (NEW_LABEL.get(f, f), ref["n"])
            row += " | ".join(cell(r["files"].get(f)) if r else "–" for r in by.values() if r)
            al = by.get("all") or {}
            row += " | %s |" % ("%.1f → %.1f" % (al["files"][f]["acc_a"], al["files"][f]["acc_b"]) if al and f in al["files"] else "–")
            L.append(row)
        if any(r and "eq_new" in r for r in by.values()):
            L.append("| **equal weight, 5 new** | | " + " | ".join(cell(r.get("eq_new")) for r in by.values() if r) + " | |")
            for lab in ("off", "on"):
                L.append("| equal weight, %s-prior | | " % lab + " | ".join(cell(r.get("eq_new_" + lab))
                                                                         for r in by.values() if r) + " | |")
        L.append("")
        al = by.get("all")
        if al and al.get("prior"):
            L += ["Position-prior split (all seeds; best constant on the test file):", "",
                  "| test file | best constant (acc) | n off / on | off-prior TF − A2 | on-prior TF − A2 |",
                  "|---|---|---|---|---|"]
            for f in files:
                pr = al["prior"].get(f)
                if pr:
                    L.append("| %s | %s (%.1f) | %d / %d | %s | %s |" % (NEW_LABEL.get(f, f), pr["const"], pr["const_acc"],
                             pr["n_off"], pr["n_on"], cell(pr.get("off")), cell(pr.get("on"))))
            L.append("")
        if block.get("pooled"):
            L += ["alg + hc pooled (per seed: mean of the two training subsets):", ""]
            for sset, pr in block["pooled"].items():
                if pr:
                    L.append("- %s (seeds %s): equal weight 5 new %s; AgenTracer %s; AFTraj %s" % (
                        sset, ",".join(map(str, pr["seeds"])), cell(pr.get("eq_new")),
                        cell(pr["files"].get("xagtr")), cell(pr["files"].get("xaftraj"))))
            L.append("")
    for v in ("", "_cap"):
        for sset in ("screen", "confirm", "all"):
            sm = allres.get("_lofo%s_%s" % (v, sset))
            if not sm:
                continue
            L += ["## LOFO%s summary, %s (seeds %s; lofo_prereg_2026-10-21.md)" % (v, sset, ",".join(map(str, sm["seeds"]))),
                  "", "| held-out fold | TF − A2 |", "|---|---|"]
            for fold, st in sm["folds"].items():
                L.append("| %s | %s |" % (fold, cell(st)))
            L.append("| **equal weight over %d folds** | %s |" % (sm["n_folds"], cell(sm["eq_folds"])))
            L.append("| equal weight, off-prior | %s |" % cell(sm["eq_folds_off"]))
            L.append("")
            L.append("- folds with TF − A2 > 0: %d / %d" % (sm["folds_positive"], sm["n_folds"]))
            for g in ("G1", "C"):
                if g in sm:
                    L.append("- gate %s: %s" % (g, ", ".join("%s=%s" % kv for kv in sm[g].items())))
            L.append("")
    L += ["## Per-run first-best val selection (secondary; sf_protocol.md §9 rule), all seeds", ""]
    for key, block in allres.items():
        if key.startswith("_lofo"):
            continue
        r2 = block.get("per_run")
        if r2:
            L.append("- %s: equal weight 5 new %s; per file %s" % (
                block["spec"]["title"], cell(r2.get("eq_new")),
                ", ".join("%s %s" % (NEW_LABEL.get(f, f), fx(v["mean"])) for f, v in r2["files"].items())))
    return "\n".join(L) + "\n"


def strip(res):
    if isinstance(res, dict):
        return {k: strip(v) for k, v in res.items() if k != "D"}
    if isinstance(res, list):
        return [strip(x) for x in res]
    return res


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="results/hpc/fasf")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--only", default="", help="comma list of comparison names (default: all)")
    ap.add_argument("--out-md", default="notes/next_round/failure_attribution/fa_readout_%s.md" % dt.date.today())
    ap.add_argument("--out-json", default="results/fa_pool/fa_recompute_%s.json" % dt.date.today())
    args = ap.parse_args(argv)
    root = Path(args.root) if Path(args.root).is_absolute() else ROOT / args.root
    allres = {}
    for spec in pairs_spec():
        if args.only and spec["name"] not in args.only.split(","):
            continue
        by = {"screen": compare(spec, root, SCREEN, args.boot, "arm"),
              "confirm": compare(spec, root, CONFIRM, args.boot, "arm"),
              "all": compare(spec, root, None, args.boot, "arm")}
        allres[spec["name"]] = {"spec": spec, "by": by, "per_run": compare(spec, root, None, args.boot, "run")}
        if by["all"]:
            print("%-14s seeds %s  eq5 %s" % (spec["name"], by["all"]["seeds"], cell(by["all"].get("eq_new"))),
                  file=sys.stderr)
    for sub in ("", "n"):
        a, h = allres.get("sf%s_alg" % sub), allres.get("sf%s_hc" % sub)
        if a and h:
            a["pooled"] = {s: pooled_alg_hc(a["by"][s], h["by"][s], args.boot) for s in ("screen", "confirm", "all")}
    for v in ("", "_cap"):
        for sset in ("screen", "confirm", "all"):
            allres["_lofo%s_%s" % (v, sset)] = lofo_summary(allres, v, sset, args.boot)
    md = report(allres, root, args.boot)
    out_md = ROOT / args.out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)
    out_js = ROOT / args.out_json
    out_js.parent.mkdir(parents=True, exist_ok=True)
    out_js.write_text(json.dumps(strip({k: ({kk: vv for kk, vv in v.items() if kk != "spec"} if v and not k.startswith("_lofo")
                                            else v) for k, v in allres.items()}),
                                 indent=1, default=str))
    print("wrote %s and %s" % (out_md, out_js), file=sys.stderr)
    return allres


if __name__ == "__main__":
    main()
