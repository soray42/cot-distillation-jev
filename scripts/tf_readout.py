"""Tree Forcing readout (notes/prereg_TF.md): arms A2 / G3 / GFP / TFL / TFA / TFLM on K&K. CPU only, reads local
copies of HPC run directories, numpy only.

Per run (and per arm: mean and range over seeds):
  kk 11-20      PRIMARY endpoint (prereg_TF "Endpoints"): kk_deep argmax accuracy on 11-20 people (1,300 items)
  dA            analyze_e2.run_summary["A"] (400 diag programs, mean correctness over R1/R2/C)
  kk_deep bins  4-10 (trained range, 700), 11-16 (900), 17-20 (400) and per people count; the people count is the
                "/dN" suffix of the group: "kk/dN" for 4-16 but "kkx/dN" for 17-20 in data/eval/kk_deep.jsonl
  local         local-step probe (eval key local_kk; local_kk_anc for a run evaluated in the ancestors view), overall
                and by node depth 0 / 1-3 / 4-7 / 8-12 / >=13 (prereg_TF bins, rebuilt from the exact "depth/N" keys)
  bbh/jevbench/folio/val_facts   analyze_e2.run_summary (bbh = macro over tasks)
  train h       history s_per_step x step / 3600 (ablation_table.train_hours: mid-training evals in, final eval out)
  click         first evaluation with diagA (diag_A_mid, 450 items; not dA) >= .40, curve points plus the
                end-of-training evaluation as the last point (scripts/curve_metrics.py); never -> censored at the last
                step ("never"); reported as epoch and step. AUC = mean diagA over all those points (prereg_T 10-01 23:45)
Contrasts (paired by seed label; A2 ignores --order-seed, so pairs with A2 share the label only):
  H1 TFL-TFLM, H2 TFL-GFP (one-sided, Holm m = 2), H3 TFA-TFL (two-sided, descriptive), and the references
  TFL-A2, TFL-G3 (reported, not tested), each on kk 11-20 (primary), dA, kk 4-10 / 11-16 / 17-20.
  Pooled paired bootstrap: the same resampled units (kk_deep item ids; diag programs for dA) for both arms and every
  paired seed, averaging the per-seed paired differences; p = share of draws <= 0. Per-seed differences each with a
  unit bootstrap CI, and a seed-level t-CI (ablation_table.t_q). Without a common seed: difference of arm means with
  jointly resampled units, flagged "unpaired". All conditional on the trained runs.
  H4 (extrapolation) for each comparator that passed H1/H2: DiD = D(11-20) - D(4-10), D = TFL - comparator, items
  resampled within bins, one-sided, Holm over the H4 tests that run; plus D per bin, the slope of D on people (4-20)
  and the chance-normalised DiD (D / (comparator - 1/6) per bin).
Then the decision rules of notes/prereg_TF.md, hard-coded in decide() (the file is prose); PREREG_MD5 is the version
they were transcribed from and a changed file is flagged in the output.

Arm and variant: a run dir is <ARM>[-<tag>]-<model>-<data>-s<seed> (hpc/train_arm.sbatch OUT). One (tag, data) variant
is used per arm: --pick ARM=tag[/data], else the variant with the most runs, ties to the most training hours (with the
existing runs this gives the registered references A2-c8 v3 and G3-c4 v3f); the others are listed as "not used".
--alias OLD=NEW renames arms (stand-ins, e.g. TF=TFL TFM=TFLM), shown in the header. Only runs trained on --data-re
(default ^student_v3: v3, v3f, v3s, v3sl) with preds_diag.jsonl are read.

  python scripts/tf_readout.py 'results/hpc/tf/*' 'results/hpc/w1/*' --out results/tf_readout/tf
  python scripts/tf_readout.py 'results/hpc/w1/*' 'results/hpc/c4v3f/*' --alias TF=TFL --alias TFM=TFLM \
      --out results/tf_readout/standin_w1_c4v3f        # stand-in test on the exploratory TF/TFM runs
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation_table import t_q, train_hours  # noqa: E402
from analyze_e2 import correct, diag_table, primary, read_preds, run_summary  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "notes" / "prereg_TF.md"
PREREG_MD5 = "f8b473042e1e69b0e8f51fc7b84ecab4"      # the draft of 2026-10-06 12:23 the rules below were taken from
ARMS = ["A2", "G3", "GFP", "TFL", "TFA", "TFLM"]
REGISTERED = {"TFL": [12, 13, 14], "GFP": [12, 13, 14], "TFLM": [12, 13], "TFA": [12, 13], "G3": [12, 13, 14],
              "A2": [12, 13, 14]}                   # prereg_TF "Fixed settings" (G3-c4 v3f and A2-c8: references)
CONTRASTS = [("H1", "TFL", "TFLM"), ("H2", "TFL", "GFP"), ("H3", "TFA", "TFL"), ("ref", "TFL", "A2"),
             ("ref", "TFL", "G3")]
BINS = {"4-10": (4, 10), "11-16": (11, 16), "17-20": (17, 20), "11-20": (11, 20)}
PRIMARY = "kk11-20"
ENDPOINTS = [PRIMARY, "dA", "kk4-10", "kk11-16", "kk17-20"]
LOCAL_BINS = [("0", 0, 0), ("1-3", 1, 3), ("4-7", 4, 7), ("8-12", 8, 12), (">=13", 13, 10 ** 6)]
CLICK = 0.40
CHANCE = 1 / 6                                      # kk_deep: 6 options
PASS_MEAN, NULL_BAND, NO_HARM, NEG_GAP, H3_GAP = 0.03, 0.03, -0.02, 0.05, 0.03   # prereg_TF "Decision rules"
NAME = re.compile(r"^(?P<left>.+?)-(?P<model>[^-]*Qwen[^/]*?)-(?P<data>student_[A-Za-z0-9_]+)-s(?P<seed>\d+)$")


# ---------------------------------------------------------------- runs
def parse_name(name: str) -> dict | None:
    m = NAME.match(name)
    if not m:
        return None
    arm, _, tag = m["left"].partition("-")
    return {"arm": arm, "tag": tag, "model": m["model"], "data": m["data"], "seed": int(m["seed"])}


def people(group: str | None) -> int | None:
    m = re.search(r"/d(\d+)$", group or "")
    return int(m[1]) if m else None


def kkdeep_units(run: Path) -> dict[str, tuple[int, int]]:
    """item_id -> (people, correct)."""
    f = run / "preds_kkdeep.jsonl"
    out = {}
    if f.exists():
        for p in read_preds(f):
            n = people(p.get("group"))
            if n is not None and p.get("gold_label") in p["labels"]:
                out[p["item_id"]] = (n, int(correct(p)))
    return out


def curve(metrics: dict) -> dict:
    """diagA curve: trajectory points plus the end-of-training evaluation as the last point (curve_metrics.py)."""
    pts = [(x["step"], x.get("epoch"), x["diagA"]["acc"]) for x in metrics.get("trajectory") or [] if "diagA" in x]
    end = (metrics.get("eval") or {}).get("diagA")
    hist = metrics.get("history") or []
    if end is not None:
        step = hist[-1]["step"] if hist else (pts[-1][0] if pts else 0)
        pts.append((step, (metrics.get("args") or {}).get("epochs"), end["acc"]))
    if not pts:
        return {"click_step": None, "click_epoch": None, "clicked": None, "auc": None}
    hit = next((p for p in pts if p[2] >= CLICK), None)
    return {"click_step": hit[0] if hit else pts[-1][0], "click_epoch": hit[1] if hit else None,
            "clicked": hit is not None, "auc": sum(p[2] for p in pts) / len(pts), "points": len(pts)}


def local_probe(metrics: dict) -> tuple[str | None, dict, int | None]:
    ev = metrics.get("eval") or {}
    for key in ("local_kk", "local_kk_anc"):
        if key in ev:
            m = ev[key]
            out = {"all": m["all"]["acc"]} if "all" in m else {}
            for lab, lo, hi in LOCAL_BINS:
                cells = [(v["n"], v["acc"]) for k, v in m.items()
                         if k.startswith("depth/") and lo <= int(k.split("/")[1]) <= hi]
                n = sum(c[0] for c in cells)
                if n:
                    out[lab] = sum(c[0] * c[1] for c in cells) / n
            return key, out, m.get("all", {}).get("n")
    return None, {}, None


def load_run(d: Path, info: dict) -> dict:
    metrics = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
    s = run_summary(d)
    t = diag_table(d)
    units = {"dA": {g: primary(t, [g]) for g in t}}
    kk = kkdeep_units(d)
    by_n: dict[int, list[int]] = defaultdict(list)
    for n, c in kk.values():
        by_n[n].append(c)
    for name, (lo, hi) in BINS.items():
        units["kk" + name] = {i: float(c) for i, (n, c) in kk.items() if lo <= n <= hi}
    units["people"] = {i: n for i, (n, c) in kk.items()}
    lkey, loc, ln = local_probe(metrics)
    r = dict(info, dir=str(d), name=d.name, dA=s["A"], bbh=s.get("bbh"), jevbench=s.get("jevbench"),
             folio=s.get("folio"), val_facts=s.get("val_facts"),
             hours=train_hours({"metrics": metrics}), **curve(metrics),
             people_acc={n: sum(v) / len(v) for n, v in sorted(by_n.items())},
             people_n={n: len(v) for n, v in sorted(by_n.items())}, local_key=lkey, local=loc, local_n=ln)
    for name in BINS:
        u = units["kk" + name]
        r["kk" + name] = sum(u.values()) / len(u) if u else None
        r["kk" + name + "_n"] = len(u)
    r["_units"] = units
    return r


def collect(patterns: list[str], alias: dict, picks: dict, data_re: str) -> tuple[dict, list, list]:
    """-> ({arm: {seed: run}}, not-used runs, skipped dirs)."""
    dirs = sorted({Path(p) for pat in patterns for p in (glob.glob(pat) or [pat])})
    found, skipped = [], []
    for d in dirs:
        info = parse_name(d.name) if d.is_dir() else None
        if info is None or not re.search(data_re, info["data"]) or not (d / "preds_diag.jsonl").exists():
            skipped.append((str(d), "name/data/preds_diag"))
            continue
        info["orig_arm"] = info["arm"]
        info["arm"] = alias.get(info["arm"], info["arm"])
        if info["arm"] not in ARMS:
            skipped.append((str(d), f"arm {info['arm']} not in {'/'.join(ARMS)}"))
            continue
        found.append(load_run(d, info))
    runs: dict[str, dict[int, dict]] = {}
    unused = []
    for arm in ARMS:
        variants = defaultdict(list)
        for r in found:
            if r["arm"] == arm:
                variants[(r["tag"], r["data"])].append(r)
        if not variants:
            continue
        if arm in picks:
            tag, _, data = picks[arm].partition("/")
            keys = [k for k in variants if k[0] == tag and (not data or k[1] == data)]
            if len(keys) != 1:
                raise SystemExit(f"--pick {arm}={picks[arm]}: {'ambiguous, add /data' if keys else 'no such variant'}"
                                 f" ({sorted(variants)})")
            key = keys[0]
        else:
            key = max(variants, key=lambda k: (len(variants[k]), np.mean([r["hours"] or 0 for r in variants[k]]), k))
        for k, v in variants.items():
            if k == key:
                dup = [r["seed"] for r in v if sum(x["seed"] == r["seed"] for x in v) > 1]
                if dup:
                    raise SystemExit(f"{arm} {key}: duplicate seed(s) {sorted(set(dup))}")
                runs[arm] = {r["seed"]: r for r in v}
            else:
                unused += v
    return runs, unused, skipped


# ---------------------------------------------------------------- bootstrap
def boot_means(vec: np.ndarray, draws: int, rng: np.random.Generator, chunk: int = 1000) -> np.ndarray:
    """Means of `draws` resamples of vec (chunked so the index matrix stays small)."""
    out = np.empty(draws)
    for s in range(0, draws, chunk):
        k = min(chunk, draws - s)
        out[s:s + k] = vec[rng.integers(0, len(vec), size=(k, len(vec)))].mean(1)
    return out


def ci95(bs: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.md5("|".join(map(str, key)).encode()).hexdigest()[:8], 16))


def _vectors(s1: dict, s2: dict, endpoint: str):
    allruns = list(s1.values()) + list(s2.values())
    sets = [set(r["_units"][endpoint]) for r in allruns]
    keys = sorted(set.intersection(*sets))
    vec = lambda r: np.array([r["_units"][endpoint][k] for k in keys], dtype=float)  # noqa: E731
    return keys, vec, max(len(s) for s in sets) - len(keys)


def seed_t_ci(xs: list[float]):
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    q = t_q(0.025, n - 1)
    return (m - q * sd / math.sqrt(n), m + q * sd / math.sqrt(n))


def contrast(runs: dict, a1: str, a2: str, endpoint: str, boot: int, boot_seed: int) -> dict:
    s1, s2 = runs.get(a1, {}), runs.get(a2, {})
    res = {"contrast": f"{a1}-{a2}", "endpoint": endpoint, "seeds": sorted(set(s1) & set(s2)),
           "only1": sorted(set(s1) - set(s2)), "only2": sorted(set(s2) - set(s1)), "seed_label_only": "A2" in (a1, a2)}
    if not s1 or not s2:
        res["missing"] = [a for a, s in ((a1, s1), (a2, s2)) if not s]
        return res
    keys, vec, dropped = _vectors(s1, s2, endpoint)
    if not keys:
        res["missing"] = ["no common units"]
        return res
    res["units"], res["units_dropped"] = len(keys), dropped
    if res["seeds"]:
        per = {s: vec(s1[s]) - vec(s2[s]) for s in res["seeds"]}
        res["per_seed"] = {s: float(v.mean()) for s, v in per.items()}
        res["per_seed_ci"] = {s: ci95(boot_means(v, boot_seed, _rng(a1, a2, endpoint, s))) for s, v in per.items()}
        d = np.mean(list(per.values()), axis=0)       # same resampled units for every paired seed
        res["paired"] = True
        res["t_ci"] = seed_t_ci(list(res["per_seed"].values()))
    else:                                             # no common seed: difference of arm means
        d = np.mean([vec(r) for r in s1.values()], axis=0) - np.mean([vec(r) for r in s2.values()], axis=0)
        res["paired"] = False
    bs = boot_means(d, boot, _rng(a1, a2, endpoint, "pooled"))
    res["mean"] = float(d.mean())
    res["ci"] = ci95(bs)
    res["p_one"] = float((bs <= 0).mean())            # prereg_TF: p = share of draws <= 0
    res["p_two"] = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
    return res


def did(runs: dict, a1: str, a2: str, boot: int) -> dict:
    """H4: DiD = D(11-20) - D(4-10), D = a1 - a2 pooled over paired seeds, items resampled within bins."""
    s1, s2 = runs.get(a1, {}), runs.get(a2, {})
    seeds = sorted(set(s1) & set(s2))
    res = {"contrast": f"{a1}-{a2}", "seeds": seeds}
    if not seeds:
        res["missing"] = True
        return res
    p1, p2 = {s: s1[s] for s in seeds}, {s: s2[s] for s in seeds}
    parts, per_seed = {}, defaultdict(dict)
    for b in ("kk11-20", "kk4-10", "kk11-16", "kk17-20"):
        keys, vec, _ = _vectors(p1, p2, b)
        if not keys:
            res["missing"] = True
            return res
        per = {s: vec(p1[s]) - vec(p2[s]) for s in seeds}
        parts[b] = (np.mean(list(per.values()), axis=0),
                    float(np.mean([vec(p2[s]).mean() for s in seeds])))      # comparator accuracy in the bin
        for s, v in per.items():
            per_seed[s][b] = float(v.mean())
    hi, lo = parts["kk11-20"][0], parts["kk4-10"][0]
    bs = boot_means(hi, boot, _rng(a1, a2, "did-hi")) - boot_means(lo, boot, _rng(a1, a2, "did-lo"))
    res.update(did=float(hi.mean() - lo.mean()), ci=ci95(bs), p_one=float((bs <= 0).mean()),
               per_seed={s: v["kk11-20"] - v["kk4-10"] for s, v in per_seed.items()},
               delta={b: float(parts[b][0].mean()) for b in parts},
               did_norm=float(hi.mean() / (parts["kk11-20"][1] - CHANCE) - lo.mean() / (parts["kk4-10"][1] - CHANCE)))
    # slope of D on the number of people, 4-20 (OLS over items of the seed-averaged paired difference)
    keys, _, _ = _vectors(p1, p2, "people")
    d = np.mean([_item_diff(p1[s], p2[s], keys) for s in seeds], axis=0)
    x = np.array([p1[seeds[0]]["_units"]["people"][k] for k in keys], dtype=float)
    res["slope"] = float(np.polyfit(x, d, 1)[0]) if len(set(x)) > 1 else None
    return res


def _item_diff(r1: dict, r2: dict, keys: list[str]) -> np.ndarray:
    u1, u2 = {}, {}
    for b in ("kk4-10", "kk11-20"):
        u1.update(r1["_units"][b])
        u2.update(r2["_units"][b])
    return np.array([u1[k] - u2[k] for k in keys], dtype=float)


def holm(ps: dict[str, float], m: int | None = None) -> dict[str, float]:
    m = m or len(ps)
    out, run = {}, 0.0
    for j, (k, p) in enumerate(sorted(ps.items(), key=lambda kv: kv[1])):
        run = max(run, min(1.0, (m - j) * p))
        out[k] = run
    return out


# ---------------------------------------------------------------- prereg_TF decision rules
def test_passes(c: dict | None, holm_p: float | None, stat: str = "mean") -> bool:
    """prereg_TF: a test passes when Holm p < .05, every paired seed > 0 and the pooled mean >= +.03."""
    return bool(c and holm_p is not None and c.get("per_seed") and holm_p < 0.05
                and all(x > 0 for x in c["per_seed"].values()) and c[stat] >= PASS_MEAN)


def decide(runs: dict, C: dict, H4: dict, holm1: dict, holm4: dict) -> dict:
    """The decision rules of notes/prereg_TF.md ("Decision rules", "Hypotheses and analysis"), on the primary endpoint
    kk_deep 11-20 unless stated. Returns {rule: text}."""
    out = {}
    h = {k: C.get((k, PRIMARY)) for k in ("H1", "H2", "H3")}
    passed = {k: test_passes(h[k], holm1.get(k)) for k in ("H1", "H2")}
    for k, comp in (("H1", "TFLM"), ("H2", "GFP")):
        c = h[k]
        if not c or "mean" not in c:
            out[k] = f"not testable: {comp} or TFL missing"
            continue
        n, signs = len(c.get("per_seed", {})), "".join("+" if x > 0 else "-" for x in c.get("per_seed", {}).values())
        out[k] = (f"{'PASS' if passed[k] else 'fail'}: TFL-{comp} {c['mean']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}],"
                  f" Holm p {holm1.get(k, float('nan')):.4f}, seeds {signs or '(unpaired)'} (n {n}"
                  + (f", registered {len(REGISTERED[comp])})" if n < min(len(REGISTERED['TFL']), len(REGISTERED[comp]))
                     else ")"))
    for k, comp in (("H1", "TFLM"), ("H2", "GFP")):
        e = H4.get(comp)
        if e is None:
            out[f"H4 vs {comp}"] = f"not run ({k} did not pass)"
        elif e.get("missing"):
            out[f"H4 vs {comp}"] = "not testable (no paired seed)"
        else:
            # H4 pass = the general prereg_TF test definition (Holm p < .05, every paired seed's DiD > 0, pooled DiD
            # >= +.03), applied to the DiD; the strong rule's parenthesis "(DiD CI excludes 0)" is weaker and is
            # printed beside it. INTERPRETATION, to be confirmed by the owner before the first TFL result.
            ok = (holm4.get(comp, 1) < 0.05 and all(x > 0 for x in e["per_seed"].values()) and e["did"] >= PASS_MEAN)
            e["passes"] = ok
            out[f"H4 vs {comp}"] = (f"{'PASS' if ok else 'fail'}: DiD {e['did']:+.3f} [{e['ci'][0]:+.3f},"
                                    f"{e['ci'][1]:+.3f}] (CI {'excludes' if e['ci'][0] > 0 else 'includes'} 0), Holm p "
                                    f"{holm4.get(comp, float('nan')):.4f}; D 4-10 {e['delta']['kk4-10']:+.3f}, 11-16 "
                                    f"{e['delta']['kk11-16']:+.3f}, 17-20 {e['delta']['kk17-20']:+.3f}; slope/person "
                                    + (f"{e['slope']:+.4f}" if e.get("slope") is not None else "-")
                                    + f"; chance-normalised DiD {e['did_norm']:+.3f}")
    h4 = {comp: bool(H4.get(comp, {}).get("passes")) for comp in ("TFLM", "GFP")}
    g3 = C.get(("ref-G3", PRIMARY))
    g3m = g3.get("mean") if g3 else None
    # headline readings
    if passed["H1"] and passed["H2"] and h4["TFLM"] and h4["GFP"]:
        out["reading"] = ("STRONG 'Tree Forcing holds'" + (
            "; TFL - G3-c4 on 11-20 " + (f"{g3m:+.3f} >= 0: recipe candidate" if g3m is not None and g3m >= 0 else
                                         (f"{g3m:+.3f} < 0: mechanism claim only, recipe stays G3" if g3m is not None
                                          else "missing (G3): recipe status open"))))
    elif passed["H1"] and passed["H2"]:
        out["reading"] = "H1 and H2 pass, H4 fails (or not both): in-range benefit only, no extrapolation claim"
    elif passed["H1"] or passed["H2"]:
        k, other, comp = ("H1", "H2", "GFP") if passed["H1"] else ("H2", "H1", "TFLM")
        if not (h[other] and "mean" in h[other]):           # the other test has no data yet: no prereg reading
            out["reading"] = f"{k} passes; {other} pending ({comp} or a paired seed missing): no reading until it is in"
        elif k == "H1":
            out["reading"] = ("H1 passes, H2 fails: structure matters within the local format; 'locality' dropped "
                              "(GFP >= TFL, compute not excluded)" + ("" if h4["TFLM"] else
                                                                       "; H4 vs TFLM not passed: in-range only"))
        else:
            out["reading"] = ("H2 passes, H1 fails: 'decomposed result practice', structure claim dropped"
                              + ("" if h4["GFP"] else "; H4 vs GFP not passed: in-range only"))
    else:
        out["reading"] = "neither H1 nor H2 passes" + ("" if all(h[k] and "mean" in h[k] for k in ("H1", "H2"))
                                                       else " (not all testable yet)")
    # weak reading: per failed comparator, |D| < .03 or mixed signs, but TFL clicks earlier in every paired seed and
    # has a higher pooled diagA AUC
    for k, comp in (("H1", "TFLM"), ("H2", "GFP")):
        c = h[k]
        if passed[k] or not c or "mean" not in c or not c.get("per_seed"):
            continue
        xs = list(c["per_seed"].values())
        if abs(c["mean"]) < NULL_BAND or (min(xs) <= 0 < max(xs)):
            seeds = c["seeds"]
            earlier = all(_click_key(runs["TFL"][s]) < _click_key(runs[comp][s]) for s in seeds)
            auc = [runs["TFL"][s]["auc"] - runs[comp][s]["auc"] for s in seeds
                   if runs["TFL"][s]["auc"] is not None and runs[comp][s]["auc"] is not None]
            higher = bool(auc) and sum(auc) / len(auc) > 0
            out[f"weak vs {comp}"] = (("WEAK 'per-update speed-up' (no extrapolation, no structure claim)" if earlier
                                       and higher else "no weak reading") + f": clicks earlier in every seed "
                                      f"{'yes' if earlier else 'no'}, pooled AUC diff "
                                      + (f"{sum(auc) / len(auc):+.3f}" if auc else "-"))
    # all four arms within +-.03 of G3-c4 -> Tree Forcing retired
    within = {a: C.get((f"vsG3-{a}", PRIMARY), {}).get("mean") for a in ("TFL", "TFA", "TFLM", "GFP")}
    if all(v is not None for v in within.values()):
        out["vs G3"] = ("RETIRED: all four arms within +-.03 of G3-c4 (recipe stays G3)"
                        if all(abs(v) < NULL_BAND for v in within.values()) else "not all within +-.03 of G3-c4") + \
            " (" + ", ".join(f"{a} {v:+.3f}" for a, v in within.items()) + ")"
    else:
        out["vs G3"] = "four-arm check vs G3-c4 not decidable (missing: " + \
            ", ".join(a for a, v in within.items() if v is None) + ")"
    # negative result
    tfl = runs.get("TFL", {})
    never = [s for s, r in tfl.items() if r["clicked"] is False]
    neg = (g3m is not None and g3m <= -NEG_GAP) or len(never) >= 2
    out["negative"] = (("NEGATIVE: local-only node training hurts the one-pass final" if neg else "no") +
                       f" (TFL - G3-c4 on 11-20 " + (f"{g3m:+.3f}" if g3m is not None else "-") +
                       f"; TFL never clicked in {len(never)}/{len(tfl)} seed(s))")
    # H3 descriptive rule
    c = h["H3"]
    if c and c.get("per_seed"):
        xs = list(c["per_seed"].values())
        ok = len(xs) >= 2 and all(x >= H3_GAP for x in xs)
        out["H3"] = (("ancestor context helps (path information beyond locality; compute not excluded)" if ok else
                      "no TFA > TFL by >= .03 in both seeds") + f": TFA-TFL {c['mean']:+.3f} [{c['ci'][0]:+.3f},"
                     f"{c['ci'][1]:+.3f}], two-sided p {c['p_two']:.3f}, per seed "
                     + ", ".join(f"s{s} {x:+.3f}" for s, x in c["per_seed"].items()))
    else:
        out["H3"] = "not testable (TFA or TFL missing / no paired seed)"
    # no-harm: pooled TFL - G3-c4 at the same seeds on BBH and JevBench >= -.02
    seeds = sorted(set(tfl) & set(runs.get("G3", {})))
    if seeds:
        nh = {k: sum(tfl[s][k] - runs["G3"][s][k] for s in seeds) / len(seeds) for k in ("bbh", "jevbench")
              if all(tfl[s][k] is not None and runs["G3"][s][k] is not None for s in seeds)}
        bad = [k for k, v in nh.items() if v < NO_HARM]
        out["no-harm"] = ("FAILS: not adoptable for the release" if bad else "holds") + \
            f" (seeds {seeds}: " + ", ".join(f"{k} {v:+.3f}" for k, v in nh.items()) + ")"
    else:
        out["no-harm"] = "not decidable (no TFL/G3 seed in common)"
    # intention to treat: sensitivity without never-clicked seeds (decides nothing)
    for k in ("H1", "H2"):
        c = h[k]
        if c and c.get("per_seed"):
            a1, a2 = c["contrast"].split("-")
            drop = [s for s in c["seeds"] if runs[a1][s]["clicked"] is False or runs[a2][s]["clicked"] is False]
            if drop:
                keep = [c["per_seed"][s] for s in c["seeds"] if s not in drop]
                out[f"{k} sensitivity"] = (f"without never-clicked seeds {drop}: "
                                           + (f"mean {sum(keep) / len(keep):+.3f} (n {len(keep)})" if keep else
                                              "no seed left") + " (decides nothing)")
    return out


def _click_key(r: dict) -> float:
    return r["click_step"] if r["clicked"] else math.inf


# ---------------------------------------------------------------- output
def f3(x, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:+.3f}" if sign else f"{x:.3f}"


def fclick(r: dict) -> str:
    if r["clicked"] is None:
        return "-"
    if not r["clicked"]:
        return f"never (>{r['click_step']})"
    return (f"{r['click_epoch']:.2f}" if isinstance(r["click_epoch"], (int, float)) else "end") + f" @{r['click_step']}"


def _row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(head, rows):
    return [_row(head), _row(["---"] * len(head))] + [_row(r) for r in rows]


def _mm(rs, k):
    xs = [r[k] for r in rs if r.get(k) is not None]
    if not xs:
        return "-"
    return f3(sum(xs) / len(xs)) + (f" ({min(xs):.3f}-{max(xs):.3f})" if len(xs) > 1 else "")


def report(runs, unused, skipped, contrasts, H4, holm1, holm4, verdicts, alias, picks, prereg_note) -> list[str]:
    L = ["# Tree Forcing readout", "",
         "alias (stand-ins): " + (", ".join(f"{k} -> {v}" for k, v in alias.items()) or "none"),
         "variant per arm: " + ", ".join(
             f"{a} = {next(iter(runs[a].values()))['tag'] or '-'}/{next(iter(runs[a].values()))['data']}"
             + (" (--pick)" if a in picks else "") for a in ARMS if a in runs),
         "missing arms: " + (", ".join(a for a in ARMS if a not in runs) or "none"),
         "missing registered seeds: " + (", ".join(f"{a} s{s}" for a in ARMS for s in REGISTERED[a]
                                                   if s not in runs.get(a, {})) or "none"),
         f"rules: {prereg_note}", "",
         "## Per run (kk 11-20 = PRIMARY; click = first diagA [diag_A_mid] >= .40 as epoch @step; AUC = mean diagA)"]
    head = ["arm", "seed", "run", "kk 11-20", "dA", "kk 4-10", "kk 11-16", "kk 17-20", "local", "bbh", "jevbench",
            "folio", "val_facts", "train h", "click", "AUC"]
    rows = []
    for a in ARMS:
        rs = runs.get(a, {})
        for s in sorted(rs):
            r = rs[s]
            lab = r["name"].split("-Qwen")[0] + "/" + r["data"].replace("student_", "")
            rows.append([a, s, lab, f3(r["kk11-20"]), f3(r["dA"]), f3(r["kk4-10"]), f3(r["kk11-16"]),
                         f3(r["kk17-20"]), f3(r["local"].get("all")), f3(r["bbh"]), f3(r["jevbench"]), f3(r["folio"]),
                         f3(r["val_facts"]), f"{r['hours']:.2f}" if r["hours"] else "-", fclick(r), f3(r["auc"])])
        if rs:
            v = list(rs.values())
            loc = [r["local"]["all"] for r in v if "all" in r["local"]]
            rows.append([f"**{a}**", f"mean n={len(v)}", "", _mm(v, "kk11-20"), _mm(v, "dA"), _mm(v, "kk4-10"),
                         _mm(v, "kk11-16"), _mm(v, "kk17-20"), f3(sum(loc) / len(loc)) if loc else "-", _mm(v, "bbh"),
                         _mm(v, "jevbench"), _mm(v, "folio"), _mm(v, "val_facts"), _mm(v, "hours").split(" ")[0],
                         f"clicked {sum(bool(r['clicked']) for r in v)}/{len(v)}", _mm(v, "auc")])
    L += _table(head, rows)
    allr = [r for a in ARMS for r in runs.get(a, {}).values()]
    ns = sorted({n for r in allr for n in r["people_acc"]})
    if ns:
        L += ["", "## kk_deep accuracy per people count (trained on 4-10)"]
        rows = [[f"{r['arm']} s{r['seed']}"] + [f"{r['people_acc'][n]:.2f}" if n in r["people_acc"] else "-"
                                                for n in ns] for r in allr]
        for a in ARMS:
            if a in runs:
                rows.append([f"**{a} mean**"] + [
                    f"{np.mean([r['people_acc'][n] for r in runs[a].values() if n in r['people_acc']]):.2f}"
                    if any(n in r["people_acc"] for r in runs[a].values()) else "-" for n in ns])
        L += _table(["arm s"] + [f"d{n}" for n in ns], rows)
        L.append("items per people count: " + ", ".join(f"d{n} {allr[0]['people_n'].get(n, 0)}" for n in ns))
    loc = [r for r in allr if r["local"]]
    L += ["", "## Local-step probe (local_kk; local_kk_anc if that is the key), by node depth"]
    if loc:
        L += _table(["arm s", "key", "n", "all"] + [b[0] for b in LOCAL_BINS],
                    [[f"{r['arm']} s{r['seed']}", r["local_key"], r["local_n"], f3(r["local"].get("all"))]
                     + [f3(r["local"].get(b[0])) for b in LOCAL_BINS] for r in loc])
    else:
        L.append("no run has a local_kk / local_kk_anc eval")
    L += ["", "## Contrasts (paired by seed label; pooled paired bootstrap over items [kk_deep] / programs [dA]; "
              "p = share of draws <= 0)"]
    rows = []
    for c in contrasts:
        lab = c["label"]
        if "mean" not in c:
            rows.append([lab, c["contrast"], c["endpoint"], "-", "missing: " + ", ".join(c.get("missing", [])),
                         "", "", "", ""])
            continue
        a1, a2 = c["contrast"].split("-")
        per = (", ".join(f"s{s} {c['per_seed'][s]:+.3f} [{c['per_seed_ci'][s][0]:+.3f},{c['per_seed_ci'][s][1]:+.3f}]"
                         for s in c["seeds"]) if c["paired"] else
               f"unpaired arm means ({a1} {c['only1']} vs {a2} {c['only2']})")
        note = "; ".join(x for x in (
            "seed label only" if c["seed_label_only"] else "",
            f"unpaired {a1} {c['only1']} / {a2} {c['only2']}" if c["paired"] and (c["only1"] or c["only2"]) else "",
            f"{c['units_dropped']} units not in every run" if c.get("units_dropped") else "") if x)
        p = c["p_two"] if lab == "H3" else c["p_one"]
        hp = holm1.get(lab) if c["endpoint"] == PRIMARY and lab in ("H1", "H2") else None
        rows.append([lab, c["contrast"], c["endpoint"] + (" (primary)" if c["endpoint"] == PRIMARY else ""),
                     len(c.get("per_seed", {})), f3(c["mean"], True), f"[{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]",
                     ("two-sided " if lab == "H3" else "") + f"{p:.4f}" + (f" (Holm {hp:.4f})" if hp is not None
                                                                            else ""),
                     f"[{c['t_ci'][0]:+.3f},{c['t_ci'][1]:+.3f}]" if c.get("t_ci") else "-",
                     per + (f" ({note})" if note else "")])
    L += _table(["test", "contrast", "endpoint", "seeds", "mean", "boot 95%", "p", "seed t 95%",
                 "per seed [item boot 95%]"], rows)
    L += ["", "## H4 extrapolation (DiD = D(11-20) - D(4-10); only vs a comparator that passed H1/H2)"]
    if H4:
        L += _table(["vs", "seeds", "DiD", "boot 95%", "p (Holm)", "D 4-10", "D 11-16", "D 17-20", "slope/person",
                     "chance-norm DiD", "per seed DiD"],
                    [[comp, len(e["seeds"]), "-", "missing", "", "", "", "", "", "", ""] if e.get("missing") else
                     [comp, len(e["seeds"]), f3(e["did"], True), f"[{e['ci'][0]:+.3f},{e['ci'][1]:+.3f}]",
                      f"{e['p_one']:.4f} ({holm4.get(comp, float('nan')):.4f})", f3(e["delta"]["kk4-10"], True),
                      f3(e["delta"]["kk11-16"], True), f3(e["delta"]["kk17-20"], True),
                      f"{e['slope']:+.4f}" if e.get("slope") is not None else "-", f3(e["did_norm"], True),
                      ", ".join(f"s{s} {x:+.3f}" for s, x in e["per_seed"].items())] for comp, e in H4.items()])
    else:
        L.append("not run: neither H1 nor H2 passed")
    L += ["", "## Decision rules (notes/prereg_TF.md)"]
    L += [f"- **{k}**: {v}" for k, v in verdicts.items()]
    L += ["", "not used (other variant of a used arm): " +
          (", ".join(f"{r['name']} (11-20 {f3(r['kk11-20'])}, dA {r['dA']:.3f})" for r in unused) or "none"),
          f"skipped dirs: {len(skipped)} (not one of the arms, other training data, or no preds_diag)"]
    return L


def prereg_status() -> str:
    if not PREREG.exists():
        return "notes/prereg_TF.md ABSENT; rules as hard-coded in tf_readout.decide()"
    txt = PREREG.read_bytes()
    md5 = hashlib.md5(txt).hexdigest()
    s = "notes/prereg_TF.md, hard-coded in tf_readout.decide()"
    if md5 != PREREG_MD5:
        s += f" -- WARNING: the file changed since the rules were transcribed (md5 {md5[:8]} != {PREREG_MD5[:8]})"
    if b"<COMMIT>" in txt or b"<UTC>" in txt:
        s += " -- note: the registration still has placeholders (<UTC>/<COMMIT>)"
    return s


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run directories or glob patterns (quote them)")
    ap.add_argument("--alias", action="append", default=[], metavar="OLD=NEW", help="rename an arm (stand-ins)")
    ap.add_argument("--pick", action="append", default=[], metavar="ARM=tag[/data]", help="variant used for an arm")
    ap.add_argument("--data-re", default=r"^student_v3", help="training data (from the dir name) to accept")
    ap.add_argument("--boot", type=int, default=10000, help="pooled bootstrap draws (prereg_TF: 10,000)")
    ap.add_argument("--boot-seed", type=int, default=2000, help="draws for each per-seed CI")
    ap.add_argument("--out", default="results/tf_readout/tf_readout", help="writes <out>.json and <out>.md")
    args = ap.parse_args(argv)
    alias = dict(a.split("=", 1) for a in args.alias)
    picks = dict(p.split("=", 1) for p in args.pick)
    runs, unused, skipped = collect(args.runs, alias, picks, args.data_re)
    contrasts, C = [], {}
    for lab, a1, a2 in CONTRASTS:
        for ep in ENDPOINTS:
            c = contrast(runs, a1, a2, ep, args.boot, args.boot_seed)
            c["label"] = lab
            contrasts.append(c)
            C[(lab if lab != "ref" else f"ref-{a2}", ep)] = c
    for a in ("TFL", "TFA", "TFLM", "GFP"):           # the four-arm check against G3-c4 (primary only)
        C[(f"vsG3-{a}", PRIMARY)] = contrast(runs, a, "G3", PRIMARY, args.boot, args.boot_seed)
    holm1 = holm({k: C[(k, PRIMARY)]["p_one"] for k in ("H1", "H2") if "p_one" in C[(k, PRIMARY)]}, m=2)
    H4 = {}
    for k, comp in (("H1", "TFLM"), ("H2", "GFP")):
        if test_passes(C[(k, PRIMARY)], holm1.get(k)):
            H4[comp] = did(runs, "TFL", comp, args.boot)
    holm4 = holm({comp: e["p_one"] for comp, e in H4.items() if "p_one" in e})
    verdicts = decide(runs, C, H4, holm1, holm4)
    lines = report(runs, unused, skipped, contrasts, H4, holm1, holm4, verdicts, alias, picks, prereg_status())
    print("\n".join(lines))
    out = {"alias": alias, "picks": picks, "prereg": prereg_status(), "verdicts": verdicts, "holm_h1h2": holm1,
           "holm_h4": holm4, "h4": H4,
           "runs": {a: {s: {k: v for k, v in r.items() if not k.startswith("_")} for s, r in rs.items()}
                    for a, rs in runs.items()},
           "not_used": [r["dir"] for r in unused], "skipped": skipped, "contrasts": contrasts}
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        Path(str(p) + ".json").write_text(json.dumps(out, indent=1, default=str))
        Path(str(p) + ".md").write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    main()
