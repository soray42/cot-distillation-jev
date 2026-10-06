"""Tree Forcing readout (notes/prereg_TF.md, "FINAL RECONCILIATION" + item 7 addendum): arms A2 / G3 / TFL / GFPS / GFP /
TFLM / TFLN / TFA on K&K. CPU only, reads local copies of HPC run directories, numpy only.

Co-primary family (Holm, m = 2, fixed), on the PRIMARY endpoint kk_deep 11-20 (1,300 items), one-sided:
  H2'  TFL - GFPS     locality (GFPS: every node with the statement lines only). If no GFPS run exists (GFPS failed its
                      pre-launch verification) H2' falls back to TFL - GFP and every line that uses it says FALLBACK.
  H5   TFL - A2       vs answers only; the registered comparator is A2-c12 on student_v3sl (12 epochs, about TFL's GPU
                      time). When no A2-c12/v3sl run exists the A2 arm is filled with the most-runs A2 variant (A2-c8/v3)
                      as a labelled REFERENCE: H5 is then computed but can never pass.
  Pass (both): Holm p < .05 from the pooled item-paired bootstrap, every paired seed > 0, pooled mean >= +.03.
  Fewer than 3 paired seeds (TFL/GFPS/A2 are registered at s12-14): the verdict is marked "provisional".
Report-only (not in the Holm family): TFL-TFLM (label noise), TFL-TFLN (nothing stated), TFL-GFP (options in node
  views), TFA-TFL (two-sided), TFL-G3 (reference, G3-c4 v3f).
Endpoints of every contrast: kk 11-20 (primary), E1' = mean(R1, C) per diag program, dA = mean(R1, R2, C), R2 alone,
  kk 4-10, 11-16, 11-16 split by the option-majority shortcut stratum (sc = shortcut-valid, ns = the rest), 17-20 (shown
  with the unseen-name share).
Option-majority shortcut stratum (notes/next_round/tree_forcing/redteam.md F2, its kkshort.py logic; computed here from the
  kk_deep eval items, --kkdeep-items, joined to preds_kkdeep.jsonl by item_id): parse the six options "X) Name: role, ..."
  of the prompt; per person take the majority role over the 6 options, a 3-3 tie counted as knave; the heuristic picks
  the option(s) at minimum Hamming distance to that majority vector; an item is shortcut-valid iff that argmin set is
  exactly {gold}. This reproduces redteam F2 exactly on data/eval/kk_deep.jsonl: shortcut-valid 226/900 at 11-16 and
  109/400 at 17-20; heuristic accuracy (ties split evenly) .179 / .206 / .340 / .355 at 4-6 / 7-10 / 11-16 / 17-20.
  Unseen-name share: persons whose name is not in the 16 training names (cotdistill.sources.NAMES); .333 at 17-20.
DiD extrapolation interaction (report-only, no Holm, no gate): D = TFL - X pooled over paired seeds, DiD = D(11-20) -
  D(4-10), and the ready.md form D(11-16 ns) - D(4-10); items resampled within bins, one-sided p; plus D per bin, the
  slope of D on people (4-20) and the chance-normalised DiD (D / (X - 1/6)); X = the H2' comparator, A2, G3.
local_kk/derived_nontrivial (local-step probe, eval key local_kk or local_kk_anc) against the .797 lookup baseline
  (redteam F4: template lookup on derived local views); in-format for TFL/TFLM/TFLN/TFA only.
Learning speed: click = first diagA (diag_A_mid) >= .40 over the curve points plus the end evaluation, censored at the
  last step if never; AUC = mean diagA over those points (prereg_T 10-01 18:39 / 23:45).
No-harm: pooled TFL - G3-c4 at the same seeds on BBH (macro) and JevBench >= -.02, else not adoptable for the release.
Wording: the item 5 table of the FINAL RECONCILIATION, hard-coded in decide() (the file is prose); PREREG_MD5 is the
  version transcribed and a changed file is flagged. Removed with the reconciliation (the older H1/H2/H3/H4 rules, the
  "weak / faster learning only" reading, the four-arm "retired vs G3" rule and the G3-based negative rule); that
  behaviour is in git at f810934 (scripts/tf_readout.py), not behind a flag.

Arm and variant: a run dir is <ARM>[-<tag>]-<model>-<data>-s<seed> (hpc/train_arm.sbatch OUT; NODEW unset = no -nw
suffix). One (tag, data) variant per arm: --pick ARM=tag[/data]; else A2 takes c12/student_v3sl when present, and every
arm otherwise the variant with the most runs, ties to the most training hours; the others are listed as "not used".
--alias OLD=NEW renames an arm; --standin NEW=OLD copies OLD's runs as NEW and keeps OLD (stand-in tests only).
Pairing is by seed label (A2 ignores --order-seed, so pairs with A2 share the label only). H2'/H5 use the registered
seeds s12-s14 only (an extra seed such as s11 is a new registration, never pooled); Holm takes only seed-paired tests.

  python scripts/tf_readout.py 'results/hpc/tf/*' 'results/hpc/w1/*' --out results/tf_readout/tf
  python scripts/tf_readout.py 'results/hpc/w1/*' 'results/hpc/c4v3f/*' --alias TF=TFL --alias TFM=TFLM \
      --standin GFPS=G3 --out results/tf_readout/standin_w1_c4v3f      # stand-in test on exploratory runs
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation_table import t_q, train_hours  # noqa: E402
from analyze_e2 import correct, diag_table, primary, read_preds, run_summary  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cotdistill.sources import NAMES as TRAIN_NAMES  # noqa: E402  (the 16 names K&K training uses)

PREREG = ROOT / "notes" / "prereg_TF.md"
PREREG_MD5 = "df083b08b17a2e76029d27cb454375c4"      # FINAL RECONCILIATION + item 7 addendum (13:50 CEST, before any run)
KKDEEP_ITEMS = ROOT / "data" / "eval" / "kk_deep.jsonl"
ARMS = ["A2", "G3", "TFL", "GFPS", "GFP", "TFLM", "TFLN", "TFA"]
REGISTERED = {"TFL": [12, 13, 14], "GFPS": [12, 13, 14], "A2": [12, 13, 14], "GFP": [12, 13], "TFLM": [12, 13],
              "TFLN": [12, 13], "TFA": [12, 13], "G3": [12, 13, 14]}      # item 3 (G3-c4 v3f: reference)
A2_REGISTERED = ("c12", "student_v3sl")
REPORT_ONLY = [("TFL", "TFLM"), ("TFL", "TFLN"), ("TFL", "GFP"), ("TFA", "TFL"), ("TFL", "G3")]
BINS = {"4-10": (4, 10), "11-16": (11, 16), "17-20": (17, 20), "11-20": (11, 20)}
PRIMARY = "kk11-20"
ENDPOINTS = [PRIMARY, "E1p", "dA", "R2", "kk4-10", "kk11-16", "kk11-16sc", "kk11-16ns", "kk17-20"]
EP_LABEL = {"E1p": "E1' mean(R1,C)", "kk11-16sc": "kk11-16 shortcut", "kk11-16ns": "kk11-16 non-shortcut"}
LOCAL_BINS = [("0", 0, 0), ("1-3", 1, 3), ("4-7", 4, 7), ("8-12", 8, 12), (">=13", 13, 10 ** 6)]
LOOKUP = 0.797                                      # redteam F4: template lookup on derived local views
IN_FORMAT = ("TFL", "TFLM", "TFLN", "TFA")
CLICK = 0.40
CHANCE = 1 / 6                                      # kk_deep: 6 options
PASS_MEAN, NO_HARM, NEG_GAP = 0.03, -0.02, 0.05     # prereg_TF items 1, 5 and "Endpoints" (no-harm)
EPS = 1e-9                                          # float slack so a pooled mean of exactly .03 / -.05 / -.02 counts
COPRIMARY_SEEDS = set(REGISTERED["TFL"])            # s12-s14; extra seeds are a new registration, never pooled
NAME = re.compile(r"^(?P<left>.+?)-(?P<model>[^-]*Qwen[^/]*?)-(?P<data>student_[A-Za-z0-9_]+)-s(?P<seed>\d+)$")
OPTION = re.compile(r"^([A-F])\) (.+)$")


# ---------------------------------------------------------------- kk_deep items: shortcut stratum, unseen names
def parse_options(prompt: str) -> dict[str, dict[str, str]]:
    out = {}
    for line in prompt.split("\n"):
        m = OPTION.match(line.strip())
        if m:
            out[m[1]] = dict(x.split(": ", 1) for x in m[2].split(", "))
    return out


def shortcut_info(item: dict) -> dict | None:
    """redteam F2 (kkshort.py): per-person majority role over the options (3-3 -> knave), heuristic = option(s) at
    minimum Hamming distance to it; shortcut-valid iff the argmin set is exactly {gold}."""
    opts = parse_options(item.get("prompt", ""))
    if len(opts) < 2 or item.get("gold_label") not in opts:
        return None
    names = list(opts[next(iter(opts))])
    maj = {}
    for nm in names:
        c = Counter(o.get(nm) for o in opts.values())
        maj[nm] = "knight" if c["knight"] > c["knave"] else "knave"
    dist = {L: sum(o.get(nm) != maj[nm] for nm in names) for L, o in opts.items()}
    best = [L for L, d in dist.items() if d == min(dist.values())]
    return {"shortcut": best == [item["gold_label"]], "heur_acc": (item["gold_label"] in best) / len(best),
            "unseen": sum(nm not in TRAIN_NAMES for nm in names) / len(names), "persons": len(names)}


def load_kkdeep_items(path: Path | None) -> dict[str, dict]:
    """item_id -> shortcut_info (empty if the file is absent)."""
    if path is None or not Path(path).exists():
        return {}
    out = {}
    with open(path) as f:
        lines = [line for line in f if line.strip()]
    for line in lines:
        if line:
            it = json.loads(line)
            info = shortcut_info(it)
            if info is not None:
                out[it["item_id"]] = dict(info, people=people(it.get("group")))
    return out


def stratum_summary(items: dict) -> dict:
    out = {}
    for lab, lo, hi in (("4-6", 4, 6), ("7-10", 7, 10), ("11-16", 11, 16), ("17-20", 17, 20)):
        xs = [v for v in items.values() if v["people"] is not None and lo <= v["people"] <= hi]
        if xs:
            out[lab] = {"n": len(xs), "shortcut": sum(v["shortcut"] for v in xs),
                        "heur_acc": sum(v["heur_acc"] for v in xs) / len(xs),
                        "unseen": sum(v["unseen"] for v in xs) / len(xs),
                        "items_with_unseen": sum(v["unseen"] > 0 for v in xs)}
    return out


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
            if "derived_nontrivial" in m:
                out["derived_nontrivial"] = m["derived_nontrivial"]["acc"]
                out["derived_nontrivial_n"] = m["derived_nontrivial"]["n"]
            for lab, lo, hi in LOCAL_BINS:
                cells = [(v["n"], v["acc"]) for k, v in m.items()
                         if k.startswith("depth/") and lo <= int(k.split("/")[1]) <= hi]
                n = sum(c[0] for c in cells)
                if n:
                    out[lab] = sum(c[0] * c[1] for c in cells) / n
            return key, out, m.get("all", {}).get("n")
    return None, {}, None


def load_run(d: Path, info: dict, items: dict) -> dict:
    metrics = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
    s = run_summary(d)
    t = diag_table(d)
    units = {"dA": {g: primary(t, [g]) for g in t},
             "E1p": {g: (t[g]["R1"]["correct"] + t[g]["C"]["correct"]) / 2 for g in t if "R1" in t[g] and "C" in t[g]},
             "R2": {g: float(t[g]["R2"]["correct"]) for g in t if "R2" in t[g]}}
    kk = kkdeep_units(d)
    by_n: dict[int, list[int]] = defaultdict(list)
    for n, c in kk.values():
        by_n[n].append(c)
    for name, (lo, hi) in BINS.items():
        units["kk" + name] = {i: float(c) for i, (n, c) in kk.items() if lo <= n <= hi}
    for b in ("11-16", "17-20"):
        units[f"kk{b}sc"] = {i: c for i, c in units["kk" + b].items() if i in items and items[i]["shortcut"]}
        units[f"kk{b}ns"] = {i: c for i, c in units["kk" + b].items() if i in items and not items[i]["shortcut"]}
    units["people"] = {i: n for i, (n, c) in kk.items()}
    u1720 = [items[i]["unseen"] for i in units["kk17-20"] if i in items]
    lkey, loc, ln = local_probe(metrics)
    r = dict(info, dir=str(d), name=d.name, dA=s["A"], R1=s["R1"], R2=s["R2"], C=s["C"], E1p=(s["R1"] + s["C"]) / 2,
             bbh=s.get("bbh"), jevbench=s.get("jevbench"), folio=s.get("folio"), val_facts=s.get("val_facts"),
             hours=train_hours({"metrics": metrics}), **curve(metrics),
             people_acc={n: sum(v) / len(v) for n, v in sorted(by_n.items())},
             people_n={n: len(v) for n, v in sorted(by_n.items())}, local_key=lkey, local=loc, local_n=ln,
             kk_unjoined=sum(i not in items for i in kk) if items else None,
             unseen17_20=sum(u1720) / len(u1720) if u1720 else None)
    for name in [k for k in units if k.startswith("kk")]:
        u = units[name]
        r[name] = sum(u.values()) / len(u) if u else None
        r[name + "_n"] = len(u)
    r["_units"] = units
    return r


def collect(patterns: list[str], alias: dict, standin: dict, picks: dict, data_re: str, items: dict
            ) -> tuple[dict, list, list, dict]:
    """-> ({arm: {seed: run}}, not-used runs, skipped dirs, {arm: variant note})."""
    dirs = sorted({Path(p) for pat in patterns for p in (glob.glob(pat) or [pat])})
    found, skipped = [], []
    for d in dirs:
        info = parse_name(d.name) if d.is_dir() else None
        if info is None or not re.search(data_re, info["data"]) or not (d / "preds_diag.jsonl").exists():
            skipped.append((str(d), "name/data/preds_diag"))
            continue
        info["orig_arm"] = info["arm"]
        info["arm"] = alias.get(info["arm"], info["arm"])
        copies = [new for new, old in standin.items() if old == info["arm"]]
        if info["arm"] not in ARMS and not copies:
            skipped.append((str(d), f"arm {info['arm']} not in {'/'.join(ARMS)}"))
            continue
        r = load_run(d, info, items)
        if info["arm"] in ARMS:
            found.append(r)
        for new in copies:
            found.append(dict(r, arm=new, standin_of=info["arm"]))
    runs: dict[str, dict[int, dict]] = {}
    unused, notes = [], {}
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
        elif arm == "A2" and A2_REGISTERED in variants:
            key = A2_REGISTERED
        else:
            key = max(variants, key=lambda k: (len(variants[k]), np.mean([r["hours"] or 0 for r in variants[k]]), k))
        if arm == "A2":
            notes[arm] = ("registered A2-c12/v3sl" if key == A2_REGISTERED else
                          f"REFERENCE A2-{key[0] or '-'}/{key[1].replace('student_', '')}, not the registered "
                          "A2-c12/v3sl: H5 cannot pass on it")
        for k, v in variants.items():
            if k == key:
                dup = [r["seed"] for r in v if sum(x["seed"] == r["seed"] for x in v) > 1]
                if dup:
                    raise SystemExit(f"{arm} {key}: duplicate seed(s) {sorted(set(dup))}")
                runs[arm] = {r["seed"]: r for r in v}
            else:
                unused += [x for x in v if not x.get("standin_of")]
    return runs, unused, skipped, notes


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
    sets = [set(r["_units"].get(endpoint, {})) for r in allruns]
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


def contrast(runs: dict, a1: str, a2: str, endpoint: str, boot: int, boot_seed: int,
             seeds_allowed: set | None = None) -> dict:
    """seeds_allowed: the co-primary tests use the registered seeds only ("no extra seeds under this registration";
    an unregistered seed, e.g. s11, is listed in `excluded_seeds` and never pooled)."""
    s1, s2 = runs.get(a1, {}), runs.get(a2, {})
    excluded = []
    if seeds_allowed is not None:
        excluded = sorted((set(s1) | set(s2)) - set(seeds_allowed))
        s1 = {s: r for s, r in s1.items() if s in seeds_allowed}
        s2 = {s: r for s, r in s2.items() if s in seeds_allowed}
    res = {"contrast": f"{a1}-{a2}", "endpoint": endpoint, "seeds": sorted(set(s1) & set(s2)),
           "only1": sorted(set(s1) - set(s2)), "only2": sorted(set(s2) - set(s1)), "seed_label_only": "A2" in (a1, a2),
           "excluded_seeds": excluded}
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
    """DiD = D(hi) - D(4-10), D = a1 - a2 pooled over paired seeds, items resampled within bins; hi = 11-20 and
    11-16 non-shortcut (ready.md). Report-only."""
    s1, s2 = runs.get(a1, {}), runs.get(a2, {})
    seeds = sorted(set(s1) & set(s2))
    res = {"contrast": f"{a1}-{a2}", "seeds": seeds}
    if not seeds:
        res["missing"] = True
        return res
    p1, p2 = {s: s1[s] for s in seeds}, {s: s2[s] for s in seeds}
    parts, per_seed = {}, defaultdict(dict)
    for b in ("kk11-20", "kk4-10", "kk11-16", "kk17-20", "kk11-16ns"):
        keys, vec, _ = _vectors(p1, p2, b)
        if not keys:
            if b == "kk11-16ns":                      # no stratum (no items file): that DiD is skipped
                continue
            res["missing"] = True
            return res
        per = {s: vec(p1[s]) - vec(p2[s]) for s in seeds}
        parts[b] = (np.mean(list(per.values()), axis=0),
                    float(np.mean([vec(p2[s]).mean() for s in seeds])))      # comparator accuracy in the bin
        for s, v in per.items():
            per_seed[s][b] = float(v.mean())
    lo = parts["kk4-10"][0]
    blo = boot_means(lo, boot, _rng(a1, a2, "did-lo"))
    res["delta"] = {b: float(parts[b][0].mean()) for b in parts}
    for hi_b, tag in (("kk11-20", ""), ("kk11-16ns", "_ns")):
        if hi_b not in parts:
            continue
        hi = parts[hi_b][0]
        bs = boot_means(hi, boot, _rng(a1, a2, "did-hi", hi_b)) - blo
        res.update({f"did{tag}": float(hi.mean() - lo.mean()), f"ci{tag}": ci95(bs),
                    f"p_one{tag}": float((bs <= 0).mean()),
                    f"per_seed{tag}": {s: v[hi_b] - v["kk4-10"] for s, v in per_seed.items()}})
    res["did_norm"] = float(parts["kk11-20"][0].mean() / (parts["kk11-20"][1] - CHANCE)
                            - lo.mean() / (parts["kk4-10"][1] - CHANCE))
    keys, _, _ = _vectors(p1, p2, "people")           # slope of D on people, 4-20 (OLS over items, seed-averaged)
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


# ---------------------------------------------------------------- prereg_TF FINAL RECONCILIATION
def h2_comparator(runs: dict) -> tuple[str, bool]:
    """-> (comparator, fallback): GFPS, or GFP when no GFPS run exists (item 1)."""
    if "GFPS" not in runs and "GFP" in runs:
        return "GFP", True
    return "GFPS", False


def test_passes(c: dict | None, holm_p: float | None) -> bool:
    """Item 2: Holm p < .05 from the pooled item-paired bootstrap, every paired seed > 0, pooled mean >= +.03."""
    return bool(c and holm_p is not None and c.get("per_seed") and holm_p < 0.05
                and all(x > 0 for x in c["per_seed"].values()) and c["mean"] >= PASS_MEAN - EPS)


def _signs(c: dict) -> str:
    return " ".join(f"s{s}{'+' if x > 0 else '-'}" for s, x in c.get("per_seed", {}).items()) or "(unpaired)"


def decide(runs: dict, C: dict, holm1: dict, a2_registered: bool, comp2: str, fallback: bool) -> dict:
    """FINAL RECONCILIATION items 1, 2, 4 and 5 (+ no-harm from "Endpoints"). Returns {rule: text}."""
    out, passed, testable = {}, {}, {}
    for k, comp in (("H2'", comp2), ("H5", "A2")):
        c = C.get((k, PRIMARY))
        ref = k == "H5" and not a2_registered and "A2" in runs
        testable[k] = bool(c and "mean" in c and c.get("per_seed")) and not ref
        passed[k] = test_passes(c, holm1.get(k)) and testable[k]
        lab = f"TFL-{comp}" + (" [FALLBACK: no GFPS run, H2' = TFL-GFP]" if k == "H2'" and fallback else "")
        if ref:
            out[k] = ("not testable as registered: no A2-c12/v3sl run; against the REFERENCE " + notes_a2(runs)
                      + (f": {c['mean']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}], seeds {_signs(c)} "
                         f"(n {len(c.get('per_seed', {}))}), decides nothing" if c and "mean" in c else ": no data"))
            continue
        if not testable[k]:
            out[k] = f"not testable: {lab} " + ("has no paired seed" if c and "mean" in c else "(arm missing)") + (
                f"; unregistered seed(s) {c['excluded_seeds']} not used" if c and c.get("excluded_seeds") else "")
            continue
        n = len(c["per_seed"])
        prov = f"; provisional: {n} of 3 registered seeds paired" if n < 3 else ""
        if c.get("excluded_seeds"):
            prov += f"; unregistered seed(s) {c['excluded_seeds']} not pooled"
        crit_ok = test_passes(c, holm1.get(k))
        out[k] = (f"{'PASS' if passed[k] else 'fail'}: {lab} {c['mean']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}], "
                  f"Holm p {holm1.get(k, float('nan')):.4f}, seeds {_signs(c)} (n {n}){prov}"
                  + ("" if crit_ok else "; criteria not met: " + ", ".join(
                      x for x, bad in (("Holm p >= .05", holm1.get(k, 1) >= 0.05),
                                       ("a paired seed <= 0", any(v <= 0 for v in c["per_seed"].values())),
                                       ("pooled mean < +.03", c["mean"] < PASS_MEAN)) if bad)))
    # negative result (item 5): TFL below GFPS (H2' comparator) or A2-c12 by >= .05 pooled
    neg = []
    for k, comp in (("H2'", comp2), ("H5", "A2")):
        c = C.get((k, PRIMARY))
        if c and "mean" in c and c["mean"] <= -NEG_GAP + EPS and (k != "H5" or a2_registered):
            neg.append(f"{comp} (TFL-{comp} {c['mean']:+.3f})")
    # item 5 wording table
    if passed["H2'"] and passed["H5"]:
        w = "'Tree Forcing' (local premise views beat full-problem node supervision and answers-only training)"
    elif passed["H5"] and testable["H2'"]:
        w = "H2' fails, H5 passes: 'dense node supervision', not Tree Forcing"
    elif testable["H5"] and not passed["H5"]:
        w = "H5 fails: no method claim on K&K"
    else:
        w = "pending: " + ", ".join(k for k in ("H2'", "H5") if not testable[k]) + " not testable yet; no wording"
    if neg:
        w += "; NEGATIVE RESULT: TFL below " + ", ".join(neg) + " (<= -.05 pooled)"
    if fallback and (testable["H2'"] or any(comp2 in x for x in neg)):
        w += " [H2' on the FALLBACK comparator GFP (GFPS failed pre-launch verification); say so]"
    if any(t and len(C[(k, PRIMARY)]["per_seed"]) < 3 for k, t in testable.items()):
        w += " [provisional: fewer than 3 paired seeds]"
    out["wording"] = w
    # "...and the dependency edges are used": TFL > TFLN and TFL > TFLM, >= 3 seeds each, in a later wave
    parts = []
    for comp in ("TFLN", "TFLM"):
        c = C.get((f"TFL-{comp}", PRIMARY))
        parts.append(f"TFL-{comp} " + (f"{c['mean']:+.3f} seeds {_signs(c)} (n {len(c.get('per_seed', {}))})"
                                       if c and "mean" in c else "missing"))
    out["edges"] = ("'...and the dependency edges are used' is not licensable in this wave (needs TFL > TFLN and "
                    "TFL > TFLM with >= 3 seeds each in a later wave); report-only: " + "; ".join(parts))
    tfl = runs.get("TFL", {})
    never = [s for s, r in tfl.items() if r["clicked"] is False]
    out["negative"] = (("NEGATIVE RESULT: TFL below " + ", ".join(neg) + " (<= -.05 pooled)") if neg else "no") + \
        f" (TFL never clicked in {len(never)}/{len(tfl)} seed(s); counted, intention to treat)"
    # no-harm: pooled TFL - G3-c4 at the same seeds on BBH and JevBench >= -.02
    seeds = sorted(set(tfl) & set(runs.get("G3", {})))
    if seeds:
        nh = {k: sum(tfl[s][k] - runs["G3"][s][k] for s in seeds) / len(seeds) for k in ("bbh", "jevbench")
              if all(tfl[s][k] is not None and runs["G3"][s][k] is not None for s in seeds)}
        bad = [k for k, v in nh.items() if v < NO_HARM - EPS]
        out["no-harm"] = ("FAILS: not adoptable for the release" if bad else "holds") + \
            f" (TFL-G3, seeds {seeds}: " + ", ".join(f"{k} {v:+.3f}" for k, v in nh.items()) + ")"
    else:
        out["no-harm"] = "not decidable (no TFL/G3 seed in common)"
    # learning speed (report-only): click order per paired seed and pooled AUC difference
    for comp in (comp2, "A2"):
        ss = sorted(set(tfl) & set(runs.get(comp, {})))
        if not ss:
            continue
        earlier = ["<" if _click_key(tfl[s]) < _click_key(runs[comp][s]) else
                   ("=" if _click_key(tfl[s]) == _click_key(runs[comp][s]) else ">") for s in ss]
        auc = [tfl[s]["auc"] - runs[comp][s]["auc"] for s in ss
               if tfl[s]["auc"] is not None and runs[comp][s]["auc"] is not None]
        out[f"speed vs {comp}"] = (f"TFL click vs {comp} per seed " + " ".join(f"s{s}{e}" for s, e in zip(ss, earlier))
                                   + " (< = earlier); pooled diagA AUC diff "
                                   + (f"{sum(auc) / len(auc):+.3f}" if auc else "-") + " (report-only)")
    # intention to treat: sensitivity without never-clicked seeds (decides nothing)
    for k in ("H2'", "H5"):
        c = C.get((k, PRIMARY))
        if c and c.get("per_seed"):
            a1, a2 = c["contrast"].split("-")
            drop = [s for s in c["seeds"] if any(runs.get(a, {}).get(s, {}).get("clicked") is False for a in (a1, a2))]
            if drop:
                keep = [c["per_seed"][s] for s in c["seeds"] if s not in drop]
                out[f"{k} sensitivity"] = (f"without never-clicked seeds {drop}: "
                                           + (f"mean {sum(keep) / len(keep):+.3f} (n {len(keep)})" if keep else
                                              "no seed left") + " (decides nothing)")
    return out


def notes_a2(runs: dict) -> str:
    r = next(iter(runs["A2"].values()))
    return f"A2-{r['tag'] or '-'}/{r['data'].replace('student_', '')}"


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


def _lab(r):
    return r["name"].split("-Qwen")[0] + "/" + r["data"].replace("student_", "") + \
        (f" (as {r['arm']})" if r.get("standin_of") else "")


def report(runs, unused, skipped, contrasts, D, holm1, verdicts, alias, standin, picks, notes, strat, comp2, fallback,
           prereg_note, items_note) -> list[str]:
    L = ["# Tree Forcing readout (prereg_TF FINAL RECONCILIATION)", "",
         "alias (stand-ins): " + (", ".join([f"{k} -> {v}" for k, v in alias.items()]
                                            + [f"{old} copied as {new}" for new, old in standin.items()]) or "none"),
         "variant per arm: " + ", ".join(
             f"{a} = {next(iter(runs[a].values()))['tag'] or '-'}/{next(iter(runs[a].values()))['data']}"
             + (" (--pick)" if a in picks else "") for a in ARMS if a in runs),
         "A2: " + notes.get("A2", "absent"),
         "H2' comparator: " + (f"{comp2} (FALLBACK: no GFPS run)" if fallback else comp2),
         "missing arms: " + (", ".join(a for a in ARMS if a not in runs) or "none"),
         "missing registered seeds: " + (", ".join(f"{a} s{s}" for a in ARMS for s in REGISTERED[a]
                                                   if s not in runs.get(a, {})) or "none"),
         "unregistered seeds present (in the tables and report-only contrasts; never pooled into H2'/H5): " + (", ".join(
             f"{a} s{s}" for a in ARMS for s in sorted(runs.get(a, {})) if s not in REGISTERED[a]) or "none"),
         f"rules: {prereg_note}", f"kk_deep items: {items_note}", "",
         "## Per run (kk 11-20 = PRIMARY; E1' = mean(R1,C); click = first diagA [diag_A_mid] >= .40 as epoch @step; "
         "AUC = mean diagA)"]
    head = ["arm", "seed", "run", "kk 11-20", "E1'", "dA", "R2", "bbh", "jevbench", "folio", "val_facts", "train h",
            "click", "AUC"]
    rows = []
    for a in ARMS:
        rs = runs.get(a, {})
        for s in sorted(rs):
            r = rs[s]
            rows.append([a, s, _lab(r), f3(r["kk11-20"]), f3(r["E1p"]), f3(r["dA"]), f3(r["R2"]), f3(r["bbh"]),
                         f3(r["jevbench"]), f3(r["folio"]), f3(r["val_facts"]),
                         f"{r['hours']:.2f}" if r["hours"] else "-", fclick(r), f3(r["auc"])])
        if rs:
            v = list(rs.values())
            rows.append([f"**{a}**", f"mean n={len(v)}", "", _mm(v, "kk11-20"), _mm(v, "E1p"), _mm(v, "dA"),
                         _mm(v, "R2"), _mm(v, "bbh"), _mm(v, "jevbench"), _mm(v, "folio"), _mm(v, "val_facts"),
                         _mm(v, "hours").split(" ")[0], f"clicked {sum(bool(r['clicked']) for r in v)}/{len(v)}",
                         _mm(v, "auc")])
    L += _table(head, rows)
    L += ["", "## kk_deep bins (sc = option-majority shortcut-valid, ns = the rest; unseen = share of persons with a "
              "name outside the 16 training names)"]
    if strat:
        L.append("stratum on the eval items: " + "; ".join(
            f"{b}: shortcut-valid {v['shortcut']}/{v['n']}, heuristic acc {v['heur_acc']:.3f}, unseen {v['unseen']:.3f}"
            for b, v in strat.items()))
    head = ["arm s", "4-10", "11-16", "11-16 sc", "11-16 ns", "17-20", "17-20 sc", "17-20 ns", "unseen 17-20", "11-20"]
    rows = []
    for a in ARMS:
        for s, r in sorted(runs.get(a, {}).items()):
            rows.append([f"{a} s{s}", f3(r["kk4-10"]), f3(r["kk11-16"]), f3(r.get("kk11-16sc")),
                         f3(r.get("kk11-16ns")), f3(r["kk17-20"]), f3(r.get("kk17-20sc")), f3(r.get("kk17-20ns")),
                         f3(r["unseen17_20"]), f3(r["kk11-20"])])
        if a in runs:
            v = list(runs[a].values())
            rows.append([f"**{a} mean**"] + [_mm(v, k).split(" ")[0] for k in (
                "kk4-10", "kk11-16", "kk11-16sc", "kk11-16ns", "kk17-20", "kk17-20sc", "kk17-20ns", "unseen17_20",
                "kk11-20")])
    L += _table(head, rows)
    allr = [r for a in ARMS for r in runs.get(a, {}).values()]
    ns = sorted({n for r in allr for n in r["people_acc"]})
    if ns:
        L += ["", "## kk_deep accuracy per people count (trained on 4-10)"]
        rows = [[f"{r['arm']} s{r['seed']}"] + [f"{r['people_acc'][n]:.2f}" if n in r["people_acc"] else "-"
                                                for n in ns] for r in allr]
        L += _table(["arm s"] + [f"d{n}" for n in ns], rows)
        L.append("items per people count: " + ", ".join(f"d{n} {allr[0]['people_n'].get(n, 0)}" for n in ns))
    loc = [r for r in allr if r["local"]]
    L += ["", f"## Local-step probe (local_kk; local_kk_anc if that is the key): derived_nontrivial vs the {LOOKUP} "
              "lookup baseline, and by node depth (in-format for " + "/".join(IN_FORMAT) + " only)"]
    if loc:
        L += _table(["arm s", "key", "n", "all", "derived_nontrivial", f"- {LOOKUP}"] + [b[0] for b in LOCAL_BINS],
                    [[f"{r['arm']} s{r['seed']}" + ("" if r["arm"] in IN_FORMAT else " (transfer)"), r["local_key"],
                      r["local_n"], f3(r["local"].get("all")), f3(r["local"].get("derived_nontrivial")),
                      f3(r["local"]["derived_nontrivial"] - LOOKUP, True) if "derived_nontrivial" in r["local"]
                      else "-"] + [f3(r["local"].get(b[0])) for b in LOCAL_BINS] for r in loc])
    else:
        L.append("no run has a local_kk / local_kk_anc eval")
    L += ["", "## Contrasts (paired by seed label; pooled paired bootstrap over items [kk_deep] / programs [diag]; "
              "p = share of draws <= 0; Holm m = 2 over H2' and H5 on the primary only)"]
    rows = []
    for c in contrasts:
        lab = c["label"]
        ep = EP_LABEL.get(c["endpoint"], c["endpoint"]) + (" (primary)" if c["endpoint"] == PRIMARY else "")
        if c["endpoint"] == "kk17-20":
            us = [r["unseen17_20"] for a in c["contrast"].split("-") for r in runs.get(a, {}).values()
                  if r["unseen17_20"] is not None]
            if us:
                ep += f" (unseen names {sum(us) / len(us):.3f})"
        if "mean" not in c:
            rows.append([lab, c["contrast"], ep, "-", "missing: " + ", ".join(c.get("missing", [])),
                         "", "", "", ""])
            continue
        a1, a2 = c["contrast"].split("-")
        per = (", ".join(f"s{s} {c['per_seed'][s]:+.3f} [{c['per_seed_ci'][s][0]:+.3f},{c['per_seed_ci'][s][1]:+.3f}]"
                         for s in c["seeds"]) if c["paired"] else
               f"unpaired arm means ({a1} {c['only1']} vs {a2} {c['only2']})")
        note = "; ".join(x for x in (
            "seed label only" if c["seed_label_only"] else "",
            f"unpaired {a1} {c['only1']} / {a2} {c['only2']}" if c["paired"] and (c["only1"] or c["only2"]) else "",
            f"{c['units_dropped']} units not in every run" if c.get("units_dropped") else "",
            f"unregistered {c['excluded_seeds']} not pooled" if c.get("excluded_seeds") else "") if x)
        two = lab == "TFA-TFL"
        p = c["p_two"] if two else c["p_one"]
        hp = holm1.get(lab) if c["endpoint"] == PRIMARY and lab in ("H2'", "H5") else None
        rows.append([lab, c["contrast"], ep, len(c.get("per_seed", {})), f3(c["mean"], True),
                     f"[{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]",
                     ("two-sided " if two else "") + f"{p:.4f}" + (f" (Holm {hp:.4f})" if hp is not None else ""),
                     f"[{c['t_ci'][0]:+.3f},{c['t_ci'][1]:+.3f}]" if c.get("t_ci") else "-",
                     per + (f" ({note})" if note else "")])
    L += _table(["test", "contrast", "endpoint", "seeds", "mean", "boot 95%", "p", "seed t 95%",
                 "per seed [item boot 95%]"], rows)
    L += ["", "## DiD extrapolation interaction (report-only; D = TFL - X; DiD = D(11-20) - D(4-10) and "
              "D(11-16 ns) - D(4-10); one-sided p)"]
    rows = []
    for comp, e in D.items():
        if e.get("missing"):
            rows.append([comp, len(e["seeds"]), "missing"] + [""] * 10)
            continue
        rows.append([comp, len(e["seeds"]), f3(e["did"], True), f"[{e['ci'][0]:+.3f},{e['ci'][1]:+.3f}]",
                     f"{e['p_one']:.4f}",
                     f3(e.get("did_ns"), True) + (f" [{e['ci_ns'][0]:+.3f},{e['ci_ns'][1]:+.3f}] p {e['p_one_ns']:.4f}"
                                                  if "did_ns" in e else ""),
                     f3(e["delta"]["kk4-10"], True), f3(e["delta"]["kk11-16"], True),
                     f3(e["delta"].get("kk11-16ns"), True), f3(e["delta"]["kk17-20"], True),
                     f"{e['slope']:+.4f}" if e.get("slope") is not None else "-", f3(e["did_norm"], True),
                     ", ".join(f"s{s} {x:+.3f}" for s, x in e["per_seed"].items())
                     + ("; ns " + ", ".join(f"s{s} {x:+.3f}" for s, x in e["per_seed_ns"].items())
                        if "per_seed_ns" in e else "")])
    L += _table(["X", "seeds", "DiD 11-20", "boot 95%", "p", "DiD 11-16 ns", "D 4-10", "D 11-16", "D 11-16 ns",
                 "D 17-20", "slope/person", "chance-norm DiD", "per seed DiD"], rows)
    L += ["", "## Verdicts (notes/prereg_TF.md FINAL RECONCILIATION, items 1, 2, 4, 5)"]
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
    s = "notes/prereg_TF.md FINAL RECONCILIATION, hard-coded in tf_readout.decide()"
    if md5 != PREREG_MD5:
        s += f" -- WARNING: the file changed since the rules were transcribed (md5 {md5[:8]} != {PREREG_MD5[:8]})"
    ph = [p for p in ("<ANALYSIS_COMMIT>", "<COMMIT>", "<UTC>") if p.encode() in txt]
    if ph:
        s += " -- note: the registration still has placeholders (" + "/".join(ph) + ")"
    return s


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run directories or glob patterns (quote them)")
    ap.add_argument("--alias", action="append", default=[], metavar="OLD=NEW", help="rename an arm (stand-ins)")
    ap.add_argument("--standin", action="append", default=[], metavar="NEW=OLD",
                    help="also use OLD's runs as arm NEW, keeping OLD (stand-in tests)")
    ap.add_argument("--pick", action="append", default=[], metavar="ARM=tag[/data]", help="variant used for an arm")
    ap.add_argument("--data-re", default=r"^student_v3", help="training data (from the dir name) to accept")
    ap.add_argument("--kkdeep-items", default=str(KKDEEP_ITEMS), help="kk_deep eval items (shortcut stratum, names)")
    ap.add_argument("--boot", type=int, default=10000, help="pooled bootstrap draws (prereg_TF: 10,000)")
    ap.add_argument("--boot-seed", type=int, default=2000, help="draws for each per-seed CI")
    ap.add_argument("--out", default="results/tf_readout/tf_readout", help="writes <out>.json and <out>.md")
    args = ap.parse_args(argv)
    alias = dict(a.split("=", 1) for a in args.alias)
    standin = dict(a.split("=", 1) for a in args.standin)
    picks = dict(p.split("=", 1) for p in args.pick)
    items = load_kkdeep_items(Path(args.kkdeep_items) if args.kkdeep_items else None)
    strat = stratum_summary(items)
    runs, unused, skipped, notes = collect(args.runs, alias, standin, picks, args.data_re, items)
    unj = sum(r["kk_unjoined"] or 0 for rs in runs.values() for r in rs.values())
    items_note = (f"{args.kkdeep_items}: {len(items)} items parsed" + (f"; {unj} run preds not found there (no stratum)"
                                                                       if unj else "; every run pred joined")
                  if items else f"{args.kkdeep_items} absent: no shortcut stratum or unseen-name share")
    a2_registered = "A2" in runs and notes.get("A2", "").startswith("registered")
    comp2, fallback = h2_comparator(runs)
    family = [("H2'", "TFL", comp2), ("H5", "TFL", "A2")] + [(f"{a}-{b}", a, b) for a, b in REPORT_ONLY]
    contrasts, C = [], {}
    for lab, a1, a2 in family:
        for ep in ENDPOINTS:
            c = contrast(runs, a1, a2, ep, args.boot, args.boot_seed,
                         COPRIMARY_SEEDS if lab in ("H2'", "H5") else None)
            c["label"] = lab
            contrasts.append(c)
            C[(lab, ep)] = c
    # Holm over the co-primary tests that ran as registered, m = 2 fixed: seed-paired only (an unpaired arm-mean
    # difference decides nothing) and not H5 on a reference A2 variant
    holm1 = holm({k: C[(k, PRIMARY)]["p_one"] for k in ("H2'", "H5") if C[(k, PRIMARY)].get("per_seed")
                  and (k != "H5" or a2_registered)}, m=2)
    D = {comp: did(runs, "TFL", comp, args.boot) for comp in dict.fromkeys((comp2, "A2", "G3"))
         if comp in runs and "TFL" in runs}
    verdicts = decide(runs, C, holm1, a2_registered, comp2, fallback)
    lines = report(runs, unused, skipped, contrasts, D, holm1, verdicts, alias, standin, picks, notes, strat, comp2,
                   fallback, prereg_status(), items_note)
    print("\n".join(lines))
    out = {"alias": alias, "standin": standin, "picks": picks, "prereg": prereg_status(), "verdicts": verdicts,
           "holm": holm1, "h2_comparator": comp2, "h2_fallback": fallback, "a2_registered": a2_registered,
           "a2_variant": notes.get("A2"), "did": D, "stratum": strat,
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
