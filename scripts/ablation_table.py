"""Pre-written analysis of the ablation ladder (notes/next_round/ablation_design.md §7-8; the registered text is the
prereg_T addendum "ablation ladder (R0-R3)"). CPU only; reads local copies of HPC run directories.

Step 1 (assertion): every run's metrics.json args are checked against its registered row (testbed x arm x tag):
tree_full, tree_eligible_only, grouped_aux, aux_kind, subq, subq_target, label_smoothing, permute_final = 0,
lambda_brier = 0, epochs, order_seed (= seed), grad_accum, the other training knobs at their defaults (node format,
fact/depth stages, subq k/weight/frac, ...), model (basename: the home differs between accounts), train
file and, when recorded, the code commit (never -dirty; equal to --commit when given). A run that fails is excluded
and listed with the fields that differ. Runs that match no registered row are listed as unused.
Step 2 (table): per testbed the primary endpoint per run (rule: rule_ho accuracy; K&K: end-of-training dA, the mean
over the 400 diag programs of R1/R2/C correctness, computed with scripts/analyze_e2.py) and the contrasts of §7,
paired by seed: mean, 95% t-CI over seeds, per-seed signs, the test registered for it (one-/two-sided, TOST for (a)),
Holm within the registered family (K&K: (b), (c), (d); rule: (d) alone), a per-seed bootstrap (programs for K&K,
held-out cases for rule) and the training MIG-hours of both arms (history: s_per_step x steps, mid-training evals
included, final evaluation not). Missing arms and seeds are reported as missing. Then the claim rules of the
addendum, the K&K pilot gate and an intention-to-treat sensitivity line without runs that never clicked.

  python scripts/ablation_table.py results/hpc/rule/* results/hpc/abl/* --commit <sha> [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_e2 import diag_table, primary  # noqa: E402

RULE, KK = "rule", "K&K"
BASE_MODEL = "Qwen3.5-2B-Base"
RULE_TRAIN, V3_TRAIN, V3F_TRAIN = "data/student_rule/train.jsonl", "data/student_v3/train.jsonl", "data/student_v3f/train.jsonl"
# arm -> registered training args (with the defaults the sbatch passes, so a leaked XARGS/GA shows up)
ARM = {
    "A2":   dict(final="teacher", subq="none", subq_target="cot", tree_full="", tree_eligible_only=False, grouped_aux=0,
                 aux_kind="cot", grad_accum=16),
    "G0":   dict(final="teacher", subq="cot", subq_target="cot", tree_full="", tree_eligible_only=False, grouped_aux=2,
                 aux_kind="placebo", aux_weight=0.5, grad_accum=48),
    "G3":   dict(final="teacher", subq="cot", subq_target="cot", tree_full="", tree_eligible_only=False, grouped_aux=2,
                 aux_kind="cot", aux_weight=0.5, grad_accum=48),
    "G4":   dict(final="teacher", subq="random", subq_target="fresh", tree_full="", tree_eligible_only=False,
                 grouped_aux=2, aux_kind="random", aux_weight=0.5, grad_accum=48),
    "TFE":  dict(final="teacher", subq="cot", subq_target="cot", tree_full="true", tree_eligible_only=True,
                 grouped_aux=0, aux_kind="cot", aux_total=1.0, tree_cap=10, grad_accum=16),
    "TFME": dict(final="teacher", subq="cot", subq_target="cot", tree_full="matched", tree_eligible_only=True,
                 grouped_aux=0, aux_kind="cot", aux_total=1.0, tree_cap=10, grad_accum=16),
    "GFE":  dict(final="teacher", subq="cot", subq_target="cot", tree_full="plain", tree_eligible_only=True,
                 grouped_aux=0, aux_kind="cot", aux_total=1.0, tree_cap=10, grad_accum=16),
    "TF":   dict(final="teacher", subq="cot", subq_target="cot", tree_full="true", tree_eligible_only=False,
                 grouped_aux=0, aux_kind="cot", aux_total=1.0, tree_cap=10, grad_accum=16),
    "TFM":  dict(final="teacher", subq="cot", subq_target="cot", tree_full="matched", tree_eligible_only=False,
                 grouped_aux=0, aux_kind="cot", aux_total=1.0, tree_cap=10, grad_accum=16),
}
ARM["G3P"] = ARM["G3"]
COMMON = dict(permute_final=0.0, lambda_brier=0.0, lr=1e-5, micro_bs=2, max_len=1536, precision="fp32master",
              optim="adamw", items_per_update=32, lambda_sub=1.0, lora_r=0, warmup=0.05,
              # the other training knobs at their defaults, so a leaked XARGS/SUBQK/DSTAGES/... shows up too
              node_format="yesno", aux_sources="", fact_withdraw=0.0, depth_stages=0, fact_stages=0, facts_always=False,
              final_replay=0, rationale_lm=False, lambda_lm=1.0, level_balanced=False, reset_optim=False,
              subq_k=0, subq_weight="split", subq_frac=0.0, no_grad_ckpt=False)
# (testbed, arm, tag) -> (train file, epochs, label smoothing, registered seeds, note)
ROWS = {
    (RULE, "A2", ""):   (RULE_TRAIN, 2.0, 0.0, [31, 32, 33], "new"),
    (RULE, "G0", ""):   (RULE_TRAIN, 2.0, 0.0, [31, 32, 33], "seen (prereg_M G2)"),
    (RULE, "G3", ""):   (RULE_TRAIN, 2.0, 0.0, [31, 32, 33], "seen (prereg_M G2)"),
    (RULE, "G4", ""):   (RULE_TRAIN, 2.0, 0.0, [31, 32, 33], "seen (prereg_M G2)"),
    (RULE, "G3P", ""):  ("data/student_rule_p/train.jsonl", 2.0, 0.0, [31, 32, 33], "seen (prereg_M G3P)"),
    (RULE, "TFE", ""):  (RULE_TRAIN, 2.0, 0.0, [31, 32, 33, 34, 35, 36], "new"),
    (RULE, "TFME", ""): (RULE_TRAIN, 2.0, 0.0, [31, 32, 33, 34, 35, 36], "new"),
    (RULE, "GFE", ""):  (RULE_TRAIN, 2.0, 0.0, [31, 32, 33], "new"),
    (KK, "A2", "c4"):   (V3_TRAIN, 4.0, 0.1, [11, 12, 13, 14], "s11 seen; s12 relay46; s13/s14 new"),
    (KK, "A2", "c8"):   (V3_TRAIN, 8.0, 0.1, [11], "new, exploratory"),
    (KK, "G3", "c4"):   (V3F_TRAIN, 4.0, 0.1, [12, 13, 14], "new"),
    (KK, "G4", "c4"):   (V3F_TRAIN, 4.0, 0.1, [12, 13, 14], "new"),
    (KK, "TFE", "c4"):  (V3F_TRAIN, 4.0, 0.1, [11, 12, 13, 14, 15], "new (s11 = pilot)"),
    (KK, "TFME", "c4"): (V3F_TRAIN, 4.0, 0.1, [11, 12, 13, 14, 15], "new (s11 = pilot)"),
    (KK, "GFE", "c4"):  (V3F_TRAIN, 4.0, 0.1, [12, 13, 14], "new"),
    (KK, "TFE", "c6"):  (V3F_TRAIN, 6.0, 0.1, [11, 12, 13, 14, 15], "gate failed"),
    (KK, "TFME", "c6"): (V3F_TRAIN, 6.0, 0.1, [11, 12, 13, 14, 15], "gate failed"),
    (KK, "GFE", "c6"):  (V3F_TRAIN, 6.0, 0.1, [12, 13, 14], "gate failed"),
    (KK, "TF", "c4"):   (V3F_TRAIN, 4.0, 0.1, [11, 12], "exploratory full-tree pair (18:45)"),
    (KK, "TFM", "c4"):  (V3F_TRAIN, 4.0, 0.1, [11, 12], "exploratory full-tree pair (18:45)"),
}
ABSENT = {"tree_eligible_only": False, "node_format": "yesno"}   # args added after a run was made: value at the time
# (label, arm1, arm2, test, family) per testbed; test: tost (equivalence +-.02), one (one-sided > 0), two (two-sided)
CONTRASTS = {
    RULE: [("(a) G0 - A2 (equivalence +-.02)", "G0", "A2", "tost", None),
           ("(a-reg) G4 - A2", "G4", "A2", "two", None),
           ("(b) G3 - A2 (seen arms vs new A2)", "G3", "A2", "two", None),
           ("(c) G3 - G4 (seen, prereg_M)", "G3", "G4", "two", None),
           ("(d) TFE - TFME", "TFE", "TFME", "two", "rule"),
           ("(d') TFE - GFE", "TFE", "GFE", "two", None),
           ("(d') TFME - GFE", "TFME", "GFE", "two", None)],
    KK: [("(a') A2-c8 - A2-c4 (exploratory)", "A2-c8", "A2-c4", "two", None),
         ("(a-reg) G4 - A2", "G4-c4", "A2-c4", "two", None),
         ("(b) G3 - A2", "G3-c4", "A2-c4", "one", "kk"),
         ("(c) G3 - G4", "G3-c4", "G4-c4", "one", "kk"),
         ("(d) TFE - TFME", "TFE-EO", "TFME-EO", "one", "kk"),
         ("(d') TFE - GFE", "TFE-EO", "GFE-EO", "two", None),
         ("(d') TFME - GFE", "TFME-EO", "GFE-EO", "two", None)],
}
FAMILY_SIZE = {"kk": 3, "rule": 1}
PAIRED_ORDER = {"A2"}       # A2 shuffles with the training rng: contrasts against it are paired by seed label only
NAME = re.compile(r"^(?P<arm>[A-Z][A-Z0-9]*)(?:-(?P<tag>c\d+))?-(?P<model>.+)-(?P<data>student_[a-z0-9_]+)-s(?P<seed>\d+)$")


# ---- Student t without scipy (regularized incomplete beta, Numerical Recipes betacf)
def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam, c, d = a + b, a + 1, a - 1, 1.0, 1 - (a + b) * x / (a + 1)
    d = 1 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1 + aa * d
            d = 1 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1 + aa / c if abs(1 + aa / c) > 1e-30 else 1e-30
            h *= d * c
        if abs(d * c - 1) < 1e-12:
            break
    return h


def _ibeta(a: float, b: float, x: float) -> float:
    if x <= 0 or x >= 1:
        return 0.0 if x <= 0 else 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return bt * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * _betacf(b, a, 1 - x) / b


def t_sf(t: float, df: int) -> float:
    """P(T > t)."""
    p = 0.5 * _ibeta(df / 2, 0.5, df / (df + t * t))
    return p if t >= 0 else 1 - p


def t_q(p: float, df: int) -> float:
    """Upper quantile: P(T > q) = p."""
    lo, hi = 0.0, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_sf(mid, df) > p else (lo, mid)
    return (lo + hi) / 2


# ---- runs
def parse(run: Path) -> dict | None:
    m = NAME.match(run.name)
    if not m or not (run / "metrics.json").exists():
        return None
    arm, tag, data = m["arm"], m["tag"] or "", m["data"]
    if data.startswith("student_rule"):
        testbed = RULE
        if arm == "G3" and data == "student_rule_p":
            arm = "G3P"
    elif data in ("student_v3", "student_v3f"):
        testbed = KK
    else:
        return None
    return {"dir": run, "testbed": testbed, "arm": arm, "tag": tag, "seed": int(m["seed"]), "data": data,
            "metrics": json.loads((run / "metrics.json").read_text())}


def check(r: dict, commit: str) -> tuple[list[str], list[str]]:
    """(mismatches, notes) of one run against its registered row."""
    row = ROWS.get((r["testbed"], r["arm"], r["tag"]))
    if row is None:
        return ["no registered row"], []
    train, epochs, ls, seeds, _ = row
    a = r["metrics"]["args"]
    want = dict(COMMON, **ARM[r["arm"]], train=train, epochs=epochs, label_smoothing=ls, seed=r["seed"],
                order_seed=r["seed"])
    bad = []
    for k, v in want.items():
        got = a.get(k, ABSENT.get(k, "<absent>"))
        same = (abs(float(got) - float(v)) < 1e-9 if isinstance(v, float) and isinstance(got, (int, float))
                else got == v)
        if not same:
            bad.append(f"{k}={got!r} (registered {v!r})")
    if Path(str(a.get("model", ""))).name != BASE_MODEL:
        bad.append(f"model={a.get('model')!r} (registered .../{BASE_MODEL})")
    notes = []
    if r["seed"] not in seeds:
        notes.append(f"seed {r['seed']} not in the registered seeds {seeds}")
    cc = a.get("code_commit") or ""
    if not cc or cc == "unknown":       # unknown: git failed inside the job (the job log names the checkout)
        notes.append("commit not recorded" if not cc else "commit unknown in the job")
    elif cc.endswith("-dirty"):
        bad.append(f"code_commit={cc!r} (a clean checkout is registered)")
    elif commit and not (cc.startswith(commit) or commit.startswith(cc)):
        bad.append(f"code_commit={cc[:12]} (registered {commit[:12]})")
    return bad, notes


def endpoint(r: dict) -> tuple[float | None, dict]:
    """Primary endpoint and the per-unit scores for the bootstrap ({unit: score})."""
    if r["testbed"] == RULE:
        f = r["dir"] / "preds_rule_ho.jsonl"
        if f.exists():
            units = {}
            for l in open(f):
                if l.strip():
                    p = json.loads(l)
                    units[p["item_id"]] = float(p["labels"][max(range(len(p["probs"])), key=p["probs"].__getitem__)]
                                                == p["gold_label"])
            return sum(units.values()) / len(units), units
        m = r["metrics"]["eval"].get("rule_ho")
        return (m["acc"], {}) if m else (None, {})
    if not (r["dir"] / "preds_diag.jsonl").exists():
        return None, {}
    t = diag_table(r["dir"])
    progs = sorted(t)
    return primary(t, progs), {g: primary(t, [g]) for g in progs}


def train_hours(r: dict) -> float | None:
    h = r["metrics"].get("history") or []
    return h[-1]["s_per_step"] * h[-1]["step"] / 3600 if h else None


def clicked(r: dict) -> bool | None:
    """K&K: diagA >= .40 at some curve point or at the end (None without a curve)."""
    tr = [x["diagA"]["acc"] for x in r["metrics"].get("trajectory") or [] if "diagA" in x]
    end = r["metrics"]["eval"].get("diagA", {}).get("acc")
    pts = tr + ([end] if end is not None else [])
    return max(pts) >= 0.40 if pts else None


def gate_point(r: dict, epoch: float = 1.998) -> float | None:
    for x in r["metrics"].get("trajectory") or []:
        if abs(x.get("epoch", -1) - epoch) < 1e-6 and "diagA" in x:
            return x["diagA"]["acc"]
    return None


def contrast(runs: dict, a1: str, a2: str, test: str, boot: int) -> dict:
    s1, s2 = runs.get(a1, {}), runs.get(a2, {})
    seeds = sorted(set(s1) & set(s2))
    res = {"seeds": seeds, "only1": sorted(set(s1) - set(s2)), "only2": sorted(set(s2) - set(s1))}
    if not seeds:
        return res
    d = [s1[s]["y"] - s2[s]["y"] for s in seeds]
    n, mean = len(d), sum(d) / len(d)
    res.update(n=n, mean=mean, per_seed={s: x for s, x in zip(seeds, d)},
               hours=(_mean([s1[s]["h"] for s in seeds]), _mean([s2[s]["h"] for s in seeds])))
    if n >= 2:
        sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
        se, df = sd / math.sqrt(n), n - 1
        q = t_q(0.025, df)
        res.update(sd=sd, ci95=(mean - q * se, mean + q * se))
        t = mean / se if se > 0 else math.copysign(math.inf, mean) if mean else 0.0
        if test == "one":
            res["p"] = t_sf(t, df)
        elif test == "two":
            res["p"] = 2 * min(t_sf(t, df), 1 - t_sf(t, df))
        else:                                       # TOST at +-.02: equivalence if the 90% CI lies inside
            q90 = t_q(0.05, df)
            res["ci90"] = (mean - q90 * se, mean + q90 * se)
            res["p"] = max(t_sf((mean + 0.02) / se, df) if se else 0.0, 1 - t_sf((mean - 0.02) / se, df) if se else 0.0)
            res["equivalent"] = -0.02 < res["ci90"][0] and res["ci90"][1] < 0.02
    rng = random.Random(0)                          # per-seed bootstrap over programs (K&K) or held-out cases (rule)
    res["boot"] = {}
    for s in seeds:
        u1, u2 = s1[s]["units"], s2[s]["units"]
        keys = sorted(set(u1) & set(u2))
        if not keys:
            continue
        diffs = [u1[k] - u2[k] for k in keys]
        bs = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(boot))
        res["boot"][s] = (bs[int(0.025 * boot)], bs[int(0.975 * boot) - 1])
    return res


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def holm(ps: dict, m: int) -> dict:
    """Holm-adjusted p for the family members present; absent members count as p = 1 (m stays registered)."""
    items = sorted(ps.items(), key=lambda kv: kv[1])
    out, run = {}, 0.0
    for j, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - j) * p))
        out[k] = run
    return out


def fmt(x, w=7, d=3, sign=True):
    return "-".rjust(w) if x is None else (f"{x:+.{d}f}" if sign else f"{x:.{d}f}").rjust(w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories (local copies of runs/<name>)")
    ap.add_argument("--commit", default="", help="registered ablation commit (prefix ok)")
    ap.add_argument("--eo-tag", default="auto", choices=["auto", "c4", "c6"],
                    help="K&K EO runs to pool: c4 (gate passed) or c6 (gate failed); auto = c6 if any c6 EO run exists")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--json", default="", help="write the table and exclusions here")
    args = ap.parse_args()

    found, unused = [], []
    for p in map(Path, args.runs):
        r = parse(p) if p.is_dir() else None
        (found if r else unused).append(r or p)
    print("== Step 1: metrics.json args vs the registered rows")
    ok, excluded, seen = [], [], {}
    for r in sorted(found, key=lambda r: (r["testbed"], r["arm"], r["tag"], r["seed"])):
        key = (r["testbed"], r["arm"], r["tag"], r["seed"], r["data"])   # v3 and v3f runs of one arm/seed are distinct
        if key in seen:
            print(f"  DUPLICATE {r['dir']} (kept {seen[key]})")
            continue
        seen[key] = r["dir"]
        if (r["testbed"], r["arm"], r["tag"]) not in ROWS:      # not a ladder arm (e.g. an older A2 run): unused
            print(f"  unused   {r['dir']} (no registered row {r['testbed']} {r['arm']} {r['tag'] or '-'})")
            continue
        bad, notes = check(r, args.commit)
        label = f"{r['testbed']:5s} {r['arm']}{'-' + r['tag'] if r['tag'] else ''} s{r['seed']}".ljust(22)
        if bad:
            excluded.append({"run": str(r["dir"]), "why": bad})
            print(f"  EXCLUDED {label} {r['dir'].name}: " + "; ".join(bad))
        else:
            ok.append(r)
            print(f"  ok       {label}" + (f" ({'; '.join(notes)})" if notes else ""))
    for p in unused:
        print(f"  unused   {p} (no metrics.json or not a ladder run)")

    eo_tag = args.eo_tag
    if eo_tag == "auto":
        eo_tag = "c6" if any(r["testbed"] == KK and r["tag"] == "c6" for r in ok) else "c4"
    table = {"eo_tag": eo_tag, "excluded": excluded, "testbeds": {}}
    for testbed in (RULE, KK):
        runs: dict[str, dict] = {}
        for r in ok:
            if r["testbed"] != testbed:
                continue
            y, units = endpoint(r)
            if y is None:
                print(f"  {testbed} {r['dir'].name}: no primary endpoint file (excluded from the table)")
                continue
            name = r["arm"] + ("-" + r["tag"] if r["tag"] else "")
            if testbed == KK and r["arm"] in ("TFE", "TFME", "GFE"):
                if r["tag"] != eo_tag:
                    continue
                name = r["arm"] + "-EO"
            runs.setdefault(name, {})[r["seed"]] = {"y": y, "units": units, "h": train_hours(r), "run": r}
        ep = "rule_ho acc" if testbed == RULE else "dA (diag A, 400 programs)"
        print(f"\n== Step 2: {testbed}, primary endpoint {ep}" + (f"; EO runs pooled: {eo_tag}" if testbed == KK else ""))
        for (tb, arm, tag), (_, _, _, seeds, note) in ROWS.items():
            if tb != testbed or (tb == KK and arm in ("TFE", "TFME", "GFE") and tag != eo_tag):
                continue
            name = arm + ("-EO" if tb == KK and arm in ("TFE", "TFME", "GFE") else ("-" + tag if tag else ""))
            have = runs.get(name, {})
            cells = " ".join(f"s{s} {have[s]['y']:.3f}" if s in have else f"s{s} missing" for s in seeds)
            hrs = _mean([v["h"] for v in have.values()])
            print(f"  {name:8s} {cells}" + (f" | train MIG-h {hrs:.2f}/run" if hrs else "") + f"   [{note}]")
        rows, pvals = [], {}
        for label, a1, a2, test, fam in CONTRASTS[testbed]:
            c = contrast(runs, a1, a2, test, args.boot)
            c.update(label=label, arm1=a1, arm2=a2, test=test, family=fam,
                     order_paired=not ({a1.split("-")[0], a2.split("-")[0]} & PAIRED_ORDER))
            rows.append(c)
            if fam and "p" in c:
                pvals[label] = c["p"]
        fams = {f for *_, f in CONTRASTS[testbed] if f}
        adj = {}
        for f in fams:
            adj.update(holm({k: v for k, v in pvals.items() if any(r["label"] == k and r["family"] == f for r in rows)},
                            FAMILY_SIZE[f]))
        print(f"\n  {'contrast':38s} {'n':>2s} {'mean':>7s} {'95% t-CI':>17s} {'signs':8s} {'test':5s} {'p':>6s} "
              f"{'Holm':>6s}  {'MIG-h a1/a2':12s} order")
        for c in rows:
            if not c["seeds"]:
                miss = [a for a in (c["arm1"], c["arm2"]) if not runs.get(a)]
                print(f"  {c['label']:38s}  MISSING " + (", ".join(miss) if miss else
                      f"no common seed ({c['arm1']} {c['only1']}, {c['arm2']} {c['only2']})"))
                continue
            c["holm"] = adj.get(c["label"])
            signs = "".join("+" if x > 0 else "-" if x < 0 else "0" for x in c["per_seed"].values())
            ci = f"[{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}]" if "ci95" in c else "(1 seed)".rjust(17)
            h1, h2 = c["hours"]
            print(f"  {c['label']:38s} {c['n']:2d} {fmt(c['mean'])} {ci:>17s} {signs:8s} {c['test']:5s} "
                  f"{fmt(c.get('p'), 6, 3, False)} {fmt(c.get('holm'), 6, 3, False)}  "
                  f"{(f'{h1:.2f}/{h2:.2f}' if h1 and h2 else '-'):12s} {'shared' if c['order_paired'] else 'seed label'}")
            extra = []
            if c["only1"] or c["only2"]:
                extra.append(f"unpaired seeds: {c['arm1']} {c['only1']}, {c['arm2']} {c['only2']}")
            if c["test"] == "tost" and "ci90" in c:
                extra.append(f"90% CI [{c['ci90'][0]:+.3f},{c['ci90'][1]:+.3f}] -> "
                             f"{'equivalent within +-.02' if c['equivalent'] else 'equivalence NOT shown'}")
            if c["boot"]:
                extra.append("per-seed bootstrap " + ", ".join(f"s{s} {d:+.3f} [{lo:+.3f},{hi:+.3f}]" for s, (lo, hi), d
                                                               in ((s, c["boot"][s], c["per_seed"][s]) for s in c["boot"])))
            for e in extra:
                print(f"  {'':38s}    {e}")
        # intention to treat: K&K sensitivity without runs that never reached diagA .40
        if testbed == KK:
            for c in rows:
                if not c["seeds"]:
                    continue
                drop = [s for s in c["seeds"] if clicked(runs[c["arm1"]][s]["run"]) is False
                        or clicked(runs[c["arm2"]][s]["run"]) is False]
                if drop:
                    keep = {a: {s: v for s, v in runs[a].items() if s not in drop} for a in (c["arm1"], c["arm2"])}
                    cs = contrast(keep, c["arm1"], c["arm2"], c["test"], 0)
                    print(f"  sensitivity {c['label']}: without never-clicked seeds {drop}: "
                          + (f"n {cs['n']}, mean {cs['mean']:+.3f}" if cs["seeds"] else "no seed left"))
        table["testbeds"][testbed] = {"runs": {a: {s: v["y"] for s, v in d.items()} for a, d in runs.items()},
                                      "contrasts": [{k: v for k, v in c.items() if k not in ("boot",)} for c in rows]}

        # claim rules (addendum)
        d = next(c for c in rows if c["label"].startswith("(d) "))
        dg = next(c for c in rows if c["arm1"].startswith("TFE") and c["arm2"].startswith("GFE"))
        bound = 0.015 if testbed == RULE else 0.05
        bound_only = False                          # K&K: A2-c4 cross-seed SD over s11-13 > .04 -> (d) as a bound only
        print(f"\n  claim rules ({testbed}):")
        if testbed == KK:
            a2 = [v["y"] for s, v in runs.get("A2-c4", {}).items() if s in (11, 12, 13)]
            if len(a2) >= 2:
                m = sum(a2) / len(a2)
                sd = math.sqrt(sum((x - m) ** 2 for x in a2) / (len(a2) - 1))
                bound_only = sd > 0.04
                print(f"    A2-c4 cross-seed SD over s11-13 ({len(a2)} runs): {sd:.3f} -> "
                      + ("> .04: K&K (d) is reported as a bound only" if bound_only else "<= .04"))
            else:
                print("    A2-c4 cross-seed SD over s11-13: not computable yet (fewer than 2 runs)")
            pilot = [runs_all for runs_all in ok if runs_all["testbed"] == KK and runs_all["arm"] in ("TFE", "TFME")
                     and runs_all["seed"] == 11 and runs_all["tag"] == "c4"]
            for r in pilot:
                g = gate_point(r)
                print(f"    pilot gate {r['arm']}-c4 s11: diagA at epoch 1.998 = "
                      + ("missing" if g is None else f"{g:.3f} ({'pass' if g >= 0.40 else 'FAIL'})"))
        if "p" not in d:
            print(f"    (d) not testable yet ({d.get('n', 0)} paired seed(s))")
        elif bound_only:
            print(f"    (d) reported as a bound only (A2-c4 SD guard): mean {d['mean']:+.3f}, "
                  f"95% t-CI [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]; no direction is claimed")
        else:
            passed = (d["holm"] if testbed == KK else d["p"]) < 0.05     # K&K one-sided + Holm; rule two-sided
            direction = "TFE > TFME" if d["mean"] > 0 else "TFE < TFME"
            if passed and d["mean"] < 0:
                print("    (d) significant in the registered alternative direction TFE < TFME (stated inputs as a "
                      "shortcut, §4.2)")
            elif passed and dg.get("mean") is not None and dg["mean"] > 0:
                print(f"    (d) passes ({direction}) and TFE - GFE {dg['mean']:+.3f} > 0: 'the edges matter' on {testbed}")
            elif passed:
                print(f"    (d) passes ({direction}) but TFE - GFE is "
                      + (f"{dg['mean']:+.3f}" if dg.get("mean") is not None else "missing")
                      + ": read with §4.2 (e.g. 'non-parent facts hurt', not 'edges help')")
            else:
                hi = d["ci95"][1]
                print(f"    (d) does not pass; 95% upper bound {hi:+.3f} "
                      + (f"< {bound}: 'the edges add less than {bound}'" if hi < bound else f">= {bound}: inconclusive"))
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=1, default=str))


if __name__ == "__main__":
    main()
