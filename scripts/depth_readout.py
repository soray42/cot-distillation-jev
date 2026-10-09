"""Depth readout: how far along the difficulty axis an arm stays accurate, and whether its advantage grows with the TRUE
reasoning complexity of an item rather than with its nominal size. CPU only, numpy (no scipy).

Runs are found and grouped as in ladder_readout.py (<ARM>[-tag]-<model>-<data>-s<seed>; K&K = student_v3sl, numeric
chains = "numgen" in the data name). The comparison is --arm (GFPS, c4 preferred) vs --ref (A2, c12 preferred), paired by
seed label over their common seeds; every other arm gets its own curve and a single-seed-pair interaction row.

Part 1, effective depth (per arm pooled over seeds and per seed). Difficulty x = people count (kk_deep, 4-20, trained
4-10) or chain depth (numgen d1-d8, trained d1-d4). Per level accuracy -> non-increasing isotonic fit (PAV, weights =
items) -> the first x where the fit reaches each threshold (linear interpolation between levels; "<=x_min" if the
easiest level is already at/below it, ">x_max" if the hardest is still above: censored, the bound is used in the
bootstrap and the censored share of draws is printed). Secondary: a 2-parameter logistic in x fitted to the level
means (may extrapolate; flagged). Shift = x(arm) - x(ref) at each threshold. A2-equivalent level of the arm at each
beyond-training level d: the x where the ref's isotonic curve equals the arm's isotonic accuracy at d (gain = d - that).
k*: the largest k with arm(d) >= ref(d-k) at every level d with d-k in range (raw level means and isotonic; all d and
beyond-training d only). Bootstrap: items resampled within each level (the design strata), the same draw for every
seed of both arms (paired), --boot draws.

Part 2, true complexity. Each kk_deep / kk_heldout item is re-solved with the solver that built the training trees
(teacher_cache/solver_kk_v3f/build_reasoning_trees.py via build_local_kk.load_solver) and cleaned exactly as the
training trees (build_solver_kk.clean: generic rule and verify nodes dropped, repeats merged, node_depth recomputed):
  depth  = max node_depth on the cleaned graph (longest derivation chain)
  width  = number of derived (non-parse) nodes
  case   = the solver's maximum nesting of hypothetical cases (ProofBuilder.max_case_depth)
The reconstruction is checked against the stored trees of data/student_v3sl/val.jsonl, and the training K&K trees give
the reference percentiles for "beyond training complexity" bins. Numeric chains carry their program: size = number of
operations (= the nominal depth d by construction), level = longest derivation path (< d when fan-in runs in
parallel), fanin = number of fan-in operations.
Item-level logistic regression, rows = item x run of the pair: correct ~ arm + c + arm:c + seed FE (+ controls), c
z-scored over the items; the arm:c coefficient (log-odds per SD of c) with a cluster bootstrap over items (all rows of
a drawn item travel together; drawn within the design strata), percentile 95% CI and two-sided p = 2 min(share <= 0,
share >= 0). Controls: people (or depth) fixed effects -> the interaction is identified from within-level variation;
"+arm:people" also lets the arm effect drift linearly with the nominal level. Linear-probability check: item gap (arm
- ref, mean over paired seeds) on c with level fixed effects. Re-binned accuracy: items binned by depth / width
relative to the training trees (<= median, <= p95, <= max, > max), paired difference with tf_readout.contrast.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python scripts/depth_readout.py 'results/hpc/l4b/*' \
      --out results/depth_readout/l4b_1009
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder_readout as lr  # noqa: E402
import tf_readout as tf  # noqa: E402
from analyze_e2 import correct, read_preds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / "teacher_cache" / "solver_kk_v3f" / "build_reasoning_trees.py"
THRESH = {"kk": (0.60, 0.55, 0.50), "numgen": (0.50, 0.45, 0.40)}
TRAIN_MAX = {"kk": 10, "numgen": 4}                    # largest trained people count / chain depth
LEVEL_NAME = {"kk": "people", "numgen": "depth"}
NG_FILE = re.compile(r"^preds_ng_?d(\d+)\.jsonl$")
RIDGE = 1e-4


# ---------------------------------------------------------------- numerics
def pav_dec(y, w) -> np.ndarray:
    """Weighted non-increasing isotonic fit (pool adjacent violators)."""
    blocks: list[list[float]] = []                    # [sum w*y, sum w, count]
    for yi, wi in zip(y, w):
        blocks.append([wi * yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1] - 1e-15:
            b = blocks.pop()
            blocks[-1] = [blocks[-1][0] + b[0], blocks[-1][1] + b[1], blocks[-1][2] + b[2]]
    out = []
    for s, ws, c in blocks:
        out += [s / ws] * c
    return np.array(out)


def crossing(xs, f, theta) -> tuple[float, str]:
    """First x where the non-increasing curve f reaches theta (linear between levels). Censored: ("low", x_min) when
    f(x_min) <= theta, ("high", x_max) when f(x_max) > theta."""
    if f[0] <= theta:
        return float(xs[0]), "low"
    for j in range(1, len(xs)):
        if f[j] <= theta:
            return float(xs[j - 1] + (f[j - 1] - theta) / (f[j - 1] - f[j]) * (xs[j] - xs[j - 1])), "ok"
    return float(xs[-1]), "high"


def irls(X, y, w=None, beta=None, iters=60, tol=1e-9, ridge=RIDGE) -> tuple[np.ndarray, bool]:
    """Weighted logistic regression by Newton steps (y may be fractional: aggregated binomial); a tiny ridge on every
    coefficient but the intercept keeps near-separated fixed effects finite. -> (beta, converged)."""
    n, k = X.shape
    w = np.ones(n) if w is None else w
    beta = np.zeros(k) if beta is None else beta.copy()
    R = ridge * np.eye(k)
    R[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
        W = w * p * (1 - p)
        H = X.T @ (X * W[:, None]) + R
        g = X.T @ (w * (y - p)) - R @ beta
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < tol:
            return beta, True
    return beta, False


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def boot_ci(draws) -> list[float]:
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def p_two(draws) -> float:
    return float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())))


# ---------------------------------------------------------------- items: complexity
def kk_complexity(items: list[dict], solver_path: Path = SOLVER) -> dict[str, dict]:
    """item_id -> {people, depth, width, case}: the training-tree solver re-run on the prompt, cleaned as the training
    trees (build_solver_kk.clean)."""
    from build_local_kk import load_solver
    from build_solver_kk import clean
    solver = load_solver(solver_path)
    out = {}
    for it in items:
        pz = solver.Puzzle.read({"item_id": it["item_id"], "prompt": it["prompt"]}, "eval")
        b = solver.ProofBuilder(pz)
        sol, ans = b.solve()
        if ans != it.get("gold_label", ans):
            raise AssertionError(f"{it['item_id']}: solver {ans} gold {it['gold_label']}")
        rec = {"domain": "knights_knaves", "prompt": it["prompt"],
               "predicates": [{"question": f"Is {k} a knight?", "truth": v} for k, v in sol.items()]}
        subqs, _ = clean(b.nodes, rec, random.Random(0))
        out[it["item_id"]] = {"people": len(pz.names), "depth": max(s["node_depth"] for s in subqs),
                              "width": sum(s["type"] != "parse" for s in subqs), "case": b.max_case_depth}
    return out


def stored_tree_stats(path: Path, limit: int | None = None) -> list[dict]:
    """K&K items of a student data file -> [{item_id, prompt, gold_label, people, depth, width}] from the stored
    (cleaned) solver trees."""
    out = []
    with open(path) as f:
        for line in f:
            if '"knights_knaves"' not in line:
                continue
            it = json.loads(line)
            sq = it.get("subqs") or []
            if it.get("source") != "knights_knaves" or not sq:
                continue
            out.append({"item_id": it["item_id"], "prompt": it["prompt"], "gold_label": it.get("gold_label"),
                        "people": len(re.findall(r'^- [A-Z][a-z]+ says:', it["prompt"], re.M)),
                        "depth": max(s["node_depth"] for s in sq), "width": sum(s.get("type") != "parse" for s in sq)})
            if limit and len(out) >= limit:
                break
    return out


def ng_complexity(items: list[dict]) -> dict[str, dict]:
    """item_id -> {depth (= size, operations), level (longest derivation path), fanin, leaves (stated inputs used),
    irrelevant}."""
    out = {}
    for it in items:
        p = it["program"]
        by = {q["id"]: q for q in p["quantities"]}
        cone, stack = set(), [p["answer"]]
        while stack:
            x = stack.pop()
            if x not in cone:
                cone.add(x)
                stack += by[x]["inputs"]
        lev = {}
        for q in p["quantities"]:
            lev[q["id"]] = 0 if q["stated"] else 1 + max([lev[i] for i in q["inputs"]] or [0])
        size = sum(not by[x]["stated"] for x in cone)
        out[it["item_id"]] = {"depth": int(it.get("depth", size)), "size": size, "level": lev[p["answer"]],
                              "fanin": int(it.get("fanin", sum(len(by[x]["inputs"]) > 1 for x in cone))),
                              "leaves": sum(by[x]["stated"] for x in cone),
                              "irrelevant": int(it.get("n_irrelevant", 0))}
    return out


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- runs
def find_runs(patterns: list[str]) -> tuple[list[dict], list]:
    runs, skipped = [], []
    for d in lr.expand(patterns):
        if not d.is_dir() or not (d / "metrics.json").exists():
            if d.is_dir() and lr.TAIL.match(d.name):
                skipped.append((d.name, "no metrics.json"))
            continue
        try:
            hint = Path(json.loads((d / "metrics.json").read_text()).get("args", {}).get("model") or "").name or None
        except (json.JSONDecodeError, AttributeError):
            hint = None
        info = lr.parse_run_name(d.name, hint)
        if info is None:
            skipped.append((d.name, "bad name"))
            continue
        runs.append(dict(info, domain=lr.domain_of(info["data"]), dir=str(d), name=d.name))
    return runs, skipped


def run_units(r: dict) -> dict[str, dict[str, int]]:
    """{set: {item_id: correct}}: kkdeep / kk (K&K), ng (numeric, all depth files)."""
    d = Path(r["dir"])
    out = {}
    if r["domain"] == "kk":
        out["kkdeep"] = {i: c for i, (_, c) in tf.kkdeep_units(d).items()}
        if (d / "preds_kk.jsonl").exists():
            out["kk"] = {p["item_id"]: int(correct(p)) for p in read_preds(d / "preds_kk.jsonl")
                         if p.get("gold_label") in p["labels"]}
    else:
        u = {}
        for f in sorted(d.glob("preds_ng*.jsonl")):
            if NG_FILE.match(f.name):
                u.update({p["item_id"]: int(correct(p)) for p in read_preds(f) if p.get("gold_label") in p["labels"]})
        out["ng"] = u
    return out


# ---------------------------------------------------------------- part 1: curves
class Curves:
    """Per-level accuracy of every run of a section on the common items, with paired stratified bootstrap draws."""

    def __init__(self, groups: dict, units_key: str, level_of: dict, boot: int, seed: int):
        runs = [(lab, s, r) for lab, g in groups.items() for s, r in g.items()]
        common = set.intersection(*[set(r["_u"].get(units_key, {})) for _, _, r in runs]) & set(level_of)
        self.levels = sorted({level_of[i] for i in common})
        self.items = {lv: sorted(i for i in common if level_of[i] == lv) for lv in self.levels}
        self.n = np.array([len(self.items[lv]) for lv in self.levels], dtype=float)
        rng = np.random.default_rng(seed)
        self.point, self.draws = {}, {}
        idx = {lv: rng.integers(0, len(self.items[lv]), size=(boot, len(self.items[lv]))) for lv in self.levels}
        for lab, s, r in runs:
            u = r["_u"][units_key]
            vecs = [np.array([u[i] for i in self.items[lv]], dtype=float) for lv in self.levels]
            self.point[(lab, s)] = np.array([v.mean() for v in vecs])
            self.draws[(lab, s)] = np.stack([v[idx[lv]].mean(1) for v, lv in zip(vecs, self.levels)], axis=1)
        self.boot = boot

    def curve(self, lab: str, seeds) -> tuple[np.ndarray, np.ndarray]:
        return (np.mean([self.point[(lab, s)] for s in seeds], axis=0),
                np.mean([self.draws[(lab, s)] for s in seeds], axis=0))


def iso_cross(xs, acc, w, theta) -> tuple[float, str]:
    return crossing(xs, pav_dec(acc, w), theta)


def logistic_curve(xs, acc, w) -> tuple[float, float]:
    X = np.stack([np.ones(len(xs)), np.asarray(xs, dtype=float)], axis=1)
    beta, _ = irls(X, np.clip(acc, 1e-6, 1 - 1e-6), w, ridge=0.0)
    return float(beta[0]), float(beta[1])


def logistic_cross(a, b, theta) -> float | None:
    return None if abs(b) < 1e-12 else (logit(theta) - a) / b


def summarize_cross(pt: tuple[float, str], draws: list[tuple[float, str]]) -> dict:
    v = np.array([x for x, _ in draws])
    cens = Counter(c for _, c in draws)
    return {"x": pt[0], "code": pt[1], "ci": boot_ci(v), "censored_low": cens["low"] / len(draws),
            "censored_high": cens["high"] / len(draws)}


def arm_crossings(C: Curves, lab: str, seeds, thetas) -> dict:
    xs = np.array(C.levels, dtype=float)
    pt, dr = C.curve(lab, seeds)
    out = {"seeds": list(seeds), "acc": dict(zip(C.levels, pt.tolist())),
           "iso": dict(zip(C.levels, pav_dec(pt, C.n).tolist()))}
    a, b = logistic_curve(xs, pt, C.n)
    fit = 1 / (1 + np.exp(-(a + b * xs)))
    out["logistic"] = {"a": a, "b": b, "max_abs_resid": float(np.max(np.abs(fit - pt)))}
    for th in thetas:
        p0 = iso_cross(xs, pt, C.n, th)
        dd = [iso_cross(xs, dr[k], C.n, th) for k in range(C.boot)]
        lx = logistic_cross(a, b, th)
        out[f"{th:.2f}"] = dict(summarize_cross(p0, dd), logistic=lx,
                                logistic_extrapolated=lx is None or not (xs[0] <= lx <= xs[-1]))
    return out


def kstar(xs, ga, ra, only_from=None) -> int | None:
    """Largest k >= 0 with ga(d) >= ra(d - k) at every level d (d - k a level; d >= only_from if given), over the k
    that have at least one such d; None if k = 0 already fails."""
    pos = {x: j for j, x in enumerate(xs)}
    best = None
    for k in range(0, len(xs)):
        pairs = [(pos[d], pos[d - k]) for d in xs if d - k in pos and (only_from is None or d >= only_from)]
        if not pairs:
            break
        if all(ga[i] >= ra[j] - 1e-12 for i, j in pairs):
            best = k
        else:
            break
    return best


def pair_shift(C: Curves, arm: str, ref: str, seeds, thetas, beyond_from: int) -> dict:
    """Arm - ref horizontal shift at each threshold, A2-equivalent levels and k*, pooled over `seeds` (paired)."""
    xs = np.array(C.levels, dtype=float)
    ga, gd = C.curve(arm, seeds)
    ra, rd = C.curve(ref, seeds)
    out = {"seeds": list(seeds)}
    for th in thetas:
        pg, pr = iso_cross(xs, ga, C.n, th), iso_cross(xs, ra, C.n, th)
        sd = []
        cg, cr = Counter(), Counter()
        for k in range(C.boot):
            g, r = iso_cross(xs, gd[k], C.n, th), iso_cross(xs, rd[k], C.n, th)
            sd.append(g[0] - r[0])
            cg[g[1]] += 1
            cr[r[1]] += 1
        sd = np.array(sd)
        a1, b1 = logistic_curve(xs, ga, C.n)
        a2, b2 = logistic_curve(xs, ra, C.n)
        l1, l2 = logistic_cross(a1, b1, th), logistic_cross(a2, b2, th)
        out[f"{th:.2f}"] = {"shift": pg[0] - pr[0], "codes": [pg[1], pr[1]], "ci": boot_ci(sd), "p_two": p_two(sd),
                            "arm_censored_high": cg["high"] / C.boot, "ref_censored_low": cr["low"] / C.boot,
                            "arm_censored_low": cg["low"] / C.boot, "ref_censored_high": cr["high"] / C.boot,
                            "logistic_shift": None if l1 is None or l2 is None else l1 - l2}
    # A2-equivalent level of the arm at each beyond-training level
    gi, ri = pav_dec(ga, C.n), pav_dec(ra, C.n)
    giso_d = np.stack([pav_dec(gd[k], C.n) for k in range(C.boot)])
    riso_d = np.stack([pav_dec(rd[k], C.n) for k in range(C.boot)])
    eq = {}
    for j, d in enumerate(C.levels):
        if d < beyond_from:
            continue
        x0, c0 = crossing(xs, ri, gi[j])
        dd = [crossing(xs, riso_d[k], giso_d[k, j]) for k in range(C.boot)]
        gains = np.array([d - x for x, _ in dd])
        eq[d] = {"ref_equiv": x0, "code": c0, "gain": d - x0, "gain_ci": boot_ci(gains),
                 "censored_low": sum(c == "low" for _, c in dd) / C.boot}
    out["equiv"] = eq
    lv = list(C.levels)
    ks = {"raw_all": kstar(lv, ga, ra), "raw_beyond": kstar(lv, ga, ra, beyond_from),
          "iso_all": kstar(lv, gi, ri), "iso_beyond": kstar(lv, gi, ri, beyond_from)}
    kd = [kstar(lv, giso_d[k], riso_d[k], beyond_from) for k in range(C.boot)]
    kd = np.array([-1 if k is None else k for k in kd])
    ks["iso_beyond_draws"] = {int(v): float((kd == v).mean()) for v in sorted(set(kd.tolist()))}
    ks["iso_beyond_ci"] = boot_ci(kd)
    ks["table"] = {}                                  # share of levels where arm(d) >= ref(d - k), raw means
    pos = {x: j for j, x in enumerate(lv)}
    for k in range(0, 9):
        pairs = [(pos[d], pos[d - k]) for d in lv if d - k in pos]
        if pairs:
            ks["table"][k] = {"all": float(np.mean([ga[i] >= ra[j] for i, j in pairs])),
                              "beyond": float(np.mean([ga[i] >= ra[j] for i, j in pairs if lv[i] >= beyond_from]))
                              if any(lv[i] >= beyond_from for i, _ in pairs) else None}
    out["kstar"] = ks
    return out


# ---------------------------------------------------------------- part 2: interaction regressions
class Pair:
    """Rows = (run, item) of arm vs ref on their common seeds and common items, item-major blocks per run."""

    def __init__(self, groups: dict, arm: str, ref: str, units_key: str, items: list[str], seeds=None):
        seeds = sorted(set(groups[arm]) & set(groups[ref])) if seeds is None else list(seeds)
        self.seeds = seeds
        self.runs = [(ref, s) for s in seeds] + [(arm, s) for s in seeds]
        self.items = items
        self.Y = np.concatenate([np.array([groups[lab][s]["_u"][units_key][i] for i in items], dtype=float)
                                 for lab, s in self.runs])
        n, R = len(items), len(self.runs)
        self.item_of_row = np.tile(np.arange(n), R)
        self.arm = np.repeat([float(lab == arm) for lab, _ in self.runs], n)
        self.seed = np.repeat([s for _, s in self.runs], n)

    def design(self, c, fe=None, arm_inter=(), mains=()) -> tuple[np.ndarray, list[str]]:
        """[1, arm, c, arm:c, seed FE, level FE, mains, arm:inter]; per-item arrays are broadcast to rows."""
        r = self.item_of_row
        cols, names = [np.ones(len(r)), self.arm, c[r], self.arm * c[r]], ["const", "arm", "c", "arm:c"]
        for s in self.seeds[1:]:
            cols.append((self.seed == s).astype(float))
            names.append(f"seed{s}")
        if fe is not None:
            for v in sorted(set(fe.tolist()))[1:]:
                cols.append((fe[r] == v).astype(float))
                names.append(f"fe{v}")
        for nm, m in mains:
            cols.append(m[r])
            names.append(nm)
        for nm, m in arm_inter:
            cols += [m[r], self.arm * m[r]]
            names += [nm, f"arm:{nm}"]
        return np.stack(cols, axis=1), names


def zs(x) -> tuple[np.ndarray, float, float]:
    x = np.asarray(x, dtype=float)
    sd = float(x.std()) or 1.0
    return (x - x.mean()) / sd, float(x.mean()), sd


def strata_draws(strata: np.ndarray, boot: int, rng: np.random.Generator) -> np.ndarray:
    """(boot, n_items) multiplicity of each item in a draw resampled within its stratum."""
    out = np.zeros((boot, len(strata)))
    for v in np.unique(strata):
        ids = np.flatnonzero(strata == v)
        pick = ids[rng.integers(0, len(ids), size=(boot, len(ids)))]
        for b in range(boot):
            out[b] += np.bincount(pick[b], minlength=len(strata))
    return out


def fit_interaction(P: Pair, comp: np.ndarray, counts: np.ndarray, fe=None, arm_inter=(), mains=(),
                    per_seed=True) -> dict:
    cz, mu, sd = zs(comp)
    X, names = P.design(cz, fe, arm_inter, mains)
    beta, conv = irls(X, P.Y)
    draws = np.empty(len(counts))
    arms = np.empty(len(counts))
    nconv = 0
    for b in range(len(counts)):
        bb, ok = irls(X, P.Y, counts[b][P.item_of_row], beta=beta, iters=30)
        draws[b], arms[b] = bb[3], bb[1]
        nconv += ok
    res = {"beta_int": float(beta[3]), "ci": boot_ci(draws), "p_two": p_two(draws), "beta_arm": float(beta[1]),
           "beta_arm_ci": boot_ci(arms), "beta_c": float(beta[2]), "per_unit": float(beta[3] / sd), "sd": sd,
           "mean": mu, "converged": bool(conv), "boot_converged": nconv / len(counts), "rows": len(P.Y),
           "items": len(P.items), "seeds": P.seeds, "params": len(names)}
    if per_seed and len(P.seeds) > 1:
        res["per_seed"] = {}
        for s in P.seeds:
            keep = P.seed == s
            Xs = np.delete(X[keep], [j for j, nm in enumerate(names) if nm.startswith("seed")], axis=1)
            bs, _ = irls(Xs, P.Y[keep])
            res["per_seed"][s] = float(bs[3])
    return res


def lpm_gap(P: Pair, comp: np.ndarray, fe: np.ndarray, counts: np.ndarray) -> dict:
    """OLS of the item gap (arm - ref, mean over paired seeds) on z(comp) with level fixed effects."""
    n = len(P.items)
    Y = P.Y.reshape(len(P.runs), n)
    k = len(P.seeds)
    gap = Y[k:].mean(0) - Y[:k].mean(0)
    cz, _, sd = zs(comp)
    cols = [np.ones(n), cz] + [(fe == v).astype(float) for v in sorted(set(fe.tolist()))[1:]]
    X = np.stack(cols, axis=1)

    def ols(w):
        Xw = X * w[:, None]
        return np.linalg.lstsq(Xw.T @ X + 1e-9 * np.eye(X.shape[1]), Xw.T @ gap, rcond=None)[0][1]
    b0 = ols(np.ones(n))
    draws = np.array([ols(c) for c in counts])
    return {"slope": float(b0), "ci": boot_ci(draws), "p_two": p_two(draws), "per_unit": float(b0 / sd), "sd": sd}


# ---------------------------------------------------------------- re-binning
def train_bins(train: list[dict], key: str) -> list[tuple[str, float, float]]:
    v = np.array([t[key] for t in train], dtype=float)
    med, p95, mx = float(np.median(v)), float(np.percentile(v, 95)), float(v.max())
    return [(f"<= {med:g} (train median)", -1, med), (f"{med:g}-{p95:g} (<= train p95)", med, p95),
            (f"{p95:g}-{mx:g} (<= train max)", p95, mx), (f"> {mx:g} (beyond every training tree)", mx, 1e9)]


def rebin(groups: dict, arm: str, ref: str, units_key: str, comp: dict, key: str, bins, meta: dict,
          boot: int, boot_seed: int) -> list[dict]:
    rows = []
    sub = {lab: {s: {"_units": {}} for s in groups[lab]} for lab in (arm, ref)}
    for lab_bin, lo, hi in bins:
        ids = [i for i, c in comp.items() if lo < c[key] <= hi]
        for lab in (arm, ref):
            for s, r in groups[lab].items():
                u = r["_u"][units_key]
                sub[lab][s]["_units"][lab_bin] = {i: float(u[i]) for i in ids if i in u}
        con = tf.contrast(sub, arm, ref, lab_bin, boot, 200)
        seeds = con.get("seeds") or []
        accs = {lab: float(np.mean([np.mean(list(sub[lab][s]["_units"][lab_bin].values())) for s in seeds]))
                if ids and seeds else None for lab in (arm, ref)}
        rows.append({"bin": lab_bin, "n": len(ids),
                     "people_mean": float(np.mean([comp[i]["people"] for i in ids])) if ids else None,
                     "people_range": [min(comp[i]["people"] for i in ids), max(comp[i]["people"] for i in ids)]
                     if ids else None,
                     "shortcut": float(np.mean([meta[i]["shortcut"] for i in ids if i in meta])) if ids else None,
                     "unseen_names": float(np.mean([meta[i]["unseen"] > 0 for i in ids if i in meta])) if ids
                     else None,
                     "acc": accs, "diff": con.get("mean"), "ci": con.get("ci"), "per_seed": con.get("per_seed")})
    return rows


# ---------------------------------------------------------------- formatting
def fx(c: dict) -> str:
    s = {"low": "<=", "high": ">"}.get(c["code"], "") + f"{c['x']:.1f}"
    s += f" [{c['ci'][0]:.1f},{c['ci'][1]:.1f}]"
    cen = [f"{k[9:]} {c[k]:.1%}" for k in ("censored_low", "censored_high") if c[k] > 0]
    return s + (f" (cens {', '.join(cen)})" if cen else "")


def fl(c: dict) -> str:
    x = c.get("logistic")
    return "-" if x is None else f"{x:.1f}" + ("*" if c.get("logistic_extrapolated") else "")


def fshift(c: dict) -> str:
    pre = ">=" if c["codes"][0] == "high" or c["codes"][1] == "low" else ("<=" if c["codes"][0] == "low" or
                                                                          c["codes"][1] == "high" else "")
    s = f"{pre}{c['shift']:+.1f} [{c['ci'][0]:+.1f},{c['ci'][1]:+.1f}]"
    cen = []
    if c["arm_censored_high"]:
        cen.append(f"arm>max {c['arm_censored_high']:.0%}")
    if c["ref_censored_low"]:
        cen.append(f"ref<=min {c['ref_censored_low']:.0%}")
    if c["arm_censored_low"]:
        cen.append(f"arm<=min {c['arm_censored_low']:.0%}")
    if c["ref_censored_high"]:
        cen.append(f"ref>max {c['ref_censored_high']:.0%}")
    if cen:
        s += " (" + ", ".join(cen) + ")"
    ls = c.get("logistic_shift")
    return s + ("" if ls is None else f"; logistic {ls:+.1f}")


def fk(k) -> str:
    return "none (fails at k=0)" if k is None else str(k)


def fbeta(r: dict) -> str:
    s = f"{r['beta_int']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] p={r['p_two']:.3f}"
    return s + ("" if r["converged"] else " (not converged)")


def fseeds(r: dict) -> str:
    ps = r.get("per_seed")
    return "-" if not ps else ", ".join(f"s{s} {v:+.3f}" for s, v in ps.items())


T = tf._table


# ---------------------------------------------------------------- main
def section(dom: str, groups: dict, arm: str, ref: str, args, meta_items: dict, comp: dict, extra: dict) -> tuple:
    L, J = [], {"arm": arm, "ref": ref}
    thetas = THRESH[dom]
    lvl = LEVEL_NAME[dom]
    key = "kkdeep" if dom == "kk" else "ng"
    level_of = {i: (c["people"] if dom == "kk" else c["depth"]) for i, c in comp[key].items()}
    C = Curves(groups, key, level_of, args.boot, args.seed)
    beyond = TRAIN_MAX[dom] + 1
    L += [f"### Part 1. Effective depth: where accuracy crosses {', '.join(f'{t:.2f}' for t in thetas)} "
          f"(x = {lvl}; levels {C.levels[0]}-{C.levels[-1]}, trained <= {TRAIN_MAX[dom]}; items per level "
          f"{int(C.n.min())}-{int(C.n.max())})", "",
          "Isotonic (PAV, non-increasing) fit of the per-level accuracy, linear interpolation; 95% CI = paired "
          f"stratified item bootstrap ({args.boot} draws); <=/> = censored at the range; logistic = 2-parameter fit "
          "in x to the level means (* = extrapolated beyond the levels).", ""]
    rows, J["crossings"] = [], {}
    for lab, g in groups.items():
        seeds = list(g)
        variants = ([("pooled", seeds)] if len(seeds) > 1 else []) + [(f"s{s}", [s]) for s in seeds]
        for vname, ss in variants:
            res = arm_crossings(C, lab, ss, thetas)
            J["crossings"][f"{lab}/{vname}"] = res
            rows.append([lab, vname] + [f"{fx(res[f'{t:.2f}'])}" for t in thetas]
                        + [" / ".join(fl(res[f"{t:.2f}"]) for t in thetas), f"{res['logistic']['max_abs_resid']:.3f}"])
    L += T(["arm", "seeds"] + [f"x @ {t:.2f}" for t in thetas] + ["logistic x", "logistic max |resid|"], rows)
    L += ["", f"#### Shift {arm} - {ref} (horizontal, in {lvl}; positive = {arm} reaches the threshold later)", ""]
    common = sorted(set(groups[arm]) & set(groups[ref]))
    J["shift"] = {}
    rows = []
    for vname, ss in [("pooled", common)] + [(f"s{s}", [s]) for s in common]:
        res = pair_shift(C, arm, ref, ss, thetas, beyond)
        J["shift"][vname] = res
        rows.append([vname] + [fshift(res[f"{t:.2f}"]) for t in thetas])
    L += T(["seeds"] + [f"shift @ {t:.2f}" for t in thetas], rows)
    pooled = J["shift"]["pooled"]
    L += ["", f"#### {ref}-equivalent {lvl} of {arm} at each beyond-training level (pooled {'/'.join(f's{s}' for s in common)};"
          f" isotonic curves; gain = d - equivalent)", ""]
    L += T([lvl, f"{arm} iso acc", f"{ref}-equivalent", "gain [95% CI]"],
           [[d, f"{J['crossings'][f'{arm}/pooled' if len(groups[arm]) > 1 else f'{arm}/s{common[0]}']['iso'][d]:.3f}",
             ("<=" if e["code"] == "low" else "") + f"{e['ref_equiv']:.1f}",
             f"{e['gain']:+.1f} [{e['gain_ci'][0]:+.1f},{e['gain_ci'][1]:+.1f}]"
             + (f" (ref<=min {e['censored_low']:.1%})" if e["censored_low"] else "")]
            for d, e in pooled["equiv"].items()])
    ks = pooled["kstar"]
    L += ["", f"#### \"{arm} at d >= {ref} at d-k\" (pooled; raw level means)", "",
          f"k* (largest k holding at every d): raw all d {fk(ks['raw_all'])}, raw d >= {beyond} "
          f"{fk(ks['raw_beyond'])}; isotonic all d {fk(ks['iso_all'])}, isotonic d >= {beyond} {fk(ks['iso_beyond'])} "
          f"(bootstrap 95% interval [{ks['iso_beyond_ci'][0]:.0f},{ks['iso_beyond_ci'][1]:.0f}]; -1 = fails at k=0)", ""]
    L += T(["k"] + list(ks["table"]),
           [["share of d (all)"] + [f"{v['all']:.2f}" for v in ks["table"].values()],
            [f"share of d >= {beyond}"] + ["-" if v["beyond"] is None else f"{v['beyond']:.2f}"
                                           for v in ks["table"].values()]])
    for s in common:
        k1 = J["shift"][f"s{s}"]["kstar"]
        L.append(f"s{s}: k* raw all {fk(k1['raw_all'])}, raw beyond {fk(k1['raw_beyond'])}, iso all "
                 f"{fk(k1['iso_all'])}, iso beyond {fk(k1['iso_beyond'])}")
    # ---------------- part 2
    L += ["", "### Part 2. True complexity", ""]
    items = sorted(set.intersection(*[set(r["_u"][key]) for g in groups.values() for r in g.values()]) & set(comp[key]))
    P = Pair(groups, arm, ref, key, items)
    cm = comp[key]
    rng = np.random.default_rng(args.seed + 1)
    if dom == "kk":
        people = np.array([cm[i]["people"] for i in items], dtype=float)
        depth = np.array([cm[i]["depth"] for i in items], dtype=float)
        width = np.array([cm[i]["width"] for i in items], dtype=float)
        case = np.array([cm[i]["case"] for i in items], dtype=float)
        sc = np.array([float(meta_items.get(i, {}).get("shortcut", 0)) for i in items])
        un = np.array([float(meta_items.get(i, {}).get("unseen", 0) > 0) for i in items])
        L += _kk_complexity_md(cm, items, extra)
        J["complexity_summary"] = _kk_complexity_summary(cm, items, extra)
        counts = strata_draws(people.astype(int), args.boot_reg, rng)
        pz = zs(people)[0]
        specs = [("people (nominal, baseline)", people, None, (), ()),
                 ("tree depth", depth, None, (), ()), ("tree width (derived nodes)", width, None, (), ()),
                 ("case nesting", case, None, (), ()),
                 ("tree depth + people FE", depth, people, (), ()),
                 ("tree width + people FE", width, people, (), ()),
                 ("tree depth + people FE + arm:people", depth, people, (("people_z", pz),), ()),
                 ("tree width + people FE + arm:people", width, people, (("people_z", pz),), ()),
                 ("tree depth + people FE + arm:shortcut", depth, people, (("shortcut", sc),), ()),
                 ("tree depth + people FE + arm:unseen names", depth, people, (("unseen", un),), ())]
        fe_lpm = people
    else:
        depth = np.array([cm[i]["depth"] for i in items], dtype=float)
        level = np.array([cm[i]["level"] for i in items], dtype=float)
        fanin = np.array([cm[i]["fanin"] for i in items], dtype=float)
        dz = zs(depth)[0]
        L += _ng_complexity_md(cm, items)
        J["complexity_summary"] = _ng_complexity_summary(cm, items)
        counts = strata_draws(depth.astype(int), args.boot_reg, rng)
        specs = [("size = operations = nominal depth (baseline)", depth, None, (), ()),
                 ("derivation level", level, None, (), ()),
                 ("derivation level + depth FE", level, depth, (), ()),
                 ("derivation level + depth FE + arm:depth", level, depth, (("depth_z", dz),), ()),
                 ("fan-in operations + depth FE", fanin, depth, (), ())]
        fe_lpm = depth
    L += ["", f"#### Item-level logistic regression, {arm} vs {ref} (seeds {'/'.join(f's{s}' for s in P.seeds)}, "
          f"{len(items)} items x {len(P.runs)} runs; seed FE; cluster bootstrap by item within {lvl} strata, "
          f"{args.boot_reg} draws)", "",
          "arm:c = log-odds change of the arm effect per SD of c (positive = the arm's advantage grows with c); "
          "arm = arm effect at mean c.", ""]
    rows, J["interaction"] = [], {}
    for name, cvec, fe, inter, mains in specs:
        res = fit_interaction(P, cvec, counts, fe=fe, arm_inter=inter, mains=mains)
        J["interaction"][name] = res
        rows.append([name, f"{res['sd']:.2f}", fbeta(res), f"{res['per_unit']:+.4f}",
                     f"{res['beta_arm']:+.3f} [{res['beta_arm_ci'][0]:+.3f},{res['beta_arm_ci'][1]:+.3f}]",
                     fseeds(res)])
    L += T(["c", "SD of c", "arm:c per SD [95% CI]", "arm:c per unit", "arm (log-odds) [95% CI]", "arm:c per seed"],
           rows)
    lp_specs = ([("tree depth", depth), ("tree width", width)] if dom == "kk"
                else [("derivation level", level), ("fan-in operations", fanin)])
    J["lpm"] = {}
    parts = []
    for nm, cv in lp_specs:
        r = J["lpm"][nm] = lpm_gap(P, cv, fe_lpm, counts)
        parts.append(f"{nm}: {r['slope']:+.4f} per SD [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] p={r['p_two']:.3f}")
    L += ["", f"Linear-probability check (item gap {arm} - {ref}, mean over paired seeds, on z(c) with {lvl} FE): "
          + "; ".join(parts)]
    bad = [nm for nm, r in J["interaction"].items() if r["boot_converged"] < 0.99]
    if bad:
        L.append("note: < 99% of bootstrap refits converged for " + ", ".join(bad))
    if dom == "kk":
        J["rebin"] = {}
        for k, nm in (("depth", "tree depth"), ("width", "derived nodes")):
            bins = train_bins(extra["train"], k)
            rows_b = rebin(groups, arm, ref, key, {i: cm[i] for i in items}, k, bins, meta_items, args.boot, args.seed)
            J["rebin"][k] = rows_b
            L += ["", f"#### kk_deep re-binned by {nm} relative to the {len(extra['train'])} training K&K trees "
                  f"(paired difference {arm} - {ref}, tf_readout.contrast)", ""]
            L += T([nm, "items", "people mean (range)", "shortcut share", "unseen-name share", f"{ref} acc",
                    f"{arm} acc", "diff [95% CI]", "per seed"],
                   [[b["bin"], b["n"], "-" if not b["n"] else f"{b['people_mean']:.1f} ({b['people_range'][0]}-"
                     f"{b['people_range'][1]})", lr.f3(b["shortcut"]), lr.f3(b["unseen_names"]),
                     lr.f3(b["acc"][ref]), lr.f3(b["acc"][arm]),
                     "-" if b["diff"] is None else f"{b['diff']:+.3f} [{b['ci'][0]:+.3f},{b['ci'][1]:+.3f}]",
                     "-" if not b["per_seed"] else ", ".join(f"s{s} {v:+.3f}" for s, v in b["per_seed"].items())]
                    for b in rows_b])
        # kk_heldout (4-12 people): secondary
        if all("kk" in r["_u"] for g in (groups[arm], groups[ref]) for r in g.values()) and comp.get("kk"):
            hk = sorted(set.intersection(*[set(r["_u"]["kk"]) for g in (groups[arm], groups[ref])
                                           for r in g.values()]) & set(comp["kk"]))
            if hk:
                Ph = Pair(groups, arm, ref, "kk", hk)
                ch = comp["kk"]
                pe = np.array([ch[i]["people"] for i in hk], dtype=float)
                cnt = strata_draws(pe.astype(int), args.boot_reg, rng)
                J["kk_heldout"] = {}
                rows = []
                for nm, cv, fe in (("people", pe, None),
                                   ("tree depth", np.array([ch[i]["depth"] for i in hk], float), None),
                                   ("tree width", np.array([ch[i]["width"] for i in hk], float), None),
                                   ("tree depth + people FE", np.array([ch[i]["depth"] for i in hk], float), pe),
                                   ("tree width + people FE", np.array([ch[i]["width"] for i in hk], float), pe)):
                    res = fit_interaction(Ph, cv, cnt, fe=fe)
                    J["kk_heldout"][nm] = res
                    rows.append([nm, fbeta(res), fseeds(res)])
                L += ["", f"#### kk_heldout ({len(hk)} items, people {int(pe.min())}-{int(pe.max())}): same model", ""]
                L += T(["c", "arm:c per SD [95% CI]", "per seed"], rows)
    # other arms vs ref, single seed pair
    others = [lab for lab in groups if lab not in (arm, ref) and set(groups[lab]) & set(groups[ref])]
    if others:
        J["other_arms"] = {}
        rows = []
        base = [specs[0], specs[1]] + ([specs[4]] if dom == "kk" else [specs[2]])
        for lab in others:
            ss = sorted(set(groups[lab]) & set(groups[ref]))
            Po = Pair(groups, lab, ref, key, items, seeds=ss)
            J["other_arms"][lab] = {}
            cells = []
            for name, cvec, fe, inter, mains in base:
                res = fit_interaction(Po, cvec, counts[: args.boot_other], fe=fe, arm_inter=inter, mains=mains,
                                      per_seed=False)
                J["other_arms"][lab][name] = res
                cells.append(fbeta(res))
            rows.append([lab, "/".join(f"s{s}" for s in ss)] + cells)
        L += ["", f"#### Every arm vs {ref}: arm:c per SD (common seeds only; {args.boot_other} draws)", ""]
        L += T(["arm", "seeds"] + [b[0] for b in base], rows)
    return L, J


def _kk_complexity_summary(cm, items, extra) -> dict:
    by = defaultdict(list)
    for i in items:
        by[cm[i]["people"]].append(cm[i])
    pe = np.array([cm[i]["people"] for i in items], float)
    out = {"per_people": {n: {k: [float(np.mean([v[k] for v in vs])), min(v[k] for v in vs), max(v[k] for v in vs)]
                              for k in ("depth", "width", "case")} for n, vs in sorted(by.items())},
           "corr_people": {k: float(np.corrcoef(pe, [cm[i][k] for i in items])[0, 1]) for k in ("depth", "width", "case")},
           "train": {k: {"median": float(np.median([t[k] for t in extra["train"]])),
                         "p95": float(np.percentile([t[k] for t in extra["train"]], 95)),
                         "max": max(t[k] for t in extra["train"])} for k in ("depth", "width")},
           "check_val": extra.get("check")}
    return out


def _kk_complexity_md(cm, items, extra) -> list[str]:
    s = _kk_complexity_summary(cm, items, extra)
    L = ["Per item (training-tree solver, cleaned as the training trees): depth = longest derivation chain, width = "
         "derived nodes, case = nesting of hypothetical cases. " + (
             f"Reconstruction check on data/student_v3sl/val.jsonl: {extra['check']['match']}/{extra['check']['n']} "
             "K&K items give the stored tree's depth and width." if extra.get("check") else "No reconstruction check "
                                                                                                 "(val file absent)."),
         f"Training trees ({len(extra['train'])} K&K items, people 4-10): depth median {s['train']['depth']['median']:g}"
         f" / p95 {s['train']['depth']['p95']:g} / max {s['train']['depth']['max']}; width median "
         f"{s['train']['width']['median']:g} / p95 {s['train']['width']['p95']:g} / max {s['train']['width']['max']}. "
         f"Correlation with people over kk_deep: depth {s['corr_people']['depth']:.2f}, width "
         f"{s['corr_people']['width']:.2f}, case {s['corr_people']['case']:.2f}.", ""]
    L += T(["people", "depth mean (min-max)", "width mean (min-max)", "case mean (max)"],
           [[n, f"{v['depth'][0]:.1f} ({v['depth'][1]}-{v['depth'][2]})",
             f"{v['width'][0]:.1f} ({v['width'][1]}-{v['width'][2]})", f"{v['case'][0]:.2f} ({v['case'][2]})"]
            for n, v in s["per_people"].items()])
    return L


def _ng_complexity_summary(cm, items) -> dict:
    by = defaultdict(list)
    for i in items:
        by[cm[i]["depth"]].append(cm[i])
    return {"per_depth": {d: {"size_eq_depth": float(np.mean([v["size"] == d for v in vs])),
                              "level": dict(sorted(Counter(v["level"] for v in vs).items())),
                              "fanin_mean": float(np.mean([v["fanin"] for v in vs]))} for d, vs in sorted(by.items())}}


def _ng_complexity_md(cm, items) -> list[str]:
    s = _ng_complexity_summary(cm, items)
    L = ["Program per item: size = operations in the answer's cone, level = longest derivation path, fanin = fan-in "
         "operations. Size equals the nominal depth for "
         f"{np.mean([v['size_eq_depth'] for v in s['per_depth'].values()]):.0%} of items, so size and depth are the "
         "same variable here; within a depth only level and fanin vary.", ""]
    L += T(["depth", "level counts", "fanin mean"],
           [[d, ", ".join(f"L{k}: {n}" for k, n in v["level"].items()), f"{v['fanin_mean']:.2f}"]
            for d, v in s["per_depth"].items()])
    return L


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--arm", default="GFPS:c4", help="ARM[:preferred tag]")
    ap.add_argument("--ref", default="A2:c12", help="ARM[:preferred tag]")
    ap.add_argument("--eval-dir", default=str(ROOT / "data" / "eval"))
    ap.add_argument("--train-kk", default=str(ROOT / "data" / "student_v3sl" / "train.jsonl"))
    ap.add_argument("--val-kk", default=str(ROOT / "data" / "student_v3sl" / "val.jsonl"))
    ap.add_argument("--check-val", type=int, default=200, help="val K&K items re-solved to check the trees (0: off)")
    ap.add_argument("--solver", default=str(SOLVER))
    ap.add_argument("--boot", type=int, default=2000, help="draws for curves, shifts and re-binned differences")
    ap.add_argument("--boot-reg", type=int, default=1000, help="cluster-bootstrap draws for the GFPS-vs-A2 models")
    ap.add_argument("--boot-other", type=int, default=300, help="draws for the other-arm interaction rows")
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    runs, skipped = find_runs(args.runs)
    for r in runs:
        r["_u"] = run_units(r)
    sections, warn = lr.group(runs)
    ev = Path(args.eval_dir)
    L = ["# Depth readout", "",
         f"runs: {len(runs)} ({', '.join(sorted({r['label'] + '/' + r['domain'] + ' s' + str(r['seed']) for r in runs}))})"]
    L += [f"skipped: {s}" for s in skipped] + [f"WARNING: {w}" for w in warn] + [""]
    out = {"args": vars(args), "sections": {}, "skipped": skipped, "warnings": warn}
    arm_a, _, tag_a = args.arm.partition(":")
    arm_r, _, tag_r = args.ref.partition(":")
    for sec, groups in sections.items():
        dom = sec[0]
        arm, ref = lr.pick_ref(groups, arm_a, tag_a), lr.pick_ref(groups, arm_r, tag_r)
        L += [f"## {lr.DOMAINS[dom]}" + (f" [{sec[1]}]" if sec[1] else ""), ""]
        if arm is None or ref is None:
            L += [f"no {args.arm if arm is None else args.ref} runs", ""]
            continue
        extra, meta = {}, {}
        if dom == "kk":
            deep = read_jsonl(ev / "kk_deep.jsonl")
            held = read_jsonl(ev / "kk_heldout.jsonl") if (ev / "kk_heldout.jsonl").exists() else []
            comp = {"kkdeep": kk_complexity(deep, Path(args.solver)),
                    "kk": kk_complexity(held, Path(args.solver)) if held else {}}
            meta = tf.load_kkdeep_items(ev / "kk_deep.jsonl")
            extra["train"] = stored_tree_stats(Path(args.train_kk)) if Path(args.train_kk).exists() else []
            if args.check_val and Path(args.val_kk).exists():
                vals = stored_tree_stats(Path(args.val_kk), args.check_val)
                re_ = kk_complexity(vals, Path(args.solver))
                extra["check"] = {"n": len(vals), "match": sum(re_[v["item_id"]]["depth"] == v["depth"] and
                                                               re_[v["item_id"]]["width"] == v["width"] for v in vals)}
            if not extra["train"]:
                extra["train"] = [dict(v, item_id=i) for i, v in comp["kkdeep"].items() if v["people"] <= 10]
                L.append("WARNING: no training file; training-range bins use kk_deep items with people <= 10")
        else:
            items = [it for f in sorted(ev.glob("numgen_d*.jsonl")) for it in read_jsonl(f)]
            comp = {"ng": ng_complexity(items)}
        Ls, Js = section(dom, groups, arm, ref, args, meta, comp, extra)
        L += Ls + [""]
        Js["complexity_items"] = {k: v for k, v in comp.items()}
        out["sections"][dom + (f"/{sec[1]}" if sec[1] else "")] = Js
    print("\n".join(L))
    if args.out:
        p = Path(re.sub(r"\.(md|json)$", "", args.out))
        p.parent.mkdir(parents=True, exist_ok=True)
        Path(str(p) + ".md").write_text("\n".join(L) + "\n")
        Path(str(p) + ".json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    main()
