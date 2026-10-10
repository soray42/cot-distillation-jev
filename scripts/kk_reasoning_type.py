"""K&K reasoning type: is Tree Forcing's gain over an answer-only model tied to how much hypothetical case analysis a
puzzle needs? CPU only, numpy (no scipy).

Hypothesis under test: K&K trees are dominated by hypothetical cases ("If Karen were a knight, would Alice be a
knave?"), i.e. search by proof by contradiction, while numeric chains are forward chaining over definite facts; so
GFPS - A2 should be larger on puzzles that need little case analysis and smaller / absent on puzzles that need much.

Classification. Every item is re-solved with the solver that built the training trees (teacher_cache/solver_kk_v3f/
build_reasoning_trees.py via build_local_kk.load_solver), with the calls of depth_readout.kk_complexity (Puzzle.read,
ProofBuilder.solve, gold asserted, build_solver_kk.clean on the nodes); depth / width / case are checked against
depth_readout.kk_complexity item by item. From the solver's own proof:
  case    = maximum nesting of hypothetical cases (ProofBuilder.max_case_depth)
  splits  = roles settled unconditionally by refuting a hypothesis (a top-level role fact that depends on a case node);
            people - splits roles follow by unconditional forward propagation
  hyps    = hypotheses tried (distinct non-empty case scopes = calls of ProofBuilder.refute)
  failed  = refuted branches (contradiction nodes), sat = satisfiable branches examined
  case_nodes / derive_nodes / case_share = hypothetical vs unconditional derived nodes of the cleaned training-style tree
            (case_share = case / (case + derive)), depth / width as depth_readout
  min_viol (exploratory, not part of the hypothesis) = fewest statements a distractor option violates (speakers whose
            statement truth differs from their role under that option): how hard the 6 options are to tell apart by
            checking statements one by one
Classes: U = no hypothesis at all (unit propagation alone), F = one flat hypothesis on one person, every other role by
forward propagation (case 1, splits 1), M = several flat hypotheses in sequence (case 1, splits >= 2), N = nested
hypotheses (case >= 2). Binary split for the tests: F vs S = M + N. The solver picks the split person by degree with
the first name as the tie-break, so the class is the training-tree count, an upper bound on the search a puzzle needs;
--orders K re-solves with K random name orders (only ties change) and reports class stability and the most forward
class over the orders (cls_min, a robustness row).

Eval sets: kk_deep (2,000 items, 4-20 people, primary), kk_heldout (450, 4-12), diag_kk (400 programs x R0 / R1 / R2 /
C; R1 permutes the options and R2 rewords the statements, so neither parses: they take their R0 program's class; C
changes one statement and is classified on its own; bootstrap clusters = programs).

Per pair ARM:REF (paired by seed label over their common seeds): accuracy per class, D = ARM - REF per class (mean over
paired seeds of the item difference), the interaction DiD = D(F) - D(S) (and F - N), all with a cluster bootstrap over
items (programs for diag) drawn within people strata, the same draws for every quantity; people control: the
people-stratified DiD (within each people count D(F) - D(S), weights nF nS / (nF + nS)), the item-level logistic
regression correct ~ arm + S + arm:S + seed FE + people FE (depth_readout.Pair / fit_interaction; arm:S in log-odds
per unit, i.e. the S-vs-F change of the arm effect; also + arm:people), the linear-probability gap on S with people FE
(depth_readout.lpm_gap) and the continuous measures (case, splits, hyps, case_share, case_nodes, min_viol) with people
FE (log-odds per SD). The hypothesis predicts DiD(F - S) > 0.

Numeric chains (optional section): programs of data/eval/numgen_d*.jsonl (depth_readout.ng_complexity); L = linear
chain (no fan-in: every operation takes one derived quantity at most) vs B = branching (>= 1 operation combines two
derived quantities), strata = depth; and 1 vs 2 irrelevant stated quantities.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python scripts/kk_reasoning_type.py 'results/hpc/l4b/*' \
      --out results/kk_reasoning_type/2026-10-10
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_readout as dr  # noqa: E402
import ladder_readout as lr  # noqa: E402
import tf_readout as tf  # noqa: E402
from analyze_e2 import correct, read_preds  # noqa: E402

ROOT = dr.ROOT
SOLVER = dr.SOLVER
CLASSES = ("U", "F", "M", "N")
CLASS_NAME = {"U": "no hypothesis (unit propagation alone)",
              "F": "one flat hypothesis, then forward propagation",
              "M": "several flat hypotheses in sequence",
              "N": "nested hypotheses (case depth >= 2)"}
KK_BINARY = ("F", ("F",), "S", ("M", "N"))
NG_BINARY = ("L", ("L",), "B", ("B",))
PEOPLE_BINS = ((4, 6), (7, 10), (11, 16), (17, 20))
DEFAULT_PAIRS = "GFPS-c4:A2-c12lr3,GFPS-c4lr3:A2-c12lr3,GFPS-c4lr5:A2-c12lr3,GFPS-c4:A2-c12,A2-c12lr3:A2-c12"
DEFAULT_NG_PAIRS = "GFPS-c4lr3:A2-c12lr3,GFPS-c4:A2-c12,GFPS-c4:A2-c12lr3"
CONT = (("case", "case depth (nesting)"), ("splits", "splits (roles settled by a refuted hypothesis)"),
        ("hyps", "hypotheses tried"), ("case_share", "case share of derived nodes"),
        ("case_nodes", "case nodes (cleaned tree)"), ("min_viol", "min statements a distractor violates (exploratory)"))


# ---------------------------------------------------------------- classification
def classify(case: int, splits: int) -> str:
    if case == 0:
        return "U"
    if case == 1 and splits == 1:
        return "F"
    return "M" if case == 1 else "N"


def proof_counts(b) -> dict:
    """Split structure of a solved ProofBuilder: splits, hyps, failed, sat, case."""
    types = {n["id"]: n["type"] for n in b.nodes}
    splits = 0
    scopes = set()
    for n in b.nodes:
        pred = b.predicates[n["id"]]
        if pred[0] in ("fact", "conflict") and pred[1]:
            scopes.add(pred[1])
        if (n["type"] == "derive" and pred[0] == "fact" and pred[2][0] == "role"
                and any(types.get(d) == "case" for d in n["depends_on"])):
            splits += 1
    return {"case": b.max_case_depth, "splits": splits, "hyps": len(scopes), "failed": b.failed_cases,
            "sat": b.satisfiable_cases}


def min_violations(pz, gold: str) -> int | None:
    """Fewest statements any distractor option violates (speaker's statement truth != speaker's role)."""
    viol = [sum(pz.expressions[s].evaluate(a) != a[s] for s in pz.names)
            for letter, a in pz.options.items() if letter != gold]
    return min(viol) if viol else None


def kk_reasoning(items: list[dict], solver_path: Path = SOLVER, orders: int = 0, order_seed: int = 0) -> dict:
    """item_id -> reasoning-type record (see the module docstring)."""
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
        c = proof_counts(b)
        ty = Counter(s["type"] for s in subqs)
        r = dict(c, people=len(pz.names), forward=len(pz.names) - c["splits"], case_nodes=ty["case"],
                 derive_nodes=ty["derive"], case_share=ty["case"] / max(1, ty["case"] + ty["derive"]),
                 depth=max(s["node_depth"] for s in subqs), width=sum(s["type"] != "parse" for s in subqs),
                 min_viol=min_violations(pz, ans), cls=classify(c["case"], c["splits"]))
        if orders:
            rng = random.Random(f"{order_seed}:{it['item_id']}")
            seen = [r["cls"]]
            for _ in range(orders):
                names = list(pz.names)
                rng.shuffle(names)
                b2 = solver.ProofBuilder(dataclasses.replace(pz, names=names))
                sol2, _ = b2.solve()
                assert sol2 == sol, it["item_id"]
                c2 = proof_counts(b2)
                seen.append(classify(c2["case"], c2["splits"]))
            r["cls_orders"] = dict(Counter(seen))
            r["cls_min"] = min(seen, key=CLASSES.index)
            r["stable"] = len(set(seen)) == 1
        out[it["item_id"]] = r
    return out


def check_against_depth_readout(info: dict, items: list[dict], solver_path: Path = SOLVER) -> dict:
    """depth / width / case of every item against depth_readout.kk_complexity."""
    ref = dr.kk_complexity(items, solver_path)
    bad = [i for i, v in ref.items() if any(info[i][k] != v[k] for k in ("people", "depth", "width", "case"))]
    return {"n": len(ref), "match": len(ref) - len(bad), "mismatch": bad[:10]}


def diag_reasoning(items: list[dict], solver_path: Path = SOLVER, orders: int = 0, order_seed: int = 0) -> dict:
    """diag_kk: R0 and C solved; R1 / R2 take their program's R0 record."""
    own = [it for it in items if it["group"] in ("diag/R0", "diag/C")]
    info = kk_reasoning(own, solver_path, orders, order_seed)
    r0 = {it["meta"]["program"]: info[it["item_id"]] for it in own if it["group"] == "diag/R0"}
    out = {}
    for it in items:
        prog = it["meta"]["program"]
        rec = info.get(it["item_id"]) or r0[prog]
        out[it["item_id"]] = dict(rec, program=prog, renderer=it["group"].split("/")[1])
    return out


def train_distribution(path: Path, solver_path: Path = SOLVER) -> dict:
    """Classes and node types of the training K&K trees (stored subqs; classes re-solved)."""
    types, items = Counter(), []
    with open(path) as f:
        for line in f:
            if '"knights_knaves"' not in line:
                continue
            it = json.loads(line)
            sq = it.get("subqs") or []
            if it.get("source") != "knights_knaves" or not sq:
                continue
            types.update(s.get("type") for s in sq)
            items.append({"item_id": it["item_id"], "prompt": it["prompt"], "gold_label": it.get("gold_label")})
    info = kk_reasoning(items, solver_path)
    return {"n": len(items), "node_types": dict(types), "classes": class_counts(info),
            "by_people": class_by_people(info)}


def class_counts(info: dict, key: str = "cls") -> dict:
    c = Counter(v[key] for v in info.values())
    return {k: c.get(k, 0) for k in CLASSES}


def class_by_people(info: dict, key: str = "cls") -> dict:
    by = defaultdict(Counter)
    for v in info.values():
        by[v["people"]][v[key]] += 1
    return {p: {k: by[p].get(k, 0) for k in CLASSES} for p in sorted(by)}


# ---------------------------------------------------------------- runs
def diag_units(d: Path) -> dict[str, int]:
    f = d / "preds_diag.jsonl"
    return {p["item_id"]: int(correct(p)) for p in read_preds(f) if p.get("gold_label") in p["labels"]} \
        if f.exists() else {}


def load_groups(patterns: list[str]) -> tuple[dict, list]:
    """-> ({domain: {label: {seed: run}}}, warnings); run["_u"] = {kkdeep, kk, diag} or {ng}."""
    runs, skipped = dr.find_runs(patterns)
    for r in runs:
        r["_u"] = dr.run_units(r)
        if r["domain"] == "kk":
            r["_u"]["diag"] = diag_units(Path(r["dir"]))
    sections, warn = lr.group(runs)
    out = {}
    for sec, groups in sections.items():
        if sec[0] in out:
            warn.append(f"several models in {sec[0]}: {sec} not used")
            continue
        out[sec[0]] = groups
    return out, warn + [f"skipped {a}: {b}" for a, b in skipped]


def parse_pairs(spec: str) -> list[tuple[str, str]]:
    return [tuple(p.split(":")) for p in spec.split(",") if p]


# ---------------------------------------------------------------- bootstrap draws
def cluster_draws(strata: np.ndarray, cluster: np.ndarray, boot: int, rng: np.random.Generator) -> np.ndarray:
    """(boot, n_items) multiplicities: clusters resampled within their stratum, every item of a drawn cluster
    travels with it."""
    ids, inv = np.unique(cluster, return_inverse=True)
    cs = np.zeros(len(ids), dtype=strata.dtype)
    for j in range(len(ids)):
        s = np.unique(strata[inv == j])
        assert len(s) == 1, f"cluster {ids[j]} spans strata {s}"
        cs[j] = s[0]
    return dr.strata_draws(cs, boot, rng)[:, inv]


def wmean(counts: np.ndarray, v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n = counts[:, mask].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, counts[:, mask] @ v[mask] / np.maximum(n, 1e-12), np.nan)


def ci_p(draws: np.ndarray) -> dict:
    d = draws[~np.isnan(draws)]
    if not len(d):
        return {"ci": None, "p_two": None}
    return {"ci": dr.boot_ci(d), "p_two": dr.p_two(d)}


def strat_did(gap: np.ndarray, lo: np.ndarray, hi: np.ndarray, strata: np.ndarray, counts: np.ndarray) -> dict:
    """Within each stratum D(lo) - D(hi), weighted nlo nhi / (nlo + nhi); point (counts = 1) and draws."""
    def one(c):                                       # c: (B, n)
        num, den = np.zeros(len(c)), np.zeros(len(c))
        for v in np.unique(strata):
            ml, mh = lo & (strata == v), hi & (strata == v)
            if not ml.any() or not mh.any():
                continue
            nl, nh = c[:, ml].sum(1), c[:, mh].sum(1)
            ok = (nl > 0) & (nh > 0)
            dl = np.where(ok, c[:, ml] @ gap[ml] / np.maximum(nl, 1e-12), 0.0)
            dh = np.where(ok, c[:, mh] @ gap[mh] / np.maximum(nh, 1e-12), 0.0)
            w = np.where(ok, nl * nh / np.maximum(nl + nh, 1e-12), 0.0)
            num += w * (dl - dh)
            den += w
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    pt = float(one(np.ones((1, len(gap))))[0])
    used = [int(v) for v in np.unique(strata) if (lo & (strata == v)).any() and (hi & (strata == v)).any()]
    return dict(point=pt, **ci_p(one(counts)), strata_used=used)


# ---------------------------------------------------------------- pair analysis
def pair_analysis(groups: dict, arm: str, ref: str, key: str, info: dict, binary: tuple, boot: int, boot_reg: int,
                  seed: int, strata_key: str = "people", cluster_key: str | None = None, cont=CONT,
                  classes=CLASSES, alt_keys=("cls_min",), bins=PEOPLE_BINS) -> dict:
    """GFPS - A2 style contrast of arm vs ref per reasoning class with the interaction tests."""
    seeds = sorted(set(groups[arm]) & set(groups[ref]))
    if not seeds:
        return {"arm": arm, "ref": ref, "missing": "no common seed"}
    runs = [groups[lab][s] for lab in (arm, ref) for s in seeds]
    items = sorted(set.intersection(*[set(r["_u"].get(key, {})) for r in runs]) & set(info))
    P = dr.Pair(groups, arm, ref, key, items, seeds)
    n, k = len(items), len(seeds)
    Y = P.Y.reshape(len(P.runs), n)                   # rows: ref seeds, then arm seeds
    ref_acc, arm_acc = Y[:k], Y[k:]
    per_seed_gap = arm_acc - ref_acc
    gap = per_seed_gap.mean(0)
    cls = np.array([info[i]["cls"] for i in items])
    strata = np.array([int(info[i][strata_key]) for i in items])
    cluster = np.array([info[i][cluster_key] if cluster_key else i for i in items])
    rng = np.random.default_rng(seed)
    counts = cluster_draws(strata, cluster, boot, rng)
    lo_name, lo_set, hi_name, hi_set = binary
    lo, hi = np.isin(cls, lo_set), np.isin(cls, hi_set)
    masks = {"all": np.ones(n, bool)} | {c: cls == c for c in classes} | {hi_name: hi}
    res = {"arm": arm, "ref": ref, "seeds": seeds, "items": n, "classes": {}}
    for name, m in masks.items():
        if not m.any():
            res["classes"][name] = {"n": 0}
            continue
        dd = wmean(counts, gap, m)
        res["classes"][name] = {"n": int(m.sum()), "ref_acc": float(ref_acc[:, m].mean()),
                                "arm_acc": float(arm_acc[:, m].mean()), "diff": float(gap[m].mean()), **ci_p(dd),
                                "per_seed": {s: float(per_seed_gap[j, m].mean()) for j, s in enumerate(seeds)},
                                "ref_per_seed": {s: float(ref_acc[j, m].mean()) for j, s in enumerate(seeds)},
                                "arm_per_seed": {s: float(arm_acc[j, m].mean()) for j, s in enumerate(seeds)}}
    # raw interaction (same draws)
    res["did"] = {}
    for nm, m2 in ((f"{lo_name}-{hi_name}", hi),) + tuple((f"{lo_name}-{c}", cls == c) for c in classes
                                                          if c not in lo_set and c in hi_set and (cls == c).any()):
        if not lo.any() or not m2.any():
            continue
        dd = wmean(counts, gap, lo) - wmean(counts, gap, m2)
        res["did"][nm] = {"point": float(gap[lo].mean() - gap[m2].mean()), **ci_p(dd),
                          "per_seed": {s: float(per_seed_gap[j, lo].mean() - per_seed_gap[j, m2].mean())
                                       for j, s in enumerate(seeds)}}
    res["did_people"] = strat_did(gap, lo, hi, strata, counts) if lo.any() and hi.any() else None
    # people-controlled logistic / LPM (indicator = hi class; per unit = S vs F)
    creg = counts[:boot_reg]
    res["logit"] = {}
    if lo.any() and hi.any():
        ind = hi.astype(float)
        sd = float(ind.std())
        for nm, inter in ((f"arm:{hi_name} + {strata_key} FE", ()),
                          (f"arm:{hi_name} + {strata_key} FE + arm:{strata_key}", (("strata_z", dr.zs(strata)[0]),))):
            f = dr.fit_interaction(P, ind, creg, fe=strata, arm_inter=inter)
            res["logit"][nm] = dict(f, per_unit_ci=[x / sd for x in f["ci"]],
                                    per_seed_unit={s: v / sd for s, v in (f.get("per_seed") or {}).items()})
        lp = dr.lpm_gap(P, ind, strata, creg)
        res["lpm"] = dict(lp, per_unit_ci=[x / sd for x in lp["ci"]])
    res["continuous"] = {}
    for ck, nm in cont:
        v = np.array([np.nan if info[i].get(ck) is None else info[i][ck] for i in items], dtype=float)
        if np.isnan(v).any() or v.std() == 0:
            continue
        f = dr.fit_interaction(P, v, creg, fe=strata)
        res["continuous"][ck] = dict(f, label=nm)
    # robustness: other class assignments
    res["robust"] = {}
    for ak in alt_keys:
        if not all(ak in info[i] for i in items):
            continue
        c2 = np.array([info[i][ak] for i in items])
        l2, h2 = np.isin(c2, lo_set), np.isin(c2, hi_set)
        if l2.any() and h2.any():
            dd = wmean(counts, gap, l2) - wmean(counts, gap, h2)
            res["robust"][ak] = {"n_lo": int(l2.sum()), "n_hi": int(h2.sum()),
                                 "did": float(gap[l2].mean() - gap[h2].mean()), **ci_p(dd),
                                 "did_people": strat_did(gap, l2, h2, strata, counts)}
    if all("stable" in info[i] for i in items):
        st = np.array([info[i]["stable"] for i in items])
        l2, h2 = lo & st, hi & st
        if l2.any() and h2.any():
            dd = wmean(counts, gap, l2) - wmean(counts, gap, h2)
            res["robust"]["stable_only"] = {"n_lo": int(l2.sum()), "n_hi": int(h2.sum()),
                                            "did": float(gap[l2].mean() - gap[h2].mean()), **ci_p(dd),
                                            "did_people": strat_did(gap, l2, h2, strata, counts)}
    # per strata bin x class
    res["bins"] = []
    for a, b in bins:
        mb = (strata >= a) & (strata <= b)
        if not mb.any():
            continue
        row = {"bin": f"{a}-{b}", "n": int(mb.sum())}
        for nm, m in ((lo_name, lo), (hi_name, hi)):
            mm = mb & m
            row[nm] = {"n": int(mm.sum())} | ({"ref_acc": float(ref_acc[:, mm].mean()),
                                               "arm_acc": float(arm_acc[:, mm].mean()),
                                               "diff": float(gap[mm].mean()), **ci_p(wmean(counts, gap, mm))}
                                              if mm.any() else {})
        res["bins"].append(row)
    return res


def accuracy_table(groups: dict, labels: list[str], key: str, info: dict, classes: list[str],
                   hi: tuple[str, tuple]) -> dict:
    """label -> {seed or "pooled": {class: acc}}"""
    out = {}
    for lab in labels:
        if lab not in groups:
            continue
        out[lab] = {}
        for s, r in groups[lab].items():
            u = {i: c for i, c in r["_u"].get(key, {}).items() if i in info}
            row = {"all": float(np.mean(list(u.values()))) if u else None}
            for c in list(classes) + [hi[0]]:
                cs = hi[1] if c == hi[0] else (c,)
                v = [x for i, x in u.items() if info[i]["cls"] in cs]
                row[c] = float(np.mean(v)) if v else None
            out[lab][s] = row
        if len(groups[lab]) > 1:
            seeds = list(groups[lab])
            out[lab]["pooled"] = {c: (None if any(out[lab][s][c] is None for s in seeds)
                                      else float(np.mean([out[lab][s][c] for s in seeds])))
                                  for c in out[lab][seeds[0]]}
    return out


# ---------------------------------------------------------------- formatting
T = tf._table
f3 = tf.f3


def fci(d: dict | None, key="diff") -> str:
    if not d or d.get(key) is None:
        return "-"
    s = f"{d[key]:+.3f}"
    if d.get("ci"):
        s += f" [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]"
    return s


def fseed(ps: dict | None) -> str:
    return "-" if not ps else ", ".join(f"s{s} {v:+.3f}" for s, v in ps.items())


def class_md(name: str, info: dict, check: dict | None) -> list[str]:
    cc = class_counts(info)
    L = [f"#### {name}: {len(info)} items; " + ", ".join(f"{c} {cc[c]}" for c in CLASSES)
         + (f"; matches depth_readout.kk_complexity on {check['match']}/{check['n']} items" if check else ""), ""]
    by = defaultdict(list)
    for v in info.values():
        by[v["people"]].append(v)
    rows = []
    for p, vs in sorted(by.items()):
        c = Counter(v["cls"] for v in vs)
        rows.append([p, len(vs)] + [c.get(k, 0) for k in CLASSES]
                    + [f"{np.mean([v['case'] for v in vs]):.2f}", f"{np.mean([v['splits'] for v in vs]):.2f}",
                       f"{np.mean([v['hyps'] for v in vs]):.2f}", f"{np.mean([v['case_share'] for v in vs]):.2f}",
                       f"{np.mean([v['depth'] for v in vs]):.1f}"]
                    + ([f"{np.mean([v['stable'] for v in vs]):.2f}", Counter(v["cls_min"] for v in vs).get("F", 0)]
                       if "stable" in vs[0] else ["-", "-"]))
    L += T(["people", "items"] + list(CLASSES) + ["case mean", "splits mean", "hyps mean", "case share",
                                                    "tree depth", "class stable over orders", "F over some order"],
           rows)
    return L


def pair_md(res: dict, binary: tuple, level: str = "people") -> list[str]:
    lo_name, _, hi_name, _ = binary
    arm, ref = res["arm"], res["ref"]
    if res.get("missing"):
        return [f"#### {arm} - {ref}: {res['missing']}", ""]
    L = [f"#### {arm} - {ref} (seeds {'/'.join(f's{s}' for s in res['seeds'])}, {res['items']} items)", ""]
    rows = []
    for c, v in res["classes"].items():
        if not v.get("n"):
            rows.append([c, 0, "-", "-", "-", "-"])
            continue
        rows.append([c, v["n"], f3(v["ref_acc"]), f3(v["arm_acc"]), fci(v) + (f" p={v['p_two']:.3f}"
                                                                             if v.get("p_two") is not None else ""),
                     fseed(v["per_seed"])])
    L += T(["class", "items", f"{ref} acc", f"{arm} acc", f"{arm} - {ref} [95% CI]", "per seed"], rows)
    L.append("")
    parts = [f"DiD {nm} = {fci(v, 'point')} p={v['p_two']:.3f} (per seed {fseed(v['per_seed'])})"
             for nm, v in res["did"].items()]
    dp = res.get("did_people")
    if dp:
        parts.append(f"{level}-stratified DiD {lo_name}-{hi_name} = {fci(dp, 'point')} p={dp['p_two']:.3f} "
                     f"({level} {dp['strata_used'][0]}-{dp['strata_used'][-1]} with both classes)")
    L += ["- " + p for p in parts]
    for nm, f in res.get("logit", {}).items():
        L.append(f"- logistic {nm}: arm:{hi_name} {f['beta_int'] / f['sd']:+.3f} log-odds "
                 f"[{f['per_unit_ci'][0]:+.3f},{f['per_unit_ci'][1]:+.3f}] p={f['p_two']:.3f}; arm at mean "
                 f"{f['beta_arm']:+.3f} [{f['beta_arm_ci'][0]:+.3f},{f['beta_arm_ci'][1]:+.3f}]"
                 + (f"; per seed {fseed(f['per_seed_unit'])}" if f.get("per_seed_unit") else "")
                 + ("" if f["converged"] else " (not converged)"))
    if res.get("lpm"):
        lp = res["lpm"]
        L.append(f"- linear-probability gap on {hi_name} with {level} FE: {lp['per_unit']:+.3f} "
                 f"[{lp['per_unit_ci'][0]:+.3f},{lp['per_unit_ci'][1]:+.3f}] p={lp['p_two']:.3f} "
                 f"(= {hi_name} minus {lo_name} within {level}; the hypothesis predicts < 0)")
    if res.get("continuous"):
        L += ["", f"Continuous measures, logistic arm:c with {level} FE (log-odds per SD of c; the hypothesis predicts "
              "< 0 for the search measures):", ""]
        L += T(["c", "SD", "arm:c per SD [95% CI]", "per seed"],
               [[f["label"], f"{f['sd']:.2f}", dr.fbeta(f), dr.fseeds(f)] for f in res["continuous"].values()])
    if res.get("robust"):
        L += [""] + [f"- robustness {nm}: {lo_name} {v['n_lo']} vs {hi_name} {v['n_hi']}, DiD {v['did']:+.3f} "
                     f"[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}], {level}-stratified {fci(v['did_people'], 'point')}"
                     for nm, v in res["robust"].items()]
    if res.get("bins"):
        L += ["", f"By {level} bin:", ""]
        L += T([level, "items", f"{lo_name} n", f"{lo_name} {ref} / {arm}", f"{lo_name} diff [95% CI]",
                f"{hi_name} n", f"{hi_name} {ref} / {arm}", f"{hi_name} diff [95% CI]"],
               [[b["bin"], b["n"]] + sum([[b[nm]["n"], (f"{f3(b[nm].get('ref_acc'))} / {f3(b[nm].get('arm_acc'))}"
                                                        if b[nm]["n"] else "-"), fci(b[nm]) if b[nm]["n"] else "-"]
                                          for nm in (lo_name, hi_name)], [])
                for b in res["bins"]])
    return L + [""]


def acc_md(tab: dict, cols: list[str]) -> list[str]:
    rows = []
    for lab, d in tab.items():
        for s, row in d.items():
            rows.append([f"**{lab}**" if s == "pooled" else lab, "pooled" if s == "pooled" else f"s{s}"]
                        + [f3(row.get(c)) for c in cols])
    return T(["arm", "seed"] + cols, rows)


def verdict(res: dict) -> str:
    if res.get("missing") or not res.get("did_people"):
        return "n/a"
    dp = res["did_people"]
    lg = next(iter(res["logit"].values()))
    lo, hi = dp["ci"], lg["per_unit_ci"]
    if lo[0] > 0 and hi[1] < 0:
        return "supported (people-stratified DiD > 0 and logistic arm:S < 0, both CIs exclude 0)"
    if lo[1] < 0 and hi[0] > 0:
        return "reversed (the arm gains more on the search-heavy class; both CIs exclude 0)"
    return (f"not supported: people-stratified DiD {dp['point']:+.3f} [{lo[0]:+.3f},{lo[1]:+.3f}], logistic arm:S "
            f"{lg['beta_int'] / lg['sd']:+.3f} [{hi[0]:+.3f},{hi[1]:+.3f}] (CIs include 0)")


# ---------------------------------------------------------------- main
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--pairs", default=DEFAULT_PAIRS, help="ARM:REF,... (labels as ladder_readout); first = primary")
    ap.add_argument("--ng-pairs", default=DEFAULT_NG_PAIRS, help="numeric-chain pairs ('' = skip the section)")
    ap.add_argument("--eval-dir", default=str(ROOT / "data" / "eval"))
    ap.add_argument("--train-kk", default=str(ROOT / "data" / "student_v3sl" / "train.jsonl"))
    ap.add_argument("--solver", default=str(SOLVER))
    ap.add_argument("--orders", type=int, default=8, help="random name orders per item for the stability check")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--boot-reg", type=int, default=1000, help="logistic cluster-bootstrap draws, primary pair")
    ap.add_argument("--boot-other", type=int, default=300, help="logistic draws for the other pairs and sets")
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    solver = Path(args.solver)
    ev = Path(args.eval_dir)
    doms, warn = load_groups(args.runs)
    pairs = parse_pairs(args.pairs)
    out = {"args": vars(args), "warnings": warn, "kk": {}, "numgen": {}}
    L = ["# K&K reasoning type: Tree Forcing gain vs hypothetical case analysis", "",
         "Classes (training-tree solver, re-solved per item): " + "; ".join(f"{c} = {CLASS_NAME[c]}" for c in CLASSES)
         + ". S = M + N. The hypothesis predicts DiD = D(F) - D(S) > 0 (D = arm - ref, paired by seed).", ""]
    L += [f"WARNING: {w}" for w in warn]
    kk = doms.get("kk", {})
    labels = list(dict.fromkeys(x for p in pairs for x in p))
    out["kk"]["runs"] = {lab: sorted(kk[lab]) for lab in labels if lab in kk}
    L += ["runs: " + ", ".join(f"{lab} ({'/'.join(f's{s}' for s in kk[lab])})" if lab in kk else f"{lab} (absent)"
                               for lab in labels), ""]
    sets = []
    deep = dr.read_jsonl(ev / "kk_deep.jsonl")
    info = kk_reasoning(deep, solver, args.orders)
    sets.append(("kkdeep", "kk_deep (4-20 people; trained 4-10)", info, check_against_depth_readout(info, deep, solver),
                 None))
    if (ev / "kk_heldout.jsonl").exists():
        held = dr.read_jsonl(ev / "kk_heldout.jsonl")
        ih = kk_reasoning(held, solver, args.orders)
        sets.append(("kk", "kk_heldout (4-12 people)", ih, check_against_depth_readout(ih, held, solver), None))
    if (ev / "diag_kk.jsonl").exists():
        dg = dr.read_jsonl(ev / "diag_kk.jsonl")
        sets.append(("diag", "diag_kk (5-9 people; R0/R1/R2 by program, C own)", diag_reasoning(dg, solver, args.orders),
                     None, "program"))
    L += ["## Classification", "",
          "case = max nesting of hypotheses; splits = roles settled by refuting a hypothesis (the rest follow by "
          "forward propagation); hyps = hypotheses tried; case share = case / (case + derive) nodes of the cleaned "
          f"tree; stability = same class under {args.orders} random name orders (the solver's tie-break).", ""]
    for key, name, inf, chk, _ in sets:
        L += class_md(name, inf, chk) + [""]
        out["kk"].setdefault("classification", {})[key] = {
            "counts": class_counts(inf), "by_people": class_by_people(inf), "check": chk,
            "counts_min_over_orders": class_counts(inf, "cls_min") if args.orders else None,
            "stable_share": float(np.mean([v["stable"] for v in inf.values()])) if args.orders else None}
    if Path(args.train_kk).exists():
        td = train_distribution(Path(args.train_kk), solver)
        out["kk"]["train"] = td
        nt = td["node_types"]
        L += [f"Training trees (data/student_v3sl train, {td['n']} K&K items): classes "
              + ", ".join(f"{c} {td['classes'][c]}" for c in CLASSES)
              + "; stored node types " + ", ".join(f"{k} {v:,}" for k, v in sorted(nt.items(), key=lambda x: -x[1]))
              + f" (case share of case + derive {nt.get('case', 0) / max(1, nt.get('case', 0) + nt.get('derive', 0)):.2f}).",
              ""]
    for key, name, inf, _, ckey in sets:
        L += [f"## {name}", "", "### Accuracy per class (pooled = mean over seeds)", ""]
        tab = accuracy_table(kk, labels, key, inf, list(CLASSES[1:]), ("S", ("M", "N")))
        L += acc_md(tab, ["all", "F", "M", "N", "S"]) + [""]
        J = {"accuracy": tab, "pairs": {}}
        for j, (a, r) in enumerate(pairs):
            if a not in kk or r not in kk:
                J["pairs"][f"{a}:{r}"] = {"missing": "absent"}
                L += [f"#### {a} - {r}: absent", ""]
                continue
            primary = key == "kkdeep" and j == 0
            res = pair_analysis(kk, a, r, key, inf, KK_BINARY, args.boot,
                                args.boot_reg if primary else args.boot_other, args.seed + j,
                                cluster_key=ckey)
            res["verdict"] = verdict(res)
            J["pairs"][f"{a}:{r}"] = res
            L += pair_md(res, KK_BINARY) + [f"Verdict: {res['verdict']}", ""]
        out["kk"][key] = J
        out["kk"][key]["items"] = inf
    # summary table across sets
    rows = []
    for key, name, *_ in sets:
        for pk, res in out["kk"][key]["pairs"].items():
            if res.get("missing"):
                continue
            c = res["classes"]
            lg = next(iter(res["logit"].values())) if res.get("logit") else None
            rows.append([key, pk.replace(":", " - "), "/".join(f"s{s}" for s in res["seeds"]), fci(c["all"]),
                         fci(c["F"]), fci(c["S"]), fci(res["did"].get("F-S"), "point"),
                         fci(res["did_people"], "point") if res.get("did_people") else "-",
                         "-" if lg is None else f"{lg['beta_int'] / lg['sd']:+.3f} [{lg['per_unit_ci'][0]:+.3f},"
                                                f"{lg['per_unit_ci'][1]:+.3f}]"])
    S = ["## Summary (D = arm - ref; DiD = D(F) - D(S); logistic arm:S with people FE, log-odds; hypothesis: DiD > 0, "
         "arm:S < 0)", ""]
    S += T(["set", "pair", "seeds", "D all", "D F", "D S", "DiD F-S", "DiD people-stratified", "logistic arm:S"], rows)
    L[4:4] = S + [""]
    # numeric chains
    ng = doms.get("numgen", {})
    ng_pairs = parse_pairs(args.ng_pairs)
    if ng_pairs and ng:
        items = [it for f in sorted(ev.glob("numgen_d*.jsonl")) for it in dr.read_jsonl(f)]
        comp = dr.ng_complexity(items)
        for v in comp.values():
            v["cls"] = "L" if v["fanin"] == 0 else "B"
        L += ["## Numeric chains (optional): linear (L, no fan-in) vs branching (B, >= 1 operation combines two derived "
              "quantities), strata = depth", ""]
        by = defaultdict(Counter)
        for v in comp.values():
            by[v["depth"]][v["cls"]] += 1
        L += ["items per depth (L/B): " + ", ".join(f"d{d} {c['L']}/{c['B']}" for d, c in sorted(by.items())), ""]
        nlabels = list(dict.fromkeys(x for p in ng_pairs for x in p))
        tab = accuracy_table(ng, nlabels, "ng", comp, ["L"], ("B", ("B",)))
        L += acc_md(tab, ["all", "L", "B"]) + [""]
        J = {"accuracy": tab, "pairs": {}, "irrelevant": {}}
        for j, (a, r) in enumerate(ng_pairs):
            if a not in ng or r not in ng:
                L += [f"#### {a} - {r}: absent", ""]
                continue
            res = pair_analysis(ng, a, r, "ng", comp, NG_BINARY, args.boot, args.boot_other, args.seed + 100 + j,
                                strata_key="depth", cont=(("fanin", "fan-in operations"),
                                                          ("irrelevant", "irrelevant stated quantities")),
                                classes=("L", "B"), alt_keys=(), bins=((1, 4), (5, 8)))
            J["pairs"][f"{a}:{r}"] = res
            L += pair_md(res, NG_BINARY, "depth")
            ci = {i: dict(v, cls="I1" if v["irrelevant"] <= 1 else "I2") for i, v in comp.items()}
            ri = pair_analysis(ng, a, r, "ng", ci, ("I1", ("I1",), "I2", ("I2",)), args.boot, args.boot_other,
                               args.seed + 200 + j, strata_key="depth", cont=(), classes=("I1", "I2"), alt_keys=(),
                               bins=())
            J["irrelevant"][f"{a}:{r}"] = ri
            c = ri["classes"]
            L += [f"1 vs 2 irrelevant quantities: D(1) {fci(c['I1'])}, D(2) {fci(c['I2'])}, DiD "
                  f"{fci(ri['did'].get('I1-I2'), 'point')}, depth-stratified {fci(ri['did_people'], 'point')}", ""]
        out["numgen"] = J
    print("\n".join(L))
    if args.out:
        p = Path(re.sub(r"\.(md|json)$", "", args.out))
        p.parent.mkdir(parents=True, exist_ok=True)
        Path(str(p) + ".md").write_text("\n".join(L) + "\n")
        Path(str(p) + ".json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    main()
