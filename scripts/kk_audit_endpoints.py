"""Endpoints of the K&K causal audit from the per-pair predictions that scripts/kk_audit.py writes (<out>/preds.jsonl,
plus <out>/audit.json for the v2 gate inputs, fits and probes). CPU only, no model.

--preset v2 (default): the endpoints of the post-hoc amendment of 2026-10-09 (notes/prereg_M.md, K&K addendum).
  Cells = (layer, rank k) of the run (layers kept by the layer screen, k in {1, 4, 8}); CIs: percentile bootstrap over
  test programs, level 1 - 0.05 / n_cells (Bonferroni over every tested cell). Primary set "both": test pairs where the
  unpatched model is right on base and source (pairs line in_both; in_test in v1 files); secondary "all" (in_all).
  Gate   stmt:source@last DAS IIA (both) >= .50 at some cell; reported by k. If it fails, E1/E2 "holds" is None
         ("not interpretable").
  E1     per cell: role:local@last IIA >= .60, IIA - max(coincidence, random, role:z@last, noise at k >= 2) >= .30,
         inv:base@last retention of the model's own unpatched answer (base_pred) >= .80 under the DAS subspaces.
         Coincidence = P(unpatched base answer == target) on the same pairs (an intervention that changes nothing).
  E2     per cell: as registered (DiD of prop, zprop, local, z on their common pair ids, CI above 0, IIA_prop >= .50,
         inv/mid own-answer retention >= .80 and to_t_prop(das) <= to_t_prop(random), DiD > 0 in the three strata);
         min n 80 pairs on "both" (primary), 250 on "all" (secondary); "against": DiD <= 0 at every cell.
  E3     dropped (the statement-site exchange is inert; stmt:source@last is the gate).
  Contrasts (report only, one seed per arm): E2 DiD at each model's best cell on the pair ids common to the models.

--preset v1: the registered endpoints of 2026-10-06, unchanged (below; they read the all-pairs sets).

All quantities use the all-pairs test sets (in_all: every test pair of the condition, target = the high-level
counterfactual, whatever the unpatched model answers) and the DAS method unless named otherwise. CIs: percentile
bootstrap over test programs (pairs of one program move together), level 1 - 0.05 / n_layers (Bonferroni over the
fixed layer set, since each endpoint asks for "some layer").

  E1 (per layer)  role:local@last IIA >= .60, >= .30 above max(random subspace, role:z@last), and inv:base@last
                  base retention >= .80 (k = 1: the noise control is not a floor and is not used).
  E2 (per layer)  on the pair ids in the all-pairs test sets of role/mid:prop, :zprop, :local and :z (the mid pairs
                  with zprop defined): DiD = (IIA_prop - IIA_zprop) - (IIA_local - IIA_z), CI excluding 0 above it;
                  IIA_prop >= .50 on the same ids; inv/mid:base@last (prop subspaces) retention >= .80 and its
                  to_t_prop no higher than under the random subspace; DiD > 0 in each of the strata desc_agree=none,
                  viol_gap prop=local and prop_clean (n reported). "Against": DiD <= 0 at every layer. Fewer than
                  250 E2 pairs (--min-e2): underpowered, neither a pass nor "against".
  E3 (per layer)  stmt:source@stmt IIA - max(random, noise) >= .30 (stmt:source@last is reported, never used).
  Arm contrasts   --contrast TFL,TFLM --contrast TFL,A2: arm = model name up to its last "-s<seed>"; the E2 DiD is
                  taken at each model's best layer (max over the fixed set, chosen the same way for every model),
                  averaged over the arm's seeds, on the pair ids common to every model involved, paired bootstrap
                  over programs; two-sided bootstrap p, Holm across the contrasts given; "supported" = Holm p < .05,
                  diff > 0 and every seed of the first arm above the second arm's mean ("preliminary" below 3 seeds).

  Refit (registered fit-set rule): pass the refit dir after the main one, e.g. runs/kkaudit/TFL-s12
  runs/kkaudit_refit/TFL-s12; dirs with the same model name merge into one model, the later dir's conditions win.

  python scripts/kk_audit_endpoints.py runs/kkaudit/TFL-s12 runs/kkaudit/TFL-s13 runs/kkaudit/A2-s12 ... \
      --contrast TFL,TFLM --contrast TFL,A2 > endpoints.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MIN_E2 = 250          # fewer E2 pairs (slot skipping): E2 is reported as underpowered
E2_CONDS = ("role/mid:prop@last", "role/mid:zprop@last", "role/mid:local@last", "role/mid:z@last")


def load(d: Path) -> dict:
    """{cond: {"pairs": [meta...], "preds": {layer: {method: [pred...]}}, "cells": {(layer, k): {method: [...]}}}},
    the model name and audit.json (if any). v1 lines have no k: k = 1. "preds" (the v1 view) holds the lowest k."""
    conds: dict = {}
    name = None
    with open(d / "preds.jsonl") as f:
        lines = f.read().splitlines()
    for line in lines:
        r = json.loads(line)
        name = r["model"]
        c = conds.setdefault(r["cond"], {"pairs": None, "preds": defaultdict(dict), "cells": defaultdict(dict)})
        if r["kind"] == "pairs":
            c["pairs"] = r["pairs"]
        else:
            c["cells"][(int(r["layer"]), int(r.get("k") or 1))][r["method"]] = r["preds"]
    for c in conds.values():
        if c["cells"]:
            k0 = min(k for _, k in c["cells"])
            for (L, k), by in c["cells"].items():
                if k == k0:
                    c["preds"][L] = by
    audit = json.loads((d / "audit.json").read_text()) if (d / "audit.json").exists() else {}
    return {"name": name or d.name, "conds": conds, "audit": audit}


def hits(m: dict, cond: str, L: int, method: str = "das", field: str = "target") -> dict:
    """{pair_id: (program, hit)} over the condition's all-pairs test set; hit = patched answer == p[field]."""
    c = m["conds"].get(cond)
    if not c or L not in c["preds"] or method not in c["preds"][L]:
        return {}
    return {p["pair_id"]: (p["program"], int(p.get(field) is not None and k == p[field]))
            for p, k in zip(c["pairs"], c["preds"][L][method]) if p["in_all"]}


def meta(m: dict, cond: str) -> dict:
    c = m["conds"].get(cond)
    return {p["pair_id"]: p for p in (c["pairs"] if c else []) if p["in_all"]}


class Boot:
    """Percentile bootstrap over programs of means of per-pair values (several aligned vectors at once)."""

    def __init__(self, programs: list[str], reps: int, seed: int):
        self.progs = sorted(set(programs))
        idx = {p: i for i, p in enumerate(self.progs)}
        self.pid = np.array([idx[p] for p in programs], dtype=int)
        rng = np.random.default_rng(seed)
        n = len(self.progs)
        self.w = rng.multinomial(n, [1 / n] * n, size=reps).astype(float) if n else np.zeros((reps, 0))

    def means(self, v: np.ndarray) -> np.ndarray:
        """Bootstrap distribution of the pair-weighted mean of v (one value per pair)."""
        n = len(self.progs)
        s = np.bincount(self.pid, weights=v, minlength=n)
        c = np.bincount(self.pid, minlength=n).astype(float)
        den = self.w @ c
        return np.where(den > 0, (self.w @ s) / np.maximum(den, 1e-12), np.nan)


def ci(dist: np.ndarray, level: float) -> list:
    a = (1 - level) / 2
    d = dist[~np.isnan(dist)]
    return [round(float(np.quantile(d, a)), 4), round(float(np.quantile(d, 1 - a)), 4)] if len(d) else [None, None]


def rate(h: dict, ids=None) -> float | None:
    v = [x for k, (_, x) in h.items() if ids is None or k in ids]
    return round(sum(v) / len(v), 4) if v else None


def layers_of(m: dict) -> list[int]:
    return sorted({L for c in m["conds"].values() for L in c["preds"]})


def did_vectors(m: dict, L: int, ids: list[str]) -> np.ndarray:
    h = [hits(m, c, L) for c in E2_CONDS]
    return np.array([h[0][i][1] - h[1][i][1] - h[2][i][1] + h[3][i][1] for i in ids], dtype=float)


def e2_ids(models: list[dict]) -> set:
    ids = None
    for m in models:
        for c in E2_CONDS:
            s = set(meta(m, c))
            ids = s if ids is None else ids & s
    return ids or set()


def model_endpoints(m: dict, reps: int, seed: int, min_e2: int = MIN_E2) -> dict:
    Ls = layers_of(m)
    level = 1 - 0.05 / max(1, len(Ls))
    out: dict = {"model": m["name"], "layers": Ls, "ci_level": round(level, 4), "E1": {}, "E2": {}, "E3": {}}
    for L in Ls:
        das, rnd = (rate(hits(m, "role:local@last", L, k)) for k in ("das", "random"))
        z = rate(hits(m, "role:z@last", L))
        ret = rate(hits(m, "inv:base@last", L))
        floor = max(x for x in (rnd, z, 0.0) if x is not None)
        out["E1"][L] = {"iia": das, "random": rnd, "z": z, "inv_retention": ret,
                        "holds": None if None in (das, ret) else das >= .60 and das - floor >= .30 and ret >= .80}
        s = rate(hits(m, "stmt:source@stmt", L))
        sr = [rate(hits(m, "stmt:source@stmt", L, k)) for k in ("random", "noise")]
        sfl = max([x for x in sr if x is not None] or [0.0])
        out["E3"][L] = {"iia": s, "random": sr[0], "noise": sr[1],
                        "stmt_at_last (not used)": rate(hits(m, "stmt:source@last", L)),
                        "holds": None if s is None else s - sfl >= .30}
    ids = sorted(e2_ids([m]))
    mt = meta(m, E2_CONDS[0])
    out["E2_n_pairs"] = len(ids)
    out["E2_underpowered"] = len(ids) < min_e2
    if ids:
        boot = Boot([mt[i]["program"] for i in ids], reps, seed)
        strata = {"desc_agree=none": lambda p: p.get("src_desc_agree") == 0,
                  "viol_gap prop=local": lambda p: (p.get("viol_base") or {}).get("prop") is not None
                  and p["viol_base"]["prop"] == p["viol_base"]["local"],
                  "prop_clean": lambda p: p.get("prop_extra_viol") == 0}
        for L in Ls:
            d = did_vectors(m, L, ids)
            lo, hi = ci(boot.means(d), level)
            prop = rate(hits(m, E2_CONDS[0], L), set(ids))
            inv = hits(m, "inv/mid:base@last", L)
            inv_ret = rate(inv)
            to_prop = {k: rate(hits(m, "inv/mid:base@last", L, k, field="t_prop")) for k in ("das", "random")}
            st = {}
            for nm, f in strata.items():
                keep = np.array([f(mt[i]) for i in ids])
                st[nm] = {"did": round(float(d[keep].mean()), 4) if keep.any() else None, "n": int(keep.sum())}
            ok = (lo is not None and lo > 0 and prop is not None and prop >= .50 and inv_ret is not None
                  and inv_ret >= .80 and None not in to_prop.values() and to_prop["das"] <= to_prop["random"]
                  and all(v["did"] is not None and v["did"] > 0 for v in st.values()))
            out["E2"][L] = {"did": round(float(d.mean()), 4), "ci": [lo, hi], "iia_prop": prop,
                            "iia_zprop": rate(hits(m, E2_CONDS[1], L), set(ids)),
                            "iia_local": rate(hits(m, E2_CONDS[2], L), set(ids)),
                            "iia_z": rate(hits(m, E2_CONDS[3], L), set(ids)),
                            "inv_mid_retention": inv_ret, "inv_mid_to_t_prop": to_prop, "strata": st,
                            "holds": bool(ok)}
        out["E2_against"] = all(v["did"] <= 0 for v in out["E2"].values()) and not out["E2_underpowered"]
    for e in ("E1", "E2", "E3"):
        out[f"{e}_holds_at_some_layer"] = any(v.get("holds") for v in out[e].values()) if out[e] else None
    if out["E2_underpowered"]:
        out["E2_holds_at_some_layer"] = None             # reported, never a pass or "against"
    return out


def arm_of(name: str) -> str:
    return re.sub(r"-s\d+$", "", name)


def contrast(models: list[dict], a: str, b: str, reps: int, seed: int) -> dict:
    ms = {x: [m for m in models if arm_of(m["name"]) == x] for x in (a, b)}
    if not ms[a] or not ms[b]:
        return {"contrast": f"{a}-{b}", "error": "missing arm", "models": {x: [m["name"] for m in v] for x, v in ms.items()}}
    involved = ms[a] + ms[b]
    ids = sorted(e2_ids(involved))
    if not ids:
        return {"contrast": f"{a}-{b}", "error": "no common E2 pairs"}
    mt = meta(involved[0], E2_CONDS[0])
    boot = Boot([mt[i]["program"] for i in ids], reps, seed)
    def arm_dist(group):     # per bootstrap rep: mean over seeds of (max over layers of the DiD)
        per_model = []
        for m in group:
            per_model.append(np.nanmax(np.stack([boot.means(did_vectors(m, L, ids)) for L in layers_of(m)]), axis=0))
        return np.mean(per_model, axis=0)
    def arm_point(group):
        return float(np.mean([max(did_vectors(m, L, ids).mean() for L in layers_of(m)) for m in group]))
    diff = arm_dist(ms[a]) - arm_dist(ms[b])
    seed_vals = {m["name"]: round(max(did_vectors(m, L, ids).mean() for L in layers_of(m)), 4) for m in involved}
    d = diff[~np.isnan(diff)]
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))) if len(d) else None
    return {"contrast": f"{a}-{b}", "n_pairs": len(ids), "n_programs": len(boot.progs),
            "seeds": {a: [m["name"] for m in ms[a]], b: [m["name"] for m in ms[b]]},
            "did_best_layer": {a: round(arm_point(ms[a]), 4), b: round(arm_point(ms[b]), 4)},
            "diff": round(arm_point(ms[a]) - arm_point(ms[b]), 4), "ci95": ci(diff, .95), "p_boot": p,
            "per_seed": seed_vals,
            "every_seed_above": all(seed_vals[m["name"]] > arm_point(ms[b]) for m in ms[a]),
            "preliminary": min(len(ms[a]), len(ms[b])) < 3}


def holm(rows: list[dict]) -> None:
    ok = sorted([r for r in rows if r.get("p_boot") is not None], key=lambda r: r["p_boot"])
    m, prev = len(ok), 0.0
    for i, r in enumerate(ok):
        prev = max(prev, min(1.0, (m - i) * r["p_boot"]))
        r["p_holm"] = round(prev, 4)
        r["supported"] = bool(r["p_holm"] < .05 and r["diff"] > 0 and r["every_seed_above"])


# ---------------------------------------------------------------------------------------------------------------------
# v2 endpoints (post-hoc amendment 2026-10-09)
# ---------------------------------------------------------------------------------------------------------------------

GATE = "stmt:source@last"
GATE_MIN = 0.50
MIN_E2_BOTH = 80
SETS = ("both", "all")


def in_set(p: dict, sset: str) -> bool:
    return bool(p.get("in_all")) if sset == "all" else bool(p.get("in_both", p.get("in_test")))


def hits2(m: dict, cond: str, cell: tuple, sset: str, method: str = "das", field: str = "target") -> dict:
    """{pair_id: (program, hit)} on the condition's `sset` test pairs at cell (layer, k); hit = patched == p[field]."""
    c = m["conds"].get(cond)
    if not c or cell not in c["cells"] or method not in c["cells"][cell]:
        return {}
    return {p["pair_id"]: (p["program"], int(p.get(field) is not None and k == p[field]))
            for p, k in zip(c["pairs"], c["cells"][cell][method]) if in_set(p, sset)}


def meta2(m: dict, cond: str, sset: str) -> dict:
    c = m["conds"].get(cond)
    return {p["pair_id"]: p for p in (c["pairs"] if c else []) if in_set(p, sset)}


def coinc(m: dict, cond: str, sset: str, ids=None) -> float | None:
    """P(unpatched base answer == target): the IIA of an intervention that changes nothing."""
    v = [int(p["base_pred"] == p["target"]) for i, p in meta2(m, cond, sset).items() if ids is None or i in ids]
    return round(sum(v) / len(v), 4) if v else None


def cells_of(m: dict, cond: str) -> list:
    c = m["conds"].get(cond)
    return sorted(c["cells"]) if c else []


def did2(m: dict, cell: tuple, ids: list, sset: str) -> np.ndarray:
    h = [hits2(m, c, cell, sset) for c in E2_CONDS]
    return np.array([h[0][i][1] - h[1][i][1] - h[2][i][1] + h[3][i][1] for i in ids], dtype=float)


def e2_ids2(models: list[dict], sset: str) -> set:
    ids = None
    for m in models:
        for c in E2_CONDS:
            s = set(meta2(m, c, sset))
            ids = s if ids is None else ids & s
    return ids or set()


def gate_of(m: dict) -> dict:
    cells = cells_of(m, GATE)
    iia = {f"L{L}/k{k}": rate(hits2(m, GATE, (L, k), "both")) for L, k in cells}
    by_k: dict = {}
    for (L, k) in cells:
        v = iia[f"L{L}/k{k}"]
        if v is not None:
            by_k[k] = max(by_k.get(k, 0.0), v)
    return {"cond": GATE, "set": "both", "threshold": GATE_MIN, "iia": iia, "best_by_k": by_k,
            "passed_k": {k: v >= GATE_MIN for k, v in by_k.items()},
            "passed": bool(by_k) and any(v >= GATE_MIN for v in by_k.values()),
            "n": len(meta2(m, GATE, "both"))}


def fits_summary(m: dict) -> dict:
    """Per condition: DAS fits that lowered the held-out loss (rel_drop), stopped early, ended below chance."""
    out: dict = {}
    for key, f in (m["audit"].get("fits") or {}).items():
        cond = key.split("|")[0]
        o = out.setdefault(cond, {"n": 0, "converged": 0, "below_chance": 0, "rel_drop": []})
        o["n"] += 1
        o["converged"] += bool(f.get("converged"))
        o["below_chance"] += bool(f.get("below_chance"))
        if f.get("rel_drop") is not None:
            o["rel_drop"].append(f["rel_drop"])
    for o in out.values():
        r = sorted(o.pop("rel_drop"))
        o["median_rel_drop"] = r[len(r) // 2] if r else None
    return out


def model_endpoints_v2(m: dict, reps: int, seed: int, min_both: int = MIN_E2_BOTH, min_all: int = MIN_E2) -> dict:
    cells = cells_of(m, "role:local@last") or cells_of(m, E2_CONDS[0])
    level = 1 - 0.05 / max(1, len(cells))
    gate = gate_of(m)
    head = (m["audit"].get("head") or {})
    out: dict = {"model": m["name"], "version": "v2", "cells": [f"L{L}/k{k}" for L, k in cells],
                 "ci_level": round(level, 4), "gate": gate, "layer_screen": head.get("layer_screen"),
                 "decode_fidelity": m["audit"].get("decode_fidelity"), "fits": fits_summary(m),
                 "probes": head.get("probes"), "E3": "dropped (amendment 2026-10-09)"}
    why = None if gate["passed"] else "gate failed: stmt:source@last DAS IIA < .50 at every cell"
    for sset in SETS:
        e1 = {}
        for L, k in cells:
            c = (L, k)
            h = hits2(m, "role:local@last", c, sset)
            das, rnd, noi = (rate(hits2(m, "role:local@last", c, sset, x)) for x in ("das", "random", "noise"))
            z = rate(hits2(m, "role:z@last", c, sset))
            co = coinc(m, "role:local@last", sset)
            ret = rate(hits2(m, "inv:base@last", c, sset, field="base_pred"))
            floor_parts = {"coinc": co, "random": rnd, "z": z, "noise": noi if k >= 2 else None}
            floor = max([v for v in floor_parts.values() if v is not None] or [0.0])
            ok = None if None in (das, ret) else das >= .60 and das - floor >= .30 and ret >= .80
            row = {"iia": das, "floor": round(floor, 4), **floor_parts, "noise_reported_k1": noi if k == 1 else None,
                   "inv_retention_own": ret, "n": len(h), "holds_raw": ok, "holds": ok if why is None else None}
            if h:
                ids = sorted(h)
                hr = hits2(m, "role:local@last", c, sset, "random")
                boot = Boot([h[i][0] for i in ids], reps, seed)
                d = np.array([h[i][1] - hr[i][1] for i in ids], dtype=float) if set(hr) >= set(ids) else None
                row["ci_iia"] = ci(boot.means(np.array([h[i][1] for i in ids], dtype=float)), level)
                row["ci_iia_minus_random"] = ci(boot.means(d), level) if d is not None else None
            e1[f"L{L}/k{k}"] = row
        ids = sorted(e2_ids2([m], sset))
        min_n = min_both if sset == "both" else min_all
        e2 = {}
        if ids:
            mt = meta2(m, E2_CONDS[0], sset)
            boot = Boot([mt[i]["program"] for i in ids], reps, seed)
            strata = {"desc_agree=none": lambda p: p.get("src_desc_agree") == 0,
                      "viol_gap prop=local": lambda p: (p.get("viol_base") or {}).get("prop") is not None
                      and p["viol_base"]["prop"] == p["viol_base"]["local"],
                      "prop_clean": lambda p: p.get("prop_extra_viol") == 0}
            for L, k in cells:
                c = (L, k)
                if any(not hits2(m, x, c, sset) for x in E2_CONDS):
                    continue
                d = did2(m, c, ids, sset)
                lo, hi = ci(boot.means(d), level)
                prop = rate(hits2(m, E2_CONDS[0], c, sset), set(ids))
                inv_ret = rate(hits2(m, "inv/mid:base@last", c, sset, field="base_pred"))
                to_prop = {x: rate(hits2(m, "inv/mid:base@last", c, sset, x, field="t_prop")) for x in ("das", "random")}
                st = {}
                for nm, f in strata.items():
                    keep = np.array([f(mt[i]) for i in ids])
                    st[nm] = {"did": round(float(d[keep].mean()), 4) if keep.any() else None, "n": int(keep.sum())}
                ok = (lo is not None and lo > 0 and prop is not None and prop >= .50 and inv_ret is not None
                      and inv_ret >= .80 and None not in to_prop.values() and to_prop["das"] <= to_prop["random"]
                      and all(v["did"] is not None and v["did"] > 0 for v in st.values()))
                e2[f"L{L}/k{k}"] = {"did": round(float(d.mean()), 4), "ci": [lo, hi], "iia_prop": prop,
                                    "iia_zprop": rate(hits2(m, E2_CONDS[1], c, sset), set(ids)),
                                    "iia_local": rate(hits2(m, E2_CONDS[2], c, sset), set(ids)),
                                    "iia_z": rate(hits2(m, E2_CONDS[3], c, sset), set(ids)),
                                    "coinc_prop": coinc(m, E2_CONDS[0], sset, set(ids)),
                                    "inv_mid_retention_own": inv_ret, "inv_mid_to_t_prop": to_prop, "strata": st,
                                    "holds_raw": bool(ok), "holds": bool(ok) if why is None else None}
        under = len(ids) < min_n
        res = {"E1": e1, "E2": e2, "E2_n_pairs": len(ids), "E2_min_n": min_n, "E2_underpowered": under,
               "E1_holds_at_some_cell": None if why else (any(v["holds"] for v in e1.values()) if e1 else None),
               "E2_holds_at_some_cell": None if (why or under) else (any(v["holds"] for v in e2.values())
                                                                     if e2 else None),
               "E2_against": None if (why or under or not e2) else all(v["did"] <= 0 for v in e2.values())}
        if why:
            res["not_interpretable"] = why
        out[sset] = res
    out["primary"] = "both"
    return out


def contrast_v2(models: list[dict], a: str, b: str, reps: int, seed: int, sset: str) -> dict:
    """E2 DiD at each model's best cell, mean over the arm's seeds, on the pair ids common to every model involved."""
    ms = {x: [m for m in models if arm_of(m["name"]) == x] for x in (a, b)}
    if not ms[a] or not ms[b]:
        return {"contrast": f"{a}-{b}", "set": sset, "error": "missing arm",
                "models": {x: [m["name"] for m in v] for x, v in ms.items()}}
    involved = ms[a] + ms[b]
    ids = sorted(e2_ids2(involved, sset))
    if not ids:
        return {"contrast": f"{a}-{b}", "set": sset, "error": "no common E2 pairs"}
    mt = meta2(involved[0], E2_CONDS[0], sset)
    boot = Boot([mt[i]["program"] for i in ids], reps, seed)
    cells = {m["name"]: [c for c in cells_of(m, E2_CONDS[0]) if all(hits2(m, x, c, sset) for x in E2_CONDS)]
             for m in involved}

    def best(m):
        return max(did2(m, c, ids, sset).mean() for c in cells[m["name"]])

    def dist(group):
        return np.mean([np.nanmax(np.stack([boot.means(did2(m, c, ids, sset)) for c in cells[m["name"]]]), axis=0)
                        for m in group], axis=0)
    diff = dist(ms[a]) - dist(ms[b])
    d = diff[~np.isnan(diff)]
    pa, pb = float(np.mean([best(m) for m in ms[a]])), float(np.mean([best(m) for m in ms[b]]))
    return {"contrast": f"{a}-{b}", "set": sset, "n_pairs": len(ids), "n_programs": len(boot.progs),
            "did_best_cell": {a: round(pa, 4), b: round(pb, 4)}, "diff": round(pa - pb, 4), "ci95": ci(diff, .95),
            "p_boot": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))) if len(d) else None,
            "per_seed": {m["name"]: round(best(m), 4) for m in involved},
            "gates_passed": {m["name"]: gate_of(m)["passed"] for m in involved},
            "report_only": True, "preliminary": min(len(ms[a]), len(ms[b])) < 3}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="kk_audit.py --out directories (each with preds.jsonl)")
    ap.add_argument("--preset", choices=("v2", "v1"), default="v2",
                    help="v2: endpoints of the 2026-10-09 amendment; v1: the registered 2026-10-06 endpoints")
    ap.add_argument("--contrast", action="append", default=[], help="ARM_A,ARM_B, e.g. TFL,TFLM (repeatable)")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-e2", type=int, default=MIN_E2, help="fewer E2 pairs: underpowered (registered: 250; v2: "
                                                               "the all-pairs secondary)")
    ap.add_argument("--min-e2-both", type=int, default=MIN_E2_BOTH, help="v2 primary (both correct): 80")
    args = ap.parse_args(argv)
    models: list[dict] = []
    for d in args.dirs:                     # same model name in several dirs (a registered refit): one model, the later
        m = load(Path(d))                   # dir's conditions replace the earlier ones
        old = next((x for x in models if x["name"] == m["name"]), None)
        if old:
            old["conds"].update(m["conds"])
        else:
            models.append(m)
    if args.preset == "v2":
        res = {"preset": "v2", "models": [model_endpoints_v2(m, args.reps, args.seed, args.min_e2_both, args.min_e2)
                                          for m in models], "contrasts": []}
        for c in args.contrast:
            a, b = c.split(",")
            for sset in SETS:
                res["contrasts"].append(contrast_v2(models, a, b, args.reps, args.seed, sset))
    else:
        res = {"models": [model_endpoints(m, args.reps, args.seed, args.min_e2) for m in models], "contrasts": []}
        for c in args.contrast:
            a, b = c.split(",")
            res["contrasts"].append(contrast(models, a, b, args.reps, args.seed))
        holm(res["contrasts"])
    json.dump(res, sys.stdout, indent=1, default=str)
    print()
    return res


if __name__ == "__main__":
    main()
