"""Registered endpoints of the K&K causal audit (notes/prereg_M.md, addendum "K&K causal-audit endpoints",
2026-10-06) from the per-pair predictions that scripts/kk_audit.py writes (<out>/preds.jsonl). CPU only, no model.

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
    """{cond: {"pairs": [meta...], "preds": {layer: {method: [pred...]}}}} and the model name."""
    conds: dict = {}
    name = None
    with open(d / "preds.jsonl") as f:
        lines = f.read().splitlines()
    for line in lines:
        r = json.loads(line)
        name = r["model"]
        c = conds.setdefault(r["cond"], {"pairs": None, "preds": defaultdict(dict)})
        if r["kind"] == "pairs":
            c["pairs"] = r["pairs"]
        else:
            c["preds"][int(r["layer"])][r["method"]] = r["preds"]
    return {"name": name or d.name, "conds": conds}


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


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="kk_audit.py --out directories (each with preds.jsonl)")
    ap.add_argument("--contrast", action="append", default=[], help="ARM_A,ARM_B, e.g. TFL,TFLM (repeatable)")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-e2", type=int, default=MIN_E2, help="fewer E2 pairs: underpowered (registered: 250)")
    args = ap.parse_args(argv)
    models: list[dict] = []
    for d in args.dirs:                     # same model name in several dirs (a registered refit): one model, the later
        m = load(Path(d))                   # dir's conditions replace the earlier ones
        old = next((x for x in models if x["name"] == m["name"]), None)
        if old:
            old["conds"].update(m["conds"])
        else:
            models.append(m)
    res = {"models": [model_endpoints(m, args.reps, args.seed, args.min_e2) for m in models], "contrasts": []}
    for c in args.contrast:
        a, b = c.split(",")
        res["contrasts"].append(contrast(models, a, b, args.reps, args.seed))
    holm(res["contrasts"])
    json.dump(res, sys.stdout, indent=1)
    print()
    return res


if __name__ == "__main__":
    main()
