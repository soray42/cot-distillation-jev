"""Paired comparison of two FA arms under StepFinder's protocol (notes/next_round/failure_attribution/sf_protocol.md
§8), per test file and pooled over the real-log test files, with seeds averaged per item. CPU only, numpy only.

  python scripts/fa_pool.py --a 'results/hpc/fasf/A2-sfx6-*-student_fa_sf_alg-s*' \\
      --b 'results/hpc/fasf/GFPS-sfs3-*-student_fa_sf_alg_t2-s*' --at-a final --at-b e2 --name alg_a2_vs_gfpst2
  -> results/fa_pool/<name>.md, <name>.json (B minus A everywhere)

Arms: --a / --b are run-directory globs (one run per seed; hpc/train_arm.sbatch names <ARM>-<tag>-<model>-<data>-s<seed>).
Checkpoint per arm (--at-a / --at-b): "final" (preds_<name>.jsonl), "val" (per run, the --eval-at point or the end
with the best accuracy on --val, the first best as StepFinder's main.py; fa_readout.sf_select) or "e<epoch>" (that
--eval-at point). Mid-training points read <run>/mid/preds_<name>_e<epoch>.jsonl (train_student.py --eval-mid-preds),
which exist only for the run's EVALMID sets; a file without predictions at the point is left out and listed, unless
--fallback-final scores it from the final predictions (marked "final*").
Items: an eval name is used when its file (metrics.json args.eval, else fa_readout.DEFAULT_EVAL) holds step-level FA
items; a run's predictions are scored as fa_readout does (rank of the gold option in the stable descending order of
probs, ties to the earlier option: rank 1 = the trainer's argmax). Per file, the items predicted by every run of both
arms; per item and arm the mean over its seeds of correct (Acc), rank <= 3 (Acc@3) and 1 / rank if rank <= 3 (MRR@3).
Statistics (difference d = B - A per item, seeds averaged):
  CI           item-level paired bootstrap of the mean of d (--boot draws, percentile 95%; pooled: items resampled within
               each file, so file sizes are fixed), for Acc, Acc@3 and MRR@3
  McNemar      exact two-sided test on Acc for every seed pair (runs with the same seed in both arms; one run per arm
               with different seeds is paired and marked "unpaired seeds"): discordant counts b (A right, B wrong) and
               c (A wrong, B right), p = min(1, 2 P(X <= min(b, c))), X ~ Bin(b + c, 1/2)
  sign-flip    a paired sign-flip test on the seed-averaged d (exact = McNemar when d is in {-1, 0, 1}; else Monte Carlo,
               --flips draws), the multi-seed analogue of McNemar
  power        Connor's (1987) McNemar sample size n = (z_{1-a/2} sqrt(psi) + z_{1-b} sqrt(psi - delta^2))^2 / delta^2
               for delta = --deltas points, at --alpha and --power, with psi = the observed discordance rate
               (b + c) / n (mean over seed pairs), and the power the available n gives; also the normal-approximation n
               from the variance of the seed-averaged d (what seed averaging buys once several seeds exist)
Pools: "real-log" = the test trajectories of real multi-agent runs (logs with an annotated or injected decisive error),
as opposed to StepFinder's regenerated trajectories (val) and synthetic logs (fa0): Who&When in the protocol's step
files (data/eval/fa_sf_ww_{alg,hc}.jsonl) and the five public test sets after the StepFinder-contamination filter
(data/eval/fa_sf_x_<src>_sfclean.jsonl, scripts/build_fa_sf_x.py). ww_alg_step (the same 126 Who&When trajectories as
fa_sf_ww_alg at another cut), the _gt files (reference answer shown), agent-level files and val are never pooled.
Pools reported: "all" (every real-log file present) and "new" (the five public sets only); --pool NAMES adds one more.
Pooled values: item-weighted (every trajectory counts once) and the equal-weight mean of the per-file differences.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fa_readout_for_pool", ROOT / "scripts/fa_readout.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

WW_RE = re.compile(r"(?:^|/)fa_sf_ww_(?:alg|hc)\.jsonl$")
NEW_RE = re.compile(r"(?:^|/)fa_sf_x_[a-z]+_sfclean\.jsonl$")
METRICS = ("acc", "acc3", "mrr3")
SKIP_PREDS = re.compile(r"^(subq|local_kk)")


# ------------------------------------------------------------------------------------------------- statistics
def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p: min(1, 2 P(X <= min(b, c))), X ~ Bin(b + c, 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def sign_flip(d: np.ndarray, flips: int, rng: np.random.Generator) -> float:
    """Two-sided paired sign-flip p of mean(d) = 0: exact McNemar when every d is -1, 0 or 1, else Monte Carlo."""
    nz = d[d != 0]
    if len(nz) == 0:
        return 1.0
    if np.all(np.isin(nz, (-1.0, 1.0))):
        return mcnemar(int(np.sum(nz < 0)), int(np.sum(nz > 0)))
    obs = abs(nz.sum())
    hits = 0
    for s in range(0, flips, 2000):
        k = min(2000, flips - s)
        sg = rng.choice((-1.0, 1.0), size=(k, len(nz)))
        hits += int(np.sum(np.abs(sg @ nz) >= obs - 1e-12))
    return (1 + hits) / (1 + flips)


def boot_strata(ds: list[np.ndarray], draws: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap draws of the item-weighted mean and of the equal-weight mean over strata (items resampled within each
    stratum; one stratum = a plain paired bootstrap)."""
    N = sum(len(d) for d in ds)
    item_w = np.zeros(draws)
    eq_w = np.zeros(draws)
    for d in ds:
        n = len(d)
        means = np.empty(draws)
        for s in range(0, draws, 500):
            b = min(500, draws - s)
            means[s:s + b] = d[rng.integers(0, n, (b, n))].mean(1)
        item_w += means * n / N
        eq_w += means / len(ds)
    return item_w, eq_w


def pct_ci(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def connor_n(psi: float, delta: float, alpha: float, power: float) -> float | None:
    """McNemar sample size (Connor 1987); None when psi <= delta^2 (no discordance budget for that difference)."""
    if psi <= delta ** 2 or psi <= 0:
        return None
    za, zb = NormalDist().inv_cdf(1 - alpha / 2), NormalDist().inv_cdf(power)
    return (za * math.sqrt(psi) + zb * math.sqrt(psi - delta ** 2)) ** 2 / delta ** 2


def connor_power(psi: float, delta: float, alpha: float, n: int) -> float | None:
    if psi <= delta ** 2 or n <= 0:
        return None
    za = NormalDist().inv_cdf(1 - alpha / 2)
    return NormalDist().cdf((delta * math.sqrt(n) - za * math.sqrt(psi)) / math.sqrt(psi - delta ** 2))


def var_n(var: float, delta: float, alpha: float, power: float) -> float | None:
    if var <= 0:
        return None
    za, zb = NormalDist().inv_cdf(1 - alpha / 2), NormalDist().inv_cdf(power)
    return (za + zb) ** 2 * var / delta ** 2


# ------------------------------------------------------------------------------------------------- runs
def load_arm(pattern: str) -> list[dict]:
    runs = R.load_runs([pattern if Path(pattern).is_absolute() else str(ROOT / pattern)])
    if not runs:
        raise SystemExit(f"no run directory matches {pattern}")
    seeds = [r["seed"] for r in runs]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"{pattern}: several runs share a seed ({seeds}); one run per seed per arm")
    return runs


def final_epoch(r: dict) -> float | None:
    ep = ((r["metrics"] or {}).get("args") or {}).get("epochs")
    return float(ep) if ep is not None else None


def checkpoint(r: dict, at: str, val: str) -> float | None:
    """The epoch to score (None = final predictions)."""
    fe = final_epoch(r)
    if at == "final":
        return None
    if at == "val":
        pts = R.sf_points(r, set(), val)
        ep = R.sf_select(pts, val)
        if ep is None:
            raise SystemExit(f"{r['name']}: no {val} accuracy to select a checkpoint")
    elif re.fullmatch(r"e[0-9.]+", at):
        ep = float(at[1:])
    else:
        raise SystemExit(f"--at {at}: final, val or e<epoch>")
    return None if fe is not None and ep == fe else ep


def run_preds(r: dict, ep: float | None, fallback: bool) -> tuple[dict, dict]:
    """({eval name: preds}, {eval name: source label}) of a run at a checkpoint."""
    d = Path(r["dir"])
    final = {p.stem[len("preds_"):]: p for p in sorted(d.glob("preds_*.jsonl"))
             if not SKIP_PREDS.match(p.stem[len("preds_"):])}
    if ep is None:
        return {n: R.read_preds(p) for n, p in final.items()}, {n: "final" for n in final}
    mid = R.mid_preds(d).get(ep, {})
    preds, src = dict(mid), {n: f"e{ep:g}" for n in mid}
    if fallback:
        for n, p in final.items():
            if n not in preds:
                preds[n], src[n] = R.read_preds(p), "final*"
    return preds, src


_ITEMS: dict = {}


def step_items(path: Path) -> dict | None:
    """item_id -> gold option index of a step-level FA file (None when the file is not one)."""
    key = str(path)
    if key not in _ITEMS:
        out = {}
        if path.exists() and R.is_fa_file(path):
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    it = json.loads(line)
                    if not all(R.STEP_OPT.match(o) for o in R.options(it)) or it.get("gold_label") not in it["labels"]:
                        continue
                    out[it["item_id"]] = (it["labels"], it["labels"].index(it["gold_label"]))
        _ITEMS[key] = out or None
    return _ITEMS[key]


def score(preds: dict, items: dict) -> dict:
    """item_id -> (acc, acc3, mrr3) for the predicted items of a step file."""
    out = {}
    for iid, p in preds.items():
        if iid not in items:
            continue
        labels, g = items[iid]
        if p["labels"] != labels or p.get("gold_label") != labels[g]:
            raise ValueError(f"{iid}: predictions and eval item disagree on labels / gold")
        rk = R.ranks(p["probs"], g)
        out[iid] = (float(rk == 1), float(rk <= 3), 1.0 / rk if rk <= 3 else 0.0)
    return out


def collect(runs: list[dict], at: str, args) -> list[dict]:
    """Per run: seed, checkpoint, {eval name: (file path, scores)} for its step-level FA files."""
    out = []
    for r in runs:
        ep = checkpoint(r, at, args.val)
        preds, src = run_preds(r, ep, args.fallback_final)
        emap = R.eval_map(Path(r["dir"]), r["metrics"])
        files = {}
        for name, pr in preds.items():
            if not pr:
                continue
            path = R.resolve(name, emap, next(iter(pr)), r.get("data") or "")
            items = step_items(path) if path is not None else None
            if not items:
                continue
            files[name] = (str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), score(pr, items),
                           src[name])
        out.append({"name": r["name"], "seed": r["seed"], "epoch": ep, "files": files})
    return out


def seed_pairs(A: list[dict], B: list[dict]) -> list[tuple[int, int, bool]]:
    """(index in A, index in B, unpaired-seed flag): equal seeds; else one run each."""
    sa = {r["seed"]: i for i, r in enumerate(A)}
    pairs = [(sa[r["seed"]], j, False) for j, r in enumerate(B) if r["seed"] in sa]
    if not pairs and len(A) == 1 and len(B) == 1:
        pairs = [(0, 0, True)]
    return pairs


# ------------------------------------------------------------------------------------------------- analysis
def file_block(name: str, A: list[dict], B: list[dict], args) -> dict | None:
    """Per-item arrays and statistics of one eval name present in every run of both arms."""
    if not all(name in r["files"] for r in A + B):
        return None
    paths = {r["files"][name][0] for r in A + B}
    ids = None
    for r in A + B:
        s = set(r["files"][name][1])
        ids = s if ids is None else ids & s
    ids = sorted(ids)
    if not ids:
        return None
    arr = {arm: np.array([[[r["files"][name][1][i][m] for i in ids] for m in range(3)] for r in runs])
           for arm, runs in (("a", A), ("b", B))}                         # [runs, metric, items]
    a, b = arr["a"].mean(0), arr["b"].mean(0)
    d = b - a
    out = {"name": name, "paths": sorted(paths), "n": len(ids), "ids": ids, "d": d,
           "src_a": sorted({r["files"][name][2] for r in A}), "src_b": sorted({r["files"][name][2] for r in B}),
           "pool": "new" if all(NEW_RE.search(p) for p in paths) else ("ww" if all(WW_RE.search(p) for p in paths)
                                                                         else None)}
    for k, m in enumerate(METRICS):
        out[f"{m}_a"], out[f"{m}_b"], out[f"{m}_d"] = float(a[k].mean()), float(b[k].mean()), float(d[k].mean())
    pairs = []
    for i, j, unp in seed_pairs(A, B):
        ca, cb = arr["a"][i, 0], arr["b"][j, 0]
        bb, cc = int(np.sum((ca == 1) & (cb == 0))), int(np.sum((ca == 0) & (cb == 1)))
        pairs.append({"seed_a": A[i]["seed"], "seed_b": B[j]["seed"], "unpaired_seeds": unp, "b_a_only": bb,
                      "c_b_only": cc, "p": mcnemar(bb, cc), "psi": (bb + cc) / len(ids)})
    out["pairs"] = pairs
    return out


def stats(blocks: list[dict], args, key: str) -> dict:
    """Bootstrap CIs (stratified by file), sign-flip p, discordance and power for a set of file blocks."""
    rng = np.random.default_rng(R.stable("fa_pool", key))
    res = {"files": [b["name"] for b in blocks], "n": sum(b["n"] for b in blocks)}
    for k, m in enumerate(METRICS):
        ds = [b["d"][k] for b in blocks]
        iw, ew = boot_strata(ds, args.boot, rng)
        allv = np.concatenate(ds)
        res[f"{m}_d"] = float(allv.mean())
        res[f"{m}_ci"] = pct_ci(iw)
        res[f"{m}_d_eq"] = float(np.mean([x.mean() for x in ds]))
        res[f"{m}_ci_eq"] = pct_ci(ew)
        res[f"{m}_a"] = float(sum(b[f"{m}_a"] * b["n"] for b in blocks) / res["n"])
        res[f"{m}_b"] = float(sum(b[f"{m}_b"] * b["n"] for b in blocks) / res["n"])
    acc_d = np.concatenate([b["d"][0] for b in blocks])
    res["sign_flip_p"] = sign_flip(acc_d, args.flips, rng)
    npairs = min(len(b["pairs"]) for b in blocks)
    pairs = []
    for t in range(npairs):
        bb = sum(b["pairs"][t]["b_a_only"] for b in blocks)
        cc = sum(b["pairs"][t]["c_b_only"] for b in blocks)
        pairs.append({"seed_a": blocks[0]["pairs"][t]["seed_a"], "seed_b": blocks[0]["pairs"][t]["seed_b"],
                      "unpaired_seeds": blocks[0]["pairs"][t]["unpaired_seeds"], "b_a_only": bb, "c_b_only": cc,
                      "p": mcnemar(bb, cc), "psi": (bb + cc) / res["n"]})
    res["pairs"] = pairs
    psi = float(np.mean([p["psi"] for p in pairs])) if pairs else None
    var = float(np.var(acc_d, ddof=1)) if len(acc_d) > 1 else None
    res["psi"], res["var_d"] = psi, var
    res["power"] = {}
    for dl in args.deltas:
        de = dl / 100
        res["power"][f"+{dl:g}"] = {
            "n_needed": connor_n(psi, de, args.alpha, args.power) if psi is not None else None,
            "power_at_n": connor_power(psi, de, args.alpha, res["n"]) if psi is not None else None,
            "n_needed_var": var_n(var, de, args.alpha, args.power) if var is not None else None}
    return res


def analyse(args) -> dict:
    ra, rb = load_arm(args.a), load_arm(args.b)
    A, B = collect(ra, args.at_a, args), collect(rb, args.at_b, args)
    names = sorted({n for r in A + B for n in r["files"]})
    blocks, missing = [], {}
    for n in names:
        blk = file_block(n, A, B, args)
        if blk is None:
            missing[n] = {"a": [r["name"] for r in A if n not in r["files"]],
                          "b": [r["name"] for r in B if n not in r["files"]]}
            continue
        blk["stats"] = stats([blk], args, n)
        blocks.append(blk)
    pools = {}
    seen: dict = {}
    for blk in blocks:                  # the same file under two eval names counts once (first name)
        if blk["pool"] and blk["paths"][0] not in seen:
            seen[blk["paths"][0]] = blk
    real = list(seen.values())
    if real:
        pools["all"] = stats(real, args, "pool_all")
    new = [b for b in real if b["pool"] == "new"]
    if new:
        pools["new"] = stats(new, args, "pool_new")
    if args.pool:
        want = [x for x in args.pool.split(",") if x]
        sel = [b for b in blocks if b["name"] in want]
        if sel:
            pools["custom:" + ",".join(b["name"] for b in sel)] = stats(sel, args, "pool_custom")
    return {"args": vars(args), "a": [{k: r[k] for k in ("name", "seed", "epoch")} for r in A],
            "b": [{k: r[k] for k in ("name", "seed", "epoch")} for r in B],
            "label_a": args.label_a or ra[0]["arm"], "label_b": args.label_b or rb[0]["arm"],
            "files": [{k: v for k, v in b.items() if k not in ("d", "ids")} for b in blocks], "pools": pools,
            "missing": missing}


# ------------------------------------------------------------------------------------------------- report
def pt(x: float | None, nd: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{nd}f}"


def sd(x: float, ci: tuple) -> str:
    return f"{100 * x:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"


def fmt_p(p: float | None) -> str:
    return "–" if p is None else (f"{p:.3f}" if p >= 0.001 else f"{p:.1e}")


def fmt_n(x: float | None) -> str:
    return "–" if x is None else f"{math.ceil(x):,}"


def report(res: dict) -> str:
    la, lb = res["label_a"], res["label_b"]
    L = [f"# FA paired comparison: {lb} minus {la}", "",
         f"A = {la}: " + ", ".join(f"{r['name']} @ {'final' if r['epoch'] is None else 'e%g' % r['epoch']}"
                                   for r in res["a"]),
         f"B = {lb}: " + ", ".join(f"{r['name']} @ {'final' if r['epoch'] is None else 'e%g' % r['epoch']}"
                                   for r in res["b"]), "",
         f"Seeds averaged per item; paired item bootstrap ({res['args']['boot']} draws, 95% percentile); McNemar exact "
         f"per seed pair (b = A right only, c = B right only); power: Connor's McNemar n at alpha "
         f"{res['args']['alpha']}, power {res['args']['power']}, psi = observed discordance.", "",
         "| file | n | pool | ckpt A / B | Acc A | Acc B | ΔAcc [95% CI] | McNemar b / c, p | ΔAcc@3 [CI] | ΔMRR@3 "
         "[CI] | psi | n for " + " / ".join(f"+{d:g}" for d in res["args"]["deltas"]) + " |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(label, n, pool, ck, s):
        mc = "; ".join(f"{p['b_a_only']} / {p['c_b_only']}, {fmt_p(p['p'])}" + (" (unpaired seeds)"
                                                                                  if p["unpaired_seeds"] else "")
                       for p in s["pairs"]) or "–"
        if len(s["pairs"]) != 1:
            mc += f"; sign-flip {fmt_p(s['sign_flip_p'])}"
        ns = " / ".join(fmt_n(s["power"][f"+{d:g}"]["n_needed"]) for d in res["args"]["deltas"])
        psi = "–" if s["psi"] is None else f"{s['psi']:.3f}"
        return (f"| {label} | {n} | {pool} | {ck} | {pt(s['acc_a'])} | {pt(s['acc_b'])} | {sd(s['acc_d'], s['acc_ci'])} "
                f"| {mc} | {sd(s['acc3_d'], s['acc3_ci'])} | {sd(s['mrr3_d'], s['mrr3_ci'])} | {psi} | {ns} |")
    for f in res["files"]:
        ck = "/".join(f["src_a"]) + " / " + "/".join(f["src_b"])
        L.append(row(f"{f['name']} ({Path(f['paths'][0]).name})", f["n"], f["pool"] or "–", ck, f["stats"]))
    for k, s in res["pools"].items():
        L.append(row(f"**pool {k}** ({', '.join(s['files'])})", s["n"], "", "", s))
    L.append("")
    for k, s in res["pools"].items():
        L.append(f"- pool {k}: n {s['n']}; ΔAcc item-weighted {sd(s['acc_d'], s['acc_ci'])}, equal-weight over files "
                 f"{sd(s['acc_d_eq'], s['acc_ci_eq'])}; sign-flip p {fmt_p(s['sign_flip_p'])}; psi "
                 f"{'–' if s['psi'] is None else round(s['psi'], 3)}; " + "; ".join(
                     f"{d}: n needed {fmt_n(v['n_needed'])} (variance-based {fmt_n(v['n_needed_var'])}), power at "
                     f"n {s['n']} {pt(v['power_at_n'], 0)}%" for d, v in s["power"].items()))
    if res["missing"]:
        L += ["", "Left out (no predictions at the checkpoint in some run; --fallback-final scores them from the final "
              "predictions): " + "; ".join(f"{n} (A {len(v['a'])}, B {len(v['b'])} runs without)"
                                           for n, v in res["missing"].items())]
    return "\n".join(L)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", required=True, help="run-directory glob of arm A (one run per seed)")
    ap.add_argument("--b", required=True, help="run-directory glob of arm B; differences are B - A")
    ap.add_argument("--at-a", default="final", help="checkpoint of A: final, val or e<epoch>")
    ap.add_argument("--at-b", default="final", help="checkpoint of B: final, val or e<epoch>")
    ap.add_argument("--label-a", default="")
    ap.add_argument("--label-b", default="")
    ap.add_argument("--val", default="val", help="eval name that selects the checkpoint for --at val")
    ap.add_argument("--fallback-final", action="store_true",
                    help="score files without predictions at a mid point from the final predictions (final*)")
    ap.add_argument("--pool", default="", help="an extra pool: comma list of eval names")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--flips", type=int, default=20000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.8)
    ap.add_argument("--deltas", default="3,5", help="differences in points for the power estimate")
    ap.add_argument("--name", default="fa_pool")
    ap.add_argument("--out-dir", default="results/fa_pool")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    args.deltas = [float(x) for x in str(args.deltas).split(",") if x]
    res = analyse(args)
    out = Path(args.out_dir)
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(res, indent=1, default=float))
    md = report(res)
    (out / f"{args.name}.md").write_text(md + "\n")
    if not args.quiet:
        print(md)
    return res


if __name__ == "__main__":
    main()
