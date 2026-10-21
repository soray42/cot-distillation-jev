#!/usr/bin/env python3
"""Pre-registered readout of CHTC batch 14 (cloud/chtc/README_14.md), as a markdown report.

Card B (confirmation, seeds 13-15; seed 12 = screening seed, listed separately, never in a test):
  * Epoch rule: per method (A2, GFPQ), the epoch (1 or 2) with the higher seed-mean valx accuracy at the end of
    training (preds_valx.jsonl), over ALL of seeds 13/14/15; ties -> 1 epoch. Any of the 6 runs missing -> PENDING
    (provisional numbers shown, no verdict on fewer seeds).
  * 3-epoch follow-up rule: >= 2 of the 3 seeds with GFPQ 2-epoch valx > 1-epoch valx -> next batch adds GFPQ and A2
    at 3 epochs (seeds 13-15).
  * C1: cue-free pool (gsm8kx+gsm1kx), GFPQ - A2 at the epochs of the rule, paired by seed. PASS = seed mean >= +1.5
    points, all three per-seed differences > 0, two-level bootstrap 95% CI lower bound > 0; the seed-level paired t
    (df = 2) is reported, it is not part of the rule.
  * C1': the same difference epoch-matched (1 vs 1, 2 vs 2); PASS = both have the same non-zero sign as C1.
    C1 holds = C1 PASS and C1' PASS.
  * C2: cue-free pool, GFPQ 2 epochs - GFPQR 2 epochs. PASS = mean >= +1.5 and all three seeds > 0.
  * C1 and C2 are also given per half (gsm8kx, gsm1kx) with the half that carries the result named, and on the
    original pool (gsm8kv2+gsm1k, report only). 9B / 12B only if C1 holds and C2 passes.
Card A (TSF / PTAD pilot, seed 12):
  * Early stop ("use" detector): at the 1-epoch eval point of TSF, val accuracy forced with the teacher's vectors
    minus intact (metrics.json trajectory, keys val / val@force); < 2 points -> stop the slot line (FAIL).
  * G1 (end of training, gsm8kv2 = the first necessity set): TSF mean-ablation drop >= 2 and forceswap drop >= 3
    points; TSFN, TSFR, TSFX: both drops < 1 point.
  * G3: TSF >= A2 (seed 12, the same training problems; its epoch = card B's A2 epoch rule) + 2 points on both the
    original and the cue-free pool; the epoch-matched difference (A2 2 epochs) and TSF - GFPQ (seed 12, 2 epochs)
    are reported. Single seed: direction only, k = 1 (item bootstrap CI shown for information).
  * PTAD: the same accuracy gate; PTADR and PTADX must be clearly lower (README gives no number: +1.5 points on both
    pools, the memo's G3 margin for TSFR / TSFX).

Accuracy = argmax(probs) label == gold_label (paired_stats.top1); pools concatenate the sets keyed "<set>:<item_id>"
(paired_stats.load_matrix, which also checks that all runs of a comparison hold the same items and golds); paired
differences, the seed-level t and the two-level bootstrap come from paired_stats.paired_stats. All internal numbers
are fractions; the thresholds are in points and are converted only when printed.

  python scripts/analyze_batch14.py [--root results/chtc_repo/runs] [--boot 10000] [--out notes/batch14_readout.md]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import paired_stats as ps  # noqa: E402

MODEL = "Qwen3.5-4B"
DATA_VT = "student_gsm8k_tf_vt"
DATA_SI = "student_gsm8k_tf_si"
CONFIRM_SEEDS = (13, 14, 15)
SCREEN_SEED = 12
EPOCHS = (1, 2)
VALX = "valx"
TSF_ARMS = ("TSF", "TSFN", "TSFR", "TSFX", "TSF6")
PTAD_ARMS = ("PTAD", "PTADR", "PTADX")
TSF_CONTROLS = ("TSFN", "TSFR", "TSFX")

POOLS = {"cuefree": ("gsm8kx", "gsm1kx"), "gsm8kx": ("gsm8kx",), "gsm1kx": ("gsm1kx",),
         "orig": ("gsm8kv2", "gsm1k"), "gsm8kv2": ("gsm8kv2",), "gsm1k": ("gsm1k",)}
POOL_LABEL = {"cuefree": "无线索合并 gsm8kx+gsm1kx", "gsm8kx": "gsm8kx 一半", "gsm1kx": "gsm1kx 一半",
              "orig": "原测试集合并 gsm8kv2+gsm1k", "gsm8kv2": "gsm8kv2 一半", "gsm1k": "gsm1k 一半"}
B_POOLS = ("cuefree", "gsm8kx", "gsm1kx", "orig", "gsm8kv2", "gsm1k")
A_POOLS = ("orig", "cuefree", "gsm8kx", "gsm1kx")
HALVES = {"cuefree": ("gsm8kx", "gsm1kx"), "orig": ("gsm8kv2", "gsm1k")}

# thresholds (fractions; README_14 states them in points)
C1_MEAN = 0.015
C2_MEAN = 0.015
EARLY_STOP = 0.02
G1_MEAN_DROP = 0.02
G1_SWAP_DROP = 0.03
G1_CONTROL = 0.01
G3_MARGIN = 0.02
PTAD_CONTROL_MARGIN = 0.015
EPS = 1e-9
TIE = 1e-12                      # the epoch rule's tie tolerance (as paired_stats.select_epoch)

PASS, FAIL, PENDING = "PASS", "FAIL", "PENDING"


def ge(x, thr) -> bool:
    return x is not None and x >= thr - EPS


def lt(x, thr) -> bool:
    return x is not None and x < thr - EPS


def combine(parts) -> str:
    """AND of sub-verdicts: any FAIL -> FAIL, all PASS -> PASS, else PENDING."""
    parts = list(parts)
    if FAIL in parts:
        return FAIL
    if parts and all(p == PASS for p in parts):
        return PASS
    return PENDING


def pts(x, signed: bool = True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:+.2f}" if signed else f"{100 * x:.2f}"


def ci(lo, hi) -> str:
    return "n/a" if lo is None or hi is None else f"[{pts(lo)}, {pts(hi)}]"


def brief(items, n: int = 2) -> str:
    items = list(items)
    if len(items) <= n:
        return "; ".join(items)
    return "; ".join(items[:n]) + f" 等 {len(items)} 项（见文末 run 目录状态）"


# ----------------------------------------------------------------------------------------------------------
# Run dirs
# ----------------------------------------------------------------------------------------------------------

def vt_name(arm: str, ep: int, seed: int) -> str:
    return f"{arm}-c{ep}lr3ch-{MODEL}-{DATA_VT}-s{seed}"


def tsf_name(arm: str) -> str:
    """hpc/train_arm.sbatch with the batch-14 env files (PROBE=1, TAG=c2lr3ch, no TSF/PTAD knob set)."""
    return f"{arm}-pr-c2lr3ch-{MODEL}-{DATA_SI}-s{SCREEN_SEED}"


def tsf_regex(arm: str) -> re.Pattern:
    """The arm, then any knob segments (-tg<x>, -sk<k>, -al12-20 ...), then -pr-c2lr3ch-...; "TSF" never matches
    TSFN / TSFR / TSFX / TSF6 (the arm must be followed by '-')."""
    return re.compile(rf"^{re.escape(arm)}(?:-[^-]+)*?-pr-c2lr3ch-{re.escape(MODEL)}-{re.escape(DATA_SI)}"
                      rf"-s{SCREEN_SEED}$")


@dataclass
class Lookup:
    status: str                          # ok | missing | ambiguous | error
    expected: str
    run: ps.Run | None = None
    why: str = ""
    matches: list = field(default_factory=list)

    def has(self, sets) -> bool:
        return self.status == "ok" and all((self.run.path / f"preds_{s}.jsonl").exists() for s in sets)

    def lacks(self, sets) -> str:
        if self.status != "ok":
            return f"`{self.expected}` {self.why or self.status}"
        miss = [f"preds_{s}.jsonl" for s in sets if not (self.run.path / f"preds_{s}.jsonl").exists()]
        return f"`{self.run.name}` 缺 {', '.join(miss)}" if miss else ""


class Runs:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.names = sorted(p.name for p in self.root.iterdir() if p.is_dir()) if self.root.is_dir() else []

    def _make(self, name: str, seed: int, ep) -> Lookup:
        path = self.root / name
        try:
            args = ps.read_args(path)
        except Exception as e:                                          # noqa: BLE001
            return Lookup("error", name, why=f"metrics.json 读不了 ({e})")
        if args:
            if args.get("seed") is not None and int(args["seed"]) != seed:
                return Lookup("error", name, why=f"metrics.json 的 seed 是 {args['seed']}，目录名是 {seed}")
            if args.get("epochs") is not None:
                if ep is not None and abs(float(args["epochs"]) - float(ep)) > 1e-9:
                    return Lookup("error", name, why=f"metrics.json 的 epochs 是 {args['epochs']}，目录名是 {ep}")
                ep = float(args["epochs"])
        why = "" if args is not None else "无 metrics.json（未跑完？）"
        return Lookup("ok", name, ps.Run(name, path, seed, float(ep) if ep is not None else None), why=why)

    def vt(self, arm: str, ep: int, seed: int) -> Lookup:
        name = vt_name(arm, ep, seed)
        if not (self.root / name).is_dir():
            return Lookup("missing", name, why="目录不存在")
        return self._make(name, seed, ep)

    def tsf(self, arm: str) -> Lookup:
        rx = tsf_regex(arm)
        hits = [n for n in self.names if rx.match(n)]
        if not hits:
            return Lookup("missing", tsf_name(arm), why="目录不存在")
        if len(hits) > 1:
            return Lookup("ambiguous", tsf_name(arm), why="多个目录匹配: " + ", ".join(f"`{h}`" for h in hits),
                          matches=hits)
        return self._make(hits[0], SCREEN_SEED, None)


@contextmanager
def preds_cache():
    """Each preds file is parsed once per report (paired_stats.load_matrix reads through ps.read_preds)."""
    orig, cache = ps.read_preds, {}

    def cached(path):
        key = str(path)
        if key not in cache:
            cache[key] = orig(path)
        return cache[key]

    ps.read_preds = cached
    try:
        yield
    finally:
        ps.read_preds = orig


def read_metrics(lk: Lookup) -> dict | None:
    if lk.status != "ok":
        return None
    p = lk.run.path / "metrics.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:                                                   # noqa: BLE001
        return None


def read_trajectory(lk: Lookup) -> list | None:
    m = read_metrics(lk)
    if m is not None and m.get("trajectory"):
        return m["trajectory"]
    if lk.status == "ok" and (lk.run.path / "trajectory.json").exists():
        try:                                     # rewritten at every eval point: may be caught half-written
            with open(lk.run.path / "trajectory.json") as f:
                return json.load(f)
        except Exception:                                               # noqa: BLE001
            return None
    return None


# ----------------------------------------------------------------------------------------------------------
# Accuracy and paired comparisons (paired_stats)
# ----------------------------------------------------------------------------------------------------------

def accuracy(lk: Lookup, sets) -> dict:
    if not lk.has(sets):
        return {"status": "pending", "why": lk.lacks(sets)}
    try:
        m = ps.load_matrix([lk.run], list(sets))
    except Exception as e:                                              # noqa: BLE001
        return {"status": "error", "why": str(e)}
    c = m["correct"][0]
    return {"status": "ok", "acc": float(c.mean()), "n": int(c.size)}


def valx_accuracy(lk: Lookup) -> float | None:
    if not lk.has([VALX]):
        return None
    try:
        rows = ps.read_preds(lk.run.path / f"preds_{VALX}.jsonl")
        return sum(ps.top1(r)[0] for r in rows) / len(rows) if rows else None
    except Exception:                                                   # noqa: BLE001  (truncated file: pending)
        return None


@dataclass
class Opt:
    boot: int = 10000
    boot_seed: int = 0


def compare(la: list[Lookup], lb: list[Lookup], pool: str, opt: Opt) -> dict:
    """A - B paired by position (same seed). status ok = every pair usable; partial = provisional on fewer seeds."""
    sets = list(POOLS[pool])
    res = {"pool": pool, "sets": sets, "k_expected": len(la), "missing": []}
    pairs = []
    for a, b in zip(la, lb):
        miss = [x for x in (a.lacks(sets), b.lacks(sets)) if x]
        if miss:
            res["missing"] += miss
        else:
            pairs.append((a, b))
    if not pairs:
        res["status"] = "pending"
        return res
    runs_a, runs_b = [a.run for a, _ in pairs], [b.run for _, b in pairs]
    try:
        m = ps.load_matrix(runs_a + runs_b, sets)
    except Exception as e:                                              # noqa: BLE001
        res.update(status="error", why=str(e))
        return res
    k = len(pairs)
    st = ps.paired_stats(m["correct"][:k], m["correct"][k:], opt.boot, opt.boot_seed)
    res.update(status="ok" if k == len(la) else "partial", stats=st, seeds=[r.seed for r in runs_a],
               runs_a=[r.name for r in runs_a], runs_b=[r.name for r in runs_b])
    return res


def compare_pools(la, lb, pools, opt) -> dict:
    return {p: compare(la, lb, p, opt) for p in pools}


# ----------------------------------------------------------------------------------------------------------
# Card B
# ----------------------------------------------------------------------------------------------------------

def epoch_rule(lookups: dict, arm: str, seeds=CONFIRM_SEEDS) -> dict:
    """lookups[(arm, ep, seed)]; the epoch with the higher seed-mean valx over ALL seeds, ties -> 1."""
    table = {ep: {s: valx_accuracy(lookups[(arm, ep, s)]) for s in seeds} for ep in EPOCHS}
    missing = [lookups[(arm, ep, s)].lacks([VALX]) for ep in EPOCHS for s in seeds if table[ep][s] is None]
    means = {ep: (float(np.mean([v for v in table[ep].values() if v is not None]))
                  if any(v is not None for v in table[ep].values()) else None) for ep in EPOCHS}
    out = {"arm": arm, "table": table, "means": means, "missing": missing, "chosen": None}
    if missing:
        out.update(status=PENDING, why=f"valx 缺 {len(missing)} 个（需要 {len(EPOCHS) * len(seeds)} 个）")
        return out
    best = max(means.values())
    chosen = min(ep for ep in EPOCHS if means[ep] >= best - TIE)
    tie = all(abs(means[ep] - best) <= TIE for ep in EPOCHS)
    out.update(status="DECIDED", chosen=chosen, tie=tie,
               why=f"valx 均值 1 轮 {pts(means[1], False)} / 2 轮 {pts(means[2], False)} -> {chosen} 轮"
                   + ("（打平，选 1 轮）" if tie else ""))
    return out


def three_epoch_rule(er_gfpq: dict, seeds=CONFIRM_SEEDS) -> dict:
    t = er_gfpq["table"]
    known = [s for s in seeds if t[1][s] is not None and t[2][s] is not None]
    up = [s for s in known if t[2][s] - t[1][s] > TIE]
    unknown = len(seeds) - len(known)
    if len(up) >= 2:
        status = "TRIGGERED"
    elif len(up) + unknown < 2:
        status = "NOT TRIGGERED"
    else:
        status = PENDING
    return {"status": status, "up": up, "known": known,
            "why": f"GFPQ 2 轮 valx > 1 轮的种子: {', '.join(f's{s}' for s in up) or '无'}"
                   f"（{len(up)}/{len(known)} 个已出，需 ≥ 2）"}


def c1_criteria(st: dict, mean_thr: float = C1_MEAN) -> tuple[bool, list[str]]:
    ok_mean = ge(st["mean"], mean_thr)
    ok_all = all(d > 0 for d in st["d"])
    ok_ci = st["two_level"][0] > 0
    notes = [f"均值 {pts(st['mean'])} {'≥' if ok_mean else '<'} {pts(mean_thr)}",
             f"各种子 {' / '.join(pts(d) for d in st['d'])} {'都 > 0' if ok_all else '不全 > 0'}",
             f"两级 CI 下界 {pts(st['two_level'][0])} {'> 0' if ok_ci else '≤ 0'}"]
    return ok_mean and ok_all and ok_ci, notes


def c2_criteria(st: dict) -> tuple[bool, list[str]]:
    ok_mean = ge(st["mean"], C2_MEAN)
    ok_all = all(d > 0 for d in st["d"])
    notes = [f"均值 {pts(st['mean'])} {'≥' if ok_mean else '<'} {pts(C2_MEAN)}",
             f"各种子 {' / '.join(pts(d) for d in st['d'])} {'都 > 0' if ok_all else '不全 > 0'}",
             f"（两级 CI {ci(*st['two_level'])}，仅报告）"]
    return ok_mean and ok_all, notes


def verdict_from(cmp: dict | None, crit) -> dict:
    if cmp is None:
        return {"status": PENDING, "why": "轮数规则未定"}
    if cmp["status"] == "error":
        return {"status": PENDING, "why": "错误: " + cmp["why"]}
    if cmp["status"] != "ok":
        k = cmp["stats"]["k"] if "stats" in cmp else 0
        return {"status": PENDING, "why": f"只有 {k}/{cmp['k_expected']} 个种子；缺 " + brief(cmp["missing"])}
    ok, notes = crit(cmp["stats"])
    return {"status": PASS if ok else FAIL, "why": "；".join(notes)}


def sign(x: float) -> int:
    return 0 if abs(x) < 1e-12 else (1 if x > 0 else -1)


def c1p_verdict(c1: dict | None, al1: dict, al2: dict) -> dict:
    known = {e: c["stats"]["mean"] for e, c in ((1, al1), (2, al2)) if c["status"] == "ok"}
    txt = "；".join(f"{e} 轮对 {e} 轮 {pts(v)}" for e, v in known.items())
    if len(known) == 2 and (sign(known[1]) != sign(known[2]) or sign(known[1]) == 0):
        return {"status": FAIL, "why": txt + "：两个轮数对齐差方向不一致（或为 0）"}
    if c1 is None or c1["status"] != "ok":
        return {"status": PENDING, "why": (txt + "；" if txt else "") + "C1 未出"}
    s = sign(c1["stats"]["mean"])
    if len(known) < 2:
        if any(sign(v) != s for v in known.values()) or s == 0:
            return {"status": FAIL, "why": txt + f"：与 C1 {pts(c1['stats']['mean'])} 方向不一致"}
        return {"status": PENDING, "why": (txt + "；" if txt else "") + "轮数对齐比较未齐"}
    ok = s != 0 and all(sign(v) == s for v in known.values())
    return {"status": PASS if ok else FAIL,
            "why": txt + f"；C1 {pts(c1['stats']['mean'])} -> " + ("方向一致" if ok else "方向不一致")}


def halves_note(cmps: dict, pooled: str, crit) -> str:
    """Which half carries a pooled comparison: each half alone against the gate's criteria, and the item-weighted
    share of the pooled difference."""
    h = HALVES[pooled]
    if cmps[pooled]["status"] != "ok" or any(cmps[x]["status"] != "ok" for x in h):
        return "（两半未齐，暂不能判断哪一半撑起）"
    st = cmps[pooled]["stats"]
    alone = {x: crit(cmps[x]["stats"])[0] for x in h}
    parts = [f"{x} 单独 {pts(cmps[x]['stats']['mean'])}（{'过线' if alone[x] else '不过线'}）" for x in h]
    n_tot = st["n_items"]
    share = {}
    if abs(st["mean"]) > 1e-12:
        share = {x: cmps[x]["stats"]["n_items"] * cmps[x]["stats"]["mean"] / (n_tot * st["mean"]) for x in h}
        parts.append("按题数加权，合并差的 " + " / ".join(f"{100 * share[x]:.0f}% 来自 {x}" for x in h))
    passing = [x for x in h if alone[x]]
    if len(passing) == 1:
        concl = f"只有 {passing[0]} 单独过线 -> 结论由 {passing[0]} 撑起"
    elif len(passing) == 2:
        concl = "两半各自都过线"
    elif share and st["mean"] > 0:
        top = max(h, key=lambda x: share[x])
        concl = f"两半都不单独过线；合并差主要来自 {top}"
    else:
        concl = "两半都不单独过线"
    return "；".join(parts) + "。" + concl + "。"


def card_b(runs: Runs, opt: Opt) -> dict:
    S = CONFIRM_SEEDS
    lk = {(arm, ep, s): runs.vt(arm, ep, s) for arm in ("A2", "GFPQ") for ep in EPOCHS for s in S}
    lk.update({("GFPQR", 2, s): runs.vt("GFPQR", 2, s) for s in S})
    er = {arm: epoch_rule(lk, arm) for arm in ("A2", "GFPQ")}
    three = three_epoch_rule(er["GFPQ"])

    def cmp(a_arm, a_ep, b_arm, b_ep):
        return compare_pools([lk[(a_arm, a_ep, s)] for s in S], [lk[(b_arm, b_ep, s)] for s in S], B_POOLS, opt)

    aligned = {ep: cmp("GFPQ", ep, "A2", ep) for ep in EPOCHS}
    eA, eG = er["A2"]["chosen"], er["GFPQ"]["chosen"]
    c1 = (aligned[eA] if eA == eG else cmp("GFPQ", eG, "A2", eA)) if eA and eG else None
    c2 = cmp("GFPQ", 2, "GFPQR", 2)
    v_c1 = verdict_from(c1["cuefree"] if c1 else None, c1_criteria)
    v_c1p = c1p_verdict(c1["cuefree"] if c1 else None, aligned[1]["cuefree"], aligned[2]["cuefree"])
    v_c1_holds = {"status": combine([v_c1["status"], v_c1p["status"]]), "why": f"C1 {v_c1['status']}，C1' {v_c1p['status']}"}
    v_c2 = verdict_from(c2["cuefree"], c2_criteria)
    v_go = {"status": combine([v_c1_holds["status"], v_c2["status"]]),
            "why": f"C1 成立 {v_c1_holds['status']}，C2 {v_c2['status']}"}
    return {"lookups": lk, "epoch_rule": er, "three_epoch": three, "aligned": aligned, "c1": c1, "c2": c2,
            "v_c1": v_c1, "v_c1p": v_c1p, "v_c1_holds": v_c1_holds, "v_c2": v_c2, "v_go": v_go,
            "eA": eA, "eG": eG}


# ----------------------------------------------------------------------------------------------------------
# Seed 12 (screening seed, not in any test)
# ----------------------------------------------------------------------------------------------------------

S12_RUNS = (("A2", 1), ("A2", 2), ("GFPQ", 1), ("GFPQ", 2), ("GFPQVTR", 2))


def s12_section(runs: Runs, opt: Opt, eA, eG) -> dict:
    lk = {(arm, ep): runs.vt(arm, ep, SCREEN_SEED) for arm, ep in S12_RUNS}
    acc = {key: {p: accuracy(l, POOLS[p]) for p in B_POOLS} for key, l in lk.items()}
    for key, l in lk.items():
        acc[key]["valx"] = valx_accuracy(l)
        acc[key]["val"] = accuracy(l, ["val"])
    comps = [(f"GFPQ 1 轮 − A2 1 轮", ("GFPQ", 1), ("A2", 1)), (f"GFPQ 2 轮 − A2 2 轮", ("GFPQ", 2), ("A2", 2))]
    if eA and eG and eA != eG:
        comps.append((f"GFPQ {eG} 轮 − A2 {eA} 轮（B 卡定的轮数）", ("GFPQ", eG), ("A2", eA)))
    comps.append(("GFPQ 2 轮 − GFPQVTR 2 轮（第 13 批的打乱对照，验证树描述打乱；不是 GFPQR）", ("GFPQ", 2), ("GFPQVTR", 2)))
    out = []
    for label, a, b in comps:
        out.append((label, compare_pools([lk[a]], [lk[b]], ("cuefree", "gsm8kx", "gsm1kx", "orig"), opt)))
    return {"lookups": lk, "acc": acc, "comps": out}


# ----------------------------------------------------------------------------------------------------------
# Card A
# ----------------------------------------------------------------------------------------------------------

def detector_rows(lk: Lookup) -> list[dict]:
    """Every eval point of a TSF run: the four accuracies on each detector set (keys <set>@mean|force|forceswap)."""
    traj = read_trajectory(lk) or []
    rows = []
    for r in traj:
        sets = sorted({k.split("@")[0] for k in r if isinstance(k, str) and k.endswith("@force")})
        for s in sets:
            g = lambda key: (r.get(key) or {}).get("acc")              # noqa: E731
            rows.append({"epoch": r.get("epoch"), "set": s, "intact": g(s), "mean": g(f"{s}@mean"),
                         "force": g(f"{s}@force"), "forceswap": g(f"{s}@forceswap"),
                         "detect": (r.get(s) or {}).get("detect") or {}})
    return rows


def early_stop(lk: Lookup) -> dict:
    """README_14 早停判据: at 1 epoch, val accuracy forced with the teacher's vectors minus intact < 2 -> stop."""
    if lk.status != "ok":
        return {"status": PENDING, "why": lk.lacks([])}
    traj = read_trajectory(lk)
    if not traj:
        return {"status": PENDING, "why": f"`{lk.run.name}` 无 trajectory"}
    row = next((r for r in traj if r.get("epoch") is not None and abs(float(r["epoch"]) - 1.0) < 1e-6), None)
    if row is None:
        return {"status": PENDING, "why": f"`{lk.run.name}` trajectory 无 1 轮评测点"}
    sets = [k.split("@")[0] for k in row if isinstance(k, str) and k.endswith("@force")]
    if not sets:
        return {"status": PENDING, "why": f"`{lk.run.name}` 1 轮评测点无 <set>@force 键（检测器没跑？）"}
    s = "val" if "val" in sets else sets[0]
    intact, force = (row.get(s) or {}).get("acc"), (row.get(f"{s}@force") or {}).get("acc")
    if intact is None or force is None:
        return {"status": PENDING, "why": f"`{lk.run.name}` 1 轮 {s} / {s}@force 无 acc"}
    diff = force - intact
    note = "" if s == "val" else f"（无 val@force，改用 {s}）"
    det = (row.get(s) or {}).get("detect") or {}
    status = FAIL if lt(diff, EARLY_STOP) else PASS
    why = (f"1 轮 {s}: 强制 {pts(force, False)} − 原样 {pts(intact, False)} = {pts(diff)} "
           f"{'<' if status == FAIL else '≥'} +2.00{note}" + ("  -> 停掉槽位这条线" if status == FAIL else ""))
    return {"status": status, "why": why, "set": s, "diff": diff, "intact": intact, "force": force,
            "mean": (row.get(f"{s}@mean") or {}).get("acc"), "forceswap": (row.get(f"{s}@forceswap") or {}).get("acc"),
            "detect": det}


def tsf_drops(lk: Lookup) -> dict:
    """End-of-training necessity drops (intact - mode, fractions) on the first necessity set (gsm8kv2)."""
    m = read_metrics(lk)
    if m is None:
        return {"status": "pending", "why": lk.lacks([]) if lk.status != "ok" else f"`{lk.run.name}` 无 metrics.json"}
    pil = m.get("tsf_pilot") or {}
    sn = m.get("slot_necessity") or {}
    out = {"status": "ok", "src": "tsf_pilot", "set": pil.get("necessity_set", "gsm8kv2")}
    if pil.get("mean_drop") is not None and pil.get("forceswap_drop") is not None:
        out.update(mean_drop=pil["mean_drop"], forceswap_drop=pil["forceswap_drop"],
                   mean_drop_chain=pil.get("mean_drop_chain"), forceswap_drop_chain=pil.get("forceswap_drop_chain"),
                   force_drop=pil.get("force_drop"))
    else:
        g = sn.get("gsm8kv2") or {}
        if (g.get("mean") or {}).get("acc_drop") is None or (g.get("forceswap") or {}).get("acc_drop") is None:
            return {"status": "pending", "why": f"`{lk.run.name}` 无 tsf_pilot / slot_necessity.gsm8kv2 的 mean、forceswap"}
        out.update(src="slot_necessity", set="gsm8kv2", mean_drop=g["mean"]["acc_drop"],
                   forceswap_drop=g["forceswap"]["acc_drop"], mean_drop_chain=g["mean"].get("acc_drop_chain"),
                   forceswap_drop_chain=g["forceswap"].get("acc_drop_chain"),
                   force_drop=(g.get("force") or {}).get("acc_drop"))
    other = {}
    for s, modes in sn.items():
        if s != out["set"]:
            other[s] = {md: (v or {}).get("acc_drop") for md, v in modes.items()}
    out["other_sets"] = other
    return out


def g1_verdict(drops: dict) -> dict:
    parts, notes = [], []
    t = drops["TSF"]
    if t["status"] != "ok":
        parts.append(PENDING)
        notes.append("TSF " + t["why"])
    else:
        ok = ge(t["mean_drop"], G1_MEAN_DROP) and ge(t["forceswap_drop"], G1_SWAP_DROP)
        parts.append(PASS if ok else FAIL)
        notes.append(f"TSF（{t['set']}）mean 掉 {pts(t['mean_drop'])}（需 ≥ +2）、forceswap 掉 {pts(t['forceswap_drop'])}"
                     f"（需 ≥ +3）-> {'过' if ok else '不过'}")
    for arm in TSF_CONTROLS:
        c = drops[arm]
        if c["status"] != "ok":
            parts.append(PENDING)
            notes.append(f"{arm} pending")
            continue
        ok = lt(c["mean_drop"], G1_CONTROL) and lt(c["forceswap_drop"], G1_CONTROL)
        parts.append(PASS if ok else FAIL)
        notes.append(f"{arm} mean {pts(c['mean_drop'])} / forceswap {pts(c['forceswap_drop'])}（都需 < 1）"
                     f"-> {'过' if ok else '不过'}")
    return {"status": combine(parts), "why": "；".join(notes)}


def margin_verdict(cmps: dict, margin: float, pools=("orig", "cuefree")) -> tuple[str, str]:
    parts, notes = [], []
    for p in pools:
        c = cmps[p]
        if c["status"] != "ok":
            parts.append(PENDING)
            notes.append(f"{POOL_LABEL[p]} pending")
            continue
        d = c["stats"]["mean"]
        ok = ge(d, margin)
        parts.append(PASS if ok else FAIL)
        notes.append(f"{POOL_LABEL[p]} {pts(d)} {'≥' if ok else '<'} {pts(margin)}")
    return combine(parts), "，".join(notes)


def accuracy_gate(cmp_by_ep: dict, eA, label: str) -> dict:
    """X - A2(eA) >= +2 on both pools; with eA undecided, a verdict only if both candidate epochs agree."""
    per = {ep: margin_verdict(cmp_by_ep[ep], G3_MARGIN) for ep in EPOCHS}
    if eA is not None:
        st, why = per[eA]
        return {"status": st, "why": f"{label} − A2 {eA} 轮（valx 规则）: {why}"}
    sts = {per[ep][0] for ep in EPOCHS}
    txt = "；".join(f"若 A2 {ep} 轮: {per[ep][1]}" for ep in EPOCHS)
    if len(sts) == 1 and PENDING not in sts:
        return {"status": sts.pop(), "why": f"A2 轮数未定，但两个候选轮数结论相同 — {txt}"}
    return {"status": PENDING, "why": f"A2 轮数未定（B 卡 valx 规则）— {txt}"}


def card_a(runs: Runs, opt: Opt, eA) -> dict:
    lk = {arm: runs.tsf(arm) for arm in TSF_ARMS + PTAD_ARMS}
    a2 = {ep: runs.vt("A2", ep, SCREEN_SEED) for ep in EPOCHS}
    gfpq2 = runs.vt("GFPQ", 2, SCREEN_SEED)
    det = {arm: detector_rows(lk[arm]) for arm in TSF_ARMS}
    es = early_stop(lk["TSF"])
    drops = {arm: tsf_drops(lk[arm]) for arm in TSF_ARMS}
    g1 = g1_verdict(drops)

    def cp(a, b):
        return compare_pools([a], [b], A_POOLS, opt)

    tsf_vs = {("A2", ep): cp(lk["TSF"], a2[ep]) for ep in EPOCHS}
    tsf_vs[("GFPQ", 2)] = cp(lk["TSF"], gfpq2)
    for arm in ("TSFR", "TSFX"):
        tsf_vs[(arm, None)] = cp(lk["TSF"], lk[arm])
    g3 = accuracy_gate({ep: tsf_vs[("A2", ep)] for ep in EPOCHS}, eA, "TSF")
    ptad_vs = {("A2", ep): cp(lk["PTAD"], a2[ep]) for ep in EPOCHS}
    for arm in ("PTADR", "PTADX"):
        ptad_vs[(arm, None)] = cp(lk["PTAD"], lk[arm])
    p_acc = accuracy_gate({ep: ptad_vs[("A2", ep)] for ep in EPOCHS}, eA, "PTAD")
    p_ctl = [margin_verdict(ptad_vs[(arm, None)], PTAD_CONTROL_MARGIN) for arm in ("PTADR", "PTADX")]
    ptad = {"status": combine([p_acc["status"]] + [s for s, _ in p_ctl]),
            "why": p_acc["why"] + "；PTAD − PTADR: " + p_ctl[0][1] + "；PTAD − PTADX: " + p_ctl[1][1]}
    acc = {}
    for arm in TSF_ARMS + PTAD_ARMS:
        acc[arm] = {p: accuracy(lk[arm], POOLS[p]) for p in A_POOLS}
        acc[arm]["valx"] = valx_accuracy(lk[arm])
        acc[arm]["val"] = accuracy(lk[arm], ["val"])
    return {"lookups": lk, "a2": a2, "gfpq2": gfpq2, "detector": det, "early_stop": es, "drops": drops, "g1": g1,
            "tsf_vs": tsf_vs, "g3": g3, "ptad_vs": ptad_vs, "ptad": ptad, "acc": acc, "eA": eA}


# ----------------------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------------------

def analyze(root: Path, opt: Opt) -> dict:
    runs = Runs(root)
    with preds_cache():
        b = card_b(runs, opt)
        s12 = s12_section(runs, opt, b["eA"], b["eG"])
        a = card_a(runs, opt, b["eA"])
    return {"root": str(root), "opt": opt, "b": b, "s12": s12, "a": a}


CMP_HEAD = ("| 比较 | 集合 | k | 准确率 A / B | 各种子 Δ | 均值 Δ | 两级 bootstrap 95% CI | seed-t 95% CI, p | "
            "题级 bootstrap 95% CI |\n|---|---|---|---|---|---|---|---|---|")


def cmp_row(label: str, c: dict) -> str:
    pool = POOL_LABEL[c["pool"]]
    if c["status"] in ("pending", "error"):
        why = c.get("why") or "缺 " + brief(c["missing"])
        return f"| {label} | {pool} | 0/{c['k_expected']} | {'ERROR' if c['status'] == 'error' else 'pending'}: {why} |" \
               + " |" * 5
    st, t = c["stats"], c["stats"]["t"]
    k = f"{st['k']}" + ("" if c["status"] == "ok" else f"/{c['k_expected']}（临时）")
    ds = " / ".join(f"s{s} {pts(d)}" for s, d in zip(c["seeds"], st["d"]))
    tci = "n/a (k=1)" if t["lo"] is None else f"{ci(t['lo'], t['hi'])} (df={t['df']}), p={ps.fmt_p(t['p'])}"
    return (f"| {label} | {pool} | {k} | {pts(np.mean(st['acc_a']), False)} / {pts(np.mean(st['acc_b']), False)} | "
            f"{ds} | **{pts(st['mean'])}** | {ci(*st['two_level'])} | {tci} | {ci(*st['item_boot'])} |")


def acc_cell(a) -> str:
    if a is None:
        return "n/a"
    if isinstance(a, float):
        return pts(a, False)
    return pts(a["acc"], False) if a["status"] == "ok" else ("ERROR" if a["status"] == "error" else "pending")


def render(res: dict) -> str:
    b, a, s12, opt = res["b"], res["a"], res["s12"], res["opt"]
    er = b["epoch_rule"]
    L = ["# 第 14 批预注册读数（cloud/chtc/README_14.md）", "",
         f"run 目录: `{res['root']}`；准确率 = argmax(probs) 的标签 == gold_label；差 = 前者 − 后者，单位为百分点；"
         f"bootstrap B = {opt.boot}（seed {opt.boot_seed}）。合并集按 `<set>:<item_id>` 拼接（paired_stats.load_matrix），"
         "配对 t 与两级 bootstrap 来自 paired_stats.paired_stats。", ""]

    def er_txt(x):
        return x["why"] if x["status"] == "DECIDED" else f"{x['why']}；临时均值 1 轮 {pts(x['means'][1], False)} / 2 轮 {pts(x['means'][2], False)}"

    L += ["## 总览", "", "| 关卡 | 判定 | 决定它的数字 |", "|---|---|---|",
          f"| B·轮数规则 A2 | {er['A2']['status'] if er['A2']['status'] == PENDING else 'DECIDED: ' + str(er['A2']['chosen']) + ' 轮'} | {er_txt(er['A2'])} |",
          f"| B·轮数规则 GFPQ | {er['GFPQ']['status'] if er['GFPQ']['status'] == PENDING else 'DECIDED: ' + str(er['GFPQ']['chosen']) + ' 轮'} | {er_txt(er['GFPQ'])} |",
          f"| B·C1 主检验（无线索合并，GFPQ − A2，按规则定的轮数） | **{b['v_c1']['status']}** | {b['v_c1']['why']} |",
          f"| B·C1' 轮数对齐（方向须与 C1 一致） | **{b['v_c1p']['status']}** | {b['v_c1p']['why']} |",
          f"| B·C1 成立（C1 且 C1'） | **{b['v_c1_holds']['status']}** | {b['v_c1_holds']['why']} |",
          f"| B·C2 内容对照（GFPQ 2 轮 − GFPQR 2 轮） | **{b['v_c2']['status']}** | {b['v_c2']['why']} |",
          f"| B·3 轮补跑规则 | {b['three_epoch']['status']} | {b['three_epoch']['why']} |",
          f"| B·上 9B / 12B（C1 成立 且 C2 过） | **{b['v_go']['status']}** | {b['v_go']['why']} |",
          f"| A·早停（use 检测器，TSF 1 轮 val 强制 − 原样 ≥ 2） | **{a['early_stop']['status']}** | {a['early_stop']['why']} |",
          f"| A·G1 必要性 | **{a['g1']['status']}** | {a['g1']['why']} |",
          f"| A·G3 准确率（TSF ≥ A2 + 2，两套测试集） | **{a['g3']['status']}** | {a['g3']['why']} |",
          f"| A·PTAD（同门槛；PTADR / PTADX 低 ≥ 1.5） | **{a['ptad']['status']}** | {a['ptad']['why']} |", ""]

    # ---- card B
    L += ["## B. 无线索测试集上的确认（种子 13 / 14 / 15）", "",
          "### 轮数规则（valx，训练结束；三个种子的平均，打平选 1 轮）", "",
          "| 方法 | 轮数 | s13 | s14 | s15 | 均值 |", "|---|---|---|---|---|---|"]
    for arm in ("A2", "GFPQ"):
        x = er[arm]
        for ep in EPOCHS:
            mark = " ←" if x["chosen"] == ep else ""
            L.append(f"| {arm} | {ep}{mark} | " + " | ".join(acc_cell(x["table"][ep][s]) for s in CONFIRM_SEEDS)
                     + f" | {acc_cell(x['means'][ep])} |")
    for arm in ("A2", "GFPQ"):
        L.append(f"- {arm}: {er[arm]['status']} — {er[arm]['why']}"
                 + ("" if not er[arm]["missing"] else "；缺: " + brief(er[arm]["missing"], 3)))
    L += ["", f"### 3 轮补跑规则（现在不加）: {b['three_epoch']['status']}", "",
          f"- {b['three_epoch']['why']}。触发则下一批补 GFPQ 3 轮和 A2 3 轮（种子 13–15）。", ""]

    def b_block(title, cmps, crit, verdict, pooled_pools=("cuefree", "gsm8kx", "gsm1kx")):
        out = [f"### {title}", ""]
        if verdict is not None:
            out += [f"**{verdict['status']}** — {verdict['why']}", ""]
        if cmps is None:
            out += ["（轮数规则未定，暂不能比较）", ""]
            return out
        out.append(CMP_HEAD)
        out += [cmp_row(title.split("：")[0], cmps[p]) for p in pooled_pools]
        out += ["", "哪一半撑起: " + halves_note(cmps, "cuefree", crit), ""]
        return out

    eA, eG = b["eA"], b["eG"]
    c1_title = f"C1 主检验：GFPQ {eG or '?'} 轮 − A2 {eA or '?'} 轮（无线索合并）"
    L += b_block(c1_title, b["c1"], c1_criteria, b["v_c1"])
    L += [f"### C1' 轮数对齐：GFPQ − A2，1 对 1、2 对 2", "", f"**{b['v_c1p']['status']}** — {b['v_c1p']['why']}", "",
          CMP_HEAD]
    for ep in EPOCHS:
        L += [cmp_row(f"{ep} 轮对 {ep} 轮", b["aligned"][ep][p]) for p in ("cuefree", "gsm8kx", "gsm1kx")]
    L.append("")
    for ep in EPOCHS:
        L.append(f"- {ep} 轮对 {ep} 轮，哪一半撑起（按 C1 的标准）: " + halves_note(b["aligned"][ep], "cuefree", c1_criteria))
    L.append("")
    L += b_block("C2 内容对照：GFPQ 2 轮 − GFPQR 2 轮（无线索合并）", b["c2"], c2_criteria, b["v_c2"])
    L += ["### 另报：原测试集（gsm8kv2 + gsm1k）上的同样差值（不判定）", "", CMP_HEAD]
    orig_rows = []
    if b["c1"] is not None:
        orig_rows.append((f"C1: GFPQ {eG} − A2 {eA}", b["c1"]))
    orig_rows += [(f"C1': {ep} 对 {ep}", b["aligned"][ep]) for ep in EPOCHS] + [("C2: GFPQ 2 − GFPQR 2", b["c2"])]
    for label, cm in orig_rows:
        L += [cmp_row(label, cm[p]) for p in ("orig", "gsm8kv2", "gsm1k")]
    L.append("")

    # ---- seed 12
    L += ["## 种子 12（筛选种子：只单独列出，不进确认检验，也不进轮数规则）", "",
          "| run | 原合并 | gsm8kv2 | gsm1k | 无线索合并 | gsm8kx | gsm1kx | valx | val |", "|---|---|---|---|---|---|---|---|---|"]
    for key, l in s12["lookups"].items():
        r = s12["acc"][key]
        name = f"`{l.run.name}`" if l.status == "ok" else f"`{l.expected}`（{l.status}）"
        L.append(f"| {name} | " + " | ".join(acc_cell(r[p]) for p in ("orig", "gsm8kv2", "gsm1k", "cuefree", "gsm8kx",
                                                                        "gsm1kx")) + f" | {acc_cell(r['valx'])} | {acc_cell(r['val'])} |")
    L += ["", "k = 1：两级 bootstrap 即题级 bootstrap，无 seed-t。", "", CMP_HEAD]
    for label, cm in s12["comps"]:
        L += [cmp_row(label, cm[p]) for p in ("cuefree", "gsm8kx", "gsm1kx", "orig")]
    L.append("")

    # ---- card A
    L += ["## A. 教师状态槽位 + 强制读取（TSF / PTAD，种子 12）", "", "### 早停判据 / use 检测器", "",
          f"**{a['early_stop']['status']}** — {a['early_stop']['why']}", ""]
    if a["early_stop"].get("detect"):
        d = a["early_stop"]["detect"]
        L += [f"- 带链题子集（n_chain = {d.get('n_chain')}，仅参考）: force − intact {pts(d.get('force_minus_intact'))}，"
              f"intact − mean {pts(d.get('intact_minus_mean'))}，intact − forceswap {pts(d.get('intact_minus_forceswap'))}", ""]
    L += ["| 臂 | 轮 | 集合 | 原样 | 换平均值 | 强制老师向量 | 强制别题向量 | 强制 − 原样 | 原样 − 平均 | 原样 − 别题 | 带链子集 force − intact |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    any_det = False
    for arm in TSF_ARMS:
        for r in a["detector"][arm]:
            any_det = True
            f = lambda x, y: None if x is None or y is None else x - y   # noqa: E731
            L.append(f"| {arm} | {r['epoch']} | {r['set']} | {acc_cell(r['intact'])} | {acc_cell(r['mean'])} | "
                     f"{acc_cell(r['force'])} | {acc_cell(r['forceswap'])} | {pts(f(r['force'], r['intact']))} | "
                     f"{pts(f(r['intact'], r['mean']))} | {pts(f(r['intact'], r['forceswap']))} | "
                     f"{pts(r['detect'].get('force_minus_intact'))} |")
    if not any_det:
        L.append("| pending | | | | | | | | | | |")
    L += ["", "### G1 必要性（训练结束，第一个必要性集 gsm8kv2）", "", f"**{a['g1']['status']}** — {a['g1']['why']}", "",
          "| 臂 | 来源 | 集合 | mean 掉分 | forceswap 掉分 | force 掉分 | mean 掉分（带链） | forceswap 掉分（带链） | 其他集合 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for arm in TSF_ARMS:
        x = a["drops"][arm]
        if x["status"] != "ok":
            L.append(f"| {arm} | pending: {x['why']} | | | | | | | |")
            continue
        oth = "; ".join(f"{s}: " + ", ".join(f"{m} {pts(v)}" for m, v in md.items()) for s, md in x["other_sets"].items())
        L.append(f"| {arm} | {x['src']} | {x['set']} | {pts(x['mean_drop'])} | {pts(x['forceswap_drop'])} | "
                 f"{pts(x.get('force_drop'))} | {pts(x.get('mean_drop_chain'))} | {pts(x.get('forceswap_drop_chain'))} | {oth or '–'} |")
    L += ["", "门槛（README_14）：TSF mean ≥ 2 且 forceswap ≥ 3；TSFN、TSFR、TSFX 两种掉分都 < 1（备忘录 §3.8 里 TSFN 只作归因、"
          "不进通过线，这里按 README_14 计入）。TSF6 只列出。", ""]
    L += ["### G3 准确率", "", f"**{a['g3']['status']}** — {a['g3']['why']}", "",
          "基线 A2 为第 13 批种子 12、同一批训练题（student_gsm8k_tf_vt）；A2 的轮数由 B 卡三个种子的 valx 平均定"
          f"（当前: {str(eA) + ' 轮' if eA else '未定'}）。单种子只看方向，过线才补种子。", "", CMP_HEAD]
    for (arm, ep), cm in a["tsf_vs"].items():
        lab = f"TSF − {arm}" + (f" {ep} 轮 s12" if ep else "")
        if arm == "A2" and ep == 2:
            lab += "（轮数对齐）"
        if arm == "A2" and ep == eA:
            lab += "（valx 规则）"
        L += [cmp_row(lab, cm[p]) for p in A_POOLS]
    L += ["", "### PTAD", "", f"**{a['ptad']['status']}** — {a['ptad']['why']}", "",
          "README_14 只写\"PTADR、PTADX 要明显更低\"，没给数；这里用备忘录 G3 对 TSFR / TSFX 的 +1.5 分（两套测试集都要）。", "",
          CMP_HEAD]
    for (arm, ep), cm in a["ptad_vs"].items():
        lab = f"PTAD − {arm}" + (f" {ep} 轮 s12" if ep else "")
        if arm == "A2" and ep == eA:
            lab += "（valx 规则）"
        L += [cmp_row(lab, cm[p]) for p in ("orig", "cuefree")]
    L += ["", "### 各臂终点准确率", "", "| 臂 | run | 原合并 | 无线索合并 | gsm8kx | gsm1kx | valx | val |",
          "|---|---|---|---|---|---|---|---|"]
    for arm in TSF_ARMS + PTAD_ARMS:
        l, r = a["lookups"][arm], a["acc"][arm]
        name = f"`{l.run.name}`" if l.status == "ok" else f"{l.status}: {l.why}"
        L.append(f"| {arm} | {name} | " + " | ".join(acc_cell(r[p]) for p in A_POOLS)
                 + f" | {acc_cell(r['valx'])} | {acc_cell(r['val'])} |")
    L.append("")

    # ---- run status
    L += ["## run 目录状态", ""]
    rows = [(f"B {arm} {ep} 轮 s{s}", l) for (arm, ep, s), l in b["lookups"].items()]
    rows += [(f"A {arm}", l) for arm, l in a["lookups"].items()]
    rows += [(f"s12 {arm} {ep} 轮", l) for (arm, ep), l in s12["lookups"].items()]
    n_ok = sum(1 for _, l in rows if l.status == "ok")
    L.append(f"{n_ok}/{len(rows)} 个目录在。")
    L.append("")
    for label, l in rows:
        if l.status == "ok":
            extra = "" if l.has([VALX]) else "（无 preds_valx）"
            extra += f"（{l.why}）" if l.why else ""
            L.append(f"- {label}: ok `{l.run.name}`{extra}")
        else:
            L.append(f"- {label}: **{l.status}** `{l.expected}` {l.why}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=ps.DEFAULT_ROOT)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="also write the markdown here")
    args = ap.parse_args(argv)
    root = Path(args.root)
    root = root if root.is_absolute() else (Path.cwd() / root if (Path.cwd() / root).exists() else ROOT / root)
    md = render(analyze(root, Opt(boot=args.boot, boot_seed=args.boot_seed)))
    print(md, end="")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
