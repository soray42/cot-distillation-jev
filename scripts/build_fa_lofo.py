#!/usr/bin/env python3
"""Leave-one-framework-out (LOFO) fold files for the failure-attribution study, materialised from
data/student_fa_lofo/plan.json (scripts/build_fa_new_sources.py): for every held-out framework, the training pools of
the other frameworks minus their trajectories that overlap the fold's held-out test files, as one A2 file (answer only,
the pools' plain rows) and one tree-forcing file (the same rows in the same order with the reasoning-tree nodes).

Standard library only (no numpy, no tokenizer), Python >= 3.8, so it runs on the HPC login node as well as locally;
rows are copied as the pools' raw lines (byte for byte), so the outputs are identical wherever it runs (md5sums.txt).

  python3 scripts/build_fa_lofo.py                       # every fold, main + capped variants, under data/
  python3 scripts/build_fa_lofo.py --check MANIFEST      # rebuild in memory and compare with a manifest's md5s
  python3 scripts/build_fa_lofo.py --root ~/cotd         # on the login node (data/ under ~/cotd)

Pools (plan.json names -> A2 dir, tree dir):
  sf_alg  student_fa_sf_alg        student_fa_sf_alg_t3      (StepFinder regenerated Who&When, v2 trees + v3 clean
  sf_hc   student_fa_sf_hc         student_fa_sf_hc_t3        nodes, index-free pointer render)
  aftraj  student_fa_aftraj        student_fa_aftraj_t3
  real    student_fa_real          student_fa_real_t3
  agtr    student_fa_agtr_diag     student_fa_agtr_diag_t2f  (AgenTracer, Diagnosis Error lines ONLY; the Injection
                                                              lines of student_fa_agtr are never used)
  wwpro   student_fa_wwpro         student_fa_wwpro_t2f
  ce      student_fa_ce            student_fa_ce_t2f
--trees-new t2 takes the new sources' index-pointer trees (_t2: "Step k [Agent]" options) instead of the index-free
_t2f; the default t2f keeps one pointer render (index-free, as _t3) across every pool of a fold.

Per fold and variant: rows of the fold's pools (plan.json "train_pools"), train and val separately, minus the uids in
the pool's plan.json "dropped_overlap_with_held_out" (train + val), in a round-robin interleave over the pools (pool
order of plan.json, each pool in its file order), so that the arm script's 16-row smoke head holds rows of every pool.
The tree file is built from the SAME uid list in the SAME order; every tree row must equal its A2 row except the node
fields (subqs, has_tree, tree_source, fa.tree_teacher), and every multiple-choice node must pass the trainer's MC rule
(treeplan.mc_check: one-line question, 2-62 distinct one-line options, int answer index).

Capped variant (--cap, default "largest"; folds whose pools include StepFinder): StepFinder (sf_alg + sf_hc, one
framework) is subsampled to at most the largest other pool's train count in the fold ("largest"), or to a share F of
the fold's train rows ("frac:F", e.g. frac:0.3); split over alg / hc in proportion to their sizes; drawn by question
group (the Task line), round-robin over the questions in a seeded order, so every question keeps ~the same number of
trajectories (StepFinder has ~18 regenerated trajectories per question); val is capped by the same kept fraction, so
val-based epoch selection sees the training mixture. The whowhen fold has no StepFinder pool: no capped variant.

Outputs (dir basenames are unique: train_arm.sbatch names run dirs by basename(DATA)):
  data/student_fa_lofo_<fold>_a2/{train,val}.jsonl, data/student_fa_lofo_<fold>_tf/{train,val}.jsonl
  data/student_fa_lofo_<fold>_cap_a2/..., data/student_fa_lofo_<fold>_cap_tf/...
  each with summary.json and md5sums.txt; data/student_fa_lofo/build_manifest.json (every file's md5, counts, the
  slice-hour estimates) and data/student_fa_lofo/md5_folds.txt (md5sum -c format, paths relative to --root).
Slice-hours: plan.json time_model (seconds per epoch = a N + b sum(L) + c sum(L^2), L = A2 tokens of an item, fitted on
the 0.8B fasf runs) with L from the pools' exact token totals (summary.json) spread over items by prompt length (pools
without a token total: the mean chars-per-token of the others); tree / A2 time per epoch = TOKEN_TIME x (tree tokens /
A2 tokens), TOKEN_TIME calibrated on the AFTraj _t2 pair (measured time ratio 4.468 / token ratio 5.25); evaluation
EVAL_S per --eval-at point (the largest measured point, +10% for the larger LOFO val) x 6 points; + overhead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = "data/student_fa_lofo/plan.json"
POOLS = OrderedDict([   # plan name -> (A2 dir, tree dir for t2f, tree dir for t2)
    ("sf_alg", ("data/student_fa_sf_alg", "data/student_fa_sf_alg_t3", "data/student_fa_sf_alg_t3")),
    ("sf_hc", ("data/student_fa_sf_hc", "data/student_fa_sf_hc_t3", "data/student_fa_sf_hc_t3")),
    ("aftraj", ("data/student_fa_aftraj", "data/student_fa_aftraj_t3", "data/student_fa_aftraj_t3")),
    ("real", ("data/student_fa_real", "data/student_fa_real_t3", "data/student_fa_real_t3")),
    ("agtr", ("data/student_fa_agtr_diag", "data/student_fa_agtr_diag_t2f", "data/student_fa_agtr_diag_t2")),
    ("wwpro", ("data/student_fa_wwpro", "data/student_fa_wwpro_t2f", "data/student_fa_wwpro_t2")),
    ("ce", ("data/student_fa_ce", "data/student_fa_ce_t2f", "data/student_fa_ce_t2")),
])
SF = ("sf_alg", "sf_hc")
NODE_FIELDS = ("subqs", "has_tree", "tree_source")
MC_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"    # 62, the trainer's option alphabet
TOKEN_TIME = 4.468 / 5.25          # tree / A2 time per epoch over tree / A2 tokens per epoch (AFTraj _t2 pair)
EVAL_S = 416.5 * 1.10              # seconds per --eval-at point (plan.json eval_s_per_point_max, +10% for val)
N_POINTS = 6                       # 5 --eval-at points + the final evaluation
OVERHEAD_H = 0.25                  # model load, smoke stage, final default evals (kk / jl / jevbench / td / bbeh)


# ------------------------------------------------------------------------------------------------- reading
def read_lines(path: Path) -> list:
    """(raw line, parsed row) for every non-empty line; raw lines keep their bytes (newline added on write)."""
    out = []
    with open(path, "rb") as f:
        for raw in f:
            s = raw.rstrip(b"\n")
            if s.strip():
                out.append((s, json.loads(s.decode("utf-8"))))
    return out


def uid(row: dict) -> str:
    return row["fa"]["uid"]


def mc_check(sq: dict) -> None:
    """The trainer's rule for a multiple-choice node (treeplan.mc_check at 915dbd6 and HEAD)."""
    o, a = sq.get("options"), sq.get("answer")
    ok = (isinstance(o, list) and 2 <= len(o) <= len(MC_SYMBOLS)
          and all(isinstance(t, str) and t.strip() and "\n" not in t for t in o)
          and len(set(o)) == len(o) and isinstance(a, int) and not isinstance(a, bool) and 0 <= a < len(o))
    if not ok or not str(sq.get("question") or "").strip() or "\n" in sq["question"]:
        raise ValueError("MC node %r: needs a one-line question, 2-62 distinct one-line options and an answer index"
                         % sq.get("id"))


def strip_nodes(row: dict) -> dict:
    r = {k: v for k, v in row.items() if k not in NODE_FIELDS}
    r["fa"] = {k: v for k, v in row["fa"].items() if k != "tree_teacher"}
    return r


def question_key(row: dict) -> str:
    """StepFinder question group: the Task line of the prompt, whitespace-normalised, lower case."""
    m = re.search(r"^Task: (.*)$", row["prompt"], flags=re.M)
    t = m.group(1) if m else row["prompt"][:400]
    return " ".join(t.lower().split())


class Pool:
    """One pool's aligned A2 / tree rows for train and val."""

    def __init__(self, root: Path, name: str, trees_new: str):
        a2_dir, t2f_dir, t2_dir = POOLS[name]
        self.name, self.a2_dir = name, a2_dir
        self.tf_dir = t2f_dir if trees_new == "t2f" else t2_dir
        self.rows = {}
        for split in ("train", "val"):
            a = read_lines(root / a2_dir / ("%s.jsonl" % split))
            t = read_lines(root / self.tf_dir / ("%s.jsonl" % split))
            if len(a) != len(t):
                raise SystemExit("%s %s: %d A2 rows vs %d tree rows" % (name, split, len(a), len(t)))
            for (ra, pa), (rt, pt) in zip(a, t):
                if uid(pa) != uid(pt) or pa["item_id"] != pt["item_id"]:
                    raise SystemExit("%s %s: rows out of order (%s vs %s)" % (name, split, uid(pa), uid(pt)))
                if strip_nodes(pa) != strip_nodes(pt):
                    raise SystemExit("%s %s %s: the tree row differs from its A2 row outside the node fields"
                                     % (name, split, uid(pa)))
                for sq in pt.get("subqs") or []:
                    if "options" in sq:
                        mc_check(sq)
            self.rows[split] = [(ra, pa, rt, pt) for (ra, pa), (rt, pt) in zip(a, t)]
        self.summaries = {}
        for d in (a2_dir, self.tf_dir):
            p = root / d / "summary.json"
            self.summaries[d] = json.loads(p.read_text()) if p.exists() else {}


# ------------------------------------------------------------------------------------------------- tokens / hours
def find_key(o, key):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key:
                yield v
            for x in find_key(v, key):
                yield x


def pool_tokens(pool: Pool) -> dict:
    """Exact train tokens per epoch (A2, tree) from the pools' summaries, None when not recorded."""
    s = pool.summaries.get(pool.tf_dir) or {}
    tr = (s.get("trees") or {}).get("tokens_train") or {}
    a2, gf = tr.get("tokens_per_epoch_a2"), tr.get("tokens_per_epoch_gfps")
    if a2 is None:
        a2 = next(iter(find_key(pool.summaries.get(pool.a2_dir) or {}, "tokens_per_epoch_a2")), None)
    n = len(pool.rows["train"])
    if a2 is not None and tr.get("n") not in (None, n):
        a2 = gf = None
    return {"a2": a2, "tree": gf}


def node_count(row: dict) -> int:
    return sum(1 for sq in row.get("subqs") or [] if "options" in sq)


def hours(plan: dict, rows: list, chars_per_tok: dict, tree_ratio: dict) -> dict:
    """Slice-hour estimates of the A2 (6 epochs) and tree (3 epochs) jobs on the train rows [(pool, A2 row, tree row)]."""
    c = plan["time_model"]["coef_s"]
    n, s1, s2, tok_a2, tok_tree = 0, 0.0, 0.0, 0.0, 0.0
    for p, pa, pt in rows:
        L = len(pa["prompt"]) / chars_per_tok[p]
        n += 1
        s1 += L
        s2 += L * L
        tok_a2 += L
        tok_tree += L * tree_ratio[p](pt)
    sec_a2 = c["per_item"] * n + c["per_1k_tokens"] * s1 / 1e3 + c["per_1M_token2"] * s2 / 1e6
    ratio = TOKEN_TIME * tok_tree / tok_a2 if tok_a2 else 0.0
    ev = N_POINTS * EVAL_S
    return {"train_items": n, "a2_tokens_per_epoch_est": int(tok_a2), "tree_tokens_per_epoch_est": int(tok_tree),
            "a2_s_per_epoch": round(sec_a2, 1), "tree_over_a2_time": round(ratio, 3),
            "slice_h_a2_6ep": round((6 * sec_a2 + ev) / 3600 + OVERHEAD_H, 2),
            "slice_h_tree_3ep": round((3 * sec_a2 * ratio + ev) / 3600 + OVERHEAD_H, 2)}


# ------------------------------------------------------------------------------------------------- folds
def interleave(lists: list) -> list:
    out, i = [], 0
    while any(i < len(x) for x in lists):
        for x in lists:
            if i < len(x):
                out.append(x[i])
        i += 1
    return out


def cap_rows(rows: list, quota: int, seed: str) -> list:
    """quota rows of one StepFinder subset, round-robin over its question groups (seeded order), file order kept."""
    if quota >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    groups = OrderedDict()
    for k, r in enumerate(rows):
        groups.setdefault(question_key(r[1]), []).append(k)
    keys = list(groups)
    rng.shuffle(keys)
    order = {key: rng.sample(groups[key], len(groups[key])) for key in keys}
    pick, rnd = [], 0
    while len(pick) < quota:
        for key in keys:
            if rnd < len(order[key]) and len(pick) < quota:
                pick.append(order[key][rnd])
        rnd += 1
    keep = set(pick)
    return [r for k, r in enumerate(rows) if k in keep]


def parse_cap(spec: str):
    if spec in ("none", "largest"):
        return spec, None
    m = re.fullmatch(r"frac:([0-9.]+)", spec)
    if not m or not 0 < float(m.group(1)) < 1:
        raise SystemExit("--cap: none, largest or frac:F with 0 < F < 1")
    return "frac", float(m.group(1))


def build_fold(fold: str, ent: dict, pools: dict, cap: str, seed: int, plan: dict, cpt: dict, tr: dict) -> dict:
    """{"files": {relpath: bytes}, "summary": {...}} of one fold variant (cap "none" = main)."""
    names = [p for p in ent["train_pools"]]
    kept = {}
    for p in names:
        drop = set(ent["pools"][p]["dropped_overlap_with_held_out"]) if p in ent["pools"] else set()
        kept[p] = {s: [r for r in pools[p].rows[s] if uid(r[1]) not in drop] for s in ("train", "val")}
        if p != "agtr":     # agtr: plan.json counted all 414 AgenTracer rows; the Diagnosis subset is smaller
            want = (ent["pools"][p]["train"], ent["pools"][p]["val"])
            got = (len(kept[p]["train"]), len(kept[p]["val"]))
            if want != got:
                raise SystemExit("%s / %s: %s rows after the plan's drops, plan.json says %s" % (fold, p, got, want))
    mode, frac = parse_cap(cap)
    capinfo = None
    if mode != "none":
        sf = [p for p in names if p in SF]
        if not sf:
            return {}
        other = [p for p in names if p not in SF]
        n_sf = sum(len(kept[p]["train"]) for p in sf)
        n_other = sum(len(kept[p]["train"]) for p in other)
        target = max(len(kept[p]["train"]) for p in other) if mode == "largest" else int(round(frac / (1 - frac) * n_other))
        target = min(target, n_sf)
        capinfo = {"rule": cap, "sf_train_before": n_sf, "sf_train_target": target, "subsets": {}}
        left = target
        for i, p in enumerate(sf):
            nt, nv = len(kept[p]["train"]), len(kept[p]["val"])
            qt = left if i == len(sf) - 1 else int(round(target * nt / n_sf))     # quotas sum to the target
            left -= qt
            qv = int(round(nv * qt / nt)) if nt else 0
            kept[p] = {"train": cap_rows(kept[p]["train"], qt, "lofo-cap-%s-%s-%d-train" % (fold, p, seed)),
                       "val": cap_rows(kept[p]["val"], qv, "lofo-cap-%s-%s-%d-val" % (fold, p, seed))}
            capinfo["subsets"][p] = {"train": [nt, len(kept[p]["train"])], "val": [nv, len(kept[p]["val"])],
                                     "questions_train": len({question_key(r[1]) for r in kept[p]["train"]})}
    tag = "%s_cap" % fold if mode != "none" else fold
    files, summ = {}, {"fold": fold, "variant": "cap" if mode != "none" else "main", "pools": OrderedDict(),
                       "held_out_tests": ent["held_out_tests"], "cap": capinfo}
    for s in ("train", "val"):
        rows = interleave([[(p,) + r for r in kept[p][s]] for p in names])
        ids = [r[2]["item_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise SystemExit("%s %s: item_id not unique across pools" % (tag, s))
        u = [uid(r[2]) for r in rows]
        if len(set(u)) != len(u):
            raise SystemExit("%s %s: uid not unique across pools" % (tag, s))
        files["data/student_fa_lofo_%s_a2/%s.jsonl" % (tag, s)] = b"".join(r[1] + b"\n" for r in rows)
        files["data/student_fa_lofo_%s_tf/%s.jsonl" % (tag, s)] = b"".join(r[3] + b"\n" for r in rows)
        if s == "train":
            summ["hours"] = hours(plan, [(r[0], r[2], r[4]) for r in rows], cpt, tr)
            types = Counter(sq.get("type") for r in rows for sq in r[4].get("subqs") or [] if "options" in sq)
            summ["tree_node_types"] = dict(sorted(types.items()))
            summ["tree_rows_with_nodes"] = sum(1 for r in rows if node_count(r[4]))
            summ["head16_pools"] = dict(Counter(r[0] for r in rows[:16]))
    for p in names:
        summ["pools"][p] = {"a2_dir": pools[p].a2_dir, "tree_dir": pools[p].tf_dir,
                            "train": len(kept[p]["train"]), "val": len(kept[p]["val"]),
                            "dropped": len(ent["pools"].get(p, {}).get("dropped_overlap_with_held_out", []))}
    n_tr = sum(v["train"] for v in summ["pools"].values())
    summ["train"], summ["val"] = n_tr, sum(v["val"] for v in summ["pools"].values())
    summ["sf_share_train"] = round(sum(summ["pools"][p]["train"] for p in names if p in SF) / n_tr, 3)
    return {"files": files, "summary": summ, "tag": tag}


def build_all(root: Path, plan_path: str, trees_new: str, cap: str, seed: int, folds: list, variants: list) -> dict:
    plan = json.loads((root / plan_path).read_text())
    use = [f for f in plan["folds"] if not folds or f in folds]
    names = sorted({p for f in use for p in plan["folds"][f]["train_pools"]}, key=list(POOLS).index)
    pools = {p: Pool(root, p, trees_new) for p in names}
    # chars per token per pool (exact token totals where recorded) and the tree / A2 token factor per row
    cpt, ks, toks = {}, {}, {}
    for p, pool in pools.items():
        toks[p] = pool_tokens(pool)
        if toks[p]["a2"]:
            cpt[p] = sum(len(r[1]["prompt"]) for r in pool.rows["train"]) / toks[p]["a2"]
    mean_cpt = sum(cpt.values()) / len(cpt) if cpt else 3.5      # no recorded totals: ~3.5 characters per token
    for p, pool in pools.items():
        cpt.setdefault(p, mean_cpt)
        t = toks[p]
        if t["a2"] and t["tree"]:       # tree tokens = sum L (1 + k n): k = node view / final view tokens
            ln = sum(len(r[1]["prompt"]) / cpt[p] * node_count(r[3]) for r in pool.rows["train"])
            ks[p] = (t["tree"] - t["a2"]) / ln if ln else 0.0
    mean_k = sum(ks.values()) / len(ks) if ks else 1.0
    tr = {p: (lambda row, k=ks.get(p, mean_k): 1 + k * node_count(row)) for p in pools}
    out = {"args": {"plan": plan_path, "trees_new": trees_new, "cap": cap, "seed": seed}, "folds": OrderedDict(),
           "files": OrderedDict()}
    for f in use:
        for v in variants:
            res = build_fold(f, plan["folds"][f], pools, "none" if v == "main" else cap, seed, plan, cpt, tr)
            if not res:
                continue
            for path, data in res["files"].items():
                out["files"][path] = {"md5": hashlib.md5(data).hexdigest(), "lines": data.count(b"\n"), "data": data}
            out["folds"][res["tag"]] = res["summary"]
    return out


def write(root: Path, out: dict) -> None:
    dirs = OrderedDict()
    for path, ent in out["files"].items():
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(ent["data"])
        dirs.setdefault(p.parent, []).append((p.name, ent["md5"]))
    for d, fs in dirs.items():
        (d / "md5sums.txt").write_text("".join("%s  %s\n" % (m, n) for n, m in fs))
    for tag, s in out["folds"].items():
        for kind in ("a2", "tf"):
            d = root / ("data/student_fa_lofo_%s_%s" % (tag, kind))
            (d / "summary.json").write_text(json.dumps(dict(s, kind=kind, args=out["args"]), indent=1) + "\n")
    man = {"args": out["args"], "folds": out["folds"],
           "files": {k: {"md5": v["md5"], "lines": v["lines"]} for k, v in out["files"].items()}}
    m = root / "data/student_fa_lofo/build_manifest.json"
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text(json.dumps(man, indent=1) + "\n")
    (root / "data/student_fa_lofo/md5_folds.txt").write_text(
        "".join("%s  %s\n" % (v["md5"], k) for k, v in out["files"].items()))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(ROOT), help="directory holding data/ (the repo, or ~/cotd on the cluster)")
    ap.add_argument("--plan", default=PLAN)
    ap.add_argument("--trees-new", default="t2f", choices=["t2f", "t2"],
                    help="tree render of the new sources agtr / wwpro / ce (t2f: index-free, as the _t3 pools)")
    ap.add_argument("--cap", default="largest", help="capped variant: largest | frac:F | none")
    ap.add_argument("--variants", default="main,cap")
    ap.add_argument("--folds", default="", help="comma list (default: every fold of plan.json)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", default="", help="a build_manifest.json: rebuild in memory, compare md5s, write nothing")
    args = ap.parse_args(argv)
    root = Path(args.root).expanduser()
    variants = [v for v in args.variants.split(",") if v]
    if not set(variants) <= {"main", "cap"}:
        raise SystemExit("--variants: main, cap")
    out = build_all(root, args.plan, args.trees_new, args.cap, args.seed,
                    [f for f in args.folds.split(",") if f], variants)
    if args.check:
        man = json.loads(Path(args.check).expanduser().read_text())
        bad = [k for k, v in man["files"].items() if k in out["files"] and out["files"][k]["md5"] != v["md5"]]
        miss = [k for k in man["files"] if k not in out["files"]]
        print("check: %d files, %d differ, %d not rebuilt" % (len(man["files"]), len(bad), len(miss)))
        for k in bad + miss:
            print("  DIFFERS " + k if k in bad else "  MISSING " + k)
        if bad or miss:
            sys.exit(1)
        return out
    write(root, out)
    tot = {"a2": 0.0, "tree": 0.0}
    for tag, s in out["folds"].items():
        h = s["hours"]
        tot["a2"] += h["slice_h_a2_6ep"]
        tot["tree"] += h["slice_h_tree_3ep"]
        print("%-18s train %5d val %4d  SF share %.2f  A2 %5.2f h  tree %5.2f h  (tree/A2 time %.2f)  head16 %s"
              % (tag, s["train"], s["val"], s["sf_share_train"], h["slice_h_a2_6ep"], h["slice_h_tree_3ep"],
                 h["tree_over_a2_time"], s["head16_pools"]))
    print("total A2 %.1f h + tree %.1f h = %.1f slice-h over %d fold variants"
          % (tot["a2"], tot["tree"], tot["a2"] + tot["tree"], len(out["folds"])))
    return out


if __name__ == "__main__":
    main()
