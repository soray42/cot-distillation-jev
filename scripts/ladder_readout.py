"""Ladder readout: one command for single- (or multi-) seed arm runs on the Tree Forcing domains. CPU only, numpy.

Run dirs are <ARM>[-<tag>]-<model>-<data>-s<seed> (hpc/train_arm.sbatch OUT), any model name; runs are grouped by
(arm, tag, data) (and by model too when several models are passed). Domain from the data part of the name: "numgen" in
it -> numeric chains (student_numgen_tf), "pronto" -> PrOntoQA-OOD (student_pronto_tf), else K&K (student_v3sl).
Arguments are run dirs, globs, or a parent dir whose children are run dirs (e.g. results/hpc/<pull>/). Dirs without
metrics.json (still running) are listed as skipped.

K&K, per run: kk_deep accuracy by people bins 4-10 / 11-16 / 17-20 / 11-20 and per people count (group kk/dN or kkx/dN),
  dA = analyze_e2.run_summary(run)["A"] (needs preds_diag), bbh (macro over tasks), jevbench, folio, val, training hours
  (history steps x s/step, i.e. the whole run incl. mid evals; "t" = fallback to the last trajectory s_elapsed), click
  = the first trajectory point (or the end eval) with diagA >= .40, as epoch @step (tf_readout.curve).
Numeric chains, per run: accuracy on ng_d1..ng_d8 and ng_ood (eval keys ng_d1 / ng1, ng_ood / ngood), mean over d1-d4
  (in range) and d5-d8 (beyond the training depth), and drop = beyond - in range.
PrOntoQA-OOD, per run: as numeric chains with proof depth = hops (eval keys pq_d1..pq_d8, trained on hops 1-4) and
  pq_wide (hops 1-4 with two extra chain pairs: in-range depth, out-of-range distractor width) in place of ood.
All: problem-view node accuracy by node depth (preds_subq, kind "cot", source knights_knaves / numgen / pronto,
  `correct`) with the depth joined by (item_id, question) from data/<data>/val.jsonl subqs' node_depth (also their
  "opposite" wording) and, for what that misses, preds_local_kk.jsonl of any run of the same domain (the numeric local
  probe runs on local_deep.jsonl, whose items are not the val items, so it alone would join nothing); bins 0 / 1-3 /
  4-7 / 8-12 / >=13, join coverage printed. Local-step probe (eval key local_kk or local_kk_anc, every domain): all,
  derived_nontrivial, by depth bin.
Contrasts: every arm vs A2 (c12 preferred) and vs GFPS (c4 preferred) of the same domain, on kk 11-20 and dA (K&K) and
  beyond d5-d8 (numeric; pronto: beyond hops 5-8 and wide): mean difference and paired item-level bootstrap 95% CI
  (items / diag programs joined by item_id; tf_readout.contrast: paired by seed label, pooled over common seeds; no
  common seed -> difference of arm means, still item-paired, labelled).
LR sweep (appended to every domain section; the tables above are unchanged). The LR is part of the arm name (tag suffix
  lrN = N e-6, e.g. A2-c12lr3 = A2-c12 at 3e-6; no suffix = the default 1e-5) and is read from metrics.json args.lr
  (tag only as a fallback; a disagreement is a WARNING). Method = arm + tag without the LR suffix (A2-c12, GFPS-c4,
  G3-c4kk). Full curve: every method x LR, per seed and mean, validation key + the domain's test metrics (K&K: kk 4-10,
  11-20, dA, bbh, jevbench, td, gsm8k; numeric: in d1-4, beyond d5-8, ood; pronto: in hops 1-4, beyond 5-8, wide; both
  + jevbench, td). Selection, by the validation key only (never a test set), fixed in
  notes/next_round/tree_forcing/lr_check.md before the sweep results: K&K valkk, numeric / pronto val, accuracy of the
  end-of-training evaluation (the last trajectory point if the run has none), averaged over the --sweep-seeds (default
  s12) run at every LR of the method, so seeds added later at the selected LR never re-select (no sweep seed at every
  LR -> the seeds common to every LR, then each LR's own seeds, flagged); ties (1e-9) -> lower NLL of the same key,
  then the LR nearest 1e-5. Numeric / pronto reuse the method's K&K-selected LR (same model) when it was run there,
  unless their own val prefers another LR strongly: own best - K&K LR >= .03 on val with a paired item-bootstrap 95% CI
  above 0 (sweep seeds). Best-vs-best: --best-arm (GFPS, c4 preferred) - --best-ref (A2, c12 preferred) at their
  selected LRs on every test metric, tf.contrast (paired item bootstrap, pooled over common seeds; bbh as the macro
  over tasks: item values weighted N / (tasks x task size), so the item mean is the macro mean), next to the same
  contrast at 1e-5 for both on the same seeds (only the LR differs). Forgetting curve (only when a run's trajectory has
  a general-ability EVALMID key: bbh500, bbh, jevbench, td, gsm8k): accuracy at the first, 25/50/75% and last trajectory
  points and the minimum, with epochs.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python scripts/ladder_readout.py 'results/hpc/l4b/*' --out results/ladder_readout/l4b
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tf_readout as tf  # noqa: E402
from ablation_table import train_hours  # noqa: E402
from analyze_e2 import correct, diag_table, primary, read_preds, run_summary  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARM_ORDER = ["A2", "G3", "GFPS", "TFL", "GTF", "TFLP", "MIX"]
REFS = [("A2", "c12"), ("GFPS", "c4")]                  # comparator arm, preferred tag
DOMAINS = {"kk": "K&K", "numgen": "Numeric chains", "pronto": "PrOntoQA-OOD"}
NODE_SOURCE = {"kk": "knights_knaves", "numgen": "numgen", "pronto": "pronto"}
ENDPOINTS = {"kk": [("kk11-20", "kk 11-20"), ("dA", "dA")], "numgen": [("ng_beyond", "beyond d5-d8")],
             "pronto": [("pq_beyond", "beyond hops 5-8"), ("pq_wide", "wide")]}
KK_BINS = ["4-10", "11-16", "17-20", "11-20"]
NODE_BINS = tf.LOCAL_BINS                               # 0 / 1-3 / 4-7 / 8-12 / >=13
NG_DEPTH = re.compile(r"^ng_?d?(\d+)$")
NG_OOD = re.compile(r"^ng_?ood$")
# chain domains: depth sets <p>_d1..d8 (trained on d1-d4) and one extra set (numeric: ood; pronto: wide)
CHAIN = {"numgen": {"p": "ng", "extra": "ood", "depth": NG_DEPTH, "extra_re": NG_OOD,
                    "title": "### Per run (accuracy per chain depth; trained on d1-d4; in = mean d1-d4, beyond = mean "
                             "d5-d8, drop = beyond - in; train h from history, t = last trajectory point)",
                    "in": "in d1-4", "beyond": "beyond d5-8"},
         "pronto": {"p": "pq", "extra": "wide", "depth": re.compile(r"^pq_?d?(\d+)$"),
                    "extra_re": re.compile(r"^pq_?wide$"),
                    "title": "### Per run (accuracy per proof depth = hops; trained on hops 1-4; in = mean d1-d4, "
                             "beyond = mean d5-d8, drop = beyond - in; wide = pronto_wide, hops 1-4 with two extra "
                             "chain pairs; train h from history, t = last trajectory point)",
                    "in": "in hops 1-4", "beyond": "beyond hops 5-8"}}
TAIL = re.compile(r"^(?P<head>.+)-(?P<data>student_[A-Za-z0-9_]+)-s(?P<seed>\d+)$")
TAG_TOKEN = re.compile(r"^[a-z][a-z0-9.]*$")
# ---- LR sweep (see the docstring and notes/next_round/tree_forcing/lr_check.md; fixed before the sweep results)
LR_SUFFIX = re.compile(r"lr(\d+)(?=-|$)")               # c12lr3 -> c12 at 3e-6
LR_DEFAULT = 1e-5                                       # hpc/train_arm.sbatch LR default (no suffix)
SEL_KEY = {"kk": "valkk", "numgen": "val", "pronto": "val"}
LR_TESTS = {"kk": [("kk4-10", "kk 4-10"), ("kk11-20", "kk 11-20"), ("dA", "dA"), ("bbh", "bbh"),
                   ("jevbench", "jevbench"), ("td", "td"), ("gsm8k", "gsm8k")],
            "numgen": [("ng_in", "in d1-4"), ("ng_beyond", "beyond d5-8"), ("ng_ood", "ood"),
                       ("jevbench", "jevbench"), ("td", "td")],
            "pronto": [("pq_in", "in hops 1-4"), ("pq_beyond", "beyond hops 5-8"), ("pq_wide", "wide"),
                       ("jevbench", "jevbench"), ("td", "td")]}
STRONG = 0.03                                           # numeric / pronto overrule the K&K LR: >= .03 and CI > 0
TIE = 1e-9
FORGET = ("bbh500", "bbh", "jevbench", "td", "gsm8k")  # general-ability EVALMID keys (forgetting curve)
SWEEP_SEEDS = (12,)                                     # the one-seed LR sweep (--sweep-seeds)


# ---------------------------------------------------------------- names, dirs
def parse_run_name(name: str, model_hint: str | None = None) -> dict | None:
    """-> {arm, tag, model, data, seed}. The model is metrics.json args.model's basename when the name contains it (any
    model), else tf_readout's Qwen pattern, else the tokens after arm + lowercase tag tokens (c12, c4kk, sf1, nw0.5)."""
    m = TAIL.match(name)
    if not m:
        return None
    head = m["head"]
    if model_hint and head.endswith("-" + model_hint) and len(head) > len(model_hint) + 1:
        left, model = head[: -len(model_hint) - 1], model_hint
    elif (info := tf.parse_name(name)) is not None:
        return {k: info[k] for k in ("arm", "tag", "model", "data", "seed")}
    else:
        toks = head.split("-")
        i = 1
        while i < len(toks) - 1 and TAG_TOKEN.match(toks[i]):
            i += 1
        left, model = "-".join(toks[:i]), "-".join(toks[i:])
    arm, _, tag = left.partition("-")
    return {"arm": arm, "tag": tag, "model": model, "data": m["data"], "seed": int(m["seed"])}


def expand(patterns: list[str]) -> list[Path]:
    """Globs; a dir given without a wildcard that is not a run (no metrics.json, no -s<seed> name) expands to its
    child dirs."""
    out = set()
    for pat in patterns:
        wild = glob.has_magic(pat)
        for p in map(Path, glob.glob(pat) or [pat]):
            if not wild and p.is_dir() and not (p / "metrics.json").exists() and not TAIL.match(p.name):
                out.update(c for c in p.iterdir() if c.is_dir())
            else:
                out.add(p)
    return sorted(out)


def domain_of(data: str) -> str:
    return "numgen" if "numgen" in data else "pronto" if "pronto" in data else "kk"


def flr(lr: float | None) -> str:
    """3e-06 -> 3e-6."""
    return "?" if lr is None else re.sub(r"e([+-])0*(\d)", r"e\1\2", f"{lr:g}")


def split_lr(tag: str) -> tuple[str, float | None]:
    """Tag -> (tag without the LR suffix, LR from the suffix: lrN = N e-6; None without one)."""
    m = LR_SUFFIX.search(tag)
    if not m:
        return tag, None
    base = (tag[:m.start()] + tag[m.end():]).replace("--", "-").strip("-")
    return base, float(f"{int(m[1]) * 1e-6:.6g}")


# ---------------------------------------------------------------- per run
def set_acc(d: Path, ev: dict, name: str) -> float | None:
    """preds_<name>.jsonl (bbh: macro over tasks, as analyze_e2.run_summary), else metrics eval acc."""
    f = d / f"preds_{name}.jsonl"
    if f.exists():
        ps = [p for p in read_preds(f) if p.get("gold_label") in p["labels"]]
        if not ps:
            return None
        if name == "bbh":
            by = defaultdict(list)
            for p in ps:
                g = p.get("group") or ""
                by[g.split("/")[0] + "/" + g.split("/")[-1]].append(correct(p))
            return sum(sum(v) / len(v) for v in by.values()) / len(by)
        return sum(correct(p) for p in ps) / len(ps)
    x = ev.get(name)
    return x.get("acc") if isinstance(x, dict) else None


def hours(metrics: dict, traj: list) -> tuple[float | None, str]:
    h = train_hours({"metrics": metrics})
    if h is not None:
        return h, "history"
    el = [x["s_elapsed"] for x in traj or [] if x.get("s_elapsed") is not None]
    return (max(el) / 3600, "trajectory") if el else (None, "")


def ng_sets(d: Path, ev: dict, spec: dict = CHAIN["numgen"]) -> dict:
    """{depth (int) or the extra set ("ood" / "wide"): eval name} from metrics eval keys and preds files."""
    names = set(ev) | {f.name[len("preds_"):-len(".jsonl")] for f in d.glob(f"preds_{spec['p']}*.jsonl")}
    out = {}
    for n in sorted(names):
        m = spec["depth"].match(n)
        if m:
            out.setdefault(int(m[1]), n)
        elif spec["extra_re"].match(n):
            out.setdefault(spec["extra"], n)
    return out


def final_metric(d: Path, ev: dict, traj: list, name: str) -> tuple[float | None, float | None]:
    """(acc, nll) of eval set `name`: preds_<name>.jsonl accuracy (set_acc) with the end-of-training metrics NLL, else
    the last trajectory point that has it."""
    acc, x = set_acc(d, ev, name), ev.get(name)
    if acc is not None:
        return acc, x.get("nll") if isinstance(x, dict) else None
    pts = [p[name] for p in traj or [] if isinstance(p.get(name), dict)]
    return (pts[-1].get("acc"), pts[-1].get("nll")) if pts else (None, None)


def item_units(d: Path, name: str, macro: bool = False) -> dict[str, float]:
    """item_id -> correct from preds_<name>.jsonl. macro (bbh): weighted N / (tasks x task size), so the plain item
    mean equals the macro mean over tasks (tasks as set_acc)."""
    f = d / f"preds_{name}.jsonl"
    if not f.exists():
        return {}
    ps = [p for p in read_preds(f) if p.get("gold_label") in p["labels"]]
    u = {p["item_id"]: float(correct(p)) for p in ps}
    if macro and ps:
        task = {p["item_id"]: (p.get("group") or "").split("/")[0] + "/" + (p.get("group") or "").split("/")[-1]
                for p in ps}
        size = defaultdict(int)
        for t in task.values():
            size[t] += 1
        u = {i: c * len(u) / (len(size) * size[task[i]]) for i, c in u.items()}
    return u


def run_lr(metrics: dict, tag: str) -> tuple[float | None, str, str | None]:
    """(LR, source, warning): metrics.json args.lr, else the tag suffix, else the default."""
    _, from_tag = split_lr(tag)
    a = (metrics.get("args") or {}).get("lr")
    if a is not None:
        a = float(f"{float(a):.6g}")
        bad = from_tag is not None and abs(a - from_tag) > 1e-3 * a
        return a, "args", (f"tag LR {flr(from_tag)} but args.lr {flr(a)}" if bad else None)
    return (from_tag, "tag", None) if from_tag is not None else (LR_DEFAULT, "default", None)


def node_preds(d: Path, source: str) -> list[tuple[str, str, int]]:
    f = d / "preds_subq.jsonl"
    if not f.exists():
        return []
    return [(p["item_id"], p["question"], int(p["correct"])) for p in read_preds(f)
            if p.get("kind") == "cot" and p.get("source") == source and p.get("correct") is not None]


def load_run(d: Path, info: dict) -> dict:
    metrics = json.loads((d / "metrics.json").read_text())
    traj = json.loads((d / "trajectory.json").read_text()) if (d / "trajectory.json").exists() else None
    if traj and not metrics.get("trajectory"):
        metrics["trajectory"] = traj
    traj = metrics.get("trajectory") or traj or []
    ev = metrics.get("eval") or {}
    dom = domain_of(info["data"])
    r = dict(info, domain=dom, dir=str(d), name=d.name)
    for k in ("val", "jevbench"):
        r[k] = set_acc(d, ev, k)
    r["hours"], r["hours_src"] = hours(metrics, traj)
    lkey, loc, ln = tf.local_probe(metrics)
    r.update(local_key=lkey, local=loc, local_n=ln)
    r["_nodes"] = node_preds(d, NODE_SOURCE[dom])
    units: dict[str, dict] = {}
    # LR sweep fields: LR, the validation key at the end of training, td (all domains), gsm8k (K&K), item units
    r["lr"], r["lr_src"], r["_lr_warn"] = run_lr(metrics, info["tag"])
    sel = SEL_KEY[dom]
    r["sel_acc"], r["sel_nll"] = final_metric(d, ev, traj, sel)
    r["_traj"] = traj
    r["td"] = set_acc(d, ev, "td")
    for k in ("val", "jevbench", "td") + (("valkk", "gsm8k") if dom == "kk" else ()):
        units[k] = item_units(d, k)
    if dom == "kk":
        r["valkk"], r["gsm8k"] = set_acc(d, ev, "valkk"), set_acc(d, ev, "gsm8k")
        units["bbh"] = item_units(d, "bbh", macro=True)
        for k in ("bbh", "folio"):
            r[k] = set_acc(d, ev, k)
        r["dA"] = None
        if (d / "preds_diag.jsonl").exists():
            r["dA"] = run_summary(d)["A"]
            t = diag_table(d)
            units["dA"] = {g: primary(t, [g]) for g in t}
        kk = tf.kkdeep_units(d)
        for b in KK_BINS:
            lo, hi = tf.BINS[b]
            u = {i: float(c) for i, (n, c) in kk.items() if lo <= n <= hi}
            units["kk" + b] = u
            r["kk" + b] = sum(u.values()) / len(u) if u else None
            r[f"kk{b}_n"] = len(u)
        by_n = defaultdict(list)
        for n, c in kk.values():
            by_n[n].append(c)
        r["people_acc"] = {n: sum(v) / len(v) for n, v in sorted(by_n.items())}
        r["people_n"] = {n: len(v) for n, v in sorted(by_n.items())}
        r.update(tf.curve(metrics))
    else:                                             # chain domains (numeric: ng_*, pronto: pq_*)
        spec = CHAIN[dom]
        p, x = spec["p"], spec["extra"]
        sets = ng_sets(d, ev, spec)
        r[f"{p}_names"] = {str(k): v for k, v in sets.items()}
        r[p], per_item = {}, {}
        for k, name in sets.items():
            f = d / f"preds_{name}.jsonl"
            if f.exists():
                ps = [q for q in read_preds(f) if q.get("gold_label") in q["labels"]]
                per_item[k] = {f"{name}/{q['item_id']}": float(correct(q)) for q in ps}
                r[p][str(k)] = sum(per_item[k].values()) / len(ps) if ps else None
            else:
                r[p][str(k)] = set_acc(d, ev, name)
        r[f"{p}_{x}"] = r[p].pop(x, None)
        units[f"{p}_{x}"] = per_item.get(x, {})
        for lab, ks in ((f"{p}_in", (1, 2, 3, 4)), (f"{p}_beyond", (5, 6, 7, 8))):
            xs = [r[p][str(k)] for k in ks if r[p].get(str(k)) is not None]
            r[lab] = sum(xs) / len(xs) if xs else None
            r[lab + "_sets"] = len(xs)
            u = {}
            for k in ks:
                u.update(per_item.get(k, {}))
            units[lab] = u
        r[f"{p}_drop"] = (r[f"{p}_beyond"] - r[f"{p}_in"] if r[f"{p}_in"] is not None and r[f"{p}_beyond"] is not None
                          else None)
    r["_units"] = units
    return r


# ---------------------------------------------------------------- node depths
def depth_maps(data_dir: Path, datas: set[str], runs: list[dict]) -> tuple[dict, dict]:
    """(item_id, question) -> node depth: (from data/<data>/val.jsonl, from runs' preds_local_kk.jsonl)."""
    from_data, from_local = {}, {}
    for data in sorted(datas):
        f = data_dir / data / "val.jsonl"
        if not f.exists():
            continue
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                it = json.loads(line)
                for q in it.get("subqs") or []:
                    if q.get("node_depth") is None:
                        continue
                    from_data[(it["item_id"], q["question"])] = q["node_depth"]
                    if q.get("opposite"):
                        from_data.setdefault((it["item_id"], q["opposite"]), q["node_depth"])
    for r in runs:
        f = Path(r["dir"]) / "preds_local_kk.jsonl"
        if f.exists():
            for p in read_preds(f):
                if p.get("depth") is not None:
                    from_local.setdefault((p["item_id"], p["question"]), p["depth"])
    return from_data, from_local


def node_depth_acc(r: dict, from_data: dict, from_local: dict) -> None:
    joined, n_data, n_local = defaultdict(list), 0, 0
    for iid, q, c in r["_nodes"]:
        k = (iid, q)
        if k in from_data:
            dep, n_data = from_data[k], n_data + 1
        elif k in from_local:
            dep, n_local = from_local[k], n_local + 1
        else:
            continue
        for lab, lo, hi in NODE_BINS:
            if lo <= dep <= hi:
                joined[lab].append(c)
    nodes = r["_nodes"]
    r["node_all"] = sum(c for _, _, c in nodes) / len(nodes) if nodes else None
    r["node_n"] = len(nodes)
    r["node_join"] = {"data": n_data, "local": n_local, "none": len(nodes) - n_data - n_local}
    r["node_bins"] = {lab: sum(v) / len(v) for lab, v in joined.items()}
    r["node_bins_n"] = {lab: len(v) for lab, v in joined.items()}


# ---------------------------------------------------------------- grouping, contrasts
def arm_key(label_parts: tuple) -> tuple:
    arm = label_parts[0]
    return (ARM_ORDER.index(arm) if arm in ARM_ORDER else len(ARM_ORDER), arm) + label_parts[1:]


def group(runs: list[dict]) -> tuple[dict, list[str]]:
    """-> ({section: {label: {seed: run}}}, warnings). section = (domain, model or None)."""
    multi_model = len({r["model"] for r in runs}) > 1
    sections: dict = defaultdict(lambda: defaultdict(dict))
    warn = []
    datas = defaultdict(set)
    for r in runs:
        datas[(r["domain"], r["model"] if multi_model else None)].add(r["data"])
    for r in sorted(runs, key=lambda x: x["dir"]):
        sec = (r["domain"], r["model"] if multi_model else None)
        lab = r["arm"] + (f"-{r['tag']}" if r["tag"] else "")
        if len(datas[sec]) > 1:
            lab += "/" + r["data"].replace("student_", "")
        r["label"] = lab
        if r["seed"] in sections[sec][lab]:
            warn.append(f"duplicate {lab} s{r['seed']}: {r['dir']} not used ({sections[sec][lab][r['seed']]['dir']} is)")
            continue
        sections[sec][lab][r["seed"]] = r
    out = {}
    for sec in sorted(sections, key=lambda s: (list(DOMAINS).index(s[0]), s[1] or "")):
        labs = sorted(sections[sec], key=lambda lab: arm_key((next(iter(sections[sec][lab].values()))["arm"], lab)))
        out[sec] = {lab: dict(sorted(sections[sec][lab].items())) for lab in labs}
    return out, warn


def pick_ref(groups: dict, arm: str, tag: str) -> str | None:
    c = [lab for lab, g in groups.items() if next(iter(g.values()))["arm"] == arm]
    if not c:
        return None
    return max(c, key=lambda lab: (next(iter(groups[lab].values()))["tag"] == tag, len(groups[lab]), lab))


def contrasts(groups: dict, dom: str, boot: int, boot_seed: int) -> list[dict]:
    out, done = [], set()
    for ref_arm, ref_tag in REFS:
        ref = pick_ref(groups, ref_arm, ref_tag)
        for lab in groups:
            if lab == ref or (ref is not None and frozenset((lab, ref)) in done):
                continue
            row = {"arm": lab, "vs": ref or ref_arm, "ref_missing": ref is None, "endpoints": {}}
            if ref is not None:
                done.add(frozenset((lab, ref)))
                for ep, _ in ENDPOINTS[dom]:
                    c = tf.contrast(groups, lab, ref, ep, boot, boot_seed)
                    row["endpoints"][ep] = {k: v for k, v in c.items() if k != "seed_label_only"}
            out.append(row)
    return out


# ---------------------------------------------------------------- LR sweep
def lr_methods(groups: dict) -> tuple[dict, list[str]]:
    """-> ({method: {arm, base, lrs: {lr: label}}}, warnings). method = the label with the LR suffix taken off its tag
    (the /data suffix of a mixed section kept)."""
    out, warn = {}, []
    for lab, g in groups.items():
        r = next(iter(g.values()))
        core = r["arm"] + (f"-{r['tag']}" if r["tag"] else "")
        base, _ = split_lr(r["tag"])
        m = r["arm"] + (f"-{base}" if base else "") + lab[len(core):]
        warn += [f"{lab} s{s}: {x['_lr_warn']}" for s, x in g.items() if x.get("_lr_warn")]
        lrs = sorted({x["lr"] for x in g.values()})
        if len(lrs) > 1:
            warn.append(f"{lab}: seeds at different LRs ({', '.join(map(flr, lrs))}); listed at {flr(r['lr'])}")
        meta = out.setdefault(m, {"arm": r["arm"], "base": base, "lrs": {}})
        if r["lr"] in meta["lrs"]:
            warn.append(f"{m} at {flr(r['lr'])}: {lab} not used ({meta['lrs'][r['lr']]} is)")
            continue
        meta["lrs"][r["lr"]] = lab
    for meta in out.values():
        meta["lrs"] = dict(sorted(meta["lrs"].items()))
    return {m: out[m] for m in sorted(out, key=lambda m: arm_key((out[m]["arm"], m)))}, warn


def select_lr(groups: dict, methods: dict, dom: str, kk_choice: dict | None, boot: int, boot_seed: int,
              sweep: tuple = SWEEP_SEEDS) -> dict:
    """Per method: the LR chosen on the validation key (module docstring; lr_check.md). Selection seeds = the sweep
    seeds run at every LR of the method (seeds added later at the chosen LR never re-select); none -> the seeds common
    to every LR, then each LR's own seeds (flagged)."""
    key = SEL_KEY[dom]
    out = {}
    for m, meta in methods.items():
        have = {lr: {s for s, r in groups[lab].items() if r.get("sel_acc") is not None}
                for lr, lab in meta["lrs"].items()}
        cand = {lr: meta["lrs"][lr] for lr, ss in have.items() if ss}
        row = {"key": key, "chosen": None, "notes": []}
        out[m] = row
        if not cand:
            row["notes"].append(f"no run has {key}")
            continue
        every = set.intersection(*[have[lr] for lr in cand])
        common = sorted(every & set(sweep)) or sorted(every)
        seeds = {lr: common or sorted(have[lr]) for lr in cand}
        score = {lr: float(np.mean([groups[cand[lr]][s]["sel_acc"] for s in seeds[lr]])) for lr in cand}
        nlls = {lr: [groups[cand[lr]][s]["sel_nll"] for s in seeds[lr]] for lr in cand}
        nll = {lr: float(np.mean(v)) if all(x is not None for x in v) else math.inf for lr, v in nlls.items()}
        top = max(score.values())
        tied = [lr for lr in cand if score[lr] >= top - TIE]
        own = min(tied, key=lambda lr: (nll[lr], abs(math.log10(lr / LR_DEFAULT)), lr))
        row.update(sweep_seeds=common, seeds={lr: seeds[lr] for lr in cand}, score=score, nll=nll, own=own,
                   chosen=own)
        if len(cand) == 1:
            row["notes"].append("one LR only")
        elif not common:
            row["notes"].append("no seed at every LR: each LR's own seeds")
        elif not every & set(sweep):
            row["notes"].append(f"no sweep seed ({'/'.join(f's{x}' for x in sweep)}) at every LR: common seeds")
        if len(tied) > 1:
            row["notes"].append(f"{key} tie: lower NLL, then nearest 1e-5")
        if dom == "kk":
            continue
        k_lr = (kk_choice or {}).get(m)
        row["kk_lr"] = k_lr
        if k_lr is None:
            row["notes"].append("no K&K selection for this method: own val")
        elif k_lr not in cand:
            row["notes"].append(f"K&K LR {flr(k_lr)} not run here: own val")
        elif k_lr == own:
            row["notes"].append("own val agrees with K&K")
        else:
            c = tf.contrast(groups, cand[own], cand[k_lr], key, boot, boot_seed,
                            seeds_allowed=set(common) if common else None)
            strong = "mean" in c and c["mean"] >= STRONG - TIE and c["ci"][0] > 0
            row["overrule"] = {k: v for k, v in c.items() if k != "seed_label_only"}
            row["chosen"] = own if strong else k_lr
            row["notes"].append(f"own val prefers {flr(own)} "
                                + ("strongly: own val" if strong else "not strongly: K&K LR"))
    return out


def pick_method(methods: dict, groups: dict, arm: str, tag: str) -> str | None:
    c = [m for m, meta in methods.items() if meta["arm"] == arm]
    if not c:
        return None
    return max(c, key=lambda m: (methods[m]["base"] == tag,
                                 sum(len(groups[lab]) for lab in methods[m]["lrs"].values()), m))


def best_vs_best(groups: dict, methods: dict, sel: dict, dom: str, arm: tuple, ref: tuple, boot: int,
                 boot_seed: int) -> dict:
    """arm - ref at each one's selected LR (and at 1e-5 for both, on the same seeds) on every test metric of the
    domain."""
    a, b = pick_method(methods, groups, *arm), pick_method(methods, groups, *ref)
    res = {"arm": a or arm[0], "ref": b or ref[0]}
    if a is None or b is None or sel[a]["chosen"] is None or sel[b]["chosen"] is None:
        res["missing"] = [x for x, y in ((arm[0], a), (ref[0], b)) if y is None or sel[y]["chosen"] is None]
        return res
    la, lb = methods[a]["lrs"][sel[a]["chosen"]], methods[b]["lrs"][sel[b]["chosen"]]
    da, db = methods[a]["lrs"].get(LR_DEFAULT), methods[b]["lrs"].get(LR_DEFAULT)
    res.update(arm_lr=sel[a]["chosen"], ref_lr=sel[b]["chosen"], arm_label=la, ref_label=lb, endpoints={})
    for ep, _ in LR_TESTS[dom]:
        c = {k: v for k, v in tf.contrast(groups, la, lb, ep, boot, boot_seed).items() if k != "seed_label_only"}
        ss = c["seeds"] if c.get("paired") else None
        means = {lab: _mean([r for s, r in groups[lab].items() if ss is None or s in ss], ep) for lab in (la, lb)}
        e = {"best": c, "means": means}
        if da and db and (da, db) != (la, lb):          # the same seeds at 1e-5 for both: only the LR differs
            e["default"] = {k: v for k, v in tf.contrast(groups, da, db, ep, boot, boot_seed,
                                                         seeds_allowed=set(ss) if ss else None).items()
                            if k != "seed_label_only"}
        res["endpoints"][ep] = e
    return res


# ---------------------------------------------------------------- output
f3, _table = tf.f3, tf._table


def fh(r):
    return "-" if r["hours"] is None else f"{r['hours']:.2f}" + ("t" if r["hours_src"] == "trajectory" else "")


def fhm(rs):
    h = _mean(rs, "hours")
    return "-" if h is None else f"{h:.2f}"


def _mean(rs, k):
    xs = [r[k] for r in rs if r.get(k) is not None]
    return sum(xs) / len(xs) if xs else None


def _rows(groups, cells, mean_cells=None):
    rows = []
    for lab, g in groups.items():
        for s, r in g.items():
            rows.append([lab, f"s{s}"] + cells(r))
        if len(g) > 1 and mean_cells:
            rows.append([f"**{lab}**", f"mean n={len(g)}"] + mean_cells(list(g.values())))
    return rows


def fcon(c: dict | None) -> str:
    if not c:
        return "-"
    if "mean" not in c:
        return "missing: " + ", ".join(map(str, c.get("missing", [])))
    s = f"{c['mean']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] n{c['units']}"
    if c.get("units_dropped"):
        s += f" ({c['units_dropped']} unjoined)"
    return s


def fseeds(c: dict | None) -> str:
    if not c or "mean" not in c:
        return "-"
    if not c["paired"]:
        return f"no common seed: arm means ({'/'.join(f's{x}' for x in c['only1'])} vs " \
               f"{'/'.join(f's{x}' for x in c['only2'])})"
    s = ",".join(f"s{x}" for x in c["seeds"])
    if len(c["seeds"]) > 1:
        s += " (" + ", ".join(f"s{x} {v:+.3f}" for x, v in c["per_seed"].items()) + ")"
    un = [f"{a} " + "/".join(f"s{x}" for x in xs) for a, xs in (("arm", c["only1"]), ("ref", c["only2"])) if xs]
    return s + ("; unpaired " + ", ".join(un) if un else "")


def section_md(sec, groups, cons, cover) -> list[str]:
    dom, model = sec
    datas = sorted({r["data"] for g in groups.values() for r in g.values()})
    L = [f"## {DOMAINS[dom]} ({', '.join(datas)})" + (f" [{model}]" if model else ""), ""]
    allr = [r for g in groups.values() for r in g.values()]
    if dom == "kk":
        L.append("### Per run (kk_deep by people; dA = mean(R1,R2,C) on diag; bbh macro; train h from history, "
                 "t = last trajectory point; click = first diagA >= .40, epoch @step)")
        L += _table(["arm", "seed"] + [f"kk {b}" for b in KK_BINS] + ["dA", "bbh", "jevbench", "folio", "val",
                                                                       "train h", "click"],
                    _rows(groups, lambda r: [f3(r["kk" + b]) for b in KK_BINS] + [
                        f3(r["dA"]), f3(r["bbh"]), f3(r["jevbench"]), f3(r["folio"]), f3(r["val"]), fh(r),
                        tf.fclick(r)],
                        lambda v: [f3(_mean(v, "kk" + b)) for b in KK_BINS] + [
                            f3(_mean(v, k)) for k in ("dA", "bbh", "jevbench", "folio", "val")] + [
                            fhm(v),
                            f"clicked {sum(bool(r.get('clicked')) for r in v)}/{len(v)}"]))
        L.append("items: " + ", ".join(f"{b} {allr[0][f'kk{b}_n']}" for b in KK_BINS))
        ns = sorted({n for r in allr for n in r["people_acc"]})
        if ns:
            L += ["", "### kk_deep accuracy per people count (trained on 4-10)"]
            L += _table(["arm", "seed"] + [f"d{n}" for n in ns],
                        _rows(groups, lambda r: [f"{r['people_acc'][n]:.2f}" if n in r["people_acc"] else "-"
                                                 for n in ns]))
            L.append("items per count: " + ", ".join(f"d{n} {allr[0]['people_n'].get(n, 0)}" for n in ns))
    else:
        spec = CHAIN[dom]
        p, x = spec["p"], spec["extra"]
        ds = sorted({int(k) for r in allr for k in r[p]})
        L.append(spec["title"])
        L += _table(["arm", "seed"] + [f"d{k}" for k in ds] + [x, spec["in"], spec["beyond"], "drop", "val",
                                                               "jevbench", "train h"],
                    _rows(groups, lambda r: [f3(r[p].get(str(k))) for k in ds] + [
                        f3(r[f"{p}_{x}"]), f3(r[f"{p}_in"]), f3(r[f"{p}_beyond"]), f3(r[f"{p}_drop"], True),
                        f3(r["val"]), f3(r["jevbench"]), fh(r)],
                        lambda v: [f3(_mean([r[p] for r in v], str(k))) for k in ds] + [
                            f3(_mean(v, f"{p}_{x}")), f3(_mean(v, f"{p}_in")), f3(_mean(v, f"{p}_beyond")),
                            f3(_mean(v, f"{p}_drop"), True), f3(_mean(v, "val")), f3(_mean(v, "jevbench")),
                            fhm(v)]))
        short = [r["label"] + f" s{r['seed']}" for r in allr if r[f"{p}_in"] is not None and
                 (r[f"{p}_in_sets"] < 4 or r[f"{p}_beyond_sets"] < 4)]
        if short:
            L.append("note: fewer than 4 depth sets in a mean for " + ", ".join(short))
    L += ["", f"### Problem-view node accuracy by node depth (preds_subq, kind cot, source {NODE_SOURCE[dom]})"]
    if any(r["node_n"] for r in allr):
        L += _table(["arm", "seed", "n", "all"] + [b[0] for b in NODE_BINS],
                    _rows(groups, lambda r: [r["node_n"], f3(r["node_all"])] + [f3(r["node_bins"].get(b[0]))
                                                                               for b in NODE_BINS]))
        r0 = next(r for r in allr if r["node_n"])
        L.append("depth join (item_id, question): " + cover + "; nodes per bin: "
                 + ", ".join(f"{b[0]} {r0['node_bins_n'].get(b[0], 0)}" for b in NODE_BINS)
                 + "; per run joined via data/local/none: "
                 + ", ".join(f"{r['label']} s{r['seed']} {r['node_join']['data']}/{r['node_join']['local']}/"
                             f"{r['node_join']['none']}" for r in allr if r["node_n"]))
    else:
        L.append("no preds_subq nodes of this domain")
    L += ["", "### Local-step probe (eval key local_kk / local_kk_anc; node in its local view)"]
    loc = [r for r in allr if r["local"]]
    if loc:
        L += _table(["arm", "seed", "key", "n", "all", "derived_nontrivial"] + [b[0] for b in tf.LOCAL_BINS],
                    [[r["label"], f"s{r['seed']}", r["local_key"], r["local_n"], f3(r["local"].get("all")),
                      f3(r["local"].get("derived_nontrivial"))] + [f3(r["local"].get(b[0])) for b in tf.LOCAL_BINS]
                     for r in loc])
        if dom == "kk":
            L.append(f"(K&K template-lookup baseline on derived local views: {tf.LOOKUP})")
    else:
        L.append("no run has a local-step probe")
    L += ["", "### vs A2 / vs GFPS (difference arm - ref, paired item-level bootstrap 95% CI, n = items or diag "
              "programs joined by item_id; paired by seed label, pooled over common seeds)"]
    heads = [lab for _, lab in ENDPOINTS[dom]]
    rows = []
    for c in cons:
        if c["ref_missing"]:
            rows.append([c["arm"], c["vs"] + " (no run)"] + ["-"] * len(heads) + ["-"])
            continue
        eps = [c["endpoints"].get(ep) for ep, _ in ENDPOINTS[dom]]
        rows.append([c["arm"], c["vs"]] + [fcon(e) for e in eps]
                    + [fseeds(next((e for e in eps if e and "mean" in e), None))])
    L += _table(["arm", "vs"] + heads + ["seeds"], rows) if rows else ["no contrast (one arm only)"]
    return L + [""]


def forget_md(groups, methods) -> list[str]:
    """Mid-training general ability per run (trajectory keys FORGET); [] when no run has one."""
    rows = []
    for m, meta in methods.items():
        for lr, lab in meta["lrs"].items():
            for s, r in groups[lab].items():
                for k in FORGET:
                    pts = [(p.get("epoch"), p[k]["acc"]) for p in r["_traj"] if isinstance(p.get(k), dict)]
                    if not pts:
                        continue
                    at = lambda e, a: f"{a:.3f} @{e:.2f}" if isinstance(e, (int, float)) else f"{a:.3f}"  # noqa: E731
                    picks = [pts[int(f * (len(pts) - 1) + 0.5)] for f in (0, 0.25, 0.5, 0.75, 1)]
                    lo = min(pts, key=lambda x: x[1])
                    rows.append([m, flr(lr), f"s{s}", k] + [at(*x) for x in picks] + [at(*lo), len(pts)])
    if not rows:
        return []
    return (["### Forgetting curve (general-ability EVALMID keys in trajectory.json; accuracy @epoch; mid sets are "
             "cut to 600 items by stride)", ""]
            + _table(["method", "LR", "seed", "set", "first", "25%", "50%", "75%", "last", "min", "points"], rows)
            + [""])


def lr_md(dom, groups, methods, sel, bvb, warn, sweep=SWEEP_SEEDS) -> list[str]:
    """The LR-sweep block appended to a domain section (### headings only, so every section above stays a prefix)."""
    key, tests = SEL_KEY[dom], LR_TESTS[dom]
    L = ["### LR sweep: full curve (method x LR)", "",
         "Method = arm + tag without the LR suffix; LR from metrics.json args.lr (tag suffix lrN = N e-6 only as a "
         f"fallback). Per seed and mean over the seeds at that LR; * = selected LR; {key} = selection key (end of "
         "training); train h as above."]
    L += [f"LR WARNING: {w}" for w in warn] + [""]
    rows = []
    for m, meta in methods.items():
        for lr, lab in meta["lrs"].items():
            g = groups[lab]
            lrc = flr(lr) + (" *" if sel[m]["chosen"] == lr else "")
            rows += [[m, lrc, lab, f"s{s}", f3(r["sel_acc"])] + [f3(r.get(ep)) for ep, _ in tests] + [fh(r)]
                     for s, r in g.items()]
            if len(g) > 1:
                v = list(g.values())
                rows.append([f"**{m}**", lrc, lab, f"mean n={len(g)}", f3(_mean(v, "sel_acc"))]
                            + [f3(_mean(v, ep)) for ep, _ in tests] + [fhm(v)])
    L += _table(["method", "LR", "run label", "seed", key] + [h for _, h in tests] + ["train h"], rows)
    rule = (f"Rule (fixed before the sweep, notes/next_round/tree_forcing/lr_check.md): highest end-of-training {key} "
            f"accuracy on the sweep seed(s) {'/'.join(f's{x}' for x in sweep)} run at every LR of the method (seeds "
            "added at the selected LR never re-select); ties -> lower NLL, then the LR nearest 1e-5. Test sets are "
            "never used.")
    if dom != "kk":
        rule += (f" {DOMAINS[dom]} reuses the method's K&K-selected LR unless its own val prefers another LR by >= "
                 f"{STRONG:.2f} with a paired item-bootstrap 95% CI above 0.")
    L += ["", "### LR selection (validation only)", "", rule, ""]
    head = ["method", "LRs run (seeds)", "sweep seeds", f"{key} per LR (sweep seeds)", "own best"]
    if dom != "kk":
        head += ["K&K-selected", f"own best - K&K LR on {key} [95% CI]"]
    rows = []
    for m, meta in methods.items():
        s = sel[m]
        runs_c = ", ".join(f"{flr(lr)} ({'/'.join(f's{x}' for x in groups[lab])})" for lr, lab in meta["lrs"].items())
        if "own" not in s:
            rows.append([m, runs_c, "-", "-", "-"] + (["-", "-"] if dom != "kk" else []) + ["-", "; ".join(s["notes"])])
            continue
        sc = ", ".join(f"{flr(lr)} {v:.3f}"
                       + ("" if s["sweep_seeds"] else f" ({'/'.join(f's{x}' for x in s['seeds'][lr])})")
                       for lr, v in s["score"].items())
        row = [m, runs_c, "/".join(f"s{x}" for x in s["sweep_seeds"]) or "none", sc, flr(s["own"])]
        if dom != "kk":
            row += ["-" if s.get("kk_lr") is None else flr(s["kk_lr"]), fcon(s.get("overrule"))]
        rows.append(row + [flr(s["chosen"]), "; ".join(s["notes"]) or "-"])
    L += _table(head + ["selected", "note"], rows)
    L += [""] + forget_md(groups, methods)
    if "missing" in bvb:
        L += [f"### Best-vs-best: no {' / '.join(bvb['missing'])} run with {key}", ""]
        return L
    a, b = bvb["arm"], bvb["ref"]
    ha, hb = f"{a} @{flr(bvb['arm_lr'])}", f"{b} @{flr(bvb['ref_lr'])}"
    L += [f"### Best-vs-best: {ha} - {hb} (each at its selected LR; paired item-level bootstrap 95% CI, pooled over "
          "common seeds; means over the contrast's seeds; bbh = macro over tasks)", ""]
    rows = []
    for ep, h in tests:
        e = bvb["endpoints"][ep]
        c = e["best"]
        dflt = "-" if "default" not in e else fcon(e["default"])
        rows.append([h, f3(e["means"][bvb["ref_label"]]), f3(e["means"][bvb["arm_label"]]), fcon(c), fseeds(c), dflt])
    L += _table(["endpoint", hb, ha, f"{a.split('-')[0]} - {b.split('-')[0]} [95% CI] n", "seeds",
                 "same seeds at 1e-5 / 1e-5"], rows)
    return L + [""]


def _lr_json(x):
    """LR floats (as values and dict keys) -> '3e-6' strings for the JSON."""
    if isinstance(x, dict):
        return {(flr(k) if isinstance(k, float) else k): _lr_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_lr_json(v) for v in x]
    return None if isinstance(x, float) and math.isinf(x) else x


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run dirs, globs (quote them) or parent dirs of run dirs")
    ap.add_argument("--data-dir", default=str(ROOT / "data"), help="node depths from <data-dir>/<data>/val.jsonl")
    ap.add_argument("--boot", type=int, default=10000, help="pooled bootstrap draws")
    ap.add_argument("--boot-seed", type=int, default=2000, help="draws for each per-seed CI")
    ap.add_argument("--out", default="", help="also write <out>.md and <out>.json (a .md/.json suffix is dropped)")
    ap.add_argument("--best-arm", default="GFPS:c4", help="best-vs-best arm, ARM[:preferred tag without LR suffix]")
    ap.add_argument("--best-ref", default="A2:c12", help="best-vs-best reference, ARM[:preferred tag]")
    ap.add_argument("--sweep-seeds", default=",".join(map(str, SWEEP_SEEDS)),
                    help="seeds of the one-seed LR sweep: the LR is selected on these only (lr_check.md)")
    args = ap.parse_args(argv)
    sweep = tuple(int(x) for x in args.sweep_seeds.split(",") if x)
    best_arm, best_ref = tuple(args.best_arm.partition(":")[::2]), tuple(args.best_ref.partition(":")[::2])
    runs, skipped = [], []
    for d in expand(args.runs):
        if not d.is_dir():
            skipped.append((str(d), "not a dir"))
            continue
        if not (d / "metrics.json").exists():
            if TAIL.match(d.name):
                skipped.append((str(d), "no metrics.json (running or failed?)"))
            continue
        try:
            hint = Path(json.loads((d / "metrics.json").read_text()).get("args", {}).get("model") or "").name or None
        except (json.JSONDecodeError, AttributeError):
            hint = None
        info = parse_run_name(d.name, hint)
        if info is None:
            skipped.append((str(d), "name not <ARM>[-tag]-<model>-<data>-s<seed>"))
            continue
        runs.append(load_run(d, info))
    sections, warn = group(runs)
    data_dir = Path(args.data_dir)
    try:
        shown = data_dir.resolve().relative_to(ROOT)
    except ValueError:
        shown = data_dir
    L = ["# Ladder readout", "",
         f"runs: {len(runs)} in " + ", ".join(f"{DOMAINS[s[0]]}{f' [{s[1]}]' if s[1] else ''}: "
                                              + ", ".join(f"{lab} ({'/'.join(f's{x}' for x in g)})"
                                                          for lab, g in gs.items())
                                              for s, gs in sections.items())
         + f"; models: {', '.join(sorted({r['model'] for r in runs})) or '-'}"]
    if skipped:
        L.append(f"skipped {len(skipped)}: " + "; ".join(f"{Path(p).name} ({why})" for p, why in skipped))
    L += [f"WARNING: {w}" for w in warn] + [""]
    out = {"sections": {}, "skipped": skipped, "warnings": warn}
    kk_choice = {}                                    # model (or None) -> {method: K&K-selected LR}
    for dom in DOMAINS:
        secs = [s for s in sections if s[0] == dom]
        if not secs:
            L += [f"## {DOMAINS[dom]}: no runs", ""]
        for sec in secs:
            groups = sections[sec]
            allr = [r for g in groups.values() for r in g.values()]
            fd, fl = depth_maps(data_dir, {r["data"] for r in allr}, allr)
            for r in allr:
                node_depth_acc(r, fd, fl)
            nn = sum(r["node_n"] for r in allr)
            jd = sum(r["node_join"]["data"] for r in allr)
            jl = sum(r["node_join"]["local"] for r in allr)
            cover = (f"{jd + jl}/{nn} nodes joined ({jd} from {shown}/<data>/val.jsonl, {jl} from preds_local_kk)"
                     if nn else "no nodes")
            cons = contrasts(groups, dom, args.boot, args.boot_seed)
            L += section_md(sec, groups, cons, cover)
            methods, lwarn = lr_methods(groups)
            sel = select_lr(groups, methods, dom, kk_choice.get(sec[1]), args.boot, args.boot_seed, sweep)
            if dom == "kk":
                kk_choice[sec[1]] = {m: s["chosen"] for m, s in sel.items() if s["chosen"] is not None}
            bvb = best_vs_best(groups, methods, sel, dom, best_arm, best_ref, args.boot, args.boot_seed)
            L += lr_md(dom, groups, methods, sel, bvb, lwarn, sweep)
            out["sections"][f"{dom}" + (f"/{sec[1]}" if sec[1] else "")] = {
                "runs": {lab: {s: {k: v for k, v in r.items() if not k.startswith("_")} for s, r in g.items()}
                         for lab, g in groups.items()},
                "depth_join": cover, "contrasts": cons,
                "lr_sweep": {"methods": {m: dict(meta, lrs={flr(lr): lab for lr, lab in meta["lrs"].items()})
                                         for m, meta in methods.items()},
                             "selection": {m: _lr_json(s) for m, s in sel.items()}, "best_vs_best": _lr_json(bvb),
                             "warnings": lwarn}}
    print("\n".join(L))
    if args.out:
        p = Path(re.sub(r"\.(md|json)$", "", args.out))
        p.parent.mkdir(parents=True, exist_ok=True)
        Path(str(p) + ".md").write_text("\n".join(L) + "\n")
        Path(str(p) + ".json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    main()
