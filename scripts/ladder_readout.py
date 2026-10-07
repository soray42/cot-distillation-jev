"""Ladder readout: one command for single- (or multi-) seed arm runs on the two Tree Forcing domains. CPU only, numpy.

Run dirs are <ARM>[-<tag>]-<model>-<data>-s<seed> (hpc/train_arm.sbatch OUT), any model name; runs are grouped by
(arm, tag, data) (and by model too when several models are passed). Domain from the data part of the name: "numgen" in
it -> numeric chains (student_numgen_tf), else K&K (student_v3sl). Arguments are run dirs, globs, or a parent dir whose
children are run dirs (e.g. results/hpc/<pull>/). Dirs without metrics.json (still running) are listed as skipped.

K&K, per run: kk_deep accuracy by people bins 4-10 / 11-16 / 17-20 / 11-20 and per people count (group kk/dN or kkx/dN),
  dA = analyze_e2.run_summary(run)["A"] (needs preds_diag), bbh (macro over tasks), jevbench, folio, val, training hours
  (history steps x s/step, i.e. the whole run incl. mid evals; "t" = fallback to the last trajectory s_elapsed), click
  = the first trajectory point (or the end eval) with diagA >= .40, as epoch @step (tf_readout.curve).
Numeric chains, per run: accuracy on ng_d1..ng_d8 and ng_ood (eval keys ng_d1 / ng1, ng_ood / ngood), mean over d1-d4
  (in range) and d5-d8 (beyond the training depth), and drop = beyond - in range.
Both: problem-view node accuracy by node depth (preds_subq, kind "cot", source knights_knaves / numgen, `correct`) with the
  depth joined by (item_id, question) from data/<data>/val.jsonl subqs' node_depth (also their "opposite" wording) and,
  for what that misses, preds_local_kk.jsonl of any run of the same domain (the numeric local probe runs on
  local_deep.jsonl, whose items are not the val items, so it alone would join nothing); bins 0 / 1-3 / 4-7 / 8-12 / >=13,
  join coverage printed. Local-step probe (eval key local_kk or local_kk_anc, both domains): all, derived_nontrivial,
  by depth bin.
Contrasts: every arm vs A2 (c12 preferred) and vs GFPS (c4 preferred) of the same domain, on kk 11-20 and dA (K&K) and
  beyond d5-d8 (numeric): mean difference and paired item-level bootstrap 95% CI (items / diag programs joined by
  item_id; tf_readout.contrast: paired by seed label, pooled over common seeds; no common seed -> difference of arm means,
  still item-paired, labelled).

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python scripts/ladder_readout.py 'results/hpc/l4b/*' --out results/ladder_readout/l4b
"""
from __future__ import annotations

import argparse
import glob
import json
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
DOMAINS = {"kk": "K&K", "numgen": "Numeric chains"}
NODE_SOURCE = {"kk": "knights_knaves", "numgen": "numgen"}
ENDPOINTS = {"kk": [("kk11-20", "kk 11-20"), ("dA", "dA")], "numgen": [("ng_beyond", "beyond d5-d8")]}
KK_BINS = ["4-10", "11-16", "17-20", "11-20"]
NODE_BINS = tf.LOCAL_BINS                               # 0 / 1-3 / 4-7 / 8-12 / >=13
NG_DEPTH = re.compile(r"^ng_?d?(\d+)$")
NG_OOD = re.compile(r"^ng_?ood$")
TAIL = re.compile(r"^(?P<head>.+)-(?P<data>student_[A-Za-z0-9_]+)-s(?P<seed>\d+)$")
TAG_TOKEN = re.compile(r"^[a-z][a-z0-9.]*$")


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
    return "numgen" if "numgen" in data else "kk"


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


def ng_sets(d: Path, ev: dict) -> dict:
    """{depth (int) or "ood": eval name} from metrics eval keys and preds files."""
    names = set(ev) | {f.name[len("preds_"):-len(".jsonl")] for f in d.glob("preds_ng*.jsonl")}
    out = {}
    for n in sorted(names):
        m = NG_DEPTH.match(n)
        if m:
            out.setdefault(int(m[1]), n)
        elif NG_OOD.match(n):
            out.setdefault("ood", n)
    return out


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
    if dom == "kk":
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
    else:
        sets = ng_sets(d, ev)
        r["ng_names"] = {str(k): v for k, v in sets.items()}
        r["ng"], per_item = {}, {}
        for k, name in sets.items():
            f = d / f"preds_{name}.jsonl"
            if f.exists():
                ps = [p for p in read_preds(f) if p.get("gold_label") in p["labels"]]
                per_item[k] = {f"{name}/{p['item_id']}": float(correct(p)) for p in ps}
                r["ng"][str(k)] = sum(per_item[k].values()) / len(ps) if ps else None
            else:
                r["ng"][str(k)] = set_acc(d, ev, name)
        r["ng_ood"] = r["ng"].pop("ood", None)
        for lab, ks in (("ng_in", (1, 2, 3, 4)), ("ng_beyond", (5, 6, 7, 8))):
            xs = [r["ng"][str(k)] for k in ks if r["ng"].get(str(k)) is not None]
            r[lab] = sum(xs) / len(xs) if xs else None
            r[lab + "_sets"] = len(xs)
            u = {}
            for k in ks:
                u.update(per_item.get(k, {}))
            units[lab] = u
        r["ng_drop"] = r["ng_beyond"] - r["ng_in"] if r["ng_in"] is not None and r["ng_beyond"] is not None else None
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
        ds = sorted({int(k) for r in allr for k in r["ng"]})
        L.append("### Per run (accuracy per chain depth; trained on d1-d4; in = mean d1-d4, beyond = mean d5-d8, "
                 "drop = beyond - in; train h from history, t = last trajectory point)")
        L += _table(["arm", "seed"] + [f"d{k}" for k in ds] + ["ood", "in d1-4", "beyond d5-8", "drop", "val",
                                                               "jevbench", "train h"],
                    _rows(groups, lambda r: [f3(r["ng"].get(str(k))) for k in ds] + [
                        f3(r["ng_ood"]), f3(r["ng_in"]), f3(r["ng_beyond"]), f3(r["ng_drop"], True), f3(r["val"]),
                        f3(r["jevbench"]), fh(r)],
                        lambda v: [f3(_mean([r["ng"] for r in v], str(k))) for k in ds] + [
                            f3(_mean(v, "ng_ood")), f3(_mean(v, "ng_in")), f3(_mean(v, "ng_beyond")),
                            f3(_mean(v, "ng_drop"), True), f3(_mean(v, "val")), f3(_mean(v, "jevbench")),
                            fhm(v)]))
        short = [r["label"] + f" s{r['seed']}" for r in allr if r["ng_in"] is not None and
                 (r["ng_in_sets"] < 4 or r["ng_beyond_sets"] < 4)]
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


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run dirs, globs (quote them) or parent dirs of run dirs")
    ap.add_argument("--data-dir", default=str(ROOT / "data"), help="node depths from <data-dir>/<data>/val.jsonl")
    ap.add_argument("--boot", type=int, default=10000, help="pooled bootstrap draws")
    ap.add_argument("--boot-seed", type=int, default=2000, help="draws for each per-seed CI")
    ap.add_argument("--out", default="", help="also write <out>.md and <out>.json (a .md/.json suffix is dropped)")
    args = ap.parse_args(argv)
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
            out["sections"][f"{dom}" + (f"/{sec[1]}" if sec[1] else "")] = {
                "runs": {lab: {s: {k: v for k, v in r.items() if not k.startswith("_")} for s, r in g.items()}
                         for lab, g in groups.items()},
                "depth_join": cover, "contrasts": cons}
    print("\n".join(L))
    if args.out:
        p = Path(re.sub(r"\.(md|json)$", "", args.out))
        p.parent.mkdir(parents=True, exist_ok=True)
        Path(str(p) + ".md").write_text("\n".join(L) + "\n")
        Path(str(p) + ".json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    main()
