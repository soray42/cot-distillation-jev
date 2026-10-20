"""Reasoning trees for failure attribution on StepFinder's training trajectories (src/cotdistill/fa_trees.py).

  sample    pick the pilot trajectories: per subset, the decisive speaker must equal mistake_agent, one trajectory per
            distinct question, equal numbers from each step-count tercile (seeded)
  render    write the teacher views (task, labels, log at the documented truncation rule) for a list of trajectories
  assemble  read the source trees (one JSON per trajectory, written by a teacher), render and shuffle the options,
            run every check, write <prefix>{alg,hc}.jsonl (passing trees), <prefix>rejected.jsonl and a summary

  python3 scripts/extract_fa_trees.py sample --n-alg 15 --n-hc 15 --out data/raw/stepfinder_trees/pilot_ids.json
  python3 scripts/extract_fa_trees.py render --ids data/raw/stepfinder_trees/pilot_ids.json --out-dir <scratch>/views
  python3 scripts/extract_fa_trees.py assemble --ids data/raw/stepfinder_trees/pilot_ids.json \
      --src-dir data/raw/stepfinder_trees/pilot_src --prefix data/raw/stepfinder_trees/pilot_

v3 (index-free pointers + clean nodes; teacher brief data/raw/stepfinder_trees/BRIEF_v3_addon.md):
  v3-targets  clean-node targets for every passing v2 tree (prior-biased + one random earlier step, length-rank
              targets), the add-on teacher views, the work list and --shards mixed shards (data/raw/fa_trees_v3/)
  v3-check    assemble + check add-on sources (src_v3/<subset>__<id>.json): <prefix>{subset}.jsonl, rejected, summary
  v3-merge    a v2 trees file + its add-on file -> v3 lines (index-free render at the teacher cap) and a summary

  python3 scripts/extract_fa_trees.py v3-targets
  python3 scripts/extract_fa_trees.py v3-check --ids <entries.json> --prefix <scratch>/chk_KK_ --teacher claude-opus
  python3 scripts/extract_fa_trees.py v3-merge --trees data/raw/aftraj_trees/aftraj.jsonl \
      --addon data/raw/aftraj_trees/v3add_aftraj.jsonl --out <scratch>/aftraj_v3.jsonl

No network: the teacher's trees are files on disk.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cotdistill import fa_trees as ft  # noqa: E402

RAW = ROOT / "data/raw/stepfinder/repo/data"
POOL_FILES = [ROOT / f"data/{d}/{s}.jsonl" for d in ("student_fa1", "student_fa1d") for s in ("train", "val")]
UID_PREFIX = {"alg": "alg-", "hc": "han-"}           # fa_public's StepFinder uids
# v3: one check command over mixed shards needs the raw export, the v2 trees and the add-on sources of every subset
RAW_BY_SUBSET = {"alg": RAW, "hc": RAW, "aftraj": ROOT / "data/raw/aftraj_sf", "real": ROOT / "data/raw/real_sf"}
TREES_DIR = {"alg": ROOT / "data/raw/stepfinder_trees", "hc": ROOT / "data/raw/stepfinder_trees",
             "aftraj": ROOT / "data/raw/aftraj_trees", "real": ROOT / "data/raw/real_trees"}
V2_TREES = {sub: TREES_DIR[sub] / f"{sub}.jsonl" for sub in TREES_DIR}
V3_DIR = ROOT / "data/raw/fa_trees_v3"


def src_name(traj_id: str) -> str:
    return traj_id.replace("/", "__") + ".json"


def fa1_pool(files=POOL_FILES) -> set[str]:
    """traj_ids that survive fa_public's dedup (in data/student_fa1{,d} train / val), if those files exist."""
    keep: set[str] = set()
    inv = {v: k for k, v in UID_PREFIX.items()}
    for f in files:
        if not Path(f).exists():
            continue
        with open(f) as fh:
            for line in fh:
                fa = json.loads(line).get("fa", {})
                if fa.get("origin") == "stepfinder":
                    uid = fa["uid"]
                    for pre, sub in inv.items():
                        if uid.startswith(pre):
                            keep.add(f"{sub}/{uid[len(pre):]}")
    return keep


def qkey(q: str) -> str:
    return re.sub(r"\W+", " ", q.lower())[:200]


def sample(raw: Path, n_by_subset: dict, seed: int, pool: set[str]) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for subset, n in n_by_subset.items():
        trajs = [ft.load_traj(p, subset) for p in ft.traj_paths(raw, subset)]
        ok = [t for t in trajs if t["steps"][t["mistake_step"]]["agent"] == t["mistake_agent"]]
        lens = sorted(len(t["steps"]) for t in ok)
        cuts = [lens[len(lens) // 3], lens[2 * len(lens) // 3]]
        tercile = lambda t: sum(len(t["steps"]) >= c for c in cuts)  # noqa: E731
        per = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
        used_q: set[str] = set()
        for ter in range(3):
            cand = [t for t in ok if tercile(t) == ter]
            rng.shuffle(cand)
            got = 0
            for t in cand:
                if got == per[ter]:
                    break
                if qkey(t["question"]) in used_q:
                    continue
                used_q.add(qkey(t["question"]))
                out.append({"traj_id": t["traj_id"], "subset": subset, "n_steps": len(t["steps"]),
                            "mistake_step": t["mistake_step"], "tercile": ter, "in_fa1_pool": t["traj_id"] in pool})
                got += 1
        rest = [t for t in ok if t["traj_id"] not in {r["traj_id"] for r in out}]
        if sum(r["subset"] == subset for r in out) < n:
            rng.shuffle(rest)
        else:
            rest = []
        for t in rest:                                   # a short tercile is filled from the others
            if sum(r["subset"] == subset for r in out) >= n:
                break
            if qkey(t["question"]) in used_q:
                continue
            used_q.add(qkey(t["question"]))
            out.append({"traj_id": t["traj_id"], "subset": subset, "n_steps": len(t["steps"]),
                        "mistake_step": t["mistake_step"], "tercile": tercile(t), "in_fa1_pool": t["traj_id"] in pool})
        print(f"{subset}: {len(trajs)} trajectories, {len(trajs) - len(ok)} with speaker != mistake_agent excluded, "
              f"tercile cuts at {cuts} steps", file=sys.stderr)
    return out


def load_by_id(raw: Path, traj_id: str) -> dict:
    subset, stem = traj_id.split("/")
    return ft.load_traj(raw / ft.SUBSET_DIRS[subset] / "train" / f"{stem}.json", subset)


# ---------------------------------------------------------------- v3: clean-node add-on pass

def rel(p: Path) -> str:
    p = Path(p)
    return str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(p)


def src_v3_path(traj_id: str) -> Path:
    return TREES_DIR[traj_id.split("/")[0]] / "src_v3" / src_name(traj_id)


def addon_view(traj: dict, entry: dict, line: dict) -> str:
    """The add-on teacher view: the targets with their index-free question, the decisive step's v2 content node as the
    form example, then the v2 teacher view (task, ground truth, decisive step, reason, log)."""
    out = [f"v3 ADD-ON (clean nodes) for {traj['traj_id']}  ({len(traj['steps'])} steps; source {traj['subset']})",
           f"Write: {entry['src']}", "Targets (one node per target: a clean node, or a skip if the step is itself wrong):"]
    for i, t in enumerate(entry["targets"], 1):
        out += [f"  c{i}  cite_step {t['step']}  {ft.step_label(traj, t['step'])}  target_len_rank {t['target_len_rank']}"
                f"  ({t['why']})",
                f"      student question: {ft.CLEAN_QUESTION.format(ref=t['ref'])}"]
    ms = traj["mistake_step"]
    out.append(f"[teacher only] Decisive step: {ft.step_label(traj, ms)} (every target is strictly before it)")
    cn = next((nd for nd in line.get("nodes") or [] if nd["type"] == "content"), None)
    if cn:
        out += ["Form example (the decisive step's content node in the v2 tree; write clean options of the same kind):",
                f"  Q: {cn['question']}", f"  true:  {cn['options'][cn['answer']]}"]
        out += [f"  false: {o}" for i, o in enumerate(cn["options"]) if i != cn["answer"]]
    return "\n".join(out) + "\n\n" + "-" * 100 + "\n" + ft.render_view(traj) + "\n"


def v3_targets(args) -> None:
    """Targets, views, work list and shards for the add-on pass over every passing v2 tree."""
    out = Path(args.out_dir)
    (out / "views").mkdir(parents=True, exist_ok=True)
    (out / "shards").mkdir(exist_ok=True)
    lines_by_sub = {}
    for sub in V2_TREES:
        p = Path(args.trees.format(subset=sub)) if args.trees else V2_TREES[sub]
        with open(p) as f:
            lines_by_sub[sub] = [json.loads(x) for x in f if x.strip()]
    old_path = out / "all_entries.json"          # resumable: an entry whose add-on source exists keeps its targets
    old = {e["traj_id"]: e for e in json.loads(old_path.read_text())} if old_path.exists() else {}
    kept = {t: e for t, e in old.items() if (ROOT / e["src"]).exists()}
    priors, speakers, entries, zero = {}, {}, [], collections.Counter()
    for sub, lines in lines_by_sub.items():
        pool = [ft.load_traj(p, sub) for p in ft.traj_paths(RAW_BY_SUBSET[sub], sub)]
        priors[sub] = ft.position_prior(pool)
        speakers[sub] = sorted(ft.clean_speakers(pool))
        for line in lines:
            t = load_by_id(RAW_BY_SUBSET[sub], line["traj_id"])
            tg = ft.clean_targets(t, priors[sub], set(speakers[sub]), args.seed)
            if not tg:
                zero[sub] += line["traj_id"] not in kept
                continue
            for x in tg:
                x["ref"] = ft.free_ref_for(t, x["step"])
            entries.append({"traj_id": t["traj_id"], "subset": sub, "n_steps": len(t["steps"]),
                            "mistake_step": t["mistake_step"], "targets": tg,
                            "view": rel(out / "views" / (src_name(t["traj_id"])[:-5] + ".txt")),
                            "src": rel(src_v3_path(t["traj_id"]))})
    entries.sort(key=lambda e: e["traj_id"])
    random.Random(f"v3-addon|{args.seed}").shuffle(entries)
    i = 0
    for e in entries:          # balanced length-rank targets, consecutive within a tree; assigned over the fresh
        for x in e["targets"]:  # targets of every entry, then the kept entries are put back, so a written entry
            x["target_len_rank"] = i % 4                 # never shifts the ranks of the others (stable re-runs)
            i += 1
    fresh = {e["traj_id"] for e in entries}
    entries = [dict(kept[e["traj_id"]]) if e["traj_id"] in kept else e for e in entries]
    entries += [dict(e) for t, e in sorted(kept.items()) if t not in fresh]   # kept, but no target under the rule
    by_id = {}
    for sub, lines in lines_by_sub.items():
        for line in lines:
            by_id[line["traj_id"]] = line
    for e in entries:
        t = load_by_id(RAW_BY_SUBSET[e["subset"]], e["traj_id"])
        for x in e["targets"]:                          # the reference at the current excerpt rule
            x["ref"] = ft.free_ref_for(t, x["step"])
        (ROOT / e["view"]).write_text(addon_view(t, e, by_id[e["traj_id"]]))
    (out / "all_entries.json").write_text(json.dumps(entries, indent=0, ensure_ascii=False) + "\n")
    for k in range(args.shards):
        (out / "shards" / f"shard_{k:02d}.json").write_text(json.dumps(entries[k::args.shards], indent=0,
                                                                       ensure_ascii=False) + "\n")
    for sub in TREES_DIR:
        (TREES_DIR[sub] / "src_v3").mkdir(exist_ok=True)
    nt = collections.Counter(len(e["targets"]) for e in entries)
    why = collections.Counter(x["why"] for e in entries for x in e["targets"])
    lr = collections.Counter(x["target_len_rank"] for e in entries for x in e["targets"])
    rel_pos = collections.Counter(round(x["step"] / max(1, e["n_steps"] - 1), 1) for e in entries for x in e["targets"])
    summ = {"seed": args.seed, "trees": {s: len(v) for s, v in lines_by_sub.items()},
            "entries": len(entries), "kept_with_source": len(kept),
            "by_subset": dict(collections.Counter(e["subset"] for e in entries)),
            "no_target": dict(zero), "targets_per_entry": dict(sorted(nt.items())), "targets": sum(why.values()),
            "why": dict(why), "target_len_rank": dict(sorted(lr.items())),
            "target_relative_position": dict(sorted(rel_pos.items())), "shards": args.shards,
            "shard_sizes": [len(entries[k::args.shards]) for k in range(args.shards)],
            "speakers": speakers,
            "prior_top": {s: {"abs": sorted(p["abs"].items(), key=lambda kv: -kv[1])[:6],
                              "end": sorted(p["end"].items(), key=lambda kv: -kv[1])[:6], "n": p["n"]}
                          for s, p in priors.items()}}
    (out / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps({k: summ[k] for k in ("trees", "entries", "kept_with_source", "by_subset", "no_target",
                                           "targets_per_entry", "why",
                                           "target_len_rank", "shard_sizes")}, indent=1))


def v3_summary(records: list[dict], mtimes: dict, start: float | None) -> dict:
    s: dict = {"n": len(records), "passed": sum(r["checks"]["pass"] for r in records)}
    s["pass_rate"] = round(s["passed"] / max(1, s["n"]), 4)
    s["by_subset"] = {sub: {"n": sum(r["subset"] == sub for r in records),
                            "passed": sum(r["subset"] == sub and r["checks"]["pass"] for r in records)}
                      for sub in sorted({r["subset"] for r in records})}
    s["failures_by_code"] = dict(collections.Counter(f.split(":")[0] for r in records
                                                     for f in r["checks"]["failures"]).most_common())
    nodes = [nd for r in records for nd in r["nodes"]]
    skips = [x for r in records for x in r["skips"]]
    tg = sum(len(r["meta"]["targets"]) for r in records)
    s["targets"], s["clean_nodes"], s["skips"] = tg, len(nodes), len(skips)
    s["skip_rate"] = round(len(skips) / max(1, tg), 3)
    s["skip_reasons"] = [f"{r['traj_id']} c{x['id'][1:]} step {x['cite_step']}: {x['reason']}" for r in records
                         for x in r["skips"]]
    lr = collections.Counter(v for r in records for v in r["checks"].get("length_rank", {}).values())
    s["length_rank_hist (0 = shortest)"] = dict(sorted(lr.items()))
    s["answer_index_hist"] = dict(sorted(collections.Counter(nd["answer"] for nd in nodes).items()))
    orank = collections.Counter(v for r in records for v in r["checks"].get("overlap_rank", {}).values())
    s["overlap_rank_hist (1 = most log words)"] = dict(sorted(orank.items()))
    sr = [v for r in records for v in r["checks"].get("stem_recall", {}).values()]
    s["stem_recall_of_true_mean"] = round(statistics.mean(sr), 3) if sr else None
    bs = [v for r in records for v in r["checks"].get("best_step", {}).values()]
    s["true_option_best_matches_its_step"] = f"{sum(bs)} / {len(bs)}"
    ts = sorted(mtimes.values())
    if ts:
        first = start if start is not None else ts[0]
        s["authoring_wall"] = {"span_s": round(ts[-1] - first, 1), "per_tree_s": round((ts[-1] - first) / len(ts), 1),
                               "note": "source-file mtimes; includes reading the views and fixing"}
    return s


def v3_check(args) -> None:
    ids = json.loads(Path(args.ids).read_text())
    known = {}
    if any("targets" not in e for e in ids):
        known = {e["traj_id"]: e for e in json.loads(Path(args.entries).read_text())}
    records, mtimes = [], {}
    for e in ids:
        e = e if "targets" in e else known[e["traj_id"]]
        sub = e["traj_id"].split("/")[0]
        t = load_by_id(RAW_BY_SUBSET[sub], e["traj_id"])
        p = ROOT / e["src"] if e.get("src") else src_v3_path(e["traj_id"])
        if not p.exists():
            records.append({"traj_id": t["traj_id"], "subset": sub, "teacher": "", "nodes": [], "skips": [],
                            "checks": {"pass": False, "failures": ["missing: no add-on source"]},
                            "meta": {"targets": e["targets"]}})
            continue
        mtimes[t["traj_id"]] = os.path.getmtime(p)
        try:
            src = json.loads(p.read_text())
        except json.JSONDecodeError as ex:
            records.append({"traj_id": t["traj_id"], "subset": sub, "teacher": "", "nodes": [], "skips": [],
                            "checks": {"pass": False, "failures": [f"schema: invalid JSON ({ex})"]},
                            "meta": {"targets": e["targets"]}})
            continue
        rec = ft.addon_record(t, src, e, args.seed)
        if src.get("traj_id") != t["traj_id"]:
            rec["checks"]["failures"].append("schema: traj_id mismatch")
            rec["checks"]["pass"] = False
        if args.teacher and src.get("teacher") != args.teacher:
            rec["checks"]["failures"].append(f"schema: teacher {src.get('teacher')!r}")
            rec["checks"]["pass"] = False
        records.append(rec)
    prefix = args.prefix
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    outs = {}
    for sub in sorted({r["subset"] for r in records}):
        path = Path(f"{prefix}{sub}.jsonl")
        with open(path, "w") as f:
            for r in records:
                if r["subset"] == sub and r["checks"]["pass"]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        outs[path.name] = path
    rej = Path(f"{prefix}rejected.jsonl")
    with open(rej, "w") as f:
        for r in records:
            if not r["checks"]["pass"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    outs[rej.name] = rej
    summ = v3_summary(records, mtimes, args.start_time)
    summ["teacher"] = sorted({r["teacher"] for r in records if r["teacher"]})
    summ["seed"] = args.seed
    summ["md5"] = {k: md5(v) for k, v in outs.items()}
    Path(f"{prefix}summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: summ[k] for k in ("n", "passed", "pass_rate", "failures_by_code", "skips",
                                           "length_rank_hist (0 = shortest)")}, indent=1))
    for r in records:
        if not r["checks"]["pass"]:
            print(r["traj_id"], r["checks"]["failures"])


def v3_merge(args) -> None:
    """v2 trees + add-on lines -> v3 lines at the teacher cap (the builders do the same per item at the student cap)."""
    addons = {}
    if args.addon:
        with open(args.addon) as f:
            for x in f:
                if x.strip():
                    a = json.loads(x)
                    addons[a["traj_id"]] = a
    st: collections.Counter = collections.Counter()
    n_lines = not_free = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.trees) as f, open(args.out, "w") as g:
        for x in f:
            if not x.strip():
                continue
            line = json.loads(x)
            sub = line["traj_id"].split("/")[0]
            t = load_by_id(RAW_BY_SUBSET[sub], line["traj_id"])
            v3 = ft.merge_v3(t, line, addons.get(line["traj_id"]), render=args.render, stats=st)
            not_free += sum(not ft.node_text_index_free(nd) for nd in v3["nodes"]) if args.render == "free" else 0
            n_lines += 1
            g.write(json.dumps(v3, ensure_ascii=False) + "\n")
    summ = {"lines": n_lines, "render": args.render, "nodes_with_a_step_index": not_free,
            "stats": dict(sorted(st.items())), "md5": md5(Path(args.out))}
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps(summ, indent=1))


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def q(xs, p):
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))] if xs else None


def summarize(records: list[dict], views: dict, src_chars: dict, mtimes: dict, start: float | None) -> dict:
    s: dict = {"n": len(records), "passed": sum(r["checks"]["pass"] for r in records)}
    s["pass_rate"] = round(s["passed"] / max(1, s["n"]), 4)
    s["by_subset"] = {sub: {"n": sum(r["subset"] == sub for r in records),
                            "passed": sum(r["subset"] == sub and r["checks"]["pass"] for r in records)}
                      for sub in sorted({r["subset"] for r in records})}
    fail_codes = collections.Counter(f.split(":")[0] for r in records for f in r["checks"]["failures"])
    s["failures_by_code"] = dict(fail_codes.most_common())
    s["drops_by_reason"] = dict(collections.Counter(r["checks"]["failures"][0].split(":")[0]
                                                    for r in records if not r["checks"]["pass"]).most_common())
    nodes = [nd for r in records for nd in r["nodes"]]
    s["nodes"] = {"total": len(nodes), "per_tree_mean": round(len(nodes) / max(1, len(records)), 2),
                  "by_type": dict(collections.Counter(nd["type"] for nd in nodes)),
                  "options_by_type": {t: dict(collections.Counter(len(nd["options"]) for nd in nodes
                                                                  if nd["type"] == t)) for t in ft.NODE_TYPES}}
    s["chain_length"] = dict(collections.Counter(len(r["checks"].get("chain", [])) for r in records))
    ranks = collections.Counter()
    opt_k = collections.Counter()
    for r in records:
        for nid, rk in r["checks"].get("length_rank", {}).items():
            nd = next(x for x in r["nodes"] if x["id"] == nid)
            ranks[rk] += 1
            opt_k[len(nd["options"])] += 1
    s["length_rank_of_correct"] = {"hist (1 = longest)": dict(sorted(ranks.items())),
                                   "longest_share": round(ranks[1] / max(1, sum(ranks.values())), 3),
                                   "chance_longest": round(sum(c / k for k, c in opt_k.items())
                                                           / max(1, sum(opt_k.values())), 3)}
    pos = collections.Counter(nd["answer"] for nd in nodes)
    s["answer_index_hist"] = dict(sorted(pos.items()))
    probes = [v for r in records for v in r["checks"].get("question_probe", {}).values()]
    s["question_probe"] = {"n": len(probes), "question_only": sum(p["question_only"] for p in probes),
                           "question_match_tell": sum(p["question_match_tell"] for p in probes)}
    gr = [v for r in records for v in r["checks"].get("grounding", {}).values()]
    s["grounding"] = {"n": len(gr), "correct_best_step_is_cited": sum(g["correct_best_step"] == g["cited"] for g in gr),
                      "distractors_best_step_not_cited": sum(sum(b != g["cited"] for b in g["distractor_best_steps"])
                                                             for g in gr),
                      "distractors": sum(len(g["distractor_best_steps"]) for g in gr)}
    ar = [r["checks"].get("answer_rule", {}) for r in records]
    s["answer_rule"] = {"rule_found": sum(a.get("rule") is not None for a in ar),
                        "by_how": dict(collections.Counter(a.get("how") for a in ar)),
                        "agree": sum(a.get("agree") is True for a in ar),
                        "disagree": sum(a.get("agree") is False for a in ar)}
    caps = [c for r in records for c in r["checks"].get("quote_min_cap", {}).values() if c is not None]
    s["quote_min_cap"] = {"max": max(caps) if caps else None, "p90": q(caps, .9),
                          "over_step_cap": sum(c > ft.STEP_CAP for c in caps)}
    s["label_notes"] = sum(bool(r["checks"].get("label_note")) for r in records)
    s["in_fa1_pool"] = sum(bool(r["meta"].get("in_fa1_pool")) for r in records)
    vc = [len(views[r["traj_id"]]) for r in records]
    sc = [src_chars[r["traj_id"]] for r in records if r["traj_id"] in src_chars]
    s["chars"] = {"view_mean": round(statistics.mean(vc)) if vc else None, "view_max": max(vc) if vc else None,
                  "source_tree_mean": round(statistics.mean(sc)) if sc else None,
                  "note": "tokens ~ chars / 3.5 (English with code and numbers); no Claude tokenizer offline"}
    ts = sorted(mtimes.values())
    if ts:
        first = start if start is not None else ts[0]
        s["authoring_wall"] = {"start": first, "last_tree": ts[-1], "span_s": round(ts[-1] - first, 1),
                               "per_tree_s": round((ts[-1] - first) / len(ts), 1),
                               "note": "file mtimes of the source trees; includes reading the views and fixing"}
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("sample")
    a.add_argument("--raw", default=str(RAW))
    a.add_argument("--n-alg", type=int, default=15)
    a.add_argument("--n-hc", type=int, default=15)
    a.add_argument("--seed", type=int, default=20261010)
    a.add_argument("--out", required=True)
    b = sub.add_parser("render")
    b.add_argument("--raw", default=str(RAW))
    b.add_argument("--ids", required=True)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--no-teacher-info", action="store_true")
    c = sub.add_parser("assemble")
    c.add_argument("--raw", default=str(RAW))
    c.add_argument("--ids", required=True)
    c.add_argument("--src-dir", required=True)
    c.add_argument("--prefix", required=True, help="output prefix, e.g. data/raw/stepfinder_trees/pilot_")
    c.add_argument("--seed", type=int, default=20261010)
    c.add_argument("--start-time", type=float, default=None, help="epoch seconds when authoring started")
    c.add_argument("--teacher", default=None, help="require this teacher name in every source tree")
    d = sub.add_parser("v3-targets")
    d.add_argument("--trees", default="", help="v2 trees per subset, a path with {subset} (default: the four files)")
    d.add_argument("--out-dir", default=str(V3_DIR))
    d.add_argument("--shards", type=int, default=12)
    d.add_argument("--seed", type=int, default=20261019)
    e = sub.add_parser("v3-check")
    e.add_argument("--ids", required=True, help="JSON list of work-list entries (or {traj_id} looked up in --entries)")
    e.add_argument("--entries", default=str(V3_DIR / "all_entries.json"))
    e.add_argument("--prefix", required=True)
    e.add_argument("--seed", type=int, default=20261010)
    e.add_argument("--start-time", type=float, default=None)
    e.add_argument("--teacher", default=None)
    m = sub.add_parser("v3-merge")
    m.add_argument("--trees", required=True)
    m.add_argument("--addon", default="")
    m.add_argument("--render", choices=("index", "free"), default="free")
    m.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd in ("v3-targets", "v3-check", "v3-merge"):
        return {"v3-targets": v3_targets, "v3-check": v3_check, "v3-merge": v3_merge}[args.cmd](args)
    raw = Path(args.raw)

    if args.cmd == "sample":
        rows = sample(raw, {"alg": args.n_alg, "hc": args.n_hc}, args.seed, fa1_pool())
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1) + "\n")
        for r in rows:
            print(r)
        return

    ids = json.loads(Path(args.ids).read_text())
    if args.cmd == "render":
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for r in ids:
            t = load_by_id(raw, r["traj_id"])
            (out / (src_name(t["traj_id"])[:-5] + ".txt")).write_text(
                ft.render_view(t, teacher_info=not args.no_teacher_info) + "\n")
        print(f"wrote {len(ids)} views to {out}")
        return

    src_dir = Path(args.src_dir)
    records, views, src_chars, mtimes = [], {}, {}, {}
    for r in ids:
        t = load_by_id(raw, r["traj_id"])
        views[t["traj_id"]] = ft.render_view(t)
        p = src_dir / src_name(t["traj_id"])
        meta = {"in_fa1_pool": r.get("in_fa1_pool"), "tercile": r.get("tercile")}
        if not p.exists():
            records.append({"traj_id": t["traj_id"], "subset": t["subset"], "teacher": "", "nodes": [],
                            "checks": {"pass": False, "failures": ["missing: no source tree"]}, "meta": meta})
            continue
        text = p.read_text()
        src_chars[t["traj_id"]] = len(text)
        mtimes[t["traj_id"]] = os.path.getmtime(p)
        try:
            src = json.loads(text)
        except json.JSONDecodeError as e:
            records.append({"traj_id": t["traj_id"], "subset": t["subset"], "teacher": "", "nodes": [],
                            "checks": {"pass": False, "failures": [f"schema: invalid JSON ({e})"]}, "meta": meta})
            continue
        rec = ft.output_record(t, src, args.seed, meta)
        if src.get("traj_id") != t["traj_id"]:
            rec["checks"]["failures"].append("schema: traj_id mismatch")
            rec["checks"]["pass"] = False
        if args.teacher and src.get("teacher") != args.teacher:
            rec["checks"]["failures"].append(f"schema: teacher {src.get('teacher')!r}")
            rec["checks"]["pass"] = False
        records.append(rec)

    prefix = args.prefix
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    outs = {}
    for subset in sorted({rec["subset"] for rec in records}):
        path = Path(f"{prefix}{subset}.jsonl")
        with open(path, "w") as f:
            for rec in records:
                if rec["subset"] == subset and rec["checks"]["pass"]:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        outs[path.name] = path
    rej = Path(f"{prefix}rejected.jsonl")
    with open(rej, "w") as f:
        for rec in records:
            if not rec["checks"]["pass"]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    outs[rej.name] = rej
    summ = summarize(records, views, src_chars, mtimes, args.start_time)
    summ["teacher"] = sorted({r["teacher"] for r in records if r["teacher"]})
    summ["spend_usd"] = 0.0
    summ["seed"] = args.seed
    summ["step_cap"], summ["task_cap"] = ft.STEP_CAP, ft.TASK_CAP
    summ["md5"] = {k: md5(v) for k, v in outs.items()}
    Path(f"{prefix}summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps({k: summ[k] for k in ("n", "passed", "pass_rate", "failures_by_code")}, indent=1))
    for rec in records:
        if not rec["checks"]["pass"]:
            print(rec["traj_id"], rec["checks"]["failures"])


if __name__ == "__main__":
    main()
