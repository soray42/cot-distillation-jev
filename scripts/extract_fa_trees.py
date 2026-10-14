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
    args = ap.parse_args()
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
