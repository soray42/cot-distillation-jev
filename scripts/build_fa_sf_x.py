"""Extra test files for StepFinder's failure-attribution protocol (notes/next_round/failure_attribution/sf_protocol.md
§8): the public step-level FA test sets, checked against the rendering of the fa_sf training items and decontaminated
against StepFinder's questions (the training data of the fa_sf models), so that the per-subset StepFinder models can be
tested on far more than Who&When's 126 + 39 trajectories. CPU only; the tokenizer is only needed for the format check.

  cd <scratch> && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2 ~/.venvs/sev/bin/python -P -E \\
      ~/projects/jev/scripts/build_fa_sf_x.py

Sources (the cleaned file where cleaning changed it, as the fa1c / fa1d runs evaluate them; data/eval, kept untouched):
  wwpro          fa1_wwpro_test_c (+ _gt_c)        Who&When Pro, 10% of the tasks
  aftraj         fa1_aftraj_test (+ _gt)           AFTraj test_unsafe (clean v1 changes nothing)
  agentracer     fa1_agentracer_test_c (+ _gt_c)   AgenTracer test, meta-leak trajectories dropped by clean v1
  traceelephant  fa1_traceelephant_test (+ _gt)    TraceElephant (clean v1 changes nothing)
  correct        fa1_correct_test                  CORRECT-Error test (no reference answers: no _gt)

1. Format check (default on; --no-format-check skips it): every source item is rebuilt from its normalized record
   (data/student_fa1/norm, data/student_fa1d/norm/correct.jsonl) with the code that builds the fa_sf training items
   (fa_public.clean_record v1, build_fa_public.fit_caps with the fa_sf budget 6,144 / cap 80-3,000 characters,
   fa_public.make_items: the same header, step lines, question, options "Step k [Agent]" under A-Z a-z 0-9, no reference
   line; _gt items with it) and must equal the stored item exactly. clean v1 is checked to leave every StepFinder record
   unchanged, so the fa_sf training items are clean-v1 items too, and the length debias acts only on training pools.
   A failure stops the build.
2. Contamination against StepFinder (the fa_sf training and val questions of both subsets: every trajectory of
   data/raw/stepfinder/repo/data/*/train, checked equal to the uids of data/student_fa_sf_{alg,hc}/{train,val}.jsonl): a
   test trajectory is marked when its task (the normalized record's task, not the cut prompt) matches a StepFinder
   question by
     exact   equal fa_public.norm_task text
     ngram   fa_public.OverlapIndex (8-gram containment of either text >= --overlap after dropping boilerplate 8-grams
             in >= --boilerplate-df reference texts; the fa1 / fa_sf rule), in both directions: the StepFinder questions
             as references (every reference over the threshold, not only the best) and the test tasks as references
     id      a shared GAIA / AssistantBench uuid (the record's ids against every uuid in the StepFinder file)
   and the same three criteria against Who&When (both subsets; ids = question_ID), reported only. For test trajectories
   whose task matches a Who&When question, the log is compared with those Who&When trajectories (8-gram containment of
   the step texts, the smaller side): ww_same_trace when >= .5.
Outputs (data/eval):
  fa_sf_x_<src>.jsonl, fa_sf_x_<src>_gt.jsonl          every source item, with fa.sf_overlap / fa.sf_subsets /
                                                      fa.ww_overlap / fa.ww_subsets / fa.ww_trace (lists of criteria
                                                      and subsets; empty = none)
  fa_sf_x_<src>_sfclean.jsonl, _gt_sfclean.jsonl      without the trajectories marked against StepFinder
  fa_sf_x_summary.json                                counts, per-criterion hits, examples, Who&When overlap, the
                                                      format check, tokens per file, md5s
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import random
import sys
import types
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cotdistill import fa_public as fp  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_for_sf_x", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SOURCES = {   # name: (normalized records, step file, _gt file or None) under data/eval
    "wwpro": ("data/student_fa1/norm/wwpro.jsonl", "fa1_wwpro_test_c", "fa1_wwpro_test_gt_c"),
    "aftraj": ("data/student_fa1/norm/aftraj.jsonl", "fa1_aftraj_test", "fa1_aftraj_test_gt"),
    "agentracer": ("data/student_fa1/norm/agentracer.jsonl", "fa1_agentracer_test_c", "fa1_agentracer_test_gt_c"),
    "traceelephant": ("data/student_fa1/norm/traceelephant.jsonl", "fa1_traceelephant_test", "fa1_traceelephant_test_gt"),
    "correct": ("data/student_fa1d/norm/correct.jsonl", "fa1_correct_test", None),
}
CRITERIA = ("exact", "ngram", "id")


# ------------------------------------------------------------------------------------------------- io
def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def read_lines(path: Path) -> list[str]:
    with open(path) as f:
        return [x.rstrip("\n") for x in f if x.strip()]


def norm_records(path: Path, uids: set) -> dict:
    """uid -> normalized record, for the given uids only (streamed: the AgenTracer file is 130 MB)."""
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["uid"] in uids:
                out[r["uid"]] = r
    return out


# ------------------------------------------------------------------------------------------------- overlap
def all_matches(idx: "fp.OverlapIndex", text: str) -> tuple[float, list[str]]:
    """OverlapIndex.match with every reference at or above the threshold (match keeps only the best one): (best ratio,
    keys of all references with ratio >= idx.thr); an equal normalized text is ratio 1.0, as in match."""
    t = fp.norm_task(text)
    keys: list[str] = []
    best = 0.0
    if t in idx.exact:
        best = 1.0
        keys += idx.keys[idx.exact[t]]
    g = fp.grams(t) - idx.boiler
    hits: collections.Counter = collections.Counter()
    for x in g:
        for i in idx.index.get(x, ()):
            hits[i] += 1
    for i, s in hits.items():
        if len(g) < idx.min_grams and idx.size[i] >= idx.min_grams:
            continue
        r = max(s / max(1, len(g)), s / max(1, idx.size[i]))
        best = max(best, r)
        if r >= idx.thr:
            keys += [k for k in idx.keys[i] if k not in keys]
    return best, keys


def overlap_marks(tests: list[dict], refs: list[dict], ref_ids: dict, df: int, thr: float) -> dict:
    """Per test record uid: {"exact" | "ngram" | "id": reference uids matched, "forward" / "reverse": ngram counts by
    direction}. refs are records with uid, task, subset. ngram is the union of both directions (references indexed and
    every test task queried; test tasks indexed and every distinct reference text queried), each with every match at
    or above thr. ref_ids: uuid -> set of reference uids."""
    exact: dict = collections.defaultdict(set)
    for r in refs:
        exact[fp.norm_task(r["task"])].add(r["uid"])
    fwd = fp.OverlapIndex([(r["uid"], r["task"]) for r in refs], df=df, thr=thr)
    rev = fp.OverlapIndex([(t["uid"], t["task"]) for t in tests], df=df, thr=thr)
    rev_hits: dict = collections.defaultdict(set)
    for nt, keys in exact.items():                      # one query per distinct reference text
        _, who = all_matches(rev, nt)
        for u in who:
            rev_hits[u] |= keys
    out = {}
    for t in tests:
        _, who = all_matches(fwd, t["task"])
        out[t["uid"]] = {"exact": sorted(exact.get(fp.norm_task(t["task"]), set())),
                         "ngram": sorted(set(who) | rev_hits.get(t["uid"], set())),
                         "id": sorted({k for i in t.get("ids", []) for k in ref_ids.get(i, ())}),
                         "forward": len(who), "reverse": len(rev_hits.get(t["uid"], set()))}
    return out


def hit(m: dict) -> list[str]:
    return [c for c in CRITERIA if m[c]]


def subsets(m: dict, by_uid: dict) -> list[str]:
    return sorted({by_uid[k]["subset"] for c in CRITERIA for k in m[c]})


def log_grams(rec: dict) -> set:
    return fp.grams(" ".join(s["text"] for s in rec["steps"]))


def trace_overlap(rec: dict, cands: list[dict]) -> tuple[float, str | None]:
    """Largest 8-gram containment (the smaller side) of rec's step texts in a candidate trajectory's, and its key."""
    g = log_grams(rec)
    best, at = 0.0, None
    for c in cands:
        h = c.get("_grams")
        if h is None:
            h = c["_grams"] = log_grams(c)
        s = len(g & h) / max(1, min(len(g), len(h)))
        if s > best:
            best, at = s, c["uid"]
    return round(best, 4), at


# ------------------------------------------------------------------------------------------------- format check
def rebuild_items(recs: dict, items: list[dict], gt: bool, tok, budget: int, cap_max: int, cap_min: int) -> list[dict]:
    """The items as the fa_sf builder renders them: clean v1, fit_caps, make_items (no reference line, the fa1 node draw;
    _gt items with the line and no nodes), with the stored item ids and group prefixes."""
    bfp = _load("build_fa_public")
    rr = {}
    for u, r in recs.items():
        r2, why = fp.clean_record(r, "v1")
        if why == "meta_leak":
            raise ValueError(f"{u}: a meta-leak record is in a cleaned test file")
        rr[u] = r2
    lst = list(rr.values())
    caps = bfp.fit_caps(lst, tok, types.SimpleNamespace(budget=budget, cap_max=cap_max, cap_min=cap_min), {})
    out = []
    for it in items:
        u = it["fa"]["uid"]
        grp = it["group"].rsplit("/", 1)[0]
        if gt:
            s, _ = fp.make_items(rr[u], it["item_id"], grp, cap=caps[u], with_gt=True, rng=random.Random(0),
                                 nodes=False)
            s["has_tree"] = False
        else:
            s, _ = fp.make_items(rr[u], it["item_id"], grp, cap=caps[u], with_gt=False,
                                 rng=random.Random(f"fa1|0|{u}|items"))
        out.append(s)
    return out


def format_check(name: str, recs: dict, items: list[dict], gt: bool, tok, args) -> dict:
    rebuilt = rebuild_items(recs, items, gt, tok, args.budget, args.cap_max, args.cap_min)
    diff = collections.Counter()
    same = 0
    for a, b in zip(items, rebuilt):
        if a == b:
            same += 1
            continue
        for k in set(a) | set(b):
            if a.get(k) != b.get(k):
                diff[k] += 1
    return {"file": name, "n": len(items), "identical": same, "differing_keys": dict(diff)}


# ------------------------------------------------------------------------------------------------- main
def annotate(line: str, m_sf: dict, m_ww: dict, sf_by: dict, ww_by: dict, trace: tuple) -> str:
    """The source line with the overlap marks in fa (criteria hit and subsets matched; ww_trace = the largest log
    containment in a Who&When trajectory of a matched question, None when no Who&When question matched)."""
    it = json.loads(line)
    fa = it["fa"]
    fa["sf_overlap"] = hit(m_sf)
    fa["sf_subsets"] = subsets(m_sf, sf_by)
    fa["ww_overlap"] = hit(m_ww)
    fa["ww_subsets"] = subsets(m_ww, ww_by)
    fa["ww_trace"] = {"containment": trace[0], "ww": trace[1]} if trace[1] else None
    return json.dumps(it, ensure_ascii=False)


def ww_traces(recs: dict, m_ww: dict, ww_by: dict) -> dict:
    """uid -> (largest log containment, Who&When uid) over the Who&When trajectories its task matched."""
    out = {}
    for u, r in recs.items():
        cands = sorted({k for c in CRITERIA for k in m_ww[u][c]})
        out[u] = trace_overlap(r, [ww_by[k] for k in cands]) if cands else (0.0, None)
    return out


def source_files(lines: list[str], items: list[dict], m_sf: dict, m_ww: dict, sf_by: dict, ww_by: dict,
                 traces: dict) -> tuple[list[str], list[str]]:
    """(every line annotated, the annotated lines of the trajectories without a StepFinder match)."""
    out, clean = [], []
    for x, it in zip(lines, items):
        u = it["fa"]["uid"]
        a = annotate(x, m_sf[u], m_ww[u], sf_by, ww_by, traces[u])
        out.append(a)
        if not hit(m_sf[u]):
            clean.append(a)
    return out, clean


def hit_counts(items: list[dict], m_sf: dict, m_ww: dict, sf_by: dict, ww_by: dict, traces: dict,
               trace_thr: float) -> dict:
    st: collections.Counter = collections.Counter()
    for it in items:
        u = it["fa"]["uid"]
        s, w = m_sf[u], m_ww[u]
        for c in CRITERIA:
            st[f"sf/{c}"] += bool(s[c])
            st[f"ww/{c}"] += bool(w[c])
        st["sf/any"] += bool(hit(s))
        st["sf/id_only"] += hit(s) == ["id"]
        st["sf/ngram_reverse_only"] += bool(s["reverse"]) and not s["forward"]
        for sub in ("alg", "hc"):
            st[f"sf/any_{sub}"] += sub in subsets(s, sf_by)
            st[f"ww/any_{sub}"] += sub in subsets(w, ww_by)
        st["ww/any"] += bool(hit(w))
        st["ww/any_in_sfclean"] += bool(hit(w)) and not hit(s)
        if traces[u][1]:
            st["ww/trace_checked"] += 1
            st[f"ww/same_trace_ge{trace_thr:g}"] += traces[u][0] >= trace_thr
            st["ww/trace_containment_max_x1000"] = max(st["ww/trace_containment_max_x1000"], int(traces[u][0] * 1000))
    return dict(sorted(st.items()))


def write(path: Path, lines: list[str], files: dict) -> None:
    with open(path, "w") as f:
        for x in lines:
            f.write(x + "\n")
    files[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = len(lines)


def fa_sf_uids(dirs: str) -> dict:
    """subset -> uids of the fa_sf train + val files (the fa_sf models' training and selection data)."""
    out = {}
    for d in [x for x in dirs.split(",") if x]:
        us = set()
        for split in ("train", "val"):
            p = ROOT / d / f"{split}.jsonl"
            if p.exists():
                us |= {x["fa"]["uid"] for x in read_jsonl(p)}
        out["alg" if d.endswith("_alg") else "hc"] = us
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--out-dir", default="data/eval")
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--boilerplate-df", type=int, default=5)
    ap.add_argument("--budget", type=int, default=6144)
    ap.add_argument("--cap-max", type=int, default=3000)
    ap.add_argument("--cap-min", type=int, default=80)
    ap.add_argument("--trace-thr", type=float, default=0.5, help="log containment that counts as the same trace")
    ap.add_argument("--no-format-check", action="store_true")
    ap.add_argument("--sf-dirs", default="data/student_fa_sf_alg,data/student_fa_sf_hc",
                    help="fa_sf builds whose train / val uids must equal the StepFinder records read here")
    args = ap.parse_args(argv)
    names = [s for s in args.sources.split(",") if s]
    if not set(names) <= set(SOURCES):
        raise SystemExit(f"--sources: {sorted(SOURCES)}")
    bfp = _load("build_fa_public")
    bsf = _load("build_fa_sf")
    ev, outd = ROOT / args.eval_dir, ROOT / args.out_dir
    outd.mkdir(parents=True, exist_ok=True)
    raw = ROOT / args.raw
    # ---- references: StepFinder (both subsets, every trajectory = fa_sf train + val) and Who&When
    counts: collections.Counter = collections.Counter()
    sf = list(fp.read_stepfinder(str(raw / "stepfinder/repo/data"), counts))
    sf_check = {}
    for sub, us in fa_sf_uids(args.sf_dirs).items():
        mine = {r["uid"] for r in sf if r["subset"] == sub}
        sf_check[sub] = {"fa_sf_train_val_uids": len(us), "stepfinder_records": len(mine), "equal": us == mine}
        if us and us != mine:
            raise SystemExit(f"{sub}: fa_sf train / val uids differ from the StepFinder records read here")
    sf_by = {r["uid"]: r for r in sf}
    sf_ids: dict = collections.defaultdict(set)
    for r in sf:
        for i in r["ids"]:
            sf_ids[i].add(r["uid"])
    ww = [dict(r, uid=f"ww-{s}-{r['file'][:-5]}") for s in ("alg", "hc") for r in bsf.read_ww(raw, s)]
    ww_by = {r["uid"]: r for r in ww}
    ww_ids: dict = collections.defaultdict(set)
    for r in ww:
        if r["question_ID"]:
            ww_ids[r["question_ID"]].add(r["uid"])
    sf_clean_v1 = sum(fp.clean_record(r, "v1", {})[0] != r for r in sf)
    if sf_clean_v1:
        raise SystemExit(f"clean v1 changes {sf_clean_v1} StepFinder records: the fa_sf items are not clean-v1 items")
    print(f"StepFinder {len(sf)} trajectories ({len({fp.norm_task(r['task']) for r in sf})} distinct questions), "
          f"clean v1 changes 0, fa_sf uids {sf_check}; Who&When {len(ww)}", flush=True)
    tok = None if args.no_format_check else bfp.Tok(bfp.TOKENIZER)
    if tok is not None and tok.tok is None:
        raise SystemExit("format check needs the tokenizer (sev venv): " + tok.method)
    # ---- sources
    loaded = {}
    for name in names:
        norm, step_f, gt_f = SOURCES[name]
        lines = read_lines(ev / f"{step_f}.jsonl")
        gl = read_lines(ev / f"{gt_f}.jsonl") if gt_f else []
        items, gitems = [json.loads(x) for x in lines], [json.loads(x) for x in gl]
        uids = {it["fa"]["uid"] for it in items + gitems}
        recs = norm_records(ROOT / norm, uids)
        if len(recs) != len(uids):
            raise SystemExit(f"{name}: {len(uids) - len(recs)} uids without a normalized record")
        if len({it["fa"]["uid"] for it in items}) != len(items):
            raise SystemExit(f"{name}: a trajectory occurs twice in {step_f}")
        loaded[name] = (lines, items, gl, gitems, recs)
    tests = [r for v in loaded.values() for r in v[4].values()]
    m_sf = overlap_marks(tests, sf, sf_ids, args.boilerplate_df, args.overlap)
    m_ww = overlap_marks(tests, ww, ww_ids, args.boilerplate_df, args.overlap)
    files: dict = {}
    summary: dict = {"args": vars(args), "stepfinder": {"trajectories": len(sf), "distinct_questions":
                     len({fp.norm_task(r["task"]) for r in sf}), "clean_v1_changed": sf_clean_v1,
                     "fa_sf_uid_check": sf_check},
                     "who_and_when": {"trajectories": len(ww)}, "sources": {}, "format_check": [], "files": files}
    pool_n = 0
    for name, (lines, items, gl, gitems, recs) in loaded.items():
        norm, step_f, gt_f = SOURCES[name]
        if tok is not None:
            for fn, its, gt in ((step_f, items, False), (gt_f, gitems, True)):
                if fn:
                    fc = format_check(fn, recs, its, gt, tok, args)
                    summary["format_check"].append(fc)
                    print("format", fc, flush=True)
                    if fc["identical"] != fc["n"]:
                        raise SystemExit(f"FORMAT CHECK FAILED {fn}: {fc}")
        traces = ww_traces(recs, m_ww, ww_by)
        out_lines, clean = source_files(lines, items, m_sf, m_ww, sf_by, ww_by, traces)
        write(outd / f"fa_sf_x_{name}.jsonl", out_lines, files)
        write(outd / f"fa_sf_x_{name}_sfclean.jsonl", clean, files)
        n_gt_clean = 0
        if gt_f:
            g_out, g_clean = source_files(gl, gitems, m_sf, m_ww, sf_by, ww_by, traces)
            write(outd / f"fa_sf_x_{name}_gt.jsonl", g_out, files)
            write(outd / f"fa_sf_x_{name}_gt_sfclean.jsonl", g_clean, files)
            n_gt_clean = len(g_clean)
        st = hit_counts(items, m_sf, m_ww, sf_by, ww_by, traces, args.trace_thr)
        ex = [{"item_id": it["item_id"], "criteria": hit(m_sf[it["fa"]["uid"]]), "task": recs[it["fa"]["uid"]]["task"][:160],
               "stepfinder": [sf_by[k]["task"][:160] for k in (m_sf[it["fa"]["uid"]]["ngram"] or
                                                               m_sf[it["fa"]["uid"]]["id"])[:1]]}
              for it in items if hit(m_sf[it["fa"]["uid"]])][:6]
        id_only = [{"item_id": it["item_id"], "task": recs[it["fa"]["uid"]]["task"][:200],
                    "stepfinder": sf_by[m_sf[it["fa"]["uid"]]["id"][0]]["task"][:200],
                    "shared_ids": sorted(set(recs[it["fa"]["uid"]].get("ids", [])) &
                                         set(sf_by[m_sf[it["fa"]["uid"]]["id"][0]]["ids"]))}
                   for it in items if hit(m_sf[it["fa"]["uid"]]) == ["id"]]
        summary["sources"][name] = {"step_file": f"{args.eval_dir}/{step_f}.jsonl",
                                    "gt_file": f"{args.eval_dir}/{gt_f}.jsonl" if gt_f else None, "n": len(items),
                                    "n_sfclean": len(clean), "n_gt": len(gitems), "n_gt_sfclean": n_gt_clean,
                                    "hits": st, "examples": ex, "id_only": id_only}
        pool_n += len(clean)
        print(name, len(items), "->", len(clean), st, flush=True)
    summary["pool_sfclean_n"] = pool_n
    if tok is not None:
        summary["tokens"] = {p: int(sum(tok.count([json.loads(x)["prompt"] + "\n\nAnswer:" for x in
                                                   read_lines(ROOT / p)]))) for p in files}
    summary["md5"] = {p: bfp.md5(ROOT / p) for p in files}
    (outd / "fa_sf_x_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({"pool_sfclean_n": pool_n, "files": files, "md5": summary["md5"]}, indent=1), flush=True)
    print(f"rss {bfp.rss_mb()} MB")
    return summary


if __name__ == "__main__":
    main()
