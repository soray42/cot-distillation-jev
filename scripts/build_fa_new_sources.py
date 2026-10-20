"""Training pools for the leave-one-framework-out (LOFO) failure-attribution study: AgenTracer (agtr), Who&When Pro
text (wwpro) and CORRECT-Error (ce) as one-pass student data in the StepFinder-protocol format (the schema of
data/student_fa_aftraj), disjoint at the task level from every registered FA test set; the tree-teacher inputs for
their reasoning trees; and the LOFO plan (which pools train which fold, tokens, slice-hours). CPU only; the Qwen3.5
tokenizer is the only model file read.

  cd <scratch> && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2 ~/.venvs/sev/bin/python -P -E \\
      ~/projects/jev/scripts/build_fa_new_sources.py [--sources agtr,wwpro,ce] [--no-export] [--no-lofo]

Records: the normalized records of the fa1 / fa1d builds (data/student_fa1/norm/{agentracer,wwpro}.jsonl,
data/student_fa1d/norm/correct.jsonl; the same records the registered test files were rendered from), streamed.
Per source, in this order (summary.json "filters" counts every step per group):
  1. steps      2-62 steps (the option alphabet A-Z a-z 0-9)
  2. own_test   the framework's own held-out set, as whole groups: AgenTracer's test split (790); Who&When Pro: the task
                groups of build_fa_public.wwpro_test(.1, seed 0) (the 618 trajectories of fa1_wwpro_test, reproduced and
                asserted); CORRECT: the question groups of build_fa_public.correct_test(.1, seed 0) (181 trajectories,
                question text joined with the question id without its run suffix; the 92 of fa1_correct_test inside)
  3. group      label-quality group policy (GROUP_POLICY, with the evidence); --groups overrides it
  4. clean      fa_public.clean_record v1: the injector text rules (the registered _c test files are rendered with
                them) and the meta-leak drop (trajectories that narrate the injection); AgenTracer also AGTR_META (the
                injector narration variants fa_public.META misses)
  5. labels     agent_mismatch (labelled agent != speaker of the labelled step), CORRECT's CORRECT_DROP_FLAGS
                (same_speaker_next: a structural injection tell; no_final_answer: labels re-checked 2 / 10 plausible),
                gold agent not among the agents, Who&When Pro alfagent runs whose decisive step has an empty action
                (empty_action_tell: 18 / 311 decisive steps, 0 / 9,019 other steps)
  6. dedup      task-level overlap with EVERY registered FA test set (references, below), by the rules of the _sfclean
                files (build_fa_sf_x.overlap_marks: equal normalized text; 8-gram containment >= .5 of either text after
                dropping boilerplate 8-grams of >= 5 texts, in both directions with every match over the threshold; a
                shared GAIA / AssistantBench uuid), on the record's full task text (not the 1,500-character cut)
  7. debias     build_fa_public.debias_length (the fa1b / fa1d step-length debias: within-group probes, rejection
                rounds, groups that cannot be fixed dropped; tau .01)
  8. split      val = min(--n-val, round(--val-frac x pool)) trajectories by task group (no task in val and train),
                train = the rest, at most --n-train; subsampling is stratified: per subset (framework) quotas
                proportional to the pool with a --subset-cap share, within a subset quotas over step-count terciles,
                within those equal quotas over relative-position deciles of the decisive step (build_fa_public.pick)
  9. items      build_fa_public.fit_caps (6,144 tokens with the reference line, per-step cap 80-3,000 characters) and
                build_fa_sf.build_items (fa_public.make_items: header, Task, "Step k [Agent]:" lines, the step question,
                options "Step k [Agent]", label-derived yes/no nodes, the reference line on half the items)
References (registered test sets; all records of each held-out split, also those its file leaves out): Who&When alg +
hc (184), Who&When Pro test groups, AgenTracer test (790), AFTraj test_unsafe (163), TraceElephant (219), CORRECT test
groups (181), REAL in-domain test (data/eval/fa_real_test.jsonl, 218), RootSE (all 102), LongRCA (all 1,140 runs, not
only the 290 of fa_longrca_fit). Every registered file's uids are asserted to lie in its reference set and its md5 is
recorded (summary.json references.files).

Outputs (names: agtr = AgenTracer, wwpro = Who&When Pro text, ce = CORRECT-Error):
  data/student_fa_<s>/train.jsonl, val.jsonl, summary.json, md5sums.txt     step items (schema of student_fa_aftraj)
  data/student_fa_agtr_diag/train.jsonl, val.jsonl                           the Diagnosis Error lines of
                                                                             student_fa_agtr (byte-identical subset)
  data/raw/<s>_sf/<DIR>/train/<uid>.json        train + val in StepFinder's file format (the tree teachers' input)
  data/raw/<s>_trees/views/<s>__<uid>.txt       teacher views (fa_trees.render_view, "[teacher only]" lines)
                     all_ids.json, shards/shard_KK.json (~--per-shard trajectories each), src/
  data/student_fa_lofo/plan.json                LOFO folds: training pools per held-out framework, the trajectories of
                                                the existing pools that overlap the fold's test sets, tokens per epoch,
                                                estimated 0.8B slice-hours (fitted on results/hpc/fasf)
With --trees <tree lines, a path with {subset}, e.g. data/raw/{subset}_trees/{subset}.jsonl> (opt-in; without it every
output above is byte-identical, and with it the A2 files are written first and stay byte-identical):
  data/student_fa_<s>_t2/train.jsonl, val.jsonl, summary.json, md5sums.txt   the same items with the label-derived
                                                nodes replaced by the tree's MC nodes (build_fa_sf.tree_items: converted
                                                and checked as build_fa_sf / build_fa_aftraj; no tree or a refused one:
                                                the final question only); AgenTracer: the Diagnosis Error lines only
                                                (data/student_fa_agtr_diag_t2/, the block the tree shards cover)
  v3 flags as build_fa_sf: --pointer-render free (index-free pointers, fa_trees.free_render at the item's step cap)
  and / or --trees-addon write _t3 unless --trees-suffix names the variant (e.g. --pointer-render free --trees-suffix
  _t2f)
No test file is written: the registered test files stay as they are.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
import types
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cotdistill import fa_public as fp  # noqa: E402
from cotdistill import fa_trees as ft  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_for_newsrc", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sf = _load("build_fa_sf")           # build_items, token_report, cut_stats, write, read_ww
bfp = sf.bfp                        # fit_caps, Tok, pick, rel_bin, wwpro_test, correct_test, debias_length, baselines
sfx = _load("build_fa_sf_x")        # overlap_marks, hit, subsets

SOURCES = {
    "agtr": {"origin": "agentracer", "norm": "data/student_fa1/norm/agentracer.jsonl", "dir": "AGTR",
             "name": "AgenTracer data-v1.0.0 (TracerTraj)", "test_files": ["data/eval/fa_sf_x_agentracer.jsonl",
                                                                         "data/eval/fa_sf_x_agentracer_sfclean.jsonl"]},
    "wwpro": {"origin": "wwpro", "norm": "data/student_fa1/norm/wwpro.jsonl", "dir": "WWPRO",
              "name": "Who&When Pro (text split)", "test_files": ["data/eval/fa_sf_x_wwpro.jsonl",
                                                                 "data/eval/fa_sf_x_wwpro_sfclean.jsonl"]},
    "ce": {"origin": "correct", "norm": "data/student_fa1d/norm/correct.jsonl", "dir": "CE",
           "name": "CORRECT-Error", "test_files": ["data/eval/fa_sf_x_correct.jsonl",
                                                   "data/eval/fa_sf_x_correct_sfclean.jsonl"]},
}
# label-quality groups (fa_public.clean_group keys) left out of the pools, with the evidence (label_audit.md,
# tier1_public.md fa1d, fa1b_debias.md). AgenTracer agentic Injection Error is KEPT (fa_public.CLEAN_DROP_GROUPS drops
# it: a lexical injected-think tell, +18 points held-out for the best single phrase after the meta-leak drop; AGTR_META
# below drops the explicit narration); with it out, AgenTracer is the Diagnosis Error runs only (the subset
# data/student_fa_agtr_diag). WW-Pro macnet is not listed: the length debias decides (fa1b dropped it: decisive = the
# shortest step in 89%).
GROUP_POLICY = {
    "agtr": {"agentracer/coding/coding": "labels: the labelled step is the agent's logged input prompt in 56% "
                                         "(637 / 1,141 train), the error is in the next step; next-step shift "
                                         "visibly erroneous in 10 / 24 checked",
             "agentracer/math/math": "labels: after the meta-leak drop 6 / 8 checked items have no error at the "
                                     "labelled step; 12% end with the reference answer"},
    "wwpro": {"wwpro/debate": fp.CLEAN_DROP_GROUPS["wwpro/debate"],
              "wwpro/dylan": fp.CLEAN_DROP_GROUPS["wwpro/dylan"]},
    "ce": {},
}
PROVENANCE = {   # how the decisive-step label was made
    "agentracer/agentic/Injection Error": "injected (programmatic fault injection into a run; the injected step)",
    "agentracer/agentic/Diagnosis Error": "observed (a failed run; decisive step located by the dataset's "
                                          "counterfactual diagnosis)",
    "wwpro": "injected (one realistic error injected at a known step and replayed to the end)",
    "correct": "injected (contextual error injection into a Magentic-One run; the injected step)",
}
# AgenTracer injector narration that fa_public.META misses (its "I am at step N" needs no "now"; it has "exactly as
# specified" only): in the agentic pool after the v1 meta-leak drop, at the decisive step / at any other step of
# Injection Error runs: "I am (now) at step N" 60 / 0, "specified|required implementation|approach|steps" 108 / 2,
# "the requirement is to" 34 / 4, "as instructed" 13 / 7, "exactly as described|..." 7 / 0, "where the next task is
# expected" 6 / 0; none of them in Diagnosis Error runs (0 / 2,098 steps). Such trajectories are dropped like meta_leak.
AGTR_META = re.compile(r"\bI(?: am|'m)(?: now)? at step \d+|where the next task is (?:expected|required)|"
                       r"exactly as (?:specified|described|instructed|required|stated|provided|given)|"
                       r"(?:specified|required|mandated|prescribed) "
                       r"(?:implementation|approach|steps?|method|procedure)|"
                       r"\bthe requirement is to\b|\bas instructed\b", re.I)
SHARD_BLOCKS = {"agentracer/agentic/Injection Error": 1}      # shards of block 0 first (needed under either policy)
SHARD_BLOCK_NAMES = {"agtr": {0: "Diagnosis Error", 1: "Injection Error"}}
REAL_TEST = "data/eval/fa_real_test.jsonl"
REGISTERED = {   # reference set -> the registered test files drawn from it
    "whowhen": ["data/eval/fa_sf_ww_alg.jsonl", "data/eval/fa_sf_ww_hc.jsonl"],
    "wwpro_test": SOURCES["wwpro"]["test_files"] + ["data/eval/fa1_wwpro_test_c.jsonl"],
    "agentracer_test": SOURCES["agtr"]["test_files"] + ["data/eval/fa1_agentracer_test_c.jsonl"],
    "aftraj_test": ["data/eval/fa_sf_x_aftraj.jsonl", "data/eval/fa_sf_x_aftraj_sfclean.jsonl",
                    "data/eval/fa_aftraj_test.jsonl"],
    "traceelephant": ["data/eval/fa_sf_x_traceelephant.jsonl", "data/eval/fa_sf_x_traceelephant_sfclean.jsonl"],
    "correct_test": SOURCES["ce"]["test_files"] + ["data/eval/fa1_correct_test.jsonl"],
    "real_test": [REAL_TEST],
    "rootse": ["data/eval/fa_rse.jsonl"],
    "longrca": ["data/eval/fa_longrca_fit.jsonl"],
}
LONGRCA_SPEAKERS = {"human", "Computer_terminal"}            # build_fa_longrca.PROMPT_SPEAKERS
VITA_TASK = "Customer-service dialogue (VitaBench): the agent must resolve the user's request with its tools."


# ------------------------------------------------------------------------------------------------- io
def stream(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def file_uids(path: Path) -> set:
    return {x["fa"]["uid"] for x in stream(path)}


def light(r: dict, refset: str) -> dict:
    """A reference record: uid (prefixed with its set), task, ids, subset = the reference set's name."""
    return {"uid": f"{refset}:{r['uid']}", "task": r["task"], "ids": list(r.get("ids") or fp.uuids(r["task"])),
            "subset": refset, "src_uid": r["uid"]}


def md5(path: Path) -> str:
    return bfp.md5(path)


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def fits(r: dict) -> bool:
    return 2 <= len(r["steps"]) <= fp.MAX_STEPS


# ------------------------------------------------------------------------------------------------- references
def held_wwpro(norm: Path) -> set:
    """uids of the Who&When Pro trajectories in the held-out task groups of the fa1 build (build_fa_public.stage_build:
    the 2-62-step records, wwpro_test(frac .1, seed 0))."""
    recs = [{"uid": r["uid"], "task": r["task"], "subset": r["subset"]} for r in stream(norm) if fits(r)]
    groups = bfp.wwpro_test(recs, 0.1, 0)
    return {r["uid"] for r in recs if fp.norm_task(r["task"]) in groups}


def held_correct(norm: Path) -> set:
    """uids of the CORRECT-Error trajectories in the held-out question groups of the fa1d build (correct_test(.1, 0)
    on the 2-62-step records)."""
    recs = [{"uid": r["uid"], "task": r["task"], "subset": r["subset"], "qid": r.get("qid")}
            for r in stream(norm) if fits(r)]
    return bfp.correct_test(recs, 0.1, 0)


def longrca_refs() -> list[dict]:
    """Every LongRCA run (1,140 files) with the task text build_fa_longrca renders: history[0] when it is the prompt,
    else VitaBench's fixed line."""
    out = []
    for f in sorted(glob.glob(str(ROOT / "data/raw/longrca/data/full/*/*.json"))):
        d = json.loads(Path(f).read_text())
        qid, hist = d["question_ID"], d["history"]
        dom = qid.split("__")[0]
        h0 = hist[0] if hist else {}
        if dom != "vitabench" and str(h0.get("name")) in LONGRCA_SPEAKERS and int(h0.get("step", -1)) == 0:
            task = fp.cut(fp.sanitize(str(h0.get("content", ""))), 4 * fp.TASK_CAP)
        else:
            task = VITA_TASK
        out.append({"uid": f"lrca-{qid}", "task": task, "ids": fp.uuids(task)})
    return out


def real_records() -> tuple[list[dict], list[dict]]:
    """(TrajErrBench + AgentProcessBench records, RootSE records) via build_fa_real's readers."""
    import warnings
    br = _load("build_fa_real")
    c: collections.Counter = collections.Counter()
    with warnings.catch_warnings():                      # read_apb leaves its files to the garbage collector
        warnings.simplefilter("ignore", ResourceWarning)
        out = br.read_teb(c) + br.read_apb(c), br.read_rse(c)
        import gc
        gc.collect()
    return out


def load_references(args) -> dict:
    """refset -> reference records, plus the own-test uid sets, the file checks and the md5s."""
    raw = ROOT / "data/raw"
    refs: dict = {}
    ww = [dict(r, uid=f"ww-{s}-{r['file'][:-5]}") for s in ("alg", "hc") for r in sf.read_ww(raw, s)]
    refs["whowhen"] = [light(r, "whowhen") for r in ww]
    held = {"wwpro": held_wwpro(ROOT / SOURCES["wwpro"]["norm"]), "ce": held_correct(ROOT / SOURCES["ce"]["norm"])}
    refs["wwpro_test"] = [light(r, "wwpro_test") for r in stream(ROOT / SOURCES["wwpro"]["norm"])
                          if r["uid"] in held["wwpro"]]
    refs["agentracer_test"] = [light(r, "agentracer_test") for r in stream(ROOT / SOURCES["agtr"]["norm"])
                               if r["split"] == "test"]
    refs["aftraj_test"] = [light(r, "aftraj_test") for r in stream(ROOT / "data/student_fa1/norm/aftraj.jsonl")
                           if r["split"] == "test"]
    refs["traceelephant"] = [light(r, "traceelephant") for r in
                             stream(ROOT / "data/student_fa1/norm/traceelephant.jsonl")]
    refs["correct_test"] = [light(r, "correct_test") for r in stream(ROOT / SOURCES["ce"]["norm"])
                            if r["uid"] in held["ce"]]
    real, rse = real_records()
    real_test = file_uids(ROOT / REAL_TEST)
    refs["real_test"] = [light(r, "real_test") for r in real if r["uid"] in real_test]
    refs["rootse"] = [light(r, "rootse") for r in rse]
    refs["longrca"] = [light(r, "longrca") for r in longrca_refs()]
    held["agtr"] = {r["src_uid"] for r in refs["agentracer_test"]}
    files, checks = {}, {}
    for rs, paths in REGISTERED.items():
        have = {r["src_uid"] for r in refs[rs]}
        for p in paths:
            us = file_uids(ROOT / p)
            missing = us - have
            checks[p] = {"n": len(us), "in_reference_set": rs, "missing": len(missing)}
            if missing:
                raise SystemExit(f"{p}: {len(missing)} uids outside the reference set {rs}, e.g. {sorted(missing)[:3]}")
            files[p] = md5(ROOT / p)
    fa1_ww = ROOT / "data/eval/fa1_wwpro_test.jsonl"     # the reproduced split = the fa1 build's, exactly
    if fa1_ww.exists() and file_uids(fa1_ww) != held["wwpro"]:
        raise SystemExit("wwpro_test(.1, 0) does not reproduce fa1_wwpro_test")
    sizes = {k: len(v) for k, v in refs.items()}
    print("references", sizes, f"rss {bfp.rss_mb()} MB", flush=True)
    return {"refs": refs, "held": held, "files": files, "checks": checks, "sizes": sizes,
            "held_sizes": {k: len(v) for k, v in held.items()}}


def ref_index(refs: dict) -> tuple[list[dict], dict, dict]:
    allr = [r for rs in refs.values() for r in rs]
    ids: dict = collections.defaultdict(set)
    for r in allr:
        for i in r["ids"]:
            ids[i].add(r["uid"])
    return allr, ids, {r["uid"]: r for r in allr}


def overlap(recs: list[dict], R: dict, df: int, thr: float) -> dict:
    """uid -> {"criteria": [...], "refsets": [...], "refs": [every matched reference uid]} for the records with any
    hit (reference uids are "<refset>:<uid>")."""
    allr, ids, by = R["index"]
    marks = sfx.overlap_marks(recs, allr, ids, df, thr)
    out = {}
    for u, m in marks.items():
        h = sfx.hit(m)
        if h:
            out[u] = {"criteria": h, "refsets": sfx.subsets(m, by),
                      "refs": sorted({k for c in sfx.CRITERIA for k in m[c]})}
    return out


# ------------------------------------------------------------------------------------------------- pools
def pool_records(src: str, R: dict, policy: dict, counts: collections.Counter, raw_task: dict) -> list[dict]:
    """Steps 1-5 of the module docstring for one source: the cleaned records that pass, counts per group; raw_task
    gets uid -> the task before the clean v1 text rules (the dedup compares it, as the test files were marked)."""
    cfg = SOURCES[src]
    held = R["held"][src]
    out = []
    for r in stream(ROOT / cfg["norm"]):
        g = fp.clean_group(r)
        if src == "agtr" and r["split"] == "test":
            counts[f"{g}/test_split_reference"] += 1
            continue
        counts[f"{g}/0_read"] += 1
        if not fits(r):
            counts[f"{g}/1_steps_out_of_2_62"] += 1
            continue
        if r["uid"] in held:
            counts[f"{g}/2_own_test_group"] += 1
            continue
        if g in policy:
            counts[f"{g}/3_group_policy"] += 1
            continue
        r2, why = fp.clean_record(r, "v1", drop_groups={})
        if why:
            counts[f"{g}/4_clean_{why}"] += 1
            continue
        if r["origin"] == "agentracer" and any(AGTR_META.search(t) for t in [r["task"]] + [x["text"]
                                                                                         for x in r["steps"]]):
            counts[f"{g}/4_clean_meta_leak_agtr"] += 1
            continue
        bad = []
        if "agent_mismatch" in r2["flags"]:
            bad.append("agent_mismatch")
        if r["origin"] == "correct":
            bad += sorted(set(r2["flags"]) & set(fp.CORRECT_DROP_FLAGS))
        if not bad and not r2["single_agent"] and r2.get("gold_agent") not in fp.agents_in_order(r2):
            bad.append("gold_agent_not_an_agent")
        if not bad and empty_action_tell(r2):
            bad.append("empty_action_tell")
        if bad:
            counts[f"{g}/5_label_{bad[0]}"] += 1
            continue
        counts[f"{g}/5_kept"] += 1
        counts["clean_changed_task"] += r2["task"] != r["task"]
        raw_task[r["uid"]] = r["task"]
        out.append(r2)
    return out


def empty_action_tell(r: dict) -> bool:
    """Who&When Pro alfagent: the decisive step's action is empty ("Action: " then the observation) in 18 / 311 runs and
    no other step of any run is (0 / 9,019): an exact format tell (the injected step's reasoning is not rendered)."""
    if (r["origin"], r["subset"]) != ("wwpro", "alfagent"):
        return False
    return r["steps"][r["dec"]]["text"].split("\nObservation:")[0].replace("Action:", "").strip() == ""


def length_tercile_cuts(recs: list[dict]) -> list[int]:
    lens = sorted(len(r["steps"]) for r in recs)
    return [lens[len(lens) // 3], lens[2 * len(lens) // 3]] if lens else [0, 0]


def proportional(sizes: dict, n: int, cap_share: float | None) -> dict:
    """Integer quotas summing to min(n, total) proportional to sizes, each <= its size and (when cap_share) <=
    cap_share x n; the excess goes to the others (water-filling, largest remainder)."""
    n = min(n, sum(sizes.values()))
    limit = {k: min(v, math.floor(cap_share * n) if cap_share else v) for k, v in sizes.items()}
    q = _fill(sizes, min(n, sum(limit.values())), limit)
    if sum(q.values()) < n:                          # the cap cannot be met: the rest from the remaining capacity
        extra = _fill(sizes, n - sum(q.values()), {k: sizes[k] - q[k] for k in sizes})
        q = {k: q[k] + extra[k] for k in q}
    return q


def _fill(sizes: dict, n: int, limit: dict) -> dict:
    """n units over the keys proportionally to sizes, each key at most limit[key] (water-filling, largest remainder)."""
    q = {k: 0 for k in sizes}
    free = dict(limit)
    left = n
    while left > 0:
        act = {k: v for k, v in free.items() if v > 0}
        tot = sum(sizes[k] for k in act)
        exact = {k: left * sizes[k] / tot for k in act}
        add = {k: min(free[k], int(x)) for k, x in exact.items()}
        if not any(add.values()):                    # remainders only
            for k in sorted(act, key=lambda k: (-(exact[k] - int(exact[k])), k))[:left]:
                add[k] = min(free[k], add.get(k, 0) + 1)
        for k, v in add.items():
            q[k] += v
            free[k] -= v
            left -= v
    return q


def strat_pick(recs: list[dict], n: int, key: str, cap_share: float | None, cuts: list[int],
               quotas: dict | None = None) -> list[dict]:
    """n records: subset quotas (proportional, capped; or the given quotas, topped up proportionally where a subset
    runs short), then step-count tercile quotas within a subset (proportional), then bfp.pick with equal quotas over
    relative-position deciles of the decisive step."""
    if n >= len(recs):
        return sorted(recs, key=lambda r: fp.stable_hash(key, r["uid"]))
    by = collections.defaultdict(list)
    for r in recs:
        by[r["subset"]].append(r)
    if quotas is None:
        q = proportional({k: len(v) for k, v in by.items()}, n, cap_share)
    else:
        q = {k: min(len(v), quotas.get(k, 0)) for k, v in by.items()}
        extra = proportional({k: len(v) - q[k] for k, v in by.items() if len(v) > q[k]}, n - sum(q.values()), None)
        for k, v in extra.items():
            q[k] += v
    out = []
    for s in sorted(by):
        ter = collections.defaultdict(list)
        for r in by[s]:
            ter[sum(len(r["steps"]) >= c for c in cuts)].append(r)
        qt = proportional({k: len(v) for k, v in ter.items()}, q[s], None)
        for t in sorted(ter):
            out += bfp.pick(ter[t], qt[t], f"{key}|{s}|t{t}", True)
    return out


def split_pool(pool: list[dict], args, src: str) -> tuple[list[dict], list[dict], dict]:
    """val by task group (<= 8 trajectories a group), then train from the other groups (module docstring, step 8)."""
    n_val = min(args.n_val, round(args.val_frac * len(pool)))
    cuts = length_tercile_cuts(pool)
    sub = len(pool) - n_val > args.n_train              # train is subsampled: val drawn with the same strata
    groups = collections.defaultdict(list)
    for r in pool:
        groups[fp.norm_task(r["task"])].append(r)
    cand, vg = [], set()
    sizes = collections.Counter(r["subset"] for r in pool)        # val quotas per subset = the train composition
    vq = proportional(dict(sizes), n_val, args.subset_cap.get(src))
    need = {k: (math.ceil(1.5 * q) if sub else q) for k, q in vq.items()}
    have: collections.Counter = collections.Counter()
    for g in sorted(groups, key=lambda g: fp.stable_hash(args.seed, "newval", src, g)):
        if all(have[k] >= need[k] for k in need):
            break
        if not any(have[r["subset"]] < need[r["subset"]] for r in groups[g]):
            continue
        vg.add(g)
        got = bfp.pick(groups[g], 8, f"{args.seed}|{src}|valg|{g}", False)
        cand += got
        have.update(r["subset"] for r in got)
    val = strat_pick(cand, n_val, f"{args.seed}|{src}|val", args.subset_cap.get(src), cuts, quotas=vq)
    rest = [r for r in pool if fp.norm_task(r["task"]) not in vg]
    train = strat_pick(rest, args.n_train, f"{args.seed}|{src}|train", args.subset_cap.get(src), cuts)
    assert not {fp.norm_task(r["task"]) for r in val} & {fp.norm_task(r["task"]) for r in train}
    info = {"n_val_target": n_val, "val_groups": len(vg), "val_candidates": len(cand),
            "discarded_in_val_groups": sum(len(groups[g]) for g in vg) - len(val),
            "train_pool": len(rest), "train_subsampled": len(rest) > args.n_train, "length_tercile_cuts": cuts,
            "subset_cap": args.subset_cap.get(src), "val_quotas": vq}
    return train, val, info


def recheck(train: list[dict], val: list[dict], raw_task: dict, R: dict, args) -> tuple[list, list, dict]:
    """The dedup rule again on the final files, with train + val, train and val each as the queried set (the reverse
    direction's boilerplate is computed on the queried set, so a smaller set can expose an overlap the whole pool hid,
    e.g. several pool questions on the same paper); hits are removed until every slice passes."""
    removed: dict = {}
    rounds = 0
    for rounds in range(1, 9):
        hits: dict = {}
        for name, q in (("train+val", train + val), ("train", train), ("val", val)):
            qs = [[dict(r, task=raw_task[r["uid"]]) for r in q]]
            if any(r["task"] != raw_task[r["uid"]] for r in q):      # the cleaned text too, where it differs
                qs.append(q)
            for qq in qs:
                for u, m in overlap(qq, R, args.boilerplate_df, args.overlap).items():
                    hits.setdefault(u, {"slice": name, "refsets": m["refsets"], "refs": m["refs"][:3]})
        if not hits:
            break
        removed.update(hits)
        train = [r for r in train if r["uid"] not in hits]
        val = [r for r in val if r["uid"] not in hits]
    else:
        raise SystemExit("recheck: no fixpoint in 8 rounds")
    return train, val, {"rounds": rounds, "removed": removed}


# ------------------------------------------------------------------------------------------------- stats
def position_stats(recs: list[dict]) -> dict:
    if not recs:
        return {"n": 0}
    n = len(recs)
    return {"n": n, "steps_pct": bfp.pct([len(r["steps"]) for r in recs]),
            "steps_hist_by5": bfp.hist(min(len(r["steps"]) // 5 * 5, 40) for r in recs),
            "decisive_index": bfp.hist(min(r["dec"], 20) for r in recs),
            "decisive_index_note": "index 20 = 20 or more",
            "decisive_rel_decile": bfp.hist(bfp.rel_bin(r["dec"], len(r["steps"])) for r in recs),
            "decisive_first": round(sum(r["dec"] == 0 for r in recs) / n, 3),
            "decisive_last": round(sum(r["dec"] == len(r["steps"]) - 1 for r in recs) / n, 3),
            "decisive_rel_mean": round(sum(r["dec"] / max(1, len(r["steps"]) - 1) for r in recs) / n, 3)}


def agent_stats(recs: list[dict]) -> dict:
    if not recs:
        return {"n": 0}
    n = len(recs)
    dec_agent = collections.Counter(r["steps"][r["dec"]]["agent"] for r in recs)
    return {"decisive_agent": dict(dec_agent.most_common(15)), "distinct_decisive_agents": len(dec_agent),
            "n_agents": bfp.hist(len(fp.agents_in_order(r)) for r in recs),
            "multi_agent_share": round(sum(len(fp.agents_in_order(r)) >= 2 for r in recs) / n, 3),
            "single_agent_framework_share": round(sum(bool(r["single_agent"]) for r in recs) / n, 3),
            "decisive_speaker_is_environment_or_user": sum(r["steps"][r["dec"]]["agent"].lower() in fp.NOT_AGENTS
                                                           for r in recs)}


def provenance(r: dict) -> str:
    g = fp.clean_group(r)
    return PROVENANCE.get(g) or PROVENANCE[r["origin"]]


def composition(recs: list[dict]) -> dict:
    return {"by_group": dict(collections.Counter(fp.clean_group(r) for r in recs).most_common()),
            "by_benchmark": dict(collections.Counter(f"{r['subset']}/{r['benchmark']}" for r in recs).most_common()),
            "provenance": dict(collections.Counter(provenance(r).split(" (")[0] for r in recs))}


# ------------------------------------------------------------------------------------------------- tree inputs
def mistake_reason(r: dict) -> str:
    """The view's "[teacher only] Reason" (the norm records keep no free-text reason; none of the three sources has
    one for these groups). Who&When Pro: the error-mode label only (no field of the raw record is shown)."""
    if r["origin"] == "wwpro":
        m = r.get("mode")
        name, gloss = fp.MODES.get(m, (m, m)) if m else (None, None)
        lab = f"error-mode label {m} {name} = {gloss}" if m else "no error-mode label"
        return f"(no text reason given; {lab}; find the error from the log and the reference answer)"
    if r["origin"] == "agentracer":
        if r["benchmark"] == "Diagnosis Error":
            return ("(not given: a failed run labelled by the dataset's diagnosis of the decisive step; find the "
                    "error from the log)")
        return "(not given: an injected error; find it from the log)"
    return "(not given: an injected error, and the dataset has no reference answer; find the error from the log)"


def export_sf(src: str, recs: list[dict], args, files: dict) -> dict:
    """train + val in StepFinder's file format, the teacher views, all_ids.json and the shards."""
    cfg = SOURCES[src]
    base = ROOT / args.raw_out / f"{src}_sf" / cfg["dir"] / "train"
    tb = ROOT / args.raw_out / f"{src}_trees"
    for d in (base, tb / "views", tb / "shards", tb / "src"):
        d.mkdir(parents=True, exist_ok=True)
    for old in list(base.glob("*.json")) + list((tb / "views").glob("*.txt")) + list((tb / "shards").glob("*.json")):
        old.unlink()                                     # a rebuild replaces the export (src/ is never touched)
    ids, h, block = [], hashlib.md5(), {}
    for r in sorted(recs, key=lambda r: r["uid"]):
        hist = [{"step_id": k, "name": s["agent"], "role": s["agent"], "content": s["text"]}
                for k, s in enumerate(r["steps"])]
        agent = r.get("gold_agent") or r["steps"][r["dec"]]["agent"]
        if hist[r["dec"]]["name"] != agent:
            raise SystemExit(f"{src}/{r['uid']}: decisive speaker {hist[r['dec']]['name']} != gold agent {agent}")
        d = {"question": r["task"], "ground_truth": r.get("answer") or "", "history": hist, "mistake_agent": agent,
             "mistake_step": r["dec"], "mistake_reason": mistake_reason(r)}
        p = base / f"{r['uid']}.json"
        p.write_text(json.dumps(d, ensure_ascii=False, indent=1))
        t = ft.load_traj(p, src)                         # = extract_fa_trees render (SUBSET_DIRS not needed here)
        view = ft.render_view(t) + "\n"
        (tb / "views" / f"{src}__{r['uid']}.txt").write_text(view)
        h.update(view.encode())
        ids.append({"traj_id": f"{src}/{r['uid']}", "subset": src, "n_steps": len(hist), "mistake_step": r["dec"],
                    "tercile": None, "in_fa1_pool": None})
        block[ids[-1]["traj_id"]] = SHARD_BLOCKS.get(fp.clean_group(r), 0)
    random.Random(f"{src}-trees|20261019").shuffle(ids)
    for i, x in enumerate(ids):                          # balanced length-rank targets (as build_fa_aftraj)
        x["target_len_rank"] = {"content": i % 4, "conflict": (i // 4 + i) % 4}
    (tb / "all_ids.json").write_text(json.dumps(ids, indent=0))
    shards, blocks = [], {}                              # shards per block (AgenTracer: Diagnosis Error first)
    for b in sorted(set(block.values())):
        xs = [x for x in ids if block[x["traj_id"]] == b]
        n_sh = max(1, math.ceil(len(xs) / args.per_shard))
        blocks[b] = list(range(len(shards), len(shards) + n_sh))
        shards += [xs[k::n_sh] for k in range(n_sh)]
    for k, xs in enumerate(shards):
        (tb / "shards" / f"shard_{k:02d}.json").write_text(json.dumps(xs, indent=0))
    files[rel(tb / "all_ids.json")] = len(ids)
    return {"trajectories": len(ids), "shards": len(shards), "per_shard": [len(x) for x in shards],
            "shard_blocks": {SHARD_BLOCK_NAMES.get(src, {}).get(b, "all"): v for b, v in blocks.items()},
            "sf_dir": rel(base), "trees_dir": rel(tb),
            "views_md5": h.hexdigest(), "subset_dirs_entry": {src: cfg["dir"]}}


# ------------------------------------------------------------------------------------------------- build
def build_source(src: str, R: dict, tok, args) -> dict:
    cfg = SOURCES[src]
    counts: collections.Counter = collections.Counter()
    policy = args.groups.get(src, GROUP_POLICY[src])
    raw_task: dict = {}
    pool = pool_records(src, R, policy, counts, raw_task)
    n5 = len(pool)
    ov = overlap([dict(r, task=raw_task[r["uid"]]) for r in pool], R, args.boilerplate_df, args.overlap)
    ov2 = overlap([r for r in pool if r["task"] != raw_task[r["uid"]]], R, args.boilerplate_df, args.overlap)
    ov.update({u: m for u, m in ov2.items() if u not in ov})     # the cleaned task text too, where it differs
    dd = collections.Counter()
    for u, m in ov.items():
        dd["removed"] += 1
        for c in m["criteria"]:
            dd[f"criterion/{c}"] += 1
        for s in m["refsets"]:
            dd[f"refset/{s}"] += 1
    by_uid = {r["uid"]: r for r in pool}
    for u, m in ov.items():
        counts[f"{fp.clean_group(by_uid[u])}/6_dedup_removed"] += 1
        for rs in m["refsets"]:
            dd[f"group_refset/{fp.clean_group(by_uid[u])}/{rs}"] += 1
    examples = [{"uid": u, "task": by_uid[u]["task"][:160], "criteria": m["criteria"], "refsets": m["refsets"],
                 "reference": R["index"][2][m["refs"][0]]["task"][:160]} for u, m in sorted(ov.items())[:8]]
    pool = [r for r in pool if r["uid"] not in ov]
    print(f"{src}: {n5} after the label filters, dedup removed {len(ov)} -> {len(pool)}", dict(dd), flush=True)
    db_args = types.SimpleNamespace(debias_length=cfg["origin"], debias_tau=0.01, debias_min_share=1 / 3,
                                    debias_min_n=40, debias_folds=5, debias_rounds=8, cap_max=args.cap_max,
                                    seed=args.seed)
    n6 = len(pool)
    pool, debias = bfp.debias_length(pool, db_args)
    kept = {r["uid"] for r in pool}
    for u, r in by_uid.items():
        if u not in ov and u not in kept and fits(r):
            counts[f"{fp.clean_group(r)}/7_debias_removed"] += 1
    for r in pool:
        counts[f"{fp.clean_group(r)}/7_pool"] += 1
    train, val, sinfo = split_pool(pool, args, src)
    train, val, sinfo["recheck"] = recheck(train, val, raw_task, R, args)
    for u, why in sinfo["recheck"]["removed"].items():
        counts[f"{fp.clean_group(by_uid[u])}/8_recheck_removed"] += 1
    for name, rs in (("train", train), ("val", val)):
        for r in rs:
            counts[f"{fp.clean_group(r)}/8_{name}"] += 1
    caps, cap_stats = {}, {}
    for name, rs in (("train", train), ("val", val)):
        st: dict = {}
        caps.update(bfp.fit_caps(rs, tok, args, st))
        cap_stats[name] = st | {"cut": sf.cut_stats(rs, caps)}
    fails: list = []
    tr, _ = sf.build_items(train, caps, src, "train", args, fails)
    va, _ = sf.build_items(val, caps, src, "val", args, fails)
    if fails:
        raise SystemExit(f"{src}: {len(fails)} items fail fa_public.audit_item, e.g. {fails[:3]}")
    out = ROOT / args.out_root / f"student_fa_{src}"
    files: dict = {}
    sf.write(out / "train.jsonl", tr, files)
    sf.write(out / "val.jsonl", va, files)
    variants = {}
    if src == "agtr":                                   # the strict variant: Diagnosis Error only (line subset)
        vd = ROOT / args.out_root / "student_fa_agtr_diag"
        for name, items in (("train", tr), ("val", va)):
            keep = [it for it in items if it["fa"]["benchmark"] == "Diagnosis Error"]
            sf.write(vd / f"{name}.jsonl", keep, files)
            variants[f"student_fa_agtr_diag/{name}"] = len(keep)
    export = export_sf(src, train + val, args, files) if args.export else None
    rec_by = {r["uid"]: r for r in train + val}
    trees = tree_variant(src, tr, va, rec_by, tok, args) if args.trees else None
    summary = {
        "source": cfg["name"], "subset_tag": src, "norm": cfg["norm"], "norm_md5": md5(ROOT / cfg["norm"]),
        "group_policy_dropped": policy, "provenance": {g: p for g, p in PROVENANCE.items()
                                                       if g.startswith(cfg["origin"])},
        "filters": dict(sorted(counts.items())),
        "pool": {"after_label_filters": n5, "after_dedup": n6, "after_debias": len(pool)},
        "dedup": {"removed": len(ov), "counts": dict(sorted(dd.items())), "examples": examples,
                  "rule": f"build_fa_sf_x.overlap_marks: exact / 8-gram both directions (thr {args.overlap}, "
                          f"boilerplate df {args.boilerplate_df}) / shared uuid, against every reference set"},
        "debias": debias, "split": sinfo, "sizes": {"train": len(tr), "val": len(va)} | variants,
        "composition": {"train": composition(train), "val": composition(val)},
        "positions": {"pool": position_stats(pool), "train": position_stats(train), "val": position_stats(val)},
        "agents": {"train": agent_stats(train), "val": agent_stats(val)},
        "baselines": {"train": bfp.baselines(train, train), "val": bfp.baselines(val, train)},
        "caps": cap_stats, "tokens_train": sf.token_report(tr, tok, args.budget),
        "tokens_val": sf.token_report(va, tok, args.budget), "export": export,
        "label_quality": LABEL_NOTES[src], "md5": sf.md5_all(files)}
    if trees is not None:
        summary["trees"] = trees
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    with open(out / "md5sums.txt", "w") as f:
        for p, h in sorted(summary["md5"].items()):
            f.write(f"{h}  {p}\n")
    assert len(rec_by) == len(tr) + len(va)
    print(json.dumps({"source": src, "sizes": summary["sizes"], "md5": summary["md5"]}, indent=1), flush=True)
    return summary


TREE_ONLY = {"agtr": ("student_fa_agtr_diag", "Diagnosis Error")}     # source -> (dir, the fa benchmark kept)


def tree_variant(src: str, tr: list[dict], va: list[dict], recs: dict, tok, args) -> dict:
    """--trees: the source's tree variant (default _t2; v3 flags as build_fa_sf), from the A2 items tr / va (already
    written; tree_items builds new dicts) and the train + val records (uid -> record). AgenTracer keeps its Diagnosis
    Error items only, before the conversion (TREE_ONLY). Returns the tree statistics for the A2 summary."""
    tpath = ROOT / args.trees.format(subset=src)
    if not tpath.exists():
        print(f"{src}: no trees file {tpath}", flush=True)
        return {"file": rel(tpath), "missing": True}
    tl = sf.read_lines_by_tail(tpath, src)
    name, bench = TREE_ONLY.get(src, (f"student_fa_{src}", None))
    if bench is not None:
        tr = [it for it in tr if it["fa"]["benchmark"] == bench]
        va = [it for it in va if it["fa"]["benchmark"] == bench]
    tstats = {"file": rel(tpath), "lines": len(tl), "items": 0, "items_with_tree": 0,
              "refused": collections.Counter(), "types": collections.Counter(), "n_options": collections.Counter(),
              "answer_pos": collections.Counter(), "depth": collections.Counter(),
              "answer_step_is_decisive": collections.Counter(), "facts": collections.Counter()}
    addons, suffix = sf.v3_setup(args, tstats, src)
    t_tr = sf.tree_items(tr, recs, tl, args.trees_keep_failed, tstats, addons, args.pointer_render)
    t_va = sf.tree_items(va, recs, tl, args.trees_keep_failed, tstats, addons, args.pointer_render)
    for it in t_tr + t_va:
        sf.statement_block(it)                    # the GFP-S view of every item builds
    rows = {it["fa"]["uid"] for it in tr + va}
    tstats["lines_without_row"] = sum(u not in rows for u in tl)
    d = ROOT / args.out_root / f"{name}{suffix}"
    files: dict = {}
    sf.write(d / "train.jsonl", t_tr, files)
    sf.write(d / "val.jsonl", t_va, files)
    trees = {k: (dict(sorted(v.items(), key=lambda kv: str(kv[0]))) if isinstance(v, collections.Counter) else v)
             for k, v in tstats.items()}
    trees["dir"] = rel(d)
    trees["rows"] = {"train": len(t_tr), "val": len(t_va), "train_with_tree": sum(it["has_tree"] for it in t_tr),
                     "val_with_tree": sum(it["has_tree"] for it in t_va)}
    trees["tokens_train"] = sf.token_report(t_tr, tok, args.budget)
    trees["md5"] = sf.md5_all(files)
    (d / "summary.json").write_text(json.dumps({"args": vars(args), "source": src, "trees": trees}, indent=1,
                                               ensure_ascii=False, default=str))
    with open(d / "md5sums.txt", "w") as f:
        for p, h in sorted(trees["md5"].items()):
            f.write(f"{h}  {p}\n")
    print(f"{src}: trees {rel(tpath)} -> {rel(d)}: rows {trees['rows']}, refused {trees['refused']}", flush=True)
    return trees


LABEL_NOTES = {
    "agtr": ["Injection Error (kept) carries a lexical injected-think tell ('as specified', 'required', 'implement'): "
             "+18 points held-out for the best single phrase after the meta-leak drop (fa1c); AGTR_META drops the "
             "explicit narration variants fa_public.META misses ('I am now at step N', 'the specified implementation', "
             "'the requirement is to'); the length debias removes the length part. Milder instruction-like wording "
             "can remain. data/student_fa_agtr_diag is the Diagnosis-Error-only variant.",
             "Agentic questions are generated per source document: train questions can ask about the same paper or "
             "article as a test question (a different question); the 8-gram rule treats a shared title as "
             "boilerplate when 5+ pool questions share it.",
             "Diagnosis Error: 'doubtful' in the 30-item audit (a final finish step labelled); decisive = last step "
             "in about half (fa1b: .52).",
             "agentic logs are single-speaker ('assistant'): the agent nodes are trivial.",
             "No licence in the AgenTracer release (data-v1.0.0)."],
    "wwpro": ["Labels are injected steps; the label audit found the injected step not visibly wrong in 2 / 6 sampled "
              "items (smolagents, magentic-one) and an earlier wrong answer before the injection in 19% of mathchat "
              "runs (debate / dylan 57%: dropped).",
              "Injection length tell: removed by the debias within each framework (macnet: the decisive step is the "
              "shortest in 89%; fa1b dropped it).",
              "alfagent: the reasoning field is not rendered (non-null exactly up to the injected step), so the "
              "decisive step shows only its action and observation; runs whose decisive action is empty (an exact "
              "tell, 18 / 311) are dropped.",
              "The registered test file is the uncleaned-by-debias test split (decisive = last step 42.7%)."],
    "ce": ["Labels: injected; the 20-item re-check found 8 / 10 answered runs plausible (gold = the final answer in "
           "7 / 10) and 2 / 10 runs without a final answer (those are dropped here, as in fa1d).",
           "Strong position prior: the decisive step is the last step in most runs (fa1d train 83.7%); "
           "no stratification can remove it from a pool this size.",
           "No reference answers (no reference line in any item, no Ground truth for the teacher)."],
}


# ------------------------------------------------------------------------------------------------- LOFO plan
EXISTING_POOLS = {   # pool -> (student dir, how to get the records' full tasks)
    "sf_alg": "data/student_fa_sf_alg", "sf_hc": "data/student_fa_sf_hc", "aftraj": "data/student_fa_aftraj",
    "real": "data/student_fa_real"}
FOLDS = {   # held-out framework -> (its training pools, left out of the fold; its registered test files)
    "whowhen": (["sf_alg", "sf_hc"], ["data/eval/fa_sf_ww_alg.jsonl", "data/eval/fa_sf_ww_hc.jsonl"]),
    "aftraj": (["aftraj"], ["data/eval/fa_sf_x_aftraj_sfclean.jsonl"]),
    "real": (["real"], ["data/eval/fa_real_test.jsonl"]),
    "agentracer": (["agtr"], ["data/eval/fa_sf_x_agentracer_sfclean.jsonl"]),
    "wwpro": (["wwpro"], ["data/eval/fa_sf_x_wwpro_sfclean.jsonl"]),
    "correct": (["ce"], ["data/eval/fa_sf_x_correct_sfclean.jsonl"]),
}
ALWAYS_HELD = ["data/eval/fa_sf_x_traceelephant_sfclean.jsonl", "data/eval/fa_rse.jsonl",
               "data/eval/fa_longrca_fit.jsonl"]                       # no training pool: held out in every fold
FILE_REFSET = {f: rs for rs, fs in REGISTERED.items() for f in fs}
# A2 runs (6 epochs) and their GFPS-t2 partners (3 epochs) used for the time model: (pool dir, A2 run, GFPS run)
TIMING_RUNS = [("data/student_fa_sf_alg", "A2-sfx6e-Qwen3.5-0.8B-Base-student_fa_sf_alg-s12",
                "GFPS-sfs3e-Qwen3.5-0.8B-Base-student_fa_sf_alg_t2-s12"),
               ("data/student_fa_sf_hc", "A2-sfx6e-Qwen3.5-0.8B-Base-student_fa_sf_hc-s12",
                "GFPS-sfs3e-Qwen3.5-0.8B-Base-student_fa_sf_hc_t2-s12"),
               ("data/student_fa_sf_comb", "A2-sfx6e-Qwen3.5-0.8B-Base-student_fa_sf_comb-s12",
                "GFPS-sfs3e-Qwen3.5-0.8B-Base-student_fa_sf_comb_t2-s12"),
               ("data/student_fa_aftraj", "A2-sfx6e-Qwen3.5-0.8B-Base-student_fa_aftraj-s12",
                "GFPS-sfs3e-Qwen3.5-0.8B-Base-student_fa_aftraj_t2-s12"),
               ("data/student_fa_real", "A2-sfx6r-Qwen3.5-0.8B-Base-student_fa_real-s12",
                "GFPS-sfs3r-Qwen3.5-0.8B-Base-student_fa_real_t2-s12")]


def existing_records(R: dict) -> dict:
    """pool -> {uid: record with the full task} for the train + val uids of the existing pools."""
    want = {p: set() for p in EXISTING_POOLS}
    for p, d in EXISTING_POOLS.items():
        for s in ("train", "val"):
            want[p] |= file_uids(ROOT / d / f"{s}.jsonl")
    out = {p: {} for p in EXISTING_POOLS}
    for r in stream(ROOT / "data/student_fa1/norm/stepfinder.jsonl"):
        for p in ("sf_alg", "sf_hc"):
            if r["uid"] in want[p]:
                out[p][r["uid"]] = r
    for r in stream(ROOT / "data/student_fa1/norm/aftraj.jsonl"):
        if r["uid"] in want["aftraj"]:
            out["aftraj"][r["uid"]] = r
    real, _ = real_records()
    for r in real:
        if r["uid"] in want["real"]:
            out["real"][r["uid"]] = r
    for p in EXISTING_POOLS:
        if set(out[p]) != want[p]:
            raise SystemExit(f"{p}: {len(want[p] - set(out[p]))} uids without a record")
    return out


def item_tokens(path: Path, tok) -> dict:
    """uid -> A2 tokens (prompt + "\\n\\nAnswer:") of a student file."""
    its = list(stream(path))
    n = tok.count([it["prompt"] + "\n\nAnswer:" for it in its])
    return {it["fa"]["uid"]: t for it, t in zip(its, n)}


def run_epoch_seconds(run: str) -> dict | None:
    """Training seconds per epoch of a run (results/hpc/fasf): the s_elapsed of its first --eval-at point (no
    evaluation before it) over that point's epoch; the mid-point evaluation seconds from the later gaps."""
    p = ROOT / "results/hpc/fasf" / run / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    t = m["trajectory"]
    e1, s1 = t[0]["epoch"], t[0]["s_elapsed"]
    per = s1 / e1
    gaps = [(b["s_elapsed"] - a["s_elapsed"]) - per * (b["epoch"] - a["epoch"]) for a, b in zip(t, t[1:])]
    return {"train_s_per_epoch": per, "eval_s_per_point": sum(gaps) / len(gaps) if gaps else None,
            "epochs": m["args"]["epochs"], "points": len(t)}


def fit_time_model(tok) -> dict:
    """Seconds per epoch = a N + b sum(L) + c sum(L^2) (L = A2 tokens of an item; least squares over the A2 runs) and
    the GFPS-t2 / A2 seconds-per-epoch ratio of the matched runs."""
    import numpy as np
    rows, ys, used, ratios, evals = [], [], [], {}, []
    for d, a2, gf in TIMING_RUNS:
        ra, rg = run_epoch_seconds(a2), run_epoch_seconds(gf)
        if ra is None or not (ROOT / d / "train.jsonl").exists():
            continue
        L = np.array(list(item_tokens(ROOT / d / "train.jsonl", tok).values()), dtype=float)
        rows.append([len(L), L.sum() / 1e3, (L ** 2).sum() / 1e6])
        ys.append(ra["train_s_per_epoch"])
        used.append({"pool": d, "a2_run": a2, "a2_train_s_per_epoch": round(ra["train_s_per_epoch"], 1),
                     "a2_eval_s_per_point": round(ra["eval_s_per_point"] or 0, 1), "items": len(L),
                     "tokens_per_epoch": int(L.sum())})
        evals.append(ra["eval_s_per_point"] or 0)
        if rg is not None:
            ratios[d] = round(rg["train_s_per_epoch"] / ra["train_s_per_epoch"], 3)
            used[-1]["gfps_train_s_per_epoch"] = round(rg["train_s_per_epoch"], 1)
    X, y = np.array(rows), np.array(ys)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    if (coef < 0).any():                                 # keep the model monotone: refit without the negative terms
        keep = coef >= 0
        c2, *_ = np.linalg.lstsq(X[:, keep], y, rcond=None)
        coef = np.zeros(3)
        coef[keep] = c2
    pred = X @ coef
    for u, p in zip(used, pred):
        u["model_s_per_epoch"] = round(float(p), 1)
    return {"coef_s": {"per_item": float(coef[0]), "per_1k_tokens": float(coef[1]), "per_1M_token2": float(coef[2])},
            "runs": used, "gfps_over_a2_per_epoch": ratios,
            "gfps_ratio_mean": round(sum(ratios.values()) / max(1, len(ratios)), 3),
            "eval_s_per_point_max": round(max(evals), 1),
            "note": "train seconds per epoch on one 4g.40gb slice, MBS 2 x GA 16, TOKBUDGET 12288 (the fasf settings);"
                    " eval seconds per --eval-at point with the full X_EVALS + lrca/real/rse list (sfx6r runs)"}


def epoch_seconds(model: dict, L: list[int]) -> float:
    c = model["coef_s"]
    return c["per_item"] * len(L) + c["per_1k_tokens"] * sum(L) / 1e3 + c["per_1M_token2"] * sum(x * x for x in L) / 1e6


def lofo_plan(R: dict, tok, args, new: dict) -> dict:
    """Per held-out framework: the other frameworks' pools minus their trajectories that overlap the fold's test sets
    or the always-held-out ones (TraceElephant, RootSE, LongRCA), sizes, A2 tokens per epoch and slice-hours."""
    ex = existing_records(R)
    pools: dict = {}
    for p, d in EXISTING_POOLS.items():
        pools[p] = {"dir": d, "recs": ex[p]}
    for s in new:                                     # the new pools: re-checked here (built disjoint, expect 0)
        d = f"{args.out_root}/student_fa_{s}"
        us = file_uids(ROOT / d / "train.jsonl") | file_uids(ROOT / d / "val.jsonl")
        recs = {r["uid"]: r for r in stream(ROOT / SOURCES[s]["norm"]) if r["uid"] in us and r["split"] != "test"}
        pools[s] = {"dir": d, "recs": recs, "new": True}
    marks = {p: overlap(list(v["recs"].values()), R, args.boilerplate_df, args.overlap) for p, v in pools.items()}
    file_refs = {f: {f"{FILE_REFSET[f]}:{u}" for u in file_uids(ROOT / f)} for f in FILE_REFSET}
    split_of, toks = {}, {}
    for p, v in pools.items():
        for split in ("train", "val"):
            f = ROOT / v["dir"] / f"{split}.jsonl"
            t = item_tokens(f, tok)
            toks[(p, split)] = t
            for u in t:
                split_of[(p, u)] = split
    model = fit_time_model(tok)
    folds = {}
    for fold, (own, tests) in FOLDS.items():
        held_files = tests + ALWAYS_HELD
        held = set().union(*(file_refs[f] for f in held_files))
        ent = {"held_out_tests": held_files, "train_pools": [], "pools": {}}
        L_all, nv = [], 0
        for p in pools:
            if p in own:
                continue
            drop = sorted(u for u, m in marks[p].items() if set(m["refs"]) & held)
            other = sorted(u for u, m in marks[p].items() if not set(m["refs"]) & held)
            tr = {u: t for u, t in toks[(p, "train")].items() if u not in drop}
            va = [u for u in toks[(p, "val")] if u not in drop]
            ent["train_pools"].append(p)
            ent["pools"][p] = {"train": len(tr), "val": len(va), "dropped_overlap_with_held_out": drop,
                               "dropped_by_test_file": {f: sum(bool(set(marks[p][u]["refs"]) & file_refs[f])
                                                               for u in drop) for f in held_files},
                               "overlap_with_other_registered_tests_kept": len(other),
                               "tokens_per_epoch_a2": int(sum(tr.values()))}
            L_all += list(tr.values())
            nv += len(va)
        sec = epoch_seconds(model, L_all)
        ev = model["eval_s_per_point_max"]
        a2_h = (6 * sec + 6 * ev) / 3600
        gf_h = (3 * sec * model["gfps_ratio_mean"] + 6 * ev) / 3600
        ent.update(train=len(L_all), val=nv, tokens_per_epoch_a2=int(sum(L_all)),
                   train_hours_per_epoch_a2=round(sec / 3600, 2),
                   slice_hours_a2_6ep=round(a2_h + args.job_overhead_h, 2),
                   slice_hours_gfps_3ep=round(gf_h + args.job_overhead_h, 2))
        folds[fold] = ent
    out = {"folds": folds, "time_model": model, "job_overhead_h": args.job_overhead_h,
           "estimate": "A2 6 epochs: 6 x train epoch + 5 mid-point evals + 1 final eval (each the largest measured "
                       "eval point) + job overhead; GFPS-t2 3 epochs: train epoch x the measured GFPS / A2 ratio "
                       "(mean over the matched runs; the new pools have no trees yet) x 3 + the same evals",
           "pool_overlaps_any_reference": {p: {"n": len(m), "by_refset": dict(collections.Counter(
               s for v in m.values() for s in v["refsets"]))} for p, m in marks.items()},
           "pool_overlaps_registered_files": {p: {f: sum(bool(set(v["refs"]) & file_refs[f]) for v in m.values())
                                                  for f in file_refs} for p, m in marks.items()}}
    d = ROOT / args.out_root / "student_fa_lofo"
    d.mkdir(parents=True, exist_ok=True)
    (d / "plan.json").write_text(json.dumps(out, indent=1, default=str))
    return out


# ------------------------------------------------------------------------------------------------- main
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--out-root", default="data", help="student dirs (and student_fa_lofo) under this, repo-relative")
    ap.add_argument("--raw-out", default="data/raw", help="<s>_sf and <s>_trees under this, repo-relative")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=150)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--subset-cap", default='{"wwpro": 0.4}', help="JSON: source -> max share of one subset")
    ap.add_argument("--groups", default="{}", help="JSON: source -> {group: reason} replacing GROUP_POLICY")
    ap.add_argument("--gt-p", type=float, default=0.5)
    ap.add_argument("--budget", type=int, default=6144)
    ap.add_argument("--cap-max", type=int, default=3000)
    ap.add_argument("--cap-min", type=int, default=80)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--boilerplate-df", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-shard", type=int, default=85)
    ap.add_argument("--tokenizer", default=bfp.TOKENIZER)
    ap.add_argument("--no-export", dest="export", action="store_false")
    ap.add_argument("--no-lofo", dest="lofo", action="store_false")
    ap.add_argument("--job-overhead-h", type=float, default=0.15, help="model load, smoke stage, final save")
    ap.add_argument("--trees", default="", help="MC tree lines per source, a path with {subset}, e.g. "
                    "data/raw/{subset}_trees/{subset}.jsonl (writes the _t2 variants)")
    ap.add_argument("--trees-keep-failed", action="store_true", help="also convert trees whose own checks failed")
    sf.add_v3_args(ap)
    args = ap.parse_args(argv)
    args.subset_cap = json.loads(args.subset_cap)
    args.groups = json.loads(args.groups)
    names = [s for s in args.sources.split(",") if s]
    if not set(names) <= set(SOURCES):
        raise SystemExit(f"--sources: {sorted(SOURCES)}")
    tok = bfp.Tok(args.tokenizer)
    if tok.tok is None:
        raise SystemExit("the Qwen3.5 tokenizer is needed (sev venv): " + tok.method)
    R = load_references(args)
    R["index"] = ref_index(R["refs"])
    out = {"references": {k: R[k] for k in ("sizes", "held_sizes", "files", "checks")}, "sources": {}}
    for s in names:
        out["sources"][s] = build_source(s, R, tok, args)
        out["sources"][s]["references"] = out["references"]
        p = ROOT / args.out_root / f"student_fa_{s}" / "summary.json"
        p.write_text(json.dumps(out["sources"][s], indent=1, ensure_ascii=False, default=str))
    if args.lofo and set(names) == set(SOURCES):
        out["lofo"] = lofo_plan(R, tok, args, names)
        for fold, v in out["lofo"]["folds"].items():
            print(f"LOFO {fold}: pools {v['train_pools']} train {v['train']} val {v['val']} tokens/epoch "
                  f"{v['tokens_per_epoch_a2']} A2 6ep {v['slice_hours_a2_6ep']} h, "
                  f"GFPS 3ep {v['slice_hours_gfps_3ep']} h", flush=True)
    print(f"rss {bfp.rss_mb()} MB")
    return out


if __name__ == "__main__":
    main()
