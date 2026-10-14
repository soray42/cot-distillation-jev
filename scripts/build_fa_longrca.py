"""LongRCA Bench (Hugging Face CLoud5-real/longrca-bench; arXiv 2608.15242) as a step-level failure-attribution test
file in the StepFinder-protocol format (fa_public.make_items: the log, "which step contains the decisive error", options
"Step k [Agent]" under A-Z a-z 0-9, no reference answer).

LongRCA: 1,140 observed (non-injected) failed trajectories from SWE-bench Pro, Terminal-Bench 2, TravelPlanner,
VitaBench and WebArena Verified; human labels of the responsible role and the earliest decisive step (Who&When schema:
question_ID, history[{step, content, role, name}], mistake_agent, mistake_step indexing history, mistake_reason).

Rendering:
  - task: history[0] when it is the prompt (speaker "human" or "Computer_terminal" at step 0: SWE-bench Pro,
    Terminal-Bench 2, TravelPlanner, WebArena); the steps are history[1:] and the gold index mistake_step - 1. VitaBench
    opens with the agent's greeting, so every entry is a step and the task is a fixed line.
  - agent names as Who&When's (text before " (", e.g. "DiagnostAgent (-> ActionAgent)" -> "DiagnostAgent").
  - only trajectories of 2-62 steps fit the option alphabet; the summary counts the rest per domain. Most LongRCA logs
    are far longer (median 165-258 steps in three domains), so this file is the fitting subset, not the benchmark.
  - per-step caps by build_fa_public.fit_caps at --budget tokens (default 16384: the FA runs evaluate at 20,480).

  CUDA_VISIBLE_DEVICES= ~/.venvs/sev/bin/python -P -E scripts/build_fa_longrca.py
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cotdistill import fa_public as fp  # noqa: E402

spec = importlib.util.spec_from_file_location("build_fa_public_for_lrca", ROOT / "scripts/build_fa_public.py")
bfp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfp)

PROMPT_SPEAKERS = {"human", "Computer_terminal"}
VITA_TASK = "Customer-service dialogue (VitaBench): the agent must resolve the user's request with its tools."


def ww_name(x) -> str:
    return str(x or "unknown").split(" (")[0].strip()


def record(d: dict, counts: collections.Counter) -> dict | None:
    qid = d["question_ID"]
    dom = qid.split("__")[0]
    hist = list(d["history"])
    k = int(d["mistake_step"])
    if dom != "vitabench" and str(hist[0].get("name")) in PROMPT_SPEAKERS and int(hist[0]["step"]) == 0:
        task, steps, dec = str(hist[0]["content"]), hist[1:], k - 1
    else:
        task, steps, dec = VITA_TASK, hist, k
    if dec < 0:
        counts[f"{dom}/drop/label_on_prompt"] += 1
        return None
    if not 2 <= len(steps) <= fp.MAX_STEPS:
        counts[f"{dom}/drop/steps_over_{fp.MAX_STEPS}"] += 1
        return None
    gold = ww_name(d["mistake_agent"])
    if ww_name(steps[dec].get("name")) != gold:
        counts[f"{dom}/speaker_differs_from_mistake_agent"] += 1
    counts[f"{dom}/kept"] += 1
    return fp._rec(f"lrca-{qid}", "longrca", dom, "natural", task, None,
                   [(ww_name(s.get("name")), str(s.get("content", ""))) for s in steps], dec, gold, split="test")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default="data/raw/longrca/data/full")
    ap.add_argument("--out", default="data/eval/fa_longrca_fit.jsonl")
    ap.add_argument("--budget", type=int, default=16384)
    ap.add_argument("--cap-max", type=int, default=3000)
    ap.add_argument("--cap-min", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default=bfp.TOKENIZER)
    args = ap.parse_args()
    counts: collections.Counter = collections.Counter()
    recs = []
    for f in sorted((ROOT / args.src).glob("*/*.json")):
        r = record(json.loads(f.read_text()), counts)
        if r is not None:
            recs.append(r)
    tok = bfp.Tok(args.tokenizer)
    st: dict = {}
    caps = bfp.fit_caps(recs, tok, args, st)
    items, fails = [], []
    for r in recs:
        s, _ = fp.make_items(r, f"fa-lrca-{r['uid'][5:]}", f"fa_sf/longrca/{r['subset']}", cap=caps[r["uid"]],
                             with_gt=False, rng=random.Random(f"fa1|{args.seed}|{r['uid']}|items"))
        b = fp.audit_item(s)
        if b:
            fails.append((s["item_id"], b[:3]))
        items.append(s)
    if fails:
        raise SystemExit(f"{len(fails)} items fail fa_public.audit_item, e.g. {fails[:3]}")
    out = ROOT / args.out
    with open(out, "w") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    pos = collections.Counter()
    for r in recs:
        n = len(r["steps"])
        pos["first"] += r["dec"] == 0
        pos["last"] += r["dec"] == n - 1
    summary = {"args": vars(args), "counts": dict(sorted(counts.items())), "items": len(items),
               "by_domain": dict(collections.Counter(r["subset"] for r in recs)), "caps": st,
               "gold_first_step": pos["first"], "gold_last_step": pos["last"],
               "md5": __import__("hashlib").md5(out.read_bytes()).hexdigest()}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in ("items", "by_domain", "gold_first_step", "gold_last_step", "md5")}))
    print(json.dumps(summary["counts"]))


if __name__ == "__main__":
    main()
