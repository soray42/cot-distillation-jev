"""Run the teacher pipeline on N generated policy cases (M1 pilot).

Usage:
  export DEEPSEEK_API_KEY=...
  python3 scripts/run_pilot.py --n 50 --k 2 --run pilot1
Outputs go to teacher_cache/<run>/ (gitignored): one JSON per item, calls.jsonl, items.jsonl.
Re-running skips items that already have an output file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.deepseek import DeepSeek, is_peak  # noqa: E402
from cotdistill.openrouter import OpenRouter  # noqa: E402
from cotdistill.policygen import HELDOUT_DOMAINS, TRAIN_DOMAINS, generate  # noqa: E402
from cotdistill.teacher import run_item, solve  # noqa: E402


def result_name(rec: dict) -> str:
    """Result file for an item; ids such as "case_001/action" contain "/" and must not become paths."""
    return rec["item_id"].replace("/", "__") + ".json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", default="pilot1")
    ap.add_argument("--domains", default=",".join(TRAIN_DOMAINS),
                    help=f"comma list; held-out: {','.join(HELDOUT_DOMAINS)}")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--backend", default="deepseek", choices=["deepseek", "openrouter"])
    ap.add_argument("--provider", default=None, help="OpenRouter provider to pin, e.g. Parasail")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget", type=float, default=1.0, help="stop submitting new items above this USD spend")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--permute", action="store_true", help="shuffle option order for traces 2..K")
    ap.add_argument("--efforts", default=None, help="comma list cycled over traces, e.g. low,high")
    ap.add_argument("--solve-only", action="store_true", help="skip sub-question stages")
    ap.add_argument("--no-random", action="store_true", help="skip the random-matched sub-question control")
    ap.add_argument("--offpeak-only", action="store_true", help="hold new items while DeepSeek peak pricing applies")
    ap.add_argument("--items", default=None, help="jsonl of pre-built records (benchmark items) instead of the generator")
    args = ap.parse_args()

    out = ROOT / "teacher_cache" / args.run
    out.mkdir(parents=True, exist_ok=True)
    if args.items:
        recs = [json.loads(l) for l in open(args.items) if l.strip()][:args.n]
    else:
        recs = generate(args.n, tuple(args.domains.split(",")), seed=args.seed)
    with open(out / "items.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    if args.backend == "openrouter":
        client = OpenRouter(model=args.model, provider=args.provider, log_path=str(out / "calls.jsonl"))
    else:
        client = DeepSeek(model=args.model, log_path=str(out / "calls.jsonl"))
    todo = [r for r in recs if not (out / result_name(r)).exists()]
    print(f"{len(recs)} items, {len(todo)} to run, k={args.k}, model={args.model}")

    def work(rec: dict) -> str:
        while args.offpeak_only and args.backend == "deepseek" and is_peak():
            time.sleep(60)
        if client.spent_usd > args.budget:
            return f"{rec['item_id']}: skipped (budget)"
        if args.solve_only:
            traces = solve(client, rec, args.k, effort=args.effort, temperature=args.temperature,
                           permute=args.permute, efforts=args.efforts.split(",") if args.efforts else None,
                           seed=args.seed)
            res = {"item": {k2: v for k2, v in rec.items() if k2 != "prompt"}, "prompt": rec["prompt"],
                   "traces": traces, "subquestions": []}
        else:
            res = run_item(client, rec, k=args.k, seed=args.seed, effort=args.effort,
                           random_matched=not args.no_random, temperature=args.temperature, permute=args.permute,
                           efforts=args.efforts.split(",") if args.efforts else None)
        (out / result_name(rec)).write_text(json.dumps(res))
        return f"{rec['item_id']}: ok, {len(res['subquestions'])} sub-questions"

    done, streak = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, r): r for r in todo}
        for fu in as_completed(futs):
            done += 1
            try:
                msg = fu.result()
                streak = 0
            except Exception as e:
                streak += 1
                msg = f"{futs[fu]['item_id']}: ERROR {type(e).__name__}: {str(e)[:200]}\n{traceback.format_exc(limit=2)}"
            print(f"[{done}/{len(todo)}] ${client.spent_usd:.4f} {msg}", flush=True)
            if streak >= 20:             # e.g. balance exhausted or results unwritable: stop spending
                print("20 consecutive errors; cancelling the remaining items", flush=True)
                ex.shutdown(wait=True, cancel_futures=True)
                break
    print(f"done: {client.n_calls} calls, ${client.spent_usd:.4f}")


if __name__ == "__main__":
    main()
