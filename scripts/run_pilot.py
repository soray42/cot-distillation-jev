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
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.deepseek import DeepSeek  # noqa: E402
from cotdistill.policygen import HELDOUT_DOMAINS, TRAIN_DOMAINS, generate  # noqa: E402
from cotdistill.teacher import run_item  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", default="pilot1")
    ap.add_argument("--domains", default=",".join(TRAIN_DOMAINS),
                    help=f"comma list; held-out: {','.join(HELDOUT_DOMAINS)}")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget", type=float, default=1.0, help="stop submitting new items above this USD spend")
    args = ap.parse_args()

    out = ROOT / "teacher_cache" / args.run
    out.mkdir(parents=True, exist_ok=True)
    recs = generate(args.n, tuple(args.domains.split(",")), seed=args.seed)
    with open(out / "items.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    client = DeepSeek(model=args.model, log_path=str(out / "calls.jsonl"))
    todo = [r for r in recs if not (out / f"{r['item_id']}.json").exists()]
    print(f"{len(recs)} items, {len(todo)} to run, k={args.k}, model={args.model}")

    def work(rec: dict) -> str:
        if client.spent_usd > args.budget:
            return f"{rec['item_id']}: skipped (budget)"
        res = run_item(client, rec, k=args.k, seed=args.seed, effort=args.effort)
        (out / f"{rec['item_id']}.json").write_text(json.dumps(res))
        return f"{rec['item_id']}: ok, {len(res['subquestions'])} sub-questions"

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, r): r for r in todo}
        for fu in as_completed(futs):
            done += 1
            try:
                msg = fu.result()
            except Exception:
                msg = f"{futs[fu]['item_id']}: ERROR\n{traceback.format_exc(limit=2)}"
            print(f"[{done}/{len(todo)}] ${client.spent_usd:.4f} {msg}", flush=True)
    print(f"done: {client.n_calls} calls, ${client.spent_usd:.4f}")


if __name__ == "__main__":
    main()
