"""Focused correction pass over tree runs: for CoTs that say they made an error, add the changed judgements
as "corrected" nodes (teacher.add_corrections). Files are updated in place; done items are skipped.

  python3 scripts/add_corrections.py --runs label_kk_v1_tree label_jl_v1_tree --workers 8 --budget 1 --offpeak-only
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
from cotdistill.teacher import add_corrections, self_corrects  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--offpeak-only", action="store_true")
    args = ap.parse_args()
    ids = set(args.ids.split(",")) if args.ids else None
    for run in args.runs:
        d = ROOT / "teacher_cache" / run
        todo = []
        for p in sorted(d.glob("*.json")):
            if ids and p.stem.replace("__", "/") not in ids:
                continue
            res = json.loads(p.read_text())
            if "item" in res and "corrections_added" not in res and self_corrects(res["traces"][0].get("reasoning")):
                todo.append(p)
        client = DeepSeek(log_path=str(d / "calls_corrections.jsonl"))
        print(f"{run}: {len(todo)} self-correcting items to process", flush=True)

        def work(p: Path) -> str:
            while args.offpeak_only and is_peak():
                time.sleep(60)
            if client.spent_usd > args.budget:
                return f"{p.stem}: skipped (budget)"
            new = add_corrections(client, json.loads(p.read_text()))
            p.write_text(json.dumps(new))
            n = sum(x["status"] == "corrected" for x in new["subquestions"])
            return f"{p.stem}: +{new['corrections_added']} nodes, {n} corrected in total"

        done, streak = 0, 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(work, p): p for p in todo}
            for fu in as_completed(futs):
                done += 1
                try:
                    msg, streak = fu.result(), 0
                except Exception as e:
                    streak += 1
                    msg = f"{futs[fu].stem}: ERROR {type(e).__name__}: {str(e)[:200]}\n{traceback.format_exc(limit=2)}"
                print(f"[{done}/{len(todo)}] ${client.spent_usd:.4f} {msg}", flush=True)
                if streak >= 20:
                    print("20 consecutive errors; cancelling", flush=True)
                    ex.shutdown(wait=True, cancel_futures=True)
                    break
        print(f"{run}: {client.n_calls} calls, ${client.spent_usd:.4f}", flush=True)


if __name__ == "__main__":
    main()
