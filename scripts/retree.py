"""Re-extract sub-questions of already solved items as reasoning trees (no new solves).

Reads teacher_cache/<run>/*.json, writes teacher_cache/<run>_tree/ with the same item files, where
"subquestions" now holds typed tree nodes (parse / derive / case / verify, dependencies, corrected
nodes with their first belief, value-commitment confidence).

  python3 scripts/retree.py --runs label_kk_v1 label_jl_v1 --n 10 --workers 4 --budget 0.2 --offpeak-only
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
from cotdistill.teacher import self_corrects, tree_item  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=None, help="items per run (in file order)")
    ap.add_argument("--ids", default=None, help="comma list of item ids to process instead")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget", type=float, default=1.0, help="stop submitting new items above this logged USD")
    ap.add_argument("--offpeak-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thinking", action="store_true", help="extract every tree in thinking mode")
    ap.add_argument("--think-on-corrections", action="store_true",
                    help="thinking mode only for CoTs that say they made an error (self_corrects)")
    ap.add_argument("--suffix", default="_tree", help="output run = <run><suffix>")
    args = ap.parse_args()
    ids = set(args.ids.split(",")) if args.ids else None
    for run in args.runs:
        src, out = ROOT / "teacher_cache" / run, ROOT / "teacher_cache" / f"{run}{args.suffix}"
        out.mkdir(parents=True, exist_ok=True)
        files = [p for p in sorted(src.glob("*.json")) if p.name != "summary.json"]
        if ids:
            files = [p for p in files if p.stem.replace("__", "/") in ids]
        files = files[:args.n] if args.n else files
        todo = [p for p in files if not (out / p.name).exists()]
        client = DeepSeek(log_path=str(out / "calls.jsonl"))
        print(f"{run}: {len(files)} items, {len(todo)} to run", flush=True)

        def work(p: Path) -> str:
            while args.offpeak_only and is_peak():
                time.sleep(60)
            if client.spent_usd > args.budget:
                return f"{p.stem}: skipped (budget)"
            res = json.loads(p.read_text())
            if "item" not in res or not res.get("traces"):
                return f"{p.stem}: skipped (no trace)"
            think = args.thinking or (args.think_on_corrections and self_corrects(res["traces"][0].get("reasoning")))
            new = tree_item(client, res, seed=args.seed, thinking=think)
            (out / p.name).write_text(json.dumps(new))
            kinds = {}
            for nd in new["subquestions"]:
                kinds[nd["type"]] = kinds.get(nd["type"], 0) + 1
            return f"{p.stem}: {'think ' if think else ''}{len(new['subquestions'])} nodes {kinds}"

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
                    print("20 consecutive errors; cancelling the remaining items", flush=True)
                    ex.shutdown(wait=True, cancel_futures=True)
                    break
        print(f"{run}: {client.n_calls} calls, ${client.spent_usd:.4f}", flush=True)


if __name__ == "__main__":
    main()
