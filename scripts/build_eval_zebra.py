"""ZebraLogic multiple-choice mode (Lin et al. 2025, "ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning",
arXiv 2502.01100) as a choice evaluation file.

ZebraLogic puzzles are logic grid puzzles: N houses (2-6), M attributes per house (2-6), a list of clues, and a unique
solution grid. The mc_mode config (huggingface.co/datasets/WildEval/ZebraLogic, file mc_mode/test-00000-of-00001.parquet;
3259 questions from 991 of the 1000 grid_mode puzzles) asks for one cell of the solution, "What is <Attribute> of the
person who lives in House <k>?", with the N values of that attribute as choices and one answer. The ungated
allenai/ZebraLogicBench copy of mc_mode ships no answers. Each item is rendered as the puzzle verbatim, then
"Question: <question>" and the choices in a seeded random order. group = zebra/<N*M>, the grid size in the notation of
the grid_mode "size" field, read from the -NxM- part of the source id. Items whose size, answer or choices cannot be
parsed are dropped and counted.

  python3 scripts/build_eval_zebra.py        # -> data/eval/zebra.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = Path("/home/soray/.claude/jobs/3b5d797f/tmp/bench/raw/zebra")
SIZE = re.compile(r"-(\d+)x(\d+)-")


def convert(r: dict, stats: collections.Counter) -> dict | None:
    size = SIZE.search(r["id"])
    choices = [str(c).strip() for c in r["choices"]]
    answer = str(r["answer"]).strip()
    puzzle, question = r["puzzle"].strip(), r["question"].strip()
    if (size is None or not puzzle or not question or answer not in choices
            or len(set(choices)) != len(choices) or "" in choices):
        stats["dropped (parse)"] += 1
        return None
    # The source id "lgp-test-6x4-37#mc-16" keeps its parts; "#" becomes "-" so the id is plain text.
    item_id = "zebra-" + r["id"].replace("#", "-")
    order = list(choices)
    random.Random(f"zebra-{item_id}").shuffle(order)
    labels = [chr(65 + i) for i in range(len(order))]
    opt_txt = "\n".join(f"{L}) {t}" for L, t in zip(labels, order))
    prompt = f"{puzzle}\n\nQuestion: {question}\nOptions:\n{opt_txt}"
    return {"item_id": item_id, "source": "zebra", "group": f"zebra/{size.group(1)}*{size.group(2)}", "type": "choice",
            "prompt": prompt, "labels": labels, "label_order": labels, "gold_label": labels[order.index(answer)],
            "label_names": dict(zip(labels, order)), "gold_probs": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW, help="directory with mc_mode_test.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "data/eval/zebra.jsonl")
    args = ap.parse_args()
    rows = sorted(pd.read_parquet(args.raw / "mc_mode_test.parquet").to_dict("records"), key=lambda r: r["id"])
    stats = collections.Counter()
    out = [it for it in (convert(r, stats) for r in rows) if it]
    print(f"{len(rows)} raw, {len(out)} kept, {stats['dropped (parse)']} dropped (parse)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for it in out:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(out)} items to {args.out}")


if __name__ == "__main__":
    main()
