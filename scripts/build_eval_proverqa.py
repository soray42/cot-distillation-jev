"""ProverQA (Qi et al., ICLR 2025) dev set as a FOLIO-format evaluation file.

ProverQA is first-order-logic reasoning generated with the Prover9 prover: a context of premises, one statement, and
the answer True, False or Uncertain, in three difficulty levels (easy 1-2 steps, medium 3-5, hard 6-9; 500 items each).
The official release (huggingface.co/datasets/opendatalab/ProverQA, files dev/{easy,medium,hard}.json) asks
"Based on the above information, is the following statement true, false, or uncertain? <statement>" with fixed
options A) True, B) False, C) Uncertain. Each item is rendered in the frame of data/eval/folio_val.jsonl: premises one
per line, then "Statement: ...", then the FOLIO question, with the three options in a seeded random order.

Premise lines come from walking the context and matching the longest premise sentence listed in the item's nl2fol map
at each position, so a sentence that occurs twice in the context stays twice; if the walk does not cover the context,
the context paragraph is kept as one line. depth = number of premise lines, as for FOLIO. Items whose statement or
answer cannot be parsed are dropped and counted. The reasoning chains and FOL fields are not used.

  python3 scripts/build_eval_proverqa.py      # -> data/eval/proverqa.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = Path("/home/soray/.claude/jobs/3b5d797f/tmp/bench/raw/proverqa")
LEVELS = ["easy", "medium", "hard"]
ANSWERS = ["True", "False", "Uncertain"]
QPREFIX = "Based on the above information, is the following statement true, false, or uncertain?"
OPTION = re.compile(r"^([A-Z])\)\s*(.+?)\s*$")


def premise_lines(context: str, sentences: list[str]) -> list[str] | None:
    """Split the context into the given premise sentences, longest match first; None if they do not cover it."""
    lines, pos = [], 0
    while True:
        while pos < len(context) and context[pos].isspace():
            pos += 1
        if pos == len(context):
            return lines
        hit = max((s for s in sentences if s and context.startswith(s, pos)), key=len, default=None)
        if hit is None:
            return None
        lines.append(hit)
        pos += len(hit)


def convert(r: dict, level: str, stats: collections.Counter) -> dict | None:
    q = r["question"].strip()
    if not q.startswith(QPREFIX):
        stats["dropped (parse)"] += 1
        return None
    statement = q[len(QPREFIX):].strip()
    options = dict(m.groups() for m in (OPTION.match(o.strip()) for o in r["options"]) if m)
    gold = options.get(r["answer"].strip())
    if not statement or gold not in ANSWERS:
        stats["dropped (parse)"] += 1
        return None
    context = r["context"].strip()
    lines = premise_lines(context, [s.strip() for s in r["nl2fol"]])
    if lines is None:
        stats["context kept as one paragraph"] += 1
        lines = [context]
    item_id = f"proverqa-{level}-{r['id']}"
    order = list(ANSWERS)
    random.Random(f"proverqa-{item_id}").shuffle(order)
    labels = [chr(65 + i) for i in range(len(order))]
    opt_txt = "\n".join(f"{L}) {t}" for L, t in zip(labels, order))
    prem = "\n".join(lines)
    prompt = ("Read the premises and judge the conclusion using only the premises and valid logical reasoning.\n\n"
              f"Premises:\n{prem}\n\nStatement: {statement}\n\n"
              f"Question: Is the statement true, false, or uncertain given the premises?\nOptions:\n{opt_txt}")
    return {"item_id": item_id, "source": "proverqa", "group": f"proverqa/{level}", "type": "choice",
            "prompt": prompt, "labels": labels, "label_order": labels, "gold_label": labels[order.index(gold)],
            "label_names": dict(zip(labels, order)), "gold_probs": None, "depth": len(lines)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW, help="directory with the dev files easy/medium/hard.json")
    ap.add_argument("--out", type=Path, default=ROOT / "data/eval/proverqa.jsonl")
    args = ap.parse_args()
    out = []
    for level in LEVELS:
        rows = sorted(json.load(open(args.raw / f"{level}.json")), key=lambda r: r["id"])
        stats = collections.Counter()
        kept = [it for it in (convert(r, level, stats) for r in rows) if it]
        print(f"{level}: {len(rows)} raw, {len(kept)} kept, {stats['dropped (parse)']} dropped (parse), "
              f"{stats['context kept as one paragraph']} with the context kept as one paragraph")
        out += kept
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for it in out:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(out)} items to {args.out}")


if __name__ == "__main__":
    main()
