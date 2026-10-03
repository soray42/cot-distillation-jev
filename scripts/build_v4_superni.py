"""v4 instruction-following classification items from Super-NaturalInstructions (allenai/natural-instructions).

Candidates come from data/raw/v4/superni/candidates.json: English tasks with 2 to 10 distinct short outputs, at
least 64 instances, and no source matching the exclusion list. Each task is one sub-task; up to --per-task instances
are drawn. The task's label set becomes the options, shuffled per item; the first listed output is gold.

  python scripts/build_v4_superni.py      # -> data/v4/superni_train.jsonl, data/v4/superni_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=64)
    ap.add_argument("--max-def", type=int, default=1500)
    ap.add_argument("--max-chars", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cands = json.load(open(ROOT / "data/raw/v4/superni/candidates.json"))
    summary = {}
    with open(ROOT / "data/v4/superni_train.jsonl", "w") as f:
        for c in cands:
            t = json.load(open(ROOT / f"data/raw/v4/superni/ni/tasks/{c['task']}.json"))
            definition = " ".join(t["Definition"]).strip()
            if len(definition) > args.max_def:
                summary[c["task"]] = "skipped (definition too long)"
                continue
            labels = c["labels"]
            rng = random.Random(f"{args.seed}-{c['task']}")
            insts = t["Instances"][:]
            rng.shuffle(insts)
            made = 0
            for inst in insts:
                if made >= args.per_task:
                    break
                gold = inst["output"][0].strip()
                if gold not in labels:
                    continue
                opts = labels[:]
                rng.shuffle(opts)
                letters = [chr(65 + i) for i in range(len(opts))]
                prompt = (f"Instruction: {definition}\n\nInput: {inst['input'].strip()}\n\n"
                          f"Question: Which answer does the instruction call for?\nOptions:\n" +
                          "\n".join(f"{L}) {o}" for L, o in zip(letters, opts)))
                if len(prompt) > args.max_chars:
                    continue
                g = letters[opts.index(gold)]
                f.write(json.dumps({
                    "item_id": f"sni-{c['task'].split('_')[0]}-{hashlib.md5(inst['id'].encode()).hexdigest()[:10]}",
                    "source": f"superni/{c['task']}", "family": f"sni/{c['task']}", "type": "choice",
                    "prompt": prompt, "labels": letters, "gold_label": g,
                    "teacher": {L: (1.0 if L == g else 0.0) for L in letters},
                    "subqs": [], "random_subqs": []}) + "\n")
                made += 1
            summary[c["task"]] = made
    (ROOT / "data/v4/superni_summary.json").write_text(json.dumps(summary, indent=1))
    kept = {k: v for k, v in summary.items() if isinstance(v, int) and v >= 32}
    print(f"{len(kept)} tasks with >= 32 items, {sum(v for v in summary.values() if isinstance(v, int))} items")


if __name__ == "__main__":
    main()
