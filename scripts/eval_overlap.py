"""13-gram overlap of evaluation items with training files (the reverse of scripts/decontam.py, which cleans training
files): an evaluation item overlaps when it shares a 13-gram with any training prompt, after ignoring 13-grams that
occur in at least --boilerplate distinct passages of the evaluation file itself (a benchmark's fixed preamble; a passage
asked several times, such as a logic-grid puzzle or an LSAT game setup, counts once). Only the evaluation
13-grams are held in memory; training files are streamed. With --drop the evaluation file is rewritten without the
overlapping items, and their ids go to <file>.overlap.json.

  python scripts/eval_overlap.py data/eval/xkk.jsonl --train data/student_v4t/train.jsonl \\
      data/student_v4tf/train.jsonl data/student_v3f/train.jsonl --drop
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

TOK = re.compile(r"[a-z0-9]+")
N = 13


def grams(text: str) -> set[int]:
    w = TOK.findall(text.lower())
    return {hash(tuple(w[i:i + N])) for i in range(len(w) - N + 1)}


def head(prompt: str) -> str:
    """The passage of an item: the prompt before its last question line (or before the options)."""
    body = prompt.rpartition("\nOptions:\n")[0] or prompt
    return body.rpartition("\n\nQuestion:")[0] or body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval", nargs="+")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--boilerplate", type=int, default=5)
    ap.add_argument("--drop", action="store_true")
    args = ap.parse_args()
    for path in args.eval:
        items = [json.loads(l) for l in open(path) if l.strip()]
        per = [grams(it["prompt"]) for it in items]
        # boilerplate is counted over distinct passages, not items: a puzzle or game setup asked several times is
        # one passage, so its text is compared with training instead of being taken for a shared template
        heads = {}
        for it, gs in zip(items, per):
            heads.setdefault(head(it["prompt"]), set()).update(gs)
        freq = collections.Counter(g for gs in heads.values() for g in gs)
        owner = collections.defaultdict(set)
        for i, gs in enumerate(per):
            for g in gs:
                if freq[g] < args.boilerplate:
                    owner[g].add(i)
        hit: dict[int, str] = {}
        for tf in args.train:
            for line in open(tf):
                r = json.loads(line)
                for g in grams(r.get("prompt", "")):
                    for i in owner.get(g, ()):
                        hit.setdefault(i, f"{Path(tf).parent.name}:{r.get('item_id')}")
        ids = [items[i]["item_id"] for i in sorted(hit)]
        print(f"{path}: {len(items)} items, {len(ids)} overlap a training prompt")
        if args.drop:
            keep = [it for i, it in enumerate(items) if i not in hit]
            with open(path, "w") as f:
                for it in keep:
                    f.write(json.dumps(it) + "\n")
            Path(path + ".overlap.json").write_text(json.dumps({"dropped": ids, "match": {
                items[i]["item_id"]: m for i, m in hit.items()}}, indent=1))
            print(f"  kept {len(keep)}; dropped ids in {path}.overlap.json")


if __name__ == "__main__":
    main()
