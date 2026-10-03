"""13-gram decontamination of training files against every evaluation file.

Text is lower-cased and split into word tokens. Evaluation 13-grams that occur in at least --boilerplate evaluation
items (shared instructions and templates, e.g. a benchmark's fixed preamble) are ignored. A training item is
contaminated if it shares any remaining 13-gram with any evaluation item; contaminated items are written to
<file>.contaminated.jsonl with the matching evaluation file, and the clean items to <file>.clean.jsonl.

  python scripts/decontam.py data/v4/tasksource_train.jsonl data/v4/superni_train.jsonl ...
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOK = re.compile(r"[a-z0-9]+")
N = 13


def grams(text: str) -> set[int]:
    w = TOK.findall(text.lower())
    return {hash(tuple(w[i:i + N])) for i in range(len(w) - N + 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("train", nargs="+")
    ap.add_argument("--evals", default="data/eval/*.jsonl")
    ap.add_argument("--boilerplate", type=int, default=20)
    args = ap.parse_args()
    df: collections.Counter = collections.Counter()
    owner: dict[int, str] = {}
    for path in sorted(glob.glob(str(ROOT / args.evals))):
        name = Path(path).stem
        for line in open(path):
            if not line.strip():
                continue
            g = grams(json.loads(line).get("prompt", ""))
            df.update(g)
            for h in g:
                owner.setdefault(h, name)
    ok = {h for h, c in df.items() if c < args.boilerplate}
    print(f"evaluation 13-grams: {len(df)}, kept after dropping boilerplate: {len(ok)}")
    del df
    for path in args.train:
        p = Path(path)
        clean = open(p.with_suffix(".clean.jsonl"), "w")
        bad = open(p.with_suffix(".contaminated.jsonl"), "w")
        n = nb = 0
        hits: collections.Counter = collections.Counter()
        for line in open(p):
            r = json.loads(line)
            n += 1
            m = grams(r["prompt"]) & ok
            if m:
                nb += 1
                src = owner[next(iter(m))]
                hits[src] += 1
                bad.write(json.dumps({**r, "contaminated_with": src}) + "\n")
            else:
                clean.write(line)
        clean.close(); bad.close()
        print(f"{p.name}: {nb}/{n} contaminated ({nb / max(n, 1):.2%}) by {dict(hits.most_common(6))}")


if __name__ == "__main__":
    main()
