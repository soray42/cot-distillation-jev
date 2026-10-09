"""Stratified fixed-seed subset of an eval file, for cheap mid-training tracking (train_student.py --eval-mid).

Equal allocation over the strata (the item field --by, default "group": the 23 BBH tasks): every stratum gets
n // strata items and n % strata strata (picked by the seeded shuffle) one more; a stratum smaller than its share gives
the rest to the others (water filling). Items inside a stratum are a seeded random sample; records are written
unchanged, in the source file's order. Equal allocation makes the plain item accuracy of the subset an estimate of the
macro accuracy over strata (the BBH number of ladder_readout / analyze_e2). CPU only, no dependencies.

  python scripts/eval_subset.py data/eval/bbh.jsonl data/eval/bbh_sub500.jsonl --n 500 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def allocate(sizes: dict[str, int], n: int, rng: random.Random) -> dict[str, int]:
    """Equal allocation of n over strata capped at their sizes; the remainder to seeded-shuffled strata."""
    if n > sum(sizes.values()):
        raise ValueError(f"asked for {n} items, the strata hold {sum(sizes.values())}")
    take = {g: 0 for g in sizes}
    order = sorted(sizes)
    rng.shuffle(order)
    left = n
    while left:
        open_ = [g for g in order if take[g] < sizes[g]]
        share = max(1, left // len(open_))
        for g in open_:
            k = min(share, sizes[g] - take[g], left)
            take[g] += k
            left -= k
            if not left:
                break
    return take


def subset(items: list[dict], n: int, seed: int, by: str = "group") -> list[dict]:
    rng = random.Random(seed)
    strata = defaultdict(list)
    for j, it in enumerate(items):
        strata[str(it.get(by))].append(j)
    take = allocate({g: len(v) for g, v in strata.items()}, n, rng)
    keep = set()
    for g in sorted(strata):
        keep.update(rng.sample(strata[g], take[g]))
    return [it for j, it in enumerate(items) if j in keep]


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--by", default="group", help="item field that defines the strata")
    args = ap.parse_args(argv)
    with open(args.src) as f:
        items = [json.loads(line) for line in f if line.strip()]
    out = subset(items, args.n, args.seed, args.by)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(it) + "\n" for it in out))
    per = Counter(str(it.get(args.by)) for it in out)
    print(f"{args.out}: {len(out)} of {len(items)} items, {len(per)} strata, {min(per.values())}-{max(per.values())} "
          f"per stratum (seed {args.seed})")
    return out


if __name__ == "__main__":
    main()
