"""Depth-extrapolation test set for Knights & Knaves (program gold, no teacher calls).

Training covers 4-10 inhabitants and kk_heldout 4-12 (50 per size). This set has 4-10 (100 per size, in range),
11-16 (150 per size, beyond the training range, same name pool) and 17-20 (100 per size, which needs 8 names
never used in training; group "kkx/..." so they are reported apart). Seeds 30000+n are disjoint from training
(100+n) and kk_heldout (9000+n).

  python3 scripts/build_kk_deep.py        # -> data/eval/kk_deep.jsonl
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cotdistill import sources as S  # noqa: E402
from build_items import to_eval  # noqa: E402

EXTRA_NAMES = ["Quinn", "Rosa", "Sam", "Tina", "Umar", "Vera", "Will", "Yara"]


def main() -> None:
    train_prompts = {hashlib.sha1(json.loads(l)["prompt"].encode()).hexdigest() for l in open(ROOT / "data/bench/train_kk.jsonl")}
    out = []
    for n in range(4, 21):
        count = 100 if n <= 10 or n > 16 else 150
        names = None if n <= 16 else S.NAMES + EXTRA_NAMES
        recs = S.knights_knaves(count, seed=30000 + n, people_range=(n, n), names=names)
        for r in recs:
            r["item_id"] = r["item_id"].replace("kk-", "kkdeep-")
            e = to_eval(r, "kk")
            if n > 16:
                e["group"] = f"kkx/d{n}"
            if hashlib.sha1(e["prompt"].encode()).hexdigest() in train_prompts:
                continue
            out.append(e)
        print(f"n={n}: {count} items", flush=True)
    path = ROOT / "data/eval/kk_deep.jsonl"
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"{path.relative_to(ROOT)}: {len(out)}")


if __name__ == "__main__":
    main()
