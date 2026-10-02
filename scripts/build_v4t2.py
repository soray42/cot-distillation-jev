"""TF-v4t2 data: data/student_v4t with the v3 reasoning families (K&K, JustLogic, ShARC, policy) upsampled x2.

Rationale: in TF-c4 (v3 only) K&K kept improving from 2 to 4 epochs (diag A .51 -> .61) while broad benchmarks fell, so the
reasoning families get a second copy (distinct item ids, `#2`) and everything else stays at one copy, i.e. 4 passes for
the reasoning items within a 2-epoch run. Kept ready in case TF-v4t is weak on K&K.

  python scripts/build_v4t2.py      # -> data/student_v4t2/{train,val}.jsonl
"""
from __future__ import annotations

import collections
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = {"knights_knaves", "justlogic", "sharc", "returns", "expense", "subscription"}

rows = [json.loads(l) for l in open(ROOT / "data/student_v4t/train.jsonl")]
extra = [dict(r, item_id=r["item_id"] + "#2") for r in rows if r.get("source") in CORE]
out = rows + extra
random.Random("v4t2").shuffle(out)
d = ROOT / "data/student_v4t2"
d.mkdir(parents=True, exist_ok=True)
with open(d / "train.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
shutil.copy(ROOT / "data/student_v4t/val.jsonl", d / "val.jsonl")
print(f"{len(rows)} + {len(extra)} upsampled = {len(out)} items;", dict(collections.Counter(r["source"] for r in extra)))
