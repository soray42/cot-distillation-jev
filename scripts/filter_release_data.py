"""Release training data: a student data directory without the sources the licence audit flags for removal.

The licence audit (data/license/per_source.csv, one row per training source with its licence class and a flag) marks
sources that no release may train on: DROP_BOTH (non-commercial-no-derivatives, withdrawn, request-form or LDC data)
and CONTAM / CONTAM? (benchmarks that Decision Index evaluates, missed by the dataset-name exclusions). Items are
matched on their "source" field. Validation items are kept as they are. A summary with the removed counts per source
and the md5 of each output file is written next to the data.

  python scripts/filter_release_data.py --data data/student_v4tf --out data/student_v4tf_rel
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default="data/license/per_source.csv")
    ap.add_argument("--flags", nargs="+", default=["DROP_BOTH", "CONTAM", "CONTAM?"])
    args = ap.parse_args()
    flagged = {r["source"]: r["flag"] for r in csv.DictReader(open(ROOT / args.sources)) if r["flag"] in args.flags}
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = {"data": args.data, "sources_file": args.sources, "flags": args.flags, "splits": {}}
    for split in ("train", "val"):
        removed, kept = collections.Counter(), 0
        with open(ROOT / args.data / f"{split}.jsonl") as f, open(out / f"{split}.jsonl", "w") as g:
            for line in f:
                src = json.loads(line).get("source") or ""
                if split == "train" and src in flagged:
                    removed[src] += 1
                    continue
                g.write(line)
                kept += 1
        md5 = hashlib.md5((out / f"{split}.jsonl").read_bytes()).hexdigest()
        summary["splits"][split] = {"kept": kept, "removed": sum(removed.values()), "md5": md5,
                                    "removed_by_flag": dict(collections.Counter(flagged[s] for s in removed.elements())),
                                    "removed_by_source": dict(sorted(removed.items()))}
        print(f"{split}: kept {kept}, removed {sum(removed.values())}, md5 {md5}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
