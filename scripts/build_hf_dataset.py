"""Hugging Face dataset of a student training set, split by what the sources' licences allow us to redistribute.

Every row keeps its source; the licence class and licence text come from the release data card (data_card.csv).
- Rows whose class is PERM (permissive, incl. our own generators), ATTR (attribution) or SA (share-alike) are published
  in full, with their source and licence on each row.
- Rows whose class is NC (non-commercial or research-only terms, which often forbid redistribution) or UNK (unknown or
  unverified terms) are withheld: only item_id, source and licence are published, so the set can be rebuilt from the
  original datasets with the repository's builders.
Writes <out>/data/{train,validation}.jsonl, <out>/data/withheld_ids.jsonl, <out>/data_card.csv and a summary.

  python scripts/build_hf_dataset.py --data data/student_v5 --card notes/release/sev-2b-v5/data_card.csv --out hf_data
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

PUBLISH = {"PERM", "ATTR", "SA"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    card = {r["source"]: r for r in csv.DictReader(open(args.card))}
    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    summary = {}
    with open(out / "data/withheld_ids.jsonl", "w") as wh:
        for split, name in (("train", "train"), ("val", "validation")):
            kept, held = collections.Counter(), collections.Counter()
            with open(out / f"data/{name}.jsonl", "w") as f:
                for line in open(Path(args.data) / f"{split}.jsonl"):
                    r = json.loads(line)
                    c = card.get(r["source"])
                    cls = c["class"] if c else "UNK"
                    lic = c["license"] if c else "not in the data card"
                    if cls in PUBLISH:
                        f.write(json.dumps({**r, "license_class": cls, "license": lic}, ensure_ascii=False) + "\n")
                        kept[cls] += 1
                    else:
                        wh.write(json.dumps({"split": name, "item_id": r["item_id"], "source": r["source"],
                                             "license_class": cls, "license": lic}, ensure_ascii=False) + "\n")
                        held[cls] += 1
            summary[name] = {"published": dict(kept), "withheld": dict(held),
                             "published_total": sum(kept.values()), "withheld_total": sum(held.values())}
    (out / "data_card.csv").write_bytes(Path(args.card).read_bytes())
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
