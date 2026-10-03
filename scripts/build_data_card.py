"""data_card.csv for a release: one row per training source of a student data directory, with item, twin and tree
counts and the licence reading of the audit.

Licence rows come from data/license/per_source.csv. Sources it does not list are filled from
- the exam-style replay summary (data/v5/replay_mc.summary.json, by_source: origin, effective class, licence), for
  replay/<name> sources;
- our own generators, for claims_mc/* (program-made statements about rule-decision cases).
A source with no licence row at all is an error.

  python scripts/build_data_card.py --data data/student_v5 --out notes/release/sev-2b/data_card.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["source", "dataset", "items", "twins", "tree_items", "class", "flag", "license", "evidence"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default="data/license/per_source.csv")
    ap.add_argument("--replay-summary", default="data/v5/replay_mc.summary.json")
    args = ap.parse_args()
    lic = {r["source"]: r for r in csv.DictReader(open(ROOT / args.sources))}
    replay = json.loads((ROOT / args.replay_summary).read_text())["by_source"] \
        if (ROOT / args.replay_summary).exists() else {}
    items, twins, trees = collections.Counter(), collections.Counter(), collections.Counter()
    for line in open(ROOT / args.data / "train.jsonl"):
        r = json.loads(line)
        s = r["source"]
        items[s] += 1
        twins[s] += r.get("augment") == "claim_twin"
        trees[s] += any("id" in q for q in r.get("subqs") or [])
    rows, missing = [], []
    for s in sorted(items, key=lambda k: (-items[k], k)):
        if s in lic:
            e = lic[s]
            row = {k: e.get(k, "") for k in ("dataset", "class", "flag", "license", "evidence")}
        elif s.startswith("replay/") and s.split("/", 1)[1] in replay:
            e = replay[s.split("/", 1)[1]]
            row = {"dataset": e["origin"], "class": e["effective_class"], "flag": "", "license": e["licence"],
                   "evidence": "replay audit (scripts/build_v5_replay.py)"}
        elif s.startswith("claims_mc/"):
            row = {"dataset": "own", "class": "PERM", "flag": "",
                   "license": "own generator (scripts/build_v5.py, src/cotdistill/rulegen.py)", "evidence": "code"}
        else:
            missing.append(s)
            continue
        rows.append({"source": s, "items": items[s], "twins": twins[s], "tree_items": trees[s], **row})
    if missing:
        raise SystemExit(f"{len(missing)} sources without a licence row, e.g. {missing[:5]}")
    with open(ROOT / args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    by_class, flagged = collections.Counter(), collections.Counter()
    for r in rows:
        by_class[r["class"]] += r["items"]
        if r["flag"]:
            flagged[r["flag"]] += r["items"]
    print(f"{len(rows)} sources, {sum(items.values())} items, {sum(twins.values())} twins, {sum(trees.values())} with a "
          f"tree; by class {dict(by_class.most_common())}; flagged {dict(flagged) or 'none'}")


if __name__ == "__main__":
    main()
