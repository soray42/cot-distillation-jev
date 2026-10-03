"""v4 public-source items from tasksource-jev-typed-decisions (670 sources, Jev typed-decision format).

Every source whose name matches an exclusion pattern (data/raw/v4/exclusions.json: all Decision Index datasets
including ARC, our own and other boards' evaluation sets) is dropped. From each remaining source, up to --per-source
rows are drawn by reservoir sampling (reproducible, one pass over the parquet shards). A fraction of the sources is
held out as unseen families for checkpoint selection and per-family calibration.

Rendering follows the typed-decision evaluation layout ("State:", question, lettered options):
- choice / score: the source's options in their given order;
- noul (yes/no): "Yes" and "No" in random order, so no letter carries a fixed meaning; the target is P(yes).

Targets are the source's label distribution (annotator votes where it has them), stored as the "teacher" field the
student code reads. Items carry no tree nodes.

  python scripts/build_v4_tasksource.py      # -> data/v4/tasksource_{train,heldout}.jsonl
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def excluded(source: str, patterns: list[str]) -> list[str]:
    s = source.lower()
    return [p for p in patterns if p in s]


def render(row: dict, rng: random.Random) -> dict | None:
    state, question, kind = row["state"], (row["question"] or "").strip(), row["kind"]
    target = list(row["target"] or [])
    if kind == "noul":
        if len(target) != 1:
            return None
        p_yes = min(max(float(target[0]), 0.0), 1.0)
        names = ["Yes", "No"] if rng.random() < 0.5 else ["No", "Yes"]
        probs = [p_yes if n == "Yes" else 1 - p_yes for n in names]
    else:
        names = [str(o) for o in (row["options"] or [])]
        if len(names) < 2 or len(names) > 10 or len(target) != len(names):
            return None
        z = sum(target)
        if z <= 0:
            return None
        probs = [t / z for t in target]
    letters = [chr(65 + i) for i in range(len(names))]
    prompt = f"State:\n{state}\n\nQuestion: {question}\nOptions:\n" + "\n".join(
        f"{L}) {n}" for L, n in zip(letters, names))
    gold = letters[max(range(len(probs)), key=probs.__getitem__)]
    return {"prompt": prompt, "labels": letters, "gold_label": gold, "label_names": dict(zip(letters, names)),
            "teacher": dict(zip(letters, probs)), "type": kind}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=64)
    ap.add_argument("--heldout-frac", type=float, default=0.05)
    ap.add_argument("--max-chars", type=int, default=5000, help="drop rows whose rendered prompt is longer")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import pyarrow.parquet as pq

    patterns = [p.lower() for p in json.load(open(ROOT / "data/raw/v4/exclusions.json"))["name_patterns"]]
    rng = random.Random(args.seed)
    keep: dict[str, list[dict]] = {}
    seen: dict[str, int] = {}
    dropped: dict[str, list[str]] = {}
    cols = ["state", "kind", "id", "options", "target", "question", "source", "variant", "license_use"]
    for f in sorted(glob.glob(str(ROOT / "data/raw/v4/tasksource_jev/data/train-*.parquet"))):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=20000, columns=cols):
            for row in batch.to_pylist():
                src = row["source"]
                if src in dropped:
                    continue
                why = excluded(src, patterns)
                if why:
                    dropped[src] = why
                    continue
                if len(row["state"] or "") + len(row["question"] or "") > args.max_chars - 300:
                    continue                                  # too long for the 1536-token training window
                n = seen.get(src, 0) + 1
                seen[src] = n
                res = keep.setdefault(src, [])
                if len(res) < args.per_source * 2:          # oversample: some rows fail to render
                    res.append(row)
                else:
                    j = rng.randrange(n)
                    if j < len(res):
                        res[j] = row
    sources = sorted(keep)
    held = {s for s in sources if int(hashlib.md5(s.encode()).hexdigest(), 16) % 10000 < args.heldout_frac * 10000}
    out_dir = ROOT / "data/v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_items = {"train": 0, "heldout": 0}
    with open(out_dir / "tasksource_train.jsonl", "w") as ft, open(out_dir / "tasksource_heldout.jsonl", "w") as fh:
        for src in sources:
            r2 = random.Random(f"{args.seed}-{src}")
            rows = keep[src][:]
            r2.shuffle(rows)
            made = 0
            for row in rows:
                if made >= args.per_source:
                    break
                rec = render(row, r2)
                if rec is None or len(rec["prompt"]) > args.max_chars:
                    continue
                made += 1
                split = "heldout" if src in held else "train"
                rec.update({"item_id": f"ts-{hashlib.md5((src + str(row['id'])).encode()).hexdigest()[:12]}",
                            "source": f"tasksource/{src}", "family": src, "variant": row["variant"],
                            "license_use": row["license_use"], "subqs": [], "random_subqs": []})
                (fh if split == "heldout" else ft).write(json.dumps(rec) + "\n")
                n_items[split] += 1
    summary = {"sources_kept": len(sources), "sources_heldout": len(held), "sources_dropped": len(dropped),
               "items": n_items, "dropped": dropped, "heldout_sources": sorted(held)}
    (out_dir / "tasksource_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("dropped", "heldout_sources")}))


if __name__ == "__main__":
    main()
