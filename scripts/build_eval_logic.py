"""Three multiple-choice logical-reading sets as evaluation files: LogiQA 2.0 English test, AGIEval LSAT-LR and LSAT-AR.

LogiQA 2.0 (Liu et al., IEEE/ACM TASLP 2023; github.com/csitfun/LogiQA2.0, logiqa/DATA/LOGIQA/test.txt, 1572 items)
is the English translation of Chinese civil-service exam questions: fields text, question, four options without letter
prefixes, answer = option index. AGIEval v1.1 (Zhong et al. 2023; github.com/ruixiangcui/AGIEval, data/v1_1/
lsat-lr.jsonl, 510 items, and lsat-ar.jsonl, 230 items) holds LSAT logical-reasoning and analytical-reasoning
questions: fields passage, question, five options prefixed "(A)".."(E)", label = gold letter. Each item is rendered
as "<passage>\n\nQuestion: <question>\nOptions:\nA) ...", options in a seeded random order with the original letter
prefixes stripped and whitespace runs collapsed.

Item ids are "<set>-<raw line index>" (the LogiQA id field is not unique in the test file). Per set, counted:
- dropped, parse failure: missing field, option prefix not the expected "(X)", gold out of range;
- dropped, ambiguous gold: the gold text occurs twice among the options;
- collapsed: two options have the same text but the gold is not one of them (translation duplicates in LogiQA); the
  repeated copy is removed, so the item keeps one option fewer;
- dropped, repeat: same passage, question and gold text as an earlier item (exact repeats, and repeats with one
  distractor changed); the first one is kept;
- dropped, seen setup: the passage shares a 13-gram with a training prompt (--train). LSAT-AR asks 4 to 7 questions
  about each of its 40 game setups, so a setup's 13-grams occur in 5 or more items and the per-item overlap rule
  (scripts/eval_overlap.py) ignores them as boilerplate; a model trained on other questions about the same game would
  then be scored on it. Here 13-grams are taken from the passage only and a 13-gram counts as boilerplate when it
  occurs in at least --boilerplate distinct passages of the set. Every item of a matched passage is dropped. The
  registered per-item rule is still run on the outputs afterwards.

  python3 scripts/build_eval_logic.py --record setup_overlap.json   # -> data/eval/{logiqa2,lsat_lr,lsat_ar}.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_overlap import grams  # noqa: E402

RAW = Path("/home/soray/.claude/jobs/3b5d797f/tmp/bench/raw/logic")
SETS = {"logiqa2": "logiqa2_test.txt", "lsat_lr": "lsat-lr.jsonl", "lsat_ar": "lsat-ar.jsonl"}
TRAIN = [ROOT / "data/student_v4t/train.jsonl", ROOT / "data/student_v4tf/train.jsonl",
         ROOT / "data/student_v3f/train.jsonl"]
PREFIX = re.compile(r"^\(([A-E])\)\s*")
SEP = "\n\nQuestion: "


def clean(s: object) -> str:
    return " ".join(str(s).split()) if isinstance(s, str) else ""


def parse(key: str, r: dict) -> tuple[str, str, list[str], int] | None:
    """(passage, question, options, gold index) of a raw record, or None if it does not parse."""
    if key == "logiqa2":
        passage, options, gold = r.get("text"), [clean(o) for o in r.get("options") or []], r.get("answer")
    else:
        passage, options = r.get("passage"), []
        for i, o in enumerate(r.get("options") or []):
            m = PREFIX.match(o.strip()) if isinstance(o, str) else None
            if m is None or m.group(1) != chr(65 + i):
                return None
            options.append(clean(o.strip()[m.end():]))
        gold = ord(r["label"]) - 65 if isinstance(r.get("label"), str) and len(r["label"]) == 1 else None
    passage, question = clean(passage), clean(r.get("question"))
    if not passage or not question or len(options) < 2 or not all(options) \
            or not isinstance(gold, int) or not 0 <= gold < len(options):
        return None
    return passage, question, options, gold


def convert(key: str, rows: list[dict], stats: collections.Counter) -> list[dict]:
    out, seen = [], set()
    for idx, r in enumerate(rows):
        parsed = parse(key, r)
        if parsed is None:
            stats["parse"] += 1
            continue
        passage, question, options, gold = parsed
        gold_text = options[gold]
        if options.count(gold_text) > 1:
            stats["ambiguous gold"] += 1
            continue
        if len(set(options)) < len(options):
            stats["collapsed"] += 1
            options = list(dict.fromkeys(options))
            gold = options.index(gold_text)
        sig = (passage, question, gold_text)
        if sig in seen:
            stats["repeat"] += 1
            continue
        seen.add(sig)
        item_id = f"{key}-{idx}"
        order = list(range(len(options)))
        random.Random(f"{key}-{item_id}").shuffle(order)
        labels = [chr(65 + i) for i in range(len(options))]
        names = {L: options[j] for L, j in zip(labels, order)}
        opt_txt = "\n".join(f"{L}) {t}" for L, t in names.items())
        out.append({"item_id": item_id, "source": key, "group": key, "type": "choice",
                    "prompt": f"{passage}{SEP}{question}\nOptions:\n{opt_txt}",
                    "labels": labels, "label_order": labels, "gold_label": labels[order.index(gold)],
                    "label_names": names, "gold_probs": None})
    return out


def setup(item: dict) -> str:
    """The passage of a rendered item."""
    return item["prompt"].split(SEP)[0]


def seen_setups(built: dict[str, list[dict]], train: list[Path], boilerplate: int) -> dict[str, str]:
    """Passage -> "<train dir>:<item_id>" of the first training prompt that shares a non-boilerplate passage 13-gram."""
    owner: dict[int, set[str]] = collections.defaultdict(set)
    for items in built.values():
        per = {p: grams(p) for p in {setup(it) for it in items}}
        freq = collections.Counter(g for gs in per.values() for g in gs)
        for p, gs in per.items():
            for g in gs:
                if freq[g] < boilerplate:
                    owner[g].add(p)
    hit: dict[str, str] = {}
    for tf in train:
        for line in open(tf):
            r = json.loads(line)
            for g in grams(r.get("prompt", "")):
                for p in owner.get(g, ()):
                    hit.setdefault(p, f"{tf.parent.name}:{r.get('item_id')}")
    return hit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW,
                    help="directory with logiqa2_test.txt (LogiQA 2.0 test.txt), lsat-lr.jsonl and lsat-ar.jsonl")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/eval")
    ap.add_argument("--train", type=Path, nargs="*", default=TRAIN,
                    help="training files whose prompts exclude a passage (setup-level overlap)")
    ap.add_argument("--boilerplate", type=int, default=5)
    ap.add_argument("--record", type=Path, help="JSON file for the ids dropped as seen setups and their matches")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    built, stats = {}, {}
    for key, name in SETS.items():
        rows = [json.loads(l) for l in open(args.raw / name) if l.strip()]
        stats[key] = collections.Counter(raw=len(rows))
        built[key] = convert(key, rows, stats[key])
    hit = seen_setups(built, args.train, args.boilerplate)
    record = {}
    for key, items in built.items():
        s = stats[key]
        record[key] = {it["item_id"]: hit[setup(it)] for it in items if setup(it) in hit}
        keep = [it for it in items if it["item_id"] not in record[key]]
        n_setups, n_seen = len({setup(it) for it in items}), len({setup(it) for it in items if setup(it) in hit})
        gold = collections.Counter(it["gold_label"] for it in keep)
        print(f"{key}: {s['raw']} raw, {len(keep)} kept; dropped: {s['parse']} parse failure, {s['ambiguous gold']} "
              f"ambiguous gold, {s['repeat']} repeat, {len(record[key])} seen setup ({n_seen} of {n_setups} "
              f"passages); {s['collapsed']} with a repeated option collapsed; gold {dict(sorted(gold.items()))}")
        path = args.out_dir / f"{key}.jsonl"
        with open(path, "w") as f:
            for it in keep:
                f.write(json.dumps(it) + "\n")
        print(f"  wrote {path}")
    if args.record:
        args.record.write_text(json.dumps(record, indent=1))
        print(f"seen-setup drops in {args.record}")


if __name__ == "__main__":
    main()
