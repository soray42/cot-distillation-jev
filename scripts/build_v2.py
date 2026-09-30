"""Build the v2 student data: every solved item gives a final-question example to every arm; items whose
reasoning tree is ready also carry CoT sub-questions (and matched controls). Augmentations from the v1 error
analysis (hedge-option bias, prose-only states):
- K&K and ARC items: a "None of the above" option is added as a wrong option (8%) or replaces the gold option
  as the right answer (2%), so a hedge option is not a safe default;
- policy-generator items: half render the case facts as a JSON object.

  python3 scripts/build_v2.py --out data/student_v2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from build_student_data import convert  # noqa: E402

RUNS = ["label_kk_v1", "label_jl_v1", "label_policy_v1", "label_sharc_v1", "label_folio_v1",
        "label_sharc2_v1", "label_policy2_v1", "label_arc_v1"]
NONE_TEXT = "None of the above"
OPT = re.compile(r"^([A-Z])\) (.*)$")


def _split_options(prompt: str) -> tuple[str, list[tuple[str, str]]] | None:
    head, sep, tail = prompt.rpartition("\nOptions:\n")
    if not sep:
        return None
    opts = [OPT.match(l) for l in tail.splitlines()]
    if not opts or not all(opts):
        return None
    return head, [(m.group(1), m.group(2)) for m in opts]


def _join(head: str, texts: list[str]) -> tuple[str, list[str]]:
    letters = [chr(65 + i) for i in range(len(texts))]
    return head + "\nOptions:\n" + "\n".join(f"{L}) {t}" for L, t in zip(letters, texts)), letters


def inject_none(rec: dict, rng: random.Random, p_wrong: float = 0.08, p_right: float = 0.02) -> str | None:
    parts = _split_options(rec["prompt"])
    if parts is None or not rec.get("teacher"):
        return None
    head, opts = parts
    u = rng.random()
    if u < p_wrong:                                   # extra wrong option, target mass 0
        texts = [t for _, t in opts] + [NONE_TEXT]
        rec["prompt"], rec["labels"] = _join(head, texts)
        rec["teacher"] = dict(rec["teacher"], **{rec["labels"][-1]: 0.0})
        return "none_wrong"
    if u < p_wrong + p_right and rec.get("gold_label") in dict(opts):  # "None of the above" becomes right
        gold = rec["gold_label"]
        keep = [(L, t) for L, t in opts if L != gold]
        texts = [t for _, t in keep] + [NONE_TEXT]
        new_prompt, letters = _join(head, texts)
        teacher = {letters[i]: rec["teacher"].get(L, 0.0) for i, (L, _) in enumerate(keep)}
        teacher[letters[-1]] = rec["teacher"].get(gold, 0.0)
        rec.update(prompt=new_prompt, labels=letters, teacher=teacher, gold_label=letters[-1])
        return "none_right"
    return None


def jsonify_case(rec: dict) -> bool:
    m = re.search(r"\nCase:\n((?:- [^\n]+\n?)+)", rec["prompt"])
    if not m:
        return False
    facts = {}
    for line in m.group(1).strip().splitlines():
        k, _, v = line[2:].partition(": ")
        v = v.strip()
        facts[re.sub(r"[^a-z0-9]+", "_", k.lower()).strip("_")] = None if v.startswith("unknown") else v
    rec["prompt"] = rec["prompt"].replace(m.group(0), "\nCase (JSON; null = not stated):\n" + json.dumps(facts) + "\n")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/student_v2")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    splits, stats = {"train": [], "val": []}, {}
    for run in RUNS:
        tree_dir, solve_dir = ROOT / "teacher_cache" / f"{run}_tree", ROOT / "teacher_cache" / run
        for p in sorted(solve_dir.glob("*.json")):
            if p.name in ("summary.json", "calls.jsonl") or not p.name[0].isalpha():
                continue
            src = tree_dir / p.name if (tree_dir / p.name).exists() else p
            res = json.loads(src.read_text())
            if "item" not in res or not res.get("traces"):
                continue
            if src == p:                                   # no tree yet: final question only (shared by all arms)
                res = dict(res, subquestions=[], random_subquestions=[])
            rec = convert(res)
            rng = random.Random(f"{args.seed}-{rec['item_id']}")
            dom = res["item"].get("domain", "")
            aug = None
            if dom == "knights_knaves" or dom.startswith("arc"):
                aug = inject_none(rec, rng)
            elif dom in ("returns", "subscription", "expense") and rng.random() < 0.5:
                aug = "json" if jsonify_case(rec) else None
            rec["augment"] = aug
            rec["has_tree"] = src != p
            key = f"{run}:{dom}"
            st = stats.setdefault(key, {"items": 0, "tree": 0, "subq": 0, "aug": {}})
            st["items"] += 1; st["tree"] += rec["has_tree"]; st["subq"] += len(rec["subqs"])
            if aug:
                st["aug"][aug] = st["aug"].get(aug, 0) + 1
            h = int(hashlib.sha1(rec["item_id"].encode()).hexdigest(), 16) % 1000 / 1000
            splits["val" if h < args.val_frac else "train"].append(rec)
    for name, recs in splits.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(recs)} items, {sum(len(r['subqs']) for r in recs)} sub-questions, "
              f"{sum(len(r['random_subqs']) for r in recs)} controls")
    for k, v in stats.items():
        print(f"  {k:36s} {v}")


if __name__ == "__main__":
    main()
