"""v4 procedural reasoning items from Reasoning Gym (Apache-2.0), converted to 4-option multiple choice.

Run with the reasoning-gym environment (.venv-rg). Generators that overlap an evaluation set are skipped: our own
K&K family, GSM8K-derived, BBH-like (word sorting, boolean circuits, dates and time intervals), Boxoban-like
(sokoban), CRUXEval-like code execution, and the ARC-AGI grid tasks. Each kept generator becomes one sub-task.

Distractors are other items' answers from the same generator, or for numeric answers nearby numbers; every
distractor is checked with the generator's own scorer and kept only if it scores 0, so exactly one option is right.
Answers longer than --max-answer characters are skipped (grids, long lists).

  .venv-rg/bin/python scripts/build_v4_rg.py      # -> data/v4/rg_train.jsonl, data/v4/rg_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"knights_knaves", "gsm_symbolic", "word_sorting", "calendar_arithmetic", "time_intervals", "sokoban",
        "codeio", "circuit_logic", "arc_agi", "rearc", "arc_1d", "composite"}
NUM = re.compile(r"^-?\d+(\.\d+)?$")


def numeric_distractors(ans: str, rng: random.Random) -> list[str]:
    if "." in ans:
        v, d = float(ans), len(ans.split(".")[1])
        cands = {round(v + rng.choice([-1, 1]) * rng.choice([0.5, 1, 2, 10 ** -d, 3 * 10 ** -d]), d) for _ in range(12)}
        return [f"{c:.{d}f}" for c in cands if f"{c:.{d}f}" != ans]
    v = int(ans)
    cands = {v + rng.choice([-1, 1]) * rng.choice([1, 2, 3, 5, 10, max(1, abs(v) // 10)]) for _ in range(12)}
    return [str(c) for c in cands if str(c) != ans]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=64)
    ap.add_argument("--max-answer", type=int, default=60)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import reasoning_gym as rg
    from reasoning_gym.factory import DATASETS

    out_path = ROOT / "data/v4/rg_train.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    with open(out_path, "w") as f:
        for name in sorted(DATASETS):
            if name in SKIP:
                summary[name] = "skipped (overlaps an evaluation set)"
                continue
            try:
                ds = rg.create_dataset(name, size=args.per_task * 3, seed=args.seed)
                entries = [ds[i] for i in range(len(ds))]
            except Exception as e:                       # some generators need extra config
                summary[name] = f"failed: {type(e).__name__}"
                continue
            answers = [str(e["answer"]).strip() for e in entries]
            if sum(len(a) <= args.max_answer for a in answers) < args.per_task or len(set(answers)) < 4:
                summary[name] = "skipped (long or too few distinct answers)"
                continue
            rng = random.Random(f"{args.seed}-{name}")
            made = 0
            for e, ans in zip(entries, answers):
                if made >= args.per_task:
                    break
                if len(ans) > args.max_answer or len(e["question"]) > args.max_chars:
                    continue
                pool = [a for a in set(answers) if a != ans and len(a) <= args.max_answer]
                if NUM.match(ans):
                    pool = numeric_distractors(ans, rng) + pool
                rng.shuffle(pool)
                dis = []
                for d in pool:
                    try:
                        if ds.score_answer(d, e) == 0:
                            dis.append(d)
                    except Exception:
                        continue
                    if len(dis) == 3:
                        break
                if len(dis) < 3:
                    continue
                opts = dis + [ans]
                rng.shuffle(opts)
                letters = ["A", "B", "C", "D"]
                prompt = f"Question: {e['question'].strip()}\nOptions:\n" + "\n".join(
                    f"{L}) {o}" for L, o in zip(letters, opts))
                gold = letters[opts.index(ans)]
                f.write(json.dumps({
                    "item_id": f"rg-{name}-{hashlib.md5(e['question'].encode()).hexdigest()[:10]}",
                    "source": f"reasoning_gym/{name}", "family": f"rg/{name}", "type": "choice",
                    "prompt": prompt, "labels": letters, "gold_label": gold,
                    "teacher": {L: (1.0 if L == gold else 0.0) for L in letters},
                    "subqs": [], "random_subqs": []}) + "\n")
                made += 1
            summary[name] = made
    (ROOT / "data/v4/rg_summary.json").write_text(json.dumps(summary, indent=1))
    kept = {k: v for k, v in summary.items() if isinstance(v, int) and v > 0}
    print(f"{len(kept)} generators, {sum(kept.values())} items; skipped/failed {len(summary) - len(kept)}")


if __name__ == "__main__":
    main()
