"""External Knights & Knaves test set (Xie et al. 2024, "On Memorization of Large Language Models in Logical
Reasoning", arXiv 2410.23123; HF K-and-K/knights-and-knaves, test split, 100 puzzles per people count 2..8) as a
multiple-choice evaluation file.

The source answer is free-form, so each puzzle becomes a choice item: the options are full role assignments written
like our K&K options ("Name: knight, Name: knave, ...", names in the order the puzzle introduces them); the gold option
is the solution. The structured statements are evaluated by brute force on every assignment: an item is kept only
when the solution is the unique assignment that contradicts no statement, so every distractor is wrong. Distractors
follow the kk_heldout rule (src/cotdistill/sources.py, knights_knaves): near misses in the statement sense, sampled
from the wrong assignments that contradict the fewest statements (at most one more than the minimum), 5 of them
(6 options; 2-person puzzles have only 3 wrong assignments, so 4 options). Near misses by role flips of the solution
would make the solution the per-position majority and Hamming medoid of the options, answerable without the puzzle.
The quiz text is kept verbatim except its closing question, which is replaced by the Question line of
data/eval/kk_heldout.jsonl.

  python3 scripts/build_eval_xkk.py        # -> data/eval/xkk.jsonl
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = Path("/home/soray/.claude/jobs/3b5d797f/tmp/bench/raw/xkk")
PEOPLE = range(2, 9)
N_OPTIONS = 6
TAIL = " So who is a knight and who is a knave?"
QUESTION = "Who is a knight and who is a knave?"
MEET = re.compile(r"You meet \d+ inhabitants: (.*?)\. ")


def holds(st: tuple, roles: tuple[bool, ...]) -> bool:
    """Truth of a structured statement under roles (True = knight)."""
    op = st[0]
    if op == "telling-truth":
        return roles[st[1]]
    if op == "lying":
        return not roles[st[1]]
    if op == "not":
        return not holds(st[1], roles)
    if op == "and":
        return all(holds(s, roles) for s in st[1:])
    if op == "or":
        return any(holds(s, roles) for s in st[1:])
    if op == "->":
        return not holds(st[1], roles) or holds(st[2], roles)
    if op == "<=>":
        return holds(st[1], roles) == holds(st[2], roles)
    raise ValueError(f"unknown operator {op!r}")


def violations(stmts: tuple, roles: tuple[bool, ...]) -> int:
    """Number of speakers whose statement contradicts their role (knights say true statements, knaves false ones)."""
    return sum(holds(st, roles) != roles[i] for i, st in enumerate(stmts))


def distractors(stmts: tuple, gold: tuple[bool, ...], rng: random.Random) -> list[tuple[bool, ...]]:
    """Up to N_OPTIONS - 1 wrong assignments sampled from those contradicting at most min + 1 statements."""
    scored = sorted((violations(stmts, b), b) for b in itertools.product([True, False], repeat=len(gold)) if b != gold)
    pool = [b for c, b in scored if c <= scored[0][0] + 1]
    return rng.sample(pool, min(N_OPTIONS - 1, len(pool)))


def convert(r: dict, p: int) -> dict | None:
    """One raw puzzle -> eval item, or None when it fails a parse or consistency check."""
    names, quiz = r["names"], r["quiz"]
    m = MEET.search(quiz)
    if not quiz.endswith(TAIL) or not m or re.split(r", and |, | and ", m.group(1)) != names or len(names) != p:
        return None
    try:
        stmts = ast.literal_eval(r["statements"])
        gold = tuple(bool(b) for b in r["solution"])
        sols = [b for b in itertools.product([True, False], repeat=p) if violations(stmts, b) == 0]
    except (ValueError, SyntaxError, IndexError, TypeError):
        return None
    if len(stmts) != p or sols != [gold]:
        return None
    item_id = f"xkk-p{p}-{r['index']}"
    rng = random.Random(f"xkk-{item_id}")
    opts = [gold] + distractors(stmts, gold, rng)
    rng.shuffle(opts)
    letters = [chr(65 + i) for i in range(len(opts))]
    text = {L: ", ".join(f"{n}: {'knight' if b else 'knave'}" for n, b in zip(names, o)) for L, o in zip(letters, opts)}
    opt_txt = "\n".join(f"{L}) {text[L]}" for L in letters)
    prompt = f"{quiz[:-len(TAIL)]}\n\nQuestion: {QUESTION}\nOptions:\n{opt_txt}"
    return {"item_id": item_id, "source": "xkk", "group": f"xkk/p{p}", "type": "choice", "prompt": prompt,
            "labels": letters, "label_order": letters, "gold_label": letters[opts.index(gold)], "label_names": text,
            "gold_probs": None, "depth": p}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=ROOT / "data/eval/xkk.jsonl")
    args = ap.parse_args()
    out, dropped = [], 0
    for p in PEOPLE:
        rows = [json.loads(l) for l in open(args.raw / f"people{p}_num100.jsonl") if l.strip()]
        items = [convert(r, p) for r in rows]
        kept = [it for it in items if it is not None]
        dropped += len(items) - len(kept)
        out += kept
        print(f"p={p}: {len(rows)} puzzles, {len(kept)} kept")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for it in out:
            f.write(json.dumps(it) + "\n")
    print(f"{args.out}: {len(out)} items, {dropped} dropped for parse/consistency failures")


if __name__ == "__main__":
    main()
