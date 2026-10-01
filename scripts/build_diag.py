"""Locked diagnostic set for Knights & Knaves (collaborator review, E1): each underlying program (statements and the
unique solution) is rendered three semantics-preserving ways plus one solver-validated counterfactual.

  R0  training template ("knight"/"knave", "X says: ...")
  R1  R0 with renamed inhabitants, shuffled statement order and shuffled options
  R2  held-out renderer never used in training ("truth-teller"/"liar", "X claims ...", "A only if B", "A exactly when B")
  C   one inhabitant's statement replaced so that the puzzle still has a unique solution, but a different one (R0 template)

Programs come from seeds 50000+ (disjoint from training 100+n, kk_heldout 9000+n, kk_deep 30000+n). Each record keeps
the program id, the renderer and every inhabitant's true role (meta) for paired and conditional analyses.

  python3 scripts/build_diag.py        # -> data/eval/diag_kk.jsonl
"""
from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill import sources as S  # noqa: E402

RENAME = ["Quinn", "Rosa", "Sam", "Tina", "Umar", "Vera", "Will", "Yara", "Zane", "Ines", "Omar", "Lena", "Hugo",
          "Nina", "Eli", "Cleo"]


def say_r2(st: tuple) -> str:
    """Held-out renderer: the same propositions in different words."""
    if st[0] == "is":
        return f"{st[1]} is a {'truth-teller' if st[2] else 'liar'}"
    a, b = say_r2(st[1]), say_r2(st[2])
    return {"and": f"both {a} and {b}", "or": f"{a}, or {b}, or both", "if": f"{a} only if {b}",
            "iff": f"{a} exactly when {b}"}[st[0]]


def solutions(people: list[str], stmts: dict) -> list[tuple]:
    return [bits for bits in itertools.product([True, False], repeat=len(people))
            if S._consistent(stmts, dict(zip(people, bits))) == 0]


def distractors(people: list[str], stmts: dict, sol: tuple, rng: random.Random, n: int = 5) -> list[tuple]:
    scored = sorted((S._consistent(stmts, dict(zip(people, b))), b)
                    for b in itertools.product([True, False], repeat=len(people)) if b != sol)
    pool = [b for c, b in scored if c <= scored[0][0] + 1] or [b for _, b in scored]
    return rng.sample(pool, min(n, len(pool)))


def render(people: list[str], stmts: dict, sol: tuple, dis: list[tuple], rng: random.Random, style: str,
           order: list[str] | None = None) -> tuple[str, list[str], str]:
    order = order or people
    if style == "r2":
        word = lambda b: "truth-teller" if b else "liar"
        intro = ("Every resident of this island is either a truth-teller, who only makes true statements, or a liar, "
                 f"who only makes false statements. Here are claims made by {len(people)} residents.")
        lines = [f"- {s} claims: \"{say_r2(stmts[s])[0].upper() + say_r2(stmts[s])[1:]}.\"" for s in order]
        q = "Which residents are truth-tellers and which are liars?"
    else:
        word = lambda b: "knight" if b else "knave"
        intro = ("On an island, knights always tell the truth and knaves always lie. "
                 f"Each of the {len(people)} inhabitants below is either a knight or a knave.")
        lines = [f"- {s} says: \"{S._say(stmts[s])[0].upper() + S._say(stmts[s])[1:]}.\"" for s in order]
        q = "Who is a knight and who is a knave?"
    opts = [sol] + dis
    rng.shuffle(opts)
    texts = [", ".join(f"{p}: {word(b)}" for p, b in zip(people, bits)) for bits in opts]
    letters = [chr(65 + i) for i in range(len(opts))]
    prompt = intro + "\n\n" + "\n".join(lines) + f"\n\nQuestion: {q}\nOptions:\n" + \
        "\n".join(f"{L}) {t}" for L, t in zip(letters, texts))
    return prompt, letters, letters[opts.index(sol)]


def rename(stmts: dict, mapping: dict) -> dict:
    def r(st):
        return (st[0], mapping[st[1]], st[2]) if st[0] == "is" else (st[0], r(st[1]), r(st[2]))
    return {mapping[s]: r(st) for s, st in stmts.items()}


def counterfactual(people: list[str], stmts: dict, sol: tuple, rng: random.Random, tries: int = 400):
    for _ in range(tries):
        s = rng.choice(people)
        new = dict(stmts, **{s: S._statement(rng, people, s, 0.7)})
        sols = solutions(people, new)
        if len(sols) == 1 and sols[0] != sol:
            return new, sols[0], s
    return None


def main() -> None:
    out, n_prog = [], 0
    for size in range(5, 10):                          # inside the training range (4-10 inhabitants)
        for r in S.knights_knaves(80, seed=50000 + size, people_range=(size, size)):
            rng = random.Random(f"diag-{r['item_id']}")
            people, stmts = r["meta"]["people"], {k: tuple(v) if not isinstance(v, tuple) else v
                                                  for k, v in r["meta"]["statements"].items()}
            sols = solutions(people, stmts)
            if len(sols) != 1:
                continue
            sol = sols[0]
            cf = counterfactual(people, stmts, sol, rng)
            if cf is None:
                continue
            pid = f"kkdiag-{size}-{n_prog:04d}"
            n_prog += 1
            mapping = dict(zip(people, rng.sample(RENAME, len(people))))
            renamed = [mapping[p] for p in people]
            views = []
            p0 = render(people, stmts, sol, distractors(people, stmts, sol, rng), rng, "r0")
            views.append(("R0", p0, people, sol))
            order = renamed[:]
            rng.shuffle(order)
            st1 = rename(stmts, mapping)
            views.append(("R1", render(renamed, st1, sol, distractors(renamed, st1, sol, rng), rng, "r0", order),
                          renamed, sol))
            views.append(("R2", render(people, stmts, sol, distractors(people, stmts, sol, rng), rng, "r2"), people, sol))
            cst, csol, changed = cf
            views.append(("C", render(people, cst, csol, distractors(people, cst, csol, rng), rng, "r0"), people, csol))
            for tag, (prompt, letters, gold), names, s in views:
                out.append({"item_id": f"{pid}-{tag}", "source": "diag_kk", "group": f"diag/{tag}", "type": "choice",
                            "prompt": prompt, "labels": letters, "label_order": letters, "gold_label": gold,
                            "gold_probs": None, "depth": size,
                            "meta": {"program": pid, "renderer": tag, "people": names,
                                     "roles": {p: bool(b) for p, b in zip(names, s)},
                                     "changed_speaker": changed if tag == "C" else None}})
    path = ROOT / "data/eval/diag_kk.jsonl"
    with open(path, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(f"{path.relative_to(ROOT)}: {len(out)} items from {n_prog} programs")


if __name__ == "__main__":
    main()
