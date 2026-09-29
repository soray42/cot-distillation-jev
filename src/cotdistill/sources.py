"""Hard training sources turned into typed decision records (same schema as policygen).

- knights_knaves: generated Knights & Knaves puzzles (knights always tell the truth, knaves always
  lie) with a unique solution; Choice over full role assignments (the solution plus near-miss
  distractors). Per-person roles are program-verifiable predicates.
- justlogic: JustLogic (MIT) items as True / False / Uncertain choices, with their depth.
"""
from __future__ import annotations

import ast
import csv
import itertools
import random
import re

NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Irene", "Jack",
         "Karen", "Leo", "Mia", "Noah", "Olivia", "Paul"]


# ---------------------------------------------------------------- Knights & Knaves
def _atom(rng: random.Random, people: list[str]) -> tuple:
    return ("is", rng.choice(people), rng.random() < 0.5)          # (is, name, knight?)


def _statement(rng: random.Random, people: list[str], speaker: str, p_compound: float) -> tuple:
    others = [p for p in people if p != speaker] or people
    if rng.random() >= p_compound:
        return _atom(rng, others)
    op = rng.choice(["and", "or", "if", "iff"])
    a, b = _atom(rng, others), _atom(rng, others)
    while b[1] == a[1] and len(others) > 1:
        b = _atom(rng, others)
    return (op, a, b)


def _eval(st: tuple, roles: dict[str, bool]) -> bool:
    if st[0] == "is":
        return roles[st[1]] == st[2]
    a, b = _eval(st[1], roles), _eval(st[2], roles)
    return {"and": a and b, "or": a or b, "if": (not a) or b, "iff": a == b}[st[0]]


def _say(st: tuple) -> str:
    if st[0] == "is":
        return f"{st[1]} is a {'knight' if st[2] else 'knave'}"
    a, b = _say(st[1]), _say(st[2])
    return {"and": f"{a} and {b}", "or": f"{a} or {b}", "if": f"if {a}, then {b}",
            "iff": f"{a} if and only if {b}"}[st[0]]


def _consistent(stmts: dict[str, tuple], roles: dict[str, bool]) -> int:
    """Number of speakers whose statement is inconsistent with their role (0 = a solution)."""
    return sum(_eval(st, roles) != roles[s] for s, st in stmts.items())


def knights_knaves(n: int, seed: int = 0, people_range: tuple[int, int] = (8, 10), p_compound: float = 0.7,
                   n_options: int = 6, max_tries: int = 5000, question: str = "assignment") -> list[dict]:
    """question="assignment": choose the full role assignment (solution + near-miss distractors).
    question="count": how many inhabitants are knights (options 0..k) - no option can be checked
    against the statements without solving the whole puzzle."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        k = rng.randint(*people_range)
        people = rng.sample(NAMES, k)
        for _ in range(max_tries):
            truth = {p: rng.random() < 0.5 for p in people}
            stmts = {}
            for s in people:
                for _ in range(50):
                    st = _statement(rng, people, s, p_compound)
                    if _eval(st, truth) == truth[s]:
                        stmts[s] = st
                        break
            if len(stmts) < k:
                continue
            scored = []
            for bits in itertools.product([True, False], repeat=k):
                roles = dict(zip(people, bits))
                scored.append((_consistent(stmts, roles), bits))
            sols = [b for c, b in scored if c == 0]
            if len(sols) != 1:
                continue
            # distractors: assignments violating the fewest statements (near misses)
            near = sorted((c, b) for c, b in scored if c > 0)
            pool = [b for c, b in near if c <= near[0][0] + 1] or [b for _, b in near]
            distract = rng.sample(pool, min(n_options - 1, len(pool)))
            opts = [sols[0]] + distract
            rng.shuffle(opts)
            keys = [f"o{i}" for i in range(len(opts))]
            fmt = lambda bits: ", ".join(f"{p}: {'knight' if b else 'knave'}" for p, b in zip(people, bits))
            options = {key: fmt(b) for key, b in zip(keys, opts)}
            gold = keys[opts.index(sols[0])]
            lines = [f"- {s} says: \"{_say(stmts[s])[0].upper() + _say(stmts[s])[1:]}.\"" for s in people]
            problem = ("On an island, knights always tell the truth and knaves always lie. "
                       f"Each of the {k} inhabitants below is either a knight or a knave.\n\n" + "\n".join(lines))
            if question == "count":
                options = {f"c{c}": str(c) for c in range(k + 1)}
                gold = f"c{sum(sols[0])}"
                label_order = list(options)                 # counts stay in numeric order
                qtext = "How many of the inhabitants are knights?"
            else:
                label_order = keys[:]
                rng.shuffle(label_order)
                qtext = "Who is a knight and who is a knave?"
            opt_txt = "\n".join(f"{chr(65 + i)}) {options[key]}" for i, key in enumerate(label_order))
            prompt = f"{problem}\n\nQuestion: {qtext}\nOptions:\n{opt_txt}"
            preds = [{"pid": f"knight_{p}", "question": f"Is {p} a knight?", "truth": truth[p], "kind": "lookup"}
                     for p in people]
            out.append({"item_id": f"kk{'c' if question == 'count' else ''}-{seed}-{len(out):05d}",
                        "domain": "knights_knaves", "hidden": None,
                        "gold": gold, "label_order": label_order, "gold_label": chr(65 + label_order.index(gold)),
                        "depth": k, "n_rules": k, "path": preds, "predicates": preds, "prompt": prompt,
                        "options": options,
                        "meta": {"n_people": k, "n_compound": sum(st[0] != "is" for st in stmts.values()),
                                 "statements": stmts, "people": people}})
            break
    return out


_KK_ONE = re.compile(r"^(?:is|was) (\w+) (?:a|actually a|really a) (knight|knave)\??$", re.I)
_KK_STMT = re.compile(r"^(?:is|was) (\w+)'s statement (true|false)\??$", re.I)
_KK_ALL = re.compile(r"^are ((?:\w+, )*\w+,? and \w+|\w+ and \w+) (?:all |both )?(knights|knaves)\??$", re.I)


def kk_subq_truth(question: str, rec: dict) -> bool | None:
    """Program truth for role questions ("Is X a knight?", "Is X's statement true?", "Are X, Y and Z
    knaves?"); None for anything else (e.g. "Does X's statement force Y to be a knight?")."""
    roles = {p["question"][3:-10]: p["truth"] for p in rec.get("predicates", [])}
    q = " ".join(question.strip().split())
    m = _KK_ONE.match(q)
    if m and m.group(1) in roles:
        return roles[m.group(1)] == (m.group(2).lower() == "knight")
    m = _KK_STMT.match(q)
    if m and m.group(1) in roles:
        return roles[m.group(1)] == (m.group(2).lower() == "true")
    m = _KK_ALL.match(q)
    if m:
        names = [x for x in re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group(1)) if x]
        if names and all(x in roles for x in names):
            return all(roles[x] == (m.group(2).lower() == "knights") for x in names)
    return None


# ---------------------------------------------------------------- JustLogic
JL_OPTIONS = {"TRUE": "True", "FALSE": "False", "UNCERTAIN": "Uncertain"}


def justlogic(path: str, n: int | None = None, seed: int = 0, min_depth: int = 1) -> list[dict]:
    csv.field_size_limit(10 ** 9)
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["depth"]) >= min_depth]
    rng = random.Random(seed)
    rng.shuffle(rows)
    out = []
    for r in rows[:n] if n else rows:
        gold = r["label"].strip().upper()
        order = list(JL_OPTIONS)
        rng.shuffle(order)
        opt_txt = "\n".join(f"{chr(65 + i)}) {JL_OPTIONS[k]}" for i, k in enumerate(order))
        prompt = (f"Read the passage and judge the statement using only the passage and valid logical reasoning.\n\n"
                  f"Passage: {r['paragraph']}\n\nStatement: {r['question']}\n\n"
                  f"Question: Is the statement true, false, or uncertain given the passage?\nOptions:\n{opt_txt}")
        try:
            statements = ast.literal_eval(r["statements"])
        except (ValueError, SyntaxError):
            statements = {}
        out.append({"item_id": f"jl-{r['id']}", "domain": "justlogic", "hidden": None, "gold": gold,
                    "label_order": order, "gold_label": chr(65 + order.index(gold)), "depth": int(r["depth"]),
                    "n_rules": int(r["depth"]), "path": [], "predicates": [], "prompt": prompt,
                    "options": dict(JL_OPTIONS), "meta": {"arg": r["arg"], "statements": statements}})
    return out
