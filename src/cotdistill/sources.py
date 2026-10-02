"""Hard training sources turned into typed decision records (same schema as policygen).

- knights_knaves: generated Knights & Knaves puzzles (knights always tell the truth, knaves always
  lie) with a unique solution; Choice over full role assignments (the solution plus near-miss
  distractors). Per-person roles are program-verifiable predicates.
- justlogic: JustLogic (MIT) items as True / False / Uncertain choices, with their depth.
- sharc: ShARC (CC BY-SA 3.0) rule-application utterances (real government policy text, a user scenario and
  earlier follow-ups) as Yes / No / Irrelevant / More-information choices. Its evidence follow-ups (human
  question-answer pairs about the conditions) are kept as predicates.
- folio: FOLIO v2 (CC BY-SA 4.0 via tasksource) first-order-logic stories as True / False / Uncertain choices.
- arc: AI2 ARC (CC BY-SA 4.0) grade-school science multiple choice, Challenge first, options reshuffled.
"""
from __future__ import annotations

import ast
import json
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
                   n_options: int = 6, max_tries: int = 5000, question: str = "assignment",
                   names: list[str] | None = None) -> list[dict]:
    """question="assignment": choose the full role assignment (solution + near-miss distractors).
    question="count": how many inhabitants are knights (options 0..k) - no option can be checked
    against the statements without solving the whole puzzle."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        k = rng.randint(*people_range)
        people = rng.sample(names or NAMES, k)
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


_KK_PAIR = re.compile(r"^(?:do|are) (\w+) and (\w+) (?:have |of )?(?:the )?(same|opposite) roles?\??$", re.I)
_KK_HYP = re.compile(r"^(?:with|if|when|given that|assuming) (.+?),? (?:is|would|does|do|are|will) (.+?)\??$", re.I)
_KK_ASSIGN = re.compile(r"\b([A-Z][a-z]+)(?: were| was| is| being| as| to be)? (?:a |an )?(knight|knave)s?\b")
_KK_SAID = re.compile(r"^(\w+)'s statement (?:be )?(true|false)$", re.I)
_KK_CONTRA = re.compile(r"^(?:the |all (?:the |four |five |six |seven |eight |nine |ten )?)?statements (?:be )?"
                        r"(contradict each other|contradictory|inconsistent|consistent(?: with each other)?|all hold)$", re.I)


def _people_in(rec: dict) -> list[str]:
    return [p["question"][3:-10] for p in rec.get("predicates", [])]


# Rule nodes ("Is X a knight exactly when S?") restate what X's statement says; their truth is a property of the
# puzzle text, not of its solution: yes when S is equivalent to X's statement (for "knave": to its negation). S is
# parsed from the many ways the teacher rewrites a statement; anything ambiguous or unrecognised gives None.
_KK_RULE = re.compile(r"^(?:is|was) (\w+) (?:a|actually a|really a) (knight|knave),? (?:exactly when|precisely when|"
                      r"just when|if and only if|iff),? (.+?)\??$", re.I)
_KK_ROLE = r"an? (knight|knave)"


def _kk_atom(s: str, people: set[str]) -> tuple | None:
    m = re.match(rf"^(\w+) (?:is|being|is being|were|was) (not )?{_KK_ROLE}$", s, re.I)
    if m and m.group(1) in people:
        return ("is", m.group(1), (m.group(3).lower() == "knight") != bool(m.group(2)))
    m = re.match(r"^(\w+) (?:is telling the truth|tells the truth|is truthful|is lying|lies)$", s, re.I)
    if m and m.group(1) in people:
        return ("is", m.group(1), "lie" not in s.lower() and "lying" not in s.lower())
    return None


def _kk_formula(s: str, people: set[str]) -> tuple | None:
    """A role sentence as is/not/and/or/if/iff over role atoms; None when it is not unambiguously one."""
    s = " ".join(s.strip().rstrip("?.").split())
    s = re.sub(r"^(?:it is true that|it holds that|the claim that) ", "", s, flags=re.I)
    s = re.sub(r"\b(\w+)'s being\b", r"\1 being", s)
    m = re.match(r"^the (?:statement|claim)(?: that)? ['\"]?(.+?)['\"]? is (true|false)$", s, re.I)
    if m:
        a = _kk_formula(m.group(1), people)
        return (a if m.group(2).lower() == "true" else ("not", a)) if a else None
    m = re.match(r"^it is not (?:the case|true) that (.+)$", s, re.I)
    if m:
        a = _kk_formula(m.group(1), people)
        return ("not", a) if a else None
    m = re.match(r"^(?:(\w+) and (\w+) are not both|not both (\w+) and (\w+) are) (knights|knaves)$", s, re.I)
    if m:
        a, b = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        k = m.group(5).lower() == "knights"
        return ("not", ("and", ("is", a, k), ("is", b, k))) if {a, b} <= people else None
    m = re.match(r"^(?:exactly one of (\w+) and (\w+) is a knight|one of (\w+) and (\w+) is a knight and the other "
                 r"is a knave|(\w+) and (\w+) do not have the same roles?|(\w+) has the opposite role (?:of|from|to) (\w+))$",
                 s, re.I)
    if m:
        a, b = [x for x in m.groups() if x]
        return ("iff", ("is", a, True), ("is", b, False)) if {a, b} <= people else None
    m = re.match(r"^(\w+) has the same role as (\w+)$|^(\w+) and (\w+) are either both knights or both knaves$", s, re.I)
    if m:
        a, b = [x for x in m.groups() if x]
        return ("iff", ("is", a, True), ("is", b, True)) if {a, b} <= people else None
    m = re.match(rf"^neither (\w+) nor (\w+) is {_KK_ROLE}$", s, re.I)
    if m and {m.group(1), m.group(2)} <= people:
        k = m.group(3).lower() == "knight"
        return ("and", ("is", m.group(1), not k), ("is", m.group(2), not k))
    m = re.match(r"^(\w+) and (\w+) (?:have|share|are of|are) (?:the )?same (?:roles?|types?)$|"
                 r"^(\w+) and (\w+) are (?:both knights or both knaves|both knaves or both knights)$", s, re.I)
    if m:
        a, b = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        return ("iff", ("is", a, True), ("is", b, True)) if {a, b} <= people else None
    m = re.match(r"^(\w+) and (\w+) (?:have|are of|are) (?:the )?(?:opposite|different) (?:roles?|types?)$", s, re.I)
    if m:
        a, b = m.group(1), m.group(2)
        return ("iff", ("is", a, True), ("is", b, False)) if {a, b} <= people else None
    m = re.match(r"^(?:both )?(\w+) and (\w+) are (?:both )?(knights|knaves)$", s, re.I)
    if m and {m.group(1), m.group(2)} <= people:
        k = m.group(3).lower() == "knights"
        return ("and", ("is", m.group(1), k), ("is", m.group(2), k))
    m = re.match(rf"^(?:either )?(\w+) or (\w+) is {_KK_ROLE}$", s, re.I)
    if m and {m.group(1), m.group(2)} <= people:
        k = m.group(3).lower() == "knight"
        return ("or", ("is", m.group(1), k), ("is", m.group(2), k))
    for pat, op in ((r"^if (.+?),? then (.+)$", "if"), (r"^if (.+?), (.+)$", "if"),
                    (r"^(.+?) (?:implies|implying)(?: that)? (.+)$", "if"),
                    (r"^(.+?) (?:if and only if|iff|exactly when|precisely when|is equivalent to|matches) (.+)$", "iff"),
                    (r"^(.+?) (?<!and )only if (.+)$", "if"),
                    (r"^(.+?) and (.+?) (?:have the same truth value|are equivalent)$", "iff"),
                    (r"^(.+?) and (.+?) differ in truth value$", "xor"),
                    (r"^either (.+?) or (.+)$", "or")):
        m = re.match(pat, s, re.I)
        if m:
            a, b = _kk_formula(m.group(1), people), _kk_formula(m.group(2), people)
            if not (a and b):
                return None
            return ("not", ("iff", a, b)) if op == "xor" else (op, a, b)
    has_and, has_or = s.count(" and "), s.count(" or ")
    if has_and + has_or == 1:                                # one connective: no precedence to guess
        a, _, b = s.partition(" and " if has_and else " or ")
        a, b = _kk_formula(a, people), _kk_formula(b, people)
        return ("and" if has_and else "or", a, b) if a and b else None
    if has_and + has_or == 0:
        return _kk_atom(s, people)
    return None


def _kk_val(t: tuple, roles: dict[str, bool]) -> bool:
    if t[0] == "not":
        return not _kk_val(t[1], roles)
    if t[0] == "is":
        return roles[t[1]] == t[2]
    a, b = _kk_val(t[1], roles), _kk_val(t[2], roles)
    return {"and": a and b, "or": a or b, "if": (not a) or b, "iff": a == b}[t[0]]


def _kk_names(t: tuple) -> set[str]:
    return {t[1]} if t[0] == "is" else set().union(*(_kk_names(x) for x in t[1:]))


def _kk_tuple(st) -> tuple:
    return ("is", st[1], bool(st[2])) if st[0] == "is" else (st[0], _kk_tuple(st[1]), _kk_tuple(st[2]))


def kk_rule_truth(question: str, rec: dict) -> bool | None:
    """Truth of "Is X a knight/knave exactly when S?" under the puzzle's rules: True/False when S is equivalent to
    X's statement or to its negation; None when S does not parse or is equivalent to neither (for example the
    statement of another person: such a node is a transcription error, not a fact to be decided from the solution)."""
    m = _KK_RULE.match(" ".join(question.strip().split()))
    meta = rec.get("meta") or {}
    stmts, people = meta.get("statements") or {}, set(meta.get("people") or _people_in(rec))
    if not m or m.group(1) not in stmts:
        return None
    s = _kk_formula(m.group(3), people)
    if s is None:
        return None
    st = _kk_tuple(stmts[m.group(1)])
    names = sorted(_kk_names(s) | _kk_names(st))
    rows = [dict(zip(names, bits)) for bits in itertools.product([True, False], repeat=len(names))]
    knight = m.group(2).lower() == "knight"
    if all(_kk_val(s, r) == _kk_val(st, r) for r in rows):
        return knight
    if all(_kk_val(s, r) != _kk_val(st, r) for r in rows):
        return not knight
    return None


def kk_node_truth(question: str, rec: dict, rule: bool = True) -> bool | None:
    """Program truth for K&K tree nodes: single roles (kk_subq_truth), rule nodes (kk_rule_truth; rule=False
    reproduces the data built before it existed), same/opposite roles, a statement's truth under a hypothetical
    assignment, and whether a hypothesis makes the statements contradictory.
    None whenever the question is not in one of these forms or leaves a needed role unassigned."""
    t = kk_subq_truth(question, rec)
    if t is None and rule:
        t = kk_rule_truth(question, rec)
    if t is not None:
        return t
    meta = rec.get("meta") or {}
    stmts, people = meta.get("statements"), meta.get("people") or _people_in(rec)
    sol = {p["question"][3:-10]: p["truth"] for p in rec.get("predicates", [])}
    if not stmts or not sol:
        return None
    q = " ".join(question.strip().split())
    m = _KK_PAIR.match(q)
    if m and m.group(1) in sol and m.group(2) in sol:
        same = sol[m.group(1)] == sol[m.group(2)]
        return same if m.group(3).lower() == "same" else not same
    m = _KK_HYP.match(q)
    if not m:
        return None
    hyp, cons = m.group(1), m.group(2).strip()
    assign = {}
    for name, role in _KK_ASSIGN.findall(hyp):
        if name not in sol or assign.get(name, role == "knight") != (role == "knight"):
            return None
        assign[name] = role == "knight"
    named = {w for w in re.findall(r"\b[A-Z][a-z]+\b", hyp) if w in sol}
    if not assign or named - set(assign):                  # a person mentioned without a role: not parsed
        return None
    m2 = _KK_SAID.match(cons)
    if m2 and m2.group(1) in stmts:
        st = stmts[m2.group(1)]
        needed = set(re.findall(r"\b[A-Z][a-z]+\b", json.dumps(st))) & set(sol)
        if not needed <= set(assign):
            return None
        val = _eval(st, assign)
        return val if m2.group(2).lower() == "true" else not val
    m3 = _KK_CONTRA.match(cons)
    if m3:
        free = [p for p in people if p not in assign]
        ok = any(_consistent(stmts, dict(assign, **dict(zip(free, bits)))) == 0
                 for bits in itertools.product([True, False], repeat=len(free)))
        word = m3.group(1).lower()
        return (not ok) if word.startswith(("contradict", "inconsistent")) else ok
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


# ---------------------------------------------------------------- ShARC
SHARC_OPTIONS = {"yes": "Yes", "no": "No", "irrelevant": "The rule does not apply to this question",
                 "more": "More information is needed before answering"}


def _sharc_label(answer: str) -> str:
    a = answer.strip().lower()
    return a if a in ("yes", "no", "irrelevant") else "more"


def sharc(path: str, n: int | None = None, seed: int = 0, min_evidence: int = 2) -> list[dict]:
    """ShARC utterances needing at least `min_evidence` follow-ups (history + evidence; not required for the
    "does not apply" class), at most one per (rule, user question, answer class), classes balanced."""
    rows = json.load(open(path))
    rng = random.Random(seed)
    rng.shuffle(rows)
    seen, by_cls = set(), {k: [] for k in SHARC_OPTIONS}
    for r in rows:
        lab = _sharc_label(r["answer"])
        key = (r["tree_id"], r["question"].strip().lower(), lab)
        need = 0 if lab == "irrelevant" else min_evidence    # "does not apply" items carry no follow-ups
        if len(r.get("evidence") or []) + len(r.get("history") or []) < need or key in seen:
            continue
        seen.add(key)
        by_cls[lab].append(r)
    per = (n // len(SHARC_OPTIONS) + 1) if n else None
    picked = [r for lab in SHARC_OPTIONS for r in (by_cls[lab][:per] if per else by_cls[lab])]
    rng.shuffle(picked)
    out = []
    for r in picked[:n] if n else picked:
        gold = _sharc_label(r["answer"])
        order = list(SHARC_OPTIONS)
        rng.shuffle(order)
        hist = "\n".join(f"Q: {h['follow_up_question']}\nA: {h['follow_up_answer']}" for h in r.get("history") or [])
        opt_txt = "\n".join(f"{chr(65 + i)}) {SHARC_OPTIONS[k]}" for i, k in enumerate(order))
        prompt = (f"Rule text:\n{r['snippet'].strip()}\n\n"
                  + (f"User scenario: {r['scenario'].strip()}\n\n" if r.get("scenario") else "")
                  + (f"Earlier follow-up questions and answers:\n{hist}\n\n" if hist else "")
                  + f"User question: {r['question'].strip()}\n\n"
                  "Question: Based only on the rule text, the scenario and the earlier answers, how should the user's "
                  f"question be answered?\nOptions:\n{opt_txt}")
        preds = [{"pid": f"ev{i}", "question": e["follow_up_question"], "truth": e["follow_up_answer"].strip().lower() == "yes",
                  "kind": "evidence"} for i, e in enumerate(r.get("evidence") or [])
                 if e.get("follow_up_answer", "").strip().lower() in ("yes", "no")]
        out.append({"item_id": f"sharc-{r['utterance_id'][:12]}", "domain": "sharc", "hidden": None, "gold": gold,
                    "label_order": order, "gold_label": chr(65 + order.index(gold)),
                    "depth": len(r.get("evidence") or []), "n_rules": len(r.get("evidence") or []), "path": preds,
                    "predicates": preds, "prompt": prompt, "options": dict(SHARC_OPTIONS),
                    "meta": {"tree_id": r["tree_id"], "source_url": r.get("source_url")}})
    return out


# ---------------------------------------------------------------- FOLIO
FOLIO_OPTIONS = {"True": "True", "False": "False", "Uncertain": "Uncertain"}


def folio(path: str, n: int | None = None, seed: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    out = []
    for r in rows[:n] if n else rows:
        gold = r["label"]
        if gold not in FOLIO_OPTIONS:
            continue
        order = list(FOLIO_OPTIONS)
        rng.shuffle(order)
        opt_txt = "\n".join(f"{chr(65 + i)}) {FOLIO_OPTIONS[k]}" for i, k in enumerate(order))
        prem = r["premises"].strip()
        prompt = ("Read the premises and judge the conclusion using only the premises and valid logical reasoning.\n\n"
                  f"Premises:\n{prem}\n\nStatement: {r['conclusion'].strip()}\n\n"
                  f"Question: Is the statement true, false, or uncertain given the premises?\nOptions:\n{opt_txt}")
        out.append({"item_id": f"folio-{r['example_id']}", "domain": "folio", "hidden": None, "gold": gold,
                    "label_order": order, "gold_label": chr(65 + order.index(gold)),
                    "depth": len([x for x in prem.splitlines() if x.strip()]), "n_rules": 0, "path": [],
                    "predicates": [], "prompt": prompt, "options": dict(FOLIO_OPTIONS),
                    "meta": {"story_id": r.get("story_id")}})
    return out


# ---------------------------------------------------------------- ARC
_CALC = re.compile(r"<<[^=<>]*=([^<>]+)>>")


def _num(x: str) -> float | None:
    try:
        return float(x.replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def _fmt(v: float) -> str:
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.2f}".rstrip("0").rstrip(".")


def gsm8k(path: str, n: int | None = None, seed: int = 0, n_options: int = 5) -> list[dict]:
    """GSM8K word problems as multiple choice. Distractors are the solution's own intermediate results (from the
    <<a*b=c>> calculator notes) plus one-sided perturbations of the answer or of an intermediate result, so that
    neither "largest", "smallest", "middle value" nor "has a near neighbour" picks the answer. depth = number of
    calculator steps in the reference solution."""
    import pandas as pd
    rng = random.Random(seed)
    rows = list(pd.read_parquet(path).itertuples(index=False))
    split = "test" if "test" in path else "train"
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        q, sol = rows[i].question.strip(), rows[i].answer
        gold = _num(sol.rsplit("####", 1)[-1])
        if gold is None or gold < 0:
            continue
        steps = [v for v in (_num(m) for m in _CALC.findall(sol)) if v is not None]
        inter = [v for v in dict.fromkeys(steps) if v >= 0 and abs(v - gold) > 1e-9]
        pick = rng.sample(inter, min(2, len(inter)))
        pool = []
        for base in [gold] + inter:
            d = rng.choice([1, 2, 5, 10]) if base >= 20 else rng.choice([1, 2])
            pool += [base + d if rng.random() < 0.5 else base - d, base * 2]
            if base % 2 == 0:
                pool.append(base / 2)
        pool = [v for v in dict.fromkeys(pool) if v > 0 and abs(v - gold) > 1e-9 and v not in pick]
        rng.shuffle(pool)
        dis = (pick + pool)[:n_options - 1]
        if len(dis) < n_options - 1:
            continue
        texts = [_fmt(gold)] + [_fmt(v) for v in dis]
        if len(set(texts)) < n_options:
            continue
        keys = [f"o{j}" for j in range(n_options)]
        options = dict(zip(keys, texts))
        order = keys[:]
        rng.shuffle(order)
        opt_txt = "\n".join(f"{chr(65 + j)}) {options[k]}" for j, k in enumerate(order))
        out.append({"item_id": f"gsm8k-{split}-{i}", "domain": "gsm8k", "hidden": None, "gold": "o0",
                    "label_order": order, "gold_label": chr(65 + order.index("o0")), "depth": max(1, len(steps)),
                    "n_rules": 0, "path": [], "predicates": [], "prompt": f"Question: {q}\nOptions:\n{opt_txt}",
                    "options": options, "meta": {"split": split, "steps": len(steps)}})
        if n and len(out) >= n:
            break
    return out


def arc(paths: list[str], n: int | None = None, seed: int = 0) -> list[dict]:
    """ARC questions from the given parquet files in order (Challenge first), options reshuffled."""
    import pandas as pd
    rng = random.Random(seed)
    out = []
    for path in paths:
        df = pd.read_parquet(path)
        rows = list(df.itertuples(index=False))
        rng.shuffle(rows)
        split = "challenge" if "Challenge" in path else "easy"
        for r in rows:
            texts, labs = list(r.choices["text"]), [str(x) for x in r.choices["label"]]
            if str(r.answerKey) not in labs or len(texts) < 3:
                continue
            keys = [f"o{i}" for i in range(len(texts))]
            options = dict(zip(keys, texts))
            gold = keys[labs.index(str(r.answerKey))]
            order = keys[:]
            rng.shuffle(order)
            opt_txt = "\n".join(f"{chr(65 + i)}) {options[k]}" for i, k in enumerate(order))
            prompt = f"Question: {r.question.strip()}\nOptions:\n{opt_txt}"
            out.append({"item_id": f"arc-{r.id}", "domain": f"arc_{split}", "hidden": None, "gold": gold,
                        "label_order": order, "gold_label": chr(65 + order.index(gold)), "depth": 1 if split == "easy" else 2,
                        "n_rules": 0, "path": [], "predicates": [], "prompt": prompt, "options": options, "meta": {"split": split}})
            if n and len(out) >= n:
                return out
    return out