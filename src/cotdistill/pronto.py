"""PrOntoQA-OOD-style deduction problems (Saparov et al., NeurIPS 2023; github.com/asaparov/prontoqa, Apache-2.0) with
exact proof trees, for the Tree Forcing depth-extrapolation study (source "pronto", scripts/build_pronto_tf.py).

Own generator, written to the paper's definitions (the upstream run_experiment.py needs scipy and was not run): a
fictional ontology (nonsense "-pus" concept nouns, PrOntoQA's 13 property families), universally quantified rules in
PrOntoQA's three surface forms ("Every X is a Y." / "Each X is a Y." / "Xes are Ys.", negated with "not"), rules in
random order with the entity facts last, and a query "True or false: E is (not) P." whose answer is False by negating
the proven conclusion (as upstream). Only the modus-ponens rule type (PrOntoQA-OOD's "ModusPonens"); hops k = number
of rule applications in the proof of the query: E's fact "E is a c_0", k - 1 concept rules c_0 -> c_1 -> ... ->
c_{k-1}, and the property rule "every c_{k-1} is (not) P".

Distractors (upstream: a second fact puts the entity on a distractor chain, and "every D is not A" contradicts the
query predicate). Here, per problem:
  - main pair: E's chain C = c_0..c_{k-1} and a twin chain D = d_0..d_{k-1} of a second entity F ("F is a d_0"), same
    length, whose last concept carries the query property with the OPPOSITE polarity; so the property appears in two
    rules of opposite sign and only tracing E's chain decides which one applies;
  - distractor pair: X (E's second fact) and Y (F's second fact), equal length L in [1, k + 1], never the query
    property;
  - side branches ("every c is an s", s with a property; p = side_p per chain concept), and one property per concept
    with probability prop_p (each adjective at most once per problem, so no entity can derive p and not p);
  - twin negatives (our extension, not in PrOntoQA): for every pair (A, B) and level i >= 1, "every a_{i-1} is not a
    b_i" and "every b_{i-1} is not an a_i". They make the No nodes below provable in one step with the same surface
    form as the Yes nodes ("Is E a d_i?"), so "not" in a node question never predicts its answer, and they appear on
    both pairs so they do not single out the chain pair that matters. They derive only negative memberships, which
    no rule takes as a premise, so the query's shortest proof stays exactly k steps (checked by build_pronto_tf.py).
  - width (OOD file): extra pairs whose first chain hangs off E and second off F, and side_p .5.

Proof tree (subqs; fields as data/student_numgen_tf; every truth exact: a Yes node's statement is in the deductive
closure of the problem, a No node's negation is; node_depth = proof depth):
  read    r0 "Is E a c_0?" (Yes), local_lines = [E's fact sentence]
  parse   for i = 1..k-1: p{i} "Is every c_{i-1} a c_i?" (Yes) / p{i}x "Is every c_{i-1} not a c_i?" (No), q{i} "Is
          every c_{i-1} not a d_i?" (Yes) / q{i}x "Is every c_{i-1} a d_i?" (No), each with local_lines = [the rule's
          sentence]; and pP / pPx the property rule of c_{k-1} as stated (Yes) and with its polarity flipped (No)
  derive  for i = 1..k-1: m{i} "Is E a c_i?" (Yes) and n{i} "Is E a d_i?" (No), both depends_on [m{i-1} (r0 for
          i = 1), p{i}, q{i}] (shared parents, as numgen's v/n pair: the local view is the same and the asked concept
          must be matched to the right rule)
  not built: the last step "Is E (not) P?" (the final question in disguise).
"""
from __future__ import annotations

import random
import re

LOCAL_PREAMBLE = "The following facts and rules describe a fictional world."
QUESTION = "Question: True or false: {query}"
FINAL_RE = r"Question: True or false: [A-Z][a-z]+ is (?:not )?[a-z]+\."      # treeplan.FINAL_QUESTION["pronto"]
OPTIONS = {"A": "True", "B": "False"}
SOURCE = "pronto"

# PrOntoQA's entity names (run_experiment.available_entity_names) and more short first names
ENTITIES = ["Fae", "Rex", "Sally", "Max", "Alex", "Sam", "Polly", "Stella", "Wren",
            "Ada", "Bea", "Cal", "Dora", "Eli", "Finn", "Gus", "Hal", "Ivy", "Jo", "Kit", "Lou", "Mae", "Ned", "Opal",
            "Pip", "Quinn", "Roy", "Sid", "Tess", "Uma", "Vic", "Wes", "Zoe", "Abe", "Bo", "Cy", "Dee", "Gil", "Hope",
            "Ike", "Jude", "Liv", "Mo", "Nell", "Otto", "Ray", "Ruth", "Ty", "Val"]
# PrOntoQA's fictional concept names (fictional ontology + irrelevant names); generated "-pus" nouns extend the pool
PRONTO_CONCEPTS = ["wumpus", "yumpus", "zumpus", "dumpus", "rompus", "numpus", "tumpus", "vumpus", "impus", "jompus",
                   "gorpus", "shumpus", "lempus", "sterpus", "grimpus", "lorpus", "brimpus",
                   "timpus", "yimpus", "rempus", "fompus", "worpus", "terpus", "gerpus", "kerpus", "scrompus", "zhorpus",
                   "bompus", "jelpus", "felpus", "chorpus", "hilpus", "storpus", "yerpus", "boompus", "gwompus",
                   "rorpus", "quimpus"]
_ONSETS = ["", "b", "bl", "br", "ch", "d", "dr", "f", "fl", "fr", "g", "gl", "gr", "h", "j", "k", "kl", "kr", "l", "m",
           "n", "p", "pl", "pr", "qu", "r", "s", "sc", "sh", "sk", "sl", "sn", "sp", "st", "sw", "t", "tr", "v", "w",
           "y", "z", "zh"]
_REAL = {"rumpus", "grampus", "tempus", "pompus", "scirpus", "wampus", "campus", "corpus", "olpus", "humpus"}
PROPERTY_FAMILIES = [["blue", "red", "brown", "orange"], ["small", "large"], ["metallic", "wooden", "luminous", "liquid"],
                     ["transparent", "opaque"], ["nervous", "happy", "feisty", "shy"], ["bright", "dull"],
                     ["sweet", "sour", "spicy", "bitter"], ["floral", "fruity", "earthy"], ["hot", "cold", "temperate"],
                     ["kind", "mean", "angry", "amenable", "aggressive"], ["melodic", "muffled", "discordant", "loud"],
                     ["slow", "moderate", "fast"], ["windy", "sunny", "overcast", "rainy", "snowy"]]
ADJECTIVES = [a for fam in PROPERTY_FAMILIES for a in fam]


def concept_pool() -> list[str]:
    """PrOntoQA's 38 names first, then onset + vowel + mpus/rpus/lpus nouns (no English words), deterministic order."""
    out = list(PRONTO_CONCEPTS)
    for o in _ONSETS:
        for v in "aeiou":
            for c in ("mpus", "rpus", "lpus"):
                w = o + v + c
                if w not in _REAL and w not in out:
                    out.append(w)
    return out


CONCEPTS = concept_pool()
assert not set(ADJECTIVES) & set(CONCEPTS) and not any(a.endswith("pus") for a in ADJECTIVES)


def article(w: str) -> str:
    return "an" if w[0] in "aeiou" else "a"


def plural(w: str) -> str:
    return w + "es"


# ------------------------------------------------------------------------------------------------------- rendering
# rule = (subject concept, object, kind "concept" | "prop", negated); fact = (entity, concept)
def rule_sentence(rule, template: int) -> str:
    s, o, kind, neg = rule
    nt = "not " if neg else ""
    if template == 2:
        obj = plural(o) if kind == "concept" else o
        return f"{plural(s).capitalize()} are {nt}{obj}."
    obj = f"{article(o)} {o}" if kind == "concept" else o
    return f"{'Every' if template == 0 else 'Each'} {s} is {nt}{obj}."


def fact_sentence(fact) -> str:
    e, c = fact
    return f"{e} is {article(c)} {c}."


def literal_sentence(e: str, p: str, neg: bool) -> str:
    return f"{e} is {'not ' if neg else ''}{p}."


def membership_q(e: str, c: str) -> str:
    return f"Is {e} {article(c)} {c}?"


def rule_q(rule) -> str:
    s, o, kind, neg = rule
    obj = f"{article(o)} {o}" if kind == "concept" else o
    return f"Is every {s} {'not ' if neg else ''}{obj}?"


_RULE_SG = re.compile(r"^(Every|Each) ([a-z]+) is (not )?(?:(?:a|an) ([a-z]+pus)|([a-z]+))\.$")
_RULE_PL = re.compile(r"^([A-Z][a-z]*puses) are (not )?([a-z]+)\.$")
_FACT = re.compile(r"^([A-Z][a-z]+) is (a|an) ([a-z]+pus)\.$")
_RULE_Q = re.compile(r"^Is every ([a-z]+) (not )?(?:(?:a|an) ([a-z]+pus)|([a-z]+))\?$")
_MEMBER_Q = re.compile(r"^Is ([A-Z][a-z]+) (?:a|an) ([a-z]+pus)\?$")
_QUERY = re.compile(r"^([A-Z][a-z]+) is (not )?([a-z]+)\.$")


def parse_sentence(s: str):
    """("rule", (subj, obj, kind, neg)) or ("fact", (entity, concept)); None if the sentence has none of the forms."""
    m = _RULE_SG.match(s)
    if m:
        conc = m.group(4)
        return "rule", (m.group(2), conc or m.group(5), "concept" if conc else "prop", bool(m.group(3)))
    m = _RULE_PL.match(s)
    if m:
        o = m.group(3)
        conc = o.endswith("puses")
        return "rule", (m.group(1)[:-2].lower(), o[:-2] if conc else o, "concept" if conc else "prop", bool(m.group(2)))
    m = _FACT.match(s)
    if m:
        if article(m.group(3)) != m.group(2):
            return None
        return "fact", (m.group(1), m.group(3))
    return None


def parse_rule_q(q: str):
    m = _RULE_Q.match(q)
    if not m:
        return None
    conc = m.group(3)
    return (m.group(1), conc or m.group(4), "concept" if conc else "prop", bool(m.group(2)))


def parse_member_q(q: str):
    m = _MEMBER_Q.match(q)
    return (m.group(1), m.group(2)) if m else None


def split_sentences(body: str) -> list[str]:
    return [s + "." for s in body[:-1].split(". ")] if body.endswith(".") else []


# --------------------------------------------------------------------------------------------------------- solver
def closure(facts, rules) -> dict:
    """Forward chaining (modus ponens) from the facts: {(entity, predicate): (negated, steps)} with steps = the length
    of the shortest chain of rule applications (facts 0). Raises ValueError on a contradiction (p and not p)."""
    by_subj: dict = {}
    for r in rules:
        by_subj.setdefault(r[0], []).append(r)
    known: dict = {}
    frontier = []
    for e, c in facts:
        if (e, c) in known and known[(e, c)][0]:
            raise ValueError(f"contradiction on {(e, c)}")
        if (e, c) not in known:
            known[(e, c)] = (False, 0)
            frontier.append((e, c))
    step = 0
    while frontier:
        step += 1
        nxt = []
        for e, c in frontier:
            for _, o, _, neg in by_subj.get(c, []):
                if (e, o) in known:
                    if known[(e, o)][0] != neg:
                        raise ValueError(f"contradiction on {(e, o)}")
                    continue
                known[(e, o)] = (neg, step)
                if not neg:
                    nxt.append((e, o))
        frontier = nxt
    return known


def parse_body(body: str):
    facts, rules = [], []
    for s in split_sentences(body):
        p = parse_sentence(s)
        if p is None:
            raise ValueError(f"unparsed sentence {s!r}")
        (facts if p[0] == "fact" else rules).append(p[1])
    return facts, rules


# ------------------------------------------------------------------------------------------------------ generation
def make_problem(k: int, answer: bool, query_neg: bool, rng: random.Random, *, width: int = 0, side_p: float = 0.25,
                 prop_p: float = 0.6) -> dict:
    """One k-hop problem (k >= 1) with the given answer and query polarity (the proven polarity follows)."""
    if k < 1:
        raise ValueError("k >= 1")
    e_q, e_f = rng.sample(ENTITIES, 2)
    n_pairs = 2 + width
    lens = [k] + [rng.randint(1, k + 1) for _ in range(n_pairs - 1)]
    names = iter(rng.sample(CONCEPTS, 4 * sum(lens)))                 # chain concepts, then at most one side each
    pairs = [([next(names) for _ in range(n)], [next(names) for _ in range(n)]) for n in lens]
    adjs = ADJECTIVES[:]
    rng.shuffle(adjs)
    prop = adjs.pop()
    rule_neg = query_neg if answer else not query_neg                 # E is (not) prop iff the main rule says so
    rules = []
    for a, b in pairs:
        for ch in (a, b):
            rules += [(ch[i - 1], ch[i], "concept", False) for i in range(1, len(ch))]
        for i in range(1, len(a)):                                    # twin negatives
            rules += [(a[i - 1], b[i], "concept", True), (b[i - 1], a[i], "concept", True)]
    c_chain, d_chain = pairs[0]
    main_rule = (c_chain[-1], prop, "prop", rule_neg)
    rules += [main_rule, (d_chain[-1], prop, "prop", not rule_neg)]
    sp = .5 if width else side_p
    owners = [c for a, b in pairs for c in a + b]
    sides = []
    for c in owners:
        if rng.random() < sp:
            s = next(names)
            sides.append(s)
            rules.append((c, s, "concept", False))
    for c in owners + sides:
        if adjs and rng.random() < prop_p:
            rules.append((c, adjs.pop(), "prop", rng.random() < 0.5))
    facts = [(e_q, a[0]) for a, _ in pairs] + [(e_f, b[0]) for _, b in pairs]
    rng.shuffle(rules)
    rng.shuffle(facts)
    templates = [rng.randrange(3) for _ in rules]
    return {"hops": k, "width": width, "entity": e_q, "other": e_f, "property": prop, "query_neg": query_neg,
            "rule_neg": rule_neg, "answer": answer, "pairs": [[a, b] for a, b in pairs], "sides": sides,
            "rules": [list(r) + [t] for r, t in zip(rules, templates)], "facts": [list(f) for f in facts]}


def body_text(th: dict) -> str:
    return " ".join([rule_sentence(tuple(r[:4]), r[4]) for r in th["rules"]] + [fact_sentence(f) for f in th["facts"]])


def query_text(th: dict) -> str:
    return literal_sentence(th["entity"], th["property"], th["query_neg"])


def prompt_of(th: dict) -> str:
    return (f"{LOCAL_PREAMBLE}\n\n{body_text(th)}\n\n{QUESTION.format(query=query_text(th))}\nOptions:\n"
            + "\n".join(f"{k}) {v}" for k, v in OPTIONS.items()))


def canonical(th: dict) -> tuple:
    """Logical identity of a problem (sentence templates and order ignored): no two items may share it."""
    return (frozenset(tuple(r[:4]) for r in th["rules"]), frozenset(tuple(f) for f in th["facts"]), th["entity"],
            th["property"], th["query_neg"])


def _sentence_of(th: dict, rule) -> str:
    for r in th["rules"]:
        if tuple(r[:4]) == tuple(rule):
            return rule_sentence(tuple(r[:4]), r[4])
    raise KeyError(rule)


def _node(nid, typ, question, truth, deps, lines=None, **kw) -> dict:
    out = {"question": question, "p_cot": 1.0 if truth else 0.0, "p_fresh": None, "truth": truth, "type": typ,
           "depends_on": deps, "id": nid, "initial": None, "status": "direct"}
    if lines:
        out["local_lines"] = lines
    out.update(kw)
    return out


def proof_tree(th: dict) -> list[dict]:
    """The proof nodes of the query (module docstring), node_depth = proof depth, without the final step."""
    e = th["entity"]
    c, d = th["pairs"][0]
    k = th["hops"]
    nodes = [_node("r0", "read", membership_q(e, c[0]), True, [], [fact_sentence((e, c[0]))], kind="fact", level=0,
                   opposite=f"Is {e} not {article(c[0])} {c[0]}?", node_depth=0)]
    prev = "r0"
    for i in range(1, k):
        pos, neg = (c[i - 1], c[i], "concept", False), (c[i - 1], d[i], "concept", True)
        for nid, r in ((f"p{i}", pos), (f"q{i}", neg)):
            flip = r[:3] + (not r[3],)
            line = [_sentence_of(th, r)]
            nodes.append(_node(nid, "parse", rule_q(r), True, [], line, kind="concept_rule" if not r[3] else
                               "twin_rule", level=i, opposite=rule_q(flip), node_depth=0))
            nodes.append(_node(nid + "x", "parse", rule_q(flip), False, [], line, kind="concept_rule" if not r[3] else
                               "twin_rule", level=i, opposite=rule_q(r), node_depth=0))
        deps = [prev, f"p{i}", f"q{i}"]
        nodes.append(_node(f"m{i}", "derive", membership_q(e, c[i]), True, deps, kind="member", level=i,
                           concept=c[i], opposite=f"Is {e} not {article(c[i])} {c[i]}?", node_depth=i))
        nodes.append(_node(f"n{i}", "derive", membership_q(e, d[i]), False, deps, kind="member", level=i,
                           concept=d[i], opposite=f"Is {e} not {article(d[i])} {d[i]}?", node_depth=i))
        prev = f"m{i}"
    pr = (c[-1], th["property"], "prop", th["rule_neg"])
    flip = pr[:3] + (not pr[3],)
    line = [_sentence_of(th, pr)]
    nodes.append(_node("pP", "parse", rule_q(pr), True, [], line, kind="property_rule", level=k, opposite=rule_q(flip),
                       node_depth=0))
    nodes.append(_node("pPx", "parse", rule_q(flip), False, [], line, kind="property_rule", level=k,
                       opposite=rule_q(pr), node_depth=0))
    return nodes


def node_leak(question: str, th: dict) -> bool:
    """Whether a node asks the final question in disguise: it names both the query entity and the query property
    (any polarity), or contains the query sentence."""
    words = set(re.findall(r"[A-Za-z]+", question))
    return (th["entity"] in words and th["property"] in words) or query_text(th)[:-1] in question


def final_item(th: dict, item_id: str) -> dict:
    gold = "A" if th["answer"] else "B"
    return {"item_id": item_id, "source": SOURCE, "prompt": prompt_of(th), "labels": list(OPTIONS), "gold_label": gold,
            "options": dict(OPTIONS), "depth": th["hops"], "hops": th["hops"], "width": th["width"],
            "answer": th["answer"], "query_neg": th["query_neg"], "rule_neg": th["rule_neg"],
            "n_sentences": len(th["rules"]) + len(th["facts"]), "theory": th}


def train_item(th: dict, item_id: str) -> dict:
    it = final_item(th, item_id)
    it["teacher"] = {L: 1.0 if L == it["gold_label"] else 0.0 for L in it["labels"]}
    it["subqs"] = proof_tree(th)
    it["has_tree"] = bool(it["subqs"])
    it["tree_source"] = "pronto_proof_tf"
    it["local_preamble"] = LOCAL_PREAMBLE
    it["tree_fix"] = True
    return it


def eval_item(th: dict, item_id: str, group: str) -> dict:
    it = final_item(th, item_id)
    it.update({"group": group, "type": "choice", "label_order": list(OPTIONS), "gold_probs": None})
    return it
