"""Lean F1 pilot generator (source "lean_f1", scripts/build_lean_tf.py; design: notes/next_round/lean/feasibility.md
section 2). Torch-free; numpy only for the exhaustive one-period brute force (bf_status), imported lazily.

Fragment F1: integer variables, congruence hypotheses "x % m = r" (literal m >= 2, 0 <= r < m, Lean's Int `%` = the
Euclidean remainder, as Python's `%` for m > 0) and linear definitions "y = a*x + b" / "y = a*x + c*w + b" (a > 0, b != 0).
No inequalities, so the brute force over one period of every free variable decides every label exactly.

Generator (formal first):
  - a main chain of `depth` levels from a root x0 (congruence x0 % m0 = r0); every level consumes a fresh premise:
      subst     y = a*u + b with gcd(a, M) = 1          (u % M known  ->  y % M)
      lift      y = a*u + b with a in {2, 3}             (u % M known  ->  y % (a*M), a residue lifted to a multiple)
      add_root  y = a*u + c*w + b and a fresh root with w % M = s   (two congruences combined through a sum)
      add_prev  y = a*u + c*p + b, p an earlier chain variable      (fan-in inside the chain)
      crt       a fresh congruence u % m2 = s on the current variable, gcd(m2, M) = 1 (combine: u % (M*m2))
  - distractor premises (irrelevant): fresh roots with a congruence, fresh variables defined from existing ones. They
    never constrain a chain variable, so they change no label and no depth;
  - premises shuffled; informal sentences from templates (CONG_TPL, DEF_TPL).
Depth = the shortest derivation in the rule set R = {subst, lift, add, combine (crt / lcm of two congruences of one
variable), reduce (mod a divisor)}, moduli <= CLOSURE_CAP, computed by round-based forward closure (closure(); like
PrOntoQA's hops). An item of depth d keeps a key claim z % M = R whose closure depth is exactly d, whose last rule is not
`reduce`, and whose shortest proof uses >= d + 1 premises; the proof's nodes are the tree (node_depth = closure depth).

Final question (format (i), 5 options), property-level wording QUESTION ("What is the remainder when z is divided by
M?"), so that a true statement about anything else is not an answer ("semantic drift" axis, owner 2026-10-11):
  key           z % M = R                                      entailed: omega (grind second certifier)
  on_refuted    z % M = r, r != R  (1 or 2 per item)             refuted: omega proves the negation
  independent   off-target claim true in some models, false in others: two witness assignments (decide)
  off_refuted   off-target refuted claim
  drift_true    off-target but TRUE (Lean proves it), generator tag "off_target" with subtype
                a = an entailed claim about another variable, b = a restated congruence premise,
                c = a true claim about z at another divisor m' (never a multiple of M)
Off-target = another variable, or z at a divisor other than M (multiples of M are never used: z % 16 = 12 would imply
the asked remainder). Under this wording no on-target option can be independent (z % M is determined); independents
are off-target. Tautologies (subtype d) are not used: no templated tautology is clearly not an answer here.

Tree nodes (format (iii); fields as data/student_numgen_tf / student_pronto_tf): read (congruence premise) / parse
(definition premise) leaves and derive nodes (every intermediate claim of the key's shortest proof, the key itself
excluded), each with a No twin (a refuted variant: another residue at the same modulus, or a definition with another
constant of the same sign). Every node is certified from its PARENTS ONLY; every critical premise is necessary
(a witness of the other premises where the key fails).
"""
from __future__ import annotations

import collections
import functools
import hashlib
import json
import math
import random
import re

SOURCE = "lean_f1"
TREE_SOURCE = "lean_f1_proof_tf"
PREAMBLE = ('Every letter below stands for an integer, which may be negative. "n leaves remainder r when divided by m" '
            'means that r is one of 0, 1, ..., m - 1 and n - r is a multiple of m.')
QUESTION = "Question: What is the remainder when {v} is divided by {m}?"
FINAL_RE = r"Question: What is the remainder when [a-z] is divided by [1-9][0-9]*\?"   # treeplan.statement_block
LETTERS = "ABCDE"
VARS = "bcdfghjkmnpqrstuvwxyz"            # no a / e / i / l / o (English words, digit look-alikes)
HYP_MODULI = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12)
HYP_WEIGHTS = (1, 2, 3, 2, 3, 1, 3, 2, 2, 3)   # composites weighted up: drift subtype (c) needs a second divisor
COEFS = (1, 2, 3, 4, 5, 7)
MMAX = 60                                 # chain moduli
CLOSURE_CAP = 360                         # moduli of the rule set's closure (depth definition)
MAX_CHAIN_ROOTS = 2
OFF_MODULI = tuple(range(2, 25))
NZ_MODULI = tuple(range(2, 13))
GRID_CAP = 1_000_000
N_MODELS = 300
ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
CONG_TPL = ("{v} leaves remainder {r} when divided by {m}.", "When {v} is divided by {m}, the remainder is {r}.",
            "Dividing {v} by {m} leaves remainder {r}.")
DEF_TPL = ("{v} equals {e}.", "{v} is equal to {e}.")


# ------------------------------------------------------------------------------------------------------ basics
def lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


def lcm_all(xs) -> int:
    return functools.reduce(lcm, xs, 1)


def divisors(n: int) -> list[int]:
    return [d for d in range(2, n) if n % d == 0]


def crt_pair(m1: int, r1: int, m2: int, r2: int):
    """(lcm, r) with r = r1 mod m1 and r = r2 mod m2, or None if inconsistent."""
    g = math.gcd(m1, m2)
    if (r1 - r2) % g:
        return None
    L = lcm(m1, m2)
    for r in range(r1 % m1, L, m1):
        if r % m2 == r2 % m2:
            return L, r
    return None


def seeded(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16))


def wchoice(rng: random.Random, items, weights):
    return rng.choices(list(items), weights=list(weights), k=1)[0]


def cong(v: str, m: int, r: int) -> dict:
    return {"kind": "cong", "var": v, "m": m, "r": r % m}


def defn(v: str, terms, c: int) -> dict:
    if not terms or terms[0][0] <= 0 or c == 0:
        raise ValueError("definitions: first coefficient > 0, constant != 0")
    return {"kind": "def", "var": v, "terms": [[int(a), u] for a, u in terms], "c": int(c)}


def claim_key(c: dict) -> tuple:
    return (c["var"], c["m"], c["r"])


# ------------------------------------------------------------------------------------------------------ rendering
def expr_text(terms, c: int) -> str:
    out = []
    for i, (a, u) in enumerate(terms):
        t = u if abs(a) == 1 else f"{abs(a)} times {u}"
        out.append(t if i == 0 else ("plus " if a > 0 else "minus ") + t)
    if c:
        out.append(("plus " if c > 0 else "minus ") + str(abs(c)))
    return " ".join(out)


def expr_lean(terms, c: int) -> str:
    s = ""
    for i, (a, u) in enumerate(terms):
        t = u if abs(a) == 1 else f"{abs(a)} * {u}"
        s = t if i == 0 else s + (" + " if a > 0 else " - ") + t
    if c:
        s += (" + " if c > 0 else " - ") + str(abs(c))
    return s


def lean_prop(h: dict) -> str:
    if h["kind"] == "cong":
        return f"{h['var']} % {h['m']} = {h['r']}"
    return f"{h['var']} = {expr_lean(h['terms'], h['c'])}"


def sentence(h: dict) -> str:
    if h["kind"] == "cong":
        return CONG_TPL[h.get("tpl", 0)].format(v=h["var"], m=h["m"], r=h["r"])
    return DEF_TPL[h.get("tpl", 0)].format(v=h["var"], e=expr_text(h["terms"], h["c"]))


def question_of(h: dict) -> str:
    if h["kind"] == "cong":
        return f"Does {h['var']} leave remainder {h['r']} when divided by {h['m']}?"
    return f"Is {h['var']} equal to {expr_text(h['terms'], h['c'])}?"


def option_text(c: dict) -> str:
    return f"{c['var']} leaves remainder {c['r']} when divided by {c['m']}"


def body_text(hyps: list[dict]) -> str:
    return "\n".join(sentence(h) for h in hyps)


def prompt_of(prog: dict, options: list[dict]) -> str:
    k = prog["key"]
    return (f"{PREAMBLE}\n\n{body_text(prog['hyps'])}\n\n{QUESTION.format(v=k['var'], m=k['m'])}\nOptions:\n"
            + "\n".join(f"{L}) {option_text(o)}" for L, o in zip(LETTERS, options)))


# ------------------------------------------------------------------------------------------------------ parsing
_CONG_RE = [(re.compile(r"^([a-z]) leaves remainder (\d+) when divided by (\d+)\.$"), (1, 3, 2)),
            (re.compile(r"^When ([a-z]) is divided by (\d+), the remainder is (\d+)\.$"), (1, 2, 3)),
            (re.compile(r"^Dividing ([a-z]) by (\d+) leaves remainder (\d+)\.$"), (1, 2, 3))]
_DEF_RE = re.compile(r"^([a-z]) (?:equals|is equal to) (.+)\.$")
_CONG_Q = re.compile(r"^Does ([a-z]) leave remainder (\d+) when divided by (\d+)\?$")
_DEF_Q = re.compile(r"^Is ([a-z]) equal to (.+)\?$")
_OPTION = re.compile(r"^([a-z]) leaves remainder (\d+) when divided by (\d+)$")


def parse_expr(e: str):
    """'2 times x minus 3 times w plus 1' -> ([[2, 'x'], [-3, 'w']], 1); None if not of that shape."""
    w = e.split()
    terms, c, sign, i = [], 0, 1, 0
    while i < len(w):
        if i > 0:
            if w[i] not in ("plus", "minus"):
                return None
            sign = 1 if w[i] == "plus" else -1
            i += 1
        if i < len(w) and w[i].isdigit() and i + 2 < len(w) and w[i + 1] == "times" and re.fullmatch(r"[a-z]", w[i + 2]):
            terms.append([sign * int(w[i]), w[i + 2]])
            i += 3
        elif i < len(w) and re.fullmatch(r"[a-z]", w[i]):
            terms.append([sign, w[i]])
            i += 1
        elif i < len(w) and w[i].isdigit() and i == len(w) - 1:
            c = sign * int(w[i])
            i += 1
        else:
            return None
    return (terms, c) if terms else None


def parse_sentence(s: str):
    for rx, (iv, im, ir) in _CONG_RE:
        m = rx.match(s)
        if m:
            return cong(m.group(iv), int(m.group(im)), int(m.group(ir))) | {"r": int(m.group(ir))}
    m = _DEF_RE.match(s)
    if m:
        p = parse_expr(m.group(2))
        if p:
            return {"kind": "def", "var": m.group(1), "terms": p[0], "c": p[1]}
    return None


def parse_question(q: str):
    """A node question -> ('cong', claim) or ('eq', definition-shaped claim); None otherwise."""
    m = _CONG_Q.match(q)
    if m:
        return "cong", {"kind": "cong", "var": m.group(1), "m": int(m.group(3)), "r": int(m.group(2))}
    m = _DEF_Q.match(q)
    if m:
        p = parse_expr(m.group(2))
        if p:
            return "eq", {"kind": "def", "var": m.group(1), "terms": p[0], "c": p[1]}
    return None


def parse_option(t: str):
    m = _OPTION.match(t)
    return {"kind": "cong", "var": m.group(1), "m": int(m.group(3)), "r": int(m.group(2))} if m else None


def parse_body(body: str) -> list[dict]:
    out = []
    for s in body.split("\n"):
        h = parse_sentence(s)
        if h is None:
            raise ValueError(f"unparsed premise {s!r}")
        out.append(h)
    return out


def strip_meta(h: dict) -> dict:
    """The logical content of a premise (no id / template / role)."""
    if h["kind"] == "cong":
        return {"kind": "cong", "var": h["var"], "m": h["m"], "r": h["r"]}
    return {"kind": "def", "var": h["var"], "terms": [list(t) for t in h["terms"]], "c": h["c"]}


# ------------------------------------------------------------------------------------------------------ semantics
def affine_forms(hyps: list[dict], extra_vars=()) -> tuple[list[str], dict]:
    """Free variables (no definition) and every variable as an affine form ({root: coef}, const) over them."""
    defs = {}
    for h in hyps:
        if h["kind"] == "def":
            if h["var"] in defs:
                raise ValueError(f"{h['var']} defined twice")
            defs[h["var"]] = h
    names = set(extra_vars)
    for h in hyps:
        names.add(h["var"])
        if h["kind"] == "def":
            names.update(u for _, u in h["terms"])
    roots = sorted(v for v in names if v not in defs)
    forms: dict = {}

    def form(v, stack=()):
        if v in forms:
            return forms[v]
        if v in stack:
            raise ValueError(f"cyclic definition through {v}")
        if v not in defs:
            forms[v] = ({v: 1}, 0)
            return forms[v]
        h = defs[v]
        co, k = collections.Counter(), h["c"]
        for a, u in h["terms"]:
            fu, ku = form(u, stack + (v,))
            for x, cx in fu.items():
                co[x] += a * cx
            k += a * ku
        forms[v] = ({x: c for x, c in co.items() if c}, k)
        return forms[v]
    for v in sorted(names):
        form(v)
    return roots, forms


def eval_form(f, vals: dict) -> int:
    return sum(c * vals[x] for x, c in f[0].items()) + f[1]


def evaluate(hyps: list[dict], root_vals: dict) -> dict:
    roots, forms = affine_forms(hyps)
    return {v: eval_form(forms[v], root_vals) for v in sorted(forms)}


def holds(h: dict, vals: dict) -> bool:
    if h["kind"] == "cong":
        return vals[h["var"]] % h["m"] == h["r"]
    return vals[h["var"]] == sum(a * vals[u] for a, u in h["terms"]) + h["c"]


def _grid(hyps: list[dict], extra: list[tuple], cap: int = GRID_CAP):
    """Exhaustive enumeration over one period (numpy). extra = [(form, m)] periodic predicates that must also be
    decided (claims). Returns (roots, forms, root arrays, sat mask). Each root runs over [0, P) with P the lcm of the
    moduli of every predicate whose form has a coefficient on it that is nonzero mod m (so every predicate depends on
    the root only through root mod P), filtered by the congruences stated directly on that root."""
    import numpy as np
    roots, forms = affine_forms(hyps, extra_vars=[x for f, _ in extra for x in f[0]])
    preds = [(forms[h["var"]], h["m"], h["r"]) for h in hyps if h["kind"] == "cong"]
    periodic = [(f, m) for f, m, _ in preds] + list(extra)
    axes = []
    for x in roots:
        P = 1
        for f, m in periodic:
            if f[0].get(x, 0) % m:
                P = lcm(P, m)
        direct = [(m, r) for f, m, r in preds if f == ({x: 1}, 0)]
        axes.append([v for v in range(P) if all(v % m == r for m, r in direct)])
    size = 1
    for a in axes:
        size *= max(1, len(a))
    if size > cap:
        raise OverflowError(f"grid {size} > {cap}")
    if any(not a for a in axes):
        return roots, forms, ({}, lambda f: np.zeros(0, dtype=np.int64)), np.zeros(0, dtype=bool)
    mesh = np.meshgrid(*[np.array(a, dtype=np.int64) for a in axes], indexing="ij") if axes else []
    arr = {x: g.ravel() for x, g in zip(roots, mesh)}
    n = size if axes else 1

    def val(f):
        out = np.full(n, f[1], dtype=np.int64)
        for x, c in f[0].items():
            out += c * arr[x]
        return out
    sat = np.ones(n, dtype=bool)
    for f, m, r in preds:
        sat &= (val(f) % m) == r
    return roots, forms, (arr, val), sat          # definitions hold by construction (affine forms)


def bf_status(hyps: list[dict], claim: dict, cap: int = GRID_CAP) -> str:
    """Exact status of a claim given hyps by brute force over one period: entailed / refuted / independent / unsat.
    A congruence claim is periodic; an equality claim ('eq', definition-shaped) is decided when its two sides differ
    by a constant once definitions are substituted (else 'undecided')."""
    if claim["kind"] == "cong":
        roots, forms = affine_forms(hyps, extra_vars=[claim["var"]])
        f = forms[claim["var"]]
        _, _, ev, sat = _grid(hyps, [(f, claim["m"])], cap)
        if not sat.any():
            return "unsat"
        c = (ev[1](f)[sat] % claim["m"]) == claim["r"]
    else:
        roots, forms = affine_forms(hyps, extra_vars=[claim["var"]] + [u for _, u in claim["terms"]])
        lhs = forms[claim["var"]]
        co = collections.Counter(lhs[0])
        k = lhs[1] - claim["c"]
        for a, u in claim["terms"]:
            for x, cx in forms[u][0].items():
                co[x] -= a * cx
            k -= a * forms[u][1]
        if any(co.values()):
            return "undecided"
        _, _, _, sat = _grid(hyps, [], cap)
        if not sat.any():
            return "unsat"
        return "entailed" if k == 0 else "refuted"
    if c.all():
        return "entailed"
    return "refuted" if not c.any() else "independent"


def sample_models(hyps: list[dict], rng: random.Random, n: int = N_MODELS, extra_vars=(), extra=()) -> list[dict]:
    """Up to n distinct satisfying assignments (dict var -> int, every variable): rows drawn at random from the exact
    one-period solution grid (_grid; extra = [(form, m)] periods to include, e.g. a claim to decide), each free variable
    then shifted by a random multiple (-2 .. 27) of its period, which keeps every premise true and varies the residues
    at other moduli. [] if unsatisfiable. Deterministic given rng."""
    import numpy as np
    roots, forms = affine_forms(hyps, extra_vars=list(extra_vars))
    _, _, (arr, val), sat = _grid(hyps, list(extra))
    idx = np.flatnonzero(sat)
    if not len(idx):
        return []
    period = {}
    for x in roots:
        P = 1
        for h in hyps:
            if h["kind"] == "cong":
                f = forms[h["var"]]
                if f[0].get(x, 0) % h["m"]:
                    P = lcm(P, h["m"])
        for f, m in extra:
            if f[0].get(x, 0) % m:
                P = lcm(P, m)
        period[x] = P
    out, seen = [], set()
    for _ in range(4 * n):
        if len(out) >= n:
            break
        i = int(idx[rng.randrange(len(idx))])
        rv = {x: int(arr[x][i]) + period[x] * rng.randrange(-2, 28) if x in arr else rng.randrange(-1, 8)
              for x in roots}
        vals = {v: eval_form(forms[v], rv) for v in sorted(forms)}
        key = tuple(sorted(vals.items()))
        if key not in seen:
            seen.add(key)
            out.append(vals)
    return out


def small_first(models: list[dict]) -> list[dict]:
    return sorted(models, key=lambda v: (max(abs(x) for x in v.values()), sum(abs(x) for x in v.values()),
                                         tuple(sorted(v.items()))))


# ------------------------------------------------------------------------------------------------------ closure
def closure(hyps: list[dict], cap: int = CLOSURE_CAP, max_rounds: int = 64) -> dict:
    """Round-based forward closure of the rule set: {(var, m): {r, depth, rule, parents}}; depth = round at which the
    claim is first derived = its shortest derivation (longest parent path); parents = premise ids ("h", id) and claims
    ("c", var, m). Rule priority within a round: subst, lift, add, combine, reduce (the first derivation is kept).
    ValueError on an inconsistency (a contradiction derived)."""
    known: dict = {}
    byvar: dict = collections.defaultdict(dict)       # var -> {m: r}
    for h in hyps:
        if h["kind"] == "cong":
            k = (h["var"], h["m"])
            if k in known:
                if known[k]["r"] != h["r"]:
                    raise ValueError(f"inconsistent premises on {k}")
                continue
            known[k] = {"r": h["r"], "depth": 0, "rule": "premise", "parents": [("h", h["id"])]}
            byvar[h["var"]][h["m"]] = h["r"]
    defs = sorted((h for h in hyps if h["kind"] == "def"), key=lambda h: h["var"])
    for t in range(1, max_rounds + 1):
        new: dict = {}

        def offer(v, m, r, rule, parents):
            k = (v, m)
            if k in known:
                if known[k]["r"] != r:
                    raise ValueError(f"contradiction on {k}")
                return
            if k in new:
                if new[k]["r"] != r:
                    raise ValueError(f"contradiction on {k}")
                return
            new[k] = {"r": r, "depth": t, "rule": rule, "parents": parents}
        snap = {v: dict(ms) for v, ms in byvar.items()}
        for h in defs:
            v, terms, c = h["var"], h["terms"], h["c"]
            if len(terms) == 1:
                a, u = terms[0]
                for m, r in sorted(snap.get(u, {}).items()):
                    offer(v, m, (a * r + c) % m, "subst", [("c", u, m), ("h", h["id"])])
                for m, r in sorted(snap.get(u, {}).items()):
                    if abs(a) >= 2 and abs(a) * m <= cap:
                        M = abs(a) * m
                        offer(v, M, (a * r + c) % M, "lift", [("c", u, m), ("h", h["id"])])
            else:
                (a, u), (b, w) = terms
                for m in sorted(set(snap.get(u, {})) & set(snap.get(w, {}))):
                    offer(v, m, (a * snap[u][m] + b * snap[w][m] + c) % m, "add", [("c", u, m), ("c", w, m),
                                                                                     ("h", h["id"])])
        for v in sorted(snap):
            ms = sorted(snap[v])
            for i, m1 in enumerate(ms):
                for m2 in ms[i + 1:]:
                    L = lcm(m1, m2)
                    if L > m2 and L <= cap:
                        cr = crt_pair(m1, snap[v][m1], m2, snap[v][m2])
                        if cr is None:
                            raise ValueError(f"contradiction combining {v} mod {m1}, {m2}")
                        offer(v, L, cr[1], "combine", [("c", v, m1), ("c", v, m2)])
        for v in sorted(snap):
            for M, R in sorted(snap[v].items()):
                for m in divisors(M):
                    offer(v, m, R % m, "reduce", [("c", v, M)])
        if not new:
            break
        for k in sorted(new):
            known[k] = new[k]
            byvar[k[0]][k[1]] = new[k]["r"]
    return known


def proof_of(known: dict, key: tuple) -> tuple[list[tuple], list[str]]:
    """(derived claims of the shortest proof of key, in depth order, key last; premise ids used, sorted)."""
    seen, prem, order = set(), set(), []

    def go(k):
        if k in seen:
            return
        seen.add(k)
        for p in known[k]["parents"]:
            if p[0] == "h":
                prem.add(p[1])
            else:
                go((p[1], p[2]))
        if known[k]["rule"] != "premise":
            order.append(k)
    go(key)
    order.sort(key=lambda k: (known[k]["depth"], k))
    return order, sorted(prem, key=lambda s: int(s[1:]))


def closure_label(known: dict, v: str, m: int, r: int):
    """entailed / refuted by the closure (sound), else None."""
    if (v, m) in known:
        return "entailed" if known[(v, m)]["r"] == r else "refuted"
    for (u, m0), k in known.items():
        if u == v:
            g = math.gcd(m, m0)
            if g > 1 and r % g != k["r"] % g:
                return "refuted"
    return None


# ------------------------------------------------------------------------------------------------------ generation
def _chain(depth: int, rng: random.Random, names) -> tuple[list[dict], dict]:
    hyps: list[dict] = []
    x0 = next(names)
    m0 = wchoice(rng, HYP_MODULI, HYP_WEIGHTS)
    hyps.append(dict(cong(x0, m0, rng.randrange(m0)), role="chain"))
    roots, cur, M = [x0], x0, m0
    chain = [{"var": x0, "m": m0, "level": 0}]
    steps = []
    for level in range(1, depth + 1):
        sub = [a for a in COEFS if math.gcd(a, M) == 1]
        lif = [a for a in (2, 3) if a * M <= MMAX]
        prevs = [c for c in chain[:-1] if c["m"] % M == 0 and c["var"] != cur]
        m2s = [m for m in HYP_MODULI if math.gcd(m, M) == 1 and M * m <= MMAX]
        kinds = [("subst", 30)] + ([("lift", 20)] if lif else []) + \
                ([("add_root", 15)] if len(roots) < MAX_CHAIN_ROOTS else []) + \
                ([("add_prev", 15)] if prevs else []) + ([("crt", 20)] if m2s else [])
        kind = wchoice(rng, [k for k, _ in kinds], [w for _, w in kinds])
        b = rng.choice([x for x in range(-12, 13) if x])
        if kind in ("subst", "lift"):
            a = rng.choice(sub if kind == "subst" else lif)
            v = next(names)
            hyps.append(dict(defn(v, [[a, cur]], b), role="chain"))
            if kind == "lift":
                M *= a
            cur = v
        elif kind in ("add_root", "add_prev"):
            a = rng.choice(sub)
            c = rng.choice([x for x in (1, 2, 3, 5) if math.gcd(x, M) == 1]) * rng.choice((1, -1))
            if kind == "add_root":
                w = next(names)
                roots.append(w)
                hyps.append(dict(cong(w, M, rng.randrange(M)), role="chain"))
            else:
                w = rng.choice(prevs)["var"]
            v = next(names)
            hyps.append(dict(defn(v, [[a, cur], [c, w]], b), role="chain"))
            cur = v
        else:                                                         # crt: a fresh congruence on cur
            m2 = rng.choice(m2s)
            ach = achievable_residues(hyps, cur, m2)
            if len(ach) < 2:
                return [], {"fail": "crt_forced"}
            hyps.append(dict(cong(cur, m2, rng.choice(ach)), role="chain"))
            M *= m2
        steps.append(kind)
        chain.append({"var": cur, "m": M, "level": level})
    return hyps, {"var": cur, "M": M, "steps": steps}


def achievable_residues(hyps: list[dict], v: str, m: int) -> list[int]:
    roots, forms = affine_forms(hyps, extra_vars=[v])
    f = forms[v]
    _, _, ev, sat = _grid(hyps, [(f, m)])
    return sorted({int(x) for x in (ev[1](f)[sat] % m).tolist()})


def _distractors(hyps: list[dict], n: int, rng: random.Random, names) -> list[dict]:
    have = sorted({h["var"] for h in hyps} | {u for h in hyps if h["kind"] == "def" for _, u in h["terms"]})
    out = []
    for _ in range(n):
        if rng.random() < 0.35:
            u = next(names)
            m = wchoice(rng, HYP_MODULI, HYP_WEIGHTS)
            out.append(dict(cong(u, m, rng.randrange(m)), role="distractor"))
        else:
            u = next(names)
            b = rng.choice([x for x in range(-12, 13) if x])
            src = rng.sample(have, 2) if len(have) >= 2 and rng.random() < 0.4 else [rng.choice(have)]
            terms = [[rng.choice(COEFS), src[0]]] + ([[rng.choice((1, 2, 3)) * rng.choice((1, -1)), src[1]]]
                                                      if len(src) == 2 else [])
            out.append(dict(defn(u, terms, b), role="distractor"))
        have.append(u)
    return out


def make_program(depth: int, rng: random.Random, n_irrelevant: int) -> tuple[dict | None, str]:
    names = iter(rng.sample(VARS, len(VARS)))
    chain, fin = _chain(depth, rng, names)
    if not chain:
        return None, fin["fail"]
    hyps = chain + _distractors(chain, n_irrelevant, rng, names)
    rng.shuffle(hyps)
    for i, h in enumerate(hyps, 1):
        h["id"] = f"h{i}"
        h["tpl"] = rng.randrange(len(CONG_TPL) if h["kind"] == "cong" else len(DEF_TPL))
    try:
        known = closure(hyps)
    except ValueError:
        return None, "inconsistent"
    z, M = fin["var"], fin["M"]
    cands = [m for m in divisors(M) + [M] if (z, m) in known and known[(z, m)]["depth"] == depth
             and known[(z, m)]["rule"] not in ("reduce", "premise")]
    if not cands:
        return None, "depth"
    m_key = wchoice(rng, cands, [2 if divisors(m) else 1 for m in cands])
    key = {"kind": "cong", "var": z, "m": m_key, "r": known[(z, m_key)]["r"]}
    order, prem = proof_of(known, (z, m_key))
    if len(prem) < depth + 1:
        return None, "few_premises"
    return {"hyps": hyps, "key": key, "depth": depth, "steps": fin["steps"], "final_modulus": M,
            "n_irrelevant": n_irrelevant, "critical": prem, "_known": known, "_order": order}, "ok"


# ------------------------------------------------------------------------------------------------------ certificates
def _binders(names) -> str:
    return f"({' '.join(names)} : Int)" if names else ""


def cert_forall(role: str, target: str, hyps: list[dict], goal: str, tactic: str = "omega") -> dict:
    """theorem NAME (vars : Int) (p1 : ...) ... : goal := by tactic, with the variables of hyps and goal only."""
    names = set(re.findall(r"\b[a-z]\b", goal))
    for h in hyps:
        names |= set(re.findall(r"\b[a-z]\b", lean_prop(h)))
    hs = " ".join(f"(p{i} : {lean_prop(h)})" for i, h in enumerate(hyps, 1))
    sig = " ".join(x for x in (_binders(sorted(names)), hs) if x)
    return {"role": role, "target": target, "tactic": tactic,
            "body": f"{sig} : {goal} := by {tactic}"}


def cert_exists(role: str, target: str, hyps: list[dict], extra: str, vals: dict) -> dict:
    """theorem NAME : ∃ vars : Int, hyps ∧ extra := ⟨values, by decide⟩ (a concrete model, checked by the kernel)."""
    names = sorted(vals)
    props = [lean_prop(h) for h in hyps] + [extra]
    return {"role": role, "target": target, "tactic": "decide",
            "body": f": ∃ {' '.join(names)} : Int, {' ∧ '.join(props)} := ⟨{', '.join(str(vals[x]) for x in names)}, "
                    f"by decide⟩", "witness": {x: vals[x] for x in names}}


def neg(p: str) -> str:
    return f"¬ ({p})"


def cert_line(name: str, cert: dict) -> str:
    return f"theorem {name} {cert['body']}"


LEAN_HEADER = "set_option linter.unusedVariables false\nset_option maxHeartbeats 400000\n"
_AXIOMS = re.compile(r"^'([^']+)' (?:depends on axioms: \[(.*)\]|does not depend on any axioms)\s*$", re.S)


def lean_file(entries: list[tuple[str, dict]]) -> tuple[str, dict]:
    """File text for [(name, cert)] (one theorem per line, then #print axioms on the next) and line -> name."""
    lines = LEAN_HEADER.rstrip("\n").split("\n")
    where = {}
    for name, c in entries:
        lines.append(cert_line(name, c))
        where[len(lines)] = name
        lines.append(f"#print axioms {name}")
        where[len(lines)] = name
    return "\n".join(lines) + "\n", where


def parse_lean_output(text: str, where: dict) -> tuple[dict, list[str]]:
    """Per-declaration results from `lean --json` output: {name: {"errors": [...], "axioms": [...] | None}} and the
    lines that are not JSON messages. A declaration is certified (certified()) only with no error on its lines and an
    axioms report within ALLOWED_AXIOMS (fail-closed: no report, no certificate)."""
    res = {n: {"errors": [], "axioms": None} for n in sorted(set(where.values()))}
    other = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            other.append(line)
            continue
        sev, data = msg.get("severity"), str(msg.get("data", ""))
        name = where.get((msg.get("pos") or {}).get("line"))
        if sev == "error":
            if name is None:
                other.append(line)
            else:
                res[name]["errors"].append(data[:300])
        elif sev == "information":
            m = _AXIOMS.match(data)
            if m and m.group(1) in res:
                ax = [a.strip() for a in (m.group(2) or "").split(",") if a.strip()]
                res[m.group(1)]["axioms"] = ax
    return res, other


def certified(rec: dict | None) -> bool:
    return bool(rec) and not rec["errors"] and rec["axioms"] is not None and set(rec["axioms"]) <= ALLOWED_AXIOMS


# ------------------------------------------------------------------------------------------------------ item
def _twin_r(m: int, r: int, rng: random.Random, pnums: set | None = None) -> int:
    """Another residue mod m; for a leaf twin (pnums = the numbers written in the premises) one that also occurs in
    the premises when there is one, so that "the number is in the text" does not predict Yes."""
    alt = [x for x in range(m) if x != r]
    if pnums is not None:
        alt = [x for x in alt if str(x) in pnums] or alt
    return rng.choice(alt)


def _twin_c(c: int, rng: random.Random, pnums: set | None = None) -> int:
    """Another constant of the same sign, the same number of digits when possible (no length cue), written in the
    premises when possible."""
    alt = [c + d for d in (-3, -2, -1, 1, 2, 3) if c + d != 0 and (c + d > 0) == (c > 0)]
    same = [x for x in alt if len(str(abs(x))) == len(str(abs(c)))]
    seen = [x for x in alt if pnums is not None and str(abs(x)) in pnums]
    return rng.choice([x for x in same if x in seen] or seen or same or alt)


def tree_nodes(prog: dict, rng: random.Random) -> list[dict]:
    """subqs: the key's shortest proof without the key; Yes nodes and refuted No twins (same parents)."""
    known, order, key = prog["_known"], prog["_order"], prog["key"]
    hyp = {h["id"]: h for h in prog["hyps"]}
    nodes, cid = [], {}
    pnums = set(re.findall(r"\d+", body_text(prog["hyps"])))
    for k, rec in known.items():                       # a premise congruence used as a parent claim
        if rec["rule"] == "premise":
            cid[("c",) + k] = rec["parents"][0][1]
    for hid in prog["critical"]:
        h = hyp[hid]
        cid[("h", hid)] = hid
        if h["kind"] == "cong":
            tw = dict(strip_meta(h), r=_twin_r(h["m"], h["r"], rng, pnums))
            typ, kind = "read", "congruence"
        else:
            tw = dict(strip_meta(h), c=_twin_c(h["c"], rng, pnums))
            typ, kind = "parse", "definition"
        q, qx = question_of(h), question_of(tw)
        common = {"p_fresh": None, "type": typ, "depends_on": [], "initial": None, "status": "direct",
                  "local_lines": [sentence(h)], "kind": kind, "level": 0, "node_depth": 0}
        nodes.append({"question": q, "p_cot": 1.0, "truth": True, "id": hid, "opposite": qx,
                      "claim": strip_meta(h), "lean": lean_prop(h), **common})
        nodes.append({"question": qx, "p_cot": 0.0, "truth": False, "id": hid + "x", "opposite": q,
                      "claim": tw, "lean": lean_prop(tw), **common})
    kk = (key["var"], key["m"])
    for j, k in enumerate([k for k in order if k != kk], 1):
        cid[("c",) + k] = f"n{j}"
    for k in order:
        if k == kk:
            continue
        rec = known[k]
        c = {"kind": "cong", "var": k[0], "m": k[1], "r": rec["r"]}
        tw = dict(c, r=_twin_r(k[1], rec["r"], rng))
        deps = [cid[tuple(p)] for p in rec["parents"]]
        q, qx = question_of(c), question_of(tw)
        nid = cid[("c",) + k]
        common = {"p_fresh": None, "type": "derive", "depends_on": deps, "initial": None, "status": "direct",
                  "kind": rec["rule"], "level": rec["depth"], "node_depth": rec["depth"]}
        nodes.append({"question": q, "p_cot": 1.0, "truth": True, "id": nid, "opposite": qx, "claim": c,
                      "lean": lean_prop(c), **common})
        nodes.append({"question": qx, "p_cot": 0.0, "truth": False, "id": nid + "x", "opposite": q, "claim": tw,
                      "lean": lean_prop(tw), **common})
    prog["_cid"] = cid
    return nodes


def _parent_hyps(prog: dict, parents) -> list[dict]:
    hyp = {h["id"]: h for h in prog["hyps"]}
    out = []
    for p in parents:
        if p[0] == "h":
            out.append(strip_meta(hyp[p[1]]))
        else:
            out.append({"kind": "cong", "var": p[1], "m": p[2], "r": prog["_known"][(p[1], p[2])]["r"]})
    return out


def option_pools(prog: dict, models: list[dict], nodes: list[dict]) -> dict:
    """Candidate options by class (module docstring), each {var, m, r, cls, subtype?}."""
    known, key = prog["_known"], prog["key"]
    z, M, R = key["var"], key["m"], key["r"]
    premises = {claim_key(strip_meta(h)) for h in prog["hyps"] if h["kind"] == "cong"}
    excluded = {claim_key(n["claim"]) for n in nodes if n["type"] == "derive" or (n["type"] == "read" and not n["truth"])}
    excluded.add((z, M, R))
    pools = collections.defaultdict(list)
    for r in range(M):
        if r != R and (z, M, r) not in excluded:
            pools["on_refuted"].append({"var": z, "m": M, "r": r})
    allvars = sorted({h["var"] for h in prog["hyps"]} | {u for h in prog["hyps"] if h["kind"] == "def"
                                                         for _, u in h["terms"]})

    def classify(v, m):
        cnt = collections.Counter(mv[v] % m for mv in models)
        for r in range(m):
            k = (v, m, r)
            lab = closure_label(known, v, m, r)
            if k in excluded and not (lab == "entailed" and k in premises):
                continue
            yield r, lab, (0 < cnt[r] < len(models))
    zmods = sorted(({m for (u, m) in known if u == z and m <= MMAX} | set(OFF_MODULI)) - {M})
    for m in zmods:
        if m % M == 0:
            continue
        for r, lab, ind in classify(z, m):
            c = {"var": z, "m": m, "r": r}
            if lab == "entailed":
                pools["drift"].append(dict(c, subtype="b" if (z, m, r) in premises else "c", nz=False))
            elif lab == "refuted":
                pools["z_refuted"].append(c)
            elif ind:
                pools["z_indep"].append(c)
    for v in allvars:
        if v == z:
            continue
        for m in sorted({m for (u, m) in known if u == v and m <= MMAX} | set(NZ_MODULI)):
            for r, lab, ind in classify(v, m):
                c = {"var": v, "m": m, "r": r}
                if lab == "entailed":
                    pools["drift"].append(dict(c, subtype="b" if (v, m, r) in premises else "a", nz=True))
                elif lab == "refuted":
                    pools["nz_refuted"].append(c)
                elif ind:
                    pools["nz_indep"].append(c)
    return pools


def premise_tokens(prog: dict) -> set:
    return set(re.findall(r"[a-z]+|\d+", body_text(prog["hyps"]).lower()))


def overlap(text: str, ptoks: set) -> float:
    t = re.findall(r"[a-z]+|\d+", text.lower())
    return sum(x in ptoks for x in t) / len(t) if t else 0.0


def rank_span(vals: list, i: int) -> tuple[int, int]:
    """(best, worst) rank of vals[i] when higher is rank 1, ties counted either way."""
    c = vals[i]
    others = [x for j, x in enumerate(vals) if j != i]
    return 1 + sum(x > c for x in others), 1 + sum(x >= c for x in others)


def form_class(o: dict) -> tuple:
    """Surface form of an option (fa_trees.option_form's analogue for the one template): remainder 0 or not, digits
    of the divisor, digits of the remainder."""
    return (o["r"] == 0, len(str(o["m"])), len(str(o["r"])))


def form_odd_one_out(opts: list[dict]) -> bool:
    """Exactly one option differs from the other four in its form class (fa_trees form_tell)."""
    cnt = collections.Counter(form_class(o) for o in opts)
    return len(cnt) == 2 and min(cnt.values()) == 1


def choose_options(prog: dict, pools: dict, plan: dict, rng: random.Random, tries: int = 1000):
    """The 4 distractors for a plan {n_on, drift, n_drift, non_z, t_len, t_ov}: n_on on-target refuted, one off-target
    independent, the drift options (drift items) or one non-z false option (non-drift items with non_z), the rest
    off-target false; at most one option about another variable; distinct (var, divisor) among off-target options;
    the key's length rank, premise-overlap rank and divisor rank (ties either way) hit the seeded targets; no form
    odd-one-out."""
    key = prog["key"]
    ptoks = premise_tokens(prog)
    n_on, k_off = plan["n_on"], 4 - plan["n_on"]
    drift_z = [c for c in pools.get("drift", []) if not c["nz"]]
    drift_nz = [c for c in pools.get("drift", []) if c["nz"]]
    slots = ["on_refuted"] * n_on
    if plan["drift"]:
        nd = min(plan["n_drift"], k_off - 1)
        nz_slot = plan["non_z"] and bool(drift_nz)
        if not drift_z and not drift_nz:
            return None, "no_drift_candidates"
        zs = nd - (1 if nz_slot else 0)
        if zs and not drift_z:
            if not drift_nz:
                return None, "no_drift_candidates"
            nz_slot, zs = True, nd - 1
            if zs:
                nd, zs = 1, 0
        slots += ["drift_nz"] * (1 if nz_slot else 0) + ["drift_z"] * zs + ["z_indep"]
    else:
        nz_slot = plan["non_z"] and bool(pools.get("nz_refuted") or pools.get("nz_indep"))
        slots += ["z_indep"] + (["nz_false"] if nz_slot else [])
    slots += ["z_false"] * (4 - len(slots))
    pool_of = {"on_refuted": [dict(c, cls="refuted", target="on") for c in pools.get("on_refuted", [])],
               "z_indep": [dict(c, cls="independent", target="off") for c in pools.get("z_indep", [])],
               "drift_z": [dict(c, cls="drift_true", target="off") for c in drift_z],
               "drift_nz": [dict(c, cls="drift_true", target="off") for c in drift_nz],
               "nz_false": [dict(c, cls="refuted", target="off") for c in pools.get("nz_refuted", [])]
               + [dict(c, cls="independent", target="off") for c in pools.get("nz_indep", [])],
               "z_false": [dict(c, cls="refuted", target="off") for c in pools.get("z_refuted", [])]
               + [dict(c, cls="independent", target="off") for c in pools.get("z_indep", [])]}
    if any(len(pool_of[s]) < slots.count(s) for s in set(slots)):
        return None, "pool_" + next(s for s in sorted(set(slots)) if len(pool_of[s]) < slots.count(s))
    keyo = dict(key, cls="key", target="on")
    for _ in range(tries):
        picked, used, ok = [], set(), True
        for s in slots:
            cand = [c for c in pool_of[s] if (c["var"], c["m"], c["r"]) not in used
                    and (s == "on_refuted" or (c["var"], c["m"]) not in {(p["var"], p["m"]) for p in picked
                                                                        if p["target"] == "off"})]
            if not cand:
                ok = False
                break
            c = rng.choice(cand)
            picked.append(c)
            used.add((c["var"], c["m"], c["r"]))
        if not ok:
            continue
        opts = [keyo] + picked
        texts = [option_text(o) for o in opts]
        lr = rank_span([len(t) for t in texts], 0)
        orr = rank_span([overlap(t, ptoks) for t in texts], 0)
        mr = rank_span([-o["m"] for o in opts], 0)                  # rank 1 = smallest divisor
        # ties counted either way (fa_trees target_overlap_rank). The key always ties with the on-target refuted
        # options in divisor (and often in length / overlap), so "pick the extreme" oracles that split ties sit a bit
        # below 1/5 by construction; build_lean_tf.py reports them with their "avoid the extreme" counterparts
        if (lr[0] <= plan["t_len"] <= lr[1] and orr[0] <= plan["t_ov"] <= orr[1] and mr[0] <= plan["t_mod"] <= mr[1]
                and not form_odd_one_out(opts)):
            return opts, "ok"
    return None, "rank_targets"


def build_item(depth: int, seed_parts: tuple, plan: dict, max_tries: int = 400) -> tuple[dict | None, collections.Counter]:
    """One certified-to-be item (before Lean): program, tree, options, certificates. plan: {drift, n_drift, n_on,
    non_z, n_irrelevant}; rank targets are drawn here. Returns (item, rejection counts)."""
    rej = collections.Counter()
    if "t_len" not in plan:                            # rank targets fixed per item slot (not per attempt)
        trng = seeded(*seed_parts, "targets")
        plan = dict(plan, t_len=trng.randint(1, 5), t_ov=trng.randint(1, 5), t_mod=trng.randint(1, 5))
    for attempt in range(max_tries):
        rng = seeded(*seed_parts, attempt)
        prog, why = make_program(depth, rng, plan["n_irrelevant"])
        if prog is None:
            rej[why] += 1
            continue
        try:
            item, why = _finish(prog, plan, rng)
        except OverflowError:
            item, why = None, "grid_cap"
        if item is None:
            rej[why] += 1
            continue
        item["gen"] = {"attempt": attempt, "rejections": dict(rej)}
        return item, rej
    return None, rej


def _finish(prog: dict, plan: dict, rng: random.Random):
    hyps = [strip_meta(h) for h in prog["hyps"]]
    key = prog["key"]
    models = small_first(sample_models(hyps, rng))
    if len(models) < 10:
        return None, "few_models"
    hyp = {h["id"]: h for h in prog["hyps"]}
    necessity = {}
    for hid in prog["critical"]:
        rest = [strip_meta(h) for h in prog["hyps"] if h["id"] != hid]
        _, fr = affine_forms(rest, extra_vars=[key["var"]])
        ce = [m for m in sample_models(rest, rng, n=200, extra_vars=[key["var"]], extra=[(fr[key["var"]], key["m"])])
              if not holds(key, m)]
        if not ce:
            return None, "padding"
        necessity[hid] = small_first(ce)[0]
    nodes = tree_nodes(prog, rng)
    pools = option_pools(prog, models, nodes)
    opts, why = choose_options(prog, pools, plan, rng)
    if opts is None:
        return None, why
    rest = opts[1:]
    rng.shuffle(rest)
    g = plan["gold_pos"] if plan.get("gold_pos") is not None else rng.randrange(5)   # balanced by the schedule
    opts = rest[:g] + [opts[0]] + rest[g:]
    gold = LETTERS[[o["cls"] for o in opts].index("key")]
    # certificates
    certs = []
    allh = [strip_meta(h) for h in prog["hyps"]]
    for L, o in zip(LETTERS, opts):
        c = {"kind": "cong", "var": o["var"], "m": o["m"], "r": o["r"]}
        p = lean_prop(c)
        if o["cls"] == "key":
            certs.append(cert_forall("key/omega", L, allh, p, "omega"))
            certs.append(cert_forall("key/grind", L, allh, p, "grind"))
        elif o["cls"] == "drift_true":
            certs.append(cert_forall("drift_true/omega", L, allh, p, "omega"))
        elif o["cls"] == "refuted":
            certs.append(cert_forall("refuted/omega", L, allh, neg(p), "omega"))
        else:
            yes = [m for m in models if holds(c, m)]
            no = [m for m in models if not holds(c, m)]
            if not yes or not no:
                return None, "indep_witness"
            certs.append(cert_exists("independent/holds", L, allh, p, yes[0]))
            certs.append(cert_exists("independent/fails", L, allh, neg(p), no[0]))
    kp = lean_prop(key)
    for hid in prog["critical"]:
        rest = [strip_meta(h) for h in prog["hyps"] if h["id"] != hid]
        certs.append(cert_exists("necessary", hid, rest, neg(kp), necessity[hid]))
    known = prog["_known"]
    for n in nodes:
        if n["type"] == "derive":
            k = (n["claim"]["var"], n["claim"]["m"])
            par = _parent_hyps(prog, known[k]["parents"])
        else:
            par = [strip_meta(hyp[n["id"].rstrip("x")])]
        goal = n["lean"] if n["truth"] else neg(n["lean"])
        certs.append(cert_forall("node/yes" if n["truth"] else "node/no", n["id"], par, goal, "omega"))
    kk = (key["var"], key["m"])
    certs.append(cert_forall("final/parents", "key", _parent_hyps(prog, known[kk]["parents"]), kp, "omega"))
    drift = [o for o in opts if o["cls"] == "drift_true"]
    item = {
        "source": SOURCE, "prompt": prompt_of(prog, opts), "labels": list(LETTERS), "gold_label": gold,
        "options": {L: option_text(o) for L, o in zip(LETTERS, opts)},
        "option_kinds": {L: o["cls"] for L, o in zip(LETTERS, opts)},
        "option_meta": {L: {"var": o["var"], "m": o["m"], "r": o["r"], "target": o["target"],
                            **({"tag": "off_target", "subtype": o["subtype"]} if o["cls"] == "drift_true" else {})}
                        for L, o in zip(LETTERS, opts)},
        "depth": prog["depth"], "drift": bool(drift), "n_drift": len(drift),
        "drift_subtypes": sorted(o["subtype"] for o in drift), "n_on_target": 1 + sum(o["target"] == "on"
                                                                                     and o["cls"] != "key" for o in opts),
        "n_irrelevant": prog["n_irrelevant"], "steps": prog["steps"],
        "targets": {"length_rank": plan["t_len"], "overlap_rank": plan["t_ov"], "divisor_rank": plan["t_mod"]},
        "plan": {k: plan.get(k) for k in ("drift", "n_drift", "n_on", "non_z", "n_irrelevant", "gold_pos")},
        "program": {"hyps": [{k: h[k] for k in ("id", "kind", "var", "m", "r", "terms", "c", "tpl", "role") if k in h}
                             for h in prog["hyps"]],
                    "key": key, "critical": prog["critical"], "final_modulus": prog["final_modulus"]},
        "formal": formal_view(prog, opts),
        "subqs": nodes,
        "certificates": certs,
    }
    return item, "ok"


def formal_view(prog: dict, opts: list[dict]) -> dict:
    names = sorted({x for h in prog["hyps"] for x in re.findall(r"\b[a-z]\b", lean_prop(h))})
    hs = " ".join(f"({h['id']} : {lean_prop(h)})" for h in prog["hyps"])
    return {"statement": f"theorem lean_f1 ({' '.join(names)} : Int) {hs} : {lean_prop(prog['key'])}",
            "hyps": {h["id"]: lean_prop(h) for h in prog["hyps"]},
            "options": {L: lean_prop({"kind": "cong", **{k: o[k] for k in ("var", "m", "r")}})
                        for L, o in zip(LETTERS, opts)}}


def canonical(item: dict) -> tuple:
    """Logical identity: the premise set (Lean text) and the key; templates, order and letters ignored."""
    return (tuple(sorted(item["formal"]["hyps"].values())), lean_prop(item["program"]["key"]))


def finalize(item: dict, item_id: str, kind: str, group: str | None = None) -> dict:
    """Split-specific fields: train / val (tree fields as data/student_pronto_tf) or eval (as data/eval/pronto_d*)."""
    it = {"item_id": item_id, **{k: v for k, v in item.items() if k not in ("subqs", "gen")}}
    if kind == "train":
        it["teacher"] = {L: 1.0 if L == it["gold_label"] else 0.0 for L in it["labels"]}
        it["subqs"] = item["subqs"]
        it["has_tree"] = bool(item["subqs"])
        it["tree_source"] = TREE_SOURCE
        it["local_preamble"] = PREAMBLE
        it["tree_fix"] = True
    else:
        it.update({"group": group, "type": "choice", "label_order": list(LETTERS), "gold_probs": None})
    return it


# ------------------------------------------------------------------------------------------------------ text audit
LOCAL_HEADER = "Known results:"
_FACTLINE = re.compile(r"^Q: (.+) A: (Yes|No)$")


def solve_view(ctx: str, question: str):
    """Answer a node question from a rendered local view's TEXT alone (preamble + one premise line, or preamble +
    "Known results:" Q/A lines): True / False when the stated facts entail / refute it (brute force), else None."""
    pre, _, rest = ctx.partition("\n\n")
    if pre != PREAMBLE:
        return None
    facts = []
    if rest.startswith(LOCAL_HEADER + "\n"):
        for line in rest[len(LOCAL_HEADER) + 1:].split("\n"):
            m = _FACTLINE.match(line)
            if not m or m.group(2) != "Yes":
                return None
            p = parse_question(m.group(1))
            if p is None:
                return None
            facts.append(p[1])
    else:
        for s in rest.split("\n"):
            h = parse_sentence(s)
            if h is None:
                return None
            facts.append(h)
    p = parse_question(question)
    if p is None:
        return None
    try:
        st = bf_status(facts, p[1])
    except (OverflowError, ValueError):
        return None
    return {"entailed": True, "refuted": False}.get(st)
