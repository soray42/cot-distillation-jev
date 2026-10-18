"""GSM8K value nodes for Tree Forcing: one node per ATOMIC operation of GSM8K's own calculator notes, asked as a
three-option value question ("What is 4 × 1.5?  A) 6  B) 5.5  C) 2.5"), with the Socratic sub-question and an
operation-selection variant as alternative renderings (arms GFPQV / GFPQVQ / GFPQO / GFPVB / GFPQVM in
hpc/train_arm.sbatch; student.build_full_tree --node-format value / yesno+value).

Rows: every row of data/student_gsm8k_tf/{train,val}.jsonl (scripts/build_gsm8k_tf.py), byte for byte the same
fields (finals, options, gold, the yes/no tree in `subqs`, has_tree, ...), plus ONE new field `value_nodes`: the
value-node list. The yes/no tree stays in `subqs` so the yes/no arms (GFPQ, ...) train exactly as on
data/student_gsm8k_tf and GFPQVM can ask both formats in one problem group; the value arms read `value_nodes` only.
Subsets data/student_gsm8k_tf_ops_f{10,25,50}: the same problem ids, in the same order, as
data/student_gsm8k_tf_f{10,25,50}/train.jsonl; their val is the full val (as there).

Source of the operations: the <<expr=value>> notes of GSM8K's socratic solutions (data/raw/bench_math/gsm8k/socratic,
parsed by build_gsm8k_socratic.parse_problem exactly as build_gsm8k_tf does; each row's problem must reproduce its
prompt, gold and step count). Each note's expression is split into binary operations in evaluation order (a post-order
walk of its Python AST: precedence, then left to right, parentheses respected), e.g. "4*1.5+3" -> "4 × 1.5 = 6",
"6 + 3 = 9"; "4+6+8+2" -> "4 + 6 = 10", "10 + 8 = 18", "18 + 2 = 20". All arithmetic is exact (fractions.Fraction of
the literals as written), numbers are printed exactly (integers without ".0", decimals without float noise or trailing
zeros, a leading "0." for ".5"; no thousands separators, as in the notes), at most MAX_DEC decimals. A percentage is
rendered "P% of X" only when the step's sentence or the problem writes "P%" / "P percent" and the note multiplies X by
P/100 in one of the forms P*.01, P/100, (X*P)*.01, (X*P)/100 or a decimal literal equal to P/100 (one node instead of
two: "What is 50% of 4?"); otherwise the literal operations are asked ("What is 20 × 0.01?"). A fraction p/q of
two integer literals that multiplies something is one number, as written ("1/2*4" -> "What is 1/2 × 4?"); a division
note on its own stays a division ("<<1/2=.5>>" -> "What is 1 ÷ 2?").

A node (`value_nodes` entry, id s<step>o<k>, k = 1-based index of the operation within its step's note):
  step, op_index, n_ops, last_of_step, expr (the note), subq (the Socratic sub-question, ask_text), kind bin / pct,
  a, op (+ − × ÷, or "% of"), b, value (exact strings; pct: a = P, b = X), depends_on (built nodes whose results are an
  operand: an earlier operation of the same note, or the last operation of a parent step through
  parse_problem's certain edges), node_depth (operation-level dependency depth over every operation of the problem,
  built or not), gold_ancestor, gold_step, flags, percent (pct nodes: written_in sentence / problem, form compound
  (P*.01, P/100, ...) / decimal (a literal equal to P/100)), and `renderings`:
    op     "What is 4 × 1.5?" / "What is 50% of 4?"
    qop    "<sub-question> It is 4 × 1.5. What is the value?" for the operation that yields the step's value;
           "<sub-question> It is 4 × 1.5 + 3. What is 4 × 1.5?" for an earlier operation of a multi-operation note
           (the note with the operations before it reduced to their values); never for the gold step's earlier
           operations ("<final question> It is 4 + 6 + 8 + 2. What is 4 + 6?" would state the final answer's
           computation: the final question in disguise)
    opsel  operation selection (last operation of a step only): "<sub-question> Which computation gives it?" with three
           expressions as options
  each {question, options (3 strings), answer (index), wrong (the kind of each wrong option)}. op and qop share their
  options and answer (one draw); opsel has its own. qop and opsel exist only where the sub-question is true as worded
  for the step (not for a step whose sub-question text is shared with another step of a different value or equals the
  problem's final question while its value is not the gold: build_gsm8k_tf drops such yes/no nodes) and, for opsel,
  where enough wrong expressions exist.
Value options (op, qop): the true value and two wrong values of the same kind (positive, an integer if the truth is
one, at most max(2, decimals of the truth) decimals, within 10x of the truth, never the problem's final answer, never
equal to the truth or to each other as printed), one per tier in this order of preference: (1) "operator": the same
operands with another operator (a+b, a−b, b−a, a×b, a÷b, b÷a), or "complement": a fractional factor f of a × / ÷
replaced by 1−f (f < 1) or f−1 (1 < f < 2) (pct: (100−P)% and (100+P)% of X); (2) "other": another operation's value
of the same problem; (3) "perturb": the truth moved keeping parity and roundness (build_gsm8k_tf.perturbs), the side
drawn 50/50. The first wrong comes from the most preferred tier available, the second from the next tier available
(the same tier when no other one has a candidate), so the truth's rank among the options does not follow from the
operator (with two operator slips, "+" would always be the middle value). Option order: a seeded shuffle
(random.Random("<seed>/<problem_id>/<node id>/value"), "/opsel" for opsel), one-hot target on the truth.
Operation-selection options (opsel): the true expression and two wrong expressions whose values differ from the
truth's, both of one design so the truth is not the option that shares the operator with one wrong option and the
operands with the other: "operator" (the same operands, two other operators) or "operand" (the same operator, the
same operand position replaced by a number of the problem text or an earlier step's value, never the gold), the design
drawn 50/50 where both exist; "mixed" (one of each) only where neither design exists.
Not built (counted in summary.json build):
  - every operation of a step build_gsm8k_tf blocks (a gold-valued step other than the gold step, steps downstream of
    one, steps with an uncertain edge to one) and of the steps after the gold step; the gold step's operations when the
    gold step reads a blocked step; the gold step's last operation (it yields the final answer: final_op); the gold
    step's earlier operations are built (they are partial sums / products, not the answer);
  - an operation whose value equals the gold (and the later operations of its note that use it);
  - trivial operations (x × 1, 1 × x, x ÷ 1, x + 0, 0 + x, x − 0, 100% of x); operations with an unsupported operator
    (//, %, **), a negative or inexact operand, a non-positive value or one with more than MAX_DEC decimals or a
    non-terminating decimal; operations without two wrong values.
Audit (audit_item, from the stored fields only): every node's operation re-evaluated exactly (Fraction) equals its
value; the last operation of every built step equals the exact value of the whole note; question texts, option
uniqueness, truth at the answer index, no value option equal to the gold, opsel wrong options of a different value;
depends_on / node_depth consistent; the rows' other fields equal data/student_gsm8k_tf's. node_audit.txt shows 5
problems (gsm8k-train-660 first) with all their nodes and renderings.

  python3 scripts/build_gsm8k_ops.py      # pandas (system python3); writes data/student_gsm8k_tf_ops*/
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import random
import re
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_gsm8k_tf as T  # noqa: E402

FIELD = "value_nodes"
RENDERINGS = ("op", "qop", "opsel")
MAX_DEC = 4
MINUS, TIMES, DIVIDE = "−", "×", "÷"
SYM = {ast.Add: "+", ast.Sub: MINUS, ast.Mult: TIMES, ast.Div: DIVIDE}
PREC = {ast.Add: 1, ast.Sub: 1, ast.Mult: 2, ast.Div: 2}
ATOM = 9
FN = {"+": lambda a, b: a + b, MINUS: lambda a, b: a - b, TIMES: lambda a, b: a * b,
      DIVIDE: lambda a, b: a / b if b else None}
TIERS = ("operator", "other", "perturb")
FRACTIONS = (10, 25, 50)
PCT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)")
NUMTOK = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?")


# ------------------------------------------------------------------------------------------------------ numbers
def fmt(x: Fraction | None) -> str | None:
    """Exact decimal text of x (no trailing zeros, '0.5' not '.5', integers without '.0'); None when x needs more than
    MAX_DEC decimals or does not terminate."""
    if x is None:
        return None
    for k in range(MAX_DEC + 1):
        y = x * 10 ** k
        if y.denominator == 1:
            n = y.numerator
            if k == 0:
                return str(n)
            s = f"{abs(n):0{k + 1}d}"
            return ("-" if n < 0 else "") + s[:-k] + "." + s[-k:]
    return None


def num(s: str) -> Fraction:
    return Fraction(s.replace(",", ""))


def lit(c: ast.Constant) -> Fraction:
    return Fraction(str(c.value)) if isinstance(c.value, float) else Fraction(c.value)


def decimals(x: Fraction) -> int:
    t = fmt(x)
    return len(t.partition(".")[2]) if t else MAX_DEC + 1


def written_percents(*texts: str) -> set:
    return {Fraction(m) for t in texts for m in PCT.findall(t.lower())}


def is_const(n, v=None) -> bool:
    return isinstance(n, ast.Constant) and not isinstance(n.value, bool) and (v is None or lit(n) == v)


# ------------------------------------------------------------------------------------------------------ splitting
def frac_literal(n) -> str | None:
    """'p/q' when n is a fraction written as p/q of two positive integer literals whose value is not an integer (an
    operand of a multiplication: '1/2*4', '36*(1/3)'), else None."""
    if (isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) and is_const(n.left) and is_const(n.right)
            and isinstance(n.left.value, int) and isinstance(n.right.value, int) and n.left.value > 0
            and n.right.value > 0 and n.left.value % n.right.value):
        return f"{n.left.value}/{n.right.value}"
    return None


def split_note(tree: ast.AST, pcts: set, refs: dict) -> tuple[list[dict], dict, dict]:
    """The binary operations of one parsed note in evaluation order (post-order), the descriptor of its root and the
    fraction literals (id of the AST node -> 'p/q'; a fraction p/q multiplying something is a number, not an operation).
    An operation: {k, kind bin / pct, sym, a, b (operand descriptors), val (Fraction or None), node (the AST node),
    skip (None or a reason)}. An operand descriptor: {val, text, op (index of the operation it is the result of, or
    None), ref (parent step index through parse_problem's certain edges, or None), neg}."""
    ops: list[dict] = []

    def factor(n) -> Fraction | None:
        """P when n is the factor P/100 of a written percentage P."""
        if is_const(n):
            v = lit(n)
            return v * 100 if v.denominator != 1 and v > 0 and v * 100 in pcts else None
        if isinstance(n, ast.BinOp) and is_const(n.left) and is_const(n.right):
            a, b = lit(n.left), lit(n.right)
            if isinstance(n.op, ast.Mult) and b == Fraction(1, 100) and a in pcts:
                return a
            if isinstance(n.op, ast.Mult) and a == Fraction(1, 100) and b in pcts:
                return b
            if isinstance(n.op, ast.Div) and b == 100 and a in pcts:
                return a
        return None

    def percent(n: ast.BinOp):
        """(P, the other operand's AST node) when n multiplies by a written percentage, else None."""
        if isinstance(n.op, ast.Mult):
            for f, o in ((n.left, n.right), (n.right, n.left)):
                P = factor(f)
                if P is not None and factor(o) is None:
                    return P, o, "decimal" if is_const(f) else "compound"
        inner = n.left
        if (isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Mult)
                and (isinstance(n.op, ast.Mult) and is_const(n.right, Fraction(1, 100))
                     or isinstance(n.op, ast.Div) and is_const(n.right, 100))):
            for c, o in ((inner.right, inner.left), (inner.left, inner.right)):    # (X * P) * .01, (X * P) / 100
                if is_const(c) and lit(c) in pcts:
                    return lit(c), o, "compound"
        return None

    def desc(v, op=None, ref=None):
        return {"val": v, "text": fmt(v) if v is not None else None, "op": op, "ref": ref,
                "neg": v is not None and v < 0}

    def emit(kind, sym, a, b, v, node, skip=None, P=None, other=None):
        ops.append({"k": len(ops), "kind": kind, "sym": sym, "a": a, "b": b, "val": v, "node": node, "skip": skip,
                    "P": P, "pct_other": other})
        return desc(v, op=len(ops) - 1)

    lits: dict = {}

    def walk(n):
        if isinstance(n, ast.Constant):
            return desc(lit(n), ref=refs.get(id(n)))
        if id(n) in lits:
            return dict(desc(Fraction(lits[id(n)])), text=lits[id(n)])
        if isinstance(n, ast.UnaryOp):
            d = walk(n.operand)
            if isinstance(n.op, ast.USub):
                d = dict(d, val=-d["val"], text=fmt(-d["val"]), neg=True)
            return d
        pc = percent(n)
        if pc is not None:
            P, other, form = pc
            x = walk(other)
            v = None if x["val"] is None else P * x["val"] / 100
            d = emit("pct", "% of", desc(P), x, v, n, P=P, other=other)
            ops[-1]["pct_form"] = form
            return d
        if isinstance(n.op, ast.Mult):
            for c in (n.left, n.right):
                if frac_literal(c):
                    lits[id(c)] = frac_literal(c)
        a, b = walk(n.left), walk(n.right)
        t = type(n.op)
        if t not in SYM:
            v = None
            try:
                v = Fraction(T.soc().OPS[t](a["val"], b["val"]))
            except (ZeroDivisionError, ValueError, OverflowError, TypeError):
                pass
            return emit("bin", t.__name__, a, b, v, n, skip="unsupported_operator")
        v = None if a["val"] is None or b["val"] is None else FN[SYM[t]](a["val"], b["val"])
        return emit("bin", SYM[t], a, b, v, n)

    root = walk(tree)
    return ops, root, lits


def render(n, ops_by_node: dict, upto: int, lits: dict | None = None) -> tuple[str | None, int]:
    """The note's text with the operations before index `upto` replaced by their values (minimal parentheses: a left
    operand of lower precedence, a right operand of lower or equal precedence; "P% of" binds its operand; fraction
    literals as written)."""
    lits = lits or {}
    if id(n) in lits:
        return lits[id(n)], ATOM
    rec = ops_by_node.get(id(n))
    if rec is not None and rec["k"] < upto:
        t = fmt(rec["val"]) if rec["val"] is not None else None
        return (t if t is None or not t.startswith("-") else f"({t})"), ATOM
    if rec is not None and rec["kind"] == "pct":
        other = rec["pct_other"]
        x, px = render(other, ops_by_node, upto, lits)
        if x is None:
            return None, ATOM
        return f"{fmt(rec['P'])}% of {x if px == ATOM else f'({x})'}", PREC[ast.Mult]
    if isinstance(n, ast.Constant):
        t = fmt(lit(n))
        return (t if t is None or not t.startswith("-") else f"({t})"), ATOM
    if isinstance(n, ast.UnaryOp):
        x, px = render(n.operand, ops_by_node, upto, lits)
        if x is None:
            return None, ATOM
        return (f"(-{x})" if isinstance(n.op, ast.USub) else x), ATOM
    if not isinstance(n, ast.BinOp) or type(n.op) not in SYM:
        return None, ATOM
    (a, pa), (b, pb) = render(n.left, ops_by_node, upto, lits), render(n.right, ops_by_node, upto, lits)
    if a is None or b is None:
        return None, ATOM
    p = PREC[type(n.op)]
    a = f"({a})" if pa < p else a
    b = f"({b})" if pb <= p else b
    return f"{a} {SYM[type(n.op)]} {b}", p


def expr_text(kind: str, a: str, sym: str, b: str) -> str:
    return f"{a}% of {b}" if kind == "pct" else f"{a} {sym} {b}"


def expr_value(kind: str, a: Fraction, sym: str, b: Fraction) -> Fraction | None:
    return a * b / 100 if kind == "pct" else FN[sym](a, b)


def trivial(kind: str, sym: str, a: Fraction, b: Fraction) -> bool:
    if kind == "pct":
        return a == 100
    return (sym == TIMES and (a == 1 or b == 1)) or (sym == DIVIDE and b == 1) or (sym == "+" and (a == 0 or b == 0)) \
        or (sym == MINUS and b == 0)


def op_question(kind: str, a: str, sym: str, b: str) -> str:
    return f"What is {expr_text(kind, a, sym, b)}?"


def qop_question(subq: str, reduced: str, kind: str, a: str, sym: str, b: str, last: bool) -> str:
    e = expr_text(kind, a, sym, b)
    return f"{subq} It is {e}. What is the value?" if last else f"{subq} It is {reduced}. What is {e}?"


def opsel_question(subq: str) -> str:
    return f"{subq} Which computation gives it?"


# ------------------------------------------------------------------------------------------------------ options
def value_ok(w: Fraction | None, t: Fraction, gold: Fraction) -> bool:
    if w is None or w <= 0 or fmt(w) is None or fmt(w) == fmt(t) or w == gold:
        return False
    if t.denominator == 1 and w.denominator != 1:
        return False
    return decimals(w) <= max(2, decimals(t)) and max(w / t, t / w) <= 10


def operator_slips(kind: str, a: Fraction, sym: str, b: Fraction) -> list[tuple[Fraction, str]]:
    """Tier 1: the same operands with another operator ("operator") and the complement of a fractional factor
    ("complement")."""
    out = []
    if kind == "pct":
        if a < 100:
            out.append((b * (100 - a) / 100, "complement"))
        out.append((b * (100 + a) / 100, "complement"))
        for s in ("+", MINUS, TIMES, DIVIDE):
            out.append((FN[s](b, a), "operator"))
        return out
    for s, x, y in (("+", a, b), (MINUS, a, b), (MINUS, b, a), (TIMES, a, b), (DIVIDE, a, b), (DIVIDE, b, a)):
        if not (s == sym and x == a and y == b):
            out.append((FN[s](x, y), "operator"))
    if sym in (TIMES, DIVIDE):
        for pos, f in (("a", a), ("b", b)):
            if f.denominator != 1 and 0 < f < 2 and f != 1:
                g = 1 - f if f < 1 else f - 1
                out.append((FN[sym](g, b) if pos == "a" else FN[sym](a, g), "complement"))
    return out


def trailing_zeros(n: int) -> int:
    k = 0
    while n and n % 10 ** (k + 1) == 0 and k < 3:
        k += 1
    return k


def near_values(t: Fraction) -> list[Fraction]:
    """Tier 3: build_gsm8k_tf.perturbs on both sides (parity and roundness kept), plus, so that small values have a
    neighbour below, t +- 1 for an integer with no trailing zero when t +- 1 has none either, and t +- (1, 2, 5) units
    of its last decimal for a non-integer."""
    out = [Fraction(str(round(x, MAX_DEC))) for side in (True, False) for x in T.perturbs(float(t), side)]
    if t.denominator == 1:
        n = t.numerator
        if trailing_zeros(n) == 0:
            out += [Fraction(m) for m in (n - 1, n + 1) if m > 0 and trailing_zeros(m) == 0]
    else:
        u = Fraction(1, 10 ** decimals(t))
        out += [t + d * u for d in (1, 2, 5, -1, -2, -5)]
    return out


RANK_SIDES = {0: (True, True), 1: (False, True), 2: (False, False)}   # truth's rank (0 = smallest) -> wrongs above?


def value_options(t: Fraction, slips: list, others: list, gold: Fraction, rng: random.Random):
    """(options, answer index, wrong kinds) or None (no two wrong values). The truth's rank among the three values is
    drawn uniformly first (smallest / middle / largest; the next rank in a seeded order when it cannot be met), and each
    wrong value on its required side comes from the most preferred tier with a candidate there: operator /
    complement slips, then other operations' values, then perturbations. So the truth's rank does not follow from the
    operator (with plain tier preference "÷" would be the smallest option three times in four)."""
    pools = {"operator": {}, "other": {}, "perturb": {}}
    for w, kind in slips:
        if value_ok(w, t, gold):
            pools["operator"].setdefault(fmt(w), (w, kind))
    for w in others:
        if value_ok(w, t, gold):
            pools["other"].setdefault(fmt(w), (w, "other"))
    for w in near_values(t):
        if value_ok(w, t, gold):
            pools["perturb"].setdefault(fmt(w), (w, "perturb"))
    ranks = [0, 1, 2]
    rng.shuffle(ranks)
    for rank in ranks:
        chosen = []
        for above in RANK_SIDES[rank]:
            for tier in TIERS:
                cand = sorted(x for x, (w, _) in pools[tier].items() if (w > t) == above and x not in dict(chosen))
                if cand:
                    x = rng.choice(cand)
                    chosen.append((x, pools[tier][x][1]))
                    break
        if len(chosen) == 2:
            break
    else:
        return None
    opts = [fmt(t)] + [c for c, _ in chosen]
    kinds = [None] + [k for _, k in chosen]
    perm = list(range(3))
    rng.shuffle(perm)
    return [opts[i] for i in perm], perm.index(0), [kinds[i] for i in perm]


def opsel_options(kind: str, a: Fraction, at: str, sym: str, b: Fraction, bt: str, t: Fraction, cands: dict,
                  gold: Fraction, rng: random.Random):
    """(options, answer index, wrong kinds, design) or None. at / bt: the operands' texts; cands: the replacement
    numbers per operand position ("a", "b"; pct: a = the percentage, b = its base), never 0, 1 or the gold, never
    making a trivial expression."""
    truth = expr_text(kind, at, sym, bt)

    def good(text, v):
        return text != truth and v is not None and v > 0 and v != t

    oper = []
    if kind == "pct":
        for s in ("+", MINUS, TIMES, DIVIDE):
            v = FN[s](b, a)
            oper.append((f"{bt} {s} {at}", v))
    else:
        for s in ("+", MINUS, TIMES, DIVIDE):
            if s != sym:
                oper.append((f"{at} {s} {bt}", FN[s](a, b)))
    oper = sorted({x: v for x, v in oper if good(x, v)}.items())
    by_pos = {}
    for pos, old in (("a", a), ("b", b)):
        got = {}
        for c in cands[pos]:
            if c == old or c == gold or c <= 0 or c == 1 or fmt(c) is None:
                continue
            x, y = (c, b) if pos == "a" else (a, c)
            if trivial(kind, sym, x, y):
                continue
            v = expr_value(kind, x, sym, y)
            text = expr_text(kind, fmt(c), sym, bt) if pos == "a" else expr_text(kind, at, sym, fmt(c))
            if good(text, v):
                got.setdefault(text, v)
        by_pos[pos] = sorted(got.items())
    designs = []
    if len(oper) >= 2:
        designs.append("operator")
    if any(len(v) >= 2 for v in by_pos.values()):
        designs.append("operand")
    if designs:
        design = rng.choice(designs)
        if design == "operator":
            wrong = [(x, "operator") for x, _ in rng.sample(oper, 2)]
        else:
            pos = rng.choice([p for p in ("a", "b") if len(by_pos[p]) >= 2])
            wrong = [(x, "operand") for x, _ in rng.sample(by_pos[pos], 2)]
    else:
        operand = [x for p in ("a", "b") for x in by_pos[p]]
        if not oper or not operand:
            return None
        design = "mixed"
        wrong = [(rng.choice(oper)[0], "operator"), (rng.choice(operand)[0], "operand")]
        if wrong[0][0] == wrong[1][0]:
            return None
    opts = [truth] + [x for x, _ in wrong]
    kinds = [None] + [k for _, k in wrong]
    perm = list(range(3))
    rng.shuffle(perm)
    return [opts[i] for i in perm], perm.index(0), [kinds[i] for i in perm], design


# ------------------------------------------------------------------------------------------------------ nodes
def blocked_steps(p: dict, st: collections.Counter) -> set:
    """build_gsm8k_tf.tf_nodes' blocked set without the gold step itself: gold-valued steps, steps downstream of one
    and steps with an uncertain edge to one (j < gold step)."""
    S = T.soc()
    steps, gs, gold = p["steps"], p["gold_step"], p["gold"]
    blocked = {j for j, s in enumerate(steps) if S.close(s["value"], gold) and j != gs}
    st["steps_blocked_gold_valued"] += len(blocked)
    for j in range(gs):
        if j in blocked:
            continue
        if T.ancestors(steps, j) & blocked:
            blocked.add(j)
            st["steps_blocked_downstream_of_gold_valued"] += 1
        elif T.reads_blocked(steps, j, blocked):
            blocked.add(j)
            st["steps_blocked_uncertain_edge_to_gold_valued"] += 1
    return blocked


def value_nodes(p: dict, pid: str, seed: int, st: collections.Counter) -> list[dict]:
    """The value nodes of one parsed problem (build_gsm8k_socratic.parse_problem, status ok)."""
    steps, gs = p["steps"], p["gold_step"]
    gold = Fraction(str(p["gold"]))
    blocked = blocked_steps(p, st)
    gold_reads_blocked = bool(T.ancestors(steps, gs) & blocked) or T.reads_blocked(steps, gs, blocked)
    fin = T.norm(T.final_sentence(p["question"]))
    by_text = collections.defaultdict(set)
    for s in steps:
        by_text[T.norm(s["subq"])].add(round(s["value"], 6))
    gold_anc = T.ancestors(steps, gs) | {gs}
    qnums = sorted({num(x) for x in NUMTOK.findall(p["question"])})
    # every operation of every step (built or not): values, depths, ids
    split = []
    for j, s in enumerate(steps):
        ops, _, lits = split_note(s["tree"], written_percents(s["sent"], p["question"]), s["refs"])
        nodes_by = {id(r["node"]): r for r in ops}
        split.append((ops, nodes_by, lits))
        st["ops_total"] += len(ops)
        st["percent_nodes_split"] += sum(r["kind"] == "pct" for r in ops)
        if not ops:
            st["steps_without_operation"] += 1
    depth, last_id = {}, {}
    for j, (ops, _, _) in enumerate(split):
        for r in ops:
            par = []
            for d in (r["a"], r["b"]):
                if d["op"] is not None:
                    par.append(f"s{j + 1}o{d['op'] + 1}")
                elif d["ref"] is not None and d["ref"] in last_id:
                    par.append(last_id[d["ref"]])
            r["id"], r["parents"] = f"s{j + 1}o{r['k'] + 1}", par
            depth[r["id"]] = 1 + max(depth[x] for x in par) if par else 0
        if ops:
            last_id[j] = ops[-1]["id"]
    all_vals = [r["val"] for ops, _, _ in split for r in ops if r["val"] is not None] + \
        [Fraction(str(round(s["value"], MAX_DEC))) for s in steps]
    out, built = [], set()
    for j, (ops, nodes_by, lits) in enumerate(split):
        s = steps[j]
        if j > gs:
            st["ops_dropped_after_gold_step"] += len(ops)
            continue
        if j in blocked:
            st["ops_blocked_step"] += len(ops)
            continue
        if j == gs and gold_reads_blocked:
            st["ops_blocked_gold_step_reads_blocked"] += len(ops)
            continue
        subq = T.ask_text(s["subq"])
        if len(by_text[T.norm(s["subq"])]) > 1:
            subq_ok = "shared_other_value"
        elif T.norm(s["subq"]) == fin and j != gs:
            subq_ok = "is_final_question"
        else:
            subq_ok = None
        tainted = set()                             # operations of this note downstream of a gold-valued one
        earlier = [Fraction(str(round(x["value"], MAX_DEC))) for x in steps[:j]]
        for r in ops:
            last = r["k"] == len(ops) - 1
            a, b, v = r["a"], r["b"], r["val"]
            reason = None
            if j == gs and last:
                reason = "final_op"
            elif r["skip"]:
                reason = r["skip"]
            elif any(d["op"] in tainted for d in (a, b) if d["op"] is not None):
                reason = "downstream_of_gold_valued_op"
            elif v is not None and v == gold:
                reason = "gold_valued_op"
            elif a["neg"] or b["neg"]:
                reason = "operand_negative"
            elif a["text"] is None or b["text"] is None:
                reason = "operand_inexact"
            elif v is None or v <= 0:
                reason = "value_not_positive"
            elif fmt(v) is None:
                reason = "value_inexact"
            elif trivial(r["kind"], r["sym"], a["val"], b["val"]):
                reason = "trivial"
            if reason in ("gold_valued_op", "downstream_of_gold_valued_op"):
                tainted.add(r["k"])
            if reason:
                st[f"ops_skipped_{reason}"] += 1
                continue
            kind, sym = r["kind"], r["sym"]
            rng = random.Random(f"{seed}/{pid}/{r['id']}/value")
            others = [x for x in all_vals if x != v]
            vo = value_options(v, operator_slips(kind, a["val"], sym, b["val"]), others, gold, rng)
            if vo is None:
                st["ops_skipped_no_wrong_values"] += 1
                continue
            opts, ans, wrong = vo
            ren = {"op": {"question": op_question(kind, a["text"], sym, b["text"]), "options": opts, "answer": ans,
                          "wrong": wrong}}
            if subq_ok:
                st[f"qop_opsel_absent_subq_{subq_ok}"] += 1
            else:
                reduced, _ = render(s["tree"], nodes_by, r["k"], lits)
                if j == gs:                         # "<final question> It is <the gold note>": the final in disguise
                    st["qop_absent_gold_step_partial"] += 1
                elif reduced is None:
                    st["qop_absent_unrenderable"] += 1
                else:
                    ren["qop"] = {"question": qop_question(subq, reduced, kind, a["text"], sym, b["text"], last),
                                  "options": opts, "answer": ans, "wrong": wrong}
                if last:
                    nums = sorted(set(qnums) | set(earlier))
                    pcts = sorted(written_percents(s["sent"], p["question"]) | {100 - a["val"]}) if kind == "pct" else nums
                    cands = {"a": pcts, "b": nums}
                    so = opsel_options(kind, a["val"], a["text"], sym, b["val"], b["text"], v, cands, gold,
                                       random.Random(f"{seed}/{pid}/{r['id']}/opsel"))
                    if so is None:
                        st["opsel_absent_no_wrong_expressions"] += 1
                    else:
                        o2, a2, w2, design = so
                        ren["opsel"] = {"question": opsel_question(subq), "options": o2, "answer": a2, "wrong": w2,
                                        "design": design}
                else:
                    st["opsel_absent_not_last_op"] += 1
            node = {"id": r["id"], "type": "op", "step": j + 1, "op_index": r["k"] + 1, "n_ops": len(ops),
                    "last_of_step": last, "expr": s["expr"], "subq": subq, "kind": kind, "a": a["text"], "op": sym,
                    "b": b["text"], "value": fmt(v), "depends_on": [x for x in r["parents"] if x in built],
                    "node_depth": depth[r["id"]], "gold_ancestor": j in gold_anc, "gold_step": j == gs,
                    "flags": s["flags"], "renderings": ren}
            if kind == "pct":                       # where "P%" is written; P*.01-style vs a decimal literal
                node["percent"] = {"written_in": "sentence" if r["P"] in written_percents(s["sent"]) else "problem",
                                   "form": r["pct_form"]}
                st[f"pct_nodes_written_in_{node['percent']['written_in']}_{r['pct_form']}"] += 1
            out.append(node)
            built.add(r["id"])
            st["nodes"] += 1
            st["nodes_gold_step_partial"] += j == gs
            for k in ren:
                st[f"rendering_{k}"] += 1
    return out



# ------------------------------------------------------------------------------------------------------ audit
OPT_BIN = re.compile(r"^(\S+) ([+\u2212\u00d7\u00f7]) (\S+)$")
OPT_PCT = re.compile(r"^(\S+)% of (\S+)$")


def exact_eval(n) -> Fraction | None:
    """A parsed note evaluated exactly (Fraction); None for an unsupported node."""
    S = T.soc()
    if isinstance(n, ast.Constant):
        return lit(n)
    if isinstance(n, ast.UnaryOp):
        v = exact_eval(n.operand)
        return None if v is None else (-v if isinstance(n.op, ast.USub) else v)
    if isinstance(n, ast.BinOp):
        a, b = exact_eval(n.left), exact_eval(n.right)
        if a is None or b is None:
            return None
        t = type(n.op)
        if t in SYM:
            return FN[SYM[t]](a, b)
        try:
            return Fraction(S.OPS[t](a, b))
        except (ZeroDivisionError, ValueError, OverflowError, TypeError):
            return None
    return None


def canon(x: Fraction, text: str) -> bool:
    """text is x printed exactly, or a fraction literal p/q of value x."""
    return fmt(x) == text or (re.fullmatch(r"[1-9]\d*/[1-9]\d*", text) is not None and num(text) == x)


def parse_option(text: str) -> tuple[str, Fraction, str, Fraction] | None:
    m = OPT_PCT.match(text)
    if m:
        return "pct", num(m.group(1)), "% of", num(m.group(2))
    m = OPT_BIN.match(text)
    if m:
        return "bin", num(m.group(1)), m.group(2), num(m.group(3))
    return None


def audit_item(item: dict) -> tuple[list[str], collections.Counter]:
    """Problems of one built item from its own fields (empty: clean) and check counts (ops_checked, op_mismatch,
    step_chain_checked, step_chain_mismatch)."""
    S = T.soc()
    bad, c = [], collections.Counter()
    iid = item.get("item_id")
    gold = num(T.gold_text(item))
    nodes = item.get(FIELD)
    if not isinstance(nodes, list):
        return [f"{iid}: no {FIELD} list"], c
    by = {n["id"]: n for n in nodes}
    if len(by) != len(nodes):
        bad.append("duplicate node ids")
    for n in nodes:
        nid = n["id"]
        try:
            A, B, V = num(n["a"]), num(n["b"]), num(n["value"])
        except (ValueError, ZeroDivisionError):
            bad.append(f"{nid}: operands / value do not parse")
            continue
        c["ops_checked"] += 1
        got = expr_value(n["kind"], A, n["op"], B)
        if got is None or got != V or fmt(V) != n["value"] or not canon(A, n["a"]) or not canon(B, n["b"]):
            c["op_mismatch"] += 1
            bad.append(f"{nid}: {n['a']} {n['op']} {n['b']} is not {n['value']} (exact: {got})")
        if V <= 0 or V == gold or trivial(n["kind"], n["op"], A, B):
            bad.append(f"{nid}: value {n['value']} not positive, equal to the gold or trivial")
        ren = n.get("renderings") or {}
        if "op" not in ren or set(ren) - set(RENDERINGS):
            bad.append(f"{nid}: renderings {sorted(ren)}")
            continue
        for k, r in ren.items():
            o, a = r["options"], r["answer"]
            if len(o) != 3 or len(set(o)) != 3 or not isinstance(a, int) or not 0 <= a < 3 or r["wrong"][a] is not None \
                    or any(w is None for i, w in enumerate(r["wrong"]) if i != a) or "\n" in r["question"]:
                bad.append(f"{nid}/{k}: options {o} answer {a} wrong {r['wrong']}")
                continue
            if k in ("op", "qop"):
                if o[a] != n["value"] or str(fmt(gold)) in o or any(fmt(num(x)) != x for x in o):
                    bad.append(f"{nid}/{k}: truth not at the answer, gold among the options or a non-canonical number")
                if k == "qop" and (o, a) != (ren["op"]["options"], ren["op"]["answer"]):
                    bad.append(f"{nid}/qop: options differ from op")
            else:
                if not n["last_of_step"] or o[a] != expr_text(n["kind"], n["a"], n["op"], n["b"]):
                    bad.append(f"{nid}/opsel: not a last operation or the truth is not the expression")
                for i, x in enumerate(o):
                    po = parse_option(x)
                    v = expr_value(*po) if po else None
                    if v is None or (i != a and v == V):
                        bad.append(f"{nid}/opsel: option {x!r} has no value or the truth's value")
        e = expr_text(n["kind"], n["a"], n["op"], n["b"])
        if ren["op"]["question"] != f"What is {e}?":
            bad.append(f"{nid}/op: question {ren['op']['question']!r}")
        if "qop" in ren and n["gold_step"]:
            bad.append(f"{nid}/qop: a qop of the gold step states the final answer's computation")
        if "qop" in ren:
            q = ren["qop"]["question"]
            want_end = " What is the value?" if n["last_of_step"] else f" What is {e}?"
            if not q.startswith(n["subq"] + " It is ") or not q.endswith(want_end) or \
                    (n["last_of_step"] and q != f"{n['subq']} It is {e}. What is the value?"):
                bad.append(f"{nid}/qop: question {q!r}")
        if "opsel" in ren and ren["opsel"]["question"] != f"{n['subq']} Which computation gives it?":
            bad.append(f"{nid}/opsel: question {ren['opsel']['question']!r}")
        d = 0
        for x in n["depends_on"]:
            if x not in by:
                bad.append(f"{nid}: parent {x} missing")
                continue
            if not any(S.matches(float(c), float(num(by[x]["value"]))) for c in (A, B)):
                bad.append(f"{nid}: parent {x} value {by[x]['value']} is not an operand")
            d = max(d, by[x]["node_depth"] + 1)
        if n["node_depth"] < d:
            bad.append(f"{nid}: node_depth below its parents' + 1")
        if n["last_of_step"]:                       # the chain of operations ends at the note's exact value
            c["step_chain_checked"] += 1
            tree = S.parse_expr(n["expr"])
            full = exact_eval(tree) if tree is not None else None
            if full is None or full != V:
                c["step_chain_mismatch"] += 1
                bad.append(f"{nid}: the note {n['expr']!r} is {full}, its last operation gives {n['value']}")
    return [f"{iid}: {b}" for b in bad], c


# ------------------------------------------------------------------------------------------------------ stats
def option_values(r: dict) -> list[Fraction]:
    return [num(x) for x in r["options"]]


def cue_stats(items: list[dict]) -> dict:
    """Shortcut checks on the training options: how often picking the largest / smallest / middle value (op, by
    operator) or the option that shares most with the others (opsel) finds the truth (chance 1/3)."""
    out = {}
    rows = collections.defaultdict(list)
    in_prob = collections.Counter()
    for it in items:
        q = it["prompt"].partition("\nOptions:")[0]
        qn = {num(x) for x in NUMTOK.findall(q)}
        for n in it[FIELD]:
            r = n["renderings"]["op"]
            vs = option_values(r)
            srt = sorted(range(3), key=lambda i: vs[i])
            hit = {"max": srt[2] == r["answer"], "min": srt[0] == r["answer"], "median": srt[1] == r["answer"]}
            rows["all"].append(hit)
            rows[n["op"]].append(hit)
            in_prob["truth_in_problem"] += vs[r["answer"]] in qn
            in_prob["wrong_in_problem"] += sum(v in qn for i, v in enumerate(vs) if i != r["answer"])
            in_prob["n"] += 1
    for k, hs in rows.items():
        out[f"op/{k}"] = {"n": len(hs), **{h: round(sum(x[h] for x in hs) / len(hs), 3) for h in ("max", "min", "median")}}
    out["op/in_problem"] = {"truth_share": round(in_prob["truth_in_problem"] / max(1, in_prob["n"]), 3),
                            "wrong_share_per_option": round(in_prob["wrong_in_problem"] / max(1, 2 * in_prob["n"]), 3)}
    cen, n_sel = 0.0, 0
    for it in items:
        for n in it[FIELD]:
            r = n["renderings"].get("opsel")
            if not r:
                continue
            ps = [parse_option(x) for x in r["options"]]
            score = []
            for i, (k, a, s, b) in enumerate(ps):
                sc = 0
                for j, (k2, a2, s2, b2) in enumerate(ps):
                    if i != j:
                        sc += (s == s2) + (sorted([a, b]) == sorted([a2, b2])) + (a == a2) + (b == b2)
                score.append(sc)
            top = [i for i in range(3) if score[i] == max(score)]
            cen += (r["answer"] in top) / len(top)
            n_sel += 1
    out["opsel/centre"] = {"n": n_sel, "acc": round(cen / max(1, n_sel), 3)}
    prior = collections.Counter(n["op"] for it in items for n in it[FIELD] if "opsel" in n["renderings"])
    rank = {s_: i for i, (s_, _) in enumerate(prior.most_common())}
    hit, m = 0.0, 0
    for it in items:
        for n in it[FIELD]:
            r = n["renderings"].get("opsel")
            if not r or r["design"] != "operator":
                continue
            ps = [parse_option(x)[2] for x in r["options"]]
            best = min(rank.get(x, 99) for x in ps)
            top = [i for i, x in enumerate(ps) if rank.get(x, 99) == best]
            hit += (r["answer"] in top) / len(top)
            m += 1
    out["opsel/operator_prior"] = {"n": m, "acc": round(hit / max(1, m), 3), "order": [s_ for s_, _ in prior.most_common()]}
    return out


def file_stats(items: list[dict]) -> dict:
    nodes = [n for it in items for n in it[FIELD]]
    ren = {}
    for k in RENDERINGS:
        rs = [n["renderings"][k] for n in nodes if k in n["renderings"]]
        ren[k] = {"nodes": len(rs), "items_with": sum(any(k in n["renderings"] for n in it[FIELD]) for it in items),
                  "gold_letter": dict(sorted(collections.Counter("ABC"[r["answer"]] for r in rs).items())),
                  "wrong": dict(sorted(collections.Counter(w for r in rs for w in r["wrong"] if w).items()))}
        if k == "opsel":
            ren[k]["design"] = dict(sorted(collections.Counter(r["design"] for r in rs).items()))
    yn = sum(len(it["subqs"]) for it in items)
    return {"items": len(items), "items_with_value_nodes": sum(bool(it[FIELD]) for it in items),
            "value_nodes": len(nodes), "value_nodes_per_item": round(len(nodes) / max(1, len(items)), 2),
            "yesno_nodes": yn, "yesno_steps": yn // 2, "yesno_per_item": round(yn / max(1, len(items)), 2),
            "kind": dict(collections.Counter(n["kind"] for n in nodes)),
            "operator": dict(sorted(collections.Counter(n["op"] for n in nodes).items())),
            "node_depth": dict(sorted(collections.Counter(n["node_depth"] for n in nodes).items())),
            "gold_step_partial": sum(n["gold_step"] for n in nodes),
            "multi_op_note": sum(n["n_ops"] > 1 for n in nodes),
            "renderings": ren}


# ------------------------------------------------------------------------------------------------------ audit file
def show_problem(item: dict, answer: str) -> str:
    out = [f"=== {item['item_id']}  (gold {T.gold_text(item)}, {len(item['subqs'])} yes/no nodes, "
           f"{len(item[FIELD])} value nodes)", item["prompt"], "", "GSM8K solution:"]
    out += ["  " + line for line in answer.strip().split("\n")]
    out += ["", "yes/no nodes (data/student_gsm8k_tf): " + " | ".join(n["question"] for n in item["subqs"] if n["truth"])
            or "yes/no nodes: none", ""]
    for n in item[FIELD]:
        out.append(f"[{n['id']}] {expr_text(n['kind'], n['a'], n['op'], n['b'])} = {n['value']}   (note {n['expr']}, "
                   f"op {n['op_index']}/{n['n_ops']}, depends_on {n['depends_on']}, depth {n['node_depth']}"
                   f"{', gold step' if n['gold_step'] else ''})")
        for k in RENDERINGS:
            r = n["renderings"].get(k)
            if not r:
                out.append(f"    {k:5s}  (absent)")
                continue
            opts = "  ".join(f"{'*' if i == r['answer'] else ' '}{L}) {o}" + (f" [{r['wrong'][i]}]" if r['wrong'][i] else "")
                             for i, (L, o) in enumerate(zip("ABC", r["options"])))
            out.append(f"    {k:5s}  {r['question']}")
            out.append(f"           {opts}" + (f"   design {r['design']}" if k == "opsel" else ""))
    return "\n".join(out) + "\n"


def pick_audit(items: list[dict]) -> list[dict]:
    """gsm8k-train-660, then the first train items with a percent node, a multi-operation note (not the gold step),
    a decimal division, and a node without qop / opsel."""
    by = {it["item_id"]: it for it in items}
    out = [by["gsm8k-train-660"]] if "gsm8k-train-660" in by else []
    tests = [lambda n: n["kind"] == "pct",
             lambda n: n["n_ops"] > 1 and not n["gold_step"],
             lambda n: n["op"] == DIVIDE and "." in n["value"],
             lambda n: "qop" not in n["renderings"]]
    for f in tests:
        for it in items:
            if it not in out and any(f(n) for n in it[FIELD]) and len(it[FIELD]) <= 6:
                out.append(it)
                break
    return out


# ------------------------------------------------------------------------------------------------------ build
def dumps(r: dict) -> str:
    return json.dumps(r, ensure_ascii=False)


def build(args) -> None:
    import pandas as pd
    S = T.soc()
    src, out = ROOT / args.src, ROOT / args.out
    df = pd.read_parquet(T.RAW / "train-00000-of-00001.parquet")
    st = collections.Counter()
    fails, built, answers = [], {}, {}
    for split in ("train", "val"):
        built[split] = []
        with open(src / f"{split}.jsonl") as f:
            lines = [x.rstrip("\n") for x in f if x.strip()]
        for line in lines:
            r = json.loads(line)
            i = int(r["problem_id"].rsplit("-", 1)[1])
            p = S.parse_problem(i, df.question[i], df.answer[i])
            if p["status"] != "ok" or f"Question: {p['question']}\nOptions:\n" != r["prompt"][:len(p["question"]) + 20]:
                fails.append(f"{r['item_id']}: raw row {i} does not reproduce the prompt")
                continue
            if T._fmt(p["gold"]) != T.gold_text(r) or len(p["steps"]) != r["n_steps"]:
                fails.append(f"{r['item_id']}: gold / step count differs from the raw solution")
                continue
            o = dict(r)
            o[FIELD] = value_nodes(p, r["problem_id"], args.seed, st)
            st["items"] += 1
            st["items_with_value_nodes"] += bool(o[FIELD])
            rest = {k: v for k, v in o.items() if k != FIELD}
            if dumps(rest) != line:
                fails.append(f"{r['item_id']}: row differs from {args.src} beyond {FIELD}")
            built[split].append(o)
            answers[r["item_id"]] = df.answer[i]
    print("built", {k: len(v) for k, v in built.items()}, f"rss {T.rss_mb()} MB", flush=True)
    audit = collections.Counter()
    for it in built["train"] + built["val"]:
        b, c = audit_item(it)
        fails += b
        audit.update(c)
    summary = {"script": "scripts/build_gsm8k_ops.py", "args": vars(args), "field": FIELD, "md5": {}, "files": {}}
    for split in ("train", "val"):
        key = f"{out.name}/{split}.jsonl"
        summary["md5"][key] = T.write(out / f"{split}.jsonl", built[split])
        summary["files"][key] = file_stats(built[split])
    by_id = {it["item_id"]: it for it in built["train"]}
    summary["subsets"] = {}
    for fr in FRACTIONS:
        sdir = ROOT / f"{args.src}_f{fr}"
        odir = ROOT / f"{args.out}_f{fr}"
        if not (sdir / "train.jsonl").exists():
            fails.append(f"missing {sdir}/train.jsonl")
            continue
        ids = [json.loads(x)["item_id"] for x in open(sdir / "train.jsonl") if x.strip()]
        sval = [json.loads(x)["item_id"] for x in open(sdir / "val.jsonl") if x.strip()]
        if sval != [it["item_id"] for it in built["val"]]:
            fails.append(f"{sdir}/val.jsonl is not the full val")
        miss = [x for x in ids if x not in by_id]
        if miss:
            fails.append(f"{sdir}: {len(miss)} train ids not built, e.g. {miss[:3]}")
        rows_ = [by_id[x] for x in ids if x in by_id]
        for name, rr in (("train", rows_), ("val", built["val"])):
            key = f"{odir.name}/{name}.jsonl"
            summary["md5"][key] = T.write(odir / f"{name}.jsonl", rr)
        summary["files"][f"{odir.name}/train.jsonl"] = file_stats(rows_)
        summary["subsets"][odir.name] = {"source_ids": f"{sdir.relative_to(ROOT)}/train.jsonl", "n_train": len(rows_),
                                         "same_ids_same_order": [it["item_id"] for it in rows_] == ids,
                                         "source_train_md5": T.md5(sdir / "train.jsonl")}
    summary["source_md5"] = {f"{args.src}/{s}.jsonl": T.md5(src / f"{s}.jsonl") for s in ("train", "val")}
    summary["build"] = dict(sorted(st.items()))
    summary["audit"] = {"failures": len(fails), "examples": fails[:20], **dict(sorted(audit.items()))}
    summary["cues_train"] = cue_stats(built["train"])
    summary["leaks"] = {"value_option_equals_gold": sum(T.gold_text(it) in n["renderings"][k]["options"]
                                                        for it in built["train"] + built["val"] for n in it[FIELD]
                                                        for k in ("op", "qop") if k in n["renderings"]),
                        "node_value_equals_gold": sum(n["value"] == T.gold_text(it)
                                                      for it in built["train"] + built["val"] for n in it[FIELD]),
                        "gold_number_in_op_question": sum(T.gold_text(it) in NUMTOK.findall(n["renderings"]["op"]["question"])
                                                          for it in built["train"] + built["val"] for n in it[FIELD]),
                        "gold_number_in_op_question_not_in_problem": sum(
                            T.gold_text(it) in NUMTOK.findall(n["renderings"]["op"]["question"])
                            and T.gold_text(it) not in NUMTOK.findall(it["prompt"].partition("\nOptions:")[0])
                            for it in built["train"] + built["val"] for n in it[FIELD])}
    picks = pick_audit(built["train"])
    (out / "node_audit.txt").write_text("\n".join(show_problem(it, answers[it["item_id"]]) for it in picks))
    summary["node_audit"] = [it["item_id"] for it in picks]
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("files", "subsets", "build", "audit", "cues_train", "leaks")}, indent=1,
                     ensure_ascii=False))
    print(f"rss {T.rss_mb()} MB")
    if fails:
        print("AUDIT FAILED", fails[:10])
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/student_gsm8k_tf")
    ap.add_argument("--out", default="data/student_gsm8k_tf_ops")
    ap.add_argument("--seed", type=int, default=0)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
