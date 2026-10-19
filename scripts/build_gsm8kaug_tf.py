"""Tree Forcing data from GSM8K-Aug (Deng et al. 2023, the 385k GPT-4 augmentation of GSM8K train; HF whynlp/gsm8k-aug
rev 62440e946e: equation steps "<<600*30/100=180>>"; whynlp/gsm8k-aug-nl rev 88f8ada364: the same problems, row for
row, with natural-language steps "600 x 30/100 = 180 employees were promoted."), in the row schema of
data/student_gsm8k_tf_si (scripts/build_gsm8k_tf.py + build_gsm8k_ops.py + build_gsm8k_si.py: finals, the yes/no tree
in `subqs`, `value_nodes`, `worked_steps`), so every GSM8K arm (A2, GFPQ, GFPQV / GFPQO, SIB / SIYB, GFPQC, ...) runs on
it unchanged with XARGS="--tree-sources gsm8k_tf".

Identity: source "gsm8k_tf" (the arms' --tree-sources and treeplan.problem_no_options key on it); the rows are told
apart by item_id / problem_id "gsm8kaug-train-<row of the gsm8k-aug train parquet>", group "gsm8kaug_tf/s<n>" and
tree_source "gsm8kaug_tf". Licence: the dataset cards state none (licence_class UNK). Key set and key order are those of
a data/student_gsm8k_tf_si row; every node key set is that of a GSM8K node of the same kind.

Steps (verify_row; a row failing any check is dropped, the reason counted in summary.json filters):
  each step must be exactly "<<expr=value>>" (no text, empty step, unclosed note); expr may use digits, . , $ ( ) + - * /
  x X × ÷ ^ and "P%" (read as (P/100)); any other symbol (variables, ~, ≈) drops the row. expr is executed EXACTLY
  (fractions.Fraction of the literals as written, build_gsm8k_ops.exact_eval) and must equal the stated value exactly
  (an integer, a decimal or p/q): "65/60=1.08333" is dropped (value_rounded: the exact value rounded to the stated
  decimals; value_wrong otherwise). The last step's value must equal the answer (stripped of $ and ,; a number or
  p/q), and the answer must be positive and printable at 2 decimals (the option format). The verified steps are then
  parsed by build_gsm8k_socratic.parse_problem exactly as GSM8K's calculator notes are (operand edges, flags); a
  problem must have status ok (an "orphan" operand, a number from nowhere, drops it as in build_gsm8k_tf). The gold step
  is the last step. n_steps = depth = the number of equations.

Finals (build_final): the GSM8K recipe of build_gsm8k_socratic.build_rows, unchanged: 5 options, the gold + 4
  distractors from the recipe drawn per problem (qo2 .4 / inter .3 / slip .3, fallback inter, slip, qo2): qo2 = the
  frozen QO-v2 recipe (partial results a op b of the problem's numbers, perturbations of the gold / partials), inter =
  one other step value of the solution + one partial result + 2 perturbations, slip = one planted-slip final (an
  operator swapped or an operand moved by 1, propagated through the step graph) + one partial result + 2 perturbations.
  Options are distinct as printed (audit), the gold position is a seeded per-problem shuffle (gold letter histogram in
  summary.json). Seeds: random.Random("gsm8kaug_soc/<pid>") (options, order), "<seed>/<pid>/recipe" (recipe).

Yes/no nodes (`subqs`, GFPQ / GFPQC; yesno_nodes): GSM8K-Aug has no Socratic sub-questions, so a node states a
  solution step and asks whether it is correct. Which steps get nodes, the wrong values, ids, parents and depths are
  exactly build_gsm8k_tf.tf_nodes': per step j that is not the gold step (no node for the step that yields the final
  answer), not gold-valued, not downstream of a gold-valued step and with a value exact at 2 decimals: "s<j>v" (Yes) and
  "s<j>n" (No) with the wrong value of build_gsm8k_tf.wrong_value (a planted slip .5, another step's value .25, a
  parity/roundness-keeping perturbation .25; never the gold, within 10x, the value's integer-ness), so every item and
  step is 50% Yes. (tf_nodes' two sub-question checks, a text shared by steps of different values and a text equal to
  the final question, have nothing to check here.) The statement:
  - "Is this step correct? <sentence>": the natural-language step sentence of gsm8k-aug-nl when the row's sentences
    align with its equations 1:1 (align: an order-preserving injective map, each equation to a sentence whose results
    "= x" (an "=" followed by a number that is not followed by an operator) are all its value; the questions of both
    datasets must agree), and the sentence is usable for that step (nl_spans): one line; every literal of the verified
    equation (but 100 / .01) written before the result, as itself or as a percentage, so the sentence states that
    computation; the value not written before its result (an operand "60 + 60 = 120" is fine, a value stated earlier
    is not); the gold not written in it (the final answer leaking into a sub-step). The No twin is the sentence with the result and its later repeats replaced
    by the wrong value, written in the same number format (thousands separators, decimals; fmt_like), else the step
    falls back;
  - otherwise "Is it true that <expr> = <x>?", expr the note as build_gsm8k_ops renders it ("600 × 30 ÷ 100"; fraction
    literals p/q as written), x the value or the wrong value.
  "opposite" holds the reversed wording ("Is this step wrong? ..." / "Is it false that ...?"). Unlike GSM8K TF's
  sub-question + probe, both forms state the computation, so a node is a step-verification (arithmetic) check in the
  problem's context, not a sub-question to solve.
Value nodes (`value_nodes`, GFPQV / GFPQO / GFPQVQ; build_gsm8k_ops.value_nodes unchanged): one node per atomic
  operation of the equations, the same skips and options. Its qop / opsel renderings need a description of the step's
  quantity in place of the Socratic sub-question: the step's NL sentence with its computation and result masked
  (cloze: "So a total of ___ employees received a promotion or a bonus."), only where the masked span is a computation
  that evaluates exactly to the value and the cloze writes neither the value nor the gold and has >= 2 words; steps
  without a cloze (unaligned rows, ...) get the op rendering only (build_gsm8k_ops' absence rule for an unusable
  sub-question).
Worked steps (`worked_steps`, SI / SIB / SIYB / SIS; build_gsm8k_si.worked_steps unchanged): the complete chain of atomic
  operations up to the gold step, its last line the gold.

Decontamination (Screen; the question text only; any hit drops the row), in this order after exact duplicate
  questions within GSM8K-Aug (the first verified copy kept):
  - test: the normalised text equal; a word 8-gram (build_gsm8k_tf's rule); a raw 13-gram; a digit-masked 10-gram
    (build_gsm8k_socratic); masked word-TRIGRAM-set Jaccard >= .5. An n-gram in >= 5 target groups (test problems:
    GSM8K test id, GSM1k id; every derived file maps to its source problem through orig_id) is boilerplate. Targets:
    GSM8K test (raw parquet), the GSM8K-Aug test split, GSM1k (raw arrow), and data/eval/gsm8k_mc, gsm8k_mc_v2,
    gsm8k_s*, gsm1k_mc, gsm_plus_mc, gsm_symbolic_mc (prompt stems). build_gsm8k_socratic's masked word-SET Jaccard >= .5
    is not used against the test sets: on GSM8K-Aug it flags unrelated short problems that share function words
    ("Laura has 30 candies. She gives 2 candies to each of her 5 friends. ..." vs "Erin has 7 lollipops. ...", 4% of
    the rows in a 3k sample, all spot-checked hits unrelated);
  - val (the 393 problems of data/student_gsm8k_tf/val.jsonl, group = problem): the same detectors with the word-SET
    Jaccard >= .5 (its hits here are mostly GPT-4 rewrites of a val problem: names, objects and numbers changed), then
    seed attribution (Seeds): the row's nearest GSM8K-train problem by masked trigram Jaccard is a val problem with
    Jaccard >= .12. GSM8K-Aug rewrites every GSM8K-train problem, val ones included; loose rewrites below these
    thresholds cannot be told from ordinary GSM-style problems.

Outputs:
  data/student_gsm8kaug_tf/train.jsonl       every clean row, in GSM8K-Aug order
  data/student_gsm8kaug_tf/val.jsonl         data/student_gsm8k_tf_si/val.jsonl copied: the 393 GSM8K val rows of
                                             data/student_gsm8k_tf/val.jsonl byte for byte plus value_nodes /
                                             worked_steps (both md5s in summary.json)
  data/student_gsm8kaug_tf_{24k,96k}/        nested seeded subsets, stratified by n_steps (largest remainder), rows in
                                             the full file's order; the same val
  summary.json (filters, decontamination, step counts, node stats, audits, md5s; token stats after --tokens),
  audit.txt (5 rows in full), tokens.json (--tokens).
Two streaming passes (7 GB machine): pass 1 decides every row (cheap checks, finals, screen) and keeps an index; pass
2 builds the trees and writes all files at once (md5s computed while writing).

  python3 -P scripts/build_gsm8kaug_tf.py                                    # pandas / pyarrow (system python3)
  ~/.venvs/sev/bin/python scripts/build_gsm8kaug_tf.py --tokens              # tokenizers + torch (token report)
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import math
import random
import re
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_gsm8k_tf as T  # noqa: E402
import build_gsm8k_ops as B  # noqa: E402
import build_gsm8k_si as SI  # noqa: E402

_SOC = T.soc()
T.soc = lambda: _SOC        # build_gsm8k_tf.soc() inserts into sys.path on every call (61k calls / 1.5k rows here)

SOURCE = T.SOURCE                       # "gsm8k_tf": the arms' --tree-sources and problem_no_options key on it
TREE_SOURCE = "gsm8kaug_tf"
PREFIX = "gsm8kaug-train-"
GROUP = "gsm8kaug_tf"
RAW_AUG = ROOT / "data/raw/gsm8k_aug"
RAW_NL = ROOT / "data/raw/gsm8k_aug_nl"
REVS = {"whynlp/gsm8k-aug": "62440e946e", "whynlp/gsm8k-aug-nl": "88f8ada364"}
LICENCE = ("unspecified: whynlp/gsm8k-aug and gsm8k-aug-nl state no licence (GPT-4 augmentation of GSM8K train, "
           "GSM8K MIT; Deng et al. 2023)")
VAL_SRC = ROOT / "data/student_gsm8k_tf_si/val.jsonl"       # = data/student_gsm8k_tf/val.jsonl + value_nodes, chains
VAL_TF = ROOT / "data/student_gsm8k_tf/val.jsonl"
SUBSETS = {"24k": 24000, "96k": 96000}
RECIPES = (("qo2", 0.4), ("inter", 0.3), ("slip", 0.3))
EVAL_FILES = ("gsm8k_mc", "gsm8k_mc_v2", "gsm1k_mc", "gsm_plus_mc", "gsm_symbolic_mc")   # + gsm8k_s*.jsonl
BOILER = 5
JACCARD = 0.5
NL_Q, EQ_Q = "Is this step correct? ", "Is it true that "
NL_OPP, EQ_OPP = "Is this step wrong? ", "Is it false that "
BLANK = "___"

STEP = re.compile(r"^<<([^<>=]*)=([^<>=]*)>>$")
EXPR_CHARS = re.compile(r"^[\d\s.,$()+\-*/xX×÷^%]+$")
PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
DEC = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)$")
FRACV = re.compile(r"^(\d+)/(\d+)$")
NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+"
NUMTOK = re.compile(rf"(?<![\w.,])(?:{NUM})(?![\w]|[.,]\d)")
RESULT = re.compile(rf"=\s*\$?\s*({NUM})(?![\w]|[.,]\d)(?!\s*(?:[-+*/×÷^]|x\s*[\d$(])\s*[\d$(.])")
LHS = re.compile(r"[\d\s.,$()+\-*/xX×÷%]*$")
WORD = re.compile(r"[A-Za-z]{2,}")
TOK = re.compile(r"[a-z0-9]+")


# ------------------------------------------------------------------------------------------------------ numbers
def frac(text: str) -> Fraction | None:
    """An integer, a decimal or p/q (after removing $ , and spaces), exactly; None otherwise."""
    t = text.strip().replace(",", "").replace("$", "").replace(" ", "")
    if DEC.match(t):
        return Fraction(t)
    m = FRACV.match(t)
    if m and int(m.group(2)):
        return Fraction(int(m.group(1)), int(m.group(2)))
    return None


def decimals_of(text: str) -> int:
    return len(text.partition(".")[2])


def clean_expr(e: str) -> str | None:
    """The expression with "P%" read as (P/100), or None when it has any other symbol."""
    e = e.strip()
    if not e or not EXPR_CHARS.match(e):
        return None
    e = PCT.sub(r"(\1/100)", e)
    return None if "%" in e else e


def eval_exact(e: str) -> tuple[ast.AST | None, Fraction | None]:
    """(parsed tree, exact value) of a clean expression; the value is None when it does not parse or execute."""
    S = T.soc()
    tree = S.parse_expr(e)
    if tree is None:
        return None, None
    try:
        S.run(tree)                                  # power guard, division by zero, overflow
        return tree, B.exact_eval(tree)
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return tree, None


# ------------------------------------------------------------------------------------------------------ steps
def verify_row(question: str, steps: list[str], answer: str) -> tuple[str | None, list[tuple[str, str]], Fraction]:
    """(drop reason or None, [(clean expr, exact value text)], gold) for one GSM8K-Aug row."""
    q = question.strip()
    if not q or "\n" in q or "\r" in q or "Question:" in q or "Options:" in q:
        return "question_shape", [], None
    if not steps:
        return "no_steps", [], None
    out = []
    for s in steps:
        m = STEP.match(s.strip())
        if not m:
            return "step_format", [], None
        e = clean_expr(m.group(1))
        if e is None:
            return "expr_symbol", [], None
        stated = frac(m.group(2))
        if stated is None:
            return "value_unparsable", [], None
        tree, exact = eval_exact(e)
        if tree is None or exact is None:
            return "expr_unparsable", [], None
        if exact != stated:
            d = decimals_of(m.group(2).strip().replace(",", "").replace("$", ""))
            rounded = Fraction(round(exact * 10 ** d), 10 ** d) == stated
            return "value_rounded" if rounded else "value_wrong", [], None
        vt = B.fmt(stated)
        if vt is None:
            return "value_over_4_decimals", [], None
        out.append((e, vt))
    gold = frac(answer)
    if gold is None:
        return "answer_not_number", [], None
    if Fraction(out[-1][1]) != gold:
        return "last_step_not_answer", [], None
    if gold <= 0 or not T.exact2(float(gold)):
        return "answer_not_positive_2dec", [], None
    return None, out, gold


def parse_verified(i: int, question: str, eqs: list[tuple[str, str]], gold: Fraction) -> dict:
    """build_gsm8k_socratic.parse_problem on the verified equations (one calculator note per step, no sub-question)."""
    S = T.soc()
    ans = "\n".join(f"_ ** <<{e}={v}>>{v}" for e, v in eqs) + f"\n#### {B.fmt(gold)}"      # "_": no sub-question
    p = S.parse_problem(i, question.strip(), ans)
    if p["status"] == "ok" and (p["gold_step"] != len(p["steps"]) - 1 or len(p["steps"]) != len(eqs)):
        p["status"] = "gold_step_not_last"
    return p


# ------------------------------------------------------------------------------------------------------ NL sentences
def results(sent: str) -> list[tuple[int, int, int, Fraction]]:
    """(position of "=", start, end of the number, value) of every result "= x" of a sentence (x not followed by an
    operator, i.e. not the start of a longer expression)."""
    out = []
    for m in RESULT.finditer(sent):
        v = frac(m.group(1))
        if v is not None:
            out.append((m.start(), m.start(1), m.end(1), v))
    return out


def align(values: list[Fraction], sents: list[str]) -> list[int] | None:
    """Sentence index per equation: order-preserving and injective, each sentence's results all equal to its equation's
    value (greedy, earliest; it finds a map whenever one exists); None when there is none."""
    out, k = [], 0
    for v in values:
        while k < len(sents):
            rs = results(sents[k])
            if rs and all(r[3] == v for r in rs):
                break
            k += 1
        if k >= len(sents):
            return None
        out.append(k)
        k += 1
    return out


def num_tokens(sent: str) -> list[tuple[int, int, Fraction]]:
    out = []
    for m in NUMTOK.finditer(sent):
        v = frac(m.group())
        if v is not None:
            out.append((m.start(), m.end(), v))
    return out


PCT_CONSTS = {Fraction(100), Fraction(1, 100)}


def nl_spans(sent: str, v: Fraction, gold: Fraction, operands: list[Fraction] | None = None
             ) -> tuple[str | None, list[tuple[int, int]]]:
    """(reason the sentence cannot state step value v, or None; spans of v to replace in the No twin: the first result
    and the later number tokens equal to v). operands: the literals of the step's verified equation; each (but 100 and
    .01) must be written before the result, as itself or as a percentage (0.7 as 70%), so the sentence states the
    equation's computation and not some other one with the same result ("360 minutes = 6 hours." for 360/60)."""
    if not sent.strip() or "\n" in sent or "\r" in sent:
        return "sentence_shape", []
    rs = [r for r in results(sent) if r[3] == v]
    if not rs:
        return "no_result", []
    toks = num_tokens(sent)
    if any(x[2] == gold for x in toks):
        return "gold_in_sentence", []
    start = rs[0][1]
    if any(x[2] == v and x[0] < start for x in toks):
        return "value_before_result", []
    before = {x[2] for x in toks if x[0] < start}
    if any(x not in PCT_CONSTS and x not in before and x * 100 not in before for x in operands or []):
        return "operands_not_in_sentence", []
    spans = [(x[0], x[1]) for x in toks if x[2] == v and x[0] >= start]
    if not spans or spans[0] != (rs[0][1], rs[0][2]):
        return "result_not_a_token", []
    return None, spans


def fmt_like(token: str, x: Fraction) -> str | None:
    """x written in the number format of token (thousands separators, number of decimals), None when it cannot be."""
    d = decimals_of(token)
    y = x * 10 ** d
    if y.denominator != 1 or x < 0:
        return None
    n = y.numerator
    ip, fp = divmod(n, 10 ** d)
    s = f"{ip:,}" if "," in token else str(ip)
    return s + (f".{fp:0{d}d}" if d else "")


def nl_statement(sent: str, spans: list[tuple[int, int]], x: Fraction) -> str | None:
    """The sentence with the spans written as x (the true sentence when x is its value)."""
    out, last = [], 0
    for a, b in spans:
        t = fmt_like(sent[a:b], x)
        if t is None:
            return None
        out += [sent[last:a], t]
        last = b
    return " ".join(("".join(out) + sent[last:]).split())


def cloze(sent: str, v: Fraction, gold: Fraction) -> str | None:
    """The sentence with its computation "lhs = v" masked as ___, when the masked lhs is an expression whose exact value
    is v, v is written nowhere else, the gold nowhere, and >= 2 words remain; else None."""
    rs = [r for r in results(sent) if r[3] == v]
    if not rs or "\n" in sent:
        return None
    eq, a, b, _ = rs[0]
    m = LHS.search(sent[:eq])
    lhs = m.group()
    k = 0
    while k < len(lhs) and lhs[k] not in "0123456789($.":
        k += 1
    lhs_start = m.start() + k
    e = clean_expr(sent[lhs_start:eq])
    if e is None or not re.search(r"\d\s*[-+*/xX×÷^]", e):
        return None
    _, val = eval_exact(e)
    if val != v:
        return None
    c = " ".join((sent[:lhs_start] + BLANK + sent[b:]).split())
    if any(t[2] in (v, gold) for t in num_tokens(c)) or len(WORD.findall(c)) < 2 or c.count(BLANK) != 1:
        return None
    return c


def eq_text(step: dict) -> str:
    """A note as build_gsm8k_ops renders it ("600 × 30 ÷ 100"; fraction literals p/q as written)."""
    _, _, lits = B.split_note(step["tree"], set(), step["refs"])
    t, _ = B.render(step["tree"], {}, 0, lits)
    return t if t is not None else " ".join(step["expr"].replace("*", " × ").split())


# ------------------------------------------------------------------------------------------------------ finals
def build_final(p: dict, pid: str, seed: int) -> dict | None:
    """The final row of build_gsm8k_socratic.build_rows (recipe, options, order) without its node rows."""
    S = T.soc()
    qo = S.qo
    q, gold, steps, gs = p["question"], p["gold"], p["steps"], p["gold_step"]
    recipe = random.Random(f"{seed}/{pid}/recipe").choices([r for r, _ in RECIPES], [w for _, w in RECIPES])[0]
    orng = random.Random(f"gsm8kaug_soc/{pid}")
    chain = list(dict.fromkeys(s["value"] for s in steps if S.ok_num(s["value"], gold)))
    orng.shuffle(chain)
    sl = S.slip_finals(p, orng) if gs is not None else []
    pair, goldn = qo.partial_candidates(q, gold)
    part = pair + goldn
    orng.shuffle(part)
    cands = {"inter": chain[:1] + part + chain[1:], "slip": sl[:1] + part + sl[1:]}
    opts = used = kinds = None
    for r in [recipe] + [x for x in ("inter", "slip", "qo2") if x != recipe]:
        if r == "qo2":
            o = qo.qo_v2(q, gold, f"gsm8kaug_soc/{pid}")
        else:
            o = S.mixed_options(gold, cands[r], orng) if {"inter": chain, "slip": sl}[r] else None
        if o is not None:
            opts, used, kinds = [gold] + [x for x, _ in o], r, ["gold"] + [k for _, k in o]
            break
    if opts is None:
        return None
    texts = [T._fmt(x) for x in opts]
    if len(set(texts)) != len(texts):
        return None
    prompt, labs, gl, shown = S.mc(f"Question: {q}", texts, 0, orng)
    kind_of = dict(zip(texts, kinds))
    n = len(steps)
    return {"item_id": pid, "source": SOURCE, "group": f"{GROUP}/s{n}", "type": "choice", "prompt": prompt,
            "labels": labs, "gold_label": gl, "teacher": {L: float(L == gl) for L in labs}, "options": shown,
            "option_kinds": [kind_of[t] for t in shown], "recipe": used, "depth": max(1, n), "dep_level": p["level"],
            "subqs": [], "random_subqs": [], "rationale": "", "augment": None, "has_tree": False, "tree_fix": True,
            "row_kind": "final", "problem_id": pid, "node_status": p["status"], "licence_class": "UNK",
            "licence": LICENCE, "label_basis": "gold", "node_basis": "program", "pool_rank": None, "n_steps": n,
            "tree_source": TREE_SOURCE}


# ------------------------------------------------------------------------------------------------------ yes/no nodes
def nl_info(p: dict, nl_steps: list[str] | None, st: collections.Counter) -> dict:
    """Per step: the aligned sentence (or None) and its cloze; row alignment counted."""
    steps = p["steps"]
    gold = Fraction(B.fmt(Fraction(str(p["gold"]))))
    vals = [Fraction(s["vt"]) for s in steps]
    amap = align(vals, nl_steps) if nl_steps else None
    st["rows_nl_aligned" if amap is not None else "rows_nl_unaligned"] += 1
    info = {}
    for j, s in enumerate(steps):
        sent = nl_steps[amap[j]] if amap is not None else None
        info[j] = {"sent": sent, "cloze": cloze(sent, vals[j], gold) if sent else None}
    return info


def yesno_nodes(p: dict, info: dict, rng: random.Random, st: collections.Counter) -> list[dict]:
    """build_gsm8k_tf.tf_nodes with step statements in place of sub-questions (module docstring)."""
    S = T.soc()
    steps, gs, gold = p["steps"], p["gold_step"], p["gold"]
    goldf = Fraction(B.fmt(Fraction(str(gold))))
    blocked = {gs} | {j for j, s in enumerate(steps) if S.close(s["value"], gold)}
    st["blocked_gold_valued"] += len(blocked) - 1
    for j in range(gs):
        if j in blocked:
            continue
        if T.ancestors(steps, j) & blocked:
            blocked.add(j)
            st["blocked_downstream_of_gold_valued"] += 1
        elif T.reads_blocked(steps, j, blocked):
            blocked.add(j)
            st["blocked_uncertain_edge_to_gold_valued"] += 1
    gold_anc = T.ancestors(steps, gs)
    nodes, ids = [], {}
    for j, s in enumerate(steps):
        v = s["value"]
        if j in blocked:
            continue
        if not T.exact2(v):
            st["dropped_value_not_exact2" if v > 0 else "dropped_value_not_positive"] += 1
            continue
        others = [x["value"] for k, x in enumerate(steps) if k != j]
        w = T.wrong_value(v, gold, T.slip_values(p, j), others, rng)
        if w is None:
            st["dropped_no_wrong_value"] += 1
            continue
        vf, wf = Fraction(s["vt"]), Fraction(T._fmt(w[0]))
        texts, why = None, None
        sent = info[j]["sent"]
        if sent is not None:
            why, spans = nl_spans(sent, vf, goldf, [B.lit(c) for c in S.constants(s["tree"])])
            if why is None:
                yes, no = nl_statement(sent, spans, vf), nl_statement(sent, spans, wf)
                if yes is None or no is None or yes == no:
                    why = "wrong_value_format"
                else:
                    texts = {"v": (NL_Q + yes, NL_OPP + yes), "n": (NL_Q + no, NL_OPP + no)}
            st[f"nl_fallback_{why}" if why else "statement_nl"] += 1
        if texts is None:
            e = eq_text(s)
            texts = {sfx: (f"{EQ_Q}{e} = {T._fmt(x)}?", f"{EQ_OPP}{e} = {T._fmt(x)}?")
                     for sfx, x in (("v", v), ("n", w[0]))}
            st["statement_equation"] += 1
        par = [f"s{k + 1}v" for k in s["parents"] if f"s{k + 1}v" in ids]
        st["parent_edges"] += len(s["parents"])
        st["parent_edges_unbuilt"] += len(s["parents"]) - len(par)
        meta = {"step": j + 1, "expr": s["expr"], "value": v, "flags": s["flags"], "gold_ancestor": j in gold_anc,
                "node_depth": s["level"]}
        for suffix, x, truth, kind in (("v", v, True, None), ("n", w[0], False, w[1])):
            n = {"id": f"s{j + 1}{suffix}", "question": texts[suffix][0], "opposite": texts[suffix][1],
                 "truth": truth, "p_cot": 1.0 if truth else 0.0, "p_fresh": None, "type": "step", "depends_on": par,
                 "initial": None, "status": "direct", "probe": x, **meta}
            if kind:
                n["wrong"] = kind
                st[f"wrong_{kind}"] += 1
            nodes.append(n)
        ids[f"s{j + 1}v"] = j
    return nodes


# ------------------------------------------------------------------------------------------------------ rows
def build_row(final: dict, p: dict, info: dict, seed: int, st: collections.Counter) -> dict:
    """The full row: the final, yes/no nodes, value nodes (qop / opsel only where a cloze exists), worked steps."""
    out = dict(final)
    out["subqs"] = yesno_nodes(p, info, random.Random(f"{seed}/{final['problem_id']}"), st)
    out["has_tree"] = bool(out["subqs"])
    for j, s in enumerate(p["steps"]):                  # the quantity's description for qop / opsel
        s["subq"] = info[j]["cloze"] or ""
    vn = B.value_nodes(p, final["problem_id"], seed, collections.Counter())
    for n in vn:
        if not info[n["step"] - 1]["cloze"]:
            n["subq"] = ""
            for k in ("qop", "opsel"):
                n["renderings"].pop(k, None)
        for k in n["renderings"]:
            st[f"value_rendering_{k}"] += 1
    st["value_nodes"] += len(vn)
    out["value_nodes"] = vn
    out["worked_steps"] = SI.worked_steps(p, out, seed, st)
    return out


def verdict(i: int, q: str, steps: list[str], ans: str, test: "Screen", val: "Screen", seeds: "Seeds", seen: set,
            seed: int) -> tuple[str | None, dict | None, object]:
    """(drop reason or None, the parsed problem, the screen hits) of one row in pass-1 order: verification, parse,
    exact duplicate (seen: normalised questions kept so far; updated), test screen, val screen, seed attribution, the
    final's options."""
    why, eqs, gold = verify_row(q, steps, ans)
    if why:
        return why, None, None
    p = parse_verified(i, q, eqs, gold)
    if p["status"] != "ok":
        return f"status_{p['status']}", p, None
    nq = norm_text(q)
    if nq in seen:
        return "duplicate_question", p, None
    seen.add(nq)
    h = test.hits(q)
    if h:
        return "test_overlap", p, h
    h = val.hits(q)
    if h:
        return "val_near_duplicate", p, h
    sh = seeds.hit(q)
    if sh:
        return "val_seed_rewrite", p, sh
    if build_final(p, f"{PREFIX}{i}", seed) is None:
        return "no_final_options", p, None
    return None, p, None


def build_item(i: int, q: str, steps: list[str], ans: str, nq: str | None, nsteps: list[str] | None, seed: int,
               st: collections.Counter, pool_rank: int | None = None) -> dict:
    """The row of a GSM8K-Aug row that passed pass 1 (nq / nsteps: its gsm8k-aug-nl question and steps)."""
    why, eqs, gold = verify_row(q, steps, ans)
    if why:
        raise ValueError(f"row {i}: {why}")
    p = parse_verified(i, q, eqs, gold)
    for s, (e, vt) in zip(p["steps"], eqs):
        s["vt"] = vt
    st["nl_question_differs"] += nq != q
    info = nl_info(p, nsteps if nq == q else None, st)
    for j, s in enumerate(p["steps"]):                   # the step's sentence for build_gsm8k_ops' percent rule
        s["sent"] = info[j]["sent"] or ""
    final = build_final(p, f"{PREFIX}{i}", seed)
    final["pool_rank"] = pool_rank
    return build_row(final, p, info, seed, st)


# ------------------------------------------------------------------------------------------------------ audit
def audit_row(item: dict) -> list[str]:
    """Problems of one built row from its own fields (empty: clean): the final, every yes/no node (arithmetic, truth,
    twins, wording, parents, no gold), build_gsm8k_ops.audit_item and build_gsm8k_si.audit_item."""
    S = T.soc()
    from cotdistill.treeplan import problem_no_options
    bad, iid = [], item["item_id"]
    gold = T.gold_text(item)
    opts = [str(x) for x in item["options"]]
    body = "\n".join(f"{L}) {o}" for L, o in zip(item["labels"], opts))
    if (len(opts) != 5 or len(set(opts)) != 5 or item["labels"] != list("ABCDE") or opts.count(gold) != 1
            or item["teacher"] != {L: float(L == item["gold_label"]) for L in item["labels"]}
            or not item["prompt"].endswith("\nOptions:\n" + body)):
        bad.append("final options / labels / teacher")
    if item["source"] != SOURCE or item["n_steps"] != item["depth"] or item["tree_fix"] is not True:
        bad.append("item fields")
    try:
        if problem_no_options(item) + "\nOptions:\n" + body != item["prompt"]:
            bad.append("problem_no_options")
    except ValueError as e:
        bad.append(f"problem_no_options: {e}")
    by = {n["id"]: n for n in item["subqs"]}
    if len(by) != len(item["subqs"]):
        bad.append("duplicate node ids")
    for n in item["subqs"]:
        tree = S.parse_expr(n["expr"])
        got = B.exact_eval(tree) if tree is not None else None
        if got is None or float(got) != n["value"] or not T.exact2(n["value"]):
            bad.append(f"{n['id']}: {n['expr']!r} is not {n['value']}")
        if (n["truth"] is True) != (T._fmt(n["probe"]) == T._fmt(n["value"])) or n["p_cot"] != float(n["truth"]):
            bad.append(f"{n['id']}: truth")
        if n["id"].endswith("v") != n["truth"]:
            bad.append(f"{n['id']}: id")
        q = n["question"]
        if q.startswith(EQ_Q):
            if not q.endswith(f" = {T._fmt(n['probe'])}?") or n["opposite"] != EQ_OPP + q[len(EQ_Q):]:
                bad.append(f"{n['id']}: equation wording")
        elif q.startswith(NL_Q):
            if n["opposite"] != NL_OPP + q[len(NL_Q):] or not any(
                    x[2] == Fraction(T._fmt(n["probe"])) for x in num_tokens(q)):
                bad.append(f"{n['id']}: sentence wording")
        else:
            bad.append(f"{n['id']}: question form")
        if T._fmt(n["probe"]) == gold or T._fmt(n["value"]) == gold:
            bad.append(f"{n['id']}: probe or value equals the gold {gold}")
        if q.startswith(NL_Q) and Fraction(gold) in [x[2] for x in num_tokens(q[len(NL_Q):])]:
            bad.append(f"{n['id']}: the gold {gold} is written in the step sentence")
        twin = by.get(n["id"][:-1] + ("n" if n["truth"] else "v"))
        if not twin or twin["value"] != n["value"] or twin["truth"] == n["truth"] or twin["question"] == q:
            bad.append(f"{n['id']}: no Yes/No twin")
        if q.startswith(NL_Q) != (twin or {}).get("question", "").startswith(NL_Q):
            bad.append(f"{n['id']}: twin form differs")
        if "options" in n or "answer" in n:
            bad.append(f"{n['id']}: yes/no node looks multiple-choice")
        consts = [abs(float(c.value)) for c in S.constants(tree)] if tree is not None else []
        depth = 0
        for d in n["depends_on"]:
            par = by.get(d)
            if not par or not par["truth"]:
                bad.append(f"{n['id']}: parent {d}")
                continue
            if not any(S.matches(c, par["value"]) for c in consts):
                bad.append(f"{n['id']}: parent {d} not an operand")
            depth = max(depth, par["node_depth"] + 1)
        if n["node_depth"] < depth:
            bad.append(f"{n['id']}: node_depth")
        if not isinstance(n["step"], int) or n["step"] < 1:
            bad.append(f"{n['id']}: step")
    b1, _ = B.audit_item(item)
    b2, _ = SI.audit_item(item)
    for n in item["value_nodes"]:
        for k in ("qop", "opsel"):
            r = n["renderings"].get(k)
            if r and (not n["subq"] or BLANK not in n["subq"]):
                bad.append(f"{n['id']}/{k}: without a cloze")
    return [f"{iid}: {x}" for x in bad] + b1 + b2


# ------------------------------------------------------------------------------------------------------ screen
def toks(text: str, mask: bool = False) -> list[str]:
    w = TOK.findall(text.lower().replace("’", "'"))
    return ["<n>" if x.isdigit() else x for x in w] if mask else w


def norm_text(text: str) -> str:
    return " ".join(toks(text))


DETECTORS = (("word8", 8, False), ("raw13", 13, False), ("masked10", 10, True))


def gram_keys(w: list[str], n: int) -> set[int]:
    return {hash(tuple(w[i:i + n])) for i in range(len(w) - n + 1)}


def word_set(text: str, kind: str) -> set:
    """The masked word set ("uni": digit-masked words) or the set of masked word trigrams ("tri") of a text."""
    w = toks(text, True)
    return set(w) if kind == "uni" else gram_keys(w, 3)


class Screen:
    """Overlap detectors against a fixed target set (texts with a group = the underlying problem, and a name for
    reports): normalised-text equality, word 8-grams, raw 13-grams, masked 10-grams (an n-gram in >= boiler groups is
    boilerplate) and the Jaccard of masked word sets (sets "uni") or of masked word-trigram sets ("tri") >= jaccard
    (exact, prefix filtering on the target side)."""

    def __init__(self, targets: list[tuple[str, str, str]], boiler: int = BOILER, jaccard: float = JACCARD,
                 sets: str = "tri"):
        self.boiler, self.t, self.sets = boiler, jaccard, sets
        self.jname = f"jaccard_{sets}{int(round(jaccard * 100))}"
        self.exact = {}
        self.grams = {name: {} for name, _, _ in DETECTORS}
        for text, group, name in targets:
            self.exact.setdefault(norm_text(text), (group, name))
            w, wm = toks(text), toks(text, True)
            for det, n, mask in DETECTORS:
                d = self.grams[det]
                for g in gram_keys(wm if mask else w, n):
                    cur = d.get(g)
                    if cur is None:
                        d[g] = (name, group)
                    elif cur is not True and cur[1] != group:
                        gs = cur[1] if isinstance(cur[1], set) else {cur[1]}
                        gs.add(group)
                        d[g] = True if len(gs) >= boiler else (cur[0], gs)
        self.n_boiler = {det: sum(v is True for v in d.values()) for det, d in self.grams.items()}
        uniq = {}
        for text, group, name in targets:
            ws = frozenset(word_set(text, sets))
            if ws:
                uniq.setdefault(ws, (group, name))
        self.B = list(uniq)
        self.B_meta = [uniq[x] for x in self.B]
        df = collections.Counter(x for b in self.B for x in b)
        self.key = lambda x: (df.get(x, 0), x)                      # noqa: E731
        self.index = collections.defaultdict(list)
        for j, b in enumerate(self.B):
            o = sorted(b, key=self.key)
            for x in o[:len(o) - math.ceil(jaccard * len(o)) + 1]:
                self.index[x].append(j)

    def jaccard(self, a: set) -> tuple[float, int]:
        if not a:
            return 0.0, -1
        o = sorted(a, key=self.key)
        cand = {j for x in o[:len(o) - math.ceil(self.t * len(o)) + 1] for j in self.index.get(x, ())}
        best = (0.0, -1)
        for j in cand:
            b = self.B[j]
            if not (self.t * len(b) <= len(a) <= len(b) / self.t):
                continue
            jac = len(a & b) / len(a | b)
            if jac > best[0]:
                best = (jac, j)
        return best

    def hits(self, text: str) -> dict[str, str]:
        """detector -> the name of a target it matched (empty: clean)."""
        out = {}
        e = self.exact.get(norm_text(text))
        if e:
            out["exact"] = e[1]
        w, wm = toks(text), toks(text, True)
        for det, n, mask in DETECTORS:
            d = self.grams[det]
            for g in gram_keys(wm if mask else w, n):
                v = d.get(g)
                if v is not None and v is not True:
                    out[det] = v[0]
                    break
        jac, j = self.jaccard(word_set(text, self.sets))
        if jac >= self.t:
            out[self.jname] = self.B_meta[j][1]
        return out


class Seeds:
    """Seed attribution: GSM8K-Aug rows are GPT-4 rewrites of GSM8K-train problems (names, objects and numbers
    changed, the template kept), which the n-gram detectors catch only when the wording is kept. A row whose nearest
    GSM8K-train problem by masked word-trigram Jaccard (candidates: problems sharing a trigram that occurs in <= cap
    of them; the best of the 20 with most shared ones) is one of the held-out ids with Jaccard >= floor is attributed
    to that problem."""

    def __init__(self, questions: list[str], held_out: set[int], floor: float = 0.12, cap: int = 300):
        self.S = [word_set(q, "tri") for q in questions]
        self.held, self.floor, self.cap = held_out, floor, cap
        self.post = collections.defaultdict(list)
        for j, x in enumerate(self.S):
            for g in x:
                self.post[g].append(j)

    def nearest(self, text: str) -> tuple[float, int]:
        import itertools
        a = word_set(text, "tri")
        c = collections.Counter(itertools.chain.from_iterable(
            self.post[g] for g in a if g in self.post and len(self.post[g]) <= self.cap))
        return max(((len(a & self.S[j]) / len(a | self.S[j]), j) for j, _ in c.most_common(20)), default=(0.0, -1))

    def hit(self, text: str) -> tuple[float, int] | None:
        jac, j = self.nearest(text)
        return (jac, j) if j in self.held and jac >= self.floor else None


def eval_stem(prompt: str) -> str:
    s = prompt.partition("\nOptions:")[0]
    return s[len("Question: "):] if s.startswith("Question: ") else s


def test_targets() -> tuple[list[tuple[str, str, str]], dict]:
    """GSM8K test, the GSM8K-Aug test split, GSM1k and the GSM-family data/eval files; group = source problem."""
    import pandas as pd
    out, info = [], {}
    tq = list(pd.read_parquet(ROOT / "data/raw/bench_math/gsm8k/main/test-00000-of-00001.parquet").question)
    by_norm = {norm_text(q): f"gsm8k-test-{i}" for i, q in enumerate(tq)}
    out += [(q, f"gsm8k-test-{i}", "gsm8k_test") for i, q in enumerate(tq)]
    aq = list(pd.read_parquet(RAW_AUG / "data/test-00000-of-00001.parquet").question)
    out += [(q, by_norm.get(norm_text(q), f"gsm8kaug-test-{i}"), "gsm8kaug_test") for i, q in enumerate(aq)]
    info["gsm8kaug_test_equal_to_gsm8k_test"] = sum(norm_text(q) in by_norm for q in aq)
    import glob
    import pyarrow as pa
    n1k = 0
    for a in sorted(glob.glob(str(ROOT / "data/raw/bench_math/gsm1k/data/test/*.arrow"))):
        with open(a, "rb") as fa:
            tab = pa.ipc.open_stream(fa).read_all()
        for q in tab.column("question").to_pylist():
            out.append((q, f"gsm1k-{n1k}", "gsm1k"))
            n1k += 1
    info["gsm1k_raw"] = n1k
    files = list(EVAL_FILES) + sorted(Path(x).stem for x in glob.glob(str(ROOT / "data/eval/gsm8k_s*.jsonl")))
    info["eval_files"] = {}
    for name in files:
        path = ROOT / "data/eval" / f"{name}.jsonl"
        k = 0
        for r in read_jsonl(path):
            key = r.get("orig_id") or r["item_id"]
            if key.startswith("gsm8k-test-"):
                key = "gsm8k-test-" + key.rsplit("-", 1)[1]
            out.append((eval_stem(r["prompt"]), key, f"eval/{name}"))
            k += 1
        info["eval_files"][name] = {"rows": k, "md5": md5_file(path)}
    info["targets"] = dict(collections.Counter(n for _, _, n in out))
    return out, info


def gsm8k_train_seeds(vt: list[tuple[str, str, str]]) -> tuple["Seeds", list[str]]:
    """Seeds over every GSM8K-train question (raw parquet), the val problems held out (checked: each val stem is its
    train question)."""
    import pandas as pd
    q = list(pd.read_parquet(ROOT / "data/raw/bench_math/gsm8k/main/train-00000-of-00001.parquet").question)
    held = set()
    for text, iid, _ in vt:
        k = int(iid.rsplit("-", 1)[1])
        if not iid.startswith("gsm8k-train-") or norm_text(q[k]) != norm_text(text):
            raise ValueError(f"val {iid} is not GSM8K train question {k}")
        held.add(k)
    return Seeds(q, held), q


def val_targets(path: Path) -> list[tuple[str, str, str]]:
    return [(eval_stem(r["prompt"]), r["item_id"], "val") for r in read_jsonl(path)]


# ------------------------------------------------------------------------------------------------------ subsets
def subset_key(seed: int, pid: str) -> str:
    return hashlib.sha256(f"gsm8kaug_tf/subset/{seed}/{pid}".encode()).hexdigest()


def quotas(counts: dict[int, int], n: int) -> dict[int, int]:
    """Largest-remainder allocation of n over strata in proportion to counts."""
    tot = sum(counts.values())
    raw = {s: n * c / tot for s, c in counts.items()}
    q = {s: int(math.floor(x)) for s, x in raw.items()}
    for s in sorted(raw, key=lambda s: (-(raw[s] - q[s]), s))[:n - sum(q.values())]:
        q[s] += 1
    return q


def choose_subsets(index: list[tuple[str, int, str]], sizes: dict[str, int]) -> dict[str, set]:
    """Nested stratified subsets: per n_steps stratum, rows ranked by their seeded key; subset s takes the first
    quota_s(stratum) rows (quota of a smaller subset capped by that of every larger one, so they nest)."""
    by = collections.defaultdict(list)
    for pid, ns, key in index:
        by[ns].append((key, pid))
    for v in by.values():
        v.sort()
    counts = {s: len(v) for s, v in by.items()}
    qs = {name: quotas(counts, n) for name, n in sizes.items()}
    names = sorted(sizes, key=lambda x: -sizes[x])
    for a, b in zip(names, names[1:]):
        for s in counts:
            if qs[b][s] > qs[a][s]:
                raise ValueError(f"subset {b} stratum {s} larger than in {a}")
    return {name: {pid for s, v in by.items() for _, pid in v[:qs[name][s]]} for name in sizes}


# ------------------------------------------------------------------------------------------------------ io
def num_hist(c: collections.Counter, prefix: str) -> dict:
    return {int(k[len(prefix):]): v for k, v in sorted(((k, v) for k, v in c.items() if k.startswith(prefix)),
                                                         key=lambda x: int(x[0][len(prefix):]))}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def parquet_rows(path: Path, batch: int = 4000):
    import pyarrow.parquet as pq
    for b in pq.ParquetFile(path).iter_batches(batch_size=batch):
        yield from b.to_pylist()


def rows(limit: int | None = None, nl: bool = False):
    """(row index, question, steps, answer, nl question, nl steps) of the GSM8K-Aug train parquet, streamed; the NL
    file is read row for row alongside (same row count, checked)."""
    a_it = parquet_rows(RAW_AUG / "data/train-00000-of-00001.parquet")
    n_it = parquet_rows(RAW_NL / "data/train-00000-of-00001.parquet") if nl else None
    for i, a in enumerate(a_it):
        if limit is not None and i >= limit:
            return
        n = next(n_it) if nl else {"question": None, "steps": None}
        yield i, a["question"], list(a["steps"]), a["answer"], n["question"], list(n["steps"] or [])
    if nl and limit is None and next(n_it, None) is not None:
        raise ValueError("gsm8k-aug-nl has more rows than gsm8k-aug")


def stage(msg: str, t0: float) -> None:
    import resource
    print(f"[{time.time() - t0:7.1f}s] {msg}  (peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB)",
          flush=True)


# ------------------------------------------------------------------------------------------------------ build
def pass1(args, t0: float) -> tuple[list, dict]:
    """Every row's verdict: verification, parse, the final, exact duplicates, the test and val screens."""
    S = T.soc()
    filt = collections.Counter()
    dec = {"test_hits": collections.Counter(), "test_hits_by_target": collections.Counter(),
           "test_only_one_detector": collections.Counter(), "val_hits": collections.Counter(),
           "val_seed_jaccard": collections.Counter(), "examples_test": [], "examples_val": [], "examples_val_seed": []}
    tg, tinfo = test_targets()
    test = Screen(tg, sets="tri")
    del tg
    vt = val_targets(VAL_TF)
    val = Screen(vt, sets="uni")
    seeds, train_q = gsm8k_train_seeds(vt)
    dec["targets"] = tinfo
    dec["boilerplate_grams"] = {"test": test.n_boiler, "val": val.n_boiler}
    dec["rules"] = {"test": f"exact, {', '.join(d for d, _, _ in DETECTORS)}, {test.jname} (boilerplate: an n-gram in "
                            f">= {BOILER} test problems)",
                    "val": f"exact, {', '.join(d for d, _, _ in DETECTORS)}, {val.jname} vs the 393 val problems; "
                           f"then seed attribution (Seeds: nearest of the {len(train_q)} GSM8K-train problems by masked "
                           f"trigram Jaccard is a val problem, Jaccard >= {seeds.floor})"}
    stage(f"screens built ({len(test.B)} distinct test trigram sets)", t0)
    seen = set()
    index = []
    for i, q, steps, ans, _, _ in rows(args.limit):
        filt["rows"] += 1
        why, p, h = verdict(i, q, steps, ans, test, val, seeds, seen, args.seed)
        if why == "test_overlap":
            dec["test_hits"].update(h.keys())
            dec["test_hits_by_target"].update({f"{k}:{v}" for k, v in h.items()})
            if len(h) == 1:
                dec["test_only_one_detector"].update(h.keys())
            if len(dec["examples_test"]) < 8:
                dec["examples_test"].append({"row": i, "how": h, "question": q[:200]})
        elif why == "val_near_duplicate":
            dec["val_hits"].update(h.keys())
            if len(dec["examples_val"]) < 8:
                dec["examples_val"].append({"row": i, "how": h, "question": q[:200]})
        elif why == "val_seed_rewrite":
            dec["val_seed_jaccard"][f"{int(h[0] * 20) / 20:.2f}"] += 1
            if len(dec["examples_val_seed"]) < 8:
                dec["examples_val_seed"].append({"row": i, "jaccard": round(h[0], 3), "question": q[:200],
                                                 "val_problem": train_q[h[1]][:200]})
        if filt["rows"] % 50000 == 0:
            stage(f"pass 1: {filt['rows']} rows, kept {filt['kept']}", t0)
        if why:
            filt[f"drop_{why}"] += 1
            continue
        pid = f"{PREFIX}{i}"
        filt["kept"] += 1
        index.append((pid, len(p["steps"]), subset_key(args.seed, pid)))
    stage(f"pass 1 done: {dict(filt)}", t0)
    dec = {k: dict(v) if isinstance(v, collections.Counter) else v for k, v in dec.items()}
    return index, {"filters": dict(filt), "decontam": dec}


def show_row(item: dict, raw: dict) -> str:
    out = [f"=== {item['item_id']}  ({item['n_steps']} steps, gold {T.gold_text(item)}, recipe {item['recipe']}, "
           f"option_kinds {item['option_kinds']})", item["prompt"], "",
           "GSM8K-Aug equations: " + "  ".join(raw["steps"]), "GSM8K-Aug-NL steps:"]
    out += [f"  {k + 1}. {s}" for k, s in enumerate(raw["nl_steps"] or [])]
    out += ["", f"yes/no nodes (subqs, {len(item['subqs'])}):"]
    for n in item["subqs"]:
        out.append(f"  [{n['id']}] {'Yes' if n['truth'] else 'No '}  {n['question']}   (step {n['step']}, "
                   f"depends_on {n['depends_on']}, depth {n['node_depth']}{', wrong ' + n['wrong'] if 'wrong' in n else ''})")
    out += ["", f"value nodes ({len(item['value_nodes'])}):"]
    for n in item["value_nodes"]:
        out.append(f"  [{n['id']}] {B.expr_text(n['kind'], n['a'], n['op'], n['b'])} = {n['value']}   (note "
                   f"{n['expr']}, depends_on {n['depends_on']}, depth {n['node_depth']}"
                   f"{', gold step' if n['gold_step'] else ''})")
        for k in B.RENDERINGS:
            r = n["renderings"].get(k)
            if r:
                opts = "  ".join(f"{'*' if i == r['answer'] else ' '}{L}) {o}" for i, (L, o) in
                                 enumerate(zip("ABC", r["options"])))
                out.append(f"      {k:5s} {r['question']}   {opts}")
            else:
                out.append(f"      {k:5s} (absent)")
    ws = item["worked_steps"]
    out += ["", "worked steps:" if ws else "worked steps: null"]
    out += [f"  Step {i}: {ln['text']}    (SIS: {ln['wrong_text']})" for i, ln in enumerate(ws or [], 1)]
    return "\n".join(out) + "\n"


def build(args) -> None:
    t0 = time.time()
    out = ROOT / args.out
    subdirs = {name: ROOT / f"{args.out}_{name}" for name in SUBSETS}
    index, rep = pass1(args, t0)
    sizes = dict(SUBSETS)
    if args.limit:                                          # dry run: subsets scaled to the rows read
        sizes = {k: max(1, len(index) * v // 385620) for k, v in SUBSETS.items()}
    member = choose_subsets(index, sizes)
    pool = {pid: r for r, (_, pid) in enumerate(sorted((key, pid) for pid, _, key in index))}
    keep = {pid for pid, _, _ in index}
    del index
    stage(f"subsets chosen {({k: len(v) for k, v in member.items()})}", t0)
    # ---- pass 2
    st = collections.Counter()
    fails = []
    files = {"full": out / "train.jsonl", **{k: d / "train.jsonl" for k, d in subdirs.items()}}
    for f in files.values():
        f.parent.mkdir(parents=True, exist_ok=True)
    fh = {k: open(f.with_suffix(".jsonl.tmp"), "w") for k, f in files.items()}
    md5 = {k: hashlib.md5() for k in files}
    per = {k: collections.Counter() for k in files}
    audit_pick, audit_raw, cue_rows = [], {}, []
    want_audit = [lambda it: it["n_steps"] == 4 and it["subqs"] and it["subqs"][0]["question"].startswith(NL_Q),
                  lambda it: any(n["question"].startswith(EQ_Q) for n in it["subqs"]),
                  lambda it: any(n["kind"] == "pct" for n in it["value_nodes"]),
                  lambda it: it["n_steps"] >= 6 and it["worked_steps"],
                  lambda it: any("opsel" in n["renderings"] for n in it["value_nodes"]) and it["recipe"] == "slip"]
    for i, q, steps, ans, nq, nsteps in rows(args.limit, nl=True):
        pid = f"{PREFIX}{i}"
        if pid not in keep:
            continue
        item = build_item(i, q, steps, ans, nq, nsteps, args.seed, st, pool_rank=pool[pid])
        bad = audit_row(item)
        if bad:
            fails += bad
        line = json.dumps(item, ensure_ascii=False) + "\n"
        enc = line.encode()
        targets = ["full"] + [k for k in SUBSETS if pid in member[k]]
        for k in targets:
            fh[k].write(line)
            md5[k].update(enc)
            c = per[k]
            c["items"] += 1
            c[f"n_steps/{item['n_steps']}"] += 1
            c["with_tree"] += item["has_tree"]
            c["yesno_nodes"] += len(item["subqs"])
            c["yes"] += sum(n["truth"] for n in item["subqs"])
            c["yesno_nl"] += sum(n["question"].startswith(NL_Q) for n in item["subqs"])
            c["value_nodes"] += len(item["value_nodes"])
            c["value_nodes_qop"] += sum("qop" in n["renderings"] for n in item["value_nodes"])
            c["value_nodes_opsel"] += sum("opsel" in n["renderings"] for n in item["value_nodes"])
            c["with_worked_steps"] += item["worked_steps"] is not None
            c["worked_lines"] += len(item["worked_steps"] or [])
            c[f"gold_letter/{item['gold_label']}"] += 1
            c[f"recipe/{item['recipe']}"] += 1
            gold = T.gold_text(item)
            qnums = T.NUMTOK.findall(item["prompt"].partition("\nOptions:")[0])
            for n in item["subqs"]:
                c[f"node_depth/{n['node_depth']}"] += 1
                inq = gold in T.NUMTOK.findall(n["question"])
                c["leak_gold_number_in_node_text"] += inq
                c["leak_gold_number_in_node_text_not_in_problem"] += inq and gold not in qnums
                c["leak_probe_equals_gold"] += T._fmt(n["probe"]) == gold
        if pid in member["24k"]:
            cue_rows += T.cue_features(item)
        for k, f in enumerate(want_audit):
            if k not in [x[0] for x in audit_pick] and len(audit_pick) < 5 and f(item):
                audit_pick.append((k, item))
                audit_raw[item["item_id"]] = {"steps": steps, "nl_steps": nsteps if nq == q else None}
                break
        st["built"] += 1
        if st["built"] % 25000 == 0:
            stage(f"pass 2: {st['built']} built, {len(fails)} audit failures", t0)
    for k in files:
        fh[k].close()
        files[k].with_suffix(".jsonl.tmp").replace(files[k])
    stage(f"pass 2 done: {st['built']} rows", t0)
    # ---- val: the 393 GSM8K val rows (si version: + value_nodes, worked_steps)
    vlines = [x for x in VAL_SRC.read_text().split("\n") if x]
    tlines = [x for x in VAL_TF.read_text().split("\n") if x]
    same = len(vlines) == len(tlines) and all(
        json.dumps({k: v for k, v in json.loads(a).items() if k not in ("value_nodes", "worked_steps")},
                   ensure_ascii=False) == b for a, b in zip(vlines, tlines))
    if not same:
        fails.append("val: student_gsm8k_tf_si/val.jsonl is not student_gsm8k_tf/val.jsonl + value_nodes / chains")
    for d in [out] + list(subdirs.values()):
        shutil.copyfile(VAL_SRC, d / "val.jsonl")
    # ---- summary
    def file_stats(c: collections.Counter) -> dict:
        n = max(1, c["items"])
        return {"items": c["items"], "with_tree": c["with_tree"],
                "n_steps": num_hist(c, "n_steps/"),
                "yesno_nodes": c["yesno_nodes"], "yesno_per_item": round(c["yesno_nodes"] / n, 2),
                "yes_share": round(c["yes"] / max(1, c["yesno_nodes"]), 4),
                "yesno_nl_statement_share": round(c["yesno_nl"] / max(1, c["yesno_nodes"]), 3),
                "node_depth": num_hist(c, "node_depth/"),
                "value_nodes": c["value_nodes"], "value_nodes_per_item": round(c["value_nodes"] / n, 2),
                "value_nodes_with_qop": c["value_nodes_qop"], "value_nodes_with_opsel": c["value_nodes_opsel"],
                "with_worked_steps": c["with_worked_steps"],
                "worked_lines_per_chain": round(c["worked_lines"] / max(1, c["with_worked_steps"]), 2),
                "gold_letter": {k.split("/")[1]: v for k, v in sorted(c.items()) if k.startswith("gold_letter/")},
                "recipe": {k.split("/")[1]: v for k, v in sorted(c.items()) if k.startswith("recipe/")},
                "leaks": {k[5:]: v for k, v in sorted(c.items()) if k.startswith("leak_")}}
    summary = {"script": "scripts/build_gsm8kaug_tf.py", "args": vars(args), "sources": REVS,
               "raw_md5": {str(p.relative_to(ROOT)): md5_file(p) for p in
                           (RAW_AUG / "data/train-00000-of-00001.parquet", RAW_NL / "data/train-00000-of-00001.parquet")},
               "md5": {}, "files": {}}
    for k, f in files.items():
        key = f"{f.parent.name}/{f.name}"
        summary["md5"][key] = md5[k].hexdigest()
        summary["files"][key] = file_stats(per[k])
    summary["md5"]["val.jsonl (all dirs)"] = md5_file(VAL_SRC)
    summary["val"] = {"copied_from": str(VAL_SRC.relative_to(ROOT)), "md5": md5_file(VAL_SRC),
                      "equals_student_gsm8k_tf_val_without_value_nodes_worked_steps": same,
                      "student_gsm8k_tf_val_md5": md5_file(VAL_TF), "items": len(vlines)}
    summary["subsets"] = {f"{out.name}_{k}": {"n": len(member[k]), "seeded_key": "sha256('gsm8kaug_tf/subset/<seed>/"
                                              "<pid>'), first quota per n_steps stratum (largest remainder)",
                                              "nested_in": [f"{out.name}_{x}" for x in SUBSETS if SUBSETS[x] > SUBSETS[k]]
                                              + [out.name]} for k in SUBSETS}
    summary.update(rep)
    summary["build"] = dict(sorted(st.items()))
    summary["cues_gfpq_view_24k"] = {
        "n": len(cue_rows), "probe_in_problem": T.stump(cue_rows, 1), "probe_round10": T.stump(cue_rows, 2),
        "probe_even": T.stump(cue_rows, 3), "probe_above_all_problem_numbers": T.stump(cue_rows, 4),
        "probe_one_op_of_problem_numbers": T.stump(cue_rows, 5),
        "no_probes_above_value": round(sum(r[6] for r in cue_rows if r[6] is not None)
                                       / max(1, sum(r[6] is not None for r in cue_rows)), 3),
        "note": "single-feature stumps on the yes/no targets (build_gsm8k_tf.cue_features); every node also states "
                "its computation, so a node is decidable by checking the arithmetic"}
    summary["audit"] = {"failures": len(fails), "examples": fails[:30], "rows_audited": st["built"]}
    summary["node_design"] = {
        "yesno": "per non-final step: 'Is this step correct? <NL sentence>' (Yes) and the sentence with its result "
                 "replaced by build_gsm8k_tf.wrong_value (No) when the row's NL steps align 1:1 with its equations and "
                 "the sentence passes nl_spans; else 'Is it true that <expr> = <x>?'. Node selection, wrong values, "
                 "ids, parents and depths as build_gsm8k_tf.tf_nodes.",
        "value_nodes": "build_gsm8k_ops.value_nodes; qop / opsel use the NL cloze (computation masked as ___) as the "
                       "quantity's description, absent without one.",
        "worked_steps": "build_gsm8k_si.worked_steps"}
    (out / "audit.txt").write_text("\n".join(show_row(it, audit_raw[it["item_id"]]) for _, it in
                                             sorted(audit_pick, key=lambda x: x[0])))
    summary["audit_rows"] = [it["item_id"] for _, it in sorted(audit_pick, key=lambda x: x[0])]
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("filters", "decontam", "md5", "files", "audit")}, indent=1,
                     ensure_ascii=False)[:20000])
    stage("done", t0)
    if fails:
        print("AUDIT FAILED", fails[:10])
        sys.exit(1)


# ------------------------------------------------------------------------------------------------------ tokens
def tokens(args) -> None:
    """Token lengths and per-epoch tokens of the arms' views (needs tokenizers + torch: cotdistill.student builds the
    views exactly as the trainer does). Every row: the A2 final view and the SI stage-0 view (final + the whole chain);
    every row of the 24k subset and of data/student_gsm8k_tf_si/train.jsonl (the GSM8K reference): every view of A2,
    GFPQ, GFPQV (op), GFPQO (opsel) and SIB (4 stages + the final stage, final-only)."""
    from tokenizers import Tokenizer
    from cotdistill import student as M
    tok = Tokenizer.from_file(str(ROOT / T.TOKENIZER))
    n_tok = lambda s: len(tok.encode(s, add_special_tokens=False).ids)   # noqa: E731
    L = args.max_len

    def summ(xs: list[int]) -> dict:
        xs = sorted(xs)
        if not xs:
            return {}
        return {"n": len(xs), "mean": round(sum(xs) / len(xs), 1), "p50": xs[len(xs) // 2],
                "p99": xs[min(len(xs) - 1, int(0.99 * len(xs)))], "max": xs[-1],
                "share_le_max_len": round(sum(x <= L for x in xs) / len(xs), 5)}

    def arm_views(it: dict) -> dict[str, list[str]]:
        rng = random.Random(0)
        kw = dict(mode="plain", rng=rng, cap=0, view="statements+question", sources={SOURCE})
        out = {"A2": [e.text for e in M.build_examples(it, final="teacher", subq="none", subq_target="cot",
                                                       lambda_sub=0.0, rng=random.Random(0)) if e.kind == "final"]}
        out["GFPQ"] = [e.text for e in M.build_full_tree(it, **kw)]
        out["GFPQV"] = [e.text for e in M.build_full_tree(dict(it), **dict(kw, rng=random.Random(0)),
                                                          node_format="value", node_render="op")]
        out["GFPQO"] = [e.text for e in M.build_full_tree(dict(it), **dict(kw, rng=random.Random(0)),
                                                          node_format="value", node_render="opsel")]
        g = M.build_full_tree(it, **dict(kw, rng=random.Random(0)))
        out["SIB"] = [e.text for k in range(5) for e in M.si_group(g, it, stage=k, n_stages=4, final_only=True,
                                                                 remove="back", seed=0)]
        return out

    rep = {"tokenizer": T.TOKENIZER, "max_len": L, "files": {}}
    for rel in (f"{args.out}/train.jsonl",):
        fin, si0 = [], []
        for it in read_jsonl(ROOT / rel):
            f = M.FINAL_TEMPLATE.format(problem=it["prompt"])
            fin.append(n_tok(f))
            ws = M.worked_steps(it)
            si0.append(n_tok(M.insert_chain(f, M.chain_block([s["text"] for s in ws]))) if ws else fin[-1])
        rep["files"][rel] = {"final_view": summ(fin), "si_stage0_view": summ(si0), "epoch_tokens_A2": sum(fin)}
        print(rel, rep["files"][rel], flush=True)
    for rel in (f"{args.out}_24k/train.jsonl", "data/student_gsm8k_tf_si/train.jsonl"):
        per_arm, lens, n = collections.Counter(), collections.defaultdict(list), 0
        for it in read_jsonl(ROOT / rel):
            n += 1
            for arm, texts in arm_views(it).items():
                ts = [n_tok(t) for t in texts]
                per_arm[arm] += sum(ts)
                lens[arm] += ts
        rep["files"][rel] = {"items": n, "epoch_tokens": dict(per_arm),
                             "tokens_per_item": {a: round(v / n, 1) for a, v in per_arm.items()},
                             "view_lengths": {a: summ(v) for a, v in lens.items()}}
        print(rel, json.dumps(rep["files"][rel]["tokens_per_item"]), flush=True)
    ref = rep["files"]["data/student_gsm8k_tf_si/train.jsonl"]
    sub = rep["files"][f"{args.out}_24k/train.jsonl"]
    full_n = rep["files"][f"{args.out}/train.jsonl"]["final_view"]["n"]
    rep["epoch_cost_vs_gsm8k"] = {}
    for name, n in (("24k", SUBSETS["24k"]), ("96k", SUBSETS["96k"]), ("full", full_n)):
        rep["epoch_cost_vs_gsm8k"][name] = {
            arm: {"tokens": round(sub["tokens_per_item"][arm] * n), "x_gsm8k": round(
                sub["tokens_per_item"][arm] * n / ref["epoch_tokens"][arm], 1)} for arm in ref["epoch_tokens"]}
    (ROOT / args.out / "tokens.json").write_text(json.dumps(rep, indent=1))
    s = json.loads((ROOT / args.out / "summary.json").read_text())
    s["tokens"] = {"file": f"{args.out}/tokens.json", "max_len": L,
                   "full_final_view": rep["files"][f"{args.out}/train.jsonl"]["final_view"],
                   "full_si_stage0_view": rep["files"][f"{args.out}/train.jsonl"]["si_stage0_view"],
                   "view_lengths_24k": sub["view_lengths"], "epoch_cost_vs_gsm8k": rep["epoch_cost_vs_gsm8k"]}
    (ROOT / args.out / "summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False))
    print(json.dumps(rep["epoch_cost_vs_gsm8k"], indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/student_gsm8kaug_tf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="dry run on the first N rows")
    ap.add_argument("--tokens", action="store_true", help="only the token report (needs tokenizers + torch)")
    ap.add_argument("--max-len", type=int, default=1536)
    args = ap.parse_args()
    if args.tokens:
        tokens(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
