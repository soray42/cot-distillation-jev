"""Tier-0 failure-attribution data (notes/next_round/failure_attribution/pilot_design.md §2.1, §3, §3.4): simulated
multi-agent group chats that solve numgen word problems, with programmatic fault injection and exact labels. CPU only,
no model and no API: every label comes from executing the simulator.

A trajectory is a numgen problem (numgen.generate: every intermediate value exact) solved by a team of 3-6 agents
(names drawn from pools): a Planner (writes the plan: one formula per operation over named variables), an optional
Reader (lists the values stated in the task), a Calculator (one operation per message) and/or a Coder plus an Executor
(writes a Python snippet / prints its output), a Verifier (re-checks one result per message, corrects it and has the
dependent results recomputed when it finds a mistake) and an optional Reporter (the final answer; otherwise the
Verifier or the Planner reports). Side remarks (a sum of two results nobody uses later) and chatter fill the log.

Semantics (exact, per step t, from the simulator; the trajectory record keeps them):
  V_t  valid: the step does what it should given the messages before it, taken as they are. Calculator / Coder: the
       plan's formula applied to the values as last stated (a step that follows a wrong plan or a wrong value is valid);
       Planner: the program's formulas; Reader: the task's numbers; Verifier: the checked step's correct local result
       (confirm if that step was valid, else correct it); Reporter: the current value of the plan's answer variable;
       Executor: always valid (prints what the code computes). V_t = No iff a fault of step t is active (its text
       differs from the correct local text, which the simulator renders too).
  C_t  true: every value, formula or answer the step states equals the program's truth.
  uses(t) the steps whose output step t reads (value provenance: the step that last established each value it uses,
       the plan for formulas, the code for the Executor, the checked step for the Verifier). D_{t,s} = s in uses(t).
  I_t  inherited: not C_t and V_t and some used step is not C (the builder asserts that every valid untrue step has one).
  R_t  (invalid steps only) re-simulation with step t's own fault removed, every other step's fault kept, succeeds.
  decisive = the earliest invalid step with R_t (the earliest mistake whose correction alone lets the task succeed);
       a trajectory is kept only if the run fails and such a step exists.
Faults are properties of steps, re-applied to whatever values a re-simulation gives them (an arithmetic slip stays a
slip); verifier faults are reactive (a false acceptance only exists while the checked step is wrong).

Fault catalogue (type: where):
  plan_op / plan_const / plan_skip   Planner: one formula with the inverse operation / a misread constant / an
                                     operation left out (later formulas use its input instead)
  given                              Reader (or Planner): a misread number (a relevant one is fatal, a "not needed" one
                                     never is)
  arith / operand / op / input       Calculator: wrong result / a misread operand / the wrong operation / the wrong
                                     earlier variable;  Coder: op / operand / input in the code
  premature                          Calculator or Verifier: declares an intermediate result final and terminates
  accept / botch / reject            Verifier: confirms a wrong result / corrects it to another wrong value / "corrects"
                                     a right result to a wrong value
  slip / wrongq                      Reporter: a copy slip / the value of another variable
  side                               an arithmetic slip in a side remark (never fatal)
"""
from __future__ import annotations

import dataclasses
import random
import re
from fractions import Fraction

from . import numgen as ng
from .treeplan import node_depths

SOURCE = "fa_tier0"
TREE_SOURCE = "fa_tier0_sim"
HEADER = "The log below comes from a multi-agent system that failed to solve its task."
Q_STEP = ("Question: Which step contains the decisive error, i.e. the earliest mistake whose correction would have "
          "let the system solve the task?")
Q_AGENT = "Question: Which agent made the decisive error that caused the failure?"
LETTERS = [chr(65 + i) for i in range(26)]
MAXV = 10 ** 7

POOLS = {
    "planner": ["Planner", "Task_Planner", "Manager", "Lead_Expert", "Orchestrator", "Coordinator", "Problem_Analyst"],
    "reader": ["Data_Extractor", "Reader", "Info_Collector", "Problem_Reader", "Input_Parser"],
    "calculator": ["Calculator", "Math_Expert", "Arithmetic_Expert", "Solver", "Calculation_Expert", "Quant_Solver"],
    "coder": ["Python_Expert", "Coder", "Code_Writer", "Programming_Expert", "Python_Coder"],
    "executor": ["Computer_terminal", "Code_Executor", "Executor", "Python_Runner"],
    "verifier": ["Verifier", "Checker", "Verification_Expert", "Reviewer", "Quality_Checker", "Auditor"],
    "reporter": ["Reporter", "Summarizer", "Answer_Writer", "Final_Responder"],
}
SWAP = {"ADD": "SUB", "SUB": "ADD", "MUL": "DIV", "DIV": "MUL", "PCT": "PCTOFF", "PCTOFF": "PCT", "SUM": "DIFF"}
LOCAL_FAULTS = ("arith", "operand", "op", "input")


# ---------------------------------------------------------------------------------------------------- formulas
@dataclasses.dataclass(frozen=True)
class Formula:
    cls: str                       # ADD SUB MUL DIV PCT PCTOFF SUM DIFF DOZEN
    const: int | None
    inputs: tuple


def evalf(f: Formula, vals: list) -> int | None:
    """Exact value of a formula on input values; None unless a positive integer <= MAXV (such runs are rejected)."""
    v = [Fraction(x) for x in vals]
    if f.cls == "DIFF":
        x = v[0] - v[1]
    elif f.cls == "DOZEN":
        x = Fraction(12 * f.const)
    else:
        x = ng.apply(f.cls, v, f.const)
    if x is None or x.denominator != 1 or not 1 <= x <= MAXV:
        return None
    return int(x)


def expr(f: Formula, terms: list, sym: dict) -> str:
    """The right-hand side of a formula over terms (variable names or numbers)."""
    c, m, d = f.const, sym["mul"], sym["div"]
    a = terms[0] if terms else None
    return {"ADD": lambda: f"{a} + {c}", "SUB": lambda: f"{a} - {c}", "MUL": lambda: f"{a} {m} {c}",
            "DIV": lambda: f"{a} {d} {c}", "PCT": lambda: f"{a} {m} {c} {d} 100",
            "PCTOFF": lambda: f"{a} {m} (100 - {c}) {d} 100", "SUM": lambda: f"{a} + {terms[1]}",
            "DIFF": lambda: f"{a} - {terms[1]}", "DOZEN": lambda: f"12 {m} {c}"}[f.cls]()


def code_expr(f: Formula, names: list) -> str:
    a, c = (names[0] if names else None), f.const
    return {"ADD": lambda: f"{a} + {c}", "SUB": lambda: f"{a} - {c}", "MUL": lambda: f"{a} * {c}",
            "DIV": lambda: f"{a} // {c}", "PCT": lambda: f"{a} * {c} // 100",
            "PCTOFF": lambda: f"{a} * (100 - {c}) // 100", "SUM": lambda: f"{a} + {names[1]}",
            "DIFF": lambda: f"{a} - {names[1]}", "DOZEN": lambda: f"12 * {c}"}[f.cls]()


def mutate(v: int, salt: str, family: str) -> int | None:
    """A deterministic wrong number near v: family "slip" (an arithmetic slip) or "misread" (a misread number)."""
    r = random.Random(f"{family}|{salt}|{v}")
    s = str(v)
    cands = {v + 1, v - 1, v + 10, v - 10}
    for i in range(len(s) - 1):
        if s[i] != s[i + 1] and not (i == 0 and s[1] == "0"):
            cands.add(int(s[:i] + s[i + 1] + s[i] + s[i + 2:]))
    if family == "slip":
        cands |= {v + 2, v - 2} | ({v + 100, v - 100} if v >= 200 else set())
        cands |= {v - v % 10 + d for d in range(10)}
    else:
        cands |= {v * 10} if v < 100 else set()
    cands = sorted(x for x in cands if 0 < x <= MAXV and x != v)
    return r.choice(cands) if cands else None


def misread_const(c: int, salt: str, cls: str) -> int | None:
    r = random.Random(f"const|{salt}|{c}")
    s = str(c)
    cands = {c + 1, c - 1}
    if len(s) == 2 and s[0] != s[1] and s[1] != "0":
        cands.add(int(s[::-1]))
    if cls in ("PCT", "PCTOFF"):
        cands = {c + 5, c - 5, c + 10, c - 10} | ({int(s[::-1])} if len(s) == 2 and s[1] != "0" else set())
        cands = {x for x in cands if 0 < x < 100}
    if cls in ("MUL", "DIV"):
        cands.discard(1)
    cands = sorted(x for x in cands if x > 0 and x != c)
    return r.choice(cands) if cands else None


# ---------------------------------------------------------------------------------------------------- trajectory
@dataclasses.dataclass
class Ev:
    eid: str                      # stable key of a scripted action: plan, given, op:q3, check:1, side:1, chat:1, report
    kind: str                     # plan given op check side chat report term
    role: str
    target: object = None         # op: qid; check: qid or "given"; side: (qa, qb, "+"/"-")


@dataclasses.dataclass
class Traj:
    p: object                     # numgen.Problem
    seed: str
    team: dict                    # role -> agent name (report_role / given_role resolved)
    route: dict                   # qid -> "calc" | "code"
    script: list
    style: dict
    noise: list                   # [(sentence, number)] listed by the given step as "not needed" (may be empty)
    vars: dict                    # qid -> variable name
    has_given: bool

    @property
    def st(self):
        return self.p.story

    def q(self, qid: str):
        return self.st.by(qid)

    def true_formula(self, qid: str) -> Formula:
        q = self.q(qid)
        return Formula(ng.OPCLS[q.op], q.const, tuple(q.inputs))

    @property
    def ops(self) -> list:
        return [q.id for q in self.st.qs if q.op]

    @property
    def gold(self) -> int:
        return self.st.qs[-1].value

    def task(self) -> str:
        return f"{self.p.body} {self.p.question}"


def _var_base(q, frame: dict) -> str:
    if q.kind == "count":
        pl = frame["item"][1]
        w = pl.split(" of ")[0].split()[0] if " of " in pl else pl.split()[-1]
    elif q.kind == "box":
        w = frame["container"][1]
    else:
        w = {"money": "money", "minutes": "minutes", "hours": "hours"}[q.kind]
    return re.sub(r"[^a-z]", "", w.lower()) or "x"


def var_names(p) -> dict:
    frame = ng.FRAME_BY_ID[p.frame]
    seen: dict = {}
    out = {}
    for q in p.story.qs:
        b = _var_base(q, frame)
        seen[b] = seen.get(b, 0) + 1
        out[q.id] = f"{b}_{seen[b]}"
    return out


def make_traj(p, seed: str, rng: random.Random) -> Traj:
    """Team, routes and the scripted actions of one run (before any fault)."""
    roles = {"planner": rng.choice(POOLS["planner"]), "verifier": rng.choice(POOLS["verifier"])}
    route_mode = rng.choices(["calc", "code", "both"], weights=[0.35, 0.25, 0.4])[0]
    if route_mode in ("calc", "both"):
        roles["calculator"] = rng.choice(POOLS["calculator"])
    if route_mode in ("code", "both"):
        roles["coder"] = rng.choice(POOLS["coder"])
        roles["executor"] = rng.choice(POOLS["executor"])
    if rng.random() < 0.4:
        roles["reader"] = rng.choice(POOLS["reader"])
    if rng.random() < 0.6:
        roles["reporter"] = rng.choice(POOLS["reporter"])
    team = dict(roles)
    team["report_role"] = "reporter" if "reporter" in roles else rng.choice(["verifier", "planner"])
    has_given = "reader" in roles or rng.random() < 0.45
    team["given_role"] = "reader" if "reader" in roles else "planner"
    ops = [q.id for q in p.story.qs if q.op]
    route = {}
    for qid in ops:
        route[qid] = route_mode if route_mode != "both" else rng.choice(["calc", "code"])
    noise = []
    if has_given and rng.random() < 0.5:
        for s in p.noise:
            m = re.search(r"\d+", s)
            if m:
                noise.append((s, int(m.group())))
    style = {"sym": rng.choice([{"mul": "*", "div": "/"}, {"mul": "×", "div": "/"}, {"mul": "×", "div": "÷"}]),
             "plan_fmt": rng.randrange(3), "verbose": rng.random() < 0.15, "assign": rng.random() < 0.5}
    script = []
    head = [Ev("plan", "plan", "planner")]
    if has_given:
        g = Ev("given", "given", team["given_role"])
        head = [g] + head if "reader" in roles and rng.random() < 0.35 else head + [g]
    script += head
    n_chk = rng.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
    chk_after = set(rng.sample(range(len(ops)), k=min(n_chk, len(ops))))
    n_side = 1 if rng.random() < 0.35 else 0
    n_chat = rng.choices([0, 1, 2], weights=[0.45, 0.4, 0.15])[0]
    body = []
    done: list = []
    k_chk = 0
    for i, qid in enumerate(ops):
        body.append(Ev(f"op:{qid}", "op", "calculator" if route[qid] == "calc" else "coder", qid))
        done.append(qid)
        if i in chk_after:
            k_chk += 1
            tgt = qid if rng.random() < 0.6 else rng.choice(done)
            if has_given and rng.random() < 0.12:
                tgt = "given"
            body.append(Ev(f"check:{k_chk}", "check", "verifier", tgt))
    script += body
    # side remarks and chatter at random places after the plan (and after two results for a side remark)
    for j in range(n_side):
        pos = [i for i in range(len(head), len(script) + 1)
               if sum(e.kind == "op" for e in script[:i]) >= 2]
        if pos:
            script.insert(rng.choice(pos), Ev(f"side:{j + 1}", "side", rng.choice(
                [r for r in ("calculator", "planner") if r in roles]), None))
    for j in range(n_chat):
        pos = list(range(len(head), len(script) + 1))
        who = rng.choice([r for r in ("planner", "verifier", "reporter", "reader") if r in roles])
        script.insert(rng.choice(pos), Ev(f"chat:{j + 1}", "chat", who))
    script.append(Ev("report", "report", team["report_role"]))
    if rng.random() < 0.4:
        script.append(Ev("term", "term", rng.choice(["planner", "verifier"])))
    tr = Traj(p=p, seed=seed, team=team, route=route, script=script, style=style, noise=noise, vars=var_names(p),
              has_given=has_given)
    for e in script:                                      # side remarks: two results of the same kind
        if e.kind == "side":
            before = [x.target for x in script[:script.index(e)] if x.kind == "op"]
            kinds: dict = {}
            for qid in before:
                kinds.setdefault(tr.q(qid).kind, []).append(qid)
            pairs = [v for v in kinds.values() if len(v) >= 2]
            if pairs:
                a, b = rng.sample(rng.choice(pairs), 2)
                e.target = (a, b, rng.choice(["+", "-"]))
    tr.script = [e for e in script if not (e.kind == "side" and e.target is None)]
    return tr


# ---------------------------------------------------------------------------------------------------- simulation
@dataclasses.dataclass
class Step:
    idx: int
    eid: str
    role: str
    agent: str
    text: str
    valid: bool
    true: bool
    uses: list
    fault: str | None = None
    declares: int | None = None


@dataclasses.dataclass
class Comp:
    """The latest computation of a quantity."""
    route: str
    idx: int                       # the step that established the value (calc step, or the Executor's output)
    code_idx: int | None
    formula: Formula               # as used (after faults)
    shown: list                    # operand values as written
    produced: int
    correct: int                   # the correct local result (plan formula on the values as last stated)
    correct_vals: list
    provs: list                    # provenance of the plan formula's operands (as last stated)
    used_provs: list               # provenance of the operands as used
    valid: bool


class Broken(Exception):
    """A run left the exact integer domain (such a configuration is rejected)."""


class Sim:
    def __init__(self, tr: Traj, faults: dict):
        self.tr, self.faults = tr, faults
        self.sym = tr.style["sym"]
        self.steps: list[Step] = []
        self.plan: dict = {}
        self.order: list = []
        self.plan_idx: int | None = None
        self.given_idx: int | None = None
        self.listed: dict = {}                # qid -> listed value (given step)
        self.believed: dict = {q.id: (q.value, None) for q in tr.st.qs if q.stated}
        self.comp: dict = {}
        self.ended = False
        self.declared: int | None = None
        self.active: set = set()
        self.n_re: dict = {}

    # ---- helpers
    def T(self, key: str) -> random.Random:
        return random.Random(f"{self.tr.seed}|tpl|{key}")

    def agent(self, role: str) -> str:
        return self.tr.team[role]

    def emit(self, eid, role, text, correct_text, uses, asserts, declares=None, fault=None) -> Step:
        truth = {q.id: q.value for q in self.tr.st.qs}
        ok = True
        for k, v in asserts:
            if k == "final":
                ok &= v == self.tr.gold
            elif k in ("plan", "formula", "side", "noise"):
                ok &= bool(v)
            else:
                ok &= truth[k] == v
        valid = text == correct_text
        if valid != (fault is None):
            raise AssertionError(f"{self.tr.seed} {eid}: fault {fault} but valid={valid}")
        if fault:
            self.active.add(eid.split("/")[0])
        st = Step(len(self.steps), eid, role, self.agent(role), text, valid, ok,
                  sorted({u for u in uses if u is not None}), fault, declares)
        self.steps.append(st)
        return st

    def fmt(self, qid: str, v: int) -> str:
        return self.tr.q(qid).fmt(v)

    def var(self, qid: str) -> str:
        return self.tr.vars[qid]

    def plan_no(self, qid: str) -> int:
        return self.order.index(qid) + 1

    # ---- run
    def run(self) -> "Sim":
        for ev in self.tr.script:
            if self.ended:
                break
            getattr(self, "do_" + ev.kind)(ev)
        if not self.ended:
            raise AssertionError(f"{self.tr.seed}: run did not end")
        return self

    @property
    def success(self) -> bool:
        return self.declared == self.tr.gold

    # ---- plan
    def plan_text(self, plan: dict, order: list) -> str:
        tr, r = self.tr, self.T("plan")
        v = tr.vars
        fmt = tr.style["plan_fmt"]
        lines = []
        head = r.choice(["Plan:", "Here is the plan for this task:", "Let's solve this step by step. Plan:",
                         "We can solve this in a few steps:"])
        if tr.style["verbose"]:
            head = (f"You are given the following task: {tr.task()}\nWe need to find {tr.q(tr.ops[-1]).np}. Every "
                    "quantity gets a variable name; each result is computed from earlier ones.\n\n" + head)
        lines.append(head)
        for i, qid in enumerate(order, 1):
            f = plan[qid]
            rhs = expr(f, [v[x] for x in f.inputs], self.sym)
            np_ = tr.q(qid).np
            lines.append([f"{i}. {v[qid]} = {rhs} ({np_})", f"{i}) {np_}: {v[qid]} = {rhs}",
                          f"({i}) {v[qid]} = {rhs}, i.e. {np_}"][fmt])
        stated = sorted({x for qid in order for x in plan[qid].inputs if tr.q(x).stated},
                        key=lambda x: [q.id for q in tr.st.qs].index(x))
        if stated:
            lines.append("Inputs from the task: " + "; ".join(f"{v[x]} = {tr.q(x).np}" for x in stated) + ".")
        lines.append(r.choice([f"The answer is {v[order[-1]]}.", f"Then report {v[order[-1]]} as the answer.",
                               f"{v[order[-1]]} answers the question."]))
        if tr.style["assign"]:
            who: dict = {}
            for i, qid in enumerate(order, 1):
                who.setdefault(self.agent("calculator" if tr.route[qid] == "calc" else "coder"), []).append(str(i))
            lines.append("Assignments: " + "; ".join(f"{a}: steps {', '.join(n)}" for a, n in who.items()) + ".")
        return "\n".join(lines)

    def do_plan(self, ev: Ev) -> None:
        tr = self.tr
        true_plan = {qid: tr.true_formula(qid) for qid in tr.ops}
        plan, order = dict(true_plan), list(tr.ops)
        f = self.faults.get(ev.eid)
        if f:
            qid = f["qid"]
            t = f["type"]
            if t == "plan_op":
                g = plan[qid]
                plan[qid] = Formula(SWAP[g.cls], g.const, g.inputs)
            elif t == "plan_const":
                g = plan[qid]
                c = misread_const(g.const, f["salt"], g.cls)
                if c is None:
                    raise Broken("const")
                plan[qid] = Formula(g.cls, c, g.inputs)
            elif t == "plan_skip":
                src = plan[qid].inputs[0]
                order.remove(qid)
                del plan[qid]
                for x in order:
                    g = plan[x]
                    plan[x] = Formula(g.cls, g.const, tuple(src if i == qid else i for i in g.inputs))
        self.plan, self.order = plan, order
        text = self.plan_text(plan, order)
        ctext = self.plan_text(true_plan, list(tr.ops))
        ok = plan == true_plan and order == tr.ops
        st = self.emit("plan", "planner", text, ctext, [], [("plan", ok)], fault=f["type"] if f else None)
        self.plan_idx = st.idx

    # ---- given values
    def given_text(self, vals: dict, noise: list) -> str:
        tr, r = self.tr, self.T("given")
        head = r.choice(["Values given in the task:", "From the task, the stated numbers are:",
                         "Here are the numbers stated in the task:", "Input values extracted from the task:"])
        lines = [head] + [f"- {tr.vars[q]} = {vals[q]} ({tr.q(q).np})" for q in vals]
        for s in noise:
            lines.append(r.choice(["- not needed: ", "- irrelevant: "]) + s)
        if tr.style["verbose"]:
            lines.append("All other numbers in the task are constants inside the operations.")
        return "\n".join(lines)

    def do_given(self, ev: Ev) -> None:
        tr = self.tr
        stated = [q.id for q in tr.st.qs if q.stated]
        vals = {q: tr.q(q).value for q in stated}
        noise = [s for s, _ in tr.noise]
        cvals, cnoise = dict(vals), list(noise)
        f = self.faults.get(ev.eid)
        if f:
            items = stated + [f"noise{k}" for k in range(len(tr.noise))]
            it = f["item"]
            if it not in items:
                raise Broken("given item")
            if it.startswith("noise"):
                k = int(it[5:])
                s, n = tr.noise[k]
                m = mutate(n, f["salt"], "misread")
                noise[k] = re.sub(r"\d+", str(m), s, count=1)
            else:
                vals[it] = mutate(vals[it], f["salt"], "misread")
                if vals[it] is None:
                    raise Broken("given")
        text, ctext = self.given_text(vals, noise), self.given_text(cvals, cnoise)
        asserts = [(q, vals[q]) for q in stated] + [("noise", a == b) for a, b in zip(noise, cnoise)]
        st = self.emit("given", ev.role, text, ctext, [], asserts, fault=f["type"] if f else None)
        self.given_idx = st.idx
        self.listed = dict(vals)
        for q in stated:
            self.believed[q] = (vals[q], st.idx)

    # ---- operations
    def do_op(self, ev: Ev, recompute: str | None = None, trigger: int | None = None) -> None:
        tr = self.tr
        qid = ev.target
        if qid not in self.order:                       # left out by the plan
            return
        f_plan = self.plan[qid]
        ins = list(f_plan.inputs)
        cvals = [self.believed[i][0] for i in ins]
        provs = [self.believed[i][1] for i in ins]
        correct = evalf(f_plan, cvals)
        if correct is None:
            raise Broken(f"{qid} correct")
        fault = None if recompute else self.faults.get(ev.eid)
        formula, shown, used_ins, used_provs = f_plan, list(cvals), list(ins), list(provs)
        premature = False
        if fault:
            t = fault["type"]
            if t == "op":
                if f_plan.cls not in SWAP:
                    raise Broken("op")
                formula = Formula(SWAP[f_plan.cls], f_plan.const, f_plan.inputs)
            elif t == "operand":
                if not ins:
                    raise Broken("operand")
                i = fault["salt_i"] % len(ins)
                shown[i] = mutate(cvals[i], fault["salt"], "misread")
                if shown[i] is None:
                    raise Broken("operand")
            elif t == "input":
                if not ins:
                    raise Broken("input")
                i = fault["salt_i"] % len(ins)
                kind = tr.q(ins[i]).kind
                cands = [x for x in self.believed if x not in ins and x != qid and tr.q(x).kind == kind
                         and self.believed[x][0] != cvals[i]]
                if not cands:
                    raise Broken("input cands")
                cands.sort(key=lambda x: [q.id for q in tr.st.qs].index(x))
                r = cands[fault["salt_i"] % len(cands)]
                used_ins[i] = r
                shown[i] = self.believed[r][0]
                used_provs[i] = self.believed[r][1]
                formula = Formula(f_plan.cls, f_plan.const, tuple(used_ins))
            elif t == "premature":
                if qid == self.order[-1]:
                    raise Broken("premature on final")
                premature = True
        produced = evalf(formula, shown)
        if produced is None:
            raise Broken(f"{qid} produced")
        if fault and fault["type"] == "arith":
            produced = mutate(correct, fault["salt"], "slip")
            if produced is None:
                raise Broken("arith")
        if fault and not premature and produced == correct:
            raise Broken("fault without effect")
        route = tr.route[qid]
        uses = [self.plan_idx] + used_provs + ([trigger] if trigger is not None else [])
        ftype = fault["type"] if fault else None
        key = ev.eid if not recompute else recompute
        if route == "calc":
            text = self.calc_text(key, qid, formula, used_ins, shown, produced, recompute)
            ctext = self.calc_text(key, qid, f_plan, ins, cvals, correct, recompute)
            if premature:
                text += self.premature_tail(key, qid, produced)
            asserts = [(i, v) for i, v in zip(used_ins, shown)] + [(qid, produced)]
            if premature:
                asserts.append(("final", produced))
            st = self.emit(key, "calculator", text, ctext, uses, asserts, declares=produced if premature else None,
                           fault=ftype)
            self.comp[qid] = Comp("calc", st.idx, None, formula, shown, produced, correct, cvals, provs,
                                  used_provs, st.valid)
            if premature:
                self.declared, self.ended = produced, True
        else:
            text = self.code_text(key, qid, formula, used_ins, shown, recompute)
            ctext = self.code_text(key, qid, f_plan, ins, cvals, recompute)
            true_f = tr.true_formula(qid)
            asserts = [(i, v) for i, v in zip(used_ins, shown)] + [("formula", formula == true_f)]
            cst = self.emit(key, "coder", text, ctext, uses, asserts, fault=ftype)
            out = self.exec_text(key + "/exec", produced)
            est = self.emit(key + "/exec", "executor", out, out, [cst.idx], [(qid, produced)])
            self.comp[qid] = Comp("code", est.idx, cst.idx, formula, shown, produced, correct, cvals, provs,
                                  used_provs, cst.valid)
        self.believed[qid] = (produced, self.comp[qid].idx)

    def calc_text(self, key, qid, f, ins, vals, out, recompute) -> str:
        tr, r = self.tr, self.T(key)
        v = tr.vars
        names = expr(f, [v[i] for i in ins], self.sym)
        nums = expr(f, vals, self.sym)
        if recompute:
            return r.choice([f"Recomputing {v[qid]} with the corrected value: {v[qid]} = {nums} = {out}.",
                             f"Updated {v[qid]} = {names} = {nums} = {out}.",
                             f"With the correction, {tr.q(qid).np} is {nums} = {tr.q(qid).fmt(out)}."])
        i = self.plan_no(qid)
        k = r.randrange(4)
        text = [f"{v[qid]} = {names} = {nums} = {out}.",
                f"Step {i} of the plan: {v[qid]} = {nums} = {out}.",
                f"Computing {tr.q(qid).np}: {nums} = {out}, so {v[qid]} = {out}.",
                f"{ng.cap(tr.q(qid).np)} is {nums} = {tr.q(qid).fmt(out)} ({v[qid]} = {out})."][k]
        if tr.style["verbose"]:
            text = f"Working on step {i} of the plan ({tr.q(qid).np}).\n" + text
        return text

    def premature_tail(self, key, qid, v) -> str:
        r = self.T(key + "|pre")
        return r.choice([f" This answers the task, so the final answer is {self.fmt(qid, v)}.\nTERMINATE",
                         f" That is what the question asks for. Final answer: {self.fmt(qid, v)}\nTERMINATE",
                         f" So the answer is {self.fmt(qid, v)}. We are done.\nTERMINATE"])

    def code_text(self, key, qid, f, ins, vals, recompute) -> str:
        tr, r = self.tr, self.T(key)
        v = tr.vars
        lines = []
        for i, x in zip(ins, vals):
            if f"{v[i]} = {x}" not in lines:
                lines.append(f"{v[i]} = {x}")
        lines.append(f"{v[qid]} = {code_expr(f, [v[i] for i in ins])}")
        lines.append(f"print({v[qid]})")
        code = "```python\n" + "\n".join(lines) + "\n```"
        if recompute:
            intro = r.choice([f"Updated code for {v[qid]} with the corrected value:",
                              f"Re-running {v[qid]} with the corrected input:"])
        else:
            i = self.plan_no(qid)
            intro = r.choice([f"I'll compute {tr.q(qid).np} with Python.", f"Code for step {i} of the plan:",
                              f"Here is the code for {v[qid]}:", f"Let me write code for step {i} ({v[qid]})."])
        if tr.style["verbose"]:
            intro += " The script prints the result so it can be checked."
        return intro + "\n" + code

    def exec_text(self, key, out) -> str:
        r = self.T(key)
        k = r.randrange(3)
        text = [f"exitcode: 0 (execution succeeded)\nCode output: {out}",
                f"exitcode: 0 (execution succeeded)\nCode output: \n{out}\n",
                f"Execution finished with exit code 0.\nOutput:\n{out}"][k]
        if self.tr.style["verbose"]:
            text += f"\n[runtime 0.0{r.randrange(1, 9)}s, python 3.11, sandbox ok]"
        return text

    # ---- verifier
    def dependents(self, qid: str) -> list:
        out, changed = [], {qid}
        for x in self.order:
            if x in self.comp and x != qid and any(i in changed for i in self.plan[x].inputs):
                changed.add(x)
                out.append(x)
        return out

    def do_check(self, ev: Ev) -> None:
        tr = self.tr
        tgt = ev.target
        f = self.faults.get(ev.eid)
        if tgt == "given":
            if self.given_idx is None:
                return
            if f:
                raise Broken("given check fault")
            wrong = [q for q, x in self.listed.items() if x != tr.q(q).value]
            text = self.check_given_text(ev.eid, wrong)
            asserts = [(q, tr.q(q).value) for q in self.listed]
            st = self.emit(ev.eid, "verifier", text, text, [self.given_idx], asserts)
            for q in wrong:
                self.listed[q] = tr.q(q).value
                self.believed[q] = (tr.q(q).value, st.idx)
            self.after_correction(ev.eid, st.idx, wrong)
            return
        rec = self.comp.get(tgt)
        if rec is None or tgt not in self.order:
            return
        invalid = rec.produced != rec.correct
        mode, val = ("catch", rec.correct) if invalid else ("confirm", rec.produced)
        cmode, cval = mode, val
        ftype = None
        premature = False
        if f:
            t = f["type"]
            if t == "accept" and invalid:
                mode, val, ftype = "accept", rec.produced, t
            elif t == "botch" and invalid:
                mode, val, ftype = "catch", mutate(rec.correct, f["salt"], "slip"), t
                if val is None or val == rec.produced:
                    raise Broken("botch")
            elif t == "reject" and not invalid:
                mode, val, ftype = "catch", mutate(rec.correct, f["salt"], "slip"), t
                if val is None:
                    raise Broken("reject")
            elif t == "premature" and tgt != self.order[-1]:
                premature, ftype = True, t
        deps = self.dependents(tgt) if mode == "catch" else []
        cdeps = self.dependents(tgt) if cmode == "catch" else []
        text = self.check_text(ev.eid, tgt, rec, mode, val, deps)
        ctext = self.check_text(ev.eid, tgt, rec, cmode, cval, cdeps)
        new_val = val if mode in ("catch",) else self.believed[tgt][0]
        if premature:
            text += self.premature_tail(ev.eid, tgt, new_val)
        uses = [rec.code_idx, rec.idx, self.plan_idx] + (rec.used_provs if mode == "accept" else rec.provs)
        if mode == "accept":
            asserts = [(i, x) for i, x in zip(rec.formula.inputs, rec.shown)] + [(tgt, rec.produced)]
        else:
            asserts = [(i, x) for i, x in zip(self.plan[tgt].inputs, rec.correct_vals)] + [(tgt, val)]
        if premature:
            asserts.append(("final", new_val))
        st = self.emit(ev.eid, "verifier", text, ctext, uses, asserts, declares=new_val if premature else None,
                       fault=ftype)
        if premature:
            self.declared, self.ended = new_val, True
            return
        if mode == "catch":
            self.believed[tgt] = (val, st.idx)
            self.comp[tgt] = dataclasses.replace(rec, idx=st.idx, code_idx=None, produced=val)
            self.after_correction(ev.eid, st.idx, [tgt])

    def after_correction(self, eid: str, idx: int, qids: list) -> None:
        todo = []
        for q in qids:
            for d in self.dependents(q) if q in self.comp else self.dependents_of_stated(q):
                if d not in todo:
                    todo.append(d)
        for d in [x for x in self.order if x in todo]:
            n = self.n_re.get(d, 0) + 1
            self.n_re[d] = n
            self.do_op(Ev(f"op:{d}", "op", "calculator" if self.tr.route[d] == "calc" else "coder", d),
                       recompute=f"re:{d}:{n}", trigger=idx)

    def dependents_of_stated(self, q: str) -> list:
        out, changed = [], {q}
        for x in self.order:
            if x in self.comp and any(i in changed for i in self.plan[x].inputs):
                changed.add(x)
                out.append(x)
        return out

    def check_text(self, key, tgt, rec: Comp, mode, val, deps) -> str:
        tr, r = self.tr, self.T(key)
        v = tr.vars
        who = self.steps[rec.code_idx if rec.code_idx is not None else rec.idx].agent
        if mode == "accept":                     # echo the checked step's own expression
            f, vals = rec.formula, rec.shown
        else:
            f, vals = self.plan[tgt], rec.correct_vals
        names = expr(f, [v[i] for i in f.inputs], self.sym)
        nums = expr(f, vals, self.sym)
        k = r.randrange(5)
        if mode in ("confirm", "accept"):
            text = [f"Checking {v[tgt]}: {nums} = {val}. Correct.",
                    f"I verified {who}'s result: {v[tgt]} = {names} = {nums} = {val}. Confirmed.",
                    f"Verification of {tr.q(tgt).np}: {nums} = {tr.q(tgt).fmt(val)}, which matches. No issues.",
                    f"No mistake in {v[tgt]}: {names} = {nums} = {val}, as {who} wrote.",
                    f"I re-did {v[tgt]} to be sure it is not off: {nums} = {val}. Correct."][k]
        else:
            text = [f"Checking {v[tgt]}: {nums} = {val}, not {rec.produced}. Correction: {v[tgt]} = {val}.",
                    f"There is a mistake in {v[tgt]}: {names} = {nums} = {val}, but {who} got {rec.produced}. "
                    f"The correct value is {val}.",
                    f"Verification of {tr.q(tgt).np} failed: {nums} = {val}, not {rec.produced}. "
                    f"Using {v[tgt]} = {val} from now on.",
                    f"{who} made a mistake in {v[tgt]}: {names} = {nums} = {val}, not {rec.produced}.",
                    f"I re-did {v[tgt]}: {nums} = {val}. {who}'s {rec.produced} is off; {v[tgt]} = {val}."][k]
            if deps:
                text += " " + r.choice([f"{', '.join(v[d] for d in deps)} used the wrong value and must be recomputed.",
                                        f"Please recompute {', '.join(v[d] for d in deps)}."])
        if tr.style["verbose"]:
            text = "I will re-derive this result from the plan and the earlier values.\n" + text
        return text

    def check_given_text(self, key, wrong: list) -> str:
        tr, r = self.tr, self.T(key)
        v = tr.vars
        if not wrong:
            return r.choice(["Checking the given values against the task: " +
                             ", ".join(f"{v[q]} = {x}" for q, x in self.listed.items()) + ". All correct.",
                             "I compared the listed values with the task text. They match."])
        return "Checking the given values against the task: " + " ".join(
            f"The task says {tr.q(q).np} is {tr.q(q).value}, not {self.listed[q]}." for q in wrong) + \
            " Correction: " + ", ".join(f"{v[q]} = {tr.q(q).value}" for q in wrong) + "."

    # ---- side remarks, chatter, report
    def do_side(self, ev: Ev) -> None:
        tr = self.tr
        a, b, op = ev.target
        if a not in self.comp or b not in self.comp:          # a result left out by the plan: no side remark
            return
        va, vb = self.believed[a][0], self.believed[b][0]
        if op == "-" and va < vb:
            a, b, va, vb = b, a, vb, va
        correct = va + vb if op == "+" else va - vb
        if correct <= 0:
            return
        f = self.faults.get(ev.eid)
        val = mutate(correct, f["salt"], "slip") if f else correct
        if val is None:
            raise Broken("side")
        text = self.side_text(ev.eid, a, b, op, va, vb, val)
        ctext = self.side_text(ev.eid, a, b, op, va, vb, correct)
        ta, tb = tr.q(a).value, tr.q(b).value
        tv = ta + tb if op == "+" else ta - tb
        self.emit(ev.eid, ev.role, text, ctext, [self.believed[a][1], self.believed[b][1]],
                  [(a, va), (b, vb), ("side", val == tv)], fault=f["type"] if f else None)

    def side_text(self, key, a, b, op, va, vb, val) -> str:
        tr, r = self.tr, self.T(key)
        v = tr.vars
        if op == "+":
            return r.choice([f"For reference, {v[a]} + {v[b]} = {va} + {vb} = {val}.",
                             f"Side note: {tr.q(a).np} and {tr.q(b).np} together come to {va} + {vb} = {val}.",
                             f"Just to keep track: {v[a]} ({va}) and {v[b]} ({vb}) add up to {val}."])
        return r.choice([f"For reference, {v[a]} - {v[b]} = {va} - {vb} = {val}.",
                         f"Side note: {v[a]} is {va} - {vb} = {val} more than {v[b]}.",
                         f"Just to keep track: the difference between {v[a]} ({va}) and {v[b]} ({vb}) is {val}."])

    def do_chat(self, ev: Ev) -> None:
        r = self.T(ev.eid)
        text = r.choice({"planner": ["Good, let's continue with the plan.", "Thanks. Please go ahead with the next step.",
                                     "Looks fine so far. Continue."],
                         "verifier": ["I will check the results as we go.", "Noted. Continue, I will verify later.",
                                      "Fine by me so far."],
                         "reporter": ["I will report the result once all steps are done.",
                                      "Waiting for the remaining steps before I write the answer."],
                         "reader": ["The values above are all the numbers we need from the task.",
                                    "Let me know if any other value from the task is needed."]}[ev.role])
        self.emit(ev.eid, ev.role, text, text, [], [])

    def do_report(self, ev: Ev) -> None:
        tr = self.tr
        fq = self.order[-1]
        v, prov = self.believed[fq]
        f = self.faults.get(ev.eid)
        val, src = v, prov
        if f:
            if f["type"] == "slip":
                val = mutate(v, f["salt"], "slip")
                if val is None:
                    raise Broken("report slip")
            elif f["type"] == "wrongq":
                cands = [x for x in self.order if x != fq and x in self.comp and self.believed[x][0] != v
                         and tr.q(x).kind == tr.q(fq).kind]
                if not cands:
                    raise Broken("wrongq")
                x = cands[-1 - f["salt_i"] % min(2, len(cands))]
                val, src = self.believed[x][0], self.believed[x][1]
        term = not any(e.kind == "term" for e in tr.script)
        text = self.report_text(ev.eid, fq, val, term)
        ctext = self.report_text(ev.eid, fq, v, term)
        self.emit(ev.eid, ev.role, text, ctext, [src, self.plan_idx], [("final", val)], declares=val,
                  fault=f["type"] if f else None)
        self.declared = val
        if term:
            self.ended = True

    def report_text(self, key, fq, val, term) -> str:
        tr, r = self.tr, self.T(key)
        s = tr.q(fq).fmt(val)
        text = r.choice([f"The final answer is {s}.", f"All steps of the plan are done. {ng.cap(tr.q(fq).np)}: {s}.",
                         f"Answer: {s}", f"Based on the results above, {tr.vars[fq]} = {val}, so the answer is {s}."])
        if tr.style["verbose"]:
            text = "Summary of the work: the plan was followed step by step and the results were checked.\n" + text
        return text + ("\nTERMINATE" if term else "")

    def do_term(self, ev: Ev) -> None:
        self.emit(ev.eid, ev.role, "TERMINATE", "TERMINATE", [], [])
        self.ended = True


def simulate(tr: Traj, faults: dict) -> Sim:
    return Sim(tr, faults).run()


# ---------------------------------------------------------------------------------------------------- labels
def labels(tr: Traj, faults: dict, run: Sim | None = None) -> dict:
    """Exact labels of a faulty run: per-step V, C, uses, I; R by re-simulation for every invalid step; the decisive
    step (earliest invalid step whose correction alone succeeds). Raises Broken if any run leaves the integer domain,
    AssertionError on an internal inconsistency."""
    run = run or simulate(tr, faults)
    steps = run.steps
    if set(faults) != run.active:
        raise Broken(f"inactive faults {set(faults) - run.active}")
    out = {"success": run.success, "declared": run.declared, "R": {}, "cf_len": {}}
    for s in steps:
        if not s.valid:
            key = s.eid.split("/")[0]
            cf = simulate(tr, {k: v for k, v in faults.items() if k != key})
            if [x.text for x in cf.steps[:s.idx]] != [x.text for x in steps[:s.idx]]:
                raise AssertionError(f"{tr.seed}: counterfactual prefix differs at {s.idx}")
            if cf.steps[s.idx].text != _correct_text(tr, faults, key, s.idx):
                raise AssertionError(f"{tr.seed}: corrected step {s.idx} is not the correct local step")
            out["R"][s.idx] = cf.success
            out["cf_len"][s.idx] = len(cf.steps)
    C = {s.idx: s.true for s in steps}
    I = {}
    for s in steps:
        bad_in = any(not C[u] for u in s.uses)
        I[s.idx] = (not s.true) and s.valid and bad_in
        if s.valid and not s.true and not bad_in:
            raise AssertionError(f"{tr.seed}: step {s.idx} valid and untrue without an untrue input")
        if not s.valid and s.true:
            raise Broken("invalid but true (coincidence)")
    cands = [i for i in sorted(out["R"]) if out["R"][i]]
    out["I"] = I
    out["decisive"] = cands[0] if cands else None
    out["candidates"] = cands
    return out


def _correct_text(tr: Traj, faults: dict, key: str, idx: int) -> str:
    """The text the simulator renders as step idx's correct local version in the actual run (Sim.emit compares it)."""
    sim = Sim(tr, faults)
    texts: dict = {}
    orig = sim.emit

    def spy(eid, role, text, correct_text, *a, **kw):
        texts[len(sim.steps)] = correct_text
        return orig(eid, role, text, correct_text, *a, **kw)
    sim.emit = spy
    sim.run()
    return texts[idx]


# ---------------------------------------------------------------------------------------------------- rendering
def step_ref(s: Step) -> str:
    return f"Step {s.idx} [{s.agent}]"


def render_log(tr: Traj, steps: list, with_gt: bool) -> str:
    lines = [HEADER, "", f"Task: {tr.task()}"]
    if with_gt:
        lines.append(f"Correct final answer (for reference): {tr.q(tr.ops[-1]).fmt(tr.gold)}")
    lines.append("")
    for s in steps:
        lines.append(f"Step {s.idx} [{s.agent}]: {s.text}")
    return "\n".join(lines)


def agents_in_order(steps: list) -> list:
    out = []
    for s in steps:
        if s.agent not in out:
            out.append(s.agent)
    return out


# ---------------------------------------------------------------------------------------------------- nodes
def node_texts(s: Step, kind: str, other: Step | None = None) -> tuple[str, str | None]:
    a = step_ref(s)
    if kind == "validity":
        return (f"Is {a} correct given the messages before it (taking them as they are)?",
                f"Does {a} make a mistake of its own, given the messages before it (taking them as they are)?")
    if kind == "correctness":
        return f"Is everything {a} states actually true?", f"Does {a} state something that is actually false?"
    if kind == "inheritance":
        return (f"Is {a} wrong only because it relied on an earlier wrong step?",
                f"Is {a} either right, or wrong through a mistake of its own?")
    if kind == "repair":
        return (f"If only {a} were corrected, would the task succeed?",
                f"If only {a} were corrected, would the task still fail?")
    if kind == "dependency":
        return f"Does {a} use the output of {step_ref(other)}?", None
    raise ValueError(kind)


def build_nodes(steps: list, lab: dict, rng: random.Random) -> list[dict]:
    """V, C, I on every step, R on every invalid step, D on the pairs whose source is untrue plus a sample of other
    used pairs and as many unused pairs (yes/no balance); V, C, I, R asked in plain or opposite wording by a fair coin
    (truth follows the wording). depends_on: I_t -> V_t, C_t and, for every used step s whose D pair is asked, D_{t,s}
    and C_s (the composition I = not C and V and some used s with D and not C_s); every other node is a leaf."""
    C = {s.idx: s.true for s in steps}
    nodes = []

    def add(nid, kind, s, truth, deps, other=None, flip=True):
        q, opp = node_texts(s, kind, other)
        pol = "plain"
        if flip and opp and rng.random() < 0.5:
            q, opp, truth, pol = opp, q, not truth, "opposite"
        nodes.append({"id": nid, "question": q, "opposite": opp, "truth": bool(truth),
                      "p_cot": 1.0 if truth else 0.0, "p_fresh": None, "type": kind, "depends_on": deps,
                      "status": "direct", "initial": None, "step": s.idx, "polarity": pol,
                      **({"source_step": other.idx} if other is not None else {})})
    # dependency pairs
    pos_bad = [(s.idx, u) for s in steps for u in s.uses if not C[u]]
    pos_other = [(s.idx, u) for s in steps for u in s.uses if C[u]]
    n_extra = min(len(pos_other), max(2, len(steps) // 3))
    pos = sorted(set(pos_bad) | set(rng.sample(pos_other, n_extra)))
    neg_pool = [(s.idx, u) for s in steps for u in range(s.idx) if u not in s.uses]
    near = [x for x in neg_pool if x[0] - x[1] <= 2]
    neg: set = set()
    for _ in range(len(pos)):
        pool = [x for x in (near if near and rng.random() < 0.5 else neg_pool) if x not in neg]
        if not pool:
            pool = [x for x in neg_pool if x not in neg]
        if not pool:
            break
        neg.add(rng.choice(pool))
    asked_d = set(pos) | neg
    by = {s.idx: s for s in steps}
    for s in steps:
        add(f"v{s.idx}", "validity", s, s.valid, [])
        add(f"c{s.idx}", "correctness", s, s.true, [])
    for t, u in sorted(asked_d):
        add(f"d{t}_{u}", "dependency", by[t], u in by[t].uses, [], other=by[u], flip=False)
    for s in steps:
        deps = [f"v{s.idx}", f"c{s.idx}"]
        for u in s.uses:
            if (s.idx, u) in asked_d:
                deps += [f"d{s.idx}_{u}", f"c{u}"]
        add(f"i{s.idx}", "inheritance", s, lab["I"][s.idx], deps)
    for s in steps:
        if not s.valid:
            add(f"r{s.idx}", "repair", s, lab["R"][s.idx], [])
    depth = node_depths(nodes)
    for n in nodes:
        n["node_depth"] = depth[n["id"]]
    return nodes


LEAK_WORDS = re.compile(r"\b(decisive|inject\w*|fault\w*|slip\w*|perturb\w*|corrupt\w*)\b", re.I)


def leak_check(item: dict) -> list[str]:
    """Leak rule 1 (§3.4): no node question states or asks the decisive step, a fault type or injection metadata;
    the prompt holds no metadata and no answer line after the question."""
    bad = []
    for n in item.get("subqs", []):
        for t in (n["question"], n.get("opposite") or ""):
            if LEAK_WORDS.search(t):
                bad.append(f"{n['id']}: {t}")
    head, _, tail = item["prompt"].rpartition("\n\nQuestion: ")
    if LEAK_WORDS.search(head):
        bad.append("log mentions metadata")
    if "Answer:" in tail or "Question:" in head:
        bad.append("prompt layout")
    return bad


# ---------------------------------------------------------------------------------------------------- configuration
PATTERNS = {"rec_pre": 3.5, "dead_pre": 2.5, "react": 1.5, "rec_post": 2.5, "dead_post": 1.0, "combo": 1.5}
# premature: a premature termination always ends the run (the decisive step is the last one); reject: a Verifier that
# "corrects" a right result writes a correction message, and so does a correct Verifier that recovers a mistake
# (rec_pre / rec_post), so the share of correction messages that are decisive stays low (surface-probe check)
TYPE_WEIGHT = {"premature": 0.3, "reject": 0.25}
CHECK_DOWNWEIGHT = 0.6                 # share of draws where Verifier checks cannot be the decisive candidate


def _fault(tr: Traj, eid: str, typ: str, rng: random.Random, **kw) -> dict:
    return {"type": typ, "salt": f"{tr.seed}|{eid}|{rng.randrange(10 ** 9)}", "salt_i": rng.randrange(10 ** 6), **kw}


def decisive_options(tr: Traj, steps: list) -> list:
    """(step index, event id, fault types) that can carry a fatal fault, from a clean run."""
    out = []
    final = tr.ops[-1]
    for s in steps:
        if s.eid == "plan":
            ts = ["plan_op", "plan_skip"] + (["plan_const"] if any(
                tr.true_formula(q).const is not None and tr.true_formula(q).cls != "SUM" for q in tr.ops) else [])
            out.append((s.idx, "plan", ts))
        elif s.eid == "given":
            if any(q.stated for q in tr.st.qs):
                out.append((s.idx, "given", ["given"]))
        elif s.eid.startswith("op:") and "/" not in s.eid:
            q = s.eid[3:]
            f = tr.true_formula(q)
            ts = (["arith"] if s.role == "calculator" else []) + (["operand", "input"] if f.inputs else []) + \
                 (["op"] if f.cls in SWAP else []) + (["premature"] if s.role == "calculator" and q != final else [])
            out.append((s.idx, s.eid, ts))
        elif s.eid.startswith("check:"):
            ev = next(e for e in tr.script if e.eid == s.eid)
            if ev.target != "given":
                out.append((s.idx, s.eid, ["reject"] + (["premature"] if ev.target != final else [])))
        elif s.eid == "report":
            out.append((s.idx, "report", ["slip", "wrongq"]))
    return [o for o in out if o[2]]


def _decisive_fault(tr: Traj, eid: str, typ: str, rng: random.Random) -> dict:
    if typ.startswith("plan_"):
        qs = [q for q in tr.ops if (typ != "plan_const" or (tr.true_formula(q).const is not None
                                                            and tr.true_formula(q).cls != "SUM"))
              and (typ == "plan_const" or tr.true_formula(q).cls in SWAP)]
        return _fault(tr, eid, typ, rng, qid=rng.choice(qs))
    if typ == "given":
        return _fault(tr, eid, typ, rng, item=rng.choice([q.id for q in tr.st.qs if q.stated]))
    return _fault(tr, eid, typ, rng)


def _ev_index(script: list, eid: str) -> int:
    return next(i for i, e in enumerate(script) if e.eid == eid)


def _qty_of(tr: Traj, eid: str, faults: dict) -> str | None:
    """The quantity whose value a fault at eid corrupts (checks on it after the fault would recover it)."""
    if eid.startswith("op:"):
        return eid[3:]
    if eid == "given":
        return "given"
    if eid.startswith("check:"):
        return next(e for e in tr.script if e.eid == eid).target
    return None


def _protect(tr: Traj, script: list, eid: str, keep: set, rng: random.Random) -> list:
    """Checks after the fault at eid that target its quantity (a correct verifier would recover it) are pointed at
    another result computed before them, or dropped. Checks in keep are left alone."""
    q = _qty_of(tr, eid, {})
    if q is None:
        return script
    i0 = _ev_index(script, eid)
    out = []
    for i, e in enumerate(script):
        if i > i0 and e.kind == "check" and e.eid not in keep and e.target == q:
            done = [x.target for x in script[:i] if x.kind == "op" and x.target != q]
            if not done:
                continue
            e = dataclasses.replace(e, target=rng.choice(done))
        out.append(e)
    return out


def _insert(script: list, ev: Ev, lo: int, hi: int, rng: random.Random) -> list:
    """Insert ev at a script position in [lo, hi] (indices of the new list), never after the report."""
    rep = _ev_index(script, "report")
    hi = min(hi, rep)
    if lo > hi:
        return None
    pos = rng.randint(lo, hi)
    return script[:pos] + [ev] + script[pos:]


def _side_slots(tr: Traj, script: list, lo: int, hi: int) -> list:
    """Positions in [lo, hi] where a side remark on two results of the same kind is possible, with the pairs."""
    out = []
    rep = _ev_index(script, "report")
    for pos in range(lo, min(hi, rep) + 1):
        kinds: dict = {}
        for x in script[:pos]:
            if x.kind == "op":
                kinds.setdefault(tr.q(x.target).kind, []).append(x.target)
        pairs = [v for v in kinds.values() if len(v) >= 2]
        if pairs:
            out.append((pos, pairs))
    return out


def configure(tr: Traj, clean: Sim, rng: random.Random, p_multi: float,
              u: float | None = None) -> tuple[list, dict, dict] | None:
    """Faults for one trajectory: a decisive candidate k drawn uniformly over the clean run's steps that can carry a
    fatal fault (then a fault type uniformly among those that fit the step), and with probability p_multi a second
    (or third) mistake from one of PATTERNS:
      rec_pre   a mistake j < k that a correct Verifier check catches (inserted after j, before k): never fatal
      dead_pre  a slip in a side remark or in a "not needed" listed number before k: never fatal
      react     the Verifier confirms k's wrong result (accept) or corrects it to another wrong value (botch)
      rec_post  a mistake j > k caught by a correct check;  dead_post  a side-remark slip after k
      combo     one *_pre and one of react / rec_post / dead_post
    Returns (script, faults, meta) or None when the draw does not fit this run."""
    opts = decisive_options(tr, clean.steps)
    if not opts:
        return None
    u = rng.random() if u is None else u                     # a uniform relative position, then the nearest step
    n = max(1, len(clean.steps) - 1)
    if rng.random() < CHECK_DOWNWEIGHT:                       # fewer decisive Verifier steps (see TYPE_WEIGHT)
        opts = [o for o in opts if not o[1].startswith("check:")] or opts
    best = min(abs(o[0] / n - u) for o in opts)
    k_idx, k_eid, types = rng.choice([o for o in opts if abs(o[0] / n - u) == best])
    k_type = rng.choices(types, weights=[TYPE_WEIGHT.get(t, 1.0) for t in types])[0]
    faults = {k_eid: _decisive_fault(tr, k_eid, k_type, rng)}
    script = [dataclasses.replace(e) for e in tr.script]
    keep: set = set()
    meta = {"intended_eid": k_eid, "intended_type": k_type, "pattern": "single"}
    if rng.random() < p_multi:                  # the first pattern (weighted order) that fits this k; else single
        pats, w = list(PATTERNS), list(PATTERNS.values())
        order = []
        while pats:
            i = rng.choices(range(len(pats)), weights=w)[0]
            order.append(pats.pop(i))
            w.pop(i)
        for pat in order:
            parts = {"combo": [rng.choice(["rec_pre", "dead_pre"]),
                               rng.choice(["react", "rec_post", "dead_post"])]}.get(pat, [pat])
            f2, k2, sc = dict(faults), set(keep), script
            for part in parts:
                sc = _add_pattern(tr, sc, f2, k2, k_eid, k_type, part, rng)
                if sc is None:
                    break
            if sc is not None:
                script, faults, keep = sc, f2, k2
                meta["pattern"] = pat
                break
    script = _protect(tr, script, k_eid, keep, rng)
    return script, faults, meta


def _local_type(tr: Traj, ev: Ev, rng: random.Random) -> str | None:
    f = tr.true_formula(ev.target)
    ts = (["arith"] if ev.role == "calculator" else []) + (["operand", "input"] if f.inputs else []) + \
         (["op"] if f.cls in SWAP else [])
    return rng.choice(ts) if ts else None


def _add_pattern(tr, script, faults, keep, k_eid, k_type, part, rng) -> list | None:
    ki = _ev_index(script, k_eid)
    n_chk = sum(e.kind == "check" for e in script)
    if part in ("rec_pre", "rec_post"):
        if part == "rec_pre":
            cands = [e for e in script[:ki] if (e.kind == "op" or (e.kind == "given" and tr.noise is not None))
                     and e.eid not in faults]
        else:
            if k_type in ("premature", "slip", "wrongq") or k_eid == "report":
                return None
            cands = [e for e in script[ki + 1:] if e.kind == "op" and e.eid not in faults]
        cands = [e for e in cands if e.kind != "given" or any(q.stated for q in tr.st.qs)]
        if not cands:
            return None
        j = rng.choice(cands)
        if j.kind == "given":
            faults[j.eid] = _fault(tr, j.eid, "given", rng, item=rng.choice([q.id for q in tr.st.qs if q.stated]))
            tgt = "given"
        else:
            t = _local_type(tr, j, rng)
            if t is None:
                return None
            faults[j.eid] = _fault(tr, j.eid, t, rng)
            tgt = j.target
        ji = _ev_index(script, j.eid)
        hi = ki if part == "rec_pre" else len(script)
        ev = Ev(f"check:{n_chk + 1}", "check", "verifier", tgt)
        keep.add(ev.eid)
        return _insert(script, ev, ji + 1, max(ji + 1, hi), rng)
    if part == "react":
        if k_type not in LOCAL_FAULTS:
            return None
        ev = Ev(f"check:{n_chk + 1}", "check", "verifier", k_eid[3:])
        faults[ev.eid] = _fault(tr, ev.eid, rng.choices(["accept", "botch"], weights=[0.8, 0.2])[0], rng)
        keep.add(ev.eid)
        return _insert(script, ev, ki + 1, len(script), rng)
    if part in ("dead_pre", "dead_post"):
        lo, hi = (0, ki) if part == "dead_pre" else (ki + 1, len(script))
        if part == "dead_post" and (k_type in ("premature",) or k_eid == "report"):
            return None
        gi = next((i for i, e in enumerate(script) if e.kind == "given"), None)
        use_noise = (part == "dead_pre" and tr.noise and gi is not None and gi < ki and "given" not in faults
                     and rng.random() < 0.4)
        if use_noise:
            faults["given"] = _fault(tr, "given", "given", rng, item=f"noise{rng.randrange(len(tr.noise))}")
            return script
        existing = [e for i, e in enumerate(script) if e.kind == "side" and lo <= i < hi and e.eid not in faults]
        if existing:
            e = rng.choice(existing)
            faults[e.eid] = _fault(tr, e.eid, "side", rng)
            return script
        slots = _side_slots(tr, script, max(lo, 1), hi)
        if not slots:
            return None
        pos, pairs = rng.choice(slots)
        a, b = rng.sample(rng.choice(pairs), 2)
        n_side = sum(e.kind == "side" for e in script)
        roles = [r for r in ("calculator", "planner") if r in tr.team]
        ev = Ev(f"side:{n_side + 1}", "side", rng.choice(roles), (a, b, rng.choice(["+", "-"])))
        faults[ev.eid] = _fault(tr, ev.eid, "side", rng)
        return script[:pos] + [ev] + script[pos:]
    raise ValueError(part)


def build_trajectory(p, seed: str, rng: random.Random, *, p_multi: float = 0.5, min_steps: int = 6,
                     max_steps: int = 16, tries: int = 30) -> dict | None:
    """One failing trajectory with exact labels for problem p, or None after `tries` draws. Returns the Traj (with
    the configured script), the faulty run, the faults, the labels and the configuration meta; counts in
    result["rejects"] why earlier draws were dropped."""
    rejects: dict = {}
    u = rng.random()                   # the decisive step's target relative position, kept over the retries

    def rej(why):
        rejects[why] = rejects.get(why, 0) + 1
    for a in range(tries):
        tr = make_traj(p, f"{seed}/{a}", rng)
        try:
            clean = simulate(tr, {})
        except Broken:
            rej("clean_broken")
            continue
        if not clean.success:
            raise AssertionError(f"{tr.seed}: clean run fails")
        if not min_steps <= len(clean.steps) <= max_steps:
            rej("clean_len")
            continue
        cfg = configure(tr, clean, rng, p_multi, u)
        if cfg is None:
            rej("config")
            continue
        script, faults, meta = cfg
        tr = dataclasses.replace(tr, script=script)
        try:
            clean2 = simulate(tr, {})
            if not clean2.success:
                raise AssertionError(f"{tr.seed}: clean run with the configured script fails")
            run = simulate(tr, faults)
            if run.success:
                rej("not_failing")
                continue
            if not min_steps <= len(run.steps) <= max_steps or len(run.steps) > len(LETTERS):
                rej("len")
                continue
            lab = labels(tr, faults, run)
        except Broken as e:
            rej("broken_" + str(e).split()[0])
            continue
        if lab["decisive"] is None:
            rej("no_single_fix")
            continue
        return {"traj": tr, "run": run, "faults": faults, "labels": lab, "meta": meta, "rejects": rejects,
                "attempts": a + 1}
    return None


# ---------------------------------------------------------------------------------------------------- items
def options_block(names: list) -> str:
    return "\n".join(f"{L}) {n}" for L, n in zip(LETTERS, names))


def make_items(res: dict, item_id: str, with_gt: bool, rng: random.Random) -> tuple[dict, dict]:
    """The step-level item (ww_alg_step layout; subqs = the V/C/D/I/R nodes) and the agent-level item (ww_alg_agent
    layout, no nodes) of one trajectory. Extra fields (fa, trajectory, program) carry the exact labels for later
    renderers (e.g. the marks format of pilot_design §4.2); the trainer reads none of them."""
    tr, run, lab, meta = res["traj"], res["run"], res["labels"], res["meta"]
    steps = run.steps
    n = len(steps)
    if n > len(LETTERS):
        raise ValueError(f"{item_id}: {n} steps do not fit letter options")
    labs = LETTERS[:n]
    names = [f"Step {s.idx} [{s.agent}]" for s in steps]
    dec = lab["decisive"]
    nodes = build_nodes(steps, lab, rng)
    invalid = [s.idx for s in steps if not s.valid]
    traj = [{"step": s.idx, "agent": s.agent, "role": s.role, "event": s.eid, "valid": s.valid, "true": s.true,
             "inherited": lab["I"][s.idx], "uses": s.uses, "repair": lab["R"].get(s.idx), "fault": s.fault}
            for s in steps]
    agents = agents_in_order(steps)
    fa = {"decisive_step": dec, "decisive_agent": steps[dec].agent, "decisive_role": steps[dec].role,
          "decisive_fault": steps[dec].fault, "n_steps": n, "n_agents": len(agents), "invalid_steps": invalid,
          "inherited_steps": [i for i in range(n) if lab["I"][i]], "candidates": lab["candidates"],
          "multi_error": len(invalid) >= 2, "first_invalid_is_decisive": invalid[0] == dec,
          "pattern": meta["pattern"], "intended_event": meta["intended_eid"],
          "gold_answer": tr.gold, "declared_answer": run.declared, "traj_seed": tr.seed,
          "roles": {tr.team[r]: r for r in POOLS if r in tr.team}, "verbose": tr.style["verbose"],
          "problem_sig": ng.signature(tr.p)}
    log = render_log(tr, steps, with_gt)
    teacher = {L: float(i == dec) for i, L in enumerate(labs)}
    step_item = {"item_id": item_id, "source": SOURCE, "group": "fa0/step_gt" if with_gt else "fa0/step",
                 "type": "choice", "prompt": log + "\n\n" + Q_STEP + "\nOptions:\n" + options_block(names),
                 "labels": labs, "label_order": labs, "label_names": dict(zip(labs, names)),
                 "gold_label": labs[dec], "gold_probs": None, "teacher": teacher, "subqs": nodes, "has_tree": True,
                 "tree_source": TREE_SOURCE, "tree_fix": True, "with_gt": with_gt, "depth": tr.p.depth,
                 "frame": tr.p.frame, "fa": fa, "trajectory": traj, "program": ng.program(tr.p)}
    alabs = LETTERS[:len(agents)]
    ga = agents.index(steps[dec].agent)
    agent_item = {"item_id": item_id + "-agent", "source": SOURCE, "group": "fa0/agent", "type": "choice",
                  "prompt": render_log(tr, steps, False) + "\n\n" + Q_AGENT + "\nOptions:\n" + options_block(agents),
                  "labels": alabs, "label_order": alabs, "label_names": dict(zip(alabs, agents)),
                  "gold_label": alabs[ga], "gold_probs": None, "teacher": {L: float(i == ga) for i, L in
                                                                           enumerate(alabs)},
                  "with_gt": False, "depth": tr.p.depth, "frame": tr.p.frame,
                  "fa": {"decisive_step": dec, "decisive_agent": steps[dec].agent, "decisive_role": steps[dec].role,
                         "n_steps": n, "n_agents": len(agents), "traj_item": item_id}}
    return step_item, agent_item


def audit_item(item: dict) -> list[str]:
    """Format and label checks of one step-level item (empty list: passes)."""
    bad = leak_check(item)
    labs, p = item["labels"], item["prompt"]
    head, sep, tail = p.rpartition("\n\nQuestion: ")
    opts = tail.split("\nOptions:\n")[1].split("\n") if "\nOptions:\n" in tail else []
    if not sep or len(opts) != len(labs) or not all(re.match(rf"^{L}\) Step {i} \[[A-Za-z_]+\]$", o)
                                                    for i, (L, o) in enumerate(zip(labs, opts))):
        bad.append("options")
    if item["gold_label"] not in labs or max(item["teacher"], key=item["teacher"].get) != item["gold_label"]:
        bad.append("gold")
    tr = item["trajectory"]
    dec = item["fa"]["decisive_step"]
    cand = [t["step"] for t in tr if not t["valid"] and t["repair"]]
    if not cand or cand[0] != dec or labs.index(item["gold_label"]) != dec:
        bad.append("decisive is not the earliest invalid step whose repair succeeds")
    ids = {n["id"] for n in item["subqs"]}
    by = {n["id"]: n for n in item["subqs"]}
    if any(d not in ids for n in item["subqs"] for d in n["depends_on"]):
        bad.append("depends_on outside the item")
    if node_depths(item["subqs"]) != {n["id"]: n["node_depth"] for n in item["subqs"]}:
        bad.append("node_depth")
    C = {t["step"]: t["true"] for t in tr}
    for n in item["subqs"]:                         # composition: I = not C and V and some asked D with untrue source
        if n["type"] != "inheritance":
            continue
        t = n["step"]
        val = lambda i: by[i]["truth"] if by[i]["polarity"] == "plain" else not by[i]["truth"]
        comp = (not val(f"c{t}")) and val(f"v{t}") and any(
            val(d) and not val(f"c{d.split('_')[1]}") for d in n["depends_on"] if d.startswith("d"))
        if comp != val(n["id"]) or comp != tr[t]["inherited"]:
            bad.append(f"{n['id']}: composition")
    for t in tr:
        if t["valid"] and not t["true"] and not any(not C[u] for u in t["uses"]):
            bad.append(f"step {t['step']}: untrue without an untrue input")
    return bad
