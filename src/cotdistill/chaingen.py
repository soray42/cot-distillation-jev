"""Decision-chain cases (testbed TB-C): derived checks whose results are never stated, a final that reads only the top
check of one chain, and a same-depth mirror chain that nothing reads.

A case is built on rulegen's domains, facts and predicates. Two facts of the domain are reserved (no check reads them;
the control questions ask about them); the predicates of the other facts feed two chains of named checks:

  path    check 1 is an AND/OR of two predicate literals; check k >= 2 reads check k-1 (passes/fails) and one predicate
          literal, and from level 3 on, with probability `fanin`, also one older check of its own chain
  mirror  built the same way and independently; it reads only predicates and its own checks, and nothing reads it
  final   "if the <path top> check passes, <outcome 1>; otherwise, if <predicate>, <outcome 2>; otherwise <outcome 3>",
          four lettered options (the domain's outcome set; the fourth is never right)

Pivotality knob q: each literal of a check k >= 2 other than its link to check k-1 is made non-controlling with
probability q (AND: it holds; OR: it fails), so check k-1 decides check k; otherwise its polarity is drawn at random
and may short-circuit the gate. Training sets are not filtered (rejecting short certificates would make every side
literal irrelevant in every item, a learnable shortcut). The depth an item needs is its certificate depth
(`cert_depth`: the number of check levels in the shallowest proof of the top check's value); evaluation sets keep only
cases with cert_depth == depth (generate(full_depth=True)), so depth k there means k levels are needed. On such items
every side literal along the proof is non-controlling, so they cannot tell whether a model reads facts above level 1
(a "level 1 plus link polarities" solver is exact on them; it still has to walk the whole chain). Each path check also
records whether overriding it flips the final (`pivotal`) and the item `piv_run`, the run of pivotal checks from the
top; with fan-in this is not the depth needed (two downstream inputs can cancel). Check names are nonce words drawn per
item and the check lines are shuffled, so neither names nor positions carry depth or chain membership. Renderers:
`chain_shuf` (check list, "Request details" line; training) and `chain_prose` (held-out: checks, decision and facts
as sentences).

Tree nodes (subqs) are the checks of both chains: exact truth, `depends_on` = the check parents (predicates are
facts of the prompt, not nodes), `node_depth` = the level in its chain (1-based; node_depths() of the same nodes is
node_depth - 1), `on_path`, and per parent `cf` (does negating that one parent, the other inputs as they are, flip
the node) and `cf_value` (the node's value then). Nothing here calls a teacher.
"""
from __future__ import annotations

import dataclasses
import random
from typing import Any

from . import rulegen as rg

OPS = ("and", "or")
_ON = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "gr", "kr", "pr", "tr", "st",
       "sk", "th"]
_VOW = ["a", "e", "i", "o", "u", "ai", "ou"]
_CODA = ["", "n", "r", "l", "s", "k", "m", "th", "rn", "sk"]


@dataclasses.dataclass(frozen=True)
class Lit:
    kind: str                       # "p" (predicate pid) | "c" (check id)
    ref: str
    want: bool


@dataclasses.dataclass(frozen=True)
class Check:
    cid: str                        # "P1".."Pd" (path), "M1".."Md" (mirror); never shown in the prompt
    name: str                       # nonce name shown in the prompt
    op: str
    lits: tuple
    level: int
    on_path: bool


@dataclasses.dataclass
class ChainCase:
    item_id: str
    domain: rg.Domain
    facts: dict
    checks: list                    # path checks by level, then mirror checks by level (topological order)
    side: str                       # pid of the final's second condition
    outs: list                      # outcome keys: [top passes, side holds, neither]
    label_order: list
    line_order: list                # check ids in prompt order
    reserved: list                  # fact names no check and no side literal reads
    q: float = 0.8
    renderer: str = "chain_shuf"            # "chain_shuf" (training) | "chain_prose" (held-out)

    @property
    def depth(self) -> int:
        return sum(c.on_path for c in self.checks)

    @property
    def top(self) -> str:
        return f"P{self.depth}"

    def by_id(self) -> dict[str, Check]:
        return {c.cid: c for c in self.checks}

    def evaluate(self, overrides: dict | None = None) -> dict:
        return run(self, overrides)

    def gold(self) -> str:
        return self.evaluate()["decision"]


# ---------------------------------------------------------------- engine
def predicates(dom: rg.Domain, names: list[str]) -> list[rg.Pred]:
    """Every predicate of the given facts: bool facts as such, categories by value, numbers and days by cut."""
    out = []
    for f in dom.facts:
        if f.name in names:
            out += [rg.Pred(f.name)] if f.kind == "bool" else \
                [rg.Pred(f.name, a) for a in (f.values if f.kind == "cat" else f.cuts)]
    return out


def gate(op: str, vals: list[bool]) -> bool:
    return all(vals) if op == "and" else any(vals)


def lit_value(lit: Lit, pv: dict, cv: dict) -> bool:
    return (pv[lit.ref] if lit.kind == "p" else cv[lit.ref]) == lit.want


def pred_values(case: ChainCase) -> dict[str, bool]:
    pids = {l.ref for c in case.checks for l in c.lits if l.kind == "p"} | {case.side}
    return {pid: rg.pred_value(case.domain, rg.Pred.parse(pid), case.facts) for pid in sorted(pids)}


def run(case: ChainCase, overrides: dict | None = None) -> dict:
    """Evaluate every check (in order) and the final; `overrides` maps check ids to forced values (downstream checks
    read the forced value)."""
    ov = overrides or {}
    pv = pred_values(case)
    cv: dict[str, bool] = {}
    for c in case.checks:
        cv[c.cid] = ov.get(c.cid, gate(c.op, [lit_value(l, pv, cv) for l in c.lits]))
    o1, o2, o3 = case.outs
    return {"p": pv, "c": cv, "decision": o1 if cv[case.top] else (o2 if pv[case.side] else o3)}


def local_cf(case: ChainCase, check: Check, cv: dict, pv: dict) -> dict[str, bool]:
    """The check's value when one check parent is negated and every other input keeps its value."""
    out = {}
    for l in check.lits:
        if l.kind == "c":
            out[l.ref] = gate(check.op, [lit_value(x, pv, {**cv, l.ref: not cv[l.ref]}) for x in check.lits])
    return out


def pivotal(case: ChainCase) -> dict[str, bool]:
    """Path checks: does overriding the check (and recomputing downstream) flip the final? Mirror checks: False."""
    res = case.evaluate()
    return {c.cid: c.on_path and case.evaluate({c.cid: not res["c"][c.cid]})["decision"] != res["decision"]
            for c in case.checks}


def piv_run(case: ChainCase) -> int:
    """Length of the run of pivotal path checks from the top down. Not the depth a solver needs: with fan-in, flipping
    a check changes two inputs downstream and the effects can cancel (cert_depth is the depth measure)."""
    piv, n = pivotal(case), 0
    for k in range(case.depth, 0, -1):
        if not piv[f"P{k}"]:
            break
        n += 1
    return n


def cert_depth(case: ChainCase, cid: str | None = None) -> int:
    """Certificate depth of a check's value (default: the top check): the number of check levels in the shallowest
    proof of it. AND true / OR false needs every literal (1 + the max over them), AND false / OR true needs one
    controlling literal (1 + the min over those); predicate literals cost 0. An item needs its whole chain exactly
    when cert_depth == depth."""
    res = case.evaluate()
    pv, cv, by = res["p"], res["c"], case.by_id()
    memo: dict[str, int] = {}

    def cert(x: str) -> int:
        if x not in memo:
            c = by[x]
            sub = [(lit_value(l, pv, cv), 0 if l.kind == "p" else cert(l.ref)) for l in c.lits]
            if (c.op == "and") == cv[x]:
                memo[x] = 1 + max(d for _, d in sub)
            else:
                memo[x] = 1 + min(d for v, d in sub if v == (c.op == "or"))
        return memo[x]
    return cert(cid or case.top)


# ---------------------------------------------------------------- sampling
def nonce_names(rng: random.Random, n: int, banned: set[str]) -> list[str]:
    out: list[str] = []
    while len(out) < n:
        w = (rng.choice(_ON) + rng.choice(_VOW) + rng.choice(_ON) + rng.choice(_VOW) + rng.choice(_CODA)).capitalize()
        if 5 <= len(w) <= 9 and w not in out and w.lower() not in banned:
            out.append(w)
    return out


def _vocab(dom: rg.Domain) -> set[str]:
    text = " ".join([dom.title, dom.role, dom.ask, dom.event, dom.request] +
                    [f"{f.label} {f.pos} {f.neg} {f.noun} {' '.join(map(str, f.values))}" for f in dom.facts] +
                    [f"{o[1]} {o[2]}" for o in dom.outcomes])
    return {w.strip(".,;:?()'").lower() for w in text.split()}


def _chain(rng: random.Random, tag: str, depth: int, preds: list[rg.Pred], pv: dict, q: float,
           fanin: float) -> list[Check]:
    """One chain of `depth` checks; values are tracked while building so non-controlling literals can be placed."""
    checks: list[Check] = []
    cv: dict[str, bool] = {}

    def pred_lit(op: str, used: set, control: bool) -> Lit:
        p = rng.choice([x for x in preds if x.fact not in used])
        used.add(p.fact)
        want = (pv[p.pid] if op == "and" else not pv[p.pid]) if control else rng.random() < 0.5
        return Lit("p", p.pid, want)

    for level in range(1, depth + 1):
        op = rng.choice(OPS)
        used: set = set()
        if level == 1:
            lits = [pred_lit(op, used, False), pred_lit(op, used, False)]
        else:
            lits = [Lit("c", f"{tag}{level - 1}", rng.random() < 0.5), pred_lit(op, used, rng.random() < q)]
            if level >= 3 and rng.random() < fanin:          # one more input: an older check of the same chain
                old = f"{tag}{rng.randrange(1, level - 1)}"
                want = (cv[old] if op == "and" else not cv[old]) if rng.random() < q else rng.random() < 0.5
                lits.append(Lit("c", old, want))
        rng.shuffle(lits)
        cid = f"{tag}{level}"
        cv[cid] = gate(op, [lit_value(l, pv, cv) for l in lits])
        checks.append(Check(cid, "", op, tuple(lits), level, tag == "P"))
    return checks


def make_case(dom: rg.Domain, rng: random.Random, item_id: str, depth: int, q: float = 0.8, fanin: float = 0.3,
              n_reserved: int = 2, renderer: str = "chain_shuf") -> ChainCase:
    reserved = rng.sample([f.name for f in dom.facts], n_reserved)
    preds = predicates(dom, [f.name for f in dom.facts if f.name not in reserved])
    cuts: dict[str, list[int]] = {}
    for p in preds:
        if isinstance(p.arg, int):
            cuts.setdefault(p.fact, []).append(p.arg)
    facts = rg.sample_facts(dom, rng, cuts)
    pv = {p.pid: rg.pred_value(dom, p, facts) for p in preds}
    checks = _chain(rng, "P", depth, preds, pv, q, fanin) + _chain(rng, "M", depth, preds, pv, q, fanin)
    names = nonce_names(rng, len(checks), _vocab(dom))
    checks = [dataclasses.replace(c, name=n) for c, n in zip(checks, names)]
    side = rng.choice(preds).pid
    outs = rng.sample([o[0] for o in dom.outcomes], 3)
    order = [o[0] for o in dom.outcomes]
    rng.shuffle(order)
    lines = [c.cid for c in checks]
    rng.shuffle(lines)
    return ChainCase(item_id, dom, facts, checks, side, outs, order, lines, sorted(reserved), q, renderer)


def generate(n: int, depth: int, domains: tuple, seed: Any, q: float = 0.8, fanin: float = 0.3,
             prefix: str = "ch", full_depth: bool = False, renderer: str = "chain_shuf") -> list[ChainCase]:
    """`n` cases of one depth over `domains` (drawn at random), half with the top check passing (label balance).
    full_depth keeps only cases whose top value needs the whole chain (cert_depth == depth; evaluation sets)."""
    rng = random.Random(f"{seed}-d{depth}" if renderer == "chain_shuf" else f"{seed}-d{depth}-{renderer}")
    want = {True: n // 2 + n % 2, False: n // 2}
    out = []
    while len(out) < n:
        dom = rg.DOMAINS[rng.choice(domains)]
        case = make_case(dom, rng, f"{prefix}-d{depth}-{seed}-{len(out):05d}", depth, q, fanin, renderer=renderer)
        top = case.evaluate()["c"][case.top]
        if want[top] > 0 and (not full_depth or cert_depth(case) == depth):
            want[top] -= 1
            out.append(case)
    return out


# ---------------------------------------------------------------- text
def lit_text(case: ChainCase, lit: Lit, by: dict) -> str:
    if lit.kind == "c":
        return f"the {by[lit.ref].name} check {'passes' if lit.want else 'fails'}"
    return rg.clause(case.domain, rg.Pred.parse(lit.ref), lit.want)


def check_line(case: ChainCase, c: Check, by: dict) -> str:
    parts = [lit_text(case, l, by) for l in c.lits]
    if len(parts) == 2:
        body = f" {c.op} ".join(parts)
    else:
        body = ("all of: " if c.op == "and" else "at least one of: ") + "; ".join(parts)
    return f"- The {c.name} check passes if {body}."


def decision_line(case: ChainCase) -> str:
    dom, by = case.domain, case.by_id()
    o1, o2, o3 = case.outs
    return (f"Decision: if the {by[case.top].name} check passes, {dom.imperative(o1)}; otherwise, if "
            f"{rg.clause(dom, rg.Pred.parse(case.side), True)}, {dom.imperative(o2)}; otherwise, {dom.imperative(o3)}.")


def check_sentence(case: ChainCase, c: Check, by: dict) -> str:
    """Held-out prose form of a check line."""
    parts = [lit_text(case, l, by) for l in c.lits]
    if len(parts) == 2:
        body = f" {'and' if c.op == 'and' else 'or'} ".join(parts)
    else:
        body = (("all of these hold: " if c.op == "and" else "at least one of these holds: ")
                + ", ".join(parts[:-1]) + (" and " if c.op == "and" else " or ") + parts[-1])
    return f"The {c.name} check passes if {body}."


def render(case: ChainCase) -> str:
    dom, by = case.domain, case.by_id()
    opts = "\n".join(f"{chr(65 + i)}) {dom.option(k)}" for i, k in enumerate(case.label_order))
    if case.renderer == "chain_prose":
        o1, o2, o3 = case.outs
        sents = " ".join(check_sentence(case, by[i], by) for i in case.line_order)
        return (f"You work as {dom.role}. Several checks apply to {dom.title}, and no check result is given: work each "
                f"one out from the facts. {sents} If the {by[case.top].name} check passes, you {dom.imperative(o1)}; "
                f"otherwise, if {rg.clause(dom, rg.Pred.parse(case.side), True)}, you {dom.imperative(o2)}; "
                f"otherwise, you {dom.imperative(o3)}.\n\n" + " ".join(rg.case_sentences(dom, case.facts))
                + f"\n\n{dom.ask}\nOptions:\n{opts}")
    if case.renderer != "chain_shuf":
        raise ValueError(case.renderer)
    lines = [check_line(case, by[i], by) for i in case.line_order]
    details = "; ".join(f"{a} = {b}" for a, b in rg.case_lines(dom, case.facts))
    return (f"Role: {dom.role}.\nChecks for {dom.title} (no check result is given; work each one out from the request "
            f"details):\n" + "\n".join(lines) + f"\n{decision_line(case)}\n\nRequest details: {details}\n\n"
            f"{dom.ask}\nOptions:\n{opts}")


# ---------------------------------------------------------------- nodes, controls, records
def tree_nodes(case: ChainCase) -> list[dict]:
    """Every check of both chains (path first, by level) with its exact value and per-parent counterfactuals."""
    res, piv = case.evaluate(), pivotal(case)
    out = []
    for c in case.checks:
        cfv = local_cf(case, c, res["c"], res["p"])
        v = res["c"][c.cid]
        out.append({"id": c.cid, "question": f"Does the {c.name} check pass?", "truth": v, "p_cot": float(v),
                    "p_fresh": None, "type": "derive", "depends_on": [l.ref for l in c.lits if l.kind == "c"],
                    "node_depth": c.level, "on_path": c.on_path, "pivotal": piv[c.cid], "op": c.op,
                    "cf": {j: x != v for j, x in cfv.items()}, "cf_value": cfv, "var": f"c:{c.cid}", "kind": "check"})
    return out


def control_questions(case: ChainCase, rng: random.Random, n: int) -> list[dict]:
    """Questions about the reserved facts (read by no check and not by the final), with the share of Yes answers
    matched to the tree nodes as far as the pool allows (rulegen.control_questions)."""
    dom = case.domain
    pool = []
    for p in predicates(dom, case.reserved):
        v = rg.pred_value(dom, p, case.facts)
        pool.append({"question": rg.question(dom, p), "truth": v, "p_fresh": float(v), "var": f"ctl:{p.pid}",
                     "type": "control", "kind": "arith" if dom.fact(p.fact).kind in ("num", "days") else "lookup"})
    rng.shuffle(pool)
    nodes = tree_nodes(case)
    want_yes = round(n * sum(x["truth"] for x in nodes) / max(len(nodes), 1))
    yes = [x for x in pool if x["truth"]]
    no = [x for x in pool if not x["truth"]]
    k_yes = min(want_yes, len(yes))
    out = yes[:k_yes] + no[:n - k_yes]
    out += yes[k_yes:][:n - len(out)]
    rng.shuffle(out)
    return out


def program_json(case: ChainCase) -> dict:
    return {"checks": [{"id": c.cid, "name": c.name, "op": c.op, "level": c.level, "on_path": c.on_path,
                        "lits": [[l.kind, l.ref, l.want] for l in c.lits]} for c in case.checks],
            "top": case.top, "side": case.side, "outs": case.outs, "line_order": case.line_order,
            "reserved": case.reserved, "q": case.q}


def case_from_record(rec: dict) -> ChainCase:
    dom, prog = rg.DOMAINS[rec["domain"]], rec["program"]
    checks = [Check(c["id"], c["name"], c["op"], tuple(Lit(*l) for l in c["lits"]), c["level"], c["on_path"])
              for c in prog["checks"]]
    return ChainCase(rec["item_id"], dom, rg.facts_from_json(dom, rec["facts"]), checks, prog["side"],
                     list(prog["outs"]), list(rec["label_order"]), list(prog["line_order"]), list(prog["reserved"]),
                     prog["q"], rec["renderer"])


def eval_record(case: ChainCase) -> dict:
    res = case.evaluate()
    labels = [chr(65 + i) for i in range(len(case.label_order))]
    return {"item_id": case.item_id, "source": "chain", "group": f"chain/d{case.depth}", "type": "choice",
            "prompt": render(case), "labels": labels, "label_order": case.label_order,
            "gold_label": labels[case.label_order.index(res["decision"])], "gold": res["decision"],
            "label_names": [case.domain.option(k) for k in case.label_order],
            "depth": case.depth, "cert_depth": cert_depth(case), "piv_run": piv_run(case),
            "top_passes": res["c"][case.top], "domain": case.domain.name,
            "renderer": case.renderer, "program": program_json(case), "facts": rg._facts_json(case.facts)}


def train_record(case: ChainCase, rng: random.Random) -> dict:
    """Student record: one-hot final target, both chains' checks as `subqs` (truth-valued, `on_path` marks the chain
    the final reads) and matched questions about the reserved facts as `random_subqs`."""
    ev = eval_record(case)
    nodes = tree_nodes(case)
    return {**{k: ev[k] for k in ("item_id", "source", "group", "domain", "renderer", "prompt", "labels",
                                  "gold_label", "depth", "cert_depth", "piv_run", "top_passes")},
            "teacher": {L: (1.0 if L == ev["gold_label"] else 0.0) for L in ev["labels"]},
            "subqs": nodes, "random_subqs": control_questions(case, rng, len(nodes)),
            "rationale": "", "augment": None, "has_tree": True, "tree_fix": True,
            "label_order": case.label_order, "program": ev["program"], "facts": ev["facts"]}
