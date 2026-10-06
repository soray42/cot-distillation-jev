"""Base/source pairs for the K&K causal audit (notes/next_round/causal_audit/kk_design.md), built on CPU from fresh
held-out programs and checked by the deterministic solver that built the training trees.

Programs: sources.knights_knaves with seeds 71000 + n_people (4-8 people), disjoint from training (100+n), kk_heldout
(9000+n), kk_deep (30000+n) and diag_kk (50000+n). Any program whose statement block also occurs in a K&K item of
data/student_v3*/{train,val}.jsonl is dropped (statement-line set, order ignored); overlaps with the K&K eval sets are
counted in the summary.

Each base program is solved by the solver (teacher_cache/solver_kk_v3f/build_reasoning_trees.py: ProofBuilder.solve,
validate_proof, independent_models). Its unconditional "derive" facts form the proof's causal model: a role fact
derived by case refutation is a root when its refutation subproof uses no earlier unconditional fact, else it is
recomputed from the facts that subproof uses (forced value over the subproof speakers' rules); a fact derived by propagation is the unique value its gate (speaker gate:
role(S) <-> truth(S); expression gate: truth(S) = S's statement over the roles it names) allows given its parent facts.
do(role(X) = not v) recomputes every downstream fact in proof order; the result is undefined when a gate then allows
no value or two values.

Families (one pair per program x person and family when a source is found; every source differs from the base in
exactly one statement and has a unique solution; base and source prompts share one option list, same letters):
  role  source: one statement (any speaker) replaced so that X's role flips. Options hold the base solution, the source
        solution, local = base with only X flipped, prop = do(role(X)) propagated through the proof (when defined and
        different from local), z = base with only Z flipped (Z: a person whose role is the same in base and source; the
        spurious-target control) and near-miss distractors. The source solution is never local, prop or z.
  inv   same-value control: one statement replaced, the solution changes but X's role does not. Options: base, source,
        local (= flip X), distractors. The fitted role(X) subspace should leave the base answer in place.
  stmt  X's own statement replaced (the solution changes); target = the source's answer. Site: the last token of X's
        statement line (also the readout).
Character offsets of X's name (line start) and of the closing quote of X's statement line are stored for both prompts
(the audit maps them to token positions).

  python3 scripts/build_kk_audit_pairs.py            # -> data/eval/kk_audit_pairs.jsonl (+ .summary.json)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import itertools
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill import sources as S  # noqa: E402

SOLVER = ROOT / "teacher_cache/solver_kk_v3f/build_reasoning_trees.py"
SEED0 = 71000
GENERIC = ("Is each of the ", "Must a knight's statement", "Must a knave's statement")   # as build_solver_kk.py
STMT_LINE = re.compile(r'^- ([A-Z][a-z]+) says: "([^"\n]+)"$', re.M)                    # as the solver's Puzzle.read
TRAIN_DIRS = ("data/student_v3", "data/student_v3f", "data/student_v3s", "data/student_v3sl")
EVAL_SETS = ("data/eval/kk_heldout.jsonl", "data/eval/kk_deep.jsonl", "data/eval/diag_kk.jsonl")
N_OPT = 6


def load_solver(path: Path = SOLVER):
    spec = importlib.util.spec_from_file_location("kk_solver", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kk_solver"] = mod          # dataclasses resolve the module by name
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ puzzles
def solutions(people: list[str], stmts: dict) -> list[tuple]:
    return [bits for bits in itertools.product([True, False], repeat=len(people))
            if S._consistent(stmts, dict(zip(people, bits))) == 0]


def flip(sol: tuple, i: int) -> tuple:
    return sol[:i] + (not sol[i],) + sol[i + 1:]


def statement_key(prompt: str) -> tuple:
    """Order-free key of a K&K prompt's statement block (speaker, statement) lines."""
    return tuple(sorted(STMT_LINE.findall(prompt)))


def render(people: list[str], stmts: dict, opts: list[tuple]) -> str:
    """The R0 training template (as sources.knights_knaves and build_diag.render), options in the given order."""
    lines = [f"- {s} says: \"{S._say(stmts[s])[0].upper() + S._say(stmts[s])[1:]}.\"" for s in people]
    intro = ("On an island, knights always tell the truth and knaves always lie. "
             f"Each of the {len(people)} inhabitants below is either a knight or a knave.")
    texts = [", ".join(f"{p}: {'knight' if b else 'knave'}" for p, b in zip(people, bits)) for bits in opts]
    return intro + "\n\n" + "\n".join(lines) + "\n\nQuestion: Who is a knight and who is a knave?\nOptions:\n" + \
        "\n".join(f"{chr(65 + i)}) {t}" for i, t in enumerate(texts))


def spans(prompt: str, person: str) -> dict:
    """Character offsets in the prompt: last character of the person's name at the start of their statement line and
    the closing quote of that line."""
    ms = [m for m in STMT_LINE.finditer(prompt) if m.group(1) == person]
    assert len(ms) == 1, person
    m = ms[0]
    return {"name_end": m.start(1) + len(person) - 1, "stmt_end": m.end() - 1}


def near_misses(people: list[str], stmts: dict, exclude: list[tuple], rng: random.Random, n: int) -> list[tuple]:
    scored = sorted((S._consistent(stmts, dict(zip(people, b))), b)
                    for b in itertools.product([True, False], repeat=len(people)) if b not in exclude)
    out: list[tuple] = []
    for c in sorted({c for c, _ in scored}):            # fewest violated statements first
        tier = [b for cc, b in scored if cc == c]
        rng.shuffle(tier)
        out += tier[:n - len(out)]
        if len(out) >= n:
            break
    return out


def option_list(people, stmts, must: list[tuple], rng: random.Random) -> list[tuple]:
    must = list(dict.fromkeys(must))
    assert len(must) <= N_OPT
    opts = must + near_misses(people, stmts, must, rng, N_OPT - len(must))
    rng.shuffle(opts)
    return opts


# ------------------------------------------------------------------ the proof's causal model
class ProofModel:
    """Unconditional derive facts of a solver proof, with do(role(X)) propagation through the proof's gates."""

    def __init__(self, solver, people: list[str], stmts: dict, sol: tuple, pid: str):
        self.people, self.stmts = people, stmts
        prompt = render(people, stmts, [sol] + near_misses(people, stmts, [sol], random.Random(pid), N_OPT - 1))
        puzzle = solver.Puzzle.read({"item_id": pid, "prompt": prompt}, "kk_audit")
        b = solver.ProofBuilder(puzzle)
        found, _ = b.solve()
        assert found == dict(zip(people, sol)), pid
        models, _ = solver.independent_models(puzzle)
        assert len(models) == 1 and models[0] == found, pid
        solver.validate_proof(b)
        by = {n["id"]: n for n in b.nodes}
        gates = {(g.speaker, g.kind): g for g in b.gates}
        # facts in proof order: id -> dict(var (gate variable), ch (role|truth|requirement), base, gate, parents, speaker)
        self.facts: dict[str, dict] = {}
        for n in b.nodes:
            if n["type"] != "derive":
                continue
            _, scope, (ch, name), value = b.predicates[n["id"]]
            assert scope == ()
            var = ("role", name) if ch == "role" else ("truth", name)    # a requirement fills the truth slot
            parents = [d for d in n["depends_on"] if d in self.facts]
            other = [by[d] for d in n["depends_on"] if d not in self.facts]
            sp = [b.predicates[o["id"]][1] for o in other if b.predicates[o["id"]][0] == "parse"]
            if any(o["type"] == "case" for o in other):       # forced by refuting a case
                assert ch == "role" and not parents and not sp
                gate, speaker = None, None
                # The refutation subproof (case nodes, transitively) may use earlier unconditional facts: those are
                # this root's parents in the proof (TF trains the case nodes on them), so under do() the root is
                # recomputed from them (do_facts). A case root whose refutation uses no fact is exogenous.
                cfacts, cspk, stack, seen = set(), set(), [d for d in n["depends_on"]], set()
                while stack:
                    d = stack.pop()
                    if d in seen:
                        continue
                    seen.add(d)
                    if d in self.facts:
                        cfacts.add(d)
                    elif by[d]["type"] == "case":
                        stack += by[d]["depends_on"]
                    elif by[d]["type"] == "parse" and b.predicates[d][0] == "parse":
                        cspk.add(b.predicates[d][1])
                    else:
                        assert by[d]["type"] == "parse", (pid, d)   # domain rules
                parents = sorted(cfacts, key=lambda x: int(x[1:]))
                case_speakers = sorted(cspk)
            else:
                case_speakers = None
                assert len(sp) == 1 and parents, (pid, n["id"])
                speaker = sp[0]
                pvars = {self.facts[x]["var"] for x in parents} | {var}
                g = gates[(speaker, "speaker")]
                gate = g if pvars <= set(g.variables) else gates[(speaker, "expression")]
                assert pvars <= set(gate.variables), (pid, n["id"])
            self.facts[n["id"]] = {"var": var, "ch": ch, "base": value, "gate": gate, "parents": parents,
                                   "speaker": speaker, "case_speakers": case_speakers}
        self.assigns = [dict(zip(people, bits)) for bits in itertools.product([True, False], repeat=len(people))]
        for i, f in self.facts.items():                        # each case root is forced by its refutation's inputs
            if f["gate"] is None:
                assert self.case_values(f, {p: self.facts[p]["base"] for p in f["parents"]}) == {f["base"]}, (pid, i)
        self.role_node = {f["var"][1]: i for i, f in self.facts.items() if f["ch"] == "role"}
        assert set(self.role_node) == set(people), pid
        assert self.do(None) == tuple(sol), pid               # replaying the proof gives the solution
        # depths: longest path over the proof without generic and verify nodes (treeplan.node_depths on the cleaned
        # graph of build_solver_kk) and over the unconditional facts alone (0 = a case-refuted root whose refutation
        # uses no earlier fact)
        keep = {n["id"]: n for n in b.nodes if n["type"] != "verify" and not n["question"].startswith(GENERIC)}
        memo: dict = {}

        def depth(i):
            if i not in memo:
                ds = [d for d in keep[i]["depends_on"] if d in keep]
                memo[i] = 1 + max(depth(d) for d in ds) if ds else 0
            return memo[i]
        cd: dict = {}
        for i, f in self.facts.items():
            cd[i] = 1 + max(cd[p] for p in f["parents"]) if f["parents"] else 0
        self.tree_depth = {p: depth(self.role_node[p]) for p in people}
        self.chain_depth = {p: cd[self.role_node[p]] for p in people}
        self.how = {p: "case" if self.facts[self.role_node[p]]["gate"] is None else "prop" for p in people}
        self.n_nodes = len(b.nodes)

    def do_facts(self, person: str | None) -> dict | None:
        """Fact values after do(role(person) = not base), recomputed through the proof in order; None if a gate then
        allows no value or several. person=None replays the proof unchanged."""
        val: dict = {}
        for i, f in self.facts.items():
            if f["var"] == ("role", person):
                val[i] = not f["base"]
            elif f["gate"] is None:
                if all(val[p] == self.facts[p]["base"] for p in f["parents"]):
                    val[i] = f["base"]
                else:                                          # a refutation input changed: recompute the root
                    vals = self.case_values(f, {p: val[p] for p in f["parents"]})
                    if len(vals) != 1:
                        return None
                    val[i] = vals.pop()
            else:
                g = f["gate"]
                known = {self.facts[p]["var"]: val[p] for p in f["parents"]}
                rows = [r for r in g.rows if all(r[g.variables.index(v)] == x for v, x in known.items())]
                vals = {r[g.variables.index(f["var"])] for r in rows}
                if len(vals) != 1:
                    return None
                val[i] = vals.pop()
        return val

    def qty(self, f: dict, r: dict) -> bool:
        """A fact's quantity in a role assignment: the evaluated statement for truth, else the role (a requirement's
        value is its speaker's role)."""
        name = f["var"][1]
        return S._eval(self.stmts[name], r) if f["ch"] == "truth" else r[name]

    def case_values(self, f: dict, pval: dict) -> set:
        """Values of a case-refuted root over all role assignments that obey the rules of the speakers its refutation
        parses and agree with the refutation's input facts at the values pval (the base proof gives {base})."""
        name = f["var"][1]
        return {r[name] for r in self.assigns
                if all(r[sp] == S._eval(self.stmts[sp], r) for sp in f["case_speakers"])
                and all(self.qty(self.facts[p], r) == v for p, v in pval.items())}

    def do(self, person: str | None) -> tuple | None:
        val = self.do_facts(person)
        return None if val is None else tuple(val[self.role_node[p]] for p in self.people)


def entailed_cf(model: ProofModel, person: str) -> bool:
    """Independent check of a propagated counterfactual (the validate_proof idea, under the intervention): every proof
    fact downstream of role(person) must be forced, over all role assignments, by its listed parent facts at their
    counterfactual values plus its speaker's rule (role <-> evaluated statement); for a case-refuted root downstream of
    role(person) through its refutation, the rules of the speakers that refutation parses. Uses only the statements and
    sources._eval, not the gate tables."""
    people, stmts = model.people, model.stmts
    val = model.do_facts(person)
    assigns = [dict(zip(people, bits)) for bits in itertools.product([True, False], repeat=len(people))]

    def qty(f, r):
        name = f["var"][1]
        return S._eval(stmts[name], r) if f["ch"] == "truth" else r[name]

    changed = {model.role_node[person]}
    for i, f in model.facts.items():
        if i in changed or not (set(f["parents"]) & changed):
            continue
        changed.add(i)
        sps = [f["speaker"]] if f["gate"] is not None else f["case_speakers"]   # a case root: its refutation's rules
        ok = [r for r in assigns if all(r[sp] == S._eval(stmts[sp], r) for sp in sps)
              and all(qty(model.facts[p], r) == val[p] for p in f["parents"])]
        if {qty(f, r) for r in ok} != {val[i]}:
            return False
    return True


def prop_extra_viol(model: ProofModel, person: str, prop: tuple) -> int:
    """Statements the prop world violates besides X's own and the cut edge (the gate that derived role(X), or for a
    case-refuted X the speakers its refutation parsed). Nonzero when the proof derived a descendant's role from others
    and never used that descendant's own statement: proof-edge propagation is then not a consistent world, and a
    model that propagates through every statement would not land on prop."""
    f = model.facts[model.role_node[person]]
    allowed = {person} | ({f["speaker"]} if f["gate"] is not None else set(f["case_speakers"]))
    w = dict(zip(model.people, prop))
    return sum(w[sp] != S._eval(model.stmts[sp], w) for sp in model.people if sp not in allowed)


# ------------------------------------------------------------------ sources
def find_source(people, stmts, sol, rng, ok, speakers=None, tries=800, banned: set = frozenset()):
    """A one-statement edit with a unique solution != sol that satisfies ok(solution); never a puzzle in `banned`
    (statement keys of training items)."""
    for _ in range(tries):
        e = rng.choice(speakers or people)
        new = S._statement(rng, people, e, 0.7)
        if new == stmts[e]:
            continue
        st = dict(stmts, **{e: new})
        sols = solutions(people, st)
        if len(sols) == 1 and sols[0] != sol and ok(sols[0]):
            if banned and statement_key(render(people, st, [sols[0]])) in banned:
                continue
            return st, sols[0], e
    return None


def training_keys() -> tuple[set, int]:
    keys, n = set(), 0
    for d in TRAIN_DIRS:
        for split in ("train", "val"):
            p = ROOT / d / f"{split}.jsonl"
            if not p.exists():
                continue
            with open(p) as f:
                for line in f:
                    if '"knights_knaves"' not in line:
                        continue
                    r = json.loads(line)
                    if r.get("source") == "knights_knaves":
                        keys.add(statement_key(r["prompt"]))
                        n += 1
    return keys, n


def eval_keys() -> dict:
    out = {}
    for p in EVAL_SETS:
        if (ROOT / p).exists():
            with open(ROOT / p) as f:
                out[p] = {statement_key(json.loads(l)["prompt"]) for l in f if l.strip()}
    return out


def build(solver, n_per_size: int, sizes: range, train_keys: set, ev_keys: dict, cap_inv: int = 2, cap_stmt: int = 2,
          log=print):
    out, c = [], collections.Counter()
    for size in sizes:
        progs = S.knights_knaves(n_per_size, seed=SEED0 + size, people_range=(size, size))
        for r in progs:
            people = r["meta"]["people"]
            stmts = {k: tuple(v) for k, v in r["meta"]["statements"].items()}
            sols = solutions(people, stmts)
            assert len(sols) == 1
            sol = sols[0]
            pid = f"kkaud-{size}-{r['item_id'].rsplit('-', 1)[1]}"
            key = statement_key(render(people, stmts, [sol]))
            if key in train_keys:
                c["dropped_train_overlap"] += 1
                continue
            for p, ks in ev_keys.items():
                c[f"eval_overlap/{Path(p).name}"] += key in ks
            pm = ProofModel(solver, people, stmts, sol, pid)
            c["programs"] += 1
            caps = {"inv": cap_inv, "stmt": cap_stmt}
            got = collections.Counter()
            order = list(range(len(people)))
            random.Random(f"kkaudit-order-{pid}").shuffle(order)          # which people get inv/stmt pairs
            for i in order:
                x = people[i]
                rng = random.Random(f"kkaudit-{pid}-{x}")
                local = flip(sol, i)
                prop = pm.do(x)
                if prop is not None:
                    assert prop[i] != sol[i] and entailed_cf(pm, x), (pid, x)
                    c["prop_defined"] += 1
                else:
                    c["prop_undefined"] += 1
                n_desc = None if prop is None else sum(a != b for a, b in zip(prop, sol)) - 1
                meta = {"program": pid, "n_people": size, "person": x, "how": pm.how[x],
                        "chain_depth": pm.chain_depth[x], "tree_depth": pm.tree_depth[x], "n_desc": n_desc,
                        "subfam": "undef" if prop is None else ("mid" if n_desc else "leaf")}
                # role: X flips in the source; the source answer is none of the hypotheses' targets
                cand = {local} | ({prop} if prop else set())
                src = find_source(people, stmts, sol, rng, lambda s: s[i] != sol[i] and s not in cand, banned=train_keys)
                if src:
                    st, ssol, e = src
                    zs = [j for j in range(len(people)) if j != i and ssol[j] == sol[j]
                          and flip(sol, j) not in cand | {ssol}]
                    # zprop: do(role(Z)) propagated (Z unchanged in the source), the plausibility-matched floor of prop.
                    # For mid pairs (X has proof descendants) Z is drawn from the people whose zprop is valid when there
                    # are any, so Z also has descendants and the zprop control covers most mid pairs (uniform draw:
                    # 591/1712); otherwise, and for leaf/undef pairs, Z is drawn uniformly as before (one rng.choice
                    # either way, so leaf/undef/inv/stmt pairs are unchanged).
                    def zprop_of(j):
                        zp = pm.do(people[j])
                        return None if zp is None or zp == flip(sol, j) or zp in cand | {ssol, sol} else zp
                    zs_m = [j for j in zs if zprop_of(j) is not None] if meta["subfam"] == "mid" else []
                    z = rng.choice(zs_m or zs) if zs else None
                    zprop = zprop_of(z) if z is not None else None
                    if zprop is not None:
                        assert entailed_cf(pm, people[z]), (pid, people[z])
                    must = [sol, ssol, local] + ([prop] if prop and prop != local else []) + \
                        ([flip(sol, z)] if z is not None else []) + ([zprop] if zprop is not None else [])
                    desc = [j for j in range(len(people)) if prop and j != i and prop[j] != sol[j]]
                    extra = {"src_desc_agree": round(sum(ssol[j] == prop[j] for j in desc) / len(desc), 4)
                             if desc else None,
                             "prop_extra_viol": prop_extra_viol(pm, x, prop) if prop else None,
                             "viol_base": {k: None if t is None else S._consistent(stmts, dict(zip(people, t)))
                                           for k, t in (("local", local), ("prop", prop),
                                                        ("z", None if z is None else flip(sol, z)),
                                                        ("zprop", zprop), ("source", ssol))}}
                    out.append(pair(people, stmts, st, sol, ssol, must, rng, "role", e, dict(meta, **extra),
                                    local=local, prop=prop, z=None if z is None else flip(sol, z),
                                    z_person=None if z is None else people[z], zprop=zprop))
                    c["role"] += 1
                    c[f"role_{meta['subfam']}"] += 1
                # inv: the solution changes, X's role does not
                if got["inv"] < caps["inv"]:
                    src = find_source(people, stmts, sol, rng, lambda s: s[i] == sol[i] and s != local,
                                      banned=train_keys)
                    if src:
                        st, ssol, e = src
                        # prop in the options too: a role(X) subspace fitted to prop should not move inv pairs there
                        pp = prop if prop and prop != local and prop != ssol else None
                        out.append(pair(people, stmts, st, sol, ssol, [sol, ssol, local] + ([pp] if pp else []), rng,
                                        "inv", e, meta, local=local, prop=pp))
                        got["inv"] += 1
                        c["inv"] += 1
                # stmt: X's own statement replaced
                if got["stmt"] < caps["stmt"]:
                    src = find_source(people, stmts, sol, rng, lambda s: True, speakers=[x], banned=train_keys)
                    if src:
                        st, ssol, e = src
                        out.append(pair(people, stmts, st, sol, ssol, [sol, ssol], rng, "stmt", e, meta))
                        got["stmt"] += 1
                        c["stmt"] += 1
        log(f"size {size}: {len(out)} pairs so far")
    return out, c


def pair(people, stmts, sst, sol, ssol, must, rng, family, edited, meta, local=None, prop=None, z=None,
         z_person=None, zprop=None) -> dict:
    opts = option_list(people, stmts, must, rng)
    bp, sp = render(people, stmts, opts), render(people, sst, opts)
    ix = lambda a: None if a is None else opts.index(a)
    x = meta["person"]
    rec = {"pair_id": f"{meta['program']}-{x}-{family}", "family": family, **meta, "edited": edited,
           "base_prompt": bp, "source_prompt": sp, "labels": [chr(65 + k) for k in range(len(opts))],
           "options": [[int(b) for b in o] for o in opts], "people": people,
           "base_gold": ix(sol), "source_gold": ix(ssol), "t_local": ix(local),
           "t_prop": ix(prop), "t_z": ix(z), "z_person": z_person, "t_zprop": ix(zprop),
           "slot": people.index(x), "z_slot": None if z_person is None else people.index(z_person),
           "base_spans": spans(bp, x), "source_spans": spans(sp, x),
           "base_statements": {k: list(v) for k, v in stmts.items()},
           "source_statements": {k: list(v) for k, v in sst.items()}}
    return rec


def solve_prompt(solver, prompt: str, item_id: str) -> tuple[dict, str]:
    """The solver's solution and option letter for a rendered prompt (its unique-solution check included)."""
    puzzle = solver.Puzzle.read({"item_id": item_id, "prompt": prompt}, "kk_audit")
    sol, letter = solver.ProofBuilder(puzzle).solve()
    models, _ = solver.independent_models(puzzle)
    assert len(models) == 1 and models[0] == sol, item_id
    return sol, letter


def verify_pair(solver, rec: dict) -> None:
    """Re-check one pair from its prompts and statements alone (raises AssertionError)."""
    people, x = rec["people"], rec["person"]
    i = people.index(x)
    assert rec["slot"] == i
    opts = [tuple(bool(b) for b in o) for o in rec["options"]]
    assert len(opts) == N_OPT == len(set(opts)) and rec["labels"] == [chr(65 + k) for k in range(N_OPT)]
    bsol, bl = solve_prompt(solver, rec["base_prompt"], rec["pair_id"] + "-b")
    ssol, sl = solve_prompt(solver, rec["source_prompt"], rec["pair_id"] + "-s")
    assert bl == rec["labels"][rec["base_gold"]] and sl == rec["labels"][rec["source_gold"]], rec["pair_id"]
    b, s = tuple(bsol[p] for p in people), tuple(ssol[p] for p in people)
    assert b != s
    bst = {k: tuple(_tup(v)) for k, v in rec["base_statements"].items()}
    sst = {k: tuple(_tup(v)) for k, v in rec["source_statements"].items()}
    diff = [k for k in people if bst[k] != sst[k]]
    assert diff == [rec["edited"]], rec["pair_id"]
    assert render(people, bst, opts) == rec["base_prompt"] and render(people, sst, opts) == rec["source_prompt"]
    bl_, sl_ = rec["base_prompt"].split("\n"), rec["source_prompt"].split("\n")
    assert sum(u != v for u, v in zip(bl_, sl_)) == 1 and len(bl_) == len(sl_)
    for key, pr in (("base_spans", rec["base_prompt"]), ("source_spans", rec["source_prompt"])):
        sp = rec[key]
        assert pr[sp["stmt_end"]] == '"' and pr[sp["name_end"] - len(x) + 1:sp["name_end"] + 1] == x
    if rec["family"] in ("role", "inv"):
        assert opts[rec["t_local"]] == flip(b, i)
    if rec["family"] == "role":
        assert s[i] != b[i]
        pm = ProofModel(solver, people, bst, b, rec["program"])
        prop = pm.do(x)
        assert (prop is None) == (rec["t_prop"] is None)
        if prop is not None:
            assert opts[rec["t_prop"]] == prop and entailed_cf(pm, x)
        hyp = [opts[rec["t_local"]]] + ([prop] if prop else [])
        if rec["t_z"] is not None:
            z = people.index(rec["z_person"])
            assert z != i and s[z] == b[z] and opts[rec["t_z"]] == flip(b, z) and rec["z_slot"] == z
            hyp.append(opts[rec["t_z"]])
            zprop = pm.do(rec["z_person"])
            if rec["t_zprop"] is not None:
                assert zprop is not None and opts[rec["t_zprop"]] == zprop and zprop != flip(b, z)
                assert entailed_cf(pm, rec["z_person"]) and zprop not in hyp
                hyp.append(zprop)
        else:
            assert rec["t_zprop"] is None
        assert s not in hyp and b not in hyp, rec["pair_id"]
    elif rec["family"] == "inv":
        assert s[i] == b[i] and s != flip(b, i)
        if rec["t_prop"] is not None:
            pm = ProofModel(solver, people, bst, b, rec["program"])
            assert opts[rec["t_prop"]] == pm.do(x) != flip(b, i) and entailed_cf(pm, x)
    else:
        assert rec["family"] == "stmt" and rec["edited"] == x


def _tup(v):
    return ("is", v[1], bool(v[2])) if v[0] == "is" else (v[0], _tup(v[1]), _tup(v[2]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval/kk_audit_pairs.jsonl")
    ap.add_argument("--n-per-size", type=int, default=60)
    ap.add_argument("--sizes", default="4-8")
    ap.add_argument("--cap-inv", type=int, default=2, help="inv pairs per program")
    ap.add_argument("--cap-stmt", type=int, default=2, help="stmt pairs per program")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.sizes.split("-"))
    solver = load_solver()
    tk, n_train = training_keys()
    pairs, c = build(solver, args.n_per_size, range(lo, hi + 1), tk, eval_keys(), args.cap_inv, args.cap_stmt)
    for p in pairs:                                     # every pair re-solved from its own prompts
        verify_pair(solver, p)
    path = ROOT / args.out
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    fam = collections.Counter(p["family"] for p in pairs)
    summ = {"file": args.out, "md5": hashlib.md5(path.read_bytes()).hexdigest(), "pairs": len(pairs),
            "by_family": dict(fam), "counts": dict(sorted(c.items())), "seed0": SEED0, "sizes": args.sizes,
            "n_per_size": args.n_per_size, "cap_inv": args.cap_inv, "cap_stmt": args.cap_stmt, "verified_pairs": len(pairs), "training_kk_items_checked": n_train, "training_dirs": TRAIN_DIRS,
            "solver": str(SOLVER.relative_to(ROOT)),
            "solver_md5": hashlib.md5(SOLVER.read_bytes()).hexdigest(),
            "builder_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
            "by_family_size": {f"{k[0]}/{k[1]}": v for k, v in
                               sorted(collections.Counter((p["family"], p["n_people"]) for p in pairs).items())},
            "role_subfam_size": {f"{k[0]}/{k[1]}": v for k, v in sorted(collections.Counter(
                (p["subfam"], p["n_people"]) for p in pairs if p["family"] == "role").items())},
            "role_mid_src_desc_agree": dict(sorted(collections.Counter(
                "none" if p["src_desc_agree"] == 0 else ("all" if p["src_desc_agree"] == 1 else "some")
                for p in pairs if p["family"] == "role" and p["subfam"] == "mid").items())),
            "role_mid_viol_gap": dict(sorted(collections.Counter(
                (lambda v: "prop<local" if v["prop"] < v["local"] else ("prop=local" if v["prop"] == v["local"]
                                                                       else "prop>local"))(p["viol_base"])
                for p in pairs if p["family"] == "role" and p["subfam"] == "mid").items())),
            "role_zprop_defined": dict(sorted(collections.Counter(
                f'{p["subfam"]}/{p["t_zprop"] is not None}' for p in pairs if p["family"] == "role").items())),
            "role_by_slot": dict(sorted(collections.Counter(
                f'{p["subfam"]}/{p["slot"]}' for p in pairs if p["family"] == "role").items())),
            "role_mid_prop_extra_viol": dict(sorted(collections.Counter(
                str(min(p["prop_extra_viol"], 2)) for p in pairs if p["family"] == "role" and p["subfam"] == "mid"
            ).items())),
            "inv_prop_defined": sum(p["t_prop"] is not None for p in pairs if p["family"] == "inv"),
            "role_by_how_ndesc": {f"{k[0]}/{k[1]}": v for k, v in sorted(collections.Counter(
                (p["how"], "undef" if p["n_desc"] is None else str(min(p["n_desc"], 3))) for p in pairs
                if p["family"] == "role").items())}}
    Path(str(path) + ".summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
