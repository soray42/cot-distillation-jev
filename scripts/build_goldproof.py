"""Gold-proof logic slice L (notes/next_round/data_expansion/plan.md §1.2, §2 row 6): final questions with yes/no tree
nodes whose truth comes from a gold proof or from our own solver, never from a teacher.

Sources:
- L1 FLD v2 (data/raw/bench_math/fld_v2, `star` config by default: the `default` config's tasksource copy is one of the
  22 S-fam families of data/eval/v4_heldout.jsonl, which must stay unseen; --fld-config default overrides). Nodes are
  the proof's intermediate conclusions (`intN`), asked as 'Can the statement "X" be proved from the facts?' (yes) or
  in the opposite wording '... disproved ...' (no). Conclusions derived while an assumption is open (`void ->
  assumpN` until `[assumpN] & ...` discharges it) are not entailed by the facts and are dropped, as is the step that
  reaches the hypothesis (the answer). depends_on: the nearest kept intermediate premises. The "disproved" wording is
  exact under the assumption that an FLD context is consistent (the generator's design; not checked here). UNKNOWN
  items are finals only: their proofs never reach the hypothesis and their steps cannot be checked here (a structural
  check of them is reported).
- L2 BoardgameQA (data/raw/bench_math/boardgameqa): the gold `proof` text is split on its fixed phrases. Nodes are
  the facts it starts from ("We know F": parse), every conclusion but the last one ("so we can conclude C": derive,
  depending on the facts and conclusions it uses) and every antecedent the proof could not prove ("we cannot prove
  the antecedent A": fail, truth no). The last conclusion is the answer and is dropped. Unknown items are finals only.
- L3 RuleTaker rows of the local tasksource pool (data/raw/v4/tasksource_jev, Apache-2.0), and L3' ProofWriter rows
  (licence unverified: written to a separate lab-only file, never into train.jsonl / val.jsonl). Contexts are parsed
  into facts and Horn rules ("X is (not) A", "X (does not) V the Y", "If ... then ...", "All A, B things are C",
  "A things are C") and solved by forward chaining: RuleTaker under the closed world (negation as failure,
  stratified by predicate), ProofWriter under the open world (negated facts are literals; a context that derives a
  literal and its negation is dropped). A row is kept only if the solver's answer equals the pool label, so the
  self-check holds on 100% of kept rows by construction; the agreement rate over all parsed rows is the real audit and
  is reported with every drop reason. Nodes: the proof tree of the queried literal (or of its negation), minus its
  top (the answer), and for underivable literals the first failing condition of every rule that could derive them.
  Half of the nodes are asked about the negated literal, whose truth the solver also gives (exact under both worlds).

Every row keeps the student_v3f schema (prompt, labels, gold_label, one-hot teacher, subqs with
id/question/truth/p_cot/depends_on/node_depth/type) plus provenance: licence_class, label_basis, node_basis, tier
(lab / release / 4b: the smallest plan size that includes the row; rows are ordered so that each tier is a prefix per
source), decontam. Decontamination: scripts/decontam.py (13-gram on the prompt, against every data/eval file) on a
temporary copy, then a digit-masked 10-gram and masked word-Jaccard >= .5 gate (plan §3.4 step 2) over the prompt and
every node question, against every data/eval/*.jsonl item; rows hit by either are dropped and counted per eval file.

  ~/.venvs/sev/bin/python scripts/build_goldproof.py --out data/student_goldproof \
      --tmp /home/soray/.claude/jobs/3b5d797f/tmp/build
  (needs pandas/pyarrow; under ~/.venvs/sev, which has neither, use python3)
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.treeplan import node_depths  # noqa: E402

TIERS = {"fld": (2500, 4000, 5000), "bgqa": (1500, 2000, 3000), "ruletaker": (1500, 2500, 3000)}
LETTERS = "ABCDEFGH"

# ----------------------------------------------------------------------------------------------------------------
# RuleTaker / ProofWriter: parser and forward-chaining solver
# ----------------------------------------------------------------------------------------------------------------

ADJ = ("blue round red cold kind big nice green young rough smart white quiet furry").split()
VERBS = ("see like chase need eat visit").split()
PRON = {"someone", "something", "they", "it", "them", "people", "things"}
ENT = r"(?:[A-Z][a-z]+|[Tt]he (?:bald eagle|[a-z]+)|someone|something|they|it|them)"
RE_IS = re.compile(rf"^({ENT}) (?:is|are) (not )?([a-z]+)$")
RE_ALL = re.compile(r"^(?:All )?([A-Za-z]+(?:,? (?:and )?[a-z]+)*) (?:things|people) are (not )?([a-z]+)$")
RE_STATE = re.compile(r"(?s)^(?:A: |Passage A:\n|text_A: |First text:\n)(.*?)\n+(?:B: |Passage B:\n|text_B: |Second text:\n)(.*)$")


class ParseError(ValueError):
    pass


def _term(t: str) -> str:
    t = t.strip()
    if t.lower() in PRON:
        return "?x"
    if t.lower().startswith("the "):
        return "the " + t[4:]
    return t


def _verb(v: str) -> str:
    for lemma in VERBS:
        if v in (lemma, lemma + "s", lemma + "es"):
            return lemma
    raise ParseError(f"verb {v}")


def parse_clause(text: str, prev_subj: str | None = None) -> tuple:
    """One literal (subject, predicate, object, negated): predicate "is" with an adjective object, or a verb lemma
    with an entity object. A bare adjective ("round", "not round") takes the previous clause's subject."""
    text = text.strip().rstrip(".")
    m = RE_IS.match(text)
    if m and m.group(3) in ADJ:
        return (_term(m.group(1)), "is", m.group(3), bool(m.group(2)))
    m = re.match(rf"^({ENT}) (does not |do not )?([a-z]+) ({ENT})$", text)
    if m and m.group(3) not in ("is", "are"):
        return (_term(m.group(1)), _verb(m.group(3)), _term(m.group(4)), bool(m.group(2)))
    m = re.match(r"^(not )?([a-z]+)$", text)
    if m and m.group(2) in ADJ and prev_subj is not None:
        return (prev_subj, "is", m.group(2), bool(m.group(1)))
    raise ParseError(text)


def parse_conj(text: str) -> list[tuple]:
    out, prev = [], None
    for part in re.split(r",? and ", text):
        lit = parse_clause(part, prev)
        out.append(lit)
        prev = lit[0]
    return out


def parse_sentence(s: str) -> tuple[str, object]:
    s = s.strip().rstrip(".")
    m = re.match(r"^If (.+?),? then (.+)$", s)
    if m:
        return "rule", (parse_conj(m.group(1)), parse_conj(m.group(2)))
    m = RE_ALL.match(s)
    if m and m.group(3) in ADJ:
        adjs = [a.lower() for a in re.split(r",? (?:and )?|, ", m.group(1)) if a]
        if all(a in ADJ for a in adjs):
            return "rule", ([("?x", "is", a, False) for a in adjs], [("?x", "is", m.group(3), bool(m.group(2)))])
    lit = parse_clause(s)
    if "?x" in (lit[0], lit[2]):
        raise ParseError(s)
    return "fact", lit


def parse_context(ctx: str) -> tuple[list[tuple], list[tuple]]:
    facts, rules = [], []
    for sent in re.split(r"(?<=\.)\s+", ctx.strip()):
        if not sent:
            continue
        kind, x = parse_sentence(sent)
        (facts if kind == "fact" else rules).append(x)
    return facts, rules


def render(lit: tuple) -> str:
    s, p, o, neg = lit
    if p == "is":
        return f"{s} is {'not ' if neg else ''}{o}"
    return f"{s} does not {p} {o}" if neg else f"{s} {p}{'es' if p.endswith('s') else 's'} {o}"


def cap(t: str) -> str:
    return t[0].upper() + t[1:]


def ground(lits: list[tuple], e: str) -> list[tuple]:
    return [tuple(e if v == "?x" else v for v in lit[:3]) + (lit[3],) for lit in lits]


def neg(lit: tuple) -> tuple:
    return lit[:3] + (not lit[3],)


def _key(lit: tuple) -> tuple:
    return lit[:3]                                # ground atom: local stratification


class Solver:
    """Forward chaining over ground literals. closed=True: positives only, negated conditions by negation as
    failure, rules evaluated stratum by stratum (predicate-level stratification); closed=False: negated literals are
    first-class (stated or derived), monotone fixpoint."""

    def __init__(self, facts, rules, entities, closed: bool):
        self.closed = closed
        self.grules = []
        for ri, (body, heads) in enumerate(rules):
            uses_var = any("?x" in (lit[0], lit[2]) for lit in body + heads)
            for e in (entities if uses_var else [None]):
                b = ground(body, e) if e else [tuple(l) for l in body]
                for h in (ground(heads, e) if e else [tuple(l) for l in heads]):
                    if closed and h[3]:
                        continue                      # CWA: positives come from derivation alone
                    self.grules.append((ri, b, h))
        self.known: dict[tuple, tuple | None] = {}   # literal -> (rule index, body) or None for a stated fact
        for f in facts:
            if closed and f[3]:
                continue
            self.known.setdefault(tuple(f), None)
        if closed:
            self._strata()
        else:
            self._fixpoint(self.grules)
            self.contradiction = any(neg(l) in self.known for l in self.known)

    def holds(self, lit: tuple) -> bool:
        if self.closed and lit[3]:
            return lit[:3] + (False,) not in self.known
        return lit in self.known

    def _fixpoint(self, grules):
        changed = True
        while changed:
            changed = False
            for ri, body, h in grules:
                if h in self.known:
                    continue
                if all(self.holds(b) for b in body):
                    self.known[h] = (ri, body)
                    changed = True

    def _strata(self):
        keys = {_key(h) for _, _, h in self.grules} | {_key(b) for _, body, _ in self.grules for b in body}
        st = {k: 0 for k in keys}
        for _ in range(len(keys) + 1):
            changed = False
            for _, body, h in self.grules:
                for b in body:
                    need = st[_key(b)] + (1 if b[3] else 0)
                    if st[_key(h)] < need:
                        st[_key(h)] = need
                        changed = True
            if not changed:
                break
        else:
            raise ParseError("not stratifiable")
        if max(st.values(), default=0) > len(keys):
            raise ParseError("not stratifiable")
        for s in sorted(set(st.values())):
            self._fixpoint([g for g in self.grules if st[_key(g[2])] == s])

    def failing(self, lit: tuple) -> list[tuple]:
        """For an underivable literal: the first condition that fails in every ground rule that could derive it."""
        out = []
        for ri, body, h in self.grules:
            if h == lit:
                for b in body:
                    if not self.holds(b):
                        out.append(b)
                        break
        return list(dict.fromkeys(out))


def solver_nodes(sv: Solver, target: tuple, rng: random.Random, closed: bool) -> tuple[list[dict], int]:
    """Nodes for one queried literal (see the module docstring). Returns (nodes, proof depth of the deciding side)."""
    answer_atoms = {target[:3]}
    nodes: dict[tuple, dict] = {}

    def add(lit: tuple, typ: str, deps: list[tuple]) -> tuple:
        if lit not in nodes:
            nodes[lit] = {"lit": lit, "type": typ, "deps": deps}
        return lit

    def proof(lit: tuple) -> tuple:
        """Proof tree of a literal that holds. Returns its key."""
        if lit in nodes:
            return lit
        if closed and lit[3]:                         # NAF condition: "lit" holds because its positive fails
            return add(lit, "naf", [])
        how = sv.known[lit]
        if how is None:
            return add(lit, "parse", [])
        deps = [proof(b) for b in how[1]]
        return add(lit, "derive", deps)

    def depth(lit: tuple) -> int:
        how = sv.known.get(lit) if not (closed and lit[3]) else None
        return 0 if how is None else 1 + max((depth(b) for b in how[1]), default=0)

    decided = None
    for side in (target, neg(target)):
        if (not closed or not side[3]) and side in sv.known:
            decided = side
            break
    if decided is not None:                           # proof of the literal (or of its negation): drop its top
        top = proof(decided)
        d = depth(decided)
    else:                                             # nothing derives it: why every candidate rule fails
        d = 0
        pos = target[:3] + (False,)
        sides = [pos] if closed else [pos, neg(pos)]
        for s in sides:
            for b in sv.failing(s):
                if closed and b[3]:                   # NAF condition fails: its positive is derived
                    proof(b[:3] + (False,))
                else:                                 # condition not derivable (CWA) / not provable (OWA)
                    add(b, "fail", [])
        top = None
    out, ids = [], {}
    for lit, n in nodes.items():
        if lit == top or lit[:3] in answer_atoms:
            continue
        ids[lit] = f"n{len(ids) + 1}"
    for lit, n in nodes.items():
        if lit not in ids:
            continue
        ask = neg(lit) if rng.random() < 0.5 else lit
        truth = sv.holds(ask)
        if closed:
            q = f"Does it follow from the context that {render(ask)}? What is not explicitly stated as true is considered false."
        else:
            q = f"Can it be proved from the context that {render(ask)}?"
        deps = [ids[x] for x in n["deps"] if x in ids]
        out.append({"id": ids[lit], "question": q, "truth": truth, "type": n["type"], "depends_on": deps,
                    "flipped": ask != lit, "content": render(ask)})
    return out, d


def pool_rows(pattern: str, sources=("ruletaker", "proofwriter")) -> list[dict]:
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(glob.glob(str(ROOT / pattern))):
        t = pq.read_table(f)
        src = t.column("source").to_pylist()
        rows += t.take([i for i, s in enumerate(src) if s in sources]).to_pylist()
    return rows


def build_rule_items(rows: list[dict], source: str, rng: random.Random, c: collections.Counter) -> list[dict]:
    closed = source == "ruletaker"
    seen, out = set(), []
    for r in rows:
        if r["source"] != source or r["variant"] != "direct" or r["kind"] != "choice":
            continue
        c[f"{source}/rows_direct"] += 1
        if r["group_id"] in seen:
            c[f"{source}/drop_duplicate"] += 1
            continue
        seen.add(r["group_id"])
        m = RE_STATE.match(r["state"])
        if not m:
            c[f"{source}/drop_format"] += 1
            continue
        ctx, stmt = m.group(1).strip(), m.group(2).strip()
        label = r["options"][max(range(len(r["target"])), key=lambda i: r["target"][i])]
        try:
            facts, rules = parse_context(ctx)
            target = parse_clause(stmt)
            if "?x" in (target[0], target[2]):
                raise ParseError(stmt)
        except ParseError:
            c[f"{source}/drop_unparsed"] += 1
            continue
        lits = facts + [lit for b, h in rules for lit in b + h] + [target]
        ents = sorted({x for lit in lits for x in ((lit[0], lit[2]) if lit[1] != "is" else (lit[0],)) if x != "?x"})
        try:
            sv = Solver(facts, rules, ents, closed)
        except ParseError:
            c[f"{source}/drop_not_stratifiable"] += 1
            continue
        c[f"{source}/parsed"] += 1
        if not closed and sv.contradiction:
            c[f"{source}/drop_contradiction"] += 1
            continue
        if closed:
            pred = "entailment" if sv.holds(target) else "not entailment"
        else:
            pred = "True" if target in sv.known else "False" if neg(target) in sv.known else "Unknown"
        c[f"{source}/label_{label}"] += 1
        if pred != label:
            c[f"{source}/drop_label_mismatch"] += 1
            c[f"{source}/mismatch_{label}->{pred}"] += 1
            continue
        nodes, d = solver_nodes(sv, target, rng, closed)
        if closed:
            names = ["entailment", "not entailment"]
            q = "Does the statement follow from the context? What is not explicitly stated as true is considered false."
        else:
            names = ["True", "False", "Unknown"]
            q = "Based only on the context, is the statement true, false, or unknown?"
        out.append({"pool_id": r["id"], "split": r["split"], "context": ctx, "statement": stmt, "question": q,
                    "names": names, "gold": label, "nodes": nodes, "depth": d,
                    "licence": r["license"], "license_use": r["license_use"]})
    c[f"{source}/kept"] += len(out)
    return out


# ----------------------------------------------------------------------------------------------------------------
# FLD v2
# ----------------------------------------------------------------------------------------------------------------

RE_STEP = re.compile(r"^(.*?)\s*->\s*([a-z]+\d*)\s*(?::\s*(.*))?$")


def fld_nodes(proof: str, context: str, c: collections.Counter) -> list[dict] | None:
    sents = set(re.findall(r"\b(sent\d+):", context))
    steps, defined = [], set()
    for raw in proof.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        m = RE_STEP.match(raw)
        if not m:
            c["fld/proof_unparsed_step"] += 1
            return None
        prem = [p.strip() for p in m.group(1).split("&")]
        refs = []
        for p in prem:
            b = p.strip("[]")
            if p == "void":
                continue
            if b.startswith("sent"):
                if b not in sents:
                    c["fld/proof_missing_sent"] += 1
                    return None
                continue
            if b not in defined:
                c["fld/proof_undefined_ref"] += 1
                return None
            refs.append((b, p.startswith("[")))
        name = m.group(2)
        defined.add(name)
        steps.append((name, refs, (m.group(3) or "").strip().rstrip(".").strip()))
    open_: dict[str, frozenset] = {}
    up: dict[str, list[str]] = {}
    kept = []
    for name, refs, text in steps:
        o = set()
        for b, discharged in refs:
            if discharged:
                continue
            o |= open_.get(b, frozenset())
        if name.startswith("assump"):
            o.add(name)
        for b, discharged in refs:
            if discharged:
                o.discard(b)
        open_[name] = frozenset(o)
        up[name] = [b for b, _ in refs]
        if name.startswith("int") and not o and text and text != "this is contradiction":
            kept.append((name, text))
        elif name.startswith("int"):
            c["fld/node_under_assumption"] += 1
    keep = {n for n, _ in kept}

    def nearest(b: str, seen: set) -> list[str]:
        if b in keep:
            return [b]
        out = []
        for x in up.get(b, []):
            if x not in seen:
                seen.add(x)
                out += nearest(x, seen)
        return out
    return [{"name": n, "text": t, "deps": list(dict.fromkeys(x for b in up[n] for x in nearest(b, set())))}
            for n, t in kept]


def build_fld(cfg: str, split: str, rng: random.Random, c: collections.Counter) -> list[dict]:
    import pandas as pd
    path = glob.glob(str(ROOT / f"data/raw/bench_math/fld_v2/{cfg}/{split}-*.parquet"))[0]
    out = []
    for i, r in enumerate(pd.read_parquet(path).to_dict("records")):
        label = r["world_assump_label"]
        assert label == r["proof_label"], (i, label, r["proof_label"])
        sentences = [s.strip() for s in re.split(r"\s*sent\d+:\s*", r["context"]) if s.strip()]
        nodes = []
        if label in ("PROVED", "DISPROVED"):
            got = fld_nodes(r["proofs"][0], r["context"], c)
            if got is None:
                c[f"fld/drop_proof_{label}"] += 1
                continue
            answer = {str(h).strip().rstrip(".").lower() for h in (r["hypothesis"], r["negative_hypothesis"]) if isinstance(h, str)}
            for k, n in enumerate(got):
                if n["text"].lower() in answer:
                    c["fld/node_equals_hypothesis"] += 1
                    continue
                flip = rng.random() < 0.5
                word = "disproved" if flip else "proved"
                nodes.append({"id": n["name"], "question": f'Can the statement "{n["text"]}" be {word} from the facts?',
                              "truth": not flip, "type": "derive", "depends_on": n["deps"], "flipped": flip,
                              "content": n["text"]})
        else:
            c["fld/unknown_items"] += 1
            ok = all(fld_nodes(p, r["context"], collections.Counter()) is not None for p in r["proofs"])
            c["fld/unknown_proof_structurally_ok"] += ok and len(r["proofs"]) > 0
            c["fld/unknown_no_proof"] += len(r["proofs"]) == 0
        out.append({"pool_id": f"fld_v2/{cfg}/{split}/{i}", "split": split, "context": "\n".join(sentences),
                    "statement": r["hypothesis"].strip(), "gold": label.lower(), "nodes": nodes,
                    "depth": int(r["depth"]) if r["depth"] == r["depth"] else 0,
                    "names": ["proved", "disproved", "unknown"],
                    "question": "Based only on the facts, is the hypothesis proved, disproved, or neither (unknown)?"})
    return out


# ----------------------------------------------------------------------------------------------------------------
# BoardgameQA
# ----------------------------------------------------------------------------------------------------------------

RE_BG_STEP = re.compile(r'^We know (?P<know>.+?), and according to Rule\d+ "(?P<rule>.*?)"(?P<mid>, and .*?)?, so we can conclude "(?P<concl>.*)"$', re.S)
RE_BG_FAIL = re.compile(r'we cannot prove the antecedent "(.*?)"(?= and for Rule\d+ we cannot|$)', re.S)


def bgqa_nodes(proof: str, c: collections.Counter) -> list[dict] | None:
    body, _, tail = proof.rpartition(' So the statement "')
    if not body or not re.match(r'^.*" is (proved|disproved) and the answer is "(yes|no)"\.$', tail, re.S):
        c["bgqa/proof_unparsed_tail"] += 1
        return None
    steps = re.split(r'(?<="\.) (?=We know )', body.strip())
    nodes, concl_id = [], {}
    for k, s in enumerate(steps):
        s = s.strip()
        if not s.endswith('".'):
            c["bgqa/proof_unparsed_step"] += 1
            return None
        m = RE_BG_STEP.match(s[:-1])
        if not m:
            c["bgqa/proof_unparsed_step"] += 1
            return None
        know, mid, concl = m.group("know"), m.group("mid") or "", m.group("concl")
        deps = []
        rest = know
        for prev in sorted(concl_id, key=len, reverse=True):
            if prev in rest:
                deps.append(concl_id[prev])
                rest = rest.replace(prev, "\x00")
        for piece in rest.split("\x00"):
            piece = re.sub(r"^[\s,]*(?:and\s+)?|[\s,]*(?:\band)?[\s,]*$", "", piece)
            if piece and piece != "and":
                nid = f"n{len(nodes) + 1}"
                nodes.append({"id": nid, "text": piece, "type": "parse", "truth": True, "deps": []})
                deps.append(nid)
        if mid and "we cannot prove the antecedent" in mid:
            seg = mid[mid.index("we cannot prove the antecedent"):]
            for a in RE_BG_FAIL.findall(seg):
                nid = f"n{len(nodes) + 1}"
                nodes.append({"id": nid, "text": a, "type": "fail", "truth": False, "deps": []})
                deps.append(nid)
        nid = f"n{len(nodes) + 1}"
        nodes.append({"id": nid, "text": concl, "type": "derive", "truth": True, "deps": deps, "last": k == len(steps) - 1})
        concl_id[concl] = nid
    return nodes


def build_bgqa(split: str, rng: random.Random, c: collections.Counter) -> list[dict]:
    import pandas as pd
    path = glob.glob(str(ROOT / f"data/raw/bench_math/boardgameqa/data/{split}-*.parquet"))[0]
    out = []
    for i, r in enumerate(pd.read_parquet(path).to_dict("records")):
        label = r["label"]
        nodes = []
        if label in ("proved", "disproved"):
            got = bgqa_nodes(r["proof"], c)
            if got is None:
                c[f"bgqa/drop_proof_{label}"] += 1
                continue
            stmt = re.search(r' So the statement "(.*)" is (?:proved|disproved) and', r["proof"], re.S).group(1)
            last = [n for n in got if n.get("last")][0]
            drop = {last["id"]}
            low = {stmt.lower(), last["text"].lower()}
            for n in got:
                if n["text"].lower() in low:
                    drop.add(n["id"])
            for n in got:
                if n["id"] in drop:
                    continue
                flip = n["type"] != "fail" and rng.random() < 0.5
                word = "disproved" if flip else "proved"
                nodes.append({"id": n["id"], "question": f'Can the statement "{n["text"]}" be {word} from the game state and the rules?',
                              "truth": n["truth"] and not flip, "type": n["type"],
                              "depends_on": [d for d in n["deps"] if d not in drop], "flipped": flip,
                              "content": n["text"]})
            c["bgqa/answer_nodes_dropped"] += len(drop)
        else:
            c["bgqa/unknown_items"] += 1
        m = re.match(r"^(.*?)\s*(Based on the game state and the rules and preferences, .*\?)\s*$", r["example"], re.S)
        if not m:
            c["bgqa/drop_question"] += 1
            continue
        out.append({"pool_id": f"boardgameqa/{split}/{i}", "split": split, "context": m.group(1).strip(),
                    "statement": None, "question": m.group(2).strip(), "gold": label, "nodes": nodes,
                    "depth": int(re.search(r"depth(\d+)", r["config"]).group(1)), "config": r["config"],
                    "names": ["proved", "disproved", "unknown"]})
    return out


# ----------------------------------------------------------------------------------------------------------------
# Items, decontamination, output
# ----------------------------------------------------------------------------------------------------------------

OPTION_TEXT = {"proved": "Proved", "disproved": "Disproved", "unknown": "Unknown"}
BG_OPTION_TEXT = {"proved": "Yes (it is proved)", "disproved": "No (it is disproved)", "unknown": "Unknown (neither can be shown)"}
PROV = {
    "fld": ("PERM", "Apache-2.0 (hitachi-nlp/FLD repo; card none)", "gold", "gold_proof"),
    "boardgameqa": ("ATTR", "CC BY 4.0 (tasksource/Boardgame-QA card)", "gold", "gold_proof"),
    "ruletaker": ("PERM", "Apache-2.0 (tasksource/ruletaker card)", "gold", "program_solver"),
    "proofwriter": ("UNVERIFIED", "unverified (lab only, plan §6.1.5)", "gold", "program_solver"),
}


def to_item(raw: dict, source: str, rng: random.Random) -> dict:
    names = list(raw["names"])
    rng.shuffle(names)
    labels = list(LETTERS[:len(names)])
    if source == "boardgameqa":
        opts = [BG_OPTION_TEXT[n] for n in names]
        prompt = f"{raw['context']}\n\nQuestion: {raw['question']}"
    elif source == "fld":
        opts = [OPTION_TEXT[n] for n in names]
        prompt = f"Facts:\n{raw['context']}\n\nHypothesis: {raw['statement']}\n\nQuestion: {raw['question']}"
    else:
        opts = names
        prompt = f"Context: {raw['context']}\n\nStatement: {raw['statement']}\n\nQuestion: {raw['question']}"
    prompt += "\nOptions:\n" + "\n".join(f"{L}) {o}" for L, o in zip(labels, opts))
    gold = labels[names.index(raw["gold"])]
    ids = {n["id"] for n in raw["nodes"]}
    subqs = [{"question": n["question"], "p_cot": 1.0 if n["truth"] else 0.0, "p_fresh": None, "truth": bool(n["truth"]),
              "type": n["type"], "depends_on": [d for d in n["depends_on"] if d in ids], "id": n["id"], "initial": None, "status": "direct",
              "source_tree": PROV[source][3], "flipped": n["flipped"]} for n in raw["nodes"]]
    depth = node_depths(subqs)
    for sq in subqs:
        sq["node_depth"] = depth[sq["id"]]
    lic = PROV[source]
    iid = f"{source}-{hashlib.sha1(raw['pool_id'].encode()).hexdigest()[:10]}"
    return {"item_id": iid, "source": f"goldproof/{source}", "family": source, "prompt": prompt, "labels": labels,
            "gold_label": gold, "teacher": {L: float(L == gold) for L in labels}, "depth": raw["depth"],
            "subqs": subqs, "has_tree": bool(subqs), "tree_fix": True, "orig_id": raw["pool_id"],
            "_content": raw["context"] + "\n" + (raw["statement"] or raw["question"]),
            "_node_content": [n["content"] for n in raw["nodes"]],
            "licence_class": lic[0], "licence": lic[1], "label_basis": lic[2], "node_basis": lic[3] if subqs else "none",
            **({"config": raw["config"]} if "config" in raw else {})}


DIGIT = re.compile(r"\d+")
WORD = re.compile(r"[a-z0-9]+")


def mtoks(text: str) -> list[str]:
    return WORD.findall(DIGIT.sub("0", text.lower()))


TOK13 = re.compile(r"[a-z0-9]+")


def ngrams(w: list[str], n: int) -> set[int]:
    return {hash(tuple(w[i:i + n])) for i in range(len(w) - n + 1)}


def eval_texts(pattern: str):
    for path in sorted(glob.glob(str(ROOT / pattern))):
        name = Path(path).stem
        for line in open(path):
            if line.strip():
                yield name, json.loads(line).get("prompt", "")


def eval_grams(pattern: str, tok, n: int, keep, boiler: int = 20) -> dict[int, str]:
    """n-grams of the eval prompts that are also in keep (sorted int64 array of the training candidates' n-grams) ->
    first eval file holding it, minus boilerplate (in >= boiler eval items). Restricting to keep bounds the memory."""
    import numpy as np
    df, owner = collections.Counter(), {}
    for name, text in eval_texts(pattern):
        g = np.fromiter(ngrams(tok(text), n), dtype=np.int64)
        if not len(g) or not len(keep):
            continue
        pos = np.minimum(np.searchsorted(keep, g), len(keep) - 1)
        hit = {int(h) for h in g[keep[pos] == g]}
        df.update(hit)
        for h in hit:
            owner.setdefault(h, name)
    return {h: owner[h] for h, k in df.items() if k < boiler}


class JaccardIndex:
    """Set-similarity join by prefix filtering: two sets with Jaccard >= t share a token among the first
    |W| - ceil(t|W|) + 1 tokens of each, tokens sorted by ascending global frequency."""

    def __init__(self, sets: list[tuple[str, frozenset]], t: float, extra: list[frozenset]):
        self.t, self.sets = t, sets
        self.freq = collections.Counter(tok for _, s in sets for tok in s)
        self.freq.update(tok for s in extra for tok in s)
        self.post = collections.defaultdict(list)
        for i, (_, s) in enumerate(sets):
            for tok in self.prefix(s):
                self.post[tok].append(i)

    def prefix(self, s: frozenset) -> list[str]:
        order = sorted(s, key=lambda x: (self.freq.get(x, 0), x))
        return order[:len(s) - math.ceil(self.t * len(s)) + 1]

    def hit(self, s: frozenset) -> tuple[str, float] | None:
        cand = {i for tok in self.prefix(s) for i in self.post.get(tok, ())}
        best = None
        for i in cand:
            name, e = self.sets[i]
            if min(len(e), len(s)) < self.t * max(len(e), len(s)):
                continue
            j = len(e & s) / len(e | s)
            if j >= self.t and (best is None or j > best[1]):
                best = (name, j)
        return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/student_goldproof")
    ap.add_argument("--tmp", default="/home/soray/.claude/jobs/3b5d797f/tmp/build")
    ap.add_argument("--fld-config", default="star", choices=["star", "data"],
                    help="star (default) or data (= FLD.v2/default, an S-fam family of v4_heldout)")
    ap.add_argument("--pool", default="data/raw/v4/tasksource_jev/data/*.parquet")
    ap.add_argument("--evals", default="data/eval/*.jsonl")
    ap.add_argument("--n-val", type=int, default=300, help="val items per source (FLD, BGQA)")
    ap.add_argument("--jaccard", type=float, default=0.5)
    ap.add_argument("--template-df", type=int, default=20,
                    help="n-grams in >= this many candidate items of one family are templates (0 = strict)")
    ap.add_argument("--headroom", type=float, default=1.3, help="train candidates per source = 4B size x this")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    c: collections.Counter = collections.Counter()
    out, tmp = ROOT / args.out, Path(args.tmp)
    out.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)

    # ---- raw items per source and split
    raw: dict[str, dict[str, list[dict]]] = {}
    fld_tr = build_fld(args.fld_config, "train", rng, c)
    fld_va = build_fld(args.fld_config, "validation", rng, c)
    raw["fld"] = {"train": fld_tr, "val": fld_va}
    raw["boardgameqa"] = {"train": build_bgqa("train", rng, c), "val": build_bgqa("valid", rng, c)}
    pool = pool_rows(args.pool)
    for src in ("ruletaker", "proofwriter"):
        items = build_rule_items(pool, src, rng, c)
        # pool train rows -> train; pool dev/test rows plus a hashed 5% of train -> val
        va = [x for x in items if x["split"] != "train" or int(hashlib.sha1(x["pool_id"].encode()).hexdigest(), 16) % 20 == 0]
        vids = {x["pool_id"] for x in va}
        tr = [x for x in items if x["pool_id"] not in vids]
        raw[src] = {"train": tr, "val": va}

    # ---- selection: label-balanced / config-stratified order, tier = smallest plan size that includes the row
    def order(items: list[dict], key) -> list[dict]:
        groups = collections.defaultdict(list)
        for x in items:
            groups[key(x)].append(x)
        for g in groups.values():
            rng.shuffle(g)
        ks = sorted(groups)
        res = []
        while any(groups.values()):
            for k in ks:
                if groups[k]:
                    res.append(groups[k].pop())
        return res

    sel = {}
    for src in raw:
        key = (lambda x: x["config"]) if src == "boardgameqa" else (lambda x: x["gold"])
        tiers = TIERS.get("ruletaker" if src == "proofwriter" else "bgqa" if src == "boardgameqa" else src)
        tr = order(raw[src]["train"], key)[:int(tiers[2] * args.headroom)]   # headroom for the decontam drops
        va = order(raw[src]["val"], key)[:args.n_val]
        sel[src] = {"train": tr, "val": va}

    items = {src: {sp: [to_item(x, src, rng) for x in sel[src][sp]] for sp in ("train", "val")} for src in sel}

    # ---- decontamination, on dataset-derived content only (context + statement, node statements; our wrapper text
    # is not evidence of leakage). (1) scripts/decontam.py (13-gram, eval-side boilerplate) on a temporary copy whose
    # prompt is that content: the strict verdict, reported. (2) Our gate: raw 13-gram and digit-masked 10-gram with the
    # same eval-side boilerplate rule plus the symmetric training-side rule (n-grams in >= --template-df candidate
    # items of the same family are generator templates; 0 = strict), and masked word-Jaccard >= --jaccard on the
    # prompt content. Rows hit by (2) are dropped.
    import numpy as np
    del raw, sel, pool, fld_tr, fld_va                # free the source rows before the n-gram passes
    allit = [it for src in items for sp in items[src] for it in items[src][sp]]
    pre = tmp / "goldproof_content.jsonl"
    with open(pre, "w") as f:
        for it in allit:
            f.write(json.dumps({"item_id": it["item_id"], "family": it["family"], "prompt": it["_content"]}) + "\n")
    log = subprocess.run([sys.executable, str(ROOT / "scripts/decontam.py"), str(pre), "--evals", args.evals],
                         capture_output=True, text=True, cwd=ROOT)
    print(log.stdout + log.stderr)
    strict13 = collections.Counter()
    for line in open(pre.with_suffix(".contaminated.jsonl")):
        r = json.loads(line)
        strict13[f"{r['family']}:{r['contaminated_with']}"] += 1
    raw13 = lambda t: TOK13.findall(t.lower())
    texts = lambda it: [it["_content"]] + it["_node_content"]
    toks = {"13": (raw13, 13), "m10": (mtoks, 10)}
    gm = {}
    for k, (tok, n) in toks.items():
        union = np.unique(np.concatenate([np.fromiter(ngrams(tok(t), n), dtype=np.int64) for it in allit for t in texts(it)]))
        gm[k] = eval_grams(args.evals, tok, n, union)
        del union
    fam_df = {k: collections.Counter() for k in gm}
    item_grams = {}
    for it in allit:
        item_grams[it["item_id"]] = {}
        for k, (tok, n) in toks.items():
            g = {h for t in texts(it) for h in ngrams(tok(t), n)} & gm[k].keys()     # only grams an eval item has
            item_grams[it["item_id"]][k] = g
            fam_df[k].update((it["family"], h) for h in g)
    sets = [(name, frozenset(mtoks(t))) for name, t in eval_texts(args.evals) if len(set(mtoks(t))) >= 5]
    jidx = JaccardIndex(sets, args.jaccard, [frozenset(mtoks(it["_content"])) for it in allit])
    hits, strict_hits, gate = collections.Counter(), collections.Counter(), {}
    for it in allit:
        why = strict = None
        for k in ("13", "m10"):
            for h in item_grams[it["item_id"]][k]:
                if h in gm[k]:
                    strict = strict or (k, gm[k][h])
                    if not args.template_df or fam_df[k][(it["family"], h)] < args.template_df:
                        why = (k, gm[k][h])
                        break
            if why:
                break
        if not why:
            j = jidx.hit(frozenset(mtoks(it["_content"])))
            if j:
                why = strict = ("jaccard", j[0])
        if strict:
            strict_hits[f"{it['family']}:{strict[0]}:{strict[1]}"] += 1
        if why:
            gate[it["item_id"]] = why
            hits[f"{it['family']}:{why[0]}:{why[1]}"] += 1

    # ---- write
    summary = {"args": vars(args), "counts": {}, "decontam_dropped": dict(hits.most_common()),
               "decontam_strict_no_template_rule": dict(strict_hits.most_common()),
               "decontam_py_13gram_on_content": dict(strict13.most_common()),
               "dropped_items": {k: list(v) for k, v in gate.items()}, "build": dict(sorted(c.items()))}
    files = {"train": open(out / "train.jsonl", "w"), "val": open(out / "val.jsonl", "w"),
             "train_pw": open(out / "train_proofwriter_lab.jsonl", "w"), "val_pw": open(out / "val_proofwriter_lab.jsonl", "w")}
    for src in items:
        tiers = TIERS.get("ruletaker" if src == "proofwriter" else "bgqa" if src == "boardgameqa" else src)
        for sp in ("train", "val"):
            kept = [it for it in items[src][sp] if it["item_id"] not in gate]
            dropped = len(items[src][sp]) - len(kept)
            if sp == "train":
                kept = kept[:tiers[2]]
                for k, it in enumerate(kept):
                    it["tier"] = "lab" if k < tiers[0] else "release" if k < tiers[1] else "4b"
            for it in kept:
                it["decontam"] = {"13gram": "clean", "masked10": "clean", "jaccard50": "clean",
                                  "template_df": args.template_df}
                it.pop("_content", None)
                it.pop("_node_content", None)
                files[sp + ("_pw" if src == "proofwriter" else "")].write(json.dumps(it) + "\n")
            nodes = [sq for it in kept for sq in it["subqs"]]
            summary["counts"][f"{src}/{sp}"] = {
                "items": len(kept), "decontam_dropped": dropped,
                "gold": dict(collections.Counter(it["gold_label"] for it in kept)),
                "with_nodes": sum(bool(it["subqs"]) for it in kept), "nodes": len(nodes),
                "node_yes_share": round(sum(sq["truth"] for sq in nodes) / max(1, len(nodes)), 3),
                "node_types": dict(collections.Counter(sq["type"] for sq in nodes)),
                "node_depth_max": dict(collections.Counter(max((sq["node_depth"] for sq in it["subqs"]), default=-1) for it in kept)),
                "tiers": dict(collections.Counter(it.get("tier") for it in kept)),
            }
    for f in files.values():
        f.close()
    summary["md5"] = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(out.glob("*.jsonl"))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
