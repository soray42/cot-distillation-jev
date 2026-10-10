"""Local-view fields for Tree Forcing on Knights & Knaves: data/student_v3s plus, for every K&K item, the text a node
sees in a local view (--tree-view local/ancestors in scripts/train_student.py), and a sufficiency audit of the cleaned
proof graph.

Added fields (nothing else changes: finals, other families, ids, node order and every existing field are kept; the
script checks that removing the new keys gives back the v3s records exactly):
  item["local_preamble"]       the prompt text before the first statement line (the island rules and the head count)
  subq["local_lines"]          parse nodes only: the speaker's statement line as the prompt renders it, e.g.
                               '- Frank says: "Alice is a knight."' (a parse node reads exactly one statement)
  subq["local_trivial"]        derived nodes only: True when the node's answer holds in every role assignment, i.e. the
                               question's own hypothesis decides it and no stated result is needed (the local-step
                               probe reports these apart)

Sufficiency audit: every K&K puzzle is re-solved with the solver that built the trees
(teacher_cache/solver_kk_v3f/build_reasoning_trees.py, deterministic; its node list must equal the stored tree), which
gives each raw node's proposition as a set of role assignments (the validate_proof masks). For every kept non-parse node
the conjunction of its CLEANED parents (build_solver_kk.py: generic rule nodes and verify nodes dropped, repeated
questions merged into their first occurrence) must entail the node over all role assignments, i.e. the node follows from
its stated parents (plus the preamble's rules, which hold in every assignment) alone. Wording flips do not matter: a
flipped node states the same proposition with the opposite answer. The script counts the nodes that lost or had a parent
remapped in cleaning, by cause, and exits non-zero if any node is not entailed.

  python scripts/build_local_kk.py --base data/student_v3s --out data/student_v3sl
"""
from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import importlib.util
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERIC = ("Is each of the ", "Must a knight's statement", "Must a knave's statement")   # as build_solver_kk.py
_PARSE_Q = re.compile(r'^Is ([A-Z][a-z]+) a (?:knight|knave) exactly when "(.+)" is true\?$')
_STMT_LINE = re.compile(r'^- ([A-Z][a-z]+) says: "([^"\n]+)"$', re.M)          # as Puzzle.read
NEW_ITEM_KEYS, NEW_SUBQ_KEYS = ("local_preamble",), ("local_lines", "local_trivial")


def load_solver(path: Path):
    spec = importlib.util.spec_from_file_location("kk_solver", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kk_solver"] = mod          # dataclasses resolve the module by name
    spec.loader.exec_module(mod)
    return mod


def preamble(prompt: str) -> str:
    m = _STMT_LINE.search(prompt)
    assert m, "no statement line"
    return prompt[:m.start()].rstrip()


def statement_line(prompt: str, question: str) -> str:
    """The prompt line a parse node reads: its speaker's statement, rendered exactly as in the prompt."""
    m = _PARSE_Q.match(question)
    assert m, question
    lines = [x for x in _STMT_LINE.finditer(prompt) if x.group(1) == m.group(1)]
    assert len(lines) == 1 and lines[0].group(2) == m.group(2), (question, [x.group(0) for x in lines])
    return lines[0].group(0)


def masks(solver, builder) -> tuple[dict, int]:
    """Raw node id -> bit mask of the role assignments where the node's proposition (with its answer) holds; the
    mask logic of validate_proof in the solver module."""
    puzzle = builder.puzzle
    assignments = [dict(zip(puzzle.names, bits)) for bits in itertools.product((False, True), repeat=len(puzzle.names))]
    universe = (1 << len(assignments)) - 1
    out = {}
    for node in builder.nodes:
        pred = builder.predicates[node["id"]]
        if pred[0] == "domain":
            mask = universe
        elif pred[0] in {"parse", "verify"}:
            sp = pred[1]
            mask = sum(1 << i for i, r in enumerate(assignments) if r[sp] == puzzle.expressions[sp].evaluate(r))
        else:
            scope = pred[1]

            def holds(r, pred=pred, scope=scope):
                if any(r[n] != v for n, v in scope):
                    return True
                if pred[0] == "conflict":
                    return False
                _, _, (channel, name), value = pred
                actual = r[name] if channel in {"role", "requirement"} else puzzle.expressions[name].evaluate(r)
                return actual == value
            mask = sum(1 << i for i, r in enumerate(assignments) if holds(r))
        out[node["id"]] = mask
    return out, universe


def cleaning_map(raw: list[dict]) -> tuple[dict, set, set]:
    """The cleaning of build_solver_kk.clean, replayed: alias (merged duplicate -> first occurrence), the dropped
    generic ids and the dropped verify ids."""
    alias, first, generic, verify = {}, {}, set(), set()
    for n in raw:
        if n["question"].startswith(GENERIC):
            generic.add(n["id"])
        elif n["type"] == "verify":
            verify.add(n["id"])
        elif n["question"] in first:
            alias[n["id"]] = first[n["question"]]
        else:
            first[n["question"]] = n["id"]
    return alias, generic, verify


def audit_item(solver, it: dict, trees: Path, c: collections.Counter, bad: list) -> dict:
    """Sufficiency of the cleaned parents (counted into c, failures into bad); returns {derived node id: trivial}."""
    puzzle = solver.Puzzle.read({"item_id": it["item_id"], "prompt": it["prompt"]}, "v3s")
    builder = solver.ProofBuilder(puzzle)
    builder.solve()
    raw = json.load(open(trees / f"trees/{it['item_id']}.json"))["nodes"]
    assert [(n["id"], n["question"], n["depends_on"]) for n in builder.nodes] == \
        [(n["id"], n["question"], n["depends_on"]) for n in raw], f"{it['item_id']}: re-solved tree differs"
    return audit_nodes(solver, builder, raw, it, c, bad)


def audit_nodes(solver, builder, raw: list[dict], it: dict, c: collections.Counter, bad: list) -> dict:
    """audit_item's check for a solved builder (its proof's raw node list `raw`) and the item's cleaned nodes
    (it["subqs"]); also used by scripts/build_kk_tf.py on puzzles solved in memory."""
    m, universe = masks(solver, builder)
    alias, generic, verify = cleaning_map(raw)
    rawby = {n["id"]: n for n in raw}
    trivial = {}
    for sq in it["subqs"]:
        if sq["type"] == "parse":
            continue
        c["derived_nodes"] += 1
        trivial[sq["id"]] = m[sq["id"]] == universe
        c["trivial"] += trivial[sq["id"]]
        rd = rawby[sq["id"]]["depends_on"]
        lost = collections.Counter("generic" if d in generic else "verify" if d in verify else "merged" if d in alias
                                   else "kept" if d in sq["depends_on"] else "other" for d in rd)
        for k in ("generic", "verify", "merged", "other"):
            if lost[k]:
                c[f"nodes_with_parent_{k}"] += 1
                c[f"parents_{k}"] += lost[k]
        if lost["generic"] or lost["verify"] or lost["other"]:
            c["nodes_lost_a_parent"] += 1
        prem = universe
        for d in sq["depends_on"]:
            prem &= m[d]
        if prem & (universe ^ m[sq["id"]]):
            c["not_entailed"] += 1
            bad.append((it["item_id"], sq["id"], sq["question"]))
        else:
            c["entailed"] += 1
        # the same check on the raw parents (sanity: the solver's own validation)
        rprem = universe
        for d in rd:
            rprem &= m[d]
        c["raw_entailed"] += not (rprem & (universe ^ m[sq["id"]]))
        if lost["generic"] or lost["verify"] or lost["other"]:
            c["lost_parent_still_entailed"] += not (prem & (universe ^ m[sq["id"]]))
    c["kk_items"] += 1
    return trivial


def audit_views(solver, data: Path, trees: Path, seeds: int = 1) -> dict:
    """Information content of the training views, by the same masks: for every derived node of every K&K item in
    data/{train,val}.jsonl, whether its answer follows from nothing (the question's own hypothesis decides it), from its
    parents (TFL; must be every node), and, for the views where the matched control states non-parent results (TFLM,
    full_tree_plan with cap 0, seeds 0..seeds-1), whether those results happen to entail it anyway."""
    import random
    sys.path.insert(0, str(ROOT / "src"))
    from cotdistill.treeplan import full_tree_plan
    c = collections.Counter()
    for split in ("train", "val"):
        for line in open(data / f"{split}.jsonl"):
            it = json.loads(line)
            if it.get("source") != "knights_knaves":
                continue
            puzzle = solver.Puzzle.read({"item_id": it["item_id"], "prompt": it["prompt"]}, "v3s")
            builder = solver.ProofBuilder(puzzle)
            builder.solve()
            m, universe = masks(solver, builder)
            ent = lambda ids, i: not (functools.reduce(lambda a, b: a & b, (m[j] for j in ids), universe) & (universe ^ m[i]))
            for sq in it["subqs"]:
                if sq["type"] == "parse":
                    continue
                c["derived"] += 1
                c["trivial_no_results"] += ent([], sq["id"])
                c["parents_entail"] += ent(sq["depends_on"], sq["id"])
            for seed in range(seeds):
                for e in full_tree_plan(it, cap=0, rng=random.Random(seed)):
                    if e["parents"]:
                        c["matched_views"] += 1
                        i = e["node"]["id"]
                        c["matched_entail"] += ent(e["matched"], i)
                        c["matched_entail_nontrivial"] += ent(e["matched"], i) and not ent([], i)
                        c["trivial_among_matched_views"] += ent([], i)
                    elif e["node"]["type"] != "parse":
                        c["matched_frozen_views"] += 1
    return dict(sorted(c.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/student_v3s")
    ap.add_argument("--out", default="data/student_v3sl")
    ap.add_argument("--trees", default="teacher_cache/solver_kk_v3f/reasoning_trees")
    ap.add_argument("--solver", default="teacher_cache/solver_kk_v3f/build_reasoning_trees.py")
    ap.add_argument("--audit-views", action="store_true", help="only audit the views of an existing --out (TFL "
                    "sufficiency, trivial nodes, accidental sufficiency of the TFLM matched results) into "
                    "<out>/view_audit.json")
    args = ap.parse_args()
    solver = load_solver(ROOT / args.solver)
    trees = ROOT / args.trees
    if args.audit_views:
        a = audit_views(solver, ROOT / args.out, trees)
        (ROOT / args.out / "view_audit.json").write_text(json.dumps(a, indent=1))
        print(json.dumps(a, indent=1))
        return
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    c, bad, summary = collections.Counter(), [], {}
    for split in ("train", "val"):
        n_kk = 0
        h_base, h_strip = hashlib.md5(), hashlib.md5()
        with open(out / f"{split}.jsonl", "w") as f:
            for line in open(ROOT / args.base / f"{split}.jsonl"):
                it = json.loads(line)
                if it.get("source") == "knights_knaves":
                    trivial = audit_item(solver, it, trees, c, bad)
                    new = dict(it, local_preamble=preamble(it["prompt"]))
                    new["subqs"] = [dict(sq, local_lines=[statement_line(it["prompt"], sq["question"])])
                                    if sq["type"] == "parse" else dict(sq, local_trivial=trivial[sq["id"]])
                                    for sq in it["subqs"]]
                    for sq in new["subqs"]:
                        if sq["type"] == "parse":
                            c["parse_nodes"] += 1
                            c["parse_flipped"] += " a knave exactly when " in sq["question"]
                    n_kk += 1
                    it2 = new
                else:
                    it2 = it
                s = json.dumps(it2) + "\n"
                f.write(s)
                # byte check: dropping the new keys gives back the base record
                back = {k: v for k, v in it2.items() if k not in NEW_ITEM_KEYS}
                if "subqs" in back:
                    back["subqs"] = [{k: v for k, v in sq.items() if k not in NEW_SUBQ_KEYS} for sq in back["subqs"]]
                assert json.dumps(back) == json.dumps(it), it["item_id"]
                h_base.update(line.rstrip("\n").encode())
                h_strip.update(json.dumps(back).encode())
        summary[split] = {"kk_items": n_kk, "md5": hashlib.md5((out / f"{split}.jsonl").read_bytes()).hexdigest(),
                          "base_equal_after_strip": h_base.hexdigest() == h_strip.hexdigest()}
        assert summary[split]["base_equal_after_strip"], f"{split}: base lines are not json.dumps round trips"
    summary.update({"sufficiency": dict(sorted(c.items())), "not_entailed_examples": bad[:20], "base": args.base,
                    "added_item_fields": list(NEW_ITEM_KEYS), "added_subq_fields": list(NEW_SUBQ_KEYS)})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if bad:
        raise SystemExit(f"{len(bad)} derived nodes are not entailed by their cleaned parents")


if __name__ == "__main__":
    main()
