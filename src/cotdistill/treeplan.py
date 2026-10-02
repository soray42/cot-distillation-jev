"""Tree-node planning for the full-tree arms (TF true parents vs TFM matched non-parents), torch-free so that data
checks (scripts/tree_balance.py) and tests run without a GPU stack. Re-exported by cotdistill.student."""
from __future__ import annotations

import random
import re


_KK_ONE_ROLE = re.compile(r"^(?:is|was) (\w+) (?:a|actually a|really a) (?:knight|knave)\??$", re.I)


def answer_component(sq: dict, item: dict) -> bool:
    """Whether a node asks for a component of the final answer: in Knights & Knaves, the role of one inhabitant
    (the options are full role assignments). Other families have no such nodes once the final question in disguise
    is filtered out (scripts/build_student_data.py, leaks_answer)."""
    return (str(item.get("source") or "") == "knights_knaves"
            and bool(_KK_ONE_ROLE.match(" ".join(sq.get("question", "").split()))))


def node_depths(nodes: list[dict]) -> dict[str, int]:
    """Depth of each tree node: 0 for nodes that depend on no other node (facts read off the problem), else one
    more than the deepest node it depends on. Unknown ids and cycles count as depth 0."""
    by = {n["id"]: n for n in nodes if "id" in n}
    memo: dict[str, int] = {}

    def depth(i: str, seen: frozenset) -> int:
        if i in memo:
            return memo[i]
        if i in seen or i not in by:
            return 0
        deps = [x for x in (by[i].get("depends_on") or []) if x in by]
        memo[i] = 1 + max(depth(x, seen | {i}) for x in deps) if deps else 0
        return memo[i]
    return {i: depth(i, frozenset()) for i in by}


def _node_p(sq: dict, subq_target: str = "cot") -> float | None:
    if subq_target == "truth" and sq.get("truth") is not None:
        return 1.0 if sq["truth"] else 0.0
    p = sq.get("p_cot") if subq_target != "fresh" else sq.get("p_fresh")
    return sq.get("p_fresh") if p is None else p


def _fact_line(sq: dict) -> str | None:
    p = (1.0 if sq["truth"] else 0.0) if sq.get("truth") is not None else sq.get("p_cot")
    if p is None or 0.3 < p < 0.7:
        return None
    return f"Q: {sq['question']} A: {'Yes' if p >= 0.5 else 'No'}"


def _descendants(nodes: list[dict]) -> dict[str, set]:
    by = {sq["id"] for sq in nodes}
    children = {i: [sq["id"] for sq in nodes if i in (sq.get("depends_on") or [])] for i in by}
    out = {}
    for i in by:
        seen, todo = set(), list(children[i])
        while todo:
            j = todo.pop()
            if j not in seen:
                seen.add(j)
                todo += children[j]
        out[i] = seen
    return out


def full_tree_plan(item: dict, *, cap: int, rng: random.Random, subq_target: str = "cot") -> list[dict]:
    """The nodes of a full-tree group (at most `cap`, sampled if the tree is larger) with both fact sets:
    "parents" (true parent results) and "matched" (as many results from other nodes of the same item, depth-matched
    where possible, excluding the node, its parents, its descendants and the near-answer nodes: sinks at the maximum
    depth). A node gets facts in both modes only if it is eligible: it has parents, every parent has a confident
    answer, and enough matched replacements exist. Otherwise it is asked plainly in every mode, so the two tree
    modes differ only in which results are stated, never in how many.
    Items built after the tree audit (item["tree_fix"]) also exclude the node's ancestors from the pool (audit F3:
    42% of matched results were ancestors), replace each parent only by a node of the same answer-component class
    (no such node: the view is not eligible), the nearest in depth (node_depth, on the graph before filtering), then
    with the same stated answer, ties at random (F4), and never list a node as its own parent (F12). Older data keeps
    the original plan, so registered arms reproduce."""
    nodes = [sq for sq in item.get("subqs", []) if "id" in sq]
    asked = [sq for sq in nodes if _node_p(sq, subq_target) is not None]
    if len(asked) > cap:
        asked = rng.sample(asked, cap)
    if not nodes:
        return []
    fix = bool(item.get("tree_fix"))
    by = {sq["id"]: sq for sq in nodes}
    depth = ({i: by[i]["node_depth"] for i in by} if fix and all("node_depth" in sq for sq in nodes)
             else node_depths(nodes))
    desc = _descendants(nodes)
    maxd = max(depth.values())
    sinks = {i for i in by if not any(i in (sq.get("depends_on") or []) for sq in nodes)}
    confident = {i for i in by if _fact_line(by[i])}
    comp = {i: answer_component(by[i], item) for i in by} if fix else {}
    yes = {i: _fact_line(by[i]).endswith("A: Yes") for i in confident} if fix else {}
    plan = []
    for sq in asked:
        i = sq["id"]
        par = [x for x in (sq.get("depends_on") or []) if x in by and not (fix and x == i)]
        entry = {"node": sq, "parents": [], "matched": []}
        if par and all(x in confident for x in par):
            pool = [j for j in by if j in confident and j != i and j not in par and j not in desc[i]
                    and not (j in sinks and depth[j] == maxd) and not (fix and i in desc[j])]
            if len(pool) >= len(par):
                rest, matched = pool[:], []
                for x in par:
                    if fix:     # the same answer-component class (required), then the nearest depth, then the same
                        cand = [j for j in rest if comp[j] == comp[x]]          # stated answer; ties at random
                        if not cand:
                            break
                        key = {j: (abs(depth[j] - depth[x]), yes[j] != yes[x]) for j in cand}
                        j = rng.choice([j for j in cand if key[j] == min(key.values())])
                    else:
                        same = [j for j in rest if depth[j] == depth[x]]
                        cand = same or sorted(rest, key=lambda j: abs(depth[j] - depth[x]))[:max(1, len(rest))]
                        j = rng.choice(same) if same else cand[0]
                    matched.append(j)
                    rest.remove(j)
                if len(matched) == len(par):       # else (no same-class replacement) asked plainly in both arms
                    entry = {"node": sq, "parents": par, "matched": matched}
        plan.append(entry)
    return plan
