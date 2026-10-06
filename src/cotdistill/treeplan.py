"""Tree-node planning for the full-tree arms (TF true parents vs TFM matched non-parents), torch-free so that data
checks (scripts/tree_balance.py) and tests run without a GPU stack. Re-exported by cotdistill.student."""
from __future__ import annotations

import random
import re


_KK_ONE_ROLE = re.compile(r"^(?:is|was) (\w+) (?:a|actually a|really a) (?:knight|knave)\??$", re.I)


def answer_component(sq: dict, item: dict) -> bool:
    """Whether a node asks for a component of the final answer: in Knights & Knaves, the role of one inhabitant
    (the options are full role assignments); in decision chains (source "chain"), the top check of either chain (the
    path top is the final's decision bit; the mirror top is marked too, so path and mirror arms drop one node each).
    Other families have no such nodes once the final question in disguise is filtered out
    (scripts/build_student_data.py, leaks_answer)."""
    src = str(item.get("source") or "")
    if src == "chain":
        return "on_path" in sq and sq.get("node_depth") == item.get("depth")
    return src == "knights_knaves" and bool(_KK_ONE_ROLE.match(" ".join(sq.get("question", "").split())))


def subq_keep(sq: dict, subq_filter: str = "all") -> bool:
    """--subq-filter: which tree nodes may be asked. "all" every node; "path" / "mirror" only the nodes of the chain the
    final reads / of the same-depth chain nothing reads (decision-chain data, on_path; nodes without it are dropped)."""
    if subq_filter == "all":
        return True
    if subq_filter not in ("path", "mirror"):
        raise ValueError(subq_filter)
    return sq.get("on_path") is (subq_filter == "path")


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


def full_tree_plan(item: dict, *, cap: int, rng: random.Random, subq_target: str = "cot",
                   subq_filter: str = "all") -> list[dict]:
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
    the original plan, so registered arms reproduce. subq_filter (subq_keep) restricts the asked nodes only; parents
    and the matched pool still come from every node of the item."""
    nodes = [sq for sq in item.get("subqs", []) if "id" in sq]
    asked = [sq for sq in nodes if _node_p(sq, subq_target) is not None]
    if subq_filter != "all":
        asked = [sq for sq in asked if subq_keep(sq, subq_filter)]
    if cap and len(asked) > cap:                # cap 0: no cap (every node)
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


# ---- Local views (Tree Forcing, --tree-view local / ancestors) ----------------------------------------------------
# A node asked WITHOUT the original problem: the item's preamble (K&K: the island rules and the head count, field
# local_preamble of data/student_v3sl) and either the statement line(s) the node reads (parse nodes, field local_lines)
# or the results of the nodes it depends on (its parents, or every ancestor), never the problem's other text or the
# final question's options.
LOCAL_HEADER = "Known results:"
_KK_ROLE_RESULT = re.compile(r"^Is ([A-Z][a-z]+) a (knight|knave)\?$")


def ancestors(nodes: list[dict]) -> dict[str, set]:
    """Transitive dependencies of every node (ids of the item; unknown ids ignored)."""
    by = {sq["id"]: sq for sq in nodes if "id" in sq}
    out: dict[str, set] = {}
    for i in by:
        seen, todo = set(), [x for x in (by[i].get("depends_on") or []) if x in by]
        while todo:
            j = todo.pop()
            if j not in seen:
                seen.add(j)
                todo += [x for x in (by[j].get("depends_on") or []) if x in by]
        out[i] = seen
    return out


def is_local_leaf(sq: dict) -> bool:
    """Nodes that read the problem text (local view: their statement lines) rather than other nodes' results."""
    return bool(sq.get("local_lines"))


def local_context(item: dict, sq: dict, stated: list[dict] | None, *, depth: dict | None = None,
                  rng=None, order: str = "shuffle") -> str:
    """The text above "Intermediate question:" in a local view: preamble, then the node's statement line(s) (parse
    nodes; `stated` is ignored) or "Known results:" with one "Q: ... A: Yes/No" line per stated node (_fact_line:
    program truth, else a confident CoT answer). order "shuffle" draws the line order from rng (local view, as the
    problem-view tree arms do), "depth" lists shallow to deep, ties in node order (ancestors view)."""
    pre = item.get("local_preamble")
    if pre is None:
        raise ValueError(f"{item.get('item_id')}: no local_preamble (local views need e.g. data/student_v3sl)")
    if is_local_leaf(sq):
        return pre + "\n\n" + "\n".join(sq["local_lines"])
    if not stated:                  # TFLN (--tree-full plain --tree-view local): derived nodes, nothing stated
        return pre
    if order == "depth":
        pos = {x["id"]: k for k, x in enumerate(item.get("subqs", [])) if "id" in x}
        stated = sorted(stated, key=lambda x: (depth[x["id"]], pos[x["id"]]))
    lines = [_fact_line(x) for x in stated]
    if order == "shuffle":
        rng.shuffle(lines)
    return pre + f"\n\n{LOCAL_HEADER}\n" + "\n".join(lines)


_KK_STATEMENT = re.compile(r'^- [A-Z][a-z]+ says: ".*"$')


def statement_block(item: dict) -> str:
    """The problem WITHOUT its final question and options (--tree-view statements, GFP-S): the context of a GFP view
    minus the final question's text and its answer options.
    - knights_knaves: the prompt before "\\n\\nQuestion: ", i.e. the island preamble and the "- X says: ..." lines;
    - numgen (data/student_numgen*): the prompt's first line without the "Question: " label and without the answer
      quantity's question sentence (program.quantities[answer].question, the last sentence), i.e. the frame intro, the
      operation sentences and the noise sentences in prompt order. The options follow on later lines and are dropped.
    Any other source, or a prompt that does not have this shape, raises ValueError (no clean definition)."""
    src, p, iid = item.get("source"), item.get("prompt", ""), item.get("item_id")
    if src == "knights_knaves":
        head, sep, tail = p.partition("\n\nQuestion: ")
        pre, _, body = head.partition("\n\n")
        lines = body.split("\n")
        if (not sep or "\n\nQuestion: " in tail or not body or not all(_KK_STATEMENT.match(x) for x in lines)
                or item.get("local_preamble") not in (None, pre)):
            raise ValueError(f"{iid}: K&K prompt is not preamble + statement lines + Question")
        return head
    if src == "numgen":
        first, sep, _opts = p.partition("\nOptions:\n")
        prog = item.get("program") or {}
        ans = [q for q in prog.get("quantities", []) if q.get("id") == prog.get("answer")]
        ask = ans[0].get("question") if len(ans) == 1 else None
        if not (sep and ask and first.startswith("Question: ") and first.endswith(" " + ask) and "\n" not in first):
            raise ValueError(f"{iid}: numgen prompt is not 'Question: <body> <answer question>' + options")
        return first[len("Question: "):-len(ask) - 1]
    raise ValueError(f"{iid}: --tree-view statements has no statement block for source {src!r} "
                     "(defined for knights_knaves and numgen; restrict with --tree-sources)")


def kk_people(item: dict) -> list[str]:
    return re.findall(r"^- ([A-Z][a-z]+) says: ", item.get("prompt", ""), re.M)


def view_leak(item: dict, sq: dict, stated: list[dict]) -> str | None:
    """Why a node view would state (part of) the final answer, else None: "question" / "stated" when leaks_final flags
    the asked question / a stated result's question (the final question in disguise), "assignment" when the stated
    unconditional role results name a role for every inhabitant (K&K: the full answer stated as results)."""
    from .teacher import leaks_final       # lazy: stdlib only, but not needed by the default arms
    kk = item.get("source") == "knights_knaves"
    people = kk_people(item) if kk else []
    rec = {"domain": "knights_knaves", "meta": {"people": people}, "prompt": item.get("prompt", "")} if kk else item
    if leaks_final(sq["question"], rec):
        return "question"
    if any(leaks_final(x["question"], rec) for x in stated):
        return "stated"
    if kk and people:
        named = {m.group(1) for m in (_KK_ROLE_RESULT.match(x["question"]) for x in stated) if m}
        if named >= set(people):
            return "assignment"
    return None
