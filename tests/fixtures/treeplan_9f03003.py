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


# ---- multiple-choice tree nodes (opt-in data; e.g. scripts/build_fa_sf.py --trees) -------------------------------------
# A node that carries its own "options" (option texts) and "answer" (the gold index) is asked like a final question:
# its options under the letters MC_SYMBOLS[:k], read out at "Answer:" over those letters, one-hot target on the answer
# (label smoothing, if any, comes from the loss as for the final question). The trigger is the "options" key; no node
# of the existing data carries one, so every default path is unchanged. MC nodes are read by the full-tree arms in the
# problem / statements / statements+question views (GF, GFP-S, GFPSP, ...) and by evaluate_subq; they never state a
# result (_fact_line is None for them) and the local Tree Forcing views refuse them.
MC_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"   # fa_public.SYMBOLS (Intern-Decision's)


def is_mc_node(sq: dict) -> bool:
    """Whether a tree node is a multiple-choice node (it has an "options" key); mc_check validates it."""
    return isinstance(sq, dict) and "options" in sq


def mc_check(sq: dict) -> None:
    """ValueError unless the MC node has 2..62 distinct one-line option texts and an int answer index among them."""
    o, a = sq.get("options"), sq.get("answer")
    ok = (isinstance(o, list) and 2 <= len(o) <= len(MC_SYMBOLS) and all(isinstance(t, str) and t.strip()
                                                                          and "\n" not in t for t in o)
          and len(set(o)) == len(o) and isinstance(a, int) and not isinstance(a, bool) and 0 <= a < len(o))
    if not ok or not str(sq.get("question") or "").strip() or "\n" in sq["question"]:
        raise ValueError(f"MC node {sq.get('id')!r}: needs a one-line question, 2-{len(MC_SYMBOLS)} distinct one-line "
                         f"options and an answer index (got options {o!r}, answer {a!r})")


def mc_labels(sq: dict) -> list[str]:
    mc_check(sq)
    return list(MC_SYMBOLS[:len(sq["options"])])


def mc_options_block(sq: dict) -> str:
    """The node's 'X) text' option lines (the layout of a final question's options)."""
    return "\n".join(f"{L}) {t}" for L, t in zip(mc_labels(sq), sq["options"]))


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
                   subq_filter: str = "all", node_types: set | frozenset | None = None,
                   stats: dict | None = None) -> list[dict]:
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
    and the matched pool still come from every node of the item. node_types (None = every node; --tree-node-types)
    likewise keeps only the asked nodes whose "type" is listed (a node without one is dropped), before the cap; depths,
    parents and the matched pool are untouched (a dropped parent is simply not asked). With node_types, stats counts
    the asked nodes it drops (node_type_dropped); without, stats is not touched. Multiple-choice nodes (is_mc_node) are
    asked under every subq_target (their target is their answer index); they are never confident, so never stated."""
    nodes = [sq for sq in item.get("subqs", []) if "id" in sq]
    asked = [sq for sq in nodes if _node_p(sq, subq_target) is not None or is_mc_node(sq)]   # MC: its own answer
    if subq_filter != "all":
        asked = [sq for sq in asked if subq_keep(sq, subq_filter)]
    if node_types is not None:
        kept = [sq for sq in asked if sq.get("type") in node_types]
        if stats is not None:
            stats["node_type_dropped"] = stats.get("node_type_dropped", 0) + len(asked) - len(kept)
        asked = kept
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
_AGENT_ASK = r"Question: What should the agent do\?"
# Sources whose prompt is "<statements>\n\n<final question>\nOptions:\n<options>": the final question is the last
# paragraph before the options, in the one form its generator writes (a BoardgameQA question names the claim it asks
# about, as numgen's answer question names the quantity; the judged statement / hypothesis of JustLogic, FOLIO, FLD
# and RuleTaker is a paragraph of the problem and stays). Not here, no clean cut: arc_challenge / arc_easy (the
# question is the whole problem) and gsm8k_soc (one free-text paragraph whose last sentence may carry premises; its
# Tree Forcing build, source gsm8k_tf, has its own branch: GFPQ on every item, GFP-S only on bare questions, GSM8K_TF).
FINAL_QUESTION = {
    "justlogic": r"Question: Is the statement true, false, or uncertain given the passage\?",
    "folio": r"Question: Is the statement true, false, or uncertain given the premises\?",
    "sharc": r"Question: Based only on the rule text, the scenario and the earlier answers, how should the user's "
             r"question be answered\?",
    "returns": _AGENT_ASK, "expense": _AGENT_ASK, "subscription": _AGENT_ASK,             # policygen.py
    "rule": r"(?:Question: )?What should the [a-z][a-z' -]* do\?",                       # rulegen.py: list / bullets, prose
    "goldproof/fld": r"Question: Based only on the facts, is the hypothesis proved, disproved, or neither \(unknown\)\?",
    "goldproof/boardgameqa": r"Question: Based on the game state and the rules and preferences, [^\n]+\?",
    "goldproof/ruletaker": r"Question: Does the statement follow from the context\? What is not explicitly stated as "
                           r"true is considered false\.",
    "pronto": r"Question: True or false: [A-Z][a-z]+ is (?:not )?[a-z]+\.",       # PrOntoQA-OOD (pronto.py): the query
}


def statement_block(item: dict) -> str:
    """The problem WITHOUT its final question and options (--tree-view statements, GFP-S): the context of a GFP view
    minus the final question's text and its answer options.
    - knights_knaves: the prompt before "\\n\\nQuestion: ", i.e. the island preamble and the "- X says: ..." lines;
    - numgen (data/student_numgen*): the prompt's first line without the "Question: " label and without the answer
      quantity's question sentence (program.quantities[answer].question, the last sentence), i.e. the frame intro, the
      operation sentences and the noise sentences in prompt order. The options follow on later lines and are dropped.
    - fa_tier0 (data/student_fa0, failure attribution): the log (header, task, optional reference answer, the steps)
      before "\n\nQuestion: ", i.e. without the attribution question and its step / agent options.
    - fa_public (data/student_fa1, public failure-attribution logs): the log before the step or agent question (exact
      text, the last occurrence) and its options; the log may contain "Question:" (tasks of MC benchmarks) but never a
      line opening with "Options:" (fa_public.sanitize).
    - the sources of FINAL_QUESTION: the prompt before its last paragraph, which must be that source's final question
      (the only "Options:" block follows it; no earlier paragraph opens with "Question: ").
    - gsm8k_tf (data/student_gsm8k_tf, GSM8K with Socratic step nodes): the problem before its last sentence, only when
      that sentence is a bare question (gsm8k_question_cut); a question sentence that carries premises ("If she sells
      3 more, how many ...?", about 60% of GSM8K train) raises ValueError, as gsm8k_soc does for every item.
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
    if src == "fa_tier0":
        head, sep, tail = p.rpartition("\n\nQuestion: ")
        if (not sep or "\nOptions:\n" not in tail or "Question:" in head or "\nOptions:" in head
                or not head.startswith(_FA_HEADER)):
            raise ValueError(f"{iid}: fa_tier0 prompt is not log + Question + options")
        return head
    if src == "fa_public":
        for q in (_FA_Q_STEP, _FA_Q_AGENT):
            head, sep, tail = p.rpartition("\n\n" + q + "\nOptions:\n")
            if sep:
                break
        if not sep or not tail or "\n\n" in tail or "\nOptions:" in head or not head.startswith(_FA_HEADER):
            raise ValueError(f"{iid}: fa_public prompt is not log + the step or agent question + options")
        return head
    if src == GSM8K_TF:
        cut = gsm8k_question_cut(_gsm8k_tf_problem(item))
        if cut is None:
            raise ValueError(f"{iid}: no statement block: the last sentence of this gsm8k_tf problem is not a bare "
                             "question (it carries a number, a condition or a clause; GFPQ uses problem_no_options)")
        return cut[0]
    if src in FINAL_QUESTION:
        head, sep, opts = p.partition("\nOptions:\n")
        body, cut, last = head.rpartition("\n\n")
        if (not sep or "\nOptions:" in opts or not cut or not body.strip() or not re.fullmatch(FINAL_QUESTION[src], last)
                or "\n\nQuestion: " in body or body.startswith("Question: ")):
            raise ValueError(f"{iid}: no statement block: the {src} prompt is not <statements> + its final question "
                             "paragraph + options")
        return body
    raise ValueError(f"{iid}: --tree-view statements has no statement block for source {src!r} "
                     "(defined for knights_knaves, numgen, fa_tier0, fa_public and treeplan.FINAL_QUESTION; restrict "
                     "with --tree-sources)")


_FA_HEADER = "The log below comes from a multi-agent system that failed to solve its task.\n\nTask: "
_FA_Q_STEP = ("Question: Which step contains the decisive error, i.e. the earliest mistake whose correction would have "
              "let the system solve the task?")
_FA_Q_AGENT = "Question: Which agent made the decisive error that caused the failure?"

# GSM8K with Tree Forcing nodes (scripts/build_gsm8k_tf.py): "Question: <problem>\nOptions:\n<numbers>", the problem
# one line of free text whose last sentence asks for the answer and often carries premises, so GFPQ
# (problem_no_options) is defined for every item and GFP-S (statement_block) only where that sentence is a bare
# question: it ends with "?", opens with a wh-word other than "When", and holds no digit, number word, comma, colon,
# semicolon, %, $ or condition / time-clause word; the text before it is one or more sentences without a "?".
GSM8K_TF = "gsm8k_tf"
_GSM_CUT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_GSM_ASK = re.compile(r"^(?:How|What|Which|Who|Whom|Whose|Where|Why)\b")
_GSM_NUMWORD = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|"
    r"billion|half|halves|twice|double|doubled|triple|tripled|thrice|third|thirds|quarter|quarters|fourth|fifth|dozen|"
    r"dozens|percent|once)\b", re.I)
_GSM_PREMISE = re.compile(
    r"\b(?:if|when|whenever|after|before|assuming|assume|given|suppose|supposing|since|because|including|excluding|"
    r"except|without|unless|while|until|now|then|instead)\b", re.I)


def _gsm8k_tf_problem(item: dict) -> str:
    """The problem text of a gsm8k_tf prompt ("Question: <one line>\nOptions:\n<options>"), else ValueError."""
    p, iid = item.get("prompt", ""), item.get("item_id")
    head, sep, opts = p.partition("\nOptions:\n")
    body = head[len("Question: "):]
    if (not sep or "\nOptions:" in opts or not head.startswith("Question: ") or "\n" in head or not body.strip()
            or "Question:" in body):
        raise ValueError(f"{iid}: gsm8k_tf prompt is not 'Question: <problem>' + options")
    return body


def gsm8k_question_cut(problem: str) -> tuple[str, str] | None:
    """(statements, question sentence) of a GSM8K problem when its last sentence is a bare question (see GSM8K_TF),
    else None. The statements are the exact text before the last sentence boundary."""
    cuts = list(_GSM_CUT.finditer(problem))
    if not cuts:
        return None
    body, ask = problem[:cuts[-1].start()], problem[cuts[-1].end():]
    if (not body.strip() or "?" in body or not ask.endswith("?") or not _GSM_ASK.match(ask)
            or re.search(r"[\d,;:%$]", ask) or _GSM_NUMWORD.search(ask) or _GSM_PREMISE.search(ask)):
        return None
    return body, ask


def problem_no_options(item: dict) -> str:
    """The problem WITHOUT its answer options but WITH its final question (--tree-view statements+question, GFPQ): the
    prompt before "\nOptions:" (K&K: preamble, statement lines and the question line; numgen: the "Question: ..." line
    with the answer quantity's question sentence; FINAL_QUESTION sources: the statement block, a blank line and the
    final question paragraph; gsm8k_tf: the "Question: <problem>" line, for every item, whether or not its last
    sentence has a clean cut). The final question names what is asked (e.g. which quantity a word problem wants), which
    GFP-S drops. Defined for the sources statement_block covers and gsm8k_tf; anything else raises ValueError."""
    if item.get("source") == GSM8K_TF:                     # own shape check: statement_block may refuse the item
        _gsm8k_tf_problem(item)
        return item["prompt"].partition("\nOptions:")[0]
    statement_block(item)                                  # same sources and shape checks as GFP-S
    head, sep, _ = item["prompt"].partition("\nOptions:")
    if not sep:
        raise ValueError(f"{item.get('item_id')}: prompt has no options block")
    return head


# Views that put the statement block (above) in place of some nodes' local views, in rng lockstep with TFL:
# "local+ctxparse" (TFLP) for parse nodes and nodes without parents, "mix" (MIX) for a seeded coin's share of all nodes.
CTX_VIEWS = ("local+ctxparse", "mix")


def mix_statements(seed: int, epoch: int, item_id, node_id, p: float) -> bool:
    """--tree-view mix: whether this node is asked in the statements view this epoch (else in its local view). Each
    (seed, epoch, item, node) has a coin of its own: random.Random seeds a str key through sha512, so the draw depends
    on nothing else (not on PYTHONHASHSEED, the training rng, the node order or the other nodes). p 0 never, p 1
    always (random() < 1)."""
    return random.Random(f"mix|{seed}|{epoch}|{item_id}|{node_id}").random() < p


# ---- GTF (grounded Tree Forcing): GFP-S with a fading scaffold and depth-weighted nodes --------------------------------
def scaffold_fade(t: int, total: int, fade: float) -> float:
    """--tree-scaffold F: the probability p(t) = max(0, 1 - t / (F x T)) that a derived node is asked with its parents'
    results stated at update t (0-based) of T updates; 1 at the start, 0 from t = F x T on. F 0 (off): 0."""
    if fade <= 0:
        return 0.0
    return max(0.0, 1.0 - t / (fade * max(1, total)))


def scaffold_rng(seed: int, epoch: int, item_id, node_id) -> random.Random:
    """The scaffold's own generator for one (seed, epoch, item, node): its first random() is the coin (scaffold iff
    < p(t)), the rest orders the stated lines. Seeded through a str key (sha512) like MIX's coin, so it depends on
    nothing else (not on PYTHONHASHSEED, the training rng, the node order or the other nodes)."""
    return random.Random(f"scaffold|{seed}|{epoch}|{item_id}|{node_id}")


def depth_split(total: float, depths: list[int], g: float) -> list[float]:
    """--tree-depth-weight G: total split over the views in proportion to (1 + node depth)^G (sums to total)."""
    raw = [(1.0 + d) ** g for d in depths]
    s = sum(raw)
    return [total * r / s for r in raw]


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
