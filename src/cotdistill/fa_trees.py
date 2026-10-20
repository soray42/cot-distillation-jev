"""Reasoning trees for failure attribution (FA) on StepFinder's training trajectories (KDD'26, arXiv 2606.03467).

A tree is a short backward chain of multiple-choice questions that a one-pass student answers from the task text
and the log alone (StepFinder's "without G" protocol): which step states the team's final answer, which earlier
step that answer (or a later decision) relies on, down to the decisive step, then what that step did, why it is
wrong given the task or evidence in the log, and who made it. The decisive step itself (the final question) is
the root and not a node.

Teacher output ("source tree", one JSON object per trajectory; any teacher):

  {"traj_id": "alg/trajectory_0000", "teacher": "<name>", "answer_step": k, "label_note": "",
   "nodes": [
     {"id": "n1", "type": "answer_step", "question": "...", "answer_step": k, "distractor_steps": [..],
      "quote": "<verbatim from step k>", "parents": []},
     {"id": "n2", "type": "provenance", "question": "...", "answer_step": j, "distractor_steps": [..],
      "quote": "<verbatim from step j>", "parents": ["n1"]},
     {"id": "n3", "type": "content", "question": "...", "correct": "...", "distractors": ["...", ...],
      "cite_step": <mistake_step>, "quote": "<verbatim from it>", "parents": ["n2"]},
     {"id": "n4", "type": "conflict", ..., "cite_step": <int, or null = the task text>, "quote": "...",
      "parents": ["n3"]},
     {"id": "n5", "type": "agent", "question": "...", "answer_agent": "X", "distractor_agents": [..],
      "parents": ["n2"]}]}

A teacher that finds the label unsupported by the log writes {"traj_id": ..., "teacher": ..., "drop": "<reason>"}
instead; the trajectory is then rejected with the failure "teacher_drop: <reason>". "label_note" records doubts about
a label the teacher still builds a tree for. "rule_override": "<one sentence>" accepts a disagreement with
answer_step_rule when the rule picks a step that is not the team's answer (e.g. a later remark "Awaiting further data
..." after "Final answer: X"); the reason is kept in checks["answer_rule"]["override"].

assemble() renders the options (pointer options exactly "Step k [AgentName]" with k = history step_id), shuffles
them with a seeded RNG, and returns the output node list {"id", "type", "question", "options", "answer",
"parents", "cite_step", "quote"} (+ "option_steps" on pointer nodes); check_tree() runs every automatic check.

Truncation rule (documented, the same head / tail shape as build_whowhen_eval.cut): every step is shown to the
teacher as cut(content, STEP_CAP) = the first 1,000 and the last 500 characters joined by " [...] " when longer
than 1,500 characters; the task as cut(question, TASK_CAP). Quotes must lie inside that kept text (and in the
original), so they survive any student cap >= STEP_CAP; checks also record the smallest cap at which each quote
survives (quote_min_cap).
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

SUBSET_DIRS = {"alg": "Algorithm-Generated", "hc": "Hand-Crafted", "aftraj": "AFTraj",   # aftraj, real: exports of
               "real": "REAL",                                     # build_fa_aftraj.py / build_fa_real.py --export-sf
               "agtr": "AGTR", "wwpro": "WWPRO", "ce": "CE"}      # build_fa_new_sources.py (LOFO training pools)
STEP_CAP = 1500
TASK_CAP = 1500
NODE_TYPES = ("answer_step", "provenance", "content", "conflict", "agent")
POINTER_TYPES = ("answer_step", "provenance")
TEXT_TYPES = ("content", "conflict")
NON_AGENTS = {"Computer_terminal", "user"}           # tool output and the human task message
MIN_NODES, MAX_NODES = 3, 7
MIN_QUOTE_CHARS = 12                                 # or the whole cited text (a bare answer)
LENGTH_TELL_RATIO = 1.25                              # correct option longer than this x the longest distractor
QUESTION_TELL_MARGIN = 0.25

BOILERPLATE = re.compile(
    r"^(terminate|task complete|session (closed|complete)|done|end|completed|task closed|end of task|"
    r"answer delivered|workflow complete|returning the result)[.!]?$", re.I)
ANSWER_MARKER = re.compile(
    r"(?im)^[\s*#>-]*(final(?:\s+[\w-]+){0,4}|answer)\s*(\([^)\n]*\))?\s*\**\s*[:：]"
    r"|\bfinal answer\s*(\([^)\n]*\))?\s*\**\s*[:：]")
# "Final note:", "Final caution:", "Final output format reminder:", "Final computed totals snapshot:" are headers, not answers
MARKER_NOT_ANSWER = {"note", "notes", "caution", "cautions", "reminder", "reminders", "snapshot", "warning", "warnings",
                     "remark", "remarks", "comment", "comments", "caveat", "caveats", "thought", "thoughts", "check",
                     "checks", "verification", "plan"}


REQUEST_BEFORE = re.compile(r"(?i)\b(prepare|provide|give|submit|send|share|post|need|await\w*|wait(ing)? for|"
                            r"request\w*|deliver)\s+(the|your|a|our)?\s*$")


def is_answer_marker(text: str) -> bool:
    """ANSWER_MARKER anywhere in text, except "Final <words>:" headers whose words name a note / caution / reminder /
    snapshot / check rather than the result (MARKER_NOT_ANSWER)."""
    for m in ANSWER_MARKER.finditer(text):
        head = m.group(1)
        if head and set(re.findall(r"[a-z]+", head.lower())[1:]) & MARKER_NOT_ANSWER:
            continue
        if REQUEST_BEFORE.search(text[max(0, m.start() - 40):m.start()]):   # "Please prepare the final answer:"
            continue
        return True
    return False


def is_note_header(text: str) -> bool:
    """A step opening with such a header ("Final note: the answer relies on ...") is a trailing remark, skipped like a
    closing line by the bare-answer rule."""
    m = re.match(r"(?i)^[\s*#>-]*final((?:\s+[\w-]+){0,4})\s*(\([^)\n]*\))?\s*\**\s*[:：]", text.strip())
    return bool(m and set(re.findall(r"[a-z]+", m.group(1).lower())) & MARKER_NOT_ANSWER)


BARE_MAX = 120
LEAD_YES_NO = re.compile(r"(?i)^(yes|no)([.,!:]|\s*$)")      # "No." / "Yes, ..." but not "No local files ..."              # "Yes. If the green plots ..." also counts as bare
NOT_BARE = re.compile(r"(?i)^(if|noted|acknowledged|understood|thanks|thank you|confirmed|proceed\w*|review\w*|ok|okay|"
                      r"great|sure|we|i|i'll|i'm|let|let's|please|standing by|no further|session|task|workflow|"
                      r"summary|completed?|closing|done)\b")
CLOSING_MAX = 200
CLOSING = re.compile(r"(?i)^if\b|\b(task (is )?(complete|completed|closed)|terminat\w*|session (closed|complete|ended)|"
                     r"answer (provided|delivered|submitted)|(has|have) been (submitted|delivered|provided|sent)|"
                     r"end of task|workflow complete|concluding|finaliz\w*|wrap(ping)?[- ]up|no further|"
                     r"end of (the )?(log|conversation|session|chat|transcript|discussion)|sign(ing|ed)?[- ]off|"
                     r"standing by|on standby|ready to (revisit|assist|help|continue|resume)|logging completion|"
                     r"already concluded)\b")
META_LEAK = re.compile(r"ground[ -]?truth|mistake_reason|is_mistake|\bFM-\d|annotat|gold (step|label|answer)|"
                       r"reference answer|hidden answer", re.I)
STEP_REF = re.compile(r"\bStep\s+(\d+)(?:\s*\[([^\]\n]*)\])?")
STOP = set("""a an the and or of to in on for with by from at as is are was were be been being this that these those it its
into than then so not no but if any all each other which who whom what when where why how does did do done has have had
can could should would will may might must step steps team task their there them they we our you your he she his her
only also just more most some such same about after before over under one two""".split())


# ---------------------------------------------------------------- loading and rendering

def agent_of(name: str) -> str:
    return str(name).split(" (")[0].strip()


def load_traj(path: str | Path, subset: str) -> dict:
    d = json.loads(Path(path).read_text())
    steps = [{"k": int(x["step_id"]), "agent": agent_of(x.get("name") or x.get("role") or "unknown"),
              "role": x.get("role", ""), "text": str(x.get("content", ""))} for x in d["history"]]
    if [s["k"] for s in steps] != list(range(len(steps))):
        raise ValueError(f"{path}: step_ids are not 0..n-1")
    return {"traj_id": f"{subset}/{Path(path).stem}", "subset": subset, "question": str(d["question"]),
            "ground_truth": str(d.get("ground_truth", "")), "mistake_step": int(d["mistake_step"]),
            "mistake_agent": str(d["mistake_agent"]), "mistake_reason": str(d.get("mistake_reason", "")),
            "steps": steps}


def traj_paths(root: str | Path, subset: str) -> list[Path]:
    return sorted((Path(root) / SUBSET_DIRS[subset] / "train").glob("*.json"))


def cut(text: str, cap: int) -> str:
    """build_whowhen_eval.cut: head 2/3 + " [...] " + tail 1/3 of the cap."""
    text = str(text).strip()
    if len(text) <= cap:
        return text
    head = cap * 2 // 3
    return text[:head] + " [...] " + text[-(cap - head):]


def step_label(traj: dict, k: int) -> str:
    return f"Step {k} [{traj['steps'][k]['agent']}]"


def kept_step(traj: dict, k: int, cap: int = STEP_CAP) -> str:
    return cut(traj["steps"][k]["text"], cap)


def kept_task(traj: dict, cap: int = TASK_CAP) -> str:
    return cut(traj["question"], cap)


def agents_in_log(traj: dict) -> list[str]:
    seen: list[str] = []
    for s in traj["steps"]:
        if s["agent"] not in seen:
            seen.append(s["agent"])
    return seen


def render_view(traj: dict, *, teacher_info: bool = True) -> str:
    """The text a teacher reads: task, (labels), and the log at the documented truncation rule."""
    lines = [f"traj_id: {traj['traj_id']}  ({len(traj['steps'])} steps)", "", f"Task: {kept_task(traj)}", ""]
    if teacher_info:
        lines += [f"[teacher only] Ground truth: {traj['ground_truth']}",
                  f"[teacher only] Decisive step: {step_label(traj, traj['mistake_step'])}; "
                  f"agent: {traj['mistake_agent']}",
                  f"[teacher only] Reason: {traj['mistake_reason']}", ""]
    for s in traj["steps"]:
        lines.append(f"{step_label(traj, s['k'])}: {kept_step(traj, s['k'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------- rule for the answer step

def is_bare_answer(text: str) -> bool:
    """A short answer-only message: <= 120 characters, one sentence, not a closing / conversational line."""
    t = text.strip()
    return (0 < len(t) <= BARE_MAX and not BOILERPLATE.match(t) and "TERMINATE" not in t
            and not re.search(r"[.!?]\s+\S", t) and not NOT_BARE.match(t))


def is_closing(text: str) -> bool:
    """A closing / hand-off line after the answer: boilerplate, or short (<= 200 characters) with a closing phrase
    ("Task complete", "TERMINATE", "answer has been submitted", "finalizing", ...) or opening with "If" (an offer
    such as "If you need the date, we can append it")."""
    t = text.strip()
    return bool(not t or BOILERPLATE.match(t) or t == "TERMINATE" or (len(t) <= CLOSING_MAX and CLOSING.search(t)))


def answer_step_rule(traj: dict) -> tuple[int | None, str]:
    """The step that states the team's final answer, by string match; (None, "none") when the rule fails.
    Among agent steps (not Computer_terminal / user):
    1) marker: the last step with "Final answer:" anywhere, or a line opening "Answer:" or "Final ...:" with at most
       four words after "Final" ("Final:", "Final answer:", "Final score:", "Final result submitted:", "Final answer
       and summary:", "Final answer to the user:"; an optional parenthetical before the colon, markdown bullets /
       bold allowed);
    2) bare: the last step that is not a closing line (is_closing: "TERMINATE", "Task complete.", "Answer provided.
       Task complete.", ...), when it is a bare answer (is_bare_answer: "9", "185 yards", "+70.0", a title) or opens
       with a bare "Yes" / "No";
    the later of the two when both exist."""
    agent_steps = [s for s in traj["steps"] if s["agent"] not in NON_AGENTS]
    marked = [s["k"] for s in agent_steps if is_answer_marker(s["text"])]
    bare = None
    for s in reversed(agent_steps):              # skip the closing lines at the end of the log
        if is_closing(s["text"]) or is_note_header(s["text"]):
            continue
        bare = s["k"] if is_bare_answer(s["text"]) or LEAD_YES_NO.match(s["text"].strip()) else None
        break
    if marked and (bare is None or bare <= marked[-1]):
        return marked[-1], "marker"
    if bare is not None:
        return bare, "bare"
    return None, "none"


# ---------------------------------------------------------------- text helpers

def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _norm_text(s: str) -> str:
    return norm_ws(unicodedata.normalize("NFKC", str(s))).casefold()


def gt_leak(gt: str, text: str) -> bool:
    """Whether text contains the ground-truth answer string (case / whitespace insensitive; step references
    "Step k [X]" are ignored; short or numeric answers must stand alone, also with thousands separators)."""
    g = _norm_text(gt).strip(" .\"'")
    if not g:
        return False
    t = _norm_text(STEP_REF.sub(" ", str(text)))
    variants = {g}
    if re.fullmatch(r"\d{4,}", g):
        variants.add(f"{int(g):,}")
    for v in variants:
        if re.fullmatch(r"[\d.,:/ -]+", v) or len(v) <= 3:
            pat = r"(?<![\w.,])" + re.escape(v) + r"(?!\w|[.,]\d)"
        else:
            pat = r"(?<!\w)" + re.escape(v) + r"(?!\w)"
        if re.search(pat, t):
            return True
    return False


def find_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Span of quote in text, whitespace-insensitive (any run of whitespace matches any other), case-sensitive."""
    toks = str(quote).split()
    if not toks:
        return None
    m = re.search(r"\s+".join(re.escape(t) for t in toks), str(text))
    return (m.start(), m.end()) if m else None


def quote_min_cap(text: str, quote: str) -> int | None:
    """Smallest cut() cap at which the quote is still in the kept text (None if it is not in the text)."""
    text = str(text).strip()
    span = find_quote(text, quote)
    if span is None:
        return None
    s, e = span
    n = len(text)

    def ok(c: int) -> bool:
        if c >= n:
            return True
        head = c * 2 // 3
        return e <= head or s >= n - (c - head)

    lo, hi = 1, max(n, 1)
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def option_form(o: str) -> tuple:
    """Surface form class of an option: first word -ing / -ed / other, has a colon or semicolon, opens with a quote."""
    w = re.findall(r"[A-Za-z']+", o)
    f = w[0].lower() if w else ""
    v = "ing" if f.endswith("ing") else "ed" if f.endswith("ed") else "other"
    return (v, bool(re.search(r"[:;]", o)), o.lstrip()[:1] in "\"'\u201c\u2018")


def content_tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", _norm_text(s)) if len(w) >= 3 and w not in STOP}


def recall(tokens: set[str], pool: set[str]) -> float:
    return len(tokens & pool) / len(tokens) if tokens else 0.0


def seeded_rng(seed: int, *parts: str) -> random.Random:
    h = hashlib.sha256("|".join([str(seed), *parts]).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


# ---------------------------------------------------------------- assembly

class TreeError(ValueError):
    pass


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise TreeError(msg)


def assemble(traj: dict, src: dict, seed: int = 20261010) -> list[dict]:
    """Source tree -> output nodes with rendered, shuffled options. Raises TreeError on schema problems."""
    _need(isinstance(src, dict), "schema: not an object")
    if src.get("drop"):
        raise TreeError(f"teacher_drop: {src['drop']}")
    _need(isinstance(src.get("nodes"), list), "schema: no node list")
    out = []
    n = len(traj["steps"])
    for i, sn in enumerate(src["nodes"]):
        _need(isinstance(sn, dict), f"schema: node {i} is not an object")
        nid, typ = sn.get("id"), sn.get("type")
        _need(isinstance(nid, str) and nid, f"schema: node {i} has no id")
        _need(typ in NODE_TYPES, f"schema: {nid} type {typ!r}")
        q = sn.get("question")
        _need(isinstance(q, str) and q.strip() != "", f"schema: {nid} question")
        parents = sn.get("parents", [])
        _need(isinstance(parents, list) and all(isinstance(p, str) for p in parents), f"schema: {nid} parents")
        rng = seeded_rng(seed, traj["traj_id"], nid)
        node = {"id": nid, "type": typ, "question": q.strip(), "options": [], "answer": None,
                "parents": list(parents), "cite_step": None, "quote": ""}
        if typ in POINTER_TYPES:
            a, ds = sn.get("answer_step"), sn.get("distractor_steps")
            _need(isinstance(a, int) and not isinstance(a, bool), f"schema: {nid} answer_step")
            _need(isinstance(ds, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in ds),
                  f"schema: {nid} distractor_steps")
            steps = [a] + list(ds)
            _need(all(0 <= k < n for k in steps), f"pointer_range: {nid}")
            order = steps[:]
            rng.shuffle(order)
            node.update(options=[step_label(traj, k) for k in order], answer=order.index(a), option_steps=order,
                        cite_step=a, quote=str(sn.get("quote", "")).strip())
        elif typ == "agent":
            a, ds = sn.get("answer_agent"), sn.get("distractor_agents")
            _need(isinstance(a, str) and isinstance(ds, list) and all(isinstance(x, str) for x in ds),
                  f"schema: {nid} agents")
            opts = [a] + list(ds)
            order = opts[:]
            rng.shuffle(order)
            node.update(options=order, answer=order.index(a))
        else:
            c, ds = sn.get("correct"), sn.get("distractors")
            _need(isinstance(c, str) and isinstance(ds, list) and all(isinstance(x, str) for x in ds),
                  f"schema: {nid} options")
            cs = sn.get("cite_step")
            _need(cs is None or (isinstance(cs, int) and not isinstance(cs, bool)), f"schema: {nid} cite_step")
            opts = [c.strip()] + [x.strip() for x in ds]
            order = opts[:]
            rng.shuffle(order)
            node.update(options=order, answer=order.index(opts[0]), cite_step=cs,
                        quote=str(sn.get("quote", "")).strip())
        out.append(node)
    return out


# ---------------------------------------------------------------- checks

def _node_texts(node: dict) -> list[str]:
    """Free text of a node: question, quote and the options written by the teacher (pointer options are rendered
    step labels and agent options are agent names, both taken from the log)."""
    texts = [node["question"], node.get("quote", "")]
    if node["type"] in TEXT_TYPES:
        texts += node["options"]
    return texts


def _quote_source(traj: dict, cite: int | None) -> tuple[str, str]:
    """(kept text, original text) of a cited step, or of the task when cite is None."""
    if cite is None:
        return kept_task(traj), traj["question"]
    return kept_step(traj, cite), traj["steps"][cite]["text"]


def check_tree(traj: dict, src: dict, nodes: list[dict] | None = None, seed: int = 20261010) -> dict:
    """Every automatic check. Returns {"pass", "failures", ...stats}; nodes = assemble(traj, src) if not given."""
    fails: list[str] = []
    res: dict = {"pass": False, "failures": fails}
    if nodes is None:
        try:
            nodes = assemble(traj, src, seed)
        except TreeError as e:
            fails.append(str(e))
            return res
    n = len(traj["steps"])
    ms, ma = traj["mistake_step"], traj["mistake_agent"]
    by_id: dict[str, dict] = {}
    types = [nd["type"] for nd in nodes]
    res["types"] = {t: types.count(t) for t in NODE_TYPES if t in types}
    res["n_nodes"] = len(nodes)
    if not MIN_NODES <= len(nodes) <= MAX_NODES:
        fails.append(f"node_count: {len(nodes)}")
    for t in ("answer_step", "content", "conflict"):
        if types.count(t) != 1:
            fails.append(f"type_count: {t} x{types.count(t)}")
    if types.count("agent") > 1:
        fails.append(f"type_count: agent x{types.count('agent')}")
    real_agents = [a for a in agents_in_log(traj) if a not in NON_AGENTS]
    if "agent" not in types and len(real_agents) >= 2:
        fails.append("type_count: agent x0")
    # ids, parents, acyclicity (parents must be earlier nodes)
    for nd in nodes:
        if nd["id"] in by_id:
            fails.append(f"ids_dup: {nd['id']}")
        for p in nd["parents"]:
            if p not in by_id:
                fails.append(f"parent_order: {nd['id']} <- {p}")
        by_id[nd["id"]] = nd
    # options: counts, distinctness, correct index
    for nd in nodes:
        k = len(nd["options"])
        lo = {"answer_step": min(3, n), "provenance": min(3, n), "content": 3, "conflict": 3, "agent": 2}[nd["type"]]
        if not lo <= k <= 5:
            fails.append(f"option_count: {nd['id']} {k}")
        if len({_norm_text(o) for o in nd["options"]}) != k:
            fails.append(f"options_dup: {nd['id']}")
        if not isinstance(nd["answer"], int) or not 0 <= nd["answer"] < max(k, 1):
            fails.append(f"answer_index: {nd['id']}")
    # pointer chain: answer_step -> provenance* -> mistake_step, strictly backward
    chain: list[int] = []
    end_node = None
    ans_nodes = [nd for nd in nodes if nd["type"] == "answer_step"]
    if ans_nodes:
        cur = ans_nodes[0]
        if cur["parents"]:
            fails.append("chain: answer_step has parents")
        seen = set()
        while cur is not None and cur["id"] not in seen:
            seen.add(cur["id"])
            k = cur["option_steps"][cur["answer"]]
            if chain and k >= chain[-1]:
                fails.append(f"chain_not_backward: {cur['id']}")
            chain.append(k)
            end_node = cur
            kids = [nd for nd in nodes if nd["type"] == "provenance" and cur["id"] in nd["parents"]]
            if len(kids) > 1:
                fails.append(f"chain_branch: {cur['id']}")
            cur = kids[0] if kids else None
        orphans = [nd["id"] for nd in nodes if nd["type"] == "provenance" and nd["id"] not in seen]
        if orphans:
            fails.append(f"chain_orphan: {','.join(orphans)}")
        if not chain or chain[-1] != ms:
            fails.append(f"chain_not_mistake: ends at {chain[-1] if chain else None}, decisive {ms}")
    res["chain"] = chain
    # answer step rule vs teacher
    rule_k, how = answer_step_rule(traj)
    teacher_k = chain[0] if chain else None
    agree = None if rule_k is None or teacher_k is None else rule_k == teacher_k
    override = (src.get("rule_override") or "").strip()
    res["answer_rule"] = {"rule": rule_k, "how": how, "teacher": teacher_k, "agree": agree,
                          **({"override": override} if override else {})}
    if agree is False and not override:
        fails.append(f"answer_rule_disagree: rule {rule_k}, teacher {teacher_k}")
    if src.get("answer_step") is not None and teacher_k is not None and src.get("answer_step") != teacher_k:
        fails.append("answer_step_field_mismatch")
    # node-specific: content / conflict / agent anchored at the chain end
    for nd in nodes:
        if nd["type"] in ("content", "agent") and end_node is not None and end_node["id"] not in nd["parents"]:
            fails.append(f"parents: {nd['id']} not under the chain end")
        if nd["type"] == "conflict" and not any(by_id.get(p, {}).get("type") in ("content", "provenance",
                                                                                  "answer_step")
                                                for p in nd["parents"]):
            fails.append(f"parents: {nd['id']} has no content / pointer parent")
        if nd["type"] == "content" and nd["cite_step"] != ms:
            fails.append(f"content_cite: {nd['id']} cites {nd['cite_step']}, decisive {ms}")
        if nd["type"] == "agent":
            a = nd["options"][nd["answer"]]
            if a != ma:
                fails.append(f"agent_wrong: {a} vs {ma}")
            if traj["steps"][ms]["agent"] != ma:
                fails.append("agent_speaker_mismatch")
            log_agents = set(agents_in_log(traj))
            if any(o not in log_agents for o in nd["options"]):
                fails.append(f"agent_option_unknown: {nd['id']}")
    # quotes: verbatim (whitespace-normalized) inside the kept text and the original
    qcap: dict[str, int | None] = {}
    for nd in nodes:
        if nd["type"] == "agent":
            continue
        cite, quote = nd["cite_step"], nd.get("quote", "")
        if cite is not None and not (isinstance(cite, int) and 0 <= cite < n):
            fails.append(f"cite_range: {nd['id']}")
            continue
        kept, orig = _quote_source(traj, cite)
        if len(norm_ws(quote)) < MIN_QUOTE_CHARS and norm_ws(quote) != norm_ws(orig):
            fails.append(f"quote_short: {nd['id']}")
            continue
        if "[...]" in quote or find_quote(kept, quote) is None or find_quote(orig, quote) is None:
            fails.append(f"quote_not_found: {nd['id']}")
            continue
        qcap[nd["id"]] = quote_min_cap(orig, quote)
    res["quote_min_cap"] = qcap
    # leaks: ground truth, labels / meta words, the node's own answer in its question
    gt = traj["ground_truth"]
    for nd in nodes:
        # the student sees question + options; quotes are evidence copied from the visible log, never shown as text
        if any(gt_leak(gt, t) for t in [nd["question"]] + (nd["options"] if nd["type"] in TEXT_TYPES else [])):
            fails.append(f"gt_leak: {nd['id']}")
        if any(META_LEAK.search(t) for t in _node_texts(nd)):
            fails.append(f"meta_leak: {nd['id']}")
        correct = nd["options"][nd["answer"]] if nd["options"] else ""
        if nd["type"] in POINTER_TYPES:
            k = nd["option_steps"][nd["answer"]]
            if any(int(m.group(1)) == k for m in STEP_REF.finditer(nd["question"])):
                fails.append(f"self_leak: {nd['id']} names its answer step")
        elif correct and _norm_text(correct) in _norm_text(nd["question"]):
            fails.append(f"self_leak: {nd['id']} question contains the correct option")
        # step references in free text must match the log
        for t in _node_texts(nd):
            for m in STEP_REF.finditer(t):
                k = int(m.group(1))
                if k >= n or (m.group(2) is not None and m.group(2).strip() != traj["steps"][k]["agent"]):
                    fails.append(f"step_ref_bad: {nd['id']} '{m.group(0)}'")
    # length tell and rank of the correct option (content / conflict)
    ranks: dict[str, int] = {}
    for nd in nodes:
        if nd["type"] not in TEXT_TYPES or not nd["options"]:
            continue
        lens = [len(o) for o in nd["options"]]
        c = lens[nd["answer"]]
        others = [x for i, x in enumerate(lens) if i != nd["answer"]]
        ranks[nd["id"]] = 1 + sum(x > c for x in others)
        if others and c > LENGTH_TELL_RATIO * max(others):
            fails.append(f"length_tell: {nd['id']} {c} vs {max(others)}")
    res["length_rank"] = ranks
    # answerable from the log, not the task text alone (cheap token-overlap probe; soft)
    task_tok = content_tokens(kept_task(traj))
    log_tok: set[str] = set()
    for s in traj["steps"]:
        if s["agent"] != "user":
            log_tok |= content_tokens(kept_step(traj, s["k"]))
    qo: dict[str, dict] = {}
    for nd in nodes:
        if nd["type"] not in TEXT_TYPES or not nd["options"]:
            continue
        toks = [content_tokens(o) for o in nd["options"]]
        qr = [recall(t, task_tok) for t in toks]
        lr = recall(toks[nd["answer"]], log_tok)
        c = qr[nd["answer"]]
        others = [x for i, x in enumerate(qr) if i != nd["answer"]]
        qo[nd["id"]] = {"q_recall": round(c, 3), "log_recall": round(lr, 3),
                        "question_only": c > lr,
                        "question_match_tell": bool(others) and c >= max(others) + QUESTION_TELL_MARGIN}
    res["question_probe"] = qo
    # log-overlap rank of the correct option (content / conflict; 1 = most of its words in the log, 4 = fewest): the
    # pilot-scale trees put verbatim actions of other steps in the distractors, so "pick the option least present in the
    # log" was a tell; a source node may carry target_overlap_rank (1-4), checked with ties counted either way
    orank: dict[str, int] = {}
    src_nodes = {x.get("id"): x for x in (src.get("nodes") or [])}
    for nd in nodes:
        if nd["type"] not in TEXT_TYPES or not nd["options"]:
            continue
        rec = [recall(content_tokens(o), log_tok) for o in nd["options"]]
        c = rec[nd["answer"]]
        others = [x for i, x in enumerate(rec) if i != nd["answer"]]
        hi, lo = 1 + sum(x > c for x in others), 1 + sum(x >= c for x in others)
        orank[nd["id"]] = hi
        want = (src_nodes.get(nd["id"]) or {}).get("target_overlap_rank")
        if want is not None and not hi <= int(want) <= lo:
            fails.append(f"overlap_rank_miss: {nd['id']} rank {hi}-{lo}, target {want}")
    res["overlap_rank"] = orank
    # grammatical form tell (content / conflict): exactly one option differs from the other three in its form class
    # (first word ending in -ing / -ed / other, a colon or semicolon, an opening quote), e.g. a correct option quoted as
    # an imperative among three past-tense distractors; "pick the odd one" scored 51% on content nodes before the fix
    for nd in nodes:
        if nd["type"] not in TEXT_TYPES or len(nd["options"]) < 3:
            continue
        fm = [option_form(o) for o in nd["options"]]
        cnt = collections.Counter(fm)
        if len(cnt) == 2 and min(cnt.values()) == 1:
            odd = next(i for i, f in enumerate(fm) if cnt[f] == 1)
            fails.append(f"form_tell: {nd['id']} option {odd} {fm[odd]} vs {fm[(odd + 1) % len(fm)]}")
    # grounding of content options: the step whose kept text shares most tokens with each option (soft)
    step_tok = [content_tokens(kept_step(traj, s["k"])) for s in traj["steps"]]
    ground: dict[str, dict] = {}
    for nd in nodes:
        if nd["type"] != "content" or not nd["options"]:
            continue
        best = []
        for o in nd["options"]:
            t = content_tokens(o)
            sc = [recall(t, st) for st in step_tok]
            best.append(max(range(n), key=lambda i: (sc[i], -abs(i - ms))))
        ground[nd["id"]] = {"correct_best_step": best[nd["answer"]], "cited": nd["cite_step"],
                            "distractor_best_steps": [b for i, b in enumerate(best) if i != nd["answer"]]}
    res["grounding"] = ground
    res["pass"] = not fails
    return res


def output_record(traj: dict, src: dict, seed: int = 20261010, extra_meta: dict | None = None) -> dict:
    """One output line: {"traj_id", "subset", "teacher", "nodes", "checks", "meta"}."""
    try:
        nodes = assemble(traj, src, seed)
    except TreeError as e:
        nodes, checks = [], {"pass": False, "failures": [str(e)]}
    else:
        checks = check_tree(traj, src, nodes, seed)
    if src.get("label_note"):
        checks["label_note"] = src["label_note"]
    meta = {"n_steps": len(traj["steps"]), "mistake_step": traj["mistake_step"],
            "mistake_agent": traj["mistake_agent"], "step_cap": STEP_CAP, "task_cap": TASK_CAP, "seed": seed}
    meta.update(extra_meta or {})
    return {"traj_id": traj["traj_id"], "subset": traj["subset"], "teacher": src.get("teacher", ""),
            "nodes": nodes, "checks": checks, "meta": meta}


# ================================================================ v3 (2026-10-19): index-free render and clean nodes
# Why: pointer nodes (answer_step / provenance, options "Step k [Agent]") carry the training position prior, while
# content / conflict / agent nodes transfer across frameworks (results/fa_pool/abl_*_vsA2.md). v3 (teacher brief:
# data/raw/stepfinder_trees/BRIEF_v3_addon.md)
#   1. renders every node without step numbers (free_render): a pointer option is "<Agent>: “<excerpt>”", a step
#      reference in a question or text option becomes "the <Agent> step “<excerpt>”". ONE excerpt rule for the correct
#      option and the distractors alike: the first W whitespace-separated words of the step's visible text (the head
#      of cut(text, cap) without a possibly partial last word; at most EXCERPT_MAX_CHARS characters), "…" when the
#      step is longer, where W is the smallest
#      width >= EXCERPT_MIN_WORDS (at most EXCERPT_MAX_WORDS) at which (agent, first W words) identifies the step among
#      ALL steps of the log; a pointer node uses one W for all its options (the largest any option needs), a reference
#      in free text the W of its step. A node whose steps cannot be told apart (an exact duplicate of the same agent,
#      a longer common opening, no words), whose excerpt contains a step index, whose question quotes its own answer
#      step, or whose rewritten text option changes the correct option's length rank / adds a length or form tell is
#      dropped; its children are re-attached to its parents. Kept nodes carry render = "free" and the steps they
#      quote (free_steps); types stay answer_step / provenance / ... so --tree-node-types selects them as before;
#   2. adds teacher-written "clean" nodes: for 1-2 steps strictly before the decisive step (chosen by code, biased to
#      the absolute positions where the source's decisive errors are most common, plus one random earlier step;
#      clean_targets), "Which
#      statement about <step> is supported by the task and the log?" with four statements of the content node's form
#      (one true, three false by the log), no evaluative wording. Add-on source per trajectory:
#        {"traj_id", "teacher", "nodes": [{"id": "c1", "type": "clean", "cite_step": k, "correct": "...",
#          "distractors": [3 x "..."], "quote": "<verbatim from step k>"},
#         {"id": "c2", "type": "clean", "cite_step": j, "skip": "<why step j is itself wrong>"}]}
#      assemble_addon() renders the fixed question (index form; free_render rewrites it) and shuffles the options with
#      the v2 seeded RNG; check_addon() runs the clean-node checks against the work-list entry (targets).
# Every v2 function above is unchanged; nothing here runs unless called.
EXCERPT_MIN_WORDS, EXCERPT_MAX_WORDS = 6, 40
EXCERPT_MAX_CHARS = 160                    # a longer W-word opening (JSON, URLs) is cut here, then "…" (0.24% of them)
ELLIPSIS = "…"
OPEN_Q, CLOSE_Q = "“", "”"
INDEX_WORD = re.compile(r"(?i)\bsteps?\s*(?:#|no\.?\s*|number\s*)?\d")          # "Step 3", "steps 2", "step #4"
FREE_REF = re.compile(r"(?i)\bstep\s+(\d+)(?:\s*\[([^\]\n]*)\])?")
CLEAN_QUESTION = "Which statement about {ref} is supported by the task and the log?"
CLEAN_NON_AGENTS = {"Computer_terminal", "user", "environment", "tool", "system", "human"}
CLEAN_ID = re.compile(r"^c\d+$")
CLEAN_OPTIONS = 4
EVALUATIVE = re.compile(
    r"(?i)\b(correct(ly)?|incorrect(ly)?|fine|valid(ly)?|invalid(ly)?|proper(ly)?|improper(ly)?|accurate(ly)?|"
    r"inaccurate(ly)?|wrong(ly)?|mistak\w*|erroneous(ly)?|flawed|faulty|appropriate(ly)?|inappropriate(ly)?|"
    r"sound(ly)?|reasonabl[ey]|justified|unjustified|legitimate(ly)?|sensibl[ey]|rightly|decisive|"
    r"(no|without( any| an)?)\s+(errors?|issues?|problems?|flaws?))\b")
STEM_MATCH_MARGIN = 0.25
PRIOR_POWER = 4                            # clean_targets: weight P_abs(k) ** 4 (when a top-3 position is eligible, the
                                           # pick lands there 57-95% of the time by source, 38-88% with P_abs ** 1)


def step_words(text: str, cap: int = STEP_CAP) -> tuple[list[str], bool]:
    """Words of a step's visible opening and whether the step goes on after them: the whole text when it fits the
    cap, else the head of cut(text, cap) (2/3 of the cap) without its last word when that word is cut."""
    t = str(text).strip()
    if len(t) <= cap:
        return t.split(), False
    h = cap * 2 // 3
    w = t[:h].split()
    if w and not t[h - 1].isspace() and not t[h].isspace():          # the head ends inside a word
        w = w[:-1]
    return w, True


def _step_table(traj: dict, cap: int) -> tuple[list[str], list[list[str]], list[bool]]:
    agents = [s["agent"] for s in traj["steps"]]
    sw = [step_words(s["text"], cap) for s in traj["steps"]]
    return agents, [w for w, _ in sw], [tr for _, tr in sw]


def excerpt_text(words: list[str], truncated: bool, w: int) -> str:
    """The first w words, cut at EXCERPT_MAX_CHARS characters; "…" when the step goes on."""
    s = " ".join(words[:w])
    if len(s) > EXCERPT_MAX_CHARS:
        return s[:EXCERPT_MAX_CHARS].rstrip() + ELLIPSIS
    return s + (ELLIPSIS if truncated or len(words) > w else "")


def excerpt_width(agents: list[str], words: list[list[str]], trunc: list[bool], ks: list[int],
                  wmin: int = EXCERPT_MIN_WORDS, wmax: int = EXCERPT_MAX_WORDS) -> int | None:
    """Smallest W in [wmin, wmax] at which (agent, excerpt at W without its "…", case / whitespace normalized) of
    every step in ks differs from that of every other step of the log; None when some step of ks cannot be
    identified that way (an exact duplicate, a longer common opening, a common first EXCERPT_MAX_CHARS characters,
    no words)."""
    if any(not words[k] for k in ks):
        return None
    for w in range(wmin, wmax + 1):
        key = [(agents[j], _norm_text(excerpt_text(words[j], trunc[j], w).rstrip(ELLIPSIS)))
               for j in range(len(words))]
        if all(not any(j != k and key[j] == key[k] for j in range(len(words))) for k in ks):
            return w
    return None


def free_option(agent: str, ex: str) -> str:
    return f"{agent}: {OPEN_Q}{ex}{CLOSE_Q}"


def free_ref(agent: str, ex: str) -> str:
    return f"the {agent} step {OPEN_Q}{ex}{CLOSE_Q}"


def _free_rewrite(text: str, agents, words, trunc) -> tuple[str | None, str, list[int]]:
    """Every step reference in text ("Step k [Agent]", "Step k", any case) -> free_ref; (None, reason, []) when a
    reference is out of range / names another agent / cannot be identified / quotes a step index, or an index word
    is left outside the inserted excerpts."""
    out, plain, refs, pos = [], [], [], 0
    for m in FREE_REF.finditer(text):
        k = int(m.group(1))
        if k >= len(agents):
            return None, "free_ref_range", []
        if m.group(2) is not None and m.group(2).strip() != agents[k]:
            return None, "free_ref_agent", []
        w = excerpt_width(agents, words, trunc, [k])
        if w is None:
            return None, "free_ambiguous", []
        ex = excerpt_text(words[k], trunc[k], w)
        if INDEX_WORD.search(ex):
            return None, "free_excerpt_index", []
        rep = free_ref(agents[k], ex)
        before = "".join(out) + text[pos:m.start()]
        if not before.strip() or before.rstrip()[-1] in ".?!":
            rep = rep[0].upper() + rep[1:]
        out += [text[pos:m.start()], rep]
        plain += [text[pos:m.start()], "the step"]
        refs.append(k)
        pos = m.end()
    out.append(text[pos:])
    plain.append(text[pos:])
    if INDEX_WORD.search("".join(plain)):
        return None, "free_index_left", []
    return "".join(out), "", refs


def _len_rank(options: list[str], answer: int) -> int:
    c = len(options[answer])
    return 1 + sum(len(o) > c for i, o in enumerate(options) if i != answer)


def _length_tell(options: list[str], answer: int) -> bool:
    others = [len(o) for i, o in enumerate(options) if i != answer]
    return bool(others) and len(options[answer]) > LENGTH_TELL_RATIO * max(others)


def _form_tell(options: list[str]) -> bool:
    if len(options) < 3:
        return False
    cnt = collections.Counter(option_form(o) for o in options)
    return len(cnt) == 2 and min(cnt.values()) == 1


def free_render(traj: dict, nodes: list[dict], *, cap: int = STEP_CAP, parent_key: str = "parents",
                stats: collections.Counter | None = None) -> list[dict]:
    """Index-free copy of assembled nodes (see the v3 notes above); stats counts kept nodes, dropped nodes by reason
    (free_drop/<reason>) and the excerpt widths of pointer nodes (free_width/<W>)."""
    st = stats if stats is not None else collections.Counter()
    agents, words, trunc = _step_table(traj, cap)
    out, dropped = [], {}
    for nd in nodes:
        new = copy.deepcopy(nd)
        why = ""
        refs: list[int] = []
        q, why, qrefs = _free_rewrite(nd["question"], agents, words, trunc)
        refs += qrefs
        if not why and nd["type"] in POINTER_TYPES:
            ks = list(nd["option_steps"])
            w = excerpt_width(agents, words, trunc, ks)
            if w is None:
                why = "free_ambiguous"
            else:
                exs = [excerpt_text(words[k], trunc[k], w) for k in ks]
                opts = [free_option(agents[k], ex) for k, ex in zip(ks, exs)]
                if any(INDEX_WORD.search(ex) for ex in exs):
                    why = "free_excerpt_index"
                elif len({_norm_text(o) for o in opts}) != len(opts):
                    why = "free_options_dup"
                elif _norm_text(exs[nd["answer"]].rstrip(ELLIPSIS)) in _norm_text(q):
                    why = "free_self_leak"
                else:
                    new["options"] = opts
                    refs += ks
                    st[f"free_width/{w}"] += 1
        elif not why and nd["type"] in TEXT_TYPES + ("clean",):
            opts = []
            for o in nd["options"]:
                o2, why, orefs = _free_rewrite(o, agents, words, trunc)
                if why:
                    break
                opts.append(o2)
                refs += orefs
            if not why and opts != nd["options"]:
                st["free_text_options_rewritten"] += 1
                if (_len_rank(opts, nd["answer"]) != _len_rank(nd["options"], nd["answer"])
                        or _length_tell(opts, nd["answer"])):
                    why = "free_option_rank"
                elif _form_tell(opts) and not _form_tell(nd["options"]):
                    why = "free_option_form"
                else:
                    new["options"] = opts
        if why:
            st[f"free_drop/{why}"] += 1
            st[f"free_drop_type/{nd['type']}"] += 1
            dropped[nd["id"]] = list(nd.get(parent_key) or [])
            continue
        new["question"] = q
        new["render"] = "free"
        new["free_steps"] = sorted(set(refs))
        st["free_kept"] += 1
        st[f"free_kept_type/{nd['type']}"] += 1
        out.append(new)

    def lift(ps: list[str], seen: frozenset = frozenset()) -> list[str]:
        res: list[str] = []
        for p in ps:
            for x in (lift(dropped[p], seen | {p}) if p in dropped and p not in seen else [p]):
                if x not in res and x not in dropped:
                    res.append(x)
        return res
    for nd in out:
        nd[parent_key] = lift(list(nd.get(parent_key) or []))
    return out


def node_text_index_free(nd: dict) -> bool:
    """Whether a node's student-visible text (question and options) has no step index outside the quoted excerpts."""
    strip = lambda s: re.sub(OPEN_Q + "[^" + CLOSE_Q + "]*" + CLOSE_Q, " ", s)     # noqa: E731
    return not any(INDEX_WORD.search(strip(t)) for t in [nd["question"]] + list(nd["options"]))


# ---------------------------------------------------------------- v3 clean nodes: targets, assembly, checks

def position_prior(trajs: list[dict]) -> dict:
    """The source's decisive-position histograms: absolute step index and steps from the end."""
    ab, en = collections.Counter(), collections.Counter()
    for t in trajs:
        ab[t["mistake_step"]] += 1
        en[len(t["steps"]) - 1 - t["mistake_step"]] += 1
    n = max(1, len(trajs))
    return {"abs": {k: v / n for k, v in sorted(ab.items())}, "end": {k: v / n for k, v in sorted(en.items())},
            "n": len(trajs)}


def clean_speakers(trajs: list[dict]) -> set[str]:
    """Speakers a clean node may target: those that are the decisive speaker of some label of the source, except tool
    output / environment / user / system speakers (CLEAN_NON_AGENTS)."""
    return {t["steps"][t["mistake_step"]]["agent"] for t in trajs} - CLEAN_NON_AGENTS


def clean_targets(traj: dict, prior: dict, speakers: set[str], seed: int = 20261019) -> list[dict]:
    """Up to two clean-node targets strictly before the decisive step, among the steps of eligible speakers that have
    an index-free reference (free_ref_for): one drawn with weight P_abs(k) ** PRIOR_POWER (the source's decisive-
    position histogram over absolute step indices, the indices the final question's options carry, sharpened so that
    the pick lands on the most common error positions: show correct steps where the prior expects an error; with a
    plain P_abs weight the flat Who&When-style histograms made the pick nearly uniform), one uniform among the
    remaining eligible earlier steps. Seeded per trajectory."""
    ms = traj["mistake_step"]
    elig = [k for k in range(ms) if traj["steps"][k]["agent"] in speakers and free_ref_for(traj, k) is not None]
    if not elig:
        return []
    rng = seeded_rng(seed, traj["traj_id"], "clean")
    wts = [prior["abs"].get(k, 0.0) ** PRIOR_POWER + 1e-12 for k in elig]
    first = rng.choices(elig, weights=wts)[0]
    out = [{"step": first, "why": "prior", "prior_p_abs": round(prior["abs"].get(first, 0.0), 4)}]
    rest = [k for k in elig if k != first]
    if rest:
        out.append({"step": rng.choice(rest), "why": "random"})
    for x in out:
        x["agent"] = traj["steps"][x["step"]]["agent"]
    return sorted(out, key=lambda x: x["step"])


def free_ref_for(traj: dict, k: int, cap: int = STEP_CAP) -> str | None:
    """The index-free reference of step k ("the <Agent> step “<excerpt>”"), None when it cannot be identified."""
    agents, words, trunc = _step_table(traj, cap)
    w = excerpt_width(agents, words, trunc, [k])
    if w is None:
        return None
    ex = excerpt_text(words[k], trunc[k], w)
    return None if INDEX_WORD.search(ex) else free_ref(agents[k], ex)


def assemble_addon(traj: dict, src: dict, seed: int = 20261010) -> tuple[list[dict], list[dict]]:
    """Add-on source -> (assembled clean nodes, skips). Raises TreeError on schema problems."""
    _need(isinstance(src, dict), "schema: not an object")
    _need(isinstance(src.get("nodes"), list), "schema: no node list")
    n = len(traj["steps"])
    nodes, skips = [], []
    for i, sn in enumerate(src["nodes"]):
        _need(isinstance(sn, dict), f"schema: node {i} is not an object")
        nid = sn.get("id")
        _need(isinstance(nid, str) and bool(CLEAN_ID.match(nid)), f"schema: node {i} id {nid!r} (c1, c2, ...)")
        _need(sn.get("type") == "clean", f"schema: {nid} type {sn.get('type')!r}")
        k = sn.get("cite_step")
        _need(isinstance(k, int) and not isinstance(k, bool) and 0 <= k < n, f"schema: {nid} cite_step")
        if "skip" in sn:
            _need(isinstance(sn["skip"], str) and sn["skip"].strip() != "", f"schema: {nid} skip reason")
            skips.append({"id": nid, "cite_step": k, "reason": sn["skip"].strip()})
            continue
        c, ds = sn.get("correct"), sn.get("distractors")
        _need(isinstance(c, str) and c.strip() != "" and isinstance(ds, list)
              and all(isinstance(x, str) and x.strip() for x in ds), f"schema: {nid} options")
        opts = [" ".join(c.split())] + [" ".join(x.split()) for x in ds]
        order = opts[:]
        seeded_rng(seed, traj["traj_id"], nid).shuffle(order)
        nodes.append({"id": nid, "type": "clean", "question": CLEAN_QUESTION.format(ref=step_label(traj, k)),
                      "options": order, "answer": order.index(opts[0]), "parents": [], "cite_step": k,
                      "quote": str(sn.get("quote", "")).strip()})
    return nodes, skips


def check_addon(traj: dict, src: dict, entry: dict, seed: int = 20261010) -> dict:
    """Assemble + every clean-node check against the work-list entry (its "targets": [{"step", "target_len_rank"}]).
    Returns {"pass", "failures", "nodes", "skips", ...stats}."""
    fails: list[str] = []
    res: dict = {"pass": False, "failures": fails, "nodes": [], "skips": []}
    if isinstance(src, dict) and src.get("drop"):
        fails.append(f"teacher_drop: {src['drop']}")
        return res
    try:
        nodes, skips = assemble_addon(traj, src, seed)
    except TreeError as e:
        fails.append(str(e))
        return res
    res["nodes"], res["skips"] = nodes, skips
    n, ms, gt = len(traj["steps"]), traj["mistake_step"], traj["ground_truth"]
    targets = {int(t["step"]): t for t in entry.get("targets") or []}
    ids = [x["id"] for x in nodes] + [x["id"] for x in skips]
    if len(set(ids)) != len(ids):
        fails.append("ids_dup")
    steps = [x["cite_step"] for x in nodes] + [x["cite_step"] for x in skips]
    if sorted(steps) != sorted(targets):
        fails.append(f"addon_targets: wrote {sorted(steps)}, targets {sorted(targets)}")
    log_tok: set[str] = set()
    for s in traj["steps"]:
        if s["agent"] != "user":
            log_tok |= content_tokens(kept_step(traj, s["k"]))
    res.update(length_rank={}, overlap_rank={}, stem_recall={}, best_step={})
    for nd in nodes:
        nid, k, opts, a = nd["id"], nd["cite_step"], nd["options"], nd["answer"]
        if k >= ms:
            fails.append(f"clean_not_before_decisive: {nid} step {k}, decisive {ms}")
        if len(opts) != CLEAN_OPTIONS:
            fails.append(f"option_count: {nid} {len(opts)}")
        if len({_norm_text(o) for o in opts}) != len(opts):
            fails.append(f"options_dup: {nid}")
        kept, orig = kept_step(traj, k), traj["steps"][k]["text"]
        quote = nd["quote"]
        if len(norm_ws(quote)) < MIN_QUOTE_CHARS and norm_ws(quote) != norm_ws(orig):
            fails.append(f"quote_short: {nid}")
        elif "[...]" in quote or find_quote(kept, quote) is None or find_quote(orig, quote) is None:
            fails.append(f"quote_not_found: {nid}")
        if any(gt_leak(gt, t) for t in [nd["question"]] + opts):
            fails.append(f"gt_leak: {nid}")
        if any(META_LEAK.search(t) for t in opts + [quote]):
            fails.append(f"meta_leak: {nid}")
        ev = [m.group(0) for o in opts for m in [EVALUATIVE.search(o)] if m]
        if ev:
            fails.append(f"evaluative_word: {nid} {ev[0]!r}")
        if any(INDEX_WORD.search(o) for o in opts):
            fails.append(f"clean_index: {nid} an option names a step number")
        if _norm_text(opts[a]) in _norm_text(nd["question"]):
            fails.append(f"self_leak: {nid}")
        # length: 0 = shortest .. 3 = longest among the four, ties either way; the target comes from the work list
        lens = [len(o) for o in opts]
        lo, hi = sum(x < lens[a] for x in lens), sum(x <= lens[a] for x in lens) - 1
        res["length_rank"][nid] = lo
        want = (targets.get(k) or {}).get("target_len_rank")
        if want is not None and not lo <= int(want) <= hi:
            fails.append(f"length_rank_miss: {nid} rank {lo}{'-' + str(hi) if hi != lo else ''}, target {want} "
                         f"(characters: true {lens[a]}, others {sorted(x for i, x in enumerate(lens) if i != a)})")
        if _length_tell(opts, a):
            fails.append(f"length_tell: {nid}")
        if _form_tell(opts):
            fm = [option_form(o) for o in opts]
            cnt = collections.Counter(fm)
            odd = next(i for i, f in enumerate(fm) if cnt[f] == 1)
            fails.append(f"form_tell: {nid} option {odd} {fm[odd]}")
        # stem match (index-free question): the true statement must not be the one that reuses the excerpt's words
        ref = free_ref_for(traj, k)
        if ref is None:
            fails.append(f"clean_unreferable: {nid} step {k} has no unique excerpt")
        else:
            et = content_tokens(ref.split(OPEN_Q, 1)[1])
            rc = [recall(content_tokens(o), et) for o in opts]
            res["stem_recall"][nid] = round(rc[a], 3)
            others = [x for i, x in enumerate(rc) if i != a]
            if rc[a] > 0 and rc[a] >= max(others) + STEM_MATCH_MARGIN:
                fails.append(f"stem_match_tell: {nid} recall {rc[a]:.2f} vs {max(others):.2f}")
        # soft: log-overlap rank (1 = most of its words in the log) and the best-matching step of the true option
        rec_ = [recall(content_tokens(o), log_tok) for o in opts]
        res["overlap_rank"][nid] = 1 + sum(x > rec_[a] for i, x in enumerate(rec_) if i != a)
        step_tok = [content_tokens(kept_step(traj, s["k"])) for s in traj["steps"]]
        t = content_tokens(opts[a])
        res["best_step"][nid] = max(range(n), key=lambda i: (recall(t, step_tok[i]), -abs(i - k))) == k
    res["pass"] = not fails
    return res


def addon_record(traj: dict, src: dict, entry: dict, seed: int = 20261010) -> dict:
    """One add-on output line: {"traj_id", "subset", "teacher", "nodes" (clean), "skips", "checks", "meta"}."""
    chk = check_addon(traj, src, entry, seed)
    nodes, skips = chk.pop("nodes"), chk.pop("skips")
    meta = {"n_steps": len(traj["steps"]), "mistake_step": traj["mistake_step"], "seed": seed,
            "targets": entry.get("targets") or []}
    return {"traj_id": traj["traj_id"], "subset": traj["subset"], "teacher": (src or {}).get("teacher", ""),
            "nodes": nodes, "skips": skips, "checks": chk, "meta": meta}


def merge_v3(traj: dict, line: dict, addon: dict | None, *, render: str = "free", cap: int = STEP_CAP,
             stats: collections.Counter | None = None) -> dict:
    """A v2 tree line (output_record) + its add-on line (addon_record; used only when its checks passed) -> the v3
    line: the v2 nodes, then the clean nodes; render "free" applies free_render to all of them at this cap (the
    student's per-step cap in the builders). The v2 line is not modified."""
    st = stats if stats is not None else collections.Counter()
    if render not in ("index", "free"):
        raise ValueError(f"render {render!r}: index or free")
    nodes = copy.deepcopy(line.get("nodes") or [])
    if addon is not None:
        if (addon.get("checks") or {}).get("pass"):
            have = {x["id"] for x in nodes}
            add = [x for x in addon.get("nodes") or [] if x["id"] not in have]
            nodes += copy.deepcopy(add)
            st["addon_merged"] += 1
            st["clean_nodes"] += len(add)
            st["clean_skips"] += len(addon.get("skips") or [])
        else:
            st["addon_failed"] += 1
    else:
        st["addon_missing"] += 1
    if render == "free":
        nodes = free_render(traj, nodes, cap=cap, stats=st)
    out = dict(line, nodes=nodes)
    out["v3"] = {"render": render, "addon": addon is not None and bool((addon.get("checks") or {}).get("pass")),
                 "cap": cap}
    return out
