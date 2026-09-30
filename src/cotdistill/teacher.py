"""Teacher pipeline: solve with CoT, self-extract sub-questions, answer them, read logprobs.

One item = one case record from `policygen.generate`. For each item we
  1. sample K thinking-mode solutions (CoT + answer-token logprobs + reasoning-token logprobs);
  2. ask the teacher (non-thinking, JSON) to list the yes/no judgements its first CoT made,
     including ones it revised, each anchored by a verbatim quote;
  3. ask the teacher (non-thinking) each sub-question once per trace with that trace's CoT in
     context, and once fresh without any CoT; read P(yes) from the answer-letter logprobs
     (Yes/No letters are shuffled);
  4. (evaluation only) map each sub-question to the rule engine's checkable predicates.
"""
from __future__ import annotations

import json
import random
import re

from .deepseek import DeepSeek
from .policygen import render_prompt
from .readout import label_distribution, locate_span, span_stats, trace_confidence, value_commitment
from .sources import kk_subq_truth

SOLVE_SUFFIX = "\n\nThink it through, then end with exactly one line: ANSWER: <letter>"

EXTRACT_PROMPT = """Below is a decision problem and the reasoning written while solving it.

<problem>
{problem}
</problem>

<reasoning>
{cot}
</reasoning>

List the intermediate yes/no judgements this reasoning made about the case before reaching its decision.
Rules:
- Each item must be a yes/no question about this problem, answerable from the problem text alone (possibly with reasoning).
- Do not restate the final decision itself or ask which option is correct.
- Include judgements the reasoning first got wrong and later corrected; mark them "revised".
- For each item give "question"; "quote": a short verbatim span (at most 30 words) copied from the reasoning where the judgement is made; "status": "direct" or "revised"; "answer": the reasoning's final answer to it, "yes" or "no".
- Use only names and terms that appear in the problem; do not use variables, symbols or abbreviations introduced in the reasoning.
- Do not ask whether an option is correct or satisfies all conditions; a question may name an option only to check one specific statement or condition against it.
- At most 12 items, in the order they appear in the reasoning.
Return JSON: {{"subquestions": [{{"question": "...", "quote": "...", "status": "direct", "answer": "yes"}}]}}"""

ANSWER_PROMPT = """{problem}
{notes}
Answer one intermediate question about the case.
Question: {question}
Options:
A) {a}
B) {b}
Answer with exactly one letter."""

MATCH_PROMPT = """Below are yes/no questions extracted from a solver's reasoning about one case, and a reference list of checkable conditions for the same case.

Reference conditions:
{refs}

Extracted questions:
{qs}

For each extracted question, give the id of the reference condition that asks the same thing, or its exact negation; use null if none matches.
Return JSON: {{"matches": [{{"i": 1, "pid": "some_id or null", "negated": false}}]}}"""


def _compact(tokens: list[dict], k: int = 5) -> list:
    """Keep token, logprob and the top-k alternatives (enough for forks and span stats)."""
    return [[t["token"], t["logprob"], [[a["token"], a["logprob"]] for a in (t.get("top_logprobs") or [])[:k]]]
            for t in tokens]


def _expand(compact: list) -> list[dict]:
    return [{"token": t, "logprob": lp, "top_logprobs": [{"token": a, "logprob": b} for a, b in alts]}
            for t, lp, alts in compact]


def solve(client: DeepSeek, rec: dict, k: int, effort: str | None = None, *, temperature: float | None = None,
          permute: bool = False, efforts: list[str] | None = None, seed: int = 0) -> list[dict]:
    """K thinking-mode solutions. Trace i may see a shuffled option order (permute) and its own
    reasoning effort (efforts, cycled); each trace's distribution is also mapped to outcome keys."""
    rng = random.Random(f"solve-{seed}-{rec['item_id']}")
    traces = []
    for i in range(k):
        order = list(rec["label_order"])
        if permute and i > 0:
            rng.shuffle(order)
        prompt = rec["prompt"] if order == rec["label_order"] else render_prompt(rec, order)
        labels = [chr(65 + j) for j in range(len(order))]
        eff = efforts[i % len(efforts)] if efforts else effort
        resp = client.chat([{"role": "user", "content": prompt + SOLVE_SUFFIX}], thinking=True, logprobs=True,
                           max_tokens=16000, effort=eff, temperature=temperature, tag=f"{rec['item_id']}/solve{i}")
        ch = resp["choices"][0]
        lp = ch.get("logprobs") or {}
        content_lp = lp.get("content") or []
        reas_lp = lp.get("reasoning_content") or []
        dist = label_distribution(content_lp, labels, "ANSWER:")
        traces.append({
            "order": order, "effort": eff, "temperature": temperature,
            "content": ch["message"].get("content") or "",
            "reasoning": ch["message"].get("reasoning_content") or "",
            "finish_reason": ch.get("finish_reason"),
            "dist": dist,
            "dist_outcome": {order[ord(l) - 65]: p for l, p in dist["probs"].items()},
            "conf": trace_confidence(reas_lp),
            "reasoning_lp": _compact(reas_lp),
            "usage": resp.get("usage", {}),
            "fingerprint": resp.get("system_fingerprint"),
        })
    return traces


def extract_subquestions(client: DeepSeek, rec: dict, trace: dict) -> list[dict]:
    msg = EXTRACT_PROMPT.format(problem=rec["prompt"], cot=trace["reasoning"])
    resp = client.chat([{"role": "user", "content": msg}], thinking=False, logprobs=False,
                       max_tokens=2000, json_mode=True, tag=f"{rec['item_id']}/extract")
    try:
        items = json.loads(resp["choices"][0]["message"]["content"]).get("subquestions", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    out = []
    for it in items[:12]:
        if isinstance(it, dict) and it.get("question"):
            out.append({"question": str(it["question"]).strip(), "quote": str(it.get("quote", "")).strip(),
                        "status": it.get("status", "direct"), "stated": str(it.get("answer", "")).lower()})
    return out


TREE_PROMPT = """You will turn a solver's reasoning into a tree of intermediate judgements. Each node is one yes/no question about the problem whose answer the reasoning settled on the way to its decision.

Node types:
- "parse": what a sentence, rule or statement of the problem means (e.g. whether a sentence asserts something or only states a condition).
- "derive": an intermediate fact the reasoning derived from earlier nodes.
- "case": a hypothesis the reasoning tried, asked as "If ..., would ...?" (e.g. whether it leads to a contradiction).
- "verify": a check of a candidate answer against one statement or condition.

Fields per node:
- "id": "n1", "n2", ... in the order the reasoning settles them.
- "type": one of the types above.
- "depends_on": ids of earlier nodes this judgement uses ([] if none).
- "question": a self-contained yes/no question in the problem's own words.
- "answer": "yes" or "no", the reasoning's FINAL belief (after any correction).
- "status": "direct" if the reasoning never believed otherwise; "corrected" if it first believed the opposite and later fixed it.
- "initial_answer": for "corrected" nodes, the first (wrong) belief; otherwise null.
- "quote": a verbatim span (at most 30 words) copied from the reasoning where the final value of this node is settled.

Rules:
1. Use only names and terms from the problem. No variables, symbols or abbreviations introduced by the reasoning, and no questions about the reasoning itself ("did the solver ...").
2. Do not ask the final question, and do not ask whether an option is correct or satisfies all conditions.
3. Include judgements the reasoning got wrong at first (status "corrected") and hypotheses it abandoned (type "case").
4. Choose the natural wording (knight or knave, true or false, asserts or only conditionally states, ...) so that about half of the answers are "no". Never use awkward negations such as "fail to".
5. Between 4 and 15 nodes; skip trivial restatements of the problem text.

Example.
<problem>
A says: "B is a knave." B says: "A and C are both knights." C says: "B is a knight." Who is a knight?
</problem>
<reasoning>
Let A,B,C true=knight. A = not B. B = A and C. C says B knight, so C = not B? wait, C says B is a knight, so C = B. Case B true: then A false, but B = A and C needs A true, contradiction. So B false. Then A true, C = B = false. Check B: A and C = T and F = F, so B's statement is false, consistent with B knave.
</reasoning>
{{"nodes": [
 {{"id": "n1", "type": "parse", "depends_on": [], "question": "Is A a knight exactly when B is a knave?", "answer": "yes", "status": "direct", "initial_answer": null, "quote": "A = not B."}},
 {{"id": "n2", "type": "parse", "depends_on": [], "question": "Is B a knight exactly when A and C are both knights?", "answer": "yes", "status": "direct", "initial_answer": null, "quote": "B = A and C."}},
 {{"id": "n3", "type": "parse", "depends_on": [], "question": "Do C and B have opposite roles?", "answer": "no", "status": "corrected", "initial_answer": "yes", "quote": "C says B is a knight, so C = B."}},
 {{"id": "n4", "type": "case", "depends_on": ["n1", "n2"], "question": "If B were a knight, would the statements contradict each other?", "answer": "yes", "status": "direct", "initial_answer": null, "quote": "Case B true: then A false, but B = A and C needs A true, contradiction."}},
 {{"id": "n5", "type": "derive", "depends_on": ["n4"], "question": "Is B a knave?", "answer": "yes", "status": "direct", "initial_answer": null, "quote": "So B false."}},
 {{"id": "n6", "type": "derive", "depends_on": ["n1", "n5"], "question": "Is A a knave?", "answer": "no", "status": "direct", "initial_answer": null, "quote": "Then A true"}},
 {{"id": "n7", "type": "derive", "depends_on": ["n3", "n5"], "question": "Is C a knight?", "answer": "no", "status": "direct", "initial_answer": null, "quote": "C = B = false."}},
 {{"id": "n8", "type": "verify", "depends_on": ["n6", "n7"], "question": "With A a knight and C a knave, is B's statement true?", "answer": "no", "status": "direct", "initial_answer": null, "quote": "A and C = T and F = F, so B's statement is false"}}
]}}

Now do the same for this problem and reasoning. Return only JSON: {{"nodes": [...]}}

<problem>
{problem}
</problem>
<reasoning>
{cot}
</reasoning>"""

TREE_TYPES = {"parse", "derive", "case", "verify"}
_META = re.compile(r"\b(the reasoning|the solver|the solution process|reasoning step|did the model)\b", re.I)
_AWKWARD = re.compile(r"\bfail(?:s|ed)? to\b", re.I)


def parse_tree(raw: str) -> list[dict]:
    """Validated nodes from the tree-extraction JSON: known types, yes/no answers, dependencies only on
    earlier kept nodes, no meta questions about the reasoning and no "fail to" negations."""
    try:
        nodes = json.loads(raw).get("nodes", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    out, seen = [], set()
    for n in nodes[:15]:
        if not isinstance(n, dict):
            continue
        q, ans, typ = str(n.get("question", "")).strip(), str(n.get("answer", "")).lower().strip(), n.get("type")
        nid = str(n.get("id", "")).strip()
        if not q or ans not in ("yes", "no") or typ not in TREE_TYPES or not nid or nid in seen:
            continue
        if _META.search(q) or _AWKWARD.search(q):
            continue
        status = "corrected" if n.get("status") == "corrected" else "direct"
        init = str(n.get("initial_answer") or "").lower().strip()
        seen.add(nid)
        out.append({"id": nid, "type": typ, "depends_on": [d for d in n.get("depends_on") or [] if d in seen],
                    "question": q, "stated": ans, "status": status,
                    "initial": init if status == "corrected" and init in ("yes", "no") else None,
                    "quote": str(n.get("quote", "")).strip()})
    return out


def extract_tree(client: DeepSeek, rec: dict, trace: dict) -> list[dict]:
    """Reasoning tree of typed yes/no nodes from one CoT (passed as text: DeepSeek drops earlier
    reasoning_content in multi-turn chats). The fixed instructions and example come first so the prefix
    is cached across items."""
    resp = client.chat([{"role": "user", "content": TREE_PROMPT.format(problem=rec["prompt"], cot=trace["reasoning"])}],
                       thinking=False, logprobs=False, max_tokens=4000, json_mode=True, tag=f"{rec['item_id']}/tree")
    return parse_tree(resp["choices"][0]["message"].get("content") or "")


NEGATE_PROMPT = """Below is a problem and some yes/no questions about it. Rewrite each question so that its correct answer is the opposite, by asking about the negation of what it asks (for example "Is Alice a knight?" -> "Is Alice a knave?", "Does the premise imply X?" -> "Does the premise fail to imply X?"). Keep every name and term; add no new information; keep each question natural and self-contained.

<problem>
{problem}
</problem>

Questions:
{qs}

Return JSON: {{"rewritten": ["...", "..."]}} with one entry per question, in order."""


def rebalance_polarity(client: DeepSeek, rec: dict, subqs: list[dict], rng: random.Random) -> None:
    """Rewrite a subset of sub-questions into their negations so that about half have the answer "no".

    Extracted judgements are mostly phrased affirmatively (about 90% "yes" on K&K); left alone, a student
    could fit the sub-questions with a label prior. Flipped items keep the original under "orig_question"."""
    stated = [s for s in subqs if s.get("stated") in ("yes", "no")]
    n_yes = sum(s["stated"] == "yes" for s in stated)
    surplus, major = (n_yes - len(stated) // 2, "yes") if n_yes * 2 > len(stated) else \
        (len(stated) - n_yes - len(stated) // 2, "no")
    pick = rng.sample([s for s in stated if s["stated"] == major], max(0, surplus))
    if not pick:
        return
    qs = "\n".join(f"{i}. {s['question']}" for i, s in enumerate(pick, start=1))
    resp = client.chat([{"role": "user", "content": NEGATE_PROMPT.format(problem=rec["prompt"], qs=qs)}],
                       thinking=False, logprobs=False, max_tokens=2000, json_mode=True, tag=f"{rec['item_id']}/negate")
    try:
        new = json.loads(resp["choices"][0]["message"]["content"]).get("rewritten", [])
    except (json.JSONDecodeError, AttributeError):
        return
    if len(new) != len(pick):
        return
    for s, q in zip(pick, new):
        q = str(q).strip()
        if q and q != s["question"]:
            s["orig_question"], s["question"] = s["question"], q
            s["stated"] = "no" if s["stated"] == "yes" else "yes"


NOTES = """
Your earlier reasoning notes on this case:
<reasoning>
{cot}
</reasoning>
"""


def answer_subquestion(client: DeepSeek, rec: dict, trace: dict | None, question: str, rng: random.Random,
                       tag: str) -> dict:
    """P(yes) for one sub-question; with the trace's CoT in context, or fresh if trace is None."""
    yes_first = rng.random() < 0.5
    a, b = ("Yes", "No") if yes_first else ("No", "Yes")
    notes = NOTES.format(cot=trace["reasoning"]) if trace is not None else ""
    msg = ANSWER_PROMPT.format(problem=rec["prompt"], notes=notes, question=question, a=a, b=b)
    resp = client.chat([{"role": "user", "content": msg}], thinking=False, logprobs=True,
                       max_tokens=3, tag=tag)
    tokens = (resp["choices"][0].get("logprobs") or {}).get("content") or []
    d = label_distribution(tokens, ["A", "B"], marker=None)
    out = {"p_yes": None, "mass": d["mass"], "missing": d["missing"], "yes_first": yes_first, "bound": d["bound"]}
    out["p_yes"] = resolve_p_yes(out, d["probs"])
    return out


def resolve_p_yes(ans: dict, probs: dict | None = None) -> float | None:
    """P(yes) from a sub-question answer. A letter outside the top-20 is censored at a tiny bound, so when
    only the "no" letter was found P(yes) is ~0 (symmetric with a missing "no" giving ~1); None when
    neither letter was found."""
    yes_lab = "A" if ans.get("yes_first") else "B"
    no_lab = "B" if yes_lab == "A" else "A"
    missing = set(ans.get("missing") or [])
    if probs:
        return probs.get(yes_lab, 0.0)
    if ans.get("p_yes") is not None:
        return ans["p_yes"]
    if (ans.get("mass") or 0) > 0 and yes_lab in missing and no_lab not in missing:
        return 0.0
    return None


GENERIC_PROMPT = """Below is a decision problem.

<problem>
{problem}
</problem>

Write {n} yes/no questions about the specific entities, statements, facts, rules or numbers of THIS problem (for example what a named person says, what a particular premise or clause states, or how two stated values compare). They must be answerable by reading the problem text, without solving it. Make roughly half of them have the answer "no" (for example by misquoting a statement or swapping a name or value).
Do not ask about the general framing or rules that any problem of this kind shares (for example what knights or knaves do, or what the options are), do not ask about the final decision or how to decide it, and do not reason about the solution.
Return JSON: {{"questions": ["...", "..."]}}"""


def generic_subquestions(client: DeepSeek, rec: dict, n: int) -> list[str]:
    """Matched-count control: yes/no questions written from the problem only (no CoT)."""
    if n <= 0:
        return []
    resp = client.chat([{"role": "user", "content": GENERIC_PROMPT.format(problem=rec["prompt"], n=n)}],
                       thinking=False, logprobs=False, max_tokens=1500, json_mode=True,
                       tag=f"{rec['item_id']}/generic")
    try:
        qs = json.loads(resp["choices"][0]["message"]["content"]).get("questions", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    return [str(q).strip() for q in qs if str(q).strip()][:n]


def match_predicates(client: DeepSeek, rec: dict, subqs: list[dict]) -> list[dict]:
    if not subqs or not rec.get("predicates"):
        return [{"pid": None, "negated": False} for _ in subqs]
    refs = "\n".join(f"{p['pid']}: {p['question']}" for p in rec["predicates"])
    qs = "\n".join(f"{i}. {s['question']}" for i, s in enumerate(subqs, start=1))
    resp = client.chat([{"role": "user", "content": MATCH_PROMPT.format(refs=refs, qs=qs)}], thinking=False,
                       logprobs=False, max_tokens=1500, json_mode=True, tag=f"{rec['item_id']}/match")
    try:
        ms = json.loads(resp["choices"][0]["message"]["content"]).get("matches", [])
    except (json.JSONDecodeError, AttributeError):
        ms = []
    valid = {p["pid"] for p in rec["predicates"]}
    out = [{"pid": None, "negated": False} for _ in subqs]
    for m in ms:
        try:
            i = int(m.get("i")) - 1
        except (TypeError, ValueError):
            continue
        pid = m.get("pid")
        if 0 <= i < len(out) and pid in valid:
            out[i] = {"pid": pid, "negated": bool(m.get("negated", False))}
    return out


def run_item(client: DeepSeek, rec: dict, k: int = 2, seed: int = 0, effort: str | None = None,
             random_matched: bool = True, extraction: str = "tree", rebalance: bool = False, **solve_kw) -> dict:
    """Solve, then extract sub-questions: "tree" (typed reasoning-tree nodes, default) or "flat" (the
    first pipeline's list of judgements; `rebalance` re-enables its negation rewrite, which produced
    garbled "fail to" questions and wrong labels, so it is off)."""
    rng = random.Random(f"{seed}-{rec['item_id']}")
    traces = solve(client, rec, k, effort=effort, seed=seed, **solve_kw)
    if extraction == "tree":
        res = {"item": {k2: v for k2, v in rec.items() if k2 != "prompt"}, "prompt": rec["prompt"],
               "traces": traces, "subquestions": [], "random_subquestions": []}
        return tree_item(client, res, seed=seed, random_matched=random_matched)
    subqs = extract_subquestions(client, rec, traces[0]) if traces[0]["reasoning"] else []
    if rebalance:
        rebalance_polarity(client, rec, subqs, rng)
    reas0 = _expand(traces[0]["reasoning_lp"])
    truth = {p["pid"]: p["truth"] for p in rec.get("predicates", [])}
    program_truth = rec.get("domain") == "knights_knaves"     # regex check; the LLM matcher conflates claims
    matches = [{"pid": None, "negated": False} for _ in subqs] if program_truth else match_predicates(client, rec, subqs)
    for j, sq in enumerate(subqs):
        sq["answers"] = [answer_subquestion(client, rec, tr, sq["question"], rng, tag=f"{rec['item_id']}/sq{j}.t{i}")
                         for i, tr in enumerate(traces)]
        sq["answer_nocot"] = answer_subquestion(client, rec, None, sq["question"], rng, tag=f"{rec['item_id']}/sq{j}.fresh")
        span = locate_span(reas0, sq["quote"])
        sq["span"] = list(span) if span else None
        sq["span_stats"] = span_stats(reas0, span)
        m = matches[j] if j < len(matches) else {"pid": None, "negated": False}
        sq["match"] = m
        if program_truth:
            sq["truth"] = kk_subq_truth(sq["question"], rec)
        else:
            t = truth.get(m["pid"]) if m["pid"] else None
            sq["truth"] = (not t if m["negated"] else t) if t is not None else None
    randoms = []
    if random_matched:
        for j, q in enumerate(generic_subquestions(client, rec, len(subqs))):
            randoms.append({"question": q, "answer_nocot": answer_subquestion(client, rec, None, q, rng,
                                                                              tag=f"{rec['item_id']}/rq{j}.fresh")})
    return {"item": {k2: v for k2, v in rec.items() if k2 != "prompt"}, "prompt": rec["prompt"],
            "traces": traces, "subquestions": subqs, "random_subquestions": randoms}


def tree_item(client: DeepSeek, res: dict, seed: int = 0, random_matched: bool = True) -> dict:
    """Re-extract the sub-questions of an already solved item as a reasoning tree (no new solve).

    Each node is answered with the CoT in context and without it, anchored to its quote in the CoT, given
    the teacher's value-commitment confidence there, and program truth for K&K. The matched control keeps
    the item's existing problem-only questions, topped up if the tree has more nodes."""
    rec = dict(res["item"], prompt=res["prompt"])
    rng = random.Random(f"tree-{seed}-{rec['item_id']}")
    tr = res["traces"][0]
    nodes = extract_tree(client, rec, tr) if tr.get("reasoning") else []
    toks = _expand(tr["reasoning_lp"])
    for j, nd in enumerate(nodes):
        nd["answers"] = [answer_subquestion(client, rec, tr, nd["question"], rng, tag=f"{rec['item_id']}/node{j}.t0")]
        nd["answer_nocot"] = answer_subquestion(client, rec, None, nd["question"], rng, tag=f"{rec['item_id']}/node{j}.fresh")
        span = locate_span(toks, nd["quote"])
        nd["span"] = list(span) if span else None
        nd["span_stats"] = span_stats(toks, span)
        nd["commit"] = value_commitment(toks, span)
        nd["truth"] = kk_subq_truth(nd["question"], rec) if rec.get("domain") == "knights_knaves" else None
    randoms = list(res.get("random_subquestions", []))
    if random_matched and len(randoms) < len(nodes):
        for j, q in enumerate(generic_subquestions(client, rec, len(nodes) - len(randoms))):
            randoms.append({"question": q, "answer_nocot": answer_subquestion(client, rec, None, q, rng,
                                                                              tag=f"{rec['item_id']}/rq{len(randoms) + j}.fresh")})
    return dict(res, subquestions=nodes, random_subquestions=randoms[:max(len(nodes), 1)], extraction="tree")