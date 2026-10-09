"""Tier-1 public failure-attribution data (notes/next_round/failure_attribution/tier1_public.md; builder
scripts/build_fa_public.py): public multi-agent failure logs with a labelled decisive step, rendered in the Who&When
eval layout of scripts/build_whowhen_eval.py, with yes/no attribution nodes for Tree Forcing. CPU only, no model.

Two stages, because the readers need pandas (python3's user site) and the token counts need tokenizers (the sev venv):
  normalize  read every raw source (untrusted data: parsed, never executed) into one record per trajectory
             {uid, origin, subset, benchmark, task, answer, steps [{agent, text}], dec, gold_agent, mode, accepted,
             split, ids}; drops are counted by reason. Step texts are pre-cut to PRECUT characters with cut() (lossless
             for any later per-step cap <= PRECUT, see precut_ok).
  build      held-out sets, dedup, splits, per-trajectory step caps fitted to a token budget, items and nodes.

Sources and their decisive-step conventions (checked on the data, tier1_public.md):
  wwpro         Who&When Pro text split. Rendered steps = the trajectory without its user entry and final_answer
                entries; debate / dylan rounds are flattened to one step per turn. Gold: smolagents / alfagent the
                action whose step_number equals ground_truth.step, or the planning entry when it is 0 (traces with a
                second planning entry are dropped: their numbering is ambiguous); magentic-one the (round, position)
                of "r.p"; mathchat / macnet / metagpt / debate / dylan the entry flagged is_injected, which equals the
                ground-truth coordinate in every trace (asserted). Debate / dylan traces with two injected agents are
                dropped. alfagent is rendered without its reasoning field (non-null exactly up to the injected step).
  aftraj        AFTraj-2K unsafe split. mistake_step indexes turns with turns[0] = the user task; turns[0] is not
                rendered (the Task line has it), so the rendered index is mistake_step - 1. Tool-call ids are not
                rendered (call_verify_shortcut / _misread mark injected steps). splits_test.json test_unsafe -> test.
  stepfinder    StepFinder regenerated training set (Algorithm-Generated, Hand-Crafted): Who&When convention,
                0-based over history (is_mistake equals mistake_step in every file; never rendered).
  agentracer    AgenTracer data-v1.0.0 (agentic, coding, math; train -> pool, test -> test): mistake_step is the value
                of the step field (1-based in agentic, 0-based in coding / math).
  traceelephant TraceElephant (eval only): mistake_step is the 1-based step_id.
  correct       CORRECT-Error (opt-in, build_fa_public.py --add-sources): history[0] is the user prompt; mistake_step
                indexes history, so the rendered index (history[1:]) is mistake_step - 1 (see "new sources" below).
  aegis         Aegis (opt-in): agent-level labels only (the single faulty agent), no step items.
Never rendered: is_injected, is_replayed, is_live, phase, injection_via, ledger, accepted_predictions, is_mistake,
mistake_reason, tool-call ids.
"""
from __future__ import annotations

import collections
import hashlib
import io
import json
import random
import re

from .treeplan import node_depths

SOURCE = "fa_public"
TREE_SOURCE = "fa_public_labels"
HEADER = "The log below comes from a multi-agent system that failed to solve its task."
Q_STEP = ("Question: Which step contains the decisive error, i.e. the earliest mistake whose correction would have "
          "let the system solve the task?")
Q_AGENT = "Question: Which agent made the decisive error that caused the failure?"
SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"   # Intern-Decision's ANSWER_SYMBOLS
MAX_STEPS = len(SYMBOLS)
TASK_CAP, GT_CAP, PRECUT = 1500, 300, 6000
GT_PREFIX = "Correct final answer (for reference): "
NOT_AGENTS = {"user", "human", "environment"}
SINGLE_AGENT = "Agent"

# Who&When Pro taxonomy (taxonomy.yaml): mode code -> (name, gloss used in node questions); category letter -> gloss
MODES = {
    "P.1": ("Visual misidentification", "a visual misidentification (misreading an image)"),
    "P.2": ("Spatial/grounding error", "a spatial grounding error (targeting the wrong region of an image or screen)"),
    "R.1": ("Hallucination", "a hallucination (a claim not grounded in anything the agents observed)"),
    "R.2": ("Reasoning error", "a reasoning error (a wrong conclusion drawn from correct information, or a broken "
                               "derivation step)"),
    "R.3": ("Numerical/calculation error", "a numerical or calculation error (arithmetic, counting, units or "
                                           "rounding)"),
    "R.4": ("Task misunderstanding", "a misunderstanding of the task (answering a different question, or a wrong "
                                     "kind of solution for it)"),
    "PL.1": ("Ineffective planning", "an ineffective plan (an unsound strategy or decomposition)"),
    "PL.2": ("Goal drift", "goal drift (moving away from the original objective)"),
    "A.1": ("Tool parameter/invocation error", "a wrong tool call (wrong arguments or a malformed invocation)"),
    "A.2": ("Output format/syntax error", "an output format or syntax error"),
    "A.3": ("Premature termination", "a premature termination (stopping before the task was done)"),
    "A.4": ("Repetitive/looping behavior", "repetitive looping (repeating actions without progress)"),
    "V.1": ("Context/memory loss", "a loss of context (forgetting or ignoring earlier information)"),
    "V.2": ("Inadequate or incorrect verification", "an inadequate or wrong verification (a missing check, or "
                                                    "confirming a wrong result)"),
    "C.1": ("Delegation/orchestration error", "a delegation error (a subtask given to the wrong agent or split "
                                              "badly)"),
    "C.2": ("Communication failure", "a communication failure (information withheld, or another agent's input "
                                     "ignored)"),
    "C.3": ("Over-reliance on other agents", "over-reliance on another agent (giving up a sound answer for another "
                                             "agent's worse one)"),
}
CATEGORIES = {
    "P": "a perception error (misreading an image or a screen)",
    "R": "a reasoning error in the broad sense (a hallucination, a wrong inference, a calculation error or a "
         "misunderstood task)",
    "PL": "a planning error (an unsound plan, or drifting away from the goal)",
    "A": "an action error (a wrong tool call, a malformed output, stopping too early, or looping)",
    "V": "a verification or memory error (lost context, or a missing or wrong check)",
    "C": "a coordination error between agents (delegation, communication, or over-reliance on another agent)",
}
TEXT_MODES = [m for m in MODES if not m.startswith(("P.", "PL.2"))]        # the modes the text split uses


def category(mode: str | None) -> str | None:
    return mode.split(".")[0] if mode else None


# ------------------------------------------------------------------------------------------------- text helpers
def cut(text: str, cap: int) -> str:
    """scripts/build_whowhen_eval.cut: the head (2/3 of cap) and the tail of a text longer than cap."""
    text = str(text).strip()
    if len(text) <= cap:
        return text
    head = cap * 2 // 3
    return text[:head] + " [...] " + text[-(cap - head):]


def precut_ok(cap: int) -> bool:
    """cut(cut(x, PRECUT), cap) == cut(x, cap) for every x when cap <= PRECUT (head and tail of the smaller cut lie
    inside the pre-cut's head and tail)."""
    return cap <= PRECUT


def sanitize(text) -> str:
    """Step and task text as rendered: str, no carriage returns, and no line that opens with 'Options:' (the trainer
    finds the answer options at the first '\\nOptions:' of a prompt, treeplan.problem_no_options)."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\nOptions:", "\nOptions :")


def agent_name(x) -> str:
    """Display name of an agent: brackets and newlines removed (the option and step lines are 'Step i [Name]')."""
    s = re.sub(r"[\[\]\n\r\t]", " ", str(x or "unknown"))
    return " ".join(s.split()) or "unknown"


def norm_task(text: str) -> str:
    return " ".join(str(text).lower().split())


UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|\b[0-9a-f]{64}\b")


def uuids(text: str) -> list[str]:
    return sorted(set(UUID.findall(str(text).lower())))


def _j(x):
    return json.loads(x) if isinstance(x, str) else x


def _rec(uid, origin, subset, benchmark, task, answer, steps, dec, gold_agent, *, mode=None, accepted=None,
         split="pool", ids=(), single=False, flags=()) -> dict:
    steps = [{"agent": agent_name(a), "text": cut(sanitize(t), PRECUT)} for a, t in steps]
    return {"uid": uid, "origin": origin, "subset": subset, "benchmark": benchmark,
            "task": cut(sanitize(task), 4 * TASK_CAP),
            "answer": None if answer in (None, "") else cut(sanitize(answer), 4 * GT_CAP), "steps": steps,
            "dec": dec, "gold_agent": None if gold_agent is None else agent_name(gold_agent), "mode": mode,
            "accepted": accepted, "split": split, "ids": sorted(set(ids)), "single_agent": single,
            "flags": list(flags)}


# ------------------------------------------------------------------------------------------------- readers
def read_wwpro(path: str, counts: collections.Counter):
    """Who&When Pro text.jsonl -> records (see the module docstring for the index conventions)."""
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            fw, bench = r["framework"], r["benchmark"]
            task, tr, gt, ex = _j(r["task"]), _j(r["trajectory"]), _j(r["ground_truth"]), _j(r["extras"]) or {}
            agents = ex.get("agents") or []
            roles = [str(a.get("role")) for a in agents]
            unique = len({x.lower() for x in roles}) == len(roles)
            disp = {a.get("id"): (a.get("role") if unique else a.get("id")) for a in agents}
            name = lambda i: disp.get(i, i)                                     # noqa: E731
            steps, dec, gold, mode, accepted, flags = [], None, None, gt.get("mode"), None, []
            single = fw in ("smolagents", "alfagent")
            if single:
                ent = [s for s in tr if s.get("kind") in ("action", "planning")]
                if sum(s["kind"] == "planning" for s in ent) > 1:
                    counts["wwpro/drop/second_planning_step"] += 1
                    continue
                for s in ent:
                    if s["kind"] == "planning":
                        text = s.get("plan")
                    elif fw == "alfagent":           # reasoning omitted: non-null exactly up to the injected step
                        text = f"Action: {s.get('action')}\nObservation: {s.get('observation')}"
                    else:
                        parts = [str(s.get("reasoning") or "").strip()]
                        if s.get("code"):
                            parts.append(f"Code:\n```py\n{s['code']}\n```")
                        if s.get("observation"):
                            parts.append(str(s["observation"]).strip())
                        if s.get("error"):
                            parts.append(f"Error: {sanitize(s['error'])}")
                        text = "\n".join(p for p in parts if p)
                    if (s["kind"] == "planning" and gt.get("step") == 0) or (
                            s["kind"] == "action" and s.get("step_number") == gt.get("step")):
                        dec = len(steps)
                    steps.append((SINGLE_AGENT, text))
            elif fw == "magentic-one":
                rr, pp = (int(x) for x in str(gt["step"]).split("."))
                for s in tr:
                    if s.get("kind") != "agent":
                        continue
                    if (s.get("round"), s.get("position")) == (rr, pp):
                        if "injected" not in str(s.get("phase")) or s.get("agent") != gt.get("agent"):
                            flags.append("gt_mismatch")
                        dec = len(steps)
                    steps.append((name(s.get("agent")), s.get("output")))
                gold = name(gt.get("agent"))
            elif fw in ("mathchat", "macnet", "metagpt"):
                for s in tr:
                    if "agent_id" not in s:
                        continue
                    if s.get("is_injected"):
                        dec = len(steps)
                        if s["agent_id"] != gt.get("agent") or (fw == "metagpt" and s.get("stage") != gt.get("stage")) \
                                or (fw == "mathchat" and (s.get("round"), s.get("position")) !=
                                    (gt.get("round"), gt.get("position"))):
                            flags.append("gt_mismatch")
                    steps.append((name(s["agent_id"]), s.get("output")))
                gold = name(gt.get("agent"))
                if gt.get("accepted_predictions"):        # macnet: alternates; the mode at the gold coordinate
                    accepted = gt["accepted_predictions"]
                    coord = f"{gt.get('round')}.{gt.get('position')}"
                    at = [a.get("mode") for a in accepted if a.get("step_coord") == coord]
                    mode = at[0] if at else None
            elif fw in ("debate", "dylan"):
                if len(gt.get("agents") or []) != 1:
                    counts[f"wwpro/drop/{fw}_two_injected_agents"] += 1
                    continue
                for s in tr:
                    if s.get("kind") != "round":
                        continue
                    for u in _j(s["turns"]):
                        if u.get("is_injected"):
                            dec = len(steps)
                            if (s.get("round"), u.get("agent_id")) != (gt.get("round"), gt["agents"][0]):
                                flags.append("gt_mismatch")
                        steps.append((name(u.get("agent_id")), u.get("output")))
                gold = name(gt["agents"][0])
            else:
                counts[f"wwpro/drop/unknown_framework_{fw}"] += 1
                continue
            if dec is None or "gt_mismatch" in flags:
                counts["wwpro/drop/" + ("no_gold_step" if dec is None else "gt_mismatch")] += 1
                continue
            yield _rec(r["id"], "wwpro", fw, bench, task.get("query", ""), task.get("answer"), steps, dec,
                       None if single else gold, mode=mode, accepted=accepted, ids=uuids(task.get("query", "")),
                       single=single)


def _aftraj_action(action: str) -> str:
    """Tool calls of an AFTraj turn without their ids (an id like call_verify_shortcut marks an injected step)."""
    if not action:
        return ""
    try:
        calls = json.loads(action)
    except (ValueError, TypeError):
        return "Action: " + re.sub(r'"(?:call_)?id"\s*:\s*"[^"]*",?\s*', "", str(action))
    out = []
    for c in calls if isinstance(calls, list) else [calls]:
        if isinstance(c, dict):
            out.append(f"Action: {c.get('name')}({c.get('arguments', '')})")
        else:
            out.append(f"Action: {c}")
    return "\n".join(out)


def _aftraj_content(content: str) -> str:
    """Tool results of an environment turn as 'Result of <tool>: <result>' lines (call ids dropped); other text as
    is."""
    s = str(content or "")
    if s.startswith("[{") and '"result"' in s:
        try:
            res = json.loads(s)
            return "\n".join(f"Result of {x.get('name')}: {x.get('result')}" for x in res if isinstance(x, dict))
        except ValueError:
            pass
    return s


def read_aftraj(unsafe_parquet: str, splits_json: str, counts: collections.Counter):
    import pandas as pd                       # python3's user site (stage normalize)
    test = set(json.load(open(splits_json))["test_unsafe"])
    df = pd.read_parquet(unsafe_parquet)
    for _, r in df.iterrows():
        turns = list(r["turns"])
        k = int(r["mistake_step"])
        if not 1 <= k < len(turns) or turns[0]["role"] != "user" or turns[k]["role"] != r["mistake_agent"]:
            counts["aftraj/drop/index"] += 1
            continue
        steps = []
        for t in turns[1:]:
            parts = []
            if t.get("thought"):
                parts.append("Thought: " + str(t["thought"]).strip())
            c = _aftraj_content(t.get("content"))
            if c.strip():
                parts.append(c.strip())
            a = _aftraj_action(t.get("action"))
            if a:
                parts.append(a)
            steps.append((t["role"], "\n".join(parts)))
        cid = str(r["conv_id"])
        yield _rec(cid, "aftraj", str(r["domain"]), str(r["unsafe_source"]), r["task"], r["gold_answer"], steps, k - 1,
                   r["mistake_agent"], split="test" if cid in test else "pool", ids=uuids(r["task"]))


def ww_agent(step: dict) -> str:
    """scripts/build_whowhen_eval.agent_of."""
    name = step.get("name") or step.get("role") or "unknown"
    return str(name).split(" (")[0].strip()


def _anorm(a: str) -> str:
    return str(a).lower().replace("_", "").replace(" ", "")


def read_stepfinder(root: str, counts: collections.Counter):
    import glob
    import os
    for sub in ("Algorithm-Generated", "Hand-Crafted"):
        files = sorted(glob.glob(os.path.join(root, sub, "train", "*.json")))
        for fn in files:
            with open(fn) as f:
                x = json.load(f)
            h = x["history"]
            k = int(x["mistake_step"])
            if not 0 <= k < len(h):
                counts["stepfinder/drop/index"] += 1
                continue
            flags = [] if _anorm(ww_agent(h[k])) == _anorm(x["mistake_agent"]) else ["agent_mismatch"]
            uid = f"{sub[:3].lower()}-{os.path.basename(fn)[:-5]}"
            yield _rec(uid, "stepfinder", "alg" if sub.startswith("Alg") else "hc", "whowhen_like", x["question"],
                       x.get("ground_truth"), [(ww_agent(s), s.get("content", "")) for s in h], k, ww_agent(h[k]),
                       ids=uuids(json.dumps(x)), flags=flags)


def read_agentracer(zip_path: str, counts: collections.Counter):
    import zipfile
    z = zipfile.ZipFile(zip_path)
    for dom in ("agentic", "coding", "math"):
        for sp in ("train", "test"):
            with z.open(f"agentracer-data-v1.0.0/{dom}/{sp}.jsonl") as f:
                for i, line in enumerate(io.TextIOWrapper(f, encoding="utf-8")):
                    x = json.loads(line)
                    h = _j(x["history"])
                    hit = [j for j, s in enumerate(h) if str(s.get("step")) == str(x["mistake_step"]).strip()]
                    if len(hit) != 1:
                        counts[f"agentracer/drop/index_{sp}"] += 1
                        continue
                    k = hit[0]
                    names = [str(s.get("name") or s.get("role") or "assistant") for s in h]
                    single = len({n for n in names if n.lower() not in NOT_AGENTS}) < 2
                    flags = [] if names[k] == str(x["mistake_agent"]) else ["agent_mismatch"]
                    yield _rec(f"{dom}-{sp}-{i:05d}", "agentracer", dom, str(x.get("error_source") or dom),
                               x["question"], x.get("ground_truth"),
                               [(n, s.get("content", "")) for n, s in zip(names, h)],
                               k, None if single else names[k], split="test" if sp == "test" else "pool",
                               ids=uuids(str(x.get("question_ID")) + " " + x["question"]), single=single, flags=flags)


def _te_text(s: dict) -> str:
    out = s.get("output") or {}
    msg = (out.get("choices") or [{}])[0].get("message", {}) if isinstance(out, dict) else {}
    parts = []
    c = msg.get("content")
    if c not in (None, "", "None"):
        parts.append(str(c).strip())
    calls = s.get("tool_logs") or []
    for c in calls if isinstance(calls, list) else []:
        fn = c.get("function", {}) if isinstance(c, dict) else {}
        if fn:
            parts.append(f"Action: {fn.get('name')}({fn.get('arguments', '')})")
    return "\n".join(parts)


def _te_observation(next_step: dict | None) -> str:
    """The tool results a swe-agent step produced: the trailing 'tool' messages of the next step's input."""
    if not next_step:
        return ""
    msgs = (next_step.get("input") or {}).get("messages") or []
    obs = []
    for m in reversed(msgs):
        if m.get("role") != "tool":
            break
        c = m.get("content")
        if isinstance(c, list):
            c = "\n".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
        obs.append(str(c))
    return "\n".join(reversed(obs)).strip()


def read_traceelephant(zip_path: str, counts: collections.Counter):
    import os
    import zipfile
    z = zipfile.ZipFile(zip_path)
    runs = sorted({os.path.dirname(n) for n in z.namelist() if n.endswith("/trace_metadata.json")})
    for d in runs:
        system = d.split("/")[1]
        md = json.loads(z.read(d + "/trace_metadata.json"))
        sr = json.loads(z.read(d + "/step_records.json"))
        m = re.fullmatch(r"(?:step\s*)?(\d+)", str(md.get("mistake_step")).strip(), re.I)
        if not m:
            counts["traceelephant/drop/no_step_label"] += 1
            continue
        hit = [j for j, s in enumerate(sr) if str(s.get("step_id")) == m.group(1)]
        if len(hit) != 1:
            counts["traceelephant/drop/index"] += 1
            continue
        k = hit[0]
        steps = []
        for j, s in enumerate(sr):
            text = _te_text(s)
            if system.startswith("swe-agent"):
                ob = _te_observation(sr[j + 1] if j + 1 < len(sr) else None)
                if ob:
                    text += "\nObservation: " + ob
            steps.append((s.get("agent_name"), text))
        names = [agent_name(s.get("agent_name")) for s in sr]
        flags = [] if names[k] == agent_name(md.get("mistake_agent")) else ["agent_mismatch"]
        yield _rec(d.split("/")[-1], "traceelephant", system, system.split("-runs-")[-1],
                   md.get("task_instruction", ""), md.get("ground_truth"), steps, k, md.get("mistake_agent"),
                   split="test", ids=uuids(str(md.get("task_id"))), flags=flags)


def read_whowhen(root: str) -> list[dict]:
    """The Who&When benchmark (test only): question, question_ID and subset of every trajectory (dedup references)."""
    import glob
    import os
    out = []
    for sub, tag in (("Algorithm-Generated", "alg"), ("Hand-Crafted", "hc")):
        for fn in sorted(glob.glob(os.path.join(root, "Who&When", sub, "*.json"))):
            with open(fn) as f:
                x = json.load(f)
            out.append({"subset": tag, "file": os.path.basename(fn), "question": str(x.get("question", "")),
                        "question_ID": str(x.get("question_ID", "")).lower()})
    return out


# ------------------------------------------------------------------------------------------------- 8-gram overlap
TOK = re.compile(r"[a-z0-9]+")


def grams(text: str, n: int = 8) -> set:
    w = TOK.findall(str(text).lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


class OverlapIndex:
    """Reference texts (held-out tasks) with their 8-grams. Boilerplate = 8-grams that occur in at least `df` distinct
    reference texts (benchmark instructions shared by many tasks); they are ignored on both sides. A query overlaps a
    reference when the shared non-boilerplate 8-grams are at least `thr` of the query's (containment of the query) or
    of the reference's (containment of the reference). Texts with fewer than `min_grams` grams after the filter match
    only by equal normalized text."""

    def __init__(self, refs: list[tuple[str, str]], df: int = 5, thr: float = 0.5, min_grams: int = 3):
        self.thr, self.min_grams = thr, min_grams
        texts = {}
        for key, text in refs:
            texts.setdefault(norm_task(text), []).append(key)
        freq: collections.Counter = collections.Counter()
        gs = {}
        for t in texts:
            gs[t] = grams(t)
            freq.update(gs[t])
        self.boiler = {g for g, c in freq.items() if c >= df}
        self.keys = list(texts.values())
        self.exact = {t: i for i, t in enumerate(texts)}
        self.size, self.index = [], collections.defaultdict(list)
        for i, t in enumerate(texts):
            g = gs[t] - self.boiler
            self.size.append(len(g))
            for x in g:
                self.index[x].append(i)

    def match(self, text: str) -> tuple[float, list[str]]:
        """(best overlap ratio, keys of the best reference); ratio 1.0 for an equal normalized text."""
        t = norm_task(text)
        if t in self.exact:
            return 1.0, self.keys[self.exact[t]]
        g = grams(t) - self.boiler
        hits: collections.Counter = collections.Counter()
        for x in g:
            for i in self.index.get(x, ()):
                hits[i] += 1
        best, who = 0.0, []
        for i, s in hits.items():
            if len(g) < self.min_grams and self.size[i] >= self.min_grams:
                continue
            r = max(s / max(1, len(g)), s / max(1, self.size[i]))
            if r > best:
                best, who = r, self.keys[i]
        return best, who


# ------------------------------------------------------------------------------------------------- rendering
def agents_in_order(rec: dict) -> list[str]:
    out = []
    for s in rec["steps"]:
        a = s["agent"]
        if a.lower() not in NOT_AGENTS and a not in out:
            out.append(a)
    return out


def step_ref(rec: dict, i: int) -> str:
    return f"Step {i} [{rec['steps'][i]['agent']}]"


def render_log(rec: dict, cap: int, with_gt: bool) -> str:
    lines = [HEADER, "", f"Task: {cut(rec['task'], TASK_CAP)}"]
    if with_gt and rec.get("answer"):
        lines.append(GT_PREFIX + cut(rec["answer"], GT_CAP))
    lines.append("")
    for i, s in enumerate(rec["steps"]):
        lines.append(f"Step {i} [{s['agent']}]: {cut(s['text'], cap)}")
    return "\n".join(lines)


def options_block(names: list[str]) -> str:
    return "\n".join(f"{L}) {n}" for L, n in zip(SYMBOLS, names))


def step_prompt(rec: dict, cap: int, with_gt: bool) -> str:
    names = [step_ref(rec, i) for i in range(len(rec["steps"]))]
    return render_log(rec, cap, with_gt) + "\n\n" + Q_STEP + "\nOptions:\n" + options_block(names)


def log_chars(rec: dict, cap: int) -> int:
    """len(step_prompt(rec, cap, True)) up to a constant, fast (for the cap search)."""
    return sum(min(len(s["text"].strip()), cap) + (7 if len(s["text"].strip()) > cap else 0) for s in rec["steps"])


def agent_prompt(rec: dict, cap: int) -> str:
    """The agent-level prompt of make_items / make_agent_item (log without the reference line, agent question)."""
    return render_log(rec, cap, False) + "\n\n" + Q_AGENT + "\nOptions:\n" + options_block(agents_in_order(rec))


def fit_cap(rec: dict, char_budget: int, cmax: int, cmin: int, prompt=None) -> int:
    """The largest per-step cap c in [cmin, cmax] (water-filling: every step cut to at most c characters) whose step
    prompt (or prompt(rec, cap), e.g. agent_prompt for agent-only records) is at most char_budget characters; cmin if
    none is."""
    empty = dict(rec, steps=[dict(s, text="") for s in rec["steps"]])
    fixed = len(step_prompt(empty, 0, True) if prompt is None else prompt(empty, 0))
    if fixed + log_chars(rec, cmax) <= char_budget:
        return cmax
    lo, hi = cmin, cmax
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fixed + log_chars(rec, mid) <= char_budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ------------------------------------------------------------------------------------------------- nodes
def _node(nid: str, kind: str, q: str, opp: str, truth: bool, deps: list, rng: random.Random, **extra) -> dict:
    pol = "plain"
    if rng.random() < 0.5:
        q, opp, truth, pol = opp, q, not truth, "opposite"
    return {"id": nid, "question": q, "opposite": opp, "truth": bool(truth), "p_cot": 1.0 if truth else 0.0,
            "p_fresh": None, "type": kind, "depends_on": deps, "status": "direct", "initial": None,
            "polarity": pol, **extra}


def build_nodes(rec: dict, rng: random.Random) -> list[dict]:
    """Yes/no nodes supported by the labels, each family balanced by selection (as many plain-Yes as plain-No
    questions per item), then asked in plain or opposite wording by a fair coin (truth follows the wording):
      prefix    "Does the decisive error occur at or before Step t [A]?" for t = g-1 and g (g the gold step) plus one
                more pair a < g-1 (No) / b > g (Yes) when both exist; t = n-1 is never asked (always Yes), and the
                family
                is skipped when g = 0 or g = n-1 (one polarity only). Ordered: each depends on the previous threshold.
      step      "Does Step t [A] contain the decisive error?" for the gold and one other step (adjacent with
                probability .5); depends on prefix t and prefix t-1 when asked (Dec_t = P_t and not P_t-1).
      agent     "Was the decisive error made by X?" for the gold agent and one other agent (traces with >= 2 agents
                and a gold agent).
      category  (Who&When Pro) the gold mode's category and one other category; mode: the gold mode and one other
                text mode (from the same category with probability .5); a mode depends on its category's node.
    No node states a result; no question names a fault, an injection or a reason."""
    n, g = len(rec["steps"]), rec["dec"]
    nodes: list[dict] = []
    ref = lambda t: step_ref(rec, t)                                            # noqa: E731
    # prefix
    pre: dict = {}
    if 1 <= g <= n - 2:
        ts = [g - 1, g]
        lo, hi = list(range(0, g - 1)), list(range(g + 1, n - 1))
        if lo and hi:
            ts += [rng.choice(lo), rng.choice(hi)]
        prev = None
        for t in sorted(ts):
            nid = f"p{t}"
            nodes.append(_node(nid, "prefix", f"Does the decisive error occur at or before {ref(t)}?",
                               f"Does the decisive error occur after {ref(t)}?", g <= t,
                               [prev] if prev else [], rng, step=t))
            pre[t] = nid
            prev = nid
    # step
    others = [t for t in range(n) if t != g]
    if others:
        adj = [t for t in (g - 1, g + 1) if 0 <= t < n]
        h = rng.choice(adj) if adj and rng.random() < 0.5 else rng.choice(others)
        for t in sorted((g, h)):
            deps = [pre[x] for x in (t, t - 1) if x in pre]
            nodes.append(_node(f"s{t}", "step", f"Does {ref(t)} contain the decisive error?",
                               f"Is {ref(t)} free of the decisive error?", t == g, deps, rng, step=t))
    # agent
    agents = agents_in_order(rec)
    if rec.get("gold_agent") and rec["gold_agent"] in agents and len(agents) >= 2:
        other = rng.choice([a for a in agents if a != rec["gold_agent"]])
        for k, a in enumerate(x for x in agents if x in (rec["gold_agent"], other)):
            nodes.append(_node(f"a{k}", "agent", f"Was the decisive error made by {a}?",
                               f"Was the decisive error made by an agent other than {a}?", a == rec["gold_agent"],
                               [], rng, agent=a))
    # category and mode
    mode = rec.get("mode")
    if mode in MODES and mode in TEXT_MODES:
        cat = category(mode)
        same = [m for m in TEXT_MODES if m != mode and category(m) == cat]
        neg = rng.choice(same) if same and rng.random() < 0.5 else rng.choice([m for m in TEXT_MODES if m != mode])
        cats_text = sorted({category(m) for m in TEXT_MODES})
        neg_cat = category(neg) if category(neg) != cat else rng.choice([c for c in cats_text if c != cat])
        cid = {}
        for c in sorted((cat, neg_cat), key=cats_text.index):
            cid[c] = f"k{c}"
            nodes.append(_node(cid[c], "category", f"Is the decisive error, broadly, {CATEGORIES[c]}?",
                               f"Is the decisive error, broadly, something other than {CATEGORIES[c]}?", c == cat, [],
                               rng, category=c))
        for m in sorted((mode, neg), key=TEXT_MODES.index):
            gl = MODES[m][1]
            nodes.append(_node(f"m{m}", "mode", f"Is the decisive error {gl}?",
                               f"Is the decisive error something other than {gl}?", m == mode,
                               [cid[category(m)]] if category(m) in cid else [], rng, mode=m))
    depth = node_depths(nodes)
    for x in nodes:
        x["node_depth"] = depth[x["id"]]
    return nodes


def plain_truth(n: dict) -> bool:
    return n["truth"] if n["polarity"] == "plain" else not n["truth"]


# ------------------------------------------------------------------------------------------------- items
def make_items(rec: dict, item_id: str, group: str, *, cap: int, with_gt: bool, rng: random.Random,
               nodes: bool = True) -> tuple[dict, dict | None]:
    """The step-level item (ww_alg_step layout with up to 62 options A-Z, a-z, 0-9; subqs = the nodes) and the
    agent-level item (ww_alg_agent layout; None for single-agent traces or without a gold agent among the options)."""
    n = len(rec["steps"])
    if n > MAX_STEPS:
        raise ValueError(f"{item_id}: {n} steps do not fit {MAX_STEPS} option symbols")
    labs = list(SYMBOLS[:n])
    names = [step_ref(rec, i) for i in range(n)]
    g = rec["dec"]
    with_gt = bool(with_gt and rec.get("answer"))
    agents = agents_in_order(rec)
    meta = {"origin": rec["origin"], "subset": rec["subset"], "benchmark": rec["benchmark"], "uid": rec["uid"],
            "decisive_step": g, "decisive_agent": rec["steps"][g]["agent"], "gold_agent": rec.get("gold_agent"),
            "mode": rec.get("mode"), "category": category(rec.get("mode")), "n_steps": n, "n_agents": len(agents),
            "step_cap": cap, "flags": rec.get("flags", [])}
    if rec.get("accepted"):
        meta["accepted_predictions"] = rec["accepted"]
    subqs = build_nodes(rec, rng) if nodes else []
    step = {"item_id": item_id, "source": SOURCE, "group": group + ("/step_gt" if with_gt else "/step"),
            "type": "choice", "prompt": step_prompt(rec, cap, with_gt), "labels": labs, "label_order": labs,
            "label_names": dict(zip(labs, names)), "gold_label": labs[g], "gold_probs": None,
            "teacher": {L: float(i == g) for i, L in enumerate(labs)}, "subqs": subqs, "has_tree": bool(subqs),
            "tree_source": TREE_SOURCE, "tree_fix": True, "with_gt": with_gt, "fa": meta}
    agent = None
    ga = rec.get("gold_agent")
    if not rec.get("single_agent") and ga in agents and 2 <= len(agents) <= MAX_STEPS:
        al = list(SYMBOLS[:len(agents)])
        k = agents.index(ga)
        agent = {"item_id": item_id + "-agent", "source": SOURCE, "group": group + "/agent", "type": "choice",
                 "prompt": render_log(rec, cap, False) + "\n\n" + Q_AGENT + "\nOptions:\n" + options_block(agents),
                 "labels": al, "label_order": al, "label_names": dict(zip(al, agents)), "gold_label": al[k],
                 "gold_probs": None, "teacher": {L: float(i == k) for i, L in enumerate(al)}, "with_gt": False,
                 "fa": dict(meta, traj_item=item_id)}
    return step, agent


# metadata that must never reach a prompt (field names, phase markers, AFTraj's injected tool-call ids); natural words
# such as "inject" occur in tasks and logs and are allowed there, but not in node questions (NODE_LEAK)
LEAK = re.compile(r"\b(is_mistake|is_injected|is_replayed|is_live|injection_via|call_verify_\w*|mistake_reason|"
                  r"accepted_predictions|mistake_step|mistake_agent)\b|\[\"injected\"\]", re.I)
NODE_LEAK = re.compile(r"\b(inject\w*|fault\w*|corrupt\w*|perturb\w*|slip\w*|reason for)\b", re.I)
_OPT = re.compile(r"^([A-Za-z0-9])\) Step (\d+) \[([^\[\]\n]+)\]$")


def audit_item(item: dict) -> list[str]:
    """Format, label and leak checks of a step-level item (empty list: passes)."""
    bad = []
    p, labs = item["prompt"], item["labels"]
    head, sep, tail = p.rpartition("\n\n" + Q_STEP + "\nOptions:\n")
    if not sep or not head.startswith(HEADER + "\n\nTask: ") or "\nOptions:" in head:
        bad.append("layout")
    opts = tail.split("\n")
    if len(opts) != len(labs) or labs != list(SYMBOLS[:len(labs)]):
        bad.append("options")
    for i, (L, o) in enumerate(zip(labs, opts)):
        m = _OPT.match(o)
        if not m or m.group(1) != L or int(m.group(2)) != i or f"\nStep {i} [{m.group(3)}]: " not in head + "\n":
            bad.append(f"option {L}")
            break
    if item["gold_label"] not in labs or labs.index(item["gold_label"]) != item["fa"]["decisive_step"] \
            or max(item["teacher"], key=item["teacher"].get) != item["gold_label"]:
        bad.append("gold")
    if (GT_PREFIX in head) != item["with_gt"]:
        bad.append("with_gt")
    m = LEAK.search(head)
    if m:
        bad.append(f"log mentions {m.group(0)}")
    by = {n["id"]: n for n in item["subqs"]}
    if len(by) != len(item["subqs"]) or any(d not in by for n in item["subqs"] for d in n["depends_on"]):
        bad.append("node ids")
    if node_depths(item["subqs"]) != {n["id"]: n["node_depth"] for n in item["subqs"]}:
        bad.append("node_depth")
    g = item["fa"]["decisive_step"]
    fam = collections.Counter()
    for n in item["subqs"]:
        q = re.sub(r"\[[^\[\]]*\]", "[]", n["question"]).replace(str(n.get("agent")), "")   # names are data
        if LEAK.search(q) or NODE_LEAK.search(q) or "Answer:" in q:
            bad.append(f"{n['id']}: wording")
        t = plain_truth(n)
        fam[(n["type"], t)] += 1
        want = {"prefix": lambda: g <= n.get("step", -1), "step": lambda: n.get("step") == g,
                "agent": lambda: n.get("agent") == item["fa"]["gold_agent"],
                "category": lambda: n.get("category") == item["fa"]["category"],
                "mode": lambda: n.get("mode") == item["fa"]["mode"]}[n["type"]]()
        if t != want:
            bad.append(f"{n['id']}: truth")
    for kind in {k for k, _ in fam}:
        if fam[(kind, True)] != fam[(kind, False)]:
            bad.append(f"{kind}: unbalanced")
    return bad


def stable_hash(*parts) -> int:
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


# ------------------------------------------------------------------------------------------------- cleaning (--clean)
# Version v1 (notes/next_round/failure_attribution/label_audit.md; the n-gram probe scripts/fa_ngram_probe.py):
#   text rules   injector-produced strings rewritten in every step, the task and the reference answer of every record of
#                their scope (pool and test alike, gold step or not), so no rule string survives anywhere
#   meta leak    agentracer / aftraj trajectories whose original text narrates the injection (the agent discusses the
#                injected instruction or its boilerplate): not attribution items; dropped from train / val and from the
#                cleaned test files
#   group drops  (origin/subset[/benchmark] as debias_group) dropped from train / val, kept in the test files
#                (CLEAN_DROP_GROUPS: mislabelled or lexically separable groups, with the evidence)
#   flags        agentracer coding test records carry "label_unreliable" (the labelled step is the agent's logged input
#                prompt in 56% of the trajectories; shifting it to the next output step was checked on 24 items and
#                recovered a visibly erroneous step in 10)
TEXT_RULES = (
    # (name, scope: origins or origin/subset keys, pattern, replacement)
    ("attacked_filename", ("*",), re.compile(r"_attacked_solution", re.I), "_solution"),
    ("i_am_at_step", ("*",), re.compile(r"\bI am at step \d+\b[^.!?\n]*[.!?]?[ \t]*", re.I), ""),
    ("system_override", ("*",), re.compile(r"\b(?:mandatory\s+)?system\s+override(\s+instructions?)?\b", re.I),
     lambda m: "instructions" if m.group(1) else "instruction"),
    # Who&When Pro smolagents: replayed steps keep the original model's ```python fences, the injected and live steps
    # use <code> tags (the parsed code follows as "Code:\n```py" in both)
    ("code_fence_python", ("wwpro/smolagents",), re.compile(r"```python\b"), "```py"),
    ("code_tags", ("wwpro/smolagents",), re.compile(r"</?code>|<end_code>"), ""),
)
META_ORIGINS = ("agentracer", "aftraj")
META = re.compile(
    r"inject(?:ed|ion)?\s+(?:error|instruction|information|requirement|content)|error[ -]injection|stealth(?:y|ily)|"
    r"system\s+override|mandatory\s+(?:\w+\s+){0,2}(?:requirement|override|instruction)|implementation requirement|"
    r"exactly as specified|(?:verify|correct) (?:the )?injected|\bI am at step \d+", re.I)
CLEAN_DROP_GROUPS = {
    "agentracer/coding/coding": "labels: the labelled step is the logged input prompt in 56%; next-step shift visibly "
                                "erroneous in 10 / 24 checked",
    "wwpro/debate": "an agent already wrong before the injected turn in 57% of MC traces; n-gram probe +41 points",
    "wwpro/dylan": "an agent already wrong before the injected turn in 57% of MC traces; single phrase 'answer' (-) "
                   "+24 points held-out",
    "agentracer/agentic/Injection Error": "injected think boilerplate ('as specified', 'required', 'implement') "
                                         "+18 points held-out after the meta-leak drop",
    "aftraj/agentic/injected": "single phrase 'Thought: I need' (-) +20 points held-out (n 88)",
    "agentracer/math/math": "labels: after the meta-leak drop 6 / 8 checked items have no error at the labelled step "
                            "(a correct step labelled, or the error in another step) and 12% end with the reference "
                            "answer (the injection did not take)",
}
LABEL_FLAGS = {"agentracer/coding/coding": "label_unreliable"}


def clean_group(rec: dict) -> str:
    """build_fa_public.debias_group: origin/subset, with the benchmark (provenance) for aftraj and agentracer."""
    key = f"{rec['origin']}/{rec['subset']}"
    return key + f"/{rec['benchmark']}" if rec["origin"] in ("aftraj", "agentracer") else key


def _in_scope(scope: tuple, rec: dict) -> bool:
    return "*" in scope or rec["origin"] in scope or f"{rec['origin']}/{rec['subset']}" in scope


def meta_leak(rec: dict) -> bool:
    """The original text narrates the injection (META in a step or the task; agentracer and aftraj only)."""
    if rec["origin"] not in META_ORIGINS:
        return False
    return bool(META.search(rec["task"])) or any(META.search(s["text"]) for s in rec["steps"])


def clean_text(rec: dict, counts=None) -> dict:
    """A copy of rec with every TEXT_RULES rule of its scope applied to every step, the task and the answer; counts
    (a Counter) gets rule -> number of strings changed. Agent names, labels and step count are unchanged."""
    out = dict(rec)

    def sub(text):
        if not text:
            return text
        for name, scope, pat, rep in TEXT_RULES:
            if _in_scope(scope, rec):
                text, k = pat.subn(rep, text)
                if k and counts is not None:
                    counts[name] += k
        return text
    out["steps"] = [dict(s, text=sub(s["text"])) for s in rec["steps"]]
    out["task"] = sub(rec["task"])
    out["answer"] = sub(rec.get("answer"))
    return out


def clean_record(rec: dict, version: str, drop_groups=None, counts=None) -> tuple[dict, str | None]:
    """(cleaned record, drop reason or None). Reasons: "meta_leak" (drop everywhere) and "group:<key>" (drop from train
    / val only; test records keep their place). Label flags of LABEL_FLAGS are added to the record's flags."""
    if version != "v1":
        raise ValueError(f"unknown clean version {version!r}")
    groups = CLEAN_DROP_GROUPS if drop_groups is None else drop_groups
    leak = meta_leak(rec)
    out = clean_text(rec, counts)
    g = clean_group(rec)
    if g in LABEL_FLAGS and LABEL_FLAGS[g] not in out["flags"]:
        out["flags"] = list(out["flags"]) + [LABEL_FLAGS[g]]
    if leak:
        return out, "meta_leak"
    if g in groups:
        return out, f"group:{g}"
    return out, None


# ------------------------------------------------------------------------------------------------- new sources (fa1d)
# build_fa_public.py --add-sources correct,aegis (notes/next_round/failure_attribution/tier1_public.md, fa1d). Both are
# cleaned by their readers (stage normalize), so their records are the same with or without --clean:
#   correct  CORRECT-Error (Yu et al., ICML 2026; Apache-2.0): 2,226 Magentic-One trajectories (Planner, WebSurfer,
#            Coder, CodeExecutor, FileSurfer) with one injected decisive error each. history[0] is always the user
#            task prompt and mistake_step indexes history (the role at mistake_step equals mistake_agent in 2,001 /
#            2,226; the other 225: 43 label step 0 = the user prompt with agent "Unknown", 70 bracketed names
#            "[Planner]" that agent_name() reads as "Planner", the rest name another agent) -> rendered steps
#            history[1:], rendered index mistake_step - 1. groundtruth and mistake_reason are empty in every row (no
#            reference line, no _gt items). Literal tells of the injector, removed here:
#              - 5,691 / 10,146 non-gold WebSurfer contents are Python literal lists "['...']" (0 / 491 gold ones):
#                parsed with ast.literal_eval (never eval) and joined (literal_text)
#              - the page line: "open to the page [<title>](<url>)" in 93% of the original WebSurfer steps, 0% "(<url>)";
#                the injected ones write "(<url>)" in 89% -> every form rewritten to "open to the page (<url>)"
#              - the injector's "Agent: <name>\nContent:" wrapper (120 / 2,183 gold steps, 0 others) -> removed
#              - trajectories whose text carries the injector's plan, refusal or failure message (CE_META) -> dropped
#   aegis    Aegis (Kong et al., 2025; MIT): MAS trajectories with errors injected into one or more agents; label =
#            the set of (faulty agent, MAST failure mode); no step index (no step items). Kept: trajectories with
#            exactly one faulty agent (agent-level finals; Aegis-Training train + val -> pool, Aegis-Bench test ->
#            test). Contents that are Python literal lists of {"type", "text"} (smolagents) are parsed with
#            ast.literal_eval; "System" entries (the smolagents system prompt, never faulty) are not rendered;
#            is_injected (macnet), phase and agent_role are never rendered; trajectories with injector debris or text
#            that narrates the injection (AEGIS_META) are dropped. The faulty agent's MAST mode maps to this taxonomy
#            for category / mode nodes (MAST_MAP; only when the agent has exactly one error type).

def literal_text(s) -> tuple[str, bool]:
    """(text, parsed): a content string that is a Python literal list of strings or of {"text": ...} dicts (CORRECT-Error
    WebSurfer "['...']", Aegis smolagents "[{'type': 'text', 'text': ...}]") parsed with ast.literal_eval (never eval)
    and joined by newlines; any other string, or a literal of another shape, is returned unchanged (None -> "")."""
    import ast
    if s is None:
        return "", False
    if not isinstance(s, str):
        return str(s), False
    t = s.strip()
    if len(t) < 4 or t[0] != "[" or t[-1] != "]" or t[1] not in "'\"{":
        return s, False
    try:
        v = ast.literal_eval(t)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return s, False
    if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
        return "\n".join(v), True
    if isinstance(v, list) and v and all(isinstance(x, dict) and isinstance(x.get("text"), str) for x in v):
        return "\n".join(x["text"] for x in v), True
    return s, False


def _ddg(m) -> str:
    from urllib.parse import quote_plus
    return f"open to the page (https://duckduckgo.com/?q={quote_plus(m.group(1))}&ia=web)."


CE_TEXT_RULES = (
    ("agent_content_wrapper", re.compile(r"\A\s*Agent:[^\n]*\n\s*Content:[ \t]*\n?"), ""),
    ("page_title_link", re.compile(r"open to the page \[[^\n]*?\]\((https?://[^\s)]+)\)"), r"open to the page (\1)"),
    ("page_title_url", re.compile(r"open to the page (?![\[(])[^\n]*? at DuckDuckGo\s*\((https?://[^\s)]+)\)"),
     r"open to the page (\1)"),
    ("page_title_only", re.compile(r"open to the page (?![\[(])([^\n]*?) at DuckDuckGo\."), _ddg),
)
_INJECTOR_TEXT = (
    ("error_in_generation", re.compile(r"Error in content generation", re.I)),
    ("injection_plan", re.compile(r"INJECTION PLAN|Subtlety Rationale|Expected Impact\s*:|Error Pattern\s*:|"
                                  r"Injection (?:Strategy|Point|Type)\s*:", re.I)),
    ("upper_tag", re.compile(r"</?[A-Z][A-Z0-9]*_[A-Z0-9_]+>")),
    ("introduced_error", re.compile(r"\bintroduc\w* (?:an? )?(?:subtle |deliberate |small )?(?:error|mistake|bug|flaw)|"
                                    r"\bsubtle (?:error|mistake|flaw)", re.I)),
    ("refusal", re.compile(r"(?:can[’']?t|cannot|won[’']?t|unable to) (?:help|assist|comply)[^.\n]{0,80}(?:inject|insert|"
                           r"deceptive|misleading|misinformation|incorrect|error|modif|false)", re.I)),
    ("narrated", re.compile(r"Preserving the original content|deliberately (?:incorrect|wrong|mislead|introduc|flaw|"
                            r"erroneous)|intentionally (?:incorrect|wrong|mislead|introduc|flaw|erroneous)|"
                            r"\bto incorrectly\b", re.I)),
    ("inject_phrase", re.compile(r"\binject(?:ed|ing|ion)? (?:false|misleading|deceptive|an? error|errors|error|incorrect)",
                                 re.I)),
)
CE_META = _INJECTOR_TEXT
AEGIS_META = _INJECTOR_TEXT + (
    ("injector_log", re.compile(r"Error in inject\w*|Injection for \w+|🎯|🔍|chat\.completions\.create|Raw OpenAI API",
                                re.I)),
    ("injector_path", re.compile(r"malicious_agent", re.I)),
    ("as_instructed", re.compile(r"\bas instructed\b", re.I)),
    ("mast_code", re.compile(r"\bFM-\d\.\d\b")),
    ("refusal_plain", re.compile(r"I[’']m sorry, but I can[’']?t (?:assist|help)|I can[’']?t (?:assist|help) with "
                                 r"(?:that|this)", re.I)),
)


FINAL_ANSWER = re.compile(r"final answer", re.I)
# CORRECT-Error records with these flags leave train / val and the test files (build_fa_public.py), with the evidence
CORRECT_DROP_FLAGS = {
    "same_speaker_next": "structural injection tell: the step after a non-final gold Planner step is another Planner "
                         "step in 534 / 534 trajectories, after other Planner steps in 1.4% (153 / 10,765)",
    "no_final_answer": "the run ends without a final answer (773 of the 783 runs that reach 20 steps): the 20-item "
                       "label re-check found the labelled step decisive in 2 / 10 such runs (the Planner corrects the "
                       "injected claim, or the loop and the step limit cause the failure) against 8 / 10 answered runs",
}


def injector_hits(texts, rules) -> list[str]:
    """Names of the rules (name, regex) that match any of the texts."""
    return [n for n, p in rules if any(p.search(t) for t in texts if t)]


def ce_clean_text(text: str, counts=None) -> str:
    text, _ = literal_text(text)
    for name, pat, rep in CE_TEXT_RULES:
        text, k = pat.subn(rep, text)
        if k and counts is not None:
            counts[f"correct/text/{name}"] += k
    return text


def correct_records(rows, counts: collections.Counter):
    """CORRECT-Error rows (dicts with the parquet fields; history a list of {role, content} or its Python-literal
    string) -> records, split "pool" (the test set is held out by question in the build). Drops: history[0] not the
    user prompt or a mistake_step outside history ("index"), mistake_step 0 ("user_step": the labelled step is the
    user prompt), injector text anywhere in the trajectory ("meta_leak"). agent_mismatch is flagged as for StepFinder
    (dropped from the pool by the build, kept in the test file)."""
    import ast
    for r in rows:
        h = r["history"]
        if isinstance(h, str):
            h = ast.literal_eval(h)
        h = [dict(e) for e in h]
        k = int(r["mistake_step"])
        if not h or str(h[0].get("role") or h[0].get("name")).lower() != "user" or not 0 <= k < len(h):
            counts["correct/drop/index"] += 1
            continue
        if k == 0:
            counts["correct/drop/user_step"] += 1
            continue
        texts = [ce_clean_text(e.get("content"), counts) for e in h[1:]]
        hits = injector_hits(texts, CE_META)
        if hits:
            counts["correct/drop/meta_leak"] += 1
            for n in hits:
                counts[f"correct/meta/{n}"] += 1
            continue
        names = [str(e.get("role") or e.get("name") or "unknown") for e in h[1:]]
        flags = [] if agent_name(names[k - 1]) == agent_name(r["mistake_agent"]) else ["agent_mismatch"]
        if k < len(names) and agent_name(names[k]) == agent_name(names[k - 1]):
            flags.append("same_speaker_next")
        if not FINAL_ANSWER.search(texts[-1]):
            flags.append("no_final_answer")
        rec = _rec(str(r["trajectory_id"]), "correct", str(r["dataset"]), str(r["generator_model"]), r["question"],
                   r.get("groundtruth"), list(zip(names, texts)), k - 1, r["mistake_agent"], ids=uuids(r["question"]),
                   flags=flags)
        rec["qid"] = f"{r['dataset']}/{re.sub(r'_[0-9]+$', '', str(r['question_id']))}"
        yield rec


def read_correct(path: str, counts: collections.Counter):
    """CORRECT-Error from its parquet file (pandas, python3's user site) or from a .jsonl file of the same fields."""
    if str(path).endswith(".jsonl"):
        with open(path) as f:
            rows = [json.loads(line) for line in f]
    else:
        import pandas as pd
        df = pd.read_parquet(path)
        rows = []
        for _, x in df.iterrows():
            d = {c: x[c] for c in ("trajectory_id", "dataset", "generator_model", "question", "groundtruth",
                                   "question_id", "mistake_agent", "mistake_step")}
            d["history"] = x["history"] if isinstance(x["history"], str) else [dict(e) for e in x["history"]]
            rows.append(d)
    yield from correct_records(rows, counts)


# MAST failure mode (Aegis error_type) -> (this taxonomy's mode or None, its category or None, MAST name, rationale)
MAST_MAP = {
    "FM-1.1": ("R.4", "R", "Disobey task specification", "a wrong kind of solution for the task as specified"),
    "FM-1.2": (None, None, "Disobey role specification", "no counterpart: acting outside one's role is neither a "
                                                         "delegation nor a communication error"),
    "FM-1.3": ("A.4", "A", "Step repetition", "repeating steps without progress"),
    "FM-1.4": ("V.1", "V", "Loss of conversation history", "forgetting earlier information"),
    "FM-1.5": (None, "A", "Unaware of termination conditions", "an action error (not stopping when done); neither "
                                                                "premature termination nor looping exactly"),
    "FM-2.1": ("V.1", "V", "Conversation reset", "progress and context lost by restarting"),
    "FM-2.2": (None, "C", "Fail to ask for clarification", "a coordination failure; not one of C.1-C.3 exactly"),
    "FM-2.3": ("PL.2", "PL", "Task derailment", "goal drift (moving away from the original objective)"),
    "FM-2.4": ("C.2", "C", "Information withholding", "communication failure: information withheld"),
    "FM-2.5": ("C.2", "C", "Ignored other agent's input", "communication failure: another agent's input ignored"),
    "FM-2.6": (None, "R", "Reasoning-action mismatch", "a reasoning error in the broad sense; no exact mode"),
    "FM-3.1": ("A.3", "A", "Premature termination", "premature termination"),
    "FM-3.2": ("V.2", "V", "No or incomplete verification", "a missing check"),
    "FM-3.3": ("V.2", "V", "Incorrect verification", "confirming a wrong result"),
}
AEGIS_MODES = sorted({m for m, _, _, _ in MAST_MAP.values() if m}, key=list(MODES).index)
AEGIS_CATEGORIES = [c for c in CATEGORIES if c in {c for _, c, _, _ in MAST_MAP.values() if c}]
AEGIS_HIDDEN = {"System"}          # the smolagents system prompt (never a faulty agent): not rendered
# Aegis frameworks dropped from train / val (kept in the test file), with the evidence
AEGIS_DROP_GROUPS = {
    "aegis/macnet": "position-determined label: the faulty agent is the second agent (Node0) in all 160 single-faulty "
                    "macnet trajectories kept (train 130, val 24, test 6; agent-probe position-only CV 100%)",
    "aegis/dylan": "length and format tells: the faulty agent has the shortest mean turn in 47% of 686 pool "
                   "trajectories (chance 25%); its turns drop the peer-evaluation block (\"let's evaluate\" in 11% of "
                   "faulty vs 29% of other agents); agent probes on fa1d train: n-gram CV 35.7 -> 58.6, length "
                   "(surface) fit on train 54.8 vs structure 21.4 on its Aegis-Bench items",
}


def aegis_records(lines, split: str, counts: collections.Counter):
    for line in lines:
        x = json.loads(line)
        md = x.get("metadata") or {}
        fw = str(md.get("framework"))
        faulty = (x.get("output") or {}).get("faulty_agents") or []
        names = sorted({str(a.get("agent_name")) for a in faulty})
        if len(names) != 1:
            counts[f"aegis/drop/{'no' if not names else 'several'}_faulty_agents_{split}"] += 1
            continue
        hist = (x.get("input") or {}).get("conversation_history") or []
        steps = []
        for s in hist:
            a = str(s.get("agent_name"))
            if a in AEGIS_HIDDEN:
                counts["aegis/hidden_steps"] += 1
                continue
            steps.append((a, literal_text(s.get("content"))[0]))
        hits = injector_hits([t for _, t in steps], AEGIS_META)
        if hits:
            counts[f"aegis/drop/meta_leak_{split}"] += 1
            for n in hits:
                counts[f"aegis/meta/{fw}/{n}"] += 1
            continue
        types = sorted({str(a.get("error_type")) for a in faulty})
        mast = types[0] if len(types) == 1 else None
        mode, cat = MAST_MAP.get(mast, (None, None, None, None))[:2] if mast else (None, None)
        q = (x.get("input") or {}).get("query", "")
        gt = (x.get("ground_truth") or {}).get("correct_answer")
        rec = _rec(str(x["id"]), "aegis", fw, str(md.get("benchmark")), q, gt, steps, None, names[0], mode=mode,
                   split="test" if split == "test" else "pool", ids=uuids(q))
        rec.update(mast=mast, mast_all=types, category=cat, aegis_split=split,
                   strategy=sorted({str(a.get("injection_strategy")) for a in faulty}))
        yield rec


def read_aegis(root: str, counts: collections.Counter):
    """Aegis-Training/{train,val}.jsonl (pool) and Aegis-Bench/test.jsonl (test), streamed."""
    import os
    for rel, split in (("Aegis-Training/train.jsonl", "train"), ("Aegis-Training/val.jsonl", "val"),
                       ("Aegis-Bench/test.jsonl", "test")):
        with open(os.path.join(root, rel)) as f:
            yield from aegis_records(f, split, counts)


def build_agent_nodes(rec: dict, rng: random.Random) -> list[dict]:
    """Nodes of an agent-only record (Aegis), each family balanced as in build_nodes: agent (the gold agent and one other),
    category (the gold category and one other of AEGIS_CATEGORIES) when the record's MAST mode maps to a category, mode
    (the gold mode and one other of AEGIS_MODES, from the same category with probability .5) when it maps to a mode; a
    mode depends on its category's node."""
    nodes: list[dict] = []
    agents = agents_in_order(rec)
    ga = rec.get("gold_agent")
    if ga in agents and len(agents) >= 2:
        other = rng.choice([a for a in agents if a != ga])
        for k, a in enumerate(x for x in agents if x in (ga, other)):
            nodes.append(_node(f"a{k}", "agent", f"Was the decisive error made by {a}?",
                               f"Was the decisive error made by an agent other than {a}?", a == ga, [], rng, agent=a))
    cat, mode = rec.get("category"), rec.get("mode")
    if cat in AEGIS_CATEGORIES:
        neg_mode = None
        if mode in AEGIS_MODES:
            same = [m for m in AEGIS_MODES if m != mode and category(m) == cat]
            neg_mode = rng.choice(same) if same and rng.random() < 0.5 else rng.choice(
                [m for m in AEGIS_MODES if m != mode])
        if neg_mode and category(neg_mode) != cat:
            neg_cat = category(neg_mode)
        else:
            neg_cat = rng.choice([c for c in AEGIS_CATEGORIES if c != cat])
        cid = {}
        for c in sorted((cat, neg_cat), key=AEGIS_CATEGORIES.index):
            cid[c] = f"k{c}"
            nodes.append(_node(cid[c], "category", f"Is the decisive error, broadly, {CATEGORIES[c]}?",
                               f"Is the decisive error, broadly, something other than {CATEGORIES[c]}?", c == cat, [],
                               rng, category=c))
        if neg_mode:
            for m in sorted((mode, neg_mode), key=AEGIS_MODES.index):
                gl = MODES[m][1]
                nodes.append(_node(f"m{m}", "mode", f"Is the decisive error {gl}?",
                                   f"Is the decisive error something other than {gl}?", m == mode,
                                   [cid[category(m)]] if category(m) in cid else [], rng, mode=m))
    depth = node_depths(nodes)
    for x in nodes:
        x["node_depth"] = depth[x["id"]]
    return nodes


def make_agent_item(rec: dict, item_id: str, group: str, *, cap: int, rng: random.Random, nodes: bool = True):
    """The agent-level item of an agent-only record (make_items' agent layout) with nodes (build_agent_nodes) as subqs;
    None when the gold agent is not among two or more options."""
    agents = agents_in_order(rec)
    ga = rec.get("gold_agent")
    if ga not in agents or not 2 <= len(agents) <= MAX_STEPS:
        return None
    al = list(SYMBOLS[:len(agents)])
    k = agents.index(ga)
    meta = {"origin": rec["origin"], "subset": rec["subset"], "benchmark": rec["benchmark"], "uid": rec["uid"],
            "decisive_step": None, "decisive_agent": None, "gold_agent": ga, "mode": rec.get("mode"),
            "category": rec.get("category"), "mast": rec.get("mast"), "mast_all": rec.get("mast_all"),
            "n_steps": len(rec["steps"]), "n_agents": len(agents), "step_cap": cap, "flags": rec.get("flags", [])}
    subqs = build_agent_nodes(rec, rng) if nodes else []
    return {"item_id": item_id + "-agent", "source": SOURCE, "group": group + "/agent", "type": "choice",
            "prompt": render_log(rec, cap, False) + "\n\n" + Q_AGENT + "\nOptions:\n" + options_block(agents),
            "labels": al, "label_order": al, "label_names": dict(zip(al, agents)), "gold_label": al[k],
            "gold_probs": None, "teacher": {L: float(i == k) for i, L in enumerate(al)}, "subqs": subqs,
            "has_tree": bool(subqs), "tree_source": TREE_SOURCE, "tree_fix": True, "with_gt": False, "fa": meta}


def audit_agent_item(item: dict) -> list[str]:
    """Format, label and leak checks of an agent-level item (empty list: passes); node truths and per-family balance as
    audit_item."""
    bad = []
    p, labs = item["prompt"], item["labels"]
    head, sep, tail = p.rpartition("\n\n" + Q_AGENT + "\nOptions:\n")
    if not sep or not head.startswith(HEADER + "\n\nTask: ") or "\nOptions:" in head or GT_PREFIX in head:
        bad.append("layout")
    opts = tail.split("\n")
    if len(opts) != len(labs) or labs != list(SYMBOLS[:len(labs)]):
        bad.append("options")
    names = [item["label_names"][L] for L in labs]
    if [o.split(") ", 1)[-1] for o in opts] != names:
        bad.append("option names")
    for a in names:
        if not re.search(r"\nStep \d+ \[" + re.escape(a) + r"\]: ", head + "\n"):
            bad.append(f"agent {a} not in log")
            break
    if item["gold_label"] not in labs or names[labs.index(item["gold_label"])] != item["fa"]["gold_agent"] \
            or max(item["teacher"], key=item["teacher"].get) != item["gold_label"]:
        bad.append("gold")
    m = LEAK.search(head)
    if m:
        bad.append(f"log mentions {m.group(0)}")
    by = {n["id"]: n for n in item.get("subqs", [])}
    if len(by) != len(item.get("subqs", [])) or any(d not in by for n in item.get("subqs", []) for d in n["depends_on"]):
        bad.append("node ids")
    fam = collections.Counter()
    for n in item.get("subqs", []):
        q = re.sub(r"\[[^\[\]]*\]", "[]", n["question"]).replace(str(n.get("agent")), "")
        if LEAK.search(q) or NODE_LEAK.search(q) or "Answer:" in q:
            bad.append(f"{n['id']}: wording")
        t = plain_truth(n)
        fam[(n["type"], t)] += 1
        want = {"agent": lambda: n.get("agent") == item["fa"]["gold_agent"],
                "category": lambda: n.get("category") == item["fa"]["category"],
                "mode": lambda: n.get("mode") == item["fa"]["mode"]}.get(n["type"], lambda: None)()
        if t != want:
            bad.append(f"{n['id']}: truth")
    for kind in {k for k, _ in fam}:
        if fam[(kind, True)] != fam[(kind, False)]:
            bad.append(f"{kind}: unbalanced")
    return bad
