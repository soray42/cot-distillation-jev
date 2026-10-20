"""Post-checks for the failure-attribution reasoning trees (src/cotdistill/fa_trees.py): the v2 trees of every source
(alg, hc, aftraj, real, agtr, wwpro, ce) and the v3 clean-node add-on. Finds problems the hard checks (check_tree /
check_addon) do not look for, and writes the label-audit statistics for the paper. CPU only, no model, no network.
Report-only by default; it never edits a source file, a brief, a shard or a trees / add-on file.

  ~/.venvs/sev/bin/python -I scripts/fa_trees_postcheck.py \
      --report notes/next_round/failure_attribution/tree_v3_audit.md --flags <scratch>/postcheck/flags.json
  ~/.venvs/sev/bin/python -I scripts/fa_trees_postcheck.py --apply --out-dir <scratch>/postcheck/filtered

Inputs per source (SOURCES): the teacher work list (ids), the v2 source trees (src/<subset>__<id>.json), the assembled
tree lines (<subset>.jsonl; for a source whose lines file does not exist yet, the passing trees are assembled in memory
with fa_trees.output_record, the same call as `extract_fa_trees.py assemble`), the raw trajectories, and for alg / hc /
aftraj / real the v3 work list (data/raw/fa_trees_v3/all_entries.json) with its add-on sources (src_v3/) and the
add-on lines of `v3-check` (data/raw/fa_trees_v3/v3add_<subset>.jsonl, used by --apply).

Detectors (flag type: what is flagged; ids are "<traj_id>#<node id>", or "<traj_id>" for a whole tree)
  gt_in_stem       a clean node whose code-rendered reference to its target step (the excerpt of free_ref_for, shown in
                   the index-free question) contains the reference answer (GT) or a gold value derived from it; a v2
                   content / conflict node whose excerpts inserted by the index-free render (free_render: the question
                   and rewritten text options) contain it. The hard gt_leak check runs on the index-form text only.
                   Matching: fa_trees.gt_leak (case / whitespace insensitive, stand-alone, thousands separators) on
                   the GT and on its parts (dabench "@key[value]" values, list elements, comma-separated parts), plus
                   numbers with tolerance (equal up to 1e-9 relative, or a more precise number that rounds to a GT
                   with decimals: 1.0103 for 1.01; "5,429,112.0" for 5429112) and simpleqa "acceptable range" GTs.
                   Strong when the whole GT (or its number / range) matches and has >= 4 characters (not a yes / no /
                   none word) or is a number with a decimal point or >= 3 digits; else gt_in_stem_weak (a short GT
                   such as "3", "No", "A", or only one part of a list GT such as "bananas" of "pears, bananas").
  gt_in_pointer    (side statistic, not removed) a pointer node's index-free option or question excerpt contains it.
  negation_pair    a distractor that is the near-negation of the true option (content / conflict / clean): after
                   removing negation words (not, no, never, n't, without -> with + not, ...), auxiliaries and articles,
                   the two word multisets (light stemming) differ in at most NEG_MAX_DIFF word, the word sequences
                   match >= NEG_OVERLAP (difflib ratio), and exactly one side carried a negation ("read the notes" /
                   "did not read the notes"; not "joins with a space" / "joins with no separator", nor "b*d = e" /
                   "d*b = d, not e").
  agent_lt2        an agent node with fewer than 2 options. agent_trivial: fewer than 2 options that are real agents
                   (tool output / user / environment speakers excluded), so the answer is the only agent left.
                   Trees without an agent node (allowed by check_tree when the log has < 2 real agents) are counted
                   per source (no_agent_node), not flagged.
  not_failed       a kept tree whose team final answer (extracted from the TEACHER's answer step: final_answer(...) /
                   the printed value after it, \\boxed{...}, "FINAL ANSWER:", "Final answer:", "Answer:",
                   "ANSWER_VERIFIED:", a bare short answer) equals the reference (answer_match, strong): normalized
                   string, a bare number with tolerance, multiple-choice letter, dabench pairs, the same list in the
                   same order, a number in a simpleqa range. not_failed_weak (review by hand): equal only without a
                   leading article, the number with a unit word the GT lacks ("185 yards" / "185"), the same elements
                   in another order (the tasks ask for an order), a near spelling of a short name, the GT
                   (strong-strength) inside a short answer; math (mathchat) also a number within 1%. GT "", "None"
                   (JSON null), reference code, dacode output-file lists and tablebench chart references count as no
                   reference. A step with no extractable answer is not flagged.
  unknown_field    a field outside the source schema (V2_* / ADDON_* allowlists below; e.g. "gt_in_ref"), in the src
                   files or in the tree / add-on lines. Id "<traj_id>#<node id or top>:<field>".

Label audit (per source): work-list items, source files written, teacher drops and drop rate, drop reasons grouped by
keyword (DROP_CLASSES, first match in this order: not failed / other: no valid tree (the label is kept, the checks
cannot be met) / truncated evidence / error corrected later or not causal / labelled step fine, error elsewhere / no
evidence in log; else unclassified), label_note and rule_override counts,
trees removed after QA (excluded_after_qa.txt), kept trees; for the v3 add-on: entries, targets, clean nodes, skips,
skip rate and the skips that say the target step already makes the labelled step's error (SKIP_LATE: label possibly
late).

--apply writes filtered COPIES (never in place, never under a src / src_v3 directory) of the tree lines and add-on lines
to --out-dir, with the same file names, and apply_summary.json:
  * a flagged clean node (types in --remove) is removed from its add-on line;
  * a flagged v2 node is removed and its children are re-attached to its parents (the rule free_render uses for the
    nodes it drops), or with --v2-node-policy drop-line the whole tree line (and its add-on line) is dropped;
  * not_failed (strong) drops the tree line and its add-on line;
  * every modified line gets "postcheck": {"removed": [{"id", "flag"}], ...}.
The flags are computed from the source files, the filter acts on the jsonl files: a trees / add-on file older than its
newest source file is reported STALE and --apply refuses it (re-run assemble / v3-check, or --allow-stale).
"""
from __future__ import annotations

import argparse
import ast
import collections
import copy
import datetime
import difflib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cotdistill import fa_trees as ft  # noqa: E402

V3_SUBSETS = ("alg", "hc", "aftraj", "real")
SUBSETS = ("alg", "hc", "aftraj", "real", "agtr", "wwpro", "ce")


def _src(sub: str) -> dict:
    d = {"alg": "stepfinder_trees", "hc": "stepfinder_trees"}.get(sub, f"{sub}_trees")
    full = "stepfinder_trees/full" if sub in ("alg", "hc") else d
    raw = "stepfinder/repo/data" if sub in ("alg", "hc") else f"{sub}_sf"
    return {"raw": f"data/raw/{raw}", "src": f"data/raw/{full}/src",
            "ids": f"data/raw/{full}/{'final_ids' if sub in ('alg', 'hc') else 'all_ids'}.json",
            "trees": f"data/raw/{d}/{sub}.jsonl", "shards": f"data/raw/{full}/shards",
            "excluded": f"data/raw/{d}/excluded_after_qa.txt"}


SOURCES = {s: _src(s) for s in SUBSETS}
V3_ENTRIES = "data/raw/fa_trees_v3/all_entries.json"
V3_SHARDS = "data/raw/fa_trees_v3/shards"
V3_ADDON = "data/raw/fa_trees_v3/v3add_{sub}.jsonl"

# ---------------------------------------------------------------- schema allowlists (fa_trees.py docstrings)
V2_TOP = {"traj_id", "teacher", "answer_step", "label_note", "nodes", "rule_override", "drop"}
V2_NODE_COMMON = {"id", "type", "question", "parents"}
V2_NODE = {"answer_step": {"answer_step", "distractor_steps", "quote"},
           "provenance": {"answer_step", "distractor_steps", "quote"},
           "content": {"correct", "distractors", "cite_step", "quote", "target_overlap_rank"},
           "conflict": {"correct", "distractors", "cite_step", "quote", "target_overlap_rank"},
           "agent": {"answer_agent", "distractor_agents"}}
ADDON_TOP = {"traj_id", "teacher", "nodes"}
ADDON_NODE = {"id", "type", "cite_step", "correct", "distractors", "quote"}
ADDON_SKIP = {"id", "type", "cite_step", "skip"}
LINE_TOP = {"traj_id", "subset", "teacher", "nodes", "checks", "meta", "postcheck", "v3"}
ADDON_LINE_TOP = {"traj_id", "subset", "teacher", "nodes", "skips", "checks", "meta", "postcheck"}
LINE_NODE = {"id", "type", "question", "options", "answer", "parents", "cite_step", "quote", "option_steps",
             "render", "free_steps"}

NON_REAL_AGENTS = ft.NON_AGENTS | ft.CLEAN_NON_AGENTS | {"ComputerTerminal", "Computer terminal", "environment",
                                                         "Environment", "tool", "Tool"}
TEXT_NODE_TYPES = ("content", "conflict", "clean")
NODE_FLAGS = ("gt_in_stem", "gt_in_stem_weak", "gt_in_pointer", "negation_pair", "agent_lt2", "agent_trivial")
TREE_FLAGS = ("not_failed", "not_failed_weak")
ALL_FLAGS = NODE_FLAGS + TREE_FLAGS + ("unknown_field",)
DEFAULT_REMOVE = ("gt_in_stem", "negation_pair", "agent_lt2", "not_failed")

# ---------------------------------------------------------------- text normalization, numbers, gold values

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s))).strip().casefold()


NUM = re.compile(r"(?<![\w.])[-+]?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w]|\.\d)")
NUM_FULL = re.compile(r"^[-+]?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?$")
COMMON_WORDS = {"none", "no", "yes", "true", "false", "n/a", "na", "null", "nan", "all", "both", "nil", "zero"}
RANGE = re.compile(r"(?i)between\s+([-+]?[\d,]*\.?\d+)\s*[^\d\s]*\s+and\s+([-+]?[\d,]*\.?\d+)")


def is_number(s: str) -> bool:
    return bool(NUM_FULL.match(str(s).strip()))


def to_float(s: str) -> float:
    return float(str(s).strip().replace("$", "").replace(",", "").rstrip("%").lstrip("+"))


def decimals(s: str) -> int:
    t = str(s).strip().rstrip("%")
    return len(t.split(".", 1)[1]) if "." in t else 0


def num_equal(x: str, g: str) -> bool:
    """x (a number in the text) equals the gold number g: up to 1e-9 relative, or x is more precise than a g that has
    decimals and rounds to it (1.0103 for 1.01)."""
    try:
        a, b = to_float(x), to_float(g)
    except ValueError:
        return False
    tol = 1e-9 * max(1.0, abs(b))
    if abs(a - b) <= tol:
        return True
    dg, dx = decimals(g), decimals(x)
    return dg >= 1 and dx > dg and abs(round(a, dg) - b) <= tol + 10 ** -(dg + 3)


def numbers_in(text: str) -> list[str]:
    return [m.group(0) for m in NUM.finditer(str(text))]


def no_reference(gt: str) -> str:
    """Why a GT is no usable reference ("" when it is one)."""
    g = str(gt).strip()
    if not g:
        return "empty"
    if g == "None":
        return "none"                       # JSON null (load_traj renders it "None"); the briefs treat it as no answer
    if re.match(r"^\[\s*['\"][^'\"]+\.(json|csv|png|jpe?g|txt|md|xlsx?|py|pdf|sqlite|db|html|parquet)['\"]", g, re.I):
        return "output_files"                # dacode: names of the files the task writes
    if g.startswith("y_references"):
        return "chart"                       # tablebench chart reference
    if len(g) > 300 or ("\n" in g and re.search(r"\b(def|return|import|for|if|while|print)\b|[{};=]", g)):
        return "code"
    return ""


def strength(v: str) -> str:
    v = str(v).strip()
    if is_number(v):
        digits = re.sub(r"\D", "", v.split(".")[0]).lstrip("0")
        return "strong" if decimals(v) >= 1 or len(digits) >= 3 else "weak"
    n = norm(v).strip(" .\"'")
    return "weak" if len(n) <= 3 or n in COMMON_WORDS else "strong"


def _literal_list(s: str):
    s = s.strip()
    if not s.startswith("["):
        return None
    try:
        x = ast.literal_eval(s)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    out = []

    def walk(y):
        if isinstance(y, (list, tuple)):
            for z in y:
                walk(z)
        else:
            out.append(str(y))
    if isinstance(x, (list, tuple)):
        walk(x)
        return out
    return None


def gold_values(gt: str) -> dict:
    """{"main": str, "values": [(value, kind full|part, strength)], "range": (lo, hi) | None}; {} when no reference."""
    if no_reference(gt):
        return {}
    g = str(gt).strip()
    main = re.sub(r"(?i)\s*\(acceptable range[^)]*\)\s*$", "", g).strip() or g
    vals = [(main, "full")]
    parts = re.findall(r"@[\w-]+\[([^\]]*)\]", main)
    if not parts:
        parts = _literal_list(main) or []
    if not parts and "," in main and len(main) <= 80 and not is_number(main):
        parts = [p.strip() for p in main.split(",")]
    seen = {norm(main)}
    for p in parts:
        p = str(p).strip().strip("'\"")
        if p and norm(p) not in seen:
            seen.add(norm(p))
            vals.append((p, "part"))
    rng = None
    m = RANGE.search(g)
    if m:
        try:
            lo, hi = sorted((to_float(m.group(1)), to_float(m.group(2))))
            rng = (lo, hi)
        except ValueError:
            rng = None
    if not is_number(main):
        lead = re.match(r"^([-+]?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*[a-zA-Z%°]{0,12}$", main)
        if lead:                                              # "17.2 meters", "185 yards", "25%"
            vals.append((lead.group(1), "number"))
    return {"main": main, "values": [(v, k, strength(v)) for v, k in vals], "range": rng}


def text_hits(gold: dict, text: str) -> list[dict]:
    """Every gold value found in text (string match by fa_trees.gt_leak, numbers with tolerance, the range)."""
    if not gold or not str(text).strip():
        return []
    hits, nums = [], None
    for v, kind, st in gold["values"]:
        how = ""
        if ft.gt_leak(v, text):
            how = "string"
        elif is_number(v):
            nums = numbers_in(text) if nums is None else nums
            x = next((x for x in nums if num_equal(x, v)), None)
            if x is not None:
                how = f"numeric:{x}"
        if how:
            hits.append({"value": v, "kind": kind, "strength": st, "how": how})
    if gold.get("range") and not hits:
        nums = numbers_in(text) if nums is None else nums
        lo, hi = gold["range"]
        x = next((x for x in nums if lo <= to_float(x) <= hi), None)
        if x is not None:
            hits.append({"value": f"[{lo}, {hi}]", "kind": "range", "strength": "strong", "how": f"range:{x}"})
    return hits


def hit_tier(hits: list[dict]) -> str | None:
    """strong: the whole GT (or its number / range) of strong strength; weak: a short GT or only a part of a list GT."""
    if not hits:
        return None
    return "strong" if any(h["strength"] == "strong" and h["kind"] != "part" for h in hits) else "weak"


# ---------------------------------------------------------------- (a) excerpts of the index-free render

QUOTED = re.compile(ft.OPEN_Q + "([^" + ft.CLOSE_Q + "]*)" + ft.CLOSE_Q)


def inserted_excerpts(orig: str, new: str) -> list[str]:
    """The “…” excerpts in the rendered text that the teacher's text did not already contain (without the "…")."""
    have = collections.Counter(QUOTED.findall(str(orig)))
    out = []
    for ex in QUOTED.findall(str(new)):
        if have[ex] > 0:
            have[ex] -= 1
            continue
        out.append(ex.rstrip(ft.ELLIPSIS).strip())
    return out


def excerpt_gt_flags(traj: dict, nodes: list[dict], gold: dict, cap: int = ft.STEP_CAP) -> list[dict]:
    """(a) GT in the excerpts inserted by free_render, per kept node: question and options. Nodes free_render drops
    are not shown in v3, so they are not flagged. A hit whose value also occurs in the task text ("which city,
    Wharvton or Algrimand") is weak: the student reads it there anyway."""
    if not gold or not nodes:
        return []
    task = traj["question"]
    orig = {nd["id"]: nd for nd in nodes}
    out = []
    for nd in ft.free_render(traj, nodes, cap=cap):
        o = orig[nd["id"]]
        places = [("question", o["question"], nd["question"])]
        places += [(f"option{i}", a, b) for i, (a, b) in enumerate(zip(o["options"], nd["options"]))]
        for where, a, b in places:
            for ex in inserted_excerpts(a, b):
                hits = text_hits(gold, ex)
                if not hits:
                    continue
                for h in hits:
                    h["in_task"] = bool(text_hits({"values": [(h["value"], h["kind"], h["strength"])],
                                                   "range": None}, task)) if h["kind"] != "range" else False
                tier = hit_tier([h for h in hits if not h["in_task"]]) or "weak"
                why = ("strong" if tier == "strong" else
                       "in_task" if hit_tier(hits) == "strong" else
                       "part" if any(h["kind"] == "part" and h["strength"] == "strong" for h in hits) else "short")
                if nd["type"] in TEXT_NODE_TYPES:
                    flag = "gt_in_stem" if tier == "strong" else "gt_in_stem_weak"
                else:
                    flag = "gt_in_pointer"
                out.append({"flag": flag, "node": nd["id"], "node_type": nd["type"], "where": where,
                            "excerpt": ex, "tier": tier, "why": why, "hits": hits,
                            "correct_option": where.startswith("option") and int(where[6:]) == nd["answer"]})
    return out


# ---------------------------------------------------------------- (b) near-negation distractors

NEG_WORDS = {"not", "no", "never", "none", "nothing", "neither", "nor", "cannot", "nobody", "nowhere"}
AUX = {"do", "does", "did", "is", "are", "was", "were", "be", "been", "has", "have", "had", "can", "could", "would",
       "will", "should", "may", "might", "must", "it", "its", "the", "a", "an", "any"}
NEG_OVERLAP = 0.85                         # difflib ratio of the word sequences without the negation
NEG_MAX_DIFF = 1                           # at most one word differs besides the negation ("also", a verb form)


def _stem(w: str) -> str:
    for suf, rep, mn in (("ies", "y", 5), ("ing", "", 6), ("ed", "", 5), ("es", "", 5), ("s", "", 4)):
        if w.endswith(suf) and len(w) >= mn:
            return w[: len(w) - len(suf)] + rep
    return w


def neg_parse(text: str) -> tuple[list[str], int]:
    """(stemmed words in order without negations / auxiliaries / articles, number of negations)."""
    toks = re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[=<>+*/-]", norm(text).replace("\u2019", "'"))
    words, neg = [], 0
    for t in toks:
        if t.endswith("n't") or t in NEG_WORDS:
            neg += 1
            continue
        if t == "without":
            neg += 1
            t = "with"
        if t in AUX:
            continue
        words.append(_stem(t))
    return words, neg


def negation_pair(a: str, b: str, thr: float = NEG_OVERLAP) -> float | None:
    """Similarity of a and b when one is the near-negation of the other, else None: exactly one side negated; besides
    the negation, at most NEG_MAX_DIFF word (multiset) differs and the word sequences match >= thr (difflib ratio, so
    "b*d = e" / "d*b = d, not e" is not one)."""
    wa, na = neg_parse(a)
    wb, nb = neg_parse(b)
    if (na == 0) == (nb == 0):
        return None
    ca, cb = collections.Counter(wa), collections.Counter(wb)
    common = sum((ca & cb).values())
    diff = sum(((ca - cb) + (cb - ca)).values())
    if common < 2 or diff > NEG_MAX_DIFF:
        return None
    ratio = difflib.SequenceMatcher(None, wa, wb, autojunk=False).ratio()
    return round(ratio, 3) if ratio >= thr else None


def negation_flags(nd: dict) -> list[dict]:
    if nd["type"] not in TEXT_NODE_TYPES or not nd.get("options"):
        return []
    c = nd["options"][nd["answer"]]
    out = []
    for i, o in enumerate(nd["options"]):
        if i == nd["answer"]:
            continue
        ov = negation_pair(c, o)
        if ov is not None:
            out.append({"flag": "negation_pair", "node": nd["id"], "node_type": nd["type"], "correct": c,
                        "distractor": o, "overlap": ov})
    return out


# ---------------------------------------------------------------- (c) agent nodes

def real_agents(traj: dict) -> list[str]:
    return [a for a in ft.agents_in_log(traj) if a not in NON_REAL_AGENTS]


def agent_flags(nd: dict) -> list[dict]:
    if nd["type"] != "agent":
        return []
    opts = list(nd.get("options") or [])
    if len(opts) < 2:
        return [{"flag": "agent_lt2", "node": nd["id"], "node_type": "agent", "options": opts}]
    real = [o for o in opts if o not in NON_REAL_AGENTS]
    if len(real) < 2:
        return [{"flag": "agent_trivial", "node": nd["id"], "node_type": "agent", "options": opts}]
    return []


# ---------------------------------------------------------------- (d) run not failed

MARKER = re.compile(r"(?im)(FINAL ANSWER|ANSWER_VERIFIED|final answer(?:\s+[\w-]+){0,4}|final(?:\s+[\w-]+){0,3}|"
                    r"answer)\s*(\([^)\n]*\))?\s*\**\s*[:：]\s*(.+)$")


def _balanced(t: str, i: int, op: str, cl: str) -> str | None:
    """Text inside the bracket that opens at t[i]."""
    depth = 0
    for j in range(i, len(t)):
        if t[j] == op:
            depth += 1
        elif t[j] == cl:
            depth -= 1
            if depth == 0:
                return t[i + 1:j]
    return None


def _unquote(s: str) -> str:
    s = s.strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, str):
            return v
        if isinstance(v, (int, float, list, tuple)):
            return str(list(v) if isinstance(v, tuple) else v)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        pass
    return s


def extract_answer(text: str) -> tuple[str, str]:
    """(the team's final answer in an answer step, how it was found)."""
    t = str(text)
    calls = list(re.finditer(r"final_answer\(", t))
    if calls:
        last = calls[-1]
        out = list(re.finditer(r"Last output from code snippet:\s*(.+)", t))
        if out and out[-1].start() > last.start():
            return out[-1].group(1).strip(), "printed_after_final_answer"
        arg = _balanced(t, last.end() - 1, "(", ")")
        if arg is not None and arg.strip():
            a = arg.strip()
            if re.fullmatch(r"[A-Za-z_]\w*", a):            # a variable: no value in the text
                return "", "final_answer_variable"
            return _unquote(re.sub(r"^answer\s*=\s*", "", a)), "final_answer_call"
    m = list(re.finditer(r"tool_name:\s*final_answer\s+arguments:\s*(\{.*)", t))
    if m:
        a = m[-1].group(1)
        mm = re.search(r"['\"]answer['\"]\s*:\s*(.+?)\}?\s*$", a.splitlines()[0])
        if mm:
            return _unquote(mm.group(1).rstrip("}").strip()), "tool_final_answer"
    boxes = [i for i in (m.end() - 1 for m in re.finditer(r"\\boxed\{", t))]
    if boxes:
        b = _balanced(t, boxes[-1], "{", "}")
        if b is not None:
            return b.strip(), "boxed"
    marks = []
    for m in MARKER.finditer(t):
        head = norm(m.group(1))
        words = re.findall(r"[a-z]+", head)
        if words and words[0] == "final" and set(words[1:]) & ft.MARKER_NOT_ANSWER:
            continue
        marks.append(m)
    if marks:
        return marks[-1].group(3).strip(), "marker"
    s = t.strip()
    if 0 < len(s) <= 200:
        return s, "bare"
    return "", "none"


def norm_answer(s: str, articles: bool = True) -> str:
    s = norm(s)
    s = re.sub(r"\bterminate\b", " ", s)
    s = re.sub(r"[*`]+", "", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("$", "")
    s = re.sub(r"\b(usd|dollars?)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" \t.\"'“”‘’:;!")
    if articles:
        s = re.sub(r"^(the|a|an)\s+", "", s)
    return s


def _elems(s: str) -> list[str] | None:
    lit = _literal_list(s)
    if lit is not None:
        return [norm_answer(x) for x in lit]
    if "," in s and len(s) <= 120 and not is_number(s.strip()):
        return [norm_answer(p) for p in s.split(",") if p.strip()]
    return None


def _same_elem(a: str, b: str) -> bool:
    return a == b or (is_number(a) and is_number(b) and num_equal(a, b))


def _strip_math(s: str) -> str:
    t = str(s).strip().strip("$").strip()
    t = re.sub(r"\\(left|right|!|,|;|displaystyle)", "", t).replace("\u2212", "-")
    return t.strip()


def math_value(s: str) -> float | None:
    """Value of a plain number, a / b, \\frac{a}{b} (\\dfrac, \\tfrac, with a sign); None otherwise."""
    t = _strip_math(s)
    m = re.fullmatch(r"([-+]?)\s*\\[dt]?frac\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}", t)
    if not m:
        m = re.fullmatch(r"([-+]?)\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", t)
    if m:
        den = float(m.group(3))
        if den == 0:
            return None
        v = float(m.group(2)) / den
        return -v if m.group(1) == "-" else v
    return to_float(t) if is_number(t) else None


def math_elems(s: str) -> list[float] | None:
    """Values of a tuple / list / comma-separated answer of numbers and fractions ("\\left( \\frac{7}{45},
    \\frac{4}{45} \\right)", "-1,0,1", "[0.6, 0.936]"); None when some element is not a number."""
    t = _strip_math(s)
    if t[:1] in "([" and t[-1:] in ")]":
        t = t[1:-1]
    parts = [p for p in t.split(",") if p.strip()]
    if len(parts) < 2:
        return None
    vals = [math_value(p) for p in parts]
    return None if any(v is None for v in vals) else vals


def close(a: float, b: float, rel: float) -> bool:
    return abs(a - b) <= rel * max(abs(b), 1e-12) + 1e-12


def answer_match(ans: str, gt: str, ordered: bool = True, math: bool = False) -> tuple[str | None, str]:
    """("strong" | "weak" | None, how): does the team's final answer equal the reference?
    strong: equal after normalization (case, whitespace, $, quotes, a trailing period, markdown); a bare number equal
    with tolerance; the multiple-choice letter; dabench @key[value] pairs; a number inside a simpleqa acceptable
    range; the same list in the same order.
    math=True (wwpro mathchat): fractions and decimals are compared by value ("0.35" for "\\dfrac{7}{20}"), tuples
    element-wise, a number within 1% is weak; elsewhere "1/2" is not "2/4" (GAIA asks for the fractions as written).
    ordered=False (solution sets): the same elements in any order are strong.
    weak: equal only without a leading article ("Nightmare on Elm Street" for "A Nightmare on Elm Street"); the number
    with a unit word the GT does not have ("185 yards" for "185"); the same elements in another order (tasks ask for
    an order); a number within 1% of the GT (math only); a near spelling of a short name (8-60 characters, <= 4 words,
    no digits, same first letter, edit distance <= max(1, len / 12): "Niloufar Bayani" for "Niloufar Bayan"); the GT
    inside a short answer (<= 3 x its length + 20 characters: "Hotels, Motels" for "Hotels")."""
    gold = gold_values(gt)
    if not gold or not str(ans).strip():
        return None, ""
    a, g = norm_answer(ans, articles=False), norm_answer(gold["main"], articles=False)
    if not a:
        return None, ""
    if a == g:
        return "strong", "equal"
    if re.fullmatch(r"[a-j]", g):                                     # multiple choice letter
        m = re.match(r"^\(?([a-j])\)?(?:[\s.):]|$)", a)
        return ("strong", "letter") if m and m.group(1) == g else (None, "")
    pairs_g = re.findall(r"@([\w-]+)\[([^\]]*)\]", gold["main"])
    if pairs_g:
        pairs_a = re.findall(r"@([\w-]+)\[([^\]]*)\]", str(ans))
        if pairs_a and {(k, norm_answer(v)) for k, v in pairs_a} == {(k, norm_answer(v)) for k, v in pairs_g}:
            return "strong", "dabench_pairs"
    m = re.match(r"^([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(%|\s+[a-z°%]+(?:\s[a-z]+)?)?$", a)
    num_a, unit_a = (m.group(1), (m.group(2) or "").strip()) if m else (None, "")
    if num_a is not None and gold.get("range") and gold["range"][0] <= to_float(num_a) <= gold["range"][1]:
        return "strong", f"range:{num_a}"
    if num_a is not None:
        for v, kind, _ in gold["values"]:
            if kind in ("full", "number") and is_number(v) and num_equal(num_a, v):
                gl = re.match(r"^[-+]?\$?[\d,.]+\s*(.*)$", g)
                same_unit = not unit_a or (gl is not None and gl.group(1).strip() == unit_a)
                return ("strong", f"numeric:{num_a}") if same_unit else ("weak", f"numeric_with_unit:{a}")
    va, vg = (math_value(ans), math_value(gold["main"])) if math else (None, None)
    if va is not None and vg is not None:
        if close(va, vg, 1e-6) or (is_number(_strip_math(gold["main"])) and is_number(_strip_math(ans))
                                   and num_equal(_strip_math(ans), _strip_math(gold["main"]))):
            return "strong", f"math_value:{va:g}"
        if close(va, vg, 1e-2):
            return "weak", f"numeric_close:{va:g}"
    ma, mg = (math_elems(ans), math_elems(gold["main"])) if math else (None, None)
    if ma and mg and len(ma) == len(mg):
        if all(close(x, y, 1e-6) for x, y in zip(ma, mg)):
            return "strong", "math_tuple"
        rest = list(mg)
        for x in ma:
            j = next((i for i, y in enumerate(rest) if close(x, y, 1e-6)), None)
            if j is None:
                break
            rest.pop(j)
        else:
            return ("weak", "same_values_other_order") if ordered else ("strong", "same_value_set")
    if norm_answer(ans) == norm_answer(gold["main"]):
        return "weak", "equal_without_article"
    ea, eg = _elems(str(ans)), _elems(gold["main"])
    if ea and eg and len(ea) == len(eg) and len(eg) >= 2:
        if all(_same_elem(x, y) for x, y in zip(ea, eg)):
            return "strong", "same_list"
        rest = list(eg)
        for x in ea:
            j = next((i for i, y in enumerate(rest) if _same_elem(x, y)), None)
            if j is None:
                break
            rest.pop(j)
        else:
            return ("weak", "same_elements_other_order") if ordered else ("strong", "same_element_set")
    if (8 <= len(g) <= 60 and len(g.split()) <= 4 and not re.search(r"[\d\\@\[]", g + a) and a[:1] == g[:1]
            and levenshtein(a, g) <= max(1, len(g) // 12)):
        return "weak", "near_string"
    if strength(gold["main"]) == "strong" and len(str(ans).strip()) <= 3 * len(gold["main"]) + 20:
        if ft.gt_leak(gold["main"], ans):
            return "weak", "contains"
        if is_number(gold["main"]) and any(num_equal(x, gold["main"]) for x in numbers_in(ans)):
            return "weak", "contains_number"
    return None, ""


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def not_failed_flag(traj: dict, answer_step: int | None) -> dict | None:
    if answer_step is None or not 0 <= answer_step < len(traj["steps"]):
        return None
    gt = traj["ground_truth"]
    if no_reference(gt):
        return None
    text = traj["steps"][answer_step]["text"]
    ans, how = extract_answer(text)
    math = "_mathchat_" in traj["traj_id"]                  # math answers: values, solution sets
    tier, why = answer_match(ans, gt, ordered=not math, math=math) if ans else (None, "")
    if tier is None:
        return None
    return {"flag": "not_failed" if tier == "strong" else "not_failed_weak", "answer_step": answer_step,
            "answer": ans[:200], "extracted_by": how, "gt": gt[:200], "match": why}


# ---------------------------------------------------------------- (e) fields outside the schema

def unknown_fields_v2(src: dict) -> list[str]:
    out = [f"top:{k}" for k in src if k not in V2_TOP]
    for i, nd in enumerate(src.get("nodes") or []):
        if not isinstance(nd, dict):
            continue
        allowed = V2_NODE_COMMON | V2_NODE.get(nd.get("type"), set())
        out += [f"{nd.get('id', i)}:{k}" for k in nd if k not in allowed]
    return out


def unknown_fields_addon(src: dict) -> list[str]:
    out = [f"top:{k}" for k in src if k not in ADDON_TOP]
    for i, nd in enumerate(src.get("nodes") or []):
        if not isinstance(nd, dict):
            continue
        allowed = ADDON_SKIP if "skip" in nd else ADDON_NODE
        out += [f"{nd.get('id', i)}:{k}" for k in nd if k not in allowed]
    return out


def unknown_fields_line(line: dict, addon: bool = False) -> list[str]:
    out = [f"line:{k}" for k in line if k not in (ADDON_LINE_TOP if addon else LINE_TOP)]
    for nd in line.get("nodes") or []:
        out += [f"line:{nd.get('id')}:{k}" for k in nd if k not in LINE_NODE]
    return out


# ---------------------------------------------------------------- (f) drop reasons and skips

DROP_CLASSES = [
    ("not failed", re.compile(r"(?i)\bnot failed\b|matches the reference|equals the reference|same as the reference|"
                              r"run (did not|didn't) fail|run succeeded|task (is|was) solved")),
    ("other: no valid tree (label kept)", re.compile(
        r"(?i)no (valid )?tree (can|could|is|possible)|tree cannot|cannot pass|can't pass|"
        r"answer-step rule (conflict|fires|fixes)|rule conflict|leak check|decisive step is empty|(has|have) no "
        r"text|empty (reply|step|message)|nothing to quote|label (itself )?(holds|is supported|is plausible|is right)"
        r"|\(label itself")),
    ("truncated evidence", re.compile(
        r"(?i)\bcut-obs\b|truncat|elided|^\s*label not (supported|shown) by the (kept|visible) log|"
        r"\b(only|is|are|sits|lies|lie|appears?|stands?|remains?|hidden|left)( \w+){0,4} in the cut\b|"
        r"\b(is|are|was|were|being|mostly|partly|both|all|gets?|got) cut\b|in the cut (part|middle|portion|text|"
        r"metadata|section|tail|head|policy|output|observation)|outside the (kept|visible)|not in the (kept|visible)|"
        r"cut (part|middle|portion) of|cut in the source")),
    ("error corrected later or not causal", re.compile(
        r"(?i)recover|redone|re-?d(one|id)\b|repair|correct(s|ed)? (it|this|that|the)\b|later (fixed|corrected|"
        r"repaired|caught)|caught (in|by|at)|fix(es|ed)? (it|this|that) (later|in)|no effect|changes nothing|"
        r"nothing changes|never used|unused|not used|no later step uses|does not (affect|change|reach|feed|carry|"
        r"matter|enter|alter|decide|drive)|correct(s|ed) (at once|by|in|later|immediately|there)|(is|are|was|were) "
        r"corrected|the same one|keeps? to the end|(un|re)done (in|by|at)|"
        r"harmless|abandon|discard|dropped (in|by|at) step|set aside|not causal|as (the )?decisive|"
        r"no bearing|does not bear|without effect|same (answer|result|value|number|count|final)\b|still (rounds|"
        r"gives|yields)|only costs|supersed|overridden|overrides? (it|this|that|the)\b|overwritten|replaced (in|by)|"
        r"rejects|withdrawn|"
        r"not tied to the failure|fails for another reason|run fails because|does not (rely|produce|lead)|"
        r"changes no\b|adds nothing|introduces? nothing|nothing (from it )?(is )?used|never acted on|"
        r"no visible effect|not what makes|answer (is|was) changed when|answer changes only|already wrong when|"
        r"the answer does not|sets? aside|only (adds|printed|prints)|never enters?|no (numerical )?effect|"
        r"notices this")),
    ("labelled step fine, error elsewhere", re.compile(
        r"(?i)first (given|comes?|appears?|stated|introduced|made|derived|recommended|named|guessed|computed|chosen|"
        r"proposed|adopted|claimed|asserted|reported|picked|shown|shows|taken|enters?|used|arises?|occurs?|written|"
        r"suggested|selected|drawn|inferred|assumed|found|offered|set|fixed)|comes later|"
        r"(a|the) later step (makes|introduces|gives|states|picks|takes|adopts)|in a later step|"
        r"(is|are|was|were) (made|introduced|given|taken|drawn|done) (later|in|by|at) step|"
        r"(starts?|lies|lie|sits?|happens?|occurs?) (elsewhere|later|earlier|in step|at step|before)|elsewhere|"
        r"introduced by|comes from step|the visible (error|errors|fault|failure|mistake)|error (is|was|lies) (in|at)|"
        r"after the answer step|decisive step after|instead (in|at) step|the error is in|is made in step|"
        r"\bonly (runs|asks|fetches|repeats|restates|visits|reports|looks|lists|crawls|prints|loads|delegates|"
        r"searches|launches|requests|opens|calls|echoes|sends|relays|says|checks|re-runs|reruns|copies|forwards|"
        r"plans|announces|goes|tries|writes|offers|gives|shows|names|proposes|notes|uses|states|confirms|"
        r"re-?checks|reads|greets|summari[sz]es|explains|verifies|queries|clicks|scrolls|navigates|types|"
        r"acknowledges|agrees|accepts|forwards|passes|hands|transfers|starts|begins|sets|describes|re-?confirms)|"
        r"comes from (the|a|an|another|its|step)\b|"
        r"(error|mistake|wrong \w+)( \w+){0,3} (comes|come|came) (from|in|later)|the wrong \w+( \w+)? (is|are|was) "
        r"(first )?(given|made|introduced|taken)|begins? (in|at) step|not the labell?ed step|not the earliest|"
        r"fails? (earlier|before)|already fails|(guessed|chosen|made|produced|added|returned|picked|adopted) "
        r"(by [\w\s\[\]_-]{1,40} )?in step|first (produced|added|returned)|"
        r"rests on (the |a |step)|visible slips?|failures? shown|the fault (is|lies)|comes? from ste|"
        r"(is|are) in step \d|made later|goes wrong (after|when|later)|the visible wrong")),
    ("no evidence in log", re.compile(
        r"(?i)nothing (in|visible|shown|on)|no (step|page|evidence|source|text|observation|tool result|line|"
        r"output|other step)\b|uncontradicted|unsourced|agrees? with|consistent with|matches the|"
        r"answer-matches-data|cannot (show|be shown|be checked|be seen|tell)|can't|not (shown|visible) (by|in) the "
        r"log|nothing wrong|no (visible )?(error|mistake|fault)|is (a )?reasonable|defensible|plain reading|"
        r"never (appears|shown|visible)|not contradicted|does not show|doesn't show|cannot be verified|"
        r"no way to|domain knowledge|outside knowledge|is (a )?legitimate|a sound|is not wrong|not an error|"
        r"\b(rightly|correctly|correct|accurate(ly)?|faithful(ly)?|sound|reasonable|legitimate|valid|plausible|"
        r"policy-consistent|necessary|needed|allowed|permitted|as (asked|requested|planned|required))\b|"
        r"match(es)? (the|every|all|what)|follows? the (policy|plan|task|rules?|request)|not visible|"
        r"only against the (hidden|gold)|only from the ground truth|stated nowhere|absent from|no wrong \w+|"
        r"explicitly asks|plain (delegation|search|request|lookup|query)|the right (target|office|nominator|"
        r"company|answer|value|person|city|name|item|order|file|column)|policy requires|expected (company|answer|"
        r"value|city|person|name)|fits? the (log|policy|task)|checks out|exactly what the user asked|ordinary|"
        r"the right move|the task needs|names no|shows no|can be shown|visible only")),
]
DROP_UNCLASSIFIED = "unclassified"
# neutral mentions of the cut part ("(kept and cut)", "kept or cut", "nothing cut") are not truncated evidence
CUT_NEUTRAL = re.compile(r"(?i)\(?\b(kept|visible)( text| part| lines?)? (and|or|as well as) (in )?(the )?cut"
                         r"( (part|middle|text|portion))?( alike)?\)?|\bnothing (is )?cut\b|\bnot cut\b|"
                         r"\b(short|whole|full),? (and )?(nothing|not) cut")
SKIP_LATE = re.compile(r"(?i)\balready\b|\bthe same\b|\bexactly the\b|\bthen\b|(labell?ed|decisive|next|later|final)"
                       r"( \w+){0,2} (step|check|verification|validation|output|answer)|\bverifier\b|\bvalidator\b|"
                       r"\bthe (decisive|labell?ed)\b")
SKIP_OWN = re.compile(r"(?i)\bchang(e|es|ed|ing)\b|outcome|every later|would (give|yield|make|lead)|leads? to|"
                      r"wrong (result|answer|count|value|total|figure|number|difference|inputs?)|by itself|"
                      r"sets the wrong|"
                      r"moves the|inflates|understates?|comes? from|rests? on|produces")
SKIP_CLASSES = ("already makes the labelled step's error (label possibly late)",
                "own earlier error that changes the outcome (earliest error before the label)", "other")


def classify_drop(reason: str) -> tuple[str, list[str]]:
    """(first matching class in DROP_CLASSES order, every matching class) of a teacher drop reason. Neutral mentions
    of the cut part (CUT_NEUTRAL) are removed first; the leading "label not supported by the log:" counts for no class
    (only its "kept log" / "visible log" form is a truncation signal)."""
    r = CUT_NEUTRAL.sub(" ", str(reason))
    every = [name for name, rx in DROP_CLASSES if rx.search(r)]
    return (every[0] if every else DROP_UNCLASSIFIED), every


def classify_skip(reason: str) -> str:
    """SKIP_CLASSES[0]: the skip says the target step already makes the labelled step's error (the labelled step
    then runs / adopts / approves it); [1]: an own, different earlier error that changes the outcome; both put an
    error that matters before the labelled decisive step."""
    if SKIP_LATE.search(reason):
        return SKIP_CLASSES[0]
    if SKIP_OWN.search(reason):
        return SKIP_CLASSES[1]
    return SKIP_CLASSES[2]


# ---------------------------------------------------------------- loading

def rp(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def src_name(traj_id: str) -> str:
    return traj_id.replace("/", "__") + ".json"


class Trajs:
    def __init__(self):
        self.cache: dict[str, dict] = {}

    def get(self, traj_id: str) -> dict | None:
        if traj_id not in self.cache:
            sub, stem = traj_id.split("/", 1)
            p = rp(SOURCES[sub]["raw"]) / ft.SUBSET_DIRS[sub] / "train" / f"{stem}.json"
            self.cache[traj_id] = ft.load_traj(p, sub) if p.exists() else None
        return self.cache[traj_id]


def read_json(p: Path):
    try:
        return json.loads(p.read_text()), ""
    except (OSError, json.JSONDecodeError) as e:
        return None, f"{type(e).__name__}: {e}"


def read_jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def shard_progress(shard_dir: Path, path_of, sub: str | None = None) -> list[tuple[str, int, int]]:
    """(shard, entries, source files present) per shard (entries of subset sub only, when given)."""
    out = []
    for f in sorted(shard_dir.glob("shard_*.json")):
        sh, _ = read_json(f)
        if not isinstance(sh, list):
            continue
        if sub:
            sh = [e for e in sh if str(e["traj_id"] if isinstance(e, dict) else e).split("/", 1)[0] == sub]
            if not sh:
                continue
        out.append((f.stem, len(sh), sum(path_of(e).exists() for e in sh)))
    return out


# ---------------------------------------------------------------- one run over every source

def answer_step_of(line: dict, src: dict | None) -> int | None:
    ch = (line.get("checks") or {}).get("chain") or []
    if ch:
        return ch[0]
    a = (src or {}).get("answer_step")
    return a if isinstance(a, int) and not isinstance(a, bool) else None


def run(args) -> dict:
    trajs = Trajs()
    res: dict = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "sources": {}, "v3": {},
                 "flags": collections.defaultdict(list), "details": [], "status": {}, "lines": {}, "addon_lines": {}}
    flags, details = res["flags"], res["details"]

    def flag(kind: str, ident: str, sub: str, **kw):
        if ident not in flags[kind]:
            flags[kind].append(ident)
        details.append({"flag": kind, "id": ident, "source": sub, **kw})

    entries = {}
    if any(s in V3_SUBSETS for s in args.subsets):
        ent, err = read_json(rp(V3_ENTRIES))
        entries = {e["traj_id"]: e for e in ent or []}
    addon_lines: dict[str, dict] = {}
    for sub in args.subsets:
        cfg = dict(SOURCES[sub])
        if sub in args.trees:
            cfg["trees"] = args.trees[sub]
        st: dict = collections.Counter()
        S: dict = {"cfg": cfg, "n": st}
        res["sources"][sub] = S
        ids, _ = read_json(rp(cfg["ids"]))
        ids = [e["traj_id"] if isinstance(e, dict) else str(e) for e in ids or []]
        ids = [t for t in ids if t.split("/", 1)[0] == sub]
        st["items"] = len(ids)
        srcs: dict[str, dict] = {}
        unreadable = []
        newest_src = 0.0
        for tid in ids:
            p = rp(cfg["src"]) / src_name(tid)
            if not p.exists():
                continue
            newest_src = max(newest_src, p.stat().st_mtime)
            s, err = read_json(p)
            if s is None or not isinstance(s, dict):
                unreadable.append(f"{tid}: {err or 'not an object'}")
                continue
            srcs[tid] = s
        st["written"] = len(srcs)
        st["unreadable"] = len(unreadable)
        S["unreadable"] = unreadable
        res["status"][sub] = shard_progress(rp(cfg["shards"]), lambda e, c=cfg: rp(c["src"]) / src_name(
            e["traj_id"] if isinstance(e, dict) else e), sub)
        # (f) audit + (e) unknown fields on src
        drop_first, drop_any, drop_ex = collections.Counter(), collections.Counter(), collections.defaultdict(list)
        for tid, s in srcs.items():
            for u in unknown_fields_v2(s):
                flag("unknown_field", f"{tid}#{u}", sub, where="v2 src")
            if s.get("drop"):
                st["drops"] += 1
                c1, every = classify_drop(s["drop"])
                drop_first[c1] += 1
                for c in every:
                    drop_any[c] += 1
                drop_ex[c1].append((tid, s["drop"]))
                continue
            st["label_note"] += bool(str(s.get("label_note") or "").strip())
            st["rule_override"] += bool(str(s.get("rule_override") or "").strip())
        S["drop_first"], S["drop_any"], S["drop_examples"] = drop_first, drop_any, drop_ex
        exf = rp(cfg["excluded"])
        S["excluded_after_qa"] = exf.read_text().strip() if exf.exists() else ""
        # tree lines: the assembled file, else assembled in memory (passing trees only)
        tp = rp(cfg["trees"])
        lines: list[dict] = []
        S["stale"] = []
        if tp.exists():
            S["lines_from"] = str(cfg["trees"])
            lines = read_jsonl(tp)
            if newest_src > tp.stat().st_mtime:
                S["stale"].append(f"{cfg['trees']} is older than the newest source file "
                                  f"({datetime.datetime.fromtimestamp(newest_src).isoformat(timespec='seconds')}): "
                                  f"re-run extract_fa_trees.py assemble")
        else:
            S["lines_from"] = "assembled in memory (fa_trees.output_record; passing trees)"
            for tid, s in srcs.items():
                if s.get("drop"):
                    continue
                t = trajs.get(tid)
                if t is None:
                    st["no_raw"] += 1
                    continue
                rec = ft.output_record(t, s)
                if s.get("traj_id") != tid:
                    rec["checks"]["pass"] = False
                    rec["checks"]["failures"].append("schema: traj_id mismatch")
                if rec["checks"]["pass"]:
                    lines.append(rec)
                else:
                    st["hard_fail"] += 1
                    S.setdefault("hard_fail_ids", []).append(f"{tid}: {rec['checks']['failures'][:3]}")
        st["kept"] = len(lines)
        res["lines"][sub] = lines
        # v3 add-on of this source
        add_by_id: dict[str, dict] = {}
        if sub in V3_SUBSETS:
            V = collections.Counter()
            skip_cls, skip_ex = collections.Counter(), collections.defaultdict(list)
            res["v3"][sub] = {"n": V, "skip_classes": skip_cls, "skip_examples": skip_ex}
            sub_entries = [e for e in entries.values() if e["subset"] == sub]
            V["entries"] = len(sub_entries)
            V["targets"] = sum(len(e["targets"]) for e in sub_entries)
            newest_v3 = 0.0
            for e in sub_entries:
                p = rp(e["src"])
                if not p.exists():
                    V["missing"] += 1
                    continue
                newest_v3 = max(newest_v3, p.stat().st_mtime)
                s, err = read_json(p)
                if not isinstance(s, dict):
                    V["unreadable"] += 1
                    continue
                V["written"] += 1
                for u in unknown_fields_addon(s):
                    flag("unknown_field", f"{e['traj_id']}#{u}", sub, where="add-on src")
                t = trajs.get(e["traj_id"])
                if t is None:
                    V["no_raw"] += 1
                    continue
                rec = ft.addon_record(t, s, e)
                V["passed"] += bool(rec["checks"]["pass"])
                V["written_targets"] += len(e["targets"])
                V["clean_nodes"] += len(rec["nodes"])
                V["skips"] += len(rec["skips"])
                for x in rec["skips"]:
                    c = classify_skip(x["reason"])
                    skip_cls[c] += 1
                    skip_ex[c].append((f"{e['traj_id']}#{x['id']}", x["reason"]))
                if rec["checks"]["pass"]:
                    add_by_id[e["traj_id"]] = rec
            ap = rp(V3_ADDON.format(sub=sub))
            res["addon_lines"][sub] = read_jsonl(ap) if ap.exists() else []
            if ap.exists() and newest_v3 > ap.stat().st_mtime:
                S["stale"].append(f"{V3_ADDON.format(sub=sub)} is older than the newest add-on source "
                                  f"({datetime.datetime.fromtimestamp(newest_v3).isoformat(timespec='seconds')}): "
                                  f"re-run extract_fa_trees.py v3-check")
            for ln in res["addon_lines"][sub]:
                for u in unknown_fields_line(ln, addon=True):
                    flag("unknown_field", f"{ln['traj_id']}#{u}", sub, where=f"add-on line {ap.name}")
            # (a) + (b) on clean nodes of add-ons whose tree is not kept are still counted (they are not merged)
        addon_lines.update(add_by_id)
        # detectors on the kept trees (+ their add-on clean nodes)
        no_agent_single = []
        for line in lines:
            tid = line["traj_id"]
            for u in unknown_fields_line(line):
                flag("unknown_field", f"{tid}#{u}", sub, where="tree line")
            t = trajs.get(tid)
            if t is None:
                st["no_raw"] += 1
                continue
            gold = gold_values(t["ground_truth"])
            st["with_reference"] += bool(gold)
            nodes = list(line.get("nodes") or [])
            add = add_by_id.get(tid)
            clean = [nd for nd in (add or {}).get("nodes") or []]
            types = [nd["type"] for nd in nodes]
            if "agent" not in types:
                st["no_agent_node"] += 1
                if len(real_agents(t)) < 2:
                    st["no_agent_node_single_speaker"] += 1
                    no_agent_single.append(tid)
            for nd in nodes + clean:
                for f in negation_flags(nd) + agent_flags(nd):
                    flag(f["flag"], f"{tid}#{nd['id']}", sub, **{k: v for k, v in f.items() if k != "flag"})
            have = {nd["id"] for nd in nodes}
            for f in excerpt_gt_flags(t, nodes + [c for c in clean if c["id"] not in have], gold, cap=args.cap):
                flag(f["flag"], f"{tid}#{f['node']}", sub, **{k: v for k, v in f.items() if k != "flag"})
            ak = answer_step_of(line, srcs.get(tid))
            if gold and ak is not None and 0 <= ak < len(t["steps"]):
                st["answer_extracted"] += bool(extract_answer(t["steps"][ak]["text"])[0])
            nf = not_failed_flag(t, ak)
            if nf:
                flag(nf["flag"], tid, sub, **{k: v for k, v in nf.items() if k != "flag"})
        S["no_agent_single_examples"] = no_agent_single[:10]
        # clean nodes of add-ons whose tree line is not kept: count only
        if sub in V3_SUBSETS:
            kept_ids = {ln["traj_id"] for ln in lines}
            res["v3"][sub]["n"]["addon_without_kept_tree"] = sum(t not in kept_ids for t in add_by_id)
    v3_prog = shard_progress(rp(V3_SHARDS), lambda e: rp(e["src"])) if entries else []
    res["status"]["v3"] = v3_prog
    return res


# ---------------------------------------------------------------- report

def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%" if b else "-"


def per_source_flag_counts(res: dict) -> dict:
    c = collections.defaultdict(collections.Counter)
    for d in res["details"]:
        key = d["flag"]
        if key in ("gt_in_stem", "gt_in_stem_weak"):
            key = f"{key}/{d.get('node_type')}"
        c[d["source"]][key] += 1
    # node-level unique counts for the main flags
    u = collections.defaultdict(collections.Counter)
    for kind, ids in res["flags"].items():
        for i in ids:
            sub = i.split("/", 1)[0]
            u[sub][kind] += 1
    return {"raw": c, "unique": u}


def write_report(res: dict, path: Path, flags_path: str | None, cmdline: str) -> None:
    L: list[str] = []
    subs = list(res["sources"])
    L += ["# Tree v3 audit: post-checks and label audit", "",
          f"Generated {res['time']} by `scripts/fa_trees_postcheck.py` (report only; no source file edited).", "",
          f"Command: `{cmdline}`", ""]
    if flags_path:
        det = Path(flags_path).with_name("flags_detail.jsonl")
        L += [f"Flagged ids per problem type: `{flags_path}` (details: `{det}`)", ""]
    # status
    L += ["## Status of the inputs at run time", ""]
    for sub in subs + ["v3"]:
        prog = res["status"].get(sub) or []
        open_ = [f"{n} {d}/{t}" for n, t, d in prog if d < t]
        if prog:
            done = sum(d for _, _, d in prog)
            tot = sum(t for _, t, _ in prog)
            L.append(f"- {sub}: {done}/{tot} source files in the shards"
                     + (f"; unfinished: {', '.join(open_)}" if open_ else "; all shards complete"))
    for sub in subs:
        S = res["sources"][sub]
        L.append(f"- {sub} tree lines: {S['lines_from']}"
                 + ("; " + "; ".join(f"STALE: {x}" for x in S["stale"]) if S.get("stale") else "; up to date"))
        if S.get("unreadable"):
            L.append(f"  - unreadable source files (being written?): {S['unreadable'][:5]}")
        if S["n"].get("hard_fail"):
            L.append(f"  - {S['n']['hard_fail']} non-dropped trees fail the hard checks now (not counted as kept): "
                     f"{S.get('hard_fail_ids', [])[:3]}")
    L.append("")
    # detector table
    fc = per_source_flag_counts(res)
    u = fc["unique"]
    L += ["## Post-check detectors (counts per source)", "",
          "Node counts are unique nodes; not_failed counts are trees. `ref` = sources with a usable reference answer "
          "(GT): real, agtr and ce have none, aftraj GT `None` and wwpro alfagent / dacode / chart / code references "
          "are not answers, so (a) and (d) are vacuous there. (d) only checks trees whose answer step yields an "
          "answer (`answer extracted`); the others (long steps without a marker, final_answer(<variable>) without "
          "printed output) are not checked.", "",
          "| source | kept trees | with GT | gt_in_stem strong (content / conflict / clean) | weak | gt_in_pointer "
          "| negation_pair | agent_lt2 | agent_trivial | no agent node (single speaker) | answer extracted (of with "
          "GT) | not_failed strong | weak | unknown_field |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    raw = fc["raw"]

    def nodes_of(sub, kind, typ):
        return len({d["id"] for d in res["details"] if d["source"] == sub and d["flag"] == kind
                    and d.get("node_type") == typ})
    tot = collections.Counter()
    for sub in subs:
        n = res["sources"][sub]["n"]
        row = {"kept": n["kept"], "ref": n["with_reference"],
               "s_ct": nodes_of(sub, "gt_in_stem", "content"), "s_cf": nodes_of(sub, "gt_in_stem", "conflict"),
               "s_cl": nodes_of(sub, "gt_in_stem", "clean"), "weak": u[sub]["gt_in_stem_weak"],
               "ptr": u[sub]["gt_in_pointer"], "neg": u[sub]["negation_pair"], "lt2": u[sub]["agent_lt2"],
               "triv": u[sub]["agent_trivial"], "noag": n["no_agent_node"], "noag1": n["no_agent_node_single_speaker"],
               "nf": u[sub]["not_failed"], "nfw": u[sub]["not_failed_weak"], "unk": u[sub]["unknown_field"],
               "ax": n["answer_extracted"]}
        tot.update(row)
        L.append(f"| {sub} | {row['kept']} | {row['ref']} | {row['s_ct'] + row['s_cf'] + row['s_cl']} ({row['s_ct']} / "
                 f"{row['s_cf']} / {row['s_cl']}) | {row['weak']} | {row['ptr']} | {row['neg']} | {row['lt2']} | "
                 f"{row['triv']} | {row['noag']} ({row['noag1']}) | {row['ax']} | {row['nf']} | {row['nfw']} | "
                 f"{row['unk']} |")
    L.append(f"| all | {tot['kept']} | {tot['ref']} | {tot['s_ct'] + tot['s_cf'] + tot['s_cl']} ({tot['s_ct']} / "
             f"{tot['s_cf']} / {tot['s_cl']}) | {tot['weak']} | {tot['ptr']} | {tot['neg']} | {tot['lt2']} | "
             f"{tot['triv']} | {tot['noag']} ({tot['noag1']}) | {tot['ax']} | {tot['nf']} | {tot['nfw']} | "
             f"{tot['unk']} |")
    L.append("")
    neg_types = collections.Counter(d["node_type"] for d in res["details"] if d["flag"] == "negation_pair")
    weak_why = collections.Counter()
    for i in res["flags"].get("gt_in_stem_weak", []):
        ws = {d.get("why") for d in res["details"] if d["id"] == i and d["flag"] == "gt_in_stem_weak"}
        weak_why["in_task" if "in_task" in ws else "part" if "part" in ws else "short"] += 1
    nfw = collections.Counter(d["match"].split(":")[0] for d in res["details"] if d["flag"] == "not_failed_weak")
    triv = collections.Counter(", ".join(sorted(set(d["options"]) & NON_REAL_AGENTS)) for d in res["details"]
                               if d["flag"] == "agent_trivial")
    L += [f"- negation_pair by node type: {dict(neg_types)}.",
          f"- gt_in_stem_weak by reason (unique nodes): {dict(weak_why)} (short = a GT of <= 3 characters / one or "
          f"two digits / yes-no word; part = one element of a list GT; in_task = the GT also stands in the task text).",
          f"- not_failed_weak by match: {dict(nfw)}.",
          f"- agent_trivial: the non-agent options that leave one real agent: {dict(triv.most_common(6))}.", ""]
    # examples
    L += ["## Examples per problem type", ""]
    exn = 6

    def ex_lines(kind, fmt):
        ds = [d for d in res["details"] if d["flag"] == kind]
        seen, out = set(), []
        by_src = collections.defaultdict(list)
        for d in ds:
            by_src[d["source"]].append(d)
        order = []
        while any(by_src.values()) and len(order) < exn:     # round-robin over sources
            for sub in list(by_src):
                if by_src[sub] and len(order) < exn:
                    order.append(by_src[sub].pop(0))
        for d in order:
            if d["id"] in seen:
                continue
            seen.add(d["id"])
            out.append(f"- `{d['id']}` ({d['source']}): {fmt(d)}")
        return out or ["- none"]

    def short(s, n=140):
        s = re.sub(r"\s+", " ", str(s))
        return s if len(s) <= n else s[:n] + "…"
    L += ["### (a) gt_in_stem (strong)", ""] + ex_lines(
        "gt_in_stem", lambda d: f"{d['node_type']} {d['where']}"
                                f"{' (the TRUE option)' if d.get('correct_option') else ''}"
                                f": excerpt “{short(d['excerpt'])}” contains {d['hits'][0]['value']!r} "
                                f"({d['hits'][0]['kind']}, {d['hits'][0]['how']})") + [""]
    L += ["### (a) gt_in_stem_weak (short GT, one part of a list GT, or a GT that also stands in the task)",
          ""] + ex_lines(
        "gt_in_stem_weak", lambda d: f"{d['node_type']} {d['where']}: “{short(d['excerpt'], 100)}” "
                                     f"has {d['hits'][0]['value']!r} ({d.get('why')})") + [""]
    L += ["### (a') gt_in_pointer (side statistic, not removed)", ""] + ex_lines(
        "gt_in_pointer", lambda d: f"{d['node_type']} {d['where']}: “{short(d['excerpt'], 100)}” "
                                   f"has {d['hits'][0]['value']!r} ({d['tier']})") + [""]
    L += ["### (b) negation_pair", ""] + ex_lines(
        "negation_pair", lambda d: f"{d['node_type']}: true “{short(d['correct'], 110)}” vs distractor "
                                   f"“{short(d['distractor'], 110)}” (overlap {d['overlap']})") + [""]
    L += ["### (c) agent_lt2 (fewer than 2 options)", ""] + ex_lines(
        "agent_lt2", lambda d: f"options {d['options']}") + ["", "### (c') agent_trivial", ""] + ex_lines(
        "agent_trivial", lambda d: f"options {d['options']} (fewer than 2 real agents)") + [""]
    single = [x for sub in subs for x in res["sources"][sub].get("no_agent_single_examples", [])[:2]]
    L += [f"Trees without an agent node in a single-speaker log (allowed by check_tree, not flagged), e.g. "
          f"{', '.join(f'`{x}`' for x in single[:8])}.", ""]
    L += ["### (d) not_failed (strong: the final answer equals the reference)", ""] + ex_lines(
        "not_failed", lambda d: f"step {d['answer_step']} answer {short(d['answer'], 80)!r} ({d['extracted_by']}) "
                                f"vs GT {short(d['gt'], 80)!r}: {d['match']}") + [""]
    L += ["### (d) not_failed_weak (the GT appears inside the answer; review by hand)", ""] + ex_lines(
        "not_failed_weak", lambda d: f"step {d['answer_step']} answer {short(d['answer'], 80)!r} "
                                     f"({d['extracted_by']}) vs GT {short(d['gt'], 60)!r}: {d['match']}") + [""]
    L += ["### (e) unknown_field", ""] + ex_lines("unknown_field", lambda d: d["where"]) + [""]
    allow = {"v2 top": sorted(V2_TOP), "v2 node": {k: sorted(V2_NODE_COMMON | v) for k, v in V2_NODE.items()},
             "add-on top": sorted(ADDON_TOP), "add-on node": sorted(ADDON_NODE), "add-on skip": sorted(ADDON_SKIP)}
    L += [f"Allowlists: `{json.dumps(allow)}`", ""]
    # label audit
    L += ["## Label audit per source (v2 trees)", "",
          "items = work list; written = teacher source files present; drop rate = drops / written; kept = tree lines "
          "(assembled file, or passing trees assembled in memory). Drop reasons: first matching class in the order "
          "of the columns (keyword rules in `DROP_CLASSES`).", "",
          "| source | items | written | drops | drop rate | " + " | ".join(c for c, _ in DROP_CLASSES)
          + " | unclassified | label_note | rule_override | removed after QA | kept |",
          "|---|---|---|---|---|" + "---|" * (len(DROP_CLASSES) + 5)]
    classes = [c for c, _ in DROP_CLASSES] + [DROP_UNCLASSIFIED]
    T = collections.Counter()
    for sub in subs:
        S = res["sources"][sub]
        n, df = S["n"], S["drop_first"]
        qa = S["excluded_after_qa"]
        nqa = len([x for x in qa.rsplit(":", 1)[-1].split(",") if x.strip()]) if qa else 0
        row = [n["items"], n["written"], n["drops"]]
        T.update({"items": n["items"], "written": n["written"], "drops": n["drops"], "label_note": n["label_note"],
                  "rule_override": n["rule_override"], "qa": nqa, "kept": n["kept"], **{c: df[c] for c in classes}})
        L.append(f"| {sub} | {n['items']} | {n['written']} | {n['drops']} | {pct(n['drops'], n['written'])} | "
                 + " | ".join(str(df[c]) for c in classes)
                 + f" | {n['label_note']} | {n['rule_override']} | {nqa} | {n['kept']} |")
    L.append(f"| all | {T['items']} | {T['written']} | {T['drops']} | {pct(T['drops'], T['written'])} | "
             + " | ".join(str(T[c]) for c in classes)
             + f" | {T['label_note']} | {T['rule_override']} | {T['qa']} | {T['kept']} |")
    L.append("")
    L += ["Multi-label view (a reason can name several causes; share of drops whose text matches each class):", "",
          "| source | " + " | ".join(classes[:-1]) + " |", "|---|" + "---|" * (len(classes) - 1)]
    for sub in subs:
        S = res["sources"][sub]
        L.append(f"| {sub} | " + " | ".join(f"{S['drop_any'][c]} ({pct(S['drop_any'][c], S['n']['drops'])})"
                                             for c in classes[:-1]) + " |")
    L.append("")
    L += ["Drop examples (first of each class per source):", ""]
    for sub in subs:
        S = res["sources"][sub]
        for c in classes:
            exs = S["drop_examples"].get(c) or []
            if exs:
                tid, r = exs[0]
                L.append(f"- {sub} / {c}: `{tid}`: {short(r, 220)}")
    L.append("")
    unc = [(sub, tid, r) for sub in subs for tid, r in res["sources"][sub]["drop_examples"].get(DROP_UNCLASSIFIED, [])]
    L += [f"Unclassified drop reasons: {len(unc)} of {T['drops']}" + (":" if unc else "."), ""]
    L += [f"- `{tid}`: {short(r, 240)}" for sub, tid, r in unc[:15]] + ([""] if unc else [])
    qa_rows = [f"- {sub}: {res['sources'][sub]['excluded_after_qa']}" for sub in subs
               if res["sources"][sub]["excluded_after_qa"]]
    if qa_rows:
        L += ["Removed after QA (excluded_after_qa.txt):", ""] + qa_rows + [""]
    # v3
    if res["v3"]:
        L += ["## v3 clean-node add-on", "",
              "targets = clean-node targets in the work list (1-2 per tree, strictly before the decisive step); skip "
              "rate = skips / targets of the written entries; late = the skip says the target step already makes "
              "the labelled step's error (`SKIP_LATE`: the decisive label is possibly late).", "",
              "| source | entries | written | add-on check passed | targets | clean nodes | skips | skip rate "
              "| already makes the labelled step's error (label possibly late) | own earlier outcome-changing error "
              "| other | late or own (share of skips) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        VT = collections.Counter()
        sk = list(SKIP_CLASSES)
        for sub, V in res["v3"].items():
            n, sc = V["n"], V["skip_classes"]
            VT.update(n)
            VT.update({f"sk/{k}": sc[k] for k in sk})
            L.append(f"| {sub} | {n['entries']} | {n['written']} | {n['passed']} | {n['targets']} | "
                     f"{n['clean_nodes']} | "
                     f"{n['skips']} | {pct(n['skips'], n['written_targets'])} | {sc[sk[0]]} | {sc[sk[1]]} | "
                     f"{sc[sk[2]]} | {sc[sk[0]] + sc[sk[1]]} ({pct(sc[sk[0]] + sc[sk[1]], n['skips'])}) |")
        both = VT['sk/' + sk[0]] + VT['sk/' + sk[1]]
        L.append(f"| all | {VT['entries']} | {VT['written']} | {VT['passed']} | {VT['targets']} | {VT['clean_nodes']} "
                 f"| {VT['skips']} | {pct(VT['skips'], VT['written_targets'])} | {VT['sk/' + sk[0]]} | "
                 f"{VT['sk/' + sk[1]]} | {VT['sk/' + sk[2]]} | {both} ({pct(both, VT['skips'])}) |")
        L.append("")
        L += ["Skip examples:", ""]
        for c in sk:
            exs = [x for V in res["v3"].values() for x in V["skip_examples"].get(c, [])]
            for tid, r in exs[:3]:
                L.append(f"- {c}: `{tid}`: {short(r, 200)}")
        L.append("")
    L += ["## Detector definitions", "", "See the docstring of `scripts/fa_trees_postcheck.py` (thresholds: "
          f"NEG_OVERLAP = {NEG_OVERLAP}; excerpts rendered at cap {ft.STEP_CAP}; numbers equal up to 1e-9 relative or "
          "a more precise number rounding to a GT with decimals).", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- --apply

def remove_nodes(nodes: list[dict], drop: set[str]) -> list[dict]:
    """Remove the nodes in drop; their children are re-attached to their parents (as free_render does)."""
    parents = {nd["id"]: list(nd.get("parents") or []) for nd in nodes}

    def lift(ps, seen=frozenset()):
        out = []
        for p in ps:
            for x in (lift(parents.get(p, []), seen | {p}) if p in drop and p not in seen else [p]):
                if x not in out and x not in drop:
                    out.append(x)
        return out
    kept = []
    for nd in nodes:
        if nd["id"] in drop:
            continue
        nd = copy.deepcopy(nd)
        nd["parents"] = lift(list(nd.get("parents") or []))
        kept.append(nd)
    return kept


def apply_filters(lines: list[dict], addon: list[dict], flags: dict, remove: set[str],
                  policy: str = "drop-node") -> tuple[list[dict], list[dict], collections.Counter]:
    """Filtered copies of tree lines and add-on lines (inputs are not modified)."""
    st = collections.Counter()
    node_flag: dict[str, str] = {}
    for kind in NODE_FLAGS:
        if kind in remove:
            for i in flags.get(kind, []):
                node_flag.setdefault(i, kind)
    drop_tree = {i for kind in TREE_FLAGS if kind in remove for i in flags.get(kind, [])}
    if policy == "drop-line":
        for i, kind in node_flag.items():
            tid, nid = i.split("#", 1)
            if not re.match(r"^c\d+$", nid):
                drop_tree.add(tid)
    out_lines = []
    for ln in lines:
        tid = ln["traj_id"]
        if tid in drop_tree:
            st["tree_lines_dropped"] += 1
            continue
        rm = {nd["id"] for nd in ln.get("nodes") or [] if f"{tid}#{nd['id']}" in node_flag}
        if rm:
            new = dict(ln, nodes=remove_nodes(ln["nodes"], rm))
            new["postcheck"] = {"removed": [{"id": x, "flag": node_flag[f"{tid}#{x}"]} for x in sorted(rm)]}
            st["tree_nodes_removed"] += len(rm)
            st["tree_lines_modified"] += 1
            out_lines.append(new)
        else:
            out_lines.append(ln)
    out_add = []
    for ln in addon:
        tid = ln["traj_id"]
        if tid in drop_tree:
            st["addon_lines_dropped"] += 1
            continue
        rm = {nd["id"] for nd in ln.get("nodes") or [] if f"{tid}#{nd['id']}" in node_flag}
        if rm:
            new = dict(ln, nodes=[copy.deepcopy(nd) for nd in ln["nodes"] if nd["id"] not in rm])
            new["postcheck"] = {"removed": [{"id": x, "flag": node_flag[f"{tid}#{x}"]} for x in sorted(rm)]}
            st["addon_nodes_removed"] += len(rm)
            st["addon_lines_modified"] += 1
            out_add.append(new)
        else:
            out_add.append(ln)
    return out_lines, out_add, st


def _forbidden_out(out: Path, inputs: list[Path]) -> str:
    o = out.resolve()
    for p in inputs:
        if p.resolve().parent == o:
            return f"--out-dir {out} holds the input {p}"
    raw = (ROOT / "data/raw").resolve()
    if o == raw or raw in o.parents:
        if any(part.startswith("src") for part in o.relative_to(raw).parts):
            return f"--out-dir {out} is a source directory"
    return ""


def do_apply(args, res: dict) -> dict:
    out = Path(args.out_dir)
    remove = set(args.remove)
    inputs = []
    jobs = []
    for sub in res["sources"]:
        tp = rp(res["sources"][sub]["cfg"]["trees"])
        if tp.exists():
            inputs.append(tp)
            jobs.append(("trees", sub, tp))
        if sub in V3_SUBSETS and rp(V3_ADDON.format(sub=sub)).exists():
            inputs.append(rp(V3_ADDON.format(sub=sub)))
            jobs.append(("addon", sub, rp(V3_ADDON.format(sub=sub))))
    why = _forbidden_out(out, inputs)
    if why:
        raise SystemExit(f"refusing to write: {why}")
    stale = [x for sub in res["sources"] for x in res["sources"][sub].get("stale", [])]
    if stale and not getattr(args, "allow_stale", False):
        raise SystemExit("refusing to filter stale files (the flags come from newer sources): " + "; ".join(stale)
                         + " (or pass --allow-stale)")
    out.mkdir(parents=True, exist_ok=True)
    summ = {"remove": sorted(remove), "policy": args.v2_node_policy, "files": {}}
    for kind, sub, p in jobs:
        if kind == "trees":
            lines, _, st = apply_filters(read_jsonl(p), [], res["flags"], remove, args.v2_node_policy)
        else:
            _, lines, st = apply_filters([], read_jsonl(p), res["flags"], remove, args.v2_node_policy)
        dst = out / p.name
        with open(dst, "w") as f:
            for ln in lines:
                f.write(json.dumps(ln, ensure_ascii=False) + "\n")
        summ["files"][str(dst)] = {"from": str(p), "lines": len(lines), **st}
    missing = [s for s in res["sources"] if not rp(res["sources"][s]["cfg"]["trees"]).exists()]
    if missing:
        summ["not_written"] = {s: "no assembled trees file yet (run extract_fa_trees.py assemble first)"
                               for s in missing}
    (out / "apply_summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    return summ


# ---------------------------------------------------------------- main

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    ap.add_argument("--trees", action="append", default=[], metavar="SUB=PATH",
                    help="tree lines of a source (default <subset>.jsonl next to its sources, else in memory)")
    ap.add_argument("--report", default="notes/next_round/failure_attribution/tree_v3_audit.md")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--flags", default=None, help="write the flagged ids per problem type here (+ flags_detail.jsonl)")
    ap.add_argument("--cap", type=int, default=ft.STEP_CAP, help="per-step cap of the index-free render")
    ap.add_argument("--apply", action="store_true", help="write filtered copies of the tree / add-on lines")
    ap.add_argument("--out-dir", default=None, help="--apply: output directory (never a src directory)")
    ap.add_argument("--remove", default=",".join(DEFAULT_REMOVE),
                    help=f"--apply: flag types to remove (of {', '.join(NODE_FLAGS + TREE_FLAGS)})")
    ap.add_argument("--v2-node-policy", choices=("drop-node", "drop-line"), default="drop-node")
    ap.add_argument("--allow-stale", action="store_true",
                    help="--apply even when a trees / add-on file is older than its newest source file")
    args = ap.parse_args(argv)
    args.subsets = [s for s in args.subsets.split(",") if s]
    bad = [s for s in args.subsets if s not in SOURCES]
    if bad:
        ap.error(f"unknown subsets {bad}")
    args.trees = dict(x.split("=", 1) for x in args.trees)
    args.remove = [x for x in args.remove.split(",") if x]
    if any(x not in NODE_FLAGS + TREE_FLAGS for x in args.remove):
        ap.error(f"--remove: unknown flag types {[x for x in args.remove if x not in NODE_FLAGS + TREE_FLAGS]}")
    if args.apply and not args.out_dir:
        ap.error("--apply needs --out-dir")
    res = run(args)
    cmdline = "python3 scripts/fa_trees_postcheck.py " + " ".join(sys.argv[1:] if argv is None else argv)
    if args.flags:
        fp = Path(args.flags)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps({k: res["flags"].get(k, []) for k in ALL_FLAGS}, indent=1) + "\n")
        with open(fp.with_name("flags_detail.jsonl"), "w") as f:
            for d in res["details"]:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    if not args.no_report:
        write_report(res, rp(args.report), args.flags, cmdline)
    summary = {k: len(res["flags"].get(k, [])) for k in ALL_FLAGS}
    print(json.dumps({"flags": summary, "kept": {s: res["sources"][s]["n"]["kept"] for s in res["sources"]}},
                     indent=1))
    if args.apply:
        print(json.dumps(do_apply(args, res), indent=1))


if __name__ == "__main__":
    main()
