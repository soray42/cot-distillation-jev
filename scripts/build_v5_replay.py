"""v5 replay set: exam-style multiple choice (4-8 options, every item's options average >= 6 words).

Diagnosis (notes/forgetting_report.md section 2.9): in data/student_v4t exam-style multiple choice (line 272: >= 4
options AND options averaging >= 8 words) was only 3.6% of rows, while short 2-3-option classification and yes/no
dominated; general reasoning (BBH, MuSR, LogiQA2, LSAT-LR) dropped. This script builds a replay set of exam-style items
from the local tasksource dump (data/raw/v4/tasksource_jev, parquet; variant == "direct", kind == "choice", one-hot
target), which the v4 builders only sampled 64 rows per source from.

Composition rule: every item has >= 4 options whose mean length is >= --min-opt-words (6) words, so no word-level
options remain; within a source, items whose options average >= 8 words (the report's exam-style definition) are
taken first. The summary reports the exam-style share by the report's definition (>= 8 words) as the headline.
Cap per source: 600, or 900 when the source has >= 1,200 raw items whose options average >= 8 words.
SuperNI (data/raw/v4/superni) was surveyed for lettered multiple choice with long options: its long-option tasks are
renderings of datasets taken here from tasksource (RACE, CaseHOLD, HEAD-QA, CODAH), excluded benchmarks (HellaSwag,
MMLU, ARC), licence UNK (MuTual), or have word-level options (QASC, AQuA, ReCoRD); no SuperNI task is used.

Screening (every considered source is listed in the summary with its outcome):
  - name patterns of data/raw/v4/exclusions.json plus our evaluation sets (LogiQA, ReClor, AGIEval/LSAT, FOLIO,
    ProverQA, ZebraLogic, Knights and Knaves, BBEH, JevBench) and CommonsenseQA derivatives (CoS-E, ECQA, X-CSR);
  - licence: a used source must have a row in data/license/per_source.csv (class PERM / ATTR / NC, empty flag).
    Sources without a row are not used; the class from the logic of classify.py is shown for information only.
    UPSTREAM records licences of the text a dataset was built from (wikiHow, DailyDialog, ...); the effective class is
    the more restrictive of the CSV class and the upstream class, and SA / UNK effective classes are dropped.
    per_source.csv files CC BY-NC-SA as NC (26 rows), and the same rule is applied to upstream NC-SA text.

Item filters: 4-8 distinct options, a single gold option, parsable question, options averaging >= --min-opt-words,
no option that is 1-3 bare letters, no option that names other options by letter ("both A and B": the shuffle would
break it), no stem that refers to a missing image or figure, rendered prompt <= --max-chars, and not on DENY (gold
label checked wrong or ambiguous). Raw records whose stem and option set occur again with a different gold are dropped.
CICERO dialogues taken from MuTual (licence UNK) are dropped: lower-cased dialogues (MuTual style) and dialogues
sharing a 13-gram with the local MuTual dump.
Options that refer to their position ("all of the above", "None of the above choices .") stay last; the other
options are shuffled with random.Random(item_id).

Reuse: items already in data/student_v4t/{train,val}.jsonl are dropped. Besides exact keys (raw item text, sorted
option set, passage >= 30 words), a containment check over the full text of every v4t row catches other renderings
(SuperNI "Input:" rows, X-CODAH sentences, label-verification and claim-twin rows): a candidate is reused if one v4t
row holds >= 95% of the stem's distinct tokens and at least one option, or every option and >= 60% of the stem tokens,
or the whole stem or the passage (>= 12 tokens) verbatim (a passage reused by another question or task counts as
reuse); plus 13-gram containment (>= 50% of the item's 13-grams in v4t).
Within the set, an item text, option set or stem-token-set + option set is used once and a passage at most twice.

Decontamination: 13-gram overlap with every file in data/eval/*.jsonl (grams() of scripts/decontam.py, grams in >= 20
evaluation items ignored); --gate then runs scripts/decontam.py on the output as the official check. Text from
excluded benchmarks: items sharing any 13-gram (grams in >= 20 reference rows ignored) with rows of excluded sources in
the local dumps (tasksource sources hit by the exclusion patterns, SWAG, and the SuperNI renderings of HellaSwag, SWAG,
OpenBookQA, ARC, CommonsenseQA, X-CSR, WinoGrande, MMLU) are dropped, as are items sharing text with the local MuTual
dump. This check runs last, on a pool over-selected by 20% per source, and counts the pool's 13-grams while the
reference is streamed (the reference has ~18M 13-grams; indexing it took 3.5 GB).

Layout follows data/eval: "<passage>\\n\\nQuestion: <question>\\nOptions:\\nA) ..." and "Question: <question>\\n
Options:..." when an item has no passage (as in gsm8k_mc). Item ids are assigned after a deterministic sort.

  python scripts/build_v5_replay.py --gate     # -> data/v5/replay_mc.jsonl, data/v5/replay_mc.summary.json
"""
from __future__ import annotations

import argparse
import array
import ast
import collections
import csv
import gc
import glob
import hashlib
import importlib.util
import json
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from decontam import grams  # noqa: E402  (same tokenizer and 13-gram hashing as the official check)

TOK = re.compile(r"[a-z0-9]+")
MIN_OPT, MAX_OPT = 4, 8
EXAM_WORDS = 8  # notes/forgetting_report.md line 272: exam-style = >= 4 options and options averaging >= 8 words
EXTRA_PATTERNS = [  # our evaluation sets and CommonsenseQA derivatives (not all are in exclusions.json)
    "logiqa", "reclor", "lsat", "agieval", "folio", "proverqa", "prover_qa", "zebra", "knights", "knave", "bbeh",
    "jevbench", "cos_e", "cos-e", "ecqa", "x_csr", "xcsr", "x-csqa", "csqa", "commonsense_qa", "commonsenseqa",
    "measuring_massive_multitask", "mmmlu"]
CLASSIFY_PATHS = [ROOT / "data/license/classify.py", Path("/home/soray/.claude/jobs/3b5d797f/tmp/license/classify.py")]
RANK = {"PERM": 0, "ATTR": 1, "NC": 2, "SA": 3, "UNK": 4}


def norm(s: str) -> str:
    return " ".join(TOK.findall(s.lower()))


def nwords(s: str) -> int:
    return len(s.split())


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def clean_block(s: str) -> str:
    s = "\n".join(line.rstrip() for line in (s or "").strip().split("\n"))
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


# ----------------------------------------------------------------------------------------------- item-level rules
LETTER_REF = re.compile(  # "both A and B", "A, B and C", "neither A nor C", "(A) or (D)"; case-sensitive (article "a")
    r"(?<![A-Za-z])\(?[A-H]\)?(?:\s*,\s*\(?[A-H]\)?)*\s*(?:and|or|nor|&)\s*\(?[A-H]\)?(?![A-Za-z])")
POSITIONAL = re.compile(r"\b(?:all|none|both|neither|any)\b.{0,30}\b(?:above|below|aforementioned)\b", re.I)
BARE_LETTERS = re.compile(r"^\(?[A-Ha-h]{1,3}\)?\.?$")
IMAGE_REF = re.compile(
    r"linked to (?:an |the )?image|\b(?:see|in|on) the (?:image|figure|picture|diagram|photo|photograph)\b|"
    r"\bshown (?:in|on|by) the (?:image|figure|picture|diagram|photo)|"
    r"\bthe (?:image|figure|picture|diagram|photo) (?:shows|below|above)\b|\bfig(?:ure|\.)\s*\d", re.I)
DENY = {  # md5(norm(context + " " + question)): gold checked wrong or ambiguous by the verifier (2026-10-03)
    "da72eef13981e016a1b8f42dffc69f95": "race/high Macao: gold 'it is an interesting place', passage supports "
                                        "'not far away from Hong Kong'",
    "354fe1e0254d7b08ad8773e2c6b62a79": "race/high Singapore 'jewel' development: ambiguous gold",
    "52a3899d6757f812514fd9cdc9646f16": "cosmos_qa New Iberia storm: ambiguous gold ('None of the above choices')",
    "5d86739d44397e9283242c33c52989e8": "cicero accounting/art dialogue: gold likely wrong",
    "c82d6af799bfecbb76c5a9f316ce4149": "race-c: raw duplicate with a different gold",
    "67c635c6a8ef2f1cc34b265c025bc8fe": "medmcqa confabulation: raw duplicate with a different gold",
}


def mutual_style(state: str) -> bool:
    """CICERO dialogue taken from MuTual: MuTual text is lower-cased (DailyDialog and DREAM are cased). On the local
    dump, 2,373 of the 2,646 CICERO dialogues sharing a 13-gram with MuTual rows are lower-cased."""
    body = (state or "").split("\n\nTarget utterance:")[0]
    utt = " ".join(re.sub(r"^[AB]: ", "", line) for line in body.split("\n"))
    return not re.search(r"[A-Z]", utt)


def positional(opt: str) -> bool:
    return nwords(opt) <= 10 and bool(POSITIONAL.search(opt))


def option_order(options: list[str], item_id: str) -> list[int]:
    """Shuffle with random.Random(item_id); options that refer to their position stay last (original order)."""
    pinned = [i for i, o in enumerate(options) if positional(o)]
    free = [i for i in range(len(options)) if i not in pinned]
    random.Random(item_id).shuffle(free)
    return free + pinned


# ----------------------------------------------------------------------------------------------- question splitting
QWORD = re.compile(r"(what|which|where|who|whom|whose|why|when|how|is|are|was|were|does|do|did|can|could|would|will|"
                   r"should|has|have|had)\b", re.I)
BOUND = re.compile(r"(?:[.!?…][\"'”’)]*|\n)\s+")


def split_question(text: str, need_context: bool) -> tuple[str, str] | None:
    """Split '<context> <question>?' at the start of the trailing question; None if that is not possible."""
    t = (text or "").strip()
    if not t.endswith("?"):
        return None
    bounds = [m.end() for m in BOUND.finditer(t) if m.end() < len(t) - 1]
    for b in reversed(bounds):
        if QWORD.match(t[b:]):
            return clean_block(t[:b]), clean(t[b:])
    if not need_context and QWORD.match(t):
        return "", clean(t)
    return None


# ----------------------------------------------------------------------------------------------- tasksource parsers
# each returns (context, question) from the row's state, or None (parse failure)
def p_race(state):
    head, _, rest = (state or "").strip().partition("\n")
    if not head.strip() or nwords(rest) < 30:
        return None
    return clean_block(rest), clean(head)


def p_trailing(need_context):
    def f(state):
        r = split_question(state, need_context)
        if r is None or (need_context and nwords(r[0]) < 15) or nwords(r[1]) > 80:
            return None
        return r
    return f


def p_cicero(state):
    body, _, last = (state or "").strip().rpartition("\n")
    if not last.strip().endswith("?") or "Target utterance:" not in body:
        return None
    return clean_block(body), clean(last)


def p_casehold(state):
    if "<HOLDING>" not in (state or ""):
        return None
    return clean_block(state), "Which holding statement best fills the <HOLDING> placeholder in the citation above?"


def p_paradise(state):
    title = clean(state)
    if not title:
        return None
    return "", f'Which piece of advice belongs to the how-to guide "{title}"?'


def p_goal(state):
    m = re.match(r"\s*Step:\s*(.+?)\s*\nGoal:\s*$", state or "", re.S)
    if not m:
        return None
    return "", f'Which goal is the step "{clean(m.group(1))}" most likely part of?'


def p_step(state):
    m = re.match(r"\s*Goal:\s*(.+?)\s*\nStep:\s*$", state or "", re.S)
    if not m:
        return None
    return "", f'Which of the following is a step toward the goal "{clean(m.group(1))}"?'


def p_stem(state):
    q = clean_block(state)
    return ("", q) if q else None


def p_codah(state):
    s = clean(state)
    return (s, "Which option is the most plausible continuation of the text above?") if s else None


# ----------------------------------------------------------------------------------------------- SuperNI option parsing
# options and passage of SuperNI inputs already in v4t (any of these renderings), for the exact-key reuse check only
def split_opts(text: str, marker: re.Pattern) -> tuple[str, list[tuple[str, str]]]:
    parts = marker.split(text)
    head, rest = parts[0], parts[1:]
    return head, [(rest[i], rest[i + 1]) for i in range(0, len(rest) - 1, 2)]


GENERIC_OPT_MARKERS = [re.compile(r"\(([A-Z])\)\s*"), re.compile(r"<(\d)>\s*"), re.compile(r"\n?\s*Option ([A-E]):\s*"),
                       re.compile(r"Completion ([A-H]):\s*"), re.compile(r"\b([a-e]) \)\s*"),
                       re.compile(r"\b([a-e])\. ")]


def _run(keys: list[str]) -> int:
    """Length of the consecutive run A, B, C ... (or 1, 2, 3 ...) at the start of keys."""
    n = 0
    for i, k in enumerate(keys):
        if k != chr(ord(keys[0]) + i) or keys[0] not in "Aa01":
            break
        n += 1
    return n


def generic_options(text: str) -> list[str]:
    """Options of a SuperNI input: the marker style with the longest consecutive run wins (so '(A)' inside a HEAD-QA
    option does not beat the '<1> <2> ...' markers)."""
    best: list[str] = []
    for mk in GENERIC_OPT_MARKERS:
        _, opts = split_opts(text, mk)
        n = _run([k for k, _ in opts])
        if n >= 3 and n > len(best):
            best = [re.sub(r"[\s,]+$", "", v) for _, v in opts[:n]]
    return best


def generic_passage(task: str, inp: str) -> str:
    m = re.match(r"\s*Article:\s*(.*?)\n\s*Question:", inp, re.S)
    if m:
        return m.group(1)
    if task.startswith("task302_") and "\n Questions:" in inp:
        return inp.partition("\n Questions:")[0]
    m = re.match(r"\s*Fact1:\s*(.*?),\s*Question:", inp, re.S)
    if m:
        return m.group(1).replace("Fact2:", " ")
    return ""


# ----------------------------------------------------------------------------------------------- source table
# slug, tasksource name, parser. Cap per source: BASE (600), or DEEP (900) when the source has >= DEEP_MIN raw items
# whose options average >= 8 words, so the size target is approached without word-level options.
BASE, DEEP, DEEP_MIN = 600, 900, 1200
SPECS = [
    ("race_high", "race/high", p_race),
    ("race_middle", "race/middle", p_race),
    ("race_c", "race-c", p_race),
    ("cosmos_qa", "cosmos_qa", p_trailing(True)),
    ("quail", "quail", p_trailing(True)),
    ("spartqa", "spartqa-mchoice", p_trailing(True)),
    ("cicero", "cicero", p_cicero),
    ("casehold", "lex_glue/case_hold", p_casehold),
    ("paradise", "PARADISE", p_paradise),
    ("wikihow_goal", "goal-step-wikihow/goal", p_goal),
    ("wikihow_step", "goal-step-wikihow/step", p_step),
    ("headqa", "head_qa/en", p_stem),
    ("codah", "codah/codah", p_codah),
    ("medmcqa", "medmcqa", p_stem),
]
TS_PARSERS = {name: p for _, name, p in SPECS}

# upstream text licences not visible in the dataset's own licence (effective class = max of CSV class and this)
WIKIHOW = ("NC", "wikiHow article text (CC BY-NC-SA 3.0 upstream); per_source.csv files NC-SA licences as NC")
UPSTREAM = {
    ("ts", "PARADISE"): WIKIHOW,
    ("ts", "goal-step-wikihow/goal"): WIKIHOW,
    ("ts", "goal-step-wikihow/step"): WIKIHOW,
    ("ts", "quail"): ("NC", "dataset licence cc-by-nc-sa-4.0; per_source.csv files NC-SA licences as NC (26 rows)"),
    ("ts", "cicero"): ("NC", "dialogues from DailyDialog (CC BY-NC-SA 4.0, filed NC), DREAM (NC) and MuTual (UNK in "
                             "per_source.csv); items sharing a 13-gram with the local MuTual dump are dropped"),
    ("ts", "cosmos_qa"): ("NC", "contexts from the ICWSM 2009 Spinn3r personal-narrative blog corpus (research-use "
                                "agreement)"),
    ("ts", "wikimedqa/medwiki"): ("SA", "Wikipedia text (CC BY-SA upstream)"),
}

# considered and not used for a format reason (licence / exclusion reasons are computed)
FORMAT_REJECTS = {
    ("ts", "sciq"): "word-level options (3 of 5,762 items average >= 6 words)",
    ("ts", "wikimedqa/medwiki"): "word-level options (16 of 14,089 items average >= 6 words)",
    ("ts", "cycic_multiplechoice"): "word-level options (268 of 6,520 items average >= 6 words)",
    ("ts", "prost"): "word-level options (object names; 0 of 6,686 items average >= 6 words)",
    ("ts", "qasc"): "word-level options (5 of 5,867 items average >= 6 words)",
    ("ts", "math_qa"): "word-level options (numbers)",
    ("ts", "multilingual/exams/multilingual"): "multilingual school exams (not English)",
    ("ts", "IntentGrasp/all"): "intent classification with a fixed label menu, not exam-style",
    ("ts", "scruples/verdict_votes"): "fixed 5-way verdict scale with soft votes",
    ("ts", "persuasion"): "fixed rating scale",
    ("ts", "cloth"): "single-sentence vocabulary cloze",
    ("ts", "wiki_hop/original"): "entity options, up to 79 per item",
    ("ts", "ekar_english"): "word-analogy triples without a passage",
    ("sni", "task1297_qasc_question_answering"): "word-level options",
    ("sni", "task750_aqua_multiple_choice_answering"): "word-level options (numbers); 95 instances with bare-letter "
                                                       "options",
    ("sni", "task302_record_classification"): "word-level options (entities)",
    ("sni", "task309_race_answer_generation"): "same items as tasksource race/* (one rendering per dataset)",
    ("sni", "task268_casehold_legal_answer_generation"): "same items as tasksource lex_glue/case_hold",
    ("sni", "task287_casehold_legal_incorrect_answer_generation"): "same items as tasksource lex_glue/case_hold",
    ("sni", "task1431_head_qa_answer_generation"): "same items as tasksource head_qa/en",
    ("sni", "task156_codah_classification_adversarial"): "same items as tasksource codah/codah",
    ("sni", "task1420_mathqa_general"): "word-level options (numbers)",
    ("sni", "task058_multirc_question_answering"): "multiple correct answers per item",
    ("sni", "task1549_wiqa_answer_generation_missing_step"): "92 instances; options are step positions",
    ("sni", "task247_dream_answer_generation"): "3 options",
    ("sni", "task385_socialiqa_incorrect_answer_generation"): "3 options",
}
# derived from an excluded benchmark although no name pattern matches (checked by 8-gram overlap with the raw dumps)
DERIVED = {
    ("ts", "dgen"): "derived from ARC: DGen cloze items were converted from science MCQ sets including the AI2/ARC "
                    "question bank; 171 of 2084 direct rows share >= 30% of their 8-grams with tasksource ai2_arc rows",
}
SCREEN_ONLY = [  # considered and screened by name pattern / licence only
    ("ts", "logiqa"), ("ts", "reclor"), ("ts", "lsat-rc"), ("ts", "lsat-ar"), ("ts", "lsat_qa/all"),
    ("ts", "hellaswag"), ("ts", "commonsense_qa"), ("ts", "openbookqa"), ("ts", "ai2_arc/ARC-Challenge/challenge"),
    ("ts", "ai2_arc/ARC-Easy/challenge"), ("ts", "multilingual/xcsr/X-CSQA-en"),
    ("ts", "multilingual/xcsr/X-CODAH-en"), ("ts", "swag/regular"), ("ts", "discosense"), ("ts", "mutual"),
    ("ts", "MedQA-USMLE-4-options-hf"), ("ts", "onestop_qa"), ("ts", "riddle_sense"), ("ts", "ScienceQA_text_only"),
    ("ts", "brainteasers/SP"), ("ts", "brainteasers/WP"), ("ts", "privacy-200k-Mistral-Large-3"), ("ts", "dream"),
    ("sni", "task073_commonsenseqa_answer_generation"), ("sni", "task1135_xcsr_en_commonsense_mc_classification"),
    ("sni", "task228_arc_answer_generation_easy"), ("sni", "task229_arc_answer_generation_hard"),
    ("sni", "task1286_openbookqa_question_answering"), ("sni", "task1389_hellaswag_completion"),
    ("sni", "task1391_winogrande_easy_answer_generation"), ("sni", "task697_mmmlu_answer_generation_formal_logic"),
    ("sni", "task043_essential_terms_answering_incomplete_questions"),
    ("sni", "task047_miscellaneous_answering_science_questions"), ("sni", "task611_mutual_multi_turn_dialogue"),
    ("sni", "task164_mcscript_question_answering_text"),
    ("sni", "task104_semeval_2019_task10_closed_vocabulary_mathematical_answer_generation"),
]
# reference text for the excluded-benchmark overlap check (besides tasksource sources hit by the exclusion patterns)
REF_TS_EXTRA = {"swag/regular": "swag (HellaSwag source corpus)", "mutual": "mutual (licence UNK; CICERO dialogues)"}
REF_SNI = re.compile(r"_(?:hellaswag|swag|openbookqa|arc|commonsenseqa|xcsr_en|winogrande|mmmlu|mutual)_")


# ----------------------------------------------------------------------------------------------- licence + exclusion
def load_classifier():
    path = next(p for p in CLASSIFY_PATHS if p.exists())
    spec = importlib.util.spec_from_file_location("license_classify", path)
    cl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cl)
    post = None  # the POST override table lives inside classify.main(); read it from the source
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "POST" for t in node.targets):
            post = ast.literal_eval(node.value)
    return cl, post or {}, str(path)


class Screen:
    def __init__(self):
        self.cl, self.post, self.cl_path = load_classifier()
        self.table = {r["source"]: r for r in csv.DictReader(open(ROOT / "data/license/per_source.csv"))}
        self.patterns = [p.lower() for p in json.load(open(ROOT / "data/raw/v4/exclusions.json"))["name_patterns"]]
        self.patterns += EXTRA_PATTERNS
        self._sni_meta: dict[str, dict] = {}

    def sni_meta(self, task: str) -> dict:
        if task not in self._sni_meta:
            t = json.load(open(ROOT / f"data/raw/v4/superni/ni/tasks/{task}.json"))
            self._sni_meta[task] = {k: t[k] for k in ("Source", "URL", "Instance License", "Input_language")}
        return self._sni_meta[task]

    def excluded(self, raw: str, name: str) -> list[str]:
        """Substring match on the source / task name and SuperNI Source (as the v4 builders); token-boundary match on
        the Hub dataset id and URLs, where a bare substring hits unrelated words ('esci' in 'openlifescienceai')."""
        if raw == "ts":
            loose, strict = [name], [str(self.cl.S.get(name, {}).get("dataset") or "")]
        else:
            m = self.sni_meta(name)
            loose, strict = [name] + [str(x) for x in m["Source"]], [str(x) for x in m["URL"]]
        loose_s, strict_s = " ".join(loose).lower(), " ".join(strict).lower()
        hits = [p for p in self.patterns if p in loose_s
                or re.search(r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])", strict_s)]
        if raw == "sni":
            src = [str(x).lower() for x in self.sni_meta(name)["Source"]]
            if "arc" in src:
                hits.append("Source=arc")
            if any("measuring_massive_multitask" in x for x in src):
                hits.append("Source=MMLU")
        return sorted(set(hits))

    def classify_logic(self, raw: str, name: str) -> dict:
        """Class from the logic of classify.py (only used for sources without a per_source.csv row)."""
        key = f"{'tasksource' if raw == 'ts' else 'superni'}/{name}"
        ds = None
        if raw == "ts":
            cls, lic, ev, flag = self.cl.ts_class(name)
        else:
            m = self.sni_meta(name)
            ds, L = m["Source"][0], m["Instance License"]
            cls, flag = self.cl.ni_map(L)
            lic, ev = " | ".join(L), "NI"
            if cls == "UNK" and ds in self.cl.NI_SRC:
                cls, lic, ev = self.cl.NI_SRC[ds]
            elif ds in self.cl.NI_DEDICATED:
                cls, lic, ev, flag = self.cl.NI_DEDICATED[ds]
            elif ds == "emotion":
                cls, lic, ev = self.cl.NI_SRC["emotion"]
            if ds == "arc":
                flag = "CONTAM"
            if ds == "x_csr" and name.startswith("task1135"):
                flag = "CONTAM?"
            if "dedicated term" in lic.lower() and ds not in self.cl.NI_DEDICATED:
                cls = "UNK"  # dedicated terms of use are UNK unless reviewed (as NI_DEDICATED does per source)
        for pre, (c2, l2, e2, f2) in self.post.items():
            if key.startswith(pre):
                if c2:
                    cls, lic, ev = c2, l2, e2
                if f2:
                    flag = f2
        if ds in ("starcon", "emo", "social_iqa") and cls == "UNK":
            cls = {"starcon": "PERM", "emo": "SA", "social_iqa": "ATTR"}[ds]
        return {"class": cls, "flag": flag, "license": lic, "evidence": ev}

    def licence(self, raw: str, name: str) -> dict:
        key = f"{'tasksource' if raw == 'ts' else 'superni'}/{name}"
        if key in self.table:
            r = self.table[key]
            out = {"csv_class": r["class"], "class": r["class"], "flag": r["flag"], "license": r["license"],
                   "evidence": r["evidence"], "via": "per_source.csv"}
        else:
            c = self.classify_logic(raw, name)
            out = {"csv_class": None, "class": c["class"], "flag": c["flag"], "license": c["license"],
                   "evidence": c["evidence"], "via": "no per_source.csv row (class from classify.py logic, "
                                                     "information only)"}
        up = UPSTREAM.get((raw, name))
        out["upstream_note"] = up[1] if up else ""
        out["effective_class"] = max(out["class"], up[0], key=RANK.get) if up else out["class"]
        return out

    def verdict(self, raw: str, name: str) -> tuple[str, dict]:
        lic = self.licence(raw, name)
        hits = self.excluded(raw, name)
        if (raw, name) in DERIVED:
            return f"excluded: {DERIVED[(raw, name)]}", lic
        if hits:
            return f"excluded name pattern {hits}", lic
        if lic["csv_class"] is None:
            return "not used: no per_source.csv row", lic
        if lic["flag"]:
            return f"licence flag {lic['flag']}", lic
        if lic["effective_class"] not in ("PERM", "ATTR", "NC"):
            return f"licence class {lic['effective_class']}", lic
        return "ok", lic


# ----------------------------------------------------------------------------------------------- v4t reuse
def opt_key(opts) -> str:
    return "|".join(sorted(norm(o) for o in opts))


def v4t_rows():
    for split in ("train", "val"):
        with open(ROOT / f"data/student_v4t/{split}.jsonl") as f:
            for line in f:
                yield json.loads(line)


def v4t_keys() -> dict[str, set]:
    """Exact keys of v4t items: raw item text, sorted option set, passage (>= 30 words)."""
    keys = {"raw": set(), "opts": set(), "passage": set()}
    n = 0
    for r in v4t_rows():
        n += 1
        p, src = r["prompt"], r.get("source", "")
        opts: list[str] = []
        passage = ""
        if p.startswith("State:\n"):
            o = p.rfind("\nOptions:\n")
            q = p.rfind("\n\nQuestion: ", 0, o if o > 0 else len(p))
            state = p[len("State:\n"):q if q > 0 else len(p)]
            keys["raw"].add(norm(state))
            ln = r.get("label_names")
            if isinstance(ln, dict):
                opts = [str(v) for v in ln.values()]
            elif isinstance(ln, list):
                opts = [str(v) for v in ln]
            elif o > 0:
                opts = re.findall(r"^[A-Z]\) (.*)$", p[o:], re.M)
            parser = TS_PARSERS.get(src[len("tasksource/"):]) if src.startswith("tasksource/") else None
            if parser:
                pr = parser(state)
                passage = pr[0] if pr else ""
        elif p.startswith("Instruction: ") and "\n\nInput: " in p:
            i = p.find("\n\nInput: ") + len("\n\nInput: ")
            j = p.rfind("\n\nQuestion: Which answer does the instruction call for?")
            inp = p[i:j if j > i else len(p)]
            keys["raw"].add(norm(inp))
            task = src.split("/", 1)[1] if src.startswith("superni/") else ""
            opts = generic_options(inp)
            passage = generic_passage(task, inp)
        else:
            keys["raw"].add(norm(p))
        if len(opts) >= 3:
            keys["opts"].add(opt_key(opts))
        if nwords(passage) >= 30:
            keys["passage"].add(norm(passage))
    print(f"v4t rows {n}: raw {len(keys['raw'])}, option sets {len(keys['opts'])}, passages {len(keys['passage'])}")
    return keys


class V4tText:
    """Full normalized text of every v4t row with a 4-gram -> row index (grams in >= `boilerplate` rows dropped) for
    retrieval. reuse() asks whether one row holds the candidate's stem and options in any rendering."""
    MIX = (0x9E3779B97F4A7C15 - (1 << 64), 0x6A09E667F3BCC909, 0x3C6EF372FE94F82B, 0x510E527FADE682D1)

    def __init__(self, boilerplate: int):
        import numpy as np
        self.np = np
        self.vocab: dict[str, int] = {}
        self.rows: list[str] = []
        ks, os_ = [], []
        for r in v4t_rows():
            toks = TOK.findall(r["prompt"].lower())
            self.rows.append(" " + " ".join(toks) + " ")
            g = np.unique(self._grams(np.array([self.vocab.setdefault(t, len(self.vocab)) for t in toks],
                                               dtype=np.int64)))
            ks.append(g)
            os_.append(np.full(len(g), len(self.rows) - 1, dtype=np.int32))
        K, O = np.concatenate(ks), np.concatenate(os_)
        del ks, os_
        order = np.argsort(K, kind="stable")
        K, O = K[order], O[order]
        del order
        _, counts = np.unique(K, return_counts=True)
        keep = np.repeat(counts < boilerplate, counts)
        self.K, self.O = K[keep], O[keep]
        del K, O, keep, counts
        print(f"v4t text index: rows {len(self.rows)}, vocab {len(self.vocab)}, 4-gram postings {len(self.K)}")

    def _grams(self, ids):
        np = self.np
        if len(ids) < 4:
            return np.zeros(0, dtype=np.int64)
        a, b, c, d = (np.int64(m) for m in self.MIX)
        with np.errstate(over="ignore"):
            g = ids[:-3] * a + ids[1:-2] * b + ids[2:-1] * c + ids[3:] * d
        ok = (ids[:-3] >= 0) & (ids[1:-2] >= 0) & (ids[2:-1] >= 0) & (ids[3:] >= 0)
        return g[ok]

    def reuse(self, stem: str, options: list[str], passage: str = "", top: int = 50) -> str | None:
        """stem, options and passage (the item's context) are normalized (norm()); returns the reason or None."""
        np = self.np
        g = [self._grams(np.array([self.vocab.get(w, -1) for w in t.split()], dtype=np.int64))
             for t in [stem] + options]
        g = np.unique(np.concatenate(g)) if g else np.zeros(0, dtype=np.int64)
        if not len(g):
            return None
        lo, hi = np.searchsorted(self.K, g, "left"), np.searchsorted(self.K, g, "right")
        hit = lo < hi
        if not hit.any():
            return None
        rows = np.concatenate([self.O[a:b] for a, b in zip(lo[hit], hi[hit])])
        ids, cnt = np.unique(rows, return_counts=True)
        stoks = stem.split()
        sset = set(stoks)
        for r in ids[np.argsort(-cnt, kind="stable")[:top]]:
            text = self.rows[int(r)]
            n_hit = sum(1 for o in options if o and f" {o} " in text)
            cov = len(sset & set(text.split())) / len(sset) if sset else 0.0
            if cov >= 0.95 and n_hit >= 1:
                return "v4t reuse: stem tokens + option in one row"
            if n_hit == len(options) and cov >= 0.6:
                return "v4t reuse: every option + stem in one row"
            if len(stoks) >= 12 and f" {stem} " in text:
                return "v4t reuse: stem verbatim in one row"
            if nwords(passage) >= 12 and f" {passage} " in text:
                return "v4t reuse: passage verbatim in one row (other question or task)"
        return None


class GramIndex:
    """13-gram index (grams() of scripts/decontam.py) of tagged reference texts; grams in >= `boilerplate` texts are
    ignored. hit() returns the tag of the first matching gram, or None."""

    def __init__(self, texts, boilerplate: int, label: str):
        import numpy as np
        self.np = np
        self.tags: list[str] = []
        tag_id: dict[str, int] = {}
        hs, own = array.array("q"), array.array("h")
        n = 0
        for tag, text in texts:
            g = grams(text)
            if not g:
                continue
            n += 1
            t = tag_id.setdefault(tag, len(tag_id))
            if t == len(self.tags):
                self.tags.append(tag)
            hs.extend(g)
            own.extend([t] * len(g))
        h, o = np.frombuffer(hs, dtype=np.int64), np.frombuffer(own, dtype=np.int16)
        uniq, first, counts = np.unique(h, return_index=True, return_counts=True)
        keep = counts < boilerplate
        self.h, self.owner = uniq[keep], o[first[keep]]
        self.stats = {"texts": n, "tags": len(self.tags), "13grams": int(len(uniq)),
                      "kept_after_boilerplate": int(keep.sum())}
        del hs, own, h, o, uniq, first, counts
        print(f"{label} index: {self.stats}")

    def frac(self, text: str) -> float:
        g = grams(text)
        if not g or not len(self.h):
            return 0.0
        a = self.np.fromiter(g, dtype=self.np.int64, count=len(g))
        idx = self.np.searchsorted(self.h, a).clip(0, len(self.h) - 1)
        return float((self.h[idx] == a).mean())

    def hit(self, text: str) -> str | None:
        g = grams(text)
        if not g or not len(self.h):
            return None
        a = self.np.fromiter(g, dtype=self.np.int64, count=len(g))
        idx = self.np.searchsorted(self.h, a).clip(0, len(self.h) - 1)
        m = self.h[idx] == a
        return self.tags[int(self.owner[idx[m][0]])] if m.any() else None


class PoolGrams:
    """13-grams of the over-selected pool, counted against streamed reference texts (the inverse of GramIndex, so the
    large reference never sits in memory). A pool gram counts if it occurs in 1 to `boilerplate` - 1 reference texts."""

    def __init__(self, texts: list[str], boilerplate: int):
        import numpy as np
        self.np, self.boilerplate = np, boilerplate
        self.item_grams = [np.fromiter(g, dtype=np.int64, count=len(g)) for g in (grams(t) for t in texts)]
        self.U = np.unique(np.concatenate(self.item_grams)) if texts else np.zeros(0, dtype=np.int64)
        self.cnt = np.zeros(len(self.U), dtype=np.int32)
        self.tag = np.full(len(self.U), -1, dtype=np.int32)
        self.tags: list[str] = []
        self._tag_id: dict[str, int] = {}
        self.n_ref = 0

    def feed(self, tag: str, text: str) -> None:
        np = self.np
        g = grams(text)
        if not g or not len(self.U):
            return
        self.n_ref += 1
        a = np.fromiter(g, dtype=np.int64, count=len(g))
        idx = np.searchsorted(self.U, a).clip(0, len(self.U) - 1)
        ii = idx[self.U[idx] == a]
        if len(ii):
            t = self._tag_id.setdefault(tag, len(self._tag_id))
            if t == len(self.tags):
                self.tags.append(tag)
            self.cnt[ii] += 1
            new = ii[self.tag[ii] < 0]
            self.tag[new] = t

    def hit(self, i: int) -> str | None:
        np = self.np
        a = self.item_grams[i]
        if not len(a):
            return None
        idx = np.searchsorted(self.U, a)
        c = self.cnt[idx]
        m = (c >= 1) & (c < self.boilerplate)
        return self.tags[int(self.tag[idx[m][0]])] if m.any() else None


def v4t_texts():
    for r in v4t_rows():
        yield "v4t", r["prompt"]


def eval_texts(pattern: str):
    for path in sorted(glob.glob(str(ROOT / pattern))):
        for line in open(path):
            if line.strip():
                yield Path(path).stem, json.loads(line).get("prompt", "")


def ts_files():
    return sorted(glob.glob(str(ROOT / "data/raw/v4/tasksource_jev/data/train-*.parquet")))


def reference_texts(ts_names: dict[str, str], sni_tasks: list[str]):
    """Text of excluded-benchmark rows (and MuTual) in the local dumps, tagged by source."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    names = pa.array(sorted(ts_names))
    for f in ts_files():
        for b in pq.ParquetFile(f).iter_batches(batch_size=5000, columns=["state", "options", "source", "variant"]):
            b = b.filter(pc.and_(pc.is_in(b.column("source"), value_set=names), pc.equal(b.column("variant"), "direct")))
            for r in b.to_pylist():
                yield ts_names[r["source"]], (r["state"] or "") + "\n" + "\n".join(str(o) for o in r["options"] or [])
    for task in sni_tasks:
        t = json.load(open(ROOT / f"data/raw/v4/superni/ni/tasks/{task}.json"))
        insts = t["Instances"]
        del t
        for inst in insts:
            yield f"superni/{task}", inst["input"]
        del insts
        gc.collect()


# ----------------------------------------------------------------------------------------------- candidates
def make_cand(slug, state, ctx, question, opts, gold, origin, min_words):
    opts = [clean(o) for o in opts]
    if not (MIN_OPT <= len(opts) <= MAX_OPT):
        return None, "option count"
    if any(not o for o in opts) or len({norm(o) for o in opts}) < len(opts):
        return None, "empty or duplicate options"
    if not question:
        return None, "parse failure"
    mw = sum(nwords(o) for o in opts) / len(opts)
    if mw < min_words:
        return None, f"options average < {min_words:g} words"
    if any(BARE_LETTERS.match(o) for o in opts):
        return None, "bare-letter option"
    if any(LETTER_REF.search(o) for o in opts):
        return None, "option names other options by letter"
    if IMAGE_REF.search(ctx) or IMAGE_REF.search(question):
        return None, "refers to a missing image or figure"
    if sum(positional(o) for o in opts) == len(opts):
        return None, "only positional options"
    if slug == "cicero" and mutual_style(state):
        return None, "CICERO dialogue from MuTual (lower-cased; MuTual licence UNK)"
    key = md5(norm(ctx + " " + question))
    if key in DENY:
        return None, "deny list (gold checked wrong or ambiguous)"
    return {"slug": slug, "raw_key": norm(state), "context": ctx, "question": question, "options": opts,
            "gold": gold, "origin": origin, "opt_words": mw}, None


def render(c: dict, order: list[int]) -> tuple[str, list[str], str]:
    letters = [chr(65 + i) for i in range(len(order))]
    head = (f"{c['context']}\n\n" if c["context"] else "") + f"Question: {c['question']}\nOptions:\n"
    prompt = head + "\n".join(f"{L}) {c['options'][k]}" for L, k in zip(letters, order))
    return prompt, letters, letters[order.index(c["gold"])]


def collect_ts(specs, k_mult, seed, min_words, drops):
    """Stream the dump once; keep a reservoir of k_mult * DEEP candidates per source and tier (>= 8-word options,
    6-8 words). Every valid raw record (before parsing) also feeds the gold-conflict table."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    by_name = {name: (slug, parser) for slug, name, parser in specs}
    res: dict[tuple, list] = collections.defaultdict(list)
    seen = collections.Counter()
    rngs = {slug: random.Random(f"{seed}-{slug}") for slug, *_ in by_name.values()}
    golds: dict[bytes, bytes] = {}
    conflicts: set[bytes] = set()
    names = pa.array(sorted(by_name))
    cols = ["state", "kind", "id", "options", "target", "source", "variant"]
    for f in ts_files():
        for b in pq.ParquetFile(f).iter_batches(batch_size=5000, columns=cols):
            b = b.filter(pc.is_in(b.column("source"), value_set=names))
            if b.num_rows == 0:
                continue
            for r in b.to_pylist():
                slug, parser = by_name[r["source"]]
                d = drops[slug]
                if r["variant"] != "direct":
                    d["variant not direct"] += 1
                    continue
                if r["kind"] != "choice":
                    d["kind not choice"] += 1
                    continue
                opts, t = [str(o) for o in (r["options"] or [])], list(r["target"] or [])
                if len(t) != len(opts) or not opts or not (MIN_OPT <= len(opts) <= MAX_OPT):
                    d["option count"] += 1
                    continue
                if max(t) != 1.0 or sum(t) != 1.0:
                    d["no single gold"] += 1
                    continue
                gi = t.index(1.0)
                rk = hashlib.md5(f"{slug}|{norm(r['state'])}|{opt_key(opts)}".encode()).digest()[:10]
                gk = hashlib.md5(norm(opts[gi]).encode()).digest()[:8]
                if golds.setdefault(rk, gk) != gk:
                    conflicts.add(rk)
                pr = parser(r["state"])
                if pr is None:
                    d["parse failure"] += 1
                    continue
                c, why = make_cand(slug, r["state"], pr[0], pr[1], opts, gi, f"tasksource/{r['source']}#{r['id']}",
                                   min_words)
                if c is None:
                    d[why] += 1
                    continue
                c["rk"] = rk
                tier = 0 if c["opt_words"] >= EXAM_WORDS else 1
                seen[slug] += 1
                seen[(slug, tier)] += 1
                lst, K = res[(slug, tier)], k_mult * DEEP
                if len(lst) < K:
                    lst.append(c)
                else:
                    j = rngs[slug].randrange(seen[(slug, tier)])
                    if j < K:
                        lst[j] = c
    print(f"gold-conflict table: {len(golds)} records, {len(conflicts)} with two golds")
    del golds
    return res, seen, conflicts


def ts_source_names() -> set[str]:
    import pyarrow.parquet as pq
    out = set()
    for f in ts_files():
        for b in pq.ParquetFile(f).iter_batches(batch_size=50000, columns=["source"]):
            out.update(b.column("source").unique().to_pylist())
    return out


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/v5/replay_mc.jsonl")
    ap.add_argument("--k-mult", type=int, default=3, help="candidates kept per source and tier before reuse filters")
    ap.add_argument("--min-opt-words", type=float, default=6, help="drop items whose options average fewer words")
    ap.add_argument("--max-chars", type=int, default=5000, help="drop items whose rendered prompt is longer")
    ap.add_argument("--max-per-passage", type=int, default=2)
    ap.add_argument("--v4t-frac", type=float, default=0.5, help="drop if this share of 13-grams is in v4t")
    ap.add_argument("--v4t-boilerplate-rows", type=int, default=50, help="4-grams in this many v4t rows are ignored")
    ap.add_argument("--evals", default="data/eval/*.jsonl")
    ap.add_argument("--boilerplate", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0, help="print N rendered items per source and stop (no filters)")
    ap.add_argument("--gate", action="store_true", help="run scripts/decontam.py on the output and record it")
    args = ap.parse_args()

    scr = Screen()
    screening, specs = {}, []
    for slug, name, parser in SPECS:
        v, lic = scr.verdict("ts", name)
        screening[f"ts:{name}"] = {"outcome": "used" if v == "ok" else v, **lic}
        if v == "ok":
            specs.append((slug, name, parser))
    for (raw, name), why in FORMAT_REJECTS.items():
        v, lic = scr.verdict(raw, name)
        screening[f"{raw}:{name}"] = {"outcome": v if v != "ok" else f"not used: {why}", **lic}
    for raw, name in SCREEN_ONLY:
        v, lic = scr.verdict(raw, name)
        screening[f"{raw}:{name}"] = {"outcome": v if v != "ok" else "screen passed but not configured", **lic}
    for raw, name in DERIVED:
        v, lic = scr.verdict(raw, name)
        screening[f"{raw}:{name}"] = {"outcome": v, **lic}
    for k, s in screening.items():
        print(f"  {k:62s} {str(s['csv_class']):4s} {s['effective_class']:4s} {str(s['flag'] or ''):9s} {s['outcome'][:90]}")

    drops: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    cands, seen, conflicts = collect_ts(specs, args.k_mult, args.seed, args.min_opt_words, drops)
    print("valid candidates per source:", {k: v for k, v in seen.items() if isinstance(k, str)})
    gc.collect()

    if args.sample:
        for slug, *_ in specs:
            for c in (cands[(slug, 0)] + cands[(slug, 1)])[:args.sample]:
                p, _, g = render(c, option_order(c["options"], c["origin"]))
                print(f"\n===== {slug} ({c['origin']}) gold {g}\n{p}")
        return

    caps = {slug: DEEP if seen.get((slug, 0), 0) >= DEEP_MIN else BASE for slug, *_ in specs}
    keys = v4t_keys()
    v4text = V4tText(args.v4t_boilerplate_rows)
    v4g = GramIndex(v4t_texts(), args.boilerplate, "v4t 13-gram")
    ev = GramIndex(eval_texts(args.evals), args.boilerplate, "eval")
    gc.collect()

    # phase 1: every filter except the excluded-text check, over-selecting each source by 20%
    used_raw, used_opts, used_bag, used_pass = set(), set(), set(), collections.Counter()
    pool: dict[str, list] = {}
    for slug, name, parser in specs:
        lst = cands.pop((slug, 0), []) + cands.pop((slug, 1), [])
        # >= 8-word options first, then 6-8 words, each in a fixed pseudo-random order
        lst.sort(key=lambda c: (0 if c["opt_words"] >= EXAM_WORDS else 1, md5(c["origin"])))
        d, take, limit = drops[slug], [], caps[slug] + max(30, caps[slug] // 5)
        for c in lst:
            if len(take) >= limit:
                d["unused (cap reached)"] += 1
                continue
            nopts = [norm(o) for o in c["options"]]
            ok_ = "|".join(sorted(nopts))
            stem = c["raw_key"] or norm(c["context"] + " " + c["question"])
            pk = norm(c["context"]) if nwords(c["context"]) >= 30 else ""
            bag = md5(" ".join(sorted(set(stem.split()))) + "||" + ok_)
            if c["rk"] in conflicts:
                d["raw duplicate with a different gold"] += 1
                continue
            if c["raw_key"] in keys["raw"]:
                d["v4t reuse: item text"] += 1
                continue
            if ok_ in keys["opts"]:
                d["v4t reuse: option set"] += 1
                continue
            if pk and pk in keys["passage"]:
                d["v4t reuse: passage"] += 1
                continue
            why = v4text.reuse(stem, nopts, norm(c["context"]))
            if why:
                d[why] += 1
                continue
            core = "\n".join([c["context"], c["question"]] + c["options"])
            if v4g.frac(core) >= args.v4t_frac:
                d[f"v4t reuse: 13-gram containment >= {args.v4t_frac}"] += 1
                continue
            if c["raw_key"] in used_raw or ok_ in used_opts or bag in used_bag:
                d["in-set duplicate"] += 1
                continue
            if pk and used_pass[pk] >= args.max_per_passage:
                d["in-set passage limit"] += 1
                continue
            prompt, _, _ = render(c, list(range(len(c["options"]))))
            if len(prompt) > args.max_chars:
                d["prompt too long"] += 1
                continue
            hit = ev.hit(prompt)
            if hit:
                d[f"decontam: {hit}"] += 1
                continue
            used_raw.add(c["raw_key"])
            used_opts.add(ok_)
            used_bag.add(bag)
            if pk:
                used_pass[pk] += 1
            take.append(c)
        pool[slug] = take
        del lst
    stats_idx = {"v4t_text": {"rows": len(v4text.rows), "postings": int(len(v4text.K))}, "v4t_13gram": v4g.stats,
                 "eval": ev.stats}
    del keys, ev, v4g, v4text, cands
    gc.collect()

    # phase 2: text shared with excluded benchmarks (tasksource sources hit by the exclusion patterns, SWAG, SuperNI
    # renderings) or with MuTual (licence UNK), counted against the pool so the reference is only streamed
    ref_ts = {}
    for name in sorted(ts_source_names()):
        if name in REF_TS_EXTRA:
            ref_ts[name] = REF_TS_EXTRA[name]
        elif "X-CODAH" in name:
            continue  # X-CSR is excluded as a family, but X-CODAH is CODAH (a used source), not CommonsenseQA text
        elif scr.excluded("ts", name) or (("ts", name) in DERIVED):
            ref_ts[name] = name
    ref_sni = sorted(Path(p).stem for p in glob.glob(str(ROOT / "data/raw/v4/superni/ni/tasks/task*.json"))
                     if REF_SNI.search(Path(p).stem + "_"))
    flat = [c for slug, *_ in specs for c in pool[slug]]
    pg = PoolGrams([render(c, list(range(len(c["options"]))))[0] for c in flat], args.boilerplate)
    for tag, text in reference_texts(ref_ts, ref_sni):
        pg.feed(tag, text)
    print(f"excluded-text reference: {len(ref_ts)} tasksource sources, {len(ref_sni)} SuperNI tasks, "
          f"{pg.n_ref} texts; pool {len(flat)} items, {len(pg.U)} 13-grams")
    for i, c in enumerate(flat):
        c["excl"] = pg.hit(i)
    stats_idx["excluded_text"] = {"reference_texts": pg.n_ref, "pool_items": len(flat), "pool_13grams": int(len(pg.U)),
                                  "tasksource_sources": ref_ts, "superni_tasks": ref_sni}
    del pg, flat
    gc.collect()

    selected: list[dict] = []
    for slug, name, parser in specs:
        d, take = drops[slug], []
        for c in pool[slug]:
            if c["excl"]:
                d[f"text shared with excluded or UNK source: {c['excl']}"] += 1
            elif len(take) >= caps[slug]:
                d["unused (cap reached)"] += 1
            else:
                take.append(c)
        take.sort(key=lambda c: md5(c["origin"]))
        for n, c in enumerate(take):
            c["item_id"] = f"v5mc-{slug}-{n}"
        selected += take
        print(f"{slug:14s} supply {seen.get(slug, 0):6d} (>=8w {seen.get((slug, 0), 0):6d}) cap {caps[slug]} "
              f"-> {len(take)}  {dict(d)}")
    del pool
    gc.collect()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = collections.defaultdict(lambda: collections.Counter())
    n_opts, gold_pos = collections.Counter(), collections.Counter()
    pinned = collections.Counter()
    with open(out, "w") as f:
        for c in selected:
            order = option_order(c["options"], c["item_id"])
            prompt, letters, g = render(c, order)
            f.write(json.dumps({"item_id": c["item_id"], "source": f"replay/{c['slug']}", "prompt": prompt,
                                "labels": letters, "gold_label": g,
                                "teacher": {L: (1.0 if L == g else 0.0) for L in letters},
                                "subqs": [], "random_subqs": [], "augment": "replay_mc"}) + "\n")
            s = stats[c["slug"]]
            s["items"] += 1
            s["opt_words"] += c["opt_words"]
            s["prompt_chars"] += len(prompt)
            s["prompt_words"] += nwords(prompt)
            s["with_passage"] += bool(c["context"])
            s["exam_style"] += c["opt_words"] >= EXAM_WORDS
            s["ge6"] += c["opt_words"] >= 6
            s["lt3"] += c["opt_words"] < 3
            n_opts[len(letters)] += 1
            gold_pos[g] += 1
            pin = [i for i, o in enumerate(c["options"]) if positional(o)]
            if pin:
                pinned["items"] += 1
                pinned["gold_on_pinned"] += c["gold"] in pin

    spec_by_slug = {slug: name for slug, name, _ in specs}
    total = len(selected)
    by_source, by_class, by_csv = {}, collections.Counter(), collections.Counter()
    for slug, s in stats.items():
        name, cap = spec_by_slug[slug], caps[slug]
        lic = screening[f"ts:{name}"]
        by_class[lic["effective_class"]] += s["items"]
        by_csv[lic["csv_class"]] += s["items"]
        by_source[slug] = {
            "origin": f"tasksource/{name}", "items": s["items"], "cap": cap,
            "cap_reason": f"{'DEEP' if cap == DEEP else 'BASE'}: {seen.get((slug, 0), 0)} raw items with >= 8-word "
                          f"options ({'>=' if cap == DEEP else '<'} {DEEP_MIN})",
            "csv_class": lic["csv_class"], "effective_class": lic["effective_class"], "licence": lic["license"],
            "upstream_note": lic["upstream_note"],
            "mean_option_words": round(s["opt_words"] / s["items"], 2),
            "exam_style_share": round(s["exam_style"] / s["items"], 3),
            "share_with_passage": round(s["with_passage"] / s["items"], 3),
            "mean_prompt_chars": round(s["prompt_chars"] / s["items"]),
            "mean_prompt_words": round(s["prompt_words"] / s["items"], 1),
            "valid_supply": seen.get(slug, 0), "valid_supply_ge8w": seen.get((slug, 0), 0),
            "drops": dict(drops[slug])}
    drop_tot = collections.Counter()
    for d in drops.values():
        drop_tot.update(d)
    tot = lambda k: sum(s[k] for s in stats.values())  # noqa: E731
    summary = {
        "file": args.out, "items": total, "sources": len(by_source),
        "exam_style_definition": "notes/forgetting_report.md line 272: >= 4 options AND options averaging >= 8 words",
        "exam_style_items": tot("exam_style"),
        "exam_style_share": round(tot("exam_style") / max(total, 1), 3),
        "share_items_option_words_ge6": round(tot("ge6") / max(total, 1), 3),
        "items_option_words_lt3": tot("lt3"),
        "mean_option_words": round(tot("opt_words") / max(total, 1), 2),
        "share_with_passage": round(tot("with_passage") / max(total, 1), 3),
        "mean_prompt_chars": round(tot("prompt_chars") / max(total, 1)),
        "mean_prompt_words": round(tot("prompt_words") / max(total, 1), 1),
        "by_licence_class": dict(by_class),
        "by_csv_class": dict(by_csv),
        "by_option_count": {str(k): v for k, v in sorted(n_opts.items())},
        "gold_letter": dict(sorted(gold_pos.items())),
        "positional_options": {**dict(pinned), "rule": "options such as 'all of the above' stay last; the others are "
                                                        "shuffled with random.Random(item_id)"},
        "by_source": dict(sorted(by_source.items(), key=lambda kv: -kv[1]["items"])),
        "drops_by_reason": dict(drop_tot.most_common()),
        "source_screening": screening,
        "indexes": stats_idx,
        "deny_list": DENY,
        "layout": "'<passage>\\n\\nQuestion: <question>\\nOptions:\\nA) ...'; items without a passage start at "
                  "'Question:' (as data/eval/gsm8k_mc.jsonl)",
        "licence_classifier": scr.cl_path,
        "params": vars(args),
    }
    summ_path = out.with_suffix(".summary.json")
    summ_path.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("items", "sources", "exam_style_share", "exam_style_items",
                                              "share_items_option_words_ge6", "items_option_words_lt3",
                                              "by_licence_class", "by_csv_class", "by_option_count",
                                              "mean_option_words", "share_with_passage", "mean_prompt_chars",
                                              "positional_options", "drops_by_reason")}, indent=1))

    if args.gate:
        r = subprocess.run(["nice", "-n", "10", sys.executable, str(ROOT / "scripts/decontam.py"), str(out),
                            "--evals", args.evals, "--boilerplate", str(args.boilerplate)],
                           capture_output=True, text=True, cwd=ROOT)
        print(r.stdout, r.stderr)
        clean_p, bad_p = out.with_suffix(".clean.jsonl"), out.with_suffix(".contaminated.jsonl")
        n_bad = sum(1 for _ in open(bad_p)) if bad_p.exists() else -1
        summary["official_decontam_gate"] = {"command": f"python scripts/decontam.py {args.out}",
                                             "stdout": r.stdout.strip(), "returncode": r.returncode,
                                             "contaminated": n_bad}
        if r.returncode == 0 and n_bad == 0:
            clean_p.unlink(missing_ok=True)   # identical to the output
            bad_p.unlink(missing_ok=True)
        summ_path.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
