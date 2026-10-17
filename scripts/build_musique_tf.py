"""Tree Forcing data for a multi-hop reading domain: MuSiQue-Ans (Trivedi et al., TACL 2022; CC BY 4.0), the answerable
subset as mirrored on the Hugging Face hub (dgslibisey/MuSiQue @ HF_REV: musique_ans_v1.0_train.jsonl, 19,938 questions;
musique_ans_v1.0_dev.jsonl, 2,417; downloaded to data/raw/musique/ and read as data only, json.loads per line). The
trees are read off MuSiQue's own question_decomposition (per hop: a single-hop sub-question, its answer and its
supporting paragraph); no model writes or labels anything.

Finals: "<paragraphs>\n\nQuestion: <the multi-hop question, verbatim>\nOptions:\nA) ..\nE) ..", five options, labels
A-E as GSM8K, teacher one-hot on the gold. The context is MuSiQue's own paragraph list: every supporting paragraph
plus distractor paragraphs of the same item, rendered "Title: <title>\n<text>" (whitespace collapsed), blocks separated
by a blank line, in a seeded random order (a fixed random key per paragraph, so the supporting ones are anywhere).
Distractor paragraphs are added in a seeded random order while every view of the item fits the token budget:
max(final view, GFP node view of the item's longest node question) <= --max-len - --margin tokens (Qwen3.5 tokenizer,
TOKENIZER; FINAL_TEMPLATE / SUBQ_TEMPLATE of cotdistill.student; the GFP view, the full prompt + a node question, is
the longest view any arm or --eval-subq builds, so GFPS / GFPQ views fit too). The margin covers a special prefix
(Gemma's BOS) and tokenizer differences. Items whose supporting paragraphs alone do not fit are dropped (counted).
source "musique_tf", group "musique_tf/s<n>" (n = number of hops = n_steps = depth); musique_type (2hop, 3hop1, ...).

Options (gold + 4 distractors of the gold's answer class): the class is read off the final hop: the answer's form
(year, date, number with digits), else its relation for a "X >> relation" hop (REL), else the natural sub-question's
first wh-word (who: person; where: place; what / which + a type noun within 5 words: that noun's class, NOUN_CLASS;
when / how many with a non-numeric answer: "other"); an answer that is a paragraph title takes the kind its lead
sentence gives it (lead_type: "X is a city in ...": place; "X (born 1970) is an American singer": person; country,
org, work), since e.g. "who won ..." is often answered by a team. Candidates, each with a tier:
  "hop"      answers of the item's other hops of the same class;
  "context"  strings of the shown paragraphs whose class is the gold's: answers of the split's pool (a string counts for
             the class it answers most often; a name only as a whole name, "David" does not occur in "David Bowie"),
             the shown paragraphs' titles and proper-noun spans typed by their lead sentence or the pool, and for years /
             dates / numbers every such expression of the paragraphs;
  "span"     in-context strings no type is known for: a person-like name span (2-3 words, no place / organisation /
             work word, month or demonym), a span the text uses as a place ("in X", "X, Minnesota"), a span with an
             organisation word; for the mixed class "other", strings of the gold's shape (other_like: name spans and
             titles for a name, number / decade / century expressions for digits, number words, else clause fragments
             within 2 words of the gold's length); language-like words for a language;
  "pool"     answers of other items of the class, not in the paragraphs; "any": any pool answer (counted, rare).
The four best by score: the tier's cost (TIER_COST), the gap in surface shape to the gold (article, number of content
words, capitalisation, digits) and in length, -0.3 for the gold's fine key (relation or type noun), a random jitter;
so distractors are in the paragraphs and not the odd one out by form (summary.json cues_*). A candidate is rejected when
it names the gold or an alias (same_answer: equal, containing or contained at word level after normalising case,
accents and articles, or an acronym of it), repeats a chosen option, or occurs in the question.
Pools: train items draw from the train split only; val from train + val; test (MuSiQue dev) from train + dev, so no
val / test answer is ever a train option. The gold's position cycles over A-E in a seeded order over the split
(balanced), the distractors fill the other positions in random order.

Nodes (each a yes/no question with an exact truth by the decomposition; p_cot = truth): per hop j that is not the
last one, "s<j>v" "<hop question> Is it <answer>?" (Yes) and "s<j>n" the same with a wrong answer (No): every item,
depth and hop is exactly 50% Yes. "opposite" holds "Is it different from <x>?". The hop question: a natural
sub-question verbatim (first letter upper-cased, "?" appended when it has no end mark); a "X >> relation" hop through
the relation's question template (REL, e.g. "Who is the performer of X?"; an unlisted relation: "What is the
<relation> of X?"). The wrong answer has the hop's answer class and is chosen as a distractor is (same candidates,
tiers and score, against the hop's answer); it never names the hop's answer, the gold or an alias (same_answer), never
occurs in the node question and never in the hop's own supporting paragraph (so it is false by the context also for
relations with several true values: "a cast member of X"). "wrong" records its tier.
Placeholders ("#k": hop k's answer; only k < j counts, so a literal "#9 Dream" stays): --placeholders describe (the
default) never writes another hop's answer into a node question, as GSM8K's Socratic sub-questions never state an
earlier result and as GFPS asks every node with no premise written out (substituting would turn GFPS into a TFL-like
view with the parents' results stated). "#k" becomes a description of hop k: "the <relation> of <X>" for a
"X >> relation" hop (REL's noun phrase), "the ..." for a natural "What / Who / Which is the ...?" hop, recursively
(rendering "inline"); a referenced hop with no such phrase (e.g. "Who wrote X?") stays "#k" and the node opens with its
definition, in hop order: 'Let #1 be the answer to "Who wrote X?". In which city did #1 die?' (rendering "defined";
a definition's own references are described or defined the same way). Either way no earlier answer is written and the
node is resolved only from the paragraphs, like the final question. --placeholders
substitute writes the earlier hops' answers instead (field "resolved" always holds that text, for audits and a
one-flag rebuild). depends_on = the Yes nodes of the referenced hops that got nodes; node_depth = the hop's level in
the decomposition graph (0 without references, else 1 + the deepest referenced hop), as GSM8K's step level.
Not built (counted in summary.json, build.*): the last hop (it yields the final answer: the final question in
disguise; always); a hop whose answer names the gold or an alias (same_answer, so also "Toronto" for the gold
"Toronto Coach Terminal": the Yes node would point at an option; blocked_gold_valued) and every hop downstream of one
(blocked_downstream_of_gold_valued); a hop whose answer is not a whole-word span of its supporting paragraph; a node
whose question contains the gold or an alias, or is the final question (normalised); a node whose question repeats
another node's; a hop with no wrong answer. Items are dropped when the gold is not a whole-word span of its supporting
paragraph ("bio" inside "biological") or the supporting paragraphs alone exceed the budget.
Every item is re-verified from its own fields (audit_item): prompt shape and the treeplan cut, options, one-hot
teacher, gold not repeated nor aliased among the options, Yes/No twins, truth = (probe == value), the value in its
supporting paragraph and the No probe not, no unresolved placeholder, parents are Yes nodes, node_depth >= 1 + deepest
parent, the last hop never asked.

Splits (question level; MuSiQue's test labels are hidden, so the official dev split is the test file):
  data/student_musique_tf/train.jsonl   MuSiQue train minus val
  data/student_musique_tf/val.jsonl     --n-val (400) problems of MuSiQue train, stratified by musique_type, seeded;
                                        disjoint from train by question id and question text. MuSiQue reuses
                                        single-hop questions across its composed questions, so val shares hops with
                                        train (reported, not dropped: it would cost ~30% of train); the official dev
                                        split shares no sub-question id with train.
  data/eval/musique_tf_test.jsonl       MuSiQue-Ans dev, the same row format (finals with their nodes)
  data/student_musique_tf_f25/          a 25% train subset stratified by group (n hops), the recipe of
                                        data/student_gsm8k_tf_f25 (one random.Random("musique_tf|f25|<date>"), groups
                                        sorted, each shuffled and cut to max(1, round(0.25 n)), file order kept); val
                                        copied byte for byte.
Duplicate question texts within a split keep the first (a text with two answers is dropped).

Reports (summary.json): source, revision, licence and raw md5s; item drops by reason; hop distribution; node stats and
renderings; option kinds and gold positions; token lengths per view vs --max-len and epoch tokens (A2 vs GFPS /
GFPQ); leak counts (teacher.leaks_answer as GSM8K reports it); single-feature cue stumps (option / probe in the
shown paragraphs); val / train hop overlap; md5 of every file.

  CUDA_VISIBLE_DEVICES="" ~/.venvs/sev/bin/python scripts/build_musique_tf.py    # tokenizers (stdlib otherwise)
"""
from __future__ import annotations

import argparse
import ast
import collections
import functools
import hashlib
import json
import math
import random
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.treeplan import musique_tf_cut, problem_no_options, statement_block  # noqa: E402

SOURCE = "musique_tf"
TREE_SOURCE = "musique_decomposition_tf"
RAW_DIR = "data/raw/musique"
RAW_FILES = {"train": "musique_ans_v1.0_train.jsonl", "test": "musique_ans_v1.0_dev.jsonl"}
HF_REPO = "dgslibisey/MuSiQue"
HF_REV = "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321"
LICENCE = ("CC BY 4.0 (MuSiQue-Ans v1.0, Trivedi et al. 2022, StonyBrookNLP/musique; HF mirror dgslibisey/MuSiQue "
           "@ c8f4f8c)")
TOKENIZER = "results/weights/T1-Qwen3.5-2B-Base-student_v3-s2/model/tokenizer.json"
LABELS = ["A", "B", "C", "D", "E"]
N_DISTRACTORS = 4

# ------------------------------------------------------------------------------------------------------ relations
# "X >> relation" hops (Wikidata properties): relation -> (question template, noun phrase template or None, class).
# {X} is the subject. A noun phrase describes the hop's answer when a later hop refers to it ("#k"); None: no phrase
# (the later node defines "#k" instead). Unlisted relations: "What is the <rel> of {X}?", "the <rel> of {X}", "other".
REL = {
    "country": ("Which country is {X} in?", "the country of {X}", "country"),
    "place of birth": ("Where was {X} born?", "the place of birth of {X}", "place"),
    "located in the administrative territorial entity": (
        "In which administrative territorial entity is {X} located?",
        "the administrative territorial entity containing {X}", "place"),
    "performer": ("Who is the performer of {X}?", "the performer of {X}", "person"),
    "place of death": ("Where did {X} die?", "the place of death of {X}", "place"),
    "country of citizenship": ("What is the country of citizenship of {X}?", "the country of citizenship of {X}",
                               "country"),
    "author": ("Who is the author of {X}?", "the author of {X}", "person"),
    "headquarters location": ("Where is the headquarters of {X} located?", "the headquarters location of {X}", "place"),
    "employer": ("Who is the employer of {X}?", "the employer of {X}", "org"),
    "member of sports team": ("Which sports team is {X} a member of?", "the sports team {X} is a member of", "org"),
    "educated at": ("Where was {X} educated?", "the institution where {X} was educated", "org"),
    "shares border with": ("Which place shares a border with {X}?", "a place that shares a border with {X}", "place"),
    "director": ("Who is the director of {X}?", "the director of {X}", "person"),
    "continent": ("Which continent is {X} on?", "the continent of {X}", "continent"),
    "composer": ("Who is the composer of {X}?", "the composer of {X}", "person"),
    "part of": ("What is {X} part of?", "the whole that {X} is part of", "other"),
    "creator": ("Who is the creator of {X}?", "the creator of {X}", "person"),
    "location of formation": ("Where was {X} formed?", "the place where {X} was formed", "place"),
    "cast member": ("Who is a cast member of {X}?", "a cast member of {X}", "person"),
    "location": ("Where is {X} located?", "the location of {X}", "place"),
    "owned by": ("Who owns {X}?", "the owner of {X}", "org"),
    "operator": ("Who operates {X}?", "the operator of {X}", "org"),
    "producer": ("Who is the producer of {X}?", "the producer of {X}", "person"),
    "lyrics by": ("Who wrote the lyrics of {X}?", "the lyricist of {X}", "person"),
    "screenwriter": ("Who is the screenwriter of {X}?", "the screenwriter of {X}", "person"),
    "record label": ("What is the record label of {X}?", "the record label of {X}", "org"),
    "residence": ("Where does {X} reside?", "the residence of {X}", "place"),
    "manufacturer": ("Who is the manufacturer of {X}?", "the manufacturer of {X}", "org"),
    "spouse": ("Who is the spouse of {X}?", "the spouse of {X}", "person"),
    "located on terrain feature": ("On which terrain feature is {X} located?", "the terrain feature {X} is located on",
                                   "place"),
    "military branch": ("Which military branch does {X} belong to?", "the military branch of {X}", "org"),
    "member of political party": ("Which political party is {X} a member of?", "the political party of {X}", "org"),
    "work location": ("Where did {X} work?", "the work location of {X}", "place"),
    "developer": ("Who is the developer of {X}?", "the developer of {X}", "org"),
    "conflict": ("Which conflict did {X} take part in?", "the conflict {X} took part in", "event"),
    "contains administrative territorial entity": ("Which administrative territorial entity is located in {X}?",
                                                   "an administrative territorial entity located in {X}", "place"),
    "original broadcaster": ("What is the original broadcaster of {X}?", "the original broadcaster of {X}", "org"),
    "publisher": ("Who is the publisher of {X}?", "the publisher of {X}", "org"),
    "located in or next to body of water": ("Which body of water is {X} located in or next to?",
                                            "the body of water {X} is located in or next to", "place"),
    "member of": ("What is {X} a member of?", "the organization {X} is a member of", "org"),
    "father": ("Who is the father of {X}?", "the father of {X}", "person"),
    "mother": ("Who is the mother of {X}?", "the mother of {X}", "person"),
    "child": ("Who is a child of {X}?", "a child of {X}", "person"),
    "sibling": ("Who is a sibling of {X}?", "a sibling of {X}", "person"),
    "mouth of the watercourse": ("Into which body of water does {X} flow?", "the body of water {X} flows into",
                                 "place"),
    "named after": ("What is {X} named after?", "the namesake of {X}", "other"),
    "licensed to broadcast to": ("Which place is {X} licensed to broadcast to?",
                                 "the place {X} is licensed to broadcast to", "place"),
    "country of origin": ("What is the country of origin of {X}?", "the country of origin of {X}", "country"),
    "applies to jurisdiction": ("Which jurisdiction does {X} apply to?", "the jurisdiction {X} applies to", "place"),
    "present in work": ("In which work does {X} appear?", "the work {X} appears in", "work"),
    "capital of": ("What is {X} the capital of?", "the place whose capital is {X}", "place"),
    "capital": ("What is the capital of {X}?", "the capital of {X}", "place"),
    "sport": ("Which sport is {X} associated with?", "the sport of {X}", "other"),
    "main subject": ("What is the main subject of {X}?", "the main subject of {X}", "other"),
    "winner": ("Who won {X}?", "the winner of {X}", "person"),
    "followed by": ("What followed {X}?", "the successor of {X}", "other"),
    "follows": ("What did {X} follow?", "the predecessor of {X}", "other"),
    "original language of film or TV show": ("What is the original language of {X}?", "the original language of {X}",
                                             "language"),
    "part of the series": ("Which series is {X} part of?", "the series {X} is part of", "work"),
    "platform": ("On which platform was {X} released?", "the platform of {X}", "other"),
    "narrative location": ("Where is {X} set?", "the narrative location of {X}", "place"),
    "instance of": ("What is {X} an instance of?", None, "other"),
    "genre": ("What is the genre of {X}?", "the genre of {X}", "other"),
    "distributed by": ("Who distributed {X}?", "the distributor of {X}", "org"),
    "participant": ("Who is a participant of {X}?", "a participant of {X}", "org"),
    "languages spoken, written or signed": ("Which language does {X} speak, write or sign?",
                                            "the language {X} speaks, writes or signs", "language"),
    "founded by": ("Who founded {X}?", "the founder of {X}", "person"),
    "crosses": ("What does {X} cross?", "the feature {X} crosses", "place"),
    "religion": ("What is the religion of {X}?", "the religion of {X}", "other"),
    "instrument": ("Which instrument does {X} play?", "the instrument {X} plays", "other"),
    "has part": ("What is a part of {X}?", "a part of {X}", "other"),
    "language of work or name": ("What is the language of {X}?", "the language of {X}", "language"),
    "production company": ("What is the production company of {X}?", "the production company of {X}", "org"),
    "native language": ("What is the native language of {X}?", "the native language of {X}", "language"),
    "occupant": ("Who is the occupant of {X}?", "the occupant of {X}", "org"),
    "position held": ("What position did {X} hold?", "the position held by {X}", "other"),
}
NOUN_CLASS = {}
for _c, _ws in {"country": "country countries nation nations kingdom empire",
                "continent": "continent continents",
                "place": ("city cities town towns village state states province county counties region district island "
                          "place location area territory capital municipality neighborhood borough parish ocean sea "
                          "river lake mountain mountains"),
                "language": "language languages",
                "org": ("company organization organisation team band group club university college school label network "
                        "party agency institution army navy league station channel church newspaper airline "
                        "corporation firm studio business"),
                "person": ("person man woman president king queen emperor author writer director actor actress singer "
                           "artist composer player leader musician rapper"),
                "year": "year"}.items():
    for _w in _ws.split():
        NOUN_CLASS[_w] = _c
MONTHS = ("January February March April May June July August September October November December").split()
WEEKDAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
_MONTH = r"(?:%s)" % "|".join(MONTHS)
YEAR_ANS = re.compile(r"^(?:in |c\. ?|circa )?\d{3,4}(?: ?(?:BC|BCE|AD|CE))?$", re.I)
DATE_ANS = re.compile(r"\b%s\b.*\d|\d.*\b%s\b" % (_MONTH, _MONTH), re.I)
NUM_ANS = re.compile(r"^(?:about |around |approximately |over |nearly |more than |some |c\. )?[$£€]?\s?\d[\d,.]*"
                     r"(?:\s?(?:%|percent|million|billion|thousand|hundred))?(?:\s?[-–]\s?\d[\d,.]*)?"
                     r"(?:\s[a-z]+){0,2}$", re.I)
CTX_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
CTX_DATE = re.compile(r"\b(?:\d{1,2} %s,? \d{4}|%s \d{1,2},? \d{4}|%s \d{4})\b" % (_MONTH, _MONTH, _MONTH))
CTX_NUM = re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w,])|(?<![\w.,])\d+(?:\.\d+)?(?![\w.,])")
PLACEHOLDER = re.compile(r"#(\d+)")
WH = re.compile(r" (what year|which year|how many|how much|when|what|which|who|whom|whose|where) ")
WTOK = re.compile(r"[a-z0-9]+")
STOP = frozenset("a an the of in on at to for from by with and or is are was were be".split())
NATURAL_NP = re.compile(r"^(?:what|who|which) (?:is|was|are|were) ((?:the|a|an) [^?]+? of [^?]+?)\s*[?.]?$", re.I)
NP_BAD_END = re.compile(r"\b(?:to|by|with|from|at|in|on|for|of|about|into|as|than|after|before|did|does|do|is|was|"
                        r"are|were|has|have|had)$", re.I)
ENTITY_CLASSES = frozenset({"person", "place", "country", "continent", "org", "language"})


# ------------------------------------------------------------------------------------------------------ text
def clean(s: str) -> str:
    return " ".join(str(s).split())


def norm(s: str) -> str:
    """Lower case, accents stripped, alphanumeric words, leading articles dropped."""
    s = unicodedata.normalize("NFKD", s.replace("’", "'"))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    w = WTOK.findall(s)
    while w and w[0] in ("the", "a", "an"):
        w = w[1:]
    return " ".join(w)


def overlaps(a: str, b: str) -> bool:
    """a and b are the same answer at word level: equal, or one contains the other as a word sequence (normalised)."""
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return na == nb
    return na == nb or f" {na} " in f" {nb} " or f" {nb} " in f" {na} "


def acronym_of(a: str, b: str) -> bool:
    """a is the initials of b's capitalised words ("NES" / "Nintendo Entertainment System", "UK" / "United Kingdom")."""
    x = re.sub(r"[.\s]", "", re.sub(r"^(?:the|The)\s+", "", a.strip()))
    if not (2 <= len(x) <= 6 and x.isupper() and x.isalpha()):
        return False
    caps = [w[0] for w in re.findall(r"[A-Za-z][\w'-]*", b) if w[0].isupper() and w.lower() not in STOP]
    return len(caps) >= 2 and "".join(caps) == x


def same_answer(a: str, b: str) -> bool:
    """a and b may name the same thing: word-level overlap (overlaps) or one is the other's acronym."""
    return overlaps(a, b) or acronym_of(a, b) or acronym_of(b, a)


NAME_STOP = frozenset("""a an the of in on at to for from by with and or is are was were be as after before during
when while since under between its his her their this that these those it he she they we its which who whom whose
but not also both either neither nor than then there here following including later however today eventually
meanwhile originally recently formerly currently initially afterwards subsequently finally thus therefore""".split())
_PREV_WORD = re.compile(r"([A-Za-z][\w'’.-]*)( *)$")
_NEXT_WORD = re.compile(r"^( *)([A-Za-z][\w'’-]*)")


def has_name(text: str, name: str) -> bool:
    """name occurs in the text as a whole name: whole words (case as given, or its first letter flipped), and neither
    the word right before nor the word right after is a capitalised word other than a function word ("David" in
    "David Bowie", "Jersey" in "New Jersey" do not count)."""
    for cand in {name, name[:1].swapcase() + name[1:]}:
        i = text.find(cand)
        while i >= 0:
            j = i + len(cand)
            ok = (i == 0 or not (text[i - 1].isalnum() or text[i - 1] in "-'’")) and (
                j >= len(text) or not (text[j].isalnum() or text[j] in "-'’"))
            if ok:
                m = _PREV_WORD.search(text[max(0, i - 40):i])
                if m and m.group(1)[:1].isupper() and m.group(1).lower().rstrip(".") not in NAME_STOP:
                    ok = False
                m = _NEXT_WORD.match(text[j:j + 40])
                if m and m.group(2)[:1].isupper() and m.group(2).lower() not in NAME_STOP:
                    ok = False
            if ok:
                return True
            i = text.find(cand, i + 1)
    return False


def has_phrase(text_low: str, phrase: str) -> bool:
    """phrase occurs in the (lower-cased) text as whole words."""
    p = phrase.lower()
    if not p:
        return False
    i = text_low.find(p)
    while i >= 0:
        a, b = i - 1, i + len(p)
        if (a < 0 or not text_low[a].isalnum()) and (b >= len(text_low) or not text_low[b].isalnum()):
            return True
        i = text_low.find(p, i + 1)
    return False


def ask_text(q: str) -> str:
    q = re.sub(r"\s+([?,.!;:])", r"\1", clean(q))
    q = q[:1].upper() + q[1:]
    return q if q[-1:] in ("?", ".", "!") else q + "?"


def node_question(q: str, x: str) -> str:
    return f"{q} Is it {x}?"


def node_opposite(q: str, x: str) -> str:
    return f"{q} Is it different from {x}?"


# ------------------------------------------------------------------------------------------------------ hops
def refs(q: str, j: int) -> list[int]:
    """Hop numbers (1-based) that hop j's text refers to: "#k" with k < j only (a title such as "#9 Dream" stays)."""
    return sorted({int(k) for k in PLACEHOLDER.findall(q) if 0 < int(k) < j})


def sub_refs(text: str, j: int, fn) -> str:
    """text with every placeholder "#k" (k < j) replaced by fn(k); an article in front of the placeholder is dropped
    when the replacement opens with one ("the the edict" -> "the edict")."""
    def one(m):
        k = int(m.group(2))
        if not 0 < k < j:
            return m.group(0)
        x = fn(k)
        if m.group(1) and re.match(r"(?:the|a|an) ", x, re.I):
            return x
        return (m.group(1) or "") + x
    return re.sub(r"(\b(?:[Tt]he|[Aa]n?) )?#(\d+)", one, text)


def split_triple(q: str) -> tuple[str, str] | None:
    if ">>" not in q:
        return None
    subj, _, rel = q.rpartition(">>")
    return clean(subj), clean(rel)


def hop_class(q: str, answer: str) -> tuple[str, str]:
    """(class, fine key) of a hop's answer: the answer's form (year, date, number), else the relation of a triple, else
    a wh-word / type noun of the natural question; fine key: the relation, type noun or first two words."""
    t = split_triple(q)
    fine = t[1] if t else (" ".join(WTOK.findall(q.lower())[:2]) or "?")
    a = clean(answer)
    if YEAR_ANS.match(a):
        return "year", "year"
    if DATE_ANS.search(a):
        return "date", "date"
    if NUM_ANS.match(a) and not re.search(r"[A-Z]", a):
        return "number", "number"
    if t:
        return (REL[t[1]][2] if t[1] in REL else "other"), fine
    low = " " + " ".join(WTOK.findall(q.lower())) + " "
    m = WH.search(low)                                      # the first wh-word decides
    if not m:                                               # no wh-word: a leading type noun ("country in the ...")
        first = low.split()[:2]
        for w in first:
            if w in NOUN_CLASS:
                return NOUN_CLASS[w], w
        return "other", fine
    wh = m.group(1)
    if wh in ("when", "what year", "which year"):           # a time answer not of a date / year form
        return "other", "when"
    if wh in ("how many", "how much"):                      # a quantity in words ("two", "a dozen")
        return "other", "how many"
    if wh in ("what", "which"):
        for w in low[m.end():].split()[:5]:
            if w in NOUN_CLASS:
                return NOUN_CLASS[w], w
        return "other", fine
    if wh in ("who", "whom", "whose"):
        return "person", "who"
    return "place", "where"


def render_hop(q: str) -> str:
    """A hop's own question text, placeholders kept: a triple through REL, a natural question verbatim."""
    t = split_triple(q)
    if t:
        tmpl = REL.get(t[1], ("What is the %s of {X}?" % t[1], None, None))[0]
        return tmpl.replace("{X}", t[0])
    return ask_text(q)


def np_template(hops: list[dict], k: int) -> tuple[str, str] | None:
    """(noun phrase template with {X}, subject text with its placeholders) for hop k's answer (1-based): REL's phrase for
    a triple, "the ..." of a natural "What / Who / Which is the ... of ...?" (not ending in a preposition or
    auxiliary), else None."""
    q = hops[k - 1]["question"]
    t = split_triple(q)
    if t:
        tmpl = REL[t[1]][1] if t[1] in REL else "the %s of {X}" % t[1]
        return None if tmpl is None else (tmpl, t[0])
    m = NATURAL_NP.match(re.sub(r"\s+([?,.!;:])", r"\1", clean(q)))
    if not m or NP_BAD_END.search(m.group(1)):
        return None
    return "{X}", m.group(1)


def describe(hops: list[dict], k: int, depth: int = 0) -> str | None:
    """A noun phrase for hop k's answer built from hop k's own text, its placeholders described recursively; None
    when hop k (or a hop it refers to) has no phrase."""
    t = np_template(hops, k)
    if t is None or depth > 5:
        return None
    parts = {r: describe(hops, r, depth + 1) for r in refs(t[1], k)}
    if any(v is None for v in parts.values()):
        return None
    return t[0].replace("{X}", sub_refs(t[1], k, lambda r: parts[r]))


def closure(hops: list[dict], j: int) -> list[int]:
    """Every hop that hop j refers to, directly or through another hop (sorted)."""
    seen, todo = set(), refs(hops[j - 1]["question"], j)
    while todo:
        k = todo.pop()
        if k not in seen:
            seen.add(k)
            todo += refs(hops[k - 1]["question"], k)
    return sorted(seen)


def hop_text(hops: list[dict], j: int, mode: str) -> tuple[str, str, str]:
    """(asked text, rendering, resolved text) of hop j's node question (without the probe). resolved: the earlier
    hops' answers written in (substitute). describe: each "#k" as hop k's noun phrase (describe); a hop without one
    stays "#k" and is defined up front ('Let #k be the answer to "<hop k's text, described the same way>"'), in hop
    order, so a definition only uses hops defined before it."""
    base = render_hop(hops[j - 1]["question"])
    rs = refs(hops[j - 1]["question"], j)
    resolved = sub_refs(base, j, lambda k: clean(hops[k - 1]["answer"]))
    if not rs:
        return base, "plain", resolved
    if mode == "substitute":
        return resolved, "substitute", resolved
    defs: dict = {}

    def ref(k: int) -> str:
        t = np_template(hops, k)
        if t is not None:                                   # its own references described or defined in turn
            return t[0].replace("{X}", sub_refs(t[1], k, ref))
        if k not in defs:
            defs[k] = None                                  # reserve (hop k's own references are earlier hops)
            defs[k] = sub_refs(render_hop(hops[k - 1]["question"]), k, ref)
        return f"#{k}"
    q = sub_refs(base, j, ref)
    q = q[:1].upper() + q[1:]
    if not defs:
        return q, "inline", resolved
    lead = "Let " + ", and let ".join(f'#{k} be the answer to "{defs[k]}"' for k in sorted(defs))
    return f"{lead}. {q}", "defined", resolved


def levels(hops: list[dict]) -> list[int]:
    out = []
    for j, h in enumerate(hops, 1):
        rs = refs(h["question"], j)
        out.append(1 + max(out[k - 1] for k in rs) if rs else 0)
    return out


# ------------------------------------------------------------------------------------------------------ pools
_LEAD_PLACE = ("city|town|village|hamlet|state|county|province|region|river|lake|island|archipelago|mountain|"
               "mountain range|peak|municipality|commune|district|capital|neighbourhood|neighborhood|suburb|borough|"
               "parish|census-designated place|unincorporated community|settlement|metropolis|territory|prefecture|"
               "canton|ocean|sea|bay|gulf|strait|peninsula|desert|valley|oblast|voivodeship|governorate|township")
_LEAD_COUNTRY = "country|sovereign state|nation"
_LEAD_ORG = ("company|corporation|band|group|team|club|university|college|school|organization|organisation|party|"
             "agency|record label|label|network|channel|station|newspaper|magazine|airline|manufacturer|publisher|"
             "studio|institution|association|federation|league|conglomerate|firm|brand|retailer|bank|foundation|"
             "institute")
_LEAD_WORK = ("film|movie|album|song|single|novel|book|series|sitcom|play|opera|musical|video game|game|poem|painting|"
              "EP|soundtrack|show|episode|comic|manga|anime|documentary|short story|ballet|symphony")
_LEAD_PERSON = ("singer|actor|actress|politician|writer|author|novelist|composer|musician|footballer|player|painter|poet|"
                "director|filmmaker|producer|rapper|journalist|businessman|businesswoman|scientist|philosopher|lawyer|"
                "judge|king|queen|emperor|empress|prince|princess|general|officer|soldier|admiral|bishop|priest|saint|"
                "monk|economist|historian|architect|sculptor|artist|photographer|dancer|comedian|entrepreneur|engineer|"
                "inventor|physician|mathematician|physicist|chemist|astronomer|explorer|navigator|activist|diplomat|"
                "statesman|ruler|sultan|pharaoh|pope|cardinal|archbishop|guitarist|drummer|bassist|pianist|violinist|"
                "songwriter|lyricist|screenwriter|playwright|cricketer|boxer|wrestler|athlete|coach|presenter|"
                "broadcaster|educator|teacher|professor|aristocrat|nobleman|noblewoman|duke|duchess|earl|baron|"
                "president|governor|mayor|senator|minister|chancellor|monarch|consort")
_LEAD = re.compile(r"\b(?:is|was|are|were) (?:an?|the|one of the) (?:[\w'’,-]+ ){0,6}?(%s)\b(?! (?:music|group|"
                   r"song|band|singer|album|club))" % "|".join((_LEAD_PERSON, _LEAD_COUNTRY, "continent", _LEAD_PLACE,
                                                               _LEAD_ORG, _LEAD_WORK)), re.I)
_LEAD_LIFE = re.compile(r"\((?:born |b\. |c\. |né |née )|\(\d{1,2} [A-Z][a-z]+ \d{3,4}|\([A-Z][a-z]+ \d{1,2}, \d{3,4} ?"
                        r"(?:[–-]|--)|\(\d{3,4} ?(?:[–-]|--) ?\d{3,4}\)")


def lead_type(text: str) -> str | None:
    """The kind of entity a Wikipedia lead sentence defines ("X is a city in ...": place; "X (born 1970) is an American
    singer": person; country, continent, org, work), else None."""
    first = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip(), maxsplit=1)[0][:300]
    m = _LEAD.search(first)
    if not m:
        paren = re.search(r"\(([^()]{0,120})\)", first[:200])
        if _LEAD_LIFE.search(first[:160]) or (paren and re.search(
                r"\bborn\b|\d{3,4}\s?(?:[–-]|--)\s?(?:[A-Za-z]+\.? )?(?:\d{1,2},? )?\d{3,4}", paren.group(1))):
            return "person"
        return None
    w = m.group(1).lower()
    for c, pat in (("person", _LEAD_PERSON), ("country", _LEAD_COUNTRY), ("continent", "continent"),
                   ("place", _LEAD_PLACE), ("org", _LEAD_ORG), ("work", _LEAD_WORK)):
        if re.fullmatch(pat, w, re.I):
            return c
    return None


def title_text(title: str) -> str:
    """A paragraph title without its disambiguation ("War (Bone Thugs-n-Harmony song)" -> "War")."""
    return clean(re.sub(r"\s*\([^)]*\)", "", title))


class Pool:
    """Answer strings of a set of items by class, with an index for finding those that occur in a text."""

    def __init__(self, raws: list[dict]):
        tv = collections.defaultdict(collections.Counter)  # paragraph titles typed by their lead sentence
        for r in raws:
            for p in r["paragraphs"]:
                tv[norm(title_text(p["title"]))][lead_type(clean(p["paragraph_text"]))] += 1
        self.title_type = {}
        for k, v in tv.items():
            v.pop(None, None)
            if k and v:
                self.title_type[k] = max(v, key=lambda c: (v[c], c))
        seen = set()
        self.entries: list[tuple[str, str, str, frozenset]] = []      # (text, class, fine, words)
        votes = collections.defaultdict(collections.Counter)            # a string's classes over the pool
        for r in raws:
            for h in r["question_decomposition"]:
                a = clean(h["answer"])
                votes[norm(a)][self.refine(hop_class(h["question"], a)[0], a)] += 1
        for r in raws:
            for h in r["question_decomposition"]:
                a = clean(h["answer"])
                c, f = hop_class(h["question"], a)
                c = self.refine(c, a)
                key = (norm(a), c)
                if not key[0] or key in seen or "\n" in a or len(a) > 80:
                    continue
                if votes[key[0]][c] < max(votes[key[0]].values()):
                    continue                                # "Britain" answers "who ..." once, "which country" more
                if c in ENTITY_CLASSES and not re.match(r"(?:(?:the|The) )?[A-Z0-9]", a):
                    continue                                # "north", "rap": not a name
                seen.add(key)
                self.entries.append((a, c, f, frozenset(WTOK.findall(norm(a)))))
        self.dominant = {k: max(v, key=lambda c: (v[c], c)) for k, v in votes.items()}
        self.by_class = collections.defaultdict(list)
        self.index = collections.defaultdict(lambda: collections.defaultdict(list))
        for i, (a, c, f, ws) in enumerate(self.entries):
            self.by_class[c].append(i)
            key = next((w for w in WTOK.findall(norm(a)) if w not in STOP), None)
            if key:
                self.index[c][key].append(i)

    def class_of(self, x: str) -> str | None:
        """The kind a paragraph's lead sentence gives the string when it is a paragraph title, else the class it answers
        most often in the pool (None: neither)."""
        k = norm(x)
        return self.title_type.get(k) or self.dominant.get(k)

    def refine(self, cls: str, answer: str) -> str:
        """An answer's class with its title kind taking over when it is a paragraph title whose lead sentence says
        what it is ("Chicago Cubs" answers "who won ..." but is an organisation)."""
        t = self.title_type.get(norm(answer))
        return t if t and cls not in ("year", "date", "number") else cls

    def in_text(self, cls: str, text: str, text_low: str, words: set) -> list[int]:
        """Entries of class cls that occur in the text: whole words, case-insensitive; a name (ENTITY_CLASSES) only
        as a whole name (has_name: "David" does not occur in "David Bowie")."""
        out = []
        for w in words & self.index[cls].keys():
            for i in self.index[cls][w]:
                a, _, _, ws = self.entries[i]
                if ws <= words and has_phrase(text_low, a) and (cls not in ENTITY_CLASSES or has_name(text, a)):
                    out.append(i)
        return sorted(set(out))


_SPAN_TOK = re.compile(r"[A-Za-z0-9][\w'’.&-]*|[,;:()\"“”?!]")
NAME_LINK = frozenset("de da van von der del la le du al bin y".split())
OF_HEADS = frozenset("""University Kingdom Republic Bank Duke Duchess Battle Church Museum Institute College School Department
House Bay Gulf Isle Isles Lake Mount Cape City County Province State Army Order Society Academy League Council Ministry
Office Board Prince Princess King Queen Emperor Earl Count Countess Archbishop Bishop Treaty Siege Sea Port Union
Federation Cathedral Basilica Palace Diocese Governor President Secretary Minister Republic Commonwealth Isle""".split())
_ABBR = re.compile(r"(?:[A-Z]\.)+|Mr\.|Mrs\.|Ms\.|Dr\.|St\.|Jr\.|Sr\.|Mt\.|Ft\.|Co\.|Inc\.|Ltd\.")


def name_spans(text: str) -> list[str]:
    """Proper-noun spans of a text: maximal runs of capitalised words within a sentence (inner links "de", "van", ...,
    and "of" after a head such as University / Kingdom / Battle), a capitalised function word in front dropped
    ("In London" -> "London"), at most 5 words; the "Title: " label skipped."""
    out = []
    for line in text.split("\n"):
        if line.startswith("Title: "):
            line = line[len("Title: "):]
        toks = _SPAN_TOK.findall(line)
        i = 0
        while i < len(toks):
            if not toks[i][:1].isupper():
                i += 1
                continue
            j = i
            while j < len(toks):
                t = toks[j]
                if t[:1].isupper():
                    j += 1
                    if t.endswith(".") and not _ABBR.fullmatch(t):
                        break                               # sentence end
                    continue
                link = t in NAME_LINK or (t == "of" and toks[j - 1] in OF_HEADS)
                if link and j > i and j + 1 < len(toks) and toks[j + 1][:1].isupper():
                    j += 1
                    continue
                break
            run = [t if _ABBR.fullmatch(t) else t.rstrip(".") for t in toks[i:j]]
            while run and run[0].lower() in NAME_STOP:
                run = run[1:]
            if 1 <= len(run) <= 5 and all(len(t) > 1 or t.isalpha() for t in run):
                out.append(" ".join(run))
            i = max(j, i + 1)
    return list(dict.fromkeys(out))


NOT_PERSON = frozenset("""County City River University Records Street Church Party Company Inc Corporation Lake Mountain
Mountains Island Islands Bay Ocean Sea Park School College Station Airport Hospital Museum Bridge Road Avenue Kingdom
Republic Empire Army Navy Force Club Team Association Society Council Committee Department Ministry Office Award Awards
Prize Cup Championship Championships League Festival Studios Pictures Entertainment Music Film Show Series Album Song
Province State States Region District Valley Hall Center Centre Institute Academy Library Theatre Theater Cathedral
Palace Castle Court Group Network Television Radio Channel News Times Journal Magazine Press Games Olympics War Battle
Revolution Act Treaty Day Genre Various Country North South East West Northern Southern Eastern Western Central Old New
Saint Mount Fort Port Bank Corps Division Regiment Brigade Records Label Orchestra Band Choir Quartet Trio Brothers
Fair Exposition Tournament Open""".split())
_DEMONYM = re.compile(r"^[A-Z][a-z]+(?:ese|ish)$|^(?:American|English|French|German|British|Chinese|Japanese|Spanish|"
                      r"Italian|Russian|Indian|African|European|Asian|Canadian|Australian|Mexican|Irish|Scottish|"
                      r"Muslim|Christian|Jewish|Catholic|Protestant|Arab|Roman|Greek|Persian|Dutch|Swedish|Polish)$")


def person_like(x: str) -> bool:
    """A name span that may be a person's name: 2-3 words, no place / organisation / work word, no month, no
    demonym."""
    ws = x.split()
    return (2 <= len(ws) <= 3 and not any(w in NOT_PERSON or w in MONTHS or _DEMONYM.match(w) for w in ws)
            and not any(c.isdigit() for c in x))


ORG_WORDS = frozenset("""University College School Institute Academy Company Corporation Inc Ltd Records Party Club Team
Association Society Council Church Bank Airlines Airways Studios Pictures Entertainment Network Television Radio Group
Foundation Agency Department Ministry Committee Commission League Federation Union Army Navy Force Corps Orchestra
Press Publishing Books Magazine Times News Journal FC Motors Industries Railway Railways Company's Brothers""".split())


def span_like(x: str, cls: str, text: str) -> bool:
    """Whether an untyped name span may stand in for the class: a person (person_like), a place (the text uses it after
    in / at / near / from / to / of, or before ", <Name>"; no organisation or person-list word), an organisation (an
    organisation word: University, Records, Party, ...)."""
    ws = x.split()
    if cls == "person":
        return person_like(x)
    if cls == "org":
        return 2 <= len(ws) <= 5 and any(w in ORG_WORDS for w in ws)
    if cls == "place":
        if len(ws) > 4 or any(w in ORG_WORDS for w in ws) or any(c.isdigit() for c in x):
            return False
        return x in locative_names(text)
    return False


_LOC_AFTER = re.compile(r"\b(?:in|at|near|from|to|of|outside|across) (?:the )?([A-Z][\w'’.-]*(?: [A-Z][\w'’.-]*){0,3})")
_LOC_BEFORE = re.compile(r"(?<=[a-z] )([A-Z][\w'’.-]*(?: [A-Z][\w'’.-]*){0,3}), [A-Z][a-z]+")   # not a sentence start


@functools.lru_cache(maxsize=8)
def locative_names(text: str) -> frozenset:
    """Capitalised runs the text uses as places (after in / at / near / from / to / of, or before ", <Name>" when not
    at a sentence start: "Later, Nielsen" is no place), with their prefixes and suffixes ("in Duluth Harbor" also gives
    "Duluth")."""
    out = set()
    for m in list(_LOC_AFTER.finditer(text)) + list(_LOC_BEFORE.finditer(text)):
        ws = m.group(1).rstrip(".").split()
        out |= {" ".join(ws[:k]) for k in range(1, len(ws) + 1)}
        out |= {" ".join(ws[k:]) for k in range(len(ws))}
    return frozenset(out)


NUMWORD = re.compile(r"\b(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:-(?:one|two|three|four|five|six|"
                     r"seven|eight|nine))?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
                     r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|hundred|thousand|dozen)\b", re.I)
_CHUNK_CUT = re.compile(r"[.;:!?()\[\]\"“”]|,\s|\s(?:and|or|but|which|who|whom|whose|that|while|when|where|because|"
                        r"after|before|although|though|since|until|as|with|without|during)\s")


@functools.lru_cache(maxsize=8)
def chunks(text: str) -> tuple:
    """Clause fragments of the paragraphs (cut at punctuation and at conjunctions / relative words), 1-12 words."""
    out = []
    for line in text.split("\n"):
        if line.startswith("Title: "):
            continue
        for c in _CHUNK_CUT.split(line):
            c = c.strip(" ,'’")
            if c and 1 <= len(c.split()) <= 12 and any(len(w) >= 3 and w.lower() not in NAME_STOP
                                                       for w in re.findall(r"[A-Za-z]+", c)):
                out.append(c)
    return tuple(dict.fromkeys(out))


def other_like(target: str, text: str) -> list[str]:
    """In-context candidates for a gold of the mixed class "other", by its shape: name spans and titles for a name;
    number, decade and century expressions (and fragments with digits) for a gold with digits; number words for a
    number word; clause fragments within 2 words of its length for a phrase."""
    art, nw, cap, dig = shape(target)
    n = len(target.split())
    if dig:
        out = context_values("number", text) + context_values("year", text)
        out += re.findall(r"\b\d{1,2}(?:st|nd|rd|th) century\b|\b\d{3}0s\b", text)
        return out + [c for c in chunks(text) if any(ch.isdigit() for ch in c) and abs(len(c.split()) - n) <= 2]
    if NUMWORD.fullmatch(target.strip()):
        return sorted({m.group(0) for m in NUMWORD.finditer(text)})
    if cap == "all":
        titles = [title_text(t) for t in re.findall(r"^Title: (.+)$", text, re.M)]
        return [x for x in dict.fromkeys(titles + name_spans(text))
                if not any(w in MONTHS or w in WEEKDAYS for w in x.split())]
    return [c for c in chunks(text) if abs(len(c.split()) - n) <= 2 and not any(ch.isdigit() for ch in c)]


def context_values(cls: str, text: str) -> list[str]:
    """Year / date / number expressions of a text (for answers of those classes)."""
    if cls == "year":
        return sorted(set(CTX_YEAR.findall(text)))
    if cls == "date":
        return sorted(set(CTX_DATE.findall(text)))
    if cls == "number":
        out = set()
        for m in CTX_NUM.finditer(text):
            x = m.group(0)
            if len(x) == 4 and x.isdigit() and 1000 <= int(x) <= 2029:
                continue
            out.add(x)
            nxt = re.match(r" ([a-z]+|%)", text[m.end():m.end() + 20])
            if nxt and nxt.group(1) not in STOP:
                out.add(f"{x} {nxt.group(1)}" if nxt.group(1) != "%" else x + "%")
        return sorted(out)
    return []


def length_gap(x: str, target: str) -> float:
    """How far x's length is from the target's (characters on a log scale)."""
    return abs(math.log((len(x) + 1) / (len(target) + 1)))


@functools.lru_cache(maxsize=200000)
def shape(x: str) -> tuple:
    """Surface shape of an answer: leading article, content words (1-4+), capitalisation of the content words (all /
    some / none), digits."""
    ws = x.split()
    art = bool(ws) and ws[0].lower() in ("the", "a", "an")
    core = [w for w in (ws[1:] if art else ws) if w.lower() not in STOP]
    caps = [w[:1].isupper() for w in core if w[:1].isalpha()]
    cap = "all" if caps and all(caps) else "none" if not any(caps) else "some"
    return art, min(len(core), 4), cap, any(c.isdigit() for c in x)


def shape_gap(x: str, target: str) -> float:
    a, b = shape(x), shape(target)
    return 0.5 * (a[0] != b[0]) + min(1.5, 0.5 * abs(a[1] - b[1])) + 1.0 * (a[2] != b[2]) + 1.5 * (a[3] != b[3])


TIER_COST = {"hop": 0.0, "context": 0.6, "span": 1.4, "pool": 2.6, "any": 6.0}
SPAN_CLASSES = frozenset({"person", "place", "org"})    # untyped name spans: span_like decides


def pick(cands: list[tuple[str, str, bool]], n: int, reject, rng: random.Random, target: str) -> list[tuple[str, str]]:
    """Up to n (text, tier) from the candidates (text, tier, fine-key match), best score first: the tier's cost
    (TIER_COST: the item's other hops, then the shown paragraphs, then the pool), the surface shape and length gap to
    the target (the gold / the true value, so it is not the odd one out by form or length), -0.3 for the fine key, a
    random jitter; reject(text, chosen) filters."""
    best = {}
    for x, tier, fine in cands:
        sc = TIER_COST[tier] + shape_gap(x, target) + 0.5 * length_gap(x, target) - 0.3 * fine
        if x not in best or sc < best[x][0]:
            best[x] = (sc, tier)
    order = sorted(best, key=lambda x: best[x][0] + rng.random() * 0.3)
    out = []
    for x in order:
        if len(out) >= n:
            break
        if not reject(x, [o for o, _ in out]):
            out.append((x, best[x][1]))
    return out


def candidates(cls: str, fine: str, other_hops: list[str], pool: Pool, text: str, text_low: str, words: set,
               rng_pool: random.Random, target: str, n_pool: int = 300) -> list[tuple[str, str, bool]]:
    """Distractor / wrong-probe candidates of one answer class: (text, tier, fine-key match). For the mixed class
    "other", a name-like target (capitalised words) also takes the paragraphs' name spans and a target with digits
    their number expressions (tier "span")."""
    inside = pool.in_text(cls, text, text_low, words)
    out = [(x, "hop", True) for x in other_hops]
    out += [(pool.entries[i][0], "context", pool.entries[i][2] == fine) for i in inside
            if cls != "other" or pool.entries[i][2] == fine]   # "other" mixes kinds: the same question form only
    out += [(x, "context", False) for x in context_values(cls, text)]
    if cls in ENTITY_CLASSES or cls == "work":              # names of the shown paragraphs, typed by the pool
        titles = [title_text(t) for t in re.findall(r"^Title: (.+)$", text, re.M)]     # answers, else by lead
        for x in list(dict.fromkeys(titles + name_spans(text))):
            if not x or any(w in MONTHS or w in WEEKDAYS or _DEMONYM.match(w) for w in x.split()):
                continue
            c = pool.class_of(x)
            if c == cls:
                out.append((x, "context", False))
            elif c is None and cls in SPAN_CLASSES and span_like(x, cls, text):
                out.append((x, "span", False))
    elif cls == "other":                                     # a mixed class: in-context strings of the gold's shape
        out += [(x, "span", False) for x in other_like(target, text)]
    elif cls == "language":
        out += [(x, "span", False) for x in sorted(set(re.findall(r"\b[A-Z][a-z]+(?:ese|ish|ian|ic|i)\b", text)))]
    ins = set(inside)
    ids = pool.by_class.get(cls, [])
    out += [(pool.entries[i][0], "pool", pool.entries[i][2] == fine)
            for i in rng_pool.sample(ids, min(n_pool, len(ids))) if i not in ins]
    out += [(pool.entries[i][0], "any", False)
            for i in rng_pool.sample(range(len(pool.entries)), min(50, len(pool.entries)))]
    return out


# ------------------------------------------------------------------------------------------------------ items
def paragraph_block(p: dict) -> str:
    return f"Title: {clean(p['title'])}\n{clean(p['paragraph_text'])}"


def bad_paragraph(p: dict) -> bool:
    """Paragraphs that could blur the prompt layout (an empty text, a "Question:" / "Options:" label)."""
    b = paragraph_block(p)
    return not clean(p["paragraph_text"]) or not clean(p["title"]) or "Question:" in b or "Options:" in b


def text_words(text: str) -> set:
    return set(WTOK.findall(norm(text))) | set(WTOK.findall(text.lower()))


def final_options(r: dict, gold: str, aliases: list[str], cls: str, fine: str, pool: Pool, text: str, words: set,
                  rng: random.Random, st: collections.Counter) -> list[tuple[str, str]] | None:
    hops = r["question_decomposition"]
    golds = [gold] + aliases
    other = [clean(h["answer"]) for h in hops[:-1]
             if pool.refine(hop_class(h["question"], h["answer"])[0], h["answer"]) == cls]
    tiers = candidates(cls, fine, other, pool, text, text.lower(), words, random.Random(f"{rng.random()}"), gold)

    q_low = clean(r["question"]).lower()

    def reject(x, chosen):
        return (not x or "\n" in x or len(x) > 80 or any(same_answer(x, g) for g in golds)
                or any(same_answer(x, c) for c in chosen) or has_phrase(q_low, x))
    got = pick(tiers, N_DISTRACTORS, reject, rng, gold)
    if len(got) < N_DISTRACTORS:
        st["dropped_item_no_distractors"] += 1
        return None
    return got


def tf_nodes(r: dict, gold: str, aliases: list[str], pool: Pool, text: str, words: set, shown: dict,
             rng: random.Random, st: collections.Counter, mode: str) -> list[dict]:
    """The yes/no nodes of one item (every hop but the last), given the shown paragraphs (text) and their blocks by
    idx (shown)."""
    hops = r["question_decomposition"]
    n = len(hops)
    lv = levels(hops)
    golds = [gold] + aliases
    final_q = norm(r["question"])
    blocked = set()
    for j in range(1, n):
        if any(same_answer(clean(hops[j - 1]["answer"]), g) for g in golds):
            blocked.add(j)
            st["blocked_gold_valued"] += 1
    for j in range(1, n):
        if j not in blocked and set(closure(hops, j)) & blocked:
            blocked.add(j)
            st["blocked_downstream_of_gold_valued"] += 1
    tl = text.lower()
    final_anc = set(closure(hops, n))
    nodes, built, asked = [], {}, set()
    for j in range(1, n):
        if j in blocked:
            continue
        h = hops[j - 1]
        v = clean(h["answer"])
        q, rendering, resolved = hop_text(hops, j, mode)
        if any(has_phrase(q.lower(), g) for g in golds if g):
            st["dropped_node_gold_in_question"] += 1
            continue
        if norm(q) == final_q or norm(resolved) == final_q:
            st["dropped_node_is_final_question"] += 1
            continue
        if norm(q) in asked:
            st["dropped_node_repeats_another"] += 1
            continue
        sup = shown.get(h["paragraph_support_idx"], "").lower()
        if not has_phrase(sup, v):
            st["dropped_node_value_not_whole_words_in_support"] += 1
            continue
        cls, fine = hop_class(h["question"], v)
        cls = pool.refine(cls, v)
        others = [clean(x["answer"]) for k, x in enumerate(hops, 1) if k != j
                  and pool.refine(hop_class(x["question"], x["answer"])[0], x["answer"]) == cls]

        q_low = q.lower()

        def reject(x, chosen):
            return (not x or "\n" in x or len(x) > 80 or same_answer(x, v) or any(same_answer(x, g) for g in golds)
                    or has_phrase(sup, x) or has_phrase(q_low, x))
        tiers = candidates(cls, fine, others, pool, text, tl, words, random.Random(f"{rng.random()}"), v)
        got = pick(tiers, 1, reject, rng, v)
        if not got:
            st["dropped_node_no_wrong_value"] += 1
            continue
        w, kind = got[0]
        par = [f"s{k}v" for k in refs(h["question"], j) if k in built]
        st["parent_edges"] += len(refs(h["question"], j))
        st["parent_edges_unbuilt"] += len(refs(h["question"], j)) - len(par)
        t = split_triple(h["question"])
        meta = {"step": j, "hop_question": clean(h["question"]), "resolved": resolved, "rendering": rendering,
                "relation": t[1] if t else None, "answer_class": cls, "value": v,
                "support_idx": h["paragraph_support_idx"], "flags": [], "gold_ancestor": j in final_anc,
                "node_depth": lv[j - 1]}
        for suffix, x, truth, k in (("v", v, True, None), ("n", w, False, kind)):
            nd = {"id": f"s{j}{suffix}", "question": node_question(q, x), "opposite": node_opposite(q, x),
                  "truth": truth, "p_cot": 1.0 if truth else 0.0, "p_fresh": None, "type": "step", "depends_on": par,
                  "initial": None, "status": "direct", "probe": x, **meta}
            if k:
                nd["wrong"] = k
                st[f"wrong_{k}"] += 1
            nodes.append(nd)
        st[f"rendering_{rendering}"] += 1
        built[j] = True
        asked.add(norm(q))
    return nodes


RECS: dict = {}          # item id -> item_tokens record (filled by build_item)


class Lengths:
    """Token counts with the repo tokenizer (cached per string)."""

    def __init__(self, path: Path):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(path))
        self.T = templates()
        self.cache: dict = {}
        head, sep, tail = self.T["SUBQ_TEMPLATE"].partition("{question}")
        assert sep and head.endswith(": "), head
        self.node_head = head[:-1]                          # "{problem}\n\nIntermediate question:"
        self.node_tail = " {question}" + tail               # " {question}\nOptions:\nA) {a}\nB) {b}\nAnswer:"
        self.checked = self.mismatch = 0
        self.long_cache: dict = {}

    def n(self, s: str) -> int:
        if len(s) > 400:                                    # long views: cached for the current item only (memory)
            if s not in self.long_cache:
                self.long_cache[s] = len(self.tok.encode(s, add_special_tokens=False).ids)
            return self.long_cache[s]
        if s not in self.cache:
            if len(self.cache) > 200000:
                self.cache.clear()
            self.cache[s] = len(self.tok.encode(s, add_special_tokens=False).ids)
        return self.cache[s]

    def final(self, prompt: str) -> int:
        return self.n(self.T["FINAL_TEMPLATE"].format(problem=prompt))

    def node_exact(self, context: str, q: str) -> int:
        return self.n(self.T["SUBQ_TEMPLATE"].format(problem=context, question=q, a="Yes", b="No"))

    def node(self, context: str, q: str) -> int:
        """Tokens of a node view, as the context part ("...Intermediate question:") plus the question part (" <q>
        ...Answer:"): the pre-tokenizer cuts between the colon and the space-led question, so the sum is exact (checked
        against the whole view on the first items: check())."""
        return (self.n(self.node_head.replace("{problem}", context))
                + self.n(self.node_tail.format(question=q, a="Yes", b="No")))

    def check(self, context: str, q: str) -> None:
        self.checked += 1
        self.mismatch += self.node(context, q) != self.node_exact(context, q)


def templates() -> dict:
    """SUBQ_TEMPLATE / FINAL_TEMPLATE of cotdistill.student, read from its source (importing it needs torch)."""
    tree = ast.parse((ROOT / "src/cotdistill/student.py").read_text())
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and getattr(n.targets[0], "id", "") in (
                "SUBQ_TEMPLATE", "FINAL_TEMPLATE"):
            out[n.targets[0].id] = ast.literal_eval(n.value)
    return out


def assemble(r: dict, chosen: list[dict], keys: dict, gold_pos: str, pool: Pool, seed: int, mode: str,
             st: collections.Counter) -> dict | None:
    """The item for one choice of shown paragraphs (chosen, displayed in the order of their random keys)."""
    iid = r["id"]
    rng = random.Random(f"{SOURCE}|item|{seed}|{iid}")
    order = sorted(chosen, key=lambda p: keys[p["idx"]])
    blocks = [paragraph_block(p) for p in order]
    text = "\n\n".join(blocks)
    shown = {p["idx"]: paragraph_block(p) for p in order}
    hops = r["question_decomposition"]
    gold = clean(r["answer"])
    aliases = [clean(a) for a in r.get("answer_aliases") or [] if clean(a)]
    cls, fine = hop_class(hops[-1]["question"], gold)
    cls = pool.refine(cls, gold)
    words = text_words(text)
    opts = final_options(r, gold, aliases, cls, fine, pool, text, words, rng, st)
    if opts is None:
        return None
    gi = LABELS.index(gold_pos)
    rest = opts[:]
    rng.shuffle(rest)
    options, kinds = [], []
    for i in range(len(LABELS)):
        if i == gi:
            options.append(gold)
            kinds.append("gold")
        else:
            x, k = rest.pop()
            options.append(x)
            kinds.append(k)
    q = clean(r["question"])
    prompt = (text + f"\n\nQuestion: {q}\nOptions:\n" + "\n".join(f"{L}) {o}" for L, o in zip(LABELS, options)))
    subqs = tf_nodes(r, gold, aliases, pool, text, words, shown, rng, st, mode)
    n = len(hops)
    sup = sorted({h["paragraph_support_idx"] for h in hops})
    return {"item_id": f"musique-{iid}", "source": SOURCE, "group": f"{SOURCE}/s{n}", "type": "choice",
            "prompt": prompt, "labels": list(LABELS), "gold_label": gold_pos,
            "teacher": {L: float(L == gold_pos) for L in LABELS}, "options": options, "option_kinds": kinds,
            "recipe": "musique_mc5", "depth": n, "dep_level": levels(hops)[-1], "subqs": subqs, "random_subqs": [],
            "rationale": "", "augment": None, "has_tree": bool(subqs), "tree_fix": True, "row_kind": "final",
            "problem_id": iid, "node_status": "ok", "licence_class": "PERM", "licence": LICENCE,
            "label_basis": "gold", "node_basis": "decomposition", "n_steps": n, "tree_source": TREE_SOURCE,
            "musique_type": iid.split("__")[0], "answer_class": cls,
            "musique": {"id": iid, "answer": gold, "answer_aliases": aliases,
                        "paragraphs_shown": [p["idx"] for p in order], "supporting": sup,
                        "n_paragraphs": len(r["paragraphs"])}}


def max_view(it: dict, L: Lengths) -> int:
    """Tokens of the longest view of the item: the final view, or the GFP view (whole prompt) of its longest node."""
    m = L.final(it["prompt"])
    if it["subqs"]:
        q = max((n["question"] for n in it["subqs"]), key=L.n)
        m = max(m, L.node(it["prompt"], q))
    return m


def build_item(r: dict, gold_pos: str, pool: Pool, L: Lengths, args, st: collections.Counter) -> dict | None:
    """Supporting paragraphs + as many distractor paragraphs (seeded order) as the token budget allows."""
    iid = r["id"]
    paras = r["paragraphs"]
    L.long_cache.clear()
    if any(bad_paragraph(p) for p in paras if p["is_supporting"]):
        st["dropped_item_bad_supporting_paragraph"] += 1
        return None
    by = {p["idx"]: p for p in paras}
    sup_idx = {h["paragraph_support_idx"] for h in r["question_decomposition"]}
    if not sup_idx <= set(by) or any(not by[i]["is_supporting"] for i in sup_idx):
        st["dropped_item_support_mismatch"] += 1
        return None
    prng = random.Random(f"{SOURCE}|paragraphs|{args.seed}|{iid}")
    keys = {p["idx"]: prng.random() for p in paras}
    sup = [by[i] for i in sorted(sup_idx)] + [p for p in paras if p["is_supporting"] and p["idx"] not in sup_idx]
    dis = [p for p in paras if not p["is_supporting"]]
    prng.shuffle(dis)
    n_bad = sum(bad_paragraph(p) for p in dis)
    dis = [p for p in dis if not bad_paragraph(p)]
    st["distractor_paragraphs_unusable"] += n_bad
    budget = args.max_len - args.margin
    if "Options:" in r["question"] or "Question:" in r["question"]:
        st["dropped_item_question_has_label"] += 1
        return None
    last = r["question_decomposition"][-1]
    if not has_phrase(paragraph_block(by[last["paragraph_support_idx"]]).lower(), clean(r["answer"])):
        st["dropped_item_gold_not_whole_words_in_support"] += 1       # e.g. "bio" inside "biological"
        return None
    # first attempt: supporting only, to measure the question / options / node overhead
    st0 = collections.Counter()
    it = assemble(r, sup, keys, gold_pos, pool, args.seed, args.placeholders, st0)
    if it is None:
        st.update(st0)
        return None
    base = max_view(it, L)
    if base > budget:
        st["dropped_item_supporting_over_budget"] += 1
        return None
    room = budget - base
    m = 0
    for p in dis:                                           # estimate: paragraph tokens + the blank line
        c = L.n(paragraph_block(p)) + 1
        if c > room:
            break
        room -= c
        m += 1
    while True:
        st1 = collections.Counter()
        it = assemble(r, sup + dis[:m], keys, gold_pos, pool, args.seed, args.placeholders, st1)
        if it is not None:
            mv = max_view(it, L)
            if mv <= budget:
                break
        if m == 0:
            st.update(st1)
            st["dropped_item_over_budget"] += 1
            return None
        m -= 1
        st["budget_retries"] += 1
    st.update(st1)
    it["musique"]["distractor_paragraphs_shown"] = m
    it["musique"]["distractor_paragraphs_left_out"] = len(dis) - m
    it["musique"]["max_view_tokens"] = mv
    RECS[it["item_id"]] = item_tokens(it, L, check=L.checked < 900)      # while this item's views are cached
    return it


# ------------------------------------------------------------------------------------------------------ audit
def gold_text(item: dict) -> str:
    return item["options"][item["labels"].index(item["gold_label"])]


def audit_item(item: dict) -> list[str]:
    """Problems of one built item, from its own fields only (empty: clean)."""
    bad, iid = [], item.get("item_id")
    try:
        body, ask = musique_tf_cut(item)
        if statement_block(item) != body or problem_no_options(item) != body + "\n\n" + ask:
            bad.append("treeplan cut")
    except ValueError as e:
        bad.append(f"shape: {e}")
        return [f"{iid}: {b}" for b in bad]
    m = item["musique"]
    opts = item["options"]
    if (item.get("source") != SOURCE or item["labels"] != LABELS or len(opts) != 5 or item.get("tree_fix") is not True
            or item["teacher"] != {L: float(L == item["gold_label"]) for L in LABELS}
            or gold_text(item) != m["answer"] or item["n_steps"] != item["depth"]
            or item["group"] != f"{SOURCE}/s{item['n_steps']}" or item["has_tree"] != bool(item["subqs"])
            or item["prompt"].rpartition("\nOptions:\n")[2] != "\n".join(f"{L}) {o}" for L, o in zip(LABELS, opts))
            or item["option_kinds"][item["labels"].index(item["gold_label"])] != "gold"):
        bad.append("item fields")
    golds = [m["answer"]] + m["answer_aliases"]
    for i, o in enumerate(opts):
        if not o or "\n" in o:
            bad.append(f"option {i} empty or multi-line")
        if item["labels"][i] != item["gold_label"] and any(same_answer(o, g) for g in golds):
            bad.append(f"option {o!r} is the gold or an alias")
        if any(same_answer(o, x) for x in opts[:i]):
            bad.append(f"option {o!r} repeats another")
    if not has_phrase(body.lower(), m["answer"]):
        bad.append("gold not in the paragraphs")
    by = {n["id"]: n for n in item["subqs"]}
    if len(by) != len(item["subqs"]):
        bad.append("duplicate node ids")
    for n in item["subqs"]:
        i = n["id"]
        if (n["truth"] is True) != (n["probe"] == n["value"]) or n["p_cot"] != float(n["truth"]):
            bad.append(f"{i}: truth is not (probe == value)")
        if not n["truth"] and same_answer(n["probe"], n["value"]):
            bad.append(f"{i}: No probe overlaps the value")
        if n["id"].endswith("v") != n["truth"] or not n["question"].endswith(f" Is it {n['probe']}?"):
            bad.append(f"{i}: id / wording")
        if n["step"] >= item["n_steps"]:
            bad.append(f"{i}: last hop asked")
        if any(same_answer(n["value"], g) or same_answer(n["probe"], g) for g in golds):
            bad.append(f"{i}: value or probe is the gold")
        stem = n["question"][:-len(f" Is it {n['probe']}?")]
        defined = set(re.findall(r"[Ll]et #(\d+) be the answer", stem))
        if any(k not in defined for k in re.findall(r"#(\d+)", stem) if 0 < int(k) < n["step"]):
            bad.append(f"{i}: unresolved placeholder")
        sup = [b for b in body.split("\n\n") if b.startswith("Title: ")]
        if not any(has_phrase(b.lower(), n["value"]) for b in sup):
            bad.append(f"{i}: value not in the paragraphs")
        twin = by.get(i[:-1] + ("n" if n["truth"] else "v"))
        if not twin or twin["value"] != n["value"] or twin["truth"] == n["truth"]:
            bad.append(f"{i}: no Yes/No twin")
        depth = 0
        for d in n["depends_on"]:
            par = by.get(d)
            if not par or not par["truth"]:
                bad.append(f"{i}: parent {d} missing or not Yes")
                continue
            depth = max(depth, par["node_depth"] + 1)
        if n["node_depth"] < depth:
            bad.append(f"{i}: node_depth {n['node_depth']} below its parents' + 1")
    return [f"{iid}: {b}" for b in bad]


def audit_support(item: dict, raw: dict) -> list[str]:
    """Checks that need the raw item: each value in its own supporting paragraph, the No probe not."""
    bad = []
    by = {p["idx"]: paragraph_block(p).lower() for p in raw["paragraphs"]}
    for n in item["subqs"]:
        s = by[n["support_idx"]]
        if not has_phrase(s, n["value"]):
            bad.append(f"{item['item_id']} {n['id']}: value not in its supporting paragraph")
        if not n["truth"] and has_phrase(s, n["probe"]):
            bad.append(f"{item['item_id']} {n['id']}: No probe in the supporting paragraph")
    return bad


# ------------------------------------------------------------------------------------------------------ io / stats
def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def write(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return md5(path)


def read_raw(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summ(xs: list[int], max_len: int) -> dict:
    xs = sorted(xs)
    if not xs:
        return {}
    return {"n": len(xs), "mean": round(sum(xs) / len(xs), 1), "p50": xs[len(xs) // 2],
            "p99": xs[min(len(xs) - 1, int(0.99 * len(xs)))], "max": xs[-1],
            "share_le_max_len": round(sum(x <= max_len for x in xs) / len(xs), 4)}


def stump(rows: list[tuple]) -> float | None:
    """Accuracy of the best single-feature rule (feature -> label or its negation)."""
    xs = [r for r in rows if r[1] is not None]
    if not xs:
        return None
    acc = sum(t == f for t, f in xs) / len(xs)
    return round(max(acc, 1 - acc), 3)


def stats(items: list[dict]) -> dict:
    nodes = [n for it in items for n in it["subqs"]]
    pairs = [n for n in nodes if n["truth"]]
    return {"items": len(items), "with_tree": sum(it["has_tree"] for it in items),
            "n_steps": dict(sorted(collections.Counter(it["n_steps"] for it in items).items())),
            "musique_type": dict(sorted(collections.Counter(it["musique_type"] for it in items).items())),
            "nodes": len(nodes), "nodes_per_item": round(len(nodes) / max(1, len(items)), 2),
            "node_pairs_per_item_by_hops": {s: round(sum(len(it["subqs"]) for it in items if it["n_steps"] == s) / 2
                                                     / max(1, sum(it["n_steps"] == s for it in items)), 2)
                                            for s in sorted({it["n_steps"] for it in items})},
            "yes_share": round(sum(n["truth"] for n in nodes) / max(1, len(nodes)), 4),
            "node_depth": dict(sorted(collections.Counter(n["node_depth"] for n in nodes).items())),
            "rendering": dict(sorted(collections.Counter(n["rendering"] for n in pairs).items())),
            "wrong": dict(sorted(collections.Counter(n["wrong"] for n in nodes if not n["truth"]).items())),
            "answer_class_final": dict(sorted(collections.Counter(it["answer_class"] for it in items).items())),
            "option_kinds": dict(sorted(collections.Counter(k for it in items for k in it["option_kinds"]
                                                            if k != "gold").items())),
            "gold_label": dict(sorted(collections.Counter(it["gold_label"] for it in items).items())),
            "paragraphs_shown": dict(sorted(collections.Counter(len(it["musique"]["paragraphs_shown"])
                                                                for it in items).items())),
            "items_with_all_paragraphs": sum(it["musique"]["distractor_paragraphs_left_out"] == 0 for it in items),
            "gold_ancestor_share": round(sum(n["gold_ancestor"] for n in nodes) / max(1, len(nodes)), 3)}


def cue_rows(items: list[dict]) -> dict:
    """Single-feature stumps: is an option / probe in the shown paragraphs (whole words); its length."""
    opt, opt_len, prb, prb_q = [], [], [], []
    elim, strict = [], collections.Counter()
    for it in items:
        body = musique_tf_cut(it)[0].lower()
        q = musique_tf_cut(it)[1].lower()
        lens = [len(o) for o in it["options"]]
        g = it["labels"].index(it["gold_label"])
        rest = lens[:g] + lens[g + 1:]
        strict["longest"] += lens[g] > max(rest)
        strict["shortest"] += lens[g] < min(rest)
        elim.append(1 / (1 + sum(has_phrase(body, o) for i, o in enumerate(it["options"]) if i != g)))
        for L, o in zip(it["labels"], it["options"]):
            g = L == it["gold_label"]
            opt.append((g, has_phrase(body, o)))
            opt_len.append((g, len(o) == max(lens)))
        for n in it["subqs"]:
            prb.append((n["truth"], has_phrase(body, n["probe"])))
            prb_q.append((n["truth"], has_phrase(q, n["probe"])))
    share = lambda rows, t: round(sum(f for g, f in rows if g == t) / max(1, sum(g == t for g, _ in rows)), 3)  # noqa
    return {"options": {"n": len(opt), "gold_in_paragraphs": share(opt, True),
                        "distractor_in_paragraphs": share(opt, False),
                        "gold_is_longest_option": share(opt_len, True),
                        "distractor_is_longest_option": share(opt_len, False),
                        "gold_strictly_longest": round(strict["longest"] / max(1, len(items)), 3),
                        "gold_strictly_shortest": round(strict["shortest"] / max(1, len(items)), 3),
                        "elimination_baseline_acc": round(sum(elim) / max(1, len(elim)), 3),
                        "note": "chance 0.2; elimination = a uniform pick among the options found in the paragraphs"},
            "nodes": {"n": len(prb), "stump_probe_in_paragraphs": stump(prb),
                      "yes_probe_in_paragraphs": share(prb, True), "no_probe_in_paragraphs": share(prb, False),
                      "stump_probe_in_final_question": stump(prb_q)}}


def item_tokens(it: dict, L: Lengths, check: bool = False) -> dict:
    """Tokens of the final view and of every node view (GFPS: statements, GFPQ: + question, GFP: whole prompt)."""
    sb, pq = statement_block(it), problem_no_options(it)
    out = {"final": L.final(it["prompt"]), "gfps": [], "gfpq": [], "gfp": []}
    for n in it["subqs"]:
        for key, ctx in (("gfps", sb), ("gfpq", pq), ("gfp", it["prompt"])):
            out[key].append(L.node(ctx, n["question"]))
            if check:
                L.check(ctx, n["question"])
    return out


def token_report(recs: list[dict], max_len: int) -> dict:
    """Token lengths per view (item_tokens records) and epoch tokens per arm (A2: finals; GFPS / GFPQ: finals + every
    node view in that context)."""
    fin = [r["final"] for r in recs]
    gfps = [x for r in recs for x in r["gfps"]]
    gfpq = [x for r in recs for x in r["gfpq"]]
    gfp = [x for r in recs for x in r["gfp"]]
    tot = {"A2": sum(fin), "GFPS": sum(fin) + sum(gfps), "GFPQ": sum(fin) + sum(gfpq)}
    return {"final_view": summ(fin, max_len), "gfps_node_view": summ(gfps, max_len),
            "gfpq_node_view": summ(gfpq, max_len), "gfp_node_view": summ(gfp, max_len),
            "epoch_tokens_A2": tot["A2"], "epoch_tokens_GFPS": tot["GFPS"], "epoch_tokens_GFPQ": tot["GFPQ"],
            "GFPS_over_A2": round(tot["GFPS"] / max(1, tot["A2"]), 2),
            "GFPQ_over_A2": round(tot["GFPQ"] / max(1, tot["A2"]), 2)}


def subset(items: list[dict], frac: float, seed: str) -> list[int]:
    """Indices (file order) of the stratified subset: data/student_gsm8k_tf_f{10,25,50}'s recipe (as
    build_strategyqa_tf.subset)."""
    groups = collections.defaultdict(list)
    for i, it in enumerate(items):
        groups[it["group"]].append(i)
    rng = random.Random(seed)
    keep = set()
    for g in sorted(groups):
        idx = groups[g][:]
        rng.shuffle(idx)
        keep |= set(idx[:max(1, round(len(idx) * frac))])
    return sorted(keep)


# ------------------------------------------------------------------------------------------------------ splits
def prepare(raw: list[dict], st: collections.Counter, name: str) -> list[dict]:
    """2-4 hop answerable items, whitespace-normalised; one item per normalised question text."""
    out, by = [], collections.defaultdict(list)
    for r in raw:
        n = len(r.get("question_decomposition") or [])
        if not r.get("answerable", True) or not 2 <= n <= 4:
            st[f"dropped_{name}_not_2to4_answerable"] += 1
            continue
        r["question"] = clean(r["question"])
        by[norm(r["question"])].append(r)
        out.append(r)
    keep = []
    for r in out:
        rs = by[norm(r["question"])]
        if len({norm(x["answer"]) for x in rs}) > 1:
            st[f"dedup_{name}_dropped_conflicting_answer"] += 1
            continue
        if rs[0] is not r:
            st[f"dedup_{name}_dropped_same_question"] += 1
            continue
        keep.append(r)
    return keep


def pick_val(raw: list[dict], n_val: int, seed: int) -> set:
    """n_val ids stratified by musique_type: each type (sorted) shuffled by one generator, round(n x frac) taken."""
    groups = collections.defaultdict(list)
    for r in raw:
        groups[r["id"].split("__")[0]].append(r["id"])
    frac = n_val / max(1, len(raw))
    rng = random.Random(f"{SOURCE}|val|{seed}")
    out = set()
    for g in sorted(groups):
        ids = sorted(groups[g])
        rng.shuffle(ids)
        out |= set(ids[:round(len(ids) * frac)])
    return out


def gold_positions(ids: list[str], seed: int, split: str) -> dict:
    ids = sorted(ids)
    random.Random(f"{SOURCE}|gold-pos|{seed}|{split}").shuffle(ids)
    return {i: LABELS[k % len(LABELS)] for k, i in enumerate(ids)}


# ------------------------------------------------------------------------------------------------------ build
def build(args) -> None:
    from cotdistill.teacher import leaks_answer
    raw_dir = ROOT / args.raw_dir
    st = collections.Counter()
    raws = {k: read_raw(raw_dir / v) for k, v in RAW_FILES.items()}
    if args.limit:                                          # development runs: a seeded sample, file order kept
        raws = {k: [v[i] for i in sorted(random.Random(f"limit|{k}").sample(range(len(v)), min(args.limit, len(v))))]
                for k, v in raws.items()}
    st["raw_train"], st["raw_dev"] = len(raws["train"]), len(raws["test"])
    tr_all = prepare(raws["train"], st, "train")
    test = prepare(raws["test"], st, "test")
    val_ids = pick_val(tr_all, args.n_val, args.seed)
    splits = {"train": [r for r in tr_all if r["id"] not in val_ids],
              "val": [r for r in tr_all if r["id"] in val_ids], "test": test}
    pools = {"train": Pool(splits["train"]), "val": Pool(splits["train"] + splits["val"]),
             "test": Pool(splits["train"] + splits["test"])}
    L = Lengths(ROOT / TOKENIZER)
    items, raw_by = {}, {}
    for name, rows in splits.items():
        pos = gold_positions([r["id"] for r in rows], args.seed, name)
        sst = collections.Counter()
        its = []
        for k, r in enumerate(rows):
            it = build_item(r, pos[r["id"]], pools[name], L, args, sst)
            if it is not None:
                its.append(it)
                raw_by[it["item_id"]] = r
            if (k + 1) % 2000 == 0:
                print(f"{name}: {k + 1}/{len(rows)} items, kept {len(its)}", flush=True)
        items[name] = its
        st.update({f"{name}_{k}": v for k, v in sst.items()})
        print(name, len(its), dict(sst), flush=True)
    ids = {k: {it["item_id"] for it in v} for k, v in items.items()}
    qs = {k: {norm(musique_tf_cut(it)[1]) for it in v} for k, v in items.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not ids[a] & ids[b], (a, b)
        assert not qs[a] & qs[b], (a, b)
    # ---- hop overlap (MuSiQue reuses single-hop questions across composed questions)
    hop_ids = {k: [{h["id"] for h in raw_by[it["item_id"]]["question_decomposition"]} for it in v]
               for k, v in items.items()}
    tr_hops = set().union(*hop_ids["train"])
    va_hops = set().union(*hop_ids["val"]) if hop_ids["val"] else set()
    overlap = {"val_items_sharing_a_hop_with_train": sum(bool(h & tr_hops) for h in hop_ids["val"]),
               "train_items_sharing_a_hop_with_val": sum(bool(h & va_hops) for h in hop_ids["train"]),
               "test_items_sharing_a_hop_with_train": sum(bool(h & tr_hops) for h in hop_ids["test"])}
    # ---- audit and leaks
    fails = []
    for v in items.values():
        for it in v:
            fails += audit_item(it)
            fails += audit_support(it, raw_by[it["item_id"]])
    leak = collections.Counter()
    for v in items.values():
        for it in v:
            gold = gold_text(it)
            for n in it["subqs"]:
                leak["nodes"] += 1
                leak["teacher_leaks_answer"] += leaks_answer(n["question"], it)
                leak["probe_or_value_is_gold"] += overlaps(n["probe"], gold) or overlaps(n["value"], gold)
                leak["gold_in_node_text"] += has_phrase(n["question"].lower(), gold)
                leak["node_states_another_hop_answer"] += any(
                    has_phrase(n["question"].rpartition(" Is it ")[0].lower(), m["value"])
                    for m in it["subqs"] if m["truth"] and m["step"] != n["step"]
                    and m["step"] in [int(d[1:-1]) for d in n["depends_on"]])
    # ---- write
    out, edir = ROOT / args.out, ROOT / args.eval_dir
    summary = {"script": "scripts/build_musique_tf.py", "args": vars(args),
               "source": {"dataset": "MuSiQue-Ans v1.0 (Trivedi et al., TACL 2022; StonyBrookNLP/musique)",
                          "hf_repo": HF_REPO, "revision": HF_REV,
                          "url": f"https://huggingface.co/datasets/{HF_REPO}/tree/{HF_REV}",
                          "licence": "CC BY 4.0", "raw": {v: md5(raw_dir / v) for v in RAW_FILES.values()},
                          "test_is": "the official answerable dev split (test labels are hidden)"},
               "md5": {}, "files": {}, "tokens": {}}
    for path, rows_ in ((out / "train.jsonl", items["train"]), (out / "val.jsonl", items["val"]),
                        (edir / "musique_tf_test.jsonl", items["test"])):
        key = rel(path)
        summary["md5"][key] = write(path, rows_)
        summary["files"][key] = stats(rows_)
        summary["tokens"][key] = token_report([RECS[it["item_id"]] for it in rows_], args.max_len)
        print(key, json.dumps(summary["tokens"][key]), flush=True)
    summary["tokens"]["tokenizer"] = TOKENIZER
    summary["tokens"]["additive_count_check"] = {"node_views_checked": L.checked, "mismatches": L.mismatch}
    if L.mismatch:
        fails.append(f"token counts: {L.mismatch} of {L.checked} node views differ from the whole-view count")
    summary["tokens"]["budget"] = {"max_len": args.max_len, "margin": args.margin,
                                   "rule": "max(final view, GFP view of the longest node) <= max_len - margin"}
    summary["build"] = dict(sorted(st.items()))
    summary["hop_overlap"] = overlap
    summary["leaks"] = dict(leak)
    summary["cues_train"] = cue_rows(items["train"])
    summary["cues_test"] = cue_rows(items["test"])
    summary["audit"] = {"failures": len(fails), "examples": fails[:20]}
    summary["val_ids"] = sorted(it["problem_id"] for it in items["val"])
    summary["fields"] = {"item": ["source musique_tf", "group musique_tf/s<n_hops>", "n_steps = depth = hops",
                                  "dep_level (level of the last hop)", "subqs", "has_tree", "tree_source",
                                  "option_kinds (gold / hop / context / pool / any)", "musique_type", "answer_class",
                                  "musique {id, answer, answer_aliases, paragraphs_shown, supporting, n_paragraphs, "
                                  "distractor_paragraphs_shown, distractor_paragraphs_left_out, max_view_tokens}"],
                         "subq": ["id s<j>v / s<j>n", "question", "opposite", "truth", "p_cot", "p_fresh", "type step",
                                  "depends_on", "initial", "status", "probe (str)", "wrong (No: hop / context / pool "
                                  "/ any)", "step", "hop_question", "resolved", "rendering (plain / inline / defined "
                                  "/ substitute)", "relation", "answer_class", "value (str)", "support_idx", "flags",
                                  "gold_ancestor", "node_depth"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    # ---- f25: a stratified subset of train (data/student_gsm8k_tf_f25's recipe), val copied
    tr = read_jsonl(out / "train.jsonl")
    lines = [ln for ln in (out / "train.jsonl").read_text().split("\n") if ln]
    sub_dir = ROOT / f"{args.out}_f{round(args.sub_frac * 100)}"
    tag = f"{SOURCE}|f{round(args.sub_frac * 100)}|{args.sub_date}"
    keep = subset(tr, args.sub_frac, tag)
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "train.jsonl").write_text("".join(lines[i] + "\n" for i in keep))
    (sub_dir / "val.jsonl").write_bytes((out / "val.jsonl").read_bytes())
    sub_items = [tr[i] for i in keep]
    sub = {"source": args.out, "fraction": args.sub_frac, "stratified_by": "group (n_steps = hops)", "seed": tag,
           "n_train": len(keep), "train_md5": md5(sub_dir / "train.jsonl"), "val_md5": md5(sub_dir / "val.jsonl"),
           "train_stats": stats(sub_items),
           "train_tokens": token_report([RECS[it["item_id"]] for it in sub_items], args.max_len)}
    (sub_dir / "summary.json").write_text(json.dumps(sub, indent=1))
    summary["md5"][rel(sub_dir / "train.jsonl")] = sub["train_md5"]
    summary["md5"][rel(sub_dir / "val.jsonl")] = sub["val_md5"]
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("files", "build", "hop_overlap", "leaks", "cues_train", "cues_test",
                                              "audit", "md5")}, indent=1))
    if fails:
        print("AUDIT FAILED", fails[:10])
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--out", default="data/student_musique_tf")
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--margin", type=int, default=32, help="tokens kept free under --max-len (special prefix, "
                                                           "tokenizer differences)")
    ap.add_argument("--placeholders", choices=("describe", "substitute"), default="describe")
    ap.add_argument("--sub-frac", type=float, default=0.25)
    ap.add_argument("--sub-date", default="20261016", help="date in the subset's seed string")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first N raw items per file (development runs; 0 = all)")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
