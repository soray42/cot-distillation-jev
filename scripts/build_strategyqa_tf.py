"""Tree Forcing data for an implicit multi-hop yes/no domain: StrategyQA (Geva et al. 2021, the original AI2 release in
data/raw/strategyqa: strategyqa_train.json, 2290 questions with a human decomposition, facts and evidence per question).
The trees are read off the dataset's own annotations (decomposition steps answered from the annotated facts); no
model writes or labels anything. Downloaded files are read as data only (json.load).

Finals: "Question: <the question, verbatim>\nOptions:\nA) Yes\nB) No", labels A / B = Yes / No (fixed order, as the
yes/no BBH tasks), teacher one-hot on the gold, options ["Yes", "No"]. The problem context is the question only: no
fact, term description or paragraph is in any prompt. source "strategyqa_tf", group "strategyqa_tf/s<n>" (n = number
of decomposition steps = n_steps = depth).

Nodes (each a yes/no question with an exact truth by the annotated facts; p_cot = truth): per decomposition step j
that is not the last one, "s<j>v" "<step> Is it <answer>?" (Yes) and "s<j>n" the same with a wrong answer (No), so
every item and step is exactly 50% Yes. "opposite" holds "Is it different from <x>?". The step is the decomposition's
own text with every "#k" replaced by step k's answer ("?" appended when it has no end mark). Answers are taken from the
one fact that the step aligns with (alignment: every capitalised word of the step and at least ALIGN_MIN of its content
words occur in the fact, and no other fact ties it) and only in two forms:
  - number (a "how many / much / old / long / tall ..." or quantity step): the fact holds exactly one number
    expression that the step does not state (digits, number words, a scale word: "1.6 million"); its unit word, when
    one follows, is kept ("195 pounds", "15%", "-280 °F"). The No probe moves the number only: a nearby value of the
    same integer-ness and roundness at least MIN_RATIO away and at most 10x (factors or small offsets), or the same
    kind of value of another item of the split (same unit), above or below the value 50/50;
  - year (a "when / what year" step): exactly one year (1000-2020; a day and month in front are allowed) and no other
    number; the No probe is a year 5-50 years away with the same roundness (never after 2020), or another item's year
    within 50 years;
  - entity (a "who" / "where" step, or "which / what <type noun>" with a singular type noun): the fact holds exactly
    one capitalised name that the step does not contain (not a possessive, month, weekday, demonym or a single
    sentence-initial word; "where": after in / from / at / near / on / to); the No probe is the answer of the same step
    class (person / place / the type noun) of another item of the split that occurs nowhere in this item's question,
    decomposition or facts and shares no word with the answer.
A step "#k" is resolved only when step k got nodes; depends_on = the Yes nodes of the referenced steps, node_depth = 1 +
the deepest of them (0 without references); only Yes nodes are parents.
Not built (counted in summary.json, build.dropped_*): the last step (the final operation: the final question in
disguise; dropped always) and any step whose normalised text is the final question; yes/no steps ("Is / Does / Can
...": their answer is a judgement, not stated by any fact); operation steps (every annotator marked "operation"); steps
with an unresolved "#k"; steps no fact aligns with (or two facts tie); steps whose fact holds no / several candidate
values; hedged values ("over", "about", "up to", ... in front, "or more" after: a No probe could be true); plural or
other non-entity questions; steps with no wrong value; a step whose answer repeats an earlier step's answer.
Every node is re-verified from its own fields (audit_item): the answer occurs in its fact, truth = (probe == answer),
the wording, the Yes/No twin, parents are Yes nodes whose answer the node's question contains, node_depth, no "#",
no node restates the final question (teacher.leaks_answer) and no fact is in the prompt.

Splits (question level, deduplicated by normalised question text; the official test set has no labels, so the test
set is a seeded share of the official train set, stratified by group):
  data/eval/strategyqa_tf_test.jsonl       --test-frac (20%) of the questions
  data/student_strategyqa_tf/val.jsonl     --val-frac (10%)
  data/student_strategyqa_tf/train.jsonl   the rest
  data/student_strategyqa_tf_f25/          a --sub-frac (25%) subset of train stratified by group (the recipe of
                                           data/student_gsm8k_tf_f{10,25,50}: one random.Random("<tag>|f25|<date>"),
                                           each group in sorted order shuffled in file order, max(1, round(n x f))
                                           items kept, file order), val copied.
Near duplicates across splits (content-word Jaccard >= NEAR_DUP of the questions, same normalised term) are dropped
from train / val and counted. Wrong answers are drawn from the item's own split only.

Reports (summary.json): drop counts by reason, node stats, Yes share, leak checks, single-feature cue stumps for the
node targets in the GFPQ view (the question only), splits, md5s.

  python3 scripts/build_strategyqa_tf.py            # stdlib only
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SOURCE = "strategyqa_tf"
TREE_SOURCE = "strategyqa_decomposition_tf"
RAW = "data/raw/strategyqa/strategyqa_train.json"
LICENCE = "StrategyQA (allenai/strategyqa release, strategyqa_dataset.zip); licence not checked (owner: licence aside)"
LABELS = ["A", "B"]
OPTIONS = ["Yes", "No"]
ALIGN_MIN = 0.7           # share of a step's content words that its fact must contain
MIN_RATIO = 1.2           # a wrong number is at least this factor away from the answer
MAX_RATIO = 10.0
YEAR_MIN, YEAR_MAX = 1000, 2020   # StrategyQA was written in 2020: no year after it is a fair probe
NEAR_DUP = 0.8
WRONG_W = {"other": 0.5, "perturb": 0.5}

WTOK = re.compile(r"[a-z0-9]+")
WORD = re.compile(r"[A-Za-z0-9#][A-Za-z0-9'’&.-]*")
STOP = frozenset("""a an the of in on at to for from by with and or is are was were be been being do does did what which
who whom whose where when why how many much that this these those it its their his her as into than then there has have
had can could would will should may might must not no about any some all each every other s""".split())
WH = re.compile(r"^(?:what|which|who|whom|where|when|how|in|at|on|during|from|by|to|of|for|according|as|through|under|"
                r"among|with|out|compared|after|before|since|about|between)\b", re.I)
YESNO = re.compile(r"^(?:is|are|was|were|does|do|did|can|could|would|will|has|have|had|should|may|might|must|if|"
                   r"considering)\b", re.I)
YEAR_Q = re.compile(r"^(?:(?:in|during|by|since|from|until) )?(?:when\b|what year\b|which year\b|what (?:was|is) the year\b)",
                    re.I)
NUM_Q = re.compile(
    r"\bhow (?:many|much|old|tall|long|far|big|large|heavy|fast|deep|high|wide|often|hot|cold|expensive|short|small|"
    r"far away)\b|\bwhat (?:is|was|are|were) the (?:\w+ )?(?:population|number|age|height|weight|length|size|cost|"
    r"price|area|speed|temperature|distance|capacity|duration|lifespan|life expectancy|rate|percentage|amount|depth|"
    r"width|diameter|mass|volume|salary|income|revenue|budget|count|total|net worth|wingspan|circumference|"
    r"elevation|altitude|top speed|boiling point|melting point|density|value|worth|gross|runtime|running time)\b"
    r"|\bwhat percent", re.I)
MONTHS = ("January February March April May June July August September October November December "
          "Jan Feb Mar Apr Jun Jul Aug Sep Sept Oct Nov Dec").split()
WEEKDAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
_DATE = re.compile(r"\b(?:%s)\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+(?=\d{4}\b)|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:%s)"
                   r",?\s+(?=\d{4}\b)" % ("|".join(MONTHS), "|".join(MONTHS)))
ONES = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                   "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
SCALES = {"hundred": 100, "thousand": 1000, "million": 10 ** 6, "billion": 10 ** 9, "trillion": 10 ** 12}
FRACTION_WORDS = {"half", "halves", "quarter", "quarters", "third", "thirds", "dozen", "dozens", "twice", "double",
                  "triple", "thrice"}
_NUMWORD = r"(?:%s)" % "|".join(sorted(list(ONES) + list(TENS), key=len, reverse=True))
NUM_EXPR = re.compile(
    r"(?P<pre>\$)?(?P<sign>[-−])?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s+(?P<scale>hundred|thousand|million|billion|trillion)\b)?"
    r"|(?P<word>\b%s(?:[- ]%s)?\b(?:\s+(?P<wscale>hundred|thousand|million|billion|trillion)\b)?)" % (_NUMWORD, _NUMWORD),
    re.I)
HEDGE_BEFORE = re.compile(
    r"(?:\b(?:over|under|about|around|approximately|approximate|approx|roughly|nearly|almost|some|estimated|up to|at least|"
    r"at most|more than|less than|fewer than|close to|upwards of|as many as|as much as|as few as|as little as|below|"
    r"above|beyond|exceeding|in excess of|only about|just over|just under|circa|ca|between|from|reach|reaches|"
    r"reached|reaching|near|starting|as \w+ as|exceed|exceeds|exceeded|surpass|surpasses|surpassed|top|tops|topped)\s*"
    r"(?:\$\s*)?|[~<>≈])$", re.I)
QUALIFIERS = frozenset("""maximum minimum max min record peak highest lowest largest smallest longest shortest tallest
fastest slowest oldest youngest heaviest lightest deepest biggest up reach reaches reached reaching""".split())
RANGE_AFTER = re.compile(r"^\s*(?:to|-|–|or|and|through)\s*(?:\$\s*)?(?:\d|%s\b)" % _NUMWORD, re.I)
RANGE_BEFORE = re.compile(r"(?:\d|\b%s)\s*(?:to|-|–|or|and|through)\s*(?:\$\s*)?$" % _NUMWORD, re.I)
HEDGE_AFTER = re.compile(r"^(?:\+|\s*(?:or (?:more|less|so|fewer)|plus|-ish)\b)", re.I)
UNIT_WORDS = frozenset("""% percent percentage °F °C degrees mph MPH km/h kph km kg lb lbs oz mm cm m ft in sq foot feet inch
inches mile miles meter meters metre metres kilometer kilometers kilometre kilometres gram grams kilogram kilograms pound
pounds ton tons tonne tonnes ounce ounces liter liters litre litres gallon gallons year years month months week weeks day
days hour hours minute minutes second seconds dollar dollars euro euros calorie calories acre acres people""".split())
TYPE_NOUNS = frozenset("""country city state company team band language religion university college school river ocean sea
continent island mountain nation empire kingdom province county region party network brand studio league church
planet god goddess deity president king queen emperor author writer director actor actress singer artist composer
painter person man woman scientist inventor founder leader character franchise publisher manufacturer airline
dynasty tribe army navy capital town village organization organisation agency club label station""".split())
DEMONYMS = frozenset("""American British English French German Italian Spanish Mexican Chinese Japanese Korean Indian
Russian Canadian Australian Irish Scottish Welsh Greek Roman Egyptian Arab Arabic Persian Turkish Dutch Swedish
Norwegian Danish Finnish Polish Brazilian Argentine African European Asian Christian Catholic Protestant Muslim Islamic
Jewish Hindu Buddhist Sikh Mormon Orthodox Native Western Eastern Northern Southern""".split())
LEAD_DROP = frozenset("""The A An In On At During After Before Since As It There This That These Those Most Many Some All
Only Every Each His Her Their Its Our My Your When While If Although Though Because However Also Both Neither Either
According Today Currently""".split())
CAP_JOIN = frozenset({"of", "the", "de", "da", "del", "der", "van", "von", "la", "le", "du", "&", "y"})
TITLES = frozenset("""President King Queen Prince Princess Duke Duchess Emperor Empress Governor Mayor Senator Pope
Bishop Archbishop Chancellor Minister Secretary Chairman CEO Director Professor Dean Lord Lady Sir Earl Count Baron
General Admiral Captain Chief Head Leader Founder Mother Father Saint St CEO""".split())
CLOSED_CLASSES = {"continent": frozenset("Africa Antarctica Asia Australia Europe Oceania".split()
                                         + ["North America", "South America"]),
                  "planet": frozenset("Mercury Venus Earth Mars Jupiter Saturn Uranus Neptune Pluto".split())}
NOT_COUNTRIES = frozenset(["Arctic", "Antarctic", "Atlantic", "Pacific", "Indian Ocean", "Mediterranean", "Caribbean",
                           "Middle East", "Sahara", "Siberia", "Scandinavia", "Balkans", "Arctic Circle"])
ENTITY_PREP = frozenset({"in", "from", "at", "near", "on", "to", "of"})


# ------------------------------------------------------------------------------------------------------ text
def norm(s: str) -> str:
    return " ".join(WTOK.findall(s.lower().replace("’", "'")))


def stem(w: str) -> str:
    """A word lower-cased, possessive and edge punctuation dropped (the unit content words compare)."""
    return w.lower().replace("’", "'").strip(".-'").removesuffix("'s")


def content(s: str) -> set:
    """Content words (stem) of a text, without stop words and #k references."""
    return {stem(w) for w in WORD.findall(s) if stem(w) and stem(w) not in STOP and not w.startswith("#")}


def capitals(s: str) -> set:
    """Content words of the capitalised words of a step (its first word, a question word, excluded)."""
    ws = WORD.findall(s)
    return {stem(w) for w in ws[1:] if w[:1].isupper() and stem(w) and stem(w) not in STOP and not w.startswith("#")}


def same_word(a: str, b: str) -> bool:
    """Two content words match: equal, the same first 5 letters, or one a prefix (>= 3 letters) of the other
    (die / died, found / founded, kid / kids)."""
    if a == b or (len(a) >= 5 and len(b) >= 5 and a[:5] == b[:5]):
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short) >= 3 and long_.startswith(short)


def covered(words: set, other: set) -> set:
    """The words of `words` that some word of `other` matches."""
    return {w for w in words if any(same_word(w, o) for o in other)}


def ask_text(sq: str) -> str:
    sq = " ".join(sq.split())
    return sq if sq[-1:] in (".", "?", "!") else sq + "?"


def node_question(step: str, x: str) -> str:
    return f"{ask_text(step)} Is it {x}?"


def node_opposite(step: str, x: str) -> str:
    return f"{ask_text(step)} Is it different from {x}?"


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


# ------------------------------------------------------------------------------------------------------ numbers
def word_value(text: str) -> int | None:
    parts = re.split(r"[- ]+", text.lower())
    v = 0
    for p in parts:
        if p in ONES:
            v += ONES[p]
        elif p in TENS:
            v += TENS[p]
        else:
            return None
    return v


def fmt_mantissa(x: float, decimals: int, commas: bool) -> str:
    s = f"{x:,.{decimals}f}" if commas else f"{x:.{decimals}f}"
    return s


def trailing_zeros(n: int, cap: int = 3) -> int:
    k = 0
    while n and n % 10 ** (k + 1) == 0 and k < cap:
        k += 1
    return k


def number_mentions(text: str) -> list[dict]:
    """Every number expression of a text, in order: span, sign, "$", mantissa value and text form (decimals, commas),
    scale, whether it is clean (not glued to a letter or hyphen: "1990s", "B-52", "18th", "6-7" are not), whether it is
    a capitalised number word inside a name ("Rogue One")."""
    out = []
    for m in NUM_EXPR.finditer(text):
        a, b = m.span()
        before, after = text[:a], text[b:]
        d = {"span": (a, b), "text": m.group(0), "pre": m.group("pre") or "", "sign": "-" if m.group("sign") else ""}
        if m.group("num"):
            raw = m.group("num")
            d["mantissa"] = float(raw.replace(",", ""))
            d["decimals"] = len(raw.split(".")[1]) if "." in raw else 0
            d["commas"] = "," in raw
            d["scale"] = (m.group("scale") or "").lower()
            glued = bool(re.match(r"[A-Za-z0-9]|-[A-Za-z0-9]|\.\d|/\d", after)) or bool(re.search(r"[A-Za-z/]-?$", before))
            d["clean"] = not glued
            d["name_word"] = False
        else:
            w = m.group("word")
            base = re.sub(r"\s+(?:hundred|thousand|million|billion|trillion)$", "", w, flags=re.I)
            d["mantissa"] = float(word_value(base))
            d["decimals"], d["commas"] = 0, False
            d["scale"] = (m.group("wscale") or "").lower()
            d["clean"] = not re.match(r"-[A-Za-z]", after)
            d["name_word"] = w[:1].isupper() and bool(before.strip()) and not re.search(r"[.!?]\s*$", before)
            if w.lower() == "one" and (re.match(r"\s+(?:of|another|who|that|which|day|time)\b", after, re.I)
                                       or re.search(r"\b(?:the|no|any|each|every|this|that|which|someone|anyone|no)\s+$",
                                                    before, re.I)):
                continue                             # "one of the ...", "the one", "no one": not a count
        d["value"] = d["mantissa"] * SCALES.get(d["scale"], 1) * (-1 if d["sign"] else 1)
        out.append(d)
    return out


def other_number_tokens(text: str, mentions: list[dict]) -> int:
    """Digit runs and fraction words outside the parsed mentions (ordinals, decades, ranges, "half", "dozen" ...)."""
    covered = [m["span"] for m in mentions]
    n = 0
    for m in re.finditer(r"\d+|\b(?:%s)\b" % "|".join(FRACTION_WORDS), text, re.I):
        if not any(a <= m.start() < b for a, b in covered):
            n += 1
    return n


NOT_UNITS = frozenset("""is was has does as this his its plus less minus across versus thus yes various previous
famous serious""".split())


def unit_after(text: str, end: int) -> str:
    """The unit right after a number, with the space in front as the fact writes it ('' when none): %, a degree sign,
    "degrees [Celsius|Fahrenheit]", "square / cubic <unit>", a unit word or a plural noun ("12 disciples"); kept with
    the value in both probes."""
    m = re.match(r"(\s*)(%|°\s?[FC]\b|degrees(?: (?:Fahrenheit|Celsius|F|C))?\b|(?:square|cubic) [a-z]+\b|"
                 r"[A-Za-z][A-Za-z/]*)", text[end:], re.I)
    if not m:
        return ""
    sp, u = m.group(1)[:1], m.group(2)
    if u == "%" or u.startswith("°") or u.lower().startswith("degrees") or re.match(r"(?:square|cubic) ", u, re.I):
        return sp + u.replace("° ", "°")
    lw = u.lower()
    if lw in STOP or lw in SCALES or lw in NOT_UNITS or not sp:
        return ""
    if u in UNIT_WORDS or lw in UNIT_WORDS:
        return sp + u
    if u == lw and lw.endswith("s") and not lw.endswith("ss") and len(lw) > 3 and "/" not in lw:
        return sp + u                                # a counted noun ("12 disciples", "9 digits")
    return ""


def render_number(d: dict, mantissa: float) -> str:
    s = d["pre"] + d["sign"] + fmt_mantissa(mantissa, d["decimals"], d["commas"])
    if d["scale"]:
        s += " " + d["scale"]
    return s + d["unit"]


def ok_wrong_number(w: float, d: dict, item_numbers: set) -> bool:
    v = d["mantissa"]
    if w <= 0 or v <= 0 or abs(w - v) < 1e-9:
        return False
    if d["decimals"] == 0 and abs(w - round(w)) > 1e-9:
        return False
    if d["decimals"] == 0 and trailing_zeros(int(round(w))) != trailing_zeros(int(round(v))):
        return False
    r = max(w / v, v / w)
    if d["unit"].strip().lower() in ("%", "percent") and w > 100:
        return False                                 # a share above 100% would be false on its face
    full = w * SCALES.get(d["scale"], 1) * (-1 if d["sign"] else 1)
    return MIN_RATIO - 1e-9 <= r <= MAX_RATIO and round(full, 6) not in item_numbers


def number_candidates(d: dict) -> list[float]:
    """Nearby values of the mantissa: factors 1/4 .. 4 rounded to its precision and roundness, and small offsets."""
    v = d["mantissa"]
    out = set()
    if d["decimals"] == 0:
        k = trailing_zeros(int(round(v)))
        q = 10 ** k
        for f in (0.25, 1 / 3, 0.4, 0.5, 0.6, 2 / 3, 0.75, 1.25, 4 / 3, 1.5, 5 / 3, 2, 2.5, 3, 4):
            out.add(float(round(v * f / q) * q))
        if v <= 20:
            out |= {float(v + s) for s in range(-6, 7) if s}
    else:
        p = 10 ** -d["decimals"]
        for f in (0.25, 0.4, 0.5, 0.6, 0.75, 1.25, 1.5, 2, 2.5, 3, 4):
            out.add(round(round(v * f / p) * p, d["decimals"]))
    return sorted(out)


def year_candidates(y: int) -> list[int]:
    k = trailing_zeros(y, cap=2)
    out = []
    for dlt in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 40, 50):
        for w in (y - dlt, y + dlt):
            if YEAR_MIN <= w <= YEAR_MAX and trailing_zeros(w, cap=2) == k:
                out.append(w)
    return sorted(set(out))


# ------------------------------------------------------------------------------------------------------ answers
def align(step: str, facts: list[str]) -> tuple[int | None, str]:
    """The index of the one fact that answers the step, else (None, reason)."""
    cs, caps = content(step), capitals(step)
    if not cs:
        return None, "no_content_words"
    scored = []
    for i, f in enumerate(facts):
        cf = content(f)
        if caps - covered(caps, cf):
            continue
        scored.append((len(covered(cs, cf)) / len(cs), i))
    scored.sort(reverse=True)
    if not scored or scored[0][0] < ALIGN_MIN:
        return None, "no_fact_aligns"
    if len(scored) > 1 and scored[1][0] == scored[0][0]:
        return None, "two_facts_align"
    return scored[0][1], ""


def numeric_answer(step: str, fact: str, kind: str) -> tuple[dict | None, str]:
    """The answer of a number / year step from its fact, else (None, reason)."""
    text = _DATE.sub(lambda m: " " * len(m.group(0)), fact) if kind == "year" else fact
    ms = number_mentions(text)
    step_vals = {round(m["value"], 6) for m in number_mentions(step)} | {
        round(m["mantissa"], 6) for m in number_mentions(step)}
    cand = [m for m in ms if round(m["value"], 6) not in step_vals and round(m["mantissa"], 6) not in step_vals
            and not m["name_word"]]
    extra = other_number_tokens(text, ms)
    if not cand:
        return None, "no_value_in_fact"
    if len(cand) > 1 or extra:
        return None, "several_values_in_fact"
    d = dict(cand[0])
    a, b = d["span"]
    if not d["clean"]:
        return None, "value_glued_to_word"
    if RANGE_AFTER.match(text[b:]) or RANGE_BEFORE.search(text[:a]):
        return None, "value_is_a_range"              # "2 to 7 players" (7 also in the step: "7 Wonders")
    if HEDGE_BEFORE.search(text[max(0, a - 30):a]) or HEDGE_AFTER.match(text[b:]):
        return None, "hedged_value"
    if {stem(w) for w in WORD.findall(fact)} & QUALIFIERS - {stem(w) for w in WORD.findall(step)}:
        return None, "fact_qualifies_value"          # "the maximum length is ...", "the record is ..." for a plain ask
    if kind == "year":
        y = d["mantissa"]
        if d["scale"] or d["pre"] or d["sign"] or d["decimals"] or not YEAR_MIN <= y <= YEAR_MAX or d.get("word"):
            return None, "when_answer_not_a_year"
        if re.match(r"\s*(?:BC|BCE|B\.C\.)\b", text[b:]):
            return None, "when_answer_not_a_year"
        d["unit"] = ""
        d["kind"] = "year"
        d["answer"] = str(int(y))
        return d, ""
    d["unit"] = unit_after(text, b)
    if d["unit"].strip() == "%" and d["scale"]:
        return None, "several_values_in_fact"
    if (not d["unit"] and not d["scale"] and not d["pre"] and not d["commas"] and not d["decimals"]
            and YEAR_MIN <= d["mantissa"] <= 2030 and re.fullmatch(r"\d{4}", d["text"])):
        return None, "number_answer_is_a_year"     # "How old was she when ...?" answered by the year of the event
    d["kind"] = "number"
    d["answer"] = render_number(d, d["mantissa"])
    return d, ""


def cap_spans(fact: str) -> list[tuple[int, int, str]]:
    """Maximal runs of capitalised words (joined by of / the / de / ...), leading determiners dropped."""
    toks = list(re.finditer(r"[A-Za-z0-9][A-Za-z0-9'’.&-]*", fact))
    spans, i = [], 0
    while i < len(toks):
        w = toks[i].group()
        if w[:1].isupper():
            j = i
            while j + 1 < len(toks):
                nxt = toks[j + 1].group()
                if nxt[:1].isupper() and not re.search(r"[,;:()\"]", fact[toks[j].end():toks[j + 1].start()]):
                    j += 1
                elif (nxt.lower() in CAP_JOIN and j + 2 < len(toks) and toks[j + 2].group()[:1].isupper()
                      and not re.search(r"[,;:()\"]", fact[toks[j].end():toks[j + 2].start()])):
                    j += 2
                else:
                    break
            k = i
            while k <= j and toks[k].group() in LEAD_DROP:
                k += 1
            if k <= j:
                spans.append((toks[k].start(), toks[j].end(), fact[toks[k].start():toks[j].end()].rstrip(".")))
            i = j + 1
        else:
            i += 1
    return spans


def entity_class(step: str) -> tuple[str | None, str]:
    """The answer class of an entity step: person (who), place (where), the singular type noun of "which / what <noun>"."""
    s = step.strip()
    m = re.match(r"^(?:(?:in|at|on|from|of|for|to|by|with|during|according to) )?(who|whom|where|which|what)\b\s*(.*)$",
                 s, re.I)
    if not m:
        return None, "not_an_entity_question"
    wh, rest = m.group(1).lower(), m.group(2)
    if wh in ("who", "whom"):
        if re.match(r"^(?:were|are|have|do)\b", rest, re.I):
            return None, "plural_answer"
        pred = [w for w in WORD.findall(re.sub(r"^(?:is|was)\b", "", rest, flags=re.I))
                if stem(w) and stem(w) not in STOP and not w.startswith("#")
                and (w[:1].islower() or w in TITLES or (len(w) > 1 and w.isupper()))]
        if not pred:
            return None, "who_is_a_description"     # "Who is Quiet?": asks what the name is, not for a name
        return "person", ""
    if wh == "where":
        return "place", ""
    if re.match(r"^(?:number|percentage|percent|amount|year|age|size|time|date|day|month|century|decade)\b", rest, re.I):
        return None, "not_an_entity_question"
    words = re.findall(r"[A-Za-z]+", rest)[:3]
    for w in words:
        lw = w.lower()
        if lw in TYPE_NOUNS:
            return lw, ""
        if lw.endswith("s") and (lw[:-1] in TYPE_NOUNS or lw[:-2] in TYPE_NOUNS or lw[:-3] + "y" in TYPE_NOUNS):
            return None, "plural_answer"
    return None, "not_an_entity_question"


def entity_answer(step: str, fact: str, cls: str) -> tuple[dict | None, str]:
    st = content(step)
    cands = []
    for a, b, span in cap_spans(fact):
        ws = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.&-]*", span)
        if any(stem(w) in st for w in ws):
            continue                                 # the step's own subject
        if span.endswith(("'s", "’s")) or fact[b:b + 2] in ("'s", "’s"):
            cands.append(None)                       # a possessive: the answer is what it owns, not a name
            continue
        if span in MONTHS or span in WEEKDAYS or any(w in MONTHS or w in WEEKDAYS for w in ws):
            continue
        if len(ws) == 1 and (span in DEMONYMS or a == 0 or re.match(r"[.!?]\s*$", fact[:a].rstrip() + " ")):
            cands.append(None)                       # a demonym or a lone sentence-initial word: not a clean name
            continue
        if re.search(r"\d", span) or ws[-1] in DEMONYMS or (len(ws) > 1 and ws[0] in TITLES and ws[1] in CAP_JOIN):
            cands.append(None)                       # "South Korean", "President of South Africa": not a clean name
            continue
        cands.append((a, b, span))
    if not cands:
        return None, "no_name_in_fact"
    if len(cands) > 1 or cands[0] is None:
        return None, "several_names_in_fact"
    a, b, span = cands[0]
    if cls == "person" and len(span.split()) < 2:
        return None, "person_not_a_full_name"
    if cls in CLOSED_CLASSES and span not in CLOSED_CLASSES[cls]:
        return None, "answer_not_of_the_class"
    if cls in ("country", "nation", "state") and (span in CLOSED_CLASSES["continent"] or span in NOT_COUNTRIES):
        return None, "answer_not_of_the_class"      # "What country is at the latitude of 75° N?" -> "Arctic"
    if cls == "place" and fact[:a].split()[-1:] and fact[:a].split()[-1].lower() not in ENTITY_PREP:
        return None, "place_not_after_preposition"
    return {"kind": "entity", "class": cls, "answer": span, "span": (a, b), "value": span}, ""


# ------------------------------------------------------------------------------------------------------ steps
def step_refs(step: str) -> list[int]:
    return sorted({int(x) for x in re.findall(r"#(\d+)", step)})


def resolve(step: str, answers: dict) -> str | None:
    refs = step_refs(step)
    if any(k not in answers for k in refs):
        return None
    return re.sub(r"#(\d+)", lambda m: answers[int(m.group(1))], step)


def operation_step(r: dict, j: int) -> bool:
    evs = [ann[j] for ann in r.get("evidence") or [] if j < len(ann)]
    return bool(evs) and all(any(e == "operation" for e in ev) for ev in evs)


def step_answers(r: dict, st: collections.Counter) -> list[dict]:
    """The answered steps of one raw question (no wrong values yet): step index, resolved text, fact, answer."""
    d, facts = r["decomposition"], r["facts"]
    fin = norm(r["question"])
    answers, out, seen = {}, [], set()
    for j, s in enumerate(d):
        if j == len(d) - 1:
            st["dropped_last_step"] += 1
            continue
        s = " ".join(s.split())
        if norm(s) == fin:
            st["dropped_step_is_final_question"] += 1
            continue
        if operation_step(r, j):
            st["dropped_operation_step"] += 1
            continue
        if YESNO.match(s) or not WH.match(s):
            st["dropped_yesno_step" if YESNO.match(s) else "dropped_not_a_wh_step"] += 1
            continue
        txt = resolve(s, answers)
        if txt is None:
            st["dropped_unresolved_ref"] += 1
            continue
        kind = "year" if YEAR_Q.match(txt) else "number" if NUM_Q.search(txt) else None
        cls = None
        if kind is None:
            cls, why = entity_class(txt)
            if cls is None:
                st[f"dropped_{why}"] += 1
                continue
        fi, why = align(txt, facts)
        if fi is None:
            st[f"dropped_{why}"] += 1
            continue
        a, why = numeric_answer(txt, facts[fi], kind) if kind else entity_answer(txt, facts[fi], cls)
        if a is None:
            st[f"dropped_{why}"] += 1
            continue
        if a["answer"].lower() in seen:
            st["dropped_same_answer_as_other_step"] += 1
            continue
        seen.add(a["answer"].lower())
        answers[j + 1] = a["answer"]
        out.append({"j": j, "step": s, "resolved": txt, "fact": facts[fi], "fact_index": fi, "refs": step_refs(s),
                    **a})
    return out


def item_numbers(r: dict) -> set:
    """Every number value of the question, decomposition and facts (a wrong number is none of them)."""
    out = set()
    for t in [r["question"], *r["decomposition"], *r["facts"]]:
        for m in number_mentions(t):
            out.add(round(m["value"], 6))
            out.add(round(m["mantissa"], 6))
    return out


def item_text(r: dict) -> str:
    return " ".join([r["question"], *r["decomposition"], *r["facts"]]).lower()


def wrong_answer(a: dict, r: dict, pools: dict, rng: random.Random) -> tuple[str, float | str, str] | None:
    """(probe text, probe value, how) for the No node of an answered step, else None."""
    up = rng.random() < 0.5
    if a["kind"] == "entity":
        text = item_text(r)
        val_words = {w.lower() for w in re.findall(r"[A-Za-z0-9]+", a["answer"])}
        pool = [x for x in pools.get(("entity", a["class"]), []) if x.lower() not in text
                and not ({w.lower() for w in re.findall(r"[A-Za-z0-9]+", x)} & val_words)]
        if not pool:
            return None
        n = len(a["answer"].split())
        same = [x for x in pool if len(x.split()) == n]
        x = rng.choice(same or pool)
        return x, x, "other"
    nums = item_numbers(r)
    if a["kind"] == "year":
        y = int(a["mantissa"])
        k = trailing_zeros(y, cap=2)
        sides = {}
        for side in (True, False):
            per = [w for w in year_candidates(y) if (w > y) == side and w not in nums]
            oth = sorted({w for w in pools.get(("year", ""), []) if (w > y) == side and 5 <= abs(w - y) <= 50
                          and trailing_zeros(w, cap=2) == k and w not in nums and YEAR_MIN <= w <= YEAR_MAX})
            sides[side] = {"perturb": per, "other": oth}
        for side in (up, not up):
            avail = [k_ for k_ in ("other", "perturb") if sides[side][k_]]
            if avail:
                how = rng.choices(avail, weights=[WRONG_W[x] for x in avail])[0]
                w = rng.choice(sides[side][how])
                return str(w), float(w), how
        return None
    d = a
    sides = {}
    for side in (True, False):
        per = [w for w in number_candidates(d) if (w > d["mantissa"]) == side and ok_wrong_number(w, d, nums)]
        oth = sorted({w for w in pools.get(("number", (d["unit"].strip().lower(), d["scale"]), d["decimals"]), [])
                      if (w > d["mantissa"]) == side and ok_wrong_number(w, d, nums)})
        sides[side] = {"perturb": per, "other": oth}
    for side in (up, not up):
        avail = [k_ for k_ in ("other", "perturb") if sides[side][k_]]
        if avail:
            how = rng.choices(avail, weights=[WRONG_W[x] for x in avail])[0]
            w = rng.choice(sides[side][how])
            full = w * SCALES.get(d["scale"], 1) * (-1 if d["sign"] else 1)
            return render_number(d, w), full, how
    return None


def answer_pools(answered: dict) -> dict:
    """Values of every answered step of a split, by kind: entity answers by class, years, numbers by (unit, scale,
    decimals). Wrong answers of an item come from these (its own are excluded by wrong_answer's checks)."""
    pools = collections.defaultdict(set)
    for steps in answered.values():
        for a in steps:
            if a["kind"] == "entity":
                pools[("entity", a["class"])].add(a["answer"])
            elif a["kind"] == "year":
                pools[("year", "")].add(int(a["mantissa"]))
            else:
                pools[("number", (a["unit"].strip().lower(), a["scale"]), a["decimals"])].add(a["mantissa"])
    return {k: sorted(v) for k, v in pools.items()}


def tf_nodes(r: dict, steps: list[dict], pools: dict, rng: random.Random, st: collections.Counter) -> list[dict]:
    """The yes/no nodes of one question from its answered steps (step_answers)."""
    last = len(r["decomposition"])
    reads = {j + 1: step_refs(s) for j, s in enumerate(r["decomposition"])}

    def ancestors_of_last() -> set:
        seen, todo = set(), list(reads[last])
        while todo:
            k = todo.pop()
            if k not in seen and k in reads:
                seen.add(k)
                todo += reads[k]
        return seen
    gold_anc = ancestors_of_last()
    nodes, depth, built = [], {}, set()
    for a in steps:
        k = a["j"] + 1
        if any(x not in built for x in a["refs"]):
            st["dropped_parent_without_nodes"] += 1      # its reference was answered but got no wrong value
            continue
        w = wrong_answer(a, r, pools, rng)
        if w is None:
            st[f"dropped_no_wrong_{a['kind']}"] += 1
            continue
        par = [f"s{x}v" for x in a["refs"]]
        depth[k] = 1 + max((depth[x] for x in a["refs"]), default=-1)
        meta = {"step": k, "step_text": a["step"], "resolved": a["resolved"], "kind": a["kind"],
                "value": a["value"] if a["kind"] == "entity" else (int(a["mantissa"]) if a["kind"] == "year"
                                                                   else a["value"]),
                "value_text": a["answer"], "unit": a.get("unit", ""), "entity_class": a.get("class"),
                "fact": a["fact"], "fact_index": a["fact_index"], "gold_ancestor": k in gold_anc,
                "node_depth": depth[k]}
        for suffix, (ptxt, pval, how), truth in (("v", (a["answer"], meta["value"], None), True), ("n", w, False)):
            n = {"id": f"s{k}{suffix}", "question": node_question(a["resolved"], ptxt),
                 "opposite": node_opposite(a["resolved"], ptxt), "truth": truth, "p_cot": 1.0 if truth else 0.0,
                 "p_fresh": None, "type": "step", "depends_on": par, "initial": None, "status": "direct",
                 "probe": pval, "probe_text": ptxt, **meta}
            if how:
                n["wrong"] = how
                st[f"wrong_{a['kind']}_{how}"] += 1
            nodes.append(n)
        built.add(k)
        st[f"built_{a['kind']}"] += 1
    return nodes


# ------------------------------------------------------------------------------------------------------ items
def final_item(r: dict) -> dict:
    q = " ".join(r["question"].split())
    n = len(r["decomposition"])
    gold = "A" if r["answer"] else "B"
    return {"item_id": f"strategyqa-{r['qid']}", "source": SOURCE, "group": f"{SOURCE}/s{n}", "type": "choice",
            "prompt": f"Question: {q}\nOptions:\nA) Yes\nB) No", "labels": list(LABELS), "gold_label": gold,
            "teacher": {L: float(L == gold) for L in LABELS}, "options": list(OPTIONS), "depth": n,
            "subqs": [], "random_subqs": [], "rationale": "", "augment": None, "has_tree": False, "tree_fix": True,
            "row_kind": "final", "problem_id": f"strategyqa-{r['qid']}", "answer": bool(r["answer"]),
            "term": r.get("term"), "licence_class": "UNCHECKED", "licence": LICENCE, "label_basis": "gold",
            "node_basis": "decomposition+facts", "n_steps": n, "n_facts": len(r["facts"]), "tree_source": TREE_SOURCE}


def gold_text(item: dict) -> str:
    return item["options"][item["labels"].index(item["gold_label"])]


def question_of(item: dict) -> str:
    head = item["prompt"].partition("\nOptions:\n")[0]
    return head[len("Question: "):]


def audit_item(item: dict) -> list[str]:
    """Problems of one built item, from its own fields only (empty: clean)."""
    from cotdistill.teacher import leaks_answer
    bad, iid = [], item.get("item_id")
    q = question_of(item)
    if (item.get("source") != SOURCE or item.get("n_steps") != item.get("depth") or item.get("tree_fix") is not True
            or item["labels"] != LABELS or item["options"] != OPTIONS
            or item["prompt"] != f"Question: {q}\nOptions:\nA) Yes\nB) No" or "\n" in q
            or gold_text(item) != ("Yes" if item["answer"] else "No")
            or item["teacher"] != {L: float(L == item["gold_label"]) for L in LABELS}
            or item["has_tree"] != bool(item["subqs"]) or item["group"] != f"{SOURCE}/s{item['n_steps']}"):
        bad.append("item fields")
    by = {n["id"]: n for n in item["subqs"]}
    if len(by) != len(item["subqs"]):
        bad.append("duplicate node ids")
    for n in item["subqs"]:
        i = n["id"]
        if n["fact"] in item["prompt"] or n["fact"].strip() in q:
            bad.append(f"{i}: its fact is in the prompt")
        if n["value_text"] not in n["fact"] and n["kind"] == "entity":
            bad.append(f"{i}: answer {n['value_text']!r} not in its fact")
        if n["kind"] != "entity":
            vals = {round(m["value"], 6) for m in number_mentions(_DATE.sub(" ", n["fact"]) if n["kind"] == "year"
                                                                  else n["fact"])}
            if round(float(n["value"]), 6) not in vals:
                bad.append(f"{i}: value {n['value']} not in its fact")
        if (n["truth"] is True) != (n["probe_text"] == n["value_text"]) or n["p_cot"] != float(n["truth"]):
            bad.append(f"{i}: truth is not (probe == answer)")
        if n["kind"] != "entity" and (n["truth"] is True) != (abs(float(n["probe"]) - float(n["value"])) < 1e-9):
            bad.append(f"{i}: numeric truth is not (probe == value)")
        if (n["id"].endswith("v") != n["truth"] or n["question"] != node_question(n["resolved"], n["probe_text"])
                or n["opposite"] != node_opposite(n["resolved"], n["probe_text"])):
            bad.append(f"{i}: id / wording")
        if "#" in n["question"] or YESNO.match(n["resolved"]) or n["step"] >= item["n_steps"]:
            bad.append(f"{i}: unresolved reference, yes/no step or last step")
        if norm(n["resolved"]) == norm(q) or leaks_answer(n["question"], item):
            bad.append(f"{i}: restates the final question")
        twin = by.get(i[:-1] + ("n" if n["truth"] else "v"))
        if not twin or twin["value_text"] != n["value_text"] or twin["truth"] == n["truth"]:
            bad.append(f"{i}: no Yes/No twin")
        if not n["truth"] and n["kind"] == "entity" and n["probe_text"].lower() in n["fact"].lower():
            bad.append(f"{i}: wrong entity occurs in the fact")
        depth = 0
        for d in n["depends_on"]:
            par = by.get(d)
            if not par or not par["truth"]:
                bad.append(f"{i}: parent {d} missing or not Yes")
                continue
            if par["value_text"] not in n["resolved"]:
                bad.append(f"{i}: parent {d} answer not in the question")
            depth = max(depth, par["node_depth"] + 1)
        if n["node_depth"] != depth:
            bad.append(f"{i}: node_depth {n['node_depth']} != {depth}")
    return [f"{iid}: {b}" for b in bad]


def cue_features(item: dict) -> list[tuple]:
    """(truth, probe in the question, probe a multiple of 10, probe even, probe has 2+ words, No probe above the
    value, probe longer than the answer in characters) per node, as seen in the GFPQ view (the question only)."""
    q = question_of(item).lower()
    rows = []
    for n in item["subqs"]:
        num = n["kind"] != "entity"
        x = float(n["probe"]) if num else None
        rows.append((n["truth"], n["probe_text"].lower() in q,
                     None if not num or abs(x - round(x)) > 1e-9 else round(x) % 10 == 0,
                     None if not num or abs(x - round(x)) > 1e-9 else round(x) % 2 == 0,
                     len(n["probe_text"].split()) >= 2,
                     None if n["truth"] or not num else x > float(n["value"]),
                     None if n["truth"] else len(n["probe_text"]) > len(n["value_text"])))
    return rows


def stump(rows: list[tuple], k: int) -> float | None:
    xs = [(r[0], r[k]) for r in rows if r[k] is not None]
    if not xs:
        return None
    acc = sum(t == f for t, f in xs) / len(xs)
    return round(max(acc, 1 - acc), 3)


# ------------------------------------------------------------------------------------------------------ io
def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def write(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return md5(path)


def stats(items: list[dict]) -> dict:
    nodes = [n for it in items for n in it["subqs"]]
    return {"items": len(items), "with_tree": sum(it["has_tree"] for it in items),
            "with_tree_share": round(sum(it["has_tree"] for it in items) / max(1, len(items)), 3),
            "gold_yes_share": round(sum(it["answer"] for it in items) / max(1, len(items)), 4),
            "n_steps": dict(sorted(collections.Counter(it["n_steps"] for it in items).items())),
            "nodes": len(nodes), "nodes_per_item": round(len(nodes) / max(1, len(items)), 2),
            "nodes_per_tree_item": round(len(nodes) / max(1, sum(it["has_tree"] for it in items)), 2),
            "yes_share": round(sum(n["truth"] for n in nodes) / max(1, len(nodes)), 4),
            "node_kind": dict(sorted(collections.Counter(n["kind"] for n in nodes).items())),
            "node_depth": dict(sorted(collections.Counter(n["node_depth"] for n in nodes).items())),
            "gold_ancestor_share": round(sum(n["gold_ancestor"] for n in nodes) / max(1, len(nodes)), 3),
            "yes_share_by_gold": {g: round(sum(n["truth"] for it in items if it["answer"] == (g == "Yes")
                                               for n in it["subqs"])
                                           / max(1, sum(len(it["subqs"]) for it in items
                                                        if it["answer"] == (g == "Yes"))), 4)
                                  for g in ("Yes", "No")}}


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------------------------------------------ splits
def dedup(raw: list[dict], st: collections.Counter) -> list[dict]:
    """One question per normalised text (the first in file order); a text with two different answers is dropped."""
    by = collections.defaultdict(list)
    for r in raw:
        by[norm(r["question"])].append(r)
    keep = []
    for r in raw:
        rs = by[norm(r["question"])]
        if len({x["answer"] for x in rs}) > 1:
            st["dedup_dropped_conflicting_answer"] += 1
            continue
        if rs[0] is not r:
            st["dedup_dropped_same_question"] += 1
            continue
        keep.append(r)
    return keep


def split(raw: list[dict], test_frac: float, val_frac: float, seed: int) -> dict[str, list[dict]]:
    """Stratified by group (number of decomposition steps): each group in sorted order is shuffled (file order first)
    by one generator; round(n x test_frac) go to test, round(n x val_frac) to val, the rest to train; file order kept."""
    groups = collections.defaultdict(list)
    for i, r in enumerate(raw):
        groups[len(r["decomposition"])].append(i)
    rng = random.Random(f"{SOURCE}|split|{seed}")
    where = {}
    for g in sorted(groups):
        idx = groups[g][:]
        rng.shuffle(idx)
        nt, nv = round(len(idx) * test_frac), round(len(idx) * val_frac)
        for k, i in enumerate(idx):
            where[i] = "test" if k < nt else "val" if k < nt + nv else "train"
    out = {"train": [], "val": [], "test": []}
    for i, r in enumerate(raw):
        out[where[i]].append(r)
    return out


def near_duplicates(splits: dict, st: collections.Counter) -> dict:
    """Drop train / val questions that nearly repeat a test question (and train ones that nearly repeat a val one):
    same normalised term and content-word Jaccard >= NEAR_DUP."""
    ex = []
    for name, against in (("val", ["test"]), ("train", ["test", "val"])):
        ref = [(norm(r.get("term") or ""), content(r["question"]), r["qid"]) for a in against for r in splits[a]]
        keep = []
        for r in splits[name]:
            c, t = content(r["question"]), norm(r.get("term") or "")
            hit = next((q for tt, cc, q in ref if tt == t and jaccard(c, cc) >= NEAR_DUP), None)
            if hit:
                st[f"near_dup_dropped_{name}"] += 1
                if len(ex) < 6:
                    ex.append({"dropped": r["qid"], "split": name, "near": hit})
                continue
            keep.append(r)
        splits[name] = keep
    return {"examples": ex}


def subset(rows_text: list[str], items: list[dict], frac: float, seed: str) -> list[int]:
    """Indices (file order) of the stratified subset: data/student_gsm8k_tf_f{10,25,50}'s recipe (checked by md5)."""
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


# ------------------------------------------------------------------------------------------------------ build
def build(args) -> None:
    from cotdistill.teacher import leaks_answer
    raw_path = ROOT / args.raw
    raw = json.loads(raw_path.read_text())
    st = collections.Counter()
    st["raw_questions"] = len(raw)
    for r in raw:
        r["question"] = " ".join(r["question"].split())
    raw = dedup(raw, st)
    splits = split(raw, args.test_frac, args.val_frac, args.seed)
    nd = near_duplicates(splits, st)
    items, fails = {}, []
    for name, rows in splits.items():
        sst = collections.Counter()
        answered = {r["qid"]: step_answers(r, sst) for r in rows}
        pools = answer_pools(answered)
        its = []
        for r in rows:
            it = final_item(r)
            it["subqs"] = tf_nodes(r, answered[r["qid"]], pools, random.Random(f"{args.seed}/{r['qid']}"), sst)
            it["has_tree"] = bool(it["subqs"])
            its.append(it)
        items[name] = its
        for k, v in sst.items():
            st[k] += v
        st[f"steps_{name}"] = sum(len(r["decomposition"]) for r in rows)
    ids = {k: {it["item_id"] for it in v} for k, v in items.items()}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
    qs = {k: {norm(question_of(it)) for it in v} for k, v in items.items()}
    assert not (qs["train"] & qs["val"]) and not (qs["train"] & qs["test"]) and not (qs["val"] & qs["test"])
    # ---- audit and leaks
    for v in items.values():
        for it in v:
            fails += audit_item(it)
    leak = collections.Counter()
    for v in items.values():
        for it in v:
            q = question_of(it)
            facts_in_prompt = [n for n in it["subqs"] if n["fact"] in it["prompt"]]
            leak["facts_in_prompt"] += len(facts_in_prompt)
            for n in it["subqs"]:
                leak["nodes"] += 1
                leak["teacher_leaks_answer"] += leaks_answer(n["question"], it)
                leak["node_is_final_question"] += norm(n["resolved"]) == norm(q)
                leak["node_yesno_form"] += bool(YESNO.match(n["resolved"]))
                leak["node_from_last_step"] += n["step"] == it["n_steps"]
                leak["node_text_has_ref"] += "#" in n["question"]
                leak["node_question_jaccard_ge_0.8_with_final"] += jaccard(content(n["resolved"]), content(q)) >= 0.8
    # ---- cues in the GFPQ view (the question only), train
    rows = [r_ for it in items["train"] for r_ in cue_features(it)]
    cues = {"n": len(rows), "probe_in_question": stump(rows, 1), "probe_round10": stump(rows, 2),
            "probe_even": stump(rows, 3), "probe_2plus_words": stump(rows, 4),
            "no_probes_above_value": round(sum(r_[5] for r_ in rows if r_[5] is not None)
                                           / max(1, sum(r_[5] is not None for r_ in rows)), 3),
            "no_probe_longer_than_answer": round(sum(r_[6] for r_ in rows if r_[6] is not None)
                                                 / max(1, sum(r_[6] is not None for r_ in rows)), 3),
            "yes_probe_in_question_share": round(sum(r_[1] for r_ in rows if r_[0]) / max(1, sum(r_[0] for r_ in rows)), 3),
            "no_probe_in_question_share": round(sum(r_[1] for r_ in rows if not r_[0])
                                                / max(1, sum(not r_[0] for r_ in rows)), 3)}
    # ---- write
    out, edir = ROOT / args.out, ROOT / args.eval_dir
    summary = {"script": "scripts/build_strategyqa_tf.py", "args": vars(args),
               "raw": {"file": args.raw, "md5": md5(raw_path)}, "md5": {}, "files": {}}
    for path, rows_ in ((out / "train.jsonl", items["train"]), (out / "val.jsonl", items["val"]),
                        (edir / "strategyqa_tf_test.jsonl", items["test"])):
        key = str(path.relative_to(ROOT))
        summary["md5"][key] = write(path, rows_)
        summary["files"][key] = stats(rows_)
    summary["build"] = dict(sorted(st.items()))
    summary["dropped_steps"] = {k: v for k, v in sorted(st.items()) if k.startswith("dropped_")}
    summary["near_duplicates"] = nd
    summary["leaks"] = dict(leak)
    summary["cues_gfpq_view"] = cues
    summary["audit"] = {"failures": len(fails), "examples": fails[:20]}
    summary["fields"] = {"item": ["source strategyqa_tf", "group strategyqa_tf/s<n_steps>", "n_steps = depth", "subqs",
                                  "has_tree", "tree_source", "answer", "term", "n_facts"],
                         "subq": ["id s<j>v / s<j>n", "question", "opposite", "truth", "p_cot", "p_fresh", "type step",
                                  "depends_on", "initial", "status", "probe", "probe_text",
                                  "wrong (No: other / perturb)", "step", "step_text", "resolved", "kind", "value",
                                  "value_text", "unit", "entity_class", "fact", "fact_index", "gold_ancestor",
                                  "node_depth"]}
    summary["notes"] = ("test = a seeded stratified share of the official train set (the official test set has no "
                        "labels); facts are only in node metadata (field fact), never in a prompt or node question")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    # ---- f25: a stratified subset of train (data/student_gsm8k_tf_f25's recipe), val copied
    lines = [ln for ln in (out / "train.jsonl").read_text().split("\n") if ln]
    tr = [json.loads(ln) for ln in lines]
    sub_dir = ROOT / f"{args.out}_f{round(args.sub_frac * 100)}"
    tag = f"{SOURCE}|f{round(args.sub_frac * 100)}|{args.sub_date}"
    keep = subset(lines, tr, args.sub_frac, tag)
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "train.jsonl").write_text("".join(lines[i] + "\n" for i in keep))
    (sub_dir / "val.jsonl").write_bytes((out / "val.jsonl").read_bytes())
    sub = {"source": args.out, "fraction": args.sub_frac, "stratified_by": "group (n_steps)", "seed": tag,
           "n_train": len(keep), "train_md5": md5(sub_dir / "train.jsonl"), "val_md5": md5(sub_dir / "val.jsonl"),
           "train_stats": stats([tr[i] for i in keep])}
    (sub_dir / "summary.json").write_text(json.dumps(sub, indent=1))
    summary["md5"][str((sub_dir / "train.jsonl").relative_to(ROOT))] = sub["train_md5"]
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("files", "dropped_steps", "leaks", "cues_gfpq_view", "audit", "md5")},
                     indent=1))
    print("build", json.dumps({k: v for k, v in summary["build"].items() if not k.startswith("dropped_")}))
    if fails:
        print("AUDIT FAILED", fails[:10])
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out", default="data/student_strategyqa_tf")
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--sub-frac", type=float, default=0.25)
    ap.add_argument("--sub-date", default="20261014", help="date in the subset's seed string (gsm8k_tf: 20261012)")
    ap.add_argument("--seed", type=int, default=0)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
