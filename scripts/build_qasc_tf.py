"""Tree Forcing data for a knowledge-composition domain: QASC (Khot et al. 2020, allenai/qasc on Hugging Face, revision
REVISION, CC BY 4.0), whose questions each come with the two human-chosen facts that answer them (fact1, fact2) and a
combined fact that composes them, so the trees come from the annotation and no new labelling is needed.

Finals: the 8-option question as released (labels A-H in QASC's own order, gold = answerKey, teacher one-hot):
"Question: <question>\nOptions:\nA) ...\n...\nH) ...". The context is the question only: the facts are never in the
prompt. Item fields are those of data/student_gsm8k_tf (source "qasc_tf", group "qasc_tf/n<pairs>" = the number of fact
pairs kept, n_steps = depth = 2, subqs, has_tree, tree_source "qasc_facts_tf", tree_fix True, ...) plus "qasc" (split,
id, fact1, fact2, combinedfact: the raw annotation, for audits; never read by training).

Nodes (yes/no, exact truth by construction; p_cot = truth; node_depth 0, depends_on []): per fact k in (1, 2)
  "f<k>v"  "Is it true that <fact k>?" (Yes; the fact's text, whitespace collapsed, end punctuation dropped, a common
           first word lower-cased);
  "f<k>n"  the same with one noun phrase swapped (No; no tagger, no LLM). Phrases: QASC's option texts (used at least
           MIN_PHRASE_FREQ times, 1-3 words, content words at both edges) and frequent single fact words; a phrase is a
           noun phrase when its uses in QASC train say so (rarely after a verb slot "to / can / it ...", a predicate
           slot "is / become / very ..." or before another content word; see noun_phrases), and a span is swappable in
           a noun slot (head_slot: not a modifier, not before a determiner). A maximal such span of the fact is replaced
           by one of the same length and grammatical number (final -s) from another item's fact (one occurrence drawn
           uniformly, so replacements follow the facts' own phrase frequencies), with a / an kept right. Neither span
           may share a content word (plural-folded) with the question, any option of the item, the other span or the
           rest of the fact, and the replacement occurs in none of the item's texts: twins have the same overlap with
           the question and the same option and gold mentions (also checked phrase by phrase). Falsity filters: the
           result is no known QASC fact (fact1 / fact2 / combined), and the replacement co-occurs with min(2, n) of the
           fact's other content words in no known fact (a likely true swap). The draw comes from a generator keyed by
           the seed and the fact's text, so items sharing a fact (QASC reuses fact1 across about 10 questions) mostly
           share its No twin and No sentences repeat about as often as Yes ones. No negation templates ("not" would be
           a cue). Residual noise: some swaps are still true or odd (a hand read of 50: about 2-4 arguably true).
  type "fact", or "fact_answer" when the Yes fact contains the correct option's text (phrase match after
  normalisation; both twins carry the pair's type, so --tree-node-types fact masks those pairs whole and every item
  stays 50% Yes); answer_in_node per node.
Not built (counted in summary.json):
  - the combined fact, in every item: it contains the correct option's text in 87% of train items and paraphrases it
    in the rest, i.e. it is the final question in disguise (the gold step of build_gsm8k_tf); without it the model
    must compose fact1 and fact2 for the final;
  - a fact that contains the correct option's text AND covers at least NEAR_FINAL of the question's content words (it
    restates question + answer on its own, e.g. "Climate is the average weather in an area over a long period of
    time." for "What is the average weather in a place over time?"; dropped_near_final);
  - a fact with no No twin under the filters (dropped_no_twin), so every item and fact keeps Yes = No.
Items dropped: the correct option's text is in the question (trivial final), the gold option equals another option
after normalisation (ambiguous), an empty option, and (train pool) a question seen before (dedup below).

Splits:
  data/student_qasc_tf/train.jsonl   QASC train minus val
  data/student_qasc_tf/val.jsonl     about --val-frac of the train pool as whole fact1 clusters (seeded): QASC's dev is
                                     fact1-disjoint from train, so a per-item val would share fact1 with train and read
                                     high
  data/eval/qasc_tf_test.jsonl       QASC dev, every item (the official test has no labels); with nodes, built from dev
                                     facts only, for --eval-subq
  data/student_qasc_tf_f25, _f10     seeded 25% / 10% train subsets stratified by group (the recipe of
                                     data/student_gsm8k_tf_f25: one random.Random("qasc_tf|f25|<date>"), groups sorted,
                                     each group shuffled in file order, round(frac x n) kept, file order), full val
Dedup: by normalised question text: train items whose question is a dev question, or a question seen earlier in train,
are dropped; the dev file keeps all its items (its own repeats are counted).

Reports (summary.json): drop counts, node stats, Yes rate, leak checks, single-feature cue stumps for the node targets
in the GFPQ view (question + node), overlap of dev facts with train facts, the audit (every node re-checked from its own
fields and the raw facts).

  python3 scripts/build_qasc_tf.py           # pandas (system python3); CPU, < 1 min, < 0.5 GB
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.treeplan import QASC_TF, problem_no_options  # noqa: E402

SOURCE = QASC_TF
TREE_SOURCE = "qasc_facts_tf"
REVISION = "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070"
LICENCE = "CC BY 4.0 (allenai/qasc)"
LABELS = list("ABCDEFGH")
FRAME = "Is it true that {}?"
OPPOSITE = "Is it false that {}?"
NEAR_FINAL = 0.5            # share of the question's content words a gold-bearing fact may cover before it is dropped
MAX_SPAN = 3
MIN_PHRASE_FREQ = 2
MIN_WORD_FREQ = 3           # a fact word joins the phrase vocabulary from this many uses in QASC train facts
TRIES = 300                 # replacement draws per candidate span
WTOK = re.compile(r"[a-z0-9]+")
TOK = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
STOP = frozenset("""a about above after again against all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each either else every few for from further had has
have having he her here hers him his how i if in into is it its itself just let me more most much must my no nor not now
of off on once only or other our out over own per same she should so some such than that the their theirs them then
there these they this those through to too under until up upon us very was we were what when where whether which while
who whom whose why will with within without would yet you your although though however therefore thus whereas""".split())


# ------------------------------------------------------------------------------------------------------ text
def clean(s: str) -> str:
    return " ".join(str(s).split())


def norm(s: str) -> str:
    return " ".join(WTOK.findall(str(s).lower().replace("’", "'")))


def has(text: str, phrase: str) -> bool:
    """Whether the phrase occurs in the text as whole words after normalisation (lower case, alphanumeric words)."""
    p = norm(phrase)
    return bool(p) and f" {p} " in f" {norm(text)} "


def stem(w: str) -> str:
    """Plural folding only (plants -> plant, species -> specie is harmless: both sides fold alike)."""
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 4 and (w[-3] in "sxz" or w.endswith(("ches", "shes"))):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        return w[:-1]
    return w


def content(s: str) -> set:
    return {stem(w) for w in WTOK.findall(str(s).lower().replace("’", "'")) if w not in STOP}


def plural(key: str) -> bool:
    w = key.split()[-1]
    return w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3


def statement(fact: str) -> str:
    """A fact as a clause: whitespace collapsed, end punctuation and stray spaces dropped."""
    return clean(fact).rstrip(" .!?;:,")


def lower_first(s: str, lower: set) -> str:
    """Lower-case the first letter when the first word is Title-case and the corpus uses it in lower case."""
    m = TOK.match(s)
    if not m:
        return s
    w = m.group()
    if w[0].isupper() and w[1:] == w[1:].lower() and w.lower() in lower:
        return s[0].lower() + s[1:]
    return s


def ask(stmt: str, lower: set) -> str:
    return FRAME.format(lower_first(stmt, lower))


def opposite(stmt: str, lower: set) -> str:
    return OPPOSITE.format(lower_first(stmt, lower))


# ------------------------------------------------------------------------------------------------------ phrases
# No tagger: a phrase counts as a noun phrase from its own uses in QASC train (facts, combined facts, questions). Its
# left neighbour is a verb slot (VERB_LEFT: "to evaporate", "can reproduce"), a predicate slot (PRED_LEFT: "is heated",
# "become ill") or neither, and its right neighbour a content word (a modifier: "genetic material", "cell division") or
# not. NOUN_MAX bounds the shares; words ending in -ly / -ed / -ing also need NOUN_LY_ED of uses after a determiner,
# preposition or the sentence start.
DET_PREP = frozenset("""the a an this that these those their its his her our your my some many most all any each every
no several few both such other of in on for by with from into about through during without between among under over
like as than at against within onto upon""".split())
VERB_LEFT = frozenset("to can will may might must should would could do does did not cannot often usually also then they "
                      "we you i it he she which who".split())
DET_RIGHT = frozenset("the a an its their his her our your my this these those".split())   # "holds the", "moves its"
PRED_LEFT = frozenset("""very more less so too is are was were be been being become becomes becoming became get gets
getting got remain remains stay stays feel feels seem seems look looks keep keeps""".split())
NOUN_MAX = {"verb_left": 0.15, "pred_left": 0.2, "modifier": 0.6}
NOUN_LY_ED = 0.4
NOUN_MIN_USES = 3
FACT_WORD_NP_LEFT = 0.3     # a fact word (not an option text) also needs this share of uses after a determiner / ...
FACT_WORD_SKIP = ("ing", "est", "ible", "able", "ous", "ful", "less", "ic", "ive")   # gerunds and adjective suffixes


def spans(stmt: str, vocab: set, maximal: bool = True) -> list[tuple]:
    """(start, end, surface, key, n_words, left word, right word) of the spans of 1..MAX_SPAN space-separated words
    whose normalised text is in vocab; maximal: only those not strictly inside a longer one. left / right: the
    neighbouring word in lower case, "<s>" / "</s>" at the sentence edge or across punctuation."""
    toks = list(TOK.finditer(stmt))
    out = []
    for i in range(len(toks)):
        for n in range(1, MAX_SPAN + 1):
            j = i + n
            if j > len(toks) or (n > 1 and stmt[toks[j - 2].end():toks[j - 1].start()] != " "):
                break
            surf = stmt[toks[i].start():toks[j - 1].end()]
            key = norm(surf)
            if key in vocab and len(key.split()) == n:
                left = ("<s>" if i == 0 or stmt[toks[i - 1].end():toks[i].start()] != " "
                        else toks[i - 1].group().lower())
                right = ("</s>" if j == len(toks) or stmt[toks[j - 1].end():toks[j].start()] != " "
                         else toks[j].group().lower())
                out.append((toks[i].start(), toks[j - 1].end(), surf, key, n, left, right))
    if maximal:
        out = [s for s in out if not any(o[0] <= s[0] and s[1] <= o[1] and o[1] - o[0] > s[1] - s[0] for o in out)]
    return out


def phrase_vocab(option_lists, facts) -> tuple[set, set]:
    """The phrase vocabulary and its fact words. Phrases: normalised option texts used at least MIN_PHRASE_FREQ times,
    1..MAX_SPAN words, first and last word a content word (no article, preposition or conjunction at an edge), no
    number, no stop word inside but and / or / of, no verb / predicate word (get shorter), no gerund first (reducing
    heat energy); fact words: single words of the facts used at least MIN_WORD_FREQ times, alphabetic, 3+ letters, not a
    stop word, not ending in FACT_WORD_SKIP, not already a phrase."""
    c = collections.Counter(norm(o) for opts in option_lists for o in opts)
    phrases = {k for k, n in c.items() if n >= MIN_PHRASE_FREQ and 1 <= len(k.split()) <= MAX_SPAN
               and k.split()[0] not in STOP and k.split()[-1] not in STOP and not any(w.isdigit() for w in k.split())
               and all(w in ("and", "or", "of") or w not in STOP | VERB_LEFT | PRED_LEFT for w in k.split())
               and not (len(k.split()) > 1 and k.split()[0].endswith("ing"))}
    u = collections.Counter(w for f in facts for w in WTOK.findall(statement(f).lower()))
    words = {w for w, n in u.items() if n >= MIN_WORD_FREQ and w.isalpha() and len(w) >= 3 and w not in STOP
             and not w.endswith(FACT_WORD_SKIP)} - phrases
    return phrases | words, words


def noun_phrases(rows: list[dict], vocab: set, fact_words: set) -> set:
    """The phrases of vocab used as nouns in the given rows (see the comment above); fact words also need
    FACT_WORD_NP_LEFT of their uses after a determiner, a preposition or the sentence start."""
    ev = collections.defaultdict(collections.Counter)
    for r in rows:
        for t in (r["fact1"], r["fact2"], r["combinedfact"], r["question"]):
            for *_, key, _n, left, right in spans(statement(t), vocab, maximal=False):
                e = ev[key]
                e["n"] += 1
                e["np_left"] += left in DET_PREP or left == "<s>"
                e["verb_left"] += left in VERB_LEFT
                e["pred_left"] += left in PRED_LEFT
                e["modifier"] += right not in STOP and right != "</s>"
    out = set()
    for key, e in ev.items():
        n = e["n"]
        if n < NOUN_MIN_USES or any(e[k] / n > v for k, v in NOUN_MAX.items()):
            continue
        if key.split()[-1].endswith(("ly", "ed", "ing")) and e["np_left"] / n < NOUN_LY_ED:
            continue
        if key in fact_words and e["np_left"] / n < FACT_WORD_NP_LEFT:
            continue
        out.add(key)
    return out


def head_slot(left: str, right: str, nouns: set) -> bool:
    """A span in a noun slot of its sentence: not after a verb / predicate word or a pronoun, not before another noun
    (the head of its phrase, not a modifier as "air" in "air pollution") nor before a determiner (a verb: "head holds
    the reproductive organs")."""
    return left not in VERB_LEFT and left not in PRED_LEFT and right not in nouns and right not in DET_RIGHT


class Corpus:
    """What the No twins are drawn from and checked against: the replacement pool (noun phrases in a noun slot of other
    items' facts), the known facts (every fact1 / fact2 / combined fact of the given rows, with their content words and
    phrases) and the noun phrases (noun_phrases of QASC train)."""

    def __init__(self, pool_rows: list[dict], known_rows: list[dict], vocab: set, nouns: set, lower: set):
        self.vocab, self.nouns, self.lower = vocab, nouns, lower
        self.pool = collections.defaultdict(list)          # (n_words, plural) -> [(row id, surface, key)]
        for r in pool_rows:
            for k in ("fact1", "fact2"):
                st = statement(r[k])
                for s0, _s1, surf, key, n, left, right in spans(st, vocab):
                    if key not in nouns or not head_slot(left, right, nouns):
                        continue
                    w = TOK.match(surf).group()
                    if s0 == 0 and w[0].isupper() and w[1:] == w[1:].lower() and w.lower() in lower:
                        surf = surf[0].lower() + surf[1:]   # sentence-initial capital of a common word
                    self.pool[(n, plural(key))].append((r["id"], surf, key))
        self.known_norm = set()
        self.known_words: list[set] = []
        self.by_phrase = collections.defaultdict(list)     # key -> indices of known facts containing it
        for r in known_rows:
            for k in ("fact1", "fact2", "combinedfact"):
                st = statement(r[k])
                if not st:
                    continue
                self.known_norm.add(norm(st))
                i = len(self.known_words)
                self.known_words.append(content(st))
                for key in {s[3] for s in spans(st, vocab, maximal=False)}:
                    self.by_phrase[key].append(i)
        self.df = collections.Counter(w for ws in self.known_words for w in ws)


def no_twin(stmt: str, rows: list[dict], corpus: Corpus, rng: random.Random, st: collections.Counter) -> dict | None:
    """The No twin of one fact (see the module docstring) under the constraints of every row in rows (the items that
    ask this fact: their questions, options and texts; a twin valid for all of them), or None."""
    q_words = set().union(*(content(r["question"]) for r in rows))
    options = [o for r in rows for o in r["options"]]
    opt_words = set().union(*(content(o) for o in options))
    own = [t for r in rows for t in (r["question"], r["fact1"], r["fact2"], r["combinedfact"], *r["options"])]
    ids = {r["id"] for r in rows}
    cands = [s for s in spans(stmt, corpus.vocab) if s[3] in corpus.nouns and head_slot(s[5], s[6], corpus.nouns)
             and not (content(s[2]) & (q_words | opt_words))]
    if not cands:
        st["no_twin_no_span"] += 1
        return None
    rng.shuffle(cands)
    yes_q = ask(stmt, corpus.lower)
    for s0, s1, surf, key, n, _left, _right in cands:
        s_words = content(surf)
        rest = content(stmt[:s0] + " " + stmt[s1:])
        prev = WTOK.findall(stmt[:s0].lower())[-1:]
        pool = corpus.pool.get((n, plural(key)), [])
        if not pool:
            continue
        for _ in range(TRIES):
            rid, rsurf, rkey = pool[rng.randrange(len(pool))]
            r_words = content(rsurf)
            if (rid in ids or rkey == key or not r_words or r_words & (s_words | q_words | opt_words | rest)
                    or any(has(t, rsurf) for t in own)):
                continue
            if prev == ["a"] and rkey[0] in "aeiou" or prev == ["an"] and rkey[0] not in "aeiou":
                continue
            if s0 == 0:
                rsurf = rsurf[0].upper() + rsurf[1:]
            new = stmt[:s0] + rsurf + stmt[s1:]
            if norm(new) in corpus.known_norm:
                st["twin_reject_known_fact"] += 1
                continue
            need = min(2, len(rest))
            if need and any(len(corpus.known_words[i] & rest) >= need for i in corpus.by_phrase.get(rkey, ())):
                st["twin_reject_cooccurs"] += 1
                continue
            no_q = ask(new, corpus.lower)
            if any(has(yes_q, o) != has(no_q, o) for o in options):
                st["twin_reject_option_mention"] += 1
                continue
            return {"statement": new, "question": no_q, "swap": {"span": surf, "replacement": rsurf, "from": rid,
                                                                  "start": s0}}
    st["no_twin_no_replacement"] += 1
    return None


# ------------------------------------------------------------------------------------------------------ items
def raw_rows(path: Path, split: str) -> list[dict]:
    import pandas as pd
    df = pd.read_parquet(path)
    out = []
    for r in df.itertuples(index=False):
        labs = [str(x) for x in r.choices["label"]]
        out.append({"split": split, "id": str(r.id), "question": clean(r.question),
                    "options": [clean(t) for t in r.choices["text"]], "option_labels": labs,
                    "answerKey": str(r.answerKey), "fact1": str(r.fact1), "fact2": str(r.fact2),
                    "combinedfact": str(r.combinedfact)})
    return out


def gold_text(row_or_item: dict) -> str:
    if "gold_label" in row_or_item:
        return row_or_item["options"][row_or_item["labels"].index(row_or_item["gold_label"])]
    return row_or_item["options"][row_or_item["option_labels"].index(row_or_item["answerKey"])]


def item_problem(row: dict) -> str | None:
    """Why a raw row cannot be a final, else None."""
    if row["option_labels"] != LABELS or row["answerKey"] not in LABELS:
        return "labels_not_A_to_H"
    if any(not o for o in row["options"]) or not row["question"]:
        return "empty_text"
    if "\n" in row["question"] or "Question:" in row["question"] or any("\n" in o for o in row["options"]):
        return "bad_text"
    g = gold_text(row)
    if any(norm(o) == norm(g) for o, L in zip(row["options"], LABELS) if L != row["answerKey"]):
        return "gold_equals_other_option"
    if has(row["question"], g):
        return "gold_in_question"
    if not statement(row["fact1"]) or not statement(row["fact2"]):
        return "empty_fact"
    return None


def coverage(question: str, stmt: str) -> float:
    q = content(question)
    return len(q & content(stmt)) / len(q) if q else 0.0


def near_final(row: dict, stmt: str) -> bool:
    """A fact that contains the correct option's text and covers NEAR_FINAL of the question's content words."""
    return has(stmt, gold_text(row)) and coverage(row["question"], stmt) >= NEAR_FINAL


def shared_twins(rows: list[dict], corpus: Corpus, seed: int, st: collections.Counter) -> dict:
    """The twin cache of a split family, seeded: one No twin per fact text for all the rows that ask it (QASC reuses
    fact1 across about 10 questions), drawn under all their constraints from a generator keyed by the seed and the
    fact; facts with no twin that suits every row start empty. tf_nodes reuses the first cached twin that suits the
    item and caches the items' own draws, so No sentences repeat about as often as Yes ones (no "seen before -> Yes"
    cue). Rows are processed in file order, so the cache is deterministic."""
    by = collections.defaultdict(list)
    for r in rows:
        for k in (1, 2):
            stmt = statement(r[f"fact{k}"])
            if not near_final(r, stmt) and r not in by[stmt]:
                by[stmt].append(r)
    out = {}
    for stmt, rs in by.items():
        tw = no_twin(stmt, rs, corpus, random.Random(f"{seed}/{norm(stmt)}"), st)
        out[stmt] = [tw] if tw else []
    return out


def twin_suits(tw: dict, stmt: str, row: dict, corpus: Corpus) -> bool:
    """Whether a twin drawn for another item sharing this fact passes this item's own checks (the question / option
    words, its texts, the source item, option mentions); the other filters do not depend on the item."""
    sw = tw["swap"]
    q_words = content(row["question"])
    opt_words = set().union(*(content(o) for o in row["options"]))
    own = [row["question"], row["fact1"], row["fact2"], row["combinedfact"], *row["options"]]
    yes_q = ask(stmt, corpus.lower)
    return (sw["from"] != row["id"] and not (content(sw["span"]) | content(sw["replacement"])) & (q_words | opt_words)
            and not any(has(t, sw["replacement"]) for t in own)
            and all(has(yes_q, o) == has(tw["question"], o) for o in row["options"]))


def tf_nodes(row: dict, corpus: Corpus, seed: int, st: collections.Counter, shared: dict | None = None) -> list[dict]:
    gold = gold_text(row)
    nodes = []
    for k in (1, 2):
        stmt = statement(row[f"fact{k}"])
        ans = has(stmt, gold)
        cov = round(coverage(row["question"], stmt), 4)
        st["facts"] += 1
        st["facts_with_gold_text"] += ans
        if near_final(row, stmt):
            st["dropped_near_final"] += 1
            continue
        cache = shared.setdefault(stmt, []) if shared is not None else []
        tw = next((x for x in cache if twin_suits(x, stmt, row, corpus)), None)
        st["twin_reused" if tw else "twin_drawn"] += 1
        if tw is None:                                      # this item's own draw (same key: the fact), then cached
            tw = no_twin(stmt, [row], corpus, random.Random(f"{seed}/{norm(stmt)}"), st)
            if tw is not None:
                cache.append(tw)
        if tw is None:
            st["dropped_no_twin"] += 1
            continue
        meta = {"p_fresh": None, "type": "fact_answer" if ans else "fact", "depends_on": [], "initial": None,
                "status": "direct", "fact": k, "answer_in_fact": ans, "question_coverage": cov,
                "gold_ancestor": True, "flags": [], "node_depth": 0}
        yes_q = ask(stmt, corpus.lower)
        nodes.append({"id": f"f{k}v", "question": yes_q, "opposite": opposite(stmt, corpus.lower), "truth": True,
                      "p_cot": 1.0, **meta, "statement": stmt, "answer_in_node": has(yes_q, gold)})
        nodes.append({"id": f"f{k}n", "question": tw["question"], "opposite": opposite(tw["statement"], corpus.lower),
                      "truth": False, "p_cot": 0.0, **meta, "statement": tw["statement"],
                      "answer_in_node": has(tw["question"], gold), "swap": tw["swap"]})
        st[f"built_{meta['type']}"] += 1
    return nodes


def make_item(row: dict, corpus: Corpus, seed: int, st: collections.Counter, shared: dict | None = None) -> dict:
    iid = f"qasc-{'dev' if row['split'] == 'validation' else row['split']}-{row['id']}"
    opts = "\n".join(f"{L}) {o}" for L, o in zip(LABELS, row["options"]))
    subqs = tf_nodes(row, corpus, seed, st, shared)
    pairs = len(subqs) // 2
    return {"item_id": iid, "source": SOURCE, "group": f"{SOURCE}/n{pairs}", "type": "choice",
            "prompt": f"Question: {row['question']}\nOptions:\n{opts}", "labels": LABELS[:],
            "gold_label": row["answerKey"], "teacher": {L: float(L == row["answerKey"]) for L in LABELS},
            "options": row["options"][:], "option_kinds": ["gold" if L == row["answerKey"] else "distractor"
                                                           for L in LABELS],
            "recipe": "qasc_official", "depth": 2, "random_subqs": [], "rationale": "", "augment": None,
            "has_tree": bool(subqs), "tree_fix": True, "row_kind": "final", "problem_id": iid,
            "node_status": "ok" if subqs else "no_nodes", "licence_class": "PERM", "licence": LICENCE,
            "label_basis": "gold", "node_basis": "annotation", "n_steps": 2, "tree_source": TREE_SOURCE,
            "subqs": subqs, "qasc": {"split": row["split"], "id": row["id"], "fact1": row["fact1"],
                                     "fact2": row["fact2"], "combinedfact": row["combinedfact"]}}


# ------------------------------------------------------------------------------------------------------ audit
def audit_item(item: dict, lower: set | None = None) -> list[str]:
    """Problems of one built item from its own fields and its raw facts (empty: clean)."""
    bad, iid = [], item.get("item_id")
    q = item.get("qasc") or {}
    if (item.get("source") != SOURCE or item.get("tree_fix") is not True or item.get("labels") != LABELS
            or item.get("gold_label") not in LABELS or item.get("n_steps") != 2 or item.get("depth") != 2):
        bad.append("item fields")
    if sum(item["teacher"].values()) != 1.0 or item["teacher"].get(item["gold_label"]) != 1.0:
        bad.append("teacher is not one-hot on the gold")
    try:
        ctx = problem_no_options(item)
        if "\n" in ctx or ctx + "\nOptions:\n" != item["prompt"][:len(ctx) + 10]:
            bad.append("GFPQ context is not the question line")
    except ValueError as e:
        bad.append(f"problem_no_options: {e}")
    gold = gold_text(item)
    if has(question_of(item), gold):
        bad.append("gold text in the question")
    by = {n["id"]: n for n in item["subqs"]}
    if len(by) != len(item["subqs"]):
        bad.append("duplicate node ids")
    if item["group"] != f"{SOURCE}/n{len(by) // 2}" or item["has_tree"] != bool(by):
        bad.append("group / has_tree")
    comb = norm(statement(q.get("combinedfact", "")))
    for n in item["subqs"]:
        i = n["id"]
        if "options" in n or n["depends_on"] or n["node_depth"] != 0 or n["p_cot"] != float(n["truth"]):
            bad.append(f"{i}: node fields")
        if not re.fullmatch(r"f[12][vn]", i) or (i[-1] == "v") != n["truth"]:
            bad.append(f"{i}: id / truth")
        if not n["question"].startswith("Is it true that ") or not n["question"].endswith("?"):
            bad.append(f"{i}: wording")
        if norm(n["statement"]) == comb or norm(n["question"][len("Is it true that "):]) == comb:
            bad.append(f"{i}: the combined fact")
        twin = by.get(i[:-1] + ("n" if n["truth"] else "v"))
        if not twin or twin["truth"] == n["truth"] or twin["type"] != n["type"]:
            bad.append(f"{i}: no Yes/No twin")
            continue
        if any(has(n["question"], o) != has(twin["question"], o) for o in item["options"]):
            bad.append(f"{i}: twins differ in an option mention")
        if n["truth"]:
            raw = statement(q.get(f"fact{i[1]}", ""))
            if n["statement"] != raw or norm(n["question"]) != norm(FRAME.format(raw)):
                bad.append(f"{i}: Yes node is not fact{i[1]}")
            if (n["type"] == "fact_answer") != has(raw, gold) or n["answer_in_fact"] != has(raw, gold):
                bad.append(f"{i}: answer flag")
            if has(raw, gold) and coverage(question_of(item), raw) >= NEAR_FINAL:
                bad.append(f"{i}: near-final fact kept")
        else:
            sw = n["swap"]
            y = twin["statement"]
            rep = sw["replacement"]
            exp = y[:sw["start"]] + rep + y[sw["start"] + len(sw["span"]):]
            if y[sw["start"]:sw["start"] + len(sw["span"])] != sw["span"] or n["statement"] != exp:
                bad.append(f"{i}: No node is not its Yes fact with one span swapped")
            if norm(n["statement"]) == norm(y) or content(rep) & content(question_of(item)):
                bad.append(f"{i}: swap")
    yes = sum(n["truth"] for n in item["subqs"])
    if 2 * yes != len(item["subqs"]):
        bad.append("Yes != No")
    return [f"{iid}: {b}" for b in bad]


def question_of(item: dict) -> str:
    """The question text of a built item (its prompt before the options, without the "Question: " label)."""
    return item["prompt"].partition("\nOptions:")[0][len("Question: "):]


# ------------------------------------------------------------------------------------------------------ cues
def stump_bool(rows: list[tuple]) -> float | None:
    xs = [(t, f) for t, f in rows if f is not None]
    if not xs:
        return None
    acc = sum(t == f for t, f in xs) / len(xs)
    return round(max(acc, 1 - acc), 4)


def stump_num(rows: list[tuple]) -> float | None:
    """Best single-threshold accuracy of a numeric feature for the Yes/No target (fit on the same rows)."""
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r[1])
    n, ny = len(rows), sum(t for t, _ in rows)
    best, yes_le = max(ny, n - ny), 0
    for i, (t, x) in enumerate(rows):
        yes_le += t
        if i + 1 < n and rows[i + 1][1] == x:
            continue
        le, gt_yes = i + 1, ny - yes_le
        best = max(best, (le - yes_le) + gt_yes, yes_le + (n - le - gt_yes))
    return round(best / n, 4)


def cues(items: list[dict], corpus: Corpus) -> dict:
    """Single-feature stumps for the node targets in the GFPQ view (question + node question), on train nodes."""
    feats = collections.defaultdict(list)
    words = collections.defaultdict(lambda: [0, 0])
    pair_rarer = pair_longer = pairs = 0
    copies = collections.Counter(norm(n["statement"]) for it in items for n in it["subqs"])
    for it in items:
        qn = content(question_of(it))
        gold = gold_text(it)
        by = {n["id"]: n for n in it["subqs"]}
        for n in it["subqs"]:
            t = n["truth"]
            text = n["question"]
            ws = content(n["statement"])
            feats["contains_gold"].append((t, has(text, gold)))
            feats["contains_an_option"].append((t, any(has(text, o) for o in it["options"])))
            feats["negation_word"].append((t, bool(re.search(r"\b(?:not|no|never|cannot)\b|n't", text, re.I))))
            feats["question_overlap"].append((t, len(ws & qn)))
            feats["words"].append((t, len(text.split())))
            feats["text_in_2plus_items"].append((t, copies[norm(n["statement"])] >= 2))
            feats["min_fact_df"].append((t, min((corpus.df.get(w, 0) for w in ws), default=0)))
            feats["mean_fact_df"].append((t, sum(corpus.df.get(w, 0) for w in ws) / max(1, len(ws))))
            for w in ws:
                words[w][t] += 1
            if t and n["id"][:-1] + "n" in by:
                m = by[n["id"][:-1] + "n"]
                pairs += 1
                dfy = min((corpus.df.get(w, 0) for w in ws), default=0)
                dfn = min((corpus.df.get(w, 0) for w in content(m["statement"])), default=0)
                pair_rarer += dfn < dfy
                pair_longer += len(m["question"].split()) > len(text.split())
    out = {"n": len(feats["words"])}
    for k in ("contains_gold", "contains_an_option", "negation_word", "text_in_2plus_items"):
        out[k] = stump_bool(feats[k])
    for k in ("question_overlap", "words", "min_fact_df", "mean_fact_df"):
        out[k] = stump_num(feats[k])
    out["pairs_no_rarer_word"] = round(pair_rarer / max(1, pairs), 4)
    out["pairs_no_longer"] = round(pair_longer / max(1, pairs), 4)
    skew = sorted(((abs(y / (y + n_) - 0.5), w, y + n_, round(y / (y + n_), 3)) for w, (n_, y) in words.items()
                   if y + n_ >= 100), reverse=True)[:5]
    out["word_yes_rate_most_skewed_ge100"] = [{"word": w, "nodes": c, "yes_rate": r} for _, w, c, r in skew]
    return out


# ------------------------------------------------------------------------------------------------------ io
def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def write(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return md5(path)


def rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def stats(items: list[dict]) -> dict:
    nodes = [n for it in items for n in it["subqs"]]
    return {"items": len(items), "with_tree": sum(it["has_tree"] for it in items),
            "groups": dict(sorted(collections.Counter(it["group"] for it in items).items())),
            "nodes": len(nodes), "nodes_per_item": round(len(nodes) / max(1, len(items)), 3),
            "yes_share": round(sum(n["truth"] for n in nodes) / max(1, len(nodes)), 4),
            "node_types": dict(sorted(collections.Counter(n["type"] for n in nodes).items())),
            "node_depth": dict(sorted(collections.Counter(n["node_depth"] for n in nodes).items())),
            "answer_in_node": {"yes": sum(n["answer_in_node"] for n in nodes if n["truth"]),
                               "no": sum(n["answer_in_node"] for n in nodes if not n["truth"])},
            "gold_position": dict(sorted(collections.Counter(it["gold_label"] for it in items).items())),
            "question_words_mean": round(sum(len(question_of(it).split()) for it in items) / max(1, len(items)), 2),
            "node_words_mean": round(sum(len(n["question"].split()) for n in nodes) / max(1, len(nodes)), 2)}


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def subset(train: list[dict], frac: float, seed: str) -> list[dict]:
    """The recipe of data/student_gsm8k_tf_f25 / _f10 / _f50 (reproduced from those files): one random.Random(seed);
    for each group in sorted order, its items in file order shuffled by that generator, the first round(frac x n)
    kept; the subset in file order."""
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for it in train:
        by[it["group"]].append(it["item_id"])
    keep = set()
    for g in sorted(by):
        ids = list(by[g])
        rng.shuffle(ids)
        keep |= set(ids[:round(frac * len(ids))])
    return [it for it in train if it["item_id"] in keep]


# ------------------------------------------------------------------------------------------------------ build
def build(args) -> None:
    raw = ROOT / args.raw / "data"
    files = {sp: raw / f"{sp}-00000-of-00001.parquet" for sp in ("train", "validation", "test")}
    rows = {sp: raw_rows(files[sp], sp) for sp in ("train", "validation")}
    assert not {r["id"] for r in rows["train"]} & {r["id"] for r in rows["validation"]}, "QASC ids shared by splits"
    print("raw", {k: len(v) for k, v in rows.items()}, f"rss {rss_mb()} MB", flush=True)
    vocab, fact_words = phrase_vocab((r["options"] for r in rows["train"]),
                                     [r[k] for r in rows["train"] for k in ("fact1", "fact2")])
    nouns = noun_phrases(rows["train"], vocab, fact_words)
    lower = set()
    for r in rows["train"] + rows["validation"]:
        for t in (r["question"], r["fact1"], r["fact2"], r["combinedfact"], *r["options"]):
            lower |= {w for w in TOK.findall(t) if w.islower()}
    c_train = Corpus(rows["train"], rows["train"], vocab, nouns, lower)                 # train / val nodes
    c_test = Corpus(rows["validation"], rows["train"] + rows["validation"], vocab, nouns, lower)   # dev: dev facts
    print("corpus", {"vocab": len(vocab), "fact_words": len(fact_words), "nouns": len(nouns), "pool_train": sum(map(len, c_train.pool.values())),
                     "pool_dev": sum(map(len, c_test.pool.values())), "known_train": len(c_train.known_words)},
          f"rss {rss_mb()} MB", flush=True)
    st = {"train": collections.Counter(), "test": collections.Counter()}
    raw_facts = {}
    for sp, rs in rows.items():
        c = collections.Counter()
        for r in rs:
            g = gold_text(r)
            c["n"] += 1
            c["gold_in_combined"] += has(r["combinedfact"], g)
            c["gold_in_fact1"] += has(r["fact1"], g)
            c["gold_in_fact2"] += has(r["fact2"], g)
            c["gold_in_fact1_or_fact2"] += has(r["fact1"], g) or has(r["fact2"], g)
        raw_facts[sp] = dict(c)
    # ---- test: every dev item; train pool: item filters, dedup by question text (vs dev, then first in train)
    test_rows = []
    for r in rows["validation"]:
        why = item_problem(r)
        if why:
            st["test"][f"dropped_item_{why}"] += 1
            continue
        test_rows.append(r)
    dev_q = {norm(r["question"]) for r in rows["validation"]}
    seen, pool_rows = set(), []
    for r in rows["train"]:
        why = item_problem(r)
        if why:
            st["train"][f"dropped_item_{why}"] += 1
            continue
        k = norm(r["question"])
        if k in dev_q:
            st["train"]["dropped_item_question_in_dev"] += 1
            continue
        if k in seen:
            st["train"]["dropped_item_repeated_question"] += 1
            continue
        seen.add(k)
        pool_rows.append(r)
    sh_st = {"train": collections.Counter(), "test": collections.Counter()}
    sh_train = shared_twins(pool_rows, c_train, args.seed, sh_st["train"])
    sh_test = shared_twins(test_rows, c_test, args.seed, sh_st["test"])
    test = [make_item(r, c_test, args.seed, st["test"], sh_test) for r in test_rows]
    st["test"]["repeated_question_kept"] = len(test) - len({norm(question_of(it)) for it in test})
    pool = [make_item(r, c_train, args.seed, st["train"], sh_train) for r in pool_rows]
    print("built", {"pool": len(pool), "test": len(test)}, f"rss {rss_mb()} MB", flush=True)
    # ---- val: whole fact1 clusters, seeded, about val_frac of the pool
    clusters = collections.defaultdict(list)
    for it in pool:
        clusters[norm(statement(it["qasc"]["fact1"]))].append(it["item_id"])
    keys = sorted(clusters)
    random.Random(f"{args.seed}-val").shuffle(keys)
    want, val_ids, val_clusters = round(args.val_frac * len(pool)), set(), 0
    for k in keys:
        if len(val_ids) >= want:
            break
        val_ids |= set(clusters[k])
        val_clusters += 1
    val = [it for it in pool if it["item_id"] in val_ids]
    train = [it for it in pool if it["item_id"] not in val_ids]
    tr_f1 = {norm(statement(it["qasc"]["fact1"])) for it in train}
    tr_facts = tr_f1 | {norm(statement(it["qasc"]["fact2"])) for it in train}
    overlap = {"val_fact1_in_train_fact1": sum(norm(statement(it["qasc"]["fact1"])) in tr_f1 for it in val),
               "val_fact2_in_train_facts": sum(norm(statement(it["qasc"]["fact2"])) in tr_facts for it in val),
               "test_fact1_in_train_facts": sum(norm(statement(it["qasc"]["fact1"])) in tr_facts for it in test),
               "test_fact2_in_train_facts": sum(norm(statement(it["qasc"]["fact2"])) in tr_facts for it in test),
               "test_any_fact_in_train_facts": sum(norm(statement(it["qasc"]["fact1"])) in tr_facts
                                                   or norm(statement(it["qasc"]["fact2"])) in tr_facts for it in test),
               "test_yes_node_statement_in_train_yes_nodes": None}
    tr_yes = {norm(n["statement"]) for it in train for n in it["subqs"] if n["truth"]}
    overlap["test_yes_node_statement_in_train_yes_nodes"] = sum(norm(n["statement"]) in tr_yes for it in test
                                                                for n in it["subqs"] if n["truth"])
    q_tr = {norm(question_of(it)) for it in train}
    assert not q_tr & {norm(question_of(it)) for it in val} and not q_tr & {norm(question_of(it)) for it in test}
    assert not {it["item_id"] for it in train} & {it["item_id"] for it in val}
    # ---- audit and leaks
    fails = []
    for it in train + val + test:
        fails += audit_item(it)
    from cotdistill.teacher import leaks_answer
    leak = collections.Counter()
    for it in train + val + test:
        comb = norm(statement(it["qasc"]["combinedfact"]))
        for n in it["subqs"]:
            leak["nodes"] += 1
            leak["yes_nodes_with_gold_text"] += n["truth"] and n["answer_in_node"]
            leak["no_nodes_with_gold_text"] += (not n["truth"]) and n["answer_in_node"]
            leak["nodes_with_a_distractor_text"] += any(has(n["question"], o) for o, L in zip(it["options"], LABELS)
                                                        if L != it["gold_label"])
            leak["node_is_combined_fact"] += norm(n["statement"]) == comb
            leak["near_final_kept"] += n["answer_in_fact"] and n["question_coverage"] >= NEAR_FINAL
            leak["teacher_leaks_answer"] += leaks_answer(n["question"], it)
    # ---- write
    out, edir = ROOT / args.out, ROOT / args.eval_dir
    summary = {"script": "scripts/build_qasc_tf.py", "args": vars(args),
               "raw": {"repo": "allenai/qasc", "revision": REVISION, "licence": LICENCE,
                       "md5": {p.name: md5(p) for p in files.values()}, "rows": {k: len(v) for k, v in rows.items()},
                       "answer_text_in_raw_facts": raw_facts},
               "md5": {}, "files": {}}
    for path, rows_ in ((out / "train.jsonl", train), (out / "val.jsonl", val), (edir / "qasc_tf_test.jsonl", test)):
        key = f"{path.parent.name}/{path.name}"
        summary["md5"][key] = write(path, rows_)
        summary["files"][key] = stats(rows_)
    summary["val_split"] = {"by": "whole fact1 clusters", "clusters": val_clusters, "of_clusters": len(clusters),
                            "target_items": want}
    summary["build"] = {k: dict(sorted(v.items())) for k, v in st.items()}
    summary["build"]["shared_twin_draws"] = {k: dict(sorted(v.items())) for k, v in sh_st.items()}
    summary["build"]["twins_per_fact"] = {k: dict(sorted(collections.Counter(len(v) for v in sh.values()).items()))
                                          for k, sh in (("train", sh_train), ("test", sh_test))}
    summary["dedup"] = {"by": "normalised question text", "train_vs_dev": st["train"]["dropped_item_question_in_dev"],
                        "within_train": st["train"]["dropped_item_repeated_question"],
                        "dev_repeats_kept": st["test"]["repeated_question_kept"]}
    summary["fact_overlap"] = overlap
    summary["leaks"] = dict(leak)
    summary["cues_gfpq_view"] = cues(train, c_train)
    summary["audit"] = {"failures": len(fails), "examples": fails[:20]}
    summary["subsets"] = {}
    for frac, name in ((0.25, "f25"), (0.1, "f10")):
        d = ROOT / f"{args.out}_{name}"
        seed = f"{SOURCE}|{name}|{args.subset_date}"
        sub = subset(train, frac, seed)
        sm = {"source": args.out, "fraction": frac, "stratified_by": "group (fact pairs kept)", "seed": seed,
              "n_train": len(sub), "train_md5": write(d / "train.jsonl", sub)}
        (d / "val.jsonl").write_bytes((out / "val.jsonl").read_bytes())
        (d / "summary.json").write_text(json.dumps(sm, indent=1))
        summary["subsets"][d.name] = sm
    summary["fields"] = {"item": ["source qasc_tf", "group qasc_tf/n<pairs>", "n_steps 2", "depth 2", "subqs",
                                  "has_tree", "tree_source qasc_facts_tf", "qasc (raw annotation)"],
                         "subq": ["id f<k>v / f<k>n", "question", "opposite", "truth", "p_cot", "p_fresh", "type fact / "
                                  "fact_answer", "depends_on []", "initial", "status", "fact", "answer_in_fact",
                                  "question_coverage", "gold_ancestor", "flags", "node_depth 0", "statement",
                                  "answer_in_node", "swap (No: span, replacement, from, start)"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("md5", "files", "build", "dedup", "fact_overlap", "leaks",
                                              "cues_gfpq_view", "audit", "subsets")}, indent=1))
    print(f"rss {rss_mb()} MB")
    if fails:
        print("AUDIT FAILED", fails[:10])
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/qasc")
    ap.add_argument("--out", default="data/student_qasc_tf")
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset-date", default="20261014", help="the date in the f25 / f10 subset seeds")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
