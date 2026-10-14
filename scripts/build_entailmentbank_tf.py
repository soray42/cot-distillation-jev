"""Tree Forcing data for science question answering: EntailmentBank v3 (Dalvi et al. 2021, allenai/entailment_bank),
ARC grade-school science multiple-choice questions with human-written entailment trees. The trees are the annotation;
no new labelling, no LLM.

Raw data (read as data only; run this script with `python3 -P -E`, the safe path of -I with the user site kept for
pandas): data/raw/entailmentbank/entailment_trees_emnlp2021_data_v3.zip, extracted beside it (allenai.org/data/
entailmentbank -> Google Drive folder v3_May6_2022; md5 and sha256 in summary.json), task_1 (the gold leaves of each
tree, no distractor sentences), and the WorldTree corpus of supporting_data (a known-true filter). EntailmentBank has
the question and the correct answer text but not the other options: the options and the answer key come from ARC
(data/raw/arc, the allenai/ai2_arc parquet files). EntailmentBank train and dev questions are ARC *train* questions,
its test questions ARC *validation* questions (no ARC test question is used).

Finals: the ARC question with its 4 options (labels A-D, the options in a seeded order; questions with 3 or 5 options
are dropped), the answer by ARC's answer key (keys "1".."4" read by position), "Question: <question>\\nOptions:\\nA)
..." as data/student_v3's arc items. The context is the question only: the tree's leaf facts are not given (the model
must know or infer them), as in ARC itself. Fields as data/student_gsm8k_tf: source "entailmentbank_tf", group
"entailmentbank_tf/d<depth>", depth = the proof depth and n_steps = the inference steps, both on the proof edges the
root reaches (EntailmentBank's own depth_of_proof / length_of_proof kept as eb_*), teacher = one-hot gold, has_tree,
tree_source, tree_fix; plus arc_set (challenge / easy), eb_split, eb_id.

Nodes (yes/no claims about the tree's sentences, asked in the problem's context: "Is it true that <sentence>?"; exact
truth by construction; p_cot = truth): for every sentence the tree uses (its leaf facts sent<k>, type "fact", and its
intermediate conclusions int<k>, type "inference") that survives the drops below, a Yes node "<sid>v" (the sentence as
written, spacing before punctuation removed) and a No twin "<sid>n", a minimal false variant of it, so every item is
exactly 50% Yes. The No variant is drawn among the edits available for the sentence (antonym, swap and reverse with
equal odds among those available, else a negation; none: the pair is dropped; field wrong, the edit in edit):
  - antonym (lexical negation): one word replaced by its antonym from a fixed list (increases / decreases, more / less,
    absorbs / reflects, conductor / insulator, physical / chemical, north / south, ...; the article a / an fixed); not
    when the sentence already has the antonym's stem (no "night and night"), and in a taxonomy sentence ("X is a kind
    of Y") only in Y ("a herbivore is a kind of animal" -> "a carnivore ..." would stay true);
  - swap: the object of a typed relation (is a kind / type of, is a part of, is a property of, is a source of, is a
    measure of, is used for / to, is made of, is located in, is formed by, is when, consists of, means, and "is a / an"
    with an object of at most 3 words) replaced by the object of the same relation in a sentence of another *train*
    tree (never a val / test sentence), with a length within 2 words; not in a hedged sentence (some, can, may, often,
    usually, ...); the new object shares no content stem with the old object, the item's question, options, hypothesis
    or tree sentences (no option is eliminated by a No node, no on-topic accidental truth), no rare stem (in at most
    300 known-true sentences) of it co-occurs with one of the subject in a known-true sentence, it is not a known object
    of the subject under that relation, nor (taxonomy) a known ancestor of it in the is-a-kind-of closure of
    EntailmentBank + WorldTree. Generic verb relations (causes, contains, requires, uses, has, ...) are not swapped:
    a swapped object ("predators eat to survive") or subject ("fungi have the ability to reproduce") came out true or
    vague in 1 in 8 to 1 in 5 of the variants checked by hand; they are negated;
  - reverse: the arguments of an asymmetric relation exchanged ("predators eat prey" -> "prey eats predators", "the
    moon orbits the earth" -> "the earth orbits the moon"; causes, contains, requires, needs, eats, produces, orbits,
    carries, transports, emits, releases, uses, provides, makes, prevents, protects, is a part of), both arguments
    short noun phrases (at most 4 words, no function word but an article or possessive), the verb agreeing, not in a
    hedged sentence;
  - negate: a single negation removed ("cannot" -> "can", "is not" -> "is", "do not X" -> "X", "does not X" -> "Xs"),
    or one inserted in the main clause: after its first copula / modal ("can" -> "cannot"), or at its verb ("causes"
    -> "does not cause", "have" -> "do not have"); never in sentences with a hedge or quantifier (some, may, often,
    usually, always, all, ...) or that open with a subordinate clause.
  A variant is never a known-true sentence (every EntailmentBank sentence of any split and the WorldTree corpus, its
  "a / b" alternatives expanded), adds no stem of any option, and passes the same gold-leak checks as a Yes node.
  depends_on: the Yes nodes of the step's premises (the tree edges) that were built; only Yes nodes are parents, both
  twins share them. node_depth: the sentence's height in the full tree (leaf 0, conclusion 1 + its deepest premise),
  computed before any drop.
Not built (counted in summary.json build):
  - the root (the hypothesis: the question and its answer as one statement);
  - gold leak (gold_leak): a sentence containing the correct option's text (normalized word sequence), or, after light
    stemming, at least 60% of the gold option's content words ("venus reflects more sunlight toward earth than other
    planets" for "It reflects the most sunlight toward Earth."), or at least half of them and half more than of every
    distractor (it singles out the gold: "a decomposer is a kind of organism that breaks down dead organisms" for
    "breaks down dead plants and animals"), or one teacher.leaks_answer flags (the question restated: "fossil fuels
    are formed from the remains of organisms" for "Fossil fuels were formed from" / "the remains of living things");
    a sentence equal to the hypothesis, and (train / val) a sentence equal to the hypothesis of a later split's item
    (val / test);
  - a sentence whose text repeats an earlier one of the tree; a sentence with no false variant (the pair).
Every node is re-checked from its own fields (audit_item): twins, wording, truth, parents among the premises and Yes,
depth, no gold leak in any claim.

Splits: EntailmentBank train -> data/student_entailmentbank_tf/train.jsonl, dev -> val.jsonl, test ->
data/eval/entailmentbank_tf_test.jsonl. A question with two trees (EntailmentBank repeats 6 ids) keeps the tree whose
answer is ARC's gold, else the first. Dedup by question text (normalized) across splits (test, then val, win) and by
word 8-grams (as build_gsm8k_tf: a shared 8-gram that is in fewer than 5 questions drops the train / val question).
data/student_entailmentbank_tf_f25: a seeded 25% train subset stratified by group, as data/student_gsm8k_tf_f25 (one
generator seeded "entailmentbank_tf|f25|<date>", groups sorted, each shuffled and cut to round(0.25 n), file order kept;
val copied).

Reports (summary.json): drop counts, node stats, yes rate, the No edits, leak checks, single-feature cue stumps of the
node targets in the GFPQ view (question only) and the GFP view (question and options), the contamination of other
training files by these questions, the raw files' hashes.

  python3 -P -E scripts/build_entailmentbank_tf.py
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import itertools
import json
import random
import re
import resource
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SOURCE = "entailmentbank_tf"
TREE_SOURCE = "entailmentbank_v3_task1"
RAW_DIR = ROOT / "data/raw/entailmentbank"
RAW_ZIP = RAW_DIR / "entailment_trees_emnlp2021_data_v3.zip"
RAW = RAW_DIR / "entailment_trees_emnlp2021_data_v3"
RAW_URL = ("https://drive.google.com/file/d/1kVr-YsUVFisceiIklvpWEe0kHNSIFtNh (allenai.org/data/entailmentbank -> "
           "Google Drive folder v3_May6_2022)")
ARC_DIR = ROOT / "data/raw/arc"
ARC_FILES = ("ARC-Challenge-train", "ARC-Easy-train", "ARC-Challenge-validation", "ARC-Easy-validation")
SPLITS = (("train", "train"), ("dev", "val"), ("test", "test"))     # EntailmentBank split -> ours
LABELS = ["A", "B", "C", "D"]
LICENCE = "EntailmentBank v3 (AI2; licence not checked) on ARC questions (CC BY-SA 4.0)"
GOLD_RECALL_MAX = 0.6        # a sentence holding this share of the gold option's content stems states the answer,
GOLD_POINTER = (0.5, 0.5)    # or this share with this margin over every distractor's share (it singles out the gold)
COOC_DF_MAX = 300            # stems in more known-true sentences than this are too common for the relatedness check
WRONG_W = {"antonym": 1.0, "swap": 1.0, "reverse": 1.0}
SWAP_LEN = 2                 # a swapped-in object has the old one's length +- this many words
YES_Q, NO_Q = "Is it true that {}?", "Is it false that {}?"

WTOK = re.compile(r"[a-z0-9]+")
HTOK = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
STEP = re.compile(r"^\s*(.+?)\s*->\s*(hypothesis|int\d+)\s*(?::\s*(.*?))?\s*$")
STOP = frozenset("""a about above after again against all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each either else every few for from further had has
have having he her here hers him his how i if in into is it its itself just let me more most much must my no nor not
now of off on once only or other our out over own per same she should so some such than that the their theirs them
then there these they this those through to too under until up upon us very was we were what when where whether which
while who whom whose why will with within without would yet you your""".split())


# ------------------------------------------------------------------------------------------------------ text
def clean(s: str) -> str:
    """One line, single spaces, no space before , ; : . ! ? and no final period (EntailmentBank writes " , ")."""
    s = re.sub(r"\s+([,;:.!?])", r"\1", " ".join(str(s).split()))
    return s[:-1] if s.endswith(".") else s


def words(s: str) -> list[str]:
    return WTOK.findall(str(s).lower().replace("°", ""))


def norm(s: str) -> str:
    return " ".join(words(s))


def stem(w: str) -> str:
    """A light suffix stripper (reflects / reflecting / reflected -> reflect, revolves / revolving -> revolv)."""
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    elif len(w) > 4 and w.endswith("es"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


def stems(s: str) -> set:
    """Content stems: words that are not stop words, stemmed."""
    return {stem(w) for w in words(s) if w not in STOP}


def phrase_in(phrase: str, s: str) -> bool:
    """Whether the phrase's word sequence occurs in s at word boundaries (normalized: lower case, punctuation and the
    degree sign dropped)."""
    p = norm(phrase)
    return bool(p) and f" {p} " in f" {norm(s)} "


def recall(option: str, s: str) -> float:
    """Share of an option's content stems that s has (all its stems when it has no content word)."""
    g, have = stems(option), stems(s)
    if not g:
        g, have = {stem(w) for w in words(option)}, {stem(w) for w in words(s)}
    return len(g & have) / len(g) if g else 0.0


def gold_leak(item: dict, s: str) -> str | None:
    """Why sentence s states or singles out the item's answer, else None: the gold option's text ("gold_text"), at
    least GOLD_RECALL_MAX of its content stems ("gold_recall"), or at least GOLD_POINTER[0] of them and GOLD_POINTER[1]
    more than of any distractor's ("gold_pointer"), or the final question in disguise (teacher.leaks_answer: an answer
    request or the question restated, "restates_question")."""
    from cotdistill.teacher import leaks_answer
    gold = gold_text(item)
    if phrase_in(gold, s):
        return "gold_text"
    r = recall(gold, s)
    if r >= GOLD_RECALL_MAX:
        return "gold_recall"
    if r >= GOLD_POINTER[0] and r - max(recall(d, s) for d in item["options"] if d != gold) >= GOLD_POINTER[1]:
        return "gold_pointer"
    if leaks_answer(YES_Q.format(s), item):
        return "restates_question"
    return None


# ------------------------------------------------------------------------------------------------------ known truths
REL_RE = re.compile(r"^(.+?) (?:is|are) (?:a |an )?(?:kind|type)s? of (.+)$")
ARTICLE = re.compile(r"^(?:a|an|the) ")


def np_key(s: str) -> str:
    return ARTICLE.sub("", norm(s))


def head(s: str) -> str:
    w = np_key(s).split()
    return stem(w[-1]) if w else ""


def alternatives(s: str, cap: int = 16) -> list[str]:
    """WorldTree writes alternatives as "a / b / c" of single words: every reading (at most cap)."""
    toks = s.split()
    slots, i = [], 0
    while i < len(toks):
        alt = [toks[i]]
        while i + 2 < len(toks) and toks[i + 1] == "/":
            alt.append(toks[i + 2])
            i += 2
        slots.append(alt)
        i += 1
    return [" ".join(x) for x in itertools.islice(itertools.product(*slots), cap)]


class Known:
    """Known-true sentences (normalized), the is-a-kind-of graph, the (subject, relation, object head) triples and a
    stem -> sentences index of every EntailmentBank sentence of every split and the WorldTree corpus."""

    def __init__(self, sentences):
        self.true, self.kind, self.triples = set(), collections.defaultdict(set), set()
        self.index = collections.defaultdict(set)
        for k, s in enumerate(dict.fromkeys(norm(x) for x in sentences)):
            for x in stems(s):
                self.index[x].add(k)
        for s in sentences:
            for a in alternatives(" ".join(str(s).lower().split())):
                self.true.add(norm(a))
                m = REL_RE.match(norm(a))
                if m:
                    sub, obj = np_key(m.group(1)), np_key(m.group(2))
                    for x in (sub, head(sub)):
                        self.kind[x] |= {obj, head(obj)}
                r = split_rel(clean(a).lower())
                if r:
                    for x in (np_key(r[0]), head(r[0])):
                        self.triples.add((x, REL_CANON[r[1]], head(r[2])))
                        if REL_CANON[r[1]] in TAXONOMIC:
                            self.kind[x] |= {np_key(r[2]), head(r[2])}

    def related(self, a: set, b: set, df_max: int | None = COOC_DF_MAX) -> bool:
        """Whether some known-true sentence holds a stem of a and a stem of b (stems in more than df_max sentences
        ignored; None: every stem): the two are related, so a swap of one next to the other may come out true."""
        keep = lambda xs: [self.index[x] for x in xs if 0 < len(self.index.get(x, ())) <= (df_max or 10 ** 9)]  # noqa
        return any(not ia.isdisjoint(ib) for ia in keep(a) for ib in keep(b))

    def ancestors(self, np: str) -> set:
        seen, todo = set(), [np_key(np), head(np)]
        while todo:
            x = todo.pop()
            for y in self.kind.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    todo.append(y)
        return seen


# ------------------------------------------------------------------------------------------------------ No variants
_ANT_PAIRS = """increase decrease|increases decreases|increased decreased|increasing decreasing|more less|most least|
higher lower|highest lowest|high low|larger smaller|largest smallest|large small|warmer cooler|warm cool|hotter colder|
hot cold|heating cooling|faster slower|fast slow|quickly slowly|longer shorter|positive negative|attract repel|
attracts repels|attracted repelled|absorb reflect|absorbs reflects|absorbed reflected|absorbing reflecting|
conductor insulator|conductors insulators|renewable nonrenewable|inherited learned|gain lose|gains loses|north south|
northern southern|east west|summer winter|strong weak|stronger weaker|wet dry|inside outside|above below|
melting freezing|melt freeze|melts freezes|evaporation condensation|evaporate condense|evaporates condenses|
expand contract|expands contracts|push pull|pushes pulls|dominant recessive|herbivore carnivore|herbivores carnivores|
producer consumer|producers consumers|physical chemical|vertebrate invertebrate|vertebrates invertebrates|
always never|easier harder|safe dangerous|helpful harmful|transparent opaque|sunrise sunset|top bottom|
upward downward|add remove|adds removes|adding removing|added removed|warm-blooded cold-blooded"""
_ANT_ONEWAY = "bigger smaller|big small|nonliving living"
ANT = {}
for _p in _ANT_PAIRS.replace("\n", "").split("|"):
    _a, _b = _p.split()
    ANT[_a], ANT[_b] = _b, _a
for _p in _ANT_ONEWAY.split("|"):
    _a, _b = _p.split()
    ANT[_a] = _b

# Relations whose object a swap replaces, in priority order at the same position (the typed WorldTree relations first,
# then definitions, short "is a / an" predicates and a fixed list of verbs); plural forms map to the singular one.
_REL_TYPED = ("is a kind of", "is a type of", "is a part of", "is a property of", "is a source of", "is a measure of",
              "is used for", "is used to", "is made of", "is located in", "is formed by", "is when",
              "consists of")
_REL_VERBS = ("lives in", "causes", "contains", "requires", "means", "produces", "eats", "uses", "provides", "converts",
              "helps", "makes", "transports", "transfers", "carries", "emits", "releases", "performs", "recycles",
              "prevents", "protects", "needs", "affects", "orbits", "has")
_PLURAL = {"lives in": "live in", "has": "have", "carries": "carry", "consists of": "consist of"}
RELS, REL_CANON = [], {}
for _r in _REL_TYPED + ("is an", "is a") + _REL_VERBS:
    for _f in (_r, _r.replace("is ", "are ", 1) if _r.startswith("is ") else _PLURAL.get(_r, _r[:-1] if _r.endswith("s")
                                                                                          and " " not in _r else _r)):
        if f" {_f} " not in REL_CANON:
            RELS.append(f" {_f} ")
            REL_CANON[f" {_f} "] = f" {_r.replace('type of', 'kind of').replace('is an', 'is a')} "
RELS = tuple(RELS)
VERB_RELS = {f" {r} " for r in _REL_VERBS} - {" means "}   # generic verbs: no swap (a swapped object or subject came
NEG_VERBS = VERB_RELS | {" means "}                        # out true or vague in 1 in 8 to 1 in 5 checked); negated
HEDGE_SWAP = re.compile(r"\b(?:some|sometimes|may|might|could|can|often|usually|many|several|most|likely|probably|"
                        r"generally|mostly|typically)\b")   # a swap in a hedged sentence may stay true
TAXONOMY = " is a kind of "
TAXONOMIC = (" is a kind of ", " is a ")       # the closure check applies; an antonym may not touch the subject
SHORT_PRED = 3                                 # an "is a / an" object has at most this many words
SUBORD = re.compile(r"^(?:if|as|when|whenever|because|since|although|while|after|before|once|unless)\b")
HEDGE = re.compile(r"\b(?:some|sometimes|may|might|could|often|usually|many|several|most|likely|probably|generally|"
                   r"mostly|typically|always|never|only|either|neither|nor|all|every|each|any)\b")
NEG = re.compile(r"\b(?:not|cannot|never|no|none|nothing|neither|nor|without)\b|n't\b")
REL_CLAUSE = re.compile(r"\b(?:that|which|who|whose|where|when|if|because|as)\b")
AUX = re.compile(r"\b(is|are|was|were|will|can|should|must)\b")


def split_rel(s: str) -> tuple[str, str, str] | None:
    """(subject, relation, object) of a simple relational sentence (the earliest relation of RELS), else None: no
    subordinate opening, "then" or comma."""
    if SUBORD.match(s) or " then " in s or "," in s or ";" in s:
        return None
    hits = sorted((m.start(), k, rel) for k, rel in enumerate(RELS) for m in re.finditer(re.escape(rel), s))
    for i, _, rel in hits:
        sub, obj = s[:i], s[i + len(rel):]
        if (stems(sub) and stems(obj) and not REL_CLAUSE.search(sub)
                and (REL_CANON[rel] != " is a " or len(obj.split()) <= SHORT_PRED)):
            return sub, rel, obj
    return None


def third(base: str) -> str:
    if re.search(r"(?:s|sh|ch|x|z|o)$", base):
        return base + "es"
    if re.search(r"[^aeiou]y$", base):
        return base[:-1] + "ies"
    return base + "s"


def antonym_edits(s: str) -> list[tuple[str, str, str]]:
    """(variant, old word, new word) for every antonym edit of s (see the docstring)."""
    toks = list(HTOK.finditer(s))
    have = {stem(t.group()) for t in toks}
    r = split_rel(s)
    start = len(r[0]) + len(r[1]) if r and REL_CANON[r[1]] in TAXONOMIC else 0
    out = []
    for i, t in enumerate(toks):
        w = t.group()
        if w not in ANT or t.start() < start or stem(ANT[w]) in have:
            continue
        a, lo = ANT[w], t.start()
        if i > 0 and toks[i - 1].group() in ("a", "an") and toks[i - 1].end() == t.start() - 1:
            a, lo = ("an " if a[0] in "aeiou" else "a ") + a, toks[i - 1].start()
        out.append((s[:lo] + a + s[t.end():], w, ANT[w]))
    return out


def negate_edit(s: str) -> tuple[str, str, str] | None:
    """(variant, old, new): one negation removed, or "not" inserted after the main clause's first copula / modal."""
    negs = NEG.findall(s)
    if len(negs) == 1:
        for pat, rep in ((r"\bcannot\b", "can"), (r"\bcan not\b", "can"),
                         (r"\b(is|are|was|were|will|should|must|has|have) not\b", r"\1")):
            m = re.search(pat, s)
            if m:
                return s[:m.start()] + m.expand(rep) + s[m.end():], m.group(), m.expand(rep)
        m = re.search(r"\bdo not ([a-z]+)\b", s)
        if m:
            return s[:m.start()] + m.group(1) + s[m.end():], m.group(), m.group(1)
        m = re.search(r"\bdoes not ([a-z]+)\b", s)
        if m and m.group(1) not in ("be", "have"):
            return s[:m.start()] + third(m.group(1)) + s[m.end():], m.group(), third(m.group(1))
        return None
    if negs or HEDGE.search(s) or SUBORD.match(s):
        return None
    m = AUX.search(s)
    r = split_rel(s)
    if r and REL_CANON[r[1]] in NEG_VERBS and (not m or m.start() > len(r[0])):     # "X causes Y": the main verb
        verb, *rest = r[1].split()
        singular = f" {r[1].strip()} " == REL_CANON[r[1]]
        base = "have" if verb in ("has", "have") else verb if not singular else (
            verb[:-3] + "y" if verb.endswith("ies") else verb[:-2] if verb.endswith(("sses", "shes", "ches", "xes"))
            else verb[:-1])
        new = " ".join([("does not " if singular else "do not ") + base] + rest)
        return r[0] + " " + new + " " + r[2], r[1].strip(), new
    if not m or REL_CLAUSE.search(s[:m.start()]) or not stems(s[:m.start()]):
        return None
    new = "cannot" if m.group() == "can" else m.group() + " not"
    return s[:m.start()] + new + s[m.end():], m.group(), new


class Swapper:
    """The donors of the swap edit: the objects of each typed relation in the train trees' sentences."""

    def __init__(self, donors):
        pool = collections.defaultdict(dict)
        for iid, s in donors:
            r = split_rel(s)
            if r and REL_CANON[r[1]] not in VERB_RELS:
                pool[REL_CANON[r[1]]].setdefault(r[2], iid)
        self.pool = {k: [(o, iid, stems(o), len(o.split()), head(o)) for o, iid in sorted(v.items())]
                     for k, v in pool.items()}

    def edits(self, s: str, avoid: set, known: Known) -> list[tuple[str, str, str, str]]:
        """(variant, old object, new object, donor item) for every admissible object swap of s (typed relations only,
        never in a hedged sentence)."""
        r = split_rel(s)
        if not r or REL_CANON[r[1]] in VERB_RELS or HEDGE_SWAP.search(s):
            return []
        sub, rel, obj = r
        canon = REL_CANON[rel]
        o_stems, s_stems, n = stems(obj), stems(sub), len(obj.split())
        keys = (np_key(sub), head(sub))
        anc = known.ancestors(sub) if canon in TAXONOMIC else set()
        out = []
        for o, iid, st, k, h in self.pool.get(canon, ()):
            if abs(k - n) > SWAP_LEN or st & o_stems or st & avoid or not st:
                continue
            if (any((x, canon, h) in known.triples for x in keys) or (anc and (np_key(o) in anc or h in anc))
                    or known.related(s_stems, st)):
                continue
            out.append((sub + rel + o, obj, o, iid))
        return out


REVERSIBLE = {" causes ", " contains ", " requires ", " needs ", " eats ", " produces ", " orbits ", " carries ",
              " transports ", " emits ", " releases ", " uses ", " provides ", " makes ", " prevents ", " protects ",
              " is a part of "}                          # asymmetric relations (canonical forms; "is made of" left
#                                                          # out: "particles are made of matter" stays true)
REV_WORDS = 4                                            # each argument of a reversal: at most this many words
_NP_OK = frozenset("a an the its their his her".split())


_PLURAL_HEAD = re.compile(r"(?:[^su]s|people|children|mice|teeth|feet|geese|men|women|cattle|bacteria|fungi)$")


def plural_np(np: str) -> bool:
    w = np.split()
    return w[0] not in ("a", "an", "one", "each", "every") and bool(_PLURAL_HEAD.search(w[-1]))


def reverse_edit(s: str, known: Known) -> tuple[str, str, str] | None:
    """(variant, old, new): "X causes Y" -> "Y causes X" for an asymmetric relation (REVERSIBLE) between two short noun
    phrases (no function word but an article or possessive, no "a / b"), the verb agreeing with the new subject; not in
    a hedged sentence, not a known triple."""
    r = split_rel(s)
    if not r or REL_CANON[r[1]] not in REVERSIBLE or HEDGE_SWAP.search(s):
        return None
    sub, rel, obj = r
    for np in (sub, obj):
        ws = np.split()
        if len(ws) > REV_WORDS or "/" in np or any(w in STOP and w not in _NP_OK or w == "to" for w in ws):
            return None
    canon = REL_CANON[rel]
    if not plural_np(obj):
        new = canon
    elif canon.startswith(" is "):
        new = canon.replace(" is ", " are ", 1)
    else:
        v = canon.strip()
        new = f" {v[:-3] + 'y' if v.endswith('ies') else v[:-1]} "
    if any((x, canon, head(sub)) in known.triples for x in (np_key(obj), head(obj))):
        return None
    return obj + new + sub, f"{sub} |{rel.strip()}| {obj}", f"{obj} |{new.strip()}| {sub}"


def false_variant(s: str, item: dict, rng: random.Random, swapper: Swapper, known: Known, avoid: set,
                  opt_stems: set) -> dict | None:
    """The No twin of true sentence s: {claim, wrong, edit {from, to}[, donor]}, or None (no admissible edit). Each
    edit type's candidates in a random order, the first admissible one (not s, not known true, no new option stem, no
    gold leak); then the type by WRONG_W among those with one."""
    def ok(v: str, new: str) -> bool:
        return (norm(v) != norm(s) and norm(v) not in known.true and not (stems(new) - stems(s)) & opt_stems
                and not gold_leak(item, v))
    cands = {}
    rev = reverse_edit(s, known)
    for kind, xs in (("antonym", [(v, a, b, None) for v, a, b in antonym_edits(s)]),
                     ("swap", swapper.edits(s, avoid, known)), ("reverse", [(*rev, None)] if rev else [])):
        rng.shuffle(xs)
        cands[kind] = next((x for x in xs if ok(x[0], x[2] if kind != "reverse" else "")), None)
    avail = [k for k in ("antonym", "swap", "reverse") if cands[k]]
    if avail:
        kind = rng.choices(avail, weights=[WRONG_W[k] for k in avail])[0]
        v, a, b, donor = cands[kind]
    else:
        e = negate_edit(s)
        if e is None or not ok(e[0], e[2]):
            return None
        kind, (v, a, b), donor = "negate", e, None
    out = {"claim": v, "wrong": kind, "edit": {"from": a, "to": b}}
    if donor:
        out["donor"] = donor
    return out


# ------------------------------------------------------------------------------------------------------ trees
def parse_tree(r: dict) -> dict:
    """The tree of one EntailmentBank task_1 record: texts {id: sentence}, premises {int: [ids]}, the root, the leaves
    and conclusions the root's proof reaches in id order (a conclusion no step uses, an annotation slip in 3 records,
    is left out with the leaves only it uses: "dangling"), heights on the proof edges (EntailmentBank's depth_of_proof
    disagrees with its own proof by one in 46 records: the edges win; "eb_depth_differs"); ValueError when the proof
    does not parse or disagrees with meta."""
    m = r["meta"]
    root = m["hypothesis_id"]
    texts = {**m["triples"], **m["intermediate_conclusions"]}
    prem = {}
    for step in (x for x in r["proof"].split(";") if x.strip()):
        mm = STEP.match(step)
        if not mm:
            raise ValueError(f"{r['id']}: proof step {step!r}")
        tgt = root if mm.group(2) == "hypothesis" else mm.group(2)
        ps = [x.strip() for x in mm.group(1).split("&")]
        if tgt in prem or not set(ps) <= set(texts) or tgt not in m["intermediate_conclusions"]:
            raise ValueError(f"{r['id']}: proof step {step!r} does not match meta")
        if mm.group(3) and norm(mm.group(3)) != norm(texts[tgt]):
            raise ValueError(f"{r['id']}: {tgt} text differs between proof and meta")
        prem[tgt] = ps
    if root not in prem:
        raise ValueError(f"{r['id']}: the root is not concluded")
    used, todo = set(), [root]
    while todo:
        for p in prem.get(todo.pop(), []):
            if p not in used:
                used.add(p)
                todo.append(p)
    dangling = sorted((set(prem) | {p for ps in prem.values() for p in ps}) - used - {root})
    memo: dict = {}

    def height(i: str, seen: frozenset = frozenset()) -> int:
        if i not in memo:
            if i in seen:
                raise ValueError(f"{r['id']}: cycle at {i}")
            memo[i] = 1 + max(height(p, seen | {i}) for p in prem[i]) if i in prem else 0
        return memo[i]
    key = lambda x: (x.startswith("int"), int(re.sub(r"\D", "", x)))      # noqa: E731
    order = (sorted((x for x in used if x.startswith("sent")), key=key)
             + sorted((x for x in used if x.startswith("int")), key=key))
    tree = {"texts": {k: clean(v) for k, v in texts.items()}, "premises": {k: v for k, v in prem.items() if k in used
                                                                           or k == root},
            "root": root, "order": order, "height": {i: height(i) for i in order + [root]}, "dangling": dangling,
            "steps": 1 + sum(x.startswith("int") for x in used)}
    if len(prem) != r["length_of_proof"]:
        raise ValueError(f"{r['id']}: proof length differs from the record")
    tree["eb_depth_differs"] = tree["height"][root] != r["depth_of_proof"]
    return tree


def build_nodes(tree: dict, item: dict, hyp: str, held: set, rng: random.Random, swapper: Swapper, known: Known,
                st: collections.Counter) -> list[dict]:
    """The Yes / No node pairs of one tree (see the docstring); st counts the drops."""
    texts, prem = tree["texts"], tree["premises"]
    st["dropped_root"] += 1
    q = item["prompt"].partition("\nOptions:\n")[0]
    opt_stems = set().union(*(stems(o) for o in item["options"]))
    avoid = opt_stems | stems(q) | stems(hyp) | set().union(*(stems(texts[i]) for i in tree["order"]))
    nodes, built, seen = [], set(), set()
    for sid in tree["order"]:
        s = texts[sid]
        why = gold_leak(item, s)
        if why is None and norm(s) == norm(hyp):
            why = "equals_hypothesis"
        if why is None and norm(s) in held:
            why = "states_heldout_hypothesis"
        if why is None and norm(s) in seen:
            why = "duplicate_text"
        if why:
            st[f"dropped_{why}"] += 1
            continue
        seen.add(norm(s))
        no = false_variant(s, item, rng, swapper, known, avoid, opt_stems)
        if no is None:
            st["dropped_no_false_variant"] += 1
            continue
        ps = prem.get(sid, [])
        par = [f"{p}v" for p in ps if p in built]
        st["parent_edges"] += len(ps)
        st["parent_edges_unbuilt"] += len(ps) - len(par)
        meta = {"type": "inference" if sid in prem else "fact", "depends_on": par, "initial": None,
                "status": "direct", "p_fresh": None, "sentence": s, "sent_id": sid, "premises": ps,
                "node_depth": tree["height"][sid]}
        nodes.append({"id": f"{sid}v", "question": YES_Q.format(s), "opposite": NO_Q.format(s), "truth": True,
                      "p_cot": 1.0, "claim": s, **meta})
        nodes.append({"id": f"{sid}n", "question": YES_Q.format(no["claim"]), "opposite": NO_Q.format(no["claim"]),
                      "truth": False, "p_cot": 0.0, **no, **meta})
        st[f"wrong_{no['wrong']}"] += 1
        built.add(sid)
    return nodes


# ------------------------------------------------------------------------------------------------------ audit
def gold_text(item: dict) -> str:
    return str(item["options"][item["labels"].index(item["gold_label"])])


def prompt_of(question: str, options: list[str]) -> str:
    return f"Question: {question}\nOptions:\n" + "\n".join(f"{L}) {t}" for L, t in zip(LABELS, options))


def audit_item(item: dict) -> list[str]:
    """Problems of one built item from its own fields (empty: clean): the final, the GFPQ context, every node's twin,
    wording, truth, parents (Yes nodes of its premises), depth, and no gold leak (gold_leak) in any claim."""
    from cotdistill.treeplan import problem_no_options
    bad, iid = [], item.get("item_id")
    gold = gold_text(item)
    head_, _, opts = item["prompt"].partition("\nOptions:\n")
    if (item.get("source") != SOURCE or item.get("tree_fix") is not True or item.get("labels") != LABELS
            or len(set(map(norm, item["options"]))) != 4 or item["teacher"] != {L: float(L == item["gold_label"])
                                                                                for L in LABELS}
            or item["prompt"] != prompt_of(head_[len("Question: "):], item["options"])
            or item.get("has_tree") is not bool(item["subqs"]) or item["group"] != f"{SOURCE}/d{item['depth']}"):
        bad.append("item fields")
    try:
        ctx = problem_no_options(item)
        if ctx != head_ or "\n" in ctx:
            bad.append("GFPQ context is not the prompt before its options")
    except ValueError as e:
        bad.append(f"problem_no_options: {e}")
    by = {n["id"]: n for n in item["subqs"]}
    if len(by) != len(item["subqs"]):
        bad.append("duplicate node ids")
    if sum(n["truth"] for n in item["subqs"]) * 2 != len(item["subqs"]):
        bad.append("Yes / No not balanced")
    for n in item["subqs"]:
        sid, i = n["sent_id"], n["id"]
        twin = by.get(f"{sid}{'n' if n['truth'] else 'v'}")
        if (i != f"{sid}{'v' if n['truth'] else 'n'}" or not twin or twin["sentence"] != n["sentence"]
                or twin["truth"] == n["truth"] or twin["depends_on"] != n["depends_on"]):
            bad.append(f"{i}: id / twin")
        if (n["question"] != YES_Q.format(n["claim"]) or n["opposite"] != NO_Q.format(n["claim"])
                or n["p_cot"] != float(n["truth"]) or (n["claim"] == n["sentence"]) != n["truth"]
                or (not n["truth"] and n.get("wrong") not in ("antonym", "swap", "reverse", "negate"))):
            bad.append(f"{i}: wording / truth")
        if gold_leak(item, n["claim"]) or gold_leak(item, n["sentence"]):
            bad.append(f"{i}: gold leak")
        if n["type"] != ("inference" if n["premises"] else "fact") or (n["type"] == "fact") != (n["node_depth"] == 0):
            bad.append(f"{i}: type / depth")
        for d in n["depends_on"]:
            par = by.get(d)
            if not par or not par["truth"] or par["sent_id"] not in n["premises"]:
                bad.append(f"{i}: parent {d} missing, not Yes or not a premise")
            elif n["node_depth"] < par["node_depth"] + 1:
                bad.append(f"{i}: node_depth below a parent's + 1")
    return [f"{iid}: {b}" for b in bad]


# ------------------------------------------------------------------------------------------------------ cues
def cue_rows(item: dict, worldtree: set) -> list[tuple]:
    """(truth, negation word, content share in the question, content share in question + options, claim in the
    WorldTree corpus, words, inference) per node."""
    q = item["prompt"].partition("\nOptions:\n")[0]
    qs, qo = stems(q), stems(item["prompt"])
    rows = []
    for n in item["subqs"]:
        c = stems(n["claim"]) or {""}
        rows.append((n["truth"], bool(NEG.search(n["claim"])), len(c & qs) / len(c), len(c & qo) / len(c),
                     norm(n["claim"]) in worldtree, len(n["claim"].split()), n["type"] == "inference"))
    return rows


def stump(rows: list[tuple], k: int) -> float | None:
    """Accuracy of the best one-feature rule for the node target (binary feature: either sign; numeric: the best
    threshold, either side)."""
    xs = [(r[0], r[k]) for r in rows]
    if not xs:
        return None
    if isinstance(xs[0][1], bool):
        acc = sum(t == f for t, f in xs) / len(xs)
        return round(max(acc, 1 - acc), 3)
    xs.sort(key=lambda x: x[1])
    n, pos = len(xs), sum(t for t, _ in xs)
    best, below_yes = max(pos, n - pos), 0
    for j, (t, v) in enumerate(xs):
        below_yes += t
        if j + 1 < n and xs[j + 1][1] == v:
            continue
        below = j + 1
        acc = below_yes + (n - below) - (pos - below_yes)               # below -> Yes, above -> No
        best = max(best, acc, n - acc)
    return round(best / n, 3)


# ------------------------------------------------------------------------------------------------------ dedup
def grams8(text: str) -> set:
    w = words(text)
    return {" ".join(w[i:i + 8]) for i in range(len(w) - 7)}


def stem_of(prompt: str) -> str:
    s = prompt.partition("\nOptions:")[0]
    return s[len("Question: "):] if s.startswith("Question: ") else s


# ------------------------------------------------------------------------------------------------------ io
def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def write(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return md5(path)


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def stats(items: list[dict]) -> dict:
    nodes = [n for it in items for n in it["subqs"]]
    c = lambda xs: dict(sorted(collections.Counter(xs).items()))       # noqa: E731
    return {"items": len(items), "with_tree": sum(it["has_tree"] for it in items),
            "arc_set": c(it["arc_set"] for it in items), "depth": c(it["depth"] for it in items),
            "gold_label": c(it["gold_label"] for it in items),
            "nodes": len(nodes), "nodes_per_item": round(len(nodes) / max(1, len(items)), 2),
            "nodes_per_item_hist": c(len(it["subqs"]) for it in items),
            "yes_share": round(sum(n["truth"] for n in nodes) / max(1, len(nodes)), 4),
            "node_type": c(n["type"] for n in nodes), "node_depth": c(n["node_depth"] for n in nodes),
            "wrong": c(n["wrong"] for n in nodes if not n["truth"]),
            "claim_words_mean": {k: round(sum(len(n["claim"].split()) for n in nodes if n["truth"] is v)
                                          / max(1, sum(n["truth"] is v for n in nodes)), 2)
                                 for k, v in (("yes", True), ("no", False))}}


def load_arc() -> dict:
    import pandas as pd
    out = {}
    for name in ARC_FILES:
        for r in pd.read_parquet(ARC_DIR / f"{name}.parquet").itertuples(index=False):
            out[r.id] = {"file": name, "question": str(r.question), "texts": [str(x) for x in r.choices["text"]],
                         "keys": [str(x) for x in r.choices["label"]], "key": str(r.answerKey)}
    return out


def contamination(stems_by_split: dict, skip: set) -> dict:
    """Items of other training files (data/student_*/train.jsonl, ARC lines only) whose question is one of ours."""
    out = {}
    for path in sorted(glob.glob(str(ROOT / "data/student_*/train.jsonl"))):
        rel = str(Path(path).relative_to(ROOT))
        if rel.split("/")[1] in skip:
            continue
        hits = collections.Counter()
        with open(path) as f:
            for line in f:
                if '"arc' not in line:
                    continue
                r = json.loads(line)
                s = norm(stem_of(r.get("prompt", "")))
                for sp, ss in stems_by_split.items():
                    hits[sp] += s in ss
        if sum(hits.values()):
            out[rel] = dict(hits)
    return out


# ------------------------------------------------------------------------------------------------------ build
def finals(eb: list[dict], arc: dict, split: str, seed: int, st: collections.Counter, fails: list) -> list[tuple]:
    """(final item, record, tree) per question of one EntailmentBank split: ARC options, gold and question checked."""
    by_id = collections.defaultdict(list)
    for r in eb:
        by_id[r["id"]].append(r)
    out = []
    for i, rs in by_id.items():
        a = arc.get(i)
        if a is None:
            st[f"{split}_dropped_not_in_arc"] += 1
            continue
        if len(a["texts"]) != 4:
            st[f"{split}_dropped_{len(a['texts'])}_options"] += 1
            continue
        if a["key"] not in a["keys"] or len({norm(t) for t in a["texts"]}) != 4:
            st[f"{split}_dropped_bad_arc_record"] += 1
            continue
        g = a["keys"].index(a["key"])
        match = [r for r in rs if norm(r["answer"]) == norm(a["texts"][g])]
        st[f"{split}_eb_answer_text_differs_from_arc_gold"] += sum(norm(r["answer"]) != norm(a["texts"][g]) for r in rs)
        if len(rs) > 1:
            st[f"{split}_repeated_id_records_dropped"] += len(rs) - 1
        r = (match or rs)[0]
        if not match:
            st[f"{split}_kept_eb_answer_not_arc_gold"] += 1
        if clean(r["question"]) != clean(a["question"]):
            fails.append(f"{i}: EntailmentBank question differs from ARC")
            continue
        try:
            tree = parse_tree(r)
        except ValueError as e:
            fails.append(str(e))
            continue
        order = list(range(4))
        random.Random(f"{seed}/options/{i}").shuffle(order)
        st[f"{split}_eb_depth_of_proof_differs"] += tree["eb_depth_differs"]
        st[f"{split}_dangling_sentences_skipped"] += len(tree["dangling"])
        st[f"{split}_trees_with_dangling"] += bool(tree["dangling"])
        depth = tree["height"][tree["root"]]
        options = [clean_opt(a["texts"][j]) for j in order]
        gl = LABELS[order.index(g)]
        q = " ".join(a["question"].split())
        if "Options:" in q or "Question:" in q:
            st[f"{split}_dropped_question_markup"] += 1
            continue
        item = {"item_id": f"eb-{i}", "source": SOURCE, "group": f"{SOURCE}/d{depth}", "type": "choice",
                "prompt": prompt_of(q, options), "labels": list(LABELS), "gold_label": gl,
                "teacher": {L: float(L == gl) for L in LABELS}, "options": options,
                "option_kinds": ["gold" if j == g else "distractor" for j in order], "recipe": "arc",
                "depth": depth, "n_steps": tree["steps"], "subqs": [], "random_subqs": [],
                "rationale": "", "augment": None, "has_tree": False, "tree_fix": True, "row_kind": "final",
                "problem_id": i, "node_status": None, "licence_class": "CHECK", "licence": LICENCE,
                "label_basis": "gold", "node_basis": "human_tree", "tree_source": TREE_SOURCE,
                "arc_set": "challenge" if "Challenge" in a["file"] else "easy",
                "eb_split": {"val": "dev"}.get(split, split), "eb_id": i,
                "eb_depth_of_proof": r["depth_of_proof"], "eb_length_of_proof": r["length_of_proof"]}
        out.append((item, r, tree))
    return out


def clean_opt(t: str) -> str:
    return " ".join(str(t).split())


def build(args) -> None:
    global GOLD_RECALL_MAX, GOLD_POINTER
    GOLD_RECALL_MAX, GOLD_POINTER = args.gold_recall_max, (args.gold_pointer_min, args.gold_pointer_margin)
    st = collections.Counter()
    fails: list[str] = []
    arc = load_arc()
    eb = {sp: read_jsonl(RAW / "dataset/task_1" / f"{sp}.jsonl") for sp, _ in SPLITS}
    wt = json.loads((RAW / "supporting_data/worldtree_corpus_sentences_extended.json").read_text())
    worldtree = {norm(a) for s in wt.values() for a in alternatives(" ".join(str(s).lower().split()))}
    every = [s for rs in eb.values() for r in rs for s in list(r["meta"]["triples"].values())
             + list(r["meta"]["intermediate_conclusions"].values()) + [r["hypothesis"]]]
    known = Known(every + list(wt.values()))
    print(f"known true {len(known.true)}, kind-of keys {len(known.kind)}, triples {len(known.triples)}, "
          f"rss {rss_mb()} MB", flush=True)
    rows = {}
    for sp, out in SPLITS:
        rows[out] = finals(eb[sp], arc, out, args.seed, st, fails)
    # ---- dedup: question text (test, then val, win), then 8-grams shared by fewer than 5 questions
    dedup = {"by_text": {}, "by_8gram": {}, "examples": []}
    seen = set()
    for out in ("test", "val", "train"):
        keep = []
        for x in rows[out]:
            k = norm(stem_of(x[0]["prompt"]))
            if k in seen:
                dedup["by_text"][out] = dedup["by_text"].get(out, 0) + 1
                continue
            seen.add(k)
            keep.append(x)
        rows[out] = keep
    owners = collections.defaultdict(set)
    for out in rows:
        for x in rows[out]:
            for g in grams8(stem_of(x[0]["prompt"])):
                owners[g].add(x[0]["item_id"])
    later = {"train": ("val", "test"), "val": ("test",), "test": ()}
    ids = {out: {x[0]["item_id"] for x in rows[out]} for out in rows}
    for out in ("val", "train"):
        held = set().union(*(ids[o] for o in later[out]))
        keep = []
        for x in rows[out]:
            g = sorted(g for g in grams8(stem_of(x[0]["prompt"])) if len(owners[g]) < 5 and owners[g] & held)
            if g:
                dedup["by_8gram"][out] = dedup["by_8gram"].get(out, 0) + 1
                if len(dedup["examples"]) < 6:
                    dedup["examples"].append({"item_id": x[0]["item_id"], "grams": g[:2],
                                              "with": sorted(set().union(*(owners[y] for y in g)) & held)[:3]})
                continue
            keep.append(x)
        rows[out] = keep
    ids = {out: {x[0]["item_id"] for x in rows[out]} for out in rows}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
    # ---- nodes: donors from train trees only; a node never states a later split's hypothesis
    hyps = {out: {norm(x[1]["hypothesis"]) for x in rows[out]} for out in rows}
    donors = [(x[0]["item_id"], x[2]["texts"][s]) for x in rows["train"] for s in x[2]["order"]]
    swapper = Swapper(donors)
    items = {}
    for out in ("train", "val", "test"):
        held = set().union(*(hyps[o] for o in later[out]))
        stx = collections.Counter()
        built = []
        for item, r, tree in rows[out]:
            nodes = build_nodes(tree, item, clean(r["hypothesis"]), held,
                                random.Random(f"{args.seed}/nodes/{item['problem_id']}"), swapper, known, stx)
            item = dict(item, subqs=nodes, has_tree=bool(nodes), node_status="ok" if nodes else "no_nodes")
            stx["sentences"] += len(tree["order"]) + 1
            built.append(item)
        items[out] = built
        st.update({f"{out}_{k}": v for k, v in stx.items()})
        print(out, len(built), dict(stx), f"rss {rss_mb()} MB", flush=True)
    # ---- audit and leaks
    from cotdistill.teacher import leaks_answer
    leak = collections.Counter()
    recall_hist = collections.Counter()
    for out, its in items.items():
        held = set().union(*(hyps[o] for o in later[out]))
        for it in its:
            fails += audit_item(it)
            gold = gold_text(it)
            dis = [o for o in it["options"] if o != gold]
            for n in it["subqs"]:
                leak["nodes"] += 1
                why = gold_leak(it, n["claim"])
                if why:
                    leak[f"gold_leak_{why}"] += 1
                leak["teacher_leaks_answer"] += leaks_answer(n["question"], it)
                leak["claim_is_heldout_hypothesis"] += norm(n["claim"]) in held
                leak["no_claim_known_true"] += (not n["truth"]) and norm(n["claim"]) in known.true
                rr = recall(gold, n["claim"])
                recall_hist[f"{out}/{'<0.25' if rr < .25 else '0.25-0.5' if rr < .5 else '>=0.5'}"] += 1
                for t in (True, False):
                    if n["truth"] is t:
                        k = "yes" if t else "no"
                        leak[f"{k}_claims_with_a_distractor_text"] += any(phrase_in(o, n["claim"]) for o in dis)
                        leak[f"{k}_claims"] += 1
    if (leak["no_claim_known_true"] or leak["claim_is_heldout_hypothesis"] or leak["teacher_leaks_answer"]
            or any(k.startswith("gold_leak_") and v for k, v in leak.items())):
        fails.append("a No claim is known true, or a claim states a held-out hypothesis or leaks the gold")
    # shared knowledge: train Yes claims that are sentences of val / test trees (not dropped: facts, not answers)
    later_sents = {norm(x[2]["texts"][s]) for o in ("val", "test") for x in rows[o] for s in x[2]["order"]}
    leak["train_yes_claims_in_val_test_trees"] = sum(n["truth"] and norm(n["claim"]) in later_sents
                                                     for it in items["train"] for n in it["subqs"])
    # ---- cues (train)
    cue = {}
    rows_ = [r for it in items["train"] for r in cue_rows(it, worldtree)]
    for k, name in enumerate(("negation_word", "share_in_question_gfpq", "share_in_question_options_gfp",
                              "claim_in_worldtree", "claim_words", "inference_node"), start=1):
        cue[name] = stump(rows_, k)
    cue["n"] = len(rows_)
    cue["negation_word_share"] = {k: round(sum(r[1] for r in rows_ if r[0] is v) / max(1, sum(r[0] is v
                                                                                              for r in rows_)), 3)
                                  for k, v in (("yes", True), ("no", False))}
    # ---- contamination of other training files by these questions
    contam = None
    if not args.no_contam:
        skip = {Path(args.out).name, Path(args.out).name + "_f25"}
        contam = contamination({o: {norm(stem_of(it["prompt"])) for it in items[o]} for o in items}, skip)
    # ---- write
    out_dir, f25_dir = ROOT / args.out, ROOT / (args.out + "_f25")
    summary = {"script": "scripts/build_entailmentbank_tf.py", "args": vars(args),
               "raw": {"url": RAW_URL, "zip": str(RAW_ZIP.relative_to(ROOT)), "zip_md5": md5(RAW_ZIP),
                       "zip_sha256": hashlib.sha256(RAW_ZIP.read_bytes()).hexdigest(),
                       "task_1_md5": {sp: md5(RAW / "dataset/task_1" / f"{sp}.jsonl") for sp, _ in SPLITS},
                       "arc": {n: md5(ARC_DIR / f"{n}.parquet") for n in ARC_FILES},
                       "eb_test_is_arc_validation": True},
               "md5": {}, "files": {}}
    paths = {"train": out_dir / "train.jsonl", "val": out_dir / "val.jsonl",
             "test": ROOT / args.eval_dir / "entailmentbank_tf_test.jsonl"}
    for out, p in paths.items():
        key = str(p.relative_to(ROOT))
        summary["md5"][key] = write(p, items[out])
        summary["files"][key] = stats(items[out])
    summary["build"] = dict(sorted(st.items()))
    summary["dedup"] = dedup
    summary["leaks"] = dict(sorted(leak.items()))
    summary["gold_recall_hist_kept_claims"] = dict(sorted(recall_hist.items()))
    summary["cues_train"] = cue
    summary["contamination_other_train_files"] = contam
    summary["audit"] = {"failures": len(fails), "examples": fails[:20]}
    summary["fields"] = {"item": ["source entailmentbank_tf", "group entailmentbank_tf/d<depth>", "depth (proof depth)",
                                  "n_steps (proof length)", "subqs", "has_tree", "tree_source", "arc_set", "eb_split",
                                  "eb_id", "option_kinds"],
                         "subq": ["id <sid>v / <sid>n", "question", "opposite", "truth", "p_cot", "p_fresh",
                                  "type fact / inference", "depends_on", "initial", "status", "claim", "sentence",
                                  "sent_id", "premises", "node_depth", "wrong (No: antonym / swap / negate)",
                                  "edit {from, to}", "donor (swap)"]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    f25 = subset(items["train"], paths["val"], f25_dir, args.f25_fraction, args.f25_seed,
                 str(out_dir.relative_to(ROOT)))
    print(json.dumps({k: summary[k] for k in ("md5", "files", "build", "dedup", "leaks", "gold_recall_hist_kept_claims",
                                              "cues_train", "contamination_other_train_files", "audit")}, indent=1))
    print("f25", json.dumps(f25))
    print(f"rss {rss_mb()} MB")
    if fails:
        print("AUDIT FAILED", fails[:10])
        sys.exit(1)


def subset(train: list[dict], val_path: Path, out_dir: Path, frac: float, seed: str, source: str) -> dict:
    """A seeded share of train stratified by group, as data/student_gsm8k_tf_f25: one generator, groups in sorted
    order, each group (file order) shuffled and cut to round(frac x size); kept in file order; val copied."""
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for it in train:
        by[it["group"]].append(it)
    keep = set()
    for g in sorted(by):
        xs = list(by[g])
        rng.shuffle(xs)
        keep |= {it["item_id"] for it in xs[:round(len(xs) * frac)]}
    rows = [it for it in train if it["item_id"] in keep]
    out_dir.mkdir(parents=True, exist_ok=True)
    s = {"source": source, "fraction": frac, "stratified_by": "group (proof depth)", "seed": seed,
         "n_train": len(rows), "train_md5": write(out_dir / "train.jsonl", rows)}
    shutil.copyfile(val_path, out_dir / "val.jsonl")
    (out_dir / "summary.json").write_text(json.dumps(s, indent=1))
    s["groups"] = {g: [len(by[g]), sum(it["group"] == g for it in rows)] for g in sorted(by)}
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/student_entailmentbank_tf")
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gold-recall-max", type=float, default=GOLD_RECALL_MAX)
    ap.add_argument("--gold-pointer-min", type=float, default=GOLD_POINTER[0])
    ap.add_argument("--gold-pointer-margin", type=float, default=GOLD_POINTER[1])
    ap.add_argument("--f25-fraction", type=float, default=0.25)
    ap.add_argument("--f25-seed", default="entailmentbank_tf|f25|20261014")
    ap.add_argument("--no-contam", action="store_true", help="skip the scan of data/student_*/train.jsonl")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
