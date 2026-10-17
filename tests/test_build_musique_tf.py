"""MuSiQue Tree Forcing data (scripts/build_musique_tf.py, source musique_tf) and its treeplan branch:
- treeplan: statement_block (GFPS) is the paragraphs, problem_no_options (GFPQ) adds the question line, shape refusals,
  placebo_text is the paragraphs (needs torch);
- builder units: placeholders ("#k" with k < j only), triple / natural renderings, the descriptive and the defined
  forms (no earlier answer written), substitute, answer identity (aliases, containment, acronyms), whole names, shapes;
- synthetic MuSiQue records built end to end (needs the tokenizer file): schema and types parity with a GSM8K row
  for the fields the trainer reads, gold never repeated or aliased among the options, Yes/No twins and balance, no
  placeholder answer written, deterministic output for a fixed seed, the item audit (and that it catches a corruption);
- the built files (skipped when absent): schema parity, the audit on a handful, balance, every view within the token
  budget, val / test disjoint from train by id and question text, summary md5s, the f25 subset and its recipe against
  data/student_gsm8k_tf_f25;
- with torch: GFPS / GFPQ / placebo (GFPSX) views of a few built rows.
CPU only. Data-backed tests read data/ of this checkout, or COTD_DATA_DIR when set."""
import argparse
import collections
import copy
import hashlib
import json
import os
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_musique_tf as B  # noqa: E402
from cotdistill.treeplan import MUSIQUE_TF, problem_no_options, statement_block  # noqa: E402

DATA = Path(os.environ.get("COTD_DATA_DIR") or ROOT / "data")
FILES = {"train": DATA / "student_musique_tf/train.jsonl", "val": DATA / "student_musique_tf/val.jsonl",
         "test": DATA / "eval/musique_tf_test.jsonl"}
HAVE_TOKENIZER = (ROOT / B.TOKENIZER).exists()
try:
    import tokenizers  # noqa: F401
except ImportError:
    HAVE_TOKENIZER = False
try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# Fields the trainer, the evaluators and the controls read, with the GSM8K row's types (data/student_gsm8k_tf).
# probe / value are numbers in GSM8K and strings here (as strategyqa_tf's entity nodes): placebo_tree_numbers and the
# readouts accept both.
GSM8K_ITEM_TYPES = {"item_id": str, "source": str, "group": str, "type": str, "prompt": str, "labels": list,
                    "gold_label": str, "teacher": dict, "options": list, "option_kinds": list, "recipe": str,
                    "depth": int, "dep_level": int, "subqs": list, "random_subqs": list, "rationale": str,
                    "augment": type(None), "has_tree": bool, "tree_fix": bool, "row_kind": str, "problem_id": str,
                    "node_status": str, "licence_class": str, "licence": str, "label_basis": str, "node_basis": str,
                    "n_steps": int, "tree_source": str}
GSM8K_NODE_TYPES = {"id": str, "question": str, "opposite": str, "truth": bool, "p_cot": float, "p_fresh": type(None),
                    "type": str, "depends_on": list, "initial": type(None), "status": str, "step": int, "flags": list,
                    "gold_ancestor": bool, "node_depth": int}


def para(idx, title, text, sup=False):
    return {"idx": idx, "title": title, "paragraph_text": text, "is_supporting": sup}


FILLER = [para(10 + k, t, x) for k, (t, x) in enumerate([
    ("Bob Lane", "Bob Lane was born in Austin in 1961 and worked with Carol Diaz and Frank Ortiz."),
    ("Fargo", "Fargo is a city in North Dakota, founded in 1871. Its mayor is Tim Mahoney."),
    ("Austin", "Austin is the capital of Texas. Its mayor in 2015 was Steve Adler, married to Diane Land."),
    ("Grace Kim", "Grace Kim is a novelist from Canada, born in 1975 in Toronto. She lives in Ottawa."),
    ("Paul Neri", "Paul Neri is an Italian writer from Italy. He was married to Laura Neri in 1980."),
    ("Henry Cole", "Henry Cole is a British author from England, born in 1950 in Leeds."),
    ("Students", "In Canada about 15,000 students are called rookies, and 9,500 are called novices."),
])]
RAW = [
    {"id": "2hop__101_102", "question": "Where was the performer of Green Song born?", "answer": "Duluth",
     "answer_aliases": ["Duluth, Minnesota"], "answerable": True,
     "question_decomposition": [
         {"id": 101, "question": "Green Song >> performer", "answer": "Alice Moran", "paragraph_support_idx": 0},
         {"id": 102, "question": "Where was #1 born?", "answer": "Duluth", "paragraph_support_idx": 1}],
     "paragraphs": [para(0, "Green Song", "Green Song is a 1999 single by Alice Moran, produced by Bob Lane.", True),
                    para(1, "Alice Moran", "Alice Moran (born 1970 in Duluth) is a singer. She moved to Fargo.", True)]
     + FILLER},
    {"id": "3hop1__103_104_105", "question": "Who is the spouse of the mayor of the city where Alice Moran was born?",
     "answer": "Doug Larson", "answer_aliases": [], "answerable": True,
     "question_decomposition": [
         {"id": 103, "question": "Alice Moran >> place of birth", "answer": "Duluth", "paragraph_support_idx": 1},
         {"id": 104, "question": "Who is the mayor of #1?", "answer": "Emily Larson", "paragraph_support_idx": 2},
         {"id": 105, "question": "#2 >> spouse", "answer": "Doug Larson", "paragraph_support_idx": 3}],
     "paragraphs": [para(1, "Alice Moran", "Alice Moran (born 1970 in Duluth) is a singer. She moved to Fargo.", True),
                    para(2, "Duluth", "Duluth is a city in Minnesota. Its mayor is Emily Larson since 2016.", True),
                    para(3, "Emily Larson", "Emily Larson is an American politician. She is married to Doug Larson.",
                         True)] + FILLER},
    {"id": "4hop3__106_107_108_109", "question": "How many people called what new students were called live in the "
     "country of the author of Blue Days?", "answer": "15,000", "answer_aliases": [], "answerable": True,
     "question_decomposition": [
         {"id": 106, "question": "Blue Days >> author", "answer": "Grace Kim", "paragraph_support_idx": 4},
         {"id": 107, "question": "#1 >> country of citizenship", "answer": "Canada", "paragraph_support_idx": 13},
         {"id": 108, "question": "what were new students once called", "answer": "rookies",
          "paragraph_support_idx": 5},
         {"id": 109, "question": "How many #3 live in #2 ?", "answer": "15,000", "paragraph_support_idx": 16}],
     "paragraphs": [para(4, "Blue Days", "Blue Days is a 2001 novel by Grace Kim, set in Ottawa.", True),
                    para(5, "Freshman", "New students were once called rookies by others, later freshers.", True)]
     + [dict(p, is_supporting=p["idx"] in (13, 16)) for p in FILLER]},
    {"id": "3hop1__110_111_112", "question": "Which country is the city where the writer of Red Sky died in?",
     "answer": "Italy", "answer_aliases": ["Italia"], "answerable": True,
     "question_decomposition": [
         {"id": 110, "question": "Who wrote Red Sky?", "answer": "Paul Neri", "paragraph_support_idx": 6},
         {"id": 111, "question": "In which city did #1 die?", "answer": "Turin", "paragraph_support_idx": 7},
         {"id": 112, "question": "#2 >> country", "answer": "Italy", "paragraph_support_idx": 8}],
     "paragraphs": [para(6, "Red Sky", "Red Sky is a 1990 play written by Paul Neri.", True),
                    para(7, "Paul Neri death", "Paul Neri died in Turin in 2004 after living in Milan.", True),
                    para(8, "Turin", "Turin is a city in Italy, near France and Switzerland.", True)] + FILLER},
]


def args(**kw):
    a = argparse.Namespace(seed=0, max_len=1536, margin=32, placeholders="describe")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def gsm8k_row() -> dict | None:
    p = DATA / "student_gsm8k_tf/train.jsonl"
    if not p.exists():
        return None
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            if r["subqs"]:
                return r
    return None


def check_types(test, it):
    for k, t in GSM8K_ITEM_TYPES.items():
        test.assertIn(k, it, k)
        test.assertIsInstance(it[k], t, k)
    test.assertEqual(it["source"], MUSIQUE_TF)
    test.assertTrue(all(isinstance(v, float) for v in it["teacher"].values()))
    test.assertTrue(all(isinstance(o, str) for o in it["options"]))
    for n in it["subqs"]:
        for k, t in GSM8K_NODE_TYPES.items():
            test.assertIn(k, n, k)
            test.assertIsInstance(n[k], t, k)
        test.assertIsInstance(n["probe"], str)
        test.assertIsInstance(n["value"], str)
        test.assertEqual("wrong" in n, not n["truth"])


def check_item(test, it):
    """Options, balance and leaks of one built item."""
    gold = B.gold_text(it)
    golds = [it["musique"]["answer"]] + it["musique"]["answer_aliases"]
    test.assertEqual(it["options"].count(gold), 1)
    for L, o in zip(it["labels"], it["options"]):
        if L != it["gold_label"]:
            test.assertFalse(any(B.same_answer(o, g) for g in golds), (it["item_id"], o))
    test.assertEqual(len({B.norm(o) for o in it["options"]}), 5)
    nodes = it["subqs"]
    test.assertEqual(sum(n["truth"] for n in nodes) * 2, len(nodes))
    ids = {n["id"] for n in nodes}
    for n in nodes:
        test.assertIn(n["id"][:-1] + ("n" if n["truth"] else "v"), ids)
        test.assertLess(n["step"], it["n_steps"])                        # the last hop is never asked
        test.assertFalse(B.same_answer(n["value"], gold))
    test.assertEqual(B.audit_item(it), [], it["item_id"])


# ---- treeplan branch ---------------------------------------------------------------------------------------------
PARAS = "Title: Green Song\nGreen Song is a single by Alice Moran.\n\nTitle: Alice Moran\nAlice Moran is a singer."
OPTS = "\n".join(f"{L}) option {L}" for L in "ABCDE")


def item(prompt=None):
    return {"item_id": "musique-x", "source": MUSIQUE_TF,
            "prompt": prompt or f"{PARAS}\n\nQuestion: Where was the performer of Green Song born?\nOptions:\n{OPTS}"}


class TestTreeplanBranch(unittest.TestCase):
    def test_statements_and_question(self):
        self.assertEqual(statement_block(item()), PARAS)
        self.assertEqual(problem_no_options(item()), PARAS + "\n\nQuestion: Where was the performer of Green Song born?")

    def test_refusals(self):
        bad = [f"Question: q?\nOptions:\n{OPTS}",                                   # no paragraphs
               f"{PARAS}\n\nQuestion: a\nb?\nOptions:\n{OPTS}",                     # two-line question
               f"{PARAS}\nQuestion: x\n\nQuestion: q?\nOptions:\n{OPTS}",           # a second question line
               f"{PARAS} Options: x\n\nQuestion: q?\nOptions:\n{OPTS}",             # an options label in the text
               f"{PARAS}\n\nQuestion: q?"]                                          # no options
        for p in bad:
            for fn in (statement_block, problem_no_options):
                with self.assertRaises(ValueError, msg=p):
                    fn(item(p))

    @unittest.skipUnless(HAVE_TORCH, "cotdistill.student needs torch")
    def test_placebo_text_is_the_paragraphs(self):
        from cotdistill.student import placebo_text
        self.assertEqual(placebo_text(item()), PARAS)


# ---- builder units -----------------------------------------------------------------------------------------------
class TestUnits(unittest.TestCase):
    def test_refs(self):
        self.assertEqual(B.refs("Where was #1 born?", 2), [1])
        self.assertEqual(B.refs("#9 Dream >> performer", 1), [])                 # a title, not a placeholder
        self.assertEqual(B.refs("How many #3 live in #2 ?", 4), [2, 3])

    def test_renderings(self):
        hops = RAW[1]["question_decomposition"]
        self.assertEqual(B.render_hop(hops[0]["question"]), "Where was Alice Moran born?")
        q, kind, res = B.hop_text(hops, 2, "describe")
        self.assertEqual((q, kind), ("Who is the mayor of the place of birth of Alice Moran?", "inline"))
        self.assertEqual(res, "Who is the mayor of Duluth?")
        self.assertNotIn("Duluth", q)                                           # no earlier answer written
        self.assertEqual(B.hop_text(hops, 2, "substitute")[0], "Who is the mayor of Duluth?")
        q4, kind4, _ = B.hop_text(RAW[2]["question_decomposition"], 4, "describe")    # natural referent: defined
        self.assertEqual(kind4, "defined")
        self.assertEqual(q4, 'Let #3 be the answer to "What were new students once called?". How many #3 live in the '
                             'country of citizenship of the author of Blue Days?')
        for ans in ("Canada", "rookies", "Grace Kim"):
            self.assertNotIn(ans, q4)
        q2, kind2, _ = B.hop_text(RAW[3]["question_decomposition"], 2, "describe")
        self.assertEqual((q2, kind2), ('Let #1 be the answer to "Who wrote Red Sky?". In which city did #1 die?',
                                       "defined"))                              # "Who wrote ...?" has no noun phrase
        hops = [{"question": "Who wrote X?", "answer": "a"}, {"question": "#1 >> spouse", "answer": "b"},
                {"question": "Where did #2 die?", "answer": "c"}, {"question": "z"}]
        self.assertEqual(B.hop_text(hops, 3, "describe")[0],
                         'Let #1 be the answer to "Who wrote X?". Where did the spouse of #1 die?')

    def test_natural_noun_phrase_guard(self):
        hops = [{"question": "Who was the edict addressed to?", "answer": "x"},
                {"question": "Where did the emperor greet the #1 ?", "answer": "y"}]
        self.assertIsNone(B.describe(hops, 1))
        hops[0]["question"] = "What is the capital of Peru?"
        self.assertEqual(B.hop_text(hops, 2, "describe")[0], "Where did the emperor greet the capital of Peru?")

    def test_same_answer(self):
        self.assertTrue(B.same_answer("the Soviet Union", "Soviet Union"))
        self.assertTrue(B.same_answer("New York", "New York City"))
        self.assertTrue(B.same_answer("NES", "Nintendo Entertainment System"))
        self.assertTrue(B.same_answer("The U.S.", "the United States"))
        self.assertFalse(B.same_answer("Kim Jong-un", "Kim Jong-il"))
        self.assertFalse(B.same_answer("Austria", "Australia"))

    def test_whole_names_and_spans(self):
        self.assertFalse(B.has_name("He met David Bowie in 1970.", "David"))
        self.assertTrue(B.has_name("In London, the band played.", "London"))
        self.assertEqual(B.name_spans("Title: Green Day\nIn 1990 Billy Joe Armstrong of Oakland, California met the "
                                      "University of Southampton."),
                         ["Green Day", "Billy Joe Armstrong", "Oakland", "California", "University of Southampton"])

    def test_classes(self):
        self.assertEqual(B.hop_class("Where did X live when he was young?", "Los Angeles")[0], "place")
        self.assertEqual(B.hop_class("When was #1 founded?", "1960")[0], "year")
        self.assertEqual(B.hop_class("When did X end?", "c. AD 400 or later"), ("other", "when"))
        self.assertEqual(B.hop_class("X >> performer", "Papa Roach"), ("person", "performer"))

    def test_pick_prefers_shape_and_tier(self):
        cands = [("the French", "context", False), ("California", "context", False), ("Dave Koz", "context", False),
                 ("Ben Affleck", "pool", False)]
        got = B.pick(cands, 1, lambda x, c: False, random.Random(0), "George Benson")
        self.assertEqual(got, [("Dave Koz", "context")])


# ---- synthetic records end to end --------------------------------------------------------------------------------
@unittest.skipUnless(HAVE_TOKENIZER, "needs the tokenizers package and the repo tokenizer file")
class TestSyntheticBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.L = B.Lengths(ROOT / B.TOKENIZER)

    def build(self, a=None, raw=None):
        raw = copy.deepcopy(raw or RAW)
        pool = B.Pool(raw)
        pos = B.gold_positions([r["id"] for r in raw], 0, "train")
        st = collections.Counter()
        return [B.build_item(r, pos[r["id"]], pool, self.L, a or args(), st) for r in raw], st

    def test_items(self):
        items, st = self.build()
        self.assertTrue(all(it is not None for it in items), st)
        for it in items:
            check_types(self, it)
            check_item(self, it)
            self.assertLessEqual(it["musique"]["max_view_tokens"], 1536 - 32)
            r = next(x for x in RAW if x["id"] == it["problem_id"])
            self.assertEqual(B.audit_support(it, r), [])
            for n in it["subqs"]:                                              # no earlier hop's answer written
                stem = n["question"].rpartition(" Is it ")[0]
                for h in r["question_decomposition"][:n["step"] - 1]:
                    self.assertFalse(B.has_phrase(stem.lower(), h["answer"]), (n["question"], h["answer"]))
        by = {it["problem_id"]: it for it in items}
        three = by["3hop1__103_104_105"]
        self.assertEqual([n["id"] for n in three["subqs"]], ["s1v", "s1n", "s2v", "s2n"])
        self.assertEqual(three["subqs"][2]["depends_on"], ["s1v"])
        self.assertEqual(three["subqs"][2]["node_depth"], 1)
        self.assertEqual(B.problem_no_options(three), three["prompt"].partition("\nOptions:")[0])
        self.assertTrue(statement_block(three).startswith("Title: "))

    def test_deterministic(self):
        a, _ = self.build()
        b, _ = self.build()
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        c, _ = self.build(args(seed=1))
        self.assertNotEqual(json.dumps(a, sort_keys=True), json.dumps(c, sort_keys=True))

    def test_gold_hop_blocked(self):
        raw = copy.deepcopy(RAW)                                               # (the others fill the pool)
        raw[0]["answer"] = raw[0]["question_decomposition"][1]["answer"] = "Alice Moran Hall"
        raw[0]["paragraphs"][1]["paragraph_text"] += " The Alice Moran Hall is in Duluth."
        items, st = self.build(raw=raw)
        self.assertEqual(items[0]["subqs"], [])
        self.assertEqual(st["blocked_gold_valued"], 1)

    def test_audit_catches(self):
        items, _ = self.build()
        it = copy.deepcopy(next(x for x in items if x["subqs"]))
        it["subqs"][1]["probe"] = it["subqs"][1]["value"]                      # a No probe made true
        self.assertTrue(B.audit_item(it))
        it = copy.deepcopy(items[0])
        g = it["labels"].index(it["gold_label"])
        it["options"][(g + 1) % 5] = it["musique"]["answer_aliases"][0]        # an alias among the options
        self.assertTrue(any("alias" in x or "item fields" in x for x in B.audit_item(it)))

    @unittest.skipUnless(gsm8k_row() is not None, "no data/student_gsm8k_tf")
    def test_parity_with_gsm8k_row(self):
        g = gsm8k_row()
        items, _ = self.build()
        for k, t in GSM8K_ITEM_TYPES.items():                                  # the reference types are GSM8K's own
            self.assertIsInstance(g[k], t, k)
        for k, t in GSM8K_NODE_TYPES.items():
            self.assertIsInstance(g["subqs"][0][k], t, k)
        for it in items:
            self.assertTrue(set(GSM8K_ITEM_TYPES) <= set(it) & set(g))
            self.assertEqual(it["labels"], g["labels"])
            self.assertEqual(set(it["teacher"]), set(g["teacher"]))


# ---- the built files ---------------------------------------------------------------------------------------------
def read(path, n=None):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
            if n and len(out) >= n:
                break
    return out


@unittest.skipUnless(all(p.exists() for p in FILES.values()), "built MuSiQue files absent")
class TestBuiltFiles(unittest.TestCase):
    HANDFUL = 60

    @classmethod
    def setUpClass(cls):
        cls.items = {k: read(p) for k, p in FILES.items()}

    def test_schema_audit_balance(self):
        for name, items in self.items.items():
            rng = random.Random(name)
            for it in rng.sample(items, self.HANDFUL):
                check_types(self, it)
                check_item(self, it)
            for it in items:
                self.assertEqual(sum(n["truth"] for n in it["subqs"]) * 2, len(it["subqs"]))
                self.assertLessEqual(it["musique"]["max_view_tokens"], 1536)

    def test_gold_positions_balanced(self):
        for items in self.items.values():
            c = collections.Counter(it["gold_label"] for it in items)
            self.assertEqual(set(c), set("ABCDE"))
            # positions are assigned in a balanced cycle before item drops (gold not a whole-word span, over budget)
            self.assertLessEqual(max(c.values()) - min(c.values()), max(8, len(items) // 50))

    def test_splits_disjoint(self):
        ids = {k: {it["problem_id"] for it in v} for k, v in self.items.items()}
        qs = {k: {B.norm(B.musique_tf_cut(it)[1]) for it in v} for k, v in self.items.items()}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            self.assertFalse(ids[a] & ids[b], (a, b))
            self.assertFalse(qs[a] & qs[b], (a, b))
        self.assertTrue(all(i.startswith(("2hop", "3hop", "4hop")) for v in ids.values() for i in v))
        self.assertGreater(len(self.items["val"]), 300)

    def test_summary_md5(self):
        s = json.loads((DATA / "student_musique_tf/summary.json").read_text())
        self.assertEqual(s["audit"]["failures"], 0)
        self.assertEqual(s["tokens"]["additive_count_check"]["mismatches"], 0)
        self.assertEqual(s["hop_overlap"]["test_items_sharing_a_hop_with_train"], 0)
        for key, m in s["md5"].items():
            rel = key[len("data/"):] if key.startswith("data/") else key
            self.assertEqual(hashlib.md5((DATA / rel).read_bytes()).hexdigest(), m, key)

    def test_f25_subset(self):
        sub = DATA / "student_musique_tf_f25"
        if not (sub / "train.jsonl").exists():
            self.skipTest("no f25 subset")
        lines = [x for x in FILES["train"].read_text().split("\n") if x]
        sl = [x for x in (sub / "train.jsonl").read_text().split("\n") if x]
        index = {x: i for i, x in enumerate(lines)}
        pos = [index[x] for x in sl]
        self.assertEqual(pos, sorted(pos))                                  # byte-identical lines, file order
        meta = json.loads((sub / "summary.json").read_text())
        self.assertEqual(pos, B.subset(self.items["train"], 0.25, meta["seed"]))
        groups = collections.Counter(it["group"] for it in self.items["train"])
        self.assertEqual(len(sl), sum(max(1, round(v * 0.25)) for v in groups.values()))
        self.assertEqual((sub / "val.jsonl").read_bytes(), FILES["val"].read_bytes())

    def test_subset_recipe_matches_gsm8k_f25(self):
        src, sub = DATA / "student_gsm8k_tf/train.jsonl", DATA / "student_gsm8k_tf_f25/train.jsonl"
        if not (src.exists() and sub.exists()):
            self.skipTest("no gsm8k_tf data")
        lines = [x for x in src.read_text().split("\n") if x]
        keep = B.subset([json.loads(x) for x in lines], 0.25, "gsm8k_tf|f25|20261012")
        out = "".join(lines[i] + "\n" for i in keep).encode()
        self.assertEqual(hashlib.md5(out).hexdigest(), hashlib.md5(sub.read_bytes()).hexdigest())

    @unittest.skipUnless(HAVE_TORCH, "cotdistill.student needs torch")
    def test_gfps_gfpq_and_placebo_views(self):
        from cotdistill.student import PlaceboBank, build_full_tree
        rows = [it for it in self.items["train"][:200] if it["subqs"]][:20]
        bank = PlaceboBank(self.items["train"][:400])
        for it in rows:
            for view in ("statements", "statements+question"):
                ex = build_full_tree(it, mode="plain", rng=random.Random(0), cap=0, view=view)
                self.assertEqual(len(ex), 1 + len(it["subqs"]))
                ctx = statement_block(it) if view == "statements" else problem_no_options(it)
                self.assertTrue(all(e.text.startswith(ctx) for e in ex[1:]))
                self.assertNotIn("\nOptions:\nA) ", ex[1].text.partition("Intermediate question:")[0])
                plc = build_full_tree(it, mode="plain", rng=random.Random(0), cap=0, view=view, target_seed=0,
                                      target_kind="placebo", placebo_bank=bank)
                self.assertEqual([e.target for e in plc], [e.target for e in ex])   # same targets as GFPS / GFPQ


if __name__ == "__main__":
    unittest.main()
