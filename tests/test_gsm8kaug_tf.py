"""GSM8K-Aug Tree Forcing data (scripts/build_gsm8kaug_tf.py -> data/student_gsm8kaug_tf{,_24k,_96k}):
- equation parsing and exact verification (fractions, %, decimals; rounded / wrong values, unparsable steps and
  answers dropped; the last step must be the answer);
- the NL statements: alignment of the NL sentences with the equations, the result spans of the No twin, number
  formats, the cloze for qop / opsel, the equation fallback;
- the fixture rows (tests/fixtures/gsm8kaug_tf_items_v1.json, raw GSM8K-Aug / -NL rows and the built rows): rebuilt
  byte for byte from the raw rows, audit clean, the key set / key order of a data/student_gsm8k_tf_si row and GSM8K's
  node key sets, the fields the trainer reads with their types, 5 distinct options with the gold once, no node for the
  gold step, 50% Yes;
- decontamination: a planted GSM8K-test question (verbatim, numbers changed, name changed) is dropped, a planted val
  question too, seed attribution of a rewrite; clean questions pass;
- subsets: exact sizes, nested, stratified by n_steps;
- the built files (skipped when absent): audit clean, md5s, nesting on disk, val = the GSM8K val, no val / test
  question in train;
- with torch: a CPU dry construction through cotdistill.student of A2, GFPQ, GFPQV, GFPQO, SIB and GFPQC on 3 rows,
  and the trainer's data checks for these arms.
CPU only, no model. Data-backed tests read data/ of this checkout, or COTD_DATA_DIR when set."""
import collections
import json
import os
import random
import re
import sys
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

DATA = Path(os.environ.get("COTD_DATA_DIR") or ROOT / "data")
FX = json.loads((ROOT / "tests/fixtures/gsm8kaug_tf_items_v1.json").read_text())
GSM_FX = json.loads((ROOT / "tests/fixtures/gsm8k_si_items_v1.json").read_text())["gsm8k"]
AUG = DATA / "student_gsm8kaug_tf"
SRC = {"gsm8k_tf"}

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

Q600 = ("Out of 600 employees in a company, 30% got promoted while 10% received bonus. How many employees did not get "
        "either a promotion or a bonus?")
EQ600 = ["<<600*30/100=180>>", "<<600*10/100=60>>", "<<180+60=240>>", "<<600-240=360>>"]
NL600 = ["600 x 30/100 = 180 employees were promoted.", "600 x 10/100 = 60 employees received a bonus.",
         "So a total of 180+60=240 employees received a promotion or a bonus.",
         "Therefore, 600 - 240 = 360 employees did not get either a promotion or a bonus."]


def G():
    import build_gsm8kaug_tf as m
    return m


def lines(path: Path) -> list[str]:
    return [x for x in path.read_text().split("\n") if x]            # not splitlines: U+2028 in texts


@unittest.skipUnless(HAVE_NUMPY, "build_gsm8k_socratic needs numpy")
class TestVerify(unittest.TestCase):
    def test_exact_steps(self):
        g = G()
        why, eqs, gold = g.verify_row(Q600, EQ600, "360")
        self.assertIsNone(why)
        self.assertEqual(eqs, [("600*30/100", "180"), ("600*10/100", "60"), ("180+60", "240"), ("600-240", "360")])
        self.assertEqual(gold, 360)
        cases = {("<<150/0.2=750>>",): ("750", [("150/0.2", "750")]),           # decimals
                 ("<<20%*50=10>>",): ("10", [("(20/100)*50", "10")]),             # P% read as P/100
                 ("<<1/4*2=1/2>>",): ("0.5", [("1/4*2", "0.5")]),                 # a fraction as the stated value
                 ("<<3/4*8=6>>", "<<6*1.5=9>>"): ("$9", [("3/4*8", "6"), ("6*1.5", "9")]),
                 ("<<1,200*3=3,600>>",): ("3,600", [("1,200*3", "3600")]),        # thousands separators
                 ("<<2^3=8>>",): ("8", [("2^3", "8")])}
        for steps, (ans, want) in cases.items():
            why, eqs, _ = g.verify_row("Q?", list(steps), ans)
            self.assertIsNone(why, steps)
            self.assertEqual(eqs, want, steps)

    def test_drops(self):
        g = G()
        cases = [(["<<65/60=1.08333>>"], "1.08333", "value_rounded"),        # exact 1.083333...: not the stated value
                 (["<<20/3=6.67>>"], "6.67", "value_rounded"),
                 (["<<4*3=13>>"], "13", "value_wrong"),
                 (["<<4*3=12>>", "<<12+1=13>>"], "12", "last_step_not_answer"),
                 ([""], "1", "step_format"), (["<<<88-28=60>>"], "60", "step_format"),
                 (["<<28/3=>>"], "9", "value_unparsable"), (["The answer is <<2*3=6>>"], "6", "step_format"),
                 (["<<x=5>>"], "5", "expr_unparsable"), (["<<1500/19≈79>>"], "79", "step_format"),
                 (["<<1500/19=78.9~79>>"], "79", "value_unparsable"), (["<<a*b=6>>"], "6", "expr_symbol"),
                 (["<<1/4+1/3=7/12>>"], "7/12", "value_over_4_decimals"), (["<<6-6=0>>"], "0", "answer_not_positive_2dec"),
                 (["<<10/8=1.25>>", "<<1.25/2=0.625>>"], "0.625", "answer_not_positive_2dec"),
                 (["<<2*3=6>>"], "~6", "answer_not_number"), (["<<2*3=6>>"], "2:45", "answer_not_number"),
                 ([], "1", "no_steps")]
        for steps, ans, want in cases:
            self.assertEqual(g.verify_row("Q?", steps, ans)[0], want, steps)
        self.assertEqual(g.verify_row("Line one\nline two?", ["<<2*3=6>>"], "6")[0], "question_shape")

    def test_parse(self):
        g = G()
        _, eqs, gold = g.verify_row(Q600, EQ600, "360")
        p = g.parse_verified(0, Q600, eqs, gold)
        self.assertEqual(p["status"], "ok")
        self.assertEqual(p["gold_step"], 3)
        self.assertEqual([s["parents"] for s in p["steps"]], [[], [], [0], [2]])     # 60: a unit constant, no edge
        self.assertIn("parents_uncertain", p["steps"][2]["flags"])
        self.assertEqual([s["level"] for s in p["steps"]], [0, 0, 1, 2])
        _, eqs, gold = g.verify_row("Ann has 3 cats. How many?", ["<<3*17=51>>"], "51")
        self.assertEqual(g.parse_verified(1, "Ann has 3 cats. How many?", eqs, gold)["status"], "orphan")


@unittest.skipUnless(HAVE_NUMPY, "build_gsm8k_socratic needs numpy")
class TestStatements(unittest.TestCase):
    def test_align(self):
        g = G()
        self.assertEqual(g.align([Fraction(x) for x in (180, 60, 240, 360)], NL600), [0, 1, 2, 3])
        extra = ["Let S be the weekly salary."] + NL600                       # extra narrative sentence: still 1:1
        self.assertEqual(g.align([Fraction(x) for x in (180, 60, 240, 360)], extra), [1, 2, 3, 4])
        two = ["The blanket has 20 rows * 20 squares = 400 squares",
               "Done: 400 * 20/100 = 80 squares, so 400 - 80 = 320 squares left"]   # two results in one sentence
        self.assertIsNone(g.align([Fraction(400), Fraction(80), Fraction(320)], two))
        # "S = 150 / 0.2" is not a result (an operator follows), "= 750" is
        self.assertEqual([r[3] for r in g.results("we get S = 150 / 0.2 = 750 So she earns $750.")], [Fraction(750)])
        self.assertEqual([r[3] for r in g.results("3 x $1,200 = $3,600 for rent")], [Fraction(3600)])

    def test_spans_and_formats(self):
        g = G()
        s = NL600[2]
        why, spans = g.nl_spans(s, Fraction(240), Fraction(360))
        self.assertIsNone(why)
        self.assertEqual(g.nl_statement(s, spans, Fraction(240)), s)
        self.assertEqual(g.nl_statement(s, spans, Fraction(230)),
                         "So a total of 180+60=230 employees received a promotion or a bonus.")
        self.assertEqual(g.nl_spans("Then 60 + 60 = 120 apples, and 120 apples in all.", Fraction(120), Fraction(7))[1],
                         [(15, 18), (31, 34)])                          # the result and its later repeat
        self.assertEqual(g.nl_spans("She had 120 before: 60 + 60 = 120.", Fraction(120), Fraction(7))[0],
                         "value_before_result")
        self.assertEqual(g.nl_spans("So 2 + 3 = 5, leaving 360 for the rest.", Fraction(5), Fraction(360))[0],
                         "gold_in_sentence")
        # the sentence must write the verified equation's operands (100 / .01 aside; 0.7 may be written 70%)
        self.assertEqual(g.nl_spans("360 minutes = 6 hours.", Fraction(6), Fraction(9), [Fraction(360), Fraction(60)])[0],
                         "operands_not_in_sentence")
        self.assertIsNone(g.nl_spans("It leaves 300 apples * 70% = 210 apples.", Fraction(210), Fraction(9),
                                     [Fraction(300), Fraction("0.7")])[0])
        self.assertIsNone(g.nl_spans(NL600[0], Fraction(180), Fraction(360),
                                     [Fraction(600), Fraction(30), Fraction(100)])[0])
        s = "He pays 3 x $1,200 = $3,600 for rent."
        _, spans = g.nl_spans(s, Fraction(3600), Fraction(10))
        self.assertEqual(g.nl_statement(s, spans, Fraction(3700)), "He pays 3 x $1,200 = $3,700 for rent.")
        self.assertEqual((g.fmt_like("12.50", Fraction(13)), g.fmt_like("1,200", Fraction(13000)),
                          g.fmt_like("12.5", Fraction("12.25")), g.fmt_like("7", Fraction("7.5"))),
                         ("13.00", "13,000", None, None))

    def test_cloze(self):
        g = G()
        self.assertEqual(g.cloze(NL600[0], Fraction(180), Fraction(360)), "___ employees were promoted.")
        self.assertEqual(g.cloze(NL600[2], Fraction(240), Fraction(360)),
                         "So a total of ___ employees received a promotion or a bonus.")
        self.assertEqual(g.cloze("He pays 3 x $1,200 = $3,600 for rent.", Fraction(3600), Fraction(10)),
                         "He pays ___ for rent.")
        self.assertIsNone(g.cloze("we get S = 150 / 0.2 = 750 So she earns $750.", Fraction(750), Fraction(9)))  # repeat
        self.assertIsNone(g.cloze("600 - 240 = 360.", Fraction(360), Fraction(9)))        # no words left
        self.assertIsNone(g.cloze("She has 5 * 4 = 21 pens.", Fraction(21), Fraction(9)))  # lhs is not 21

    def test_eq_text(self):
        g = G()
        _, eqs, gold = g.verify_row("Ann has 4 bags of 1/2 kg and 3 kg. How many kg?", ["<<1/2*4=2>>", "<<2+3=5>>"], "5")
        p = g.parse_verified(0, "Ann has 4 bags of 1/2 kg and 3 kg. How many kg?", eqs, gold)
        self.assertEqual(g.eq_text(p["steps"][0]), "1/2 × 4")
        _, eqs, gold = g.verify_row(Q600, EQ600, "360")
        p = g.parse_verified(0, Q600, eqs, gold)
        self.assertEqual([g.eq_text(s) for s in p["steps"]], ["600 × 30 ÷ 100", "600 × 10 ÷ 100", "180 + 60",
                                                             "600 − 240"])


@unittest.skipUnless(HAVE_NUMPY, "build_gsm8k_socratic needs numpy")
class TestRows(unittest.TestCase):
    def test_reproduces_fixture_rows(self):
        """build_item on the raw GSM8K-Aug / -NL rows gives the fixture rows (taken from the built file) byte for byte."""
        g = G()
        for it in FX["items"]:
            r = FX["raw"][it["item_id"]]
            got = g.build_item(r["row"], r["question"], r["steps"], r["answer"], r["nl_question"], r["nl_steps"], 0,
                               collections.Counter(), pool_rank=it["pool_rank"])
            self.assertEqual(json.dumps(got, ensure_ascii=False), json.dumps(it, ensure_ascii=False), it["item_id"])

    def test_audit_clean(self):
        g = G()
        for it in FX["items"]:
            self.assertEqual(g.audit_row(it), [], it["item_id"])

    def test_row_600(self):
        """GSM8K-Aug train row 0: NL statements, the No twin's result replaced, no node for the gold step, the clozes,
        the chain."""
        g = G()
        it = g.build_item(0, Q600, EQ600, "360", Q600, NL600, 0, collections.Counter(), pool_rank=0)
        yes = [n["question"] for n in it["subqs"] if n["truth"]]
        self.assertEqual(yes, ["Is this step correct? " + s for s in NL600[:3]])
        for n in it["subqs"]:
            self.assertNotEqual(n["step"], 4)                                # the gold step has no node
            if not n["truth"]:
                twin = next(x for x in it["subqs"] if x["id"] == n["id"][:-1] + "v")
                self.assertNotEqual(n["question"], twin["question"])
                self.assertIn(g.T._fmt(n["probe"]), n["question"])
        self.assertEqual([n["subq"] for n in it["value_nodes"]],
                         ["___ employees were promoted.", "___ employees received a bonus.",
                          "So a total of ___ employees received a promotion or a bonus."])
        self.assertEqual([ln["text"] for ln in it["worked_steps"]],
                         ["30% of 600 = 180", "10% of 600 = 60", "180 + 60 = 240", "600 − 240 = 360"])
        # unaligned NL (or a differing question): equation statements, op renderings only
        it2 = g.build_item(0, Q600, EQ600, "360", Q600, ["It is 600 x 30/100 = 180 and 600 x 10/100 = 60."], 0,
                           collections.Counter(), pool_rank=0)
        self.assertEqual([n["question"] for n in it2["subqs"] if n["truth"]],
                         ["Is it true that 600 × 30 ÷ 100 = 180?", "Is it true that 600 × 10 ÷ 100 = 60?",
                          "Is it true that 180 + 60 = 240?"])
        self.assertTrue(all(set(n["renderings"]) == {"op"} and n["subq"] == "" for n in it2["value_nodes"]))
        self.assertEqual(g.audit_row(it), [])
        self.assertEqual(g.audit_row(it2), [])
        self.assertEqual({k: v for k, v in it.items() if k not in ("subqs", "value_nodes")},
                         {k: v for k, v in it2.items() if k not in ("subqs", "value_nodes")})

    def test_schema_parity(self):
        """Key set and order of a data/student_gsm8k_tf_si row; GSM8K's node key sets (yes/no, value nodes, chain
        lines); the trainer-read fields with their types."""
        gsm_keys = list(GSM_FX[0])
        yn_keys = {frozenset(n) for it in GSM_FX for n in it["subqs"]}
        vn_keys = {frozenset(n) for it in GSM_FX for n in it["value_nodes"]} | {
            frozenset(set(n) | {"percent"}) for it in GSM_FX for n in it["value_nodes"]}
        ws_keys = {frozenset(ln) for it in GSM_FX for ln in it["worked_steps"] or []}
        ren_keys = {k: {frozenset(n["renderings"][k]) for it in GSM_FX for n in it["value_nodes"]
                        if k in n["renderings"]} for k in ("op", "qop", "opsel")}
        si = DATA / "student_gsm8k_tf_si/train.jsonl"
        if si.exists():
            with open(si) as f:
                self.assertEqual(list(json.loads(f.readline())), gsm_keys)
        for it in FX["items"]:
            self.assertEqual(list(it), gsm_keys, it["item_id"])
            self.assertEqual(it["source"], "gsm8k_tf")
            self.assertTrue(it["item_id"].startswith("gsm8kaug-train-") and it["problem_id"] == it["item_id"])
            self.assertEqual(it["labels"], list("ABCDE"))
            self.assertIn(it["gold_label"], it["labels"])
            self.assertEqual(it["teacher"], {L: float(L == it["gold_label"]) for L in it["labels"]})
            self.assertTrue(it["prompt"].startswith("Question: ") and "\n" not in it["prompt"].partition("\nOptions:")[0])
            self.assertEqual((it["row_kind"], it["tree_fix"], it["depth"]), ("final", True, it["n_steps"]))
            ids = {n["id"] for n in it["subqs"]}
            for n in it["subqs"]:
                self.assertIn(frozenset(n), yn_keys, n["id"])
                self.assertIsInstance(n["truth"], bool)
                self.assertEqual(n["p_cot"], float(n["truth"]))
                self.assertEqual(n["type"], "step")
                self.assertTrue(isinstance(n["step"], int) and n["step"] >= 1)
                self.assertIsInstance(n["node_depth"], int)
                self.assertTrue(set(n["depends_on"]) <= ids and all(d.endswith("v") for d in n["depends_on"]))
                self.assertNotIn("options", n)                           # not a multiple-choice node
            for n in it["value_nodes"]:
                self.assertIn(frozenset(n), vn_keys, n["id"])
                for k, r in n["renderings"].items():
                    self.assertIn(frozenset(r), ren_keys[k])
                    self.assertEqual(len(r["options"]), 3)
                    self.assertIsInstance(r["answer"], int)
            for ln in it["worked_steps"] or []:
                self.assertIn(frozenset(ln), ws_keys)
                self.assertTrue(all(isinstance(ln[k], str) for k in ("text", "wrong_text", "wrong_result_text")))

    def test_finals_and_nodes(self):
        g = G()
        for it in FX["items"]:
            gold = g.T.gold_text(it)
            self.assertEqual(len(set(it["options"])), 5)
            self.assertEqual(it["options"].count(gold), 1)
            self.assertEqual(it["option_kinds"][it["labels"].index(it["gold_label"])], "gold")
            self.assertIn(it["recipe"], ("qo2", "inter", "slip"))
            nodes = it["subqs"]
            self.assertEqual(sum(n["truth"] for n in nodes) * 2, len(nodes))
            for n in nodes:
                self.assertLess(n["step"], it["n_steps"])                 # never the gold (last) step
                self.assertNotEqual(g.T._fmt(n["value"]), gold)
                self.assertNotEqual(g.T._fmt(n["probe"]), gold)
                self.assertTrue(n["question"].startswith(("Is this step correct? ", "Is it true that ")))
            if it["worked_steps"]:
                self.assertEqual(it["worked_steps"][-1]["value"], gold)
        forms = collections.Counter(n["question"].split("?")[0] for it in FX["items"] for n in it["subqs"])
        self.assertTrue(forms["Is this step correct"] and any(k.startswith("Is it true that") for k in forms))


@unittest.skipUnless(HAVE_NUMPY, "build_gsm8k_socratic needs numpy")
class TestScreen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        g = G()
        cls.targets = []
        for name in ("gsm8k_mc_v2", "gsm1k_mc"):
            for line in lines(DATA / f"eval/{name}.jsonl"):
                r = json.loads(line)
                cls.targets.append((g.eval_stem(r["prompt"]), r.get("orig_id") or r["item_id"], f"eval/{name}"))
        cls.test = g.Screen(cls.targets, sets="tri")
        cls.val_t = g.val_targets(DATA / "student_gsm8k_tf/val.jsonl")
        cls.val = g.Screen(cls.val_t, sets="uni")

    def test_planted_test_question(self):
        """A GSM8K test question dropped verbatim, with its numbers changed and with a name changed; clean questions
        (the fixture rows) pass; verdict() drops a planted row as test_overlap and keeps it with a fresh question."""
        g = G()
        q = self.targets[0][0]                                         # gsm8k-test-0 (Janet's ducks)
        self.assertIn("exact", self.test.hits(q))
        renum = re.sub(r"\d+", lambda m: str(int(m.group()) + 3), q)
        self.assertTrue({"masked10", "jaccard_tri50"} & set(self.test.hits(renum)), renum)
        rename = q.replace("Janet", "Maria").replace("Janet’s", "Maria’s")
        self.assertTrue({"word8", "raw13"} & set(self.test.hits(rename)), rename)
        for it in FX["items"]:
            self.assertEqual(self.test.hits(g.eval_stem(it["prompt"])), {}, it["item_id"])
        seeds = g.Seeds(["Tom has 3 red apples and buys 4 more at the market. How many apples does Tom have?",
                         "A train travels 60 miles per hour for 3 hours. How far does it go?"], held_out={1})
        self.assertTrue(q.startswith("Janet’s ducks lay 16 eggs per day."))
        for question, steps, ans in ((q, ["<<16-3-4=9>>", "<<9*2=18>>"], "18"),
                                     (renum, ["<<19-3-4=12>>", "<<12*5=60>>"], "60")):
            why, p, _ = g.verdict(0, question, steps, ans, self.test, self.val, seeds, set(), 0)
            self.assertEqual((why, p["status"]), ("test_overlap", "ok"), question)
            self.assertEqual(g.verify_row(question, steps, ans)[0], None)
        raw = FX["raw"][FX["items"][0]["item_id"]]
        why, _, _ = g.verdict(0, raw["question"], raw["steps"], raw["answer"], self.test, self.val, seeds, set(), 0)
        self.assertIsNone(why)
        seen = set()
        g.verdict(0, raw["question"], raw["steps"], raw["answer"], self.test, self.val, seeds, seen, 0)
        self.assertEqual(g.verdict(1, raw["question"], raw["steps"], raw["answer"], self.test, self.val, seeds, seen,
                                   0)[0], "duplicate_question")

    def test_val_and_seeds(self):
        g = G()
        q = self.val_t[0][0]
        self.assertIn("exact", self.val.hits(q))
        self.assertTrue(self.val.hits(re.sub(r"\d+", lambda m: str(int(m.group()) + 1), q)))
        seeds = g.Seeds(["Carl buys index cards for his class. He gives each student 10 index cards. He teaches 6 "
                         "periods a day and each class has 30 students. If a 50 pack of index cards cost $3, how much "
                         "did he spend on them all?",
                         "A train travels 60 miles per hour for 3 hours. How far does it go?"], held_out={0})
        rewrite = ("Susan buys water bottles for her gym. She gives each member 2 water bottles. She teaches 5 "
                   "classes a day and each class has 10 members. If a pack of 4 water bottles cost $2, how much did "
                   "she spend on them all?")
        self.assertIsNotNone(seeds.hit(rewrite))
        self.assertIsNone(seeds.hit("A car travels 50 miles per hour for 2 hours. How far does it go?"))   # not held


class TestSubsets(unittest.TestCase):
    def test_nested_stratified(self):
        g = G()
        rng = random.Random(5)
        index = [(f"p{i}", rng.choice([1, 2, 2, 3, 3, 3, 4, 9]), g.subset_key(0, f"p{i}")) for i in range(5000)]
        m = g.choose_subsets(index, {"24k": 300, "96k": 1200})
        self.assertEqual((len(m["24k"]), len(m["96k"])), (300, 1200))
        self.assertTrue(m["24k"] <= m["96k"])
        counts = collections.Counter(ns for _, ns, _ in index)
        for name, n in (("24k", 300), ("96k", 1200)):
            got = collections.Counter(ns for pid, ns, _ in index if pid in m[name])
            for s, c in counts.items():
                self.assertLessEqual(abs(got[s] - n * c / 5000), 1, (name, s))
        self.assertEqual(m, g.choose_subsets(list(reversed(index)), {"24k": 300, "96k": 1200}))   # order-free


@unittest.skipUnless((AUG / "summary.json").exists(), "no data/student_gsm8kaug_tf (python3 -P scripts/build_gsm8kaug_tf.py)")
class TestBuiltData(unittest.TestCase):
    ID = re.compile(r'^\{"item_id": "([^"]+)"')
    PROMPT = re.compile(r'"prompt": "((?:[^"\\]|\\.)*)"')

    def ids(self, path: Path) -> list[str]:
        out = []
        with open(path) as f:
            for line in f:
                out.append(self.ID.match(line).group(1))
        return out

    def test_summary_md5_audit(self):
        g = G()
        s = json.loads((AUG / "summary.json").read_text())
        self.assertEqual(s["audit"]["failures"], 0)
        for key, want in s["md5"].items():
            if key.endswith("train.jsonl"):
                self.assertEqual(g.md5_file(DATA / key), want, key)
        self.assertEqual(s["files"]["student_gsm8kaug_tf_24k/train.jsonl"]["items"], 24000)
        self.assertEqual(s["files"]["student_gsm8kaug_tf_96k/train.jsonl"]["items"], 96000)

    def test_nesting_on_disk(self):
        full = self.ids(AUG / "train.jsonl")
        sub = {k: self.ids(DATA / f"student_gsm8kaug_tf_{k}/train.jsonl") for k in ("24k", "96k")}
        self.assertEqual(len(set(full)), len(full))
        pos = {x: i for i, x in enumerate(full)}
        for k, xs in sub.items():
            self.assertEqual([pos[x] for x in xs], sorted(pos[x] for x in xs), k)   # the full file's order
        self.assertTrue(set(sub["24k"]) <= set(sub["96k"]) <= set(full))
        self.assertEqual((len(sub["24k"]), len(sub["96k"])), (24000, 96000))

    def test_val(self):
        for d in ("student_gsm8kaug_tf", "student_gsm8kaug_tf_24k", "student_gsm8kaug_tf_96k"):
            self.assertEqual((DATA / d / "val.jsonl").read_bytes(), (DATA / "student_gsm8k_tf_si/val.jsonl").read_bytes())
        tf = lines(DATA / "student_gsm8k_tf/val.jsonl")
        got = [json.dumps({k: v for k, v in json.loads(x).items() if k not in ("value_nodes", "worked_steps")},
                          ensure_ascii=False) for x in lines(AUG / "val.jsonl")]
        self.assertEqual(got, tf)

    def test_no_val_or_test_question_in_train(self):
        g = G()
        banned = {g.norm_text(t) for t, _, _ in g.val_targets(DATA / "student_gsm8k_tf/val.jsonl")}
        for name in ("gsm8k_mc_v2", "gsm1k_mc", "gsm8k_mc"):
            banned |= {g.norm_text(g.eval_stem(json.loads(x)["prompt"])) for x in lines(DATA / f"eval/{name}.jsonl")}
        n = 0
        with open(AUG / "train.jsonl") as f:
            for line in f:
                p = json.loads('"' + self.PROMPT.search(line).group(1) + '"')
                self.assertNotIn(g.norm_text(g.eval_stem(p)), banned)
                n += 1
        self.assertGreater(n, 96000)


@unittest.skipUnless(HAVE_TORCH and HAVE_NUMPY, "needs torch + transformers")
class TestDryConstruction(unittest.TestCase):
    """The arms' views built by cotdistill.student from 3 rows, as the trainer builds them (no model)."""
    KW = dict(mode="plain", view="statements+question", cap=0, sources=SRC)

    @classmethod
    def setUpClass(cls):
        cls.items = FX["items"][:3]

    def test_a2(self):
        from cotdistill.student import FINAL_TEMPLATE, build_examples
        for it in self.items:
            ex = build_examples(it, final="teacher", subq="none", subq_target="cot", lambda_sub=0.0,
                                rng=random.Random(0))
            self.assertEqual([e.kind for e in ex], ["final"])
            self.assertEqual(ex[0].text, FINAL_TEMPLATE.format(problem=it["prompt"]))
            self.assertEqual(ex[0].labels, list("ABCDE"))
            self.assertEqual(ex[0].target, [float(L == it["gold_label"]) for L in "ABCDE"])

    def test_gfpq_and_gfpqc(self):
        from cotdistill.student import SUBQ_TEMPLATE, build_full_tree, curriculum_item, curriculum_stage
        from cotdistill.treeplan import problem_no_options
        for it in self.items:
            ex = build_full_tree(it, rng=random.Random(1), **self.KW)
            self.assertEqual([e.kind for e in ex], ["final"] + ["subq"] * len(it["subqs"]))
            ctx = problem_no_options(it)
            for e in ex[1:]:
                self.assertTrue(e.text.startswith(ctx + "\n\nIntermediate question: "))
                self.assertEqual(e.labels, ["A", "B"])
                self.assertAlmostEqual(e.weight, 1.0 / len(it["subqs"]))
            texts = {SUBQ_TEMPLATE.format(problem=ctx, question=n["question"], a=a, b=b): (n["truth"], a == "Yes")
                     for n in it["subqs"] for a, b in (("Yes", "No"), ("No", "Yes"))}
            for e in ex[1:]:
                truth, yes_first = texts[e.text]
                self.assertEqual(e.target.index(max(e.target)), 0 if truth == yes_first else 1)
            stages = [curriculum_item(it, stage=k, n_stages=4, mode="exact") for k in range(1, 5)]
            self.assertEqual(sorted(n["id"] for s in stages for n in s["subqs"]), sorted(n["id"] for n in it["subqs"]))
            for n in it["subqs"]:
                curriculum_stage(n["step"], 4)
            for k in range(1, 5):
                build_full_tree(curriculum_item(it, stage=k, n_stages=4), rng=random.Random(1), **self.KW)

    def test_gfpqv_gfpqo(self):
        from cotdistill.student import build_full_tree
        from cotdistill.treeplan import problem_no_options
        for it in self.items:
            for render in ("op", "qop", "opsel"):
                ex = build_full_tree(it, rng=random.Random(1), node_format="value", node_render=render, **self.KW)
                want = [n for n in it["value_nodes"] if render in n["renderings"]]
                self.assertEqual([e.kind for e in ex], ["final"] + ["subq"] * len(want))
                for e, n in zip(ex[1:], want):
                    r = n["renderings"][render]
                    self.assertTrue(e.text.startswith(problem_no_options(it) + "\n\nIntermediate question: "
                                                      + r["question"] + "\nOptions:\nA) "))
                    self.assertEqual(e.labels, ["A", "B", "C"])
                    self.assertEqual(e.target, [float(i == r["answer"]) for i in range(3)])
        self.assertTrue(any("opsel" in n["renderings"] for it in FX["items"] for n in it["value_nodes"]))

    def test_sib(self):
        from cotdistill.student import CHAIN_HEADER, build_full_tree, si_group
        for it in self.items:
            g = build_full_tree(it, rng=random.Random(1), **self.KW)
            fin = [e for e in g if e.kind == "final"][0]
            ws = it["worked_steps"]
            for k in range(5):
                out = si_group(g, it, stage=k, n_stages=4, final_only=True, remove="back", seed=0, smooth=0.0)
                self.assertEqual([e.kind for e in out], ["final"])
                n = len(ws or [])
                m = -(-k * n // 4)                                   # steps removed at stage k (from the back)
                if not ws or m >= n:
                    self.assertEqual(out[0].text, fin.text)
                    continue
                head, _, tail = out[0].text.partition(f"\n\n{CHAIN_HEADER}\n")
                self.assertEqual(head, fin.text.partition("\nOptions:\n")[0])
                shown = tail.partition("\nOptions:\n")[0].split("\n")
                self.assertEqual(shown, [f"Step {i}: {ln['text']}" for i, ln in enumerate(ws[:n - m], 1)])
        self.assertTrue(any(it["worked_steps"] for it in self.items))

    def test_trainer_data_checks(self):
        """What train_student.py checks before training these arms: --tree-sources gsm8k_tf has items, value nodes
        with each rendering, a step on every tree node (GFPQC), worked steps (SI)."""
        from cotdistill.student import VALUE_FIELD, curriculum_stage, value_mc_node, worked_steps
        items = FX["items"]
        self.assertTrue(any(it.get("source") in SRC for it in items))
        for render in ("op", "qop", "opsel"):
            self.assertTrue(any(value_mc_node(sq, render) for it in items for sq in it.get(VALUE_FIELD) or []), render)
        for it in items:
            for sq in it["subqs"]:
                curriculum_stage(sq.get("step"), 4)
        self.assertTrue(any(worked_steps(it) for it in items))


if __name__ == "__main__":
    unittest.main()
