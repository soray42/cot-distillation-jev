"""GSM8K value nodes (scripts/build_gsm8k_ops.py, item field value_nodes) and the value node formats of
student.build_full_tree (--node-format value / yesno+value, --node-render op / qop / opsel, --tree-view bare; arms
GFPQV, GFPQVQ, GFPQO, GFPVB, GFPQVM):
- builder units: exact numbers, atomic splitting in evaluation order, percent / fraction / division / decimal
  rendering, reduced expressions, skips (final operation, trivial, gold-valued), option tiers, uniqueness, rank and
  gold-letter balance, operation-selection designs, the item audit;
- the built files (skipped when absent): audit clean, rows byte-identical to data/student_gsm8k_tf but for value_nodes,
  subsets with the same ids and order, md5s, gold-letter balance;
- with torch: value-node examples (prompt text, labels, one-hot target, per-problem weights), the bare view, GFPQVM's
  yes/no views and half weights, the yes/no arms (GFPQ / GFPS / GFP / GFPQR) byte-identical to the examples the code
  before value nodes built (tests/fixtures/gsm8k_tf_yesno_examples_v1.json, commit in the file), guards,
  evaluate_subq on value nodes, and the trainer end to end on a tiny random model.
CPU only. Data-backed tests read data/ of this checkout, or COTD_DATA_DIR when set."""
import ast
import collections
import dataclasses
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from cotdistill.treeplan import problem_no_options, statement_block  # noqa: E402

DATA = Path(os.environ.get("COTD_DATA_DIR") or ROOT / "data")
YESNO_FX = json.loads((ROOT / "tests/fixtures/gsm8k_tf_yesno_examples_v1.json").read_text())
OPS_FX = json.loads((ROOT / "tests/fixtures/gsm8k_ops_items_v1.json").read_text())["items"]
OPS = DATA / "student_gsm8k_tf_ops"
TF = DATA / "student_gsm8k_tf"
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

Q = ("Ann has 44 apples. She sells 25% of them. Then she buys 3 bags of 6 apples and 2 loose apples and gives away "
     "1/2 of the new apples. How many apples does Ann have now?")
A = ("How many apples does Ann sell? ** She sells 25% of 44 = <<25*.01*44=11>>11 apples.\n"
     "How many apples are left? ** She has 44 - 11 = <<44-11=33>>33 apples left.\n"
     "How many apples does she buy? ** She buys 3 * 6 + 2 = <<3*6+2=20>>20 apples.\n"
     "How many apples does she give away? ** She gives away 20 * 1/2 = <<20*(1/2)=10>>10 apples.\n"
     "How many apples does Ann have now? ** She has 33 + 20 - 10 = <<33+20-10=43>>43 apples.\n#### 43")


def rows(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@unittest.skipUnless(HAVE_NUMPY, "build_gsm8k_socratic needs numpy")
class TestBuilderUnits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import build_gsm8k_ops as B
        cls.B = B
        cls.S = B.T.soc()

    def ops(self, expr, pcts=()):
        tree = self.S.parse_expr(expr)
        ops, _, lits = self.B.split_note(tree, {Fraction(p) for p in pcts}, {})
        return [(r["kind"], r["a"]["text"], r["sym"], r["b"]["text"], self.B.fmt(r["val"])) for r in ops], tree, ops, lits

    def test_fmt(self):
        F, fmt = Fraction, self.B.fmt
        self.assertEqual([fmt(F(6)), fmt(F(3, 10)), fmt(F("1.50")), fmt(F(1, 8)), fmt(F(-5, 2)), fmt(F(1, 3))],
                         ["6", "0.3", "1.5", "0.125", "-2.5", None])
        self.assertEqual(fmt(F(1, 100000)), None)                     # more than MAX_DEC decimals
        self.assertEqual(self.ops("0.1+0.2")[0], [("bin", "0.1", "+", "0.2", "0.3")])   # no float noise
        self.assertEqual(self.ops(".5*4")[0], [("bin", "0.5", "×", "4", "2")])
        self.assertEqual(self.ops("2,000-5.0")[0], [("bin", "2000", "−", "5", "1995")])

    def test_split_in_evaluation_order(self):
        cases = {"4*1.5+3": [("bin", "4", "×", "1.5", "6"), ("bin", "6", "+", "3", "9")],
                 "4+6+8+2": [("bin", "4", "+", "6", "10"), ("bin", "10", "+", "8", "18"),
                             ("bin", "18", "+", "2", "20")],
                 "100-(20+30)": [("bin", "20", "+", "30", "50"), ("bin", "100", "−", "50", "50")],
                 "2*100+2*50": [("bin", "2", "×", "100", "200"), ("bin", "2", "×", "50", "100"),
                                ("bin", "200", "+", "100", "300")],
                 "12/3": [("bin", "12", "÷", "3", "4")],
                 "10-4": [("bin", "10", "−", "4", "6")],
                 "7/2": [("bin", "7", "÷", "2", "3.5")],
                 "(30/5)*2": [("bin", "30", "÷", "5", "6"), ("bin", "6", "×", "2", "12")]}
        for expr, want in cases.items():
            self.assertEqual(self.ops(expr)[0], want, expr)

    def test_percent_only_when_written(self):
        self.assertEqual(self.ops("4*.5", pcts=[50])[0], [("pct", "50", "% of", "4", "2")])
        self.assertEqual(self.ops("4*.5")[0], [("bin", "4", "×", "0.5", "2")])
        self.assertEqual(self.ops("(15/100)*80", pcts=[15])[0], [("pct", "15", "% of", "80", "12")])
        self.assertEqual(self.ops("(15/100)*80")[0], [("bin", "15/100", "×", "80", "12")])     # a fraction literal
        self.assertEqual(self.ops("15/100")[0], [("bin", "15", "÷", "100", "0.15")])          # alone: a division
        self.assertEqual(self.ops("25*.01*24", pcts=[25])[0], [("pct", "25", "% of", "24", "6")])
        self.assertEqual(self.ops("18*50*.01", pcts=[50])[0], [("pct", "50", "% of", "18", "9")])
        self.assertEqual(self.ops("20*30/100", pcts=[30])[0], [("pct", "30", "% of", "20", "6")])
        self.assertEqual(self.ops("100*1.2", pcts=[120])[0], [("pct", "120", "% of", "100", "120")])
        self.assertEqual(self.B.op_question("pct", "50", "% of", "4"), "What is 50% of 4?")
        self.assertEqual(self.B.written_percents("a 20% tip and 7.5 percent tax"), {Fraction(20), Fraction("7.5")})

    def test_fraction_literals(self):
        self.assertEqual(self.ops("1/2*4")[0], [("bin", "1/2", "×", "4", "2")])
        self.assertEqual(self.ops("36*(1/3)")[0], [("bin", "36", "×", "1/3", "12")])
        self.assertEqual(self.ops("(2/4)*8")[0], [("bin", "2/4", "×", "8", "4")])     # as written, not reduced
        self.assertTrue(self.B.canon(Fraction(1, 3), "1/3") and not self.B.canon(Fraction(1, 3), "0.33"))

    def test_render_and_questions(self):
        B = self.B
        for expr, want in {"4*1.5+3": ["4 × 1.5 + 3", "6 + 3"], "100-(20+30)": ["100 − (20 + 30)", "100 − 50"],
                           "4*(3-1)": ["4 × (3 − 1)", "4 × 2"], "(4+6)*2": ["(4 + 6) × 2", "10 × 2"],
                           "36*(1/3)+2": ["36 × 1/3 + 2", "12 + 2"]}.items():
            _, tree, ops, lits = self.ops(expr)
            by = {id(r["node"]): r for r in ops}
            self.assertEqual([B.render(tree, by, k, lits)[0] for k in range(len(ops))], want, expr)
        _, tree, ops, lits = self.ops("25*.01*44-3", pcts=[25])
        self.assertEqual(B.render(tree, {id(r["node"]): r for r in ops}, 0, lits)[0], "25% of 44 − 3")
        self.assertEqual(B.op_question("bin", "4", "×", "1.5"), "What is 4 × 1.5?")
        self.assertEqual(B.op_question("bin", "12", "÷", "3"), "What is 12 ÷ 3?")
        self.assertEqual(B.op_question("bin", "10", "−", "4"), "What is 10 − 4?")
        self.assertEqual(B.qop_question("How many ants does Beth see?", "4 × 1.5", "bin", "4", "×", "1.5", True),
                         "How many ants does Beth see? It is 4 × 1.5. What is the value?")
        self.assertEqual(B.qop_question("How many?", "4 × 1.5 + 3", "bin", "4", "×", "1.5", False),
                         "How many? It is 4 × 1.5 + 3. What is 4 × 1.5?")
        self.assertEqual(B.opsel_question("How many?"), "How many? Which computation gives it?")

    def nodes(self, q=Q, a=A, seed=0):
        p = self.S.parse_problem(7, q, a)
        self.assertEqual(p["status"], "ok")
        st = collections.Counter()
        return p, self.B.value_nodes(p, "gsm8k-train-7", seed, st), st

    def test_nodes(self):
        p, nodes, st = self.nodes()
        by = {n["id"]: n for n in nodes}
        self.assertEqual(sorted(by), ["s1o1", "s2o1", "s3o1", "s3o2", "s4o1", "s5o1"])  # s5o2 yields the gold
        self.assertEqual(st["ops_skipped_final_op"], 1)
        op = {i: by[i]["renderings"]["op"]["question"] for i in by}
        self.assertEqual(op, {"s1o1": "What is 25% of 44?", "s2o1": "What is 44 − 11?", "s3o1": "What is 3 × 6?",
                              "s3o2": "What is 18 + 2?", "s4o1": "What is 20 × 1/2?", "s5o1": "What is 33 + 20?"})
        self.assertEqual(by["s3o1"]["renderings"]["qop"]["question"],
                         "How many apples does she buy? It is 3 × 6 + 2. What is 3 × 6?")
        self.assertEqual(by["s3o2"]["renderings"]["qop"]["question"],
                         "How many apples does she buy? It is 18 + 2. What is the value?")
        self.assertEqual(by["s3o2"]["renderings"]["opsel"]["question"],
                         "How many apples does she buy? Which computation gives it?")
        self.assertNotIn("opsel", by["s3o1"]["renderings"])                          # not a step's last operation
        self.assertEqual(sorted(by["s5o1"]["renderings"]), ["op"])                   # gold step: op only
        self.assertEqual(st["qop_absent_gold_step_partial"], 1)
        self.assertEqual([by[i]["value"] for i in sorted(by)], ["11", "33", "18", "20", "10", "53"])
        self.assertEqual(by["s2o1"]["depends_on"], ["s1o1"])
        self.assertEqual(by["s3o2"]["depends_on"], ["s3o1"])
        self.assertEqual(by["s4o1"]["depends_on"], ["s3o2"])
        self.assertEqual(sorted(by["s5o1"]["depends_on"]), ["s2o1", "s3o2"])
        self.assertEqual([by[i]["node_depth"] for i in sorted(by)], [0, 1, 0, 1, 2, 2])
        self.assertTrue(by["s5o1"]["gold_step"] and not by["s4o1"]["gold_step"])
        self.assertEqual(by["s1o1"]["percent"], {"written_in": "sentence", "form": "compound"})
        for n in nodes:
            for k, r in n["renderings"].items():
                self.assertEqual(len(set(r["options"])), 3, (n["id"], k))
                self.assertNotIn("43", r["options"])
                if k != "opsel":
                    self.assertEqual(r["options"][r["answer"]], n["value"])
                else:
                    self.assertEqual(r["options"][r["answer"]], self.B.expr_text(n["kind"], n["a"], n["op"], n["b"]))
                    for i, x in enumerate(r["options"]):
                        v = self.B.expr_value(*self.B.parse_option(x))
                        self.assertTrue(i == r["answer"] or v != Fraction(n["value"]), (n["id"], x))
            if "qop" in n["renderings"]:
                self.assertEqual(n["renderings"]["qop"]["options"], n["renderings"]["op"]["options"])
        it = self.item(p, nodes)
        self.assertEqual(self.B.audit_item(it)[0], [])
        bad = [dict(n, renderings=dict(n["renderings"], qop=dict(n["renderings"]["op"], question="x It is 33 + 20 "
                                                                  "− 10. What is 33 + 20?")))
               if n["id"] == "s5o1" else n for n in nodes]
        self.assertTrue(any("gold step" in e for e in self.B.audit_item(self.item(p, bad))[0]))

    def item(self, p, nodes):
        opts = ["43", "53", "33", "10", "40"]
        return {"item_id": "gsm8k-train-7", "problem_id": "gsm8k-train-7", "source": "gsm8k_tf",
                "prompt": f"Question: {p['question']}\nOptions:\n" + "\n".join(f"{L}) {o}" for L, o in zip("ABCDE", opts)),
                "labels": list("ABCDE"), "gold_label": "A", "options": opts, "subqs": [], "value_nodes": nodes}

    def test_audit_catches(self):
        p, nodes, _ = self.nodes()
        for bad, frag in ((dict(nodes[1], value="34"), "is not 34"), (dict(nodes[1], a="45"), "is not"),
                          (dict(nodes[2], renderings=dict(nodes[2]["renderings"], op=dict(
                              nodes[2]["renderings"]["op"], answer=(nodes[2]["renderings"]["op"]["answer"] + 1) % 3))),
                           "s3o1/op: options")):
            it = self.item(p, [bad if n["id"] == bad["id"] else n for n in nodes])
            errs, c = self.B.audit_item(it)
            self.assertTrue(any(frag in e for e in errs), errs)

    def test_skips(self):
        a = ("How many pens? ** He has 12 * 1 = <<12*1=12>>12 pens.\n"
             "How many pens in all? ** He has 12 + 5 = <<12+5=17>>17 pens.\n#### 17")
        p, nodes, st = self.nodes("Tom has 12 pens and 5 more. How many pens in all?", a)
        self.assertEqual(nodes, [])
        self.assertEqual((st["ops_skipped_trivial"], st["ops_skipped_final_op"]), (1, 1))
        a = ("How many pens? ** He has 4 + 5 = <<4+5=9>>9 pens.\n"                    # a partial result = the gold
             "How many pens in all? ** He has 4 + 5 + 3 - 3 = <<4+5+3-3=9>>9 pens.\n#### 9")
        p, nodes, st = self.nodes("Tom has 4 pens, buys 5 and 3 and loses 3. How many pens in all?", a)
        self.assertEqual(st["steps_blocked_gold_valued"], 1)
        self.assertEqual(st["ops_blocked_gold_step_reads_blocked"], 0)
        self.assertTrue(all(n["value"] != "9" for n in nodes))
        self.assertEqual(st["ops_skipped_gold_valued_op"], 1)                        # 4 + 5 inside the gold step
        self.assertEqual(st["ops_skipped_downstream_of_gold_valued_op"], 1)          # (4 + 5) + 3 uses it

    def test_shared_subquestion_has_no_qop(self):
        a = ("How many floors? ** It has 12 - 1 = <<12-1=11>>11 floors.\n"
             "How many floors? ** It has 11 + 12 = <<11+12=23>>23 floors.\n"
             "How many rooms? ** It has 23 * 4 = <<23*4=92>>92 rooms.\n#### 92")
        p, nodes, st = self.nodes("A tower has 12 floors minus 1 and then 12 more; each floor has 4 rooms. How many "
                                  "rooms?", a)
        by = {n["id"]: n for n in nodes}
        self.assertEqual(sorted(by), ["s1o1", "s2o1"])
        for n in nodes:
            self.assertEqual(sorted(n["renderings"]), ["op"])
        self.assertEqual(st["qop_opsel_absent_subq_shared_other_value"], 2)

    def test_value_options_tiers_balance(self):
        B, F = self.B, Fraction
        t, gold = F(6), F(43)
        ranks, kinds, letters = collections.Counter(), collections.Counter(), collections.Counter()
        for s in range(600):
            opts, ans, wrong = B.value_options(t, B.operator_slips("bin", F(4), "×", F("1.5")), [F(11), F(33)], gold,
                                               random.Random(s))
            vals = [F(x) for x in opts]
            self.assertEqual(len(set(opts)), 3)
            self.assertEqual(opts[ans], "6")
            self.assertNotIn("43", opts)
            self.assertIsNone(wrong[ans])
            ranks[sorted(vals).index(t)] += 1
            if sorted(vals).index(t) > 0:              # a wrong value below the truth: the tier-1 one (4 × 0.5 = 2)
                self.assertEqual(wrong[opts.index("2")] if "2" in opts else None, "complement")
            letters[ans] += 1
            kinds.update(w for w in wrong if w)
            for v in vals:
                self.assertTrue(v.denominator == 1)                                  # an integer truth: integers
                self.assertLessEqual(max(v / t, t / v), 10)
        for counts in (ranks, letters):
            self.assertEqual(sorted(counts), [0, 1, 2])
            self.assertTrue(all(150 <= c <= 250 for c in counts.values()), counts)
        self.assertGreater(kinds["other"], kinds["perturb"])          # above the truth: tier 2 before tier 3
        slips = dict((B.fmt(v), k) for v, k in B.operator_slips("bin", F(4), "×", F("1.5")))
        self.assertEqual(slips.get("5.5"), "operator")                               # 4 + 1.5
        self.assertEqual(slips.get("2"), "complement")                               # 4 × 0.5
        pct = dict((B.fmt(v), k) for v, k in B.operator_slips("pct", F(20), "% of", F(30)))
        self.assertEqual((pct.get("24"), pct.get("36")), ("complement", "complement"))

    def test_opsel_designs(self):
        B, F = self.B, Fraction
        seen = collections.Counter()
        for s in range(200):
            o, a, w, design = B.opsel_options("bin", F(4), "4", "×", F("1.5"), "1.5", F(6),
                                              {"a": [F(3), F(10)], "b": [F(3), F(10), F("0.5")]}, F(42),
                                              random.Random(s))
            seen[design] += 1
            self.assertEqual(o[a], "4 × 1.5")
            self.assertEqual(len(set(o)), 3)
            self.assertEqual(len({x for x in w if x}), 1)                            # one design per node
            for i, x in enumerate(o):
                if i != a:
                    self.assertNotEqual(B.expr_value(*B.parse_option(x)), F(6))
        self.assertEqual(set(seen), {"operator", "operand"})


@unittest.skipUnless((OPS / "summary.json").exists() and HAVE_NUMPY, "data/student_gsm8k_tf_ops not built (or no numpy)")
class TestBuiltData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import build_gsm8k_ops as B
        cls.B = B
        cls.s = json.loads((OPS / "summary.json").read_text())

    def test_audit_and_md5(self):
        s = self.s
        self.assertEqual(s["audit"]["failures"], 0)
        self.assertEqual(s["audit"].get("op_mismatch", 0), 0)
        self.assertEqual(s["audit"].get("step_chain_mismatch", 0), 0)
        self.assertEqual(s["audit"]["ops_checked"], s["build"]["nodes"])
        self.assertEqual((s["leaks"]["value_option_equals_gold"], s["leaks"]["node_value_equals_gold"]), (0, 0))
        for name, want in s["md5"].items():
            self.assertEqual(self.B.T.md5(DATA / name), want, name)
        for name, want in s["source_md5"].items():
            self.assertEqual(self.B.T.md5(ROOT / name), want, name)

    def test_rows_identical_but_value_nodes(self):
        for split in ("train", "val"):
            with open(TF / f"{split}.jsonl") as a, open(OPS / f"{split}.jsonl") as b:
                la = [x for x in a.read().split("\n") if x]               # not splitlines: U+2028 in texts
                lb = [x for x in b.read().split("\n") if x]
            self.assertEqual(len(la), len(lb))
            for x, y in zip(la, lb):
                r = json.loads(y)
                self.assertEqual(list(r)[-1], "value_nodes")
                del r["value_nodes"]
                self.assertEqual(json.dumps(r, ensure_ascii=False), x)

    def test_subsets(self):
        val = (OPS / "val.jsonl").read_bytes()
        for fr in (10, 25, 50):
            src = [r["item_id"] for r in rows(DATA / f"student_gsm8k_tf_f{fr}/train.jsonl")]
            got = [r["item_id"] for r in rows(DATA / f"student_gsm8k_tf_ops_f{fr}/train.jsonl")]
            self.assertEqual(got, src, fr)
            self.assertEqual((DATA / f"student_gsm8k_tf_ops_f{fr}/val.jsonl").read_bytes(), val)

    def test_balance_and_options(self):
        train = rows(OPS / "train.jsonl")
        for k in ("op", "qop", "opsel"):
            c = collections.Counter(n["renderings"][k]["answer"] for it in train for n in it["value_nodes"]
                                    if k in n["renderings"])
            tot = sum(c.values())
            self.assertGreater(tot, 5000)
            for L in range(3):
                self.assertLess(abs(c[L] / tot - 1 / 3), 0.03, (k, c))
        for it in train[:2000]:
            gold = it["options"][it["labels"].index(it["gold_label"])]
            for n in it["value_nodes"]:
                for k, r in n["renderings"].items():
                    self.assertEqual(len(set(r["options"])), 3)
                    if k != "opsel":
                        self.assertNotIn(gold, r["options"])
                        self.assertEqual(r["options"][r["answer"]], n["value"])
        cues = self.s["cues_train"]
        for k, v in cues.items():
            if k.startswith("op/") and "max" in v and v["n"] > 500:
                self.assertLess(max(v["max"], v["min"], v["median"]), 0.42, (k, v))
        self.assertLess(cues["opsel/centre"]["acc"], 0.36)

    def test_node_audit_file(self):
        txt = (OPS / "node_audit.txt").read_text()
        self.assertTrue(txt.startswith("=== gsm8k-train-660"))
        self.assertEqual(txt.count("\n=== "), 4)
        for k in ("op ", "qop ", "opsel "):
            self.assertIn(f"    {k}", txt)


def fx_items():
    return [json.loads(json.dumps(it)) for it in OPS_FX]


def asdicts(exs):
    return [dataclasses.asdict(e) for e in exs]


@unittest.skipUnless(HAVE_TORCH, "needs torch + transformers")
class TestValueViews(unittest.TestCase):
    YESNO_ARMS = {"GFPQ": dict(mode="plain", view="statements+question"), "GFPS": dict(mode="plain", view="statements"),
                  "GFP": dict(mode="plain", view="problem"),
                  "GFPQR": dict(mode="plain", view="statements+question", target_seed=12)}

    def build(self, items, **kw):
        from cotdistill.student import build_full_tree
        rng, out = random.Random(12), []
        for it in items:
            if kw.get("view") == "statements":
                try:
                    statement_block(it)
                except ValueError:
                    continue
            out.append(build_full_tree(it, rng=rng, cap=0, sources=SRC, **kw))
        return out

    def test_yesno_arms_byte_identical(self):
        """GFPQ / GFPS / GFP / GFPQR on the fixture rows, and on the same rows with value nodes, equal the examples of
        the code before value nodes (commit in the fixture)."""
        plain = YESNO_FX["items"]
        for arm, kw in self.YESNO_ARMS.items():
            self.assertEqual([asdicts(x) for x in self.build(plain, **kw)], YESNO_FX["examples"][arm], arm)
            self.assertEqual([asdicts(x) for x in self.build(fx_items(), **kw)], YESNO_FX["examples"][arm], arm)

    def test_yesno_digest_first300(self):
        p = TF / "train.jsonl"
        if not p.exists():
            self.skipTest("no data/student_gsm8k_tf")
        from cotdistill.student import build_full_tree
        with open(p) as f:
            its = [json.loads(x) for _, x in zip(range(300), f)]
        for arm in ("GFPQ", "GFPQR"):
            rng, h = random.Random(12), hashlib.md5()
            for it in its:
                for e in build_full_tree(it, rng=rng, cap=0, sources=SRC, **self.YESNO_ARMS[arm]):
                    h.update(json.dumps(dataclasses.asdict(e), sort_keys=True).encode())
            self.assertEqual(h.hexdigest(), YESNO_FX["digest_first300_train"][arm], arm)

    def test_value_examples(self):
        from cotdistill.student import build_full_tree
        it = {x["item_id"]: x for x in fx_items()}["gsm8k-train-660"]
        ctx = problem_no_options(it)
        nodes = it["value_nodes"]
        for render in ("op", "qop", "opsel"):
            ex = build_full_tree(it, mode="plain", view="statements+question", rng=random.Random(0), cap=0,
                                 sources=SRC, node_format="value", node_render=render)
            asked = [n for n in nodes if render in n["renderings"]]
            self.assertEqual(len(ex), 1 + len(asked))
            self.assertEqual(ex[0].kind, "final")
            self.assertEqual(ex[0].text, it["prompt"] + "\n\nAnswer:")
            for e, n in zip(ex[1:], asked):
                r = n["renderings"][render]
                want = (f"{ctx}\n\nIntermediate question: {r['question']}\nOptions:\n"
                        + "\n".join(f"{L}) {o}" for L, o in zip("ABC", r["options"])) + "\nAnswer:")
                self.assertEqual(e.text, want)
                self.assertEqual(e.labels, ["A", "B", "C"])
                self.assertEqual(e.target, [float(i == r["answer"]) for i in range(3)])
                self.assertAlmostEqual(e.weight, 1.0 / len(asked))
                self.assertEqual(e.kind, "subq")
        e = build_full_tree(it, mode="plain", view="statements+question", rng=random.Random(0), cap=0, sources=SRC,
                            node_format="value")[1]
        self.assertEqual(e.text, f"{ctx}\n\nIntermediate question: What is 4 + 2?\nOptions:\nA) 4\nB) 2\nC) 6\nAnswer:")
        self.assertEqual(e.target, [0.0, 0.0, 1.0])
        bare = build_full_tree(it, mode="plain", view="bare", rng=random.Random(0), cap=0, sources=SRC,
                               node_format="value")
        self.assertEqual(bare[1].text, "Intermediate question: What is 4 + 2?\nOptions:\nA) 4\nB) 2\nC) 6\nAnswer:")
        self.assertEqual(bare[0].text, ex[0].text)
        other = build_full_tree(it, mode="plain", view="bare", rng=random.Random(0), cap=0, sources={"numgen"},
                                node_format="value")
        self.assertEqual(len(other), 1)

    def test_gfpqvm(self):
        """GFPQVM: GFPQ's yes/no views (same order, text, target) at half weight, then GFPQV's value views at half
        weight; per problem total aux_total; one format only: the whole."""
        vm = self.build(fx_items(), mode="plain", view="statements+question", node_format="yesno+value")
        v = self.build(fx_items(), mode="plain", view="statements+question", node_format="value")
        for g, gq, gv in zip(vm, YESNO_FX["examples"]["GFPQ"], v):
            n_yn = len(gq) - 1
            yn, val = g[1:1 + n_yn], g[1 + n_yn:]
            self.assertEqual(asdicts(g[:1]), gq[:1])
            for e, d in zip(yn, gq[1:]):
                self.assertEqual(dataclasses.asdict(dataclasses.replace(e, weight=d["weight"])), d)
                self.assertEqual(e.weight, d["weight"] * (0.5 if val else 1.0))
            self.assertEqual([e.text for e in val], [e.text for e in gv[1:]])
            for e, f in zip(val, gv[1:]):
                self.assertAlmostEqual(e.weight, f.weight * (0.5 if n_yn else 1.0))
            self.assertAlmostEqual(sum(e.weight for e in g[1:]), 1.0)
        it = fx_items()[0]
        only_yn = dict(it, value_nodes=[])
        from cotdistill.student import build_full_tree
        g = build_full_tree(only_yn, mode="plain", view="statements+question", rng=random.Random(1), cap=0,
                            sources=SRC, node_format="yesno+value")
        self.assertAlmostEqual(sum(e.weight for e in g[1:]), 1.0)
        only_v = dict(it, subqs=[])
        g = build_full_tree(only_v, mode="plain", view="statements+question", rng=random.Random(1), cap=0,
                            sources=SRC, node_format="yesno+value")
        self.assertEqual(len(g) - 1, len(it["value_nodes"]))
        self.assertAlmostEqual(sum(e.weight for e in g[1:]), 1.0)

    def test_guards(self):
        from cotdistill.student import build_full_tree, value_mc_node
        it = fx_items()[-1]
        for kw in (dict(mode="true", view="statements+question", node_format="value"),
                   dict(mode="plain", view="local", node_format="value"),
                   dict(mode="plain", view="bare", node_format="yesno+value"),
                   dict(mode="plain", view="bare"),
                   dict(mode="plain", view="statements+question", node_format="value", node_render="sq"),
                   dict(mode="plain", view="statements+question", node_format="value", target_seed=3)):
            with self.assertRaises(ValueError, msg=str(kw)):
                build_full_tree(it, rng=random.Random(0), cap=0, sources=SRC, **kw)
        n = [x for x in it["value_nodes"] if "opsel" not in x["renderings"]][0]
        self.assertIsNone(value_mc_node(n, "opsel"))
        self.assertEqual(value_mc_node(n, "op")["type"], "value_op")

    def test_eval_subq(self):
        """evaluate_subq scores value nodes in the given rendering (kind value, groups all_value / value/mc /
        value/depth/<d>); the yes/no predictions equal those of the items without value nodes."""
        import test_student as ts
        from cotdistill.student import evaluate_subq
        tok, model = ts.tiny()
        model.eval()
        its = fx_items()[-2:]
        base = [{k: v for k, v in it.items() if k != "value_nodes"} for it in its]
        m0, p0 = evaluate_subq(model, tok, base, 512, 4, {}, seed=5)
        for render in ("op", "opsel"):
            m, p = evaluate_subq(model, tok, its, 512, 4, {}, seed=5, value_render=render)
            val = [x for x in p if x["kind"] == "value"]
            n = sum(render in sq["renderings"] for it in its for sq in it["value_nodes"])
            self.assertEqual(len(val), n)
            self.assertEqual(m["all_value"]["n"], n)
            self.assertEqual(m["value/mc"]["n"], n)
            self.assertEqual(m[f"value/type/value_{render}"]["n"], n)
            self.assertNotIn("cot/mc", m)
            self.assertEqual(m["all_cot"], m0["all_cot"])
            yn = [x for x in p if x["kind"] != "value"]
            self.assertEqual([(x["question"], x["yes_first"]) for x in yn], [(x["question"], x["yes_first"]) for x in p0])
            for x in val:
                self.assertEqual((len(x["probs"]), x["render"]), (3, render))
                self.assertEqual(x["correct"], int(x["pred"] == x["answer"]))
        m, p = evaluate_subq(model, tok, its, 512, 4, {}, seed=5, value_render=None)
        self.assertNotIn("all_value", m)


@unittest.skipUnless(HAVE_TORCH, "needs torch + transformers")
class TestTrainValue(unittest.TestCase):
    def test_arms_end_to_end(self):
        """GFPQV, GFPVB and GFPQVM through main() on four fixture rows: they run, count their value views, the node
        losses are finite and --eval-subq scores the value nodes; the CLI guards refuse bad combinations."""
        from test_tree_forcing import ENV, SCRIPT, tiny
        arms = {"GFPQV": ["--tree-view", "statements+question", "--node-format", "value"],
                "GFPVB": ["--tree-view", "bare", "--node-format", "value"],
                "GFPQVM": ["--tree-view", "statements+question", "--node-format", "yesno+value"],
                "GFPQO": ["--tree-view", "statements+question", "--node-format", "value", "--node-render", "opsel"]}
        with tempfile.TemporaryDirectory() as d:
            tiny(d)
            items = fx_items()[-4:]
            Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in items))
            common = [sys.executable, SCRIPT, "--model", d + "/m", "--train", d + "/t.jsonl", "--eval",
                      "v=" + d + "/t.jsonl", "--eval-subq", d + "/t.jsonl", "--epochs", "1", "--items-per-update",
                      "2", "--micro-bs", "4", "--max-len", "512", "--eval-max-len", "512", "--precision",
                      "fp32master", "--no-grad-ckpt", "--log-every", "1", "--lr", "1e-3", "--tree-full", "plain",
                      "--tree-cap", "0", "--tree-sources", "gsm8k_tf", "--subq", "cot", "--subq-target", "cot"]
            for arm, extra in arms.items():
                r = subprocess.run(common + ["--out", f"{d}/{arm}", *extra], capture_output=True, text=True,
                                   timeout=900, env=ENV)
                self.assertEqual(r.returncode, 0, (arm, r.stderr[-2000:]))
                m = json.loads(Path(f"{d}/{arm}/metrics.json").read_text())
                render = extra[extra.index("--node-render") + 1] if "--node-render" in extra else "op"
                n_val = sum(render in n["renderings"] for it in items for n in it["value_nodes"])
                self.assertEqual(m["tree_view_stats"]["value_views"], n_val, arm)
                n_yn = sum(len(it["subqs"]) for it in items) if arm == "GFPQVM" else 0
                self.assertEqual(m["tree_view_stats"].get("views", 0), n_yn, arm)
                self.assertTrue(all(math.isfinite(h["loss"]) for h in m["history"]))
                self.assertEqual(m["eval"]["subq"]["all_value"]["n"], n_val, arm)
            for bad in (["--tree-view", "bare"], ["--node-render", "qop"],
                        ["--tree-view", "local", "--node-format", "value"],
                        ["--tree-view", "statements+question", "--node-format", "value", "--tree-target", "shuffled"]):
                r = subprocess.run(common + ["--out", f"{d}/bad", *bad], capture_output=True, text=True, timeout=300,
                                   env=ENV)
                self.assertNotEqual(r.returncode, 0, bad)
            Path(d + "/t2.jsonl").write_text("\n".join(json.dumps({k: v for k, v in x.items() if k != "value_nodes"})
                                                       for x in items))
            r = subprocess.run([x if x != d + "/t.jsonl" else d + "/t2.jsonl" for x in common]
                               + ["--out", f"{d}/bad2", "--tree-view", "statements+question", "--node-format", "value"],
                               capture_output=True, text=True, timeout=600, env=ENV)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("has value nodes", r.stderr + r.stdout)


if __name__ == "__main__":
    unittest.main()
