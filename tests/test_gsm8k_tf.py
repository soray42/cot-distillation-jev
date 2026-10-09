"""GSM8K Tree Forcing data (scripts/build_gsm8k_tf.py, source gsm8k_tf) and its treeplan branch:
- treeplan: problem_no_options (GFPQ) for every gsm8k_tf prompt, statement_block only on bare final questions, shape
  refusals; every other source byte-identical to BASE_REF (git show of treeplan.py; fixtures and every local item of the
  files below);
- builder units on synthetic Socratic solutions: nodes, wrong values, blocks and drops, the item audit;
- the built files (skipped when absent): every node re-verified (audit_item), Yes/No balance, splits, le4 view,
  per-step evaluation files, dedup, finals byte-identical to data/student_gsm8k_soc but for the tree fields;
- with torch: the GFPQ views of build_full_tree are the problem views with the prompt cut before its options.
CPU only. Data-backed tests read data/ of this checkout, or COTD_DATA_DIR when set."""
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from cotdistill.treeplan import GSM8K_TF, gsm8k_question_cut, problem_no_options, statement_block  # noqa: E402

DATA = Path(os.environ.get("COTD_DATA_DIR") or ROOT / "data")
BASE_REF = "2509b67"          # the commit before the gsm8k_tf branch (pinned: HEAD would compare the branch with itself;
#                               move the pin when another source's branch of treeplan.py changes)

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

CLEAN = "Carla has 3 apples. She buys 4 more apples at the store. How many apples does Carla have in her basket?"
CLEAN2 = "Carla has 3 apples. She buys 4 more apples at the store. How many apples does Carla have?"
PREMISE = "Carla has 3 apples. If she buys 4 more, how many apples does she have?"
OPTS = "A) 7\nB) 12\nC) 1\nD) 8\nE) 6"


def gsm(problem: str, source: str = GSM8K_TF, opts: str = OPTS) -> dict:
    return {"item_id": "gsm8k-train-x", "source": source, "prompt": f"Question: {problem}\nOptions:\n{opts}"}


def outcome(fn, item):
    try:
        return ("ok", fn(item))
    except ValueError:
        return ("raise", None)


def old_treeplan():
    """treeplan.py at BASE_REF as a standalone module (None outside a git checkout)."""
    try:
        src = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE_REF}:src/cotdistill/treeplan.py"],
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    d = tempfile.mkdtemp()
    path = Path(d) / "treeplan_base.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("treeplan_base", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OTHER_FILES = ["student_v3sl/val.jsonl", "student_v3sl/train.jsonl", "student_numgen_tf/val.jsonl",
               "student_numgen_tf/train.jsonl", "student_pronto_tf/val.jsonl", "student_goldproof/val.jsonl",
               "student_rule/val.jsonl", "student_fa0/val.jsonl", "student_fa1/val.jsonl",
               "student_gsm8k_soc/val.jsonl", "student_v8/val.jsonl", "eval/kk_heldout.jsonl",
               "eval/gsm8k_mc_v2.jsonl", "eval/numgen_d5.jsonl"]


class TestTreeplanBranch(unittest.TestCase):
    def test_gfpq_every_item_gfps_only_bare_questions(self):
        for prob in (CLEAN, CLEAN2, PREMISE):
            self.assertEqual(problem_no_options(gsm(prob)), f"Question: {prob}")
        self.assertEqual(statement_block(gsm(CLEAN)), "Carla has 3 apples. She buys 4 more apples at the store.")
        self.assertEqual(statement_block(gsm(CLEAN2)), "Carla has 3 apples. She buys 4 more apples at the store.")
        with self.assertRaises(ValueError):
            statement_block(gsm(PREMISE))

    def test_question_cut(self):
        bare = ["How many apples does Carla have?", "What is the total cost of the shoes?",
                "How much money does Betty need to buy the wallet?"]
        carry = ["If she sells 3, how many are left?", "How many pages does he read in 5 days?",
                 "How many apples does she have after eating two?", "How much does she pay, in dollars?",
                 "When does the train arrive?", "How much does each cost if they split it?",
                 "Calculate the total number of apples.", "How many apples does she have left now?",
                 "How many apples does Carla have in total in her basket and in her bag at 50% off?"]
        for ask in bare:
            self.assertEqual(gsm8k_question_cut(f"Carla has 3 apples. {ask}"), ("Carla has 3 apples.", ask), ask)
        for ask in carry:
            self.assertIsNone(gsm8k_question_cut(f"Carla has 3 apples. {ask}"), ask)
        self.assertIsNone(gsm8k_question_cut("How many apples does Carla have?"))            # nothing before it
        self.assertIsNone(gsm8k_question_cut("Is it red? Carla has 3 apples. How many apples does she have?"))

    def test_refusals(self):
        bad = [{"item_id": "x", "source": GSM8K_TF, "prompt": f"Question: {CLEAN}"},                  # no options
               gsm(CLEAN, opts=OPTS + "\nOptions:\nA) 2"),                                          # two blocks
               {"item_id": "x", "source": GSM8K_TF, "prompt": f"{CLEAN}\nOptions:\n{OPTS}"},         # no label
               gsm("Carla has 3 apples.\nHow many apples does Carla have?"),                        # two lines
               gsm(f"{CLEAN} Question: How many?"), gsm("  ")]
        for it in bad:
            for fn in (statement_block, problem_no_options):
                with self.assertRaises(ValueError, msg=repr(it["prompt"][:60])):
                    fn(it)

    def test_gsm8k_soc_still_refused(self):
        for prob in (CLEAN, PREMISE):
            for fn in (statement_block, problem_no_options):
                with self.assertRaises(ValueError):
                    fn(gsm(prob, source="gsm8k_soc"))

    def test_other_sources_unchanged_against_base_ref(self):
        old = old_treeplan()
        if old is None:
            self.skipTest("not a git checkout")
        import test_statement_block as tsb
        fixtures = [tsb.KK, tsb.NUMGEN] + [tsb.new_item(s, *v) for s, v in tsb.NEW.items()] + [
            gsm(CLEAN, source="gsm8k_soc"), gsm(CLEAN, source="arc_challenge"), gsm(CLEAN, source=None)]
        for it in fixtures:
            for name in ("statement_block", "problem_no_options"):
                self.assertEqual(outcome(getattr(sys.modules["cotdistill.treeplan"], name), it),
                                 outcome(getattr(old, name), it), f"{it.get('source')} {name}")
        n, sources = 0, set()
        for rel in OTHER_FILES:
            p = DATA / rel
            if not p.exists():
                continue
            with open(p) as f:
                for line in f:
                    it = json.loads(line)
                    if it.get("source") == GSM8K_TF:
                        continue
                    for name in ("statement_block", "problem_no_options"):
                        self.assertEqual(outcome(getattr(sys.modules["cotdistill.treeplan"], name), it),
                                         outcome(getattr(old, name), it), f"{rel} {it.get('item_id')} {name}")
                    n += 1
                    sources.add(it.get("source"))
        if not n:
            self.skipTest(f"no local data under {DATA}")
        self.assertGreater(len(sources), 10)


# ---- builder units -----------------------------------------------------------------------------------------------
Q = ("Tom buys 3 boxes of 6 pens each. He gives away 5 pens. Then he buys twice as many pens as he has left. "
     "How many pens does Tom have now?")
S1, S2, S3, S4 = ("How many pens does Tom buy?", "How many pens does Tom have after giving some away?",
                  "How many more pens does Tom buy?", "How many pens does Tom have now?")
A = (f"{S1} ** Tom buys 3 * 6 = <<3*6=18>>18 pens.\n"
     f"{S2} ** He has 18 - 5 = <<18-5=13>>13 pens.\n"
     f"{S3} ** He buys 13 * 2 = <<13*2=26>>26 more pens.\n"
     f"{S4} ** He has 13 + 26 = <<13+26=39>>39 pens.\n#### 39")


@unittest.skipUnless(HAVE_NUMPY, "build_gsm8k_socratic needs numpy")
class TestBuilderUnits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import build_gsm8k_tf as B
        cls.B = B
        cls.S = B.soc()

    def nodes(self, q=Q, a=A, seed=0):
        p = self.S.parse_problem(0, q, a)
        self.assertEqual(p["status"], "ok")
        st = __import__("collections").Counter()
        return p, self.B.tf_nodes(p, random.Random(seed), st), st

    def item(self, q=Q, a=A, seed=0):
        p, nodes, _ = self.nodes(q, a, seed)
        gold = self.B._fmt(p["gold"])
        opts = [gold, "20", "22", "14", "7"]
        it = {"item_id": "gsm8k-train-0", "problem_id": "gsm8k-train-0", "source": "gsm8k_soc",
              "prompt": f"Question: {q}\nOptions:\n" + "\n".join(f"{L}) {o}" for L, o in zip("ABCDE", opts)),
              "labels": list("ABCDE"), "gold_label": "A", "options": opts, "depth": len(p["steps"]),
              "tree_fix": True, "subqs": []}
        return self.B.convert(it, p, random.Random(seed), __import__("collections").Counter())

    def test_nodes(self):
        p, nodes, st = self.nodes()
        by = {n["id"]: n for n in nodes}
        self.assertEqual(sorted(by), ["s1n", "s1v", "s2n", "s2v", "s3n", "s3v"])       # s4 = the gold step
        self.assertEqual(by["s1v"]["question"], "How many pens does Tom buy? Is it 18?")
        self.assertEqual(by["s1v"]["opposite"], "How many pens does Tom buy? Is it different from 18?")
        self.assertEqual([by[f"s{j}v"]["truth"] for j in (1, 2, 3)], [True] * 3)
        self.assertEqual([by[f"s{j}n"]["truth"] for j in (1, 2, 3)], [False] * 3)
        self.assertEqual(by["s2v"]["depends_on"], ["s1v"])
        self.assertEqual(by["s3n"]["depends_on"], ["s2v"])
        self.assertEqual([by[f"s{j}v"]["node_depth"] for j in (1, 2, 3)], [0, 1, 2])
        for n in nodes:
            self.assertNotEqual(self.B._fmt(n["probe"]), "39")
            self.assertEqual(n["flags"], [])
            self.assertTrue(n["gold_ancestor"])
            if not n["truth"]:
                self.assertIn(n["wrong"], ("slip", "other", "perturb"))
                self.assertTrue(self.B.ok_wrong(n["probe"], n["value"], 39.0))

    def test_wrong_values_balanced_and_clean(self):
        above, kinds = 0, __import__("collections").Counter()
        for seed in range(200):
            _, nodes, _ = self.nodes(seed=seed)
            for n in nodes:
                if not n["truth"]:
                    above += n["probe"] > n["value"]
                    kinds[n["wrong"]] += 1
                    self.assertEqual(n["probe"], round(n["probe"]))
        self.assertTrue(0.4 < above / 600 < 0.6, above)
        self.assertEqual(set(kinds), {"slip", "other", "perturb"})

    def test_slips(self):
        p, _, _ = self.nodes()
        # step 2 = (3 * 6) - 5 with 3 * 6 = step 1: in step 1 3/6 -> -4.5, 2*6 -> 7, 4*6 -> 19, 3*5 -> 10, 3*7 -> 16; in
        # step 2 18+5 -> 23, 18-6 -> 12, 18-4 -> 14 (the parent's result is never bumped)
        self.assertEqual(set(self.B.slip_values(p, 1)), {-4.5, 7.0, 19.0, 10.0, 16.0, 23.0, 12.0, 14.0})
        self.assertEqual(set(self.B.slip_values(p, 0)), {0.5, 12.0, 24.0, 15.0, 21.0})

    def test_perturbs(self):
        self.assertEqual(self.B.perturbs(37, True), [39, 41, 47])
        self.assertEqual(self.B.perturbs(40, False), [30, 20, -10, -60])
        self.assertEqual(self.B.perturbs(2.5, True), [3.5, 4.5, 7.5, 12.5])
        self.assertFalse(self.B.ok_wrong(21, 20, 21))          # the gold
        self.assertFalse(self.B.ok_wrong(20.5, 20, 30))        # integer value, non-integer probe
        self.assertFalse(self.B.ok_wrong(201, 20, 30))         # beyond 10x
        self.assertFalse(self.B.ok_wrong(1 / 3, 0.5, 30))      # not exact at 2 decimals
        self.assertTrue(self.B.ok_wrong(0.25, 0.5, 30))

    def test_blocks_and_drops(self):
        # step 2 equals the gold (blocked) and step 3 reads it (blocked downstream)
        q = "Ann has 4 bags of 5 apples. She eats 0 of them. Then she doubles them and gives half away. How many?"
        a = ("How many apples does Ann have? ** 4 * 5 = <<4*5=20>>20\n"
             "How many apples are left? ** 20 - 0 = <<20-0=20>>20\n"
             "How many after doubling? ** 20 * 2 = <<20*2=40>>40\n"
             "How many are left at the end? ** 40 / 2 = <<40/2=20>>20\n#### 20")
        p, nodes, st = self.nodes(q, a)
        self.assertEqual(nodes, [])
        self.assertEqual(st["blocked_gold_valued"], 2)
        self.assertEqual(st["blocked_downstream_of_gold_valued"], 1)
        # an uncertain edge (12 is a unit constant: no parent link) to a gold-valued step still blocks
        q6 = "Sam has 3 bags with 4 apples each. He buys 5 more apples and then gives 5 apples away. How many now?"
        a6 = ("How many apples are in the bags? ** 3 * 4 = <<3*4=12>>12\n"
              "How many apples after buying? ** 12 + 5 = <<12+5=17>>17\n"
              "How many apples does he have now? ** 17 - 5 = <<17-5=12>>12\n#### 12")
        p, nodes, st = self.nodes(q6, a6)
        self.assertIn("parents_uncertain", p["steps"][1]["flags"])
        self.assertEqual((nodes, st["blocked_uncertain_edge_to_gold_valued"]), ([], 1))
        # a sub-question reused for a partial result, and one that restates the final question
        a2 = A.replace(S2, S1)
        _, nodes, st = self.nodes(Q, a2)
        self.assertEqual(sorted({n["step"] for n in nodes}), [3])
        self.assertEqual(st["dropped_subq_shared_other_value"], 2)
        a3 = A.replace(S3, S4)
        _, nodes, st = self.nodes(Q, a3)
        self.assertEqual(sorted({n["step"] for n in nodes}), [1, 2])
        a4 = A.replace(S4, "How many pens does Tom end up with?").replace(S3, "How many pens does Tom have, now")
        _, nodes, st = self.nodes(Q, a4)
        self.assertEqual((st["dropped_subq_is_final_question"], st["dropped_subq_shared_other_value"]), (1, 0))
        self.assertEqual(sorted({n["step"] for n in nodes}), [1, 2])
        # a value the formatter cannot print exactly
        q5 = "Ben splits 10 cakes among 3 friends and then 3 more friends join. How many cakes in all?"
        a5 = ("How much does each friend get? ** 10 / 3 = <<10/3=3.3333333333>>3.33\n"
              "How many friends are there in all? ** 3 + 3 = <<3+3=6>>6\n"
              "How many cakes in all? ** 6 + 10 = <<6+10=16>>16\n#### 16")
        _, nodes, st = self.nodes(q5, a5)
        self.assertEqual(st["dropped_value_not_exact2"], 1)
        self.assertEqual(sorted({n["step"] for n in nodes}), [2])

    def test_audit(self):
        it = self.item()
        self.assertEqual(it["source"], GSM8K_TF)
        self.assertEqual((it["n_steps"], it["group"], it["has_tree"]), (4, "gsm8k_tf/s4", True))
        self.assertEqual(self.B.audit_item(it), [])
        bad = json.loads(json.dumps(it))
        bad["subqs"][0]["truth"] = False
        self.assertTrue(self.B.audit_item(bad))
        bad = json.loads(json.dumps(it))
        bad["subqs"][2]["expr"] = "12-6"
        self.assertTrue(any("expression" in x for x in self.B.audit_item(bad)))
        bad = json.loads(json.dumps(it))
        bad["subqs"][2]["depends_on"] = ["s3v"]
        self.assertTrue(self.B.audit_item(bad))
        bad = json.loads(json.dumps(it))
        bad["subqs"][1]["probe"] = 39.0
        bad["subqs"][1]["question"] = "How many pens does Tom buy? Is it 39?"
        self.assertTrue(any("gold" in x for x in self.B.audit_item(bad)))


# ---- the built files ---------------------------------------------------------------------------------------------
TF = DATA / "student_gsm8k_tf"
LE4 = DATA / "student_gsm8k_tf_le4"


def rows(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@unittest.skipUnless((TF / "train.jsonl").exists() and HAVE_NUMPY, "data/student_gsm8k_tf not built (or no numpy)")
class TestBuiltData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import build_gsm8k_tf as B
        cls.B = B
        cls.train, cls.val = rows(TF / "train.jsonl"), rows(TF / "val.jsonl")

    def test_every_node_verified(self):
        fails = [f for it in self.train + self.val for f in self.B.audit_item(it)]
        self.assertEqual(fails[:5], [])
        summ = json.loads((TF / "summary.json").read_text())
        self.assertEqual(summ["audit"]["failures"], 0)
        self.assertEqual(summ["leaks"]["yes_node_of_gold_step_or_gold_value"], 0)
        self.assertEqual(summ["leaks"]["probe_equals_gold"], 0)

    def test_balance_and_fields(self):
        for it in self.train + self.val:
            yes = sum(n["truth"] for n in it["subqs"])
            self.assertEqual(2 * yes, len(it["subqs"]), it["item_id"])
            self.assertEqual(it["has_tree"], bool(it["subqs"]))
            self.assertEqual(it["group"], f"gsm8k_tf/s{it['n_steps']}")
            self.assertTrue(it["prompt"].startswith(problem_no_options(it) + "\nOptions:\n"))
            for n in it["subqs"]:
                self.assertTrue({"id", "question", "opposite", "truth", "depends_on", "node_depth"} <= set(n))
        self.assertGreater(len(self.train), 5000)
        self.assertTrue(380 <= len(self.val) <= 420)

    def test_splits(self):
        tr, va = {it["problem_id"] for it in self.train}, {it["problem_id"] for it in self.val}
        self.assertFalse(tr & va)
        self.assertEqual(len(tr), len(self.train))
        soc_val = {r["problem_id"] for r in rows(DATA / "student_gsm8k_soc/val.jsonl")} \
            if (DATA / "student_gsm8k_soc/val.jsonl").exists() else set()
        self.assertFalse(tr & soc_val)
        le4 = rows(LE4 / "train.jsonl")
        self.assertEqual(le4, [it for it in self.train if it["n_steps"] <= 4])
        self.assertEqual(rows(LE4 / "val.jsonl"), [it for it in self.val if it["n_steps"] <= 4])

    def test_finals_unchanged(self):
        p = DATA / "student_gsm8k_soc/train.jsonl"
        if not p.exists():
            self.skipTest("no data/student_gsm8k_soc")
        soc = {r["item_id"]: r for r in rows(p) + rows(DATA / "student_gsm8k_soc/val.jsonl") if r.get("row_kind") == "final"}
        for it in self.train + self.val:
            r = soc[it["item_id"]]
            diff = {k for k in set(it) | set(r) if it.get(k) != r.get(k)}
            self.assertLessEqual(diff, self.B.CHANGED, it["item_id"])
            self.assertEqual(r["node_status"], "ok")

    def test_step_eval_files(self):
        ev = DATA / "eval"
        if not (ev / "gsm8k_s2.jsonl").exists():
            self.skipTest("no step files")
        v2 = {json.dumps(r, ensure_ascii=False): r for r in rows(ev / "gsm8k_mc_v2.jsonl")}
        seen = []
        for s in range(2, 9):
            for r in rows(ev / f"gsm8k_s{s}.jsonl"):
                self.assertIn(json.dumps(r, ensure_ascii=False), v2)
                self.assertEqual(min(r["depth"], 8), s)
                seen.append(r["item_id"])
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(len(seen), sum(min(r["depth"], 8) >= 2 for r in v2.values()))
        s5p = [r["item_id"] for r in rows(ev / "gsm8k_s5p.jsonl")]
        self.assertEqual(sorted(s5p), sorted(r["item_id"] for r in v2.values() if r["depth"] >= 5))

    def test_no_test_problem_8gram(self):
        """No train / val question shares a non-boilerplate 8-gram (in < 5 test problems) with GSM8K test, GSM1k or
        any file derived from them."""
        ev = DATA / "eval"
        if not (ev / "gsm1k_mc.jsonl").exists():
            self.skipTest("no eval files")
        ours = set().union(*(self.B.grams8(self.B.stem(it["prompt"])) for it in self.train + self.val))
        owners = self.B.eval_targets([self.B.stem(r["prompt"]) for r in rows(ev / "gsm8k_mc_v2.jsonl")], ours)
        hits = [it["item_id"] for it in self.train + self.val if self.B.shared(self.B.stem(it["prompt"]), owners)]
        self.assertEqual(hits, [])


@unittest.skipUnless(HAVE_TORCH and (TF / "val.jsonl").exists(), "needs torch and data/student_gsm8k_tf")
class TestGFPQViews(unittest.TestCase):
    def test_gfpq_is_problem_view_without_options(self):
        from cotdistill.student import build_full_tree
        items = [it for it in rows(TF / "val.jsonl") if it["subqs"]][:40]
        for it in items:
            ctx = problem_no_options(it)
            q = build_full_tree(it, mode="plain", view="statements+question", rng=random.Random(5), cap=0,
                                sources={GSM8K_TF})
            g = build_full_tree(it, mode="plain", view="problem", rng=random.Random(5), cap=0, sources={GSM8K_TF})
            self.assertEqual(len(q), 1 + len(it["subqs"]))
            self.assertEqual(q[0].text, g[0].text)
            for eq, eg in zip(q[1:], g[1:]):
                self.assertEqual(eq.text, eg.text.replace(it["prompt"], ctx, 1))
                self.assertEqual((eq.target, eq.weight), (eg.target, eg.weight))
                head = eq.text.split("\n\nIntermediate question: ")[0]
                self.assertEqual(head, ctx)
                self.assertNotIn("Options:", head)
            other = build_full_tree(it, mode="plain", view="statements+question", rng=random.Random(5), cap=0,
                                    sources={"numgen"})
            self.assertEqual(len(other), 1)                       # not a tree source: its final only


if __name__ == "__main__":
    unittest.main()
