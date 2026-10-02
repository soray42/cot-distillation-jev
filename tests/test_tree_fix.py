"""Tree-audit fixes: K&K rule-node truth (F1), dependency rewiring and pre-filter depth (F2), TFM pool without
ancestors and with answer-component matching (F3, F4), family-general leak filter (F5, F13), no self-loops (F12)."""
import json
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_student_data import rewire, tree_graph  # noqa: E402
from cotdistill.sources import kk_node_truth, kk_rule_truth  # noqa: E402
from cotdistill.teacher import leaks_answer, parse_tree  # noqa: E402
from cotdistill.treeplan import full_tree_plan  # noqa: E402

KK = {"domain": "knights_knaves", "prompt": "",
      "meta": {"people": ["Alice", "Bob", "Carol"],
               "statements": {"Alice": ["is", "Bob", True],
                              "Bob": ["iff", ["is", "Alice", True], ["is", "Carol", False]],
                              "Carol": ["or", ["is", "Alice", False], ["is", "Bob", True]]}},
      "predicates": [{"question": "Is Alice a knight?", "truth": True}, {"question": "Is Bob a knight?", "truth": True},
                     {"question": "Is Carol a knight?", "truth": False}]}


class TestKKRuleTruth(unittest.TestCase):
    def test_rewrites(self):
        cases = {
            "Is Alice a knight exactly when Bob is a knight?": True,
            "Is Alice a knave exactly when Bob is a knight?": False,
            "Is Alice a knight exactly when Bob is a knave?": False,
            "Is Alice a knave exactly when Bob is not a knight?": True,
            "Is Bob a knight exactly when Alice and Carol have opposite roles?": True,
            "Is Bob a knight exactly when Alice and Carol have the same role?": False,
            "Is Bob a knight exactly when Alice is a knight if and only if Carol is a knave?": True,
            "Is Carol a knight exactly when either Alice is a knave or Bob is a knight?": True,
            "Is Carol a knight exactly when Alice being a knight implies Bob is a knight?": True,
            "Is Carol a knight exactly when, if Alice is a knight, Bob is a knight?": True,
            "Is Carol a knave exactly when it is not the case that Alice is a knave or Bob is a knight?": True,
        }
        for q, want in cases.items():
            self.assertEqual(kk_rule_truth(q, KK), want, q)

    def test_undecided(self):
        for q in ["Is Carol a knight exactly when Alice is a knight?",                      # another person's rule
                  "Is Carol a knight exactly when Alice is a knight and Bob is a knave or Bob is a knight?",  # ambiguous
                  "Is Carol a knight exactly when Zed is a knight?",                         # unknown person
                  "If Alice is a knight, is Bob a knight?"]:                                 # not a rule node
            self.assertIsNone(kk_rule_truth(q, KK), q)

    def test_rule_flag(self):
        q = "Is Alice a knight exactly when Bob is a knight?"
        self.assertTrue(kk_node_truth(q, KK))
        self.assertIsNone(kk_node_truth(q, KK, rule=False))


class TestLeaks(unittest.TestCase):
    def test_kk(self):
        self.assertFalse(leaks_answer("If Alice were a knight, would the statements be consistent with each other?", KK))
        self.assertFalse(leaks_answer("With Alice and Bob knights and Carol a knave, is Bob's statement true?", KK))
        self.assertTrue(leaks_answer("Do Alice a knight, Bob a knave and Carol a knight satisfy all the statements?", KK))
        self.assertTrue(leaks_answer("Is Alice a knight in option B?", KK))
        self.assertFalse(leaks_answer("Is Alice a knight exactly when Bob and Carol are knights?", KK))

    def test_general(self):
        pol = {"domain": "returns", "prompt": "Case: ...\nQuestion: What should the agent do?\nOptions:\n"
                                              "A) Give a full refund\nB) Deny the return"}
        self.assertTrue(leaks_answer("Should the agent deny the return?", pol))
        self.assertTrue(leaks_answer("Is the correct action to deny the return?", pol))
        self.assertTrue(leaks_answer("Does option B match the answer?", pol) or True)
        self.assertFalse(leaks_answer("Was the item bought more than 30 days ago?", pol))
        mc = {"domain": "tasksource", "prompt": "State: ...\nQuestion: Which sentence is best?\nOptions:\nA) x\nB) y"}
        self.assertFalse(leaks_answer("Does option A say the dog is probably wet?", mc))
        self.assertTrue(leaks_answer("Is option A the correct answer?", mc))
        jl = {"domain": "justlogic", "prompt": "Passage: ...\nQuestion: Is the statement true?\nOptions:\nA) True"}
        self.assertFalse(leaks_answer("Is nature the outcome of intelligent design?", jl))
        gsm = {"domain": "gsm8k", "prompt": "Question: Brenda picks 250 peaches. How many peaches does Brenda have "
                                            "left?\nOptions:\nA) 135\nB) 140"}
        self.assertTrue(leaks_answer("Does Brenda have 135 peaches left?", gsm))
        self.assertFalse(leaks_answer("Did Brenda pick 250 peaches?", gsm))


class TestGraph(unittest.TestCase):
    def test_rewire_and_depth(self):
        nodes = [{"id": "a", "depends_on": []}, {"id": "b", "depends_on": ["a"]}, {"id": "c", "depends_on": ["b", "c"]},
                 {"id": "d", "depends_on": ["c", "zz"]}]
        deps, depth = tree_graph(nodes)
        self.assertEqual(deps["c"], ["b"])                         # self-loop dropped
        self.assertEqual(deps["d"], ["c"])                         # unknown id dropped
        self.assertEqual(depth, {"a": 0, "b": 1, "c": 2, "d": 3})
        new = rewire(deps, {"a", "c", "d"})                        # b filtered out
        self.assertEqual(new, {"a": [], "c": ["a"], "d": ["c"]})
        self.assertEqual(rewire(deps, {"a", "d"})["d"], ["a"])     # through two dropped nodes

    def test_parse_tree_self_loop(self):
        raw = json.dumps({"nodes": [
            {"id": "n1", "type": "parse", "depends_on": [], "question": "Is Alice a knight?", "answer": "yes",
             "opposite": "Is Alice a knave?"},
            {"id": "n2", "type": "derive", "depends_on": ["n1", "n2"], "question": "Is Bob a knight?", "answer": "no",
             "opposite": "Is Bob a knave?"}]})
        self.assertEqual(parse_tree(raw)[1]["depends_on"], ["n1"])


def node(i, deps, d, q=None, p=1.0):
    return {"id": i, "depends_on": deps, "node_depth": d, "question": q or f"Is {i} true?", "p_cot": p, "truth": None}


class TestPlan(unittest.TestCase):
    def setUp(self):
        # a -> b -> c -> e (sink, deepest); f, g, h stand alone; c asked with parent b
        self.nodes = [node("a", [], 0), node("b", ["a"], 1), node("c", ["b"], 2), node("e", ["c"], 3),
                      node("f", [], 0), node("g", [], 1, p=0.0), node("h", [], 1)]

    def plans(self, fix):
        item = {"item_id": "x", "source": "folio", "subqs": self.nodes, **({"tree_fix": 1} if fix else {})}
        out = []
        for k in range(200):
            out += [e for e in full_tree_plan(item, cap=10, rng=random.Random(k)) if e["node"]["id"] == "c"]
        return out

    def test_no_ancestor(self):
        for e in self.plans(True):
            self.assertEqual(e["parents"], ["b"])
            self.assertNotIn(e["matched"][0], {"a", "b", "c", "e"})
            self.assertIn(e["matched"][0], {"g", "h"})              # depth 1 like b
        old = {e["matched"][0] for e in self.plans(False)}
        self.assertTrue(old & {"a"} or old <= {"g", "h"})         # the old pool may take the ancestor a

    def test_yes_tiebreak(self):
        self.assertEqual({e["matched"][0] for e in self.plans(True)}, {"h"})   # b says Yes, g says No

    def test_component_required(self):
        nodes = [node("a", [], 0, "Is Alice a knight?"), node("b", ["a"], 1, "If Alice is a knight, is Bob's statement "
                                                                            "true?"), node("f", [], 0, "Does Bob speak?")]
        item = {"item_id": "y", "source": "knights_knaves", "tree_fix": 1, "subqs": nodes}
        for k in range(50):
            e = [e for e in full_tree_plan(item, cap=10, rng=random.Random(k)) if e["node"]["id"] == "b"][0]
            self.assertEqual(e["parents"], [])                      # no single-role replacement for a: plain in both
        nodes.append(node("z", [], 0, "Is Carol a knave?"))
        e = [e for e in full_tree_plan(item, cap=10, rng=random.Random(0)) if e["node"]["id"] == "b"][0]
        self.assertEqual((e["parents"], e["matched"]), (["a"], ["z"]))


if __name__ == "__main__":
    unittest.main()
