import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill import sources as S  # noqa: E402
from cotdistill.readout import value_commitment  # noqa: E402
from cotdistill.teacher import parse_tree, tree_item  # noqa: E402


def tok(t, alts):
    return {"token": t, "logprob": alts[0][1], "top_logprobs": [{"token": a, "logprob": lp} for a, lp in alts]}


class TestParseTree(unittest.TestCase):
    def test_validation(self):
        raw = json.dumps({"nodes": [
            {"id": "n1", "type": "parse", "depends_on": ["n9"], "question": "Is A a knight exactly when B is a knave?",
             "answer": "yes", "status": "direct", "initial_answer": None, "quote": "A = not B"},
            {"id": "n2", "type": "derive", "depends_on": ["n1"], "question": "Did the reasoning treat A as a knight?",
             "answer": "yes", "status": "direct", "quote": "x"},
            {"id": "n3", "type": "derive", "depends_on": ["n1"], "question": "Does B fail to be a knave?",
             "answer": "no", "status": "direct", "quote": "x"},
            {"id": "n4", "type": "conclusion", "depends_on": [], "question": "Is the answer C?", "answer": "yes"},
            {"id": "n5", "type": "case", "depends_on": ["n1", "n2"], "question": "If B were a knight, would A lie?",
             "answer": "YES", "status": "corrected", "initial_answer": "no", "quote": "Case B true"},
            {"id": "n5", "type": "derive", "depends_on": [], "question": "Duplicate id?", "answer": "no"}]})
        nodes = parse_tree(raw)
        self.assertEqual([n["id"] for n in nodes], ["n1", "n5"])        # meta, "fail to", bad type, duplicate dropped
        self.assertEqual(nodes[0]["depends_on"], [])                       # forward/unknown dependency dropped
        self.assertEqual(nodes[1]["depends_on"], ["n1"])                   # n2 was dropped, so only n1 remains
        self.assertEqual((nodes[1]["stated"], nodes[1]["status"], nodes[1]["initial"]), ("yes", "corrected", "no"))
        self.assertEqual(parse_tree("not json"), [])


class TestValueCommitment(unittest.TestCase):
    def test_value_fork_vs_wording(self):
        toks = [tok("So", [("So", -0.7), ("Then", -0.7)]),                         # wording fork: ignored
                tok(" Bob", [(" Bob", 0.0)]),
                tok(" knave", [(" knave", -0.357), (" knight", -1.204), (" is", -3.0)]),
                tok(" not", [(" not", -0.5), (" knight", -1.0)])]                  # "not" is no value
        c = value_commitment(toks, (0, 4))
        self.assertEqual(c["token"], " knave")
        self.assertAlmostEqual(c["conf"], 0.7, places=2)
        self.assertIsNone(value_commitment(toks[:2], (0, 2)))
        self.assertIsNone(value_commitment(toks, None))


class FakeClient:
    """Canned DeepSeek responses keyed by the call tag, so tree_item runs offline."""
    def __init__(self, tree):
        self.tree, self.tags = tree, []

    def chat(self, messages, **kw):
        tag = kw.get("tag", "")
        self.tags.append(tag)
        if tag.endswith("/tree"):
            return {"choices": [{"message": {"content": json.dumps(self.tree)}}]}
        if tag.endswith("/generic"):
            return {"choices": [{"message": {"content": json.dumps({"questions": ["Does Bob speak first?"] * 5})}}]}
        lp = {"content": [tok("A", [("A", -0.05), ("B", -3.0)])]}
        return {"choices": [{"message": {"content": "A"}, "logprobs": lp}]}


class TestTreeItem(unittest.TestCase):
    def test_offline_tree_item(self):
        rec = S.knights_knaves(1, seed=7, people_range=(4, 4))[0]
        p0, p1 = rec["meta"]["people"][:2]
        role0 = "knight" if rec["predicates"][0]["truth"] else "knave"
        cot = f"Let us solve. So {p0} is a {role0}. Then {p1} follows."
        reasoning_lp = [[t, -0.01, [[t, -0.01]]] for t in cot.split(" ")]
        reasoning_lp = [[t if i == 0 else " " + t, lp, [[t if i == 0 else " " + t, -0.01], [" knave" if "knight" in t else " knight", -4.0]]
                         if t.startswith(role0) else alts] for i, (t, lp, alts) in enumerate(reasoning_lp)]
        res = {"item": {k: v for k, v in rec.items() if k != "prompt"}, "prompt": rec["prompt"],
               "traces": [{"reasoning": cot, "reasoning_lp": reasoning_lp}], "subquestions": [],
               "random_subquestions": [{"question": "Does the island have 4 people?", "answer_nocot": {"p_yes": 1.0}}]}
        tree = {"nodes": [
            {"id": "n1", "type": "derive", "depends_on": [], "question": f"Is {p0} a {role0}?", "answer": "yes",
             "status": "direct", "initial_answer": None, "quote": f"So {p0} is a {role0}."},
            {"id": "n2", "type": "derive", "depends_on": ["n1"], "question": f"Is {p1} a knight?", "answer": "no",
             "status": "corrected", "initial_answer": "yes", "quote": f"Then {p1} follows."}]}
        client = FakeClient(tree)
        out = tree_item(client, res)
        self.assertEqual(out["extraction"], "tree")
        self.assertEqual(len(out["subquestions"]), 2)
        n1 = out["subquestions"][0]
        self.assertTrue(n1["truth"])                                  # program truth for K&K role questions
        self.assertIsNotNone(n1["span"])
        self.assertGreater(n1["commit"]["conf"], 0.9)
        self.assertEqual(len(out["random_subquestions"]), 2)          # control topped up to the node count
        self.assertTrue(any(t.endswith("/generic") for t in client.tags))

        sys.path.insert(0, str(ROOT / "scripts"))
        from build_student_data import convert
        out["traces"][0]["dist_outcome"] = {rec["gold"]: 1.0}
        rec_out = convert(out)
        s1, s2 = rec_out["subqs"]
        self.assertEqual((s1["type"], s2["initial"], s2["depends_on"]), ("derive", "yes", ["n1"]))
        self.assertGreater(s1["p_commit"], 0.9)                           # stated "yes", confident commitment
        self.assertEqual(rec_out["teacher"][rec["gold_label"]], 1.0)


if __name__ == "__main__":
    unittest.main()
