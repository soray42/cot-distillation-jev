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


class TestLeaksFinal(unittest.TestCase):
    def test_kk_and_justlogic(self):
        from cotdistill.teacher import leaks_final
        rec = S.knights_knaves(1, seed=7, people_range=(4, 4))[0]
        ppl = rec["meta"]["people"]
        branch = "If " + ", ".join(f"{p} were a knight" for p in ppl) + f", would {ppl[0]}'s statement be false?"
        self.assertFalse(leaks_final(branch, rec))                       # one statement under a hypothesis
        self.assertTrue(leaks_final("With " + ", ".join(f"{p} a knight" for p in ppl) + ", does every statement hold?", rec))
        self.assertTrue(leaks_final(f"Is the assignment with {ppl[0]} a knight consistent with all statements?", rec))
        self.assertFalse(leaks_final(f"Is {ppl[0]} a knight?", rec))
        jl = {"domain": "justlogic", "prompt": "Passage: ...\n\nStatement: Stops are acts.\n\nQuestion: ..."}
        self.assertTrue(leaks_final("Is the statement uncertain given the passage?", jl))
        self.assertTrue(leaks_final("Is it true that stops are acts?", jl))
        self.assertFalse(leaks_final("Does the passage say that meadow voles live in meadows?", jl))
        jl2 = {"domain": "justlogic", "prompt": "Passage: ...\n\nStatement: Given that usefulness is quality, it can be "
               "inferred that prostate cancer affects more men than any other cancer except skin cancer.\n\nQuestion: ..."}
        self.assertTrue(leaks_final("Must it be true that if usefulness is quality, then prostate cancer affects more men "
                                    "than any other cancer except skin cancer?", jl2))
        self.assertFalse(leaks_final("Must it be true that either visual proprioception is present at birth and appears "
                                     "early on in evolution or if usefulness is quality, then prostate cancer affects more "
                                     "men than any other cancer except skin cancer?", jl2))
        self.assertFalse(leaks_final("Can the statement 'if elk eat vegetation, then postmodern spirituality is different "
                                     "from postmodern philosophy' be considered false?", jl2))
        self.assertTrue(leaks_final("Given only the passage and valid logical reasoning, is the statement uncertain?", jl2))


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
             "opposite": f"Is {p1} a knave?", "status": "corrected", "initial_answer": "yes", "quote": f"Then {p1} follows."},
            {"id": "n3", "type": "verify", "depends_on": ["n1", "n2"], "question": "Is the assignment consistent with all statements?",
             "answer": "yes", "status": "direct", "quote": "x"}]}
        client = FakeClient(tree)
        out = tree_item(client, res)
        self.assertEqual(out["extraction"], "tree")
        self.assertEqual(len(out["subquestions"]), 2)                # the full-assignment check is dropped as a leak
        self.assertEqual(len(out["leaked_nodes"]), 1)
        n2 = out["subquestions"][1]
        if n2["polarity"] == "opposite":                              # swapped wording flips the answers
            self.assertEqual((n2["question"], n2["stated"], n2["initial"]), (f"Is {p1} a knave?", "yes", "no"))
        else:
            self.assertEqual((n2["stated"], n2["initial"]), ("no", "yes"))
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
        s1 = rec_out["subqs"][0]                                       # n2 may be dropped as inconsistent (fake answers say yes)
        self.assertEqual(s1["type"], "derive")
        self.assertGreater(s1["p_commit"], 0.9)                           # stated "yes", confident commitment
        self.assertEqual(rec_out["teacher"][rec["gold_label"]], 1.0)


if __name__ == "__main__":
    unittest.main()
