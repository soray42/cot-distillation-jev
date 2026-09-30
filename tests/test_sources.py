import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill import sources as S  # noqa: E402


class TestKnightsKnaves(unittest.TestCase):
    def test_unique_solution_and_gold_option(self):
        for rec in S.knights_knaves(15, seed=3, people_range=(5, 8)):
            people = [p["question"][3:-10] for p in rec["predicates"]]          # "Is X a knight?"
            truth = {p: n["truth"] for p, n in zip(people, rec["predicates"])}
            gold_text = rec["options"][rec["gold"]]
            self.assertEqual(gold_text, ", ".join(f"{p}: {'knight' if truth[p] else 'knave'}" for p in people))
            self.assertIn(f"{rec['gold_label']}) {gold_text}", rec["prompt"])
            self.assertEqual(len(set(rec["options"].values())), len(rec["options"]))

    def test_gold_satisfies_all_statements_and_distractors_do_not(self):
        for rec in S.knights_knaves(10, seed=5, people_range=(6, 9)):
            people, stmts = rec["meta"]["people"], rec["meta"]["statements"]
            for key, text in rec["options"].items():
                roles = {p: part.endswith("knight") for p, part in zip(people, text.split(", "))}
                violated = S._consistent(stmts, roles)
                if key == rec["gold"]:
                    self.assertEqual(violated, 0)
                else:
                    self.assertGreater(violated, 0)


    def test_kk_subq_truth(self):
        rec = S.knights_knaves(1, seed=7, people_range=(5, 5))[0]
        people, roles = rec["meta"]["people"], {p: n["truth"] for p, n in zip(rec["meta"]["people"], rec["predicates"])}
        a, b, c = people[:3]
        self.assertEqual(S.kk_subq_truth(f"Is {a} a knight?", rec), roles[a])
        self.assertEqual(S.kk_subq_truth(f"Is {a} a knave?", rec), not roles[a])
        self.assertEqual(S.kk_subq_truth(f"Is {b}'s statement true?", rec), roles[b])
        self.assertEqual(S.kk_subq_truth(f"Are {a}, {b}, and {c} knights?", rec), roles[a] and roles[b] and roles[c])
        self.assertEqual(S.kk_subq_truth(f"Are {a} and {b} both knaves?", rec), not roles[a] and not roles[b])
        self.assertIsNone(S.kk_subq_truth(f"Does {a}'s statement force {b} to be a knight?", rec))
        self.assertIsNone(S.kk_subq_truth("Is Zed a knight?", rec))


class TestResolvePYes(unittest.TestCase):
    def test_censored_letters_are_symmetric(self):
        from cotdistill.teacher import resolve_p_yes
        self.assertEqual(resolve_p_yes({"yes_first": True, "missing": ["A"], "mass": 0.99, "p_yes": None}), 0.0)
        self.assertEqual(resolve_p_yes({"yes_first": False, "missing": ["B"], "mass": 0.99, "p_yes": None}), 0.0)
        self.assertEqual(resolve_p_yes({"yes_first": True, "missing": ["B"], "mass": 0.99}, {"A": 1.0}), 1.0)
        self.assertIsNone(resolve_p_yes({"yes_first": True, "missing": ["A", "B"], "mass": 0.0, "p_yes": None}))
        self.assertEqual(resolve_p_yes({"yes_first": True, "missing": [], "mass": 1.0}, {"A": 0.3, "B": 0.7}), 0.3)


class TestKKNodeTruth(unittest.TestCase):
    def test_hypotheticals_and_pairs(self):
        import itertools
        rec = json.loads(json.dumps(S.knights_knaves(1, seed=11, people_range=(5, 5))[0]))   # as stored on disk
        ppl, stm = rec["meta"]["people"], rec["meta"]["statements"]
        sol = {p: n["truth"] for p, n in zip(ppl, rec["predicates"])}
        a, b = ppl[:2]
        self.assertEqual(S.kk_node_truth(f"Do {a} and {b} have the same role?", rec), sol[a] == sol[b])
        self.assertEqual(S.kk_node_truth(f"Are {a} and {b} opposite roles?", rec), sol[a] != sol[b])
        full = ", ".join(f"{p} {'a knight' if sol[p] else 'a knave'}" for p in ppl)
        for sp in ppl:
            self.assertEqual(S.kk_node_truth(f"If {full}, would {sp}'s statement be true?", rec), sol[sp])
        self.assertEqual(S.kk_node_truth(f"If {full}, would the statements contradict each other?", rec), False)
        wrong = ", ".join(f"{p} {'a knave' if sol[p] else 'a knight'}" for p in ppl)
        self.assertEqual(S.kk_node_truth(f"If {wrong}, would the statements contradict each other?", rec), True)
        # a single-person hypothesis: contradiction iff no consistent completion exists
        flip = f"If {a} were {'a knave' if sol[a] else 'a knight'}, would the statements contradict each other?"
        self.assertEqual(S.kk_node_truth(flip, rec), True)                       # the solution is unique
        self.assertIsNone(S.kk_node_truth(f"If {a} were a knight, would {b} be happy?", rec))
        self.assertIsNone(S.kk_node_truth("Is the sky blue?", rec))


@unittest.skipUnless((ROOT / "data/raw/justlogic/train_dataset.csv").exists(), "JustLogic data not downloaded")
class TestJustLogic(unittest.TestCase):
    def test_records(self):
        recs = S.justlogic(str(ROOT / "data/raw/justlogic/train_dataset.csv"), n=50, seed=0, min_depth=5)
        self.assertEqual(len(recs), 50)
        for r in recs:
            self.assertGreaterEqual(r["depth"], 5)
            self.assertIn(r["gold"], S.JL_OPTIONS)
            self.assertIn(f"{r['gold_label']}) {S.JL_OPTIONS[r['gold']]}", r["prompt"])


if __name__ == "__main__":
    unittest.main()
