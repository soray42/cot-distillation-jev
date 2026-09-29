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
