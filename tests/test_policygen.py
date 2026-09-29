import random
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cotdistill import policygen as pg  # noqa: E402


class TestPolicyGen(unittest.TestCase):
    def test_deterministic(self):
        a = pg.generate(30, seed=7)
        b = pg.generate(30, seed=7)
        self.assertEqual([r["prompt"] for r in a], [r["prompt"] for r in b])

    def test_records_are_consistent(self):
        for r in pg.generate(200, pg.TRAIN_DOMAINS + pg.HELDOUT_DOMAINS, seed=1):
            self.assertIn(r["gold"], r["label_order"])
            self.assertEqual(r["label_order"][ord(r["gold_label"]) - 65], r["gold"])
            self.assertGreaterEqual(r["depth"], 0)
            self.assertIn(f"{r['gold_label']}) ", r["prompt"])
            if r["hidden"] is None:
                self.assertNotEqual(r["gold"], pg.ASK)
                self.assertTrue(all(n["truth"] is not None for n in r["path"]))
            else:
                self.assertIn("unknown (not stated)", r["prompt"])

    def test_ask_iff_hidden_fact_matters(self):
        rng = random.Random(3)
        seen = Counter()
        for i in range(400):
            case = pg.make_case(rng.choice(pg.TRAIN_DOMAINS), rng, f"x{i}", p_hidden=1.0)
            outs = {case._evaluate({**case.facts, case.hidden: v})[0] for v in case.fact_space[case.hidden]}
            self.assertEqual(case.gold() == pg.ASK, len(outs) > 1)
            seen[case.gold() == pg.ASK] += 1
        self.assertGreater(seen[True], 20)
        self.assertGreater(seen[False], 20)

    def test_outcomes_are_spread(self):
        recs = pg.generate(300, seed=2)
        by_dom = {}
        for r in recs:
            by_dom.setdefault(r["domain"], Counter())[r["gold"]] += 1
        for dom, c in by_dom.items():
            self.assertGreaterEqual(len(c), 3, (dom, c))
            self.assertLess(max(c.values()) / sum(c.values()), 0.5, (dom, c))


if __name__ == "__main__":
    unittest.main()
