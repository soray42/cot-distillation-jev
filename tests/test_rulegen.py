import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cotdistill import rulegen as rg  # noqa: E402


def cases(n=120, domains=rg.TRAIN_DOMAINS + rg.HELDOUT_DOMAINS, seed=3):
    return rg.generate(n, domains, seed=seed, renderers=rg.RENDERERS)


class TestRuleGen(unittest.TestCase):
    def test_deterministic(self):
        a = [rg.render(c) for c in cases(40, seed=7)]
        b = [rg.render(c) for c in cases(40, seed=7)]
        self.assertEqual(a, b)

    def test_vocabularies_disjoint(self):
        self.assertFalse(set(rg.TRAIN_DOMAINS) & set(rg.HELDOUT_DOMAINS))

    def test_engine_and_overrides(self):
        for c in cases():
            res = c.evaluate()
            self.assertEqual(c.evaluate(rg.variables(res))["decision"], res["decision"])
            if res["fire"]:
                self.assertTrue(res["a"][res["fire"] - 1])
                self.assertFalse(any(res["a"][:res["fire"] - 1]))
                off = c.evaluate({f"a:{res['fire']}": False})
                self.assertNotEqual(off["fire"], res["fire"])
            for k in range(1, len(c.rules) + 1):
                forced = c.evaluate({f"a:{k}": True, **{f"a:{j}": False for j in range(1, k)}})
                self.assertEqual(forced["decision"], c.rules[k - 1].outcome)

    def test_every_rule_can_fire(self):
        rng = random.Random(0)
        for c in cases(60):
            self.assertTrue(rg.valid_policy(c.domain, c.rules, rng, n=600))

    def test_render_has_gold_and_facts(self):
        for c in cases():
            rec = rg.eval_record(c)
            self.assertIn(f"{rec['gold_label']}) {c.domain.option(rec['gold'])}", rec["prompt"])
            self.assertIn(c.facts["_request"].isoformat(), rec["prompt"])
            heads = {rg.render(c, r).split("Options:\n")[1] for r in rg.RENDERERS}
            self.assertEqual(len(heads), 1)

    def test_record_round_trip(self):
        for c in cases(60):
            rec = rg.eval_record(c)
            back = rg.case_from_record(rec)
            self.assertEqual(rg.render(back), rec["prompt"])
            self.assertEqual(back.gold(), rec["gold"])

    def test_tree_nodes(self):
        for c in cases():
            res = c.evaluate()
            nodes = rg.tree_nodes(c)
            by = {n["id"]: n for n in nodes}
            rules = [n for n in nodes if n["var"].startswith("a:")]
            self.assertEqual(len(rules), res["fire"] or len(c.rules))
            for n in nodes:
                kind, name = n["var"].split(":", 1)
                self.assertEqual(n["truth"], res["p"][name] if kind == "p" else res["a"][int(name) - 1])
                self.assertEqual(n["p_cot"], float(n["truth"]))
                if kind == "a":
                    want = {f"p:{p.pid}" for p, _ in c.rules[int(name) - 1].conds}
                    self.assertEqual({by[d]["var"] for d in n["depends_on"]}, want)

    def test_controls_are_unread_facts(self):
        rng = random.Random(1)
        for c in cases():
            used = {p.fact for r in c.rules for p, _ in r.conds}
            for q in rg.control_questions(c, rng, 4):
                self.assertNotIn(rg.Pred.parse(q["var"][4:]).fact, used)
                self.assertEqual(q["p_fresh"], float(q["truth"]))

    def test_edit_flips_exactly_one_predicate(self):
        rng = random.Random(2)
        n = 0
        for c in cases():
            before = c.evaluate()["p"]
            for meta, cf in rg.edit_pairs(c, rng):
                after = cf.evaluate()["p"]
                self.assertEqual({k for k in before if before[k] != after[k]}, {meta["var"][2:]})
                self.assertEqual(meta["sub"] == "sensitive", cf.gold() != c.gold())
                self.assertEqual(cf.rules, c.rules)
                n += 1
        self.assertGreater(n, 200)

    def test_distractors_and_rule_twins(self):
        rng = random.Random(4)
        for c in cases():
            for meta, cf in rg.distractor_pairs(c, rng):
                self.assertEqual(cf.gold(), c.gold())
                self.assertNotEqual(rg.render(cf), rg.render(c))
            for meta, cf in rg.rule_twins(c, rng):
                self.assertEqual(cf.facts, c.facts)
                self.assertEqual(meta["gold_change"], cf.gold() != c.gold())
                self.assertNotEqual(rg.render(cf), rg.render(c))

    def test_third_outcome_pairs(self):
        rng = random.Random(5)
        n = 0
        for c in cases():
            b = c.evaluate()
            for meta, src in rg.third_outcome_pairs(c, rng):
                s = src.evaluate()
                self.assertEqual(src.rules, c.rules)
                sv = rg.variables(s)
                self.assertNotEqual(rg.variables(b)[meta["var"]], sv[meta["var"]])
                target = c.evaluate({meta["var"]: sv[meta["var"]]})["decision"]
                self.assertEqual(target, meta["target_out"])
                self.assertNotIn(target, (b["decision"], s["decision"]))
                n += 1
        self.assertGreater(n, 150)

    def test_train_record_fields(self):
        rng = random.Random(6)
        for c in cases(30, rg.TRAIN_DOMAINS):
            r = rg.train_record(c, rng)
            self.assertEqual(sum(r["teacher"].values()), 1.0)
            self.assertEqual(r["teacher"][r["gold_label"]], 1.0)
            self.assertTrue(r["subqs"] and all(q["truth"] is not None for q in r["subqs"]))


if __name__ == "__main__":
    unittest.main()
