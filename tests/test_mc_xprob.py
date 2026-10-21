"""Other-problem test sets (scripts/build_mc_xprob.py: data/eval/gsm8k_mc_v2_xprob.jsonl, data/eval/gsm1k_mc_xprob.jsonl):
- construction units on a synthetic file: 4 distinct distractors per item, none equal to the gold, each the answer of
  another item, exact rank quotas and letter balance, determinism, the item audit;
- the built files (skipped when absent): md5s, the same items / question texts / golds in the same order as the
  originals, no duplicate options, uniform gold rank and letter, a rebuild reproduces the files byte for byte, and the
  options-only adversaries of the summary near chance.
CPU only, numpy; data-backed tests read data/ of this checkout, or COTD_DATA_DIR when set."""
import collections
import hashlib
import json
import os
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_mc_xprob as X  # noqa: E402

DATA = Path(os.environ.get("COTD_DATA_DIR") or ROOT / "data")
EVAL = DATA / "eval"


def toy_file(n=300, seed=0):
    rng = random.Random(seed)
    items = []
    for i in range(n):
        g = rng.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 36, 40, 50, 60, 75, 100, 120, 150, 200, 250, 360,
                        500, 640, 1000, 1200, 2500, 4000, 10000, 70000]) + rng.choice([0, 0, 0, 1, 2, 7])
        opts = [g, g + 1, g + 3, g * 2 + 5, g + 11]
        order = list(range(5))
        rng.shuffle(order)
        letters = {L: opts[j] for L, j in zip(X.LETTERS, order)}
        gl = X.LETTERS[order.index(0)]
        items.append({"item_id": f"toy-{i}", "source": "toy", "group": "toy/d2/m2", "type": "choice",
                      "prompt": f"Question: Problem {i} asks for a number?\nOptions:\n"
                                + "\n".join(f"{L}) {v}" for L, v in letters.items()),
                      "labels": X.LETTERS, "label_order": X.LETTERS, "gold_label": gl,
                      "label_names": {L: f"o{j}" for L, j in zip(X.LETTERS, order)}, "gold_probs": None, "depth": 2,
                      "level": 2, "mag": "m2", "nq": 1, "orig_id": f"toy-{i}", "variant": None,
                      "option_kinds": {L: "gold" if L == gl else "perturb_gold" for L in X.LETTERS}, "recipe": "qo_v2:x"})
    return items


class TestConstruction(unittest.TestCase):
    def test_options(self):
        items = toy_file()
        out, st, fails = X.build_file(items, "toy", 0, "m")
        self.assertEqual(fails, [])
        n = len(items)
        golds = {it["item_id"]: X.gold_of(it) for it in items}
        for it, o in zip(items, out):
            q0, opts0 = X.split_prompt(it["prompt"])
            q1, opts1 = X.split_prompt(o["prompt"])
            self.assertEqual(q1, q0)
            self.assertEqual(opts1[o["gold_label"]], opts0[it["gold_label"]])
            vals = [X.num(opts1[L]) for L in X.LETTERS]
            self.assertEqual(len(set(vals)), 5)
            for L in X.LETTERS:
                if L == o["gold_label"]:
                    continue
                src = o["xprob"]["src"][L]
                self.assertNotEqual(src, it["item_id"])
                self.assertEqual(golds[src], opts1[L])
                self.assertNotEqual(X.num(opts1[L]), X.num(golds[it["item_id"]]))
            self.assertEqual(o["option_kinds"][o["gold_label"]], "gold")
            self.assertEqual(sum(v == "other_problem" for v in o["option_kinds"].values()), 4)
            self.assertEqual(o["label_names"][o["gold_label"]], "o0")
            self.assertEqual(sorted(o["label_names"].values()), [f"o{j}" for j in range(5)])
            self.assertEqual(o["xprob"]["gold_rank"], sorted(vals).index(X.num(opts1[o["gold_label"]])))
        ranks = collections.Counter(o["xprob"]["gold_rank"] for o in out)
        self.assertLessEqual(max(ranks.values()) - min(ranks.values()), 1 + 2 * st["quota_overflow"], ranks)
        letters = collections.Counter(o["gold_label"] for o in out)
        self.assertLessEqual(max(letters.values()) - min(letters.values()), 1, letters)
        self.assertEqual(sum(letters.values()), n)
        again, _, _ = X.build_file(items, "toy", 0, "m")
        self.assertEqual(again, out)
        other, _, _ = X.build_file(items, "toy", 1, "m")
        self.assertNotEqual([o["prompt"] for o in other], [o["prompt"] for o in out])

    def test_window(self):
        """Distractors fall in a window of width ln 2 on the ln(x + 5) scale around the gold (about 0.5x-2x for large
        golds) unless it had to grow; the window is not centred on the gold (the window rule is not the gold's)."""
        import math
        items = toy_file(600, 3)
        out, st, _ = X.build_file(items, "toy", 0, "m")
        for o in out:
            g = X.num(X.gold_of(o))
            vals = [X.num(v) for L, v in X.split_prompt(o["prompt"])[1].items() if L != o["gold_label"]]
            if o["xprob"]["window_ratio"] == 2.0:
                us = [X.scale(v) for v in vals] + [X.scale(g)]
                self.assertLessEqual(max(us) - min(us), math.log(2) + 1e-9, (g, vals))
            if g >= 1000 and o["xprob"]["window_ratio"] == 2.0:
                self.assertTrue(all(0.49 * g <= v <= 2.01 * g for v in vals), (g, vals))
        S, Y = X.option_sets(out)
        self.assertLess(X.window_rule(S, Y), 0.3)

    def test_audit_catches(self):
        items = toy_file(50)
        out, _, _ = X.build_file(items, "toy", 0, "m")
        golds = {it["item_id"]: X.gold_of(it) for it in items}
        o, it = dict(out[0]), items[0]
        self.assertEqual(X.audit_item(o, it, golds), [])
        bad = dict(o, prompt=o["prompt"].replace("Problem 0", "Problem zero"))
        self.assertTrue(any("question" in b for b in X.audit_item(bad, it, golds)))
        L = next(x for x in X.LETTERS if x != o["gold_label"])
        q, opts = X.split_prompt(o["prompt"])
        opts = dict(opts, **{L: opts[o["gold_label"]]})
        bad = dict(o, prompt=f"Question: {q}\nOptions:\n" + "\n".join(f"{k}) {opts[k]}" for k in X.LETTERS))
        self.assertTrue(X.audit_item(bad, it, golds))
        bad = dict(o, depth=99)
        self.assertTrue(any("other fields" in b for b in X.audit_item(bad, it, golds)))


@unittest.skipUnless((EVAL / "mc_xprob.summary.json").exists(), "data/eval/*_xprob.jsonl not built")
class TestBuiltFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = json.loads((EVAL / "mc_xprob.summary.json").read_text())

    def test_md5_and_audit(self):
        self.assertEqual(self.s["audit"]["failures"], 0)
        for name, want in list(self.s["md5"].items()) + list(self.s["source_md5"].items()):
            self.assertEqual(hashlib.md5((DATA / name[len("data/"):]).read_bytes()).hexdigest(), want, name)

    def test_items(self):
        for src, dst in X.SETS:
            orig, new = X.read(EVAL / f"{src}.jsonl"), X.read(EVAL / f"{dst}.jsonl")
            self.assertEqual([o["item_id"] for o in new], [o["item_id"] for o in orig])
            golds = {it["item_id"]: X.gold_of(it) for it in orig}
            ranks, letters = collections.Counter(), collections.Counter()
            for a, o in zip(new, orig):
                self.assertEqual(X.audit_item(a, o, golds), [])
                self.assertEqual(X.split_prompt(a["prompt"])[0], X.split_prompt(o["prompt"])[0])
                self.assertEqual(X.gold_of(a), X.gold_of(o))
                vals = [X.num(v) for v in X.split_prompt(a["prompt"])[1].values()]
                self.assertEqual(len(set(vals)), 5)
                ranks[sorted(vals).index(X.num(X.gold_of(a)))] += 1
                letters[a["gold_label"]] += 1
            n = len(new)
            self.assertLessEqual(max(abs(c - n / 5) for c in ranks.values()), 0.01 * n, (dst, ranks))
            self.assertLessEqual(max(letters.values()) - min(letters.values()), 1, (dst, letters))

    def test_rebuild_identical(self):
        md5 = self.s["recipe_md5"]
        self.assertEqual(md5, X.recipe_md5())
        for src, dst in X.SETS:
            out, _, _ = X.build_file(X.read(EVAL / f"{src}.jsonl"), src, self.s["args"]["seed"], md5)
            self.assertEqual("".join(json.dumps(r) + "\n" for r in out), (EVAL / f"{dst}.jsonl").read_text(), dst)

    def test_adversaries_near_chance(self):
        a = self.s["adversary"]
        for src, dst in X.SETS:
            self.assertGreater(a[src]["optonly"], 0.26, src)            # the original sets: the known cue
            r = a[dst]
            self.assertLess(r["optonly"], 0.23, dst)
            for k in ("optonly_xprob_train", "crossfit_optonly", "crossfit_rich", "window_rule", "hub", "largest",
                      "smallest", "middle"):
                self.assertLess(r[k], 0.25, (dst, k))


if __name__ == "__main__":
    unittest.main()
