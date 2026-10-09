"""scripts/eval_subset.py: equal allocation over strata, fixed seed, records unchanged; data/eval/bbh_sub500.jsonl when
present. CPU only:

  OMP_NUM_THREADS=2 ~/.venvs/sev/bin/python -m unittest tests/test_eval_subset.py
"""
import json
import random
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_subset as es  # noqa: E402


def items(sizes: dict[str, int]) -> list[dict]:
    return [{"item_id": f"{g}-{j:03d}", "group": g, "prompt": f"{g} {j}"} for g, n in sizes.items() for j in range(n)]


class TestSubset(unittest.TestCase):
    def test_equal_allocation(self):
        src = items({f"bbh/t{k}": 250 if k < 20 else 146 + k for k in range(23)})
        out = es.subset(src, 500, seed=0)
        per = Counter(it["group"] for it in out)
        self.assertEqual(len(out), 500)
        self.assertEqual(len(per), 23)
        self.assertEqual(sorted(set(per.values())), [21, 22])
        self.assertEqual(sum(v == 22 for v in per.values()), 500 - 23 * 21)
        self.assertEqual(len({it["item_id"] for it in out}), 500)
        pos = {it["item_id"]: j for j, it in enumerate(src)}
        self.assertEqual([pos[it["item_id"]] for it in out], sorted(pos[it["item_id"]] for it in out))  # file order
        self.assertTrue(all(it == src[pos[it["item_id"]]] for it in out))                                # unchanged

    def test_seed(self):
        src = items({"a": 40, "b": 40, "c": 40})
        ids = lambda s: [it["item_id"] for it in es.subset(src, 30, seed=s)]  # noqa: E731
        self.assertEqual(ids(3), ids(3))
        self.assertNotEqual(ids(3), ids(4))

    def test_small_stratum_water_filling(self):
        out = es.subset(items({"a": 3, "b": 100, "c": 100}), 60, seed=1)
        per = Counter(it["group"] for it in out)
        self.assertEqual(per["a"], 3)
        self.assertEqual(per["b"] + per["c"], 57)
        self.assertLessEqual(abs(per["b"] - per["c"]), 1)
        with self.assertRaises(ValueError):
            es.allocate({"a": 2}, 3, random.Random(0))

    def test_cli(self):
        with tempfile.TemporaryDirectory() as t:
            src, dst = Path(t) / "src.jsonl", Path(t) / "o" / "sub.jsonl"
            src.write_text("".join(json.dumps(it) + "\n" for it in items({"x": 10, "y": 10})))
            with redirect_stdout(StringIO()):
                es.main([str(src), str(dst), "--n", "6", "--seed", "0"])
            self.assertEqual(len(dst.read_text().splitlines()), 6)

    @unittest.skipUnless((ROOT / "data/eval/bbh_sub500.jsonl").exists() and (ROOT / "data/eval/bbh.jsonl").exists(),
                         "bbh files absent")
    def test_bbh_sub500_is_reproducible(self):
        with open(ROOT / "data/eval/bbh.jsonl") as f:
            bbh = [json.loads(line) for line in f if line.strip()]
        with open(ROOT / "data/eval/bbh_sub500.jsonl") as f:
            sub = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(sub, es.subset(bbh, 500, seed=0))
        per = Counter(it["group"] for it in sub)
        self.assertEqual((len(sub), len(per), min(per.values()), max(per.values())), (500, 23, 21, 22))


if __name__ == "__main__":
    unittest.main()
