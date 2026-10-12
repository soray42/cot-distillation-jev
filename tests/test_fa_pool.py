"""scripts/fa_pool.py (paired FA arm comparison under StepFinder's protocol): exact McNemar and the sign-flip test,
Connor's sample size, per-item seed averaging and the per-seed-pair McNemar counts on synthetic runs, the pools
(Who&When protocol files + the _sfclean public files; val never pooled), checkpoint selection on val with the
mid-training predictions (files without them left out, or scored from the final predictions with --fallback-final),
and, when the two existing fasf runs are present, the registered result (GFPS-t2 e2 minus A2 final on Who&When Alg:
+4.8 [-3.2, +12.7], McNemar p .33). CPU only:

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/sev/bin/python -m unittest tests/test_fa_pool.py
"""
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = load_script("fa_pool")
LABELS = ["A", "B", "C"]


def item(iid: str, gold: int) -> dict:
    return {"item_id": iid, "source": "fa_public", "group": "t/step", "type": "choice", "prompt": "log",
            "labels": LABELS, "label_names": {L: f"Step {i} [X{i}]" for i, L in enumerate(LABELS)},
            "gold_label": LABELS[gold], "fa": {"uid": iid}}


def pred(it: dict, right: bool, second: bool = False) -> dict:
    g = LABELS.index(it["gold_label"])
    top = g if right else (g + 1) % 3
    probs = [0.1, 0.1, 0.1]
    probs[top] = 0.7
    if second and not right:
        probs[g] = 0.2                     # gold ranked 2nd
    return {"item_id": it["item_id"], "group": it["group"], "labels": LABELS, "probs": probs,
            "gold_label": it["gold_label"], "gold_probs": None}


class TestStats(unittest.TestCase):
    def test_mcnemar(self):
        self.assertAlmostEqual(P.mcnemar(10, 16), 0.3269, places=4)
        self.assertEqual(P.mcnemar(10, 16), P.mcnemar(16, 10))
        self.assertEqual(P.mcnemar(0, 0), 1.0)
        self.assertAlmostEqual(P.mcnemar(0, 6), 2 / 64)

    def test_sign_flip(self):
        rng = np.random.default_rng(0)
        d = np.array([1.0] * 16 + [-1.0] * 10 + [0.0] * 100)
        self.assertEqual(P.sign_flip(d, 1000, rng), P.mcnemar(10, 16))     # binary d: exact McNemar
        d2 = np.array([0.5] * 30 + [-0.5] * 5 + [0.0] * 50)
        p = P.sign_flip(d2, 20000, rng)
        self.assertAlmostEqual(p, P.mcnemar(5, 30), delta=0.002)           # +-.5 steps: the same binomial
        self.assertGreater(P.sign_flip(np.array([0.5, -0.5, 1.0, -1.0]), 2000, rng), 0.9)

    def test_connor(self):
        n = P.connor_n(0.206, 0.05, 0.05, 0.8)
        self.assertTrue(640 < n < 650, n)
        self.assertGreater(P.connor_n(0.206, 0.03, 0.05, 0.8), 1700)
        self.assertIsNone(P.connor_n(0.002, 0.05, 0.05, 0.8))
        self.assertAlmostEqual(P.connor_power(0.206, 0.05, 0.05, math.ceil(n)), 0.8, places=2)


class TestRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.files = {"xfoo": t / "fa_sf_x_foo_sfclean.jsonl", "sfalg": t / "fa_sf_ww_alg.jsonl", "val": t / "val.jsonl"}
        self.items = {k: [item(f"{k}-{i}", i % 3) for i in range(n)] for k, n in (("xfoo", 40), ("sfalg", 30),
                                                                                 ("val", 20))}
        for k, p in self.files.items():
            p.write_text("".join(json.dumps(x) + "\n" for x in self.items[k]))
        # correctness patterns per (arm, seed): item index i is right when the rule holds
        self.rule = {("A2", 12): lambda i: i % 2 == 0, ("A2", 13): lambda i: i % 3 == 0,
                     ("GFPS", 12): lambda i: i % 2 == 0 or i % 5 == 1, ("GFPS", 13): lambda i: i % 4 != 3}
        for (arm, seed), rule in self.rule.items():
            d = t / "runs" / f"{arm}-tg-M-student_x-s{seed}"
            (d / "mid").mkdir(parents=True)
            for k, its in self.items.items():
                (d / f"preds_{k}.jsonl").write_text("".join(json.dumps(pred(x, rule(i), second=True)) + "\n"
                                                            for i, x in enumerate(its)))
            # mid point e1: val and xfoo only; GFPS's val is better there than at the end
            for k in ("val", "xfoo"):
                (d / "mid" / f"preds_{k}_e1.jsonl").write_text("".join(json.dumps(pred(x, True)) + "\n"
                                                                       for x in self.items[k]))
            vacc = float(np.mean([rule(i) for i in range(20)]))
            m = {"args": {"epochs": 3, "eval": [f"{k}={p}" for k, p in self.files.items()]},
                 "eval": {"val": {"acc": vacc}},
                 "trajectory": [{"epoch": 1.0, "val": {"acc": 1.0 if arm == "GFPS" else 0.0}}]}
            (d / "metrics.json").write_text(json.dumps(m))
        self.t = t

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, **kw):
        a = ["--a", str(self.t / "runs/A2-*"), "--b", str(self.t / "runs/GFPS-*"), "--boot", "500", "--flips", "500",
             "--quiet", "--out-dir", str(self.t / "out")]
        for k, v in kw.items():
            a += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
        return a

    def test_final(self):
        res = P.main(self.args())
        files = {f["name"]: f for f in res["files"]}
        self.assertEqual(set(files), {"xfoo", "sfalg", "val"})
        for k, n in (("xfoo", 40), ("sfalg", 30)):
            ra = [np.mean([self.rule[("A2", s)](i) for s in (12, 13)]) for i in range(n)]
            rb = [np.mean([self.rule[("GFPS", s)](i) for s in (12, 13)]) for i in range(n)]
            f = files[k]
            self.assertAlmostEqual(f["acc_a"], float(np.mean(ra)))
            self.assertAlmostEqual(f["acc_d"], float(np.mean(rb) - np.mean(ra)))
            for p in f["pairs"]:
                s = p["seed_a"]
                self.assertEqual(p["seed_b"], s)
                ca = [self.rule[("A2", s)](i) for i in range(n)]
                cb = [self.rule[("GFPS", s)](i) for i in range(n)]
                self.assertEqual(p["b_a_only"], sum(a and not b for a, b in zip(ca, cb)))
                self.assertEqual(p["c_b_only"], sum(b and not a for a, b in zip(ca, cb)))
            # Acc@3 = 1 for every item (3 options); MRR@3: 1 if right else 1/2 (gold second)
            self.assertAlmostEqual(f["acc3_d"], 0.0)
            self.assertAlmostEqual(f["mrr3_d"], f["acc_d"] / 2)
        self.assertEqual(files["xfoo"]["pool"], "new")
        self.assertEqual(files["sfalg"]["pool"], "ww")
        self.assertIsNone(files["val"]["pool"])
        self.assertEqual(res["pools"]["all"]["n"], 70)
        self.assertEqual(res["pools"]["new"]["n"], 40)
        self.assertEqual(sorted(res["pools"]["all"]["files"]), ["sfalg", "xfoo"])
        w = (40 * files["xfoo"]["acc_d"] + 30 * files["sfalg"]["acc_d"]) / 70
        self.assertAlmostEqual(res["pools"]["all"]["acc_d"], w)
        self.assertAlmostEqual(res["pools"]["all"]["acc_d_eq"], (files["xfoo"]["acc_d"] + files["sfalg"]["acc_d"]) / 2)
        pa = res["pools"]["all"]["pairs"]
        self.assertEqual([p["b_a_only"] for p in pa],
                         [sum(files[k]["pairs"][j]["b_a_only"] for k in ("xfoo", "sfalg")) for j in range(2)])

    def test_val_selection(self):
        res = P.main(self.args(at_b="val"))
        self.assertEqual({r["epoch"] for r in res["b"]}, {1.0})
        self.assertEqual({r["epoch"] for r in res["a"]}, {None})
        files = {f["name"]: f for f in res["files"]}
        self.assertEqual(set(files), {"xfoo", "val"})                 # sfalg has no e1 predictions
        self.assertIn("sfalg", res["missing"])
        self.assertAlmostEqual(files["xfoo"]["acc_b"], 1.0)
        res2 = P.main(self.args(at_b="e1", fallback_final=True))
        f2 = {f["name"]: f for f in res2["files"]}
        self.assertEqual(f2["sfalg"]["src_b"], ["final*"])
        self.assertEqual(f2["xfoo"]["src_b"], ["e1"])


class TestRegistered(unittest.TestCase):
    A = "results/hpc/fasf/A2-sfx6-Qwen3.5-0.8B-Base-student_fa_sf_alg-s12"
    B = "results/hpc/fasf/GFPS-sfs3-Qwen3.5-0.8B-Base-student_fa_sf_alg_t2-s12"

    def test_alg_runs(self):
        if not ((ROOT / self.A / "metrics.json").exists() and (ROOT / self.B / "mid").exists()
                and (ROOT / "data/eval/fa_sf_ww_alg.jsonl").exists()):
            self.skipTest("fasf runs not present")
        with tempfile.TemporaryDirectory() as t:
            res = P.main(["--a", self.A, "--b", self.B, "--at-b", "e2", "--quiet", "--out-dir", t])
        f = {x["name"]: x for x in res["files"]}["sfalg"]
        s = f["stats"]
        self.assertEqual(f["n"], 126)
        self.assertEqual((round(100 * f["acc_a"], 1), round(100 * f["acc_b"], 1)), (27.0, 31.7))
        self.assertEqual(round(100 * s["acc_d"], 1), 4.8)
        self.assertEqual((round(100 * s["acc_ci"][0], 1), round(100 * s["acc_ci"][1], 1)), (-3.2, 12.7))
        self.assertEqual((f["pairs"][0]["b_a_only"], f["pairs"][0]["c_b_only"]), (10, 16))
        self.assertAlmostEqual(f["pairs"][0]["p"], 0.327, places=3)
        self.assertEqual(res["pools"]["all"]["n"], 126)


if __name__ == "__main__":
    unittest.main()
