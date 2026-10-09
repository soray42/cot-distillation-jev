"""scripts/fa_readout.py: prompt parsing, ranks with ties, baselines, paired bootstrap, and the readout of the Tier-0
runs in results/hpc/fa0 (accuracy equal to metrics.json). CPU only."""
from __future__ import annotations

import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cotdistill import fa_public as fp  # noqa: E402


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = load_script("fa_readout")
FA0 = ROOT / "results/hpc/fa0"


def rec(n=6, dec=2, texts=None):
    texts = texts or [f"message {i} " * (3 + 4 * i) for i in range(n)]
    agents = ["Planner", "Coder", "environment"]
    steps = [(agents[i % 3], texts[i]) for i in range(n)]
    return fp._rec(f"u{n}{dec}", "wwpro", "mathchat", "bench", "Compute the total.", "42", steps, dec, agents[dec % 3])


class TestParse(unittest.TestCase):
    def test_fa_public_item(self):
        r = rec(n=7, dec=3)
        cap = 40
        step, agent = fp.make_items(r, "x", "fa1/wwpro", cap=cap, with_gt=True, rng=random.Random(0))
        info = R.item_info(step)
        self.assertEqual(info["kind"], "step")
        self.assertEqual(info["gold"], 3)
        self.assertEqual(info["key"], "wwpro/mathchat")
        self.assertEqual(info["agents"], [s["agent"] for s in r["steps"]])
        self.assertEqual(info["lens"], [len(fp.cut(s["text"], cap)) for s in r["steps"]])
        a = R.item_info(agent)
        self.assertEqual(a["kind"], "agent")
        self.assertEqual(a["options"], fp.agents_in_order(r))
        self.assertEqual(a["agents"][-1], r["steps"][-1]["agent"])
        self.assertEqual(a["lens"], info["lens"])                    # parsed without the step options

    def test_ww_layout_without_label_names(self):
        log = ("The log below comes from a multi-agent system that failed to solve its task.\n\nTask: t\n\n"
               "Step 0 [A_Expert]: hello\nworld\nStep 1 [Computer_terminal]: ok\nStep 2 [A_Expert]: done")
        item = {"item_id": "ww-1-step", "group": "ww/alg/step", "labels": ["A", "B", "C"], "gold_label": "B",
                "prompt": log + "\n\nQuestion: Which step?\nOptions:\nA) Step 0 [A_Expert]\n"
                                "B) Step 1 [Computer_terminal]\nC) Step 2 [A_Expert]"}
        info = R.item_info(item)
        self.assertEqual(info["lens"], [len("hello\nworld"), 2, 4])
        self.assertEqual(info["agents"], ["A_Expert", "Computer_terminal", "A_Expert"])
        self.assertEqual(info["key"], "ww/alg/step")


class TestStats(unittest.TestCase):
    def test_ranks_follow_first_argmax(self):
        self.assertEqual(R.ranks([0.5, 0.5, 0.0], 0), 1)
        self.assertEqual(R.ranks([0.5, 0.5, 0.0], 1), 2)
        self.assertEqual(R.ranks([0.2, 0.5, 0.5], 2), 3 - 1)
        self.assertEqual(R.ranks([0.1, 0.2, 0.7], 0), 3)

    def test_paired_identical(self):
        v = np.array([1.0, 0.0, 1.0, 1.0, np.nan])
        d = R.paired(v, v.copy(), 300, ("t",))
        self.assertEqual((d["n"], d["diff"], d["ci"], d["beats"]), (4, 0.0, (0.0, 0.0), False))
        d = R.paired(np.ones(50), np.zeros(50), 300, ("t",))
        self.assertTrue(d["beats"])

    def test_baselines(self):
        def info(n, g):
            return {"labels": list(fp.SYMBOLS[:n]), "gold": g, "kind": "step", "agents": ["a"] * n,
                    "lens": list(range(1, n + 1))}
        infos = [info(5, 4), info(3, 2), info(4, 1), info(6, 1)]
        pri = R.TrainPriors(None, None, 0)
        b = R.baselines(infos, pri)
        v = b["vectors"]
        self.assertEqual(v["last_step"].tolist(), [1, 1, 0, 0])
        self.assertEqual(v["step_1"].tolist(), [0, 0, 1, 1])
        self.assertEqual(v["first_step"].tolist(), [0, 0, 0, 0])
        self.assertAlmostEqual(b["chance"], np.mean([1 / 5, 1 / 3, 1 / 4, 1 / 6]))
        self.assertEqual(v["best_const_index"].mean(), 0.5)
        self.assertEqual(v["best_const_rel"].mean(), 0.5)
        self.assertNotIn("surface_probe", v)
        ag = [{"labels": ["A", "B"], "gold": 1, "kind": "agent", "options": ["X", "Y"], "agents": ["X", "Y"],
               "lens": [1, 1]}, {"labels": ["A", "B", "C"], "gold": 2, "kind": "agent", "options": ["X", "Y", "Z"],
                                 "agents": ["Y", "Z"], "lens": [1, 1]}]
        pri.agent_counts.update({"Z": 5, "Y": 3})
        b = R.baselines(ag, pri)["vectors"]
        self.assertEqual(b["most_frequent_train_agent"].tolist(), [1, 1])
        self.assertEqual(b["last_step_agent"].tolist(), [1, 1])
        self.assertEqual(b["option_A"].tolist(), [0, 0])

    def test_probe_learns_length(self):
        rng = random.Random(0)
        infos = []
        for k in range(200):
            n = rng.randint(3, 8)
            g = rng.randrange(n)
            lens = [rng.randint(500, 900) for _ in range(n)]
            lens[g] = 50                                             # the gold step is always the shortest
            infos.append({"labels": list(fp.SYMBOLS[:n]), "gold": g, "kind": "step", "agents": ["a"] * n,
                          "lens": lens})
        y = np.array([x["gold"] for x in infos])
        acc = {}
        for level in ("position", "surface"):
            X, mask = R.tensor(infos, level)
            w = R.fit_clogit(X, mask, y)
            acc[level] = float((R.clogit_probs(X, mask, w).argmax(1) == y).mean())
        self.assertGreater(acc["surface"], 0.95)
        self.assertLess(acc["position"], 0.5)


@unittest.skipUnless(len(list(FA0.glob("*/preds_ww_step.jsonl"))) >= 2, "results/hpc/fa0 runs not present")
class TestTier0Runs(unittest.TestCase):
    def test_readout(self):
        with tempfile.TemporaryDirectory() as d:
            res = R.main([str(FA0 / "*"), "--name", "t", "--out-dir", d, "--probe-n", "300", "--boot", "200",
                          "--quiet"])
            self.assertTrue((Path(d) / "t.md").exists() and (Path(d) / "t.json").exists())
            json.loads((Path(d) / "t.json").read_text())
        self.assertTrue(res["acc_checks"])
        self.assertTrue(all(c["ok"] for c in res["acc_checks"]), res["acc_checks"])
        files = res["files"]
        for f in ("data/eval/ww_alg_step.jsonl", "data/eval/ww_alg_step_gt.jsonl", "data/eval/ww_alg_agent.jsonl",
                  "data/eval/ww_hc_agent.jsonl"):
            self.assertIn(f, files)
            self.assertEqual(len(files[f]["runs"]), 2)
            self.assertEqual(len(files[f]["contrasts"]), 1)
            c = files[f]["contrasts"][0]
            self.assertEqual(c["contrast"], "GFPS-A2")
            self.assertEqual(c["n"], files[f]["n"])
            for s in files[f]["runs"].values():
                self.assertIn("vs_best_trivial", s)
                self.assertLessEqual(s["acc"], s["acc3"])
        ww = files["data/eval/ww_alg_step.jsonl"]
        self.assertEqual(ww["n"], 126)
        self.assertIn("surface_probe", ww["baselines"])
        self.assertEqual(sum(v for k, v in next(iter(ww["runs"].values()))["rel_error"].items()
                             if k in [b[2] for b in R.REL_BINS]), 126)
        self.assertAlmostEqual(ww["baselines"]["step_1"]["acc"], 34 / 126)      # tier1_public.md: W&W Alg 27.0%
        self.assertIn("kk", next(iter(res["other_evals"].values())))


if __name__ == "__main__":
    unittest.main()
