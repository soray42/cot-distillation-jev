"""scripts/fa_readout.py: prompt parsing, ranks with ties, baselines, paired bootstrap, the readout of the Tier-0
runs in results/hpc/fa0 (accuracy equal to metrics.json; only the two student_fa0 runs, whatever else the directory
holds), and the opt-in StepFinder protocol (--sf: metrics, baselines, checkpoint selection on synthetic runs). CPU
only."""
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
TIER0 = "*-student_fa0-s*"            # the Tier-0 pair (A2 / GFPS on data/student_fa0); later runs share the directory


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


class TestRuns(unittest.TestCase):
    def test_parse_run_and_dropped(self):
        r = R.parse_run("GFPS-fa1c2-Qwen3.5-0.8B-Base-student_fa1b-s12")
        self.assertEqual((r["arm"], r["tag"], r["model"], r["data"], r["seed"]),
                         ("GFPS", "fa1c2", "Qwen3.5-0.8B-Base", "student_fa1b", 12))
        self.assertEqual(R.parse_run("A2-x-s3")["arm"], "A2")
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".fa_readout_test_") as d:
            (Path(d) / "summary.json").write_text(json.dumps({"debias": {"dropped": ["wwpro/macnet"]}}))
            train = str((Path(d) / "train_all.jsonl").relative_to(ROOT))
            self.assertEqual(R.dropped_for({"args": {"train": train}}, ["x/y"]), ["wwpro/macnet", "x/y"])
            (Path(d) / "summary.json").write_text(json.dumps({"debias": {"dropped": ["wwpro/macnet"]},  # --clean
                                                              "clean": {"dropped": ["wwpro/debate"]}}))
            self.assertEqual(R.dropped_for({"args": {"train": train}}, []), ["wwpro/debate", "wwpro/macnet"])
        self.assertEqual(R.dropped_for(None, []), [])
        p = R.resolve("val", {}, "fa1-val-00000", "student_fa0")
        self.assertEqual(p, ROOT / ("data/student_fa0/val.jsonl" if (ROOT / "data/student_fa0/val.jsonl").exists()
                                    else "data/student_fa1/val.jsonl"))
        self.assertEqual(R.resolve("ww_hc", R.eval_map(Path("."), None), "x"), ROOT / "data/eval/ww_hc_agent.jsonl")


def sf_info(n: int, gold: int, agents=None, label=None) -> dict:
    agents = agents or [("P", "C")[i % 2] for i in range(n)]
    return {"labels": list(fp.SYMBOLS[:n]), "gold": gold, "kind": "step", "opt_steps": list(range(n)),
            "opt_agents": agents, "agent_label": label, "agents": agents, "lens": [10] * n}


class TestSF(unittest.TestCase):
    def test_item_metrics(self):
        x = sf_info(8, 3)
        m = R.sf_score([5, 3, 0, 1, 2, 4, 6, 7], x)               # gold at rank 2, top-1 two steps late
        self.assertEqual((m["acc"], m["acc2"], m["acc3"], m["mrr3"]), (0.0, 1.0, 1.0, 0.5))
        self.assertEqual([m[f"tol{k}"] for k in R.SF_DELTAS], [0.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(m["agent"], 1.0)                           # steps 5 and 3 are both C's
        self.assertEqual(R.sf_score([4, 3], x)["agent"], 0.0)       # step 4 is P's
        self.assertEqual(R.sf_score([5, 3], dict(x, agent_label="p"))["agent"], 0.0)       # the label wins
        self.assertEqual(R.sf_score([4, 3], dict(x, agent_label="p"))["agent"], 1.0)
        m = R.sf_score([0, 1, 2, 3], sf_info(8, 3))                 # rank 4: no MRR@3 credit
        self.assertEqual((m["acc3"], m["mrr3"]), (0.0, 0.0))
        self.assertEqual(R.sf_order([0.2, 0.5, 0.5, 0.1]), [1, 2, 0, 3])
        self.assertEqual(R.ranks([0.2, 0.5, 0.5, 0.1], 2), R.sf_order([0.2, 0.5, 0.5, 0.1]).index(2) + 1)

    def test_expected_random_and_of_all(self):
        rng = random.Random(0)
        for _ in range(20):
            n = rng.randint(2, 9)
            x = sf_info(n, rng.randrange(n), label=rng.choice(["P", "C", "Z"]))
            e = R.sf_expected_random(x)
            perms = [rng.sample(range(n), n) for _ in range(4000)]
            mc = R.sf_mean([R.sf_score(o, x) for o in perms])
            for k in ("acc", "acc2", "acc3", "mrr3", "tol1", "tol3", "agent"):
                self.assertAlmostEqual(e[k], mc[k], delta=0.04, msg=k)
        rows = [{"acc": 1.0, "acc2": 1.0, "acc3": 1.0, "mrr3": 1.0, "agent": 1.0, **{f"tol{k}": 1.0 for k in
                                                                                       R.SF_DELTAS}}] * 3
        m = R.sf_mean(rows, n_total=6)
        self.assertEqual((m["acc"], m["acc_of_all"], m["n"], m["n_total"]), (1.0, 0.5, 3, 6))
        self.assertNotIn("acc_of_all", R.sf_mean(rows, n_total=3))

    def test_runs_and_selection(self):
        """A synthetic run with final and mid-training predictions: --sf leaves every other output unchanged, scores
        the final checkpoint, selects the first best val point (main.py's strict >) and the best test point."""
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".fa_readout_sf_") as d:
            d = Path(d)
            rs = [rec(n=6 + k % 3, dec=k % 5) for k in range(12)]
            test, val = [], []
            for k, r in enumerate(rs):
                r = dict(r, uid=f"u{k}")
                s, _ = fp.make_items(r, f"t{k}", "fa1/wwpro", cap=40, with_gt=False, rng=random.Random(k))
                (test if k < 8 else val).append(s)
            (d / "test.jsonl").write_text("\n".join(json.dumps(x) for x in test))
            (d / "val.jsonl").write_text("\n".join(json.dumps(x) for x in val))
            run = d / "runs" / "A2-x-Qwen-student_fa_sf_alg-s12"
            (run / "mid").mkdir(parents=True)

            def preds(items, hit):                     # hit(k): gold first, else gold last
                out = []
                for k, it in enumerate(items):
                    n, g = len(it["labels"]), it["labels"].index(it["gold_label"])
                    p = [0.01] * n
                    p[g if hit(k) else (g + 1) % n] = 0.6
                    if not hit(k):
                        p[g] = 0.3                     # second: Acc@2 = 1
                    z = sum(p)
                    out.append({"item_id": it["item_id"], "labels": it["labels"], "probs": [v / z for v in p],
                                "gold_label": it["gold_label"]})
                return "\n".join(json.dumps(x) for x in out)
            plan = {1.0: (lambda k: k < 2, lambda k: k < 1), 2.0: (lambda k: k < 6, lambda k: k < 3),
                    3.0: (lambda k: k < 7, lambda k: k < 3), 4.0: (lambda k: k < 3, lambda k: k < 2)}
            for ep, (ht, hv) in plan.items():
                if ep < 4:
                    (run / "mid" / f"preds_sfx_e{ep:g}.jsonl").write_text(preds(test, ht))
                    (run / "mid" / f"preds_val_e{ep:g}.jsonl").write_text(preds(val, hv))
            (run / "preds_sfx.jsonl").write_text(preds(test, plan[4.0][0]))
            (run / "preds_val.jsonl").write_text(preds(val, plan[4.0][1]))
            rel = lambda p: str(p.relative_to(ROOT))                # noqa: E731
            (run / "metrics.json").write_text(json.dumps({
                "args": {"epochs": 4.0, "eval": [f"sfx={rel(d / 'test.jsonl')}", f"val={rel(d / 'val.jsonl')}"]},
                "eval": {"sfx": {"acc": 3 / 8, "n": 8}, "val": {"acc": 2 / 4, "n": 4}}, "trajectory": [],
                "history": []}))
            args = [str(run), "--name", "t", "--out-dir", str(d / "o"), "--boot", "100", "--quiet",
                    "--train-step", "", "--train-agent", ""]
            a = R.main(args)
            b = R.main(args + ["--sf"])
            self.assertEqual(json.dumps(a["files"], default=str), json.dumps(b["files"], default=str))
            self.assertNotIn("sf", a)
            f = b["sf"][rel(d / "test.jsonl")]
            res = f["runs"][run.name]
            self.assertAlmostEqual(res["final"]["acc"], 3 / 8)
            self.assertAlmostEqual(res["final"]["acc2"], 1.0)
            self.assertEqual(res["val_selected"]["epoch"], 2.0)    # val 3/4 at e2 and e3: the first
            self.assertAlmostEqual(res["val_selected"]["acc"], 6 / 8)
            self.assertEqual(res["test_selected"]["epoch"], 3.0)
            self.assertAlmostEqual(res["test_selected"]["acc"], 7 / 8)
            self.assertIn("random (expected)", f["baselines"])
            md = (d / "o" / "t.md").read_text()
            self.assertIn("## StepFinder protocol", md)
            self.assertTrue(md.startswith(R.report(a)))


@unittest.skipUnless(len(list(FA0.glob(TIER0 + "/preds_ww_step.jsonl"))) == 2, "results/hpc/fa0 Tier-0 runs not present")
class TestTier0Runs(unittest.TestCase):
    def test_readout(self):
        with tempfile.TemporaryDirectory() as d:
            res = R.main([str(FA0 / TIER0), "--name", "t", "--out-dir", d, "--probe-n", "300", "--boot", "200",
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
