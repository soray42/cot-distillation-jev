"""Tier-0 failure-attribution data (src/cotdistill/fa_tier0.py, scripts/build_fa_tier0.py): the simulator, exact labels
by brute-force re-simulation, the item format the trainer reads, the node DAG, leak rules, determinism, and the
fa_tier0 branch of treeplan.statement_block (every other source unchanged against HEAD). CPU only:

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python3 -m unittest tests/test_fa_tier0.py
"""
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill import fa_tier0 as fa  # noqa: E402
from cotdistill import numgen as ng  # noqa: E402
from cotdistill import treeplan  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

KK = json.loads((ROOT / "tests/fixtures/kk_local_item.json").read_text())
FRAMES = [f for f in ng.FRAMES if f["id"] not in ng.HELD_OUT]


def trajectories(n: int, seed: int = 0, **kw) -> list[dict]:
    rng = random.Random(seed)
    out = []
    i = 0
    while len(out) < n:
        try:
            p = ng.generate(rng.choice(FRAMES), rng.choice([2, 3, 4, 5, 6]), rng)
        except RuntimeError:
            continue
        r = fa.build_trajectory(p, f"t{seed}-{i}", random.Random(f"{seed}-{i}"), **kw)
        i += 1
        if r is not None:
            out.append(r)
    return out


TRAJ = trajectories(60)


class TestSimulator(unittest.TestCase):
    def test_clean_runs_succeed(self):
        rng = random.Random(3)
        for i in range(40):
            p = ng.generate(rng.choice(FRAMES), rng.choice([2, 3, 4, 5, 6]), rng)
            tr = fa.make_traj(p, f"c{i}", rng)
            run = fa.simulate(tr, {})
            self.assertTrue(run.success)
            self.assertEqual(run.declared, p.answer.value)
            self.assertTrue(all(s.valid and s.true for s in run.steps))
            self.assertTrue(all(u < s.idx for s in run.steps for u in s.uses))

    def test_labels_by_brute_force(self):
        """Every invalid step: remove its fault alone, re-simulate; the decisive step is the earliest success, the
        prefix before the corrected step is unchanged, and the run with all faults fails."""
        for r in TRAJ:
            tr, faults, run, lab = r["traj"], r["faults"], r["run"], r["labels"]
            self.assertFalse(run.success)
            self.assertNotEqual(run.declared, tr.gold)
            invalid = [s for s in run.steps if not s.valid]
            self.assertEqual({s.eid.split("/")[0] for s in invalid}, set(faults))
            wins = []
            for s in invalid:
                key = s.eid.split("/")[0]
                cf = fa.simulate(tr, {k: v for k, v in faults.items() if k != key})
                self.assertEqual([x.text for x in cf.steps[:s.idx]], [x.text for x in run.steps[:s.idx]])
                self.assertTrue(cf.steps[s.idx].valid)
                self.assertEqual(cf.success, lab["R"][s.idx])
                if cf.success:
                    wins.append(s.idx)
            self.assertTrue(wins)
            self.assertEqual(lab["decisive"], min(wins))
            for s in run.steps:                              # inheritance: valid, untrue, some untrue input
                bad_in = any(not run.steps[u].true for u in s.uses)
                self.assertEqual(lab["I"][s.idx], s.valid and not s.true and bad_in)
                if not s.valid:
                    self.assertFalse(s.true)

    def test_determinism(self):
        a = trajectories(8, seed=11)
        b = trajectories(8, seed=11)
        for x, y in zip(a, b):
            self.assertEqual([s.text for s in x["run"].steps], [s.text for s in y["run"].steps])
            self.assertEqual(x["labels"]["decisive"], y["labels"]["decisive"])
            ix = fa.make_items(x, "i", True, random.Random(1))
            iy = fa.make_items(y, "i", True, random.Random(1))
            self.assertEqual(json.dumps(ix), json.dumps(iy))

    def test_multi_error_share(self):
        n_multi = sum(len([s for s in r["run"].steps if not s.valid]) >= 2 for r in TRAJ)
        n_first = sum([s.idx for s in r["run"].steps if not s.valid][0] != r["labels"]["decisive"] for r in TRAJ)
        self.assertGreaterEqual(n_multi / len(TRAJ), 0.3)
        self.assertGreaterEqual(n_first / len(TRAJ), 0.1)
        single = trajectories(20, seed=5, p_multi=0.0)
        self.assertTrue(all(len([s for s in r["run"].steps if not s.valid]) == 1 for r in single))

    def test_fault_types_and_lengths(self):
        types = {s.fault for r in trajectories(150, seed=2) for s in r["run"].steps if s.fault}
        for t in ("plan_op", "plan_const", "plan_skip", "given", "arith", "operand", "op", "input", "premature",
                  "accept", "reject", "slip", "wrongq", "side"):
            self.assertIn(t, types)
        for r in TRAJ:
            self.assertTrue(6 <= len(r["run"].steps) <= 16)


class TestItems(unittest.TestCase):
    def setUp(self):
        self.items = [fa.make_items(r, f"x-{i}", i % 2 == 0, random.Random(i)) for i, r in enumerate(TRAJ)]

    def test_format_and_audit(self):
        for (step, agent), r in zip(self.items, TRAJ):
            self.assertEqual(fa.audit_item(step), [])
            n = len(r["run"].steps)
            self.assertEqual(step["labels"], fa.LETTERS[:n])
            self.assertEqual(step["label_order"], step["labels"])
            self.assertEqual(step["gold_label"], fa.LETTERS[r["labels"]["decisive"]])
            self.assertEqual(step["teacher"][step["gold_label"]], 1.0)
            self.assertEqual(sum(step["teacher"].values()), 1.0)
            self.assertIsNone(step["gold_probs"])
            self.assertEqual((step["source"], step["type"], step["tree_fix"]), (fa.SOURCE, "choice", True))
            head, _, tail = step["prompt"].rpartition("\n\n" + fa.Q_STEP + "\nOptions:\n")
            self.assertTrue(head.startswith(fa.HEADER + "\n\nTask: "))
            self.assertEqual(("Correct final answer (for reference): " in head), step["with_gt"])
            for i, line in enumerate(tail.split("\n")):
                self.assertRegex(line, rf"^{fa.LETTERS[i]}\) Step {i} \[[A-Za-z_]+\]$")
                self.assertIn(f"\nStep {i} [", head)
            # agent level: agents in order of first appearance, gold = the decisive step's agent
            agents = list(agent["label_names"].values())
            self.assertEqual(agents, fa.agents_in_order(r["run"].steps))
            self.assertEqual(agent["label_names"][agent["gold_label"]], r["run"].steps[r["labels"]["decisive"]].agent)
            self.assertTrue(agent["prompt"].endswith(fa.Q_AGENT + "\nOptions:\n" + fa.options_block(agents)))
            self.assertNotIn("Correct final answer", agent["prompt"])

    def test_nodes(self):
        yes, total = 0, 0
        for (step, _), r in zip(self.items, TRAJ):
            nodes = step["subqs"]
            by = {n["id"]: n for n in nodes}
            self.assertEqual(len(by), len(nodes))
            self.assertEqual(treeplan.node_depths(nodes), {n["id"]: n["node_depth"] for n in nodes})
            self.assertLessEqual(max(n["node_depth"] for n in nodes), 1)
            self.assertEqual({n["type"] for n in nodes} - {"validity", "correctness", "dependency", "inheritance",
                                                          "repair"}, set())
            steps = r["run"].steps
            for n in nodes:
                plain = n["truth"] if n["polarity"] == "plain" else not n["truth"]
                s = steps[n["step"]]
                want = {"validity": s.valid, "correctness": s.true, "inheritance": r["labels"]["I"][s.idx],
                        "repair": r["labels"]["R"].get(s.idx),
                        "dependency": n.get("source_step") in s.uses}[n["type"]]
                self.assertEqual(plain, want, n)
                self.assertEqual(n["p_cot"], 1.0 if n["truth"] else 0.0)
                if n["type"] == "repair":
                    self.assertFalse(s.valid)
                self.assertNotRegex(n["question"], r"(?i)decisive|inject|fault|slip")
                yes += n["truth"]
                total += 1
            for s in steps:
                for kind in ("v", "c", "i"):
                    self.assertIn(f"{kind}{s.idx}", by)
        self.assertTrue(0.4 < yes / total < 0.6)

    def test_no_metadata_in_prompts(self):
        for step, agent in self.items:
            for p in (step["prompt"], agent["prompt"]):
                self.assertNotRegex(p, r"(?i)\b(decisive)\b(?!.*\?)|inject|fault")
                self.assertEqual(p.count("Question: "), 1)
                self.assertNotIn("Answer:", p.rpartition("Question: ")[2])
            self.assertEqual(fa.leak_check(step), [])


class TestStatementBlock(unittest.TestCase):
    def head_treeplan(self):
        """treeplan.py as committed at HEAD (before the fa_tier0 branch)."""
        src = subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:src/cotdistill/treeplan.py"], capture_output=True,
                             text=True)
        if src.returncode != 0:
            self.skipTest("git HEAD not readable")
        spec = importlib.util.spec_from_loader("treeplan_head", loader=None)
        mod = importlib.util.module_from_spec(spec)
        mod.__dict__["__package__"] = "cotdistill"
        exec(compile(src.stdout, "treeplan_head.py", "exec"), mod.__dict__)
        return mod

    def numgen_item(self):
        p = ng.generate(FRAMES[0], 3, random.Random(0))
        return ng.final_item(p, "n0", "mix", random.Random(0))

    def test_fa_source(self):
        step, agent = fa.make_items(TRAJ[0], "x", True, random.Random(0))
        block = treeplan.statement_block(step)
        self.assertTrue(block.startswith(fa.HEADER))
        self.assertTrue(step["prompt"].startswith(block + "\n\n" + fa.Q_STEP + "\nOptions:\n"))
        self.assertNotIn("Question:", block)
        self.assertEqual(treeplan.problem_no_options(step), block + "\n\n" + fa.Q_STEP)
        self.assertIn("\nCorrect final answer (for reference): ", block)
        plain, agent = fa.make_items(TRAJ[0], "x", False, random.Random(0))
        self.assertEqual(treeplan.statement_block(agent), treeplan.statement_block(plain))
        self.assertTrue(agent["prompt"].startswith(treeplan.statement_block(plain) + "\n\n" + fa.Q_AGENT))
        for bad in (step["prompt"].replace("\n\nQuestion: ", "\n\nAsk: "), step["prompt"].replace(fa.HEADER, "Log:"),
                    step["prompt"].split("\nOptions:\n")[0]):
            with self.assertRaises(ValueError):
                treeplan.statement_block(dict(step, prompt=bad))

    def test_other_sources_unchanged(self):
        head = self.head_treeplan()
        items = [KK, self.numgen_item()]
        tf = ROOT / "data/student_numgen_tf/val.jsonl"
        if tf.exists():
            with open(tf) as f:
                items += [json.loads(next(f)) for _ in range(3)]
        for it in items:
            self.assertEqual(treeplan.statement_block(it), head.statement_block(it))
            self.assertEqual(treeplan.problem_no_options(it), head.problem_no_options(it))
        step, _ = fa.make_items(TRAJ[0], "x", False, random.Random(0))
        for it in (dict(step, source="justlogic"), {"item_id": "o", "source": None, "prompt": "P\nOptions:\nA) a"},
                   dict(KK, prompt=KK["prompt"].replace("\n\nQuestion: ", "\n\nAsk: "))):
            for fn in ("statement_block", "problem_no_options"):
                with self.assertRaises(ValueError) as new:
                    getattr(treeplan, fn)(it)
                with self.assertRaises(ValueError) as old:
                    getattr(head, fn)(it)
                self.assertEqual(str(new.exception), str(old.exception))
        with self.assertRaises(ValueError):                       # HEAD had no fa_tier0 branch
            head.statement_block(step)

    @unittest.skipUnless(HAVE_TORCH, "needs torch")
    def test_trainer_views(self):
        from cotdistill.student import build_full_tree, build_group
        step, _ = fa.make_items(TRAJ[1], "x", False, random.Random(0))
        n = len(step["subqs"])
        block = treeplan.statement_block(step)
        for view, ctx in (("problem", step["prompt"]), ("statements", block),
                          ("statements+question", treeplan.problem_no_options(step))):
            ex = build_full_tree(step, mode="plain", view=view, rng=random.Random(0), cap=0, final="teacher")
            self.assertEqual(len(ex), 1 + n)
            self.assertEqual(ex[0].text, step["prompt"] + "\n\nAnswer:")
            self.assertEqual(ex[0].gold, step["labels"].index(step["gold_label"]))
            for e in ex[1:]:
                self.assertTrue(e.text.startswith(ctx + "\n\nIntermediate question: "))
            self.assertAlmostEqual(sum(e.weight for e in ex[1:]), 1.0)
        g = build_group(step, aux_kind="cot", k=3, aux_weight=0.5, rng=random.Random(0), subq_target="truth")
        self.assertEqual(len(g), 4)
        sc = build_full_tree(step, mode="plain", view="statements", rng=random.Random(0), cap=0, scaffold_p=1.0,
                             mix_seed=0, mix_epoch=0, stats={})
        self.assertEqual(len(sc), 1 + n)


class TestBuilder(unittest.TestCase):
    def test_small_build(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".fa0_test_") as d:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
            cmd = [sys.executable, str(ROOT / "scripts/build_fa_tier0.py"), "--out", f"{d}/out", "--eval-dir",
                   f"{d}/eval", "--n-train", "60", "--n-val", "20", "--n-test", "20", "--no-numgen-exclusion",
                   "--tokenizer", "missing.json"]
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
            self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
            s = json.loads(Path(f"{d}/out/summary.json").read_text())
            self.assertEqual(s["audit"]["failures"], 0)
            self.assertEqual(s["disjoint"]["test_train_signatures"], 0)
            counts = {Path(k).name: v for k, v in s["files"].items()}
            self.assertEqual(counts, {"fa0_test.jsonl": 20, "fa0_test_gt.jsonl": 20, "fa0_test_agent.jsonl": 20,
                                      "val.jsonl": 20, "val_agent.jsonl": 20, "train.jsonl": 60,
                                      "train_agent.jsonl": 60})
            tasks = {}
            for name in ("out/train.jsonl", "eval/fa0_test.jsonl"):
                with open(f"{d}/{name}") as f:
                    tasks[name] = {re.search(r"\nTask: (.*)\n", json.loads(l)["prompt"]).group(1) for l in f}
            self.assertFalse(tasks["out/train.jsonl"] & tasks["eval/fa0_test.jsonl"])
            with open(f"{d}/eval/fa0_test_gt.jsonl") as f:
                self.assertTrue(all("Correct final answer (for reference): " in json.loads(l)["prompt"] for l in f))
            r2 = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
            s2 = json.loads(Path(f"{d}/out/summary.json").read_text())
            self.assertEqual(s["md5"], s2["md5"])                 # deterministic


if __name__ == "__main__":
    unittest.main()
