"""Step curriculum (train_student.py --curriculum-steps; arm GFPQC): K node stages by solution step, then the final
questions, in the full-tree loop.
- curriculum_stage / curriculum_item: a node's stage is its step with every step beyond K folded into stage K;
  cumulative stage j keeps the nodes of steps <= j, exact stage j those of step j (every node in exactly one exact
  stage, stage K cumulative = the whole tree), every other field is the item's, nodes without a step refused;
- build_full_tree on a stage item: GFPQ's view (problem without its options) of exactly that stage's nodes, the
  per-problem weight split over them;
- the training loop on a tiny random model (6 GSM8K items of tests/fixtures/gsm8k_si_items_v1.json, steps 1-5): stage
  counts and views per stage (no final view in a node stage, finals only in the last; --curriculum-final-nodes: GFPQ's
  groups there), epochs / (K + 1) per stage, stage-end evaluations, one schedule, an empty exact stage with no update,
  --reset-optim, the argument checks;
- the existing loops (GFPQ, A2, A10 with stages and an optimizer reset, SIB) give CUR_BASE_REF's losses update for
  update.
CPU only (CUDA_VISIBLE_DEVICES="")."""
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

FX = json.loads((ROOT / "tests/fixtures/gsm8k_si_items_v1.json").read_text())
ITEMS = FX["gsm8k"]
# The last commit before the step curriculum: the existing loops are compared with it.
CUR_BASE_REF = os.environ.get("CUR_BASE_REF", "8aa9a0f")

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def nodes(it):
    return [sq for sq in it["subqs"] if "id" in sq]


class TestStages(unittest.TestCase):
    def test_stage_of_step(self):
        from cotdistill.student import curriculum_stage
        self.assertEqual([curriculum_stage(s, 4) for s in (1, 2, 3, 4, 5, 8)], [1, 2, 3, 4, 4, 4])
        self.assertEqual(curriculum_stage(3, 1), 1)
        for bad in (0, -1, None, 1.5, "2", True):
            with self.assertRaises(ValueError):
                curriculum_stage(bad, 4)

    def test_membership(self):
        """cumulative: nested stages, the last = every node; exact: a partition of the nodes; steps beyond K in K."""
        from cotdistill.student import curriculum_item
        before = json.dumps(ITEMS)
        for K in (1, 2, 3, 4, 6):
            for it in ITEMS:
                ids = [sq["id"] for sq in nodes(it)]
                step = {sq["id"]: sq["step"] for sq in nodes(it)}
                prev = []
                for j in range(1, K + 1):
                    cum = curriculum_item(it, stage=j, n_stages=K)
                    ex = curriculum_item(it, stage=j, n_stages=K, mode="exact")
                    self.assertEqual({k: v for k, v in cum.items() if k != "subqs"},
                                     {k: v for k, v in it.items() if k != "subqs"})
                    got = [sq["id"] for sq in cum["subqs"]]
                    self.assertEqual(got, [i for i in ids if min(step[i], K) <= j])       # data order kept
                    self.assertEqual([sq["id"] for sq in ex["subqs"]], [i for i in ids if min(step[i], K) == j])
                    self.assertTrue(set(prev) <= set(got))
                    prev = got
                    self.assertTrue(all(a is b for a, b in zip(cum["subqs"], [sq for sq in nodes(it) if sq["id"] in got])))
                self.assertEqual(prev, ids)
                parts = [sq["id"] for j in range(1, K + 1)
                         for sq in curriculum_item(it, stage=j, n_stages=K, mode="exact")["subqs"]]
                self.assertEqual(sorted(parts), sorted(ids))
        self.assertEqual(json.dumps(ITEMS), before)                                   # the items are not modified
        self.assertEqual(max(sq["step"] for it in ITEMS for sq in nodes(it)), 5)      # the fixture exercises the fold

    def test_refusals(self):
        from cotdistill.student import curriculum_item
        it = ITEMS[0]
        for kw in ({"stage": 0, "n_stages": 4}, {"stage": 5, "n_stages": 4}, {"stage": 1, "n_stages": 4, "mode": "x"}):
            with self.assertRaises(ValueError):
                curriculum_item(it, **kw)
        bad = dict(it, subqs=[{k: v for k, v in sq.items() if k != "step"} for sq in it["subqs"]])
        with self.assertRaises(ValueError):
            curriculum_item(bad, stage=1, n_stages=4)

    def test_views_660(self):
        """gsm8k-train-660 (steps 1-3, two nodes each): the stage views are GFPQ's views of the stage's nodes
        (problem without its options, the node question, Yes/No), weight 1 / views; exact stage 4 has none."""
        from cotdistill.student import build_full_tree, curriculum_item
        from cotdistill.treeplan import problem_no_options
        it = next(x for x in ITEMS if x["item_id"] == "gsm8k-train-660")
        kw = dict(mode="plain", cap=0, aux_total=1.0, final="teacher", subq_target="cot",
                  view="statements+question", sources={"gsm8k_tf"})
        q = {sq["id"]: sq["question"] for sq in nodes(it)}
        want = {1: ["s1v", "s1n"], 2: ["s1v", "s1n", "s2v", "s2n"], 3: list(q), 4: list(q)}
        for j, ids in want.items():
            g = build_full_tree(curriculum_item(it, stage=j, n_stages=4), rng=random.Random(0), **kw)
            self.assertEqual(g[0].kind, "final")
            views = g[1:]
            self.assertEqual([e.text.split("Intermediate question: ")[1].split("\n")[0] for e in views],
                             [q[i] for i in ids])
            for e in views:
                self.assertEqual(e.kind, "subq")
                self.assertTrue(e.text.startswith(problem_no_options(it) + "\n\nIntermediate question: "))
                self.assertNotIn("Options:\nA) 48", e.text)
                self.assertAlmostEqual(e.weight, 1.0 / len(ids))
        self.assertEqual(build_full_tree(curriculum_item(it, stage=4, n_stages=4, mode="exact"),
                                         rng=random.Random(0), **kw)[1:], [])
        fin = build_full_tree(dict(it, subqs=[]), rng=random.Random(0), **kw)
        self.assertEqual([e.kind for e in fin], ["final"])
        self.assertTrue(fin[0].text.endswith("E) 4\n\nAnswer:"))


@unittest.skipUnless(HAVE_TORCH, "needs torch + transformers")
class TestTrain(unittest.TestCase):
    @staticmethod
    def common(d):
        from test_tree_forcing import SCRIPT
        return [sys.executable, SCRIPT, "--model", d + "/m", "--train", d + "/t.jsonl", "--eval", "v=" + d + "/t.jsonl",
                "--items-per-update", "2", "--micro-bs", "4", "--grad-accum", "1", "--max-len", "640",
                "--eval-max-len", "640", "--precision", "fp32master", "--no-grad-ckpt", "--log-every", "1",
                "--lr", "1e-3", "--seed", "3", "--final", "teacher", "--subq-target", "cot"]

    GFPQ = ["--subq", "cot", "--tree-full", "plain", "--tree-view", "statements+question", "--tree-cap", "0",
            "--tree-sources", "gsm8k_tf"]

    def run_(self, cmd, out, *extra, script=None):
        from test_tree_forcing import ENV
        if script:
            cmd = [cmd[0], script] + cmd[2:]
        return subprocess.run(cmd + ["--out", out, *extra], capture_output=True, text=True, timeout=900, env=ENV)

    def test_existing_loops_match_ref(self):
        """GFPQ (full-tree loop), A2 (plain loop), A10 (fact stages, per-stage schedules, --reset-optim) and SIB (SI
        stages, stage-end evaluations): CUR_BASE_REF's losses, learning rates, sequences, trajectory and evaluation,
        update for update."""
        from test_tree_forcing import head_tree, tiny
        with tempfile.TemporaryDirectory() as d:
            if head_tree(d + "/h", CUR_BASE_REF) is None:
                self.skipTest("not a git checkout")
            tiny(d)
            Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in ITEMS))
            arms = {"GFPQ": self.GFPQ + ["--epochs", "2"], "A2": ["--subq", "none", "--epochs", "2"],
                    "A10": ["--subq", "none", "--fact-stages", "2", "--reset-optim", "--epochs", "1"],
                    "SIB": self.GFPQ + ["--si-stages", "2", "--si-final-only", "--si-remove", "back", "--epochs", "3"]}
            for arm, extra in arms.items():
                res = {}
                for name, script in (("ref", d + "/h/scripts/train_student.py"), ("now", None)):
                    r = self.run_(self.common(d), f"{d}/{name}-{arm}", *extra, script=script)
                    self.assertEqual(r.returncode, 0, (arm, name, r.stderr[-2000:]))
                    m = json.loads(Path(f"{d}/{name}-{arm}/metrics.json").read_text())
                    traj = [{k: v for k, v in t.items() if k != "s_elapsed"} for t in m["trajectory"]]   # (wall clock)
                    res[name] = ([(h["step"], h["loss"], h["lr"], h.get("seqs")) for h in m["history"]],
                                 traj, m["eval"]["v"], m.get("tree_view_stats"), m.get("si"))
                self.assertEqual(res["ref"], res["now"], arm)
                self.assertGreater(len(res["now"][0]), 2)
                self.assertNotIn("curriculum", json.loads(Path(f"{d}/now-{arm}/metrics.json").read_text()))

    def test_curriculum_end_to_end(self):
        """GFPQC through main(): K + 1 stages of epochs / (K + 1) (one pass each here), node views of the stage only and
        no final in the node stages, finals only in the last (with --curriculum-final-nodes: GFPQ's groups), one
        schedule, stage-end evaluations; exact mode with an empty stage; --reset-optim; the argument checks."""
        from cotdistill.student import curriculum_item
        from test_tree_forcing import tiny
        with tempfile.TemporaryDirectory() as d:
            tiny(d)
            Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in ITEMS))
            K, n_fin = 4, len(ITEMS)

            def views(j, k, mode="cumulative"):     # node views of stage j and the problems that have one
                c = [len(curriculum_item(it, stage=j, n_stages=k, mode=mode)["subqs"]) for it in ITEMS]
                return sum(c), sum(x > 0 for x in c)

            cur = ["--curriculum-steps", str(K), "--epochs", str(K + 1)]
            runs = {"C": cur, "CF": cur + ["--curriculum-final-nodes"], "CR": cur + ["--reset-optim"],
                    "CE": ["--curriculum-steps", "6", "--curriculum-mode", "exact", "--epochs", "7"]}
            ms = {}
            for name, extra in runs.items():
                r = self.run_(self.common(d) + self.GFPQ, f"{d}/{name}", *extra)
                self.assertEqual(r.returncode, 0, (name, r.stderr[-2000:]))
                ms[name] = json.loads(Path(f"{d}/{name}/metrics.json").read_text())
            all_nodes = sum(len(nodes(it)) for it in ITEMS)
            for name in ("C", "CF", "CR"):
                m = ms[name]
                h = m["history"]
                ups = [math.ceil(views(j, K)[1] / 2) for j in range(1, K + 1)] + [math.ceil(n_fin / 2)]
                self.assertEqual(len(h), sum(ups), name)
                self.assertEqual([c["stage"] for c in m["curriculum"]], [1, 2, 3, 4, "final"])
                self.assertEqual([c["pass"] for c in m["curriculum"]], [0, 1, 2, 3, 4])
                self.assertEqual([c["updates"] for c in m["curriculum"]], ups)
                at = 0
                for j, u in enumerate(ups, 1):
                    rows, at = h[at:at + u], at + u
                    c = m["curriculum"][j - 1]
                    if j <= K:
                        self.assertEqual((c["node_views"], c["problems"], c["finals"]), (*views(j, K), 0))
                        self.assertEqual(sum(x["seqs"] for x in rows), views(j, K)[0])
                        self.assertTrue(all(x["loss_final"] == 0.0 and x["loss_node"] > 0 for x in rows), (name, j))
                    elif name == "CF":
                        self.assertEqual((c["node_views"], c["finals"]), (all_nodes, n_fin))
                        self.assertEqual(sum(x["seqs"] for x in rows), all_nodes + n_fin)
                    else:
                        self.assertEqual((c["node_views"], c["finals"], c["problems"]), (0, n_fin, n_fin))
                        self.assertEqual(sum(x["seqs"] for x in rows), n_fin)
                        self.assertTrue(all(x["loss_node"] == 0.0 and x["loss_final"] > 0 for x in rows))
                self.assertEqual([t["epoch"] for t in m["trajectory"]], [1.0, 2.0, 3.0, 4.0])
                self.assertEqual([t["step"] for t in m["trajectory"]], [sum(ups[:k]) for k in (1, 2, 3, 4)])
                lrs = [x["lr"] for x in h]
                self.assertEqual(lrs, sorted(lrs, reverse=True))           # one schedule (warmup 1 step), no restart
                self.assertTrue(all(math.isfinite(x["loss"]) for x in h))
                self.assertEqual(m["args"]["curriculum_steps"], K)
            self.assertEqual([x["loss"] for x in ms["C"]["history"][:ups[0]]],
                             [x["loss"] for x in ms["CR"]["history"][:ups[0]]])   # the reset acts from stage 2 on
            self.assertNotEqual([x["loss"] for x in ms["C"]["history"]], [x["loss"] for x in ms["CR"]["history"]])
            # exact, K = 6: the fixture's steps end at 5, so stage 6 has no problem and no update
            m = ms["CE"]
            ups = [math.ceil(views(j, 6, "exact")[1] / 2) for j in range(1, 7)] + [math.ceil(n_fin / 2)]
            self.assertEqual(ups[5], 0)
            self.assertEqual(len(m["history"]), sum(ups))
            self.assertEqual([c["node_views"] for c in m["curriculum"][:5]], [views(j, 6, "exact")[0] for j in range(1, 6)])
            self.assertEqual([c["stage"] for c in m["curriculum"]], [1, 2, 3, 4, 5, "final"])   # stage 6 never built
            # refused
            bad_runs = ((self.GFPQ, ["--curriculum-mode", "exact"]), (self.GFPQ, ["--curriculum-final-nodes"]),
                        (["--subq", "none"], cur), (self.GFPQ, cur + ["--si-stages", "2"]),
                        (self.GFPQ, cur + ["--eval-at", "1"]), (self.GFPQ, cur + ["--stop-at-step", "2"]),
                        (self.GFPQ, cur + ["--tree-target", "shuffled"]),
                        ([x if x != "plain" else "true" for x in self.GFPQ], cur),
                        (self.GFPQ, cur + ["--node-format", "value"]),
                        ([x if x != "statements+question" else "problem" for x in self.GFPQ], cur + ["--node-format", "mc"]),
                        (self.GFPQ, ["--curriculum-steps", "-1"]))
            for base, extra in bad_runs:
                r = self.run_(self.common(d) + base, f"{d}/bad", *extra)
                self.assertNotEqual(r.returncode, 0, extra)
            Path(d + "/t2.jsonl").write_text("\n".join(json.dumps(dict(x, subqs=[{k: v for k, v in sq.items() if k != "step"}
                                                                                 for sq in x["subqs"]])) for x in ITEMS))
            cmd = [x if x != d + "/t.jsonl" else d + "/t2.jsonl" for x in self.common(d)]
            r = self.run_(cmd + self.GFPQ, f"{d}/bad2", *cur)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("node step None", r.stderr + r.stdout)


if __name__ == "__main__":
    unittest.main()
