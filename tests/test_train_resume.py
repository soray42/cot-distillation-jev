"""CPU tests for the 4B training infrastructure in scripts/train_student.py: token-budget micro-batches (same update
as fixed micro-batches), checkpoint/resume (bit-identical continuation, exit code 3) and the torchao optimizer path
(plumbing only, through a stub; the real torchao state round trip is an HPC smoke item, hpc/mem4b.sbatch)."""
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import torch
    import transformers  # noqa: F401
    import test_student as ts
    HAVE = True
except ImportError:
    HAVE = False

SCRIPT = str(Path(__file__).resolve().parents[1] / "scripts/train_student.py")
STUB = '''import torch


class AdamW8bit(torch.optim.AdamW):
    """Test stub of torchao.optim.AdamW8bit: plain AdamW with the LR stored as a tensor, as torchao does."""
    def __init__(self, params, lr=1e-3, weight_decay=1e-2, bf16_stochastic_round=False):
        assert bf16_stochastic_round and weight_decay == 0.0
        super().__init__(params, lr=torch.tensor(lr), weight_decay=weight_decay, foreach=False)
'''


def items(n):
    """Full-tree items whose prompts differ in length, so length-sorted micro-batches differ from fixed ones."""
    return [dict(ts.TestFullTree.ITEM, item_id=f"i{j}",
                 prompt="P " + "word " * (5 * j) + "\nOptions:\nA) a\nB) b\nC) c") for j in range(n)]


def run(d, *extra, env=None):
    return subprocess.run([sys.executable, SCRIPT, "--model", d + "/m", "--train", d + "/t.jsonl",
                           "--eval", "v=" + d + "/t.jsonl", "--tree-full", "true", "--items-per-update", "2",
                           "--max-len", "128", "--no-grad-ckpt", "--log-every", "1", *extra],
                          capture_output=True, text=True, timeout=900,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", **(env or {})})


def setup_dir(d, n):
    tok, model = ts.tiny()
    model.save_pretrained(d + "/m")
    tok.save_pretrained(d + "/m")
    Path(d + "/t.jsonl").write_text("\n".join(json.dumps(it) for it in items(n)))


def ckpt(out):
    load = lambda f, **kw: torch.load(Path(out) / "ckpt" / f, map_location="cpu", **kw)
    return load("model.pt", weights_only=True), load("optim.pt", weights_only=False), load("state.pt", weights_only=False)


class TestTokenBudgetPartition(unittest.TestCase):
    @unittest.skipUnless(HAVE, "needs torch")
    def test_partition(self):
        from cotdistill.student import token_budget_batches
        rng = random.Random(0)
        for _ in range(200):
            lens = [rng.randint(1, 90) for _ in range(rng.randint(1, 40))]
            budget, cap = rng.choice([64, 100, 300]), rng.choice([2, 4, 16])
            mbs = token_budget_batches(lens, budget, cap)
            flat = [i for b in mbs for i in b]
            self.assertEqual(sorted(flat), list(range(len(lens))))                   # every sequence exactly once
            self.assertEqual([lens[i] for i in flat], sorted(lens, reverse=True))    # longest first
            for b in mbs:
                self.assertLessEqual(len(b), cap)
                self.assertTrue(len(b) == 1 or len(b) * lens[b[0]] <= budget)        # padded size within budget
        self.assertEqual(token_budget_batches([70, 10, 20], 60), [[0], [2, 1]])      # over budget: alone


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestTokenBudgetGradient(unittest.TestCase):
    def test_update_gradient_does_not_depend_on_the_split(self):
        """The full-tree loop's update gradient (sum over micro-batches of kl_loss x len(mb) / norm) in fp32 on the
        tiny model: fixed micro-batches vs several token-budget splits, with label smoothing, unequal example weights
        (final weight, aux_total 0.5, a zero-weight view) and max_len truncation. Without the len(mb) factor the
        gradients differ, so the comparison does detect a normalisation error."""
        from cotdistill.student import build_full_tree, kl_loss, label_logits, micro_batches
        tok, model = ts.tiny()
        rng = random.Random(0)
        upd = [e for it in items(5) for e in build_full_tree(dict(it, weight=1.5), mode="true", rng=rng,
                                                              aux_total=0.5, permute_final=0.5)]
        upd[3].weight = 0.0
        self.assertGreater(len({e.weight for e in upd}), 2)
        norm, cache = 3 * 5, {}

        def grad(mbs, max_len, by_len=True):
            model.zero_grad(set_to_none=True)
            for mb in mbs:
                loss = kl_loss(label_logits(model, tok, mb, max_len, cache), mb, 0.1) * (len(mb) if by_len else 1) / norm
                loss.backward()
            return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

        def close(g, h):
            return all((g[k] - h[k]).abs().max().item() <= 1e-5 * max(g[k].abs().max().item(), 1e-8) for k in g)

        for max_len in (128, 24):                           # 24 truncates the longer prompts (collate keeps the end)
            fixed = micro_batches(tok, upd, max_len, 2)
            ref = grad(fixed, max_len)
            sizes = {tuple(map(len, fixed))}
            for budget, cap in ((300, 16), (120, 4), (60, 3), (1, 16)):
                split = micro_batches(tok, upd, max_len, 2, budget, cap)
                self.assertEqual(sorted(id(e) for mb in split for e in mb), sorted(id(e) for e in upd))
                sizes.add(tuple(map(len, split)))
                self.assertTrue(close(ref, grad(split, max_len)), (max_len, budget, cap))
            self.assertGreaterEqual(len(sizes), 4)          # the splits really differ
            self.assertFalse(close(ref, grad(micro_batches(tok, upd, max_len, 2, 120, 4), max_len, by_len=False)))


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestTokenBudgetTraining(unittest.TestCase):
    def test_same_updates_as_fixed_micro_batches(self):
        with tempfile.TemporaryDirectory() as d:
            setup_dir(d, 6)                                     # 3 updates per pass; stop (checkpoint) after 2
            common = ["--epochs", "1", "--lr", "1e-3", "--precision", "fp32master", "--stop-at-step", "2"]
            fixed = run(d, *common, "--micro-bs", "3", "--out", d + "/a")
            budget = run(d, *common, "--micro-bs", "3", "--token-budget", "300", "--out", d + "/b")
            self.assertEqual(fixed.returncode, 3, fixed.stderr[-2000:])
            self.assertEqual(budget.returncode, 3, budget.stderr[-2000:])
            w0 = transformers.AutoModelForCausalLM.from_pretrained(d + "/m").state_dict()
            (wa, _, sa), (wb, _, sb) = ckpt(d + "/a"), ckpt(d + "/b")
            fwd = [r["fwd"] for r in sb["history"]]
            self.assertNotIn("fwd", sa["history"][0])                                 # rows unchanged when off
            self.assertNotEqual(fwd, [-(-r["seqs"] // 3) for r in sa["history"]])     # the partitions differ
            for ra, rb in zip(sa["history"], sb["history"]):
                self.assertAlmostEqual(ra["loss"], rb["loss"], places=5)
            for k in wa:
                da, db = wa[k] - w0[k], wb[k] - w0[k]
                self.assertLessEqual((da - db).abs().max().item(), 1e-3 * max(da.abs().max().item(), 1e-6), k)
            self.assertGreater(max((wa[k] - w0[k]).abs().max().item() for k in wa), 1e-4)  # the weights did move


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestResume(unittest.TestCase):
    def test_resumed_run_is_identical(self):
        with tempfile.TemporaryDirectory() as d:
            setup_dir(d, 5)                                     # 3 updates per pass x 2 epochs = 6 updates
            common = ["--epochs", "2", "--precision", "fp32master", "--permute-final", "0.5", "--seed", "3"]
            straight = run(d, *common, "--stop-at-step", "4", "--out", d + "/a")
            self.assertEqual(straight.returncode, 3, straight.stderr[-2000:])
            self.assertIn("planned stop after step 4/6", straight.stdout)
            first = run(d, *common, "--stop-at-step", "2", "--out", d + "/b")
            self.assertEqual(first.returncode, 3, first.stderr[-2000:])
            self.assertFalse(Path(d + "/b/metrics.json").exists())
            other = run(d, *common, "--lr", "2e-5", "--stop-at-step", "4", "--out", d + "/b")
            self.assertNotEqual(other.returncode, 0)                                  # refuses other training args
            self.assertIn("other training arguments", other.stderr)
            second = run(d, *common, "--stop-at-step", "4", "--out", d + "/b")          # resumes across the epoch end
            self.assertEqual(second.returncode, 3, second.stderr[-2000:])
            self.assertIn("resumed from", second.stdout)
            (wa, oa, sa), (wb, ob, sb) = ckpt(d + "/a"), ckpt(d + "/b")
            self.assertEqual(wa.keys(), wb.keys())
            for k in wa:
                self.assertTrue(torch.equal(wa[k], wb[k]), k)
            for p, st in oa["state"].items():
                for k, v in st.items():
                    self.assertTrue(torch.equal(v, ob["state"][p][k]), (p, k))
            self.assertEqual(oa["param_groups"], ob["param_groups"])
            strip = lambda h: [{k: v for k, v in r.items() if k not in ("s_per_step", "approx_tok_per_s")} for r in h]
            self.assertEqual(strip(sa["history"]), strip(sb["history"]))
            self.assertEqual(len(sa["history"]), 4)
            for k in ("step", "rng", "order_rng", "sched", "passes", "seen_tok"):
                self.assertEqual(sa[k], sb[k], k)
            # both to the end (any resume flag turns auto-resume on): same predictions, checkpoint removed
            for o in ("a", "b"):
                done = run(d, *common, "--ckpt-every-min", "1000", "--out", f"{d}/{o}")
                self.assertEqual(done.returncode, 0, done.stderr[-2000:])
                self.assertFalse(Path(f"{d}/{o}/ckpt").exists())
            self.assertEqual(Path(d + "/a/preds_v.jsonl").read_text(), Path(d + "/b/preds_v.jsonl").read_text())
            ha, hb = (json.loads(Path(f"{d}/{o}/metrics.json").read_text())["history"] for o in ("a", "b"))
            self.assertEqual(strip(ha), strip(hb))
            self.assertEqual([r["step"] for r in ha], [1, 2, 3, 4, 5, 6])

    def test_recovery_after_a_kill_during_the_swap(self):
        """A kill between the two renames leaves the new checkpoint in ckpt.tmp and the previous one in ckpt.old: the
        rerun continues from ckpt.tmp (the newer); leftover ckpt.old / incomplete ckpt.tmp next to a ckpt are removed;
        --ckpt-single (one copy on disk) continues the same way."""
        import shutil
        with tempfile.TemporaryDirectory() as d:
            setup_dir(d, 5)
            common = ["--epochs", "2", "--precision", "fp32master", "--seed", "1"]
            for o, at in (("a", 5), ("b", 1), ("c", 2)):
                r = run(d, *common, "--stop-at-step", str(at), "--out", f"{d}/{o}")
                self.assertEqual(r.returncode, 3, r.stderr[-2000:])
            shutil.copytree(d + "/b/ckpt", d + "/c/ckpt.old")              # step 1: older, complete
            Path(d + "/c/ckpt").rename(d + "/c/ckpt.tmp")                  # step 2: newer, complete, not swapped in
            r = run(d, *common, "--stop-at-step", "3", "--ckpt-single", "--out", d + "/c")
            self.assertEqual(r.returncode, 3, r.stderr[-2000:])
            self.assertIn("recovered", r.stdout)
            self.assertIn("resumed from", r.stdout)
            self.assertIn("at step 2", r.stdout)
            Path(d + "/c/ckpt").rename(d + "/c/ckpt.tmp")                  # step 3, complete, newer than ...
            shutil.copytree(d + "/b/ckpt", d + "/c/ckpt")                  # ... a step-1 ckpt (kill before the swap)
            shutil.copytree(d + "/b/ckpt", d + "/c/ckpt.old")
            r = run(d, *common, "--stop-at-step", "4", "--out", d + "/c")
            self.assertEqual(r.returncode, 3, r.stderr[-2000:])
            self.assertIn("at step 3", r.stdout)
            shutil.copytree(d + "/b/ckpt", d + "/c/ckpt.old")              # leftovers next to a complete ckpt
            (Path(d) / "c/ckpt.tmp").mkdir()
            (Path(d) / "c/ckpt.tmp/model.pt").write_bytes(b"partial")
            r = run(d, *common, "--stop-at-step", "5", "--out", d + "/c")
            self.assertEqual(r.returncode, 3, r.stderr[-2000:])
            self.assertIn("at step 4", r.stdout)
            self.assertEqual(sorted(p.name for p in Path(d + "/c").glob("ckpt*")), ["ckpt"])
            (wa, _, sa), (wc, _, sc) = ckpt(d + "/a"), ckpt(d + "/c")
            for k in wa:
                self.assertTrue(torch.equal(wa[k], wc[k]), k)
            self.assertEqual((sa["step"], sa["rng"], sa["order_rng"]), (sc["step"], sc["rng"], sc["order_rng"]))
            r = run(d, *common, "--ckpt-single", "--out", d + "/o")       # needs a resume flag
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("--ckpt-single needs", r.stderr)

    def test_flag_checks(self):
        with tempfile.TemporaryDirectory() as d:
            setup_dir(d, 2)
            r = run(d, "--precision", "fp32master", "--optim", "adamw8bit-sr", "--out", d + "/o")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("needs --precision bf16", r.stderr)
            r = subprocess.run([sys.executable, SCRIPT, "--model", d + "/m", "--train", d + "/t.jsonl",
                                "--token-budget", "300", "--out", d + "/o"], capture_output=True, text=True,
                               timeout=300, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("full-tree loop only", r.stderr)
            if importlib.util.find_spec("torchao") is None:
                r = run(d, "--precision", "bf16", "--optim", "adamw8bit-sr", "--out", d + "/o")
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("needs torchao (pip install", r.stderr)

    @unittest.skipIf(HAVE and importlib.util.find_spec("torchao") is not None, "real torchao installed")
    def test_torchao_path_with_stub(self):
        """Tensor LR (torchao keeps it as a tensor) through the scheduler, the JSON log and a checkpoint round trip."""
        with tempfile.TemporaryDirectory() as d:
            setup_dir(d, 4)
            (Path(d) / "stub/torchao/optim").mkdir(parents=True)
            (Path(d) / "stub/torchao/__init__.py").write_text('__version__ = "0.0.0"\n')   # transformers reads it
            (Path(d) / "stub/torchao/optim/__init__.py").write_text(STUB)
            (Path(d) / "stub/torchao-0.0.0.dist-info").mkdir(parents=True)   # hide a real torchao from transformers
            (Path(d) / "stub/torchao-0.0.0.dist-info/METADATA").write_text("Metadata-Version: 2.1\nName: torchao\nVersion: 0.0.0\n")
            env = {"PYTHONPATH": d + "/stub" + os.pathsep + os.environ.get("PYTHONPATH", "")}
            common = ["--epochs", "1", "--precision", "bf16", "--optim", "adamw8bit-sr", "--token-budget", "300"]
            r = run(d, *common, "--stop-at-step", "1", "--out", d + "/o", env=env)
            self.assertEqual(r.returncode, 3, r.stderr[-2000:])
            r = run(d, *common, "--ckpt-every-min", "1000", "--out", d + "/o", env=env)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertIn("resumed from", r.stdout)
            hist = json.loads(Path(d + "/o/metrics.json").read_text())["history"]
            self.assertEqual([r["step"] for r in hist], [1, 2])
            self.assertTrue(all(isinstance(r["lr"], float) for r in hist))


if __name__ == "__main__":
    unittest.main()
