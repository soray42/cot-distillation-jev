"""CPU tests for the 12B feasibility-review changes: the masters-only optimizer state (CPUOffloadAdamW.state_dict
(moments=False)), the quota-aware mem12b roundtrip (--rt-free-gb, --rt-moments none, OSError -> skip) and the
cost12b --no-resume / measured accumulation rate. Tiny models only."""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import torch

from cotdistill.offload import CPUOffloadAdamW

try:
    from test_mem12b import HAVE, Args, tiny_full
except ImportError:
    HAVE = False


def _toy(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(6, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3))


def _update(model, opt, x):
    model(x).pow(2).sum().backward()
    opt.clip_grad_norm_(1.0)
    opt.step()
    opt.zero_grad()


class TestLeanState(unittest.TestCase):
    def test_masters_only_state_restarts_adam(self):
        m = _toy()
        opt = CPUOffloadAdamW(m.parameters(), lr=1e-2, num_threads=1)
        x = torch.randn(4, 6)
        for _ in range(3):
            _update(m, opt, x)
        lean = opt.state_dict(moments=False)          # references: snapshot it the way torch.save would
        buf = io.BytesIO()
        torch.save(lean, buf)
        buf.seek(0)
        lean = torch.load(buf, weights_only=False)
        self.assertTrue(all(set(e) == {"master"} for e in lean["state"].values()))
        full = opt.state_dict()
        self.assertTrue(all({"master", "exp_avg", "exp_avg_sq", "step"} <= set(e) for e in full["state"].values()))
        masters = [t.clone() for t in opt.master]
        _update(m, opt, x)                               # move away, then load the lean state back
        opt.load_state_dict(lean)
        for a, b in zip(opt.master, masters):
            self.assertTrue(torch.equal(a, b))
        for p, a in zip(m.parameters(), masters):
            self.assertTrue(torch.equal(p.detach(), a))
        _update(m, opt, x)
        self.assertTrue(all(e["step"] == 1.0 for e in opt.state_dict()["state"].values()))   # AdamW restarted
        # the restarted update equals a fresh AdamW from the same masters
        ref = _toy()
        with torch.no_grad():
            for p, a in zip(ref.parameters(), masters):
                p.copy_(a)
        ro = CPUOffloadAdamW(ref.parameters(), lr=1e-2, num_threads=1)
        _update(ref, ro, x)
        for a, b in zip(opt.master, ro.master):
            self.assertTrue(torch.equal(a, b))
        opt.release()
        ro.release()


@unittest.skipUnless(HAVE, "needs transformers with Gemma4Unified")
class TestQuotaRoundtrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mem12b
        cls.mem12b = mem12b
        cls.tmp = tempfile.TemporaryDirectory()
        tiny_full().eval().save_pretrained(Path(cls.tmp.name) / "ckpt")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.rows = []
        orig = self.mem12b.emit
        self.mem12b.emit = self.rows.append
        self.addCleanup(setattr, self.mem12b, "emit", orig)
        m, _ = self.mem12b.load_text_only(str(Path(self.tmp.name) / "ckpt"), torch.bfloat16)
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.train()
        self.m = m
        self.opt = CPUOffloadAdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.0,
                                   num_threads=1)
        self.addCleanup(self.opt.release)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: 1.0 / (1 + s))
        self.lids = torch.tensor([200 + i for i in range(6)])

    def run_rt(self, d, **kw):
        a = Args(rt_dir=d)
        for k, v in kw.items():
            setattr(a, k, v)
        self.mem12b.part_roundtrip(self.m, self.opt, self.sched, a, self.lids, [2], 300, torch.device("cpu"))
        return self.rows[-1]

    def test_lean_roundtrip_ok(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.run_rt(d, rt_moments="none")
            self.assertEqual(list(Path(d).iterdir()), [])
        self.assertTrue(r["ok"], r)
        self.assertFalse(r["moments"])
        self.assertTrue(r["adam_restarted"] and r["load_exact"] and r["weights_restored"])

    def test_quota_cap_skips_without_writing(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.run_rt(d, rt_free_gb=0.0)
            self.assertEqual(list(Path(d).iterdir()), [])
        self.assertIn("skip", r)
        self.assertNotIn("ok", r)

    def test_oserror_is_a_skip_and_cleans_up(self):
        real = torch.save

        def full_disk(obj, f, *a, **k):
            Path(f).write_bytes(b"partial")
            raise OSError(122, "Disk quota exceeded")
        torch.save = full_disk
        self.addCleanup(setattr, torch, "save", real)
        with tempfile.TemporaryDirectory() as d:
            r = self.run_rt(d)                           # must not raise SystemExit
            self.assertEqual(list(Path(d).iterdir()), [])
        self.assertIn("quota", r["skip"])


class TestCostReview(unittest.TestCase):
    def setUp(self):
        import cost12b
        self.c = cost12b

    def test_acc_rate_moves_the_floor(self):
        self.assertAlmostEqual(self.c.offload_floor(16, 60, 120), 11.90735 * 2 / 16, places=3)   # PCIe binds
        self.assertAlmostEqual(self.c.offload_floor(16, 60, None), self.c.offload_floor(16, 60), places=9)

    def test_from_log_reads_acc(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "log"
            f.write_text("MEM12B " + json.dumps({"part": "bench", "d2h_gbs": 14.0, "acc_add_bf16_gparam_s": 5.5})
                         + "\n")
            m = self.c.from_log(str(f))
        self.assertEqual((m["pcie"], m["acc_gbs"]), (14.0, 55.0))

    @unittest.skipUnless((ROOT / "notes/next_round/tree_forcing/cost.json").exists()
                         and (ROOT / "data/student_v3f/train.jsonl").exists(), "cost.json / v3f not in this checkout")
    def test_no_resume_has_no_job_chains(self):
        import contextlib
        import io
        out = {}
        for flag in ([], ["--no-resume"]):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.c.main(["--md"] + flag)
            out[bool(flag)] = buf.getvalue()
        self.assertIn("jobs", out[False])
        self.assertNotIn(" jobs", out[True].split("\n", 2)[2])


if __name__ == "__main__":
    unittest.main()
