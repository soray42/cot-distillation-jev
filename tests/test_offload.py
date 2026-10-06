"""CPU tests for the CPU-offloaded AdamW (src/cotdistill/offload.py); the CUDA cases run only with COTD_CUDA_TESTS=1."""
import io
import math
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
    HAVE = True
except ImportError:
    HAVE = False

if HAVE:
    from cotdistill.offload import CPUOffloadAdamW, estimate, _probe_impl


def tiny(dtype=None, device="cpu", seed=0):
    torch.manual_seed(seed)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(11, 8)
            self.l1 = torch.nn.Linear(8, 16)
            self.l2 = torch.nn.Linear(16, 8)
            self.norm = torch.nn.LayerNorm(8)
            self.unused = torch.nn.Linear(3, 3)      # never receives a gradient

        def forward(self, x):
            h = self.emb(x).mean(1)
            h = self.norm(self.l2(torch.tanh(self.l1(h))))
            return h @ self.emb.weight.T             # tied head: one parameter used twice

    m = Net()
    if dtype is not None:
        m = m.to(dtype)
    return m.to(device)


def batches(steps, ga, seed=1, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    return [[(torch.randint(0, 11, (4, 5), generator=g).to(device), torch.randint(0, 11, (4,), generator=g).to(device))
             for _ in range(ga)] for _ in range(steps)]


def groups(model, lr):
    decay = [model.l1.weight, model.l2.weight, model.emb.weight]
    ids = {id(p) for p in decay}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return [{"params": decay, "weight_decay": 0.1}, {"params": rest, "lr": lr * 3}]


def sched(opt, total=20, warm=4):
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, s / total))))


def run(model, opt, data, clip, offload, sch=None, autocast=False):
    norms = []
    for mbs in data:
        for x, y in mbs:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                loss = torch.nn.functional.cross_entropy(model(x).float(), y) / len(mbs)
            loss.backward()
        if clip:
            n = opt.clip_grad_norm_(clip) if offload else torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            norms.append(float(n))
        opt.step()
        if sch is not None:
            sch.step()
        opt.zero_grad(set_to_none=True)
    return norms


@unittest.skipUnless(HAVE, "needs torch")
class TestOffloadCPU(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def compare(self, impl, async_acc, clip=0.5, bucket=64):
        lr = 3e-3
        ref, ours = tiny(), tiny()
        ropt = torch.optim.AdamW(groups(ref, lr), lr=lr, weight_decay=0.0, foreach=False, fused=False)
        oopt = CPUOffloadAdamW(groups(ours, lr), lr=lr, impl=impl, async_accumulate=async_acc,
                               bucket_numel=bucket, num_threads=2)
        rs, os_ = sched(ropt), sched(oopt)
        data = batches(20, 4)
        rn = run(ref, ropt, data, clip, False, rs)
        on = run(ours, oopt, data, clip, True, os_)
        self.assertTrue(any(n > clip for n in rn), "the clip never triggered; the test would not cover it")
        for a, b in zip(rn, on):
            self.assertAlmostEqual(a, b, delta=1e-5 * max(1.0, a))
        for (n, a), b in zip(ref.named_parameters(), ours.parameters()):
            torch.testing.assert_close(b, a, atol=2e-6, rtol=1e-5, msg=f"{impl} {n}")
        self.assertTrue(torch.equal(ours.unused.weight, tiny().unused.weight), "a never-used param moved")
        self.assertEqual([g["lr"] for g in ropt.param_groups], [g["lr"] for g in oopt.param_groups])
        return oopt

    def test_matches_torch_adamw_all_impls(self):
        impls = ["for", "foreach"]
        try:
            _probe_impl("fused")
            impls.append("fused")
        except Exception:
            pass
        for impl in impls:
            for async_acc in (True, False):
                with self.subTest(impl=impl, async_acc=async_acc):
                    self.compare(impl, async_acc)

    def test_no_clip_and_one_bucket(self):
        lr = 1e-3
        ref, ours = tiny(), tiny()
        ropt = torch.optim.AdamW(ref.parameters(), lr=lr, weight_decay=0.0, foreach=False)
        oopt = CPUOffloadAdamW(ours.parameters(), lr=lr, bucket_numel=10 ** 9)
        data = batches(20, 3, seed=5)
        run(ref, ropt, data, 0.0, False)
        run(ours, oopt, data, 0.0, True)
        for a, b in zip(ref.parameters(), ours.parameters()):
            torch.testing.assert_close(b, a, atol=2e-6, rtol=1e-5)

    def test_grads_leave_the_device_params(self):
        m = tiny()
        opt = CPUOffloadAdamW(m.parameters(), lr=1e-3)
        x, y = batches(1, 1)[0][0]
        torch.nn.functional.cross_entropy(m(x), y).backward()
        self.assertTrue(all(p.grad is None for p in m.parameters()))
        gs = opt.grads()
        self.assertIsNone(gs[-1])                      # the unused Linear's bias
        self.assertTrue(all(g.dtype == torch.float32 for g in gs if g is not None))

    def test_reentrant_checkpoint_shared_param(self):
        """With reentrant checkpointing a parameter used in two segments gets its grad accumulated (and the hook
        fired) once per segment; the offloaded accumulator must still hold the sum."""
        from torch.utils.checkpoint import checkpoint

        def make():
            torch.manual_seed(7)
            return torch.nn.Linear(6, 6), torch.nn.Linear(6, 2)

        def fwd(shared, head, x, reentrant):
            h = checkpoint(lambda t: torch.tanh(shared(t)), x, use_reentrant=reentrant)
            h = checkpoint(lambda t: torch.tanh(shared(t)), h, use_reentrant=reentrant)
            return head(h).sum()

        x = torch.randn(3, 6, requires_grad=True)       # reentrant checkpoint needs an input that requires grad
        for reentrant in (True, False):
            s_ref, h_ref = make()
            s, h = make()
            opt = CPUOffloadAdamW(list(s.parameters()) + list(h.parameters()), lr=1e-3)
            fired = []
            s.weight.register_post_accumulate_grad_hook(lambda p: fired.append(1))
            for _ in range(2):                           # two micro-steps
                fwd(s_ref, h_ref, x, reentrant).backward()
                fwd(s, h, x, reentrant).backward()
            ref = [p.grad for p in list(s_ref.parameters()) + list(h_ref.parameters())]
            for a, b in zip(ref, opt.grads()):
                torch.testing.assert_close(b, a, atol=1e-6, rtol=1e-6)
            self.assertEqual(len(fired), 4 if reentrant else 2, "hook firing count changed; recheck the design")

    def test_zero_grad_restarts_accumulation(self):
        m = tiny()
        opt = CPUOffloadAdamW(m.parameters(), lr=1e-3)
        (x1, y1), (x2, y2) = batches(1, 2)[0]
        torch.nn.functional.cross_entropy(m(x1), y1).backward()
        opt.zero_grad()
        torch.nn.functional.cross_entropy(m(x2), y2).backward()
        got = [g.clone() if g is not None else None for g in opt.grads()]
        ref = tiny()
        torch.nn.functional.cross_entropy(ref(x2), y2).backward()
        for g, p in zip(got, ref.parameters()):
            if p.grad is None:
                self.assertIsNone(g)
            else:
                self.assertTrue(torch.equal(g, p.grad))

    def test_state_round_trip_exact(self):
        lr = 2e-3
        a = tiny()
        oa = CPUOffloadAdamW(groups(a, lr), lr=lr, bucket_numel=100)
        sa = sched(oa)
        data = batches(10, 2, seed=3)
        run(a, oa, data[:5], 1.0, True, sa)
        buf = io.BytesIO()
        torch.save({"opt": oa.state_dict(), "sched": sa.state_dict(), "model": a.state_dict()}, buf)
        buf.seek(0)
        ck = torch.load(buf, weights_only=False)

        b = tiny(seed=123)                              # different init: everything must come from the checkpoint
        ob = CPUOffloadAdamW(groups(b, lr), lr=lr, bucket_numel=100)
        sb = sched(ob)
        ob.load_state_dict(ck["opt"])
        sb.load_state_dict(ck["sched"])
        for p, q in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(p, q))
        sd_a, sd_b = oa.state_dict(), ob.state_dict()
        for i in sd_a["state"]:
            self.assertEqual(sd_a["state"][i].keys(), sd_b["state"][i].keys())
            for k, v in sd_a["state"][i].items():
                self.assertTrue(torch.equal(v, sd_b["state"][i][k]) if torch.is_tensor(v) else v == sd_b["state"][i][k])
        self.assertEqual([g["lr"] for g in oa.param_groups], [g["lr"] for g in ob.param_groups])
        run(a, oa, data[5:], 1.0, True, sa)
        run(b, ob, data[5:], 1.0, True, sb)
        for p, q in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(p, q))

    def test_scheduler_drives_lr(self):
        m = tiny()
        opt = CPUOffloadAdamW(groups(m, 1e-2), lr=1e-2)
        s = sched(opt, total=10, warm=5)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 1e-2 * 0.2)
        self.assertAlmostEqual(opt.param_groups[1]["lr"], 3e-2 * 0.2)
        run(m, opt, batches(3, 1), 0.0, True, s)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], s.get_last_lr()[0])
        for _, inner in opt._buckets:                  # the inner optimizers got the scheduled lr at the last step
            for ig in inner.param_groups:
                self.assertIn(ig["_outer"], (0, 1))
        # zero lr and no weight decay leaves the weights untouched
        for g in opt.param_groups:
            g["lr"], g["weight_decay"] = 0.0, 0.0
        before = [p.detach().clone() for p in m.parameters()]
        run(m, opt, batches(1, 1, seed=9), 0.0, True)
        for p, q in zip(before, m.parameters()):
            self.assertTrue(torch.equal(p, q))

    def test_bf16_params_keep_fp32_master(self):
        m = tiny(dtype=torch.bfloat16)
        opt = CPUOffloadAdamW(m.parameters(), lr=1e-5)  # updates far below one bf16 ulp of O(0.1) weights
        start = [p.detach().clone() for p in m.parameters()]
        run(m, opt, batches(5, 2), 0.0, True)
        moved_master = 0
        for p, mast, s0 in zip(m.parameters(), opt.master, start):
            self.assertEqual(mast.dtype, torch.float32)
            self.assertTrue(torch.equal(p.detach(), mast.to(torch.bfloat16)))
            moved_master += int((mast != s0.float()).sum())
        self.assertGreater(moved_master, 0)
        # what a bf16-master optimizer would do: most weights never move
        self.assertLess(sum(int((p.detach() != s0).sum()) for p, s0 in zip(m.parameters(), start)), moved_master)

    def test_host_bytes_and_release(self):
        m = tiny()
        n = sum(p.numel() for p in m.parameters())
        opt = CPUOffloadAdamW(m.parameters(), lr=1e-3)
        self.assertEqual(opt.host_bytes(), n * (8 + 4))
        run(m, opt, batches(1, 1), 0.0, True)
        self.assertGreater(opt.host_bytes(), n * 12)
        opt.release()
        self.assertEqual(opt.host_bytes(), 0)
        torch.nn.functional.cross_entropy(m(batches(1, 1)[0][0][0]), batches(1, 1)[0][0][1]).backward()
        self.assertIsNotNone(m.l1.weight.grad)         # hooks removed: plain autograd again

    def test_estimate(self):
        e = estimate(11.9e9, 16)
        self.assertAlmostEqual(e["host_GB"], 11.9 * 18, places=6)
        self.assertAlmostEqual(e["d2h_GB"], 16 * 23.8, places=6)


@unittest.skipUnless(HAVE and os.environ.get("COTD_CUDA_TESTS") == "1" and torch.cuda.is_available(),
                     "GPU test: set COTD_CUDA_TESTS=1 (in-job smoke) on a CUDA machine")
class TestOffloadCUDA(unittest.TestCase):
    def test_cuda_fp32_matches_torch(self):
        lr = 3e-3
        ref, ours = tiny(device="cuda"), tiny(device="cuda")
        ropt = torch.optim.AdamW(ref.parameters(), lr=lr, weight_decay=0.0)
        oopt = CPUOffloadAdamW(ours.parameters(), lr=lr, bucket_numel=64, max_inflight_bytes=256)
        data = batches(20, 4, device="cuda")
        run(ref, ropt, data, 0.5, False)
        run(ours, oopt, data, 0.5, True)
        torch.cuda.synchronize()
        for a, b in zip(ref.parameters(), ours.parameters()):
            torch.testing.assert_close(b, a, atol=5e-6, rtol=1e-5)

    def test_cuda_bf16_weights_follow_master(self):
        m = tiny(dtype=torch.bfloat16, device="cuda")
        opt = CPUOffloadAdamW(m.parameters(), lr=1e-4)
        run(m, opt, batches(6, 3, device="cuda"), 1.0, True)
        torch.cuda.synchronize()
        for p, mast in zip(m.parameters(), opt.master):
            self.assertTrue(torch.equal(p.detach().cpu(), mast.to(torch.bfloat16)))


if __name__ == "__main__":
    unittest.main()
