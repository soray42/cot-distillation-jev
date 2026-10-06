"""Numerics review additions for src/cotdistill/offload.py and scripts/mem12b.py (CPU only, tiny tensors):
reset_state (the --reset-optim replacement for opt.state.clear()), the skip-on-non-finite pattern, host_bytes with
pinned rounding (CUDA only), and the chunked position-weighted bit digest of the 12B roundtrip."""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import torch
    HAVE = True
except ImportError:
    HAVE = False

if HAVE:
    from cotdistill.offload import CPUOffloadAdamW


def net(seed=0, dtype=None, device="cpu"):
    torch.manual_seed(seed)
    m = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3))
    if dtype is not None:
        m = m.to(dtype)
    return m.to(device)


def one_update(model, opt, x, y, clip=None, offload=True):
    torch.nn.functional.cross_entropy(model(x).float(), y).backward()
    if clip:
        opt.clip_grad_norm_(clip) if offload else torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    opt.zero_grad()


@unittest.skipUnless(HAVE, "needs torch")
class TestResetState(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_reset_state_equals_fresh_adamw(self):
        """After reset_state the next update is that of a fresh AdamW started from the current weights (step 1, m = v
        = 0); opt.state.clear() would be a silent no-op on this class."""
        g = torch.Generator().manual_seed(0)
        data = [(torch.randn(5, 6, generator=g), torch.randint(0, 3, (5,), generator=g)) for _ in range(6)]
        ours = net()
        opt = CPUOffloadAdamW(ours.parameters(), lr=1e-2, bucket_numel=20)
        for x, y in data[:4]:
            one_update(ours, opt, x, y)
        opt.state.clear()                                   # what train_student --reset-optim calls today
        self.assertTrue(all(opt._inner_state(i) for i in range(len(opt._params))), "state.clear() did reset; recheck")
        opt.reset_state()
        self.assertTrue(not any(opt._inner_state(i) for i in range(len(opt._params))))
        ref = net()
        ref.load_state_dict(ours.state_dict())
        ropt = torch.optim.AdamW(ref.parameters(), lr=1e-2, weight_decay=0.0, foreach=False)
        for x, y in data[4:]:
            one_update(ours, opt, x, y)
            one_update(ref, ropt, x, y, offload=False)
        for a, b in zip(ref.parameters(), ours.parameters()):
            torch.testing.assert_close(b, a, atol=2e-6, rtol=1e-5)
        self.assertEqual(float(opt._inner_state(0)["step"]), 2.0)

    def test_skip_nonfinite_update_leaves_state_untouched(self):
        """The integration pattern for a bad micro-batch: total = clip_grad_norm_(); if not finite, zero_grad() and
        no step(). The masters, moments and weights must be exactly as before, and the next update must work."""
        m = net()
        opt = CPUOffloadAdamW(m.parameters(), lr=1e-2)
        x, y = torch.randn(4, 6), torch.randint(0, 3, (4,))
        one_update(m, opt, x, y)
        before = [t.clone() for t in opt.master] + [opt._inner_state(i)["exp_avg_sq"].clone()
                                                    for i in range(len(opt._params))]
        w0 = [p.detach().clone() for p in m.parameters()]
        (m(x).sum() * float("inf")).backward()
        total = opt.clip_grad_norm_(1.0)
        self.assertFalse(bool(torch.isfinite(total)))
        opt.zero_grad()                                     # skip: no step()
        after = [t for t in opt.master] + [opt._inner_state(i)["exp_avg_sq"] for i in range(len(opt._params))]
        for a, b in zip(before, after):
            self.assertTrue(torch.equal(a, b))
        for a, p in zip(w0, m.parameters()):
            self.assertTrue(torch.equal(a, p))
        one_update(m, opt, x, y, clip=1.0)
        self.assertTrue(all(bool(torch.isfinite(p).all()) for p in m.parameters()))

    def test_host_bytes_unpinned_exact(self):
        m = net()
        opt = CPUOffloadAdamW(m.parameters(), lr=1e-3)
        n = sum(p.numel() for p in m.parameters())
        self.assertEqual(opt.host_bytes(), n * 12)


@unittest.skipUnless(HAVE and os.environ.get("COTD_CUDA_TESTS") == "1" and torch.cuda.is_available(),
                     "GPU test: set COTD_CUDA_TESTS=1 (in-job smoke) on a CUDA machine")
class TestOffloadNumericsCUDA(unittest.TestCase):
    def test_pinned_rounding_in_host_bytes_and_reset(self):
        m = net(dtype=torch.bfloat16, device="cuda")
        opt = CPUOffloadAdamW(m.parameters(), lr=1e-3)
        n = sum(p.numel() for p in m.parameters())
        rounded = sum(1 << (p.numel() * 2 - 1).bit_length() for p in m.parameters())
        self.assertEqual(opt.host_bytes(), n * 8 + rounded)
        x, y = torch.randn(4, 6, device="cuda", dtype=torch.bfloat16), torch.randint(0, 3, (4,), device="cuda")
        one_update(m, opt, x, y, clip=1.0)
        opt.reset_state()
        one_update(m, opt, x, y, clip=1.0)
        torch.cuda.synchronize()
        for p, mast in zip(m.parameters(), opt.master):
            self.assertTrue(torch.equal(p.detach().cpu(), mast.to(torch.bfloat16)))
        opt.release()


@unittest.skipUnless(HAVE, "needs torch")
class TestBitDigest(unittest.TestCase):
    def setUp(self):
        try:
            import mem12b
        except Exception as e:                              # transformers missing etc.
            self.skipTest(f"mem12b not importable: {e}")
        self.mem12b = mem12b

    def test_sum_part_equals_plain_sum_and_chunking_invariant(self):
        t = torch.randn(10_007)
        bits = t.view(torch.int32)
        plain = int(bits.sum(dtype=torch.int64))
        a = self.mem12b.bit_digest(bits, chunk=1000)
        self.assertEqual(a[0], plain)
        self.assertEqual(self.mem12b.bit_digest(bits, chunk=1000), a)
        self.assertEqual(self.mem12b.bit_digest(bits)[0], plain)

    def test_detects_swap_and_opposite_flips(self):
        t = torch.randn(4096)
        d0 = self.mem12b.bit_digest(t.view(torch.int32), chunk=1024)
        s = t.clone()
        s[[3, 2000]] = s[[2000, 3]]                         # a swap: the plain sum is unchanged
        self.assertEqual(int(s.view(torch.int32).sum(dtype=torch.int64)), d0[0])
        self.assertNotEqual(self.mem12b.bit_digest(s.view(torch.int32), chunk=1024), d0)
        u = t.clone().view(torch.int32)
        u[10] += 1
        u[11] -= 1                                          # opposite one-ulp flips: the plain sum is unchanged
        self.assertNotEqual(self.mem12b.bit_digest(u, chunk=1024), d0)

    def test_bf16_view(self):
        t = torch.randn(3000).to(torch.bfloat16)
        d = self.mem12b.bit_digest(t.view(torch.int16), chunk=512)
        self.assertEqual(d[0], int(t.view(torch.int16).sum(dtype=torch.int64)))


if __name__ == "__main__":
    unittest.main()
