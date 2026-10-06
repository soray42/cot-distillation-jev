"""CPU tests for scripts/mem12b.py (the 12B memory/speed/resume smoke): text-only load, soft-capped readout,
synthetic batches, the offloaded update loop and the bit-exact resume round trip, on a tiny random Gemma 4 Unified
model (vocabulary 300, no downloads, no weights). Skipped when transformers has no Gemma4Unified classes."""
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

try:
    import torch
    import transformers
    HAVE = hasattr(transformers, "Gemma4UnifiedForConditionalGeneration")
except ImportError:
    HAVE = False

if HAVE:
    import mem12b
    from cotdistill.offload import CPUOffloadAdamW


class LetterTok:
    """' A'..' Z' -> 200 + index; other characters -> ord % 150 + 5; add_special_tokens prepends <bos> = 2."""
    pad_token_id, eos_token_id = 0, 1

    def encode(self, text, add_special_tokens=True):
        if len(text) == 2 and text[0] == " " and text[1].isupper():
            return [200 + ord(text[1]) - 65]
        return ([2] if add_special_tokens else []) + [ord(c) % 150 + 5 for c in text]


def tiny_full():
    tc = dict(vocab_size=300, hidden_size=32, intermediate_size=64, num_attention_heads=2, num_key_value_heads=1,
              head_dim=16, global_head_dim=32, num_global_key_value_heads=1, num_hidden_layers=2,
              layer_types=["sliding_attention", "full_attention"], sliding_window=4, attention_k_eq_v=True,
              final_logit_softcapping=30.0, tie_word_embeddings=True)
    vc = dict(mm_embed_dim=32, output_proj_dims=32, mm_posemb_size=8, num_soft_tokens=4)
    ac = dict(audio_embed_dim=16, hidden_size=16, output_proj_dims=16)
    cfg = transformers.Gemma4UnifiedConfig(text_config=tc, vision_config=vc, audio_config=ac)
    torch.manual_seed(0)
    return transformers.Gemma4UnifiedForConditionalGeneration(cfg)


ITEMS = [{"item_id": "a", "prompt": "Is it so?\nOptions:\nA) yes\nB) no", "labels": ["A", "B"], "gold_label": "B"},
         {"item_id": "b", "prompt": "Pick one.\nA) x\nB) y\nC) z", "labels": ["A", "B", "C"], "gold_label": "A"},
         {"item_id": "c", "prompt": "Q", "labels": ["A", "B", "C", "D"], "gold_label": "D"}]


class Args:
    ga, steps, baseline_micro, inflight_gib, max_len = 2, 3, 1, 2.0, 64

    def __init__(self, rt_dir="."):
        self.rt_dir = rt_dir


@unittest.skipUnless(HAVE, "needs transformers with Gemma4Unified")
class TestMem12b(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.full = tiny_full().eval()
        cls.full.save_pretrained(Path(cls.tmp.name) / "ckpt")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def load(self, dtype=torch.float32):
        m, info = mem12b.load_text_only(str(Path(self.tmp.name) / "ckpt"), dtype)
        return m

    def capture(self):
        rows = []
        orig = mem12b.emit
        mem12b.emit = rows.append
        self.addCleanup(setattr, mem12b, "emit", orig)
        return rows

    def test_text_only_load_and_parts(self):
        m = self.load()
        self.assertIsNone(m.model.embed_vision)
        self.assertEqual(mem12b.softcap_of(m), 30.0)
        body, head = mem12b.text_parts(m)
        self.assertIs(head.weight, m.model.language_model.embed_tokens.weight)     # tied
        self.assertEqual(mem12b.special_prefix(LetterTok()), [2])

    def test_readout_equals_model_logits(self):
        m = self.load().eval()
        r = mem12b.readout_items(m, LetterTok(), ITEMS, 64, hf_check=True)
        self.assertLess(r["hf_check"]["letters_maxabs"], 1e-4)
        self.assertLess(r["hf_check"]["lse_absdiff"], 1e-4)
        self.assertTrue(0.0 < r["letter_mass_min"] <= r["letter_mass"] <= 1.0)
        self.assertAlmostEqual(r["chance"], (1 / 2 + 1 / 3 + 1 / 4) / 3, places=4)
        # the same mass computed directly from the HF forward (soft-capped full logits)
        tok = LetterTok()
        with torch.no_grad():
            masses = []
            for it in ITEMS:
                ids = torch.tensor([mem12b.encode_tail(tok, it["prompt"] + "\n\nAnswer:", 64, [2])])
                z = m(input_ids=ids).logits[0, -1].float()
                lid = [200 + ord(L) - 65 for L in it["labels"]]
                masses.append(float(torch.softmax(z, 0)[lid].sum()))
        self.assertAlmostEqual(r["letter_mass"], sum(masses) / 3, places=4)
        nob = mem12b.readout_items(m, tok, ITEMS, 64, use_prefix=False)
        self.assertNotEqual(nob["p_gold"], r["p_gold"])                              # BOS changes the input

    def test_part_readout_rows(self):
        m = self.load().train()
        rows = self.capture()
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "e.jsonl"
            f.write_text("".join(json.dumps(dict(it, group="g")) + "\n" for it in ITEMS * 3))
            mem12b.part_readout(m, LetterTok(), [str(f), str(Path(d) / "missing.jsonl")], 4, 64)
        self.assertEqual(rows[0]["bos"]["n"], 4)
        self.assertEqual(rows[0]["groups"], {"g": 4})
        self.assertEqual(rows[1]["skip"], "missing")
        self.assertTrue(m.training)                                                   # mode restored

    def test_make_batch(self):
        g = torch.Generator().manual_seed(0)
        b = mem12b.make_batch(3, 20, 300, [2], 6, g, torch.device("cpu"))
        self.assertEqual(b["input_ids"].shape, (3, 20))
        self.assertTrue((b["input_ids"][:, 0] == 2).all())
        self.assertEqual(b["last"].tolist(), [19, 19, 12])
        self.assertEqual(b["attention_mask"][-1].sum().item(), 13)
        self.assertEqual(b["tokens"], 53)
        self.assertTrue(((b["input_ids"][:2] >= 3) & (b["input_ids"][:2] < 300))[:, 1:].all())
        self.assertEqual(mem12b.parse_setting("4x1024:6"), (4, 1024, 6.0))
        self.assertEqual(mem12b.parse_setting("2x1536"), (2, 1536, None))

    def test_letter_loss_gradient_matches_plain_adamw_update(self):
        """The offloaded update loop on CPU equals a plain fp32 AdamW loop (same loss, clip, GA)."""
        m1, m2 = self.load(), self.load()
        for m in (m1, m2):
            m.train()
        lids = torch.tensor([200 + i for i in range(6)])
        mk = lambda: [mem12b.make_batch(2, 16, 300, [2], 6, torch.Generator().manual_seed(k), torch.device("cpu"))
                      for k in range(4)]
        p1 = [p for p in m1.parameters() if p.requires_grad]
        opt = CPUOffloadAdamW(p1, lr=1e-3, weight_decay=0.0, num_threads=1)
        bs = mk()
        for step in range(2):
            mem12b.run_update(m1, opt, None, lambda k: dict(bs[2 * step + k], lids=lids), 2, torch.device("cpu"))
        p2 = [p for p in m2.parameters() if p.requires_grad]
        ref = torch.optim.AdamW(p2, lr=1e-3, weight_decay=0.0, foreach=False)
        for step in range(2):
            for k in range(2):
                b = bs[2 * step + k]
                (mem12b.letter_loss(m2, b, lids) / (2 * 2)).backward()
            torch.nn.utils.clip_grad_norm_(p2, 1.0)
            ref.step()
            ref.zero_grad()
        for a, b in zip(p1, p2):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
        opt.release()

    def test_mem_and_roundtrip(self):
        m = self.load(torch.bfloat16)
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.train()
        dev = torch.device("cpu")
        lids = torch.tensor([200 + i for i in range(6)])
        params = [p for p in m.parameters() if p.requires_grad]
        rows = self.capture()
        base = mem12b.part_baseline(m, ["2x24"], Args(), lids, [2], 300, dev)
        self.assertEqual(rows[-1]["status"], "ok")
        self.assertTrue(all(p.grad is None for p in params))
        opt = CPUOffloadAdamW(params, lr=1e-3, weight_decay=0.0, num_threads=1)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 / (1 + s))
        w0 = [p.detach().clone() for p in params]
        mem12b.part_mem(m, opt, sched, ["2x24", "3x16:0.001"], Args(), lids, [2], 300, dev, base, float("inf"))
        r = rows[-2]
        self.assertEqual((r["status"], r["updates"], r["tokens_per_update"]), ("ok", 3, 2 * (24 + 17)))
        for k in ("s_per_update", "fb_s", "clip_s", "step_s", "tok_s", "fb_over_gpu_only", "host_rss_peak_gib"):
            self.assertIn(k, r)
        self.assertAlmostEqual(rows[-1]["inflight_gib"], 0.001, places=6)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(w0, params)))
        with tempfile.TemporaryDirectory() as d:
            mem12b.part_roundtrip(m, opt, sched, Args(rt_dir=d), lids, [2], 300, dev)
            self.assertEqual(list(Path(d).iterdir()), [])                              # state file removed
        rt = rows[-1]
        self.assertTrue(rt["ok"], rt)
        self.assertGreater(rt["moved"], 0)
        self.assertEqual(rt["n_params_weights_differ"], 0)
        opt.release()

    def test_roundtrip_detects_a_bad_load(self):
        """The digests are sensitive: perturbing one master element after the load breaks load_exact."""
        m = self.load(torch.bfloat16)
        params = [p for p in m.parameters() if p.requires_grad]
        opt = CPUOffloadAdamW(params, lr=1e-3, num_threads=1)
        mem12b.fixed_grad_update(opt, None, seed=1)
        d0 = mem12b.digest_cpu(opt.state_dict())
        opt.master[3].view(-1)[0] += 1e-7
        self.assertNotEqual(mem12b.digest_cpu(opt.state_dict()), d0)
        g0 = mem12b.digest_gpu(params)
        with torch.no_grad():
            params[5].view(-1)[0] += 1.0
        self.assertNotEqual(mem12b.digest_gpu(params), g0)
        self.assertFalse(mem12b.weights_match_masters(opt))
        opt.release()

    @unittest.skipUnless((ROOT / "notes/next_round/scale12b/hf/base_config.json").exists(), "no saved 12B config")
    def test_tiny_from_real_config(self):
        """--tiny: the real 12B config shrunk to 2 layers keeps the class, soft-cap, K = V full layer and vocabulary."""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text((ROOT / "notes/next_round/scale12b/hf/base_config.json").read_text())
            cfg = mem12b.tiny_config(d)
        self.assertIsNone(cfg.vision_config)
        m = mem12b.build_tiny(cfg, torch.float32)
        self.assertEqual(type(m).__name__, "Gemma4UnifiedForConditionalGeneration")
        self.assertEqual(mem12b.softcap_of(m), 30.0)
        self.assertEqual(m.get_output_embeddings().weight.shape, (262144, 64))
        self.assertLess(sum(p.numel() for p in m.parameters()), 17_000_000)
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.train()
        b = mem12b.make_batch(2, 80, 262144, [2], 6, torch.Generator().manual_seed(0), torch.device("cpu"))
        lids = torch.tensor([562, 603, 565, 622, 645, 633])
        mem12b.letter_loss(m, b, lids).backward()
        self.assertTrue(all(p.grad is not None for p in m.parameters() if p.requires_grad))

    def test_fixed_grad_is_exact(self):
        p = torch.nn.Parameter(torch.randn(5, 3).to(torch.bfloat16))
        g = torch.Generator().manual_seed(4)
        c = (torch.randn(p.shape, generator=g) * 1e-3).to(p.dtype)
        (p * c).sum().backward()
        self.assertTrue(torch.equal(p.grad, c))


class TestCost(unittest.TestCase):
    def setUp(self):
        import cost12b
        self.c = cost12b

    def test_floor_and_step(self):
        self.assertAlmostEqual(self.c.offload_floor(16, 60), 11.90735 * 10 / 60, places=3)   # CPU add binds
        self.assertAlmostEqual(self.c.offload_floor(10, 1000), 11.90735 * 2 / 10, places=3)  # PCIe binds
        self.assertAlmostEqual(self.c.step_s(60), 11.90735 * 46 / 60, places=3)

    @unittest.skipUnless((ROOT / "notes/next_round/tree_forcing/cost.json").exists()
                         and (ROOT / "data/student_v3f/train.jsonl").exists(), "cost.json / v3f not in this checkout")
    def test_hours_reproduce_g3_rate(self):
        """At the 2B-calibrated 12B rate (2B G3-c4 all-in rate / 7.94) and no offload cost, G3-c4's training hours
        are the 2B training hours x 7.94."""
        g3 = [a for a in self.c.arms() if a["arm"] == "G3-c4"][0]
        rate_2b = g3["pad"] * 4 / (2.3 * 3600)                     # 2.6 h - 0.3 h evals
        h = self.c.hours(g3, rate_2b / 7.94, 0.0, pcie=1e9, cpu=1e9, eval_h=0.0)
        self.assertAlmostEqual(h["gpu_only_h"], 2.3 * 7.94, places=2)

    def test_padding_simulation(self):
        pad, real = self.c.padded_random([100] * 64 + [300] * 64, 2, 32)
        self.assertGreaterEqual(pad, real)
        pad1, real1 = self.c.padded_random([100] * 64, 8, 32)
        self.assertEqual(pad1, real1)

    def test_missing_inputs_skip_cleanly(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.c.main(["--cost-json", "/nonexistent/cost.json"])
        self.assertIn("cost table skipped", buf.getvalue())

    def test_from_log(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "log"
            rows = [{"part": "bench", "d2h_gbs": 14.0}, {"part": "baseline", "status": "ok", "setting": "2x1536",
                     "tok_s": 700.0}, {"part": "baseline", "status": "ok", "setting": "4x1024", "tok_s": 750.0},
                    {"part": "mem", "step_s": 8.0, "clip_s": 1.0}]
            f.write_text("noise\n" + "".join("MEM12B " + json.dumps(r) + "\n" for r in rows))
            m = self.c.from_log(str(f))
        self.assertEqual((m["rate"], m["rate_setting"], m["pcie"], m["step_s"]), (750.0, "4x1024", 14.0, 9.0))
        self.assertAlmostEqual(m["cpu_gbs"], 11.90735 * 46 / 9, places=0)


if __name__ == "__main__":
    unittest.main()
