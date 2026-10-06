"""CPU tests for the Gemma 4 readout changes: BOS prefix, logit soft-cap, text-only gemma4_unified loading.

Tiny random models only (no weights, no downloads); the gemma4_unified tests are skipped when the installed
transformers has no Gemma4Unified classes (needs >= 5.10).
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
    import transformers
    HAVE = True
except ImportError:
    HAVE = False
HAVE_G4U = HAVE and hasattr(transformers, "Gemma4UnifiedForConditionalGeneration")

if HAVE:
    from cotdistill.student import (Example, collate, encode_tail, head_logits, label_logits, load_model,
                                    logit_softcap, micro_batches, special_prefix)


class FakeTok:
    """Character tokenizer: id = ord(c); with bos set, add_special_tokens=True prepends it (as Gemma's does)."""

    def __init__(self, bos=None, eos_suffix=None):
        self.bos, self.eos_suffix = bos, eos_suffix
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text, add_special_tokens=True):
        ids = [ord(c) for c in text]
        if add_special_tokens:
            ids = ([self.bos] if self.bos is not None else []) + ids + ([self.eos_suffix] if self.eos_suffix else [])
        return ids


@unittest.skipUnless(HAVE, "needs torch")
class TestPrefix(unittest.TestCase):
    def test_prefix_detection(self):
        self.assertEqual(special_prefix(FakeTok(bos=2)), [2])
        self.assertEqual(special_prefix(FakeTok()), [])
        self.assertEqual(special_prefix(FakeTok(eos_suffix=1)), [])      # a suffix is not a prefix

    def test_truncation_keeps_bos_and_tail(self):
        t = FakeTok(bos=2)
        self.assertEqual(encode_tail(t, "abcdef", 4), [2, ord("d"), ord("e"), ord("f")])
        self.assertEqual(encode_tail(t, "ab", 10), [2, ord("a"), ord("b")])
        self.assertEqual(encode_tail(FakeTok(), "abcdef", 4), [ord(c) for c in "cdef"])

    def test_collate_and_micro_batches(self):
        ex = [Example("abcdef", ["A", "B"], [0.5, 0.5], 1.0, "final", "0"),
              Example("xy", ["A", "B"], [0.5, 0.5], 1.0, "final", "1")]
        b = collate(FakeTok(bos=2), ex, 4, "cpu")
        self.assertEqual(b["input_ids"][:, 0].tolist(), [2, 2])
        self.assertEqual(b["last"].tolist(), [3, 2])
        b0 = collate(FakeTok(), ex, 4, "cpu")                            # no BOS: as before the change
        self.assertEqual(b0["input_ids"][0].tolist(), [ord(c) for c in "cdef"])
        mbs = micro_batches(FakeTok(bos=2), ex, 4, 1, token_budget=7)    # lengths 4 (cut) and 3 (BOS + xy)
        self.assertEqual([[e.item_id for e in mb] for mb in mbs], [["0"], ["1"]])
        mbs = micro_batches(FakeTok(bos=2), ex, 4, 1, token_budget=8)
        self.assertEqual([[e.item_id for e in mb] for mb in mbs], [["0", "1"]])


@unittest.skipUnless(HAVE, "needs torch")
class TestHeadLogits(unittest.TestCase):
    def test_no_cap_is_previous_readout(self):
        torch.manual_seed(0)
        h, w = torch.randn(8, dtype=torch.bfloat16), torch.randn(3, 8, dtype=torch.bfloat16)
        self.assertTrue(torch.equal(head_logits(h, w), h.to(w.dtype) @ w.T))

    def test_cap(self):
        h, w = torch.tensor([10.0, 0.0]), torch.tensor([[4.5, 0.0], [3.8, 0.0], [2.0, 0.0]])
        z = head_logits(h, w, softcap=30.0)
        torch.testing.assert_close(z, 30.0 * torch.tanh(torch.tensor([45.0, 38.0, 20.0]) / 30.0))
        self.assertLess(float(torch.softmax(z, -1)[0]), 0.83)            # uncapped: 0.999
        self.assertEqual(z.dtype, torch.float32)


def _tiny_g4u():
    tc = dict(vocab_size=300, hidden_size=32, intermediate_size=64, num_attention_heads=2, num_key_value_heads=1,
              head_dim=16, global_head_dim=32, num_global_key_value_heads=1, num_hidden_layers=2,
              layer_types=["sliding_attention", "full_attention"], sliding_window=4, attention_k_eq_v=True,
              final_logit_softcapping=30.0, tie_word_embeddings=True)
    vc = dict(mm_embed_dim=32, output_proj_dims=32, mm_posemb_size=8, num_soft_tokens=4)
    ac = dict(audio_embed_dim=16, hidden_size=16, output_proj_dims=16)
    cfg = transformers.Gemma4UnifiedConfig(text_config=tc, vision_config=vc, audio_config=ac)
    torch.manual_seed(0)
    return transformers.Gemma4UnifiedForConditionalGeneration(cfg).eval()


class LetterTok(FakeTok):
    """Ids stay below the tiny vocabulary: ' A' -> 200 + letter index, other characters -> ord % 150 + 5."""

    def encode(self, text, add_special_tokens=True):
        if len(text) == 2 and text[0] == " " and text[1].isupper():
            return [200 + ord(text[1]) - 65]
        ids = [ord(c) % 150 + 5 for c in text]
        return ([self.bos] if add_special_tokens else []) + ids


@unittest.skipUnless(HAVE_G4U, "needs transformers with Gemma4Unified")
class TestGemma4Unified(unittest.TestCase):
    def test_text_only_load_readout_and_save(self):
        full = _tiny_g4u()
        tok = LetterTok(bos=2)
        with tempfile.TemporaryDirectory() as d:
            full.save_pretrained(Path(d) / "ckpt")
            m = load_model(str(Path(d) / "ckpt"), torch.float32).eval()
            self.assertIsNone(m.model.embed_vision)
            self.assertIsNone(m.model.embed_audio)
            self.assertEqual(logit_softcap(m), 30.0)
            ex = [Example("Is it so?\n\nAnswer:", ["A", "B"], [0.5, 0.5], 1.0, "final", "0"),
                  Example("Q\n\nAnswer:", ["A", "B", "C"], [0.2, 0.3, 0.5], 1.0, "final", "1")]
            with torch.no_grad():
                zs = label_logits(m, tok, ex, 64, {})
                for z, e in zip(zs, ex):
                    ids = torch.tensor([encode_tail(tok, e.text, 64)])
                    self.assertEqual(int(ids[0, 0]), 2)
                    ref = full(input_ids=ids).logits[0, -1, [200 + ord(L) - 65 for L in e.labels]]
                    torch.testing.assert_close(z, ref, atol=1e-5, rtol=1e-5)
            m.save_pretrained(Path(d) / "saved")
            r = load_model(str(Path(d) / "saved"), torch.float32).eval()
            ids = torch.tensor([[2, 10, 11, 12, 13, 14, 15]])
            with torch.no_grad():
                torch.testing.assert_close(r(input_ids=ids).logits, full(input_ids=ids).logits)


if __name__ == "__main__":
    unittest.main()
