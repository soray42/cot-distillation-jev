"""--chat-template (train_student.py / eval_student.py): opt-in chat-turn rendering of every view, thinking off, read out
at the assistant turn's first token. Default path unchanged (encodes, micro-batch lengths, letter ids); with the
renderer set, a final question renders exactly as scripts/eval_llm_baseline.py renders the prompt. CPU only, fake
tokenizer and processor; COTD_CHAT_TOKENIZER=<dir with a chat template> adds a check on a real tokenizer."""
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False

TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


class FakeTok:
    """Word-piece-like toy tokenizer: ' A' and 'A' are single (different) tokens."""
    bos_token = None
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=True):
        return [self.vocab.setdefault(t, len(self.vocab) + 1) for t in re.findall(r" ?[^\s]+|\s", text)]


class FakeProc:
    def __init__(self, think_open=False, bos=""):
        self.think_open, self.bos = think_open, bos

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        c = msgs[0]["content"]
        c = c[0]["text"] if isinstance(c, list) else c
        out = f"{self.bos}<|im_start|>user\n{c}<|im_end|>\n<|im_start|>assistant\n"
        return out + ("<think>\n" if self.think_open or enable_thinking else "<think>\n\n</think>\n\n")


PROMPT = "Question: Is the road wet?\nOptions:\nA) Yes\nB) No\nC) Unknown"


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestChatTemplate(unittest.TestCase):
    def setUp(self):
        from cotdistill import student as S
        self.S = S

    def test_default_path_unchanged(self):
        S, tok = self.S, FakeTok()
        text = S.FINAL_TEMPLATE.format(problem=PROMPT)
        ids = tok.encode(text, add_special_tokens=False)
        self.assertEqual(S.text_ids(tok, text), ids)
        self.assertEqual(S.encode_tail(tok, text, 5), ids[-5:])
        self.assertEqual(S.letter_token_ids(tok, ["A", "B"]), [tok.encode(" A")[0], tok.encode(" B")[0]])
        ex = [S.Example(text, ["A", "B", "C"], [1, 0, 0], 1.0, "final", "x"),
              S.Example(text + " more words here", ["A", "B", "C"], [1, 0, 0], 1.0, "final", "y")]
        mb = S.micro_batches(tok, ex, 1536, 2, token_budget=40)
        lens = [len(tok.encode(e.text, add_special_tokens=False)) for e in ex]
        self.assertEqual([[e.item_id for e in b] for b in mb],
                         [[ex[j].item_id for j in b] for b in S.token_budget_batches(lens, 40, 16)])

    def test_final_question_matches_llm_baseline(self):
        import eval_llm_baseline as base
        S, tok, proc = self.S, FakeTok(), FakeProc()
        S.set_chat_render(tok, S.chat_render(proc))
        text = S.FINAL_TEMPLATE.format(problem=PROMPT)
        want = base.render(proc, PROMPT, False)
        self.assertEqual(S._chat(tok)(text), want)
        self.assertEqual(S.CHAT_INSTRUCTION, base.INSTRUCTION)
        self.assertTrue(want.endswith(TAIL))
        self.assertEqual(S.text_ids(tok, text), tok.encode(want, add_special_tokens=False))
        self.assertEqual(S.encode_tail(tok, text, 4), tok.encode(want, add_special_tokens=False)[-4:])
        self.assertEqual(S.letter_token_ids(tok, ["A", "B"]), [tok.encode("A")[0], tok.encode("B")[0]])
        other = FakeTok()                                   # keyed by the tokenizer object: others stay default
        self.assertEqual(S.text_ids(other, text), other.encode(text))
        self.assertEqual(S.letter_token_ids(other, ["A"]), [other.encode(" A")[0]])

    def test_subq_view_and_lengths(self):
        S, tok = self.S, FakeTok()
        S.set_chat_render(tok, S.chat_render(FakeProc(), typed=True))
        sub = S.SUBQ_TEMPLATE.format(problem="P", question="Is it?", a="Yes", b="No")
        r = S._chat(tok)(sub)
        self.assertEqual(r, f"<|im_start|>user\nP\n\nIntermediate question: Is it?\nOptions:\nA) Yes\nB) No"
                            f"{S.CHAT_INSTRUCTION}<|im_end|>\n{TAIL}")
        self.assertNotIn("Answer:", r)
        ex = [S.Example(sub, ["A", "B"], [1, 0], 1.0, "subq", "x"), S.Example(sub + "?", ["A", "B"], [1, 0], 1.0, "subq", "y")]
        lens = [len(tok.encode(S._chat(tok)(e.text))) for e in ex]
        mb = S.micro_batches(tok, ex, 1536, 2, token_budget=lens[0] + 1)
        self.assertEqual([[e.item_id for e in b] for b in mb],
                         [[ex[j].item_id for j in b] for b in S.token_budget_batches(lens, lens[0] + 1, 16)])

    def test_think_block_closed_and_bos_cut(self):
        S = self.S
        r = S.chat_render(FakeProc(think_open=True, bos="<bos>"), bos="<bos>")("X\n\nAnswer:")
        self.assertTrue(r.startswith("<|im_start|>user\nX"))
        self.assertTrue(r.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

    def test_rationale_lm_refused(self):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
        p = subprocess.run([sys.executable, str(ROOT / "scripts/train_student.py"), "--model", "none", "--train", "none",
                            "--out", str(ROOT / "runs/never"), "--chat-template", "--rationale-lm"],
                           capture_output=True, text=True, env=env, timeout=300)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("--chat-template does not combine with --rationale-lm", p.stderr)
        self.assertFalse((ROOT / "runs/never").exists())

    def test_resume_default(self):
        src = (ROOT / "scripts/train_student.py").read_text()
        self.assertRegex(src, r'NEW_ARG_DEFAULTS = \{[^}]*"chat_template": False')

    @unittest.skipUnless(os.environ.get("COTD_CHAT_TOKENIZER"), "COTD_CHAT_TOKENIZER not set")
    def test_real_tokenizer(self):
        import transformers as tf
        S = self.S
        tok = tf.AutoTokenizer.from_pretrained(os.environ["COTD_CHAT_TOKENIZER"])
        sample = S.use_chat_template(tok, os.environ["COTD_CHAT_TOKENIZER"])
        self.assertIn("Answer with the letter of the correct option only.", sample)
        self.assertTrue(sample.rstrip("\n").endswith("</think>") or sample.endswith("assistant\n"), repr(sample[-60:]))
        ids = S.letter_token_ids(tok, ["A", "B", "C"])
        self.assertEqual(ids, [tok.encode(x, add_special_tokens=False)[0] for x in "ABC"])


if __name__ == "__main__":
    unittest.main()
