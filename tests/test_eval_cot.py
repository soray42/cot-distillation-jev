"""scripts/eval_cot.py: answer parsing, prompt rendering, token counting, the prediction / metrics format, sharding and
merging, with a fake tokenizer and a fake generate (no model, CPU only). Also the rationale cleaning of
scripts/build_gsm8k_rationale.py (no pandas needed)."""
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ec = _load("eval_cot", "scripts/eval_cot.py")
bgr = _load("build_gsm8k_rationale", "scripts/build_gsm8k_rationale.py")
LAB5 = ["A", "B", "C", "D", "E"]


class FakeTok:
    """Character-level tokenizer with a few special tokens, left padding, and a Qwen-like chat template."""
    SPECIAL = {"<pad>": 0, "<|im_end|>": 1, "<|endoftext|>": 2, "<think>": 3, "</think>": 4, "<|im_start|>": 5}

    def __init__(self, opens_think=True):
        self.inv = {v: k for k, v in self.SPECIAL.items()}
        self.pad_token_id, self.eos_token_id, self.unk_token_id = 0, 2, None
        self.padding_side = "right"
        self.opens_think = opens_think
        self.rendered = []

    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for s, v in self.SPECIAL.items():
                if text.startswith(s, i):
                    ids.append(v)
                    i += len(s)
                    break
            else:
                ids.append(ord(text[i]) + 10)
                i += 1
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.inv[i] if i in self.inv else chr(i - 10) for i in ids
                       if not (skip_special_tokens and i in self.inv))

    def convert_tokens_to_ids(self, t):
        return self.SPECIAL.get(t)

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=False):
        enc = [self.encode(t) for t in texts]
        L = max(map(len, enc))
        ids = torch.full((len(enc), L), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(enc), L), dtype=torch.long)
        for r, e in enumerate(enc):
            if self.padding_side == "left":
                ids[r, L - len(e):], mask[r, L - len(e):] = torch.tensor(e), 1
            else:
                ids[r, :len(e)], mask[r, :len(e)] = torch.tensor(e), 1
        return {"input_ids": ids, "attention_mask": mask}

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        c = msgs[0]["content"]
        c = c[0]["text"] if isinstance(c, list) else c
        out = f"<|im_start|>user\n{c}<|im_end|>\n<|im_start|>assistant\n"
        if self.opens_think:          # Qwen3.5: "<think>\n" when thinking, else an empty think block
            out += "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
        return out


class FakeModel(torch.nn.Module):
    """generate() answers each prompt with a scripted response keyed by its question text, ends it with EOS (id 1)
    unless the response is marked no-EOS, cuts at max_new_tokens and pads after the EOS; asserts left padding and
    greedy settings."""

    def __init__(self, tok, responses, no_eos=()):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.tok, self.responses, self.no_eos = tok, responses, set(no_eos)
        self.generation_config = types.SimpleNamespace(eos_token_id=[1, 2])
        self.calls = []

    def generate(self, input_ids, attention_mask, max_new_tokens, do_sample, temperature, top_p, top_k,
                 eos_token_id, pad_token_id):
        assert do_sample is False and temperature is None and top_p is None and top_k is None
        self.calls.append(input_ids.shape[0])
        rows = []
        for ids, m in zip(input_ids.tolist(), attention_mask.tolist()):
            first = m.index(1)
            assert all(m[first:]) and not any(m[:first]), "not left-padded"
            prompt = self.tok.decode(ids[first:])
            key = prompt.split("Question: ", 1)[1].split("\n", 1)[0]
            new = self.tok.encode(self.responses[key]) + ([] if key in self.no_eos else [1])
            rows.append(new[:max_new_tokens])
        L = max(map(len, rows))
        new = torch.tensor([r + [pad_token_id] * (L - len(r)) for r in rows], dtype=torch.long)
        return torch.cat([input_ids, new], 1)


def item(key, gold="B", group="g/1", labels=LAB5):
    opts = "\n".join(f"{L}) {i}" for i, L in enumerate(labels))
    return {"item_id": f"id-{key}", "group": group, "prompt": f"Question: {key}\nOptions:\n{opts}", "labels": labels,
            "gold_label": gold, "gold_probs": None}


def quiet(*a, **k):
    pass


class TestParse(unittest.TestCase):
    def check(self, text, want, how=None, labels=LAB5, thinking=False):
        got, method = ec.parse_answer(text, labels, thinking)
        self.assertEqual(got, want, text)
        if how:
            self.assertEqual(method, how, text)

    def test_explicit_forms(self):
        self.check("Step 1 ...\nAnswer: B", "B", "explicit")
        self.check("so 18.\n\n**Answer:** C", "C", "explicit")
        self.check("**Answer: D**", "D", "explicit")
        self.check("The answer is (E).", "E", "explicit")
        self.check("The final answer is $\\boxed{C}$", "C", "explicit")
        self.check("answer: \\boxed{\\text{A}}", "A", "explicit")
        self.check("\\boxed{D) 18}", "D", "explicit")
        self.check("answer is option B", "B", "explicit")
        self.check("Answer:A", "A", "explicit")

    def test_last_explicit_wins(self):
        self.check("I first thought Answer: A. Rechecking...\nAnswer: C", "C")
        self.check("\\boxed{B}\n\nAnswer: E", "E")

    def test_labels_and_word_boundaries(self):
        self.check("Answer: F", None, "none")                         # F is not among A-E
        self.check("Answer: F", "F", "explicit", labels=LAB5 + ["F"])
        self.check("The answer is 18.", None, "none")
        self.check("Answer: Paul: knight, Mia: knave", None, "none")   # a name, not a letter
        self.check("To answer this, compute 3 x 6 = 18.", None, "none")

    def test_last_line_fallback(self):
        self.check("3 x 6 = 18\nSo the correct option is D.", "D", "last_line")
        self.check("work\nAnswer: 18 (D)", "D", "last_line")
        self.check("work\nA farmer would earn 18 dollars.", None, "none")   # article "A", not an option
        self.check("so it is B\n\nwhich gives 18 in total.", None, "none")  # only the last non-empty line counts

    def test_thinking_region(self):
        self.check("<think>\nmaybe Answer: B\n</think>\n\nAnswer: C", "C", "explicit", thinking=True)
        self.check("hmm, Answer: B and more thinking", None, "no_think_end", thinking=True)   # never left thinking
        self.check("reasoning</think>\n\nAnswer: A", "A", thinking=False)
        self.check("no think tags. Answer: D", "D", thinking=False)


class TestPieces(unittest.TestCase):
    def test_render(self):
        tok = FakeTok()
        off = ec.render(tok, "Question: q", False, False)
        self.assertTrue(off.endswith("<think>\n\n</think>\n\n"))
        self.assertIn("Question: q" + ec.INSTRUCTION, off)
        self.assertTrue(ec.render(tok, "Question: q", True, True).endswith("<think>\n"))
        typed = ec.render(tok, "Question: q", True, False)
        self.assertEqual(typed, off)

        class OpensAlways(FakeTok):          # a template that ignores enable_thinking and opens a think block
            def apply_chat_template(self, msgs, **kw):
                return f"<|im_start|>user\n{msgs[0]['content']}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        self.assertTrue(ec.render(OpensAlways(), "Question: q", False, False).endswith(ec.NO_THINK_CLOSE))

    def test_count_generated(self):
        self.assertEqual(ec.count_generated([20, 21, 1, 0, 0], {1, 2}, 8), (3, False, [20, 21]))
        self.assertEqual(ec.count_generated([20, 2, 1], {1, 2}, 8), (2, False, [20]))
        self.assertEqual(ec.count_generated([20, 21, 22], {1, 2}, 3), (3, True, [20, 21, 22]))

    def test_eos_ids(self):
        tok = FakeTok()
        m = FakeModel(tok, {})
        self.assertEqual(ec.eos_ids(m, tok), [1, 2])
        m.generation_config = types.SimpleNamespace(eos_token_id=None)
        self.assertEqual(ec.eos_ids(m, tok), [1, 2])     # tokenizer eos + <|im_end|>

    def test_p90_and_shard(self):
        self.assertEqual(ec.p90(list(range(1, 11))), 9)
        self.assertEqual(ec.p90([5]), 5)
        self.assertEqual(ec.parse_shard("1/3"), (1, 3))
        self.assertIsNone(ec.parse_shard(""))
        with self.assertRaises(SystemExit):
            ec.parse_shard("3/3")


RESP = {  # key -> response; gold is B for every item below
    "q1": "First 2 + 3 = 5.\nAnswer: B",                  # correct
    "q2": "3 x 2 = 6\nAnswer: C",                         # wrong
    "q3": "I am not sure about this one.",                # unparsable
    "q4": "a very long chain of thought " * 4,          # truncated, no EOS
    "q5": "so the correct option is B.",                  # correct via the last line
}


class TestRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.items = [item(k, group=f"g/{i % 2}") for i, k in enumerate(RESP)]
        self.eval_path = self.dir / "set.jsonl"
        self.eval_path.write_text("".join(json.dumps(it) + "\n" for it in self.items))

    def tearDown(self):
        self.tmp.cleanup()

    def run_eval(self, out, extra=(), max_new=60, bs=2):
        tok = FakeTok()
        model = FakeModel(tok, RESP, no_eos={"q4"})
        args = ec.parse_args(["--eval", f"s={self.eval_path}", "--out", str(out), "--max-new-tokens", str(max_new),
                              "--bs", str(bs), *extra])
        res = ec.run(args, model, tok, tok, False, log=quiet)
        return res, model

    def test_end_to_end(self):
        out = self.dir / "run"
        res, model = self.run_eval(out)
        self.assertEqual(model.calls, [2, 2, 1])                 # batches of bs=2
        preds = [json.loads(l) for l in (out / "preds_s.jsonl").read_text().splitlines()]
        self.assertEqual([p["item_id"] for p in preds], [it["item_id"] for it in self.items])   # input order
        self.assertEqual(list(preds[0])[:6], ["item_id", "group", "labels", "probs", "gold_label", "gold_probs"])
        by = {p["item_id"]: p for p in preds}
        self.assertEqual(by["id-q1"]["probs"], [0.0, 1.0, 0.0, 0.0, 0.0])
        self.assertEqual(by["id-q2"]["pred_label"], "C")
        self.assertEqual(by["id-q3"]["probs"], [0.2] * 5)
        self.assertFalse(by["id-q3"]["parsed"])
        self.assertEqual(by["id-q5"]["parse_method"], "last_line")
        tok = FakeTok()
        self.assertEqual(by["id-q1"]["gen_tokens"], len(tok.encode(RESP["q1"])) + 1)   # + EOS, pads not counted
        self.assertEqual(by["id-q4"]["gen_tokens"], 60)
        self.assertTrue(by["id-q4"]["truncated"])
        self.assertEqual(by["id-q1"]["prompt_tokens"], len(tok.encode(ec.render(tok, self.items[0]["prompt"], False,
                                                                                   False))))
        gens = [json.loads(l) for l in (out / "gens_s.jsonl").read_text().splitlines()]
        self.assertEqual(gens[0]["text"], RESP["q1"])           # EOS cut, no pads
        m = json.loads((out / "metrics.json").read_text())["eval"]["s"]
        self.assertEqual((m["n"], m["n_correct"]), (5, 2))
        self.assertAlmostEqual(m["acc"], 0.4)                    # unparsable q3 / q4 count as wrong
        self.assertAlmostEqual(m["unparsable_rate"], 0.4)
        self.assertAlmostEqual(m["truncated_rate"], 0.2)
        self.assertEqual(m["unparsable_truncated"], 1)
        toks = sorted(p["gen_tokens"] for p in preds)
        self.assertEqual(m["gen_tokens"]["median"], toks[2])
        self.assertEqual(m["gen_tokens"]["p90"], toks[4])
        self.assertEqual(m["gen_tokens"]["total"], sum(toks))
        self.assertEqual(m["parse_methods"], {"explicit": 2, "none": 2, "last_line": 1})
        self.assertEqual(m["groups"]["g/0"]["n"], 3)
        self.assertIn("decode_steps_per_s", m)
        self.assertGreater(m["s_per_item"], 0)

    def test_limit(self):
        out = self.dir / "lim"
        self.run_eval(out, ["--limit", "2"])
        self.assertEqual(len((out / "preds_s.jsonl").read_text().splitlines()), 2)

    def test_shards_merge_to_the_full_run(self):
        full = self.dir / "full"
        self.run_eval(full)
        sh = self.dir / "sh"
        for i in range(2):
            self.run_eval(sh, ["--shard", f"{i}/2"])
        self.assertEqual(len((sh / "preds_s.shard0of2.jsonl").read_text().splitlines()), 3)   # items 0, 2, 4
        self.assertEqual(len((sh / "preds_s.shard1of2.jsonl").read_text().splitlines()), 2)
        ec.merge(ec.parse_args(["--merge", "--shards", "2", "--eval", f"s={self.eval_path}", "--out", str(sh)]),
                 log=quiet)
        self.assertEqual((sh / "preds_s.jsonl").read_text(), (full / "preds_s.jsonl").read_text())
        self.assertEqual((sh / "gens_s.jsonl").read_text(), (full / "gens_s.jsonl").read_text())
        a = json.loads((sh / "metrics.json").read_text())["eval"]["s"]
        b = json.loads((full / "metrics.json").read_text())["eval"]["s"]
        for k in ("n", "acc", "unparsable_rate", "truncated_rate", "gen_tokens", "groups", "parse_methods"):
            self.assertEqual(a[k], b[k], k)

    def test_merge_refuses_missing_shard(self):
        sh = self.dir / "sh1"
        self.run_eval(sh, ["--shard", "0/2"])
        with self.assertRaises(SystemExit):
            ec.merge(ec.parse_args(["--merge", "--shards", "2", "--eval", f"s={self.eval_path}", "--out", str(sh)]),
                     log=quiet)


class TestRationaleCleaning(unittest.TestCase):
    def test_clean_solution(self):
        rat, final = bgr.clean_solution("He spent 3 nickels because 5 - 2 = <<5-2=3>>3\n"
                                        "He got 6 pieces of gum because 3 x 2 = <<3*2=6>>6\n#### 6")
        self.assertEqual(rat, "He spent 3 nickels because 5 - 2 = 3\nHe got 6 pieces of gum because 3 x 2 = 6")
        self.assertEqual(final, "6")
        with self.assertRaises(ValueError):
            bgr.clean_solution("no answer line")
        self.assertTrue(bgr.same_number("4,000", "4000"))
        self.assertFalse(bgr.same_number("40", "4000"))

    def test_fill_changes_only_the_rationale(self):
        row = {"item_id": "gsm8k-train-0", "prompt": "Question: Q?\nOptions:\nA) 6\nB) 7", "labels": ["A", "B"],
               "gold_label": "A", "options": ["6", "7"], "rationale": "", "row_kind": "final", "z": 1}
        new, st = bgr.fill([row], [("Q?", "x = <<2*3=6>>6\n#### 6")])
        self.assertEqual(new[0]["rationale"], "x = 6")
        self.assertEqual(list(new[0]), list(row))
        self.assertEqual({k: v for k, v in new[0].items() if k != "rationale"},
                         {k: v for k, v in row.items() if k != "rationale"})
        self.assertEqual(st["with_rationale"], 1)
        with self.assertRaises(ValueError):                      # the GSM8K answer must be the gold option
            bgr.fill([dict(row, gold_label="B")], [("Q?", "x = 6\n#### 6")])


if __name__ == "__main__":
    unittest.main()
