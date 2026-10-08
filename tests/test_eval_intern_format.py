"""scripts/eval_intern_format.py: request rendering as Intern-Decision's compile_row, MC conversion, option permutation,
marker positions (added-token and split paths). CPU only; the tokenizer tests skip without local tokenizer files."""
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eif", ROOT / "scripts/eval_intern_format.py")
eif = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eif)
TOKDIR = ROOT / "results/weights/T1-Qwen3.5-2B-Base-student_v3-s2/model"


class TestRender(unittest.TestCase):
    def test_user_text_and_skeleton(self):
        q = {"type": "choice", "instructions": "Which box is red?", "criteria": {"left": "The left box", "right": "R"}}
        n = {"type": "noul", "instructions": "Is one red?"}
        s = {"type": "score", "instructions": "How many?", "criteria": ["Zero", "One"]}
        msg = eif.messages({"a": 1}, [("box", q), ("red", n), ("cnt", s)])
        self.assertEqual(msg[0]["content"], eif.SYSTEM_PROMPT)
        self.assertEqual(msg[1]["content"],
                         "Return one answer for every field using the supplied answer symbols.\n\n## State\n"
                         '{\n  "a": 1\n}\n## Decision schema\n'
                         "box: Which box is red?\n    A = left: The left box\n    B = right: R\n"
                         f"red: Is one red?\n    A = no: {eif.NOUL_NO}\n    B = yes: {eif.NOUL_YES}\n"
                         "cnt: How many?\n    A = 0: Zero\n    B = 1: One")
        self.assertEqual(json.loads(msg[2]["content"]), {"box": "<decision>", "red": "<decision>", "cnt": "<decision>"})

    def test_from_mc_and_permute(self):
        it = {"item_id": "x", "prompt": "Some state.\n\nQuestion: Pick one?\nOptions:\nA) red\nB) blue\nC) red",
              "labels": ["A", "B", "C"], "gold_label": "C"}
        r = eif.from_mc(it)
        self.assertEqual(r["state"], "Some state.")
        q = r["questions"]["decision"]
        self.assertEqual(list(q["criteria"]), ["red", "blue", "red (C)"])
        self.assertEqual(r["targets"]["decision"]["label"], "red (C)")
        self.assertEqual(list(eif.permute(q, "reverse", "k")["criteria"]), ["red (C)", "blue", "red"])
        sh = eif.permute(q, "shuffle", "k")
        self.assertEqual(sorted(sh["criteria"]), sorted(q["criteria"]))
        self.assertEqual(sh, eif.permute(q, "shuffle", "k"))
        n = {"type": "noul", "instructions": "?"}
        self.assertIs(eif.permute(n, "reverse", "k"), n)

    def test_jevraw_rows(self):
        p = ROOT / "data/eval/intern_bench/jevbench/original.jsonl"
        if not p.exists():
            self.skipTest("no local JevBench copy")
        for r in eif.load_rows(str(p), "jevraw"):
            q = r["questions"]["decision"]
            self.assertIn(eif.norm_label(q, r["targets"]["decision"]["label"]), [v for v, _ in eif.options(q)])


@unittest.skipUnless((TOKDIR / "tokenizer.json").exists(), "no local tokenizer")
class TestMarkers(unittest.TestCase):
    def setUp(self):
        import transformers
        self.tok = transformers.AutoTokenizer.from_pretrained(str(TOKDIR))

    def test_split_path(self):
        text = 'x {\n    "a": "<decision>",\n    "b": "<decision>"\n}'
        ids, pos = eif.marker_positions(self.tok, text)
        self.assertEqual(len(pos), 2)
        self.assertEqual(self.tok.decode(ids[:pos[0] + 1]), 'x {\n    "a": "')
        self.assertEqual(self.tok.decode(ids[:pos[1] + 1]), 'x {\n    "a": "",\n    "b": "')

    def test_added_token_path(self):
        self.tok.add_tokens([eif.DECISION_TOKEN], special_tokens=True)
        text = 'x {"a": "<decision>", "b": "<decision>"}'
        ids, pos = eif.marker_positions(self.tok, text)
        mid = self.tok.convert_tokens_to_ids(eif.DECISION_TOKEN)
        self.assertEqual([ids[p + 1] for p in pos], [mid, mid])

    def test_symbols_single_tokens(self):
        sym = [self.tok.convert_tokens_to_ids(s) for s in eif.ANSWER_SYMBOLS]
        self.assertEqual(len(set(sym)), 62)
        self.assertNotIn(self.tok.unk_token_id, sym)


if __name__ == "__main__":
    unittest.main()
