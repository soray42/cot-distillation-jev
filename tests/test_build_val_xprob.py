"""scripts/build_val_xprob.py: the cue-free validation set (data/eval/gsm8k_val_xprob.jsonl) is reproducible, keeps
every question and gold, draws its distractors from other problems' answers, and loads as an --eval / --eval-mid set
of train_student.py (A2 and TSF, whose use detector reads its worked steps)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import build_val_xprob as bvx  # noqa: E402
from build_mc_xprob import LETTERS, md5, read, split_prompt  # noqa: E402


class TestBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.val, cls.pool = read(bvx.VAL), read(bvx.POOL)
        cls.out, cls.st = bvx.build(cls.val, cls.pool)

    def test_file_is_this_build(self):
        summ = json.loads(bvx.OUT.with_suffix(".summary.json").read_text())
        self.assertEqual(summ["md5"]["data/eval/gsm8k_val_xprob.jsonl"], md5(bvx.OUT))
        self.assertEqual(read(bvx.OUT), json.loads(json.dumps(self.out)))

    def test_questions_golds_and_sources(self):
        golds = {it["item_id"]: split_prompt(it["prompt"])[1][it["gold_label"]] for it in self.val + self.pool}
        self.assertEqual(len(self.out), 393)
        for a, b in zip(self.val, self.out):
            q0, o0 = split_prompt(a["prompt"])
            q1, o1 = split_prompt(b["prompt"])
            self.assertEqual(q0, q1)
            self.assertEqual(o1[b["gold_label"]], o0[a["gold_label"]])
            self.assertEqual(b["options"], [o1[L] for L in LETTERS])
            self.assertEqual(b["teacher"][b["gold_label"]], 1.0)
            self.assertEqual(sorted(b["xprob"]["src"]), sorted(L for L in LETTERS if L != b["gold_label"]))
            for L, src in b["xprob"]["src"].items():
                self.assertNotEqual(src, b["item_id"])
                self.assertEqual(golds[src], o1[L])
            self.assertEqual(b["worked_steps"], a["worked_steps"])
            self.assertFalse(set(bvx.DROP) & set(b))

    def test_window_like_the_test_build(self):
        self.assertGreaterEqual(self.st["window_ratio"]["2"], 0.95 * 393)
        self.assertGreaterEqual(min(self.st["gold_rank"].values()), 60)


try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestLoads(unittest.TestCase):
    def run_arm(self, extra: list[str]) -> dict:
        from test_tsf import common4, tiny4_dir
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            tiny4_dir(d)
            Path(f"{d}/t.jsonl").write_text("".join(json.dumps(r) + "\n" for r in read(bvx.VAL)[:4]))
            Path(f"{d}/vx.jsonl").write_text("".join(json.dumps(r) + "\n" for r in read(bvx.OUT)[:4]))
            cmd = common4(d) + ["--eval", f"v={d}/t.jsonl", f"valx={d}/vx.jsonl", "--eval-mid", "v,valx",
                                "--eval-at", "0.5", "--epochs", "1", "--out", f"{d}/o"] + extra
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            self.assertEqual(len(read(Path(f"{d}/o/preds_valx.jsonl"))), 4)
            return json.loads(Path(f"{d}/o/metrics.json").read_text())

    def test_a2(self):
        m = self.run_arm([])
        self.assertTrue(any("valx" in t for t in m["trajectory"]))

    def test_tsf_detector(self):
        from test_tsf import GFPQ, TSF_ARMS
        m = self.run_arm(GFPQ + TSF_ARMS["TSF"] + ["--teacher-grad-probe", "0,1"])
        self.assertTrue(any("valx@force" in t for t in m["trajectory"]), [sorted(t) for t in m["trajectory"]])


if __name__ == "__main__":
    unittest.main()
