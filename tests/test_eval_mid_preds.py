"""Opt-in evaluation outputs for the StepFinder protocol (notes/next_round/failure_attribution/sf_protocol.md):
train_student.py --eval-mid-preds (the predictions of every --eval-at point, for checkpoint selection on a val set)
and eval_student.py --time (forward-only seconds per item, batch 1). Without the flags nothing new is written and the
outputs are unchanged. CPU only, tiny random Qwen2 with the GPT-2 tokenizer."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    from test_mc_nodes import toy_fa
    from test_tree_forcing import ENV, SCRIPT, tiny
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestMidPredsAndTiming(unittest.TestCase):
    def test_eval_mid_preds(self):
        with tempfile.TemporaryDirectory() as d:
            tiny(d)
            items = [dict(toy_fa(), item_id=f"t{j}") for j in range(4)]
            Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in items))
            outs = {}
            for name, extra in (("off", []), ("on", ["--eval-mid-preds"])):
                out = f"{d}/{name}"
                r = subprocess.run([sys.executable, SCRIPT, "--model", d + "/m", "--train", d + "/t.jsonl",
                                    "--eval", "val=" + d + "/t.jsonl", "--epochs", "2", "--items-per-update", "2",
                                    "--micro-bs", "2", "--max-len", "512", "--eval-max-len", "512",
                                    "--precision", "fp32master", "--no-grad-ckpt", "--log-every", "1", "--seed", "1",
                                    "--eval-at", "0.5,1", "--eval-mid", "val", "--out", out, *extra],
                                   capture_output=True, text=True, timeout=900, env=ENV)
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                m = json.loads(Path(out + "/metrics.json").read_text())
                outs[name] = ([{k: v for k, v in h.items() if k not in ("s_per_step", "approx_tok_per_s")}
                               for h in m["history"]], m["eval"],
                              [{k: v for k, v in t.items() if k != "s_elapsed"} for t in m["trajectory"]],
                              Path(out + "/preds_val.jsonl").read_text())
            self.assertFalse(Path(d + "/off/mid").exists())
            self.assertEqual(outs["off"], outs["on"])                # training and evaluations unchanged
            for t in outs["on"][2]:
                p = Path(f"{d}/on/mid/preds_val_e{t['epoch']:g}.jsonl")
                self.assertTrue(p.exists(), p)
                preds = [json.loads(x) for x in p.read_text().splitlines()]
                self.assertEqual(len(preds), 4)
                acc = sum(max(range(len(q["probs"])), key=q["probs"].__getitem__) ==
                          q["labels"].index(q["gold_label"]) for q in preds) / 4
                self.assertAlmostEqual(acc, t["val"]["acc"])

    def test_eval_student_time(self):
        with tempfile.TemporaryDirectory() as d:
            tiny(d)
            items = [dict(toy_fa(), item_id=f"t{j}") for j in range(3)]
            Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in items))
            script = str(ROOT / "scripts/eval_student.py")
            res = {}
            for name, extra in (("off", []), ("on", ["--time"])):
                r = subprocess.run([sys.executable, script, "--model", d + "/m", "--eval", "x=" + d + "/t.jsonl",
                                    "--max-len", "512", "--bs", "2", "--out", f"{d}/{name}", *extra],
                                   capture_output=True, text=True, timeout=900, env=ENV)
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                res[name] = (json.loads(Path(f"{d}/{name}/metrics.json").read_text()),
                             Path(f"{d}/{name}/preds_x.jsonl").read_text())
            self.assertFalse(Path(f"{d}/off/timing_x.json").exists())
            self.assertEqual(res["off"][1], res["on"][1])
            self.assertEqual(res["off"][0]["eval"], res["on"][0]["eval"])
            tm = json.loads(Path(f"{d}/on/timing_x.json").read_text())
            self.assertEqual((tm["n"], tm["batch"], len(tm["per_item_s"])), (3, 1, 3))
            self.assertGreater(tm["s_mean"], 0)
            self.assertGreater(tm["tokens_mean"], 10)
            self.assertIn("timing", res["on"][0])


if __name__ == "__main__":
    unittest.main()
