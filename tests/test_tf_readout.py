import json
import os
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import tf_readout  # noqa: E402

LABELS = ["A", "B", "C", "D", "E", "F"]


def pred(item_id, group, ok):
    probs = [0.1] * 6
    probs[0 if ok else 1] = 0.5
    return {"item_id": item_id, "group": group, "labels": LABELS, "probs": probs, "gold_label": "A"}


def make_run(root, name, p_lo, p_hi, seed, click_at=None, local=False):
    """p_lo: accuracy on diag and kk_deep 4-10, p_hi: on kk_deep 11-20 (Bernoulli per item, fixed rng)."""
    d = Path(root) / name
    d.mkdir(parents=True)
    rng = random.Random(name)
    with open(d / "preds_diag.jsonl", "w") as f:
        for g in range(40):
            for tag in ("R0", "R1", "R2", "C"):
                f.write(json.dumps(pred(f"diag-{g:03d}-{tag}", "diag", rng.random() < p_lo)) + "\n")
    with open(d / "preds_kkdeep.jsonl", "w") as f:
        for n in range(4, 21):
            for j in range(30):
                grp = f"kk/d{n}" if n <= 16 else f"kkx/d{n}"
                f.write(json.dumps(pred(f"kkdeep-{n}-{j}", grp, rng.random() < (p_lo if n <= 10 else p_hi))) + "\n")
    traj = [{"step": 50 * (i + 1), "epoch": 0.5 * (i + 1),
             "diagA": {"acc": 0.45 if click_at is not None and i >= click_at else 0.3}} for i in range(6)]
    ev = {"diagA": {"acc": 0.45 if click_at is not None else 0.3}}
    if local:
        ev["local_kk"] = {"all": {"n": 40, "acc": 0.75}, "depth/0": {"n": 10, "acc": 1.0},
                          "depth/2": {"n": 10, "acc": 0.8}, "depth/3": {"n": 10, "acc": 0.6},
                          "depth/14": {"n": 10, "acc": 0.6}, "depth_bin/0": {"n": 10, "acc": 1.0}}
    (d / "metrics.json").write_text(json.dumps({
        "args": {"epochs": 4.0, "seed": seed}, "history": [{"step": 300, "s_per_step": 36.0}],
        "trajectory": traj, "eval": ev}))
    return d


def run(argv):
    with redirect_stdout(StringIO()) as buf:
        out = tf_readout.main(argv)
    return out, buf.getvalue()


class TestTfReadout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        r = cls.tmp.name
        M = "Qwen3.5-2B-Base"
        for s in (12, 13):
            make_run(r, f"TF-c4-{M}-student_v3f-s{s}", 0.6, 0.85, s, click_at=1, local=True)   # stand-in TFL
            make_run(r, f"TFM-c4-{M}-student_v3f-s{s}", 0.6, 0.35, s, click_at=3)                # stand-in TFLM
            make_run(r, f"GFP-{M}-student_v3sl-s{s}", 0.6, 0.40, s, click_at=2)
            make_run(r, f"G3-c4-{M}-student_v3f-s{s}", 0.6, 0.50, s, click_at=2)
            make_run(r, f"A2-c8-{M}-student_v3-s{s}", 0.4, 0.30, s, click_at=None)              # never clicks
        make_run(r, f"A2-c16-{M}-student_v3-s12", 0.5, 0.40, 12, click_at=4)
        make_run(r, f"A2-v5s-{M}-student_v5_10k-s0", 0.5, 0.40, 0)                             # other data
        make_run(r, f"G3R-c4-{M}-student_v3f-s12", 0.5, 0.40, 12)                               # not an arm
        cls.argv = [r + "/*", "--alias", "TF=TFL", "--alias", "TFM=TFLM", "--boot", "400", "--boot-seed", "200",
                    "--out", r + "/out/readout"]
        cls.out, cls.text = run(cls.argv)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_parse_name(self):
        self.assertEqual(tf_readout.parse_name("A2-c4ls0-Qwen3.5-2B-Base-student_v3-s13"),
                         {"arm": "A2", "tag": "c4ls0", "model": "Qwen3.5-2B-Base", "data": "student_v3", "seed": 13})
        self.assertEqual(tf_readout.parse_name("GFP-Qwen3.5-2B-Base-student_v3sl-s12")["tag"], "")
        self.assertIsNone(tf_readout.parse_name("base-subq-Qwen3.5-2B-Base"))

    def test_kkx_groups_land_in_17_20(self):
        r = self.out["runs"]["TFL"][12]
        self.assertEqual(r["kk17-20_n"], 120)
        self.assertEqual(r["kk11-20_n"], 300)
        self.assertEqual(r["kk4-10_n"], 210)
        self.assertIn(18, r["people_acc"])
        self.assertEqual(r["people_n"][20], 30)

    def test_alias_variant_and_filters(self):
        self.assertEqual(sorted(self.out["runs"]), ["A2", "G3", "GFP", "TFL", "TFLM"])
        self.assertEqual(self.out["runs"]["TFL"][12]["orig_arm"], "TF")
        self.assertEqual(self.out["runs"]["A2"][12]["tag"], "c8")        # most runs wins
        self.assertTrue(any("A2-c16" in d for d in self.out["not_used"]))
        self.assertTrue(any("student_v5_10k" in d for d, _ in self.out["skipped"]))
        self.assertTrue(any("G3R" in d for d, _ in self.out["skipped"]))
        out, _ = run(self.argv[:-2] + ["--pick", "A2=c16", "--out", ""])
        self.assertEqual(list(out["runs"]["A2"]), [12])
        self.assertEqual(out["runs"]["A2"][12]["tag"], "c16")

    def test_click_auc_hours_local(self):
        tfl, a2 = self.out["runs"]["TFL"][12], self.out["runs"]["A2"][12]
        self.assertEqual((tfl["click_step"], tfl["click_epoch"], tfl["clicked"]), (100, 1.0, True))
        self.assertEqual((a2["clicked"], a2["click_step"]), (False, 300))   # censored at the last (end) point
        self.assertAlmostEqual(a2["auc"], 0.3)
        self.assertAlmostEqual(tfl["hours"], 3.0)
        self.assertEqual(tfl["local_key"], "local_kk")
        self.assertAlmostEqual(tfl["local"]["1-3"], 0.7)                   # depth 2 and 3 pooled by n
        self.assertAlmostEqual(tfl["local"][">=13"], 0.6)
        self.assertNotIn("4-7", tfl["local"])

    def test_contrast_matches_exact_paired_difference(self):
        c = next(c for c in self.out["contrasts"] if c["contrast"] == "TFL-TFLM" and c["endpoint"] == "kk11-20")
        runs = self.out["runs"]
        exact = sum(runs["TFL"][s]["kk11-20"] - runs["TFLM"][s]["kk11-20"] for s in (12, 13)) / 2
        self.assertAlmostEqual(c["mean"], exact)
        self.assertLessEqual(c["ci"][0], exact)
        self.assertGreaterEqual(c["ci"][1], exact)
        self.assertAlmostEqual(c["per_seed"][12], runs["TFL"][12]["kk11-20"] - runs["TFLM"][12]["kk11-20"])
        self.assertEqual(c["seeds"], [12, 13])
        self.assertTrue(c["paired"])
        a2 = next(c for c in self.out["contrasts"] if c["contrast"] == "TFL-A2" and c["endpoint"] == "dA")
        self.assertTrue(a2["seed_label_only"])

    def test_unpaired_fallback(self):
        runs = {"X": {1: {"_units": {"e": {"a": 1.0, "b": 1.0}}}}, "Y": {2: {"_units": {"e": {"a": 0.0, "b": 1.0}}}}}
        c = tf_readout.contrast(runs, "X", "Y", "e", 200, 50)
        self.assertFalse(c["paired"])
        self.assertAlmostEqual(c["mean"], 0.5)

    def test_decision_rules(self):
        v = self.out["verdicts"]
        self.assertTrue(v["H1"].startswith("PASS"), v["H1"])
        self.assertTrue(v["H2"].startswith("PASS"), v["H2"])
        self.assertIn("TFLM", self.out["h4"])                              # H4 runs only after H1/H2 pass
        self.assertTrue(v["H4 vs TFLM"].startswith("PASS"), v["H4 vs TFLM"])
        self.assertTrue(v["reading"].startswith("STRONG"), v["reading"])
        self.assertIn("recipe candidate", v["reading"])                    # TFL - G3 on 11-20 >= 0
        self.assertTrue(v["H3"].startswith("not testable"))
        self.assertIn("not decidable", v["vs G3"])                         # TFA missing
        self.assertTrue(v["H4 vs GFP"].startswith("PASS"), v["H4 vs GFP"])
        self.assertLessEqual(self.out["holm_h1h2"]["H1"], 0.05)

    def test_pending_reading_without_gfp(self):
        out, _ = run([self.tmp.name + "/[!G]*", self.tmp.name + "/G3-*", "--alias", "TF=TFL", "--alias", "TFM=TFLM",
                      "--boot", "300", "--boot-seed", "100", "--out", ""])
        self.assertNotIn("GFP", out["runs"])
        v = out["verdicts"]
        self.assertTrue(v["H1"].startswith("PASS"))
        self.assertTrue(v["H2"].startswith("not testable"))
        self.assertIn("H2 pending", v["reading"])
        self.assertNotIn("dropped", v["reading"])

    def test_holm(self):
        h = tf_readout.holm({"a": 0.01, "b": 0.04}, m=2)
        self.assertAlmostEqual(h["a"], 0.02)
        self.assertAlmostEqual(h["b"], 0.04)
        self.assertAlmostEqual(tf_readout.holm({"a": 0.03}, m=2)["a"], 0.06)

    def test_files_written(self):
        base = Path(self.tmp.name) / "out" / "readout"
        j = json.loads(Path(str(base) + ".json").read_text())
        self.assertIn("verdicts", j)
        md = Path(str(base) + ".md").read_text()
        self.assertIn("| TFL | 12 |", md)
        self.assertIn("alias (stand-ins): TF -> TFL, TFM -> TFLM", md)
        self.assertEqual(md.strip(), self.text.strip())


if __name__ == "__main__":
    unittest.main()
