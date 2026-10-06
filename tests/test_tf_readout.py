import hashlib
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
NAMES = tf_readout.TRAIN_NAMES + ["Quinn", "Rosa", "Sam", "Tina", "Umar", "Vera", "Will", "Yara"]
M = "Qwen3.5-2B-Base"


def pred(item_id, group, ok):
    probs = [0.1] * 6
    probs[0 if ok else 1] = 0.5
    return {"item_id": item_id, "group": group, "labels": LABELS, "probs": probs, "gold_label": "A"}


def kk_group(n):
    return f"kk/d{n}" if n <= 16 else f"kkx/d{n}"


def opt_line(letter, names, roles):
    return f"{letter}) " + ", ".join(f"{nm}: {r}" for nm, r in zip(names, roles))


def kk_item(n, j):
    """Gold A. j % 3 == 0: shortcut-valid (each distractor flips one person, so the majority is the gold vector);
    otherwise every distractor is the same other vector (the majority points away from gold)."""
    names = (tf_readout.TRAIN_NAMES if n <= 16 else NAMES[-8:] + tf_readout.TRAIN_NAMES)[:n]
    gold = ["knight" if (i + j) % 2 else "knave" for i in range(n)]
    flip = lambda v, i: [("knave" if r == "knight" else "knight") if k == i else r for k, r in enumerate(v)]  # noqa
    if j % 3 == 0:
        opts = [gold] + [flip(gold, i) for i in range(5)]
    else:
        other = flip(flip(gold, 0), 1)
        opts = [gold] + [other] * 5
    prompt = "Puzzle\n\nQuestion: Who?\nOptions:\n" + "\n".join(opt_line(L, names, o) for L, o in zip(LABELS, opts))
    return {"item_id": f"kkdeep-{n}-{j}", "group": kk_group(n), "prompt": prompt, "gold_label": "A"}


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
                f.write(json.dumps(pred(f"kkdeep-{n}-{j}", kk_group(n), rng.random() < (p_lo if n <= 10 else p_hi)))
                        + "\n")
    traj = [{"step": 50 * (i + 1), "epoch": 0.5 * (i + 1),
             "diagA": {"acc": 0.45 if click_at is not None and i >= click_at else 0.3}} for i in range(6)]
    ev = {"diagA": {"acc": 0.45 if click_at is not None else 0.3}}
    if local:
        ev["local_kk"] = {"all": {"n": 40, "acc": 0.75}, "depth/0": {"n": 10, "acc": 1.0},
                          "depth/2": {"n": 10, "acc": 0.8}, "depth/3": {"n": 10, "acc": 0.6},
                          "depth/14": {"n": 10, "acc": 0.6}, "depth_bin/0": {"n": 10, "acc": 1.0},
                          "derived_nontrivial": {"n": 12, "acc": 0.85}}
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
        cls.items = r + "/kk_deep.jsonl"
        Path(cls.items).write_text("\n".join(json.dumps(kk_item(n, j)) for n in range(4, 21) for j in range(30)))
        cls.runs = r + "/runs"
        for s in (12, 13, 14):
            make_run(cls.runs, f"TF-c4-{M}-student_v3f-s{s}", 0.6, 0.85, s, click_at=1, local=True)  # stand-in TFL
            make_run(cls.runs, f"GFPS-c4-{M}-student_v3sl-s{s}", 0.6, 0.45, s, click_at=2, local=True)
            make_run(cls.runs, f"A2-c12-{M}-student_v3sl-s{s}", 0.5, 0.40, s, click_at=3)
            make_run(cls.runs, f"G3-c4-{M}-student_v3f-s{s}", 0.6, 0.50, s, click_at=2)
        for s in (11, 12, 13, 14):
            make_run(cls.runs, f"A2-c8-{M}-student_v3-s{s}", 0.4, 0.30, s, click_at=None)          # never clicks
        for s in (12, 13):
            make_run(cls.runs, f"TFM-c4-{M}-student_v3f-s{s}", 0.6, 0.35, s, click_at=3)            # stand-in TFLM
            make_run(cls.runs, f"TFLN-c4-{M}-student_v3sl-s{s}", 0.6, 0.55, s, click_at=2, local=True)
            make_run(cls.runs, f"GFP-c4-{M}-student_v3sl-s{s}", 0.6, 0.40, s, click_at=2)
        make_run(cls.runs, f"A2-v5s-{M}-student_v5_10k-s0", 0.5, 0.40, 0)                         # other data
        make_run(cls.runs, f"G3R-c4-{M}-student_v3f-s12", 0.5, 0.40, 12)                           # not an arm
        cls.base = ["--alias", "TF=TFL", "--alias", "TFM=TFLM", "--kkdeep-items", cls.items, "--boot", "400",
                    "--boot-seed", "200"]
        cls.argv = [cls.runs + "/*"] + cls.base + ["--out", r + "/out/readout"]
        cls.out, cls.text = run(cls.argv)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _glob_without(self, *prefixes):
        return [str(p) for p in Path(self.runs).iterdir() if not p.name.startswith(prefixes)]

    def test_parse_name(self):
        self.assertEqual(tf_readout.parse_name("A2-c4ls0-Qwen3.5-2B-Base-student_v3-s13"),
                         {"arm": "A2", "tag": "c4ls0", "model": "Qwen3.5-2B-Base", "data": "student_v3", "seed": 13})
        self.assertEqual(tf_readout.parse_name("GFPS-c4-Qwen3.5-2B-Base-student_v3sl-s12")["arm"], "GFPS")
        self.assertIsNone(tf_readout.parse_name("base-subq-Qwen3.5-2B-Base"))

    def test_kkx_groups_land_in_17_20(self):
        r = self.out["runs"]["TFL"][12]
        self.assertEqual(r["kk17-20_n"], 120)
        self.assertEqual(r["kk11-20_n"], 300)
        self.assertEqual(r["kk4-10_n"], 210)
        self.assertEqual(r["people_n"][20], 30)

    def test_shortcut_info(self):
        sc = tf_readout.shortcut_info(kk_item(12, 0))
        self.assertTrue(sc["shortcut"])
        self.assertEqual(sc["heur_acc"], 1.0)
        no = tf_readout.shortcut_info(kk_item(12, 1))
        self.assertFalse(no["shortcut"])
        self.assertEqual(no["heur_acc"], 0.0)
        self.assertEqual(no["unseen"], 0.0)
        self.assertAlmostEqual(tf_readout.shortcut_info(kk_item(20, 1))["unseen"], 8 / 20)
        # 3-3 tie on a person counts as knave (redteam F2 / kkshort.py)
        names = ["Alice", "Bob"]
        opts = [["knave", "knight"], ["knight", "knight"], ["knave", "knave"], ["knight", "knave"],
                ["knight", "knave"], ["knave", "knave"]]          # Alice 3-3 -> knave; Bob 2 knight / 4 knave
        it = {"prompt": "\n".join(opt_line(L, names, o) for L, o in zip(LABELS, opts)), "gold_label": "C"}
        tie = tf_readout.shortcut_info(it)
        self.assertEqual(tie["heur_acc"], 0.5)                    # C and F are both (knave, knave)
        self.assertFalse(tie["shortcut"])                         # (a knight tie-break would pick D/E)

    def test_stratum_endpoints(self):
        r = self.out["runs"]["TFL"][12]
        self.assertEqual(r["kk11-16sc_n"], 60)                    # j % 3 == 0 of 30 items x 6 sizes
        self.assertEqual(r["kk11-16ns_n"], 120)
        self.assertEqual(r["kk_unjoined"], 0)
        self.assertAlmostEqual(r["unseen17_20"], sum(8 / n for n in range(17, 21)) / 4)
        self.assertEqual(self.out["stratum"]["11-16"]["shortcut"], 60)

    @unittest.skipUnless(tf_readout.KKDEEP_ITEMS.exists(), "data/eval/kk_deep.jsonl absent")
    def test_stratum_reproduces_redteam_f2(self):
        st = tf_readout.stratum_summary(tf_readout.load_kkdeep_items(tf_readout.KKDEEP_ITEMS))
        self.assertEqual((st["11-16"]["shortcut"], st["11-16"]["n"]), (226, 900))
        self.assertEqual((st["17-20"]["shortcut"], st["17-20"]["n"]), (109, 400))
        for b, acc in (("4-6", .179), ("7-10", .206), ("11-16", .340), ("17-20", .355)):
            self.assertAlmostEqual(st[b]["heur_acc"], acc, places=3)
        self.assertAlmostEqual(st["17-20"]["unseen"], 1 / 3, places=3)
        self.assertEqual(st["17-20"]["items_with_unseen"], 400)
        self.assertEqual(st["11-16"]["unseen"], 0)

    def test_variants_a2_prefers_c12(self):
        self.assertEqual(sorted(self.out["runs"]), ["A2", "G3", "GFP", "GFPS", "TFL", "TFLM", "TFLN"])
        self.assertEqual(self.out["runs"]["A2"][12]["tag"], "c12")          # c8 has more runs, c12 still wins
        self.assertTrue(self.out["a2_registered"])
        self.assertEqual(self.out["runs"]["TFL"][12]["orig_arm"], "TF")
        self.assertTrue(any("A2-c8" in d for d in self.out["not_used"]))
        self.assertTrue(any("student_v5_10k" in d for d, _ in self.out["skipped"]))
        self.assertTrue(any("G3R" in d for d, _ in self.out["skipped"]))

    def test_h5_on_reference_a2(self):
        out, text = run(self._glob_without("A2-c12") + self.base + ["--out", ""])
        self.assertEqual(out["runs"]["A2"][12]["tag"], "c8")
        self.assertFalse(out["a2_registered"])
        self.assertIn("REFERENCE", out["a2_variant"])
        self.assertTrue(out["verdicts"]["H5"].startswith("not testable as registered"), out["verdicts"]["H5"])
        self.assertNotIn("H5", out["holm"])                                  # not in the Holm family
        self.assertAlmostEqual(out["holm"]["H2'"], min(1.0, 2 * next(
            c for c in out["contrasts"] if c["label"] == "H2'" and c["endpoint"] == "kk11-20")["p_one"]))
        self.assertTrue(out["verdicts"]["wording"].startswith("pending"))
        self.assertIn("REFERENCE A2-c8/v3", text)

    def test_fallback_gfp(self):
        out, text = run(self._glob_without("GFPS") + self.base + ["--out", ""])
        self.assertEqual((out["h2_comparator"], out["h2_fallback"]), ("GFP", True))
        self.assertIn("FALLBACK", out["verdicts"]["H2'"])
        self.assertIn("FALLBACK", out["verdicts"]["wording"])
        self.assertIn("H2' comparator: GFP (FALLBACK", text)
        c = next(c for c in out["contrasts"] if c["label"] == "H2'" and c["endpoint"] == "kk11-20")
        self.assertEqual(c["contrast"], "TFL-GFP")

    def test_click_auc_hours_local(self):
        tfl, a2 = self.out["runs"]["TFL"][12], self.out["runs"]["A2"][12]
        self.assertEqual((tfl["click_step"], tfl["click_epoch"], tfl["clicked"]), (100, 1.0, True))
        self.assertEqual((a2["clicked"], a2["click_step"]), (True, 200))
        self.assertAlmostEqual(tfl["hours"], 3.0)
        self.assertEqual(tfl["local_key"], "local_kk")
        self.assertAlmostEqual(tfl["local"]["1-3"], 0.7)
        self.assertAlmostEqual(tfl["local"]["derived_nontrivial"], 0.85)
        self.assertIn("+0.053", self.text)                                   # .85 - .797
        self.assertIn("GFPS s12 (transfer)", self.text)

    def test_endpoints_and_exact_paired_difference(self):
        runs = self.out["runs"]
        for lab, comp in (("H2'", "GFPS"), ("H5", "A2")):
            c = next(c for c in self.out["contrasts"] if c["label"] == lab and c["endpoint"] == "kk11-20")
            exact = sum(runs["TFL"][s]["kk11-20"] - runs[comp][s]["kk11-20"] for s in (12, 13, 14)) / 3
            self.assertAlmostEqual(c["mean"], exact)
            self.assertLessEqual(c["ci"][0], exact)
            self.assertEqual(c["seeds"], [12, 13, 14])
        e1 = next(c for c in self.out["contrasts"] if c["label"] == "H2'" and c["endpoint"] == "E1p")
        self.assertAlmostEqual(e1["per_seed"][12], runs["TFL"][12]["E1p"] - runs["GFPS"][12]["E1p"])
        labels = {c["label"] for c in self.out["contrasts"]}
        self.assertEqual(labels, {"H2'", "H5", "TFL-TFLM", "TFL-TFLN", "TFL-GFP", "TFA-TFL", "TFL-G3"})
        self.assertEqual({c["endpoint"] for c in self.out["contrasts"]}, set(tf_readout.ENDPOINTS))
        self.assertIn("(unseen names", self.text)

    def test_unpaired_fallback(self):
        runs = {"X": {1: {"_units": {"e": {"a": 1.0, "b": 1.0}}}}, "Y": {2: {"_units": {"e": {"a": 0.0, "b": 1.0}}}}}
        c = tf_readout.contrast(runs, "X", "Y", "e", 200, 50)
        self.assertFalse(c["paired"])
        self.assertAlmostEqual(c["mean"], 0.5)

    def test_verdicts_tree_forcing(self):
        v = self.out["verdicts"]
        self.assertTrue(v["H2'"].startswith("PASS"), v["H2'"])
        self.assertTrue(v["H5"].startswith("PASS"), v["H5"])
        self.assertTrue(v["wording"].startswith("'Tree Forcing'"), v["wording"])
        self.assertNotIn("provisional", v["wording"])
        self.assertIn("not licensable in this wave", v["edges"])
        self.assertTrue(v["negative"].startswith("no"))
        self.assertIn("speed vs GFPS", v)
        self.assertLessEqual(self.out["holm"]["H2'"], 0.05)
        for k in ("H1", "H2", "H4 vs GFP", "reading", "vs G3"):              # legacy rules removed
            self.assertNotIn(k, v)

    def test_did_report_only(self):
        d = self.out["did"]
        self.assertEqual(list(d), ["GFPS", "A2", "G3"])
        e = d["GFPS"]
        self.assertAlmostEqual(e["did"], e["delta"]["kk11-20"] - e["delta"]["kk4-10"])
        self.assertAlmostEqual(e["did_ns"], e["delta"]["kk11-16ns"] - e["delta"]["kk4-10"])
        self.assertEqual(sorted(e["per_seed_ns"]), [12, 13, 14])

    def _decide(self, m2, s2, m5, s5, a2_registered=True):
        C = {}
        for k, m, ss in (("H2'", m2, s2), ("H5", m5, s5)):
            C[(k, "kk11-20")] = {"contrast": "TFL-X", "mean": m, "ci": (m - .02, m + .02), "seeds": [12, 13, 14],
                                 "per_seed": dict(zip((12, 13, 14), ss))}
        holm1 = tf_readout.holm({"H2'": 0.001 if m2 > 0 else 0.9, "H5": 0.001 if m5 > 0 else 0.9}, m=2)
        return tf_readout.decide({}, C, holm1, a2_registered, "GFPS", False)

    def test_wording_table(self):
        self.assertTrue(self._decide(.05, (.04, .05, .06), .06, (.05, .06, .07))["wording"].startswith(
            "'Tree Forcing'"))
        self.assertIn("dense node supervision", self._decide(.01, (.02, -.01, .02), .06, (.05, .06, .07))["wording"])
        self.assertIn("no method claim", self._decide(.06, (.05, .06, .07), .02, (.01, .03, .02))["wording"])
        # one negative seed fails the test even with a large pooled mean
        self.assertIn("dense node supervision", self._decide(.08, (.2, .1, -.06), .06, (.05, .06, .07))["wording"])
        neg = self._decide(-.06, (-.05, -.06, -.07), .06, (.05, .06, .07))
        self.assertTrue(neg["negative"].startswith("NEGATIVE"), neg["negative"])
        neg5 = self._decide(.06, (.05, .06, .07), -.05, (-.05, -.06, -.04))
        self.assertTrue(neg5["negative"].startswith("NEGATIVE"))
        self.assertIn("provisional", self._decide(.05, (.04, .05), .06, (.05, .06))["wording"])
        # the negative result reaches the wording line (item 5), not only the "negative" key
        self.assertIn("NEGATIVE RESULT: TFL below A2", neg5["wording"])
        self.assertIn("NEGATIVE RESULT: TFL below GFPS", neg["wording"])
        self.assertNotIn("NEGATIVE", self._decide(-.04, (-.04, -.04, -.04), .06, (.05, .06, .07))["wording"])

    def test_wording_fallback_whatever_the_outcome(self):
        C = {(k, "kk11-20"): {"contrast": "TFL-X", "mean": m, "ci": (m - .02, m + .02), "seeds": [12, 13, 14],
                              "per_seed": {12: m, 13: m, 14: m}} for k, m in (("H2'", .06), ("H5", .01))}
        v = tf_readout.decide({}, C, tf_readout.holm({"H2'": .001, "H5": .3}, m=2), True, "GFP", True)
        self.assertTrue(v["wording"].startswith("H5 fails"), v["wording"])
        self.assertIn("FALLBACK", v["wording"])

    def test_coprimary_registered_seeds_only_and_holm_paired_only(self):
        extra = Path(self.tmp.name) / "extra"
        if not extra.exists():
            for arm, d in (("TF", "student_v3f"), ("GFPS", "student_v3sl"), ("A2-c12", "student_v3sl")):
                name = f"{arm}-c4-{M}-{d}-s11" if arm != "A2-c12" else f"A2-c12-{M}-{d}-s11"
                make_run(extra, name, 0.6, 0.05 if arm == "TF" else 0.9, 11, click_at=1)   # s11: TFL far below
        out, _ = run([self.runs + "/*", str(extra) + "/*"] + self.base + ["--out", ""])
        for k in ("H2'", "H5"):
            c = next(c for c in out["contrasts"] if c["label"] == k and c["endpoint"] == "kk11-20")
            self.assertEqual(c["seeds"], [12, 13, 14])
            self.assertEqual(c["excluded_seeds"], [11])
            self.assertTrue(out["verdicts"][k].startswith("PASS"), out["verdicts"][k])
            self.assertIn("unregistered seed(s) [11] not pooled", out["verdicts"][k])
        r = next(c for c in out["contrasts"] if c["label"] == "TFL-G3" and c["endpoint"] == "kk11-20")
        self.assertEqual(r["excluded_seeds"], [])                           # report-only rows are not restricted
        # an unpaired co-primary contrast (no common registered seed) never enters the Holm family
        runs = {"TFL": {12: {"_units": {"e": {"a": 1.0, "b": 1.0}}}}, "GFPS": {15: {"_units": {"e": {"a": 0.0}}}}}
        c = tf_readout.contrast(runs, "TFL", "GFPS", "e", 200, 50, tf_readout.COPRIMARY_SEEDS)
        self.assertEqual(c["missing"], ["GFPS"])
        self.assertEqual(c["excluded_seeds"], [15])
        nopair = [str(p) for p in Path(self.runs).iterdir() if not p.name.startswith("GFPS")]
        nopair += [str(make_run(Path(self.tmp.name) / "gfps15", f"GFPS-c4-{M}-student_v3sl-s15", .6, .45, 15))] \
            if not (Path(self.tmp.name) / "gfps15").exists() else [str(next((Path(self.tmp.name) / "gfps15").iterdir()))]
        out, _ = run(nopair + self.base + ["--out", ""])
        self.assertNotIn("H2'", out["holm"])
        self.assertIn("H5", out["holm"])

    def test_holm(self):
        h = tf_readout.holm({"a": 0.01, "b": 0.04}, m=2)
        self.assertAlmostEqual(h["a"], 0.02)
        self.assertAlmostEqual(h["b"], 0.04)
        self.assertAlmostEqual(tf_readout.holm({"a": 0.03}, m=2)["a"], 0.06)

    def test_prereg_md5_is_current(self):
        if tf_readout.PREREG.exists():
            self.assertEqual(hashlib.md5(tf_readout.PREREG.read_bytes()).hexdigest(), tf_readout.PREREG_MD5)
            self.assertNotIn("WARNING", tf_readout.prereg_status())

    def test_standin_copies_keep_original(self):
        out, text = run([str(p) for p in Path(self.runs).iterdir() if not p.name.startswith("GFP")] + self.base
                        + ["--standin", "GFPS=G3", "--out", ""])
        self.assertIn("G3", out["runs"])
        self.assertEqual(out["runs"]["GFPS"][12]["standin_of"], "G3")
        self.assertFalse(out["h2_fallback"])
        self.assertIn("G3 copied as GFPS", text)
        self.assertFalse(any("G3-c4" in d for d in out["not_used"]))

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
