import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ladder_readout as lr  # noqa: E402

M = "Qwen3.5-4B"


def pred(item_id, group, ok, n=6):
    labels = "ABCDEF"[:n]
    probs = [0.1] * n
    probs[0 if ok else 1] = 0.5
    return {"item_id": item_id, "group": group, "labels": list(labels), "probs": probs, "gold_label": "A"}


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def subq(item_id, source, k, ok):
    return {"item_id": item_id, "source": source, "kind": "cot", "type": "derive", "question": f"q{k}?",
            "correct": int(ok)}


def data_dir(root):
    """val.jsonl per domain: item <dom>-v-0 with nodes q0..q19 at node_depth = k."""
    for data, src in (("student_v3sl", "knights_knaves"), ("student_numgen_tf", "numgen")):
        d = Path(root) / "data" / data
        d.mkdir(parents=True)
        write(d / "val.jsonl", [{"item_id": f"{src}-v-0", "source": src,
                                 "subqs": [{"question": f"q{k}?", "node_depth": k} for k in range(20)]}])
    return Path(root) / "data"


def kk_run(root, name, hi_ok, click=None):
    """kk_deep: 4-10 all correct; 11-20 correct where j < 10 * hi_ok (10 items per people count). Nodes: depth k
    correct iff k < 4."""
    d = Path(root) / name
    d.mkdir(parents=True)
    write(d / "preds_kkdeep.jsonl", [pred(f"kkdeep-{n}-{j}", f"kk/d{n}" if n <= 16 else f"kkx/d{n}",
                                          n <= 10 or j < 10 * hi_ok) for n in range(4, 21) for j in range(10)])
    write(d / "preds_diag.jsonl", [pred(f"diag-{g:03d}-{t}", "diag", g < 10 * hi_ok)
                                   for g in range(10) for t in ("R0", "R1", "R2", "C")])
    write(d / "preds_subq.jsonl", [subq("knights_knaves-v-0", "knights_knaves", k, k < 4) for k in range(20)])
    traj = [{"step": 50 * i, "epoch": 0.5 * i, "s_elapsed": 600.0 * i,
             "diagA": {"acc": 0.45 if click is not None and 0.5 * i >= click else 0.3}} for i in range(1, 7)]
    write(d / "preds_bbh.jsonl", [pred(f"bbh-{t}-{j}", f"bbh/{t}", t == "a" or j == 0) for t in "ab" for j in range(2)])
    ev = {"val": {"acc": 0.7}, "jevbench": {"acc": 0.6}, "folio": {"acc": 0.5},
          "local_kk": {"all": {"n": 20, "acc": 0.9}, "derived_nontrivial": {"n": 10, "acc": 0.8},
                       "depth/0": {"n": 4, "acc": 1.0}, "depth/5": {"n": 4, "acc": 0.75}}}
    (d / "trajectory.json").write_text(json.dumps(traj))
    (d / "metrics.json").write_text(json.dumps({"args": {"model": f"/m/{M}", "epochs": 4.0},
                                                "history": [{"step": 360, "s_per_step": 50.0}], "eval": ev}))
    return d


def ng_run(root, name, beyond_ok):
    """ng_d1..d4 all correct, ng_d5..d8 correct where j < 20 * beyond_ok (20 items each); ood half."""
    d = Path(root) / name
    d.mkdir(parents=True)
    for k in range(1, 9):
        write(d / f"preds_ng_d{k}.jsonl", [pred(f"numgen-d{k}-{j:05d}", f"numgen/d{k}", k <= 4 or j < 20 * beyond_ok, 5)
                                           for j in range(20)])
    write(d / "preds_ng_ood.jsonl", [pred(f"numgen-ood-{j:05d}", "numgen/ood", j % 2 == 0, 5) for j in range(20)])
    write(d / "preds_subq.jsonl", [subq("numgen-v-0", "numgen", k, k < 8) for k in range(20)])
    write(d / "preds_local_kk.jsonl", [{"item_id": "numgen-deep-0", "question": "other?", "depth": 3, "correct": 1}])
    ev = {"val": {"acc": 0.9}, "jevbench": {"acc": 0.55}, "local_kk": {"all": {"n": 8, "acc": 0.5}}}
    traj = [{"step": 100, "epoch": 1.0, "s_elapsed": 7200.0, "ng_d4": {"acc": 0.9}}]
    (d / "metrics.json").write_text(json.dumps({"args": {"model": f"/m/{M}"}, "trajectory": traj, "eval": ev}))
    return d


class TestParse(unittest.TestCase):
    def test_names(self):
        p = lr.parse_run_name
        self.assertEqual(p("A2-c12-Qwen3.5-4B-student_v3sl-s12"),
                         {"arm": "A2", "tag": "c12", "model": "Qwen3.5-4B", "data": "student_v3sl", "seed": 12})
        self.assertEqual(p("TFL-c4-Qwen3.5-2B-Base-student_v3sl-s13")["model"], "Qwen3.5-2B-Base")
        g = p("G3-c4kk-gemma-3-4b-pt-student_numgen_tf-s12", "gemma-3-4b-pt")
        self.assertEqual((g["arm"], g["tag"], g["model"], g["data"]), ("G3", "c4kk", "gemma-3-4b-pt", "student_numgen_tf"))
        h = p("GFPS-nw0.5-c4-Mistral-7B-student_v3sl-s3")              # no hint, no Qwen: lowercase tag tokens
        self.assertEqual((h["arm"], h["tag"], h["model"]), ("GFPS", "nw0.5-c4", "Mistral-7B"))
        self.assertIsNone(p("fsdp12b-686772"))
        self.assertEqual(lr.domain_of("student_numgen_tf"), "numgen")
        self.assertEqual(lr.domain_of("student_v3sl"), "kk")


class TestReadout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        dd = data_dir(root)
        runs = root / "pull"
        kk_run(runs, f"A2-c12-{M}-student_v3sl-s12", 0.5)
        kk_run(runs, f"GFPS-c4-{M}-student_v3sl-s12", 0.7, click=1.0)
        kk_run(runs, f"TFL-c4-{M}-student_v3sl-s12", 1.0, click=1.5)
        ng_run(runs, f"A2-c12-{M}-student_numgen_tf-s12", 0.5)
        ng_run(runs, f"TFL-c4-{M}-student_numgen_tf-s12", 0.75)
        (runs / f"GTF-c4-{M}-student_v3sl-s12").mkdir()             # still running: no metrics.json
        cls.out = root / "o" / "ladder.md"
        with redirect_stdout(StringIO()) as buf:
            cls.res = lr.main([str(runs), "--data-dir", str(dd), "--boot", "300", "--boot-seed", "100",
                               "--out", str(cls.out)])
        cls.md = buf.getvalue()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_files_and_skips(self):
        self.assertTrue(self.out.exists() and self.out.with_suffix(".json").exists())
        self.assertEqual(self.out.read_text().strip(), self.md.strip())
        self.assertIn("GTF-c4", self.md.split("\n")[3])                  # skipped line names the running dir
        self.assertEqual(set(self.res["sections"]), {"kk", "numgen"})

    def test_kk(self):
        runs = self.res["sections"]["kk"]["runs"]
        a2, tfl = runs["A2-c12"][12], runs["TFL-c4"][12]
        self.assertEqual((a2["kk4-10"], a2["kk11-20"], tfl["kk11-20"]), (1.0, 0.5, 1.0))
        self.assertEqual(a2["kk11-20_n"], 100)
        self.assertEqual(a2["people_acc"][20], 0.5)
        self.assertAlmostEqual(a2["dA"], 0.5)
        self.assertAlmostEqual(a2["bbh"], 0.75)                          # macro: task a 1.0, task b 0.5
        self.assertEqual((a2["val"], a2["jevbench"], a2["folio"]), (0.7, 0.6, 0.5))
        self.assertAlmostEqual(a2["hours"], 5.0)
        self.assertEqual((tfl["click_epoch"], runs["GFPS-c4"][12]["click_epoch"], a2["clicked"]), (1.5, 1.0, False))
        self.assertEqual(tfl["node_bins"], {"0": 1.0, "1-3": 1.0, "4-7": 0.0, "8-12": 0.0, ">=13": 0.0})
        self.assertEqual(tfl["node_join"], {"data": 20, "local": 0, "none": 0})
        self.assertEqual(tfl["local"]["derived_nontrivial"], 0.8)

    def test_kk_contrasts(self):
        cons = {(c["arm"], c["vs"]): c for c in self.res["sections"]["kk"]["contrasts"]}
        self.assertEqual(set(cons), {("GFPS-c4", "A2-c12"), ("TFL-c4", "A2-c12"), ("TFL-c4", "GFPS-c4")})
        c = cons[("TFL-c4", "A2-c12")]["endpoints"]["kk11-20"]
        self.assertAlmostEqual(c["mean"], 0.5)
        self.assertEqual((c["units"], c["seeds"]), (100, [12]))
        self.assertTrue(c["ci"][0] < 0.5 < c["ci"][1] and c["ci"][0] > 0.3)
        self.assertAlmostEqual(cons[("TFL-c4", "GFPS-c4")]["endpoints"]["dA"]["mean"], 0.3)
        self.assertIn("| TFL-c4 | A2-c12 | +0.500 [", self.md)

    def test_numgen(self):
        sec = self.res["sections"]["numgen"]
        a2, tfl = sec["runs"]["A2-c12"][12], sec["runs"]["TFL-c4"][12]
        self.assertEqual((a2["ng_in"], a2["ng_beyond"], a2["ng_drop"], a2["ng_ood"]), (1.0, 0.5, -0.5, 0.5))
        self.assertEqual((tfl["ng_beyond"], tfl["ng"]["8"]), (0.75, 0.75))
        self.assertEqual((a2["hours"], a2["hours_src"]), (2.0, "trajectory"))
        self.assertIsNone(a2.get("click_epoch"))
        # the local probe items are not val items: every node joins through data/<data>/val.jsonl
        self.assertEqual(tfl["node_join"], {"data": 20, "local": 0, "none": 0})
        self.assertEqual(tfl["node_bins"]["4-7"], 1.0)
        self.assertEqual(tfl["node_bins"]["8-12"], 0.0)
        cons = {(c["arm"], c["vs"]): c for c in sec["contrasts"]}
        c = cons[("TFL-c4", "A2-c12")]["endpoints"]["ng_beyond"]
        self.assertAlmostEqual(c["mean"], 0.25)
        self.assertEqual(c["units"], 80)
        self.assertTrue(cons[("TFL-c4", "GFPS")]["ref_missing"])
        self.assertIn("GFPS (no run)", self.md)


if __name__ == "__main__":
    unittest.main()
