import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ladder_readout as lr  # noqa: E402

M = "Qwen3.5-4B"
BASE = "73b3097"            # the last commit before the LR sweep / pronto domain: its tables must stay byte-identical
L4B = ROOT / "results" / "hpc" / "l4b"


def load_base(tmp: Path, name: str = "ladder_readout", commit: str = BASE):
    """scripts/<name>.py at `commit` as a module (None if git cannot read it)."""
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:scripts/{name}.py"], capture_output=True,
                         text=True)
    if src.returncode:
        return None
    f = tmp / f"{name}_base_{commit}.py"
    f.write_text(src.stdout)
    spec = importlib.util.spec_from_file_location(f"{name}_base_{commit}", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = ROOT                     # paths are printed relative to the repo, not to the temp copy
    return mod


def sections(md: str) -> list[tuple[str, str]]:
    """[(heading, text)] split at '## ' headings; the preamble has heading ''."""
    out, head, buf = [], "", []
    for line in md.split("\n"):
        if line.startswith("## "):
            out.append((head, "\n".join(buf)))
            head, buf = line, []
        buf.append(line)
    return out + [(head, "\n".join(buf))]


def run_md(mod, argv) -> str:
    with redirect_stdout(StringIO()) as buf:
        mod.main(argv)
    return buf.getvalue()


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
    for data, src in (("student_v3sl", "knights_knaves"), ("student_numgen_tf", "numgen"),
                      ("student_pronto_tf", "pronto")):
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
        self.assertEqual(lr.domain_of("student_pronto_tf"), "pronto")
        self.assertEqual(p("A2-c12lr3-Qwen3.5-4B-student_v3sl-s12")["tag"], "c12lr3")    # the LR stays in the arm name

    def test_lr_names(self):
        self.assertEqual(lr.split_lr("c12lr3"), ("c12", 3e-6))
        self.assertEqual(lr.split_lr("c4lr1"), ("c4", 1e-6))
        self.assertEqual(lr.split_lr("nw0.5-c4lr5"), ("nw0.5-c4", 5e-6))
        self.assertEqual(lr.split_lr("lr3-c4"), ("c4", 3e-6))
        self.assertEqual(lr.split_lr("c4kk"), ("c4kk", None))
        self.assertEqual(lr.split_lr(""), ("", None))
        self.assertEqual([lr.flr(x) for x in (3e-06, 1e-05, 2.5e-06, 1e-4)], ["3e-6", "1e-5", "2.5e-6", "0.0001"])
        self.assertEqual(lr.run_lr({"args": {"lr": 3e-06}}, "c12lr3"), (3e-6, "args", None))
        self.assertEqual(lr.run_lr({"args": {}}, "c12lr3"), (3e-6, "tag", None))
        self.assertEqual(lr.run_lr({}, "c12"), (1e-5, "default", None))
        self.assertIn("args.lr 1e-5", lr.run_lr({"args": {"lr": 1e-5}}, "c4lr3")[2])


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



# ---------------------------------------------------------------- LR sweep, pronto, legacy tables
def lr_fields(d, lr_=None, valkk=None, val=None, nll=1.0, n=100):
    """args.lr in metrics.json; preds_valkk / preds_val with item j correct iff j < acc * n (so a lower accuracy's
    correct set is a subset of a higher one's), their end-of-training NLL; td and gsm8k preds."""
    m = json.loads((d / "metrics.json").read_text())
    if lr_ is not None:
        m["args"]["lr"] = lr_
    for name, acc in (("valkk", valkk), ("val", val)):
        if acc is not None:
            write(d / f"preds_{name}.jsonl", [pred(f"{name}-{j:03d}", name, j < round(acc * n)) for j in range(n)])
            m["eval"][name] = {"n": n, "acc": acc, "nll": nll}
    write(d / "preds_td.jsonl", [pred(f"td-{j:03d}", "td/x", j < 20) for j in range(40)])
    write(d / "preds_gsm8k.jsonl", [pred(f"gsm-{j:03d}", "gsm8k", j < 10) for j in range(40)])
    (d / "metrics.json").write_text(json.dumps(m))
    return d


def pq_run(root, name, beyond_ok, wide_ok):
    """pq_d1..d4 all correct, pq_d5..d8 correct where j < 20 * beyond_ok; pq_wide where j < 20 * wide_ok."""
    d = Path(root) / name
    d.mkdir(parents=True)
    for k in range(1, 9):
        write(d / f"preds_pq_d{k}.jsonl", [pred(f"pronto-d{k}-{j:05d}", f"pronto/d{k}", k <= 4 or j < 20 * beyond_ok, 2)
                                           for j in range(20)])
    write(d / "preds_pq_wide.jsonl", [pred(f"pronto-wide-d{1 + j % 4}-{j:05d}", "pronto/wide", j < 20 * wide_ok, 2)
                                      for j in range(20)])
    write(d / "preds_subq.jsonl", [subq("pronto-v-0", "pronto", k, k < 6) for k in range(20)])
    ev = {"val": {"acc": 0.95}, "jevbench": {"acc": 0.6}}
    traj = [{"step": 100, "epoch": 1.0, "s_elapsed": 3600.0, "pq_d4": {"acc": 0.9}}]
    (d / "metrics.json").write_text(json.dumps({"args": {"model": f"/m/{M}", "lr": 1e-5}, "trajectory": traj,
                                                "eval": ev}))
    return d


def fake(arm, tag, lr_, seed, acc, nll=1.0):
    return {"arm": arm, "tag": tag, "lr": lr_, "seed": seed, "sel_acc": acc, "sel_nll": nll, "_units": {}}


def fake_groups(*runs):
    g = {}
    for r in runs:
        g.setdefault(r["arm"] + (f"-{r['tag']}" if r["tag"] else ""), {})[r["seed"]] = r
    return g


class TestLRSelection(unittest.TestCase):
    def select(self, *runs, dom="kk", kk=None):
        g = fake_groups(*runs)
        methods, warn = lr.lr_methods(g)
        return methods, lr.select_lr(g, methods, dom, kk, 200, 50)

    def test_sweep_seed_only(self):
        """1e-5 has s12 .70 and s13 .80 (mean .75 = 3e-6's s12): the sweep seed s12 decides, so 3e-6."""
        methods, sel = self.select(fake("A2", "c12", 1e-5, 12, 0.70), fake("A2", "c12", 1e-5, 13, 0.80),
                                   fake("A2", "c12lr3", 3e-6, 12, 0.75), fake("A2", "c12lr1", 1e-6, 12, 0.60))
        self.assertEqual(list(methods), ["A2-c12"])
        self.assertEqual(methods["A2-c12"]["lrs"], {1e-6: "A2-c12lr1", 3e-6: "A2-c12lr3", 1e-5: "A2-c12"})
        s = sel["A2-c12"]
        self.assertEqual((s["sweep_seeds"], s["chosen"]), ([12], 3e-6))
        self.assertAlmostEqual(s["score"][1e-5], 0.70)

    def test_extra_seeds_never_reselect(self):
        """After the sweep (s12 picks 3e-6), s13/s14 at 3e-6 are worse than 1e-5's: the selection stays on s12."""
        runs = [fake("A2", "c12", 1e-5, s, a) for s, a in ((12, 0.70), (13, 0.80), (14, 0.80))]
        runs += [fake("A2", "c12lr3", 3e-6, s, a) for s, a in ((12, 0.75), (13, 0.60), (14, 0.60))]
        _, sel = self.select(*runs)
        self.assertEqual((sel["A2-c12"]["sweep_seeds"], sel["A2-c12"]["chosen"]), ([12], 3e-6))
        g = fake_groups(*runs)
        methods, _ = lr.lr_methods(g)
        s3 = lr.select_lr(g, methods, "kk", None, 200, 50, sweep=(12, 13, 14))["A2-c12"]
        self.assertEqual(s3["chosen"], 1e-5)                             # a 3-seed sweep would choose differently
        s0 = lr.select_lr(g, methods, "kk", None, 200, 50, sweep=(99,))["A2-c12"]
        self.assertEqual(s0["sweep_seeds"], [12, 13, 14])                # no sweep seed: common seeds, flagged
        self.assertIn("no sweep seed (s99)", s0["notes"][0])

    def test_ties(self):
        _, sel = self.select(fake("G", "c4", 1e-5, 12, 0.8, nll=1.0), fake("G", "c4lr3", 3e-6, 12, 0.8, nll=0.9))
        self.assertEqual(sel["G-c4"]["chosen"], 3e-6)                    # tie on accuracy: lower NLL
        _, sel = self.select(fake("G", "c4", 1e-5, 12, 0.8), fake("G", "c4lr3", 3e-6, 12, 0.8))
        self.assertEqual(sel["G-c4"]["chosen"], 1e-5)                    # and NLL: nearest 1e-5
        _, sel = self.select(fake("G", "c4lr1", 1e-6, 12, 0.8), fake("G", "c4lr3", 3e-6, 12, 0.8))
        self.assertEqual(sel["G-c4"]["chosen"], 3e-6)
        self.assertIn("tie", sel["G-c4"]["notes"][0])

    def test_no_common_seed_and_missing_key(self):
        _, sel = self.select(fake("G", "c4", 1e-5, 13, 0.8), fake("G", "c4lr3", 3e-6, 12, 0.7),
                             fake("T", "c4", 1e-5, 12, None))
        self.assertEqual(sel["G-c4"]["chosen"], 1e-5)
        self.assertIn("no seed at every LR", sel["G-c4"]["notes"][0])
        self.assertIsNone(sel["T-c4"]["chosen"])

    def test_methods_keep_other_tags(self):
        methods, _ = self.select(fake("G3", "c4kk", 1e-5, 12, 0.7), fake("G3", "c4", 1e-5, 12, 0.6),
                                 fake("G3", "c4kklr3", 3e-6, 12, 0.8))
        self.assertEqual({m: list(v["lrs"].values()) for m, v in methods.items()},
                         {"G3-c4kk": ["G3-c4kklr3", "G3-c4kk"], "G3-c4": ["G3-c4"]})


class TestLRSweepReadout(unittest.TestCase):
    """K&K: A2 sweep 1e-6 (tag fallback) / 3e-6 / 1e-5, GFPS 3e-6 / 1e-5 (valkk tie at s12 -> lower NLL: 3e-6), TFL
    without valkk. Numeric: A2's own val beats its K&K LR strongly (1e-5 kept), GFPS's not (K&K 3e-6). Pronto: A2
    and GFPS at 1e-5 only."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.dd = data_dir(root)
        cls.pull, cls.pq = root / "pull", root / "pq"
        R = f"{M}-student_v3sl"
        lr_fields(kk_run(cls.pull, f"A2-c12-{R}-s12", 0.4), 1e-5, valkk=0.70)
        lr_fields(kk_run(cls.pull, f"A2-c12-{R}-s13", 0.4), 1e-5, valkk=0.80)
        d = lr_fields(kk_run(cls.pull, f"A2-c12lr3-{R}-s12", 0.5), 3e-6, valkk=0.75)
        traj = json.loads((d / "trajectory.json").read_text())        # a forgetting curve: bbh500 .58 -> .48
        for i, p in enumerate(traj):
            p["bbh500"] = {"acc": 0.58 - 0.02 * i}
        (d / "trajectory.json").write_text(json.dumps(traj))
        lr_fields(kk_run(cls.pull, f"A2-c12lr1-{R}-s12", 0.3), None, valkk=0.60)
        lr_fields(kk_run(cls.pull, f"GFPS-c4-{R}-s12", 0.7), 1e-5, valkk=0.80, nll=1.0)
        lr_fields(kk_run(cls.pull, f"GFPS-c4-{R}-s13", 0.7), 1e-5, valkk=0.90)
        lr_fields(kk_run(cls.pull, f"GFPS-c4lr3-{R}-s12", 0.6), 3e-6, valkk=0.80, nll=0.9)
        kk_run(cls.pull, f"TFL-c4-{R}-s12", 1.0)
        N = f"{M}-student_numgen_tf"
        lr_fields(ng_run(cls.pull, f"A2-c12-{N}-s12", 0.5), 1e-5, val=0.90, n=200)
        lr_fields(ng_run(cls.pull, f"A2-c12lr3-{N}-s12", 0.25), 3e-6, val=0.80, n=200)
        lr_fields(ng_run(cls.pull, f"GFPS-c4-{N}-s12", 0.5), 1e-5, val=0.82, n=200)
        lr_fields(ng_run(cls.pull, f"GFPS-c4lr3-{N}-s12", 0.75), 3e-6, val=0.80, n=200)
        pq_run(cls.pq, f"A2-c12-{M}-student_pronto_tf-s12", 0.5, 0.5)
        pq_run(cls.pq, f"GFPS-c4-{M}-student_pronto_tf-s12", 0.75, 1.0)
        cls.args = ["--data-dir", str(cls.dd), "--boot", "300", "--boot-seed", "100"]
        with redirect_stdout(StringIO()) as buf:
            cls.res = lr.main([str(cls.pull), str(cls.pq)] + cls.args)
        cls.md = buf.getvalue()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_kk_selection_and_curve(self):
        sw = self.res["sections"]["kk"]["lr_sweep"]
        self.assertEqual(sw["methods"]["A2-c12"]["lrs"], {"1e-6": "A2-c12lr1", "3e-6": "A2-c12lr3", "1e-5": "A2-c12"})
        self.assertEqual({m: s["chosen"] for m, s in sw["selection"].items()},
                         {"A2-c12": 3e-6, "GFPS-c4": 3e-6, "TFL-c4": None})
        self.assertEqual(self.res["sections"]["kk"]["runs"]["A2-c12lr1"][12]["lr_src"], "tag")
        self.assertIn("| A2-c12 | 3e-6 * | A2-c12lr3 | s12 | 0.750 |", self.md)
        self.assertIn("| A2-c12 | 1e-5 | A2-c12 | s13 | 0.800 |", self.md)
        self.assertIn("| **GFPS-c4** | 1e-5 | GFPS-c4 | mean n=2 | 0.850 |", self.md)

    def test_kk_best_vs_best(self):
        b = self.res["sections"]["kk"]["lr_sweep"]["best_vs_best"]
        self.assertEqual((b["arm_label"], b["ref_label"]), ("GFPS-c4lr3", "A2-c12lr3"))
        e = b["endpoints"]["kk11-20"]
        self.assertAlmostEqual(e["best"]["mean"], 0.1)                   # .6 - .5 at the selected LRs
        self.assertEqual(e["best"]["seeds"], [12])
        self.assertAlmostEqual(e["default"]["mean"], 0.3)                # .7 - .4 at 1e-5, same seed s12
        self.assertEqual(e["default"]["seeds"], [12])
        self.assertAlmostEqual(b["endpoints"]["bbh"]["best"]["mean"], 0.0)
        self.assertEqual(b["endpoints"]["td"]["best"]["units"], 40)
        self.assertIn("### Best-vs-best: GFPS-c4 @3e-6 - A2-c12 @3e-6", self.md)

    def test_forgetting_curve(self):
        self.assertIn("| A2-c12 | 3e-6 | s12 | bbh500 | 0.580 @0.50 | 0.560 @1.00 | 0.520 @2.00 | 0.500 @2.50 | "
                      "0.480 @3.00 | 0.480 @3.00 | 6 |", self.md)
        self.assertEqual(self.md.count("| bbh500 |"), 1)                 # only the run that has the key

    def test_bbh_units_are_macro(self):
        """Tasks of unequal size: the item mean of the weighted units is the macro mean of set_acc."""
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            write(d / "preds_bbh.jsonl", [pred(f"a{j}", "bbh/a", True) for j in range(3)]
                  + [pred(f"b{j}", "bbh/b", j == 0) for j in range(5)])
            u = lr.item_units(d, "bbh", macro=True)
            self.assertAlmostEqual(sum(u.values()) / len(u), lr.set_acc(d, {}, "bbh"))
            self.assertAlmostEqual(lr.set_acc(d, {}, "bbh"), (1.0 + 0.2) / 2)

    def test_numgen_reuses_kk_lr_unless_strong(self):
        sel = self.res["sections"]["numgen"]["lr_sweep"]["selection"]
        a2, g = sel["A2-c12"], sel["GFPS-c4"]
        self.assertEqual((a2["own"], a2["kk_lr"], a2["chosen"]), (1e-5, 3e-6, 1e-5))
        self.assertAlmostEqual(a2["overrule"]["mean"], 0.1)
        self.assertGreater(a2["overrule"]["ci"][0], 0)
        self.assertEqual((g["own"], g["kk_lr"], g["chosen"]), (1e-5, 3e-6, 3e-6))   # .02 < .03: K&K LR kept
        b = self.res["sections"]["numgen"]["lr_sweep"]["best_vs_best"]
        self.assertEqual((b["arm_label"], b["ref_label"]), ("GFPS-c4lr3", "A2-c12"))
        self.assertAlmostEqual(b["endpoints"]["ng_beyond"]["best"]["mean"], 0.25)
        self.assertAlmostEqual(b["endpoints"]["ng_beyond"]["default"]["mean"], 0.0)

    def test_pronto(self):
        sec = self.res["sections"]["pronto"]
        g = sec["runs"]["GFPS-c4"][12]
        self.assertEqual((g["pq_in"], g["pq_beyond"], g["pq_wide"], g["pq_drop"]), (1.0, 0.75, 1.0, -0.25))
        self.assertEqual(g["pq_names"]["wide"], "pq_wide")
        self.assertEqual(g["node_join"], {"data": 20, "local": 0, "none": 0})
        self.assertEqual(g["node_bins"]["4-7"], 0.5)                     # nodes 4, 5 of 4..7 correct
        cons = {(c["arm"], c["vs"]): c for c in sec["contrasts"]}
        c = cons[("GFPS-c4", "A2-c12")]["endpoints"]
        self.assertAlmostEqual(c["pq_beyond"]["mean"], 0.25)
        self.assertAlmostEqual(c["pq_wide"]["mean"], 0.5)
        sel = sec["lr_sweep"]["selection"]["A2-c12"]
        self.assertEqual(sel["chosen"], 1e-5)
        self.assertIn("K&K LR 3e-6 not run here", sel["notes"][-1])
        self.assertIn("## PrOntoQA-OOD (student_pronto_tf)", self.md)
        self.assertIn("| arm | seed | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | wide | in hops 1-4 | beyond hops 5-8 |",
                      self.md)

    def assert_legacy(self, old_md: str, new_md: str):
        """Every section of the old readout is a byte-identical prefix of the new one; the only new section is the
        pronto 'no runs' line."""
        old, new = sections(old_md), dict(sections(new_md))
        self.assertEqual(old[0][1], new[""])
        for head, text in old[1:]:
            self.assertTrue(new[head].startswith(text), head)
            self.assertTrue(new[head][len(text):].lstrip("\n").startswith("### LR sweep") or "no runs" in head, head)
        self.assertEqual(set(new) - {h for h, _ in old}, {"## PrOntoQA-OOD: no runs"})

    def test_legacy_tables_unchanged(self):
        with tempfile.TemporaryDirectory() as t:
            base = load_base(Path(t))
            if base is None:
                self.skipTest(f"git commit {BASE} not readable")
            argv = [str(self.pull)] + self.args
            self.assert_legacy(run_md(base, argv), run_md(lr, argv))

    @unittest.skipUnless(L4B.is_dir(), "results/hpc/l4b absent")
    def test_legacy_tables_unchanged_current_runs(self):
        """The real 4B runs (pronto runs left out: the base commit read them as K&K)."""
        with tempfile.TemporaryDirectory() as t:
            base = load_base(Path(t))
            if base is None:
                self.skipTest(f"git commit {BASE} not readable")
            dirs = sorted(str(d) for d in L4B.iterdir() if d.is_dir() and "pronto" not in d.name)
            argv = dirs + ["--data-dir", str(ROOT / "data"), "--boot", "300", "--boot-seed", "100"]
            self.assert_legacy(run_md(base, argv), run_md(lr, argv))

if __name__ == "__main__":
    unittest.main()
