"""scripts/build_fa_lofo.py: the LOFO fold files. Synthetic pools (drops, A2 / tree alignment, interleave, the
StepFinder cap by question group, determinism, refusals) and, when the real build exists, the real fold files (md5s
reproduce, no held-out uid, AgenTracer Diagnosis rows only, tree rows = A2 rows outside the node fields). CPU only:

  ~/.venvs/sev/bin/python -m unittest tests.test_build_fa_lofo
"""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_fa_lofo", ROOT / "scripts/build_fa_lofo.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


def row(pool, split, k, task, nodes=2, tree=True):
    u = "%s-%s-%04d" % (pool, split, k)
    a2 = {"item_id": "fasf-%s-%s-%05d" % (pool, split, k), "source": "fa_public", "prompt": "Task: %s\n\nStep 0 [A]: x"
          % task + " y" * (k % 7), "labels": ["A", "B"], "gold_label": "A",
          "subqs": [{"id": "p1", "question": "label node?", "truth": True}], "has_tree": True,
          "tree_source": "fa_public_labels", "fa": {"uid": u, "origin": pool}}
    t = dict(a2)
    t["subqs"] = [{"id": "n%d" % j, "type": ["content", "clean"][j % 2], "question": "Which step %d?" % j,
                   "options": ["one", "two", "three"], "answer": j % 3} for j in range(nodes)] if tree else []
    t["has_tree"], t["tree_source"] = tree, "fa_sf_trees"
    t["fa"] = dict(a2["fa"], tree_teacher="claude-opus" if tree else None)
    return a2, t


def make_root(d: Path, sizes=None, bad=None):
    """Pools with sizes {pool: (train, val)}; StepFinder pools have 4 rows per task (question groups)."""
    sizes = sizes or {"sf_alg": (24, 4), "sf_hc": (40, 8), "aftraj": (10, 2), "real": (6, 2), "agtr": (3, 1),
                      "wwpro": (12, 2), "ce": (8, 2)}
    for p, (a2d, t2f, _) in B.POOLS.items():
        ntr, nv = sizes[p]
        for split, n in (("train", ntr), ("val", nv)):
            rows = [row(p, split, k, "%s question %d" % (p, k // 4 if p.startswith("sf") else k)) for k in range(n)]
            if bad == "order" and p == "wwpro" and split == "train":
                rows[0], rows[1] = (rows[0][0], rows[1][1]), (rows[1][0], rows[0][1])
            if bad == "mc" and p == "ce" and split == "val":
                rows[0][1]["subqs"][0]["options"] = ["same", "same"]
            if bad == "field" and p == "real" and split == "train":
                rows[2][1]["prompt"] += " changed"
            for dd, i in ((a2d, 0), (t2f, 1)):
                f = d / dd / ("%s.jsonl" % split)
                f.parent.mkdir(parents=True, exist_ok=True)
                with open(f, "a") as fh:
                    for r in rows:
                        fh.write(json.dumps(r[i]) + "\n")
    plan = {"folds": {}, "time_model": {"coef_s": {"per_item": 0.07, "per_1k_tokens": 0.04, "per_1M_token2": 0.03}}}
    own = {"whowhen": ["sf_alg", "sf_hc"], "aftraj": ["aftraj"], "wwpro": ["wwpro"]}
    for fold, o in own.items():
        pools = [p for p in B.POOLS if p not in o]
        ent = {"held_out_tests": ["data/eval/x.jsonl"], "train_pools": pools, "pools": {}}
        for p in pools:
            drop = ["%s-train-0001" % p, "%s-val-0000" % p] if p == "aftraj" else []
            ntr, nv = sizes[p]
            ent["pools"][p] = {"train": ntr - sum(1 for x in drop if "-train-" in x),
                               "val": nv - sum(1 for x in drop if "-val-" in x), "dropped_overlap_with_held_out": drop}
        plan["folds"][fold] = ent
    (d / "data/student_fa_lofo").mkdir(parents=True, exist_ok=True)
    (d / B.PLAN).write_text(json.dumps(plan))
    return d


def lines(p: Path):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


class Synthetic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = make_root(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, **kw):
        a = ["--root", str(self.d)] + sum((["--" + k.replace("_", "-"), v] for k, v in kw.items()), [])
        return B.main(a)

    def test_folds_drops_alignment(self):
        out = self.build()
        self.assertEqual(sorted(out["folds"]), ["aftraj", "aftraj_cap", "whowhen", "wwpro", "wwpro_cap"])
        ww = self.d / "data/student_fa_lofo_whowhen_a2/train.jsonl"
        a2, tf = lines(ww), lines(self.d / "data/student_fa_lofo_whowhen_tf/train.jsonl")
        self.assertFalse(any(r["fa"]["origin"].startswith("sf") for r in a2))           # own pools left out
        self.assertNotIn("aftraj-train-0001", [r["fa"]["uid"] for r in a2])             # the plan's drops
        self.assertEqual(len(a2), 10 - 1 + 6 + 3 + 12 + 8)
        self.assertEqual([r["fa"]["uid"] for r in a2], [r["fa"]["uid"] for r in tf])    # same rows, same order
        for x, y in zip(a2, tf):
            self.assertEqual(B.strip_nodes(x), B.strip_nodes(y))
            self.assertNotIn("options", x["subqs"][0])
            self.assertIn("options", y["subqs"][0])
        self.assertEqual([r["fa"]["origin"] for r in a2[:5]], ["aftraj", "real", "agtr", "wwpro", "ce"])  # interleave
        v = lines(self.d / "data/student_fa_lofo_whowhen_a2/val.jsonl")
        self.assertNotIn("aftraj-val-0000", [r["fa"]["uid"] for r in v])
        # raw bytes: every output line is a pool line verbatim
        pool_lines = set((self.d / "data/student_fa_aftraj_t3/train.jsonl").read_text().splitlines())
        out_lines = (self.d / "data/student_fa_lofo_whowhen_tf/train.jsonl").read_text().splitlines()
        self.assertTrue({x for x in out_lines if '"aftraj"' in x} <= pool_lines)
        md5 = (self.d / "data/student_fa_lofo_whowhen_tf/md5sums.txt").read_text().split()
        self.assertEqual(md5[0], hashlib.md5((self.d / "data/student_fa_lofo_whowhen_tf/train.jsonl").read_bytes()).hexdigest())

    def test_cap_largest_by_question(self):
        out = self.build()
        s = out["folds"]["aftraj_cap"]
        others = {p: s["pools"][p]["train"] for p in s["pools"] if p not in B.SF}
        sf = s["pools"]["sf_alg"]["train"] + s["pools"]["sf_hc"]["train"]
        self.assertEqual(sf, max(others.values()))                                     # wwpro 12
        self.assertEqual((s["pools"]["sf_alg"]["train"], s["pools"]["sf_hc"]["train"]), (4, 8))   # 24:40 of 12
        rows = lines(self.d / "data/student_fa_lofo_aftraj_cap_a2/train.jsonl")
        per_q = Counter(B.question_key(r) for r in rows if r["fa"]["origin"] == "sf_hc")
        self.assertEqual(len(per_q), 8)                                                 # one per question (10 q)
        self.assertLessEqual(max(per_q.values()) - min(per_q.values()), 1)
        self.assertEqual(s["pools"]["sf_hc"]["val"], round(8 * 8 / 40))                  # val: same kept fraction
        main = out["folds"]["aftraj"]
        self.assertEqual(main["pools"]["sf_hc"]["train"], 40)                           # main wave: not capped
        self.assertNotIn("whowhen_cap", out["folds"])

    def test_cap_frac_and_determinism(self):
        out = self.build(cap="frac:0.3", folds="aftraj")
        s = out["folds"]["aftraj_cap"]
        n_other = sum(v["train"] for p, v in s["pools"].items() if p not in B.SF)
        self.assertEqual(s["pools"]["sf_alg"]["train"] + s["pools"]["sf_hc"]["train"], round(0.3 / 0.7 * n_other))
        m1 = {k: v["md5"] for k, v in out["files"].items()}
        m2 = {k: v["md5"] for k, v in self.build(cap="frac:0.3", folds="aftraj")["files"].items()}
        self.assertEqual(m1, m2)
        man = self.d / "data/student_fa_lofo/build_manifest.json"
        self.assertEqual(B.main(["--root", str(self.d), "--cap", "frac:0.3", "--folds", "aftraj", "--check", str(man)])
                         ["files"].keys(), out["files"].keys())

    def test_hours_positive(self):
        out = self.build(folds="aftraj", variants="main")
        h = out["folds"]["aftraj"]["hours"]
        self.assertGreater(h["slice_h_tree_3ep"], h["slice_h_a2_6ep"] - 1)
        self.assertGreater(h["a2_s_per_epoch"], 0)


class Refusals(unittest.TestCase):
    def check(self, bad, msg):
        with tempfile.TemporaryDirectory() as t:
            d = make_root(Path(t), bad=bad)
            with self.assertRaises((SystemExit, ValueError)) as cm:
                B.main(["--root", str(d)])
            self.assertIn(msg, str(cm.exception))

    def test_order(self):
        self.check("order", "out of order")

    def test_mc(self):
        self.check("mc", "MC node")

    def test_field(self):
        self.check("field", "differs from its A2 row")


MAN = ROOT / "data/student_fa_lofo/build_manifest.json"


@unittest.skipUnless(MAN.exists(), "no real LOFO build")
class RealBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.man = json.loads(MAN.read_text())
        cls.plan = json.loads((ROOT / B.PLAN).read_text())

    def test_md5_on_disk(self):
        for path, ent in self.man["files"].items():
            self.assertEqual(hashlib.md5((ROOT / path).read_bytes()).hexdigest(), ent["md5"], path)

    def test_held_out_and_agtr(self):
        diag = {json.loads(x)["fa"]["uid"] for s in ("train", "val")
                for x in (ROOT / ("data/student_fa_agtr_diag/%s.jsonl" % s)).read_text().splitlines() if x.strip()}
        for tag, s in self.man["folds"].items():
            fold = s["fold"]
            drops = {u for p in self.plan["folds"][fold]["pools"].values() for u in p["dropped_overlap_with_held_out"]}
            for split in ("train", "val"):
                a2 = lines(ROOT / ("data/student_fa_lofo_%s_a2/%s.jsonl" % (tag, split)))
                tf = lines(ROOT / ("data/student_fa_lofo_%s_tf/%s.jsonl" % (tag, split)))
                self.assertEqual(len(a2), len(tf))
                us = [r["fa"]["uid"] for r in a2]
                self.assertFalse(set(us) & drops, tag)
                self.assertEqual(us, [r["fa"]["uid"] for r in tf])
                for x, y in zip(a2, tf):
                    self.assertEqual(B.strip_nodes(x), B.strip_nodes(y))
                ag = {r["fa"]["uid"] for r in a2 if r["fa"]["origin"] == "agentracer"}
                self.assertTrue(ag <= diag, tag)
                own = {"whowhen": "stepfinder", "aftraj": "aftraj", "agentracer": "agentracer", "wwpro": "wwpro",
                       "correct": "correct"}.get(fold)
                self.assertFalse(any(r["fa"]["origin"] == own for r in a2), tag)
            if s["variant"] == "cap":
                self.assertLessEqual(s["sf_share_train"], 0.35)


if __name__ == "__main__":
    unittest.main()
