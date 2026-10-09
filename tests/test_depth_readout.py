import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import depth_readout as dr  # noqa: E402

M = "Qwen3.5-4B"
HAVE_SOLVER = dr.SOLVER.exists()


def pred(item_id, group, ok, n=5):
    probs = [0.1] * n
    probs[0 if ok else 1] = 0.6
    return {"item_id": item_id, "group": group, "labels": list("ABCDEF"[:n]), "probs": probs, "gold_label": "A"}


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def chain_program(d, fan):
    """d operations; with fan the last one adds a stated count (fan-in): level stays d, one more stated leaf."""
    qs = [{"id": "q1", "stated": True, "inputs": []}]
    for k in range(2, d + 2):
        qs.append({"id": f"q{k}", "stated": False, "inputs": [f"q{k - 1}"]})
    if fan:
        qs.insert(0, {"id": "s0", "stated": True, "inputs": []})
        qs[-1]["inputs"] = qs[-1]["inputs"] + ["s0"]
    return {"answer": qs[-1]["id"], "quantities": qs}


def p_ok(arm, d, j, n):
    """Accuracy falls 0.9 -> 0.2 over depths 1..8; the arm's curve is the ref's shifted two levels to the right."""
    acc = {"ref": lambda x: max(0.2, 1.0 - 0.1 * x), "arm": lambda x: max(0.2, 1.0 - 0.1 * (x - 2))}[arm](d)
    return j < round(acc * n)


class Numerics(unittest.TestCase):
    def test_pav_and_crossing(self):
        np.testing.assert_allclose(dr.pav_dec([0.8, 0.6, 0.7, 0.5], [1, 1, 1, 1]), [0.8, 0.65, 0.65, 0.5])
        np.testing.assert_allclose(dr.pav_dec([0.5, 0.9], [3, 1]), [0.6, 0.6])
        self.assertEqual(dr.crossing([4, 5, 6], [0.8, 0.6, 0.4], 0.5), (5.5, "ok"))
        self.assertEqual(dr.crossing([4, 5, 6], [0.8, 0.6, 0.4], 0.9), (4.0, "low"))
        self.assertEqual(dr.crossing([4, 5, 6], [0.8, 0.6, 0.4], 0.3), (6.0, "high"))

    def test_irls_recovers_coefficients(self):
        rng = np.random.default_rng(0)
        X = np.stack([np.ones(40000), rng.normal(size=40000), rng.integers(0, 2, 40000)], axis=1)
        b = np.array([0.3, -0.8, 0.5])
        y = (rng.random(40000) < 1 / (1 + np.exp(-X @ b))).astype(float)
        est, ok = dr.irls(X, y)
        self.assertTrue(ok)
        np.testing.assert_allclose(est, b, atol=0.06)

    def test_kstar(self):
        ref = [0.9, 0.8, 0.7, 0.6, 0.5]
        self.assertEqual(dr.kstar([1, 2, 3, 4, 5], [0.9, 0.9, 0.9, 0.8, 0.7], ref), 2)
        self.assertIsNone(dr.kstar([1, 2, 3], [0.8, 0.8, 0.6], [0.9, 0.8, 0.7]))
        self.assertEqual(dr.kstar([1, 2, 3, 4, 5], [0.9, 0.9, 0.9, 0.8, 0.7], ref, only_from=4), 2)

    def test_ng_complexity(self):
        items = [{"item_id": "a", "depth": 3, "program": chain_program(3, False)},
                 {"item_id": "b", "depth": 3, "program": chain_program(3, True)}]
        c = dr.ng_complexity(items)
        self.assertEqual((c["a"]["size"], c["a"]["level"], c["a"]["leaves"]), (3, 3, 1))
        self.assertEqual((c["b"]["size"], c["b"]["level"], c["b"]["leaves"], c["b"]["fanin"]), (3, 3, 2, 1))

    def test_interaction_sign(self):
        """Ref accuracy falls steeply with c, the arm's less steeply: arm:c > 0 with a CI above 0; equal slopes: CI
        covers 0."""
        rng = np.random.default_rng(1)
        n = 3000
        comp = rng.normal(size=n)
        items = [f"i{k}" for k in range(n)]

        def groups(slope_arm):
            g = {}
            for lab, a, b in (("ref", 0.0, -1.0), ("arm", 0.4, slope_arm)):
                g[lab] = {}
                for s in (12, 13):
                    y = rng.random(n) < 1 / (1 + np.exp(-(a + b * comp)))
                    g[lab][s] = {"_u": {"x": dict(zip(items, y.astype(int).tolist()))}}
            return g
        strata = np.zeros(n, dtype=int)
        counts = dr.strata_draws(strata, 200, np.random.default_rng(2))
        res = dr.fit_interaction(dr.Pair(groups(-0.5), "arm", "ref", "x", items), comp, counts)
        self.assertGreater(res["ci"][0], 0)
        self.assertAlmostEqual(res["beta_int"], 0.5, delta=0.15)
        res0 = dr.fit_interaction(dr.Pair(groups(-1.0), "arm", "ref", "x", items), comp, counts)
        self.assertLess(res0["ci"][0], 0)
        self.assertGreater(res0["ci"][1], 0)


class EndToEnd(unittest.TestCase):
    def test_numgen_shift_and_models(self):
        """Arm curve = ref curve shifted two depths right: horizontal shift 2 at every threshold the two curves cross
        in range, A2-equivalent gain 2, k* 2."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ev = tmp / "eval"
            n = 40
            for d in range(1, 9):
                write(ev / f"numgen_d{d}.jsonl", [{"item_id": f"numgen-d{d}-{j:05d}", "depth": d, "fanin": j % 2,
                                                   "program": chain_program(d, j % 2 == 1)} for j in range(n)])
            runs = tmp / "runs"
            for arm, lab in (("ref", "A2-c12"), ("arm", "GFPS-c4")):
                for s in (12, 13):
                    d0 = runs / f"{lab}-{M}-student_numgen_tf-s{s}"
                    d0.mkdir(parents=True)
                    (d0 / "metrics.json").write_text(json.dumps({"args": {"model": f"/m/{M}"}, "eval": {}}))
                    for d in range(1, 9):
                        write(d0 / f"preds_ng_d{d}.jsonl",
                              [pred(f"numgen-d{d}-{j:05d}", f"numgen/d{d}", p_ok(arm, d, j, n)) for j in range(n)])
            with redirect_stdout(StringIO()):
                out = dr.main([str(runs), "--eval-dir", str(ev), "--boot", "50", "--boot-reg", "20",
                               "--boot-other", "10", "--out", str(tmp / "o" / "x")])
            self.assertTrue((tmp / "o" / "x.md").exists() and (tmp / "o" / "x.json").exists())
            sec = out["sections"]["numgen"]
            sh = sec["shift"]["pooled"]
            for th in ("0.50", "0.45"):
                self.assertAlmostEqual(sh[th]["shift"], 2.0, places=6)
            self.assertAlmostEqual(sh["equiv"][5]["gain"], 2.0, places=6)
            self.assertEqual(sh["kstar"]["iso_all"], 2)
            self.assertIn("size = operations = nominal depth (baseline)", sec["interaction"])
            self.assertEqual(sec["complexity_summary"]["per_depth"][3]["size_eq_depth"], 1.0)

    @unittest.skipUnless(HAVE_SOLVER, "K&K solver (teacher_cache) absent")
    def test_kk_complexity_generated_items(self):
        from build_items import to_eval
        from cotdistill import sources as S
        items = [to_eval(r, "kk") for r in S.knights_knaves(6, seed=777, people_range=(4, 6))]
        c = dr.kk_complexity(items)
        for it in items:
            v = c[it["item_id"]]
            self.assertEqual(v["people"], it["depth"])
            self.assertGreaterEqual(v["depth"], 1)
            self.assertGreaterEqual(v["width"], v["people"])     # at least one derived role node per inhabitant
            self.assertGreaterEqual(v["case"], 0)

    @unittest.skipUnless(HAVE_SOLVER and (ROOT / "data/student_v3sl/val.jsonl").exists(), "solver or v3sl val absent")
    def test_reconstruction_matches_stored_trees(self):
        vals = dr.stored_tree_stats(ROOT / "data/student_v3sl/val.jsonl", limit=15)
        c = dr.kk_complexity(vals)
        for v in vals:
            self.assertEqual((c[v["item_id"]]["depth"], c[v["item_id"]]["width"]), (v["depth"], v["width"]))


if __name__ == "__main__":
    unittest.main()
