"""Tests for scripts/paired_stats.py on synthetic data where the truth is known."""
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import paired_stats as ps  # noqa: E402

LABELS = ["A", "B", "C", "D"]


def pred_row(item_id: str, correct: bool, conf: float = 0.7) -> dict:
    """Gold is always A (fixed per item, as in real preds); the top option is A when correct, else B."""
    rest = (1.0 - conf) / 3
    probs = [conf, rest, rest, rest] if correct else [rest, conf, rest, rest]
    return {"item_id": item_id, "labels": LABELS, "probs": probs, "gold_label": "A"}


def make_run(root: Path, name: str, seed: int, epochs: float, sets: dict, metrics_eval: dict | None = None) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"args": {"seed": seed, "epochs": epochs},
                                                "eval": metrics_eval or {}}))
    for s, rows in sets.items():
        if isinstance(rows, (bytes, str)):
            (d / f"preds_{s}.jsonl").write_bytes(rows if isinstance(rows, bytes) else rows.encode())
        else:
            (d / f"preds_{s}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def rows_from(prefix: str, correct) -> list[dict]:
    return [pred_row(f"{prefix}-{i}", bool(c)) for i, c in enumerate(correct)]


def simulate(rng, k, n, effect=0.0, seed_sd=0.15, theta=None):
    """Correctness (k, n) for two arms sharing item difficulty (items correlated across seeds and arms)."""
    theta = rng.normal(0.8, 1.2, n) if theta is None else theta
    ua = rng.normal(0, seed_sd, (k, 1))
    ub = rng.normal(0, seed_sd, (k, 1))
    pa = 1 / (1 + np.exp(-(theta + effect + ua)))
    pb = 1 / (1 + np.exp(-(theta + ub)))
    return (rng.random((k, n)) < pa).astype(np.int8), (rng.random((k, n)) < pb).astype(np.int8)


class TestStudentT(unittest.TestCase):
    def test_quantiles(self):
        for df, q in [(1, 12.7062), (2, 4.3027), (4, 2.7764), (30, 2.0423)]:
            self.assertAlmostEqual(ps.t_ppf(0.975, df), q, places=3)

    def test_p_values(self):
        self.assertAlmostEqual(ps.t_two_sided_p(1.0, 1), 0.5, places=6)
        self.assertAlmostEqual(ps.t_two_sided_p(2.0, 10), 0.07339, places=4)
        self.assertAlmostEqual(ps.t_two_sided_p(0.0, 3), 1.0, places=9)

    def test_k1_has_no_interval(self):
        t = ps.seed_t([0.05])
        self.assertIsNone(t["lo"])
        self.assertIn("k=1", t["note"])
        rng = np.random.default_rng(0)
        a, b = simulate(rng, 1, 2000, effect=1.5)          # huge effect, one seed
        st = ps.paired_stats(a, b, n_boot=500)
        self.assertGreater(st["two_level"][0], 0)          # bootstrap alone would say "excludes 0"
        self.assertFalse(st["significant"])                 # ... but k=1 never qualifies

    def test_k2_interval_is_wide(self):
        t = ps.seed_t([0.01, 0.03])
        self.assertAlmostEqual(t["hi"] - t["mean"], 12.7062 * 0.02 / np.sqrt(2) / np.sqrt(2), places=4)
        self.assertTrue(t["lo"] < 0 < t["hi"])
        rng = np.random.default_rng(1)
        a, b = simulate(rng, 2, 2500, effect=0.12, seed_sd=0.05)
        st = ps.paired_stats(a, b, n_boot=1000)
        tw = st["t"]["hi"] - st["t"]["lo"]
        bw = st["two_level"][1] - st["two_level"][0]
        self.assertGreater(tw, 2.5 * bw, (tw, bw))            # df=1: t_.975 = 12.7


class TestCoverage(unittest.TestCase):
    def test_zero_effect_covers_zero_at_nominal_rate(self):
        rng = np.random.default_rng(20261020)
        reps, k, n = 160, 5, 300
        cov_t = cov_2l = cov_item = sig = 0
        for r in range(reps):
            a, b = simulate(rng, k, n)
            st = ps.paired_stats(a, b, n_boot=400, boot_seed=r)
            cov_t += st["t"]["lo"] <= 0 <= st["t"]["hi"]
            cov_2l += st["two_level"][0] <= 0 <= st["two_level"][1]
            cov_item += st["item_boot"][0] <= 0 <= st["item_boot"][1]
            sig += st["significant"]
        self.assertGreaterEqual(cov_t / reps, 0.90)
        self.assertLessEqual(cov_t / reps, 0.995)
        self.assertGreaterEqual(cov_2l / reps, 0.88)
        self.assertLessEqual(sig / reps, 0.06)
        self.assertLess(cov_item, cov_2l)          # item-only ignores seed variance -> undercovers (why 2-level is primary)

    def test_real_effect_is_detected(self):
        rng = np.random.default_rng(7)
        a, b = simulate(rng, 5, 2500, effect=0.35, seed_sd=0.05)
        st = ps.paired_stats(a, b, n_boot=1000)
        self.assertTrue(st["significant"], st)


class TestSharedIndices(unittest.TestCase):
    def width(self, D, independent):
        lo, hi, _ = ps.item_bootstrap(D, 3000, np.random.default_rng(3), independent=independent)
        return hi - lo

    def test_shared_vs_independent_when_items_correlated(self):
        rng = np.random.default_rng(11)
        n, k = 600, 4
        d_item = rng.choice([-1, 0, 1], size=n, p=[0.2, 0.6, 0.2]).astype(np.int8)
        D = np.tile(d_item, (k, 1))                         # each item favours the same arm in every seed
        ratio = self.width(D, False) / self.width(D, True)
        self.assertGreater(ratio, 1.6, ratio)               # ~sqrt(k) = 2
        # realistic: per-item arm effect + shared difficulty (positively correlated across seeds)
        theta = rng.normal(0.8, 1.2, n)
        gap = rng.normal(0, 2.0, n)
        pa, pb = 1 / (1 + np.exp(-(theta + gap))), 1 / (1 + np.exp(-theta))
        D2 = ((rng.random((k, n)) < pa).astype(np.int8) - (rng.random((k, n)) < pb).astype(np.int8))
        self.assertGreater(self.width(D2, False) / self.width(D2, True), 1.15)

    def test_no_difference_when_items_uncorrelated(self):
        rng = np.random.default_rng(12)
        D = rng.choice([-1, 0, 1], size=(4, 600), p=[0.2, 0.6, 0.2]).astype(np.int8)
        ratio = self.width(D, False) / self.width(D, True)
        self.assertTrue(0.85 < ratio < 1.18, ratio)


class RunTree(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def opt(self, **kw):
        kw.setdefault("boot", 300)
        return ps.Options(root=self.root, **kw)


class TestValSelection(RunTree):
    def build(self, test_rows_for):
        """X-c{ep}-s{seed}: ep1 is better on val, ep2 better on 'test' (when test is valid)."""
        val = {1: [1] * 70 + [0] * 30, 2: [1] * 60 + [0] * 40}
        for seed in (12, 13):
            for ep in (1, 2):
                make_run(self.root, f"X-c{ep}-s{seed}", seed, float(ep),
                         {"val": rows_from("v", val[ep]), "gsm8kv2": test_rows_for(ep, "t"),
                          "gsm1k": test_rows_for(ep, "k")},
                         metrics_eval={"gsm8kv2": {"acc": 0.5 + 0.2 * ep}, "val": {"acc": 0.0}})
            make_run(self.root, f"B-s{seed}", seed, 1.0,
                     {"val": rows_from("v", [1] * 50 + [0] * 50), "gsm8kv2": rows_from("t", [1] * 50 + [0] * 50),
                      "gsm1k": rows_from("k", [1] * 50 + [0] * 50)})

    def record(self):
        seen = []
        orig = ps.read_preds

        def hook(path):
            seen.append(Path(path).name)
            return orig(path)
        ps.read_preds = hook
        self.addCleanup(setattr, ps, "read_preds", orig)
        return seen

    def test_selection_never_reads_test_files(self):
        self.build(lambda ep, p: b"\x00\xffnot json at all{{{")   # test files are garbage
        seen = self.record()
        arm = ps.ArmSpec("X", ["X-c{ep}-s{seed}"], [1.0, 2.0])
        variants = ps.resolve_arm(arm, self.root)
        buf = []
        ep, info = ps.select_epoch("X", variants, {12, 13}, set(), buf)
        self.assertEqual(ep, 1.0)
        self.assertTrue(seen and all(n == "preds_val.jsonl" for n in seen), seen)
        self.assertIn("ep1", buf[0])

    def test_full_compare_selects_on_val_not_test(self):
        self.build(lambda ep, p: rows_from(p, [1] * (50 + 30 * (ep == 2)) + [0] * (50 - 30 * (ep == 2))))
        seen = self.record()
        a = ps.ArmSpec("X", ["X-c{ep}-s{seed}"])                 # epochs discovered
        b = ps.ArmSpec("B", ["B-s{seed}"])
        with redirect_stderr(StringIO()):
            res = ps.compare(a, b, self.opt(select_by="val", depth=False))
        self.assertEqual(res["ep_a"], 1.0)                       # val winner, although ep2 wins on test
        first_test = next(i for i, n in enumerate(seen) if n != "preds_val.jsonl")
        self.assertTrue(all(n == "preds_val.jsonl" for n in seen[:first_test]))
        self.assertNotIn("preds_val.jsonl", seen[first_test:])   # selection done before any test read
        self.assertEqual(res["seeds"], [12, 13])

    def test_tie_goes_to_fewer_epochs(self):
        for ep in (2, 4):
            make_run(self.root, f"T-c{ep}-s12", 12, float(ep), {"val": rows_from("v", [1, 0, 1, 0])})
        variants = ps.resolve_arm(ps.ArmSpec("T", ["T-c{ep}-s{seed}"]), self.root)
        ep, _ = ps.select_epoch("T", variants, None, set(), [])
        self.assertEqual(ep, 2.0)

    def test_multiple_epochs_without_selection_is_an_error(self):
        self.build(lambda ep, p: rows_from(p, [1, 0]))
        with self.assertRaises(ValueError):
            ps.compare(ps.ArmSpec("X", ["X-c{ep}-s{seed}"]), ps.ArmSpec("B", ["B-s{seed}"]), self.opt())


class TestPairingAndIO(RunTree):
    def test_only_common_seeds_and_exclusion(self):
        for s in (12, 13, 14):
            make_run(self.root, f"A-s{s}", s, 1.0, {"gsm8kv2": rows_from("t", [1, 1, 0, 1]), "gsm1k": rows_from("k", [1, 0])})
        for s in (13, 14, 15):
            make_run(self.root, f"B-s{s}", s, 1.0, {"gsm8kv2": rows_from("t", [1, 0, 0, 1]), "gsm1k": rows_from("k", [0, 0])})
        res = ps.compare(ps.ArmSpec("A", ["A-s{seed}"]), ps.ArmSpec("B", ["B-s{seed}"]), self.opt(exclude_seeds=[13]))
        self.assertEqual(res["seeds"], [14])
        self.assertEqual(res["dropped"], {"a_only": [12], "b_only": [15], "excluded": [13]})
        self.assertEqual(res["main"]["n_items"], 6)             # 4 + 2 concatenated, keyed by set
        self.assertAlmostEqual(res["main"]["mean"], 2 / 6)
        self.assertIsNone(res["main"]["t"]["lo"])
        self.assertFalse(res["main"]["significant"])
        self.assertIn("k=1", ps.text_report(res))

    def test_same_item_id_in_two_sets_is_two_items(self):
        make_run(self.root, "A-s1", 1, 1.0, {"gsm8kv2": rows_from("x", [1]), "gsm1k": rows_from("x", [0])})
        m = ps.load_matrix([ps.Run("A-s1", self.root / "A-s1", 1, 1.0)], ["gsm8kv2", "gsm1k"])
        self.assertEqual(m["keys"], ["gsm1k:x-0", "gsm8kv2:x-0"])

    def test_missing_items_error(self):
        make_run(self.root, "A-s1", 1, 1.0, {"gsm8kv2": rows_from("t", [1, 1, 0]), "gsm1k": rows_from("k", [1])})
        make_run(self.root, "B-s1", 1, 1.0, {"gsm8kv2": rows_from("t", [1, 1]), "gsm1k": rows_from("k", [1])})
        with self.assertRaises(ValueError):
            ps.compare(ps.ArmSpec("A", ["A-s{seed}"]), ps.ArmSpec("B", ["B-s{seed}"]), self.opt())

    def test_seed_mismatch_with_metrics_is_an_error(self):
        make_run(self.root, "A-s12", 13, 1.0, {})
        with self.assertRaises(ValueError):
            ps.resolve_arm(ps.ArmSpec("A", ["A-s{seed}"]), self.root)

    def test_pattern_is_strict(self):
        for n in ("GFPQ-c1lr3ch-Q-s12", "GFPQ-c4lr3chcs0-Q-s12", "GFPQ-nw0.5-c4lr3ch-Q-s12", "GFPQ-c2lr3ch-Q-s12x"):
            (self.root / n).mkdir()
        rx = ps.pattern_regex("GFPQ-c{ep}lr3ch-Q-s{seed}")
        self.assertEqual([n for n in ps._listdir(self.root) if rx.match(n)], ["GFPQ-c1lr3ch-Q-s12"])

    def test_depth_groups_and_extra_sets(self):
        sets = {"gsm8kv2": rows_from("t", [1, 0]), "gsm1k": rows_from("k", [1]), "gs2": rows_from("a", [1, 1]),
                "gs3": rows_from("b", [0]), "gs4": rows_from("c", [1]), "gs5": rows_from("d", [0, 0]),
                "gs6": rows_from("e", [1]), "gs7": rows_from("f", [0]), "gs8": rows_from("g", [0]),
                "bbh500": rows_from("h", [1, 0, 1])}
        for s in (1, 2):
            make_run(self.root, f"A-s{s}", s, 1.0, sets)
            make_run(self.root, f"B-s{s}", s, 1.0, {k: v for k, v in sets.items() if k not in ("gs8",)})
        res = ps.compare(ps.ArmSpec("A", ["A-s{seed}"]), ps.ArmSpec("B", ["B-s{seed}"]), self.opt(sets=["bbh=bbh500"]))
        self.assertEqual(res["extras"]["gs2-4"]["n_items"], 4)
        self.assertTrue(res["extras"]["gs5-8"]["missing"])      # B lacks gs8 -> flagged, not silently partial
        self.assertEqual(res["extras"]["bbh"]["n_items"], 3)

    def test_ece_matches_calibration(self):
        rows = [pred_row(f"t-{i}", i % 3 != 0, conf=0.55 + 0.04 * (i % 10)) for i in range(60)]
        make_run(self.root, "A-s1", 1, 1.0, {"gsm8kv2": rows})
        m = ps.load_matrix([ps.Run("A-s1", self.root / "A-s1", 1, 1.0)], ["gsm8kv2"])
        ref = ps.calibration([r["probs"] for r in rows], [LABELS.index(r["gold_label"]) for r in rows], bins=15)
        self.assertAlmostEqual(m["ece"][0], ref["ece"])
        self.assertAlmostEqual(m["correct"].mean(), ref["acc"])


class TestTable(RunTree):
    def test_table_markdown(self):
        rng = np.random.default_rng(5)
        for s in (12, 13, 14):
            for name, p in (("BASE", 0.6), ("ARM-c1", 0.62), ("ARM-c2", 0.65)):
                c = (rng.random(200) < p).astype(int)
                ep = 2.0 if name.endswith("c2") else 1.0
                make_run(self.root, f"{name}-s{s}", s, ep, {"gsm8kv2": rows_from("t", c[:150]), "gsm1k": rows_from("k", c[150:]),
                                                            "val": rows_from("v", c[:40])})
        spec = {"root": str(self.root), "select_by": "val", "boot": 200, "depth": False,
                "tables": [{"title": "toy", "baseline": {"name": "base", "pattern": "BASE-s{seed}"},
                            "arms": [{"name": "arm", "pattern": "ARM-c{ep}-s{seed}", "epochs": [1, 2]},
                                     {"name": "arm c2 excl s12", "pattern": "ARM-c2-s{seed}", "exclude_seeds": [12]}]}]}
        sp = self.root / "spec.json"
        sp.write_text(json.dumps(spec))
        out = self.root / "t.md"
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            ps.main(["--table", str(sp), "--out", str(out)])
        md = out.read_text()
        self.assertIn("### toy", md)
        self.assertIn("| arm |", md)
        self.assertIn("(val)", md)
        self.assertIn("select-by val", md)
        row = next(l for l in md.splitlines() if l.startswith("| arm c2 excl s12 |"))
        self.assertEqual(row.split(" | ")[2], "2")                # k = 2 after excluding s12


if __name__ == "__main__":
    unittest.main()
