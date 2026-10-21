"""Tests for scripts/analyze_batch14.py: the batch-13 seed-12 runs (smoke, README_14's quoted numbers) and synthetic
run dirs for the seed 13-15 rules (epoch rule, 3-epoch rule, C1 / C1' / C2, halves) and the TSF detector keys."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import analyze_batch14 as ab  # noqa: E402

REAL_ROOT = ROOT / "results/chtc_repo/runs"
LABELS = ["A", "B", "C", "D"]
SIZES = {"gsm8kx": 40, "gsm1kx": 30, "gsm8kv2": 40, "gsm1k": 30, "valx": 20, "val": 20}
BOOT = 300
OPT = ab.Opt(boot=BOOT, boot_seed=0)


def pred_row(item_id: str, correct: bool) -> dict:
    probs = [0.7, 0.1, 0.1, 0.1] if correct else [0.1, 0.7, 0.1, 0.1]
    return {"item_id": item_id, "labels": LABELS, "probs": probs, "gold_label": "A"}


def write_run(root: Path, name: str, seed: int, epochs, counts: dict, metrics: dict | None = None,
              skip=(), no_metrics: bool = False, ids: dict | None = None) -> Path:
    """counts: set -> number of correct items (the first ones); sets not given get half correct."""
    d = root / name
    d.mkdir(parents=True)
    if not no_metrics:
        m = {"args": {"seed": seed, "epochs": epochs}, "eval": {}}
        m.update(metrics or {})
        (d / "metrics.json").write_text(json.dumps(m))
    for s, n in SIZES.items():
        if s in skip:
            continue
        c = counts.get(s, n // 2)
        prefix = (ids or {}).get(s, s)
        (d / f"preds_{s}.jsonl").write_text("".join(json.dumps(pred_row(f"{prefix}-{i}", i < c)) + "\n"
                                                   for i in range(n)))
    return d


def vt(arm, ep, seed):
    return ab.vt_name(arm, ep, seed)


def build_b(root: Path, a2=None, gfpq=None, gfpqr=None, seeds=(13, 14, 15), valx=None, skip_valx=()):
    """Default card-B fixture: A2 gsm8kx 20 / gsm1kx 15 correct; GFPQ +0 on gsm8kx and +5/+6/+7 on gsm1kx (seed
    13/14/15) at both epochs; GFPQR = A2. valx: {(arm, ep, seed): n correct of 20}."""
    a2 = a2 or (lambda ep, s: {"gsm8kx": 20, "gsm1kx": 15, "gsm8kv2": 25, "gsm1k": 18})
    gfpq = gfpq or (lambda ep, s: {"gsm8kx": 20, "gsm1kx": 15 + (s - 8), "gsm8kv2": 25, "gsm1k": 18})
    gfpqr = gfpqr or a2
    valx = valx or {}
    for s in seeds:
        for ep in (1, 2):
            for arm, f in (("A2", a2), ("GFPQ", gfpq)):
                c = dict(f(ep, s))
                c["valx"] = valx.get((arm, ep, s), 10)
                write_run(root, vt(arm, ep, s), s, ep, c, skip=("valx",) if (arm, ep, s) in skip_valx else ())
        c = dict(gfpqr(2, s))
        c["valx"] = valx.get(("GFPQR", 2, s), 10)
        write_run(root, vt("GFPQR", 2, s), s, 2, c)


def traj_row(epoch, intact, force, mean=None, swap=None, set_name="val", detect=None):
    r = {"step": int(100 * epoch), "epoch": epoch, set_name: {"acc": intact}}
    r[f"{set_name}@force"] = {"acc": force, "acc_drop": intact - force}
    r[f"{set_name}@mean"] = {"acc": intact if mean is None else mean}
    r[f"{set_name}@forceswap"] = {"acc": intact if swap is None else swap}
    if detect is not None:
        r[set_name]["detect"] = detect
    return r


def tsf_metrics(mean_drop, swap_drop, traj=None, pilot=True):
    m = {"trajectory": traj or []}
    if pilot:
        m["tsf_pilot"] = {"necessity_set": "gsm8kv2", "mean_drop": mean_drop, "forceswap_drop": swap_drop,
                          "g1_own": mean_drop >= 0.02 and swap_drop >= 0.03}
    m["slot_necessity"] = {"gsm8kv2": {"mean": {"acc_drop": mean_drop}, "forceswap": {"acc_drop": swap_drop},
                                       "force": {"acc_drop": -0.05}},
                           "gsm1k": {"mean": {"acc_drop": 0.001}, "forceswap": {"acc_drop": 0.002}}}
    return m


class TmpRoot(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def analyze(self):
        return ab.analyze(self.root, OPT)


# ----------------------------------------------------------------------------------------------------------
# Real batch-13 seed-12 runs
# ----------------------------------------------------------------------------------------------------------

@unittest.skipUnless((REAL_ROOT / vt("A2", 1, 12)).is_dir(), "batch-13 s12 runs not pulled")
class TestRealS12(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = ab.analyze(REAL_ROOT, OPT)
        cls.md = ab.render(cls.res)

    def acc(self, arm, ep, pool):
        a = self.res["s12"]["acc"][(arm, ep)][pool]
        self.assertEqual(a["status"], "ok", a)
        return round(100 * a["acc"], 1)

    def test_readme_numbers(self):
        """README_14 quotes: A2 1 / 2 epochs cue-free 58.2 / 55.6, original 70.4 / 69.6; GFPQ 2 epochs 70.7 / 60.2;
        gsm8kx GFPQVTR = GFPQ = 62.6."""
        self.assertEqual(self.acc("A2", 1, "cuefree"), 58.2)
        self.assertEqual(self.acc("A2", 2, "cuefree"), 55.6)
        self.assertEqual(self.acc("A2", 1, "orig"), 70.4)
        self.assertEqual(self.acc("A2", 2, "orig"), 69.6)
        self.assertEqual(self.acc("GFPQ", 2, "orig"), 70.7)
        self.assertEqual(self.acc("GFPQ", 2, "cuefree"), 60.2)
        self.assertEqual(self.acc("GFPQ", 2, "gsm8kx"), 62.6)
        self.assertEqual(self.acc("GFPQVTR", 2, "gsm8kx"), 62.6)

    def test_s12_comparisons_k1(self):
        label, cm = self.res["s12"]["comps"][1]
        self.assertIn("GFPQ 2 轮 − A2 2 轮", label)
        st = cm["cuefree"]["stats"]
        self.assertEqual(st["k"], 1)
        self.assertIsNone(st["t"]["lo"])                                  # k = 1: no seed-t
        self.assertAlmostEqual(100 * st["mean"], 60.23 - 55.63, delta=0.02)
        self.assertEqual(cm["cuefree"]["seeds"], [12])
        self.assertIn("GFPQVTR", self.res["s12"]["comps"][-1][0])

    def test_seed12_not_in_confirmation(self):
        lk = self.res["b"]["lookups"]
        self.assertEqual({s for (_, _, s) in lk}, {13, 14, 15})
        if not (REAL_ROOT / vt("A2", 1, 13)).is_dir():                      # batch 14 not pulled yet
            self.assertEqual(self.res["b"]["epoch_rule"]["A2"]["status"], ab.PENDING)
            self.assertEqual(self.res["b"]["v_c1"]["status"], ab.PENDING)
            self.assertEqual(self.res["b"]["v_c2"]["status"], ab.PENDING)
            self.assertEqual(self.res["b"]["v_go"]["status"], ab.PENDING)
        # the non-_vt seed-13 runs (A2-c1lr3ch-...student_gsm8k_tf-s13) must never be picked up
        for l in lk.values():
            if l.status == "ok":
                self.assertIn(ab.DATA_VT, l.run.name)

    def test_card_a_pending_and_render(self):
        if not any(n.startswith("TSF") for n in os.listdir(REAL_ROOT)):
            self.assertEqual(self.res["a"]["early_stop"]["status"], ab.PENDING)
            self.assertEqual(self.res["a"]["g1"]["status"], ab.PENDING)
            self.assertEqual(self.res["a"]["g3"]["status"], ab.PENDING)
        for head in ("## 总览", "## B.", "## 种子 12", "## A.", "## run 目录状态"):
            self.assertIn(head, self.md)
        self.assertIn("58.17", self.md)


# ----------------------------------------------------------------------------------------------------------
# Card B on synthetic runs
# ----------------------------------------------------------------------------------------------------------

class TestEpochRule(TmpRoot):
    def test_tie_goes_to_one_epoch_and_higher_mean_wins(self):
        valx = {("A2", 1, 13): 10, ("A2", 1, 14): 12, ("A2", 1, 15): 11,      # mean 11 = 11
                ("A2", 2, 13): 12, ("A2", 2, 14): 10, ("A2", 2, 15): 11,
                ("GFPQ", 1, 13): 10, ("GFPQ", 1, 14): 10, ("GFPQ", 1, 15): 10,
                ("GFPQ", 2, 13): 11, ("GFPQ", 2, 14): 10, ("GFPQ", 2, 15): 10}
        build_b(self.root, valx=valx)
        er = self.analyze()["b"]["epoch_rule"]
        self.assertEqual(er["A2"]["status"], "DECIDED")
        self.assertEqual(er["A2"]["chosen"], 1)
        self.assertTrue(er["A2"]["tie"])
        self.assertEqual(er["GFPQ"]["chosen"], 2)
        self.assertAlmostEqual(er["GFPQ"]["means"][2], 31 / 60)

    def test_two_of_three_seeds_is_pending(self):
        build_b(self.root, seeds=(13, 14), valx={("A2", 2, 13): 15, ("A2", 2, 14): 15})
        res = self.analyze()
        er = res["b"]["epoch_rule"]["A2"]
        self.assertEqual(er["status"], ab.PENDING)
        self.assertIsNone(er["chosen"])
        self.assertAlmostEqual(er["means"][2], 0.75)                       # provisional numbers still shown
        self.assertEqual(res["b"]["v_c1"]["status"], ab.PENDING)
        c = res["b"]["aligned"][1]["cuefree"]                              # provisional k = 2, verdict PENDING
        self.assertEqual(c["status"], "partial")
        self.assertEqual(c["stats"]["k"], 2)
        self.assertEqual(ab.verdict_from(c, ab.c1_criteria)["status"], ab.PENDING)
        self.assertEqual(res["b"]["v_c2"]["status"], ab.PENDING)

    def test_no_valx_file_is_pending_but_accuracy_rows_render(self):
        build_b(self.root, skip_valx={("GFPQ", 1, 14)})
        res = self.analyze()
        self.assertEqual(res["b"]["epoch_rule"]["GFPQ"]["status"], ab.PENDING)
        self.assertEqual(res["b"]["epoch_rule"]["A2"]["status"], "DECIDED")
        self.assertEqual(res["b"]["aligned"][1]["cuefree"]["status"], "ok")
        self.assertEqual(res["b"]["v_c1"]["status"], ab.PENDING)
        md = ab.render(res)
        self.assertIn("无 preds_valx", md)


class TestThreeEpochRule(unittest.TestCase):
    @staticmethod
    def er(v1, v2):
        return {"table": {1: dict(zip((13, 14, 15), v1)), 2: dict(zip((13, 14, 15), v2))}}

    def test_rule(self):
        self.assertEqual(ab.three_epoch_rule(self.er([.5, .5, .5], [.6, .55, .4]))["status"], "TRIGGERED")
        self.assertEqual(ab.three_epoch_rule(self.er([.5, .5, .5], [.6, .5, .4]))["status"], "NOT TRIGGERED")  # tie
        self.assertEqual(ab.three_epoch_rule(self.er([.5, .5, None], [.6, .4, None]))["status"], ab.PENDING)
        self.assertEqual(ab.three_epoch_rule(self.er([.5, .5, None], [.4, .4, None]))["status"], "NOT TRIGGERED")
        self.assertEqual(ab.three_epoch_rule(self.er([.5, .5, None], [.6, .6, None]))["status"], "TRIGGERED")


class TestConfirmation(TmpRoot):
    def test_c1_passes_carried_by_gsm1kx_and_seed12_excluded(self):
        build_b(self.root)
        # seed 12 with a huge negative effect must not enter any card-B number
        write_run(self.root, vt("A2", 1, 12), 12, 1, {"gsm8kx": 40, "gsm1kx": 30})
        write_run(self.root, vt("GFPQ", 1, 12), 12, 1, {"gsm8kx": 0, "gsm1kx": 0})
        res = self.analyze()
        b = res["b"]
        self.assertEqual((b["eA"], b["eG"]), (1, 1))                       # all valx equal -> tie -> 1
        c1 = b["c1"]["cuefree"]
        self.assertEqual(c1["seeds"], [13, 14, 15])
        self.assertEqual([round(70 * d) for d in c1["stats"]["d"]], [5, 6, 7])
        self.assertEqual(b["v_c1"]["status"], ab.PASS, b["v_c1"])
        self.assertEqual(b["v_c1p"]["status"], ab.PASS, b["v_c1p"])
        self.assertEqual(b["v_c1_holds"]["status"], ab.PASS)
        self.assertEqual(b["c1"]["gsm8kx"]["stats"]["mean"], 0.0)
        note = ab.halves_note(b["c1"], "cuefree", ab.c1_criteria)
        self.assertIn("只有 gsm1kx 单独过线", note)
        self.assertIn("100% 来自 gsm1kx", note)
        # seed-level t is reported with df = 2
        self.assertEqual(c1["stats"]["t"]["df"], 2)
        # C2: GFPQR = A2 here, so the same differences -> PASS; GO
        self.assertEqual(b["v_c2"]["status"], ab.PASS)
        self.assertEqual(b["v_go"]["status"], ab.PASS)
        self.assertIn("只有 gsm1kx 单独过线", ab.render(res))

    def test_c1_fails_when_one_seed_is_not_positive(self):
        build_b(self.root, gfpq=lambda ep, s: {"gsm8kx": 20, "gsm1kx": 15 + (0 if s == 14 else 9)})
        b = self.analyze()["b"]
        self.assertGreaterEqual(b["c1"]["cuefree"]["stats"]["mean"], ab.C1_MEAN)
        self.assertEqual(b["v_c1"]["status"], ab.FAIL)
        self.assertIn("不全 > 0", b["v_c1"]["why"])
        self.assertEqual(b["v_go"]["status"], ab.FAIL)

    def test_c1_mean_below_threshold_fails(self):
        # +1 item of 70 per seed = +1.43 points < +1.5
        build_b(self.root, gfpq=lambda ep, s: {"gsm8kx": 20, "gsm1kx": 16})
        b = self.analyze()["b"]
        self.assertAlmostEqual(b["c1"]["cuefree"]["stats"]["mean"], 1 / 70)
        self.assertEqual(b["v_c1"]["status"], ab.FAIL)

    def test_c1_prime_direction(self):
        # GFPQ better at 1 epoch, worse at 2 epochs; valx picks 1 epoch for both -> C1 PASS, C1' FAIL
        def gfpq(ep, s):
            return {"gsm8kx": 20, "gsm1kx": 21 if ep == 1 else 13}
        valx = {("GFPQ", 1, s): 15 for s in (13, 14, 15)}
        build_b(self.root, gfpq=gfpq, valx=valx)
        b = self.analyze()["b"]
        self.assertEqual((b["eA"], b["eG"]), (1, 1))
        self.assertEqual(b["v_c1"]["status"], ab.PASS)
        self.assertEqual(b["v_c1p"]["status"], ab.FAIL)
        self.assertEqual(b["v_c1_holds"]["status"], ab.FAIL)

    def test_cross_epoch_c1(self):
        """A2 chosen at 2 epochs, GFPQ at 1: C1 = GFPQ c1 - A2 c2, distinct from both aligned comparisons."""
        valx = {("A2", 2, s): 15 for s in (13, 14, 15)}
        build_b(self.root, a2=lambda ep, s: {"gsm8kx": 20, "gsm1kx": 15 if ep == 1 else 10}, valx=valx)
        b = self.analyze()["b"]
        self.assertEqual((b["eA"], b["eG"]), (2, 1))
        self.assertEqual([round(70 * d) for d in b["c1"]["cuefree"]["stats"]["d"]], [10, 11, 12])
        self.assertEqual(b["c1"]["cuefree"]["runs_b"][0], vt("A2", 2, 13))
        self.assertEqual(b["c1"]["cuefree"]["runs_a"][0], vt("GFPQ", 1, 13))

    def test_c2_fails_on_equal_seed(self):
        build_b(self.root, gfpqr=lambda ep, s: {"gsm8kx": 20, "gsm1kx": 15 + (s - 8) if s == 15 else 15})
        b = self.analyze()["b"]
        self.assertEqual([round(70 * d) for d in b["c2"]["cuefree"]["stats"]["d"]], [5, 6, 0])
        self.assertEqual(b["v_c2"]["status"], ab.FAIL)

    def test_item_mismatch_is_an_error_row_not_a_crash(self):
        build_b(self.root)
        d = self.root / vt("GFPQR", 2, 14)
        rows = [json.loads(x) for x in (d / "preds_gsm8kx.jsonl").read_text().splitlines()]
        rows[0]["item_id"] = "other-0"
        (d / "preds_gsm8kx.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        res = self.analyze()
        self.assertEqual(res["b"]["c2"]["cuefree"]["status"], "error")
        self.assertEqual(res["b"]["v_c2"]["status"], ab.PENDING)
        self.assertIn("ERROR", ab.render(res))

    def test_empty_root_all_pending(self):
        res = self.analyze()
        for k in ("v_c1", "v_c1p", "v_c1_holds", "v_c2", "v_go"):
            self.assertEqual(res["b"][k]["status"], ab.PENDING, k)
        for k in ("early_stop", "g1", "g3", "ptad"):
            self.assertEqual(res["a"][k]["status"], ab.PENDING, k)
        md = ab.render(res)
        self.assertIn("0/28 个目录在", md)
        res = ab.analyze(self.root / "does-not-exist", OPT)
        self.assertEqual(res["b"]["v_go"]["status"], ab.PENDING)


# ----------------------------------------------------------------------------------------------------------
# Card A on synthetic runs
# ----------------------------------------------------------------------------------------------------------

class TestTSFNames(TmpRoot):
    def test_knob_suffix_and_prefix(self):
        write_run(self.root, f"TSF-tg0.69-pr-c2lr3ch-{ab.MODEL}-{ab.DATA_SI}-s12", 12, 2, {})
        write_run(self.root, ab.tsf_name("TSFN"), 12, 2, {})
        write_run(self.root, ab.tsf_name("TSF6"), 12, 2, {})
        write_run(self.root, f"TSFR-al12-20-pr-c2lr3ch-{ab.MODEL}-{ab.DATA_SI}-s12", 12, 2, {})
        write_run(self.root, f"TSFX-pr-c2lr3ch-{ab.MODEL}-{ab.DATA_SI}-s13", 13, 2, {})       # wrong seed
        runs = ab.Runs(self.root)
        self.assertEqual(runs.tsf("TSF").run.name, f"TSF-tg0.69-pr-c2lr3ch-{ab.MODEL}-{ab.DATA_SI}-s12")
        self.assertEqual(runs.tsf("TSFN").run.name, ab.tsf_name("TSFN"))
        self.assertEqual(runs.tsf("TSFR").status, "ok")
        self.assertEqual(runs.tsf("TSFX").status, "missing")
        self.assertEqual(runs.tsf("PTAD").status, "missing")
        self.assertEqual(ab.tsf_name("TSF"), "TSF-pr-c2lr3ch-Qwen3.5-4B-student_gsm8k_tf_si-s12")

    def test_two_matching_dirs_are_ambiguous_and_pending(self):
        traj = [traj_row(1.0, 0.70, 0.80)]
        write_run(self.root, ab.tsf_name("TSF"), 12, 2, {}, metrics=tsf_metrics(0.05, 0.06, traj))
        write_run(self.root, f"TSF-tg0.69-pr-c2lr3ch-{ab.MODEL}-{ab.DATA_SI}-s12", 12, 2, {},
                  metrics=tsf_metrics(0.05, 0.06, traj))
        res = self.analyze()
        self.assertEqual(res["a"]["lookups"]["TSF"].status, "ambiguous")
        self.assertEqual(res["a"]["early_stop"]["status"], ab.PENDING)
        self.assertEqual(res["a"]["g1"]["status"], ab.PENDING)
        self.assertIn("多个目录匹配", ab.render(res))


class TestDetectorAndG1(TmpRoot):
    def tsf(self, arm, mean_drop, swap_drop, traj=None, **kw):
        write_run(self.root, ab.tsf_name(arm), 12, 2, kw.pop("counts", {}),
                  metrics=tsf_metrics(mean_drop, swap_drop, traj, **kw))

    def test_early_stop_threshold_in_points(self):
        det = {"n_chain": 300, "force_minus_intact": 0.08, "intact_minus_mean": 0.01, "intact_minus_forceswap": 0.02}
        traj = [traj_row(0.5, 0.68, 0.70), traj_row(1.0, 0.700, 0.719, mean=0.69, swap=0.66, detect=det),
                traj_row(1.5, 0.70, 0.75)]
        self.tsf("TSF", 0.0, 0.0, traj)
        es = self.analyze()["a"]["early_stop"]
        self.assertEqual(es["status"], ab.FAIL)                            # +1.9 < +2
        self.assertAlmostEqual(es["diff"], 0.019)
        self.assertIn("停掉槽位这条线", es["why"])

    def test_early_stop_pass_and_detector_table(self):
        traj = [traj_row(0.5, 0.68, 0.70), traj_row(1.0, 0.700, 0.725), traj_row(1.5, 0.70, 0.75)]
        self.tsf("TSF", 0.0, 0.0, traj)
        self.tsf("TSFN", 0.0, 0.0, [traj_row(1.0, 0.70, 0.70)])
        res = self.analyze()
        self.assertEqual(res["a"]["early_stop"]["status"], ab.PASS)
        self.assertEqual(len(res["a"]["detector"]["TSF"]), 3)
        self.assertEqual(len(res["a"]["detector"]["TSFN"]), 1)
        self.assertIn("| TSF | 1.0 | val |", ab.render(res))

    def test_early_stop_fallbacks(self):
        # no metrics.json yet (job running): trajectory.json; no val@force: the first <set>@force, noted
        d = write_run(self.root, ab.tsf_name("TSF"), 12, 2, {}, no_metrics=True)
        (d / "trajectory.json").write_text(json.dumps([traj_row(1.0, 0.70, 0.75, set_name="gsm8kv2")]))
        es = self.analyze()["a"]["early_stop"]
        self.assertEqual(es["status"], ab.PASS)
        self.assertEqual(es["set"], "gsm8kv2")
        self.assertIn("改用 gsm8kv2", es["why"])

    def test_corrupt_files_are_pending_not_a_crash(self):
        d = write_run(self.root, ab.tsf_name("TSF"), 12, 2, {}, no_metrics=True)
        (d / "trajectory.json").write_text('[{"epoch": 1.0, "val": {"ac')          # caught mid-write
        (d / "preds_valx.jsonl").write_text('{"item_id": "valx-0", "lab')
        res = self.analyze()
        self.assertEqual(res["a"]["early_stop"]["status"], ab.PENDING)
        self.assertIsNone(res["a"]["acc"]["TSF"]["valx"])

    def test_early_stop_no_one_epoch_point(self):
        self.tsf("TSF", 0.0, 0.0, [traj_row(0.5, 0.70, 0.75)])
        self.assertEqual(self.analyze()["a"]["early_stop"]["status"], ab.PENDING)

    def test_g1_units_and_controls(self):
        self.tsf("TSF", 0.025, 0.035)
        for arm in ab.TSF_CONTROLS:
            self.tsf(arm, 0.005, 0.009)
        self.assertEqual(self.analyze()["a"]["g1"]["status"], ab.PASS)

    def test_g1_swap_drop_below_three_points(self):
        self.tsf("TSF", 0.025, 0.025)                                     # 2.5 >= 2 but 2.5 < 3
        for arm in ab.TSF_CONTROLS:
            self.tsf(arm, 0.0, 0.0)
        self.assertEqual(self.analyze()["a"]["g1"]["status"], ab.FAIL)

    def test_g1_tsfn_control_counts(self):
        self.tsf("TSF", 0.03, 0.04)
        self.tsf("TSFN", 0.012, 0.0)                                       # README_14: TSFN must be < 1 too
        self.tsf("TSFR", 0.0, 0.0)
        self.tsf("TSFX", 0.0, 0.0)
        g1 = self.analyze()["a"]["g1"]
        self.assertEqual(g1["status"], ab.FAIL)
        self.assertIn("TSFN mean +1.20", g1["why"])

    def test_g1_fail_known_even_with_pending_controls(self):
        self.tsf("TSF", 0.0, 0.0)
        self.assertEqual(self.analyze()["a"]["g1"]["status"], ab.FAIL)

    def test_g1_slot_necessity_fallback(self):
        self.tsf("TSF", 0.03, 0.04, pilot=False)
        for arm in ab.TSF_CONTROLS:
            self.tsf(arm, 0.0, 0.0, pilot=False)
        res = self.analyze()
        self.assertEqual(res["a"]["drops"]["TSF"]["src"], "slot_necessity")
        self.assertEqual(res["a"]["g1"]["status"], ab.PASS)
        self.assertIn("gsm1k", res["a"]["drops"]["TSF"]["other_sets"])


class TestG3AndPTAD(TmpRoot):
    def base(self, valx=None):
        build_b(self.root, valx=valx)
        write_run(self.root, vt("A2", 1, 12), 12, 1, {"gsm8kv2": 20, "gsm1k": 15, "gsm8kx": 20, "gsm1kx": 15})
        write_run(self.root, vt("A2", 2, 12), 12, 2, {"gsm8kv2": 22, "gsm1k": 15, "gsm8kx": 18, "gsm1kx": 15})
        write_run(self.root, vt("GFPQ", 2, 12), 12, 2, {"gsm8kv2": 22, "gsm1k": 16, "gsm8kx": 22, "gsm1kx": 16})

    def arm(self, arm, counts):
        write_run(self.root, ab.tsf_name(arm), 12, 2, counts, metrics=tsf_metrics(0.0, 0.0))

    def test_g3_pass_against_valx_chosen_epoch(self):
        self.base()                                                        # valx tie -> A2 1 epoch
        self.arm("TSF", {"gsm8kv2": 22, "gsm1k": 15, "gsm8kx": 22, "gsm1kx": 15})   # +2/70 = +2.86 on both pools
        res = self.analyze()
        self.assertEqual(res["a"]["eA"], 1)
        self.assertEqual(res["a"]["g3"]["status"], ab.PASS, res["a"]["g3"])
        self.assertAlmostEqual(res["a"]["tsf_vs"][("A2", 1)]["orig"]["stats"]["mean"], 2 / 70)
        self.assertEqual(res["a"]["tsf_vs"][("GFPQ", 2)]["cuefree"]["status"], "ok")   # reported, not gating

    def test_g3_fails_on_one_pool(self):
        self.base()
        self.arm("TSF", {"gsm8kv2": 22, "gsm1k": 15, "gsm8kx": 21, "gsm1kx": 15})   # cue-free +1/70 < +2
        g3 = self.analyze()["a"]["g3"]
        self.assertEqual(g3["status"], ab.FAIL)
        self.assertIn("无线索合并 gsm8kx+gsm1kx +1.43 < +2.00", g3["why"])

    def test_g3_undecided_epoch(self):
        # card B missing -> A2 epoch undecided: a verdict only when both candidate epochs agree
        write_run(self.root, vt("A2", 1, 12), 12, 1, {"gsm8kv2": 20, "gsm1k": 15, "gsm8kx": 20, "gsm1kx": 15})
        write_run(self.root, vt("A2", 2, 12), 12, 2, {"gsm8kv2": 22, "gsm1k": 15, "gsm8kx": 18, "gsm1kx": 15})
        self.arm("TSF", {"gsm8kv2": 22, "gsm1k": 15, "gsm8kx": 22, "gsm1kx": 15})
        g3 = self.analyze()["a"]["g3"]
        self.assertEqual(g3["status"], ab.PENDING)                         # orig: +2.86 vs A2 c1, 0 vs A2 c2
        shutil_dir = self.root / ab.tsf_name("TSF")
        for p in shutil_dir.iterdir():
            p.unlink()
        shutil_dir.rmdir()
        self.arm("TSF", {"gsm8kv2": 18, "gsm1k": 15, "gsm8kx": 18, "gsm1kx": 15})
        g3 = self.analyze()["a"]["g3"]
        self.assertEqual(g3["status"], ab.FAIL)
        self.assertIn("两个候选轮数结论相同", g3["why"])

    def test_ptad_controls_margin(self):
        self.base()
        self.arm("PTAD", {"gsm8kv2": 23, "gsm1k": 15, "gsm8kx": 23, "gsm1kx": 15})  # +3/70 = +4.29 over A2 c1
        self.arm("PTADR", {"gsm8kv2": 21, "gsm1k": 15, "gsm8kx": 21, "gsm1kx": 15})  # +2/70 = +2.86 lower
        self.arm("PTADX", {"gsm8kv2": 22, "gsm1k": 15, "gsm8kx": 22, "gsm1kx": 15})  # +1/70 = +1.43 < +1.5
        p = self.analyze()["a"]["ptad"]
        self.assertEqual(p["status"], ab.FAIL)
        self.assertIn("PTAD − PTADX: 原测试集合并 gsm8kv2+gsm1k +1.43 < +1.50", p["why"])
        d = self.root / ab.tsf_name("PTADX")
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
        self.arm("PTADX", {"gsm8kv2": 20, "gsm1k": 15, "gsm8kx": 20, "gsm1kx": 15})
        self.assertEqual(self.analyze()["a"]["ptad"]["status"], ab.PASS)


class TestCLI(TmpRoot):
    def test_main_writes_out(self):
        build_b(self.root)
        out = self.root / "report" / "b14.md"
        from contextlib import redirect_stdout
        from io import StringIO
        buf = StringIO()
        with redirect_stdout(buf):
            ab.main(["--root", str(self.root), "--boot", "100", "--out", str(out)])
        self.assertTrue(out.exists())
        self.assertEqual(out.read_text(), buf.getvalue())
        self.assertIn("**PASS**", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
