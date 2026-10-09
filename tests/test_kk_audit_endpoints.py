"""CPU tests for scripts/kk_audit_endpoints.py on synthetic kk_audit.py preds.jsonl files (no model): the registered v1
endpoints (--preset v1) and the v2 endpoints of the 2026-10-09 amendment (gate, both-correct scoring, coincidence
floor, own-answer retention); the v2 reading of the real v1 run when results/hpc/kkaudit is present."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import kk_audit_endpoints as E  # noqa: E402

LAYERS = (4, 8)


def write_model(d: Path, name: str, hit: dict, n_prog: int = 40, per_prog: int = 3) -> None:
    """hit: {cond: fn(i, L) -> bool}; the patched answer equals the target (1) when fn says so, else option 5."""
    pairs = []
    for i in range(n_prog * per_prog):
        pairs.append({"pair_id": f"p{i}", "program": f"g{i // per_prog}", "target": 1, "in_all": True,
                      "in_test": True, "src_desc_agree": 0 if i % 2 else 1,
                      "viol_base": {"prop": 1, "local": 1 if i % 3 == 0 else 2}, "prop_extra_viol": 0,
                      "base_gold": 0, "t_prop": 2})
    conds = ["role:local@last", "role:z@last", "inv:base@last", "inv/mid:base@last", "stmt:source@stmt",
             *E.E2_CONDS]
    with open(d / "preds.jsonl", "w") as f:
        for c in conds:
            tgt = 0 if c.startswith("inv") else 1
            ps = [dict(p, target=tgt) for p in pairs]
            f.write(json.dumps({"kind": "pairs", "model": name, "cond": c, "pairs": ps}) + "\n")
            for L in LAYERS:
                for m in ("das", "random", "noise", "full"):
                    fn = hit.get((c, m), hit.get(c, lambda i, L: False)) if m == "das" else \
                        hit.get((c, m), lambda i, L: False)
                    f.write(json.dumps({"kind": "preds", "model": name, "cond": c, "layer": L, "method": m,
                                        "preds": [tgt if fn(i, L) else 5 for i in range(len(ps))]}) + "\n")


def run(dirs, *extra):
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = E.main([str(d) for d in dirs] + list(extra) + ["--reps", "300", "--min-e2", "0", "--preset", "v1"])
    return res


class TestEndpoints(unittest.TestCase):
    def test_tree_like_model_passes_e1_e2(self):
        yes = lambda i, L: True
        no = lambda i, L: False
        hit = {"role:local@last": yes, "inv:base@last": yes, "inv/mid:base@last": yes,
               "role/mid:prop@last": lambda i, L: L == 8, "role/mid:zprop@last": no,
               "role/mid:local@last": lambda i, L: L == 4, "role/mid:z@last": no,
               "stmt:source@stmt": yes}
        with tempfile.TemporaryDirectory() as d:
            write_model(Path(d), "TFL-s12", hit)
            r = run([d])["models"][0]
        self.assertTrue(r["E1_holds_at_some_layer"])
        self.assertTrue(r["E3_holds_at_some_layer"])
        self.assertEqual(r["E2"][8]["did"], 1.0)
        self.assertEqual(r["E2"][4]["did"], -1.0)
        self.assertTrue(r["E2"][8]["holds"] and not r["E2"][4]["holds"])
        self.assertTrue(r["E2_holds_at_some_layer"])
        self.assertFalse(r["E2_against"])
        self.assertEqual(r["E2_n_pairs"], 120)
        self.assertAlmostEqual(r["ci_level"], 0.975)

    def test_underpowered_e2_is_neither_pass_nor_against(self):
        hit = {"role/mid:prop@last": lambda i, L: True}
        with tempfile.TemporaryDirectory() as d:
            write_model(Path(d), "TFL-s12", hit)
            buf = io.StringIO()
            with redirect_stdout(buf):
                r = E.main([d, "--reps", "100", "--preset", "v1"])["models"][0]   # registered minimum: 250 > 120
        self.assertTrue(r["E2_underpowered"])
        self.assertIsNone(r["E2_holds_at_some_layer"])
        self.assertFalse(r["E2_against"])

    def test_doubt_direction_fails_e2(self):
        # prop and zprop both reached (plausibility), local and z both missed: DiD = 0; and the prop subspace carries
        # inv pairs off base: no E2
        yes = lambda i, L: True
        no = lambda i, L: False
        hit = {"role/mid:prop@last": yes, "role/mid:zprop@last": yes, "role/mid:local@last": no,
               "role/mid:z@last": no, "inv/mid:base@last": no}
        with tempfile.TemporaryDirectory() as d:
            write_model(Path(d), "A2-s12", hit)
            r = run([d])["models"][0]
        self.assertEqual(r["E2"][4]["did"], 0.0)
        self.assertFalse(r["E2_holds_at_some_layer"])
        self.assertTrue(r["E2_against"])
        self.assertFalse(r["E1_holds_at_some_layer"])

    def test_arm_contrast_paired_and_holm(self):
        yes = lambda i, L: True
        no = lambda i, L: False
        tree = {"role/mid:prop@last": yes, "role/mid:zprop@last": no, "role/mid:local@last": no,
                "role/mid:z@last": no}
        flat = {"role/mid:prop@last": lambda i, L: i % 4 == 0, "role/mid:zprop@last": no,
                "role/mid:local@last": no, "role/mid:z@last": no}
        with tempfile.TemporaryDirectory() as d:
            dirs = []
            for name, h in (("TFL-s12", tree), ("TFL-s13", tree), ("A2-s12", flat), ("A2-s13", flat),
                            ("TFLM-s12", tree), ("TFLM-s13", tree)):
                (Path(d) / name).mkdir()
                write_model(Path(d) / name, name, h)
                dirs.append(Path(d) / name)
            res = run(dirs, "--contrast", "TFL,A2", "--contrast", "TFL,TFLM")
        c = {r["contrast"]: r for r in res["contrasts"]}
        self.assertEqual(c["TFL-A2"]["did_best_layer"], {"TFL": 1.0, "A2": 0.25})
        self.assertAlmostEqual(c["TFL-A2"]["diff"], 0.75)
        self.assertLess(c["TFL-A2"]["p_boot"], 0.01)
        self.assertEqual(c["TFL-TFLM"]["diff"], 0.0)
        self.assertEqual(c["TFL-TFLM"]["p_holm"], 1.0)
        self.assertTrue(c["TFL-A2"]["supported"] and not c["TFL-TFLM"]["supported"])
        self.assertTrue(c["TFL-A2"]["preliminary"])
        self.assertEqual(c["TFL-A2"]["seeds"]["TFL"], ["TFL-s12", "TFL-s13"])

    def test_refit_dir_merges_into_one_model(self):
        no = lambda i, L: False
        base = {"role/mid:prop@last": lambda i, L: i % 2 == 0, "role/mid:zprop@last": no,
                "role/mid:local@last": no, "role/mid:z@last": no}
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "main", Path(d) / "refit"
            a.mkdir(); b.mkdir()
            write_model(a, "TFL-s12", base)
            write_model(b, "TFL-s12", {"role/mid:prop@last": lambda i, L: True})
            with open(b / "preds.jsonl") as f:             # the refit holds only the refitted condition
                keep = [l for l in f if json.loads(l)["cond"] == "role/mid:prop@last"]
            (b / "preds.jsonl").write_text("".join(keep))
            res = run([a, b], "--contrast", "TFL,TFL")
        self.assertEqual(len(res["models"]), 1)
        self.assertEqual(res["models"][0]["E2"][4]["iia_prop"], 1.0)
        self.assertEqual(res["models"][0]["E2"][4]["iia_local"], 0.0)
        self.assertEqual(res["contrasts"][0]["seeds"]["TFL"], ["TFL-s12"])


CELLS = ((16, 1), (16, 4), (20, 1), (20, 4))
V2_CONDS = ["stmt:source@last", "role:local@last", "role:z@last", "inv:base@last", "inv/mid:base@last", *E.E2_CONDS]


def write_v2(d: Path, name: str, hit: dict, n_prog: int = 40, per_prog: int = 3, both_every: int = 2,
             base_wrong_hits_target: bool = False) -> None:
    """A v2 preds.jsonl. Every pair is in the all-pairs set; every `both_every`-th pair is both-correct (in_both).
    Pairs outside "both" have a wrong unpatched base answer: base_pred = 1 (= the target) if base_wrong_hits_target,
    else 4. hit: {cond or (cond, method): fn(i, L, k) -> bool}: the patched answer is the target (1) when fn says so,
    else the unpatched answer (base_pred). inv conditions: target = base_pred; base_gold = 0; t_prop = 2."""
    pairs = []
    for i in range(n_prog * per_prog):
        both = i % both_every == 0
        bp = 0 if both else (1 if base_wrong_hits_target else 4)
        pairs.append({"pair_id": f"p{i}", "program": f"g{i // per_prog}", "in_all": True, "in_both": both,
                      "base_pred": bp, "source_pred": 3, "base_ok": both, "source_ok": both, "base_gold": 0,
                      "src_desc_agree": 0 if (i // 2) % 2 else 1,
                      "viol_base": {"prop": 1, "local": 1 if i % 3 == 0 else 2}, "prop_extra_viol": 0, "t_prop": 2})
    with open(d / "preds.jsonl", "w") as f:
        for c in V2_CONDS:
            inv = c.startswith("inv")
            ps = [dict(p, target=p["base_pred"] if inv else 1) for p in pairs]
            f.write(json.dumps({"kind": "pairs", "version": "v2", "model": name, "cond": c, "pairs": ps}) + "\n")
            for L, k in CELLS:
                for m in ("das", "random", "noise", "full"):
                    fn = hit.get((c, m), hit.get(c, lambda i, L, k: False)) if m == "das" else \
                        hit.get((c, m), lambda i, L, k: False)
                    pr = [p["target"] if fn(i, L, k) else (5 if inv else p["base_pred"]) for i, p in enumerate(ps)]
                    if inv and m != "das" and (c, m) not in hit:
                        pr = [p["base_pred"] for p in ps]        # controls leave inv pairs at their own answer
                    f.write(json.dumps({"kind": "preds", "version": "v2", "model": name, "cond": c, "layer": L, "k": k,
                                        "method": m, "preds": pr}) + "\n")


def run2(dirs, *extra):
    buf = io.StringIO()
    with redirect_stdout(buf):
        return E.main([str(d) for d in dirs] + list(extra) + ["--reps", "300"])


YES = lambda i, L, k: True        # noqa: E731
NO = lambda i, L, k: False        # noqa: E731
TREE = {"stmt:source@last": lambda i, L, k: k == 4, "role:local@last": YES, "inv:base@last": YES,
        "inv/mid:base@last": YES, "role/mid:prop@last": lambda i, L, k: L == 20, "role/mid:zprop@last": NO,
        "role/mid:local@last": lambda i, L, k: L == 16, "role/mid:z@last": NO}


class TestEndpointsV2(unittest.TestCase):
    def test_gate_passes_and_tree_model_holds(self):
        with tempfile.TemporaryDirectory() as d:
            write_v2(Path(d), "TFL-s12", TREE)
            r = run2([d], "--min-e2-both", "0")["models"][0]
        self.assertTrue(r["gate"]["passed"])
        self.assertEqual(r["gate"]["passed_k"], {1: False, 4: True})
        self.assertEqual(r["cells"], ["L16/k1", "L16/k4", "L20/k1", "L20/k4"])
        self.assertAlmostEqual(r["ci_level"], 1 - 0.05 / 4)
        b = r["both"]
        self.assertTrue(b["E1_holds_at_some_cell"])
        self.assertEqual(b["E1"]["L16/k1"]["coinc"], 0.0)
        self.assertEqual(b["E1"]["L16/k1"]["inv_retention_own"], 1.0)
        self.assertEqual(b["E2"]["L20/k1"]["did"], 1.0)
        self.assertEqual(b["E2"]["L16/k1"]["did"], -1.0)
        self.assertTrue(b["E2_holds_at_some_cell"])
        self.assertFalse(b["E2_against"])
        self.assertEqual(b["E2_n_pairs"], 60)
        self.assertEqual(r["all"]["E2_n_pairs"], 120)
        self.assertEqual(r["primary"], "both")
        self.assertIn("dropped", r["E3"])

    def test_failed_gate_makes_endpoints_not_interpretable(self):
        hit = dict(TREE, **{"stmt:source@last": NO})
        with tempfile.TemporaryDirectory() as d:
            write_v2(Path(d), "TFL-s12", hit)
            r = run2([d], "--min-e2-both", "0")["models"][0]
        self.assertFalse(r["gate"]["passed"])
        for s in ("both", "all"):
            self.assertIsNone(r[s]["E1_holds_at_some_cell"])
            self.assertIsNone(r[s]["E2_holds_at_some_cell"])
            self.assertIn("gate failed", r[s]["not_interpretable"])
        self.assertTrue(r["both"]["E1"]["L16/k1"]["holds_raw"])           # computed, never read as a pass
        self.assertIsNone(r["both"]["E1"]["L16/k1"]["holds"])

    def test_coincidence_floor_on_all_pairs(self):
        # the patch changes nothing; where the base is wrong its unpatched answer happens to be the target:
        # IIA (all) = coincidence rate = .5, so E1 fails on the all-pairs set by its floor, and the both set has IIA 0
        hit = {"stmt:source@last": YES, "inv:base@last": YES, "role:local@last": NO}
        with tempfile.TemporaryDirectory() as d:
            write_v2(Path(d), "A2-s12", hit, base_wrong_hits_target=True)
            r = run2([d])["models"][0]
        e = r["all"]["E1"]["L20/k4"]
        self.assertEqual(e["iia"], 0.5)
        self.assertEqual(e["coinc"], 0.5)
        self.assertEqual(e["floor"], 0.5)
        self.assertFalse(e["holds"])
        self.assertEqual(r["both"]["E1"]["L20/k4"]["iia"], 0.0)

    def test_inv_retention_against_own_answer_and_noise_floor_from_k2(self):
        # half the inv pairs have a wrong base answer (base_pred 4 != base_gold 0); a patch that keeps the model's own
        # answer has retention 1.0 (scored against gold it would be .5); noise reaching the target: floor at k = 4 only
        hit = dict(TREE)
        hit[("role:local@last", "noise")] = YES
        with tempfile.TemporaryDirectory() as d:
            write_v2(Path(d), "TFL-s12", hit)
            r = run2([d])["models"][0]
        self.assertEqual(r["all"]["E1"]["L16/k1"]["inv_retention_own"], 1.0)
        self.assertEqual(r["all"]["E1"]["L16/k1"]["floor"], 0.0)
        self.assertEqual(r["all"]["E1"]["L16/k1"]["noise_reported_k1"], 1.0)
        self.assertEqual(r["all"]["E1"]["L16/k4"]["floor"], 1.0)
        self.assertFalse(r["all"]["E1"]["L16/k4"]["holds"])
        self.assertTrue(r["all"]["E1"]["L16/k1"]["holds"])

    def test_primary_e2_min_n(self):
        with tempfile.TemporaryDirectory() as d:
            write_v2(Path(d), "TFL-s12", TREE)               # 60 both-correct E2 pairs < 80
            r = run2([d])["models"][0]
        self.assertTrue(r["both"]["E2_underpowered"])
        self.assertIsNone(r["both"]["E2_holds_at_some_cell"])
        self.assertIsNone(r["both"]["E2_against"])
        self.assertEqual(r["both"]["E2_min_n"], 80)

    def test_v1_file_reads_as_k1_and_contrast_report_only(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "TFL-s12", Path(d) / "GFPS-s12"
            a.mkdir()
            b.mkdir()
            write_model(a, "TFL-s12", {"role/mid:prop@last": lambda i, L: True})        # v1 format: no k, in_test
            write_v2(b, "GFPS-s12", TREE)
            for x in (a, b):
                with open(x / "preds.jsonl") as f:
                    rows = [json.loads(l) for l in f]
                with open(x / "preds.jsonl", "w") as f:
                    for row in rows:
                        if row["kind"] == "pairs":
                            row["pairs"] = [dict(p, base_pred=p.get("base_pred", 0), in_test=True) for p in row["pairs"]]
                        f.write(json.dumps(row) + "\n")
            res = run2([a, b], "--contrast", "TFL,GFPS", "--min-e2-both", "0")
        m = {x["model"]: x for x in res["models"]}
        self.assertEqual(m["TFL-s12"]["cells"], ["L4/k1", "L8/k1"])
        self.assertEqual(m["TFL-s12"]["both"]["E2_n_pairs"], 120)
        self.assertFalse(m["TFL-s12"]["gate"]["passed"])                  # no stmt:source@last in this file
        c = [x for x in res["contrasts"] if x["set"] == "both"][0]
        self.assertTrue(c["report_only"] and c["preliminary"])

    @unittest.skipUnless((ROOT / "results/hpc/kkaudit/GFPS2B-s12/preds.jsonl").exists(), "v1 audit results absent")
    def test_real_v1_run_is_a_measurement_failure(self):
        r = run2([ROOT / "results/hpc/kkaudit/GFPS2B-s12", ROOT / "results/hpc/kkaudit/TFL2B-s12"])
        for m in r["models"]:
            self.assertFalse(m["gate"]["passed"])                         # stmt:source@last DAS IIA .05-.06
            self.assertLess(m["gate"]["best_by_k"][1], 0.1)
            e = m["all"]["E1"]["L20/k1"]
            self.assertLess(abs(e["iia"] - e["coinc"]), 0.01)              # IIA = coincidence on all pairs
            self.assertGreater(m["all"]["E1"]["L4/k1"]["inv_retention_own"], 0.9)


if __name__ == "__main__":
    unittest.main()
