"""CPU tests for scripts/kk_audit_endpoints.py on synthetic kk_audit.py preds.jsonl files (no model)."""
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
        res = E.main([str(d) for d in dirs] + list(extra) + ["--reps", "300", "--min-e2", "0"])
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
                r = E.main([d, "--reps", "100"])["models"][0]           # registered minimum: 250 > 120 pairs
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


if __name__ == "__main__":
    unittest.main()
