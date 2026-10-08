"""scripts/build_v8.py: the config is valid, staging swaps in Tree Forcing trees only for byte-identical prompts and
drops node rows, and the post-build check flags trees of ineligible sources, missing statement blocks and leaks.
CPU only, no data needed."""
import collections
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_v8 as b8  # noqa: E402

CFG = ROOT / "configs/v8/v8_release_4b.json"
KK = json.loads((ROOT / "tests/fixtures/kk_local_item.json").read_text())


def jl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class TestBuildV8(unittest.TestCase):
    def test_config(self):
        self.assertEqual(b8.main([str(CFG), "--check-only"]), 0)
        cfg = json.loads(CFG.read_text())
        self.assertEqual(b8.check_v8(cfg), [])
        elig = set(cfg["gfps"]["sources"])
        self.assertEqual(elig, {"knights_knaves", "numgen", "goldproof/fld", "goldproof/boardgameqa",
                                "goldproof/ruletaker", "rule"})
        for s in cfg["sources"]:                                 # answer-only sources drop their trees
            if s["name"] in ("v3sl_ao", "gsm8k_soc", "br40k"):
                self.assertEqual(s["nodes"], "drop")
            else:
                self.assertEqual(s["nodes"], "keep")
            self.assertNotIn("node_rows", s)                     # no standalone node rows anywhere
        bad = dict(cfg, gfps={})
        self.assertTrue(b8.check_v8(bad))

    def test_stage_trees_and_node_rows(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ms = [{"item_id": "a", "source": "numgen", "v7_source": "numgen", "prompt": "P1", "subqs": [{"id": "q"}]},
                  {"item_id": "b", "source": "numgen", "v7_source": "numgen", "prompt": "P2", "subqs": [{"id": "q"}],
                   "tree_source": "numgen_program"},
                  {"item_id": "a-n1", "source": "numgen_node", "v7_source": "numgen", "row_kind": "node", "prompt": "N"},
                  {"item_id": "g", "source": "gsm8k_soc", "v7_source": "gsm8k_soc", "prompt": "G"}]
            tf = [{"item_id": "a", "prompt": "P1", "subqs": [{"id": "x"}, {"id": "y"}], "local_preamble": "L",
                   "tree_fix": True, "tree_source": "numgen_program_tf", "has_tree": True},
                  {"item_id": "b", "prompt": "P2 changed", "subqs": [{"id": "z"}]}]
            jl(d / "ms.jsonl", ms)
            jl(d / "tf.jsonl", tf)
            c = collections.Counter()
            st = {"v7_source": "numgen", "drop_node_rows": True}
            b8.stage_file(d / "ms.jsonl", d / "out.jsonl", st, b8.load_trees(d, ["tf.jsonl"]), c)
            out = [json.loads(l) for l in (d / "out.jsonl").read_text().splitlines()]
            self.assertEqual([r["item_id"] for r in out], ["a", "b"])
            self.assertEqual(out[0]["subqs"], tf[0]["subqs"])
            self.assertEqual((out[0]["local_preamble"], out[0]["tree_source"]), ("L", "numgen_program_tf"))
            self.assertEqual(out[0]["prompt"], "P1")
            self.assertEqual((out[1]["subqs"], out[1]["has_tree"]), ([], False))          # prompt differs: no tree
            self.assertNotIn("tree_source", out[1])
            self.assertEqual(c, collections.Counter({"rows": 2, "trees_replaced": 1, "tree_nodes": 2,
                                                     "no_tree_final_only": 1, "probe_subqs_dropped": 1,
                                                     "node_rows_dropped": 1}))

    def test_post_build_checks(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "train.jsonl"
            arc = {"item_id": "arc", "source": "arc_challenge", "prompt": "Question: Q?\nOptions:\nA) a\nB) b",
                   "subqs": [{"id": "n1", "question": "q?", "p_cot": 0.9}]}
            leaky = json.loads(json.dumps(KK))
            leaky["item_id"] = "kk-leak"
            leaky["subqs"] = leaky["subqs"][:1] + [{"id": "zz", "question": "Is option A correct?", "truth": True,
                                                    "p_cot": 1.0, "depends_on": []}]
            broken = dict(KK, item_id="kk-broken", prompt=KK["prompt"].replace("\n\nQuestion: ", "\n\nAsk: "))
            jl(p, [KK, arc, leaky, broken])
            fails = []
            st = b8.gfps_stats(p, {"knights_knaves"}, "train", fails)
            why = {i: w for i, w in fails}
            self.assertEqual(set(why), {"arc", "kk-leak", "kk-broken"})
            self.assertIn("not GFP-S-eligible", why["arc"])
            self.assertIn("view_leak", why["kk-leak"])
            kk = st["knights_knaves"]
            n = len([sq for sq in KK["subqs"] if "id" in sq and b8._node_p(sq, "cot") is not None])
            self.assertEqual((kk["rows"], kk["items_with_tree"]), (3, 2))
            self.assertEqual(kk["nodes"], n + 2)
            fails = []
            b8.gfps_stats(p, {"knights_knaves"}, "val", fails)        # val keeps trees of every source
            self.assertNotIn("arc", {i for i, _ in fails})


if __name__ == "__main__":
    unittest.main()
