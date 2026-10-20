"""The built tree variants of the failure-attribution student data, on the real files (each check is skipped until its
directories exist):
  _t3   scripts/build_fa_sf.py / build_fa_aftraj.py / build_fa_real.py --trees <post-check copies> --trees-addon
        <post-check add-on copies> --pointer-render free (v2 tree + clean nodes, index-free render)
  _t2   scripts/build_fa_new_sources.py --trees (agtr: the Diagnosis Error lines only), and _t2f with
        --pointer-render free --trees-suffix _t2f
Every row is its A2 row (and its _t2 row) byte for byte except the node fields (subqs, has_tree, tree_source,
fa.tree_teacher); the recorded md5s hold, also the _t2 / A2 md5s of the earlier builds (v2 byte-identity); every node
is a valid MC node with the expected render, index-free when free; a _t3 / _t2f item's v2 node ids are a subset of its
_t2 item's (render drops and post-check removals only), clean nodes only in _t3; the post-check removals are absent;
--tree-node-types "content,conflict,agent", "+clean" and all six types select exactly their nodes
(treeplan.full_tree_plan) and pass train_student's every-type-present check. CPU only, no model:

  flock <lock> ~/.venvs/sev/bin/python -m unittest tests.test_fa_student_t3
"""
import collections
import hashlib
import json
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cotdistill import fa_trees as ft  # noqa: E402
from cotdistill.treeplan import full_tree_plan, mc_check  # noqa: E402

D = ROOT / "data"
TYPES = ("answer_step", "provenance", "content", "conflict", "agent", "clean")
SELECTIONS = (("content", "conflict", "agent"), ("content", "conflict", "agent", "clean"), TYPES)
NODE_KEYS = {"subqs", "has_tree", "tree_source"}
# variant -> (its _t2 reference or None, A2 dir, render)
SETS = {"sf_alg_t3": ("sf_alg_t2", "sf_alg", "free"), "sf_alg_disj_t3": ("sf_alg_disj_t2", "sf_alg_disj", "free"),
        "sf_hc_t3": ("sf_hc_t2", "sf_hc", "free"), "sf_hc_disj_t3": ("sf_hc_disj_t2", "sf_hc_disj", "free"),
        "aftraj_t3": ("aftraj_t2", "aftraj", "free"), "real_t3": ("real_t2", "real", "free"),
        "agtr_diag_t2": (None, "agtr_diag", "index"), "agtr_diag_t2f": ("agtr_diag_t2", "agtr_diag", "free"),
        "wwpro_t2": (None, "wwpro", "index"), "wwpro_t2f": ("wwpro_t2", "wwpro", "free"),
        "ce_t2": (None, "ce", "index"), "ce_t2f": ("ce_t2", "ce", "free")}
T3 = [k for k in SETS if k.endswith("_t3")]


def built(name: str) -> bool:
    return (D / f"student_fa_{name}/train.jsonl").exists()


def rows(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def strip(it: dict) -> dict:
    out = {k: v for k, v in it.items() if k not in NODE_KEYS}
    out["fa"] = {k: v for k, v in it["fa"].items() if k != "tree_teacher"}
    return out


_CACHE: dict = {}


def load(name: str, split: str) -> list[dict]:
    key = (name, split)
    if key not in _CACHE:
        _CACHE[key] = rows(D / f"student_fa_{name}/{split}.jsonl")
    return _CACHE[key]


class TestBuiltVariants(unittest.TestCase):
    def names(self) -> list[str]:
        have = [n for n in SETS if built(n)]
        if not have:
            self.skipTest("no tree variant built yet")
        return have

    def test_rows_align_with_a2_and_t2(self):
        for name in self.names():
            ref, a2, _ = SETS[name]
            for sp in ("train", "val"):
                new, base = load(name, sp), load(a2, sp)
                self.assertEqual(len(new), len(base), (name, sp))
                for a, b in zip(new, base):
                    self.assertEqual(strip(a), strip(b), (name, sp, a["item_id"]))
                    self.assertEqual(a["prompt"], b["prompt"])
                if ref and built(ref):
                    old = load(ref, sp)
                    self.assertEqual(len(new), len(old), (name, sp))
                    for a, b in zip(new, old):
                        self.assertEqual(strip(a), strip(b), (name, sp, a["item_id"]))
                        self.assertEqual(a["tree_source"], b["tree_source"])
                        self.assertEqual(a["has_tree"], b["has_tree"], (name, a["item_id"]))   # no tree is lost
                        self.assertLessEqual({x["id"] for x in a["subqs"] if x["type"] != "clean"},
                                             {x["id"] for x in b["subqs"]}, (name, a["item_id"]))

    def test_nodes(self):
        for name in self.names():
            render = SETS[name][2]
            n_clean = 0
            for sp in ("train", "val"):
                for it in load(name, sp):
                    self.assertEqual(it["has_tree"], bool(it["subqs"]))
                    ids = [sq["id"] for sq in it["subqs"]]
                    self.assertEqual(len(ids), len(set(ids)))
                    for sq in it["subqs"]:
                        mc_check(sq)
                        self.assertIn(sq["type"], TYPES)
                        self.assertLessEqual(set(sq["depends_on"]), set(ids))
                        n_clean += sq["type"] == "clean"
                        if render == "free":
                            self.assertEqual(sq.get("render"), "free", (name, it["item_id"], sq["id"]))
                            self.assertTrue(ft.node_text_index_free(sq), (name, it["item_id"], sq["id"]))
                        else:
                            self.assertNotIn("render", sq)
            if name.endswith("_t3"):
                self.assertGreater(n_clean, 0, name)
            else:
                self.assertEqual(n_clean, 0, name)

    def test_summary_md5(self):
        for name in self.names():
            summ = json.loads((D / f"student_fa_{name}/summary.json").read_text())
            md = summ.get("md5") or summ["trees"]["md5"]
            mine = {p: h for p, h in md.items() if Path(p).parent == Path(f"data/student_fa_{name}")}
            self.assertEqual(set(mine), {f"data/student_fa_{name}/{s}.jsonl" for s in ("train", "val")}, name)
            for p, h in mine.items():
                self.assertEqual(md5(ROOT / p), h, p)
            mfile = D / f"student_fa_{name}/md5sums.txt"
            if mfile.exists():
                got = {tuple(x.split("  ")) for x in mfile.read_text().splitlines() if x}
                self.assertEqual(got, {(h, p) for p, h in mine.items()})

    def test_postcheck_removals_absent(self):
        for name in self.names():
            if not name.endswith("_t3"):
                continue
            args = json.loads((D / f"student_fa_{name}/summary.json").read_text())["args"]
            sub = name.split("_")[1] if name.startswith("sf_") else name.split("_")[0]
            removed = collections.defaultdict(set)
            for key in ("trees", "trees_addon"):
                p = Path(args[key].format(subset=sub))
                p = p if p.is_absolute() else ROOT / p
                if not p.exists():
                    self.skipTest(f"post-check copy {p} not found")
                for ln in rows(p):
                    for x in (ln.get("postcheck") or {}).get("removed", []):
                        removed[ln["traj_id"].split("/", 1)[1]].add(x["id"])
            self.assertTrue(removed, name)
            hit = 0
            for sp in ("train", "val"):
                for it in load(name, sp):
                    uid = it["fa"]["uid"]
                    key = uid.split("-", 1)[1] if name.startswith("sf_") else uid
                    if key in removed:
                        hit += 1
                        self.assertFalse(removed[key] & {sq["id"] for sq in it["subqs"]}, (name, uid))
            self.assertGreater(hit, 0, name)

    def test_node_type_selection(self):
        for name in self.names():
            if not name.endswith("_t3"):
                continue
            items = load(name, "train")
            have = {sq["type"] for it in items for sq in it["subqs"]}
            for sel in SELECTIONS:
                self.assertFalse(set(sel) - have, (name, sel))        # train_student: every listed type occurs
                asked = want = 0
                for it in items:
                    st: dict = {}
                    plan = full_tree_plan(it, cap=0, rng=random.Random(0), subq_target="truth",
                                          node_types=frozenset(sel), stats=st)
                    self.assertTrue(all(e["node"]["type"] in sel for e in plan))
                    asked += len(plan)
                    want += sum(sq["type"] in sel for sq in it["subqs"])
                self.assertEqual(asked, want, (name, sel))
                self.assertGreater(asked, 0)


class TestEarlierBuildsUnchanged(unittest.TestCase):
    """v2 byte-identity: the A2, _t2 and eval files of the earlier builds still have the md5s their A2 summaries
    recorded (the _t3 / _t2 builds never rewrite them)."""

    def test_recorded_md5(self):
        n = 0
        for d in ("sf_alg", "sf_hc", "aftraj", "real", "agtr", "wwpro", "ce"):
            p = D / f"student_fa_{d}/summary.json"
            if not p.exists():
                continue
            for f, h in json.loads(p.read_text())["md5"].items():
                if (ROOT / f).exists():
                    self.assertEqual(md5(ROOT / f), h, f)
                    n += 1
        if not n:
            self.skipTest("no earlier build")


if __name__ == "__main__":
    unittest.main()
