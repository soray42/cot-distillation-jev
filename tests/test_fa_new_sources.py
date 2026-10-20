"""scripts/build_fa_new_sources.py (AgenTracer / Who&When Pro / CORRECT-Error training pools for the LOFO study): the
registered test files are untouched; the pools are disjoint from every registered test set (the _sfclean rule on every
slice of the built files, an independent exact Task-line check, planted duplicates removed by the dedup and by the final
recheck); rows have the schema of a data/student_fa_aftraj row and pass fa_public.audit_item; no injection field,
label field or teacher-only line reaches a row; labels are valid positions; the teacher exports agree with the rows;
the sampling helpers; a deterministic rebuild (CORRECT: md5s equal the recorded ones; all three with FA_NEW_FULL=1).
CPU only:

  flock <lock> ~/.venvs/sev/bin/python -m unittest tests.test_fa_new_sources
"""
import collections
import importlib.util
import json
import os
import random
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill import fa_public as fp  # noqa: E402


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = load_script("build_fa_new_sources")
SRCS = tuple(B.SOURCES)
DIRS = {s: ROOT / f"data/student_fa_{s}" for s in SRCS}
BUILT = all((DIRS[s] / "summary.json").exists() for s in SRCS)
AFTRAJ_ROW = ROOT / "data/student_fa_aftraj/train.jsonl"


def rows(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def md5(path: Path) -> str:
    return B.md5(path)


_REFS = {}


def references():
    if not _REFS:
        R = B.load_references(None)
        R["index"] = B.ref_index(R["refs"])
        _REFS["R"] = R
    return _REFS["R"]


def task_line(prompt: str) -> str:
    """The rendered Task text of an item (between "Task: " and the first step line, without the reference line)."""
    head = prompt.split("\n\nStep 0 [", 1)[0]
    if "\n\nTask: " not in head:
        return ""
    return fp.norm_task(head.split("\n\nTask: ", 1)[1].split("\n" + fp.GT_PREFIX, 1)[0])


# ------------------------------------------------------------------------------------------------- helpers (synthetic)
class TestSampling(unittest.TestCase):
    def test_proportional(self):
        q = B.proportional({"a": 600, "b": 300, "c": 100}, 100, None)
        self.assertEqual(sum(q.values()), 100)
        self.assertEqual(q, {"a": 60, "b": 30, "c": 10})
        q = B.proportional({"a": 3000, "b": 300, "c": 100}, 100, 0.4)
        self.assertEqual(sum(q.values()), 100)
        self.assertLessEqual(q["a"], 40)
        q = B.proportional({"a": 3000, "b": 5}, 100, 0.4)            # the cap cannot be met: relaxed
        self.assertEqual(q, {"a": 95, "b": 5})
        self.assertEqual(sum(B.proportional({"a": 3, "b": 2}, 100, None).values()), 5)

    def _recs(self, n=400, seed=0):
        r = random.Random(seed)
        out = []
        for i in range(n):
            k = r.randint(2, 30)
            out.append({"uid": f"u{i}", "subset": r.choice("xyz"), "task": f"task {i // 3}", "dec": r.randrange(k),
                        "steps": [{"agent": "A", "text": "t"}] * k})
        return out

    def test_strat_pick_deterministic_and_sized(self):
        recs = self._recs()
        cuts = B.length_tercile_cuts(recs)
        a = B.strat_pick(recs, 120, "k", 0.4, cuts)
        b = B.strat_pick(list(reversed(recs)), 120, "k", 0.4, cuts)
        self.assertEqual(len(a), 120)
        self.assertEqual(sorted(r["uid"] for r in a), sorted(r["uid"] for r in b))
        self.assertLessEqual(max(collections.Counter(r["subset"] for r in a).values()), 48)
        self.assertEqual(len(B.strat_pick(recs, 1000, "k", 0.4, cuts)), len(recs))
        q = B.strat_pick(recs, 30, "k", None, cuts, quotas={"x": 20, "y": 10, "z": 0})
        self.assertEqual(collections.Counter(r["subset"] for r in q), {"x": 20, "y": 10})

    def test_split_pool_task_disjoint(self):
        import argparse
        args = argparse.Namespace(n_val=30, val_frac=0.15, n_train=200, seed=0, subset_cap={"t": 0.4})
        tr, va, info = B.split_pool(self._recs(), args, "t")
        self.assertEqual(len(va), 30)
        self.assertEqual(len(tr), 200)
        self.assertFalse({fp.norm_task(r["task"]) for r in tr} & {fp.norm_task(r["task"]) for r in va})
        self.assertTrue(info["train_subsampled"])


# ------------------------------------------------------------------------------------------------- planted duplicates
class TestPlanted(unittest.TestCase):
    """Copies of registered test tasks planted among pool records: the dedup (and the final recheck) removes each,
    keeps an unrelated control."""

    @classmethod
    def setUpClass(cls):
        cls.R = references()
        allr = cls.R["index"][0]
        boiler = fp.OverlapIndex([(r["uid"], r["task"]) for r in allr], df=5).boiler
        cls.picked = {}
        for r in allr:                                  # one reference per set with >= 12 non-boilerplate 8-grams
            if r["subset"] not in cls.picked and len(fp.grams(r["task"]) - boiler) >= 12:
                cls.picked[r["subset"]] = r

    def planted(self):
        out = []
        for k, (rs, r) in enumerate(sorted(self.picked.items())):
            out.append({"uid": f"plant-exact-{rs}", "task": r["task"], "ids": [], "subset": "p"})
            out.append({"uid": f"plant-embed-{rs}", "task": "Please answer in a concise format - use only a single "
                        "word or a number as your final answer.\n\n" + r["task"], "ids": [], "subset": "p"})
        ww = self.R["refs"]["whowhen"][0]
        out.append({"uid": "plant-id", "task": "zebra quartz fjord nymph " * 6, "ids": fp.uuids(ww["task"]) or
                    [i for x in self.R["refs"]["whowhen"] for i in x["ids"]][:1], "subset": "p"})
        out.append({"uid": "control", "task": "Count the vowels in the word 'ornithopter' and multiply the count by "
                    "seventeen, then report the remainder after dividing by eleven.", "ids": [], "subset": "p"})
        return out

    def test_dedup_removes_planted(self):
        pool = self.planted()
        ov = B.overlap(pool, self.R, 5, 0.5)
        for r in pool:
            if r["uid"] == "control":
                self.assertNotIn("control", ov)
            else:
                self.assertIn(r["uid"], ov, r["uid"])
        for rs in self.picked:
            self.assertIn(rs, ov[f"plant-exact-{rs}"]["refsets"])
        self.assertEqual(set(self.picked), set(B.REGISTERED))       # every reference set was planted

    def test_recheck_removes_planted_from_val(self):
        import argparse
        pool = self.planted()
        for r in pool:
            r.update(steps=[{"agent": "A", "text": "x"}] * 3, dec=1)
        raw = {r["uid"]: r["task"] for r in pool}
        args = argparse.Namespace(boilerplate_df=5, overlap=0.5)
        tr, va, info = B.recheck([pool[-1]], pool[:-1], raw, self.R, args)
        self.assertEqual([r["uid"] for r in tr], ["control"])
        self.assertEqual(va, [])
        self.assertEqual(len(info["removed"]), len(pool) - 1)


# ------------------------------------------------------------------------------------------------- built files
@unittest.skipUnless(BUILT, "build scripts/build_fa_new_sources.py first")
class TestRegisteredUntouched(unittest.TestCase):
    def test_md5_equal_to_their_builders(self):
        recorded = {}
        recorded.update(json.loads((ROOT / "data/eval/fa_sf_x_summary.json").read_text())["md5"])
        for d in ("data/student_fa_sf_alg", "data/student_fa_sf_hc", "data/student_fa_aftraj", "data/student_fa_real"):
            recorded.update(json.loads((ROOT / d / "summary.json").read_text())["md5"])
        recorded["data/eval/fa_longrca_fit.jsonl"] = json.loads(
            (ROOT / "data/eval/fa_longrca_fit.summary.json").read_text())["md5"]
        here = json.loads((DIRS["ce"] / "summary.json").read_text())["references"]["files"]
        for p, h in here.items():
            self.assertEqual(md5(ROOT / p), h, p)                  # unchanged since the build
            if p in recorded:
                self.assertEqual(h, recorded[p], p)                # = what its own builder registered
        self.assertTrue({"data/eval/fa_sf_ww_alg.jsonl", "data/eval/fa_sf_x_agentracer_sfclean.jsonl",
                         "data/eval/fa_sf_x_wwpro_sfclean.jsonl", "data/eval/fa_sf_x_correct_sfclean.jsonl",
                         "data/eval/fa_aftraj_test.jsonl", "data/eval/fa_real_test.jsonl", "data/eval/fa_rse.jsonl",
                         "data/eval/fa_longrca_fit.jsonl"} <= set(here) & set(recorded))


@unittest.skipUnless(BUILT, "build scripts/build_fa_new_sources.py first")
class TestDisjoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.R = references()
        cls.items = {s: {sp: rows(DIRS[s] / f"{sp}.jsonl") for sp in ("train", "val")} for s in SRCS}
        cls.recs = {}
        for s in SRCS:
            us = {it["fa"]["uid"] for sp in ("train", "val") for it in cls.items[s][sp]}
            cls.recs[s] = {r["uid"]: r for r in B.stream(ROOT / B.SOURCES[s]["norm"])
                           if r["uid"] in us and r["split"] != "test"}

    def test_no_overlap_any_slice(self):
        for s in SRCS:
            sl = {sp: [self.recs[s][it["fa"]["uid"]] for it in self.items[s][sp]] for sp in ("train", "val")}
            for name, q in (("train+val", sl["train"] + sl["val"]), ("train", sl["train"]), ("val", sl["val"])):
                ov = B.overlap(q, self.R, 5, 0.5)
                self.assertEqual(ov, {}, f"{s} {name}: {list(ov.items())[:2]}")

    def test_not_in_own_test_split(self):
        held = self.R["held"]
        for s in SRCS:
            self.assertFalse(set(self.recs[s]) & held[s], s)
            for f in B.SOURCES[s]["test_files"]:
                self.assertFalse({x["fa"]["uid"] for x in rows(ROOT / f)} & set(self.recs[s]), f)

    def test_task_lines_independent(self):
        """No pool row shows the same Task line as a row of any registered test file (parsed from the prompts)."""
        test_tasks = set()
        for p in B.FILE_REFSET:
            test_tasks |= {task_line(x["prompt"]) for x in rows(ROOT / p)}
        test_tasks.discard("")
        for s in SRCS:
            for sp in ("train", "val"):
                mine = {task_line(x["prompt"]) for x in self.items[s][sp]}
                self.assertNotIn("", mine)
                self.assertFalse(mine & test_tasks, f"{s}/{sp}: {sorted(mine & test_tasks)[:2]}")

    def test_val_train_task_disjoint(self):
        for s in SRCS:
            tv = {fp.norm_task(self.recs[s][x["fa"]["uid"]]["task"]) for x in self.items[s]["val"]}
            tt = {fp.norm_task(self.recs[s][x["fa"]["uid"]]["task"]) for x in self.items[s]["train"]}
            self.assertFalse(tv & tt, s)


@unittest.skipUnless(BUILT and AFTRAJ_ROW.exists(), "build scripts/build_fa_new_sources.py first")
class TestRows(unittest.TestCase):
    BANNED_KEYS = re.compile(r"inject|^is_|phase|ledger|mistake_reason|ground_truth|replay|teacher_only", re.I)

    @classmethod
    def setUpClass(cls):
        with open(AFTRAJ_ROW) as f:
            cls.ref = json.loads(f.readline())
        cls.items = {s: rows(DIRS[s] / "train.jsonl") + rows(DIRS[s] / "val.jsonl") for s in SRCS}

    def keys(self, x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                yield f"{path}/{k}", k
                yield from self.keys(v, f"{path}/{k}")
        elif isinstance(x, list):
            for v in x:
                yield from self.keys(v, path)

    def test_schema_parity(self):
        ref = self.ref
        node_keys = {k for n in ref["subqs"] for k in n}
        for s in SRCS:
            for it in self.items[s]:
                self.assertEqual(set(it), set(ref), it["item_id"])
                self.assertEqual(set(it["fa"]), set(ref["fa"]), it["item_id"])
                for k, v in ref.items():
                    self.assertIs(type(it[k]), type(v), f"{it['item_id']} {k}")
                self.assertEqual(it["source"], "fa_public")
                self.assertEqual(it["tree_source"], "fa_public_labels")
                self.assertTrue(it["group"] in (f"fa_sf/{s}/step", f"fa_sf/{s}/step_gt"), it["group"])
                self.assertTrue(re.fullmatch(rf"fasf-{s}-(train|val)-\d{{5}}", it["item_id"]), it["item_id"])
                for n in it["subqs"]:
                    self.assertLessEqual(set(n), node_keys | {"agent", "category", "mode", "step"}, n)
            self.assertEqual(len({it["item_id"] for it in self.items[s]}), len(self.items[s]))
            self.assertEqual(len({it["fa"]["uid"] for it in self.items[s]}), len(self.items[s]))

    def test_no_injection_or_teacher_fields(self):
        for s in SRCS:
            for it in self.items[s]:
                for path, k in self.keys(it):
                    self.assertFalse(self.BANNED_KEYS.search(k), f"{it['item_id']}: key {path}")
                self.assertNotIn("[teacher only]", it["prompt"])
                self.assertEqual(fp.audit_item(it), [], it["item_id"])
                for n in it["subqs"]:
                    self.assertFalse(fp.NODE_LEAK.search(n["question"]), n["question"])
                for w in ("is_injected", "injection_via", "is_replayed", '"phase":', '"ledger":', "accepted_pred",
                          "mistake_reason"):
                    self.assertNotIn(w, it["prompt"], it["item_id"])                # raw field names / JSON keys

    def test_label_positions(self):
        for s in SRCS:
            for it in self.items[s]:
                fa, labs = it["fa"], it["labels"]
                g = fa["decisive_step"]
                self.assertTrue(2 <= fa["n_steps"] <= 62 and len(labs) == fa["n_steps"])
                self.assertTrue(0 <= g < fa["n_steps"])
                self.assertEqual(it["gold_label"], labs[g])
                self.assertEqual(labs, list(fp.SYMBOLS[:len(labs)]))
                self.assertEqual(it["teacher"], {L: float(i == g) for i, L in enumerate(labs)})
                self.assertEqual(it["label_names"][it["gold_label"]], f"Step {g} [{fa['decisive_agent']}]")
                self.assertIn(f"\nStep {g} [{fa['decisive_agent']}]: ", it["prompt"])

    def test_budget_and_reference_line(self):
        for s in SRCS:
            summ = json.loads((DIRS[s] / "summary.json").read_text())
            self.assertEqual(summ["tokens_train"]["over_budget"], 0)
            gt = [it["with_gt"] for it in self.items[s]]
            if s in ("ce", "agtr"):
                self.assertFalse(any(gt))                            # no reference answers (agentic, CORRECT)
            else:
                self.assertTrue(0.35 < sum(gt) / len(gt) < 0.65)
                self.assertTrue(all(fp.GT_PREFIX in it["prompt"] for it in self.items[s] if it["with_gt"]))

    def test_summary_md5(self):
        for s in SRCS:
            summ = json.loads((DIRS[s] / "summary.json").read_text())
            for p, h in summ["md5"].items():
                self.assertEqual(md5(ROOT / p), h, p)
            lines = (DIRS[s] / "md5sums.txt").read_text().split("\n")
            self.assertEqual({tuple(x.split("  ")) for x in lines if x}, {(h, p) for p, h in summ["md5"].items()})
            self.assertEqual(summ["sizes"]["train"], len(rows(DIRS[s] / "train.jsonl")))

    def test_diag_variant_is_line_subset(self):
        for sp in ("train", "val"):
            full = (DIRS["agtr"] / f"{sp}.jsonl").read_text().splitlines()
            diag = (ROOT / f"data/student_fa_agtr_diag/{sp}.jsonl").read_text().splitlines()
            self.assertEqual(diag, [x for x in full if json.loads(x)["fa"]["benchmark"] == "Diagnosis Error"])


@unittest.skipUnless(BUILT, "build scripts/build_fa_new_sources.py first")
class TestTeacherExport(unittest.TestCase):
    def test_exports_views_shards(self):
        from cotdistill import fa_trees as ft
        for s in SRCS:
            summ = json.loads((DIRS[s] / "summary.json").read_text())
            items = {it["fa"]["uid"]: it for sp in ("train", "val") for it in rows(DIRS[s] / f"{sp}.jsonl")}
            tb = ROOT / f"data/raw/{s}_trees"
            ids = json.loads((tb / "all_ids.json").read_text())
            self.assertEqual({x["traj_id"] for x in ids}, {f"{s}/{u}" for u in items})
            shards = [json.loads(p.read_text()) for p in sorted((tb / "shards").glob("shard_*.json"))]
            self.assertEqual(sorted(x["traj_id"] for sh in shards for x in sh), sorted(x["traj_id"] for x in ids))
            self.assertTrue(all(len(sh) <= 90 for sh in shards))
            self.assertEqual(len(shards), summ["export"]["shards"])
            for x in ids:
                u = x["traj_id"].split("/", 1)[1]
                it = items[u]
                t = ft.load_traj(ROOT / f"data/raw/{s}_sf/{B.SOURCES[s]['dir']}/train/{u}.json", s)
                self.assertEqual(t["mistake_step"], it["fa"]["decisive_step"])
                self.assertEqual(len(t["steps"]), it["fa"]["n_steps"])
                self.assertEqual(x["mistake_step"], it["fa"]["decisive_step"])
                view = (tb / "views" / f"{s}__{u}.txt").read_text()
                self.assertEqual(view, ft.render_view(t) + "\n")
                teach = [ln for ln in view.split("\n") if ln.startswith("[teacher only]")]
                self.assertEqual(len(teach), 3)
                self.assertIn(f"Decisive step: Step {t['mistake_step']} [", teach[1])
                if s == "wwpro":
                    self.assertFalse(re.search(r"inject", " ".join(teach), re.I), u)
                for w in ("is_injected", "injection_via", '"phase":', '"ledger":', "mistake_reason"):
                    self.assertNotIn(w, view, u)                                    # raw field names / JSON keys
                if s == "agtr":
                    self.assertFalse(B.AGTR_META.search(view), u)
        blocks = json.loads((DIRS["agtr"] / "summary.json").read_text())["export"]["shard_blocks"]
        self.assertEqual(blocks["Diagnosis Error"], list(range(len(blocks["Diagnosis Error"]))))


@unittest.skipUnless(BUILT and (ROOT / "data/student_fa_lofo/plan.json").exists(), "build first")
class TestLofoPlan(unittest.TestCase):
    def test_folds(self):
        plan = json.loads((ROOT / "data/student_fa_lofo/plan.json").read_text())
        for fold, (own, tests) in B.FOLDS.items():
            f = plan["folds"][fold]
            self.assertFalse(set(own) & set(f["train_pools"]), fold)
            self.assertEqual(f["train"], sum(p["train"] for p in f["pools"].values()))
            self.assertEqual(set(f["held_out_tests"]), set(tests + B.ALWAYS_HELD))
        for s in SRCS:
            self.assertEqual(sum(plan["pool_overlaps_registered_files"][s].values()), 0, s)


# ------------------------------------------------------------------------------------------------- determinism
@unittest.skipUnless(BUILT, "build scripts/build_fa_new_sources.py first")
class TestDeterministic(unittest.TestCase):
    def rebuild(self, sources: str):
        tmp = Path(tempfile.mkdtemp(dir=ROOT, prefix=".verify_newsrc_"))
        try:
            B.main(["--sources", sources, "--out-root", str(tmp / "data"), "--raw-out", str(tmp / "raw"), "--no-lofo"])
            out = {}
            for s in sources.split(","):
                rec = json.loads((DIRS[s] / "summary.json").read_text())
                for p, h in rec["md5"].items():
                    q = tmp / p.replace("data/raw/", "raw/", 1) if p.startswith("data/raw/") else tmp / p
                    out[p] = (md5(q), h)
                v = json.loads((tmp / f"data/student_fa_{s}/summary.json").read_text())["export"]["views_md5"]
                out[f"{s}/views"] = (v, rec["export"]["views_md5"])
            return out
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_rebuild_ce(self):
        for p, (a, b) in self.rebuild("ce").items():
            self.assertEqual(a, b, p)

    @unittest.skipUnless(os.environ.get("FA_NEW_FULL"), "set FA_NEW_FULL=1 for the full rebuild (~2 min)")
    def test_rebuild_all(self):
        for p, (a, b) in self.rebuild(",".join(SRCS)).items():
            self.assertEqual(a, b, p)


# ------------------------------------------------------------------------------------------------- --trees
@unittest.skipUnless(BUILT and (ROOT / "data/raw/ce_trees/ce.jsonl").exists(), "build first; needs the ce trees")
class TestTreesPath(unittest.TestCase):
    """--trees (build_fa_sf.tree_items, as build_fa_aftraj): the A2 files stay byte-identical to the recorded build;
    the _t2 rows are the A2 rows with the tree's MC nodes (final questions byte-identical, refused / missing trees
    keep no nodes); --pointer-render free --trees-suffix _t2f renders the same trees index-free."""

    @classmethod
    def setUpClass(cls):
        from cotdistill import fa_trees as ft
        cls.ft = ft
        cls.tmp = Path(tempfile.mkdtemp(dir=ROOT, prefix=".verify_newsrc_trees_"))
        common = ["--sources", "ce", "--out-root", str(cls.tmp / "data"), "--raw-out", str(cls.tmp / "raw"),
                  "--no-lofo", "--no-export", "--trees", "data/raw/{subset}_trees/{subset}.jsonl"]
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            cls.out = B.main(common)
            cls.out_f = B.main(common + ["--pointer-render", "free", "--trees-suffix", "_t2f"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_a2_byte_identical(self):
        rec = json.loads((DIRS["ce"] / "summary.json").read_text())
        n = 0
        for p, h in rec["md5"].items():
            if p.startswith("data/student_fa_ce/"):
                self.assertEqual(md5(self.tmp / p), h, p)
                n += 1
        self.assertEqual(n, 2)

    def test_t2_rows(self):
        lines = {json.loads(x)["traj_id"].split("/", 1)[1]
                 for x in (ROOT / "data/raw/ce_trees/ce.jsonl").read_text().splitlines() if x.strip()}
        from cotdistill.treeplan import mc_check
        for suffix, out in (("_t2", self.out), ("_t2f", self.out_f)):
            t = out["sources"]["ce"]["trees"]
            self.assertEqual(t["dir"], str((self.tmp / f"data/student_fa_ce{suffix}").relative_to(ROOT)))
            self.assertEqual(t["lines_without_row"], 0)
            with_tree = 0
            for sp in ("train", "val"):
                a2 = rows(DIRS["ce"] / f"{sp}.jsonl")
                t2 = rows(self.tmp / f"data/student_fa_ce{suffix}/{sp}.jsonl")
                self.assertEqual(len(a2), len(t2))
                for a, b in zip(a2, t2):
                    self.assertEqual(a["prompt"], b["prompt"])
                    skip = ("subqs", "has_tree", "tree_source", "fa")
                    self.assertEqual({k: v for k, v in a.items() if k not in skip},
                                     {k: v for k, v in b.items() if k not in skip})
                    self.assertEqual({k: v for k, v in a["fa"].items()},
                                     {k: v for k, v in b["fa"].items() if k != "tree_teacher"})
                    self.assertEqual(b["tree_source"], "fa_sf_trees")
                    self.assertEqual(b["has_tree"], bool(b["subqs"]))
                    if b["has_tree"]:
                        self.assertIn(b["fa"]["uid"], lines)
                    with_tree += b["has_tree"]
                    for sq in b["subqs"]:
                        mc_check(sq)
                        if suffix == "_t2f":
                            self.assertEqual(sq["render"], "free")
                            self.assertTrue(self.ft.node_text_index_free(sq))
                        else:
                            self.assertNotIn("render", sq)
            self.assertEqual(with_tree, t["items_with_tree"])
            self.assertGreater(with_tree, 0.9 * len(lines))
            summ = json.loads((self.tmp / f"data/student_fa_ce{suffix}/summary.json").read_text())
            for p, h in summ["trees"]["md5"].items():
                self.assertEqual(md5(ROOT / p), h, p)
        self.assertLessEqual(self.out_f["sources"]["ce"]["trees"]["items_with_tree"],
                             self.out["sources"]["ce"]["trees"]["items_with_tree"])

    def test_built_files_match(self):
        """The data/student_fa_ce_t2 / _t2f files in the repo (when copied there) are this build's."""
        for suffix in ("_t2", "_t2f"):
            d = ROOT / f"data/student_fa_ce{suffix}"
            if not d.exists():
                continue
            for sp in ("train", "val"):
                self.assertEqual(md5(d / f"{sp}.jsonl"), md5(self.tmp / f"data/student_fa_ce{suffix}/{sp}.jsonl"))


if __name__ == "__main__":
    unittest.main()
