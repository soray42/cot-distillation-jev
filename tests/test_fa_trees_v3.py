"""Tree v3 (src/cotdistill/fa_trees.py v3 section, scripts/extract_fa_trees.py v3-*, the builders' --trees-addon /
--pointer-render / --trees-suffix): v2 byte-identity, the index-free render (no step numbers, one excerpt rule for the
correct option and the distractors, distinct options, references in questions, dropped nodes re-attached), the
clean-node targets and checks (a node at / after the decisive step, a quote not in the step, an evaluative giveaway,
length rank, stem match, targets), the builder merge by traj_id and --tree-node-types selection. Synthetic data, CPU
only, no model:

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/sev/bin/python -m unittest tests.test_fa_trees_v3
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import random
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from cotdistill import fa_public as fp  # noqa: E402
from cotdistill import fa_trees as ft  # noqa: E402
from cotdistill.treeplan import full_tree_plan, mc_check  # noqa: E402

import test_build_fa_sf as TB  # noqa: E402
import test_fa_trees as TT  # noqa: E402


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_v3test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


X = load_script("extract_fa_trees")

HIST = [
    ("Planner", "assistant", "Plan: load the sheet, keep only Blu-Ray rows, report the oldest title."),
    ("Computer_terminal", "user", "exitcode: 0 (execution succeeded)\nCode output: rows=25"),
    ("Coder", "assistant", "Loaded the sheet; the Title column holds section markers DVD, Blu-Ray and VHS."),
    ("Coder", "assistant", "Counted the rows per section: DVD 9, Blu-Ray 8 and VHS 8, sorted by year."),
    ("Analyst", "assistant", "Let us merge DVD and Blu-Ray into one Disc group and pick the oldest Disc item."),
    ("Coder", "assistant", "Wrote code that sorts the Disc group by year and prints its first title."),
    ("Computer_terminal", "user", "exitcode: 0 (execution succeeded)\nCode output: Cereal Killer III"),
    ("Analyst", "assistant", "Final answer: Cereal Killer III"),
    ("Planner", "assistant", "TERMINATE"),
]
MS = 4


def make_raw(hist=HIST, ms: int = MS, gt: str = "Time-Parking 2") -> dict:
    return {"question": "What is the title of the oldest Blu-Ray in the spreadsheet? Return it as written.",
            "ground_truth": gt, "mistake_agent": hist[ms][0], "mistake_step": ms, "mistake_reason": "merged formats",
            "history": [{"step_id": i, "role": r, "name": n, "content": c} for i, (n, r, c) in enumerate(hist)]}


def v2_source(traj: dict) -> dict:
    """A v2 source tree (passes every v2 check): answer 7 -> output 6 -> code 5 -> merge 4."""
    return {"traj_id": traj["traj_id"], "teacher": "fake", "answer_step": 7, "nodes": [
        {"id": "n1", "type": "answer_step", "question": "Which step states the team's final answer?",
         "answer_step": 7, "distractor_steps": [6, 2, 8], "quote": "Final answer: Cereal Killer III", "parents": []},
        {"id": "n2", "type": "provenance", "question": "Step 7 [Analyst] repeats a printed title. Which step printed it?",
         "answer_step": 6, "distractor_steps": [1, 3, 5], "quote": "Code output: Cereal Killer III",
         "parents": ["n1"]},
        {"id": "n3", "type": "provenance", "question": "That output comes from code. Which step wrote the code?",
         "answer_step": 5, "distractor_steps": [2, 3, 0], "quote": "sorts the Disc group by year",
         "parents": ["n2"]},
        {"id": "n4", "type": "provenance", "question": "The code works on a merged group. Which step proposed the merge?",
         "answer_step": 4, "distractor_steps": [0, 2, 3], "quote": "merge DVD and Blu-Ray into one Disc group",
         "parents": ["n3"]},
        {"id": "n5", "type": "content", "question": "What did Step 4 [Analyst] propose?",
         "correct": "Pool two formats into one group and take its oldest item",
         "distractors": ["Keep only the Blu-Ray rows and report the oldest title",
                         "Read the section markers DVD, Blu-Ray and VHS from the Title column",
                         "Sort the merged group by year and print the first title"],
         "cite_step": 4, "quote": "merge DVD and Blu-Ray into one Disc group", "parents": ["n4"]},
        {"id": "n6", "type": "conflict", "question": "Why does that proposal fail the task?",
         "correct": "The task asks for the oldest Blu-Ray only, so DVDs must not compete",
         "distractors": ["The sheet has no section markers, so formats cannot be told apart",
                         "The answer must be given in upper case letters in the reply",
                         "The terminal could not open the spreadsheet file at that path"],
         "cite_step": None, "quote": "oldest Blu-Ray in the spreadsheet", "parents": ["n5"]},
        {"id": "n7", "type": "agent", "question": "Which agent proposed the merge?", "answer_agent": "Analyst",
         "distractor_agents": ["Planner", "Coder"], "parents": ["n4"]}]}


def traj_of(raw: dict, subset: str = "alg", stem: str = "trajectory_0000") -> dict:
    with tempfile.TemporaryDirectory() as td:
        return ft.load_traj(TT.write_raw(Path(td), subset, stem, raw), subset)


def v2_line(traj: dict, src: dict | None = None) -> dict:
    rec = ft.output_record(traj, src or v2_source(traj))
    assert rec["checks"]["pass"], rec["checks"]["failures"]
    return rec


def by_id(nodes: list[dict]) -> dict:
    return {nd["id"]: nd for nd in nodes}


OPT = re.compile("^([A-Za-z_]+): “([^”]*)”$")


class TestV2Unchanged(unittest.TestCase):
    def test_output_record_golden(self):
        """A v2 record of tests/test_fa_trees.py's fixture is byte-identical to the one recorded before the v3 change."""
        t = TT.traj_from(TT.make_raw())
        rec = ft.output_record(t, TT.fake_teacher(t), 20261010, {"tercile": 1})
        s = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        self.assertEqual(hashlib.md5(s.encode()).hexdigest(), "534463064b375dca2411d3d50c4b5ad5")


class TestFreeRender(unittest.TestCase):
    def setUp(self):
        self.t = traj_of(make_raw())
        self.line = v2_line(self.t)
        self.st = collections.Counter()
        self.free = ft.free_render(self.t, self.line["nodes"], stats=self.st)

    def test_no_step_numbers_anywhere(self):
        self.assertEqual(len(self.free), len(self.line["nodes"]))
        for nd in self.free:
            self.assertEqual(nd["render"], "free")
            self.assertTrue(ft.node_text_index_free(nd), nd)
            for t in [nd["question"]] + nd["options"]:
                self.assertIsNone(re.search(r"(?i)\bstep\s*\d", re.sub("“[^”]*”", "", t)), t)

    def test_pointer_options_uniform_rule(self):
        """Every pointer option is '<Agent>: “<first W words>…”' with ONE W for the correct option and the distractors,
        the smallest W >= 6 that identifies each option step among all steps; options distinct; answers unchanged."""
        orig = by_id(self.line["nodes"])
        words = [ft.step_words(s["text"])[0] for s in self.t["steps"]]
        for nd in self.free:
            if nd["type"] not in ft.POINTER_TYPES:
                continue
            self.assertEqual(nd["option_steps"], orig[nd["id"]]["option_steps"])
            self.assertEqual(nd["answer"], orig[nd["id"]]["answer"])
            self.assertEqual(len({o.casefold() for o in nd["options"]}), len(nd["options"]))
            widths = set()
            for o, k in zip(nd["options"], nd["option_steps"]):
                m = OPT.match(o)
                self.assertIsNotNone(m, o)
                self.assertEqual(m.group(1), self.t["steps"][k]["agent"])
                ex = m.group(2)
                w = ex.rstrip(ft.ELLIPSIS).split()
                self.assertEqual(w, words[k][:len(w)])
                self.assertEqual(ex.endswith(ft.ELLIPSIS), len(words[k]) > len(w))
                widths.add(len(w) if ex.endswith(ft.ELLIPSIS) else None)
            widths.discard(None)
            self.assertLessEqual(len(widths), 1, nd)
        # n2's options include the two terminal outputs, identical for their first 6 words: W = 7 for every option
        n2 = by_id(self.free)["n2"]
        self.assertEqual(n2["options"][n2["option_steps"].index(1)],
                         "Computer_terminal: “exitcode: 0 (execution succeeded) Code output: rows=25”")
        self.assertIn("Planner: “TERMINATE”", by_id(self.free)["n1"]["options"])
        self.assertEqual(self.st["free_width/7"], 2)

    def test_references_in_questions(self):
        f = by_id(self.free)
        self.assertEqual(f["n5"]["question"],
                         "What did the Analyst step “Let us merge DVD and Blu-Ray…” propose?")
        self.assertTrue(f["n2"]["question"].startswith("The Analyst step “Final answer: Cereal Killer III”"
                                                       " repeats"))
        self.assertEqual(f["n2"]["free_steps"], [1, 3, 5, 6, 7])
        self.assertEqual(f["n6"]["options"], by_id(self.line["nodes"])["n6"]["options"])   # no reference: unchanged

    def test_duplicate_steps_dropped_and_children_relinked(self):
        hist = copy.deepcopy(HIST)
        hist[8] = ("Analyst", "assistant", "Final answer: Cereal Killer III")       # = step 7, same agent
        t = traj_of(make_raw(hist))
        src = v2_source(t)
        src["rule_override"] = "the rule picks the repeat in the last step"
        line = v2_line(t, src)
        st = collections.Counter()
        free = by_id(ft.free_render(t, line["nodes"], stats=st))
        self.assertNotIn("n1", free)                                       # options 7 and 8 cannot be told apart
        self.assertNotIn("n2", free)                                       # its question refers to step 7
        self.assertEqual(free["n3"]["parents"], [])                        # n2 -> n1 -> no parents
        self.assertEqual(free["n4"]["parents"], ["n3"])
        self.assertEqual(free["n5"]["parents"], ["n4"])
        self.assertEqual(st["free_drop/free_ambiguous"], 2)

    def test_excerpt_with_a_step_index_is_dropped(self):
        hist = copy.deepcopy(HIST)
        hist[0] = ("Planner", "assistant", "Step 1: load the sheet; step 2: keep only Blu-Ray rows.")
        t = traj_of(make_raw(hist))
        st = collections.Counter()
        free = by_id(ft.free_render(t, v2_line(t)["nodes"], stats=st))
        self.assertNotIn("n3", free)
        self.assertNotIn("n4", free)
        self.assertEqual(st["free_drop/free_excerpt_index"], 2)
        for nd in free.values():
            self.assertTrue(ft.node_text_index_free(nd))

    def test_self_leak_and_index_left(self):
        src = v2_source(self.t)
        src["nodes"][3]["question"] = "Which step said 'Let us merge DVD and Blu-Ray' first?"
        src["nodes"][4]["question"] = "What did steps 4 and 5 do together?"
        line = v2_line(self.t, src)
        st = collections.Counter()
        free = by_id(ft.free_render(self.t, line["nodes"], stats=st))
        self.assertNotIn("n4", free)
        self.assertNotIn("n5", free)
        self.assertEqual(st["free_drop/free_self_leak"], 1)
        self.assertEqual(st["free_drop/free_index_left"], 1)

    def test_text_option_rewrite_keeps_rank_or_drops(self):
        src = v2_source(self.t)
        # a short distractor that names a step: rewritten, the correct option stays longest -> kept
        src["nodes"][5]["distractors"][1] = "Step 5 [Coder] sorted by year"
        line = v2_line(self.t, src)
        st = collections.Counter()
        free = by_id(ft.free_render(self.t, line["nodes"], stats=st))
        self.assertIn("n6", free)
        self.assertIn("The Coder step “Wrote code that sorts the Disc…” sorted by year", free["n6"]["options"])
        # the excerpt of step 2 has a semicolon: the rewritten option becomes the odd form -> dropped
        src["nodes"][5]["distractors"][1] = "Step 2 [Coder] named VHS"
        st = collections.Counter()
        free = by_id(ft.free_render(self.t, v2_line(self.t, src)["nodes"], stats=st))
        self.assertNotIn("n6", free)
        self.assertEqual(st["free_drop/free_option_form"], 1)
        # the correct option names a step: its rewrite makes it the longest -> dropped
        src["nodes"][5]["correct"] = "Step 0 [Planner] said Blu-Ray only"
        line = v2_line(self.t, src)
        st = collections.Counter()
        free = by_id(ft.free_render(self.t, line["nodes"], stats=st))
        self.assertNotIn("n6", free)
        self.assertEqual(st["free_drop/free_option_rank"], 1)

    def test_student_cap(self):
        """At a small per-step cap the excerpt comes from the visible head of the cut text."""
        hist = copy.deepcopy(HIST)
        hist[5] = ("Coder", "assistant", "Wrote code that sorts the Disc group by year " + "z" * 400 + " tail")
        t = traj_of(make_raw(hist))
        free = by_id(ft.free_render(t, v2_line(t)["nodes"], cap=60))
        o = free["n3"]["options"][free["n3"]["option_steps"].index(5)]
        self.assertEqual(o, "Coder: “Wrote code that sorts the Disc…”")
        self.assertEqual(ft.step_words("abcdef ghij klmnop", 12), (["abcdef"], True))      # head "abcdef g" -> 1 word
        self.assertEqual(ft.step_words("abcdef gh klmnop", 14), (["abcdef", "gh"], True))   # head ends at a word end
        self.assertEqual(ft.step_words("abcdef gh  klmnop", 15), (["abcdef", "gh"], True))  # head ends at a space


def clean_src(traj: dict, k1: int = 2, k2: int = 3) -> dict:
    return {"traj_id": traj["traj_id"], "teacher": "fake", "nodes": [
        {"id": "c1", "type": "clean", "cite_step": k1,
         "correct": "Found DVD, Blu-Ray and VHS markers in the Title field",
         "distractors": ["Kept only the Blu-Ray rows of the sheet",
                         "Printed twenty-five rows from the terminal session",
                         "Sorted all rows of every format by release year first"],
         "quote": "section markers DVD, Blu-Ray and VHS"},
        {"id": "c2", "type": "clean", "cite_step": k2,
         "correct": "Said there were eight Blu-Ray rows",
         "distractors": ["Said there were nine Blu-Ray rows",
                         "Said the VHS rows outnumbered the DVD rows",
                         "Said that DVD and VHS had the same count"],
         "quote": "DVD 9, Blu-Ray 8 and VHS 8"}]}


def entry(k1: int = 2, k2: int = 3, r1: int = 2, r2: int = 1) -> dict:
    return {"traj_id": "alg/trajectory_0000", "targets": [{"step": k1, "target_len_rank": r1},
                                                          {"step": k2, "target_len_rank": r2}]}


class TestCleanNodes(unittest.TestCase):
    def setUp(self):
        self.t = traj_of(make_raw())

    def check(self, src=None, ent=None) -> dict:
        return ft.check_addon(self.t, src or clean_src(self.t), ent or entry())

    def test_good_addon_passes(self):
        r = self.check()
        self.assertTrue(r["pass"], r["failures"])
        self.assertEqual([nd["type"] for nd in r["nodes"]], ["clean", "clean"])
        c1 = r["nodes"][0]
        self.assertEqual(c1["question"], "Which statement about Step 2 [Coder] is supported by the task and the log?")
        self.assertEqual(c1["parents"], [])
        mc_check(c1)
        self.assertEqual(r["length_rank"], {"c1": 2, "c2": 1})
        free = ft.free_render(self.t, r["nodes"])
        self.assertEqual(free[0]["question"], "Which statement about the Coder step “Loaded the sheet; the Title"
                                              " column…” is supported by the task and the log?")
        self.assertEqual(free[0]["options"], c1["options"])

    def test_not_before_decisive(self):
        for k in (MS, MS + 1):              # at and after the decisive step
            src = clean_src(self.t, k1=k)
            src["nodes"][0]["quote"] = self.t["steps"][k]["text"][:20]
            r = self.check(src, entry(k1=k))
            self.assertFalse(r["pass"])
            self.assertTrue(any(f.startswith("clean_not_before_decisive: c1") for f in r["failures"]), r["failures"])

    def test_quote_must_be_verbatim_in_the_step(self):
        src = clean_src(self.t)
        src["nodes"][0]["quote"] = "section markers DVD, BluRay and VHS"
        self.assertIn("quote_not_found: c1", self.check(src)["failures"])
        src["nodes"][0]["quote"] = "DVD 9, Blu-Ray 8 and VHS 8"                    # step 3's text, not step 2's
        self.assertIn("quote_not_found: c1", self.check(src)["failures"])

    def test_evaluative_giveaway(self):
        for word in ("Correctly found DVD, Blu-Ray and VHS markers", "Found the markers without any errors",
                     "Validly found DVD, Blu-Ray and VHS markers"):
            src = clean_src(self.t)
            src["nodes"][0]["correct"] = word
            self.assertTrue(any(f.startswith("evaluative_word: c1") for f in self.check(src)["failures"]), word)
        src = clean_src(self.t)
        src["nodes"][0]["distractors"][0] = "Kept only the Blu-Ray rows, which was the right call"   # "right" alone ok
        self.assertFalse(any(f.startswith("evaluative_word") for f in self.check(src)["failures"]))

    def test_length_rank_target_and_tell(self):
        r = self.check(ent=entry(r1=0))
        self.assertTrue(any(f.startswith("length_rank_miss: c1 rank 2-3, target 0") for f in r["failures"]))   # ties with a distractor
        src = clean_src(self.t)
        src["nodes"][1]["correct"] = "Said there were eight Blu-Ray rows, nine DVD rows and eight VHS rows " * 2
        r = self.check(src, entry(r2=3))
        self.assertTrue(any(f.startswith("length_tell: c2") for f in r["failures"]), r["failures"])

    def test_stem_match_tell(self):
        src = clean_src(self.t)
        src["nodes"][0]["correct"] = "Loaded the sheet with its Title column"
        r = self.check(src, entry(r1=0))
        self.assertTrue(any(f.startswith("stem_match_tell: c1") for f in r["failures"]), r["failures"])

    def test_targets_options_leaks(self):
        r = self.check(ent=entry(k2=1))
        self.assertTrue(any(f.startswith("addon_targets") for f in r["failures"]))
        src = clean_src(self.t)
        src["nodes"][0]["distractors"] = src["nodes"][0]["distractors"][:2]
        self.assertIn("option_count: c1 3", self.check(src)["failures"])
        src = clean_src(self.t)
        src["nodes"][0]["distractors"][2] = "Reported Time-Parking 2 as the oldest title"
        self.assertIn("gt_leak: c1", self.check(src)["failures"])
        src = clean_src(self.t)
        src["nodes"][0]["distractors"][2] = "Pointed back to Step 1 for the row count"
        self.assertTrue(any(f.startswith("clean_index: c1") for f in self.check(src)["failures"]))
        src = clean_src(self.t)
        src["nodes"][0]["distractors"][0] = "Found DVD, Blu-Ray and VHS markers in the Title field"
        self.assertIn("options_dup: c1", self.check(src)["failures"])

    def test_skip_and_schema(self):
        src = clean_src(self.t)
        src["nodes"][1] = {"id": "c2", "type": "clean", "cite_step": 3, "skip": "it miscounts the rows"}
        r = self.check(src)
        self.assertTrue(r["pass"], r["failures"])
        self.assertEqual(r["skips"], [{"id": "c2", "cite_step": 3, "reason": "it miscounts the rows"}])
        src["nodes"][1]["skip"] = " "
        self.assertFalse(self.check(src)["pass"])
        src = clean_src(self.t)
        src["nodes"][0]["id"] = "n9"
        self.assertTrue(self.check(src)["failures"][0].startswith("schema"))

    def test_shuffle_seeded_and_balanced(self):
        pos = collections.Counter()
        for i in range(200):
            src = clean_src(self.t)
            t = dict(self.t, traj_id=f"alg/trajectory_{i:04d}")
            nodes, _ = ft.assemble_addon(t, src)
            self.assertEqual(nodes, ft.assemble_addon(t, src)[0])
            pos.update(nd["answer"] for nd in nodes)
        self.assertEqual(sorted(pos), [0, 1, 2, 3])
        self.assertGreater(min(pos.values()), 70)                     # 400 nodes, 100 expected per index

    def test_targets(self):
        trajs = [self.t, dict(self.t, mistake_step=0)]
        prior = ft.position_prior(trajs)
        self.assertEqual(prior["abs"], {0: 0.5, 4: 0.5})
        sp = ft.clean_speakers([self.t, dict(self.t, mistake_step=1)])
        self.assertEqual(sp, {"Analyst"})                              # Computer_terminal is never a target speaker
        sp = {"Analyst", "Coder", "Planner"}
        tg = ft.clean_targets(self.t, prior, sp)
        self.assertEqual(tg, ft.clean_targets(self.t, prior, sp))      # seeded
        self.assertEqual(len(tg), 2)
        self.assertTrue(all(x["step"] < MS and x["agent"] != "Computer_terminal" for x in tg))
        self.assertEqual(sorted(x["why"] for x in tg), ["prior", "random"])
        self.assertEqual(ft.clean_targets(dict(self.t, mistake_step=0), prior, sp), [])
        self.assertEqual([x["step"] for x in ft.clean_targets(dict(self.t, mistake_step=1), prior, sp)], [0])
        # the prior pick follows the weights: a dominant position wins most draws
        prior = {"abs": {3: 1.0}, "end": {}}
        firsts = collections.Counter(next(x["step"] for x in ft.clean_targets(
            dict(self.t, traj_id=f"alg/t{i}"), prior, sp) if x["why"] == "prior") for i in range(100))
        self.assertGreater(firsts[3], 90)


class TestScriptV3(unittest.TestCase):
    """v3-targets, v3-check and v3-merge end to end on a temporary raw / trees layout (the module's path maps
    patched)."""

    def test_targets_views_shards_resumable(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            raw, tdir, out = d / "raw", d / "trees", d / "v3"
            tdir.mkdir()
            lines = []
            for i in range(6):
                TT.write_raw(raw, "alg", f"trajectory_{i:04d}", make_raw())
                t = ft.load_traj(raw / f"Algorithm-Generated/train/trajectory_{i:04d}.json", "alg")
                lines.append(json.dumps(v2_line(t)))
            for i, ms in enumerate((0, 2, 2, 3), 6):     # pool-only labels: Planner and Coder are decisive speakers too
                TT.write_raw(raw, "alg", f"trajectory_{i:04d}", make_raw(ms=ms))
            (tdir / "alg.jsonl").write_text("\n".join(lines) + "\n")
            old = (X.RAW_BY_SUBSET, X.TREES_DIR, X.V2_TREES)
            X.RAW_BY_SUBSET, X.TREES_DIR, X.V2_TREES = {"alg": raw}, {"alg": tdir}, {"alg": tdir / "alg.jsonl"}
            args = argparse.Namespace(trees="", out_dir=str(out), shards=4, seed=20261019)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    X.v3_targets(args)
                ents = json.loads((out / "all_entries.json").read_text())
                self.assertEqual(len(ents), 6)
                ranks = collections.Counter(x["target_len_rank"] for e in ents for x in e["targets"])
                self.assertEqual(sorted(ranks), [0, 1, 2, 3])
                self.assertLessEqual(max(ranks.values()) - min(ranks.values()), 1)          # balanced
                for e in ents:
                    self.assertTrue(all(x["step"] < MS for x in e["targets"]))
                    v = Path(e["view"]).read_text()
                    self.assertIn("student question: Which statement about the ", v)
                    self.assertIn("Form example", v)
                    self.assertTrue(e["src"].endswith(f"src_v3/alg__{e['traj_id'].split('/')[1]}.json"))
                self.assertEqual(sum(len(json.loads((out / f"shards/shard_{k:02d}.json").read_text()))
                                     for k in range(4)), 6)
                # an entry whose add-on source exists keeps its targets when the work list is rebuilt
                e0 = ents[0]
                Path(e0["src"]).write_text("{}")
                ents2 = json.loads((out / "all_entries.json").read_text())
                for e in ents2:
                    if e["traj_id"] == e0["traj_id"]:
                        e["targets"] = [dict(x, target_len_rank=(x["target_len_rank"] + 1) % 4) for x in e["targets"]]
                (out / "all_entries.json").write_text(json.dumps(ents2))
                with contextlib.redirect_stdout(io.StringIO()):
                    X.v3_targets(args)
                ents3 = {e["traj_id"]: e for e in json.loads((out / "all_entries.json").read_text())}
                want = next(e for e in ents2 if e["traj_id"] == e0["traj_id"])["targets"]
                self.assertEqual([(x["step"], x["target_len_rank"]) for x in ents3[e0["traj_id"]]["targets"]],
                                 [(x["step"], x["target_len_rank"]) for x in want])
                for e in ents:                  # every other entry: the first run's targets, ranks and position
                    if e["traj_id"] != e0["traj_id"]:
                        self.assertEqual([(x["step"], x["target_len_rank"]) for x in ents3[e["traj_id"]]["targets"]],
                                         [(x["step"], x["target_len_rank"]) for x in e["targets"]])
                self.assertEqual(list(ents3), [e["traj_id"] for e in ents])
                self.assertEqual(json.loads((out / "summary.json").read_text())["kept_with_source"], 1)
            finally:
                X.RAW_BY_SUBSET, X.TREES_DIR, X.V2_TREES = old

    def test_check_and_merge(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            raw = d / "raw"
            TT.write_raw(raw, "alg", "trajectory_0000", make_raw())
            t = ft.load_traj(raw / "Algorithm-Generated/train/trajectory_0000.json", "alg")
            src_dir = d / "trees" / "src_v3"
            src_dir.mkdir(parents=True)
            (src_dir / "alg__trajectory_0000.json").write_text(json.dumps(dict(clean_src(t), teacher="claude-opus")))
            old = (X.RAW_BY_SUBSET, X.TREES_DIR)
            X.RAW_BY_SUBSET, X.TREES_DIR = {"alg": raw}, {"alg": d / "trees"}
            try:
                ids = d / "ids.json"
                ids.write_text(json.dumps([dict(entry(), src=str(src_dir / "alg__trajectory_0000.json"))]))
                with contextlib.redirect_stdout(io.StringIO()):
                    X.v3_check(argparse.Namespace(ids=str(ids), entries="", prefix=str(d / "chk_"), seed=20261010,
                                                  start_time=None, teacher="claude-opus"))
                summ = json.loads((d / "chk_summary.json").read_text())
                self.assertEqual((summ["n"], summ["passed"], summ["clean_nodes"], summ["skips"]), (1, 1, 2, 0))
                (d / "v2.jsonl").write_text(json.dumps(v2_line(t)) + "\n")
                with contextlib.redirect_stdout(io.StringIO()):
                    X.v3_merge(argparse.Namespace(trees=str(d / "v2.jsonl"), addon=str(d / "chk_alg.jsonl"),
                                                  render="free", out=str(d / "v3.jsonl")))
                v3 = json.loads((d / "v3.jsonl").read_text())
                self.assertEqual([nd["id"] for nd in v3["nodes"]], ["n1", "n2", "n3", "n4", "n5", "n6", "n7", "c1",
                                                                    "c2"])
                self.assertTrue(all(ft.node_text_index_free(nd) for nd in v3["nodes"]))
                self.assertEqual(json.loads((d / "v3.summary.json").read_text())["nodes_with_a_step_index"], 0)
                # a failing add-on is not merged
                bad = json.loads((d / "chk_alg.jsonl").read_text())
                bad["checks"]["pass"] = False
                st = collections.Counter()
                m = ft.merge_v3(t, v2_line(t), bad, render="index", stats=st)
                self.assertEqual(len(m["nodes"]), 7)
                self.assertEqual(st["addon_failed"], 1)
                self.assertEqual(m["nodes"], v2_line(t)["nodes"])                  # render index: v2 nodes as they are
            finally:
                X.RAW_BY_SUBSET, X.TREES_DIR = old


# ---------------------------------------------------------------- builders
# _t2 files of tests/test_build_fa_sf.py's synthetic build with its trees file, recorded 2026-10-19 with the code before
# the v3 change (scratchpad treev3/baseline/synth_md5.json). A mismatch means the v2 tree build changed; if fa_public's
# item rendering was changed on purpose, re-record.
V2_GOLDEN = {"out/student_fa_sf_alg_t2/train.jsonl": "1ee143f1c2896060a8e0703e21de052d",
             "out/student_fa_sf_alg_t2/val.jsonl": "a74250ee604f6cd0bca5d2b77f254c5e",
             "out/student_fa_sf_alg_disj_t2/train.jsonl": "6fb13756c017d67198f79d35b4c8abca",
             "out/student_fa_sf_hc_t2/train.jsonl": "a5d37aa72f81aa90a6f1e57c92ad838b",
             "out/student_fa_sf_hc_t2/val.jsonl": "1ff142764b2e4077bd104bf4d8652fcd",
             "out/student_fa_sf_hc_disj_t2/train.jsonl": "3359773061a53e15f652f3f212aec2b2",
             "out/student_fa_sf_alg/train.jsonl": "bd09e4c2125c7b2e599140cd3b97adbc",
             "out/student_fa_sf_hc/train.jsonl": "e6296674fcaa71515626c6b1d89c61ee"}
KINDS = ["ok"] * 6 + ["pointer", "checks", "parent", "answer", "cycle", "leak", "final"]


def md5s(d: Path) -> dict:
    return {str(p.relative_to(d)): hashlib.md5(p.read_bytes()).hexdigest() for p in sorted((d / "out").rglob("*.jsonl"))}


class TestV2Golden(unittest.TestCase):
    def test_synthetic_build_byte_identical(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            TB.make_raw(d / "raw")
            recs = {r["uid"]: r for r in fp.read_stepfinder(str(d / "raw/stepfinder/repo/data"),
                                                            collections.Counter())}
            (d / "trees").mkdir()
            for s in ("alg", "hc"):
                with open(d / "trees" / f"{s}.jsonl", "w") as f:
                    for k, kind in enumerate(KINDS):
                        stem = f"trajectory_{k:04d}"
                        f.write(json.dumps(TB.tree_line(s, stem, recs[f"{TB.B.UID_PREFIX[s]}-{stem}"], kind=kind))
                                + "\n")
            TB.build(d, "--trees", str(d / "trees/{subset}.jsonl"))
            got = md5s(d)
            for p, m in V2_GOLDEN.items():
                self.assertEqual(got[p], m, p)


class TestBuilderV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.d = Path(cls.tmp.name)
        TB.make_raw(d / "raw")
        for p in (d / "raw/stepfinder/repo/data").rglob("*.json"):       # "alg q0 t0 step 3: ..." -> "... msg3: ..."
            x = json.loads(p.read_text())
            for h in x["history"]:
                h["content"] = re.sub(r"\bstep (\d+):", r"msg\1:", h["content"])
            p.write_text(json.dumps(x))
        recs = {r["uid"]: r for r in fp.read_stepfinder(str(d / "raw/stepfinder/repo/data"), collections.Counter())}
        (d / "trees").mkdir()
        cls.n_addon = {}
        for s in ("alg", "hc"):
            with open(d / "trees" / f"{s}.jsonl", "w") as f, open(d / "trees" / f"add_{s}.jsonl", "w") as g:
                n = 0
                for k, kind in enumerate(KINDS):
                    stem = f"trajectory_{k:04d}"
                    rec = recs[f"{TB.B.UID_PREFIX[s]}-{stem}"]
                    f.write(json.dumps(TB.tree_line(s, stem, rec, kind=kind)) + "\n")
                    if rec["dec"] >= 1:
                        g.write(json.dumps(cls.addon_line(s, stem, rec, ok=k != 1)) + "\n")
                        n += kind == "ok" and k != 1
                cls.n_addon[s] = n
        cls.v2 = TB.build(d, "--trees", str(d / "trees/{subset}.jsonl"))
        cls.v2_md5 = md5s(d)
        cls.v3 = TB.build(d, "--trees", str(d / "trees/{subset}.jsonl"), "--trees-addon",
                          str(d / "trees/add_{subset}.jsonl"), "--pointer-render", "free")
        cls.v3_md5 = md5s(d)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def addon_line(s: str, stem: str, rec: dict, ok: bool = True) -> dict:
        k = 0
        node = {"id": "c1", "type": "clean", "question": f"Which statement about {fp.step_ref(rec, k)} is supported by "
                "the task and the log?", "options": ["a count", "a date", "a name", "a place"], "answer": 2,
                "parents": [], "cite_step": k, "quote": rec["steps"][k]["text"][:20]}
        return {"traj_id": f"{s}/{stem}", "subset": s, "teacher": "synthetic", "nodes": [node], "skips": [],
                "checks": {"pass": ok, "failures": [] if ok else ["length_tell: c1"]}, "meta": {}}

    def items(self, rel: str) -> list[dict]:
        return [json.loads(x) for x in (self.d / rel).read_text().splitlines()]

    def test_v3_flags_leave_v2_files(self):
        # the v3 build writes only the _t3 directories: every other file is byte for byte the v2 build's
        for p, m in self.v2_md5.items():
            self.assertEqual(self.v3_md5[p], m, p)
        self.assertEqual(sorted(set(self.v3_md5) - set(self.v2_md5)),
                         sorted(f"out/student_fa_sf_{s}{x}/{f}.jsonl" for s in ("alg", "hc")
                                for x in ("_t3", "_disj_t3") for f in ("train", "val")))

    def test_t3_items(self):
        for s in ("alg", "hc"):
            t = self.v3[s]["trees"]
            self.assertEqual(t["pointer_render"], "free")
            self.assertEqual(t["items_with_tree"], self.v2[s]["trees"]["items_with_tree"])
            self.assertEqual(t["v3"]["addon_merged"], self.n_addon[s])
            self.assertEqual(t["types"].get("clean"), self.n_addon[s])
            self.assertEqual(t["refused"], self.v2[s]["trees"]["refused"])
            t3 = self.items(f"out/student_fa_sf_{s}_t3/train.jsonl") + self.items(f"out/student_fa_sf_{s}_t3/val.jsonl")
            t2 = self.items(f"out/student_fa_sf_{s}_t2/train.jsonl") + self.items(f"out/student_fa_sf_{s}_t2/val.jsonl")
            self.assertEqual([it["prompt"] for it in t3], [it["prompt"] for it in t2])
            seen_free_pointer = 0
            for it in t3:
                for sq in it["subqs"]:
                    mc_check(sq)
                    self.assertEqual(sq.get("render"), "free")
                    self.assertTrue(ft.node_text_index_free(sq), sq)
                    self.assertFalse(any(TB.B.STEP_WORD.match(o) for o in sq["options"]))
                    if sq["type"] in ft.POINTER_TYPES:
                        seen_free_pointer += 1
                        self.assertEqual(len(sq["option_steps"]), len(sq["options"]))
                        for o in sq["options"]:
                            self.assertRegex(o, OPT)
                            ex = OPT.match(o).group(2).rstrip(ft.ELLIPSIS)
                            self.assertIn(" ".join(ex.split()), " ".join(it["prompt"].split()))   # visible
            self.assertGreater(seen_free_pointer, 0)
            self.assertEqual(t["answer_step_is_decisive"], self.v2[s]["trees"]["answer_step_is_decisive"])
            summ = json.loads((self.d / f"out/student_fa_sf_{s}_t3/summary.json").read_text())
            self.assertEqual(summ["trees"]["v3"]["addon_merged"], self.n_addon[s])

    def test_node_type_selection(self):
        """--tree-node-types (treeplan.full_tree_plan node_types) selects clean nodes and the index-free pointers."""
        it = next(x for x in self.items("out/student_fa_sf_alg_t3/train.jsonl")
                  if any(sq["type"] == "clean" for sq in x["subqs"])
                  and any(sq["type"] == "answer_step" for sq in x["subqs"]))
        for types in ({"clean"}, {"answer_step", "provenance"}, {"content", "agent", "clean"}):
            st: dict = {}
            plan = full_tree_plan(it, cap=0, rng=random.Random(0), subq_target="truth", node_types=types, stats=st)
            got = {e["node"]["type"] for e in plan}
            self.assertTrue(got and got <= types, (types, got))
            self.assertEqual(st["node_type_dropped"], len(it["subqs"]) - len(plan))
        try:
            from cotdistill.student import build_full_tree
        except ImportError:
            return
        ex = build_full_tree(it, mode="plain", view="statements", rng=random.Random(0), cap=0, subq_target="truth",
                             node_types=frozenset({"clean"}))
        mc = [e for e in ex if e.kind == "subq"]
        self.assertEqual(len(mc), sum(sq["type"] == "clean" for sq in it["subqs"]))

    def test_render_without_addon_and_suffix(self):
        with tempfile.TemporaryDirectory() as d2:
            d2 = Path(d2)
            TB.make_raw(d2 / "raw")
            (d2 / "trees").mkdir()
            for s in ("alg", "hc"):
                (d2 / "trees" / f"{s}.jsonl").write_text((self.d / "trees" / f"{s}.jsonl").read_text())
            summ = TB.build(d2, "--trees", str(d2 / "trees/{subset}.jsonl"), "--pointer-render", "index",
                            "--trees-addon", str(self.d / "trees/add_{subset}.jsonl"), "--trees-suffix", "_t3i")
            items = [json.loads(x) for x in (d2 / "out/student_fa_sf_alg_t3i/train.jsonl").read_text().splitlines()]
            clean = [sq for it in items for sq in it["subqs"] if sq["type"] == "clean"]
            self.assertTrue(clean)
            self.assertTrue(all(re.search(r"Step \d+ \[", sq["question"]) for sq in clean))   # index render
            self.assertNotIn("render", clean[0])
            self.assertEqual(summ["alg"]["trees"]["v3"]["addon_merged"], self.n_addon["alg"])

    def test_answer_cue_inside_excerpt(self):
        """An index-free node may quote a log excerpt that holds "Answer:" (the step reference of a final-answer step);
        the teacher's own wording may not, and an index-render node keeps the strict v2 rule."""
        rec = {"steps": [{"agent": "Lead", "text": "Answer: 3"}, {"agent": "Coder", "text": "Counted the clicks: 3"}]}
        item = {"fa": {"step_cap": 3000}}

        def line(q: str, free: bool = True) -> dict:
            nd = {"id": "n1", "type": "provenance", "question": q, "answer": 0, "parents": [], "cite_step": None,
                  "quote": "", "options": ["Coder: “Counted the clicks: 3”", "Lead: “Answer: 3”"],
                  "option_steps": [1, 0]}
            if free:
                nd["render"] = "free"
            return {"nodes": [nd], "checks": {"pass": True}}
        quoted = "The Lead step “Answer: 3” reports a click count. Which step first counted the clicks?"
        nodes, bad, _ = TB.B.convert_tree(line(quoted), rec, item)
        self.assertEqual(bad, [])
        self.assertEqual(nodes[0]["question"], quoted)
        _, bad, _ = TB.B.convert_tree(line("Which step counted the clicks? Answer: the Coder step"), rec, item)
        self.assertEqual(bad, ["leak:n1"])
        _, bad, _ = TB.B.convert_tree(line("Which step gives the mistake_step “Answer: 3”?"), rec, item)
        self.assertEqual(bad, ["leak:n1"])                                       # label words: whole text
        _, bad, _ = TB.B.convert_tree(line(quoted, free=False), rec, item)
        self.assertEqual(bad, ["leak:n1"])                                       # v2 (index render): unchanged


PC = load_script("fa_trees_postcheck")


class TestPostcheckFilteredV3(unittest.TestCase):
    """_t3 built from fa_trees_postcheck --apply copies (apply_filters, drop-node policy): a flagged v2 node leaves the
    item and its child is re-attached to the removed node's parents, a flagged clean node leaves its add-on line; all
    other nodes and every final question are those of the unfiltered _t3 build; the "postcheck" record of a filtered
    line never reaches an item."""
    N = 8

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.d = Path(cls.tmp.name)
        TB.make_raw(d / "raw")
        for p in (d / "raw/stepfinder/repo/data").rglob("*.json"):       # no "step N:" index word in the excerpts
            x = json.loads(p.read_text())
            for h in x["history"]:
                h["content"] = re.sub(r"\bstep (\d+):", r"msg\1:", h["content"])
            p.write_text(json.dumps(x))
        recs = {r["uid"]: r for r in fp.read_stepfinder(str(d / "raw/stepfinder/repo/data"), collections.Counter())}
        lines, adds = [], []
        cls.flag_v2, cls.flag_clean = set(), set()
        for s in ("alg", "hc"):
            for k in range(cls.N):
                stem = f"trajectory_{k:04d}"
                rec = recs[f"{TB.B.UID_PREFIX[s]}-{stem}"]
                ln = TB.tree_line(s, stem, rec)
                ln["nodes"].append({"id": "n4", "type": "conflict", "question": "Why does that claim fail the task?",
                                    "options": ["It names the wrong year", "It skips one of the rows",
                                                "It counts the header line", "It reads another file"],
                                    "answer": 1, "parents": ["n2"], "cite_step": None, "quote": ""})
                lines.append(ln)
                if k % 2 == 0:
                    cls.flag_v2.add(ln["traj_id"])
                if rec["dec"] >= 1:
                    adds.append(TestBuilderV3.addon_line(s, stem, rec))
                    if k % 3 == 0:
                        cls.flag_clean.add(ln["traj_id"])
        flags = {"gt_in_stem": sorted(f"{t}#n2" for t in cls.flag_v2) + sorted(f"{t}#c1" for t in cls.flag_clean),
                 "gt_in_stem_weak": [f"{lines[1]['traj_id']}#n3"]}               # not in the removal set
        f_lines, _, cls.st_lines = PC.apply_filters(lines, [], flags, {"gt_in_stem"})
        _, f_adds, cls.st_adds = PC.apply_filters([], adds, flags, {"gt_in_stem"})
        (d / "trees").mkdir()
        for name, xs in (("{s}.jsonl", lines), ("add_{s}.jsonl", adds), ("f_{s}.jsonl", f_lines),
                         ("f_add_{s}.jsonl", f_adds)):
            for s in ("alg", "hc"):
                (d / "trees" / name.format(s=s)).write_text(
                    "".join(json.dumps(x) + "\n" for x in xs if x["traj_id"].startswith(s + "/")))
        t = str(d / "trees")
        cls.full = TB.build(d, "--trees", t + "/{subset}.jsonl", "--trees-addon", t + "/add_{subset}.jsonl",
                            "--pointer-render", "free")
        cls.filt = TB.build(d, "--trees", t + "/f_{subset}.jsonl", "--trees-addon", t + "/f_add_{subset}.jsonl",
                            "--pointer-render", "free", "--trees-suffix", "_t3p")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def items(self, rel: str) -> list[dict]:
        return [json.loads(x) for x in (self.d / rel).read_text().splitlines()]

    def test_filter_counts(self):
        self.assertEqual(self.st_lines["tree_nodes_removed"], len(self.flag_v2))
        self.assertEqual(self.st_lines["tree_lines_modified"], len(self.flag_v2))
        self.assertEqual(self.st_adds["addon_nodes_removed"], len(self.flag_clean))
        self.assertGreater(len(self.flag_clean), 0)

    def test_merge(self):
        seen = collections.Counter()
        for s in ("alg", "hc"):
            for sp in ("train", "val"):
                full = self.items(f"out/student_fa_sf_{s}_t3/{sp}.jsonl")
                filt = self.items(f"out/student_fa_sf_{s}_t3p/{sp}.jsonl")
                a2 = self.items(f"out/student_fa_sf_{s}/{sp}.jsonl")
                self.assertEqual(len(full), len(filt))
                for a, b, c in zip(full, filt, a2):
                    self.assertEqual({k: v for k, v in a.items() if k != "subqs"},
                                     {k: v for k, v in b.items() if k != "subqs"})
                    self.assertEqual(b["prompt"], c["prompt"])
                    self.assertNotIn("postcheck", json.dumps(b))
                    tid = f"{s}/" + b["fa"]["uid"].split("-", 1)[1]
                    fa_ = {x["id"]: x for x in a["subqs"]}
                    fb = {x["id"]: x for x in b["subqs"]}
                    want = dict(fa_)
                    if tid in self.flag_v2 and "n2" in fa_:
                        seen["v2"] += 1
                        self.assertNotIn("n2", fb)
                        want.pop("n2")
                        self.assertEqual(fb["n4"]["depends_on"], fa_["n2"]["depends_on"])     # re-attached
                        self.assertEqual(fa_["n4"]["depends_on"], ["n2"])
                        want["n4"] = dict(fa_["n4"], depends_on=fb["n4"]["depends_on"],
                                          node_depth=fb["n4"]["node_depth"])
                        self.assertEqual(fb["n4"]["node_depth"], fa_["n2"]["node_depth"])
                    if tid in self.flag_clean and "c1" in fa_:
                        seen["clean"] += 1
                        self.assertNotIn("c1", fb)
                        want.pop("c1")
                    self.assertEqual(fb, want)
                    for sq in b["subqs"]:
                        self.assertEqual(sq["render"], "free")
                        mc_check(sq)
        self.assertGreater(seen["v2"], 0)
        self.assertGreater(seen["clean"], 0)


if __name__ == "__main__":
    unittest.main()
