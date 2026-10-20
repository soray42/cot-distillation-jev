"""scripts/fa_trees_postcheck.py: the post-check detectors (GT in a code-rendered stem excerpt, near-negation
distractors, agent nodes with < 2 options, runs that did not fail, fields outside the schema), the drop / skip reason
classes, and --apply (filtered copies, children re-attached, inputs untouched, no writes into a src directory).
Synthetic data, CPU only, no model:

  ~/.venvs/sev/bin/python -m unittest tests.test_fa_trees_postcheck
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cotdistill import fa_trees as ft  # noqa: E402

_spec = importlib.util.spec_from_file_location("fa_trees_postcheck_test", ROOT / "scripts" / "fa_trees_postcheck.py")
PC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PC)

HIST = [
    ("Manager", "assistant", "Plan: find the film, then report the studio that released it."),
    ("search_agent", "assistant", "Searching the archive for the release list of 1985 films."),
    ("search_agent", "assistant", "Time-Parking 2 is listed first among the DVD rows of the release sheet."),
    ("environment", "user", "Result: 3 rows; total 5,429,112.0 units counted in the sheet."),
    ("Manager", "assistant", "Let us take the second row as the answer and skip the check."),
    ("search_agent", "assistant", "The second row names Cereal Killer III as the release."),
    ("Manager", "assistant", "Final answer: Cereal Killer III"),
]
MS = 4


def make_traj(gt: str = "Time-Parking 2", hist=HIST, ms: int = MS, traj_id: str = "aftraj/x_0001") -> dict:
    sub, stem = traj_id.split("/")
    return {"traj_id": traj_id, "subset": sub, "question": "Which film of the sheet came out first? Give its title.",
            "ground_truth": gt, "mistake_step": ms, "mistake_agent": hist[ms][0], "mistake_reason": "skipped the check",
            "steps": [{"k": i, "agent": a, "role": r, "text": t} for i, (a, r, t) in enumerate(hist)]}


def raw_of(traj: dict) -> dict:
    return {"question": traj["question"], "ground_truth": traj["ground_truth"], "mistake_agent": traj["mistake_agent"],
            "mistake_step": traj["mistake_step"], "mistake_reason": traj["mistake_reason"],
            "history": [{"step_id": s["k"], "name": s["agent"], "role": s["role"], "content": s["text"]}
                        for s in traj["steps"]]}


def v2_src(traj_id: str = "aftraj/x_0001", **extra) -> dict:
    return {"traj_id": traj_id, "teacher": "fake", "answer_step": 6, "label_note": "", "nodes": [
        {"id": "n1", "type": "answer_step", "question": "Which step states the team's final answer?",
         "answer_step": 6, "distractor_steps": [5, 2, 1], "quote": "Final answer: Cereal Killer III", "parents": []},
        {"id": "n2", "type": "provenance", "question": "Step 6 [Manager] repeats a title. Which step chose its row?",
         "answer_step": 4, "distractor_steps": [0, 1, 2], "quote": "take the second row as the answer",
         "parents": ["n1"]},
        {"id": "n3", "type": "content", "question": "What did Step 2 [search_agent] report?",
         "correct": "The first DVD row of the release sheet", "distractors": [
             "A list of 1985 films from the archive", "The studio of the second row",
             "A total count of the units in the sheet"], "cite_step": 4, "quote": "take the second row",
         "parents": ["n2"]},
        {"id": "n4", "type": "conflict", "question": "Why is that choice wrong?",
         "correct": "The task asks for the first film, not the second row",
         "distractors": ["The sheet has no DVD rows at all", "The archive lists only 1984 films",
                         "The environment reported an error"], "cite_step": None,
         "quote": "Which film of the sheet came out first?", "parents": ["n3"]},
        {"id": "n5", "type": "agent", "question": "Who made the decisive choice?", "answer_agent": "Manager",
         "distractor_agents": ["search_agent", "environment"], "parents": ["n2"]}], **extra}


def clean_node(nid: str, k: int, options: list[str], answer: int = 0) -> dict:
    return {"id": nid, "type": "clean", "question": ft.CLEAN_QUESTION.format(ref=f"Step {k} [{HIST[k][0]}]"),
            "options": options, "answer": answer, "parents": [], "cite_step": k, "quote": HIST[k][2][:20]}


class TestGTInStem(unittest.TestCase):
    def test_clean_reference_excerpt_contains_gt(self):
        t = make_traj()
        nd = clean_node("c1", 2, ["Named a film from the sheet", "Asked for a studio", "Ran a search", "Listed rows"])
        self.assertTrue(ft.gt_leak(t["ground_truth"], ft.free_ref_for(t, 2)))       # the excerpt has the GT ...
        self.assertFalse(ft.gt_leak(t["ground_truth"], nd["question"]))              # ... the index form does not
        fl = PC.excerpt_gt_flags(t, [nd], PC.gold_values(t["ground_truth"]))
        self.assertEqual([(f["flag"], f["node"], f["where"]) for f in fl], [("gt_in_stem", "c1", "question")])

    def test_v2_content_question_under_free_render(self):
        t = make_traj()
        line = ft.output_record(t, v2_src())
        fl = PC.excerpt_gt_flags(t, line["nodes"], PC.gold_values(t["ground_truth"]))
        self.assertIn(("gt_in_stem", "n3"), {(f["flag"], f["node"]) for f in fl})
        self.assertNotIn("n4", {f["node"] for f in fl})

    def test_numbers_with_tolerance_and_weak_tiers(self):
        g = PC.gold_values("5429112")
        self.assertEqual(PC.hit_tier(PC.text_hits(g, "total 5,429,112.0 units")), "strong")
        self.assertEqual(PC.text_hits(g, "total 5,429,113 units"), [])
        self.assertEqual(PC.hit_tier(PC.text_hits(PC.gold_values("1.01"), "it gives 1.0103 here")), "strong")
        self.assertEqual(PC.text_hits(PC.gold_values("17"), "it gives 17.4 here"), [])          # GT without decimals
        self.assertEqual(PC.hit_tier(PC.text_hits(PC.gold_values("3"), "Result: 3 rows")), "weak")
        self.assertEqual(PC.hit_tier(PC.text_hits(PC.gold_values("pears, bananas"), "bananas first")), "weak")
        rng = PC.gold_values("17.2 meters (acceptable range: anything between 17.03 and 17.37 meters)")
        self.assertEqual(PC.hit_tier(PC.text_hits(rng, "about 17.31 m tall")), "strong")
        self.assertEqual(PC.gold_values("None"), {})
        self.assertEqual(PC.gold_values("['result.json']"), {})

    def test_gt_also_in_task_is_weak(self):
        t = make_traj(gt="Time-Parking 2")
        t["question"] = "Is it Time-Parking 2 or Cereal Killer III? Give the title."
        nd = clean_node("c1", 2, ["Named a film", "Asked for a studio", "Ran a search", "Listed rows"])
        fl = PC.excerpt_gt_flags(t, [nd], PC.gold_values(t["ground_truth"]))
        self.assertEqual([(f["flag"], f["why"]) for f in fl], [("gt_in_stem_weak", "in_task")])


class TestNegationPair(unittest.TestCase):
    def test_pairs(self):
        self.assertIsNotNone(PC.negation_pair("Read the speaker notes", "Did not read the speaker notes"))
        self.assertIsNotNone(PC.negation_pair("Shared code that also reads the notes",
                                              "Shared code that does not read the notes"))
        self.assertIsNotNone(PC.negation_pair("Shared the code with the notes", "Shared the code without the notes"))
        self.assertIsNone(PC.negation_pair("Ran the tests on code that joins with a space",
                                           "Ran the tests on code that joins with no separator"))
        self.assertIsNone(PC.negation_pair("The table gives b*d = e", "The table gives d*b = d, not e"))
        self.assertIsNone(PC.negation_pair("Did not read the notes", "Never read the notes"))      # both negated

    def test_node_flags(self):
        nd = {"id": "n3", "type": "content", "options": ["Asked for the files", "Read the speaker notes",
                                                         "Did not read the speaker notes", "Listed the slides"],
              "answer": 1}
        fl = PC.negation_flags(nd)
        self.assertEqual([(f["flag"], f["distractor"]) for f in fl],
                         [("negation_pair", "Did not read the speaker notes")])
        self.assertEqual(PC.negation_flags(dict(nd, type="agent")), [])


class TestAgentNodes(unittest.TestCase):
    def test_option_counts(self):
        one = {"id": "n5", "type": "agent", "options": ["assistant"], "answer": 0}
        self.assertEqual([f["flag"] for f in PC.agent_flags(one)], ["agent_lt2"])
        triv = {"id": "n5", "type": "agent", "options": ["environment", "Manager"], "answer": 1}
        self.assertEqual([f["flag"] for f in PC.agent_flags(triv)], ["agent_trivial"])
        ok = {"id": "n5", "type": "agent", "options": ["search_agent", "Manager", "environment"], "answer": 1}
        self.assertEqual(PC.agent_flags(ok), [])
        self.assertEqual(PC.agent_flags({"id": "n3", "type": "content", "options": ["a"], "answer": 0}), [])


class TestNotFailed(unittest.TestCase):
    def flag(self, answer_text: str, gt: str, traj_id: str = "aftraj/x_0001"):
        hist = HIST[:-1] + [("Manager", "assistant", answer_text)]
        f = PC.not_failed_flag(make_traj(gt=gt, hist=hist, traj_id=traj_id), 6)
        return (f["flag"], f["match"].split(":")[0]) if f else None

    def test_strong_and_weak(self):
        self.assertEqual(self.flag("Final answer: 185", "185"), ("not_failed", "equal"))
        self.assertEqual(self.flag("Final answer: 5,429,112.0", "5429112"), ("not_failed", "numeric"))
        self.assertEqual(self.flag("Final answer: Cereal Killer III.", "cereal killer iii"), ("not_failed", "equal"))
        self.assertEqual(self.flag("Final answer: 185 yards", "185"), ("not_failed_weak", "numeric_with_unit"))
        self.assertEqual(self.flag("Final answer: bananas, pears", "pears, bananas"),
                         ("not_failed_weak", "same_elements_other_order"))
        self.assertIsNone(self.flag("Final answer: 190", "185"))
        self.assertIsNone(self.flag("Final answer: Cereal Killer III", "None"))       # no reference
        self.assertIsNone(self.flag("Final answer: Cereal Killer III", ""))

    def test_framework_answers(self):
        smol = ("Thought: done.\nCode:\n```py\nfinal_answer(average)\n```\nExecution logs:\n"
                "Last output from code snippet:\n733333.0")
        self.assertEqual(PC.extract_answer(smol), ("733333.0", "printed_after_final_answer"))
        self.assertEqual(self.flag(smol, "733333", "wwpro/tablebench_smolagents_0001"), ("not_failed", "numeric"))
        boxed = "So the probability is\n\n\\boxed{0.35}"
        self.assertEqual(self.flag(boxed, "\\dfrac{7}{20}", "wwpro/math_mathchat_0001"),
                         ("not_failed", "math_value"))
        self.assertIsNone(self.flag("Final answer: 1/2", "2/4"))                    # GAIA: fractions as written
        self.assertEqual(self.flag("\\boxed{0, 1, -1}", "-1,0,1", "wwpro/math_mathchat_0002"),
                         ("not_failed", "same_value_set"))
        self.assertEqual(self.flag("I am confident.\n\nFINAL ANSWER: Niloufar Bayan", "Niloufar Bayan",
                                   "wwpro/simpleqa_verified_magentic-one_0001"), ("not_failed", "equal"))
        self.assertEqual(self.flag("\\boxed{B}", "A", "wwpro/mmlu_pro_mathchat_0001"), None)
        self.assertEqual(self.flag("\\boxed{(A)}", "A", "wwpro/mmlu_pro_mathchat_0001"), ("not_failed", "letter"))
        self.assertEqual(PC.extract_answer('Action: python_interpreter(final_answer("Juliet"))'),
                         ("Juliet", "final_answer_call"))


class TestUnknownFields(unittest.TestCase):
    def test_v2_and_addon(self):
        self.assertEqual(PC.unknown_fields_v2(v2_src()), [])
        src = v2_src(gt_in_ref=True)
        src["nodes"][2]["gt_in_ref"] = False
        src["nodes"][4]["correct"] = "x"                       # a text-node field on an agent node
        self.assertEqual(PC.unknown_fields_v2(src), ["top:gt_in_ref", "n3:gt_in_ref", "n5:correct"])
        add = {"traj_id": "aftraj/x_0001", "teacher": "fake", "nodes": [
            {"id": "c1", "type": "clean", "cite_step": 2, "correct": "a", "distractors": ["b", "c", "d"],
             "quote": "Time-Parking 2 is listed"},
            {"id": "c2", "type": "clean", "cite_step": 3, "skip": "It already skips the check."}]}
        self.assertEqual(PC.unknown_fields_addon(add), [])
        add["nodes"][0]["gt_in_ref"] = True
        add["nodes"][1]["quote"] = "x"
        self.assertEqual(PC.unknown_fields_addon(add), ["c1:gt_in_ref", "c2:quote"])
        line = ft.output_record(make_traj(), v2_src())
        self.assertEqual(PC.unknown_fields_line(line), [])
        self.assertEqual(PC.unknown_fields_line(dict(line, gt_in_ref=1)), ["line:gt_in_ref"])


class TestReasonClasses(unittest.TestCase):
    def test_drop_classes(self):
        c = lambda r: PC.classify_drop(r)[0]  # noqa: E731
        self.assertEqual(c("not failed: final answer matches the reference (Step 5 boxes 0.35)"), "not failed")
        self.assertEqual(c("label not supported by the visible log: the date is only in the cut middle of Step 9"),
                         "truncated evidence")
        self.assertEqual(c("label not supported by the log: Step 3's slip is corrected at once by Step 4"),
                         "error corrected later or not causal")
        self.assertEqual(c("label not supported by the log: Step 2 only runs a search; the wrong name is first "
                           "given in Step 5 [Coder]"), "labelled step fine, error elsewhere")
        self.assertEqual(c("uncontradicted: Step 9 (short, nothing cut) asserts the name and no page contradicts it"),
                         "no evidence in log")
        self.assertEqual(c("no tree can pass the checks (label itself is supported)"),
                         "other: no valid tree (label kept)")
        self.assertEqual(c("zzz"), PC.DROP_UNCLASSIFIED)

    def test_skip_classes(self):
        self.assertEqual(PC.classify_skip("It already proposes the filter that the labelled step then runs."),
                         PC.SKIP_CLASSES[0])
        self.assertEqual(PC.classify_skip("It takes 1.05 g/mL, a wrong constant that changes the final x."),
                         PC.SKIP_CLASSES[1])
        self.assertEqual(PC.classify_skip("It is odd."), PC.SKIP_CLASSES[2])


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


class TestRunAndApply(unittest.TestCase):
    """End to end on a fake aftraj source: work list, v2 src, tree lines, v3 entry + add-on src + add-on line."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.d = d
        t = make_traj()
        (d / "raw/AFTraj/train").mkdir(parents=True)
        (d / "raw/AFTraj/train/x_0001.json").write_text(json.dumps(raw_of(t)))
        t2 = make_traj(gt="185", traj_id="aftraj/x_0002",
                       hist=HIST[:-1] + [("Manager", "assistant", "Final answer: 185")])
        (d / "raw/AFTraj/train/x_0002.json").write_text(json.dumps(raw_of(t2)))
        (d / "src").mkdir()
        (d / "src/aftraj__x_0001.json").write_text(json.dumps(v2_src(gt_in_ref=1)))
        (d / "src/aftraj__x_0002.json").write_text(json.dumps(v2_src("aftraj/x_0002")))
        (d / "src/aftraj__x_0003.json").write_text(json.dumps(
            {"traj_id": "aftraj/x_0003", "teacher": "fake",
             "drop": "label not supported by the log: the wrong name is first given in Step 5 [Coder]"}))
        (d / "ids.json").write_text(json.dumps([{"traj_id": f"aftraj/x_000{i}"} for i in (1, 2, 3)]))
        (d / "shards").mkdir()
        (d / "shards/shard_00.json").write_text(json.dumps([{"traj_id": f"aftraj/x_000{i}"} for i in (1, 2, 3, 4)]))
        lines = [ft.output_record(t, v2_src()), ft.output_record(t2, v2_src("aftraj/x_0002"))]
        self.trees = d / "aftraj.jsonl"
        self.trees.write_text("".join(json.dumps(x) + "\n" for x in lines))
        entry = {"traj_id": "aftraj/x_0001", "subset": "aftraj", "n_steps": 7, "mistake_step": MS,
                 "targets": [{"step": 2}, {"step": 3}],
                 "src": str(d / "src_v3/aftraj__x_0001.json")}
        (d / "src_v3").mkdir()
        (d / "src_v3/aftraj__x_0001.json").write_text(json.dumps(
            {"traj_id": "aftraj/x_0001", "teacher": "fake", "nodes": [
                {"id": "c1", "type": "clean", "cite_step": 2, "correct": "Named the first DVD row of the sheet",
                 "distractors": ["Named the studio of the second row", "Named a total count of the units",
                                 "Named a list of 1985 archive films"],
                 "quote": "is listed first among the DVD rows"},
                {"id": "c2", "type": "clean", "cite_step": 3, "skip": "It already skips the check that the "
                                                                      "labelled step then skips."}]}))
        (d / "entries.json").write_text(json.dumps([entry]))
        (d / "v3shards").mkdir()
        (d / "v3shards/shard_00.json").write_text(json.dumps([entry]))
        add = ft.addon_record(t, json.loads((d / "src_v3/aftraj__x_0001.json").read_text()), entry)
        assert add["checks"]["pass"], add["checks"]["failures"]
        self.addon = d / "v3add_aftraj.jsonl"
        self.addon.write_text(json.dumps(add) + "\n")
        self.saved = (copy.deepcopy(PC.SOURCES), PC.V3_ENTRIES, PC.V3_SHARDS, PC.V3_ADDON)
        PC.SOURCES["aftraj"] = {"raw": str(d / "raw"), "src": str(d / "src"), "ids": str(d / "ids.json"),
                                "trees": str(self.trees), "shards": str(d / "shards"),
                                "excluded": str(d / "excluded.txt")}
        PC.V3_ENTRIES, PC.V3_SHARDS = str(d / "entries.json"), str(d / "v3shards")
        PC.V3_ADDON = str(d / "v3add_{sub}.jsonl")

    def tearDown(self):
        PC.SOURCES.clear()
        PC.SOURCES.update(self.saved[0])
        PC.V3_ENTRIES, PC.V3_SHARDS, PC.V3_ADDON = self.saved[1:]
        self.tmp.cleanup()

    def args(self, **kw):
        a = argparse.Namespace(subsets=["aftraj"], trees={}, cap=ft.STEP_CAP, apply=False, out_dir=None,
                               remove=list(PC.DEFAULT_REMOVE), v2_node_policy="drop-node", allow_stale=False)
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    def test_run_flags_audit_and_apply(self):
        res = PC.run(self.args())
        fl = res["flags"]
        self.assertIn("aftraj/x_0001#n3", fl["gt_in_stem"])
        self.assertIn("aftraj/x_0001#c1", fl["gt_in_stem"])
        self.assertEqual(fl["not_failed"], ["aftraj/x_0002"])
        self.assertIn("aftraj/x_0001#top:gt_in_ref", fl["unknown_field"])
        self.assertNotIn("aftraj/x_0001#n5", fl["agent_trivial"] + fl["agent_lt2"])        # 2 real agents
        n = res["sources"]["aftraj"]["n"]
        self.assertEqual((n["items"], n["written"], n["drops"], n["kept"]), (3, 3, 1, 2))
        self.assertEqual(res["sources"]["aftraj"]["drop_first"]["labelled step fine, error elsewhere"], 1)
        self.assertEqual(res["status"]["aftraj"], [("shard_00", 4, 3)])
        V = res["v3"]["aftraj"]
        self.assertEqual((V["n"]["targets"], V["n"]["skips"]), (2, 1))
        self.assertEqual(V["skip_classes"][PC.SKIP_CLASSES[0]], 1)
        report = self.d / "report.md"
        PC.write_report(res, report, None, "test")
        self.assertIn("aftraj/x_0002", report.read_text())
        # --apply: filtered copies, inputs untouched
        before = {p: md5(p) for p in (self.trees, self.addon)}
        out = self.d / "filtered"
        summ = PC.do_apply(self.args(apply=True, out_dir=str(out), remove=["gt_in_stem", "not_failed",
                                                                            "negation_pair"]), res)
        self.assertEqual({p: md5(p) for p in (self.trees, self.addon)}, before)
        lines = [json.loads(x) for x in (out / "aftraj.jsonl").read_text().splitlines()]
        self.assertEqual([x["traj_id"] for x in lines], ["aftraj/x_0001"])                  # not_failed dropped
        nodes = {nd["id"]: nd for nd in lines[0]["nodes"]}
        self.assertNotIn("n3", nodes)
        self.assertEqual(nodes["n4"]["parents"], ["n2"])                                      # re-attached
        self.assertEqual(lines[0]["postcheck"]["removed"], [{"id": "n3", "flag": "gt_in_stem"}])
        add = [json.loads(x) for x in (out / "v3add_aftraj.jsonl").read_text().splitlines()]
        self.assertEqual([nd["id"] for nd in add[0]["nodes"]], [])                              # c1 removed
        self.assertTrue((out / "apply_summary.json").exists())
        self.assertIn(str(out / "aftraj.jsonl"), summ["files"])
        # drop-line policy drops the whole tree with a flagged v2 node
        out2 = self.d / "filtered2"
        PC.do_apply(self.args(apply=True, out_dir=str(out2), remove=["gt_in_stem"], v2_node_policy="drop-line"), res)
        self.assertEqual([json.loads(x)["traj_id"] for x in (out2 / "aftraj.jsonl").read_text().splitlines()],
                         ["aftraj/x_0002"])
        # stale: a source newer than the trees file -> --apply refuses unless --allow-stale
        src1 = self.d / "src/aftraj__x_0001.json"
        future = self.trees.stat().st_mtime + 100
        os.utime(src1, (future, future))
        res2 = PC.run(self.args())
        self.assertTrue(res2["sources"]["aftraj"]["stale"])
        with self.assertRaises(SystemExit):
            PC.do_apply(self.args(apply=True, out_dir=str(self.d / "filtered3")), res2)
        PC.do_apply(self.args(apply=True, out_dir=str(self.d / "filtered3"), allow_stale=True), res2)
        self.assertEqual(res["sources"]["aftraj"]["n"]["answer_extracted"], 2)
        # never into the input directory or a src directory
        with self.assertRaises(SystemExit):
            PC.do_apply(self.args(apply=True, out_dir=str(self.d)), res)
        self.assertIn("source directory", PC._forbidden_out(ROOT / "data/raw/aftraj_trees/src_v3/x", []))

    def test_remove_nodes_relinks(self):
        nodes = [{"id": "a", "parents": []}, {"id": "b", "parents": ["a"]}, {"id": "c", "parents": ["b"]},
                 {"id": "d", "parents": ["c", "a"]}]
        kept = PC.remove_nodes(nodes, {"b", "c"})
        self.assertEqual([(nd["id"], nd["parents"]) for nd in kept], [("a", []), ("d", ["a"])])
        self.assertEqual(nodes[2]["parents"], ["b"])                                           # input unchanged


if __name__ == "__main__":
    unittest.main()
