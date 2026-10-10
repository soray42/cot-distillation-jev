"""src/cotdistill/fa_trees.py and scripts/extract_fa_trees.py: schema, option rendering and shuffling, every check, on
synthetic trajectories with a programmatic fake teacher. No network, CPU only."""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cotdistill import fa_trees as ft  # noqa: E402

LONG_MIDDLE = "SECRET-MIDDLE-CLAIM about the grouping"


def make_raw(gt: str = "Time-Parking 2", ms: int = 3, agent: str = "Analyst") -> dict:
    hist = [
        ("Planner", "assistant", "Plan: load the sheet, keep only Blu-Ray rows, report the oldest title."),
        ("Computer_terminal", "user", "exitcode: 0 (execution succeeded)\nCode output: rows=25"),
        ("Coder", "assistant", "Loaded the sheet; the Title column holds section markers DVD, Blu-Ray and VHS."),
        ("Analyst", "assistant", "Let us merge DVD and Blu-Ray into one Disc group and pick the oldest Disc item."),
        ("Coder", "assistant", "x" * 1200 + " " + LONG_MIDDLE + " " + "y" * 1200),
        ("Computer_terminal", "user", "exitcode: 0 (execution succeeded)\nCode output: Cereal Killer III"),
        ("Analyst", "assistant", "Final answer: Cereal Killer III"),
        ("Planner", "assistant", "TERMINATE"),
    ]
    return {"question": "What is the title of the oldest Blu-Ray in the spreadsheet? Return it as written.",
            "ground_truth": gt, "mistake_agent": agent, "mistake_step": ms,
            "mistake_reason": "merged formats against the task",
            "history": [{"step_id": i, "role": r, "name": n, "content": c, "is_mistake": int(i == ms)}
                        for i, (n, r, c) in enumerate(hist)]}


def write_raw(root: Path, subset: str, stem: str, raw: dict) -> Path:
    d = root / ft.SUBSET_DIRS[subset] / "train"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{stem}.json"
    p.write_text(json.dumps(raw))
    return p


def fake_teacher(traj: dict) -> dict:
    """A programmatic teacher: answer step by the rule, one provenance hop to the decisive step, then content,
    conflict and agent nodes with quotes copied from the log."""
    ms = traj["mistake_step"]
    ans, _ = ft.answer_step_rule(traj)
    hop = 5
    return {"traj_id": traj["traj_id"], "teacher": "fake", "answer_step": ans, "nodes": [
        {"id": "n1", "type": "answer_step", "question": "Which step states the team's final answer?",
         "answer_step": ans, "distractor_steps": [5, 2, 7], "quote": "Final answer: Cereal Killer III",
         "parents": []},
        {"id": "n2", "type": "provenance", "question": "The title given as the final answer was printed by which step?",
         "answer_step": hop, "distractor_steps": [1, 2, 4], "quote": "Code output: Cereal Killer III",
         "parents": ["n1"]},
        {"id": "n3", "type": "provenance",
         "question": "The code behind that output used a merged group of formats. Which step proposed the merge?",
         "answer_step": ms, "distractor_steps": [0, 2, 4], "quote": "merge DVD and Blu-Ray into one Disc group",
         "parents": ["n2"]},
        {"id": "n4", "type": "content", "question": "What did Step 3 [Analyst] propose?",
         "correct": "Merge DVD and Blu-Ray into one group and take its oldest item",
         "distractors": ["Keep only the Blu-Ray rows and report the oldest title",
                         "Read the section markers DVD, Blu-Ray and VHS from the Title column",
                         "Stop the session and return the last printed value"],
         "cite_step": ms, "quote": "merge DVD and Blu-Ray into one Disc group", "parents": ["n3"]},
        {"id": "n5", "type": "conflict", "question": "Why does that proposal fail the task?",
         "correct": "The task asks for the oldest Blu-Ray only, so other formats must not compete",
         "distractors": ["The sheet has no section markers, so the formats cannot be told apart",
                         "The answer must be given in upper case letters",
                         "The terminal could not open the spreadsheet file"],
         "cite_step": None, "quote": "oldest Blu-Ray in the spreadsheet", "parents": ["n4"]},
        {"id": "n6", "type": "agent", "question": "Which agent proposed the merge?", "answer_agent": "Analyst",
         "distractor_agents": ["Planner", "Coder"], "parents": ["n3"]},
    ]}


def traj_from(raw: dict, subset: str = "alg", stem: str = "trajectory_0000") -> dict:
    with tempfile.TemporaryDirectory() as td:
        return ft.load_traj(write_raw(Path(td), subset, stem, raw), subset)


class TestBasics(unittest.TestCase):
    def setUp(self):
        self.t = traj_from(make_raw())

    def test_load_and_labels(self):
        self.assertEqual(self.t["traj_id"], "alg/trajectory_0000")
        self.assertEqual(ft.step_label(self.t, 3), "Step 3 [Analyst]")
        self.assertEqual(ft.agents_in_log(self.t), ["Planner", "Computer_terminal", "Coder", "Analyst"])

    def test_cut_matches_build_whowhen_eval(self):
        spec = importlib.util.spec_from_file_location("bwe", ROOT / "scripts/build_whowhen_eval.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for s in ["short", "a" * 1499, "b" * 1500, "c" * 1501 + "tail", " pad " * 900]:
            self.assertEqual(ft.cut(s, 1500), mod.cut(s, 1500))

    def test_answer_rule(self):
        self.assertEqual(ft.answer_step_rule(self.t), (6, "marker"))
        raw = make_raw()
        raw["history"][6]["content"] = "Cereal Killer III"          # bare answer, then TERMINATE
        self.assertEqual(ft.answer_step_rule(traj_from(raw)), (6, "bare"))
        raw["history"][6]["content"] = "A long summary without a marker. " * 10
        self.assertEqual(ft.answer_step_rule(traj_from(raw)), (None, "none"))
        raw["history"][6]["content"] = "**Final answer (as written):** X"
        self.assertEqual(ft.answer_step_rule(traj_from(raw)), (6, "marker"))
        # pilot cases: "Final <up to 4 words>:" at a line start, "final answer:" inside a line
        for text in ["Final score: The student earned 90 points.", "Final answer to the user: 28806.",
                     "Final: Minimum number of clicks = 1.", "Cross-checked both views. Final answer: 3."]:
            raw["history"][6]["content"] = text
            self.assertEqual(ft.answer_step_rule(traj_from(raw)), (6, "marker"), text)
        # a bare answer after closing lines wins over an earlier marker; offers / notices are closing lines
        raw = make_raw()
        raw["history"][2]["content"] = "Final string: The World of the 21st Century"
        raw["history"][6]["content"] = "The World of the 21st Century"
        raw["history"][7]["content"] = "If you need the source, we can add it, but the answer has been submitted."
        self.assertEqual(ft.answer_step_rule(traj_from(raw)), (6, "bare"))
        raw["history"][6]["content"] = "+70.0"
        raw["history"][7]["content"] = "Answer provided. Task complete."
        self.assertEqual(ft.answer_step_rule(traj_from(raw)), (6, "bare"))
        raw["history"][6]["content"] = "Yes. If the green plots form a single loop, Earl can walk it without backtracking."
        self.assertEqual(ft.answer_step_rule(traj_from(raw)), (6, "bare"))
        raw["history"][6]["content"] = "Review complete. The data is compiled. Proceed to close the task."
        self.assertEqual(ft.answer_step_rule(traj_from(raw)), (2, "marker"))

    def test_gt_leak(self):
        self.assertTrue(ft.gt_leak("Time-Parking 2", "the oldest is time-parking  2 here"))
        self.assertFalse(ft.gt_leak("Time-Parking 2", "Time-Parking 21 is different"))
        self.assertTrue(ft.gt_leak("3", "the count is 3."))
        self.assertFalse(ft.gt_leak("3", "Step 3 [Analyst] counted 2000-2009 and 13 items, 3.5 on average"))
        self.assertTrue(ft.gt_leak("100000000", "it is 100,000,000 in total"))
        self.assertFalse(ft.gt_leak("", "anything"))

    def test_quotes(self):
        self.assertEqual(ft.find_quote("a  b\nc d", "b c"), (3, 6))
        self.assertIsNone(ft.find_quote("abc", "ABC"))
        text = "x" * 1200 + " " + LONG_MIDDLE + " " + "y" * 1200
        self.assertNotIn(LONG_MIDDLE, ft.cut(text, ft.STEP_CAP))
        cap = ft.quote_min_cap(text, LONG_MIDDLE)
        self.assertIn(LONG_MIDDLE, ft.cut(text, cap))
        self.assertNotIn(LONG_MIDDLE, ft.cut(text, cap - 1))
        self.assertIsNone(ft.quote_min_cap("short text here", "absent"))

    def test_render_view(self):
        v = ft.render_view(self.t)
        self.assertIn("Step 3 [Analyst]: Let us merge", v)
        self.assertNotIn(LONG_MIDDLE, v)
        self.assertIn("[teacher only] Ground truth", v)
        self.assertNotIn("Ground truth", ft.render_view(self.t, teacher_info=False))


class TestAssembleAndChecks(unittest.TestCase):
    def setUp(self):
        self.t = traj_from(make_raw())
        self.src = fake_teacher(self.t)

    def check(self, src):
        return ft.check_tree(self.t, src)

    def test_clean_tree_passes(self):
        res = self.check(self.src)
        self.assertEqual(res["failures"], [])
        self.assertTrue(res["pass"])
        self.assertEqual(res["chain"], [6, 5, 3])
        self.assertTrue(res["answer_rule"]["agree"])

    def test_pointer_rendering_and_answers(self):
        nodes = ft.assemble(self.t, self.src)
        for nd in nodes:
            self.assertEqual(set(nd), {"id", "type", "question", "options", "answer", "parents", "cite_step", "quote"}
                             | ({"option_steps"} if nd["type"] in ft.POINTER_TYPES else set()))
            if nd["type"] in ft.POINTER_TYPES:
                for k, o in zip(nd["option_steps"], nd["options"]):
                    self.assertEqual(o, f"Step {k} [{self.t['steps'][k]['agent']}]")
        by = {nd["id"]: nd for nd in nodes}
        self.assertEqual(by["n3"]["options"][by["n3"]["answer"]], "Step 3 [Analyst]")
        self.assertEqual(by["n4"]["options"][by["n4"]["answer"]],
                         "Merge DVD and Blu-Ray into one group and take its oldest item")
        self.assertEqual(by["n6"]["options"][by["n6"]["answer"]], "Analyst")

    def test_shuffle_is_seeded(self):
        a = ft.assemble(self.t, self.src, seed=1)
        self.assertEqual(a, ft.assemble(self.t, self.src, seed=1))
        b = [ft.assemble(self.t, self.src, seed=s) for s in range(2, 12)]
        self.assertTrue(any(x != a for x in b))
        # correct option positions are spread over seeds
        pos = {nd["answer"] for x in b for nd in x if nd["id"] == "n4"}
        self.assertGreater(len(pos), 1)

    def mutate(self, fn):
        s = copy.deepcopy(self.src)
        fn(s)
        return self.check(s)

    def assertFails(self, res, code):
        self.assertFalse(res["pass"])
        self.assertTrue(any(f.startswith(code) for f in res["failures"]), res["failures"])

    def test_schema(self):
        self.assertFails(self.mutate(lambda s: s["nodes"][0].update(type="guess")), "schema")
        self.assertFails(self.mutate(lambda s: s["nodes"][3].pop("correct")), "schema")
        self.assertFails(self.mutate(lambda s: s["nodes"][1].update(answer_step="5")), "schema")
        self.assertFails(self.check({"nodes": "x"}), "schema")

    def test_pointer_range_and_chain(self):
        self.assertFails(self.mutate(lambda s: s["nodes"][1]["distractor_steps"].append(99)), "pointer_range")
        self.assertFails(self.mutate(lambda s: s["nodes"][2].update(answer_step=2, distractor_steps=[0, 3, 4],
                                                                      quote="Loaded the sheet; the Title")),
                         "chain_not_mistake")
        self.assertFails(self.mutate(lambda s: s["nodes"][1].update(answer_step=7, distractor_steps=[1, 2, 4],
                                                                      quote="TERMINATE TERMINATE")), "chain")

    def test_parents_acyclic(self):
        self.assertFails(self.mutate(lambda s: s["nodes"][1].update(parents=["n3"])), "parent_order")
        self.assertFails(self.mutate(lambda s: s["nodes"][1].update(parents=["n9"])), "parent_order")
        self.assertFails(self.mutate(lambda s: s["nodes"].__setitem__(1, dict(s["nodes"][1], id="n1"))), "ids_dup")

    def test_quotes_checked_in_kept_text(self):
        self.assertFails(self.mutate(lambda s: s["nodes"][3].update(quote="merge DVD and VHS")), "quote_not_found")
        self.assertFails(self.mutate(lambda s: s["nodes"][3].update(quote="merge")), "quote_short")
        # a quote that is only in the cut-away middle of a long step is rejected
        self.assertFails(self.mutate(lambda s: s["nodes"][2].update(answer_step=4, distractor_steps=[0, 2, 3],
                                                                      quote=LONG_MIDDLE)), "quote_not_found")
        # task quotes (cite_step null) are checked against the task text
        self.assertFails(self.mutate(lambda s: s["nodes"][4].update(quote="oldest DVD in the spreadsheet")),
                         "quote_not_found")
        res = self.mutate(lambda s: s["nodes"][3].update(quote="merge   DVD and\nBlu-Ray into one Disc group"))
        self.assertTrue(res["pass"], res["failures"])

    def test_options(self):
        self.assertFails(self.mutate(lambda s: s["nodes"][3]["distractors"].__setitem__(
            0, "merge dvd and blu-ray into one group and take its  oldest item")), "options_dup")
        self.assertFails(self.mutate(lambda s: s["nodes"][3].update(distractors=["only one"])), "option_count")
        self.assertFails(self.mutate(lambda s: s["nodes"][1].update(distractor_steps=[1, 1, 2])), "options_dup")

    def test_length_tell_and_rank(self):
        res = self.mutate(lambda s: s["nodes"][3].update(
            correct="Merge DVD and Blu-Ray into one Disc group, forward-fill the section markers, and take the "
                    "oldest item of the merged group as the answer"))
        self.assertFails(res, "length_tell")
        self.assertEqual(res["length_rank"]["n4"], 1)
        self.assertIn(self.check(self.src)["length_rank"]["n4"], {1, 2, 3, 4})

    def test_gt_and_meta_leaks(self):
        self.assertFails(self.mutate(lambda s: s["nodes"][4]["distractors"].__setitem__(
            0, "The oldest Blu-Ray is Time-Parking 2, which the team skipped")), "gt_leak")
        self.assertFails(self.mutate(lambda s: s["nodes"][1].update(
            question="Per the ground truth, which step printed the title?")), "meta_leak")
        self.assertFails(self.mutate(lambda s: s["nodes"][4].update(question="Why does FM-1.1 apply here?")),
                         "meta_leak")

    def test_self_leak_and_step_refs(self):
        self.assertFails(self.mutate(lambda s: s["nodes"][2].update(question="Did Step 3 propose the merge?")),
                         "self_leak")
        self.assertFails(self.mutate(lambda s: s["nodes"][3].update(question="What did Step 3 [Coder] propose?")),
                         "step_ref_bad")
        self.assertFails(self.mutate(lambda s: s["nodes"][3].update(question="What did Step 30 propose?")),
                         "step_ref_bad")

    def test_rule_agreement(self):
        def f(s):
            s["answer_step"] = 5
            s["nodes"][0].update(answer_step=5, distractor_steps=[6, 2, 7], quote="Code output: Cereal Killer III")
            s["nodes"][1].update(answer_step=4, distractor_steps=[1, 2, 3], quote="x" * 20)
            s["nodes"][2].update(distractor_steps=[0, 2, 1])
        res = self.mutate(f)
        self.assertFails(res, "answer_rule_disagree")
        self.assertEqual(res["answer_rule"], {"rule": 6, "how": "marker", "teacher": 5, "agree": False})

    def test_agent_checks(self):
        self.assertFails(self.mutate(lambda s: s["nodes"][5].update(answer_agent="Coder",
                                                                      distractor_agents=["Planner", "Analyst"])),
                         "agent_wrong")
        self.assertFails(self.mutate(lambda s: s["nodes"][5].update(distractor_agents=["Planner", "Nobody"])),
                         "agent_option_unknown")
        self.assertFails(self.mutate(lambda s: s["nodes"].pop(5)), "type_count")
        raw = make_raw(agent="Coder")            # labelled agent is not the speaker of the decisive step
        t = traj_from(raw)
        src = fake_teacher(t)
        src["nodes"][5].update(answer_agent="Coder", distractor_agents=["Planner", "Analyst"])
        res = ft.check_tree(t, src)
        self.assertFails(res, "agent_speaker_mismatch")

    def test_content_must_cite_decisive_step(self):
        self.assertFails(self.mutate(lambda s: s["nodes"][3].update(cite_step=2, quote="Loaded the sheet; the Title")),
                         "content_cite")

    def test_mistake_at_answer_step(self):
        raw = make_raw(ms=6)
        t = traj_from(raw)
        src = fake_teacher(t)
        src["nodes"] = [src["nodes"][0],
                        dict(src["nodes"][3], parents=["n1"], cite_step=6, quote="Final answer: Cereal Killer III"),
                        src["nodes"][4], dict(src["nodes"][5], parents=["n1"])]
        res = ft.check_tree(t, src)
        self.assertTrue(res["pass"], res["failures"])
        self.assertEqual(res["chain"], [6])

    def test_question_probe_and_grounding(self):
        res = self.check(self.src)
        p = res["question_probe"]["n5"]
        self.assertGreater(p["q_recall"], 0)
        self.assertIn("question_only", p)
        self.assertEqual(res["grounding"]["n4"]["cited"], 3)
        self.assertEqual(res["grounding"]["n4"]["correct_best_step"], 3)

    def test_teacher_drop(self):
        rec = ft.output_record(self.t, {"traj_id": self.t["traj_id"], "teacher": "fake", "drop": "label unsupported"})
        self.assertFalse(rec["checks"]["pass"])
        self.assertEqual(rec["checks"]["failures"], ["teacher_drop: label unsupported"])

    def test_short_quote_only_as_whole_step(self):
        raw = make_raw()
        raw["history"][6]["content"] = "Cereal"
        t = traj_from(raw)
        src = fake_teacher(t)
        src["nodes"][0]["quote"] = "Cereal"
        res = ft.check_tree(t, src)
        self.assertFalse(any(f.startswith("quote_short") for f in res["failures"]), res["failures"])
        src["nodes"][1]["quote"] = "Cereal"                       # a short part of a longer step
        self.assertTrue(any(f.startswith("quote_short: n2") for f in ft.check_tree(t, src)["failures"]))

    def test_output_record(self):
        rec = ft.output_record(self.t, self.src, extra_meta={"in_fa1_pool": False})
        self.assertEqual(set(rec), {"traj_id", "subset", "teacher", "nodes", "checks", "meta"})
        self.assertEqual(rec["teacher"], "fake")
        self.assertTrue(rec["checks"]["pass"])
        bad = copy.deepcopy(self.src)
        bad["nodes"][0]["type"] = "nope"
        rec = ft.output_record(self.t, bad)
        self.assertFalse(rec["checks"]["pass"])
        self.assertEqual(rec["nodes"], [])


class TestScript(unittest.TestCase):
    def test_sample_render_assemble(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw_root = td / "raw"
            for i in range(9):
                write_raw(raw_root, "alg", f"trajectory_{i:04d}", dict(make_raw(), question=f"Q{i}: the oldest Blu-Ray in the spreadsheet?"))
                write_raw(raw_root, "hc", f"trajectory_{i:04d}", dict(make_raw(), question=f"H{i}: the oldest Blu-Ray in the spreadsheet?"))
            bad = make_raw(agent="Coder")
            write_raw(raw_root, "alg", "trajectory_0100", bad)            # speaker != mistake_agent: never sampled
            script = str(ROOT / "scripts/extract_fa_trees.py")
            ids = td / "ids.json"
            run = lambda *a: subprocess.run([sys.executable, script, *a], check=True, capture_output=True, text=True)
            run("sample", "--raw", str(raw_root), "--n-alg", "3", "--n-hc", "3", "--out", str(ids))
            rows = json.loads(ids.read_text())
            self.assertEqual(len(rows), 6)
            self.assertNotIn("alg/trajectory_0100", [r["traj_id"] for r in rows])
            run("render", "--raw", str(raw_root), "--ids", str(ids), "--out-dir", str(td / "views"))
            self.assertEqual(len(list((td / "views").glob("*.txt"))), 6)
            src_dir = td / "src"
            src_dir.mkdir()
            for r in rows:
                sub, stem = r["traj_id"].split("/")
                t = ft.load_traj(raw_root / ft.SUBSET_DIRS[sub] / "train" / f"{stem}.json", sub)
                src = fake_teacher(t)
                if r is rows[0]:
                    src["nodes"][3]["quote"] = "not in the log at all"
                (src_dir / f"{sub}__{stem}.json").write_text(json.dumps(src))
            prefix = str(td / "out" / "pilot_")
            run("assemble", "--raw", str(raw_root), "--ids", str(ids), "--src-dir", str(src_dir), "--prefix", prefix,
                "--teacher", "fake")
            summ = json.loads(Path(prefix + "summary.json").read_text())
            self.assertEqual((summ["n"], summ["passed"]), (6, 5))
            self.assertEqual(summ["failures_by_code"], {"quote_not_found": 1})
            n_out = sum(len(Path(f"{prefix}{s}.jsonl").read_text().splitlines()) for s in ("alg", "hc"))
            self.assertEqual(n_out, 5)
            self.assertEqual(len(Path(prefix + "rejected.jsonl").read_text().splitlines()), 1)
            self.assertEqual(summ["spend_usd"], 0.0)
            self.assertEqual(set(summ["md5"]), {"pilot_alg.jsonl", "pilot_hc.jsonl", "pilot_rejected.jsonl"})
            line = json.loads(Path(prefix + "alg.jsonl").read_text().splitlines()[0])
            self.assertEqual(line["teacher"], "fake")
            for nd in line["nodes"]:
                self.assertIsInstance(nd["answer"], int)


if __name__ == "__main__":
    unittest.main()
