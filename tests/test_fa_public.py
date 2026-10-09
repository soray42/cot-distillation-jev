"""Tier-1 public failure-attribution data (src/cotdistill/fa_public.py, scripts/build_fa_public.py): the readers' index
conventions on synthetic records, rendering (the build_whowhen_eval cut, the per-trajectory cap), the nodes (balance,
truth, DAG), the fa_public branch of treeplan.statement_block (every other source unchanged against e7bc9f4, the last
commit before it), option symbols beyond Z in the trainer (letter_token_ids, permute_options, label_logits), a small
end-to-end build on synthetic raw data (deterministic), and, when data/student_fa1 exists, audits of the built files.
CPU only:

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/sev/bin/python -m unittest tests/test_fa_public.py
"""
import importlib.util
import io
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill import fa_public as fp  # noqa: E402
from cotdistill import treeplan  # noqa: E402

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

BASE = "e7bc9f4"          # the last commit before the fa_public branch of treeplan.statement_block
KK = json.loads((ROOT / "tests/fixtures/kk_local_item.json").read_text())
QWEN_TOK = ROOT / "results/weights/T1-Qwen3.5-2B-Base-student_v3-s2/model/tokenizer.json"
DATA = Path(os.environ.get("COTD_DATA_DIR") or ROOT / "data")


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rec(n: int = 8, dec: int = 3, agents=("Planner", "Coder", "Checker"), mode: str | None = "R.3", answer="42",
        text=lambda i: f"message {i} " * (5 + 3 * i), **kw) -> dict:
    steps = [(agents[i % len(agents)], text(i)) for i in range(n)]
    return fp._rec(kw.pop("uid", f"u{n}-{dec}"), kw.pop("origin", "wwpro"), kw.pop("subset", "mathchat"), "bench",
                   kw.pop("task", "Compute the total."), answer, steps, dec,
                   None if kw.get("single") else agents[dec % len(agents)], mode=mode, **kw)


# ------------------------------------------------------------------------------------------------- synthetic raw data
def ww_line(i: int, fw: str) -> dict:
    """One Who&When Pro row of framework fw (the field layout of text.jsonl)."""
    q = f"Task number {i} asks for the value of a quantity {i} in a long enough sentence for eight grams to exist."
    if fw == "smolagents":
        gt_step = i % 4                              # 0 = the planning entry
        traj = [{"kind": "user", "content": q}, {"kind": "planning", "plan": f"plan {i}"}] + [
            {"kind": "action", "step_number": k, "reasoning": f"Thought: step {k}", "code": f"x = {k}",
             "observation": f"Execution logs: {k}", "error": None, "is_final_answer": k == 4} for k in range(1, 5)]
        gt = {"agent": None, "step": gt_step, "mode": "R.2"}
        ex = {}
    elif fw == "alfagent":
        traj = [{"kind": "user", "content": q}] + [
            {"kind": "action", "step_number": k, "reasoning": "Repetitive loop: injected" if k == 3 else
             (f"think {k}" if k < 3 else None), "action": f"go to desk {k}", "observation": f"desk {k}",
             "is_final_answer": False} for k in range(1, 6)]
        gt = {"agent": None, "step": 3, "mode": "A.4"}
        ex = {}
    elif fw == "magentic-one":
        traj = [{"kind": "user", "content": q}]
        for r in range(3):
            for p, (a, role) in enumerate((("orchestrator", "Orchestrator"), ("websurfer", "WebSurfer"))):
                traj.append({"kind": "agent", "agent": a, "role": role, "round": r, "position": p,
                             "output": f"{role} says {r}.{p}", "ledger": "{\"secret\": 1}",
                             "phase": json.dumps(["injected"] if (r, p) == (1, 1) else ["replayed"])})
        gt = {"agent": "websurfer", "step": "1.1", "mode": "R.1"}
        ex = {"agents": [{"id": "orchestrator", "role": "Orchestrator"}, {"id": "websurfer", "role": "WebSurfer"}]}
    elif fw == "mathchat":
        traj = [{"kind": "user", "content": q}] + [
            {"agent_id": ("user_proxy", "assistant")[k % 2], "role": ("user_proxy", "assistant")[k % 2], "round": k // 2,
             "position": k % 2, "output": f"turn {k}", "is_injected": k == 3, "is_replayed": k < 3} for k in range(5)] + [
            {"kind": "final_answer", "content": "7"}]
        gt = {"agent": "assistant", "round": 1, "position": 1, "mode": "R.3"}
        ex = {"agents": [{"id": "assistant", "role": "assistant"}, {"id": "user_proxy", "role": "user_proxy"}]}
    elif fw == "debate":
        two = i % 5 == 0
        turns = lambda r: json.dumps([{"agent_id": f"agent_{a}", "output": f"r{r} a{a}", "is_replayed": False,  # noqa
                                       "is_injected": r == 0 and (a == 1 or (two and a == 2))} for a in range(3)])
        traj = [{"kind": "user", "content": q}, {"kind": "round", "round": 0, "turns": turns(0)},
                {"kind": "round", "round": 1, "turns": turns(1)}, {"kind": "final_answer", "content": "B"}]
        gt = {"agents": ["agent_1", "agent_2"] if two else ["agent_1"], "round": 0, "mode": "R.4"}
        ex = {"agents": [{"id": f"agent_{a}", "role": "debater"} for a in range(3)]}
    else:
        raise ValueError(fw)
    return {"id": f"{fw}_{i:04d}", "framework": fw, "benchmark": "bench", "task": json.dumps({"query": q, "answer": "7"}),
            "trajectory": json.dumps(traj), "ground_truth": json.dumps(gt), "extras": json.dumps(ex)}


def write_raw(d: Path, n: int = 12) -> None:
    """Synthetic raw data for every reader but AFTraj (pandas): Who&When Pro, StepFinder, AgenTracer, TraceElephant
    and Who&When (one Who&When question equals a StepFinder question, which must be removed from the pool)."""
    (d / "whowhen_pro").mkdir(parents=True)
    with open(d / "whowhen_pro/text.jsonl", "w") as f:
        for i in range(n):
            for fw in ("smolagents", "alfagent", "magentic-one", "mathchat", "debate"):
                f.write(json.dumps(ww_line(i, fw)) + "\n")
    leak_q = "Which museum in the old town holds the largest collection of maps drawn before the year eighteen hundred?"
    for sub in ("Algorithm-Generated", "Hand-Crafted"):
        p = d / "stepfinder/repo/data" / sub / "train"
        p.mkdir(parents=True)
        for i in range(n):
            q = leak_q if i == 0 else f"StepFinder question {i} of {sub} about a city with a river and {i} bridges total."
            hist = [{"step_id": str(k), "role": "assistant", "name": ("Planner_Expert", "Computer_terminal")[k % 2],
                     "content": f"content {k}", "is_mistake": str(int(k == 2))} for k in range(6)]
            x = {"question": q, "ground_truth": "3", "history": hist, "mistake_agent": "Planner_Expert",
                 "mistake_step": "2", "mistake_reason": "At step 2 the planner erred."}
            (p / f"trajectory_{i:04d}.json").write_text(json.dumps(x))
    (d / "agentracer").mkdir()
    with zipfile.ZipFile(d / "agentracer/agentracer-data-v1.0.0.zip", "w") as z:
        for dom in ("agentic", "coding", "math"):
            for sp in ("train", "test"):
                buf = io.StringIO()
                for i in range(n if sp == "train" else 3):
                    one = 1 if dom == "agentic" else 0             # agentic step fields are 1-based
                    names = ["assistant"] * 5 if dom == "agentic" else ["Team Leader", "Engineer"] * 2 + ["Engineer"]
                    h = [{"step": str(k + one), "role": "assistant", "name": names[k], "content": f"{dom} step {k}"}
                         for k in range(5)]
                    buf.write(json.dumps({"question": f"AgenTracer {dom} {sp} task {i} with a long text of eight words.",
                                          "question_ID": f"q{i}", "history": h, "mistake_step": str(2 + one),
                                          "mistake_agent": names[2], "ground_truth": "ok"}) + "\n")
                z.writestr(f"agentracer-data-v1.0.0/{dom}/{sp}.jsonl", buf.getvalue())
    (d / "traceelephant").mkdir()
    with zipfile.ZipFile(d / "traceelephant/data.zip", "w") as z:
        for i in range(4):
            run = f"data/captain-runs-gaia/gaia_task_{i}_x"
            md = {"task_id": f"0000000{i}-0000-0000-0000-000000000000", "task_instruction": f"TraceElephant task {i} "
                  "about a train timetable and the number of stops between two named stations today.",
                  "mistake_agent": "Expert", "mistake_step": "Step 3" if i == 1 else "3", "ground_truth": "5"}
            sr = [{"step_id": str(k + 1), "agent_name": ("Orchestrator", "Expert")[k % 2],
                   "output": {"choices": [{"message": {"role": "assistant", "content": f"out {k}"}}]},
                   "tool_logs": None, "input": {"messages": []}} for k in range(5)]
            z.writestr(run + "/trace_metadata.json", json.dumps(md))
            z.writestr(run + "/step_records.json", json.dumps(sr))
    ww = d / "who_and_when/Who&When/Algorithm-Generated"
    ww.mkdir(parents=True)
    (ww / "1.json").write_text(json.dumps({"question": leak_q, "question_ID": "abc"}))
    (d / "who_and_when/Who&When/Hand-Crafted").mkdir(parents=True)


class TestReaders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(dir=ROOT, prefix=".fa1_test_")
        cls.raw = Path(cls.tmp.name) / "raw"
        write_raw(cls.raw)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_wwpro_conventions(self):
        import collections
        counts = collections.Counter()
        recs = list(fp.read_wwpro(str(self.raw / "whowhen_pro/text.jsonl"), counts))
        by = collections.defaultdict(list)
        for r in recs:
            by[r["subset"]].append(r)
        for r in by["smolagents"]:                  # plan = rendered step 0 = ground_truth.step 0; action k = step k
            i = int(r["uid"].split("_")[1])
            self.assertEqual(r["dec"], i % 4)
            self.assertEqual(r["steps"][0]["text"], f"plan {i}")
            self.assertTrue(r["single_agent"])
            self.assertIsNone(r["gold_agent"])
        for r in by["alfagent"]:                     # reasoning never rendered (it marks the injected step)
            self.assertEqual(r["dec"], 2)
            self.assertFalse(any("Repetitive" in s["text"] or "think" in s["text"] for s in r["steps"]))
        for r in by["magentic-one"]:                 # "1.1" -> round 1, position 1 -> rendered index 3
            self.assertEqual(r["dec"], 3)
            self.assertEqual(r["gold_agent"], "WebSurfer")
            self.assertEqual([s["agent"] for s in r["steps"][:2]], ["Orchestrator", "WebSurfer"])
            self.assertFalse(any("injected" in s["text"] or "secret" in s["text"] for s in r["steps"]))
        for r in by["mathchat"]:                     # is_injected entry; the final_answer entry is not a step
            self.assertEqual((r["dec"], r["gold_agent"], len(r["steps"])), (3, "assistant", 5))
        self.assertEqual(counts["wwpro/drop/debate_two_injected_agents"], 3)
        for r in by["debate"]:                       # turns flattened; agent_1 of round 0
            self.assertEqual((r["dec"], r["gold_agent"], len(r["steps"])), (1, "agent_1", 6))

    def test_other_readers(self):
        import collections
        counts = collections.Counter()
        sf = list(fp.read_stepfinder(str(self.raw / "stepfinder/repo/data"), counts))
        self.assertEqual(len(sf), 24)
        self.assertTrue(all(r["dec"] == 2 and r["gold_agent"] == "Planner_Expert" and not r["flags"] for r in sf))
        self.assertFalse(any("is_mistake" in s["text"] for r in sf for s in r["steps"]))
        ar = list(fp.read_agentracer(str(self.raw / "agentracer/agentracer-data-v1.0.0.zip"), counts))
        self.assertTrue(all(r["dec"] == 2 for r in ar))          # step field 3 (agentic, 1-based) / 2 (0-based)
        self.assertTrue(all(r["single_agent"] for r in ar if r["subset"] == "agentic"))
        self.assertEqual({r["split"] for r in ar if "-test-" in r["uid"]}, {"test"})
        te = list(fp.read_traceelephant(str(self.raw / "traceelephant/data.zip"), counts))
        self.assertEqual([r["dec"] for r in te], [2] * 4)          # step_id 3 (also "Step 3") -> index 2
        self.assertTrue(all(r["split"] == "test" for r in te))

    def test_aftraj_text(self):
        a = fp._aftraj_action('[{"name": "verify_answer", "arguments": "{\\"x\\": 1}", "id": "call_verify_shortcut"}]')
        self.assertEqual(a, 'Action: verify_answer({"x": 1})')
        c = fp._aftraj_content('[{"name": "compute", "result": "3", "call_id": "call_verify_slip"}]')
        self.assertEqual(c, "Result of compute: 3")
        self.assertNotIn("call_", fp._aftraj_action('{"name": "f", "id": "call_9"}'))


class TestRender(unittest.TestCase):
    def test_cut_is_build_whowhen_eval_cut(self):
        ww = load_script("build_whowhen_eval")
        rng = random.Random(0)
        for _ in range(200):
            t = "".join(rng.choice("ab \n") for _ in range(rng.randint(0, 400)))
            c = rng.randint(1, 300)
            self.assertEqual(fp.cut(t, c), ww.cut(t, c))
            self.assertEqual(fp.cut(fp.cut(t, 350), min(c, 350)), fp.cut(t, min(c, 350)))   # pre-cut is lossless
        self.assertTrue(fp.precut_ok(fp.PRECUT) and not fp.precut_ok(fp.PRECUT + 1))

    def test_layout_and_cap(self):
        r = rec(n=40, dec=30, text=lambda i: "word " * (50 * (i % 7)))
        step, agent = fp.make_items(r, "x", "fa1/wwpro", cap=120, with_gt=True, rng=random.Random(0))
        self.assertEqual(fp.audit_item(step), [])
        p = step["prompt"]
        self.assertTrue(p.startswith(fp.HEADER + "\n\nTask: Compute the total.\n" + fp.GT_PREFIX + "42\n\nStep 0 [Planner]: "))
        self.assertEqual(step["labels"], list(fp.SYMBOLS[:40]))
        self.assertEqual(step["gold_label"], "e")                 # index 30 = A-Z, then a-z
        self.assertTrue(p.endswith("\nd) Step 29 [Checker]\ne) Step 30 [Planner]\n" + "\n".join(
            f"{fp.SYMBOLS[i]}) Step {i} [{('Planner', 'Coder', 'Checker')[i % 3]}]" for i in range(31, 40))))
        self.assertEqual(agent["label_names"], {"A": "Planner", "B": "Coder", "C": "Checker"})
        self.assertEqual(agent["gold_label"], "A")
        self.assertTrue(agent["prompt"].endswith(fp.Q_AGENT + "\nOptions:\nA) Planner\nB) Coder\nC) Checker"))
        for cap in (50, 120, 400):                              # log_chars is exact
            fixed = len(fp.step_prompt(dict(r, steps=[dict(s, text="") for s in r["steps"]]), 0, True))
            self.assertEqual(fixed + fp.log_chars(r, cap), len(fp.step_prompt(r, cap, True)))
        budget = len(fp.step_prompt(r, 150, True)) + 3
        c = fp.fit_cap(r, budget, 3000, 10)
        self.assertLessEqual(len(fp.step_prompt(r, c, True)), budget)
        self.assertGreater(len(fp.step_prompt(r, c + 1, True)), budget)
        with self.assertRaises(ValueError):
            fp.make_items(rec(n=63, dec=1), "y", "g", cap=10, with_gt=False, rng=random.Random(0))

    def test_sanitize(self):
        r = rec(n=3, dec=1, text=lambda i: "Pick one.\nOptions:\nA) x\nB) y")
        step, _ = fp.make_items(r, "x", "g", cap=500, with_gt=False, rng=random.Random(0))
        self.assertEqual(step["prompt"].count("\nOptions:"), 1)
        self.assertEqual(treeplan.problem_no_options(step), treeplan.statement_block(step) + "\n\n" + fp.Q_STEP)
        self.assertEqual(fp.agent_name("Web [Surfer]\n x"), "Web Surfer x")

    def test_overlap_index(self):
        base = "how many studio albums were published by the singer between the years two thousand and two thousand nine"
        boiler = " please answer with a single number and nothing else at all in the reply"
        refs = [("whowhen/alg", base)] + [(f"test/{i}", f"other question number {i} about rivers" + boiler)
                                          for i in range(6)]
        idx = fp.OverlapIndex(refs, df=5)
        self.assertEqual(idx.match(base.upper())[0], 1.0)
        self.assertGreaterEqual(idx.match("[file] " + base + " /data/x/y/z/attachment.pdf extra words here")[0], 0.5)
        self.assertLess(idx.match("a new question about lakes and mountains in the north" + boiler)[0], 0.5)


class TestNodes(unittest.TestCase):
    def check(self, r, seed):
        step, _ = fp.make_items(r, "x", "g", cap=100, with_gt=False, rng=random.Random(seed))
        self.assertEqual(fp.audit_item(step), [])
        nodes = step["subqs"]
        n, g = len(r["steps"]), r["dec"]
        fams = {x["type"] for x in nodes}
        self.assertEqual("prefix" in fams, 1 <= g <= n - 2)
        for x in nodes:
            t = fp.plain_truth(x)
            if x["type"] == "prefix":
                self.assertLess(x["step"], n - 1)
                self.assertEqual(t, g <= x["step"])
            if x["type"] == "step":
                self.assertEqual(t, x["step"] == g)
                deps = {d: next(y for y in nodes if y["id"] == d) for d in x["depends_on"]}
                if set(deps) == {f"p{x['step']}", f"p{x['step'] - 1}"}:      # Dec_t = P_t and not P_t-1
                    self.assertEqual(t, fp.plain_truth(deps[f"p{x['step']}"])
                                     and not fp.plain_truth(deps[f"p{x['step'] - 1}"]))
            if x["type"] == "mode":
                self.assertEqual(x["depends_on"], [f"k{fp.category(x['mode'])}"])
            self.assertNotRegex(x["question"], r"(?i)inject|fault|reason for")
            self.assertEqual(x["p_cot"], float(x["truth"]))
        self.assertEqual(treeplan.node_depths(nodes), {x["id"]: x["node_depth"] for x in nodes})
        return nodes

    def test_families(self):
        yes = tot = 0
        for seed in range(60):
            n = random.Random(seed).randint(2, 30)
            g = random.Random(seed + 1).randrange(n)
            nodes = self.check(rec(n=n, dec=g, mode=random.Random(seed).choice(fp.TEXT_MODES)), seed)
            yes += sum(x["truth"] for x in nodes)
            tot += len(nodes)
        self.assertTrue(0.4 < yes / tot < 0.6)
        single = self.check(rec(n=5, dec=2, mode=None, single=True, agents=("Agent",)), 0)
        self.assertEqual({x["type"] for x in single}, {"prefix", "step"})
        macnet = rec(n=4, dec=1, mode="R.1", accepted=[{"step_coord": "1.0", "mode": "R.1"}])
        step, _ = fp.make_items(macnet, "m", "g", cap=100, with_gt=False, rng=random.Random(0))
        self.assertIn("accepted_predictions", step["fa"])
        self.assertNotIn("accepted", step["prompt"])


class TestStatementBlock(unittest.TestCase):
    def base_treeplan(self):
        src = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE}:src/cotdistill/treeplan.py"],
                             capture_output=True, text=True)
        if src.returncode != 0:
            self.skipTest(f"git commit {BASE} not readable")
        spec = importlib.util.spec_from_loader("treeplan_base", loader=None)
        mod = importlib.util.module_from_spec(spec)
        mod.__dict__["__package__"] = "cotdistill"
        exec(compile(src.stdout, "treeplan_base.py", "exec"), mod.__dict__)
        return mod

    def other_items(self) -> list[dict]:
        from cotdistill import fa_tier0, numgen as ng
        frames = [f for f in ng.FRAMES if f["id"] not in ng.HELD_OUT]
        p = ng.generate(frames[0], 3, random.Random(0))
        items = [KK, ng.final_item(p, "n0", "mix", random.Random(0))]
        rng = random.Random(1)
        while True:
            q = ng.generate(rng.choice(frames), 3, rng)
            res = fa_tier0.build_trajectory(q, "t", random.Random(2))
            if res is not None:
                items += list(fa_tier0.make_items(res, "f0", True, random.Random(0)))
                break
        finals = {"justlogic": "Question: Is the statement true, false, or uncertain given the passage?",
                  "folio": "Question: Is the statement true, false, or uncertain given the premises?",
                  "sharc": "Question: Based only on the rule text, the scenario and the earlier answers, how should "
                           "the user's question be answered?",
                  "returns": "Question: What should the agent do?", "expense": "Question: What should the agent do?",
                  "subscription": "Question: What should the agent do?", "rule": "What should the clerk do?",
                  "goldproof/fld": "Question: Based only on the facts, is the hypothesis proved, disproved, or "
                                   "neither (unknown)?",
                  "goldproof/boardgameqa": "Question: Based on the game state and the rules and preferences, does "
                                           "the cat eat the food?",
                  "goldproof/ruletaker": "Question: Does the statement follow from the context? What is not "
                                         "explicitly stated as true is considered false."}
        # pronto came after e7bc9f4 (its own check of the other entries: tests/test_pronto_tf.py)
        self.assertEqual(set(finals), set(treeplan.FINAL_QUESTION) - {"pronto"})
        for src, q in finals.items():
            items.append({"item_id": src, "source": src, "prompt": f"Fact one.\nFact two.\n\n{q}\nOptions:\nA) x\nB) y"})
        return items

    def test_fa_public(self):
        r = rec(n=30, dec=4, task="Question: Which option? A. one B. two")
        step, agent = fp.make_items(r, "x", "g", cap=80, with_gt=True, rng=random.Random(0))
        block = treeplan.statement_block(step)
        self.assertTrue(block.startswith(fp.HEADER + "\n\nTask: Question: Which option?"))
        self.assertEqual(step["prompt"], block + "\n\n" + fp.Q_STEP + "\nOptions:\n" + step["prompt"].split(
            "\nOptions:\n")[1])
        self.assertEqual(treeplan.problem_no_options(step), block + "\n\n" + fp.Q_STEP)
        self.assertEqual(treeplan.statement_block(agent), fp.render_log(r, 80, False))
        self.assertEqual(treeplan.problem_no_options(agent), fp.render_log(r, 80, False) + "\n\n" + fp.Q_AGENT)
        for bad in (step["prompt"].replace(fp.Q_STEP, "Question: Which step?"), step["prompt"].replace(fp.HEADER, "Log"),
                    step["prompt"].split("\nOptions:\n")[0],
                    step["prompt"].replace("\nStep 1 [", "\nOptions:\nStep 1 [")):
            with self.assertRaises(ValueError):
                treeplan.statement_block(dict(step, prompt=bad))

    def test_other_sources_unchanged(self):
        base = self.base_treeplan()
        items = self.other_items()
        for it in items:
            self.assertEqual(treeplan.statement_block(it), base.statement_block(it))
            self.assertEqual(treeplan.problem_no_options(it), base.problem_no_options(it))
        broken = [dict(KK, prompt=KK["prompt"].replace("\n\nQuestion: ", "\n\nAsk: ")),
                  dict(items[2], prompt=items[2]["prompt"].replace(fp.HEADER, "Log:")),
                  dict(items[-1], prompt=items[-1]["prompt"].replace("Options:", "Choices:"))]
        for it in broken:                                   # the same refusals, with the same messages
            for fn in ("statement_block", "problem_no_options"):
                with self.assertRaises(ValueError) as new:
                    getattr(treeplan, fn)(it)
                with self.assertRaises(ValueError) as old:
                    getattr(base, fn)(it)
                self.assertEqual(str(new.exception), str(old.exception))
        step, _ = fp.make_items(rec(), "x", "g", cap=80, with_gt=False, rng=random.Random(0))
        with self.assertRaises(ValueError):                 # e7bc9f4 had no fa_public branch
            base.statement_block(step)
        with self.assertRaises(ValueError):                 # unknown sources still refused
            treeplan.statement_block(dict(step, source="no_such_source"))

    @unittest.skipUnless(HAVE_TORCH, "needs torch")
    def test_trainer_views(self):
        from cotdistill.student import build_full_tree
        step, _ = fp.make_items(rec(n=12, dec=5), "x", "g", cap=80, with_gt=False, rng=random.Random(0))
        n = len(step["subqs"])
        block = treeplan.statement_block(step)
        for view, ctx in (("problem", step["prompt"]), ("statements", block),
                          ("statements+question", treeplan.problem_no_options(step))):
            ex = build_full_tree(step, mode="plain", view=view, rng=random.Random(0), cap=0, final="teacher")
            self.assertEqual(len(ex), 1 + n)
            self.assertEqual(ex[0].text, step["prompt"] + "\n\nAnswer:")
            self.assertEqual(ex[0].gold, step["labels"].index(step["gold_label"]))
            self.assertEqual(ex[0].labels, step["labels"])
            for e in ex[1:]:
                self.assertTrue(e.text.startswith(ctx + "\n\nIntermediate question: "))
        st: dict = {}
        sc = build_full_tree(step, mode="plain", view="statements", rng=random.Random(0), cap=0, scaffold_p=1.0,
                             mix_seed=0, mix_epoch=0, stats=st)
        self.assertEqual(len(sc), 1 + n)
        self.assertGreater(st.get("scaffold_views", 0), 0)       # derived nodes (prefix chain, step, mode) scaffolded


# ------------------------------------------------------------------------------------------------- trainer labels
class DigitTok:
    """Qwen-like toy tokenizer: a space before a digit is its own token (' 0' -> ' ', '0'); ' A' / ' a' are single."""
    pad_token_id = 0
    eos_token_id = 0
    bos_token = None

    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=True):
        parts = re.findall(r" ?[A-Za-z]+| |\d|[^\sA-Za-z\d]+|\n", text)
        return [self.vocab.setdefault(t, len(self.vocab) + 1) for t in parts]


@unittest.skipUnless(HAVE_TORCH, "needs torch + transformers")
class TestLabelsBeyondZ(unittest.TestCase):
    def test_letter_token_ids(self):
        from cotdistill import student as S
        tok = DigitTok()
        ids = S.letter_token_ids(tok, list(fp.SYMBOLS))
        self.assertEqual(ids[:52], [tok.encode(" " + c)[0] for c in fp.SYMBOLS[:52]])     # A-Z, a-z: ' X' as before
        self.assertEqual(ids[52:], [tok.encode(c)[0] for c in "0123456789"])              # digits: the bare token
        self.assertEqual(len(set(ids)), 62)
        with self.assertRaises(ValueError):
            S.letter_token_ids(tok, ["10"])
        if QWEN_TOK.exists():
            from transformers import PreTrainedTokenizerFast
            q = PreTrainedTokenizerFast(tokenizer_file=str(QWEN_TOK))
            qi = S.letter_token_ids(q, list(fp.SYMBOLS))
            self.assertEqual(qi[:26], [q.encode(" " + c, add_special_tokens=False)[0] for c in fp.SYMBOLS[:26]])
            self.assertEqual(qi[52:], [q.encode(c, add_special_tokens=False)[0] for c in "0123456789"])
            self.assertEqual(len(set(qi)), 62)

    def test_permute_options(self):
        from cotdistill import student as S
        old = re.compile(r"^([A-Z])\) (.*)$")
        rng = random.Random(0)
        for k in range(50):                                  # A-Z prompts: identical to the old option regex
            n = rng.randint(2, 26)
            labs = list(fp.SYMBOLS[:n])
            prompt = "Q\nOptions:\n" + "\n".join(f"{L}) opt {i}" for i, L in enumerate(labs))
            target = [1.0 if i == k % n else 0.0 for i in range(n)]
            new = S.permute_options(prompt, labs, target, k % n, random.Random(k))
            saved = S._OPT_LINE
            try:
                S._OPT_LINE = old
                ref = S.permute_options(prompt, labs, target, k % n, random.Random(k))
            finally:
                S._OPT_LINE = saved
            self.assertEqual(new, ref)
        labs = list(fp.SYMBOLS[:60])                         # beyond Z: permuted, target follows the option text
        prompt = "Q\nOptions:\n" + "\n".join(f"{L}) Step {i} [A]" for i, L in enumerate(labs))
        out = S.permute_options(prompt, labs, [float(i == 55) for i in range(60)], 55, random.Random(3))
        self.assertIsNotNone(out)
        p2, t2, g2 = out
        self.assertEqual(p2.split("\n")[2 + g2], f"{labs[g2]}) Step 55 [A]")
        self.assertEqual(t2[g2], 1.0)

    def test_label_logits_and_loss(self):
        from cotdistill import student as S
        tok = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        tok.pad_token = tok.eos_token
        cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                       tie_word_embeddings=True)
        torch.manual_seed(0)
        model = transformers.Qwen2ForCausalLM(cfg)
        model.eval()
        r = rec(n=60, dec=57, text=lambda i: f"m{i}")
        step, _ = fp.make_items(r, "x", "g", cap=20, with_gt=False, rng=random.Random(0))
        short, _ = fp.make_items(rec(n=5, dec=2), "y", "g", cap=20, with_gt=False, rng=random.Random(0))
        cache: dict = {}
        exs = [S.Example(S.FINAL_TEMPLATE.format(problem=it["prompt"]), it["labels"],
                         [it["teacher"][L] for L in it["labels"]], 1.0, "final", it["item_id"]) for it in (step, short)]
        with torch.no_grad():
            zs = S.label_logits(model, tok, exs, 4096, cache)
            body, head = S.text_parts(model)
            b = S.collate(tok, exs[1:], 4096, "cpu")
            h = body(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state[0, b["last"][0]]
            ids = [tok.encode(" " + L)[0] for L in short["labels"]]
            self.assertTrue(torch.allclose(zs[1], (h @ head.weight[ids].T).float(), atol=1e-5))   # A-Z as before
        self.assertEqual([z.shape[0] for z in zs], [60, 5])
        _, preds = S.evaluate(model, tok, [step, short], 4096, 2, {})
        self.assertEqual(len(preds[0]["probs"]), 60)
        self.assertAlmostEqual(sum(preds[0]["probs"]), 1.0, places=5)
        model.train()
        loss = S.kl_loss(S.label_logits(model, tok, exs, 4096, cache), exs)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


# ------------------------------------------------------------------------------------------------- builder
class TestBuilder(unittest.TestCase):
    def test_small_build(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".fa1_test_") as d:
            d = Path(d)
            write_raw(d / "raw", n=20)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
            cmd = [sys.executable, "-P", str(ROOT / "scripts/build_fa_public.py"), "--raw", str(d / "raw"), "--out",
                   str(d / "out"), "--eval-dir", str(d / "eval"), "--origins",
                   "wwpro,stepfinder,agentracer,traceelephant", "--n-val", "12", "--tokenizer", "missing.json",
                   "--caps", "{}", "--share", '{"wwpro": 0.5}']
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900, cwd=d)
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
            s = json.loads((d / "out/summary.json").read_text())
            self.assertEqual(s["audit"]["failures"], 0)
            self.assertGreaterEqual(s["dedup"].get("stepfinder/whowhen/alg", 0), 2)     # the planted question
            files = {Path(k).name: v for k, v in s["files"].items()}
            for name in ("train.jsonl", "val.jsonl", "train_agent.jsonl", "fa1_wwpro_test.jsonl",
                         "fa1_agentracer_test.jsonl", "fa1_traceelephant_test.jsonl", "fa1_wwpro_test_agent.jsonl"):
                self.assertGreater(files[name], 0, name)
            tasks = {}
            for name in ("out/train.jsonl", "out/val.jsonl", "eval/fa1_wwpro_test.jsonl"):
                with open(d / name) as f:
                    tasks[name] = {re.search(r"\nTask: (.*)\n", json.loads(x)["prompt"]).group(1) for x in f}
            self.assertFalse(tasks["out/train.jsonl"] & tasks["out/val.jsonl"])
            self.assertFalse(tasks["out/train.jsonl"] & tasks["eval/fa1_wwpro_test.jsonl"])
            self.assertFalse(any("museum in the old town" in t for t in tasks["out/train.jsonl"]))
            r2 = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900, cwd=d)
            self.assertEqual(r2.returncode, 0)
            self.assertEqual(s["md5"], json.loads((d / "out/summary.json").read_text())["md5"])   # deterministic


# ------------------------------------------------------------------------------------------------- built data
@unittest.skipUnless((DATA / "student_fa1/summary.json").exists(), "data/student_fa1 not built")
class TestBuiltData(unittest.TestCase):
    @staticmethod
    def task(prompt: str) -> str:
        head = prompt.split("\n\nStep 0 [")[0][len(fp.HEADER + "\n\nTask: "):]
        return head.split("\n" + fp.GT_PREFIX)[0]

    def read(self, path: Path, limit: int | None = None) -> list[dict]:
        out = []
        with open(path) as f:
            for line in f:
                out.append(json.loads(line))
                if limit and len(out) >= limit:
                    break
        return out

    def test_files(self):
        s = json.loads((DATA / "student_fa1/summary.json").read_text())
        self.assertEqual(s["audit"]["failures"], 0)
        val = self.read(DATA / "student_fa1/val.jsonl")
        tests = {o: self.read(DATA / f"eval/fa1_{o}_test.jsonl") for o in ("wwpro", "aftraj", "agentracer",
                                                                            "traceelephant")}
        train_tasks = set()
        with open(DATA / "student_fa1/train.jsonl") as f:
            for line in f:
                it = json.loads(line)
                train_tasks.add(self.task(it["prompt"]))
                self.assertEqual(it["source"], fp.SOURCE)
        for name, items in [("val", val)] + list(tests.items()):
            for it in items:
                self.assertEqual(fp.audit_item(it), [], it["item_id"])
                self.assertEqual(treeplan.statement_block(it) + "\n\n" + fp.Q_STEP,
                                 treeplan.problem_no_options(it))
                self.assertNotIn(self.task(it["prompt"]), train_tasks, f"{name} {it['item_id']}")
        aft = {it["fa"]["uid"] for it in tests["aftraj"]}
        split = json.loads((DATA / "raw/aftraj/splits_test.json").read_text())["test_unsafe"] \
            if (DATA / "raw/aftraj/splits_test.json").exists() else None
        if split:
            self.assertEqual(aft, set(split))
        ww = json.loads((DATA / "student_fa1/norm/whowhen.json").read_text())
        idx = fp.OverlapIndex([("ww", x["question"]) for x in ww])
        hits = [t for t in train_tasks if idx.match(t)[0] >= 0.5]
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
