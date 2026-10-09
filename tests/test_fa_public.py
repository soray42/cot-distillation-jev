"""Tier-1 public failure-attribution data (src/cotdistill/fa_public.py, scripts/build_fa_public.py): the readers' index
conventions on synthetic records, rendering (the build_whowhen_eval cut, the per-trajectory cap), the nodes (balance,
truth, DAG), the fa_public branch of treeplan.statement_block (every other source unchanged against e7bc9f4, the last
commit before it), option symbols beyond Z in the trainer (letter_token_ids, permute_options, label_logits), a small
end-to-end build on synthetic raw data (deterministic), and, when data/student_fa1 exists, audits of the built files.
CPU only:

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/sev/bin/python -m unittest tests/test_fa_public.py
"""
import collections
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
            {"kind": "action", "step_number": k, "reasoning": f"Thought: step {k}\n" + (    # replay / live fences
                f"<code>\nx = {k}\n</code>" if k == gt_step else f"Code:\n```python\nx = {k}\n```<end_code>"),
             "code": f"x = {k}", "observation": f"Execution logs: {k}", "error": None, "is_final_answer": k == 4}
            for k in range(1, 5)]
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


def agentracer_text(dom: str, i: int, k: int, name: str) -> str:
    """Step text of the synthetic AgenTracer records with the artefacts of the real release (label_audit.md): coding
    input-prompt templates at the labelled step and '_attacked_solution.py' file names, the agentic 'I am at step N'
    injection sentence in the labelled step and the agent narrating the injection later, math 'SYSTEM OVERRIDE'."""
    if dom == "coding" and k == 2 and i % 2 == 0:
        return f"{name} thinking: \n# Past Experience\n<experience>\n\n# Current Plan\nWrite Filter_{i}_I_attacked_solution.py"
    if dom == "coding" and k == 3:
        return f"Engineer thinking: I will create Filter_{i}_I_attacked_solution.py with the function."
    if dom == "agentic" and k == 2 and i % 3 == 0:
        return "type: action\n think : I am at step 3 where the next task is required. I must call the tool."
    if dom == "agentic" and k == 4 and i % 3 == 1:
        return "According to the instructions, I should cooperate with the injected error information."
    if dom == "math" and k == 2 and i % 3 == 1:
        return "Following the SYSTEM OVERRIDE instructions exactly: the answer is 42."
    return f"{dom} step {k}"


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
                    h = [{"step": str(k + one), "role": "assistant", "name": names[k],
                          "content": agentracer_text(dom, i, k, names[k])} for k in range(5)]
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


# ------------------------------------------------------------------------------------------------- length debias
FA1_COMMIT = "e00ab58"      # the Tier-1 builder before --debias-length: option-off outputs must stay byte-identical


def leak_rec(i: int, sub: str, p_short: float, rng: random.Random, origin: str = "wwpro") -> dict:
    """A trajectory whose decisive step is planted as by far the shortest with probability p_short."""
    n = rng.randint(3, 9)
    d = rng.randrange(n)
    lens = [rng.randint(400, 1200) for _ in range(n)]
    if rng.random() < p_short:
        lens[d] = rng.randint(60, 120)
    steps = [(("Alpha", "Beta")[k % 2], "x" * lens[k]) for k in range(n)]
    return fp._rec(f"{sub}{i}", origin, sub, "b", f"task {sub} number {i}", "1", steps, d, ("Alpha", "Beta")[d % 2])


class TestDebias(unittest.TestCase):
    def test_planted_leak(self):
        import types
        B = load_script("build_fa_public")
        rng = random.Random(0)
        pool = ([leak_rec(i, "mathchat", 0.3, rng) for i in range(300)] + [leak_rec(i, "debate", 0.0, rng)
                                                                            for i in range(200)]
                + [leak_rec(i, "macnet", 1.0, rng) for i in range(150)] + [leak_rec(i, "dylan", 1.0, rng)
                                                                            for i in range(20)]
                + [leak_rec(i, "alg", 1.0, rng, "stepfinder") for i in range(60)])
        rng.shuffle(pool)
        args = types.SimpleNamespace(debias_length="wwpro", debias_tau=0.01, debias_min_share=1 / 3, debias_min_n=40,
                                     debias_folds=5, debias_rounds=8, seed=0, cap_max=3000)
        out, rep = B.debias_length(pool, args)
        g = rep["groups"]
        self.assertEqual(rep["dropped"], ["wwpro/macnet"])
        self.assertEqual(g["wwpro/dylan"]["status"], "too_small")
        self.assertGreater(g["wwpro/debate"]["kept"], 0.85 * 200)
        mc = g["wwpro/mathchat"]
        self.assertEqual(mc["status"], "selected")
        self.assertGreater(mc["probe_before"]["surface"] - mc["probe_before"]["structure"], 0.15)
        self.assertLessEqual(mc["probe_after"]["surface"] - mc["probe_after"]["structure"], 0.01)
        self.assertGreater(mc["before"]["decisive_shortest"], 0.4)
        self.assertLess(mc["after"]["decisive_shortest"], 0.3)
        self.assertGreater(mc["kept"], 300 / 3)
        self.assertEqual([r for r in out if r["origin"] == "stepfinder"],
                         [r for r in pool if r["origin"] == "stepfinder"])          # other origins untouched
        self.assertEqual(out, [r for r in pool if r in out])                       # pool order kept
        self.assertFalse(any(r["subset"] == "macnet" for r in out))
        out2, rep2 = B.debias_length(pool, args)                                   # deterministic
        self.assertEqual([r["uid"] for r in out2], [r["uid"] for r in out])

    def test_off_byte_identical_and_on_build(self):
        src = subprocess.run(["git", "-C", str(ROOT), "show", f"{FA1_COMMIT}:scripts/build_fa_public.py"],
                             capture_output=True, text=True)
        if src.returncode != 0:
            self.skipTest(f"git commit {FA1_COMMIT} not readable")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
        common = ["--origins", "wwpro,stepfinder,agentracer,traceelephant", "--n-val", "12", "--tokenizer",
                  "missing.json", "--caps", "{}", "--share", '{"wwpro": 0.5}']
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".fa1_test_") as d:
            d = Path(d)
            write_raw(d / "raw", n=60)
            base = d / "base"                    # e00ab58's script; its ROOT is base/ (outputs must lie under it)
            (base / "scripts").mkdir(parents=True)
            (base / "scripts/build_fa_public.py").write_text(src.stdout)
            (base / "src").symlink_to(ROOT / "src")

            def run(script: Path, out: Path, ev: Path, extra=()):
                cmd = [sys.executable, "-P", str(script), "--raw", str(d / "raw"), "--out", str(out), "--eval-dir",
                       str(ev)] + common + list(extra)
                r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900, cwd=d)
                self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
                return json.loads((out / "summary.json").read_text())
            run(base / "scripts/build_fa_public.py", base / "out", base / "eval")
            s_off = run(ROOT / "scripts/build_fa_public.py", d / "out", d / "eval")
            names = sorted(p.name for p in (base / "out").glob("*.jsonl")) + sorted(
                p.name for p in (base / "eval").glob("*.jsonl"))
            self.assertEqual(len(names), 4 + 9)
            for sub in ("out", "eval"):
                for p in sorted((base / sub).glob("*.jsonl")):
                    self.assertEqual(p.read_bytes(), (d / sub / p.name).read_bytes(), f"{sub}/{p.name}")
            self.assertNotIn("debias", s_off)
            self.assertIn("length_probe", s_off)
            s_on = run(ROOT / "scripts/build_fa_public.py", d / "out_b", d / "eval_b",
                       ["--stage", "build", "--norm", str(d / "out/norm"), "--debias-length", "wwpro,agentracer",
                        "--compare-eval", str((d / "eval").relative_to(ROOT))])
            self.assertEqual(s_on["audit"]["failures"], 0)
            self.assertEqual(s_on["debias"]["origins"], ["agentracer", "wwpro"])
            self.assertTrue(s_on["debias"]["groups"])
            self.assertEqual(set(s_on["eval_vs_compare"]["same_md5"].values()), {True})   # test sets untouched
            self.assertFalse((d / "out_b/norm").exists())


# ------------------------------------------------------------------------------------------------- cleaning
DEBIAS_COMMIT = "4a79d36"   # the builder with --debias-length, before --clean: clean-off outputs must stay byte-identical
ARTEFACTS = re.compile(r"attacked_solution|I am at step \d|system override|```python|</?code>|<end_code>", re.I)


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(x) for x in f]


class TestCleanRecord(unittest.TestCase):
    def ar(self, subset, texts, bench=None, **kw):
        return fp._rec(f"{subset}-1", "agentracer", subset, bench or subset, kw.pop("task", "Write a function."), "ok",
                       [("Engineer" if subset == "coding" else "assistant", t) for t in texts], 1,
                       "Engineer" if subset == "coding" else None, single=subset != "coding", **kw)

    def test_text_rules_everywhere(self):
        r = self.ar("coding", ["Team Leader: name it Leetcode_1_I_attacked_solution.py",
                               "Engineer thinking: I will create Leetcode_1_I_attacked_solution.py"],
                    task="Save as X_attacked_solution.py")
        c = collections.Counter()
        out, why = fp.clean_record(r, "v1", counts=c)
        self.assertEqual(why, "group:agentracer/coding/coding")
        self.assertEqual(out["flags"], ["label_unreliable"])
        self.assertFalse(any("attacked" in s["text"] for s in out["steps"]) or "attacked" in out["task"])
        self.assertIn("Leetcode_1_I_solution.py", out["steps"][1]["text"])
        self.assertEqual(c["attacked_filename"], 3)
        self.assertEqual(r["steps"][1]["text"], "Engineer thinking: I will create Leetcode_1_I_attacked_solution.py")
        self.assertEqual(r["flags"], [])                                          # the input record is not modified
        a = self.ar("agentic", ["search", "think : I am at step 3 where the next task is required. I must call it."],
                    bench="Diagnosis Error")
        out, why = fp.clean_record(a, "v1")
        self.assertEqual(out["steps"][1]["text"], "think : I must call it.")
        self.assertEqual(why, "meta_leak")                                        # judged on the original text
        m = self.ar("math", ["x", "Following the SYSTEM OVERRIDE instructions exactly: 42. Per the mandatory system "
                                  "override, done."])
        out, why = fp.clean_record(m, "v1")
        self.assertEqual(out["steps"][1]["text"], "Following the instructions exactly: 42. Per the instruction, done.")
        self.assertEqual(why, "meta_leak")
        self.assertEqual(fp.clean_record(self.ar("math", ["Let x be 2.", "So it is 4."]), "v1")[1],
                         "group:agentracer/math/math")
        ok = self.ar("agentic", ["search the web", "So the answer is 4."], bench="Diagnosis Error")
        out, why = fp.clean_record(ok, "v1")
        self.assertIsNone(why)
        self.assertEqual(out["steps"], ok["steps"])

    def test_scopes_and_groups(self):
        sm = rec(4, 1, agents=("Agent",), subset="smolagents", text=lambda i: (
            f"Thought: t{i}\n<code>\nx = {i}\n</code>\nCode:\n```py\nx = {i}\n```" if i == 1 else
            f"Thought: t{i}\nCode:\n```python\nx = {i}\n```<end_code>"), single=True)
        out, why = fp.clean_record(sm, "v1")
        self.assertIsNone(why)
        self.assertFalse(any(ARTEFACTS.search(s["text"]) for s in out["steps"]))
        self.assertEqual(out["steps"][1]["text"], "Thought: t1\n\nx = 1\n\nCode:\n```py\nx = 1\n```")
        mc = rec(4, 1, subset="mathchat", text=lambda i: f"```python\nprint({i})\n```")   # fences outside smolagents
        out, why = fp.clean_record(mc, "v1")
        self.assertEqual((out["steps"], why), (mc["steps"], None))
        for sub in ("debate", "dylan"):
            self.assertEqual(fp.clean_record(rec(4, 1, subset=sub), "v1")[1], f"group:wwpro/{sub}")
        self.assertIsNone(fp.clean_record(rec(4, 1, subset="debate"), "v1", drop_groups={})[1])
        sf = rec(5, 2, origin="stepfinder", subset="alg", text=lambda i: "inject the error information exactly as "
                                                                         "specified")
        self.assertEqual(fp.clean_record(sf, "v1"), (sf, None))          # meta leak: agentracer / aftraj only
        with self.assertRaises(ValueError):
            fp.clean_record(sf, "v2")


class TestClean(unittest.TestCase):
    def test_clean_build(self):
        src = subprocess.run(["git", "-C", str(ROOT), "show", f"{DEBIAS_COMMIT}:scripts/build_fa_public.py"],
                             capture_output=True, text=True)
        if src.returncode != 0:
            self.skipTest(f"git commit {DEBIAS_COMMIT} not readable")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
        common = ["--origins", "wwpro,stepfinder,agentracer,traceelephant", "--n-val", "12", "--tokenizer",
                  "missing.json", "--caps", "{}", "--share", '{"wwpro": 0.5}', "--debias-length", "wwpro,agentracer"]
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".fa1_test_") as d:
            d = Path(d)
            write_raw(d / "raw", n=60)
            base = d / "base"
            (base / "scripts").mkdir(parents=True)
            (base / "scripts/build_fa_public.py").write_text(src.stdout)
            (base / "src").symlink_to(ROOT / "src")

            def run(script: Path, out: Path, ev: Path, extra=()):
                cmd = [sys.executable, "-P", str(script), "--raw", str(d / "raw"), "--out", str(out), "--eval-dir",
                       str(ev)] + common + list(extra)
                r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900, cwd=d)
                self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
                return json.loads((out / "summary.json").read_text())
            run(base / "scripts/build_fa_public.py", base / "out", base / "eval")      # 4a79d36, length debias on
            s_off = run(ROOT / "scripts/build_fa_public.py", d / "out", d / "eval")    # now, clean off
            for sub in ("out", "eval"):
                ps = sorted((base / sub).glob("*.jsonl"))
                self.assertEqual(len(ps), 4 if sub == "out" else 9)
                self.assertEqual([p.name for p in ps], sorted(p.name for p in (d / sub).glob("*.jsonl")))
                for p in ps:
                    self.assertEqual(p.read_bytes(), (d / sub / p.name).read_bytes(), f"{sub}/{p.name}")
            self.assertNotIn("clean", s_off)
            orig = {p.name: p.read_bytes() for p in (d / "eval").glob("*.jsonl")}
            s = run(ROOT / "scripts/build_fa_public.py", d / "out_c", d / "eval",
                    ["--stage", "build", "--norm", str(d / "out/norm"), "--clean", "v1"])
            self.assertEqual(s["audit"]["failures"], 0)
            self.assertEqual({p.name: p.read_bytes() for p in (d / "eval").glob("*.jsonl") if p.name in orig}, orig)
            c = s["clean"]
            ef = c["eval_files"]
            self.assertTrue(ef["fa1_traceelephant_test_c.jsonl"].startswith("unchanged"))
            self.assertFalse((d / "eval/fa1_traceelephant_test_c.jsonl").exists())
            for o in ("agentracer", "wwpro"):                 # artefacts (agentracer) and smolagents fences (wwpro)
                self.assertEqual(ef[f"fa1_{o}_test_c.jsonl"], "written")
                self.assertTrue((d / f"eval/fa1_{o}_test_c.jsonl").exists())
            self.assertIn("agentracer/coding/coding", c["dropped"])
            self.assertIn("wwpro/debate", c["dropped"])
            for name in ("train.jsonl", "val.jsonl", "train_agent.jsonl", "val_agent.jsonl"):
                items = read_jsonl(d / "out_c" / name)
                self.assertTrue(items or name == "val_agent.jsonl", name)
                for it in items:
                    self.assertIsNone(ARTEFACTS.search(it["prompt"]), it["item_id"])
                    self.assertNotIn((it["fa"]["origin"], it["fa"]["subset"]), {("agentracer", "coding"),
                                                                                ("wwpro", "debate")})
            self.assertTrue(any(it["fa"]["origin"] == "agentracer" for it in read_jsonl(d / "out_c/train.jsonl")))
            old = {it["item_id"]: it for it in read_jsonl(d / "eval/fa1_agentracer_test.jsonl")}
            new = read_jsonl(d / "eval/fa1_agentracer_test_c.jsonl")
            leak = set(c["test"]["agentracer"]["dropped_meta_leak"]["uids"])
            self.assertTrue(leak)
            self.assertEqual(len(new), len(old) - len(leak))
            for it in new:                                    # ids of the uncleaned file, the same trajectory
                self.assertEqual(it["fa"]["uid"], old[it["item_id"]]["fa"]["uid"])
                self.assertEqual(it["gold_label"], old[it["item_id"]]["gold_label"])
                self.assertNotIn(it["fa"]["uid"], leak)
                self.assertIsNone(ARTEFACTS.search(it["prompt"]))
                self.assertEqual("label_unreliable" in it["fa"]["flags"], it["fa"]["subset"] == "coding")
            self.assertTrue(any(it["fa"]["subset"] == "coding" for it in new))   # dropped from train, kept in test
            ww = read_jsonl(d / "eval/fa1_wwpro_test_c.jsonl")
            self.assertEqual(len(ww), len(read_jsonl(d / "eval/fa1_wwpro_test.jsonl")))
            self.assertTrue(any(it["fa"]["subset"] == "debate" for it in ww))
            s2 = run(ROOT / "scripts/build_fa_public.py", d / "out_c", d / "eval",     # deterministic
                     ["--stage", "build", "--norm", str(d / "out/norm"), "--clean", "v1"])
            self.assertEqual(s["md5"], s2["md5"])
            r = subprocess.run([sys.executable, "-P", str(ROOT / "scripts/build_fa_public.py"), "--stage", "build",
                                "--norm", str(d / "out/norm"), "--out", str(d / "out_x"), "--eval-dir", str(d / "eval"),
                                "--clean", "v1", "--eval-suffix", ""], capture_output=True, text=True, env=env, cwd=d)
            self.assertNotEqual(r.returncode, 0)                                         # never over the originals


# ------------------------------------------------------------------------------------------------- n-gram probe
class TestNgramProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ng = load_script("fa_ngram_probe")

    def trajs(self, n, tell: float, role: bool = False, seed: int = 0):
        """n trajectories; with probability tell the gold step contains the word 'zeta'; with role, 'zeta' marks every
        step of agent Z instead and the gold step is always a Z step (a role cue the structure prior absorbs)."""
        rng = random.Random(seed)
        out = []
        for i in range(n):
            k = rng.randint(3, 8)
            agents = [rng.choice("XYZ") for _ in range(k)]
            if role:
                zs = [j for j, a in enumerate(agents) if a == "Z"] or [0]
                agents[zs[0]] = "Z"
                g = rng.choice(zs)
            else:
                g = rng.randrange(k)
            steps = []
            for j in range(k):
                w = [rng.choice(["alpha", "beta", "gamma", "delta"]) for _ in range(4)]
                if (role and agents[j] == "Z") or (not role and j == g and rng.random() < tell):
                    w.append("zeta")
                steps.append(" ".join(w))
            out.append({"id": str(i), "group": "x/y", "task": f"task {i}", "steps": steps, "agents": agents, "gold": g})
        return out

    def run_group(self, trajs, evals=None):
        H = self.ng.Hasher()
        for t in trajs + [t for ts in (evals or {}).values() for t in ts]:
            t["buckets"] = [H.buckets(s) for s in t["steps"]]
        args = type("A", (), {"folds": 5, "min_df": 10, "lams": [1e-3, 1e-2, 1e-1], "top": 5, "max_agents": 30})
        return self.ng.run_group(trajs, evals or {}, H, args)

    def test_hash_and_phrase_gains(self):
        import numpy as np
        H = self.ng.Hasher()
        t = "The quick brown fox 123 jumps over the lazy dog 7"
        b = set(H.buckets(t).tolist())
        self.assertEqual(set(H.grams_of_bucket([t], b)), b)                 # every bucket read back to its n-grams
        trajs = self.trajs(200, 0.5)
        for tr in trajs:
            tr["buckets"] = [H.buckets(s) for s in tr["steps"]]
        df = np.zeros(1 << self.ng.NB, dtype=np.int32)
        for tr in trajs:
            df[np.unique(np.concatenate(tr["buckets"]))] += 1
        kept = np.flatnonzero(df >= 5)
        fmap = np.full(1 << self.ng.NB, -1)
        fmap[kept] = np.arange(len(kept))
        B = self.ng.Block(trajs, fmap)
        p = np.random.RandomState(0).rand(B.S)
        rank, grank = self.ng.rank_layout(B, p)
        gp, gn, cov = self.ng.phrase_gains(B, rank, grank, len(kept))
        bp, bn, bc = (np.zeros(len(kept), int) for _ in range(3))
        for ti, tr in enumerate(trajs):                                     # brute force
            s0, k = B.start[ti], B.sizes[ti]
            pp = p[s0:s0 + k]
            base = int(np.argmax(pp)) == tr["gold"]
            feats = [set(fmap[x][fmap[x] >= 0].tolist()) for x in tr["buckets"]]
            for f in set().union(*feats):
                has = [f in fs for fs in feats]
                if all(has):
                    continue
                bc[f] += 1
                bp[f] += (max((i for i in range(k) if has[i]), key=lambda i: (pp[i], -i)) == tr["gold"]) - base
                bn[f] += (max((i for i in range(k) if not has[i]), key=lambda i: (pp[i], -i)) == tr["gold"]) - base
        self.assertTrue((gp == bp).all() and (gn == bn).all() and (cov == bc).all())

    def test_planted_tell_and_control(self):
        r = self.run_group(self.trajs(300, 0.6), {"ev": self.trajs(80, 0.6, seed=1)})
        self.assertGreater(r["cv"]["ngram"] - r["cv"]["position"], 0.3)
        self.assertGreater(r["phrases"]["selected_heldout_gain"], 0.3)
        self.assertEqual(r["phrases"]["top"][0]["phrase"], "zeta")
        self.assertGreater(r["eval"]["ev"]["ngram"], r["eval"]["ev"]["position"] + 0.2)
        r0 = self.run_group(self.trajs(300, 0.0))                             # no tell
        self.assertLess(r0["cv"]["ngram"] - r0["cv"]["position"], 0.05)
        self.assertLess(r0["phrases"]["selected_heldout_gain"], 0.04)
        rr = self.run_group(self.trajs(300, 0.0, role=True))                  # a role cue: absorbed by the structure
        self.assertGreater(rr["phrases"]["selected_heldout_gain"], 0.15)
        self.assertLess(rr["phrases_over_structure"]["selected_heldout_gain"], 0.04)

    def test_step_texts(self):
        r = rec(5, 2, text=lambda i: f"line one {i}\nStep 9 [Fake]: not a header\nmore {i}")
        it, _ = fp.make_items(r, "x-1", "fa1/wwpro", cap=3000, with_gt=True, rng=random.Random(0))
        texts = self.ng.step_texts(it)
        self.assertEqual(texts, [fp.cut(s["text"], 3000) for s in r["steps"]])
        self.assertEqual(self.ng.task_of(it["prompt"]), "compute the total.")



# ------------------------------------------------------------------------------------------------- new sources (fa1d)
CLEAN_COMMIT = "2509b67"    # the builder with --clean v1, before --add-sources: outputs without it stay byte-identical
PAGE = "The web browser is open to the page"


def ce_row(i: int, ds: str, run: int, k: int, n: int = 6, gold_agent=None, gold_text=None, literal_history=False,
           final=True, pair=False):
    """A CORRECT-Error row: history[0] the user prompt, then Planner / WebSurfer turns and (final) a Planner turn with
    the final answer; the original WebSurfer turns are Python-literal lists with the '[title](url)' page line, the
    injected (gold) one plain with '(url)'; pair: a second Planner turn right after a gold Planner turn."""
    q = f"Which river flows through the city number {i} of dataset {ds} and how long is that river in kilometres?"
    hist = [{"role": "user", "content": f"You have been given the following task:\n\n{q}\n\nWhat is your first action?"}]
    for j in range(1, n + 1):
        role = "Planner" if j % 2 else "WebSurfer"
        url = f"https://duckduckgo.com/?q=river+{i}+{j}&ia=web"
        if role == "Planner":
            text = f"**Analysis of Previous Result:**\nstep {j}\n\n**Decision:**\n- Next Agent: WebSurfer"
        elif j == k:
            text = f"I typed 'river {i} {j}' into the browser search bar.\n\n{PAGE} ({url}).\nThe viewport shows 20%"
        else:
            text = repr([f"I typed 'river {i} {j}' into the browser search bar.\n\n{PAGE} [river {i} {j} at "
                         f"DuckDuckGo]({url}).\nThe viewport shows 20%"])
        if j == k and gold_text is not None:
            text = gold_text
        hist.append({"role": role, "content": text})
        if pair and j == k:
            hist.append({"role": "Planner", "content": "**Analysis of Previous Result:**\nre-plan"})
    if final:
        hist.append({"role": "Planner", "content": f"**Analysis of Previous Result:**\nFINAL ANSWER: river {i}"})
    row = {"trajectory_id": f"gpt-5-nano_{ds}_task{i}_{run}", "dataset": ds, "generator_model": "gpt-5-nano",
           "question": q, "groundtruth": "", "question_id": f"task{i}_{run}", "history": hist,
           "mistake_agent": gold_agent or (hist[k]["role"] if k else "Unknown"), "mistake_step": k,
           "mistake_reason": "", "is_corrected": False, "level": 0}
    if literal_history:
        row["history"] = repr(hist)
    return row


def aegis_line(i: int, fw: str, faulty, split: str, task: str | None = None) -> dict:
    """An Aegis record; faulty = [(agent, mode), ...]; smoagents contents are Python-literal lists of text dicts and
    start with a System prompt entry."""
    q = task or f"Aegis {split} problem {i} for {fw}: compute the area of a triangle with sides {i}, {i + 3} and {i + 4}."
    if fw == "smoagents":
        names = ["System", "User", "Manager", "Searcher", "Manager"]
        hist = [{"step": k + 1, "agent_name": a, "phase": "reasoning",
                 "content": repr([{"type": "text", "text": f"{a} message {k} about {q[:20]}"}])}
                for k, a in enumerate(names)]
    else:
        names = ["Assistant1", "Assistant2", "Assistant3", "Assistant1", "Assistant2", "Assistant3"]
        hist = [{"step": k + 1, "agent_name": a, "phase": "reasoning", "content": f"{a} says {k}: the area is {k}."}
                for k, a in enumerate(names)]
        if fw == "macnet":
            for h in hist:
                h["is_injected"] = h["agent_name"] == faulty[0][0]
    fa = [{"agent_name": a, "error_type": m, "injection_strategy": "response_corruption"} for a, m in faulty]
    return {"id": f"{fw}_{split}_{i}", "metadata": {"framework": fw, "benchmark": "MATH", "model": "gpt-4o-mini"},
            "input": {"query": q, "conversation_history": hist, "final_output": "x"},
            "output": {"faulty_agents": fa},
            "ground_truth": {"correct_answer": "6", "injected_agents": fa, "is_injection_successful": True}}


def write_raw_new(d: Path, n: int = 30) -> None:
    """Synthetic CORRECT-Error (.jsonl with the parquet's fields) and Aegis files next to write_raw's sources."""
    rng = random.Random(7)
    (d / "correct_error").mkdir(parents=True)
    with open(d / "correct_error/test.jsonl", "w") as f:
        for i in range(n):
            for ds in ("hotpot", "wikimqa"):
                for run in (1, 2):                                   # one question under two run ids
                    nn = rng.choice((4, 6, 8))
                    f.write(json.dumps(ce_row(i, ds, run, rng.randint(1, nn), nn,
                                              literal_history=(i == 3 and run == 1))) + "\n")
        f.write(json.dumps(ce_row(900, "hotpot", 1, 0)) + "\n")                       # labelled user prompt
        f.write(json.dumps(ce_row(901, "hotpot", 1, 2, gold_text="INJECTION PLAN:\nError Pattern: x")) + "\n")
        f.write(json.dumps(ce_row(902, "hotpot", 1, 2, gold_agent="[WebSurfer]")) + "\n")
        f.write(json.dumps(ce_row(903, "hotpot", 1, 2, gold_agent="Coder")) + "\n")
        for i in range(904, 924):                               # flagged: out of train / val and the test files
            f.write(json.dumps(ce_row(i, "wikimqa", 1, 3, pair=i % 2 == 0, final=i % 2 == 1)) + "\n")
    for sub, split, m in (("Aegis-Training", "train", 3 * n), ("Aegis-Training", "val", n), ("Aegis-Bench", "test", n)):
        (d / "aegis" / sub).mkdir(parents=True, exist_ok=True)
        with open(d / "aegis" / sub / f"{split}.jsonl", "w") as f:
            for i in range(m):
                fw = ("dylan", "smoagents", "macnet")[i % 3]
                agents = ("Manager", "Searcher") if fw == "smoagents" else ("Assistant1", "Assistant2", "Assistant3")
                mode = sorted(fp.MAST_MAP)[i % 14]
                faulty = [(agents[i % len(agents)], mode)]
                if i % 7 == 0:
                    faulty.append((agents[(i + 1) % len(agents)], "FM-1.1"))           # two faulty agents: dropped
                shared = (split == "train" and i % 9 == 0) or (split == "test" and i < 5)
                task = f"Shared Aegis task {i % 5}: how many diagonals does a convex polygon with {i % 5 + 7} sides " \
                    "have in total?" if shared else None          # train tasks equal to test tasks: removed by dedup
                f.write(json.dumps(aegis_line(i, fw, faulty, split, task)) + "\n")
            if split == "test":
                x = aegis_line(999, "dylan", [("Assistant1", "FM-2.3")], split)
                x["input"]["conversation_history"][1]["content"] = "Error in inject_prompt: Error code: 400"
                f.write(json.dumps(x) + "\n")


class TestNewReaders(unittest.TestCase):
    def test_literal_text(self):
        self.assertEqual(fp.literal_text(repr(["a\nb", "c"])), ("a\nb\nc", True))
        self.assertEqual(fp.literal_text(repr([{"type": "text", "text": "x"}, {"type": "text", "text": "y"}])),
                         ("x\ny", True))
        for s in ("plain text", "[1, 2]", "['a', 1]", "[__import__('os').system('true')]", "['unterminated",
                  "[{'type': 'image'}]"):
            self.assertEqual(fp.literal_text(s), (s, False))
        self.assertEqual(fp.literal_text(None), ("", False))

    def test_correct_records(self):
        counts = collections.Counter()
        rows = [ce_row(1, "hotpot", 1, 4, 6), ce_row(2, "arc", 1, 0), ce_row(3, "arc", 2, 3, 4, literal_history=True),
                ce_row(4, "hotpot", 1, 2, gold_text="Agent: WebSurfer\nContent: I typed 'x'.\n\n" + PAGE +
                       " river at DuckDuckGo."),
                ce_row(5, "hotpot", 1, 2, gold_text="I’m sorry, but I can’t help create or insert deceptive errors."),
                ce_row(6, "hotpot", 1, 2, gold_agent="[WebSurfer]"), ce_row(7, "hotpot", 1, 2, gold_agent="Coder")]
        recs = {r["uid"]: r for r in fp.correct_records(rows, counts)}
        self.assertEqual(counts["correct/drop/user_step"], 1)
        self.assertEqual(counts["correct/drop/meta_leak"], 1)
        r = recs["gpt-5-nano_hotpot_task1_1"]
        self.assertEqual((r["dec"], len(r["steps"]), r["gold_agent"]), (3, 7, "WebSurfer"))   # history index 4 -> 3
        self.assertEqual(r["flags"], [])
        self.assertEqual(r["qid"], "hotpot/task1")
        self.assertIsNone(r["answer"])
        for rr in recs.values():                       # no literal wrap, one page-line form, no injector wrapper
            for st in rr["steps"]:
                self.assertFalse(st["text"].startswith("['"), st["text"])
                self.assertNotIn("at DuckDuckGo](", st["text"])
                self.assertFalse(st["text"].startswith("Agent:"))
                if st["agent"] == "WebSurfer":
                    self.assertIn(PAGE + " (https://duckduckgo.com/?q=", st["text"])
        self.assertEqual(recs["gpt-5-nano_arc_task3_2"]["dec"], 2)                  # history given as a literal
        self.assertEqual(recs["gpt-5-nano_hotpot_task6_1"]["flags"], [])            # "[WebSurfer]" = WebSurfer
        self.assertEqual(recs["gpt-5-nano_hotpot_task7_1"]["flags"], ["agent_mismatch"])
        self.assertEqual(counts["correct/text/page_title_only"], 1)
        fl = {x["uid"]: x["flags"] for x in fp.correct_records([ce_row(8, "arc", 1, 3, pair=True),
                                                                 ce_row(9, "arc", 1, 2, final=False)], counts)}
        self.assertEqual(fl, {"gpt-5-nano_arc_task8_1": ["same_speaker_next"],
                              "gpt-5-nano_arc_task9_1": ["no_final_answer"]})

    def test_aegis_records_and_items(self):
        counts = collections.Counter()
        lines = [json.dumps(aegis_line(0, "smoagents", [("Searcher", "FM-3.1")], "train")),
                 json.dumps(aegis_line(1, "dylan", [("Assistant2", "FM-1.2")], "train")),
                 json.dumps(aegis_line(2, "dylan", [("Assistant2", "FM-2.6")], "train")),
                 json.dumps(aegis_line(3, "macnet", [("Assistant3", "FM-3.2"), ("Assistant3", "FM-1.1")], "train")),
                 json.dumps(aegis_line(4, "dylan", [("Assistant1", "FM-3.2"), ("Assistant2", "FM-1.1")], "train"))]
        bad = aegis_line(5, "dylan", [("Assistant1", "FM-2.3")], "train")
        bad["input"]["conversation_history"][2]["content"] = "Path: /Users/x/malicious_agent/MASLab"
        lines.append(json.dumps(bad))
        recs = list(fp.aegis_records(lines, "train", counts))
        self.assertEqual(counts["aegis/drop/several_faulty_agents_train"], 1)
        self.assertEqual(counts["aegis/drop/meta_leak_train"], 1)
        self.assertEqual(counts["aegis/hidden_steps"], 1)
        sm, d12, d26, mac = recs
        self.assertEqual([s["agent"] for s in sm["steps"]], ["User", "Manager", "Searcher", "Manager"])
        self.assertTrue(sm["steps"][0]["text"].startswith("User message 1"))       # literal dicts parsed
        self.assertEqual((sm["dec"], sm["gold_agent"], sm["mode"], sm["category"], sm["mast"]),
                         (None, "Searcher", "A.3", "A", "FM-3.1"))
        self.assertEqual((d12["mode"], d12["category"]), (None, None))              # FM-1.2: no counterpart
        self.assertEqual((d26["mode"], d26["category"]), (None, "R"))               # category only
        self.assertEqual((mac["mast"], mac["mode"], mac["mast_all"]), (None, None, ["FM-1.1", "FM-3.2"]))
        self.assertFalse(any("is_injected" in s["text"] for s in mac["steps"]))
        kinds = {}
        for r in recs:
            for seed in range(20):
                it = fp.make_agent_item(r, f"x-{r['uid']}", "fa1/aegis", cap=3000, rng=random.Random(seed))
                self.assertEqual(fp.audit_agent_item(it), [], it["item_id"])
                self.assertEqual(it["label_names"][it["gold_label"]], r["gold_agent"])
                self.assertNotIn("User", it["label_names"].values())
                self.assertEqual(treeplan.statement_block(it) + "\n\n" + fp.Q_AGENT, treeplan.problem_no_options(it))
                kinds[r["uid"]] = sorted({n["type"] for n in it["subqs"]})
        self.assertEqual(kinds[sm["uid"]], ["agent", "category", "mode"])
        self.assertEqual(kinds[d12["uid"]], ["agent"])
        self.assertEqual(kinds[d26["uid"]], ["agent", "category"])
        it = fp.make_agent_item(sm, "y", "fa1/aegis", cap=3000, rng=random.Random(0))
        bad = json.loads(json.dumps(it))
        bad["subqs"][0]["truth"] = not bad["subqs"][0]["truth"]
        self.assertTrue(fp.audit_agent_item(bad))
        for code, (m, c, _, _) in fp.MAST_MAP.items():                              # the map stays in the taxonomy
            self.assertTrue(m is None or (m in fp.MODES and fp.category(m) == c), code)
            self.assertTrue(c is None or c in fp.CATEGORIES, code)


class TestAddSources(unittest.TestCase):
    def test_off_byte_identical_and_on_build(self):
        src = subprocess.run(["git", "-C", str(ROOT), "show", f"{CLEAN_COMMIT}:scripts/build_fa_public.py"],
                             capture_output=True, text=True)
        if src.returncode != 0:
            self.skipTest(f"git commit {CLEAN_COMMIT} not readable")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
        common = ["--n-val", "12", "--tokenizer", "missing.json", "--caps", "{}", "--share", '{"wwpro": 0.5}',
                  "--debias-length", "wwpro,agentracer", "--clean", "v1"]
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".fa1_test_") as d:
            d = Path(d)
            write_raw(d / "raw", n=60)
            write_raw_new(d / "raw")
            base = d / "base"
            (base / "scripts").mkdir(parents=True)
            (base / "scripts/build_fa_public.py").write_text(src.stdout)
            (base / "src").symlink_to(ROOT / "src")

            def run(script: Path, out: Path, ev: Path, extra=(), ok=True):
                cmd = [sys.executable, "-P", str(script), "--raw", str(d / "raw"), "--out", str(out), "--eval-dir",
                       str(ev)] + list(extra)
                r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900, cwd=d)
                if not ok:
                    return r
                self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
                return json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else None
            norm = ["--origins", "wwpro,stepfinder,agentracer,traceelephant"]
            run(base / "scripts/build_fa_public.py", base / "out", base / "eval", norm + common)   # 2509b67
            s_off = run(ROOT / "scripts/build_fa_public.py", d / "out", d / "eval", norm + common)
            for sub in ("out", "eval"):
                ps = sorted((base / sub).glob("*.jsonl"))
                self.assertGreaterEqual(len(ps), 4)
                self.assertEqual([p.name for p in ps], sorted(p.name for p in (d / sub).glob("*.jsonl")))
                for p in ps:
                    self.assertEqual(p.read_bytes(), (d / sub / p.name).read_bytes(), f"{sub}/{p.name}")
            self.assertNotIn("new_sources", s_off)
            # on: normalize the new sources into their own dir, then build with them
            run(ROOT / "scripts/build_fa_public.py", d / "out_new", d / "eval",
                ["--stage", "normalize", "--origins", "correct,aegis", "--correct-file", "correct_error/test.jsonl",
                 "--norm", str(d / "norm_new")])
            nc = json.loads((d / "norm_new/counts.json").read_text())
            self.assertEqual(nc["correct/drop/user_step"], 1)
            self.assertEqual(nc["correct/drop/meta_leak"], 1)
            self.assertGreater(nc["aegis/drop/several_faulty_agents_train"], 0)
            self.assertEqual(nc["aegis/drop/meta_leak_test"], 1)
            on = ["--stage", "build", "--norm", str(d / "out/norm"), "--add-norm", str(d / "norm_new"),
                  "--add-sources", "correct,aegis", "--n-val", "12", "--tokenizer", "missing.json", "--caps", "{}",
                  "--share", '{"wwpro": 0.45, "correct": 0.45}', "--debias-length", "wwpro,agentracer,correct",
                  "--clean", "v1", "--aegis-val", "6"]
            r = run(ROOT / "scripts/build_fa_public.py", d / "out_x", d / "eval_x", [x for x in on if x != "v1" and
                                                                                    x != "--clean"], ok=False)
            self.assertNotEqual(r.returncode, 0)                                   # --add-sources needs --clean
            orig = {p.name: p.read_bytes() for p in (d / "eval").glob("*.jsonl")}
            s = run(ROOT / "scripts/build_fa_public.py", d / "out_d", d / "eval", on)
            self.assertEqual(s["audit"]["failures"], 0)
            self.assertEqual({p.name: p.read_bytes() for p in (d / "eval").glob("*.jsonl") if p.name in orig}, orig)
            for name in ("fa1_correct_test.jsonl", "fa1_correct_test_agent.jsonl", "fa1_aegis_test_agent.jsonl"):
                self.assertTrue((d / "eval" / name).exists(), name)
            for name in ("fa1_correct_test_gt.jsonl", "fa1_correct_test_c.jsonl", "fa1_aegis_test.jsonl",
                         "fa1_aegis_test_c.jsonl", "fa1_aegis_test_agent_c.jsonl"):
                self.assertFalse((d / "eval" / name).exists(), name)
            ce_test = read_jsonl(d / "eval/fa1_correct_test.jsonl")
            ag_test = read_jsonl(d / "eval/fa1_aegis_test_agent.jsonl")
            train = read_jsonl(d / "out_d/train.jsonl")
            train_ag = read_jsonl(d / "out_d/train_agent.jsonl")
            val_ag = read_jsonl(d / "out_d/val_agent.jsonl")
            self.assertTrue(any(it["fa"]["origin"] == "correct" for it in train))
            self.assertFalse(any(it["fa"]["origin"] == "aegis" for it in train))     # agent-level only
            aeg = [it for it in train_ag if it["fa"]["origin"] == "aegis"]
            self.assertTrue(aeg and all(it["subqs"] and it["has_tree"] for it in aeg))
            self.assertFalse(any(fp.clean_group(it["fa"]) in fp.AEGIS_DROP_GROUPS for it in aeg + val_ag
                                 if it["fa"]["origin"] == "aegis"))                # AEGIS_DROP_GROUPS: train / val
            self.assertTrue(any(it["fa"]["origin"] == "aegis" for it in val_ag))
            n_other = sum(it["fa"]["origin"] != "aegis" for it in train_ag)
            self.assertLessEqual(len(aeg), 0.35 / 0.65 * n_other + 1)               # --agent-share
            q = TestBuiltData.task
            test_tasks = {q(it["prompt"]) for it in ce_test + ag_test}
            self.assertFalse(test_tasks & {q(it["prompt"]) for it in train + train_ag})
            ce_q = {it["fa"]["uid"].rsplit("_", 1)[0] for it in ce_test}               # no question in both
            self.assertFalse(ce_q & {it["fa"]["uid"].rsplit("_", 1)[0] for it in train if it["fa"]["origin"] == "correct"})
            self.assertTrue(all(not it["with_gt"] for it in ce_test))
            self.assertEqual({it["fa"]["origin"] for it in ag_test}, {"aegis"})
            self.assertTrue(any(it["fa"]["subset"] in ("macnet", "dylan") for it in ag_test))  # ... kept in the test
            self.assertEqual(s["new_sources"]["dropped"], sorted(fp.AEGIS_DROP_GROUPS))
            for it in ce_test + read_jsonl(d / "eval/fa1_correct_test_agent.jsonl") + train + read_jsonl(
                    d / "out_d/val.jsonl"):
                self.assertFalse(set(it["fa"]["flags"]) & set(fp.CORRECT_DROP_FLAGS), it["item_id"])
            self.assertFalse(any("agent_mismatch" in it["fa"]["flags"]
                                 for it in read_jsonl(d / "eval/fa1_correct_test_agent.jsonl")))
            self.assertGreater(sum(v for k, v in s["build_counts"].items() if k.startswith("correct/drop/same_")), 0)
            for it in ce_test + [x for x in train if x["fa"]["origin"] == "correct"]:
                self.assertEqual(fp.audit_item(it), [])
                self.assertNotIn("['I typed", it["prompt"])
                self.assertNotIn("at DuckDuckGo](", it["prompt"])
            ns = s["new_sources"]
            self.assertEqual(set(ns["agent_train_by_origin"]), {it["fa"]["origin"] for it in train_ag})
            self.assertIn("aegis", ns)
            self.assertIn("test_correct", s["baselines"])
            s2 = run(ROOT / "scripts/build_fa_public.py", d / "out_d", d / "eval", on)          # deterministic
            self.assertEqual(s["md5"], s2["md5"])


if __name__ == "__main__":
    unittest.main()
