"""Stepwise internalisation (train_student.py --si-stages; arms SI / SIY / SIS; scripts/build_gsm8k_si.py):
- the GSM8K worked steps (item field worked_steps): the builder reproduces the fixture rows from the raw socratic texts,
  complete chains (the gold step's last operation included, e.g. gsm8k-train-660: 4 + 2, 4 × 2, 4 ÷ 2, 4 + 6, 10 + 8,
  18 + 2 = 20), null chains where a step's computation is not written;
- numgen chains from the item's program;
- stage removal counts (ceil(k n / S), stage S removes everything), removal smoothing (its own coin, rate p);
- the SIS wrong values: never the true value, the same number of lines, gold-valued lines never an option, linked
  operands written with the linked line's wrong value;
- si_group: the final view's text gets the stage's chain before the options, the node views stay GFPQ's (rng lockstep;
  stage S is GFPQ's group exactly), final_only drops them;
- the built files (skipped when absent): audit clean, md5s, rows byte-identical to data/student_gsm8k_tf_ops but for
  worked_steps;
- the training loop: GFPQ (full-tree loop), A2 (plain loop) and A10 (fact stages with an optimizer reset) give the
  losses of SI_BASE_REF update for update; SI / SIY / SIS run end to end on a tiny random model (stage counts, stage-end
  evaluations, one schedule, SIY in step with GFPQ) and the argument checks refuse what is not defined.
CPU only (CUDA_VISIBLE_DEVICES=""); data-backed tests read data/ of this checkout, or COTD_DATA_DIR when set."""
import collections
import dataclasses
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

DATA = Path(os.environ.get("COTD_DATA_DIR") or ROOT / "data")
FX = json.loads((ROOT / "tests/fixtures/gsm8k_si_items_v1.json").read_text())
GSM = {it["item_id"]: it for it in FX["gsm8k"]}
SI_DIR = DATA / "student_gsm8k_tf_si"
OPS_DIR = DATA / "student_gsm8k_tf_ops"
# The last commit before stepwise internalisation: the existing loops are compared with it.
SI_BASE_REF = os.environ.get("SI_BASE_REF", "28813ed")
PROMPT_660 = ("Question: Abe finds 4 ants on the sidewalk.  Beth sees 50% more ants than Abe.  CeCe watches twice as "
              "many ants as Abe.  Duke discovers half as many ants as Abe.  How many ants do the four children find "
              "together?")
OPTS_660 = "\nOptions:\nA) 48\nB) 20\nC) 53\nD) 49\nE) 4"
LINES_660 = ["4 + 2 = 6", "4 × 2 = 8", "4 ÷ 2 = 2", "4 + 6 = 10", "10 + 8 = 18", "18 + 2 = 20"]

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def asdicts(xs):
    return [dataclasses.asdict(x) for x in xs]


@unittest.skipUnless(HAVE_NUMPY, "build_gsm8k_socratic needs numpy")
class TestBuilder(unittest.TestCase):
    def test_reproduces_fixture_rows(self):
        """worked_steps from the raw socratic texts equals the fixture rows (5699: null, its gold step is "<<36=36>>");
        the audit of every fixture row is clean."""
        import build_gsm8k_si as SI
        S = SI.T.soc()
        for iid, it in GSM.items():
            raw = FX["raw"][iid]
            p = S.parse_problem(int(iid.rsplit("-", 1)[1]), raw["question"], raw["answer"])
            self.assertEqual(p["status"], "ok", iid)
            st = collections.Counter()
            self.assertEqual(SI.worked_steps(p, it, 0, st), it["worked_steps"], iid)
            bad, _ = SI.audit_item(it)
            self.assertEqual(bad, [], iid)
        self.assertIsNone(GSM["gsm8k-train-5699"]["worked_steps"])

    def test_complete_chains(self):
        """660: every operation of the four notes, the gold step's last one (no value node) included; 3365: a percent
        line; 3929: a fraction literal; links to earlier results (same note, certain edges)."""
        ws = GSM["gsm8k-train-660"]["worked_steps"]
        self.assertEqual([x["text"] for x in ws], LINES_660)
        self.assertEqual([x["node"] for x in ws], ["s1o1", "s2o1", "s3o1", "s4o1", "s4o2", None])
        self.assertEqual([(x["a_from"], x["b_from"]) for x in ws],
                         [(None, None), (None, None), (None, None), (None, 0), (3, 1), (4, 2)])
        self.assertEqual([x["text"] for x in GSM["gsm8k-train-3365"]["worked_steps"]],
                         ["3 × 20 = 60", "10% of 60 = 6", "60 + 6 = 66"])
        self.assertEqual([x["text"] for x in GSM["gsm8k-train-3929"]["worked_steps"]],
                         ["1/2 × 4 = 2", "3 × 2 = 6"])
        for it in FX["gsm8k"]:
            if it["worked_steps"]:
                gold = it["options"][it["labels"].index(it["gold_label"])]
                self.assertEqual(it["worked_steps"][-1]["value"], gold, it["item_id"])
                self.assertTrue({n["id"] for n in it["value_nodes"]} <= {x["id"] for x in it["worked_steps"]})

    def test_wrong_values(self):
        """SIS values: never the truth, one per line; gold-valued lines never an option; operands that are earlier
        results carry those lines' wrong values (wrong_text), wrong_result_text keeps the true operands; value-node
        lines take one of the node's own wrong options."""
        ws = GSM["gsm8k-train-660"]["worked_steps"]
        self.assertEqual([x["wrong_text"] for x in ws],
                         ["4 + 2 = 4", "4 × 2 = 18", "4 ÷ 2 = 6", "4 + 4 = 2", "2 + 18 = 2", "2 + 6 = 16"])
        self.assertEqual([x["wrong_result_text"] for x in ws][-2:], ["10 + 8 = 2", "18 + 2 = 16"])
        for it in FX["gsm8k"]:
            nodes = {n["id"]: n for n in it.get("value_nodes") or []}
            for x in it["worked_steps"] or []:
                self.assertNotEqual(x["wrong"], x["value"])
                if x["value"] == it["options"][it["labels"].index(it["gold_label"])]:
                    self.assertNotIn(x["wrong"], it["options"])
                if x["node"]:
                    r = nodes[x["node"]]["renderings"]["op"]
                    self.assertIn(x["wrong"], [o for k, o in enumerate(r["options"]) if k != r["answer"]])
                for k, side in (("a_from", "a"), ("b_from", "b")):
                    want = it["worked_steps"][x[k]]["wrong"] if x[k] is not None else x[side]
                    self.assertIn(f"{want}", x["wrong_text"].split(" = ")[0])
                self.assertEqual(x["wrong_result_text"].split(" = ")[0], x["text"].split(" = ")[0])


@unittest.skipUnless((SI_DIR / "summary.json").exists() and (OPS_DIR / "train.jsonl").exists(),
                     "no data/student_gsm8k_tf_si (python3 scripts/build_gsm8k_si.py)")
class TestBuiltData(unittest.TestCase):
    def test_audit_md5_and_rows(self):
        import build_gsm8k_si as SI
        s = json.loads((SI_DIR / "summary.json").read_text())
        self.assertEqual(s["audit"]["failures"], 0)
        self.assertNotIn("sis_operand_leaks", s["audit"])
        for name, want in s["md5"].items():
            self.assertEqual(SI.T.md5(DATA / name), want, name)
        self.assertEqual(s["md5"]["student_gsm8k_tf_si/train.jsonl"], FX["si_train_md5"])
        for split in ("train", "val"):
            with open(OPS_DIR / f"{split}.jsonl") as a, open(SI_DIR / f"{split}.jsonl") as b:
                la = [x for x in a.read().split("\n") if x]               # not splitlines: U+2028 in texts
                lb = [x for x in b.read().split("\n") if x]
            self.assertEqual(len(la), len(lb))
            for x, y in zip(la, lb):
                r = json.loads(y)
                self.assertEqual(list(r)[-1], "worked_steps")
                ws = r.pop("worked_steps")
                self.assertEqual(json.dumps(r, ensure_ascii=False), x)
                for ln in ws or []:
                    self.assertNotEqual(ln["wrong"], ln["value"])
        for fr in (10, 25, 50):
            got = [json.loads(x)["item_id"] for x in (DATA / f"student_gsm8k_tf_si_f{fr}/train.jsonl").read_text()
                   .split("\n") if x]
            want = [json.loads(x)["item_id"] for x in (DATA / f"student_gsm8k_tf_ops_f{fr}/train.jsonl").read_text()
                    .split("\n") if x]
            self.assertEqual(got, want, fr)


@unittest.skipUnless(HAVE_TORCH, "needs torch + transformers")
class TestStages(unittest.TestCase):
    def test_removal_counts(self):
        from cotdistill.student import si_removed
        for n in range(1, 15):
            for S in (1, 2, 3, 4, 6):
                ms = [si_removed(n, k, S, 1.0, 0.1) for k in range(S + 1)]
                self.assertEqual(ms, [math.ceil(k * n / S) for k in range(S + 1)], (n, S))
                self.assertEqual((ms[0], ms[-1]), (0, n))
                self.assertEqual([si_removed(n, k, S, 0.0, 0.1) for k in range(S + 1)],
                                 [min(n, m + 1) for m in ms])          # smoothing: one more, at most n
        self.assertEqual([si_removed(6, k, 4, 1.0, 0.1) for k in range(5)], [0, 2, 3, 5, 6])

    def test_smoothing_coin(self):
        """The coin is keyed by (seed, stage, pass, item) only, and hits at rate p."""
        from cotdistill.student import si_rng
        self.assertEqual(si_rng(12, 1, 1, "x").random(), si_rng(12, 1, 1, "x").random())
        coins = [si_rng(12, 1, 1, f"i{j}").random() for j in range(4000)]
        self.assertAlmostEqual(sum(c < 0.1 for c in coins) / len(coins), 0.1, delta=0.015)
        self.assertNotEqual(coins[:5], [si_rng(13, 1, 1, f"i{j}").random() for j in range(5)])
        self.assertNotEqual(coins[:5], [si_rng(12, 1, 2, f"i{j}").random() for j in range(5)])

    def test_660_prompts_by_stage(self):
        """The written chain of 660 at every stage of 4 (front removal, kept steps keep their numbers; one more step
        with a smoothing hit; back removal); stage 4 is the test format, byte for byte."""
        from cotdistill.student import FINAL_TEMPLATE, insert_chain, si_view
        ws = GSM["gsm8k-train-660"]["worked_steps"]
        it = GSM["gsm8k-train-660"]
        self.assertEqual(it["prompt"], PROMPT_660 + OPTS_660)
        shown = {0: range(1, 7), 1: range(3, 7), 2: range(4, 7), 3: range(6, 7), 4: range(0)}
        for k, keep in shown.items():
            block, m = si_view(ws, stage=k, n_stages=4, chain="true", coin=1.0, smooth=0.1)
            want = "".join(f"\nStep {i}: {LINES_660[i - 1]}" for i in keep)
            self.assertEqual(block, f"\n\nWorked steps:{want}" if want else "", k)
            self.assertEqual(m, 6 - len(keep))
            text = insert_chain(FINAL_TEMPLATE.format(problem=it["prompt"]), block)
            self.assertEqual(text, PROMPT_660 + (f"\n\nWorked steps:{want}" if want else "") + OPTS_660 + "\n\nAnswer:")
        block, m = si_view(ws, stage=1, n_stages=4, chain="true", coin=0.05, smooth=0.1)
        self.assertEqual((m, block.split("\n")[3]), (3, "Step 4: 4 + 6 = 10"))
        block, _ = si_view(ws, stage=1, n_stages=4, chain="true", coin=1.0, smooth=0.1, remove="back")
        self.assertEqual(block.split("\n")[3:], [f"Step {i}: {LINES_660[i - 1]}" for i in range(1, 5)])
        block, _ = si_view(ws, stage=0, n_stages=4, chain="wrong", coin=1.0, smooth=0.1)
        self.assertEqual(block.split("\n")[-1], "Step 6: 2 + 6 = 16")

    def test_numgen_chain(self):
        """numgen: one line per derived quantity in program order, checked by numgen.apply; the last line is the gold;
        wrong values never the truth, the answer's never an option."""
        from cotdistill.numgen import OPCLS
        from cotdistill.student import numgen_chain, worked_steps
        first = FX["numgen"][0]
        self.assertEqual([x["text"] for x in numgen_chain(first)], ["18 ÷ 2 = 9"])
        seen = set()
        for it in FX["numgen"]:
            ws = worked_steps(it)
            derived = [q for q in it["program"]["quantities"] if q["op"] and not q["stated"]]
            self.assertEqual([x["id"] for x in ws], [q["id"] for q in derived])
            self.assertEqual(int(ws[-1]["value"]), it["options"][it["gold_label"]])
            self.assertNotIn(int(ws[-1]["wrong"]), it["options"].values())
            for x in ws:
                self.assertNotEqual(x["wrong"], x["value"])
                seen.add(OPCLS[x["op"]])
        self.assertTrue({"DIV", "SUM", "PCT", "PCTOFF", "DOZEN"} <= seen, seen)

    def test_si_group_lockstep(self):
        """si_group on GFPQ's groups: the node views and every other field unchanged, the final view's text with the
        chain before the options; no draw from the training rng; stage S = GFPQ's group; final_only = the final view;
        items without a chain unchanged."""
        from cotdistill.student import build_full_tree, insert_chain, si_group, si_rng, si_view, worked_steps
        kw = dict(mode="plain", view="statements+question", cap=0, sources={"gsm8k_tf"})
        for k in range(5):
            r_ref, r_si, st = random.Random(12), random.Random(12), {}
            for it in FX["gsm8k"]:
                ref = build_full_tree(it, rng=r_ref, **kw)
                got = si_group(build_full_tree(it, rng=r_si, **kw), it, stage=k, n_stages=4, seed=12, epoch=k, stats=st)
                self.assertEqual(asdicts([e for e in got if e.kind != "final"]),
                                 asdicts([e for e in ref if e.kind != "final"]))
                (f_ref,), (f_got,) = [e for e in ref if e.kind == "final"], [e for e in got if e.kind == "final"]
                ws = worked_steps(it)
                block = si_view(ws, stage=k, n_stages=4, chain="true", coin=si_rng(12, k, k, it["item_id"]).random(),
                                smooth=0.1)[0] if ws else ""
                self.assertEqual(f_got, dataclasses.replace(f_ref, text=insert_chain(f_ref.text, block)))
                if k == 4 or not ws:
                    self.assertEqual(asdicts(got), asdicts(ref))
                fo = si_group(ref, it, stage=k, n_stages=4, seed=12, epoch=k, final_only=True)
                self.assertEqual([e.kind for e in fo], ["final"])
            self.assertEqual(r_ref.getstate(), r_si.getstate())
            self.assertEqual(st["si_items"], 5)
            self.assertEqual(st["si_items_without_chain"], 1)


@unittest.skipUnless(HAVE_TORCH, "needs torch + transformers")
class TestTrain(unittest.TestCase):
    @staticmethod
    def common(d):
        from test_tree_forcing import SCRIPT
        return [sys.executable, SCRIPT, "--model", d + "/m", "--train", d + "/t.jsonl", "--eval", "v=" + d + "/t.jsonl",
                "--items-per-update", "2", "--micro-bs", "4", "--grad-accum", "1", "--max-len", "640",
                "--eval-max-len", "640",
                "--precision", "fp32master", "--no-grad-ckpt", "--log-every", "1", "--lr", "1e-3", "--seed", "3",
                "--final", "teacher", "--subq-target", "cot"]

    GFPQ = ["--subq", "cot", "--tree-full", "plain", "--tree-view", "statements+question", "--tree-cap", "0",
            "--tree-sources", "gsm8k_tf"]

    def run_(self, cmd, out, *extra, script=None):
        from test_tree_forcing import ENV
        if script:
            cmd = [cmd[0], script] + cmd[2:]
        return subprocess.run(cmd + ["--out", out, *extra], capture_output=True, text=True, timeout=900, env=ENV)

    def test_existing_loops_match_ref(self):
        """GFPQ (full-tree loop), A2 (plain loop) and A10 (fact stages, per-stage schedules, --reset-optim):
        SI_BASE_REF's losses, learning rates, sequences and evaluation, update for update."""
        from test_tree_forcing import head_tree, tiny
        with tempfile.TemporaryDirectory() as d:
            if head_tree(d + "/h", SI_BASE_REF) is None:
                self.skipTest("not a git checkout")
            tiny(d)
            Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in FX["gsm8k"]))
            arms = {"GFPQ": self.GFPQ + ["--epochs", "2"], "A2": ["--subq", "none", "--epochs", "2"],
                    "A10": ["--subq", "none", "--fact-stages", "2", "--reset-optim", "--epochs", "1"]}
            for arm, extra in arms.items():
                res = {}
                for name, script in (("ref", d + "/h/scripts/train_student.py"), ("now", None)):
                    r = self.run_(self.common(d), f"{d}/{name}-{arm}", *extra, script=script)
                    self.assertEqual(r.returncode, 0, (arm, name, r.stderr[-2000:]))
                    m = json.loads(Path(f"{d}/{name}-{arm}/metrics.json").read_text())
                    res[name] = ([(h["step"], h["loss"], h["lr"], h.get("seqs")) for h in m["history"]], m["eval"]["v"])
                self.assertEqual(res["ref"], res["now"], arm)
                self.assertGreater(len(res["now"][0]), 2)

    def test_si_arms_end_to_end(self):
        """SI / SIY / SIS (and GFPQ for the lockstep) through main(): S + 1 stages of epochs / (S + 1), one counted
        pass per stage with the expected chain counts, mid evaluations at the end of stages 0..S-1, one cosine schedule,
        SIY with GFPQ's sequences per update; numgen SI; the argument checks."""
        from cotdistill.student import si_removed, si_rng, worked_steps
        from test_tree_forcing import tiny
        with tempfile.TemporaryDirectory() as d:
            tiny(d)
            items = FX["gsm8k"]
            Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in items))
            si = ["--si-stages", "4", "--epochs", "5"]
            arms = {"SI": si + ["--si-final-only"], "SIY": si, "SIS": si + ["--si-final-only", "--si-chain", "wrong"],
                    "GFPQ": ["--epochs", "5"]}
            ms = {}
            for arm, extra in arms.items():
                r = self.run_(self.common(d) + self.GFPQ, f"{d}/{arm}", *extra)
                self.assertEqual(r.returncode, 0, (arm, r.stderr[-2000:]))
                ms[arm] = json.loads(Path(f"{d}/{arm}/metrics.json").read_text())
            n_upd = math.ceil(len(items) / 2)
            for arm in ("SI", "SIY", "SIS"):
                m = ms[arm]
                self.assertEqual(len(m["history"]), 5 * n_upd, arm)
                self.assertEqual([p["stage"] for p in m["si"]], [0, 1, 2, 3, 4])
                for p in m["si"]:
                    coin = {it["item_id"]: si_rng(3, p["stage"], p["pass"], it["item_id"]).random() for it in items}
                    want = sum(len(worked_steps(it)) - si_removed(len(worked_steps(it)), p["stage"], 4,
                                                                  coin[it["item_id"]], 0.1)
                               for it in items if worked_steps(it))
                    self.assertEqual((p["si_items"], p["si_shown"]), (5, want), (arm, p))
                self.assertEqual([t["epoch"] for t in m["trajectory"]], [1.0, 2.0, 3.0, 4.0])
                self.assertEqual([t["step"] for t in m["trajectory"]], [n_upd * k for k in (1, 2, 3, 4)])
                lrs = [h["lr"] for h in m["history"]]
                self.assertEqual(lrs, sorted(lrs, reverse=True))           # one schedule (warmup 1 step), no restart
                self.assertTrue(all(math.isfinite(h["loss"]) for h in m["history"]))
            self.assertEqual([h["seqs"] for h in ms["SIY"]["history"]], [h["seqs"] for h in ms["GFPQ"]["history"]])
            self.assertEqual({h["seqs"] for h in ms["SI"]["history"]}, {2})             # the final views only
            self.assertNotEqual([h["loss"] for h in ms["SI"]["history"]], [h["loss"] for h in ms["SIS"]["history"]])
            # SI and SIS: the same coins and line counts at every stage (only the written values differ)
            self.assertEqual(ms["SI"]["si"], ms["SIS"]["si"])
            # numgen: chains from the programs
            Path(d + "/n.jsonl").write_text("\n".join(json.dumps(x) for x in FX["numgen"]))
            cmd = [x if x != d + "/t.jsonl" else d + "/n.jsonl" for x in self.common(d)]
            r = self.run_(cmd + [x if x != "gsm8k_tf" else "numgen" for x in self.GFPQ], f"{d}/ng",
                          *(si + ["--si-final-only"]))
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            m = json.loads(Path(f"{d}/ng/metrics.json").read_text())
            self.assertEqual(m["si"][0]["si_lines"], sum(len(worked_steps(it)) for it in FX["numgen"]))
            # refused: SI flags without --si-stages, SI outside the full-tree loop, with resume, data without chains
            for bad, extra in ((self.GFPQ, ["--si-chain", "wrong"]), (["--subq", "none"], si),
                               (self.GFPQ, si + ["--stop-at-step", "2"]), (self.GFPQ, si + ["--si-smooth", "2"])):
                r = self.run_(self.common(d) + bad, f"{d}/bad", *extra)
                self.assertNotEqual(r.returncode, 0, extra)
            Path(d + "/t2.jsonl").write_text("\n".join(json.dumps({k: v for k, v in x.items() if k != "worked_steps"})
                                                       for x in items))
            cmd = [x if x != d + "/t.jsonl" else d + "/t2.jsonl" for x in self.common(d)]
            r = self.run_(cmd + self.GFPQ, f"{d}/bad2", *si)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("has worked steps", r.stderr + r.stdout)


if __name__ == "__main__":
    unittest.main()
