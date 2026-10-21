"""GSM8K verification nodes (scripts/build_gsm8k_vt.py, item field verify_nodes) and their renderings in the value-node
path of student.build_full_tree (--node-format value --node-render vt / vtr --tree-view problem; arms GFPQVT, GFPQVTR):
- builder units: what an option value is (gold / step / given / none, the skips), the choice pool, the 4 choices (A
  and N always, the canonical one true, every other false, length-rank pick), the shuffled control;
- gsm8k-train-660 and the other fixture rows (tests/fixtures/gsm8k_vt_items_v1.json, with their raw GSM8K solutions):
  the stored nodes are what the builder makes from the raw solution, options and their correct descriptions;
- the built files (skipped when absent): audit clean, md5s, rows byte-identical to data/student_gsm8k_tf_si but for
  verify_nodes, balance;
- with torch: vt / vtr examples (problem view with options, labels A..D, one-hot targets, per-problem weights), the
  view guard, the old arms byte-identical on rows with verify_nodes (yes/no arms vs the pre-value-node fixture, value
  arms vs the ops rows), evaluate_subq on verification nodes, and the trainer end to end (GFPQVT, GFPQVTR, A2 with
  --eval-value-render vt) on a tiny random model.
CPU only. Data-backed tests read data/ of this checkout, or COTD_DATA_DIR when set."""
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
VT = DATA / "student_gsm8k_tf_vt"
SI = DATA / "student_gsm8k_tf_si"
VT_FX = json.loads((ROOT / "tests/fixtures/gsm8k_vt_items_v1.json").read_text())
OPS_FX = json.loads((ROOT / "tests/fixtures/gsm8k_ops_items_v1.json").read_text())["items"]
YESNO_FX = json.loads((ROOT / "tests/fixtures/gsm8k_tf_yesno_examples_v1.json").read_text())
SRC = {"gsm8k_tf"}

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


def vt_items():
    return [json.loads(json.dumps(it)) for it in VT_FX["items"]]


def asdicts(exs):
    return [dataclasses.asdict(e) for e in exs]


@unittest.skipUnless(HAVE_NUMPY, "build_gsm8k_socratic needs numpy")
class TestBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import build_gsm8k_vt as V
        cls.V = V
        cls.S = V.T.soc()

    def parsed(self, iid):
        r = VT_FX["raw"][iid]
        return self.S.parse_problem(r["row"], r["question"], r["answer"])

    def test_fixture_rows_rebuild(self):
        """The stored nodes of every fixture row are what verify_nodes makes from the raw solution (seed 0, one none
        node), and the audit is clean."""
        for it in vt_items():
            p = self.parsed(it["item_id"])
            st = collections.Counter()
            row = {k: v for k, v in it.items() if k != "verify_nodes"}
            self.assertEqual(self.V.verify_nodes(row, p, 0, 1, st), it["verify_nodes"], it["item_id"])
            self.assertEqual(self.V.audit_item(it, p, 0), [], it["item_id"])

    def test_660(self):
        """gsm8k-train-660 (4 ants; Beth 6, CeCe 8, Duke 2; gold 20; options 48 / 20 / 53 / 49 / 4): the gold node,
        the given number 4, one of the three none options; the pool is the three non-final sub-questions."""
        V = self.V
        it = {x["item_id"]: x for x in vt_items()}["gsm8k-train-660"]
        info = V.problem_info(self.parsed("gsm8k-train-660"))
        self.assertEqual([d for d, _, _ in info["pool"]], ["How many ants does Beth see?", "How many ants does CeCe see?",
                                                           "How many ants does Duke see?"])
        self.assertEqual(V.classify(20.0, info)[:2], ("gold", V.DESC_ANSWER))
        self.assertEqual(V.classify(4.0, info)[:2], ("given", V.DESC_GIVEN))
        for v in (48.0, 53.0, 49.0):
            self.assertEqual(V.classify(v, info)[:2], ("none", V.DESC_NONE))
        self.assertEqual(V.classify(6.0, info)[:2], ("step", "How many ants does Beth see?"))
        self.assertEqual(V.classify(18.0, info)[:2], (None, "unnamed_solution_quantity"))   # 4 + 6 + 8 = 18 inside the
        self.assertEqual(V.classify(50.0, info)[:2], (None, "given_unused"))                 # gold note; "50%"
        nodes = {n["option"]: n for n in it["verify_nodes"]}
        self.assertEqual(sorted(nodes), ["B", "C", "E"])
        want = {"B": ("20", "gold", V.DESC_ANSWER), "C": ("53", "none", V.DESC_NONE), "E": ("4", "given", V.DESC_GIVEN)}
        for L, (val, kind, desc) in want.items():
            n = nodes[L]
            r = n["renderings"]["vt"]
            self.assertEqual((n["value"], n["vt_kind"], r["options"][r["answer"]]), (val, kind, desc))
            self.assertEqual(r["question"], f"In this problem, what does option {L} ({val}) stand for?")
            self.assertEqual(len(r["options"]), 4)
            self.assertIn(V.DESC_ANSWER, r["options"])
            self.assertIn(V.DESC_NONE, r["options"])
            for o in r["options"]:
                self.assertIn(o, V.descriptions(info))
        self.assertEqual(nodes["B"]["option_kind"], "gold")

    def test_classify_rules(self):
        V = self.V
        info = {"pool": [("How many apples are left?", 33.0, 1), ("How many does she buy?", 20.0, 2),
                         ("How many pears are left?", 33.0, 4)],
                "steps": [(11.0, 0), (33.0, 1), (20.0, 2), (10.0, 3), (7.0, 4), (43.0, 5)], "gold": 43.0,
                "given": {44.0, 25.0, 3.0, 6.0, 20.0}, "quantities": [11.0, 33.0, 20.0, 10.0, 43.0, 18.0, 44.0, 3.0, 6.0]}
        self.assertEqual(V.classify(43.0, info)[0], "gold")
        k, d, true, j = V.classify(20.0, info)                  # a step value that is also written: the step wins,
        self.assertEqual((k, d, j), ("step", "How many does she buy?", 2))   # G is true too (never offered)
        self.assertEqual(true, {"How many does she buy?", V.DESC_GIVEN})
        self.assertEqual(V.classify(33.0, info)[:2], (None, "ambiguous_steps"))
        self.assertEqual(V.classify(11.0, info)[:2], (None, "unnamed_step_value"))   # a step outside the pool
        self.assertEqual(V.classify(18.0, info)[:2], (None, "unnamed_solution_quantity"))
        self.assertEqual(V.classify(25.0, info)[:2], (None, "given_unused"))
        self.assertEqual(V.classify(44.0, info)[:2], ("given", V.DESC_GIVEN))
        self.assertEqual(V.classify(45.0, info)[:2], ("none", V.DESC_NONE))

    def test_pick_choices(self):
        V = self.V
        false = [V.DESC_ANSWER, V.DESC_GIVEN, "Short?", "A much longer sub-question about the problem here?",
                 "How many ants does Beth see?"]
        ranks = collections.Counter()
        for k in range(400):
            d = V.pick_choices(V.DESC_NONE, false, None, random.Random(k))
            self.assertEqual(len(d), 3)
            self.assertEqual(len(set(d)), 3)
            self.assertIn(V.DESC_ANSWER, d)
            self.assertNotIn(V.DESC_NONE, d)
            self.assertTrue(set(d) <= set(false))
            ranks[V.length_rank(V.DESC_NONE, [V.DESC_NONE] + d)] += 1
            f = V.pick_choices(V.DESC_NONE, false, "Short?", random.Random(k))
            self.assertIn("Short?", f)
        self.assertGreater(min(ranks.values()), 60, ranks)      # every rank realisable here: drawn about uniformly

    def test_control(self):
        """vtr: vt's question and options, the answer the description of the node the permutation maps to; per
        problem the control answers are a permutation of the canonical descriptions (no node kept a true
        non-canonical one in the fixture)."""
        agree = n = 0
        for it in vt_items():
            canon, ctrl = [], []
            for nd in it["verify_nodes"]:
                r, rc = nd["renderings"]["vt"], nd["renderings"]["vtr"]
                self.assertEqual((rc["question"], rc["options"], rc["kinds"]), (r["question"], r["options"], r["kinds"]))
                canon.append(r["options"][r["answer"]])
                ctrl.append(rc["options"][rc["answer"]])
                agree += nd["control_agree"]
                n += 1
                self.assertEqual(nd["control_agree"], r["answer"] == rc["answer"])
            self.assertEqual(sorted(canon), sorted(ctrl), it["item_id"])
        self.assertLess(agree, n)
        bad = vt_items()[0]
        nd = bad["verify_nodes"][0]
        nd["renderings"]["vtr"]["answer"] = (nd["renderings"]["vtr"]["answer"] + 1) % 4
        nd["control_agree"] = nd["renderings"]["vtr"]["answer"] == nd["renderings"]["vt"]["answer"]
        self.assertTrue(self.V.audit_item(bad, self.parsed(bad["item_id"]), 0))

    def test_audit_catches(self):
        it = vt_items()[-1]
        p = self.parsed(it["item_id"])
        nd = it["verify_nodes"][0]
        r = nd["renderings"]["vt"]
        r["answer"] = (r["answer"] + 1) % 4
        self.assertTrue(self.V.audit_item(it, p, 0))
        it = vt_items()[-1]
        it["verify_nodes"][0]["value"] = "999"
        self.assertTrue(self.V.audit_item(it, p, 0))

    def test_none_nodes_all(self):
        """--none-nodes all asks every option the builder can describe; the default one none option per problem."""
        it = {x["item_id"]: x for x in vt_items()}["gsm8k-train-660"]
        row = {k: v for k, v in it.items() if k != "verify_nodes"}
        p = self.parsed("gsm8k-train-660")
        allv = self.V.verify_nodes(row, p, 0, None, collections.Counter())
        self.assertEqual([n["option"] for n in allv], ["A", "B", "C", "D", "E"])
        self.assertEqual(sum(n["vt_kind"] == "none" for n in allv), 3)


@unittest.skipUnless((VT / "summary.json").exists() and HAVE_NUMPY, "data/student_gsm8k_tf_vt not built")
class TestBuiltData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import build_gsm8k_vt as V
        cls.V = V
        cls.s = json.loads((VT / "summary.json").read_text())

    def test_audit_and_md5(self):
        self.assertEqual(self.s["audit"]["failures"], 0)
        for name, want in self.s["md5"].items():
            self.assertEqual(self.V.T.md5(DATA / name), want, name)
        for name, want in self.s["source_md5"].items():
            self.assertEqual(self.V.T.md5(ROOT / name), want, name)

    def test_rows_identical_but_verify_nodes(self):
        for split in ("train", "val"):
            with open(SI / f"{split}.jsonl") as a, open(VT / f"{split}.jsonl") as b:
                la = [x for x in a.read().split("\n") if x]               # not splitlines: U+2028 in texts
                lb = [x for x in b.read().split("\n") if x]
            self.assertEqual(len(la), len(lb))
            for x, y in zip(la, lb):
                r = json.loads(y)
                self.assertEqual(list(r)[-1], "verify_nodes")
                del r["verify_nodes"]
                self.assertEqual(json.dumps(r, ensure_ascii=False), x)

    def test_balance(self):
        f = self.s["files"]["student_gsm8k_tf_vt/train.jsonl"]
        self.assertGreater(f["nodes"], 12000)
        tot = sum(f["answer_position"].values())
        for L in "ABCD":
            self.assertLess(abs(f["answer_position"][L] / tot - 0.25), 0.02, f["answer_position"])
        for k in ("answer", "none"):                       # offered in every node
            self.assertEqual(f["type_prior"]["offered"][k], f["nodes"])
        self.assertLess(f["type_prior"]["best_prior_pick_acc"], 0.40)
        self.assertLess(max(f["shortest_pick_acc"], f["longest_pick_acc"]), 0.30)
        self.assertLess(abs(f["control_agree_share"] - 0.37), 0.05)
        self.assertLess(self.s["description_only_adversary"]["val_acc"], 0.42)
        T = self.V.T
        for it in T.read_jsonl(VT / "val.jsonl"):
            gold = T.gold_text(it)
            shown = set(T.NUMTOK.findall(it["prompt"].partition("\nOptions:")[0]))
            for n in it["verify_nodes"]:
                self.assertEqual(n["vt_kind"] == "gold", n["value"] == gold)
                for o in n["renderings"]["vt"]["options"]:      # no description shows a gold number the problem
                    self.assertTrue(gold not in T.NUMTOK.findall(o) or gold in shown, o)   # does not show


@unittest.skipUnless(HAVE_TORCH, "needs torch + transformers")
class TestVerificationViews(unittest.TestCase):
    YESNO_ARMS = {"GFPQ": dict(mode="plain", view="statements+question"), "GFPS": dict(mode="plain", view="statements"),
                  "GFP": dict(mode="plain", view="problem"),
                  "GFPQR": dict(mode="plain", view="statements+question", target_seed=12)}

    def build(self, items, **kw):
        from cotdistill.student import build_full_tree
        from cotdistill.treeplan import statement_block
        rng, out = random.Random(12), []
        for it in items:
            if kw.get("view") == "statements":
                try:
                    statement_block(it)
                except ValueError:
                    continue
            out.append(build_full_tree(it, rng=rng, cap=0, sources=SRC, **kw))
        return out

    def test_examples(self):
        from cotdistill.student import build_full_tree
        for it in vt_items():
            for render in ("vt", "vtr"):
                st = {}
                ex = build_full_tree(it, mode="plain", view="problem", rng=random.Random(0), cap=0, sources=SRC,
                                     node_format="value", node_render=render, stats=st)
                nodes = it["verify_nodes"]
                self.assertEqual(len(ex), 1 + len(nodes))
                self.assertEqual((ex[0].kind, ex[0].text), ("final", it["prompt"] + "\n\nAnswer:"))
                for e, n in zip(ex[1:], nodes):
                    r = n["renderings"][render]
                    want = (f"{it['prompt']}\n\nIntermediate question: {r['question']}\nOptions:\n"
                            + "\n".join(f"{L}) {o}" for L, o in zip("ABCD", r["options"])) + "\nAnswer:")
                    self.assertEqual(e.text, want)
                    self.assertEqual(e.labels, ["A", "B", "C", "D"])
                    self.assertEqual(e.target, [float(i == r["answer"]) for i in range(4)])
                    self.assertAlmostEqual(e.weight, 1.0 / len(nodes))
                    self.assertEqual(e.kind, "subq")
                self.assertEqual(st["value_views"], len(nodes))
                if render == "vtr":
                    self.assertEqual(st["vtr_agree"], sum(n["control_agree"] for n in nodes))
        it = {x["item_id"]: x for x in vt_items()}["gsm8k-train-660"]
        e = build_full_tree(it, mode="plain", view="problem", rng=random.Random(0), cap=0, sources=SRC,
                            node_format="value", node_render="vt")[1]
        self.assertTrue(e.text.endswith("Intermediate question: In this problem, what does option B (20) stand for?\n"
                                        "Options:\nA) None of the quantities in the solution\nB) A number written in "
                                        "the problem text\nC) How many ants does Duke see?\nD) The answer to the "
                                        "problem's final question\nAnswer:"), e.text)
        self.assertEqual(e.target, [0.0, 0.0, 0.0, 1.0])
        v = build_full_tree(it, mode="plain", view="problem", rng=random.Random(0), cap=0, sources=SRC,
                            node_format="value", node_render="vt")
        c = build_full_tree(it, mode="plain", view="problem", rng=random.Random(0), cap=0, sources=SRC,
                            node_format="value", node_render="vtr")
        self.assertEqual([x.text for x in v], [x.text for x in c])          # the control changes targets only
        self.assertNotEqual([x.target for x in v], [x.target for x in c])

    def test_guards(self):
        from cotdistill.student import build_full_tree
        it = vt_items()[0]
        for view in ("statements+question", "statements", "bare"):
            with self.assertRaises(ValueError):
                build_full_tree(it, mode="plain", view=view, rng=random.Random(0), cap=0, sources=SRC,
                                node_format="value", node_render="vt")
        no = build_full_tree({k: v for k, v in it.items() if k != "verify_nodes"}, mode="plain", view="problem",
                             rng=random.Random(0), cap=0, sources=SRC, node_format="value", node_render="vt")
        self.assertEqual(len(no), 1)

    def test_old_arms_byte_identical(self):
        """On rows with verify_nodes (and worked_steps): the yes/no arms equal the examples of the code before value
        nodes (commit in the fixture), the value arms (op / qop / opsel, GFPQVM) those of the ops rows."""
        for arm, kw in self.YESNO_ARMS.items():
            self.assertEqual([asdicts(x) for x in self.build(vt_items(), **kw)], YESNO_FX["examples"][arm], arm)
        ops = [json.loads(json.dumps(x)) for x in OPS_FX]
        self.assertEqual([x["item_id"] for x in ops], [x["item_id"] for x in vt_items()])
        for fmt, render in (("value", "op"), ("value", "qop"), ("value", "opsel"), ("yesno+value", "op")):
            for view in ("statements+question", "problem"):
                kw = dict(mode="plain", view=view, node_format=fmt, node_render=render)
                self.assertEqual([asdicts(x) for x in self.build(vt_items(), **kw)],
                                 [asdicts(x) for x in self.build(ops, **kw)], (fmt, render, view))

    def test_si_and_slot_views_unchanged(self):
        """The SI chain and the latent-slot targets read worked_steps only: the same on rows with and without
        verify_nodes."""
        from cotdistill.student import build_full_tree, si_group, slot_value_targets
        for it in vt_items():
            base = {k: v for k, v in it.items() if k != "verify_nodes"}
            kw = dict(mode="plain", view="statements+question", rng=random.Random(3), cap=0, sources=SRC)
            a = si_group(build_full_tree(it, **kw), it, stage=1, n_stages=4, seed=12, epoch=1, stats={})
            kw["rng"] = random.Random(3)
            b = si_group(build_full_tree(base, **kw), base, stage=1, n_stages=4, seed=12, epoch=1, stats={})
            self.assertEqual(asdicts(a), asdicts(b))
            self.assertEqual(slot_value_targets(it, 6, fill="last"), slot_value_targets(base, 6, fill="last"))

    def test_eval_subq(self):
        """evaluate_subq: vt scores the verification nodes (groups all_value, value/mc, value/type/value_vt,
        value/vt_kind/<kind>, no depth group), op the value nodes only; the yes/no predictions are unchanged."""
        import test_student as ts
        from cotdistill.student import evaluate_subq
        tok, model = ts.tiny()
        model.eval()
        its = vt_items()[-2:]
        base = [{k: v for k, v in it.items() if k not in ("verify_nodes", "value_nodes")} for it in its]
        m0, p0 = evaluate_subq(model, tok, base, 512, 4, {}, seed=5)
        m, p = evaluate_subq(model, tok, its, 512, 4, {}, seed=5, value_render="vt")
        n = sum(len(it["verify_nodes"]) for it in its)
        val = [x for x in p if x["kind"] == "value"]
        self.assertEqual((len(val), m["all_value"]["n"], m["value/mc"]["n"], m["value/type/value_vt"]["n"]),
                         (n, n, n, n))
        kinds = collections.Counter(nd["vt_kind"] for it in its for nd in it["verify_nodes"])
        for k, c in kinds.items():
            self.assertEqual(m[f"value/vt_kind/{k}"]["n"], c)
        self.assertFalse(any(g.startswith("value/depth/") for g in m))
        self.assertEqual(m["all_cot"], m0["all_cot"])
        for x in val:
            self.assertEqual((len(x["probs"]), x["render"]), (4, "vt"))
        m2, _ = evaluate_subq(model, tok, its, 512, 4, {}, seed=5, value_render="op")
        self.assertEqual(m2["all_value"]["n"], sum("op" in sq["renderings"] for it in its for sq in it["value_nodes"]))
        self.assertNotIn("value/type/value_vt", m2)


@unittest.skipUnless(HAVE_TORCH, "needs torch + transformers")
class TestTrainVerification(unittest.TestCase):
    def test_arms_end_to_end(self):
        """GFPQVT, GFPQVTR (the sbatch flags) and A2 with --eval-value-render vt through main() on four fixture rows:
        they run, count their views, the losses are finite, --eval-subq scores the verification nodes in vt (for the
        control too); the guards refuse a view without options and data without verify_nodes."""
        from test_tree_forcing import ENV, SCRIPT, tiny
        tree = ["--tree-full", "plain", "--tree-cap", "0", "--tree-sources", "gsm8k_tf", "--node-format", "value",
                "--tree-view", "problem"]
        arms = {"GFPQVT": tree + ["--node-render", "vt"], "GFPQVTR": tree + ["--node-render", "vtr"],
                "A2": ["--eval-value-render", "vt"]}
        with tempfile.TemporaryDirectory() as d:
            tiny(d)
            items = vt_items()[-4:]
            Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in items))
            common = [sys.executable, SCRIPT, "--model", d + "/m", "--train", d + "/t.jsonl", "--eval",
                      "v=" + d + "/t.jsonl", "--eval-subq", d + "/t.jsonl", "--epochs", "1", "--items-per-update",
                      "2", "--micro-bs", "4", "--max-len", "512", "--eval-max-len", "512", "--precision",
                      "fp32master", "--no-grad-ckpt", "--log-every", "1", "--lr", "1e-3", "--subq-target", "cot"]
            n = sum(len(it["verify_nodes"]) for it in items)
            for arm, extra in arms.items():
                sub = ["--subq", "cot" if arm != "A2" else "none"]
                r = subprocess.run(common + sub + ["--out", f"{d}/{arm}", *extra], capture_output=True, text=True,
                                   timeout=900, env=ENV)
                self.assertEqual(r.returncode, 0, (arm, r.stderr[-2000:]))
                m = json.loads(Path(f"{d}/{arm}/metrics.json").read_text())
                self.assertTrue(all(math.isfinite(h["loss"]) for h in m["history"]))
                self.assertEqual(m["eval"]["subq"]["all_value"]["n"], n, arm)
                self.assertEqual(m["eval"]["subq"]["value/type/value_vt"]["n"], n, arm)
                if arm != "A2":
                    self.assertEqual(m["tree_view_stats"]["value_views"], n, arm)
                    self.assertEqual(m["tree_view_stats"].get("views", 0), 0, arm)
                if arm == "GFPQVTR":
                    self.assertEqual(m["tree_view_stats"]["vtr_agree"],
                                     sum(nd["control_agree"] for it in items for nd in it["verify_nodes"]))
                preds = [json.loads(x) for x in Path(f"{d}/{arm}/preds_subq.jsonl").read_text().split("\n") if x]
                self.assertTrue(all(x["render"] == "vt" for x in preds if x["kind"] == "value"))
            bad = common + ["--subq", "cot", "--out", f"{d}/bad"] + tree[:-1] + ["statements+question", "--node-render",
                                                                                 "vt"]
            r = subprocess.run(bad, capture_output=True, text=True, timeout=300, env=ENV)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("--tree-view", r.stderr + r.stdout)
            Path(d + "/t2.jsonl").write_text("\n".join(json.dumps({k: v for k, v in x.items() if k != "verify_nodes"})
                                                       for x in items))
            r = subprocess.run([x if x != d + "/t.jsonl" else d + "/t2.jsonl" for x in common]
                               + ["--subq", "cot", "--out", f"{d}/bad2", *arms["GFPQVT"]],
                               capture_output=True, text=True, timeout=600, env=ENV)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("verify_nodes", r.stderr + r.stdout)


if __name__ == "__main__":
    unittest.main()
