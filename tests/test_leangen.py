"""Lean F1 pilot (src/cotdistill/leangen.py, scripts/build_lean_tf.py): generator invariants, the one-period brute
force, closure depths, certificate parsing on a FAKE Lean output (no Lean needed), determinism, the lean_f1
statement_block branch, and every other source's statement_block / problem_no_options unchanged against a frozen copy of
treeplan.py at 9f03003 (tests/fixtures/treeplan_9f03003.py: the diff only inserts lines, and the outputs agree on the
fixtures and on the local data of every source). CPU only; numpy for the brute force; the GFP-S view check needs torch.

Data-backed tests read data/ of this checkout, or COTD_DATA_DIR when set, and skip when the files are missing."""
import collections
import difflib
import importlib.util
import json
import os
import random
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from cotdistill import leangen as G  # noqa: E402
from cotdistill import treeplan  # noqa: E402
from cotdistill.treeplan import (FINAL_QUESTION, LEAN_F1_QUESTION, node_depths, problem_no_options,  # noqa: E402
                                 statement_block)

DATA = Path(os.environ.get("COTD_DATA_DIR") or ROOT / "data")
FROZEN = ROOT / "tests/fixtures/treeplan_9f03003.py"

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def frozen_treeplan():
    spec = importlib.util.spec_from_file_location("treeplan_9f03003", FROZEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def items(n_per_depth=3, depths=range(1, 9), tag="t"):
    out = []
    for d in depths:
        for j in range(n_per_depth):
            plan = {"drift": j % 2 == 0, "n_drift": 1 + (j % 3 == 0), "n_on": 1 + (j % 2), "non_z": j % 3 == 1,
                    "n_irrelevant": 1 + j % 3}
            it, _ = G.build_item(d, (tag, d, j), plan)
            assert it is not None
            out.append(it)
    return out


ITEMS = None


def get_items():
    global ITEMS
    if ITEMS is None:
        ITEMS = items()
    return ITEMS


def outcome(fn, item):
    try:
        return ("ok", fn(item))
    except ValueError:
        return ("raise", None)


class TestSemantics(unittest.TestCase):
    def test_brute_force(self):
        h = [G.cong("x", 4, 3), G.defn("y", [[2, "x"]], 1), G.defn("z", [[1, "y"]], 5)]
        st = lambda v, m, r: G.bf_status(h, {"kind": "cong", "var": v, "m": m, "r": r})   # noqa: E731
        self.assertEqual(st("y", 8, 7), "entailed")
        self.assertEqual(st("z", 4, 0), "entailed")
        self.assertEqual(st("z", 8, 4), "entailed")
        self.assertEqual(st("z", 4, 2), "refuted")
        self.assertEqual(st("z", 16, 12), "independent")
        self.assertEqual(st("z", 3, 0), "independent")
        self.assertEqual(st("z", 6, 1), "refuted")                    # z even (z = 0 mod 4), 1 mod 6 is odd
        eq = {"kind": "def", "var": "y", "terms": [[2, "x"]], "c": 4}
        self.assertEqual(G.bf_status(h, eq), "refuted")
        self.assertEqual(G.bf_status(h, dict(eq, c=1)), "entailed")
        self.assertEqual(G.bf_status(h + [G.cong("y", 2, 0)], {"kind": "cong", "var": "z", "m": 3, "r": 0}), "unsat")
        # negative values: Python % = Lean's Int % (Euclidean) for m > 0
        self.assertEqual(G.bf_status([G.cong("x", 4, 3), G.defn("y", [[1, "x"]], -11)],
                                     {"kind": "cong", "var": "y", "m": 4, "r": 0}), "entailed")

    def test_closure_depths(self):
        h = [dict(G.cong("x", 4, 3), id="h1"), dict(G.defn("y", [[2, "x"]], 1), id="h2"),
             dict(G.defn("z", [[1, "y"]], 5), id="h3")]
        k = G.closure(h)
        self.assertEqual((k[("y", 8)]["r"], k[("y", 8)]["depth"], k[("y", 8)]["rule"]), (7, 1, "lift"))
        self.assertEqual((k[("z", 4)]["r"], k[("z", 4)]["depth"]), (0, 2))      # subst twice, not lift + reduce (3)
        self.assertEqual((k[("z", 8)]["r"], k[("z", 8)]["depth"]), (4, 2))
        self.assertEqual(G.closure_label(k, "z", 6, 1), "refuted")
        self.assertIsNone(G.closure_label(k, "z", 3, 0))
        order, prem = G.proof_of(k, ("z", 8))
        self.assertEqual(prem, ["h1", "h2", "h3"])
        self.assertEqual(order, [("y", 8), ("z", 8)])
        with self.assertRaises(ValueError):
            G.closure(h + [dict(G.cong("y", 2, 0), id="h4")])

    def test_render_parse_round_trip(self):
        for h in (G.cong("x", 12, 7), G.defn("y", [[2, "x"], [-3, "w"]], -1), G.defn("y", [[1, "x"]], 9)):
            for t in range(len(G.CONG_TPL) if h["kind"] == "cong" else len(G.DEF_TPL)):
                self.assertEqual(G.strip_meta(G.parse_sentence(G.sentence(dict(h, tpl=t)))), G.strip_meta(h))
            self.assertEqual(G.strip_meta(G.parse_question(G.question_of(h))[1]), G.strip_meta(h))
        self.assertEqual(G.expr_lean([[2, "x"], [-3, "w"]], -1), "2 * x - 3 * w - 1")
        self.assertEqual(G.expr_text([[1, "x"]], 5), "x plus 5")
        self.assertEqual(G.parse_option("z leaves remainder 4 when divided by 8"), G.cong("z", 8, 4))
        self.assertTrue(re.fullmatch(G.FINAL_RE, G.QUESTION.format(v="z", m=12)))
        self.assertEqual(G.FINAL_RE, LEAN_F1_QUESTION)


class TestGenerator(unittest.TestCase):
    def test_invariants(self):
        for it in get_items():
            d, key = it["depth"], it["program"]["key"]
            hyps = [G.strip_meta(h) for h in it["program"]["hyps"]]
            ids = [h["id"] for h in it["program"]["hyps"]]
            known = G.closure(it["program"]["hyps"])
            k = known[(key["var"], key["m"])]
            self.assertEqual((k["depth"], k["r"]), (d, key["r"]))
            self.assertNotIn(k["rule"], ("reduce", "premise"))
            self.assertGreaterEqual(len(it["program"]["critical"]), d + 1)
            self.assertTrue(set(it["program"]["critical"]) <= set(ids))
            self.assertEqual(sum(h["role"] == "distractor" for h in it["program"]["hyps"]), it["n_irrelevant"])
            # options
            kinds = collections.Counter(it["option_kinds"].values())
            self.assertEqual(kinds["key"], 1)
            self.assertEqual(it["option_kinds"][it["gold_label"]], "key")
            self.assertGreaterEqual(kinds["refuted"], 1)
            self.assertGreaterEqual(kinds["independent"], 1)
            self.assertEqual(kinds["drift_true"] > 0, it["drift"])
            self.assertEqual(len(set(it["options"].values())), 5)
            z, M = key["var"], key["m"]
            nz = 0
            for L, m in it["option_meta"].items():
                kind = it["option_kinds"][L]
                on = m["var"] == z and m["m"] == M
                self.assertEqual(m["target"], "on" if on else "off")
                self.assertIn(kind if on else "off", ("key", "refuted", "off"))
                if m["var"] == z and not on:
                    self.assertNotEqual(m["m"] % M, 0, "an off-target divisor of z is never a multiple of M")
                nz += m["var"] != z
                if kind == "drift_true":
                    self.assertEqual(m["tag"], "off_target")
                    self.assertIn(m["subtype"], ("a", "b", "c"))
                    self.assertEqual(m["subtype"] == "c", m["var"] == z and m["subtype"] != "b")
                st = G.bf_status(hyps, {"kind": "cong", **{x: m[x] for x in ("var", "m", "r")}})
                self.assertEqual(st, {"key": "entailed", "drift_true": "entailed", "refuted": "refuted",
                                      "independent": "independent"}[kind], (it["prompt"], L))
            self.assertLessEqual(nz, 1, "at most one option about another variable")
            self.assertFalse(G.form_odd_one_out([it["option_meta"][L] for L in G.LETTERS]))
            # tree
            nodes = it["subqs"]
            by = {n["id"]: n for n in nodes}
            depth = node_depths(nodes)
            for n in nodes:
                self.assertEqual(depth[n["id"]], n["node_depth"])
                twin = by[n["id"][:-1]] if n["id"].endswith("x") else by[n["id"] + "x"]
                self.assertNotEqual(n["truth"], twin["truth"])
                self.assertEqual((n["depends_on"], n["type"]), (twin["depends_on"], twin["type"]))
                self.assertEqual(n["opposite"], twin["question"])
                self.assertTrue(all(by[p]["truth"] for p in n["depends_on"]))
                if n["type"] == "derive":
                    self.assertEqual(n["node_depth"], known[(n["claim"]["var"], n["claim"]["m"])]["depth"])
                    par = [by[p]["claim"] for p in n["depends_on"]]
                    self.assertEqual(G.bf_status(par, n["claim"]), "entailed" if n["truth"] else "refuted")
                else:
                    self.assertEqual(n["node_depth"], 0)
                    self.assertIn(n["local_lines"][0], G.body_text(it["program"]["hyps"]).split("\n"))
                    if n["claim"]["kind"] == "cong":
                        self.assertNotEqual(G.claim_key(n["claim"]), (z, M, key["r"]))
            yes = collections.Counter(n["type"] for n in nodes if n["truth"])
            self.assertEqual(yes, collections.Counter(n["type"] for n in nodes if not n["truth"]))
            self.assertEqual({n["id"].rstrip("x") for n in nodes if n["type"] != "derive"},
                             set(it["program"]["critical"]))
            # certificates: every option, node, the final from its parents, every critical premise
            roles = collections.Counter(c["role"] for c in it["certificates"])
            self.assertEqual(roles["node/yes"] + roles["node/no"], len(nodes))
            self.assertEqual(roles["necessary"], len(it["program"]["critical"]))
            self.assertEqual((roles["key/omega"], roles["key/grind"], roles["final/parents"]), (1, 1, 1))
            self.assertEqual(roles["independent/holds"], kinds["independent"])
            for c in it["certificates"]:
                if c["tactic"] == "decide":
                    vals = c["witness"]
                    body = c["body"].split(" := ")[0].split(", ", 1)[1]
                    for prop in body.split(" ∧ "):          # every conjunct true of the witness
                        neg = prop.startswith("¬ (")
                        p = prop[3:-1] if neg else prop
                        lhs, rhs = p.split(" = ")
                        ev = lambda e: eval(re.sub(r"\b([a-z])\b", lambda mm: f"({vals[mm.group(1)]})", e))  # noqa
                        self.assertEqual(ev(lhs) == ev(rhs), not neg, (c["role"], prop))

    def test_deterministic(self):
        plan = {"drift": True, "n_drift": 2, "n_on": 1, "non_z": False, "n_irrelevant": 2}
        a, _ = G.build_item(5, ("det", 1), plan)
        b, _ = G.build_item(5, ("det", 1), plan)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        c, _ = G.build_item(5, ("det", 2), plan)
        self.assertNotEqual(a["prompt"], c["prompt"])

    def test_local_views_decide(self):
        it = G.finalize(get_items()[10], "x", "train")
        rng = random.Random(0)
        by = {n["id"]: n for n in it["subqs"]}
        for n in it["subqs"]:
            stated = [] if n.get("local_lines") else [by[p] for p in n["depends_on"]]
            ctx = treeplan.local_context(it, n, stated, rng=rng)
            self.assertEqual(G.solve_view(ctx, n["question"]), n["truth"], ctx)
        self.assertIsNone(G.solve_view(G.PREAMBLE + "\n\nKnown results:\nQ: Does x leave remainder 3 when divided "
                                       "by 4? A: Yes", "Does x leave remainder 7 when divided by 8?"))   # undecided
        self.assertFalse(G.solve_view(G.PREAMBLE + "\n\nKnown results:\nQ: Does x leave remainder 3 when divided "
                                      "by 4? A: Yes", "Does x leave remainder 1 when divided by 8?"))


class TestLeanParsing(unittest.TestCase):
    def test_fake_output(self):
        certs = [G.cert_forall("key/omega", "A", [G.cong("x", 4, 3)], "x % 2 = 1"),
                 G.cert_forall("refuted/omega", "B", [G.cong("x", 4, 3)], "¬ (x % 2 = 0)"),
                 G.cert_exists("independent/holds", "C", [G.cong("x", 4, 3)], "x % 3 = 0", {"x": 3}),
                 G.cert_forall("node/yes", "n1", [G.cong("x", 4, 3)], "x % 8 = 3"),
                 G.cert_forall("node/no", "n1x", [G.cong("x", 4, 3)], "¬ (x % 4 = 1)"),
                 G.cert_forall("key/grind", "A", [G.cong("x", 4, 3)], "x % 2 = 1", "grind")]
        names = [f"c{i:06d}" for i in range(len(certs))]
        text, where = G.lean_file(list(zip(names, certs)))
        lines = text.split("\n")
        self.assertTrue(lines[0].startswith("set_option"))
        self.assertEqual(lines[2], "theorem c000000 (x : Int) (p1 : x % 4 = 3) : x % 2 = 1 := by omega")
        self.assertEqual(lines[3], "#print axioms c000000")
        self.assertEqual(where[3], "c000000")
        self.assertIn("⟨3, by decide⟩", lines[6])

        def msg(line, sev, data):
            return json.dumps({"caption": "", "data": data, "endPos": {"column": 6, "line": line},
                               "fileName": "b.lean", "isSilent": False, "keepFullRange": False,
                               "kind": "[anonymous]", "pos": {"column": 0, "line": line}, "severity": sev})
        out = "\n".join([
            msg(4, "information", "'c000000' depends on axioms: [propext, Quot.sound]"),
            msg(6, "information", "'c000001' depends on axioms: [propext, Classical.choice, Quot.sound]"),
            msg(7, "error", "Tactic `decide` proved that the proposition\n  3 % 4 = 3 ∧ 3 % 3 = 1\nis false"),
            msg(8, "information", "'c000002' depends on axioms: [sorryAx]"),
            msg(9, "error", "omega could not prove the goal"),
            msg(10, "information", "'c000003' depends on axioms: [sorryAx]"),
            # c000004: no axioms report at all (fail closed)
            msg(14, "information", "'c000005' does not depend on any axioms"),
            msg(13, "warning", "declaration uses 'sorry'"),
            "uncaught exception: something odd"])
        res, other = G.parse_lean_output(out, where)
        self.assertEqual([G.certified(res[n]) for n in names], [True, True, False, False, False, True])
        self.assertEqual(res["c000002"]["axioms"], ["sorryAx"])
        self.assertTrue(res["c000003"]["errors"][0].startswith("omega could not"))
        self.assertIsNone(res["c000004"]["axioms"])
        self.assertEqual(other, ["uncaught exception: something odd"])
        self.assertFalse(G.certified({"errors": [], "axioms": ["propext", "Lean.ofReduceBool"]}))
        self.assertFalse(G.certified(None))


class TestStatementBlock(unittest.TestCase):
    def test_lean_f1_cut(self):
        for it in get_items()[:8]:
            block = G.PREAMBLE + "\n\n" + G.body_text(it["program"]["hyps"])
            q = G.QUESTION.format(v=it["program"]["key"]["var"], m=it["program"]["key"]["m"])
            self.assertEqual(statement_block(it), block)
            self.assertEqual(problem_no_options(it), block + "\n\n" + q)
            self.assertTrue(it["prompt"].startswith(block + "\n\n" + q + "\nOptions:\nA) "))
            self.assertNotIn("Question", block)
            for bad in (it["prompt"].replace("What is the remainder", "Which remainder"),
                        it["prompt"] + "\nOptions:\nF) x",
                        it["prompt"].split("\nOptions:\n")[0],
                        "Question: " + it["prompt"].replace(G.PREAMBLE + "\n\n", "Question: x?\n\n", 1)):
                for fn in (statement_block, problem_no_options):
                    with self.assertRaises(ValueError):
                        fn(dict(it, prompt=bad))
        self.assertEqual(treeplan.LEAN_F1, G.SOURCE)
        self.assertNotIn(G.SOURCE, FINAL_QUESTION)


OTHER_FILES = ["student_v3sl/val.jsonl", "student_v3sl/train.jsonl", "student_numgen_tf/train.jsonl",
               "student_numgen_tf/val.jsonl", "student_numgen_tf/local_deep.jsonl", "student_pronto_tf/val.jsonl",
               "student_pronto_tf/train.jsonl", "student_gsm8k_tf/train.jsonl", "student_gsm8k_tf/val.jsonl",
               "student_kk_tf/train.jsonl", "student_goldproof/train.jsonl", "student_rule/train.jsonl",
               "student_fa0/train.jsonl", "student_fa1/train.jsonl", "student_v3s/train.jsonl",
               "eval/kk_heldout.jsonl", "eval/pronto_d5.jsonl", "eval/numgen_d5.jsonl", "eval/gsm8k_mc.jsonl",
               "eval/bbh_sub500.jsonl", "eval/fa0_test.jsonl", "eval/jl_heldout.jsonl"]


class TestOtherSourcesUnchanged(unittest.TestCase):
    """treeplan.py at 9f03003 vs now: the diff only inserts (lean_f1, entailmentbank_tf, strategyqa_tf, qasc_tf, musique_tf) lines, FINAL_QUESTION is the
    same object content, and statement_block / problem_no_options give the same outcome (text or refusal) on fixtures
    and on up to 800 rows of every local data file of every other source."""

    @classmethod
    def setUpClass(cls):
        cls.old = frozen_treeplan()

    def test_diff_only_inserts(self):
        a = FROZEN.read_text().splitlines()
        b = (ROOT / "src/cotdistill/treeplan.py").read_text().splitlines()
        ops = [op for op in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op[0] != "equal"]
        self.assertTrue(ops)
        for tag, i1, i2, j1, j2 in ops:
            self.assertEqual(tag, "insert", (a[i1:i2], b[j1:j2]))
            self.assertTrue(any(k in x.lower() for x in b[j1:j2] for k in ("lean_f1", "entailmentbank_tf", "strategyqa_tf", "qasc_tf",
                                                                         "musique_tf")),
                            b[j1:j2])
        self.assertEqual(self.old.FINAL_QUESTION, FINAL_QUESTION)

    def test_fixtures(self):
        from test_statement_block import KK, NEW, NUMGEN, new_item
        rows = [KK, NUMGEN] + [new_item(s, *v) for s, v in NEW.items()]
        rows += [dict(r, prompt=r["prompt"].replace("Question:", "Q:")) for r in rows]
        rows += [{"item_id": "g", "source": "gsm8k_tf", "prompt": "Question: Tom has 3 apples. He buys 2 more. How "
                  "many apples does Tom have?\nOptions:\nA) 5\nB) 6"},
                 {"item_id": "u", "source": "unknown", "prompt": "x\n\nQuestion: y?\nOptions:\nA) 1"}]
        for r in rows:
            for fn in ("statement_block", "problem_no_options"):
                self.assertEqual(outcome(getattr(treeplan, fn), r), outcome(getattr(self.old, fn), r), (r["source"], fn))

    def test_every_local_source(self):
        n, srcs = 0, collections.Counter()
        for relp in OTHER_FILES:
            p = DATA / relp
            if not p.exists():
                continue
            with open(p) as f:
                for i, line in enumerate(f):
                    if i >= 800:
                        break
                    it = json.loads(line)
                    if it.get("row_kind") == "node":
                        continue
                    for fn in ("statement_block", "problem_no_options"):
                        self.assertEqual(outcome(getattr(treeplan, fn), it), outcome(getattr(self.old, fn), it),
                                         (relp, it.get("item_id"), fn))
                    n += 1
                    srcs[it.get("source")] += 1
        if not n:
            self.skipTest(f"no local data under {DATA}")
        self.assertGreater(len(srcs), 3)


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class TestGFPSViews(unittest.TestCase):
    def test_statements_view(self):
        from cotdistill.student import build_full_tree
        it = G.finalize(get_items()[12], "lean-x", "train")
        block, n = statement_block(it), len(it["subqs"])
        g = build_full_tree(it, mode="plain", view="statements", rng=random.Random(0), cap=0, sources={G.SOURCE})
        p = build_full_tree(it, mode="plain", view="problem", rng=random.Random(0), cap=0, sources={G.SOURCE})
        self.assertEqual(len(g), 1 + n)
        self.assertEqual(g[0].text, p[0].text)
        for eg, ep in zip(g[1:], p[1:]):
            self.assertEqual(eg.text, ep.text.replace(it["prompt"], block, 1))
            self.assertNotIn("What is the remainder", eg.text)
            self.assertEqual((eg.target, eg.weight), (ep.target, ep.weight))


class TestData(unittest.TestCase):
    def setUp(self):
        self.out = DATA / "student_lean_tf_pilot"
        if not (self.out / "summary.json").exists():
            self.skipTest(f"no {self.out}")
        self.s = json.loads((self.out / "summary.json").read_text())

    def test_files(self):
        import build_lean_tf as B
        s = self.s
        self.assertEqual(s["audit"]["failures"], 0)
        self.assertEqual(s["brute_force"]["kept_disagree"], 0)
        for name, want in s["md5"].items():
            path = DATA / Path(name).relative_to("data")
            self.assertEqual(B.md5(path), want, name)
        a = s["args"]
        for name, depths, n in ([(f"eval/lean_f1_d{d}.jsonl", {d}, a["eval_per_depth"])
                                 for d in range(1, a["eval_depths"] + 1)]
                                + [("student_lean_tf_pilot/train.jsonl", set(range(1, a["max_depth"] + 1)),
                                    a["train_per_depth"] * a["max_depth"]),
                                   ("student_lean_tf_pilot/val.jsonl", set(range(1, a["max_depth"] + 1)),
                                    a["val_per_depth"] * a["max_depth"])]):
            rows = [json.loads(x) for x in open(DATA / name)]
            self.assertEqual(len(rows), n, name)
            self.assertEqual({r["depth"] for r in rows}, depths)
            self.assertLessEqual(abs(sum(r["drift"] for r in rows) - n / 2), len(depths), name)
            for r in rows:
                self.assertTrue(all(c["certified"] for c in r["certificates"] if c["role"] != "key/grind"))
                self.assertEqual(r["source"], G.SOURCE)
                if name.startswith("eval/"):
                    self.assertEqual((r["type"], r["label_order"]), ("choice", list(G.LETTERS)))
                    self.assertNotIn("subqs", r)
                else:
                    self.assertTrue(r["subqs"] and r["tree_fix"] and r["local_preamble"] == G.PREAMBLE)
        keys = set()
        for name in ["student_lean_tf_pilot/train.jsonl", "student_lean_tf_pilot/val.jsonl"] + \
                [f"eval/lean_f1_d{d}.jsonl" for d in range(1, a["eval_depths"] + 1)]:
            for line in open(DATA / name):
                k = G.canonical(json.loads(line))
                self.assertNotIn(k, keys)
                keys.add(k)


if __name__ == "__main__":
    unittest.main()
