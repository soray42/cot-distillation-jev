"""PrOntoQA-OOD domain for Tree Forcing (src/cotdistill/pronto.py, scripts/build_pronto_tf.py): the generator's text
parses back into its theory, the deductive closure gives the gold answer at exactly k steps, the proof tree's truths,
depths and local views are exact (the builder's audit, run on fresh problems of every hop count and width), no node
asks the final question, "not" never predicts a node's answer, and the GFP-S statement block of a pronto item is the
problem without its final question (treeplan.FINAL_QUESTION["pronto"]) while the other sources' entries are unchanged.
CPU only; the pipeline test needs torch, the data tests read data/ (or COTD_DATA_DIR) and skip without it."""
import collections
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

from cotdistill import pronto as P  # noqa: E402
from cotdistill.treeplan import FINAL_QUESTION, problem_no_options, statement_block  # noqa: E402
import build_pronto_tf as B  # noqa: E402

DATA = Path(os.environ.get("COTD_DATA_DIR") or ROOT / "data")

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# FINAL_QUESTION before the pronto entry (commit adb41e9): every other source's final question must stay as it was
_AGENT_ASK = r"Question: What should the agent do\?"
OLD_FINAL_QUESTION = {
    "justlogic": r"Question: Is the statement true, false, or uncertain given the passage\?",
    "folio": r"Question: Is the statement true, false, or uncertain given the premises\?",
    "sharc": r"Question: Based only on the rule text, the scenario and the earlier answers, how should the user's "
             r"question be answered\?",
    "returns": _AGENT_ASK, "expense": _AGENT_ASK, "subscription": _AGENT_ASK,
    "rule": r"(?:Question: )?What should the [a-z][a-z' -]* do\?",
    "goldproof/fld": r"Question: Based only on the facts, is the hypothesis proved, disproved, or neither \(unknown\)\?",
    "goldproof/boardgameqa": r"Question: Based on the game state and the rules and preferences, [^\n]+\?",
    "goldproof/ruletaker": r"Question: Does the statement follow from the context\? What is not explicitly stated as "
                           r"true is considered false\.",
}


def problems(seed=0, hops=range(1, 9), widths=(0, 2), per=4):
    rng = random.Random(seed)
    for k in hops:
        for w in widths:
            for j in range(per):
                yield P.make_problem(k, j % 2 == 0, j // 2 % 2 == 1, rng, width=w)


class TestGenerator(unittest.TestCase):
    def test_round_trip_and_closure(self):
        for th in problems():
            body = P.body_text(th)
            facts, rules = P.parse_body(body)
            self.assertEqual(sorted(facts), sorted(map(tuple, th["facts"])))
            self.assertEqual(sorted(rules), sorted(tuple(r[:4]) for r in th["rules"]))
            cl = P.closure(facts, rules)                   # raises on a contradiction
            neg, steps = cl[(th["entity"], th["property"])]
            self.assertEqual(steps, th["hops"])
            self.assertEqual(neg == th["query_neg"], th["answer"])
            self.assertEqual(th["rule_neg"], neg)
            prules = [r for r in rules if r[1] == th["property"]]
            self.assertEqual(sorted(r[3] for r in prules), [False, True])      # the twin chain's opposite rule
            self.assertFalse(re.search(r"\d", body))

    def test_determinism_and_canonical(self):
        a = list(problems(seed=3, hops=[4], widths=[0], per=4))
        b = list(problems(seed=3, hops=[4], widths=[0], per=4))
        self.assertEqual(a, b)
        self.assertEqual(len({P.canonical(t) for t in a}), 4)
        t = a[0]
        shuffled = dict(t, rules=[r[:4] + [(r[4] + 1) % 3] for r in reversed(t["rules"])])
        self.assertEqual(P.canonical(shuffled), P.canonical(t))         # templates and order are not identity

    def test_prompt_shape(self):
        th = next(problems(hops=[3], widths=[0], per=1))
        it = P.train_item(th, "x")
        head, _, opts = it["prompt"].partition("\nOptions:\n")
        self.assertEqual(opts, "A) True\nB) False")
        self.assertTrue(it["prompt"].startswith(P.LOCAL_PREAMBLE + "\n\n"))
        self.assertEqual(head.rpartition("\n\n")[2], "Question: True or false: " + P.query_text(th))
        self.assertEqual(it["gold_label"], "A" if th["answer"] else "B")
        self.assertEqual(it["teacher"][it["gold_label"]], 1.0)
        for s in ("Every wumpus is a yumpus.", "Each impus is not an impus.", "Wumpuses are not yumpuses.",
                  "Lempuses are sour.", "Each lempus is not sour.", "Wren is an impus."):
            p = P.parse_sentence(s)
            self.assertIsNotNone(p, s)
            self.assertEqual(P.rule_sentence(p[1], {"Every": 0, "Each": 1}.get(s.split()[0], 2))
                             if p[0] == "rule" else P.fact_sentence(p[1]), s)

    def test_vocabulary(self):
        self.assertEqual(len(set(P.CONCEPTS)), len(P.CONCEPTS))
        self.assertGreater(len(P.CONCEPTS), 400)
        self.assertTrue(all(re.fullmatch(r"[a-z]+pus", c) for c in P.CONCEPTS))
        self.assertTrue(all(re.fullmatch(r"[A-Z][a-z]+", e) for e in P.ENTITIES))
        self.assertEqual(len(P.ADJECTIVES), 45)
        self.assertEqual(P.CONCEPTS[:38], P.PRONTO_CONCEPTS)


class TestTree(unittest.TestCase):
    def test_audit_passes(self):
        """The builder's audit (closure truths, depths, parents, leaks, local-view re-solving) on fresh problems."""
        fails, a = [], collections.Counter()
        n = 0
        for th in problems(seed=1):
            it = P.train_item(th, f"t{n}")
            B.audit_item(it, it["subqs"], random.Random(n), a, fails)
            n += 1
        self.assertEqual(fails, [])
        self.assertEqual(a["leak_node_question"], 0)
        self.assertEqual(a["view_leak"], 0)
        self.assertEqual(a["local_views_sufficient/derive"], a["local_views/derive"])
        self.assertGreater(a["local_views/derive"], 100)

    def test_shape_balance_and_no_not_cue(self):
        for th in problems(seed=2, widths=(0,), per=2):
            k = th["hops"]
            nodes = P.proof_tree(th)
            self.assertEqual(len(nodes), 6 * k - 3)
            yes = sum(n["truth"] for n in nodes)
            self.assertEqual((yes, len(nodes) - yes), (3 * k - 1, 3 * k - 2))
            self.assertEqual(max(n["node_depth"] for n in nodes), k - 1)
            by = {n["id"]: n for n in nodes}
            for n in nodes:
                self.assertFalse(P.node_leak(n["question"], th), n["question"])
                if n["type"] == "derive":
                    self.assertEqual(n["depends_on"], [f"m{n['level'] - 1}" if n["level"] > 1 else "r0",
                                                       f"p{n['level']}", f"q{n['level']}"])
                    self.assertNotIn(" not ", n["question"])
                    self.assertTrue(all(by[d]["truth"] for d in n["depends_on"]))
            concept_parse = [n for n in nodes if n["kind"] in ("concept_rule", "twin_rule")]
            cnt = collections.Counter((" not " in n["question"], n["truth"]) for n in concept_parse)
            self.assertEqual(cnt[(True, True)], cnt[(True, False)])            # "not" says nothing about the answer
            self.assertEqual(cnt[(False, True)], cnt[(False, False)])

    def test_node_leak(self):
        th = next(problems(hops=[2], widths=[0], per=1))
        e, p = th["entity"], th["property"]
        for q in (f"Is {e} {p}?", f"Is {e} not {p}?", f"Does it follow that {e} is {p}?"):
            self.assertTrue(P.node_leak(q, th))
        self.assertFalse(P.node_leak(f"Is every {th['pairs'][0][0][-1]} {p}?", th))

    def test_solve_view_refuses_undecided(self):
        ctx = P.LOCAL_PREAMBLE + "\n\nKnown results:\nQ: Is Wren a lempus? A: Yes\nQ: Is every lempus a zumpus? A: Yes"
        self.assertTrue(B.solve_view(ctx, "Is Wren a zumpus?"))
        self.assertIsNone(B.solve_view(ctx, "Is Wren a gorpus?"))             # unknown, never "No"
        ctx2 = ctx + "\nQ: Is every lempus not a gorpus? A: Yes"
        self.assertFalse(B.solve_view(ctx2, "Is Wren a gorpus?"))
        self.assertFalse(B.solve_view(P.LOCAL_PREAMBLE + "\n\nEvery lempus is sour.", "Is every lempus not sour?"))


class TestStatementBlock(unittest.TestCase):
    def test_final_question_entry(self):
        self.assertEqual(FINAL_QUESTION["pronto"], P.FINAL_RE)
        self.assertEqual({k: v for k, v in FINAL_QUESTION.items() if k != "pronto"}, OLD_FINAL_QUESTION)

    def test_cut(self):
        for th in problems(seed=4, hops=[1, 5], widths=(0, 2), per=2):
            it = P.eval_item(th, "x", "pronto/d1")
            block = P.LOCAL_PREAMBLE + "\n\n" + P.body_text(th)
            self.assertEqual(statement_block(it), block)
            self.assertEqual(problem_no_options(it), block + "\n\nQuestion: True or false: " + P.query_text(th))
            self.assertNotIn(P.query_text(th), block)
            bad = dict(it, prompt=it["prompt"].replace("True or false:", "Is it true that"))
            with self.assertRaises(ValueError):
                statement_block(bad)


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class TestPipeline(unittest.TestCase):
    def test_arms_render(self):
        from cotdistill.student import build_full_tree, local_eval_examples
        th = P.make_problem(4, False, True, random.Random(7))
        it = P.train_item(th, "pronto-x")
        block = statement_block(it)
        n = len(it["subqs"])
        g = build_full_tree(it, mode="plain", view="statements", rng=random.Random(0), cap=0, sources={"pronto"})
        p = build_full_tree(it, mode="plain", view="problem", rng=random.Random(0), cap=0, sources={"pronto"})
        self.assertEqual(len(g), 1 + n)
        self.assertEqual(g[0].text, p[0].text)
        self.assertEqual(g[0].target, [0.0, 1.0])
        for eg, ep in zip(g[1:], p[1:]):
            self.assertEqual(eg.text, ep.text.replace(it["prompt"], block, 1))
            self.assertTrue(eg.text.startswith(block + "\n\nIntermediate question: "))
            self.assertNotIn("True or false", eg.text)
            self.assertEqual((eg.target, eg.weight), (ep.target, ep.weight))
        other = dict(it, source="numgen")
        self.assertEqual(len(build_full_tree(other, mode="plain", view="statements", rng=random.Random(0), cap=0,
                                             sources={"pronto"})), 1)
        for mode in ("true", "matched", "plain"):
            t = build_full_tree(it, mode=mode, view="local", rng=random.Random(0), cap=0, sources={"pronto"})
            self.assertEqual(len(t), 1 + n, mode)
            self.assertTrue(all(e.text.startswith(P.LOCAL_PREAMBLE) for e in t[1:]))
        exs, meta = local_eval_examples([it])
        self.assertEqual(len(exs), n)
        self.assertEqual(sum(m["target"] for m in meta), sum(x["truth"] for x in it["subqs"]))


class TestData(unittest.TestCase):
    def setUp(self):
        self.out = DATA / "student_pronto_tf"
        if not (self.out / "summary.json").exists():
            self.skipTest(f"no {self.out}")
        self.summary = json.loads((self.out / "summary.json").read_text())

    def test_files(self):
        s = self.summary
        self.assertEqual(s["audit"]["failures"], 0)
        for name, want in s["md5"].items():
            path = ROOT / name if not name.startswith("/") else Path(name)
            if DATA != ROOT / "data":
                path = DATA / Path(name).relative_to("data")
            self.assertEqual(B.md5(path), want, name)
        keys = set()
        for name, hops, n in [("student_pronto_tf/train.jsonl", {1, 2, 3, 4}, 10000),
                              ("student_pronto_tf/val.jsonl", {1, 2, 3, 4}, 400)] + \
                [(f"eval/pronto_d{k}.jsonl", {k}, 400) for k in range(1, 9)] + [("eval/pronto_wide.jsonl", {1, 2, 3, 4},
                                                                                   400)]:
            items = [json.loads(x) for x in open(DATA / name)]
            self.assertEqual(len(items), n, name)
            self.assertEqual({it["hops"] for it in items}, hops, name)
            self.assertEqual(sum(it["gold_label"] == "A" for it in items), n // 2, name)
            self.assertEqual(sum(it["query_neg"] for it in items), n // 2, name)
            for it in items:
                key = P.canonical(it["theory"])
                self.assertNotIn(key, keys, it["item_id"])
                keys.add(key)
                self.assertEqual(it["prompt"], P.prompt_of(it["theory"]))
                if name.startswith("eval/"):
                    self.assertEqual((it["type"], it["label_order"]), ("choice", ["A", "B"]))
                    self.assertTrue(it["group"].startswith("pronto/"))
                    self.assertNotIn("subqs", it)
                else:
                    self.assertEqual(it["subqs"], P.proof_tree(it["theory"]))
        deep = [json.loads(x) for x in open(self.out / "local_deep.jsonl")]
        self.assertEqual(collections.Counter(it["hops"] for it in deep), {5: 100, 6: 100, 7: 100, 8: 100})
        self.assertTrue(all(it["subqs"] and it["local_preamble"] == P.LOCAL_PREAMBLE for it in deep))
        for split, t in s["tokens_final_prompt"].items():
            self.assertGreaterEqual(t["share_le_1536"], 0.99, split)


if __name__ == "__main__":
    unittest.main()
