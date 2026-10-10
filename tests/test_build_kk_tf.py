"""scripts/build_kk_tf.py (single-domain K&K Tree Forcing data): a small build (--n 14, --val-n 7, no tokens) into a
temporary directory, run twice, checked for
- format: the builder's own check against the v3sl K&K items, plus a direct comparison with a v3sl K&K item (item key
  list, node key list per type, value types); prompt shape; one-hot teacher on the gold letter; the solver's answer;
  the GFP-S statement block; "None of the above" exactly 4 wrong + 1 right per 50 positions;
- disjointness: no prompt, statement block or statement-line set of a new item occurs in data/bench/train_kk.jsonl,
  data/student_v3sl or any data/eval/*kk*.jsonl / diag*.jsonl file; seeds outside every other K&K builder's block;
- entailment: every derived node follows from its stated parents over all role assignments, re-solved here from the
  written prompt (solver masks), and every node's stored truth is its proposition under the unique solution;
- determinism: identical md5 over two runs; --n 7 writes the first 7 lines of --n 14; val does not depend on --n;
- balance: people counts of every train prefix differ by at most one.
Needs the solver (teacher_cache/) and data/ (not in git): skipped without them. CPU only, stdlib."""
import collections
import functools
import hashlib
import itertools
import json
import random
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SOLVER = ROOT / "teacher_cache/solver_kk_v3f/build_reasoning_trees.py"
V3SL = ROOT / "data/student_v3sl/train.jsonl"
SEED_TEST = 99000                     # fresh puzzles for the forced-augmentation test only
HAVE = SOLVER.exists() and V3SL.exists() and (ROOT / "data/bench/train_kk.jsonl").exists()
PROMPT = re.compile(r"^On an island, knights always tell the truth and knaves always lie\. Each of the (\d+) "
                    r"inhabitants below is either a knight or a knave\.\n\n((?:- [A-Z][a-z]+ says: \"[^\"\n]+\"\n)+)"
                    r"\nQuestion: Who is a knight and who is a knave\?\nOptions:\n((?:[A-G]\) [^\n]+\n?)+)$")


def read(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(x) for x in f]


@unittest.skipUnless(HAVE, "needs teacher_cache/solver_kk_v3f and data/ (student_v3sl, bench/train_kk)")
class TestBuildKkTf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import build_kk_tf as B
        cls.B = B
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.sum1 = B.main(["--out", str(d / "a"), "--n", "14", "--val-n", "7", "--no-tokens"])
        cls.sum2 = B.main(["--out", str(d / "b"), "--n", "14", "--val-n", "7", "--no-tokens"])
        cls.sum3 = B.main(["--out", str(d / "c"), "--n", "7", "--val-n", "7", "--no-tokens"])
        cls.dirs = {k: d / k for k in "abc"}
        cls.train = read(d / "a/train.jsonl")
        cls.val = read(d / "a/val.jsonl")
        cls.items = cls.val + cls.train
        cls.solver = B.load_solver(SOLVER)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ---------------------------------------------------------------------------------------------- format
    def test_builder_format_check(self):
        self.assertTrue(self.sum1["format"]["equal"], self.sum1["format"]["diffs"])
        self.assertEqual(self.sum1["sufficiency"]["not_entailed"], 0)
        self.assertEqual(self.sum1["sufficiency"]["entailed"], self.sum1["sufficiency"]["derived_nodes"])
        self.assertEqual(self.sum1["sufficiency"]["kk_items"], 21)

    def test_same_layout_as_a_v3sl_kk_item(self):
        ref = None
        with open(V3SL) as f:
            for line in f:
                it = json.loads(line)
                if it.get("source") == "knights_knaves":
                    ref = it
                    break
        ref_nodes = {sq["type"]: sq for sq in ref["subqs"]}
        self.assertEqual(set(ref_nodes), {"parse", "case", "derive"})
        for it in self.items:
            self.assertEqual(list(it), list(ref), it["item_id"])
            for k in ("item_id", "source", "prompt", "labels", "gold_label", "teacher", "depth", "subqs",
                      "random_subqs", "rationale", "n_inconsistent_dropped", "tree_fix", "n_leak_dropped", "has_tree",
                      "tree_source", "local_preamble"):
                self.assertIs(type(it[k]), type(ref[k]), k)
            self.assertEqual((it["source"], it["tree_fix"], it["has_tree"]), ("knights_knaves", True, True))
            self.assertEqual(it["tree_source"], "solver_kk_tf")
            self.assertEqual(it["random_subqs"], [])
            types = {sq["type"] for sq in it["subqs"]}
            self.assertLessEqual(types, {"parse", "case", "derive"})
            self.assertIn("parse", types)
            for sq in it["subqs"]:
                r = ref_nodes[sq["type"]]
                self.assertEqual(list(sq), list(r), (it["item_id"], sq["id"]))
                for k in sq:
                    self.assertIs(type(sq[k]), type(r[k]), (sq["type"], k))
                self.assertEqual(sq["p_cot"], 1.0 if sq["truth"] else 0.0)
                self.assertEqual((sq["source_tree"], sq["status"]), ("solver", "direct"))
                self.assertNotEqual(sq["opposite"], sq["question"])
                self.assertTrue(sq["question"].endswith("?") and sq["opposite"].endswith("?"))

    def test_prompt_options_gold(self):
        from cotdistill.treeplan import statement_block
        for it in self.items:
            m = PROMPT.match(it["prompt"])
            self.assertIsNotNone(m, it["prompt"])
            n = int(m.group(1))
            self.assertEqual(n, it["depth"])
            self.assertIn(n, range(4, 11))
            names = re.findall(r"^- ([A-Z][a-z]+) says", m.group(2), re.M)
            self.assertEqual(len(names), n)
            opts = re.findall(r"^([A-G])\) (.+)$", m.group(3), re.M)
            self.assertEqual([x for x, _ in opts], it["labels"])
            self.assertEqual(len(opts), 6 + (it["augment"] == "none_wrong"))
            none = [x for x, t in opts if t == "None of the above"]
            self.assertEqual(len(none), it["augment"] is not None)
            if none:
                self.assertEqual(none[0], it["labels"][-1])
            for x, t in opts:
                if t != "None of the above":
                    self.assertEqual([p for p, _ in re.findall(r"([A-Z][a-z]+): (knight|knave)", t)], names)
            self.assertEqual(it["teacher"], {x: 1.0 if x == it["gold_label"] else 0.0 for x in it["labels"]})
            self.assertEqual(it["gold_label"] == none[0] if none else False, it["augment"] == "none_right")
            self.assertEqual(statement_block(it), it["prompt"].partition("\n\nQuestion: ")[0])
            self.assertTrue(it["prompt"].startswith(it["local_preamble"] + "\n\n- "))
            self.assertTrue(it["item_id"].startswith("kktf-"))

    def test_gold_is_the_unique_solution(self):
        S = self.solver
        for it in self.items:
            pz = S.Puzzle.read({"item_id": it["item_id"], "prompt": it["prompt"]}, "test")
            models, _ = S.independent_models(pz)
            self.assertEqual(len(models), 1)
            good = [x for x, a in pz.options.items() if a == models[0]]
            self.assertEqual(good or [S.NONE_OPTION[it["item_id"]]], [it["gold_label"]])

    def test_augmentation_exact_rate(self):
        """inject_none's rates (8% wrong / 2% right "None of the above") exactly in every block of 50 positions, and
        each written item carries the augmentation of its position."""
        B = self.B
        kind = lambda u: "none_wrong" if u < 0.08 else "none_right" if u < 0.10 else None   # noqa: E731
        for split in ("train", "val"):
            for b in range(20):
                c = collections.Counter(kind(B.aug_u(0, split, i)) for i in range(50 * b, 50 * b + 50))
                self.assertEqual((c["none_wrong"], c["none_right"]), (4, 1))
        for split, items in (("val", self.val), ("train", self.train)):
            for i, it in enumerate(items):
                self.assertEqual(it["augment"], kind(B.aug_u(0, split, i)), it["item_id"])

    def test_forced_augmentations(self):
        """Both "None of the above" variants on fresh puzzles of every size: prompt shape, gold, teacher, the solver's
        answer (make_item asserts it) and the audit."""
        import collections as C
        from cotdistill import sources as S
        B = self.B
        for k in (4, 7, 10):
            r = S.knights_knaves(1, seed=SEED_TEST + k, people_range=(k, k))[0]
            for u, aug in ((0.0, "none_wrong"), (0.09, "none_right"), (0.5, None)):
                c, bad = C.Counter(), []
                it = B.make_item(r, self.solver, 0, u, c, bad)
                self.assertEqual(it["augment"], aug)
                self.assertEqual(bad, [])
                self.assertEqual(c["entailed"], c["derived_nodes"])
                m = PROMPT.match(it["prompt"])
                self.assertIsNotNone(m)
                opts = re.findall(r"^([A-G])\) (.+)$", m.group(3), re.M)
                self.assertEqual(len(opts), 7 if aug == "none_wrong" else 6)
                self.assertEqual(opts[-1][1] == "None of the above", aug is not None)
                self.assertEqual(it["gold_label"] == opts[-1][0], aug == "none_right")
                self.assertEqual(sum(it["teacher"].values()), 1.0)
                self.assertEqual(it["teacher"][it["gold_label"]], 1.0)

    # ---------------------------------------------------------------------------------------------- disjointness
    def test_disjoint_from_training_and_eval_sets(self):
        files = [ROOT / "data/bench/train_kk.jsonl", ROOT / "data/student_v3sl/train.jsonl",
                 ROOT / "data/student_v3sl/val.jsonl"]
        files += sorted((ROOT / "data/eval").glob("*kk*.jsonl")) + sorted((ROOT / "data/eval").glob("diag*.jsonl"))
        theirs = set()
        for p in files:
            with open(p) as f:
                for line in f:
                    rec = json.loads(line)
                    for k, v in rec.items():
                        if isinstance(v, str) and (k == "prompt" or k.endswith("_prompt")):
                            theirs.add(v)
                            theirs.add(v.partition("\n\nQuestion: ")[0])
                            theirs.add(frozenset(re.findall(r'^- [A-Z][a-z]+ says: "[^"\n]+"$', v, re.M)))
        theirs.discard(frozenset())
        self.assertGreater(len(theirs), 10000)
        for it in self.items:
            p = it["prompt"]
            for key in (p, p.partition("\n\nQuestion: ")[0], frozenset(re.findall(r'^- [A-Z][a-z]+ says: "[^"\n]+"$',
                                                                                   p, re.M))):
                self.assertNotIn(key, theirs, it["item_id"])
        blocks = [it["prompt"].partition("\n\nQuestion: ")[0] for it in self.items]
        self.assertEqual(len(set(blocks)), len(blocks))
        # seeds: train 60000 + k, val 61000 + k; other K&K builders: 100, 9000, 30000, 50000, 71000 (+ k)
        for it, split in [(x, "val") for x in self.val] + [(x, "train") for x in self.train]:
            seed = int(it["item_id"].split("-")[1])
            self.assertEqual(seed, (61000 if split == "val" else 60000) + it["depth"])
        used = {s + k for s in (100, 9000, 30000, 50000, 71000) for k in range(2, 21)}
        self.assertFalse({60000 + k for k in range(4, 11)} & used or {61000 + k for k in range(4, 11)} & used)
        dedup = self.sum1["dedup"]["files"]
        for rel in ("data/bench/train_kk.jsonl", "data/eval/kk_heldout.jsonl", "data/eval/kk_deep.jsonl",
                    "data/eval/diag_kk.jsonl", "data/eval/kk_audit_pairs.jsonl", "data/student_v3sl/train.jsonl"):
            if (ROOT / rel).exists():
                self.assertGreater(dedup[rel]["prompts"], 0, rel)

    # ---------------------------------------------------------------------------------------------- entailment
    def test_every_derived_node_entailed(self):
        from build_local_kk import masks
        S = self.solver
        n_derived = 0
        for it in self.items:
            pz = S.Puzzle.read({"item_id": it["item_id"], "prompt": it["prompt"]}, "test")
            b = S.ProofBuilder(pz)
            solution, _ = b.solve()
            m, universe = masks(S, b)
            raw = {n["id"]: n for n in b.nodes}
            sol_bit = 1 << [i for i, bits in enumerate(itertools.product((False, True), repeat=len(pz.names)))
                            if dict(zip(pz.names, bits)) == solution][0]
            ids = [sq["id"] for sq in it["subqs"]]
            for sq in it["subqs"]:
                r = raw[sq["id"]]
                self.assertTrue(m[sq["id"]] & sol_bit)           # every raw node holds under the solution
                if sq["question"] == r["question"]:              # stored truth = the raw answer, or its flip
                    self.assertEqual(sq["truth"], r["answer"] == "yes")
                else:
                    self.assertEqual((sq["question"], sq["truth"]), (r["opposite"], r["answer"] != "yes"))
                self.assertLessEqual(set(sq["depends_on"]), set(ids[:ids.index(sq["id"])]))
                if sq["type"] == "parse":
                    self.assertEqual(sq["depends_on"], [])
                    self.assertEqual(len(sq["local_lines"]), 1)
                    self.assertIn(sq["local_lines"][0], it["prompt"].split("\n"))
                    continue
                n_derived += 1
                self.assertTrue(sq["depends_on"])
                prem = functools.reduce(lambda a, x: a & m[x], sq["depends_on"], universe)
                self.assertFalse(prem & (universe ^ m[sq["id"]]), (it["item_id"], sq["id"], sq["question"]))
                self.assertEqual(sq["local_trivial"], m[sq["id"]] == universe)
            depth = {}
            for sq in it["subqs"]:
                depth[sq["id"]] = 1 + max(depth[d] for d in sq["depends_on"]) if sq["depends_on"] else 0
                self.assertEqual(sq["node_depth"], depth[sq["id"]])
            self.assertFalse(any(sq["question"].startswith(("Is each of the ", "Must a knight", "Must a knave"))
                                 for sq in it["subqs"]))
        self.assertGreater(n_derived, 200)

    def test_about_half_the_wordings_flipped(self):
        S = self.solver
        flipped = total = 0
        for it in self.items:
            b = S.ProofBuilder(S.Puzzle.read({"item_id": it["item_id"], "prompt": it["prompt"]}, "test"))
            b.solve()
            raw = {n["id"]: n for n in b.nodes}
            for sq in it["subqs"]:
                total += 1
                flipped += sq["question"] != raw[sq["id"]]["question"]
        self.assertAlmostEqual(flipped / total, 0.5, delta=0.06)

    # ---------------------------------------------------------------------------------------------- determinism
    def test_deterministic_and_prefix_stable(self):
        md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()   # noqa: E731
        a, b, c = self.dirs["a"], self.dirs["b"], self.dirs["c"]
        for split in ("train", "val"):
            self.assertEqual(md5(a / f"{split}.jsonl"), md5(b / f"{split}.jsonl"), split)
        self.assertEqual(md5(a / "val.jsonl"), md5(c / "val.jsonl"))
        lines = (a / "train.jsonl").read_text().splitlines(keepends=True)
        self.assertEqual("".join(lines[:7]), (c / "train.jsonl").read_text())
        self.assertEqual(list(self.sum1["md5"].values()), list(self.sum2["md5"].values()))
        self.assertEqual(self.sum1["md5"]["a/train.jsonl"], md5(a / "train.jsonl"))

    # ---------------------------------------------------------------------------------------------- balance
    def test_people_balanced_in_every_prefix(self):
        for split in (self.train, self.val):
            c = collections.Counter()
            for i, it in enumerate(split):
                c[it["depth"]] += 1
                counts = [c[k] for k in range(4, 11)]
                self.assertLessEqual(max(counts) - min(counts), 1, i)
        self.assertEqual([it["depth"] for it in self.train[:7]], list(range(4, 11)))


class TestHelpers(unittest.TestCase):
    def test_puzzle_keys_and_budget_batches(self):
        import build_kk_tf as B
        p = ('On an island, knights always tell the truth and knaves always lie. Each of the 2 inhabitants below is '
             'either a knight or a knave.\n\n- Bob says: "Ann is a knave."\n- Ann says: "Bob is a knight."\n\n'
             'Question: Who is a knight and who is a knave?\nOptions:\nA) x')
        q = p.replace('- Bob says: "Ann is a knave."\n- Ann says: "Bob is a knight."',
                      '- Ann says: "Bob is a knight."\n- Bob says: "Ann is a knave."')
        kp, kq = B.puzzle_keys(p), B.puzzle_keys(q)
        self.assertEqual(len(kp), 3)
        self.assertNotEqual(kp[:2], kq[:2])
        self.assertEqual(kp[2], kq[2])                             # same statements, other speaker order
        self.assertEqual(B.puzzle_keys(p.replace("A) x", "A) y"))[1:], kp[1:])
        bs = B.budget_batches([100, 900, 300, 300, 2000], 1024)
        self.assertEqual(sorted(i for b in bs for i in b), list(range(5)))
        self.assertEqual(bs[0], [4])                               # longer than the budget: alone
        for b in bs:
            self.assertTrue(len(b) == 1 or len(b) * max([100, 900, 300, 300, 2000][i] for i in b) <= 1024)

    def test_epoch_updates(self):
        import build_kk_tf as B
        e = B.Epoch(1536)
        for _ in range(B.IPU + 1):
            e.add(10, [20, 30])
        r = e.row()
        self.assertEqual((r["items"], r["updates"], r["views"]), (B.IPU + 1, 2, 2 * (B.IPU + 1)))
        self.assertEqual(r["tokens_gfps"], 60 * (B.IPU + 1))
        e2 = B.Epoch(1536)
        e2.add(5000, [10])
        self.assertEqual(e2.row()["tokens_final"], 1536)          # capped at max_len as collate truncates


if __name__ == "__main__":
    unittest.main()
