"""QASC Tree Forcing data (scripts/build_qasc_tf.py, source qasc_tf) and its treeplan branch:
- treeplan: problem_no_options (GFPQ) is the question line, statement_block refuses, shape refusals;
- the built files (skipped when absent): schema, Yes/No balance per item and per fact, no leak (no combined fact, no
  near-final fact, twins with the same gold / option mentions, gold never in the question), the item audit on a
  handful of items, disjoint questions across train / val / test, md5s of summary.json, the f25 / f10 subsets;
- the subset recipe reproduces data/student_gsm8k_tf_f25 (skipped when absent);
- with torch: the GFPQ and placebo views of a few items.
CPU only. Data-backed tests read data/ of this checkout, or COTD_DATA_DIR when set."""
import hashlib
import json
import os
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_qasc_tf as B  # noqa: E402
from cotdistill.treeplan import QASC_TF, problem_no_options, statement_block  # noqa: E402

DATA = Path(os.environ.get("COTD_DATA_DIR") or ROOT / "data")
FILES = {"train": DATA / "student_qasc_tf/train.jsonl", "val": DATA / "student_qasc_tf/val.jsonl",
         "test": DATA / "eval/qasc_tf_test.jsonl"}
HANDFUL = 40
ITEM_KEYS = {"item_id", "source", "group", "type", "prompt", "labels", "gold_label", "teacher", "options",
             "option_kinds", "recipe", "depth", "random_subqs", "rationale", "augment", "has_tree", "tree_fix",
             "row_kind", "problem_id", "node_status", "licence_class", "licence", "label_basis", "node_basis",
             "n_steps", "tree_source", "subqs", "qasc"}
NODE_KEYS = {"id", "question", "opposite", "truth", "p_cot", "p_fresh", "type", "depends_on", "initial", "status",
             "fact", "answer_in_fact", "question_coverage", "gold_ancestor", "flags", "node_depth", "statement",
             "answer_in_node"}
OPTS = "\n".join(f"{L}) option {L.lower()}" for L in "ABCDEFGH")

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def read(path: Path, n: int | None = None) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
            if n and len(out) >= n:
                break
    return out


def item(q: str = "What forms beads of water?", opts: str = OPTS) -> dict:
    return {"item_id": "qasc-x", "source": QASC_TF, "prompt": f"Question: {q}\nOptions:\n{opts}"}


class TestTreeplanBranch(unittest.TestCase):
    def test_gfpq_is_the_question_line_and_gfps_refuses(self):
        it = item()
        self.assertEqual(problem_no_options(it), "Question: What forms beads of water?")
        with self.assertRaises(ValueError):
            statement_block(it)

    def test_shape_refusals(self):
        for bad in (item("two\nlines"), {**item(), "prompt": "What? \nOptions:\n" + OPTS},
                    item("Question: nested")):
            with self.assertRaises(ValueError):
                problem_no_options(bad)

    @unittest.skipUnless(HAVE_TORCH, "cotdistill.student needs torch")
    def test_placebo_text_is_the_question(self):
        from cotdistill.student import placebo_text
        self.assertEqual(placebo_text(item()), "What forms beads of water?")


class TestHelpers(unittest.TestCase):
    def test_has_is_whole_word_and_normalised(self):
        self.assertTrue(B.has("An example of water vapor is steam.", "Steam."))
        self.assertFalse(B.has("Steamboats are fast", "steam"))

    def test_near_final(self):
        row = {"question": "What is the average weather in a place over time?", "options": ["climate", "x"],
               "option_labels": ["A", "B"], "answerKey": "A"}
        self.assertTrue(B.near_final(row, "Climate is the average weather in an area over a long period of time"))
        self.assertFalse(B.near_final(row, "Weather is the state of the atmosphere"))

    def test_subset_recipe(self):
        rows = [{"item_id": f"i{k}", "group": f"g{k % 3}"} for k in range(40)]
        a, b = B.subset(rows, 0.25, "s"), B.subset(rows, 0.25, "s")
        self.assertEqual(a, b)
        self.assertEqual([r["item_id"] for r in a], [r["item_id"] for r in rows if r in a])   # file order
        self.assertEqual(len(a), sum(round(0.25 * n) for n in (14, 13, 13)))

    @unittest.skipUnless((DATA / "student_gsm8k_tf_f25/summary.json").exists(), "no gsm8k_tf data")
    def test_subset_reproduces_gsm8k_f25(self):
        sm = json.loads((DATA / "student_gsm8k_tf_f25/summary.json").read_text())
        lines = (DATA / "student_gsm8k_tf/train.jsonl").read_text().split("\n")[:-1]
        rows = [json.loads(x) for x in lines]
        keep = {r["item_id"] for r in B.subset(rows, sm["fraction"], sm["seed"])}
        text = "".join(x + "\n" for x, r in zip(lines, rows) if r["item_id"] in keep)
        self.assertEqual(hashlib.md5(text.encode()).hexdigest(), sm["train_md5"])


@unittest.skipUnless(all(p.exists() for p in FILES.values()), "no qasc_tf data (python3 scripts/build_qasc_tf.py)")
class TestBuiltFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = {k: read(p, HANDFUL) for k, p in FILES.items()}

    def test_schema(self):
        for split, its in self.items.items():
            for it in its:
                self.assertEqual(set(it), ITEM_KEYS, it["item_id"])
                self.assertEqual(it["labels"], list("ABCDEFGH"))
                self.assertEqual(it["teacher"], {L: float(L == it["gold_label"]) for L in it["labels"]})
                self.assertEqual(problem_no_options(it), it["prompt"].partition("\nOptions:")[0])
                self.assertEqual(it["group"], f"qasc_tf/n{len(it['subqs']) // 2}")
                for n in it["subqs"]:
                    self.assertEqual(set(n) - {"swap"}, NODE_KEYS)
                    self.assertEqual("swap" in n, not n["truth"])
                    self.assertNotIn("options", n)              # never a multiple-choice node
                    self.assertEqual((n["node_depth"], n["depends_on"]), (0, []))

    def test_balance(self):
        for its in self.items.values():
            for it in its:
                by = {n["id"]: n for n in it["subqs"]}
                self.assertEqual(sum(n["truth"] for n in it["subqs"]) * 2, len(it["subqs"]))
                for i, n in by.items():
                    twin = by[i[:-1] + ("n" if n["truth"] else "v")]
                    self.assertNotEqual(twin["truth"], n["truth"])
                    self.assertEqual(twin["type"], n["type"])

    def test_no_leak(self):
        from cotdistill.teacher import leaks_final
        for its in self.items.values():
            for it in its:
                gold = B.gold_text(it)
                self.assertFalse(B.has(B.question_of(it), gold))
                comb = B.norm(B.statement(it["qasc"]["combinedfact"]))
                by = {n["id"]: n for n in it["subqs"]}
                for n in it["subqs"]:
                    self.assertNotEqual(B.norm(n["statement"]), comb)
                    self.assertFalse(n["answer_in_fact"] and n["question_coverage"] >= B.NEAR_FINAL)
                    self.assertFalse(leaks_final(n["question"], it))
                    twin = by[n["id"][:-1] + ("n" if n["truth"] else "v")]
                    for o in it["options"]:
                        self.assertEqual(B.has(n["question"], o), B.has(twin["question"], o))

    def test_audit(self):
        for its in self.items.values():
            for it in its:
                self.assertEqual(B.audit_item(it), [])

    def test_splits_and_md5(self):
        qs = {}
        for k, p in FILES.items():
            qs[k] = {B.norm(B.question_of(json.loads(x))) for x in p.read_text().split("\n") if x}
        self.assertFalse(qs["train"] & qs["val"])
        self.assertFalse(qs["train"] & qs["test"])
        sm = json.loads((DATA / "student_qasc_tf/summary.json").read_text())
        self.assertEqual(sm["audit"]["failures"], 0)
        for k, p in FILES.items():
            self.assertEqual(sm["md5"][f"{p.parent.name}/{p.name}"], hashlib.md5(p.read_bytes()).hexdigest())
        for name in ("f25", "f10"):
            d = DATA / f"student_qasc_tf_{name}"
            s = json.loads((d / "summary.json").read_text())
            self.assertEqual(set(s), {"source", "fraction", "stratified_by", "seed", "n_train", "train_md5"})
            self.assertEqual((d / "val.jsonl").read_bytes(), FILES["val"].read_bytes())
            self.assertEqual(s["train_md5"], hashlib.md5((d / "train.jsonl").read_bytes()).hexdigest())


@unittest.skipUnless(HAVE_TORCH and FILES["train"].exists(), "needs torch and the qasc_tf data")
class TestViews(unittest.TestCase):
    def test_gfpq_and_placebo_views(self):
        from cotdistill.student import PlaceboBank, build_full_tree
        its = read(FILES["train"], 300)
        bank = PlaceboBank(its)
        picks = [next(it for it in its if it["group"] == g) for g in ("qasc_tf/n2", "qasc_tf/n1", "qasc_tf/n0")]
        for it in picks:
            ctx = B.question_of(it)
            for kw in ({}, {"target_seed": 0, "target_kind": "placebo", "placebo_bank": bank}):
                ex = build_full_tree(it, mode="plain", rng=random.Random(0), cap=4, view="statements+question",
                                     sources={QASC_TF}, **kw)
                self.assertEqual([e.kind for e in ex], ["final"] + ["subq"] * len(it["subqs"]))
                for e in ex[1:]:
                    self.assertTrue(e.text.startswith(f"Question: {ctx}\n\nIntermediate question: "))
                    self.assertNotIn("\nA) " + it["options"][0], e.text)      # no final options in a node view
                if kw:
                    self.assertTrue(all("Is it true that" not in e.text for e in ex[1:]))


if __name__ == "__main__":
    unittest.main()
