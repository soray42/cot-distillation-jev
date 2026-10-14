"""EntailmentBank Tree Forcing data (scripts/build_entailmentbank_tf.py, source entailmentbank_tf), its treeplan branch:
- treeplan: problem_no_options (GFPQ) is the question line of every entailmentbank_tf prompt, statement_block refuses
  every item (the question is the whole problem), shape refusals; other sources' outcomes equal the frozen treeplan;
- builder units on a synthetic EntailmentBank record: the tree (dangling conclusion, heights), finals from ARC fields,
  the drops (root, gold text, gold recall, gold pointer, question restated), Yes / No twins, edges, depths, the audit;
  the No edits (antonym, swap, reverse, negate) and their guards;
- the built files (skipped when absent): schema, per-item balance, no gold leak and the audit on a handful of items,
  splits disjoint by question, md5s as summary.json, the f25 subset;
- with torch: placebo_text is the question and every node of the synthetic item gets a placebo question.
CPU only, stdlib (+ torch for the last part). Data-backed tests read data/ of this checkout, or COTD_DATA_DIR."""
import collections
import hashlib
import importlib.util
import json
import os
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import build_entailmentbank_tf as B  # noqa: E402
from cotdistill import treeplan  # noqa: E402
from cotdistill.teacher import leaks_answer  # noqa: E402
from cotdistill.treeplan import ENTAILMENTBANK_TF, problem_no_options, statement_block  # noqa: E402

DATA = Path(os.environ.get("COTD_DATA_DIR") or ROOT / "data")
FILES = {"train": DATA / "student_entailmentbank_tf/train.jsonl", "val": DATA / "student_entailmentbank_tf/val.jsonl",
         "test": DATA / "eval/entailmentbank_tf_test.jsonl"}
GSM8K_TF_FIELDS = {"item_id", "source", "group", "type", "prompt", "labels", "gold_label", "teacher", "options",
                   "option_kinds", "recipe", "depth", "subqs", "random_subqs", "rationale", "augment", "has_tree",
                   "tree_fix", "row_kind", "problem_id", "node_status", "licence_class", "licence", "label_basis",
                   "node_basis", "n_steps", "tree_source"}
SUBQ_FIELDS = {"id", "question", "opposite", "truth", "p_cot", "p_fresh", "type", "depends_on", "initial", "status",
               "node_depth"}
try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# A synthetic EntailmentBank task_1 record (the shape of the v3 release) and its ARC question.
RECORD = {
    "id": "SYN_1",
    "question": "Which property of a mineral can be determined just by looking at it?",
    "answer": "luster",
    "hypothesis": "luster of a mineral can be determined by looking at the mineral",
    "proof": ("sent2 & sent3 -> int1: looking at a mineral can be used to observe the mineral; "
              "int1 & sent4 -> int2: observing a mineral can be used to determine its properties; "
              "sent5 & sent6 -> int3: a rock is a kind of nonliving thing; "
              "int2 & sent1 & sent7 -> hypothesis; "),
    "depth_of_proof": 2, "length_of_proof": 4,
    "meta": {"hypothesis_id": "int4",
             "triples": {"sent1": "luster is a property of a mineral",
                         "sent2": "looking at an object is a kind of observing",
                         "sent3": "a mineral is a kind of object",
                         "sent4": "observing can be used to determine the properties of an object",
                         "sent5": "a rock is a kind of object",
                         "sent6": "an object is a kind of nonliving thing",
                         "sent7": "a property of a mineral that is seen by looking is determined just by looking"},
             "intermediate_conclusions": {"int1": "looking at a mineral can be used to observe the mineral",
                                          "int2": "observing a mineral can be used to determine its properties",
                                          "int3": "a rock is a kind of nonliving thing",
                                          "int4": "luster of a mineral can be determined by looking at the mineral"}}}
ARC = {"SYN_1": {"file": "ARC-Easy-train", "question": RECORD["question"],
                 "texts": ["mass", "luster", "weight", "hardness"], "keys": ["1", "2", "3", "4"], "key": "2"}}
DONORS = [("d1", "a leaf is a kind of plant part"), ("d2", "a planet is a kind of celestial object"),
          ("d3", "the sun is a kind of star"), ("d4", "a magnet is a kind of tool for attracting metal")]


def build_synthetic(seed=0):
    st, fails = collections.Counter(), []
    rows = B.finals([RECORD], ARC, "train", seed, st, fails)
    assert not fails and len(rows) == 1, fails
    item, r, tree = rows[0]
    known = B.Known([s for _, s in DONORS] + list(RECORD["meta"]["triples"].values()))
    nodes = B.build_nodes(tree, item, B.clean(r["hypothesis"]), set(), random.Random(f"{seed}/nodes/SYN_1"),
                          B.Swapper(DONORS), known, st)
    item = dict(item, subqs=nodes, has_tree=bool(nodes), node_status="ok" if nodes else "no_nodes")
    return item, tree, st


def read(path, n=None):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if n and len(out) >= n:
                    break
    return out


class TestTreeplanBranch(unittest.TestCase):
    def item(self, prompt, src=ENTAILMENTBANK_TF):
        return {"item_id": "eb-x", "source": src, "prompt": prompt}

    def test_gfpq_and_no_statement_block(self):
        it = self.item("Question: Which gas do plants absorb?\nOptions:\nA) oxygen\nB) carbon dioxide\nC) neon")
        self.assertEqual(problem_no_options(it), "Question: Which gas do plants absorb?")
        with self.assertRaises(ValueError):
            statement_block(it)

    def test_shape_refusals(self):
        for p in ("Question: a\nb?\nOptions:\nA) x\nB) y", "Which gas?\nOptions:\nA) x", "Question: Which gas?",
                  "Question: Which gas?\nOptions:\nA) x\nOptions:\nB) y", "Question: \nOptions:\nA) x",
                  "Question: Question: Which gas?\nOptions:\nA) x"):
            for fn in (problem_no_options, statement_block):
                with self.assertRaises(ValueError, msg=repr(p)):
                    fn(self.item(p))

    def test_other_sources_unchanged(self):
        """statement_block / problem_no_options give the frozen treeplan's outcome on the other sources' fixtures."""
        from test_statement_block import KK, NEW, NUMGEN, new_item
        spec = importlib.util.spec_from_file_location("treeplan_9f03003", ROOT / "tests/fixtures/treeplan_9f03003.py")
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        rows = [KK, NUMGEN] + [new_item(s, *v) for s, v in NEW.items()]
        rows += [{"item_id": "a", "source": "arc_challenge", "prompt": "Question: Which gas?\nOptions:\nA) x\nB) y"},
                 {"item_id": "u", "source": "unknown", "prompt": "x\n\nQuestion: y?\nOptions:\nA) 1"}]

        def outcome(fn, r):
            try:
                return ("ok", fn(r))
            except ValueError:
                return ("raise", None)
        for r in rows:
            for fn in ("statement_block", "problem_no_options"):
                self.assertEqual(outcome(getattr(treeplan, fn), r), outcome(getattr(old, fn), r), (r["source"], fn))


class TestBuilderUnits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.item, cls.tree, cls.st = build_synthetic()

    def test_tree(self):
        t = self.tree
        self.assertEqual(t["root"], "int4")
        self.assertEqual(t["dangling"], ["int3", "sent5", "sent6"])          # int3 is used by no step
        self.assertEqual(t["order"], ["sent1", "sent2", "sent3", "sent4", "sent7", "int1", "int2"])
        self.assertEqual((t["height"]["int1"], t["height"]["int2"], t["height"]["int4"]), (1, 2, 3))
        self.assertTrue(t["eb_depth_differs"])                               # the record says 2; the edges give 3

    def test_final(self):
        it = self.item
        self.assertEqual(it["labels"], ["A", "B", "C", "D"])
        self.assertEqual(B.gold_text(it), "luster")
        self.assertEqual(it["prompt"].split("\n")[0], "Question: " + RECORD["question"])
        self.assertEqual(it["teacher"][it["gold_label"]], 1.0)
        self.assertEqual((it["depth"], it["n_steps"], it["group"]), (3, 3, "entailmentbank_tf/d3"))
        self.assertTrue(GSM8K_TF_FIELDS <= set(it))

    def test_drops_and_nodes(self):
        st, nodes = self.st, self.item["subqs"]
        self.assertEqual(st["dropped_root"], 1)
        self.assertEqual(st["dropped_gold_text"], 1)                         # sent1 names the gold option
        sids = {n["sent_id"] for n in nodes}
        self.assertNotIn("sent1", sids)
        self.assertNotIn("int4", sids)
        self.assertNotIn("sent5", sids)                                      # dangling
        self.assertNotIn("sent7", sids)                                      # restates the question
        self.assertEqual(st["dropped_restates_question"], 1)
        self.assertTrue(sids)
        self.assertEqual(sum(n["truth"] for n in nodes) * 2, len(nodes))
        self.assertEqual(B.audit_item(self.item), [])
        by = {n["id"]: n for n in nodes}
        if "int1v" in by:
            self.assertEqual(by["int1v"]["node_depth"], 1)
            self.assertEqual(set(by["int1v"]["depends_on"]), {f"{s}v" for s in ("sent2", "sent3") if f"{s}v" in by})
            self.assertEqual(by["int1n"]["depends_on"], by["int1v"]["depends_on"])
        for n in nodes:
            self.assertTrue(SUBQ_FIELDS <= set(n))
            self.assertEqual(n["question"], f"Is it true that {n['claim']}?")
            self.assertEqual(n["truth"], n["claim"] == n["sentence"])

    def test_gold_leak_rules(self):
        it = dict(self.item, options=["the moon orbiting the earth", "a comet", "the sun rising", "wind"],
                  gold_label="A", labels=["A", "B", "C", "D"],
                  prompt="Question: What causes the phases of the moon?\nOptions:\nA) the moon orbiting the earth\n"
                         "B) a comet\nC) the sun rising\nD) wind")
        self.assertEqual(B.gold_leak(it, "the moon orbiting the earth causes phases"), "gold_text")
        self.assertEqual(B.gold_leak(it, "the moon orbits around the earth"), "gold_recall")
        self.assertEqual(B.gold_leak(it, "the moon is a kind of satellite"), None)              # 1 of 3 stems
        it2 = dict(it, options=["breaks down dead plants and animals", "makes food", "hunts animals", "migrates"])
        self.assertEqual(B.gold_leak(it2, "a decomposer breaks down dead organisms"), "gold_pointer")
        self.assertEqual(B.gold_leak(it, "what causes the phases of the moon is known"), "restates_question")

    def test_edits(self):
        v = [x[0] for x in B.antonym_edits("as temperature increases, the rate of evaporation increases")]
        self.assertIn("as temperature decreases, the rate of evaporation increases", v)
        self.assertIn("a black object reflects light", [x[0] for x in B.antonym_edits("a black object absorbs light")])
        self.assertEqual(B.antonym_edits("a herbivore is a kind of animal"), [])          # subject of a taxonomy
        self.assertEqual(B.antonym_edits("a conductor is the opposite of an insulator"), [])
        self.assertIn("an insulator", B.antonym_edits("copper is a conductor")[0][0])         # article fixed
        self.assertEqual(B.negate_edit("a magnet can attract iron")[0], "a magnet cannot attract iron")
        self.assertEqual(B.negate_edit("plants do not move")[0], "plants move")
        self.assertEqual(B.negate_edit("a plant requires sunlight")[0], "a plant does not require sunlight")
        self.assertIsNone(B.negate_edit("some animals are herbivores"))
        self.assertIsNone(B.negate_edit("if it rains then the ground is wet"))
        known = B.Known(["the earth orbits the sun"])
        self.assertEqual(B.reverse_edit("the moon orbits the earth", known)[0], "the earth orbits the moon")
        self.assertEqual(B.reverse_edit("predators eat prey", known)[0], "prey eats predators")
        self.assertIsNone(B.reverse_edit("matter is made of particles", known))              # not reversible
        self.assertIsNone(B.reverse_edit("the sun orbits the earth", known))                 # known triple
        sw = B.Swapper([("d1", "a whale is a kind of mammal"), ("d2", "a pebble is a kind of rock"),
                        ("d3", "a cloud is a kind of weather phenomenon")])
        kn = B.Known(["a dog is a kind of mammal", "a pebble is a kind of rock"])
        objs = {x[2] for x in sw.edits("a dog is a kind of animal", set(), kn)}
        self.assertNotIn("mammal", objs)                                       # a known ancestor of "dog"
        self.assertIn("rock", objs)
        self.assertEqual(sw.edits("a dog is usually a kind of animal", set(), kn), [])          # hedged
        self.assertEqual(sw.edits("a dog eats meat", set(), kn), [])                            # verb relation

    def test_seeded(self):
        a, _, _ = build_synthetic(seed=0)
        b, _, _ = build_synthetic(seed=0)
        self.assertEqual(a, b)


@unittest.skipUnless(all(p.exists() for p in FILES.values()), "built files absent")
class TestBuiltFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.head = {k: read(p, 25) for k, p in FILES.items()}

    def test_handful(self):
        for split, items in self.head.items():
            for it in items:
                self.assertTrue(GSM8K_TF_FIELDS <= set(it), it["item_id"])
                self.assertEqual(it["source"], ENTAILMENTBANK_TF)
                self.assertEqual(B.audit_item(it), [], it["item_id"])
                self.assertEqual(sum(n["truth"] for n in it["subqs"]) * 2, len(it["subqs"]), it["item_id"])
                gold = B.gold_text(it)
                for n in it["subqs"]:
                    self.assertTrue(SUBQ_FIELDS <= set(n))
                    self.assertFalse(B.phrase_in(gold, n["claim"]), (it["item_id"], n["id"]))
                    self.assertLess(B.recall(gold, n["claim"]), B.GOLD_RECALL_MAX)
                    self.assertIsNone(B.gold_leak(it, n["claim"]))
                    self.assertFalse(leaks_answer(n["question"], it))

    def test_splits_and_md5(self):
        qs = {k: {B.norm(B.stem_of(it["prompt"])) for it in read(p)} for k, p in FILES.items()}
        self.assertFalse(qs["train"] & qs["val"] or qs["train"] & qs["test"] or qs["val"] & qs["test"])
        summary = json.loads((FILES["train"].parent / "summary.json").read_text())
        for k, p in FILES.items():
            self.assertEqual(hashlib.md5(p.read_bytes()).hexdigest(), summary["md5"][str(p.relative_to(DATA.parent))])
        yes = [n["truth"] for it in read(FILES["train"]) for n in it["subqs"]]
        self.assertEqual(sum(yes) * 2, len(yes))

    def test_f25(self):
        d = FILES["train"].parent.parent / "student_entailmentbank_tf_f25"
        if not d.exists():
            self.skipTest("f25 absent")
        s = json.loads((d / "summary.json").read_text())
        sub, full = read(d / "train.jsonl"), read(FILES["train"])
        self.assertEqual(len(sub), s["n_train"])
        pos = {it["item_id"]: i for i, it in enumerate(full)}
        self.assertEqual([pos[it["item_id"]] for it in sub], sorted(pos[it["item_id"]] for it in sub))
        self.assertEqual((d / "val.jsonl").read_bytes(), FILES["val"].read_bytes())
        g, gs = collections.Counter(it["group"] for it in full), collections.Counter(it["group"] for it in sub)
        for k in g:
            self.assertEqual(gs[k], round(g[k] * s["fraction"]), k)


@unittest.skipUnless(HAVE_TORCH, "torch absent")
class TestPlacebo(unittest.TestCase):
    def test_placebo(self):
        from cotdistill.student import PlaceboBank, placebo_questions, placebo_text
        item, _, _ = build_synthetic()
        other = dict(item, item_id="eb-other", prompt="Question: Why do leaves change color in the fall?\nOptions:\n"
                                                         "A) a\nB) b\nC) c\nD) d")
        self.assertEqual(placebo_text(item), RECORD["question"])
        q = placebo_questions(item, 0, PlaceboBank([item, other]))
        self.assertEqual(set(q), {n["id"] for n in item["subqs"]})
        self.assertEqual([q[n["id"]][1] for n in item["subqs"]], [n["truth"] for n in item["subqs"]])


if __name__ == "__main__":
    unittest.main()
