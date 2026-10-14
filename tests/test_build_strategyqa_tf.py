"""StrategyQA Tree Forcing data (scripts/build_strategyqa_tf.py, source strategyqa_tf) and its treeplan branch:
- builder units on synthetic StrategyQA records: answers read off the facts (year, number, entity), "#k" resolved into
  a parent edge, drops (last step, yes/no step, hedged or several values, unresolved reference), balanced Yes / No
  nodes, the item audit (and that it catches a corrupted item);
- treeplan: problem_no_options (GFPQ) is the question line, statement_block refuses every item, shape refusals;
- the built files (skipped when absent): schema, every item 50% Yes with a twin per step, no leak (audit_item on
  every item), disjoint splits, the f25 subset (and its recipe against data/student_gsm8k_tf_f25), summary md5s.
CPU only, stdlib only (placebo_text needs torch: skipped without it). Data-backed tests read data/ of this checkout,
or COTD_DATA_DIR when set."""
import collections
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

import build_strategyqa_tf as B  # noqa: E402
from cotdistill.treeplan import STRATEGYQA_TF, problem_no_options, statement_block  # noqa: E402

DATA = Path(os.environ.get("COTD_DATA_DIR") or ROOT / "data")
try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

ITEM_KEYS = {"item_id", "source", "group", "type", "prompt", "labels", "gold_label", "teacher", "options", "depth",
             "subqs", "random_subqs", "rationale", "augment", "has_tree", "tree_fix", "row_kind", "problem_id", "answer",
             "term", "licence_class", "licence", "label_basis", "node_basis", "n_steps", "n_facts", "tree_source"}
NODE_KEYS = {"id", "question", "opposite", "truth", "p_cot", "p_fresh", "type", "depends_on", "initial", "status",
             "probe", "probe_text", "step", "step_text", "resolved", "kind", "value", "value_text", "unit",
             "entity_class", "fact", "fact_index", "gold_ancestor", "node_depth"}

SHAKESPEARE = {"qid": "q-shakespeare", "term": "Daily Mirror", "answer": False,
               "question": "Did William Shakespeare read the Daily Mirror?",
               "decomposition": ["When did William Shakespeare die?", "When was the Daily Mirror founded?",
                                 "Is #2 before #1?"],
               "facts": ["The Daily Mirror is a British tabloid founded in 1903.", "William Shakespeare died in 1616."],
               "evidence": [[[["Daily Mirror-1"]], [["Daily Mirror-2"]], ["operation"]]]}
KHAN = {"qid": "q-khan", "term": "Genghis Khan", "answer": True,
        "question": "Was the founder of the Mongol Empire born before the year 1200?",
        "decomposition": ["Who was the founder of the Mongol Empire?", "When was #1 born?", "Is #2 before 1200?"],
        "facts": ["Genghis Khan was the founder of the Mongol Empire.", "Genghis Khan was born in 1162."],
        "evidence": []}
SPICE = {"qid": "q-spice", "term": "Spice Girls", "answer": True,
         "question": "Could the Spice Girls fit in a Tata Hexa?",
         "decomposition": ["How many members did the Spice Girls have?", "How many people can the Tata Hexa seat?",
                           "Do the Spice Girls usually travel together?", "Is #2 greater than or equal to #1?"],
         "facts": ["The Spice Girls had 5 members.", "The Tata Hexa is a car with 6 and 7 seat capacities.",
                   "The Spice Girls toured together in 1998."],
         "evidence": []}
BOMBER = {"qid": "q-bomber", "term": "B-52", "answer": False,
          "question": "Could a B-52 outrun a cheetah's top speed tenfold?",
          "decomposition": ["How fast can a B-52 travel?", "What is #1 times ten?", "Is #2 above 700 MPH?"],
          "facts": ["A B-52 can travel fast, up to 650 MPH."], "evidence": []}
# other items of the "split": the pools the wrong answers come from
OTHERS = [{"qid": f"q-other{i}", "term": "x", "answer": True, "question": f"Is it about {n}?",
           "decomposition": [f"Who was the leader of the {g} band?", f"When was the {g} band formed?", "Is #1 old?"],
           "facts": [f"The leader of the {g} band was {n}.", f"The {g} band was formed in {y}."], "evidence": []}
          for i, (n, g, y) in enumerate([("Marshall Applewhite", "Heaven", 1974), ("Trent Reznor", "Nails", 1988),
                                          ("Damon Albarn", "Gorillaz", 1998), ("Dave Mustaine", "Megadeth", 1983)])]


def build_items(raws, seed=0):
    st = collections.Counter()
    answered = {r["qid"]: B.step_answers(r, st) for r in raws}
    pools = B.answer_pools(answered)
    out = {}
    for r in raws:
        it = B.final_item(r)
        it["subqs"] = B.tf_nodes(r, answered[r["qid"]], pools, random.Random(f"{seed}/{r['qid']}"), st)
        it["has_tree"] = bool(it["subqs"])
        out[r["qid"]] = it
    return out, st


class TestBuilderUnits(unittest.TestCase):
    def setUp(self):
        self.items, self.st = build_items([SHAKESPEARE, KHAN, SPICE, BOMBER] + OTHERS)

    def test_final(self):
        it = self.items["q-shakespeare"]
        self.assertEqual(it["prompt"], "Question: Did William Shakespeare read the Daily Mirror?\nOptions:\nA) Yes\nB) No")
        self.assertEqual((it["labels"], it["options"], it["gold_label"]), (["A", "B"], ["Yes", "No"], "B"))
        self.assertEqual(it["teacher"], {"A": 0.0, "B": 1.0})
        self.assertEqual((it["source"], it["group"], it["n_steps"], it["depth"]), (STRATEGYQA_TF, "strategyqa_tf/s3", 3, 3))
        for fact in SHAKESPEARE["facts"]:
            self.assertNotIn(fact, it["prompt"])

    def test_year_nodes(self):
        nodes = {n["id"]: n for n in self.items["q-shakespeare"]["subqs"]}
        self.assertEqual(sorted(nodes), ["s1n", "s1v", "s2n", "s2v"])             # the last step: no node
        self.assertEqual(nodes["s1v"]["question"], "When did William Shakespeare die? Is it 1616?")
        self.assertEqual(nodes["s2v"]["question"], "When was the Daily Mirror founded? Is it 1903?")
        for k, y in (("s1", 1616), ("s2", 1903)):
            no = nodes[k + "n"]
            self.assertFalse(no["truth"])
            self.assertNotEqual(no["probe"], y)
            self.assertTrue(5 <= abs(no["probe"] - y) <= 50 and B.YEAR_MIN <= no["probe"] <= B.YEAR_MAX)
            self.assertEqual(no["question"], f"{nodes[k + 'v']['resolved']} Is it {no['probe_text']}?")

    def test_entity_and_reference(self):
        nodes = {n["id"]: n for n in self.items["q-khan"]["subqs"]}
        self.assertEqual(nodes["s1v"]["value_text"], "Genghis Khan")
        self.assertEqual(nodes["s1v"]["entity_class"], "person")
        no = nodes["s1n"]["probe_text"]
        self.assertIn(no, {"Marshall Applewhite", "Trent Reznor", "Damon Albarn", "Dave Mustaine"})
        self.assertEqual(nodes["s2v"]["resolved"], "When was Genghis Khan born?")         # "#1" resolved
        self.assertEqual(nodes["s2v"]["question"], "When was Genghis Khan born? Is it 1162?")
        self.assertEqual((nodes["s2v"]["depends_on"], nodes["s2v"]["node_depth"]), (["s1v"], 1))
        self.assertEqual((nodes["s1v"]["depends_on"], nodes["s1v"]["node_depth"]), ([], 0))

    def test_drops(self):
        spice = {n["id"]: n for n in self.items["q-spice"]["subqs"]}
        self.assertEqual(sorted(spice), ["s1n", "s1v"])        # step 2: "6 and 7"; step 3: yes/no; step 4: last
        self.assertEqual(spice["s1v"]["question"], "How many members did the Spice Girls have? Is it 5 members?")
        self.assertRegex(spice["s1n"]["probe_text"], r"^\d+ members$")
        self.assertEqual(self.items["q-bomber"]["subqs"], [])  # "up to 650": hedged; step 2 reads it: unresolved
        self.assertFalse(self.items["q-bomber"]["has_tree"])
        for key in ("dropped_last_step", "dropped_yesno_step", "dropped_hedged_value", "dropped_unresolved_ref"):
            self.assertGreater(self.st[key], 0, key)
        self.assertGreater(self.st["dropped_several_values_in_fact"] + self.st["dropped_value_is_a_range"], 0)

    def test_balance_and_audit(self):
        for it in self.items.values():
            nodes = it["subqs"]
            self.assertEqual(sum(n["truth"] for n in nodes) * 2, len(nodes))
            self.assertEqual(B.audit_item(it), [], it["item_id"])
            for n in nodes:
                self.assertEqual(set(n) - {"wrong"}, NODE_KEYS)
                self.assertLess(n["step"], it["n_steps"])                  # never the final operation
                self.assertNotIn("#", n["question"])

    def test_audit_catches(self):
        it = json.loads(json.dumps(self.items["q-shakespeare"]))
        it["subqs"][1]["truth"] = True                                     # the No node marked Yes
        self.assertTrue(B.audit_item(it))
        it = json.loads(json.dumps(self.items["q-khan"]))
        it["subqs"][2]["question"] = "When was #1 born? Is it 1162?"
        self.assertTrue(B.audit_item(it))
        it = json.loads(json.dumps(self.items["q-shakespeare"]))
        it["prompt"] = it["prompt"].replace("?\nOptions", " " + SHAKESPEARE["facts"][1] + "?\nOptions")
        self.assertTrue(B.audit_item(it))

    def test_numbers(self):
        a, why = B.numeric_answer("How many kids did Genghis Khan have?", "Genghis Khan had sixteen children.", "number")
        self.assertEqual((a["mantissa"], why), (16.0, ""))
        a, why = B.numeric_answer("What is the average temperature of Mercury at night?",
                                  "Mercury can reach temperatures of −280 °F at night.", "number")
        self.assertEqual((a, why), (None, "fact_qualifies_value"))       # "can reach": a maximum, not the value
        a, why = B.numeric_answer("How fast can a B-52 travel?", "A B-52 can travel fast, up to 650 MPH.", "number")
        self.assertEqual((a, why), (None, "hedged_value"))
        a, why = B.numeric_answer("How long does it take for chlorine to dissolve?",
                                  "Chlorine will dissolve in 4 and a half days.", "number")
        self.assertEqual((a, why), (None, "several_values_in_fact"))
        a, why = B.numeric_answer("When did Saddam Hussein die?", "Saddam Hussein died on December 30th, 2006.", "year")
        self.assertEqual(a["answer"], "2006")
        d = {"mantissa": 86.2, "decimals": 2, "commas": False, "scale": "", "sign": "", "pre": "", "unit": "%"}
        self.assertFalse(B.ok_wrong_number(215.5, d, set()))               # a share above 100%


class TestTreeplanBranch(unittest.TestCase):
    ITEM = {"item_id": "strategyqa-x", "source": STRATEGYQA_TF,
            "prompt": "Question: Did William Shakespeare read the Daily Mirror?\nOptions:\nA) Yes\nB) No"}

    def test_gfpq_and_no_statement_block(self):
        self.assertEqual(problem_no_options(self.ITEM), "Question: Did William Shakespeare read the Daily Mirror?")
        with self.assertRaises(ValueError):
            statement_block(self.ITEM)

    def test_refusals(self):
        for p in ("Question: Did he?\nOptions:\nA) No\nB) Yes", "Question: Did he?", "Did he?\nOptions:\nA) Yes\nB) No",
                  "Question: Did he?\nWhy?\nOptions:\nA) Yes\nB) No", "Question:  \nOptions:\nA) Yes\nB) No"):
            with self.assertRaises(ValueError, msg=p):
                problem_no_options(dict(self.ITEM, prompt=p))

    @unittest.skipUnless(HAVE_TORCH, "cotdistill.student needs torch")
    def test_placebo_text(self):
        from cotdistill.student import placebo_text
        self.assertEqual(placebo_text(self.ITEM), "Did William Shakespeare read the Daily Mirror?")


# ---- the built files ---------------------------------------------------------------------------------------------
FILES = {"train": DATA / "student_strategyqa_tf/train.jsonl", "val": DATA / "student_strategyqa_tf/val.jsonl",
         "test": DATA / "eval/strategyqa_tf_test.jsonl"}


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@unittest.skipUnless(all(p.exists() for p in FILES.values()), "built StrategyQA files absent")
class TestBuiltFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = {k: read(p) for k, p in FILES.items()}

    def test_schema_balance_and_audit(self):
        for name, items in self.items.items():
            for it in items:
                self.assertEqual(set(it), ITEM_KEYS, it["item_id"])
                self.assertEqual(problem_no_options(it), it["prompt"].partition("\nOptions:")[0])
                nodes = it["subqs"]
                self.assertEqual(sum(n["truth"] for n in nodes) * 2, len(nodes), it["item_id"])
                ids = {n["id"] for n in nodes}
                self.assertTrue(all(n["id"][:-1] + ("n" if n["truth"] else "v") in ids for n in nodes))
                self.assertEqual(B.audit_item(it), [], it["item_id"])      # incl. no restated final, no fact shown

    def test_splits(self):
        qs = {k: {B.norm(B.question_of(it)) for it in v} for k, v in self.items.items()}
        self.assertFalse(qs["train"] & qs["val"] or qs["train"] & qs["test"] or qs["val"] & qs["test"])
        n = {k: len(v) for k, v in self.items.items()}
        total = sum(n.values())
        self.assertAlmostEqual(n["test"] / total, 0.2, delta=0.01)
        self.assertAlmostEqual(n["val"] / total, 0.1, delta=0.01)
        for items in self.items.values():
            self.assertGreater(sum(it["has_tree"] for it in items), 0)

    def test_summary_md5(self):
        s = json.loads((DATA / "student_strategyqa_tf/summary.json").read_text())
        self.assertEqual(s["audit"]["failures"], 0)
        for key, m in s["md5"].items():
            rel = key[len("data/"):] if key.startswith("data/") else key
            self.assertEqual(hashlib.md5((DATA / rel).read_bytes()).hexdigest(), m, key)

    def test_f25_subset(self):
        sub = DATA / "student_strategyqa_tf_f25"
        if not (sub / "train.jsonl").exists():
            self.skipTest("no f25 subset")
        lines = [x for x in FILES["train"].read_text().split("\n") if x]
        sl = [x for x in (sub / "train.jsonl").read_text().split("\n") if x]
        pos = [lines.index(x) for x in sl]
        self.assertEqual(pos, sorted(pos))                                  # byte-identical lines, file order
        meta = json.loads((sub / "summary.json").read_text())
        self.assertEqual(pos, B.subset(lines, self.items["train"], 0.25, meta["seed"]))
        groups = collections.Counter(it["group"] for it in self.items["train"])
        self.assertEqual(len(sl), sum(max(1, round(v * 0.25)) for v in groups.values()))
        self.assertEqual((sub / "val.jsonl").read_bytes(), FILES["val"].read_bytes())

    def test_subset_recipe_matches_gsm8k_f25(self):
        src, sub = DATA / "student_gsm8k_tf/train.jsonl", DATA / "student_gsm8k_tf_f25/train.jsonl"
        if not (src.exists() and sub.exists()):
            self.skipTest("no gsm8k_tf data")
        lines = [x for x in src.read_text().split("\n") if x]
        keep = B.subset(lines, [json.loads(x) for x in lines], 0.25, "gsm8k_tf|f25|20261012")
        out = "".join(lines[i] + "\n" for i in keep).encode()
        self.assertEqual(hashlib.md5(out).hexdigest(), hashlib.md5(sub.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
