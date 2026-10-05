"""v7 mixture assembler (scripts/build_v7.py): TODO refusal of the release skeletons, the array form of decontam.py's
13-gram rule, node and label policies, own-eval self-overlap, cross-source exact / near duplicates with priority and
dedup groups, problem-level drops, licence filter of a release build (incl. the audit's proprietary-generator rows),
fill to a row target and its min_ratio refusal, node rows of a "drop" arm in the dedup, the per-family val split and
determinism of the output."""
import copy
import hashlib
import io
import json
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_v7 as b  # noqa: E402
from decontam import grams  # noqa: E402

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec "
         "romeo sierra tango uniform victor whiskey xray yankee zulu").split()


def story(rng: random.Random, n: int = 40) -> str:
    return " ".join(rng.choice(WORDS) + str(rng.randrange(1000)) for _ in range(n))


def final(iid: str, text: str, gold: str = "A", subqs=None, **kw) -> dict:
    r = {"item_id": iid, "source": kw.pop("source", "src"), "prompt": f"Question: {text}\nOptions:\nA) 1\nB) 2",
         "labels": ["A", "B"], "gold_label": gold, "teacher": {"A": float(gold == "A"), "B": float(gold == "B")},
         "subqs": subqs or [], "random_subqs": []}
    r.update(kw)
    return r


def write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


class TestTodo(unittest.TestCase):
    def test_release_skeletons_refuse(self):
        for size in ("2b", "4b"):
            cfg = ROOT / f"configs/v7/v7_release_{size}.json"
            err = io.StringIO()
            with redirect_stderr(err):
                self.assertEqual(b.main([str(cfg), "--check-only"]), 2)
            self.assertIn("TODO", err.getvalue())
            self.assertIn("$.total_rows", err.getvalue())

    def test_pilots_validate(self):
        for name in ("v7_pilot_math_MS", "v7_pilot_math_MA", "v7_pilot_math_MSgen", "v7_pilot_math_M0",
                     "v7_pilot_br40k"):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(b.main([str(ROOT / f"configs/v7/{name}.json"), "--check-only"]), 0)

    def test_todo_paths(self):
        cfg = {"a": 1, "b": {"c": "TODO: x", "d": [0, "ok", "TODO"]}, "TODO_key": 2}
        self.assertEqual(sorted(b.todo_paths(cfg)), ["$.TODO_key (key)", "$.b.c", "$.b.d[2]"])


class TestGrams13(unittest.TestCase):
    def test_matches_decontam_rule(self):
        rng = random.Random(0)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ev = [{"prompt": story(rng)} for _ in range(30)]
            boiler = "the quick brown fox jumps over the lazy dog near the old river bank today"
            for e in ev:                                   # a 16-word phrase in all 30 items: boilerplate
                e["prompt"] = boiler + " " + e["prompt"]
            write(d / "evA.jsonl", ev[:15])
            write(d / "evB.jsonl", ev[15:])
            g = b.Grams13(str(d / "ev*.jsonl"))
            self.assertEqual(g.hits([boiler + " unrelated words here"]), set())
            hit = " ".join(ev[20]["prompt"].split()[20:40])
            self.assertEqual(g.hits(["prefix " + hit]), {"evB"})
            self.assertEqual(g.hits(["short text"]), set())
            # agrees with decontam.grams item by item
            for e in ev:
                gi = grams(e["prompt"])
                self.assertTrue(gi)


class TestAssemble(unittest.TestCase):
    def build(self, cfg: dict, d: Path, evals: str, limit: int = 0, **kw) -> dict:
        args = b.argparse.Namespace(config="x", out=str(d / "out" / cfg["name"]), tmp=str(d / "tmp"), evals=evals,
                                    limit=limit, check_only=False, **kw)
        with redirect_stdout(io.StringIO()):
            return b.Assembler(cfg, args).run()

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        d = self.d = Path(self.td.name)
        rng = random.Random(1)
        self.texts = [story(rng) for _ in range(60)]
        sq = [{"id": "n1", "question": "Is alpha more than 3?", "truth": True, "p_cot": 1.0, "depends_on": []}]
        a_rows = [final(f"a{i}", self.texts[i], subqs=sq, family=f"fam{i % 3}") for i in range(30)]
        a_nodes = [{"item_id": f"a{i}-n{j}", "node_of": f"a{i}", "source": "src_node", "prompt": f"node {j} of {i}",
                    "labels": ["A", "B"], "gold_label": "B", "teacher": {"A": 0.0, "B": 1.0}, "weight": 1 / 3}
                   for i in range(30) for j in range(3)]
        b_rows = [final(f"b{i}", self.texts[30 + i], family="fb", licence_class="NC") for i in range(20)]
        b_rows.append(final("b_dup_exact", self.texts[0], family="fb", licence_class="NC"))
        near = self.texts[1].split()
        near[-1] = "zzz999"                                   # one word of 40 changed: Jaccard ~ .9
        b_rows.append(final("b_dup_near", " ".join(near), family="fb", licence_class="NC"))
        b_rows.append(final("b_soft", self.texts[55], family="fb", licence_class="NC",
                            teacher={"A": 0.7, "B": 0.3}))
        self.contam = self.texts[50]
        b_rows.append(final("b_contam", self.contam, family="fb", licence_class="NC"))
        write(d / "a.jsonl", a_rows)
        write(d / "a_nodes.jsonl", a_nodes)
        write(d / "b.jsonl", b_rows)
        (d / "ev").mkdir()
        write(d / "ev" / "bench.jsonl", [{"prompt": "Context: " + self.contam}])
        write(d / "ev" / "own_a.jsonl", [{"prompt": "x " + self.texts[2]}])
        self.evals = str(d / "ev" / "*.jsonl")

    def tearDown(self):
        self.td.cleanup()

    def cfg(self, **kw) -> dict:
        d = self.d
        c = {"name": "t", "tier": "lab", "seed": 0, "out": "unused", "val": {"mode": "sources"},
             "sources": [
                 {"name": "A", "slice": "M", "train": str(d / "a.jsonl"), "node_rows": str(d / "a_nodes.jsonl"),
                  "node_link": "node_of", "select": {"problems": "all", "rank": "file"}, "nodes": "keep",
                  "max_node_rows_per_problem": 2, "label": "program", "licence": {"class": "PERM"},
                  "gates": ["13gram"], "own_eval": ["own_"], "math": True},
                 {"name": "B", "slice": "B", "train": str(d / "b.jsonl"), "select": {"problems": "all", "rank": "hash"},
                  "nodes": "drop", "label": "as_is", "licence": {"from": "row"}, "gates": ["13gram"]}]}
        c.update(kw)
        return c

    def rows(self, name: str, split: str = "train") -> list[dict]:
        return [json.loads(l) for l in (self.d / "out" / name / f"{split}.jsonl").read_text().splitlines()]

    def test_policies_dedup_decontam(self):
        s = self.build(self.cfg(), self.d, self.evals)
        rows = self.rows("t")
        ids = {r["item_id"] for r in rows}
        self.assertNotIn("b_contam", ids)                            # 13-gram hit on bench
        self.assertNotIn("b_dup_exact", ids)                         # exact duplicate of A's a0
        self.assertNotIn("b_dup_near", ids)                          # near duplicate of A's a1
        self.assertIn("a0", ids)
        self.assertIn("a1", ids)
        self.assertIn("a2", ids)                                     # own-eval overlap: kept, flagged
        a2 = next(r for r in rows if r["item_id"] == "a2")
        self.assertIn("own-eval template overlap", a2["decontam"])
        self.assertEqual(s["per_source"]["A"]["own_eval_self_overlap_rows"], 1)
        nodes = [r for r in rows if r.get("node_of")]
        self.assertEqual(len(nodes), 60)                             # <= 2 node rows per problem
        self.assertTrue(all(abs(r["weight"] - .5) < 1e-9 for r in nodes))
        self.assertTrue(all(r["subqs"] for r in rows if r["item_id"].startswith("a") and "node_of" not in r))
        self.assertTrue(all(r["licence_class"] == "NC" and r["lab_only"] for r in rows if r["v7_source"] == "B"))
        soft = next(r for r in rows if r["item_id"] == "b_soft")
        self.assertEqual(soft["label_basis"], "teacher-soft")
        self.assertEqual(s["dedup"]["pairs"], {"B<-A": 2})
        self.assertEqual(s["lab_only_rows"], 21)

    def test_nodes_drop_and_gold_labels(self):
        c = self.cfg()
        c["sources"][0]["nodes"] = "drop"
        c["sources"][1]["label"] = "gold"
        self.build(c, self.d, self.evals)
        rows = self.rows("t")
        self.assertFalse([r for r in rows if r.get("node_of")])
        self.assertTrue(all(r["subqs"] == [] for r in rows))
        soft = next(r for r in rows if r["item_id"] == "b_soft")
        self.assertEqual(soft["teacher"], {"A": 1.0, "B": 0.0})

    def test_drop_arm_keeps_the_same_problems(self):
        # a node row of a1 shares a 13-gram with an evaluation item: the problem goes in both arms
        rows = [json.loads(l) for l in (self.d / "a_nodes.jsonl").read_text().splitlines()]
        for r in rows:
            if r["node_of"] == "a5":
                r["prompt"] = "node of a5 " + self.contam
        write(self.d / "a_nodes.jsonl", rows)
        keep = self.build(self.cfg(name="keep"), self.d, self.evals)
        c = self.cfg(name="drop")
        c["sources"][0]["nodes"] = "drop"
        drop = self.build(c, self.d, self.evals)
        ids = lambda n: {r["item_id"] for r in self.rows(n) if r["v7_source"] == "A" and "node_of" not in r}
        self.assertNotIn("a5", ids("keep"))
        self.assertEqual(ids("keep"), ids("drop"))
        self.assertEqual(keep["per_source"]["A"]["kept_problems"], drop["per_source"]["A"]["kept_problems"])
        self.assertFalse(any("_screen_texts" in r or "_policy_drop" in r for r in self.rows("drop")))

    def test_drop_arm_node_rows_take_part_in_dedup(self):
        # a B final repeats a node row of A (higher priority): it goes in the keep arm and in its drop twin
        c = self.cfg(name="keep")
        c["sources"][0].pop("max_node_rows_per_problem")
        write(self.d / "b2.jsonl", [final("b_node_dup", "", prompt="node 0 of 5", family="fb", licence_class="NC"),
                                    final("b_other", self.texts[40], family="fb", licence_class="NC")])
        c["sources"][1]["train"] = str(self.d / "b2.jsonl")
        keep = self.build(c, self.d, self.evals)
        c = copy.deepcopy(c)
        c["name"] = "drop"
        c["sources"][0]["nodes"] = "drop"
        drop = self.build(c, self.d, self.evals)
        ids = lambda n: {r["item_id"] for r in self.rows(n) if "node_of" not in r}
        self.assertNotIn("b_node_dup", ids("keep"))
        self.assertEqual(ids("keep"), ids("drop"))
        self.assertIn("b_other", ids("drop"))
        self.assertEqual(drop["dedup"]["pairs"], {"B<-A": 1})
        self.assertFalse([r for r in self.rows("drop") if r.get("node_of")])

    def test_release_drops_proprietary_generator_rows(self):
        lic = b.Licences()
        gen = "tasksource/UltraFeedback-paired"
        ok = "tasksource/PARARULE-Plus"
        self.assertTrue(lic.gen_of(gen))
        self.assertEqual((lic.v5(gen)[0] in b.RELEASE_OK, lic.v5(gen)[1]), (True, ""))
        self.assertEqual(lic.v5(ok), ("PERM", ""))
        self.assertEqual(lic.gen_of(ok), "")
        rng = random.Random(5)
        write(self.d / "c.jsonl", [final(f"c{i}", story(rng), source=gen if i % 2 else ok, family="fc")
                                   for i in range(10)])
        c = self.cfg(name="lab", sources=[{"name": "C", "slice": "A", "train": str(self.d / "c.jsonl"),
                                          "select": {"problems": "all", "rank": "file"}, "nodes": "keep",
                                          "label": "as_is", "licence": {"from": "v5"}, "gates": ["13gram"]}])
        lab = self.build(c, self.d, self.evals)
        self.assertEqual(lab["rows"]["train"], 10)
        self.assertEqual(lab["licence_gen_rows"], 5)
        self.assertFalse(any("licence_flag" in r or r["lab_only"] for r in self.rows("lab")))   # row bytes as before
        rel = self.build(dict(c, name="rel", tier="release"), self.d, self.evals)
        self.assertEqual({r["source"] for r in self.rows("rel")}, {ok})
        self.assertEqual(rel["rows"]["train"], 5)
        self.assertEqual(rel["per_source"]["C"]["dropped_rows"], {"licence": 5})
        allow = self.build(dict(c, name="allow", tier="release"), self.d, self.evals, allow_gen=True)
        self.assertEqual(allow["rows"]["train"], 10)

    def test_release_drops_lab_only(self):
        c = self.cfg(tier="release")
        s = self.build(c, self.d, self.evals)
        self.assertEqual(s["licence_class"], {"PERM": s["rows"]["train"]})
        self.assertEqual(s["lab_only_rows"], 0)

    def test_dedup_group(self):
        c = self.cfg()
        c["sources"][0]["dedup_group"] = c["sources"][1]["dedup_group"] = "gen"
        s = self.build(c, self.d, self.evals)
        ids = {r["item_id"] for r in self.rows("t")}
        self.assertIn("b_dup_near", ids)                             # same group: not a cross-source pair
        self.assertNotIn("b_contam", ids)
        self.assertEqual(s["dedup"]["near_dropped_rows"], 0)

    def test_fill_rows_and_split_val_deterministic(self):
        c = self.cfg(val={"mode": "split", "frac": .1, "min_per_family": 1, "max_per_family": 2},
                     fill={"source": "B", "target": "rows", "rows": 100})
        c["sources"][1]["select"] = {"rank": "hash", "fill": True}
        s1 = self.build(c, self.d, self.evals)
        self.assertEqual(s1["fill"]["achieved"], 100)
        self.assertEqual(s1["fill"]["shortfall"], 0)
        self.assertEqual(s1["rows"]["train"] + s1["rows"]["val"], 100)   # the split moves whole problems to val
        fams = s1["val"]["families"]
        self.assertTrue(all(v <= 2 for v in fams.values()))
        self.assertEqual(set(fams), {"fam0", "fam1", "fam2", "fb"})
        train = {r["item_id"] for r in self.rows("t")}
        val = {r["item_id"] for r in self.rows("t", "val")}
        self.assertFalse(train & val)
        self.assertFalse({r["node_of"] for r in self.rows("t") if r.get("node_of")} & val)   # problem-level split
        md5_1 = s1["md5"]
        s2 = self.build(c, self.d, self.evals)
        self.assertEqual(s2["md5"], md5_1)

    def test_fill_min_ratio(self):
        c = self.cfg(fill={"source": "B", "target": "rows", "rows": 1000, "min_ratio": .95})
        c["sources"][1]["select"] = {"rank": "hash", "fill": True}
        with self.assertRaises(SystemExit):
            self.build(c, self.d, self.evals)
        self.assertFalse((self.d / "out" / "t" / "train.jsonl").exists())     # refused before writing
        s = self.build(c, self.d, self.evals, allow_short_fill=True)
        self.assertTrue(s["fill"]["below_min_ratio"])
        self.assertEqual(s["fill"]["min_ratio"], .95)
        c["fill"]["rows"] = 60
        s = self.build(c, self.d, self.evals)
        self.assertFalse(s["fill"]["below_min_ratio"])
        self.assertEqual(s["fill"]["shortfall"], 0)

    def test_fill_counts_problems_of_a_node_dropping_source(self):
        c = self.cfg(fill={"source": "A", "target": "rows", "rows": 10})
        c["sources"] = [dict(c["sources"][0], nodes="drop", select={"rank": "file", "fill": True})]
        s = self.build(c, self.d, self.evals)
        self.assertEqual(s["fill"]["achieved"], 10)
        self.assertEqual(s["fill"]["problems_taken"], 10)
        self.assertEqual(s["rows"]["train"], 10)

    def test_minhash_deterministic(self):
        m1, m2 = b.MinHash(), b.MinHash()
        t = self.texts[3]
        self.assertTrue((m1(t) == m2(t)).all())
        self.assertIsNone(m1("too short"))
        self.assertEqual(b.hrank(0, "x", "y"), hashlib.md5(b"0|x|y").hexdigest())


if __name__ == "__main__":
    unittest.main()
