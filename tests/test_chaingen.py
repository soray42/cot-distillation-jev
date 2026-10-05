"""Decision chains (TB-C): exact truth, the final reads only the path top, the mirror is never read, per-parent
counterfactuals, no check result in the prompt, depth and label balance, certificate depth and full-depth evaluation
sets, the held-out prose renderer, chain tops as answer components, and --subq-filter (flag-off identity, guards)."""
import dataclasses
import json
import random
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cotdistill import chaingen as cg  # noqa: E402
from cotdistill import rulegen as rg  # noqa: E402
from cotdistill.treeplan import answer_component, full_tree_plan, node_depths  # noqa: E402

DEPTHS = (1, 2, 3, 4, 6, 8)


def cases(n=24, depths=DEPTHS, seed=5, q=0.8):
    return [c for d in depths for c in cg.generate(n, d, rg.TRAIN_DOMAINS, seed=seed, q=q)]


def recompute(rec: dict) -> tuple[dict, str]:
    """Independent evaluation from the stored record (facts and program JSON only)."""
    dom, prog = rg.DOMAINS[rec["domain"]], rec["program"]
    facts = rg.facts_from_json(dom, rec["facts"])
    val: dict = {}
    for c in prog["checks"]:
        lits = []
        for kind, ref, want in c["lits"]:
            v = rg.pred_value(dom, rg.Pred.parse(ref), facts) if kind == "p" else val[ref]
            lits.append(v == want)
        val[c["id"]] = all(lits) if c["op"] == "and" else any(lits)
    o1, o2, o3 = prog["outs"]
    side = rg.pred_value(dom, rg.Pred.parse(prog["side"]), facts)
    return val, (o1 if val[prog["top"]] else (o2 if side else o3))


def recompute_cert(rec: dict) -> int:
    """Independent certificate depth from the record: levels in the shallowest proof of the top check's value."""
    dom, prog = rg.DOMAINS[rec["domain"]], rec["program"]
    facts = rg.facts_from_json(dom, rec["facts"])
    val, _ = recompute(rec)
    checks = {c["id"]: c for c in prog["checks"]}

    def cert(cid):
        c, sat, cost = checks[cid], [], []
        for kind, ref, want in c["lits"]:
            v = rg.pred_value(dom, rg.Pred.parse(ref), facts) if kind == "p" else val[ref]
            sat.append(v == want)
            cost.append(0 if kind == "p" else cert(ref))
        if (c["op"] == "and") == val[cid]:
            return 1 + max(cost)
        return 1 + min(x for x, ok in zip(cost, sat) if ok == (c["op"] == "or"))
    return cert(prog["top"])


class TestChainGen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = cases()
        rng = random.Random(0)
        cls.recs = [cg.train_record(c, rng) for c in cls.cases]

    def test_deterministic(self):
        a = [cg.render(c) for c in cases(8, (2, 5), seed=9)]
        b = [cg.render(c) for c in cases(8, (2, 5), seed=9)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, [cg.render(c) for c in cases(8, (2, 5), seed=10)])

    def test_truth_recomputed_from_facts(self):
        for rec in self.recs:
            val, gold = recompute(rec)
            self.assertEqual(gold, rec["label_order"][rec["labels"].index(rec["gold_label"])])
            self.assertEqual({n["id"]: n["truth"] for n in rec["subqs"]}, val)
            self.assertTrue(all(n["p_cot"] == float(n["truth"]) for n in rec["subqs"]))
            back = cg.case_from_record(rec)
            self.assertEqual(cg.render(back), rec["prompt"])

    def test_final_reads_only_the_path_top(self):
        for c in self.cases:
            res = c.evaluate()
            mirror = [x.cid for x in c.checks if not x.on_path]
            for m in mirror:
                self.assertEqual(c.evaluate({m: not res["c"][m]})["decision"], res["decision"])
            self.assertEqual(c.evaluate({m: not res["c"][m] for m in mirror})["decision"], res["decision"])
            self.assertNotEqual(c.evaluate({c.top: not res["c"][c.top]})["decision"], res["decision"])
            self.assertEqual(res["decision"] == c.outs[0], res["c"][c.top])
            self.assertIn(f"Decision: if the {c.by_id()[c.top].name} check passes,", cg.render(c))

    def test_mirror_never_read(self):
        for c in self.cases:
            by = c.by_id()
            for x in c.checks:
                refs = [l.ref for l in x.lits if l.kind == "c"]
                self.assertTrue(all(r[0] == x.cid[0] for r in refs))                 # own chain only
                self.assertTrue(all(by[r].level < x.level for r in refs))           # older checks only
                if x.level > 1:
                    self.assertIn(f"{x.cid[0]}{x.level - 1}", refs)                 # every level reads the one below
                else:
                    self.assertEqual(refs, [])
            decision = cg.decision_line(c)
            for x in c.checks:
                self.assertEqual(x.name in decision, x.cid == c.top)
            self.assertEqual(sum(x.on_path for x in c.checks), sum(not x.on_path for x in c.checks))

    def test_cf_is_the_local_flip(self):
        for rec in self.recs:
            dom, prog = rg.DOMAINS[rec["domain"]], rec["program"]
            facts = rg.facts_from_json(dom, rec["facts"])
            val, _ = recompute(rec)
            checks = {c["id"]: c for c in prog["checks"]}
            for n in rec["subqs"]:
                c = checks[n["id"]]
                self.assertEqual(sorted(n["cf"]), sorted(n["depends_on"]))
                for par in n["depends_on"]:
                    lits = []
                    for kind, ref, want in c["lits"]:
                        v = rg.pred_value(dom, rg.Pred.parse(ref), facts) if kind == "p" else val[ref] != (ref == par)
                        lits.append(v == want)
                    flipped = all(lits) if c["op"] == "and" else any(lits)
                    self.assertEqual(n["cf_value"][par], flipped)
                    self.assertEqual(n["cf"][par], flipped != n["truth"])

    def test_no_check_result_in_prompt(self):
        for c in self.cases:
            prompt = cg.render(c)
            self.assertNotIn("Known intermediate results", prompt)
            names = [x.name for x in c.checks]
            kept = []
            for line in prompt.split("\n"):
                m = re.match(r"^- The (\w+) check passes if .+\.$", line)
                if m:
                    self.assertIn(m.group(1), names)
                    continue
                if line.startswith("Decision: if the "):
                    continue
                kept.append(line)
            rest = "\n".join(kept)
            for name in names:
                self.assertNotIn(name, rest)
            defs = [line for line in prompt.split("\n") if line.startswith("- The ")]
            self.assertEqual(len(defs), len(c.checks))
            self.assertEqual(sorted(re.match(r"^- The (\w+) check", d).group(1) for d in defs), sorted(names))
            self.assertTrue(prompt.endswith("\nOptions:\n" + "\n".join(
                f"{L}) {c.domain.option(k)}" for L, k in zip("ABCD", c.label_order))))

    def test_depth_and_balance(self):
        for d in DEPTHS:
            cs = cg.generate(40, d, rg.TRAIN_DOMAINS, seed=3)
            self.assertTrue(all(c.depth == d for c in cs))
            self.assertEqual(sum(c.evaluate()["c"][c.top] for c in cs), 20)
            for c in cs:
                nodes = cg.tree_nodes(c)
                self.assertEqual(sorted(n["node_depth"] for n in nodes if n["on_path"]), list(range(1, d + 1)))
                self.assertEqual(sorted(n["node_depth"] for n in nodes if not n["on_path"]), list(range(1, d + 1)))
                nd = node_depths(nodes)
                self.assertTrue(all(n["node_depth"] - nd[n["id"]] == 1 for n in nodes))
                e = cg.piv_run(c)
                piv = {n["id"]: n["pivotal"] for n in nodes}
                self.assertTrue(1 <= e <= d and piv[c.top])
                self.assertTrue(all(piv[f"P{k}"] for k in range(d - e + 1, d + 1)))
                self.assertTrue(e == d or not piv[f"P{d - e}"])
                self.assertFalse(any(piv[n["id"]] for n in nodes if not n["on_path"]))

    def test_q_one_without_fanin_makes_every_path_check_pivotal(self):
        for d in (3, 6):
            for c in cg.generate(10, d, rg.TRAIN_DOMAINS, seed=2, q=1.0, fanin=0.0):
                self.assertEqual(cg.piv_run(c), c.depth)
                self.assertEqual(cg.cert_depth(c), c.depth)

    def test_cert_depth_recomputed(self):
        for rec in self.recs:
            self.assertEqual(rec["cert_depth"], recompute_cert(rec))
            self.assertTrue(1 <= rec["cert_depth"] <= rec["depth"])
        self.assertTrue(any(r["cert_depth"] < r["depth"] for r in self.recs))   # training sets keep the natural mix

    def test_full_depth_sets(self):
        from build_chaingen import links_only
        for d in (2, 5, 8):
            cs = cg.generate(30, d, rg.TRAIN_DOMAINS, seed=7, full_depth=True)
            self.assertEqual(sum(c.evaluate()["c"][c.top] for c in cs), 15)
            for c in cs:
                rec = cg.eval_record(c)
                self.assertEqual(rec["cert_depth"], d)
                self.assertEqual(recompute_cert(rec), d)
                self.assertEqual(links_only(c), c.evaluate()["c"][c.top])  # facts above level 1 never decide here

    def test_prose_renderer(self):
        cs = cg.generate(12, 4, rg.HELDOUT_DOMAINS, seed=8, full_depth=True, renderer="chain_prose")
        for c in cs:
            rec = cg.eval_record(c)
            self.assertEqual(rec["renderer"], "chain_prose")
            self.assertIn(rec["domain"], rg.HELDOUT_DOMAINS)
            self.assertEqual(cg.render(cg.case_from_record(rec)), rec["prompt"])
            prompt, by = rec["prompt"], c.by_id()
            self.assertNotIn("Decision:", prompt)
            for x in c.checks:
                self.assertEqual(prompt.count(f"The {x.name} check passes if "), 1)
            self.assertIn(f"If the {by[c.top].name} check passes, you ", prompt)
            body = prompt.split("\n\n")[1]                                 # the facts paragraph
            self.assertFalse(any(x.name in body for x in c.checks))
            self.assertNotIn(" check passes", prompt.split("\n\n")[0].split(f"If the {by[c.top].name}")[1][30:])

    def test_controls_ask_about_unread_facts(self):
        for c, rec in zip(self.cases, self.recs):
            read = {rg.Pred.parse(l.ref).fact for x in c.checks for l in x.lits if l.kind == "p"}
            read.add(rg.Pred.parse(c.side).fact)
            self.assertFalse(read & set(c.reserved))
            self.assertTrue(rec["random_subqs"])
            for q in rec["random_subqs"]:
                self.assertIn(rg.Pred.parse(q["var"][4:]).fact, c.reserved)
                self.assertEqual(q["p_fresh"], float(q["truth"]))

    def test_student_schema(self):
        for rec in self.recs:
            for k in ("item_id", "prompt", "labels", "gold_label", "teacher", "subqs", "random_subqs", "tree_fix"):
                self.assertIn(k, rec)
            self.assertEqual(rec["teacher"][rec["gold_label"]], 1.0)
            for n in rec["subqs"]:
                for k in ("id", "question", "truth", "p_cot", "depends_on", "node_depth", "type", "on_path", "cf"):
                    self.assertIn(k, n)
            json.dumps(rec)


class TestAnswerComponent(unittest.TestCase):
    def test_chain_tops_only(self):
        from cotdistill.student import build_group
        rng = random.Random(3)
        for c in cases(4, (1, 2, 4), seed=13):
            it = cg.train_record(c, rng)
            comp = [sq for sq in it["subqs"] if answer_component(sq, it)]
            self.assertEqual(sorted(sq["id"] for sq in comp), sorted([f"P{c.depth}", f"M{c.depth}"]))
            for f in ("path", "mirror"):
                for seed in range(4):
                    g = build_group(it, aux_kind="cot", k=2, aux_weight=0.5, rng=random.Random(seed), subq_filter=f,
                                    aux_answer="exclude")
                    self.assertEqual(len(g), 3)
                    asked = [e for e in g[1:] if e.weight > 0]
                    self.assertEqual(len(asked), min(2, c.depth - 1))
                    tops = {sq["question"] for sq in comp}
                    self.assertFalse(any(e.text.split("Intermediate question: ")[1].split("\n")[0] in tops
                                         for e in asked))
        rule = rg.train_record(rg.generate(1, rg.TRAIN_DOMAINS, seed=4)[0], random.Random(0))
        self.assertFalse(any(answer_component(sq, rule) for sq in rule["subqs"]))


class TestSubqFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = random.Random(1)
        cls.items = [cg.train_record(c, rng) for c in cases(6, (2, 3, 4), seed=11)]

    def question(self, e):
        return e.text.split("\n\nIntermediate question: ")[1].split("\n")[0]

    def test_grouped_and_examples_ask_only_the_chain(self):
        from cotdistill.student import build_examples, build_group
        for it in self.items:
            on = {sq["question"]: sq["on_path"] for sq in it["subqs"]}
            for f, want in (("path", True), ("mirror", False)):
                for seed in range(5):
                    g = build_group(it, aux_kind="cot", k=2, aux_weight=0.5, rng=random.Random(seed), subq_filter=f)
                    self.assertEqual(len(g), 3)
                    self.assertTrue(all(on[self.question(e)] is want for e in g[1:]))
                    ex = build_examples(it, final="teacher", subq="cot", subq_target="truth", lambda_sub=1.0,
                                        rng=random.Random(seed), subq_filter=f)
                    self.assertEqual(sorted(self.question(e) for e in ex[1:]),
                                     sorted(q for q, p in on.items() if p is want))

    def test_full_tree_asks_path_matches_from_mirror(self):
        for it in self.items:
            by = {sq["id"]: sq for sq in it["subqs"]}
            plan = full_tree_plan(it, cap=10, rng=random.Random(0), subq_filter="path")
            self.assertEqual(sorted(e["node"]["id"] for e in plan), sorted(i for i in by if by[i]["on_path"]))
            elig = [e for e in plan if e["parents"]]
            self.assertTrue(elig)
            for e in elig:
                self.assertTrue(all(by[j]["on_path"] for j in e["parents"]))
                self.assertTrue(all(not by[j]["on_path"] for j in e["matched"]))

    def test_transition_views_keep_rewiring_pool(self):
        from cotdistill.student import build_transition_views
        for it in self.items:
            on = {sq["question"]: sq["on_path"] for sq in it["subqs"]}
            for seed in range(5):
                ex = build_transition_views(it, k=2, rng=random.Random(seed), shuffled=True, subq_filter="path")
                self.assertTrue(all(on[self.question(e)] for e in ex))
                self.assertTrue(any("Known intermediate results" in e.text for e in ex))

    def test_flag_off_identity(self):
        from cotdistill.student import build_examples, build_full_tree, build_group
        rule = [rg.train_record(c, random.Random(0)) for c in rg.generate(12, rg.TRAIN_DOMAINS, seed=4)]
        for it in rule + self.items:
            for seed in range(3):
                a = build_examples(it, final="teacher", subq="mix", subq_target="cot", lambda_sub=1.0,
                                   rng=random.Random(seed))
                b = build_examples(it, final="teacher", subq="mix", subq_target="cot", lambda_sub=1.0,
                                   rng=random.Random(seed), subq_filter="all")
                self.assertEqual(a, b)
                for kind in ("cot", "tree_shuf", "cot_mc"):
                    a = build_group(it, aux_kind=kind, k=2, aux_weight=0.5, rng=random.Random(seed))
                    b = build_group(it, aux_kind=kind, k=2, aux_weight=0.5, rng=random.Random(seed), subq_filter="all")
                    self.assertEqual(a, b)
                a = build_full_tree(it, mode="matched", rng=random.Random(seed), eligible_only=True)
                b = build_full_tree(it, mode="matched", rng=random.Random(seed), eligible_only=True, subq_filter="all")
                self.assertEqual([dataclasses.asdict(x) for x in a], [dataclasses.asdict(x) for x in b])

    def test_parent_sensitivity_strata(self):
        from parent_sens_eval import summarize
        preds = []
        for i, path in enumerate((True, False, None)):
            for c, p in (("true", 0.9), ("flip_true", 0.2), ("matched", 0.6), ("flip_matched", 0.55), ("plain", 0.5)):
                r = {"item_id": f"i{i}", "node": "n", "cond": c, "source": "chain", "target": True, "n_par": 1,
                     "cf_changes": True, "p_yes": p}
                if path is not None:
                    r["on_path"] = path
                preds.append(r)
        out = summarize(preds, n_boot=10)
        self.assertEqual(out["path"]["n_items"], 1)
        self.assertEqual(out["mirror/cf_changes"]["n_items"], 1)
        self.assertNotIn("path/cf_invariant", out)
        self.assertEqual(out["all"]["n_items"], 3)
        legacy = summarize([dict(r) for r in preds if "on_path" not in r], n_boot=10)
        self.assertNotIn("path", legacy)

    def test_parent_sensitivity_reads_cf_value(self):
        from parent_sens_eval import _cf_rule
        it = self.items[0]
        by = {sq["id"]: sq for sq in it["subqs"]}
        for sq in it["subqs"]:
            for par in sq["depends_on"]:
                self.assertEqual(_cf_rule(it, sq, by, par), sq["cf_value"][par])


class TestTrainingOnCpu(unittest.TestCase):
    def test_grouped_path_filter_and_chain_subq_strata(self):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            from test_student import tiny
        except ImportError:
            self.skipTest("needs torch + transformers")
        import os
        import subprocess
        import tempfile
        tok, model = tiny()
        rng = random.Random(0)
        items = [cg.train_record(c, rng) for c in cases(3, (1, 2), seed=12)]
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d + "/m"); tok.save_pretrained(d + "/m")
            Path(d + "/t.jsonl").write_text("\n".join(json.dumps(r) for r in items))
            for args in (["--grouped-aux", "2", "--aux-kind", "cot", "--items-per-update", "2"],
                         ["--tree-full", "matched", "--tree-eligible-only", "--items-per-update", "2"]):
                out = subprocess.run([sys.executable, str(ROOT / "scripts/train_student.py"), "--model", d + "/m",
                                      "--train", d + "/t.jsonl", "--eval", "v=" + d + "/t.jsonl", *args,
                                      "--subq-filter", "path", "--subq-target", "truth", "--epochs", "1",
                                      "--micro-bs", "2", "--max-len", "512", "--eval-max-len", "512", "--precision",
                                      "bf16", "--no-grad-ckpt", "--eval-subq", d + "/t.jsonl", "--out", d + "/o"],
                                     capture_output=True, text=True, timeout=600,
                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
                self.assertEqual(out.returncode, 0, out.stderr[-2000:])
                m = json.loads(Path(d + "/o/metrics.json").read_text())
                self.assertEqual(m["args"]["subq_filter"], "path")
                self.assertIn("cot/path/d2", m["eval"]["subq"])
                self.assertIn("cot/mirror/d1", m["eval"]["subq"])
                self.assertIn("cot/path/top-0", m["eval"]["subq"])
                self.assertIn("cot/mirror/item_d2", m["eval"]["subq"])
            rule = [rg.train_record(c, random.Random(0)) for c in rg.generate(4, rg.TRAIN_DOMAINS, seed=4)]
            Path(d + "/r.jsonl").write_text("\n".join(json.dumps(r) for r in rule))
            for args, msg in ((["--train", d + "/r.jsonl", "--grouped-aux", "2", "--aux-kind", "cot"], "no node of"),
                              (["--train", d + "/t.jsonl", "--grouped-aux", "2", "--aux-kind", "random"], "asks none")):
                out = subprocess.run([sys.executable, str(ROOT / "scripts/train_student.py"), "--model", d + "/m",
                                      "--eval", "v=" + d + "/t.jsonl", *args, "--subq-filter", "path", "--epochs", "1",
                                      "--precision", "bf16", "--no-grad-ckpt", "--out", d + "/o2"],
                                     capture_output=True, text=True, timeout=600,
                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
                self.assertNotEqual(out.returncode, 0)
                self.assertIn(msg, out.stderr)


class TestAudit(unittest.TestCase):
    def test_logistic_fits_a_linear_rule(self):
        import numpy as np
        from build_chaingen import accuracy, fit_logistic
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 5)).astype(np.float32)
        y = (X[:, 0] - 2 * X[:, 3] > 0).astype(np.float32)
        self.assertGreater(accuracy(fit_logistic(X, y, 1e-4), X, y), 0.95)


if __name__ == "__main__":
    unittest.main()
