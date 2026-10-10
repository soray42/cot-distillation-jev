"""Multiple-choice tree nodes (treeplan.is_mc_node: a node with its own "options" and "answer"; opt-in data such as
scripts/build_fa_sf.py --trees): asked like a final question in the full-tree views (build_full_tree, GFP-S, GFPSP
packing) and in evaluate_subq, one-hot target on the answer; the default paths unchanged against the commit before
(MC_BASE_REF) on real data/student_fa1dall (fa1d step and agent items) and data/student_v3sl items. CPU only, tiny
random Qwen2 with the GPT-2 tokenizer."""
import copy
import dataclasses
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

try:
    import torch
    import torch.nn.functional as F
    import transformers  # noqa: F401
    import test_student as ts
    from test_tree_forcing import ENV, KK, SCRIPT, head_student, head_tree, tiny
    HAVE = True
except ImportError:
    HAVE = False

# The last commit before multiple-choice tree nodes.
MC_BASE_REF = "ac9eec1"
FA_TRAIN = ROOT / "data/student_fa1dall/train.jsonl"
FA_VAL = ROOT / "data/student_fa1dall/val.jsonl"
V3SL_VAL = ROOT / "data/student_v3sl/val.jsonl"
VIEWS = [("plain", "problem"), ("plain", "statements"), ("plain", "statements+question"), ("true", "problem"),
         ("matched", "problem"), ("placebo", "problem"), ("control", "problem"), ("true", "local"),
         ("matched", "local"), ("plain", "local"), ("true", "ancestors"), ("true", "local+ctxparse"), ("true", "mix")]
GFPSP_FA = ["--tree-full", "plain", "--tree-view", "statements", "--tree-cap", "0", "--tree-sources", "fa_public",
            "--subq", "cot", "--subq-target", "truth", "--tree-pack", "8"]


def items_of(path: Path, n: int, pred=lambda it: True) -> list[dict]:
    out = []
    if path.exists():
        with open(path) as f:
            for line in f:
                it = json.loads(line)
                if pred(it):
                    out.append(it)
                    if len(out) == n:
                        break
    return out


def fa_items(n_step: int = 3, n_agent: int = 2, path: Path = FA_TRAIN) -> list[dict]:
    """fa1d step items (label-derived yes/no nodes) and agent items of data/student_fa1dall."""
    step = items_of(path, n_step, lambda it: "/agent" not in it.get("group", ""))
    agent = items_of(path, n_agent, lambda it: "/agent" in it.get("group", ""))
    return step + agent


def toy_fa() -> dict:
    """A small fa_public step item (fa_public.make_items) for when data/student_fa1dall is not built."""
    from cotdistill import fa_public as fp
    steps = [("Planner", "plan the search"), ("WebSurfer", "searched and found 12"), ("Planner", "so the answer is 12"),
             ("WebSurfer", "done")]
    rec = fp._rec("u1", "stepfinder", "alg", "whowhen_like", "How many?", "13", steps, 1, "WebSurfer")
    return fp.make_items(rec, "toy-0", "fa1/stepfinder", cap=200, with_gt=False, rng=random.Random(0))[0]


def with_mc(item: dict, link: bool = True) -> dict:
    """item plus three MC nodes: a pointer node over the step options, an agent node depending on it, and a text node
    depending on a yes/no node (when the item has one and link). link=False: the text node depends on nothing and every
    MC node carries a node_depth, so the full-tree plan of the yes/no nodes (sinks, depths, matched draws) is the
    item's own."""
    it = copy.deepcopy(item)
    names = [it["label_names"][L] for L in it["labels"]][:5]
    yn = [sq["id"] for sq in it.get("subqs", []) if "id" in sq][:1] if link else []
    it["subqs"] = it.get("subqs", []) + [
        {"id": "m1", "type": "answer_step", "question": "Which step states the team's final answer?",
         "options": names, "answer": len(names) - 1, "depends_on": []},
        {"id": "m2", "type": "agent", "question": "Who wrote the step that the final answer relies on?",
         "options": ["Planner", "WebSurfer", "Coder"], "answer": 1, "depends_on": ["m1"]},
        {"id": "m3", "type": "content", "question": "What did the decisive step claim?",
         "options": ["a count of 12", "a count of 13", "no count", "a date"], "answer": 0, "depends_on": yn}]
    if not link:
        for sq, dep in zip(it["subqs"][-3:], (0, 1, 0)):
            sq["node_depth"] = dep
    return it


def astuples(exs):
    from cotdistill.student import Pack
    out = []
    for e in exs:
        if type(e).__name__ == Pack.__name__:                # the base commit's Pack is another class
            out.append(("pack", tuple(e.parts), tuple(dataclasses.astuple(n) for n in e.nodes)))
        else:
            out.append(dataclasses.astuple(e))
    return out


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestDefaultUnchanged(unittest.TestCase):
    def items(self):
        return ((fa_items() or [toy_fa()]) + items_of(V3SL_VAL, 2, lambda it: it.get("source") == "knights_knaves")
                + [KK] + items_of(V3SL_VAL, 1, lambda it: it.get("source") == "justlogic"))

    def test_library_matches_base(self):
        """build_full_tree (every full-tree mode and view, caps, sources, targets), build_examples (--subq cot) and
        pack_views give exactly MC_BASE_REF's examples and stats on fa1d step / agent items and v3sl items."""
        from cotdistill import student as S
        from cotdistill.treeplan import statement_block
        with tempfile.TemporaryDirectory() as d:
            H = head_student(d, MC_BASE_REF, "cotdistill_mcbase")
            if H is None:
                self.skipTest("not a git checkout")
            tok, _ = ts.tiny()
            n = 0
            for it in self.items():
                for mode, view in VIEWS:
                    for cap in (0, 10):
                        for sources in (None, {"fa_public", "knights_knaves"}):
                            for tgt in ("cot", "truth"):
                                for seed in range(2):
                                    kw = dict(cap=cap, sources=sources, permute_final=0.5, subq_target=tgt)
                                    out = []
                                    for mod in (H, S):
                                        st = {}
                                        try:
                                            ex = mod.build_full_tree(it, mode=mode, view=view, rng=random.Random(seed),
                                                                     stats=st, **kw)
                                            if (mode, view) == ("plain", "statements") and any(
                                                    e.kind == "subq" for e in ex):
                                                ex = mod.pack_views(ex, tok, statement_block(it), max_len=512,
                                                                    max_nodes=4, rng=mod.pack_rng(seed, 0, "x"),
                                                                    stats=st)
                                            out.append((astuples(ex), st))
                                        except ValueError as e:
                                            out.append(repr(e))
                                    self.assertEqual(out[0], out[1], (it["item_id"], mode, view, cap, sources, tgt))
                                    n += 1
                for subq in ("cot", "none", "mix"):
                    a = [dataclasses.astuple(e) for e in H.build_examples(it, final="teacher", subq=subq,
                                                                          subq_target="truth", lambda_sub=1.0,
                                                                          rng=random.Random(1), permute_final=0.5)]
                    b = [dataclasses.astuple(e) for e in S.build_examples(it, final="teacher", subq=subq,
                                                                          subq_target="truth", lambda_sub=1.0,
                                                                          rng=random.Random(1), permute_final=0.5)]
                    self.assertEqual(a, b, (it["item_id"], subq))
            self.assertGreater(n, 300)

    def test_eval_subq_matches_base(self):
        """evaluate_subq: the same metrics and predictions as MC_BASE_REF on fa1d val items and v3sl items."""
        from cotdistill import student as S
        with tempfile.TemporaryDirectory() as d:
            H = head_student(d, MC_BASE_REF, "cotdistill_mcbase2")
            if H is None:
                self.skipTest("not a git checkout")
            tok, model = ts.tiny()
            model.eval()
            items = (fa_items(2, 1, FA_VAL) or [toy_fa()]) + [KK]
            a = H.evaluate_subq(model, tok, items, 384, 4, {}, seed=3)
            b = S.evaluate_subq(model, tok, items, 384, 4, {}, seed=3)
            self.assertEqual(a, b)
            self.assertGreater(len(b[1]), 5)

    def test_training_matches_base(self):
        """A2 and GFPSP (the fa1d settings: --tree-sources fa_public --subq-target truth --tree-pack 8) through main():
        the same history, evaluations, tree stats and predictions as MC_BASE_REF's script on fa1d items."""
        with tempfile.TemporaryDirectory() as d:
            if head_tree(d + "/h", MC_BASE_REF) is None:
                self.skipTest("not a git checkout")
            tiny(d)
            items = fa_items(4, 2) or [dict(toy_fa(), item_id=f"t{j}") for j in range(4)]
            Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in items))
            arms = {"A2": [], "GFPSP": GFPSP_FA}
            runs = {}
            for name, script in (("base", d + "/h/scripts/train_student.py"), ("now", SCRIPT)):
                for arm, extra in arms.items():
                    out = f"{d}/{name}-{arm}"
                    r = subprocess.run([sys.executable, script, "--model", d + "/m", "--train", d + "/t.jsonl",
                                        "--eval", "v=" + d + "/t.jsonl", "--eval-subq", d + "/t.jsonl", "--epochs", "1",
                                        "--items-per-update", "2", "--micro-bs", "2", "--max-len", "512",
                                        "--eval-max-len", "512", "--precision", "fp32master", "--no-grad-ckpt",
                                        "--log-every", "1", "--seed", "3", "--label-smoothing", "0.1",
                                        "--token-budget", "1200", "--out", out, *extra],
                                       capture_output=True, text=True, timeout=900, env=ENV)
                    self.assertEqual(r.returncode, 0, (name, arm, r.stderr[-2000:]))
                    m = json.loads(Path(out + "/metrics.json").read_text())
                    hist = [{k: v for k, v in h.items() if k not in ("s_per_step", "approx_tok_per_s")}
                            for h in m["history"]]
                    runs[name, arm] = (hist, m["eval"], m.get("tree_view_stats"),
                                       Path(out + "/preds_v.jsonl").read_text(),
                                       Path(out + "/preds_subq.jsonl").read_text())
            for arm in arms:
                self.assertEqual(runs["base", arm], runs["now", arm], arm)
                self.assertGreater(len(runs["now", arm][0]), 0)


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestMCViews(unittest.TestCase):
    def item(self, link: bool = True):
        return with_mc((fa_items(1, 0) or [toy_fa()])[0], link)

    def test_statements_view(self):
        """GFP-S: every MC node asked once under the statement block with its own options (A, B, ...), a one-hot
        target on its answer and the per-problem weight; the yes/no views (texts, targets, order) are those of the item
        without the MC nodes (no extra rng draw)."""
        from cotdistill.student import SUBQ_OPTS_TEMPLATE, build_full_tree
        from cotdistill.treeplan import MC_SYMBOLS, statement_block
        it = self.item(link=False)
        base = dict(it, subqs=[sq for sq in it["subqs"] if "options" not in sq])
        for seed in range(3):
            ex = build_full_tree(it, mode="plain", view="statements", rng=random.Random(seed), cap=0,
                                 subq_target="truth", aux_total=1.0)
            ref = build_full_tree(base, mode="plain", view="statements", rng=random.Random(seed), cap=0,
                                  subq_target="truth", aux_total=1.0)
            self.assertEqual(dataclasses.astuple(ex[0]), dataclasses.astuple(ref[0]))
            mc = [e for e in ex[1:] if e.labels != ["A", "B"]]
            yn = [e for e in ex[1:] if e.labels == ["A", "B"]]
            self.assertEqual([(e.text, e.target) for e in yn], [(e.text, e.target) for e in ref[1:]])
            self.assertEqual(len(mc), 3)
            block = statement_block(it)
            for e, sq in zip(mc, [sq for sq in it["subqs"] if "options" in sq]):
                opts = "\n".join(f"{L}) {t}" for L, t in zip(MC_SYMBOLS, sq["options"]))
                self.assertEqual(e.text, SUBQ_OPTS_TEMPLATE.format(problem=block, question=sq["question"], opts=opts))
                self.assertTrue(e.text.startswith(block + "\n\nIntermediate question: "))
                self.assertEqual(e.labels, list(MC_SYMBOLS[:len(sq["options"])]))
                self.assertEqual(e.target, [float(i == sq["answer"]) for i in range(len(sq["options"]))])
                self.assertEqual(e.kind, "subq")
            for e in ex[1:]:
                self.assertAlmostEqual(e.weight, 1.0 / (len(ex) - 1), places=12)
            # node types: only the MC agent node and the yes/no agent nodes
            only = build_full_tree(it, mode="plain", view="statements", rng=random.Random(seed), cap=0,
                                   subq_target="truth", node_types=frozenset({"agent"}))
            self.assertTrue(any(e.labels == ["A", "B", "C"] for e in only))

    def test_problem_true_view_states_yes_no_parents(self):
        """Mode true (problem view): an MC node whose parent is a confident yes/no node is asked with that result
        stated; an MC parent is never stated (not confident)."""
        from cotdistill.student import FACTS_HEADER, build_full_tree
        it = self.item()
        yn = [sq for sq in it["subqs"] if "options" not in sq]
        if not yn:
            self.skipTest("no yes/no node")
        stated = False
        for seed in range(6):
            for e in build_full_tree(it, mode="true", view="problem", rng=random.Random(seed), cap=0,
                                     subq_target="truth"):
                if "What did the decisive step claim?" in e.text:
                    self.assertEqual(e.labels, ["A", "B", "C", "D"])
                    stated |= FACTS_HEADER in e.text
                if "Who wrote the step" in e.text:
                    self.assertNotIn(FACTS_HEADER, e.text)          # its parent m1 is an MC node
        self.assertTrue(stated)

    def test_pack_and_readout(self):
        """GFPSP: MC views are packed with the yes/no views (same examples, targets, weights as unpacked); every
        readout (2-letter and MC) equals the model's logits on the prefix up to its cue alone, over its own labels."""
        from cotdistill.student import (Pack, build_full_tree, label_logits, letter_token_ids, pack_ids, pack_rng,
                                        pack_views, readouts, text_parts)
        from cotdistill.treeplan import statement_block
        tok, model = ts.tiny()
        model.eval()
        it = self.item()
        g = build_full_tree(it, mode="plain", view="statements", rng=random.Random(0), cap=0, subq_target="truth")
        p = pack_views(g, tok, statement_block(it), max_len=100000, max_nodes=5, rng=pack_rng(0, 0, it["item_id"]))
        packs = [e for e in p if isinstance(e, Pack)]
        self.assertTrue(packs)
        self.assertEqual(sorted((e.text, e.weight, tuple(e.target)) for e in readouts(p) if e.kind == "subq"),
                         sorted((e.text, e.weight, tuple(e.target)) for e in g if e.kind == "subq"))
        self.assertTrue(any(len(n.labels) > 2 for pk in packs for n in pk.nodes))
        # readout check on a tiny log (the GPT-2 toy model has 512 positions)
        small = with_mc(toy_fa())
        g2 = build_full_tree(small, mode="plain", view="statements", rng=random.Random(1), cap=0, subq_target="truth")
        p2 = pack_views(g2, tok, statement_block(small), max_len=512, max_nodes=8, rng=pack_rng(1, 0, "toy-0"))
        pk = [e for e in p2 if isinstance(e, Pack)][0]
        self.assertTrue(any(len(n.labels) > 2 for n in pk.nodes))
        body, head = text_parts(model)
        with torch.no_grad():
            z = label_logits(model, tok, [pk], 512, {})
            ids, pos = pack_ids(tok, pk)
            for k, q in enumerate(pos):
                h = body(input_ids=torch.tensor([ids[:q + 1]]), use_cache=False).last_hidden_state[0, -1]
                ref = h @ head.weight[letter_token_ids(tok, pk.nodes[k].labels)].T
                self.assertEqual(z[k].shape, ref.shape)
                self.assertLess((z[k] - ref).abs().max().item(), 1e-4)

    def test_loss_is_manual_ce(self):
        """kl_loss of an MC node view = weight x (-log softmax over its option letters at the answer) / batch size;
        with label smoothing s the target is (1 - s) one-hot + s / K, as for a final question."""
        from cotdistill.student import build_full_tree, kl_loss, label_logits, letter_token_ids, text_parts
        tok, model = ts.tiny()
        model.eval()
        it = with_mc(toy_fa())
        g = build_full_tree(it, mode="plain", view="statements", rng=random.Random(0), cap=0, subq_target="truth")
        mc = [e for e in g if len(e.labels) > 2 and e.kind == "subq"]
        self.assertEqual(len(mc), 3)
        body, head = text_parts(model)
        with torch.no_grad():
            z = label_logits(model, tok, mc, 512, {})
            for s in (0.0, 0.1):
                got = kl_loss(z, mc, smoothing=s).item()
                want = 0.0
                for e in mc:
                    ids = tok.encode(e.text)
                    h = body(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state[0, -1]
                    logits = h @ head.weight[letter_token_ids(tok, e.labels)].T
                    ans = e.target.index(1.0)
                    if s == 0:
                        ce = F.cross_entropy(logits[None], torch.tensor([ans])).item()
                    else:
                        lp = torch.log_softmax(logits, -1)
                        K = len(e.labels)
                        ce = -sum(((1 - s) * (i == ans) + s / K) * lp[i].item() for i in range(K))
                    want += e.weight * ce
                self.assertAlmostEqual(got, want / len(mc), places=5)

    def test_eval_subq(self):
        """evaluate_subq: MC nodes scored by argmax == answer (pred, rank, probs, group cot/mc); the yes/no
        predictions are those of the item without its MC nodes (no rng draw)."""
        from cotdistill.student import evaluate_subq
        tok, model = ts.tiny()
        model.eval()
        it = with_mc(toy_fa())
        base = dict(it, subqs=[sq for sq in it["subqs"] if "options" not in sq])
        m, preds = evaluate_subq(model, tok, [it], 512, 3, {}, seed=5)
        m0, preds0 = evaluate_subq(model, tok, [base], 512, 3, {}, seed=5)
        mc = [p for p in preds if "mc" in p]
        self.assertEqual(len(mc), 3)
        self.assertEqual([p for p in preds if "mc" not in p], preds0)
        for p, sq in zip(mc, [sq for sq in it["subqs"] if "options" in sq]):
            self.assertEqual(len(p["probs"]), len(sq["options"]))
            self.assertAlmostEqual(sum(p["probs"]), 1.0, places=5)
            self.assertEqual(p["pred"], max(range(len(p["probs"])), key=p["probs"].__getitem__))
            self.assertEqual(p["correct"], int(p["pred"] == sq["answer"]))
            self.assertEqual(p["rank"] == 1, bool(p["correct"]))
            self.assertAlmostEqual(p["p_yes"], p["probs"][sq["answer"]])
        self.assertEqual(m["cot/mc"]["n"], 3)
        self.assertEqual(m["all_cot"]["n"], m0["all_cot"]["n"] + 3)
        self.assertEqual(m["cot/program_truth"], m0["cot/program_truth"])

    def test_guards(self):
        from cotdistill.student import build_examples, build_full_tree
        from cotdistill.treeplan import mc_check
        it = self.item()
        with self.assertRaises(ValueError):
            build_examples(it, final="teacher", subq="cot", subq_target="truth", lambda_sub=1.0,
                           rng=random.Random(0))
        kk = copy.deepcopy(KK)
        kk["subqs"].append({"id": "mz", "type": "x", "question": "q?", "options": ["a", "b"], "answer": 0,
                            "depends_on": []})
        with self.assertRaises(ValueError):
            build_full_tree(kk, mode="true", view="local", rng=random.Random(0), cap=0)
        for bad in ({"options": ["a"], "answer": 0}, {"options": ["a", "a"], "answer": 0},
                    {"options": ["a", "b"], "answer": 2}, {"options": ["a", "b"], "answer": True},
                    {"options": ["a", "b\nc"], "answer": 0}, {"options": [str(i) for i in range(63)], "answer": 0}):
            with self.assertRaises(ValueError):
                mc_check(dict(bad, id="x", question="q?"))
        mc_check({"id": "x", "question": "q?", "options": [str(i) for i in range(62)], "answer": 61})


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestTrainMC(unittest.TestCase):
    def test_gfpsp_with_mc_nodes(self):
        """GFPSP through main() on items with MC nodes: runs, every node view counted, node losses finite, and the node
        eval scores the MC nodes."""
        with tempfile.TemporaryDirectory() as d:
            tiny(d)
            items = [dict(with_mc(toy_fa()), item_id=f"t{j}") for j in range(4)]
            Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in items))
            r = subprocess.run([sys.executable, SCRIPT, "--model", d + "/m", "--train", d + "/t.jsonl",
                                "--eval", "v=" + d + "/t.jsonl", "--eval-subq", d + "/t.jsonl", "--epochs", "2",
                                "--items-per-update", "2", "--micro-bs", "2", "--max-len", "512",
                                "--eval-max-len", "512", "--precision", "fp32master", "--no-grad-ckpt",
                                "--log-every", "1", "--lr", "1e-3", "--out", d + "/o", *GFPSP_FA],
                               capture_output=True, text=True, timeout=900, env=ENV)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            m = json.loads(Path(d + "/o/metrics.json").read_text())
            n_nodes = len(items[0]["subqs"])
            self.assertEqual(m["tree_view_stats"]["views"], 4 * n_nodes)
            self.assertGreater(m["tree_view_stats"]["pack_seqs"], 0)
            for h in m["history"]:
                self.assertTrue(math.isfinite(h["loss"]))
            self.assertEqual(m["eval"]["subq"]["cot/mc"]["n"], 12)
            mc = [json.loads(x) for x in Path(d + "/o/preds_subq.jsonl").read_text().splitlines()]
            self.assertEqual(sum("mc" in p for p in mc), 12)


if __name__ == "__main__":
    unittest.main()
