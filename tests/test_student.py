"""CPU tests for the student readout/loss with a tiny random Qwen2 (skipped without torch/transformers)."""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
    import transformers
    HAVE = True
except ImportError:
    HAVE = False


def tiny():
    tok = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    tok.pad_token = tok.eos_token
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                                   tie_word_embeddings=True)
    torch.manual_seed(0)
    return tok, transformers.Qwen2ForCausalLM(cfg)


def toy_item(i, rng):
    color = rng.choice(["red", "blue"])
    labels = ["A", "B"]
    gold = "A" if color == "red" else "B"
    return {"item_id": f"t{i}", "prompt": f"Case {i}. The light is {color}.\nOptions:\nA) stop\nB) go",
            "labels": labels, "gold_label": gold, "teacher": {gold: 0.95, ("B" if gold == "A" else "A"): 0.05},
            "subqs": [{"question": "Is the light red?", "p_fresh": 0.9 if color == "red" else 0.1, "truth": color == "red"}],
            "random_subqs": [{"question": "Is the case number even?", "p_fresh": 1.0 if i % 2 == 0 else 0.0}]}


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestStudent(unittest.TestCase):
    def test_padding_invariance(self):
        from cotdistill.student import Example, label_logits
        tok, model = tiny()
        model.eval()
        exs = [Example("short text\n\nAnswer:", ["A", "B"], [1, 0], 1.0, "final", "a"),
               Example("a much longer piece of text " * 8 + "\n\nAnswer:", ["A", "B", "C"], [0, 1, 0], 1.0, "final", "b")]
        cache = {}
        with torch.no_grad():
            both = label_logits(model, tok, exs, 256, cache)
            one = [label_logits(model, tok, [e], 256, cache)[0] for e in exs]
        for a, b in zip(both, one):
            self.assertTrue(torch.allclose(a, b, atol=1e-4), (a, b))

    def test_evaluate_returns_input_order(self):
        from cotdistill.student import evaluate
        tok, model = tiny()
        items = [{"item_id": f"e{i}", "prompt": "word " * n + "\nOptions:\nA) x\nB) y", "labels": ["A", "B"],
                  "gold_label": "A"} for i, n in enumerate([40, 3, 25, 1, 60])]
        _, preds = evaluate(model, tok, items, 256, 2, {})
        self.assertEqual([p["item_id"] for p in preds], [it["item_id"] for it in items])
        _, alone = evaluate(model, tok, items[:1], 256, 1, {})
        self.assertTrue(all(abs(a - b) < 1e-4 for a, b in zip(preds[0]["probs"], alone[0]["probs"])))

    def test_lm_loss_matches_direct_cross_entropy(self):
        from cotdistill.student import Example, lm_loss
        tok, model = tiny()
        model.eval()
        e = Example("Problem: two plus two.\n\nReasoning:", ["A", "B"], [], 2.0, "lm", "x",
                    continuation=" add them to get four.\n\nAnswer: B")
        e2 = Example("Short.\n\nReasoning:", ["A", "B"], [], 1.0, "lm", "y", continuation=" ok\n\nAnswer: A")
        with torch.no_grad():
            got = lm_loss(model, tok, [e, e2], 256, chunk=3)
            want = 0.0
            for ex in (e, e2):
                pre = tok.encode(ex.text, add_special_tokens=False)
                cont = tok.encode(ex.continuation, add_special_tokens=False)
                ids = torch.tensor([pre + cont])
                logits = model(input_ids=ids).logits[0, len(pre) - 1:-1]
                want += ex.weight * torch.nn.functional.cross_entropy(logits, ids[0, len(pre):]).item()
        self.assertAlmostEqual(got.item(), want, places=4)

    def test_rationale_arm_adds_lm_example(self):
        from cotdistill.student import build_examples
        it = dict(toy_item(0, random.Random(0)), rationale="The light is red, so stop.")
        ex = build_examples(it, final="teacher", subq="none", subq_target="cot", lambda_sub=1,
                            rng=random.Random(0), rationale_lm=True)
        self.assertEqual([e.kind for e in ex], ["final", "lm"])
        self.assertTrue(ex[1].continuation.endswith("Answer: " + max(it["teacher"], key=it["teacher"].get)))

    def test_mix_arm_matches_cot_count(self):
        from cotdistill.student import build_examples
        it = toy_item(0, random.Random(0))
        it["subqs"] = [dict(it["subqs"][0], question=f"cot {i}?", p_cot=0.9) for i in range(5)]
        it["random_subqs"] = [{"question": f"ctl {i}?", "p_fresh": 0.1} for i in range(5)]
        ex = build_examples(it, final="teacher", subq="mix", subq_target="cot", lambda_sub=1, rng=random.Random(1))
        qs = [e.text.split("Intermediate question: ")[1].split("\n")[0] for e in ex if e.kind == "subq"]
        self.assertEqual(len(qs), 5)
        self.assertEqual(sum(q.startswith("cot") for q in qs), 3)

    def test_arms_build_expected_examples(self):
        from cotdistill.student import build_examples
        it = toy_item(0, random.Random(0))
        r = random.Random(0)
        self.assertEqual(len(build_examples(it, final="teacher", subq="none", subq_target="fresh", lambda_sub=1, rng=r)), 1)
        ex = build_examples(it, final="none", subq="cot", subq_target="truth", lambda_sub=1, rng=r)
        self.assertEqual(len(ex), 1)
        self.assertAlmostEqual(max(ex[0].target), 1 - 1e-4, places=6)          # truth=True -> ~1 on "Yes"
        self.assertEqual(len(build_examples(it, final="gold", subq="random", subq_target="fresh", lambda_sub=1, rng=r)), 2)

    def test_learns_toy_task(self):
        from cotdistill.student import FINAL_TEMPLATE, Example, build_examples, kl_loss, label_logits
        tok, model = tiny()
        rng = random.Random(0)
        items = [toy_item(i, rng) for i in range(64)]
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        cache = {}

        def acc():
            model.eval()
            with torch.no_grad():
                ex = [Example(FINAL_TEMPLATE.format(problem=it["prompt"]), it["labels"], [0, 0], 1, "final", it["item_id"])
                      for it in items]
                zs = label_logits(model, tok, ex, 256, cache)
            model.train()
            return sum(int(z.argmax()) == it["labels"].index(it["gold_label"]) for z, it in zip(zs, items)) / len(items)

        losses = []
        for epoch in range(12):
            ex = [e for it in items for e in build_examples(it, final="teacher", subq="cot", subq_target="fresh",
                                                           lambda_sub=1.0, rng=rng)]
            rng.shuffle(ex)
            for i in range(0, len(ex), 16):
                b = ex[i:i + 16]
                loss = kl_loss(label_logits(model, tok, b, 256, cache), b)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
        self.assertLess(sum(losses[-8:]) / 8, sum(losses[:8]) / 8)
        self.assertGreater(acc(), 0.9)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE, "needs torch")
class TestPermuteAndBrier(unittest.TestCase):
    PROMPT = "Who is lying?\nOptions:\nA) Ann\nB) Bob\nC) Cy\nD) None of the above"

    def test_target_and_gold_follow_texts(self):
        import random as _r
        from cotdistill.student import permute_options
        labels, target = ["A", "B", "C", "D"], [0.1, 0.7, 0.2, 0.0]
        for seed in range(20):
            out = permute_options(self.PROMPT, labels, target, 1, _r.Random(seed))
            self.assertIsNotNone(out)
            prompt, t, g = out
            lines = prompt.split("\nOptions:\n")[1].split("\n")
            text_of = {l[0]: l[3:] for l in lines}
            self.assertEqual(text_of[labels[g]], "Bob")                         # gold follows its text
            self.assertAlmostEqual(t[labels.index(next(L for L in labels if text_of[L] == "Bob"))], 0.7)
            self.assertEqual(text_of["D"], "None of the above")                 # pinned last
            self.assertAlmostEqual(sum(t), 1.0)

    def test_skips_malformed_and_ordered(self):
        import random as _r
        from cotdistill.student import permute_options
        self.assertIsNone(permute_options("no options here", ["A", "B"], [0.5, 0.5], 0, _r.Random(0)))
        scale = "Rate it.\nOptions:\nA) 1 - poor\nB) 2 - fair\nC) 3 - good"
        self.assertIsNone(permute_options(scale, ["A", "B", "C"], [0.2, 0.3, 0.5], 2, _r.Random(0)))

    def test_build_examples_permutes_final(self):
        import random as _r
        from cotdistill.student import build_examples
        item = {"item_id": "x", "prompt": self.PROMPT, "labels": ["A", "B", "C", "D"], "gold_label": "B",
                "teacher": {"A": 0.1, "B": 0.7, "C": 0.2, "D": 0.0}, "subqs": [], "random_subqs": []}
        moved = 0
        for seed in range(10):
            ex = build_examples(item, final="teacher", subq="none", subq_target="cot", lambda_sub=1.0,
                                rng=_r.Random(seed), permute_final=1.0)[0]
            lines = ex.text.split("\nOptions:\n")[1].split("\n\nAnswer:")[0].split("\n")
            self.assertEqual(lines[ex.gold][3:], "Bob")
            self.assertAlmostEqual(ex.target[ex.gold], 0.7)
            moved += ex.gold != 1
        self.assertGreater(moved, 0)

    def test_brier_zero_at_target_and_positive_otherwise(self):
        from cotdistill.student import Example, brier_loss
        t = [0.2, 0.8]
        e = Example("q", ["A", "B"], t, 1.0, "final", "x")
        z_exact = torch.log(torch.tensor(t))
        self.assertAlmostEqual(brier_loss([z_exact], [e]).item(), 0.0, places=6)
        z = torch.tensor([2.0, -1.0], requires_grad=True)
        b = brier_loss([z], [e])
        b.backward()
        self.assertGreater(b.item(), 0.0)
        self.assertTrue(torch.isfinite(z.grad).all())


@unittest.skipUnless(HAVE, "needs torch")
class TestHiddenDump(unittest.TestCase):
    def test_hidden_in_input_order_and_preds_unchanged(self):
        from cotdistill.student import evaluate
        tok, model = tiny()
        items = [{"item_id": f"i{k}", "prompt": "x " * (k * 7 % 11 + 1) + "\nOptions:\nA) a\nB) b",
                  "labels": ["A", "B"], "gold_label": "A"} for k in range(7)]
        m0, p0 = evaluate(model, tok, items, 256, 3, {})
        hid = []
        m1, p1 = evaluate(model, tok, items, 256, 3, {}, hidden=hid)
        self.assertEqual([p["probs"] for p in p0], [p["probs"] for p in p1])
        self.assertEqual(len(hid), len(items))
        for k in (0, 4):                         # each row is that item's own readout state
            one = []
            evaluate(model, tok, [items[k]], 256, 1, {}, hidden=one)
            self.assertTrue(torch.allclose(one[0], hid[k], atol=1e-4))


@unittest.skipUnless(HAVE, "needs torch")
class TestSubqSampling(unittest.TestCase):
    def test_k_and_weights(self):
        import random as _r
        from cotdistill.student import build_examples
        item = {"item_id": "x", "prompt": "P\nOptions:\nA) a\nB) b", "labels": ["A", "B"], "gold_label": "A",
                "teacher": {"A": 0.9, "B": 0.1},
                "subqs": [{"question": f"q{i}?", "p_cot": 0.8, "p_fresh": 0.6} for i in range(6)],
                "random_subqs": [{"question": f"r{i}?", "p_fresh": 0.3} for i in range(6)]}
        kw = dict(final="teacher", subq_target="cot", lambda_sub=0.5, rng=_r.Random(0))
        all_split = [e for e in build_examples(item, subq="cot", **kw) if e.kind == "subq"]
        self.assertEqual(len(all_split), 6)
        self.assertAlmostEqual(sum(e.weight for e in all_split), 0.5)
        k_each = [e for e in build_examples(item, subq="cot", subq_k=2, subq_weight="each", **kw) if e.kind == "subq"]
        self.assertEqual(len(k_each), 2)
        self.assertTrue(all(abs(e.weight - 0.5) < 1e-9 for e in k_each))
        ctl = [e for e in build_examples(item, subq="random", subq_k=2, subq_weight="each", **kw) if e.kind == "subq"]
        self.assertEqual(len(ctl), 2)                       # the control arm is sampled the same way


@unittest.skipUnless(HAVE, "needs torch")
class TestDepthCurriculum(unittest.TestCase):
    NODES = [{"id": "n1", "question": "a?", "p_cot": 0.9, "depends_on": []},
             {"id": "n2", "question": "b?", "p_cot": 0.2, "depends_on": []},
             {"id": "n3", "question": "c?", "p_cot": 0.8, "depends_on": ["n1"]},
             {"id": "n4", "question": "d?", "p_cot": 0.7, "depends_on": ["n3", "n2"]},
             {"id": "n5", "question": "e?", "p_cot": 0.6, "depends_on": ["n4", "zz"]}]

    def test_node_depths(self):
        from cotdistill.student import node_depths
        self.assertEqual(node_depths(self.NODES), {"n1": 0, "n2": 0, "n3": 1, "n4": 2, "n5": 3})
        cyc = [{"id": "a", "depends_on": ["b"]}, {"id": "b", "depends_on": ["a"]}]
        self.assertTrue(all(isinstance(v, int) for v in node_depths(cyc).values()))

    def test_stages_cover_levels_in_order(self):
        import random as _r
        from cotdistill.student import build_stage_examples
        item = {"item_id": "x", "prompt": "P", "subqs": self.NODES}
        level = {"a?": 0, "b?": 0, "c?": 1, "d?": 2, "e?": 3}
        seen = []
        for s in range(4):                       # 4 levels, 4 stages: stage s may use levels <= s
            qs = set()
            for seed in range(200):
                for e in build_stage_examples(item, stage=s, n_stages=4, k=1, rng=_r.Random(seed)):
                    qs.add(e.text.split("Intermediate question: ")[1].split("\n")[0])
            self.assertLessEqual(max(level[q] for q in qs), s)
            self.assertIn(s, {level[q] for q in qs})          # the new level is actually drawn
            seen.append(qs)
        self.assertEqual(build_stage_examples({"item_id": "y", "prompt": "P", "subqs": []}, stage=0, n_stages=4, k=1,
                                              rng=_r.Random(0)), [])

    def test_train_script_runs_stages_on_cpu(self):
        import json as _j
        import subprocess
        import tempfile
        tok, model = tiny()
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d + "/m"); tok.save_pretrained(d + "/m")
            recs = [{"item_id": f"i{j}", "prompt": "Q\nOptions:\nA) x\nB) y", "labels": ["A", "B"], "gold_label": "A",
                     "teacher": {"A": 0.8, "B": 0.2}, "subqs": self.NODES, "random_subqs": []} for j in range(6)]
            with open(d + "/t.jsonl", "w") as f:
                f.write("\n".join(_j.dumps(r) for r in recs))
            out = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/train_student.py"),
                                  "--model", d + "/m", "--train", d + "/t.jsonl", "--eval", "v=" + d + "/t.jsonl",
                                  "--depth-stages", "2", "--epochs", "1", "--micro-bs", "2", "--grad-accum", "1",
                                  "--max-len", "64", "--precision", "bf16", "--no-grad-ckpt", "--out", d + "/o",
                                  "--log-every", "1"], capture_output=True, text=True, timeout=600)
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            self.assertIn("stage depth0: 6 examples per pass, 3 steps", out.stdout)
            self.assertIn("stage final: 6 examples per pass, 3 steps", out.stdout)
            self.assertIn("steps=9", out.stdout)
            self.assertTrue((Path(d) / "o/metrics.json").exists())


@unittest.skipUnless(HAVE, "needs torch")
class TestItemWeightAndFrac(unittest.TestCase):
    ITEM = {"item_id": "x", "prompt": "P\nOptions:\nA) a\nB) b", "labels": ["A", "B"], "gold_label": "A",
            "teacher": {"A": 0.9, "B": 0.1}, "weight": 1.5,
            "subqs": [{"id": f"n{i}", "question": f"q{i}?", "p_cot": 0.8, "depends_on": []} for i in range(6)],
            "random_subqs": []}

    def test_item_weight_scales_final_and_subqs(self):
        import random as _r
        from cotdistill.student import build_examples, build_stage_examples
        ex = build_examples(self.ITEM, final="teacher", subq="cot", subq_target="cot", lambda_sub=1.0, rng=_r.Random(0))
        self.assertAlmostEqual(next(e for e in ex if e.kind == "final").weight, 1.5)
        self.assertAlmostEqual(sum(e.weight for e in ex if e.kind == "subq"), 1.5)
        st = build_stage_examples(self.ITEM, stage=0, n_stages=2, k=1, rng=_r.Random(0))
        self.assertAlmostEqual(st[0].weight, 1.5)

    def test_subq_frac(self):
        import random as _r
        from cotdistill.student import build_examples
        ex = build_examples(self.ITEM, final="teacher", subq="cot", subq_target="cot", lambda_sub=1.0,
                            rng=_r.Random(0), subq_frac=0.5)
        sub = [e for e in ex if e.kind == "subq"]
        self.assertEqual(len(sub), 3)
        self.assertAlmostEqual(sum(e.weight for e in sub), 1.5)          # split weights still sum to lambda x item weight


@unittest.skipUnless(HAVE, "needs torch")
class TestA9b(unittest.TestCase):
    NODES = TestDepthCurriculum.NODES

    def test_level_balanced_replay(self):
        import collections
        import random as _r
        from cotdistill.student import build_stage_examples
        item = {"item_id": "x", "prompt": "P", "subqs": self.NODES}
        level = {"a?": 0, "b?": 0, "c?": 1, "d?": 2, "e?": 3}
        c = collections.Counter()
        for seed in range(2000):
            for e in build_stage_examples(item, stage=3, n_stages=4, k=1, rng=_r.Random(seed), p_new=0.0,
                                          level_balanced=True):
                c[level[e.text.split("Intermediate question: ")[1].split("\n")[0]]] += 1
        for lv in range(4):                       # each of the 4 levels ~ 1/4, although level 0 has 2 nodes
            self.assertAlmostEqual(c[lv] / 2000, 0.25, delta=0.04)

    def test_train_script_a9b_on_cpu(self):
        import json as _j
        import subprocess
        import tempfile
        tok, model = tiny()
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d + "/m"); tok.save_pretrained(d + "/m")
            recs = [{"item_id": f"i{j}", "prompt": "Q\nOptions:\nA) x\nB) y", "labels": ["A", "B"], "gold_label": "A",
                     "teacher": {"A": 0.8, "B": 0.2}, "subqs": self.NODES, "random_subqs": []} for j in range(6)]
            Path(d + "/t.jsonl").write_text("\n".join(_j.dumps(r) for r in recs))
            out = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/train_student.py"),
                                  "--model", d + "/m", "--train", d + "/t.jsonl", "--eval", "v=" + d + "/t.jsonl",
                                  "--depth-stages", "2", "--final-replay", "1", "--level-balanced", "--reset-optim",
                                  "--epochs", "1", "--micro-bs", "2", "--grad-accum", "1", "--max-len", "64",
                                  "--precision", "bf16", "--no-grad-ckpt", "--out", d + "/o"],
                                 capture_output=True, text=True, timeout=600, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""})
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            self.assertIn("stage final: 12 examples per pass, 6 steps", out.stdout)
            self.assertIn("steps=12", out.stdout)


@unittest.skipUnless(HAVE, "needs torch")
class TestFactInternalization(unittest.TestCase):
    ITEM = {"item_id": "x", "prompt": "Q\nOptions:\nA) x\nB) y", "labels": ["A", "B"], "gold_label": "A",
            "teacher": {"A": 0.8, "B": 0.2}, "random_subqs": [],
            "subqs": [{"id": "n1", "question": "a?", "p_cot": 0.9, "truth": None, "depends_on": []},
                      {"id": "n2", "question": "b?", "p_cot": 0.5, "truth": None, "depends_on": []},       # unsure
                      {"id": "n3", "question": "c?", "p_cot": 0.9, "truth": False, "depends_on": ["n1"]},  # truth wins
                      {"id": "n4", "question": "d?", "p_cot": 0.1, "truth": None, "depends_on": ["n3"]}]}

    def test_fact_block(self):
        from cotdistill.student import fact_block
        fb = fact_block(self.ITEM)
        self.assertEqual(fb.split("\n")[3:], ["Q: a? A: Yes", "Q: c? A: No", "Q: d? A: No"])
        self.assertEqual(fact_block(self.ITEM, min_depth=2).split("\n")[3:], ["Q: d? A: No"])

    def test_stages_drop_shallow_first(self):
        from cotdistill.student import build_fact_examples
        texts = [build_fact_examples(self.ITEM, stage=s, n_stages=3)[0].text for s in range(3)]
        self.assertIn("a?", texts[0]); self.assertIn("d?", texts[0])
        self.assertNotIn("a?", texts[1]); self.assertIn("d?", texts[1])
        self.assertNotIn("Known intermediate results", build_fact_examples(self.ITEM, stage=3, n_stages=3)[0].text)
        self.assertTrue(texts[0].endswith("\n\nAnswer:"))

    def test_train_script_fact_modes_on_cpu(self):
        import json as _j
        import os
        import subprocess
        import tempfile
        tok, model = tiny()
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d + "/m"); tok.save_pretrained(d + "/m")
            Path(d + "/t.jsonl").write_text("\n".join(_j.dumps(dict(self.ITEM, item_id=f"i{j}")) for j in range(6)))
            for extra, want in ((["--facts-always"], "steps=6"), (["--fact-stages", "2"], "steps=12")):
                out = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/train_student.py"),
                                      "--model", d + "/m", "--train", d + "/t.jsonl", "--eval", "v=" + d + "/t.jsonl",
                                      "--epochs", "2", "--micro-bs", "2", "--grad-accum", "1", "--max-len", "64",
                                      "--precision", "bf16", "--no-grad-ckpt", "--out", d + "/o"] + extra,
                                     capture_output=True, text=True, timeout=600, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
                self.assertEqual(out.returncode, 0, out.stderr[-2000:])
                self.assertIn(want, out.stdout, out.stdout[-800:])


try:
    import peft  # noqa: F401
    HAVE_PEFT = True
except ImportError:
    HAVE_PEFT = False


@unittest.skipUnless(HAVE and HAVE_PEFT, "needs torch and peft")
class TestLoRA(unittest.TestCase):
    def test_lora_trains_adapters_and_saves_merged(self):
        import json as _j
        import os
        import subprocess
        import tempfile
        from cotdistill.student import load_model
        tok, model = tiny()
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d + "/m"); tok.save_pretrained(d + "/m")
            recs = [{"item_id": f"i{j}", "prompt": "Q\nOptions:\nA) x\nB) y", "labels": ["A", "B"], "gold_label": "A",
                     "teacher": {"A": 0.9, "B": 0.1}, "subqs": [], "random_subqs": []} for j in range(8)]
            Path(d + "/t.jsonl").write_text("\n".join(_j.dumps(r) for r in recs))
            out = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/train_student.py"),
                                  "--model", d + "/m", "--train", d + "/t.jsonl", "--eval", "v=" + d + "/t.jsonl",
                                  "--lora-r", "4", "--lr", "1e-2", "--epochs", "3", "--micro-bs", "2", "--grad-accum", "1",
                                  "--max-len", "64", "--precision", "bf16", "--out", d + "/o", "--save"],
                                 capture_output=True, text=True, timeout=600,
                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            self.assertIn("lora r=4", out.stdout)
            n_trainable = float(out.stdout.split("params=")[1].split("B")[0])
            self.assertLess(n_trainable, 0.001)                     # adapters only
            self.assertTrue((Path(d) / "o/adapter/adapter_config.json").exists())
            merged = load_model(d + "/o/model", torch.float32)       # the merged checkpoint loads as a plain model
            base = load_model(d + "/m", torch.float32)
            diff = sum((a - b).abs().sum().item() for a, b in zip(merged.parameters(), base.parameters()))
            self.assertGreater(diff, 0.0)                            # training changed the merged weights


@unittest.skipUnless(HAVE, "needs torch")
class TestGroupedAux(unittest.TestCase):
    ITEM = {"item_id": "x", "prompt": "Q\nOptions:\nA) x\nB) y", "labels": ["A", "B"], "gold_label": "A",
            "teacher": {"A": 0.8, "B": 0.2},
            "subqs": [{"id": f"n{i}", "question": f"c{i}?", "p_cot": 0.9, "depends_on": []} for i in range(4)],
            "random_subqs": [{"question": f"r{i}?", "p_fresh": 0.2} for i in range(4)]}

    def test_group_shapes_and_weights(self):
        import random as _r
        from cotdistill.student import build_group
        for kind, want_q, want_w in (("cot", "c", 0.5), ("random", "r", 0.5), ("placebo", "c", 0.0)):
            g = build_group(self.ITEM, aux_kind=kind, k=2, aux_weight=0.5, rng=_r.Random(0))
            self.assertEqual([e.kind for e in g], ["final", "subq", "subq"])
            self.assertEqual(g[0].weight, 1.0)
            for e in g[1:]:
                self.assertAlmostEqual(e.weight, want_w)
                self.assertIn(f"Intermediate question: {want_q}", e.text)
        # same problems and same CoT views in placebo and cot arms (same rng seed)
        a = build_group(self.ITEM, aux_kind="cot", k=2, aux_weight=0.5, rng=_r.Random(3))
        b = build_group(self.ITEM, aux_kind="placebo", k=2, aux_weight=0.5, rng=_r.Random(3))
        self.assertEqual([e.text for e in a], [e.text for e in b])

    def test_fill_with_zero_weight_copies(self):
        import random as _r
        from cotdistill.student import build_group
        g = build_group(dict(self.ITEM, subqs=self.ITEM["subqs"][:1]), aux_kind="cot", k=2, aux_weight=0.5, rng=_r.Random(0))
        self.assertEqual(len(g), 3)
        self.assertEqual((g[2].kind, g[2].weight), ("final", 0.0))

    def test_grouped_training_on_cpu(self):
        import json as _j
        import os
        import subprocess
        import tempfile
        tok, model = tiny()
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d + "/m"); tok.save_pretrained(d + "/m")
            Path(d + "/t.jsonl").write_text("\n".join(_j.dumps(dict(self.ITEM, item_id=f"i{j}")) for j in range(6)))
            for kind in ("cot", "random", "placebo"):
                out = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/train_student.py"),
                                      "--model", d + "/m", "--train", d + "/t.jsonl", "--eval", "v=" + d + "/t.jsonl",
                                      "--grouped-aux", "2", "--aux-kind", kind, "--items-per-update", "2", "--epochs", "1",
                                      "--micro-bs", "2", "--max-len", "64", "--precision", "bf16", "--no-grad-ckpt",
                                      "--out", d + "/o"], capture_output=True, text=True, timeout=600,
                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
                self.assertEqual(out.returncode, 0, out.stderr[-2000:])
                self.assertIn("stage grouped: 18 examples per pass, 3 steps", out.stdout)


@unittest.skipUnless(HAVE, "needs torch")
class TestLayerDump(unittest.TestCase):
    def test_layer_stack_shape_and_last_entry(self):
        from cotdistill.student import evaluate
        tok, model = tiny()
        items = [{"item_id": f"i{k}", "prompt": "x " * (k + 1) + "\nOptions:\nA) a\nB) b", "labels": ["A", "B"],
                  "gold_label": "A"} for k in range(5)]
        plain, multi = [], []
        _, p0 = evaluate(model, tok, items, 128, 2, {}, hidden=plain)
        _, p1 = evaluate(model, tok, items, 128, 2, {}, hidden=multi, layers=[0, 1])
        self.assertEqual([p["probs"] for p in p0], [p["probs"] for p in p1])
        self.assertEqual(tuple(multi[0].shape), (3, model.config.hidden_size))
        for a, b in zip(plain, multi):
            self.assertTrue(torch.allclose(a, b[-1], atol=1e-5))


@unittest.skipUnless(HAVE, "needs torch")
class TestPlaceboGradient(unittest.TestCase):
    def test_placebo_gradient_is_final_only_gradient_over_group_size(self):
        """G0 (zero-weight auxiliary views) must give exactly the final-only gradient scaled by 1/(1+K), the factor
        coming from the loss denominator that counts every view in the update (train_student's normalisation)."""
        import random as _r
        from cotdistill.student import build_group, kl_loss, label_logits
        tok, model = tiny()
        model.eval()                                  # no dropout: both passes see identical arithmetic
        item = {"item_id": "x", "prompt": "Q one two\nOptions:\nA) x\nB) y", "labels": ["A", "B"], "gold_label": "A",
                "teacher": {"A": 0.8, "B": 0.2},
                "subqs": [{"id": f"n{i}", "question": f"is c{i} true?", "p_cot": 0.9, "depends_on": []} for i in range(3)],
                "random_subqs": []}
        group = build_group(item, aux_kind="placebo", k=2, aux_weight=0.5, rng=_r.Random(0))
        def grads(batch):
            model.zero_grad()
            z = label_logits(model, tok, batch, 64, {})
            (kl_loss(z, batch) * len(batch) / len(batch)).backward()     # total / len(batch), grad_accum = 1
            return [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
        g_group, g_final = grads(group), grads(group[:1])
        self.assertEqual(len(g_group), len(g_final))
        for a, b in zip(g_group, g_final):
            self.assertTrue(torch.allclose(a, b / 3, atol=1e-6, rtol=1e-4))


@unittest.skipUnless(HAVE, "needs torch")
class TestTreeTransitions(unittest.TestCase):
    # n1 n2 -> n3 -> n5 ; n4 independent ; n6 depends on n4
    NODES = [{"id": "n1", "question": "q1?", "p_cot": 0.9, "depends_on": []},
             {"id": "n2", "question": "q2?", "p_cot": 0.1, "depends_on": []},
             {"id": "n3", "question": "q3?", "p_cot": 0.9, "depends_on": ["n1", "n2"]},
             {"id": "n4", "question": "q4?", "p_cot": 0.8, "depends_on": []},
             {"id": "n5", "question": "q5?", "p_cot": 0.2, "depends_on": ["n3"]},
             {"id": "n6", "question": "q6?", "p_cot": 0.9, "depends_on": ["n4"]}]
    ITEM = {"item_id": "x", "prompt": "P", "labels": ["A", "B"], "gold_label": "A", "teacher": {"A": 0.7, "B": 0.3},
            "subqs": NODES, "random_subqs": []}

    def parse(self, e):
        head, q = e.text.split("\n\nIntermediate question: ")
        facts = [l.split(" A: ")[0][3:] for l in head.split("\n") if l.startswith("Q: ")]
        return q.split("\n")[0], facts

    def test_true_parents_and_shuffled_control(self):
        import random as _r
        from cotdistill.student import build_transition_views
        par = {"q3?": {"q1?", "q2?"}, "q5?": {"q3?"}, "q6?": {"q4?"}}
        desc = {"q3?": {"q5?"}, "q5?": set(), "q6?": set()}
        for seed in range(30):
            for e in build_transition_views(self.ITEM, k=2, rng=_r.Random(seed)):
                q, facts = self.parse(e)
                self.assertIn(q, par)                                  # nodes with parents are drawn first
                self.assertEqual(set(facts), par[q])
            for e in build_transition_views(self.ITEM, k=2, rng=_r.Random(seed), shuffled=True):
                q, facts = self.parse(e)
                self.assertTrue(len(facts) <= len(par[q]) and facts)
                self.assertFalse(set(facts) & (par[q] | desc[q] | {q}))
        none = build_transition_views(self.ITEM, k=2, rng=_r.Random(0), keep_prob=0.0)
        self.assertTrue(all("Known intermediate results" not in e.text for e in none))

    def test_group_and_training_on_cpu(self):
        import json as _j
        import os
        import random as _r
        import subprocess
        import tempfile
        from cotdistill.student import build_group
        g = build_group(self.ITEM, aux_kind="tree", k=2, aux_weight=0.5, rng=_r.Random(0))
        self.assertEqual([e.kind for e in g], ["final", "subq", "subq"])
        self.assertNotIn("Known intermediate results", g[0].text)        # the final question never gets facts
        tok, model = tiny()
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d + "/m"); tok.save_pretrained(d + "/m")
            Path(d + "/t.jsonl").write_text("\n".join(_j.dumps(dict(self.ITEM, item_id=f"i{j}")) for j in range(6)))
            for kind, extra in (("tree", ["--fact-withdraw", "0.75"]), ("tree_shuf", [])):
                out = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/train_student.py"),
                                      "--model", d + "/m", "--train", d + "/t.jsonl", "--eval", "v=" + d + "/t.jsonl",
                                      "--grouped-aux", "2", "--aux-kind", kind, "--items-per-update", "2", "--epochs", "2",
                                      "--micro-bs", "2", "--max-len", "96", "--precision", "bf16", "--no-grad-ckpt",
                                      "--out", d + "/o"] + extra, capture_output=True, text=True, timeout=600,
                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
                self.assertEqual(out.returncode, 0, out.stderr[-2000:])
                self.assertIn("steps=6", out.stdout)


@unittest.skipUnless(HAVE, "needs torch")
class TestEvalAt(unittest.TestCase):
    def test_trajectory_points(self):
        import json as _j
        import os
        import subprocess
        import tempfile
        tok, model = tiny()
        item = {"prompt": "Q one two\nOptions:\nA) x\nB) y", "labels": ["A", "B"], "gold_label": "A",
                "teacher": {"A": 0.8, "B": 0.2}, "subqs": [], "random_subqs": []}
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d + "/m"); tok.save_pretrained(d + "/m")
            Path(d + "/t.jsonl").write_text("\n".join(_j.dumps(dict(item, item_id=f"i{j}")) for j in range(8)))
            out = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/train_student.py"),
                                  "--model", d + "/m", "--train", d + "/t.jsonl", "--eval", "v=" + d + "/t.jsonl",
                                  "w=" + d + "/t.jsonl", "--epochs", "4", "--micro-bs", "2", "--grad-accum", "2",
                                  "--eval-at", "0.5,1,4", "--eval-mid", "v", "--eval-mid-max", "3", "--max-len", "96",
                                  "--precision", "bf16", "--no-grad-ckpt", "--out", d + "/o"],
                                 capture_output=True, text=True, timeout=600, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            traj = _j.loads(Path(d + "/o/metrics.json").read_text())["trajectory"]
            self.assertEqual([r["step"] for r in traj], [1, 2, 8])          # 8 items / 4 per step = 2 steps per epoch
            self.assertEqual([r["epoch"] for r in traj], [0.5, 1.0, 4.0])
            self.assertEqual(set(traj[0]) - {"step", "epoch", "s_elapsed"}, {"v"})
            self.assertEqual(traj[0]["v"].keys() >= {"acc", "nll", "ece"}, True)


@unittest.skipUnless(HAVE, "needs torch")
class TestAuxSources(unittest.TestCase):
    def test_inactive_items_get_zero_weight_views(self):
        import random as _r
        from cotdistill.student import build_group
        item = {"item_id": "x", "source": "sharc", "prompt": "P\nOptions:\nA) x\nB) y", "labels": ["A", "B"],
                "gold_label": "A", "teacher": {"A": 0.7, "B": 0.3},
                "subqs": [{"id": f"n{i}", "question": f"q{i}?", "p_cot": 0.9, "depends_on": []} for i in range(3)],
                "random_subqs": []}
        on = build_group(item, aux_kind="cot", k=2, aux_weight=0.5, rng=_r.Random(0))
        off = build_group(item, aux_kind="cot", k=2, aux_weight=0.5, rng=_r.Random(0), aux_active=False)
        self.assertEqual([e.text for e in on], [e.text for e in off])          # same views, same work
        self.assertEqual([e.weight for e in on][1:], [0.5, 0.5])
        self.assertEqual([e.weight for e in off][1:], [0.0, 0.0])
        self.assertEqual(on[0].weight, off[0].weight)


@unittest.skipUnless(HAVE, "needs torch")
class TestFullTree(unittest.TestCase):
    # n1 n2 -> n3 -> n5 ; n4 -> n6 ; n7 depends on n5 and is the near-answer sink at max depth
    NODES = [{"id": "n1", "question": "q1?", "p_cot": 0.9, "depends_on": []},
             {"id": "n2", "question": "q2?", "p_cot": 0.1, "depends_on": []},
             {"id": "n3", "question": "q3?", "p_cot": 0.9, "depends_on": ["n1", "n2"]},
             {"id": "n4", "question": "q4?", "p_cot": 0.8, "depends_on": []},
             {"id": "n5", "question": "q5?", "p_cot": 0.2, "depends_on": ["n3"]},
             {"id": "n6", "question": "q6?", "p_cot": 0.9, "depends_on": ["n4"]},
             {"id": "n7", "question": "q7?", "p_cot": 0.95, "depends_on": ["n5"]}]
    ITEM = {"item_id": "x", "prompt": "P\nOptions:\nA) a\nB) b\nC) c", "labels": ["A", "B", "C"], "gold_label": "A",
            "teacher": {"A": 0.7, "B": 0.2, "C": 0.1}, "subqs": NODES,
            "random_subqs": [{"question": f"c{i}?", "p_fresh": 0.3} for i in range(4)]}

    def facts_of(self, e):
        head = e.text.split("\n\nIntermediate question: ")[0]
        q = e.text.split("\n\nIntermediate question: ")[1].split("\n")[0]
        return q, [l.split(" A: ")[0][3:] for l in head.split("\n") if l.startswith("Q: ")]

    def test_plan_invariants(self):
        import random as _r
        from cotdistill.student import full_tree_plan, node_depths
        depth = node_depths(self.NODES)
        desc = {"n1": {"n3", "n5", "n7"}, "n2": {"n3", "n5", "n7"}, "n3": {"n5", "n7"}, "n4": {"n6"}, "n5": {"n7"},
                "n6": set(), "n7": set()}
        for seed in range(30):
            plan = full_tree_plan(self.ITEM, cap=10, rng=_r.Random(seed))
            self.assertEqual(len(plan), 7)
            for e in plan:
                i, par, m = e["node"]["id"], e["parents"], e["matched"]
                self.assertEqual(len(par), len(m))
                self.assertFalse(set(m) & ({i} | set(par) | desc[i] | {"n7"}))   # n7: near-answer sink
            n3 = [e for e in plan if e["node"]["id"] == "n3"][0]
            self.assertEqual(sorted(n3["parents"]), ["n1", "n2"])
            self.assertEqual(set(n3["matched"]), {"n4", "n6"})                  # the only unrelated nodes
            n6 = [e for e in plan if e["node"]["id"] == "n6"][0]                 # parent n4 has depth 0
            self.assertEqual(len(n6["matched"]), 1)
            self.assertEqual(depth[n6["matched"][0]], 0)                         # depth-matched replacement

    def test_modes_state_equal_counts(self):
        import random as _r
        from cotdistill.student import build_full_tree
        for seed in range(10):
            t = build_full_tree(self.ITEM, mode="true", rng=_r.Random(seed))
            m = build_full_tree(self.ITEM, mode="matched", rng=_r.Random(seed))
            self.assertEqual(len(t), 8)                                       # final + 7 nodes
            ct = sorted((q, len(f)) for q, f in map(self.facts_of, t[1:]))
            cm = sorted((q, len(f)) for q, f in map(self.facts_of, m[1:]))
            self.assertEqual(ct, cm)
            self.assertAlmostEqual(sum(e.weight for e in t[1:]), 1.0)
        pl = build_full_tree(self.ITEM, mode="placebo", rng=_r.Random(0))
        self.assertTrue(all(e.weight == 0 for e in pl[1:]) and pl[0].weight > 0)
        self.assertTrue(all("Known intermediate" not in e.text for e in pl))
        ctl = build_full_tree(self.ITEM, mode="control", rng=_r.Random(0))
        self.assertEqual(len(ctl), 5)

    def test_training_on_cpu(self):
        import json as _j
        import os
        import subprocess
        import tempfile
        tok, model = tiny()
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d + "/m"); tok.save_pretrained(d + "/m")
            Path(d + "/t.jsonl").write_text("\n".join(_j.dumps(dict(self.ITEM, item_id=f"i{j}")) for j in range(5)))
            for mode in ("true", "matched"):
                out = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/train_student.py"),
                                      "--model", d + "/m", "--train", d + "/t.jsonl", "--eval", "v=" + d + "/t.jsonl",
                                      "--tree-full", mode, "--items-per-update", "2", "--epochs", "2", "--micro-bs", "3",
                                      "--max-len", "128", "--precision", "bf16", "--no-grad-ckpt", "--log-every", "1", "--out", d + "/o"],
                                     capture_output=True, text=True, timeout=600,
                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
                self.assertEqual(out.returncode, 0, out.stderr[-2000:])
                self.assertIn("steps=6", out.stdout)                          # ceil(5/2) = 3 updates per pass
                self.assertIn('"seqs": 16', out.stdout)                         # 2 problems x (final + 7 nodes)


@unittest.skipUnless(HAVE, "needs torch")
class TestMCViews(unittest.TestCase):
    NODES = [{"id": f"n{i}", "question": f"q{i}?", "p_cot": p, "depends_on": []}
             for i, p in enumerate([0.95, 0.9, 0.05, 0.1, 0.02, 0.2, 0.5, 0.08])]
    ITEM = {"item_id": "x", "prompt": "P\nOptions:\nA) a\nB) b\nC) c", "labels": ["A", "B", "C"], "gold_label": "A",
            "teacher": {"A": 0.7, "B": 0.2, "C": 0.1}, "subqs": NODES,
            "random_subqs": [{"question": f"c{i}?", "p_fresh": p} for i, p in enumerate([0.9, 0.1, 0.1, 0.2])]}

    def test_views(self):
        import random as _r
        from cotdistill.student import build_group, build_mc_views
        p = {n["question"]: n["p_cot"] for n in self.NODES}
        for seed in range(40):
            views = build_mc_views(self.ITEM, k=2, rng=_r.Random(seed))
            self.assertEqual(len(views), 2)                                      # 2 Yes + 5 No confident nodes
            used = []
            for v in views:
                opts = v.text.split("Options:\n")[-1].split("\nAnswer:")[0].split("\n")
                self.assertEqual(len(opts), 4)
                self.assertTrue(opts[-1].endswith("None of them"))
                qs = [o[3:] for o in opts[:-1]]
                used += qs
                self.assertNotIn("q6?", qs)                                      # p = .5 is never used
                self.assertLessEqual(sum(p[q] >= 0.7 for q in qs), 1)
                self.assertAlmostEqual(sum(v.target), 1.0, places=5)
                best = max(range(4), key=v.target.__getitem__)
                gold = [i for i, q in enumerate(qs) if p[q] >= 0.7]
                self.assertEqual(best, gold[0] if gold else 3)
            self.assertEqual(len(used), len(set(used)))                          # disjoint questions
        ctl = build_mc_views(self.ITEM, k=2, rng=_r.Random(0), kind="random")
        self.assertEqual(len(ctl), 1)                                            # 4 control questions: one view
        for kind in ("cot_mc", "random_mc"):                                   # both arms: min of the capacities
            g = build_group(self.ITEM, aux_kind=kind, k=2, aux_weight=0.5, rng=_r.Random(0))
            self.assertEqual([e.weight for e in g][1:], [0.5, 0.0])              # padded with a zero-weight filler


@unittest.skipUnless(HAVE, "needs torch")
class TestCommonOrder(unittest.TestCase):
    def test_arms_share_problem_order_across_epochs(self):
        import random as _r
        from cotdistill.student import build_full_tree
        items = [dict(TestFullTree.ITEM, item_id=f"i{j}") for j in range(12)]
        def order(mode):
            order_rng, rng, out = _r.Random("order-0"), _r.Random(0), []
            for _ in range(2):                                   # two epochs, as fulltree_updates does
                its = items[:]
                order_rng.shuffle(its)
                for it in its:
                    build_full_tree(it, mode=mode, rng=rng)       # consumes rng differently per mode
                out.append([it["item_id"] for it in its])
            return out
        self.assertEqual(order("true"), order("matched"))
