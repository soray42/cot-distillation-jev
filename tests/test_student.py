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
