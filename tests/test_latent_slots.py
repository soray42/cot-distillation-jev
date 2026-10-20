"""Latent node slots (Tree Forcing v2, design (a); train_student.py --latent-slots, arms LS / LSU / LSR / LSY):
- the slot line: K x c copies of an existing single token, before the options (or after them), the same text in the
  training final view (slot_group) and in every evaluation (evaluate via final_text, eval_student via the model's
  latent_slots.json); the token choice (Qwen3.5: <|box_start|>, swap <|quad_start|>; GPT-2 needs an explicit token);
- slot targets: gsm8k-train-660's six worked steps 4 + 2 = 6 ... 18 + 2 = 20 on slots 1-6 (K = 6), shorter chains leave
  the last slots unsupervised, longer ones merge their tail into slot K (or truncate), no chain = no target; LSR's
  targets come from other problems at the same supervised slots, fixed by --seed and the item;
- slot_group: LS / LSR final views only (SlotExample, lambda / supervised slots), LSU plain Examples (no decoding),
  LSY = GFPQ's node views unchanged, no draw from the training rng;
- slot_forward: the letter logits of label_logits, and a decoding loss whose gradient reaches the main pass (embedding
  rows of problem-only tokens, also with gradient checkpointing; zero when the slot states are detached);
- the training loop on a tiny random model: LS / LSU / LSR / LSY end to end (slot loss logged for LS / LSR / LSY only,
  LSY in step with GFPQ, LS / LSR the same supervised slots), the necessity test (swap / drop / noise) and the decoding
  readout, eval_student on the saved model reproducing the final evaluation, resume, the argument checks;
- the existing arms (GFPQ, A2, SIB, eval_student) give LS_BASE_REF's files, and LS / LSY under --fsdp (2 gloo ranks
  and 1 rank, CPU) give the single-process updates.
v2.1 (lens readout; arms LSL / LSLU / LSLR / LSLY):
- the v2 slot arms (LS / LSU / LSR / LSY with necessity + decoding, eval_student) give LS21_BASE_REF's files;
- distinct slot tokens (prompt, ids, the merge check; Qwen3.5: the six grounding tokens, swap <|fim_pad|>), value
  targets with fill / overflow (gsm8k-train-660: 6, 8, 2, 10, 18, 20), the LSLR shuffle, --slot-layer resolution on
  32 / 24-layer Qwen3.5 configs (0.67 -> 20 / 15);
- the layer-L capture equals output_hidden_states[L + 1] (Qwen2 and a hybrid Qwen3.5) and carries the gradient with and
  without gradient checkpointing (reentrant too); the lens loss equals a manual computation; one problem overfits to
  a 100% lens readout;
- mean-ablation (own means = intact, other means change the readout), the lens readout and the pilot summary; LSL /
  LSLU / LSLR / LSLY end to end with eval_student on the saved LSL; LSL under --fsdp (2 and 1 ranks) = single process.
CPU only (CUDA_VISIBLE_DEVICES="")."""
import dataclasses
import glob
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

FX = json.loads((ROOT / "tests/fixtures/gsm8k_si_items_v1.json").read_text())
ITEMS = FX["gsm8k"]
GSM = {it["item_id"]: it for it in ITEMS}
# The last commit before the latent slots: the existing arms are compared with it.
LS_BASE_REF = os.environ.get("LS_BASE_REF", "c0b31c6")
# The last commit before latent slots v2.1: the v2 slot arms are compared with it.
LS21_BASE_REF = os.environ.get("LS21_BASE_REF", "693ca28")
VALUES_660 = ["6", "8", "2", "10", "18", "20"]
DIST = ("§", "©", "®", "°", "±", "¢")      # GPT-2: single tokens, runs and neighbours do not merge, absent from prompts
LINES_660 = ["4 + 2 = 6", "4 × 2 = 8", "4 ÷ 2 = 2", "4 + 6 = 10", "10 + 8 = 18", "18 + 2 = 20"]
TOK = "§"          # GPT-2: one token (16273), absent from the fixture prompts
SWAP = "¶"         # GPT-2: one token (26604)
QWEN = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--sora42y--Sev-2B-preview/snapshots/*/"))

try:
    import torch
    import transformers
    HAVE = True
except ImportError:
    HAVE = False


def gpt2():
    tok = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    tok.pad_token = tok.eos_token
    return tok


def gfpq_group(it, seed=0):
    from cotdistill.student import build_full_tree
    return build_full_tree(it, mode="plain", rng=random.Random(seed), cap=0, view="statements+question",
                           final="teacher", subq_target="cot", sources={"gsm8k_tf"})


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestPrompt(unittest.TestCase):
    def setUp(self):
        from cotdistill.student import set_slot_prompt
        self.tok = gpt2()
        self.addCleanup(set_slot_prompt, self.tok, None)

    def test_slot_line(self):
        from cotdistill.student import SlotSpec, slot_prompt, slot_text
        p = GSM["gsm8k-train-660"]["prompt"]
        head, _, opts = p.partition("\nOptions:\n")
        s = SlotSpec(6, 2, TOK)
        self.assertEqual(slot_prompt(p, s), head + "\n" + TOK * 12 + "\nOptions:\n" + opts)
        self.assertEqual(slot_prompt(p, dataclasses.replace(s, place="after-options")), p + "\n" + TOK * 12)
        self.assertEqual(slot_prompt("x < y?", s), "x < y?\n" + TOK * 12)              # no options block: at the end
        self.assertEqual(slot_text(p + "\n\nAnswer:", s), slot_prompt(p, s) + "\n\nAnswer:")
        with self.assertRaises(ValueError):
            slot_text(p, s)
        with self.assertRaises(ValueError):
            slot_prompt(p, dataclasses.replace(s, place="middle"))

    def test_token_choice(self):
        from cotdistill.student import choose_slot_token, slot_token_id
        self.assertEqual(slot_token_id(self.tok, TOK), 16273)
        self.assertEqual(choose_slot_token(self.tok, TOK), TOK)
        with self.assertRaises(ValueError):
            choose_slot_token(self.tok)                          # GPT-2 has no reserved token: explicit token needed
        for bad in ("<|endoftext|>", "Answer:", "§x"):
            with self.assertRaises(ValueError):
                slot_token_id(self.tok, bad)

    @unittest.skipUnless(QWEN, "no local Qwen3.5 tokenizer (Sev-2B-preview snapshot)")
    def test_qwen_tokens_and_660(self):
        """Qwen3.5: <|box_start|> (248049) is the slot, <|quad_start|> (248051) the swap token; the 660 final
        question keeps the problem, option and cue tokens of A2 around the K x c slot ids."""
        from cotdistill.student import FINAL_TEMPLATE, SlotSpec, choose_slot_token, final_text, set_slot_prompt
        tok = transformers.AutoTokenizer.from_pretrained(QWEN[0])
        t = choose_slot_token(tok)
        self.assertEqual((t, tok.convert_tokens_to_ids(t)), ("<|box_start|>", 248049))
        sw = choose_slot_token(tok, avoid=(t,))
        self.assertEqual((sw, tok.convert_tokens_to_ids(sw)), ("<|quad_start|>", 248051))
        p = GSM["gsm8k-train-660"]["prompt"]
        a2 = tok.encode(FINAL_TEMPLATE.format(problem=p), add_special_tokens=False)
        for place in ("before-options", "after-options"):
            for c in (1, 2, 4):
                set_slot_prompt(tok, SlotSpec(6, c, t, place))
                ids = tok.encode(final_text(tok, p), add_special_tokens=False)
                self.assertEqual(ids.count(248049), 6 * c)
                rest = [i for i in ids if i != 248049]
                nl = tok.encode("\n", add_special_tokens=False)
                # the slot line adds one "\n" and the slots, nothing else changes
                self.assertEqual(len(rest), len(a2) + len(nl))
        set_slot_prompt(tok, None)

    def test_train_equals_eval_text(self):
        """The final view of every slot arm and the Example evaluate builds (FSDP or not) have the same text; without a
        registered spec evaluate builds FINAL_TEMPLATE exactly."""
        import cotdistill.student as S
        seen = []

        def fake(model, tok, batch, *a, **k):
            seen.extend(e.text for e in batch)
            return [torch.zeros(len(e.labels)) for e in batch]
        model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
            vocab_size=len(self.tok), hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1))
        orig = S.label_logits
        S.label_logits = fake
        try:
            S.evaluate(model, self.tok, ITEMS, 640, 4, {})
            self.assertEqual(sorted(seen), sorted(S.FINAL_TEMPLATE.format(problem=it["prompt"]) for it in ITEMS))
            for place in S.SLOT_PLACES:
                spec = S.SlotSpec(6, 2, TOK, place)
                S.set_slot_prompt(self.tok, spec)
                seen.clear()
                S.evaluate(model, self.tok, ITEMS, 640, 4, {})
                train = [e.text for it in ITEMS for e in S.slot_group(gfpq_group(it), it, spec=spec,
                                                                        targets=S.slot_targets(it, 6), final_only=True)]
                self.assertEqual(sorted(seen), sorted(train))
                self.assertEqual(train[0], S.final_text(self.tok, ITEMS[0]["prompt"]))
                self.assertTrue(all(t.count(TOK) == 12 for t in train))
        finally:
            S.label_logits = orig


class TestTargets(unittest.TestCase):
    def test_660_alignment(self):
        from cotdistill.student import slot_targets
        it = GSM["gsm8k-train-660"]
        self.assertEqual(slot_targets(it, 6), LINES_660)                 # slot j = node j, K = 6 covers the chain
        self.assertEqual(slot_targets(it, 8), LINES_660 + [None, None])
        self.assertEqual(slot_targets(it, 4), LINES_660[:3] + ["4 + 6 = 10; 10 + 8 = 18; 18 + 2 = 20"])
        self.assertEqual(slot_targets(it, 4, "truncate"), LINES_660[:4])
        self.assertEqual(slot_targets(GSM["gsm8k-train-3365"], 6), ["3 × 20 = 60", "10% of 60 = 6", "60 + 6 = 66",
                                                                     None, None, None])
        self.assertEqual(slot_targets(GSM["gsm8k-train-5699"], 6), [None] * 6)         # no chain: no target
        long = [s["text"] for s in GSM["gsm8k-train-6669"]["worked_steps"]]
        self.assertEqual(len(long), 8)
        self.assertEqual(slot_targets(GSM["gsm8k-train-6669"], 6), long[:5] + ["; ".join(long[5:])])
        with self.assertRaises(ValueError):
            slot_targets(it, 6, "drop")

    def test_shuffled(self):
        """LSR: the same supervised slots, texts of other problems only, keyed by seed and item (not the training rng),
        identical over calls (epochs)."""
        from cotdistill.student import shuffled_slot_targets, slot_targets
        state = random.getstate()
        a = shuffled_slot_targets(ITEMS, k=6, seed=3)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(a, shuffled_slot_targets(ITEMS, k=6, seed=3))
        self.assertNotEqual(a, shuffled_slot_targets(ITEMS, k=6, seed=4))
        real = {it["item_id"]: slot_targets(it, 6) for it in ITEMS}
        for iid, ts in a.items():
            self.assertEqual([t is None for t in ts], [t is None for t in real[iid]])
            own = set(real[iid])
            others = {t for j, r in real.items() if j != iid for t in r if t is not None}
            for t in ts:
                if t is not None:
                    self.assertNotIn(t, own)
                    self.assertIn(t, others)
        self.assertEqual(sum(t is not None for ts in a.values() for t in ts), 21)

    def test_slot_group(self):
        from cotdistill.student import Example, SlotExample, SlotSpec, slot_group, slot_targets
        spec = SlotSpec(6, 1, TOK)
        it = GSM["gsm8k-train-660"]
        g = gfpq_group(it)
        self.assertGreater(len(g), 1)
        st = {}
        ls = slot_group(g, it, spec=spec, targets=slot_targets(it, 6), lam=1.0, final_only=True, stats=st)
        self.assertEqual(len(ls), 1)
        e = ls[0]
        self.assertIsInstance(e, SlotExample)
        self.assertEqual(list(e.slot_targets), LINES_660)
        self.assertAlmostEqual(e.slot_lambda, 1 / 6)
        self.assertEqual((e.labels, e.target, e.weight, e.kind, e.gold), (g[0].labels, g[0].target, g[0].weight,
                                                                           "final", g[0].gold))
        self.assertEqual(st, {"slot_items": 1, "slot_supervised": 6, "slot_items_without_chain": 0,
                              "slot_items_merged": 0})
        lsy = slot_group(g, it, spec=spec, targets=slot_targets(it, 6), lam=0.5)
        self.assertEqual([dataclasses.astuple(x) for x in lsy[1:]], [dataclasses.astuple(x) for x in g[1:]])
        self.assertAlmostEqual(lsy[0].slot_lambda, 0.5 / 6)
        lsu = slot_group(g, it, spec=spec, targets=None, final_only=True)
        self.assertIs(type(lsu[0]), Example)
        self.assertEqual(lsu[0].text, e.text)
        none = slot_group(gfpq_group(GSM["gsm8k-train-5699"]), GSM["gsm8k-train-5699"], spec=spec,
                          targets=slot_targets(GSM["gsm8k-train-5699"], 6), final_only=True)
        self.assertEqual((none[0].slot_lambda, none[0].slot_targets), (0.0, (None,) * 6))
        self.assertEqual(dataclasses.astuple(gfpq_group(it)[0]), dataclasses.astuple(g[0]))   # inputs untouched


def tiny_untied(tok, ckpt=False):
    torch.manual_seed(0)
    m = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=1024, tie_word_embeddings=False))
    if ckpt:
        m.gradient_checkpointing_enable()
    return m.train()


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestGradient(unittest.TestCase):
    def setUp(self):
        from cotdistill.student import SlotSpec, set_slot_prompt
        self.tok = gpt2()
        self.spec = SlotSpec(6, 2, TOK)
        set_slot_prompt(self.tok, self.spec)
        self.addCleanup(set_slot_prompt, self.tok, None)

    def batch(self):
        from cotdistill.student import slot_group, slot_targets
        return [slot_group(gfpq_group(it), it, spec=self.spec, targets=slot_targets(it, 6), final_only=True)[0]
                for it in (GSM["gsm8k-train-660"], GSM["gsm8k-train-3365"])]

    def test_letters_match_label_logits(self):
        from cotdistill.student import label_logits, slot_forward
        m = tiny_untied(self.tok).eval()
        b = self.batch()
        z, loss, st = slot_forward(m, self.tok, b, 640, {})
        for x, y in zip(z, label_logits(m, self.tok, b, 640, {})):
            self.assertTrue(torch.allclose(x, y, atol=1e-6))
        self.assertEqual(st["slot_n"], 9)
        self.assertTrue(math.isfinite(loss.item()) and loss.item() > 0)
        b2 = [dataclasses.replace(e, slot_lambda=2 * e.slot_lambda) for e in b]
        self.assertAlmostEqual(slot_forward(m, self.tok, b2, 640, {})[1].item(), 2 * loss.item(), places=4)
        b0 = [dataclasses.replace(e, weight=0.0) for e in b]                   # an FSDP pad copy: no slot loss
        self.assertEqual(float(slot_forward(m, self.tok, b0, 640, {})[1]), 0.0)
        # one supervised slot (slot 4 of 660, c = 2), recomputed by hand: its two final-layer states, then the
        # step's tokens, next-token CE from the last slot state on
        one = [dataclasses.replace(b[0], slot_targets=(None, None, None, LINES_660[3], None, None), slot_lambda=1.0)]
        got = slot_forward(m, self.tok, one, 640, {})
        from cotdistill.student import collate
        with torch.no_grad():
            c = collate(self.tok, one, 640, "cpu")
            hid = m.model(input_ids=c["input_ids"], attention_mask=c["attention_mask"]).last_hidden_state[0]
            pos = (c["input_ids"][0] == 16273).nonzero().flatten()
            self.assertEqual(pos.numel(), 12)
            y = self.tok.encode(LINES_660[3] + "\n", add_special_tokens=False)
            x = torch.cat([hid[pos[6:8]], m.get_input_embeddings()(torch.tensor(y[:-1]))])
            h2 = m.model(inputs_embeds=x[None]).last_hidden_state[0, 1:]
            ce = torch.nn.functional.cross_entropy(m.lm_head(h2), torch.tensor(y))
        self.assertAlmostEqual(float(got[1]), float(ce), places=5)
        self.assertEqual(got[2]["slot_n"], 1)
        self.assertAlmostEqual(got[2]["slot_ce_sum"], float(ce), places=5)

    def test_gradient_reaches_main_pass(self):
        """Decoding loss only: the embedding rows of tokens that occur only in the problem text before the slots (main
        pass) get a gradient, with and without gradient checkpointing; with the slot states detached they get none, and
        tokens that only occur after the slots (the options; causal) get none either."""
        from cotdistill.student import slot_forward
        b = self.batch()
        enc = lambda t: set(self.tok.encode(t, add_special_tokens=False))
        before = enc(b[0].text.split("\n" + TOK)[0]) | enc(b[1].text.split("\n" + TOK)[0])
        after = enc(b[0].text.split(TOK)[-1]) | enc(b[1].text.split(TOK)[-1])
        dec = {t for e in b for s in e.slot_targets if s for t in self.tok.encode(s + "\n", add_special_tokens=False)}
        only = sorted(before - after - dec - {self.tok.pad_token_id})
        late = sorted(after - before - dec - {self.tok.pad_token_id})
        self.assertGreater(len(only), 20)
        self.assertGreater(len(late), 0)
        for ckpt in (False, True):
            for detach in (False, True):
                m = tiny_untied(self.tok, ckpt)
                _, loss, _ = slot_forward(m, self.tok, b, 640, {}, _detach=detach)
                loss.backward()
                g = m.get_input_embeddings().weight.grad[only].abs().sum(-1)
                self.assertEqual(float(m.get_input_embeddings().weight.grad[late].abs().sum()), 0.0)
                if detach:
                    self.assertEqual(float(g.sum()), 0.0, ckpt)
                else:
                    self.assertTrue(bool((g > 0).all()), (ckpt, int((g > 0).sum()), len(only)))
                lm = m.lm_head.weight.grad
                self.assertGreater(float(lm.abs().sum()), 0)      # the decoding CE reads the same head

    def test_overfit_then_decode(self):
        """The training path and the greedy readout agree: after 200 Adam steps on 660 and 3365 (c = 1), every
        supervised slot decodes its step exactly (slot_decode), e.g. slot 1 of 660 -> '4 + 2 = 6'."""
        from cotdistill.student import SlotSpec, kl_loss, set_slot_prompt, slot_decode, slot_forward, slot_group, \
            slot_targets
        spec = SlotSpec(6, 1, TOK)
        set_slot_prompt(self.tok, spec)
        its = [GSM["gsm8k-train-660"], GSM["gsm8k-train-3365"]]
        b = [slot_group(gfpq_group(it), it, spec=spec, targets=slot_targets(it, 6), final_only=True)[0] for it in its]
        m = tiny_untied(self.tok)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        for _ in range(200):
            z, sl, _ = slot_forward(m, self.tok, b, 640, {})
            (kl_loss(z, b) * len(b) + sl).backward()
            opt.step()
            opt.zero_grad()
        rows = slot_decode(m, self.tok, its, max_len=640)
        self.assertEqual([s["decoded"] for s in rows[0]["slots"]], LINES_660)
        self.assertTrue(all(s["exact"] for r in rows for s in r["slots"] if s["target"] is not None))
        self.assertTrue(m.training)                                     # slot_decode restores the mode

    def test_necessity_modes(self):
        """swap and noise change the readout, drop equals the prompt without slots, the spec is restored."""
        from cotdistill.student import evaluate, parse_slot_necessity, set_slot_prompt, slot_necessity, slot_spec
        m = tiny_untied(self.tok).eval()
        intact, pi = evaluate(m, self.tok, ITEMS, 640, 3, {})
        res = slot_necessity(m, self.tok, ITEMS, 640, 3, {}, parse_slot_necessity("swap,drop,noise@0:3.0"),
                             swap_token=SWAP, intact=intact)
        self.assertEqual(slot_spec(self.tok), self.spec)
        set_slot_prompt(self.tok, None)
        _, p0 = evaluate(m, self.tok, ITEMS, 640, 3, {})
        set_slot_prompt(self.tok, self.spec)
        probs = lambda ps: [p["probs"] for p in ps]
        self.assertEqual(probs(res["drop"][1]), probs(p0))
        self.assertNotEqual(probs(res["swap"][1]), probs(pi))
        self.assertNotEqual(probs(res["noise@0:3.0"][1]), probs(pi))
        self.assertEqual(res["swap"][0]["acc_intact"], intact["acc"])
        for bad in ("noise@1:1.0", "noise@x:1", "zero"):              # layer 1 of 2 is the last one
            with self.assertRaises(ValueError):
                modes = parse_slot_necessity(bad)
                slot_necessity(m, self.tok, ITEMS[:1], 640, 3, {}, modes, swap_token=SWAP, intact=intact)


def ref_tree(dest: str, ref: str):
    """src/cotdistill, scripts/train_student.py and scripts/eval_student.py of commit ref under dest (None outside a git
    checkout)."""
    import tarfile
    try:
        blob = subprocess.run(["git", "-C", str(ROOT), "archive", ref, "src/cotdistill", "scripts/train_student.py",
                               "scripts/eval_student.py"], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    Path(dest).mkdir(parents=True, exist_ok=True)
    (Path(dest) / "ref.tar").write_bytes(blob)
    with tarfile.open(Path(dest) / "ref.tar") as t:
        t.extractall(dest, filter="data")
    return Path(dest)


def common(d, data="t.jsonl"):
    from test_tree_forcing import SCRIPT
    return [sys.executable, SCRIPT, "--model", d + "/m", "--train", f"{d}/{data}", "--eval", f"v={d}/{data}",
            "--items-per-update", "2", "--micro-bs", "4", "--grad-accum", "1", "--max-len", "640",
            "--eval-max-len", "640", "--precision", "fp32master", "--no-grad-ckpt", "--log-every", "1", "--lr", "1e-3",
            "--seed", "3", "--final", "teacher", "--subq-target", "cot"]


GFPQ = ["--subq", "cot", "--tree-full", "plain", "--tree-view", "statements+question", "--tree-cap", "0",
        "--tree-sources", "gsm8k_tf"]
SLOTS = ["--latent-slots", "6", "--slot-c", "2", "--slot-token", TOK]
ARMS = {"LS": SLOTS + ["--slot-final-only"], "LSU": SLOTS + ["--slot-final-only", "--slot-target", "none"],
        "LSR": SLOTS + ["--slot-final-only", "--slot-target", "shuffled"], "LSY": SLOTS}


def run(cmd, out, *extra, script=None):
    from test_tree_forcing import ENV
    if script:
        cmd = [cmd[0], script] + cmd[2:]
    return subprocess.run(cmd + ["--out", out, *extra], capture_output=True, text=True, timeout=900, env=ENV)


def metrics(out):
    return json.loads((Path(out) / "metrics.json").read_text())


def preds(path):
    return [json.loads(l) for l in open(path)]


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_tree_forcing import tiny
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = d = cls.tmp.name
        tiny(d)
        Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in ITEMS))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def ok(self, r):
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])

    def test_arms_end_to_end(self):
        """LS / LSU / LSR / LSY (and GFPQ) for two epochs; the necessity test and the decoding readout at the end;
        eval_student on LS's saved model rebuilds the slot prompt (latent_slots.json) and its predictions."""
        d = self.d
        nec = ["--slot-necessity", "swap,drop,noise@0:1.0", "--slot-swap-token", SWAP, "--slot-decode", "3"]
        ms = {}
        for arm, extra in dict(ARMS, GFPQ=[]).items():
            more = nec + (["--save"] if arm == "LS" else []) if arm != "GFPQ" else []
            r = run(common(d) + GFPQ, f"{d}/{arm}", *extra, "--epochs", "2", *more)
            self.ok(r)
            ms[arm] = metrics(f"{d}/{arm}")
            self.assertEqual("latent slots:" in r.stdout, arm != "GFPQ", arm)
        n_upd = 2 * math.ceil(len(ITEMS) / 2)
        for arm in ARMS:
            h = ms[arm]["history"]
            self.assertEqual(len(h), n_upd, arm)
            self.assertTrue(all(math.isfinite(x["loss"]) for x in h), arm)
            self.assertEqual("loss_slot" in h[0], arm != "LSU", arm)
            ls = ms[arm]["latent_slots"]
            self.assertEqual((ls["k"], ls["c"], ls["token"], ls["place"]), (6, 2, TOK, "before-options"))
            self.assertEqual(ls["views"]["slot_items"], 6)
            self.assertEqual(set(ms[arm]["slot_necessity"]["v"]), {"swap", "drop", "noise@0:1.0"})
            for mode in ("swap", "drop", "noise@0:1.0"):
                x = ms[arm]["slot_necessity"]["v"][mode]
                self.assertAlmostEqual(x["acc_drop"], x["acc_intact"] - x["acc"])
                self.assertTrue(Path(f"{d}/{arm}/preds_v@{re.sub(r'[^A-Za-z0-9._-]', '_', mode)}.jsonl").exists())
            dec = [json.loads(l) for l in open(f"{d}/{arm}/slot_decode_v.jsonl")]
            self.assertEqual([r["item_id"] for r in dec], [it["item_id"] for it in ITEMS[:3]])
            self.assertEqual([s["target"] for s in dec[0]["slots"]], LINES_660)
            self.assertEqual(ms[arm]["slot_decode"]["all"]["n"], 6 + 3 + 2)
        for arm in ("LS", "LSR", "LSU"):
            self.assertEqual({x["seqs"] for x in ms[arm]["history"]}, {2}, arm)          # final views only
        self.assertEqual([x["seqs"] for x in ms["LSY"]["history"]], [x["seqs"] for x in ms["GFPQ"]["history"]])
        self.assertGreater(min(x["loss_node"] for x in ms["LSY"]["history"]), 0)
        # LS and LSR: the same supervised slots per update, other texts
        self.assertEqual([x["slot_n"] for x in ms["LS"]["history"]], [x["slot_n"] for x in ms["LSR"]["history"]])
        self.assertEqual(sum(x["slot_n"] for x in ms["LS"]["history"]), 2 * 21)
        self.assertNotEqual([x["loss_slot"] for x in ms["LS"]["history"]],
                            [x["loss_slot"] for x in ms["LSR"]["history"]])
        self.assertEqual(ms["LS"]["latent_slots"]["views"]["slot_supervised"], 21)
        # LSU: the slot line without a decoding pass differs from LS only through the slot loss
        self.assertNotEqual([x["loss"] for x in ms["LSU"]["history"]], [x["loss"] for x in ms["LS"]["history"]])
        self.assertAlmostEqual(ms["LS"]["history"][0]["loss"], ms["LS"]["history"][0]["loss_final"] +
                               ms["LS"]["history"][0]["loss_slot"], places=5)
        # eval_student on the saved LS model: the same slot line from latent_slots.json, the same predictions
        side = json.loads(Path(f"{d}/LS/model/latent_slots.json").read_text())
        self.assertEqual(side, {"k": 6, "c": 2, "token": TOK, "place": "before-options"})
        from test_tree_forcing import ENV
        r = subprocess.run([sys.executable, str(ROOT / "scripts/eval_student.py"), "--model", f"{d}/LS/model",
                            "--eval", f"v={d}/t.jsonl", "--max-len", "640", "--bs", "8", "--out", f"{d}/ev",
                            "--slot-necessity", "drop", "--slot-decode", "2"], capture_output=True, text=True,
                           timeout=900, env=ENV)
        self.ok(r)
        self.assertIn("latent slots (from", r.stdout)
        a, b = preds(f"{d}/LS/preds_v.jsonl"), preds(f"{d}/ev/preds_v.jsonl")
        self.assertEqual([p["item_id"] for p in a], [p["item_id"] for p in b])
        for p, q in zip(a, b):                           # bf16 save: close, not equal
            self.assertLess(max(abs(x - y) for x, y in zip(p["probs"], q["probs"])), 2e-2)
        em = metrics(f"{d}/ev")
        self.assertIn("v@drop", em["eval"])
        self.assertEqual(em["slot_decode"]["items"], 2)

    def test_resume(self):
        """LS with a planned stop after update 2 and a resume: the uninterrupted run's history and evaluation."""
        d = self.d
        a = run(common(d) + GFPQ, f"{d}/full", *ARMS["LS"], "--epochs", "2", "--ckpt-every-min", "1000")
        self.ok(a)
        r1 = run(common(d) + GFPQ, f"{d}/res", *ARMS["LS"], "--epochs", "2", "--stop-at-step", "2")
        self.assertEqual(r1.returncode, 3, r1.stderr[-2000:])
        r2 = run(common(d) + GFPQ, f"{d}/res", *ARMS["LS"], "--epochs", "2", "--ckpt-every-min", "1000")
        self.ok(r2)
        self.assertIn("resumed from", r2.stdout)
        drop = re.compile(r"s_per_step|approx_tok_per_s")
        strip = lambda h: [{k: v for k, v in row.items() if not drop.match(k)} for row in h]
        self.assertEqual(strip(metrics(f"{d}/full")["history"]), strip(metrics(f"{d}/res")["history"]))
        self.assertEqual(metrics(f"{d}/full")["eval"], metrics(f"{d}/res")["eval"])

    def test_argument_checks(self):
        d = self.d
        bad = [(GFPQ, ["--slot-c", "2"]),                                    # slot flags without --latent-slots
               (["--subq", "cot", "--tree-full", "true"], SLOTS),            # not --tree-full plain
               (["--subq", "none"], SLOTS),                                  # not the full-tree loop
               (GFPQ, SLOTS + ["--si-stages", "2"]),
               (GFPQ, SLOTS + ["--slot-c", "0"]),
               (GFPQ, ["--latent-slots", "6"]),                              # GPT-2: no reserved token
               (GFPQ, ["--latent-slots", "6", "--slot-token", "Abe"]),        # occurs in a prompt (and is no single
               (GFPQ, ["--latent-slots", "6", "--slot-token", "A"]),          # token / is in the prompts)
               (GFPQ, SLOTS + ["--slot-necessity", "swap", "--slot-swap-token", TOK]),
               (GFPQ, SLOTS + ["--slot-necessity", "zero"]),
               (GFPQ, SLOTS + ["--chat-template"])]
        for arm, extra in bad:
            r = run(common(d) + arm, f"{d}/bad", *extra, "--epochs", "1")
            self.assertNotEqual(r.returncode, 0, extra)
        Path(d + "/nc.jsonl").write_text("\n".join(json.dumps({k: v for k, v in x.items() if k != "worked_steps"})
                                                   for x in ITEMS))
        r = run(common(d, "nc.jsonl") + GFPQ, f"{d}/bad2", *ARMS["LS"], "--epochs", "1")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("has worked steps", r.stderr + r.stdout)
        r = run(common(d, "nc.jsonl") + GFPQ, f"{d}/lsu-nc", *ARMS["LSU"], "--epochs", "1")    # LSU needs no chain
        self.ok(r)

    def test_existing_arms_match_ref(self):
        """GFPQ (full-tree loop), A2 (plain loop), SIB (SI stages) and eval_student: LS_BASE_REF's files."""
        from test_tree_forcing import ENV
        d = self.d
        if ref_tree(d + "/h", LS_BASE_REF) is None:
            self.skipTest("not a git checkout")
        arms = {"GFPQ": GFPQ + ["--epochs", "2", "--eval-subq", f"{d}/t.jsonl"], "A2": ["--subq", "none", "--epochs", "2"],
                "SIB": GFPQ + ["--si-stages", "2", "--si-final-only", "--si-remove", "back", "--epochs", "3"]}
        drop = re.compile(r"s_per_step|approx_tok_per_s")
        strip = lambda h: [{k: v for k, v in row.items() if not drop.match(k)} for row in h]
        for arm, extra in arms.items():
            outs = []
            for name, script in (("ref", d + "/h/scripts/train_student.py"), ("now", None)):
                o = f"{d}/{name}-{arm}"
                self.ok(run(common(d), o, *extra, *(["--save"] if arm == "A2" else []), script=script))
                outs.append(Path(o))
            a, b = metrics(outs[0]), metrics(outs[1])
            self.assertEqual(strip(a["history"]), strip(b["history"]), arm)
            self.assertEqual(a["eval"], b["eval"], arm)
            self.assertEqual(sorted(p.name for p in outs[0].iterdir()), sorted(p.name for p in outs[1].iterdir()))
            for f in sorted(p.name for p in outs[0].glob("preds_*.jsonl")):
                self.assertEqual((outs[0] / f).read_text(), (outs[1] / f).read_text(), (arm, f))
        for name, script in (("ref", d + "/h/scripts/eval_student.py"), ("now", str(ROOT / "scripts/eval_student.py"))):
            r = subprocess.run([sys.executable, script, "--model", f"{d}/now-A2/model", "--eval", f"v={d}/t.jsonl",
                                "--max-len", "640", "--out", f"{d}/ev-{name}", "--time"], capture_output=True,
                               text=True, timeout=900, env=ENV)
            self.ok(r)
        self.assertEqual(Path(f"{d}/ev-ref/preds_v.jsonl").read_text(), Path(f"{d}/ev-now/preds_v.jsonl").read_text())
        ta, tb = (json.loads(Path(f"{d}/ev-{n}/timing_v.json").read_text()) for n in ("ref", "now"))
        self.assertEqual((ta["tokens_mean"], ta["tokens_max"]), (tb["tokens_mean"], tb["tokens_max"]))


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestFSDP(unittest.TestCase):
    """LS and LSY under --fsdp (FSDP2 over gloo on the CPU, the byte-level tiny model of test_fsdp, slot token '~'):
    the single-process updates (fp32 shards + cpu-adamw vs fp32master + AdamW), the same evaluation, necessity test
    and decoding readout; one rank (the 4B single-GPU FSDP route) too."""

    def test_ls_lsy_fsdp_equal_single(self):
        import test_fsdp as tf
        with tempfile.TemporaryDirectory() as d:
            tf.tiny_model(d)
            self.assertFalse(any("~" in it["prompt"] or "^" in it["prompt"] for it in ITEMS))
            Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in ITEMS))
            base = ["--model", d + "/m", "--train", d + "/t.jsonl", "--eval", f"v={d}/t.jsonl", *GFPQ,
                    "--items-per-update", "3", "--micro-bs", "2", "--max-len", "1024", "--eval-max-len", "1024",
                    "--log-every", "1", "--lr", "1e-3", "--seed", "0", "--epochs", "2", "--precision", "fp32master",
                    "--latent-slots", "4", "--slot-c", "2", "--slot-token", "~", "--slot-necessity", "swap,noise@0:1.0",
                    "--slot-swap-token", "^", "--slot-decode", "2"]
            for arm, extra in (("LS", ["--slot-final-only"]), ("LSY", [])):
                runs = {"single": tf.single(d, *base, *extra, "--out", f"{d}/{arm}-s"),
                        "fsdp2": tf.fsdp(d, *base, *extra, "--fsdp-optim", "cpu-adamw", "--out", f"{d}/{arm}-f2"),
                        "fsdp1": tf.fsdp(d, *base, *extra, "--fsdp-optim", "cpu-adamw", "--out", f"{d}/{arm}-f1",
                                         nproc=1)}
                for k, r in runs.items():
                    self.assertEqual(r.returncode, 0, (arm, k, r.stderr[-3000:]))
                a = tf.metrics(f"{d}/{arm}-s")
                for tag in ("f2", "f1"):
                    b = tf.metrics(f"{d}/{arm}-{tag}")
                    self.assertEqual(len(a["history"]), len(b["history"]))
                    for x, y in zip(a["history"], b["history"]):
                        self.assertAlmostEqual(x["loss"], y["upd_loss"], delta=1e-5, msg=(arm, tag))
                        self.assertAlmostEqual(x["loss_slot"], y["loss_slot"], delta=1e-5, msg=(arm, tag))
                        self.assertEqual(x["slot_n"], y["slot_n"], (arm, tag))
                    pa, pb = tf.preds(f"{d}/{arm}-s"), tf.preds(f"{d}/{arm}-{tag}")
                    for p, q in zip(pa, pb):
                        self.assertLess(max(abs(u - v) for u, v in zip(p["probs"], q["probs"])), 1e-4)
                    for mode in ("swap", "noise@0:1.0"):
                        self.assertAlmostEqual(a["slot_necessity"]["v"][mode]["acc"],
                                               b["slot_necessity"]["v"][mode]["acc"], msg=(arm, tag, mode))
                    da = [json.loads(l) for l in open(f"{d}/{arm}-s/slot_decode_v.jsonl")]
                    db = [json.loads(l) for l in open(f"{d}/{arm}-{tag}/slot_decode_v.jsonl")]
                    self.assertEqual(da, db, (arm, tag))
                self.assertGreater(sum(x["slot_n"] for x in a["history"]), 0)



# ---------------------------------------------------------------------------------------------------- v2.1 (lens)
SLOTS21 = ["--latent-slots", "6", "--slot-c", "2", "--slot-tokens", "distinct", "--slot-token", ",".join(DIST),
           "--slot-layer", "0", "--slot-readout", "lens", "--slot-fill", "last"]
ARMS21 = {"LSL": SLOTS21 + ["--slot-final-only"], "LSLU": SLOTS21 + ["--slot-final-only", "--slot-target", "none"],
          "LSLR": SLOTS21 + ["--slot-final-only", "--slot-target", "shuffled"], "LSLY": SLOTS21}


def lens_spec(k=6, c=2, layer=0, fill="last", tokens=DIST):
    from cotdistill.student import SlotSpec
    return SlotSpec(k, c, tokens[0], tokens=tuple(tokens[:k]), layer=layer, readout="lens", fill=fill)


def qwen35_tiny(n, vocab=64):
    """A random Qwen3.5 text model (Gated DeltaNet + full attention every 4th layer, as the real configs)."""
    cfg = transformers.Qwen3_5TextConfig(num_hidden_layers=n, hidden_size=16, intermediate_size=32, num_attention_heads=2,
                                         num_key_value_heads=1, head_dim=8, linear_num_key_heads=2,
                                         linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
                                         vocab_size=vocab)
    torch.manual_seed(0)
    return transformers.Qwen3_5ForCausalLM(cfg)


def lens_batch(spec, ids=("gsm8k-train-660", "gsm8k-train-3365")):
    from cotdistill.student import slot_group, slot_value_targets
    return [slot_group(gfpq_group(GSM[i]), GSM[i], spec=spec, targets=slot_value_targets(GSM[i], spec.k, fill=spec.fill),
                       final_only=True)[0] for i in ids]


class TestV21Targets(unittest.TestCase):
    def test_values_660(self):
        from cotdistill.student import answer_text, slot_value_targets, slot_values
        it = GSM["gsm8k-train-660"]
        self.assertEqual(slot_values(it), VALUES_660)
        self.assertEqual(slot_value_targets(it, 6), VALUES_660)
        self.assertEqual(slot_value_targets(it, 6, fill="last"), VALUES_660)
        self.assertEqual(slot_value_targets(it, 8), VALUES_660 + [None, None])
        self.assertEqual(slot_value_targets(it, 8, fill="last"), VALUES_660 + ["20", "20"])
        self.assertEqual(answer_text(it), "20")                                          # B) 20
        self.assertEqual(slot_value_targets(it, 8, fill="answer"), VALUES_660 + ["20", "20"])
        self.assertEqual(slot_value_targets(it, 4), ["6", "8", "2", "20"])            # merge: the answer's step stays
        self.assertEqual(slot_value_targets(it, 4, "truncate"), ["6", "8", "2", "10"])
        it2 = GSM["gsm8k-train-3365"]
        self.assertEqual(slot_value_targets(it2, 6, fill="last"), ["60", "6", "66", "66", "66", "66"])
        self.assertEqual(slot_value_targets(it2, 6, fill="answer"), ["60", "6", "66", "66", "66", "66"])
        self.assertEqual(slot_value_targets(it2, 6), ["60", "6", "66", None, None, None])
        self.assertEqual(slot_value_targets(GSM["gsm8k-train-5699"], 6, fill="last"), [None] * 6)     # no chain
        v = [x["value"] for x in GSM["gsm8k-train-6669"]["worked_steps"]]
        self.assertEqual(slot_value_targets(GSM["gsm8k-train-6669"], 6), v[:5] + ["260"])
        with self.assertRaises(ValueError):
            slot_value_targets(it, 6, fill="first")

    def test_shuffled_values(self):
        """LSLR: LSL's supervised slots (fill included), every value from other problems, never one of the item's own;
        keyed by seed and item, not the training rng."""
        from cotdistill.student import shuffled_value_targets, slot_value_targets
        state = random.getstate()
        a = shuffled_value_targets(ITEMS, k=6, fill="last", seed=3)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(a, shuffled_value_targets(ITEMS, k=6, fill="last", seed=3))
        self.assertNotEqual(a, shuffled_value_targets(ITEMS, k=6, fill="last", seed=4))
        real = {it["item_id"]: slot_value_targets(it, 6, fill="last") for it in ITEMS}
        for iid, ts in a.items():
            self.assertEqual([t is None for t in ts], [t is None for t in real[iid]])
            others = {t for j, r in real.items() if j != iid for t in r if t is not None}
            for t in ts:
                if t is not None:
                    self.assertNotIn(t, set(real[iid]))
                    self.assertIn(t, others)
        self.assertEqual(sum(t is not None for ts in a.values() for t in ts), 30)         # 5 chains x 6 slots


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestV21Prompt(unittest.TestCase):
    def setUp(self):
        from cotdistill.student import set_slot_prompt
        self.tok = gpt2()
        self.addCleanup(set_slot_prompt, self.tok, None)

    def test_distinct_line_and_spec(self):
        from cotdistill.student import (SlotSpec, _slot_ids, choose_slot_tokens, final_text, lens_target_ids,
                                        set_slot_prompt, slot_end_id, slot_prompt)
        spec = lens_spec()
        p = GSM["gsm8k-train-660"]["prompt"]
        head, _, opts = p.partition("\nOptions:\n")
        self.assertEqual(slot_prompt(p, spec), head + "\n§§©©®®°°±±¢¢\nOptions:\n" + opts)
        set_slot_prompt(self.tok, spec)
        sids = [self.tok.encode(t)[0] for t in DIST]
        self.assertEqual(_slot_ids(self.tok), tuple(sids))
        ids = self.tok.encode(final_text(self.tok, p))
        pos = [i for i, x in enumerate(ids) if x in set(sids)]
        self.assertEqual([ids[i] for i in pos], [x for x in sids for _ in range(2)])
        self.assertEqual(pos, list(range(pos[0], pos[0] + 12)))                       # one line, slot by slot
        self.assertEqual(spec.as_dict(), {"k": 6, "c": 2, "token": "§", "place": "before-options", "tokens": list(DIST),
                                          "layer": 0, "readout": "lens", "fill": "last"})
        self.assertEqual(SlotSpec(6, 2, TOK).as_dict(), {"k": 6, "c": 2, "token": TOK, "place": "before-options"})
        self.assertEqual(choose_slot_tokens(self.tok, 6, ",".join(DIST)), DIST)
        for bad in (",".join(DIST[:5]), ",".join(DIST[:5] + ("§",)), "auto", ",".join(DIST[:5] + ("~",))):
            with self.assertRaises(ValueError):
                choose_slot_tokens(self.tok, 6, bad)
        with self.assertRaises(ValueError):                                          # 5 tokens for 6 slots
            set_slot_prompt(self.tok, SlotSpec(6, 2, "§", tokens=DIST[:5], layer=0, readout="lens"))
        end = slot_end_id(self.tok)
        self.assertEqual(end, 198)
        six = self.tok.encode("6")
        self.assertEqual(lens_target_ids(self.tok, "6", 2), (six + [end], True))
        self.assertEqual(lens_target_ids(self.tok, "6", 1), (six, True))
        long = self.tok.encode("25044.75")
        self.assertEqual(lens_target_ids(self.tok, "25044.75", 2), (long[:2], len(long) <= 2))

    @unittest.skipUnless(QWEN, "no local Qwen3.5 tokenizer (Sev-2B-preview snapshot)")
    def test_qwen_distinct_tokens(self):
        """Qwen3.5: the six grounding tokens (no vision / audio / chat id), swap <|fim_pad|>; the 660 slot line of 6 x 4
        tokens adds 25 tokens to the A2 prompt; one token per digit, so "10" fits 4 positions as 1, 0, \\n, \\n."""
        from cotdistill.student import (FINAL_TEMPLATE, SLOT_POOL, SLOT_TOKENS, SlotSpec, choose_slot_token,
                                        choose_slot_tokens, final_text, lens_target_ids, set_slot_prompt)
        tok = transformers.AutoTokenizer.from_pretrained(QWEN[0])
        ts = choose_slot_tokens(tok, 6)
        self.assertEqual(ts, SLOT_TOKENS)
        ids = [tok.convert_tokens_to_ids(t) for t in ts]
        self.assertEqual(ids, [248049, 248051, 248047, 248050, 248052, 248048])
        sw = choose_slot_token(tok, avoid=ts, pool=SLOT_POOL)
        self.assertEqual((sw, tok.convert_tokens_to_ids(sw)), ("<|fim_pad|>", 248063))
        self.assertEqual([tok.convert_tokens_to_ids(t) for t in SLOT_POOL[6:]], list(range(248063, 248066)) +
                         list(range(248060, 248063)))
        banned = set(range(248053, 248058)) | set(range(248070, 248077)) | {248044, 248045, 248046}
        self.assertFalse(banned & set(tok.convert_tokens_to_ids(list(SLOT_POOL))))
        set_slot_prompt(tok, SlotSpec(6, 4, ts[0], tokens=ts, layer=20, readout="lens", fill="last"))
        p = GSM["gsm8k-train-660"]["prompt"]
        a2 = tok.encode(FINAL_TEMPLATE.format(problem=p), add_special_tokens=False)
        got = tok.encode(final_text(tok, p), add_special_tokens=False)
        self.assertEqual(len(got) - len(a2), 25)
        self.assertEqual([i for i in got if i in set(ids)], [i for i in ids for _ in range(4)])
        nl = tok.encode("\n", add_special_tokens=False)
        self.assertEqual(lens_target_ids(tok, "10", 4), (tok.encode("1", add_special_tokens=False) +
                                                        tok.encode("0", add_special_tokens=False) + nl * 2, True))
        self.assertEqual([len(tok.encode(v, add_special_tokens=False)) for v in VALUES_660], [1, 1, 1, 2, 2, 2])
        self.assertFalse(lens_target_ids(tok, "25044", 4)[1])
        set_slot_prompt(tok, None)


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestV21Layer(unittest.TestCase):
    def test_resolve(self):
        """0.67 -> layer 20 of 32 (Qwen3.5-4B / 9B: 11 layers after it, full attention at 23, 27, 31), 15 of 24 (2B)."""
        from cotdistill.student import resolve_slot_layer
        m32, m24 = qwen35_tiny(32), qwen35_tiny(24)
        li, info = resolve_slot_layer(m32, "0.67")
        self.assertEqual((li, info["layers_after"], info["full_attention_after"]), (20, 11, [23, 27, 31]))
        self.assertEqual(resolve_slot_layer(m32, "2/3")[0], 20)
        self.assertEqual(resolve_slot_layer(m24, "0.67")[0], 15)
        self.assertEqual(resolve_slot_layer(m24, "0.67")[1]["full_attention_after"], [19, 23])
        self.assertEqual(resolve_slot_layer(m32, "-2")[0], 30)
        self.assertEqual(resolve_slot_layer(m32, "0")[0], 0)
        self.assertEqual(resolve_slot_layer(m32, "final"), (None, {}))
        for bad in ("31", "-1", "32", "1.0", "0.0", "-40"):
            with self.assertRaises(ValueError, msg=bad):
                resolve_slot_layer(m32, bad)

    def test_capture(self):
        """_SlotCapture(li) = output_hidden_states[li + 1] (Qwen2; a hybrid Qwen3.5 in a subprocess without fla, whose
        triton kernels need a GPU), in eval and in training with gradient checkpointing off / non-reentrant /
        reentrant; its gradient reaches the embeddings of the positions up to the read one, not of later ones."""
        torch.manual_seed(0)
        capture_check(transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
            vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=4, num_attention_heads=2,
            num_key_value_heads=1, max_position_embeddings=64)))
        from test_tree_forcing import ENV
        code = ("import sys; sys.modules['fla'] = None; sys.path[:0] = [%r, %r]; import test_latent_slots as T; "
                "T.capture_check(T.qwen35_tiny(8)); print('capture ok')" % (str(ROOT / "src"), str(ROOT / "tests")))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, env=ENV)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("capture ok", r.stdout)


def capture_check(m) -> None:
    from cotdistill.student import _SlotCapture, text_parts
    ids = torch.randint(3, 64, (2, 12), generator=torch.Generator().manual_seed(1))
    body, _ = text_parts(m)
    m.eval()
    with torch.no_grad():
        hs = body(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
    n = len(hs) - 1
    for li in range(n - 1):
        with torch.no_grad(), _SlotCapture(body, li) as cap:
            body(input_ids=ids, use_cache=False)
        assert torch.allclose(cap.h, hs[li + 1], atol=1e-6), li
    try:
        _SlotCapture(body, n - 1)
        raise AssertionError("the last layer was accepted")
    except ValueError:
        pass
    early = set(ids[:, :6].flatten().tolist())
    late = sorted(set(ids[:, 6:].flatten().tolist()) - early)
    assert late
    for kw in (None, {"use_reentrant": False}, {"use_reentrant": True}):
        m.train()
        if kw is None:
            m.gradient_checkpointing_disable()
        else:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kw)
        m.zero_grad()
        with _SlotCapture(body, 1) as cap:
            body(input_ids=ids, use_cache=False)
        assert torch.allclose(cap.h, hs[2], atol=1e-5), kw
        cap.h[:, 5].pow(2).sum().backward()
        g = m.get_input_embeddings().weight.grad
        assert float(g[sorted(early)].abs().sum()) > 0, kw
        assert float(g[late].abs().sum()) == 0.0, kw                 # causal: later positions get none
    m.gradient_checkpointing_disable()


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestV21Lens(unittest.TestCase):
    def setUp(self):
        from cotdistill.student import set_slot_prompt
        self.tok = gpt2()
        self.spec = lens_spec()
        set_slot_prompt(self.tok, self.spec)
        self.addCleanup(set_slot_prompt, self.tok, None)

    def test_lens_loss_manual(self):
        """slot_forward (lens): the letters of label_logits, one pass, and the loss = sum over supervised slots of
        weight x lambda / slots x mean CE of norm(layer-0 state) @ head over the c positions (value token, then \\n)."""
        from cotdistill.student import collate, label_logits, slot_forward
        m = tiny_untied(self.tok).eval()
        b = lens_batch(self.spec)
        z, loss, st = slot_forward(m, self.tok, b, 640, {})
        for x, y in zip(z, label_logits(m, self.tok, b, 640, {})):
            self.assertTrue(torch.allclose(x, y, atol=1e-6))
        self.assertEqual(st["slot_n"], 12)
        sids = [self.tok.encode(t)[0] for t in DIST]
        tot, ces = 0.0, []
        with torch.no_grad():
            c = collate(self.tok, b, 640, "cpu")
            hs = m.model(input_ids=c["input_ids"], attention_mask=c["attention_mask"],
                         output_hidden_states=True).hidden_states[1]
            for i, e in enumerate(b):
                row = c["input_ids"][i].tolist()
                for j, t in enumerate(e.slot_targets):
                    pos = [q for q, x in enumerate(row) if x == sids[j]]
                    self.assertEqual(len(pos), 2)
                    y = (self.tok.encode(t) + [198, 198])[:2]
                    ce = torch.nn.functional.cross_entropy(m.lm_head(m.model.norm(hs[i, pos])), torch.tensor(y))
                    ces.append(float(ce))
                    tot += e.weight * e.slot_lambda * float(ce)
        self.assertAlmostEqual(float(loss), tot, places=5)
        self.assertAlmostEqual(st["slot_ce_sum"], sum(ces), places=4)
        b2 = [dataclasses.replace(e, slot_lambda=2 * e.slot_lambda) for e in b]
        self.assertAlmostEqual(slot_forward(m, self.tok, b2, 640, {})[1].item(), 2 * loss.item(), places=4)
        b0 = [dataclasses.replace(e, weight=0.0) for e in b]
        self.assertEqual(float(slot_forward(m, self.tok, b0, 640, {})[1]), 0.0)

    def test_gradient_reaches_main_pass(self):
        """Lens loss only: embedding rows of problem-only tokens get a gradient (with and without checkpointing), none
        when the layer-0 slot states are detached, none for tokens after the slot line (causal)."""
        from cotdistill.student import slot_forward
        b = lens_batch(self.spec)
        enc = lambda t: set(self.tok.encode(t, add_special_tokens=False))
        before = enc(b[0].text.split("\n" + DIST[0])[0]) | enc(b[1].text.split("\n" + DIST[0])[0])
        after = enc(b[0].text.split(DIST[-1])[-1]) | enc(b[1].text.split(DIST[-1])[-1])
        dec = {t for e in b for s in e.slot_targets if s for t in self.tok.encode(s + "\n")}
        only = sorted(before - after - dec - {self.tok.pad_token_id})
        late = sorted(after - before - dec - {self.tok.pad_token_id})
        self.assertGreater(len(only), 20)
        self.assertGreater(len(late), 0)
        for ckpt in (False, True):
            for detach in (False, True):
                m = tiny_untied(self.tok, ckpt)
                _, loss, _ = slot_forward(m, self.tok, b, 640, {}, _detach=detach)
                loss.backward()
                ge = m.get_input_embeddings().weight.grad          # None: no path to the embeddings at all
                ge = torch.zeros_like(m.get_input_embeddings().weight) if ge is None else ge
                g = ge[only].abs().sum(-1)
                self.assertEqual(float(ge[late].abs().sum()), 0.0)
                if detach:
                    self.assertEqual(float(g.sum()), 0.0, ckpt)
                else:
                    self.assertTrue(bool((g > 0).all()), (ckpt, int((g > 0).sum()), len(only)))
                hg = m.lm_head.weight.grad                         # the lens head is read detached (default)
                self.assertTrue(hg is None or float(hg.abs().sum()) == 0.0)

    def test_head_stop_grad(self):
        """Slot loss alone, default (head_grad False): zero gradient on the LM head (untied; on a tied model none on the
        rows of target-only tokens), while the layer-0 slot states get one; head_grad True (--slot-head-grad) trains
        the head; the letter CE trains it either way."""
        from cotdistill.fsdp import decoder_layers
        from cotdistill.student import collate, kl_loss, set_slot_prompt, slot_forward
        b = lens_batch(self.spec)
        for head_grad in (False, True):
            spec = dataclasses.replace(self.spec, head_grad=head_grad)
            set_slot_prompt(self.tok, spec)
            m = tiny_untied(self.tok)
            seen = []

            def keep(mod, args, kwargs):
                h = args[0] if args else kwargs["hidden_states"]
                h.retain_grad()
                seen.append(h)
            hd = decoder_layers(m.model)[1].register_forward_pre_hook(keep, with_kwargs=True)
            z, sl, _ = slot_forward(m, self.tok, b, 640, {})
            hd.remove()
            sl.backward()
            hg = m.lm_head.weight.grad
            if head_grad:
                self.assertGreater(float(hg.abs().sum()), 0)
            else:
                self.assertTrue(hg is None or float(hg.abs().sum()) == 0.0)
            sid = self.tok.encode(DIST[0])[0]
            ids = collate(self.tok, b, 640, "cpu")["input_ids"]
            gs = seen[0].grad[ids == sid]                         # slot 1's layer-0 states
            self.assertGreater(float(gs.abs().sum()), 0, head_grad)
            m.zero_grad()
            kl_loss(slot_forward(m, self.tok, b, 640, {})[0], b).backward()  # the letter CE reads the head as before
            self.assertGreater(float(m.lm_head.weight.grad.abs().sum()), 0)
        set_slot_prompt(self.tok, self.spec)
        # tied head = embedding (2B / 4B): rows of target-only tokens get no gradient from the slot loss
        torch.manual_seed(0)
        tied = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
            vocab_size=len(self.tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=1024, tie_word_embeddings=True)).train()
        self.assertIs(tied.lm_head.weight, tied.get_input_embeddings().weight)
        _, sl, _ = slot_forward(tied, self.tok, b, 640, {})
        sl.backward()
        g_tied = tied.get_input_embeddings().weight.grad
        targets = sorted({t for x in b for v in x.slot_targets if v for t in self.tok.encode(v)} | {198})
        inputs = set(collate(self.tok, b, 640, "cpu")["input_ids"].flatten().tolist())
        only_t = [t for t in targets if t not in inputs]
        self.assertGreater(len(only_t), 0)
        self.assertEqual(float(g_tied[only_t].abs().sum()), 0.0)   # target-only rows: no gradient through the head

    def test_overfit_one_problem(self):
        """One problem (660), 120 Adam steps on letter CE + lens loss with the head read detached (the default): the
        layer-0 lens readout gives 6, 8, 2, 10, 18, 20 on slots 1-6 (100%). The head is re-drawn at a pretrained-like
        scale (N(0, 1) rows): with the 0.02 init, a 32-d body cannot single out one of GPT-2's 50k random rows."""
        from cotdistill.student import kl_loss, slot_forward, slot_lens_readout, slot_readout_summary
        it = GSM["gsm8k-train-660"]
        b = lens_batch(self.spec, ids=("gsm8k-train-660",))
        m = tiny_untied(self.tok)
        with torch.no_grad():
            m.lm_head.weight.normal_(0.0, 1.0, generator=torch.Generator().manual_seed(1))
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        for _ in range(120):
            z, sl, _ = slot_forward(m, self.tok, b, 640, {})
            (kl_loss(z, b) + sl).backward()
            opt.step()
            opt.zero_grad()
        rows = slot_lens_readout(m, self.tok, [it], max_len=640)
        self.assertEqual([s["read"] for s in rows[0]["slots"]], VALUES_660)
        self.assertEqual([s["target"] for s in rows[0]["slots"]], VALUES_660)
        self.assertEqual(slot_readout_summary(rows)["all"], {"n": 6, "value": 1.0, "token": 1.0, "fit": 1.0})
        self.assertTrue(m.training)

    def test_mean_ablation_and_readout(self):
        """mean: its own slot means leave an item's prediction unchanged; other items' means (and swap) change it;
        the spec is restored; the lens readout of every item."""
        from cotdistill.student import (_SlotMean, evaluate, parse_slot_necessity, set_slot_prompt, slot_layer_means,
                                        slot_lens_readout, slot_necessity, slot_readout_summary, slot_spec)
        m = tiny_untied(self.tok).eval()
        it = GSM["gsm8k-train-660"]
        _, p1 = evaluate(m, self.tok, [it], 640, 1, {})
        own = slot_layer_means(m, self.tok, [it], layer=0, max_len=640)
        self.assertEqual(tuple(own.shape), (12, 32))
        with _SlotMean(m, self.tok, 0, own):
            _, p2 = evaluate(m, self.tok, [it], 640, 1, {})
        self.assertEqual(p1[0]["probs"], p2[0]["probs"])
        intact, pi = evaluate(m, self.tok, ITEMS, 640, 3, {})
        res = slot_necessity(m, self.tok, ITEMS, 640, 3, {}, parse_slot_necessity("swap,drop,mean,mean@0"),
                             swap_token=SWAP, intact=intact, mean_items=ITEMS[1:])
        self.assertEqual(slot_spec(self.tok), self.spec)
        probs = lambda ps: [p["probs"] for p in ps]
        for mode in ("swap", "mean"):
            self.assertNotEqual(probs(res[mode][1]), probs(pi), mode)
        self.assertEqual(probs(res["mean"][1]), probs(res["mean@0"][1]))
        self.assertEqual((res["mean"][0]["mean_layer"], res["mean"][0]["mean_items"]), (0, 5))
        set_slot_prompt(self.tok, None)
        _, p0 = evaluate(m, self.tok, ITEMS, 640, 3, {})
        set_slot_prompt(self.tok, self.spec)
        self.assertEqual(probs(res["drop"][1]), probs(p0))
        for bad, kw in (("mean", {}), ("mean@1", {"mean_items": ITEMS})):         # no items; the last layer
            with self.assertRaises(ValueError):
                slot_necessity(m, self.tok, ITEMS[:1], 640, 3, {}, parse_slot_necessity(bad), swap_token=SWAP,
                               intact=intact, **kw)
        rows = slot_lens_readout(m, self.tok, ITEMS, max_len=640, bs=4)
        self.assertEqual([r["item_id"] for r in rows], [x["item_id"] for x in ITEMS])
        self.assertEqual([s["target"] for s in rows[0]["slots"]], VALUES_660)
        sm = slot_readout_summary(rows)
        self.assertEqual(sm["all"]["n"], 30)
        self.assertEqual(sorted(sm), ["all"] + [f"slot{j}" for j in range(1, 7)])


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestV21Train(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_tree_forcing import tiny
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = d = cls.tmp.name
        tiny(d)
        Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in ITEMS))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def ok(self, r):
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])

    def test_v21_arms_end_to_end(self):
        """LSL / LSLU / LSLR / LSLY (and GFPQ) for two epochs, necessity swap / drop / mean and the lens readout;
        eval_student on LSL's saved model (latent_slots.json with the v2.1 fields)."""
        d = self.d
        nec = ["--slot-necessity", "swap,drop,mean", "--slot-swap-token", SWAP, "--slot-decode", "6"]
        ms = {}
        for arm, extra in dict(ARMS21, GFPQ=[]).items():
            more = nec + (["--save"] if arm == "LSL" else []) if arm != "GFPQ" else []
            r = run(common(d) + GFPQ, f"{d}/{arm}", *extra, "--epochs", "2", *more)
            self.ok(r)
            ms[arm] = metrics(f"{d}/{arm}")
            self.assertEqual("latent slots v2.1: readout lens at layer 0" in r.stdout, arm != "GFPQ", arm)
        n_upd = 2 * math.ceil(len(ITEMS) / 2)
        for arm in ARMS21:
            h = ms[arm]["history"]
            self.assertEqual(len(h), n_upd, arm)
            self.assertTrue(all(math.isfinite(x["loss"]) for x in h), arm)
            self.assertEqual("loss_slot" in h[0], arm != "LSLU", arm)
            ls = ms[arm]["latent_slots"]
            self.assertEqual({k: ls[k] for k in ("k", "c", "token", "tokens", "layer", "readout", "fill", "place")},
                             {"k": 6, "c": 2, "token": DIST[0], "tokens": list(DIST), "layer": 0, "readout": "lens",
                              "fill": "last", "place": "before-options"}, arm)
            self.assertEqual(ls["info"]["layer"], 0)
            self.assertEqual(ls["info"]["value_targets"], 30)
            self.assertEqual(ls["views"]["slot_items"], 6)
            nm = ms[arm]["slot_necessity"]["v"]
            self.assertEqual(set(nm), {"swap", "drop", "mean"})
            self.assertEqual((nm["mean"]["mean_layer"], nm["mean"]["mean_items"]), (0, 6))
            for mode in nm:
                self.assertAlmostEqual(nm[mode]["acc_drop"], nm[mode]["acc_intact"] - nm[mode]["acc"])
            ro = ms[arm]["slot_readout"]["v"]
            self.assertEqual((ro["items"], ro["layer"], ro["all"]["n"]), (6, 0, 30))
            rows = [json.loads(l) for l in open(f"{d}/{arm}/slot_readout_v.jsonl")]
            self.assertEqual([s["target"] for s in rows[0]["slots"]], VALUES_660)
            pil = ms[arm]["slot_pilot"]
            self.assertEqual((pil["readout_set"], pil["necessity_mode"]), ("v", "mean"))
            self.assertEqual(pil["pass"], pil["readout_value"] >= 0.45 and pil["acc_drop"] >= 0.02)
            self.assertNotIn("slot_decode", ms[arm])
            self.assertFalse(Path(f"{d}/{arm}/slot_decode_v.jsonl").exists())
        for arm in ("LSL", "LSLR", "LSLU"):
            self.assertEqual({x["seqs"] for x in ms[arm]["history"]}, {2}, arm)
        self.assertEqual([x["seqs"] for x in ms["LSLY"]["history"]], [x["seqs"] for x in ms["GFPQ"]["history"]])
        self.assertGreater(min(x["loss_node"] for x in ms["LSLY"]["history"]), 0)
        self.assertEqual([x["slot_n"] for x in ms["LSL"]["history"]], [x["slot_n"] for x in ms["LSLR"]["history"]])
        self.assertEqual(sum(x["slot_n"] for x in ms["LSL"]["history"]), 2 * 30)
        self.assertNotEqual([x["loss_slot"] for x in ms["LSL"]["history"]],
                            [x["loss_slot"] for x in ms["LSLR"]["history"]])
        self.assertNotEqual([x["loss"] for x in ms["LSLU"]["history"]], [x["loss"] for x in ms["LSL"]["history"]])
        h0 = ms["LSL"]["history"][0]
        self.assertAlmostEqual(h0["loss"], h0["loss_final"] + h0["loss_slot"], places=5)
        side = json.loads(Path(f"{d}/LSL/model/latent_slots.json").read_text())
        self.assertEqual(side, {"k": 6, "c": 2, "token": DIST[0], "place": "before-options", "tokens": list(DIST),
                                "layer": 0, "readout": "lens", "fill": "last"})
        from test_tree_forcing import ENV
        r = subprocess.run([sys.executable, str(ROOT / "scripts/eval_student.py"), "--model", f"{d}/LSL/model",
                            "--eval", f"v={d}/t.jsonl", "--max-len", "640", "--bs", "8", "--out", f"{d}/ev21",
                            "--slot-necessity", "swap,mean", "--slot-swap-token", SWAP, "--slot-mean-train",
                            f"{d}/t.jsonl", "--slot-decode", "6"], capture_output=True, text=True, timeout=900, env=ENV)
        self.ok(r)
        a, b = preds(f"{d}/LSL/preds_v.jsonl"), preds(f"{d}/ev21/preds_v.jsonl")
        for p, q in zip(a, b):                           # bf16 save: close, not equal
            self.assertLess(max(abs(x - y) for x, y in zip(p["probs"], q["probs"])), 2e-2)
        em = metrics(f"{d}/ev21")
        self.assertEqual(em["latent_slots"], side)
        self.assertEqual(set(em["slot_necessity"]["v"]), {"swap", "mean"})
        self.assertEqual(em["slot_readout"]["v"]["all"]["n"], 30)
        r = subprocess.run([sys.executable, str(ROOT / "scripts/eval_student.py"), "--model", f"{d}/LSL/model",
                            "--eval", f"v={d}/t.jsonl", "--max-len", "640", "--out", f"{d}/ev21b", "--slot-necessity",
                            "mean"], capture_output=True, text=True, timeout=900, env=ENV)
        self.assertNotEqual(r.returncode, 0)                                     # mean needs --slot-mean-train

    def test_v21_argument_checks(self):
        d = self.d
        L = ["--latent-slots", "6", "--slot-c", "2", "--slot-tokens", "distinct", "--slot-token", ",".join(DIST),
             "--slot-final-only"]
        bad = [GFPQ + ["--slot-layer", "0"],                                        # no --latent-slots
               GFPQ + L + ["--slot-readout", "lens"],                               # lens without a layer
               GFPQ + L + ["--slot-readout", "lens", "--slot-layer", "1"],           # the last of 2 layers
               GFPQ + L + ["--slot-layer", "0"],                                    # a layer for decode
               GFPQ + L + ["--slot-fill", "last"],                                  # fill for decode
               GFPQ + L + ["--slot-head-grad"],                                     # head grad for decode
               GFPQ + SLOTS + ["--slot-necessity", "mean"],                          # mean without a layer
               GFPQ + ["--latent-slots", "6", "--slot-tokens", "distinct"],          # GPT-2: no reserved tokens
               GFPQ + ["--latent-slots", "6", "--slot-tokens", "distinct", "--slot-token", "§,©"],
               GFPQ + SLOTS21 + ["--slot-necessity", "swap", "--slot-swap-token", DIST[2]],
               GFPQ + SLOTS21 + ["--slot-mean-n", "0"]]
        for argv in bad:
            r = run(common(d) + argv, f"{d}/bad21", "--epochs", "1")
            self.assertNotEqual(r.returncode, 0, argv)
        r = run(common(d) + GFPQ + SLOTS21 + ["--slot-final-only", "--slot-layer", "0.5"], f"{d}/frac", "--epochs", "1")
        self.ok(r)
        self.assertEqual(metrics(f"{d}/frac")["latent_slots"]["layer"], 0)              # round(0.5 x 2) - 1

    def test_v2_arms_match_ref(self):
        """LS / LSU / LSR / LSY (necessity swap / drop / noise, decoding readout; LS saved) and eval_student on the
        saved LS model: LS21_BASE_REF's files."""
        from test_tree_forcing import ENV
        d = self.d
        h = ref_tree(d + "/h21", LS21_BASE_REF)
        if h is None:
            self.skipTest("not a git checkout")
        nec = ["--slot-necessity", "swap,drop,noise@0:1.0", "--slot-swap-token", SWAP, "--slot-decode", "3"]
        drop = re.compile(r"s_per_step|approx_tok_per_s")
        strip = lambda hh: [{k: v for k, v in row.items() if not drop.match(k)} for row in hh]
        for arm, extra in ARMS.items():
            outs = []
            for name, script in (("ref", str(h / "scripts/train_student.py")), ("now", None)):
                o = f"{d}/v2{name}-{arm}"
                self.ok(run(common(d) + GFPQ, o, *extra, "--epochs", "2", *nec, *(["--save"] if arm == "LS" else []),
                            script=script))
                outs.append(Path(o))
            a, b = metrics(outs[0]), metrics(outs[1])
            self.assertEqual(strip(a["history"]), strip(b["history"]), arm)
            for key in ("eval", "latent_slots", "slot_necessity", "slot_decode", "tree_view_stats"):
                self.assertEqual(a.get(key), b.get(key), (arm, key))
            self.assertEqual(sorted(set(a) - {"args"}), sorted(set(b) - {"args"}), arm)
            self.assertEqual(sorted(p.name for p in outs[0].iterdir()), sorted(p.name for p in outs[1].iterdir()))
            for f in sorted(p.name for p in outs[0].glob("*.jsonl")) + ["latent_slots.json"]:
                self.assertEqual((outs[0] / f).read_text(), (outs[1] / f).read_text(), (arm, f))
        self.assertEqual(Path(f"{d}/v2ref-LS/model/latent_slots.json").read_text(),
                         Path(f"{d}/v2now-LS/model/latent_slots.json").read_text())
        for name, script in (("ref", str(h / "scripts/eval_student.py")), ("now", str(ROOT / "scripts/eval_student.py"))):
            r = subprocess.run([sys.executable, script, "--model", f"{d}/v2now-LS/model", "--eval", f"v={d}/t.jsonl",
                                "--max-len", "640", "--out", f"{d}/v2ev-{name}", "--slot-necessity", "swap,drop",
                                "--slot-swap-token", SWAP, "--slot-decode", "2"], capture_output=True, text=True,
                               timeout=900, env=ENV)
            self.ok(r)
        for f in ("preds_v.jsonl", "preds_v@swap.jsonl", "preds_v@drop.jsonl", "slot_decode_v.jsonl"):
            self.assertEqual(Path(f"{d}/v2ev-ref/{f}").read_text(), Path(f"{d}/v2ev-now/{f}").read_text(), f)
        ea, eb = metrics(f"{d}/v2ev-ref"), metrics(f"{d}/v2ev-now")
        for key in ("eval", "latent_slots", "slot_necessity", "slot_decode"):
            self.assertEqual(ea.get(key), eb.get(key), key)


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestV21FSDP(unittest.TestCase):
    """LSL (and LSLY on one rank) under --fsdp on the CPU (byte-level tiny model of test_fsdp, distinct tokens ~ ^ | {,
    swap }): the single-process updates, evaluation, necessity (swap / mean / noise) and lens readout."""

    def test_lsl_fsdp_equal_single(self):
        import test_fsdp as tf
        with tempfile.TemporaryDirectory() as d:
            tf.tiny_model(d)
            self.assertFalse(any(c in it["prompt"] for it in ITEMS for c in "~^|{}"))
            Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in ITEMS))
            base = ["--model", d + "/m", "--train", d + "/t.jsonl", "--eval", f"v={d}/t.jsonl", *GFPQ,
                    "--items-per-update", "3", "--micro-bs", "2", "--max-len", "1024", "--eval-max-len", "1024",
                    "--log-every", "1", "--lr", "1e-3", "--seed", "0", "--epochs", "2", "--precision", "fp32master",
                    "--latent-slots", "4", "--slot-c", "2", "--slot-tokens", "distinct", "--slot-token", "~,^,|,{",
                    "--slot-layer", "0", "--slot-readout", "lens", "--slot-fill", "last",
                    "--slot-necessity", "swap,mean,noise@0:1.0", "--slot-swap-token", "}", "--slot-decode", "6"]
            plan = (("LSL", ["--slot-final-only"], (("f2", 2), ("f1", 1))), ("LSLY", [], (("f1", 1),)))
            for arm, extra, ranks in plan:
                runs = {"single": tf.single(d, *base, *extra, "--out", f"{d}/{arm}-s")}
                for tag, n in ranks:
                    runs[tag] = tf.fsdp(d, *base, *extra, "--fsdp-optim", "cpu-adamw", "--out", f"{d}/{arm}-{tag}",
                                        nproc=n)
                for k, r in runs.items():
                    self.assertEqual(r.returncode, 0, (arm, k, r.stderr[-3000:]))
                a = tf.metrics(f"{d}/{arm}-s")
                self.assertGreater(sum(x["slot_n"] for x in a["history"]), 0)
                for tag, _ in ranks:
                    b = tf.metrics(f"{d}/{arm}-{tag}")
                    self.assertEqual(len(a["history"]), len(b["history"]))
                    for x, y in zip(a["history"], b["history"]):
                        self.assertAlmostEqual(x["loss"], y["upd_loss"], delta=1e-5, msg=(arm, tag))
                        self.assertAlmostEqual(x["loss_slot"], y["loss_slot"], delta=1e-5, msg=(arm, tag))
                        self.assertEqual(x["slot_n"], y["slot_n"], (arm, tag))
                    for p, q in zip(tf.preds(f"{d}/{arm}-s"), tf.preds(f"{d}/{arm}-{tag}")):
                        self.assertLess(max(abs(u - v) for u, v in zip(p["probs"], q["probs"])), 1e-4)
                    for mode in ("swap", "mean", "noise@0:1.0"):
                        self.assertAlmostEqual(a["slot_necessity"]["v"][mode]["acc"],
                                               b["slot_necessity"]["v"][mode]["acc"], msg=(arm, tag, mode))
                        self.assertAlmostEqual(a["slot_necessity"]["v"][mode]["nll"],
                                               b["slot_necessity"]["v"][mode]["nll"], delta=1e-4, msg=(arm, tag, mode))
                    ra = [json.loads(l) for l in open(f"{d}/{arm}-s/slot_readout_v.jsonl")]
                    rb = [json.loads(l) for l in open(f"{d}/{arm}-{tag}/slot_readout_v.jsonl")]
                    self.assertEqual(ra, rb, (arm, tag))
                    self.assertEqual(a["slot_pilot"], b["slot_pilot"], (arm, tag))


if __name__ == "__main__":
    unittest.main()
