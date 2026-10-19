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


if __name__ == "__main__":
    unittest.main()
