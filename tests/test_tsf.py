"""TSF (teacher-state slots with forced reading) and PTAD (answer-position self-distillation, the zero-latent CODI
backup), notes/latent_comeback_memo_2026-10-21.md sections 3 / 4; train_student.py --slot-readout tsf / --answer-teacher:
- slot_step_map (truncate / merge / short chains) and parse_layer_list;
- the teacher view of gsm8k-train-660: the SI stage-0 text (insert_chain + chain_block, no slot line), the marks of
  slots 1-2 (truncate: steps 1, 2) and 1-6 (K = 6, merge: the 8-step 6669 puts step 8 on slot 6); the token positions
  (GPT-2: ' =' / ' 6'; the Qwen-style digit split of the model-dir tokenizer and of Qwen3.5 when cached: ' ' / '6');
  the head cut of max_len; shuffled partners (keyed by seed, never the item's own lines, enough steps, no draw from the
  global rng); placebo lines (the problem's sentences, cycled) and their marks (the line's last two tokens);
- slot_group with a teacher plan (TeacherExample; LSL's prompt) and answer_group (PTAD: final view only, A2 prompt);
- _LayerCapture = hidden_states; forcing replaces only the given rows / positions of layer L's output; the gradient of a
  forced pass with gradient checkpointing (non-reentrant, reentrant) equals the one without (Qwen2, and a Gated
  DeltaNet layer of a hybrid Qwen3.5);
- teacher_forward (TSF): the letters of label_logits when not forced; the forced micro-batch only when the coin says
  (p = 0 never, p = 1 always; the coin keyed by content, not order); the align loss and the teacher CE = manual
  computations; the letter CE's gradient reaches the writing layers' slot states in a normal micro-batch and not in a
  forced one, the align loss's in both; shuffled / placebo teachers give other targets; the FSDP weight-0 stand-in;
  PTAD: the align loss at "Answer:" = manual;
- tsf_eval: force = the forced training pass, forceswap another item's vectors, unchained items unchanged, the
  alignment loss; tsf_subsets;
- end to end on a tiny 4-layer model: TSF / TSFN / TSFR / TSFX / TSF6 / PTAD / PTADR / PTADX (the detector at every
  --eval-at point, necessity incl. force / forceswap, tsf_pilot, the gradient probe and its rescale, the CE guard, the
  sidecar, eval_student on a saved TSF model), resume, the argument checks;
- TSF and PTAD under --fsdp (CPU gloo, 1 and 2 ranks) = the single process;
- the existing arms (A2, GFPQ, LS, LSL / LSLU / LSLR / LSLY, eval_student; LSL under --fsdp on one rank) give
  LS23_BASE_REF's files byte for byte.
CPU only (CUDA_VISIBLE_DEVICES="")."""
import dataclasses
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

from test_latent_slots import (DIST, GSM, HAVE, ITEMS, LINES_660, QWEN, SWAP, VALUES_660, gfpq_group, gpt2,  # noqa: E402
                               metrics, preds, ref_tree, run)

# The last commit before TSF / PTAD: the existing arms are compared with it.
LS23_BASE_REF = os.environ.get("LS23_BASE_REF", "602ceb5")

if HAVE:
    import torch
    import transformers


def tsf_spec(k=2, layer=2, overflow="truncate", tokens=DIST):
    from cotdistill.student import SlotSpec
    return SlotSpec(k, 2, tokens[0], tokens=tuple(tokens[:k]), layer=layer, readout="tsf", overflow=overflow)


def tiny4(tok, ckpt=None, tied=False, n=4):
    """A random 4-layer Qwen2 (hidden 32); ckpt None / {"use_reentrant": False} / {"use_reentrant": True}."""
    torch.manual_seed(0)
    m = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=n, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=1024, tie_word_embeddings=tied))
    if ckpt is not None:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs=ckpt)
    return m.train()


def tsf_batch(spec, kind="steps", ids=("gsm8k-train-660", "gsm8k-train-3365"), seed=0, lam=1.0):
    from cotdistill.student import plan_slot_targets, slot_group, teacher_partners, teacher_plan
    out = []
    partners = teacher_partners(ITEMS, seed) if kind == "shuffled" else {}
    for i in ids:
        pl = teacher_plan(GSM[i], kind, k=spec.k, overflow=spec.overflow, partner=GSM.get(partners.get(i)))
        out += slot_group(gfpq_group(GSM[i]), GSM[i], spec=spec, targets=plan_slot_targets(pl, spec.k), lam=lam,
                          final_only=True, teacher=pl)
    return out


# ---------------------------------------------------------------------------------------------------- plans, views
class TestPlan(unittest.TestCase):
    def test_step_map_and_layers(self):
        from cotdistill.student import parse_layer_list, slot_step_map
        self.assertEqual(slot_step_map(6, 2, "truncate"), [0, 1])
        self.assertEqual(slot_step_map(6, 2, "merge"), [0, 5])
        self.assertEqual(slot_step_map(1, 2, "truncate"), [0, None])
        self.assertEqual(slot_step_map(0, 2, "truncate"), [None, None])
        self.assertEqual(slot_step_map(8, 6, "merge"), [0, 1, 2, 3, 4, 7])        # TSF6: first five + the last
        self.assertEqual(slot_step_map(6, 6, "merge"), list(range(6)))
        self.assertEqual(slot_step_map(3, 6, "merge"), [0, 1, 2, None, None, None])
        with self.assertRaises(ValueError):
            slot_step_map(3, 2, "keep")
        self.assertEqual(parse_layer_list("12-20", 32), tuple(range(12, 21)))
        self.assertEqual(parse_layer_list("12-27", 32), tuple(range(12, 28)))
        self.assertEqual(parse_layer_list("3, 1,1", 8), (1, 3))
        for bad in ("31", "x", "", "0-31", "-1"):
            with self.assertRaises(ValueError, msg=bad):
                parse_layer_list(bad, 32)

    def test_660_teacher_view(self):
        """SI stage 0: the final question with all six steps before the options; slot 1 <- step 1 ('6'), slot 2 <- step
        2 ('8'); K = 6 merge: steps 1..6 (660), and the 8-step 6669 puts step 8's 260 on slot 6."""
        from cotdistill.student import (FINAL_TEMPLATE, chain_block, insert_chain, plan_slot_targets,
                                        teacher_plan, teacher_text_marks)
        it = GSM["gsm8k-train-660"]
        text = FINAL_TEMPLATE.format(problem=it["prompt"])
        pl = teacher_plan(it, "steps", k=2, overflow="truncate")
        self.assertEqual((pl.lines, pl.slot_lines, pl.rule), (tuple(LINES_660), (0, 1), "value"))
        t, mk = teacher_text_marks(text, pl)
        self.assertEqual(t, insert_chain(text, chain_block(LINES_660)))
        self.assertIn("\n\nWorked steps:\nStep 1: 4 + 2 = 6\nStep 2: 4 × 2 = 8\n", t)
        self.assertEqual([t[a:b + 1] for a, b in mk], ["6", "8"])
        self.assertEqual(plan_slot_targets(pl, 2), LINES_660[:2])
        p6 = teacher_plan(it, "steps", k=6, overflow="merge")
        t6, mk6 = teacher_text_marks(text, p6)
        self.assertEqual(t6, t)
        self.assertEqual([t6[a:b + 1] for a, b in mk6], VALUES_660)
        x = GSM["gsm8k-train-6669"]
        px = teacher_plan(x, "steps", k=6, overflow="merge")
        tx, mx = teacher_text_marks(FINAL_TEMPLATE.format(problem=x["prompt"]), px)
        self.assertEqual(px.slot_lines, (0, 1, 2, 3, 4, 7))
        self.assertEqual([tx[a:b + 1] for a, b in mx], ["2.5", "3.75", "1.5", "6.25", "7.75", "260"])
        none = teacher_plan(GSM["gsm8k-train-5699"], "steps", k=2)              # no chain
        self.assertEqual((none.lines, none.slot_lines), ((), (None, None)))
        self.assertEqual(teacher_text_marks(text, none), ("", (None, None)))
        short = teacher_plan(GSM["gsm8k-train-3929"], "steps", k=6, overflow="merge")   # 2 steps
        self.assertEqual(short.slot_lines, (0, 1, None, None, None, None))

    @unittest.skipUnless(HAVE, "needs torch + transformers")
    def test_positions(self):
        """Token 1 <- the token BEFORE the value's first token (it predicts it), token 2 <- the value's last token:
        GPT-2 ' =' / ' 6'; a Qwen-style digit split (the tiny models' model-dir tokenizer) ' ' / '6', and for 10 + 8 =
        18 ' ' / '8'; positions in collate's row (shifted by the head cut; None when cut away)."""
        from cotdistill.student import (FINAL_TEMPLATE, collate, encode_tail, teacher_mark_positions, teacher_plan,
                                        teacher_text_marks)
        it = GSM["gsm8k-train-660"]
        text = FINAL_TEMPLATE.format(problem=it["prompt"])
        t, mk = teacher_text_marks(text, teacher_plan(it, "steps", k=6, overflow="merge"))
        g = gpt2()
        pos = teacher_mark_positions(g, t, mk, 640)
        ids = encode_tail(g, t, 640)
        self.assertEqual([[g.decode([ids[q]]) for q in pp] for pp in pos],
                         [[" =", " 6"], [" =", " 8"], [" =", " 2"], [" =", " 10"], [" =", " 18"], [" =", " 20"]])
        from cotdistill.student import Example
        row = collate(g, [Example(t, ["A"], [1.0], 1.0, "final", "x")], 640, "cpu")["input_ids"][0].tolist()
        self.assertEqual(row, ids)
        cut = len(ids) - 60                                                   # max_len 60: the head is cut
        short = teacher_mark_positions(g, t, mk, 60)
        self.assertEqual(short, [None if pp[0] - cut < 0 else (pp[0] - cut, pp[1] - cut) for pp in pos])
        self.assertEqual(short[0], None)
        with tempfile.TemporaryDirectory() as d:                              # Qwen2 class over GPT-2's vocabulary
            from test_tree_forcing import tiny
            tiny(d)
            q = transformers.AutoTokenizer.from_pretrained(d + "/m")
            qi = encode_tail(q, t, 640)
            got = [[q.decode([qi[a]]) for a in pp] for pp in teacher_mark_positions(q, t, mk, 640)]
            self.assertEqual(got, [[" ", "6"], [" ", "8"], [" ", "2"], [" ", "0"], [" ", "8"], [" ", "0"]])

    @unittest.skipUnless(HAVE and QWEN, "no local Qwen3.5 tokenizer (Sev-2B-preview snapshot)")
    def test_qwen35_positions(self):
        """Qwen3.5 writes '= 6' as ' =', ' ', '6': slot token 1 <- the bare space, 2 <- the value's last digit."""
        from cotdistill.student import FINAL_TEMPLATE, encode_tail, teacher_mark_positions, teacher_plan, teacher_text_marks
        tok = transformers.AutoTokenizer.from_pretrained(QWEN[0])
        for iid, k, ov, want in (("gsm8k-train-660", 2, "truncate", [["Ġ", "6"], ["Ġ", "8"]]),
                                 ("gsm8k-train-660", 6, "merge", [["Ġ", "6"], ["Ġ", "8"], ["Ġ", "2"], ["Ġ", "0"],
                                                                  ["Ġ", "8"], ["Ġ", "0"]]),
                                 ("gsm8k-train-6669", 6, "merge", [["Ġ", "5"], ["Ġ", "5"], ["Ġ", "5"], ["Ġ", "5"],
                                                                   ["Ġ", "5"], ["Ġ", "0"]])):
            it = GSM[iid]
            t, mk = teacher_text_marks(FINAL_TEMPLATE.format(problem=it["prompt"]), teacher_plan(it, "steps", k=k,
                                                                                                 overflow=ov))
            ids = encode_tail(tok, t, 1536)
            got = [tok.convert_ids_to_tokens([ids[a], ids[b]]) for a, b in teacher_mark_positions(tok, t, mk, 1536)]
            self.assertEqual(got, want, iid)

    def test_shuffled_and_placebo_plans(self):
        from cotdistill.student import (FINAL_TEMPLATE, placebo_lines, slot_step_map, teacher_partners, teacher_plan,
                                        teacher_text_marks, worked_steps)
        state = random.getstate()
        need = {it["item_id"]: sum(x is not None for x in slot_step_map(len(worked_steps(it) or []), 2, "truncate"))
                for it in ITEMS}
        a = teacher_partners(ITEMS, 3, need)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(a, teacher_partners(ITEMS, 3, need))
        chains = {it["item_id"]: [s["text"] for s in worked_steps(it) or []] for it in ITEMS}
        self.assertEqual(set(a), {i for i, c in chains.items() if c})                  # every chained item, 5
        for iid, p in a.items():
            self.assertNotEqual(p, iid)
            self.assertNotEqual(chains[p], chains[iid])
            self.assertGreaterEqual(len(chains[p]), need[iid])
            pl = teacher_plan(GSM[iid], "shuffled", k=2, overflow="truncate", partner=GSM[p])
            self.assertEqual(pl.lines, tuple(chains[p]))
            own = teacher_plan(GSM[iid], "steps", k=2, overflow="truncate")
            self.assertEqual([x is None for x in pl.slot_lines], [x is None for x in own.slot_lines])
        self.assertTrue(any(teacher_partners(ITEMS, s, need) != a for s in range(4, 12)))
        it = GSM["gsm8k-train-660"]
        lines = placebo_lines(it, 6)                                                   # 5 sentences, cycled
        self.assertEqual(lines[0], "Abe finds 4 ants on the sidewalk.")
        self.assertEqual(lines[5], lines[0])
        px = teacher_plan(it, "placebo", k=2, overflow="truncate")
        self.assertEqual((px.lines, px.slot_lines, px.rule), (tuple(lines), (0, 1), "last"))
        t, mk = teacher_text_marks(FINAL_TEMPLATE.format(problem=it["prompt"]), px)
        self.assertEqual([t[a:b + 1] for a, b in mk], [".", "."])
        self.assertNotIn("4 + 2 = 6", t)
        with self.assertRaises(ValueError):
            teacher_plan(it, "shuffled", k=2)                                          # no partner

    @unittest.skipUnless(HAVE, "needs torch + transformers")
    def test_groups(self):
        """slot_group with a plan: the final view as a TeacherExample (LSL's slot line; the teacher text without it,
        lambda / supervised slots); an item without a chain: no teacher view, lambda 0; inputs and rng untouched.
        answer_group (PTAD): the final view only, the A2 prompt, lambda."""
        from cotdistill.student import (FINAL_TEMPLATE, TeacherExample, answer_group, set_slot_prompt, slot_group,
                                        teacher_plan, teacher_text_marks)
        tok = gpt2()
        spec = tsf_spec()
        set_slot_prompt(tok, spec)
        self.addCleanup(set_slot_prompt, tok, None)
        it = GSM["gsm8k-train-660"]
        g0 = gfpq_group(it)
        state = random.getstate()
        b = tsf_batch(spec, lam=4.0, ids=("gsm8k-train-660", "gsm8k-train-3929", "gsm8k-train-5699"))
        self.assertEqual(random.getstate(), state)
        self.assertEqual([type(e).__name__ for e in b], ["TeacherExample"] * 3)
        e = b[0]
        self.assertIn("\n§§©©\nOptions:\n", e.text)
        self.assertNotIn("§", e.teacher_text)
        self.assertEqual(e.teacher_text, teacher_text_marks(g0[0].text, teacher_plan(it, "steps", k=2,
                                                                                     overflow="truncate"))[0])
        self.assertEqual((e.slot_targets, e.slot_lambda), (tuple(LINES_660[:2]), 2.0))
        self.assertEqual(len(e.teacher_marks), 2)
        self.assertEqual((b[1].slot_lambda, b[1].slot_targets), (2.0, ("1/2 × 4 = 2", "3 × 2 = 6")))
        self.assertEqual((b[2].teacher_text, b[2].slot_lambda, b[2].teacher_marks), ("", 0.0, (None, None)))
        self.assertEqual(dataclasses.astuple(gfpq_group(it)[0]), dataclasses.astuple(g0[0]))
        a = answer_group(g0, it, plan=teacher_plan(it, "steps"), lam=1.5)
        self.assertEqual(len(a), 1)
        self.assertIsInstance(a[0], TeacherExample)
        self.assertEqual(a[0].text, FINAL_TEMPLATE.format(problem=it["prompt"]))
        self.assertEqual((a[0].slot_targets, a[0].slot_lambda, a[0].teacher_marks), ((), 1.5, ()))
        self.assertIn("Step 6: 18 + 2 = 20\nOptions:", a[0].teacher_text)
        n = answer_group(gfpq_group(GSM["gsm8k-train-5699"]), GSM["gsm8k-train-5699"],
                         plan=teacher_plan(GSM["gsm8k-train-5699"], "steps"), lam=1.5)
        self.assertEqual((n[0].teacher_text, n[0].slot_lambda), ("", 0.0))


# ---------------------------------------------------------------------------------------------------- forward passes
@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestCapture(unittest.TestCase):
    def test_capture_and_force(self):
        """_LayerCapture = hidden_states[l + 1]; force replaces layer L's output at the given rows / positions only
        (layer L + 1 sees it), the capture of L is the un-replaced output; positions before the first replaced one
        keep their final states (causal)."""
        from cotdistill.student import _LayerCapture, text_parts
        m = tiny4(gpt2()).eval()
        body, _ = text_parts(m)
        ids = torch.randint(3, 500, (2, 12), generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            hs = body(input_ids=ids, output_hidden_states=True).hidden_states
            with _LayerCapture(body, [0, 1, 2]) as cap:
                h0 = body(input_ids=ids).last_hidden_state
            for li in (0, 1, 2):
                self.assertTrue(torch.allclose(cap.h[li], hs[li + 1], atol=1e-6))
            vals = torch.randn(3, 32, generator=torch.Generator().manual_seed(2))
            rows, pos = torch.tensor([0, 0, 1]), torch.tensor([5, 6, 9])
            seen = []
            hd = m.model.layers[3].register_forward_pre_hook(lambda mod, a, k: seen.append((a[0] if a else
                                                                                            k["hidden_states"]).clone()),
                                                             with_kwargs=True)
            with _LayerCapture(body, [1, 2], force=(2, rows, pos, vals)) as cap:
                h1 = body(input_ids=ids).last_hidden_state
            hd.remove()
            self.assertTrue(torch.allclose(cap.h[2], hs[3], atol=1e-6))              # before the replacement
            want = hs[3].clone()
            want[rows, pos] = vals
            self.assertTrue(torch.allclose(seen[0], want, atol=1e-6))
            self.assertTrue(torch.allclose(h1[0, :5], h0[0, :5], atol=1e-6))
            self.assertTrue(torch.allclose(h1[1, :9], h0[1, :9], atol=1e-6))
            self.assertFalse(torch.allclose(h1[0, 5:], h0[0, 5:], atol=1e-4))
            with self.assertRaises(ValueError):
                _LayerCapture(body, [3])                                             # the last layer: read by nobody

    def test_forced_gradient_with_checkpointing(self):
        """A forced pass's parameter gradients with gradient checkpointing (non-reentrant / reentrant) = without; the
        layer is checkpointed again afterwards; the replaced positions get no gradient from the readout."""
        from cotdistill.student import _LayerCapture, text_parts
        tok = gpt2()
        ids = torch.randint(3, 500, (2, 12), generator=torch.Generator().manual_seed(1))
        vals = torch.randn(4, 32, generator=torch.Generator().manual_seed(2))
        rows, pos = torch.tensor([0, 0, 1, 1]), torch.tensor([4, 5, 7, 8])
        grads = {}
        for kw in (None, {"use_reentrant": False}, {"use_reentrant": True}):
            m = tiny4(tok, kw)
            body, _ = text_parts(m)
            with _LayerCapture(body, [2], force=(2, rows, pos, vals)) as cap:
                self.assertFalse(m.model.layers[2].gradient_checkpointing)
                h = body(input_ids=ids, use_cache=False).last_hidden_state
            if kw is not None:
                self.assertTrue(m.model.layers[2].gradient_checkpointing)
            cap.h[2].retain_grad()
            h[:, -1].pow(2).sum().backward()
            grads[str(kw)] = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
            self.assertEqual(float(cap.h[2].grad[rows, pos].abs().sum()), 0.0, kw)
            self.assertGreater(float(cap.h[2].grad[0, 3].abs().sum()), 0.0, kw)
        ref = grads["None"]
        for k, g in grads.items():
            self.assertEqual(set(g), set(ref), k)
            for n in ref:
                self.assertTrue(torch.allclose(g[n], ref[n], atol=1e-6, rtol=1e-5), (k, n))


    def test_deltanet_layer(self):
        """The same on a hybrid Qwen3.5 (8 layers, full attention at 3 and 7) forcing at the Gated DeltaNet layer 5 (as
        layer 20 of 32 on 4B / 9B), in a subprocess without fla (its triton kernels need a GPU)."""
        from test_tree_forcing import ENV
        code = ("import sys; sys.modules['fla'] = None; sys.path[:0] = [%r, %r]; import test_tsf as T; "
                "T.deltanet_force_check(); print('deltanet ok')" % (str(ROOT / "src"), str(ROOT / "tests")))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, env=ENV)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("deltanet ok", r.stdout)


def deltanet_force_check() -> None:
    """Force at layer 5 (linear attention) of qwen35_tiny(8) in training, gradient checkpointing off / non-reentrant /
    reentrant: the captures of 4 and 5 = hidden_states, layer 6 reads the replaced states, equal parameter gradients."""
    from test_latent_slots import qwen35_tiny
    from cotdistill.fsdp import decoder_layers
    from cotdistill.student import _LayerCapture, text_parts
    ids = torch.randint(3, 64, (2, 12), generator=torch.Generator().manual_seed(1))
    vals = torch.randn(4, 16, generator=torch.Generator().manual_seed(2))
    rows, pos = torch.tensor([0, 0, 1, 1]), torch.tensor([4, 5, 7, 8])
    ref = None
    for kw in (None, {"use_reentrant": False}, {"use_reentrant": True}):
        m = qwen35_tiny(8).train()
        assert m.config.layer_types[5] == "linear_attention" and m.config.layer_types[3] == "full_attention"
        if kw is not None:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kw)
        body, _ = text_parts(m)
        dl = decoder_layers(body)
        m.eval()
        with torch.no_grad():
            hs = body(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
        m.train()
        seen = []
        hd = dl[6].register_forward_pre_hook(lambda mod, a, k: seen.append((a[0] if a else k["hidden_states"])
                                                                           .detach().clone()), with_kwargs=True)
        with _LayerCapture(body, [4, 5], force=(5, rows, pos, vals)) as cap:
            h = body(input_ids=ids, use_cache=False).last_hidden_state
        hd.remove()
        assert torch.allclose(cap.h[4], hs[5], atol=1e-5) and torch.allclose(cap.h[5], hs[6], atol=1e-5), kw
        want = hs[6].clone()
        want[rows, pos] = vals
        assert torch.allclose(seen[0], want, atol=1e-5), kw
        assert kw is None or dl[5].gradient_checkpointing
        h[:, -1].pow(2).sum().backward()
        g = torch.cat([q.grad.flatten() for q in m.parameters() if q.grad is not None])
        if ref is None:
            ref = g
        else:
            assert g.shape == ref.shape and torch.allclose(g, ref, atol=1e-6, rtol=1e-5), kw


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestTSFForward(unittest.TestCase):
    def setUp(self):
        from cotdistill.student import TeacherConfig, set_slot_prompt, set_teacher
        self.tok = gpt2()
        self.spec = tsf_spec()
        set_slot_prompt(self.tok, self.spec)
        self.cfg = TeacherConfig("tsf", "steps", (1, 2), force_p=0.0, teacher_ce=0.5, seed=0)
        set_teacher(self.tok, self.cfg)
        self.addCleanup(set_slot_prompt, self.tok, None)
        self.addCleanup(set_teacher, self.tok, None)

    def manual(self, m, b, forced: bool):
        """(letter logits, align sum, teacher CE sum) by hand: the teacher views' hidden_states at layers 1, 2 at the
        marked positions; the student pass with the layer-2 slot states replaced when forced."""
        from cotdistill.student import (Example, collate, kl_loss, label_logits, teacher_mark_positions)
        sids = [self.tok.encode(t)[0] for t in DIST[:2]]
        tex = [Example(e.teacher_text, e.labels, e.target, e.weight * 0.5, "teacher", e.item_id) for e in b
               if e.teacher_text]
        with torch.no_grad():
            ct = collate(self.tok, tex, 640, "cpu")
            ht = m.model(input_ids=ct["input_ids"], attention_mask=ct["attention_mask"], output_hidden_states=True)
            tce = float(kl_loss(label_logits(m, self.tok, tex, 640, {}), tex) * len(tex))
            T = []
            for r, e in enumerate(b):
                for pp in teacher_mark_positions(self.tok, e.teacher_text, e.teacher_marks, 640):
                    T.append(torch.stack([ht.hidden_states[li + 1][r, list(pp)] for li in (1, 2)]))
            cs = collate(self.tok, b, 640, "cpu")
            P = []
            for r, e in enumerate(b):
                row = cs["input_ids"][r].tolist()
                for j in range(2):
                    P.append((r, [q for q, x in enumerate(row) if x == sids[j]]))
            seen = {}

            def keep(li):
                def pre(mod, a, k):
                    seen.setdefault(li, a[0] if a else k["hidden_states"])
                return pre

            def force(mod, a, out):
                h = (out[0] if isinstance(out, tuple) else out).clone()
                for (r, ps), t in zip(P, T):
                    h[r, ps] = t[1]
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            hds = [m.model.layers[2].register_forward_pre_hook(keep(1), with_kwargs=True),
                   m.model.layers[3].register_forward_pre_hook(keep(2), with_kwargs=True)]
            if forced:
                hds.append(m.model.layers[2].register_forward_hook(force))
                orig = []
                hds.append(m.model.layers[2].register_forward_hook(lambda mod, a, out: orig.append(out), prepend=True))
            hidden = m.model(input_ids=cs["input_ids"], attention_mask=cs["attention_mask"]).last_hidden_state
            for hd in hds:
                hd.remove()
            if forced:
                seen[2] = orig[0][0] if isinstance(orig[0], tuple) else orig[0]
            z = [m.lm_head(hidden[r, cs["last"][r]])[self.ids(e)] for r, e in enumerate(b)]
            tot = 0.0
            for (r, ps), t, w in zip(P, T, [e.weight * e.slot_lambda for e in b for _ in range(2)]):
                s = torch.stack([seen[li][r, ps] for li in (1, 2)])
                sig = t.pow(2).mean(-1, keepdim=True).sqrt()
                tot += w * float(torch.nn.functional.smooth_l1_loss(s / sig, t / sig))
        return z, tot, tce

    def ids(self, e):
        from cotdistill.student import letter_token_ids
        return torch.tensor(letter_token_ids(self.tok, e.labels))

    def test_unforced_and_forced_manual(self):
        from cotdistill.student import label_logits, teacher_forward
        m = tiny4(self.tok).eval()
        b = tsf_batch(self.spec)
        z, parts, st = teacher_forward(m, self.tok, b, 640, {}, step=0)
        self.assertEqual((st["forced"], st["slot_n"], st["teacher_n"]), (0, 4, 2))
        for x, y in zip(z, label_logits(m, self.tok, b, 640, {})):
            self.assertTrue(torch.allclose(x, y, atol=1e-6))
        mz, align, tce = self.manual(m, b, forced=False)
        for x, y in zip(z, mz):
            self.assertTrue(torch.allclose(x, y, atol=1e-5))
        self.assertAlmostEqual(float(parts["align"]), align, places=5)
        self.assertAlmostEqual(float(parts["teacher"]), tce, places=5)
        self.cfg.force_p = 1.0
        z1, parts1, st1 = teacher_forward(m, self.tok, b, 640, {}, step=0)
        self.assertEqual((st1["forced"], st1["forced_n"]), (1, 2))
        mz1, align1, _ = self.manual(m, b, forced=True)
        for x, y, u in zip(z1, mz1, z):
            self.assertTrue(torch.allclose(x, y, atol=1e-5))
            self.assertFalse(torch.allclose(x, u, atol=1e-4))
        self.assertAlmostEqual(float(parts1["align"]), align1, places=5)   # taken before the replacement
        self.assertAlmostEqual(float(parts1["align"]), float(parts["align"]), places=5)
        self.cfg.scale = 3.0
        self.assertAlmostEqual(float(teacher_forward(m, self.tok, b, 640, {})[1]["align"]), 3 * align, places=4)

    def test_coin(self):
        """Forced iff tsf_coin(seed, step, items) < p, keyed by the items (not their order); p 0 never, 1 always."""
        from cotdistill.student import teacher_forward, tsf_coin
        m = tiny4(self.tok).eval()
        b = tsf_batch(self.spec)
        self.assertEqual(tsf_coin(0, 7, b), tsf_coin(0, 7, b[::-1]))
        self.assertNotEqual(tsf_coin(0, 7, b), tsf_coin(0, 8, b))
        self.assertNotEqual(tsf_coin(0, 7, b), tsf_coin(1, 7, b))
        frac = sum(tsf_coin(0, s, b) < 0.5 for s in range(400)) / 400
        self.assertTrue(0.4 < frac < 0.6, frac)
        self.cfg.force_p = 0.5
        for s in range(6):
            self.assertEqual(teacher_forward(m, self.tok, b, 640, {}, step=s)[2]["forced"], int(tsf_coin(0, s, b) < 0.5))
        self.cfg.force_p = 0.0
        self.assertEqual(sum(teacher_forward(m, self.tok, b, 640, {}, step=s)[2]["forced"] for s in range(6)), 0)

    def test_gradient_paths(self):
        """The letter CE's gradient reaches the layer-2 (writing) slot states in a normal micro-batch and not in a
        forced one (replaced); the align loss's reaches them in both; the teacher CE trains the weights."""
        from cotdistill.student import collate, kl_loss, teacher_forward
        b = tsf_batch(self.spec)
        sids = [self.tok.encode(t)[0] for t in DIST[:2]]
        mask = torch.isin(collate(self.tok, b, 640, "cpu")["input_ids"], torch.tensor(sids))
        for p in (0.0, 1.0):
            self.cfg.force_p = p
            for term in ("ce", "align", "teacher"):
                m = tiny4(self.tok)
                seen = []

                def grab(mod, a, out):
                    h = out[0] if isinstance(out, tuple) else out
                    if h.shape[:2] == mask.shape and not seen:
                        h.retain_grad()
                        seen.append(h)
                hd = m.model.layers[2].register_forward_hook(grab, prepend=True)
                z, parts, _ = teacher_forward(m, self.tok, b, 640, {})
                hd.remove()
                loss = {"ce": kl_loss(z, b) * len(b), "align": parts["align"], "teacher": parts["teacher"]}[term]
                loss.backward()
                g = seen[0].grad
                slot_g = float(g[mask].abs().sum()) if g is not None else 0.0
                if term == "ce":
                    self.assertEqual(slot_g > 0, p == 0.0, (p, term))
                    self.assertGreater(float(g[~mask].abs().sum()), 0)            # problem tokens still read
                elif term == "align":
                    self.assertGreater(slot_g, 0, (p, term))
                else:
                    self.assertGreater(sum(float(q.grad.abs().sum()) for q in m.parameters() if q.grad is not None), 0)

    def test_teacher_kinds_differ(self):
        """shuffled (other items' lines) and placebo (the problem's sentences) give other teacher texts and targets,
        the same supervised slots."""
        from cotdistill.student import teacher_forward
        m = tiny4(self.tok).eval()
        res = {}
        for kind in ("steps", "shuffled", "placebo"):
            b = tsf_batch(self.spec, kind=kind)
            _, parts, st = teacher_forward(m, self.tok, b, 640, {})
            res[kind] = (tuple(e.teacher_text for e in b), float(parts["align"]), st["slot_n"])
        self.assertEqual({v[2] for v in res.values()}, {4})
        for a, c in (("steps", "shuffled"), ("steps", "placebo"), ("shuffled", "placebo")):
            self.assertNotEqual(res[a][0], res[c][0])
            self.assertNotAlmostEqual(res[a][1], res[c][1], places=4)

    def test_fsdp_stand_in(self):
        """A micro-batch without a teacher view: no teacher pass alone; with _always (FSDP) a weight-0 one."""
        from cotdistill.student import teacher_forward
        m = tiny4(self.tok).eval()
        b = tsf_batch(self.spec, ids=("gsm8k-train-5699",))
        _, parts, st = teacher_forward(m, self.tok, b, 640, {})
        self.assertEqual((float(parts["teacher"]), float(parts["align"]), st["teacher_n"]), (0.0, 0.0, 0))
        _, parts, st = teacher_forward(m, self.tok, b, 640, {}, 0, True)
        self.assertEqual((float(parts["teacher"]), st["teacher_n"], st["slot_n"]), (0.0, 0, 0))
        self.assertIsNotNone(parts["teacher"].grad_fn)

    def test_ptad_manual(self):
        """PTAD: the student's "Answer:" states at layers 0-2 vs the teacher view's (stop-grad), weight x lambda; the
        teacher CE at 0.5; the letters of label_logits."""
        from cotdistill.student import (Example, TeacherConfig, answer_group, collate, kl_loss, label_logits,
                                        set_slot_prompt, set_teacher, teacher_forward, teacher_plan)
        set_slot_prompt(self.tok, None)
        set_teacher(self.tok, TeacherConfig("ptad", "steps", (0, 1, 2), teacher_ce=0.5))
        m = tiny4(self.tok).eval()
        b = [x for i in ("gsm8k-train-660", "gsm8k-train-3365", "gsm8k-train-5699")
             for x in answer_group(gfpq_group(GSM[i]), GSM[i], plan=teacher_plan(GSM[i], "steps"), lam=2.0)]
        z, parts, st = teacher_forward(m, self.tok, b, 640, {})
        self.assertEqual((st["slot_n"], st["teacher_n"]), (2, 2))
        for x, y in zip(z, label_logits(m, self.tok, b, 640, {})):
            self.assertTrue(torch.allclose(x, y, atol=1e-6))
        tex = [Example(e.teacher_text, e.labels, e.target, e.weight * 0.5, "teacher", e.item_id) for e in b[:2]]
        with torch.no_grad():
            ct, cs = collate(self.tok, tex, 640, "cpu"), collate(self.tok, b[:2], 640, "cpu")
            ht = m.model(input_ids=ct["input_ids"], attention_mask=ct["attention_mask"], output_hidden_states=True)
            hs = m.model(input_ids=cs["input_ids"], attention_mask=cs["attention_mask"], output_hidden_states=True)
            tot = 0.0
            for r in range(2):
                t = torch.stack([ht.hidden_states[li + 1][r, ct["last"][r]] for li in (0, 1, 2)])
                s = torch.stack([hs.hidden_states[li + 1][r, cs["last"][r]] for li in (0, 1, 2)])
                sig = t.pow(2).mean(-1, keepdim=True).sqrt()
                tot += b[r].weight * 2.0 * float(torch.nn.functional.smooth_l1_loss(s / sig, t / sig))
            tce = float(kl_loss(label_logits(m, self.tok, tex, 640, {}), tex) * 2)
        self.assertAlmostEqual(float(parts["align"]), tot, places=5)
        self.assertAlmostEqual(float(parts["teacher"]), tce, places=5)


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestTSFEval(unittest.TestCase):
    def setUp(self):
        from cotdistill.student import TeacherConfig, set_slot_prompt, set_teacher
        self.tok = gpt2()
        self.spec = tsf_spec()
        set_slot_prompt(self.tok, self.spec)
        self.cfg = TeacherConfig("tsf", "steps", (1, 2), force_p=1.0, seed=0)
        set_teacher(self.tok, self.cfg)
        self.addCleanup(set_slot_prompt, self.tok, None)
        self.addCleanup(set_teacher, self.tok, None)

    def test_force_forceswap_align(self):
        from cotdistill.student import evaluate, teacher_forward, teacher_partners, tsf_eval, tsf_subsets
        m = tiny4(self.tok).eval()
        intact, ip = evaluate(m, self.tok, ITEMS, 640, 4, {})
        res = tsf_eval(m, self.tok, ITEMS, max_len=640, bs=4, modes=("force", "forceswap"), align_layers=(1, 2))
        self.assertEqual(res["chained"], sorted(it["item_id"] for it in ITEMS if it.get("worked_steps")))
        self.assertEqual(res["force"][0]["n_forced"], 5)
        pf = {p["item_id"]: p["probs"] for p in res["force"][1]}
        ps = {p["item_id"]: p["probs"] for p in res["forceswap"][1]}
        pi = {p["item_id"]: p["probs"] for p in ip}
        for iid in ("gsm8k-train-660", "gsm8k-train-3365"):                    # = the forced training pass
            b = tsf_batch(self.spec, ids=(iid,))
            z, _, st = teacher_forward(m, self.tok, b, 640, {})
            self.assertEqual(st["forced"], 1)
            for u, v in zip(torch.softmax(z[0], -1).tolist(), pf[iid]):
                self.assertAlmostEqual(u, v, places=5)
            self.assertNotEqual(pf[iid], ps[iid])
            self.assertNotEqual(pf[iid], pi[iid])
        for u, v in zip(pf["gsm8k-train-5699"], pi["gsm8k-train-5699"]):          # no chain: unchanged
            self.assertAlmostEqual(u, v, places=6)
        self.assertEqual(teacher_partners(ITEMS, "eval-swap", {i: 2 for i in res["chained"]}).keys(), set(res["chained"]))
        a = res["align"]
        self.assertEqual((a["n_slots"], a["layers"]), (10, [1, 2]))                 # 5 chains x 2 slots
        self.cfg.force_p = 0.0
        tot, n = 0.0, 0
        for iid in res["chained"]:
            _, _, st = teacher_forward(m, self.tok, tsf_batch(self.spec, ids=(iid,)), 640, {})
            tot, n = tot + st["align_sum"], n + st["slot_n"]
        self.assertAlmostEqual(a["align"], tot / n, places=5)
        only = tsf_eval(m, self.tok, ITEMS, max_len=640, bs=4, modes=(), align_layers=(1, 2))
        self.assertAlmostEqual(only["align"]["align"], a["align"], places=6)
        sub = tsf_subsets(ITEMS, res["force"][1], ip, self.spec)
        self.assertEqual(sub["n_chain"], 5)
        self.assertEqual(sub["n_optmid"] + sub["n_nooptmid"], 5)
        self.assertAlmostEqual(sub["acc_drop_chain"], sub["acc_intact_chain"] - sub["acc_chain"])

    def test_option_has_intermediate(self):
        from cotdistill.student import option_has_intermediate
        it = GSM["gsm8k-train-660"]                                        # options 48 20 53 49 4; steps 6 8 2 10 18
        self.assertFalse(option_has_intermediate(it))
        x = dict(it, prompt=it["prompt"].replace("D) 49", "D) 18"))
        self.assertTrue(option_has_intermediate(x))
        y = dict(it, prompt=it["prompt"].replace("D) 49", "D) 1,0"))       # 10 written with a comma
        self.assertTrue(option_has_intermediate(y))
        self.assertFalse(option_has_intermediate(GSM["gsm8k-train-5699"]))


# ---------------------------------------------------------------------------------------------------- end to end
def tiny4_dir(d: str, n: int = 4) -> None:
    tok = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    tok.pad_token = tok.eos_token
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=n,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                                   tie_word_embeddings=True)
    torch.manual_seed(0)
    transformers.Qwen2ForCausalLM(cfg).save_pretrained(d + "/m")
    tok.save_pretrained(d + "/m")


def common4(d, data="t.jsonl"):
    from test_tree_forcing import SCRIPT
    return [sys.executable, SCRIPT, "--model", d + "/m", "--train", f"{d}/{data}", "--eval", f"v={d}/{data}",
            "--items-per-update", "2", "--micro-bs", "4", "--grad-accum", "1", "--max-len", "640",
            "--eval-max-len", "640", "--precision", "fp32master", "--log-every", "1", "--lr", "1e-3",
            "--seed", "3", "--final", "teacher", "--subq-target", "cot"]


GFPQ = ["--subq", "cot", "--tree-full", "plain", "--tree-view", "statements+question", "--tree-cap", "0",
        "--tree-sources", "gsm8k_tf"]
TSFB = ["--latent-slots", "2", "--slot-c", "2", "--slot-tokens", "distinct", "--slot-token", ",".join(DIST[:2]),
        "--slot-layer", "2", "--slot-readout", "tsf", "--slot-overflow", "truncate", "--slot-align-layers", "1-2",
        "--slot-final-only"]
TSF_ARMS = {"TSF": TSFB + ["--slot-teacher", "steps"],
            "TSFN": TSFB + ["--slot-teacher", "steps", "--slot-force-p", "0"],
            "TSFR": TSFB + ["--slot-teacher", "shuffled"],
            "TSFX": TSFB + ["--slot-teacher", "placebo"],
            "TSF6": TSFB + ["--slot-teacher", "steps", "--latent-slots", "6", "--slot-token", ",".join(DIST),
                            "--slot-overflow", "merge"]}
PTAD_ARMS = {"PTAD": ["--answer-teacher", "steps", "--answer-align-layers", "0-2"],
             "PTADR": ["--answer-teacher", "shuffled", "--answer-align-layers", "0-2"],
             "PTADX": ["--answer-teacher", "placebo", "--answer-align-layers", "0-2"]}
NEC = ["--slot-necessity", "swap,drop,mean,force,forceswap", "--slot-swap-token", SWAP]
DETECT = ["--eval-at", "0.5,1", "--teacher-grad-probe", "0,1"]


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = d = cls.tmp.name
        tiny4_dir(d)
        Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in ITEMS))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def ok(self, r):
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])

    def test_arms_end_to_end(self):
        """The eight arms for two epochs with the detector at 0.5 / 1 epochs, the gradient probe at updates 0 / 1, the
        necessity modes (TSF arms) and the held-out alignment; eval_student on the saved TSF model."""
        d = self.d
        ms = {}
        for arm, extra in {**TSF_ARMS, **PTAD_ARMS}.items():
            more = NEC + (["--save"] if arm == "TSF" else []) if arm.startswith("TSF") else []
            r = run(common4(d) + GFPQ, f"{d}/{arm}", *extra, "--epochs", "2", *DETECT, *more)
            self.ok(r)
            self.assertIn("teacher view (", r.stdout)
            ms[arm] = metrics(f"{d}/{arm}")
        n_upd = 2 * math.ceil(len(ITEMS) / 2)
        for arm, m in ms.items():
            h = m["history"]
            self.assertEqual(len(h), n_upd, arm)
            for x in h:
                self.assertAlmostEqual(x["loss"], x["loss_final"] + x["loss_align"] + x["loss_teacher"], places=5,
                                       msg=arm)
                self.assertEqual(x["seqs"], 2, arm)                                     # final views only
            t = m["teacher"]
            self.assertEqual((t["mode"], t["kind"]), ("tsf" if arm.startswith("TSF") else "ptad",
                                                      {"R": "shuffled", "X": "placebo"}.get(arm[-1], "steps")), arm)
            self.assertEqual([r_["step"] for r_ in t["grad_probe"]["rows"]], [0, 1], arm)
            self.assertEqual(t["grad_probe"]["after_step"], 1, arm)
            self.assertTrue(all(r_["ce"] > 0 and r_["align"] > 0 and r_["teacher"] > 0 for r_ in t["grad_probe"]["rows"]))
            self.assertEqual([r_["epoch"] for r_ in m["trajectory"]], [0.5, 1.0], arm)
            for row in m["trajectory"]:
                self.assertIn("align", row["v"], arm)
                self.assertIn("align_train", row, arm)
            self.assertIn("v", m["teacher_align"])
            self.assertIn("train", m["teacher_align"])
        forced = {arm: sum(x["forced_mb"] for x in ms[arm]["history"]) for arm in TSF_ARMS}
        self.assertEqual(forced["TSFN"], 0)
        for arm in ("TSF", "TSFR", "TSFX", "TSF6"):
            self.assertGreater(forced[arm], 0, arm)
            self.assertLess(forced[arm], n_upd, arm)                               # p = 0.5: not every one
        self.assertEqual(forced["TSF"], forced["TSFR"])                            # the same coins (content-keyed)
        for arm in ("TSFN", "TSFR", "TSFX"):                                        # the same supervised slots
            self.assertEqual([x["align_n"] for x in ms[arm]["history"]], [x["align_n"] for x in ms["TSF"]["history"]])
        self.assertEqual(sum(x["align_n"] for x in ms["TSF"]["history"]), 2 * 10)
        self.assertEqual(sum(x["align_n"] for x in ms["TSF6"]["history"]), 2 * (6 + 3 + 2 + 6 + 4))
        self.assertNotEqual([x["loss_align"] for x in ms["TSF"]["history"]],
                            [x["loss_align"] for x in ms["TSFR"]["history"]])
        for arm in TSF_ARMS:
            m = ms[arm]
            for row in m["trajectory"]:
                for mode in ("mean", "force", "forceswap"):
                    x = row[f"v@{mode}"]
                    self.assertAlmostEqual(x["acc_drop"], row["v"]["acc"] - x["acc"], msg=arm)
                    self.assertEqual(x["n_chain"], 5, arm)
                self.assertEqual(set(row["v"]["detect"]) >= {"force_minus_intact", "intact_minus_mean", "n_chain"}, True)
            nec = m["slot_necessity"]["v"]
            self.assertEqual(set(nec), {"swap", "drop", "mean", "force", "forceswap"}, arm)
            self.assertEqual(nec["force"]["n_forced"], 5, arm)
            self.assertEqual(nec["force"]["n_chain"], 5, arm)
            self.assertIn("g1_own", m["tsf_pilot"], arm)
            ls = m["latent_slots"]
            self.assertEqual((ls["readout"], ls["layer"], ls["overflow"]),
                             ("tsf", 2, "merge" if arm == "TSF6" else "truncate"), arm)
            self.assertEqual(ls["views"]["teacher_views"], 5, arm)
            self.assertTrue(Path(f"{d}/{arm}/preds_v@force.jsonl").exists())
            self.assertNotIn("slot_readout", m)
        for arm in PTAD_ARMS:
            self.assertNotIn("latent_slots", ms[arm])
            self.assertEqual(ms[arm]["teacher"]["views"]["teacher_views"], 5)
            self.assertEqual(sum(x["forced_mb"] for x in ms[arm]["history"]), 0)
        self.assertEqual([x["loss_final"] for x in ms["PTAD"]["history"]][0],
                         [x["loss_final"] for x in ms["PTADX"]["history"]][0])          # same first update's CE
        side = json.loads(Path(f"{d}/TSF/model/latent_slots.json").read_text())
        self.assertEqual(side, {"k": 2, "c": 2, "token": DIST[0], "place": "before-options", "tokens": list(DIST[:2]),
                                "layer": 2, "readout": "tsf", "overflow": "truncate"})
        from test_tree_forcing import ENV
        r = subprocess.run([sys.executable, str(ROOT / "scripts/eval_student.py"), "--model", f"{d}/TSF/model",
                            "--eval", f"v={d}/t.jsonl", "--max-len", "640", "--out", f"{d}/ev",
                            "--slot-necessity", "force,forceswap,drop"], capture_output=True, text=True, timeout=900,
                           env=ENV)
        self.ok(r)
        em = metrics(f"{d}/ev")
        self.assertEqual(em["latent_slots"], side)
        self.assertEqual(em["slot_necessity"]["v"]["force"]["n_forced"], 5)
        self.assertIn("tsf_pilot", em)
        for p, q in zip(preds(f"{d}/TSF/preds_v.jsonl"), preds(f"{d}/ev/preds_v.jsonl")):
            self.assertLess(max(abs(x - y) for x, y in zip(p["probs"], q["probs"])), 2e-2)

    def test_probe_rescale_and_guard(self):
        """A large alignment weight: the median gradient ratio > 3, the alignment loss rescaled once to ratio 1 after the
        last probe update; the CE guard lowers the teacher CE to 0.25 at the first point when the accuracy is low."""
        d = self.d
        r = run(common4(d) + GFPQ, f"{d}/resc", *TSF_ARMS["TSF"], "--slot-align-lambda", "1000", "--epochs", "2",
                "--eval-at", "0.5", "--teacher-grad-probe", "0,1", "--teacher-ce-guard", "0.99")
        self.ok(r)
        t = metrics(f"{d}/resc")["teacher"]
        gp = t["grad_probe"]
        ratios = sorted(x["align"] / x["ce"] for x in gp["rows"])
        self.assertGreater(gp["ratio_median"], 3)
        self.assertAlmostEqual(gp["ratio_median"], sum(ratios) / 2, places=6)
        self.assertTrue(gp["rescaled"])
        self.assertAlmostEqual(t["scale"], 1 / gp["ratio_median"], places=6)
        self.assertEqual((t["guard"]["lowered"], t["guard"]["teacher_ce"], t["teacher_ce_final"], t["guard"]["epoch"]),
                         (True, 0.25, 0.25, 0.5))
        h = metrics(f"{d}/resc")["history"]
        self.assertTrue(all(math.isfinite(x["loss"]) for x in h))
        r0 = run(common4(d) + GFPQ, f"{d}/resc0", *TSF_ARMS["TSF"], "--slot-align-lambda", "1000", "--epochs", "2",
                 "--eval-at", "0.5", "--teacher-grad-probe", "0,1")                # the same run without the guard
        self.ok(r0)
        h0 = metrics(f"{d}/resc0")["history"]
        for x, y in zip(h[:2], h0[:2]):                                   # before the rescale: lambda 1000 as given
            self.assertAlmostEqual(x["loss_align"], y["loss_align"], places=6)

    def test_resume(self):
        """TSF with the gradient probe and the guard: a planned stop after update 2 and a resume give the uninterrupted
        run's history, evaluation and teacher state."""
        d = self.d
        args = [*TSF_ARMS["TSF"], "--epochs", "2", "--eval-at", "0.5", "--teacher-grad-probe", "0,3",
                "--teacher-ce-guard", "0.99"]
        self.ok(run(common4(d) + GFPQ, f"{d}/full", *args, "--ckpt-every-min", "1000"))
        r1 = run(common4(d) + GFPQ, f"{d}/res", *args, "--stop-at-step", "2")
        self.assertEqual(r1.returncode, 3, r1.stderr[-2000:])
        r2 = run(common4(d) + GFPQ, f"{d}/res", *args, "--ckpt-every-min", "1000")
        self.ok(r2)
        self.assertIn("resumed from", r2.stdout)
        drop = re.compile(r"s_per_step|approx_tok_per_s|s_elapsed")
        strip = lambda h: [{k: v for k, v in row.items() if not drop.match(k)} for row in h]
        a, b = metrics(f"{d}/full"), metrics(f"{d}/res")
        self.assertEqual(strip(a["history"]), strip(b["history"]))
        self.assertEqual(strip(a["trajectory"]), strip(b["trajectory"]))
        self.assertEqual(a["eval"], b["eval"])
        self.assertEqual(a["teacher"], b["teacher"])

    def test_argument_checks(self):
        d = self.d
        T = TSF_ARMS["TSF"]
        bad = [GFPQ + TSFB,                                                       # tsf without --slot-teacher
               GFPQ + T[:T.index("--slot-readout")] + ["--slot-readout", "lens", "--slot-teacher", "steps"],
               GFPQ + T + ["--slot-c", "3"],
               GFPQ + T + ["--slot-fill", "last"],
               GFPQ + T + ["--slot-lambda", "2"],
               GFPQ + T + ["--slot-decode", "3"],
               GFPQ + T + ["--slot-target", "shuffled"],
               GFPQ + T + ["--slot-align-layers", "0-1"],                         # last != --slot-layer
               GFPQ + T + ["--slot-align-layers", "1-3"],                         # layer 3: the last
               GFPQ + T + ["--slot-force-p", "1.5"],
               GFPQ + T + ["--teacher-ce-guard", "0.6"],                          # no --eval-at
               GFPQ + T + ["--teacher-grad-probe", "x"],
               GFPQ + T + ["--lambda-brier", "0.1"],
               GFPQ + ["--slot-teacher", "steps"],                                # no --latent-slots
               GFPQ + ["--latent-slots", "2", "--slot-c", "2", "--slot-tokens", "distinct", "--slot-token",
                       ",".join(DIST[:2]), "--slot-layer", "2", "--slot-readout", "lens", "--slot-necessity", "force"],
               GFPQ + ["--answer-align-layers", "0-1"],                           # no --answer-teacher
               GFPQ + PTAD_ARMS["PTAD"] + ["--latent-slots", "2"],
               GFPQ + ["--answer-teacher", "steps", "--answer-align-layers", "0-3"],
               ["--subq", "none", "--answer-teacher", "steps"],                   # not the full-tree loop
               GFPQ + ["--teacher-grad-probe", "0"]]                              # no TSF / PTAD
        for argv in bad:
            r = run(common4(d) + argv, f"{d}/bad", "--epochs", "1")
            self.assertNotEqual(r.returncode, 0, argv)
        Path(d + "/nc.jsonl").write_text("\n".join(json.dumps({k: v for k, v in x.items() if k != "worked_steps"})
                                                   for x in ITEMS))
        for arm in (T, PTAD_ARMS["PTAD"]):
            r = run(common4(d, "nc.jsonl") + GFPQ, f"{d}/bad2", *arm, "--epochs", "1")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("worked steps", r.stderr + r.stdout)


# ---------------------------------------------------------------------------------------------------- FSDP
@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestFSDP(unittest.TestCase):
    """TSF (forced micro-batches, the gradient probe, the detector, force / forceswap necessity) and PTAD under --fsdp
    over gloo on the CPU (a 4-layer byte-level model, slot tokens ~ ^, swap }), 1 and 2 ranks = the single process."""

    def test_tsf_ptad_fsdp_equal_single(self):
        import test_fsdp as tf
        with tempfile.TemporaryDirectory() as d:
            tok = tf.tiny_tok()
            cfg = transformers.Qwen2Config(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                           num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                                           tie_word_embeddings=True)
            torch.manual_seed(0)
            transformers.Qwen2ForCausalLM(cfg).save_pretrained(d + "/m")
            tok.save_pretrained(d + "/m")
            tf.write_data(d)
            Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in ITEMS))
            base = ["--model", d + "/m", "--train", d + "/t.jsonl", "--eval", f"v={d}/t.jsonl", *GFPQ,
                    "--items-per-update", "3", "--micro-bs", "2", "--max-len", "1024", "--eval-max-len", "1024",
                    "--log-every", "1", "--lr", "1e-3", "--seed", "0", "--epochs", "2", "--precision", "fp32master",
                    "--eval-at", "1", "--teacher-grad-probe", "0,1"]
            tsf = ["--latent-slots", "2", "--slot-c", "2", "--slot-tokens", "distinct", "--slot-token", "~,^",
                   "--slot-layer", "2", "--slot-readout", "tsf", "--slot-overflow", "truncate", "--slot-teacher",
                   "steps", "--slot-align-layers", "1-2", "--slot-final-only", "--slot-necessity",
                   "swap,mean,force,forceswap", "--slot-swap-token", "}"]
            for arm, extra in (("TSF", tsf), ("PTAD", PTAD_ARMS["PTAD"])):
                runs = {"single": tf.single(d, *base, *extra, "--out", f"{d}/{arm}-s")}
                for tag, n in (("f1", 1), ("f2", 2)):
                    runs[tag] = tf.fsdp(d, *base, *extra, "--fsdp-optim", "cpu-adamw", "--out", f"{d}/{arm}-{tag}",
                                        nproc=n)
                for k, r in runs.items():
                    self.assertEqual(r.returncode, 0, (arm, k, r.stderr[-3000:]))
                a = tf.metrics(f"{d}/{arm}-s")
                if arm == "TSF":
                    self.assertGreater(sum(x["forced_mb"] for x in a["history"]), 0)
                for tag in ("f1", "f2"):
                    b = tf.metrics(f"{d}/{arm}-{tag}")
                    self.assertEqual(len(a["history"]), len(b["history"]))
                    for x, y in zip(a["history"], b["history"]):
                        self.assertAlmostEqual(x["loss"], y["upd_loss"], delta=1e-5, msg=(arm, tag))
                        for key in ("loss_align", "loss_teacher", "align"):
                            self.assertAlmostEqual(x[key], y[key], delta=1e-5, msg=(arm, tag, key))
                        for key in ("align_n", "teacher_n", "forced_mb", "forced_n"):
                            self.assertEqual(x[key], y[key], (arm, tag, key))
                    for u, v in zip(a["teacher"]["grad_probe"]["rows"], b["teacher"]["grad_probe"]["rows"]):
                        for key in ("ce", "teacher", "align"):
                            self.assertAlmostEqual(u[key], v[key], delta=1e-4 * max(1.0, u[key]), msg=(arm, tag, key))
                    for p, q in zip(tf.preds(f"{d}/{arm}-s"), tf.preds(f"{d}/{arm}-{tag}")):
                        self.assertLess(max(abs(u - v) for u, v in zip(p["probs"], q["probs"])), 1e-4)
                    ta, tb = a["trajectory"][0], b["trajectory"][0]
                    self.assertAlmostEqual(ta["v"]["align"]["align"], tb["v"]["align"]["align"], delta=1e-4)
                    if arm == "TSF":
                        for mode in ("mean", "force", "forceswap"):
                            self.assertAlmostEqual(ta[f"v@{mode}"]["acc"], tb[f"v@{mode}"]["acc"], msg=(tag, mode))
                            self.assertAlmostEqual(a["slot_necessity"]["v"][mode]["nll"],
                                                   b["slot_necessity"]["v"][mode]["nll"], delta=1e-4, msg=(tag, mode))
                        self.assertEqual(a["tsf_pilot"], b["tsf_pilot"])


# ---------------------------------------------------------------------------------------------------- byte identity
@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestUnchanged(unittest.TestCase):
    """A2 (plain loop), GFPQ (full-tree loop, sub-questions), LS (v2 decode), LSL / LSLU / LSLR / LSLY (v2.1 lens, the
    necessity modes; LSL saved), eval_student on the saved LSL and LSL under --fsdp on one rank: LS23_BASE_REF's files
    byte for byte; the arguments only gain the new keys at their defaults."""
    NEW = {"slot_teacher": "none", "slot_align_layers": "12-20", "slot_align_lambda": 1.0, "slot_force_p": 0.5,
           "slot_teacher_ce": 0.5, "slot_detect_sets": "auto", "answer_teacher": "none", "answer_align_layers": "12-27",
           "answer_align_lambda": 1.0, "answer_teacher_ce": 0.5, "teacher_grad_probe": "", "teacher_ce_guard": 0.0,
           "teacher_align_train_n": 200}

    def same(self, ra, rb, tag):
        drop = re.compile(r"s_per_step|approx_tok_per_s|s_elapsed|upd_s|fb_s|clip_s|opt_s|pad_tok_per_s|step_s|"
                          r"d2h_wait_s|rss_gib_ranks|rss_peak_gib_ranks|peak_gib_ranks|reserved_gib_ranks")
        strip = lambda hh: [{k: v for k, v in row.items() if not drop.fullmatch(k)} for row in hh]
        a, b = metrics(ra), metrics(rb)
        self.assertEqual(strip(a["history"]), strip(b["history"]), tag)
        self.assertEqual(strip(a.get("trajectory", [])), strip(b.get("trajectory", [])), tag)
        self.assertEqual({k: v for k, v in a.items() if k not in ("args", "history", "trajectory")},
                         {k: v for k, v in b.items() if k not in ("args", "history", "trajectory")}, tag)
        self.assertEqual({k: v for k, v in a["args"].items() if k != "out"},
                         {k: v for k, v in b["args"].items() if k != "out" and k not in self.NEW}, tag)
        self.assertEqual({k: b["args"][k] for k in self.NEW}, self.NEW, tag)
        fa = sorted(p.name for p in Path(ra).iterdir())
        self.assertEqual(fa, sorted(p.name for p in Path(rb).iterdir()), tag)
        for f in fa:
            if f.endswith(".jsonl") or f == "latent_slots.json":
                self.assertEqual((Path(ra) / f).read_text(), (Path(rb) / f).read_text(), (tag, f))

    def test_existing_arms_match_ref(self):
        from test_latent_slots import ARMS as LS_ARMS, ARMS21, common
        from test_tree_forcing import ENV, tiny
        with tempfile.TemporaryDirectory() as d:
            h = ref_tree(d + "/h", LS23_BASE_REF)
            if h is None:
                self.skipTest("not a git checkout")
            tiny(d)
            Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in ITEMS))
            nec21 = ["--slot-necessity", "swap,drop,mean", "--slot-swap-token", SWAP, "--slot-decode", "6",
                     "--eval-at", "1"]
            arms = {"A2": ["--subq", "none", "--epochs", "2", "--save"],
                    "GFPQ": GFPQ + ["--epochs", "2", "--eval-subq", f"{d}/t.jsonl", "--eval-at", "1"],
                    "LS": GFPQ + LS_ARMS["LS"] + ["--epochs", "2", "--slot-necessity", "swap,drop", "--slot-swap-token",
                                                  SWAP, "--slot-decode", "2"],
                    **{a: GFPQ + x + ["--epochs", "2", *nec21] + (["--save"] if a == "LSL" else [])
                       for a, x in ARMS21.items()}}
            for arm, extra in arms.items():
                outs = []
                for name, script in (("ref", str(h / "scripts/train_student.py")), ("now", None)):
                    o = f"{d}/{name}-{arm}"
                    self.assertEqual(run(common(d), o, *extra, script=script).returncode, 0, (arm, name))
                    outs.append(o)
                self.same(*outs, arm)
            self.assertEqual(Path(f"{d}/ref-LSL/model/latent_slots.json").read_text(),
                             Path(f"{d}/now-LSL/model/latent_slots.json").read_text())
            for name, script in (("ref", str(h / "scripts/eval_student.py")), ("now", str(ROOT / "scripts/eval_student.py"))):
                r = subprocess.run([sys.executable, script, "--model", f"{d}/now-LSL/model", "--eval", f"v={d}/t.jsonl",
                                    "--max-len", "640", "--out", f"{d}/ev-{name}", "--slot-necessity", "swap,mean,drop",
                                    "--slot-swap-token", SWAP, "--slot-mean-train", f"{d}/t.jsonl", "--slot-decode",
                                    "6"], capture_output=True, text=True, timeout=900, env=ENV)
                self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            ea, eb = metrics(f"{d}/ev-ref"), metrics(f"{d}/ev-now")
            self.assertEqual({k: v for k, v in ea.items() if k != "args"}, {k: v for k, v in eb.items() if k != "args"})
            for f in sorted(p.name for p in Path(f"{d}/ev-ref").glob("*.jsonl")):
                self.assertEqual(Path(f"{d}/ev-ref/{f}").read_text(), Path(f"{d}/ev-now/{f}").read_text(), f)

    def test_lsl_fsdp1_matches_ref(self):
        """LSL under --fsdp on one rank (the friend's single-GPU route): the reference commit's files."""
        import test_fsdp as tf
        with tempfile.TemporaryDirectory() as d:
            h = ref_tree(d + "/h", LS23_BASE_REF)
            if h is None:
                self.skipTest("not a git checkout")
            tf.tiny_model(d)
            tf.write_data(d)
            Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in ITEMS))
            base = ["--model", d + "/m", "--train", d + "/t.jsonl", "--eval", f"v={d}/t.jsonl", *GFPQ,
                    "--items-per-update", "3", "--micro-bs", "2", "--max-len", "1024", "--eval-max-len", "1024",
                    "--log-every", "1", "--lr", "1e-3", "--seed", "0", "--epochs", "2", "--precision", "fp32master",
                    "--latent-slots", "4", "--slot-c", "2", "--slot-tokens", "distinct", "--slot-token", "~,^,|,{",
                    "--slot-layer", "0", "--slot-readout", "lens", "--slot-fill", "last", "--slot-final-only",
                    "--slot-necessity", "swap,mean", "--slot-swap-token", "}", "--slot-decode", "6", "--fsdp",
                    "--fsdp-optim", "cpu-adamw"]
            outs = []
            for name, script in (("ref", str(h / "scripts/train_student.py")), ("now", tf.SCRIPT)):
                o = f"{d}/{name}"
                r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node",
                                    "1", script, *base, "--out", o], capture_output=True, text=True, timeout=900,
                                   env=tf.env(d))
                self.assertEqual(r.returncode, 0, (name, r.stderr[-3000:]))
                outs.append(o)
            self.same(*outs, "LSL-fsdp1")


if __name__ == "__main__":
    unittest.main()
