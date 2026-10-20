"""scripts/probe_slots.py (linear probes of the latent slot states, batch-12 diagnostics) and the PROBE / PROBE_ONLY /
SLOTPADW / SLOTOV logic of hpc/train_arm.sbatch:
- value features (first / last digit, digit count, top-50 value) and labels;
- the probe fitter: a class written linearly into the features is decoded at ~100% on held-out rows, a random class
  stays at the majority baseline; every probe of a batch equals the same probe fitted alone;
- the capture: one forward per batch, each hook (embeddings + every decoder layer) called once per forward, in eval
  mode without gradient, the states equal output_hidden_states (Qwen2 and a hybrid Qwen3.5), the last layer raw (its
  final norm gives the model's last_hidden_state), slot last / mean / problem / answer at the right positions, the
  lens = norm + head argmax at every layer; the training mode is restored;
- positions: the problem's last token (before the slot line and the options), the slot tokens, the "Answer:" cue;
- end to end on a tiny saved model (GPT-2 tokenizer, distinct slot tokens): the problem's last token is a digit, the
  first digit of step 1's value: decoded at ~100% at layer 0 (the embeddings), a random last digit at baseline, the
  slot positions at layer 0 (one fixed embedding) exactly at the majority; the spec from the model's sidecar, from
  --slot-spec-from (a run dir whose metrics.json gives the overflow) and from train_student's flags is the same;
  --table with --control;
- the sbatch (DRY_RUN, a scratch HOME): without the new knobs every command of a batch-11 LSL job is byte-identical to
  HEAD's (6b8ac87) sbatch; PROBE=1 adds --save --save-dir <job-local> and the probe (smoke and arm), never SAVE,
  and -pr in the run dir;
  PROBE_ONLY trains nothing and probes MODEL with LSL's slot line; run dirs -sW / -sO / PROBE-..., the one
  run_train8.sh packs.
CPU only, tiny random models (CUDA_VISIBLE_DEVICES="")."""
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
PROBE = str(ROOT / "scripts/probe_slots.py")
DIST = ("§", "©", "®", "°", "±", "¢")      # GPT-2 single tokens (test_latent_slots.DIST)
SBATCH_REF = os.environ.get("SBATCH_REF", "6b8ac87")     # the commit before SLOTPADW / SLOTOV / PROBE

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


def synth_items(n: int, seed: int, prefix: str) -> list[dict]:
    """Problems whose text ends with the digit d (" ... box 7"): step 1's value starts with d (planted), its other 1-2
    digits (so its last digit and digit count) and step 2's value are random; chains of 2 or 3 steps."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        d = rng.randrange(1, 10)
        v1 = str(d) + "".join(str(rng.randrange(10)) for _ in range(rng.randrange(1, 3)))
        vals = [v1, str(rng.randrange(1, 300))] + ([str(rng.randrange(1, 50))] if rng.random() < 0.5 else [])
        steps = [{"text": f"x = {v}", "value": v} for v in vals]
        out.append({"item_id": f"{prefix}-{i}", "prompt": f"Question: Tom counts the marbles in box {d}\nOptions:\n"
                    f"A) {vals[-1]}\nB) {int(vals[-1]) + 1}\n", "labels": ["A", "B"], "gold_label": "A",
                    "worked_steps": steps})
    return out


class TestFeatures(unittest.TestCase):
    def test_digit_features(self):
        import probe_slots as P
        self.assertEqual(P.digit_features("6"), {"first_digit": 6, "last_digit": 6, "n_digits": 0})
        self.assertEqual(P.digit_features("1200"), {"first_digit": 1, "last_digit": 0, "n_digits": 3})
        self.assertEqual(P.digit_features("0.25"), {"first_digit": 0, "last_digit": 5, "n_digits": 0})
        self.assertEqual(P.digit_features("-37.5"), {"first_digit": 3, "last_digit": 5, "n_digits": 1})
        self.assertEqual(P.digit_features("123456"), {"first_digit": 1, "last_digit": 6, "n_digits": 4})   # 5+
        self.assertIsNone(P.digit_features("x"))

    @unittest.skipUnless(HAVE, "needs torch")
    def test_labels_660(self):
        import probe_slots as P
        fx = json.loads((ROOT / "tests/fixtures/gsm8k_si_items_v1.json").read_text())
        it = {x["item_id"]: x for x in fx["gsm8k"]}
        v = P.item_values(it["gsm8k-train-660"], 6)
        self.assertEqual(v, {"step1": "6", "step2": "8", "step3": "2", "step4": "10", "step5": "18", "step6": "20",
                             "last": "20"})
        v2 = P.item_values(it["gsm8k-train-3365"], 6)
        self.assertEqual((v2["step3"], v2["step4"], v2["last"]), ("66", None, "66"))
        tg = P.target_names(6)
        vocab = P.value_vocab([v, v2], tg)
        y = P.build_labels([v, v2], tg, vocab)
        self.assertEqual(tuple(y.shape), (2, 7, 4))
        self.assertEqual(y[0, 0].tolist(), [6, 6, 0, vocab["step1"]["6"]])
        self.assertEqual(y[1, 3].tolist(), [-1, -1, -1, -1])                  # no step 4
        self.assertEqual(y[0, 3].tolist(), [1, 0, 1, vocab["step4"]["10"]])


@unittest.skipUnless(HAVE, "needs torch")
class TestFit(unittest.TestCase):
    def test_planted_vs_random(self):
        """Position 0 holds the class of target 0 / feature first_digit linearly (a code vector per class + noise):
        held-out ~100%; the random labels of target 1 stay near the majority; position 1 (noise only) at chance."""
        import probe_slots as P
        g = torch.Generator().manual_seed(0)
        N, d = 600, 24
        y = torch.full((N, 2, 4), -1, dtype=torch.long)
        y[:, 0, 0] = torch.randint(0, 10, (N,), generator=g)
        y[:, 1, 0] = torch.randint(0, 10, (N,), generator=g)
        y[:, 0, 2] = torch.randint(0, 5, (N,), generator=g)
        y[:, 0, 3] = torch.where(y[:, 0, 0] < 8, y[:, 0, 0], torch.full_like(y[:, 0, 0], P.TOP_VALUES))  # 8, 9: other
        code = torch.randn(10, d, generator=g)
        x = torch.randn(N, 2, d, generator=g)
        x[:, 0] += 3 * code[y[:, 0, 0]]
        fit, sel, ev = list(range(320)), list(range(320, 400)), {"h": list(range(400, 600))}
        r = P.fit_probes(x, y, fit, sel, ev, wds=[1e-3, 1e-2], steps=200, lr=0.05, device="cpu")
        maj, _ = P.majority(y, fit + sel, ev)
        a = r["acc"]["h"]
        self.assertGreaterEqual(float(a[0, 0, 0]), 0.97)                        # planted
        for p, t, f in ((0, 1, 0), (1, 0, 0), (1, 1, 0), (0, 0, 2)):            # random labels / no signal
            self.assertLess(float(a[p, t, f]), float(maj["h"][t, f]) + 0.12, (p, t, f))
        self.assertTrue(torch.isnan(a[:, 1, 3]).all())                          # value50 of target 1: no labels
        self.assertGreaterEqual(float(a[0, 0, 3]), 0.97)                        # value50 with "other" planted too
        self.assertGreaterEqual(float(r["acc_in"]["h"][0, 0]), 0.97)            # the in-vocabulary rows
        n_in = int(((y[400:, 0, 3] >= 0) & (y[400:, 0, 3] < P.TOP_VALUES)).sum())
        mi = P.majority_in(y, fit + sel, ev)["h"][0]
        top = int(torch.bincount(y[:400, 0, 3][y[:400, 0, 3] < 50]).argmax())
        self.assertAlmostEqual(float(mi), float((y[400:, 0, 3] == top).sum()) / n_in, places=6)
        self.assertEqual(tuple(r["wd"].shape), (2, 2, 4))

    def test_batched_equals_alone(self):
        """A probe fitted with others (more positions, targets, features, wds) = the same probe fitted alone."""
        import probe_slots as P
        g = torch.Generator().manual_seed(1)
        N, d = 200, 8
        x = torch.randn(N, 3, d, generator=g)
        y = torch.randint(0, 5, (N, 2, 4), generator=g)
        fit, sel, ev = list(range(120)), list(range(120, 160)), {"h": list(range(160, 200))}
        big = P.fit_probes(x, y, fit, sel, ev, wds=[1e-2], steps=60, lr=0.05, device="cpu")
        one = P.fit_probes(x[:, 1:2], y[:, 1:2], fit, sel, ev, wds=[1e-2], steps=60, lr=0.05, device="cpu")
        self.assertTrue(torch.allclose(big["acc"]["h"][1:2, 1:2], one["acc"]["h"], atol=0))
        self.assertTrue(torch.allclose(big["select_acc"][1:2, 1:2], one["select_acc"], atol=0))


class _Tok:
    pad_token_id = 0
    eos_token_id = 0


def capture_check(model, vocab: int) -> None:
    """probe_slots.capture on random rows (right padding, batches of 2): every probe layer, slot last / mean, problem
    and answer = output_hidden_states at those positions (layers 0..n-1), the last layer raw (its final norm = the
    body's output), the lens = argmax head(norm(h)) at the slot positions, no gradient, the training mode restored; a
    chunk of one layer gives the same states."""
    import probe_slots as P
    from cotdistill.fsdp import decoder_layers
    from cotdistill.student import SlotSpec, text_parts
    spec = SlotSpec(2, 3, "x")
    g = torch.Generator().manual_seed(0)
    encs = []
    for n in range(5):
        L = 14 + 2 * n
        ids = torch.randint(1, vocab, (L,), generator=g).tolist()
        encs.append({"ids": ids, "slots": [5, 6, 7, 8, 9, 10], "problem": 4, "answer": L - 1})
    body, head = text_parts(model)
    nl = len(decoder_layers(body)) + 1
    model.train()
    feats, lens = P.capture(model, _Tok(), spec, encs, list(range(nl)), bs=2, lens=True)
    assert model.training
    assert sorted(feats) == list(range(nl)) and tuple(lens.shape) == (5, nl, 6), (sorted(feats), lens.shape)
    model.eval()
    for n, e in enumerate(encs):
        with torch.no_grad():
            o = body(input_ids=torch.tensor([e["ids"]]), output_hidden_states=True, use_cache=False)
        hs = o.hidden_states
        for li in range(nl):
            f = feats[li][n]
            assert not f.requires_grad
            if li == nl - 1:                                        # the last layer: raw, its norm = the output
                assert torch.allclose(body.norm(f[-1:]), o.last_hidden_state[0, -1:], atol=1e-5), li
                continue
            h = hs[li][0]
            for j in range(2):
                q = e["slots"][3 * j:3 * j + 3]
                assert torch.allclose(f[2 * j], h[q[-1]], atol=1e-5), (li, j)
                assert torch.allclose(f[2 * j + 1], h[q].mean(0), atol=1e-5), (li, j)
            assert torch.allclose(f[4], h[4], atol=1e-5) and torch.allclose(f[5], h[e["answer"]], atol=1e-5), li
            with torch.no_grad():
                z = head(body.norm(h[e["slots"]]))
            assert lens[n, li].tolist() == z.argmax(-1).tolist(), li
    f2, l2 = P.capture(model, _Tok(), spec, encs, [1], bs=2, lens=False)
    assert sorted(f2) == [1] and l2 is None and torch.equal(f2[1], feats[1])


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestCapture(unittest.TestCase):
    def qwen2(self):
        torch.manual_seed(0)
        return transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
            vocab_size=500, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=256, tie_word_embeddings=False))

    def test_qwen2(self):
        capture_check(self.qwen2(), 500)

    def test_qwen35_hybrid(self):
        """The hybrid Qwen3.5 (Gated DeltaNet + full attention) in a subprocess without fla (its triton kernels need a
        GPU), as test_latent_slots.test_capture."""
        code = ("import sys; sys.modules['fla'] = None; sys.path[:0] = [%r, %r, %r]; import test_latent_slots as L; "
                "import test_probe_slots as T; T.capture_check(L.qwen35_tiny(4), 64); print('capture ok')"
                % (str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests")))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, env=ENV)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("capture ok", r.stdout)

    def test_hook_calls(self):
        """LayerCapture: each hook once per forward (embedding pre-hook + every decoder layer), only the kept layers
        stored, the hooks removed afterwards."""
        import probe_slots as P
        from cotdistill.fsdp import decoder_layers
        m = self.qwen2().eval()
        layers = decoder_layers(m.model)
        with torch.inference_mode(), P.LayerCapture(m.model, m.lm_head, layers, {0, 2}) as cap:
            for _ in range(3):
                cap.idx = torch.tensor([[1, 2]])
                m.model(input_ids=torch.tensor([[3, 4, 5, 6]]))
        self.assertEqual(dict(cap.calls), {i: 3 for i in range(4)})
        self.assertEqual(sorted(cap.got), [0, 2])
        self.assertTrue(all(len(l._forward_hooks) == 0 and len(l._forward_pre_hooks) == 0 for l in layers))


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestPositions(unittest.TestCase):
    def test_gpt2_positions(self):
        import probe_slots as P
        from cotdistill.student import SlotSpec, final_text, set_slot_prompt
        tok = gpt2()
        spec = SlotSpec(2, 2, DIST[0], tokens=DIST[:2], layer=0, readout="lens")
        set_slot_prompt(tok, spec)
        self.addCleanup(set_slot_prompt, tok, None)
        it = synth_items(1, 0, "p")[0]
        e = P.encode_item(tok, spec, it, 640)
        ids = e["ids"]
        self.assertEqual(ids, tok.encode(final_text(tok, it["prompt"])))
        self.assertEqual(tok.decode(ids[e["problem"]]), it["prompt"].split("\nOptions")[0][-2:])     # " 7"
        self.assertEqual([tok.decode(ids[i]) for i in e["slots"]], [DIST[0]] * 2 + [DIST[1]] * 2)
        self.assertEqual(tok.decode(ids[e["answer"]]), ":")
        self.assertLess(e["problem"], e["slots"][0])
        self.assertIsNone(P.encode_item(tok, spec, it, 10))                 # would be cut
        set_slot_prompt(tok, None)                                          # no slot line: the anchors only
        e0 = P.encode_item(tok, None, it, 640)
        self.assertEqual((e0["slots"], tok.decode(e0["ids"][e0["problem"]])), ([], tok.decode(ids[e["problem"]])))


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.d = cls.tmp.name
        tok = gpt2()
        torch.manual_seed(0)
        cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=48, intermediate_size=96, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                                       tie_word_embeddings=True)
        for name in ("m", "base"):
            transformers.Qwen2ForCausalLM(cfg).save_pretrained(f"{d}/{name}")
            tok.save_pretrained(f"{d}/{name}")
        side = {"k": 2, "c": 2, "token": DIST[0], "place": "before-options", "tokens": list(DIST[:2]), "layer": 0,
                "readout": "lens", "fill": "last"}
        Path(f"{d}/m/latent_slots.json").write_text(json.dumps(side))
        Path(f"{d}/run").mkdir()                                            # a run dir: sidecar + metrics overflow
        Path(f"{d}/run/latent_slots.json").write_text(json.dumps(side))
        Path(f"{d}/run/metrics.json").write_text(json.dumps({"latent_slots": dict(side, overflow="truncate")}))
        for name, n, seed in (("train", 420, 1), ("val", 120, 2), ("test", 100, 3)):
            Path(f"{d}/{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in synth_items(n, seed, name)))
        cls.common = ["--train", f"{d}/train.jsonl", "--train-n", "400", "--heldout", f"val={d}/val.jsonl",
                      f"test={d}/test.jsonl", "--n-steps", "2", "--fit-steps", "150", "--lr", "0.05", "--bs", "16",
                      "--max-len", "512", "--wd", "1e-3,1e-2"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_probe(self, *argv):
        r = subprocess.run([sys.executable, PROBE, *argv], capture_output=True, text=True, timeout=900, env=ENV)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        return r

    def test_planted_end_to_end(self):
        d = self.d
        r = self.run_probe("--model", f"{d}/m", "--out", f"{d}/p", *self.common, "--layer-chunk", "2", "--png")
        res = json.loads(Path(f"{d}/p/probe_results.json").read_text())
        self.assertEqual(res["n_layers"], 3)
        self.assertEqual(res["positions"], ["slot1_last", "slot1_mean", "slot2_last", "slot2_mean", "problem", "answer"])
        self.assertEqual(res["targets"], ["step1", "step2", "last"])
        self.assertEqual(res["items"], {"train_fit": 320, "train_select": 80, "val": 120, "test": 100})
        self.assertEqual(res["slot_probe_layer"], 1)
        self.assertEqual(res["spec"]["tokens"], list(DIST[:2]))
        self.assertEqual(set(res["sets"]), {"val", "test", "heldout"})
        pr = res["probe"]
        fd = pr["problem"]["step1"]["first_digit"]
        self.assertEqual(len(fd["acc"]["heldout"]), 3)
        self.assertEqual(fd["n"]["heldout"], 220)
        self.assertGreaterEqual(fd["acc"]["heldout"][0], 0.97)              # planted: the embedding of " d"
        for f in ("last_digit", "n_digits"):                                # random features: baseline
            x = pr["problem"]["step1"][f]
            self.assertLess(x["acc"]["heldout"][0], x["majority"]["heldout"] + 0.12, f)
        x = pr["problem"]["step2"]["value50"]
        self.assertLess(x["acc"]["heldout"][0], x["majority"]["heldout"] + 0.12)
        for pos in ("slot1_last", "slot1_mean", "slot2_mean"):              # one fixed embedding at layer 0
            x = pr[pos]["step1"]["first_digit"]
            self.assertAlmostEqual(x["acc"]["heldout"][0], x["majority"]["heldout"], places=6, msg=pos)
        v50 = pr["problem"]["step1"]["value50"]
        self.assertEqual(len(v50["acc_in"]["heldout"]), 3)
        self.assertLessEqual(v50["n_in"]["heldout"], v50["n"]["heldout"])
        self.assertIsNotNone(v50["majority_in"]["heldout"])
        lens = res["lens"]
        self.assertEqual(sorted(lens), ["slot1", "slot2"])
        self.assertEqual(len(lens["slot1"]["step1"]["first"]["heldout"]), 3)
        self.assertEqual(lens["slot1"]["step1"]["n"]["heldout"], 220)
        self.assertTrue(all(0 <= a <= 1 for a in lens["slot2"]["last"]["value"]["val"]))
        tab = Path(f"{d}/p/probe_table.txt").read_text()
        self.assertIn("problem      step1  first_digit", tab)
        self.assertIn("problem      step1  value50_in", tab)
        self.assertIn("lens (final norm", tab)
        self.assertIn(tab, r.stdout)
        self.assertTrue(Path(f"{d}/p/probe_heatmap.png").exists() or "no matplotlib" in r.stdout)
        self.assertGreater(res["timing"]["total_s"], 0)
        # --table with --control: no model; deltas against the control's probes
        t = self.run_probe("--table", f"{d}/p/probe_results.json", "--control", f"{d}/p/probe_results.json")
        self.assertIn("control", t.stdout)
        self.assertIn("+0.000", t.stdout)

    def test_spec_sources(self):
        """The base model (no sidecar) with --slot-spec-from <run dir> (overflow from metrics.json) and with
        train_student's flags; a model without a sidecar and no spec: the anchors only."""
        import probe_slots as P
        d = self.d
        argv = [*self.common, "--fit-steps", "5", "--no-lens", "--layer-chunk", "0", "--train-n", "40",
                "--heldout-n", "10"]
        self.run_probe("--model", f"{d}/base", "--out", f"{d}/q1", *argv, "--slot-spec-from", f"{d}/run")
        self.run_probe("--model", f"{d}/base", "--out", f"{d}/q2", *argv, "--latent-slots", "2", "--slot-c", "2",
                       "--slot-tokens", "distinct", "--slot-token", ",".join(DIST[:2]), "--slot-layer", "0",
                       "--slot-fill", "last", "--slot-overflow", "truncate")
        self.run_probe("--model", f"{d}/base", "--out", f"{d}/q3", *argv)
        a, b, c = (json.loads(Path(f"{d}/{q}/probe_results.json").read_text()) for q in ("q1", "q2", "q3"))
        self.assertEqual(a["spec"], b["spec"])
        self.assertEqual(a["spec"]["overflow"], "truncate")
        self.assertEqual(a["probe"]["slot1_last"]["step1"]["first_digit"]["n"],
                         b["probe"]["slot1_last"]["step1"]["first_digit"]["n"])
        self.assertEqual(c["positions"], ["problem", "answer"])
        self.assertIsNone(c["spec"])
        self.assertNotIn("lens", a)
        tok = gpt2()
        sd, src = P.read_spec_source(f"{d}/run")
        self.assertEqual((sd["overflow"], src), ("truncate", f"{d}/run/latent_slots.json"))
        self.assertEqual(P.read_spec_source(f"{d}/run/metrics.json")[0]["overflow"], "truncate")
        self.assertEqual(P.spec_from_dict(tok, sd).as_dict(), a["spec"])
        r = subprocess.run([sys.executable, PROBE, "--model", f"{d}/base", "--out", f"{d}/q4", *argv, "--no-slots",
                            "--latent-slots", "2"], capture_output=True, text=True, timeout=300, env=ENV)
        self.assertNotEqual(r.returncode, 0)


# ------------------------------------------------------------------------------------------------------ sbatch
def sbatch_home(d: str) -> Path:
    """A scratch HOME with cotd/repo -> this checkout and the data files a dry run looks for (fixture lines)."""
    fx = json.loads((ROOT / "tests/fixtures/gsm8k_si_items_v1.json").read_text())["gsm8k"]
    lines = "".join(json.dumps(x) + "\n" for x in fx)
    c = Path(d) / "cotd"
    (c / "data/student_gsm8k_tf_si").mkdir(parents=True)
    (c / "data/eval").mkdir(parents=True)
    os.symlink(ROOT, c / "repo")
    for f in ("train", "val"):
        (c / f"data/student_gsm8k_tf_si/{f}.jsonl").write_text(lines)
    for f in ("kk_heldout", "jl_heldout", "jevbench_public", "typed_decisions_test", "bbeh", "bbh_sub500", "gsm8k_mc",
              "gsm8k_mc_v2", "gsm1k_mc", "gsm8k_s2", "gsm8k_s3", "gsm8k_s4", "gsm8k_s5", "gsm8k_s6", "gsm8k_s7",
              "gsm8k_s8", "gsm8k_s5p", "gsm8k_mc_v2_steps"):
        (c / f"data/eval/{f}.jsonl").write_text(lines)
    return c


B11 = ROOT / "cloud/condor/jobs11_4b/261_4b_gsmsi_LSL-c2lr3_s12_fsdp.env"


def dry_run(home: Path, sbatch: Path, env_file: Path | None, **knobs) -> tuple[str, list[str]]:
    """The sbatch under DRY_RUN=1 and bash -x, a batch-11 env file sourced (as run_train8.sh does) plus knobs: (its
    stdout, every python command it would run, in order: the xtrace of the "[dry-run] python" echoes, smoke ones
    included, whose output goes to the smoke logs); timestamps, the smoke dir and the shell's pid normalised."""
    pre = f'set -a; . "{env_file}"; set +a; ' if env_file else ""
    sets = "".join(f" {k}={v!r}" for k, v in knobs.items())
    trace = home.parent / "xtrace.txt"                  # its own fd: the smoke stage sends 2> to its logs
    cmd = (f'cd "{home}"; {pre}export{sets} DRY_RUN=1 HOME="{home.parent}"; exec 9>"{trace}"; '
           f'BASH_XTRACEFD=9 bash -x "{sbatch}"')
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLOT", "PROBE", "SAVE", "_CONDOR", "SLURM"))}
    env["TMPDIR"] = str(home.parent / "tmp")
    (home.parent / "tmp").mkdir(exist_ok=True)
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=120, env=env)
    if r.returncode != 0:
        raise AssertionError(r.stdout[-2000:] + r.stderr[-2000:])
    norm = lambda t: re.sub(r"probe-model-\d+-", "probe-model-PID-",
                            re.sub(r"tmp\.[A-Za-z0-9]+", "tmp.X", re.sub(r"\b\d\d:\d\d:\d\d\b", "T", t)))
    cmds = [norm(l.lstrip("+ ")) for l in trace.read_text().splitlines()
            if re.match(r"^\++ echo '\[dry-run\]' python", l)]
    return norm(r.stdout), cmds


@unittest.skipUnless(B11.exists() and shutil.which("git"), "needs the batch-11 env file and git")
class TestSbatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.home = sbatch_home(cls.tmp.name)
        cls.ref = Path(cls.tmp.name) / "train_arm_ref.sbatch"
        try:
            cls.ref.write_bytes(subprocess.run(["git", "-C", str(ROOT), "show", f"{SBATCH_REF}:hpc/train_arm.sbatch"],
                                               capture_output=True, check=True).stdout)
        except subprocess.CalledProcessError:
            cls.ref = None
        cls.now = ROOT / "hpc/train_arm.sbatch"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_unchanged_without_knobs(self):
        """Without SLOTPADW / SLOTOV / PROBE / PROBE_ONLY: the same stdout and the same python commands (smoke
        included) as HEAD's sbatch, for LSL as batch 11, LSL with other knobs, LSLU, LS and SAVE=1."""
        if self.ref is None:
            self.skipTest("no git history")
        for knobs in ({}, {"SLOTK": "1", "SLOTFILL": "none", "SLOTLAM": "4", "EPOCHS": "6"}, {"ARMS": "LSLU"},
                      {"ARMS": "LS", "SLOTC": "2"}, {"SAVE": "1"}):
            a = dry_run(self.home, self.ref, B11, **knobs)
            b = dry_run(self.home, self.now, B11, **knobs)
            self.assertEqual(a, b, knobs)
            self.assertEqual(sum("train_student.py" in c for c in b[1]), 2, knobs)          # smoke + the arm

    def test_knobs_and_probe(self):
        o, cmds = dry_run(self.home, self.now, B11, SLOTK="1", SLOTOV="truncate", SLOTFILL="none", SLOTPADW="0",
                          SLOTLAM="4", PROBE="1")
        tr = [c for c in cmds if "train_student.py" in c]
        pr = [c for c in cmds if "probe_slots.py" in c]
        self.assertEqual((len(tr), len(pr)), (2, 2))                       # smoke + the arm, each
        tmp = str(self.home.parent / "tmp")
        for c in tr:
            for flag in ("--latent-slots 1 ", "--slot-fill none ", "--slot-pad-weight 0 ", "--slot-overflow truncate ",
                         "--slot-lambda 4 ", "--slot-final-only", "--save --save-dir "):
                self.assertIn(flag, c + " ", flag)
            self.assertNotIn("--save --save --save-dir", c)
        run = "runs/LSL-sk1-sl4-sFnone-sW0-sOtruncate-pr-c2lr3ch-Qwen3.5-4B-student_gsm8k_tf_si-s12"
        self.assertIn(f"--out {tmp}/tmp.X/out-LSL ", tr[0])
        self.assertIn(f"--save --save-dir {tmp}/tmp.X/pm-LSL", tr[0])
        self.assertIn(f"--out {run} ", tr[1])
        self.assertIn(f"--save --save-dir {tmp}/probe-model-PID-LSL/model ", tr[1] + " ")
        self.assertIn(f"--model {tmp}/tmp.X/pm-LSL ", pr[0])
        self.assertIn("--train-n 16 --heldout-n 4 --fit-steps 5 --layer-chunk 0", pr[0])
        self.assertIn(f"--model {tmp}/probe-model-PID-LSL/model --out {run}/probe --train "
                      "data/student_gsm8k_tf_si/train.jsonl --train-n 3000 --heldout "
                      "val=data/student_gsm8k_tf_si/val.jsonl gsm8kv2=data/eval/gsm8k_mc_v2_steps.jsonl", pr[1])
        self.assertNotIn("SAVE=1", o)
        # run_train8.sh packs the last "=== <arm> -> runs/..." line's dir; the probe line does not match it
        self.assertEqual(re.findall(r"^=== [A-Za-z0-9]* -> (runs/[^ ]*) ", o, re.M), [run])
        self.assertIn(f"=== probe of LSL: {tmp}/probe-model-PID-LSL/model into {run}/probe", o)
        _, c2 = dry_run(self.home, self.now, B11, PROBE="1", SAVE="1")    # SAVE too: $OUT/model, probed and kept
        t2 = [c for c in c2 if "train_student.py" in c][1]
        self.assertIn(" --save ", t2 + " ")
        self.assertNotIn("--save-dir", t2)
        self.assertIn("--model runs/LSL-pr-c2lr3ch-Qwen3.5-4B-student_gsm8k_tf_si-s12/model --out",
                      [c for c in c2 if "probe_slots.py" in c][1])

    def test_probe_only(self):
        o, cmds = dry_run(self.home, self.now, B11, PROBE_ONLY="1")
        self.assertFalse(any("train_student.py" in c for c in cmds))
        pr = [c for c in cmds if "probe_slots.py" in c]
        self.assertEqual(len(pr), 2)                                       # smoke + the probe
        spec = ("--latent-slots 6 --slot-c 4 --slot-tokens distinct --slot-place before-options --slot-layer 0.67 "
                "--slot-fill last --slot-overflow merge")
        for c in pr:
            self.assertIn("--model models/Qwen3.5-4B ", c)
            self.assertIn(spec, c)
        run = "runs/PROBE-LSL-Qwen3.5-4B-student_gsm8k_tf_si"
        self.assertIn(f"--out {run} ", pr[1])
        self.assertEqual(re.findall(r"^=== [A-Za-z0-9]* -> (runs/[^ ]*) ", o, re.M), [run])
        # LSL's training slot line (arm_config) has the probe's flags
        cmd = 'source <(sed -n "/^arm_config()/,/^}/p" "$0"); arm_config LSL; echo "$DX"'
        dx = subprocess.run(["bash", "-c", cmd, str(self.now)], capture_output=True, text=True, check=True).stdout
        for flag in spec.replace(" --slot-overflow merge", "").split(" --"):
            self.assertIn(flag.strip("-"), dx)
        _, c2 = dry_run(self.home, self.now, B11, PROBE_ONLY="1", PROBE_SPEC="runs/LSL-c2lr3ch-x")
        self.assertIn("--slot-spec-from runs/LSL-c2lr3ch-x", [c for c in c2 if "probe_slots.py" in c][1])
        o3, _ = dry_run(self.home, self.now, B11, PROBE_ONLY="1", SLOTK="1", SLOTOV="truncate", SLOTFILL="none")
        self.assertIn("=== PROBE -> runs/PROBE-LSL-sk1-sFnone-sOtruncate-Qwen3.5-4B-student_gsm8k_tf_si ", o3)
        with self.assertRaises(AssertionError):                             # not a v2.1 arm, no PROBE_SPEC
            dry_run(self.home, self.now, B11, PROBE_ONLY="1", ARMS="A2")
        with self.assertRaises(AssertionError):
            dry_run(self.home, self.now, B11, PROBE_ONLY="1", PROBE="1")


if __name__ == "__main__":
    unittest.main()
