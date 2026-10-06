"""CPU tests for --fsdp (src/cotdistill/fsdp.py, scripts/train_student.py): FSDP2 over 2 gloo ranks (torchrun on the
CPU, tiny models: vocabulary 285 / 300, 2 layers) against the single-process script. The update must equal the
single-process update (gradients summed over the ranks, same sequences per update), for both optimizer setups:
cpu-adamw with fp32 shards against --precision fp32master (exact up to float reordering) and 8bit-sr (through a
torchao stub: plain AdamW) against the single-process --optim adamw8bit-sr on bf16 weights (bf16 reordering only).
Also: the update split, ShardCPUAdamW against torch AdamW, evaluation gathered in the original order, --save,
the sharded checkpoint round trip, and the default (non-FSDP) paths byte-identical to the pinned base commit.
COTD_CUDA_TESTS=1 with 2 visible GPUs adds the same equivalence over NCCL (HPC smoke, hpc/fsdp12b.sbatch)."""
import dataclasses
import json
import os
import random
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import torch
    import transformers
    import test_student as ts
    import test_train_resume as tr
    HAVE = True
except ImportError:
    HAVE = False
HAVE_G4U = HAVE and hasattr(transformers, "Gemma4UnifiedForConditionalGeneration")
CUDA2 = HAVE and os.environ.get("COTD_CUDA_TESTS") == "1" and torch.cuda.is_available() and torch.cuda.device_count() >= 2

SCRIPT = str(ROOT / "scripts/train_student.py")
# The commit before --fsdp and the Gemma readout patch: the default paths must give the same files as there.
BASE_REF = os.environ.get("FSDP_BASE_REF", "acdc564")


def tiny_tok(bos: bool = False):
    """Byte-level BPE tokenizer with 285 entries: <pad> <eos> <bos>, the 256 byte symbols and ' A'..' Z' as single
    tokens (the letter readout's tokens). bos=True: add_special_tokens prepends <bos>, like Gemma's tokenizer."""
    from tokenizers import Tokenizer, decoders, models, processors
    from tokenizers.pre_tokenizers import ByteLevel
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2}
    for c in sorted(ByteLevel.alphabet()):
        vocab[c] = len(vocab)
    merges = []
    for L in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        vocab["Ġ" + L] = len(vocab)
        merges.append(("Ġ", L))
    t = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    t.pre_tokenizer = ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    if bos:
        t.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
    return transformers.PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", eos_token="<eos>",
                                                bos_token="<bos>" if bos else None)


def tiny_model(d: str, kind: str = "qwen2") -> None:
    if kind == "qwen2":
        tok = tiny_tok()
        cfg = transformers.Qwen2Config(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                                       tie_word_embeddings=True)
        torch.manual_seed(0)
        m = transformers.Qwen2ForCausalLM(cfg)
    else:                                   # Gemma 4 unified, text decoder + vision/audio embedders (dropped on load)
        tok = tiny_tok(bos=True)
        tc = dict(vocab_size=300, hidden_size=32, intermediate_size=64, num_attention_heads=2, num_key_value_heads=1,
                  head_dim=16, global_head_dim=32, num_global_key_value_heads=1, num_hidden_layers=2,
                  layer_types=["sliding_attention", "full_attention"], sliding_window=8, attention_k_eq_v=True,
                  final_logit_softcapping=30.0, tie_word_embeddings=True)
        cfg = transformers.Gemma4UnifiedConfig(text_config=tc, vision_config=dict(mm_embed_dim=32, output_proj_dims=32,
                                               mm_posemb_size=8, num_soft_tokens=4),
                                               audio_config=dict(audio_embed_dim=16, hidden_size=16, output_proj_dims=16))
        torch.manual_seed(0)
        m = transformers.Gemma4UnifiedForConditionalGeneration(cfg)
    m.save_pretrained(d + "/m")
    tok.save_pretrained(d + "/m")


def write_data(d: str) -> None:
    rng = random.Random(0)
    Path(d + "/t.jsonl").write_text("\n".join(json.dumps(ts.toy_item(i, rng)) for i in range(40)))
    Path(d + "/tree.jsonl").write_text("\n".join(json.dumps(it) for it in tr.items(12)))
    (Path(d) / "stub/torchao/optim").mkdir(parents=True, exist_ok=True)
    (Path(d) / "stub/torchao/__init__.py").write_text('__version__ = "0.0.0"\n')
    (Path(d) / "stub/torchao/optim/__init__.py").write_text(tr.STUB)
    # a stub dist-info first on the path: importlib.metadata then reports torchao 0.0.0, so transformers'
    # is_torchao_available() is False and it never imports the real torchao.prototype through the stub (HPC: the
    # real torchao 0.18 is installed, and the stub package shadowed it -> "No module named 'torchao.prototype'")
    (Path(d) / "stub/torchao-0.0.0.dist-info").mkdir(parents=True, exist_ok=True)
    (Path(d) / "stub/torchao-0.0.0.dist-info/METADATA").write_text("Metadata-Version: 2.1\nName: torchao\nVersion: 0.0.0\n")


def env(d: str, cuda: bool = False) -> dict:
    e = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": d + "/stub" + os.pathsep + os.environ.get("PYTHONPATH", "")}
    if not cuda:
        e["CUDA_VISIBLE_DEVICES"] = ""
    return e


def single(d, *argv, script=SCRIPT, cuda=False):
    return subprocess.run([sys.executable, script, *argv], capture_output=True, text=True, timeout=900, env=env(d, cuda))


def fsdp(d, *argv, nproc=2, cuda=False):
    return subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(nproc),
                           SCRIPT, *argv, "--fsdp"], capture_output=True, text=True, timeout=900, env=env(d, cuda))


def metrics(out):
    return json.loads((Path(out) / "metrics.json").read_text())


def preds(out, name="v"):
    return [json.loads(l) for l in open(Path(out) / f"preds_{name}.jsonl")]


G3 = ["--final", "teacher", "--subq", "cot", "--subq-target", "cot", "--grouped-aux", "2", "--aux-kind", "cot",
      "--aux-weight", "0.5", "--items-per-update", "8", "--micro-bs", "4", "--epochs", "1", "--max-len", "128",
      "--log-every", "1", "--lr", "1e-3", "--seed", "0"]
A2 = ["--final", "teacher", "--subq", "none", "--micro-bs", "2", "--grad-accum", "5", "--epochs", "1", "--max-len", "128",
      "--log-every", "1", "--lr", "1e-3", "--seed", "0"]
TREE = ["--tree-full", "true", "--items-per-update", "2", "--micro-bs", "3", "--max-len", "128", "--log-every", "1",
        "--lr", "1e-3", "--seed", "0", "--epochs", "1"]


# ---------------------------------------------------------------------------------------------------- no process group
@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestSplitUpdate(unittest.TestCase):
    def test_partition(self):
        from cotdistill.fsdp import split_update
        from cotdistill.student import Example
        tok = tiny_tok()
        rng = random.Random(0)
        for trial in range(60):
            upd = [Example("x" * rng.randint(1, 60) + "\n\nAnswer:", ["A", "B"], [0.5, 0.5], rng.random(), "final",
                           str(i)) for i in range(rng.randint(1, 23))]
            budget = rng.choice([0, 40, 90, 400])
            mbs, world = rng.choice([1, 2, 3, 4]), rng.choice([1, 2, 3])
            parts = split_update(tok, upd, 128, mbs, budget, 16, world)
            self.assertEqual(len({len(p) for p in parts}), 1)                     # same number of forwards per rank
            real = [e for p in parts for mb, pad, _ in p if not pad for e in mb]
            self.assertEqual(sorted(e.item_id for e in real), sorted(e.item_id for e in upd))   # each sequence once
            for p in parts:
                for mb, pad, ntok in p:
                    lens = [len(tok.encode(e.text, add_special_tokens=False)) for e in mb]
                    self.assertEqual(ntok, len(mb) * max(lens))
                    if pad:
                        self.assertEqual([e.weight for e in mb], [0.0])
                    elif not budget:
                        self.assertLessEqual(len(mb), mbs)
            n_real = sum(1 for p in parts for _, pad, _ in p if not pad)
            if any(not pad for p in parts for _, pad, _ in p) and n_real % world == 0:
                self.assertFalse(any(pad for p in parts for _, pad, _ in p))      # pads only when splitting cannot help

    def test_odd_count_is_split_not_padded(self):
        from cotdistill.fsdp import split_update
        from cotdistill.student import Example
        upd = [Example(f"q{i}\n\nAnswer:", ["A", "B"], [1.0, 0.0], 1.0, "final", str(i)) for i in range(6)]
        parts = split_update(tiny_tok(), upd, 128, 2, 0, 16, 2)                   # 3 chunks of 2 -> 4 micro-batches
        self.assertEqual([len(p) for p in parts], [2, 2])
        self.assertFalse(any(pad for p in parts for _, pad, _ in p))
        parts = split_update(tiny_tok(), upd[:3], 128, 1, 0, 16, 2)               # 3 singles: one pad
        self.assertEqual(sum(pad for p in parts for _, pad, _ in p), 1)


@unittest.skipUnless(HAVE, "needs torch")
class TestShardCPUAdamW(unittest.TestCase):
    """World size 1, plain CPU tensors: the CPU optimizer against torch AdamW on the same fp32 values."""

    def run_pair(self, dtype, seg, clip):
        from cotdistill.fsdp import ShardCPUAdamW, sharded_grad_norm
        torch.manual_seed(0)
        shapes = [(7, 5), (33,), (4, 4, 3)]
        p = [torch.randn(s).to(dtype) for s in shapes]
        ref = [x.detach().float().clone().requires_grad_(True) for x in p]
        opt = ShardCPUAdamW(p, lr=1e-2, weight_decay=0.01, seg_numel=seg, impl="for")
        ropt = torch.optim.AdamW(ref, lr=1e-2, weight_decay=0.01, foreach=False)
        for step in range(4):
            gs = [torch.randn(s) * (5.0 if clip else 0.01) for s in shapes]
            for x, g in zip(p, gs):
                x.grad = g.to(dtype)
            for r, g in zip(ref, gs):
                r.grad = g.to(dtype).float().clone()             # (fp32: .to/.float would alias x.grad)
            total = sharded_grad_norm(p, torch.device("cpu"))
            if dtype == torch.float32:                   # the norm of torch.nn.utils.clip_grad_norm_
                ref_total = torch.nn.utils.get_total_norm([r.grad for r in ref])
                torch.testing.assert_close(total, ref_total, rtol=1e-6, atol=0.0)
            coef = torch.clamp(1.0 / (total + 1e-6), max=1.0)
            opt.set_clip(coef if float(coef) < 1 else None)
            if float(coef) < 1:                          # the same coefficient on the reference's fp32 gradients
                for r in ref:
                    r.grad.mul_(coef)
            opt.step()
            ropt.step()
            opt.zero_grad()
            self.assertIsNone(p[0].grad)
            for x, m, r in zip(p, opt.master, ref):
                torch.testing.assert_close(m.view(r.shape), r.detach(), rtol=1e-5, atol=1e-6)
                self.assertTrue(torch.equal(x, m.view(r.shape).to(dtype)))        # weights = masters rounded
        return opt

    def test_fp32_bf16_segments_clip(self):
        for dtype in (torch.float32, torch.bfloat16):
            for seg in (2 ** 20, 16):                    # one bucket / parameters cut into many segments
                for clip in (False, True):
                    self.run_pair(dtype, seg, clip)

    def test_reset_and_release(self):
        opt = self.run_pair(torch.float32, 16, False)
        self.assertGreater(opt.host_bytes(), 0)
        opt.reset_state()
        self.assertTrue(all(not inner.state for _, _, inner in opt._buckets))
        opt.release()
        self.assertEqual(opt.master, [])


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestValidation(unittest.TestCase):
    def test_flag_errors(self):
        with tempfile.TemporaryDirectory() as d:
            base = ["--model", d + "/none", "--train", d + "/none.jsonl", "--out", d + "/o"]
            r = single(d, *base, "--fsdp-optim", "cpu-adamw")
            self.assertIn("--fsdp-optim needs --fsdp", r.stderr)
            r = single(d, *base, "--fsdp", "--fsdp-optim", "8bit-sr", "--precision", "fp32master")
            self.assertIn("use --precision bf16", r.stderr)
            r = single(d, *base, "--fsdp", "--precision", "bf16", "--tree-full", "true", "--stop-at-step", "1")
            self.assertIn("does not checkpoint", r.stderr)
            r = single(d, *base, "--fsdp", "--precision", "bf16", "--optim", "adamw8bit-sr")
            self.assertIn("is --fsdp-optim", r.stderr)
            r = single(d, *base, "--fsdp", "--precision", "bf16")
            self.assertIn("launch with torchrun", r.stderr)


# ---------------------------------------------------------------------------------------------------- 2 gloo ranks
@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestFSDPEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = d = cls.tmp.name
        tiny_model(d)
        write_data(d)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def ok(self, r):
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])

    def test_grouped_cpu_adamw_fp32_equals_single(self):
        """G3 (grouped loop, gradient checkpointing on): FSDP fp32 shards + cpu-adamw vs single fp32master + AdamW,
        update for update; evaluation (odd number of batches: one pad batch), sub-question eval and hidden states
        gathered in the original order; --save gives a model that reloads to the same predictions."""
        d = self.d
        ev = ["--eval", f"v={d}/t.jsonl", "--eval-subq", f"{d}/t.jsonl", "--dump-hidden", "v"]
        self.ok(single(d, "--model", d + "/m", "--train", d + "/t.jsonl", *G3, *ev, "--precision", "fp32master",
                       "--log-update-loss", "--out", d + "/g3s"))
        self.ok(fsdp(d, "--model", d + "/m", "--train", d + "/t.jsonl", *G3, *ev, "--precision", "fp32master",
                     "--fsdp-optim", "cpu-adamw", "--save", "--out", d + "/g3f"))
        a, b = metrics(d + "/g3s"), metrics(d + "/g3f")
        self.assertEqual(len(a["history"]), 5)
        self.assertEqual([r["step"] for r in a["history"]], [r["step"] for r in b["history"]])
        for x, y in zip(a["history"], b["history"]):
            self.assertAlmostEqual(x["upd_loss"], y["upd_loss"], delta=1e-6)
            self.assertAlmostEqual(y["loss"], y["upd_loss"], delta=1e-12)
            self.assertEqual(len(y["peak_gib_ranks"]), 2)
        self.assertEqual(b["fsdp"], {"world": 2, "optim": "cpu-adamw"})
        pa, pb = preds(d + "/g3s"), preds(d + "/g3f")
        self.assertEqual([p["item_id"] for p in pa], [p["item_id"] for p in pb])
        for p, q in zip(pa, pb):
            for x, y in zip(p["probs"], q["probs"]):
                self.assertAlmostEqual(x, y, delta=1e-5)
        sa, sb = preds(d + "/g3s", "subq"), preds(d + "/g3f", "subq")
        self.assertEqual([(p["item_id"], p["kind"]) for p in sa], [(p["item_id"], p["kind"]) for p in sb])
        self.assertLess(max(abs(p["p_yes"] - q["p_yes"]) for p, q in zip(sa, sb)), 1e-5)
        ha, hb = torch.load(d + "/g3s/hidden_v.pt"), torch.load(d + "/g3f/hidden_v.pt")
        self.assertEqual(ha.shape, hb.shape)
        self.assertLess(float((ha.float() - hb.float()).abs().max()), 1e-2)        # float16 storage
        # the saved full state dict reloads to the FSDP model's predictions
        from cotdistill.student import evaluate, load_model
        m = load_model(d + "/g3f/model", torch.float32).eval()
        tok = transformers.AutoTokenizer.from_pretrained(d + "/g3f/model")
        items = [json.loads(l) for l in open(d + "/t.jsonl")]
        _, pr = evaluate(m, tok, items, 6144, 8, {})
        for p, q in zip(pr, pb):
            for x, y in zip(p["probs"], q["probs"]):
                self.assertAlmostEqual(x, y, delta=1e-5)

    def test_plain_token_budget_equals_single(self):
        """A2 (plain loop): FSDP with token-budget micro-batches (not available on one process) gives the update of
        the single-process micro_bs x grad_accum loop."""
        d = self.d
        self.ok(single(d, "--model", d + "/m", "--train", d + "/t.jsonl", *A2, "--eval", f"v={d}/t.jsonl",
                       "--precision", "fp32master", "--log-update-loss", "--out", d + "/a2s"))
        self.ok(fsdp(d, "--model", d + "/m", "--train", d + "/t.jsonl", *A2, "--eval", f"v={d}/t.jsonl",
                     "--precision", "fp32master", "--token-budget", "200", "--out", d + "/a2f"))
        a, b = metrics(d + "/a2s"), metrics(d + "/a2f")
        self.assertEqual(len(a["history"]), 4)
        for x, y in zip(a["history"], b["history"]):
            self.assertAlmostEqual(x["upd_loss"], y["upd_loss"], delta=1e-6)
        self.assertNotEqual({y["fwd"] for y in b["history"]}, {5})                 # the budget changed the split
        self.assertAlmostEqual(a["eval"]["v"]["nll"], b["eval"]["v"]["nll"], delta=1e-5)

    def test_fulltree_8bit_sr_and_resume(self):
        """Full-tree loop, bf16 weights, 8bit-sr through the torchao stub: FSDP vs single --optim adamw8bit-sr (bf16
        reordering only); the sharded checkpoint (planned stop after update 3, resume) continues bit for bit."""
        d = self.d
        base = ["--model", d + "/m", "--train", d + "/tree.jsonl", "--eval", f"v={d}/tree.jsonl", *TREE,
                "--precision", "bf16"]
        self.ok(single(d, *base, "--optim", "adamw8bit-sr", "--out", d + "/ts"))
        self.ok(fsdp(d, *base, "--fsdp-optim", "8bit-sr", "--out", d + "/tf"))
        r1 = fsdp(d, *base, "--fsdp-optim", "8bit-sr", "--stop-at-step", "3", "--out", d + "/tr")
        self.assertNotEqual(r1.returncode, 0)                                    # torchrun reports exit 3 as a failure
        self.assertTrue((Path(d) / "tr/planned_stop").exists())
        self.assertEqual(sorted(p.name for p in (Path(d) / "tr/ckpt").iterdir()),
                         ["model", "optim.0.pt", "optim.1.pt", "state.pt"])
        r2 = fsdp(d, *base, "--fsdp-optim", "8bit-sr", "--ckpt-every-min", "1000", "--out", d + "/tr")
        self.ok(r2)
        self.assertIn("resumed from", r2.stdout)
        self.assertFalse((Path(d) / "tr/planned_stop").exists())
        self.assertFalse((Path(d) / "tr/ckpt").exists())
        a, b, c = metrics(d + "/ts"), metrics(d + "/tf"), metrics(d + "/tr")
        self.assertEqual(len(a["history"]), 6)
        for x, y in zip(a["history"], b["history"]):
            self.assertAlmostEqual(x["loss"], y["loss"], delta=2e-3)
        self.assertEqual([x["loss"] for x in b["history"]], [x["loss"] for x in c["history"]])
        self.assertEqual(preds(d + "/tf"), preds(d + "/tr"))


@unittest.skipUnless(HAVE_G4U, "needs transformers with Gemma4Unified")
class TestFSDPGemma(unittest.TestCase):
    def test_gemma_text_only_softcap_bos(self):
        """Gemma 4 unified (text-only load, BOS prefix, soft-capped readout) under FSDP: same updates as one process."""
        with tempfile.TemporaryDirectory() as d:
            tiny_model(d, "gemma")
            write_data(d)
            argv = ["--model", d + "/m", "--train", d + "/t.jsonl", "--eval", f"v={d}/t.jsonl", *G3,
                    "--precision", "fp32master"]
            r = single(d, *argv, "--log-update-loss", "--out", d + "/s")
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            self.assertIn("readout: special prefix [2], logit soft-cap 30.0", r.stdout)
            r = fsdp(d, *argv, "--out", d + "/f")
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            self.assertIn("gemma4_unified: text-only load", r.stdout)
            for x, y in zip(metrics(d + "/s")["history"], metrics(d + "/f")["history"]):
                self.assertAlmostEqual(x["upd_loss"], y["upd_loss"], delta=1e-6)


class TestCostTwoSlices(unittest.TestCase):
    def test_model(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import cost12b as c
        self.assertAlmostEqual(c.comm_s(14.0), 3 * 11.90735032 / 14.0, places=3)        # 2.55 s per micro-batch
        self.assertAlmostEqual(c.opt_s("cpu-adamw", 30.0), 11.90735032 / 2 * 48 / 30, places=3)
        arm = dict(arm="x", epochs=4, pad=226 * 8 * 3000.0, fwd=226 * 8.0, fwd_rank=226 * 4.0)
        h_gpu = c.hours2(arm, 790.0, 0.0, 2.55, 9.5, 0.0)                               # 3000 tok / 790 = 3.8 s > comm
        self.assertEqual(h_gpu["bound"], "GPU")
        self.assertAlmostEqual(h_gpu["s_per_update"], 4 * 3000 / 790 + 9.5 + 0.5, places=6)
        self.assertEqual(c.hours2(arm, 5000.0, 0.0, 2.55, 9.5, 0.0)["bound"], "comm")
        L = [200] * 64
        pad, n_mb, n_rank = c.a2_budget(L, 3200, per_update=32)                       # 16 x 200 per micro-batch
        self.assertAlmostEqual(n_mb / 226, 2.0)
        self.assertAlmostEqual(n_rank / 226, 1.0)


# ---------------------------------------------------------------------------------------------------- default paths
def base_script(dest: str) -> Path | None:
    """scripts/train_student.py and src/cotdistill of BASE_REF under dest (None outside a git checkout)."""
    try:
        blob = subprocess.run(["git", "-C", str(ROOT), "archive", BASE_REF, "src/cotdistill", "scripts/train_student.py"],
                              capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    Path(dest).mkdir(parents=True, exist_ok=True)
    (Path(dest) / "b.tar").write_bytes(blob)
    with tarfile.open(Path(dest) / "b.tar") as t:
        t.extractall(dest, filter="data")
    return Path(dest) / "scripts/train_student.py"


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestDefaultPathsUnchanged(unittest.TestCase):
    def test_plain_grouped_fulltree_match_base(self):
        """Without --fsdp the plain (A2), grouped (G3) and full-tree loops write the same history (minus timings),
        evaluations and prediction files as the base commit's script."""
        with tempfile.TemporaryDirectory() as d:
            old = base_script(d + "/base")
            if old is None:
                self.skipTest("not a git checkout")
            tiny_model(d)
            write_data(d)
            runs = {"a2": ["--train", d + "/t.jsonl", "--eval", f"v={d}/t.jsonl", *A2],
                    "g3": ["--train", d + "/t.jsonl", "--eval", f"v={d}/t.jsonl", "--eval-subq", f"{d}/t.jsonl", *G3],
                    "tf": ["--train", d + "/tree.jsonl", "--eval", f"v={d}/tree.jsonl", *TREE, "--token-budget", "300"]}
            drop = re.compile(r"s_per_step|approx_tok_per_s")
            for name, argv in runs.items():
                outs = []
                for tag, script in (("old", str(old)), ("new", SCRIPT)):
                    r = single(d, "--model", d + "/m", *argv, "--out", f"{d}/{name}-{tag}", script=script)
                    self.assertEqual(r.returncode, 0, r.stderr[-3000:])
                    outs.append(Path(f"{d}/{name}-{tag}"))
                a, b = (metrics(o) for o in outs)
                strip = lambda h: [{k: v for k, v in row.items() if not drop.match(k)} for row in h]
                self.assertEqual(strip(a["history"]), strip(b["history"]), name)
                self.assertEqual(a["eval"], b["eval"], name)
                for f in sorted(p.name for p in outs[0].glob("preds_*.jsonl")):
                    self.assertEqual((outs[0] / f).read_text(), (outs[1] / f).read_text(), (name, f))


# ---------------------------------------------------------------------------------------------------- 2 GPUs (HPC)
@unittest.skipUnless(CUDA2, "COTD_CUDA_TESTS=1 and 2 GPUs")
class TestFSDPCuda(unittest.TestCase):
    def test_nccl_grouped_equals_single(self):
        """NCCL on the job's two GPUs: G3, fp32 shards + cpu-adamw vs single fp32master (both autocast bf16)."""
        with tempfile.TemporaryDirectory(dir=os.environ.get("COTD_TEST_TMP")) as d:
            tiny_model(d)
            write_data(d)
            argv = ["--model", d + "/m", "--train", d + "/t.jsonl", "--eval", f"v={d}/t.jsonl", *G3,
                    "--precision", "fp32master"]
            r = single(d, *argv, "--log-update-loss", "--out", d + "/s", cuda=True)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            r = fsdp(d, *argv, "--out", d + "/f", cuda=True)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            for x, y in zip(metrics(d + "/s")["history"], metrics(d + "/f")["history"]):
                self.assertAlmostEqual(x["upd_loss"], y["upd_loss"], delta=2e-3)
            r = fsdp(d, *argv[:-2], "--precision", "bf16", "--fsdp-optim", "cpu-adamw", "--token-budget", "300",
                     "--out", d + "/fb", cuda=True)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            self.assertEqual(len(metrics(d + "/fb")["history"]), 5)


if __name__ == "__main__":
    unittest.main()
