"""scripts/grad_subspace.py (one-step gradient subspaces vs the finished update, LoRA-One link): the examples and the
loss normalisation of scripts/train_student.py (plain and full-tree loops replicated update by update), layer and
matrix selection on Qwen3.5 / Qwen2, the subspace metrics, gradient checkpointing / layer groups / micro-batching
invariance, and the CLI end to end (finetuned models, --check, --report). Tiny random models, CPU only."""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

try:
    import torch
    import transformers
    HAVE = True
except ImportError:
    HAVE = False

SCRIPT = str(ROOT / "scripts/grad_subspace.py")
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
KK = json.loads((ROOT / "tests/fixtures/kk_local_item.json").read_text())      # one data/student_v3sl K&K item
HAVE_Q35 = HAVE and hasattr(transformers, "Qwen3_5ForCausalLM")


def kk_item(j: int) -> dict:
    """The fixture K&K item with its first inhabitant renamed (distinct sequences: identical copies would make the
    last layer's final-question gradient rank 1)."""
    name = ["Paul", "Xavier", "Yolanda", "Zachary", "Quentin", "Wanda"][j % 6]
    return dict(json.loads(json.dumps(KK).replace("Paul", name)), item_id=f"k{j}")


def toy_items(n_kk: int = 4, n_toy: int = 4) -> list[dict]:
    rng = random.Random(0)
    out = [kk_item(j) for j in range(n_kk)]
    for i in range(n_toy):
        c = rng.choice(["red", "blue"])
        g = "A" if c == "red" else "B"
        out.append({"item_id": f"t{i}", "source": "toy", "prompt": f"Case {i}. The light is {c}.\nOptions:\nA) stop\nB) go",
                    "labels": ["A", "B"], "gold_label": g, "teacher": {g: 0.9, ("B" if g == "A" else "A"): 0.1},
                    "weight": 1.5 if i == 0 else 1.0})
    return out


def gpt2_tok():
    tok = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    tok.pad_token = tok.eos_token
    return tok


def tiny_qwen2(tok, layers: int = 2):
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                                   tie_word_embeddings=True)
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(cfg)


def q35(tok):
    from test_kk_audit import tiny_q35
    return tiny_q35(tok)


def run_args(**kw) -> argparse.Namespace:
    a = dict(max_len=512, micro_bs=4, token_budget=0, token_budget_seqs=16, label_smoothing=0.1, layer_groups=1)
    a.update(kw)
    return argparse.Namespace(**a)


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    model.train()


@unittest.skipUnless(HAVE, "needs torch")
class TestMetrics(unittest.TestCase):
    def test_principal_angle_similarity(self):
        import grad_subspace as gs
        torch.manual_seed(0)
        Q, _ = torch.linalg.qr(torch.randn(200, 200))
        self.assertAlmostEqual(gs.subspace_sim(Q[:, :8], Q[:, :8], 8), 1.0, places=5)
        self.assertAlmostEqual(gs.subspace_sim(Q[:, :8], Q[:, 8:16], 8), 0.0, places=5)
        # half of the directions shared: mean cos^2 = 1/2
        self.assertAlmostEqual(gs.subspace_sim(Q[:, :8], torch.cat([Q[:, :4], Q[:, 20:24]], 1), 8), 0.5, places=5)
        sims = []
        for _ in range(30):         # unrelated random subspaces: r / d on average
            A, _ = torch.linalg.qr(torch.randn(200, 8))
            B, _ = torch.linalg.qr(torch.randn(200, 8))
            sims.append(gs.subspace_sim(A, B, 8))
        self.assertAlmostEqual(sum(sims) / len(sims), 8 / 200, delta=0.01)
        self.assertAlmostEqual(gs.subspace_sim(Q[:, :4], Q[:, :4], 16), 1.0, places=5)   # r clipped to the columns

    def test_matrix_report_against_constructed_updates(self):
        import grad_subspace as gs
        torch.manual_seed(1)
        U, _ = torch.linalg.qr(torch.randn(60, 60))
        V, _ = torch.linalg.qr(torch.randn(40, 40))
        s = torch.tensor([10.0, 6, 3, 1] + [0.0] * 36)
        G = U[:, :40] @ torch.diag(s) @ V.T                       # rank 4: U[:, :4], V[:, :4]
        N = U[:, :40] @ torch.diag(torch.tensor([0.0] * 4 + [1.0] * 36)) @ V.T   # orthogonal to it
        grads = {"A2": G, "GFPS": G / 3 + N, "GFPS_nodes": N}
        dws = {"aligned": -0.01 * G, "outside": -0.01 * N}
        rep = gs.matrix_report(grads, dws, ["GFPS"], [2, 4], w_fro=1.0)
        a = rep["dw"]["aligned"]["vs"]["A2"]
        self.assertAlmostEqual(a["cos"], 1.0, places=5)
        for r in ("2", "4"):
            self.assertAlmostEqual(a["r"][r]["left"], 1.0, places=4)
            self.assertAlmostEqual(a["r"][r]["right"], 1.0, places=4)
            self.assertAlmostEqual(a["r"][r]["energy"], rep["dw"]["aligned"]["top_energy"][r], places=5)
            self.assertAlmostEqual(a["r"][r]["energy"], rep["grad"]["A2"]["top_energy"][r], places=5)
        o = rep["dw"]["outside"]["vs"]["A2"]
        self.assertAlmostEqual(o["cos"], 0.0, places=5)
        self.assertLess(o["r"]["4"]["energy"], 1e-6)
        self.assertLess(o["r"]["4"]["left"], 1e-4)
        self.assertAlmostEqual(rep["dw"]["outside"]["vs"]["GFPS_nodes"]["cos"], 1.0, places=5)
        self.assertAlmostEqual(rep["nodes"]["GFPS"]["ratio"], float(N.norm() / (G.norm() / 3)), places=4)
        self.assertAlmostEqual(rep["nodes"]["GFPS"]["cos_final"], 0.0, places=5)
        self.assertEqual(rep["grad"]["A2"]["rank"], 4)
        self.assertEqual(rep["grad"]["GFPS_nodes"]["rank"], 36)
        self.assertAlmostEqual(rep["grad"]["A2"]["top_energy"]["2"], (100 + 36) / float((s ** 2).sum()), places=5)
        self.assertEqual(set(rep["pairs"]), {"A2~GFPS", "A2~GFPS_nodes", "GFPS~GFPS_nodes"})
        self.assertAlmostEqual(rep["pairs"]["A2~GFPS_nodes"]["r"]["4"]["left"], 0.0, places=4)
        self.assertAlmostEqual(rep["dw_pairs"]["aligned~outside"]["cos"], 0.0, places=5)

    def test_baseline_and_summary(self):
        import grad_subspace as gs
        b = gs.random_baseline([96, 64], [4, 64, 128])
        self.assertAlmostEqual(b["4"]["left"], 4 / 96, places=6)
        self.assertAlmostEqual(b["4"]["right"], 4 / 64, places=6)
        self.assertAlmostEqual(b["4"]["energy"], 16 / (96 * 64), places=8)
        self.assertEqual(b["128"]["right"], 1.0)                   # r clipped to min(d_out, d_in)
        mats = [{"name": f"m{i}", "layer": L, "type": t, "shape": [2, 2], "grad": {"A2": {"fro": float(i)}}}
                for i, (L, t) in enumerate([(3, "q_proj"), (3, "up_proj"), (7, "q_proj"), (7, "down_proj")])]
        s = gs.summarize(mats)
        self.assertEqual(s["all"]["n"], 4)
        self.assertAlmostEqual(s["all"]["grad/A2/fro"], 1.5)
        self.assertAlmostEqual(s["q_proj"]["grad/A2/fro"], 1.0)
        self.assertAlmostEqual(s["mlp"]["grad/A2/fro"], 2.0)
        self.assertAlmostEqual(s["L7"]["grad/A2/fro"], 2.5)
        self.assertNotIn("layer", s["all"])


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestViews(unittest.TestCase):
    def test_batch_and_views_are_the_trainers(self):
        import grad_subspace as gs
        from cotdistill.student import build_examples, build_full_tree
        items = toy_items()
        src = {"knights_knaves"}
        b_all = gs.pick_batch(items, 6, 0, src, "all")
        self.assertEqual(len(b_all), 6)
        self.assertEqual([it["item_id"] for it in b_all], [it["item_id"] for it in gs.pick_batch(items, 6, 0, src, "all")])
        self.assertNotEqual([it["item_id"] for it in b_all],
                            [it["item_id"] for it in gs.pick_batch(items, 6, 1, src, "all")])
        b_tree = gs.pick_batch(items, 10, 0, src, "tree")          # more than eligible: all K&K items
        self.assertEqual(sorted(it["item_id"] for it in b_tree), ["k0", "k1", "k2", "k3"])
        finals, nodes, stats = gs.build_views(b_all, ["GFPS", "TFL"], seed=5, tree_sources=src)
        for it, fin in zip(b_all, finals):          # A2's final view (train_student --final teacher --subq none)
            ref = [e for e in build_examples(it, final="teacher", subq="none", subq_target="cot", lambda_sub=1.0,
                                             rng=random.Random(9)) if e.kind == "final"]
            self.assertEqual(ref, [fin])
        for arm, (mode, view) in (("GFPS", ("plain", "statements")), ("TFL", ("true", "local"))):
            rng = random.Random(5)
            for it, per in zip(b_all, nodes[arm]):
                g = build_full_tree(it, mode=mode, rng=rng, cap=0, aux_total=1.0, final="teacher", subq_target="cot",
                                    view=view, sources=src)
                self.assertEqual(g[1:], per)
                if it["source"] == "knights_knaves":
                    self.assertEqual(len(per), len(KK["subqs"]))
                    self.assertAlmostEqual(sum(e.weight for e in per), 1.0, places=9)    # aux_total per problem
                    if arm == "GFPS":                    # statement block: no final question, no options
                        self.assertTrue(all("Question:" not in e.text.split("Intermediate question:")[0] and
                                            "Options:" not in e.text.split("Intermediate question:")[0] for e in per))
                else:
                    self.assertEqual(per, [])
        # lockstep arms ask the same nodes with the same Yes/No order
        _, n2, _ = gs.build_views(b_all, ["GFPS", "GFP"], seed=5, tree_sources=src)
        for a, b in zip(n2["GFPS"], n2["GFP"]):
            self.assertEqual([(e.target, e.weight) for e in a], [(e.target, e.weight) for e in b])

    def test_preflight_and_batch_errors(self):
        import grad_subspace as gs
        items = toy_items()
        with self.assertRaises(SystemExit):
            gs.preflight(items, ["GFPS"], {"toy"})                # no statement block for this source
        gs.preflight(items, ["TFL"], {"toy"})                     # TFL has no statements view to check
        with self.assertRaises(SystemExit):
            gs.pick_batch(items, 4, 0, {"nope"}, "tree")


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestGradients(unittest.TestCase):
    def test_layer_and_matrix_selection(self):
        import grad_subspace as gs
        tok = gpt2_tok()
        m2 = tiny_qwen2(tok, layers=6)
        t2 = gs.layer_types(m2)
        self.assertEqual(gs.parse_layers("every:4", t2), [0, 4])
        self.assertEqual(gs.parse_layers("every:4:1", t2), [1, 5])
        self.assertEqual(gs.parse_layers("all", t2), list(range(6)))
        self.assertEqual(gs.parse_layers("5,2", t2), [2, 5])
        with self.assertRaises(SystemExit):
            gs.parse_layers("6", t2)
        mats = gs.select_matrices(m2, [0, 4], list(gs.ATTN + gs.MLP))
        self.assertEqual([(m["layer"], m["type"]) for m in mats][:3], [(0, "q_proj"), (0, "k_proj"), (0, "v_proj")])
        self.assertEqual(len(mats), 14)
        self.assertTrue(all(dict(m2.named_parameters())[m["name"]] is m["param"] for m in mats))
        if HAVE_Q35:
            m3 = q35(tok)
            t3 = gs.layer_types(m3)
            self.assertEqual(gs.parse_layers("every:4", t3), [3, 7])    # the full-attention layers
            mats = gs.select_matrices(m3, [3, 7], list(gs.ATTN + gs.MLP))
            self.assertEqual(len(mats), 14)
            self.assertEqual({m["type"] for m in mats if m["layer"] == 3}, set(gs.ATTN + gs.MLP))
            with self.assertRaises(SystemExit):                     # Gated DeltaNet layer: no q/k/v/o
                gs.select_matrices(m3, [0], list(gs.ATTN))
            lin = gs.select_matrices(m3, [0], list(gs.LINATTN))
            self.assertEqual({m["type"] for m in lin}, set(gs.LINATTN))

    def test_split_gradients_equal_the_trainer_loops(self):
        """G_A2 = the plain loop's update gradient (micro_bs x grad_accum normalisation); G_A2 / 3 + G_GFPS_nodes = the
        mean of the full-tree loop's update gradients (3 x items_per_update normalisation, [final] + nodes per problem,
        consecutive micro-batches), as scripts/train_student.py computes them."""
        import grad_subspace as gs
        from cotdistill.student import kl_loss, label_logits, micro_batches
        tok = gpt2_tok()
        model = tiny_qwen2(tok)
        freeze(model)
        items = toy_items(2, 2)
        batch = gs.pick_batch(items, 4, 0, {"knights_knaves"}, "all")
        finals, nodes, _ = gs.build_views(batch, ["GFPS"], seed=3, tree_sources={"knights_knaves"})
        mats = gs.select_matrices(model, [0, 1], ["q_proj", "o_proj", "down_proj"])
        args = run_args(micro_bs=3)
        G, _ = gs.gradients(model, tok, finals, nodes, mats, args, autocast=False, store="cpu")
        for m in mats:
            m["param"].requires_grad_(True)
        ls, max_len = args.label_smoothing, args.max_len

        def grads():
            out = {m["name"]: m["param"].grad.detach().clone() for m in mats}
            for m in mats:
                m["param"].grad = None
            return out
        micro_bs, grad_accum = 2, 2                    # plain loop: one update of the 4 finals
        for i in range(grad_accum):
            b = finals[i * micro_bs:(i + 1) * micro_bs]
            z = label_logits(model, tok, b, max_len, {})
            (kl_loss(z, b, ls) * len(b) / len(b) / grad_accum).backward()
        a2 = grads()
        ipu, ups = 2, []                               # full-tree loop: 2 updates of 2 problems
        groups = [[fin] + per for fin, per in zip(finals, nodes["GFPS"])]
        for u in range(0, len(groups), ipu):
            upd = [e for g in groups[u:u + ipu] for e in g]
            for mb in micro_batches(tok, upd, max_len, 5):
                z = label_logits(model, tok, mb, max_len, {})
                (kl_loss(z, mb, ls) * len(mb) / (3 * ipu)).backward()
            ups.append(grads())
        for m in mats:
            n = m["name"]
            self.assertTrue(torch.allclose(G["final"][n], a2[n], rtol=1e-4, atol=1e-7), n)
            tree = sum(g[n] for g in ups) / len(ups)
            derived = G["final"][n] / 3 + G["GFPS"][n]
            self.assertLess(float((derived - tree).norm() / tree.norm()), 1e-5, n)
            self.assertGreater(float(G["GFPS"][n].norm()), 0)

    @unittest.skipUnless(HAVE_Q35, "needs Qwen3.5 in transformers")
    def test_checkpointing_groups_and_batching_do_not_change_gradients(self):
        import grad_subspace as gs
        from cotdistill.interchange import use_reference_kernels
        tok = gpt2_tok()
        items = toy_items(2, 2)
        batch = gs.pick_batch(items, 3, 0, {"knights_knaves"}, "all")
        finals, nodes, _ = gs.build_views(batch, ["GFPS"], seed=0, tree_sources={"knights_knaves"})
        nodes = {"GFPS": [per[:6] for per in nodes["GFPS"]]}       # a few views keep the test fast

        def run(ckpt: bool, **kw):
            model = q35(tok)
            use_reference_kernels(model)
            freeze(model)
            if ckpt:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            mats = gs.select_matrices(model, [3, 7], list(gs.ATTN + gs.MLP))
            G, info = gs.gradients(model, tok, finals, nodes, mats, run_args(**kw), autocast=False, store="cpu")
            return G, info
        ref, _ = run(False)
        for ckpt, kw in ((True, {}), (False, {"layer_groups": 2}), (False, {"token_budget": 300, "micro_bs": 2})):
            G, info = run(ckpt, **kw)
            for key in ref:
                for n, t in ref[key].items():
                    self.assertGreater(float(t.norm()), 0, (key, n))
                    self.assertLess(float((G[key][n] - t).norm() / t.norm()), 1e-4, (ckpt, kw, key, n))
            if kw.get("layer_groups"):
                self.assertEqual(len(info["final"]), 2)


def save_tiny(d: Path, kind: str) -> Path:
    tok = gpt2_tok()
    m = q35(tok) if kind == "q35" else tiny_qwen2(tok)
    m.save_pretrained(d)
    tok.save_pretrained(d)
    return d


def run_cli(*a, check: bool = True):
    out = subprocess.run([sys.executable, SCRIPT, *map(str, a)], capture_output=True, text=True, env=ENV, timeout=900)
    if check and out.returncode != 0:
        raise AssertionError(out.stdout[-3000:] + out.stderr[-3000:])
    return out


@unittest.skipUnless(HAVE_Q35, "needs torch + transformers with Qwen3.5")
class TestCLI(unittest.TestCase):
    def test_finetuned_report_and_check(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            base = save_tiny(d / "base", "q35")
            (d / "train.jsonl").write_text("\n".join(json.dumps(x) for x in toy_items(3, 3)))
            common = ["--model", base, "--train", d / "train.jsonl", "--n", 3, "--batch-sources", "tree",
                      "--node-arms", "GFPS,TFL", "--ranks", "2,4", "--token-budget", 0, "--micro-bs", 4]
            run_cli(*common, "--out", d / "o0", "--save-grads", d / "g.pt")
            raw = torch.load(d / "g.pt")
            r0 = json.loads((d / "o0/grad_subspace.json").read_text())
            self.assertEqual(r0["meta"]["layers"], [3, 7])
            self.assertEqual(len(r0["matrices"]), 14)
            self.assertEqual(r0["meta"]["batch"]["sources"], {"knights_knaves": 3})
            # finetuned models whose update is exactly -eta x a gradient
            eta = 1.0                       # large: fp32 rounding of W + dW stays far below dW's spectrum
            for name, key in (("ftg", "GFPS"), ("fta", "A2")):
                m = transformers.AutoModelForCausalLM.from_pretrained(base)
                sd = m.state_dict()
                with torch.no_grad():
                    for n, gfin in raw["final"].items():
                        g = gfin if key == "A2" else gfin / 3 + raw["nodes"][key][n]
                        sd[n].add_(-eta * g)
                m.save_pretrained(d / name)
                gpt2_tok().save_pretrained(d / name)
            run_cli(*common, "--out", d / "oA", "--finetuned", f"GFPS={d / 'ftg'}")
            run_cli(*common, "--out", d / "oB", "--finetuned", f"A2-c12={d / 'fta'}")
            rA = json.loads((d / "oA/grad_subspace.json").read_text())
            rB = json.loads((d / "oB/grad_subspace.json").read_text())
            for m in rA["matrices"]:
                v = m["dw"]["GFPS"]["vs"]["GFPS"]
                self.assertGreater(v["cos"], 0.9999, m["name"])
                self.assertGreater(v["r"]["2"]["left"], 0.999, m["name"])
                self.assertGreater(v["r"]["2"]["right"], 0.999, m["name"])
                for r in ("2", "4"):
                    self.assertAlmostEqual(v["r"][r]["energy"], m["grad"]["GFPS"]["top_energy"][r], places=4)
                self.assertGreater(m["dw"]["GFPS"]["rel"], 0)
                self.assertIn("A2~GFPS_nodes", m["pairs"])
            for m in rB["matrices"]:
                self.assertGreater(m["dw"]["A2-c12"]["vs"]["A2"]["cos"], 0.9999, m["name"])
            md = (d / "oA/grad_subspace.md").read_text()
            self.assertIn("## Gradient vs finished update dW_GFPS", md)
            self.assertIn("n/a (no dW_A2)", md)
            # merge the two "accounts"
            run_cli("--report", d / "oA/grad_subspace.json", d / "oB/grad_subspace.json", "--out", d / "rep")
            rep = json.loads((d / "rep/grad_subspace.json").read_text())
            self.assertLess(rep["meta"]["repro_max_rel"], 1e-6)
            self.assertTrue(all(set(m["dw"]) == {"GFPS", "A2-c12"} for m in rep["matrices"]))
            md = (d / "rep/grad_subspace.md").read_text()
            h1 = [l for l in md.splitlines() if l.startswith("| dW_GFPS | 2 |")]
            self.assertEqual(len(h1), 1, md[-3000:])
            cells = [c.strip() for c in h1[0].strip("|").split("|")]
            self.assertGreater(float(cells[2]), 0.999)                 # sim(G_GFPS, dW_GFPS)
            self.assertGreater(float(cells[3]), 0.999)                 # sim(G_A2, dW_A2)
            self.assertTrue(cells[7].endswith("/14)"), cells)           # within dW_A2: every matrix compared
            # jobs on other batches are not merged
            bad = dict(rB, meta=dict(rB["meta"], batch_items_sha1="x"))
            (d / "bad.json").write_text(json.dumps(bad))
            out = run_cli("--report", d / "oA/grad_subspace.json", d / "bad.json", "--out", d / "rep2", check=False)
            self.assertNotEqual(out.returncode, 0)
            self.assertIn("not the same batch", out.stderr)
            # same finetuned name twice
            out = run_cli("--report", d / "oA/grad_subspace.json", d / "oA/grad_subspace.json", "--out", d / "rep3",
                          check=False)
            self.assertNotEqual(out.returncode, 0)
            # --check: the trainer-layout identity, non-zero gradients, the full run's estimate
            out = run_cli(*common, "--out", d / "chk", "--check", "--check-n", 2, "--finetuned", f"GFPS={d / 'ftg'}")
            self.assertIn("check ok", out.stdout)
            c = json.loads((d / "chk/check.json").read_text())
            self.assertLess(max(c["direct_rel_err"].values()), 1e-4)
            self.assertTrue(c["nonzero"] and c["finite"])
            self.assertEqual(c["full_run_sequences"]["final"], 3)
            self.assertGreater(c["est_full_pass_minutes"], 0)
            rc = json.loads((d / "chk/grad_subspace.json").read_text())
            self.assertEqual({m["layer"] for m in rc["matrices"]}, {3})
            self.assertEqual(rc["meta"]["batch"]["n"], 2)

    def test_cli_errors(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "train.jsonl").write_text("\n".join(json.dumps(x) for x in toy_items(1, 2)))
            base = save_tiny(d / "base", "qwen2")
            for extra, msg in ((["--node-arms", "XX"], "--node-arms"), (["--finetuned", f"GFPS={d}"], "config.json"),
                               (["--tree-sources", "nope"], "no item"), (["--layers", "9"], "--layers")):
                out = run_cli("--model", base, "--train", d / "train.jsonl", "--out", d / "o", *extra, check=False)
                self.assertNotEqual(out.returncode, 0, extra)
                self.assertIn(msg, out.stderr, extra)
            # Qwen2: every:4 starts at layer 0; all sources; one node arm
            run_cli("--model", base, "--train", d / "train.jsonl", "--out", d / "o", "--n", 3, "--ranks", "2",
                    "--token-budget", 0)
            r = json.loads((d / "o/grad_subspace.json").read_text())
            self.assertEqual(r["meta"]["layers"], [0])
            self.assertEqual(r["meta"]["batch"]["sources"], {"toy": 2, "knights_knaves": 1})
            self.assertEqual(r["meta"]["node_arms"], ["GFPS"])
            self.assertTrue(math.isfinite(r["summary"]["all"]["pairs/A2~GFPS_nodes/r/2/left"]))


if __name__ == "__main__":
    unittest.main()
