"""CPU tests for the K&K causal audit: the pair builder (solver-checked counterfactuals, no training puzzle), the token
sites and the position-aware interchange hooks (identity patches leave the logits unchanged), end-to-end runs of
scripts/kk_audit.py (--check and a tiny audit; v1 and v2) on a tiny random Qwen2, and the v2 machinery: single-token
decodes over cached prefixes against the hook-patched full forward (logits and DAS gradients; tiny Qwen2 and a tiny
Qwen3.5 with Gated DeltaNet layers and mrope; sdpa and eager attention), the per-slot DAS fit (convergence, curve,
early stop), the layer screen, the linear probes and their control task. The builder needs the gitignored solver
(teacher_cache/solver_kk_v3f/build_reasoning_trees.py); those tests are skipped without it."""
import hashlib
import json
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

import build_kk_audit_pairs as B  # noqa: E402

HAVE_SOLVER = B.SOLVER.exists()
PAIRS = ROOT / "data/eval/kk_audit_pairs.jsonl"
try:
    import torch
    import transformers  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

_SMALL: list = []


def small_pairs() -> list[dict]:
    """A few programs of 4-5 people built in-process (cached for the module)."""
    if not _SMALL:
        solver = B.load_solver()
        tk, _ = B.training_keys()
        pairs, _ = B.build(solver, 4, range(4, 6), tk, {}, log=lambda *a: None)
        _SMALL.extend(pairs)
    return _SMALL


@unittest.skipUnless(HAVE_SOLVER, "needs the K&K solver in teacher_cache/")
class TestPairBuilder(unittest.TestCase):
    def test_counterfactuals_verified_by_solver(self):
        solver = B.load_solver()
        pairs = small_pairs()
        fams = {p["family"] for p in pairs}
        self.assertEqual(fams, {"role", "inv", "stmt"})
        for p in pairs:
            B.verify_pair(solver, p)                       # re-solves both prompts with ProofBuilder + enumeration
            people = p["people"]
            opts = [tuple(bool(b) for b in o) for o in p["options"]]
            # brute force, without the solver: the gold options are the unique consistent assignments
            for st_key, gold in (("base_statements", "base_gold"), ("source_statements", "source_gold")):
                st = {k: B._tup(v) for k, v in p[st_key].items()}
                sols = B.solutions(people, st)
                self.assertEqual(sols, [opts[p[gold]]])
            if p["family"] == "role" and p["t_prop"] is not None:
                # the propagated counterfactual differs from the base at the intervened person
                i = people.index(p["person"])
                self.assertNotEqual(opts[p["t_prop"]][i], opts[p["base_gold"]][i])
                self.assertEqual(p["n_desc"], sum(a != b for a, b in zip(opts[p["t_prop"]], opts[p["base_gold"]])) - 1)

    def test_propagation_model_on_a_chain(self):
        # Paul: "Mia is a knight or Leo is a knave." Mia: "Paul is a knight." Leo: "Mia is a knight." Unique solution:
        # all knights. The solver splits on Mia (refuting Mia = knave), then propagates Paul and Leo from Mia; Paul
        # and Leo have no role descendants, and flipping the case-refuted root Mia leaves a gate with no consistent row
        solver = B.load_solver()
        people = ["Paul", "Mia", "Leo"]
        stmts = {"Paul": ("or", ("is", "Mia", True), ("is", "Leo", False)), "Mia": ("is", "Paul", True),
                 "Leo": ("is", "Mia", True)}
        sols = B.solutions(people, stmts)
        self.assertEqual(sols, [(True, True, True)])
        pm = B.ProofModel(solver, people, stmts, sols[0], "toy")
        self.assertEqual(pm.how, {"Paul": "prop", "Mia": "case", "Leo": "prop"})
        self.assertEqual(pm.do(None), (True, True, True))
        self.assertEqual(pm.do("Paul"), (False, True, True))
        self.assertEqual(pm.do("Leo"), (True, True, False))
        self.assertIsNone(pm.do("Mia"))
        self.assertTrue(B.entailed_cf(pm, "Paul") and B.entailed_cf(pm, "Leo"))

    def test_case_root_recomputed_from_its_refutation_inputs(self):
        # kkaud-6-00046 (review_validity.md F1): Emma is case-refuted, but refuting "Emma is a knave" uses the earlier
        # facts role(Irene) and Irene's requirement, so under do(Irene = knight) Emma is recomputed (forced knave),
        # not held fixed; Karen, Leo and David follow from her. The old model gave Emma knight, Karen knave, Leo knight.
        solver = B.load_solver()
        T = lambda n, v: ("is", n, v)
        people = ["Irene", "Emma", "David", "Karen", "Leo", "Henry"]
        st = {"Irene": ("and", T("Emma", False), T("Leo", False)), "Emma": ("iff", T("Karen", False), T("Irene", True)),
              "David": ("iff", T("Leo", True), T("Henry", True)), "Karen": ("if", T("Emma", False), T("David", True)),
              "Leo": ("iff", T("Irene", True), T("Karen", False)), "Henry": T("Irene", True)}
        sol = B.solutions(people, st)[0]
        pm = B.ProofModel(solver, people, st, sol, "kkaud-6-00046")
        emma = next(f for f in pm.facts.values() if f["var"] == ("role", "Emma"))
        self.assertIsNone(emma["gate"])
        self.assertTrue(emma["parents"])
        self.assertEqual(pm.do("Irene"), (True, False, False, True, False, True))
        self.assertTrue(B.entailed_cf(pm, "Irene"))

    def test_entailment_check_rejects_a_wrong_counterfactual(self):
        solver = B.load_solver()
        mid = [p for p in small_pairs() if p["family"] == "role" and p["subfam"] == "mid"]
        self.assertTrue(mid)
        for p in mid[:5]:
            people, x = p["people"], p["person"]
            st = {k: B._tup(v) for k, v in p["base_statements"].items()}
            pm = B.ProofModel(solver, people, st, B.solutions(people, st)[0], p["program"])
            val = pm.do_facts(x)
            moved = [i for i, f in pm.facts.items() if val[i] != f["base"] and i != pm.role_node[x]]
            self.assertTrue(moved)                       # a descendant fact changes
            self.assertTrue(B.entailed_cf(pm, x))
            pm.do_facts = lambda person, val=val, d=moved[-1]: {**val, d: not val[d]}
            self.assertFalse(B.entailed_cf(pm, x))

    def test_no_training_items(self):
        tk, n = B.training_keys()
        self.assertGreater(n, 1000)
        ids, prompts = set(), set()
        for d in B.TRAIN_DIRS:
            for split in ("train", "val"):
                path = ROOT / d / f"{split}.jsonl"
                if path.exists():
                    with open(path) as f:
                        for line in f:
                            r = json.loads(line)
                            ids.add(r["item_id"])
                            prompts.add(r["prompt"])
        files = [small_pairs()]
        if PAIRS.exists():
            with open(PAIRS) as f:
                files.append([json.loads(l) for l in f])
        for pairs in files:
            for p in pairs:
                self.assertNotIn(p["program"], ids)
                self.assertNotIn(p["pair_id"], ids)
                for k in ("base_prompt", "source_prompt"):
                    self.assertNotIn(p[k], prompts)
                    self.assertNotIn(B.statement_key(p[k]), tk)

    @unittest.skipUnless(PAIRS.exists(), "data/eval/kk_audit_pairs.jsonl not built")
    def test_built_file_sample_verified(self):
        solver = B.load_solver()
        with open(PAIRS) as f:
            pairs = [json.loads(l) for l in f]
        summ = json.loads(Path(str(PAIRS) + ".summary.json").read_text())
        self.assertEqual(summ["md5"], hashlib.md5(PAIRS.read_bytes()).hexdigest())
        self.assertEqual(len({p["pair_id"] for p in pairs}), len(pairs))
        for p in random.Random(0).sample(pairs, min(150, len(pairs))):
            B.verify_pair(solver, p)


@unittest.skipUnless(HAVE_SOLVER and HAVE_TORCH, "needs the solver and torch + transformers")
class TestSitesAndHooks(unittest.TestCase):
    def test_identity_patch_and_positions(self):
        from test_student import tiny
        from cotdistill import interchange as ic
        import kk_audit as K
        from cotdistill.student import FINAL_TEMPLATE
        tok, model = tiny()
        model.eval()
        pairs = [p for p in small_pairs() if p["family"] == "role"][:6]
        bp, lb = [p["base_prompt"] for p in pairs], [p["labels"] for p in pairs]
        stmt = [K.token_pos(tok, p["base_prompt"], p["base_spans"]["stmt_end"], 512) for p in pairs]
        name = [K.token_pos(tok, p["base_prompt"], p["base_spans"]["name_end"], 512) for p in pairs]
        for p, s, n in zip(pairs, stmt, name):
            ids = tok.encode(FINAL_TEMPLATE.format(problem=p["base_prompt"]), add_special_tokens=False)
            self.assertIn('"', tok.decode([ids[s]]))
            self.assertIn(tok.decode([ids[n]]).strip(), p["person"])
        cache: dict = {}
        with torch.no_grad():
            b = ic._batch(tok, bp, lb, 512, "cpu")
            ref = ic._letter_logits(model, tok, b, lb, cache)
            for L in (0, 1):
                for pos in (None, stmt, name):
                    z = ic.patched_logits(model, tok, bp, lb, L, lambda h: h, 512, cache, pos=pos)
                    self.assertLess(max(float((a - c).abs().max()) for a, c in zip(ref, z)), 1e-5)
            # a full-state exchange from the base's own states is also the identity, at any site
            for pos in (None, stmt):
                st = ic.site_states(model, tok, bp, lb, [0, 1], 512, 4, pos=pos)
                recs = [dict(p, target=p["base_gold"], base_pred=p["base_gold"], source_pred=-1,
                             **({"base_pos": q} if pos else {})) for p, q in zip(pairs, stmt)]
                for j, L in enumerate((0, 1)):
                    r = ic.iia(model, tok, recs, st[:, j], L, "full", max_len=512, bs=4, return_preds=True)
                    self.assertEqual(r["preds"], [int(z.argmax()) for z in ref])
            # a patch at the statement site changes the readout (the hook really acts there)
            z = ic.patched_logits(model, tok, bp, lb, 0, lambda h: h + 10.0, 512, cache, pos=stmt)
            self.assertGreater(max(float((a - c).abs().max()) for a, c in zip(ref, z)), 1e-4)

    def test_kk_audit_script_end_to_end(self):
        """The registered v1 audit (--preset v1) stays runnable."""
        from test_student import tiny
        tok, model = tiny()
        pairs = small_pairs()
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d + "/m")
            tok.save_pretrained(d + "/m")
            Path(d + "/p.jsonl").write_text("\n".join(json.dumps(p) for p in pairs))
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
            base = [sys.executable, str(ROOT / "scripts/kk_audit.py"), "--model", d + "/m", "--pairs", d + "/p.jsonl",
                    "--layers", "0,1", "--max-len", "512", "--bs", "4", "--preset", "v1"]
            out = subprocess.run(base + ["--out", d + "/c", "--check"], capture_output=True, text=True, timeout=600,
                                 env=env)
            self.assertEqual(out.returncode, 0, out.stderr[-3000:])
            c = json.loads(Path(d + "/c/check.json").read_text())
            self.assertTrue(all(v < 1e-5 for k, v in c.items() if k.startswith("identity_max_abs_diff")), c)
            self.assertEqual(c["stmt_site_is_quote"], 1.0)
            self.assertEqual(c["name_site_in_name"], 1.0)
            self.assertGreater(c["das_grad_norm"], 0)
            self.assertEqual(c["model_grads"], 0)
            self.assertTrue(c["weights_unchanged"])
            self.assertTrue(all(v is None or v["norm_ratio"] > 0.99 for v in c["exchange_fidelity"].values()))  # fp32
            out = subprocess.run(base + ["--out", d + "/r", "--all-pairs", "--min-pairs", "2", "--max-train", "8",
                                         "--max-test", "8", "--epochs", "1", "--name", "tiny", "--min-slot-pairs", "1",
                                         "--conds", "role:local@last,role:z@last,stmt:source@stmt,role:local@stmt"],
                                 capture_output=True, text=True, timeout=900, env=env)
            self.assertEqual(out.returncode, 0, out.stderr[-3000:])
            lines = [json.loads(l) for l in Path(d + "/r/summary.jsonl").read_text().splitlines()]
            conds = {(r["cond"], r.get("layer")) for r in lines}
            for cnd in ("role:local@last", "role:z@last", "stmt:source@stmt", "inv:base@last"):
                for L in (0, 1):
                    self.assertIn((cnd, L), conds)
            r = next(r for r in lines if r["cond"] == "role:local@last" and r["layer"] == 1)
            for m in ("das", "random", "noise", "full"):
                self.assertTrue(0 <= r[m]["iia"] <= 1)
                self.assertIn("to_t_local", r[m])
            self.assertTrue(all(l.startswith("KKAUDIT ") for l in out.stdout.splitlines() if l.startswith("KKAUDIT")))
            # controls on both test sets; per-pair predictions aligned with the pairs line and reproducing the scores
            self.assertEqual(set(r["all_pairs"]), {"das", "random", "noise", "full"})
            self.assertEqual(r["das_all_pairs"], r["all_pairs"]["das"])
            recs = [json.loads(l) for l in Path(d + "/r/preds.jsonl").read_text().splitlines()]
            meta = next(x for x in recs if x["kind"] == "pairs" and x["cond"] == "role:local@last")["pairs"]
            for x in recs:
                if x["kind"] == "preds" and x["cond"] == "role:local@last" and x["layer"] == 1:
                    keep = [k for p, k in zip(meta, x["preds"]) if p["in_all"]]
                    tg = [p["target"] for p in meta if p["in_all"]]
                    iia = sum(a == b for a, b in zip(keep, tg)) / len(tg)
                    self.assertAlmostEqual(iia, r["all_pairs"][x["method"]]["iia"], places=3)
            self.assertTrue({("inv:base@last", "pairs"), ("role:local@stmt", "pairs")}
                            <= {(x["cond"], x["kind"]) for x in recs})
            # at the statement site, role pairs whose edited line follows X's line are dropped (no-op exchange)
            st = next(r for r in lines if r["cond"] == "role:local@stmt" and r.get("layer") == 0)
            role = [p for p in pairs if p["family"] == "role"]
            later = sum(p["base_prompt"].find("- " + p["edited"] + " says") > p["base_prompt"].find("- " + p["person"]
                                                                                                      + " says")
                        for p in role)
            self.assertEqual(st["n_noop"], later)


def tiny_q35(tok, impl: str = "sdpa"):
    """A tiny Qwen3.5 text model (3 Gated DeltaNet layers + 1 full-attention layer, twice; partial mrope)."""
    cfg = transformers.Qwen3_5TextConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=96, num_hidden_layers=8,
                                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                         linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
                                         linear_num_value_heads=4, max_position_embeddings=1024,
                                         rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                                          "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1],
                                                          "mrope_interleaved": True},
                                         attn_implementation=impl)
    torch.manual_seed(0)
    return transformers.Qwen3_5ForCausalLM(cfg).eval()


@unittest.skipUnless(HAVE_TORCH, "needs torch + transformers")
class TestDecodePath(unittest.TestCase):
    PROMPTS = ["Case one. The light is red." * k + "\nOptions:\nA) x\nB) y\nC) z" for k in (1, 3, 5, 2)]

    def models(self):
        from test_student import tiny
        tok, q2 = tiny()
        out = [("qwen2-sdpa", q2.eval())]
        q2e = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(**{**q2.config.to_dict(),
                                                                         "attn_implementation": "eager"})).eval()
        q2e.load_state_dict(q2.state_dict())
        out.append(("qwen2-eager", q2e))
        if hasattr(transformers, "Qwen3_5ForCausalLM"):
            out += [("qwen3.5-sdpa", tiny_q35(tok)), ("qwen3.5-eager", tiny_q35(tok, "eager"))]
        return tok, out

    def test_decode_matches_hook_patched_forward(self):
        from cotdistill import interchange as ic
        ic.set_precision("fp32")
        tok, ms = self.models()
        labels = [["A", "B", "C"]] * 4
        for name, model in ms:
            with self.subTest(model=name):
                rep = ic.use_reference_kernels(model)
                self.assertEqual(len(rep), 4 if "3.5" in name else 0)
                self.assertEqual(ic.use_reference_kernels(model), rep)               # idempotent
                for prm in model.parameters():
                    prm.requires_grad_(False)
                n = len(ic.decoder_layers(model))
                logits, st = ic.readout_pass(model, tok, self.PROMPTS, labels, list(range(n)), 512, 2)
                store = ic.PrefixStore(model, tok, 1, 512, "cpu", 1e9)
                store.ensure(list(range(4)), self.PROMPTS)
                cache: dict = {}
                for L in range(1, n):                                  # unpatched decode from every layer
                    z = ic.decode_logits(model, tok, store, list(range(4)), st[:, L], L, labels, cache)
                    self.assertLess(max(float((a - b).abs().max()) for a, b in zip(z, logits)), 1e-4)
                L = n // 2
                src = st.flip(0)[:, L]
                s = ic.Subspace(64, 2)
                R = s.R[None].expand(4, -1, -1)
                zd = ic.decode_logits(model, tok, store, list(range(4)), ic.exchange_rows(st[:, L], src, R), L, labels,
                                      cache)
                zh = ic.patched_logits(model, tok, self.PROMPTS, labels, L,
                                       lambda hb: ic.exchange_rows(hb.float(), src, R), 512, cache)
                self.assertLess(max(float((a - b).abs().max()) for a, b in zip(zd, zh)), 1e-4)
                gd = torch.autograd.grad(sum(z[1] for z in zd), list(s.parameters()), retain_graph=True)
                gh = torch.autograd.grad(sum(z[1] for z in zh), list(s.parameters()))
                self.assertGreater(float(gh[0].norm()), 0)
                self.assertLess(max(float((a - b).abs().max()) for a, b in zip(gd, gh)), 1e-4 * (1 + float(gh[0].norm())))
                # a wrong prefix (another prompt's) changes the decoded answer logits: the cache is really read
                zw = ic.decode_logits(model, tok, store, [1, 2, 3, 0], st[:, L], L, labels, cache)
                self.assertGreater(max(float((a - b).abs().max()) for a, b in zip(zw, logits)), 1e-4)

    def test_store_lru_cap_keeps_requested(self):
        from cotdistill import interchange as ic
        from test_student import tiny
        tok, model = tiny()
        model.eval()
        store = ic.PrefixStore(model, tok, 0, 512, "cpu", 1.0)            # 1 byte: evict everything not requested
        store.ensure([0, 1], self.PROMPTS[:2])
        self.assertEqual(set(store.items), {0, 1})
        store.ensure([2], self.PROMPTS[2:3])
        self.assertEqual(set(store.items), {2})
        self.assertEqual(store.n_built, 3)

    def test_precision_policy(self):
        from cotdistill import interchange as ic
        info = ic.set_precision("fp32")
        self.assertEqual((info["autocast_bf16"], info["tf32_matmul"], info["tf32_cudnn"]), (False, False, False))
        self.assertFalse(ic._AMP["autocast"])
        self.assertTrue(ic.set_precision("fp32-autocast")["autocast_bf16"])
        ic.set_precision("fp32")


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class TestFitAndProbes(unittest.TestCase):
    def test_exchange_and_noise_rows(self):
        from cotdistill import interchange as ic
        s = ic.random_subspace(16, 3, seed=1)
        hb, hs = torch.randn(5, 16), torch.randn(5, 16)
        R = s.R.detach()[None].expand(5, -1, -1)
        self.assertTrue(torch.allclose(ic.exchange_rows(hb, hs, R), s.exchange(hb, hs), atol=1e-6))
        z = ic.noise_rows(hb, hs, R, torch.Generator().manual_seed(0)) - hb
        P = s.R.detach().T @ s.R.detach()
        self.assertTrue(torch.allclose(z @ P, z, atol=1e-5))                          # inside the subspace
        self.assertTrue(torch.allclose(z.norm(dim=-1), ((hs - hb) @ P).norm(dim=-1), atol=1e-4))   # norm matched

    def test_fit_slot_subspaces_finds_the_variable(self):
        # two slots, each with a hidden direction u_g: the target is the sign of the source's coordinate on u_g, read
        # from the patched state (base = 0). A fit must align R_g with u_g and push the loss far below chance.
        from cotdistill import interchange as ic
        g = torch.Generator().manual_seed(0)
        d = 16
        u = {0: torch.nn.functional.normalize(torch.randn(d, generator=g), dim=0),
             1: torch.nn.functional.normalize(torch.randn(d, generator=g), dim=0)}
        n = 120
        hs = torch.randn(n, d, generator=g)
        slot = [i % 2 for i in range(n)]
        tgt = [int(float(hs[i] @ u[slot[i]]) > 0) for i in range(n)]
        fit, val = list(range(0, 90)), list(range(90, n))

        def fwd(rows, train, R):
            idx = [(fit if train else val)[i] for i in rows]
            h = ic.exchange_rows(torch.zeros(len(idx), d), hs[idx], R)
            return [torch.stack([torch.zeros(()), 8 * h[j] @ u[slot[i]]]) for j, i in enumerate(idx)]
        subs, info = ic.fit_slot_subspaces(fwd, d, 1, [slot[i] for i in fit], [slot[i] for i in val],
                                           [tgt[i] for i in fit], [tgt[i] for i in val], [2] * len(fit),
                                           steps_min=100, steps_max=600, patience=60, eval_every=10, rows_per_slot=8,
                                           lr=3e-2, seed=0)
        self.assertEqual(set(subs), {0, 1})
        self.assertGreater(info["rel_drop"], 0.5)
        self.assertTrue(info["below_chance"])
        self.assertEqual(info["loss_on"], "val")
        self.assertEqual(info["n_train_by_slot"], {"0": 45, "1": 45})
        self.assertEqual(info["curve"][0]["step"], 0)
        self.assertGreaterEqual(info["steps"], 100)
        self.assertTrue(info["converged"])
        self.assertLessEqual(info["best_loss"], min(c["val_loss"] for c in info["curve"]) + 1e-4)
        for k in (0, 1):
            self.assertGreater(abs(float(subs[k].R.detach()[0] @ u[k])), 0.9)

    def test_probe_and_control(self):
        import kk_audit as K
        g = torch.Generator().manual_seed(0)
        y = torch.randint(0, 2, (400,), generator=g).float()
        X = torch.randn(400, 32, generator=g) + 2 * y[:, None] * torch.eye(32)[0]
        progs = [f"g{i // 4}" for i in range(400)]
        r = K.probe(X[:300], y[:300], progs[:300], X[300:], y[300:])
        self.assertGreater(r["acc"], 0.85)
        rnd = torch.randint(0, 2, (400,), generator=g).float()                   # labels unrelated to X
        r0 = K.probe(X[:300], rnd[:300], progs[:300], X[300:], rnd[300:])
        self.assertLess(r0["acc"], 0.65)
        self.assertEqual(K.control_label("Mia"), K.control_label("Mia"))
        self.assertEqual({K.control_label(x) for x in ("Mia", "Leo", "Paul", "Emma", "Irene", "Jack", "Noah")}, {0, 1})

    def test_layer_screen_keeps_layers_with_copy(self):
        import kk_audit as K
        from unittest import mock
        pairs = [{"pair_id": f"s{i}", "program": f"g{i}", "family": "stmt", "base_ok": True, "source_ok": True,
                  "base_pred": 0, "source_pred": 1 + i % 3} for i in range(80)]

        class Fake:
            def __init__(self, *a, **kw):
                self.fidelity = {}

            def ensure(self, recs):
                pass

            def logits_at(self, recs, L, hfun):
                return [torch.nn.functional.one_hot(torch.tensor(p["source_pred"] if L in (16, 20) or
                                                                 (L == 15 and i % 3 == 0) else 0), 6).float()
                        for i, p in enumerate(recs)]
        args = type("A", (), {"screen_pairs": 128, "screen_min": 0.5})()
        with mock.patch.object(K, "Readout", Fake):
            sel, info = K.screen_layers(args, None, None, pairs, None, None, {}, [13, 14, 15, 16, 17, 20], "cpu")
        self.assertEqual(sel, [16, 20])
        self.assertEqual(info["copy_source_full"][16], 1.0)
        self.assertLess(info["copy_source_full"][15], 0.5)
        self.assertEqual(K.candidate_layers(24), list(range(13, 21)))
        self.assertEqual(K.candidate_layers(32), list(range(17, 29)))
        self.assertEqual(K.probe_layers_auto(24), [16, 20])


@unittest.skipUnless(HAVE_SOLVER and HAVE_TORCH, "needs the solver and torch + transformers")
class TestAuditV2EndToEnd(unittest.TestCase):
    def test_check_and_run_v2(self):
        from test_student import tiny
        tok, model = tiny()
        pairs = small_pairs()
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d + "/m")
            tok.save_pretrained(d + "/m")
            Path(d + "/p.jsonl").write_text("\n".join(json.dumps(p) for p in pairs))
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
            base = [sys.executable, str(ROOT / "scripts/kk_audit.py"), "--model", d + "/m", "--pairs", d + "/p.jsonl",
                    "--max-len", "512", "--bs", "4", "--k", "1,2", "--das-bs", "8"]
            out = subprocess.run(base + ["--out", d + "/c", "--check", "--layers", "gate:0-1", "--check-fit-steps", "10"],
                                 capture_output=True, text=True, timeout=600, env=env)
            self.assertEqual(out.returncode, 0, out.stderr[-3000:])
            c = json.loads(Path(d + "/c/check.json").read_text())
            self.assertEqual(c["version"], "v2")
            self.assertEqual((c["precision"], c["autocast_bf16"], c["tf32_matmul"]), ("fp32", False, False))
            self.assertEqual(c["matmul_output_dtype"], "float32")
            self.assertLess(c["decode_max_abs_diff"], 1e-4)
            self.assertEqual(c["decode_prefix_fidelity"]["argmax_mismatch"], 0)
            self.assertTrue(c["weights_unchanged"])
            self.assertGreater(c["das_grad_norm"], 0)
            self.assertEqual(c["model_grads"], 0)
            self.assertEqual(c["posctl_fit"]["steps"], 10)
            self.assertIn("rel_drop", c["posctl_fit"])
            self.assertIn("exchange_fidelity_bf16", c)
            self.assertIn("expected", c["gpu_hours_estimate"])
            out = subprocess.run(base + ["--out", d + "/r", "--layers", "0,1", "--all-pairs", "--min-pairs", "2",
                                         "--min-slot-pairs", "1", "--pool-slots", "--max-train", "16", "--max-test", "16",
                                         "--das-steps", "10", "--das-steps-max", "20", "--das-patience", "5",
                                         "--das-eval-every", "5", "--val-frac", "0.5", "--name", "tiny-s1"],
                                 capture_output=True, text=True, timeout=900, env=env)
            self.assertEqual(out.returncode, 0, out.stderr[-3000:])
            lines = [json.loads(l) for l in Path(d + "/r/summary.jsonl").read_text().splitlines()]
            conds = {r["cond"] for r in lines}
            self.assertTrue({"_head", "_probes", "_gate", "stmt:source@last", "role:local@last", "inv:base@last"}
                            <= conds, conds)
            gate = next(r for r in lines if r["cond"] == "_gate")
            self.assertEqual(set(gate["iia"]), {"L0/k1", "L0/k2", "L1/k1", "L1/k2"})
            r = next(r for r in lines if r["cond"] == "role:local@last" and r["layer"] == 1 and r["k"] == 2)
            self.assertEqual(set(r["both"]), {"das", "random", "noise", "full"})
            self.assertIn("coinc", r["all"]["das"])
            self.assertGreaterEqual(r["fit"]["steps"], 10)
            recs = [json.loads(l) for l in Path(d + "/r/preds.jsonl").read_text().splitlines()]
            meta = next(x for x in recs if x["kind"] == "pairs" and x["cond"] == "role:local@last")["pairs"]
            for x in recs:
                if x["kind"] == "preds" and x["cond"] == "role:local@last" and x["layer"] == 1 and x["k"] == 2:
                    keep = [k for p, k in zip(meta, x["preds"]) if p["in_all"]]
                    tg = [p["target"] for p in meta if p["in_all"]]
                    self.assertAlmostEqual(sum(a == b for a, b in zip(keep, tg)) / len(tg),
                                           r["all"][x["method"]]["iia"], places=3)
            inv = next(x for x in recs if x["kind"] == "pairs" and x["cond"] == "inv:base@last")["pairs"]
            self.assertTrue(all(p["target"] == p["base_pred"] for p in inv))      # retention of the own answer
            a = json.loads(Path(d + "/r/audit.json").read_text())
            f = a["fits"]["role:local@last|L1|k2"]
            self.assertEqual(f["curve"][0]["step"], 0)
            self.assertIn(f["stop"], ("early_stop", "steps_max"))
            self.assertLess(a["decode_fidelity"]["max_abs_diff"], 1e-4)
            self.assertIsNotNone(a["gate"])
            self.assertIn("name", a["head"]["probes"])
            # the v2 endpoints read this run
            import kk_audit_endpoints as E
            import io
            from contextlib import redirect_stdout
            with redirect_stdout(io.StringIO()):
                res = E.main([d + "/r", "--reps", "50"])
            self.assertEqual(res["models"][0]["cells"], ["L0/k1", "L0/k2", "L1/k1", "L1/k2"])


if __name__ == "__main__":
    unittest.main()
