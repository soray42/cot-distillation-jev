"""CPU tests for the K&K causal audit: the pair builder (solver-checked counterfactuals, no training puzzle), the token
sites and the position-aware interchange hooks (identity patches leave the logits unchanged), and an end-to-end run of
scripts/kk_audit.py (--check and a tiny audit) on a tiny random Qwen2. The builder needs the gitignored solver
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
        from test_student import tiny
        tok, model = tiny()
        pairs = small_pairs()
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d + "/m")
            tok.save_pretrained(d + "/m")
            Path(d + "/p.jsonl").write_text("\n".join(json.dumps(p) for p in pairs))
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
            base = [sys.executable, str(ROOT / "scripts/kk_audit.py"), "--model", d + "/m", "--pairs", d + "/p.jsonl",
                    "--layers", "0,1", "--max-len", "512", "--bs", "4"]
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


if __name__ == "__main__":
    unittest.main()
