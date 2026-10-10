import itertools
import os
import random
import sys
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import kk_reasoning_type as kr  # noqa: E402

HAVE_SOLVER = kr.SOLVER.exists()
HELDOUT = ROOT / "data" / "eval" / "kk_heldout.jsonl"
HEAD = ("On an island, knights always tell the truth and knaves always lie. Each of the {n} inhabitants below is either "
        "a knight or a knave.\n\n")


def prompt(names, stmts, options):
    lines = "".join(f'- {nm} says: "{st}"\n' for nm, st in zip(names, stmts))
    opts = "".join(f"{L}) " + ", ".join(f"{nm}: {'knight' if a[nm] else 'knave'}" for nm in names) + "\n"
                   for L, a in zip("ABCDEF", options))
    return HEAD.format(n=len(names)) + lines + "\nQuestion: Who is a knight and who is a knave?\nOptions:\n" + opts.rstrip()


def models(names, stmts):
    from build_local_kk import load_solver
    s = load_solver(kr.SOLVER)
    ex = {nm: s.parse_expression(st) for nm, st in zip(names, stmts)}
    out = []
    for bits in itertools.product((False, True), repeat=len(names)):
        a = dict(zip(names, bits))
        if all(ex[nm].evaluate(a) == a[nm] for nm in names):
            out.append(a)
    return out


def item(item_id, names, stmts, seed=0, **extra):
    sol = models(names, stmts)
    assert len(sol) == 1, (names, stmts, sol)
    rng = random.Random(seed)
    others = [dict(zip(names, b)) for b in itertools.product((False, True), repeat=len(names))]
    others = [a for a in others if a != sol[0]]
    opts = rng.sample(others, 5) + [sol[0]]
    rng.shuffle(opts)
    return dict({"item_id": item_id, "prompt": prompt(names, stmts, opts),
                 "gold_label": "ABCDEF"[opts.index(sol[0])]}, **extra)


# one flat hypothesis: Carol (degree 2) is refuted as a knight, then Alice and Bob follow by propagation
F_PUZZLE = (["Alice", "Bob", "Carol"],
            ["Bob is a knave.", "Carol is a knave.", "Alice is a knight and Carol is a knave."])
# no hypothesis: Alice's statement is a tautology, everything else follows
U_PUZZLE = (["Alice", "Bob", "Carol"],
            ["Alice is a knight or Alice is a knave.", "Alice is a knave.", "Bob is a knave."])
ATOMS = ("{} is a knight", "{} is a knave")


def random_puzzles(n_items, seed=0):
    rng = random.Random(seed)
    pool = ["Alice", "Bob", "Carol", "David", "Emma", "Frank"]
    out = []
    while len(out) < n_items:
        names = rng.sample(pool, rng.choice([4, 5]))
        stmts = []
        for nm in names:
            a = rng.choice(ATOMS).format(rng.choice([x for x in names if x != nm]))
            b = rng.choice(ATOMS).format(rng.choice([x for x in names if x != nm]))
            if a.split()[0] == b.split()[0]:
                stmts.append(a + ".")
                continue
            stmts.append(rng.choice([f"{a} and {b}.", f"{a} or {b}.", f"{a} if and only if {b}.",
                                     f"If {a}, then {b}.", f"{a}."]))
        if len(models(names, stmts)) == 1:
            out.append(item(f"syn-{len(out)}", names, stmts, seed=len(out)))
    return out


class TestClassify(unittest.TestCase):
    def test_classes(self):
        self.assertEqual(kr.classify(0, 0), "U")
        self.assertEqual(kr.classify(1, 1), "F")
        self.assertEqual(kr.classify(1, 2), "M")
        self.assertEqual(kr.classify(2, 1), "N")
        self.assertEqual(kr.classify(3, 4), "N")


@unittest.skipUnless(HAVE_SOLVER, "training-tree solver absent")
class TestSolverClasses(unittest.TestCase):
    def test_hand_checked(self):
        info = kr.kk_reasoning([item("f", *F_PUZZLE), item("u", *U_PUZZLE)], orders=4)
        f, u = info["f"], info["u"]
        self.assertEqual((f["cls"], f["case"], f["splits"], f["hyps"], f["forward"], f["failed"]), ("F", 1, 1, 1, 2, 1))
        self.assertEqual((u["cls"], u["case"], u["splits"], u["hyps"], u["forward"], u["case_nodes"]),
                         ("U", 0, 0, 0, 3, 0))
        self.assertEqual(f["cls_min"], "F")
        self.assertGreaterEqual(f["min_viol"], 1)        # unique solution: every distractor breaks a statement

    def test_invariants_and_depth_readout(self):
        items = random_puzzles(40)
        info = kr.kk_reasoning(items, orders=3)
        self.assertEqual(kr.check_against_depth_readout(info, items)["match"], len(items))
        for v in info.values():
            self.assertEqual(v["cls"], kr.classify(v["case"], v["splits"]))
            self.assertEqual(v["splits"] + v["forward"], v["people"])
            self.assertEqual(v["case"] == 0, v["splits"] == 0)
            self.assertEqual(v["width"], v["case_nodes"] + v["derive_nodes"])
            self.assertLessEqual(kr.CLASSES.index(v["cls_min"]), kr.CLASSES.index(v["cls"]))
            self.assertEqual(sum(v["cls_orders"].values()), 4)
        self.assertGreater(sum(v["cls"] == "F" for v in info.values()), 0)

    def test_hyps_is_refute_calls(self):
        from build_local_kk import load_solver
        s = load_solver(kr.SOLVER)
        items = random_puzzles(25, seed=1) + [item("f", *F_PUZZLE)]
        if HELDOUT.exists():
            items += kr.dr.read_jsonl(HELDOUT)[:80]
        for it in items:
            b = s.ProofBuilder(s.Puzzle.read({"item_id": it["item_id"], "prompt": it["prompt"]}, "eval"))
            calls = []
            orig = b.refute

            def wrapped(state, scope, orig=orig, calls=calls):
                calls.append(scope)
                return orig(state, scope)
            b.refute = wrapped
            b.solve()
            self.assertEqual(kr.proof_counts(b)["hyps"], len(calls), it["item_id"])

    def test_diag_inherits_r0(self):
        names, stmts = F_PUZZLE
        r0 = item("p1-R0", names, stmts, group="diag/R0", meta={"program": "p1"})
        r1 = dict(r0, item_id="p1-R1", group="diag/R1", prompt=r0["prompt"].replace(" says:", " claims:"))
        c = item("p1-C", *U_PUZZLE, group="diag/C", meta={"program": "p1"})
        info = kr.diag_reasoning([r0, r1, c])
        self.assertEqual(info["p1-R1"]["cls"], "F")
        self.assertEqual(info["p1-C"]["cls"], "U")
        self.assertEqual({v["program"] for v in info.values()}, {"p1"})

    @unittest.skipUnless(HELDOUT.exists(), "kk_heldout absent")
    def test_heldout_has_no_unit_items(self):
        items = kr.dr.read_jsonl(HELDOUT)[:120]
        info = kr.kk_reasoning(items)
        self.assertEqual(kr.check_against_depth_readout(info, items)["match"], len(items))
        cc = kr.class_counts(info)
        self.assertEqual(cc["U"], 0)                     # propagation from no facts settles nobody
        self.assertGreater(cc["F"], 0)
        self.assertGreater(cc["N"], 0)


def fake_groups(n_per=100, strata=(4, 5, 6, 7), seeds=(12, 13), effect_f=0.2, effect_s=0.0, seed=0):
    rng = np.random.default_rng(seed)
    info, groups = {}, {"ARM": {}, "REF": {}}
    ids = []
    for p in strata:
        for j in range(n_per):
            i = f"p{p}-{j}"
            ids.append(i)
            info[i] = {"people": p, "cls": "F" if j < 0.6 * n_per else ("N" if j % 2 else "M")}
    for s in seeds:
        ref = {i: int(rng.random() < 0.5) for i in ids}
        arm = {}
        for i in ids:
            eff = effect_f if info[i]["cls"] == "F" else effect_s
            arm[i] = 1 if (ref[i] == 1 or rng.random() < eff / 0.5) else 0
        groups["REF"][s] = {"_u": {"x": ref}}
        groups["ARM"][s] = {"_u": {"x": arm}}
    return groups, info


class TestStats(unittest.TestCase):
    def test_planted_interaction(self):
        groups, info = fake_groups(n_per=200)
        res = kr.pair_analysis(groups, "ARM", "REF", "x", info, kr.KK_BINARY, 400, 60, 0, cont=(), alt_keys=())
        F = [i for i in info if info[i]["cls"] == "F"]
        direct = np.mean([np.mean([groups["ARM"][s]["_u"]["x"][i] - groups["REF"][s]["_u"]["x"][i] for s in (12, 13)])
                          for i in F])
        self.assertAlmostEqual(res["classes"]["F"]["diff"], direct)
        self.assertGreater(res["classes"]["F"]["diff"], 0.12)
        self.assertLess(abs(res["classes"]["S"]["diff"]), 1e-12)
        self.assertGreater(res["did"]["F-S"]["ci"][0], 0)
        self.assertGreater(res["did_people"]["ci"][0], 0)
        lg = next(iter(res["logit"].values()))
        self.assertLess(lg["per_unit_ci"][1], 0)
        self.assertTrue(kr.verdict(res).startswith("supported"))
        self.assertEqual(res["classes"]["U"], {"n": 0})

    def test_null_is_not_supported(self):
        groups, info = fake_groups(effect_f=0.0)
        res = kr.pair_analysis(groups, "ARM", "REF", "x", info, kr.KK_BINARY, 200, 30, 0, cont=(), alt_keys=())
        self.assertEqual(res["did"]["F-S"]["point"], 0.0)
        self.assertTrue(kr.verdict(res).startswith("not supported"))

    def test_cluster_draws(self):
        strata = np.repeat([4, 5], 12)
        cluster = np.repeat(np.arange(6), 4)             # 3 clusters of 4 items per stratum
        c = kr.cluster_draws(strata, cluster, 50, np.random.default_rng(0))
        self.assertEqual(c.shape, (50, 24))
        for k in range(6):
            blk = c[:, cluster == k]
            self.assertTrue((blk == blk[:, :1]).all())
        self.assertTrue((c[:, strata == 4].sum(1) == 12).all())
        with self.assertRaises(AssertionError):
            kr.cluster_draws(np.array([4, 5]), np.array([0, 0]), 5, np.random.default_rng(0))

    def test_strat_did(self):
        gap = np.array([1.0, 0.0, 0.0, 0.5, 0.5, 0.0])
        strata = np.array([4, 4, 4, 5, 5, 5])
        lo = np.array([True, False, False, True, True, False])
        hi = ~lo
        r = kr.strat_did(gap, lo, hi, strata, np.ones((3, 6)))
        # stratum 4: D(lo) 1 - D(hi) 0 = 1, weight 1*2/3; stratum 5: .5 - 0 = .5, weight 2*1/3
        self.assertAlmostEqual(r["point"], 0.75)
        self.assertEqual(r["strata_used"], [4, 5])


if __name__ == "__main__":
    unittest.main()
