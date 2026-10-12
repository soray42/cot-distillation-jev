"""--tree-target shuffled (GFPSR / GFPQR: the random-label control of GFPS / GFPQ): every node keeps its view, question,
option order and weight, the final question is untouched, only the node targets are replaced by
student.shuffled_targets (each problem's P(Yes) values permuted among its nodes, keyed by the seed and the item, never
the training rng; an MC node: a uniformly drawn option). The default paths unchanged against the commit before
(TARGET_BASE_REF) on real data/student_numgen_tf, data/student_gsm8k_tf and data/student_v3sl items (plus kk_tf /
pronto_tf / lean items when built); flag validation; the GFPSR / GFPQR arm lines of hpc/train_arm.sbatch. CPU only, tiny
random Qwen2 with the GPT-2 tokenizer."""
import copy
import dataclasses
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    from test_tree_forcing import ENV, KK, SCRIPT, head_student, head_tree, tiny
    HAVE = True
except ImportError:
    HAVE = False

# The last commit before --tree-target.
TARGET_BASE_REF = "fd0e1ba"
DATA = ROOT / "data"
V3SL_VAL = DATA / "student_v3sl/val.jsonl"
NUMGEN_VAL = DATA / "student_numgen_tf/val.jsonl"
GSM_VAL = DATA / "student_gsm8k_tf/val.jsonl"
GFPS = ["--tree-full", "plain", "--tree-view", "statements", "--tree-cap", "0", "--tree-sources", "knights_knaves"]
GFPQ = ["--tree-full", "plain", "--tree-view", "statements+question", "--tree-cap", "0", "--tree-sources",
        "knights_knaves"]
SHUF = ["--tree-target", "shuffled"]
VIEWS = [("plain", "problem"), ("plain", "statements"), ("plain", "statements+question"), ("true", "problem"),
         ("matched", "problem"), ("placebo", "problem"), ("control", "problem"), ("true", "local"),
         ("matched", "local"), ("plain", "local"), ("true", "ancestors"), ("true", "local+ctxparse"), ("true", "mix")]
PLAIN_VIEWS = ("problem", "statements", "statements+question")


def first_items(path: Path, source: str, n: int, tag: str = "") -> list[dict]:
    """The first n items of a source (whose item_id contains tag) in a data file; [] when it is not built."""
    out = []
    if path.exists():
        with open(path) as f:
            for line in f:
                it = json.loads(line)
                if it.get("source") == source and tag in it["item_id"]:
                    out.append(it)
                    if len(out) == n:
                        break
    return out


def real_items() -> list[dict]:
    """Items of every single-domain Tree Forcing source (those whose data is built; numgen / pronto / lean at depth 1
    and 4) and of the v3sl mixture (K&K and JustLogic), plus the K&K fixture."""
    return ([KK] + first_items(NUMGEN_VAL, "numgen", 2) + first_items(NUMGEN_VAL, "numgen", 1, "-d4-") +
            first_items(GSM_VAL, "gsm8k_tf", 2) + first_items(V3SL_VAL, "knights_knaves", 2) +
            first_items(V3SL_VAL, "justlogic", 1) + first_items(DATA / "student_kk_tf/val.jsonl", "knights_knaves", 1) +
            first_items(DATA / "student_pronto_tf/val.jsonl", "pronto", 1, "-d4-") +
            first_items(DATA / "student_lean_tf_pilot/val.jsonl", "lean_f1", 1, "-d4-"))


def build(item, mode, view, seed, **kw):
    """build_full_tree's examples as tuples, or the exception it raises (some views do not exist for some sources)."""
    from cotdistill.student import build_full_tree
    try:
        return [dataclasses.astuple(e) for e in build_full_tree(item, mode=mode, view=view, rng=random.Random(seed),
                                                                 **kw)]
    except ValueError as e:
        return repr(e)


def p_yes_t(e) -> float:
    """P(Yes) of a yes/no node view given as a tuple (text, labels, target, ...); the yes/no order is drawn per view."""
    return e[2][0] if e[0].endswith("Options:\nA) Yes\nB) No\nAnswer:") else e[2][1]


def run_child(argv: list[str], tmp: str) -> list:
    """test_train_resume.run_inproc(argv) (every update's micro-batches: text, weight, target, kind of each example read
    out) in a child process, so the suite's own process does not hold the training runs' memory."""
    res = Path(tmp) / "events.json"
    code = ("import json, sys; sys.path[:0] = [sys.argv[1] + '/src', sys.argv[1] + '/tests']; "
            "from test_train_resume import run_inproc; "
            "json.dump(run_inproc(json.loads(sys.argv[2])), open(sys.argv[3], 'w'))")
    r = subprocess.run([sys.executable, "-c", code, str(ROOT), json.dumps(argv), str(res)], capture_output=True,
                       text=True, timeout=900, env=ENV)
    if r.returncode:
        raise AssertionError(r.stderr[-2000:])
    return [([[(x[0], x[1], tuple(x[2]), x[3]) for x in m] for m in mbs], g) for mbs, g in json.loads(res.read_text())]


def with_mc(item: dict) -> dict:
    """item plus two multiple-choice nodes (their own options and answer index; no parents)."""
    it = copy.deepcopy(item)
    it["subqs"] = it["subqs"] + [
        {"id": "mc1", "question": "Which of these is the first speaker?", "options": ["Alice", "Bob", "Carol", "Dave"],
         "answer": 2, "depends_on": []},
        {"id": "mc2", "question": "How many inhabitants are there?", "options": ["2", "3", "4"], "answer": 0,
         "depends_on": []}]
    return it


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestDefaultUnchanged(unittest.TestCase):
    def test_library_matches_base(self):
        """Without --tree-target (target_seed None): build_full_tree gives exactly TARGET_BASE_REF's examples (text,
        labels, target, weight, kind) and stats, for every full-tree mode and view, cap and source filter, on real
        numgen_tf, gsm8k_tf and v3sl items (K&K and another family) and the K&K fixture."""
        with tempfile.TemporaryDirectory() as d:
            H = head_student(d, TARGET_BASE_REF, "cotdistill_ttbase")
            if H is None:
                self.skipTest("not a git checkout")
            items = real_items()
            self.assertTrue(any(it.get("source") == "numgen" for it in items) or not NUMGEN_VAL.exists())
            n = 0
            for it in items:
                for mode, view in VIEWS:
                    for cap in (0, 10):
                        for sources in (None, {it.get("source")}):
                            for seed in range(2):
                                kw = dict(cap=cap, sources=sources, permute_final=0.5, subq_target="cot")
                                sa, sb = {}, {}
                                try:
                                    a = [dataclasses.astuple(e) for e in H.build_full_tree(
                                        it, mode=mode, view=view, rng=random.Random(seed), stats=sa, **kw)]
                                except ValueError as e:
                                    a = repr(e)
                                b = build(it, mode, view, seed, stats=sb, **kw)
                                key = (it["item_id"], mode, view, cap, sources, seed)
                                self.assertEqual(a, b, key)
                                self.assertEqual(sa, sb, key)
                                self.assertEqual(build(it, mode, view, seed, target_seed=None, **kw), a, key)
                                n += 1
            self.assertGreater(n, 400)

    def test_training_matches_base(self):
        """GFPS on numgen_tf items, GFPQ on gsm8k_tf items and GFPS on v3sl items through main() without the flag: the
        same history, evaluations, tree view stats and predictions as TARGET_BASE_REF's script (the fixture when the
        data is not built)."""
        with tempfile.TemporaryDirectory() as d:
            if head_tree(d + "/h", TARGET_BASE_REF) is None:
                self.skipTest("not a git checkout")
            tiny(d)
            fixture = [dict(KK, item_id=f"k{j}") for j in range(3)]
            sets = {"numgen": (first_items(NUMGEN_VAL, "numgen", 3), [*GFPS[:-1], "numgen"]),
                    "gsm": (first_items(GSM_VAL, "gsm8k_tf", 3), [*GFPQ[:-1], "gsm8k_tf"]),
                    "v3sl": (first_items(V3SL_VAL, "knights_knaves", 2) + first_items(V3SL_VAL, "justlogic", 1)
                             or fixture, GFPS)}
            runs, n = {}, 0
            for name, (items, extra) in sets.items():
                if not items:
                    continue
                n += 1
                Path(f"{d}/{name}.jsonl").write_text("\n".join(json.dumps(x) for x in items))
                for which, script in (("base", d + "/h/scripts/train_student.py"), ("now", SCRIPT)):
                    out = f"{d}/{which}-{name}"
                    r = subprocess.run([sys.executable, script, "--model", d + "/m", "--train", f"{d}/{name}.jsonl",
                                        "--eval", f"v={d}/{name}.jsonl", "--epochs", "2", "--items-per-update", "2",
                                        "--micro-bs", "4", "--max-len", "768", "--precision", "bf16", "--no-grad-ckpt",
                                        "--log-every", "1", "--seed", "3", "--permute-final", "0.5", "--out", out,
                                        *extra], capture_output=True, text=True, timeout=900, env=ENV)
                    self.assertEqual(r.returncode, 0, (which, name, r.stderr[-2000:]))
                    m = json.loads(Path(out + "/metrics.json").read_text())
                    hist = [{k: v for k, v in h.items() if k not in ("s_per_step", "approx_tok_per_s")}
                            for h in m["history"]]
                    runs[which, name] = (hist, m["eval"], m.get("tree_view_stats"),
                                         Path(out + "/preds_v.jsonl").read_text())
                    if which == "now":
                        self.assertEqual(m["args"]["tree_target"], "real")
                self.assertEqual(runs["base", name], runs["now", name], name)
                self.assertGreater(len(runs["now", name][0]), 0)
            self.assertGreater(n, 0)


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestShuffled(unittest.TestCase):
    def pairs(self, items=None, seeds=range(2), **kw):
        """(item, view, seed, real examples, shuffled examples, shuffled stats) for the plain views each item has."""
        for it in items or real_items():
            for view in PLAIN_VIEWS:
                for seed in seeds:
                    kw2 = dict(cap=0, sources={it.get("source")}, permute_final=0.5, **kw)
                    a = build(it, "plain", view, seed, **kw2)
                    if isinstance(a, str):          # e.g. statement_block on a gsm8k_tf item
                        continue
                    st = {}
                    b = build(it, "plain", view, seed, target_seed=7, stats=st, **kw2)
                    yield it, view, seed, a, b, st

    def test_only_node_targets_change(self):
        """Same examples in the same order (text: question, view and yes/no order; labels; weight; kind; gold), the
        final question with its target unchanged (permute_final 0.5 included: rng lockstep), every node view's target
        replaced by the permuted one: per item the same multiset of P(Yes) (the yes rate and the soft values), about
        half of the hard answers changed."""
        n = agree = total = 0
        sources = set()
        for it, view, seed, a, b, st in self.pairs():
            key = (it["item_id"], view, seed)
            self.assertEqual(len(a), len(b), key)
            for x, y in zip(a, b):
                self.assertEqual(x[:2] + x[3:], y[:2] + y[3:], key)
                if x[4] == "final":
                    self.assertEqual(x[2], y[2], key)
            self.assertEqual(a[0][4], "final")
            na = [x for x in a if x[4] == "subq"]
            nb = [x for x in b if x[4] == "subq"]
            self.assertTrue(nb, key)
            self.assertEqual(sorted(map(p_yes_t, na)), sorted(map(p_yes_t, nb)), key)
            for x, y in zip(na, nb):
                self.assertAlmostEqual(sum(y[2]), 1.0, places=9)
                agree += (p_yes_t(x) >= 0.5) == (p_yes_t(y) >= 0.5)
                total += 1
            self.assertEqual(st["shuffled_views"], len(nb))
            self.assertEqual(st["shuffled_yes"], st["real_yes"])
            self.assertEqual(st["shuffled_agree"], sum((p_yes_t(x) >= 0.5) == (p_yes_t(y) >= 0.5)
                                                       for x, y in zip(na, nb)))
            sources.add(it.get("source"))
            n += 1
        self.assertGreater(n, 10)
        self.assertGreater(total - agree, 0.3 * total)        # targets really changed
        self.assertLess(total - agree, 0.7 * total)           # and not flipped wholesale (that would be informative)
        if NUMGEN_VAL.exists() and GSM_VAL.exists():
            self.assertTrue({"numgen", "gsm8k_tf", "knights_knaves"} <= sources, sources)

    def test_seeded_not_training_rng(self):
        """A node's random target depends on the target seed and the item only: the same with another training rng
        (other yes/no orders), with a cap (another node subset), in another epoch (no epoch in the key) and in another
        process; another target seed gives another labelling."""
        from cotdistill.student import _soft
        from cotdistill.student import shuffled_targets
        for it in real_items():
            src = {it.get("source")}
            view = "statements" if not isinstance(build(it, "plain", "statements", 0, cap=0), str) else \
                "statements+question"
            maps = []
            for rseed in range(3):
                ex = build(it, "plain", view, rseed, cap=0, sources=src, target_seed=5)
                maps.append([p_yes_t(e) for e in ex if e[4] == "subq"])
            self.assertEqual(maps[0], maps[1], it["item_id"])
            self.assertEqual(maps[0], maps[2], it["item_id"])
            m5 = shuffled_targets(it, 5)
            nodes = [sq for sq in it["subqs"] if "id" in sq]
            self.assertEqual(maps[0], [_soft(m5[sq["id"]], True)[0] for sq in nodes if sq["id"] in m5])
            self.assertEqual(shuffled_targets(it, 5), m5)
            by_q = {}
            for sq in nodes:
                by_q.setdefault(sq["question"], set()).add(_soft(m5[sq["id"]], True)[0])
            for rseed in range(3):                  # cap 5: a random subset, each node with its own random target
                ex = build(it, "plain", view, rseed, cap=5, sources=src, target_seed=5)
                for e in ex:
                    if e[4] == "subq":
                        q = e[0].split("\n\nIntermediate question: ")[1].split("\n")[0]
                        self.assertIn(p_yes_t(e), by_q[q])
        code = ("import json, sys; sys.path.insert(0, 'src'); from cotdistill.student import shuffled_targets; "
                "it = json.load(open('tests/fixtures/kk_local_item.json')); print(json.dumps(shuffled_targets(it, 5), "
                "sort_keys=True))")
        env = {**ENV, "PYTHONHASHSEED": "123"}
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, env=env, check=True)
        self.assertEqual(json.loads(out.stdout), shuffled_targets(KK, 5))
        diff = sum(shuffled_targets(it, 5) != shuffled_targets(it, 6) for it in real_items())
        self.assertGreater(diff, 0)

    def test_balanced_over_many_items(self):
        """Over the first 300 items of numgen_tf, gsm8k_tf and the v3sl K&K items (val): the shuffled yes rate equals
        the real one exactly (per item, so in all), and the share of nodes whose random answer is the real one is near
        the chance level of a permutation, sum over items of n_i (r_i^2 + (1 - r_i)^2) / sum n_i (r_i: the item's yes
        rate; 0.5 on balanced trees)."""
        from cotdistill.student import _node_p, shuffled_targets
        n_sets = 0
        for path, src in ((NUMGEN_VAL, "numgen"), (GSM_VAL, "gsm8k_tf"), (V3SL_VAL, "knights_knaves")):
            items = first_items(path, src, 300)
            if not items:
                continue
            n_sets += 1
            real, shuf, chance = [], [], 0.0
            for it in items:
                m = shuffled_targets(it, 0)
                ri = [_node_p(sq) >= 0.5 for sq in it["subqs"] if "id" in sq and _node_p(sq) is not None]
                si = [m[sq["id"]] >= 0.5 for sq in it["subqs"] if "id" in sq and _node_p(sq) is not None]
                self.assertEqual(sum(ri), sum(si), it["item_id"])
                if ri:
                    r = sum(ri) / len(ri)
                    chance += len(ri) * (r * r + (1 - r) * (1 - r))
                real += ri
                shuf += si
            agree = sum(x == y for x, y in zip(real, shuf)) / len(real)
            self.assertAlmostEqual(agree, chance / len(real), delta=0.05, msg=src)
            self.assertLess(agree, 0.65, src)
        self.assertGreater(n_sets, 0)

    def test_mc_nodes_get_a_random_option(self):
        """Multiple-choice nodes: the same question and options, a one-hot target on the drawn option (uniform over the
        options across seeds, seeded), the yes/no nodes as above, the training rng untouched."""
        it = with_mc(KK)
        for view in ("statements", "problem"):
            a = build(it, "plain", view, 0, cap=0)
            seen = {"mc1": set(), "mc2": set()}
            for ts in range(40):
                st = {}
                b = build(it, "plain", view, 0, cap=0, target_seed=ts, stats=st)
                self.assertEqual(b, build(it, "plain", view, 0, cap=0, target_seed=ts))
                self.assertEqual(len(a), len(b))
                mcs = 0
                for x, y in zip(a, b):
                    self.assertEqual(x[:2] + x[3:], y[:2] + y[3:])
                    if x[4] == "subq" and len(x[1]) > 2:    # an MC node (its own options)
                        self.assertEqual(sorted(y[2]), [0.0] * (len(y[2]) - 1) + [1.0])
                        k = "mc1" if len(x[1]) == 4 else "mc2"
                        seen[k].add(y[2].index(1.0))
                        mcs += 1
                self.assertEqual(mcs, 2)
                self.assertEqual(st["shuffled_views"], sum(x[4] == "subq" for x in b))
            self.assertEqual(seen, {"mc1": {0, 1, 2, 3}, "mc2": {0, 1, 2}})

    def test_refused_outside_plain_views(self):
        from cotdistill.student import build_full_tree
        bad = [dict(mode="true"), dict(mode="matched"), dict(mode="placebo"), dict(mode="control"),
               dict(mode="plain", view="local"), dict(mode="true", view="local"), dict(mode="plain", node_format="mc"),
               dict(mode="plain", eligible_only=True), dict(mode="plain", view="statements", scaffold_p=0.5)]
        for kw in bad:
            with self.assertRaises(ValueError, msg=kw):
                build_full_tree(KK, rng=random.Random(0), cap=0, target_seed=1, **kw)


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestTrainShuffled(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.d = self.tmp.name
        tiny(d)

    def tearDown(self):
        self.tmp.cleanup()

    def data(self, name, items):
        Path(f"{self.d}/{name}.jsonl").write_text("\n".join(json.dumps(x) for x in items))
        return f"{self.d}/{name}.jsonl"

    def common(self, train, out):
        d = self.d
        return ["--model", d + "/m", "--train", train, "--eval", "v=" + train, "--items-per-update", "2",
                "--epochs", "2", "--micro-bs", "2", "--max-len", "256", "--precision", "fp32master", "--no-grad-ckpt",
                "--log-every", "1", "--seed", "2", "--permute-final", "0.5", "--eval-bs", "1", "--eval-max-len", "512",
                "--out", f"{d}/{out}"]

    def check_updates(self, a, b):
        """b (shuffled) reads exactly a's examples in every update (text, weight, kind; final targets too); node targets
        are a permutation within each problem, the same in both epochs (2 epochs, as many updates each), and some
        differ from a's."""
        self.assertEqual(len(a), len(b))
        self.assertEqual(len(b) % 2, 0)
        changed, per_epoch = 0, [{}, {}]
        for u, ((ma, _), (mb, _)) in enumerate(zip(a, b)):
            ea, eb = [x for m in ma for x in m], [x for m in mb for x in m]
            self.assertEqual([(x[0], x[1], x[3]) for x in ea], [(x[0], x[1], x[3]) for x in eb])
            ya, yb = {}, {}
            for x, y in zip(ea, eb):
                if x[3] == "final":
                    self.assertEqual(x[2], y[2])
                    continue
                pa, pb = p_yes_t((x[0], None, x[2])), p_yes_t((y[0], None, y[2]))
                ctx, q = x[0].split("\n\nIntermediate question: ")
                ya.setdefault(ctx, []).append(pa)
                yb.setdefault(ctx, []).append(pb)
                changed += (pa >= 0.5) != (pb >= 0.5)
                per_epoch[2 * u // len(b)].setdefault((ctx, q.split("\n")[0]), []).append(pb)
            for k in ya:                            # per problem (its context): the same multiset of P(Yes)
                self.assertEqual(sorted(ya[k]), sorted(yb[k]))
        self.assertGreater(changed, 0)
        self.assertEqual({k: sorted(v) for k, v in per_epoch[0].items()},   # a node keeps its random target
                         {k: sorted(v) for k, v in per_epoch[1].items()})

    def test_gfpsr_gfpqr_updates(self):
        """Through main(): GFPSR reads GFPS's updates with permuted node targets (numgen_tf items, or the K&K fixture),
        GFPQR GFPQ's (gsm8k_tf items); GFPSR packed (--tree-pack 4) reads the same examples per update as unpacked."""
        ng = first_items(NUMGEN_VAL, "numgen", 4)
        gsm = first_items(GSM_VAL, "gsm8k_tf", 4)
        cases = [("ng", ng, [*GFPS[:-1], "numgen"]) if ng else
                 ("kk", [dict(KK, item_id=f"k{j}") for j in range(3)], GFPS)]
        if gsm:
            cases.append(("gsm", gsm, [*GFPQ[:-1], "gsm8k_tf"]))
        for name, items, arm in cases:
            t = self.data(name, items)
            a = run_child([*self.common(t, name + "a"), *arm], self.d)
            b = run_child([*self.common(t, name + "b"), *arm, *SHUF], self.d)
            self.check_updates(a, b)
            if "statements" in arm:
                c = run_child([*self.common(t, name + "c"), *arm, *SHUF, "--tree-pack", "4"], self.d)
                self.assertEqual(len(b), len(c))
                for (mb, _), (mc, _) in zip(b, c):
                    self.assertEqual(sorted(x for m in mb for x in m), sorted(x for m in mc for x in m))

    def test_stats_args_and_flags(self):
        t = self.data("kk", [dict(KK, item_id=f"k{j}") for j in range(3)])
        r = subprocess.run([sys.executable, SCRIPT, *self.common(t, "o"), *GFPS, *SHUF], capture_output=True,
                           text=True, timeout=900, env=ENV)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        m = json.loads(Path(self.d + "/o/metrics.json").read_text())
        self.assertEqual(m["args"]["tree_target"], "shuffled")
        st = m["tree_view_stats"]
        self.assertEqual(st["shuffled_views"], st["views"])
        self.assertEqual(st["shuffled_yes"], st["real_yes"])
        self.assertLess(st["shuffled_agree"], st["views"])
        for bad, msg in ((SHUF, "need --tree-full"),
                         (["--tree-full", "true", *SHUF], "--tree-target shuffled: for --tree-full plain"),
                         (["--tree-full", "control", *SHUF], "--tree-target shuffled: for --tree-full plain"),
                         (["--tree-full", "plain", "--tree-view", "local", *SHUF], "for --tree-full plain"),
                         (["--tree-full", "plain", "--node-format", "mc", *SHUF], "with yes/no nodes"),
                         (["--tree-full", "plain", "--tree-eligible-only", *SHUF], "without --tree-eligible-only"),
                         ([*GFPS, "--tree-scaffold", "0.5", *SHUF], "or --tree-scaffold")):
            r = subprocess.run([sys.executable, SCRIPT, *self.common(t, "bad"), *bad], capture_output=True, text=True,
                               timeout=900, env=ENV)
            self.assertNotEqual(r.returncode, 0, bad)
            self.assertIn(msg, r.stderr, bad)


class TestArmLines(unittest.TestCase):
    def test_gfpsr_gfpqr_are_gfps_gfpq_with_shuffled_targets(self):
        sb = ROOT / "hpc/train_arm.sbatch"
        cmd = ('source <(sed -n "/^arm_config()/,/^}/p" "$0"); for a in GFPS GFPSR GFPQ GFPQR; do arm_config "$a"; '
               'echo "$F|$S|$T|$RLM|$DS|$DX"; done')
        out = subprocess.run(["bash", "-c", cmd, str(sb)], capture_output=True, text=True, check=True).stdout.split("\n")
        self.assertEqual(out[1], out[0] + " --tree-target shuffled")
        self.assertEqual(out[3], out[2] + " --tree-target shuffled")
        lines = sb.read_text().split("\n")
        i = next(j for j, l in enumerate(lines) if l.lstrip().startswith("GFPSP)"))
        self.assertTrue(lines[i + 1].lstrip().startswith("GFPSR)"))
        self.assertTrue(lines[i + 2].lstrip().startswith("GFPQR)"))


if __name__ == "__main__":
    unittest.main()
