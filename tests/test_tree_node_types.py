"""--tree-node-types (GFPSNC = GFP-S without the K&K "case" nodes): default paths unchanged against the commit before it
(NODE_TYPES_BASE_REF) on data/student_v3sl items, only the listed node types asked with the same per-problem weight
total, items of other sources untouched, flag validation, and the GFPSNC arm line of hpc/train_arm.sbatch. CPU only,
tiny random Qwen2 with the GPT-2 tokenizer."""
import collections
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
    from test_train_resume import run_inproc
    from test_tree_forcing import ENV, KK, SCRIPT, head_student, head_tree, split_view, tiny
    HAVE = True
except ImportError:
    HAVE = False

V3SL_VAL = ROOT / "data/student_v3sl/val.jsonl"
# The last commit before --tree-node-types.
NODE_TYPES_BASE_REF = "adee35f"
GFPS = ["--tree-full", "plain", "--tree-view", "statements", "--tree-cap", "0", "--tree-sources", "knights_knaves"]
NC = frozenset({"parse", "derive"})
# (mode, view) pairs of the full-tree arms (GF*, TF*, GFPS, GFPQ, TFL, TFA, TFLM, TFLN, TFLP, MIX, GFC)
VIEWS = [("plain", "problem"), ("plain", "statements"), ("plain", "statements+question"), ("true", "problem"),
         ("matched", "problem"), ("placebo", "problem"), ("control", "problem"), ("true", "local"),
         ("matched", "local"), ("plain", "local"), ("true", "ancestors"), ("true", "local+ctxparse"), ("true", "mix")]


def first_items(path: Path, source: str, n: int) -> list[dict]:
    out = []
    if path.exists():
        with open(path) as f:
            for line in f:
                it = json.loads(line)
                if it.get("source") == source:
                    out.append(it)
                    if len(out) == n:
                        break
    return out


def kk_items(n: int) -> list[dict]:
    return first_items(V3SL_VAL, "knights_knaves", n) or [KK]


def qtypes(item: dict) -> dict:
    """Node question -> the set of node types asking it."""
    out = collections.defaultdict(set)
    for sq in item.get("subqs", []):
        if "id" in sq:
            out[sq["question"]].add(sq.get("type"))
    return out


def qof(text: str) -> str:
    """The node question of a node view's text."""
    return text.split("\n\nIntermediate question: ")[1].split("\n")[0]


def build(item, mode, view, seed, **kw):
    """build_full_tree's examples as tuples, or the exception it raises (some views do not exist for some sources)."""
    from cotdistill.student import build_full_tree
    try:
        return [dataclasses.astuple(e) for e in build_full_tree(item, mode=mode, view=view, rng=random.Random(seed),
                                                                 **kw)]
    except ValueError as e:
        return repr(e)


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestDefaultUnchanged(unittest.TestCase):
    def test_library_matches_base(self):
        """Without --tree-node-types (node_types None) and with every node type listed: build_full_tree gives exactly
        NODE_TYPES_BASE_REF's examples (text, labels, target, weight, kind) and stats, for every full-tree mode and
        view, on v3sl K&K items and items of other sources, with and without --tree-sources."""
        with tempfile.TemporaryDirectory() as d:
            H = head_student(d, NODE_TYPES_BASE_REF, "cotdistill_ntbase")
            if H is None:
                self.skipTest("not a git checkout")
            items = ([KK] + kk_items(2) + first_items(V3SL_VAL, "justlogic", 1) + first_items(V3SL_VAL, "sharc", 1))
            n = 0
            for it in items:
                every = {sq.get("type") for sq in it.get("subqs", [])}
                for mode, view in VIEWS:
                    for cap in (0, 10):
                        for sources in (None, {"knights_knaves"}):
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
                                self.assertEqual(build(it, mode, view, seed, node_types=None, **kw), a, key)
                                if mode != "control":
                                    self.assertEqual(build(it, mode, view, seed, node_types=every, **kw), a, key)
                                n += 1
            self.assertGreater(n, 200)

    def test_training_matches_base(self):
        """GFPS and TFL through main() without the flag: the same history, evaluations and predictions as
        NODE_TYPES_BASE_REF's script, on v3sl items (K&K and other families) when built, else the fixture."""
        with tempfile.TemporaryDirectory() as d:
            if head_tree(d + "/h", NODE_TYPES_BASE_REF) is None:
                self.skipTest("not a git checkout")
            tiny(d)
            items = (first_items(V3SL_VAL, "knights_knaves", 2) + first_items(V3SL_VAL, "justlogic", 2)
                     or [dict(KK, item_id=f"k{j}") for j in range(3)])
            Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in items))
            arms = {"GFPS": GFPS,
                    "TFL": ["--tree-full", "true", "--tree-view", "local", "--tree-cap", "0", "--tree-sources",
                            "knights_knaves"]}
            runs = {}
            for name, script in (("base", d + "/h/scripts/train_student.py"), ("now", SCRIPT)):
                for arm, extra in arms.items():
                    out = f"{d}/{name}-{arm}"
                    r = subprocess.run([sys.executable, script, "--model", d + "/m", "--train", d + "/t.jsonl",
                                        "--eval", "v=" + d + "/t.jsonl", "--epochs", "1", "--items-per-update", "2",
                                        "--micro-bs", "4", "--max-len", "768", "--precision", "bf16", "--no-grad-ckpt",
                                        "--log-every", "1", "--seed", "3", "--permute-final", "0.5", "--out", out,
                                        *extra], capture_output=True, text=True, timeout=900, env=ENV)
                    self.assertEqual(r.returncode, 0, (name, arm, r.stderr[-2000:]))
                    m = json.loads(Path(out + "/metrics.json").read_text())
                    hist = [{k: v for k, v in h.items() if k not in ("s_per_step", "approx_tok_per_s")}
                            for h in m["history"]]
                    runs[name, arm] = (hist, m["eval"], m.get("tree_view_stats"),
                                       Path(out + "/preds_v.jsonl").read_text())
            for arm in arms:
                self.assertEqual(runs["base", arm], runs["now", arm], arm)
                self.assertGreater(len(runs["now", arm][0]), 0)


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestFilter(unittest.TestCase):
    def check_group(self, it, full, filt, aux_total=1.0):
        """filt: the final view of full (same rng draw), then exactly the listed-type nodes of full, each asked once,
        sharing the same per-problem weight total."""
        types = qtypes(it)
        wi = it.get("weight", 1.0)
        fin_a = [e for e in full if e[4] == "final"]
        fin_b = [e for e in filt if e[4] == "final"]
        self.assertEqual(fin_a, fin_b)
        node_a = [e for e in full if e[4] == "subq"]
        node_b = [e for e in filt if e[4] == "subq"]
        qa = [qof(e[0]) for e in node_a]
        qb = [qof(e[0]) for e in node_b]
        self.assertTrue(node_b)
        for q in qb:
            self.assertTrue(types[q] <= NC, (q, types[q]))
        self.assertEqual(sorted(qb), sorted(q for q in qa if types[q] <= NC))
        self.assertLess(len(qb), len(qa))
        self.assertAlmostEqual(sum(e[3] for e in node_b), aux_total * wi, places=9)
        self.assertAlmostEqual(sum(e[3] for e in node_b), sum(e[3] for e in node_a), places=9)
        for e in node_b:
            self.assertAlmostEqual(e[3], aux_total * wi / len(node_b), places=12)

    def test_only_listed_types_same_total(self):
        """GFPS / GFP / GFPQ / TFL / TFLN / TFA / MIX views of v3sl K&K items with node_types {parse, derive}: no case
        node is asked, every parse / derive node is, and the per-problem weight total is unchanged (aux 1 and 2)."""
        n = 0
        for it in kk_items(4):
            n_case = sum(sq.get("type") == "case" for sq in it["subqs"])
            self.assertGreater(n_case, 0)
            for mode, view in VIEWS:
                if mode in ("control", "placebo", "matched"):
                    continue
                for aux in (1.0, 2.0):
                    for seed in range(2):
                        kw = dict(cap=0, sources={"knights_knaves"}, aux_total=aux)
                        st = {}
                        full = build(it, mode, view, seed, **kw)
                        filt = build(it, mode, view, seed, node_types=NC, stats=st, **kw)
                        self.assertIsInstance(filt, list, filt)
                        self.check_group(it, full, filt, aux)
                        self.assertEqual(st["node_type_dropped"], n_case)
                        n += 1
        self.assertGreater(n, 20)

    def test_cap_samples_kept_nodes(self):
        """With a cap, the cap is drawn from the kept nodes (cap 5 of the parse / derive nodes, never a case node)."""
        for it in kk_items(2):
            types = qtypes(it)
            for seed in range(3):
                ex = build(it, "plain", "statements", seed, cap=5, node_types=NC)
                qs = [qof(e[0]) for e in ex if e[4] == "subq"]
                self.assertEqual(len(qs), 5)
                self.assertTrue(all(types[q] <= NC for q in qs))

    def test_other_sources_untouched(self):
        """Items outside --tree-sources keep their final question only, as without the flag; without --tree-sources
        the filter applies to every source (a node without a type is not asked)."""
        others = first_items(V3SL_VAL, "justlogic", 2) + first_items(V3SL_VAL, "sharc", 1)
        if not others:
            self.skipTest("data/student_v3sl not built")
        for it in others:
            for seed in range(2):
                a = build(it, "plain", "problem", seed, cap=0, sources={"knights_knaves"})
                b = build(it, "plain", "problem", seed, cap=0, sources={"knights_knaves"}, node_types=NC)
                self.assertEqual(a, b)
                self.assertEqual([e[4] for e in b], ["final"])
                c = build(it, "plain", "problem", seed, cap=0, node_types=NC)
                types = qtypes(it)
                for e in c:
                    if e[4] == "subq":
                        self.assertTrue(types[qof(e[0])] <= NC)
        it = dict(KK, subqs=[dict(sq, type=None) if sq.get("type") == "derive" else sq for sq in KK["subqs"]])
        ex = build(it, "plain", "statements", 0, cap=0, node_types=NC)
        self.assertEqual(sum(e[4] == "subq" for e in ex), sum(sq.get("type") == "parse" for sq in KK["subqs"]))

    def test_parents_still_stated(self):
        """TFL: a derive node whose parent is a case node still gets that parent's result stated (depends_on
        untouched); the case node itself is not asked."""
        from cotdistill.student import build_full_tree
        n = 0
        for it in kk_items(3):
            by = {sq["id"]: sq for sq in it["subqs"]}
            case_q = {sq["question"] for sq in it["subqs"] if sq.get("type") == "case"}
            ex = build_full_tree(it, mode="true", view="local", rng=random.Random(0), cap=0, node_types=NC)
            stated = [s for e in ex if e.kind == "subq" for s in split_view(e)[2]]
            asked = {split_view(e)[1] for e in ex if e.kind == "subq"}
            self.assertFalse(asked & case_q)
            if any(by[p].get("type") == "case" for sq in it["subqs"] if sq.get("type") == "derive"
                   for p in sq.get("depends_on") or [] if p in by):
                self.assertTrue(any(s[3:].split(" A: ")[0] in case_q for s in stated))
                n += 1
        self.assertGreater(n, 0)

    def test_control_rejected(self):
        from cotdistill.student import build_full_tree
        with self.assertRaises(ValueError):
            build_full_tree(KK, mode="control", rng=random.Random(0), cap=0, node_types=NC)


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestTrainNodeTypes(unittest.TestCase):
    OTHER = {"item_id": "o", "source": "justlogic", "prompt": "P\nOptions:\nA) a\nB) b", "labels": ["A", "B"],
             "gold_label": "A", "teacher": {"A": 0.8, "B": 0.2},
             "subqs": [{"id": "n1", "question": "q?", "p_cot": 0.9, "depends_on": []}]}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.d = self.tmp.name
        tiny(d)
        items = [dict(KK, item_id=f"k{j}") for j in range(3)] + [dict(self.OTHER, item_id=f"o{j}") for j in range(2)]
        Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in items))

    def tearDown(self):
        self.tmp.cleanup()

    def common(self, out):
        d = self.d
        return ["--model", d + "/m", "--train", d + "/t.jsonl", "--eval", "v=" + d + "/t.jsonl",
                "--items-per-update", "2", "--epochs", "2", "--micro-bs", "4", "--max-len", "256",
                "--precision", "fp32master", "--no-grad-ckpt", "--log-every", "1", "--seed", "2", "--out", f"{d}/{out}"]

    def run_(self, *a, out="o"):
        return subprocess.run([sys.executable, SCRIPT, *self.common(out), *a], capture_output=True, text=True,
                              timeout=900, env=ENV)

    def test_gfpsnc_updates(self):
        """Through main(): GFPSNC's updates hold GFPS's problems and final questions, only parse / derive node views
        (each K&K problem's node weights summing to 1, as in GFPS), packed or not (--tree-pack)."""
        types = qtypes(KK)
        a = run_inproc([*self.common("gfps"), *GFPS])
        b = run_inproc([*self.common("nc"), *GFPS, "--tree-node-types", "parse,derive"])
        c = run_inproc([*self.common("ncp"), *GFPS, "--tree-node-types", "parse,derive", "--tree-pack", "4"])
        self.assertEqual(len(a), len(b))
        self.assertEqual(len(a), len(c))
        for (ma, _), (mb, _), (mc, _) in zip(a, b, c):
            ea, eb, ec = ([x for m in u for x in m] for u in (ma, mb, mc))
            self.assertEqual(sorted(x for x in ea if x[3] == "final"), sorted(x for x in eb if x[3] == "final"))
            self.assertEqual(sorted(eb), sorted(ec))
            na, nb = ([x for x in e if x[3] == "subq"] for e in (ea, eb))
            self.assertAlmostEqual(sum(x[1] for x in na), sum(x[1] for x in nb), places=9)
            for x in nb:
                self.assertTrue(types[qof(x[0])] <= NC, x[0])
            self.assertEqual(len(nb), len(na) // 21 * 12)       # KK: 4 parse + 8 derive of 21 nodes
        self.assertLess(sum(len(m) for m, _ in c), sum(len(m) for m, _ in b))   # packs: fewer forward passes

    def test_stats_and_flags(self):
        r = self.run_(*GFPS, "--tree-node-types", "parse,derive")
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        m = json.loads(Path(self.d + "/o/metrics.json").read_text())
        self.assertEqual(m["args"]["tree_node_types"], "parse,derive")
        st = m["tree_view_stats"]
        self.assertEqual(st["views"], 3 * 12)
        self.assertEqual(st["node_type_dropped"], 3 * 9)
        for bad, msg in ((["--tree-node-types", "parse,derive"], "need --tree-full"),
                         (["--tree-full", "control", "--tree-node-types", "parse"], "control questions have no type"),
                         ([*GFPS, "--tree-node-types", ","], "a comma list of node types"),
                         ([*GFPS, "--tree-node-types", "parse,derived"], "no node of type(s) derived"),
                         ([*GFPS[:-2], "--tree-sources", "justlogic", "--tree-node-types", "parse"],
                          "no node of type(s) parse")):
            r = self.run_(*bad, out="bad")
            self.assertNotEqual(r.returncode, 0, bad)
            self.assertIn(msg, r.stderr, bad)


class TestArmLine(unittest.TestCase):
    def test_gfpsnc_is_gfps_without_case_nodes(self):
        sb = ROOT / "hpc/train_arm.sbatch"
        cmd = ('source <(sed -n "/^arm_config()/,/^}/p" "$0"); for a in GFPS GFPSNC; do arm_config "$a"; '
               'echo "$F|$S|$T|$RLM|$DS|$DX"; done')
        out = subprocess.run(["bash", "-c", cmd, str(sb)], capture_output=True, text=True, check=True).stdout.split("\n")
        self.assertEqual(out[1], out[0] + " --tree-node-types parse,derive")
        lines = sb.read_text().split("\n")
        i = next(j for j, l in enumerate(lines) if l.lstrip().startswith("GFPQ)"))
        self.assertTrue(lines[i + 1].lstrip().startswith("GFPSNC)"))


if __name__ == "__main__":
    unittest.main()
