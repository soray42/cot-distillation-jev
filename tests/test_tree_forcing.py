"""Tree Forcing views (--tree-view local / ancestors, --tree-cap 0, --tree-sources, --tree-node-weight, --eval-local):
default behaviour identical to HEAD, rendering, filters and the matched-local control. CPU only."""
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

try:
    import torch  # noqa: F401
    import transformers
    HAVE = True
except ImportError:
    HAVE = False

KK = json.loads((ROOT / "tests/fixtures/kk_local_item.json").read_text())      # one data/student_v3sl K&K item
SCRIPT = str(ROOT / "scripts/train_student.py")
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}


# The pre-Tree-Forcing commit the default path is compared with. Pinned, not HEAD: once the Tree Forcing code is
# committed, HEAD would compare the new code with itself.
BASE_REF = os.environ.get("TF_BASE_REF", "6a4a771")


def head_tree(dest: str) -> Path | None:
    """src/ and scripts/ of the pre-Tree-Forcing commit BASE_REF under dest (None outside a git checkout)."""
    try:
        blob = subprocess.run(["git", "-C", str(ROOT), "archive", BASE_REF, "src/cotdistill", "scripts/train_student.py"],
                              capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    Path(dest).mkdir(parents=True, exist_ok=True)
    p = Path(dest) / "head.tar"
    p.write_bytes(blob)
    with tarfile.open(p) as t:
        t.extractall(dest, filter="data")
    return Path(dest)


def head_student(dest: str):
    """HEAD's cotdistill.student, imported as a separate package (cotdistill_head)."""
    root = head_tree(dest)
    if root is None:
        return None
    (root / "src/cotdistill").rename(root / "src/cotdistill_head")
    sys.path.insert(0, str(root / "src"))
    import importlib
    return importlib.import_module("cotdistill_head.student")


def tiny(d: str) -> None:
    tok = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    tok.pad_token = tok.eos_token
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                                   tie_word_embeddings=True)
    torch.manual_seed(0)
    transformers.Qwen2ForCausalLM(cfg).save_pretrained(d + "/m")
    tok.save_pretrained(d + "/m")


def split_view(e):
    head, _, rest = e.text.partition("\n\nIntermediate question: ")
    q = rest.split("\n")[0]
    stated = [l for l in head.split("\n") if l.startswith("Q: ")]
    return head, q, stated


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestDefaultUnchanged(unittest.TestCase):
    TOY = {"item_id": "x", "prompt": "P\nOptions:\nA) a\nB) b\nC) c", "labels": ["A", "B", "C"], "gold_label": "A",
           "teacher": {"A": 0.7, "B": 0.2, "C": 0.1},
           "subqs": [{"id": "n1", "question": "q1?", "p_cot": 0.9, "depends_on": []},
                     {"id": "n2", "question": "q2?", "p_cot": 0.1, "depends_on": []},
                     {"id": "n3", "question": "q3?", "p_cot": 0.9, "depends_on": ["n1", "n2"]},
                     {"id": "n4", "question": "q4?", "p_cot": 0.8, "depends_on": []},
                     {"id": "n5", "question": "q5?", "p_cot": 0.2, "depends_on": ["n3"]},
                     {"id": "n6", "question": "q6?", "p_cot": 0.9, "depends_on": ["n4"]},
                     {"id": "n7", "question": "q7?", "p_cot": 0.95, "depends_on": ["n5"]}],
           "random_subqs": [{"question": f"c{i}?", "p_fresh": 0.3} for i in range(4)]}

    def test_build_full_tree_matches_head(self):
        from cotdistill.student import build_full_tree
        with tempfile.TemporaryDirectory() as d:
            head = head_student(d)
            if head is None:
                self.skipTest("not a git checkout")
            n = 0
            for item in (self.TOY, KK):
                for mode in ("true", "matched", "plain", "placebo", "control"):
                    for elig in (False, True):
                        for fmt in ("yesno", "mc"):
                            for cap in (3, 10):
                                if elig and mode == "control":
                                    continue
                                for seed in range(4):
                                    kw = dict(mode=mode, cap=cap, eligible_only=elig, node_format=fmt,
                                              permute_final=0.5, subq_target="cot")
                                    a = head.build_full_tree(item, rng=random.Random(seed), **kw)
                                    b = build_full_tree(item, rng=random.Random(seed), **kw)
                                    self.assertEqual([dataclasses.astuple(e) for e in a],
                                                     [dataclasses.astuple(e) for e in b], (mode, elig, fmt, cap, seed))
                                    n += 1
            self.assertGreater(n, 100)

    def test_training_matches_head(self):
        """The full-tree loop with default tree options: the same losses as HEAD's script, update for update."""
        with tempfile.TemporaryDirectory() as d:
            if head_tree(d + "/h") is None:
                self.skipTest("not a git checkout")
            tiny(d)
            items = [dict(KK, item_id=f"k{j}") for j in range(3)] + [dict(self.TOY, item_id=f"t{j}") for j in range(3)]
            Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in items))
            hist = {}
            for name, script in (("head", d + "/h/scripts/train_student.py"), ("now", SCRIPT)):
                for mode in ("true", "matched"):
                    out = subprocess.run([sys.executable, script, "--model", d + "/m", "--train", d + "/t.jsonl",
                                          "--eval", "v=" + d + "/t.jsonl", "--tree-full", mode, "--items-per-update", "2",
                                          "--epochs", "1", "--micro-bs", "4", "--max-len", "256", "--precision", "bf16",
                                          "--no-grad-ckpt", "--log-every", "1", "--seed", "3", "--out", f"{d}/o-{name}-{mode}"],
                                         capture_output=True, text=True, timeout=900, env=ENV)
                    self.assertEqual(out.returncode, 0, out.stderr[-2000:])
                    m = json.loads(Path(f"{d}/o-{name}-{mode}/metrics.json").read_text())
                    hist[name, mode] = ([(h["step"], h["loss"], h["seqs"]) for h in m["history"]], m["eval"]["v"])
            for mode in ("true", "matched"):
                self.assertEqual(hist["head", mode], hist["now", mode], mode)


@unittest.skipUnless(HAVE, "needs torch")
class TestLocalViews(unittest.TestCase):
    def build(self, mode, view, seed=0, **kw):
        from cotdistill.student import build_full_tree
        st = {}
        ex = build_full_tree(KK, mode=mode, view=view, rng=random.Random(seed), cap=kw.pop("cap", 0), stats=st, **kw)
        return ex, st

    def test_local_rendering(self):
        from cotdistill.treeplan import LOCAL_HEADER, _fact_line
        by = {sq["id"]: sq for sq in KK["subqs"]}
        byq = {sq["question"]: sq for sq in KK["subqs"]}
        ex, st = self.build("true", "local")
        self.assertEqual(ex[0].kind, "final")
        self.assertEqual(ex[0].text, KK["prompt"] + "\n\nAnswer:")                    # final view unchanged
        stmt = re.findall(r"^- .+$", KK["prompt"], re.M)
        for e in ex[1:]:
            head, q, stated = split_view(e)
            sq = byq[q]
            self.assertTrue(head.startswith(KK["local_preamble"] + "\n\n"))
            self.assertNotIn("Question: Who is", e.text)                              # no final question
            self.assertNotIn("Options:\nA) " + KK["prompt"].split("Options:\nA) ")[1].split("\n")[0], e.text)
            if sq["type"] == "parse":
                self.assertEqual(head, KK["local_preamble"] + "\n\n" + sq["local_lines"][0])
                self.assertIn(sq["local_lines"][0], stmt)
            else:
                self.assertIn(LOCAL_HEADER, head)
                self.assertEqual(sorted(stated), sorted(_fact_line(by[i]) for i in sq["depends_on"]))
                self.assertFalse(any(l in head for l in stmt))                         # no problem statements
            self.assertEqual(e.labels, ["A", "B"])
        parse = [sq for sq in KK["subqs"] if sq["type"] == "parse"]
        self.assertEqual(st["leaf_views"], len(parse))
        self.assertEqual(st["views"], len(KK["subqs"]) - sum(v for k, v in st.items() if k.startswith("dropped")))

    def test_ancestors_rendering(self):
        from cotdistill.treeplan import _fact_line, ancestors
        byq = {sq["question"]: sq for sq in KK["subqs"]}
        anc = ancestors(KK["subqs"])
        depth = {sq["id"]: sq["node_depth"] for sq in KK["subqs"]}
        line_d = {_fact_line(sq): depth[sq["id"]] for sq in KK["subqs"]}
        ex, st = self.build("true", "ancestors")
        self.assertGreater(len(ex), 1)
        for e in ex[1:]:
            _, q, stated = split_view(e)
            sq = byq[q]
            if sq["type"] == "parse":
                self.assertEqual(stated, [])
                continue
            self.assertEqual(sorted(stated), sorted(_fact_line(x) for x in KK["subqs"] if x["id"] in anc[sq["id"]]))
            ds = [line_d[l] for l in stated]
            self.assertEqual(ds, sorted(ds))                                           # shallow to deep
            self.assertGreaterEqual(len(stated), len(sq["depends_on"]))

    def test_no_cap_and_cap(self):
        ex, st = self.build("true", "local", cap=0)
        self.assertEqual(len(ex) - 1 + sum(v for k, v in st.items() if k.startswith("dropped")), len(KK["subqs"]))
        ex4, _ = self.build("true", "local", cap=4)
        self.assertLessEqual(len(ex4) - 1, 4)
        from cotdistill.treeplan import full_tree_plan
        self.assertEqual(len(full_tree_plan(KK, cap=0, rng=random.Random(0))), len(KK["subqs"]))

    def test_source_filter(self):
        for mode, view in (("true", "local"), ("matched", "local"), ("true", "ancestors"), ("plain", "problem")):
            ex, _ = self.build(mode, view, sources={"justlogic"})
            self.assertEqual([e.kind for e in ex], ["final"])
            ex, _ = self.build(mode, view, sources={"knights_knaves"})
            self.assertGreater(len(ex), 1)

    def test_problem_view_every_node(self):
        """GFP: plain + problem view, no cap: every node with the full problem and no stated results."""
        ex, st = self.build("plain", "problem", sources={"knights_knaves"})
        self.assertEqual(len(ex) - 1, len(KK["subqs"]))
        for e in ex[1:]:
            self.assertTrue(e.text.startswith(KK["prompt"] + "\n\nIntermediate question: "))
        self.assertAlmostEqual(sum(e.weight for e in ex[1:]), 1.0)

    def test_leak_filter(self):
        from cotdistill.treeplan import view_leak
        people = re.findall(r"^- ([A-Z][a-z]+) says", KK["prompt"], re.M)
        roles = [{"question": f"Is {p} a knight?", "truth": True} for p in people]
        node = {"question": "Is the statement by X true?"}
        self.assertEqual(view_leak(KK, node, roles), "assignment")
        self.assertIsNone(view_leak(KK, node, roles[:-1]))
        self.assertEqual(view_leak(KK, {"question": "Is option A the solution?"}, []), "question")
        self.assertEqual(view_leak(KK, node, [{"question": "Which option is consistent?", "truth": True}]), "stated")
        # a node whose stated ancestors name every inhabitant's role is dropped and counted
        ex, st = self.build("true", "ancestors")
        for e in ex[1:]:
            _, _, stated = split_view(e)
            named = {m.group(1) for m in (re.match(r"^Q: Is ([A-Z][a-z]+) a kn(?:ight|ave)\? A:", l) for l in stated) if m}
            self.assertFalse(named >= set(people))
        self.assertEqual(st.get("dropped_leak_question", 0) + st.get("dropped_leak_stated", 0), 0)

    def test_matched_local_same_counts(self):
        from cotdistill.treeplan import full_tree_plan
        by = {sq["id"]: sq for sq in KK["subqs"]}
        for seed in range(8):
            t, st_t = self.build("true", "local", seed=seed)
            m, st_m = self.build("matched", "local", seed=seed)
            vt, vm = [split_view(e) for e in t[1:]], [split_view(e) for e in m[1:]]
            self.assertEqual([q for _, q, _ in vt], [q for _, q, _ in vm])                # same nodes, same order
            self.assertEqual([len(s) for _, _, s in vt], [len(s) for _, _, s in vm])      # same number of results
            self.assertEqual([e.target for e in t], [e.target for e in m])               # rng in lockstep
            self.assertEqual(st_t["stated_lines"], st_m["stated_lines"])
            plan = {e["node"]["question"]: e for e in full_tree_plan(KK, cap=0, rng=random.Random(seed))}
            frozen = 0
            for (ht, q, s_t), (hm, _, s_m) in zip(vt, vm):
                e = plan[q]
                if e["parents"]:                                                          # eligible: no parent stated
                    self.assertTrue(set(s_t).isdisjoint(s_m))
                    self.assertEqual(len(e["matched"]), len(s_m))
                else:                                                                     # leaf or frozen: identical
                    self.assertEqual(ht, hm)
                    frozen += bool(s_t)
            self.assertEqual(frozen, st_m.get("matched_frozen", 0))
            self.assertGreater(sum(bool(plan[q]["parents"]) for _, q, _ in vt), 0)

    def test_node_weight(self):
        ex, _ = self.build("true", "local")
        self.assertAlmostEqual(sum(e.weight for e in ex[1:]), 1.0)                     # per-problem convention
        ex, _ = self.build("true", "local", node_weight=0.1)
        self.assertTrue(all(abs(e.weight - 0.1) < 1e-12 for e in ex[1:]) and ex[0].weight == 1.0)
        ex, _ = self.build("plain", "problem", node_weight=0.1)
        self.assertTrue(all(abs(e.weight - 0.1) < 1e-12 for e in ex[1:]))

    def test_errors(self):
        for mode, view in (("matched", "ancestors"), ("plain", "ancestors"), ("control", "local"), ("placebo", "local"),
                           ("true", "statements"), ("matched", "statements"), ("control", "statements")):
            with self.assertRaises(ValueError):
                self.build(mode, view)
        for kw in (dict(eligible_only=True), dict(node_format="mc")):
            with self.assertRaises(ValueError):
                self.build("plain", "statements", **kw)
        with self.assertRaises(ValueError):
            from cotdistill.student import build_full_tree
            build_full_tree(dict(KK, local_preamble=None), mode="true", view="local", rng=random.Random(0), cap=0)

    def test_local_eval_examples(self):
        from cotdistill.student import local_eval_examples
        exs, meta = local_eval_examples([KK, {"item_id": "other", "subqs": []}])
        ex, st = self.build("true", "local")
        self.assertEqual(len(exs), st["views"])
        self.assertEqual({m["item_id"] for m in meta}, {KK["item_id"]})
        self.assertTrue(all(m["depth"] == next(sq["node_depth"] for sq in KK["subqs"] if sq["id"] == m["id"]) for m in meta))
        self.assertEqual({m["trivial"] for m in meta if m["type"] == "parse"}, {None})
        self.assertEqual(sum(bool(m["trivial"]) for m in meta),
                         sum(bool(sq.get("local_trivial")) for sq in KK["subqs"]))
        self.assertGreater(sum(m["trivial"] is False for m in meta), 0)
        norm = lambda e: sorted(e.text.split("Options:")[0].split("\n"))                 # line order is drawn
        self.assertEqual(sorted(map(norm, exs)), sorted(map(norm, ex[1:])))             # same views as training



V3SL_VAL = ROOT / "data/student_v3sl/val.jsonl"
NUMGEN_VAL = ROOT / "data/student_numgen_tf/val.jsonl"


def first_items(path: Path, source: str, n: int) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            it = json.loads(line)
            if it.get("source") == source:
                out.append(it)
                if len(out) == n:
                    break
    return out


def p_yes(e) -> float:
    """The soft target's probability of Yes (the yes/no order is drawn per view)."""
    return e.target[0] if e.text.endswith("Options:\nA) Yes\nB) No\nAnswer:") else e.target[1]


@unittest.skipUnless(HAVE, "needs torch")
class TestControlArms(unittest.TestCase):
    """GFP-S (--tree-full plain --tree-view statements) and TFLN (--tree-full plain --tree-view local)."""

    def build(self, item, mode, view, seed=0, **kw):
        from cotdistill.student import build_full_tree
        st = {}
        ex = build_full_tree(item, mode=mode, view=view, rng=random.Random(seed), cap=kw.pop("cap", 0), stats=st, **kw)
        return ex, st

    def check_gfps(self, item, seed):
        """GFP-S views: statement block only (no final question, no options), else GFP in rng lockstep."""
        from cotdistill.treeplan import statement_block
        block = statement_block(item)
        stmt = re.findall(r"^- .+$", item["prompt"], re.M)
        opt_lines = item["prompt"].split("\nOptions:\n", 1)[1].split("\n")
        self.assertEqual(block, item["prompt"].split("\n\nQuestion: ")[0])
        self.assertEqual(block.split("\n\n")[0], item["local_preamble"])
        s, _ = self.build(item, "plain", "statements", seed=seed)
        g, _ = self.build(item, "plain", "problem", seed=seed)
        self.assertEqual(s[0].text, item["prompt"] + "\n\nAnswer:")                    # final view unchanged
        self.assertEqual(len(s) - 1, len(item["subqs"]))                               # every node
        for es, eg in zip(s[1:], g[1:]):
            head, q, stated = split_view(es)
            self.assertEqual(head, block)
            self.assertEqual(stated, [])
            self.assertNotIn("Question:", head)
            self.assertNotIn("Options:", head)
            self.assertFalse(any(l[3:] in head for l in opt_lines))                    # no option text
            self.assertTrue(all(l in head for l in stmt))                              # every statement line
            self.assertEqual(es.text, eg.text.replace(item["prompt"], block, 1))       # GFP minus question/options
            self.assertEqual((es.labels, es.target, es.weight, es.kind, es.item_id),
                             (eg.labels, eg.target, eg.weight, eg.kind, eg.item_id))   # rng lockstep with GFP
        return s

    def check_tfln(self, item, seed):
        """TFLN views: parse nodes their line, derived nodes the preamble only; else TFL in rng lockstep."""
        from cotdistill.treeplan import LOCAL_HEADER
        byq = {sq["question"]: sq for sq in item["subqs"]}
        stmt = re.findall(r"^- .+$", item["prompt"], re.M)
        n, st_n = self.build(item, "plain", "local", seed=seed)
        t, st_t = self.build(item, "true", "local", seed=seed)
        self.assertEqual(len(n), len(t))
        self.assertEqual(n[0].text, t[0].text)
        for en, et in zip(n[1:], t[1:]):
            head, q, stated = split_view(en)
            sq = byq[q]
            self.assertEqual(q, split_view(et)[1])                                     # same node, same order
            self.assertEqual((en.labels, en.target, en.weight), (et.labels, et.target, et.weight))   # lockstep
            self.assertNotIn(LOCAL_HEADER, en.text)
            self.assertEqual(stated, [])
            self.assertNotIn("Question:", head)
            if sq.get("local_lines"):
                self.assertEqual(head, item["local_preamble"] + "\n\n" + "\n".join(sq["local_lines"]))
                self.assertEqual(en.text, et.text)                                     # leaves identical to TFL
            else:
                self.assertEqual(head, item["local_preamble"])                         # nothing stated
                self.assertFalse(any(l in en.text for l in stmt))
        self.assertEqual(st_n["views"], st_t["views"])
        self.assertEqual(st_n["leaf_views"], st_t["leaf_views"])
        self.assertEqual(st_n["withheld_lines"], st_t["stated_lines"])
        self.assertNotIn("stated_lines", st_n)
        return n

    def check_same_nodes(self, item, seed):
        """GFP, GFP-S, TFL and TFLN: the same nodes in the same order with the same targets (as P(Yes); the yes/no
        order is in lockstep only within GFP/GFP-S and within TFL/TFLN) and the same weights."""
        arms = {name: self.build(item, mode, view, seed=seed)[0]
                for name, (mode, view) in {"GFP": ("plain", "problem"), "GFPS": ("plain", "statements"),
                                           "TFL": ("true", "local"), "TFLN": ("plain", "local")}.items()}
        ref = arms["GFP"]
        for name, ex in arms.items():
            self.assertEqual([split_view(e)[1] for e in ex[1:]], [split_view(e)[1] for e in ref[1:]], name)
            self.assertEqual([round(p_yes(e), 12) for e in ex[1:]], [round(p_yes(e), 12) for e in ref[1:]], name)
            self.assertEqual([e.weight for e in ex], [e.weight for e in ref], name)
            self.assertEqual(ex[0].text, ref[0].text, name)

    def test_fixture(self):
        for seed in range(6):
            self.check_gfps(KK, seed)
            self.check_tfln(KK, seed)
            self.check_same_nodes(KK, seed)
        with self.assertRaises(ValueError):                                            # TFLN needs local fields
            self.build(dict(KK, local_preamble=None), "plain", "local")

    @unittest.skipUnless(V3SL_VAL.exists(), "data/student_v3sl not built")
    def test_real_v3sl_items(self):
        items = first_items(V3SL_VAL, "knights_knaves", 5)
        self.assertEqual(len(items), 5)
        for it in items:
            for seed in (0, 1):
                self.check_gfps(it, seed)
                self.check_tfln(it, seed)
                self.check_same_nodes(it, seed)

    def test_sources_and_weights(self):
        for view in ("statements", "local"):
            ex, _ = self.build(KK, "plain", view, sources={"justlogic"})
            self.assertEqual([e.kind for e in ex], ["final"])
            ex, _ = self.build(KK, "plain", view)
            self.assertAlmostEqual(sum(e.weight for e in ex[1:]), 1.0)                 # per-problem convention
            ex, _ = self.build(KK, "plain", view, node_weight=0.1)
            self.assertTrue(all(abs(e.weight - 0.1) < 1e-12 for e in ex[1:]))
        other = {"item_id": "o", "source": "justlogic", "prompt": "P\nOptions:\nA) a\nB) b", "labels": ["A", "B"],
                 "gold_label": "A", "teacher": {"A": 0.8, "B": 0.2},
                 "subqs": [{"id": "n1", "question": "q?", "p_cot": 0.9, "depends_on": []}]}
        with self.assertRaises(ValueError):                                            # no statement block defined
            self.build(other, "plain", "statements")
        with self.assertRaises(ValueError):                                            # malformed K&K prompt
            self.build(dict(KK, prompt=KK["prompt"].replace("\n\nQuestion: ", "\n\nAsk: ")), "plain", "statements")

    @unittest.skipUnless(NUMGEN_VAL.exists(), "data/student_numgen_tf not built")
    def test_numgen_statement_block(self):
        from cotdistill.treeplan import statement_block
        for it in first_items(NUMGEN_VAL, "numgen", 5):
            block = statement_block(it)
            ask = next(q["question"] for q in it["program"]["quantities"] if q["id"] == it["program"]["answer"])
            self.assertEqual(it["prompt"].split("\nOptions:\n")[0], "Question: " + block + " " + ask)
            self.assertTrue(block.startswith(it["local_preamble"] + " "))
            self.assertNotIn("?", block)
            self.assertNotIn("Question:", block)
            self.assertNotIn("\n", block)
            for sq in it["subqs"]:                                                     # every leaf sentence kept
                for line in sq.get("local_lines") or []:
                    self.assertIn(line, block)
            s, _ = self.build(it, "plain", "statements", sources={"numgen"})
            g, _ = self.build(it, "plain", "problem", sources={"numgen"})
            self.assertEqual(len(s), len(g))
            for es, eg in zip(s[1:], g[1:]):
                self.assertEqual(split_view(es)[0], block)
                self.assertEqual(es.text, eg.text.replace(it["prompt"], block, 1))
                self.assertEqual((es.target, es.weight), (eg.target, eg.weight))


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestTrainLocal(unittest.TestCase):
    def test_training_on_cpu(self):
        with tempfile.TemporaryDirectory() as d:
            tiny(d)
            other = {"item_id": "o", "source": "justlogic", "prompt": "P\nOptions:\nA) a\nB) b", "labels": ["A", "B"],
                     "gold_label": "A", "teacher": {"A": 0.8, "B": 0.2},
                     "subqs": [{"id": "n1", "question": "q?", "p_cot": 0.9, "depends_on": []}]}
            items = [dict(KK, item_id=f"k{j}") for j in range(3)] + [dict(other, item_id=f"o{j}") for j in range(3)]
            Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in items))
            run = lambda *a: subprocess.run(
                [sys.executable, SCRIPT, "--model", d + "/m", "--train", d + "/t.jsonl", "--eval", "v=" + d + "/t.jsonl",
                 "--items-per-update", "3", "--epochs", "1", "--micro-bs", "8", "--max-len", "512", "--precision", "bf16",
                 "--no-grad-ckpt", "--log-every", "1", "--out", d + "/o", *a],
                capture_output=True, text=True, timeout=900, env=ENV)
            for mode, view in (("true", "local"), ("matched", "local"), ("true", "ancestors"), ("plain", "local"),
                               ("plain", "statements")):
                out = run("--tree-full", mode, "--tree-view", view, "--tree-cap", "0", "--tree-sources", "knights_knaves",
                          "--eval-local", d + "/t.jsonl")
                self.assertEqual(out.returncode, 0, out.stderr[-2000:])
                m = json.loads(Path(d + "/o/metrics.json").read_text())
                self.assertIn("local_kk", m["eval"])
                self.assertIn("derived_nontrivial", m["eval"]["local_kk"])
                if view == "local":                    # the probe asks the TFL views of every node
                    self.assertEqual(m["eval"]["local_kk"]["all"]["n"], m["tree_view_stats"]["views"])
                self.assertGreater(m["tree_view_stats"]["views"], 0)
                self.assertIn("tree views", out.stdout)
                for h in m["history"]:                 # loss split by kind and the pre-clip gradient norm
                    self.assertAlmostEqual(h["loss_final"] + h["loss_node"], h["loss"], delta=1e-3 * max(1, h["loss"]))
                    self.assertGreater(h["loss_node"], 0)
                    self.assertGreater(h["node_ce"], 0)
                    self.assertGreater(h["gnorm"], 0)
                    self.assertTrue(0 <= h["clip_frac"] <= 1 and h["gnorm_max"] >= h["gnorm"])
            out = run("--tree-full", "matched", "--tree-view", "ancestors")
            self.assertNotEqual(out.returncode, 0)
            for mode in ("true", "matched"):
                out = run("--tree-full", mode, "--tree-view", "statements")
                self.assertNotEqual(out.returncode, 0)
            out = run("--tree-full", "plain", "--tree-view", "statements")       # justlogic items have no block
            self.assertNotEqual(out.returncode, 0)
            self.assertIn("no statement block", out.stderr)
            out = run("--tree-view", "local")
            self.assertNotEqual(out.returncode, 0)


if __name__ == "__main__":
    unittest.main()
