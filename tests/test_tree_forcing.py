"""Tree Forcing views (--tree-view local / ancestors / statements / local+ctxparse / mix, --tree-cap 0, --tree-sources,
--tree-node-weight, --eval-local): default behaviour identical to HEAD, rendering, filters, the controls, the
premise-extraction arms TFLP / MIX and GTF (--tree-scaffold, --tree-depth-weight). CPU only."""
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
# The last commit before the TFLP / MIX views: the existing Tree Forcing views (TFL, TFA, TFLM, GFP, GFP-S, TFLN) are
# compared with it.
VIEWS_REF = os.environ.get("TF_VIEWS_REF", "c289894")
# The last commit before GTF (--tree-scaffold / --tree-depth-weight): every existing view, TFLP and MIX included, is
# compared with it.
GTF_REF = os.environ.get("TF_GTF_REF", "94e283c")


def head_tree(dest: str, ref: str = BASE_REF) -> Path | None:
    """src/ and scripts/ of commit ref (default: the pre-Tree-Forcing BASE_REF) under dest (None outside a git
    checkout)."""
    try:
        blob = subprocess.run(["git", "-C", str(ROOT), "archive", ref, "src/cotdistill", "scripts/train_student.py"],
                              capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    Path(dest).mkdir(parents=True, exist_ok=True)
    p = Path(dest) / "head.tar"
    p.write_bytes(blob)
    with tarfile.open(p) as t:
        t.extractall(dest, filter="data")
    return Path(dest)


def head_student(dest: str, ref: str = BASE_REF, name: str = "cotdistill_head"):
    """cotdistill.student of commit ref, imported as a separate package (name; one name per ref)."""
    root = head_tree(dest, ref)
    if root is None:
        return None
    (root / "src/cotdistill").rename(root / f"src/{name}")
    sys.path.insert(0, str(root / "src"))
    import importlib
    return importlib.import_module(f"{name}.student")


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


def yes_no_canon(text: str) -> str:
    """A node view's text with its options in the order Yes, No (the order is drawn per view)."""
    return text.replace("Options:\nA) No\nB) Yes\nAnswer:", "Options:\nA) Yes\nB) No\nAnswer:")


def problem_sentences(item: dict) -> set:
    """The problem's statement lines (K&K "- X says: ...") / sentences (numgen) that the item's parse nodes read."""
    return {line for sq in item["subqs"] for line in sq.get("local_lines") or []}


@unittest.skipUnless(HAVE, "needs torch")
class TestCtxArms(unittest.TestCase):
    """TFLP (--tree-full true --tree-view local+ctxparse) and MIX (--tree-full true --tree-view mix): TFL with some
    nodes asked in the GFP-S statements view instead, in rng lockstep with TFL."""

    def build(self, item, mode, view, seed=0, src="knights_knaves", **kw):
        from cotdistill.student import build_full_tree
        st = {}
        ex = build_full_tree(item, mode=mode, view=view, rng=random.Random(seed), cap=kw.pop("cap", 0), stats=st,
                             sources={src}, **kw)
        return ex, st

    def against_tfl(self, item, ex, seed, src):
        """Each node view of ex is TFL's view byte for byte ("local") or TFL's view with its head replaced by the
        statement block, nothing stated ("stmt"); nodes, order, labels, soft targets (yes/no order included), weights
        and the final view are TFL's. -> (kinds, TFL's stats)."""
        from cotdistill.treeplan import LOCAL_HEADER, statement_block
        block = statement_block(item)
        t, st_t = self.build(item, "true", "local", seed, src)
        self.assertEqual(len(ex), len(t))
        self.assertEqual(ex[0].text, t[0].text)
        self.assertEqual(ex[0].text, item["prompt"] + "\n\nAnswer:")                  # final view unchanged
        kinds = []
        for e, et in zip(ex[1:], t[1:]):
            head, q, stated = split_view(e)
            head_t, q_t, _ = split_view(et)
            self.assertEqual(q, q_t)                                                     # same node, same order
            self.assertEqual((e.labels, e.target, e.weight, e.kind, e.item_id),
                             (et.labels, et.target, et.weight, et.kind, et.item_id))     # rng lockstep with TFL
            self.assertNotIn("Question:", head)
            self.assertNotIn("Options:", head)
            if head == block:
                kinds.append("stmt")
                self.assertEqual(stated, [])
                self.assertNotIn(LOCAL_HEADER, e.text)
                self.assertEqual(e.text, et.text.replace(head_t, block, 1))
            else:
                kinds.append("local")
                self.assertEqual(e.text, et.text)
        return kinds, st_t

    def check_tflp(self, item, seed, src="knights_knaves"):
        """TFLP: parse views = the whole statement block (every statement line, not only their own); derived views =
        TFL's local views, with no problem text."""
        from cotdistill.treeplan import statement_block
        byq = {sq["question"]: sq for sq in item["subqs"]}
        block, sent = statement_block(item), problem_sentences(item)
        p, st = self.build(item, "true", "local+ctxparse", seed, src)
        kinds, st_t = self.against_tfl(item, p, seed, src)
        for e, k in zip(p[1:], kinds):
            sq = byq[split_view(e)[1]]
            if sq.get("local_lines"):
                self.assertEqual(k, "stmt")
                self.assertTrue(all(line in block for line in sq["local_lines"]))
                self.assertTrue(all(line in e.text for line in sent))
            else:
                self.assertEqual(k, "local")
                self.assertTrue(e.text.startswith(item["local_preamble"] + "\n\n"))
                self.assertFalse(any(line in e.text for line in sent))                    # no problem text
        self.assertEqual(st["views"], st_t["views"])
        self.assertEqual(st["statements_views"], st_t["leaf_views"])
        self.assertEqual(st["statements_leaf_views"], st_t["leaf_views"])
        self.assertEqual(st["local_views"], st_t["views"] - st_t["leaf_views"])
        self.assertEqual(st["stated_lines"], st_t["stated_lines"])
        self.assertEqual((st["withheld_lines"], st["statements_no_parents"]), (0, 0))
        return kinds

    def check_mix(self, item, seed, src="knights_knaves", epoch=0, p=0.5):
        """MIX: one view per node (TFL's or the statements view), counted in the stats; the choice depends on (seed,
        epoch, item, node) only, not on the training rng; the same arguments give the same examples. -> kinds."""
        kw = dict(mix_p=p, mix_seed=seed, mix_epoch=epoch)
        m, st = self.build(item, "true", "mix", seed, src, **kw)
        kinds, st_t = self.against_tfl(item, m, seed, src)
        self.assertEqual(st["views"], st_t["views"])
        self.assertEqual((st["statements_views"], st["local_views"]), (kinds.count("stmt"), kinds.count("local")))
        self.assertEqual(st["stated_lines"] + st["withheld_lines"], st_t["stated_lines"])
        self.assertEqual(st["statements_no_parents"], 0)
        again, _ = self.build(item, "true", "mix", seed, src, **kw)
        self.assertEqual([dataclasses.astuple(e) for e in again], [dataclasses.astuple(e) for e in m])
        other_rng, _ = self.build(item, "true", "mix", seed + 100, src, **kw)          # another training-rng seed
        self.assertEqual(self.against_tfl(item, other_rng, seed + 100, src)[0], kinds)
        return kinds

    def check_same_nodes(self, item, seed, src="knights_knaves"):
        """TFLP and MIX vs GFP-S: the same nodes in the same order, P(Yes), weights and final view; MIX with mix_p 1
        is GFP-S up to the yes/no order, with mix_p 0 it is TFL."""
        g, _ = self.build(item, "plain", "statements", seed, src)
        t, _ = self.build(item, "true", "local", seed, src)
        arms = {"TFLP": self.build(item, "true", "local+ctxparse", seed, src)[0],
                "MIX": self.build(item, "true", "mix", seed, src, mix_seed=seed)[0]}
        for name, ex in arms.items():
            self.assertEqual([split_view(e)[1] for e in ex[1:]], [split_view(e)[1] for e in g[1:]], name)
            self.assertEqual([round(p_yes(e), 12) for e in ex[1:]], [round(p_yes(e), 12) for e in g[1:]], name)
            self.assertEqual([e.weight for e in ex], [e.weight for e in g], name)
            self.assertEqual(ex[0].text, g[0].text, name)
        for e, eg in zip(arms["TFLP"][1:], g[1:]):                                       # parse views = GFP-S views
            if split_view(e)[0] == split_view(eg)[0]:
                self.assertEqual(yes_no_canon(e.text), yes_no_canon(eg.text))
        all_stmt, _ = self.build(item, "true", "mix", seed, src, mix_p=1.0, mix_seed=seed)
        self.assertEqual([yes_no_canon(e.text) for e in all_stmt], [yes_no_canon(e.text) for e in g])
        self.assertEqual([round(p_yes(e), 12) for e in all_stmt], [round(p_yes(e), 12) for e in g])
        all_local, _ = self.build(item, "true", "mix", seed, src, mix_p=0.0, mix_seed=seed)
        self.assertEqual([dataclasses.astuple(e) for e in all_local], [dataclasses.astuple(e) for e in t])

    def test_fixture(self):
        kinds = {}
        for seed in range(6):
            k = self.check_tflp(KK, seed)
            self.assertEqual(k.count("stmt"), sum(bool(sq.get("local_lines")) for sq in KK["subqs"]))
            self.assertGreater(k.count("local"), 0)
            for epoch in (0, 1):
                kinds[seed, epoch] = self.check_mix(KK, seed, epoch=epoch)
            self.assertNotEqual(kinds[seed, 0], kinds[seed, 1])                           # epochs differ
            self.check_same_nodes(KK, seed)
        self.assertNotEqual(kinds[0, 0], kinds[1, 0])                                     # seeds differ
        share = sum(k.count("stmt") for k in kinds.values()) / sum(map(len, kinds.values()))
        self.assertTrue(0.4 < share < 0.6, share)                                          # 12 x 21 views

    @unittest.skipUnless(V3SL_VAL.exists(), "data/student_v3sl not built")
    def test_real_v3sl_items(self):
        items = first_items(V3SL_VAL, "knights_knaves", 5)
        self.assertEqual(len(items), 5)
        for seed in (0, 1):
            share = {}
            for epoch in (0, 1):
                ks = [self.check_mix(it, seed, epoch=epoch) for it in items]
                share[epoch] = sum(k.count("stmt") for k in ks) / sum(map(len, ks))
                self.assertTrue(0.4 < share[epoch] < 0.6, share)                          # about half of each kind
                share[epoch, "kinds"] = ks
            self.assertNotEqual(share[0, "kinds"], share[1, "kinds"])
            for it in items:
                self.check_tflp(it, seed)
                self.check_same_nodes(it, seed)

    @unittest.skipUnless(NUMGEN_VAL.exists(), "data/student_numgen_tf not built")
    def test_numgen_items(self):
        """5 numgen val items: the first of each depth (d1: 1 operation, read / parse leaves only; d2-d4: with
        derived nodes) and the second d4 item."""
        items, per = [], {}
        with open(NUMGEN_VAL) as f:
            for line in f:
                it = json.loads(line)
                d = it["item_id"].split("-")[2]
                per[d] = per.get(d, 0) + 1
                if it.get("source") == "numgen" and per[d] <= (2 if d == "d4" else 1):
                    items.append(it)
        self.assertEqual(len(items), 5)
        kinds = []
        for it in items:
            kinds += self.check_tflp(it, 0, "numgen")          # (depth-1 items have only read / parse leaves)
            self.check_mix(it, 0, "numgen")
            self.check_same_nodes(it, 0, "numgen")
        self.assertGreater(kinds.count("stmt") * kinds.count("local"), 0)

    def test_node_without_parents(self):
        """A non-leaf node without parents: TFL drops it; TFLP and MIX ask it in the statements view."""
        from cotdistill.treeplan import CTX_VIEWS, statement_block
        item = json.loads(json.dumps(KK))
        sq = next(x for x in item["subqs"] if not x.get("local_lines") and x["depends_on"])
        sq["depends_on"] = []
        t, st_t = self.build(item, "true", "local")
        self.assertEqual(st_t["dropped_no_parents"], 1)
        for view in CTX_VIEWS:
            for p in (0.0, 1.0):
                ex, st = self.build(item, "true", view, mix_p=p)
                self.assertEqual(len(ex), len(t) + 1)
                self.assertEqual(st["statements_no_parents"], 1)
                e = next(e for e in ex[1:] if split_view(e)[1] == sq["question"])
                self.assertEqual(split_view(e)[0], statement_block(item))

    def test_sources_weights_errors(self):
        from cotdistill.treeplan import CTX_VIEWS
        other = {"item_id": "o", "source": "justlogic", "prompt": "P\nOptions:\nA) a\nB) b", "labels": ["A", "B"],
                 "gold_label": "A", "teacher": {"A": 0.8, "B": 0.2}, "local_preamble": "P",
                 "subqs": [{"id": "n1", "question": "q?", "p_cot": 0.9, "depends_on": [], "local_lines": ["P"]}]}
        for view in CTX_VIEWS:
            ex, _ = self.build(KK, "true", view, src="justlogic")
            self.assertEqual([e.kind for e in ex], ["final"])
            ex, _ = self.build(KK, "true", view)
            self.assertAlmostEqual(sum(e.weight for e in ex[1:]), 1.0)                  # per-problem convention
            ex, _ = self.build(KK, "true", view, node_weight=0.1)
            self.assertTrue(all(abs(e.weight - 0.1) < 1e-12 for e in ex[1:]) and ex[0].weight == 1.0)
            for mode in ("matched", "plain", "placebo", "control"):
                with self.assertRaises(ValueError):
                    self.build(KK, mode, view)
            for kw in (dict(eligible_only=True), dict(node_format="mc"), dict(mix_p=1.5)):
                with self.assertRaises(ValueError):
                    self.build(KK, "true", view, **kw)
            with self.assertRaisesRegex(ValueError, "no local_preamble"):
                self.build(dict(KK, local_preamble=None), "true", view)
            with self.assertRaisesRegex(ValueError, "no statement block"):
                self.build(other, "true", view, src="justlogic")
            with self.assertRaises(ValueError):                                          # malformed K&K prompt
                self.build(dict(KK, prompt=KK["prompt"].replace("\n\nQuestion: ", "\n\nAsk: ")), "true", view)

    def test_existing_views_match_ref(self):
        """The views that existed before TFLP / MIX (TFL, TFLM, TFLN, TFA, GFP-S, GFP and the problem-view modes) give
        exactly VIEWS_REF's examples and stats."""
        from cotdistill.student import build_full_tree
        with tempfile.TemporaryDirectory() as d:
            ref = head_student(d, VIEWS_REF, "cotdistill_views")
            if ref is None:
                self.skipTest("not a git checkout")
            items = [(KK, "knights_knaves")]
            if V3SL_VAL.exists():
                items += [(it, "knights_knaves") for it in first_items(V3SL_VAL, "knights_knaves", 3)]
            if NUMGEN_VAL.exists():
                items += [(it, "numgen") for it in first_items(NUMGEN_VAL, "numgen", 3)]
            n = 0
            for item, src in items:
                for mode, view in (("true", "local"), ("matched", "local"), ("plain", "local"), ("true", "ancestors"),
                                   ("plain", "statements"), ("plain", "problem"), ("true", "problem")):
                    for nw in (0.0, 0.1):
                        for seed in (0, 1):
                            kw = dict(mode=mode, view=view, cap=0, sources={src}, node_weight=nw, permute_final=0.5)
                            sa, sb = {}, {}
                            a = ref.build_full_tree(item, rng=random.Random(seed), stats=sa, **kw)
                            b = build_full_tree(item, rng=random.Random(seed), stats=sb, **kw)
                            self.assertEqual([dataclasses.astuple(e) for e in a], [dataclasses.astuple(e) for e in b],
                                             (item["item_id"], mode, view, nw, seed))
                            self.assertEqual(sa, sb)
                            n += 1
            self.assertGreaterEqual(n, 28)


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestTrainCtxArms(unittest.TestCase):
    OTHER = {"item_id": "o", "source": "justlogic", "prompt": "P\nOptions:\nA) a\nB) b", "labels": ["A", "B"],
             "gold_label": "A", "teacher": {"A": 0.8, "B": 0.2},
             "subqs": [{"id": "n1", "question": "q?", "p_cot": 0.9, "depends_on": []}]}

    def setup_dir(self, d):
        tiny(d)
        items = [dict(KK, item_id=f"k{j}") for j in range(3)] + [dict(self.OTHER, item_id=f"o{j}") for j in range(3)]
        Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in items))

    def run_script(self, d, *a, script=SCRIPT, out="o"):
        return subprocess.run(
            [sys.executable, script, "--model", d + "/m", "--train", d + "/t.jsonl", "--eval", "v=" + d + "/t.jsonl",
             "--items-per-update", "3", "--micro-bs", "8", "--max-len", "512", "--no-grad-ckpt", "--log-every", "1",
             "--out", f"{d}/{out}", *a], capture_output=True, text=True, timeout=900, env=ENV)

    def test_training_on_cpu(self):
        """Both arms train two epochs with the local-step probe; MIX draws new coins in each epoch (pass 0 = the
        plan's counting pass, then epochs 0 and 1); the argument checks refuse the combinations that are not defined."""
        with tempfile.TemporaryDirectory() as d:
            self.setup_dir(d)
            tree = ("--tree-cap", "0", "--tree-sources", "knights_knaves", "--epochs", "2", "--precision", "bf16")
            for view in ("local+ctxparse", "mix"):
                out = self.run_script(d, "--tree-full", "true", "--tree-view", view, *tree, "--eval-local", d + "/t.jsonl")
                self.assertEqual(out.returncode, 0, out.stderr[-2000:])
                m = json.loads(Path(d + "/o/metrics.json").read_text())
                st = m["tree_view_stats"]
                self.assertEqual(m["eval"]["local_kk"]["all"]["n"], st["views"])        # the TFL node set
                self.assertIn("derived_nontrivial", m["eval"]["local_kk"])
                self.assertEqual(st["statements_views"] + st["local_views"], st["views"])
                self.assertGreater(st["statements_views"] * st["local_views"], 0)
                if view == "mix":
                    self.assertEqual(re.findall(r"^mix pass (\d+): ", out.stdout, re.M), ["0", "0", "1"])
                else:
                    self.assertEqual(st["statements_views"], st["leaf_views"])
                    self.assertNotIn("mix pass", out.stdout)
                for h in m["history"]:
                    self.assertAlmostEqual(h["loss_final"] + h["loss_node"], h["loss"], delta=1e-3 * max(1, h["loss"]))
                self.assertGreater(sum(h["loss_node"] for h in m["history"]), 0)
            for bad in (("--tree-full", "plain", "--tree-view", "mix"),
                        ("--tree-full", "matched", "--tree-view", "local+ctxparse"),
                        ("--tree-full", "true", "--tree-view", "local", "--tree-mix-p", "0.3"),
                        ("--tree-full", "true", "--tree-view", "mix", "--tree-mix-p", "1.5"),
                        ("--tree-full", "true", "--tree-view", "mix", "--tree-eligible-only")):
                out = self.run_script(d, *bad, "--tree-cap", "0", "--tree-sources", "knights_knaves")
                self.assertNotEqual(out.returncode, 0, bad)
            for view in ("local+ctxparse", "mix"):                                         # justlogic: no block
                out = self.run_script(d, "--tree-full", "true", "--tree-view", view, "--tree-cap", "0")
                self.assertNotEqual(out.returncode, 0)
                self.assertIn("no statement block", out.stderr)

    def test_mix_resume(self):
        """MIX's epoch key survives a checkpoint: a run stopped after update 1 and resumed gives the uninterrupted
        run's updates (the second epoch's coins included) and predictions."""
        with tempfile.TemporaryDirectory() as d:
            self.setup_dir(d)
            common = ("--tree-full", "true", "--tree-view", "mix", "--tree-cap", "0", "--tree-sources",
                      "knights_knaves", "--epochs", "2", "--precision", "fp32master", "--seed", "3")
            a = self.run_script(d, *common, "--ckpt-every-min", "1000", out="a")
            self.assertEqual(a.returncode, 0, a.stderr[-2000:])
            b1 = self.run_script(d, *common, "--stop-at-step", "1", out="b")
            self.assertEqual(b1.returncode, 3, b1.stderr[-2000:])
            b2 = self.run_script(d, *common, "--ckpt-every-min", "1000", out="b")
            self.assertEqual(b2.returncode, 0, b2.stderr[-2000:])
            self.assertIn("resumed from", b2.stdout)
            self.assertEqual(re.findall(r"^mix pass (\d+): ", b2.stdout, re.M), ["0", "1"])
            key = lambda o: [(h["step"], h["loss"], h["seqs"], h["loss_final"], h["loss_node"])
                             for h in json.loads(Path(f"{d}/{o}/metrics.json").read_text())["history"]]
            self.assertEqual(key("a"), key("b"))
            self.assertEqual([r[0] for r in key("a")], [1, 2, 3, 4])
            self.assertEqual(Path(d + "/a/preds_v.jsonl").read_text(), Path(d + "/b/preds_v.jsonl").read_text())

    def test_existing_arms_loop_match_ref(self):
        """TFL and GFP-S through the training loop: VIEWS_REF's losses, update for update."""
        with tempfile.TemporaryDirectory() as d:
            if head_tree(d + "/h", VIEWS_REF) is None:
                self.skipTest("not a git checkout")
            self.setup_dir(d)
            for mode, view in (("true", "local"), ("plain", "statements")):
                res = {}
                for name, script in (("ref", d + "/h/scripts/train_student.py"), ("now", SCRIPT)):
                    out = self.run_script(d, "--tree-full", mode, "--tree-view", view, "--tree-cap", "0",
                                          "--tree-sources", "knights_knaves", "--epochs", "2", "--precision", "bf16",
                                          "--seed", "3", script=script, out=f"{name}-{mode}")
                    self.assertEqual(out.returncode, 0, out.stderr[-2000:])
                    m = json.loads(Path(f"{d}/{name}-{mode}/metrics.json").read_text())
                    res[name] = ([(h["step"], h["loss"], h["seqs"]) for h in m["history"]], m["eval"]["v"],
                                 m["tree_view_stats"])
                self.assertEqual(res["ref"], res["now"], (mode, view))


def numgen_depth_items(path: Path) -> list[dict]:
    """The first numgen item of each depth d1-d4 and the second d4 item (d1: read / parse leaves only)."""
    items, per = [], {}
    with open(path) as f:
        for line in f:
            it = json.loads(line)
            d = it["item_id"].split("-")[2]
            per[d] = per.get(d, 0) + 1
            if it.get("source") == "numgen" and per[d] <= (2 if d == "d4" else 1):
                items.append(it)
    return items


@unittest.skipUnless(HAVE, "needs torch")
class TestGTF(unittest.TestCase):
    """GTF = GFP-S (--tree-full plain --tree-view statements) + scaffold fading (scaffold_p: a derived node asked with
    its parents' results under the statement block) + depth weights (depth_weight G)."""

    def build(self, item, seed=0, src="knights_knaves", p=None, g=0.0, coin_seed=None, epoch=0, mode="plain",
              view="statements", **kw):
        from cotdistill.student import build_full_tree
        st = {}
        ex = build_full_tree(item, mode=mode, view=view, rng=random.Random(seed), cap=kw.pop("cap", 0), stats=st,
                             sources={src}, scaffold_p=p, depth_weight=g,
                             mix_seed=seed if coin_seed is None else coin_seed, mix_epoch=epoch, **kw)
        return ex, st

    def check_gtf(self, item, seed, src="knights_knaves", p=1.0, epoch=0, coin_seed=None):
        """GTF vs GFP-S at the same seed: plain views byte-identical; scaffold views = GFP-S's text with "Known
        results:" and the parents' fact lines inserted after the statement block (no final question, no options, no
        leak); nodes, order, labels, soft targets (yes/no order included), weights and the final view identical.
        -> (kinds per node view, stats)."""
        from cotdistill.treeplan import LOCAL_HEADER, _fact_line, statement_block, view_leak
        block = statement_block(item)
        by = {sq["id"]: sq for sq in item["subqs"]}
        byq = {sq["question"]: sq for sq in item["subqs"]}
        opts = item["prompt"].split("\nOptions:\n", 1)[1].split("\n")
        g, st_g = self.build(item, seed, src)
        x, st = self.build(item, seed, src, p=p, epoch=epoch, coin_seed=coin_seed)
        self.assertEqual(len(x), len(g))
        self.assertEqual(dataclasses.astuple(x[0]), dataclasses.astuple(g[0]))         # final view unchanged
        self.assertEqual(x[0].text, item["prompt"] + "\n\nAnswer:")
        kinds = []
        for e, eg in zip(x[1:], g[1:]):
            head, q, stated = split_view(e)
            sq = byq[q]
            self.assertEqual(q, split_view(eg)[1])                                        # same node, same order
            self.assertEqual((e.labels, e.target, e.weight, e.kind, e.item_id),
                             (eg.labels, eg.target, eg.weight, eg.kind, eg.item_id))      # step-matched with GFP-S
            self.assertTrue(head.startswith(block))
            self.assertNotIn("Question:", head)
            self.assertNotIn("Options:", head)
            if src == "knights_knaves":
                self.assertFalse(any(o[3:] in e.text for o in opts))                     # no option text
            if stated:
                kinds.append("scaffold")
                par = [j for j in sq["depends_on"] if j in by and j != sq["id"]]
                self.assertTrue(par)
                self.assertFalse(sq.get("local_lines"))                                   # never a leaf
                self.assertEqual(head, block + "\n\n" + LOCAL_HEADER + "\n" + "\n".join(stated))
                self.assertEqual(sorted(stated), sorted(_fact_line(by[j]) for j in par))  # the parents, confident
                self.assertIsNone(view_leak(item, sq, [by[j] for j in par]))
                self.assertEqual(e.text.replace("\n\n" + LOCAL_HEADER + "\n" + "\n".join(stated), "", 1), eg.text)
            else:
                kinds.append("plain")
                self.assertEqual(e.text, eg.text)                                         # GFP-S byte for byte
        self.assertEqual(st["scaffold_views"], kinds.count("scaffold"))
        self.assertEqual(st["scaffold_lines"], sum(len(split_view(e)[2]) for e in x[1:]))
        self.assertEqual({k: v for k, v in st.items() if not k.startswith("scaffold_")}, st_g)
        self.assertEqual(sum(v for k, v in st.items() if k.startswith("scaffold_") and k not in
                             ("scaffold_views", "scaffold_lines")), st["views"])          # every node classified
        return kinds, st

    def test_fixture_rendering(self):
        n_derived = sum(not sq.get("local_lines") for sq in KK["subqs"])
        for seed in range(6):
            kinds, st = self.check_gtf(KK, seed, p=1.0)                                  # p 1: every eligible node
            self.assertEqual(kinds.count("scaffold"), st["scaffold_eligible"])
            self.assertEqual(st["scaffold_eligible"], n_derived)                          # fixture: all derived nodes
            self.assertEqual(st["scaffold_leaf"], len(KK["subqs"]) - n_derived)
            kinds, st = self.check_gtf(KK, seed, p=0.0)                                  # p 0: GFP-S exactly
            self.assertEqual(kinds.count("scaffold"), 0)
            a, _ = self.build(KK, seed, p=0.0)
            b, _ = self.build(KK, seed)
            self.assertEqual([dataclasses.astuple(e) for e in a], [dataclasses.astuple(e) for e in b])
        e = next(e for e in self.build(KK, 0, p=1.0)[0][1:] if split_view(e)[1].startswith("If Carol were a knight, "
                                                                                             "would Bob's role"))
        self.assertIn('- Henry says: "Bob is a knave."\n\nKnown results:\nQ: ', e.text)  # the §10 example (n12)

    def test_unconfident_and_leaking_parents(self):
        """Nodes with an unconfident parent or whose stated parents would give the answer away are always plain."""
        item = json.loads(json.dumps(KK))
        by = {sq["id"]: sq for sq in item["subqs"]}
        by["n9"]["truth"], by["n9"]["p_cot"] = None, 0.5                                  # n10 reads n9: unconfident
        people = re.findall(r"^- ([A-Z][a-z]+) says", item["prompt"], re.M)
        roles = [sq for sq in item["subqs"] if re.match(r"^Is [A-Z][a-z]+ a kn(?:ight|ave)\?$", sq["question"])]
        self.assertEqual({re.match(r"^Is (\w+)", sq["question"]).group(1) for sq in roles}, set(people))
        by["n24"]["depends_on"] = [sq["id"] for sq in roles]                               # every role stated: a leak
        kinds, st = self.check_gtf(item, 0, p=1.0)
        self.assertEqual(st["scaffold_leak_assignment"], 1)
        self.assertGreaterEqual(st["scaffold_unconfident"], 1)
        x, _ = self.build(item, 0, p=1.0)
        for q in (by["n24"]["question"], by["n10"]["question"]):
            self.assertEqual(split_view(next(e for e in x[1:] if split_view(e)[1] == q))[2], [])

    def test_fade_schedule(self):
        from cotdistill.treeplan import scaffold_fade
        T, F = 904, 0.5
        ps = [scaffold_fade(t, T, F) for t in range(T)]
        self.assertEqual(ps[0], 1.0)
        self.assertAlmostEqual(ps[226], 0.5)
        self.assertTrue(all(a >= b for a, b in zip(ps, ps[1:])))                         # non-increasing
        self.assertEqual(ps[452:], [0.0] * (T - 452))                                     # 0 from t = F x T on
        self.assertTrue(all(abs(p - (1 - t / (F * T))) < 1e-12 for t, p in enumerate(ps[:452])))
        self.assertEqual(scaffold_fade(0, T, 0.0), 0.0)
        self.assertEqual(scaffold_fade(10, 4, 1.0), 0.0)
        # the share of scaffold views among the eligible nodes is about p; the draws are nested in p (one uniform per
        # node), deterministic per (seed, epoch, item, node) whatever the training rng, and new per epoch and seed
        for p in (0.25, 0.5, 0.75):
            n = s = 0
            for coin in range(40):
                k, st = self.check_gtf(KK, 0, p=p, coin_seed=coin)
                n += st["scaffold_eligible"]
                s += k.count("scaffold")
            self.assertLess(abs(s / n - p), 0.07, (p, s / n))
        lo, _ = self.check_gtf(KK, 0, p=0.3, coin_seed=5)
        hi, _ = self.check_gtf(KK, 0, p=0.7, coin_seed=5)
        self.assertTrue(all(h == "scaffold" for l, h in zip(lo, hi) if l == "scaffold"))
        kinds = {}
        for coin in (0, 1):
            for epoch in (0, 1):
                kinds[coin, epoch] = self.check_gtf(KK, 0, p=0.5, coin_seed=coin, epoch=epoch)[0]
                again = self.check_gtf(KK, 7, p=0.5, coin_seed=coin, epoch=epoch)[0]      # other training rng
                self.assertEqual(again, kinds[coin, epoch])
                a, _ = self.build(KK, 0, p=0.5, coin_seed=coin, epoch=epoch)
                b, _ = self.build(KK, 0, p=0.5, coin_seed=coin, epoch=epoch)
                self.assertEqual([dataclasses.astuple(e) for e in a], [dataclasses.astuple(e) for e in b])
        self.assertNotEqual(kinds[0, 0], kinds[0, 1])                                       # epochs differ
        self.assertNotEqual(kinds[0, 0], kinds[1, 0])                                       # seeds differ

    def test_depth_weights(self):
        depth = {sq["question"]: sq["node_depth"] for sq in KK["subqs"]}
        g0, _ = self.build(KK, 0)
        for p in (None, 0.5):
            same, _ = self.build(KK, 0, p=p, g=0.0)                                       # G 0: GFP-S's weights
            self.assertEqual([e.weight for e in same], [e.weight for e in g0])
        for g in (1.0, 2.0):
            for kw, total in ((dict(), 1.0), (dict(node_weight=0.1), 0.1 * len(KK["subqs"])),
                              (dict(aux_total=2.0), 2.0)):
                for item, wi in ((KK, 1.0), (dict(KK, weight=0.5), 0.5)):
                    x, _ = self.build(item, 0, p=0.5, g=g, **kw)
                    self.assertEqual(x[0].weight, wi)                                     # final weight unchanged
                    self.assertAlmostEqual(sum(e.weight for e in x[1:]), total * wi)       # per-problem total kept
                    dw = sorted((depth[split_view(e)[1]], e.weight) for e in x[1:])
                    for (d1, w1), (d2, w2) in zip(dw, dw[1:]):
                        self.assertTrue(w1 < w2 if d1 < d2 else abs(w1 - w2) < 1e-15)     # increasing in depth
                    (d1, w1), (d2, w2) = dw[0], dw[-1]
                    self.assertAlmostEqual(w2 / w1, ((1 + d2) / (1 + d1)) ** g)
        # the same weights in the other full-tree views (any mode but control), e.g. TFL and GFP
        for mode, view in (("true", "local"), ("plain", "problem"), ("true", "problem")):
            x, _ = self.build(KK, 0, g=1.0, mode=mode, view=view)
            y, _ = self.build(KK, 0, mode=mode, view=view)
            self.assertAlmostEqual(sum(e.weight for e in x[1:]), sum(e.weight for e in y[1:]))
            self.assertEqual([e.target for e in x], [e.target for e in y])
            dw = sorted((depth[split_view(e)[1]], e.weight) for e in x[1:])
            self.assertTrue(all(w1 <= w2 for (_, w1), (_, w2) in zip(dw, dw[1:])))
        x, _ = self.build(KK, 0, g=1.0, mode="placebo", view="problem")
        self.assertTrue(all(e.weight == 0.0 for e in x[1:]))

    def test_same_nodes_as_gfps(self):
        """Step matching: GTF (p 0.5, G 1) has GFP-S's and GFP's nodes in the same order with the same P(Yes)
        (GFP-S: the same targets, yes/no order included) and the same per-problem weight; G 0 gives GFP-S's
        weights."""
        for seed in range(4):
            g, _ = self.build(KK, seed)
            f, _ = self.build(KK, seed, view="problem")
            x, _ = self.build(KK, seed, p=0.5, g=1.0)
            for ref in (g, f):
                self.assertEqual([split_view(e)[1] for e in x[1:]], [split_view(e)[1] for e in ref[1:]])
                self.assertEqual([round(p_yes(e), 12) for e in x[1:]], [round(p_yes(e), 12) for e in ref[1:]])
                self.assertEqual(x[0].text, ref[0].text)
                self.assertAlmostEqual(sum(e.weight for e in x[1:]), sum(e.weight for e in ref[1:]))
            self.assertEqual([e.target for e in x], [e.target for e in g])
            self.assertNotEqual([e.weight for e in x], [e.weight for e in g])

    @unittest.skipUnless(V3SL_VAL.exists(), "data/student_v3sl not built")
    def test_real_v3sl_items(self):
        items = first_items(V3SL_VAL, "knights_knaves", 5)
        self.assertEqual(len(items), 5)
        for it in items:
            for seed in (0, 1):
                kinds, st = self.check_gtf(it, seed, p=1.0)
                self.assertEqual(kinds.count("scaffold"), st["scaffold_eligible"])
                self.assertGreater(st["scaffold_eligible"], 0)
                self.check_gtf(it, seed, p=0.5)

    @unittest.skipUnless(NUMGEN_VAL.exists(), "data/student_numgen_tf not built")
    def test_numgen_items(self):
        items = numgen_depth_items(NUMGEN_VAL)
        self.assertEqual(len(items), 5)
        n = 0
        for it in items:
            ask = next(q["question"] for q in it["program"]["quantities"] if q["id"] == it["program"]["answer"])
            kinds, st = self.check_gtf(it, 0, "numgen", p=1.0)
            derived = [sq for sq in it["subqs"] if not sq.get("local_lines")]
            self.assertEqual(kinds.count("scaffold"), len(derived))                      # d1: none (leaves only)
            n += len(derived)
            for e in self.build(it, 0, "numgen", p=1.0)[0][1:]:
                self.assertNotIn(ask, e.text)                                             # no final question
            self.check_gtf(it, 0, "numgen", p=0.5)
        self.assertGreater(n, 0)

    def test_errors(self):
        for kw in (dict(view="local"), dict(view="problem"), dict(mode="true", view="mix"),
                   dict(mode="true", view="local+ctxparse"), dict(mode="matched", view="local")):
            with self.assertRaises(ValueError):
                self.build(KK, 0, p=0.5, **kw)
        for p in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                self.build(KK, 0, p=p)
        with self.assertRaises(ValueError):
            self.build(KK, 0, g=1.0, mode="control", view="problem")
        ex, _ = self.build(KK, 0, src="justlogic", p=0.5, g=1.0)                          # other sources: final only
        self.assertEqual([e.kind for e in ex], ["final"])

    def test_views_match_gtf_ref(self):
        """Every view and mode of GTF_REF (the problem-view modes, mc and eligible-only included, TFL, TFLM, TFLN, TFA,
        GFP-S, GFP, TFLP, MIX) gives exactly GTF_REF's examples and stats."""
        from cotdistill.student import build_full_tree
        with tempfile.TemporaryDirectory() as d:
            ref = head_student(d, GTF_REF, "cotdistill_gtfref")
            if ref is None:
                self.skipTest("not a git checkout")
            items = [(KK, "knights_knaves"), (TestDefaultUnchanged.TOY, None)]
            if V3SL_VAL.exists():
                items += [(it, "knights_knaves") for it in first_items(V3SL_VAL, "knights_knaves", 2)]
            if NUMGEN_VAL.exists():
                items += [(it, "numgen") for it in first_items(NUMGEN_VAL, "numgen", 2)]
            combos = [("true", "local", {}), ("matched", "local", {}), ("plain", "local", {}),
                      ("true", "ancestors", {}), ("plain", "statements", {}), ("true", "local+ctxparse", {}),
                      ("true", "mix", dict(mix_seed=3, mix_epoch=1)), ("true", "mix", dict(mix_p=0.2))]
            combos += [(mode, "problem", dict(eligible_only=elig, node_format=fmt))
                       for mode in ("true", "matched", "plain", "placebo") for elig in (False, True)
                       for fmt in ("yesno", "mc")] + [("control", "problem", {})]
            n = 0
            for item, src in items:
                for mode, view, extra in combos:
                    if src is None and view != "problem":
                        continue
                    for nw in (0.0, 0.1):
                        for cap in ((0, 3) if view == "problem" else (0,)):
                            kw = dict(mode=mode, view=view, cap=cap, sources={src} if src else None, node_weight=nw,
                                      permute_final=0.5, **extra)
                            sa, sb = {}, {}
                            a = ref.build_full_tree(item, rng=random.Random(n), stats=sa, **kw)
                            b = build_full_tree(item, rng=random.Random(n), stats=sb, **kw)
                            self.assertEqual([dataclasses.astuple(e) for e in a], [dataclasses.astuple(e) for e in b],
                                             (item["item_id"], mode, view, extra, nw, cap))
                            self.assertEqual(sa, sb)
                            n += 1
            self.assertGreaterEqual(n, 100)


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestTrainGTF(unittest.TestCase):
    OTHER = TestTrainCtxArms.OTHER
    setup_dir = TestTrainCtxArms.setup_dir
    run_script = TestTrainCtxArms.run_script
    GTF = ("--tree-full", "plain", "--tree-view", "statements", "--tree-cap", "0", "--tree-sources", "knights_knaves",
           "--tree-scaffold", "0.5", "--tree-depth-weight", "1.0")

    def test_training_on_cpu(self):
        """GTF trains two epochs with the local-step probe (6 problems, 3 per update: U 2, T 4, so p(t) is 1, .5, 0,
        0): pass 0 (the plan's counting pass, then epoch 0; each in its own problem order) has scaffold views, pass 1
        none; the same sequences per update as GFP-S (step-matched); the argument checks refuse what is not
        defined."""
        with tempfile.TemporaryDirectory() as d:
            self.setup_dir(d)
            common = ("--epochs", "2", "--precision", "bf16", "--seed", "3")
            out = self.run_script(d, *self.GTF, *common, "--eval-local", d + "/t.jsonl", out="gtf")
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            m = json.loads(Path(d + "/gtf/metrics.json").read_text())
            st = m["tree_view_stats"]
            self.assertIn("derived_nontrivial", m["eval"]["local_kk"])
            self.assertEqual(st["views"], 3 * len(KK["subqs"]))
            self.assertGreater(st["scaffold_views"], 0)
            self.assertLessEqual(st["scaffold_views"], st["scaffold_eligible"])
            passes = re.findall(r"^scaffold pass (\d+): (\d+) scaffold views of (\d+) eligible derived nodes, "
                                r"p\(t\) ([\d.]+) -> ([\d.]+) \(updates (\d+)-(\d+) of (\d+)\)$", out.stdout, re.M)
            self.assertEqual([x[0] for x in passes], ["0", "0", "1"])
            self.assertEqual(passes[0][3:], passes[1][3:])                                 # epoch 0's schedule
            self.assertEqual(int(passes[0][1]), st["scaffold_views"])
            self.assertEqual([(x[3], x[4], x[5], x[6], x[7]) for x in passes[1:]],
                             [("1.0000", "0.5000", "0", "1", "4"), ("0.0000", "0.0000", "2", "3", "4")])
            self.assertEqual(passes[2][1], "0")                                            # faded out
            self.assertNotIn("mix pass", out.stdout)
            for h in m["history"]:
                self.assertAlmostEqual(h["loss_final"] + h["loss_node"], h["loss"], delta=1e-3 * max(1, h["loss"]))
            self.assertGreater(sum(h["loss_node"] for h in m["history"]), 0)
            g = self.run_script(d, *self.GTF[:8], *common, out="gfps")
            self.assertEqual(g.returncode, 0, g.stderr[-2000:])
            mg = json.loads(Path(d + "/gfps/metrics.json").read_text())
            self.assertEqual([(h["step"], h["seqs"]) for h in m["history"]],
                             [(h["step"], h["seqs"]) for h in mg["history"]])              # same sequences per update
            self.assertEqual({k: v for k, v in st.items() if not k.startswith("scaffold_")}, mg["tree_view_stats"])
            self.assertNotEqual([h["loss"] for h in m["history"]], [h["loss"] for h in mg["history"]])
            for bad in (("--tree-full", "plain", "--tree-view", "local", "--tree-scaffold", "0.5"),
                        ("--tree-full", "true", "--tree-view", "mix", "--tree-scaffold", "0.5"),
                        ("--tree-full", "plain", "--tree-view", "problem", "--tree-scaffold", "0.5"),
                        ("--tree-full", "plain", "--tree-view", "statements", "--tree-scaffold", "-1"),
                        ("--tree-view", "statements", "--tree-scaffold", "0.5"),
                        ("--tree-depth-weight", "1.0"),
                        ("--tree-full", "control", "--tree-depth-weight", "1.0")):
                out = self.run_script(d, *bad, "--tree-cap", "0", "--tree-sources", "knights_knaves", out="bad")
                self.assertNotEqual(out.returncode, 0, bad)
            out = self.run_script(d, *self.GTF[:6], "--tree-scaffold", "0.5")             # justlogic: no block
            self.assertNotEqual(out.returncode, 0)
            self.assertIn("no statement block", out.stderr)

    def test_gtf_resume(self):
        """The fade survives a checkpoint: a run stopped after update 1 and resumed gives the uninterrupted run's
        updates (epoch 1's views included) and predictions."""
        with tempfile.TemporaryDirectory() as d:
            self.setup_dir(d)
            common = (*self.GTF, "--epochs", "2", "--precision", "fp32master", "--seed", "3")
            a = self.run_script(d, *common, "--ckpt-every-min", "1000", out="a")
            self.assertEqual(a.returncode, 0, a.stderr[-2000:])
            b1 = self.run_script(d, *common, "--stop-at-step", "1", out="b")
            self.assertEqual(b1.returncode, 3, b1.stderr[-2000:])
            b2 = self.run_script(d, *common, "--ckpt-every-min", "1000", out="b")
            self.assertEqual(b2.returncode, 0, b2.stderr[-2000:])
            self.assertIn("resumed from", b2.stdout)
            self.assertEqual(re.findall(r"^scaffold pass (\d+): ", b2.stdout, re.M), ["0", "1"])
            key = lambda o: [(h["step"], h["loss"], h["seqs"], h["loss_final"], h["loss_node"])
                             for h in json.loads(Path(f"{d}/{o}/metrics.json").read_text())["history"]]
            self.assertEqual(key("a"), key("b"))
            self.assertEqual([r[0] for r in key("a")], [1, 2, 3, 4])
            self.assertEqual(Path(d + "/a/preds_v.jsonl").read_text(), Path(d + "/b/preds_v.jsonl").read_text())

    def test_existing_arms_loop_match_gtf_ref(self):
        """GFP-S and MIX through the training loop: GTF_REF's losses, update for update."""
        with tempfile.TemporaryDirectory() as d:
            if head_tree(d + "/h", GTF_REF) is None:
                self.skipTest("not a git checkout")
            self.setup_dir(d)
            for mode, view in (("plain", "statements"), ("true", "mix")):
                res = {}
                for name, script in (("ref", d + "/h/scripts/train_student.py"), ("now", SCRIPT)):
                    out = self.run_script(d, "--tree-full", mode, "--tree-view", view, "--tree-cap", "0",
                                          "--tree-sources", "knights_knaves", "--epochs", "2", "--precision", "bf16",
                                          "--seed", "3", script=script, out=f"{name}-{mode}")
                    self.assertEqual(out.returncode, 0, out.stderr[-2000:])
                    m = json.loads(Path(f"{d}/{name}-{mode}/metrics.json").read_text())
                    res[name] = ([(h["step"], h["loss"], h["seqs"]) for h in m["history"]], m["eval"]["v"],
                                 m["tree_view_stats"])
                self.assertEqual(res["ref"], res["now"], (mode, view))


if __name__ == "__main__":
    unittest.main()
