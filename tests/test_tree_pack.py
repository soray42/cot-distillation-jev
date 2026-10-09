"""--tree-pack (GFPSP): GFP-S node views of one problem read in one causal pass (student.Pack, pack_views, collate /
label_logits with several readout positions per sequence), and the default paths unchanged against the commit before
it (PACK_BASE_REF). CPU only, tiny random Qwen2 with the GPT-2 tokenizer (plus the Qwen3.5 tokenizer when the local
weights are there)."""
import copy
import dataclasses
import json
import random
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

try:
    import torch
    import transformers  # noqa: F401
    import test_student as ts
    from test_train_resume import run_inproc
    from test_tree_forcing import ENV, KK, SCRIPT, head_tree, tiny
    HAVE = True
except ImportError:
    HAVE = False

V3SL_VAL = ROOT / "data/student_v3sl/val.jsonl"
NUMGEN_VAL = ROOT / "data/student_numgen_tf/val.jsonl"
QWEN_TOK = ROOT / "results/weights/T1-Qwen3.5-2B-Base-student_v3-s2/model"
# The last commit before --tree-pack and the one-GPU token budget of the plain loops.
PACK_BASE_REF = "03ec878"
GFPS = ["--tree-full", "plain", "--tree-view", "statements", "--tree-cap", "0", "--tree-sources", "knights_knaves"]


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


def gfps(item, seed, **kw):
    from cotdistill.student import build_full_tree
    return build_full_tree(item, mode="plain", view="statements", rng=random.Random(seed), cap=kw.pop("cap", 0), **kw)


def packed(item, seed, tok, max_len, max_nodes, epoch=0, stats=None, **kw):
    from cotdistill.student import pack_rng, pack_views
    from cotdistill.treeplan import statement_block
    return pack_views(gfps(item, seed, **kw), tok, statement_block(item), max_len=max_len, max_nodes=max_nodes,
                      rng=pack_rng(seed, epoch, item["item_id"]), stats=stats)


def flipped(item: dict) -> dict:
    """The item with every node's answer reversed (truth, p_cot, p_fresh): the same inputs must follow."""
    it = copy.deepcopy(item)
    for sq in it["subqs"]:
        if sq.get("truth") is not None:
            sq["truth"] = not sq["truth"]
        for k in ("p_cot", "p_fresh", "p_commit"):
            if sq.get(k) is not None:
                sq[k] = 1.0 - sq[k]
    return it


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestPackViews(unittest.TestCase):
    """(a) packed vs unpacked: the same per-node examples (targets, weights, labels), the final question unchanged."""

    def items(self):
        return ([KK] + first_items(V3SL_VAL, "knights_knaves", 3) +
                [(it, {"sources": {"numgen"}}) for it in first_items(NUMGEN_VAL, "numgen", 3)])

    def test_same_nodes_targets_weights(self):
        from cotdistill.student import PACK_CUE, Pack, build_full_tree, pack_rng, pack_views, readouts, seq_len
        from cotdistill.treeplan import statement_block
        tok, _ = ts.tiny()
        n = 0
        for entry in self.items():
            item, kw = entry if isinstance(entry, tuple) else (entry, {})
            block = statement_block(item)
            for seed in range(3):
                for max_nodes, max_len in ((2, 1536), (3, 1536), (8, 1536), (64, 1536), (8, 300)):
                    rng = random.Random(seed)
                    g = build_full_tree(item, mode="plain", view="statements", rng=rng, cap=0, **kw)
                    state = rng.getstate()
                    ref = gfps(item, seed, **kw)
                    st = {}
                    p = pack_views(g, tok, block, max_len=max_len, max_nodes=max_nodes,
                                   rng=pack_rng(seed, 0, item["item_id"]), stats=st)
                    self.assertEqual(rng.getstate(), state)                               # training rng untouched
                    self.assertEqual(dataclasses.astuple(p[0]), dataclasses.astuple(ref[0]))   # final first, as is
                    self.assertEqual(ref[0].kind, "final")
                    nodes = [e for e in ref if e.kind == "subq"]
                    flat = readouts(p[1:])
                    key = lambda e: dataclasses.astuple(e)
                    self.assertEqual(sorted(map(key, flat)), sorted(map(key, nodes)))     # same views, targets, weights
                    self.assertAlmostEqual(sum(e.weight for e in flat), sum(e.weight for e in nodes), places=12)
                    self.assertEqual(st["pack_nodes"] + st["pack_single"] + st["pack_long"], len(nodes))
                    for e in p[1:]:
                        if isinstance(e, Pack):
                            self.assertTrue(2 <= len(e.nodes) <= max_nodes)
                            self.assertEqual(e.parts[0], e.nodes[0].text)                 # the first view as is
                            for part, nd in zip(e.parts[1:], e.nodes[1:]):
                                self.assertTrue(part.startswith(PACK_CUE))
                                self.assertEqual(block + part, nd.text)                   # its question chunk
                            self.assertLessEqual(seq_len(tok, e, max_len), max_len)
                        else:
                            self.assertEqual(e.kind, "subq")
                    n += 1
        self.assertGreaterEqual(n, 15)

    def test_order_is_seeded_per_epoch(self):
        from cotdistill.student import Pack
        tok, _ = ts.tiny()
        order = lambda p: [x.text for e in p[1:] for x in (e.nodes if isinstance(e, Pack) else [e])]
        a, b = packed(KK, 0, tok, 1536, 8), packed(KK, 0, tok, 1536, 8)
        self.assertEqual(order(a), order(b))                                              # deterministic
        self.assertNotEqual(order(a), order(packed(KK, 0, tok, 1536, 8, epoch=1)))        # new order each epoch
        self.assertNotEqual(order(a), [e.text for e in gfps(KK, 0)[1:]])                 # not the plan order

    def test_long_views_and_singles_stay_plain(self):
        tok, _ = ts.tiny()
        ref = gfps(KK, 0)
        st = {}
        p = packed(KK, 0, tok, 40, 8, stats=st)                  # block alone is longer than 40 tokens
        self.assertEqual(st["pack_long"], len(ref) - 1)
        self.assertEqual(sorted(map(dataclasses.astuple, p)), sorted(map(dataclasses.astuple, ref)))
        with self.assertRaises(ValueError):                      # not GFP-S views (the problem view)
            from cotdistill.student import build_full_tree, pack_views
            from cotdistill.treeplan import statement_block
            g = build_full_tree(KK, mode="plain", view="problem", rng=random.Random(0), cap=0)
            pack_views(g, tok, statement_block(KK), max_len=1536, max_nodes=4, rng=random.Random(0))


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestPackReadout(unittest.TestCase):
    """(b) readout positions: the last token of every node's "Answer:" cue, the first view's tokens are its unpacked
    tokens, and every readout equals the model's next-token logits on the prefix up to that position alone."""

    def check_positions(self, tok, p):
        from cotdistill.student import Pack, encode_tail, pack_ids, special_prefix, text_ids
        packs = [e for e in p if isinstance(e, Pack)]
        self.assertGreater(len(packs), 0)
        for pk in packs:
            ids, pos = pack_ids(tok, pk)
            self.assertEqual(len(pos), len(pk.nodes))
            self.assertEqual(pos[-1], len(ids) - 1)
            self.assertEqual(ids[:pos[0] + 1], encode_tail(tok, pk.nodes[0].text, 10 ** 6))    # = the unpacked view
            pre = list(special_prefix(tok))
            for k, part in enumerate(pk.parts):
                pre += text_ids(tok, part)
                self.assertEqual(ids[:pos[k] + 1], pre)
                self.assertTrue(tok.decode(ids[:pos[k] + 1]).endswith("\nAnswer:"))
                self.assertEqual(ids[pos[k]], text_ids(tok, pk.nodes[k].text)[-1])           # the unpacked readout token
                if k + 1 < len(pk.parts):                                                     # next: the next chunk
                    self.assertEqual(tok.decode(ids[pos[k] + 1:pos[k + 1] + 1]), pk.parts[k + 1])
        return packs

    def test_positions_and_logits(self):
        from cotdistill.student import Pack, collate, label_logits, pack_ids, readouts, text_parts
        tok, model = ts.tiny()
        model.eval()
        p = packed(KK, 1, tok, 512, 5)
        packs = self.check_positions(tok, p)
        single = gfps(KK, 2)[3]                                   # a plain node view of another draw
        batch = [p[0], packs[0], single, packs[-1]]
        b = collate(tok, batch, 512, "cpu")
        self.assertEqual(len(b["pos"]), len(readouts(batch)))
        body, head = text_parts(model)
        with torch.no_grad():
            z = label_logits(model, tok, batch, 512, {})
            self.assertEqual(len(z), len(readouts(batch)))
            ref = []
            for e in batch:
                if isinstance(e, Pack):
                    ids, pos = pack_ids(tok, e)
                    for q in pos:                                 # the prefix alone, no padding, no later chunk
                        h = body(input_ids=torch.tensor([ids[:q + 1]]), use_cache=False).last_hidden_state[0, -1]
                        ref.append(h @ head.weight[[tok.encode(" A")[0], tok.encode(" B")[0]]].T)
                else:
                    ref.append(label_logits(model, tok, [e], 512, {})[0])
        self.assertEqual(len(ref), len(z))
        for a, r in zip(z, ref):
            self.assertLess((a - r).abs().max().item(), 1e-4)
        with torch.no_grad():                                     # a pack's readouts do not depend on its batch
            alone = label_logits(model, tok, [packs[0]], 512, {})
        for a, r in zip(alone, z[1:1 + len(packs[0].nodes)]):
            self.assertLess((a - r).abs().max().item(), 1e-4)

    @unittest.skipUnless((QWEN_TOK / "tokenizer.json").exists(), "local Qwen3.5 tokenizer not here")
    def test_qwen_tokenizer_cue_tokens(self):
        """The Qwen3.5 pre-tokenizer may join ':' with the newlines after it; parts are tokenized one by one, so every
        cue's last token is the unpacked view's readout token."""
        tok = transformers.AutoTokenizer.from_pretrained(str(QWEN_TOK))
        for item in [KK] + first_items(V3SL_VAL, "knights_knaves", 2):
            self.check_positions(tok, packed(item, 0, tok, 1536, 8))


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestPackNoLabels(unittest.TestCase):
    """(c) no answer in the input: the packed inputs do not depend on the targets, no cue is followed by an answer."""

    def test_inputs_do_not_depend_on_targets(self):
        from cotdistill.student import PACK_CUE, Pack, letter_token_ids, pack_ids, readouts
        tok, _ = ts.tiny()
        labels = set(letter_token_ids(tok, ["A", "B"]))
        for item in [KK] + first_items(V3SL_VAL, "knights_knaves", 2):
            for seed in range(3):
                a, b = packed(item, seed, tok, 1536, 8), packed(flipped(item), seed, tok, 1536, 8)
                ta = [e.parts if isinstance(e, Pack) else e.text for e in a[1:]]
                self.assertEqual(ta, [e.parts if isinstance(e, Pack) else e.text for e in b[1:]])
                ra, rb = readouts(a[1:]), readouts(b[1:])
                self.assertTrue(any(x.target != y.target for x, y in zip(ra, rb)))
                for x, y in zip(ra, rb):                          # the answer flips, the input does not
                    self.assertEqual([round(v, 9) for v in x.target], [round(v, 9) for v in y.target[::-1]])
                for e in a[1:]:
                    if not isinstance(e, Pack):
                        continue
                    self.assertIsNone(re.search(r"Answer:[ \t]*(A|B|Yes|No)\b", e.text))
                    self.assertNotIn("A: Yes", e.text)
                    self.assertNotIn("A: No", e.text)
                    self.assertEqual(e.text.count("\nAnswer:"), len(e.nodes))
                    ids, pos = pack_ids(tok, e)
                    for q in pos[:-1]:
                        self.assertNotIn(ids[q + 1], labels)
                        self.assertTrue(tok.decode(ids[q + 1:]).startswith(PACK_CUE))


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestTrainPack(unittest.TestCase):
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

    def run_(self, *a, out="o"):
        d = self.d
        return subprocess.run([sys.executable, SCRIPT, "--model", d + "/m", "--train", d + "/t.jsonl",
                               "--eval", "v=" + d + "/t.jsonl", "--items-per-update", "2", "--epochs", "2",
                               "--micro-bs", "2", "--max-len", "256", "--precision", "fp32master", "--no-grad-ckpt",
                               "--log-every", "1", "--lr", "1e-3", "--out", f"{d}/{out}", *a],
                              capture_output=True, text=True, timeout=900, env=ENV)

    def test_training_and_flags(self):
        for extra in ([], ["--token-budget", "600"]):
            r = self.run_(*GFPS, "--tree-pack", "4", *extra)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            m = json.loads(Path(self.d + "/o/metrics.json").read_text())
            st = m["tree_view_stats"]
            self.assertGreater(st["pack_seqs"], 0)
            self.assertEqual(st["pack_nodes"] + st["pack_single"] + st["pack_long"], st["views"])
            self.assertIn("pack_seqs", r.stdout)
            self.assertTrue(any(h["reads"] > h["seqs"] for h in m["history"]))
            for h in m["history"]:
                self.assertAlmostEqual(h["loss_final"] + h["loss_node"], h["loss"], delta=1e-4 * max(1, h["loss"]))
                if h["reads"] > h["seqs"]:                    # (an update of final-only problems has no node term)
                    self.assertGreater(h["loss_node"], 0)
        for bad, msg in ((["--tree-pack", "1"], "at least 2"),
                         (["--tree-pack", "4", "--tree-scaffold", "0.5"], "without --tree-scaffold"),
                         (["--tree-pack", "4", "--chat-template"], "without --tree-scaffold, --fsdp or --chat-template")):
            r = self.run_(*GFPS, *bad)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn(msg, r.stderr)
        r = self.run_("--tree-full", "plain", "--tree-view", "problem", "--tree-pack", "4")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--tree-pack 4", r.stderr)
        r = self.run_("--tree-pack", "4")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("need --tree-full", r.stderr)

    def test_same_node_terms_per_update_as_gfps(self):
        """Through main(): every update of GFPSP reads out exactly GFPS's examples of that update (final questions and
        node views: text, weight, target), in fewer forward passes."""
        d = self.d
        common = ["--model", d + "/m", "--train", d + "/t.jsonl", "--eval", "v=" + d + "/t.jsonl", *GFPS,
                  "--items-per-update", "2", "--epochs", "2", "--micro-bs", "2", "--max-len", "256",
                  "--precision", "fp32master", "--no-grad-ckpt", "--log-every", "1", "--seed", "2"]
        a = run_inproc([*common, "--out", d + "/gfps"])
        b = run_inproc([*common, "--tree-pack", "6", "--out", d + "/gfpsp"])
        self.assertEqual(len(a), len(b))
        for (ma, _), (mb, _) in zip(a, b):
            self.assertEqual(sorted(x for m in ma for x in m), sorted(x for m in mb for x in m))
        self.assertLess(sum(len(m) for m, _ in b), sum(len(m) for m, _ in a))          # fewer forward passes

    def test_resume_with_packs(self):
        """A planned stop mid-epoch (the remaining updates, Packs included, pickled in the checkpoint) and the resumed
        run end where a straight run ends."""
        common = [*GFPS, "--tree-pack", "4", "--items-per-update", "1", "--seed", "1"]
        r = self.run_(*common, "--stop-at-step", "4", out="a")
        self.assertEqual(r.returncode, 3, r.stderr[-2000:])
        r = self.run_(*common, "--ckpt-every-min", "1000", out="a")
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("resumed from", r.stdout)
        r = self.run_(*common, out="b")
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        strip = lambda h: [{k: v for k, v in x.items() if k not in ("s_per_step", "approx_tok_per_s")} for x in h]
        ma, mb = (json.loads(Path(f"{self.d}/{o}/metrics.json").read_text()) for o in ("a", "b"))
        self.assertEqual(strip(ma["history"]), strip(mb["history"]))
        self.assertEqual(Path(self.d + "/a/preds_v.jsonl").read_text(), Path(self.d + "/b/preds_v.jsonl").read_text())


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestDefaultPathsUnchanged(unittest.TestCase):
    """Flags off: the same training as PACK_BASE_REF (history, evaluations, predictions), on data/student_v3sl items
    (K&K and other families) when built, else the fixture: A2 and G3 (plain and grouped loops), GFPS, and TF with a
    token budget (full-tree loop); and the library's readout and micro-batches on the same examples."""

    def test_training_matches_base(self):
        with tempfile.TemporaryDirectory() as d:
            if head_tree(d + "/h", PACK_BASE_REF) is None:
                self.skipTest("not a git checkout")
            tiny(d)
            items = (first_items(V3SL_VAL, "knights_knaves", 3) + first_items(V3SL_VAL, "justlogic", 2) +
                     first_items(V3SL_VAL, "sharc", 1)) or [dict(KK, item_id=f"k{j}") for j in range(3)]
            Path(d + "/t.jsonl").write_text("\n".join(json.dumps(x) for x in items))
            arms = {"A2": ["--final", "teacher", "--subq", "none", "--micro-bs", "2", "--grad-accum", "2"],
                    "G3": ["--subq", "cot", "--grouped-aux", "2", "--aux-kind", "cot", "--aux-weight", "0.5",
                           "--items-per-update", "2", "--micro-bs", "2"],
                    "GFPS": [*GFPS, "--items-per-update", "2", "--micro-bs", "4"],
                    "TFtb": ["--tree-full", "true", "--items-per-update", "2", "--token-budget", "1500"]}
            runs = {}
            for name, script in (("base", d + "/h/scripts/train_student.py"), ("now", SCRIPT)):
                for arm, extra in arms.items():
                    out = f"{d}/{name}-{arm}"
                    r = subprocess.run([sys.executable, script, "--model", d + "/m", "--train", d + "/t.jsonl",
                                        "--eval", "v=" + d + "/t.jsonl", "--eval-subq", d + "/t.jsonl", "--epochs", "1",
                                        "--max-len", "768", "--precision", "bf16", "--no-grad-ckpt", "--log-every", "1",
                                        "--seed", "3", "--permute-final", "0.5", "--out", out, *extra],
                                       capture_output=True, text=True, timeout=900, env=ENV)
                    self.assertEqual(r.returncode, 0, (name, arm, r.stderr[-2000:]))
                    m = json.loads(Path(out + "/metrics.json").read_text())
                    hist = [{k: v for k, v in h.items() if k not in ("s_per_step", "approx_tok_per_s")}
                            for h in m["history"]]
                    runs[name, arm] = (hist, m["eval"], Path(out + "/preds_v.jsonl").read_text(),
                                       Path(out + "/preds_subq.jsonl").read_text())
            for arm in arms:
                self.assertEqual(runs["base", arm], runs["now", arm], arm)
                self.assertGreater(len(runs["now", arm][0]), 0)

    def test_library_matches_base(self):
        from test_tree_forcing import head_student
        from cotdistill import student as S
        with tempfile.TemporaryDirectory() as d:
            H = head_student(d, PACK_BASE_REF, "cotdistill_packbase")
            if H is None:
                self.skipTest("not a git checkout")
            tok, model = ts.tiny()
            model.eval()
            items = [KK] + first_items(V3SL_VAL, "knights_knaves", 2)
            for it in items:
                for seed in range(2):
                    for view in ("statements", "problem"):
                        a = S.build_full_tree(it, mode="plain", view=view, rng=random.Random(seed), cap=0)
                        b = H.build_full_tree(it, mode="plain", view=view, rng=random.Random(seed), cap=0)
                        self.assertEqual([dataclasses.astuple(e) for e in a], [dataclasses.astuple(e) for e in b])
                    for max_len in (128, 1536):
                        ex = a[:9]
                        hb = [H.Example(*dataclasses.astuple(e)) for e in ex]
                        ca, cb = S.collate(tok, ex, max_len, "cpu"), H.collate(tok, hb, max_len, "cpu")
                        self.assertEqual(sorted(ca), sorted(cb))
                        for k in ca:
                            self.assertTrue(torch.equal(ca[k], cb[k]), k)
                        for budget in (0, 300, 2000):
                            self.assertEqual([[e.text for e in m] for m in S.micro_batches(tok, ex, max_len, 2, budget)],
                                             [[e.text for e in m] for m in H.micro_batches(tok, hb, max_len, 2, budget)])
                        with torch.no_grad():
                            za = S.label_logits(model, tok, ex, max_len, {}, hidden_out=(ha := []))
                            zb = H.label_logits(model, tok, hb, max_len, {}, hidden_out=(hb_ := []))
                        for x, y in zip(za + ha, zb + hb_):
                            self.assertTrue(torch.equal(x, y))


if __name__ == "__main__":
    unittest.main()
