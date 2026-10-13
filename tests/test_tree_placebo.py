"""--tree-target placebo (GFPSX / GFPQX: the placebo control of GFPS / GFPQ): every node keeps its view, option order,
weight and answer, the final question is untouched, only the node question is replaced by student.placebo_questions'
surface question about the problem text (a name / number / word / sentence that is or is not in the problem), with a
correct answer, keyed by the seed and the item (never the training rng). Checked on real numgen_tf, gsm8k_tf, kk_tf,
pronto_tf and lean items (those built): same count, weights and views as GFPS / GFPQ; answers by string search in the
statement block (placebo_text) and, for "no", in the whole prompt; no derived value asked; yes rate per problem; length;
seeding; the default and shuffled paths unchanged against PLACEBO_BASE_REF; flags; main(); the GFPSX / GFPQX arm lines.
CPU only, tiny random Qwen2 with the GPT-2 tokenizer."""
import copy
import dataclasses
import json
import random
import re
import statistics
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
    from test_tree_target import (GFPQ, GFPS, GSM_VAL, NUMGEN_VAL, PLAIN_VIEWS, SHUF, VIEWS, build, first_items,
                                  p_yes_t, real_items, run_child, with_mc)
    HAVE = True
except ImportError:
    HAVE = False

# The last commit before --tree-target placebo.
PLACEBO_BASE_REF = "0bfcf1c"
DATA = ROOT / "data"
SOURCES = (("student_numgen_tf", "numgen"), ("student_gsm8k_tf", "gsm8k_tf"), ("student_kk_tf", "knights_knaves"),
           ("student_pronto_tf", "pronto"), ("student_lean_tf_pilot", "lean_f1"))
PLAC = ["--tree-target", "placebo"]
SPLIT = "\n\nIntermediate question: "
NUM = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
TEMPLATES = {"name": re.compile(r"Is (\S+) mentioned in the problem\?"),
             "number": re.compile(r"Does the problem mention the number (\S+)\?"),
             "word": re.compile(r"Does the word '(\S+)' appear in the problem\?"),
             "sentence": re.compile(r"Is the sentence '(.+)' part of the problem\?", re.S)}


def val_items(n: int = 120) -> dict:
    """The first n val items of each Tree Forcing source whose data is built, by source."""
    out = {}
    for d, src in SOURCES:
        items = first_items(DATA / d / "val.jsonl", src, n)
        if items:
            out[src] = items
    return out


def nums(text: str) -> set:
    """The values of the number tokens of a text (1,200 -> 1200; 24.0 -> 24)."""
    out = set()
    for x in NUM.findall(text):
        v = float(x.replace(",", ""))
        out.add(int(v) if v == int(v) else v)
    return out


def tree_numbers(item: dict) -> set:
    """Independently of student.placebo_tree_numbers: every number in the node questions, the nodes' value / probe
    fields, the numgen program's quantities and the options."""
    out = set()
    for sq in item.get("subqs", []):
        if "id" in sq:
            out |= nums(sq["question"])
            out |= {v for k in ("value", "probe") for v in nums(str(sq.get(k))) if isinstance(sq.get(k), (int, float))}
    prog = item.get("program")
    for q in (prog.get("quantities") or []) if isinstance(prog, dict) else []:
        out |= nums(str(q.get("value"))) if isinstance(q.get("value"), (int, float)) else set()
    return out | nums(item["prompt"].partition("\nOptions:\n")[2])


def node_words(item: dict) -> set:
    return {w.lower() for sq in item["subqs"] if "id" in sq for w in re.findall(r"[A-Za-z]+", sq["question"])}


def parse(q: str) -> tuple[str, str]:
    """(template, the asked name / number / word / quote without its "...") of a placebo question."""
    for t, rx in TEMPLATES.items():
        m = rx.fullmatch(q)
        if m:
            return t, m.group(1).removesuffix("...") if t == "sentence" else m.group(1)
    raise AssertionError(f"not a placebo question: {q!r}")


def spelled(v) -> list[str]:
    from cotdistill.student import _pl_spelled
    return _pl_spelled(v) if isinstance(v, int) else []


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestDefaultUnchanged(unittest.TestCase):
    def test_library_matches_base(self):
        """Without target_kind "placebo": build_full_tree gives exactly PLACEBO_BASE_REF's examples and stats, for every
        full-tree mode and view without a target seed and for the plain views with --tree-target shuffled's
        (target_seed), on numgen_tf, gsm8k_tf, kk_tf, pronto_tf, lean, v3sl items and the K&K fixture."""
        with tempfile.TemporaryDirectory() as d:
            H = head_student(d, PLACEBO_BASE_REF, "cotdistill_tpbase")
            if H is None:
                self.skipTest("not a git checkout")
            n = 0
            for it in real_items():
                for mode, view in VIEWS:
                    for ts in ((None, 7) if (mode, view) in (("plain", v) for v in PLAIN_VIEWS) else (None,)):
                        for cap in (0, 10):
                            for seed in range(2):
                                kw = dict(cap=cap, sources={it.get("source")}, permute_final=0.5, target_seed=ts)
                                sa, sb = {}, {}
                                try:
                                    a = [dataclasses.astuple(e) for e in H.build_full_tree(
                                        it, mode=mode, view=view, rng=random.Random(seed), stats=sa, **kw)]
                                except ValueError as e:
                                    a = repr(e).replace("shuffled", "X")
                                b = build(it, mode, view, seed, stats=sb, **kw)
                                b = b.replace("shuffled", "X") if isinstance(b, str) else b
                                key = (it["item_id"], mode, view, ts, cap, seed)
                                self.assertEqual(a, b, key)
                                self.assertEqual(sa, sb, key)
                                n += 1
            self.assertGreater(n, 400)

    def test_training_matches_base(self):
        """GFPS and GFPSR on numgen_tf items (the K&K fixture when not built) through main(): the same history,
        evaluations, tree view stats and predictions as PLACEBO_BASE_REF's script."""
        with tempfile.TemporaryDirectory() as d:
            if head_tree(d + "/h", PLACEBO_BASE_REF) is None:
                self.skipTest("not a git checkout")
            tiny(d)
            items = first_items(NUMGEN_VAL, "numgen", 3)
            arm = [*GFPS[:-1], "numgen"] if items else GFPS
            items = items or [dict(KK, item_id=f"k{j}") for j in range(3)]
            Path(f"{d}/t.jsonl").write_text("\n".join(json.dumps(x) for x in items))
            for extra in ([], SHUF):
                runs = {}
                for which, script in (("base", d + "/h/scripts/train_student.py"), ("now", SCRIPT)):
                    out = f"{d}/{which}{len(extra)}"
                    r = subprocess.run([sys.executable, script, "--model", d + "/m", "--train", f"{d}/t.jsonl",
                                        "--eval", f"v={d}/t.jsonl", "--epochs", "2", "--items-per-update", "2",
                                        "--micro-bs", "4", "--max-len", "768", "--precision", "bf16", "--no-grad-ckpt",
                                        "--log-every", "1", "--seed", "3", "--permute-final", "0.5", "--out", out,
                                        *arm, *extra], capture_output=True, text=True, timeout=900, env=ENV)
                    self.assertEqual(r.returncode, 0, (which, extra, r.stderr[-2000:]))
                    m = json.loads(Path(out + "/metrics.json").read_text())
                    hist = [{k: v for k, v in h.items() if k not in ("s_per_step", "approx_tok_per_s")}
                            for h in m["history"]]
                    runs[which] = (hist, m["eval"], m.get("tree_view_stats"), Path(out + "/preds_v.jsonl").read_text())
                self.assertEqual(runs["base"], runs["now"], extra)
                self.assertGreater(len(runs["now"][0]), 0)


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestPlacebo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cotdistill.student import PlaceboBank
        cls.items = val_items()
        cls.banks = {src: PlaceboBank(v) for src, v in cls.items.items()}

    def all_items(self):
        for src, items in self.items.items():
            for it in items:
                yield src, it

    def pairs(self, n=12, seeds=range(2)):
        """(item, view, seed, GFPS examples, placebo examples, placebo stats) for the plain views each item has."""
        for src, items in self.items.items():
            for it in items[:n]:
                for view in PLAIN_VIEWS:
                    for seed in seeds:
                        kw = dict(cap=0, sources={src}, permute_final=0.5)
                        a = build(it, "plain", view, seed, **kw)
                        if isinstance(a, str):          # e.g. statement_block on a gsm8k_tf item
                            continue
                        st = {}
                        b = build(it, "plain", view, seed, target_seed=12, target_kind="placebo",
                                  placebo_bank=self.banks[src], stats=st, **kw)
                        yield it, view, seed, a, b, st

    def test_all_sources_built(self):
        """Every built Tree Forcing source is in the checks (all five in this checkout)."""
        self.assertEqual(set(self.items), {s for d, s in SOURCES if (DATA / d / "val.jsonl").exists()})
        if NUMGEN_VAL.exists():
            self.assertGreaterEqual(len(self.items), 2)

    def test_same_count_weights_views_as_gfps(self):
        """Same examples in the same order: the final question (text and target, permute_final 0.5 included: rng
        lockstep) and, for every node view, the context, the yes/no option order, labels, weight, kind, item and target
        (the data's node targets are hard, so the placebo's correct hard answer gives the same vector); only the
        question differs from the node's. Every source and plain view it has."""
        n, sources = 0, set()
        for it, view, seed, a, b, st in self.pairs():
            key = (it["item_id"], view, seed)
            self.assertEqual(len(a), len(b), key)
            self.assertEqual(a[0], b[0], key)
            self.assertEqual(a[0][4], "final")
            nodes = 0
            for x, y in zip(a[1:], b[1:]):
                self.assertEqual(x[1:], y[1:], key)              # labels, target, weight, kind, item
                cx, _, qx = x[0].partition(SPLIT)
                cy, _, qy = y[0].partition(SPLIT)
                self.assertEqual(cx, cy, key)                     # the view's context
                self.assertEqual(qx.split("\n", 1)[1], qy.split("\n", 1)[1], key)   # options (yes/no order), cue
                self.assertNotEqual(qx.split("\n")[0], qy.split("\n")[0], key)
                parse(qy.split("\n")[0])
                nodes += 1
            self.assertGreater(nodes, 0, key)
            self.assertEqual(st["placebo_views"], nodes)
            self.assertEqual(st["placebo_yes"], st["real_yes"])
            self.assertEqual(sum(st[f"placebo_{t}"] for t in TEMPLATES), nodes)
            sources.add(it["source"])
            n += 1
        self.assertGreater(n, 10)
        self.assertEqual(sources, set(self.items))

    def test_answers_correct_by_string_search(self):
        """Yes iff the name / word (whole word, case-sensitive) / number (token value) / quote (exact substring) is in
        the statement block (placebo_text); a "no" token is not in the whole prompt either (any case for names and
        words, no number word for numbers), so the answer holds in every view (GFPQ shows the final question)."""
        from cotdistill.student import placebo_questions, placebo_text
        n, seen = 0, set()
        for src, it in self.all_items():
            text, prompt = placebo_text(it), it["prompt"]
            for q, yes, t in placebo_questions(it, 12, self.banks[src]).values():
                tt, x = parse(q)
                self.assertEqual(tt, t, q)
                key = (it["item_id"], q, yes)
                if t in ("name", "word"):
                    self.assertEqual(bool(re.search(rf"\b{re.escape(x)}\b", text)), yes, key)
                    if not yes:
                        self.assertIsNone(re.search(rf"\b{re.escape(x)}\b", prompt, re.I), key)
                elif t == "number":
                    v = nums(x).pop()
                    self.assertEqual(v in nums(text), yes, key)
                    if not yes:
                        self.assertNotIn(v, nums(prompt), key)
                        for w in spelled(v):
                            self.assertIsNone(re.search(rf"\b{w}\b", prompt, re.I), key)
                else:
                    self.assertEqual(x in text, yes, key)
                    if not yes:
                        self.assertNotIn(x, prompt, key)
                n += 1
                seen.add((t, yes))
        self.assertGreater(n, 1000)
        self.assertGreaterEqual(len(seen), 6)

    def test_no_derived_values_asked(self):
        """A number question never asks a tree number (node questions, node value / probe, numgen quantities, options);
        no placebo question holds a tree number that the statement block does not state (a derived value, a wrong
        probe or an option); a name or word question never uses a word of any node question; a "no" sentence holds no
        tree number at all."""
        from cotdistill.student import placebo_questions, placebo_text
        n, with_derived = 0, set()
        for src, it in self.all_items():
            tree, nw, text = tree_numbers(it), node_words(it), placebo_text(it)
            derived = tree - nums(text)
            if derived:
                with_derived.add(src)
            for q, yes, t in placebo_questions(it, 12, self.banks[src]).values():
                key = (it["item_id"], q)
                self.assertFalse(nums(q) & derived, key)
                _, x = parse(q)
                if t == "number":
                    self.assertNotIn(nums(x).pop(), tree, key)
                if t in ("name", "word"):
                    self.assertNotIn(x.lower(), nw, key)
                    self.assertNotIn(x.lower() + "s", nw, key)
                if t == "sentence" and not yes:
                    self.assertFalse(nums(x) & tree, key)
                n += 1
        self.assertGreater(n, 1000)
        self.assertEqual(with_derived, set(self.items) & {"numgen", "gsm8k_tf", "lean_f1"})

    def test_balanced_yes_rate_and_template_mix(self):
        """Per problem the placebo yes rate is the real nodes' (every node keeps its answer). Over each source the
        template does not tell the answer: each template's yes share is within .1 of the source's yes rate (templates
        asked 50 times or more)."""
        from cotdistill.student import _node_p, placebo_questions
        for src, items in self.items.items():
            per, ys = {}, []
            for it in items:
                m = placebo_questions(it, 12, self.banks[src])
                real = [_node_p(sq) >= 0.5 for sq in it["subqs"] if "id" in sq and _node_p(sq) is not None]
                self.assertEqual(sum(y for _, y, _ in m.values()), sum(real), it["item_id"])
                self.assertEqual(len(m), len(real), it["item_id"])
                for _, y, t in m.values():
                    per.setdefault(t, []).append(y)
                    ys.append(y)
            rate = sum(ys) / len(ys)
            for t, v in per.items():
                if len(v) >= 50:
                    self.assertAlmostEqual(sum(v) / len(v), rate, delta=0.1, msg=(src, t, len(v)))

    def test_length_close_to_node_questions(self):
        """Whitespace words: the placebo mean within max(3.5, 25%) of the node questions' mean per source (pronto's node
        questions, 4-6 words, are shorter than the shortest template, 6 words); and a long node gets a long question:
        the placebo length rises with the node's (Spearman-like check on halves)."""
        from cotdistill.student import placebo_questions
        for src, items in self.items.items():
            real, plc = [], []
            for it in items:
                by = {sq["id"]: sq for sq in it["subqs"] if "id" in sq}
                for nid, (q, _, _) in placebo_questions(it, 12, self.banks[src]).items():
                    real.append(len(by[nid]["question"].split()))
                    plc.append(len(q.split()))
            mr, mp = statistics.mean(real), statistics.mean(plc)
            self.assertLess(abs(mr - mp), max(3.5, 0.25 * mr), (src, mr, mp))   # pronto: 4.8 vs 7.8 words
            med = statistics.median(real)
            lo = [p for r, p in zip(real, plc) if r < med]
            hi = [p for r, p in zip(real, plc) if r > med]
            if lo and hi:
                self.assertGreaterEqual(statistics.mean(hi), statistics.mean(lo), src)

    def test_distinct_within_problem(self):
        """A problem's placebo questions repeat rarely, only when its text runs out of unused names / numbers / words /
        sentences: at most 10% repeats per source (pronto, about 7%: one name per problem that no node mentions, the
        distractor entity, so "Is <it> mentioned?" may come twice; template counts do not look at the answers)."""
        from cotdistill.student import placebo_questions
        for src, items in self.items.items():
            tot = rep = 0
            for it in items:
                qs = [q for q, _, _ in placebo_questions(it, 12, self.banks[src]).values()]
                tot += len(qs)
                rep += len(qs) - len(set(qs))
            self.assertLessEqual(rep, 0.1 * tot, src)
            if src in ("numgen", "gsm8k_tf"):
                self.assertEqual(rep, 0, src)

    def test_seeded_not_training_rng(self):
        """A node's placebo question depends on the seed, the item and the bank only: the same with other training rng
        seeds (other yes/no orders), with a cap (another node subset), in another process (PYTHONHASHSEED), with the
        items in another order in the bank; another seed gives other questions; nothing is drawn from the training
        rng (the generator's state after build_full_tree is GFPS's)."""
        from cotdistill.student import PlaceboBank, build_full_tree, placebo_questions
        differ = total = 0
        for src, items in self.items.items():
            bank = self.banks[src]
            rbank = PlaceboBank(list(reversed(items)))
            for it in items[:4]:
                view = "statements+question" if src == "gsm8k_tf" else "statements"
                m = placebo_questions(it, 5, bank)
                self.assertEqual(placebo_questions(it, 5, rbank), m)
                differ += placebo_questions(it, 6, bank) != m
                total += 1
                allowed = {}
                for sq in it["subqs"]:
                    if sq.get("id") in m:
                        allowed.setdefault(sq["question"], set()).add(m[sq["id"]][0])
                for rseed in range(2):
                    for cap in (0, 5):
                        r1, r2 = random.Random(rseed), random.Random(rseed)
                        a = build_full_tree(it, mode="plain", view=view, rng=r1, cap=cap)
                        b = build_full_tree(it, mode="plain", view=view, rng=r2, cap=cap, target_seed=5,
                                            target_kind="placebo", placebo_bank=bank)
                        self.assertEqual(r1.random(), r2.random())
                        self.assertEqual(len(a), len(b))
                        for x, y in zip(a[1:], b[1:]):
                            qx = x.text.partition(SPLIT)[2].split("\n")[0]
                            qy = y.text.partition(SPLIT)[2].split("\n")[0]
                            self.assertIn(qy, allowed[qx])            # the node's own placebo question
        self.assertGreater(differ, 0.8 * total)
        it = KK
        code = ("import json, sys; sys.path.insert(0, 'src'); from cotdistill.student import placebo_questions; "
                "it = json.load(open('tests/fixtures/kk_local_item.json')); "
                "print(json.dumps(placebo_questions(it, 5), sort_keys=True))")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                             env={**ENV, "PYTHONHASHSEED": "123"}, check=True)
        self.assertEqual(json.loads(out.stdout), {k: list(v) for k, v in placebo_questions(it, 5).items()})

    def test_fixture_without_bank(self):
        """Without a bank (None) the K&K fixture still gets a correct placebo for every node ("no" sentences need a
        bank; "no" words come from the fixed pool)."""
        from cotdistill.student import placebo_questions, placebo_text
        m = placebo_questions(KK, 3)
        text = placebo_text(KK)
        self.assertEqual(len(m), sum(1 for sq in KK["subqs"] if "id" in sq))
        for q, yes, t in m.values():
            self.assertFalse(t == "sentence" and not yes, q)
            _, x = parse(q)
            if t == "sentence":
                self.assertTrue(x in text)

    def test_refused(self):
        """Outside the plain views / yes-no format, with eligible_only or a scaffold, an unknown target kind, an asked
        multiple-choice node, an item without a placebo text."""
        from cotdistill.student import build_full_tree
        bad = [dict(mode="true"), dict(mode="matched"), dict(mode="placebo"), dict(mode="control"),
               dict(mode="plain", view="local"), dict(mode="true", view="local"), dict(mode="plain", node_format="mc"),
               dict(mode="plain", eligible_only=True), dict(mode="plain", view="statements", scaffold_p=0.5)]
        for kw in bad:
            with self.assertRaises(ValueError, msg=kw):
                build_full_tree(KK, rng=random.Random(0), cap=0, target_seed=1, target_kind="placebo", **kw)
        with self.assertRaises(ValueError):
            build_full_tree(KK, mode="plain", rng=random.Random(0), cap=0, target_seed=1, target_kind="nope")
        with self.assertRaises(ValueError):
            build_full_tree(with_mc(KK), mode="plain", view="statements", rng=random.Random(0), cap=0, target_seed=1,
                            target_kind="placebo")
        odd = copy.deepcopy(KK)
        odd["prompt"] = "No statement block here.\n\nQuestion: ?\nOptions:\nA) x\nB) y"
        with self.assertRaises(ValueError):
            build_full_tree(odd, mode="plain", view="problem", rng=random.Random(0), cap=0, target_seed=1,
                            target_kind="placebo")


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestTrainPlacebo(unittest.TestCase):
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
        """b (placebo) reads a's examples in every update: same context, option order, weight, kind and target; the
        node question replaced by a placebo one, the same in both epochs (a node keeps its question)."""
        self.assertEqual(len(a), len(b))
        per_epoch = [{}, {}]
        n = 0
        for u, ((ma, _), (mb, _)) in enumerate(zip(a, b)):
            ea, eb = [x for m in ma for x in m], [x for m in mb for x in m]
            self.assertEqual([x[1:] for x in ea], [x[1:] for x in eb])
            for x, y in zip(ea, eb):
                if x[3] == "final":
                    self.assertEqual(x, y)
                    continue
                cx, _, qx = x[0].partition(SPLIT)
                cy, _, qy = y[0].partition(SPLIT)
                self.assertEqual(cx, cy)
                self.assertEqual(qx.split("\n", 1)[1], qy.split("\n", 1)[1])
                parse(qy.split("\n")[0])
                per_epoch[2 * u // len(b)].setdefault((cx, qx.split("\n")[0]), set()).add(qy.split("\n")[0])
                n += 1
        self.assertGreater(n, 0)
        self.assertEqual(per_epoch[0], per_epoch[1])

    def test_gfpsx_gfpqx_updates(self):
        """Through main(): GFPSX reads GFPS's updates with placebo questions (numgen_tf items, or the K&K fixture),
        GFPQX GFPQ's (gsm8k_tf items); GFPSX packed (--tree-pack 4) reads the same examples per update as unpacked."""
        ng = first_items(NUMGEN_VAL, "numgen", 4)
        gsm = first_items(GSM_VAL, "gsm8k_tf", 4)
        cases = [("ng", ng, [*GFPS[:-1], "numgen"]) if ng else
                 ("kk", [dict(KK, item_id=f"k{j}") for j in range(3)], GFPS)]
        if gsm:
            cases.append(("gsm", gsm, [*GFPQ[:-1], "gsm8k_tf"]))
        for name, items, arm in cases:
            t = self.data(name, items)
            a = run_child([*self.common(t, name + "a"), *arm], self.d)
            b = run_child([*self.common(t, name + "b"), *arm, *PLAC], self.d)
            self.check_updates(a, b)
            if "statements" in arm:
                c = run_child([*self.common(t, name + "c"), *arm, *PLAC, "--tree-pack", "4"], self.d)
                self.assertEqual(len(b), len(c))
                for (mb, _), (mc, _) in zip(b, c):
                    self.assertEqual(sorted(x for m in mb for x in m), sorted(x for m in mc for x in m))

    def test_stats_args_and_flags(self):
        t = self.data("kk", [dict(KK, item_id=f"k{j}") for j in range(3)])
        r = subprocess.run([sys.executable, SCRIPT, *self.common(t, "o"), *GFPS, *PLAC], capture_output=True,
                           text=True, timeout=900, env=ENV)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("placebo bank: knights_knaves 3 problems", r.stdout)
        m = json.loads(Path(self.d + "/o/metrics.json").read_text())
        self.assertEqual(m["args"]["tree_target"], "placebo")
        st = m["tree_view_stats"]
        self.assertEqual(st["placebo_views"], st["views"])
        self.assertEqual(st["placebo_yes"], st["real_yes"])
        self.assertEqual(sum(st[f"placebo_{k}"] for k in TEMPLATES), st["views"])
        self.assertGreater(st["placebo_words"], 0)
        mc = self.data("mc", [dict(with_mc(KK), item_id=f"k{j}") for j in range(2)])
        for bad, msg, train in ((PLAC, "need --tree-full", t),
                                (["--tree-full", "true", *PLAC], "--tree-target placebo: for --tree-full plain", t),
                                (["--tree-full", "control", *PLAC], "--tree-target placebo: for --tree-full plain", t),
                                (["--tree-full", "plain", "--tree-view", "local", *PLAC], "for --tree-full plain", t),
                                (["--tree-full", "plain", "--node-format", "mc", *PLAC], "with yes/no nodes", t),
                                (["--tree-full", "plain", "--tree-eligible-only", *PLAC], "without --tree-eligible-only",
                                 t),
                                ([*GFPS, "--tree-scaffold", "0.5", *PLAC], "or --tree-scaffold", t),
                                ([*GFPS, *PLAC], "has multiple-choice nodes", mc)):
            r = subprocess.run([sys.executable, SCRIPT, *self.common(train, "bad"), *bad], capture_output=True,
                               text=True, timeout=900, env=ENV)
            self.assertNotEqual(r.returncode, 0, bad)
            self.assertIn(msg, r.stderr, bad)


class TestArmLines(unittest.TestCase):
    def test_gfpsx_gfpqx_are_gfps_gfpq_with_placebo_targets(self):
        sb = ROOT / "hpc/train_arm.sbatch"
        cmd = ('source <(sed -n "/^arm_config()/,/^}/p" "$0"); for a in GFPS GFPSX GFPQ GFPQX; do arm_config "$a"; '
               'echo "$F|$S|$T|$RLM|$DS|$DX"; done')
        out = subprocess.run(["bash", "-c", cmd, str(sb)], capture_output=True, text=True, check=True).stdout.split("\n")
        self.assertEqual(out[1], out[0] + " --tree-target placebo")
        self.assertEqual(out[3], out[2] + " --tree-target placebo")
        lines = sb.read_text().split("\n")
        i = next(j for j, x in enumerate(lines) if x.lstrip().startswith("GFPQR)"))
        self.assertTrue(lines[i + 1].lstrip().startswith("GFPSX)"))
        self.assertTrue(lines[i + 2].lstrip().startswith("GFPQX)"))


if __name__ == "__main__":
    unittest.main()
