"""scripts/build_fa_sf.py (StepFinder's protocol: one model per Who&When subset, trained on StepFinder's regenerated
trajectories of that subset): a build on synthetic raw data (readers, overlap by exact text / 8-grams / ids within and
across subsets, val by question group among non-overlapping questions, the disjoint variant as a byte-identical subset,
test items without the reference line and the > 62-step exclusion, the cap policy, determinism), the --trees converter
on a synthetic trees file (accepted and refused trees, MC nodes the trainer reads), and, when built, the real files.
CPU only, no model:

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/sev/bin/python -m unittest tests/test_build_fa_sf.py
"""
import collections
import contextlib
import importlib.util
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill import fa_public as fp  # noqa: E402
from cotdistill.treeplan import is_mc_node, mc_check, statement_block  # noqa: E402


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = load_script("build_fa_sf")
WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon "
         "phi chi psi omega red green blue black white orange purple yellow river mountain forest desert ocean").split()


def sentence(seed: int, n: int = 22) -> str:
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) for _ in range(n)).capitalize() + "?"


def uuid(k: int) -> str:
    return f"{k:08x}-aaaa-bbbb-cccc-{k:012x}"


QA = {i: sentence(100 + i) for i in range(1, 5)}            # Who&When Algorithm-Generated questions
QH = {i: sentence(200 + i) for i in range(1, 4)}            # Who&When Hand-Crafted questions
AGENTS = {"alg": ["Planner_Expert", "Computer_terminal", "Verification_Expert"],
          "hc": ["Orchestrator", "WebSurfer", "FileSurfer"]}


def sf_question(subset: str, q: int) -> str:
    """alg q0: a WW alg question + an attachment path (8-gram + id overlap, same subset); q1: a WW hc question verbatim
    (exact, other subset); q2: a WW alg GAIA id in an attachment path only (id overlap, same subset). hc q0: a WW alg
    question verbatim (exact, other subset); q1: a WW hc question + attachment (8-gram, same subset). Others: none."""
    if subset == "alg":
        if q == 0:
            return QA[1] + f"\n\n[Attachment File]: /data/GAIA/validation/{uuid(1)}.xlsx\nPlease analyze the file."
        if q == 1:
            return QH[1]
        if q == 2:
            return sentence(300) + f"\n\n[Attachment File]: /data/GAIA/validation/{uuid(2)}.png"
    else:
        if q == 0:
            return QA[3]
        if q == 1:
            return QH[2] + "\n\n[Attachment File]: /data/x.pdf"
    return sentence(400 + 50 * (subset == "hc") + q)


def make_raw(root: Path, n_q: dict | None = None, per_q: int = 3) -> None:
    n_q = n_q or {"alg": 12, "hc": 10}
    r = random.Random(0)
    for subset, sub in B.SUBSETS.items():
        d = root / "stepfinder/repo/data" / sub / "train"
        d.mkdir(parents=True)
        k = 0
        for q in range(n_q[subset]):
            for j in range(per_q):
                n = r.randint(4, 9)
                ag = AGENTS[subset]
                hist = [{"step_id": str(i), "role": "assistant", "name": ag[(i + j) % 3],
                         "content": f"{subset} q{q} t{j} step {i}: " + " ".join(r.choice(WORDS) for _ in
                                                                              range(r.randint(5, 60))),
                         "is_mistake": "0"} for i in range(n)]
                m = r.randrange(n)
                hist[m]["is_mistake"] = "1"
                (d / f"trajectory_{k:04d}.json").write_text(json.dumps({
                    "question": sf_question(subset, q), "ground_truth": f"ans{q}", "history": hist,
                    "mistake_agent": hist[m]["name"], "mistake_step": m, "mistake_reason": "the mistake_reason"}))
                k += 1
    wa = root / "who_and_when/Who&When/Algorithm-Generated"
    wa.mkdir(parents=True)
    for i, q in QA.items():
        n = 5 + i
        hist = [{"content": f"alg ww {i} step {s} " + " ".join(r.choice(WORDS) for _ in range(30)),
                 "name": AGENTS["alg"][s % 3], "role": "assistant"} for s in range(n)]
        (wa / f"{i}.json").write_text(json.dumps({
            "is_correct": False, "question": q, "question_ID": uuid(i), "level": 1, "ground_truth": "x",
            "history": hist, "mistake_agent": hist[i % n]["name"], "mistake_step": str(i % n),
            "mistake_reason": "r", "system_prompt": {}}))
    wh = root / "who_and_when/Who&When/Hand-Crafted"
    wh.mkdir(parents=True)
    for i, q in QH.items():
        n = 70 if i == 3 else 6 + i
        hist = [{"content": q, "role": "human"}] + [
            {"content": f"hc ww {i} step {s} " + " ".join(r.choice(WORDS) for _ in range(20)),
             "role": ["Orchestrator (thought)", "WebSurfer", "Orchestrator (-> WebSurfer)"][s % 3]}
            for s in range(1, n)]
        dec = 65 if i == 3 else 2
        (wh / f"{i}.json").write_text(json.dumps({
            "history": hist, "question": q, "groundtruth": "y", "is_corrected": False,
            "mistake_agent": hist[dec]["role"].split(" (")[0], "mistake_step": str(dec), "mistake_reason": "r",
            "question_ID": uuid(10 + i), "level": 1}))


def build(tmp: Path, *extra) -> dict:
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        B.main(["--raw", str(tmp / "raw"), "--out-prefix", str(tmp / "out/student_fa_sf"), "--eval-dir",
                str(tmp / "eval"),
                "--tokenizer", "no/such/tokenizer.json", "--compare-norm", "", *extra])
    for s in ("alg", "hc"):
        out[s] = json.loads((tmp / f"out/student_fa_sf_{s}/summary.json").read_text())
    return out


def lines(p: Path) -> list[str]:
    return p.read_text().splitlines()


class TestSyntheticBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        make_raw(cls.d / "raw")
        cls.sum = build(cls.d)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def items(self, rel: str) -> list[dict]:
        return [json.loads(x) for x in lines(self.d / rel)]

    def test_overlap_counts(self):
        """Hand counts: alg q0 (8-gram + id, same subset), q1 (exact, other), q2 (id, same); hc q0 (exact, other), q1
        (8-gram, same); 3 trajectories per question."""
        a, h = self.sum["alg"]["overlap"], self.sum["hc"]["overlap"]
        self.assertEqual(a["exact/same"], {"questions": 0, "trajectories": 0})
        self.assertEqual(a["exact/other"], {"questions": 1, "trajectories": 3})
        self.assertEqual(a["ngram/same"]["questions"], 1)
        self.assertEqual(a["id/same"], {"questions": 2, "trajectories": 6})
        self.assertEqual(a["any/either"], {"questions": 3, "trajectories": 9})
        self.assertEqual(h["exact/other"], {"questions": 1, "trajectories": 3})
        self.assertEqual(h["ngram/same"], {"questions": 1, "trajectories": 3})
        self.assertEqual(h["any/either"], {"questions": 2, "trajectories": 6})
        self.assertEqual(self.sum["alg"]["train"]["disjoint"], self.sum["alg"]["train"]["main"] - 9)

    def test_splits(self):
        for s in ("alg", "hc"):
            tr = self.items(f"out/student_fa_sf_{s}/train.jsonl")
            va = self.items(f"out/student_fa_sf_{s}/val.jsonl")
            n = 36 if s == "alg" else 30
            self.assertEqual(len(tr) + len(va), n)
            self.assertEqual(len(va), 6 if s == "alg" else 3)           # whole groups up to round(0.1 x n)
            qt = lambda its: {fp.norm_task(it["prompt"].split("\nTask: ")[1].split("\n")[0]) for it in its}  # noqa
            self.assertFalse(qt(tr) & qt(va))
            for it in va:                                                   # val questions overlap nothing
                self.assertNotIn("[Attachment File]", it["prompt"])
            self.assertEqual(sum(it["with_gt"] for it in tr + va) > 0, True)
            for it in tr + va:
                self.assertEqual(fp.audit_item(it), [])
                self.assertEqual(it["source"], "fa_public")
                statement_block(it)
            # disjoint: the same val file, train lines a subset (byte for byte) without any overlapping trajectory
            self.assertEqual((self.d / f"out/student_fa_sf_{s}_disj/val.jsonl").read_text(),
                             (self.d / f"out/student_fa_sf_{s}/val.jsonl").read_text())
            main = set(lines(self.d / f"out/student_fa_sf_{s}/train.jsonl"))
            disj = lines(self.d / f"out/student_fa_sf_{s}_disj/train.jsonl")
            self.assertTrue(set(disj) <= main)
            self.assertEqual(len(main) - len(disj), 9 if s == "alg" else 6)
            for x in disj:
                p = json.loads(x)["prompt"]
                self.assertNotIn(QH[1], p)
                self.assertNotIn(QA[1], p)
                self.assertNotIn(QA[3], p)
                self.assertNotIn(QH[2], p)
                self.assertNotIn(uuid(2), p)

    def test_test_files(self):
        alg = self.items("eval/fa_sf_ww_alg.jsonl")
        hc = self.items("eval/fa_sf_ww_hc.jsonl")
        self.assertEqual([it["fa"]["ww_file"] for it in alg], [f"{i}.json" for i in range(1, 5)])
        self.assertEqual([it["fa"]["ww_file"] for it in hc], ["1.json", "2.json"])         # 3.json: 70 steps
        self.assertEqual(self.sum["hc"]["test"]["excluded_over_62_steps"],
                         [{"file": "3.json", "n_steps": 70, "decisive_step": 65}])
        for it in alg + hc:
            self.assertFalse(it["with_gt"])
            self.assertNotIn(fp.GT_PREFIX, it["prompt"])
            self.assertEqual(fp.audit_item(it), [])
        self.assertTrue(hc[0]["prompt"].split("\n\n")[2].startswith("Step 0 [human]: "))
        self.assertEqual(hc[0]["gold_label"], "C")
        self.assertEqual(hc[0]["fa"]["mistake_agent"], hc[0]["fa"]["decisive_agent"])
        self.assertEqual(self.sum["hc"]["baselines"]["test_all"]["n"], 3)

    def test_deterministic_and_caps(self):
        with tempfile.TemporaryDirectory() as d2:
            d2 = Path(d2)
            make_raw(d2 / "raw")
            s2 = build(d2)
            for s in ("alg", "hc"):
                a = {Path(k).name: v for k, v in self.sum[s]["md5"].items()}
                b = {Path(k).name: v for k, v in s2[s]["md5"].items()}
                self.assertEqual(a, b)
            s3 = build(d2, "--budget", "300", "--subsets", "alg")             # a small budget cuts every log
            c = s3["alg"]["caps"]["train"]
            self.assertGreater(c["cut"]["items_with_a_cut_step"], 0)
            self.assertLess(c["cut"]["cap"]["mean"], 3000)


def tree_line(subset: str, stem: str, rec: dict, *, kind: str = "ok") -> dict:
    """A tree of src/cotdistill/fa_trees.output_record's shape for record rec; kind plants one defect."""
    n = len(rec["steps"])
    g = rec["dec"]
    ref = lambda k: fp.step_ref(rec, k)                                             # noqa: E731
    others = [k for k in range(n) if k != g][:2]
    ptr = [ref(k) for k in [others[0], g] + others[1:]]
    nodes = [{"id": "n1", "type": "answer_step", "question": "Which step states the team's final answer?",
              "options": ptr, "answer": 1, "parents": [], "cite_step": g, "quote": rec["steps"][g]["text"][:20],
              "option_steps": [others[0], g] + others[1:]},
             {"id": "n2", "type": "content", "question": "What did that step claim?",
              "options": ["a count", "a date", "a name"], "answer": 2, "parents": ["n1"], "cite_step": g,
              "quote": "zzz not in the log"},
             {"id": "n3", "type": "agent", "question": "Who wrote that step?",
              "options": ["Orchestrator", "WebSurfer"] if subset == "hc" else ["Planner_Expert", "Verification_Expert"],
              "answer": 0, "parents": ["n1"], "cite_step": None, "quote": ""}]
    checks = {"pass": True, "failures": []}
    if kind == "pointer":
        nodes[0]["options"][0] = f"Step {others[0]} [Nobody]"
    elif kind == "checks":
        checks = {"pass": False, "failures": ["quote_not_found: n2"]}
    elif kind == "parent":
        nodes[1]["parents"] = ["n9"]
    elif kind == "answer":
        nodes[2]["answer"] = 5
    elif kind == "cycle":
        nodes[0]["parents"] = ["n2"]
    elif kind == "leak":
        nodes[1]["question"] = "What does the mistake_step say?"
    elif kind == "final":
        nodes[1]["question"] = fp.Q_STEP.removeprefix("Question: ")
    return {"traj_id": f"{subset}/{stem}", "subset": subset, "teacher": "synthetic", "nodes": nodes, "checks": checks,
            "meta": {}}


class TestTrees(unittest.TestCase):
    def test_converter(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            make_raw(d / "raw")
            recs = {r["uid"]: r for r in fp.read_stepfinder(str(d / "raw/stepfinder/repo/data"), collections.Counter())}
            tdir = d / "trees"
            tdir.mkdir()
            kinds = ["ok"] * 6 + ["pointer", "checks", "parent", "answer", "cycle", "leak", "final"]
            for s in ("alg", "hc"):
                with open(tdir / f"{s}.jsonl", "w") as f:
                    for k, kind in enumerate(kinds):
                        stem = f"trajectory_{k:04d}"
                        f.write(json.dumps(tree_line(s, stem, recs[f"{B.UID_PREFIX[s]}-{stem}"], kind=kind)) + "\n")
            summ = build(d, "--trees", str(tdir / "{subset}.jsonl"))
            for s in ("alg", "hc"):
                t = summ[s]["trees"]
                self.assertEqual(t["lines"], len(kinds))
                self.assertEqual(t["items_with_tree"], 6)
                ref = {"pointer": 1, "tree_checks_failed": 1, "parents": 1, "mc": 1, "cycle": 1, "leak": 1,
                       "final_question": 1}
                for k, v in ref.items():
                    self.assertEqual(t["refused"].get(k), v, (s, k, t["refused"]))
                self.assertEqual(t["refused"]["no_tree"], (36 if s == "alg" else 30) - len(kinds))
                self.assertEqual(t["types"], {"agent": 6, "answer_step": 6, "content": 6})
                self.assertEqual(t["answer_step_is_decisive"], {"true": 6})
                main = [json.loads(x) for x in lines(d / f"out/student_fa_sf_{s}/train.jsonl")]
                t2 = [json.loads(x) for x in lines(d / f"out/student_fa_sf_{s}_t2/train.jsonl")]
                self.assertEqual(len(main), len(t2))
                for a, b in zip(main, t2):
                    self.assertEqual({k: v for k, v in a.items() if k not in ("subqs", "has_tree", "tree_source", "fa")},
                                     {k: v for k, v in b.items() if k not in ("subqs", "has_tree", "tree_source", "fa")})
                    self.assertEqual(b["has_tree"], bool(b["subqs"]))
                    for sq in b["subqs"]:
                        self.assertTrue(is_mc_node(sq))
                        mc_check(sq)
                        self.assertIn("node_depth", sq)
                    if b["subqs"]:
                        names = set(b["label_names"].values())
                        self.assertTrue(set(b["subqs"][0]["options"]) <= names)
                        self.assertEqual(b["subqs"][1]["depends_on"], ["n1"])
                        self.assertEqual(b["subqs"][1]["node_depth"], 1)
                disj = set(lines(d / f"out/student_fa_sf_{s}_disj_t2/train.jsonl"))
                self.assertTrue(disj <= set(lines(d / f"out/student_fa_sf_{s}_t2/train.jsonl")))
                self.assertEqual(lines(d / f"out/student_fa_sf_{s}_t2/val.jsonl"),
                                 lines(d / f"out/student_fa_sf_{s}_disj_t2/val.jsonl"))
                self.assertEqual(t["facts"]["quotes"], 12)                 # n1 and n2 of the six accepted trees
                self.assertEqual(t["facts"]["quote_visible"], 6)            # n2's quote is not in the log
            # the trainer reads the MC nodes (GFP-S view, one-hot targets)
            try:
                from cotdistill.student import build_full_tree
            except ImportError:
                return
            it = next(x for x in t2 if x["subqs"])
            ex = build_full_tree(it, mode="plain", view="statements", rng=random.Random(0), cap=0, subq_target="truth")
            mc = [e for e in ex if e.kind == "subq"]
            self.assertEqual(len(mc), 3)
            self.assertEqual([e.target.index(1.0) for e in mc], [sq["answer"] for sq in it["subqs"]])

    def test_read_trees_subset(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.jsonl"
            p.write_text(json.dumps({"traj_id": "hc/trajectory_0000", "nodes": []}) + "\n")
            with self.assertRaises(SystemExit):
                B.read_trees(p, "alg")
            self.assertEqual(list(B.read_trees(p, "hc")), ["han-trajectory_0000"])


@unittest.skipUnless((ROOT / "data/student_fa_sf_alg/summary.json").exists(), "data/student_fa_sf_* not built")
class TestBuiltData(unittest.TestCase):
    def test_files(self):
        """The registered build: md5s, sizes, the overlap numbers, the test files (Alg 126; HC 39 of 58 with the 19
        > 62-step trajectories listed), every item audited, disjoint lines a subset of the main train file."""
        want = {"alg": dict(sf=1564, q=87, exact=(11, 198), test=126, excl=0),
                "hc": dict(sf=2604, q=145, exact=(69, 1241), test=39, excl=19)}
        for s, w in want.items():
            summ = json.loads((ROOT / f"data/student_fa_sf_{s}/summary.json").read_text())
            for p, m in summ["md5"].items():
                self.assertEqual(B.bfp.md5(ROOT / p), m, p)
            ov = summ["overlap"]
            self.assertEqual((ov["distinct_questions"], ov["trajectories"]), (w["q"], w["sf"]))
            self.assertEqual(ov["exact/other"], {"questions": w["exact"][0], "trajectories": w["exact"][1]})
            self.assertEqual(ov["exact/same"]["questions"], 0)
            self.assertEqual(summ["test"]["n"], w["test"])
            self.assertEqual(len(summ["test"]["excluded_over_62_steps"]), w["excl"])
            tr = lines(ROOT / f"data/student_fa_sf_{s}/train.jsonl")
            va = lines(ROOT / f"data/student_fa_sf_{s}/val.jsonl")
            self.assertEqual(len(tr) + len(va), w["sf"])
            self.assertEqual((len(tr), len(va)), (summ["train"]["main"], summ["val"]["trajectories"]))
            disj = lines(ROOT / f"data/student_fa_sf_{s}_disj/train.jsonl")
            self.assertEqual(len(disj), summ["train"]["disjoint"])
            self.assertTrue(set(disj) <= set(tr))
            for x in tr[:300] + va:
                self.assertEqual(fp.audit_item(json.loads(x)), [])
            for x in lines(ROOT / f"data/eval/fa_sf_ww_{s}.jsonl"):
                it = json.loads(x)
                self.assertEqual(fp.audit_item(it), [])
                self.assertFalse(it["with_gt"])


if __name__ == "__main__":
    unittest.main()
