"""scripts/build_fa_sf_x.py (the public FA test sets for StepFinder's protocol, decontaminated against StepFinder's
questions): the overlap marks on synthetic records (exact text, 8-gram containment in both directions with every match
above the threshold, shared uuids, nothing for an unrelated task), the annotated / _sfclean split of a file and its
_gt file, the log containment of the Who&When trace check, and, when built, the real files (sfclean lines = the
source lines of the unmarked trajectories plus the marks; counts as registered in sf_protocol.md §8). CPU only:

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/sev/bin/python -m unittest tests/test_build_fa_sf_x.py
"""
import importlib.util
import json
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill import fa_public as fp  # noqa: E402


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


X = load_script("build_fa_sf_x")
WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon "
         "phi chi psi omega red green blue black white orange purple yellow river mountain forest desert ocean").split()


def sentence(seed: int, n: int = 24) -> str:
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) for _ in range(n))


def uuid(k: int) -> str:
    return f"{k:08x}-aaaa-bbbb-cccc-{k:012x}"


def rec(uid, task, subset="s", ids=(), steps=("one two three four five six seven eight nine ten",) * 3):
    return {"uid": uid, "task": task, "subset": subset, "ids": list(ids),
            "steps": [{"agent": "A", "text": t} for t in steps]}


class TestOverlap(unittest.TestCase):
    def setUp(self):
        self.q = {i: sentence(10 + i) for i in range(6)}
        self.refs = [rec("sf-a", self.q[0], "alg"), rec("sf-a2", self.q[0], "alg"),
                     rec("sf-b", self.q[1] + " [Attachment File]: /data/x.xlsx", "hc", ids=[uuid(7)]),
                     rec("sf-c", self.q[2], "hc", ids=[uuid(9)])]
        self.tests = [rec("t-exact", self.q[0]), rec("t-ngram", self.q[1] + " please answer briefly"),
                      rec("t-id", sentence(99), ids=[uuid(9)]), rec("t-none", self.q[5])]
        ref_ids = {uuid(7): {"sf-b"}, uuid(9): {"sf-c"}}
        self.m = X.overlap_marks(self.tests, self.refs, ref_ids, df=5, thr=0.5)
        self.by = {r["uid"]: r for r in self.refs}

    def test_criteria(self):
        m = self.m
        self.assertEqual(X.hit(m["t-exact"]), ["exact", "ngram"])
        self.assertEqual(m["t-exact"]["exact"], ["sf-a", "sf-a2"])          # every trajectory of the question
        self.assertEqual(m["t-exact"]["ngram"], ["sf-a", "sf-a2"])
        self.assertEqual(X.hit(m["t-ngram"]), ["ngram"])
        self.assertEqual(X.subsets(m["t-ngram"], self.by), ["hc"])
        self.assertEqual(X.hit(m["t-id"]), ["id"])
        self.assertEqual(X.hit(m["t-none"]), [])
        self.assertEqual(X.subsets(m["t-exact"], self.by), ["alg"])

    def test_all_matches_keeps_every_reference(self):
        idx = fp.OverlapIndex([("r1", self.q[3] + " " + self.q[4]), ("r2", self.q[3])], df=5, thr=0.5)
        best, keys = X.all_matches(idx, self.q[3])
        self.assertEqual(best, 1.0)
        self.assertEqual(sorted(keys), ["r1", "r2"])                          # match() returns only the best
        self.assertEqual(idx.match(self.q[3])[1], ["r2"])

    def test_reverse_direction(self):
        """Boilerplate of the references hides a match from the forward index; the test-indexed direction finds it."""
        common = sentence(500, 30)
        refs = [rec(f"r{i}", common + " " + sentence(600 + i, 3), "alg") for i in range(5)]
        tests = [rec("t", common + " " + sentence(700, 3))]
        m = X.overlap_marks(tests, refs, {}, df=5, thr=0.5)
        self.assertEqual(m["t"]["forward"], 0)
        self.assertEqual(m["t"]["reverse"], 5)
        self.assertEqual(X.hit(m["t"]), ["ngram"])

    def test_source_files_and_gt(self):
        items = [{"item_id": f"i{k}", "fa": {"uid": t["uid"]}, "prompt": "p"} for k, t in enumerate(self.tests)]
        lines = [json.dumps(it) for it in items]
        traces = {t["uid"]: (0.0, None) for t in self.tests}
        traces["t-exact"] = (0.03, "ww-alg-1")
        ww = X.overlap_marks(self.tests, [], {}, df=5, thr=0.5)
        out, clean = X.source_files(lines, items, self.m, ww, self.by, {}, traces)
        self.assertEqual(len(out), 4)
        self.assertEqual([json.loads(x)["item_id"] for x in clean], ["i3"])
        a = json.loads(out[0])["fa"]
        self.assertEqual(a["sf_overlap"], ["exact", "ngram"])
        self.assertEqual(a["sf_subsets"], ["alg"])
        self.assertEqual(a["ww_overlap"], [])
        self.assertEqual(a["ww_trace"], {"containment": 0.03, "ww": "ww-alg-1"})
        self.assertIsNone(json.loads(out[3])["fa"]["ww_trace"])
        g_items = [dict(it, item_id=it["item_id"] + "-gt") for it in items[1:]]     # a _gt file: same uids, subset
        _, g_clean = X.source_files([json.dumps(x) for x in g_items], g_items, self.m, ww, self.by, {}, traces)
        self.assertEqual([json.loads(x)["item_id"] for x in g_clean], ["i3-gt"])

    def test_trace_overlap(self):
        a = rec("a", "q", steps=(sentence(1), sentence(2)))
        same = rec("w1", "q", steps=(sentence(1), sentence(2), sentence(3)))
        other = rec("w2", "q", steps=(sentence(4), sentence(5)))
        self.assertEqual(X.trace_overlap(a, [other, same]), (1.0, "w1"))
        self.assertEqual(X.trace_overlap(a, [other])[0], 0.0)


class TestFiles(unittest.TestCase):
    """The built files (data/ is not committed: skipped when absent)."""
    COUNTS = {"wwpro": (618, 592), "aftraj": (163, 146), "agentracer": (498, 484), "traceelephant": (211, 79),
              "correct": (92, 89)}
    MARKS = ("sf_overlap", "sf_subsets", "ww_overlap", "ww_subsets", "ww_trace")

    def test_built(self):
        ev = ROOT / "data/eval"
        if not (ev / "fa_sf_x_summary.json").exists():
            self.skipTest("data/eval/fa_sf_x_* not built")
        for name, (n, n_clean) in self.COUNTS.items():
            src = X.SOURCES[name][1]
            full = X.read_lines(ev / f"fa_sf_x_{name}.jsonl")
            clean = X.read_lines(ev / f"fa_sf_x_{name}_sfclean.jsonl")
            source = X.read_lines(ev / f"{src}.jsonl")
            self.assertEqual((len(full), len(clean)), (n, n_clean), name)
            self.assertEqual(len(source), n)
            cl = set(clean)
            for a, b in zip(full, source):
                x, y = json.loads(a), json.loads(b)
                marks = {k: x["fa"].pop(k) for k in self.MARKS}
                self.assertEqual(x, y)                                   # the source item plus the marks only
                self.assertEqual(a in cl, not marks["sf_overlap"])
            for line in clean:
                self.assertEqual(json.loads(line)["fa"]["sf_overlap"], [])


if __name__ == "__main__":
    unittest.main()
