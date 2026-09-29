import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cotdistill.readout import label_distribution, locate_span, trace_confidence  # noqa: E402


def tok(t, p, alts=None):
    alts = alts or [(t, p)]
    return {"token": t, "logprob": math.log(p) if p > 0 else -9999.0,
            "top_logprobs": [{"token": a, "logprob": (math.log(b) if b > 0 else -9999.0)} for a, b in alts]}


class TestReadout(unittest.TestCase):
    def test_label_after_marker_and_variants(self):
        toks = [tok("So", 0.9), tok(" ANSWER", 0.99), tok(":", 0.99),
                tok(" B", 0.6, [(" B", 0.6), ("B", 0.1), (" A", 0.2), ("\n", 0.05)])]
        d = label_distribution(toks, ["A", "B", "C"])
        self.assertEqual(d["pos"], 3)
        self.assertAlmostEqual(d["raw"]["B"], 0.7)
        self.assertAlmostEqual(d["probs"]["B"], 0.7 / 0.9)
        self.assertEqual(d["missing"], ["C"])
        self.assertAlmostEqual(d["bound"], 0.05)
        self.assertEqual(d["sampled"], "B")

    def test_censored_sentinel_is_not_zero_mass(self):
        toks = [tok("A", 1.0, [("A", 1.0), ("B", 0.0)])]
        d = label_distribution(toks, ["A", "B"], marker=None)
        self.assertEqual(d["missing"], ["B"])
        self.assertAlmostEqual(d["probs"]["A"], 1.0)

    def test_parenthesised_and_bold_labels(self):
        def tok(t, alts):
            return {"token": t, "logprob": alts[0][1], "top_logprobs": [{"token": a, "logprob": lp} for a, lp in alts]}
        for opener in ("(", "**", " ("):
            toks = [tok("ANSWER:", [("ANSWER:", 0.0)]), tok(opener, [(opener, 0.0)]),
                    tok("E", [("E", -0.1), ("B", -2.5)]), tok(")", [(")", 0.0)])]
            d = label_distribution(toks, ["A", "B", "E"], "ANSWER:")
            self.assertEqual(d["sampled"], "E")
            self.assertGreater(d["probs"]["E"], d["probs"]["B"])

    def test_fused_marker_token(self):
        toks = [tok("ANSWER: C", 0.9, [("ANSWER: C", 0.9)])]
        d = label_distribution(toks, ["C", "D"])                     # label fused with the marker token
        self.assertEqual((d["pos"], d["sampled"]), (0, "C"))
        toks = [tok("ANSWER:", 0.9), tok(" C", 0.8, [(" C", 0.8), (" D", 0.2)])]
        self.assertAlmostEqual(label_distribution(toks, ["C", "D"])["probs"]["C"], 0.8)

    def test_trace_confidence_orders_peaked_above_flat(self):
        peaked = [tok("x", 0.99, [("x", 0.99), ("y", 0.005), ("z", 0.005)])] * 40
        flat = [tok("x", 0.4, [("x", 0.4), ("y", 0.3), ("z", 0.3)])] * 40
        self.assertGreater(trace_confidence(peaked)["mean"], trace_confidence(flat)["mean"])

    def test_locate_span(self):
        toks = [tok(w, 0.9) for w in ["Sep", " 2", " to", " Sep", " 13", " is", " 11", " days", "."]]
        s = locate_span(toks, "Sep 13 is 11 days")
        self.assertEqual(s, (3, 8))


if __name__ == "__main__":
    unittest.main()
