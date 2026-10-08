"""GFP-S statement blocks for every source with a clean cut (treeplan.statement_block / problem_no_options,
treeplan.FINAL_QUESTION): knights_knaves and numgen byte-identical to the f27b56b definitions (frozen copies below,
compared on the fixture and on every local K&K / numgen item), the new sources' cuts, refusals for sources without a
clean cut, and no view_leak on the local data. CPU only, stdlib only (the build_full_tree check needs torch).

Data-backed tests read data/ of this checkout, or COTD_DATA_DIR when set (a worktree has no data/), and skip when
neither has the files."""
import json
import os
import random
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.treeplan import FINAL_QUESTION, problem_no_options, statement_block, view_leak  # noqa: E402

DATA = Path(os.environ.get("COTD_DATA_DIR") or ROOT / "data")
KK = json.loads((ROOT / "tests/fixtures/kk_local_item.json").read_text())

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


# ---- frozen copies of the f27b56b definitions (K&K and numgen only) -------------------------------------------------
_OLD_KK_STATEMENT = re.compile(r'^- [A-Z][a-z]+ says: ".*"$')


def old_statement_block(item: dict) -> str:
    src, p, iid = item.get("source"), item.get("prompt", ""), item.get("item_id")
    if src == "knights_knaves":
        head, sep, tail = p.partition("\n\nQuestion: ")
        pre, _, body = head.partition("\n\n")
        lines = body.split("\n")
        if (not sep or "\n\nQuestion: " in tail or not body or not all(_OLD_KK_STATEMENT.match(x) for x in lines)
                or item.get("local_preamble") not in (None, pre)):
            raise ValueError(f"{iid}: K&K prompt is not preamble + statement lines + Question")
        return head
    if src == "numgen":
        first, sep, _opts = p.partition("\nOptions:\n")
        prog = item.get("program") or {}
        ans = [q for q in prog.get("quantities", []) if q.get("id") == prog.get("answer")]
        ask = ans[0].get("question") if len(ans) == 1 else None
        if not (sep and ask and first.startswith("Question: ") and first.endswith(" " + ask) and "\n" not in first):
            raise ValueError(f"{iid}: numgen prompt is not 'Question: <body> <answer question>' + options")
        return first[len("Question: "):-len(ask) - 1]
    raise ValueError(f"{iid}: --tree-view statements has no statement block for source {src!r} "
                     "(defined for knights_knaves and numgen; restrict with --tree-sources)")


def old_problem_no_options(item: dict) -> str:
    old_statement_block(item)
    head, sep, _ = item["prompt"].partition("\nOptions:")
    if not sep:
        raise ValueError(f"{item.get('item_id')}: prompt has no options block")
    return head


def outcome(fn, item):
    try:
        return ("ok", fn(item))
    except ValueError:
        return ("raise", None)


NUMGEN = {"item_id": "numgen-x", "source": "numgen",
          "prompt": "Question: Ravi grows pumpkins on a farm. Ravi harvested 18 pumpkins in the first week. In the "
                    "second week, Ravi harvested half as many pumpkins as in the first week. How many pumpkins did Ravi "
                    "harvest in the second week?\nOptions:\nA) 36\nB) 9\nC) 18",
          "program": {"answer": "q2", "quantities": [
              {"id": "q1", "question": "How many pumpkins did Ravi harvest in the first week?"},
              {"id": "q2", "question": "How many pumpkins did Ravi harvest in the second week?"}]}}

# one prompt per new source, in the shape its builder writes (body, final question paragraph, options)
NEW = {
    "justlogic": ("Read the passage and judge the statement using only the passage and valid logical reasoning.\n\n"
                  "Passage: If it rains, the road is wet. It rains.\n\nStatement: The road is wet.",
                  "Question: Is the statement true, false, or uncertain given the passage?", "A) True\nB) False\nC) Uncertain"),
    "folio": ("Read the premises and judge the conclusion using only the premises and valid logical reasoning.\n\n"
              "Premises:\nAll cats are animals.\nTom is a cat.\n\nStatement: Tom is an animal.",
              "Question: Is the statement true, false, or uncertain given the premises?", "A) True\nB) False\nC) Uncertain"),
    "sharc": ("Rule text:\nYou can get the grant if you live in Wales.\n\nUser question: Can I get the grant?\n\n"
              "Scenario: I live in Cardiff.",
              "Question: Based only on the rule text, the scenario and the earlier answers, how should the user's "
              "question be answered?", "A) Yes\nB) No\nC) More information is needed before answering"),
    "returns": ("You are a customer-service agent. Apply the policy to the case.\n\nPolicy (retail returns).\n1. Clearance "
                "items are final sale.\nOtherwise: deny the return.\n\nCase:\n- Price: $497\n- Opened: no",
                "Question: What should the agent do?", "A) Give a full refund\nB) Deny the return"),
    "expense": ("You are a customer-service agent. Apply the policy to the case.\n\nPolicy (expense reimbursement).\n"
                "1. Meals are reimbursed up to $100.\nOtherwise: approve the full amount.\n\nCase (JSON; null = not "
                "stated):\n{\"category\": \"meal\", \"amount\": \"$80\"}",
                "Question: What should the agent do?", "A) Approve the full amount\nB) Reject the claim"),
    "subscription": ("You are a customer-service agent. Apply the policy to the case.\n\nPolicy (subscription "
                     "cancellations).\n1. Promotional plans are non-refundable.\nOtherwise: cancel at the end of the "
                     "current period with no refund.\n\nCase:\n- Plan: monthly",
                     "Question: What should the agent do?", "A) Give a full refund\nB) Cancel at period end"),
    "rule": ("Role: a letting-agency assistant.\nRental deposit returns policy (the earliest matching rule wins):\n"
             "* Rule 1: when all keys were returned -> return the full deposit\n* No rule matches -> withhold the "
             "deposit\n\nRequest details: All keys returned = yes",
             "What should the assistant do?", "A) Withhold the deposit\nB) Return the full deposit"),
    "goldproof/fld": ("Facts:\nthe slopshop is a foolscap.\nif something is a foolscap then it is sternal.\n\n"
                      "Hypothesis: the slopshop is sternal.",
                      "Question: Based only on the facts, is the hypothesis proved, disproved, or neither (unknown)?",
                      "A) Proved\nB) Unknown\nC) Disproved"),
    "goldproof/boardgameqa": ("A few players are playing a boardgame. The current state of the game is as follows. The "
                              "catfish does not prepare armor for the sheep. And the rules of the game are as follows. "
                              "Rule1: The sheep proceeds to the spot right after the hare, in the case where the catfish "
                              "does not prepare armor for the sheep.",
                              "Question: Based on the game state and the rules and preferences, does the sheep proceed to "
                              "the spot right after the hare?",
                              "A) No (it is disproved)\nB) Yes (it is proved)\nC) Unknown (neither can be shown)"),
    "goldproof/ruletaker": ("Context: Bob is green. If something is green then it is nice.\n\nStatement: Bob is nice.",
                            "Question: Does the statement follow from the context? What is not explicitly stated as true "
                            "is considered false.", "A) not entailment\nB) entailment"),
}


def new_item(src: str, body: str, ask: str, opts: str, **kw) -> dict:
    return {"item_id": f"{src}-x", "source": src, "prompt": f"{body}\n\n{ask}\nOptions:\n{opts}", **kw}


def local_items(paths, sources: set, quick: str | None = None):
    """Every non-node row of the given data files (relative to DATA) whose source is in sources."""
    for rel in paths:
        p = DATA / rel
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                if quick and quick not in line:
                    continue
                it = json.loads(line)
                if it.get("source") in sources and it.get("row_kind") != "node":
                    yield rel, it


OLD_FILES = ["student_v3sl/train.jsonl", "student_v3sl/val.jsonl", "student_v3s/train.jsonl", "student_v3s/val.jsonl",
             "student_numgen_tf/train.jsonl", "student_numgen_tf/val.jsonl", "student_numgen_tf/local_deep.jsonl",
             "student_numgen/train.jsonl", "student_numgen/val.jsonl",
             "student_v7_pilot_math_MS/train.jsonl", "student_v7_pilot_math_MS/val.jsonl",
             "eval/kk_heldout.jsonl", "eval/val_kk.jsonl", "eval/kk_deep.jsonl", "eval/diag_kk.jsonl"] + \
            [f"eval/numgen_d{d}.jsonl" for d in range(1, 9)] + ["eval/numgen_ood.jsonl"]
NEW_FILES = ["student_v3sl/train.jsonl", "student_v3sl/val.jsonl", "student_goldproof/train.jsonl",
             "student_goldproof/val.jsonl", "student_rule/train.jsonl", "student_rule/val.jsonl"]


class TestOldSourcesUnchanged(unittest.TestCase):
    def test_fixtures(self):
        for it in (KK, NUMGEN):
            self.assertEqual(statement_block(it), old_statement_block(it))
            self.assertEqual(problem_no_options(it), old_problem_no_options(it))
        bad = dict(KK, prompt=KK["prompt"].replace("\n\nQuestion: ", "\n\nAsk: "))
        for fn in (statement_block, problem_no_options):
            with self.assertRaises(ValueError):
                fn(bad)

    def test_every_local_kk_and_numgen_item(self):
        n, by = 0, {}
        for rel, it in local_items(OLD_FILES, {"knights_knaves", "numgen"}):
            for new, old in ((statement_block, old_statement_block), (problem_no_options, old_problem_no_options)):
                self.assertEqual(outcome(new, it), outcome(old, it), f"{rel} {it.get('item_id')} {new.__name__}")
            n += 1
            by[rel] = by.get(rel, 0) + 1
        if not n:
            self.skipTest(f"no local K&K / numgen data under {DATA}")
        self.assertGreater(n, 1000)


class TestNewSources(unittest.TestCase):
    def test_cuts(self):
        self.assertEqual(set(NEW), set(FINAL_QUESTION))
        for src, (body, ask, opts) in NEW.items():
            it = new_item(src, body, ask, opts)
            self.assertEqual(statement_block(it), body, src)
            self.assertEqual(problem_no_options(it), f"{body}\n\n{ask}", src)
            self.assertTrue(it["prompt"].startswith(problem_no_options(it)))

    def test_rule_renderers(self):
        body, _, opts = NEW["rule"]
        for ask in ("What should the clerk do?", "Question: What should the agent do?",
                    "What should the pet-boarding receptionist do?"):
            self.assertEqual(statement_block(new_item("rule", body, ask, opts)), body)

    def test_refusals(self):
        for src, (body, ask, opts) in NEW.items():
            bad = [new_item(src, body, "Question: Which option is right?", opts),          # another final question
                   {"item_id": "x", "source": src, "prompt": f"{body}\n\n{ask}"},           # no options
                   new_item(src, body, ask, opts + "\nOptions:\nA) x"),                       # two options blocks
                   new_item(src, f"Question: {ask}\n\n{body}", ask, opts),                    # two question paragraphs
                   {"item_id": "x", "source": src, "prompt": f"{ask}\nOptions:\n{opts}"},   # nothing before it
                   new_item(src, body, ask + "\nMore text.", opts)]                          # not one paragraph
            for it in bad:
                for fn in (statement_block, problem_no_options):
                    with self.assertRaises(ValueError, msg=f"{src}: {it['prompt'][:80]!r}"):
                        fn(it)

    def test_no_clean_cut(self):
        arc = {"item_id": "arc-x", "source": "arc_challenge",
               "prompt": "Question: What is a worldwide increase in temperature called?\nOptions:\nA) ozone\nB) warming"}
        gsm = {"item_id": "gsm8k-x", "source": "gsm8k_soc",
               "prompt": "Question: Carla needs 30 leaves and 20 bugs in 10 days. How many items a day?\nOptions:\nA) 5\nB) 7"}
        for it in (arc, dict(arc, source="arc_easy"), gsm, dict(gsm, source="tasksource/x")):
            for fn in (statement_block, problem_no_options):
                with self.assertRaises(ValueError):
                    fn(it)

    def test_every_local_item(self):
        """Every local item of a FINAL_QUESTION source has a block; problem_no_options = block + the question
        paragraph; no node of these trees is a view_leak in the statements view (nothing stated)."""
        n = leaks = 0
        for rel, it in local_items(NEW_FILES, set(FINAL_QUESTION)):
            block, ctx = statement_block(it), problem_no_options(it)
            last = ctx[len(block):]
            self.assertTrue(last.startswith("\n\n") and "\n" not in last[2:], f"{rel} {it['item_id']}")
            self.assertTrue(re.fullmatch(FINAL_QUESTION[it["source"]], last[2:]))
            self.assertEqual(it["prompt"][:len(ctx) + 10], ctx + "\nOptions:\n")
            for sq in it.get("subqs") or []:
                leaks += bool(view_leak(it, sq, []))
            n += 1
        if not n:
            self.skipTest(f"no local data under {DATA}")
        self.assertGreater(n, 1000)
        self.assertEqual(leaks, 0)


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class TestGFPSViewsNewSource(unittest.TestCase):
    def test_statements_view_is_problem_view_with_block(self):
        from cotdistill.student import build_full_tree
        body, ask, opts = NEW["rule"]
        it = new_item("rule", body, ask, opts, labels=["A", "B"], gold_label="B", teacher={"A": 0.0, "B": 1.0},
                      subqs=[{"id": "n1", "question": "Were all keys returned?", "truth": True, "p_cot": 1.0,
                              "depends_on": []},
                             {"id": "n2", "question": "Does rule 1 apply?", "truth": True, "p_cot": 1.0,
                              "depends_on": ["n1"]}])
        for view, ctx in (("statements", body), ("statements+question", f"{body}\n\n{ask}")):
            s = build_full_tree(it, mode="plain", view=view, rng=random.Random(3), cap=0, sources={"rule"})
            g = build_full_tree(it, mode="plain", view="problem", rng=random.Random(3), cap=0, sources={"rule"})
            self.assertEqual(len(s), 3)
            self.assertEqual(s[0].text, g[0].text)
            for es, eg in zip(s[1:], g[1:]):
                self.assertEqual(es.text, eg.text.replace(it["prompt"], ctx, 1))
                self.assertNotIn("Return the full deposit", es.text)
                self.assertEqual((es.target, es.weight), (eg.target, eg.weight))


if __name__ == "__main__":
    unittest.main()
