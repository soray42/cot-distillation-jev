"""Format probes for the TD yes/no ("noul") questions: is the Yes bias a property of the declarative-claim format?

From data/eval/typed_decisions_test.jsonl, every noul item (6 question types x 100) is rewritten as:
  orig   unchanged (declarative claim, "A) No[: ...] B) Yes[: ...]")
  plain  the same claim with bare "A) No / B) Yes" options (no option explanations)
  qform  the claim as a yes/no question, bare options
  neg    the negated claim, bare options, gold flipped
  swap   the original claim with the options in the other order ("A) Yes / B) No")
and content-free inputs (the state replaced by "{}", one per question type and form) for contextual calibration.

  python scripts/build_td_probes.py      # -> data/eval/td_noul_{orig,plain,qform,neg,swap,cf}.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMS = {   # question type -> (question form, negated claim)
    "needs_review": ("Does this trace require human review?", "This trace does not require human review."),
    "needs_human": ("Does this conversation require a human agent rather than automated handling?",
                    "This conversation does not require a human agent; automated handling is enough."),
    "duplicate": ("Does this invoice duplicate an invoice already submitted?",
                  "This invoice does not duplicate any invoice already submitted."),
    "matches_order": ("Does the invoice reconcile with the purchase order and the recorded delivery?",
                      "The invoice does not reconcile with the purchase order and the recorded delivery."),
    "credential_compromise": ("Does the evidence indicate that a credential or account has been compromised?",
                              "The evidence does not indicate that any credential or account has been compromised."),
    "true_positive": ("Does this alert reflect genuinely malicious or unauthorised activity?",
                      "This alert does not reflect malicious or unauthorised activity."),
}


def split(prompt: str) -> tuple[str, str]:
    head, _, _ = prompt.rpartition("\nOptions:\n")
    state, _, q = head.rpartition("\nQuestion: ")
    return state, q


def record(it: dict, tag: str, state: str, question: str, yes_first: bool, gold_yes: bool, opts: list[str] | None = None):
    if opts is None:
        opts = ["Yes", "No"] if yes_first else ["No", "Yes"]
    names = {"A": "true" if opts[0].startswith("Yes") else "false", "B": "true" if opts[1].startswith("Yes") else "false"}
    gold = next(L for L, n in names.items() if (n == "true") == gold_yes)
    return {"item_id": f"{it['item_id']}/{tag}", "source": "typed_decisions", "group": it["group"], "type": "noul",
            "prompt": f"{state}\nQuestion: {question}\nOptions:\nA) {opts[0]}\nB) {opts[1]}", "labels": ["A", "B"],
            "gold_label": gold, "label_names": names, "label_order": ["A", "B"]}


def main() -> None:
    items = [json.loads(l) for l in open(ROOT / "data/eval/typed_decisions_test.jsonl")]
    out = {k: [] for k in ("orig", "plain", "qform", "neg", "swap", "cf")}
    seen_cf = set()
    for it in items:
        if it["type"] != "noul":
            continue
        qt = it["item_id"].split("/")[-1]
        state, claim = split(it["prompt"])
        opts = it["prompt"].rpartition("\nOptions:\n")[2].splitlines()
        orig_opts = [o[3:] for o in opts]
        gold_yes = it["label_names"][it["gold_label"]] == "true"
        q, neg = FORMS[qt]
        out["orig"].append(record(it, "orig", state, claim, False, gold_yes, orig_opts))
        out["plain"].append(record(it, "plain", state, claim, False, gold_yes))
        out["qform"].append(record(it, "qform", state, q, False, gold_yes))
        out["neg"].append(record(it, "neg", state, neg, False, not gold_yes))
        out["swap"].append(record(it, "swap", state, claim, True, gold_yes, orig_opts[::-1]))
        if qt not in seen_cf:          # content-free inputs: the state is replaced by an empty object
            seen_cf.add(qt)
            for tag, text in (("orig", claim), ("qform", q), ("neg", neg)):
                r = record({**it, "item_id": f"cf/{qt}"}, tag, "State:\n{}\n", text, False, True)
                out["cf"].append(r)
    for k, rows in out.items():
        with open(ROOT / f"data/eval/td_noul_{k}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(k, len(rows))


if __name__ == "__main__":
    main()
