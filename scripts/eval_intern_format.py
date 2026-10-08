"""Score a model in Intern-Decision's own request format (internlm/Intern-Decision, src/inputs/schema.py, reimplemented):
system prompt + user "## State / ## Decision schema" + an assistant JSON skeleton with one <decision> marker per field,
chat template with an empty think block; each field is read at the position just before its marker, softmax over that
field's answer symbols (A, B, ... from ANSWER_SYMBOLS). Used to check whether its public JevBench scores reproduce and
survive simple perturbations, next to sets it was not tuned on.

Set kinds (--set name=path:kind):
  jevraw  upstream JevBench rows (state, question{type, instructions, criteria}, expected, labels, group, family)
  canon   canonical multi-field rows (state, questions{field: question}, targets{field: {label}}), e.g. Typed Decision
  mc      our multiple-choice items (prompt "...\\n\\nQuestion: q\\nOptions:\\nA) x\\n...", labels, gold_label): state = the
          text before the question, one choice field whose options are the option texts
--perm reverse|shuffle reorders the options of choice fields (score/noul keep their order: ordinal / fixed no,yes);
the gold answer follows its option. Accuracy is decision-level (one count per field).

  python scripts/eval_intern_format.py --model models/Intern-Decision-4B --out runs/intern/id4b \\
      --set jev_hard=data/eval/intern_bench/jevbench/hard.jsonl:jevraw --perm none
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

DECISION_TOKEN = "<decision>"
ANSWER_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
SYSTEM_PROMPT = (
    "You are a careful decision assistant. Use the state and decision schema in "
    "the user message to make the requested decisions. For every field, choose "
    "exactly one answer symbol (e.g. A, B, C, ...) from its listed options and "
    "return one valid JSON object mapping each field name to its chosen symbol. "
    "Use the field names and symbols exactly as given. Do not include "
    "explanations, Markdown, or extra text."
)
NOUL_YES = "The answer is yes (affirmative, or align with the claim)."
NOUL_NO = "The answer is no (negative, or disagree with the claim)."


def options(q: dict) -> list[tuple[str, str]]:
    kind, crit = q.get("type"), q.get("criteria")
    if kind == "choice":
        return [(str(k), str(v)) for k, v in crit.items()]
    if kind == "score":
        if isinstance(crit, list):
            return [(str(i), str(v)) for i, v in enumerate(crit)]
        return [(str(k), str(v)) for k, v in crit.items()]
    if kind == "noul":
        d = crit if isinstance(crit, dict) else {}
        yes = next((str(d[k]) for k in d if str(k).lower() in {"yes", "true", "1"}), NOUL_YES)
        no = next((str(d[k]) for k in d if str(k).lower() in {"no", "false", "0"}), NOUL_NO)
        return [("no", no), ("yes", yes)]
    raise ValueError(f"unsupported question type {kind!r}")


def norm_label(q: dict, label) -> str:
    if q.get("type") == "noul":
        return "yes" if str(label).lower() in {"yes", "true", "1"} else "no"
    return str(label)


def user_text(state, fields: list[tuple[str, dict]]) -> str:
    lines = []
    for name, q in fields:
        lines.append(f"{name}: {q.get('instructions', '')}")
        for sym, (val, desc) in zip(ANSWER_SYMBOLS, options(q)):
            lines.append(f"    {sym} = {val}: {desc}")
    return ("Return one answer for every field using the supplied answer symbols.\n\n## State\n"
            f"{json.dumps(state, ensure_ascii=False, indent=2)}\n## Decision schema\n" + "\n".join(lines))


def messages(state, fields: list[tuple[str, dict]]) -> list[dict]:
    skeleton = json.dumps({n: DECISION_TOKEN for n, _ in fields}, ensure_ascii=False, indent=4)
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text(state, fields)},
            {"role": "assistant", "content": skeleton}]


def permute(q: dict, how: str, key: str) -> dict:
    if how == "none" or q.get("type") != "choice":
        return q
    items = list(q["criteria"].items())
    if how == "reverse":
        items = items[::-1]
    elif how == "shuffle":
        random.Random(f"perm|{key}").shuffle(items)
    return dict(q, criteria=dict(items))


_MC = re.compile(r"\n\nQuestion: (.*?)\nOptions:\n(.*)$", re.S)


def from_mc(it: dict) -> dict:
    m = _MC.search(it["prompt"])
    if not m:
        raise ValueError(f"{it.get('item_id')}: not a 'Question: ... Options:' prompt")
    state = it["prompt"][:m.start()]
    opts = {}
    for line in m.group(2).split("\n"):
        lab, _, text = line.partition(") ")
        if lab in it["labels"]:
            text = text.strip()
            opts[lab] = text if text not in opts.values() else f"{text} ({lab})"
    crit = {opts[l]: opts[l] for l in it["labels"]}
    q = {"type": "choice", "instructions": m.group(1).strip(), "criteria": crit}
    return {"id": it["item_id"], "group": it.get("group"), "state": state, "questions": {"decision": q},
            "targets": {"decision": {"label": opts[it["gold_label"]]}}}


def load_rows(path: str, kind: str) -> list[dict]:
    raw = [json.loads(l) for l in open(path) if l.strip()]
    if kind == "jevraw":
        return [{"id": r["id"], "group": r.get("group"), "family": r.get("family"), "state": r["state"],
                 "questions": {"decision": r["question"]}, "targets": {"decision": {"label": r["expected"]}}}
                for r in raw]
    if kind == "canon":
        return [dict(r, group=r.get("category")) for r in raw]
    if kind == "mc":
        return [from_mc(it) for it in raw if it.get("gold_label") in it.get("labels", [])]
    raise ValueError(f"unknown kind {kind}")


def marker_positions(tok, text: str) -> tuple[list[int], list[int]]:
    """Token ids of the rendered text and the positions whose logits predict each marker's field. A trained
    checkpoint has <decision> as one added token (its position - 1, as Intern-Decision's backend); otherwise the
    marker string is split out and each position is the last token of the text before it."""
    if DECISION_TOKEN in tok.get_added_vocab():
        ids = tok(text, add_special_tokens=False)["input_ids"]
        mid = tok.convert_tokens_to_ids(DECISION_TOKEN)
        return ids, [i - 1 for i, t in enumerate(ids) if t == mid]
    parts = text.split(DECISION_TOKEN)
    ids, pos = [], []
    for k, part in enumerate(parts):
        ids += tok(part, add_special_tokens=False)["input_ids"]
        if k < len(parts) - 1:
            pos.append(len(ids) - 1)
    return ids, pos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", required=True, help="name=path:kind (jevraw|canon|mc)")
    ap.add_argument("--perm", default="none", choices=["none", "reverse", "shuffle"])
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    import transformers
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cotdistill.student import load_model, text_parts

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    body, head = text_parts(model)
    sym_ids = [tok.convert_tokens_to_ids(s) for s in ANSWER_SYMBOLS]
    if len(set(sym_ids)) != len(sym_ids) or tok.unk_token_id in sym_ids:
        raise SystemExit("answer symbols are not distinct single tokens")
    print(f"model {type(model).__name__}, marker token: {DECISION_TOKEN in tok.get_added_vocab()}, perm {args.perm}",
          flush=True)
    res = {"args": vars(args), "eval": {}}
    shown = False
    for spec in args.set:
        name, rest = spec.split("=", 1)
        path, kind = rest.rsplit(":", 1)
        rows = load_rows(path, kind)[:args.limit or None]
        t0 = time.time()
        correct, total, groups, preds, n_trunc = 0, 0, defaultdict(lambda: [0, 0]), [], 0
        for r in rows:
            fields = [(n, permute(q, args.perm, f"{r['id']}|{n}")) for n, q in r["questions"].items()]
            text = tok.apply_chat_template(messages(r["state"], fields), tokenize=False, add_generation_prompt=False,
                                           enable_thinking=False)
            if not shown:
                print("first rendered request:", repr(text[-1500:]), flush=True)
                shown = True
            ids, pos = marker_positions(tok, text)
            if len(pos) != len(fields):
                raise SystemExit(f"{r['id']}: {len(pos)} marker positions for {len(fields)} fields")
            if len(ids) > args.max_len:
                n_trunc += 1
                continue
            x = torch.tensor([ids], device=dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                h = body(input_ids=x, use_cache=False).last_hidden_state[0, pos]
            logits = (h.to(head.weight.dtype) @ head.weight.T).float()
            for (n, q), lg in zip(fields, logits):
                opts = options(q)
                p = torch.softmax(lg[sym_ids[:len(opts)]], -1).tolist()
                gold = norm_label(q, r["targets"][n]["label"])
                vals = [v for v, _ in opts]
                if gold not in vals:
                    raise SystemExit(f"{r['id']}/{n}: gold {gold!r} not among options")
                ok = vals[max(range(len(p)), key=p.__getitem__)] == gold
                correct += ok
                total += 1
                g = r.get("group") or "?"
                groups[g][0] += ok
                groups[g][1] += 1
                preds.append({"id": r["id"], "field": n, "group": r.get("group"), "values": vals, "probs": p,
                              "gold": gold, "correct": ok, "gold_prob": p[vals.index(gold)]})
        m = {"n_rows": len(rows), "n_decisions": total, "acc": correct / max(total, 1), "n_truncated": n_trunc,
             "mean_gold_prob": sum(q["gold_prob"] for q in preds) / max(len(preds), 1),
             "seconds": round(time.time() - t0, 1),
             "groups": {k: {"n": v[1], "acc": v[0] / v[1]} for k, v in sorted(groups.items())}}
        res["eval"][name] = m
        with open(out / f"preds_{name}.jsonl", "w") as f:
            for q in preds:
                f.write(json.dumps(q) + "\n")
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k != "groups"}),
              flush=True)
        (out / "metrics.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
