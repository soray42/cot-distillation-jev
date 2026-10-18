"""GSM8K worked steps for stepwise internalisation (arms SI / SIY / SIS in hpc/train_arm.sbatch; student.si_group,
train_student.py --si-stages): the complete chain of atomic operations of each problem's solution, written as lines
"Step 1: 4 + 2 = 6" between the problem and its options, removed from the front stage by stage (Deng et al. 2024,
ICoT-SI).

Rows: every row of data/student_gsm8k_tf_ops/{train,val}.jsonl (scripts/build_gsm8k_ops.py), byte for byte the same
fields (finals, options, gold, the yes/no tree in `subqs`, `value_nodes`, ...), plus ONE new last field `worked_steps`
(a list, or null when the chain cannot be written exactly). Subsets data/student_gsm8k_tf_si_f{10,25,50}: the train ids
of data/student_gsm8k_tf_ops_f{10,25,50} in the same order, the full val (as there).

The chain: every operation of the GSM8K calculator notes of the steps up to and including the gold step (the last step
whose value is the gold; build_gsm8k_socratic.parse_problem), split into binary operations in evaluation order exactly
as the value nodes are (build_gsm8k_ops.split_note: precedence, then left to right; "P% of X" where the percentage is
written; fraction literals p/q as one number), so "4+6+8+2" gives "4 + 6 = 10", "10 + 8 = 18", "18 + 2 = 20". Unlike
`value_nodes` NOTHING is skipped: the gold step's last operation (the one that yields the final answer, final_op),
trivial operations (x × 1), operations of blocked steps and gold-valued operations are all written, so the chain is
complete and its last line yields the gold. Steps after the gold step (15 operations in all) are not part of it; a step
whose note restates a number of the problem has no operation and no line. A chain is null (the problem trains
without one; counted in summary.json build: null_<reason>) when it would not be complete or exact: a step up to the
gold step is flagged "bare" by parse_problem (a computed value written without its computation, "<<36=36>>"), the
gold step's note has no operation (final_not_written), an unsupported operator (//, %, **), a value or operand that
is not a terminating decimal of at most 4 places, or no wrong value for some line.

A line: {id (s<step>o<k>, the value-node id scheme), step, op_index, kind (bin / pct), a, op, b, value (exact strings
as build_gsm8k_ops.fmt; a negative operand in parentheses), text "a op b = value" ("50% of 4 = 2" for pct), node (the
value-node id when that operation has a value node, else null), a_from / b_from (index of the earlier line whose result
the operand is: an earlier operation of the same note, or the last operation of the parent step through parse_problem's
certain edges, else the latest earlier line with the operand's value when the problem text does not write that
number; else null), wrong, wrong_tier, wrong_text, wrong_result_text}.
The wrong value (SIS, the wrong-content control) is never the line's true value:
  - an operation with a value node: one of the two wrong options of its "op" rendering (the value node's distractors:
    operator slips, other operations' values, perturbations, all within 10x of the truth, never the gold), drawn by
    random.Random("<seed>/<problem_id>/<line id>/sis") (wrong_tier "node");
  - otherwise (the final operation, trivial operations, blocked steps, ...): the first tier with a candidate among
    build_gsm8k_ops' tiers (operator: same operands with another operator or the complement of a fractional factor;
    other: another line's value; perturb: build_gsm8k_ops.near_values), each candidate passing build_gsm8k_ops.value_ok
    (positive, an integer if the truth is, at most max(2, decimals of the truth) decimals, within 10x, not the gold),
    drawn by the same generator; a non-positive truth moves by 1 or 2.
  - an operation whose true value is the gold (the final operation, and any gold-valued one) never gets a wrong value
    equal to one of the final question's options, so the wrong chain never points at a specific wrong letter.
wrong_text writes every computed value of the chain wrongly: the line's result is its wrong value and an operand that
is an earlier line's result (a_from / b_from) is that line's wrong value (the chain keeps its form: each line reads the
results written above it; no true intermediate result is left as an operand: audit sis_operand_leaks, 0 by
construction; a wrong value may still equal ANOTHER line's true value, the "other" tier of the distractors:
sis_lines_with_another_lines_true_value). wrong_result_text replaces the results only (operands as in text).
Audit (from the stored fields only): every line's operation re-evaluated exactly equals its value; the last line's
value is the gold; each value node of the row is the line with its id (same operands, operator and value); wrong values
differ from the true ones and the gold-valued lines' wrong values are no option; the rows' other fields equal
data/student_gsm8k_tf_ops'. chain_audit.txt shows gsm8k-train-660 and four more problems (written, wrong).

  python3 scripts/build_gsm8k_si.py      # pandas + numpy (system python3); writes data/student_gsm8k_tf_si*/
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_gsm8k_ops as B  # noqa: E402

T = B.T
FIELD = "worked_steps"
FRACTIONS = B.FRACTIONS


def otext(d: dict) -> str | None:
    """An operand's text as written in a line (a negative operand in parentheses, as build_gsm8k_ops.render)."""
    t = d["text"]
    return None if t is None else (f"({t})" if t.startswith("-") else t)


def vtext(x: Fraction) -> str | None:
    return B.fmt(x)


def line_text(kind: str, a: str, sym: str, b: str, v: str) -> str:
    return f"{B.expr_text(kind, a, sym, b)} = {v}"


def wrong_value(t: Fraction, kind: str, a: Fraction, sym: str, b: Fraction, others: list, gold: Fraction,
                banned: set, rng: random.Random) -> tuple[Fraction, str] | None:
    """(a wrong value, its tier) for a line without a value node: the first of build_gsm8k_ops' tiers with a candidate
    (value_ok, not in banned), drawn uniformly from that tier (sorted by text, then rng.choice)."""
    if t <= 0:                                       # value_ok needs a positive truth
        cands = [t + d for d in (1, 2, -1, -2) if t + d != gold and t + d not in banned and t + d != t]
        return (rng.choice(sorted(cands)), "perturb") if cands else None
    tiers = {"operator": [w for w, _ in B.operator_slips(kind, a, sym, b)], "other": list(others),
             "perturb": B.near_values(t)}
    for tier in B.TIERS:
        pool = {B.fmt(w): w for w in tiers[tier] if B.value_ok(w, t, gold) and w not in banned}
        if pool:
            return pool[rng.choice(sorted(pool))], tier
    return None


def worked_steps(p: dict, item: dict, seed: int, st: collections.Counter) -> list[dict] | None:
    """The worked steps of one parsed problem (parse_problem, status ok) and its ops row (value_nodes, options)."""
    steps, gs = p["steps"], p["gold_step"]
    gold = Fraction(str(p["gold"]))
    pid = item["problem_id"]
    nodes = {n["id"]: n for n in item.get("value_nodes") or []}
    options = {B.num(str(x)) for x in item["options"]}
    if any("bare" in s["flags"] for s in steps[:gs + 1]):     # a computed value written without its computation
        st["null_bare_step"] += 1
        return None
    pnums = problem_numbers(item)
    lines, raw, last = [], [], {}
    for j, s in enumerate(steps[:gs + 1]):
        ops, _, _ = B.split_note(s["tree"], B.written_percents(s["sent"], p["question"]), s["refs"])
        base = len(lines)
        for r in ops:
            a, b, v = r["a"], r["b"], r["val"]
            reason = None
            if r["skip"]:
                reason = r["skip"]
            elif v is None or B.fmt(v) is None:
                reason = "value_inexact"
            elif otext(a) is None or otext(b) is None:
                reason = "operand_inexact"
            if reason:
                st[f"null_{reason}"] += 1
                return None
            frm = []
            for d in (a, b):
                if d["op"] is not None:
                    frm.append(base + d["op"])
                    st["links_note"] += 1
                elif d["ref"] is not None and d["ref"] in last:
                    frm.append(last[d["ref"]])
                    st["links_certain_edge"] += 1
                elif d["neg"]:
                    frm.append(None)
                else:                               # the latest earlier result of that value, unless the problem
                    same = [k for k, x in enumerate(raw) if x[4] == d["val"]]           # writes the number
                    frm.append(same[-1] if same and d["val"] not in pnums else None)
                    st["links_value"] += frm[-1] is not None
            lid = f"s{j + 1}o{r['k'] + 1}"
            lines.append({"id": lid, "step": j + 1, "op_index": r["k"] + 1, "kind": r["kind"], "a": otext(a),
                          "op": r["sym"], "b": otext(b), "value": B.fmt(v),
                          "text": line_text(r["kind"], otext(a), r["sym"], otext(b), B.fmt(v)),
                          "node": lid if lid in nodes else None, "a_from": frm[0], "b_from": frm[1]})
            raw.append((r["kind"], a["val"], r["sym"], b["val"], v))
        if ops:
            last[j] = len(lines) - 1
    if not lines:
        st["null_no_operation"] += 1
        return None
    if raw[-1][4] != gold:                          # the gold step's note is a bare number ("<<4=4>>")
        st["null_final_not_written"] += 1
        return None
    vals = [x[4] for x in raw]
    for i, (ln, (kind, a, sym, b, v)) in enumerate(zip(lines, raw)):
        rng = random.Random(f"{seed}/{pid}/{ln['id']}/sis")
        if ln["node"] is not None:
            r = nodes[ln["node"]]["renderings"]["op"]
            w = B.num(rng.choice([x for k, x in enumerate(r["options"]) if k != r["answer"]]))
            tier = "node"
        else:
            got = wrong_value(v, kind, a, sym, b, [x for x in vals if x != v], gold, options if v == gold else set(),
                              rng)
            if got is None:
                st["null_no_wrong_value"] += 1
                return None
            w, tier = got
        ln["wrong"], ln["wrong_tier"] = B.fmt(w), tier
        st[f"wrong_tier_{tier}"] += 1
    for ln in lines:
        wa = lines[ln["a_from"]]["wrong"] if ln["a_from"] is not None else ln["a"]
        wb = lines[ln["b_from"]]["wrong"] if ln["b_from"] is not None else ln["b"]
        wa, wb = (f"({x})" if x.startswith("-") else x for x in (wa, wb))
        ln["wrong_text"] = line_text(ln["kind"], wa, ln["op"], wb, ln["wrong"])
        ln["wrong_result_text"] = line_text(ln["kind"], ln["a"], ln["op"], ln["b"], ln["wrong"])
    return lines


def problem_numbers(item: dict) -> set:
    return {B.num(x) for x in B.NUMTOK.findall(item["prompt"].partition("\nOptions:")[0])}


def operand_values(line: dict, which: str) -> list[Fraction]:
    """The operand values written in a line's text (which: 'text' or 'wrong_text')."""
    expr = line[which].rpartition(" = ")[0]
    out = []
    for x in B.NUMTOK.findall(expr.replace("(", " ").replace(")", " ")):
        out.append(B.num(x))
    return out


def audit_item(item: dict) -> tuple[list[str], collections.Counter]:
    """Problems of one built row from its own fields (empty: clean) and counters."""
    bad, c = [], collections.Counter()
    ws = item.get(FIELD)
    iid = item["item_id"]
    if ws is None:
        c["null"] += 1
        return bad, c
    gold = B.num(T.gold_text(item))
    options = {B.num(str(x)) for x in item["options"]}
    pnums = problem_numbers(item)
    for i, ln in enumerate(ws):
        A, Bv, V = B.num(ln["a"].strip("()")), B.num(ln["b"].strip("()")), B.num(ln["value"])
        got = B.expr_value(ln["kind"], A, ln["op"], Bv)
        c["lines"] += 1
        if got != V:
            bad.append(f"{iid}/{ln['id']}: {ln['text']} evaluates to {got}")
        if ln["text"] != line_text(ln["kind"], ln["a"], ln["op"], ln["b"], ln["value"]):
            bad.append(f"{iid}/{ln['id']}: text does not match its fields")
        W = B.num(ln["wrong"])
        if B.fmt(W) == B.fmt(V):
            bad.append(f"{iid}/{ln['id']}: wrong value equals the truth")
        if V == gold and W in options:
            bad.append(f"{iid}/{ln['id']}: the wrong value of a gold-valued line is an option")
        for k in ("a_from", "b_from"):
            if ln[k] is not None and not 0 <= ln[k] < i:
                bad.append(f"{iid}/{ln['id']}: {k} {ln[k]} is not an earlier line")
        for k, side in (("a_from", A), ("b_from", Bv)):
            if ln[k] is not None and B.num(ws[ln[k]]["value"]) != side:
                c["from_value_differs"] += 1      # the literal of a certain edge, rounded in the note
        # SIS leaks: an operand left as written (no link) that is an earlier line's TRUE result, not a problem number
        earlier = {B.num(x["value"]) for x in ws[:i]} - pnums
        if any(ln[k] is None and side in earlier for k, side in (("a_from", A), ("b_from", Bv))):
            c["sis_operand_leaks"] += 1
        c["si_lines_reading_earlier_results"] += any(side in earlier for side in (A, Bv))
        c["sis_lines_with_another_lines_true_value"] += any(x in earlier for x in operand_values(ln, "wrong_text")
                                                            + [W])
        if ln["node"] is not None:
            n = next((x for x in item["value_nodes"] if x["id"] == ln["node"]), None)
            if n is None or (n["a"], n["op"], n["b"], n["value"]) != (ln["a"], ln["op"], ln["b"], ln["value"]):
                bad.append(f"{iid}/{ln['id']}: not the value node {ln['node']}")
    if B.num(ws[-1]["value"]) != gold:
        bad.append(f"{iid}: the last line {ws[-1]['text']} does not yield the gold {T.gold_text(item)}")
    built = {n["id"] for n in item.get("value_nodes") or []}
    if not built <= {ln["id"] for ln in ws}:
        bad.append(f"{iid}: value nodes {sorted(built - {ln['id'] for ln in ws})} missing from the chain")
    c["items"] += 1
    return bad, c


def show(item: dict) -> str:
    out = [f"=== {item['item_id']}  (gold {T.gold_text(item)}, options {item['options']})",
           item["prompt"].partition("\nOptions:")[0]]
    ws = item.get(FIELD)
    if ws is None:
        return "\n".join(out + ["  (no chain)"])
    out.append("written:")
    out += [f"  Step {i}: {ln['text']}    [{ln['id']}{', node' if ln['node'] else ''}]" for i, ln in enumerate(ws, 1)]
    out.append("wrong (SIS):")
    out += [f"  Step {i}: {ln['wrong_text']}    [{ln['wrong_tier']}]" for i, ln in enumerate(ws, 1)]
    return "\n".join(out) + "\n"


def file_stats(items: list[dict]) -> dict:
    ns = [len(it[FIELD]) for it in items if it.get(FIELD)]
    return {"items": len(items), "items_with_chain": len(ns), "lines": sum(ns),
            "lines_per_item": round(sum(ns) / max(1, len(ns)), 2),
            "n_lines": dict(sorted(collections.Counter(ns).items())),
            "lines_with_value_node": sum(ln["node"] is not None for it in items for ln in it.get(FIELD) or []),
            "chain_chars": sum(len(ln["text"]) + len(f"Step {i}: ") + 1 for it in items
                               for i, ln in enumerate(it.get(FIELD) or [], 1))}


def pick(items: list[dict]) -> list[dict]:
    """gsm8k-train-660, then the first train items with a percent line, a fraction literal, a gold-valued line before
    the last one and a long chain (>= 8 lines)."""
    by = {it["item_id"]: it for it in items}
    out = [by["gsm8k-train-660"]] if "gsm8k-train-660" in by else []
    tests = [lambda ws, it: any(ln["kind"] == "pct" for ln in ws),
             lambda ws, it: any("/" in ln["a"] or "/" in ln["b"] for ln in ws),
             lambda ws, it: any(ln["value"] == T.gold_text(it) for ln in ws[:-1]),
             lambda ws, it: len(ws) >= 8]
    for f in tests:
        for it in items:
            if it not in out and it.get(FIELD) and f(it[FIELD], it):
                out.append(it)
                break
    return out


def build(args) -> None:
    import pandas as pd
    S = T.soc()
    src, out = ROOT / args.src, ROOT / args.out
    df = pd.read_parquet(T.RAW / "train-00000-of-00001.parquet")
    st = collections.Counter()
    fails, built = [], {}
    for split in ("train", "val"):
        built[split] = []
        with open(src / f"{split}.jsonl") as f:
            lines = [x.rstrip("\n") for x in f if x.strip()]
        for line in lines:
            r = json.loads(line)
            i = int(r["problem_id"].rsplit("-", 1)[1])
            p = S.parse_problem(i, df.question[i], df.answer[i])
            if p["status"] != "ok" or f"Question: {p['question']}\nOptions:\n" != r["prompt"][:len(p["question"]) + 20]:
                fails.append(f"{r['item_id']}: raw row {i} does not reproduce the prompt")
                continue
            if T._fmt(p["gold"]) != T.gold_text(r) or len(p["steps"]) != r["n_steps"]:
                fails.append(f"{r['item_id']}: gold / step count differs from the raw solution")
                continue
            o = dict(r)
            o[FIELD] = worked_steps(p, r, args.seed, st)
            st["items"] += 1
            st["items_with_chain"] += o[FIELD] is not None
            if B.dumps({k: v for k, v in o.items() if k != FIELD}) != line:
                fails.append(f"{r['item_id']}: row differs from {args.src} beyond {FIELD}")
            built[split].append(o)
    print("built", {k: len(v) for k, v in built.items()}, f"rss {T.rss_mb()} MB", flush=True)
    audit = collections.Counter()
    for it in built["train"] + built["val"]:
        b, c = audit_item(it)
        fails += b
        audit.update(c)
    summary = {"script": "scripts/build_gsm8k_si.py", "args": vars(args), "field": FIELD, "md5": {}, "files": {}}
    for split in ("train", "val"):
        key = f"{out.name}/{split}.jsonl"
        summary["md5"][key] = T.write(out / f"{split}.jsonl", built[split])
        summary["files"][key] = file_stats(built[split])
    by_id = {it["item_id"]: it for it in built["train"]}
    summary["subsets"] = {}
    for fr in FRACTIONS:
        sdir, odir = ROOT / f"{args.src}_f{fr}", ROOT / f"{args.out}_f{fr}"
        if not (sdir / "train.jsonl").exists():
            fails.append(f"missing {sdir}/train.jsonl")
            continue
        ids = [json.loads(x)["item_id"] for x in open(sdir / "train.jsonl") if x.strip()]
        miss = [x for x in ids if x not in by_id]
        if miss:
            fails.append(f"{sdir}: {len(miss)} train ids not built, e.g. {miss[:3]}")
        rows_ = [by_id[x] for x in ids if x in by_id]
        for name, rr in (("train", rows_), ("val", built["val"])):
            summary["md5"][f"{odir.name}/{name}.jsonl"] = T.write(odir / f"{name}.jsonl", rr)
        summary["files"][f"{odir.name}/train.jsonl"] = file_stats(rows_)
        summary["subsets"][odir.name] = {"source_ids": f"{sdir.relative_to(ROOT)}/train.jsonl", "n_train": len(rows_),
                                         "same_ids_same_order": [it["item_id"] for it in rows_] == ids,
                                         "source_train_md5": T.md5(sdir / "train.jsonl")}
    summary["source_md5"] = {f"{args.src}/{s}.jsonl": T.md5(src / f"{s}.jsonl") for s in ("train", "val")}
    summary["build"] = dict(sorted(st.items()))
    summary["audit"] = {"failures": len(fails), "examples": fails[:20], **dict(sorted(audit.items()))}
    picks = pick(built["train"])
    (out / "chain_audit.txt").write_text("\n".join(show(it) for it in picks))
    summary["chain_audit"] = [it["item_id"] for it in picks]
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("files", "subsets", "build", "audit")}, indent=1, ensure_ascii=False))
    print(f"rss {T.rss_mb()} MB")
    if fails:
        print("AUDIT FAILED", fails[:10])
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/student_gsm8k_tf_ops")
    ap.add_argument("--out", default="data/student_gsm8k_tf_si")
    ap.add_argument("--seed", type=int, default=0)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
