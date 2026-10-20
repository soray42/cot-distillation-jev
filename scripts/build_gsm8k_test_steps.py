"""Worked steps for the GSM8K TEST problems of data/eval/gsm8k_mc_v2.jsonl (gsm8kv2), the held-out set of
scripts/probe_slots.py beside data/student_gsm8k_tf_si/val.jsonl. The eval rows carry no chain (build_math_evals writes
none), so the probe could not name the value of step j for them.

The chain is build_gsm8k_si's, computed by its own function (build_gsm8k_si.worked_steps) on the raw socratic test
solution (data/raw/bench_math/gsm8k/socratic/test-*.parquet, row orig_id gsm8k-test-<i>), for problems whose parse has
status ok (as every training row): every operation of the calculator notes up to the gold step, split into binary
operations exactly as for training. The eval rows have no value nodes and no option texts, so the call gets a stand-in
row (problem_id, prompt, no value nodes, no options); those two only steer the WRONG values (SIS) and the node ids,
never a line's operands or value. Only the true chain is kept: id, step, op_index, kind, a, op, b, value, text, a_from,
b_from (the wrong_* fields and node are dropped). A problem without a status-ok parse, or whose chain build_gsm8k_si
leaves null, gets worked_steps null (the probe skips it).

Check before writing (exit 1 on any difference): the same call with the same stand-in rows on every row of
data/student_gsm8k_tf_si/{train,val}.jsonl (raw train parquet) gives exactly their stored chains' text and value,
line by line, and a null chain exactly where theirs is null, except rows whose stored chain is null only because no
WRONG value passed the option / node constraints of their real row (null_no_wrong_value when rebuilt from it; counted
as stored_null_no_wrong_value: 3 of 6,316). Their true lines are well defined, so a test problem with that pattern
keeps its chain.

  python3 scripts/build_gsm8k_test_steps.py      # system python3 (pandas, numpy); a few seconds
  -> data/eval/gsm8k_mc_v2_steps.jsonl (every gsm8k_mc_v2 row, fields unchanged, plus worked_steps last)
     data/eval/gsm8k_mc_v2_steps.summary.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

KEEP = ("id", "step", "op_index", "kind", "a", "op", "b", "value", "text", "a_from", "b_from")


def chain(SI, S, i: int, question: str, answer: str, prompt: str, pid: str, st: collections.Counter):
    """build_gsm8k_si's worked steps of one raw problem (stand-in row: no value nodes, no options), or None."""
    p = S.parse_problem(i, question, answer)
    if p["status"] != "ok":
        st[f"parse_{p['status']}"] += 1
        return None
    item = {"problem_id": pid, "prompt": prompt, "value_nodes": [], "options": []}
    return SI.worked_steps(p, item, 0, st)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/eval/gsm8k_mc_v2.jsonl")
    ap.add_argument("--out", default="data/eval/gsm8k_mc_v2_steps.jsonl")
    ap.add_argument("--check", default="data/student_gsm8k_tf_si", help="rows whose stored chains must be reproduced")
    args = ap.parse_args()
    import pandas as pd
    import build_gsm8k_si as SI
    S = SI.T.soc()
    raw = SI.T.RAW
    fails = []
    # ---- check: the training rows' chains, recomputed with the stand-in rows
    tr = pd.read_parquet(raw / "train-00000-of-00001.parquet")
    chk = collections.Counter()
    for split in ("train", "val"):
        for line in open(ROOT / args.check / f"{split}.jsonl"):
            if not line.strip():
                continue
            r = json.loads(line)
            i = int(r["problem_id"].rsplit("-", 1)[1])
            got = chain(SI, S, i, tr.question[i], tr.answer[i], r["prompt"], r["problem_id"], collections.Counter())
            want = r.get("worked_steps")
            chk["rows"] += 1
            if got is not None and want is None:     # stored null only for want of a WRONG value (options / nodes)?
                real = collections.Counter()
                if SI.worked_steps(S.parse_problem(i, tr.question[i], tr.answer[i]), r, 0, real) is None and \
                        real["null_no_wrong_value"] == 1:
                    chk["stored_null_no_wrong_value"] += 1
                    continue
            if (got is None) != (want is None):
                fails.append(f"{r['item_id']}: chain {'null' if got is None else 'built'}, stored "
                             f"{'null' if want is None else 'built'}")
            elif got is not None and [(x["text"], x["value"]) for x in got] != [(x["text"], x["value"]) for x in want]:
                fails.append(f"{r['item_id']}: chain lines differ from the stored ones")
            else:
                chk["same"] += 1
                chk["with_chain"] += want is not None
    print("check", dict(chk), f"{len(fails)} differences", flush=True)
    # ---- the test problems
    te = pd.read_parquet(raw / "test-00000-of-00001.parquet")
    st = collections.Counter()
    rows = [json.loads(l) for l in open(ROOT / args.eval) if l.strip()]
    out = []
    for r in rows:
        i = int(r["orig_id"].rsplit("-", 1)[1])
        q = te.question[i].strip()
        if f"Question: {q}\nOptions:\n" != r["prompt"][:len(q) + 20]:
            fails.append(f"{r['item_id']}: raw test row {i} does not reproduce the prompt")
            continue
        c = chain(SI, S, i, te.question[i], te.answer[i], r["prompt"], r["orig_id"], st)
        o = dict(r)
        o["worked_steps"] = [{k: x[k] for k in KEEP} for x in c] if c else None
        st["rows"] += 1
        st["with_chain"] += c is not None
        out.append(o)
    lens = collections.Counter(len(o["worked_steps"]) for o in out if o["worked_steps"])
    summary = {"script": "scripts/build_gsm8k_test_steps.py", "args": vars(args), "check": dict(chk),
               "build": dict(sorted(st.items())), "chain_lengths": dict(sorted(lens.items())),
               "failures": len(fails), "examples": fails[:20]}
    print(json.dumps(summary, indent=1))
    if fails:
        sys.exit("NOT WRITTEN: " + "; ".join(fails[:10]))
    text = "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out)
    Path(ROOT / args.out).write_text(text, encoding="utf-8")
    summary["md5"] = hashlib.md5(text.encode("utf-8")).hexdigest()
    Path(ROOT / args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote", args.out, summary["md5"])


if __name__ == "__main__":
    main()
