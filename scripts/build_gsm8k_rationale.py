"""Rationale-LM baseline data for GSM8K (arm A7, Distilling Step-by-Step style): data/student_gsm8k_tf with the
`rationale` field filled from GSM8K's own human solution text, every other byte of every row unchanged.

Arm A7 (hpc/train_arm.sbatch: F=teacher, S=none, T=cot, --rationale-lm --lm-max-len 3072) trains each problem's final
question exactly as A2 does and adds one token-level LM example per row (cotdistill.student.build_examples): the text
RATIONALE_PREFIX = "{prompt}\\n\\nReasoning:" followed by the scored continuation
" {rationale.strip()}\\n\\nAnswer: {argmax teacher letter}". The LM example is only built when `rationale` is non-empty,
and in data/student_gsm8k_tf it is "" for every row, so A7 on that directory is A2. This script fills it.

Source and matching: item_id "gsm8k-train-<i>" is row i of the GSM8K train split (data/raw/gsm8k/train.parquet, md5
identical to data/raw/bench_math/gsm8k/main/train-00000-of-00001.parquet, which scripts/build_gsm8k_socratic.py
enumerates to name the items). The row's question must equal the prompt's "Question: ..." text exactly (checked).

Rationale = the GSM8K solution with
  - every calculator note <<expr=value>> removed ("5 - 2 = <<5-2=3>>3" -> "5 - 2 = 3"; GSM8K has no nested notes), and
  - the closing "#### <answer>" line dropped, because build_examples appends "\\n\\nAnswer: <letter>" itself (a kept
    "#### 6" would give the model two answer lines, the second one a letter). The value of the dropped line is checked
    against the gold option text (equal for every row; 68 rows differ only by thousands commas, e.g. "4,000" vs
    "4000"), so the rationale always ends at the step that computes the gold value.
Only rows with row_kind "final" get a rationale (all rows of student_gsm8k_tf are finals); every other field is
byte-identical (checked: the base rows re-serialise to the base files byte for byte, and each output row equals its base
row with only `rationale` replaced). val.jsonl is filled the same way (its items are GSM8K train problems held out
from training; the evaluation never reads `rationale`).

Two steps, two environments (pandas lives in the system python, transformers in the training venv):
  python3 scripts/build_gsm8k_rationale.py                       # build data/student_gsm8k_tf_rat/{train,val}.jsonl
  CUDA_VISIBLE_DEVICES= ~/.venvs/sev/bin/python scripts/build_gsm8k_rationale.py --token-stats \\
      results/weights/T1-Qwen3.5-2B-Base-student_v3-s2/model       # add LM token lengths to summary.json
The second step reads the built files only (md5s unchanged) and measures each A7 LM example the way
cotdistill.student.lm_loss does: len(encode_tail(prompt view, lm_max_len // 2)) + len(continuation tokens), against
--lm-max-len (default 3072), and counts the continuations lm_loss would cut (start + last 64 tokens).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALC_NOTE = re.compile(r"<<[^<>]*>>")
ITEM_ID = re.compile(r"^gsm8k-train-(\d+)$")
FILES = ("train.jsonl", "val.jsonl")


def clean_solution(answer: str) -> tuple[str, str]:
    """(rationale, final answer string) of a GSM8K solution: calculator notes removed, the closing '#### x' line
    split off. Raises ValueError if the solution does not end in exactly one '#### ' line or a note survives."""
    lines = answer.rstrip("\n").split("\n")
    if not lines[-1].startswith("#### ") or any(l.startswith("####") for l in lines[:-1]):
        raise ValueError(f"no single closing '#### ' line: {answer[-80:]!r}")
    final = lines[-1][5:].strip()
    body = CALC_NOTE.sub("", "\n".join(lines[:-1]))
    body = "\n".join(l.rstrip() for l in body.split("\n")).strip()
    if "<<" in body or ">>" in body or "####" in body or not body:
        raise ValueError(f"unclean rationale: {body[:120]!r}")
    return body, final


def same_number(a: str, b: str) -> bool:
    """Equal as written, or equal once thousands commas are dropped ('4,000' vs '4000')."""
    if a == b:
        return True
    try:
        return float(a.replace(",", "")) == float(b.replace(",", ""))
    except ValueError:
        return False


def question_of(prompt: str) -> str:
    head = prompt.split("\nOptions:\n", 1)[0]
    if not head.startswith("Question: "):
        raise ValueError(f"prompt does not start with 'Question: ': {prompt[:80]!r}")
    return head[len("Question: "):]


def dumps(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False)       # as scripts/build_gsm8k_tf.py writes the base files


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> tuple[bytes, list[dict]]:
    raw = path.read_bytes()
    rows = [json.loads(l) for l in raw.decode("utf-8").split("\n") if l.strip()]
    return raw, rows


def fill(rows: list[dict], solutions: list[tuple[str, str]]) -> tuple[list[dict], dict]:
    """Rows with `rationale` filled (finals only) and the file's counts. solutions[i] = (question, answer) of GSM8K
    train row i."""
    out, st = [], {"items": len(rows), "finals": 0, "with_rationale": 0, "notes_removed": 0,
                   "answer_exact": 0, "answer_comma_only": 0}
    chars, nlines = [], []
    for r in rows:
        if r.get("row_kind") != "final":
            out.append(r)
            continue
        st["finals"] += 1
        m = ITEM_ID.match(r["item_id"])
        if not m:
            raise ValueError(f"{r['item_id']}: not a gsm8k-train-<i> id")
        q, ans = solutions[int(m.group(1))]
        if question_of(r["prompt"]) != q.strip():
            raise ValueError(f"{r['item_id']}: prompt question differs from GSM8K train row {m.group(1)}")
        if r.get("rationale"):
            raise ValueError(f"{r['item_id']}: base row already has a rationale")
        rat, final = clean_solution(ans)
        gold = r["options"][r["labels"].index(r["gold_label"])]
        if not same_number(final, gold):
            raise ValueError(f"{r['item_id']}: GSM8K answer {final!r} is not the gold option {gold!r}")
        st["answer_exact" if final == gold else "answer_comma_only"] += 1
        st["notes_removed"] += len(CALC_NOTE.findall(ans))
        new = dict(r)                                    # same key order; only the value of "rationale" changes
        new["rationale"] = rat
        out.append(new)
        st["with_rationale"] += 1
        chars.append(len(rat))
        nlines.append(rat.count("\n") + 1)
    st["rationale_chars"] = {"mean": round(statistics.mean(chars), 1), "p50": statistics.median(chars),
                             "max": max(chars)} if chars else {}
    st["rationale_lines"] = {"mean": round(statistics.mean(nlines), 2), "max": max(nlines)} if nlines else {}
    for a, b in zip(rows, out):                          # nothing but the rationale changed
        if {k: v for k, v in a.items() if k != "rationale"} != {k: v for k, v in b.items() if k != "rationale"} \
                or list(a) != list(b):
            raise AssertionError(f"{a['item_id']}: a field other than rationale changed")
    return out, st


def build(args) -> None:
    import pandas as pd                                  # system python3 (the training venv has no pandas)
    df = pd.read_parquet(ROOT / args.raw)
    solutions = list(zip(df.question, df.answer))
    base, out = ROOT / args.base, ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = {"script": "scripts/build_gsm8k_rationale.py", "args": vars(args), "md5": {}, "base_md5": {},
               "raw_md5": md5(ROOT / args.raw), "changed_fields": ["rationale"], "files": {}}
    for name in FILES:
        raw, rows = read_rows(base / name)
        if "".join(dumps(r) + "\n" for r in rows).encode("utf-8") != raw:
            raise SystemExit(f"{base / name}: rows do not re-serialise byte for byte; the writer would change fields")
        new, st = fill(rows, solutions)
        if st["with_rationale"] != st["items"]:
            raise SystemExit(f"{name}: {st['items'] - st['with_rationale']} rows without a rationale")
        dst = out / name
        dst.write_text("".join(dumps(r) + "\n" for r in new), encoding="utf-8")
        tag = f"{out.name}/{name}"
        summary["md5"][tag] = md5(dst)
        summary["base_md5"][f"{base.name}/{name}"] = md5(base / name)
        summary["files"][tag] = st
        print(tag, json.dumps(st), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote", out / "summary.json")


def pct(xs: list[int], q: float) -> int:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def token_stats(args) -> None:
    """LM-example token lengths of A7, measured through cotdistill.student (no model)."""
    import random
    sys.path.insert(0, str(ROOT / "src"))
    import transformers
    from cotdistill.student import build_examples, encode_tail
    tok = transformers.AutoTokenizer.from_pretrained(args.token_stats)
    out = ROOT / args.out
    summary = json.loads((out / "summary.json").read_text())
    for tag in summary["md5"]:
        if md5(out / Path(tag).name) != summary["md5"][tag]:
            raise SystemExit(f"{tag}: md5 differs from summary.json; rebuild first")
    rep = {"tokenizer": args.token_stats, "lm_max_len": args.lm_max_len, "files": {}}
    for name in FILES:
        _, rows = read_rows(out / name)
        tot, pre_n, cont_n, finals = [], [], [], []
        cut = 0
        for r in rows:
            ex = build_examples(r, final="teacher", subq="none", subq_target="cot", lambda_sub=1.0,
                                rng=random.Random(0), rationale_lm=True)
            lm = [e for e in ex if e.kind == "lm"]
            if not lm:
                continue
            e = lm[0]
            pre = encode_tail(tok, e.text, args.lm_max_len // 2)          # as lm_loss
            cont = tok.encode(e.continuation, add_special_tokens=False)
            cut += len(cont) > args.lm_max_len - len(pre)
            pre_n.append(len(pre))
            cont_n.append(len(cont))
            tot.append(len(pre) + len(cont))
            fin = [x for x in ex if x.kind == "final"][0]
            finals.append(len(encode_tail(tok, fin.text, 1536)))
        def desc(xs):
            return {"n": len(xs), "mean": round(statistics.mean(xs), 1), "p50": pct(xs, 0.5), "p90": pct(xs, 0.9),
                    "p99": pct(xs, 0.99), "max": max(xs)}
        rep["files"][name] = {"lm_example_tokens": desc(tot), "prompt_view_tokens": desc(pre_n),
                              "continuation_tokens": desc(cont_n), "final_view_tokens": desc(finals),
                              "share_le_lm_max_len": round(sum(t <= args.lm_max_len for t in tot) / len(tot), 4),
                              "continuations_cut": cut,
                              "epoch_tokens_lm": sum(tot), "epoch_tokens_final": sum(finals)}
        print(name, json.dumps(rep["files"][name]), flush=True)
    summary["tokens"] = rep
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("added token stats to", out / "summary.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="data/student_gsm8k_tf")
    ap.add_argument("--out", default="data/student_gsm8k_tf_rat")
    ap.add_argument("--raw", default="data/raw/gsm8k/train.parquet")
    ap.add_argument("--token-stats", default="", metavar="TOKENIZER_DIR",
                    help="instead of building: add A7 LM-example token lengths of the built files to summary.json")
    ap.add_argument("--lm-max-len", type=int, default=3072, help="train_student.py --lm-max-len (A7: 3072)")
    args = ap.parse_args()
    if args.token_stats:
        token_stats(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
