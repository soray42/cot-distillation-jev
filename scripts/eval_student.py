"""Evaluate a model (untrained base control or a saved student) on eval sets, final question only.

  python scripts/eval_student.py --model ~/cotd/models/Qwen3.5-2B-Base \
      --eval jevbench=data/eval/jevbench_public.jsonl kk=data/eval/kk_heldout.jsonl --out runs/base-2B

Writes metrics.json and preds_<set>.jsonl; scripts/score_preds.py gives per-group breakdowns.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.student import evaluate, evaluate_subq, load_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", nargs="+", required=True, help="name=path.jsonl")
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-subq", default=None, help="student data file whose sub-questions to score")
    args = ap.parse_args()

    import transformers
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.model, torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
    results, cache = {"args": vars(args), "eval": {}}, {}
    for kv in args.eval:
        name, path = kv.split("=", 1)
        items = [json.loads(l) for l in open(path) if l.strip()]
        m, preds = evaluate(model, tok, items, args.max_len, args.bs, cache)
        results["eval"][name] = m
        with open(out / f"preds_{name}.jsonl", "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        print(name, json.dumps(m), flush=True)
    if args.eval_subq:
        items = [json.loads(l) for l in open(args.eval_subq) if l.strip()]
        sm, spreds = evaluate_subq(model, tok, items, args.max_len, args.bs, cache)
        results["eval"]["subq"] = sm
        with open(out / "preds_subq.jsonl", "w") as f:
            for p in spreds:
                f.write(json.dumps(p) + "\n")
        print("subq", json.dumps({k: round(v["acc"], 3) for k, v in sm.items()}), flush=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
