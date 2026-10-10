"""Evaluate a model (untrained base control or a saved student) on eval sets, final question only.

  python scripts/eval_student.py --model ~/cotd/models/Qwen3.5-2B-Base \
      --eval jevbench=data/eval/jevbench_public.jsonl kk=data/eval/kk_heldout.jsonl --out runs/base-2B

Writes metrics.json and preds_<set>.jsonl; scripts/score_preds.py gives per-group breakdowns.

--time (opt-in): forward-only seconds per item for a practicality table (e.g. StepFinder's seconds per trajectory):
every item of every set alone (batch 1, the readout forward of evaluate, no generation), after one warm-up item,
CUDA-synchronized wall time per item -> timing_<set>.json (mean, median, p90, max, total, prompt tokens, device).
The predictions and metrics are written as without the flag.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.student import (FINAL_TEMPLATE, encode_tail, evaluate, evaluate_subq, load_model,  # noqa: E402
                                subq_sets, use_chat_template)


def time_items(model, tok, items: list[dict], max_len: int, cache: dict) -> dict:
    """--time: seconds per item, each item alone (batch 1), after one untimed warm-up item; CUDA synchronized."""
    import time
    import numpy as np
    dev = next(model.parameters()).device
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    evaluate(model, tok, items[:1], max_len, 1, cache)                 # warm-up (kernels, allocator, label ids)
    secs, ntok = [], []
    for it in items:
        sync()
        t = time.perf_counter()
        evaluate(model, tok, [it], max_len, 1, cache)
        sync()
        secs.append(time.perf_counter() - t)
        ntok.append(len(encode_tail(tok, FINAL_TEMPLATE.format(problem=it["prompt"]), max_len)))
    a = np.array(secs)
    return {"n": len(secs), "s_mean": float(a.mean()), "s_median": float(np.median(a)),
            "s_p90": float(np.percentile(a, 90)), "s_max": float(a.max()), "s_total": float(a.sum()),
            "tokens_mean": float(np.mean(ntok)), "tokens_max": int(max(ntok)), "batch": 1, "max_len": max_len,
            "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
            "dtype": str(next(model.parameters()).dtype), "per_item_s": [round(x, 4) for x in secs]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", nargs="+", required=True, help="name=path.jsonl")
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-layers", default="", help="comma list of decoder layers whose readout states are saved "
                    "with --dump-hidden (shape [n, layers + 1, hidden]; the last entry is the final readout state)")
    ap.add_argument("--dump-hidden", default="", help="comma list of eval-set names whose readout hidden states "
                    "to save as hidden_<name>.pt (float16, preds order)")
    ap.add_argument("--eval-subq", default=None, help="student data file whose sub-questions to score, or "
                    "comma-separated name=path entries (predictions in preds_subq[_name].jsonl)")
    ap.add_argument("--chat-template", action="store_true",
                    help="read out every item as one user turn of the model's chat template, thinking off, at the "
                         "assistant turn's first token (bare letters), as train_student.py --chat-template; off: "
                         "unchanged")
    ap.add_argument("--time", action="store_true", help="also time every item alone (batch 1, forward only): "
                    "timing_<set>.json")
    args = ap.parse_args()

    import transformers
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if args.chat_template:
        print("chat template: every item rendered as", repr(use_chat_template(tok, args.model)), flush=True)
    model = load_model(args.model, torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
    results, cache = {"args": vars(args), "eval": {}}, {}
    for kv in args.eval:
        name, path = kv.split("=", 1)
        items = [json.loads(l) for l in open(path) if l.strip()]
        hid = [] if name in args.dump_hidden.split(",") else None
        m, preds = evaluate(model, tok, items, args.max_len, args.bs, cache, hidden=hid,
                            layers=[int(x) for x in args.dump_layers.split(",") if x] or None)
        if hid is not None:
            torch.save(torch.stack(hid).half(), out / f"hidden_{name}.pt")
        results["eval"][name] = m
        with open(out / f"preds_{name}.jsonl", "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        print(name, json.dumps(m), flush=True)
        if args.time:
            tm = time_items(model, tok, items, args.max_len, cache)
            (out / f"timing_{name}.json").write_text(json.dumps(tm, indent=1))
            results.setdefault("timing", {})[name] = {k: v for k, v in tm.items() if k != "per_item_s"}
            print("timing", name, json.dumps(results["timing"][name]), flush=True)
    for sname, spath in subq_sets(args.eval_subq or ""):
        items = [json.loads(l) for l in open(spath) if l.strip()]
        sm, spreds = evaluate_subq(model, tok, items, args.max_len, args.bs, cache)
        results["eval"][sname] = sm
        with open(out / f"preds_{sname}.jsonl", "w") as f:
            for p in spreds:
                f.write(json.dumps(p) + "\n")
        print(sname, json.dumps({k: round(v["acc"], 3) for k, v in sm.items()}), flush=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
