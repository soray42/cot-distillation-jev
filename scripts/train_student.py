"""Train one student arm and evaluate it on held-out sets (final question only).

Example (HPC):
  python scripts/train_student.py --model ~/cotd/models/Qwen3.5-2B-Base \
      --train data/student/train.jsonl --eval val=data/student/val.jsonl \
      --final teacher --subq cot --subq-target cot --out runs/A1S1-seed0 --seed 0

Arms: --final {none,teacher,gold} x --subq {none,cot,random}. Writes metrics.json and preds_<set>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.student import (Example, build_examples, evaluate, kl_loss, label_logits,  # noqa: E402
                                load_model)


def read_jsonl(p: str) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", nargs="*", default=[], help="name=path.jsonl")
    ap.add_argument("--final", default="teacher", choices=["none", "teacher", "gold"])
    ap.add_argument("--subq", default="none", choices=["none", "cot", "random"])
    ap.add_argument("--subq-target", default="cot", choices=["fresh", "cot", "truth"])
    ap.add_argument("--lambda-sub", type=float, default=1.0)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--precision", default="fp32master", choices=["fp32master", "bf16"])
    ap.add_argument("--optim", default="adamw", choices=["adamw", "adamw8bit"])
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--eval-bs", type=int, default=8)
    ap.add_argument("--eval-max-len", type=int, default=6144, help="long eval states (JevBench hard) need more room")
    ap.add_argument("--save", action="store_true", help="save the final weights (bf16) to <out>/model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float32 if args.precision == "fp32master" else torch.bfloat16
    model = load_model(args.model, dtype).to(dev)
    model.config.use_cache = False
    if not args.no_grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, fused=dev == "cuda")

    train_items = read_jsonl(args.train)
    evals = {kv.split("=", 1)[0]: read_jsonl(kv.split("=", 1)[1]) for kv in args.eval}

    def epoch_examples() -> list[Example]:
        ex = []
        for it in train_items:
            ex += build_examples(it, final=args.final, subq=args.subq, subq_target=args.subq_target,
                                 lambda_sub=args.lambda_sub, rng=rng)
        rng.shuffle(ex)
        return ex

    n_per_epoch = len(epoch_examples())
    total_steps = max(1, math.ceil(args.epochs * n_per_epoch / (args.micro_bs * args.grad_accum)))
    warm = max(1, int(args.warmup * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    print(f"examples/epoch={n_per_epoch} steps={total_steps} params={sum(p.numel() for p in params)/1e9:.2f}B "
          f"device={dev}", flush=True)

    cache: dict = {}
    step, micro, t0, seen_tok = 0, 0, time.time(), 0
    history = []
    ex_iter: list[Example] = []
    model.train()
    while step < total_steps:
        if len(ex_iter) < args.micro_bs:
            ex_iter += epoch_examples()
        batch, ex_iter = ex_iter[:args.micro_bs], ex_iter[args.micro_bs:]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            zs = label_logits(model, tok, batch, args.max_len, cache)
            loss = kl_loss(zs, batch) / args.grad_accum
        loss.backward()
        seen_tok += sum(min(args.max_len, len(e.text) // 3) for e in batch)   # rough token count for logging
        micro += 1
        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0 or step == total_steps:
                dt = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0.0
                row = {"step": step, "loss": loss.item() * args.grad_accum, "lr": sched.get_last_lr()[0],
                       "s_per_step": dt / step, "approx_tok_per_s": seen_tok / dt, "peak_gib": mem}
                history.append(row)
                print(json.dumps(row), flush=True)

    results = {"args": vars(args), "history": history, "eval": {}}
    for name, items in evals.items():
        m, preds = evaluate(model, tok, items, args.eval_max_len, args.eval_bs, cache)
        results["eval"][name] = m
        with open(out / f"preds_{name}.jsonl", "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        print(name, json.dumps(m), flush=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    if args.save:
        model.to(torch.bfloat16).save_pretrained(out / "model", safe_serialization=True)
        tok.save_pretrained(out / "model")


if __name__ == "__main__":
    main()
