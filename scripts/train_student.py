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

from cotdistill.student import (Example, brier_loss, build_examples, build_stage_examples, evaluate,  # noqa: E402
                                evaluate_subq, kl_loss, label_logits, lm_loss, load_model)


def read_jsonl(p: str) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", nargs="*", default=[], help="name=path.jsonl")
    ap.add_argument("--final", default="teacher", choices=["none", "teacher", "gold"])
    ap.add_argument("--subq", default="none", choices=["none", "cot", "random", "mix"])
    ap.add_argument("--subq-target", default="cot", choices=["fresh", "cot", "truth", "commit"])
    ap.add_argument("--lambda-sub", type=float, default=1.0)
    ap.add_argument("--rationale-lm", action="store_true", help="DHRD-style baseline: LM loss on teacher CoT + answer")
    ap.add_argument("--lambda-lm", type=float, default=1.0)
    ap.add_argument("--lm-max-len", type=int, default=3072)
    ap.add_argument("--subq-k", type=int, default=0, help="at most K random sub-questions per item and epoch (0 = all)")
    ap.add_argument("--subq-frac", type=float, default=0.0,
                    help="use a random fraction of each item's sub-questions per epoch (0 = all)")
    ap.add_argument("--subq-weight", default="split", choices=["split", "each"],
                    help="split: lambda-sub shared by an item's sub-questions; each: every sub-question weighs lambda-sub")
    ap.add_argument("--depth-stages", type=int, default=0,
                    help="depth curriculum: N sub-question stages ordered by tree depth (each 1 pass over the items, "
                         "--stage-k nodes per item), then the final questions for --epochs; 0 = off")
    ap.add_argument("--stage-k", type=int, default=1, help="sub-questions per item in each depth stage")
    ap.add_argument("--final-replay", type=int, default=0,
                    help="depth curriculum: also train this many tree nodes per item and epoch in the final stage")
    ap.add_argument("--level-balanced", action="store_true", help="depth curriculum: draw replay nodes level-uniformly")
    ap.add_argument("--reset-optim", action="store_true", help="depth curriculum: reset the AdamW state at each stage")
    ap.add_argument("--lambda-brier", type=float, default=0.0, help="add this x Brier score (vs the soft target)")
    ap.add_argument("--permute-final", type=float, default=0.0,
                    help="probability of reordering a final question's options each epoch (target follows the texts)")
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
    ap.add_argument("--dump-hidden", default="", help="comma list of eval-set names whose readout hidden states "
                    "to save as hidden_<name>.pt (float16, preds order)")
    ap.add_argument("--eval-subq", default=None, help="student data file (e.g. val.jsonl) whose sub-questions to score")
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
                                 lambda_sub=args.lambda_sub, rng=rng, rationale_lm=args.rationale_lm,
                                 lambda_lm=args.lambda_lm, permute_final=args.permute_final,
                                 subq_k=args.subq_k, subq_weight=args.subq_weight, subq_frac=args.subq_frac)
        rng.shuffle(ex)
        return ex

    def stage_examples(stage: int):
        def make() -> list[Example]:
            ex = []
            for it in train_items:
                ex += build_stage_examples(it, stage=stage, n_stages=args.depth_stages, k=args.stage_k, rng=rng,
                                           subq_target=args.subq_target, level_balanced=args.level_balanced)
            rng.shuffle(ex)
            return ex
        return make

    per_step = args.micro_bs * args.grad_accum
    if args.depth_stages:            # sub-question stages by depth, then the final questions, each with its own schedule
        stages = [(f"depth{s}", stage_examples(s), 1.0) for s in range(args.depth_stages)]
        def final_examples() -> list[Example]:
            ex = [e for it in train_items for e in build_examples(
                it, final=args.final, subq="none", subq_target=args.subq_target, lambda_sub=args.lambda_sub, rng=rng,
                permute_final=args.permute_final)]
            if args.final_replay:            # keep the whole tree in play while the final answer is calibrated
                ex += [e for it in train_items for e in build_stage_examples(
                    it, stage=args.depth_stages - 1, n_stages=args.depth_stages, k=args.final_replay, rng=rng,
                    subq_target=args.subq_target, p_new=0.0, level_balanced=args.level_balanced)]
            rng.shuffle(ex)
            return ex
        stages.append(("final", final_examples, args.epochs))
    else:
        stages = [("all", epoch_examples, args.epochs)]
    plan = []
    for name, make, epochs in stages:
        n = len(make())
        plan.append((name, make, n, max(1, math.ceil(epochs * n / per_step))))
        print(f"stage {name}: {n} examples per pass, {plan[-1][3]} steps", flush=True)
    total_steps = sum(x[3] for x in plan)
    print(f"examples/epoch={plan[-1][2]} steps={total_steps} params={sum(p.numel() for p in params)/1e9:.2f}B "
          f"device={dev}", flush=True)

    cache: dict = {}
    step, micro, t0, seen_tok = 0, 0, time.time(), 0
    history = []
    model.train()
    for si, (name, make, _, stage_steps) in enumerate(plan):
        if args.reset_optim and si > 0:
            opt.state.clear()
        warm = max(1, int(args.warmup * stage_steps))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s, w=warm, t=stage_steps: min(1.0, (s + 1) / w) * 0.5 * (
            1 + math.cos(math.pi * min(1.0, s / t))))
        ex_iter: list[Example] = []
        end = step + stage_steps
        while step < end:
            if len(ex_iter) < args.micro_bs:
                ex_iter += make()
            batch, ex_iter = ex_iter[:args.micro_bs], ex_iter[args.micro_bs:]
            cls = [e for e in batch if e.kind != "lm"]
            lms = [e for e in batch if e.kind == "lm"]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                total = 0.0
                if cls:
                    z = label_logits(model, tok, cls, args.max_len, cache)
                    total = kl_loss(z, cls) * len(cls)
                    if args.lambda_brier:
                        total = total + args.lambda_brier * brier_loss(z, cls) * len(cls)
                if lms:
                    total = total + lm_loss(model, tok, lms, args.lm_max_len)
                loss = total / len(batch) / args.grad_accum
            loss.backward()
            seen_tok += sum(min(args.max_len, len(e.text) // 3) for e in cls) + \
                sum(min(args.lm_max_len, (len(e.text) + len(e.continuation)) // 3) for e in lms)  # rough count
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
        hid = [] if name in args.dump_hidden.split(",") else None
        m, preds = evaluate(model, tok, items, args.eval_max_len, args.eval_bs, cache, hidden=hid)
        if hid is not None:
            torch.save(torch.stack(hid).half(), out / f"hidden_{name}.pt")
        results["eval"][name] = m
        with open(out / f"preds_{name}.jsonl", "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        print(name, json.dumps(m), flush=True)
    if args.eval_subq:
        sm, spreds = evaluate_subq(model, tok, read_jsonl(args.eval_subq), args.max_len, args.eval_bs, cache, seed=args.seed)
        results["eval"]["subq"] = sm
        with open(out / "preds_subq.jsonl", "w") as f:
            for p in spreds:
                f.write(json.dumps(p) + "\n")
        print("subq", json.dumps({k: round(v["acc"], 3) for k, v in sm.items()}), flush=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    if args.save:
        model.to(torch.bfloat16).save_pretrained(out / "model", safe_serialization=True)
        tok.save_pretrained(out / "model")


if __name__ == "__main__":
    main()
