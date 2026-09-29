"""Measure peak GPU memory and throughput of full fine-tuning a causal LM (random tokens).

Runs three optimizer/precision settings and reports peak allocated memory, seconds per step
and tokens per second. The loss reads logits at the last position only, like our readout.
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--seq", type=int, default=1024)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--steps", type=int, default=6)
a = ap.parse_args()

print(torch.__version__, torch.cuda.get_device_name(0),
      f"free/total GiB: {[round(x / 2**30, 1) for x in torch.cuda.mem_get_info()]}", flush=True)

for mode in ["fp32master_adamw_ckpt", "fp32master_adamw_nockpt", "bf16_adamw8bit_ckpt"]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dtype = torch.float32 if mode.startswith("fp32") else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).cuda()
    model.config.use_cache = False
    if "nockpt" not in mode:
        model.gradient_checkpointing_enable()
    if "adamw8bit" in mode:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
    x = torch.randint(0, model.config.vocab_size, (a.bs, a.seq), device="cuda")
    try:
        t0 = time.time()
        for s in range(a.steps):
            if s == 1:
                torch.cuda.synchronize()
                t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=x, logits_to_keep=1).logits[:, -1, :].float()
                loss = torch.nn.functional.cross_entropy(logits, x[:, 0])
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / (a.steps - 1)
        print(f"{mode}: peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
              f"{dt:.2f} s/step, {a.bs * a.seq / dt:.0f} tok/s", flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f"{mode}: OOM", flush=True)
    del model, opt
    torch.cuda.empty_cache()
