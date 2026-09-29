"""Measure peak GPU memory and throughput of full fine-tuning a causal LM (random tokens).

Runs three optimizer/precision settings and reports peak allocated memory, seconds per step
and tokens per second. The loss reads logits at the last position only, like our readout.
"""
import argparse
import time

import torch
import transformers


def load(path: str, dtype):
    """Text-only causal LM if available; else the multimodal wrapper with the vision tower frozen."""
    try:
        m = transformers.AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    except Exception as e:  # Qwen3.5 small models ship as image-text-to-text
        print("AutoModelForCausalLM failed:", repr(e)[:160], flush=True)
        m = transformers.AutoModelForImageTextToText.from_pretrained(path, torch_dtype=dtype)
    for n, p in m.named_parameters():
        if "visual" in n or "vision" in n:
            p.requires_grad = False
    return m

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--seq", type=int, default=1024)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--modes", default="fp32master_adamw_ckpt,bf16_adamw_ckpt,bf16_adamw8bit_ckpt")
a = ap.parse_args()

print(torch.__version__, torch.cuda.get_device_name(0),
      f"free/total GiB: {[round(x / 2**30, 1) for x in torch.cuda.mem_get_info()]}", flush=True)

print("transformers", transformers.__version__, flush=True)
for mode in a.modes.split(","):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dtype = torch.float32 if mode.startswith("fp32") else torch.bfloat16
    model = load(a.model, dtype).cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"{mode}: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params, "
          f"{sum(p.numel() for p in params)/1e9:.2f}B trainable", flush=True)
    model.config.use_cache = False
    if "nockpt" not in mode:
        model.gradient_checkpointing_enable()
    if "adamw8bit" in mode:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=1e-5)
    else:
        opt = torch.optim.AdamW(params, lr=1e-5, fused=True)
    cfg = model.config
    vocab = getattr(cfg, "vocab_size", None) or cfg.get_text_config().vocab_size
    x = torch.randint(0, vocab, (a.bs, a.seq), device="cuda")
    try:
        t0 = time.time()
        for s in range(a.steps):
            if s == 1:
                torch.cuda.synchronize()
                t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                try:
                    logits = model(input_ids=x, logits_to_keep=1).logits[:, -1, :].float()
                except TypeError:
                    logits = model(input_ids=x).logits[:, -1, :].float()
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
    except Exception as e:
        print(f"{mode}: FAILED {e!r}"[:300], flush=True)
    del model, opt
    torch.cuda.empty_cache()
