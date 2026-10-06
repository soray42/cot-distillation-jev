"""Memory / speed / resume smoke for full-parameter training of Gemma 4 12B (text only) on one 40 GB MIG slice.

Standalone (does not use train_student.py): bf16 weights on the GPU, activation checkpointing, CPUOffloadAdamW
(src/cotdistill/offload.py: fp32 masters, accumulators and moments on the CPU), a loss on the option-letter logits at
the last position only (soft-capped like the HF forward), synthetic worst-case micro-batches. One process, one model
load, one optimizer; every part prints one or more "MEM12B {json}" lines.

Parts (--parts, run in this order):
  check     config + tokenizer only (no torch.cuda, no weights): model type, soft-cap, BOS prefix, letter ids, eval
            files, safetensors size, free disk; the login-node dry run uses this part alone.
  bench     pinned 1 GiB D2H / H2D GB/s on this slice; CPU fused-AdamW and bf16->fp32 accumulate Gparam/s at
            --threads (the two [est] inputs of notes/next_round/scale12b/offload.md section 4).
  readout   zero-shot, before any update: the first --readout-n items of each --readout-files file with the raw
            "<prompt>\\n\\nAnswer:" prompt (BOS + text): accuracy over the item's letters, mean letter probability mass
            in the full-vocabulary soft-capped softmax, raw (uncapped) letter-logit stats, the same without BOS, and a
            check that the readout equals the model's own logits (HF forward, logits_to_keep=1) on the first item.
            On bf16 weights HF computes lm_head and the cap in bf16 and this readout in fp32, so hf_check differs by
            up to about one bf16 ulp (0.125-0.25 at |z| 16-64); that is not a mismatch (the fp32 CPU test is exact).
  baseline  pure GPU forward+backward per micro-batch of every setting, gradients discarded by a hook (no offload,
            no optimizer): the compute floor the offload overhead is measured against.
  mem       per setting BSxL[:inflightGiB]: --steps updates of --ga micro-batches; peak GPU GiB, host RSS GiB,
            s/update split into forward+backward (with the gradient offload inside) / drain+clip / CPU step + H2D,
            tokens/s; update 0 is reported apart (first touch of 48 GB of accumulators, lazy 96 GB of moments).
  roundtrip 2 updates, save the optimizer state (torch.save), 1 update ("straight"), load the saved state
            (torch.load mmap=True) into the same optimizer, the same update again ("resumed"): loaded state equals
            the saved one bit for bit, GPU weights equal the masters cast to bf16, and the straight and resumed
            updates give bit-identical weights / masters / moments. The compared update uses fixed seeded gradients
            fed through the real hook path ((p * c).sum().backward() per parameter: grad == c exactly), so GPU
            nondeterminism (sdpa backward) cannot blur the check. A second 12B optimizer (another 214 GB) is
            not affordable, so the load goes into the live one. Skipped when --rt-dir has too little free disk.

--tiny builds a 2-layer random model with the real config and tokenizer (vocabulary 262,144) instead of loading the
weights: the in-job smoke that runs every part in about a minute before the 12B run.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

GIB = 2 ** 30
FINAL_TEMPLATE = "{problem}\n\nAnswer:"          # = cotdistill.student.FINAL_TEMPLATE
T_START = time.time()


def emit(row: dict) -> None:
    print("MEM12B", json.dumps(row), flush=True)


def log(*a) -> None:
    print(f"[{time.time() - T_START:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------------------------------------- host / device stats
def host_mem() -> dict:
    """Current and peak resident set of this process (GiB), from /proc (Linux) or getrusage."""
    out = {}
    try:
        for line in open("/proc/self/status"):
            k, _, v = line.partition(":")
            key = {"VmRSS": "rss_gib", "VmHWM": "rss_peak_gib", "RssAnon": "rss_anon_gib", "RssFile": "rss_file_gib",
                   "VmPin": "pinned_gib", "VmLck": "locked_gib"}.get(k)
            if key:                     # RssFile: file pages (mmap'd checkpoint / weights), reclaimable
                out[key] = round(int(v.split()[0]) / 2 ** 20, 2)
    except OSError:
        pass
    if "rss_peak_gib" not in out:
        out["rss_peak_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)
    return out


def sync(dev) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def gpu_mem(dev) -> dict:
    if dev.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info(dev)
    return {"peak_alloc_gib": round(torch.cuda.max_memory_allocated(dev) / GIB, 2),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved(dev) / GIB, 2),
            "used_now_gib": round((total - free) / GIB, 2), "total_gib": round(total / GIB, 2)}


# ---------------------------------------------------------------------------------------------- model and tokenizer
def text_config(cfg):
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg


def softcap_of(model) -> float | None:
    return getattr(text_config(model.config), "final_logit_softcapping", None) or None


def strip_multimodal(cfg):
    """Gemma 4 Unified: drop the vision/audio sub-configs, so the model is the text decoder + tied head only
    (gemma_notes.md section 1; never Gemma4UnifiedForCausalLM.from_pretrained, which silently loads nothing)."""
    if getattr(cfg, "model_type", None) == "gemma4_unified":
        cfg.vision_config = None
        cfg.audio_config = None
    return cfg


def load_text_only(path: str, dtype, attn: str | None = None):
    import transformers
    cfg = strip_multimodal(transformers.AutoConfig.from_pretrained(path))
    kw = {"attn_implementation": attn} if attn else {}
    m, info = transformers.AutoModelForCausalLM.from_pretrained(path, config=cfg, dtype=dtype,
                                                                output_loading_info=True, **kw)
    if info.get("missing_keys"):
        raise SystemExit(f"weights missing for {len(info['missing_keys'])} keys, e.g. {sorted(info['missing_keys'])[:5]}"
                         " (a silently random model would make every number here meaningless)")
    return m, info


def tiny_config(path: str):
    """The real config shrunk to 2 layers (one sliding, one full; K = V full layer kept), real vocabulary."""
    import transformers
    cfg = strip_multimodal(transformers.AutoConfig.from_pretrained(path))
    tc = text_config(cfg)
    for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
                     num_key_value_heads=1, head_dim=32, global_head_dim=64, sliding_window=64,
                     layer_types=["sliding_attention", "full_attention"]).items():
        setattr(tc, k, v)
    return cfg


def build_tiny(cfg, dtype):
    import transformers
    torch.manual_seed(0)
    return transformers.AutoModelForCausalLM.from_config(cfg, dtype=dtype)


def text_parts(model):
    """(decoder returning last_hidden_state, output embedding) for causal or multimodal wrappers."""
    head = model.get_output_embeddings()
    inner = getattr(model, "model", model)
    body = getattr(inner, "language_model", None) or getattr(model, "language_model", None)
    if body is None:
        body = model.get_decoder() if hasattr(model, "get_decoder") else inner
    return body, head


def special_prefix(tok) -> list[int]:
    """Ids the tokenizer itself puts in front of a text (Gemma: [2] = <bos>; Qwen: [])."""
    w, wo = tok.encode("Answer:", add_special_tokens=True), tok.encode("Answer:", add_special_tokens=False)
    for k in range(len(w) - len(wo) + 1):
        if w[k:k + len(wo)] == wo:
            return list(w[:k])
    return []


def encode_tail(tok, text: str, max_len: int, prefix: list[int]) -> list[int]:
    """prefix + the last (max_len - len(prefix)) tokens of text (the question end is what matters)."""
    ids = tok.encode(text, add_special_tokens=False)
    return list(prefix) + ids[-(max_len - len(prefix)):]


def letter_ids(tok, letters) -> list[int]:
    out = []
    for L in letters:
        t = tok.encode(" " + L, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"' {L}' is not a single token: {t}")
        out.append(t[0])
    return out


def capped(z: torch.Tensor, cap: float | None) -> torch.Tensor:
    return cap * torch.tanh(z / cap) if cap else z


def body_hidden(body, input_ids, attention_mask=None):
    return body(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state


# ---------------------------------------------------------------------------------------------- readout sanity
@torch.no_grad()
def readout_items(model, tok, items: list[dict], max_len: int, *, use_prefix: bool = True,
                  hf_check: bool = False, vocab_chunk: int = 32768) -> dict:
    """Zero-shot letter readout on eval records (prompt, labels, gold_label), one item per forward (no padding)."""
    body, head = text_parts(model)
    W = head.weight
    dev = W.device
    cap = softcap_of(model)
    prefix = special_prefix(tok) if use_prefix else []
    res = collections.defaultdict(list)
    hf_diff = None
    for n, it in enumerate(items):
        ids = encode_tail(tok, FINAL_TEMPLATE.format(problem=it["prompt"]), max_len, prefix)
        x = torch.tensor([ids], device=dev)
        h = body_hidden(body, x)[0, -1].float()
        lid = torch.tensor(letter_ids(tok, it["labels"]), device=dev)
        raw = W[lid].float() @ h                                   # uncapped letter logits (fp32)
        zl = capped(raw, cap)
        lse = torch.logsumexp(torch.stack([torch.logsumexp(capped(W[s:s + vocab_chunk].float() @ h, cap), 0)
                                           for s in range(0, W.shape[0], vocab_chunk)]), 0)
        logp = torch.log_softmax(zl, 0)
        g = it["labels"].index(it["gold_label"])
        res["correct"].append(float(int(zl.argmax()) == g))
        res["p_gold"].append(float(logp[g].exp()))
        res["nll"].append(float(-logp[g]))
        res["mass"].append(float(torch.exp(torch.logsumexp(zl, 0) - lse)))
        res["raw"].extend(raw.tolist())
        res["chance"].append(1.0 / len(it["labels"]))
        if hf_check and n == 0:
            try:
                full = model(input_ids=x, logits_to_keep=1, use_cache=False).logits[0, -1].float()
            except TypeError:
                full = model(input_ids=x, use_cache=False).logits[0, -1].float()
            hf_diff = {"letters_maxabs": float((full[lid] - zl).abs().max()),
                       "lse_absdiff": float((torch.logsumexp(full, 0) - lse).abs())}
    m = lambda v: round(sum(v) / len(v), 4) if v else None
    raw = res["raw"]
    out = {"n": len(items), "acc": m(res["correct"]), "chance": m(res["chance"]), "p_gold": m(res["p_gold"]),
           "nll": m(res["nll"]), "letter_mass": m(res["mass"]),
           "letter_mass_min": round(min(res["mass"]), 4) if res["mass"] else None,
           "raw_letter_logit": {"mean": m(raw), "max": round(max(raw), 2) if raw else None,
                                "frac_gt20": m([float(r > 20) for r in raw])}}
    if hf_diff is not None:
        out["hf_check"] = hf_diff
    return out


def part_readout(model, tok, files: list[str], n_items: int, max_len: int) -> None:
    was = model.training
    model.eval()
    for f in files:
        p = Path(f)
        if not p.exists():
            emit({"part": "readout", "file": f, "skip": "missing"})
            continue
        items = []
        for line in open(p):
            if line.strip():
                it = json.loads(line)
                if it.get("gold_label") in it.get("labels", []):
                    items.append(it)
            if len(items) >= n_items:
                break
        t0 = time.time()
        row = {"part": "readout", "file": f, "groups": dict(collections.Counter(it.get("group") for it in items)),
               "bos": readout_items(model, tok, items, max_len, use_prefix=True, hf_check=True)}
        nob = readout_items(model, tok, items, max_len, use_prefix=False)
        row["no_bos"] = {k: nob[k] for k in ("acc", "p_gold", "letter_mass")}
        row["s"] = round(time.time() - t0, 1)
        emit(row)
    model.train(was)


# ---------------------------------------------------------------------------------------------- synthetic training
def parse_setting(s: str) -> tuple[int, int, float | None]:
    """'2x1536' -> (2, 1536, None); '4x1024:6' -> (4, 1024, 6.0) (6 GiB gradient in-flight cap)."""
    s, _, inflight = s.partition(":")
    bs, L = s.lower().split("x")
    return int(bs), int(L), float(inflight) if inflight else None


def make_batch(bs: int, L: int, vocab: int, prefix: list[int], n_letters: int, gen: torch.Generator, dev,
               short: int = 7) -> dict:
    """Random-token sequences of exactly L tokens (prefix first), except the last one, which is `short` tokens
    shorter and right-padded, so a real attention mask exists (the masked sdpa path is what training uses)."""
    ids = torch.randint(3, vocab, (bs, L), generator=gen)
    if prefix:
        ids[:, :len(prefix)] = torch.tensor(prefix)
    mask = torch.ones(bs, L, dtype=torch.long)
    last = torch.full((bs,), L - 1)
    if bs > 1 and L > short + len(prefix) + 1:
        ids[-1, L - short:] = 0
        mask[-1, L - short:] = 0
        last[-1] = L - short - 1
    y = torch.randint(0, n_letters, (bs,), generator=gen)
    return {"input_ids": ids.to(dev), "attention_mask": mask.to(dev), "last": last.to(dev), "y": y.to(dev),
            "tokens": int(mask.sum())}


def letter_loss(model, batch: dict, lids: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """Sum over the micro-batch of label-smoothed CE on the soft-capped letter logits at each sequence's last token
    (full-vocabulary logits are never formed)."""
    body, head = text_parts(model)
    h = body_hidden(body, batch["input_ids"], batch["attention_mask"])
    hl = h[torch.arange(h.shape[0], device=h.device), batch["last"]]
    z = capped(hl.float() @ head.weight[lids].float().T, softcap_of(model))
    return F.cross_entropy(z, batch["y"], label_smoothing=smoothing, reduction="sum")


def part_bench(dev, threads: int) -> None:
    row = {"part": "bench", "threads": threads}
    if dev.type == "cuda":
        n = GIB // 2
        g = torch.empty(n, dtype=torch.bfloat16, device=dev)
        h = torch.empty(n, dtype=torch.bfloat16, pin_memory=True)
        for name, (dst, src) in {"d2h_gbs": (h, g), "h2d_gbs": (g, h)}.items():
            ts = []
            for _ in range(4):
                sync(dev)
                t0 = time.time()
                dst.copy_(src, non_blocking=True)
                sync(dev)
                ts.append(time.time() - t0)
            row[name] = round(GIB / 1e9 / sorted(ts[1:])[1], 2)
        del g, h
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        n = 2 ** 27
        p = torch.zeros(n, requires_grad=True)
        p.grad = torch.randn(n) * 1e-3
        acc, st = torch.empty(n), torch.randn(n).to(torch.bfloat16)
        for impl in ("fused", "foreach"):
            try:
                opt = torch.optim.AdamW([p], lr=1e-5, fused=impl == "fused", foreach=impl == "foreach")
                opt.step()
                t0 = time.time()
                for _ in range(3):
                    opt.step()
                row[f"adamw_{impl}_gparam_s"] = round(3 * n / 1e9 / (time.time() - t0), 3)
                del opt
            except (RuntimeError, ValueError, TypeError, NotImplementedError) as e:
                row[f"adamw_{impl}_gparam_s"] = f"unavailable: {e!r}"[:120]
        acc.copy_(st)
        t0 = time.time()
        for _ in range(3):
            acc.add_(st)
        row["acc_add_bf16_gparam_s"] = round(3 * n / 1e9 / (time.time() - t0), 3)
        # thread sweep of the fused step (memory-bandwidth bound): how many CPUs a 12B job really needs to request
        if isinstance(row.get("adamw_fused_gparam_s"), float):
            sweep = {}
            for t in sorted({2, 4, 6, threads}):
                if t > threads:
                    continue
                torch.set_num_threads(t)
                opt = torch.optim.AdamW([p], lr=1e-5, fused=True)
                opt.step()
                t0 = time.time()
                for _ in range(2):
                    opt.step()
                sweep[str(t)] = round(2 * n / 1e9 / (time.time() - t0), 3)
                del opt
            row["adamw_fused_gparam_s_by_threads"] = sweep
        del p, acc, st
    finally:
        torch.set_num_threads(old)
    emit(row)


def part_baseline(model, settings, args, lids, prefix, vocab, dev) -> dict:
    """Forward+backward per micro-batch with every gradient freed by a hook as soon as it is accumulated."""
    params = [p for p in model.parameters() if p.requires_grad]
    hooks = [p.register_post_accumulate_grad_hook(lambda p: setattr(p, "grad", None)) for p in params]
    out = {}
    try:
        for s in settings:
            bs, L, _ = parse_setting(s)
            gen = torch.Generator().manual_seed(1)
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            ts, tok = [], 0
            try:
                for k in range(1 + args.baseline_micro):
                    b = make_batch(bs, L, vocab, prefix, len(lids), gen, dev)
                    sync(dev)
                    t0 = time.time()
                    letter_loss(model, b, lids).div(bs * args.ga).backward()
                    sync(dev)
                    if k:
                        ts.append(time.time() - t0)
                        tok += b["tokens"]
                row = {"part": "baseline", "setting": s, "status": "ok", "micro": len(ts),
                       "s_per_micro": round(sum(ts) / len(ts), 3), "tok_s": round(tok / sum(ts), 1), **gpu_mem(dev)}
                out[s] = row
            except torch.cuda.OutOfMemoryError as e:
                row = {"part": "baseline", "setting": s, "status": "OOM", "error": str(e)[:300], **gpu_mem(dev)}
            emit(row)
            for p in params:
                p.grad = None
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()
    return out


def run_update(model, opt, sched, batches_fn, ga: int, dev, clip: float = 1.0) -> dict:
    """One optimizer update of ga micro-batches; times forward+backward (incl. the overlapped gradient D2H and the
    hooks' throttling), drain + clip (pending CPU adds + fp32 norm), CPU AdamW + cast + H2D."""
    sync(dev)
    t0 = time.time()
    loss, tok = torch.zeros((), device=dev), 0
    for k in range(ga):
        b = batches_fn(k)
        l = letter_loss(model, b, b["lids"]) / (b["input_ids"].shape[0] * ga)
        l.backward()
        loss += l.detach()                 # no host sync between micro-batches
        tok += b["tokens"]
    sync(dev)
    t1 = time.time()
    norm = float(opt.clip_grad_norm_(clip))
    t2 = time.time()
    opt.step()
    if sched is not None:
        sched.step()
    sync(dev)
    t3 = time.time()
    opt.zero_grad()
    return {"loss": float(loss), "grad_norm": norm, "tokens": tok, "fb_s": t1 - t0, "clip_s": t2 - t1, "step_s": t3 - t2,
            "total_s": t3 - t0}


def part_mem(model, opt, sched, settings, args, lids, prefix, vocab, dev, base: dict, deadline: float) -> None:
    for s in settings:
        bs, L, inflight = parse_setting(s)
        opt.max_inflight_bytes = int((inflight if inflight is not None else args.inflight_gib) * GIB)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
        gen = torch.Generator().manual_seed(2)
        recs, status, err = [], "ok", None
        try:
            for step in range(args.steps):
                if recs and time.time() + recs[-1]["total_s"] > deadline:
                    status = "truncated (time budget)"
                    break
                r = run_update(model, opt, sched, lambda k: dict(make_batch(bs, L, vocab, prefix, len(lids), gen, dev),
                                                                  lids=lids), args.ga, dev)
                r.update(host_mem())
                recs.append(r)
                log(f"{s} update {step}: " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                                        for k, v in r.items()))
        except torch.cuda.OutOfMemoryError as e:
            status, err = "OOM", str(e)[:300]
            opt.zero_grad()
            for p in model.parameters():
                p.grad = None
        row = {"part": "mem", "setting": s, "status": status, "bs": bs, "max_len": L, "ga": args.ga,
               "inflight_gib": opt.max_inflight_bytes / GIB, "impl": opt.impl, "threads": opt.num_threads,
               "updates": len(recs), **gpu_mem(dev), **host_mem()}
        if err:
            row["error"] = err
        if recs:
            row["update0"] = {k: round(recs[0][k], 2) for k in ("fb_s", "clip_s", "step_s", "total_s")}
            rest = recs[1:] or recs
            mean = lambda k: sum(r[k] for r in rest) / len(rest)
            tot = mean("total_s")
            row.update({"tokens_per_update": recs[0]["tokens"], "s_per_update": round(tot, 2),
                        "fb_s": round(mean("fb_s"), 2), "clip_s": round(mean("clip_s"), 2),
                        "step_s": round(mean("step_s"), 2), "tok_s": round(mean("tokens") / tot, 1),
                        "loss": [round(r["loss"], 4) for r in recs],
                        "grad_norm": [round(r["grad_norm"], 3) for r in recs],
                        "host_rss_peak_gib": max(r["rss_peak_gib"] for r in recs)})
            nbytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
            row["d2h_gbs_lower_bound"] = round(args.ga * nbytes / 1e9 / mean("fb_s"), 2)
            if s in base:
                floor = base[s]["s_per_micro"] * args.ga
                row["fb_gpu_only_s"] = round(floor, 2)
                row["fb_over_gpu_only"] = round(mean("fb_s") / floor, 3)
                row["update_over_gpu_only"] = round(tot / floor, 3)
        emit(row)


# ---------------------------------------------------------------------------------------------- resume round trip
_DIGEST_W: dict = {}


@torch.no_grad()
def bit_digest(bits: torch.Tensor, chunk: int = 2 ** 22) -> tuple[int, int]:
    """(sum, position-weighted sum) of an int16/int32 bit-pattern view, chunk by chunk. Exact integer arithmetic
    (|bits| < 2^31, weights <= 251, chunk 2^22: every chunk sum < 2^62, combined in Python ints), so the result does not
    depend on the reduction order or thread count. Chunked because t.sum(dtype=torch.int64) on a CPU int32 tensor
    first casts the whole tensor to int64 (2x its size transient: 8 GB for the 12B embedding master). The weighted
    sum also catches swapped / permuted elements and opposite flips, which a plain sum of bit patterns misses."""
    key = (bits.device, chunk)
    w = _DIGEST_W.get(key)
    if w is None:
        w = _DIGEST_W[key] = torch.arange(chunk, device=bits.device, dtype=torch.int64) % 251 + 1
    s0 = s1 = 0
    for c, s in enumerate(range(0, bits.numel(), chunk)):
        b = bits[s:s + chunk].to(torch.int64)
        cs = int(b.sum())
        s0 += cs
        s1 += (c + 1) * cs + int((b * w[:b.numel()]).sum())
    return s0, s1


def digest_cpu(sd: dict) -> list[tuple]:
    """Bit-level digest per parameter of masters and moments: (step, bit_digest of each fp32 tensor)."""
    out = []
    for i in sorted(sd["state"]):
        e = sd["state"][i]
        row = [e.get("step")]
        for k in ("master", "exp_avg", "exp_avg_sq"):
            t = e.get(k)
            row.append(None if t is None else bit_digest(t.reshape(-1).view(torch.int32)))
        out.append(tuple(row))
    return out


@torch.no_grad()
def digest_gpu(params) -> list[tuple[int, int]]:
    out = []
    for p in params:
        out.append(bit_digest(p.detach().reshape(-1).view(torch.int16 if p.element_size() == 2 else torch.int32)))
    return out


@torch.no_grad()
def weights_match_masters(opt) -> bool:
    ok = True
    for p, m in zip(opt._params, opt.master):
        ok &= bool(torch.equal(p, m.to(p.dtype).to(p.device)))
    return ok


def fixed_grad_update(opt, sched, seed: int, ga: int = 2, scale: float = 1e-3) -> float:
    """One update whose gradients are fixed seeded tensors fed through the real hooks: (p * c).sum().backward()
    gives p.grad == c exactly (no reduction), so the update is deterministic."""
    for k in range(ga):
        for i, p in enumerate(opt._params):
            g = torch.Generator(device=p.device).manual_seed(seed * 1_000_003 + k * 10_007 + i)
            c = torch.randn(p.shape, generator=g, device=p.device, dtype=p.dtype).mul_(scale)
            (p * c).sum().backward()
            del c
    norm = float(opt.clip_grad_norm_(1.0))
    opt.step()
    if sched is not None:
        sched.step()
    sync(opt._params[0].device)
    opt.zero_grad()
    return norm


def part_roundtrip(model, opt, sched, args, lids, prefix, vocab, dev) -> None:
    """--rt-moments fp32: the exact resume (masters + m + v, 12 B/param = 143 GB at 12B). none: the quota-sized
    approximate resume (masters only, 4 B/param = 47.6 GB): the masters and weights must come back bit for bit and
    AdamW restarts (step 1 after the resumed update); the straight/resumed update equality is not expected there.
    Free space is min(df, --rt-free-gb): df cannot see a user quota, and running into the quota mid-write would also
    break every other job of the account that writes to the same filesystem."""
    rt_dir = Path(args.rt_dir)
    rt_dir.mkdir(parents=True, exist_ok=True)
    moments = getattr(args, "rt_moments", "fp32") != "none"
    need = sum(m.numel() * (12 if moments else 4) for m in opt.master) * 1.05 + 2 ** 26
    free = shutil.disk_usage(rt_dir).free
    cap = getattr(args, "rt_free_gb", None)
    if cap is not None:
        free = min(free, cap * 1e9)
    if free < need:
        emit({"part": "roundtrip", "skip": f"free {free / 1e9:.1f} GB (min of df and --rt-free-gb) < "
                                           f"{need / 1e9:.1f} GB in {rt_dir}", "moments": moments})
        return
    path = rt_dir / f"mem12b_optim_{os.getpid()}.pt"
    row = {"part": "roundtrip", "rt_dir": str(rt_dir), "moments": moments, "state_gb": round(need / 1.05 / 1e9, 1)}
    gen = torch.Generator().manual_seed(3)
    bs, L = 2, min(512, args.max_len)
    try:
        for _ in range(2):        # two real updates, so the saved state has non-trivial moments and step = 2+
            run_update(model, opt, sched, lambda k: dict(make_batch(bs, L, vocab, prefix, len(lids), gen, dev),
                                                          lids=lids), 2, dev)
        t0 = time.time()
        torch.save({"optim": opt.state_dict(moments=moments), "sched": sched.state_dict()}, path)
        row["save_s"] = round(time.time() - t0, 1)
        row["file_gb"] = round(path.stat().st_size / 1e9, 1)
        d0, g0 = digest_cpu(opt.state_dict(moments=moments)), digest_gpu(opt._params)
        lr0 = [g["lr"] for g in opt.param_groups]
        row["norm_straight"] = fixed_grad_update(opt, sched, seed=7)
        da, ga_ = digest_cpu(opt.state_dict()), digest_gpu(opt._params)
        row["moved"] = sum(x != y for x, y in zip(g0, ga_))
        rss0 = host_mem()
        t0 = time.time()
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        opt.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        row["load_s"] = round(time.time() - t0, 1)
        row["rss_before_load_gib"], row["rss_after_load_gib"] = rss0["rss_gib"], host_mem()["rss_gib"]
        del ck
        row["load_exact"] = digest_cpu(opt.state_dict(moments=moments)) == d0
        row["weights_restored"] = digest_gpu(opt._params) == g0
        row["weights_eq_masters_bf16"] = weights_match_masters(opt)
        row["lr_restored"] = [g["lr"] for g in opt.param_groups] == lr0
        row["norm_resumed"] = fixed_grad_update(opt, sched, seed=7)
        sd_b = opt.state_dict()
        db, gb_ = digest_cpu(sd_b), digest_gpu(opt._params)
        ok = (row["load_exact"] and row["weights_restored"] and row["weights_eq_masters_bf16"] and row["lr_restored"]
              and row["moved"] > 0)
        if moments:
            row["state_equal_after_step"] = db == da
            row["weights_equal_after_step"] = gb_ == ga_
            row["n_params_weights_differ"] = sum(x != y for x, y in zip(ga_, gb_))
            ok = ok and row["state_equal_after_step"] and row["weights_equal_after_step"]
        else:                     # masters-only resume: AdamW restarted, so one update later every step count is 1
            row["adam_restarted"] = all(e.get("step") == 1.0 for e in sd_b["state"].values())
            row["n_params_weights_differ"] = sum(x != y for x, y in zip(ga_, gb_))
            ok = ok and row["adam_restarted"]
        row["ok"] = bool(ok)
    except torch.cuda.OutOfMemoryError as e:
        row.update(ok=False, status="OOM", error=str(e)[:300])
    except OSError as e:           # quota / disk full mid-write: a storage limit, not a code failure
        row.update(skip=f"OSError during save/load: {e}"[:300])
    finally:
        path.unlink(missing_ok=True)
    row.update(host_mem())
    emit(row)
    if not row.get("ok") and "skip" not in row:
        raise SystemExit("roundtrip failed")


# ---------------------------------------------------------------------------------------------- check (login node)
def part_check(args) -> None:
    import transformers
    cfg = transformers.AutoConfig.from_pretrained(args.model)
    tc = text_config(cfg)
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    lt = getattr(tc, "layer_types", None) or []
    st = sorted(Path(args.model).glob("*.safetensors"))
    row = {"part": "check", "model": args.model, "model_type": cfg.model_type,
           "architectures": getattr(cfg, "architectures", None), "layers": tc.num_hidden_layers,
           "layer_types": dict(collections.Counter(lt)), "hidden": tc.hidden_size, "vocab": tc.vocab_size,
           "softcap": getattr(tc, "final_logit_softcapping", None), "tied": getattr(cfg, "tie_word_embeddings", None),
           "bos_prefix": special_prefix(tok), "letters_A_F": letter_ids(tok, list("ABCDEF")),
           "safetensors_gb": round(sum(p.stat().st_size for p in st) / 1e9, 2), "n_safetensors": len(st),
           "eval_files": {f: (sum(1 for _ in open(f)) if Path(f).exists() else "missing") for f in args.readout_files}}
    rt = Path(args.rt_dir)
    probe = rt if rt.exists() else next((q for q in rt.parents if q.exists()), Path("/"))
    row["rt_dir_free_gb"] = round(shutil.disk_usage(probe).free / 1e9, 1)
    emit(row)


# ---------------------------------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--parts", default="bench,readout,baseline,mem,roundtrip")
    ap.add_argument("--tiny", action="store_true", help="2-layer random model from the real config (in-job smoke)")
    ap.add_argument("--settings", default="2x1536,4x1024", help="comma-separated BSxL[:inflightGiB]")
    ap.add_argument("--baseline-extra", default="",
                    help="extra BSxL settings for the baseline part only (GPU-only peak memory / tok/s, no updates)")
    ap.add_argument("--ga", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--baseline-micro", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0) or None)
    ap.add_argument("--inflight-gib", type=float, default=2.0)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--attn", default=None, help="attn_implementation (default: the model's, sdpa)")
    ap.add_argument("--readout-files", default="data/eval/kk_heldout.jsonl,data/eval/bbh.jsonl,"
                                               "data/eval/jevbench_public.jsonl")
    ap.add_argument("--readout-n", type=int, default=50)
    ap.add_argument("--rt-dir", default=".")
    ap.add_argument("--rt-free-gb", type=float, default=None,
                    help="free space the roundtrip may use (GB): the lquota headroom from the login node when --rt-dir "
                         "is on the quota'd home filesystem (df cannot see the quota); 0 skips the roundtrip")
    ap.add_argument("--rt-moments", choices=["fp32", "none"], default="fp32",
                    help="fp32: exact resume state (143 GB at 12B); none: masters only (47.6 GB), AdamW restarts")
    ap.add_argument("--budget-min", type=float, default=150.0,
                    help="mem updates stop when the next one would pass this (minutes since start) minus --rt-reserve")
    ap.add_argument("--rt-reserve-min", type=float, default=40.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    args.readout_files = [f for f in args.readout_files.split(",") if f]
    parts = [p for p in args.parts.split(",") if p]
    settings = [s for s in args.settings.split(",") if s]
    extra = [s for s in args.baseline_extra.split(",") if s and s not in settings]
    for s in settings + extra:
        parse_setting(s)
    if "check" in parts:
        part_check(args)
        parts.remove("check")
        if not parts:
            return
    import transformers
    dev = torch.device(args.device)
    threads = args.threads or torch.get_num_threads()
    torch.set_num_threads(threads)
    env = {"part": "env", "torch": torch.__version__, "transformers": transformers.__version__, "threads": threads,
           "tiny": args.tiny, "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"}
    if dev.type == "cuda":
        env["free_total_gib"] = [round(x / GIB, 2) for x in torch.cuda.mem_get_info(dev)]
    emit(env)
    if "bench" in parts:
        part_bench(dev, threads)
    if not set(parts) & {"readout", "baseline", "mem", "roundtrip"}:
        return

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    if args.tiny:
        model = build_tiny(tiny_config(args.model), torch.bfloat16)
        info = {}
    else:
        model, info = load_text_only(args.model, torch.bfloat16, args.attn)
    load_s = time.time() - t0
    model.to(dev)
    model.config.use_cache = False
    text_config(model.config).use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()                       # HF checkpointing is active in training mode only
    params = [p for p in model.parameters() if p.requires_grad]
    vocab = text_config(model.config).vocab_size
    prefix = special_prefix(tok)
    lids = torch.tensor(letter_ids(tok, list("ABCDEF")), device=dev)
    emit({"part": "load", "class": type(model).__name__, "params_b": round(sum(p.numel() for p in params) / 1e9, 4),
          "n_tensors": len(params), "load_s": round(load_s, 1), "softcap": softcap_of(model), "bos_prefix": prefix,
          "attn": getattr(text_config(model.config), "_attn_implementation", None),
          "unexpected_keys": len(info.get("unexpected_keys", []) or []), **gpu_mem(dev), **host_mem()})

    if "readout" in parts:
        part_readout(model, tok, args.readout_files, args.readout_n, args.max_len)
    base = part_baseline(model, settings + extra, args, lids, prefix, vocab, dev) if "baseline" in parts else {}
    if not set(parts) & {"mem", "roundtrip"}:
        return

    from cotdistill.offload import CPUOffloadAdamW
    t0 = time.time()
    opt = CPUOffloadAdamW(params, lr=args.lr, weight_decay=0.0, num_threads=threads,
                          max_inflight_bytes=int(args.inflight_gib * GIB))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    emit({"part": "optimizer", "impl": opt.impl, "threads": opt.num_threads, "init_s": round(time.time() - t0, 1),
          "host_state_gib": round(opt.host_bytes() / GIB, 2), "n_buckets": len(opt._buckets), **host_mem()})
    deadline = T_START + 60 * (args.budget_min - (args.rt_reserve_min if "roundtrip" in parts else 0))
    if "mem" in parts:
        part_mem(model, opt, sched, settings, args, lids, prefix, vocab, dev, base, deadline)
    if "roundtrip" in parts:
        part_roundtrip(model, opt, sched, args, lids, prefix, vocab, dev)
    opt.release()
    emit({"part": "done", "elapsed_min": round((time.time() - T_START) / 60, 1), **host_mem(), **gpu_mem(dev)})


if __name__ == "__main__":
    main()
