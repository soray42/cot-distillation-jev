"""Interchange interventions at the readout position of a one-pass decision model (causal-audit tooling).

A site is (layer L, readout position): the residual stream entering decoder layer L at the last prompt token, which is
`hidden_states[L]` in Hugging Face's output (index 0 = embeddings), the same indexing as the hidden-state dumps. A
forward pre-hook on decoder layer L rewrites that one vector. Another token position can be used instead of the readout
(`pos` in site_states / patched_logits, a "base_pos" entry per pair in iia / fit_das: absolute token indices of the
collated, right-padded sequence); the letter logits are always read at the readout position:

  full    the base state is replaced by the source state at the site
  das     only a learned k-dimensional subspace is exchanged: h_b + (h_s - h_b) R^T R, where R (k x d) has orthonormal
          rows and is fitted by distributed alignment search (Geiger et al.) so that the patched base predicts the
          program's counterfactual target
  random  the same exchange in a random orthonormal subspace (baseline)
  noise   the base state plus norm-matched Gaussian noise inside the subspace (baseline)

Interchange-intervention accuracy (IIA) is the share of pairs whose patched base predicts the target outcome.
Model weights stay frozen throughout; only R is trained.
"""
from __future__ import annotations

import contextlib
import functools
import inspect
import math
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .student import FINAL_TEMPLATE, collate, letter_token_ids, text_parts, Example


_AMP = {"autocast": True}


def set_precision(mode: str) -> dict:
    """Forward precision of every pass in this module (and of kk_audit.py, which uses amp()):
      bf16           bf16 weights under bf16 autocast (audit v1 default)
      fp32-autocast  fp32 weights and residual stream, matmul inputs cast to bf16 by autocast (audit v1 --dtype fp32)
      fp32           fp32 weights, autocast off, TF32 off for matmuls and cuDNN (convolutions): every product in fp32
    Returns the settings for the run header."""
    assert mode in ("bf16", "fp32-autocast", "fp32"), mode
    _AMP["autocast"] = mode != "fp32"
    if mode == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    return {"precision": mode, "autocast_bf16": _AMP["autocast"],
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)}


def amp(dev):
    """The autocast context of the current precision (bf16 autocast on CUDA unless set_precision("fp32"))."""
    dev = torch.device(dev) if isinstance(dev, str) else dev
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda" and _AMP["autocast"])


_REF_KERNELS = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
                "causal_conv1d_update")


def use_reference_kernels(model) -> list[str]:
    """Route the linear-attention kernels that the model's decoder layers call (Qwen3.5 Gated DeltaNet: the chunked and
    recurrent delta rule, the causal conv1d and its single-token update) to the PyTorch reference implementations of
    their modeling file. Those compute in the input dtype (fp32 when the forward is fp32) and are differentiable in the
    single-token decode, where the causal-conv1d CUDA update kernel has no backward and the fla kernels are bf16-oriented.
    The modeling module's globals are replaced, so every model of that class in the process is affected. Returns the
    replaced names (empty for models without such layers, e.g. Qwen2)."""
    done = []
    for mod_name in sorted({type(m).__module__ for m in decoder_layers(model).modules()}):
        mod = sys.modules.get(mod_name)
        for name in _REF_KERNELS:
            f = getattr(mod, name, None)
            if f is None or not callable(f):
                continue
            if getattr(f, "_cotd_reference", False):
                done.append(f"{mod_name}.{name}")
                continue
            ref = inspect.unwrap(f)
            if ref.__module__ != mod_name or ref.__name__ != name:
                raise RuntimeError(f"{mod_name}.{name}: the unwrapped function is {ref.__module__}.{ref.__name__}, "
                                   "not the modeling file's reference implementation")
            sig = inspect.signature(ref).parameters
            var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.values())

            def call(*a, _ref=ref, _ok=frozenset(sig), _var=var_kw, **kw):      # the HF wrapper's kwarg filter
                return _ref(*a, **(kw if _var else {k: v for k, v in kw.items() if k in _ok}))
            call = functools.wraps(ref)(call)
            call._cotd_reference = True
            setattr(mod, name, call)
            done.append(f"{mod_name}.{name}")
    return done


def decoder_layers(model) -> nn.ModuleList:
    body, _ = text_parts(model)
    layers = getattr(body, "layers", None)
    if layers is None:
        raise ValueError("no decoder layers found on the text backbone")
    return layers


@contextlib.contextmanager
def patch_site(model, layer: int, last: torch.Tensor, fn):
    """Within the block, decoder layer `layer` receives hidden states whose readout rows (`last`, one position per
    batch row) are replaced by fn(h_last) ([B, d] -> [B, d])."""
    def pre(module, args, kwargs):
        if args:
            hs, rest = args[0], args[1:]
        else:
            hs, rest = kwargs["hidden_states"], ()
        rows = torch.arange(hs.shape[0], device=hs.device)
        new = fn(hs[rows, last]).to(hs.dtype)
        hs = hs.index_put((rows, last), new)
        if args:
            return (hs, *rest), kwargs
        kwargs = dict(kwargs, hidden_states=hs)
        return args, kwargs
    h = decoder_layers(model)[layer].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        yield
    finally:
        h.remove()


def _batch(tok, prompts: list[str], labels: list[list[str]], max_len: int, dev):
    exs = [Example(FINAL_TEMPLATE.format(problem=p), l, [0.0] * len(l), 1.0, "final", "") for p, l in zip(prompts, labels)]
    return collate(tok, exs, max_len, dev)


def _letter_logits(model, tok, b: dict, labels: list[list[str]], cache: dict) -> list[torch.Tensor]:
    body, head = text_parts(model)
    res = body(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
    h = res.last_hidden_state[torch.arange(len(labels), device=head.weight.device), b["last"]]
    out = []
    for i, l in enumerate(labels):
        key = tuple(l)
        if key not in cache:
            cache[key] = torch.tensor(letter_token_ids(tok, l), device=head.weight.device)
        out.append((h[i].to(head.weight.dtype) @ head.weight[cache[key]].T).float())
    return out


@torch.no_grad()
def site_states(model, tok, prompts: list[str], labels: list[list[str]], layers: list[int], max_len: int = 1536,
                bs: int = 8, pos: list[int] | None = None) -> torch.Tensor:
    """[n, len(layers), d] readout-position states entering each layer (float32, CPU), in input order. With `pos`
    (one token index per prompt) the states at those positions instead."""
    body, head = text_parts(model)
    dev = head.weight.device
    out = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        b = _batch(tok, [prompts[i] for i in idx], [labels[i] for i in idx], max_len, dev)
        with amp(dev):
            res = body(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=True)
        rows = torch.arange(len(idx), device=dev)
        at = b["last"] if pos is None else _at(b, [pos[i] for i in idx])
        st = torch.stack([res.hidden_states[l][rows, at] for l in layers], 1).float().cpu()
        for j, i in enumerate(idx):
            out[i] = st[j]
    return torch.stack(out)


class Subspace(nn.Module):
    """k orthonormal directions in R^d (rows of R)."""

    def __init__(self, d: int, k: int = 1, init: torch.Tensor | None = None):
        super().__init__()
        lin = nn.Linear(d, k, bias=False)
        if init is not None:
            with torch.no_grad():
                lin.weight.copy_(init)
        self.proj = nn.utils.parametrizations.orthogonal(lin)

    @property
    def R(self) -> torch.Tensor:
        return self.proj.weight                          # [k, d], orthonormal rows

    def exchange(self, hb: torch.Tensor, hs: torch.Tensor) -> torch.Tensor:
        R = self.R.to(hb.dtype)
        return hb + ((hs - hb) @ R.T) @ R


def random_subspace(d: int, k: int, seed: int) -> Subspace:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, k, generator=g))
    return Subspace(d, k, init=q.T.contiguous())


def patched_logits(model, tok, base_prompts, labels, layer: int, fn, max_len: int, cache: dict,
                   pos: list[int] | None = None) -> list[torch.Tensor]:
    dev = text_parts(model)[1].weight.device
    b = _batch(tok, base_prompts, labels, max_len, dev)
    at = b["last"] if pos is None else _at(b, pos)
    with patch_site(model, layer, at, fn):
        with amp(dev):
            return _letter_logits(model, tok, b, labels, cache)


def _at(b: dict, pos: list[int]) -> torch.Tensor:
    """Token positions as a tensor, checked to be real (unpadded) tokens at or before each row's readout."""
    at = torch.tensor(pos, device=b["last"].device)
    if not bool(((at >= 0) & (at <= b["last"])).all()):
        raise ValueError(f"patch positions outside the prompt: {pos} (readout {b['last'].tolist()})")
    return at


def _pos(chunk: list[dict]) -> list[int] | None:
    return [p["base_pos"] for p in chunk] if "base_pos" in chunk[0] else None


def iia(model, tok, pairs: list[dict], src_states: torch.Tensor, layer: int, method: str, sub: Subspace | None = None,
        max_len: int = 1536, bs: int = 8, seed: int = 0, outcomes: tuple[str, ...] = (), return_preds: bool = False) -> dict:
    """Accuracy of the patched base on the target outcome. `pairs` entries carry base_prompt, labels (letters),
    target (index), base_pred and source_pred (indices, the unpatched model's own answers; -1 = none), optionally
    base_pos (the patched token position; default the readout); `src_states[i]` is pair i's source state at `layer`
    ([d]). Also reports how often the patched answer copies the base or source answer, the share landing on each
    index field named in `outcomes` ("to_<field>"; None never matches) and, with return_preds, the patched answers."""
    cache: dict = {}
    hits = copies_b = copies_s = 0
    to = {k: 0 for k in outcomes}
    preds: list[int] = []
    g = torch.Generator().manual_seed(seed)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(pairs), bs):
            chunk = pairs[s:s + bs]
            hs = src_states[s:s + bs]

            def fn(hb, hs=hs):
                hs_ = hs.to(hb.device, hb.dtype)
                if method == "full":
                    return hs_
                if method == "noise":
                    R = sub.R.to(hb.dtype)
                    z = torch.randn(hb.shape, generator=g).to(hb.device, hb.dtype) @ R.T
                    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6) * ((hs_ - hb) @ R.T).norm(dim=-1, keepdim=True)
                    return hb + z @ R
                return sub.exchange(hb, hs_)
            zs = patched_logits(model, tok, [p["base_prompt"] for p in chunk], [p["labels"] for p in chunk], layer, fn,
                                max_len, cache, pos=_pos(chunk))
            for p, z in zip(chunk, zs):
                k = int(z.argmax())
                hits += k == p["target"]
                copies_b += k == p["base_pred"]
                copies_s += k == p["source_pred"]
                for f in outcomes:
                    to[f] += p.get(f) is not None and k == p[f]
                preds.append(k)
    n = max(1, len(pairs))
    out = {"iia": hits / n, "copy_base": copies_b / n, "copy_source": copies_s / n, "n": len(pairs)}
    out.update({f"to_{f}": v / n for f, v in to.items()})
    if return_preds:
        out["preds"] = preds
    return out


def fit_das(model, tok, pairs: list[dict], src_states: torch.Tensor, layer: int, k: int = 1, epochs: int = 3,
            lr: float = 1e-3, bs: int = 8, max_len: int = 1536, seed: int = 0, log=print) -> Subspace:
    """Fit a k-dimensional exchange subspace at `layer` by cross-entropy of the patched base on the target."""
    torch.manual_seed(seed)
    dev = text_parts(model)[1].weight.device
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sub = Subspace(src_states.shape[-1], k).to(dev)
    opt = torch.optim.Adam(sub.parameters(), lr=lr)
    cache: dict = {}
    order = list(range(len(pairs)))
    rng = random.Random(seed)
    for ep in range(epochs):
        rng.shuffle(order)
        tot = 0.0
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            chunk = [pairs[i] for i in idx]
            hs = src_states[idx].to(dev)

            def fn(hb, hs=hs):
                return sub.exchange(hb.float(), hs.float())
            zs = patched_logits(model, tok, [p["base_prompt"] for p in chunk], [p["labels"] for p in chunk], layer, fn,
                                max_len, cache, pos=_pos(chunk))
            loss = sum(F.cross_entropy(z[None], torch.tensor([p["target"]], device=z.device)) for z, p in zip(zs, chunk))
            opt.zero_grad()
            (loss / len(chunk)).backward()
            opt.step()
            tot += float(loss)
        log(f"das layer {layer} k {k} epoch {ep + 1}: loss {tot / len(order):.4f}")
    return sub


# ---------------------------------------------------------------------------------------------------------------------
# Readout-site interventions as a single-token decode over a cached prefix (audit v2).
#
# A patch at the readout (last prompt token) changes nothing at earlier positions, so the patched answer depends only on
# (a) every layer's state of the prompt minus its last token, which the patch cannot reach, and (b) the last token's
# state entering layer L. PrefixStore runs the prefix once per base prompt (batch 1, no padding) and keeps, for decoder
# layers >= its lowest layer, the full-attention keys/values and the linear-attention conv and recurrent states;
# decode_logits then runs layers L..N-1 on the one (patched) token. This is the same computation as the hook-patched full
# forward (tests/test_kk_audit.py checks logits and DAS gradients on tiny Qwen2 and Qwen3.5 models; kk_audit.py --check
# checks it on the real model), at a small fraction of its cost, so DAS can be fitted for hundreds of steps per subspace.


class _LinearState:
    """What a Gated DeltaNet layer reads from cache_params in a single-token decode (Qwen3.5 modeling)."""

    def __init__(self, conv: torch.Tensor, rec: torch.Tensor):
        self.conv_states = {0: conv}
        self.recurrent_states = {0: rec}
        self.record_past = False


class DecodeCache:
    """Duck-typed cache for one decode step: full-attention layers get the stored prefix keys/values (left padded to a
    common length) prepended to the new token's; linear-attention layers read fresh copies of their states (the layer
    updates them in place, so the stored prefix is never modified)."""

    def __init__(self, kv: dict, lin: dict):
        self.kv = kv
        self.layers = {i: _LinearState(c, r) for i, (c, r) in lin.items()}

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        K, V = self.kv[layer_idx]
        return torch.cat([K, key_states.to(K.dtype)], -2), torch.cat([V, value_states.to(V.dtype)], -2)

    def has_previous_state(self, layer_idx, state_idx: int = 0) -> bool:
        return layer_idx in self.layers

    def update_recurrent_state(self, recurrent_states, layer_idx, state_idx: int = 0, **kwargs):
        return recurrent_states


def text_config(model):
    """The decoder's own config (the inner text config of a multimodal wrapper)."""
    return text_parts(model)[0].config


def _layer_types(model) -> list[str]:
    cfg = text_config(model)
    n = len(decoder_layers(model))
    return list(getattr(cfg, "layer_types", None) or ["full_attention"] * n)


def prompt_ids(tok, prompt: str, max_len: int) -> list[int]:
    """Token ids of a prompt exactly as collate builds them (FINAL_TEMPLATE, tail cut, special prefix)."""
    from .student import encode_tail
    return encode_tail(tok, FINAL_TEMPLATE.format(problem=prompt), max_len)


class PrefixStore:
    """Per-prompt prefix state of decoder layers >= `lo`, keyed by a caller key (pair id), with an LRU memory cap.
    `device` holds the stored tensors (the GPU when they fit, else CPU); decode steps move what they need."""

    def __init__(self, model, tok, lo: int, max_len: int, device, cap_bytes: float):
        self.model, self.tok, self.lo, self.max_len = model, tok, lo, max_len
        self.device = torch.device(device)
        self.cap = cap_bytes
        self.items: dict = {}
        self.order: list = []
        self.bytes = 0
        self.n_built = 0
        self.types = _layer_types(model)

    def _build_one(self, prompt: str) -> dict:
        from transformers import DynamicCache
        body, head = text_parts(self.model)
        dev = head.weight.device
        ids = prompt_ids(self.tok, prompt, self.max_len)
        cache = DynamicCache(config=body.config)
        with torch.no_grad(), amp(dev):
            body(input_ids=torch.tensor([ids[:-1]], device=dev), past_key_values=cache, use_cache=True)
        layers, nb = {}, 0
        for i in range(self.lo, len(self.types)):
            lay = cache.layers[i]
            if self.types[i] == "linear_attention":
                t = (lay.conv_states[0].detach().to(self.device), lay.recurrent_states[0].detach().to(self.device))
                layers[i] = ("lin",) + t
            else:
                t = (lay.keys.detach().to(self.device), lay.values.detach().to(self.device))
                layers[i] = ("kv",) + t
            nb += sum(x.numel() * x.element_size() for x in t)
        return {"layers": layers, "pos": len(ids) - 1, "last_id": ids[-1], "bytes": nb}

    def ensure(self, keys: list, prompts: list[str]) -> None:
        """Build the missing entries; evict least recently used entries outside `keys` above the memory cap."""
        need = set(keys)
        for key, pr in zip(keys, prompts):
            if key in self.items:
                self.order.remove(key)
            else:
                self.items[key] = self._build_one(pr)
                self.bytes += self.items[key]["bytes"]
                self.n_built += 1
            self.order.append(key)
            while self.bytes > self.cap:
                old = next((k for k in self.order if k not in need), None)
                if old is None:
                    break
                self.order.remove(old)
                self.bytes -= self.items.pop(old)["bytes"]

    def bytes_per_prompt(self) -> float:
        return self.bytes / max(1, len(self.items))


def decode_logits(model, tok, store: PrefixStore, keys: list, h: torch.Tensor, layer: int, labels: list[list[str]],
                  cache: dict) -> list[torch.Tensor]:
    """Letter logits (float32) of each prompt in `keys` when its last token enters decoder layer `layer` with state h[i]
    ([B, d], may require grad), over the stored prefix (store.lo <= layer)."""
    assert layer >= store.lo, (layer, store.lo)
    body, head = text_parts(model)
    dev = head.weight.device
    items = [store.items[k] for k in keys]
    B = len(items)
    types = store.types
    n = len(types)
    kv, lin = {}, {}
    full = [i for i in range(layer, n) if types[i] != "linear_attention"]
    T = max((it["layers"][full[0]][1].shape[-2] for it in items), default=0) if full else 0
    for i in range(layer, n):
        if types[i] == "linear_attention":
            lin[i] = (torch.cat([it["layers"][i][1] for it in items]).to(dev),
                      torch.cat([it["layers"][i][2] for it in items]).to(dev))
        else:
            ks, vs = [], []
            for it in items:
                K, V = it["layers"][i][1], it["layers"][i][2]
                pad = T - K.shape[-2]
                ks.append(F.pad(K, (0, 0, pad, 0)) if pad else K)
                vs.append(F.pad(V, (0, 0, pad, 0)) if pad else V)
            kv[i] = (torch.cat(ks).to(dev), torch.cat(vs).to(dev))
    plen = torch.tensor([it["pos"] for it in items], device=dev)
    mask = None
    if full:
        valid = torch.arange(T + 1, device=dev)[None, :] >= (T - plen)[:, None]       # left-padded prefix + new token
        impl = getattr(body.config, "_attn_implementation", "sdpa") or "sdpa"
        if impl == "sdpa":
            mask = valid[:, None, None, :]
        elif impl == "eager":
            mask = torch.zeros(B, 1, 1, T + 1, device=dev).masked_fill(~valid[:, None, None, :],
                                                                      torch.finfo(torch.float32).min)
        else:
            raise ValueError(f"decode path supports sdpa/eager attention, not {impl}")
    pos = plen[:, None]
    hs = h.to(dev, head.weight.dtype)[:, None, :]          # the residual stream's dtype, as patch_site casts it
    if mask is not None and mask.dtype != torch.bool:
        mask = mask.to(hs.dtype)
    rot = body.rotary_emb
    with amp(dev):
        pe = rot(hs, pos[None].expand(3, B, 1) if hasattr(rot, "mrope_section") else pos)
        dc = DecodeCache(kv, lin)
        lays = decoder_layers(model)
        for i in range(layer, n):
            o = lays[i](hs, position_embeddings=pe, attention_mask=None if types[i] == "linear_attention" else mask,
                        position_ids=pos, past_key_values=dc, use_cache=True)
            hs = o[0] if isinstance(o, tuple) else o
        hs = body.norm(hs)[:, 0]
        out = []
        for j, l in enumerate(labels):
            key = tuple(l)
            if key not in cache:
                cache[key] = torch.tensor(letter_token_ids(tok, l), device=head.weight.device)
            out.append((hs[j].to(head.weight.dtype) @ head.weight[cache[key]].T).float())
    return out


@torch.no_grad()
def readout_pass(model, tok, prompts: list[str], labels: list[list[str]], layers: list[int], max_len: int = 1536,
                 bs: int = 8) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Unpatched letter logits (float32 CPU) and readout states entering each layer in `layers` ([n, len(layers), d],
    float32 CPU), in input order, from one forward per prompt at the current precision."""
    body, head = text_parts(model)
    dev = head.weight.device
    logits: list = [None] * len(prompts)
    states: list = [None] * len(prompts)
    cache: dict = {}
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        b = _batch(tok, [prompts[i] for i in idx], [labels[i] for i in idx], max_len, dev)
        with amp(dev):
            res = body(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=True)
        rows = torch.arange(len(idx), device=dev)
        st = torch.stack([res.hidden_states[l][rows, b["last"]] for l in layers], 1).float().cpu()
        h = res.last_hidden_state[rows, b["last"]]
        for j, i in enumerate(idx):
            key = tuple(labels[i])
            if key not in cache:
                cache[key] = torch.tensor(letter_token_ids(tok, labels[i]), device=head.weight.device)
            logits[i] = (h[j].to(head.weight.dtype) @ head.weight[cache[key]].T).float().cpu()
            states[i] = st[j]
    return logits, torch.stack(states) if states else torch.zeros(0, len(layers), 0)


@torch.no_grad()
def token_states(model, tok, prompts: list[str], positions: list[list[int]], layers: list[int], max_len: int = 1536,
                 bs: int = 8) -> list[torch.Tensor]:
    """States entering each layer in `layers` at the given token positions of each prompt (collate indexing): one
    [len(positions[i]), len(layers), d] float32 CPU tensor per prompt."""
    body, head = text_parts(model)
    dev = head.weight.device
    out: list = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        b = _batch(tok, [prompts[i] for i in idx], [["A"]] * len(idx), max_len, dev)
        with amp(dev):
            res = body(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=True)
        for j, i in enumerate(idx):
            at = _at({"last": b["last"][j:j + 1].expand(len(positions[i]))}, positions[i])
            out[i] = torch.stack([res.hidden_states[l][j, at] for l in layers], 1).float().cpu()
    return out


def exchange_rows(hb: torch.Tensor, hs: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """hb + ((hs - hb) R_i^T) R_i per row: hb, hs [B, d]; R [B, k, d] (each row's own subspace)."""
    c = torch.einsum("bkd,bd->bk", R, hs - hb)
    return hb + torch.einsum("bk,bkd->bd", c, R)


def noise_rows(hb: torch.Tensor, hs: torch.Tensor, R: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """hb plus Gaussian noise inside each row's subspace, scaled to the norm of the DAS exchange's change."""
    z = torch.einsum("bkd,bd->bk", R, torch.randn(hb.shape, generator=g).to(hb.device, hb.dtype))
    want = torch.einsum("bkd,bd->bk", R, hs - hb).norm(dim=-1, keepdim=True)
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6) * want
    return hb + torch.einsum("bk,bkd->bd", z, R)


def fit_slot_subspaces(fwd, n_dim: int, k: int, slot_of: list, val_slot_of: list, targets: list[int],
                       val_targets: list[int], n_options: list[int], steps_min: int = 500, steps_max: int = 1500,
                       patience: int = 200, eval_every: int = 25, rows_per_slot: int = 4, lr: float = 1e-2,
                       seed: int = 0, dev="cpu", log=None, fwd_chunk: int = 0) -> tuple[dict, dict]:
    """Fit one rank-k exchange subspace per slot, all slots in one loop (each step takes `rows_per_slot` training rows of
    every slot, cycling through each slot's rows in a fresh random order per pass, so a slot with few rows still gets
    one update per step). fwd(rows, train: bool, R: [len(rows), k, d]) -> letter logits of those rows (train rows or
    validation rows) patched with each row's subspace. Adam (lr), cross-entropy on the targets. The validation loss
    (held-out programs) is computed every `eval_every` steps; with fwd_chunk, each step's rows are run (and their
    gradients accumulated) in chunks of that size (full-forward hook path). Training runs at least `steps_min` steps
    and stops after `patience` steps without a new best validation loss (or at `steps_max`); the best-validation
    parameters are kept.
    Without validation rows the training-loss curve is used. Returns ({slot: Subspace}, info) where info has the loss
    curve, the initial / best validation loss, the relative drop, steps run, stop reason, chance level and whether the
    best validation loss is below chance (cross-entropy of a uniform guess, ln(n options))."""
    torch.manual_seed(seed)
    slots = sorted(set(slot_of))
    subs = {g: Subspace(n_dim, k).to(dev) for g in slots}
    params = [p for s in subs.values() for p in s.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    rng = random.Random(seed)
    rows_of = {g: [i for i, s in enumerate(slot_of) if s == g] for g in slots}
    queues = {g: [] for g in slots}
    vrows = [i for i, s in enumerate(val_slot_of) if s in subs]

    def take(g):
        out = []
        for _ in range(min(rows_per_slot, len(rows_of[g])) or 1):
            if not queues[g]:
                queues[g] = rng.sample(rows_of[g], len(rows_of[g]))
            out.append(queues[g].pop())
        return out

    def Rs(slot_list):
        stack = {g: subs[g].R for g in slots}
        return torch.stack([stack[g] for g in slot_list])

    def ce(zs, tg):
        return torch.stack([F.cross_entropy(z[None], torch.tensor([t], device=z.device)) for z, t in zip(zs, tg)])

    @torch.no_grad()
    def val_loss():
        if not vrows:
            return None, None
        tot, hit = 0.0, 0
        for s in range(0, len(vrows), 64):
            r = vrows[s:s + 64]
            zs = fwd(r, False, Rs([val_slot_of[i] for i in r]))
            tot += float(ce(zs, [val_targets[i] for i in r]).sum())
            hit += sum(int(z.argmax()) == val_targets[i] for z, i in zip(zs, r))
        return tot / len(vrows), hit / len(vrows)

    chance = float(sum(math.log(n_options[i]) for i in range(len(targets))) / max(1, len(targets)))
    curve, run = [], []
    v0, a0 = val_loss()
    curve.append({"step": 0, "train_loss": None, "val_loss": None if v0 is None else round(v0, 4),
                  "val_iia": None if a0 is None else round(a0, 4)})
    best = (v0 if v0 is not None else float("inf"), 0, {g: {n: p.detach().clone() for n, p in s.state_dict().items()}
                                                        for g, s in subs.items()})
    first_train = None
    reason = "steps_max"
    step = 0
    for step in range(1, steps_max + 1):
        rows = [r for g in slots for r in take(g)]
        opt.zero_grad()
        tot = 0.0
        ch = fwd_chunk or len(rows)
        for c in range(0, len(rows), ch):
            r = rows[c:c + ch]
            zs = fwd(r, True, Rs([slot_of[i] for i in r]))
            loss = ce(zs, [targets[i] for i in r]).sum() / len(rows)
            loss.backward()
            tot += float(loss.detach())
        opt.step()
        run.append(tot)
        if step % eval_every == 0 or step == steps_max:
            tl = sum(run) / len(run)
            run = []
            first_train = tl if first_train is None else first_train
            vl, va = val_loss()
            curve.append({"step": step, "train_loss": round(tl, 4), "val_loss": None if vl is None else round(vl, 4),
                          "val_iia": None if va is None else round(va, 4)})
            score = vl if vl is not None else tl
            if score < best[0] - 1e-4:
                best = (score, step, {g: {n: p.detach().clone() for n, p in s.state_dict().items()}
                                      for g, s in subs.items()})
            if log:
                log(f"  step {step}: train {tl:.4f} val {vl if vl is None else round(vl, 4)} (best {best[0]:.4f} @ "
                    f"{best[1]})")
            if step >= steps_min and step - best[1] >= patience:
                reason = "early_stop"
                break
    for g, s in subs.items():
        s.load_state_dict(best[2][g])
    init = v0 if v0 is not None else first_train
    info = {"steps": step, "stop": reason, "best_step": best[1], "init_loss": None if init is None else round(init, 4),
            "best_loss": round(best[0], 4) if best[0] != float("inf") else None, "loss_on": "val" if vrows else "train",
            "chance_loss": round(chance, 4), "n_train": len(targets), "n_val": len(vrows),
            "n_train_by_slot": {str(g): len(v) for g, v in rows_of.items()}, "curve": curve}
    if info["init_loss"] and info["best_loss"] is not None:
        info["rel_drop"] = round((info["init_loss"] - info["best_loss"]) / info["init_loss"], 4)
        info["below_chance"] = info["best_loss"] < chance
        info["converged"] = info["rel_drop"] >= 0.05 and reason == "early_stop"
    return subs, info
