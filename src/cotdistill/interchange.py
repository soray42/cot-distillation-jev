"""Interchange interventions at the readout position of a one-pass decision model (causal-audit tooling).

A site is (layer L, readout position): the residual stream entering decoder layer L at the last prompt token, which is
`hidden_states[L]` in Hugging Face's output (index 0 = embeddings), the same indexing as the hidden-state dumps. A
forward pre-hook on decoder layer L rewrites that one vector:

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
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .student import FINAL_TEMPLATE, collate, letter_token_ids, text_parts, Example


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
                bs: int = 8) -> torch.Tensor:
    """[n, len(layers), d] readout-position states entering each layer (float32, CPU), in input order."""
    body, head = text_parts(model)
    dev = head.weight.device
    out = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        b = _batch(tok, [prompts[i] for i in idx], [labels[i] for i in idx], max_len, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            res = body(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=True)
        rows = torch.arange(len(idx), device=dev)
        st = torch.stack([res.hidden_states[l][rows, b["last"]] for l in layers], 1).float().cpu()
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


def patched_logits(model, tok, base_prompts, labels, layer: int, fn, max_len: int, cache: dict) -> list[torch.Tensor]:
    dev = text_parts(model)[1].weight.device
    b = _batch(tok, base_prompts, labels, max_len, dev)
    with patch_site(model, layer, b["last"], fn):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            return _letter_logits(model, tok, b, labels, cache)


def iia(model, tok, pairs: list[dict], src_states: torch.Tensor, layer: int, method: str, sub: Subspace | None = None,
        max_len: int = 1536, bs: int = 8, seed: int = 0) -> dict:
    """Accuracy of the patched base on the target outcome. `pairs` entries carry base_prompt, labels (letters),
    target (index), base_pred and source_pred (indices, the unpatched model's own answers); `src_states[i]` is pair
    i's source state at `layer` ([d]). Also reports how often the patched answer copies the base or source answer."""
    cache: dict = {}
    hits = copies_b = copies_s = 0
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
                                max_len, cache)
            for p, z in zip(chunk, zs):
                k = int(z.argmax())
                hits += k == p["target"]
                copies_b += k == p["base_pred"]
                copies_s += k == p["source_pred"]
    n = max(1, len(pairs))
    return {"iia": hits / n, "copy_base": copies_b / n, "copy_source": copies_s / n, "n": len(pairs)}


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
                                max_len, cache)
            loss = sum(F.cross_entropy(z[None], torch.tensor([p["target"]], device=z.device)) for z, p in zip(zs, chunk))
            opt.zero_grad()
            (loss / len(chunk)).backward()
            opt.step()
            tot += float(loss)
        log(f"das layer {layer} k {k} epoch {ep + 1}: loss {tot / len(order):.4f}")
    return sub
