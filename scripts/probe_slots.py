"""Linear probes of the latent slot states (diagnostic GPU-1 of notes/latent_slots_v21_audit_2026-10-20.md): is the
value of worked step j decodable from the hidden states of slot j, at any layer, even where the frozen-head lens
(train_student --slot-readout lens) reads only the marginals? Training recipe only: nothing here changes a model.

  python scripts/probe_slots.py --model <run>/model --out <run>/probe \\
      --train data/student_gsm8k_tf_si/train.jsonl --train-n 3000 \\
      --heldout val=data/student_gsm8k_tf_si/val.jsonl gsm8kv2=data/eval/gsm8k_mc_v2_steps.jsonl
  python scripts/probe_slots.py --model models/Qwen3.5-4B --slot-spec-from <LSL run dir> ...      # base / A2 model
  python scripts/probe_slots.py --model models/Qwen3.5-4B --latent-slots 6 --slot-c 4 --slot-tokens distinct \\
      --slot-layer 0.67 ...                                     # the slot line from train_student's own flags
  python scripts/probe_slots.py --table <run>/probe/probe_results.json --control <ctrl>/probe/probe_results.json

Prompts: exactly the training / evaluation final question (student.final_text with the slot line of the spec): the
model's latent_slots.json by default, else --slot-spec-from (a run or model dir with latent_slots.json, its overflow
from the run's metrics.json if the sidecar has none, or such a json file), else the train_student slot flags given
here, else no slot line (--no-slots, or a model without a sidecar: the anchors only).

Positions (2K + 2 for K slots): slot<j>_last (slot j's last token) and slot<j>_mean (the mean over its c tokens), then
problem (the last token of the problem text: the token that holds its last character, before the slot line and the
options; Qwen3.5: "?", the "\\n" after it is a token of its own) and answer (the "Answer:" cue's last token, the
letter readout position). Layers: 0 = the embedding output (the input of decoder layer 0), l = the output of decoder
layer l - 1, l = 1..n (33 for Qwen3.5-4B; the train_student slot layer L is probe layer L + 1, 21 for 0.67 of 32),
captured by hooks (forward pre-hook on decoder layer 0, forward hooks on every decoder layer: the raw residual
stream, no final norm), in eval mode without gradient (as evaluate), each hook once per forward.

Targets per item (its worked_steps values, student.slot_values; an item without a chain is skipped): step1..stepS
(--n-steps S, default 6; step j absent from a shorter chain) and last (the chain's last value, the answer's), the same
for every position (slot j's own step is the diagonal "step<j>"; the others tell whether slot j holds another step).
Features of a value: first_digit and last_digit (10 classes; the first / last digit character, sign skipped),
n_digits (5 classes: 1, 2, 3, 4, 5+ digits of the integer part) and value50 (51 classes: the 50 most frequent values of
that target over the TRAIN subset, + other).

value50 also reports acc_in: the accuracy on the rows whose value is one of the 50 (the majority there:
the train subset's most frequent value), since "other" can be the majority class of a diverse target.

Probes: one L2-regularised multinomial logistic regression per (layer, position, target, feature), on the
features standardised by the fit rows' mean / std, full-batch Adam in float32 (--fit-steps, --lr), every --wd value
fitted; fit on the first 80% of the train subset (--select-frac), the wd chosen per probe on the other 20% (ties: the
larger wd), evaluated on every held-out set and their pool ("heldout") next to the majority baseline (the most
frequent class of the train subset). All probes of a (layer, wd) are one batched problem (separate weights and
softmax blocks, so Adam treats each probe exactly as if alone). The train subset: --train-n items with a chain drawn by
random.Random(--seed) from --train (default seed 0, the same subset and split for every model, so runs compare).

Lens (slots only, --no-lens to skip): at every layer, the model's own final norm + LM head on slot j's FIRST token
(the position train_student's lens trains to give the value's first token): first-token accuracy against each target
(lens.first) and the exact value on slot j's first n tokens (n = the value's token count, at most c; lens.value), next
to the majority first token of the train subset.

Memory: one forward pass over all items per chunk of --layer-chunk layers (default 11: three passes for 33 layers), the
chunk's states of the 2K + 2 positions kept on the CPU in the model's dtype (Qwen3.5-4B: 2560 x 14 x 2 B = 72 kB per
item and layer, 3.6 GB for 4,500 items and 11 layers); each layer goes to the GPU alone, in float32, for its probes.

Outputs in --out: probe_results.json (the grid: probe[position][target][feature] = acc / majority / n per set, each
acc a list over layers, plus wd and select_acc per layer; lens[slot][target]; counts, spec, timings), probe_table.txt
(the compact table, also printed; with --control the delta to the control's probe at the same layer), probe_heatmap.png
with --png (matplotlib; held-out acc - majority by layer, value50 and first_digit, diagonal targets).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FEATURES = ("first_digit", "last_digit", "n_digits", "value50")
N_CLASSES = {"first_digit": 10, "last_digit": 10, "n_digits": 5, "value50": 51}
TOP_VALUES = 50
ANCHORS = ("problem", "answer")


# ------------------------------------------------------------------------------------------------ targets / features
def digit_features(v: str) -> dict | None:
    """first_digit / last_digit (0-9) and n_digits (class 0..4 = 1, 2, 3, 4, 5+ digits of the integer part) of a value
    string, None without a digit."""
    s = str(v).strip().lstrip("+-$")
    digits = [ch for ch in s if ch.isdigit()]
    if not digits:
        return None
    nd = sum(ch.isdigit() for ch in s.split(".", 1)[0]) or 1
    return {"first_digit": int(digits[0]), "last_digit": int(digits[-1]), "n_digits": min(nd, 5) - 1}


def item_values(item: dict, n_steps: int) -> dict:
    """{step1..step<n_steps>, last} -> the item's worked-step value (None: no such step)."""
    from cotdistill.student import slot_values
    vals = slot_values(item)
    out = {f"step{j + 1}": vals[j] if j < len(vals) else None for j in range(n_steps)}
    out["last"] = vals[-1] if vals else None
    return out


def target_names(n_steps: int) -> list[str]:
    return [f"step{j + 1}" for j in range(n_steps)] + ["last"]


def build_labels(values: list[dict], targets: list[str], vocab: dict) -> "torch.Tensor":
    """[N, T, F] int64 class labels (-1: no target), value50 against vocab[target] (value -> class, 50 = other)."""
    import torch
    y = torch.full((len(values), len(targets), len(FEATURES)), -1, dtype=torch.long)
    for n, vs in enumerate(values):
        for t, name in enumerate(targets):
            v = vs.get(name)
            f = digit_features(v) if v is not None else None
            if f is None:
                continue
            for k, feat in enumerate(FEATURES[:3]):
                y[n, t, k] = f[feat]
            y[n, t, 3] = vocab[name].get(v, TOP_VALUES)
    return y


def value_vocab(values: list[dict], targets: list[str]) -> dict:
    """target -> {value: class} for the TOP_VALUES most frequent values of the train subset (ties: first seen)."""
    out = {}
    for name in targets:
        c = collections.Counter(vs[name] for vs in values if vs.get(name) is not None and digit_features(vs[name]))
        out[name] = {v: i for i, (v, _) in enumerate(c.most_common(TOP_VALUES))}
    return out


# ------------------------------------------------------------------------------------------------------- slot spec
def spec_from_dict(tok, sd: dict):
    """The SlotSpec of a latent_slots.json dict, as eval_student builds it."""
    from cotdistill.student import SlotSpec, choose_slot_token
    token = choose_slot_token(tok, sd.get("token", "auto"))
    return SlotSpec(sd["k"], sd.get("c", 1), token, sd.get("place", "before-options"), tokens=tuple(sd.get("tokens", ())),
                    layer=sd.get("layer"), readout=sd.get("readout", "decode"), fill=sd.get("fill", "none"),
                    head_grad=sd.get("head_grad", False), pad_weight=sd.get("pad_weight", 1.0),
                    overflow=sd.get("overflow", "merge"))


def read_spec_source(path: str) -> tuple[dict, str]:
    """--slot-spec-from: (latent_slots dict, where from). A run dir (latent_slots.json, else model/latent_slots.json; the
    overflow from its metrics.json when the sidecar has none), a model dir, or a json file (a sidecar, or a
    metrics.json whose latent_slots is taken)."""
    p = Path(path)
    if p.is_file():
        d = json.loads(p.read_text())
        if "latent_slots" in d:
            d = d["latent_slots"]
        keep = ("k", "c", "token", "place", "tokens", "layer", "readout", "fill", "head_grad", "pad_weight", "overflow")
        return {k: v for k, v in d.items() if k in keep}, str(p)
    for side in (p / "latent_slots.json", p / "model" / "latent_slots.json"):
        if side.is_file():
            d = json.loads(side.read_text())
            m = p / "metrics.json"
            if "overflow" not in d and m.is_file():
                ov = json.loads(m.read_text()).get("latent_slots", {}).get("overflow", "merge")
                if ov != "merge":
                    d["overflow"] = ov
            return d, str(side)
    raise SystemExit(f"--slot-spec-from {path}: no latent_slots.json there (nor in model/)")


def resolve_spec(args, tok, model):
    """(SlotSpec or None, source) by the rules of the docstring; registered with set_slot_prompt."""
    from cotdistill.student import (SlotSpec, choose_slot_token, choose_slot_tokens, resolve_slot_layer,
                                    set_slot_prompt)
    spec, src = None, "none"
    if args.no_slots:
        pass
    elif args.latent_slots:
        if args.slot_tokens == "distinct":
            tokens = choose_slot_tokens(tok, args.latent_slots, args.slot_token)
            token = tokens[0]
        else:
            tokens, token = (), choose_slot_token(tok, args.slot_token)
        layer, _ = resolve_slot_layer(model, args.slot_layer)
        spec = SlotSpec(args.latent_slots, args.slot_c, token, args.slot_place, tokens=tokens, layer=layer,
                        readout="lens" if layer is not None else "decode",
                        fill=args.slot_fill if layer is not None else "none",
                        pad_weight=args.slot_pad_weight if layer is not None else 1.0,
                        overflow=args.slot_overflow if layer is not None else "merge")
        src = "flags"
    elif args.slot_spec_from:
        sd, src = read_spec_source(args.slot_spec_from)
        spec = spec_from_dict(tok, sd)
    elif (Path(args.model) / "latent_slots.json").is_file():
        src = str(Path(args.model) / "latent_slots.json")
        spec = spec_from_dict(tok, json.loads(Path(src).read_text()))
    set_slot_prompt(tok, spec)
    return spec, src


def position_names(spec) -> list[str]:
    k = spec.k if spec is not None else 0
    return [f"slot{j + 1}_{w}" for j in range(k) for w in ("last", "mean")] + list(ANCHORS)


# ------------------------------------------------------------------------------------------------------- positions
def problem_end(prompt: str) -> int:
    """Character length of the problem text of a prompt: up to its last "\\nOptions:\\n" block, else all of it."""
    head, sep, _ = prompt.rpartition("\nOptions:\n")
    return len(head) if sep else len(prompt)


def encode_item(tok, spec, item: dict, max_len: int) -> dict | None:
    """The final question of item as evaluate encodes it (special prefix + text ids, no cut): ids and the positions of
    the slot tokens (k x c, slot by slot), the problem's last token and the answer cue's last token. None when the
    text does not fit max_len (it would be cut on the left)."""
    import torch
    from cotdistill.student import _slot_ids, _slot_positions, final_text, special_prefix, text_ids
    text = final_text(tok, item["prompt"])
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    if ids != text_ids(tok, text):
        raise ValueError(f"{item['item_id']}: the offsets encoding differs from the training encoding")
    pre = special_prefix(tok)
    full = pre + ids
    if len(full) > max_len:
        return None
    end = problem_end(item["prompt"])
    prob = max(i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > a and a < end) + len(pre)
    slots = []
    if spec is not None:
        slots = _slot_positions(torch.tensor(full), _slot_ids(tok), spec, item["item_id"]).tolist()
        if prob >= slots[0]:
            raise ValueError(f"{item['item_id']}: the problem's last token is not before the slot line")
    return {"ids": full, "slots": slots, "problem": prob, "answer": len(full) - 1}


# --------------------------------------------------------------------------------------------------------- capture
class LayerCapture:
    """Hooks on the text decoder: pre-hook on decoder layer 0 (its input = the embedding output, probe layer 0) and a
    forward hook on every decoder layer i (its output, probe layer i + 1). Each call gathers the rows' query positions
    (self.idx [B, Q], set before the forward) into self.got[layer] ([B, Q, d], detached), only for the layers in keep;
    with lens, every layer's first kc query positions also go through the final norm + LM head (argmax ids into
    self.lens[layer], [B, kc]). self.calls counts the calls per layer (one per forward each)."""

    def __init__(self, body, head, layers, keep: set, lens_kc: int = 0, softcap=None):
        self.body, self.head, self.layers, self.keep = body, head, layers, keep
        self.kc, self.cap = lens_kc, softcap
        self.idx, self.got, self.lens = None, {}, {}
        self.calls = collections.Counter()
        self.handles = []

    def _take(self, layer: int, h):
        import torch
        from cotdistill.student import head_logits
        self.calls[layer] += 1
        if layer not in self.keep and not self.kc:
            return
        rows = torch.arange(h.shape[0], device=h.device)[:, None]
        g = h[rows, self.idx.to(h.device)].detach()
        if layer in self.keep:
            self.got[layer] = g
        if self.kc:
            bias = getattr(self.head, "bias", None)
            z = head_logits(self.body.norm(g[:, :self.kc]), self.head.weight, bias, self.cap)
            self.lens[layer] = z.argmax(-1)

    def __enter__(self):
        def pre(mod, args, kwargs):
            self._take(0, args[0] if args else kwargs["hidden_states"])

        def post_for(i):
            def post(mod, args, out):
                self._take(i + 1, out[0] if isinstance(out, tuple) else out)
            return post
        self.handles = [self.layers[0].register_forward_pre_hook(pre, with_kwargs=True)]
        self.handles += [m.register_forward_hook(post_for(i)) for i, m in enumerate(self.layers)]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


def capture(model, tok, spec, encs: list[dict], keep: list[int], *, bs: int, lens: bool) -> tuple[dict, "torch.Tensor"]:
    """One forward pass over every encoded item (batches of bs by length, right-padded as collate), no gradient, eval
    mode: {probe layer in keep: [N, P, d] CPU tensor in the model's dtype} with P = 2k + 2 (position_names order) and,
    with lens, the lens argmax ids [N, n_layers + 1, k x c] (int32). The training mode is restored."""
    import torch
    from cotdistill.fsdp import decoder_layers
    from cotdistill.student import logit_softcap, text_parts
    body, head = text_parts(model)
    layers = decoder_layers(body)
    n_layers = len(layers) + 1
    k, c = (spec.k, spec.c) if spec is not None else (0, 1)
    kc = k * c
    dev = head.weight.device
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    N = len(encs)
    out = {}
    lens_ids = torch.zeros((N, n_layers, kc), dtype=torch.int32) if lens and kc else None
    order = sorted(range(N), key=lambda i: len(encs[i]["ids"]))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), LayerCapture(body, head, layers, set(keep), kc if lens_ids is not None else 0,
                                                  logit_softcap(model)) as cap:
            for s in range(0, N, bs):
                rows = order[s:s + bs]
                L = max(len(encs[i]["ids"]) for i in rows)
                ids = torch.full((len(rows), L), pad, dtype=torch.long)
                mask = torch.zeros((len(rows), L), dtype=torch.long)
                q = []
                for r, i in enumerate(rows):
                    e = encs[i]
                    ids[r, :len(e["ids"])] = torch.tensor(e["ids"])
                    mask[r, :len(e["ids"])] = 1
                    q.append(e["slots"] + [e["problem"], e["answer"]])
                cap.idx = torch.tensor(q, dtype=torch.long)
                cap.got, cap.lens = {}, {}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    body(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False)
                idx = torch.tensor(rows)
                for layer in keep:
                    g = cap.got[layer]
                    sl = g[:, :kc].float().view(len(rows), k, c, -1) if kc else None
                    parts = []
                    for j in range(k):
                        parts += [sl[:, j, -1], sl[:, j].mean(1)]
                    parts += [g[:, kc].float(), g[:, kc + 1].float()]
                    x = torch.stack(parts, 1).to(g.dtype).cpu()
                    if layer not in out:
                        out[layer] = torch.empty((N,) + tuple(x.shape[1:]), dtype=x.dtype)
                    out[layer][idx] = x
                if lens_ids is not None:
                    for layer, z in cap.lens.items():
                        lens_ids[idx, layer] = z.to(torch.int32).cpu()
            calls = dict(cap.calls)
    finally:
        if was_training:
            model.train()
    n_fwd = math.ceil(N / bs)
    bad = {li: n for li, n in calls.items() if n != n_fwd}
    if bad or len(calls) != n_layers:
        raise RuntimeError(f"capture: hook calls per layer {calls} for {n_fwd} forwards")
    return out, lens_ids


# ------------------------------------------------------------------------------------------------------------ fit
def fit_probes(x: "torch.Tensor", y: "torch.Tensor", fit_idx, sel_idx, eval_idx: dict, *, wds: list[float],
               steps: int, lr: float, device) -> dict:
    """Probes of one layer. x [N, P, d] (any float dtype), y [N, T, F] labels (-1 = none, the same for every
    position). Every (position, target, feature) is its own multinomial logistic regression (bias unregularised; loss
    = mean CE over its labelled fit rows + wd x |W|^2) on x standardised by the fit rows; all of them and every wd
    are fitted at once by full-batch Adam in float32 (separate weights and softmax blocks: Adam is elementwise, so each
    probe's updates are those it would get alone). Returns {"acc": {set: [P, T, F]}, "wd": [P, T, F],
    "select_acc": [P, T, F], "acc_in": {set: [P, T]}}: accuracy on eval_idx's sets with the wd of the best select
    accuracy (ties: the larger wd); NaN where a set has no label. acc_in: value50 on the rows whose value is one of
    the 50 (not "other"): the exact-value accuracy that a constant "other" cannot reach."""
    import torch
    P, d = x.shape[1], x.shape[2]
    T, F = y.shape[1], y.shape[2]
    sizes = [N_CLASSES[f] for f in FEATURES]
    G = len(wds)
    dev = torch.device(device)
    xs = x.to(dev, torch.float32).transpose(0, 1).contiguous()            # [P, N, d]
    mu = xs[:, fit_idx].mean(1, keepdim=True)
    sd = xs[:, fit_idx].std(1, keepdim=True).clamp_min(1e-6)
    xs = (xs - mu) / sd
    yd = y.to(dev)
    wd = torch.tensor(wds, device=dev, dtype=torch.float32)
    W = [torch.zeros((P, d, G * T * C), device=dev, requires_grad=True) for C in sizes]
    b = [torch.zeros((P, 1, G * T * C), device=dev, requires_grad=True) for C in sizes]
    opt = torch.optim.Adam(W + b, lr=lr)
    xf, yf = xs[:, fit_idx], yd[fit_idx]                                   # [P, n, d], [n, T, F]

    def logp(xx, f):                                                       # [P, n, G, T, C]
        z = torch.baddbmm(b[f], xx, W[f])
        return torch.log_softmax(z.view(P, xx.shape[1], G, T, sizes[f]), -1)

    masks = [(yf[:, :, f] >= 0).float() for f in range(F)]                 # [n, T]
    cnt = [m.sum(0).clamp_min(1) for m in masks]                           # [T]
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for f in range(F):
            lp = logp(xf, f)
            yy = yf[:, :, f].clamp_min(0)
            nll = -lp.gather(-1, yy[None, :, None, :, None].expand(P, -1, G, -1, 1)).squeeze(-1)    # [P, n, G, T]
            per = (nll * masks[f][None, :, None, :]).sum(1) / cnt[f]        # [P, G, T]: mean CE per probe
            reg = (W[f].view(P, d, G, T, sizes[f]) ** 2).sum((1, 4))        # [P, G, T]
            loss = loss + (per + wd[None, :, None] * reg).sum()
        loss.backward()
        opt.step()

    vf = FEATURES.index("value50")

    @torch.no_grad()
    def accuracy(idx, in_vocab=False):                                     # [P, G, T, F] (NaN: no labels)
        acc = torch.full((P, G, T, F), float("nan"), device=dev)
        if len(idx) == 0:
            return acc
        xx, yy = xs[:, idx], yd[idx]
        for f in range(F):
            if in_vocab and f != vf:
                continue
            pred = logp(xx, f).argmax(-1)                                  # [P, n, G, T]
            lab = yy[:, :, f]
            m = ((lab >= 0) & (lab < TOP_VALUES) if in_vocab else lab >= 0)[None, :, None, :]
            hit = ((pred == lab[None, :, None, :]) & m).sum(1).float()     # [P, G, T]
            n = m.sum(1).float()
            acc[..., f] = torch.where(n > 0, hit / n.clamp_min(1), acc[..., f])
        return acc

    sel = accuracy(sel_idx)
    score = torch.nan_to_num(sel, nan=-1.0) + 1e-9 * torch.arange(G, device=dev)[None, :, None, None]  # ties: larger
    best = score.argmax(1, keepdim=True)                                   # [P, 1, T, F]
    pick = lambda a: a.gather(1, best).squeeze(1)
    return {"acc": {name: pick(accuracy(idx)).cpu() for name, idx in eval_idx.items()},
            "acc_in": {name: pick(accuracy(idx, True)).cpu()[..., vf] for name, idx in eval_idx.items()},
            "wd": torch.tensor(wds)[best.squeeze(1).cpu()], "select_acc": pick(sel).cpu()}


def majority_in(y: "torch.Tensor", train_idx, eval_idx: dict) -> dict:
    """value50 on the in-vocabulary rows: the share of the train subset's most frequent value (not "other") among the
    eval rows whose value is one of the 50: {set: [T]}."""
    import torch
    f = FEATURES.index("value50")
    out = {}
    tops = []
    for t in range(y.shape[1]):
        v = y[train_idx, t, f]
        v = v[(v >= 0) & (v < TOP_VALUES)]
        tops.append(int(torch.bincount(v).argmax()) if len(v) else -1)
    top = torch.tensor(tops)
    for name, idx in eval_idx.items():
        lab = y[idx, :, f]
        m = (lab >= 0) & (lab < TOP_VALUES)
        n = m.sum(0).float()
        out[name] = torch.where(n > 0, ((lab == top[None]) & m).sum(0).float() / n.clamp_min(1),
                                torch.full_like(n, float("nan")))
    return out


def majority(y: "torch.Tensor", train_idx, eval_idx: dict) -> tuple[dict, "torch.Tensor"]:
    """Accuracy of the train subset's most frequent class per (target, feature) on each eval set: {set: [T, F]}."""
    import torch
    T, F = y.shape[1], y.shape[2]
    maj = torch.zeros((T, F), dtype=torch.long)
    for t in range(T):
        for f in range(F):
            v = y[train_idx, t, f]
            v = v[v >= 0]
            maj[t, f] = torch.bincount(v).argmax() if len(v) else -1
    out = {}
    for name, idx in eval_idx.items():
        yy = y[idx]
        m = yy >= 0
        n = m.sum(0).float()
        hit = ((yy == maj[None]) & m).sum(0).float()
        out[name] = torch.where(n > 0, hit / n.clamp_min(1), torch.full_like(n, float("nan")))
    return out, maj


def lens_scores(tok, spec, lens_ids, values: list[dict], targets: list[str], train_idx, eval_idx: dict) -> dict:
    """lens[slot<j>][target]: first (first-token accuracy of slot j's first position, per layer), value (the value's
    tokens on slot j's first n positions, n <= c: exact, per layer), majority_first (the accuracy of the train
    subset's most frequent first token), n."""
    import torch
    k, c = spec.k, spec.c
    enc = {}
    out = {}
    for t in targets:
        toks = []
        for vs in values:
            v = vs.get(t)
            if v is not None and v not in enc:
                enc[v] = tok.encode(v, add_special_tokens=False)
            toks.append(enc[v] if v is not None else None)
        first_tr = collections.Counter(toks[i][0] for i in train_idx if toks[i])
        maj = first_tr.most_common(1)[0][0] if first_tr else None
        per_set = {}
        for name, idx in eval_idx.items():
            idx = [i for i in idx if toks[i]]
            tgt = torch.full((len(idx), c), -1, dtype=torch.int32)
            fit = torch.zeros(len(idx), dtype=torch.bool)
            for r, i in enumerate(idx):
                if len(toks[i]) <= c:
                    tgt[r, :len(toks[i])] = torch.tensor(toks[i], dtype=torch.int32)
                    fit[r] = True
            per_set[name] = (idx, tgt, fit)
        for j in range(k):
            row = {"first": {}, "value": {}, "majority_first": {}, "n": {}}
            for name, (idx, tgt, fit) in per_set.items():
                row["n"][name] = len(idx)
                if not idx:
                    continue
                p = lens_ids[idx][:, :, j * c:(j + 1) * c]                     # [n, layers, c]
                first = torch.tensor([toks[i][0] for i in idx], dtype=torch.int32)
                row["first"][name] = [round(v, 5) for v in (p[:, :, 0] == first[:, None]).float().mean(0).tolist()]
                ok = ((p == tgt[:, None, :]) | (tgt[:, None, :] < 0)).all(-1) & fit[:, None]
                row["value"][name] = [round(v, 5) for v in ok.float().mean(0).tolist()]
                row["majority_first"][name] = round(float((first == maj).float().mean()), 5) if maj is not None else None
            out.setdefault(f"slot{j + 1}", {})[t] = row
    return out


# ---------------------------------------------------------------------------------------------------------- table
def table_layers(n_layers: int, slot_layer: int | None) -> list[int]:
    """The table's layer columns: every layer for small models, else 0, every 4th, the slot layer and the last."""
    if n_layers <= 12:
        return list(range(n_layers))
    ls = {0, n_layers - 1} | set(range(4, n_layers, 4)) | ({slot_layer} if slot_layer is not None else set())
    return sorted(x for x in ls if 0 <= x < n_layers)


def _f(v, width=5, signed=False) -> str:
    if v is None or v != v:
        return " " * (width - 1) + "-"
    return f"{v:+{width}.3f}" if signed else f"{v:{width}.3f}"


def _best(acc: list) -> int:
    return max(range(len(acc)), key=lambda i: -1.0 if acc[i] is None else acc[i])


def table_text(res: dict, control: dict | None = None, set_name: str = "heldout") -> str:
    """The compact table: per position its own targets (slot j: step j and last; problem / answer: step1, step2,
    last), every feature: n, majority, held-out accuracy at a few layers, the best layer (and the control probe at that
    layer with the delta); then the lens' first-token accuracy of each slot."""
    nl = res["n_layers"]
    sl = res.get("slot_probe_layer")
    cols = table_layers(nl, sl)
    lines = [f"model {res['model']}  spec {res.get('spec_source')}  set {set_name}  (layer 0 = embeddings, l = output "
             f"of decoder layer l - 1{'; slot layer = ' + str(sl) if sl is not None else ''})",
             "items: " + ", ".join(f"{k} {v}" for k, v in res["items"].items())]
    if control:
        lines.append(f"control {control['model']}: ctrl = its probe at the best layer, delta = acc - ctrl")
    head = f"{'position':<12} {'target':<6} {'feature':<11} {'n':>5} {'maj':>5} " + \
        " ".join(f"{'L' + str(x):>5}" for x in cols) + f" | {'best':>10}" + (f" {'ctrl':>5} {'delta':>6}" if control else "")
    lines += [head, "-" * len(head)]
    for pos in res["positions"]:
        tg = [f"step{int(pos[4:].split('_')[0])}", "last"] if pos.startswith("slot") else ["step1", "step2", "last"]
        for t in tg:
            if t not in res["targets"]:
                continue
            for f in FEATURES + ("value50_in",):
                r = res["probe"][pos][t][f.replace("_in", "")]
                if f == "value50_in":            # value50 on the rows whose value is one of the 50
                    if "acc_in" not in r:
                        continue
                    r = {"acc": r["acc_in"], "majority": r["majority_in"], "n": r["n_in"]}
                acc = r["acc"].get(set_name)
                if acc is None:
                    continue
                b = _best(acc)
                row = (f"{pos:<12} {t:<6} {f:<11} {r['n'].get(set_name, 0):>5} {_f(r['majority'].get(set_name))} "
                       + " ".join(_f(acc[x]) for x in cols) + f" | {_f(acc[b])}@L{b:<3}")
                if control:
                    cr = control.get("probe", {}).get(pos, {}).get(t, {}).get(f.replace("_in", ""), {})
                    ca = cr.get("acc_in" if f == "value50_in" else "acc", {}).get(set_name)
                    cv = ca[b] if ca and b < len(ca) else None
                    row += f" {_f(cv)} " + (_f(acc[b] - cv, 6, True) if cv is not None and acc[b] is not None
                                           else "     -")
                lines.append(row)
    if res.get("lens"):
        lines += ["", f"lens (final norm + LM head on slot j's first token): first-token accuracy, set {set_name}"]
        head = f"{'slot':<7} {'target':<6} {'n':>5} {'maj':>5} " + " ".join(f"{'L' + str(x):>5}" for x in cols) + \
            f" | {'best':>10} {'value@best':>10}"
        lines += [head, "-" * len(head)]
        for slot, by_t in res["lens"].items():
            for t in (f"step{int(slot[4:])}", "last"):
                r = by_t.get(t)
                if not r or set_name not in r["first"]:
                    continue
                acc = r["first"][set_name]
                b = _best(acc)
                lines.append(f"{slot:<7} {t:<6} {r['n'][set_name]:>5} {_f(r['majority_first'][set_name])} "
                             + " ".join(_f(acc[x]) for x in cols)
                             + f" | {_f(acc[b])}@L{b:<3} {_f(r['value'][set_name][b], 10)}")
    return "\n".join(lines) + "\n"


def heatmap(res: dict, path: Path, set_name: str = "heldout") -> bool:
    """probe_heatmap.png: held-out accuracy - majority by layer, value50 and first_digit, each position's diagonal
    target (slot j: step j; anchors: step1)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("--png: no matplotlib, skipped", flush=True)
        return False
    feats = ("value50", "first_digit")
    fig, axes = plt.subplots(1, len(feats), figsize=(7 * len(feats), 0.32 * len(res["positions"]) + 2), squeeze=False)
    for ax, f in zip(axes[0], feats):
        rows, labels = [], []
        for pos in res["positions"]:
            t = f"step{int(pos[4:].split('_')[0])}" if pos.startswith("slot") else "step1"
            r = res["probe"][pos][t][f]
            m = r["majority"].get(set_name)
            if set_name in r["acc"] and m is not None:
                rows.append([float("nan") if a is None else a - m for a in r["acc"][set_name]])
                labels.append(f"{pos} / {t}")
        if not rows:
            continue
        im = ax.imshow(rows, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xlabel("layer (0 = embeddings)")
        ax.set_title(f"{f}: {set_name} acc - majority")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------------------------------------- main
def read_jsonl(p: str) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", help="checkpoint or base model directory")
    ap.add_argument("--out", help="output directory (probe_results.json, probe_table.txt)")
    ap.add_argument("--train", help="training items (jsonl with worked_steps) for the probes' fit / select subset")
    ap.add_argument("--train-n", type=int, default=3000, help="items of --train with a chain (all if fewer)")
    ap.add_argument("--select-frac", type=float, default=0.2, help="share of the train subset that picks the wd")
    ap.add_argument("--heldout", nargs="*", default=[], help="name=path.jsonl held-out sets (items with worked_steps)")
    ap.add_argument("--heldout-n", type=int, default=0, help="first N items with a chain of each held-out set (0: all)")
    ap.add_argument("--seed", type=int, default=0, help="train subset draw and fit / select split (keep it fixed "
                    "across the models compared)")
    ap.add_argument("--n-steps", type=int, default=6, help="targets step1..stepN, plus last")
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--layer-chunk", type=int, default=11, help="layers captured per forward pass over the items "
                    "(0: all in one pass)")
    ap.add_argument("--fit-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", default="1e-4,1e-3,1e-2,1e-1,1", help="comma list of L2 weights (each fitted, picked on "
                    "the select split)")
    ap.add_argument("--no-lens", action="store_true", help="skip the frozen-head lens readout")
    ap.add_argument("--png", action="store_true", help="also write probe_heatmap.png (matplotlib)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--control", default="", help="probe_results.json of a control model: deltas in the table")
    ap.add_argument("--table", default="", help="probe_results.json: print its table (with --control) and exit, no "
                    "model")
    ap.add_argument("--no-slots", action="store_true", help="no slot line (anchors only)")
    ap.add_argument("--slot-spec-from", default="", help="run / model dir with latent_slots.json, or a json file")
    ap.add_argument("--latent-slots", type=int, default=0, help="the slot line from train_student's flags: K slots")
    ap.add_argument("--slot-c", type=int, default=1)
    ap.add_argument("--slot-token", default="auto")
    ap.add_argument("--slot-tokens", default="same", choices=["same", "distinct"])
    ap.add_argument("--slot-place", default="before-options", choices=["before-options", "after-options"])
    ap.add_argument("--slot-layer", default="final")
    ap.add_argument("--slot-fill", default="none", choices=["none", "last", "answer"])
    ap.add_argument("--slot-overflow", default="merge", choices=["merge", "truncate"])
    ap.add_argument("--slot-pad-weight", type=float, default=1.0)
    args = ap.parse_args()

    control = json.loads(Path(args.control).read_text()) if args.control else None
    if args.table:
        print(table_text(json.loads(Path(args.table).read_text()), control), end="")
        return
    if not (args.model and args.out and args.train and args.heldout):
        raise SystemExit("--model, --out, --train and --heldout are needed (or --table)")
    if sum(bool(x) for x in (args.no_slots, args.latent_slots, args.slot_spec_from)) > 1:
        raise SystemExit("--no-slots, --latent-slots and --slot-spec-from exclude each other")
    import torch
    import transformers
    from cotdistill.student import load_model, worked_steps
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if not tok.is_fast:
        raise SystemExit("probe_slots needs a fast tokenizer (offsets of the problem's last token)")
    model = load_model(args.model, torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
    try:
        spec, src = resolve_spec(args, tok, model)
    except ValueError as e:
        raise SystemExit(f"slot spec: {e}")
    print(f"slot spec ({src}): {spec}", flush=True)
    targets = target_names(args.n_steps)
    positions = position_names(spec)

    # ---- items: train subset (fit + select) and the held-out sets, each with a chain and fitting max_len
    skipped = collections.Counter()

    def usable(items, name, limit=0):
        keep = []
        for it in items:
            if spec is not None and any(t in it["prompt"] for t in spec.slot_tokens()):
                raise SystemExit(f"{name}: a slot token occurs in the prompt of {it['item_id']}")
            if not worked_steps(it):
                skipped[f"{name}_no_chain"] += 1
                continue
            e = encode_item(tok, spec, it, args.max_len)
            if e is None:
                skipped[f"{name}_too_long"] += 1
                continue
            keep.append((it, e))
            if limit and len(keep) == limit:
                break
        return keep

    pool = usable(read_jsonl(args.train), "train")
    rng = random.Random(f"probe|{args.seed}")
    pick = sorted(rng.sample(range(len(pool)), min(args.train_n, len(pool))))
    rng.shuffle(pick)
    train = [pool[i] for i in pick]
    n_sel = max(1, int(round(args.select_frac * len(train))))
    sets = {"train_fit": train[:len(train) - n_sel], "train_select": train[len(train) - n_sel:]}
    for kv in args.heldout:
        name, path = kv.split("=", 1)
        sets[name] = usable(read_jsonl(path), name, args.heldout_n)
    names = list(sets)
    rows = [x for n in names for x in sets[n]]
    span, s0 = {}, 0
    for n in names:
        span[n] = list(range(s0, s0 + len(sets[n])))
        s0 += len(sets[n])
    held = [n for n in names if not n.startswith("train_")]
    eval_idx = {n: span[n] for n in held}
    eval_idx["heldout"] = [i for n in held for i in span[n]]
    fit_idx, sel_idx = span["train_fit"], span["train_select"]
    train_idx = fit_idx + sel_idx
    values = [item_values(it, args.n_steps) for it, _ in rows]
    vocab = value_vocab([values[i] for i in train_idx], targets)
    y = build_labels(values, targets, vocab)
    maj, maj_cls = majority(y, train_idx, eval_idx)
    maj_in = majority_in(y, train_idx, eval_idx)
    encs = [e for _, e in rows]
    counts = {n: len(v) for n, v in sets.items()}
    print(f"items {counts}, skipped {dict(skipped)}; positions {len(positions)}, targets {targets}", flush=True)

    # ---- capture (chunks of layers) and probes, layer by layer
    from cotdistill.fsdp import decoder_layers
    from cotdistill.student import text_parts
    n_layers = len(decoder_layers(text_parts(model)[0])) + 1
    chunk = args.layer_chunk if args.layer_chunk > 0 else n_layers
    wds = [float(x) for x in args.wd.split(",") if x]
    P, T, F = len(positions), len(targets), len(FEATURES)
    acc = {n: torch.full((n_layers, P, T, F), float("nan")) for n in eval_idx}
    acc_in = {n: torch.full((n_layers, P, T), float("nan")) for n in eval_idx}
    wd_l = torch.zeros((n_layers, P, T, F))
    sel_l = torch.zeros((n_layers, P, T, F))
    lens_ids, timing = None, {"capture_s": 0.0, "fit_s": 0.0}
    for c0 in range(0, n_layers, chunk):
        keep = list(range(c0, min(n_layers, c0 + chunk)))
        t1 = time.time()
        feats, li = capture(model, tok, spec, encs, keep, bs=args.bs, lens=not args.no_lens and c0 == 0)
        if li is not None:
            lens_ids = li
        timing["capture_s"] += time.time() - t1
        print(f"captured layers {keep[0]}-{keep[-1]} of {n_layers} ({time.time() - t1:.0f} s)", flush=True)
        for layer in keep:
            t1 = time.time()
            r = fit_probes(feats.pop(layer), y, fit_idx, sel_idx, eval_idx, wds=wds, steps=args.fit_steps,
                           lr=args.lr, device=dev)
            for n in eval_idx:
                acc[n][layer] = r["acc"][n]
                acc_in[n][layer] = r["acc_in"][n]
            wd_l[layer], sel_l[layer] = r["wd"], r["select_acc"]
            timing["fit_s"] += time.time() - t1
            best = r["acc"]["heldout"][:, :, FEATURES.index("value50")]
            print(f"layer {layer}: probes fitted ({time.time() - t1:.1f} s); value50 heldout max "
                  f"{float(torch.nan_to_num(best, nan=0).max()):.3f}", flush=True)
        del feats
        if dev == "cuda":
            torch.cuda.empty_cache()

    # ---- results
    nan2none = lambda xs: [None if v != v else round(float(v), 5) for v in xs]
    probe = {}
    for p, pos in enumerate(positions):
        for t, tn in enumerate(targets):
            for f, fn in enumerate(FEATURES):
                probe.setdefault(pos, {}).setdefault(tn, {})[fn] = {
                    "acc": {n: nan2none(acc[n][:, p, t, f].tolist()) for n in eval_idx},
                    "majority": {n: (None if maj[n][t, f] != maj[n][t, f] else round(float(maj[n][t, f]), 5))
                                 for n in eval_idx},
                    "n": {n: int((y[eval_idx[n], t, f] >= 0).sum()) for n in eval_idx},
                    "n_fit": int((y[fit_idx, t, f] >= 0).sum()),
                    "wd": [float(v) for v in wd_l[:, p, t, f]], "select_acc": nan2none(sel_l[:, p, t, f].tolist())}
            vf = FEATURES.index("value50")
            probe[pos][tn]["value50"].update(
                acc_in={n: nan2none(acc_in[n][:, p, t].tolist()) for n in eval_idx},
                majority_in={n: nan2none([maj_in[n][t]])[0] for n in eval_idx},
                n_in={n: int(((y[eval_idx[n], t, vf] >= 0) & (y[eval_idx[n], t, vf] < TOP_VALUES)).sum())
                      for n in eval_idx})
    res = {"model": args.model, "spec": spec.as_dict() if spec is not None else None, "spec_source": src,
           "n_layers": n_layers, "slot_probe_layer": spec.layer + 1 if spec is not None and spec.layer is not None
           else None, "positions": positions, "targets": targets, "features": list(FEATURES),
           "n_classes": N_CLASSES, "items": counts, "skipped": dict(skipped), "sets": list(eval_idx),
           "value50_vocab": {t: list(v) for t, v in vocab.items()},
           "fit": {"wd": wds, "steps": args.fit_steps, "lr": args.lr, "optimizer": "adam full-batch float32",
                   "standardise": "fit rows mean / std", "select_frac": args.select_frac, "seed": args.seed,
                   "train": args.train, "train_n": args.train_n, "heldout": args.heldout},
           "probe": probe, "args": vars(args), "device": dev}
    if lens_ids is not None:
        res["lens"] = lens_scores(tok, spec, lens_ids, values, targets, train_idx, eval_idx)
    timing["total_s"] = time.time() - t0
    if dev == "cuda":
        timing["peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    res["timing"] = {k: round(v, 2) for k, v in timing.items()}
    (out / "probe_results.json").write_text(json.dumps(res))
    txt = table_text(res, control)
    (out / "probe_table.txt").write_text(txt)
    print(txt, end="", flush=True)
    if args.png:
        heatmap(res, out / "probe_heatmap.png")
    print("probe timing", json.dumps(res["timing"]), flush=True)


if __name__ == "__main__":
    main()
