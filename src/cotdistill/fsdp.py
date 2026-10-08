"""FSDP2 data parallelism for scripts/train_student.py --fsdp (one rank per GPU, launched by torchrun).

Design notes: notes/next_round/scale12b/fsdp2.md. In short:
  - every decoder layer is its own fully_shard unit; the root is Readout, which holds the HF model and runs the
    unchanged student.label_logits inside its forward, so the root's parameters (tied embedding / head, final norm)
    are unsharded while the letter rows are read. Modules with only frozen parameters (vision towers) are separate
    units that never run, so they are never all-gathered.
  - gradients are SUMMED over the ranks (gradient divide factor 1, sum reductions): the ranks process disjoint parts
    of one update, each sequence with the single-GPU loss scale, so the summed gradient is the single-GPU update's.
  - split_update cuts one update's sequences into micro-batches (micro_bs chunks or token-budget batches, exactly as
    student.micro_batches) and deals them to the ranks with equal counts (FSDP collectives need the same number of
    forward/backward calls on every rank).
  - two optimizer setups (ShardedOptim): "8bit-sr" = torchao AdamW8bit with bf16 stochastic rounding on plain-tensor
    aliases of the local shards (GPU); "cpu-adamw" = ShardCPUAdamW, fp32 master + AdamW of the local shard on the CPU
    with one gradient D2H and one weight H2D per update.
  - Readout.batched splits evaluation batches across the ranks and gathers the logits in the original order.
"""
from __future__ import annotations

import collections
import contextlib
import dataclasses
import datetime
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist

_HYPER = ("lr", "betas", "eps", "weight_decay")
GIB = 2 ** 30


@dataclass
class Dist:
    rank: int
    world: int
    local_rank: int
    device: torch.device
    mesh: object

    def barrier(self) -> None:
        if self.device.type == "cuda":
            dist.barrier(device_ids=[self.device.index])
        else:
            dist.barrier()


def init(timeout_min: float = 120.0, quiet: bool = True) -> Dist:
    """Process group + 1-D device mesh from the torchrun environment (RANK, WORLD_SIZE, LOCAL_RANK). CUDA: rank r uses
    cuda:LOCAL_RANK when the process sees several devices (CUDA_VISIBLE_DEVICES=0,1 as Slurm sets it for a 2-slice
    job), else cuda:0 (one MIG UUID per process); NCCL. Without CUDA: CPU and gloo (tests). quiet: ranks > 0 write
    nothing to stdout (stderr stays), so logs and json rows come from rank 0 only."""
    from torch.distributed.device_mesh import init_device_mesh
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise SystemExit("--fsdp: launch with torchrun (RANK / WORLD_SIZE missing), e.g. "
                         "python -m torch.distributed.run --standalone --nproc_per_node 2 scripts/train_student.py ...")
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    timeout = datetime.timedelta(minutes=timeout_min)
    if torch.cuda.is_available():
        dev = torch.device("cuda", local if torch.cuda.device_count() > local else 0)
        torch.cuda.set_device(dev)
        dist.init_process_group("nccl", timeout=timeout, device_id=dev)
    else:
        dev = torch.device("cpu")
        dist.init_process_group("gloo", timeout=timeout)
    mesh = init_device_mesh(dev.type, (world,))
    if quiet and rank != 0:
        sys.stdout = open(os.devnull, "w")
    return Dist(rank, world, local, dev, mesh)


def _local(t):
    """The local shard of a DTensor (a plain tensor sharing its storage), or t itself."""
    return t.to_local() if hasattr(t, "to_local") else t


# ---------------------------------------------------------------------------------------------------- model sharding
class _Call:
    """Opaque argument holder: FSDP's root pre-forward rebuilds lists and dicts it finds in the inputs (moving tensors
    to the device), which would detach hidden_out and the label-id cache from the caller's objects."""
    __slots__ = ("args",)

    def __init__(self, *args):
        self.args = args


class Readout(torch.nn.Module):
    """FSDP2 root around the HF model: forward = student.label_logits on the wrapped model (text decoder, letter rows
    of the output embedding, soft-cap), so the root unit's parameters are unsharded while they are read.
    student.label_logits dispatches to readout() when it is given this module (attribute cotd_readout)."""
    cotd_readout = True

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, call: _Call):
        from .student import label_logits
        return label_logits(self.model, *call.args)

    def readout(self, tokenizer, batch, max_len, cache, hidden_out=None, layers=None):
        return self(_Call(tokenizer, batch, max_len, cache, hidden_out, layers))

    @torch.no_grad()
    def batched(self, tokenizer, exs: list, order: list[int], bs: int, max_len: int, cache: dict,
                want_hidden: bool = False, layers=None) -> tuple[list, list]:
        """Letter logits of every example (index = position in exs), from batches of bs consecutive entries of order.
        Batch j runs on rank j mod world; every rank runs the same number of forwards (a rank short of a batch in the
        last round repeats the first, shortest batch and drops the result). The ranks' results are gathered with
        all_gather_object, so every rank returns the full lists (CPU float32 logits, and hidden states if wanted)."""
        rank, world = dist.get_rank(), dist.get_world_size()
        dev = next(self.parameters()).device
        batches = [order[s:s + bs] for s in range(0, len(order), bs)]
        rounds = -(-len(batches) // world)
        mine_z, mine_h = {}, {}
        for k in range(rounds):
            j = k * world + rank
            idx = batches[j] if j < len(batches) else batches[0]
            hs = [] if want_hidden else None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                zs = self.readout(tokenizer, [exs[i] for i in idx], max_len, cache, hs, layers)
            if j < len(batches):
                for t, i in enumerate(idx):
                    mine_z[i] = zs[t].detach().float().cpu()
                    if hs is not None:
                        mine_h[i] = hs[t]
        parts = [None] * world
        dist.all_gather_object(parts, (mine_z, mine_h))
        zs, hs = [None] * len(exs), [None] * len(exs)
        for pz, ph in parts:
            for i, z in pz.items():
                zs[i] = z
            for i, h in ph.items():
                hs[i] = h
        return zs, hs


def decoder_layers(body) -> list:
    """The outermost GradientCheckpointingLayer modules of the text decoder (its decoder layers), else body.layers."""
    found = []
    try:
        from transformers.modeling_layers import GradientCheckpointingLayer
        found = [m for m in body.modules() if isinstance(m, GradientCheckpointingLayer)]
    except ImportError:
        pass
    inner = {id(c) for m in found for c in m.modules() if c is not m}
    found = [m for m in found if id(m) not in inner]
    if not found and hasattr(body, "layers"):
        found = list(body.layers)
    if not found:
        raise SystemExit("--fsdp: no decoder layers found in the text decoder")
    return found


def frozen_modules(model, body) -> list:
    """Outermost submodules outside the text decoder whose parameters are all frozen (vision / audio towers)."""
    body_ids = {id(m) for m in body.modules()}
    out, taken = [], []
    for name, m in model.named_modules():
        if m is model or id(m) in body_ids or any(name.startswith(t + ".") for t in taken):
            continue
        ps = list(m.parameters())
        if ps and not any(p.requires_grad for p in ps) and not any(id(c) in body_ids for c in m.modules()):
            out.append(m)
            taken.append(name)
    return out


def shard_model(model, mesh, reshard_after_forward: bool = True) -> Readout:
    """fully_shard every decoder layer and every frozen tower, then the Readout root; sum (not average) gradients."""
    from torch.distributed.fsdp import FSDPModule, fully_shard
    from .student import text_parts
    body, _ = text_parts(model)
    for m in frozen_modules(model, body):
        fully_shard(m, mesh=mesh)
    for layer in decoder_layers(body):
        fully_shard(layer, mesh=mesh, reshard_after_forward=reshard_after_forward)
    root = Readout(model)
    fully_shard(root, mesh=mesh)
    for m in root.modules():
        if isinstance(m, FSDPModule):
            m.set_gradient_divide_factor(1.0)
            if hasattr(m, "set_force_sum_reduction_for_comms"):
                m.set_force_sum_reduction_for_comms(True)   # plain SUM (gloo has no premul-sum / AVG)
    return root


# ---------------------------------------------------------------------------------------------------- update split
def split_update(tokenizer, upd: list, max_len: int, micro_bs: int, token_budget: int, max_seqs: int,
                 world: int) -> list[list[tuple[list, bool, int]]]:
    """One update's sequences as micro-batches per rank: [(micro-batch, is_pad, padded tokens), ...] for each rank,
    the same count on every rank. Micro-batches are those of student.micro_batches (micro_bs consecutive chunks, or
    length-sorted token-budget batches; lengths as collate truncates them). When their number is not a multiple of
    world, the micro-batch with the most padded tokens among those with >= 2 sequences is cut in two (the gradient
    is a sum over sequences, so the split does not change it); if every micro-batch is a single sequence, the short
    ranks get a pad micro-batch (the update's shortest sequence at weight 0: zero loss and zero gradient). Dealing:
    by padded tokens, largest first, in snake order over the ranks (0, 1, 1, 0, 0, 1, ...)."""
    from .student import special_prefix, text_ids, token_budget_batches
    n_pre = len(special_prefix(tokenizer))
    lens = [min(max_len, n_pre + len(text_ids(tokenizer, e.text))) for e in upd]
    if token_budget:
        groups = token_budget_batches(lens, token_budget, max_seqs)
    else:
        groups = [list(range(i, min(i + micro_bs, len(upd)))) for i in range(0, len(upd), micro_bs)]

    def padded(g):
        return len(g) * max(lens[i] for i in g)
    while len(groups) % world:
        cand = [k for k, g in enumerate(groups) if len(g) > 1]
        if not cand:
            break
        k = max(cand, key=lambda k: padded(groups[k]))
        g = groups.pop(k)
        h = len(g) // 2
        groups[k:k] = [g[:h], g[h:]]
    order = sorted(range(len(groups)), key=lambda k: -padded(groups[k]))
    out: list[list] = [[] for _ in range(world)]
    for pos, k in enumerate(order):
        rnd, j = divmod(pos, world)
        r = j if rnd % 2 == 0 else world - 1 - j
        out[r].append(([upd[i] for i in groups[k]], False, padded(groups[k])))
    n = max(len(o) for o in out)
    short = min(range(len(upd)), key=lambda i: lens[i])
    for o in out:
        while len(o) < n:
            o.append(([dataclasses.replace(upd[short], weight=0.0)], True, lens[short]))
    return out


# ---------------------------------------------------------------------------------------------------- clipping
@torch.no_grad()
def sharded_grad_norm(params, device) -> torch.Tensor:
    """L2 norm of the full gradient (fp32 0-dim tensor on device): per-shard norms (torch._foreach_norm, as
    clip_grad_norm_), squared and summed in fp32 over the local shards, all-reduced over the ranks."""
    grads = [g for g in (_local(p.grad) for p in params if p.grad is not None) if g.numel()]
    if grads:
        sq = torch.stack([n.float() for n in torch._foreach_norm(grads, 2.0)]).pow(2).sum()
    else:
        sq = torch.zeros((), dtype=torch.float32, device=device)
    sq = sq.to(device)
    if dist.is_initialized():
        dist.all_reduce(sq)
    return sq.sqrt()


# ---------------------------------------------------------------------------------------------------- (a) 8bit-sr
def adamw8bit_sr(params, lr: float):
    """torchao AdamW8bit with bf16 stochastic rounding (the single-GPU --optim adamw8bit-sr optimizer)."""
    try:
        from torchao.optim import AdamW8bit
    except ImportError as e:
        try:                         # torchao < 0.11 keeps the low-bit optimizers under prototype
            from torchao.prototype.low_bit_optim import AdamW8bit
        except ImportError:
            raise SystemExit(f"--fsdp-optim 8bit-sr needs torchao (pip install --no-deps torchao): {e}")
    return AdamW8bit(params, lr=lr, weight_decay=0.0, bf16_stochastic_round=True)


class LocalShards:
    """Plain-tensor aliases of this rank's parameter shards (same storage) and, at step time, of their gradient
    shards, so a single-device optimizer (torchao) runs on exactly the tensors it runs on without FSDP."""

    def __init__(self, params):
        self.params = list(params)
        with torch.no_grad():
            self.alias = [torch.nn.Parameter(_local(p), requires_grad=False) for p in self.params]
        self.live = [a for a in self.alias if a.numel() > 0]       # the optimizer's parameters

    @torch.no_grad()
    def bind(self) -> None:
        for a, p in zip(self.alias, self.params):
            loc = _local(p)
            if a.data_ptr() != loc.data_ptr() or a.shape != loc.shape:   # FSDP re-registered the shard
                a.data = loc
            a.grad = None if p.grad is None else _local(p.grad)

    def unbind(self) -> None:
        for a in self.alias:
            a.grad = None


# ---------------------------------------------------------------------------------------------------- (b) cpu-adamw
class ShardCPUAdamW(torch.optim.Optimizer):
    """fp32 master weights + AdamW of this rank's shards on the CPU, bf16 shards and gradients on the device.

    The device keeps FSDP's bf16 sharded gradient (accumulated over the update's micro-batches); step() moves it to
    the CPU once per update, segment by segment (at most seg_numel elements, through two pinned bf16 buffers): D2H
    of segment k + 1 overlaps the CPU work on segment k (bf16 -> fp32 gradient, clip coefficient, torch AdamW
    on the masters (fused CPU kernel when available), masters -> bf16), and the H2D of segment k overlaps segment
    k + 1. The default stream waits for the last H2D, so the next forward's all-gathers read the new weights.
    Host memory: 12 B/param (master, m, v) + two pinned staging buffers + one fp32 segment buffer.
    Parameters are FSDP2 DTensors (their local shards are used) or plain tensors (world size 1, tests)."""

    def __init__(self, params, lr: float = 1e-5, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0,
                 *, seg_numel: int = 2 ** 26, impl: str = "auto"):
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay))
        from .offload import _probe_impl
        self.impl = _probe_impl(impl)
        self._params, self._group_of = [], []
        for gi, g in enumerate(self.param_groups):
            for p in g["params"]:
                self._params.append(p)
                self._group_of.append(gi)
        if len({id(p) for p in self._params}) != len(self._params):
            raise ValueError("ShardCPUAdamW: a parameter appears twice")
        locs = [_local(p) for p in self._params]
        if len({t.dtype for t in locs}) != 1 or len({t.device for t in locs}) != 1:
            raise ValueError("ShardCPUAdamW: parameters must share one dtype and one device")
        if not all(t.is_contiguous() for t in locs):
            raise ValueError("ShardCPUAdamW: non-contiguous local shard")
        self.device = locs[0].device
        self.cuda = self.device.type == "cuda"
        self.master: list[torch.Tensor] = []
        with torch.no_grad():
            for t in locs:
                m = torch.empty(t.numel(), dtype=torch.float32)
                m.copy_(t.detach().reshape(-1).cpu())
                self.master.append(m)
        segs = [(i, s, min(m.numel(), s + seg_numel)) for i, m in enumerate(self.master)
                for s in range(0, m.numel(), seg_numel)]
        buckets, cur, n = [], [], 0
        for sg in segs:
            ln = sg[2] - sg[1]
            if cur and (n + ln > seg_numel or self._group_of[sg[0]] != self._group_of[cur[-1][0]]):
                buckets.append(cur)
                cur, n = [], 0
            cur.append(sg)
            n += ln
        if cur:
            buckets.append(cur)
        self._buckets = []
        for bsegs in buckets:
            views = [self.master[i][s:e] for i, s, e in bsegs]
            groups = collections.OrderedDict()
            for (i, _, _), v in zip(bsegs, views):
                groups.setdefault(self._group_of[i], []).append(v)
            inner = torch.optim.AdamW([{"params": ps, "_outer": gi, **{k: self.param_groups[gi][k] for k in _HYPER}}
                                       for gi, ps in groups.items()],
                                      fused=self.impl == "fused", foreach=self.impl == "foreach")
            self._buckets.append((bsegs, views, inner))
        nbuf = max((sum(e - s for _, s, e in b) for b in buckets), default=0)
        self.stage = [torch.empty(nbuf, dtype=locs[0].dtype, pin_memory=self.cuda) for _ in range(2)]
        self.gbuf = torch.empty(nbuf, dtype=torch.float32)
        self.stream = torch.cuda.Stream(device=self.device) if self.cuda else None
        self._coef: torch.Tensor | None = None
        self.timing: dict = {}

    def set_clip(self, coef: torch.Tensor | None) -> None:
        """Clip coefficient (< 1) for the next step, applied to the fp32 CPU gradient."""
        self._coef = None if coef is None else coef.detach().float().cpu()

    def _stream_ctx(self):
        return torch.cuda.stream(self.stream) if self.cuda else contextlib.nullcontext()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        t0 = time.time()
        has = [p.grad is not None for p in self._params]
        if self.cuda:
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
        grads = [_local(p.grad).reshape(-1) if h else None for p, h in zip(self._params, has)]
        locs = [_local(p).reshape(-1) for p in self._params]

        def fetch(b):
            bsegs = self._buckets[b][0]
            buf, off = self.stage[b % 2], 0
            with self._stream_ctx():
                for i, s, e in bsegs:
                    if has[i]:
                        buf[off:off + e - s].copy_(grads[i][s:e], non_blocking=self.cuda)
                    off += e - s
                if self.cuda:
                    ev = torch.cuda.Event()
                    ev.record(self.stream)
                    return ev
            return None
        pending = {b: fetch(b) for b in range(min(2, len(self._buckets)))}
        wait = 0.0
        for b, (bsegs, views, inner) in enumerate(self._buckets):
            tw = time.time()
            ev = pending.pop(b)
            if ev is not None:
                ev.synchronize()
            wait += time.time() - tw
            buf = self.stage[b % 2]
            n = sum(e - s for _, s, e in bsegs)
            g = self.gbuf[:n]
            g.copy_(buf[:n])                                   # bf16 -> fp32
            if self._coef is not None:
                g.mul_(self._coef)
            off, live = 0, False
            for (i, s, e), v in zip(bsegs, views):
                v.grad = g[off:off + e - s] if has[i] else None   # params without a gradient are skipped, like torch
                live = live or has[i]
                off += e - s
            if live:
                for ig in inner.param_groups:
                    og = self.param_groups[ig["_outer"]]
                    for k in _HYPER:
                        ig[k] = og[k]
                inner.step()
            for v in views:
                v.grad = None
            off = 0
            for (i, s, e), v in zip(bsegs, views):
                if has[i]:
                    buf[off:off + e - s].copy_(v)               # fp32 -> bf16 (round to nearest)
                off += e - s
            with self._stream_ctx():
                off = 0
                for i, s, e in bsegs:
                    if has[i]:
                        locs[i][s:e].copy_(buf[off:off + e - s], non_blocking=self.cuda)
                    off += e - s
            if b + 2 < len(self._buckets):          # stream-ordered after this H2D, which still reads buf
                pending[b + 2] = fetch(b + 2)
        if self.cuda:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        self._coef = None
        self.timing = {"step_s": time.time() - t0, "d2h_wait_s": wait}
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self._params:
            p.grad = None

    def reset_state(self) -> None:
        """Forget the AdamW moments and step counts (masters and weights stay)."""
        for _, _, inner in self._buckets:
            inner.state.clear()

    def release(self) -> None:
        """Free the CPU state (end of training)."""
        for _, _, inner in self._buckets:
            inner.state.clear()
        self._buckets, self.master, self.stage, self.gbuf = [], [], [], torch.empty(0)
        if self.cuda and hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()

    def host_bytes(self) -> int:
        n = sum(m.numel() * 4 for m in self.master) + sum(s.numel() * s.element_size() for s in self.stage)
        n += self.gbuf.numel() * 4
        for _, _, inner in self._buckets:
            for st in inner.state.values():
                n += sum(t.numel() * t.element_size() for t in st.values() if torch.is_tensor(t) and t.dim() > 0)
        return n

    def state_dict(self):
        raise NotImplementedError("ShardCPUAdamW does not checkpoint: 12 B/param (143 GB at 12B) does not fit the quota")


class ShardedOptim:
    """The optimizer side of --fsdp: global-norm clipping over the shards and one of the two setups.
    opt is what the LR scheduler drives (torchao AdamW8bit over the shard aliases, or ShardCPUAdamW)."""

    def __init__(self, params, kind: str, lr: float, device):
        self.params, self.kind, self.device = list(params), kind, device
        self.shards = None
        if kind == "8bit-sr":
            self.shards = LocalShards(self.params)
            self.opt = adamw8bit_sr(self.shards.live, lr)
        elif kind == "cpu-adamw":
            self.opt = ShardCPUAdamW(self.params, lr=lr, weight_decay=0.0)
        else:
            raise ValueError(f"unknown --fsdp-optim {kind}")
        self.timing: dict = {}

    @torch.no_grad()
    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        """Total gradient norm before clipping (as torch.nn.utils.clip_grad_norm_ returns it); scales the gradient by
        min(1, max_norm / (norm + 1e-6))."""
        total = sharded_grad_norm(self.params, self.device)
        coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
        if float(coef) < 1.0:
            if self.kind == "cpu-adamw":
                self.opt.set_clip(coef)
            else:
                grads = [g for g in (_local(p.grad) for p in self.params if p.grad is not None) if g.numel()]
                for dt in {g.dtype for g in grads}:
                    torch._foreach_mul_([g for g in grads if g.dtype == dt], coef.to(dt))
        return total

    def step(self) -> None:
        t0 = time.time()
        if self.shards is not None:
            self.shards.bind()
            self.opt.step()
            self.shards.unbind()
        else:
            self.opt.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.timing = dict(getattr(self.opt, "timing", {}) or {}, opt_step_s=time.time() - t0)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def reset_state(self) -> None:
        if self.kind == "cpu-adamw":
            self.opt.reset_state()
        else:
            self.opt.state.clear()

    def release(self) -> None:
        if self.kind == "cpu-adamw":
            self.opt.release()
        else:
            self.opt.state.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------- state dicts
def save_sharded(root, path) -> None:
    """Sharded model state (torch.distributed.checkpoint; every rank writes its shards)."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    dcp.save({"model": get_model_state_dict(root)}, checkpoint_id=str(path))


def load_sharded(root, path) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
    sd = {"model": get_model_state_dict(root)}
    dcp.load(sd, checkpoint_id=str(path))
    set_model_state_dict(root, sd["model"])


def full_state_dict(root) -> dict:
    """The full model state dict of the wrapped HF model (keys without the Readout prefix) on rank 0's CPU; other ranks
    get {}. Collective: every rank must call it."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    sd = get_model_state_dict(root, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    return {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}


# ---------------------------------------------------------------------------------------------------- stats
def host_rss_gib() -> tuple[float, float]:
    """(current, peak) resident set of this process in GiB."""
    cur = peak = 0.0
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                cur = int(line.split()[1]) / 2 ** 20
            elif line.startswith("VmHWM:"):
                peak = int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return cur, peak


def rank_stats(device) -> dict:
    """Per-rank peak device memory (allocated / reserved GiB) and host RSS (GiB), gathered from every rank."""
    cuda = device.type == "cuda"
    cur, peak = host_rss_gib()
    v = torch.tensor([torch.cuda.max_memory_allocated(device) / GIB if cuda else 0.0,
                      torch.cuda.max_memory_reserved(device) / GIB if cuda else 0.0, cur, peak],
                     dtype=torch.float64, device=device)
    out = [torch.zeros_like(v) for _ in range(dist.get_world_size())]
    dist.all_gather(out, v)
    rows = [x.tolist() for x in out]
    return {"peak_gib_ranks": [round(r[0], 2) for r in rows], "reserved_gib_ranks": [round(r[1], 2) for r in rows],
            "rss_gib_ranks": [round(r[2], 1) for r in rows], "rss_peak_gib_ranks": [round(r[3], 1) for r in rows]}


__all__ = ["Dist", "init", "Readout", "shard_model", "split_update", "sharded_grad_norm", "LocalShards",
           "ShardCPUAdamW", "ShardedOptim", "save_sharded", "load_sharded", "full_state_dict", "rank_stats"]
