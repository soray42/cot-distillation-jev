"""CPU-offloaded AdamW with fp32 master weights for bf16 (or fp32) GPU parameters.

Layout per trainable parameter p (on the GPU, usually bf16):
  master  fp32 CPU copy of p (the weights the optimizer really updates)
  acc     fp32 CPU gradient accumulator (= master.grad; summed over gradient-accumulation micro-steps)
  m, v    fp32 CPU AdamW moments (held by an inner torch.optim.AdamW over the master tensors)
  stage   CPU buffer in p's dtype, pinned when p is on CUDA: grad D2H target, then the weight H2D source

Backward: a post-accumulate-grad hook on p copies p.grad into stage on a side CUDA stream (non-blocking), lets the
allocator free p.grad once the copy is done (record_stream + p.grad = None), and hands the fp32 accumulation
(acc.copy_/add_(stage)) to a background CPU thread, so the GPU never holds a full gradient buffer and the CPU add
overlaps the rest of the backward. The number of gradient bytes in flight on the GPU is capped (max_inflight_bytes):
when PCIe falls behind, the hook waits, which throttles backward instead of letting grads pile up until an OOM.

step(): drain the pending adds, then bucket by bucket run torch's AdamW (fused CPU kernel when available, else
foreach, else for-loop) on the fp32 masters with num_threads intra-op threads, cast the bucket's masters into the
stage buffers and copy them H2D on the side stream, so the transfer of bucket k overlaps the CPU update of bucket
k + 1. The default stream then waits on the side stream (no host sync), so the next forward sees the new weights.

The training loop must replace torch.nn.utils.clip_grad_norm_(params, c) by opt.clip_grad_norm_(c): after the
hooks, p.grad is None on the GPU, so the torch call would silently see no gradients and clip nothing.

Works with torch.optim.lr_scheduler (this is an Optimizer subclass; param_groups hold the GPU params and 'lr').
state_dict() holds the masters, moments and step counts (the bf16 model weights are derivable from the masters).
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import warnings
from typing import Iterable

import torch

FORMAT = "cotdistill.offload.CPUOffloadAdamW/v1"
_HYPER = ("lr", "betas", "eps", "weight_decay")


def _probe_impl(impl: str) -> str:
    """The fastest torch AdamW implementation that runs on CPU fp32 tensors in this torch build."""
    order = {"auto": ["fused", "foreach", "for"], "fused": ["fused"], "foreach": ["foreach"], "for": ["for"]}[impl]
    for cand in order:
        p = torch.zeros(4, requires_grad=True)
        p.grad = torch.ones(4)
        kw = {"fused": cand == "fused", "foreach": cand == "foreach"}
        try:
            torch.optim.AdamW([p], lr=1e-3, **kw).step()
            return cand
        except (RuntimeError, ValueError, TypeError, NotImplementedError) as e:
            if impl != "auto":
                raise
            warnings.warn(f"CPU AdamW impl {cand!r} unavailable ({e}); trying the next one")
    raise RuntimeError("no CPU AdamW implementation works")


class CPUOffloadAdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable, lr: float = 1e-5, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0, *, num_threads: int | None = None,
                 impl: str = "auto", bucket_numel: int = 2 ** 27, max_inflight_bytes: int = 2 * 2 ** 30,
                 async_accumulate: bool = True):
        """bucket_numel: elements per CPU-update / H2D bucket (2**27 fp32 = 512 MiB of masters).
        max_inflight_bytes: GPU gradient bytes allowed between the hook and the end of their D2H copy.
        async_accumulate: run the fp32 accumulation in a background thread (False: inline in the hook)."""
        if not 0.0 <= lr:
            raise ValueError(f"invalid lr {lr}")
        defaults = dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.num_threads = num_threads
        self.impl = _probe_impl(impl)
        self.max_inflight_bytes = int(max_inflight_bytes)
        self._params: list[torch.Tensor] = []
        self._group_of: list[int] = []
        for gi, g in enumerate(self.param_groups):
            for p in g["params"]:
                if not p.requires_grad:
                    raise ValueError("CPUOffloadAdamW: every parameter must require grad")
                self._params.append(p)
                self._group_of.append(gi)
        if len({id(p) for p in self._params}) != len(self._params):
            raise ValueError("CPUOffloadAdamW: a parameter appears twice")
        self._index = {p: i for i, p in enumerate(self._params)}
        self.cuda = any(p.device.type == "cuda" for p in self._params)
        if self.cuda and any(p.device.type != "cuda" for p in self._params):
            raise ValueError("CPUOffloadAdamW: mix of CUDA and non-CUDA parameters")
        pin = self.cuda

        self.master: list[torch.Tensor] = []
        self.stage: list[torch.Tensor] = []
        for p in self._params:
            m = torch.empty(p.shape, dtype=torch.float32, device="cpu")
            st = torch.empty(p.shape, dtype=p.dtype, device="cpu", pin_memory=pin)
            st.copy_(p.detach())           # synchronous D2H (once)
            m.copy_(st)
            # the fp32 accumulator. torch.empty does not commit pages (Linux overcommit; the cgroup charges on first
            # touch), so a RAM shortfall shows up at the first backward / first step (moments), not here.
            m.grad = torch.empty_like(m)
            self.master.append(m)
            self.stage.append(st)
        self._has_grad = [False] * len(self._params)
        self._pending: list[cf.Future | None] = [None] * len(self._params)
        self._inflight: collections.deque = collections.deque()   # (event, nbytes)
        self._inflight_bytes = 0
        self._clip_coef: torch.Tensor | None = None
        self._h2d_done: torch.cuda.Event | None = None
        self._pool = cf.ThreadPoolExecutor(1, thread_name_prefix="offload-acc") if async_accumulate else None
        self.stream = torch.cuda.Stream(device=self._params[0].device) if self.cuda else None

        # buckets: consecutive params of one group up to bucket_numel elements; one inner AdamW per bucket
        self._buckets: list[tuple[list[int], torch.optim.AdamW]] = []
        cur: list[int] = []
        n = 0

        def flush():
            if cur:
                groups = collections.OrderedDict()
                for i in cur:
                    groups.setdefault(self._group_of[i], []).append(self.master[i])
                inner_groups = [{"params": ps, "_outer": gi, **{k: self.param_groups[gi][k] for k in _HYPER}}
                                for gi, ps in groups.items()]
                inner = torch.optim.AdamW(inner_groups, fused=self.impl == "fused", foreach=self.impl == "foreach")
                self._buckets.append((list(cur), inner))
        for i, p in enumerate(self._params):
            if cur and (n + p.numel() > bucket_numel or self._group_of[i] != self._group_of[cur[-1]]):
                flush()
                cur, n = [], 0
            cur.append(i)
            n += p.numel()
        flush()

        self._hooks = [p.register_post_accumulate_grad_hook(self._hook) for p in self._params]

    # ------------------------------------------------------------------ backward side
    def _accumulate(self, i: int, ev) -> None:
        if ev is not None:
            ev.synchronize()
        acc = self.master[i].grad
        if self._has_grad[i]:
            acc.add_(self.stage[i])        # bf16 stage promotes into the fp32 accumulator
        else:
            acc.copy_(self.stage[i])
            self._has_grad[i] = True

    def _wait(self, i: int) -> None:
        f = self._pending[i]
        if f is not None:
            self._pending[i] = None
            f.result()

    @torch.no_grad()
    def _hook(self, p: torch.Tensor) -> None:
        g = p.grad
        if g is None:
            return
        i = self._index[p]
        self._wait(i)                       # the previous micro-step's add must be done before stage is reused
        if self.cuda:
            self._sync_h2d()                # stage was last the source of the H2D weight copy
            self.stream.wait_stream(torch.cuda.current_stream(g.device))
            with torch.cuda.stream(self.stream):
                self.stage[i].copy_(g, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self.stream)
            g.record_stream(self.stream)
            p.grad = None
            nb = g.numel() * g.element_size()
            self._inflight.append((ev, nb))
            self._inflight_bytes += nb
            while self._inflight and (self._inflight_bytes > self.max_inflight_bytes or self._inflight[0][0].query()):
                e0, b0 = self._inflight.popleft()
                e0.synchronize()
                self._inflight_bytes -= b0
        else:
            self.stage[i].copy_(g)
            p.grad = None
            ev = None
        if self._pool is not None:
            self._pending[i] = self._pool.submit(self._accumulate, i, ev)
        else:
            self._accumulate(i, ev)

    def _sync_h2d(self) -> None:
        if self._h2d_done is not None:
            self._h2d_done.synchronize()
            self._h2d_done = None

    def _drain(self) -> None:
        for i in range(len(self._params)):
            self._wait(i)
        self._inflight.clear()
        self._inflight_bytes = 0

    # ------------------------------------------------------------------ optimizer API
    def _threads(self):
        old = torch.get_num_threads()
        if self.num_threads and self.num_threads != old:
            torch.set_num_threads(self.num_threads)
        return old

    @staticmethod
    def _restore_threads(old: int) -> None:
        if torch.get_num_threads() != old:
            torch.set_num_threads(old)

    @torch.no_grad()
    def clip_grad_norm_(self, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
        """torch.nn.utils.clip_grad_norm_ over the accumulated fp32 CPU gradients; returns the total norm (fp32 CPU
        tensor). The scaling is applied inside the next step(), before any moment update.
        A non-finite total poisons the fp32 state on the next step() exactly as torch's clip + AdamW would (here the
        coefficient is NaN, so no scaling, and the inf/NaN gradients reach the moments): a caller that wants to
        survive one bad micro-batch must check torch.isfinite(total) and zero_grad() instead of step()."""
        self._drain()
        old = self._threads()
        try:
            accs = [self.master[i].grad for i in range(len(self._params)) if self._has_grad[i]]
            if not accs:
                return torch.tensor(0.0)
            norms = torch._foreach_norm(accs, norm_type)
            total = torch.linalg.vector_norm(torch.stack(norms), norm_type)
        finally:
            self._restore_threads(old)
        coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)   # the fp32 arithmetic of torch's clip_grad_norm_
        self._clip_coef = coef if float(coef) < 1.0 else None
        return total

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._drain()
        self._sync_h2d()      # stage buffers are rewritten below on the CPU: the last H2D from them must be done
        old = self._threads()
        try:
            if self.cuda:     # the side stream must not overwrite weights the default stream may still be reading
                self.stream.wait_stream(torch.cuda.current_stream(self._params[0].device))
            for idx, inner in self._buckets:
                live = [i for i in idx if self._has_grad[i]]
                if not live:
                    continue
                for ig in inner.param_groups:
                    og = self.param_groups[ig["_outer"]]
                    for k in _HYPER:
                        ig[k] = og[k]
                saved = {}
                for i in idx:                     # params without a gradient this update are skipped, like torch
                    if not self._has_grad[i]:
                        saved[i] = self.master[i].grad
                        self.master[i].grad = None
                if self._clip_coef is not None:
                    torch._foreach_mul_([self.master[i].grad for i in live], self._clip_coef)
                inner.step()
                for i, g in saved.items():
                    self.master[i].grad = g
                for i in live:
                    p = self._params[i]
                    if self.cuda:
                        self.stage[i].copy_(self.master[i])     # fp32 -> bf16 (round to nearest) on the CPU
                        with torch.cuda.stream(self.stream):
                            p.data.copy_(self.stage[i], non_blocking=True)
                    else:
                        p.data.copy_(self.master[i])
            if self.cuda:
                self._h2d_done = torch.cuda.Event()
                self._h2d_done.record(self.stream)
                torch.cuda.current_stream(self._params[0].device).wait_stream(self.stream)
        finally:
            self._restore_threads(old)
            self._clip_coef = None
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Forget the accumulated gradients (the next micro-step overwrites the accumulators instead of adding)."""
        self._drain()
        for p in self._params:
            p.grad = None
        self._has_grad = [False] * len(self._params)
        self._clip_coef = None

    def grads(self) -> list[torch.Tensor | None]:
        """The accumulated fp32 CPU gradients (None where no gradient arrived since zero_grad)."""
        self._drain()
        return [self.master[i].grad if self._has_grad[i] else None for i in range(len(self._params))]

    def _inner_state(self, i: int) -> dict:
        for idx, inner in self._buckets:
            if i in idx:
                return inner.state[self.master[i]]
        raise KeyError(i)

    def state_dict(self, moments: bool = True) -> dict:
        """References (not copies) to the master weights and moments; torch.save streams them.
        moments=False: masters only (4 B/param instead of 12; 47.6 GB instead of 143 GB at 12B). Loading such a
        state restarts AdamW (m = v = 0, step 0, so bias correction restarts too): an approximate resume, for disk
        quotas that cannot hold the moments; re-warm the LR for a few dozen updates after it."""
        self._drain()
        groups = []
        for gi, g in enumerate(self.param_groups):
            d = {k: v for k, v in g.items() if k != "params"}
            d["params"] = [i for i in range(len(self._params)) if self._group_of[i] == gi]
            groups.append(d)
        state = {}
        for i in range(len(self._params)):
            st = self._inner_state(i)
            e = {"master": self.master[i]}
            if st and moments:
                e.update(step=float(st["step"]), exp_avg=st["exp_avg"], exp_avg_sq=st["exp_avg_sq"])
            state[i] = e
        return {"format": FORMAT, "param_groups": groups, "state": state}

    @torch.no_grad()
    def load_state_dict(self, sd: dict) -> None:
        """Copy saved masters and moments into the existing buffers (works with torch.load(..., mmap=True)) and
        refresh the device parameters from the masters."""
        if sd.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} state dict")
        self._drain()
        if len(sd["param_groups"]) != len(self.param_groups) or len(sd["state"]) != len(self._params):
            raise ValueError("state dict does not match the optimizer's parameters")
        for g, sg in zip(self.param_groups, sd["param_groups"]):
            if len(sg["params"]) != len(g["params"]):
                raise ValueError("param group sizes differ")
            for k, v in sg.items():
                if k != "params":
                    g[k] = tuple(v) if k == "betas" else v
        for i in range(len(self._params)):
            e = sd["state"][i]
            if tuple(e["master"].shape) != tuple(self.master[i].shape):
                raise ValueError(f"param {i}: shape {tuple(e['master'].shape)} != {tuple(self.master[i].shape)}")
            self.master[i].copy_(e["master"])
            st = self._inner_state(i)
            if "exp_avg" not in e:
                st.clear()
                continue
            for k in ("exp_avg", "exp_avg_sq"):        # reuse existing moment buffers (no transient 2x RAM)
                if k in st and st[k].shape == e[k].shape:
                    st[k].copy_(e[k])
                else:
                    st[k] = e[k].to(torch.float32).clone()
            st["step"] = torch.tensor(float(e["step"]), dtype=torch.float32)
        self.zero_grad()
        self._push_all()

    def _push_all(self) -> None:
        self._sync_h2d()
        for i, p in enumerate(self._params):
            if self.cuda:
                self.stage[i].copy_(self.master[i])
                p.data.copy_(self.stage[i])
            else:
                p.data.copy_(self.master[i])
        if self.cuda:
            torch.cuda.synchronize()

    def reset_state(self) -> None:
        """Forget the AdamW moments and step counts (masters and weights stay): what opt.state.clear() does for a
        plain torch optimizer. Here self.state is the unused base-class dict, so train_student's --reset-optim
        (opt.state.clear()) would be a silent no-op; call this instead."""
        self._drain()
        for _, inner in self._buckets:
            inner.state.clear()

    def release(self) -> None:
        """Free the CPU state (masters, moments, accumulators, stages) and remove the hooks; the device weights stay
        (use at the end of training, where the plain-AdamW loop calls opt.state.clear())."""
        self._drain()
        self._sync_h2d()
        for h in self._hooks:
            h.remove()
        self._hooks = []
        for _, inner in self._buckets:
            inner.state.clear()
        self._buckets = []
        self.master, self.stage = [], []
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self.cuda and hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()   # the freed stages stay in torch's pinned cache (about 27 GB at 12B) otherwise

    def host_bytes(self) -> int:
        """CPU bytes held: masters + accumulators + stages (+ moments once allocated). Pinned stages are counted at
        what torch's caching host allocator commits for them: each block rounded up to a power of two (+12% at 12B:
        26.6 GB for 23.8 GB of bf16)."""
        def stage_bytes(s):
            b = s.numel() * s.element_size()
            return 1 << (b - 1).bit_length() if s.is_pinned() and b > 0 else b
        n = sum(m.numel() * 8 + stage_bytes(s) for m, s in zip(self.master, self.stage))
        for _, inner in self._buckets:
            for st in inner.state.values():
                n += sum(t.numel() * t.element_size() for t in st.values() if torch.is_tensor(t) and t.dim() > 0)
        return n


def estimate(n_params: float, micro_steps: int, *, pcie_gbs: float = 20.0, cpu_gbs: float = 60.0,
             bytes_param: int = 2) -> dict:
    """Back-of-envelope per-update costs (GB, s) of CPUOffloadAdamW for n_params trainable parameters."""
    gb = 1e9
    d2h = micro_steps * n_params * bytes_param / gb
    h2d = n_params * bytes_param / gb
    # accumulation: first micro-step reads stage + writes acc; later ones read stage + acc, write acc
    acc = n_params * ((bytes_param + 4) + (micro_steps - 1) * (bytes_param + 8)) / gb
    clip = n_params * 4 / gb                       # norm pass (the scaling write is rolled into adam's read below)
    clip_mul = n_params * 8 / gb                   # explicit scaling pass when the clip is active
    adam = n_params * (4 * 4 + 3 * 4) / gb         # read master, grad, m, v; write master, m, v
    cast = n_params * (4 + bytes_param) / gb       # master -> stage before H2D
    host = n_params * (4 + 4 + 4 + 4 + bytes_param) / gb
    cpu_s = (acc + clip + clip_mul + adam + cast) / cpu_gbs
    return {"host_GB": host, "d2h_GB": d2h, "h2d_GB": h2d, "pcie_s": (d2h + h2d) / pcie_gbs,
            "cpu_acc_GB": acc, "cpu_step_GB": clip + clip_mul + adam + cast, "cpu_s": cpu_s,
            "cpu_step_s": (clip + clip_mul + adam + cast) / cpu_gbs, "cpu_acc_s": acc / cpu_gbs}


__all__ = ["CPUOffloadAdamW", "estimate"]
