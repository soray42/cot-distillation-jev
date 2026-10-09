"""One-step full-gradient subspaces of the answer-only and Tree Forcing losses vs the finished fine-tuning update.

Connects Tree Forcing with LoRA-One (Zhang, Liu, Chen, ICML 2025, arXiv 2502.01235): LoRA adapters trained from a small
init align with the top singular subspaces of the ONE-STEP full fine-tuning gradient G = -grad L(W0) (B with its left,
A with its right singular vectors), and a spectral init from G puts the adapter there at step 0. Question: does node
supervision (GFP-S) rotate G toward the subspace that the finished full fine-tuning update dW = W_ft - W_base ends up
using? That would explain why GFP-S "clicks" within one epoch while answer-only training (A2) needs 3-7.

Nothing is trained. On a fixed random batch of --n problems (seeded by --seed, the same problems for every loss) the
script builds the examples scripts/train_student.py builds and accumulates, in fp32 param.grad (one backward per
micro-batch, no optimizer step), the gradient of
  A2          --final teacher --subq none (plain loop): sum over the finals of w x CE / N
  <arm>_nodes the node views of --tree-full MODE --tree-view VIEW --tree-cap 0 --tree-sources SRC (full-tree loop;
              GFPS = plain/statements, TFL = true/local, ...; NODE_ARMS): sum over the node views of w x CE / (3 N)
  <arm>       the whole full-tree loss = A2 / 3 + <arm>_nodes (derived, not recomputed)
with the trainer's soft cross-entropy (kl_loss, --label-smoothing; the analysed runs used LSMOOTH=0.1, the trainer's
own default is 0) and weights (item weight; node views aux_total / views per problem, or --tree-node-weight). The
normalisation is the trainer's per-update one averaged over the N/32 updates the batch would fill: the plain loop
divides an update's sum by micro_bs x grad_accum = 32 sequences, the full-tree loop by 3 x items_per_update (32
problems), so every G is the expected one-update gradient and node / final norm ratios are those the optimizer sees.
build_full_tree's final view is build_examples' A2 final byte for byte (permute_final 0; asserted per problem), hence
G_<arm> = G_A2 / 3 + G_<arm>_nodes exactly; --check recomputes G_<arm> in the trainer's layout ([final] + nodes per
problem, one normalisation) and compares. Node examples come from a random.Random(--seed) walking the batch in order,
so lockstep arms (GFPS / GFP / GFPQ; TFL / TFLN) ask the same nodes with the same Yes/No orders.

Precision: --precision fp32master (default) = the analysed 2B runs: fp32 weights, bf16 autocast on CUDA, fp32 grads;
fp32 = no autocast. Gradient checkpointing is non-reentrant (reentrant checkpointing with frozen embeddings returns no
gradient). Only the selected matrices require grad (no 2 GB embedding gradient). On the CPU the Qwen3.5 Gated
DeltaNet layers run the modeling file's PyTorch reference kernels (fla's Triton kernels need a GPU; --reference-kernels
forces them on the GPU too).

Matrices: --layers every:4 = every 4th decoder layer starting at the first full-attention layer (Qwen3.5: layers 3, 7,
..., 23 of 24, the only ones with q/k/v/o; Qwen2: 0, 4, ...), --modules q/k/v/o + gate/up/down (Gated DeltaNet
projections in_proj_qkv / in_proj_z / in_proj_a / in_proj_b / out_proj can be added; a module absent from a layer is
skipped). Per matrix and r in --ranks (clipped to min(d_out, d_in)):
  - for every pair of gradients: mean cos^2 of the principal angles between their top-r left singular subspaces
    (||U_a^T U_b||_F^2 / r; LoRA-One's B side) and right ones (A side); 1 = same subspace, about r/d for random ones;
    flattened cosine;
  - node / final: ||G_<arm>_nodes|| / ||G_A2 / 3|| (the final part of the same loss) and cos(G_A2, G_<arm>_nodes);
  - top-r energy share sum_{i<=r} s_i^2 / ||G||_F^2 of every gradient (how low-rank it is) and its numerical rank;
  - with --finetuned NAME=dir (dW = W_ft - W_base, W_ft read through load_model as scripts/wise_ft_eval.py does; NAME
    = the arm whose run it is, optionally NAME-tag, e.g. GFPS, A2-c12): the same left / right similarities of every G
    with dW, the energy of dW captured by G's top-r pair ||U_r^T dW V_r||_F^2 / ||dW||_F^2 (at most dW's own top-r
    energy, also reported), cos(-G, dW) (dW is about -lr x sum of updates), ||dW|| / ||W||; and dW vs dW for every
    pair of finetuned models of one job.
Outputs: <out>/grad_subspace.json (per matrix + means by matrix type, attention / MLP, layer, all) and
<out>/grad_subspace.md (tables + the H1 / H2 readout). --report a.json b.json ... --out DIR merges jobs (same base,
data, batch and settings; the gradients are recomputed in each, their norms are compared as a reproducibility check),
e.g. dW_GFPS / dW_TFL from account A with dW_A2 from account B; G x dW similarities are only computed inside a job.

Memory (estimates; --store cpu keeps the accumulated gradients in host memory between passes):
  2B (Qwen3.5-2B-Base, 24 layers, 6 full attention; hidden 2048, MLP 6144, q_proj 2048->4096 with the output gate,
  k/v 2048->512): 6 selected layers x 52.4M = 315M params = 1.26 GB per fp32 gradient. GPU: fp32 weights 7.5-9.1 GB
  (text only / with the vision tower) + autocast bf16 weight copies about 4 GB + one param.grad set 1.26 GB +
  checkpointed activations at --token-budget 8192 about 2-3 GB = about 15-17 GB peak: one 4g.40gb slice. Host: the
  finals gradient + one per node arm (3 x 1.26 GB with GFPS,TFL), 0.63 GB of bf16 weights per finetuned model (a
  transient 4.5 GB full load each): under 10 GB of the job's 48 GB.
  4B (32 layers, 8 full attention; hidden 2560, MLP 9216, q_proj 2560->8192, k/v 2560->1024): 8 x 107.5M = 860M params
  = 3.44 GB per gradient. GPU: weights 16.8-18.6 GB + autocast copies about 8 GB + 3.44 GB grads + activations about
  4 GB = about 32-35 GB: one 80 GB A100 comfortably; on a 40 GB slice use --layer-groups 2 (each group repeats every
  pass, grads 1.7 GB) and --token-budget 4096. Host: 3 x 3.44 GB + 1.7 GB per finetuned model (+ a 9.3 GB transient).
Time (2B, 4g.40gb slice, about 10 sequences/s forward + backward from the GFPS-c4 training throughput): v3sl batch
--batch-sources tree, N 256 = 256 finals + about 11.2k node views per arm (43.8 nodes per K&K problem): about 20 min
per arm (TFL's local views are shorter); --batch-sources all (14% K&K): about 3 min per arm; SVDs (6144 x 2048 at
most) a few minutes; --check prints a measured estimate.

  python scripts/grad_subspace.py --model ~/cotd/models/Qwen3.5-2B-Base --train data/student_v3sl/train.jsonl \\
      --tree-sources knights_knaves --batch-sources tree --node-arms GFPS,TFL \\
      --finetuned GFPS=runs/GFPS-c4-.../model TFL=runs/TFL-c4-.../model --out runs/gradsub/v3sl-tree-n256-s0-GFPS+TFL
  python scripts/grad_subspace.py --report runs/gradsub/v3sl-tree-n256-s0-*/grad_subspace.json \\
      --out runs/gradsub/v3sl-tree-n256-s0
--check: the whole pipeline on --check-n problems from the tree sources and the first selected layer only, plus the
trainer-layout identity check, non-zero finite gradients, timings and the full run's estimate (<out>/check.json);
exits non-zero when a check fails (the sbatch smoke stage). The analysis is CPU-runnable on tiny models (tests).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.student import (build_examples, build_full_tree, kl_loss, label_logits, load_model,  # noqa: E402
                                micro_batches, text_parts)

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
LINATTN = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
TYPE_ORDER = ATTN + MLP + LINATTN
# Node arms as hpc/train_arm.sbatch's arm_config: --tree-full MODE --tree-view VIEW (--tree-cap and --tree-sources here)
NODE_ARMS = {"GFPS": ("plain", "statements"), "GFPQ": ("plain", "statements+question"), "GFP": ("plain", "problem"),
             "TFL": ("true", "local"), "TFLN": ("plain", "local"), "TFA": ("true", "ancestors"),
             "TFLM": ("matched", "local")}
SKIP_KEYS = ("name", "layer", "type", "shape")


def read_jsonl(p: str) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def md5(p: str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rd(x):
    """JSON leaf: 6 significant digits; None for undefined values (no NaN in the file)."""
    if x is None or not math.isfinite(x):
        return None
    return float(f"{x:.6g}")


# ---- examples -------------------------------------------------------------------------------------------------------

def final_example(item: dict, subq_target: str = "cot"):
    """The A2 final view of an item (train_student's epoch_examples with --final teacher --subq none), or None."""
    ex = [e for e in build_examples(item, final="teacher", subq="none", subq_target=subq_target, lambda_sub=0.0,
                                    rng=random.Random(0)) if e.kind == "final"]
    return ex[0] if ex else None


def pick_batch(items: list[dict], n: int, seed: int, tree_sources: set | None, batch_sources: str,
               subq_target: str = "cot") -> list[dict]:
    """N problems with a final view (all sources, or only --tree-sources), drawn with their own seeded generator."""
    pool = [it for it in items if final_example(it, subq_target) is not None]
    if batch_sources == "tree":
        pool = [it for it in pool if tree_sources is None or it.get("source") in tree_sources]
    if not pool:
        raise SystemExit(f"no item with a final view{' in the tree sources' if batch_sources == 'tree' else ''}")
    if n > len(pool):
        print(f"--n {n}: only {len(pool)} eligible problems, all used", flush=True)
    return random.Random(f"gradsub-{seed}").sample(pool, min(n, len(pool)))


def build_views(batch: list[dict], arms: list[str], *, seed: int, tree_sources: set | None, tree_cap: int = 0,
                aux_total: float = 1.0, node_weight: float = 0.0, subq_target: str = "cot"):
    """finals (one per problem, A2's), {arm: [node views of problem j]}, {arm: build_full_tree stats}. Each arm's
    groups come from build_full_tree exactly as the full-tree loop calls it (permute_final 0, final teacher); its final
    view must equal A2's (so G_arm = G_A2 / 3 + G_arm_nodes)."""
    finals = [final_example(it, subq_target) for it in batch]
    nodes, stats = {}, {}
    for arm in arms:
        mode, view = NODE_ARMS[arm]
        rng, st, per = random.Random(seed), {}, []
        for it, fin in zip(batch, finals):
            g = build_full_tree(it, mode=mode, rng=rng, cap=tree_cap, aux_total=aux_total, final="teacher",
                                subq_target=subq_target, view=view, sources=tree_sources, node_weight=node_weight,
                                stats=st)
            if not g or g[0] != fin or any(e.kind != "subq" for e in g[1:]):
                raise RuntimeError(f"{arm} {it.get('item_id')}: the full-tree group does not start with A2's final view")
            per.append(g[1:])
        nodes[arm], stats[arm] = per, st
    return finals, nodes, stats


def preflight(items: list[dict], arms: list[str], tree_sources: set | None) -> None:
    """train_student's view check: every tree-source item must build the statements views (fail before the GPU work)."""
    from cotdistill.treeplan import problem_no_options, statement_block
    for arm in arms:
        view = NODE_ARMS[arm][1]
        if view not in ("statements", "statements+question"):
            continue
        fn = problem_no_options if view == "statements+question" else statement_block
        for it in items:
            if tree_sources is None or it.get("source") in tree_sources:
                try:
                    fn(it)
                except ValueError as e:
                    raise SystemExit(f"--node-arms {arm} (--tree-view {view}): {e}")


# ---- matrices -------------------------------------------------------------------------------------------------------

def decoder(model):
    body, _ = text_parts(model)
    layers = getattr(body, "layers", None)
    if layers is None:
        raise SystemExit("no decoder layers found on the text backbone")
    return body, layers


def layer_types(model) -> list[str]:
    body, layers = decoder(model)
    cfg = getattr(body, "config", None) or model.config
    cfg = getattr(cfg, "text_config", None) or cfg
    t = getattr(cfg, "layer_types", None)
    return list(t) if t and len(t) == len(layers) else ["full_attention"] * len(layers)


def parse_layers(spec: str, types: list[str]) -> list[int]:
    """all | every:K[:OFFSET|auto] (auto = the first full-attention layer) | comma list of layer indices."""
    n = len(types)
    if spec == "all":
        return list(range(n))
    if spec.startswith("every:"):
        parts = spec.split(":")
        k = int(parts[1])
        if k < 1:
            raise SystemExit(f"--layers {spec}: K >= 1")
        off = (int(parts[2]) if len(parts) > 2 and parts[2] != "auto"
               else next((i for i, t in enumerate(types) if t == "full_attention"), 0))
        return list(range(off, n, k))
    out = sorted({int(x) for x in spec.split(",") if x.strip()})
    bad = [i for i in out if not 0 <= i < n]
    if bad or not out:
        raise SystemExit(f"--layers {spec}: indices must lie in 0..{n - 1}")
    return out


def select_matrices(model, layers: list[int], modules: list[str]) -> list[dict]:
    """The 2-D weights named in modules (last name component) inside the selected decoder layers, by layer then by the
    order of modules; each as {name (model.named_parameters), layer, type, shape, param}."""
    _, dec = decoder(model)
    pname = {id(p): n for n, p in model.named_parameters()}
    out = []
    for i in layers:
        found = []
        for sub, mod in dec[i].named_modules():
            t = sub.rsplit(".", 1)[-1]
            w = getattr(mod, "weight", None)
            if t in modules and isinstance(w, torch.nn.Parameter) and w.dim() == 2 and id(w) in pname:
                found.append({"name": pname[id(w)], "layer": i, "type": t, "shape": list(w.shape), "param": w})
        out += sorted(found, key=lambda m: (modules.index(m["type"]), m["name"]))
    if not out:
        raise SystemExit(f"no matrix of {modules} in layers {layers}")
    return out


# ---- gradients ------------------------------------------------------------------------------------------------------

def grad_pass(model, tok, exs: list, norm: float, mats: list[dict], *, max_len: int, micro_bs: int,
              token_budget: int, token_budget_seqs: int, label_smoothing: float, autocast: bool, store,
              cache: dict) -> tuple[dict, dict]:
    """sum over exs of w x CE / norm, accumulated in the selected params' fp32 .grad (one backward per micro-batch);
    returns {name: gradient on `store`} and the pass's counts, time and loss."""
    for m in mats:
        m["param"].grad = None
    t0, loss_sum = time.time(), 0.0
    mbs = micro_batches(tok, exs, max_len, micro_bs, token_budget, token_budget_seqs) if exs else []
    for mb in mbs:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            z = label_logits(model, tok, mb, max_len, cache)
            loss = kl_loss(z, mb, label_smoothing) * len(mb) / norm
        loss.backward()
        loss_sum += float(loss.detach())
    out = {}
    for m in mats:
        g = m["param"].grad
        out[m["name"]] = (torch.zeros(m["shape"], dtype=torch.float32, device=store) if g is None
                          else g.detach().float().to(store, copy=True))
        m["param"].grad = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, {"seqs": len(exs), "micro_batches": len(mbs), "s": time.time() - t0, "loss": loss_sum}


def layer_groups(mats: list[dict], k: int) -> list[list[dict]]:
    layers = sorted({m["layer"] for m in mats})
    k = max(1, min(k, len(layers)))
    per = math.ceil(len(layers) / k)
    chunks = [set(layers[i:i + per]) for i in range(0, len(layers), per)]
    return [[m for m in mats if m["layer"] in c] for c in chunks]


def gradients(model, tok, finals: list, nodes: dict, mats: list[dict], args, *, autocast: bool, store) -> tuple:
    """{"final": G_A2, arm: G_arm_nodes} (each {name: tensor}) and per-pass info. With --layer-groups K every pass is
    repeated once per group of layers (only that group requires grad)."""
    n = len(finals)
    kw = dict(max_len=args.max_len, micro_bs=args.micro_bs, token_budget=args.token_budget,
              token_budget_seqs=args.token_budget_seqs, label_smoothing=args.label_smoothing, autocast=autocast,
              store=store, cache={})
    G = {"final": {}, **{arm: {} for arm in nodes}}
    info = collections.defaultdict(list)
    flat = {arm: [e for per in v for e in per] for arm, v in nodes.items()}
    for gi, group in enumerate(layer_groups(mats, args.layer_groups)):
        ids = {id(m["param"]) for m in group}
        for m in mats:
            m["param"].requires_grad_(id(m["param"]) in ids)
        for key, exs, norm in [("final", finals, n)] + [(arm, flat[arm], 3 * n) for arm in nodes]:
            g, inf = grad_pass(model, tok, exs, norm, group, **kw)
            G[key].update(g)
            info[key].append(inf)
            print(f"group {gi} {key}: {inf['seqs']} sequences, {inf['micro_batches']} micro-batches, {inf['s']:.1f} s, "
                  f"loss {inf['loss']:.4f}", flush=True)
    return G, dict(info)


def direct_full_tree(model, tok, finals: list, per_problem: list[list], mats: list[dict], args, *, autocast: bool,
                     store) -> dict:
    """The full-tree loss in the trainer's layout ([final] + node views per problem, consecutive micro-batches, one
    normalisation 3 N): G_arm without the split (--check)."""
    exs = [e for fin, per in zip(finals, per_problem) for e in [fin] + per]
    for m in mats:
        m["param"].requires_grad_(True)
    g, _ = grad_pass(model, tok, exs, 3 * len(finals), mats, max_len=args.max_len, micro_bs=args.micro_bs,
                     token_budget=0, token_budget_seqs=args.token_budget_seqs, label_smoothing=args.label_smoothing,
                     autocast=autocast, store=store, cache={})
    return g


def canon(name: str) -> str | None:
    """'layers.N.<rest>' of a text-decoder parameter name (None for vision / MTP / audio towers)."""
    m = re.search(r"(?:^|\.)(layers\.\d+\..+)$", name)
    if not m or re.search(r"visual|vision|mtp|audio", name[:m.start(1)]):
        return None
    return m.group(1)


def load_finetuned(path: str, mats: list[dict]) -> tuple[dict, dict]:
    """{name: weight on the CPU, in the checkpoint's dtype (bf16 for train_student --save)} of the selected matrices of
    a fine-tuned model directory (load_model, the class the base loads with, so the names match as in
    scripts/wise_ft_eval.py; else matched by their decoder suffix)."""
    ft = load_model(path, "auto")
    sd = ft.state_dict()
    by_canon = collections.defaultdict(list)
    for k in sd:
        c = canon(k)
        if c is not None:
            by_canon[c].append(k)
    out, keys = {}, {}
    for m in mats:
        k = m["name"] if m["name"] in sd else None
        if k is None:
            cand = by_canon.get(canon(m["name"]) or "", [])
            if len(cand) != 1:
                raise SystemExit(f"{path}: no unique tensor for {m['name']} (candidates {cand[:3]})")
            k = cand[0]
        w = sd[k]
        if list(w.shape) != m["shape"]:
            raise SystemExit(f"{path}: {k} has shape {list(w.shape)}, the base {m['shape']}")
        out[m["name"]] = w.detach().clone()
        keys[m["name"]] = k
    del ft, sd
    return out, keys


# ---- metrics --------------------------------------------------------------------------------------------------------

def factors(M: torch.Tensor, r_max: int) -> dict:
    """Top-r_max singular vectors of M (fp32), its Frobenius norm, numerical rank and cumulative energy shares."""
    U, S, Vh = torch.linalg.svd(M.float(), full_matrices=False)
    s2 = S.double() ** 2
    tot = float(s2.sum())
    tol = float(S[0]) * max(M.shape) * torch.finfo(torch.float32).eps if S.numel() else 0.0
    return {"U": U[:, :r_max], "V": Vh[:r_max].T, "fro": math.sqrt(tot), "rank": int((S > tol).sum()) if tot > 0 else 0,
            "cum": (s2.cumsum(0) / tot)[:r_max].tolist() if tot > 0 else None}


def subspace_sim(A: torch.Tensor, B: torch.Tensor, r: int) -> float:
    """Mean cos^2 of the principal angles between span(A[:, :r]) and span(B[:, :r]) (orthonormal columns)."""
    r = min(r, A.shape[1], B.shape[1])
    return float((A[:, :r].T @ B[:, :r]).pow(2).sum()) / r


def captured_energy(F: dict, D: torch.Tensor, r: int, d_fro: float) -> float | None:
    """||U_r^T D V_r||_F^2 / ||D||_F^2 with U_r, V_r the top-r singular vectors of another matrix (F)."""
    if d_fro == 0:
        return None
    r = min(r, F["U"].shape[1])
    return float((F["U"][:, :r].T @ D.float() @ F["V"][:, :r]).pow(2).sum()) / d_fro ** 2


def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    na, nb = float(a.double().norm()), float(b.double().norm())
    return None if na == 0 or nb == 0 else float((a.double() * b.double()).sum()) / (na * nb)


def top_energy(F: dict, r: int) -> float | None:
    return None if F["cum"] is None else F["cum"][min(r, len(F["cum"])) - 1]


def matrix_report(grads: dict, dws: dict, arms: list[str], ranks: list[int], w_fro: float) -> dict:
    """Metrics of one matrix. grads: {"A2", arm, arm_nodes: tensor}, dws: {finetuned name: dW}, both on one device."""
    r_max = max(ranks)
    F = {k: factors(v, r_max) for k, v in grads.items()}
    D = {k: factors(v, r_max) for k, v in dws.items()}
    rs = [str(r) for r in ranks]
    rep: dict = {"grad": {k: {"fro": rd(f["fro"]), "rank": f["rank"],
                              "top_energy": {s: rd(top_energy(f, r)) for s, r in zip(rs, ranks)}}
                          for k, f in F.items()}}
    rep["nodes"] = {}
    for arm in arms:
        fin = F["A2"]["fro"] / 3
        rep["nodes"][arm] = {"ratio": rd(F[f"{arm}_nodes"]["fro"] / fin if fin > 0 else None),
                             "cos_final": rd(cosine(grads["A2"], grads[f"{arm}_nodes"]))}

    def pair(Fa, Fb, a, b):
        return {"cos": rd(cosine(a, b)), "r": {s: {"left": rd(subspace_sim(Fa["U"], Fb["U"], r)),
                                                   "right": rd(subspace_sim(Fa["V"], Fb["V"], r))}
                                               for s, r in zip(rs, ranks)}}
    keys = list(grads)
    rep["pairs"] = {f"{a}~{b}": pair(F[a], F[b], grads[a], grads[b]) for i, a in enumerate(keys) for b in keys[i + 1:]}
    rep["dw"] = {}
    for name, dw in dws.items():
        Fd = D[name]
        vs = {}
        for g in keys:
            vs[g] = {"cos": rd(cosine(-grads[g], dw)),
                     "r": {s: {"left": rd(subspace_sim(F[g]["U"], Fd["U"], r)),
                               "right": rd(subspace_sim(F[g]["V"], Fd["V"], r)),
                               "energy": rd(captured_energy(F[g], dw, r, Fd["fro"]))} for s, r in zip(rs, ranks)}}
        rep["dw"][name] = {"fro": rd(Fd["fro"]), "rel": rd(Fd["fro"] / w_fro if w_fro > 0 else None),
                           "rank": Fd["rank"], "top_energy": {s: rd(top_energy(Fd, r)) for s, r in zip(rs, ranks)},
                           "vs": vs}
    dk = list(dws)
    rep["dw_pairs"] = {f"{a}~{b}": pair(D[a], D[b], dws[a], dws[b]) for i, a in enumerate(dk) for b in dk[i + 1:]}
    return rep


def random_baseline(shape: list[int], ranks: list[int]) -> dict:
    o, i = shape
    return {str(r): {"left": rd(min(r, o, i) / o), "right": rd(min(r, o, i) / i),
                     "energy": rd(min(r, o, i) ** 2 / (o * i))} for r in ranks}


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        p = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, p))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            out[p] = float(v)
    return out


def kind(t: str) -> str:
    return "attn" if t in ATTN else "mlp" if t in MLP else "linattn" if t in LINATTN else "other"


def summarize(mats: list[dict]) -> dict:
    """Means of every numeric leaf by matrix type, attention / MLP / linear attention, layer (L<i>) and over all."""
    groups = collections.defaultdict(list)
    for m in mats:
        flat = flatten({k: v for k, v in m.items() if k not in SKIP_KEYS})
        for g in (m["type"], kind(m["type"]), f"L{m['layer']}", "all"):
            groups[g].append(flat)
    out = {}
    for g, rows in groups.items():
        keys = sorted(set().union(*rows))
        out[g] = {"n": len(rows), **{k: rd(sum(r[k] for r in rows if k in r) / sum(k in r for r in rows))
                                     for k in keys}}
    return out


# ---- markdown -------------------------------------------------------------------------------------------------------

def fmt(x, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def lr_cell(s: dict, path: str) -> str:
    left, right = s.get(path + "/left"), s.get(path + "/right")
    return "n/a" if left is None or right is None else f"{left:.3f}/{right:.3f}"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def group_order(summary: dict) -> tuple[list[str], list[str]]:
    types = [t for t in TYPE_ORDER if t in summary] + [g for g in ("attn", "mlp", "linattn", "other") if g in summary]
    layers = sorted((g for g in summary if re.fullmatch(r"L\d+", g)), key=lambda g: int(g[1:]))
    return types + ["all"], layers


def sim_avg(m: dict, g: str, ft: str, r: int):
    v = m.get("dw", {}).get(ft, {}).get("vs", {}).get(g, {}).get("r", {}).get(str(r))
    if not v or v.get("left") is None or v.get("right") is None:
        return None
    return (v["left"] + v["right"]) / 2


def contrast(mats: list[dict], a: tuple, b: tuple, r: int) -> str:
    """Mean over matrices of sim(a) - sim(b) (each the mean of the left and right similarity; a, b = (gradient,
    finetuned)), with the number of matrices where it is positive."""
    d = []
    for m in mats:
        x, y = sim_avg(m, *a, r), sim_avg(m, *b, r)
        if x is not None and y is not None:
            d.append(x - y)
    return "n/a" if not d else f"{sum(d) / len(d):+.3f} ({sum(v > 0 for v in d)}/{len(d)})"


def own_grad(ft: str, grads: list[str]) -> str | None:
    """The gradient of the run a finetuned model comes from: its name, or the part before '-' (A2-c12 -> A2)."""
    for g in (ft, ft.split("-")[0]):
        if g in grads:
            return g
    return None


def render_md(res: dict) -> str:
    meta, mats, summ = res["meta"], res["matrices"], res["summary"]
    ranks, arms = meta["ranks"], meta["node_arms"]
    grads = ["A2"] + [x for a in arms for x in (a, f"{a}_nodes")]
    fts = sorted({ft for m in mats for ft in m.get("dw", {})})
    rows_g, rows_l = group_order(summ)
    r_mid = ranks[len(ranks) // 2]
    b = meta["batch"]
    L = ["# One-step gradient subspaces vs the finished update (LoRA-One link)", ""]
    L.append(f"- model: `{meta['model']}`; data: `{meta['train']}` (md5 {meta['train_md5']}); tree sources "
             f"{meta['tree_sources']}")
    L.append(f"- batch: {b['n']} problems ({b['batch_sources']}; seed {meta['seed']}; "
             + ", ".join(f"{k} {v}" for k, v in b["sources"].items()) + "); node views: "
             + ", ".join(f"{a} {b['node_views'][a]}" for a in arms))
    L.append(f"- loss: soft CE, label smoothing {meta['label_smoothing']}; A2 = finals / N, <arm>_nodes = node views / "
             f"(3 N), <arm> = A2 / 3 + <arm>_nodes (trainer normalisation); precision {meta['precision']}")
    L.append(f"- matrices: {len(mats)} (layers {meta['layers']}; types {meta['modules']}); ranks {ranks}")
    if meta.get("merged_from"):
        L.append(f"- merged from {len(meta['merged_from'])} jobs: " + ", ".join(f"`{p}`" for p in meta["merged_from"])
                 + f"; max relative difference of the shared gradients' norms {fmt(meta.get('repro_max_rel'), 6)}")
    L += ["", "Similarity = mean cos^2 of the principal angles between top-r singular subspaces, shown left/right "
          "(left = output side, LoRA-One's B; right = input side, A). 1 = same subspace; `rand` = r/d, the "
          "expectation for unrelated subspaces. Energy = share of ||dW||_F^2 inside G's top-r left x right pair (at "
          "most dW's own top-r energy). Group rows are means over matrices.", ""]

    L += ["## Gradient norms; node vs final part", ""]
    hdr = ["group", "n", "‖G_A2‖", f"top-{r_mid} energy G_A2"]
    for a in arms:
        hdr += [f"‖{a}_nodes‖/‖{a}_final‖", f"cos(final, {a}_nodes)", f"top-{r_mid} energy G_{a}"]
    rows = []
    for g in rows_g + rows_l:
        s = summ[g]
        row = [g, str(s["n"]), fmt(s.get("grad/A2/fro"), 4), fmt(s.get(f"grad/A2/top_energy/{r_mid}"))]
        for a in arms:
            row += [fmt(s.get(f"nodes/{a}/ratio")), fmt(s.get(f"nodes/{a}/cos_final")),
                    fmt(s.get(f"grad/{a}/top_energy/{r_mid}"))]
        rows.append(row)
    L += table(hdr, rows) + [""]

    pairs = list(mats[0].get("pairs", {})) if mats else []
    L += ["## Gradient vs gradient", ""]
    for r in ranks:
        L += [f"r = {r}", ""]
        rows = [[g] + [lr_cell(summ[g], f"pairs/{p}/r/{r}") for p in pairs] + [lr_cell(summ[g], f"random/{r}")]
                for g in rows_g]
        L += table(["group"] + pairs + ["rand"], rows) + [""]
    rows = [[g] + [fmt(summ[g].get(f"pairs/{p}/cos")) for p in pairs] for g in rows_g]
    L += ["Flattened cosine", ""] + table(["group"] + pairs, rows) + [""]

    for ft in fts:
        L += [f"## Gradient vs finished update dW_{ft}", ""]
        rows = [[g, fmt(summ[g].get(f"dw/{ft}/rel"), 4)] + [fmt(summ[g].get(f"dw/{ft}/vs/{x}/cos")) for x in grads]
                for g in rows_g]
        L += table(["group", "‖dW‖/‖W‖"] + [f"cos(-G_{x}, dW)" for x in grads], rows) + [""]
        for r in ranks:
            L += [f"r = {r}", ""]
            rows = []
            for g in rows_g:
                s = summ[g]
                rows.append([g, fmt(s.get(f"dw/{ft}/top_energy/{r}")), lr_cell(s, f"random/{r}")]
                            + [lr_cell(s, f"dw/{ft}/vs/{x}/r/{r}") for x in grads]
                            + [fmt(s.get(f"dw/{ft}/vs/{x}/r/{r}/energy")) for x in grads])
            L += table(["group", "dW top-r energy", "rand"] + [f"G_{x}" for x in grads]
                       + [f"energy G_{x}" for x in grads], rows) + [""]
        rows = [[g] + [fmt(None if summ[g].get(f"dw/{ft}/vs/{x}/r/{r_mid}/left") is None else
                           (summ[g][f"dw/{ft}/vs/{x}/r/{r_mid}/left"] + summ[g][f"dw/{ft}/vs/{x}/r/{r_mid}/right"]) / 2)
                       for x in grads] for g in rows_l]
        L += [f"By layer, r = {r_mid}, (left + right) / 2", ""] + table(["layer"] + [f"G_{x}" for x in grads], rows) + [""]
    dps = sorted({p for m in mats for p in m.get("dw_pairs", {})})
    if dps:
        L += ["## Finished update vs finished update", ""]
        for r in ranks:
            rows = [[g] + [lr_cell(summ[g], f"dw_pairs/{p}/r/{r}") for p in dps] for g in rows_g]
            L += [f"r = {r}", ""] + table(["group"] + dps, rows) + [""]

    L += ["## Readout", "", "H2 (node and final gradients share top subspaces): similarity of G_A2 (= the final part "
          "up to 1/3) and G_<arm>_nodes, all matrices, left/right, against `rand`.", ""]
    s = summ["all"]
    rows = [[a, str(r), lr_cell(s, f"pairs/A2~{a}_nodes/r/{r}"), lr_cell(s, f"random/{r}"),
             fmt(s.get(f"pairs/A2~{a}_nodes/cos")), fmt(s.get(f"nodes/{a}/ratio"))] for a in arms for r in ranks]
    L += table(["arm", "r", "sim(G_A2, G_nodes)", "rand", "cos", "‖nodes‖/‖final‖"], rows) + [""]
    L += ["H1 (G_<arm> aligns more with dW_<arm> than G_A2 with dW_A2, at small r): per matrix (left + right) / 2 "
          "similarity; differences are means over matrices with (positive / matrices). Within-column contrasts hold the "
          "update fixed (A2-c12 trained 12 epochs, the node arms 4, so dW_A2 is larger and more spread: the cross-column "
          "contrast mixes that in).", ""]
    a2 = next((ft for ft in fts if own_grad(ft, ["A2"]) == "A2"), None)
    rows = []
    for ft in fts:
        g = own_grad(ft, grads)
        if g is None or g == "A2":
            continue
        for r in ranks:
            vals = [sim_avg(m, g, ft, r) for m in mats]
            vals = [v for v in vals if v is not None]
            own = sum(vals) / len(vals) if vals else None
            a2v = [v for v in (sim_avg(m, "A2", a2, r) for m in mats) if v is not None] if a2 else []
            rows.append([f"dW_{ft}", str(r), fmt(own), fmt(sum(a2v) / len(a2v) if a2v else None),
                         contrast(mats, (g, ft), ("A2", a2), r) if a2 else "n/a (no dW_A2)",
                         contrast(mats, (g, ft), ("A2", ft), r),
                         contrast(mats, (f"{g}_nodes", ft), ("A2", ft), r) if f"{g}_nodes" in grads else "n/a",
                         contrast(mats, ("A2", a2), (g, a2), r) if a2 else "n/a (no dW_A2)"])
    if rows:
        L += table(["dW", "r", f"sim(G_arm, dW_arm)", "sim(G_A2, dW_A2)", "H1: own arm - own A2",
                    "within dW_arm: G_arm - G_A2", "within dW_arm: G_nodes - G_A2", "within dW_A2: G_A2 - G_arm"],
                   rows) + [""]
    else:
        L += ["(no finetuned node-arm model in these results)", ""]
    return "\n".join(L) + "\n"


def write_outputs(res: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    res["summary"] = summarize(res["matrices"])
    (out / "grad_subspace.json").write_text(json.dumps(res, indent=1))
    (out / "grad_subspace.md").write_text(render_md(res))
    print(f"wrote {out / 'grad_subspace.json'} and {out / 'grad_subspace.md'}", flush=True)


# ---- report (merge jobs) --------------------------------------------------------------------------------------------

FINGERPRINT = ("train_md5", "batch_items_sha1", "seed", "label_smoothing", "subq_target", "tree_cap", "aux_total",
               "tree_node_weight", "tree_sources", "ranks", "matrices_sha1")


def merge(results: list[dict], paths: list[str]) -> dict:
    """One result from several jobs on the same base, data, batch and settings: the union of their gradients, node
    arms, pairs and finetuned models per matrix. Shared gradients must have (nearly) the same norms (recorded)."""
    fp0 = {k: results[0]["meta"].get(k) for k in FINGERPRINT}
    for p, r in zip(paths[1:], results[1:]):
        fp = {k: r["meta"].get(k) for k in FINGERPRINT}
        diff = {k: (fp0[k], fp[k]) for k in FINGERPRINT if fp0[k] != fp[k]}
        if diff:
            raise SystemExit(f"{p} differs from {paths[0]} in {diff}: not the same batch / settings")
    meta = dict(results[0]["meta"])
    arms = []
    for r in results:
        arms += [a for a in r["meta"]["node_arms"] if a not in arms]
    meta["node_arms"], meta["merged_from"] = arms, list(paths)
    meta["batch"] = dict(meta["batch"], node_views={k: v for r in results for k, v in r["meta"]["batch"]["node_views"].items()})
    by_name: dict = {}
    worst = 0.0
    for r in results:
        for m in r["matrices"]:
            if m["name"] not in by_name:
                by_name[m["name"]] = json.loads(json.dumps(m))
                continue
            cur = by_name[m["name"]]
            for g, v in m["grad"].items():
                if g in cur["grad"]:
                    a, b = cur["grad"][g]["fro"], v["fro"]
                    if a and b:
                        worst = max(worst, abs(a - b) / max(abs(a), abs(b)))
                else:
                    cur["grad"][g] = v
            for k in m.get("dw", {}):
                if k in cur.get("dw", {}):
                    raise SystemExit(f"finetuned model {k} appears in more than one job: merge one of them")
            for key in ("nodes", "pairs", "dw", "dw_pairs"):
                for k, v in m.get(key, {}).items():
                    cur.setdefault(key, {}).setdefault(k, v)
    meta["repro_max_rel"] = rd(worst)
    return {"meta": meta, "matrices": list(by_name.values())}


def report(paths: list[str], out: Path) -> None:
    results = [json.loads(Path(p).read_text()) for p in paths]
    res = merge(results, paths)
    write_outputs(res, out)
    if res["meta"]["repro_max_rel"] and res["meta"]["repro_max_rel"] > 1e-2:
        print(f"WARNING: the jobs' shared gradients differ by up to {res['meta']['repro_max_rel']:.3g} in norm",
              flush=True)


# ---- main -----------------------------------------------------------------------------------------------------------

def code_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", help="base model directory (the one the finetuned runs started from)")
    ap.add_argument("--train", help="student training data (e.g. data/student_v3sl/train.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", nargs="+", default=None, help="merge these grad_subspace.json files into --out")
    ap.add_argument("--n", type=int, default=256, help="problems in the batch")
    ap.add_argument("--seed", type=int, default=0, help="batch draw and node-view generator")
    ap.add_argument("--batch-sources", default="all", choices=["all", "tree"],
                    help="draw the batch from every item (the trainer's mixture) or from --tree-sources items only")
    ap.add_argument("--tree-sources", default="knights_knaves",
                    help="sources whose items get node views (train_student --tree-sources; numgen data: numgen); "
                         "empty = all")
    ap.add_argument("--node-arms", default="GFPS", help=f"comma list of {','.join(NODE_ARMS)}")
    ap.add_argument("--tree-cap", type=int, default=0)
    ap.add_argument("--aux-total", type=float, default=1.0)
    ap.add_argument("--tree-node-weight", type=float, default=0.0)
    ap.add_argument("--subq-target", default="cot")
    ap.add_argument("--label-smoothing", type=float, default=0.1, help="the analysed runs' LSMOOTH (trainer default 0)")
    ap.add_argument("--layers", default="every:4", help="all | every:K[:OFFSET|auto] | comma list")
    ap.add_argument("--modules", default=",".join(ATTN + MLP))
    ap.add_argument("--ranks", default="4,16,64")
    ap.add_argument("--finetuned", nargs="*", default=[], help="NAME=model_dir (NAME = the run's arm, e.g. GFPS, A2-c12)")
    ap.add_argument("--precision", default="fp32master", choices=["fp32master", "fp32"])
    ap.add_argument("--attn-impl", default="", choices=["", "sdpa", "eager", "flex_attention"])
    ap.add_argument("--reference-kernels", action="store_true",
                    help="PyTorch reference Gated DeltaNet kernels on the GPU too (always on the CPU)")
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--token-budget", type=int, default=8192, help="padded tokens per micro-batch (0 = --micro-bs)")
    ap.add_argument("--token-budget-seqs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--layer-groups", type=int, default=1, help="split the layers into K groups, one set of passes each")
    ap.add_argument("--store", default="cpu", choices=["cpu", "device"], help="where accumulated gradients wait")
    ap.add_argument("--save-grads", default="", help="also torch.save the raw gradients here (large: 1.26 GB per "
                    "gradient at 2B)")
    ap.add_argument("--check", action="store_true", help="smoke: --check-n tree-source problems, first layer only")
    ap.add_argument("--check-n", type=int, default=4)
    ap.add_argument("--check-tol", type=float, default=None,
                    help="--check: max relative error of the split vs the trainer layout (default 2e-2 with bf16 "
                         "autocast, 1e-4 without)")
    args = ap.parse_args()
    out = Path(args.out)
    if args.report:
        report(args.report, out)
        return
    if not args.model or not args.train:
        raise SystemExit("--model and --train are needed (or --report)")
    arms = [a for a in args.node_arms.split(",") if a]
    bad = [a for a in arms if a not in NODE_ARMS]
    if bad or len(set(arms)) != len(arms):
        raise SystemExit(f"--node-arms {args.node_arms}: unknown or repeated ({bad}); known {list(NODE_ARMS)}")
    ranks = sorted({int(x) for x in args.ranks.split(",") if x})
    if not ranks or ranks[0] < 1:
        raise SystemExit(f"--ranks {args.ranks}")
    fts = {}
    for kv in args.finetuned:
        name, sep, path = kv.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", name) or name in fts:
            raise SystemExit(f"--finetuned {kv}: NAME=dir with a unique NAME of letters, digits, _ . + -")
        if not (Path(path) / "config.json").exists():
            raise SystemExit(f"--finetuned {kv}: no config.json in {path}")
        fts[name] = path
    modules = [m for m in args.modules.split(",") if m]
    tree_sources = {x for x in args.tree_sources.split(",") if x} or None
    t_start = time.time()

    items = read_jsonl(args.train)
    if tree_sources and not any(it.get("source") in tree_sources for it in items):
        raise SystemExit(f"--tree-sources {args.tree_sources}: no item of {args.train} has one of these sources")
    preflight(items, arms, tree_sources)
    batch = pick_batch(items, args.n, args.seed, tree_sources, args.batch_sources, args.subq_target)
    vkw = dict(seed=args.seed, tree_sources=tree_sources, tree_cap=args.tree_cap, aux_total=args.aux_total,
               node_weight=args.tree_node_weight, subq_target=args.subq_target)
    full_finals, full_nodes, _ = build_views(batch, arms, **vkw)
    n_seq_full = {"final": len(full_finals), **{a: sum(map(len, v)) for a, v in full_nodes.items()}}
    if args.check:                       # smoke: a few tree-source problems (node views present)
        batch = pick_batch(items, args.check_n, args.seed, tree_sources, "tree", args.subq_target)
    finals, nodes, stats = build_views(batch, arms, **vkw)
    for a in arms:
        if not any(nodes[a]):
            raise SystemExit(f"{a}: no node view in the batch (--batch-sources {args.batch_sources}, tree sources "
                             f"{args.tree_sources})")
    print(f"batch: {len(batch)} problems, node views " + ", ".join(f"{a} {sum(map(len, nodes[a]))}" for a in arms),
          flush=True)

    import transformers
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    autocast = dev == "cuda" and args.precision == "fp32master"
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.model, torch.float32, args.attn_impl or None).to(dev)
    model.config.use_cache = False
    kernels = []
    if dev == "cpu" or args.reference_kernels:
        from cotdistill.interchange import use_reference_kernels
        kernels = use_reference_kernels(model)
    if not args.no_grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()                        # (checkpointing is active in train mode only; the models have no dropout)
    for p in model.parameters():
        p.requires_grad_(False)
    types = layer_types(model)
    layers = parse_layers(args.layers, types)
    mats = select_matrices(model, layers, modules)
    if args.check:
        mats = [m for m in mats if m["layer"] == mats[0]["layer"]]
        layers = [mats[0]["layer"]]
    print(f"matrices: {len(mats)} in layers {sorted({m['layer'] for m in mats})} "
          f"({sum(math.prod(m['shape']) for m in mats) / 1e6:.1f}M params); kernels {kernels or 'model default'}; "
          f"device {dev}, autocast {autocast}", flush=True)
    ft_w, ft_keys = {}, {}
    for name, path in fts.items():       # before the GPU passes: a wrong directory fails early
        t0 = time.time()
        ft_w[name], ft_keys[name] = load_finetuned(path, mats)
        print(f"finetuned {name}: {path} ({time.time() - t0:.0f} s)", flush=True)

    store = "cpu" if args.store == "cpu" else dev
    G, info = gradients(model, tok, finals, nodes, mats, args, autocast=autocast, store=store)
    if args.save_grads:
        torch.save({"final": G["final"], "nodes": {a: G[a] for a in arms}, "n": len(finals),
                    "note": "final = G_A2 (finals / N); nodes[arm] = G_arm_nodes (node views / 3N)"}, args.save_grads)

    check = None
    if args.check:
        tol = args.check_tol if args.check_tol is not None else (2e-2 if autocast else 1e-4)
        check = {"tol": tol, "direct_rel_err": {}, "nonzero": True, "finite": True}
        for a in arms:
            direct = direct_full_tree(model, tok, finals, nodes[a], mats, args, autocast=autocast, store=store)
            err = 0.0
            for m in mats:
                derived = G["final"][m["name"]] / 3 + G[a][m["name"]]
                d = direct[m["name"]]
                err = max(err, float((d - derived).norm() / d.norm().clamp_min(1e-30)))
            check["direct_rel_err"][a] = err
        for key, g in G.items():
            for name, t in g.items():
                check["finite"] &= bool(torch.isfinite(t).all())
                check["nonzero"] &= float(t.norm()) > 0
        n_groups = len(layer_groups(mats, args.layer_groups))
        sps = {k: sum(x["s"] for x in v) / max(1, sum(x["seqs"] for x in v)) for k, v in info.items()}
        est = sum(n_seq_full[k] * sps[k] for k in n_seq_full) * max(1, args.layer_groups)
        check.update({"s_per_seq": sps, "full_run_sequences": n_seq_full, "groups_in_check": n_groups,
                      "est_full_pass_minutes": rd(est / 60),
                      "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev == "cuda" else None})

    res_mats = []
    t_m = time.time()
    for m in mats:
        nm = m["name"]
        g = {"A2": G["final"][nm].to(dev)}
        for a in arms:
            g[f"{a}_nodes"] = G[a][nm].to(dev)
            g[a] = g["A2"] / 3 + g[f"{a}_nodes"]
        ordered = {"A2": g["A2"], **{k: g[k] for a in arms for k in (a, f"{a}_nodes")}}
        w = m["param"].detach().float()
        dws = {name: ft_w[name][nm].to(dev).float() - w for name in fts}
        rep = matrix_report(ordered, dws, arms, ranks, float(w.norm()))
        res_mats.append({"name": nm, "layer": m["layer"], "type": m["type"], "shape": m["shape"], **rep,
                         "random": random_baseline(m["shape"], ranks)})
        del g, ordered, dws
    print(f"metrics: {len(mats)} matrices in {time.time() - t_m:.0f} s", flush=True)

    meta = {"script": "scripts/grad_subspace.py", "code_commit": code_commit(), "model": args.model,
            "train": args.train, "train_md5": md5(args.train), "seed": args.seed, "node_arms": arms,
            "tree_sources": sorted(tree_sources) if tree_sources else None, "tree_cap": args.tree_cap,
            "aux_total": args.aux_total, "tree_node_weight": args.tree_node_weight, "subq_target": args.subq_target,
            "label_smoothing": args.label_smoothing, "precision": args.precision, "autocast": autocast,
            "kernels": kernels, "layers": layers, "modules": modules, "ranks": ranks, "check": args.check,
            "finetuned": fts, "finetuned_keys_renamed": {k: {a: b for a, b in v.items() if a != b}
                                                         for k, v in ft_keys.items()},
            "matrices_sha1": hashlib.sha1("\n".join(m["name"] for m in mats).encode()).hexdigest(),
            "batch_items_sha1": hashlib.sha1("\n".join(str(it.get("item_id")) for it in batch).encode()).hexdigest(),
            "batch": {"n": len(batch), "batch_sources": "tree (check)" if args.check else args.batch_sources,
                      "sources": dict(collections.Counter(it.get("source") for it in batch).most_common()),
                      "node_views": {a: sum(map(len, nodes[a])) for a in arms}, "tree_stats": stats},
            "loss": {"A2": rd(sum(x["loss"] for x in info["final"]) / len(info["final"])),
                     **{f"{a}_nodes": rd(sum(x["loss"] for x in info[a]) / len(info[a])) for a in arms}},
            "passes": info, "seconds": round(time.time() - t_start, 1)}
    res = {"meta": meta, "matrices": res_mats}
    write_outputs(res, out)
    if check is not None:
        (out / "check.json").write_text(json.dumps(check, indent=1))
        bad = [f"{a}: split vs trainer layout {e:.2e} > {check['tol']:.0e}" for a, e in check["direct_rel_err"].items()
               if not e <= check["tol"]] + ([] if check["nonzero"] else ["a gradient is zero"]) + \
              ([] if check["finite"] else ["a gradient is not finite"])
        print("check", json.dumps({k: v for k, v in check.items()}), flush=True)
        if bad:
            raise SystemExit("CHECK FAILED: " + "; ".join(bad))
        print(f"check ok: estimated {check['est_full_pass_minutes']} min of passes for the full run "
              f"({n_seq_full} sequences)", flush=True)


if __name__ == "__main__":
    main()
