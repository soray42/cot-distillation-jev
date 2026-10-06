"""Train one student arm and evaluate it on held-out sets (final question only).

Example (HPC):
  python scripts/train_student.py --model ~/cotd/models/Qwen3.5-2B-Base \
      --train data/student/train.jsonl --eval val=data/student/val.jsonl \
      --final teacher --subq cot --subq-target cot --out runs/A1S1-seed0 --seed 0

Arms: --final {none,teacher,gold} x --subq {none,cot,random}. Writes metrics.json and preds_<set>.jsonl.

Two GPUs, one job (--fsdp, FSDP2 data parallel; notes/next_round/scale12b/fsdp2.md):
  python -m torch.distributed.run --standalone --nproc_per_node 2 scripts/train_student.py ... --precision bf16 \
      --fsdp --fsdp-optim cpu-adamw|8bit-sr [--token-budget 4096]
Every update holds the same sequences as on one GPU, split across the ranks, gradients summed; rank 0 writes the files.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.student import (Example, brier_loss, build_examples, build_fact_examples, build_full_tree, build_group,  # noqa: E402
                                build_stage_examples, evaluate, evaluate_local, evaluate_subq, kl_loss, label_logits,
                                lm_loss, load_model, logit_softcap, micro_batches, special_prefix, subq_sets)


# Arguments that do not change the training trajectory: a checkpoint may be resumed with different values.
RESUME_FREE = {"stop_after_h", "ckpt_every_min", "ckpt_single", "stop_at_step", "code_commit", "eval", "eval_subq", "eval_bs",
               "eval_max_len", "dump_hidden", "dump_layers", "log_every", "save", "eval_local", "eval_local_view",
               "log_update_loss"}

# Training arguments added after checkpoints may have been written: a checkpoint without the key had this value.
NEW_ARG_DEFAULTS = {"tree_view": "problem", "tree_sources": "", "tree_node_weight": 0.0, "attn_impl": "", "fsdp": False,
                    "fsdp_optim": None}

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]


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
    ap.add_argument("--reset-optim", action="store_true", help="curricula: reset the AdamW state at each stage")
    ap.add_argument("--fact-stages", type=int, default=0,
                    help="fact internalization: N passes with the tree nodes stated in the prompt, dropping shallow "
                         "levels first, then the plain final questions for --epochs; 0 = off")
    ap.add_argument("--facts-always", action="store_true",
                    help="upper-bound diagnostic: every final question is trained with all tree nodes stated")
    ap.add_argument("--grouped-aux", type=int, default=0,
                    help="exposure-matched grouped training: every optimizer update holds --items-per-update problems, "
                         "each as 1 final view + K auxiliary views (0 = off)")
    ap.add_argument("--aux-kind", default="cot", choices=["cot", "random", "placebo", "tree", "tree_shuf", "cot_mc", "random_mc", "cot_randlabel"],
                    help="grouped training: CoT nodes, matched controls, the CoT views with zero loss (placebo), CoT "
                         "nodes with their parents' results stated (tree), or with unrelated nodes' results (tree_shuf)")
    ap.add_argument("--tree-full", default="", choices=["", "true", "matched", "plain", "placebo", "control"],
                    help="full-tree groups: every problem with ALL its nodes (up to --tree-cap) in one update; true = "
                         "parents' results stated, matched = as many depth-matched non-parent results, plain = no "
                         "facts, placebo = plain at zero weight, control = the control questions")
    ap.add_argument("--aux-answer", default="all", choices=["all", "exclude", "only"],
                    help="grouped training: auxiliary CoT nodes that are not (exclude) / are (only) answer components")
    ap.add_argument("--subq-filter", default="all", choices=["all", "path", "mirror"],
                    help="decision chains: ask only the nodes of the chain the final reads (path) or of the mirror "
                         "chain (mirror) in --subq cot/mix, grouped and full-tree views (stated results and matched "
                         "pools still use every node); all = every node")
    ap.add_argument("--tree-eligible-only", action="store_true",
                    help="full-tree groups: keep only the eligible node views (TFE/TFME/GFE); items without one train "
                         "their final question only; the per-problem auxiliary weight is split over the kept views")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="mix every training target with the uniform distribution over its options (0 = off)")
    ap.add_argument("--order-seed", type=int, default=None,
                    help="seed of the problem order (grouped and full-tree arms); default = --seed. Varying it alone "
                         "or --seed alone (node sampling, option order) decomposes the seed variance")
    ap.add_argument("--tree-cap", type=int, default=10,
                    help="full-tree groups: at most this many nodes per problem (0 = no cap: every node every epoch)")
    ap.add_argument("--tree-view", default="problem", choices=["problem", "local", "ancestors", "statements"],
                    help="full-tree groups: how a node is asked. problem = with the original problem (the TF/TFM/GF "
                         "arms); local = Tree Forcing: no problem text, only the preamble and the parents' results "
                         "(--tree-full true) or as many matched non-parent results (matched) or nothing (plain: "
                         "TFLN, in rng lockstep with TFL); parse nodes show their statement line(s); ancestors = every "
                         "transitive ancestor's result, shallow to deep (true only). Needs the local fields of "
                         "data/student_v3sl (scripts/build_local_kk.py). statements = GFP-S (plain only): the problem "
                         "view without the final question and its options (K&K: preamble + statement lines; numgen: "
                         "the problem text without its question sentence), in rng lockstep with GFP")
    ap.add_argument("--tree-sources", default="",
                    help="full-tree groups: comma list of item sources that get node views; other items train their "
                         "final question only (every view mode); empty = all sources")
    ap.add_argument("--tree-node-weight", type=float, default=0.0,
                    help="full-tree groups: loss weight of every node view (times the item weight); 0 = the per-problem "
                         "convention, --aux-total split evenly over the problem's views")
    ap.add_argument("--node-format", default="yesno", choices=["yesno", "mc"],
                    help="full-tree groups: ask nodes as yes/no questions, or in multiple-choice form with other "
                         "confident No nodes of the item and 'None of them' as options (same stated results)")
    ap.add_argument("--aux-total", type=float, default=1.0,
                    help="full-tree groups: auxiliary weight per problem, split evenly over its node views")
    ap.add_argument("--aux-sources", default="", help="grouped training: comma list of item sources whose auxiliary "
                    "views keep their weight; items of other sources get the same views at zero weight (placebo)")
    ap.add_argument("--fact-withdraw", type=float, default=0.0,
                    help="tree aux views: the stated results are kept with probability 1 - progress / F over the first "
                         "fraction F of training and dropped entirely afterwards (0 = always kept)")
    ap.add_argument("--aux-weight", type=float, default=0.5, help="grouped training: loss weight of each auxiliary view")
    ap.add_argument("--items-per-update", type=int, default=32, help="grouped training: problems per optimizer update")
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
    ap.add_argument("--optim", default="adamw", choices=["adamw", "adamw8bit", "adamw8bit-sr", "adamw8bit_sr"],
                    help="adamw8bit = bitsandbytes (round-to-nearest bf16 writes); adamw8bit-sr = torchao AdamW8bit "
                         "with bf16 stochastic rounding (needs --precision bf16 and torchao; adamw8bit_sr = the same)")
    ap.add_argument("--token-budget", "--mb-tokens", type=int, default=0,
                    help="full-tree loop: length-sorted micro-batches of at most N padded tokens (and at most "
                         "--token-budget-seqs sequences) instead of --micro-bs; same gradient per update; 0 = off")
    ap.add_argument("--token-budget-seqs", type=int, default=16, help="--token-budget: sequences per micro-batch cap")
    ap.add_argument("--stop-after-h", type=float, default=0.0,
                    help="full-tree loop: at the first update boundary after H hours (counted from this process's "
                         "start) write <out>/ckpt and exit with code 3; a rerun with the same --out resumes; 0 = off")
    ap.add_argument("--ckpt-every-min", type=float, default=0.0,
                    help="full-tree loop: also write <out>/ckpt every M minutes (one checkpoint, replaced atomically)")
    ap.add_argument("--ckpt-single", action="store_true",
                    help="delete the previous checkpoint before writing the new one: one copy on disk instead of two "
                         "during a write (4B: about 17 instead of 34 GB), but a kill during the write leaves none")
    ap.add_argument("--stop-at-step", type=int, default=0,
                    help="planned stop (checkpoint + exit 3) after this update; for tests and the resume smoke")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--fsdp", action="store_true",
                    help="FSDP2 data parallelism over the torchrun ranks (one GPU each): every update holds the same "
                         "sequences as on one GPU, split across the ranks, gradients summed; --precision bf16 (fp32master "
                         "only with cpu-adamw, for equivalence checks); "
                         "--token-budget works in every loop; rank 0 writes the outputs (scale12b/fsdp2.md)")
    ap.add_argument("--fsdp-optim", default=None, choices=["cpu-adamw", "8bit-sr"],
                    help="--fsdp optimizer (default cpu-adamw): cpu-adamw = fp32 master + AdamW of each rank's shard on "
                         "the CPU (one gradient D2H and one weight H2D per update; no checkpoints); 8bit-sr = torchao "
                         "AdamW8bit with bf16 stochastic rounding on the GPU shards (resumable)")
    ap.add_argument("--log-update-loss", action="store_true",
                    help="grouped / plain loops: also log upd_loss, the whole update's loss (the full-tree loop's and "
                         "--fsdp's loss column); for step-by-step comparisons with --fsdp runs")
    ap.add_argument("--attn-impl", default="", choices=["", "sdpa", "eager", "flex_attention"],
                    help="attention implementation passed to from_pretrained (empty = the transformers default, sdpa). "
                         "Gemma 4: not flash_attention_2, whose kernels stop at head_dim 256 (global layers use 512)")
    ap.add_argument("--eval-bs", type=int, default=8)
    ap.add_argument("--eval-max-len", type=int, default=6144, help="long eval states (JevBench hard) need more room")
    ap.add_argument("--save", action="store_true", help="save the final weights (bf16) to <out>/model")
    ap.add_argument("--eval-at", default="", help="training-length curve: comma list of points, in epochs, at which "
                    "the --eval-mid sets are evaluated during training (e.g. 0.5,1,2,4); stored as metrics.json trajectory")
    ap.add_argument("--eval-mid", default="", help="comma list of --eval set names for the --eval-at points (default all)")
    ap.add_argument("--eval-mid-max", type=int, default=600,
                    help="at --eval-at points each set is cut to at most this many items (every k-th item, spread "
                         "across the file's task groups); the final eval always uses the full sets")
    ap.add_argument("--lora-r", type=int, default=0, help="LoRA rank (0 = full fine-tuning)")
    ap.add_argument("--lora-alpha", type=float, default=0.0, help="LoRA alpha (default 2 x rank)")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", default=",".join(LORA_TARGETS),
                    help="module names to adapt (names absent from the model are skipped)")
    ap.add_argument("--dump-layers", default="", help="comma list of decoder layers whose readout states are saved "
                    "with --dump-hidden (shape [n, layers + 1, hidden]; the last entry is the final readout state)")
    ap.add_argument("--dump-hidden", default="", help="comma list of eval-set names whose readout hidden states "
                    "to save as hidden_<name>.pt (float16, preds order)")
    ap.add_argument("--eval-local", default="", help="student data file with local fields (e.g. data/student_v3sl/"
                    "val.jsonl): local-step probe, every K&K node asked in its local view (eval key local_kk, "
                    "preds_local_kk.jsonl; accuracy by node type and depth)")
    ap.add_argument("--eval-local-view", default="local", choices=["local", "ancestors"],
                    help="--eval-local: parents' results (key local_kk) or all ancestors' results (local_kk_anc)")
    ap.add_argument("--eval-subq", default=None, help="student data file (e.g. val.jsonl) whose sub-questions to "
                    "score, or comma-separated name=path entries (predictions in preds_subq[_name].jsonl)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--code-commit", default="", help="commit of the code checkout (recorded in metrics.json args only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()
    tree_opts = args.tree_view != "problem" or args.tree_sources or args.tree_node_weight or args.tree_cap == 0
    if tree_opts and not args.tree_full:
        raise SystemExit("--tree-view / --tree-sources / --tree-node-weight / --tree-cap 0 need --tree-full")
    if args.tree_view != "problem" and not (args.tree_full == "true" and args.tree_view != "statements" or
                                             args.tree_full in ("matched", "plain") and args.tree_view == "local" or
                                             args.tree_full == "plain" and args.tree_view == "statements"):
        raise SystemExit(f"--tree-view {args.tree_view} is implemented for --tree-full true (local/ancestors), matched "
                         "(local) and plain (local, statements)")
    if args.tree_view != "problem" and (args.tree_eligible_only or args.node_format != "yesno"):
        raise SystemExit("--tree-view local/ancestors/statements does not combine with --tree-eligible-only / "
                         "--node-format mc")
    if args.tree_eligible_only and args.tree_full not in ("true", "matched", "plain", "placebo"):
        raise SystemExit("--tree-eligible-only needs --tree-full true/matched/plain/placebo")
    if args.subq_filter != "all" and (args.depth_stages or args.fact_stages or args.facts_always):
        raise SystemExit("--subq-filter is not implemented for --depth-stages / --fact-stages / --facts-always")
    if args.subq_filter != "all" and (args.aux_kind in ("random", "random_mc") if args.grouped_aux else
                                      args.tree_full == "control" if args.tree_full else args.subq in ("random", "none")):
        raise SystemExit("--subq-filter only restricts CoT nodes; this arm asks none (control questions or no "
                         "sub-questions), so the filter would do nothing")

    if args.optim == "adamw8bit_sr":     # the spelling of the plan notes
        args.optim = "adamw8bit-sr"
    if args.optim == "adamw8bit-sr" and args.precision != "bf16":
        raise SystemExit("--optim adamw8bit-sr rounds bf16 weights stochastically; it needs --precision bf16")
    resumable = bool(args.stop_after_h or args.ckpt_every_min or args.stop_at_step)
    if (resumable or args.token_budget and not args.fsdp) and not args.tree_full:
        raise SystemExit("--token-budget / --stop-after-h / --ckpt-every-min / --stop-at-step are implemented for the "
                         "full-tree loop only (--tree-full; --token-budget also with --fsdp)")
    if args.fsdp_optim and not args.fsdp:
        raise SystemExit("--fsdp-optim needs --fsdp")
    if args.fsdp:
        args.fsdp_optim = args.fsdp_optim or "cpu-adamw"
        if args.precision != "bf16" and args.fsdp_optim != "cpu-adamw":
            raise SystemExit("--fsdp-optim 8bit-sr rounds bf16 shards stochastically: use --precision bf16")
        # (--precision fp32master with cpu-adamw keeps fp32 shards on the GPUs: the exact analogue of the single-GPU
        # fp32master + AdamW recipe, for equivalence checks at small scale; 12B needs bf16 shards)
        if args.optim != "adamw":
            raise SystemExit("--fsdp: the optimizer is --fsdp-optim (8bit-sr = torchao AdamW8bit with bf16 stochastic "
                             "rounding, cpu-adamw = fp32 master + AdamW on the CPU), not --optim")
        if args.lora_r or args.rationale_lm:
            raise SystemExit("--fsdp does not support --lora-r or --rationale-lm")
        if resumable and args.fsdp_optim != "8bit-sr":
            raise SystemExit("--fsdp-optim cpu-adamw does not checkpoint (fp32 master + m + v = 12 B/param, 143 GB at "
                             "12B, over the 100 GB quota): no --stop-after-h / --ckpt-every-min / --stop-at-step")
    if args.ckpt_single and not resumable:
        raise SystemExit("--ckpt-single needs --stop-after-h, --ckpt-every-min or --stop-at-step")
    t_start = time.time()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    # Problem order for the grouped and full-tree arms comes from its own generator, so every arm with the same seed
    # sees the same problems in the same updates in every epoch (common random numbers for paired comparisons);
    # sampling inside a group still uses rng.
    order_rng = random.Random(f"order-{args.seed if args.order_seed is None else args.order_seed}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fs = None                        # --fsdp: rank, world, device and mesh (cotdistill.fsdp.Dist)
    if args.fsdp:
        import torch.distributed as dist
        from cotdistill import fsdp as fsdp_lib
        fs = fsdp_lib.init()
        print(f"fsdp: {fs.world} ranks, device {fs.device}, optimizer {args.fsdp_optim}", flush=True)
    is_main = fs is None or fs.rank == 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float32 if args.precision == "fp32master" else torch.bfloat16
    model = load_model(args.model, dtype, args.attn_impl or None)
    if fs is None:
        model = model.to(dev)
    model.config.use_cache = False
    print(f"readout: special prefix {special_prefix(tok)}, logit soft-cap {logit_softcap(model)}, "
          f"attention {model.config._attn_implementation}", flush=True)
    if not args.no_grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        if fs is None:
            model.gradient_checkpointing_enable()
        else:                        # non-reentrant checkpointing composes with FSDP2's per-layer hooks
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.lora_r:                  # frozen backbone + LoRA on attention, MLP and Gated-DeltaNet projections
        from peft import LoraConfig, get_peft_model
        names = {n.rsplit(".", 1)[-1] for n, _ in model.named_modules()}
        targets = [t for t in args.lora_targets.split(",") if t in names]
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha or 2 * args.lora_r,
                                                 lora_dropout=args.lora_dropout, target_modules=targets, bias="none"))
        if not args.no_grad_ckpt:
            model.enable_input_require_grads()
        print(f"lora r={args.lora_r} targets={targets}", flush=True)
    if fs is not None:               # shard on the CPU-loaded model: each rank moves only its shards to its GPU
        model = fsdp_lib.shard_model(model, fs.mesh)
    params = [p for p in model.parameters() if p.requires_grad]
    trainer = None                   # --fsdp: clipping over the shards + the optimizer setup (cotdistill.fsdp.ShardedOptim)
    if fs is not None:
        trainer = fsdp_lib.ShardedOptim(params, args.fsdp_optim, args.lr, fs.device)
        opt = trainer.opt
    elif args.optim == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.0)
    elif args.optim == "adamw8bit-sr":   # 8-bit block-wise states, bf16 weights written with stochastic rounding
        try:
            from torchao.optim import AdamW8bit
        except ImportError as e:
            try:                         # torchao < 0.11 keeps the low-bit optimizers under prototype
                from torchao.prototype.low_bit_optim import AdamW8bit
            except ImportError:
                raise SystemExit(f"--optim adamw8bit-sr needs torchao (pip install --no-deps torchao, so pip leaves "
                                 f"torch alone): {e}")
        opt = AdamW8bit(params, lr=args.lr, weight_decay=0.0, bf16_stochastic_round=True)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, fused=dev == "cuda")

    train_items = read_jsonl(args.train)
    if args.subq_filter != "all" and not any("on_path" in sq for it in train_items for sq in it.get("subqs", [])):
        raise SystemExit(f"--subq-filter {args.subq_filter}: no node of {args.train} has on_path (decision-chain data "
                         "only, e.g. DATA=data/student_chain); the filter would drop every node")
    evals = {kv.split("=", 1)[0]: read_jsonl(kv.split("=", 1)[1]) for kv in args.eval}

    def epoch_examples() -> list[Example]:
        ex = []
        for it in train_items:
            ex += build_examples(it, final=args.final, subq=args.subq, subq_target=args.subq_target,
                                 lambda_sub=args.lambda_sub, rng=rng, rationale_lm=args.rationale_lm,
                                 lambda_lm=args.lambda_lm, permute_final=args.permute_final,
                                 subq_k=args.subq_k, subq_weight=args.subq_weight, subq_frac=args.subq_frac,
                                 subq_filter=args.subq_filter)
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
    def fact_stage(stage: int, n: int):
        def make() -> list[Example]:
            ex = [e for it in train_items for e in build_fact_examples(it, stage=stage, n_stages=n, final=args.final)]
            rng.shuffle(ex)
            return ex
        return make

    passes = {"n": 0}
    aux_sources = {x for x in args.aux_sources.split(",") if x}

    def grouped_examples() -> list[Example]:
        """Problems in shuffled order, each as build_group's [final] + K auxiliary views, so every update covers the same
        problems with the same number of sequences in every arm. With --fact-withdraw, stated results fade out with
        training progress (counted in passes over the data)."""
        items = train_items[:]
        order_rng.shuffle(items)
        ex, n = [], len(items)
        for j, it in enumerate(items):
            progress = (passes["n"] + j / n) / max(args.epochs, 1e-9)
            keep = 1.0 if not args.fact_withdraw else max(0.0, 1.0 - progress / args.fact_withdraw)
            ex += build_group(it, aux_kind=args.aux_kind, k=args.grouped_aux, aux_weight=args.aux_weight, rng=rng,
                              final=args.final, subq_target=args.subq_target, permute_final=args.permute_final,
                              keep_prob=keep, aux_active=not aux_sources or it.get("source") in aux_sources,
                              aux_answer=args.aux_answer, subq_filter=args.subq_filter)
        passes["n"] += 1
        return ex

    tree_sources = {x for x in args.tree_sources.split(",") if x} or None
    if tree_sources and not any(it.get("source") in tree_sources for it in train_items):
        raise SystemExit(f"--tree-sources {args.tree_sources}: no item of {args.train} has one of these sources")
    if args.tree_view == "statements":         # GFP-S: every node-view item needs a statement block (fail before training)
        from cotdistill.treeplan import statement_block
        for it in train_items:
            if tree_sources is None or it.get("source") in tree_sources:
                try:
                    statement_block(it)
                except ValueError as e:
                    raise SystemExit(f"--tree-view statements: {e}")
    tree_stats: dict = {}

    def fulltree_updates() -> list[list[Example]]:
        """Problems in shuffled order, each as [final] + all its node views; items_per_update problems per update
        (the number of sequences per update varies with tree size)."""
        items = train_items[:]
        order_rng.shuffle(items)
        st: dict = {}
        groups = [g for g in (build_full_tree(it, mode=args.tree_full, rng=rng, cap=args.tree_cap,
                                              aux_total=args.aux_total, node_format=args.node_format,
                                              final=args.final, eligible_only=args.tree_eligible_only,
                                              subq_target=args.subq_target, permute_final=args.permute_final,
                                              subq_filter=args.subq_filter, view=args.tree_view,
                                              sources=tree_sources, node_weight=args.tree_node_weight, stats=st)
                              for it in items) if g]
        if tree_opts and not tree_stats:     # the plan's counting pass: views, dropped views, stated results
            tree_stats.update(st)
            print("tree views", json.dumps(st, sort_keys=True), flush=True)
        b = args.items_per_update
        return [[e for g in groups[i:i + b] for e in g] for i in range(0, len(groups), b)]

    if args.tree_full:
        per_step = 1                 # make() returns whole updates
        stages = [("fulltree", fulltree_updates, args.epochs)]
    elif args.grouped_aux:
        group = (1 + args.grouped_aux) * args.items_per_update
        if group % args.micro_bs:
            raise SystemExit(f"(1 + grouped_aux) x items_per_update = {group} must be a multiple of micro_bs")
        args.grad_accum = group // args.micro_bs
        per_step = group
        stages = [("grouped", grouped_examples, args.epochs)]
    elif args.facts_always:
        stages = [("facts", fact_stage(0, 0), args.epochs)]
    elif args.fact_stages:           # fact internalization: stated facts removed shallow-first, then plain finals
        stages = [(f"facts{s}", fact_stage(s, args.fact_stages), 1.0) for s in range(args.fact_stages)]
        stages.append(("final", epoch_examples, args.epochs))
    elif args.depth_stages:          # sub-question stages by depth, then the final questions, each with its own schedule
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
    passes["n"] = 0                  # the counting pass above does not count as training progress
    print(f"examples/epoch={plan[-1][2]} steps={total_steps} params={sum(p.numel() for p in params)/1e9:.2f}B "
          f"device={dev}", flush=True)

    cache: dict = {}
    step, micro, t0, seen_tok = 0, 0, time.time(), 0
    history, trajectory = [], []
    epochs_total = args.epochs if not args.depth_stages and not args.fact_stages else None
    eval_steps = {}
    for e in (float(x) for x in args.eval_at.split(",") if x):
        assert epochs_total, "--eval-at needs a single-stage run (no --depth-stages/--fact-stages)"
        eval_steps[max(1, round(e / epochs_total * total_steps))] = e
    mid_names = [x for x in args.eval_mid.split(",") if x in evals] or list(evals)   # absent names (smoke) skipped
    mid_sets = {}
    for name in mid_names:
        items = evals[name]
        k = max(1, -(-len(items) // args.eval_mid_max))
        mid_sets[name] = items[::k]

    def mid_eval(at_step: int) -> None:
        row = {"step": at_step, "epoch": eval_steps[at_step], "s_elapsed": round(time.time() - t0, 1)}
        for name, items in mid_sets.items():
            m, _ = evaluate(model, tok, items, args.eval_max_len, args.eval_bs, cache)
            row[name] = {k: m[k] for k in ("acc", "nll", "ece", "brier") if k in m}
        trajectory.append(row)
        print("trajectory", json.dumps(row), flush=True)
        if is_main:
            (out / "trajectory.json").write_text(json.dumps(trajectory, indent=1))

    # Resume (full-tree loop only): <out>/ckpt holds the weights, the optimizer and scheduler state, the counters, every
    # RNG state and the rest of the current epoch's updates. It is written to ckpt.tmp (state.pt last, by rename, so
    # ckpt.tmp/state.pt marks a complete write) and swapped in by two renames; a kill at any point leaves a complete
    # checkpoint (ckpt; between the renames the new ckpt.tmp, else ckpt.old), except during a --ckpt-single write.
    ck_dir, train_args = out / "ckpt", {k: v for k, v in vars(args).items() if k not in RESUME_FREE}
    ck_tmp, ck_old = out / "ckpt.tmp", out / "ckpt.old"
    resume = None
    stop_mark = out / "planned_stop"   # --fsdp: torchrun turns the ranks' exit code 3 into 1, so the job script reads this
    if resumable:
        if is_main:
            if ck_dir.exists() and (ck_tmp / "state.pt").exists():   # killed after a complete write, before the swap
                shutil.rmtree(ck_dir)
            if not ck_dir.exists():
                for d in (ck_tmp, ck_old):   # the newest complete one
                    if (d / "state.pt").exists():
                        d.rename(ck_dir)
                        print(f"recovered {d} as {ck_dir}", flush=True)
                        break
                else:
                    if ck_tmp.exists():
                        print(f"{ck_tmp} is incomplete (killed during a write) and there is no other checkpoint: "
                              "training starts from the beginning", flush=True)
            shutil.rmtree(ck_tmp, ignore_errors=True)
            shutil.rmtree(ck_old, ignore_errors=True)
            if fs is not None:
                stop_mark.unlink(missing_ok=True)
        if fs is not None:
            fs.barrier()
        if ck_dir.exists():
            resume = torch.load(ck_dir / "state.pt", map_location="cpu", weights_only=False)
            diff = {k: (resume["args"].get(k, NEW_ARG_DEFAULTS.get(k)), v) for k, v in train_args.items()
                    if resume["args"].get(k, NEW_ARG_DEFAULTS.get(k)) != v}
            if diff:
                raise SystemExit(f"{ck_dir} was written with other training arguments (checkpoint, now): {diff}")
            if fs is None:
                if not (ck_dir / "optim.pt").exists():
                    raise SystemExit(f"{ck_dir} has no optimizer state")
                model.load_state_dict(torch.load(ck_dir / "model.pt", map_location="cpu", mmap=True, weights_only=True))
            else:                            # sharded model (torch.distributed.checkpoint), one optimizer file per rank
                if resume.get("world") != fs.world or not (ck_dir / f"optim.{fs.rank}.pt").exists():
                    raise SystemExit(f"{ck_dir} was written by {resume.get('world')} ranks, this run has {fs.world}")
                fsdp_lib.load_sharded(model, ck_dir / "model")
            rng.setstate(resume["rng"])          # after the plan's counting pass, which consumed both generators
            order_rng.setstate(resume["order_rng"])
            torch.set_rng_state(resume["torch_rng"])
            if dev == "cuda" and resume["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(resume["cuda_rng"])
            passes["n"], micro, seen_tok = resume["passes"], resume["micro"], resume["seen_tok"]
            history, trajectory = resume["history"], resume["trajectory"]
            t0 = time.time() - resume["elapsed"]    # s_per_step stays an average over the whole run

    def save_ckpt(sched, upd_iter: list) -> None:
        tmp, old = ck_tmp, ck_old
        if fs is not None:               # sharded: model via torch.distributed.checkpoint, optimizer + CUDA RNG per rank
            if is_main:
                shutil.rmtree(tmp, ignore_errors=True)
                shutil.rmtree(old, ignore_errors=True)
                if args.ckpt_single:
                    shutil.rmtree(ck_dir, ignore_errors=True)
                tmp.mkdir()
            fs.barrier()
            fsdp_lib.save_sharded(model, tmp / "model")
            torch.save({"optim": opt.state_dict(), "cuda_rng": torch.cuda.get_rng_state() if dev == "cuda" else None},
                       tmp / f"optim.{fs.rank}.pt")
            fs.barrier()
            if is_main:
                torch.save({"args": train_args, "step": step, "micro": micro, "seen_tok": seen_tok, "passes": passes["n"],
                            "elapsed": time.time() - t0, "sched": sched.state_dict(), "upd_iter": upd_iter,
                            "history": history, "trajectory": trajectory, "rng": rng.getstate(),
                            "order_rng": order_rng.getstate(), "torch_rng": torch.get_rng_state(), "cuda_rng": None,
                            "world": fs.world}, tmp / "state.part")
                (tmp / "state.part").rename(tmp / "state.pt")
                if ck_dir.exists():
                    ck_dir.rename(old)
                tmp.rename(ck_dir)
                shutil.rmtree(old, ignore_errors=True)
            fs.barrier()
            print(f"checkpoint {ck_dir} at step {step} ({time.time() - t_start:.0f} s into this process)", flush=True)
            return
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(old, ignore_errors=True)
        if args.ckpt_single:             # disk: one checkpoint at a time
            shutil.rmtree(ck_dir, ignore_errors=True)
        tmp.mkdir()
        torch.save(model.state_dict(), tmp / "model.pt")
        torch.save(opt.state_dict(), tmp / "optim.pt")
        torch.save({"args": train_args, "step": step, "micro": micro, "seen_tok": seen_tok, "passes": passes["n"],
                    "elapsed": time.time() - t0, "sched": sched.state_dict(), "upd_iter": upd_iter,
                    "history": history, "trajectory": trajectory, "rng": rng.getstate(),
                    "order_rng": order_rng.getstate(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if dev == "cuda" else None}, tmp / "state.part")
        (tmp / "state.part").rename(tmp / "state.pt")
        if ck_dir.exists():
            ck_dir.rename(old)
        tmp.rename(ck_dir)
        shutil.rmtree(old, ignore_errors=True)
        print(f"checkpoint {ck_dir} at step {step} ({time.time() - t_start:.0f} s into this process)", flush=True)

    def fsdp_update(upd: list[Example], norm: float) -> dict:
        """--fsdp: one optimizer update over the sequences upd, split across the ranks (cotdistill.fsdp.split_update).
        Each sequence's loss is weight x CE / norm as in the single-GPU loops and FSDP sums the ranks' gradients, so
        the update is the single-GPU one. Clips, steps and zeroes the gradients (the caller steps the scheduler).
        Returns the update's loss over all ranks, its split by kind (Tree Forcing arms), the pre-clip gradient norm,
        forward passes (all ranks, pads included), padded tokens and the timings."""
        t_a = time.time()
        parts = fsdp_lib.split_update(tok, upd, args.max_len, args.micro_bs, args.token_budget,
                                      args.token_budget_seqs, fs.world)
        acc = torch.zeros(6, dtype=torch.float64, device=fs.device)  # loss, final, node, node CE sum, nodes, padded tok
        for mb, pad, ntok in parts[fs.rank]:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                z = label_logits(model, tok, mb, args.max_len, cache)
                loss = kl_loss(z, mb, args.label_smoothing) * len(mb) / norm
                if args.lambda_brier:
                    loss = loss + args.lambda_brier * brier_loss(z, mb) * len(mb) / norm
            loss.backward()
            if pad:                      # weight 0: no loss, no gradient, only the collectives
                continue
            acc[0] += loss.detach().double()
            acc[5] += ntok
            if tree_opts:
                with torch.no_grad():
                    for zz, e in zip(z, mb):
                        ce = kl_loss([zz.detach().float()], [dataclasses.replace(e, weight=1.0)],
                                     args.label_smoothing).double()
                        acc[1 if e.kind == "final" else 2] += e.weight * ce / norm
                        if e.kind != "final":
                            acc[3] += ce
                            acc[4] += 1
        if dev == "cuda":
            torch.cuda.synchronize()
        t_b = time.time()
        dist.all_reduce(acc)
        gnorm = float(trainer.clip_grad_norm_(1.0))
        t_c = time.time()
        trainer.step()
        trainer.zero_grad()
        a = acc.tolist()
        return {"loss": a[0], "loss_final": a[1], "loss_node": a[2], "node_ce": a[3] / max(1.0, a[4]), "gnorm": gnorm,
                "fwd": len(parts[fs.rank]) * fs.world, "pad_fwd": sum(x[1] for r in parts for x in r),
                "pad_tok": int(a[5]), "fb_s": t_b - t_a, "clip_s": t_c - t_b, "opt_s": time.time() - t_c,
                **{k: v for k, v in trainer.timing.items() if k != "opt_step_s"}}

    def fsdp_row(fu: dict) -> dict:
        """--fsdp log columns: the update's loss, forwards, padded tokens, time split and per-rank memory (collective)."""
        upd_s = fu["fb_s"] + fu["clip_s"] + fu["opt_s"]
        r = {"upd_loss": fu["loss"], "fwd": fu["fwd"], "pad_fwd": fu["pad_fwd"], "pad_tok": fu["pad_tok"],
             "upd_s": round(upd_s, 3), "fb_s": round(fu["fb_s"], 3), "clip_s": round(fu["clip_s"], 3),
             "opt_s": round(fu["opt_s"], 3), "pad_tok_per_s": round(fu["pad_tok"] / max(upd_s, 1e-9), 1)}
        r.update({k: round(v, 3) for k, v in fu.items() if k in ("step_s", "d2h_wait_s")})
        r.update(fsdp_lib.rank_stats(fs.device))
        return r

    last_ckpt = time.time()
    model.train()
    for si, (name, make, _, stage_steps) in enumerate(plan):
        if args.reset_optim and si > 0:
            if trainer is not None:
                trainer.reset_state()
            else:
                opt.state.clear()
        warm = max(1, int(args.warmup * stage_steps))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s, w=warm, t=stage_steps: min(1.0, (s + 1) / w) * 0.5 * (
            1 + math.cos(math.pi * min(1.0, s / t))))
        ex_iter: list[Example] = []
        end = step + stage_steps
        upd_iter: list[list[Example]] = []
        if resume is not None:           # single-stage full-tree run; after the scheduler, whose constructor sets the LR
            if fs is None:
                opt.load_state_dict(torch.load(ck_dir / "optim.pt", map_location="cpu", weights_only=False))
            else:
                ost = torch.load(ck_dir / f"optim.{fs.rank}.pt", map_location="cpu", weights_only=False)
                opt.load_state_dict(ost["optim"])
                if ost["cuda_rng"] is not None:
                    torch.cuda.set_rng_state(ost["cuda_rng"])
            sched.load_state_dict(resume["sched"])
            step, upd_iter = resume["step"], resume["upd_iter"]
            print(f"resumed from {ck_dir} at step {step}", flush=True)
        gwin: list[float] = []           # pre-clip gradient norms since the last log row (Tree Forcing arms)
        while name == "fulltree" and step < end:
            # One problem group per update; the loss is scaled by 1 / (3 x items_per_update), the per-sequence
            # normalisation of the 1 final + 2 auxiliary grouped arms, so the learning-rate scale is comparable.
            if not upd_iter:
                upd_iter = make()
            upd, upd_iter = upd_iter[0], upd_iter[1:]
            norm, upd_loss = 3 * args.items_per_update, 0.0
            split = {"final": 0.0, "subq": 0.0, "node_ce": 0.0, "node_n": 0}
            if fs is not None:           # the same sequences split across the ranks, gradients summed
                fu = fsdp_update(upd, norm)
                upd_loss = fu["loss"]
                split.update(final=fu["loss_final"], subq=fu["loss_node"], node_ce=fu["node_ce"], node_n=1)
                seen_tok += sum(min(args.max_len, len(e.text) // 3) for e in upd)
                if tree_opts:
                    gwin.append(fu["gnorm"])
                sched.step()
            else:
                # micro_bs chunks, or length-sorted micro-batches of at most token_budget padded tokens
                mbs = micro_batches(tok, upd, args.max_len, args.micro_bs, args.token_budget, args.token_budget_seqs)
                for mb in mbs:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                        z = label_logits(model, tok, mb, args.max_len, cache)
                        loss = kl_loss(z, mb, args.label_smoothing) * len(mb) / norm
                        if args.lambda_brier:          # as in the grouped loop (was silently ignored in full-tree arms)
                            loss = loss + args.lambda_brier * brier_loss(z, mb) * len(mb) / norm
                    loss.backward()
                    upd_loss += loss.item()
                    seen_tok += sum(min(args.max_len, len(e.text) // 3) for e in mb)
                    if tree_opts:                      # Tree Forcing arms: the loss split by kind (detached, no gradient)
                        with torch.no_grad():
                            for zz, e in zip(z, mb):
                                ce = kl_loss([zz.detach().float()], [dataclasses.replace(e, weight=1.0)],
                                             args.label_smoothing).item()
                                split[e.kind if e.kind in split else "subq"] += e.weight * ce / norm
                                if e.kind != "final":
                                    split["node_ce"] += ce
                                    split["node_n"] += 1
                gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if tree_opts:                          # pre-clip gradient norm (clipped at 1.0), per update
                    gwin.append(float(gnorm))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0 or step == total_steps:
                dt = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0.0
                row = {"step": step, "loss": upd_loss, "lr": float(sched.get_last_lr()[0]), "s_per_step": dt / step,
                       "approx_tok_per_s": seen_tok / dt, "peak_gib": mem, "seqs": len(upd)}
                if fs is not None:
                    row.update(fsdp_row(fu))
                elif args.token_budget:
                    row["fwd"] = len(mbs)
                if tree_opts:      # this update's loss by kind (weighted; the two sum to loss without Brier), its
                    # mean unweighted node CE, and the pre-clip gradient norms of the updates since the last row
                    row.update(loss_final=split["final"], loss_node=split["subq"],
                               node_ce=split["node_ce"] / max(1, split["node_n"]),
                               gnorm=sum(gwin) / len(gwin), gnorm_max=max(gwin),
                               clip_frac=sum(g > 1.0 for g in gwin) / len(gwin))
                    gwin.clear()
                history.append(row)
                print(json.dumps(row), flush=True)
            if step in eval_steps:
                mid_eval(step)
            if resumable and step < total_steps:
                stop = bool(args.stop_at_step and step >= args.stop_at_step or
                            args.stop_after_h and time.time() - t_start >= args.stop_after_h * 3600)
                ck_now = stop or args.ckpt_every_min and time.time() - last_ckpt >= args.ckpt_every_min * 60
                if fs is not None:       # clock-based decisions: every rank follows rank 0's
                    flags = torch.tensor([int(stop), int(bool(ck_now))], device=fs.device)
                    dist.broadcast(flags, 0)
                    stop, ck_now = bool(flags[0]), bool(flags[1])
                if ck_now:
                    save_ckpt(sched, upd_iter)
                    last_ckpt = time.time()
                if stop:
                    print(f"planned stop after step {step}/{total_steps}; rerun with the same --out to resume", flush=True)
                    if fs is not None:
                        if is_main:
                            stop_mark.write_text(f"{step}\n")
                        fs.barrier()
                        dist.destroy_process_group()
                    sys.exit(3)
        upd_acc = 0.0                    # --log-update-loss: the current update's loss so far
        while step < end:
            if fs is not None:           # one whole update: the grad_accum micro-batches the loop below takes, split
                upd = []                 # across the ranks (same refills, so the same sequences in the same updates)
                for _ in range(args.grad_accum):
                    if len(ex_iter) < args.micro_bs:
                        ex_iter += make()
                    batch, ex_iter = ex_iter[:args.micro_bs], ex_iter[args.micro_bs:]
                    upd += batch
                fu = fsdp_update(upd, args.micro_bs * args.grad_accum)   # = len(batch) x grad_accum below
                seen_tok += sum(min(args.max_len, len(e.text) // 3) for e in upd)
                micro += args.grad_accum
                sched.step()
                step += 1
                if step % args.log_every == 0 or step == total_steps:
                    dt = time.time() - t0
                    mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0.0
                    row = {"step": step, "loss": fu["loss"], "lr": float(sched.get_last_lr()[0]), "s_per_step": dt / step,
                           "approx_tok_per_s": seen_tok / dt, "peak_gib": mem, "gnorm": fu["gnorm"]}
                    row.update(fsdp_row(fu))
                    history.append(row)
                    print(json.dumps(row), flush=True)
                if step in eval_steps:
                    mid_eval(step)
                continue
            if len(ex_iter) < args.micro_bs:
                ex_iter += make()
            batch, ex_iter = ex_iter[:args.micro_bs], ex_iter[args.micro_bs:]
            cls = [e for e in batch if e.kind != "lm"]
            lms = [e for e in batch if e.kind == "lm"]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                total = 0.0
                if cls:
                    z = label_logits(model, tok, cls, args.max_len, cache)
                    total = kl_loss(z, cls, args.label_smoothing) * len(cls)
                    if args.lambda_brier:
                        total = total + args.lambda_brier * brier_loss(z, cls) * len(cls)
                if lms:
                    total = total + lm_loss(model, tok, lms, args.lm_max_len)
                loss = total / len(batch) / args.grad_accum
            loss.backward()
            if args.log_update_loss:     # the whole update's loss (the --fsdp loss column), for step-by-step comparisons
                upd_acc += loss.item()
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
                    row = {"step": step, "loss": loss.item() * args.grad_accum, "lr": float(sched.get_last_lr()[0]),
                           "s_per_step": dt / step, "approx_tok_per_s": seen_tok / dt, "peak_gib": mem}
                    if args.log_update_loss:
                        row["upd_loss"] = upd_acc
                    history.append(row)
                    print(json.dumps(row), flush=True)
                upd_acc = 0.0
                if step in eval_steps:
                    mid_eval(step)

    if trainer is not None:              # --fsdp: GPU 8-bit states or the CPU masters and moments
        trainer.release()
    else:
        opt.state.clear()                # the optimizer state is not needed for the evaluations (4B: about 9 GiB)
    if dev == "cuda":
        torch.cuda.empty_cache()
    results = {"args": vars(args), "history": history, "trajectory": trajectory, "eval": {}}
    if tree_opts:
        results["tree_view_stats"] = tree_stats
    if fs is not None:
        results["fsdp"] = {"world": fs.world, "optim": args.fsdp_optim}
    for name, items in evals.items():    # (--fsdp: every rank evaluates its share of the batches; rank 0 writes)
        hid = [] if name in args.dump_hidden.split(",") else None
        m, preds = evaluate(model, tok, items, args.eval_max_len, args.eval_bs, cache, hidden=hid,
                            layers=[int(x) for x in args.dump_layers.split(",") if x] or None)
        if hid is not None and is_main:
            torch.save(torch.stack(hid).half(), out / f"hidden_{name}.pt")
        results["eval"][name] = m
        if is_main:
            with open(out / f"preds_{name}.jsonl", "w") as f:
                for p in preds:
                    f.write(json.dumps(p) + "\n")
        print(name, json.dumps(m), flush=True)
    for sname, spath in subq_sets(args.eval_subq or ""):
        sm, spreds = evaluate_subq(model, tok, read_jsonl(spath), args.max_len, args.eval_bs, cache, seed=args.seed)
        results["eval"][sname] = sm
        if is_main:
            with open(out / f"preds_{sname}.jsonl", "w") as f:
                for p in spreds:
                    f.write(json.dumps(p) + "\n")
        print(sname, json.dumps({k: round(v["acc"], 3) for k, v in sm.items()}), flush=True)
    if args.eval_local:
        lname = "local_kk" if args.eval_local_view == "local" else "local_kk_anc"
        lm, lpreds = evaluate_local(model, tok, read_jsonl(args.eval_local), args.max_len, args.eval_bs, cache,
                                    view=args.eval_local_view, seed=args.seed)
        results["eval"][lname] = lm
        if is_main:
            with open(out / f"preds_{lname}.jsonl", "w") as f:
                for p in lpreds:
                    f.write(json.dumps(p) + "\n")
        print(lname, json.dumps({k: round(v["acc"], 3) for k, v in lm.items() if not k.startswith("depth/")}), flush=True)
    if fs is not None:
        if is_main:
            (out / "metrics.json").write_text(json.dumps(results, indent=1))
            if resumable:                # quota: the sharded checkpoint goes before the full model is written
                shutil.rmtree(ck_dir, ignore_errors=True)
        if args.save:                    # full bf16 state dict gathered on rank 0's CPU (collective), rank 0 writes it
            sd = fsdp_lib.full_state_dict(model)
            if is_main:
                model.model.save_pretrained(out / "model", state_dict=sd, safe_serialization=True)
                tok.save_pretrained(out / "model")
            del sd
        fs.barrier()
        dist.destroy_process_group()
        return
    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    if args.save:
        if args.lora_r:                  # adapter for reuse, merged weights so <out>/model loads like a full checkpoint
            model.save_pretrained(out / "adapter")
            model = model.merge_and_unload()
        model.to(torch.bfloat16).save_pretrained(out / "model", safe_serialization=True)
        tok.save_pretrained(out / "model")
    if resumable:                        # finished: metrics (and weights) are written, the checkpoint goes
        shutil.rmtree(ck_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
