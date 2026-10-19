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
                                build_stage_examples, curriculum_item, evaluate, evaluate_local, evaluate_subq, kl_loss,
                                label_logits,
                                lm_loss, load_model, logit_softcap, micro_batches, pack_rng, pack_views, readouts,
                                scaffold_fade, si_group, slot_forward, slot_group, special_prefix, subq_sets,
                                use_chat_template, worked_steps)


# Arguments that do not change the training trajectory: a checkpoint may be resumed with different values.
RESUME_FREE = {"stop_after_h", "ckpt_every_min", "ckpt_single", "stop_at_step", "code_commit", "eval", "eval_subq", "eval_bs",
               "eval_max_len", "dump_hidden", "dump_layers", "log_every", "save", "eval_local", "eval_local_view",
               "log_update_loss", "eval_mid_preds", "slot_necessity", "slot_necessity_sets", "slot_swap_token",
               "slot_decode", "slot_decode_set"}

# Training arguments added after checkpoints may have been written: a checkpoint without the key had this value.
NEW_ARG_DEFAULTS = {"tree_view": "problem", "tree_sources": "", "tree_node_weight": 0.0, "attn_impl": "", "fsdp": False,
                    "fsdp_optim": None, "tree_mix_p": 0.5, "tree_scaffold": 0.0, "tree_depth_weight": 0.0,
                    "chat_template": False, "tree_pack": 0, "tree_node_types": "", "tree_target": "real",
                    "node_render": "op", "si_stages": 0, "si_chain": "true", "si_final_only": False, "si_smooth": 0.1,
                    "si_schedule": "global", "si_keep_optim": False, "si_remove": "front", "curriculum_steps": 0,
                    "curriculum_mode": "cumulative", "curriculum_final_nodes": False, "latent_slots": 0, "slot_c": 1,
                    "slot_token": "auto", "slot_place": "before-options", "slot_lambda": 1.0, "slot_target": "tree",
                    "slot_overflow": "merge", "slot_final_only": False}

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
    ap.add_argument("--tree-view", default="problem",
                    choices=["problem", "local", "ancestors", "statements", "statements+question", "local+ctxparse", "mix",
                             "bare"],
                    help="full-tree groups: how a node is asked. problem = with the original problem (the TF/TFM/GF "
                         "arms); local = Tree Forcing: no problem text, only the preamble and the parents' results "
                         "(--tree-full true) or as many matched non-parent results (matched) or nothing (plain: "
                         "TFLN, in rng lockstep with TFL); parse nodes show their statement line(s); ancestors = every "
                         "transitive ancestor's result, shallow to deep (true only). Needs the local fields of "
                         "data/student_v3sl (scripts/build_local_kk.py). statements = GFP-S (plain only): the problem "
                         "view without the final question and its options (K&K: preamble + statement lines; numgen: "
                         "the problem text without its question sentence), in rng lockstep with GFP. local+ctxparse = "
                         "TFLP (true only): TFL with the parse nodes asked in the statements view; mix = MIX (true "
                         "only): each node asked once per epoch, in the statements view with probability "
                         "--tree-mix-p (a coin per seed, epoch, item and node), else in its TFL view; both in rng "
                         "lockstep with TFL; bare = GFPVB (plain, --node-format value only): the node question "
                         "with its own options and nothing else (no problem text)")
    ap.add_argument("--tree-mix-p", type=float, default=0.5,
                    help="--tree-view mix: share of node views asked in the statements view (the rest local)")
    ap.add_argument("--tree-scaffold", type=float, default=0.0,
                    help="GTF (--tree-full plain --tree-view statements only): scaffold fading over the first fraction "
                         "F of the updates: at update t of T, a derived node is asked with its parents' results stated "
                         "('Known results:' under the statement lines) with probability max(0, 1 - t / (F x T)), else "
                         "in the plain GFP-S view; a coin per seed, epoch, item and node (not the training rng, so the "
                         "run stays step-matched with GFP-S); 0 = off")
    ap.add_argument("--tree-depth-weight", type=float, default=0.0,
                    help="full-tree groups: split each problem's node-view weight (the same total) in proportion to "
                         "(1 + node depth)^G instead of evenly; 0 = off (GTF uses 1)")
    ap.add_argument("--tree-pack", type=int, default=0,
                    help="GFPSP (--tree-full plain --tree-view statements only): read up to N >= 2 node views of a "
                         "problem in one sequence (statement block, then each node question with its options and "
                         "'Answer:' cue, no answers), read out at every cue; nodes in a seeded order per seed, epoch and "
                         "problem (not the training rng: same nodes, option orders, targets and weights as GFPS); a new "
                         "pack (re-reading the block) when N or --max-len is reached; the final question stays its own "
                         "sequence; 0 = off")
    ap.add_argument("--tree-sources", default="",
                    help="full-tree groups: comma list of item sources that get node views; other items train their "
                         "final question only (every view mode); empty = all sources")
    ap.add_argument("--tree-node-types", default="",
                    help="full-tree groups: comma list of node types (the subq field type, e.g. parse,derive: GFPSNC = "
                         "GFP-S without the K&K case nodes) that get node views; nodes of other types are not asked "
                         "(their results may still be stated as parents in --tree-full true / matched / GTF); the "
                         "per-problem weight convention is unchanged, so the kept views share the same total; only the "
                         "items of --tree-sources are affected; empty = every node type")
    ap.add_argument("--tree-target", default="real", choices=["real", "shuffled", "placebo"],
                    help="full-tree groups: the node targets. real = the data's (--subq-target); shuffled = GFPSR / "
                         "GFPQR, the random-label control (--tree-full plain, views problem / statements / "
                         "statements+question, yes/no nodes): each problem's node targets permuted among its nodes "
                         "(the yes rate per problem unchanged), a multiple-choice node's answer drawn uniformly, by "
                         "generators keyed by --seed and the item (not the training rng: the same nodes, order, option "
                         "orders and weights as the real-target arm; fixed over the epochs); the final question is "
                         "untouched. placebo = GFPSX / GFPQX, the placebo control (same scope): each node question "
                         "replaced by a surface question about the problem text with a correct answer and the node's "
                         "own yes/no (is a name / number / word / sentence in the problem; never a tree token or a "
                         "derived number; student.placebo_questions, keyed by --seed and the item, fixed over the "
                         "epochs; the \"no\" sentences come from other training problems of the same source); same "
                         "views, weights, option orders and targets as the real-target arm (unrelated to --tree-full "
                         "placebo, the zero-weight views)")
    ap.add_argument("--tree-node-weight", type=float, default=0.0,
                    help="full-tree groups: loss weight of every node view (times the item weight); 0 = the per-problem "
                         "convention, --aux-total split evenly over the problem's views")
    ap.add_argument("--node-format", default="yesno", choices=["yesno", "mc", "value", "yesno+value"],
                    help="full-tree groups: ask nodes as yes/no questions, or in multiple-choice form with other "
                         "confident No nodes of the item and 'None of them' as options (same stated results); value = "
                         "the item's value nodes (field value_nodes, scripts/build_gsm8k_ops.py: one node per atomic "
                         "operation, three options, one-hot target) in the rendering --node-render instead of the "
                         "yes/no nodes (GFPQV / GFPQVQ / GFPQO / GFPVB; --tree-full plain, views problem / statements / "
                         "statements+question / bare); yesno+value = GFPQVM: the yes/no views exactly as yesno, plus "
                         "the value views, each format carrying half of the per-problem --aux-total (a problem with "
                         "one format only gives it the whole)")
    ap.add_argument("--node-render", default="op", choices=["op", "qop", "opsel"],
                    help="--node-format value / yesno+value: the value-node rendering: op = the bare operation "
                         "('What is 4 x 1.5?'), qop = the Socratic sub-question plus the operation, opsel = operation "
                         "selection ('<sub-question> Which computation gives it?', three expressions); nodes without "
                         "the rendering are not asked. --eval-subq scores the value nodes in this rendering")
    ap.add_argument("--si-stages", type=int, default=0,
                    help="stepwise internalisation (SI / SIY / SIS; Deng et al. 2024, ICoT-SI; full-tree loop): the "
                         "final question is trained with the problem's worked steps written between the problem and "
                         "its options ('Worked steps:' / 'Step 1: 4 + 2 = 6', student.si_group; the item field "
                         "worked_steps of data/student_gsm8k_tf_si*, or a numgen item's program), over S + 1 stages "
                         "of --epochs / (S + 1) epochs each; stage k removes the first ceil(k n / S) of the n steps, so "
                         "stage S trains the test format; the node views of the group are build_full_tree's (SIY: "
                         "GFPQ's) unless --si-final-only; evaluations stay final-question-only, with a mid-training "
                         "evaluation (--eval-at mechanism, --eval-mid sets) at the end of every stage but the last; "
                         "0 = off")
    ap.add_argument("--si-chain", default="true", choices=["true", "wrong", "wrong-results"],
                    help="--si-stages: the written steps. true = the solution's operations; wrong = SIS, the "
                         "wrong-content control: every computed value written wrongly (each result a distractor of "
                         "the same size, never the truth, and an operand that is an earlier result that result's wrong "
                         "value), the same lines and counts; wrong-results = only the results wrong")
    ap.add_argument("--si-final-only", action="store_true",
                    help="--si-stages: train the final question only (SI / SIS); off = also the node views of the "
                         "full-tree group (SIY with GFPQ's yes/no views)")
    ap.add_argument("--si-smooth", type=float, default=0.1,
                    help="--si-stages: removal smoothing, the probability of removing one more step (a coin per "
                         "seed, stage, pass and item, not the training rng)")
    ap.add_argument("--si-schedule", default="global", choices=["global", "stage"],
                    help="--si-stages: one warmup + cosine schedule over all stages (global) or a fresh one per "
                         "stage (stage)")
    ap.add_argument("--si-keep-optim", action="store_true",
                    help="--si-stages: keep the AdamW state across stages (default: reset at every stage, as ICoT-SI)")
    ap.add_argument("--si-remove", default="front", choices=["front", "back"],
                    help="--si-stages: remove the steps from the front (ICoT-SI; the kept steps keep their numbers) "
                         "or from the back (the steps nearest the answer first)")
    ap.add_argument("--curriculum-steps", type=int, default=0,
                    help="step curriculum (GFPQC; full-tree loop, --tree-full plain, yes/no nodes): K node stages, "
                         "then the final questions. Node stage j (1..K) trains only the node views (the arm's view, "
                         "e.g. GFPQ's) of the nodes whose solution step (node field step: GSM8K's Socratic step "
                         "index) is <= j (--curriculum-mode cumulative) or == j (exact), every step beyond K counted "
                         "as K (no node is left out); no final question in the node stages; each problem's node "
                         "views share --aux-total (the --tree-node-types convention); problems without a node of the "
                         "stage are not in it. The last stage trains the final questions only (--curriculum-final-"
                         "nodes: with all their node views, GFPQ's groups). --epochs is the total, split evenly "
                         "(epochs / (K + 1) passes per stage, each stage counted in its own updates per pass); one "
                         "warmup + cosine schedule over all stages; the AdamW state is kept across stages "
                         "(--reset-optim resets it); a mid-training evaluation (--eval-mid sets) at the end of every "
                         "stage but the last, no --eval-at; 0 = off")
    ap.add_argument("--curriculum-mode", default="cumulative", choices=["cumulative", "exact"],
                    help="--curriculum-steps: node stage j asks the steps <= j (cumulative) or == j (exact)")
    ap.add_argument("--curriculum-final-nodes", action="store_true",
                    help="--curriculum-steps: the last stage trains the final questions with all their node views "
                         "(GFPQ's groups) instead of the final questions only")
    ap.add_argument("--latent-slots", type=int, default=0,
                    help="latent node slots (Tree Forcing v2, design (a); LS / LSU / LSR / LSY; full-tree loop, "
                         "--tree-full plain, yes/no nodes): K slots of --slot-c reserved tokens each in every final "
                         "question's prompt (--slot-place), in training and in every evaluation (one forward pass, the "
                         "same letter readout); in training only, slot j's final-layer states start a separate "
                         "sequence through the same network and head that must continue with the problem's j-th worked "
                         "step ('4 + 2 = 6'; field worked_steps, data/student_gsm8k_tf_si*), loss = letter CE + "
                         "--slot-lambda x the mean slot decoding CE per problem; 0 = off")
    ap.add_argument("--slot-c", type=int, default=1, help="--latent-slots: tokens (vectors) per slot")
    ap.add_argument("--slot-token", default="auto",
                    help="--latent-slots: the slot token, an existing single token of the tokenizer (auto = the first "
                         "of <|box_start|>, <|quad_start|>, ... that is one; Qwen3.5: <|box_start|>)")
    ap.add_argument("--slot-place", default="before-options", choices=["before-options", "after-options"],
                    help="--latent-slots: the slot line between the problem and its options (before-options: the "
                         "slots see the problem only; prompts without an options block get it at the end) or after "
                         "the options, right before the 'Answer:' cue")
    ap.add_argument("--slot-lambda", type=float, default=1.0,
                    help="--latent-slots: weight of the slot decoding loss per problem (split over its supervised "
                         "slots, as --aux-total over the node views)")
    ap.add_argument("--slot-target", default="tree", choices=["tree", "none", "shuffled"],
                    help="--latent-slots: tree = slot j decodes the j-th worked step (LS / LSY); none = the slots without "
                         "any slot loss (LSU, the pause-token control; no decoding pass); shuffled = LSR, the "
                         "wrong-content control: the same supervised slots, texts drawn from other problems' slot "
                         "targets (keyed by --seed and the item, fixed over the epochs)")
    ap.add_argument("--slot-overflow", default="merge", choices=["merge", "truncate"],
                    help="--latent-slots: a chain longer than K: merge = steps K..n joined into slot K (the answer's "
                         "step stays in), truncate = the first K steps")
    ap.add_argument("--slot-final-only", action="store_true",
                    help="--latent-slots: train the final questions only (LS / LSU / LSR); off = also the node views of "
                         "the full-tree group (LSY with GFPQ's yes/no views, which carry no slots)")
    ap.add_argument("--slot-necessity", default="",
                    help="--latent-slots, after training: the necessity test, a comma list of modes, each evaluated on "
                         "--slot-necessity-sets next to the intact slots: swap (every slot token replaced by "
                         "--slot-swap-token), drop (no slot line, the A2 prompt), noise@<layer|mid>:<sigma> (Gaussian "
                         "noise of sigma x RMS on the slot states after that decoder layer); eval keys <set>@<mode>, "
                         "metrics slot_necessity, preds_<set>@<mode>.jsonl")
    ap.add_argument("--slot-necessity-sets", default="", help="--slot-necessity: comma list of eval sets (default all)")
    ap.add_argument("--slot-swap-token", default="auto",
                    help="--slot-necessity swap: the replacement token (auto = the next reserved token, <|quad_start|>)")
    ap.add_argument("--slot-decode", type=int, default=0,
                    help="--latent-slots, after training: decode every slot of the first N items of --slot-decode-set "
                         "back to text greedily (same separate-sequence path): slot_decode_<set>.jsonl and "
                         "metrics slot_decode (exact / value match by slot)")
    ap.add_argument("--slot-decode-set", default="", help="--slot-decode: the eval set (default: the first --eval set)")
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
                    help="length-sorted micro-batches of at most N padded tokens (and at most --token-budget-seqs "
                         "sequences) instead of --micro-bs chunks; same gradient per update; 0 = off. Full-tree loop: an "
                         "update is a problem group. Other loops: an update is still the --micro-bs x --grad-accum "
                         "examples of the fixed loop (same refills), only regrouped by length (not with --rationale-lm)")
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
    ap.add_argument("--chat-template", action="store_true",
                    help="train and evaluate every view as one user turn of the model's chat template, thinking off "
                         "(the text without its 'Answer:' cue + 'Answer with the letter of the correct option only.', "
                         "as scripts/eval_llm_baseline.py), read out at the assistant turn's first token (bare 'A', "
                         "not ' A'); off: unchanged")
    ap.add_argument("--eval-bs", type=int, default=8)
    ap.add_argument("--eval-max-len", type=int, default=6144, help="long eval states (JevBench hard) need more room")
    ap.add_argument("--save", action="store_true", help="save the final weights (bf16) to <out>/model")
    ap.add_argument("--eval-at", default="", help="training-length curve: comma list of points, in epochs, at which "
                    "the --eval-mid sets are evaluated during training (e.g. 0.5,1,2,4); stored as metrics.json trajectory")
    ap.add_argument("--eval-mid", default="", help="comma list of --eval set names for the --eval-at points (default all)")
    ap.add_argument("--eval-mid-preds", action="store_true",
                    help="also write each --eval-at point's predictions on the --eval-mid sets to "
                         "<out>/mid/preds_<name>_e<epoch>.jsonl (checkpoint selection on a val set, e.g. "
                         "scripts/fa_readout.py --sf); off: nothing is written")
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
    tree_opts = (args.tree_view != "problem" or args.tree_sources or args.tree_node_weight or args.tree_cap == 0 or
                 args.tree_scaffold or args.tree_depth_weight or args.tree_pack or args.tree_node_types or
                 args.tree_target != "real" or args.latent_slots)
    if tree_opts and not args.tree_full:
        raise SystemExit("--tree-view / --tree-sources / --tree-node-weight / --tree-cap 0 / --tree-scaffold / "
                         "--tree-depth-weight / --tree-pack / --tree-node-types / --tree-target / --latent-slots need "
                         "--tree-full")
    if args.tree_target != "real" and (args.tree_full != "plain" or args.node_format != "yesno" or
                                       args.tree_view not in ("problem", "statements", "statements+question") or
                                       args.tree_eligible_only or args.tree_scaffold):
        raise SystemExit(f"--tree-target {args.tree_target}: for --tree-full plain (GF / GFP / GFPS / GFPQ: views "
                         "problem, statements, statements+question) with yes/no nodes, without --tree-eligible-only "
                         "or --tree-scaffold")
    if args.tree_pack and (args.tree_pack < 2 or (args.tree_full, args.tree_view) != ("plain", "statements") or
                           args.tree_scaffold or args.fsdp or args.chat_template):
        raise SystemExit(f"--tree-pack {args.tree_pack}: at least 2 views per pack, for --tree-full plain --tree-view "
                         "statements (GFPS) without --tree-scaffold, --fsdp or --chat-template")
    value_fmt = args.node_format in ("value", "yesno+value")
    if args.node_render != "op" and not value_fmt:
        raise SystemExit(f"--node-render {args.node_render}: for --node-format value / yesno+value")
    if value_fmt and (args.tree_full != "plain" or args.tree_view not in ("problem", "statements", "statements+question")
                      + (("bare",) if args.node_format == "value" else ()) or args.tree_eligible_only or
                      args.tree_scaffold or args.tree_depth_weight or args.tree_pack or args.tree_node_types or
                      args.tree_target != "real"):
        raise SystemExit(f"--node-format {args.node_format}: for --tree-full plain, views problem / statements / "
                         "statements+question (and bare for value), without --tree-eligible-only, --tree-scaffold, "
                         "--tree-depth-weight, --tree-pack, --tree-node-types or --tree-target")
    if args.tree_view == "bare" and args.node_format != "value":
        raise SystemExit("--tree-view bare (GFPVB): for --tree-full plain --node-format value")
    if args.tree_view != "problem" and not (args.tree_full == "true" and args.tree_view not in ("statements", "statements+question", "bare") or
                                             args.tree_full in ("matched", "plain") and args.tree_view == "local" or
                                             args.tree_full == "plain" and args.tree_view in ("statements", "statements+question", "bare")):
        raise SystemExit(f"--tree-view {args.tree_view} is implemented for --tree-full true (local/ancestors/"
                         "local+ctxparse/mix), matched (local) and plain (local, statements)")
    if args.tree_view != "problem" and (args.tree_eligible_only or args.node_format not in ("yesno", "value", "yesno+value")):
        raise SystemExit("--tree-view local/ancestors/statements/local+ctxparse/mix does not combine with "
                         "--tree-eligible-only / --node-format mc")
    if not 0.0 <= args.tree_mix_p <= 1.0 or args.tree_mix_p != 0.5 and args.tree_view != "mix":
        raise SystemExit(f"--tree-mix-p {args.tree_mix_p}: a probability, for --tree-view mix only")
    if args.tree_scaffold < 0 or args.tree_scaffold and (args.tree_full, args.tree_view) != ("plain", "statements"):
        raise SystemExit(f"--tree-scaffold {args.tree_scaffold}: a fraction > 0 of the updates, for --tree-full plain "
                         "--tree-view statements only (GTF)")
    if args.tree_depth_weight and args.tree_full == "control":
        raise SystemExit("--tree-depth-weight needs tree nodes: control questions have no depth")
    node_types = frozenset(x.strip() for x in args.tree_node_types.split(",") if x.strip()) or None
    if args.tree_node_types and (node_types is None or args.tree_full == "control"):
        raise SystemExit(f"--tree-node-types {args.tree_node_types!r}: a comma list of node types, for tree nodes "
                         "(not --tree-full control: control questions have no type)")
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
    if resumable and not args.tree_full:
        raise SystemExit("--stop-after-h / --ckpt-every-min / --stop-at-step are implemented for the full-tree loop "
                         "only (--tree-full)")
    if args.token_budget and args.rationale_lm and not args.tree_full:
        raise SystemExit("--token-budget does not combine with --rationale-lm (LM continuations are not regrouped)")
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
    si_flags = (args.si_chain != "true" or args.si_final_only or args.si_smooth != 0.1 or args.si_schedule != "global"
                or args.si_keep_optim or args.si_remove != "front")
    if args.si_stages < 0 or si_flags and not args.si_stages:
        raise SystemExit("--si-chain / --si-final-only / --si-smooth / --si-schedule / --si-keep-optim / --si-remove "
                         "need --si-stages S >= 1")
    if args.si_stages and (not args.tree_full or args.tree_pack or resumable or args.depth_stages or args.fact_stages
                           or args.facts_always or not 0.0 <= args.si_smooth <= 1.0):
        raise SystemExit(f"--si-stages {args.si_stages}: for the full-tree loop (--tree-full; SI / SIS with "
                         "--si-final-only), without --tree-pack, --stop-after-h / --ckpt-every-min / --stop-at-step "
                         "(no multi-stage resume), --depth-stages, --fact-stages or --facts-always; --si-smooth a "
                         "probability")
    if args.curriculum_steps < 0 or ((args.curriculum_mode != "cumulative" or args.curriculum_final_nodes)
                                     and not args.curriculum_steps):
        raise SystemExit("--curriculum-mode / --curriculum-final-nodes need --curriculum-steps K >= 1")
    if args.curriculum_steps and (args.tree_full != "plain" or args.node_format != "yesno" or
                                  args.tree_view not in ("problem", "statements", "statements+question") or
                                  args.si_stages or args.tree_pack or resumable or args.depth_stages or
                                  args.fact_stages or args.facts_always or args.tree_eligible_only or
                                  args.tree_scaffold or args.tree_depth_weight or args.tree_target != "real" or
                                  args.eval_at):
        raise SystemExit(f"--curriculum-steps {args.curriculum_steps}: for the full-tree loop with --tree-full plain, "
                         "yes/no nodes (views problem / statements / statements+question), without --si-stages, "
                         "--tree-pack, --stop-after-h / --ckpt-every-min / --stop-at-step (no multi-stage resume), "
                         "--depth-stages, --fact-stages, --facts-always, --tree-eligible-only, --tree-scaffold, "
                         "--tree-depth-weight or --tree-target; no --eval-at (it evaluates at its stage ends)")
    slot_flags = (args.slot_c != 1 or args.slot_token != "auto" or args.slot_place != "before-options" or
                  args.slot_lambda != 1.0 or args.slot_target != "tree" or args.slot_overflow != "merge" or
                  args.slot_final_only or args.slot_necessity or args.slot_necessity_sets or
                  args.slot_swap_token != "auto" or args.slot_decode or args.slot_decode_set)
    if args.latent_slots < 0 or slot_flags and not args.latent_slots:
        raise SystemExit("--slot-* options need --latent-slots K >= 1")
    if args.latent_slots and (args.tree_full != "plain" or args.node_format != "yesno" or
                              args.tree_view not in ("problem", "statements", "statements+question") or
                              args.si_stages or args.curriculum_steps or args.tree_pack or args.tree_scaffold or
                              args.tree_target != "real" or args.tree_eligible_only or args.chat_template or
                              args.depth_stages or args.fact_stages or args.facts_always or args.slot_c < 1 or
                              args.slot_lambda < 0 or args.slot_decode < 0):
        raise SystemExit(f"--latent-slots {args.latent_slots}: for the full-tree loop with --tree-full plain and yes/no "
                         "nodes (views problem / statements / statements+question), without --si-stages, "
                         "--curriculum-steps, --tree-pack, --tree-scaffold, --tree-target, --tree-eligible-only, "
                         "--chat-template, --depth-stages, --fact-stages or --facts-always; --slot-c >= 1, "
                         "--slot-lambda >= 0, --slot-decode >= 0")
    if args.latent_slots:
        from cotdistill.student import parse_slot_necessity
        try:
            parse_slot_necessity(args.slot_necessity)
        except ValueError as e:
            raise SystemExit(str(e))
    if args.ckpt_single and not resumable:
        raise SystemExit("--ckpt-single needs --stop-after-h, --ckpt-every-min or --stop-at-step")
    if args.chat_template and args.rationale_lm:
        raise SystemExit("--chat-template does not combine with --rationale-lm (the LM continuation is not a chat turn)")
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
    if args.chat_template:
        print("chat template: every view rendered as", repr(use_chat_template(tok, args.model)), flush=True)
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
    slot = None                      # --latent-slots: the slot spec (registered for every final question evaluate builds)
    slot_tg: dict = {}               # item_id -> slot targets (LS / LSY: the worked steps; LSR: other problems' steps)
    slot_loss_on = False             # a decoding pass (LSU and --slot-lambda 0: none)
    slot_stats: dict = {}
    swap_token = None
    if args.latent_slots:
        from cotdistill.student import (SlotSpec, choose_slot_token, final_text, set_slot_prompt, shuffled_slot_targets,
                                        slot_targets, slot_token_id)
        try:
            token = choose_slot_token(tok, args.slot_token)
            if "swap" in parse_slot_necessity(args.slot_necessity):
                swap_token = choose_slot_token(tok, args.slot_swap_token, avoid=(token,))
                if swap_token == token:
                    raise ValueError(f"--slot-swap-token {swap_token!r} is the slot token")
        except ValueError as e:
            raise SystemExit(f"--latent-slots: {e}")
        for name, items in [("train", train_items)] + list(evals.items()):
            bad = [it["item_id"] for it in items if token in it["prompt"] or swap_token and swap_token in it["prompt"]]
            if bad:
                raise SystemExit(f"--latent-slots: the slot (or swap) token occurs in {len(bad)} {name} prompts, e.g. "
                                 f"{bad[0]}: pick another --slot-token / --slot-swap-token")
        slot = SlotSpec(args.latent_slots, args.slot_c, token, args.slot_place)
        set_slot_prompt(tok, slot)
        srcs = {x for x in args.tree_sources.split(",") if x} or None     # supervision: the --tree-sources items
        sup_items = [it for it in train_items if srcs is None or it.get("source") in srcs]
        if args.slot_target == "shuffled":
            slot_tg = shuffled_slot_targets(sup_items, k=slot.k, overflow=args.slot_overflow, seed=args.seed)
        elif args.slot_target == "tree":
            slot_tg = {it["item_id"]: slot_targets(it, slot.k, args.slot_overflow) for it in sup_items}
        n_sup = sum(t is not None for ts in slot_tg.values() for t in ts)
        if args.slot_target != "none" and not n_sup:
            raise SystemExit(f"--latent-slots: no {args.tree_sources or 'training'} item of {args.train} has worked "
                             "steps (field worked_steps: scripts/build_gsm8k_si.py, or a numgen program)")
        slot_loss_on = args.slot_target != "none" and args.slot_lambda > 0
        print(f"latent slots: {slot.k} x {slot.c} {token!r} (id {slot_token_id(tok, token)}), {slot.place}, target "
              f"{args.slot_target}, lambda {args.slot_lambda}, overflow {args.slot_overflow}, "
              f"{'final views only' if args.slot_final_only else 'with the node views'}; {n_sup} supervised slots over "
              f"{sum(any(t is not None for t in ts) for ts in slot_tg.values())} items; decoding pass "
              f"{'on' if slot_loss_on else 'off'}; final question e.g. {final_text(tok, train_items[0]['prompt'])!r}",
              flush=True)

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
    if node_types:                           # every listed type must occur among the nodes that may be asked
        have = {sq.get("type") for it in train_items if tree_sources is None or it.get("source") in tree_sources
                for sq in it.get("subqs") or [] if "id" in sq}
        if node_types - have:
            raise SystemExit(f"--tree-node-types {args.tree_node_types}: no node of type(s) "
                             f"{','.join(sorted(node_types - have))} in the {args.tree_sources or 'tree'} items of "
                             f"{args.train} (types there: {','.join(sorted(str(x) for x in have))})")
    if value_fmt:                            # GFPQV / GFPQVQ / GFPQO / GFPVB / GFPQVM: value nodes with the rendering
        from cotdistill.student import VALUE_FIELD                            # must exist in the tree items
        n_val = sum(bool((sq.get("renderings") or {}).get(args.node_render)) for it in train_items
                    if tree_sources is None or it.get("source") in tree_sources for sq in it.get(VALUE_FIELD) or [])
        if not n_val:
            raise SystemExit(f"--node-format {args.node_format}: no {args.tree_sources or 'tree'} item of {args.train} "
                             f"has value nodes ({VALUE_FIELD}) with the rendering {args.node_render} "
                             "(build them with scripts/build_gsm8k_ops.py)")
        print(f"value nodes ({args.node_render}): {n_val}", flush=True)
    if args.curriculum_steps:                # GFPQC: every tree node of the --tree-sources items needs its step
        from cotdistill.student import curriculum_stage
        by_stage: dict = {}
        for it in train_items:
            if tree_sources is None or it.get("source") in tree_sources:
                for sq in it.get("subqs") or []:
                    if "id" in sq:
                        try:
                            c = curriculum_stage(sq.get("step"), args.curriculum_steps)
                            by_stage[c] = by_stage.get(c, 0) + 1
                        except ValueError as e:
                            raise SystemExit(f"--curriculum-steps: {it['item_id']} node {sq['id']}: {e}")
        if not by_stage:
            raise SystemExit(f"--curriculum-steps: no {args.tree_sources or 'tree'} item of {args.train} has tree nodes")
        print(f"curriculum: {args.curriculum_steps} node stages ({args.curriculum_mode}) + the final stage"
              f"{' with all nodes' if args.curriculum_final_nodes else ''}; nodes by stage (steps beyond "
              f"{args.curriculum_steps} in the last): {json.dumps(dict(sorted(by_stage.items())))}", flush=True)
    if args.tree_view in ("statements", "statements+question", "local+ctxparse", "mix"):   # GFP-S, GFPQ, TFLP, MIX: every
        from cotdistill.treeplan import problem_no_options, statement_block   # view must build (fail before training);
        view_fn = problem_no_options if args.tree_view == "statements+question" else statement_block   # GFPQ needs
        for it in train_items:                                              # only the problem without its options
            if tree_sources is None or it.get("source") in tree_sources:
                try:
                    view_fn(it)
                except ValueError as e:
                    raise SystemExit(f"--tree-view {args.tree_view}: {e}")
    placebo_kw = {}
    if args.tree_target == "placebo":       # GFPSX / GFPQX: every asked item needs a placebo text, no MC node; the bank
        from cotdistill.student import PlaceboBank, placebo_text      # of "no" sentences: the training items of
        from cotdistill.treeplan import is_mc_node                    # --tree-sources
        tree_items = [it for it in train_items if tree_sources is None or it.get("source") in tree_sources]
        for it in tree_items:
            if any(is_mc_node(sq) for sq in it.get("subqs") or [] if "id" in sq):
                raise SystemExit(f"--tree-target placebo: {it['item_id']} has multiple-choice nodes")
            if any("id" in sq for sq in it.get("subqs") or []):
                try:
                    placebo_text(it)
                except ValueError as e:
                    raise SystemExit(f"--tree-target placebo: {e}")
        bank = PlaceboBank(tree_items)
        placebo_kw = {"target_kind": "placebo", "placebo_bank": bank}
        print("placebo bank:", ", ".join(f"{k} {len(bank.texts[k])}" for k in sorted(bank.texts, key=str)), "problems",
              flush=True)
    tree_stats: dict = {}
    # GTF's scaffold fades with the update index t = epoch x U + (position of the problem's group in the epoch) // b
    # over T updates. U (updates per pass) and T are counted here, before the plan's counting pass, from the problems
    # with a final question (the others give no group); build_examples with permute_final 0 draws nothing from its
    # rng, so a throwaway generator keeps the training rng untouched. Checked against the plan below.
    fade = None
    if args.tree_scaffold:
        n_fin = sum(any(e.kind == "final" for e in build_examples(it, final=args.final, subq="none",
                                                                 subq_target=args.subq_target, lambda_sub=0.0,
                                                                 rng=random.Random(0)))
                    for it in train_items)
        upp = math.ceil(n_fin / args.items_per_update)
        fade = {"U": upp, "T": max(1, math.ceil(args.epochs * upp))}

    si_log: list[dict] = []                  # --si-stages: the chain counts of every pass
    cur_log: list[dict] = []                 # --curriculum-steps: the problems and views of every pass

    def fulltree_updates(si_stage: int | None = None, cur_stage: int | None = None) -> list[list[Example]]:
        """Problems in shuffled order, each as [final] + all its node views; items_per_update problems per update
        (the number of sequences per update varies with tree size). si_stage (--si-stages): the final views carry
        that stage's worked steps (student.si_group, which draws from no shared generator). cur_stage
        (--curriculum-steps K): 1..K = node stage j, each problem's node views of that stage only
        (student.curriculum_item) and no final view, problems without one left out; K + 1 = the final stage, the
        final views only (--curriculum-final-nodes: the whole group)."""
        items = train_items[:]
        order_rng.shuffle(items)
        st: dict = {}
        ep = passes["n"]                     # MIX / GTF key their per-node coins by it (checkpointed; the counting
        passes["n"] += 1                     # pass is 0)
        b = args.items_per_update
        groups, ps = [], []
        sst: dict = {}                       # --si-stages: this pass's chain counts
        sst_slot: dict = {}                  # --latent-slots: this pass's slot counts
        for it in items:
            p_sc = None
            if fade is not None:             # GTF: p(t) of the update this problem's group lands in
                p_sc = scaffold_fade(ep * fade["U"] + len(groups) // b, fade["T"], args.tree_scaffold)
            src = it
            node_stage = cur_stage is not None and cur_stage <= args.curriculum_steps
            if node_stage:                   # --curriculum-steps: this node stage's nodes of a tree item
                if tree_sources is not None and it.get("source") not in tree_sources:
                    continue
                src = curriculum_item(it, stage=cur_stage, n_stages=args.curriculum_steps, mode=args.curriculum_mode)
            elif cur_stage is not None and not args.curriculum_final_nodes:   # the final stage: final views only
                src = dict(it, subqs=[])
            g = build_full_tree(src, mode=args.tree_full, rng=rng, cap=args.tree_cap,
                                aux_total=args.aux_total, node_format=args.node_format,
                                final=args.final, eligible_only=args.tree_eligible_only,
                                subq_target=args.subq_target, permute_final=args.permute_final,
                                subq_filter=args.subq_filter, view=args.tree_view,
                                sources=tree_sources, node_weight=args.tree_node_weight, stats=st,
                                mix_p=args.tree_mix_p, mix_seed=args.seed, mix_epoch=ep, scaffold_p=p_sc,
                                depth_weight=args.tree_depth_weight, node_types=node_types,
                                target_seed=args.seed if args.tree_target != "real" else None,
                                node_render=args.node_render, **placebo_kw)
            if g and args.tree_pack and any(e.kind == "subq" for e in g):   # GFPSP: the GFPS views, packed
                g = pack_views(g, tok, statement_block(it), max_len=args.max_len, max_nodes=args.tree_pack,
                               rng=pack_rng(args.seed, ep, it["item_id"]), stats=st)
            if g and si_stage is not None and (tree_sources is None or it.get("source") in tree_sources):
                g = si_group(g, it, stage=si_stage, n_stages=args.si_stages, chain=args.si_chain, seed=args.seed,
                             epoch=ep, smooth=args.si_smooth, final_only=args.si_final_only, remove=args.si_remove,
                             stats=sst)
            if g and slot is not None:       # --latent-slots: the slot line (+ targets) in the final view
                g = slot_group(g, it, spec=slot, targets=slot_tg.get(it["item_id"]) if slot_tg else None,
                               lam=args.slot_lambda, final_only=args.slot_final_only, stats=sst_slot)
            if g and node_stage:             # node stages train no final question
                g = [e for e in g if e.kind != "final"]
            if g:
                groups.append(g)
                ps.append(p_sc)
        if tree_opts and not tree_stats:     # the plan's counting pass: views, dropped views, stated results
            tree_stats.update(st)
            print("tree views", json.dumps(st, sort_keys=True), flush=True)
        if slot is not None and not slot_stats:
            slot_stats.update(sst_slot)
            print("latent slot views", json.dumps(sst_slot, sort_keys=True), flush=True)
        if args.tree_view == "mix":
            print(f"mix pass {ep}: {st.get('statements_views', 0)} statements views, {st.get('local_views', 0)} local "
                  "views", flush=True)
        if si_stage is not None:
            si_log.append({"pass": ep, "stage": si_stage, **sst})
            print(f"si pass {ep} stage {si_stage}/{args.si_stages}: {json.dumps(sst, sort_keys=True)}", flush=True)
        if cur_stage is not None:
            row = {"pass": ep, "stage": cur_stage if cur_stage <= args.curriculum_steps else "final",
                   "problems": len(groups), "node_views": sum(e.kind != "final" for g in groups for e in g),
                   "finals": sum(e.kind == "final" for g in groups for e in g), "updates": -(-len(groups) // b)}
            cur_log.append(row)
            print(f"curriculum pass {ep}: {json.dumps(row)}", flush=True)
        if fade is not None:
            print(f"scaffold pass {ep}: {st.get('scaffold_views', 0)} scaffold views of "
                  f"{st.get('scaffold_eligible', 0)} eligible derived nodes, p(t) {ps[0] if ps else 0:.4f} -> "
                  f"{ps[-1] if ps else 0:.4f} (updates {ep * fade['U']}-{ep * fade['U'] + fade['U'] - 1} of "
                  f"{fade['T']})", flush=True)
        return [[e for g in groups[i:i + b] for e in g] for i in range(0, len(groups), b)]

    if args.tree_full and args.si_stages:   # SI: S + 1 full-tree stages of epochs / (S + 1) each, steps removed stage by
        per_step = 1                          # stage
        stages = [(f"si{k}", (lambda k=k: fulltree_updates(k)), args.epochs / (args.si_stages + 1))
                  for k in range(args.si_stages + 1)]
    elif args.tree_full and args.curriculum_steps:   # step curriculum: K node stages, then the final questions, epochs
        per_step = 1                                  # / (K + 1) each
        stages = [(f"cur{k}" if k <= args.curriculum_steps else "curfinal", (lambda k=k: fulltree_updates(cur_stage=k)),
                   args.epochs / (args.curriculum_steps + 1)) for k in range(1, args.curriculum_steps + 2)]
    elif args.tree_full:
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
    for si_k, (name, make, epochs) in enumerate(stages):
        # SI: every stage has the same updates per pass (one group per problem with a final), so one counting pass as
        # in the single-stage full-tree loop (the generators then stand where GFPQ's do: SIY in rng lockstep with it)
        n = len(make()) if not args.si_stages or si_k == 0 else plan[0][2]
        # (a curriculum node stage without any problem, e.g. --curriculum-mode exact on a few items: no update)
        plan.append((name, make, n, 0 if args.curriculum_steps and not n else max(1, math.ceil(epochs * n / per_step))))
        print(f"stage {name}: {n} examples per pass, {plan[-1][3]} steps", flush=True)
    cur_log.clear()                          # the counting passes are not training
    if args.si_stages:
        si_log.clear()                       # the counting pass is not training
        if not any(worked_steps(it) for it in train_items if tree_sources is None or it.get("source") in tree_sources):
            raise SystemExit(f"--si-stages: no {args.tree_sources or 'training'} item of {args.train} has worked "
                             "steps (field worked_steps: scripts/build_gsm8k_si.py, or a numgen program)")
    total_steps = sum(x[3] for x in plan)
    passes["n"] = 0                  # the counting pass above does not count as training progress
    if fade is not None and (fade["U"], fade["T"]) != (plan[-1][2], total_steps):
        raise SystemExit(f"--tree-scaffold: counted {fade['U']} updates per pass and {fade['T']} in all, the plan has "
                         f"{plan[-1][2]} and {total_steps}")
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
    if args.si_stages or args.curriculum_steps:   # the end of every stage but the last (the final evaluation follows)
        done = 0
        for k, (_, _, _, stage_steps) in enumerate(plan[:-1]):
            done += stage_steps
            eval_steps[done] = round((k + 1) * args.epochs / (len(plan)), 6)
    mid_names = [x for x in args.eval_mid.split(",") if x in evals] or list(evals)   # absent names (smoke) skipped
    mid_sets = {}
    for name in mid_names:
        items = evals[name]
        k = max(1, -(-len(items) // args.eval_mid_max))
        mid_sets[name] = items[::k]

    def mid_eval(at_step: int) -> None:
        row = {"step": at_step, "epoch": eval_steps[at_step], "s_elapsed": round(time.time() - t0, 1)}
        for name, items in mid_sets.items():
            m, preds = evaluate(model, tok, items, args.eval_max_len, args.eval_bs, cache)
            row[name] = {k: m[k] for k in ("acc", "nll", "ece", "brier") if k in m}
            if args.eval_mid_preds and is_main:          # opt-in: the predictions of this point
                (out / "mid").mkdir(exist_ok=True)
                with open(out / "mid" / f"preds_{name}_e{eval_steps[at_step]:g}.jsonl", "w") as f:
                    for p in preds:
                        f.write(json.dumps(p) + "\n")
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
        acc = torch.zeros(6 if not slot_loss_on else 9, dtype=torch.float64, device=fs.device)  # loss, final, node,
        for mb, pad, ntok in parts[fs.rank]:      # node CE sum, nodes, padded tok (+ slot loss, slot CE sum, slots)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                if slot_loss_on:         # every rank runs both passes (pads at weight 0): same collectives
                    z, sl, sls = slot_forward(model, tok, mb, args.max_len, cache)
                    loss = (kl_loss(z, mb, args.label_smoothing) * len(mb) + sl) / norm
                else:
                    z = label_logits(model, tok, mb, args.max_len, cache)
                    loss = kl_loss(z, mb, args.label_smoothing) * len(mb) / norm
                if args.lambda_brier:
                    loss = loss + args.lambda_brier * brier_loss(z, mb) * len(mb) / norm
            loss.backward()
            if pad:                      # weight 0: no loss, no gradient, only the collectives
                continue
            acc[0] += loss.detach().double()
            acc[5] += ntok
            if slot_loss_on:
                acc[6] += sl.item() / norm
                acc[7] += sls["slot_ce_sum"]
                acc[8] += sls["slot_n"]
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
        slot_part = {"loss_slot": a[6], "slot_ce_sum": a[7], "slot_n": int(a[8])} if slot_loss_on else {}
        return {"loss": a[0], "loss_final": a[1], "loss_node": a[2], "node_ce": a[3] / max(1.0, a[4]), "gnorm": gnorm,
                "fwd": len(parts[fs.rank]) * fs.world, "pad_fwd": sum(x[1] for r in parts for x in r),
                "pad_tok": int(a[5]), "fb_s": t_b - t_a, "clip_s": t_c - t_b, "opt_s": time.time() - t_c,
                **{k: v for k, v in trainer.timing.items() if k != "opt_step_s"}, **slot_part}

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
    reset_stages = args.reset_optim or bool(args.si_stages) and not args.si_keep_optim
    gsched = None                            # --si-stages with --si-schedule global: one schedule over every stage
    if args.si_stages and args.si_schedule == "global" or args.curriculum_steps:
        gwarm = max(1, int(args.warmup * total_steps))
        gsched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s, w=gwarm, t=total_steps: min(1.0, (s + 1) / w) * 0.5 * (
            1 + math.cos(math.pi * min(1.0, s / t))))
    for si, (name, make, _, stage_steps) in enumerate(plan):
        if reset_stages and si > 0:
            if trainer is not None:
                trainer.reset_state()
            else:
                opt.state.clear()
        if gsched is not None:
            sched = gsched
        else:
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
        while (name == "fulltree" or name.startswith("si") or name.startswith("cur")) and step < end:
            # One problem group per update; the loss is scaled by 1 / (3 x items_per_update), the per-sequence
            # normalisation of the 1 final + 2 auxiliary grouped arms, so the learning-rate scale is comparable.
            if not upd_iter:
                upd_iter = make()
            upd, upd_iter = upd_iter[0], upd_iter[1:]
            norm, upd_loss = 3 * args.items_per_update, 0.0
            split = {"final": 0.0, "subq": 0.0, "node_ce": 0.0, "node_n": 0}
            if slot_loss_on:
                split.update(slot=0.0, slot_ce=0.0, slot_n=0)
            if fs is not None:           # the same sequences split across the ranks, gradients summed
                fu = fsdp_update(upd, norm)
                upd_loss = fu["loss"]
                split.update(final=fu["loss_final"], subq=fu["loss_node"], node_ce=fu["node_ce"], node_n=1)
                if slot_loss_on:
                    split.update(slot=fu["loss_slot"], slot_ce=fu["slot_ce_sum"], slot_n=fu["slot_n"])
                seen_tok += sum(min(args.max_len, len(e.text) // 3) for e in upd)
                if tree_opts:
                    gwin.append(fu["gnorm"])
                sched.step()
            else:
                # micro_bs chunks, or length-sorted micro-batches of at most token_budget padded tokens
                mbs = micro_batches(tok, upd, args.max_len, args.micro_bs, args.token_budget, args.token_budget_seqs)
                for mb in mbs:
                    rd = readouts(mb)                  # --tree-pack: one entry per node view of a pack; else mb
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                        if slot_loss_on:               # --latent-slots: the main pass + the slot decoding pass
                            z, sl, sls = slot_forward(model, tok, mb, args.max_len, cache)
                            loss = (kl_loss(z, rd, args.label_smoothing) * len(rd) + sl) / norm
                            split["slot"] += sl.item() / norm
                            split["slot_ce"] += sls["slot_ce_sum"]
                            split["slot_n"] += sls["slot_n"]
                        else:
                            z = label_logits(model, tok, mb, args.max_len, cache)
                            loss = kl_loss(z, rd, args.label_smoothing) * len(rd) / norm
                        if args.lambda_brier:          # as in the grouped loop (was silently ignored in full-tree arms)
                            loss = loss + args.lambda_brier * brier_loss(z, rd) * len(rd) / norm
                    loss.backward()
                    upd_loss += loss.item()
                    seen_tok += sum(min(args.max_len, len(e.text) // 3) for e in mb)
                    if tree_opts:                      # Tree Forcing arms: the loss split by kind (detached, no gradient)
                        with torch.no_grad():
                            for zz, e in zip(z, rd):
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
                if args.tree_pack:
                    row["reads"] = len(readouts(upd))
                if tree_opts:      # this update's loss by kind (weighted; the two sum to loss without Brier), its
                    # mean unweighted node CE, and the pre-clip gradient norms of the updates since the last row
                    row.update(loss_final=split["final"], loss_node=split["subq"],
                               node_ce=split["node_ce"] / max(1, split["node_n"]),
                               gnorm=sum(gwin) / len(gwin), gnorm_max=max(gwin),
                               clip_frac=sum(g > 1.0 for g in gwin) / len(gwin))
                    if slot_loss_on:   # the slot decoding loss (weighted, in loss) and its mean unweighted CE per slot
                        row.update(loss_slot=split["slot"], slot_ce=split["slot_ce"] / max(1, split["slot_n"]),
                                   slot_n=split["slot_n"])
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
            if args.token_budget:        # one GPU: the update of the fixed loop below (grad_accum refills of micro_bs,
                upd = []                 # so the same sequences in the same updates), regrouped by length
                for _ in range(args.grad_accum):
                    if len(ex_iter) < args.micro_bs:
                        ex_iter += make()
                    batch, ex_iter = ex_iter[:args.micro_bs], ex_iter[args.micro_bs:]
                    upd += batch
                norm, upd_loss, padded = args.micro_bs * args.grad_accum, 0.0, []   # = len(batch) x grad_accum below
                mbs = micro_batches(tok, upd, args.max_len, args.micro_bs, args.token_budget, args.token_budget_seqs,
                                    padded=padded)
                for mb in mbs:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                        z = label_logits(model, tok, mb, args.max_len, cache)
                        loss = kl_loss(z, mb, args.label_smoothing) * len(mb) / norm
                        if args.lambda_brier:
                            loss = loss + args.lambda_brier * brier_loss(z, mb) * len(mb) / norm
                    loss.backward()
                    upd_loss += loss.item()
                seen_tok += sum(min(args.max_len, len(e.text) // 3) for e in upd)
                micro += args.grad_accum
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0 or step == total_steps:
                    dt = time.time() - t0
                    mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0.0
                    row = {"step": step, "loss": upd_loss, "lr": float(sched.get_last_lr()[0]), "s_per_step": dt / step,
                           "approx_tok_per_s": seen_tok / dt, "peak_gib": mem, "upd_loss": upd_loss, "fwd": len(mbs),
                           "pad_tok": sum(padded)}
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
    if args.si_stages:
        results["si"] = si_log
    if args.curriculum_steps:
        results["curriculum"] = cur_log
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
    if slot is not None:                 # --latent-slots: the spec, the counts, the necessity test, the decoding readout
        from cotdistill.student import SLOT_SIDECAR, slot_evals
        results["latent_slots"] = {**slot.as_dict(), "target": args.slot_target, "lambda": args.slot_lambda,
                                   "overflow": args.slot_overflow, "final_only": args.slot_final_only,
                                   "swap_token": swap_token, "views": slot_stats}
        slot_evals(model, tok, evals, results, out, modes=parse_slot_necessity(args.slot_necessity),
                   sets=[x for x in args.slot_necessity_sets.split(",") if x], swap_token=swap_token,
                   decode_n=args.slot_decode, decode_set=args.slot_decode_set, max_len=args.eval_max_len,
                   bs=args.eval_bs, cache=cache, seed=args.seed, overflow=args.slot_overflow, write=is_main)
        if is_main:
            (out / SLOT_SIDECAR).write_text(json.dumps(slot.as_dict(), indent=1))
    for sname, spath in subq_sets(args.eval_subq or ""):
        sm, spreds = evaluate_subq(model, tok, read_jsonl(spath), args.max_len, args.eval_bs, cache, seed=args.seed,
                                   value_render=args.node_render)
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
                if slot is not None:     # eval_student rebuilds the slot line from it
                    shutil.copy(out / SLOT_SIDECAR, out / "model" / SLOT_SIDECAR)
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
        if slot is not None:             # eval_student rebuilds the slot line from it
            shutil.copy(out / SLOT_SIDECAR, out / "model" / SLOT_SIDECAR)
    if resumable:                        # finished: metrics (and weights) are written, the checkpoint goes
        shutil.rmtree(ck_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
