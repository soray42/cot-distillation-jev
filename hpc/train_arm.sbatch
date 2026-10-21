#!/bin/bash
#SBATCH --job-name=cotd-train
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Student arms, run one after another inside one job (SLURM_COORD: one cotd- job in the system at a time).
# From ~/cotd:  ARMS="A2 A3 A4 A1" DATA=data/student_tree EVAL_BASE=1 sbatch repo/hpc/train_arm.sbatch
# JEFF=1 evaluates Jeff with its own serving path (scripts/eval_jeff.py); BASE_SUBQ=1 scores the untrained base on
# the val sub-questions. Every arm is scored on the sub-questions of $DATA/val.jsonl (--eval-subq). DUMP=val,kk,...
# saves the readout hidden states of those eval sets (probing); PERM / BRIER / SUBQK / SUBQW set --permute-final /
# --lambda-brier / --subq-k / --subq-weight.
# LORA=r trains rank-r LoRA adapters on a frozen backbone (set PREC=bf16 and e.g. LR=1e-4); --save then writes the
# adapter and the merged weights. MODEL picks the
# student backbone (default Qwen3.5-2B-Base; e.g. MODEL=$HOME/cotd/models/Qwen3.5-2B for the post-trained release).
# Before submitting, dry-run the shell logic on the login node (Python is only echoed):
#   DRY_RUN=1 ARMS="A2 A3" DATA=data/student_tree EVAL_BASE=1 bash repo/hpc/train_arm.sbatch
# SMOKE_ONLY=1 stops after the smoke stage (checks a new account or environment in a few minutes).
# The job starts with a smoke stage (every arm on 16 items for about 2 steps, plus a 20-item eval) and exits
# at once if any of it fails. Defaults from check job 677208: fp32 master weights + AdamW, micro-batch 2 x
# grad-accum 16 (peak 32 GiB on the 4g.40gb slice). EVAL_BASE=1 also evaluates the untrained $MODEL (models are
# downloaded on the login node beforehand; compute nodes have no /userScripts).
# Arms: A1 gold answer only · A2 teacher answer only · A3 teacher + CoT sub-questions (ours) · A4 teacher +
# matched control · A5 CoT sub-questions only · A6 teacher + CoT sub-q with truth targets · A7 teacher +
# LM loss on the teacher's CoT and answer · A9 depth curriculum: DSTAGES (default 4) passes over the items, each
# drawing STAGEK (default 1) CoT nodes per item from the next band of tree depth (half new band, half replay), then
# the final questions for EPOCHS, with a fresh LR schedule per stage · A9b: A9 + one level-balanced tree node per item
# and epoch kept during the final stage, level-balanced replay, AdamW state reset per stage (Yao et al. 2505.17923;
# Deng et al. 2405.14838; Saparov et al. 2412.04703).
# A10ub: upper-bound diagnostic, final questions trained with the teacher's tree nodes stated as text (eval set
# val_facts has them too) · A10: fact internalization, FSTAGES (default 3) passes with stated nodes dropped
# shallow-first, then plain final questions for EPOCHS (stepwise internalization, Deng et al. 2405.14838).
# G0/G3/G4: exposure-matched grouped training (collaborator review E2): every update holds 32 problems, each as 1 final
# view + 2 auxiliary views (placebo = CoT views with zero loss / CoT nodes at 0.5 / matched controls at 0.5).
# G3MC/G4MC: G3/G4 with multiple-choice auxiliary views ("Which of these questions has the answer Yes?", 3 questions +
# "None of them"), the same number of views per item in both; no 2-option question is left in training.
# TF/TFM/GF/GF0: full-tree groups, every problem with ALL its nodes (up to 10) in one update (run with EPOCHS=1, about
# the grouped arms' compute): parents' results stated / as many depth-matched non-parent results (eligible nodes only,
# the same nodes and counts in both) / no stated results / plain views at zero weight.
# G3L/G3D: G3 with CoT nodes only on the logic families (K&K, JustLogic, FOLIO) or only on the decision families (ShARC,
# returns, expense, subscription); the other items keep their views at zero weight. Where does the negative transfer come from?
# T1/T1S/T2: tree transitions in the same grouped setting: each auxiliary node is asked with its parents' results
# stated (T1), with as many unrelated nodes' results stated (T1S, edge-rewiring control), or with the parents' results
# withdrawn over the first 75% of training (T2, internalization).
# TFE/TFME/GFE: TF/TFM/GF restricted to the eligible node views (--tree-eligible-only; ablation rung R3, every asked view
# differs between TFE and TFME); GFC: full tree with the control questions (--tree-full control, with G4's --subq random
# --subq-target fresh; GF + XARGS="--tree-full control" trains the same views but records --subq cot --subq-target cot).
# Decision chains (TB-C, DATA=data/student_chain; notes/next_round/tbc_readme.md): G3P2 = G3 asking only the checks of
# the chain the final reads (--subq-filter path; the "G3 (path)" arm of the notes), G3M2 = the same on the mirror chain
# the final never reads (the "G4 (mirror)" content control), G3P2X/G3M2X = the same without the chain's top check
# (--aux-answer exclude; the path top is the final's decision bit). Plain G3 there asks both chains; chain-data G4 asks
# lookups of the two facts no check reads (an easy-control reference, not the mirror arm). XARGS="--subq-filter
# path|mirror" restricts TFE/TFME/GFE/T1/T2 the same way (stated results and matched pools still come from both chains);
# the run directory then gets "-sf<filter>" after the arm name, so it never overwrites the unfiltered run.
# REPO: code checkout the job runs (default repo; ablation jobs use a frozen checkout, e.g. REPO=$HOME/cotd/repo-abl);
# its commit (with -dirty if tracked files differ) goes into the job log and metrics.json args (code_commit).
# 4B / long runs (full-tree arms only; notes/next_round/sev4b_plan.md §6-§7): PREC=bf16 OPTIM=adamw8bit-sr = torchao 8-bit
# AdamW with bf16 stochastic rounding (needs torchao in the env); TOKBUDGET=3072 = length-sorted micro-batches of at most
# 3072 padded tokens (--token-budget, same gradient as MBS; MBS/GA are then unused); STOPH=H = planned stop H hours after
# the JOB start (smoke and base evals included; leave room for the final evals and one checkpoint write): the arm writes
# $OUT/ckpt and the job exits with code 3 ("PLANNED STOP" in the log); resubmitting the same job resumes it, and arms
# already finished ($OUT/metrics.json, no $OUT/ckpt) are skipped. CKPTMIN=M also writes the checkpoint every M minutes
# (one checkpoint, replaced via ckpt.tmp + rename, so two exist during a write: about 2 x 17 GB at 4B, plus the 9.3 GB
# base). CKPTONE=1 (--ckpt-single) deletes the old checkpoint before writing the new one: one copy on disk, but a kill
# during the write leaves none (the arm then restarts from update 0). At the end, --save writes model/ (8.4 GB at 4B)
# before the checkpoint is deleted. With STOPH or CKPTMIN every arm must be a full-tree arm, and the smoke stage also
# runs a checkpoint round trip (stop after update 1, resume) on the first arm. No relay resubmits after exit 3 yet
# (plan §7: same job again with --dependency=afterany); SLURM records the job as FAILED with exit code 3.
# Tree Forcing (notes/next_round/tree_forcing/impl.md; DATA=data/student_v3sl, built by scripts/build_local_kk.py): every
# K&K node every epoch (--tree-cap 0), only K&K items get node views (--tree-sources knights_knaves; the other families
# train their final question only), the final view unchanged. TFL = each node asked WITHOUT the problem: preamble + its
# parents' results (parse nodes: their statement line) · TFA = the same with every ancestor's result, shallow to deep ·
# TFLM = structure control: as many matched non-parent results (post-audit matching; nodes without a same-class
# replacement keep their true parents in both arms) · GFP = every node with the full problem and no stated results.
# Controls: GFPS (GFP-S) = GFP with the problem cut to its statement block (preamble + "- X says" lines; no final
# question, no options; rng lockstep with GFP) · TFLN = TFL with nothing stated (parse nodes still see their line,
# derived nodes only the preamble; rng lockstep with TFL). Same knobs (NODEW, LOCAL_EVAL, run dir runs/<ARM>...).
# Premise extraction (impl.md section 9): TFLP = TFL with the parse nodes asked in the GFPS view (the whole statement
# block instead of their own line) · MIX = every node asked once per epoch, in the GFPS view with probability .5 (a
# coin per seed, epoch, item and node), else in its TFL view. Both in rng lockstep with TFL (same nodes, order, targets
# and weights as TFL and GFPS); same knobs; no new upper-case knob. Another MIX share: XARGS="--tree-mix-p P" with
# TAG=mp<P> (the run dir does not show P otherwise).
# GTF (grounded Tree Forcing, impl.md section 10) = GFPS + a fading scaffold + depth weights: a derived node is asked in
# the GFPS view with its parents' results stated ("Known results:" under the statement lines) with probability
# max(0, 1 - t / (0.5 T)) at update t of T (--tree-scaffold 0.5; a coin per seed, epoch, item and node, so the run is
# step-matched with GFPS: same nodes, order, targets), and each problem's node weight (same total) is split in
# proportion to 1 + node depth (--tree-depth-weight 1.0). Other values: XARGS="--tree-scaffold F --tree-depth-weight G"
# with TAG (argparse: the last value wins); numgen: XARGS="--tree-sources numgen" as for the other arms.
# Value nodes (DATA=data/student_gsm8k_tf_ops*, scripts/build_gsm8k_ops.py; XARGS="--tree-sources gsm8k_tf" as for
# GFPQ on GSM8K): GFPQV = GFPQ with one node per atomic operation of GSM8K's calculator notes, three numeric options
# ("What is 4 x 1.5?") instead of the yes/no nodes · GFPQVQ = the same nodes as "<sub-question> It is 4 x 1.5. What is
# the value?" · GFPQO = operation selection ("<sub-question> Which computation gives it?", three expressions; a step's
# last operation only) · GFPVB = GFPQV's nodes asked alone, no problem text (--tree-view bare) · GFPQVM = GFPQ's yes/no
# views unchanged plus GFPQV's value views, each format half of the per-problem auxiliary weight. The yes/no arms on
# the same data train exactly as on data/student_gsm8k_tf (the value nodes are in a field of their own).
# Verification nodes (DATA=data/student_gsm8k_tf_vt, scripts/build_gsm8k_vt.py; XARGS="--tree-sources gsm8k_tf" as for
# GFPQ): GFPQVT = the final + one node per asked option of the final question, in the problem view (the prompt WITH its
# options): "In this problem, what does option C (8) stand for?" with four descriptions (the answer to the problem's
# final question / a Socratic sub-question of the solution / a number written in the problem text / none of the
# solution's quantities), labels A..D, one-hot target, the per-problem aux weight split over the problem's nodes
# (about 2.5: the gold option, every option equal to a step value or a written number, one "none" option) ·
# GFPQVTR = the same stems, choices and weights with each problem's descriptions permuted among its nodes (wrong-
# content control; about 37% keep their own by chance, logged as tree_view_stats.vtr_agree). --eval-subq scores the
# verification nodes of $DATA/val.jsonl in vt for both (groups value/vt_kind/<gold|step|given|none>); other arms on
# this DATA can score them too with XARGS="... --eval-value-render vt" (A2: XARGS="--eval-value-render vt"). The rows
# are data/student_gsm8k_tf_si's plus the field verify_nodes, so every other arm trains on them exactly as on its own
# data. Other-problem test sets (scripts/build_mc_xprob.py: the same problems and golds, the 4 distractors final
# answers of other problems of the file): EXTRA_EVALS="... gsm8kx=data/eval/gsm8k_mc_v2_xprob.jsonl
# gsm1kx=data/eval/gsm1k_mc_xprob.jsonl" next to gsm8kv2 / gsm1k.
# Stepwise internalisation (Deng et al. 2024, ICoT-SI; DATA=data/student_gsm8k_tf_si*, built by scripts/build_gsm8k_si.py,
# or data/student_numgen_tf*, whose programs give the steps; XARGS="--tree-sources gsm8k_tf" / "--tree-sources numgen" as
# for GFPQ): the final question is trained with the problem's worked steps written between the problem and its options
# ("Worked steps:" / "Step 1: 4 + 2 = 6", one line per atomic operation, the answer's operation last) over SISTAGES + 1
# stages (default 4 + 1 = 5) of EPOCHS / (SISTAGES + 1) epochs each (EPOCHS defaults to SISTAGES + 1 for these arms);
# stage k removes the first ceil(k n / SISTAGES) of a problem's n steps (one more with probability 0.1), so the last
# stage trains the test format; AdamW reset at every stage, one warmup + cosine schedule over all stages; validation and
# test evaluation final-question-only, with a mid-training evaluation (the --eval-at mechanism, EVALMID sets) at the end
# of every stage but the last. SI = the final question only · SIY = SI + GFPQ's yes/no node views in every update (the
# same groups, order and option orders as GFPQ; the last stage is GFPQ) · SIS = SI with every computed value written
# wrongly (each result a distractor of the same size, never the truth; operands that are earlier results get those
# wrong values: same lines and counts, wrong content). Knobs: XARGS="--si-smooth P --si-schedule stage --si-keep-optim
# --si-remove back --si-chain wrong-results" (with TAG). No STOPH / CKPTMIN (multi-stage runs do not resume); relays
# must put SISTAGES in their reset list.
# Step curriculum (GFPQC; any data whose tree nodes have a solution step, e.g. data/student_gsm8k_tf, field step = the
# Socratic step index; XARGS="--tree-sources gsm8k_tf" as for GFPQ): CSTEPS (default 4) node stages, then the final
# questions, EPOCHS / (CSTEPS + 1) passes per stage (EPOCHS is the total): node stage j trains only GFPQ's node views of
# the steps <= j (every step beyond CSTEPS counted as CSTEPS), no final question; the last stage trains the final
# questions only. AdamW state kept across stages, one warmup + cosine schedule over all of them; mid-training
# evaluation (EVALMID sets) at the end of every stage but the last, so EVALAT stays empty. Knobs: XARGS="--curriculum-mode
# exact" (stage j = the steps == j), "--curriculum-final-nodes" (the last stage = GFPQ's groups), "--reset-optim" (with
# TAG). No STOPH / CKPTMIN; relays must put CSTEPS in their reset list.
# Latent node slots (Tree Forcing v2, design (a) of notes/latent_primer_zh_2026-10-19.md; DATA=data/student_gsm8k_tf_si*,
# whose worked_steps give the targets; XARGS="--tree-sources gsm8k_tf" as for GFPQ): every final question gets a line of
# SLOTK (default 6) x SLOTC (default 1) reserved tokens (<|box_start|> on Qwen3.5) between the problem and its options
# (SLOTPLACE=after-options: after the options), in training and in every evaluation (one forward pass, the same letter
# readout; a saved model carries latent_slots.json for eval_student). LS = in training only, slot j's final-layer
# states start a separate sequence through the same network and head that must continue with the j-th worked step
# ("4 + 2 = 6"; chains longer than SLOTK: the tail merged into the last slot), loss = letter CE + SLOTLAM (default 1)
# x the mean slot decoding CE per problem; final views only · LSU = the same slots without any slot loss (pause-token
# control, no decoding pass) · LSR = LS with each slot's target drawn from other problems' steps (wrong-content
# control; same supervised slots) · LSY = LS + GFPQ's yes/no node views (no slots in them). After training: the
# necessity test SLOTNEC (default swap,drop,noise@mid:1.0: slot tokens swapped for <|quad_start|>, no slot line, noise
# of 1 x RMS on the slot states after the middle decoder layer; on SLOTNECSETS, default the EVALMID sets, else all; keys
# <set>@<mode>, metrics slot_necessity) and the decoding readout of the first SLOTDEC (default 32) items of the first eval
# set (slot_decode_<set>.jsonl). SLOTK/SLOTC/SLOTLAM/SLOTPLACE also name the run dir (-sk<K>-sc<c>-sl<lam>-sp<place>)
# for these arms; relays must put SLOTK SLOTC SLOTLAM SLOTPLACE SLOTNEC SLOTNECSETS SLOTDEC in their reset list.
# NGPU=2 / the 1-rank FSDP route work (test_latent_slots TestFSDP); STOPH/CKPTMIN resume works on one GPU.
# Latent slots v2.1 (lens readout; same DATA / XARGS as LS): LSL = SLOTK (default 6) slots of SLOTC (default 4) tokens,
# every slot its own reserved token (SLOTTOK=distinct; Qwen3.5: <|box_start|> <|quad_start|> <|object_ref_start|>
# <|box_end|> <|quad_end|> <|object_ref_end|>, swap token <|fim_pad|>); in training only, slot j's c states at the
# OUTPUT of decoder layer SLOTL (default 0.67 = round(0.67 n) - 1: layer 20 of 32 on Qwen3.5-4B / 9B, after which come
# 11 layers, 3 of them full attention, 15 of 24 on 2B) go through the model's own final norm + LM head (no second
# pass, no new parameters) and position i must give token i of the j-th worked step's VALUE ("6"; Qwen3.5 writes one
# token per digit, so 4 positions hold 97% of the GSM8K values whole), then "\n"; chains longer than SLOTK keep steps
# 1..K-1 and the last (answer) step's value, slots after a shorter chain get the last value (SLOTFILL=last; none /
# answer); loss = letter CE + SLOTLAM x the mean slot CE per problem; final views only · LSLU = the same prompt without
# any slot loss · LSLR = LSL with every value drawn from other problems' values (same supervised slots) · LSLY = LSL +
# GFPQ's yes/no node views. After training: SLOTNEC (default swap,drop,mean: one swap token for all slots, no slot
# line, the layer-SLOTL slot states replaced by their mean over the first 300 training items) and the lens readout of
# the first SLOTDEC (default 400: all of GSM8K val) items of each set of SLOTDECSET (default: the first eval set) ->
# metrics slot_readout (exact value match per slot) and slot_pilot (pass: value readout >= 45% and mean-ablation
# acc_drop >= 2 points). Run dirs: -sk/-sc/-sl/-sp as LS, plus -sL<SLOTL> -sF<SLOTFILL> -sT<SLOTTOK>; relays must put
# SLOTL SLOTFILL SLOTTOK SLOTDECSET in their reset list too.
# Batch 12 diagnostics (notes/latent_slots_v21_audit_2026-10-20.md), v2.1 arms: SLOTPADW=W (--slot-pad-weight: weight
# of a slot's "\n" padding positions in its lens CE, 0 = the value's own tokens only, renormalised over the counted
# positions) and SLOTOV=truncate (--slot-overflow: a chain longer than SLOTK keeps its FIRST SLOTK steps; SLOTK=1
# SLOTOV=truncate SLOTFILL=none supervises only the first worked step's value); run dirs -sW<W> -sO<ov>.
# PROBE=1: after training the model is written to a job-local dir (PROBE_TMP, default $_CONDOR_SCRATCH_DIR, else
# $TMPDIR, else /tmp; never in the run dir, never uploaded, SAVE stays off), scripts/probe_slots.py probes it (linear
# probes of every layer at the slot / problem / "Answer:" positions, the frozen-head lens at every layer; fit on
# PROBE_N, default 3000, items of $DATA/train.jsonl; held-out PROBE_HELDOUT, default "val=$DATA/val.jsonl
# gsm8kv2=data/eval/gsm8k_mc_v2_steps.jsonl"), the probe outputs go to $OUT/probe/ and the weights are deleted (an
# EXIT trap deletes them too). With SAVE=1 as well, $OUT/model is probed and kept. The run dir gets -pr before the TAG
# (a probed rerun never shares a dir with the run it repeats, e.g. batch 11's LSL). PROBE_ONLY=1 with MODEL: no
# training; the untrained MODEL is probed with the slot line of each v2.1 arm in ARMS (its SLOTK / SLOTC / SLOTTOK /
# SLOTPLACE / SLOTL / SLOTFILL / SLOTOV / SLOTPADW knobs, defaults as LSL; PROBE_SPEC=<run dir | json>: that run's
# latent_slots.json instead) into runs/PROBE-<arm><knobs>-<model>-<data> (logged as "=== PROBE -> <dir>"). PROBE_ARGS:
# more probe_slots.py options. The smoke stage probes the smoke model (PROBE_ONLY: MODEL) on 16 + 4 items. Relays must
# put SLOTPADW SLOTOV PROBE PROBE_ONLY PROBE_N PROBE_HELDOUT PROBE_SPEC PROBE_TMP PROBE_ARGS in their reset list.
# Teacher-state slots with forced reading (TSF; notes/latent_comeback_memo_2026-10-21.md section 3; DATA=
# data/student_gsm8k_tf_si*, XARGS="--tree-sources gsm8k_tf" as for LSL): the deployed prompt is LSL's with SLOTK
# (default 2) slots of 2 distinct reserved tokens before the options, final views only. In training a teacher view of
# the same weights (the final question with worked-step lines before its options, SI stage 0, no slot line) gives
# the targets: slot j's two tokens <- the teacher's residual states at the outputs of layers TSFAL (default 12-20) at
# the token that predicts the value's first token of step j's line and at the value's last token (SLOTOV=truncate:
# steps 1, 2); align loss smooth L1 / teacher RMS, stop-grad, weight TSFLAM (default 1) per problem; with probability
# TSFP (default 0.5) per micro-batch the layer-SLOTL (default 0.67 = 20 of 32) slot states are REPLACED by the teacher's
# before the later layers compute the letter CE (forced reading); the teacher view trains the letter CE at TSFCE
# (default 0.5). TSF = steps · TSFN = TSF without forcing (p 0) · TSFR = another problem's steps in the teacher view
# (wrong content) · TSFX = the problem's own sentences in place of the steps (placebo) · TSF6 = TSF with 6 slots, merge
# (steps 1-5 + the last). Defaults: the gradient-norm probe at updates TSFPROBE (default 0,5,10,15,20; off = none: the
# align loss rescaled once if align / CE is outside [0.3, 3]), TSFGUARD=<A2 val accuracy at the first EVALAT point>
# (the teacher CE drops to 0.25 if the first EVALMID set's accuracy is 3 points below it; default off), the "use"
# detector at every EVALAT point on the EVALMID sets with worked steps (TSF arms; TSFDET: other sets / none): intact /
# mean / force / forceswap and the alignment loss into metrics.json trajectory; SLOTNEC default swap,drop,mean,force,
# forceswap (force modes need eval files with worked_steps: gsm8kv2=data/eval/gsm8k_mc_v2_steps.jsonl, identical to
# gsm8k_mc_v2 plus the field). PTAD / PTADR / PTADX (memo section 4, zero-latent CODI): the A2 prompt, final views only;
# the "Answer:" states at layers PTADAL (default 12-27) aligned to the teacher view's (steps / another problem's /
# placebo), weight PTADLAM (default 1), teacher CE TSFCE; PTADPROBE: the gradient probe (default off). Run dirs: TSF
# arms -sk -sp -sL -sT -sO as LSL plus -fp<TSFP> -wl<TSFLAM> -tc<TSFCE> -al<TSFAL> -tg<TSFGUARD> -gp<TSFPROBE>; PTAD
# arms -al<PTADAL> -wl<PTADLAM> -tc<TSFCE> -tg<TSFGUARD> -gp<PTADPROBE>. Relays must put TSFP TSFLAM TSFCE TSFAL
# TSFGUARD TSFPROBE TSFDET PTADAL PTADLAM PTADPROBE in their reset list.
# NODEW=w gives every node view weight w (--tree-node-weight; default: the per-problem split, aux 1.0 / views, about
# .023 per K&K node); set it to the same value for all arms of a comparison. LOCAL_EVAL=path scores the
# local-step probe (--eval-local: every K&K node in its local view, eval key local_kk); when unset it is
# $DATA/val.jsonl if that file has local fields (any arm), and LOCAL_EVAL=0 turns it off.
# NODEW also names the run dir (-nw<w> after the arm), so a dose variant never shares a dir with the default run;
# relays that compute the run dir as their skip key must add the same suffix.
# Two slices, one job (notes/next_round/scale12b/fsdp2.md): NGPU=2 runs every train_student call (smoke included) under
# torchrun with one rank per 4g.40gb slice (--fsdp: FSDP2, the same sequences per update as on one slice, split across
# the ranks, gradients summed; rank 0 writes the outputs), PREC defaults to bf16, OMP threads = CPUs / 2 per rank.
# FSDP_OPTIM picks the optimizer: cpu-adamw (default; fp32 master + AdamW of each rank's shard on the CPU, the only
# setup that fits 12B; no checkpoints, so no STOPH/CKPTMIN) or 8bit-sr (torchao AdamW8bit on the GPU shards, as
# OPTIM=adamw8bit-sr; resumable; does not fit 12B). TOKBUDGET then works for every arm (A2/G3 too). The #SBATCH lines
# above ask for one slice: `NGPU=2 ... bash repo/hpc/train_arm.sbatch [sbatch options]` on the login node submits this
# script with --gres=gpu:4g.40gb:2 --cpus-per-task=12 --mem=220G (96G for 8bit-sr) --time=24:00:00 plus the options
# given (e.g. --dependency=afterany:<job>); a plain `sbatch` with NGPU=2 must pass those itself (the job checks that it
# holds two GPUs and exits at once otherwise). 12B: MODEL=$HOME/cotd/models/gemma-4-12B TOKBUDGET=4096
# XARGS="--eval-bs 16". Planned stops under torchrun leave $OUT/planned_stop (torchrun reports exit 3 as 1).
# Relay scripts: TOKBUDGET STOPH CKPTMIN CKPTONE NODEW LOCAL_EVAL NGPU FSDP_OPTIM belong in the reset list (relay44's
# reset() lacks OPTIM too), so a 4B, Tree Forcing or two-slice setting never leaks into the next submission (le, tr,
# fs, pr, ng, mem are internal, lower case).
set -euo pipefail
JOB_T0=$(date +%s)
ng="${NGPU:-1}"
[ "$ng" = 1 ] || [ "$ng" = 2 ] || { echo "NGPU must be 1 or 2 (got $ng)"; exit 1; }
if [ "$ng" = 2 ] && [ -z "${SLURM_JOB_ID:-}" ] && [ -z "${DRY_RUN:-}" ]; then   # login node: submit with two slices
  mem=220G; [ "${FSDP_OPTIM:-cpu-adamw}" = 8bit-sr ] && mem=96G
  echo "submitting: sbatch --gres=gpu:4g.40gb:2 --cpus-per-task=12 --mem=$mem --time=24:00:00 $* $0"
  exec sbatch --gres=gpu:4g.40gb:2 --cpus-per-task=12 --mem="$mem" --time=24:00:00 "$@" "$0"
fi
cd "$HOME/cotd"
module load cuda/12.4 2>/dev/null || true
SEED="${SEED:-0}"; MODEL="${MODEL:-$HOME/cotd/models/Qwen3.5-2B-Base}"; DATA="${DATA:-data/student}"
export HF_HOME="$HOME/cotd/hf" TOKENIZERS_PARALLELISM=false
# ORDERSEED: problem-order seed (default SEED); LSMOOTH: label smoothing of all targets (e.g. 0.1).
# EVALAT=0.5,1,2,... (epochs) with EVALMID=val,kk,...: training-length curve, the named sets (cut to 600 items) are
# evaluated at those points during training and stored in metrics.json "trajectory".
# CC1D=1: use the causal-conv1d CUDA kernel built for torch 2.14.0+cu130 (installed under ~/cotd/cc1d, not in the env);
# without it transformers falls back to the PyTorch reference implementation (same maths, slower)
[ -n "${CC1D:-}" ] && export PYTHONPATH="$HOME/cotd/cc1d${PYTHONPATH:+:$PYTHONPATH}"
PY=envs/cotd/bin/python
[ -n "${DRY_RUN:-}" ] && PY="echo [dry-run] python"
tr="$PY"; fs=; pr=fp32master           # launcher, --fsdp arguments and default precision (NGPU=2: torchrun, FSDP, bf16)
if [ "$ng" = 2 ]; then
  case "${FSDP_OPTIM:-cpu-adamw}" in cpu-adamw|8bit-sr) ;; *) echo "FSDP_OPTIM must be cpu-adamw or 8bit-sr"; exit 1 ;; esac
  tr="$PY -m torch.distributed.run --standalone --nproc_per_node 2"
  fs="--fsdp --fsdp-optim ${FSDP_OPTIM:-cpu-adamw}"; pr=bf16
  if [ -z "${DRY_RUN:-}" ]; then
    n=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES:-}")
    [ "${n:-0}" -ge 2 ] || { echo "NGPU=2 but this job sees ${n:-0} GPU(s) (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}): submit with NGPU=2 bash repo/hpc/train_arm.sbatch, or sbatch --gres=gpu:4g.40gb:2 --cpus-per-task=12 --mem=220G"; exit 1; }
  fi
  export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-12} / 2 ))   # per rank (torchrun would set 1)
  [ -z "${STOPH:-}${CKPTMIN:-}" ] || [ "${FSDP_OPTIM:-cpu-adamw}" = 8bit-sr ] || \
    { echo "STOPH/CKPTMIN with NGPU=2 need FSDP_OPTIM=8bit-sr (cpu-adamw state, 143 GB at 12B, is not saved)"; exit 1; }
  echo "two slices: torchrun, 2 ranks, ${FSDP_OPTIM:-cpu-adamw}, OMP_NUM_THREADS=$OMP_NUM_THREADS per rank"
fi
REPO="${REPO:-repo}"
[ -f "$REPO/scripts/train_student.py" ] || { echo "missing $REPO/scripts/train_student.py"; exit 1; }
COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
[ "$COMMIT" = unknown ] || [ -z "$(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null)" ] || COMMIT="$COMMIT-dirty"
echo "code: $REPO @ $COMMIT"
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi here"
[ -f "$DATA/train.jsonl" ] || { echo "missing $DATA/train.jsonl"; exit 1; }
EVALS="val=$DATA/val.jsonl kk=data/eval/kk_heldout.jsonl jl=data/eval/jl_heldout.jsonl jevbench=data/eval/jevbench_public.jsonl td=data/eval/typed_decisions_test.jsonl bbeh=data/eval/bbeh.jsonl ${EXTRA_EVALS:-}"
for f in $EVALS; do [ -f "${f#*=}" ] || { echo "missing eval file ${f#*=}"; exit 1; }; done
PT=                                     # PROBE=1: the job-local model dir of the arm being probed (deleted on exit)
trap '[ -z "$PT" ] || rm -rf "$PT"' EXIT
PH=                                     # PROBE / PROBE_ONLY: the probe's held-out sets
if [ -n "${PROBE:-}${PROBE_ONLY:-}" ]; then
  [ -z "${PROBE:-}" ] || [ -z "${PROBE_ONLY:-}" ] || { echo "PROBE and PROBE_ONLY exclude each other"; exit 1; }
  [ -f "$REPO/scripts/probe_slots.py" ] || { echo "missing $REPO/scripts/probe_slots.py"; exit 1; }
  PH="${PROBE_HELDOUT:-val=$DATA/val.jsonl gsm8kv2=data/eval/gsm8k_mc_v2_steps.jsonl}"
  for f in $PH; do [ -f "${f#*=}" ] || { echo "missing probe held-out file ${f#*=}"; exit 1; }; done
fi

arm_config() {           # sets F S T RLM DS DX EPD for arm $1 (EPD: the arm's default EPOCHS, empty = 2)
  RLM=; DS=; DX=; EPD=
  local SL="--tree-full plain --tree-view statements+question --tree-cap 0 --tree-sources knights_knaves --latent-slots ${SLOTK:-6} --slot-c ${SLOTC:-1} --slot-lambda ${SLOTLAM:-1.0} --slot-place ${SLOTPLACE:-before-options} --slot-necessity ${SLOTNEC:-swap,drop,noise@mid:1.0} --slot-decode ${SLOTDEC:-32}"
  local NS="${SLOTNECSETS:-${EVALMID:-}}"; SL="$SL${NS:+ --slot-necessity-sets $NS}"
  local TSFB="--tree-full plain --tree-view statements+question --tree-cap 0 --tree-sources knights_knaves --latent-slots ${SLOTK:-2} --slot-c 2 --slot-place ${SLOTPLACE:-before-options} --slot-layer ${SLOTL:-0.67} --slot-readout tsf --slot-overflow ${SLOTOV:-truncate} --slot-tokens ${SLOTTOK:-distinct} --slot-align-layers ${TSFAL:-12-20} --slot-align-lambda ${TSFLAM:-1.0} --slot-teacher-ce ${TSFCE:-0.5} --slot-final-only --slot-necessity ${SLOTNEC:-swap,drop,mean,force,forceswap}${NS:+ --slot-necessity-sets $NS}${TSFDET:+ --slot-detect-sets $TSFDET}${TSFGUARD:+ --teacher-ce-guard $TSFGUARD}"
  [ "${TSFPROBE:-0,5,10,15,20}" = off ] || TSFB="$TSFB --teacher-grad-probe ${TSFPROBE:-0,5,10,15,20}"
  local PTB="--tree-full plain --tree-view statements+question --tree-cap 0 --tree-sources knights_knaves --answer-align-layers ${PTADAL:-12-27} --answer-align-lambda ${PTADLAM:-1.0} --answer-teacher-ce ${TSFCE:-0.5}${TSFGUARD:+ --teacher-ce-guard $TSFGUARD}${PTADPROBE:+ --teacher-grad-probe $PTADPROBE}"
  local SLL="--tree-full plain --tree-view statements+question --tree-cap 0 --tree-sources knights_knaves --latent-slots ${SLOTK:-6} --slot-c ${SLOTC:-4} --slot-lambda ${SLOTLAM:-1.0} --slot-place ${SLOTPLACE:-before-options} --slot-layer ${SLOTL:-0.67} --slot-readout lens --slot-fill ${SLOTFILL:-last}${SLOTPADW:+ --slot-pad-weight $SLOTPADW}${SLOTOV:+ --slot-overflow $SLOTOV} --slot-tokens ${SLOTTOK:-distinct} --slot-necessity ${SLOTNEC:-swap,drop,mean} --slot-decode ${SLOTDEC:-400}${SLOTDECSET:+ --slot-decode-set $SLOTDECSET}${NS:+ --slot-necessity-sets $NS}"
  case "$1" in
    A1) F=gold;    S=none;   T=cot ;;
    A2) F=teacher; S=none;   T=cot ;;
    A3) F=teacher; S=cot;    T=cot ;;
    A4) F=teacher; S=random; T=fresh ;;
    A5) F=none;    S=cot;    T=cot ;;
    A6) F=teacher; S=cot;    T=truth ;;
    A7) F=teacher; S=none;   T=cot; RLM=1 ;;
    A8) F=teacher; S=mix;    T=cot ;;           # half CoT nodes, half matched controls (same count as A3)
    A9) F=teacher; S=cot;    T=cot; DS=${DSTAGES:-4} ;;  # depth curriculum: CoT nodes by tree depth, then finals
    A9b) F=teacher; S=cot;   T=cot; DS=${DSTAGES:-4}; DX="--final-replay 1 --level-balanced --reset-optim" ;;
    A10ub) F=teacher; S=none; T=cot; DX="--facts-always" ;;      # diagnostic: tree nodes stated in every prompt
    A10) F=teacher; S=none;  T=cot; DX="--fact-stages ${FSTAGES:-3} --reset-optim" ;;  # fact internalization
    G0) F=teacher; S=cot;    T=cot; DX="--grouped-aux 2 --aux-kind placebo --aux-weight 0.5" ;;  # E2 placebo
    G3) F=teacher; S=cot;    T=cot; DX="--grouped-aux 2 --aux-kind cot --aux-weight 0.5" ;;      # E2 CoT nodes
    G4) F=teacher; S=random; T=fresh; DX="--grouped-aux 2 --aux-kind random --aux-weight 0.5" ;; # E2 matched controls
    G3MC) F=teacher; S=cot;  T=cot; DX="--grouped-aux 2 --aux-kind cot_mc --aux-weight 0.5" ;;     # MC node views
    G3R) F=teacher; S=cot;   T=cot; DX="--grouped-aux 2 --aux-kind cot_randlabel --aux-weight 0.5" ;;  # G3, random node labels
    G3NA) F=teacher; S=cot;  T=cot; DX="--grouped-aux 2 --aux-kind cot --aux-weight 0.5 --aux-answer exclude" ;; # no answer-component nodes
    G3A) F=teacher; S=cot;   T=cot; DX="--grouped-aux 2 --aux-kind cot --aux-weight 0.5 --aux-answer only" ;;    # answer-component nodes only
    G3P2) F=teacher; S=cot;  T=cot; DX="--grouped-aux 2 --aux-kind cot --aux-weight 0.5 --subq-filter path" ;;  # decision chains: path checks only
    G3M2) F=teacher; S=cot;  T=cot; DX="--grouped-aux 2 --aux-kind cot --aux-weight 0.5 --subq-filter mirror" ;;  # decision chains: mirror checks only
    G3P2X) F=teacher; S=cot; T=cot; DX="--grouped-aux 2 --aux-kind cot --aux-weight 0.5 --subq-filter path --aux-answer exclude" ;;  # path checks below the top
    G3M2X) F=teacher; S=cot; T=cot; DX="--grouped-aux 2 --aux-kind cot --aux-weight 0.5 --subq-filter mirror --aux-answer exclude" ;;  # mirror checks below the top
    G4MC) F=teacher; S=random; T=fresh; DX="--grouped-aux 2 --aux-kind random_mc --aux-weight 0.5" ;; # MC controls
    TF) F=teacher; S=cot;    T=cot; DX="--tree-full true" ;;      # full tree: every node, parents' results stated
    TFE) F=teacher; S=cot;   T=cot; DX="--tree-full true --tree-eligible-only" ;;     # eligible nodes, parents stated
    TFME) F=teacher; S=cot;  T=cot; DX="--tree-full matched --tree-eligible-only" ;;  # eligible nodes, matched results
    GFE) F=teacher; S=cot;   T=cot; DX="--tree-full plain --tree-eligible-only" ;;    # eligible nodes, no results
    GFC) F=teacher; S=random; T=fresh; DX="--tree-full control" ;;  # full tree of control questions (fresh targets)
    TFM) F=teacher; S=cot;   T=cot; DX="--tree-full matched" ;;   # full tree: as many depth-matched non-parent results
    GF) F=teacher; S=cot;    T=cot; DX="--tree-full plain" ;;     # full tree: every node, no stated results
    GF0) F=teacher; S=cot;   T=cot; DX="--tree-full placebo" ;;   # full tree: plain views at zero weight
    G3L) F=teacher; S=cot;   T=cot; DX="--grouped-aux 2 --aux-kind cot --aux-weight 0.5 --aux-sources knights_knaves,justlogic,folio" ;;
    G3D) F=teacher; S=cot;   T=cot; DX="--grouped-aux 2 --aux-kind cot --aux-weight 0.5 --aux-sources sharc,returns,expense,subscription" ;;
    T1) F=teacher; S=cot;    T=cot; DX="--grouped-aux 2 --aux-kind tree --aux-weight 0.5" ;;     # tree: true parents
    T1S) F=teacher; S=cot;   T=cot; DX="--grouped-aux 2 --aux-kind tree_shuf --aux-weight 0.5" ;; # tree: rewired edges
    T2) F=teacher; S=cot;    T=cot; DX="--grouped-aux 2 --aux-kind tree --aux-weight 0.5 --fact-withdraw 0.75" ;;
    TFL) F=teacher; S=cot;   T=cot; DX="--tree-full true --tree-view local --tree-cap 0 --tree-sources knights_knaves" ;;      # Tree Forcing: parents only
    TFA) F=teacher; S=cot;   T=cot; DX="--tree-full true --tree-view ancestors --tree-cap 0 --tree-sources knights_knaves" ;;  # all ancestors
    TFLM) F=teacher; S=cot;  T=cot; DX="--tree-full matched --tree-view local --tree-cap 0 --tree-sources knights_knaves" ;;  # matched non-parents
    GFP) F=teacher; S=cot;   T=cot; DX="--tree-full plain --tree-view problem --tree-cap 0 --tree-sources knights_knaves" ;;   # every node, full problem
    GFPS) F=teacher; S=cot;  T=cot; DX="--tree-full plain --tree-view statements --tree-cap 0 --tree-sources knights_knaves" ;; # GFP-S: no question/options
    GFPQ) F=teacher; S=cot;  T=cot; DX="--tree-full plain --tree-view statements+question --tree-cap 0 --tree-sources knights_knaves" ;; # GFP-S + the final question (no options)
    GFPSNC) F=teacher; S=cot; T=cot; DX="--tree-full plain --tree-view statements --tree-cap 0 --tree-sources knights_knaves --tree-node-types parse,derive" ;; # GFPS without the K&K case (hypothesis) nodes
    GTF) F=teacher; S=cot;   T=cot; DX="--tree-full plain --tree-view statements --tree-cap 0 --tree-sources knights_knaves --tree-scaffold 0.5 --tree-depth-weight 1.0" ;;  # GFPS + fading parents' results + depth weights
    TFLN) F=teacher; S=cot;  T=cot; DX="--tree-full plain --tree-view local --tree-cap 0 --tree-sources knights_knaves" ;;   # local view, nothing stated
    TFLP) F=teacher; S=cot;  T=cot; DX="--tree-full true --tree-view local+ctxparse --tree-cap 0 --tree-sources knights_knaves" ;;  # TFL, parse nodes in the GFPS view
    MIX) F=teacher; S=cot;   T=cot; DX="--tree-full true --tree-view mix --tree-cap 0 --tree-sources knights_knaves" ;;   # per node and epoch: TFL or GFPS view (p .5)
    GFPSP) F=teacher; S=cot; T=cot; DX="--tree-full plain --tree-view statements --tree-cap 0 --tree-sources knights_knaves --tree-pack 8" ;;  # GFPS, up to 8 node questions per sequence (one pass, no answers)
    GFPSR) F=teacher; S=cot; T=cot; DX="--tree-full plain --tree-view statements --tree-cap 0 --tree-sources knights_knaves --tree-target shuffled" ;;  # GFPS, node targets permuted within each problem (random-label control)
    GFPQR) F=teacher; S=cot; T=cot; DX="--tree-full plain --tree-view statements+question --tree-cap 0 --tree-sources knights_knaves --tree-target shuffled" ;;  # GFPQ, node targets permuted within each problem (random-label control)
    GFPSX) F=teacher; S=cot; T=cot; DX="--tree-full plain --tree-view statements --tree-cap 0 --tree-sources knights_knaves --tree-target placebo" ;;  # GFPS, node questions replaced by surface questions about the problem text, correct answers (placebo control)
    GFPQX) F=teacher; S=cot; T=cot; DX="--tree-full plain --tree-view statements+question --tree-cap 0 --tree-sources knights_knaves --tree-target placebo" ;;  # GFPQ, node questions replaced by surface questions about the problem text, correct answers (placebo control)
    GFPQV) F=teacher; S=cot; T=cot; DX="--tree-full plain --tree-view statements+question --tree-cap 0 --tree-sources knights_knaves --node-format value --node-render op" ;;  # GFPQ with value nodes (one per atomic operation, 3 numeric options): "What is 4 x 1.5?"
    GFPQVQ) F=teacher; S=cot; T=cot; DX="--tree-full plain --tree-view statements+question --tree-cap 0 --tree-sources knights_knaves --node-format value --node-render qop" ;;  # GFPQV, Socratic sub-question + the operation
    GFPQO) F=teacher; S=cot; T=cot; DX="--tree-full plain --tree-view statements+question --tree-cap 0 --tree-sources knights_knaves --node-format value --node-render opsel" ;;  # GFPQ with operation-selection nodes ("Which computation gives it?", 3 expressions)
    GFPVB) F=teacher; S=cot; T=cot; DX="--tree-full plain --tree-view bare --tree-cap 0 --tree-sources knights_knaves --node-format value --node-render op" ;;  # value nodes asked alone (no problem text)
    GFPQVM) F=teacher; S=cot; T=cot; DX="--tree-full plain --tree-view statements+question --tree-cap 0 --tree-sources knights_knaves --node-format yesno+value --node-render op" ;;  # GFPQ's yes/no nodes + the op value nodes, half the per-problem weight each
    GFPQVT) F=teacher; S=cot; T=cot; DX="--tree-full plain --tree-view problem --tree-cap 0 --tree-sources knights_knaves --node-format value --node-render vt" ;;  # final + verification nodes: what each option is (problem view, options shown)
    GFPQVTR) F=teacher; S=cot; T=cot; DX="--tree-full plain --tree-view problem --tree-cap 0 --tree-sources knights_knaves --node-format value --node-render vtr" ;;  # GFPQVT's stems and choices, descriptions permuted among a problem's options (wrong-content control)
    SI) F=teacher; S=cot; T=cot; EPD=$(( ${SISTAGES:-4} + 1 )); DX="--tree-full plain --tree-view statements+question --tree-cap 0 --tree-sources knights_knaves --si-stages ${SISTAGES:-4} --si-final-only" ;;  # stepwise internalisation: the final question with the worked steps, removed front-first stage by stage
    SIY) F=teacher; S=cot; T=cot; EPD=$(( ${SISTAGES:-4} + 1 )); DX="--tree-full plain --tree-view statements+question --tree-cap 0 --tree-sources knights_knaves --si-stages ${SISTAGES:-4}" ;;  # SI + GFPQ's yes/no node views every update
    SIS) F=teacher; S=cot; T=cot; EPD=$(( ${SISTAGES:-4} + 1 )); DX="--tree-full plain --tree-view statements+question --tree-cap 0 --tree-sources knights_knaves --si-stages ${SISTAGES:-4} --si-final-only --si-chain wrong" ;;  # SI with every computed value written wrongly (wrong-content control)
    SIB) F=teacher; S=cot; T=cot; EPD=$(( ${SISTAGES:-4} + 1 )); DX="--tree-full plain --tree-view statements+question --tree-cap 0 --tree-sources knights_knaves --si-stages ${SISTAGES:-4} --si-final-only --si-remove back" ;;  # SI, steps removed from the answer end first
    SIYB) F=teacher; S=cot; T=cot; EPD=$(( ${SISTAGES:-4} + 1 )); DX="--tree-full plain --tree-view statements+question --tree-cap 0 --tree-sources knights_knaves --si-stages ${SISTAGES:-4} --si-remove back" ;;  # SIB + GFPQ's yes/no node views
    SISB) F=teacher; S=cot; T=cot; EPD=$(( ${SISTAGES:-4} + 1 )); DX="--tree-full plain --tree-view statements+question --tree-cap 0 --tree-sources knights_knaves --si-stages ${SISTAGES:-4} --si-final-only --si-chain wrong --si-remove back" ;;  # SIB with wrong values (control)
    GFPQC) F=teacher; S=cot; T=cot; DX="--tree-full plain --tree-view statements+question --tree-cap 0 --tree-sources knights_knaves --curriculum-steps ${CSTEPS:-4}" ;;  # step curriculum: GFPQ's node views by solution step (<= 1, <= 2, ...), then the final questions
    LS) F=teacher; S=cot; T=cot; DX="$SL --slot-final-only" ;;   # latent node slots: slot j decodes worked step j (training only), final views only
    LSU) F=teacher; S=cot; T=cot; DX="$SL --slot-final-only --slot-target none" ;;   # the same slots, no slot loss (pause-token control)
    LSR) F=teacher; S=cot; T=cot; DX="$SL --slot-final-only --slot-target shuffled" ;;   # slot targets from other problems (wrong-content control)
    LSY) F=teacher; S=cot; T=cot; DX="$SL" ;;   # LS + GFPQ's yes/no node views
    LSL) F=teacher; S=cot; T=cot; DX="$SLL --slot-final-only" ;;   # v2.1: layer-L slot states read by the final norm + head, value targets
    LSLU) F=teacher; S=cot; T=cot; DX="$SLL --slot-final-only --slot-target none" ;;   # the same prompt, no slot loss
    LSLR) F=teacher; S=cot; T=cot; DX="$SLL --slot-final-only --slot-target shuffled" ;;   # values from other problems
    LSLY) F=teacher; S=cot; T=cot; DX="$SLL" ;;   # LSL + GFPQ's yes/no node views
    TSF) F=teacher; S=cot; T=cot; DX="$TSFB --slot-teacher steps --slot-force-p ${TSFP:-0.5}" ;;   # teacher-state slots + forced reading
    TSFN) F=teacher; S=cot; T=cot; DX="$TSFB --slot-teacher steps --slot-force-p 0" ;;   # TSF without forced reading (write only)
    TSFR) F=teacher; S=cot; T=cot; DX="$TSFB --slot-teacher shuffled --slot-force-p ${TSFP:-0.5}" ;;   # another problem's steps (wrong content)
    TSFX) F=teacher; S=cot; T=cot; DX="$TSFB --slot-teacher placebo --slot-force-p ${TSFP:-0.5}" ;;   # the problem's sentences as steps (placebo)
    TSF6) F=teacher; S=cot; T=cot; DX="$TSFB --slot-teacher steps --slot-force-p ${TSFP:-0.5} --latent-slots 6 --slot-overflow merge" ;;   # 6 slots: steps 1-5 + the last (readout upper bound)
    PTAD) F=teacher; S=cot; T=cot; DX="$PTB --answer-teacher steps" ;;   # zero-latent CODI: Answer: states <- the teacher view's
    PTADR) F=teacher; S=cot; T=cot; DX="$PTB --answer-teacher shuffled" ;;   # another problem's steps in the teacher view
    PTADX) F=teacher; S=cot; T=cot; DX="$PTB --answer-teacher placebo" ;;   # placebo lines in the teacher view
    *) echo "unknown ARM $1"; return 1 ;;
  esac
}

train() {                # train <arm> <train.jsonl> <evals> <out> [extra args...]
  local arm=$1 trn=$2 ev=$3 out=$4; shift 4
  arm_config "$arm"
  $tr $REPO/scripts/train_student.py --model "$MODEL" --train "$trn" --eval $ev \
    --final "$F" --subq "$S" --subq-target "$T" --lambda-sub "${LAMBDA:-1.0}" \
    --lr "${LR:-1e-5}" --max-len 1536 --precision "${PREC:-$pr}" --optim "${OPTIM:-adamw}" $fs \
    --seed "$SEED" --out "$out" --eval-subq "$DATA/val.jsonl${SUBQ_EXTRA:+,$SUBQ_EXTRA}" ${RLM:+--rationale-lm --lm-max-len 3072} \
    ${DUMP:+--dump-hidden "$DUMP,smoke"} ${PERM:+--permute-final "$PERM"} ${BRIER:+--lambda-brier "$BRIER"} \
    ${SUBQK:+--subq-k "$SUBQK"} ${SUBQW:+--subq-weight "$SUBQW"} ${DS:+--depth-stages "$DS" --stage-k "${STAGEK:-1}"} \
    ${SUBQF:+--subq-frac "$SUBQF"} ${LORA:+--lora-r "$LORA"} ${DUMPL:+--dump-layers "$DUMPL"} $DX ${XARGS:-} \
    ${EVALAT:+--eval-at "$EVALAT"} ${EVALMID:+--eval-mid "$EVALMID"} ${ORDERSEED:+--order-seed "$ORDERSEED"} \
    ${LSMOOTH:+--label-smoothing "$LSMOOTH"} ${TOKBUDGET:+--token-budget "$TOKBUDGET"} ${NODEW:+--tree-node-weight "$NODEW"} \
    ${le:+--eval-local "$le"} --code-commit "$COMMIT" "$@"
}
probe() {                # probe <model dir> <out dir> <train.jsonl> [more probe_slots.py args...]
  local m=$1 o=$2 trn=$3; shift 3
  $PY $REPO/scripts/probe_slots.py --model "$m" --out "$o" --train "$trn" --train-n "${PROBE_N:-3000}" \
    --heldout $PH --max-len 1536 --png ${PROBE_ARGS:-} "$@"
}
probe_spec() {           # PROBE_ONLY: the slot line of v2.1 arm $1 as probe_slots.py flags (PROBE_SPEC: from that run)
  if [ -n "${PROBE_SPEC:-}" ]; then echo "--slot-spec-from $PROBE_SPEC"; return 0; fi
  case "$1" in
    LSL|LSLU|LSLR|LSLY) ;;
    TSF|TSFN|TSFR|TSFX|TSF6) local k="${SLOTK:-2}" ov="${SLOTOV:-truncate}"; [ "$1" = TSF6 ] && k=6 && ov=merge
      echo "--latent-slots $k --slot-c 2 --slot-tokens ${SLOTTOK:-distinct} --slot-place ${SLOTPLACE:-before-options} --slot-layer ${SLOTL:-0.67} --slot-fill none --slot-overflow $ov"; return 0 ;;
    *) echo "PROBE_ONLY: $1 is not a v2.1 / TSF slot arm (LSL / LSLU / LSLR / LSLY / TSF*); set PROBE_SPEC" >&2; return 1 ;;
  esac
  echo "--latent-slots ${SLOTK:-6} --slot-c ${SLOTC:-4} --slot-tokens ${SLOTTOK:-distinct} --slot-place ${SLOTPLACE:-before-options} --slot-layer ${SLOTL:-0.67} --slot-fill ${SLOTFILL:-last} --slot-overflow ${SLOTOV:-merge}${SLOTPADW:+ --slot-pad-weight $SLOTPADW}"
}
stop_h() {               # hours left until STOPH hours after the job start (at least 0.01)
  awk -v h="$STOPH" -v e="$(( $(date +%s) - JOB_T0 ))" 'BEGIN { r = h - e / 3600; printf "%.3f", (r < 0.01 ? 0.01 : r) }'
}

LOCAL_EVAL="${LOCAL_EVAL:-}"            # the local-step probe (--eval-local) needs data with local fields
if [ -z "$LOCAL_EVAL" ] && grep -q -m 1 '"local_preamble"' "$DATA/val.jsonl"; then LOCAL_EVAL="$DATA/val.jsonl"; fi
[ "$LOCAL_EVAL" = 0 ] && LOCAL_EVAL=
[ -z "$LOCAL_EVAL" ] || [ -f "$LOCAL_EVAL" ] || { echo "missing LOCAL_EVAL file $LOCAL_EVAL"; exit 1; }
echo "local-step probe: ${LOCAL_EVAL:-off}"
ARMS="${ARMS:-${ARM:-}}"
[ -n "$ARMS${JEFF:-}${EVAL_BASE:-}${BASE_SUBQ:-}" ] || { echo "set ARMS (or JEFF/EVAL_BASE/BASE_SUBQ for an eval-only job)"; exit 1; }

echo "=== smoke stage $(date +%T)"
SM=$(mktemp -d)
head -n 16 "$DATA/train.jsonl" > "$SM/train.jsonl"
head -n 20 data/eval/kk_heldout.jsonl > "$SM/eval.jsonl"
[ -z "$LOCAL_EVAL" ] || grep -m 2 '"local_preamble"' "$LOCAL_EVAL" > "$SM/local.jsonl" || true   # no local fields: empty probe
[ -z "${CKPTONE:-}" ] || [ -n "${STOPH:-}${CKPTMIN:-}" ] || { echo "CKPTONE needs STOPH or CKPTMIN"; exit 1; }
sa=$ARMS; [ -z "${PROBE_ONLY:-}" ] || sa=     # PROBE_ONLY trains nothing (lower case: internal)
for ARM in $sa; do
  if [ -n "${STOPH:-}${CKPTMIN:-}" ]; then   # resume needs a full-tree arm (one-slice token budgets work for every arm
    arm_config "$ARM"                         # since e0ca874)
    [[ " $DX ${XARGS:-} " == *" --tree-full "* ]] || { echo "SMOKE FAILED $ARM: STOPH/CKPTMIN need a full-tree arm"; exit 1; }
    [[ " $DX ${XARGS:-} " != *" --si-stages "* ]] || { echo "SMOKE FAILED $ARM: SI arms (multi-stage) do not resume: no STOPH/CKPTMIN"; exit 1; }
  fi
  SUBQ_EXTRA= le=${LOCAL_EVAL:+$SM/local.jsonl} train "$ARM" "$SM/train.jsonl" "smoke=$SM/eval.jsonl" "$SM/out-$ARM" --epochs 0.5 --micro-bs 2 --grad-accum 2 \
    ${PROBE:+--save --save-dir $SM/pm-$ARM} > "$SM/$ARM.log" 2>&1 || { echo "SMOKE FAILED $ARM"; tail -n 30 "$SM/$ARM.log"; exit 1; }
  [ -n "${DRY_RUN:-}" ] || grep -q '^smoke {' "$SM/$ARM.log" || { echo "SMOKE FAILED $ARM (no eval line)"; tail -n 30 "$SM/$ARM.log"; exit 1; }
  [ -n "${DRY_RUN:-}" ] || [ ! -s "$SM/local.jsonl" ] || grep -q '^local_kk' "$SM/$ARM.log" || { echo "SMOKE FAILED $ARM (no local_kk line)"; tail -n 30 "$SM/$ARM.log"; exit 1; }
  echo "smoke ok $ARM"
  if [ -n "${PROBE:-}" ]; then          # the probe of the smoke model (16 train + 4 held-out items per set)
    probe "$SM/pm-$ARM" "$SM/probe-$ARM" "$SM/train.jsonl" --train-n 16 --heldout-n 4 --fit-steps 5 --layer-chunk 0 \
      > "$SM/probe-$ARM.log" 2>&1 || { echo "SMOKE FAILED probe $ARM"; tail -n 30 "$SM/probe-$ARM.log"; exit 1; }
    [ -n "${DRY_RUN:-}" ] || [ -f "$SM/probe-$ARM/probe_results.json" ] || { echo "SMOKE FAILED probe $ARM (no probe_results.json)"; tail -n 30 "$SM/probe-$ARM.log"; exit 1; }
    rm -rf "$SM/pm-$ARM"
    echo "smoke ok probe $ARM"
  fi
done
if [ -n "${PROBE_ONLY:-}" ]; then        # the probe of the untrained MODEL with each arm's slot line
  for ARM in $ARMS; do
    ps=$(probe_spec "$ARM") || exit 1
    probe "$MODEL" "$SM/probe-$ARM" "$SM/train.jsonl" $ps --train-n 16 --heldout-n 4 --fit-steps 5 --layer-chunk 0 \
      > "$SM/probe-$ARM.log" 2>&1 || { echo "SMOKE FAILED probe $ARM"; tail -n 30 "$SM/probe-$ARM.log"; exit 1; }
    [ -n "${DRY_RUN:-}" ] || [ -f "$SM/probe-$ARM/probe_results.json" ] || { echo "SMOKE FAILED probe $ARM (no probe_results.json)"; tail -n 30 "$SM/probe-$ARM.log"; exit 1; }
    echo "smoke ok probe $ARM ($MODEL)"
  done
fi
if [ -n "${STOPH:-}${CKPTMIN:-}" ] && [ -z "${PROBE_ONLY:-}" ]; then   # checkpoint round trip on the first arm: planned stop after update 1, resume
  a=${ARMS%% *}; rc=0                    # (lower case: internal, so relay reset lists need not know them)
  SUBQ_EXTRA= train "$a" "$SM/train.jsonl" "smoke=$SM/eval.jsonl" "$SM/res-$a" --epochs 1 --items-per-update 4 \
    ${CKPTONE:+--ckpt-single} --stop-at-step 1 > "$SM/res1.log" 2>&1 || rc=$?
  [ -n "${DRY_RUN:-}" ] || [ "$rc" = 3 ] || [ -f "$SM/res-$a/planned_stop" ] || { echo "SMOKE FAILED resume $a (planned stop gave exit $rc)"; tail -n 30 "$SM/res1.log"; exit 1; }
  SUBQ_EXTRA= train "$a" "$SM/train.jsonl" "smoke=$SM/eval.jsonl" "$SM/res-$a" --epochs 1 --items-per-update 4 \
    ${CKPTONE:+--ckpt-single} --ckpt-every-min 1000 > "$SM/res2.log" 2>&1 || { echo "SMOKE FAILED resume $a"; tail -n 30 "$SM/res2.log"; exit 1; }
  [ -n "${DRY_RUN:-}" ] || grep -q '^resumed from' "$SM/res2.log" || { echo "SMOKE FAILED resume $a (no resume line)"; tail -n 30 "$SM/res2.log"; exit 1; }
  echo "smoke ok resume $a"
fi
head -n 3 "$DATA/val.jsonl" > "$SM/val3.jsonl"
if [ -n "${EVAL_BASE:-}${BASE_SUBQ:-}" ]; then
  $PY $REPO/scripts/eval_student.py --model "$MODEL" --eval "smoke=$SM/eval.jsonl" --eval-subq "$SM/val3.jsonl" \
    --out "$SM/base" > "$SM/base.log" 2>&1 || { echo "SMOKE FAILED base eval"; tail -n 30 "$SM/base.log"; exit 1; }
  echo "smoke ok base eval"
fi
if [ -n "${JEFF:-}" ]; then
  $PY $REPO/scripts/eval_jeff.py --model models/Jeff-Qwen3.5-2B --sets jevbench td kk bbh musr --limit 20 --out "$SM/jeff" \
    > "$SM/jeff.log" 2>&1 || { echo "JEFF SMOKE SKIPPED: the Jeff eval is dropped, the job goes on"; tail -n 30 "$SM/jeff.log"; JEFF=; }
  [ -z "${JEFF:-}" ] || echo "smoke ok jeff eval"
fi
rm -rf "$SM"
echo "=== smoke passed $(date +%T)"
[ -z "${SMOKE_ONLY:-}" ] || { echo "SMOKE_ONLY: stopping after the smoke stage"; exit 0; }   # new account/env checks

if [ -n "${JEFF:-}" ]; then               # Jeff with its own prompt, readout matrix and temperature
  echo "=== jeff eval  $(date +%T)"
  $PY $REPO/scripts/eval_jeff.py --model models/Jeff-Qwen3.5-2B --sets jevbench td kk jl bbeh bbh musr policy sharc folio \
    --out runs/jeff-native \
    || echo "jeff eval failed"
fi
if [ -n "${BASE_SUBQ:-}" ]; then         # sub-question accuracy of the untrained base (reference for the arms)
  echo "=== base sub-question eval  $(date +%T)"
  $PY $REPO/scripts/eval_student.py --model "$MODEL" --eval "kk=data/eval/kk_heldout.jsonl" \
    --eval-subq "$DATA/val.jsonl" --out "runs/base-subq-$(basename "$MODEL")" || echo "base eval failed for subq"
fi
if [ -n "${EVAL_BASE:-}" ]; then          # untrained backbone control (Jeff needs its own readout: JEFF=1)
  M=$(basename "$MODEL")
  echo "=== base eval $M  $(date +%T)"
  $PY $REPO/scripts/eval_student.py --model "$MODEL" --eval $EVALS --out "runs/base-$M" ${DUMP:+--dump-hidden "$DUMP"} \
    || echo "base eval failed for $M"
fi
for ARM in $ARMS; do
  SF=$(sed -n 's/.*--subq-filter[ =]\([a-z]*\).*/\1/p' <<< "${XARGS:-}")               # XARGS chain filter -> -sf<f>
  SLD=; case "$ARM" in LS|LSU|LSR|LSY|LSL|LSLU|LSLR|LSLY) SLD="${SLOTK:+-sk$SLOTK}${SLOTC:+-sc$SLOTC}${SLOTLAM:+-sl$SLOTLAM}${SLOTPLACE:+-sp${SLOTPLACE%%-*}}" ;; esac
  case "$ARM" in LSL|LSLU|LSLR|LSLY) SLD="$SLD${SLOTL:+-sL$SLOTL}${SLOTFILL:+-sF$SLOTFILL}${SLOTTOK:+-sT$SLOTTOK}${SLOTPADW:+-sW$SLOTPADW}${SLOTOV:+-sO$SLOTOV}" ;; esac   # v2.1 knobs
  case "$ARM" in   # TSF / PTAD knobs
    TSF|TSFN|TSFR|TSFX|TSF6) SLD="${SLOTK:+-sk$SLOTK}${SLOTPLACE:+-sp${SLOTPLACE%%-*}}${SLOTL:+-sL$SLOTL}${SLOTTOK:+-sT$SLOTTOK}${SLOTOV:+-sO$SLOTOV}${TSFP:+-fp$TSFP}${TSFLAM:+-wl$TSFLAM}${TSFCE:+-tc$TSFCE}${TSFAL:+-al$TSFAL}${TSFGUARD:+-tg$TSFGUARD}${TSFPROBE:+-gp${TSFPROBE//,/_}}" ;;
    PTAD|PTADR|PTADX) SLD="${PTADAL:+-al$PTADAL}${PTADLAM:+-wl$PTADLAM}${TSFCE:+-tc$TSFCE}${TSFGUARD:+-tg$TSFGUARD}${PTADPROBE:+-gp${PTADPROBE//,/_}}" ;;
  esac
  if [ -n "${PROBE_ONLY:-}" ]; then     # no training: the untrained MODEL with this arm's slot line
    po="runs/PROBE-${ARM}${SLD}-$(basename "$MODEL")-$(basename "$DATA")"
    ps=$(probe_spec "$ARM") || exit 1
    echo "=== PROBE -> $po  $(date +%T)"
    probe "$MODEL" "$po" "$DATA/train.jsonl" $ps || echo "probe failed $ARM"
    echo "done $po  $(date +%T)"
    continue
  fi
  OUT="runs/${ARM}${SF:+-sf$SF}${NODEW:+-nw$NODEW}${SLD}${PROBE:+-pr}${TAG:+-$TAG}-$(basename "$MODEL")-$(basename "$DATA")-s${SEED}"   # TAG names recipe variants; NODEW -> -nw<w>; slot knobs -> -sk/-sc/-sl/-sp
  if [ -n "${STOPH:-}${CKPTMIN:-}" ] && [ -f "$OUT/metrics.json" ] && [ ! -d "$OUT/ckpt" ]; then
    echo "=== $ARM: $OUT finished in an earlier job, skipped"; continue
  fi
  echo "=== $ARM -> $OUT  $(date +%T)"
  rc=0
  arm_config "$ARM" || true             # EPD: the arm's default epochs (SI arms: SISTAGES + 1)
  sv=${SAVE:+--save}; pm=               # PROBE: the model to probe (job-local unless SAVE; lower case: internal)
  if [ -n "${PROBE:-}" ]; then
    if [ -n "${SAVE:-}" ]; then pm="$OUT/model"
    else PT="${PROBE_TMP:-${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}}/probe-model-${SLURM_JOB_ID:-$$}-$ARM"; pm="$PT/model"; sv="--save --save-dir $pm"
    fi
  fi
  le=$LOCAL_EVAL train "$ARM" "$DATA/train.jsonl" "$EVALS" "$OUT" --epochs "${EPOCHS:-${EPD:-2}}" --micro-bs "${MBS:-2}" \
    --grad-accum "${GA:-16}" $sv ${STOPH:+--stop-after-h "$(stop_h)"} ${CKPTMIN:+--ckpt-every-min "$CKPTMIN"} \
    ${CKPTONE:+--ckpt-single} || rc=$?
  if [ "$rc" = 3 ] || { [ "$rc" != 0 ] && [ -f "$OUT/planned_stop" ]; }; then   # (torchrun: exit 1 + marker)
    echo "PLANNED STOP $ARM: checkpoint in $OUT/ckpt; resubmit the same job to resume  $(date +%T)"; exit 3
  fi
  [ "$rc" = 0 ] || echo "arm $ARM failed"
  if [ -n "${PROBE:-}" ]; then          # the probe outputs into $OUT/probe, then the job-local weights go
    if [ "$rc" = 0 ]; then
      echo "=== probe of $ARM: $pm into $OUT/probe  $(date +%T)"
      probe "$pm" "$OUT/probe" "$DATA/train.jsonl" || echo "probe failed $ARM"
    fi
    [ -z "$PT" ] || rm -rf "$PT"
    PT=
  fi
  echo "done $OUT  $(date +%T)"
done
echo "all done $(date +%T)"
