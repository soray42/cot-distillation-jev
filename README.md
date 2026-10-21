# Tree Forcing: teaching one-pass decision models to reason

Research code for training **one-pass decision models**: language models that answer a multiple-choice decision by reading
the probability of each option letter after a single forward pass, without writing any reasoning. The main question is
whether such models can learn multi-step reasoning when every intermediate step of a verified reasoning tree is supervised
as an extra yes/no question, asked in the context of the problem, with no premise written out ("Tree Forcing", arm
`GFPS`). The repository also covers a failure-attribution application (which step and which agent caused a multi-agent
system to fail) and the tooling for a causal audit of internal variables.

Owner: Haotian "Sora" Yang. Private research repository; data, results and notes are not tracked (see `.gitignore`).

## Status (2026-10-17)

- Preprint: `paper/tree_forcing_v1/main.tex`. The university HPC has been unreachable since 2026-10-15 afternoon; new runs
  go to the CHTC batch (`cloud/condor`, `cloud/chtc`; results in `tree-forcing-lab/cot-distillation-jev-results`).
- 4B (Qwen3.5-4B), LR 3e-6 unless noted, seed 12 unless noted:
  - **GSM8K**: Tree Forcing (`GFPQ`) minus answer-only on GSM8K test + GSM1K: +5.6 / +3.7 / +2.9 over 12-epoch
    answer-only (3 seeds); +2.5 [+0.8, +4.1] over the tuned answer-only (1e-6, best of the 1e-5 ... 3e-7 sweep).
  - **Data efficiency** (GSM8K): +11.4 at 10%, +9.0 / +6.7 at 25% (2 seeds), +1.9 [+0.2, +3.6] over the placebo at 25%.
  - **Arithmetic chains**: tied with tuned answer-only and the placebo at full data; +5.8 [+3.9, +7.8] over the placebo at 25%.
  - **Knights-and-Knaves**: +4.5 over 12-epoch answer-only; 20-epoch answer-only ties in accuracy and loses 8-10 BBH points.
  - EntailmentBank at ceiling (96% every arm). QASC and further seeds finished on the HPC but cannot be pulled yet.
- **Gemma-4-12B** (seed 12): GSM8K 65.9 vs tuned 62.8 (+3.0 [+1.1, +4.8]), placebo 63.7; arithmetic chains full 83.5 vs
  tuned 79.9 (+3.6 [+1.8, +5.4]); arithmetic chains 25% 74.2 vs 12-epoch answer-only 65.8 (+8.4) and placebo 54.2.
  K&K diverges at 3e-6 (rerun at 1e-6). Batch 2 (seed-13 replications, compute-matched 20 epochs) queued.
- New in this push: MuSiQue domain (`scripts/build_musique_tf.py`, treeplan `musique_tf`), rationale-LM baseline data
  (`scripts/build_gsm8k_rationale.py`, arm A7), chain-of-thought upper bound (`scripts/eval_cot.py`, `hpc/eval_cot.sbatch`).
- Controls: `GFPSR` / `GFPQR` (node targets shuffled within a problem) and `GFPSX` / `GFPQX` (placebo: surface questions
  about the problem text, same count, views and yes rate).
- Failure attribution (separate collaboration, not in the preprint): AFTraj in-domain 80.4-83.4 Hit@1 for the 0.8B
  one-pass model (StepFinder 73.8, ReCast 84.1); real-failure study (TrajErrBench + AgentProcessBench, split by task):
  Tree Forcing 45.9 vs answer-only 40.4 in domain (+5.5 [-0.9, +12.4]); RootSE held out at chance for both.

## Repository layout

| Path | What it holds |
| --- | --- |
| `src/cotdistill/student.py` | The one-pass student: letter readout from the LM head, KL loss to targets, sub-question / tree-node example builders (`build_full_tree`, views, packing) |
| `src/cotdistill/treeplan.py` | Tree-node planning, torch-free: statement blocks per data source (`statement_block`, `problem_no_options`), node orders, leak checks |
| `src/cotdistill/numgen.py`, `pronto.py`, `chaingen.py`, `rulegen.py`, `policygen.py`, `sources.py` | Data generators and source converters (numeric chains, PrOntoQA-OOD-style proofs, decision chains, rule and policy cases) |
| `src/cotdistill/fa_tier0.py`, `fa_public.py` | Failure attribution: simulated multi-agent trajectories with replay-verified labels (Tier 0); cleaned public datasets (Tier 1) |
| `src/cotdistill/fa_trees.py`, `scripts/extract_fa_trees.py` | Failure attribution: multiple-choice reasoning trees on StepFinder's trajectories (answer step, provenance chain, content, conflict, agent) and their automatic checks (chain, quotes, leaks, length / log-overlap / surface-form tells) |
| `src/cotdistill/leangen.py`, `scripts/build_lean_tf.py` | Lean 4 pilot: generated congruence chains whose answers, distractors (refuted, independent, true-but-off-target) and tree nodes are certified by the Lean kernel |
| `scripts/build_fa_sf.py`, `build_fa_sf_x.py`, `fa_pool.py` | StepFinder's protocol: one model per Who&When subset, its training trajectories, Who&When test files (`--trees` swaps in the reasoning trees); the public test sets in the same rendering; seed-averaged paired comparisons |
| `scripts/build_fa_aftraj.py`, `build_fa_real.py`, `build_fa_longrca.py` | Failure attribution beyond StepFinder: AFTraj in-domain (ReCast's protocol), observed failures (TrajErrBench, AgentProcessBench, RootSE), LongRCA |
| `scripts/build_strategyqa_tf.py` | StrategyQA with nodes read off its decompositions and annotated facts |
| `scripts/plot_data_efficiency.py` | Accuracy vs share of the training set (figure) |
| `src/cotdistill/fsdp.py`, `offload.py` | Two-GPU FSDP2 training and CPU-offloaded AdamW (fp32 master weights) |
| `src/cotdistill/interchange.py` | Interchange interventions for the causal audit |
| `src/cotdistill/teacher.py`, `deepseek.py`, `openrouter.py`, `readout.py` | Teacher pipeline (DeepSeek only) and log-probability readout |
| `src/cotdistill/evalsets.py`, `metrics.py` | Held-out evaluation sets and torch-free metrics |
| `scripts/train_student.py` | Training entry point (all arms; `--tree-full`, `--tree-view`, `--tree-pack`, `--token-budget`, `--fsdp`, `--chat-template`) |
| `scripts/build_*.py` | Data builders (`build_numgen_tf.py`, `build_pronto_tf.py`, `build_gsm8k_tf.py`, `build_kk_tf.py`, `build_fa_tier0.py`, `build_fa_public.py`, `build_v7.py`, `build_v8.py`, ...) |
| `scripts/ladder_readout.py`, `depth_readout.py`, `fa_readout.py` | Readouts: arms x seeds x learning rates with paired bootstrap CIs; difficulty shift; failure-attribution baselines |
| `scripts/kk_audit.py`, `kk_audit_endpoints.py`, `kk_audit_diag.py` | Causal audit (DAS interchange interventions, probes) and its registered endpoints |
| `scripts/grad_subspace.py` | One-step full-gradient subspaces vs the finished update (link to LoRA-One, arXiv 2502.01235) |
| `scripts/kk_reasoning_type.py` | Knights-and-Knaves test items split by hypothesis structure (one vs several or nested case splits); arm differences per subgroup |
| `scripts/eval_intern_format.py`, `eval_llm_baseline.py` | Evaluating third-party models in their own request format; chat-template baselines |
| `hpc/train_arm.sbatch` | Slurm job for one training run; `arm_config` lists every arm (A2, G3, GFPS, GFPQ, GFPSP, TFL, GTF, ...) |
| `hpc/*.sbatch`, `hpc/*.sh` | Other jobs (audit, Intern-format eval, gradient subspaces, memory tests) and data push / fetch helpers |
| `configs/v7`, `configs/v8` | Data-mixture configs for the release model |
| `cloud/` | Rented-GPU scripts; `cloud/condor/` + `cloud/pack_condor.sh`: an HTCondor bundle for the Gemma-4-12B runs (pinned code, data, per-job settings dumped from the Slurm relay logic); `cloud/chtc/` + `cloud/pack_chtc.sh`: its CHTC variant (container, /staging, file transfer, DAG) |
| `tests/` | CPU unit tests (tiny random models) |

## Main arms

| Arm | Supervision | Context of each node question |
| --- | --- | --- |
| `A2` | Final answer only | n/a |
| `G3` | Final + 2 sampled nodes | Full problem with question and options |
| `GFPS` (Tree Forcing) | Final + every node | Problem statements, without the final question and options |
| `GFPQ` | Final + every node | Statements plus the final question, without options |
| `GFPSNC` | As GFPS without hypothetical case nodes (`--tree-node-types parse,derive`) | Statements |
| `GFPSP` | As GFPS, up to 8 node questions packed per sequence | Same; later questions see earlier questions, never answers |
| `TFL` | Final + every node | Parents' results written as text, no problem |
| `GTF` | GFPS + fading written parents + depth weights | Statements |
| `GFPSR` / `GFPQR` | As GFPS / GFPQ with each problem's node targets shuffled among its nodes | Same |
| `GFPSX` / `GFPQX` | As GFPS / GFPQ with each node question replaced by a surface question about the problem text (placebo) | Same |
| `GFPQV` / `GFPQVQ` | Final + atomic computation steps as three-option value questions ("What is 4 + 2?"; GFPQVQ prefixes the sub-question) | Statements plus the final question |
| `GFPQO` | Final + operation choice per step ("Which computation gives it? A) 4 × 2 B) 4 ÷ 2 C) 4 + 2") | Same |
| `GFPVB` | As GFPQV without the problem text | None (bare arithmetic) |
| `GFPQVM` | GFPQ's yes/no nodes + GFPQV's value nodes, half the node weight each | Statements plus the final question |
| `GFPQVT` | Final + verification nodes, one per asked option ("In this problem, what does option C (8) stand for?": the final answer / a solution step's sub-question / a number written in the problem / none of the solution's quantities) | Full problem with its options |
| `GFPQVTR` | As GFPQVT with each problem's descriptions permuted among its options (wrong-content control) | Same |
| `SI` / `SIB` | Stepwise internalisation: the worked atomic chain written before the options, removed over 4 stages (SI from the start, SIB from the answer end), last stage = test format | Final question only |
| `GFPQC` | Step curriculum: yes/no nodes of steps <= j, stage by stage, then the final questions (--curriculum-steps) | Statements plus the final question |
| `SIYB` / `SISB` | SIB + GFPQ's yes/no nodes / SIB with wrong values in the chain (control) | Same |
| `LS` | Latent node slots: K x c reserved tokens before the options in every final question (train and test); in training, slot j's final-layer states, fed to the same network as the start of a separate sequence, must continue with worked step j (`--latent-slots`) | Final question only (one pass, same readout) |
| `LSU` / `LSR` / `LSY` | The slots without slot loss (pause-token control) / slot targets from other problems (wrong-content control) / LS + GFPQ's yes/no nodes | Same / same / + statements plus the final question |

## Setup

Python 3.11+ with PyTorch 2.x and Transformers 5.x (developed with torch 2.11 and transformers 5.17). Training runs on
GPUs (two 40 GB A100 MIG slices with FSDP2, or one 80 GB A100); the local machine is for CPU work only.

```bash
pip install torch transformers tokenizers safetensors numpy
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -q
```

## Typical use

```bash
# build the numeric-chain domain (train depth <= 4, tests depth 1-8)
python scripts/build_numgen_tf.py
# one run on a cluster (see the header of hpc/train_arm.sbatch for every variable)
ARMS=GFPS SEED=12 TAG=c4 EPOCHS=4 DATA=data/student_numgen_tf MODEL=models/Qwen3.5-4B LR=3e-6 sbatch hpc/train_arm.sbatch
# readout over runs pulled to results/hpc/<batch>/
python scripts/ladder_readout.py 'results/hpc/l4b/*' --out results/ladder_readout/<name>
```

## Data and rules

- Data are not tracked. Builders write to `data/`; raw downloads go to `data/raw/<source>/`.
- Training data never come from GPT or Codex outputs; DeepSeek is the teacher model. Exception by the owner's decision:
  the failure-attribution reasoning trees are written by Claude Opus and tagged `"teacher": "claude-opus"`.
- Evaluation sets are evaluation-only and deduplicated against training data (8-gram overlap checks in the builders).
- Licences of third-party data are recorded per source; some sources are research-only.

## Changelog

The README is updated with every push.

- 2026-10-21: verification trees (GFPQVT / GFPQVTR: what does each option stand for) and other-problem distractor test sets (gsm8kx / gsm1kx; options-only adversary at chance); batch 12 results (step-1-only slots: partly written, never used); CHTC batch 13 = the verification-tree pilot plus the GSM8K-Aug 24k scale test, evaluated on both test-set kinds.
- 2026-10-20 (3): batch-11 pilot failed both gates (layer-20 readout 3%, at the shuffled-control level; necessity +0.15); read-only audit found no implementation bug (loss weight, fill and frozen-lens design). Latent-slot diagnostics: --slot-pad-weight, step-1-only K=1, scripts/probe_slots.py (every layer, logistic probes and frozen-head lens), sbatch PROBE / PROBE_ONLY; CHTC batch 12 = probes of LSL / LSLU / the base model and the step-1-only arm at 2 and 6 epochs.
- 2026-10-20 (2): latent slots v2.1 (arms LSL / LSLU / LSLR / LSLY: middle-layer slot loss, own-head value readout without a second pass, filled tail slots, distinct slot tokens; mean-ablation necessity and a pre-registered pilot gate); scripts/paired_stats.py (seed-level t, shared-index item bootstrap, two-level bootstrap, val-only epoch choice); CHTC batch 11 = the v2.1 pilot (batch 10 on hold).
- 2026-10-20: failure-attribution trees v3 (index-free pointers, teacher-written clean nodes) and LOFO training pools (AgenTracer Diagnosis, Who&When Pro text, CORRECT) with a tree post-check and label audit; CHTC batch 10 (slots + yes/no seeds 12 / 15 / 16; GSM8K-Aug 24k answer-only vs slots at 1 and 2 epochs).
- 2026-10-19 (3): CHTC batch 9 (Qwen3.5-9B, MC nodes at 1 epoch, seeds 12 / 13 / 14) for the 9B best-vs-best comparison.
- 2026-10-19 (2): Tree Forcing v2 latent node slots (arms LS / LSU / LSR / LSY, cb0e84f); CHTC batch 8 (latent-slot pilot);
  GSM8K-Aug builder (`scripts/build_gsm8kaug_tf.py`, 317,503 verified and decontaminated problems, 24k / 96k subsets).
- 2026-10-19: step curriculum (arm GFPQC); CHTC batch 7: 1- and 2-epoch runs of every new method, MC-node and answer-only
  seeds 13/14, curriculum with a full final stage. Answer-only peaks at 1 epoch on GSM8K 4B; all comparisons now best vs best.
- 2026-10-18 (3): CHTC batch 6 (`cloud/condor/jobs6_4b`): the new node designs and stepwise internalisation on GSM8K first;
  batch 5 trimmed to the GSM8K seeds and the answer-only epoch sweep.
- 2026-10-18 (2): value nodes (`scripts/build_gsm8k_ops.py`, arms GFPQV / GFPQVQ / GFPQO / GFPVB / GFPQVM) and stepwise
  internalisation v1 (`scripts/build_gsm8k_si.py`, arms SI / SIY / SIS and answer-end-first SIB / SIYB / SISB).
- 2026-10-18: CHTC batch 5 (53 jobs, `cloud/condor/jobs5_*`, `cloud/chtc/*5*`): seeds and epoch sweeps for answer-only and
  Tree Forcing in one setting (GSM8K 10/25/50/100%, K&K, arithmetic chains 25%, 12B K&K at 1e-6, 9B GSM8K); GPU preflight.
- 2026-10-17 (5): paper v2 draft with calibration (Section 4.4); CHTC batch 4 with Qwen3.5-9B (`cloud/chtc/*_9b*`,
  `cloud/condor/jobs9b`): GSM8K and arithmetic-chain 25% trios for a same-family scale ladder.
- 2026-10-17 (4): 4B batch update 2: GFPQR, step-capped GSM8K nodes (`scripts/filter_tree_steps.py`), answer-only 1e-5,
  Tree Forcing 25% 12 epochs; anchors save weights; MuSiQue Tree Forcing / placebo held until the node negatives are fixed.
- 2026-10-17 (3): after Milo's review: CHTC batch 2 reordered (compute-matched first), 40 GB GPUs, 12B answer-only at
  1e-5, weights saved for the 12B GSM8K seed-13 pair, validation tracked in answer-only runs, 4B 25% epoch-selection runs.
- 2026-10-17 (2): CHTC batch 3 with Qwen3.5-4B (`cloud/chtc/*_4b*`, `cloud/condor/jobs4b`): GSM8K anchors, MuSiQue 25%
  quartet, compute-matched 20 epochs, A7 and its single-GPU twin, GSM8K 25% placebo s14; run tags end in ch / ch1.
- 2026-10-17: MuSiQue builder and treeplan branch; rationale-LM data builder (A7) and the chain-of-thought evaluation;
  CHTC batches 1-2 (K&K 3e-6 dropped, 25% 12-epoch baselines, seed-13 replications, compute-matched 20 epochs).

- 2026-10-15: preprint draft completed (ablations in the appendix; main table default / tuned / Tree Forcing; Gemma-4-12B
  comparison; corresponding-author email); answer-only LR sweep down to 3e-7; EntailmentBank and QASC builders and runs
  (`notes/tmp/relay98.sh`, `relay99.sh`); nested 50% arithmetic-chain subset; data-efficiency figure reads the same
  arithmetic-chain run as the tables.

- 2026-10-14: shuffled-target and placebo controls (`GFPSR`, `GFPQR`, `GFPSX`, `GFPQX`); StepFinder protocol test sets
  and paired statistics (`build_fa_sf_x.py`, `fa_pool.py`); failure-attribution builders for AFTraj in-domain,
  observed-failure data and LongRCA, with the tree tools extended to them; StrategyQA domain; data-efficiency figure;
  HTCondor bundle for Gemma-4-12B (`cloud/condor`); status rewritten with tuned and schedule-matched baselines and the
  placebo results. Each HTCondor job pushes its outputs (no weights) to the private repo
  tree-forcing-lab/cot-distillation-jev-results through a write deploy key (`cloud/condor/push_results.sh`). `cloud/chtc/` +
  `cloud/pack_chtc.sh`: the same batch for CHTC (no shared filesystem): Apptainer image (torch 2.11 + cu128 for CUDA
  12.8 drivers), Gemma and the image in /staging, code and data transferred per job, a DAG whose POST script pushes
  each result.

- 2026-10-12: Lean F1 pilot (`leangen.py`, `build_lean_tf.py`, a `lean_f1` statement block; 2,400 certified items,
  surface-tell battery near chance); status rewritten with the four-domain results at tuned learning rates.

- 2026-10-10 (3): multiple-choice tree nodes (opt-in by an `options` field), `--eval-mid-preds`, `eval_student --time`;
  `build_fa_sf.py` (StepFinder protocol) and `fa_readout.py --sf` (Acc@K, MRR@3, tolerance accuracy); `fa_trees.py` /
  `extract_fa_trees.py` (reasoning trees for failure attribution with automatic checks, incl. answer-step rule fixes,
  `rule_override`, log-overlap and surface-form checks).

- 2026-10-10 (2): `--tree-node-types` and arm `GFPSNC`; `build_kk_tf.py` (single-domain Knights-and-Knaves, solver trees, 1,792 puzzles by default so Tree Forcing costs about as much per epoch as on numeric chains); status rewritten: science line on single-domain data only.

- 2026-10-10: `scripts/kk_reasoning_type.py`: the Tree Forcing vs answer-only gap on Knights-and-Knaves does not depend on how many hypothetical case splits a puzzle needs (interaction about 0). Status: learning-rate sweep on two domains (math: Tree Forcing ahead at tuned rates; Knights-and-Knaves: tie), PrOntoQA-OOD and GSM8K runs queued.

- 2026-10-09: first README. Since the previous push: Tree Forcing variants GFPQ and GFPSP (packed node questions),
  token-budget batching for every arm on one GPU, PrOntoQA-OOD and GSM8K tree domains, failure-attribution Tier 0 and
  Tier 1 data builders with cleaning and debiasing, causal-audit v2, learning-rate and difficulty-shift readouts,
  Intern-Decision-format evaluator, gradient-subspace analysis.
