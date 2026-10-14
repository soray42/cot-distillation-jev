# Tree Forcing: teaching one-pass decision models to reason

Research code for training **one-pass decision models**: language models that answer a multiple-choice decision by reading
the probability of each option letter after a single forward pass, without writing any reasoning. The main question is
whether such models can learn multi-step reasoning when every intermediate step of a verified reasoning tree is supervised
as an extra yes/no question, asked in the context of the problem, with no premise written out ("Tree Forcing", arm
`GFPS`). The repository also covers a failure-attribution application (which step and which agent caused a multi-agent
system to fail) and the tooling for a causal audit of internal variables.

Owner: Haotian "Sora" Yang. Private research repository; data, results and notes are not tracked (see `.gitignore`).

## Status (2026-10-14)

- 4B, LR 3e-6 unless noted, seed 12 unless noted:
  - **GSM8K** (trees from GSM8K's own step annotations), Tree Forcing (`GFPQ`) minus answer-only, on GSM8K test + GSM1K:
    +5.6 / +3.7 / +2.9 over 12-epoch answer-only (3 seeds); +2.5 [+0.8, +4.1] over answer-only at 1e-6;
    +2.8 [+1.2, +4.5] over 4-epoch answer-only (Tree Forcing's schedule); +1.5 [-0.2, +3.1] over the placebo control.
  - **Data efficiency** (GSM8K): with 25% of the data, +9.0 / +6.7 over answer-only (2 seeds) and +1.9 [+0.2, +3.6] over
    the placebo; Tree Forcing reaches answer-only's full-data accuracy with about 17% of the data.
  - **Arithmetic chains**: ahead of 12-epoch answer-only (+3.0 to +4.4 in range, 3 seeds), but tied with 4-epoch or
    1e-6 answer-only and with the placebo at full data; with 25% of the data, +5.8 [+3.9, +7.8] over the placebo.
  - **Knights-and-Knaves**: +4.5 over 12-epoch answer-only; 20-epoch answer-only catches up in accuracy and loses 8-10
    points on BIG-Bench Hard (2 seeds).
- Controls: `GFPSR` / `GFPQR` (node targets shuffled within a problem) and `GFPSX` / `GFPQX` (placebo: surface questions
  about the problem text, same count, views and yes rate).
- In progress: StrategyQA, EntailmentBank and QASC as further real-data domains; Gemma-4-12B runs (`cloud/condor`).
- Failure attribution: AFTraj in-domain (ReCast's protocol) 80.4 Hit@1 for the 0.8B one-pass model (StepFinder 73.8,
  ReCast 84.1); Tree Forcing +15 on AFTraj's diagnosed (real) agentic errors, -32 on its injected ones (one seed, 46 / 22
  items); real-failure study on TrajErrBench + AgentProcessBench (train / test by task) with RootSE held out, running.

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
| `cloud/` | Rented-GPU scripts; `cloud/condor/` + `cloud/pack_condor.sh`: an HTCondor bundle for the Gemma-4-12B runs (pinned code, data, per-job settings dumped from the Slurm relay logic) |
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

- 2026-10-14: shuffled-target and placebo controls (`GFPSR`, `GFPQR`, `GFPSX`, `GFPQX`); StepFinder protocol test sets
  and paired statistics (`build_fa_sf_x.py`, `fa_pool.py`); failure-attribution builders for AFTraj in-domain,
  observed-failure data and LongRCA, with the tree tools extended to them; StrategyQA domain; data-efficiency figure;
  HTCondor bundle for Gemma-4-12B (`cloud/condor`); status rewritten with tuned and schedule-matched baselines and the
  placebo results. Each HTCondor job pushes its outputs (no weights) to the private repo
  soray42/cot-distillation-jev-results through a write deploy key (`cloud/condor/push_results.sh`).

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
