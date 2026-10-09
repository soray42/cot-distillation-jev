# Tree Forcing: teaching one-pass decision models to reason

Research code for training **one-pass decision models**: language models that answer a multiple-choice decision by reading
the probability of each option letter after a single forward pass, without writing any reasoning. The main question is
whether such models can learn multi-step reasoning when every intermediate step of a verified reasoning tree is supervised
as an extra yes/no question, asked in the context of the problem, with no premise written out ("Tree Forcing", arm
`GFPS`). The repository also covers a failure-attribution application (which step and which agent caused a multi-agent
system to fail) and the tooling for a causal audit of internal variables.

Owner: Haotian "Sora" Yang. Private research repository; data, results and notes are not tracked (see `.gitignore`).

## Status (2026-10-09)

- At learning rate 1e-5 and equal compute, Tree Forcing beats answer-only training (`A2`) on Knights-and-Knaves puzzles
  larger than any seen in training (3 of 3 seeds) and on deeper arithmetic chains (2 of 2 seeds), and learns within one
  epoch in every seed.
- Supervising the same steps with their premises written out (`TFL`) reaches about 100% on the steps but lowers one-pass
  accuracy. The premises must stay latent.
- Open: with its own best learning rate (3e-6) the answer-only baseline matches Tree Forcing on the logic puzzles (one
  seed). A learning-rate sweep, 3 seeds at the selected rates, a packed variant (`GFPSP`) and data-efficiency runs
  are in progress.
- Open: the causal audit (`scripts/kk_audit.py`, version 2); the first version failed as a measurement.

## Repository layout

| Path | What it holds |
| --- | --- |
| `src/cotdistill/student.py` | The one-pass student: letter readout from the LM head, KL loss to targets, sub-question / tree-node example builders (`build_full_tree`, views, packing) |
| `src/cotdistill/treeplan.py` | Tree-node planning, torch-free: statement blocks per data source (`statement_block`, `problem_no_options`), node orders, leak checks |
| `src/cotdistill/numgen.py`, `pronto.py`, `chaingen.py`, `rulegen.py`, `policygen.py`, `sources.py` | Data generators and source converters (numeric chains, PrOntoQA-OOD-style proofs, decision chains, rule and policy cases) |
| `src/cotdistill/fa_tier0.py`, `fa_public.py` | Failure attribution: simulated multi-agent trajectories with replay-verified labels (Tier 0); cleaned public datasets (Tier 1) |
| `src/cotdistill/fsdp.py`, `offload.py` | Two-GPU FSDP2 training and CPU-offloaded AdamW (fp32 master weights) |
| `src/cotdistill/interchange.py` | Interchange interventions for the causal audit |
| `src/cotdistill/teacher.py`, `deepseek.py`, `openrouter.py`, `readout.py` | Teacher pipeline (DeepSeek only) and log-probability readout |
| `src/cotdistill/evalsets.py`, `metrics.py` | Held-out evaluation sets and torch-free metrics |
| `scripts/train_student.py` | Training entry point (all arms; `--tree-full`, `--tree-view`, `--tree-pack`, `--token-budget`, `--fsdp`, `--chat-template`) |
| `scripts/build_*.py` | Data builders (`build_numgen_tf.py`, `build_pronto_tf.py`, `build_gsm8k_tf.py`, `build_fa_tier0.py`, `build_fa_public.py`, `build_v7.py`, `build_v8.py`, ...) |
| `scripts/ladder_readout.py`, `depth_readout.py`, `fa_readout.py` | Readouts: arms x seeds x learning rates with paired bootstrap CIs; difficulty shift; failure-attribution baselines |
| `scripts/kk_audit.py`, `kk_audit_endpoints.py`, `kk_audit_diag.py` | Causal audit (DAS interchange interventions, probes) and its registered endpoints |
| `scripts/grad_subspace.py` | One-step full-gradient subspaces vs the finished update (link to LoRA-One, arXiv 2502.01235) |
| `scripts/eval_intern_format.py`, `eval_llm_baseline.py` | Evaluating third-party models in their own request format; chat-template baselines |
| `hpc/train_arm.sbatch` | Slurm job for one training run; `arm_config` lists every arm (A2, G3, GFPS, GFPQ, GFPSP, TFL, GTF, ...) |
| `hpc/*.sbatch`, `hpc/*.sh` | Other jobs (audit, Intern-format eval, gradient subspaces, memory tests) and data push / fetch helpers |
| `configs/v7`, `configs/v8` | Data-mixture configs for the release model |
| `cloud/` | Scripts used for one rented-GPU run |
| `tests/` | CPU unit tests (tiny random models) |

## Main arms

| Arm | Supervision | Context of each node question |
| --- | --- | --- |
| `A2` | Final answer only | n/a |
| `G3` | Final + 2 sampled nodes | Full problem with question and options |
| `GFPS` (Tree Forcing) | Final + every node | Problem statements, without the final question and options |
| `GFPQ` | Final + every node | Statements plus the final question, without options |
| `GFPSP` | As GFPS, up to 8 node questions packed per sequence | Same; later questions see earlier questions, never answers |
| `TFL` | Final + every node | Parents' results written as text, no problem |
| `GTF` | GFPS + fading written parents + depth weights | Statements |

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
- Training data never come from GPT, Codex or Claude outputs; DeepSeek is the only teacher model.
- Evaluation sets are evaluation-only and deduplicated against training data (8-gram overlap checks in the builders).
- Licences of third-party data are recorded per source; some sources are research-only.

## Changelog

The README is updated with every push.

- 2026-10-09: first README. Since the previous push: Tree Forcing variants GFPQ and GFPSP (packed node questions),
  token-budget batching for every arm on one GPU, PrOntoQA-OOD and GSM8K tree domains, failure-attribution Tier 0 and
  Tier 1 data builders with cleaning and debiasing, causal-audit v2, learning-rate and difficulty-shift readouts,
  Intern-Decision-format evaluator, gradient-subspace analysis.
