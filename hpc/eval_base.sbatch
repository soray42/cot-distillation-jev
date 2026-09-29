#!/bin/bash
#SBATCH --job-name=cotd-evalbase
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# Untrained base control and the same-backbone answer-only replica (Jeff-Qwen3.5-2B, Apache-2.0),
# scored with our harness and prompts. From ~/cotd:  sbatch repo/hpc/eval_base.sbatch
set -euo pipefail
cd "$HOME/cotd"
module load cuda/12.4 2>/dev/null || true
export HF_HOME="$HOME/cotd/hf" TOKENIZERS_PARALLELISM=false
PY=envs/cotd/bin/python
$PY -c "from huggingface_hub import snapshot_download as s; import os; print(s('mstrasser/Jeff-Qwen3.5-2B', local_dir=os.path.expanduser('~/cotd/models/Jeff-Qwen3.5-2B')))"
EVALS="val=data/student/val.jsonl kk=data/eval/kk_heldout.jsonl jl=data/eval/jl_heldout.jsonl jevbench=data/eval/jevbench_public.jsonl td=data/eval/typed_decisions_test.jsonl"
for M in Qwen3.5-2B-Base Jeff-Qwen3.5-2B; do
  echo "=== $M"
  $PY repo/scripts/eval_student.py --model "models/$M" --eval $EVALS --out "runs/base-$M" || echo "eval failed for $M"
done
