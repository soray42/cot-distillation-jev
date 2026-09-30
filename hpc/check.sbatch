#!/bin/bash
#SBATCH --job-name=cotd-check
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=logs/%x-%j.out
# Short check before long jobs: full-FT memory per precision and batch shape (one process each, so a
# failed attempt cannot hold memory for the next), then eval and training speed. From ~/cotd:
#   sbatch repo/hpc/check.sbatch
set -uo pipefail
cd "$HOME/cotd"
PY=envs/cotd/bin/python
export HF_HOME="$HOME/cotd/hf" TOKENIZERS_PARALLELISM=false
M=models/Qwen3.5-2B-Base
nvidia-smi -L
$PY -c "import fla, torch, transformers; print('fla', fla.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
echo "=== memory (seq bs mode)"
for mode in bf16_adamw8bit_ckpt fp32master_adamw_ckpt; do
  for cfg in "768 8" "1536 4" "1536 8"; do
    set -- $cfg
    echo "--- seq=$1 bs=$2"
    $PY repo/hpc/mem_smoke.py --model $M --seq $1 --bs $2 --steps 3 --modes $mode 2>&1 | grep -E "peak|OOM|FAILED|Error"
  done
done
T=$(mktemp -d)
head -n 100 data/eval/kk_heldout.jsonl > $T/kk100.jsonl
head -n 200 data/student/train.jsonl > $T/train200.jsonl
echo "=== eval speed (base model)  $(date +%T)"
$PY repo/scripts/eval_student.py --model $M --eval jevbench=data/eval/jevbench_public.jsonl kk100=$T/kk100.jsonl \
  --out runs/check-base 2>&1 | grep -v -i "warn\|loading weights"
echo "=== eval done $(date +%T)"
for p in "bf16 adamw8bit 4 8" "fp32master adamw 2 16"; do
  set -- $p
  echo "=== train speed (A3-style, 200 items) precision=$1 optim=$2 micro-bs=$3 grad-accum=$4  $(date +%T)"
  $PY repo/scripts/train_student.py --model $M --train $T/train200.jsonl --eval kk100=$T/kk100.jsonl \
    --final teacher --subq cot --epochs 0.1 --log-every 2 --precision $1 --optim $2 --micro-bs $3 --grad-accum $4 \
    --out runs/check-a3-$1 2>&1 | grep -v -i "warn\|loading weights"
  echo "=== train done $(date +%T)"
done
echo "done"
