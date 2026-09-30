#!/bin/bash
#SBATCH --job-name=cotd-check
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=logs/%x-%j.out
# Short check before long jobs: kernels, full-FT memory for the 2B student, eval and training speed.
# From ~/cotd:  sbatch repo/hpc/check.sbatch
set -uo pipefail
cd "$HOME/cotd"
PY=envs/cotd/bin/python
export HF_HOME="$HOME/cotd/hf" TOKENIZERS_PARALLELISM=false
M=models/Qwen3.5-2B-Base
nvidia-smi -L
$PY -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
$PY -c "import fla; print('fla', fla.__version__)" || echo "flash-linear-attention missing (torch fallback)"
echo "=== memory"
$PY repo/hpc/mem_smoke.py --model $M --seq 1536 --bs 8 --steps 4 --modes fp32master_adamw_ckpt,bf16_adamw8bit_ckpt
T=$(mktemp -d)
head -n 100 data/eval/kk_heldout.jsonl > $T/kk100.jsonl
head -n 200 data/student/train.jsonl > $T/train200.jsonl
echo "=== eval speed (base model)"
/usr/bin/time -v $PY repo/scripts/eval_student.py --model $M --eval jevbench=data/eval/jevbench_public.jsonl kk100=$T/kk100.jsonl \
  --out runs/check-base 2>&1 | grep -E "^(jevbench|kk100) |Elapsed|Maximum resident"
echo "=== train speed (A3-style, 200 items, fp32 master + AdamW)"
/usr/bin/time -v $PY repo/scripts/train_student.py --model $M --train $T/train200.jsonl --eval kk100=$T/kk100.jsonl \
  --final teacher --subq cot --epochs 0.1 --log-every 2 --out runs/check-a3 2>&1 | grep -E "examples/epoch|step|kk100 |Error|error|Elapsed"
echo "done"
