#!/bin/bash
#SBATCH --job-name=cotd-train
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# One student arm per job (SLURM_COORD: one cotd- job in the system at a time). From ~/cotd:
#   ARM=A3 SEED=0 sbatch repo/hpc/train_arm.sbatch
# Arms (final-question target x sub-question source / target):
#   A1 gold answer only            A2 teacher answer only         A3 teacher + CoT sub-questions (ours)
#   A4 teacher + matched control   A5 CoT sub-questions only      A6 teacher + CoT sub-q, program truth
# Override PREC/OPTIM after reading the memory smoke log (default: fp32 master weights + AdamW).
set -euo pipefail
cd "$HOME/cotd"
module load cuda/12.4 2>/dev/null || true
case "${ARM:?set ARM}" in
  A1) F=gold;    S=none;   T=cot ;;
  A2) F=teacher; S=none;   T=cot ;;
  A3) F=teacher; S=cot;    T=cot ;;
  A4) F=teacher; S=random; T=fresh ;;
  A5) F=none;    S=cot;    T=cot ;;
  A6) F=teacher; S=cot;    T=truth ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
SEED="${SEED:-0}"; MODEL="${MODEL:-$HOME/cotd/models/Qwen3.5-2B-Base}"
OUT="runs/${ARM}-$(basename "$MODEL")-s${SEED}"
export HF_HOME="$HOME/cotd/hf" TOKENIZERS_PARALLELISM=false
nvidia-smi -L
envs/cotd/bin/python repo/scripts/train_student.py --model "$MODEL" \
  --train data/student/train.jsonl \
  --eval val=data/student/val.jsonl kk=data/eval/kk_heldout.jsonl jl=data/eval/jl_heldout.jsonl \
         jevbench=data/eval/jevbench_public.jsonl td=data/eval/typed_decisions_test.jsonl \
  --final "$F" --subq "$S" --subq-target "$T" --lambda-sub "${LAMBDA:-1.0}" --epochs "${EPOCHS:-2}" \
  --lr "${LR:-1e-5}" --micro-bs "${MBS:-8}" --grad-accum "${GA:-4}" --max-len 1536 \
  --precision "${PREC:-fp32master}" --optim "${OPTIM:-adamw}" --seed "$SEED" --out "$OUT" ${SAVE:+--save}
echo "done $OUT"
