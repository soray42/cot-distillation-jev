#!/bin/bash
#SBATCH --job-name=cotd-train
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Student arms, run one after another inside one job (SLURM_COORD: one cotd- job in the system at a time).
# From ~/cotd:  ARMS="A1 A2 A7" SEED=0 sbatch repo/hpc/train_arm.sbatch   (or ARM=A3 for a single arm)
# Arms (final-question target x sub-question source / target):
#   A1 gold answer only            A2 teacher answer only         A3 teacher + CoT sub-questions (ours)
#   A4 teacher + matched control   A5 CoT sub-questions only      A6 teacher + CoT sub-q, program truth
#   A7 teacher + LM loss on the teacher's CoT and answer (DHRD-style rationale baseline)
# Override PREC/OPTIM after reading the memory smoke log (default: fp32 master weights + AdamW).
set -euo pipefail
cd "$HOME/cotd"
module load cuda/12.4 2>/dev/null || true
SEED="${SEED:-0}"; MODEL="${MODEL:-$HOME/cotd/models/Qwen3.5-2B-Base}"
export HF_HOME="$HOME/cotd/hf" TOKENIZERS_PARALLELISM=false
nvidia-smi -L
for ARM in ${ARMS:-$ARM}; do
  RLM=
  case "$ARM" in
    A1) F=gold;    S=none;   T=cot ;;
    A2) F=teacher; S=none;   T=cot ;;
    A3) F=teacher; S=cot;    T=cot ;;
    A4) F=teacher; S=random; T=fresh ;;
    A5) F=none;    S=cot;    T=cot ;;
    A6) F=teacher; S=cot;    T=truth ;;
    A7) F=teacher; S=none;   T=cot; RLM=1 ;;
    *) echo "unknown ARM $ARM"; exit 1 ;;
  esac
  OUT="runs/${ARM}-$(basename "$MODEL")-s${SEED}"
  echo "=== $ARM -> $OUT  $(date)"
  envs/cotd/bin/python repo/scripts/train_student.py --model "$MODEL" \
    --train data/student/train.jsonl \
    --eval val=data/student/val.jsonl kk=data/eval/kk_heldout.jsonl jl=data/eval/jl_heldout.jsonl \
           jevbench=data/eval/jevbench_public.jsonl td=data/eval/typed_decisions_test.jsonl bbeh=data/eval/bbeh.jsonl \
    --final "$F" --subq "$S" --subq-target "$T" --lambda-sub "${LAMBDA:-1.0}" --epochs "${EPOCHS:-2}" \
    --lr "${LR:-1e-5}" --micro-bs "${MBS:-8}" --grad-accum "${GA:-4}" --max-len 1536 \
    --precision "${PREC:-fp32master}" --optim "${OPTIM:-adamw}" --seed "$SEED" --out "$OUT" ${SAVE:+--save} \
    ${RLM:+--rationale-lm --lm-max-len 3072} || echo "arm $ARM failed"
  echo "done $OUT"
done
