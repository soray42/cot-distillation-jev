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
# Defaults from check job 677208: fp32 master weights + AdamW at micro-batch 2 x grad-accum 16 peaks at 32 GiB
# on the 4g.40gb slice (bf16 + 8-bit Adam at micro-batch 4 peaks at 12.6 GiB). EVAL_BASE=1 first evaluates the
# untrained base and Jeff-Qwen3.5-2B; DATA picks the student data directory.
set -euo pipefail
cd "$HOME/cotd"
module load cuda/12.4 2>/dev/null || true
SEED="${SEED:-0}"; MODEL="${MODEL:-$HOME/cotd/models/Qwen3.5-2B-Base}"; DATA="${DATA:-data/student}"
export HF_HOME="$HOME/cotd/hf" TOKENIZERS_PARALLELISM=false
PY=envs/cotd/bin/python
EVALS="val=$DATA/val.jsonl kk=data/eval/kk_heldout.jsonl jl=data/eval/jl_heldout.jsonl jevbench=data/eval/jevbench_public.jsonl td=data/eval/typed_decisions_test.jsonl bbeh=data/eval/bbeh.jsonl"
nvidia-smi -L
if [ ! -f "$DATA/train.jsonl" ]; then       # sub-question data not pushed yet: never fall back to old sub-questions
  echo "missing $DATA; answer-only arms on data/student"; DATA=data/student; ONLY_ANSWER=1
  EVALS="val=$DATA/val.jsonl kk=data/eval/kk_heldout.jsonl jl=data/eval/jl_heldout.jsonl jevbench=data/eval/jevbench_public.jsonl td=data/eval/typed_decisions_test.jsonl bbeh=data/eval/bbeh.jsonl"
fi
if [ -n "${EVAL_BASE:-}" ]; then          # untrained base control and the same-backbone answer-only replica
  USED=$(/userScripts/lquota 2>/dev/null | sed -n 's/.*USED=\([0-9.]*\)GB.*/\1/p')
  echo "lquota used ${USED:-?} GB of 50"
  if [ ! -d models/Jeff-Qwen3.5-2B ] && awk "BEGIN{exit !(${USED:-99} <= 42)}"; then   # keep >= 3 GB after 4.5 GB
    $PY -c "from huggingface_hub import snapshot_download as s; import os; s('mstrasser/Jeff-Qwen3.5-2B', local_dir=os.path.expanduser('~/cotd/models/Jeff-Qwen3.5-2B'))" \
      || echo "Jeff download failed"
  fi
  for M in Qwen3.5-2B-Base Jeff-Qwen3.5-2B; do
    echo "=== base eval $M  $(date)"
    [ -d "models/$M" ] && $PY repo/scripts/eval_student.py --model "models/$M" --eval $EVALS --out "runs/base-$M" \
      || echo "base eval failed for $M"
  done
fi
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
  if [ -n "${ONLY_ANSWER:-}" ] && [ "$S" != none -o -n "$RLM" ]; then echo "skip $ARM (no tree data)"; continue; fi
  OUT="runs/${ARM}-$(basename "$MODEL")-$(basename "$DATA")-s${SEED}"
  echo "=== $ARM -> $OUT  $(date)"
  $PY repo/scripts/train_student.py --model "$MODEL" --train "$DATA/train.jsonl" --eval $EVALS \
    --final "$F" --subq "$S" --subq-target "$T" --lambda-sub "${LAMBDA:-1.0}" --epochs "${EPOCHS:-2}" \
    --lr "${LR:-1e-5}" --micro-bs "${MBS:-2}" --grad-accum "${GA:-16}" --max-len 1536 \
    --precision "${PREC:-fp32master}" --optim "${OPTIM:-adamw}" --seed "$SEED" --out "$OUT" ${SAVE:+--save} \
    ${RLM:+--rationale-lm --lm-max-len 3072} || echo "arm $ARM failed"
  echo "done $OUT"
done
