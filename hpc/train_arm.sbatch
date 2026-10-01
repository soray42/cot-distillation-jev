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
# From ~/cotd:  ARMS="A2 A3 A4 A1" DATA=data/student_tree EVAL_BASE=1 sbatch repo/hpc/train_arm.sbatch
# JEFF=1 evaluates Jeff with its own serving path (scripts/eval_jeff.py); BASE_SUBQ=1 scores the untrained base on
# the val sub-questions. Every arm is scored on the sub-questions of $DATA/val.jsonl (--eval-subq). DUMP=val,kk,...
# saves the readout hidden states of those eval sets (probing); PERM / BRIER / SUBQK / SUBQW set --permute-final /
# --lambda-brier / --subq-k / --subq-weight.
# MODEL picks the
# student backbone (default Qwen3.5-2B-Base; e.g. MODEL=$HOME/cotd/models/Qwen3.5-2B for the post-trained release).
# Before submitting, dry-run the shell logic on the login node (Python is only echoed):
#   DRY_RUN=1 ARMS="A2 A3" DATA=data/student_tree EVAL_BASE=1 bash repo/hpc/train_arm.sbatch
# The job starts with a smoke stage (every arm on 16 items for about 2 steps, plus a 20-item eval) and exits
# at once if any of it fails. Defaults from check job 677208: fp32 master weights + AdamW, micro-batch 2 x
# grad-accum 16 (peak 32 GiB on the 4g.40gb slice). EVAL_BASE=1 also evaluates the untrained $MODEL (models are
# downloaded on the login node beforehand; compute nodes have no /userScripts).
# Arms: A1 gold answer only · A2 teacher answer only · A3 teacher + CoT sub-questions (ours) · A4 teacher +
# matched control · A5 CoT sub-questions only · A6 teacher + CoT sub-q with truth targets · A7 teacher +
# LM loss on the teacher's CoT and answer · A9 depth curriculum: DSTAGES (default 4) passes over the items, each
# drawing STAGEK (default 1) CoT nodes per item from the next band of tree depth (half new band, half replay), then
# the final questions for EPOCHS, with a fresh LR schedule per stage.
set -euo pipefail
cd "$HOME/cotd"
module load cuda/12.4 2>/dev/null || true
SEED="${SEED:-0}"; MODEL="${MODEL:-$HOME/cotd/models/Qwen3.5-2B-Base}"; DATA="${DATA:-data/student}"
export HF_HOME="$HOME/cotd/hf" TOKENIZERS_PARALLELISM=false
PY=envs/cotd/bin/python
[ -n "${DRY_RUN:-}" ] && PY="echo [dry-run] python"
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi here"
[ -f "$DATA/train.jsonl" ] || { echo "missing $DATA/train.jsonl"; exit 1; }
EVALS="val=$DATA/val.jsonl kk=data/eval/kk_heldout.jsonl jl=data/eval/jl_heldout.jsonl jevbench=data/eval/jevbench_public.jsonl td=data/eval/typed_decisions_test.jsonl bbeh=data/eval/bbeh.jsonl ${EXTRA_EVALS:-}"
for f in $EVALS; do [ -f "${f#*=}" ] || { echo "missing eval file ${f#*=}"; exit 1; }; done

arm_config() {           # sets F S T RLM DS for arm $1
  RLM=; DS=
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
    *) echo "unknown ARM $1"; return 1 ;;
  esac
}

train() {                # train <arm> <train.jsonl> <evals> <out> [extra args...]
  local arm=$1 tr=$2 ev=$3 out=$4; shift 4
  arm_config "$arm"
  $PY repo/scripts/train_student.py --model "$MODEL" --train "$tr" --eval $ev \
    --final "$F" --subq "$S" --subq-target "$T" --lambda-sub "${LAMBDA:-1.0}" \
    --lr "${LR:-1e-5}" --max-len 1536 --precision "${PREC:-fp32master}" --optim "${OPTIM:-adamw}" \
    --seed "$SEED" --out "$out" --eval-subq "$DATA/val.jsonl" ${RLM:+--rationale-lm --lm-max-len 3072} \
    ${DUMP:+--dump-hidden "$DUMP,smoke"} ${PERM:+--permute-final "$PERM"} ${BRIER:+--lambda-brier "$BRIER"} \
    ${SUBQK:+--subq-k "$SUBQK"} ${SUBQW:+--subq-weight "$SUBQW"} ${DS:+--depth-stages "$DS" --stage-k "${STAGEK:-1}"} \
    ${SUBQF:+--subq-frac "$SUBQF"} "$@"
}

ARMS="${ARMS:-${ARM:-}}"
[ -n "$ARMS" ] || { echo "set ARMS"; exit 1; }

echo "=== smoke stage $(date +%T)"
SM=$(mktemp -d)
head -n 16 "$DATA/train.jsonl" > "$SM/train.jsonl"
head -n 20 data/eval/kk_heldout.jsonl > "$SM/eval.jsonl"
for ARM in $ARMS; do
  train "$ARM" "$SM/train.jsonl" "smoke=$SM/eval.jsonl" "$SM/out-$ARM" --epochs 0.5 --micro-bs 2 --grad-accum 2 \
    > "$SM/$ARM.log" 2>&1 || { echo "SMOKE FAILED $ARM"; tail -n 30 "$SM/$ARM.log"; exit 1; }
  [ -n "${DRY_RUN:-}" ] || grep -q '^smoke {' "$SM/$ARM.log" || { echo "SMOKE FAILED $ARM (no eval line)"; tail -n 30 "$SM/$ARM.log"; exit 1; }
  echo "smoke ok $ARM"
done
head -n 3 "$DATA/val.jsonl" > "$SM/val3.jsonl"
if [ -n "${EVAL_BASE:-}${BASE_SUBQ:-}" ]; then
  $PY repo/scripts/eval_student.py --model "$MODEL" --eval "smoke=$SM/eval.jsonl" --eval-subq "$SM/val3.jsonl" \
    --out "$SM/base" > "$SM/base.log" 2>&1 || { echo "SMOKE FAILED base eval"; tail -n 30 "$SM/base.log"; exit 1; }
  echo "smoke ok base eval"
fi
if [ -n "${JEFF:-}" ]; then
  $PY repo/scripts/eval_jeff.py --model models/Jeff-Qwen3.5-2B --sets jevbench td kk bbh musr --limit 20 --out "$SM/jeff" \
    > "$SM/jeff.log" 2>&1 || { echo "JEFF SMOKE SKIPPED: the Jeff eval is dropped, the job goes on"; tail -n 30 "$SM/jeff.log"; JEFF=; }
  [ -z "${JEFF:-}" ] || echo "smoke ok jeff eval"
fi
rm -rf "$SM"
echo "=== smoke passed $(date +%T)"

if [ -n "${JEFF:-}" ]; then               # Jeff with its own prompt, readout matrix and temperature
  echo "=== jeff eval  $(date +%T)"
  $PY repo/scripts/eval_jeff.py --model models/Jeff-Qwen3.5-2B --sets jevbench td kk jl bbeh bbh musr policy sharc folio \
    --out runs/jeff-native \
    || echo "jeff eval failed"
fi
if [ -n "${BASE_SUBQ:-}" ]; then         # sub-question accuracy of the untrained base (reference for the arms)
  echo "=== base sub-question eval  $(date +%T)"
  $PY repo/scripts/eval_student.py --model "$MODEL" --eval "kk=data/eval/kk_heldout.jsonl" \
    --eval-subq "$DATA/val.jsonl" --out "runs/base-subq-$(basename "$MODEL")" || echo "base eval failed for subq"
fi
if [ -n "${EVAL_BASE:-}" ]; then          # untrained backbone control (Jeff needs its own readout: JEFF=1)
  M=$(basename "$MODEL")
  echo "=== base eval $M  $(date +%T)"
  $PY repo/scripts/eval_student.py --model "$MODEL" --eval $EVALS --out "runs/base-$M" ${DUMP:+--dump-hidden "$DUMP"} \
    || echo "base eval failed for $M"
fi
for ARM in $ARMS; do
  OUT="runs/${ARM}${TAG:+-$TAG}-$(basename "$MODEL")-$(basename "$DATA")-s${SEED}"   # TAG names recipe variants
  echo "=== $ARM -> $OUT  $(date +%T)"
  train "$ARM" "$DATA/train.jsonl" "$EVALS" "$OUT" --epochs "${EPOCHS:-2}" --micro-bs "${MBS:-2}" \
    --grad-accum "${GA:-16}" ${SAVE:+--save} || echo "arm $ARM failed"
  echo "done $OUT  $(date +%T)"
done
echo "all done $(date +%T)"
