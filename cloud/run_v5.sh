#!/usr/bin/env bash
# v5 recipe runs on a rented box with two GPUs, one run per GPU, with the training flags of TF-v4t
# (runs/TF-v4t-.../metrics.json args: full tree, 2 epochs, 32 problems per update, lr 1e-5, label smoothing .1,
# final-option permutation .5, fp32 master weights) on data/student_v5. The two runs differ only in the node format:
#   M1  --aux-total 0.5                      (yes/no nodes at half weight)
#   M2  --aux-total 0.5 --node-format mc     (the same nodes as multiple-choice views)
# A smoke stage (16 problems, half an epoch, each arm) runs first. Runs are skipped once their metrics.json exists.
#   cd ~/cotd && setsid nohup bash repo/cloud/run_v5.sh > run_v5.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."                      # ~/cotd
PY=env/bin/python
MODEL=models/Qwen3.5-2B-Base; DATA=data/student_v5; E=data/eval
EVALS="val=$DATA/val.jsonl kk=$E/kk_heldout.jsonl jl=$E/jl_heldout.jsonl jevbench=$E/jevbench_public.jsonl \
td=$E/typed_decisions_test.jsonl bbeh=$E/bbeh.jsonl bbh=$E/bbh.jsonl musr=$E/musr.jsonl policy=$E/policy_heldout.jsonl \
sharc=$E/sharc_dev.jsonl folio=$E/folio_val.jsonl kkdeep=$E/kk_deep.jsonl gsm8k=$E/gsm8k_mc.jsonl diag=$E/diag_kk.jsonl \
v4heldout=$E/v4_heldout.jsonl tdq=$E/td_noul_qform.jsonl tdn=$E/td_noul_neg.jsonl claims_ho=$E/claims_ho.jsonl \
xkk=$E/xkk.jsonl proverqa=$E/proverqa.jsonl zebra=$E/zebra.jsonl logiqa2=$E/logiqa2.jsonl lsat_lr=$E/lsat_lr.jsonl \
lsat_ar=$E/lsat_ar.jsonl"
for f in $EVALS; do [ -f "${f#*=}" ] || { echo "missing eval file ${f#*=}"; exit 1; }; done
mkdir -p runs logs
declare -A FLAGS=([M1]="--aux-total 0.5" [M2]="--aux-total 0.5 --node-format mc")
train() {                                       # train <arm> <gpu> <train.jsonl> <evals> <out> [extra]
  local arm=$1 gpu=$2 tr=$3 ev=$4 out=$5; shift 5
  CUDA_VISIBLE_DEVICES=$gpu $PY repo/scripts/train_student.py --model "$MODEL" --train "$tr" --eval $ev \
    --final teacher --subq cot --subq-target cot --lambda-sub 1.0 --tree-full true --tree-cap 10 \
    --label-smoothing 0.1 --permute-final 0.5 --lr 1e-5 --warmup 0.05 --max-len 1536 --precision fp32master \
    --optim adamw --items-per-update 32 --micro-bs 4 --grad-accum 16 --eval-bs 8 --seed 0 --out "$out" \
    --eval-subq "$DATA/val.jsonl" ${FLAGS[$arm]} "$@"
}
echo "=== smoke $(date +%T)"
SM=$(mktemp -d)
head -n 16 "$DATA/train.jsonl" > "$SM/train.jsonl"
head -n 20 "$E/kk_heldout.jsonl" > "$SM/eval.jsonl"
for arm in M1 M2; do
  train $arm 0 "$SM/train.jsonl" "smoke=$SM/eval.jsonl" "$SM/out-$arm" --epochs 0.5 --items-per-update 4 \
    > "$SM/$arm.log" 2>&1 && grep -q '^smoke {' "$SM/$arm.log" \
    || { echo "SMOKE FAILED $arm"; tail -n 30 "$SM/$arm.log"; exit 1; }
  echo "smoke ok $arm"
done
rm -rf "$SM"
echo "=== smoke passed $(date +%T)"
gpu=0
for arm in M1 M2; do
  out="runs/TF-$arm-Qwen3.5-2B-Base-student_v5-s0"
  if [ -f "$out/metrics.json" ]; then echo "skip $out (done)"; continue; fi
  echo "=== $arm on GPU $gpu -> $out $(date +%T)"
  ( train $arm $gpu "$DATA/train.jsonl" "$EVALS" "$out" --epochs 2 --save > "logs/$arm.log" 2>&1 \
      || echo "run $arm failed (logs/$arm.log)"; echo "done $out $(date +%T)" ) &
  gpu=$((gpu + 1))
done
wait
echo "all done $(date +%T)"
