#!/usr/bin/env bash
# E2 (exposure-matched grouped training, collaborator review) on a rented GPU, with the flags of the HPC G0/G3/G4 jobs
# (hpc/train_arm.sbatch). Per seed: G3 CoT nodes, G4 matched controls, G0 placebo; 451 grouped updates of 32 problems x
# (1 final + 2 auxiliary views). Runs are named <arm>-a100-...-s<seed> and skipped once their metrics.json exists, so
# rerunning the script resumes. Micro-batch 4 without gradient checkpointing (80 GB); on OOM the run is retried with
# checkpointing, then with micro-batch 2 (the grouped update and its loss normalisation do not depend on these).
#   cd ~/cotd && setsid nohup bash repo/cloud/run_e2.sh "0 1" > run_e2.out 2>&1 < /dev/null &
# Options (environment): CC1D=1 causal-conv1d kernel; EVAL_BS (default 8); DUMP_LAYERS e.g. 4,8,12,16,20,24; SMOKE=0;
# SAVE=1 keeps the final weights (bf16, ~4 GB per run) for interventions and later diagnostics.
set -uo pipefail
cd "$(dirname "$0")/../.."                      # ~/cotd
SEEDS=${1:-"0 1"}; SMOKE=${SMOKE:-1}; ARMS_LIST=${ARMS_LIST:-"G3 G4 G0"}
PY=env/bin/python
# CC1D=1: causal-conv1d CUDA kernel from /root/cc1d (built by build_cc1d.sh); keep it fixed within a seed's arms
[ -n "${CC1D:-}" ] && export PYTHONPATH="/root/cc1d${PYTHONPATH:+:$PYTHONPATH}"
MODEL=models/Qwen3.5-2B-Base; DATA=data/student_v3; E=data/eval
EVALS="val=$DATA/val.jsonl kk=$E/kk_heldout.jsonl jl=$E/jl_heldout.jsonl jevbench=$E/jevbench_public.jsonl \
td=$E/typed_decisions_test.jsonl bbeh=$E/bbeh.jsonl bbh=$E/bbh.jsonl musr=$E/musr.jsonl policy=$E/policy_heldout.jsonl \
sharc=$E/sharc_dev.jsonl folio=$E/folio_val.jsonl kkdeep=$E/kk_deep.jsonl gsm8k=$E/gsm8k_mc.jsonl \
val_facts=$E/val_facts.jsonl diag=$E/diag_kk.jsonl"
for f in $EVALS; do [ -f "${f#*=}" ] || { echo "missing eval file ${f#*=}"; exit 1; }; done
mkdir -p runs logs

arm_args() {                                    # same subq/target/aux settings as the sbatch G arms
  case $1 in
    G0) echo "--subq cot --subq-target cot --aux-kind placebo" ;;
    G3) echo "--subq cot --subq-target cot --aux-kind cot" ;;
    G4) echo "--subq random --subq-target fresh --aux-kind random" ;;
    TF) echo "--subq cot --subq-target cot --tree-full true" ;;
    TFM) echo "--subq cot --subq-target cot --tree-full matched" ;;
    GF) echo "--subq cot --subq-target cot --tree-full plain" ;;
    GF0) echo "--subq cot --subq-target cot --tree-full placebo" ;;
    G3L) echo "--subq cot --subq-target cot --aux-kind cot --aux-sources knights_knaves,justlogic,folio" ;;
    G3D) echo "--subq cot --subq-target cot --aux-kind cot --aux-sources sharc,returns,expense,subscription" ;;
    T1) echo "--subq cot --subq-target cot --aux-kind tree" ;;
    T1S) echo "--subq cot --subq-target cot --aux-kind tree_shuf" ;;
    T2) echo "--subq cot --subq-target cot --aux-kind tree --fact-withdraw 0.75" ;;
  esac
}
train() {                                       # train <arm> <seed> <train.jsonl> <evals> <out> <mbs> <ckpt 0|1> [extra]
  local arm=$1 seed=$2 tr=$3 ev=$4 out=$5 mbs=$6 ckpt=$7; shift 7
  $PY repo/scripts/train_student.py --model "$MODEL" --train "$tr" --eval $ev --final teacher $(arm_args "$arm") \
    --lambda-sub 1.0 --lr 1e-5 --max-len 1536 --precision fp32master --optim adamw --seed "$seed" --out "$out" \
    --eval-subq "$DATA/val.jsonl" --grouped-aux 2 --aux-weight 0.5 --micro-bs "$mbs" --eval-bs "${EVAL_BS:-8}" \
    $([ "$ckpt" = 1 ] || echo --no-grad-ckpt) "$@"
}
run_with_fallback() {                           # run_with_fallback <log> <train args without mbs/ckpt>...
  local log=$1; shift
  local arm=$1 seed=$2 tr=$3 ev=$4 out=$5; shift 5
  for cfg in "4 0" "4 1" "2 1"; do
    set -- $cfg "$@"; local mbs=$1 ckpt=$2; shift 2
    echo "$(date +%T) $arm s$seed: micro-batch $mbs, grad checkpointing $ckpt" | tee -a "$log"
    train "$arm" "$seed" "$tr" "$ev" "$out" "$mbs" "$ckpt" "$@" >> "$log" 2>&1 && return 0
    grep -qiE "out of memory|OutOfMemoryError" "$log" || return 1
    echo "$(date +%T) OOM, retrying with a smaller configuration" | tee -a "$log"
  done
  return 1
}

if [ "$SMOKE" = 1 ]; then
echo "=== smoke $(date +%T)"
SM=$(mktemp -d)
head -n 16 "$DATA/train.jsonl" > "$SM/train.jsonl"
head -n 20 "$E/kk_heldout.jsonl" > "$SM/eval.jsonl"
for arm in $ARMS_LIST; do
  run_with_fallback "$SM/$arm.log" "$arm" 0 "$SM/train.jsonl" "smoke=$SM/eval.jsonl" "$SM/out-$arm" \
    --epochs 0.5 --items-per-update 4 --dump-hidden smoke \
    || { echo "SMOKE FAILED $arm"; tail -n 30 "$SM/$arm.log"; exit 1; }
  grep -q '^smoke {' "$SM/$arm.log" || { echo "SMOKE FAILED $arm (no eval line)"; tail -n 30 "$SM/$arm.log"; exit 1; }
  echo "smoke ok $arm"
done
rm -rf "$SM"
echo "=== smoke passed $(date +%T)"
fi

for seed in $SEEDS; do
  for arm in $ARMS_LIST; do
    out="runs/$arm-a100-Qwen3.5-2B-Base-student_v3-s$seed"
    if [ -f "$out/metrics.json" ]; then echo "skip $out (done)"; continue; fi
    echo "=== $arm seed $seed -> $out $(date +%T)"
    ep=2; case $arm in TF|TFM|GF|GF0) ep=1 ;; esac      # full-tree arms: one pass with every node
    run_with_fallback "logs/$arm-s$seed.log" "$arm" "$seed" "$DATA/train.jsonl" "$EVALS" "$out" \
      --epochs $ep --dump-hidden val,kk,kkdeep,diag ${DUMP_LAYERS:+--dump-layers "$DUMP_LAYERS"} ${SAVE:+--save} \
      || echo "arm $arm seed $seed failed (see logs/$arm-s$seed.log)"
    grep -E '"step": (20|100|451),' "logs/$arm-s$seed.log" | tail -2 | cut -c1-120
    echo "done $out $(date +%T)"
  done
done
echo "all done $(date +%T)"
