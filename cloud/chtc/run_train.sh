#!/bin/bash
# Executable of train.sub (12B) and train_4b.sub (4B), inside the job's image on the execute node:
#   run_train.sh <env file in the bundle> <name>
# Lays the job out as ~/cotd (train_arm.sbatch works there), runs it exactly as on the original cluster, and packs the
# outputs (no weights) plus the log into results_<name>.tgz for the transfer back. Always writes that file.
# Code + data: $CODE_TGZ (default cotd_code_data.tgz, the 12B batch; train_4b.sub sets cotd_code_data_4b.tgz).
# Model: $STAGING/<basename of the env file's MODEL>.tar (models/gemma-4-12B -> gemma-4-12B.tar, models/Qwen3.5-4B ->
# Qwen3.5-4B.tar), read from the env file in a subshell right after the bundle is extracted.
set -uo pipefail
ENVF=$1; NAME=$2
TOP=${_CONDOR_SCRATCH_DIR:-$PWD}; cd "$TOP"
export HOME=$TOP; W=$HOME/cotd
mkdir -p "$W/logs" "$W/models" "$W/envs/cotd/bin"
pack() {
  local items=("logs/$NAME.out")
  [ -n "$OUTD" ] && [ -d "$W/$OUTD" ] && items+=("$OUTD")
  (cd "$W" && tar czf "$TOP/results_$NAME.tgz" --exclude='*/model' --exclude='*.safetensors' --exclude='*.pt' \
     "${items[@]}") || tar czf "$TOP/results_$NAME.tgz" -T /dev/null
}
OUTD=
echo "== $NAME on $(hostname) $(date)" > "$W/logs/$NAME.out"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader >> "$W/logs/$NAME.out" 2>&1
tar xzf "${CODE_TGZ:-cotd_code_data.tgz}" -C "$W" || { echo "bundle extract failed" >> "$W/logs/$NAME.out"; pack; exit 1; }
[ ! -f "$W/CODE_PIN" ] || echo "code pin: $(cat "$W/CODE_PIN")" >> "$W/logs/$NAME.out"   # 4B bundle only
MT=$(cd "$W" && . "$ENVF" > /dev/null 2>&1; [ -n "${MODEL:-}" ] && basename "$MODEL")
[ -n "$MT" ] || { echo "no MODEL in $ENVF" >> "$W/logs/$NAME.out"; pack; exit 1; }
tar xf "$STAGING/$MT.tar" -C "$W/models" || { echo "model extract failed ($STAGING)" >> "$W/logs/$NAME.out"; pack; exit 1; }
ln -sf "$(command -v python3)" "$W/envs/cotd/bin/python"
cd "$W"
SMOKE=${SMOKE_ONLY:-}; DRYR=${DRY_RUN:-}      # kept over the env file's blanks (DRY_RUN: local tests only)
set -a; . "$ENVF"; set +a
[ -n "$SMOKE" ] && export SMOKE_ONLY=1
[ -n "$DRYR" ] && export DRY_RUN=1
export SLURM_JOB_ID="chtc-$NAME" SLURM_CPUS_PER_TASK=12 HF_HUB_OFFLINE=1
OUTD="runs/${ARMS}${TAG:+-$TAG}-$(basename "$MODEL")-$(basename "$DATA")-s${SEED}"
bash repo/hpc/train_arm.sbatch >> "logs/$NAME.out" 2>&1
rc=$?
echo "== exit $rc $(date)" >> "logs/$NAME.out"
pack
exit $rc
