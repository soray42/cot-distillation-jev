#!/bin/bash
# Executable of the batch-5 submit files (train5_4b.sub, train5_9b.sub, train5_12b.sub), inside the job's image:
#   run_train5.sh <env file in the bundle> <name>
# = run_train.sh (md5 e61eff1e, batches 1-4; unchanged below the preflight) + a GPU preflight before anything is unpacked:
# a CUDA context, a small kernel and >= 90% free memory on every GPU the job was given, and at least NGPU_REQ GPUs
# (the DAG's ngpu). Jobs 70-73 (2026-10-18 01:15-01:20) all matched one slot on gpu2008.chtc.wisc.edu whose A100
# refused a CUDA context ("CUDA-capable device(s) is/are busy or unavailable" at torch.cuda.set_device) and died in
# 4 s each, so the slot took the next job at once. A failed preflight exits 86 within seconds (the result tgz holds the
# log); the submit file then retries the job (max_retries) away from the machines it already failed on.
# DRY_RUN=1 (local tests) skips the preflight.
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
if [ -z "${DRY_RUN:-}" ]; then
  if ! NGPU_REQ=${NGPU_REQ:-0} python3 - >> "$W/logs/$NAME.out" 2>&1 <<'PY'
import os, sys, torch
need = int(os.environ.get("NGPU_REQ") or 0)
n = torch.cuda.device_count()
print(f"preflight: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')} sees {n} GPU(s), job asked {need}")
if n < max(1, need):
    sys.exit("preflight: too few GPUs")
for i in range(n):
    torch.cuda.set_device(i)
    x = torch.ones(256, 256, device=f"cuda:{i}")
    s = float((x @ x).sum()); torch.cuda.synchronize(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f"preflight: gpu {i} {torch.cuda.get_device_name(i)} ok (matmul {s:.0f}, {free / 2**30:.1f} of {total / 2**30:.1f} GiB free)")
    if abs(s - 256.0 ** 3) > 1 or free < 0.9 * total:
        sys.exit(f"preflight: gpu {i} wrong result or busy")
PY
  then
    echo "== GPU preflight FAILED on $(hostname): exit 86 (retried on another machine) $(date)" >> "$W/logs/$NAME.out"
    pack; exit 86
  fi
fi
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
if [ -n "${SAVE:-}" ] && [ -z "$SMOKE" ] && [ "$rc" = 0 ] && [ -d "$OUTD/model" ]; then   # SAVE=1: weights -> /staging
  mkdir -p "$STAGING/weights" && tar cf "$STAGING/weights/$NAME.tar" -C "$OUTD" model \
    && echo "== weights -> $STAGING/weights/$NAME.tar ($(du -sh "$OUTD/model" | cut -f1))" >> "logs/$NAME.out" \
    || echo "== weights export to $STAGING/weights failed" >> "logs/$NAME.out"
fi
pack
exit $rc
