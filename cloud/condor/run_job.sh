#!/usr/bin/env bash
# Runs one job (one condor/jobs/*.env file) exactly as the original Slurm runs: hpc/train_arm.sbatch with NGPU=2
# (torchrun, 2 ranks, FSDP). The bundle root must be ~/cotd (train_arm.sbatch works there).
#   bash condor/run_job.sh condor/jobs/02_12b_gsm_GFPQ-c4lr3g12_s12_fsdp.env           # normal run
#   SMOKE_ONLY=1 bash condor/run_job.sh condor/jobs/02_...env                           # smoke stage only (minutes)
# Optional overrides (environment): FSDP_OPTIM=8bit-sr when the job cannot get ~220 GB of host RAM.
set -uo pipefail
ROOT=$HOME/cotd
cd "$ROOT" || { echo "the bundle must be at $ROOT"; exit 1; }
ENVF=$1; [ -f "$ENVF" ] || { echo "no env file $ENVF"; exit 1; }
SMOKE=${SMOKE_ONLY:-}; OPT=${FSDP_OPTIM_OVERRIDE:-${FSDP_OPTIM:-}}; DRYR=${DRY_RUN:-}   # kept over the env file's blanks
set -a; . "$ENVF"; set +a
[ -n "$SMOKE" ] && export SMOKE_ONLY=1
[ -n "$DRYR" ] && export DRY_RUN=1
[ -n "$OPT" ] && export FSDP_OPTIM=$OPT
export SLURM_JOB_ID="condor-$(hostname)-$$"   # any value: tells train_arm.sbatch it already runs inside a job
export SLURM_CPUS_PER_TASK=${REQUEST_CPUS:-12}
mkdir -p logs
NAME=$(basename "$ENVF" .env)
echo "job $NAME on $(hostname) $(date) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} FSDP_OPTIM=$FSDP_OPTIM SMOKE_ONLY=${SMOKE_ONLY:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
bash repo/hpc/train_arm.sbatch > "logs/$NAME${SMOKE:+-smoke}.out" 2>&1
rc=$?
echo "exit $rc ($(date))"; tail -n 5 "logs/$NAME${SMOKE:+-smoke}.out"
exit $rc
