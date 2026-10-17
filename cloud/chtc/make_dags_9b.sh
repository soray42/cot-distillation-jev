#!/bin/bash
# Writes jobs9b.dag (every job of joblist_9b.txt: "<env file>, <priority>, <GPUs>"; priority order, one retry,
# POST = push to GitHub) and smoke9b.dag (the smoke stages of job 62, GSM8K Tree Forcing, and 65, arithmetic-chain
# Tree Forcing, on 2 GPUs each). Every batch-4 job is FSDP on 2 GPUs: 12 CPUs, 200 GB (cpu-adamw keeps the fp32 master
# + AdamW states of the 9B on the host, 12 B/param x ~9.65e9 = ~116 GB, plus both ranks' bf16 copies while loading;
# batch 3 measured 27 GiB RSS per rank at 4B), GPUs >= 40 GB (4B peaked 11-12 GiB per GPU, 12B ~28 GiB).
# Priorities 53-59: below every batch-3 job (60-75), so batch 4 queues behind what is already submitted.
set -euo pipefail
cd "$(dirname "$0")"
res() { case "$1" in 2) echo 'cpus="12" mem="200GB" gpumem="40000"' ;; *) echo "batch 4 jobs use 2 GPUs (got $1)" >&2; exit 1 ;; esac; }
: > jobs9b.dag
while IFS=, read -r env prio ngpu; do
  env=$(echo "$env" | xargs); prio=$(echo "$prio" | xargs); ngpu=$(echo "$ngpu" | xargs); name=$(basename "$env" .env); n=j${name%%_*}
  r=$(res "$ngpu")
  printf 'JOB %s train_9b.sub\nVARS %s envfile="%s" name="%s" prio="%s" smoke="" ngpu="%s" %s\nSCRIPT POST %s post_push.sh %s $RETURN\nRETRY %s 1\nPRIORITY %s %s\n\n' \
    "$n" "$n" "$env" "$name" "$prio" "$ngpu" "$r" "$n" "$name" "$n" "$n" "$prio" >> jobs9b.dag
done < joblist_9b.txt
cat > smoke9b.dag <<D
JOB smoke62 train_9b.sub
VARS smoke62 envfile="condor/jobs9b/62_9b_gsm_GFPQ-c4lr3_s12_fsdp.env" name="smoke_62_9b_gsm_GFPQ" prio="0" smoke="1" ngpu="2" $(res 2)
SCRIPT POST smoke62 post_push.sh smoke_62_9b_gsm_GFPQ \$RETURN
JOB smoke65 train_9b.sub
VARS smoke65 envfile="condor/jobs9b/65_9b_math25_GFPS-c4lr3_s12_fsdp.env" name="smoke_65_9b_math25_GFPS" prio="0" smoke="1" ngpu="2" $(res 2)
SCRIPT POST smoke65 post_push.sh smoke_65_9b_math25_GFPS \$RETURN
D
echo "jobs9b.dag: $(grep -c '^JOB' jobs9b.dag) jobs; smoke9b.dag: $(grep -c '^JOB' smoke9b.dag) jobs"
