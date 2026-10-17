#!/bin/bash
# Writes jobs4b.dag (every job of joblist_4b.txt: "<env file>, <priority>, <GPUs>"; priority order, one retry,
# POST = push to GitHub) and smoke4b.dag (the smoke stages of job 46, MuSiQue Tree Forcing, and 47, MuSiQue placebo
# (never run on the HPC), on 2 GPUs each, and of job 44, A7 on 1 GPU). Resources per job: 2 GPUs -> 12 CPUs, 120 GB (4B FSDP with cpu-adamw: 27 GiB RSS per rank on the HPC);
# 1 GPU -> 8 CPUs, 80 GB (fp32 master weights and AdamW on the GPU).
set -euo pipefail
cd "$(dirname "$0")"
res() { case "$1" in 2) echo 'cpus="12" mem="120GB" gpumem="40000"' ;; 1) echo 'cpus="8" mem="80GB" gpumem="80000"' ;; *) echo "GPUs must be 1 or 2 (got $1)" >&2; exit 1 ;; esac; }
: > jobs4b.dag; : > jobs4b_hold.dag
while IFS=, read -r env prio ngpu; do
  env=$(echo "$env" | xargs); prio=$(echo "$prio" | xargs); ngpu=$(echo "$ngpu" | xargs); name=$(basename "$env" .env); n=j${name%%_*}
  r=$(res "$ngpu")
  dag=jobs4b.dag; [ "$prio" = 0 ] && { dag=jobs4b_hold.dag; prio=70; }   # priority 0 in joblist_4b.txt = held (own DAG)
  printf 'JOB %s train_4b.sub\nVARS %s envfile="%s" name="%s" prio="%s" smoke="" ngpu="%s" %s\nSCRIPT POST %s post_push.sh %s $RETURN\nRETRY %s 1\nPRIORITY %s %s\n\n' \
    "$n" "$n" "$env" "$name" "$prio" "$ngpu" "$r" "$n" "$name" "$n" "$n" "$prio" >> "$dag"
done < joblist_4b.txt
cat > smoke4b.dag <<D
JOB smoke46 train_4b.sub
VARS smoke46 envfile="condor/jobs4b/46_4b_musique25_GFPQ-c4lr3_s12_fsdp.env" name="smoke_46_4b_musique25_GFPQ" prio="0" smoke="1" ngpu="2" $(res 2)
SCRIPT POST smoke46 post_push.sh smoke_46_4b_musique25_GFPQ \$RETURN
JOB smoke47 train_4b.sub
VARS smoke47 envfile="condor/jobs4b/47_4b_musique25_GFPQX-c4lr3_s12_fsdp.env" name="smoke_47_4b_musique25_GFPQX" prio="0" smoke="1" ngpu="2" $(res 2)
SCRIPT POST smoke47 post_push.sh smoke_47_4b_musique25_GFPQX \$RETURN
JOB smoke44 train_4b.sub
VARS smoke44 envfile="condor/jobs4b/44_4b_gsmrat_A7-c12lr3_s12_a100r.env" name="smoke_44_4b_gsmrat_A7" prio="0" smoke="1" ngpu="1" $(res 1)
SCRIPT POST smoke44 post_push.sh smoke_44_4b_gsmrat_A7 \$RETURN
D
echo "jobs4b.dag: $(grep -c '^JOB' jobs4b.dag) jobs; jobs4b_hold.dag: $(grep -c '^JOB' jobs4b_hold.dag) held; smoke4b.dag: $(grep -c '^JOB' smoke4b.dag) jobs"
