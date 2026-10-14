#!/bin/bash
# Writes jobs.dag (every job of joblist.txt, priority order, one retry, POST = push to GitHub) and smoke.dag.
set -euo pipefail
cd "$(dirname "$0")"
: > jobs.dag
while IFS=, read -r env prio; do
  env=$(echo "$env" | xargs); prio=$(echo "$prio" | xargs); name=$(basename "$env" .env); n=j${name%%_*}
  printf 'JOB %s train.sub\nVARS %s envfile="%s" name="%s" prio="%s" smoke=""\nSCRIPT POST %s post_push.sh %s $RETURN\nRETRY %s 1\nPRIORITY %s %s\n\n' \
    "$n" "$n" "$env" "$name" "$prio" "$n" "$name" "$n" "$n" "$prio" >> jobs.dag
done < joblist.txt
cat > smoke.dag <<'D'
JOB smoke train.sub
VARS smoke envfile="condor/jobs/02_12b_gsm_GFPQ-c4lr3g12_s12_fsdp.env" name="smoke_02_12b_gsm_GFPQ" prio="0" smoke="1"
SCRIPT POST smoke post_push.sh smoke_02_12b_gsm_GFPQ $RETURN
D
echo "jobs.dag: $(grep -c '^JOB' jobs.dag) jobs"
