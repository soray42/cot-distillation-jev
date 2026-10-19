#!/bin/bash
# Writes the batch-7 DAGs (README_7.md), Qwen3.5-4B only, as make_dags_6.sh: jobs7_4b.dag (every job of
# joblist_7_4b.txt: "<env file>, <priority>, <GPUs>"; priority order, one DAG-level retry, POST = push to GitHub) and
# smoke7_4b.dag (the smoke stage of 201 GFPQM, multiple-choice nodes, which needs the friend's MC code in the package,
# and 207 SIYB, stepwise internalisation: SMOKE_ONLY, about 15 min each, mostly unpacking the model). Submit file
# train7_4b.sub. Resources: a 1-GPU line (the friend's fsdp1 patch, as batches 3-6 ran 4B; the joblist ships with 1)
# gets 8 CPUs / 80 GB (49 GiB RSS in 42 / 75 / 83), a 2-GPU line 12 CPUs / 120 GB; GPUs >= 40 GB. Node names j201... /
# smoke201... (batch 6: j154-j163; the friend's own jobs: 164-184), DAGMan log logs/train7_4b.log.
# The joblist is read from this directory, else from ../condor (the repo layout).
#   bash make_dags_7.sh
set -euo pipefail
cd "$(dirname "$0")"
L=joblist_7_4b.txt; [ -f "$L" ] || L=../condor/joblist_7_4b.txt
[ -f "$L" ] || { echo "no joblist_7_4b.txt"; exit 1; }
res() {
  case "$1" in
    2) echo 'cpus="12" mem="120GB" gpumem="40000"' ;;
    1) echo 'cpus="8" mem="80GB" gpumem="40000"' ;;    # 1-GPU FSDP (the friend's fsdp1 patch): 49 GiB RSS in 42 / 75 / 83
    *) echo "make_dags_7: $1 GPU(s) has no resource line" >&2; exit 1 ;;
  esac
}
node() {   # node <dag> <env> <name> <prio> <ngpu> <smoke>
  local n=j${3%%_*}; [ -n "$6" ] && n=smoke${3#smoke_}; n=${n%%_*}
  printf 'JOB %s train7_4b.sub\nVARS %s envfile="%s" name="%s" prio="%s" smoke="%s" ngpu="%s" %s\nSCRIPT POST %s post_push.sh %s $RETURN\n' \
    "$n" "$n" "$2" "$3" "$4" "$6" "$5" "$(res "$5")" "$n" "$3" >> "$1"
  [ -n "$6" ] || printf 'RETRY %s 1\nPRIORITY %s %s\n' "$n" "$n" "$4" >> "$1"
  echo >> "$1"
}
smoke="201 207"
: > jobs7_4b.dag; : > smoke7_4b.dag
while IFS=, read -r env prio ngpu; do
  env=$(echo "$env" | xargs); prio=$(echo "$prio" | xargs); ngpu=$(echo "$ngpu" | xargs); name=$(basename "$env" .env)
  node jobs7_4b.dag "$env" "$name" "$prio" "$ngpu" ""
  case " $smoke " in *" ${name%%_*} "*)
    s=$(echo "$name" | sed -E 's/^([0-9]+_[0-9a-z]+_[0-9a-z]+_[A-Z0-9]+)-.*/\1/')   # 201_4b_gsm_GFPQM
    node smoke7_4b.dag "$env" "smoke_$s" 0 "$ngpu" 1 ;;
  esac
done < "$L"
echo "jobs7_4b.dag: $(grep -c '^JOB' jobs7_4b.dag) jobs; smoke7_4b.dag: $(grep -c '^JOB' smoke7_4b.dag) jobs"
