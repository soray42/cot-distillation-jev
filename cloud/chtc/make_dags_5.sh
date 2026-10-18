#!/bin/bash
# Writes the batch-5 DAGs (README_5.md): for each model m given (default: every joblist_5_<m>.txt found), jobs5_<m>.dag
# (every job of joblist_5_<m>.txt: "<env file>, <priority>, <GPUs>"; priority order, one DAG-level retry, POST = push to
# GitHub, as batches 3-4) and smoke5_<m>.dag (the smoke stage of a few jobs: SMOKE_ONLY, about 15 min each, mostly
# unpacking the model). Submit file train5_<m>.sub. Resources per 2-GPU job: 4B 12 CPUs / 120 GB (batch 3: 27 GiB RSS
# per rank), 9B 12 CPUs / 200 GB (batch 4: cpu-adamw holds ~116 GB of fp32 master + AdamW), 12B 12 CPUs / 230 GB
# (batches 1-2); GPUs >= 40 GB everywhere (12B peaked at ~28 GiB per GPU, 9B / 4B less). A 4B line with 1 GPU (run with
# the friend's 1-GPU FSDP patch, as batch 3 ran) gets 8 CPUs / 80 GB.
# The joblists are read from this directory, else from ../condor (the repo layout).
#   bash make_dags_5.sh [4b] [9b] [12b]
set -euo pipefail
cd "$(dirname "$0")"
list() { local f=joblist_5_$1.txt; [ -f "$f" ] || f=../condor/joblist_5_$1.txt; [ -f "$f" ] && echo "$f"; }
res() {
  case "$1:$2" in
    4b:2) echo 'cpus="12" mem="120GB" gpumem="40000"' ;;
    4b:1) echo 'cpus="8" mem="80GB" gpumem="40000"' ;;    # 1-GPU FSDP (the friend's fsdp1 patch): 49 GiB RSS in 42 / 75 / 83
    9b:2) echo 'cpus="12" mem="200GB" gpumem="40000"' ;;
    12b:2) echo 'cpus="12" mem="230GB" gpumem="40000"' ;;
    *) echo "make_dags_5: model $1 with $2 GPU(s) has no resource line" >&2; exit 1 ;;
  esac
}
node() {   # node <dag> <model> <env> <name> <prio> <ngpu> <smoke>
  local n=j${4%%_*}; [ -n "$7" ] && n=smoke${4#smoke_}; n=${n%%_*}
  printf 'JOB %s train5_%s.sub\nVARS %s envfile="%s" name="%s" prio="%s" smoke="%s" ngpu="%s" %s\nSCRIPT POST %s post_push.sh %s $RETURN\n' \
    "$n" "$2" "$n" "$3" "$4" "$5" "$7" "$6" "$(res "$2" "$6")" "$n" "$4" >> "$1"
  [ -n "$7" ] || printf 'RETRY %s 1\nPRIORITY %s %s\n' "$n" "$n" "$5" >> "$1"
  echo >> "$1"
}
models=("$@"); [ ${#models[@]} -gt 0 ] || for m in 4b 9b 12b; do [ -n "$(list $m)" ] && models+=("$m"); done
for m in "${models[@]}"; do
  case $m in 4b) smoke="101 107 144" ;; 9b) smoke="153" ;; 12b) smoke="105" ;; *) echo "unknown model $m"; exit 1 ;; esac
  L=$(list "$m") || { echo "no joblist_5_$m.txt"; exit 1; }
  : > "jobs5_$m.dag"; : > "smoke5_$m.dag"
  while IFS=, read -r env prio ngpu; do
    env=$(echo "$env" | xargs); prio=$(echo "$prio" | xargs); ngpu=$(echo "$ngpu" | xargs); name=$(basename "$env" .env)
    node "jobs5_$m.dag" "$m" "$env" "$name" "$prio" "$ngpu" ""
    case " $smoke " in *" ${name%%_*} "*)
      s=$(echo "$name" | sed -E 's/^([0-9]+_[0-9a-z]+_[0-9a-z]+_[A-Z0-9]+)-.*/\1/')   # 101_4b_gsm_GFPQ
      node "smoke5_$m.dag" "$m" "$env" "smoke_$s" 0 "$ngpu" 1 ;;
    esac
  done < "$L"
  echo "jobs5_$m.dag: $(grep -c '^JOB' "jobs5_$m.dag") jobs; smoke5_$m.dag: $(grep -c '^JOB' "smoke5_$m.dag") jobs"
done
