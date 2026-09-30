#!/usr/bin/env bash
# Poll a SLURM job every 5 minutes (HPC_KEY, HPC_HOST in the environment) and exit on the first event worth
# reporting: the smoke stage passed or failed, an error in the log, or the job leaving the queue.
#   hpc/monitor.sh <jobid> <log name under ~/cotd/logs> [stop pattern, e.g. "smoke passed" or "=== A3 "]
# Every poll is appended to notes/hpc_monitor_<jobid>.log. Unreachable (off the school network) polls are
# logged and polling continues.
set -uo pipefail
JID=$1; LOG=$2; STOP=${3:-}
[ "$STOP" = stop-after-smoke ] && STOP="smoke passed"
cd "$(dirname "$0")/.."
OUT=notes/hpc_monitor_$JID.log; mkdir -p notes
for i in $(seq 1 90); do
  R=$(timeout 60 ssh.exe -o BatchMode=yes -o ConnectTimeout=20 -i "$HPC_KEY" "$HPC_HOST" \
      "sacct -j $JID -X -n -o State%20,Elapsed | head -1; echo ---; grep -E 'SMOKE|smoke passed|^=== |failed|Traceback|Error|all done' ~/cotd/logs/$LOG | tail -n 6" 2>&1 | tr -d '\r')
  if [ $? -ne 0 ] || [ -z "$R" ]; then echo "$(date -u +%T) unreachable" >> "$OUT"; sleep 300; continue; fi
  STATE=$(echo "$R" | head -1 | awk '{print $1}')
  echo "$(date -u +%T) $(echo "$R" | tr '\n' ' ' | cut -c1-300)" >> "$OUT"
  if echo "$R" | grep -qE "SMOKE FAILED|Traceback|arm .* failed|base eval failed"; then echo "EVENT error"; echo "$R"; exit 0; fi
  if [ -n "$STOP" ] && echo "$R" | grep -qE "$STOP"; then echo "EVENT reached: $STOP"; echo "$R"; exit 0; fi
  case "$STATE" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_ME*|NODE_FAIL) echo "EVENT job $STATE"; echo "$R"; exit 0 ;; esac
  sleep 300
done
echo "EVENT monitor gave up after 90 polls"
