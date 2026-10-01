#!/usr/bin/env bash
# Follow a relay chain: every 5 minutes, look at the relay's running job (else its newest submission; relays
# queue the next job ahead with a dependency) (HPC_KEY, HPC_HOST in the env)
# and exit on the first event not reported before: smoke passed or failed, an error in the job log, the first
# training-step line of an arm (with the projected training time against the walltime), the job leaving the
# queue, or the relay finishing. Reported events are kept in notes/chain_<relay log>.seen, so a relaunch only
# reports new ones.
#   hpc/monitor_chain.sh relay4.log
set -uo pipefail
RLOG=$1
cd "$(dirname "$0")/.."
mkdir -p notes
SEEN=notes/chain_${RLOG%.log}.seen; OUT=notes/hpc_monitor_chain_${RLOG%.log}.log
touch "$SEEN"
event() {   # key, message: report once
  grep -qxF "$1" "$SEEN" && return 1
  echo "$1" >> "$SEEN"; echo "EVENT $2"; return 0
}
for i in $(seq 1 22); do
  R=$(timeout 60 ssh.exe -o BatchMode=yes -o ConnectTimeout=20 -i "$HPC_KEY" "$HPC_HOST" "cd ~/cotd;
      J=; for j in \$(grep -oE 'submitted [0-9]+' logs/$RLOG | cut -d' ' -f2); do
        [ \"\$(squeue -h -j \$j -o %T 2>/dev/null)\" = RUNNING ] && J=\$j; done;
      [ -n \"\$J\" ] || J=\$(grep -oE 'submitted [0-9]+' logs/$RLOG | tail -1 | cut -d' ' -f2); echo \"JID \${J:-none}\";
      grep -qE 'relay[0-9]* done' logs/$RLOG && echo RELAYDONE;
      for j in \$(grep -oE 'submitted [0-9]+' logs/$RLOG | cut -d' ' -f2); do
        echo \"ALL \$j \$(sacct -j \$j -X -n -o State%20 | head -1)\"; done;
      if [ -n \"\$J\" ]; then sacct -j \$J -X -n -o State%20,Elapsed | head -1 | sed 's/^/STATE /';
        L=logs/cotd-train-\$J.out;
        grep -E 'SMOKE|smoke passed|JEFF SMOKE SKIPPED|^=== |failed|Traceback|Error|all done|examples/epoch' \$L 2>/dev/null | tail -n 12;
        grep -E '^\{\"step\"' \$L 2>/dev/null | tail -1 | sed 's/^/LASTSTEP /'; fi" 2>&1 | tr -d '\r')
  if [ $? -ne 0 ] || [ -z "$R" ]; then echo "$(date -u +%T) unreachable" >> "$OUT"; sleep 300; continue; fi
  echo "$(date -u +%T) $(echo "$R" | tr '\n' ' ' | cut -c1-400)" >> "$OUT"
  J=$(echo "$R" | awk '/^JID /{print $2}'); ST=$(echo "$R" | awk '/^STATE /{print $2}')
  if echo "$R" | grep -qE "SMOKE FAILED|Traceback|arm .* failed|base eval failed|jeff eval failed|JEFF SMOKE SKIPPED"; then
    event "$J:error:$(echo "$R" | grep -cE 'SMOKE FAILED|Traceback|failed|SKIPPED')" "error in job $J" && { echo "$R"; exit 0; }
  fi
  if echo "$R" | grep -q "smoke passed"; then event "$J:smoke" "smoke passed in job $J" && { echo "$R"; exit 0; }; fi
  ARM=$(echo "$R" | grep -E '^=== A[0-9]' | tail -1 | awk '{print $2}')
  STEPS=$(echo "$R" | grep -oE 'steps=[0-9]+' | tail -1 | cut -d= -f2)
  SPS=$(echo "$R" | awk '/^LASTSTEP /' | grep -oE '"s_per_step": [0-9.]+' | awk '{print $2}')
  if [ -n "$ARM" ] && [ -n "$STEPS" ] && [ -n "$SPS" ]; then
    H=$(awk -v s="$STEPS" -v t="$SPS" 'BEGIN{printf "%.2f", s*t/3600}')
    event "$J:steps:$ARM" "job $J arm $ARM training: $STEPS steps x ${SPS}s = ${H}h projected (walltime 6h incl. smoke/eval)" && { echo "$R"; exit 0; }
  fi
  while read -r _ jid jst; do                 # every relay job that has ended (also ones queued ahead)
    case "$jst" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_ME*|NODE_FAIL)
      event "$jid:end" "job $jid $jst" && { echo "$R"; exit 0; } ;; esac
  done < <(echo "$R" | grep '^ALL ')
  case "$ST" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_ME*|NODE_FAIL)
    event "$J:end" "job $J $ST" && { echo "$R"; exit 0; } ;; esac
  if echo "$R" | grep -q RELAYDONE && [ -n "$ST" ] && ! echo "$ST" | grep -qE "PENDING|RUNNING"; then
    event "relay:done" "relay finished" && { echo "$R"; exit 0; }
  fi
  sleep 300
done
echo "EVENT heartbeat: no new event in 22 polls; relaunch to keep watching"
