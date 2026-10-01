#!/usr/bin/env bash
# Pull run results (metrics, predictions, hidden states; no weights) and logs from the rented box into results/cloud/.
#   cloud/fetch.sh HOST PORT
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=$1; PORT=$2; KEY=${KEY:-$HOME/.ssh/id_vast_gemma}
mkdir -p results/cloud
ssh -p "$PORT" -i "$KEY" -o StrictHostKeyChecking=accept-new "root@$HOST" \
  'cd ~/cotd && tar czf - --exclude="*/model" --exclude="*/adapter" --exclude="*.safetensors" runs logs run_e2.out 2>/dev/null' \
  | tar xzf - -C results/cloud
ls results/cloud/runs
