#!/usr/bin/env bash
# Copy metrics and predictions of all runs (no weights) from ~/cotd/runs into results/runs (HPC_KEY, HPC_HOST).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
timeout 300 ssh.exe -o BatchMode=yes -i "$HPC_KEY" "$HPC_HOST" \
  'cd ~/cotd && tar czf - --exclude="*/model" --exclude="*.safetensors" runs logs/cotd-train-*.out' > results/runs.tgz
tar xzf results/runs.tgz -C results && rm results/runs.tgz
ls results/runs
