#!/usr/bin/env bash
# From WSL, on school wifi: copy student/eval data to ~/cotd/data on the cluster and update the repo clone.
#   hpc/push_data.sh                      # all data dirs
#   hpc/push_data.sh data/student_v3      # only the given dirs
set -euo pipefail
cd "$(dirname "$0")/.."
# Set HPC_KEY (Windows path of the ssh key) and HPC_HOST (user@login-node) in the environment.
KEY="${HPC_KEY:?set HPC_KEY}"; HOST="${HPC_HOST:?set HPC_HOST}"
WIN_TMP="${WIN_TMP:-/mnt/c/Users/$USER}"   # a Windows-visible folder for the scp.exe upload
DIRS=("$@"); [ ${#DIRS[@]} -gt 0 ] || DIRS=(data/student data/eval $(ls -d data/student_tree* data/student_v[0-9]* 2>/dev/null))
tar czf "$WIN_TMP/cotd_data.tgz" "${DIRS[@]}"
(cd "$WIN_TMP" && scp.exe -i "$KEY" cotd_data.tgz "$HOST:cotd/")
ssh.exe -i "$KEY" "$HOST" 'cd ~/cotd && tar xzf cotd_data.tgz && rm cotd_data.tgz && mkdir -p logs runs &&
  if [ -d repo/.git ]; then git -C repo pull -q; else git clone -q https://github.com/soray42/cot-distillation-jev.git repo; fi &&
  ls data/*'
