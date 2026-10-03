#!/usr/bin/env bash
# Back up both v5 arms to the private dev repo (sora42y/Sev-2B-dev) as soon as each run has saved its weights
# (its "done" line in run_v5.out): weights, metrics.json, predictions and the training log. The token is read from
# ~/.hf_tok, a repo-scoped token that the caller deletes afterwards.
#   cd ~/cotd && setsid nohup bash repo/cloud/upload_dev.sh > upload_dev.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."                      # ~/cotd
export HF_TOKEN="$(cat ~/.hf_tok)"
for arm in M1 M2; do
  out=runs/TF-$arm-Qwen3.5-2B-Base-student_v5-s0
  until grep -q "done $out" run_v5.out; do sleep 30; done
  cp "logs/$arm.log" "$out/train.log"
  env/bin/python - "$out" "$arm" <<'PY' || echo "upload failed $arm"
import sys
from huggingface_hub import HfApi
out, arm = sys.argv[1], sys.argv[2]
HfApi().upload_folder(repo_id="sora42y/Sev-2B-dev", folder_path=out, path_in_repo=f"TF-{arm}-v5-s0",
                      commit_message=f"TF-{arm} on student_v5, seed 0: weights, metrics, predictions, log",
                      ignore_patterns=["hidden_*.pt"])
print(f"uploaded {out}", flush=True)
PY
done
echo "upload_dev done $(date +%T)"
