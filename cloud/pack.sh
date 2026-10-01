#!/usr/bin/env bash
# Build cloud/bundle.tgz for a rented GPU box: the repo at HEAD (git archive, so only committed code) plus the v3 student
# data and every eval set the E2 runs use, laid out like ~/cotd on the HPC (repo/, data/...), so all paths match.
#
# Whole workflow (from the repo root on WSL; HOST/PORT from the vast.ai "SSH" button, key ~/.ssh/id_vast_gemma):
#   cloud/pack.sh                                                   # -> cloud/bundle.tgz
#   scp -P PORT -i ~/.ssh/id_vast_gemma cloud/bundle.tgz root@HOST:/root/
#   ssh -p PORT -i ~/.ssh/id_vast_gemma root@HOST 'mkdir -p cotd && tar xzf bundle.tgz -C cotd && bash cotd/repo/cloud/setup.sh'
#   ssh ... 'cd cotd && nohup bash repo/cloud/run_e2.sh "0 1" > run_e2.out 2>&1 &'
#   cloud/fetch.sh HOST PORT                                         # -> results/cloud/runs
# No API keys go to the box: training and evaluation make no API calls.
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
git archive --format=tar --prefix=repo/ HEAD | tar x -C "$TMP"
mkdir -p "$TMP/data/eval"
cp -r data/student_v3 "$TMP/data/"
for f in kk_heldout jl_heldout jevbench_public typed_decisions_test bbeh bbh musr policy_heldout sharc_dev folio_val \
         kk_deep gsm8k_mc val_facts diag_kk; do
  cp "data/eval/$f.jsonl" "$TMP/data/eval/"
done
tar czf cloud/bundle.tgz -C "$TMP" repo data
rm -rf "$TMP"
ls -la cloud/bundle.tgz
