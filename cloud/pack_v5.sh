#!/usr/bin/env bash
# Build cloud/bundle_v5.tgz: the repo at HEAD (committed code only), data/student_v5 and every evaluation set
# cloud/run_v5.sh uses, laid out like ~/cotd on the HPC. Then (HOST/PORT from the provider's SSH line):
#   scp -P PORT cloud/bundle_v5.tgz root@HOST:/root/
#   ssh -p PORT root@HOST 'mkdir -p cotd && tar xzf bundle_v5.tgz -C cotd && bash cotd/repo/cloud/setup.sh'
#   ssh -p PORT root@HOST 'cd cotd && setsid nohup bash repo/cloud/run_v5.sh > run_v5.out 2>&1 < /dev/null &'
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
git archive --format=tar --prefix=repo/ HEAD | tar x -C "$TMP"
mkdir -p "$TMP/data/eval"
cp -r data/student_v5 "$TMP/data/"
for f in kk_heldout jl_heldout jevbench_public typed_decisions_test bbeh bbh musr policy_heldout sharc_dev folio_val \
         kk_deep gsm8k_mc diag_kk v4_heldout td_noul_qform td_noul_neg claims_ho xkk proverqa zebra logiqa2 lsat_lr \
         lsat_ar; do
  cp "data/eval/$f.jsonl" "$TMP/data/eval/"
done
tar czf cloud/bundle_v5.tgz -C "$TMP" repo data
rm -rf "$TMP"
ls -la cloud/bundle_v5.tgz
