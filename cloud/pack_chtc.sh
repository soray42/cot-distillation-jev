#!/usr/bin/env bash
# Build the CHTC bundle (cloud/chtc/README.md): cotd_code_data.tgz (repo at PIN, the data of cloud/condor/needed_data.json,
# the per-job env files as condor/jobs/) plus the CHTC files, laid out as ~/cotd_chtc.
#   cloud/pack_chtc.sh [PIN]   -> cloud/cotd_chtc_bundle.tgz
set -euo pipefail
cd "$(dirname "$0")/.."
PIN=${1:-915dbd6}
T=$(mktemp -d); C=$T/code; B=$T/bundle; mkdir -p "$C" "$B/logs" "$B/results"
git archive --format=tar --prefix=repo/ "$PIN" | tar x -C "$C"
python3 - "$C" <<'PY'
import json, shutil, sys, os
dst = sys.argv[1]
need = json.load(open("cloud/condor/needed_data.json"))
for d in need["dirs"]:
    shutil.copytree(d, os.path.join(dst, d))
for f in need["files"]:
    os.makedirs(os.path.join(dst, os.path.dirname(f)), exist_ok=True)
    shutil.copy2(f, os.path.join(dst, f))
PY
mkdir -p "$C/condor" && cp -r cloud/condor/jobs "$C/condor/"
tar czf "$B/cotd_code_data.tgz" -C "$C" .
cp cloud/condor/joblist.txt cloud/chtc/{README.md,chtc_settings.inc,cotd.def,build.sub,fetch_model.sh,train.sub,run_train.sh,post_push.sh,make_dags.sh} "$B/"
( cd "$B" && bash make_dags.sh >/dev/null && touch logs/.keep results/.keep && find . -type f | sort | xargs md5sum > MANIFEST.md5 )
tar czf cloud/cotd_chtc_bundle.tgz -C "$B" .
rm -rf "$T"
ls -la cloud/cotd_chtc_bundle.tgz
