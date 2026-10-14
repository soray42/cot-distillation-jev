#!/usr/bin/env bash
# Build the HTCondor bundle for the Gemma-4-12B jobs (cloud/condor/README.md): the repo at PIN (git archive: committed
# code only), the data the jobs need (cloud/condor/needed_data.json), and cloud/condor as condor/. Laid out as ~/cotd.
#   cloud/pack_condor.sh [PIN]      -> cloud/cotd_condor_bundle.tgz
set -euo pipefail
cd "$(dirname "$0")/.."
PIN=${1:-915dbd6}
TMP=$(mktemp -d)
git archive --format=tar --prefix=repo/ "$PIN" | tar x -C "$TMP"
python3 - "$TMP" <<'PY'
import json, shutil, sys, os
dst = sys.argv[1]
need = json.load(open("cloud/condor/needed_data.json"))
for d in need["dirs"]:
    shutil.copytree(d, os.path.join(dst, d), ignore=shutil.ignore_patterns("*.pt", "*.safetensors"))
for f in need["files"]:
    os.makedirs(os.path.join(dst, os.path.dirname(f)), exist_ok=True)
    shutil.copy2(f, os.path.join(dst, f))
PY
mkdir -p "$TMP/condor"
cp -r cloud/condor/jobs cloud/condor/*.sh cloud/condor/*.sub cloud/condor/joblist.txt cloud/condor/README.md "$TMP/condor/"
( cd "$TMP" && find . -type f | sort | xargs md5sum > condor/MANIFEST.md5 )
tar czf cloud/cotd_condor_bundle.tgz -C "$TMP" .
rm -rf "$TMP"
ls -la cloud/cotd_condor_bundle.tgz
