#!/bin/bash
# Inside the interactive build job, after cotd.sif is built:  bash fetch_model.sh /staging/<netid>
# Downloads google/gemma-4-12B (Apache-2.0, not gated) at the pinned revision with the image's python, checks sha256,
# and writes /staging/<netid>/gemma-4-12B.tar (uncompressed tar, ~24 GB). /staging is the place for large files.
set -euo pipefail
STAGING=${1:?usage: bash fetch_model.sh /staging/<netid>}
HF_HUB_OFFLINE=0 apptainer exec cotd.sif python - <<'PY'
import hashlib, os
from huggingface_hub import snapshot_download
REV = "023679ed352de9bb66cc873c9009ce3482585c08"
SHA = "fe054ae05ff7f44318fd8ae90d58992531455c7ed31356704088f0f2d8c8009a"
p = snapshot_download("google/gemma-4-12B", revision=REV, local_dir="gemma-4-12B")
h = hashlib.sha256()
with open(os.path.join(p, "model.safetensors"), "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
        h.update(b)
assert h.hexdigest() == SHA, h.hexdigest()
print("gemma-4-12B ok at", REV)
PY
rm -rf gemma-4-12B/.cache
tar cf gemma-4-12B.tar gemma-4-12B
mv gemma-4-12B.tar "$STAGING/"
ls -la "$STAGING/gemma-4-12B.tar"
