#!/usr/bin/env bash
# One-time setup on the HTCondor submit host (needs internet). Run from the bundle root (~/cotd):
#   bash condor/setup.sh
# 1. a Python 3.11 venv at ~/cotd/envs/cotd with the exact package versions of the original runs;
# 2. google/gemma-4-12B (base) at the pinned revision into ~/cotd/models/gemma-4-12B, checked by sha256.
# google/gemma-4-12B is Apache-2.0 and not gated (checked 2026-10-14): no Hugging Face login or token is needed.
# Disk: venv ~10 GB, model 24 GB, data < 1 GB.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f repo/scripts/train_student.py ] || { echo "run this from the bundle root (it must contain repo/)"; exit 1; }
PYBIN=$(command -v python3.11 || true)
[ -n "$PYBIN" ] || { echo "python3.11 not found (the original runs used 3.11.16); install it or module load it, then rerun"; exit 1; }
[ -d envs/cotd ] || "$PYBIN" -m venv envs/cotd
. envs/cotd/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.14.0 torchvision==0.29.0
pip install -q transformers==5.17.0 accelerate==1.15.0 peft==0.21.1 safetensors==0.8.0 tokenizers==0.23.2 \
  huggingface_hub==1.33.0 numpy==2.4.6 torchao==0.18.0 bitsandbytes==0.50.2 einops==0.8.2 \
  flash-linear-attention==0.5.2 fla-core==0.5.2 Jinja2==3.1.6 PyYAML==6.0.3 regex==2026.9.29
python - <<'PY'
import hashlib, os
from huggingface_hub import snapshot_download
REV = "023679ed352de9bb66cc873c9009ce3482585c08"
SHA = "fe054ae05ff7f44318fd8ae90d58992531455c7ed31356704088f0f2d8c8009a"
p = snapshot_download("google/gemma-4-12B", revision=REV, local_dir="models/gemma-4-12B")
h = hashlib.sha256()
with open(os.path.join(p, "model.safetensors"), "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
        h.update(b)
assert h.hexdigest() == SHA, f"model.safetensors sha256 {h.hexdigest()} != {SHA}"
print("gemma-4-12B ok at", REV)
PY
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__)"
echo "setup done"
