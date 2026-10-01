#!/usr/bin/env bash
# One-time setup on a rented GPU box (see cloud/pack.sh for the whole workflow). Run from anywhere:
#   bash ~/cotd/repo/cloud/setup.sh
# Installs the HPC package versions (torch 2.14.0, transformers 5.17.0, flash-linear-attention 0.5.2, peft 0.21.1) into
# ~/cotd/env, tries to build causal-conv1d (the HPC runs use the slower PyTorch fallback; the kernel only changes speed),
# downloads Qwen3.5-2B-Base at the HPC revision and checks its sha256 against the HPC copy.
set -euo pipefail
cd "$(dirname "$0")/../.."                      # ~/cotd
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
PYBIN=$(command -v python3.11 || command -v python3)
"$PYBIN" -m venv env
. env/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.14.0
pip install -q transformers==5.17.0 flash-linear-attention==0.5.2 fla-core==0.5.2 peft==0.21.1 accelerate==1.15.0 \
  safetensors==0.8.0 tokenizers==0.23.2 huggingface_hub==1.33.0 numpy==2.4.6
if command -v nvcc >/dev/null; then
  pip install -q --no-build-isolation causal-conv1d || echo "causal-conv1d build failed: the PyTorch fallback is used"
else
  echo "no nvcc on this image: causal-conv1d skipped (PyTorch fallback)"
fi
python - <<'PY'
import hashlib, os
from huggingface_hub import snapshot_download
REV = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"        # the revision on the HPC (sha256 below matches it)
path = snapshot_download("Qwen/Qwen3.5-2B-Base", revision=REV, local_dir="models/Qwen3.5-2B-Base")
h = hashlib.sha256()
with open(os.path.join(path, "model.safetensors-00001-of-00001.safetensors"), "rb") as f:
    for block in iter(lambda: f.read(1 << 24), b""):
        h.update(block)
want = "928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d"
assert h.hexdigest() == want, f"weights differ from the HPC copy: {h.hexdigest()}"
print("model ok", REV)
PY
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0),
      "bf16", torch.cuda.is_bf16_supported(), "transformers", transformers.__version__)
try:
    import causal_conv1d  # noqa: F401
    print("causal-conv1d kernel: yes")
except ImportError:
    print("causal-conv1d kernel: no (fallback)")
PY
mkdir -p runs logs
echo "setup done"
