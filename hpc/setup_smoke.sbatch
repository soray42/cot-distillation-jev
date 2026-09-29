#!/bin/bash
#SBATCH --job-name=cotd-setup
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Submit from ~/cotd:  cd ~/cotd && sbatch setup_smoke.sbatch
# Builds the conda env, downloads the student base models, then measures full-FT memory.
set -euo pipefail
cd "$HOME/cotd"
module load miniconda3 2>/dev/null || module load sw/miniconda3
module load cuda/12.4 2>/dev/null || true

ENV="$HOME/cotd/envs/cotd"
if [ ! -x "$ENV/bin/python" ]; then
  conda create -y -p "$ENV" -c conda-forge --override-channels python=3.11
fi
PY="$ENV/bin/python"
$PY -m pip install -q --upgrade pip
$PY -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124
$PY -m pip install -q "transformers>=5.0" accelerate bitsandbytes huggingface_hub safetensors pillow torchvision --extra-index-url https://download.pytorch.org/whl/cu124
$PY -m pip install -q flash-linear-attention || echo "fla install failed (torch fallback will be used)"
$PY -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"

export HF_HOME="$HOME/cotd/hf"
$PY - <<'EOF'
import os
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen3.5-2B-Base", "Qwen/Qwen3.5-4B-Base"]:
    p = snapshot_download(m, local_dir=os.path.expanduser(f"~/cotd/models/{m.split('/')[1]}"))
    print("downloaded", m, "->", p, flush=True)
EOF

nvidia-smi -L
echo "=== Qwen3.5-2B-Base"
$PY "$HOME/cotd/mem_smoke.py" --model "$HOME/cotd/models/Qwen3.5-2B-Base" --seq 1024 --bs 4 || true
echo "=== Qwen3.5-4B-Base (8-bit Adam only)"
$PY "$HOME/cotd/mem_smoke.py" --model "$HOME/cotd/models/Qwen3.5-4B-Base" --seq 1024 --bs 4 --modes bf16_adamw8bit_ckpt || true
echo "done"
