#!/bin/bash
#SBATCH --job-name=cotd-wise
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# Weight interpolation between the base and a fine-tuned student (scripts/wise_ft_eval.py), evaluation only, no
# weights saved: FT=<model dir> ALPHAS=0.5,0.7,0.85 -> runs/wise-<name of FT's run>/a<alpha>/. Smoke first (4 items).
#   DRY_RUN=1 FT=runs/TF-v4t-Qwen3.5-2B-Base-student_v4t-s0/model bash repo/hpc/wise_ft.sbatch
set -euo pipefail
cd "$HOME/cotd"
module load cuda/12.4 2>/dev/null || true
export HF_HOME="$HOME/cotd/hf" TOKENIZERS_PARALLELISM=false PYTHONPATH="$HOME/cotd/cc1d${PYTHONPATH:+:$PYTHONPATH}"
PY=envs/cotd/bin/python
[ -n "${DRY_RUN:-}" ] && PY="echo [dry-run] python"
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi here"
BASE=models/Qwen3.5-2B-Base
[ -f "$FT/config.json" ] || { echo "missing model $FT"; exit 1; }
OUT=runs/wise-$(basename "$(dirname "$FT")")
SETS="jevbench=data/eval/jevbench_public.jsonl bbh=data/eval/bbh.jsonl musr=data/eval/musr.jsonl gsm8k=data/eval/gsm8k_mc.jsonl td=data/eval/typed_decisions_test.jsonl diag=data/eval/diag_kk.jsonl kk=data/eval/kk_heldout.jsonl policy=data/eval/policy_heldout.jsonl proverqa=data/eval/proverqa.jsonl xkk=data/eval/xkk.jsonl logiqa2=data/eval/logiqa2.jsonl lsat_lr=data/eval/lsat_lr.jsonl zebra=data/eval/zebra.jsonl"
for f in $SETS; do [ -f "${f#*=}" ] || { echo "missing eval file ${f#*=}"; exit 1; }; done
mkdir -p runs/wise-smoke-data
head -n 4 data/eval/gsm8k_mc.jsonl > runs/wise-smoke-data/gsm8k.jsonl
echo "=== smoke $(date +%T)"
$PY repo/scripts/wise_ft_eval.py --base $BASE --ft "$FT" --alphas 0.5 --eval gsm8k=runs/wise-smoke-data/gsm8k.jsonl \
  --out runs/wise-smoke > logs/wise-smoke.log 2>&1 || { echo "SMOKE FAILED"; tail -n 30 logs/wise-smoke.log; exit 1; }
rm -rf runs/wise-smoke runs/wise-smoke-data
echo "=== smoke passed; full $(date +%T)"
$PY repo/scripts/wise_ft_eval.py --base $BASE --ft "$FT" --alphas "${ALPHAS:-0.5,0.7,0.85}" --eval $SETS --out "$OUT"
echo "all done $(date +%T)"
