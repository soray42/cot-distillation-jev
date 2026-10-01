#!/bin/bash
#SBATCH --job-name=cotd-debug
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Eval-only debug job: score each model in MODELS (space-separated "name=path") on each set in SETS ("name=path"),
# writing runs/debug-<name>/. No training. Used for the TD duplicate-question perturbation test.
set -euo pipefail
cd "$HOME/cotd"
module load cuda/12.4 2>/dev/null || true
export HF_HOME="$HOME/cotd/hf" TOKENIZERS_PARALLELISM=false PYTHONPATH="$HOME/cotd/cc1d${PYTHONPATH:+:$PYTHONPATH}"
PY=envs/cotd/bin/python
[ -n "${DRY_RUN:-}" ] && PY="echo [dry-run] python"
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi here"
for m in $MODELS; do
  echo "=== ${m%%=*} $(date +%T)"
  $PY repo/scripts/eval_student.py --model "${m#*=}" --eval $SETS --out "runs/debug-${m%%=*}" --bs 8 || echo "eval failed ${m%%=*}"
done
echo "all done $(date +%T)"
