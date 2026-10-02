#!/bin/bash
#SBATCH --job-name=cotd-debug
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Eval-only debug job: score each model in MODELS (space-separated "name=path") on each set in SETS ("name=path"),
# writing runs/debug-<name>/. No training. Used for the TD duplicate-question perturbation test. With X0_MODEL set, also
# runs the interchange-tool sanity checks (scripts/interchange_audit.py --check) on that model -> runs/x0-check/.
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
if [ -n "${X0_MODEL:-}" ]; then
  echo "=== X0 check $(date +%T)"
  $PY repo/scripts/interchange_audit.py --model "$X0_MODEL" --out runs/x0-check --layers 4,8,12,16,20 --max-pairs 200 \
    --check || echo "x0 check failed"
fi
echo "all done $(date +%T)"
