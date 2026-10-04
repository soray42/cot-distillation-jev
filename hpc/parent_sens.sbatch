#!/bin/bash
#SBATCH --job-name=cotd-ps
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Parent-sensitivity evaluation (scripts/parent_sens_eval.py; evaluation only) of every model in PS_MODELS (default:
# the base plus every saved runs/*/model): eligible tree nodes of held-out items asked with the parents' results, with
# as many matched non-parent results, with each set's answers negated, and plainly. Output: runs/ps-<model>/.
# A smoke stage (3 items, first model) runs first; models whose ps_metrics.json exists are skipped (resubmit-safe).
#   DRY_RUN=1 bash repo/hpc/parent_sens.sbatch        PS_MODELS="runs/X/model runs/Y/model" sbatch repo/hpc/parent_sens.sbatch
set -euo pipefail
cd "$HOME/cotd"
module load cuda/12.4 2>/dev/null || true
export HF_HOME="$HOME/cotd/hf" TOKENIZERS_PARALLELISM=false PYTHONPATH="$HOME/cotd/cc1d${PYTHONPATH:+:$PYTHONPATH}"
PY=envs/cotd/bin/python
[ -n "${DRY_RUN:-}" ] && PY="echo [dry-run] python"
BASE=$HOME/cotd/models/Qwen3.5-2B-Base
MODELS=${PS_MODELS:-"$BASE $(ls -d runs/*/model 2>/dev/null | tr '\n' ' ')"}
DATA=
for kv in v3f=data/student_v3f/val.jsonl v4t=data/student_v4t/val.jsonl rule=data/student_rule/val.jsonl; do
  if [ -f "${kv#*=}" ]; then DATA="$DATA $kv"; else echo "skip missing set ${kv#*=}"; fi
done
[ -n "$DATA" ] || { echo "no data"; exit 1; }
name() { case "$1" in "$BASE") echo base ;; */model) basename "$(dirname "$1")" ;; *) basename "$1" ;; esac; }
OK=
for m in $MODELS; do
  if [ -f "$m/config.json" ]; then OK="$OK $m"; else echo "skip $m (no config.json)"; fi
done
[ -n "$OK" ] || { echo "no models"; exit 1; }
echo "models:$OK"; echo "sets:$DATA"
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi here"
echo "=== smoke $(date +%T)"
FIRST=$(echo $OK | cut -d' ' -f1)
$PY repo/scripts/parent_sens_eval.py --model "$FIRST" --data $DATA --limit 3 --out runs/ps-smoke \
  > logs/ps-smoke.log 2>&1 || { echo "SMOKE FAILED"; tail -n 30 logs/ps-smoke.log; exit 1; }
[ -n "${DRY_RUN:-}" ] || grep -q '^ps_v3f\|^ps_v4t\|^ps_rule' logs/ps-smoke.log || { echo "SMOKE FAILED (no result line)"; tail -n 30 logs/ps-smoke.log; exit 1; }
rm -rf runs/ps-smoke; echo "smoke ok"
for m in $OK; do
  out=runs/ps-$(name "$m")
  [ -f "$out/ps_metrics.json" ] && { echo "done already: $out"; continue; }
  echo "=== $m -> $out $(date +%T)"
  $PY repo/scripts/parent_sens_eval.py --model "$m" --data $DATA --bs 16 --out "$out" || echo "FAILED $m (next model)"
done
echo "all done $(date +%T)"
