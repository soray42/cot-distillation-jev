#!/bin/bash
#SBATCH --job-name=cotd-exit
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Early-exit evaluation (scripts/early_exit_eval.py) of EXIT_MODEL (a saved model dir): decision quality after L of the
# layers (logit lens and label-free tuned lens) on our evaluation sets, and the latency of a forward stopped after layer L.
# A smoke stage (8 items) runs first.   DRY_RUN=1 EXIT_MODEL=... bash repo/hpc/early_exit.sbatch
set -euo pipefail
cd "$HOME/cotd"
module load cuda/12.4 2>/dev/null || true
export HF_HOME="$HOME/cotd/hf" TOKENIZERS_PARALLELISM=false PYTHONPATH="$HOME/cotd/cc1d${PYTHONPATH:+:$PYTHONPATH}"
PY=envs/cotd/bin/python
[ -n "${DRY_RUN:-}" ] && PY="echo [dry-run] python"
M=${EXIT_MODEL:?set EXIT_MODEL}
OUT=${EXIT_OUT:-runs/early-exit-$(basename $(dirname $M))}
FIT="val=data/student_v4t/val.jsonl v4heldout=data/eval/v4_heldout.jsonl"
SETS="jevbench=data/eval/jevbench_public.jsonl bbh=data/eval/bbh.jsonl musr=data/eval/musr.jsonl td=data/eval/typed_decisions_test.jsonl diag=data/eval/diag_kk.jsonl kk=data/eval/kk_heldout.jsonl policy=data/eval/policy_heldout.jsonl folio=data/eval/folio_val.jsonl claims_ho=data/eval/claims_ho.jsonl"
for f in $FIT $SETS; do [ -f "${f#*=}" ] || { echo "missing ${f#*=}"; exit 1; }; done
[ -n "${DRY_RUN:-}" ] || [ -f "$M/config.json" ] || { echo "no model at $M"; exit 1; }
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi here"
echo "=== smoke $(date +%T)"
$PY repo/scripts/early_exit_eval.py --model "$M" --out "$OUT-smoke" --fit $FIT --eval jevbench=data/eval/jevbench_public.jsonl \
  --limit 8 --timing-items 2 > "logs/early-exit-smoke.log" 2>&1 || { echo "SMOKE FAILED"; tail -n 30 logs/early-exit-smoke.log; exit 1; }
rm -rf "$OUT-smoke"; echo "smoke ok"
echo "=== full $(date +%T)"
$PY repo/scripts/early_exit_eval.py --model "$M" --out "$OUT" --fit $FIT --eval $SETS
echo "all done $(date +%T)"
