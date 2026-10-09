#!/bin/bash
#SBATCH --job-name=cotd-intern
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Intern-Decision-4B audit (2026-10-08): scripts/eval_intern_format.py in its own request format, for Intern-Decision-4B
# and the post-trained Qwen3.5-4B it was fine-tuned from (zero-shot reference). Public JevBench easy/original/hard and
# Typed Decision (its released copies) with options as given, reversed and shuffled; our MC sets (JevBench MC copy,
# K&K, diag, kk_deep, BBH, GSM8K-MC, numeric chains, policy, FOLIO, ProverQA) as one choice field.
# -> runs/intern/<model>-<perm>/{metrics.json,preds_*.jsonl}. REPO = checkout with the script (default repo-w10).
#   sbatch [--dependency=afterany:<jid>] repo-w10/hpc/intern_eval.sbatch     (DRY_RUN=1 bash ... echoes the commands)
set -uo pipefail
cd "$HOME/cotd"
export HF_HOME="$HOME/cotd/hf" TOKENIZERS_PARALLELISM=false PYTHONPATH="$HOME/cotd/cc1d${PYTHONPATH:+:$PYTHONPATH}"
PY=envs/cotd/bin/python; REPO=${REPO:-$HOME/cotd/repo-w10}
echo "code: $REPO @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"; nvidia-smi -L 2>/dev/null
IB=data/eval/intern_bench
NATIVE="--set jev_easy=$IB/jevbench/easy.jsonl:jevraw --set jev_orig=$IB/jevbench/original.jsonl:jevraw --set jev_hard=$IB/jevbench/hard.jsonl:jevraw --set td=$IB/typed_decisions/test.jsonl:canon"
MC=""
for s in jevmc=jevbench_public kk=kk_heldout diag=diag_kk kkdeep=kk_deep bbh=bbh gsm8k=gsm8k_mc ng4=numgen_d4 ng6=numgen_d6 ng8=numgen_d8 policy=policy_heldout folio=folio_val proverqa=proverqa; do
  f=data/eval/${s#*=}.jsonl; [ -f "$f" ] && MC="$MC --set ${s%%=*}=$f:mc" || echo "skip missing $f"
done
for f in $IB/jevbench/easy.jsonl $IB/jevbench/original.jsonl $IB/jevbench/hard.jsonl $IB/typed_decisions/test.jsonl; do
  [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
run() { local o="runs/intern/$1-${OUTTAG:-$2}"; echo "=== $1 $2 -> $o $(date +%T)"
  [ -n "${DRY_RUN:-}" ] && { echo "$PY $REPO/scripts/eval_intern_format.py --model models/$1 --out $o --perm $2 ${@:3}" | cut -c1-300; return; }
  $PY "$REPO/scripts/eval_intern_format.py" --model "models/$1" --out "$o" --perm "$2" "${@:3}" || echo "FAILED $1 $2"; }
for M in ${MODELS:-Intern-Decision-4B Qwen3.5-4B}; do
  [ -f "models/$M/config.json" ] || { echo "missing models/$M"; continue; }
  [ -n "${MCONLY:-}" ] && { OUTTAG=mc run "$M" none $MC; continue; }   # -> runs/intern/<model>-mc
  run "$M" none $NATIVE $MC
  run "$M" reverse $NATIVE
  run "$M" shuffle $NATIVE
done
echo "all done $(date +%T)"
