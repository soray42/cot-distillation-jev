#!/bin/bash
#SBATCH --job-name=cotd-peers
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# Open peer decision models on our evaluation sets, each through its own prompt layout and readout
# (scripts/eval_peers.py): decider-2b v11 and Strands Decider v19 -> runs/peer-decider, runs/peer-strands.
# A smoke stage (4 items of two sets per peer) runs first; a peer whose smoke fails is skipped.
#   DRY_RUN=1 bash repo/hpc/eval_peers.sbatch
set -euo pipefail
cd "$HOME/cotd"
module load cuda/12.4 2>/dev/null || true
export HF_HOME="$HOME/cotd/hf" TOKENIZERS_PARALLELISM=false PYTHONPATH="$HOME/cotd/cc1d${PYTHONPATH:+:$PYTHONPATH}"
PY=envs/cotd/bin/python
[ -n "${DRY_RUN:-}" ] && PY="echo [dry-run] python"
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi here"
SETS="bbh=data/eval/bbh.jsonl musr=data/eval/musr.jsonl jevbench=data/eval/jevbench_public.jsonl td=data/eval/typed_decisions_test.jsonl diag=data/eval/diag_kk.jsonl kk=data/eval/kk_heldout.jsonl policy=data/eval/policy_heldout.jsonl folio=data/eval/folio_val.jsonl sharc=data/eval/sharc_dev.jsonl jl=data/eval/jl_heldout.jsonl tdq=data/eval/td_noul_qform.jsonl tdn=data/eval/td_noul_neg.jsonl claims_ho=data/eval/claims_ho.jsonl rule_ho=data/eval/rule_ho.jsonl v4heldout=data/eval/v4_heldout.jsonl"
for f in $SETS; do [ -f "${f#*=}" ] || { echo "missing eval file ${f#*=}"; exit 1; }; done
peer() {    # peer <name> <args...>
  local name=$1; shift
  echo "=== $name smoke $(date +%T)"
  if $PY repo/scripts/eval_peers.py "$@" --eval jevbench=data/eval/jevbench_public.jsonl bbh=data/eval/bbh.jsonl \
       --limit 4 --out "runs/peer-$name-smoke" > "logs/peer-$name-smoke.log" 2>&1; then
    echo "smoke ok $name"; rm -rf "runs/peer-$name-smoke"
    echo "=== $name full $(date +%T)"
    $PY repo/scripts/eval_peers.py "$@" --eval $SETS --out "runs/peer-$name" || echo "peer $name failed"
  else
    echo "SMOKE FAILED $name"; tail -n 30 "logs/peer-$name-smoke.log"
  fi
}
peer decider --peer decider --model models/decider-2b-v11
peer strands --peer strands --model models/strands-decider-2B-v19 --base models/Qwen3.5-2B-Base \
  --strands-src third_party/strands-decider/src
echo "all done $(date +%T)"
