#!/bin/bash
#SBATCH --job-name=cotd-reason
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x-%j.out
# External reasoning benchmarks (notes/prereg_release.md, addendum 2026-10-02 19:35 UTC) for every model on HPC, each
# through its own readout -> runs/reason-<name>/preds_<set>.jsonl:
#   OURS="name=model_dir ..."   our checkpoints and the untrained base (scripts/eval_student.py)
#   PEERS=1                      decider-2b v11 and Strands v19 (scripts/eval_peers.py)
#   JEFF=1                       Jeff-Qwen3.5-2B with its own prompt and readout (scripts/eval_jeff.py)
#   LLM="name=model_dir ..."    post-trained chat models with thinking off (scripts/eval_llm_baseline.py)
# Missing evaluation files and model directories are skipped with a message. A smoke stage (the first 4 items of every
# set) runs per model first; a model whose smoke fails is skipped, the others go on.
#   DRY_RUN=1 OURS="base=models/Qwen3.5-2B-Base" PEERS=1 bash repo/hpc/eval_reasoning.sbatch
set -euo pipefail
cd "$HOME/cotd"
module load cuda/12.4 2>/dev/null || true
export HF_HOME="$HOME/cotd/hf" TOKENIZERS_PARALLELISM=false PYTHONPATH="$HOME/cotd/cc1d${PYTHONPATH:+:$PYTHONPATH}"
PY=envs/cotd/bin/python
[ -n "${DRY_RUN:-}" ] && PY="echo [dry-run] python"
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi here"
ALL="bbeh=data/eval/bbeh.jsonl xkk=data/eval/xkk.jsonl proverqa=data/eval/proverqa.jsonl zebra=data/eval/zebra.jsonl zebra_ovl=data/eval/zebra_ovl.jsonl logiqa2=data/eval/logiqa2.jsonl lsat_lr=data/eval/lsat_lr.jsonl lsat_ar=data/eval/lsat_ar.jsonl"
SETS= SMOKE=
SM=runs/reason-smoke-data
mkdir -p "$SM"
for f in ${REASON_SETS:-$ALL}; do
  if [ -f "${f#*=}" ]; then
    SETS="$SETS $f"; head -n 4 "${f#*=}" > "$SM/${f%%=*}.jsonl"; SMOKE="$SMOKE ${f%%=*}=$SM/${f%%=*}.jsonl"
  else
    echo "missing eval file ${f#*=} (skipped)"
  fi
done
[ -n "$SETS" ] || { echo "no evaluation file present"; exit 1; }
echo "sets:$SETS"

run() {     # run <name> <command prefix taking the sets>: smoke on 4 items per set, then the full sets
  local name=$1; shift
  echo "=== $name smoke $(date +%T)"
  if "$@" $SMOKE --out "runs/reason-$name-smoke" > "logs/reason-$name-smoke.log" 2>&1; then
    rm -rf "runs/reason-$name-smoke"; echo "smoke ok $name"
    echo "=== $name full $(date +%T)"
    "$@" $SETS --out "runs/reason-$name" || echo "eval failed $name"
  else
    echo "SMOKE FAILED $name"; tail -n 30 "logs/reason-$name-smoke.log"
  fi
}
have() { [ -f "$1/config.json" ] || { echo "missing model $1 (skipped)"; return 1; }; }

for m in ${OURS:-}; do
  have "${m#*=}" && run "${m%%=*}" $PY repo/scripts/eval_student.py --model "${m#*=}" --bs 8 --eval
done
if [ -n "${PEERS:-}" ]; then
  have models/decider-2b-v11 && run peer-decider $PY repo/scripts/eval_peers.py --peer decider --model models/decider-2b-v11 --eval
  [ -d models/strands-decider-2B-v19 ] && run peer-strands $PY repo/scripts/eval_peers.py --peer strands \
    --model models/strands-decider-2B-v19 --base models/Qwen3.5-2B-Base --strands-src third_party/strands-decider/src --eval
fi
if [ -n "${JEFF:-}" ]; then
  have models/Jeff-Qwen3.5-2B && run jeff $PY repo/scripts/eval_jeff.py --model models/Jeff-Qwen3.5-2B --sets
fi
for m in ${LLM:-}; do
  have "${m#*=}" && run "${m%%=*}" $PY repo/scripts/eval_llm_baseline.py --model "${m#*=}" --bs 4 --eval
done
rm -rf "$SM"
echo "all done $(date +%T)"
