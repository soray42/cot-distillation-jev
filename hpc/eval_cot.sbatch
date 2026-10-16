#!/bin/bash
#SBATCH --job-name=cotd-cot
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# Chain-of-thought upper bound of the base model (scripts/eval_cot.py): the post-trained Qwen3.5-4B reasons step by step
# and names an option letter (chat template, greedy, left-padded batches) -> $OUT/preds_<set>.jsonl, gens_<set>.jsonl,
# metrics.json (accuracy with unparsable = wrong, generated tokens mean / median / p90, seconds per item). One domain
# per job:
#   DOMAIN=gsm   gsm8kv2 (GSM8K test, 1,317) + gsm1k (1,205)
#   DOMAIN=math  ng_d1 ... ng_d8 (400 each) + ng_ood (600): relay99's NG_EVALS; NOOOD=1 leaves ng_ood out
#   DOMAIN=kk    kkdeep (K&K, 4-20 people, 2,000)
# Knobs: MODEL (default models/Qwen3.5-4B) · THINK=1 (--enable-thinking) · MAXNEW (--max-new-tokens, default 2048) ·
# BS (default 16) · SHARD=i/n (0-based; items i, i+n, ...) · LIMIT=N · REPO (default repo) · OUT (default
# runs/cot-<model>[-think]-n<MAXNEW>; shards write shard-suffixed files into the same OUT).
#   DRY_RUN=1 DOMAIN=gsm bash repo/hpc/eval_cot.sbatch          # login node: checks files, echoes the Python commands
#   DOMAIN=gsm sbatch repo/hpc/eval_cot.sbatch
#   DOMAIN=math SHARD=0/2 sbatch repo/hpc/eval_cot.sbatch; DOMAIN=math SHARD=1/2 sbatch repo/hpc/eval_cot.sbatch
#   then on the login node (no GPU, no model): the merge command printed at the end of each shard's log.
# Smoke first (exits on failure): 8 GSM8K items at --max-new-tokens 256, once at bs 8 and once at bs 1; prints how many
# generations and parsed letters agree (left padding through the linear-attention layers; bf16 batch numerics may flip
# a few greedy tokens, most should agree) and the decode speed the job log reports.
# Walltime: each batch runs until its longest sequence ends or hits MAXNEW, so the ceiling is
# ceil(items / BS) x MAXNEW / (decode steps per second); at 25 steps/s and BS 16: gsm 3.6 h, math 5.4 h (4.6 h with
# NOOOD=1; or two shards), kk 2.8 h. Each set's files are written when the set finishes, so a timeout loses only the
# set in flight (kk is one set). The job log prints the model's generation_config: a repetition or presence penalty
# there still applies under greedy decoding.
set -euo pipefail
cd "$HOME/cotd"
module load cuda/12.4 2>/dev/null || true
export HF_HOME="$HOME/cotd/hf" TOKENIZERS_PARALLELISM=false PYTHONPATH="$HOME/cotd/cc1d${PYTHONPATH:+:$PYTHONPATH}"
PY=envs/cotd/bin/python
[ -n "${DRY_RUN:-}" ] && PY="echo [dry-run] python"
REPO="${REPO:-repo}"; MODEL="${MODEL:-models/Qwen3.5-4B}"; MAXNEW="${MAXNEW:-2048}"; BS="${BS:-16}"
case "${DOMAIN:-}" in
  gsm)  SETS="gsm8kv2=data/eval/gsm8k_mc_v2.jsonl gsm1k=data/eval/gsm1k_mc.jsonl" ;;
  math) SETS=; for d in 1 2 3 4 5 6 7 8; do SETS="$SETS ng_d$d=data/eval/numgen_d$d.jsonl"; done
        [ -n "${NOOOD:-}" ] || SETS="$SETS ng_ood=data/eval/numgen_ood.jsonl" ;;
  kk)   SETS="kkdeep=data/eval/kk_deep.jsonl" ;;
  *) echo "set DOMAIN=gsm, math or kk"; exit 1 ;;
esac
OUT="${OUT:-runs/cot-$(basename "$MODEL")${THINK:+-think}-n$MAXNEW}"
[ -f "$REPO/scripts/eval_cot.py" ] || { echo "missing $REPO/scripts/eval_cot.py"; exit 1; }
[ -f "$MODEL/config.json" ] || { echo "missing model $MODEL"; exit 1; }
for f in $SETS; do [ -f "${f#*=}" ] || { echo "missing eval file ${f#*=}"; exit 1; }; done
COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
echo "code: $REPO @ $COMMIT; model $MODEL; domain $DOMAIN; sets:$SETS; out $OUT; thinking $([ -n "${THINK:-}" ] && echo on || echo off); max new $MAXNEW; bs $BS${SHARD:+; shard $SHARD}"
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi here"
ARGS=(--model "$MODEL" --max-new-tokens "$MAXNEW" ${THINK:+--enable-thinking})

echo "=== smoke $(date +%T)"
SM=$(mktemp -d)
head -n 8 data/eval/gsm8k_mc_v2.jsonl > "$SM/s.jsonl"
for b in 8 1; do
  $PY "$REPO/scripts/eval_cot.py" "${ARGS[@]}" --max-new-tokens 256 --bs "$b" --eval "smoke=$SM/s.jsonl" \
    --out "$SM/bs$b" > "$SM/bs$b.log" 2>&1 || { echo "SMOKE FAILED (bs $b)"; tail -n 30 "$SM/bs$b.log"; exit 1; }
  [ -n "${DRY_RUN:-}" ] || [ -s "$SM/bs$b/preds_smoke.jsonl" ] || { echo "SMOKE FAILED (bs $b: no predictions)"; tail -n 30 "$SM/bs$b.log"; exit 1; }
done
if [ -z "${DRY_RUN:-}" ]; then
  grep -E "^smoke |steps/s" "$SM/bs8.log" | tail -n 2
  $PY - "$SM/bs8" "$SM/bs1" <<'EOF'
import json, sys
a, b = ([json.loads(l) for l in open(f"{d}/gens_smoke.jsonl")] for d in sys.argv[1:3])
same_text = sum(x["text"] == y["text"] for x, y in zip(a, b))
same_letter = sum(x["pred_label"] == y["pred_label"] for x, y in zip(a, b))
print(f"padding check, bs 8 vs bs 1 on {len(a)} items: identical generations {same_text}/{len(a)}, "
      f"same parsed letter {same_letter}/{len(a)}")
EOF
fi
rm -rf "$SM"
echo "=== smoke passed $(date +%T)"

echo "=== $DOMAIN $(date +%T)"
$PY "$REPO/scripts/eval_cot.py" "${ARGS[@]}" --bs "$BS" --eval $SETS --out "$OUT" ${SHARD:+--shard "$SHARD"} ${LIMIT:+--limit "$LIMIT"}
if [ -n "${SHARD:-}" ]; then
  echo "after all ${SHARD#*/} shards of $DOMAIN: envs/cotd/bin/python $REPO/scripts/eval_cot.py --merge --shards ${SHARD#*/} --out $OUT --eval $SETS"
fi
echo "all done $(date +%T)"
