#!/bin/bash
#SBATCH --job-name=cotd-mem12b
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# Memory / speed / resume smoke for full-parameter Gemma 4 12B (text only, bf16 weights on the GPU, activation
# checkpointing, CPUOffloadAdamW from src/cotdistill/offload.py) on one 4g.40gb slice. Standalone: scripts/mem12b.py,
# not train_student.py. Every result is one "MEM12B {json}" line (also in $W/main.log); the cost model at the end turns
# them into hours per run (scripts/cost12b.py --from-log; notes/next_round/scale12b/cost.md).
# Stages (PARTS selects; default all, in this order; each stage exits the job with "SMOKE FAILED" on an error):
#   unit     COTD_CUDA_TESTS=1 unittest tests.test_offload (incl. its 2 CUDA cases) + tests.test_mem12b + review tests (test_scale12b_review, test_offload_numerics), about 1 min
#   tiny     the whole mem12b.py path on a 2-layer random model built from the real config + tokenizer (no weights)
#   main     the 12B: bench (PCIe GB/s, CPU AdamW Gparam/s), readout (zero-shot acc / letter mass / raw letter logits,
#            first READN items of kk_heldout, bbh, jevbench_public; BOS and no-BOS), baseline (GPU-only fwd+bwd per
#            micro-batch, SETTINGS + BASE_EXTRA), mem (SETTINGS, STEPS updates x GA micro-batches each), roundtrip
#            (2 updates, save, 1 update, mmap load, the same update again; RT_MOMENTS=none (default): masters only,
#            47.6 GB, masters/weights bit-identical and AdamW restarted; RT_MOMENTS=fp32: 143 GB, bit-identical update)
#   cost     scripts/cost12b.py --from-log $W/main.log (hours per arm from the measured numbers)
# Host memory: 214 GB of optimizer state (masters, accumulators, m, v, pinned stage) + 2.8 GB because torch's pinned
# allocator rounds every stage block up to a power of two (26.6 GB pinned, not 23.8) = 217 GB anonymous, + transients,
# + the roundtrip's save (dirty page cache) and mmap load (file pages), both charged to the job's cgroup: 300G = 322 GB
# (the OPT report's figure for an mmap resume; memory is not capped by the QoS).
# The 12B parts need MODEL12B (default ~/cotd/models/gemma-4-12B, revision 023679ed) with its 23.92 GB
# model.safetensors complete; otherwise only unit and tiny run (tiny needs config.json + tokenizer only).
# Disk: the roundtrip writes one file (47.6 GB with RT_MOMENTS=none, 143 GB with fp32) into RT_DIR and deletes it.
# RT_DIR defaults to node-local scratch ($SLURM_TMPDIR, else $TMPDIR; skipped if that is tmpfs = RAM), else $W. df cannot see the user quota (100 GB
# on home, of which the 12B base model already takes 24 GB), and hitting the quota mid-write would also break every
# other job of the account that writes to home. So when RT_DIR is under $HOME the roundtrip runs only with
# RT_FREE_GB=<free GB from lquota on the LOGIN node> (lquota does not exist on compute nodes); otherwise it prints a
# "skip" line (not a failure). An OSError during the save/load is reported as a skip too.
# CPUs: 8 of the account's 12 (QoS 2026-10-06), so a 4-CPU job of the same account can still run next to it; the
# CPU AdamW step is memory-bandwidth bound, so 10 threads would gain little. (If the QoS is back to 8, nothing else
# of the account can run alongside.)
# From ~/cotd:   CC1D=1 sbatch repo/hpc/mem12b.sbatch
# Login-node dry run first (echoes every Python command; runs only the light "check" part: config + tokenizer):
#                DRY_RUN=1 bash repo/hpc/mem12b.sbatch          (DRY_CHECK=0 to skip that part too)
# Knobs: SETTINGS (default 2x1536,4x1024,8x1024; BSxL[:inflightGiB]), BASE_EXTRA (16x1024, GPU-only), STEPS (8), GA (16), READN (50),
# THREADS (default SLURM_CPUS_PER_TASK), INFLIGHT (2 GiB), BUDGETMIN (150: mem updates stop so the roundtrip still
# fits in the 3 h wall), RT_DIR, RT_FREE_GB, RT_MOMENTS (none|fp32), WORK, ATTN (sdpa default; flash_attention_2
# cannot run head_dim 512).
# A failing main stage exits before the cost stage; then run scripts/cost12b.py --from-log on the partial main.log.
set -euo pipefail
cd "$HOME/cotd"
module load cuda/12.4 2>/dev/null || true
NCPU="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="$HOME/cotd/hf" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="$NCPU" MKL_NUM_THREADS="$NCPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
[ -n "${CC1D:-}" ] && export PYTHONPATH="$HOME/cotd/cc1d${PYTHONPATH:+:$PYTHONPATH}"
PY="$HOME/cotd/envs/cotd/bin/python"
RUN="$PY"
[ -n "${DRY_RUN:-}" ] && RUN="echo [dry-run] python"
REPO="${REPO:-repo}"
MODEL12B="${MODEL12B:-$HOME/cotd/models/gemma-4-12B}"
PARTS="${PARTS:-unit tiny main cost}"
MAINPARTS="${MAINPARTS:-bench,readout,baseline,mem,roundtrip}"
SETTINGS="${SETTINGS:-2x1536,4x1024,8x1024}"; BASE_EXTRA="${BASE_EXTRA:-16x1024}"   # 8x1024 = G3/A2 mb 8 shape
STEPS="${STEPS:-8}"; GA="${GA:-16}"; READN="${READN:-50}"; THREADS="${THREADS:-$NCPU}"; INFLIGHT="${INFLIGHT:-2}"
BUDGETMIN="${BUDGETMIN:-150}"
W="${WORK:-runs/mem12b-${SLURM_JOB_ID:-local}}"
RT_DIR="${RT_DIR:-${SLURM_TMPDIR:-${TMPDIR:-$W}}}"; RT_MOMENTS="${RT_MOMENTS:-none}"
EVALS="data/eval/kk_heldout.jsonl,data/eval/bbh.jsonl,data/eval/jevbench_public.jsonl"
COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
DIRTY=$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l || echo "?")
has() { [[ " $PARTS " == *" $1 "* ]]; }
fail() { echo "SMOKE FAILED: $*"; exit 1; }

echo "code: $REPO @ $COMMIT (+$DIRTY uncommitted paths)  parts: $PARTS  main: $MAINPARTS  work: $W  rt: $RT_DIR"
echo "host: $(hostname)  cpus: $NCPU  mem limit: ${SLURM_MEM_PER_NODE:-?} MB  job: ${SLURM_JOB_ID:-none}"
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi here"
for f in "$REPO/scripts/mem12b.py" "$REPO/scripts/cost12b.py" "$REPO/src/cotdistill/offload.py" \
         "$REPO/tests/test_offload.py" "$REPO/tests/test_mem12b.py" "$REPO/tests/test_scale12b_review.py" "$REPO/tests/test_offload_numerics.py" ${EVALS//,/ }; do
  [ -f "$f" ] || fail "missing $f (sync the new files to the HPC repo first)"
done
[ -f "$MODEL12B/config.json" ] && [ -f "$MODEL12B/tokenizer.json" ] || fail "missing $MODEL12B/config.json or tokenizer.json"
ST_BYTES=$( (stat -c %s "$MODEL12B"/*.safetensors 2>/dev/null || true) | awk '{s+=$1} END {print s+0}')
HAVE12B=1
if [ "$ST_BYTES" -lt 23900000000 ]; then
  HAVE12B=
  echo "MEM12B {\"skip\": \"12B parts: safetensors in $MODEL12B total $ST_BYTES bytes < 23.9e9 (download incomplete?)\"}"
fi
mkdir -p "$W" "$RT_DIR"
WABS=$(cd "$W" && pwd)
df -h "$RT_DIR" | tail -1 | awk '{print "rt dir filesystem free:", $4, "(the user quota may be lower: lquota on the login node)"}'
RTABS=$(cd "$RT_DIR" && pwd -P); HOMEABS=$(cd "$HOME" && pwd -P)
RT_ARGS=(--rt-moments "$RT_MOMENTS")
RTFS=$(stat -f -c %T "$RTABS" 2>/dev/null || echo unknown)
if [[ "$RTFS" == tmpfs || "$RTFS" == ramfs ]]; then   # a file there is RAM charged to this job and not reclaimable
  RT_ARGS+=(--rt-free-gb 0)
  echo "roundtrip: RT_DIR $RTABS is $RTFS (RAM, on top of 217 GB of optimizer state) -> it will be skipped"
elif [ -n "${RT_FREE_GB:-}" ]; then RT_ARGS+=(--rt-free-gb "$RT_FREE_GB")
elif [[ "$RTABS/" == "$HOMEABS/"* ]]; then
  RT_ARGS+=(--rt-free-gb 0)
  echo "roundtrip: RT_DIR $RTABS is on the quota'd home filesystem and RT_FREE_GB is unset -> it will be skipped"
fi
$RUN -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, \
torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda', \
'free/total GiB', [round(x / 2**30, 2) for x in torch.cuda.mem_get_info()] if torch.cuda.is_available() else '-')" \
  || fail "torch import"

COMMON=(--model "$MODEL12B" --readout-files "$EVALS" --threads "$THREADS" --inflight-gib "$INFLIGHT")
[ -n "${ATTN:-}" ] && COMMON+=(--attn "$ATTN")

if [ -n "${DRY_RUN:-}" ] && [ "${DRY_CHECK:-1}" = 1 ]; then
  echo "=== check (config + tokenizer, runs for real in the dry run)"
  $PY "$REPO/scripts/mem12b.py" "${COMMON[@]}" --parts check --rt-dir "$RT_DIR" || fail "check"
fi

if has unit; then
  echo "=== unit tests (CUDA cases on)  $(date +%T)"
  ( cd "$REPO" && COTD_CUDA_TESTS=1 OMP_NUM_THREADS=2 $RUN -m unittest tests.test_offload tests.test_mem12b tests.test_scale12b_review tests.test_offload_numerics ) \
    > "$W/unit.log" 2>&1 || { tail -30 "$W/unit.log"; fail "unit tests"; }
  tail -3 "$W/unit.log"
fi

if has tiny; then
  echo "=== tiny smoke: every mem12b part on a 2-layer random model  $(date +%T)"
  $RUN "$REPO/scripts/mem12b.py" "${COMMON[@]}" --tiny --parts bench,readout,baseline,mem,roundtrip \
    --settings "$SETTINGS" --baseline-extra "$BASE_EXTRA" --steps 2 --ga 2 --baseline-micro 1 --readout-n 5 \
    --rt-dir "$W/tiny_rt" --rt-moments fp32 > "$W/tiny.log" 2>&1 || { tail -40 "$W/tiny.log"; fail "tiny"; }
  # (tiny always runs the exact fp32 roundtrip: about 0.2 GB (262k x 64 embedding), and the only place the bit-identical resume is checked on
  # CUDA (pinned stages, side stream) when the 12B roundtrip uses RT_MOMENTS=none)
  if [ -n "${DRY_RUN:-}" ]; then cat "$W/tiny.log"; else grep '^MEM12B' "$W/tiny.log" || true; fi
  [ -n "${DRY_RUN:-}" ] || grep -q '"part": "done"' "$W/tiny.log" || fail "tiny: no done line"
  [ -n "${DRY_RUN:-}" ] || ! grep -q '"status": "OOM"\|"ok": false' "$W/tiny.log" || fail "tiny: OOM or failed check"
fi

if has main; then
  if [ -z "$HAVE12B" ]; then
    echo "=== main skipped: 12B weights incomplete"
  else
    echo "=== 12B: $MAINPARTS  settings $SETTINGS (+baseline $BASE_EXTRA)  GA $GA  steps $STEPS  $(date +%T)"
    set +e
    $RUN "$REPO/scripts/mem12b.py" "${COMMON[@]}" --parts "$MAINPARTS" --settings "$SETTINGS" \
      --baseline-extra "$BASE_EXTRA" --steps "$STEPS" --ga "$GA" --readout-n "$READN" --rt-dir "$RT_DIR" \
      "${RT_ARGS[@]}" --budget-min "$BUDGETMIN" 2>&1 | tee "$W/main.log"
    RC=${PIPESTATUS[0]}
    set -e
    echo "main exit code: $RC"
    [ -n "${DRY_RUN:-}" ] || grep '^MEM12B' "$W/main.log" > "$W/mem12b.jsonl" || true
    [ "$RC" = 0 ] || fail "main (exit $RC; partial MEM12B lines above are still valid)"
  fi
fi

if has cost; then
  echo "=== cost model from the measured numbers  $(date +%T)"
  COSTJSON="${COSTJSON:-$REPO/notes/next_round/tree_forcing/cost.json}"   # notes/ is gitignored: sync it or set COSTJSON
  if [ ! -f "$COSTJSON" ]; then
    echo "cost model skipped: $COSTJSON missing (run scripts/cost12b.py --from-log $WABS/main.log locally)"
  elif [ -n "${DRY_RUN:-}" ] || [ -s "$W/main.log" ]; then
    $RUN "$REPO/scripts/cost12b.py" --from-log "$WABS/main.log" --cost-json "$COSTJSON" \
      --train data/student_v3f/train.jsonl || echo "cost model failed (not fatal)"
  else
    echo "no main.log: cost model skipped"
  fi
fi
rm -rf "$W/tiny_rt"
echo "all done $(date +%T)"
