#!/bin/bash
#SBATCH --job-name=cotd-fsdp12b
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# Two-slice smoke for FSDP2 training (scripts/train_student.py --fsdp; notes/next_round/scale12b/fsdp2.md §6). Every
# result is one "FSDP12B {json}" line (also in $W/main.log); the last one is the recommendation.
# Stages (PARTS, default all, in this order; "unit" failing stops the job, later stages report and go on):
#   unit      tests.test_fsdp: CPU gloo equivalence tests + the NCCL case on the job's two slices (COTD_CUDA_TESTS=1)
#   bench     all_gather / reduce_scatter GB/s between the slices -> estimated FSDP communication per 12B micro-batch
#   equiv     2B (MODEL2B): G3 on 960 problems of data/student_v3f (every 7th line: all sources; 30 updates of 32
#             problems), seed 0, LS .1, LR 1e-5: one slice vs two slices for 8bit-sr (both bf16 + torchao), cpu-adamw (vs
#             the 2B recipe fp32master + AdamW) and cpu-adamw with fp32 shards (the exact analogue); per-update loss
#             differences, val acc (200 items) and prediction agreement. The fp32-shards pair is the strict
#             sum-semantics check (recommend: < 2% relative); the bf16 pairs differ by design (stochastic rounding,
#             bf16 vs fp32 weights) and only have to track (< 10%)
#   readout   12B (MODEL12B) on the first slice: zero-shot accuracy / letter mass / raw letter logits / BOS vs no BOS,
#             first READN items of kk_heldout, bbh, jevbench_public (scripts/mem12b.py readout code)
#   train     12B, 10 updates each of ARMS12 (A2, G3 on 320 problems of v3f; TFL on 320 of v3sl with
#             --tree-sources knights_knaves; every 22nd line) for each FSDP_OPTIMS setup and TOKBUDGETS budget: peak GiB
#             per rank, s/update (fwd+bwd / est. communication / clip / optimizer), padded tok/s, host RSS. 8bit-sr is
#             expected not to fit (fsdp2.md §2): its first run is A2 at the smallest budget; an OOM there skips the rest.
#             A run is started only if RUNMIN minutes remain before BUDGETMIN (default 170 of the 180-minute wall).
#   recommend projection to hours per arm (226 updates/epoch, A2-c12 / c4) and the recommendation line
# Before submitting, on the login node: lquota (the runs write only small metrics; nothing is saved), check that
# $MODEL12B/model.safetensors is 23.92 GB, sync repo (src/cotdistill/fsdp.py, scripts/train_student.py,
# scripts/fsdp12b.py, scripts/mem12b.py, src/cotdistill/student.py, tests/test_fsdp.py, tests/test_train_resume.py,
# tests/test_student.py, this file), then the dry run (Python only echoed):
#   cd ~/cotd && DRY_RUN=1 bash repo/hpc/fsdp12b.sbatch
#   sbatch repo/hpc/fsdp12b.sbatch
# Knobs: PARTS, MODEL2B (default ~/cotd/models/Qwen3.5-2B-Base), MODEL12B (~/cotd/models/gemma-4-12B), ARMS12 ("TFL G3
# A2"), FSDP_OPTIMS ("cpu-adamw 8bit-sr"), TOKBUDGETS ("4096 3072"), READN (50), BUDGETMIN (170), RUNMIN (14), WORK, REPO,
# CC1D (causal-conv1d kernel for Qwen3.5, as in train_arm.sbatch).
set -uo pipefail
T0=$(date +%s)
cd "$HOME/cotd"
module load cuda/12.4 2>/dev/null || true
ncpu="${SLURM_CPUS_PER_TASK:-12}"
export HF_HOME="$HOME/cotd/hf" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=$(( ncpu / 2 ))
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
[ -n "${CC1D:-}" ] && export PYTHONPATH="$HOME/cotd/cc1d${PYTHONPATH:+:$PYTHONPATH}"
py="$HOME/cotd/envs/cotd/bin/python"
run="$py"
[ -n "${DRY_RUN:-}" ] && run="echo [dry-run] python"
trun="$run -m torch.distributed.run --standalone --nproc_per_node 2"
REPO="${REPO:-repo}"
MODEL2B="${MODEL2B:-$HOME/cotd/models/Qwen3.5-2B-Base}"
MODEL12B="${MODEL12B:-$HOME/cotd/models/gemma-4-12B}"
PARTS="${PARTS:-unit bench equiv readout train recommend}"
ARMS12="${ARMS12:-TFL G3 A2}"; FSDP_OPTIMS="${FSDP_OPTIMS:-cpu-adamw 8bit-sr}"; TOKBUDGETS="${TOKBUDGETS:-4096 3072}"
READN="${READN:-50}"; BUDGETMIN="${BUDGETMIN:-170}"; RUNMIN="${RUNMIN:-14}"
W="${WORK:-runs/fsdp12b-${SLURM_JOB_ID:-local}}"
mkdir -p "$W"
log="$W/main.log"
commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
has() { [[ " $PARTS " == *" $1 "* ]]; }
say() { echo "$*" | tee -a "$log"; }
fsdp_line() { grep -h '^FSDP12B' "$@" 2>/dev/null | tee -a "$log"; }
mins() { echo $(( ($(date +%s) - T0) / 60 )); }
show() { [ -z "${DRY_RUN:-}" ] || cat "$@"; }        # dry run: the echoed commands

say "code: $REPO @ $commit  parts: $PARTS  work: $W  host: $(hostname)  cpus: $ncpu  job: ${SLURM_JOB_ID:-none}"
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi here"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  OMP_NUM_THREADS=$OMP_NUM_THREADS per rank"
for f in scripts/train_student.py scripts/fsdp12b.py scripts/mem12b.py src/cotdistill/fsdp.py src/cotdistill/student.py \
         tests/test_fsdp.py tests/test_train_resume.py tests/test_student.py; do
  [ -f "$REPO/$f" ] || { say "SMOKE FAILED: missing $REPO/$f (sync the new files first)"; exit 1; }
done
for f in data/student_v3f/train.jsonl data/student_v3f/val.jsonl data/student_v3sl/train.jsonl data/eval/kk_heldout.jsonl \
         data/eval/bbh.jsonl data/eval/jevbench_public.jsonl; do
  [ -f "$f" ] || { say "SMOKE FAILED: missing $f"; exit 1; }
done
if [ -z "${DRY_RUN:-}" ]; then
  n=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES:-}")
  [ "${n:-0}" -ge 2 ] || { say "SMOKE FAILED: this job sees ${n:-0} GPU(s); it needs --gres=gpu:4g.40gb:2"; exit 1; }
fi
$run -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, \
[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])" || { say "SMOKE FAILED: torch import"; exit 1; }
have12=1
st=$( (stat -c %s "$MODEL12B"/*.safetensors 2>/dev/null || true) | awk '{s+=$1} END {print s+0}')
if [ "$st" -lt 23900000000 ]; then
  have12=
  say "FSDP12B {\"skip\": \"12B parts: safetensors in $MODEL12B total $st bytes < 23.9e9\"}"
fi
have2=1
[ -f "$MODEL2B/config.json" ] || { have2=; say "FSDP12B {\"skip\": \"equiv: no $MODEL2B/config.json\"}"; }

# data subsets: every k-th line, so every source is in (the files are sorted by source)
awk 'NR % 7 == 1' data/student_v3f/train.jsonl | head -n 960 > "$W/v3f960.jsonl"
head -n 200 data/student_v3f/val.jsonl > "$W/val200.jsonl"
awk 'NR % 22 == 1' data/student_v3f/train.jsonl | head -n 320 > "$W/v3f320.jsonl"
awk 'NR % 22 == 1' data/student_v3sl/train.jsonl | head -n 320 > "$W/v3sl320.jsonl"
awk 'NR % 14 == 1' data/eval/kk_heldout.jsonl | head -n 32 > "$W/kk32.jsonl"
common="--lr 1e-5 --max-len 1536 --label-smoothing 0.1 --epochs 1 --seed 0 --log-every 1 --code-commit $commit"
g3="--final teacher --subq cot --subq-target cot --grouped-aux 2 --aux-kind cot --aux-weight 0.5 --micro-bs 2"
arm_args() {
  case "$1" in
    A2) echo "--final teacher --subq none --subq-target cot --micro-bs 2 --grad-accum 16 --train $W/v3f320.jsonl" ;;
    G3) echo "$g3 --train $W/v3f320.jsonl" ;;
    TFL) echo "--final teacher --subq cot --subq-target cot --tree-full true --tree-view local --tree-cap 0 --tree-sources knights_knaves --micro-bs 2 --train $W/v3sl320.jsonl" ;;
    *) echo "unknown arm $1" >&2; return 1 ;;
  esac
}

if has unit; then
  say "=== unit: tests.test_fsdp (gloo CPU + NCCL on the two slices)  $(date +%T)"
  ( cd "$REPO" && COTD_CUDA_TESTS=1 OMP_NUM_THREADS=2 $run -m unittest tests.test_fsdp ) > "$W/unit.log" 2>&1 \
    || { tail -40 "$W/unit.log"; say "SMOKE FAILED: unit"; exit 1; }
  grep -q "^OK" "$W/unit.log" || [ -n "${DRY_RUN:-}" ] || { tail -40 "$W/unit.log"; say "SMOKE FAILED: unit"; exit 1; }
  tail -3 "$W/unit.log"
fi

if has bench; then
  say "=== bench: collectives between the slices  $(date +%T)"
  $trun "$REPO/scripts/fsdp12b.py" bench --gib 1 --out "$W/bench.json" > "$W/bench.log" 2>&1 || tail -20 "$W/bench.log"
  show "$W/bench.log"
  fsdp_line "$W/bench.log"
fi

if has equiv && [ -n "$have2" ]; then
  say "=== equiv: 2B G3, 30 updates, one slice vs two  $(date +%T)"
  base="--model $MODEL2B --train $W/v3f960.jsonl --eval val=$W/val200.jsonl $g3 $common"
  $run "$REPO/scripts/train_student.py" $base --precision bf16 --optim adamw8bit-sr --log-update-loss \
    --out "$W/eq-bf16-single" > "$W/eq-bf16-single.log" 2>&1 || tail -20 "$W/eq-bf16-single.log"
  show "$W/eq-bf16-single.log"
  $run "$REPO/scripts/train_student.py" $base --precision fp32master --optim adamw --log-update-loss \
    --out "$W/eq-fp32-single" > "$W/eq-fp32-single.log" 2>&1 || tail -20 "$W/eq-fp32-single.log"
  show "$W/eq-fp32-single.log"
  for v in "8bit-sr bf16 eq-bf16-single" "cpu-adamw bf16 eq-fp32-single" "cpu-adamw fp32master eq-fp32-single"; do
    set -- $v
    o="$W/eq-$1-$2-fsdp"
    $trun "$REPO/scripts/train_student.py" $base --precision "$2" --fsdp --fsdp-optim "$1" --token-budget 4096 \
      --out "$o" > "$o.log" 2>&1 || tail -20 "$o.log"
    $run "$REPO/scripts/fsdp12b.py" compare --ref "$W/$3" --fsdp "$o" --label "G3 2B $1 $2 shards vs one slice ($3)" \
      > "$o.cmp" 2>&1
    show "$o.log" "$o.cmp"
    fsdp_line "$o.cmp"
  done
fi

if has readout && [ -n "$have12" ]; then
  say "=== readout: 12B zero-shot on one slice  $(date +%T)"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" $run "$REPO/scripts/fsdp12b.py" readout --model "$MODEL12B" \
    --n "$READN" > "$W/readout.log" 2>&1 || tail -20 "$W/readout.log"
  show "$W/readout.log"
  fsdp_line "$W/readout.log"
fi

if has train && [ -n "$have12" ]; then
  say "=== train: 12B, 10 updates per setting  $(date +%T)"
  for opt in $FSDP_OPTIMS; do
    order="$TOKBUDGETS"; arms="$ARMS12"
    if [ "$opt" = 8bit-sr ]; then                       # cheapest first: an OOM there ends the 8bit-sr runs
      order=$(tr ' ' '\n' <<< "$TOKBUDGETS" | sort -n | tr '\n' ' '); arms=$(tr ' ' '\n' <<< "$ARMS12" | sort | tr '\n' ' ')
    fi
    oomed=
    for b in $order; do
      for arm in $arms; do
        tag="$arm-$opt-tb$b"
        if [ -n "$oomed" ]; then say "FSDP12B {\"part\": \"train_12b\", \"arm\": \"$arm\", \"optim\": \"$opt\", \"token_budget\": $b, \"status\": \"skipped after OOM\"}"; continue; fi
        if [ $(( $(mins) + RUNMIN )) -gt "$BUDGETMIN" ]; then
          say "FSDP12B {\"part\": \"train_12b\", \"arm\": \"$arm\", \"optim\": \"$opt\", \"token_budget\": $b, \"status\": \"skipped: time budget\"}"; continue
        fi
        echo "--- $tag  $(date +%T)"
        $trun "$REPO/scripts/train_student.py" --model "$MODEL12B" $(arm_args "$arm") --eval "smoke=$W/kk32.jsonl" \
          --eval-bs 16 $common --precision bf16 --fsdp --fsdp-optim "$opt" --token-budget "$b" --out "$W/$tag" \
          > "$W/$tag.log" 2>&1 || tail -15 "$W/$tag.log"
        $run "$REPO/scripts/fsdp12b.py" summarize --out "$W/$tag" --log "$W/$tag.log" --arm "$arm" --optim "$opt" \
          --budget "$b" --bench "$W/bench.json" > "$W/$tag.sum" 2>&1
        show "$W/$tag.log" "$W/$tag.sum"
        fsdp_line "$W/$tag.sum"
        if [ "$opt" = 8bit-sr ] && grep -q '"status": "OOM"' "$W/$tag.sum"; then oomed=1; fi
      done
    done
  done
fi

if has recommend; then
  say "=== recommendation  $(date +%T)"
  $run "$REPO/scripts/fsdp12b.py" recommend --log "$log" > "$W/recommend.log" 2>&1 || tail -20 "$W/recommend.log"
  show "$W/recommend.log"
  fsdp_line "$W/recommend.log"
  grep -h '^RECOMMENDATION' "$W/recommend.log" | tee -a "$log"
fi
echo "all done $(date +%T) ($(mins) min)"
