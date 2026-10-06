#!/bin/bash
#SBATCH --job-name=cotd-kkaudit
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# K&K causal audit (notes/next_round/causal_audit/kk_design.md): scripts/kk_audit.py on each model, one after another.
# From ~/cotd (data/eval/kk_audit_pairs.jsonl pushed with hpc/push_data.sh data/eval; the solver is not needed here):
#   MODELS="TFL-s31=runs/TFL-...-s31/model A2-s31=runs/A2-...-s31/model" sbatch repo/hpc/kk_audit.sbatch
# MODELS: space-separated name=model_dir. LAYERS (default auto: 4,8,...,n_layers-4 per model, i.e. 4..20 for the 2B and
# 4..28 for the 4B, so 2B and 4B models can share one MODELS list), CONDS (default: kk_audit.py's), K (rank, default 1),
# EPOCHS (3), MAXTR/MAXTE (train/test pairs per condition, 600/600), BS (8), XARGS (more kk_audit.py flags, e.g.
# "--all-pairs"), PAIRS (default data/eval/kk_audit_pairs.jsonl). OUTROOT (default runs/kkaudit):
# <OUTROOT>/<name>/{audit.json,summary.jsonl,preds.jsonl}, and every model's KKAUDIT lines are appended to
# <OUTROOT>/summary.jsonl (one JSON line per model x layer x condition; the "_head" line carries the job id).
# REPO: code checkout the job runs (default repo; pin a frozen checkout for registered runs, as train_arm.sbatch); its
# commit (with -dirty if tracked files differ) is printed and written next to the results (code_commit.txt).
# CC1D (default 1): puts the causal-conv1d kernel ($HOME/cotd/cc1d) on PYTHONPATH when it exists; CC1D=0 to skip.
# Time (40 GB slice, 30 TFLOP/s effective; role/mid conditions are tested on every mid test pair, --max-test-mid 0):
# 2B about 2.5-3 h per model (4 per 12 h job); 4B (about 2.3x FLOPs per pass, 7 layers) about 8 h, one per job
# (EPOCHS=2 or LAYERS=4,8,12,16,20,24 to fit two). --check prints the
# measured estimate; keep 2B and 4B models in separate jobs when the walltime is tight.
# Smoke stage first (exits on failure): kk_audit.py --check on the first model (64 pairs per family: identity patches,
# token sites, full-exchange copy rates, gradients, weights unchanged, bf16 exchange fidelity, peak memory, timing and a
# GPU-hour estimate for the full run with these settings), then a 1-layer, 1-epoch run on 24/24 pairs that must print
# KKAUDIT lines. SMOKE_ONLY=1 stops after it.
# Relay scripts: MODELS LAYERS CONDS K EPOCHS MAXTR MAXTE BS XARGS PAIRS OUTROOT REPO CC1D SMOKE_ONLY DRY_RUN belong in
# the reset list (internal variables are lower case).
# Dry-run the shell logic on the login node first (Python is only echoed):
#   DRY_RUN=1 MODELS="a=runs/x/model" bash repo/hpc/kk_audit.sbatch
set -euo pipefail
cd "$HOME/cotd"
module load cuda/12.4 2>/dev/null || true
export HF_HOME="$HOME/cotd/hf" TOKENIZERS_PARALLELISM=false
if [ "${CC1D:-1}" != 0 ]; then
  if [ -d "$HOME/cotd/cc1d" ]; then export PYTHONPATH="$HOME/cotd/cc1d${PYTHONPATH:+:$PYTHONPATH}"
  else echo "warning: $HOME/cotd/cc1d missing, Gated DeltaNet runs without the causal-conv1d kernel (slower)"; fi
fi
py=envs/cotd/bin/python
[ -n "${DRY_RUN:-}" ] && py="echo [dry-run] python"
repo="${REPO:-repo}"
[ -f "$repo/scripts/kk_audit.py" ] || { echo "missing $repo/scripts/kk_audit.py"; exit 1; }
commit=$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo unknown)
[ "$commit" = unknown ] || [ -z "$(git -C "$repo" status --porcelain --untracked-files=no 2>/dev/null)" ] || commit="$commit-dirty"
echo "code: $repo @ $commit"
pairs="${PAIRS:-data/eval/kk_audit_pairs.jsonl}"
[ -f "$pairs" ] || { echo "missing $pairs (build with scripts/build_kk_audit_pairs.py and push data/eval)"; exit 1; }
echo "pairs: $pairs md5 $(md5sum "$pairs" | cut -c1-32)"
[ -n "${MODELS:-}" ] || { echo "set MODELS=\"name=model_dir ...\""; exit 1; }
for m in $MODELS; do
  [ -n "${DRY_RUN:-}" ] || [ -f "${m#*=}/config.json" ] || { echo "missing ${m#*=}/config.json"; exit 1; }
done
layers="${LAYERS:-auto}"
outroot="${OUTROOT:-runs/kkaudit}"
mkdir -p "$outroot"
echo "$commit" > "$outroot/code_commit.txt"
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi here"
common=(--pairs "$pairs" --bs "${BS:-8}" --k "${K:-1}" --epochs "${EPOCHS:-3}" --max-train "${MAXTR:-600}"
        --max-test "${MAXTE:-600}")

echo "=== smoke stage $(date +%T)"
first="${MODELS%% *}"
sm="$outroot/_smoke"
rm -rf "$sm"; mkdir -p "$sm"
$py "$repo/scripts/kk_audit.py" --model "${first#*=}" --name smoke --out "$sm/check" --check --max-pairs 64 \
  --layers "$layers" "${common[@]}" ${CONDS:+--conds "$CONDS"} ${XARGS:-} \
  || { echo "SMOKE FAILED (check)"; exit 1; }
if [ -z "${DRY_RUN:-}" ]; then
  $py - "$sm/check/check.json" ${XARGS:-} <<'EOF' || { echo "SMOKE FAILED (check values)"; exit 1; }
import json, sys
c = json.load(open(sys.argv[1]))
bad = [k for k, v in c.items() if k.startswith("identity_max_abs_diff") and v > 1e-2]
assert not bad, ("identity patch changes the logits", {k: c[k] for k in bad})
assert c["stmt_site_is_quote"] == 1.0 and c["name_site_in_name"] == 1.0, "token sites"
assert c["das_grad_norm"] > 0 and c["model_grads"] == 0, "gradients"
assert c["weights_unchanged"], "model weights changed during the DAS fit"
print("check ok; layers", c["layers"], "| seconds per 64 pairs", c["seconds_per_64"], "| peak GB",
      c.get("peak_gpu_gb_das_fit"), "| estimated GPU hours per model:", c["gpu_hours_estimate_default"],
      "| jointly correct", c["jointly_correct"], "| bf16 exchange fidelity", c["exchange_fidelity"])
low = {L: v for L, v in c["exchange_fidelity"].items() if v and (v["norm_ratio"] < .95 or v["cos_median"] < .99)}
assert not low or "--dtype fp32" in " ".join(sys.argv[2:]), \
    ("bf16 rounds the rank-1 exchange (prereg_M K&K addendum): resubmit with XARGS=\"--dtype fp32\"", low)
EOF
fi
smoke_layer="${layers%%,*}"
[ "$smoke_layer" = auto ] && smoke_layer=8
$py "$repo/scripts/kk_audit.py" --model "${first#*=}" --name smoke --out "$sm/run" --max-pairs 200 --all-pairs \
  --layers "$smoke_layer" --pairs "$pairs" --bs "${BS:-8}" --k "${K:-1}" --epochs 1 --max-train 24 --max-test 24 --max-test-mid 24 \
  --min-pairs 8 --min-slot-pairs 2 ${CONDS:+--conds "$CONDS"} > "$sm/run.log" 2>&1 \
  || { echo "SMOKE FAILED (run)"; tail -n 30 "$sm/run.log"; exit 1; }
[ -n "${DRY_RUN:-}" ] || grep -q '^KKAUDIT .*"layer"' "$sm/run.log" || { echo "SMOKE FAILED (no KKAUDIT line)"; tail -n 30 "$sm/run.log"; exit 1; }
echo "smoke ok $(date +%T)"
[ -n "${SMOKE_ONLY:-}" ] && { echo "SMOKE_ONLY: stopping"; exit 0; }

failed=
for m in $MODELS; do
  name="${m%%=*}"; dir="${m#*=}"
  echo "=== $name ($dir) $(date +%T)"
  rm -f "$outroot/$name/summary.jsonl"
  $py "$repo/scripts/kk_audit.py" --model "$dir" --name "$name" --out "$outroot/$name" --layers "$layers" \
    "${common[@]}" ${CONDS:+--conds "$CONDS"} ${XARGS:-} || { echo "audit failed $name"; failed="$failed $name"; continue; }
  [ -n "${DRY_RUN:-}" ] || cat "$outroot/$name/summary.jsonl" >> "$outroot/summary.jsonl"
  echo "done $name $(date +%T)"
done
[ -z "$failed" ] || { echo "FAILED:$failed $(date +%T)"; exit 1; }
echo "all done $(date +%T)"
