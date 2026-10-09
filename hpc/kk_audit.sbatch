#!/bin/bash
#SBATCH --job-name=cotd-kkaudit
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# K&K causal audit (notes/next_round/causal_audit/kk_design.md; endpoints notes/prereg_M.md K&K addendum and its post-hoc
# amendment of 2026-10-09): scripts/kk_audit.py on each model, one after another.
# From ~/cotd (data/eval/kk_audit_pairs.jsonl pushed with hpc/push_data.sh data/eval; the solver is not needed here):
#   MODELS="TFL2B-s12=runs/TFL-...-s12/model GFPS2B-s12=runs/GFPS-...-s12/model" REPO=<frozen checkout> \
#     sbatch repo/hpc/kk_audit.sbatch
# MODELS: space-separated name=model_dir (names end in -s<seed>).
# PRESET (default v2): v2 = the amended audit; v1 = the registered 2026-10-06 run exactly (layers 4..20 step 4, k 1,
#   3 epochs at lr 1e-3, bf16 autocast, OUTROOT runs/kkaudit), kept for the record.
# v2 defaults (kk_audit.py --preset v2): fp32 forward with autocast and TF32 off (DTYPE; fp32-autocast = v1's "fp32"),
#   layer screen over candidates 13..20 of 24 (17..28 of 32) keeping layers whose full exchange of the readout state
#   copies the source answer on >= .5 of stmt pairs (LAYERS=gate; or a comma list), ranks K=1,4,8, one subspace per
#   slot fitted for >= DAS_STEPS (500) Adam steps at LR 1e-2, early stop on held-out train programs (patience 200,
#   max DAS_STEPS_MAX 1500), positive-control gate stmt:source@last, both-correct and all-pairs test sets, inv retention
#   of the model's own answer, linear probes at layers 16/20 (name, statement, readout per slot) with control tasks.
#   Patches at the readout run as single-token decodes over cached prefixes (exactly the hook-patched forward; checked
#   on the real model in the smoke). OUTROOT default runs/kkaudit_v2 (v1 results stay in runs/kkaudit).
# Other knobs: CONDS, MAXTR/MAXTE (600/600), BS (8, prompt batches), XARGS (more kk_audit.py flags), PAIRS, CC1D (1:
# the causal-conv1d kernel on PYTHONPATH; v2 routes the linear-attention layers to the PyTorch reference kernels anyway).
# Outputs: <OUTROOT>/<name>/{audit.json,summary.jsonl,preds.jsonl}; every model's KKAUDIT lines are appended to
# <OUTROOT>/summary.jsonl. REPO: code checkout the job runs (pin a frozen checkout; its commit, "-dirty" if tracked files
# differ, goes to code_commit.txt).
# Time, v2 (estimate, 40 GB slice, fp32 without TF32; --check prints the measured value and the smoke stage echoes it):
#   2B about 3 h per model (prediction + readout pass over 16.6k prompts ~35 min, prefix forwards ~40 min, 7 conditions
#   x ~6 screened layers x 3 ranks = ~126 DAS fits at ~25-45 s each ~1-1.5 h, evaluations and probes ~15 min; upper
#   bound ~4 h when every fit runs 1500 steps), so two 2B models fit a 12 h job; 4B about 6 h (8 h upper), one per job.
# Smoke stage first (exits on failure), on the first model:
#   1. kk_audit.py --check on 64 pairs per family: identity patches, token sites, gradients only in the subspace and
#      bit-identical weights, fp32 forward (matmul outputs float32, autocast off), decode path = hook-patched full
#      forward at every candidate layer (max |logit diff| < DECODE_TOL, default 1e-2 in fp32), the layer screen keeps a
#      layer, positive-control mini-fit (stmt:source@last, best-copy layer, top rank, 200 steps) lowers its training
#      loss by >= POSCTL_MIN (default 0.10, relative), bf16 exchange fidelity (fails only under DTYPE=bf16, as registered),
#      timings and the GPU-hour estimate;
#   2. a 1-layer, 20-step run on 24/24 pairs that must print KKAUDIT lines including the gate line.
#   SMOKE_ONLY=1 stops after it. If the decode check fails, resubmit with XARGS="--no-prefix-cache" (hook path, the v1
#   mechanism; about 100x slower, so also K=1 and LAYERS=16,20) and record that in the amendment's log.
# Relay scripts: MODELS PRESET LAYERS CONDS K DTYPE LR DAS_STEPS DAS_STEPS_MAX EPOCHS MAXTR MAXTE BS XARGS PAIRS OUTROOT
# REPO CC1D SMOKE_ONLY DRY_RUN DECODE_TOL POSCTL_MIN belong in the reset list (internal variables are lower case).
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
preset="${PRESET:-v2}"
case "$preset" in
  v2) layers="${LAYERS:-gate}"; k="${K:-1,4,8}"; dtype="${DTYPE:-fp32}"; outroot="${OUTROOT:-runs/kkaudit_v2}" ;;
  v1) layers="${LAYERS:-auto}"; k="${K:-1}"; dtype="${DTYPE:-bf16}"; outroot="${OUTROOT:-runs/kkaudit}" ;;
  *) echo "PRESET must be v2 or v1"; exit 1 ;;
esac
set -- ${XARGS:-}                                     # an explicit --dtype in XARGS wins (as in v1 relays)
while [ $# -gt 0 ]; do { [ "$1" = --dtype ] && [ $# -gt 1 ] && dtype=$2; } || true; shift; done
mkdir -p "$outroot"
echo "$commit" > "$outroot/code_commit.txt"
echo "preset $preset layers $layers k $k dtype $dtype outroot $outroot"
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi here"
common=(--preset "$preset" --pairs "$pairs" --bs "${BS:-8}" --k "$k" --dtype "$dtype" --max-train "${MAXTR:-600}"
        --max-test "${MAXTE:-600}" ${LR:+--lr "$LR"})
if [ "$preset" = v2 ]; then
  common+=(--das-steps "${DAS_STEPS:-500}" --das-steps-max "${DAS_STEPS_MAX:-1500}")
else
  common+=(--epochs "${EPOCHS:-3}")
fi

echo "=== smoke stage $(date +%T)"
first="${MODELS%% *}"
sm="$outroot/_smoke"
rm -rf "$sm"; mkdir -p "$sm"
$py "$repo/scripts/kk_audit.py" --model "${first#*=}" --name smoke --out "$sm/check" --check --max-pairs 64 \
  --layers "$layers" "${common[@]}" ${CONDS:+--conds "$CONDS"} ${XARGS:-} \
  || { echo "SMOKE FAILED (check)"; exit 1; }
smoke_layer=8
if [ -z "${DRY_RUN:-}" ]; then
  if [ "$preset" = v2 ]; then
    smoke_layer=$($py - "$sm/check/check.json" "$dtype" "${DECODE_TOL:-1e-2}" "${POSCTL_MIN:-0.10}" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
dtype, tol, pmin = sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
bad = [k for k, v in c.items() if k.startswith("identity_max_abs_diff") and v > 1e-2]
assert not bad, ("identity patch changes the logits", {k: c[k] for k in bad})
assert c["stmt_site_is_quote"] == 1.0 and c["name_site_in_name"] == 1.0, "token sites"
assert c["das_grad_norm"] > 0 and c["model_grads"] == 0, "gradients"
assert c["weights_unchanged"], "model weights changed during the DAS fit"
if dtype == "fp32":
    assert c["matmul_output_dtype"] == "float32" and not c["autocast_bf16"] and not c["tf32_matmul"], \
        ("forward is not fp32", c["matmul_output_dtype"], c["autocast_bf16"], c["tf32_matmul"])
    assert c["decode_max_abs_diff"] < tol, ("decode path differs from the hook-patched forward", c["decode_fidelity"])
assert c["decode_prefix_fidelity"]["argmax_mismatch"] == 0, ("decoded unpatched answers differ", c["decode_prefix_fidelity"])
assert c["layers_selected"], ("layer screen kept no layer", c["layer_screen"])
p = c["posctl_fit"]
assert p["rel_drop"] >= pmin, ("positive-control mini-fit did not lower the loss", {k: p[k] for k in p if k != "curve"})
low = {L: v for L, v in c["exchange_fidelity_bf16"].items() if v and (v["norm_ratio"] < .95 or v["cos_median"] < .99)}
assert not low or dtype != "bf16", ("bf16 rounds the rank-1 exchange (prereg_M K&K addendum): use DTYPE=fp32", low)
print("check ok; candidates", c["candidates"], "| screen", c["layer_screen"].get("copy_source_full"), "| selected",
      c["layers_selected"], "| decode max diff", c["decode_max_abs_diff"], "| posctl", p["first_loss"], "->",
      p["last_loss"], "| bf16 fidelity", c["exchange_fidelity_bf16"], "| cache MB/prompt",
      round(c["cache_bytes_per_prompt"] / 2 ** 20, 1), "| seconds", c["seconds"], "| jointly correct",
      c["jointly_correct"], "| estimated GPU hours per model", c["gpu_hours_estimate"], file=sys.stderr)
print(c["best_copy_layer"])
EOF
) || { echo "SMOKE FAILED (check values)"; exit 1; }
  else
    $py - "$sm/check/check.json" "$dtype" <<'EOF' || { echo "SMOKE FAILED (check values)"; exit 1; }
import json, sys
c = json.load(open(sys.argv[1]))
bad = [k for k, v in c.items() if k.startswith("identity_max_abs_diff") and v > 1e-2]
assert not bad, ("identity patch changes the logits", {k: c[k] for k in bad})
assert c["stmt_site_is_quote"] == 1.0 and c["name_site_in_name"] == 1.0, "token sites"
assert c["das_grad_norm"] > 0 and c["model_grads"] == 0, "gradients"
assert c["weights_unchanged"], "model weights changed during the DAS fit"
print("check ok; layers", c["layers"], "| seconds per 64 pairs", c["seconds_per_64"], "| estimated GPU hours per model:",
      c["gpu_hours_estimate_default"], "| bf16 exchange fidelity", c["exchange_fidelity"])
low = {L: v for L, v in c["exchange_fidelity"].items() if v and (v["norm_ratio"] < .95 or v["cos_median"] < .99)}
assert not low or sys.argv[2] != "bf16", ("bf16 rounds the rank-1 exchange (prereg_M K&K addendum): DTYPE=fp32", low)
EOF
    smoke_layer="${layers%%,*}"
    [ "$smoke_layer" = auto ] && smoke_layer=8
  fi
fi
echo "smoke run at layer $smoke_layer"
if [ "$preset" = v2 ]; then
  smoke_args=(--das-steps 20 --das-steps-max 20 --das-eval-every 10 --probe-layers "$smoke_layer")
else
  smoke_args=(--epochs 1)
fi
$py "$repo/scripts/kk_audit.py" --model "${first#*=}" --name smoke --out "$sm/run" --max-pairs 200 --all-pairs \
  --layers "$smoke_layer" "${common[@]}" --max-train 24 --max-test 24 --max-test-mid 24 --min-pairs 8 \
  --min-slot-pairs 2 "${smoke_args[@]}" ${CONDS:+--conds "$CONDS"} ${XARGS:-} > "$sm/run.log" 2>&1 \
  || { echo "SMOKE FAILED (run)"; tail -n 30 "$sm/run.log"; exit 1; }
if [ -z "${DRY_RUN:-}" ]; then
  grep -q '^KKAUDIT .*"layer"' "$sm/run.log" || { echo "SMOKE FAILED (no KKAUDIT line)"; tail -n 30 "$sm/run.log"; exit 1; }
  if [ "$preset" = v2 ] && [ -z "${CONDS:-}" ]; then
    grep -q '^KKAUDIT .*"cond": "_gate"' "$sm/run.log" || { echo "SMOKE FAILED (no gate line)"; tail -n 30 "$sm/run.log"; exit 1; }
  fi
fi
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
