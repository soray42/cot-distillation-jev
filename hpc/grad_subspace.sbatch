#!/bin/bash
#SBATCH --job-name=cotd-gradsub
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# One-step gradient subspaces vs the finished update (scripts/grad_subspace.py; LoRA-One link; notes/next_round/
# tree_forcing/grad_subspace.md). Nothing is trained or saved but the JSON / markdown results.
# On the base Qwen3.5-2B-Base: the full gradients of A2 (finals), GFPS and TFL (finals + node views) and of their node
# views alone, on a fixed batch of N problems of $DATA (seed SEED), compared with each other and with the saved 2B
# fine-tunes of this account (dW = W_ft - W_base):
#   ACCT=A: GFPS-c4 and TFL-c4 (runs/{GFPS,TFL}-c4-Qwen3.5-2B-Base-student_v3sl-s12/model)
#   ACCT=B: A2-c12 (runs/A2-c12-Qwen3.5-2B-Base-student_v3sl-s12/model)
# Both accounts compute the same gradients (same base, data, batch and seed: the merged report checks their norms), so
# every G x dW cell exists; then merge the two accounts' JSONs locally (see the last line of the log).
# From ~/cotd, login node first (Python only echoed; checks the account, the files and the shell logic):
#   ACCT=A DRY_RUN=1 bash repo/hpc/grad_subspace.sbatch      then      ACCT=A sbatch repo/hpc/grad_subspace.sbatch
#   (ACCT=B likewise on account B)
# Knobs: BATCHES (default "tree all": the batch drawn from the K&K items only, then from all of $DATA, the trainer's
# mixture), N (256), SEED (0), NODE_ARMS (GFPS,TFL), DATA (data/student_v3sl; numgen data: TREE_SOURCES=numgen),
# TREE_SOURCES (knights_knaves; numgen for *numgen* data), RUNSEED (12: the seed in the run directories), FT (override:
# "NAME=model_dir ..."), BASE (model dir), OUTROOT (runs/gradsub), XARGS (more grad_subspace.py flags, e.g. --layers,
# --modules, --ranks), REPO (code checkout, default repo; its commit goes into the log and the JSON), CC1D (1: the
# causal-conv1d kernel as in training), SMOKE_ONLY, DRY_RUN, AUSER (account A's user, 3343816).
# Smoke stage first: grad_subspace.py --check on 4 K&K problems and the first selected layer (every finetuned model
# loaded; the trainer-layout identity G_arm = G_A2 / 3 + G_arm_nodes within 2e-2 under bf16 autocast; non-zero finite
# gradients; prints the measured time estimate of the full passes); the job exits if it fails.
# Time (2B, estimate): tree batch about 20 min per node arm + 1 min of finals, all-sources batch about 3 min per arm,
# SVDs and finetuned loads a few minutes: about 1 h per account with the defaults. Memory: about 15-17 GB GPU peak,
# under 12 GB host (the script's docstring has the arithmetic).
# Outputs: $OUTROOT/<data>-<batch>-n<N>-s<SEED>-<finetuned names>/grad_subspace.{json,md}; done directories are skipped
# (resubmit-safe). Relay scripts: ACCT BATCHES N SEED NODE_ARMS DATA TREE_SOURCES RUNSEED FT BASE OUTROOT XARGS REPO CC1D
# SMOKE_ONLY DRY_RUN AUSER belong in the reset list (internal variables are lower case).
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
[ -f "$repo/scripts/grad_subspace.py" ] || { echo "missing $repo/scripts/grad_subspace.py"; exit 1; }
commit=$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo unknown)
[ "$commit" = unknown ] || [ -z "$(git -C "$repo" status --porcelain --untracked-files=no 2>/dev/null)" ] || commit="$commit-dirty"
echo "code: $repo @ $commit"

acct="${ACCT:-}"; auser="${AUSER:-3343816}"; me=$(id -un)
base="${BASE:-$HOME/cotd/models/Qwen3.5-2B-Base}"
data="${DATA:-data/student_v3sl}"
runs="$(basename "$base")-$(basename "$data")-s${RUNSEED:-12}"
case "$acct" in
  A) [ "$me" = "$auser" ] || { echo "ACCT=A but this is $me (account A is $auser)"; exit 1; }
     ft="GFPS=runs/GFPS-c4-$runs/model TFL=runs/TFL-c4-$runs/model" ;;
  B) [ "$me" != "$auser" ] || { echo "ACCT=B but this is account A ($me)"; exit 1; }
     ft="A2=runs/A2-c12-$runs/model" ;;
  *) [ -n "${FT:-}" ] || { echo "set ACCT=A or ACCT=B (or FT=\"NAME=model_dir ...\")"; exit 1; }
     ft= ;;
esac
ft="${FT:-$ft}"
case "$data" in *numgen*) src_default=numgen ;; *) src_default=knights_knaves ;; esac
src="${TREE_SOURCES:-$src_default}"
arms="${NODE_ARMS:-GFPS,TFL}"; n="${N:-256}"; seed="${SEED:-0}"; batches="${BATCHES:-tree all}"
outroot="${OUTROOT:-runs/gradsub}"
tag=$(for kv in $ft; do printf '%s+' "${kv%%=*}"; done); tag="${tag%+}"
[ -f "$base/config.json" ] || { echo "missing base model $base"; exit 1; }
[ -f "$data/train.jsonl" ] || { echo "missing $data/train.jsonl"; exit 1; }
for kv in $ft; do
  [ -f "${kv#*=}/config.json" ] || { echo "missing finetuned model ${kv#*=} (${kv%%=*})"; exit 1; }
done
for b in $batches; do case "$b" in tree|all) ;; *) echo "BATCHES: tree and/or all (got $b)"; exit 1 ;; esac; done
echo "data: $data/train.jsonl md5 $(md5sum < "$data/train.jsonl" | cut -c1-32) (v3sl registered: 53f0240ffbe379a38f5d0e0e68872e4e)"
echo "base $base | finetuned $ft | node arms $arms | tree sources $src | batches $batches | N $n | seed $seed"
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi here"
common=(--model "$base" --train "$data/train.jsonl" --tree-sources "$src" --node-arms "$arms" --n "$n" --seed "$seed")
mkdir -p "$outroot" logs

echo "=== smoke stage $(date +%T)"
sm="$outroot/_smoke-${acct:-x}"
rm -rf "$sm"
# shellcheck disable=SC2086
$py "$repo/scripts/grad_subspace.py" "${common[@]}" --check --check-n 4 --finetuned $ft ${XARGS:-} --out "$sm" \
  || { echo "SMOKE FAILED"; exit 1; }
[ -n "${DRY_RUN:-}" ] || [ -f "$sm/check.json" ] || { echo "SMOKE FAILED (no check.json)"; exit 1; }
echo "smoke ok $(date +%T)"
[ -n "${SMOKE_ONLY:-}" ] && { echo "SMOKE_ONLY: stopping"; exit 0; }

failed=
for b in $batches; do
  out="$outroot/$(basename "$data")-$b-n$n-s$seed-$tag"
  if [ -f "$out/grad_subspace.json" ]; then echo "done already: $out"; continue; fi
  echo "=== $b -> $out $(date +%T)"
  # shellcheck disable=SC2086
  $py "$repo/scripts/grad_subspace.py" "${common[@]}" --batch-sources "$b" --finetuned $ft ${XARGS:-} --out "$out" \
    || { echo "FAILED $b"; failed="$failed $b"; continue; }
  echo "done $out $(date +%T)"
done
[ -z "$failed" ] || { echo "FAILED:$failed $(date +%T)"; exit 1; }
echo "all done $(date +%T)"
echo "merge the accounts (after hpc/fetch_results.sh): python scripts/grad_subspace.py --report" \
     "results/runs/gradsub/$(basename "$data")-<batch>-n$n-s$seed-*/grad_subspace.json --out" \
     "results/runs/gradsub/$(basename "$data")-<batch>-n$n-s$seed"
