#!/usr/bin/env bash
# Pushes one finished job's outputs (no weights) to the private GitHub repo tree-forcing-lab/cot-distillation-jev-results.
# Called by run_job.sh after every job; safe to rerun by hand: bash condor/push_results.sh <env file>
# Needs the write deploy key ~/.ssh/cotd_results (README step 1b). Several jobs may finish at once: a lock serialises
# the pushes and every push rebases on the remote first.
set -uo pipefail
ROOT=$HOME/cotd; cd "$ROOT" || exit 1
ENVF=$1; NAME=$(basename "$ENVF" .env)
( set -a; . "$ENVF"; set +a
  echo "runs/${ARMS}${TAG:+-$TAG}-$(basename "$MODEL")-$(basename "$DATA")-s${SEED}" ) > "logs/.$NAME.outdir" 2>/dev/null
OUTD=$(cat "logs/.$NAME.outdir"); rm -f "logs/.$NAME.outdir"
KEY=$HOME/.ssh/cotd_results
[ -f "$KEY" ] || { echo "push_results: no deploy key $KEY, results stay in $ROOT/$OUTD"; exit 0; }
REPO_DIR=$ROOT/results_repo
URLS=("git@github.com:tree-forcing-lab/cot-distillation-jev-results.git"
      "ssh://git@ssh.github.com:443/tree-forcing-lab/cot-distillation-jev-results.git")   # port 443 if 22 is blocked
[ -n "${RESULTS_URL:-}" ] && URLS=("$RESULTS_URL")                               # tests: a local bare repo
export GIT_SSH_COMMAND="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30"
exec 8> "$ROOT/.results_push.lock"; flock -w 1800 8 || { echo "push_results: lock timeout"; exit 0; }
if [ ! -d "$REPO_DIR/.git" ]; then
  for u in "${URLS[@]}"; do git clone -q "$u" "$REPO_DIR" 2>/dev/null && break; done
  [ -d "$REPO_DIR/.git" ] || { echo "push_results: clone failed (network / key)"; exit 0; }
fi
cd "$REPO_DIR"
git config user.name "condor-${USER:-friend}"; git config user.email "condor@users.noreply.github.com"
mkdir -p "$OUTD" logs
if [ -d "$ROOT/$OUTD" ]; then
  find "$ROOT/$OUTD" -maxdepth 1 -type f \( -name 'metrics.json' -o -name 'preds_*.jsonl' -o -name 'trajectory.json' \
    -o -name '*.log' \) -size -90M -exec cp {} "$OUTD/" \;
fi
for f in "$ROOT/logs/$NAME.out" "$ROOT/logs/$NAME-smoke.out"; do [ -f "$f" ] && cp "$f" logs/; done
git add -A "$OUTD" logs
git diff --cached --quiet && { echo "push_results: nothing new for $NAME"; exit 0; }
st="unfinished"; [ -f "$ROOT/$OUTD/metrics.json" ] && st="done"
git commit -q -m "$NAME: $st ($(hostname), $(date '+%F %T'))"
for i in 1 2 3 4 5; do
  for u in "${URLS[@]}"; do
    git remote set-url origin "$u"
    git pull -q --rebase origin main 2>/dev/null && git push -q origin HEAD:main 2>/dev/null && { echo "push_results: pushed $NAME ($st)"; exit 0; }
  done
  sleep $((30 * i))
done
echo "push_results: push failed after retries; the commit stays in $REPO_DIR (rerun this script later)"
