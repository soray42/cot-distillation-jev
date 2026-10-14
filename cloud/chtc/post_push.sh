#!/bin/bash
# DAG POST script, on the submit server after each job:  post_push.sh <name> <job exit code>
# Unpacks results/results_<name>.tgz and pushes it to the private repo soray42/cot-distillation-jev-results with the
# write deploy key ~/.ssh/cotd_results (README step 2). Never blocks: without the key the files stay in results/.
# Returns the job's exit code, so a failed job is retried by the DAG.
NAME=$1; RC=${2:-0}
cd "$(dirname "$0")" || exit "$RC"
F=results/results_$NAME.tgz; KEY=$HOME/.ssh/cotd_results
[ -f "$F" ] && [ -f "$KEY" ] || exit "$RC"
export GIT_SSH_COMMAND="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30"
URLS=("git@github.com:soray42/cot-distillation-jev-results.git"
      "ssh://git@ssh.github.com:443/soray42/cot-distillation-jev-results.git")
[ -n "${RESULTS_URL:-}" ] && URLS=("$RESULTS_URL")
exec 8> .results_push.lock; flock -w 1800 8 || exit "$RC"
if [ ! -d results_repo/.git ]; then
  for u in "${URLS[@]}"; do git clone -q "$u" results_repo 2>/dev/null && break; done
fi
[ -d results_repo/.git ] || exit "$RC"
tar xzf "$F" -C results_repo || exit "$RC"
cd results_repo
git config user.name "chtc-${USER:-friend}"; git config user.email "chtc@users.noreply.github.com"
git add -A runs logs 2>/dev/null
git diff --cached --quiet && exit "$RC"
git commit -q -m "$NAME: exit $RC ($(date '+%F %T'))"
for i in 1 2 3 4 5; do
  for u in "${URLS[@]}"; do
    git remote set-url origin "$u"
    git pull -q --rebase origin main 2>/dev/null && git push -q origin HEAD:main 2>/dev/null && exit "$RC"
  done
  sleep $((30 * i))
done
exit "$RC"
