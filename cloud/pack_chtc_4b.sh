#!/usr/bin/env bash
# Build the CHTC batch-3 bundle (Qwen3.5-4B; cloud/chtc/README_4b.md): cotd_code_data_4b.tgz (repo at PIN, the data of
# NEED, the env files of JOBS as condor/jobs4b/, CODE_PIN) plus the new / changed CHTC files, to be unpacked over the
# friend's existing ~/cotd_chtc (it carries no chtc_settings.inc or post_push.sh, so those stay as they are).
#   cloud/pack_chtc_4b.sh [PIN]   -> cloud/cotd_chtc_4b_bundle.tgz
# PIN defaults to 5a0460b (HEAD of 2026-10-17: MuSiQue source in treeplan, arms A7 / GFPQX). Checks before packing:
# the md5s of NEED, every env file's DATA / EXTRA_EVALS / LOCAL_EVAL / MODEL and the evals train_arm.sbatch always runs
# are in NEED, the GPU column of the joblist equals each env file's NGPU, and PIN's train_arm.sbatch has every arm.
set -euo pipefail
cd "$(dirname "$0")/.."
PIN=${1:-5a0460b}; NEED=${NEED:-cloud/condor/needed_data_4b.json}; JOBS=${JOBS:-cloud/condor/jobs4b}
LIST=${LIST:-cloud/condor/joblist_4b.txt}; OUT=${OUT:-cloud/cotd_chtc_4b_bundle.tgz}
FULL=$(git rev-parse --verify "$PIN^{commit}")
python3 - "$NEED" "$JOBS" "$LIST" <<'PY'
import hashlib, json, os, subprocess, sys
need, jobs, lst = sys.argv[1:4]
nd = json.load(open(need)); dirs, files = set(nd["dirs"]), set(nd["files"])
bad = [f"md5 {p}" for p, m in nd["md5"].items() if hashlib.md5(open(p, "rb").read()).hexdigest() != m]
bad += [f"no md5 for {d}/{s}.jsonl" for d in dirs for s in ("train", "val") if f"{d}/{s}.jsonl" not in nd["md5"]]
always = ["data/eval/kk_heldout.jsonl", "data/eval/jl_heldout.jsonl", "data/eval/jevbench_public.jsonl",
          "data/eval/typed_decisions_test.jsonl", "data/eval/bbeh.jsonl"]          # train_arm.sbatch EVALS (+ smoke)
bad += [f"{f} (train_arm.sbatch) not in {need}" for f in always if f not in files]
gpus = {}
for line in open(lst):
    if line.strip():
        env, prio, ng = [x.strip() for x in line.split(",")]
        gpus[os.path.basename(env)] = ng
envs = sorted(os.listdir(jobs))
bad += [f"{e} in {lst} but not in {jobs}" for e in gpus if e not in envs]
for e in envs:
    sh = f'set -a; . "{jobs}/{e}"; printf "%s\\n" "$DATA" "$NGPU" "$MODEL" "$LOCAL_EVAL" "$ARMS" "$EXTRA_EVALS"'
    data, ngpu, model, le, arms, extra = subprocess.run(["bash", "-c", sh], capture_output=True, text=True, check=True).stdout.split("\n")[:6]
    if data not in dirs: bad.append(f"{e}: DATA {data} not in {need}")
    if e not in gpus: bad.append(f"{e} not in {lst}")
    elif gpus[e] != ngpu: bad.append(f"{e}: {lst} asks {gpus[e]} GPU(s), NGPU={ngpu}")
    if os.path.basename(model) != "Qwen3.5-4B": bad.append(f"{e}: MODEL {model}")
    if le not in ("", "0") and le not in files: bad.append(f"{e}: LOCAL_EVAL {le} not in {need}")
    for kv in extra.split():
        if kv.split("=", 1)[1] not in files: bad.append(f"{e}: eval {kv} not in {need}")
    print(f"{e}: ARMS={arms} DATA={data} NGPU={ngpu} evals={len(extra.split())} ok")
if bad:
    sys.exit("pack_chtc_4b: " + "; ".join(bad))
PY
for a in $(for e in "$JOBS"/*.env; do bash -c "set -a; . '$e'; echo \$ARMS"; done | tr ' ' '\n' | sort -u); do
  git show "$FULL:hpc/train_arm.sbatch" | grep -q "^    $a)" || { echo "pack_chtc_4b: $PIN has no arm $a"; exit 1; }
done
T=$(mktemp -d); C=$T/code; B=$T/bundle; mkdir -p "$C" "$B"
git archive --format=tar --prefix=repo/ "$FULL" | tar x -C "$C"
echo "$FULL" > "$C/CODE_PIN"
python3 - "$C" "$NEED" <<'PY'
import json, shutil, sys, os
dst, need = sys.argv[1:3]
nd = json.load(open(need))
for d in nd["dirs"]:
    shutil.copytree(d, os.path.join(dst, d))
for f in nd["files"]:
    os.makedirs(os.path.join(dst, os.path.dirname(f)), exist_ok=True)
    shutil.copy2(f, os.path.join(dst, f))
PY
mkdir -p "$C/condor" && cp -r "$JOBS" "$C/condor/jobs4b"
tar czf "$B/cotd_code_data_4b.tgz" -C "$C" .
cp "$LIST" "$B/joblist_4b.txt"
cp cloud/chtc/{README_4b.md,cotd_4b.def,build_4b.sub,fetch_model_4b.sh,train_4b.sub,run_train.sh,make_dags_4b.sh} "$B/"
( cd "$B" && bash make_dags_4b.sh >/dev/null && find . -type f ! -name MANIFEST_4b.md5 | sort | xargs md5sum > MANIFEST_4b.md5 )
tar czf "$OUT" -C "$B" .
rm -rf "$T"
echo "code pin $FULL"
ls -la "$OUT"
