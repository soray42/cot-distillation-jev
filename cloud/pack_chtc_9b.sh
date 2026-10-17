#!/usr/bin/env bash
# Build the CHTC batch-4 bundle (Qwen3.5-9B scale ladder; cloud/chtc/README_9b.md): cotd_code_data_9b.tgz (repo at PIN,
# the data of NEED, the env files of JOBS as condor/jobs9b/, CODE_PIN) plus the new CHTC files of batch 4, to be
# unpacked over the friend's ~/cotd_chtc (batch 3 already there: cotd_4b.sif in /staging, chtc_settings.inc,
# post_push.sh; nothing of batch 3 is replaced: run_train.sh ships byte-identical, checked below).
#   cloud/pack_chtc_9b.sh [PIN]   -> cloud/cotd_chtc_9b_bundle.tgz
# PIN defaults to 60bf5a1 (HEAD of 2026-10-17 11:00; scripts/ hpc/ src/ equal batch 3's 5a0460b except the new
# scripts/filter_tree_steps.py, which no batch-4 job uses). Checks before packing (pack_chtc_4b.sh's, plus): the md5s of
# NEED; every env file's DATA / EXTRA_EVALS / LOCAL_EVAL and the evals train_arm.sbatch always runs are in NEED; MODEL is
# Qwen3.5-9B; NGPU=2 with FSDP_OPTIM=cpu-adamw; SAVE is off; EVALMID starts with val and names only evals of the job;
# the JOBKEY says 9b; the GPU column of the joblist equals NGPU; PIN's train_arm.sbatch has every arm; run_train.sh is
# the batch-3 file (md5 RT_MD5).
set -euo pipefail
cd "$(dirname "$0")/.."
PIN=${1:-60bf5a1}; NEED=${NEED:-cloud/condor/needed_data_9b.json}; JOBS=${JOBS:-cloud/condor/jobs9b}
LIST=${LIST:-cloud/condor/joblist_9b.txt}; OUT=${OUT:-cloud/cotd_chtc_9b_bundle.tgz}
RT_MD5=${RT_MD5:-e61eff1e7070e57e092efa4335c49fa1}   # cloud/chtc/run_train.sh as shipped with batch 3 (and the 12B batch)
FULL=$(git rev-parse --verify "$PIN^{commit}")
[ "$(md5sum < cloud/chtc/run_train.sh | cut -c1-32)" = "$RT_MD5" ] || { echo "pack_chtc_9b: cloud/chtc/run_train.sh is not the batch-3 file"; exit 1; }
python3 - "$NEED" "$JOBS" "$LIST" <<'PY'
import hashlib, json, os, subprocess, sys
need, jobs, lst = sys.argv[1:4]
nd = json.load(open(need)); dirs, files = set(nd["dirs"]), set(nd["files"])
bad = [f"md5 {p}" for p, m in nd["md5"].items() if hashlib.md5(open(p, "rb").read()).hexdigest() != m]
bad += [f"no md5 for {d}/{s}.jsonl" for d in dirs for s in ("train", "val") if f"{d}/{s}.jsonl" not in nd["md5"]]
bad += [f"no md5 for {f}" for f in files if f not in nd["md5"]]
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
    sh = (f'set -a; . "{jobs}/{e}"; printf "%s\\n" "$DATA" "$NGPU" "$MODEL" "$LOCAL_EVAL" "$ARMS" "$EXTRA_EVALS" '
          f'"$FSDP_OPTIM" "$SAVE" "$EVALMID" "$JOBKEY"')
    out = subprocess.run(["bash", "-c", sh], capture_output=True, text=True, check=True).stdout.split("\n")[:10]
    data, ngpu, model, le, arms, extra, fo, save, mid, key = out
    if data not in dirs: bad.append(f"{e}: DATA {data} not in {need}")
    if e not in gpus: bad.append(f"{e} not in {lst}")
    elif gpus[e] != ngpu: bad.append(f"{e}: {lst} asks {gpus[e]} GPU(s), NGPU={ngpu}")
    if os.path.basename(model) != "Qwen3.5-9B": bad.append(f"{e}: MODEL {model}")
    if ngpu != "2" or fo != "cpu-adamw": bad.append(f"{e}: NGPU={ngpu} FSDP_OPTIM={fo} (batch 4: 2 GPUs, cpu-adamw)")
    if save: bad.append(f"{e}: SAVE={save} (batch 4 writes no weights)")
    if not key.startswith("9b "): bad.append(f"{e}: JOBKEY {key}")
    if le not in ("", "0") and le not in files: bad.append(f"{e}: LOCAL_EVAL {le} not in {need}")
    for kv in extra.split():
        if kv.split("=", 1)[1] not in files: bad.append(f"{e}: eval {kv} not in {need}")
    names = {"val", "kk", "jl", "jevbench", "td", "bbeh"} | {kv.split("=", 1)[0] for kv in extra.split()}
    mids = mid.split(",")
    if mids[0] != "val": bad.append(f"{e}: EVALMID {mid} does not start with val")
    bad += [f"{e}: EVALMID names {m}, not an eval of the job" for m in mids if m not in names]
    print(f"{e}: ARMS={arms} DATA={data} NGPU={ngpu} evals={len(extra.split())} mid={mid} ok")
if bad:
    sys.exit("pack_chtc_9b: " + "; ".join(bad))
PY
for a in $(for e in "$JOBS"/*.env; do bash -c "set -a; . '$e'; echo \$ARMS"; done | tr ' ' '\n' | sort -u); do
  git show "$FULL:hpc/train_arm.sbatch" | grep -q "^    $a)" || { echo "pack_chtc_9b: $PIN has no arm $a"; exit 1; }
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
mkdir -p "$C/condor" && cp -r "$JOBS" "$C/condor/jobs9b"
tar czf "$B/cotd_code_data_9b.tgz" -C "$C" .
cp "$LIST" "$B/joblist_9b.txt"
cp cloud/chtc/{README_9b.md,build_9b.sub,fetch_model_9b.sh,train_9b.sub,run_train.sh,make_dags_9b.sh} "$B/"
( cd "$B" && bash make_dags_9b.sh >/dev/null && find . -type f ! -name MANIFEST_9b.md5 | sort | xargs md5sum > MANIFEST_9b.md5 )
tar czf "$OUT" -C "$B" .
rm -rf "$T"
echo "code pin $FULL"
ls -la "$OUT"
