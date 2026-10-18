#!/usr/bin/env bash
# Build the CHTC batch-6 bundle (cloud/chtc/README_6.md), Qwen3.5-4B only, to be unpacked over the friend's ~/cotd_chtc;
# only new file names, so nothing of batches 1-5 is replaced (chtc_settings.inc and post_push.sh stay his):
#   cloud/pack_chtc_6.sh    -> cloud/cotd_chtc_6_4b_bundle.tgz   (code = HEAD)
# As cloud/pack_chtc_5.sh 4b (which stays as it is): the bundle holds cotd_code_data_6_4b.tgz (repo at PIN via git
# archive, so uncommitted working-tree changes never enter it; the data of NEED; the env files as condor/jobs6_4b/;
# CODE_PIN), joblist_6_4b.txt, train6_4b.sub, run_train5.sh (batch 5's executable, unchanged), make_dags_6.sh with its
# outputs jobs6_4b.dag / smoke6_4b.dag, README_6.md, MANIFEST_6_4b.md5. Checks before packing (pack_chtc_5's, adapted):
# md5s of NEED; every env file's DATA / EXTRA_EVALS / LOCAL_EVAL and the five evals train_arm.sbatch always runs are in
# NEED; MODEL Qwen3.5-4B; NGPU=2, FSDP_OPTIM=cpu-adamw, PREC=bf16 in the env, SAVE off; the joblist GPU column 1 (the
# friend's fsdp1 patch runs NGPU=2 envs on one GPU, README_5) or equal to NGPU; EVALMID starts with val and names only
# evals of the job; SI arms (--si-stages): SISTAGES set, EPOCHS = SISTAGES + 1, EVALAT empty (stage-end evaluations);
# other arms: SISTAGES empty, EVALAT points inside the run; value-node arms (--node-format) train on data with
# value_nodes, SI arms on data with worked_steps (within the 16 lines the smoke stage takes); TAG = c<EPOCHS>lr<n>ch with
# LR = <n>e-6; the job key names the model; run dirs distinct, none already done on CHTC (RUNS/runs/<dir>/metrics.json)
# and none shared with an env file of an earlier batch (cloud/condor/jobs*/: batch 5 is still in flight); PIN's
# train_arm.sbatch has every arm.
set -euo pipefail
cd "$(dirname "$0")/.."
M=4b; PIN=${PIN:-HEAD}; NEED=${NEED:-cloud/condor/needed_data_6_4b.json}; MODEL=Qwen3.5-4B; SFX=ch
JOBS=cloud/condor/jobs6_$M; LIST=cloud/condor/joblist_6_$M.txt; OUT=${OUT:-cloud/cotd_chtc_6_${M}_bundle.tgz}
RUNS=${RUNS:-results/chtc_repo}
FULL=$(git rev-parse --verify "$PIN^{commit}")
dirty=$(git status --porcelain --untracked-files=no | awk '{print $2}' | tr '\n' ' ')
[ -z "$dirty" ] || echo "pack_chtc_6: not packed (working tree only, git archive takes $FULL): $dirty"
for f in cloud/chtc/run_train5.sh cloud/chtc/train6_4b.sub cloud/chtc/make_dags_6.sh cloud/chtc/README_6.md; do
  [ -f "$f" ] || { echo "pack_chtc_6: no $f"; exit 1; }
done
git diff --quiet "$FULL" -- cloud/chtc/run_train5.sh || { echo "pack_chtc_6: cloud/chtc/run_train5.sh differs from $PIN"; exit 1; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT; C=$T/code; B=$T/bundle; mkdir -p "$C" "$B"
git show "$FULL:hpc/train_arm.sbatch" > "$T/train_arm.sbatch"
python3 - "$M" "$NEED" "$JOBS" "$LIST" "$MODEL" "$SFX" "$RUNS" "$T/train_arm.sbatch" <<'PY'
import glob, hashlib, json, os, re, subprocess, sys
m, need, jobs, lst, model, sfx, runs, sbatch = sys.argv[1:9]
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
nd = json.load(open(need)); dirs, files = set(nd["dirs"]), set(nd["files"])
bad = []
bad += [f"md5 {p}" for p, h in nd["md5"].items() if md5(p) != h]
bad += [f"no md5 for {d}/{s}.jsonl" for d in dirs for s in ("train", "val") if f"{d}/{s}.jsonl" not in nd["md5"]]
bad += [f"no md5 for {f}" for f in files if f not in nd["md5"]]
always = ["data/eval/kk_heldout.jsonl", "data/eval/jl_heldout.jsonl", "data/eval/jevbench_public.jsonl",
          "data/eval/typed_decisions_test.jsonl", "data/eval/bbeh.jsonl"]          # train_arm.sbatch EVALS (+ smoke)
bad += [f"{f} (train_arm.sbatch) not in {need}" for f in always if f not in files]
dx = {}                                     # arm -> its DX in PIN's train_arm.sbatch
for line in open(sbatch):
    a = re.match(r"^    ([A-Za-z0-9]+)\) .*DX=\"([^\"]*)\"", line)
    if a: dx[a.group(1)] = a.group(2)
gpus = {}
for line in open(lst):
    if line.strip():
        env, prio, ng = [x.strip() for x in line.split(",")]
        gpus[os.path.basename(env)] = ng
envs = sorted(os.listdir(jobs))
bad += [f"{e} in {lst} but not in {jobs}" for e in gpus if e not in envs]
keys = "DATA NGPU MODEL LOCAL_EVAL ARMS EXTRA_EVALS FSDP_OPTIM SAVE EVALMID JOBKEY PREC TAG EPOCHS LR EVALAT SEED SISTAGES".split()
def read(path):
    sh = f'set -a; . "{path}"; printf "%s\\n" ' + " ".join(f'"${k}"' for k in keys)
    return dict(zip(keys, subprocess.run(["bash", "-c", sh], capture_output=True, text=True, check=True).stdout.split("\n")))
def rundir(v): return f"runs/{v['ARMS']}-{v['TAG']}-{os.path.basename(v['MODEL'])}-{os.path.basename(v['DATA'])}-s{v['SEED']}"
older = {}                                  # run dirs of every earlier batch's env files (batch 5 is still running)
for p in glob.glob("cloud/condor/jobs*/*.env"):
    if os.path.dirname(p) != jobs:
        older[rundir(read(p))] = p
def has_field(data, field):
    with open(f"{data}/train.jsonl") as f:
        return any(f'"{field}"' in next(f, "") for _ in range(16))
outs = {}
for e in envs:
    v = read(f"{jobs}/{e}")
    if v["DATA"] not in dirs: bad.append(f"{e}: DATA {v['DATA']} not in {need}")
    if e not in gpus: bad.append(f"{e} not in {lst}")
    elif gpus[e] not in ("1", v["NGPU"]): bad.append(f"{e}: {lst} asks {gpus[e]} GPU(s), NGPU={v['NGPU']}")
    if os.path.basename(v["MODEL"]) != model: bad.append(f"{e}: MODEL {v['MODEL']}")
    if (v["NGPU"], v["FSDP_OPTIM"], v["PREC"]) != ("2", "cpu-adamw", "bf16"): bad.append(f"{e}: NGPU={v['NGPU']} FSDP_OPTIM={v['FSDP_OPTIM']} PREC={v['PREC']}")
    if v["SAVE"]: bad.append(f"{e}: SAVE={v['SAVE']} (batch 6 writes no weights)")
    if not v["JOBKEY"].startswith(m + " "): bad.append(f"{e}: JOBKEY {v['JOBKEY']}")
    if v["LOCAL_EVAL"] not in ("", "0") and v["LOCAL_EVAL"] not in files: bad.append(f"{e}: LOCAL_EVAL {v['LOCAL_EVAL']} not in {need}")
    for kv in v["EXTRA_EVALS"].split():
        if kv.split("=", 1)[1] not in files: bad.append(f"{e}: eval {kv} not in {need}")
    names = {"val", "kk", "jl", "jevbench", "td", "bbeh"} | {kv.split("=", 1)[0] for kv in v["EXTRA_EVALS"].split()}
    mids = v["EVALMID"].split(",")
    if mids[0] != "val": bad.append(f"{e}: EVALMID {v['EVALMID']} does not start with val")
    bad += [f"{e}: EVALMID names {x}, not an eval of the job" for x in mids if x not in names]
    t = re.fullmatch(r"c(\d+)lr(\d+)" + sfx, v["TAG"])
    if not t or t.group(1) != v["EPOCHS"] or float(v["LR"]) != float(t.group(2)) * 1e-6:
        bad.append(f"{e}: TAG {v['TAG']} vs EPOCHS {v['EPOCHS']} LR {v['LR']}")
    arms = v["ARMS"].split()
    for a in arms:
        if a not in dx and a != "A2": bad.append(f"{e}: arm {a} has no DX in train_arm.sbatch at the pin")
    si = [a for a in arms if "--si-stages" in dx.get(a, "")]
    if si:
        if len(si) != len(arms): bad.append(f"{e}: SI and non-SI arms in one job ({v['ARMS']})")
        if not v["SISTAGES"].isdigit() or int(v["EPOCHS"]) != int(v["SISTAGES"]) + 1:
            bad.append(f"{e}: SI arm with SISTAGES={v['SISTAGES']!r} EPOCHS={v['EPOCHS']} (want EPOCHS = SISTAGES + 1)")
        if v["EVALAT"]: bad.append(f"{e}: SI arm with EVALAT {v['EVALAT']} (SI evaluates at its stage ends)")
        if not has_field(v["DATA"], "worked_steps"): bad.append(f"{e}: {v['DATA']} has no worked_steps in its first 16 lines")
        at = "stage ends"
    else:
        if v["SISTAGES"]: bad.append(f"{e}: SISTAGES={v['SISTAGES']} in a non-SI job")
        pts = [x for x in v["EVALAT"].split(",") if x]
        if not pts or not all(0 < float(x) < float(v["EPOCHS"]) for x in pts): bad.append(f"{e}: EVALAT {v['EVALAT']}")
        at = v["EVALAT"]
    if any("--node-format" in dx.get(a, "") for a in arms) and not has_field(v["DATA"], "value_nodes"):
        bad.append(f"{e}: {v['DATA']} has no value_nodes in its first 16 lines")
    od = rundir(v)
    if od in outs: bad.append(f"{e}: run dir {od} also from {outs[od]}")
    outs[od] = e
    if od in older: bad.append(f"{e}: run dir {od} also from {older[od]}")
    if os.path.isfile(f"{runs}/{od}/metrics.json"): bad.append(f"{e}: {od} is already done on CHTC")
    print(f"{e}: {od} NGPU={v['NGPU']} (joblist {gpus.get(e)}) evals={len(v['EXTRA_EVALS'].split())} mid={v['EVALMID']} at={at}"
          f"{' SISTAGES=' + v['SISTAGES'] if si else ''} ok")
print(f"{len(older)} run dirs of earlier batches' env files checked")
if not os.path.isdir(f"{runs}/runs"): print(f"pack_chtc_6: warning: no {runs}/runs, the done-on-CHTC check was skipped")
if bad:
    sys.exit("pack_chtc_6: " + "; ".join(bad))
PY
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
mkdir -p "$C/condor"
cp -r "$JOBS" "$C/condor/jobs6_$M"
tar czf "$B/cotd_code_data_6_$M.tgz" -C "$C" .
cp "$LIST" "$B/joblist_6_$M.txt"
cp cloud/chtc/{README_6.md,train6_$M.sub,run_train5.sh,make_dags_6.sh} "$B/"
( cd "$B" && bash make_dags_6.sh >/dev/null && find . -type f ! -name "MANIFEST_6_$M.md5" | sort | xargs md5sum > "MANIFEST_6_$M.md5" )
tar czf "$OUT" -C "$B" .
echo "code pin $FULL"
ls -la "$OUT"; md5sum "$OUT"
