#!/usr/bin/env bash
# Build a CHTC batch-5 bundle (cloud/chtc/README_5.md), to be unpacked over the friend's ~/cotd_chtc; only new file
# names, so nothing of batches 1-4 is replaced (chtc_settings.inc and post_push.sh stay his):
#   cloud/pack_chtc_5.sh 4b    -> cloud/cotd_chtc_5_4b_bundle.tgz   (Qwen3.5-4B, groups A-D; code = HEAD)
#   cloud/pack_chtc_5.sh 9b    -> cloud/cotd_chtc_5_9b_bundle.tgz   (Qwen3.5-9B, group F; code = HEAD)
#   cloud/pack_chtc_5.sh 12b   -> cloud/cotd_chtc_5_12b_bundle.tgz  (Gemma-4-12B, group E; code = 915dbd6, batches 1-2)
# Each bundle: cotd_code_data_5_<m>.tgz (repo at PIN via git archive, so uncommitted working-tree changes never enter it;
# the data of NEED; the env files as condor/jobs5_<m>/; CODE_PIN), joblist_5_<m>.txt, train5_<m>.sub, run_train5.sh,
# make_dags_5.sh with its outputs jobs5_<m>.dag / smoke5_<m>.dag, README_5.md, MANIFEST_5_<m>.md5.
# 12B keeps the 12B tarball mechanism of cloud/pack_chtc.sh (pin 915dbd6, cloud/condor/needed_data.json, every env file
# of cloud/condor/jobs as condor/jobs/) and adds condor/jobs5_12b/ + CODE_PIN; it is checked member by member against
# the batch-2 tarball REF12 (requests/2026-10-17b): every member there must be here with the same md5, and nothing else
# may be new. Checks before packing (pack_chtc_9b.sh's, plus): md5s of NEED (12B: against REF12); every env file's DATA /
# EXTRA_EVALS / LOCAL_EVAL and the five evals train_arm.sbatch always runs are in NEED; MODEL is the bundle's model;
# NGPU=2, FSDP_OPTIM=cpu-adamw, PREC=bf16, SAVE off; EVALMID starts with val and names only evals of the job; EVALAT
# points lie inside the run; TAG = c<EPOCHS>lr<n><ch|g12> with LR = <n>e-6; the job key names the model; the GPU column
# of the joblist equals NGPU; run dirs are distinct and none is already done on CHTC (RUNS/runs/<dir>/metrics.json);
# PIN's train_arm.sbatch has every arm.
set -euo pipefail
cd "$(dirname "$0")/.."
M=${1:-}; case "$M" in 4b|9b|12b) ;; *) echo "usage: cloud/pack_chtc_5.sh 4b|9b|12b"; exit 1 ;; esac
case "$M" in
  4b)  PIN=${PIN:-HEAD};    NEED=${NEED:-cloud/condor/needed_data_5_4b.json}; MODEL=Qwen3.5-4B;  SFX=ch ;;
  9b)  PIN=${PIN:-HEAD};    NEED=${NEED:-cloud/condor/needed_data_9b.json};   MODEL=Qwen3.5-9B;  SFX=ch ;;
  12b) PIN=${PIN:-915dbd6}; NEED=${NEED:-cloud/condor/needed_data.json};      MODEL=gemma-4-12B; SFX=g12 ;;
esac
JOBS=cloud/condor/jobs5_$M; LIST=cloud/condor/joblist_5_$M.txt; OUT=${OUT:-cloud/cotd_chtc_5_${M}_bundle.tgz}
RUNS=${RUNS:-results/chtc_repo}; REF12=${REF12:-results/chtc_repo/requests/2026-10-17b/cotd_code_data.tgz}
FULL=$(git rev-parse --verify "$PIN^{commit}")
dirty=$(git status --porcelain --untracked-files=no | awk '{print $2}' | tr '\n' ' ')
[ -z "$dirty" ] || echo "pack_chtc_5: not packed (working tree only, git archive takes $FULL): $dirty"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT; C=$T/code; B=$T/bundle; mkdir -p "$C" "$B"
if [ "$M" = 12b ]; then [ -f "$REF12" ] || { echo "pack_chtc_5: no $REF12"; exit 1; }; mkdir -p "$T/ref"; tar xzf "$REF12" -C "$T/ref"; fi
python3 - "$M" "$NEED" "$JOBS" "$LIST" "$MODEL" "$SFX" "$RUNS" "$T/ref" <<'PY'
import hashlib, json, os, re, subprocess, sys
m, need, jobs, lst, model, sfx, runs, ref = sys.argv[1:9]
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
nd = json.load(open(need)); dirs, files = set(nd["dirs"]), set(nd["files"])
bad = []
if "md5" in nd:
    bad += [f"md5 {p}" for p, h in nd["md5"].items() if md5(p) != h]
    bad += [f"no md5 for {d}/{s}.jsonl" for d in dirs for s in ("train", "val") if f"{d}/{s}.jsonl" not in nd["md5"]]
    bad += [f"no md5 for {f}" for f in files if f not in nd["md5"]]
else:   # 12B: needed_data.json has no md5s; the batch-2 tarball is the reference
    for d in dirs:
        for f in sorted(os.listdir(d)):
            if os.path.exists(f"{ref}/{d}/{f}") and md5(f"{ref}/{d}/{f}") != md5(f"{d}/{f}"): bad.append(f"{d}/{f} differs from the batch-2 tarball")
    bad += [f"{f} differs from the batch-2 tarball" for f in files if md5(f"{ref}/{f}") != md5(f)]
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
outs = {}
keys = "DATA NGPU MODEL LOCAL_EVAL ARMS EXTRA_EVALS FSDP_OPTIM SAVE EVALMID JOBKEY PREC TAG EPOCHS LR EVALAT SEED".split()
for e in envs:
    sh = f'set -a; . "{jobs}/{e}"; printf "%s\\n" ' + " ".join(f'"${k}"' for k in keys)
    v = dict(zip(keys, subprocess.run(["bash", "-c", sh], capture_output=True, text=True, check=True).stdout.split("\n")))
    if v["DATA"] not in dirs: bad.append(f"{e}: DATA {v['DATA']} not in {need}")
    if e not in gpus: bad.append(f"{e} not in {lst}")
    elif gpus[e] != v["NGPU"]: bad.append(f"{e}: {lst} asks {gpus[e]} GPU(s), NGPU={v['NGPU']}")
    if os.path.basename(v["MODEL"]) != model: bad.append(f"{e}: MODEL {v['MODEL']}")
    if (v["NGPU"], v["FSDP_OPTIM"], v["PREC"]) != ("2", "cpu-adamw", "bf16"): bad.append(f"{e}: NGPU={v['NGPU']} FSDP_OPTIM={v['FSDP_OPTIM']} PREC={v['PREC']}")
    if v["SAVE"]: bad.append(f"{e}: SAVE={v['SAVE']} (batch 5 writes no weights)")
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
    if not all(0 < float(x) < float(v["EPOCHS"]) for x in v["EVALAT"].split(",")): bad.append(f"{e}: EVALAT {v['EVALAT']}")
    od = f"runs/{v['ARMS']}-{v['TAG']}-{model}-{os.path.basename(v['DATA'])}-s{v['SEED']}"
    if od in outs: bad.append(f"{e}: run dir {od} also from {outs[od]}")
    outs[od] = e
    if os.path.isfile(f"{runs}/{od}/metrics.json"): bad.append(f"{e}: {od} is already done on CHTC")
    print(f"{e}: {od} NGPU={v['NGPU']} evals={len(v['EXTRA_EVALS'].split())} mid={v['EVALMID']} at={v['EVALAT']} ok")
if not os.path.isdir(f"{runs}/runs"): print(f"pack_chtc_5: warning: no {runs}/runs, the done-on-CHTC check was skipped")
if bad:
    sys.exit("pack_chtc_5: " + "; ".join(bad))
PY
for a in $(for e in "$JOBS"/*.env; do bash -c "set -a; . '$e'; echo \$ARMS"; done | tr ' ' '\n' | sort -u); do
  git show "$FULL:hpc/train_arm.sbatch" | grep -q "^    $a)" || { echo "pack_chtc_5: $PIN has no arm $a"; exit 1; }
done
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
[ "$M" = 12b ] && cp -r cloud/condor/jobs "$C/condor/jobs"     # the 12B tarball carries every 12B env file (batches 1-2)
cp -r "$JOBS" "$C/condor/jobs5_$M"
if [ "$M" = 12b ]; then   # superset of the batch-2 tarball: same md5 for every member there, new = jobs5_12b + CODE_PIN
  python3 - "$T/ref" "$C" <<'PY'
import hashlib, os, sys
ref, new = sys.argv[1:3]
def walk(r): return {os.path.relpath(os.path.join(d, f), r) for d, _, fs in os.walk(r) for f in fs}
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
a, b = walk(ref), walk(new)
missing = sorted(a - b); changed = sorted(p for p in a & b if md5(f"{ref}/{p}") != md5(f"{new}/{p}"))
extra = sorted(p for p in b - a if not (p.startswith("condor/jobs5_12b/") or p == "CODE_PIN"))
if missing or changed or extra:
    sys.exit(f"pack_chtc_5: 12B tarball vs batch 2: missing {missing[:5]} changed {changed[:5]} extra {extra[:5]}")
print(f"12B tarball = batch-2 tarball ({len(a)} files, all md5-identical) + {len(b - a)} new ({', '.join(sorted(b - a))[:200]}...)")
PY
fi
tar czf "$B/cotd_code_data_5_$M.tgz" -C "$C" .
cp "$LIST" "$B/joblist_5_$M.txt"
cp cloud/chtc/{README_5.md,train5_$M.sub,run_train5.sh,make_dags_5.sh} "$B/"
( cd "$B" && bash make_dags_5.sh "$M" >/dev/null && find . -type f ! -name "MANIFEST_5_$M.md5" | sort | xargs md5sum > "MANIFEST_5_$M.md5" )
tar czf "$OUT" -C "$B" .
echo "code pin $FULL"
ls -la "$OUT"; md5sum "$OUT"
