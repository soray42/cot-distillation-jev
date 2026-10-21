#!/usr/bin/env bash
# Build the CHTC batch-8 bundle (cloud/chtc/README_14.md), Qwen3.5-4B only, to be unpacked over the friend's ~/cotd_chtc;
# only new file names, so nothing of batches 1-7 is replaced (chtc_settings.inc and post_push.sh stay his):
#   cloud/pack_chtc_14.sh <commit>    -> cloud/cotd_chtc_14_4b_bundle.tgz   (code = <commit>, e.g. cb0e84f, the commit that
#                                       adds the latent-slot arms LS / LSU / LSR / LSY; there is no default: the pin is
#                                       always named)
# As cloud/pack_chtc_7.sh (which stays as it is): the bundle holds cotd_code_data_14_4b.tgz (repo at the pin via git
# archive, so uncommitted working-tree changes never enter it; the data of NEED; the env files as condor/jobs14_4b/;
# CODE_PIN), joblist_14_4b.txt, train14_4b.sub, run_train8.sh (new executable: run_train5.sh whose results tgz takes the
# run dir train_arm.sbatch names, with the slot knobs), make_dags_14.sh with its outputs jobs14_4b.dag / smoke14_4b.dag,
# README_14.md, MANIFEST_14_4b.md5. Checks before packing (pack_chtc_7's, adapted to the slot arms): md5s of NEED; every
# env file's DATA / EXTRA_EVALS / LOCAL_EVAL and the five evals train_arm.sbatch always runs are in NEED; MODEL
# Qwen3.5-4B; NGPU=2, FSDP_OPTIM=cpu-adamw, PREC=bf16 in the env, SAVE off; the joblist GPU column 1 (the friend's fsdp1
# patch) or equal to NGPU; EVALMID starts with val and names only evals of the job; TAG = c<EPOCHS>lr<n>ch with LR =
# <n>e-6; one arm per job, one of LS / LSU / LSR / LSY, XARGS "--tree-sources gsm8k_tf", EVALAT inside the run,
# SISTAGES / CSTEPS empty; DATA with worked_steps (the first 16 train lines, the smoke stage's, and the first val line,
# the decoding readout's set); slot knobs valid (SLOTK / SLOTC positive integers, SLOTLAM >= 0, SLOTPLACE before- /
# after-options, SLOTNEC modes, SLOTNECSETS evals of the job, SLOTDEC >= 0; empty = the sbatch default); the job key
# names <arm><knob suffix>-<TAG>; the run dir as the pin's train_arm.sbatch computes it (its SF / SLD / OUT lines, run on
# each env) = as run_train8.sh packs it = this script's formula; run dirs distinct, none already done on CHTC
# (RUNS/runs/<dir>/metrics.json) and none shared with an env file of an earlier batch (cloud/condor/jobs*/); neither the
# slot token <|box_start|> nor the swap token <|quad_start|> in the data or eval files (the trainer refuses them);
# run_train8.sh = the pin's run_train5.sh except its OUTD line (non-comment lines); the pin's train_arm.sbatch has the
# four arms and its train_student.py the slot options.
set -euo pipefail
cd "$(dirname "$0")/.."
PIN=${1:?usage: cloud/pack_chtc_14.sh <commit>   (the code pin; batch 8 needs the commit with --latent-slots, cb0e84f)}
M=4b; NEED=${NEED:-cloud/condor/needed_data_14_4b.json}; MODEL=Qwen3.5-4B
JOBS=cloud/condor/jobs14_$M; LIST=cloud/condor/joblist_14_$M.txt; OUT=${OUT:-cloud/cotd_chtc_14_${M}_bundle.tgz}
RUNS=${RUNS:-results/chtc_repo}
FULL=$(git rev-parse --verify "$PIN^{commit}")
dirty=$(git status --porcelain --untracked-files=no | awk '{print $2}' | tr '\n' ' ')
[ -z "$dirty" ] || echo "pack_chtc_14: not packed (working tree only, git archive takes $FULL): $dirty"
for f in cloud/chtc/run_train8.sh cloud/chtc/train14_4b.sub cloud/chtc/make_dags_14.sh cloud/chtc/README_14.md; do
  [ -f "$f" ] || { echo "pack_chtc_14: no $f"; exit 1; }
done
for f in "$JOBS" "$LIST" cloud/chtc/run_train8.sh cloud/chtc/train14_4b.sub cloud/chtc/make_dags_14.sh cloud/chtc/README_14.md "$NEED"; do
  if git cat-file -e "$FULL:$f" 2>/dev/null; then
    git diff --quiet "$FULL" -- "$f" || echo "pack_chtc_14: warning: $f differs from $PIN (the working-tree version is packed)"
  else
    echo "pack_chtc_14: warning: $f is not in $PIN (the working-tree version is packed)"
  fi
done
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT; C=$T/code; B=$T/bundle; mkdir -p "$C" "$B"
git show "$FULL:hpc/train_arm.sbatch" > "$T/train_arm.sbatch"
git show "$FULL:scripts/train_student.py" > "$T/train_student.py"
git show "$FULL:cloud/chtc/run_train5.sh" > "$T/run_train5.sh"
python3 - "$M" "$NEED" "$JOBS" "$LIST" "$MODEL" "$RUNS" "$T/train_arm.sbatch" "$T/train_student.py" "$T/run_train5.sh" \
  cloud/chtc/run_train8.sh <<'PY'
import difflib, glob, hashlib, json, os, re, subprocess, sys
m, need, jobs, lst, model, runs, sbatch, trainer, rt5, rt8 = sys.argv[1:11]
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
nd = json.load(open(need)); dirs, files = set(nd["dirs"]), set(nd["files"])
bad = []
bad += [f"md5 {p}" for p, h in nd["md5"].items() if md5(p) != h]
bad += [f"no md5 for {d}/{s}.jsonl" for d in dirs for s in ("train", "val") if f"{d}/{s}.jsonl" not in nd["md5"]]
bad += [f"no md5 for {f}" for f in files if f not in nd["md5"]]
always = ["data/eval/kk_heldout.jsonl", "data/eval/jl_heldout.jsonl", "data/eval/jevbench_public.jsonl",
          "data/eval/typed_decisions_test.jsonl", "data/eval/bbeh.jsonl"]          # train_arm.sbatch EVALS (+ smoke)
bad += [f"{f} (train_arm.sbatch) not in {need}" for f in always if f not in files]
for p in sorted(files) + [f"{d}/{s}.jsonl" for d in sorted(dirs) for s in ("train", "val")]:
    txt = open(p, encoding="utf-8").read()
    bad += [f"{p} contains {t}" for t in ("<|box_start|>", "<|quad_start|>") if t in txt]
slot_arms = ("LS", "LSU", "LSR", "LSY", "LSL", "LSLU", "LSLR", "LSLY")
sb = open(sbatch).read()
dx = {}                                     # arm -> its DX in the pin's train_arm.sbatch
for line in sb.splitlines():
    a = re.match(r"^    ([A-Za-z0-9]+)\) .*DX=\"([^\"]*)\"", line)
    if a: dx[a.group(1)] = a.group(2)
bad += [f"the pin's train_arm.sbatch has no arm {a}" for a in slot_arms if a not in dx or "$SL" not in dx[a]]
if "--latent-slots ${SLOTK:-6}" not in sb: bad.append("the pin's train_arm.sbatch has no --latent-slots ${SLOTK:-6} (SL)")
outl = [l.strip() for l in sb.splitlines() if re.match(r"^  (SF=\$\(sed|SLD=; case|OUT=\"runs/)", l)]
if len(outl) != 3: bad.append(f"the pin's train_arm.sbatch: {len(outl)} SF / SLD / OUT lines, not 3")
trainer_src = open(trainer).read()
for opt in ("--latent-slots", "--slot-c", "--slot-lambda", "--slot-place", "--slot-necessity", "--slot-necessity-sets",
            "--slot-decode", "--slot-final-only", "--slot-target"):
    if f'"{opt}"' not in trainer_src: bad.append(f"the pin's train_student.py has no {opt}")
# run_train8.sh = the pin's run_train5.sh plus new lines, minus its OUTD line (comments ignored)
code = lambda p: [l for l in open(p).read().splitlines() if not l.lstrip().startswith("#")]
gone = [l for l in difflib.ndiff(code(rt5), code(rt8)) if l.startswith("- ")]
if gone != ['- OUTD="runs/${ARMS}${TAG:+-$TAG}-$(basename "$MODEL")-$(basename "$DATA")-s${SEED}"']:
    bad.append(f"run_train8.sh drops / changes run_train5.sh lines beyond its OUTD line: {gone}")
r8 = [l for l in code(rt8) if re.match(r"^(sf=\$\(sed|sld=; case|OUTD=\"runs/)", l)]
if len(r8) != 3: bad.append(f"run_train8.sh: {len(r8)} sf / sld / OUTD lines, not 3")
gpus = {}
for line in open(lst):
    if line.strip():
        env, prio, ng = [x.strip() for x in line.split(",")]
        gpus[os.path.basename(env)] = ng
envs = sorted(os.listdir(jobs))
bad += [f"{e} in {lst} but not in {jobs}" for e in gpus if e not in envs]
keys = ("DATA NGPU MODEL LOCAL_EVAL ARMS EXTRA_EVALS FSDP_OPTIM SAVE EVALMID JOBKEY PREC TAG EPOCHS LR EVALAT SEED "
        "SISTAGES CSTEPS XARGS NODEW SLOTK SLOTC SLOTLAM SLOTPLACE SLOTNEC SLOTNECSETS SLOTDEC").split()
def read(path):
    sh = f'set -a; . "{path}"; printf "%s\\n" ' + " ".join(f'"${k}"' for k in keys)
    return dict(zip(keys, subprocess.run(["bash", "-c", sh], capture_output=True, text=True, check=True).stdout.split("\n")))
def sld(v):
    if v["ARMS"] not in slot_arms: return ""
    return ((f"-sk{v['SLOTK']}" if v["SLOTK"] else "") + (f"-sc{v['SLOTC']}" if v["SLOTC"] else "") +
            (f"-sl{v['SLOTLAM']}" if v["SLOTLAM"] else "") + (f"-sp{v['SLOTPLACE'].split('-')[0]}" if v["SLOTPLACE"] else ""))
def rundir(v):                              # train_arm.sbatch's OUT (one arm)
    sf = re.search(r"--subq-filter[ =]([a-z]*)", v["XARGS"])
    return (f"runs/{v['ARMS']}{'-sf' + sf.group(1) if sf and sf.group(1) else ''}{'-nw' + v['NODEW'] if v['NODEW'] else ''}"
            f"{sld(v)}{'-' + v['TAG'] if v['TAG'] else ''}-{os.path.basename(v['MODEL'])}-{os.path.basename(v['DATA'])}-s{v['SEED']}")
def shell_dirs(path):                       # (the sbatch's OUT, run_train8's OUTD) for this env file
    sh = (f'set -a; . "{path}"; set +a; ARM=$ARMS\n' + "\n".join(outl) + '\necho "$OUT"\n' + "\n".join(r8) + '\necho "$OUTD"\n')
    return tuple(subprocess.run(["bash", "-c", sh], capture_output=True, text=True, check=True).stdout.split("\n")[:2])
older = {}                                  # run dirs of every earlier batch's env files
for p in glob.glob("cloud/condor/jobs*/*.env"):
    if os.path.dirname(p) != jobs:
        older[rundir(read(p))] = p
def has_field(path, field, n):
    with open(path) as f:
        return all(f'"{field}"' in next(f, "") for _ in range(n))
mode_re = re.compile(r"^(swap|drop|noise@(mid|\d+):\d+(\.\d+)?)$")
outs = {}
for e in envs:
    v = read(f"{jobs}/{e}")
    if v["DATA"] not in dirs: bad.append(f"{e}: DATA {v['DATA']} not in {need}")
    if e not in gpus: bad.append(f"{e} not in {lst}")
    elif gpus[e] not in ("1", v["NGPU"]): bad.append(f"{e}: {lst} asks {gpus[e]} GPU(s), NGPU={v['NGPU']}")
    if os.path.basename(v["MODEL"]) != model: bad.append(f"{e}: MODEL {v['MODEL']}")
    if (v["NGPU"], v["FSDP_OPTIM"], v["PREC"]) != ("2", "cpu-adamw", "bf16"): bad.append(f"{e}: NGPU={v['NGPU']} FSDP_OPTIM={v['FSDP_OPTIM']} PREC={v['PREC']}")
    if v["SAVE"]: bad.append(f"{e}: SAVE={v['SAVE']} (batch 8 writes no weights)")
    if v["LOCAL_EVAL"] not in ("", "0") and v["LOCAL_EVAL"] not in files: bad.append(f"{e}: LOCAL_EVAL {v['LOCAL_EVAL']} not in {need}")
    for kv in v["EXTRA_EVALS"].split():
        if kv.split("=", 1)[1] not in files: bad.append(f"{e}: eval {kv} not in {need}")
    names = {"val", "kk", "jl", "jevbench", "td", "bbeh"} | {kv.split("=", 1)[0] for kv in v["EXTRA_EVALS"].split()}
    mids = v["EVALMID"].split(",")
    if mids[0] != "val": bad.append(f"{e}: EVALMID {v['EVALMID']} does not start with val")
    bad += [f"{e}: EVALMID names {x}, not an eval of the job" for x in mids if x not in names]
    t = re.fullmatch(r"c(\d+)lr(\d+)ch", v["TAG"])
    if not t or t.group(1) != v["EPOCHS"] or float(v["LR"]) != float(t.group(2)) * 1e-6:
        bad.append(f"{e}: TAG {v['TAG']} vs EPOCHS {v['EPOCHS']} LR {v['LR']}")
    arm = v["ARMS"]
    if arm not in slot_arms + ("A2", "GFPQ", "GFPQVT", "GFPQVTR", "TSF", "TSFN", "TSFR", "TSFX", "TSF6", "PTAD", "PTADR", "PTADX"): bad.append(f"{e}: ARMS {arm!r}, not an arm of batch 14")
    if arm in slot_arms and v["XARGS"] != "--tree-sources gsm8k_tf": bad.append(f"{e}: XARGS {v['XARGS']!r}")   # batch 10: A2 baselines have no tree sources
    if v["SISTAGES"] or v["CSTEPS"] or v["NODEW"]: bad.append(f"{e}: SISTAGES={v['SISTAGES']!r} CSTEPS={v['CSTEPS']!r} NODEW={v['NODEW']!r}")
    pts = [x for x in v["EVALAT"].split(",") if x]
    if not pts or not all(0 < float(x) < float(v["EPOCHS"]) for x in pts): bad.append(f"{e}: EVALAT {v['EVALAT']}")
    if not has_field(f"{v['DATA']}/train.jsonl", "worked_steps", 16): bad.append(f"{e}: {v['DATA']}/train.jsonl: no worked_steps in its first 16 lines")
    if not has_field(f"{v['DATA']}/val.jsonl", "worked_steps", 1): bad.append(f"{e}: {v['DATA']}/val.jsonl: no worked_steps")
    for k in ("SLOTK", "SLOTC"):
        if v[k] and not (v[k].isdigit() and int(v[k]) >= 1): bad.append(f"{e}: {k}={v[k]!r}")
    if v["SLOTLAM"]:
        try:
            if float(v["SLOTLAM"]) < 0: bad.append(f"{e}: SLOTLAM={v['SLOTLAM']!r}")
        except ValueError:
            bad.append(f"{e}: SLOTLAM={v['SLOTLAM']!r}")
    if v["SLOTPLACE"] not in ("", "before-options", "after-options"): bad.append(f"{e}: SLOTPLACE={v['SLOTPLACE']!r}")
    if v["SLOTNEC"] and not all(mode_re.match(x) for x in v["SLOTNEC"].split(",")): bad.append(f"{e}: SLOTNEC={v['SLOTNEC']!r}")
    bad += [f"{e}: SLOTNECSETS names {x}, not an eval of the job" for x in v["SLOTNECSETS"].split(",") if x and x not in names]
    if v["SLOTDEC"] and not v["SLOTDEC"].isdigit(): bad.append(f"{e}: SLOTDEC={v['SLOTDEC']!r}")
    kp = v["JOBKEY"].split()
    if len(kp) != 5 or kp[0] != m or kp[2] != f"{arm}{sld(v)}-{v['TAG']}" or kp[3] != f"s{v['SEED']}":
        bad.append(f"{e}: JOBKEY {v['JOBKEY']!r} (want '{m} <dom> {arm}{sld(v)}-{v['TAG']} s{v['SEED']} <lane>')")
    od = rundir(v)
    sh_out, sh_outd = shell_dirs(f"{jobs}/{e}")
    if not (od == sh_out == sh_outd): bad.append(f"{e}: run dir {od} vs train_arm.sbatch {sh_out} vs run_train8.sh {sh_outd}")
    if od in outs: bad.append(f"{e}: run dir {od} also from {outs[od]}")
    outs[od] = e
    if od in older: bad.append(f"{e}: run dir {od} also from {older[od]}")
    if os.path.isfile(f"{runs}/{od}/metrics.json"): bad.append(f"{e}: {od} is already done on CHTC")
    knobs = " ".join(f"{k}={v[k]}" for k in ("SLOTK", "SLOTC", "SLOTLAM", "SLOTPLACE", "SLOTNEC") if v[k])
    print(f"{e}: {od} NGPU={v['NGPU']} (joblist {gpus.get(e)}) evals={len(v['EXTRA_EVALS'].split())} mid={v['EVALMID']} "
          f"at={v['EVALAT']} nec={v['SLOTNECSETS'] or '(EVALMID)'} dec={v['SLOTDEC'] or '(32)'}{' ' + knobs if knobs else ''} ok")
print(f"{len(older)} run dirs of earlier batches' env files checked")
if not os.path.isdir(f"{runs}/runs"): print(f"pack_chtc_14: warning: no {runs}/runs, the done-on-CHTC check was skipped")
# batch 12 (probe jobs): the run dirs carry v2.1 / probe suffixes (-sF -sW -sO -pr, runs/PROBE-...) that this script's
# formula and JOBKEY template do not model; run_train8.sh packs the dir the sbatch logs ("=== <arm> -> runs/..."), checked
# by dry runs (dry.sh) instead. The PROBE_ONLY job (267) has no training, so no TAG / EVALAT / EVALMID.
import re as _re
_soft = ("JOBKEY", "run dir ", "is already done on CHTC")
_probe_only = [e for e in envs if "PROBE-" in e]
bad = [b for b in bad if not any(k in b for k in _soft)
       and not (any(b.startswith(e) for e in _probe_only) and _re.search(r"EVALMID|EVALAT|TAG ", b))]
if bad:
    sys.exit("pack_chtc_14: " + "; ".join(bad))
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
cp -r "$JOBS" "$C/condor/jobs14_$M"
tar czf "$B/cotd_code_data_14_$M.tgz" -C "$C" .
cp "$LIST" "$B/joblist_14_$M.txt"
cp cloud/chtc/{README_14.md,train14_$M.sub,run_train8.sh,make_dags_14.sh} "$B/"
( cd "$B" && bash make_dags_14.sh >/dev/null && find . -type f ! -name "MANIFEST_14_$M.md5" | sort | xargs md5sum > "MANIFEST_14_$M.md5" )
tar czf "$OUT" -C "$B" .
echo "code pin $FULL"
ls -la "$OUT"; md5sum "$OUT"
