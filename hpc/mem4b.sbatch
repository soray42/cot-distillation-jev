#!/bin/bash
#SBATCH --job-name=cotd-mem4b
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x-%j.out
# Memory / speed / resume smoke for full-parameter 4B training on one 40 GB MIG slice (notes/next_round/sev4b_plan.md
# §4-§7, §9 C0), through the real scripts/train_student.py full-tree path. Every part prints one "MEM4B {json}" line.
#   torchao  torchao AdamW8bit (bf16 stochastic rounding) on one 4096 x 4096 bf16 matrix: 2 steps, state_dict save/load
#            into a fresh optimizer (map_location cpu, as train_student.py does), 1 more step on both from the same CUDA
#            RNG state; the weights must match exactly and the loaded m/v must still be 8-bit on the GPU (allocated MiB
#            of the load: about 32 when 8-bit, 64 when cast to dense bf16), the [unverified] torchao state round trip of
#            plan §7; plus the tensor-LR scheduler check. "ok": false (exit 1, "failed" line) when any of these fails.
#   mem      4B (MODEL4B), bf16 + adamw8bit-sr, every sequence cut to exactly MAX-LEN tokens (synthetic long prompts, finals
#            only, 4 per update) so each micro-batch is a worst-case TOKBUDGET-token one; MAX-LEN 1536 and 3072, STEPS4B
#            updates: peak GiB (max_memory_allocated) and s/update over the second half of the run (the first steps
#            include torch.compile).
#   speed    4B, the same optimizer on real TF updates (32 problems + their trees, spread over DATA), MAX-LEN 1536 and 3072,
#            STEPS4B updates: s/update against the plan §6.3 gate (15.4 s/update for 2 epochs of v5 in 22.7 h).
#   roundtrip 2B (MODEL2B), bf16 + adamw8bit-sr, 6 updates straight vs stop after update 2 (exit 3) + resume: per-step
#            loss differences after the resume next to the run-to-run differences before it (GPU nondeterminism).
#   equiv    2B, fp32master + AdamW vs bf16 + adamw8bit-sr, EQSTEPS updates (default 100) of TF on DATA (default
#            data/student_v3f), same seed, order and token-budget batching; evaluated on EQVAL items of DATA/val.jsonl.
#            Readout: loss over the last 20 updates, s/update, peak GiB, val acc/nll, item-level mean |dp| and argmax
#            agreement between the two arms (the plan §9 C0 optimizer-risk check in small).
# 4B parts are skipped with a message when MODEL4B is missing, so torchao/roundtrip/equiv can run before the download.
# From ~/cotd:  CC1D=1 sbatch repo/hpc/mem4b.sbatch     (PARTS="mem speed" etc. selects parts; WORK=<dir> for the run
# directories, default runs/mem4b-<job>; the roundtrip's checkpoint needs about 7.6 GB there (2B: 3.8 GB bf16 weights +
# 3.8 GB 8-bit m and v), the other parts < 0.1 GB; check lquota first). torchao goes in with
# "envs/cotd/bin/pip install --no-deps torchao" (then "pip check"), so pip cannot replace torch 2.14.
# Dry-run the shell logic on the login node first:  DRY_RUN=1 bash repo/hpc/mem4b.sbatch
# XARGS is added to every training run, e.g. XARGS="--node-format mc --aux-total 0.5" DATA=data/student_v5 for the v5
# recipe of the plan's speed gate (default: DATA's yes/no nodes, aux-total 1).
# The 3g.40gb slice has the same memory and fewer SMs: sbatch --gres=gpu:3g.40gb:1 ... gives its s/update.
set -euo pipefail
cd "$HOME/cotd"
module load cuda/12.4 2>/dev/null || true
export HF_HOME="$HOME/cotd/hf" TOKENIZERS_PARALLELISM=false
[ -n "${CC1D:-}" ] && export PYTHONPATH="$HOME/cotd/cc1d${PYTHONPATH:+:$PYTHONPATH}"
PY=envs/cotd/bin/python
RUN="$PY"
[ -n "${DRY_RUN:-}" ] && RUN="echo [dry-run] python"
REPO="${REPO:-repo}"
MODEL4B="${MODEL4B:-$HOME/cotd/models/Qwen3.5-4B-Base}"; MODEL2B="${MODEL2B:-$HOME/cotd/models/Qwen3.5-2B-Base}"
DATA="${DATA:-data/student_v3f}"; BUDGET="${TOKBUDGET:-3072}"; STEPS4B="${STEPS4B:-20}"; EQSTEPS="${EQSTEPS:-100}"
EQVAL="${EQVAL:-300}"; PARTS="${PARTS:-torchao mem speed roundtrip equiv}"
W="${WORK:-runs/mem4b-${SLURM_JOB_ID:-local}}"
COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
echo "code: $REPO @ $COMMIT  parts: $PARTS  work: $W"
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi here"
for f in "$REPO/scripts/train_student.py" "$DATA/train.jsonl" "$DATA/val.jsonl"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
[ -d "$MODEL2B" ] || { echo "missing $MODEL2B"; exit 1; }
HAVE4B=1; [ -d "$MODEL4B" ] || { HAVE4B=; echo "MEM4B {\"skip\": \"4B parts: $MODEL4B missing\"}"; }
$RUN -c "import torch, torchao; print('torch', torch.__version__, 'torchao', torchao.__version__, \
torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda', \
'free/total GiB', [round(x / 2**30, 2) for x in torch.cuda.mem_get_info()] if torch.cuda.is_available() else '-')" || { echo "torchao import failed"; exit 1; }
mkdir -p "$W"
has() { [[ " $PARTS " == *" $1 "* ]]; }

# data: synthetic worst-case prompts, real TF updates spread over the sources, the equivalence and roundtrip sets
$RUN - "$DATA" "$W" "$STEPS4B" "$EQSTEPS" "$EQVAL" <<'EOF'
import json, sys
from pathlib import Path
data, w, steps4b, eqsteps, eqval = Path(sys.argv[1]), Path(sys.argv[2]), *map(int, sys.argv[3:])
items = [json.loads(l) for l in open(data / "train.jsonl") if l.strip()]
spread = lambda n: items[::max(1, len(items) // n)][:n]          # every k-th item: all sources, not the first family
def write(name, rows):
    (w / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
filler = "The record continues with further routine details of no consequence. " * 400   # > 3072 tokens
write("long.jsonl", [dict(it, prompt=filler + it["prompt"], subqs=[]) for it in spread(4 * steps4b)])
write("speed.jsonl", spread(32 * steps4b))
write("eq.jsonl", spread(32 * eqsteps))
write("rt.jsonl", spread(32 * 6))
val = [json.loads(l) for l in open(data / "val.jsonl") if l.strip()]
write("val.jsonl", val[::max(1, len(val) // eqval)][:eqval])
print("data", {p.name: sum(1 for _ in open(p)) for p in sorted(w.glob("*.jsonl"))})
EOF

run() {   # run <name> <model> <train.jsonl> [args...]: one TF (full tree, parents stated) run into $W/<name>
  local name=$1 model=$2 tr=$3; shift 3
  rm -rf "${W:?}/$name"
  $RUN $REPO/scripts/train_student.py --model "$model" --train "$tr" --out "$W/$name" --final teacher --subq cot \
    --subq-target cot --tree-full true --label-smoothing 0.1 --permute-final 0.5 --token-budget "$BUDGET" --epochs 1 \
    --log-every 1 --seed 0 --code-commit "$COMMIT" ${XARGS:-} "$@" > "$W/$name.log" 2>&1
}
summ() {  # summ <name>...: one MEM4B line per run (peak GiB, s/update over the second half, mean loss of the last 10)
  [ -n "${DRY_RUN:-}" ] && { echo "[dry-run] summary $*"; return 0; }
  $PY - "$W" "$@" <<'EOF'
import json, sys
from pathlib import Path
w = Path(sys.argv[1])
for name in sys.argv[2:]:
    m, log = w / name / "metrics.json", w / f"{name}.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    if not m.exists():
        print("MEM4B", json.dumps({"run": name, "status": "OOM" if "OutOfMemoryError" in text else "failed",
                                   "log_tail": text[-400:]}), flush=True)
        continue
    h = json.loads(m.read_text())["history"]
    n, a = h[-1]["step"], h[max(0, len(h) // 2 - 1)]
    row = {"run": name, "status": "ok", "updates": n, "peak_gib": round(max(r["peak_gib"] for r in h), 2),
           "s_per_update_2nd_half": round((h[-1]["s_per_step"] * n - a["s_per_step"] * a["step"]) / (n - a["step"]), 2)
           if n > a["step"] else None,
           "s_per_update_all": round(h[-1]["s_per_step"], 2), "loss_last10": round(sum(r["loss"] for r in h[-10:]) / len(h[-10:]), 4),
           "seqs_per_update": round(sum(r["seqs"] for r in h) / len(h), 1)}
    if "fwd" in h[-1]:
        row["fwd_per_update"] = round(sum(r["fwd"] for r in h) / len(h), 1)
    print("MEM4B", json.dumps(row), flush=True)
EOF
}

if has torchao; then
  echo "=== torchao state round trip  $(date +%T)"
  $RUN - "$W" <<'EOF' || echo "MEM4B {\"part\": \"torchao\", \"status\": \"failed\"}"
import json, sys
import torch
import torchao
try:
    from torchao.optim import AdamW8bit
except ImportError:                     # torchao < 0.11
    from torchao.prototype.low_bit_optim import AdamW8bit
path = f"{sys.argv[1]}/torchao_state.pt"
def make():
    torch.manual_seed(0)
    lin = torch.nn.Linear(4096, 4096, bias=False, device="cuda", dtype=torch.bfloat16)
    return lin, AdamW8bit(lin.parameters(), lr=1e-3, weight_decay=0.0, bf16_stochastic_round=True)
g = [torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16) * 1e-3 for _ in range(3)]
a, oa = make()
w0 = a.weight.detach().clone()
for i in range(2):
    a.weight.grad = g[i]; oa.step()
moved = (a.weight != w0).float().mean().item()                  # with round-to-nearest most weights would not move
torch.save(oa.state_dict(), path)
b, ob = make()
b.load_state_dict(a.state_dict())
sd = torch.load(path, map_location="cpu", weights_only=False)
torch.cuda.synchronize(); m0 = torch.cuda.memory_allocated()
ob.load_state_dict(sd)
torch.cuda.synchronize(); load_mib = (torch.cuda.memory_allocated() - m0) / 2**20
st = ob.state[b.weight]
types_a = {k: type(v).__name__ for k, v in oa.state[a.weight].items()}
rs = torch.cuda.get_rng_state()
a.weight.grad = g[2]; oa.step()
torch.cuda.set_rng_state(rs)
b.weight.grad = g[2]; ob.step()
sched = torch.optim.lr_scheduler.LambdaLR(oa, lambda s: 0.5)
types_b = {k: type(v).__name__ for k, v in st.items()}
row = {"part": "torchao", "torch": torch.__version__, "torchao": torchao.__version__,
       "state_types": types_b, "state_types_before_save": types_a, "load_alloc_mib": round(load_mib, 1),
       "state_devices": sorted({str(v.device) for k, v in st.items() if torch.is_tensor(v) and k != "step"}),
       "frac_weights_moved_2_steps": round(moved, 4),
       "roundtrip_exact": bool(torch.equal(a.weight, b.weight)),
       "max_abs_diff": (a.weight.float() - b.weight.float()).abs().max().item(),
       "sched_lr_type": type(sched.get_last_lr()[0]).__name__,
       "sched_lr": float(sched.get_last_lr()[0]), "lr_after_sched": float(oa.param_groups[0]["lr"])}
row["ok"] = (row["roundtrip_exact"] and types_b == types_a and "8bit" in types_b.get("exp_avg", "").lower()
             and row["state_devices"] == ["cuda:0"] and load_mib < 48 and moved > 0.5
             and abs(row["lr_after_sched"] - 5e-4) < 1e-9)
print("MEM4B", json.dumps(row), flush=True)
sys.exit(0 if row["ok"] else 1)
EOF
  rm -f "$W/torchao_state.pt"
fi

SR="--precision bf16 --optim adamw8bit-sr"
if has mem && [ -n "$HAVE4B" ]; then
  for L in 1536 3072; do
    echo "=== 4B memory, every micro-batch $BUDGET padded tokens, max-len $L  $(date +%T)"
    run mem$L "$MODEL4B" "$W/long.jsonl" $SR --max-len $L --items-per-update 4 || true
    summ mem$L
  done
fi
if has speed && [ -n "$HAVE4B" ]; then
  for L in 1536 3072; do
    echo "=== 4B speed, real TF updates, max-len $L  $(date +%T)"
    run speed$L "$MODEL4B" "$W/speed.jsonl" $SR --max-len $L || true
    summ speed$L
  done
fi
if has roundtrip; then
  echo "=== 2B resume round trip (bf16 + adamw8bit-sr)  $(date +%T)"
  run rt_a "$MODEL2B" "$W/rt.jsonl" $SR --lr 1e-4 || true
  RC=0; run rt_b "$MODEL2B" "$W/rt.jsonl" $SR --lr 1e-4 --stop-at-step 2 || RC=$?
  echo "planned stop exit code: $RC (expected 3)"
  cp "$W/rt_b.log" "$W/rt_b1.log" 2>/dev/null || true
  $RUN $REPO/scripts/train_student.py --model "$MODEL2B" --train "$W/rt.jsonl" --out "$W/rt_b" --final teacher --subq cot \
    --subq-target cot --tree-full true --label-smoothing 0.1 --permute-final 0.5 --token-budget "$BUDGET" --epochs 1 \
    --log-every 1 --seed 0 --code-commit "$COMMIT" ${XARGS:-} $SR --lr 1e-4 --ckpt-every-min 1000 > "$W/rt_b.log" 2>&1 || true
  [ -n "${DRY_RUN:-}" ] || $PY - "$W" "$RC" <<'EOF'
import json, sys
from pathlib import Path
w = Path(sys.argv[1])
la, lb = ([r["loss"] for r in json.loads((w / o / "metrics.json").read_text())["history"]]
          if (w / o / "metrics.json").exists() else [] for o in ("rt_a", "rt_b"))
ok = len(la) == len(lb) == 6
d = [abs(x - y) for x, y in zip(la, lb)]
print("MEM4B", json.dumps({"part": "roundtrip", "stop_exit": int(sys.argv[2]),
                           "resumed": "resumed from" in (w / "rt_b.log").read_text(errors="replace"),
                           "loss_straight": la, "loss_resumed": lb,
                           "maxdiff_before_resume": max(d[:2]) if ok else None,
                           "maxdiff_after_resume": max(d[2:]) if ok else None,
                           "ckpt_removed": not (w / "rt_b" / "ckpt").exists()}), flush=True)
EOF
fi
if has equiv; then
  echo "=== 2B equivalence, $EQSTEPS updates: fp32master + AdamW vs bf16 + adamw8bit-sr  $(date +%T)"
  run eq_fp32 "$MODEL2B" "$W/eq.jsonl" --precision fp32master --optim adamw --max-len 1536 --eval "val=$W/val.jsonl" || true
  run eq_sr "$MODEL2B" "$W/eq.jsonl" $SR --max-len 1536 --eval "val=$W/val.jsonl" || true
  summ eq_fp32 eq_sr
  [ -n "${DRY_RUN:-}" ] || $PY - "$W" <<'EOF'
import json, sys
from pathlib import Path
w = Path(sys.argv[1])
try:
    m = {o: json.loads((w / o / "metrics.json").read_text()) for o in ("eq_fp32", "eq_sr")}
    p = {o: [json.loads(l) for l in open(w / o / "preds_val.jsonl")] for o in m}
except FileNotFoundError as e:
    print("MEM4B", json.dumps({"part": "equiv", "status": "failed", "missing": str(e)}), flush=True)
    sys.exit()
pa, pb = p["eq_fp32"], p["eq_sr"]
assert [x["item_id"] for x in pa] == [x["item_id"] for x in pb]
dp = [max(abs(u - v) for u, v in zip(x["probs"], y["probs"])) for x, y in zip(pa, pb)]
agree = [x["probs"].index(max(x["probs"])) == y["probs"].index(max(y["probs"])) for x, y in zip(pa, pb)]
h = {o: m[o]["history"] for o in m}
row = {"part": "equiv", "status": "ok", "updates": [h[o][-1]["step"] for o in m], "n_val": len(pa),
       "loss_last20": {o: round(sum(r["loss"] for r in h[o][-20:]) / len(h[o][-20:]), 4) for o in m},
       "s_per_update": {o: round(h[o][-1]["s_per_step"], 2) for o in m},
       "peak_gib": {o: round(max(r["peak_gib"] for r in h[o]), 2) for o in m},
       "val": {o: {k: m[o]["eval"]["val"].get(k) for k in ("acc", "nll", "ece")} for o in m},
       "mean_abs_dp": round(sum(dp) / len(dp), 4), "max_abs_dp": round(max(dp), 4),
       "argmax_agree": round(sum(agree) / len(agree), 4)}
print("MEM4B", json.dumps(row), flush=True)
EOF
fi
echo "all done $(date +%T)"
