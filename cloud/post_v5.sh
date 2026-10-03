#!/usr/bin/env bash
# After cloud/run_v5.sh, on the same box, per arm on its own GPU (both arms in parallel):
#   1. the 10:33 interpolation rule (notes/prereg_release.md): theta(alpha) = base + alpha * (ft - base) on the
#      selection sets v4_heldout + val over the grid .3-1.0; alpha* = argmax of the mean of the two accuracies, ties to
#      the larger alpha. alpha 1.0 is evaluated here as well (bf16 weights), so all grid points are measured alike;
#   2. theta(alpha*) on every test set -> runs/wise-<arm>-test/a<alpha*>/;
#   3. if alpha* < 1, the sets the final evaluation of run_v5.sh left out, at alpha 1 -> runs/wise-<arm>-extra/a1/.
# Each arm waits for its "done" line in run_v5.out, which run_v5.sh prints after the weights are saved.
#   cd ~/cotd && setsid nohup bash repo/cloud/post_v5.sh > post_v5.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."                      # ~/cotd
PY=env/bin/python
MODEL=models/Qwen3.5-2B-Base; DATA=data/student_v5; E=data/eval
GRID=0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
SEL="v4heldout=$E/v4_heldout.jsonl val=$DATA/val.jsonl"
TEST="kk=$E/kk_heldout.jsonl jevbench=$E/jevbench_public.jsonl td=$E/typed_decisions_test.jsonl bbh=$E/bbh.jsonl \
musr=$E/musr.jsonl policy=$E/policy_heldout.jsonl gsm8k=$E/gsm8k_mc.jsonl diag=$E/diag_kk.jsonl \
claims_ho=$E/claims_ho.jsonl xkk=$E/xkk.jsonl proverqa=$E/proverqa.jsonl logiqa2=$E/logiqa2.jsonl \
lsat_lr=$E/lsat_lr.jsonl lsat_ar=$E/lsat_ar.jsonl zebra=$E/zebra.jsonl bbeh=$E/bbeh.jsonl tdq=$E/td_noul_qform.jsonl \
tdn=$E/td_noul_neg.jsonl folio=$E/folio_val.jsonl jl=$E/jl_heldout.jsonl sharc=$E/sharc_dev.jsonl kkdeep=$E/kk_deep.jsonl"
EXTRA="lsat_ar=$E/lsat_ar.jsonl zebra=$E/zebra.jsonl bbeh=$E/bbeh.jsonl tdq=$E/td_noul_qform.jsonl \
tdn=$E/td_noul_neg.jsonl folio=$E/folio_val.jsonl jl=$E/jl_heldout.jsonl sharc=$E/sharc_dev.jsonl kkdeep=$E/kk_deep.jsonl"
for f in $SEL $TEST; do [ -f "${f#*=}" ] || { echo "missing eval file ${f#*=}"; exit 1; }; done

pick() {                                        # pick <wise dir>: alpha* by the 10:33 rule
  $PY - "$1" <<'EOF'
import json, sys
from pathlib import Path
rows = []
for d in Path(sys.argv[1]).glob("a*"):
    if not (d / "metrics.json").exists():
        continue
    ev = json.loads((d / "metrics.json").read_text())["eval"]
    rows.append((round((ev["v4heldout"]["acc"] + ev["val"]["acc"]) / 2, 9), float(d.name[1:])))
for s, a in sorted(rows, key=lambda r: r[1]):
    print(f"  alpha {a:g}: selection mean {s:.4f}", file=sys.stderr)
print(f"{max(rows)[1]:g}")                      # max over (score, alpha): ties go to the larger alpha
EOF
}

post() {                                        # post <arm> <gpu>
  local arm=$1 gpu=$2 out="runs/TF-$1-Qwen3.5-2B-Base-student_v5-s0"
  until grep -q "done $out" run_v5.out 2>/dev/null; do sleep 60; done
  [ -f "$out/model/config.json" ] || { echo "no saved weights for $arm"; return 1; }
  echo "=== $arm selection grid $(date +%T)"
  CUDA_VISIBLE_DEVICES=$gpu $PY repo/scripts/wise_ft_eval.py --base "$MODEL" --ft "$out/model" --alphas $GRID \
    --eval $SEL --out "runs/wise-$arm-sel" > "logs/wise-$arm-sel.log" 2>&1 || { echo "selection failed $arm"; return 1; }
  local a; a=$(pick "runs/wise-$arm-sel") || { echo "pick failed $arm"; return 1; }
  echo "=== $arm alpha* = $a, test sets $(date +%T)"
  CUDA_VISIBLE_DEVICES=$gpu $PY repo/scripts/wise_ft_eval.py --base "$MODEL" --ft "$out/model" --alphas "$a" \
    --eval $TEST --out "runs/wise-$arm-test" > "logs/wise-$arm-test.log" 2>&1 || { echo "test failed $arm"; return 1; }
  if [ "$a" != 1 ]; then
    echo "=== $arm extra sets at alpha 1 $(date +%T)"
    CUDA_VISIBLE_DEVICES=$gpu $PY repo/scripts/wise_ft_eval.py --base "$MODEL" --ft "$out/model" --alphas 1 \
      --eval $EXTRA --out "runs/wise-$arm-extra" > "logs/wise-$arm-extra.log" 2>&1 || echo "extra failed $arm"
  fi
  echo "=== $arm post done $(date +%T)"
}

post M1 0 & post M2 1 &
wait
echo "post all done $(date +%T)"
