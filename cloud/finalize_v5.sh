#!/usr/bin/env bash
# After cloud/post_v5.sh and the release decision: make the release checkpoint of the chosen arm and check it.
#   1. theta(alpha) of the arm, saved as a model directory (scripts/wise_ft_eval.py --save: bf16, computed exactly as
#      evaluated), evaluated on zebra_ovl on the way (the registered ZebraLogic sensitivity set needs it);
#   2. scripts/sev_infer.py on the saved weights against the evaluated predictions of that alpha (first 100 items of
#      JevBench public, BBH and TD): top-option agreement and the largest probability difference;
#   3. sha256 of the weight files.
#   cd ~/cotd && GPU=0 bash repo/cloud/finalize_v5.sh M1 0.8
set -euo pipefail
cd "$(dirname "$0")/../.."                      # ~/cotd
ARM=$1 A=$2
PY=env/bin/python E=data/eval
FT=runs/TF-$ARM-Qwen3.5-2B-Base-student_v5-s0/model REL=runs/release-$ARM
[ -f "$FT/config.json" ] || { echo "no fine-tuned weights at $FT"; exit 1; }
[ -d "runs/wise-$ARM-test/a$A" ] || { echo "no evaluated predictions at runs/wise-$ARM-test/a$A"; exit 1; }
CUDA_VISIBLE_DEVICES=${GPU:-0} $PY repo/scripts/wise_ft_eval.py --base models/Qwen3.5-2B-Base --ft "$FT" --alphas "$A" \
  --eval zebra_ovl=$E/zebra_ovl.jsonl --out "runs/wise-$ARM-ovl" --save "$REL"
CUDA_VISIBLE_DEVICES=${GPU:-0} $PY - "$REL" "runs/wise-$ARM-test/a$A" <<'EOF'
import json, sys
sys.path.insert(0, "repo/scripts")
from sev_infer import decide, load
rel, pdir = sys.argv[1], sys.argv[2]
model, tok = load(rel)
worst, agree, n = 0.0, 0, 0
for name, path in (("jevbench", "data/eval/jevbench_public.jsonl"), ("bbh", "data/eval/bbh.jsonl"),
                   ("td", "data/eval/typed_decisions_test.jsonl")):
    items = {json.loads(l)["item_id"]: json.loads(l) for l in open(path)}
    for line in list(open(f"{pdir}/preds_{name}.jsonl"))[:100]:
        p = json.loads(line)
        q = decide(model, tok, items[p["item_id"]]["prompt"], letters=p["labels"])
        qs = [q[L] for L in p["labels"]]
        worst = max(worst, max(abs(x - y) for x, y in zip(qs, p["probs"])))
        agree += max(range(len(qs)), key=qs.__getitem__) == max(range(len(qs)), key=p["probs"].__getitem__)
        n += 1
print(f"sev_infer check: {n} items, top-option agreement {agree}/{n}, largest probability difference {worst:.4f}")
EOF
sha256sum "$REL"/*.safetensors
echo "finalize done $(date +%T)"
