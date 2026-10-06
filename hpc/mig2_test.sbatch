#!/bin/bash
#SBATCH --job-name=cotd-mig2
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# Can one job train across its two MIG slices? (owner, 2026-10-06: "优先试试两个切片合起来")
# CUDA lets a process enumerate only one MIG instance, so the test runs one process per slice (CUDA_VISIBLE_DEVICES =
# that slice's MIG UUID) and checks: (1) what one process sees; (2) NCCL and gloo all_reduce between the slices
# (correctness, GB/s); (3) one FSDP2 (fully_shard) training step of a small model over NCCL, gradients compared with a
# single-process step. Prints one "MIG2 {json}" line per check.
#   From ~/cotd: sbatch repo-w3/hpc/mig2_test.sbatch   (or DRY_RUN=1 bash ... on the login node: shell logic only)
set -uo pipefail
cd "$HOME/cotd"
PY=envs/cotd/bin/python
[ -n "${DRY_RUN:-}" ] && PY="echo [dry-run] python"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; nvidia-smi -L 2>/dev/null
mapfile -t UUIDS < <(nvidia-smi -L 2>/dev/null | grep -oE 'MIG-[0-9a-f-]+')
echo "MIG UUIDs: ${UUIDS[*]:-none}"
$PY -c "import torch, json; print('MIG2', json.dumps({'check': 'one_process', 'device_count': torch.cuda.device_count(), 'names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))"
[ "${#UUIDS[@]}" -ge 2 ] || { echo "MIG2 {\"check\": \"uuids\", \"ok\": false}"; [ -n "${DRY_RUN:-}" ] || exit 1; UUIDS=(MIG-a MIG-b); }
T=$(mktemp -d)
cat > "$T/t.py" <<'EOF'
import json, os, sys, time, datetime
import torch, torch.distributed as dist
backend, rank = sys.argv[1], int(os.environ["RANK"])
dist.init_process_group(backend, timeout=datetime.timedelta(seconds=180))
dev = torch.device("cuda", 0)
torch.cuda.set_device(dev)
out = {"check": f"allreduce_{backend}", "rank": rank}
try:
    x = torch.full((64 * 2**20,), float(rank + 1), device=dev)          # 256 MB fp32
    dist.all_reduce(x); torch.cuda.synchronize()
    ok = bool((x == 3.0).all())
    t = time.time()
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    out.update(ok=ok, gb_per_s=round(5 * x.numel() * 4 / (time.time() - t) / 1e9, 2))
except Exception as e:
    out.update(ok=False, error=repr(e)[:300])
if backend == "nccl" and out.get("ok"):
    try:
        from torch.distributed.fsdp import fully_shard
        torch.manual_seed(0)
        def net():
            return torch.nn.Sequential(*[torch.nn.Linear(4096, 4096) for _ in range(8)]).to(dev)
        ref, m = net(), net()
        m.load_state_dict(ref.state_dict())
        for layer in m:
            fully_shard(layer)
        fully_shard(m)
        torch.manual_seed(1)
        xs = torch.randn(16, 4096, device=dev)                       # same data on both ranks
        m(xs).pow(2).mean().backward()
        ref(xs).pow(2).mean().backward()
        g = m[0].weight.grad.full_tensor()
        out["fsdp_grad_max_abs_diff"] = (g - ref[0].weight.grad).abs().max().item()
        out["fsdp_ok"] = out["fsdp_grad_max_abs_diff"] < 1e-4
        out["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except Exception as e:
        out.update(fsdp_ok=False, fsdp_error=repr(e)[:300])
print("MIG2", json.dumps(out), flush=True)
dist.destroy_process_group()
EOF
for backend in nccl gloo; do
  PORT=$((29500 + RANDOM % 2000))
  for r in 0 1; do
    CUDA_VISIBLE_DEVICES=${UUIDS[$r]} RANK=$r WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT NCCL_DEBUG=WARN \
      timeout 600 $PY "$T/t.py" $backend > "$T/$backend.$r.log" 2>&1 &
  done
  wait
  for r in 0 1; do grep -h "^MIG2" "$T/$backend.$r.log" || { echo "MIG2 {\"check\": \"allreduce_$backend\", \"rank\": $r, \"ok\": false, \"no_output\": true}"; tail -n 15 "$T/$backend.$r.log"; }; done
done
rm -rf "$T"
echo "all done $(date +%T)"
