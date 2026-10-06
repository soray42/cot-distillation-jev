"""Cost model: hours per training run of Gemma 4 12B (full parameters, CPUOffloadAdamW) on one 4g.40gb MIG slice.

Per update with M micro-batches (forward + backward each) of padded tokens T_m:
  micro-batch time = max(T_m / R + b,  D2H,  ACC)      D2H = bf16 gradient bytes / PCIe GB/s, ACC = 10 B/param / CPU GB/s
                                                         (both overlap backward; they bind only when the GPU work is short)
  update time      = sum over micro-batches + STEP + tail  STEP = 46 B/param / CPU GB/s (serial: norm, clip, AdamW, cast)
  run hours        = epochs x updates/epoch x update time + evals
R = 12B forward+backward tokens/s with activation checkpointing (8N FLOP/token, N = 10.90B non-embedding params).

Inputs: notes/next_round/tree_forcing/cost.json (per-epoch sequences, padded tokens and forward passes of every arm on
v3f/v3s/v3sl at MBS 2 and token budgets 3072/4096, Qwen3.5 tokenizer; Gemma's averages 3.97 chars/token on v3f finals,
about the same) and data/student_v3f/train.jsonl (A2 finals; padding of larger fixed micro-batches simulated by
random batches of real lengths). Rate scenarios (tok/s, s per forward):
  1090, 0.3  the 2B time model's marginal cost (122 s per M padded tokens x 0.946, cost.json fit) x 7.94
  790,  0.3  the 2B dense ceiling on this slice (6,250 tok/s incl. its optimizer step) / 7.94
  340,  0    2B G3-c4's all-in rate (22.1 M padded tokens in 2.6 - 0.3 h at MBS 2) / 7.94: reproduces the 2B
             calibration exactly, but carries 2B's small-batch overheads, so it is pessimistic for large micro-batches.
The smoke (hpc/mem12b.sbatch) measures R ("baseline" tok_s), the D2H floor and STEP; rerun with --rate/--pcie/--cpu-gbs set from it.

  python scripts/cost12b.py [--rate 790 --b 0.3] [--pcie 16 --cpu-gbs 60] [--md]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N_TEXT = 11_907_350_320          # text model params (tied embedding included): bytes moved per update
N_NONEMB = 10_900_717_360        # FLOP-relevant params
N_2B_NONEMB = 1.373e9
UPD_PER_EPOCH = 226              # v3f: 7,207 problems / 32 per update
CHARS_PER_TOKEN = 3.97           # v3f finals: 866 chars (prompt + "\n\nAnswer:") = 218 tokens (cost.json)


def offload_floor(pcie_gbs: float, cpu_gbs: float, acc_gbs: float | None = None) -> float:
    """Seconds per micro-batch below which the gradient D2H or the CPU fp32 accumulation binds (acc_gbs: the
    accumulation's own GB/s when measured, else cpu_gbs)."""
    return max(N_TEXT * 2 / 1e9 / pcie_gbs, N_TEXT * 10 / 1e9 / (acc_gbs or cpu_gbs))


def step_s(cpu_gbs: float) -> float:
    return N_TEXT * 46 / 1e9 / cpu_gbs


COST_JSON = ROOT / "notes/next_round/tree_forcing/cost.json"
TRAIN = ROOT / "data/student_v3f/train.jsonl"


def a2_lengths(train: Path = TRAIN) -> list[int]:
    out = []
    for line in open(train):
        if line.strip():
            out.append(round((len(json.loads(line)["prompt"]) + 9) / CHARS_PER_TOKEN))
    return out


def padded_random(lengths: list[int], mb: int, per_update: int, seed: int = 0) -> tuple[float, float]:
    """(padded tokens, real tokens) per epoch when each update's per_update sequences (shuffled) are cut into
    consecutive micro-batches of mb, each padded to its longest sequence (the non-token-budget loop)."""
    rng = random.Random(seed)
    order = lengths[:]
    rng.shuffle(order)
    pad = real = 0
    for u in range(0, len(order) - per_update + 1, per_update):
        upd = order[u:u + per_update]
        for k in range(0, per_update, mb):
            chunk = upd[k:k + mb]
            pad += max(chunk) * len(chunk)
            real += sum(chunk)
    n_upd = len(order) // per_update
    scale = UPD_PER_EPOCH / n_upd
    return pad * scale, real * scale


def arms(cost_json: Path = COST_JSON, train: Path = TRAIN) -> list[dict]:
    """Per-epoch padded tokens and forward passes for each arm / batching (epochs as run at 2B)."""
    c = json.loads(Path(cost_json).read_text())["arms"]
    out = []
    L = a2_lengths(Path(train))
    for mb, ga in ((2, 16), (8, 4), (16, 2)):
        pad, real = padded_random(L, mb, 32)
        out.append(dict(arm="A2-c8", batching=f"mb {mb} x GA {ga}", epochs=8, pad=pad, fwd=UPD_PER_EPOCH * ga,
                        tokens=real, resumable=False))
    g3 = c["G3 v3f"]
    out.append(dict(arm="G3-c4", batching="mb 2 x GA 48 (as run)", epochs=4, pad=g3["pad_mbs"], fwd=g3["fwd_mbs"],
                    tokens=g3["tokens"], resumable=False))
    for mb in (8, 16, 32):
        # finals-based padding factor of random micro-batches, relative to MBS 2 (views have similar spread)
        f = padded_random(L, mb, 96)[0] / padded_random(L, 2, 96)[0]
        out.append(dict(arm="G3-c4", batching=f"mb {mb} x GA {96 // mb}", epochs=4, pad=g3["pad_mbs"] * f,
                        fwd=UPD_PER_EPOCH * 96 // mb, tokens=g3["tokens"], resumable=False))
    for name, key in (("TF-c4 (cap 10, v3f)", "TF cap10 v3f"), ("TFE-c4 (v3s)", "TFE cap10 v3s"),
                      ("TFL-c4", "TFL"), ("TFLM-c4", "TFLM"), ("TFA-c4", "TFA"), ("GFP-c4", "GFP")):
        a = c[key]
        for tb in ("mbs", "tb3072", "tb4096"):
            out.append(dict(arm=name, batching={"mbs": "mb 2", "tb3072": "budget 3072", "tb4096": "budget 4096"}[tb],
                            epochs=4, pad=a[f"pad_{tb}"], fwd=a[f"fwd_{tb}"], tokens=a["tokens"], resumable=True))
    return out


def hours(a: dict, rate: float, b: float, pcie: float, cpu: float, eval_h: float, acc: float | None = None) -> dict:
    M = a["fwd"] / UPD_PER_EPOCH
    t_gpu = a["pad"] / a["fwd"] / rate + b                   # mean micro-batch GPU time
    floor = offload_floor(pcie, cpu, acc)
    upd = M * max(t_gpu, floor) + step_s(cpu) + 0.5
    n_upd = a["epochs"] * UPD_PER_EPOCH
    train_h = n_upd * upd / 3600
    gpu_only_h = n_upd * M * t_gpu / 3600
    return {"M": M, "tok_per_mb": a["pad"] / a["fwd"], "t_gpu": t_gpu, "bound": "offload" if floor > t_gpu else "GPU",
            "s_per_update": upd, "train_h": train_h, "gpu_only_h": gpu_only_h, "total_h": train_h + eval_h}


def from_log(path: str) -> dict:
    """Measured inputs from the smoke's MEM12B lines (hpc/mem12b.sbatch)."""
    rows = []
    for line in open(path, errors="replace"):
        if line.startswith("MEM12B "):
            try:
                rows.append(json.loads(line[7:]))
            except json.JSONDecodeError:
                pass
    out = {}
    base = [r for r in rows if r.get("part") == "baseline" and r.get("status") == "ok"]
    if base:
        best = max(base, key=lambda r: r["tok_s"])
        out.update(rate=best["tok_s"], rate_setting=best["setting"])
    bench = [r for r in rows if r.get("part") == "bench" and isinstance(r.get("d2h_gbs"), (int, float))]
    if bench:
        out["pcie"] = bench[-1]["d2h_gbs"]
        acc = bench[-1].get("acc_add_bf16_gparam_s")
        if isinstance(acc, (int, float)) and acc > 0:
            out["acc_gbs"] = round(acc * 10, 1)        # 10 B/param: read bf16 stage + read/write fp32 accumulator
    mem = [r for r in rows if r.get("part") == "mem" and r.get("step_s")]
    if mem:
        st = sum(r["step_s"] + r.get("clip_s", 0) for r in mem) / len(mem)
        out.update(step_s=round(st, 2), cpu_gbs=round(N_TEXT * 46 / 1e9 / st, 1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, nargs="+", default=[1090.0, 790.0, 340.0])
    ap.add_argument("--b", type=float, nargs="+", default=[0.3, 0.3, 0.0], help="s per forward pass, one per --rate")
    ap.add_argument("--pcie", type=float, default=16.0)
    ap.add_argument("--cpu-gbs", type=float, default=60.0)
    ap.add_argument("--eval-h", type=float, default=2.4, help="12B evals per run (2B c4 grid 0.3 h x 7.94)")
    ap.add_argument("--wall-h", type=float, default=24.0)
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--acc-gbs", type=float, default=None, help="CPU fp32 accumulation GB/s (default: --cpu-gbs)")
    ap.add_argument("--no-resume", action="store_true",
                    help="no resume chains (a 12B optimizer checkpoint does not fit the 100 GB home quota): runs over "
                         "the wall are NO instead of 'N jobs'")
    ap.add_argument("--from-log", help="a log with MEM12B lines: rate = baseline tok/s (largest ok setting), "
                                       "PCIe = bench d2h, CPU GB/s = 46 B/param / measured step_s")
    ap.add_argument("--cost-json", default=str(COST_JSON), help="tree_forcing/cost.json (per-arm epoch totals)")
    ap.add_argument("--train", default=str(TRAIN), help="data/student_v3f/train.jsonl (A2 final lengths)")
    a = ap.parse_args(argv)
    if a.from_log:
        m = from_log(a.from_log)
        print("measured:", json.dumps(m))
        if m.get("rate"):
            a.rate, a.b = [m["rate"]], [0.0]
        a.pcie = m.get("pcie", a.pcie)
        a.cpu_gbs = m.get("cpu_gbs", a.cpu_gbs)
        a.acc_gbs = m.get("acc_gbs", a.acc_gbs)
    missing = [f for f in (a.cost_json, a.train) if not Path(f).exists()]
    if missing:
        print(f"cost table skipped: missing {missing} (pass --cost-json / --train)")
        return
    floor, st = offload_floor(a.pcie, a.cpu_gbs, a.acc_gbs), step_s(a.cpu_gbs)
    print(f"FLOP ratio 12B/2B = {N_NONEMB / N_2B_NONEMB:.2f}; offload floor {floor:.2f} s per micro-batch "
          f"(PCIe {a.pcie} GB/s, CPU {a.cpu_gbs} GB/s); CPU step {st:.1f} s per update; evals {a.eval_h} h")
    hdr = ["arm", "batching", "ep", "padded M tok/ep", "mb/update", "tok/mb"]
    for r in a.rate:
        hdr += [f"s/upd @{r:g}", f"bound @{r:g}", f"h @{r:g}"]
    hdr += ["fits 24 h (per rate; NO = loop cannot resume)"]
    rows = []
    for arm in arms(a.cost_json, a.train):
        row = [arm["arm"], arm["batching"], str(arm["epochs"]), f"{arm['pad'] / 1e6:.2f}"]
        hs = [hours(arm, r, b, a.pcie, a.cpu_gbs, a.eval_h, a.acc_gbs) for r, b in zip(a.rate, a.b)]
        row += [f"{hs[0]['M']:.1f}", f"{hs[0]['tok_per_mb']:.0f}"]
        for h in hs:
            row += [f"{h['s_per_update']:.0f}", h["bound"], f"{h['total_h']:.1f}"]
        cap = a.wall_h - 1                      # 1 h of the wall for load, smoke and the final save
        fits = []
        for h in hs:
            fits.append("ok" if h["total_h"] <= cap else
                        (f"{-(-h['total_h'] // cap):.0f} jobs" if arm["resumable"] and not a.no_resume else "NO"))
        row.append(" / ".join(fits))
        rows.append(row)
    if a.md:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        for r in rows:
            print("| " + " | ".join(r) + " |")
    else:
        w = [max(len(x) for x in col) for col in zip(hdr, *rows)]
        for r in [hdr] + rows:
            print("  ".join(x.ljust(n) for x, n in zip(r, w)))


if __name__ == "__main__":
    main()
