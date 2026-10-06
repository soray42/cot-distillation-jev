"""Helpers of the two-slice smoke hpc/fsdp12b.sbatch (notes/next_round/scale12b/fsdp2.md §6). Every result is one
"FSDP12B {json}" line.

  bench      (torchrun, 2 ranks) all_gather / reduce_scatter / all_reduce GB/s between the two slices (bf16, --gib per
             rank), and the per-micro-batch FSDP communication this implies for the 12B (forward all-gather + backward
             re-gather + reduce-scatter of 11.907B bf16 parameters, half per rank)
  readout    (one process, one slice) zero-shot letter readout of the 12B: accuracy, letter probability mass, raw letter
             logits, BOS / no-BOS (scripts/mem12b.py readout_items) on the first --n items of each --files file
  compare    a single-GPU run against a --fsdp run of the same arm and seed (metrics.json, preds_<set>.jsonl): per-update
             upd_loss differences, final accuracy of both, prediction agreement
  summarize  one --fsdp training run (10 updates of the 12B): per-rank peak GiB, s/update split (fwd+bwd / est.
             communication / clip / optimizer), padded tok/s, host RSS; OOM or failure from its log
  recommend  reads the FSDP12B lines of a log, projects hours per arm (226 updates per epoch, cost12b epochs) and prints
             the recommendation line

  python scripts/fsdp12b.py compare --ref runs/x-single --fsdp runs/x-fsdp --label G3/cpu-adamw
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

N_TEXT = 11_907_350_320          # Gemma 4 12B text parameters (bf16 bytes moved by FSDP per pass: 2 x this)
UPD_PER_EPOCH = 226
EPOCHS = {"A2": 12, "G3": 4, "TFL": 4, "TFLM": 4, "TFLN": 4, "GFPS": 4, "GFP": 4, "TFA": 4}   # A2-c12, the rest c4
WALL_H = 23.0                    # 24 h stud wall minus load, smoke and the final save


def emit(row: dict) -> None:
    print("FSDP12B", json.dumps(row), flush=True)


# ------------------------------------------------------------------------------------------------------------ bench
def cmd_bench(a) -> None:
    import torch
    import torch.distributed as dist
    from cotdistill import fsdp as F
    fs = F.init(quiet=True)
    w = fs.world
    n = int(a.gib * 2 ** 30 // 2)                       # bf16 elements per rank
    shard = torch.ones(n, dtype=torch.bfloat16, device=fs.device)
    full = torch.empty(n * w, dtype=torch.bfloat16, device=fs.device)

    def timed(fn, reps=5):
        fn()
        torch.cuda.synchronize() if fs.device.type == "cuda" else None
        t = time.time()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize() if fs.device.type == "cuda" else None
        return (time.time() - t) / reps
    sent = n * 2 * (w - 1)                               # bytes each rank sends per collective
    t_ag = timed(lambda: dist.all_gather_into_tensor(full, shard))
    t_rs = timed(lambda: dist.reduce_scatter_tensor(shard, full))
    t_ar = timed(lambda: dist.all_reduce(shard))
    row = {"part": "bench", "world": w, "gib_per_rank": a.gib, "allgather_gbs": round(sent / t_ag / 1e9, 2),
           "reducescatter_gbs": round(sent / t_rs / 1e9, 2), "allreduce_algbw_gbs": round(n * 2 / t_ar / 1e9, 2)}
    per_rank = N_TEXT * 2 / w * (w - 1)                  # bytes a rank sends per all-gather of the 12B
    row["comm_s_per_microbatch_12b"] = round(2 * per_rank / (sent / t_ag) + per_rank / (sent / t_rs), 2)
    if fs.rank == 0:
        emit(row)
        if a.out:
            Path(a.out).write_text(json.dumps(row))
    dist.destroy_process_group()


# ------------------------------------------------------------------------------------------------------------ readout
def cmd_readout(a) -> None:
    import mem12b
    import torch
    mem12b.emit = lambda row: emit(dict(row, part="readout_12b") if row.get("part") == "readout" else row)
    t0 = time.time()
    model, info = mem12b.load_text_only(a.model, torch.bfloat16, a.attn)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    emit({"part": "load_12b", "class": type(model).__name__, "load_s": round(time.time() - t0, 1),
          "params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 4), "softcap": mem12b.softcap_of(model),
          "bos_prefix": mem12b.special_prefix(tok), "unexpected_keys": len(info.get("unexpected_keys", []) or [])})
    mem12b.part_readout(model, tok, [f for f in a.files.split(",") if f], a.n, a.max_len)


# ------------------------------------------------------------------------------------------------------------ compare
def _metrics(d) -> dict:
    return json.loads((Path(d) / "metrics.json").read_text())


def _preds(d, name) -> list[dict]:
    p = Path(d) / f"preds_{name}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def cmd_compare(a) -> None:
    row = {"part": "equivalence", "label": a.label}
    try:
        r, f = _metrics(a.ref), _metrics(a.fsdp)
    except (OSError, ValueError) as e:
        emit(dict(row, status="failed", error=repr(e)[:300]))
        return
    lr = [h.get("upd_loss", h["loss"]) for h in r["history"]]
    lf = [h.get("upd_loss", h["loss"]) for h in f["history"]]
    d = [abs(x - y) for x, y in zip(lr, lf)]
    row.update(status="ok", updates=[len(lr), len(lf)], loss_absdiff_max=round(max(d), 6) if d else None,
               loss_absdiff_mean=round(sum(d) / len(d), 6) if d else None,
               loss_reldiff_max=round(max(x / max(abs(y), 1e-9) for x, y in zip(d, lr)), 5) if d else None,
               loss_ref=[round(x, 4) for x in lr], loss_fsdp=[round(x, 4) for x in lf])
    for name in r.get("eval", {}):
        er, ef = r["eval"].get(name, {}), f["eval"].get(name, {})
        if "acc" in er:
            pr, pf = _preds(a.ref, name), _preds(a.fsdp, name)
            agree = [int(max(range(len(p["probs"])), key=p["probs"].__getitem__) ==
                         max(range(len(q["probs"])), key=q["probs"].__getitem__)) for p, q in zip(pr, pf)]
            row[name] = {"acc_ref": er.get("acc"), "acc_fsdp": ef.get("acc"), "nll_ref": er.get("nll"),
                         "nll_fsdp": ef.get("nll"), "same_order": [p["item_id"] for p in pr] == [q["item_id"] for q in pf],
                         "agreement": round(sum(agree) / len(agree), 4) if agree else None,
                         "prob_absdiff_max": round(max((abs(x - y) for p, q in zip(pr, pf)
                                                        for x, y in zip(p["probs"], q["probs"])), default=0.0), 5)}
    emit(row)


# ------------------------------------------------------------------------------------------------------------ summarize
def cmd_summarize(a) -> None:
    row = {"part": "train_12b", "arm": a.arm, "optim": a.optim, "token_budget": a.budget}
    log = Path(a.log).read_text(errors="replace") if a.log and Path(a.log).exists() else ""
    mpath = Path(a.out) / "metrics.json"
    hist = []
    if mpath.exists():
        m = json.loads(mpath.read_text())
        hist = m.get("history", [])
        row["eval"] = {k: {kk: v.get(kk) for kk in ("n", "acc", "nll")} for k, v in m.get("eval", {}).items()
                       if isinstance(v, dict) and "acc" in v}
    else:                               # crashed or still running: the json rows printed so far
        for line in log.splitlines():
            if line.startswith("{\"step\""):
                try:
                    hist.append(json.loads(line))
                except ValueError:
                    pass
    if re.search(r"OutOfMemoryError|CUDA out of memory", log):
        row["status"] = "OOM"
        m = re.search(r"Tried to allocate ([0-9.]+ [GM]iB)", log)
        row["oom"] = m.group(1) if m else True
    elif not mpath.exists():
        row["status"] = "failed"
        errs = [l.strip() for l in log.splitlines() if re.search(r"(Error|Exception|SystemExit)\b", l)
                and "ChildFailedError" not in l and "elastic" not in l]
        row["error"] = (errs[-1] if errs else " | ".join([l for l in log.splitlines() if l.strip()][-3:]))[-400:]
    else:
        row["status"] = "ok"
    rows = hist[1:] or hist             # update 1 includes lazy state allocation and compile
    if rows:
        mean = lambda k: round(sum(r.get(k, 0.0) for r in rows) / len(rows), 3)
        row.update(updates=len(hist), upd_s=mean("upd_s"), fb_s=mean("fb_s"), clip_s=mean("clip_s"),
                   opt_s=mean("opt_s"), fwd_per_update=mean("fwd"), pad_fwd=mean("pad_fwd"),
                   pad_tok_per_update=mean("pad_tok"), pad_tok_per_s=mean("pad_tok_per_s"),
                   update1_s=round(hist[0].get("upd_s", 0.0), 2), loss=[round(r["loss"], 4) for r in hist])
        if any("d2h_wait_s" in r for r in rows):
            row.update(cpu_step_s=mean("step_s"), d2h_wait_s=mean("d2h_wait_s"))
        for k in ("peak_gib_ranks", "reserved_gib_ranks", "rss_peak_gib_ranks"):
            vals = [r[k] for r in hist if k in r]
            if vals:
                row[k] = [max(v[i] for v in vals) for i in range(len(vals[0]))]
        if a.bench and Path(a.bench).exists():
            b = json.loads(Path(a.bench).read_text())
            per_mb = b.get("comm_s_per_microbatch_12b")
            if per_mb:
                n_mb = row["fwd_per_update"] / b.get("world", 2)
                row["comm_s_est"] = round(n_mb * per_mb, 2)           # if nothing overlapped
                row["compute_s_lower"] = round(max(0.0, row["fb_s"] - row["comm_s_est"]), 2)
    emit(row)


# ------------------------------------------------------------------------------------------------------------ recommend
def cmd_recommend(a) -> None:
    rows = []
    for line in open(a.log, errors="replace"):
        if line.startswith("FSDP12B "):
            try:
                rows.append(json.loads(line[8:]))
            except ValueError:
                pass
    train = [r for r in rows if r.get("part") == "train_12b"]
    ok = [r for r in train if r.get("status") == "ok" and r.get("upd_s")]
    best, proj = {}, {}
    for r in ok:
        if r["arm"] not in best or r["upd_s"] < best[r["arm"]]["upd_s"]:
            best[r["arm"]] = r
    for arm, r in best.items():
        h = EPOCHS.get(arm, 4) * UPD_PER_EPOCH * r["upd_s"] / 3600 + a.eval_h
        proj[arm] = {"optim": r["optim"], "token_budget": r["token_budget"], "s_per_update": r["upd_s"],
                     "hours": round(h, 1), "fits_24h": h <= WALL_H,
                     "peak_gib_max": max(r.get("peak_gib_ranks") or [0])}
    oom = sorted({f"{r['arm']}/{r['optim']}/{r['token_budget']}" for r in train if r.get("status") == "OOM"})
    # The fp32-shards pair is the sum-semantics check (same optimizer, fp32 weights on both sides): strict. The bf16
    # pairs differ by design (stochastic rounding with different element -> RNG mappings; bf16 vs fp32 weights) and
    # only have to track (loose tolerance).
    eq = [r for r in rows if r.get("part") == "equivalence"]
    eq_status = {r.get("label", "?"): r.get("status") == "ok" and (r.get("loss_reldiff_max") or 0) <
                 (a.eq_tol if "fp32master" in r.get("label", "") else a.eq_tol_bf16) for r in eq}
    eq_ok = bool(eq) and all(eq_status.values())
    setups = sorted({r["optim"] for r in ok})
    if not ok:
        text = "no 12B setting trained: do not submit 12B runs; read the OOM / error lines above"
    else:
        choice = "cpu-adamw" if "8bit-sr" not in setups else min(
            setups, key=lambda s: sum(r["upd_s"] for r in ok if r["optim"] == s) / max(1, sum(r["optim"] == s for r in ok)))
        text = (f"12B on two slices: FSDP_OPTIM={choice}; per arm: " +
                ", ".join(f"{k} TOKBUDGET={v['token_budget']} {v['s_per_update']} s/update ~{v['hours']} h"
                          f"{'' if v['fits_24h'] else ' (over 24 h)'}" for k, v in sorted(proj.items())))
        if not eq_ok:
            text += "; CHECK the 2B equivalence lines first (missing or loss differences above tolerance)"
    emit({"part": "recommendation", "equivalence_ok": eq_ok, "equivalence": eq_status, "oom": oom, "projection": proj,
          "text": text})
    print("RECOMMENDATION:", text, flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--gib", type=float, default=1.0)
    b.add_argument("--out", default="")
    r = sub.add_parser("readout")
    r.add_argument("--model", required=True)
    r.add_argument("--files", default="data/eval/kk_heldout.jsonl,data/eval/bbh.jsonl,data/eval/jevbench_public.jsonl")
    r.add_argument("--n", type=int, default=50)
    r.add_argument("--max-len", type=int, default=1536)
    r.add_argument("--attn", default=None)
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--fsdp", required=True)
    c.add_argument("--label", default="")
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    s.add_argument("--log", default="")
    s.add_argument("--arm", required=True)
    s.add_argument("--optim", required=True)
    s.add_argument("--budget", type=int, required=True)
    s.add_argument("--bench", default="")
    m = sub.add_parser("recommend")
    m.add_argument("--log", required=True)
    m.add_argument("--eval-h", type=float, default=1.5, help="12B final evals on two slices (h per run)")
    m.add_argument("--eq-tol", type=float, default=0.02,
                   help="2B fp32-shards pair (the sum-semantics check): max relative per-update loss difference")
    m.add_argument("--eq-tol-bf16", type=float, default=0.10,
                   help="2B bf16 pairs (stochastic rounding / bf16 vs fp32 weights): max relative loss difference")
    a = ap.parse_args(argv)
    {"bench": cmd_bench, "readout": cmd_readout, "compare": cmd_compare, "summarize": cmd_summarize,
     "recommend": cmd_recommend}[a.cmd](a)


if __name__ == "__main__":
    main()
