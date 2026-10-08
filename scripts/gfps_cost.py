"""Cost of one GFP-S epoch (--tree-full plain --tree-view statements, optional --tree-cap) versus answer-only (A2) on a
student data set, from measured 4B speeds.

Views are built as train_student.py builds them (seed 0; build_full_tree / build_examples), tokenised with a Qwen3.5
tokenizer.json (left-truncated to --max-len as collate does) and grouped into updates as the full-tree loop groups them
(--items-per-update problems per update, all their views; A2: as many finals per update). Each update's sequences are
split into length-sorted token-budget micro-batches (student.token_budget_batches), giving per update the padded
tokens and forward passes. The time model, seconds per update = c + a x padded tokens + b x forward passes, is fitted
on the history of finished runs (metrics.json: per logged update pad_tok, fwd, upd_s; --calib run directories); the
wall-clock factor s_per_step / upd_s of the same runs (in-training evals, logging) gives the range. Those runs are what
they are (e.g. 4B, FSDP over two 4g.40gb slices, cpu-adamw, token budget 4096): the estimate holds for that setup;
--one-slice also reports the same work on one slice (forward/backward time doubled, update overhead kept).

  ~/.venvs/sev/bin/python scripts/gfps_cost.py --tokenizer <dir with tokenizer.json> --data data/student_v8/train.jsonl \
      --sources knights_knaves,numgen,... --caps 0,24,16 --calib results/hpc/l4b/*-Qwen3.5-4B-* --out cost.json
CPU only, streaming.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.student import build_examples, build_full_tree, token_budget_batches  # noqa: E402


def fit(calib: list[str]) -> dict:
    """Least squares upd_s = c + a pad_tok + b fwd over the runs' mean logged update, and the wall-clock factors."""
    import numpy as np
    rows = []
    for d in calib:
        m = json.loads((Path(d) / "metrics.json").read_text())
        h = [x for x in m["history"] if "pad_tok" in x and "upd_s" in x]
        if not h:
            continue
        rows.append({"run": Path(d).name, "pad_tok": float(np.mean([x["pad_tok"] for x in h])),
                     "fwd": float(np.mean([x["fwd"] for x in h])), "upd_s": float(np.mean([x["upd_s"] for x in h])),
                     "fb_s": float(np.mean([x["fb_s"] for x in h])), "s_per_step": m["history"][-1]["s_per_step"],
                     "steps": m["history"][-1]["step"], "fsdp": m.get("fsdp"),
                     "token_budget": m["args"].get("token_budget")})
    if len(rows) < 3:
        raise SystemExit(f"--calib: {len(rows)} usable runs (need >= 3 with pad_tok / fwd / upd_s in the history)")
    X = np.array([[1.0, r["pad_tok"], r["fwd"]] for r in rows])
    y = np.array([r["upd_s"] for r in rows])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    for r, p in zip(rows, X @ coef):
        r["fit_upd_s"] = round(float(p), 3)
        r["wall_factor"] = round(r["s_per_step"] / r["upd_s"], 3)
    f = [r["wall_factor"] for r in rows]
    return {"c": float(coef[0]), "a_per_pad_tok": float(coef[1]), "b_per_fwd": float(coef[2]),
            "max_abs_resid_s": round(float(np.max(np.abs(y - X @ coef))), 3),
            "wall_factor": [min(f), max(f)], "runs": rows}


def epoch(tok, path: Path, arm: str, sources: set | None, cap: int, ipu: int, budget: int, max_len: int,
          extra: int) -> dict:
    """Totals of one epoch (file order; the order changes no total) and per-update padded tokens / forward passes."""
    rng = random.Random(0)
    c = collections.Counter()
    by_src = collections.defaultdict(collections.Counter)
    upd_pad, upd_fwd, cur, cur_n = [], [], [], 0

    def close():
        lens = [x for x, _ in cur]
        bs = token_budget_batches(lens, budget, 16)
        upd_pad.append(sum(len(b) * lens[b[0]] for b in bs))
        upd_fwd.append(len(bs))

    with open(path) as f:
        for line in f:
            it = json.loads(line)
            it.pop("rationale", None)
            if arm == "A2":
                ex = [e for e in build_examples(it, final="teacher", subq="none", subq_target="cot", lambda_sub=0.0,
                                                rng=rng) if e.kind == "final"]
            else:
                ex = build_full_tree(it, mode="plain", rng=rng, cap=cap, view="statements", sources=sources,
                                     final="teacher", subq_target="cot")
            if not ex:
                continue
            if cur_n == ipu:
                close()
                cur, cur_n = [], 0
            cur_n += 1
            src = str(it.get("source"))
            for e, t in zip(ex, tok.encode_batch([e.text for e in ex], add_special_tokens=False)):
                n = len(t.ids) + extra
                c["over_max_len"] += n > max_len
                n = min(n, max_len)
                cur.append((n, e.kind))
                c["seqs"] += 1
                c["tokens"] += n
                c[f"seqs_{e.kind}"] += 1
                c[f"tokens_{e.kind}"] += n
                by_src[src][f"seqs_{e.kind}"] += 1
                by_src[src][f"tokens_{e.kind}"] += n
            if arm != "A2":
                n_nodes = sum(1 for sq in it.get("subqs") or [] if "id" in sq and
                              (sq.get("p_cot") is not None or sq.get("truth") is not None or
                               sq.get("p_fresh") is not None))
                if sources is None or it.get("source") in sources:
                    by_src[src]["tree_nodes"] += n_nodes
    if cur:
        close()
    out = dict(c)
    out.update({"updates": len(upd_pad), "pad_tok": sum(upd_pad), "fwd": sum(upd_fwd), "upd_pad": upd_pad,
                "upd_fwd": upd_fwd, "by_source": {k: dict(v) for k, v in sorted(by_src.items())}})
    return out


def hours(e: dict, model: dict, one_slice: bool = False) -> dict:
    var = [model["a_per_pad_tok"] * p + model["b_per_fwd"] * f for p, f in zip(e["upd_pad"], e["upd_fwd"])]
    k = 2.0 if one_slice else 1.0
    s = sum(model["c"] + k * v for v in var)
    lo, hi = model["wall_factor"]
    return {"s_per_update": round(s / max(1, e["updates"]), 2), "h": round(s / 3600, 2),
            "h_wall": [round(s * lo / 3600, 2), round(s * hi / 3600, 2)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="directory with tokenizer.json (Qwen3.5)")
    ap.add_argument("--data", required=True, help="train.jsonl")
    ap.add_argument("--sources", required=True, help="--tree-sources of the GFP-S arm (comma list)")
    ap.add_argument("--caps", default="0", help="--tree-cap values to cost (0 = every node)")
    ap.add_argument("--calib", nargs="+", required=True, help="finished run directories (metrics.json)")
    ap.add_argument("--items-per-update", type=int, default=32)
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--extra-tokens", type=int, default=0, help="tokens added to every sequence (e.g. a chat template)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))
    model = fit(args.calib)
    print("time model", json.dumps({k: v for k, v in model.items() if k != "runs"}), flush=True)
    sources = {x for x in args.sources.split(",") if x}
    res = {"data": args.data, "sources": sorted(sources), "items_per_update": args.items_per_update,
           "budget": args.budget, "max_len": args.max_len, "extra_tokens": args.extra_tokens, "model": model, "arms": {}}
    arms = [("A2", 0)] + [(f"GFPS cap{c}" if c else "GFPS", c) for c in (int(x) for x in args.caps.split(","))]
    for name, cap in arms:
        e = epoch(tok, Path(args.data), "A2" if name == "A2" else "GFPS", sources, cap, args.items_per_update,
                  args.budget, args.max_len, args.extra_tokens)
        e["time_measured_setup"] = hours(e, model)
        e["time_one_slice"] = hours(e, model, one_slice=True)
        e.pop("upd_pad")
        e.pop("upd_fwd")
        res["arms"][name] = e
        print(name, json.dumps({k: v for k, v in e.items() if k != "by_source"}), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
