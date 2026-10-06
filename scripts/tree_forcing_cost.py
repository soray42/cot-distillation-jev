"""Per-epoch sequence and token totals of the Tree Forcing arms and their references, and an HPC time estimate.

For each arm one epoch of training views is built exactly as train_student.py builds them (seed 0), tokenised with
the Qwen3.5 tokenizer (left-truncated to --max-len as collate does) and split into micro-batches as the training loop
splits them (micro_bs consecutive sequences, or --token-budget length-sorted batches per update). The time model,
seconds per epoch = a x padded tokens + b x forward passes, is fitted on finished c4 runs (their s_per_step x steps /
epochs). CPU only; streams the data.

  python scripts/tree_forcing_cost.py --tokenizer <dir with tokenizer.json> --out notes/next_round/tree_forcing/cost.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.student import build_full_tree, build_group, token_budget_batches  # noqa: E402

ARMS = {   # name: (data, kind, kwargs)
    "TF cap10 v3f": ("data/student_v3f", "tree", dict(mode="true", cap=10)),
    "TFM cap10 v3f": ("data/student_v3f", "tree", dict(mode="matched", cap=10)),
    "G3 v3f": ("data/student_v3f", "group", {}),
    "G3 v3s": ("data/student_v3s", "group", {}),
    "TFE cap10 v3s": ("data/student_v3s", "tree", dict(mode="true", cap=10, eligible_only=True)),
    "TFL": ("data/student_v3sl", "tree", dict(mode="true", cap=0, view="local", sources={"knights_knaves"})),
    "TFA": ("data/student_v3sl", "tree", dict(mode="true", cap=0, view="ancestors", sources={"knights_knaves"})),
    "TFLM": ("data/student_v3sl", "tree", dict(mode="matched", cap=0, view="local", sources={"knights_knaves"})),
    "GFP": ("data/student_v3sl", "tree", dict(mode="plain", cap=0, view="problem", sources={"knights_knaves"})),
    "GFPS": ("data/student_v3sl", "tree", dict(mode="plain", cap=0, view="statements", sources={"knights_knaves"})),
    "TFLN": ("data/student_v3sl", "tree", dict(mode="plain", cap=0, view="local", sources={"knights_knaves"})),
}
CALIB = {"TF cap10 v3f": "results/hpc/c4v3f/TF-c4-Qwen3.5-2B-Base-student_v3f-s12",
         "G3 v3f": "results/hpc/w1/G3-c4-Qwen3.5-2B-Base-student_v3f-s12",
         "TFE cap10 v3s": "results/hpc/w1/TFE-c4-Qwen3.5-2B-Base-student_v3s-s12"}


def epoch_views(path: Path, kind: str, kw: dict, ipu: int = 32):
    """Yield (update index, item source, examples) for one epoch: problems in file order (the order does not change the
    totals), items_per_update problems per update."""
    rng = random.Random(0)
    with open(path) as f:
        for j, line in enumerate(f):
            it = json.loads(line)
            it.pop("rationale", None)
            if kind == "tree":
                ex = build_full_tree(it, rng=rng, final="teacher", subq_target="cot", **kw)
            else:
                ex = build_group(it, aux_kind="cot", k=2, aux_weight=0.5, rng=rng, final="teacher", subq_target="cot")
            yield j // ipu, it.get("source"), ex


def measure(tok, data: str, kind: str, kw: dict, max_len: int, micro_bs: int, budgets: list[int]) -> dict:
    from collections import Counter
    c = Counter()
    upd: list[int] = []
    cur = -1
    lens_by_kind = Counter()
    over = 0

    def close(lens):
        c["fwd_mbs"] += -(-len(lens) // micro_bs)
        c["pad_mbs"] += sum(max(lens[i:i + micro_bs]) * len(lens[i:i + micro_bs]) for i in range(0, len(lens), micro_bs))
        for b in budgets:
            bs = token_budget_batches(lens, b, 16)
            c[f"fwd_tb{b}"] += len(bs)
            c[f"pad_tb{b}"] += sum(len(x) * lens[x[0]] for x in bs)
    stats = {}
    for u, src, ex in epoch_views(ROOT / data / "train.jsonl", kind, dict(kw, stats=stats) if kind == "tree" else kw):
        if u != cur and upd:
            close(upd)
            upd = []
        cur = u
        enc = tok.encode_batch([e.text for e in ex], add_special_tokens=False)
        for e, t in zip(ex, enc):
            n = len(t.ids)
            over += n > max_len
            n = min(n, max_len)
            upd.append(n)
            c["seqs"] += 1
            c["tokens"] += n
            k = "final" if e.kind == "final" else ("node_kk" if src == "knights_knaves" else "node_other")
            c[f"seqs_{k}"] += 1
            lens_by_kind[k] += n
    if upd:
        close(upd)
    c["updates"] = cur + 1
    out = dict(c)
    out["over_max_len"] = over
    out["mean_tokens"] = {k: round(v / c[f"seqs_{k}"], 1) for k, v in lens_by_kind.items() if c[f"seqs_{k}"]}
    if stats:
        out["view_stats"] = stats
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="directory with tokenizer.json (Qwen3.5)")
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--micro-bs", type=int, default=2)
    ap.add_argument("--budgets", default="3072,4096")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))
    budgets = [int(x) for x in args.budgets.split(",") if x]
    res = {}
    for name in args.arms.split(","):
        data, kind, kw = ARMS[name]
        res[name] = measure(tok, data, kind, kw, args.max_len, args.micro_bs, budgets)
        print(name, json.dumps(res[name]), flush=True)
    calib = {}
    for name, run in CALIB.items():
        p = ROOT / run / "metrics.json"
        if name in res and p.exists():
            m = json.loads(p.read_text())
            h = m["history"][-1]
            calib[name] = {"s_per_epoch": h["s_per_step"] * h["step"] / m["args"]["epochs"], "steps": h["step"],
                           "epochs": m["args"]["epochs"], "s_per_step": h["s_per_step"]}
    out = {"arms": res, "calibration": calib}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
