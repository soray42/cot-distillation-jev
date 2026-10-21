"""Cue-free validation set for epoch selection (batch-14 confirmation): the 393 GSM8K validation finals of
data/student_gsm8k_tf_si/val.jsonl (the same finals as data/student_gsm8k_tf_vt/val.jsonl) with the 4 wrong options
replaced by final answers of OTHER problems, built exactly as scripts/build_mc_xprob.py builds the test sets
(exchangeable ratio-2 window on u = ln(x + 5), rank and letter quotas, per-item seeded draws), so that the epoch
chosen on it matches the cue-free test sets (gsm8kx / gsm1kx) rather than the cue-bearing val (QO-v2 / inter options).

Pool: 393 validation golds alone are too few for the ratio-2 window, so the distractor values are drawn from the
validation golds plus the GSM8K test golds (data/eval/gsm8k_mc_v2.jsonl, 1,317 items); neither is trained on.
The quotas run over the whole pool, so the validation subset's gold ranks and letters are balanced only on average
(the summary lists them).

Each row is the validation row (same fields, so the val loader reads it) with prompt, gold_label, options,
option_kinds and teacher rebuilt; the node fields (subqs, random_subqs, value_nodes) are dropped and worked_steps kept
(the TSF use detector reads them). New field xprob {src, window_ratio, gold_rank}.

  data/eval/gsm8k_val_xprob.jsonl
  data/eval/gsm8k_val_xprob.summary.json

  python3 scripts/build_val_xprob.py          # system python3; a second
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_mc_xprob import (LETTERS, build_options, convert, fmt, gold_of, md5, num, read, recipe_md5,  # noqa: E402
                            split_prompt)

VAL = ROOT / "data/student_gsm8k_tf_si/val.jsonl"
POOL = ROOT / "data/eval/gsm8k_mc_v2.jsonl"
OUT = ROOT / "data/eval/gsm8k_val_xprob.jsonl"
DROP = ("subqs", "random_subqs", "value_nodes", "verify_nodes")
SEED = 0


def build(val: list[dict], pool: list[dict], seed: int = SEED) -> tuple[list[dict], dict]:
    probs = [(it["item_id"], num(gold_of(it))) for it in val + pool]
    if len({p[0] for p in probs}) != len(probs):
        raise ValueError("duplicate item ids across validation and pool")
    plan = build_options(probs, seed, "gsm8k_val")
    md5_ = recipe_md5()
    golds = {it["item_id"]: gold_of(it) for it in val + pool}
    out, bad = [], []
    for it in val:
        base = {"item_id": it["item_id"], "prompt": it["prompt"], "gold_label": it["gold_label"]}
        x = convert(base, plan[it["item_id"]], md5_, seed)
        q0, o0 = split_prompt(it["prompt"])
        q1, o1 = split_prompt(x["prompt"])
        g = o0[it["gold_label"]]
        if q1 != q0 or o1[x["gold_label"]] != g or len({fmt(num(v)) for v in o1.values()}) != 5:
            bad.append(it["item_id"])
        for L, src in x["xprob"]["src"].items():
            if src == it["item_id"] or golds.get(src) != o1[L]:
                bad.append(f"{it['item_id']}:{L}")
        row = {k: v for k, v in it.items() if k not in DROP}
        row.update(prompt=x["prompt"], gold_label=x["gold_label"], options=[o1[L] for L in LETTERS],
                   option_kinds=[x["option_kinds"][L] for L in LETTERS],
                   teacher={L: float(L == x["gold_label"]) for L in LETTERS}, recipe=x["recipe"], xprob=x["xprob"])
        out.append(row)
    if bad:
        raise ValueError(f"audit failed: {bad[:10]}")
    ps = [plan[it["item_id"]] for it in val]

    def within2(it: dict) -> bool:
        g = num(gold_of(it))
        return all(0.5 * g - 1e-9 <= v <= 2 * g + 1e-9 for _, v in plan[it["item_id"]]["distractors"])

    src_kind = collections.Counter("val" if s.startswith("gsm8k-train") else "test"
                                   for p in ps for s, _ in p["distractors"])
    st = {"items": len(out), "pool": len(probs),
          "gold_rank": dict(sorted(collections.Counter(p["rank"] for p in ps).items())),
          "gold_letter": dict(sorted(collections.Counter(p["gold_letter"] for p in ps).items())),
          "window_ratio": {str(k): v for k, v in sorted(collections.Counter(round(p["window_ratio"]) for p in ps).items())},
          "within_0.5x_2x": sum(within2(it) for it in val),
          "distractor_source": dict(src_kind)}
    return out, st


def main() -> None:
    val, pool = read(VAL), read(POOL)
    out, st = build(val, pool)
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    summary = {"script": "scripts/build_val_xprob.py", "seed": SEED, "recipe_md5": recipe_md5(),
               "source_md5": {str(VAL.relative_to(ROOT)): md5(VAL), str(POOL.relative_to(ROOT)): md5(POOL)},
               "md5": {str(OUT.relative_to(ROOT)): md5(OUT)}, "stats": st}
    OUT.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
