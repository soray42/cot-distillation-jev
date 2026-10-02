"""RLCD stage-2 data: a short calibration fine-tune that starts from the stage-1 model (TF-v4t).

  pool    python scripts/build_rlcd_data.py pool       -> data/rlcd/pool.jsonl
          20k final questions drawn round-robin over the public source families of data/student_v4t/train.jsonl (claim
          twins excluded), plus every item of the v3 reasoning families (K&K, JustLogic, ShARC, policy). The stage-1 model is evaluated on this file (eval-only job) to get its own distributions.
  build   python scripts/build_rlcd_data.py build --preds runs/TF-v4t-.../preds_pool.jsonl
          -> data/student_rlcd/{train,val}.jsonl
          - every pool item, target = (1 - a) one-hot(gold) + a stage-1 distribution (a = --retain, default 0.3):
            anchored to gold, softened by the model's own beliefs, which keeps stage 2 close to stage 1 (retention, as
            in self-distillation / replay) and lowers overconfidence;
          - every claim/negation twin of student_v4t (the yes/no bias fix), one-hot targets;
          - no tree nodes (stage 2 trains final readouts only).
          Train with --final teacher, --permute-final 1, --lambda-brier, a light --label-smoothing and a low lr.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = {"knights_knaves", "justlogic", "sharc", "returns", "expense", "subscription"}


def family(r: dict) -> str:
    s = r.get("source") or "?"
    return "/".join(s.split("/")[:2])


def pool(n: int, seed: int) -> None:
    rows = [json.loads(l) for l in open(ROOT / "data/student_v4t/train.jsonl")]
    rows = [r for r in rows if r.get("augment") != "claim_twin"]
    core = [r for r in rows if r["source"] in CORE]          # the v3 reasoning families: all kept
    by = collections.defaultdict(list)
    for r in rows:
        if r["source"] not in CORE:
            by[family(r)].append(r)
    rng = random.Random(seed)
    for v in by.values():
        rng.shuffle(v)
    out, i = [], 0
    while len(out) < n:                       # round-robin over families, so small families are fully represented
        layer = [v[i] for v in by.values() if i < len(v)]
        if not layer:
            break
        out += layer
        i += 1
    out = out[:n] + core
    path = ROOT / "data/rlcd/pool.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps({"item_id": r["item_id"], "source": r["source"], "prompt": r["prompt"],
                                "labels": r["labels"], "gold_label": r["gold_label"]}) + "\n")
    print(f"pool: {len(out)} items ({n} from {len(by)} public families, {len(core)} from the v3 reasoning families) "
          f"-> {path.relative_to(ROOT)}")


def build(preds: str, retain: float) -> None:
    train = {json.loads(l)["item_id"]: json.loads(l) for l in open(ROOT / "data/student_v4t/train.jsonl")}
    out, missing = [], 0
    for line in open(ROOT / "data/rlcd/pool.jsonl"):
        it = json.loads(line)
        out.append(dict(train[it["item_id"]], subqs=[], random_subqs=[]))
    stage1 = {}
    for line in open(preds):
        p = json.loads(line)
        stage1[p["item_id"]] = dict(zip(p["labels"], p["probs"]))
    rows = []
    for r in out:
        s = stage1.get(r["item_id"])
        if s is None:
            missing += 1
            continue
        gold = {L: float(L == r["gold_label"]) for L in r["labels"]}
        r["teacher"] = {L: (1 - retain) * gold[L] + retain * s.get(L, 0.0) for L in r["labels"]}
        r["augment"] = "rlcd_retain"
        rows.append(r)
    twins = [dict(r) for r in train.values() if r.get("augment") == "claim_twin"]
    rows += twins
    random.Random(0).shuffle(rows)
    d = ROOT / "data/student_rlcd"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(d / "val.jsonl", "w") as f:            # the stage-1 val (v3 families, with nodes) for --eval-subq
        for line in open(ROOT / "data/student_v4t/val.jsonl"):
            f.write(line)
    print(f"stage-2 train: {len(rows)} items ({len(rows) - len(twins)} retained finals, {len(twins)} claim twins, "
          f"{missing} pool items without a stage-1 prediction)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pool", "build"])
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--preds", default=None)
    ap.add_argument("--retain", type=float, default=0.3)
    args = ap.parse_args()
    if args.step == "pool":
        pool(args.n, args.seed)
    else:
        build(args.preds, args.retain)


if __name__ == "__main__":
    main()
