"""Build training items (for teacher labelling) and held-out eval sets from the raw data.

  bash scripts/fetch_data.sh && python3 scripts/build_items.py

data/bench/train_kk.jsonl, train_jl.jsonl   training items (labelled by scripts/run_pilot.py)
data/eval/kk_heldout.jsonl                  K&K, 4-12 people x 50 (11-12 are beyond the training range)
data/eval/jl_heldout.jsonl                  JustLogic validation split, depth 1-7 x 150
data/eval/jevbench_public.jsonl             JevBench public tiers (evaluation only)
data/eval/typed_decisions_test.jsonl        Typed Decisions test, one record per decision (evaluation only)
data/eval/bbeh.jsonl                        BBEH closed-answer tasks, 1,920 items (evaluation only; hard for the teacher too)
data/eval/bbeh_sub.jsonl                    the first 50 per task in a fixed shuffle, for teacher (System 2) runs
data/bench/train_{policy,sharc,folio}.jsonl second training batch (decision formats): 500 / 400 / 300
data/eval/{policy_heldout,sharc_dev,folio_val}.jsonl  their in-family held-out sets (policy: unseen domains;
                                            ShARC: official dev, no shared rules; FOLIO: official validation)
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill import evalsets as E  # noqa: E402
from cotdistill import sources as S  # noqa: E402
from cotdistill.policygen import HELDOUT_DOMAINS, TRAIN_DOMAINS, generate  # noqa: E402

RAW = ROOT / "data/raw"
JL_TRAIN_DEPTHS = {1: 100, 2: 100, 3: 100, 4: 200, 5: 200, 6: 200, 7: 200}


def to_eval(r: dict, src: str) -> dict:
    letters = [chr(65 + i) for i in range(len(r["label_order"]))]
    return {"item_id": r["item_id"], "source": src, "group": f"{src}/d{r['depth']}", "type": "choice",
            "prompt": r["prompt"], "labels": letters, "label_order": letters, "gold_label": r["gold_label"],
            "label_names": dict(zip(letters, r["label_order"])), "gold_probs": None, "depth": r["depth"]}


def write(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"{path.relative_to(ROOT)}: {len(recs)}")


def main() -> None:
    kk = [r for n in range(4, 11) for r in S.knights_knaves(158, seed=100 + n, people_range=(n, n))]
    random.Random(0).shuffle(kk)
    got, jl = collections.Counter(), []
    for r in S.justlogic(str(RAW / "justlogic/train_dataset.csv"), seed=100):
        if got[r["depth"]] < JL_TRAIN_DEPTHS[r["depth"]]:
            jl.append(r)
            got[r["depth"]] += 1
    random.Random(1).shuffle(jl)

    kk_eval = [r for n in range(4, 13) for r in S.knights_knaves(50, seed=9000 + n, people_range=(n, n))]
    for r in kk_eval:
        r["item_id"] = r["item_id"].replace("kk-", "kkeval-")
    jl_eval = S.justlogic(str(RAW / "justlogic/validate_dataset.csv"), seed=9000)
    # second training batch (decision-format sources): policy generator, ShARC rules, FOLIO logic
    policy = generate(500, TRAIN_DOMAINS, seed=101)
    sharc = S.sharc(str(RAW / "sharc/sharc_train.json"), n=400, seed=1)
    folio = S.folio(str(RAW / "folio/folio_v2_train.jsonl"), n=300, seed=1)
    evals = {
        "kk_heldout": [to_eval(r, "kk") for r in kk_eval],
        "jl_heldout": [to_eval(r, "jl") for r in jl_eval],
        "jevbench_public": E.jevbench({t: str(RAW / f"jevbench/{t}.jsonl") for t in ("original", "easy", "hard")}),
        "typed_decisions_test": E.typed_decisions(str(RAW / "typed_decisions/test.parquet")),
        "bbeh": E.bbeh(str(RAW / "bbeh"), seed=1),
        "policy_heldout": [to_eval(r, "policy") for r in generate(300, HELDOUT_DOMAINS, seed=9001)],
        "sharc_dev": [to_eval(r, "sharc") for r in S.sharc(str(RAW / "sharc/sharc_dev.json"), n=300, seed=9000)],
        "folio_val": [to_eval(r, "folio") for r in S.folio(str(RAW / "folio/folio_v2_validation.jsonl"), seed=9000)],
        "bbeh_sub": E.bbeh(str(RAW / "bbeh"), n_per_task=50, seed=1),
    }
    train_prompts = {r["prompt"] for r in kk + jl + policy + sharc + folio}
    for name, recs in evals.items():
        for r in recs:
            r.setdefault("label_order", r["labels"])
        assert not train_prompts & {r["prompt"] for r in recs}, f"train/eval prompt overlap in {name}"
        write(ROOT / f"data/eval/{name}.jsonl", recs)
    write(ROOT / "data/bench/train_kk.jsonl", kk)
    write(ROOT / "data/bench/train_jl.jsonl", jl)
    write(ROOT / "data/bench/train_policy.jsonl", policy)
    write(ROOT / "data/bench/train_sharc.jsonl", sharc)
    write(ROOT / "data/bench/train_folio.jsonl", folio)


if __name__ == "__main__":
    main()
