"""Keep only the tree nodes of the first K solution steps (default 3) of a Tree Forcing data set: the step-capped GSM8K
variant asked by the 12B depth analysis (deep GSM8K problems carry 7-9 nodes, the late ones about quantities a single
pass may not compute). GSM8K nodes have a `step` index (1-based, the Socratic step); depends_on is empty for 73% of them,
so the dependency depth cannot express this cap. Train rows lose the nodes with step > K (final rows unchanged); the
val file is copied unchanged so node and final evaluations stay comparable with the uncapped runs.

  python3 scripts/filter_tree_steps.py --src data/student_gsm8k_tf --out data/student_gsm8k_tf_s3 --max-step 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default="data/student_gsm8k_tf")
    ap.add_argument("--out", default="data/student_gsm8k_tf_s3")
    ap.add_argument("--max-step", type=int, default=3)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_rows = n_all = n_keep = n_rows_cut = 0
    with open(src / "train.jsonl") as fi, open(out / "train.jsonl", "w") as fo:
        for line in fi:
            x = json.loads(line)
            sq = x.get("subqs") or []
            keep = [q for q in sq if int(q.get("step", 0)) <= args.max_step]
            n_rows += 1; n_all += len(sq); n_keep += len(keep); n_rows_cut += len(keep) < len(sq)
            if len(keep) < len(sq):
                x["subqs"] = keep
                x["has_tree"] = bool(keep)
            fo.write(json.dumps(x, ensure_ascii=False) + "\n")
    shutil.copy2(src / "val.jsonl", out / "val.jsonl")
    md5 = {f: hashlib.md5((out / f).read_bytes()).hexdigest() for f in ("train.jsonl", "val.jsonl")}
    summary = {"source": str(src), "max_step": args.max_step, "rows": n_rows, "rows_with_nodes_cut": n_rows_cut,
               "nodes_in": n_all, "nodes_kept": n_keep, "nodes_dropped": n_all - n_keep,
               "val": "copied unchanged", "md5": md5}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
