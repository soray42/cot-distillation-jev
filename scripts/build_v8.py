"""v8 candidate release mixture: one JSON config (configs/v8/*.json) -> data/student_v8/{train,val}.jsonl + summary.json.

v8 = the v7 assembler (scripts/build_v7.py: selection, label/licence provenance, 13-gram + digit-masked decontamination
against every data/eval/*.jsonl, cross-source exact/near dedup, val sampling, summary) over sources prepared for GFP-S
(Tree Forcing: every node of a problem's tree asked as a yes/no question under the problem's statement block,
treeplan.statement_block; train_student.py --tree-full plain --tree-view statements --tree-cap 0 --tree-sources S).

A source keeps its tree only if it is GFP-S-eligible (config gfps.sources): program-verified trees whose prompt has a
clean statement-block cut. Every other source is answer-only (v7 "nodes": "drop": subqs and random_subqs emptied in
train). The config's v7 sources follow configs/v7 (see build_v7's docstring), plus an optional "stage" block that
derives the source's input files before assembly (written under --tmp, never into data/):
  stage: {"from": [train file, val file], "v7_source": name (keep rows whose v7_source is name, e.g. the math slice of a
          built v7 set), "drop_node_rows": true (standalone node rows left out), "trees_from": [files] (numgen: subqs,
          local_preamble, tree_fix, tree_source taken from the Tree Forcing build, data/student_numgen_tf, for every
          item with the same item_id and byte-identical prompt; items not there keep their final question only)}
The staged train / val files replace the source's "train" and "val.file".

After assembly every train and val row is checked: a row of an eligible source has a statement block (the training
script calls statement_block on every item of --tree-sources before the first update) and no node of its tree is a
view_leak in the statements view (nothing stated); no train row of another source carries subqs. summary.json (the v7
summary) gets a "v8" section: rows by source, GFP-S-eligible items and nodes per source (mean / p90 / max per item,
node_depth histogram, yes share), the --tree-sources string to train with, staging counts, decontamination removals
by evaluation file, md5s.

The data root (--root, default this checkout) holds data/ and notes/ (both gitignored, so a git worktree passes the
main checkout's path); build_v7 and build_breadth read their inputs relative to it.

CPU only, streaming, about 1 GB RSS:

  python3 scripts/build_v8.py configs/v8/v8_release_4b.json [--root /path/to/checkout] [--limit 50]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_v7 as b7  # noqa: E402
from cotdistill.treeplan import _node_p, node_depths, statement_block, view_leak  # noqa: E402


def set_root(root: Path) -> None:
    """Point build_v7 / build_breadth (module-level ROOT, NOTES) at the data root."""
    import build_breadth as bb
    b7.ROOT, b7.NOTES = root, root / "notes/next_round/data_expansion"
    bb.ROOT, bb.NOTES, bb.POOL = root, root / "notes/next_round/data_expansion", root / "data/raw/v4/tasksource_jev"


def check_v8(cfg: dict) -> list[str]:
    """Structural errors of the v8 additions (empty = fine)."""
    err = []
    g = cfg.get("gfps") or {}
    if not g.get("sources"):
        err.append("gfps.sources: the GFP-S-eligible item sources")
    for s in cfg.get("sources", []):
        st = s.get("stage")
        if st and not (isinstance(st.get("from"), list) and 1 <= len(st["from"]) <= 2):
            err.append(f"{s.get('name')}: stage.from must be [train] or [train, val]")
        if st and len(st.get("from", [])) == 2 and not s.get("val"):
            err.append(f"{s.get('name')}: stage.from has a val file but the source has no val spec")
    return err


# ---------------------------------------------------------------------------------------------------------- staging

def load_trees(root: Path, files: list[str]) -> dict[str, dict]:
    """item_id -> tree fields of a Tree Forcing build (numgen_tf)."""
    out = {}
    for f in files:
        for line in b7.lines(root / f):
            r = json.loads(line)
            out[r["item_id"]] = {k: r[k] for k in ("prompt", "subqs", "local_preamble", "tree_fix", "tree_source",
                                                   "has_tree") if k in r}
    return out


def stage_file(src: Path, dst: Path, st: dict, trees: dict | None, c: collections.Counter) -> None:
    with open(dst, "w") as g:
        for line in b7.lines(src):
            r = json.loads(line)
            if st.get("v7_source") and r.get("v7_source") != st["v7_source"]:
                continue
            if st.get("drop_node_rows") and r.get("row_kind") == "node":
                c["node_rows_dropped"] += 1
                continue
            if trees is not None and r.get("row_kind") != "node":
                t = trees.get(r["item_id"])
                if t is not None and t["prompt"] == r["prompt"]:
                    r.update({k: v for k, v in t.items() if k != "prompt"})
                    c["trees_replaced"] += 1
                    c["tree_nodes"] += len(t["subqs"])
                else:
                    c["no_tree_final_only"] += 1
                    c["probe_subqs_dropped"] += len(r.get("subqs") or [])
                    r["subqs"], r["random_subqs"] = [], []
                    r["has_tree"] = False
                    r.pop("tree_source", None)
            c["rows"] += 1
            g.write(json.dumps(r) + "\n")


def stage(cfg: dict, root: Path, tmp: Path) -> dict:
    """Write the staged inputs; point each staged source's train (and val file) at them. Returns counts per source."""
    tmp.mkdir(parents=True, exist_ok=True)
    out = {}
    for s in cfg["sources"]:
        st = s.get("stage")
        if not st:
            continue
        trees = load_trees(root, st["trees_from"]) if st.get("trees_from") else None
        info = {"from": st["from"], "inputs_md5": {f: b7.md5_file(root / f) for f in st["from"] + st.get("trees_from", [])}}
        for split, f in zip(("train", "val"), st["from"]):
            c = collections.Counter()
            dst = tmp / f"stage_{s['name']}.{split}.jsonl"
            stage_file(root / f, dst, st, trees, c)
            info[split] = dict(c)
            if split == "train":
                s["train"] = str(dst)
            else:
                s["val"] = dict(s["val"], file=str(dst))
        out[s["name"]] = info
        print(f"[v8] staged {s['name']}: {json.dumps({k: v for k, v in info.items() if k != 'inputs_md5'})}", flush=True)
    return out


# ---------------------------------------------------------------------------------------------------------- checks

def gfps_stats(path: Path, eligible: set, split: str, fails: list) -> dict:
    """Per source: rows, items with a tree, nodes, nodes per item, node depths, yes share, statement-block and
    view_leak failures (appended to fails)."""
    per = collections.defaultdict(lambda: {"rows": 0, "items": 0, "items_with_tree": 0, "nodes": 0,
                                           "depth": collections.Counter(), "yes": 0, "per_item": []})
    for line in b7.lines(path):
        r = json.loads(line)
        src = r.get("source")
        d = per[src]
        d["rows"] += 1
        if r.get("row_kind") == "node":
            continue
        d["items"] += 1
        nodes = [sq for sq in r.get("subqs") or [] if "id" in sq and _node_p(sq, "cot") is not None]
        if src not in eligible:
            if nodes and split == "train":
                fails.append((r["item_id"], f"{src} is not GFP-S-eligible but keeps {len(nodes)} nodes"))
            continue
        try:
            statement_block(r)
        except ValueError as e:
            fails.append((r["item_id"], str(e)))
            continue
        leaks = [sq["id"] for sq in nodes if view_leak(r, sq, [])]
        if leaks:
            fails.append((r["item_id"], f"view_leak in the statements view: {leaks}"))
        if not nodes:
            continue
        d["items_with_tree"] += 1
        d["nodes"] += len(nodes)
        d["per_item"].append(len(nodes))
        all_nodes = [sq for sq in r["subqs"] if "id" in sq]
        dep = ({sq["id"]: sq["node_depth"] for sq in all_nodes} if all("node_depth" in sq for sq in all_nodes)
               else node_depths(all_nodes))
        for sq in nodes:
            d["depth"][dep.get(sq["id"], 0)] += 1
            d["yes"] += _node_p(sq, "cot") >= 0.5
    out = {}
    for src, d in sorted(per.items(), key=lambda kv: str(kv[0])):
        L = sorted(d.pop("per_item")) or [0]
        e = {k: v for k, v in d.items() if k not in ("depth", "yes")}
        if src in eligible:
            e.update({"gfps": True, "nodes_per_tree_item": {"mean": round(sum(L) / max(1, d["items_with_tree"]), 2),
                                                            "p90": L[int(.9 * (len(L) - 1))], "max": L[-1]},
                      "node_depth": dict(sorted(d["depth"].items())),
                      "yes_share": round(d["yes"] / max(1, d["nodes"]), 4)})
        out[str(src)] = e
    return out


def removed_by_eval(summ: dict) -> dict:
    """Train/val rows removed by the decontamination gates, by evaluation file (problem-level drops included)."""
    out = collections.Counter()
    for why, n in summ.get("decontam", {}).get("by_reason", {}).items():
        out[why] += n
    return dict(out.most_common())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--root", default=str(ROOT), help="checkout whose data/ and notes/ hold the inputs and outputs")
    ap.add_argument("--out", default="", help="output dir relative to --root (default: the config's out)")
    ap.add_argument("--tmp", default=b7.DEFAULT_TMP)
    ap.add_argument("--evals", default="data/eval/*.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: at most this many problems per source")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--allow-short-fill", action="store_true")
    ap.add_argument("--allow-gen", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    todos = b7.todo_paths(cfg)
    err = ([f"TODO field {p}" for p in todos] + b7.check_config(cfg) + check_v8(cfg))
    if err:
        print(f"{args.config}: invalid config:\n  " + "\n  ".join(err), file=sys.stderr)
        return 2
    if args.check_only:
        print(f"{args.config}: ok")
        return 0
    set_root(root)
    tmp = Path(args.tmp) / cfg["name"]
    staged = stage(cfg, root, tmp / "stage")
    eligible = set(cfg["gfps"]["sources"])
    summ = b7.Assembler(cfg, args).run()
    out = root / (args.out or cfg["out"])
    fails: list = []
    by_split = {split: gfps_stats(out / f"{split}.jsonl", eligible, split, fails) for split in ("train", "val")}
    tr = by_split["train"]
    v8 = {
        "builder": "scripts/build_v8.py", "builder_md5": b7.md5_file(Path(__file__)),
        "config": str(cfg_path), "config_md5": b7.md5_file(cfg_path), "root": str(root),
        "gfps": {"view": cfg["gfps"].get("view", "statements"), "sources": sorted(eligible),
                 "tree_sources_flag": ",".join(sorted(eligible)),
                 "train_flags": f"--tree-full plain --tree-view {cfg['gfps'].get('view', 'statements')} --tree-cap 0 "
                                f"--tree-sources {','.join(sorted(eligible))}",
                 "answer_only": cfg["gfps"].get("answer_only", {}),
                 "eligible_items": sum(v["items"] for k, v in tr.items() if k in eligible),
                 "eligible_items_with_tree": sum(v["items_with_tree"] for k, v in tr.items() if k in eligible),
                 "total_nodes": sum(v["nodes"] for k, v in tr.items() if k in eligible),
                 "val_nodes": sum(v["nodes"] for k, v in by_split["val"].items() if k in eligible)},
        "rows_by_source": {"train": {k: v["rows"] for k, v in tr.items()},
                           "val": {k: v["rows"] for k, v in by_split["val"].items()}},
        "per_source": by_split,
        "staging": staged,
        "decontam_removed_by_reason": removed_by_eval(summ),
        "checks": {"failures": len(fails), "examples": fails[:20]},
        "md5": {"train": b7.md5_file(out / "train.jsonl"), "val": b7.md5_file(out / "val.jsonl")},
    }
    summ["v8"] = v8
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    (out / "config.json").write_text(cfg_path.read_text())    # the config as written (the assembler saw staged paths:
    #                                                           summary inputs_md5 / v8.staging)
    print(json.dumps({k: v8[k] for k in ("gfps", "rows_by_source", "checks", "md5")}, indent=1))
    if fails:
        print(f"[v8] {len(fails)} check failures (see summary.json v8.checks)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
