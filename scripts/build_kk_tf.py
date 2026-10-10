"""Single-domain Knights & Knaves data for Tree Forcing: data/student_kk_tf, in the layout of the other *_tf sets
(data/student_numgen_tf, student_pronto_tf, student_gsm8k_tf), so the K&K runs share the one-domain design of the
math / PrOntoQA / GSM8K runs instead of the 9-source mixture data/student_v3sl (K&K 995 of 7,207 train items, trees
on K&K only). notes/next_round/tree_forcing/kk_tf.md.

Puzzles: the repo's generator (cotdistill.sources.knights_knaves with its defaults, as scripts/build_items.py made
data/bench/train_kk: p_compound 0.7, 6 options = the solution + 5 near-miss distractors in shuffled letter order, the
training name pool), one people count per call, 4-10 inhabitants (the training range). Seeds: train 60000 + k, val
61000 + k for k inhabitants, disjoint from training (100 + k), kk_heldout (9000 + k), kk_deep (30000 + k), diag_kk
(50000 + k) and kk_audit_pairs (71000 + k). The generator is prefix-stable (the first m puzzles of a seed do not depend
on how many are asked for) and the files list the sizes round-robin (4, 5, ..., 10, 4, ...), so any prefix of
train.jsonl is size-balanced (counts differ by at most one) and --n M writes exactly the first M lines of a larger
build.
Augmentation as for data/student_v2 / v3f / v3sl (scripts/build_v2.py inject_none, its rates and its rewriting):
"None of the above" added as a wrong 7th option (8%) or replacing the gold option as the right answer (2%). v3sl draws
one coin per item; here the rates are exact in every block of AUG_BLOCK consecutive items of a file (4 wrong + 1 right
per 50, positions drawn per block; aug_u), so every prefix (any --n) has them too.
Finals: teacher = the program's one-hot gold (as the other *_tf sets; no DeepSeek labels), random_subqs [] (no
control questions), rationale "".
Trees: the deterministic solver of the v3sl trees (teacher_cache/solver_kk_v3f/build_reasoning_trees.py: Puzzle.read,
ProofBuilder.solve on the final prompt, augmented options included; every puzzle checked by independent_models (unique
solution, equal to the solver's) and validate_proof (every raw node follows from its listed parents over all role
assignments)), cleaned by scripts/build_solver_kk.clean (generic rule nodes and verify nodes dropped, repeated
questions merged, final-answer leaks dropped, about half of the nodes asked in their opposite wording, node_depth
recomputed; the flips drawn from a generator keyed by the item id) and given the local-view fields of
scripts/build_local_kk.py (local_preamble, local_lines, local_trivial) with its sufficiency audit (audit_nodes: every
derived node entailed by its cleaned parents over all role assignments; the build fails otherwise). tree_fix True,
tree_source "solver_kk_tf". Item keys, node keys per type (parse / case / derive) and value types are those of a v3sl
K&K item (checked against data/student_v3sl when present: summary.json "format").
Dedup: a puzzle is skipped when its prompt (before or after augmentation), its statement block or its set of
statement lines occurs in data/bench/train_kk.jsonl, data/student_v3sl/{train,val}.jsonl, any data/eval/*kk*.jsonl or
data/eval/diag*.jsonl (every "prompt" / "*_prompt" string of every record), in val, or earlier in train; puzzles with
fewer than 6 generated options (the solver's format) are skipped too. Each skip is counted.
Size: --n train puzzles (default DEFAULT_N = 1792), --val-n val puzzles (300). tokens.json (needs `tokenizers`;
--no-tokens skips it): train tokens per epoch with the Qwen3.5 tokenizer, final view (A2: FINAL_TEMPLATE over the
prompt) and GFP-S node views (every node, --tree-cap 0: SUBQ_TEMPLATE over treeplan.statement_block, as
student.build_full_tree builds them), for every N of --grid (the build then generates max(grid) train puzzles and
writes the first --n), next to the same counts for data/student_numgen_tf (--tree-sources numgen); padded tokens /
forward passes per epoch for token budgets 2048 and 4096 (student.token_budget_batches, IPU problems per update in
file order). DEFAULT_N = the largest train
prefix whose GFP-S tokens per epoch stay within BUDGET x numgen's (tokens.json "crossover": 1792 = 256 per size, 1.10x
numgen, so a 4-epoch GFP-S run costs about what the numgen one did; "rule" re-checks it and also gives the largest grid
N within the budget, 1000 = 0.61x).
summary.json: sizes, md5s, people-count / gold-position / augmentation counts (v3sl K&K alongside), node statistics,
cleaning and sufficiency counters (v3sl's names), dedup counts, format check.

  ~/.venvs/sev/bin/python scripts/build_kk_tf.py                          # data/student_kk_tf + tokens.json
  ~/.venvs/sev/bin/python scripts/build_kk_tf.py --n 4000 --out data/student_kk_tf_n4000 --no-tokens
CPU only, about 4 min and < 0.3 GB for the default grid (8,300 puzzles solved and tokenised).
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import os
import random
import re
import resource
import statistics
import sys
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cotdistill import sources as S  # noqa: E402
from cotdistill.treeplan import _node_p, statement_block  # noqa: E402
from build_gsm8k_tf import templates  # noqa: E402
from build_local_kk import audit_nodes, load_solver, preamble, statement_line  # noqa: E402
from build_solver_kk import clean  # noqa: E402
from build_v2 import inject_none  # noqa: E402

PEOPLE = tuple(range(4, 11))
SEED_TRAIN, SEED_VAL = 60000, 61000
SOURCE, TREE_SOURCE, ID_PREFIX = "knights_knaves", "solver_kk_tf", "kktf-"
ITEM_KEYS = ("item_id", "source", "prompt", "labels", "gold_label", "teacher", "depth", "subqs", "random_subqs",
             "rationale", "n_inconsistent_dropped", "tree_fix", "n_leak_dropped", "augment", "has_tree", "tree_source",
             "local_preamble")                               # a v3sl K&K item's keys, in its order
DEDUP = ("data/bench/train_kk.jsonl", "data/student_v3sl/train.jsonl", "data/student_v3sl/val.jsonl",
         "data/eval/*kk*.jsonl", "data/eval/diag*.jsonl")
V3SL = "data/student_v3sl"
SOLVER = "teacher_cache/solver_kk_v3f/build_reasoning_trees.py"
TOKENIZER = "results/weights/T1-Qwen3.5-2B-Base-student_v3-s2/model/tokenizer.json"
NUMGEN = "data/student_numgen_tf/train.jsonl"
GRID = "1000,2000,4000,8000"
DEFAULT_N, VAL_N = 1792, 300        # DEFAULT_N: tokens.json crossover (the largest prefix within BUDGET), 256 per size
BUDGET = 1.10                       # GFP-S tokens per epoch at most 1.1 x numgen's (DEFAULT_N rule)
AUG_BLOCK, AUG_WRONG, AUG_RIGHT = 50, 4, 1  # per block: inject_none's p_wrong 0.08 / p_right 0.02, exact
IPU = 32                            # train_student.py --items-per-update default (full-tree loop: problems per update)
TOKEN_BUDGETS = (2048, 4096)        # gjob: kk 2048, math / pronto / gsm 4096
_STMT = re.compile(r'^- [A-Z][a-z]+ says: "[^"\n]+"$', re.M)


# ------------------------------------------------------------------------------------------------------ helpers
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def puzzle_keys(prompt: str) -> list[str]:
    """Dedup keys of a K&K prompt: the whole prompt, the statement block (everything before the question) and the
    sorted set of statement lines (the same puzzle in another speaker order)."""
    out = ["prompt:" + sha1(prompt), "block:" + sha1(prompt.partition("\n\nQuestion: ")[0])]
    lines = sorted(_STMT.findall(prompt))
    if lines:
        out.append("lines:" + sha1("\n".join(lines)))
    return out


def dedup_index(patterns: tuple[str, ...]) -> tuple[dict, dict]:
    """key -> first file holding it, over every "prompt" / "*_prompt" string of every record of the matching files;
    and per file: records and prompt strings read."""
    owners, files = {}, {}
    paths = sorted({p for pat in patterns for p in glob.glob(str(ROOT / pat))})
    for p in paths:
        rel = str(Path(p).relative_to(ROOT))
        n_rec = n_prompt = 0
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                n_rec += 1
                for k, v in rec.items():
                    if isinstance(v, str) and (k == "prompt" or k.endswith("_prompt")):
                        n_prompt += 1
                        for key in puzzle_keys(v):
                            owners.setdefault(key, rel)
        files[rel] = {"records": n_rec, "prompts": n_prompt}
    return owners, files


class SizeStream:
    """Puzzles of one people count from one seed, in generator order (knights_knaves is prefix-stable, so asking for
    more puzzles later only appends)."""

    def __init__(self, seed: int, k: int, first: int):
        self.seed, self.k, self.want, self.recs, self.i = seed, k, max(8, first), [], 0

    def next(self) -> dict:
        if self.i >= len(self.recs):
            self.want = max(self.want, 2 * len(self.recs))
            recs = S.knights_knaves(self.want, seed=self.seed, people_range=(self.k, self.k))
            assert [r["prompt"] for r in recs[:len(self.recs)]] == [r["prompt"] for r in self.recs], "not prefix-stable"
            self.recs = recs
        r = self.recs[self.i]
        self.i += 1
        return r


# ------------------------------------------------------------------------------------------------------ one item
class _Coin:
    """A stand-in for inject_none's generator: its one draw is u."""

    def __init__(self, u: float):
        self.u = u

    def random(self) -> float:
        return self.u


def aug_u(seed: int, split: str, i: int) -> float:
    """inject_none's draw for position i of a file: in every block of AUG_BLOCK positions, AUG_WRONG positions get a
    wrong "None of the above" option (u 0.0 < p_wrong), AUG_RIGHT one a right one (u 0.09), the rest none (u 0.5)."""
    b, j = divmod(i, AUG_BLOCK)
    pos = random.Random(f"{seed}-aug-{split}-{b}").sample(range(AUG_BLOCK), AUG_WRONG + AUG_RIGHT)
    return 0.0 if j in pos[:AUG_WRONG] else 0.09 if j in pos[AUG_WRONG:] else 0.5


def make_item(r: dict, solver, seed: int, u: float, c: collections.Counter, bad: list) -> dict:
    """Student record of one generated puzzle: final (one-hot teacher, inject_none with draw u), solver tree cleaned
    as v3s, local fields and sufficiency audit as v3sl (counted into c, failures into bad)."""
    iid = ID_PREFIX + r["item_id"].split("-", 1)[1]
    letters = [chr(65 + i) for i in range(len(r["label_order"]))]
    rec = {"prompt": r["prompt"], "labels": letters, "gold_label": r["gold_label"],
           "teacher": {x: 1.0 if x == r["gold_label"] else 0.0 for x in letters}}
    aug = inject_none(rec, _Coin(u))
    puzzle = solver.Puzzle.read({"item_id": iid, "prompt": rec["prompt"]}, "kk_tf")
    builder = solver.ProofBuilder(puzzle)
    solution, answer = builder.solve()
    assert answer == rec["gold_label"], (iid, answer, rec["gold_label"])
    models, _ = solver.independent_models(puzzle)
    assert len(models) == 1 and models[0] == solution, iid
    assert solution == {p["question"][3:-10]: p["truth"] for p in r["predicates"]}, iid   # the planted roles
    check = solver.validate_proof(builder)
    c["raw_nodes"] += check["nodes"]
    crec = dict(r, item_id=iid, prompt=rec["prompt"],
                predicates=[{"question": f"Is {k} a knight?", "truth": v} for k, v in solution.items()])
    subqs, cc = clean(builder.nodes, crec, random.Random(f"{seed}-flip-{iid}"))
    for k, v in cc.items():
        c[f"clean_{k}"] += v
    it = {"item_id": iid, "source": SOURCE, "prompt": rec["prompt"], "labels": rec["labels"],
          "gold_label": rec["gold_label"], "teacher": rec["teacher"], "depth": len(solution), "subqs": subqs,
          "random_subqs": [], "rationale": "", "n_inconsistent_dropped": 0, "tree_fix": True,
          "n_leak_dropped": cc.get("leak", 0), "augment": aug, "has_tree": True, "tree_source": TREE_SOURCE,
          "local_preamble": preamble(rec["prompt"])}
    assert tuple(it) == ITEM_KEYS
    trivial = audit_nodes(solver, builder, builder.nodes, it, c, bad)
    it["subqs"] = [dict(sq, local_lines=[statement_line(it["prompt"], sq["question"])]) if sq["type"] == "parse"
                   else dict(sq, local_trivial=trivial[sq["id"]]) for sq in subqs]
    for sq in it["subqs"]:
        if sq["type"] == "parse":
            c["parse_nodes"] += 1
            c["parse_flipped"] += " a knave exactly when " in sq["question"]
    statement_block(it)                                    # the GFP-S view must exist
    return it


# ------------------------------------------------------------------------------------------------------ tokens
class Tokens:
    """Token lengths of an item's final view and GFP-S node views, as train_student.py builds them (build_full_tree,
    mode plain, view statements, cap 0: one view per node with a target; tokenised without special tokens)."""

    def __init__(self, path: Path):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(path))
        self.T = templates()

    def item(self, it: dict) -> tuple[int, list[int]]:
        ctx = statement_block(it)
        texts = [self.T["FINAL_TEMPLATE"].format(problem=it["prompt"])] + [
            self.T["SUBQ_TEMPLATE"].format(problem=ctx, question=sq["question"], a="Yes", b="No")
            for sq in it.get("subqs") or [] if "id" in sq and _node_p(sq, "cot") is not None]
        lens = [len(e.ids) for e in self.tok.encode_batch(texts, add_special_tokens=False)]
        return lens[0], lens[1:]


def budget_batches(lengths: list[int], budget: int, max_seqs: int = 16) -> list[list[int]]:
    """student.token_budget_batches (copied: importing student needs torch)."""
    out: list[list[int]] = []
    for i in sorted(range(len(lengths)), key=lambda j: -lengths[j]):
        if out and len(out[-1]) < max_seqs and (len(out[-1]) + 1) * lengths[out[-1][0]] <= budget:
            out[-1].append(i)
        else:
            out.append([i])
    return out


class Epoch:
    """Running per-epoch totals over items in file order: tokens of the finals (A2) and of the GFP-S node views,
    views, updates of IPU problems, and per token budget the padded tokens / forward passes of the full-tree loop."""

    def __init__(self, max_len: int):
        self.max_len, self.n, self.fin, self.nodes, self.views = max_len, 0, 0, 0, 0
        self.cur: list[int] = []
        self.pad = {b: 0 for b in TOKEN_BUDGETS}
        self.fwd = {b: 0 for b in TOKEN_BUDGETS}

    def add(self, fin: int, nodes: list[int]) -> None:
        if self.n and self.n % IPU == 0:
            self._close()
        fin, nodes = min(fin, self.max_len), [min(x, self.max_len) for x in nodes]
        self.n += 1
        self.fin += fin
        self.nodes += sum(nodes)
        self.views += len(nodes)
        self.cur += [fin] + nodes

    def _close(self) -> None:
        for b in TOKEN_BUDGETS:
            bs = budget_batches(self.cur, b)
            self.pad[b] += sum(len(x) * self.cur[x[0]] for x in bs)
            self.fwd[b] += len(bs)
        self.cur = []

    def row(self) -> dict:
        pad, fwd = dict(self.pad), dict(self.fwd)
        if self.cur:                                       # the open update, without closing it
            for b in TOKEN_BUDGETS:
                bs = budget_batches(self.cur, b)
                pad[b] += sum(len(x) * self.cur[x[0]] for x in bs)
                fwd[b] += len(bs)
        return {"items": self.n, "tokens_final": self.fin, "tokens_nodes": self.nodes,
                "tokens_gfps": self.fin + self.nodes, "views": self.views, "updates": math.ceil(self.n / IPU),
                **{f"pad_tok_tb{b}": pad[b] for b in TOKEN_BUDGETS}, **{f"fwd_tb{b}": fwd[b] for b in TOKEN_BUDGETS}}


def dist(xs: list[int], max_len: int) -> dict:
    xs = sorted(xs)
    if not xs:
        return {}
    return {"n": len(xs), "mean": round(sum(xs) / len(xs), 1), "p50": xs[len(xs) // 2],
            "p99": xs[min(len(xs) - 1, int(0.99 * len(xs)))], "max": xs[-1],
            "share_le_max_len": round(sum(x <= max_len for x in xs) / len(xs), 4)}


def numgen_reference(tk: Tokens, path: Path, max_len: int) -> dict:
    e, fins, views = Epoch(max_len), [], []
    with open(path) as f:
        for line in f:
            it = json.loads(line)
            fin, nodes = tk.item(it)
            e.add(fin, nodes)
            fins.append(fin)
            views += nodes
    return {"file": str(path.relative_to(ROOT)), "md5": md5(path), **e.row(), "final_view": dist(fins, max_len),
            "gfps_node_view": dist(views, max_len)}


# ------------------------------------------------------------------------------------------------------ stats
def item_stats(rows: list[dict]) -> dict:
    """Counts over items (light per-item rows, row_of)."""
    n = len(rows)
    if not n:
        return {"items": 0}
    m = lambda xs: round(statistics.mean(xs), 2)   # noqa: E731
    nodes = [x["nodes"] for x in rows]
    tc, nd = collections.Counter(), collections.Counter()
    for x in rows:
        tc.update(x["types"])
        nd.update(x["node_depth"])
    return {"items": n,
            "people": dict(sorted(collections.Counter(x["people"] for x in rows).items())),
            "gold_label": dict(sorted(collections.Counter(x["gold"] for x in rows).items())),
            "gold_label_unaugmented": dict(sorted(collections.Counter(x["gold"] for x in rows
                                                                      if x["augment"] is None).items())),
            "augment": dict(sorted(collections.Counter(str(x["augment"]) for x in rows).items())),
            "n_options": dict(sorted(collections.Counter(x["n_options"] for x in rows).items())),
            "nodes": sum(nodes), "nodes_per_item": [m(nodes), min(nodes), max(nodes)],
            "nodes_per_item_by_people": {k: m([x["nodes"] for x in rows if x["people"] == k])
                                         for k in sorted({x["people"] for x in rows})},
            "node_types": dict(sorted(tc.items())),
            "node_depth": {str(k): v for k, v in sorted(nd.items())},
            "max_depth": [m([x["max_depth"] for x in rows]), min(x["max_depth"] for x in rows),
                          max(x["max_depth"] for x in rows)],
            "share_with_parents": m([x["with_parents"] for x in rows]),
            "yes_share": round(sum(x["yes"] for x in rows) / max(1, sum(nodes)), 4)}


def row_of(it: dict) -> dict:
    sq = it["subqs"]
    return {"people": it["depth"], "gold": it["gold_label"], "augment": it["augment"], "n_options": len(it["labels"]),
            "nodes": len(sq), "types": collections.Counter(s["type"] for s in sq),
            "node_depth": collections.Counter(s["node_depth"] for s in sq),
            "max_depth": max(s["node_depth"] for s in sq),
            "with_parents": sum(bool(s["depends_on"]) for s in sq) / len(sq), "yes": sum(bool(s["truth"]) for s in sq)}


class Layout:
    """Every key list (with order) and value type seen in items and, per node type, in nodes."""

    def __init__(self):
        self.item_keys, self.item_types = set(), collections.defaultdict(set)
        self.node_keys, self.node_types = collections.defaultdict(set), collections.defaultdict(set)

    def add(self, it: dict) -> None:
        self.item_keys.add(tuple(it))
        for k, v in it.items():
            self.item_types[k].add(type(v).__name__)
        for sq in it["subqs"]:
            self.node_keys[sq["type"]].add(tuple(sq))
            for k, v in sq.items():
                self.node_types[(sq["type"], k)].add(type(v).__name__)

    def as_dict(self) -> dict:
        return {"item_keys": [list(x) for x in sorted(self.item_keys)],
                "item_types": {k: sorted(v) for k, v in sorted(self.item_types.items())},
                "node_keys": {t: [list(x) for x in sorted(v)] for t, v in sorted(self.node_keys.items())},
                "node_types": {f"{t}.{k}": sorted(v) for (t, k), v in sorted(self.node_types.items())}}


def v3sl_reference(path: Path) -> tuple[dict, Layout]:
    """The same counts for the K&K items of data/student_v3sl, and their layout (format check)."""
    out, lay = {}, Layout()
    for split in ("train", "val"):
        rows = []
        with open(path / f"{split}.jsonl") as f:
            for line in f:
                it = json.loads(line)
                if it.get("source") == SOURCE:
                    rows.append(row_of(it))
                    lay.add(it)
        out[split] = item_stats(rows)
    return out, lay


def format_check(ours: Layout, ref: Layout) -> dict:
    """Ours vs v3sl K&K: the same single item key list, the same node types, per type the same single key list, and
    every value type of ours among v3sl's for that field ("augment" is None or a string in both)."""
    diffs = []
    if len(ours.item_keys) != 1 or ours.item_keys != ref.item_keys:
        diffs.append(f"item keys {sorted(ours.item_keys)} vs {sorted(ref.item_keys)}")
    for k, ts in ours.item_types.items():
        if not ts <= ref.item_types.get(k, set()):
            diffs.append(f"item {k}: {sorted(ts)} vs {sorted(ref.item_types.get(k, set()))}")
    if set(ours.node_keys) != set(ref.node_keys):
        diffs.append(f"node types {sorted(ours.node_keys)} vs {sorted(ref.node_keys)}")
    for t, ks in ours.node_keys.items():
        if len(ks) != 1 or ks != ref.node_keys.get(t):
            diffs.append(f"{t} node keys {sorted(ks)} vs {sorted(ref.node_keys.get(t, set()))}")
    for (t, k), ts in ours.node_types.items():
        if not ts <= ref.node_types.get((t, k), set()):
            diffs.append(f"{t} node {k}: {sorted(ts)} vs {sorted(ref.node_types.get((t, k), set()))}")
    return {"equal": not diffs, "diffs": diffs}


# ------------------------------------------------------------------------------------------------------ build
def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/student_kk_tf")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="train puzzles written")
    ap.add_argument("--val-n", type=int, default=VAL_N)
    ap.add_argument("--seed", type=int, default=0, help="keys the augmentation and wording-flip generators")
    ap.add_argument("--grid", default=GRID, help="train sizes for tokens.json (the build generates max(grid, n))")
    ap.add_argument("--no-tokens", action="store_true", help="no tokens.json (no tokenizer needed; generates --n only)")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--numgen", default=NUMGEN, help="reference set for tokens.json")
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--dedup", default=",".join(DEDUP), help="comma list of globs (relative to the repo)")
    ap.add_argument("--solver", default=SOLVER)
    ap.add_argument("--v3sl", default=V3SL, help="reference K&K items (format check, balance table); skipped if absent")
    args = ap.parse_args(argv)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    grid = [] if args.no_tokens else sorted({int(x) for x in args.grid.split(",") if x})
    pool = max([args.n] + grid)
    solver = load_solver(ROOT / args.solver)
    owners, dedup_files = dedup_index(tuple(x for x in args.dedup.split(",") if x))
    print(f"dedup index: {len(owners)} keys from {len(dedup_files)} files, rss {rss_mb()} MB", flush=True)
    tk = ng = None
    tok_rep = {"tokenizer": args.tokenizer, "max_len": args.max_len, "items_per_update": IPU,
               "views": "final: FINAL_TEMPLATE(prompt); GFP-S node: SUBQ_TEMPLATE(statement_block, node question), "
                        "every node (--tree-cap 0); no special tokens; lengths capped at max_len",
               "files": {}, "grid": {}}
    if not args.no_tokens:
        tk = Tokens(ROOT / args.tokenizer)
        ng = numgen_reference(tk, ROOT / args.numgen, args.max_len)
        print(f"numgen reference: {ng['tokens_gfps']} GFP-S tokens per epoch, rss {rss_mb()} MB", flush=True)
    cap = BUDGET * ng["tokens_gfps"] if ng else None
    seen: set = set()
    skips = collections.Counter()
    c, bad = collections.Counter(), []
    lay = Layout()
    summary = {"script": "scripts/build_kk_tf.py", "args": vars(args),
               "seeds": {"train": f"{SEED_TRAIN} + k", "val": f"{SEED_VAL} + k", "k": list(PEOPLE)},
               "md5": {}, "files": {}}

    def take(stream: SizeStream, seed: int, u: float) -> dict:
        while True:
            r = stream.next()
            if len(r["label_order"]) != 6:
                skips["fewer_than_6_options"] += 1
                continue
            keys = puzzle_keys(r["prompt"])
            hit = next((owners[k] for k in keys if k in owners), None)
            if hit:
                skips[f"in {hit}"] += 1
                continue
            if any(k in seen for k in keys):
                skips["repeat within the new set"] += 1
                continue
            it = make_item(r, solver, seed, u, c, bad)
            keys2 = puzzle_keys(it["prompt"])
            hit = next((owners[k] for k in keys2 if k in owners), None)
            if hit:                                       # the augmented prompt (never seen so far; checked anyway)
                skips[f"augmented prompt in {hit}"] += 1
                continue
            seen.update(keys + keys2)
            return it

    crossing = None
    for split, count, seed0 in (("val", args.val_n, SEED_VAL), ("train", pool, SEED_TRAIN)):
        n_write = args.val_n if split == "val" else args.n
        streams = {k: SizeStream(seed0 + k, k, math.ceil(count / len(PEOPLE)) + 8) for k in PEOPLE}
        rows, fins, views, cuts = [], [], [], {}
        ep = Epoch(args.max_len)
        path = out / f"{split}.jsonl"
        with open(path, "w") as f:
            for i in range(count):
                it = take(streams[PEOPLE[i % len(PEOPLE)]], args.seed, aug_u(args.seed, split, i))
                if i < n_write:
                    f.write(json.dumps(it) + "\n")
                    rows.append(row_of(it))
                    lay.add(it)
                if tk is not None:
                    fin, nodes = tk.item(it)
                    ep.add(fin, nodes)
                    if i < n_write:
                        fins.append(fin)
                        views += nodes
                    if split == "train" and crossing is None and ep.fin + ep.nodes > cap:
                        crossing = ep.n - 1                # the largest prefix within the budget
                    if (split == "train" and ep.n in grid) or ep.n == n_write:
                        cuts[ep.n] = ep.row()
                if (i + 1) % 1000 == 0:
                    print(f"{split}: {i + 1} / {count}, rss {rss_mb()} MB", flush=True)
        key = f"{out.name}/{split}.jsonl"
        summary["md5"][key] = md5(path)
        summary["files"][key] = item_stats(rows)
        if tk is not None:
            tok_rep["files"][key] = {"final_view": dist(fins, args.max_len),
                                     "gfps_node_view": dist(views, args.max_len), **cuts[n_write]}
            if split == "train":
                tok_rep["grid"] = {str(n): cuts[n] for n in grid}
        print(f"{split}: wrote {n_write} of {count}, rss {rss_mb()} MB", flush=True)
    summary["skipped"] = dict(sorted(skips.items()))
    summary["sufficiency"] = dict(sorted((k, v) for k, v in c.items()
                                         if not k.startswith("clean_") and k != "raw_nodes"))
    summary["sufficiency"]["not_entailed"] = c["not_entailed"]
    summary["sufficiency_scope"] = f"every generated puzzle (val {args.val_n} + train {pool}), written or not"
    summary["not_entailed_examples"] = bad[:20]
    summary["cleaning"] = {k[len("clean_"):]: v for k, v in sorted(c.items()) if k.startswith("clean_")}
    summary["cleaning"]["raw_nodes"] = c["raw_nodes"]
    summary["dedup"] = {"files": dedup_files, "keys": len(owners),
                        "key_kinds": "prompt (before and after augmentation), statement block, sorted statement lines"}
    v3 = ROOT / args.v3sl
    if (v3 / "train.jsonl").exists():
        summary["v3sl_kk"], ref = v3sl_reference(v3)
        summary["format"] = format_check(lay, ref)
    else:
        summary["format"] = {"equal": None, "diffs": [f"{args.v3sl} absent: not checked"]}
    summary["layout"] = lay.as_dict()
    if tk is not None:
        tok_rep["numgen_tf"] = ng
        for row in list(tok_rep["grid"].values()) + [tok_rep["files"][f"{out.name}/train.jsonl"]]:
            row["gfps_over_numgen"] = round(row["tokens_gfps"] / ng["tokens_gfps"], 3)
            row["final_over_numgen"] = round(row["tokens_final"] / ng["tokens_final"], 3)
            for b in TOKEN_BUDGETS:
                row[f"pad_tok_tb{b}_over_numgen"] = round(row[f"pad_tok_tb{b}"] / ng[f"pad_tok_tb{b}"], 3)
        ok = [int(n) for n, row in tok_rep["grid"].items() if row["tokens_gfps"] <= cap]
        tok_rep["rule"] = {"budget": BUDGET, "numgen_tokens_gfps": ng["tokens_gfps"], "cap_tokens_gfps": round(cap),
                           "largest_grid_n_within_budget": max(ok) if ok else None, "default_n": DEFAULT_N,
                           "default_n_follows_rule": crossing == DEFAULT_N}
        tok_rep["crossover"] = {"n": crossing, "n_multiple_of_7": None if crossing is None else crossing // 7 * 7,
                                "note": "largest train prefix with GFP-S tokens per epoch <= cap_tokens_gfps "
                                        "(None: the whole pool fits)"}
        (out / "tokens.json").write_text(json.dumps(tok_rep, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("md5", "skipped", "sufficiency", "format")}, indent=1))
    if tk is not None:
        print(json.dumps({k: tok_rep[k] for k in ("grid", "rule", "crossover")}, indent=1))
    print(f"rss {rss_mb()} MB")
    if bad:
        raise SystemExit(f"{len(bad)} derived nodes are not entailed by their cleaned parents")
    if summary["format"]["equal"] is False:
        raise SystemExit(f"format differs from v3sl K&K: {summary['format']['diffs'][:5]}")
    return summary


if __name__ == "__main__":
    main()
