"""v7 mixture assembler: one JSON config (configs/v7/*.json) -> data/student_v7_<name>/{train,val}.jsonl + summary.json.

Plan: notes/next_round/data_expansion/plan.md §1 (mixture, label and node keys, provenance columns), §2 row 8, §3.4
(decontamination), §4 (lab arms MS / MA / MS-gen / BR-40k), §6.1 (licence traps), §6.4 (program-family shares).

Config (top level):
  name, tier ("lab" | "release"), out, seed, description, plan
  sources        list, in priority order (an earlier source wins every cross-source duplicate), each:
    name, slice (plan ids: G, A, R, M, L, B, ...), train (jsonl), node_rows (jsonl of standalone node rows, optional),
    node_link (field of a node row naming its problem: "node_of" / "problem_id"), problem_key (field of a final naming
    its problem; default item_id), include_source / exclude_source (regex on the row `source`),
    select: {problems | rows: int | "all", weight: share of the config's total_problems / total_rows,
             rank: "hash" (md5 of seed, source name, problem id) | "pool_rank" (row field) | "file",
             rank_key: the name in the md5 rank (default the source name; two sources reading one generator file
             with the same rank_key see one order, so offset gives fresh problems), offset: problems skipped in rank
             order, fill: true (problems are taken in rank order until the config's `fill` target is met; candidates:
             how many are screened), skip_selected: true (problems whose ids an earlier source already holds are not
             candidates)}
    dedup_group: sources sharing it count as one source for the cross-source duplicate check (one generator's
    template-similar problems are not duplicates of each other), family_key, max_node_rows_per_problem,
    where: {field: [allowed values]} on finals (e.g. the goldproof size tiers lab / release / 4b)
    nodes: "keep" (subqs kept, node rows included; max_node_rows_per_problem samples them by md5 rank and re-splits
           the weight 1 / kept rows) | "drop" (subqs and random_subqs emptied, node rows left out; both are still
           screened by the gates, so a "drop" arm keeps exactly the problems of its "keep" twin)
    label: "as_is" (the row's teacher field: teacher-soft or gold as built) | "gold" (one-hot on gold_label unless the
           row's label_basis is vote_share) | "program" (must already be one-hot on gold_label, else the row is dropped)
    label_basis / node_basis (provenance written to rows that lack them; label_basis otherwise "gold" when the
    teacher field is one-hot on gold_label, else "teacher-soft"), licence: {"class": C, "text": T} (every row)
    | {"from": "row"} (the row's licence_class) | {"from": "v5"} (data/license/per_source.csv, the replay_mc
    effective classes of data/v5/replay_mc.summary.json for replay/*, and for tasksource/* the more restrictive of that
    and the effective class of notes/next_round/data_expansion/tasksource_audit.csv), lab_only: true (licence not
    cleared, e.g. ProofWriter)
    gates: ["13gram"] and/or "masked" (plan §3.4; math sources), own_eval: eval-file name prefixes of the source's own
    in-family test files (hits there are template self-overlap: counted, not dropped), math, gsm_derived (bools)
    val: {file, node_rows, max_per_family, problems} or null
  val            {"mode": "sources"} (each source's val spec; the file's problems are sampled per family) |
                 {"mode": "copy", "path": P} (byte copy, e.g. the v5 val shared by v5-40k and BR-40k) |
                 {"mode": "split", "frac", "min_per_family", "max_per_family", "key"} (per-family stratified hold-out of
                 the kept train problems)
  fill           {"source": name, "target": "rows" | "math_chars", "rows": n, "reference": dir of a built v7 set
                 whose summary gives math_chars (token match), "min_ratio": r (optional: refuse to write the set when
                 achieved / target < r, e.g. a token-matched arm whose generator pool ran out; --allow-short-fill
                 builds it anyway and marks summary fill.below_min_ratio)}
  dedup          {"near": .8, "shingle": 5, "perms": 64, "bands": 16}
  caps           {"slice_share_chars": {"M": .25}, "gsm_share_of_math_problems": .15}: checked; a release build
                 refuses when one is exceeded, a lab build reports it
  total_problems / total_rows, required when a source has select.weight

Any string "TODO" anywhere in a config makes the script refuse (exit 2, every path listed): the release skeletons leave
their proportions as TODO until the lab rounds decide them.

Pipeline:
  1. Selection, per source in config order, streaming: problem ids and rank keys are read first, then only the rows of
     the selected problems (finals, node rows of the train file, node rows of node_rows) are transformed (node policy,
     label policy, provenance: v7_source, slice, licence_class, lab_only, label_basis, node_basis, decontam) and written
     to a per-source temp file; only a small record per row stays in memory.
  2. Decontamination of every row (train and val): the 13-gram rule of scripts/decontam.py (its grams(), N = 13,
     evaluation 13-grams in >= 20 evaluation items ignored) against every data/eval/*.jsonl, held as a sorted array of
     (gram, file) pairs instead of two dicts, so it fits the memory limit; all text fields count (prompt, sub-question
     and control-question text, rationale). For sources with "masked": build_breadth.MaskedGate (digit-masked 10-gram
     vs data/eval and the local math test sets, masked word-set Jaccard >= .5 vs the math sets) on the prompts of finals
     and node rows. A problem with any hit outside its own_eval files is dropped with all of its rows.
  3. Licence: a release build keeps PERM/ATTR rows with no per_source.csv flag only; a lab build keeps the rest
     flagged lab_only and counts them. Rows of a tasksource source whose audit row names a proprietary generator
     (tasksource_audit.csv `gen`, plan §6.1 item 2) are dropped from a release build (--allow-gen keeps them) and
     counted in a lab build (licence_gen_rows; row bytes unchanged).
  4. Cross-source duplicates: exact (normalised prompt md5, or the same item_id) and near (MinHash over word
     5-grams, crc32 shingles, 64 permutations in 16 bands; estimated Jaccard >= .8); the problem of the lower-priority
     source is dropped. Node rows a "drop" arm leaves out take part as in the "keep" twin, so both keep the same
     problems. Val problems duplicating any train row (exact: any source; near: other sources) are dropped.
  5. Fill (token match / top-up), val split, caps check.
  6. Output in md5(seed, item_id) order (the files are read back by byte offset), summary.json (rows per source, row
     kind and type, node coverage, licence mix, label/node basis, decontamination by reason and eval file, duplicates,
     slice shares, compute chars, fill/token match, md5 of every input, evaluation file, config and output, an md5 of
     each source's kept item ids so arms can be checked to share e.g. the G20 replay), per_source.json, config.json.

CPU only, streaming, <= ~1.5 GB RSS on the pilot configs:

  python3 scripts/build_v7.py configs/v7/v7_pilot_math_MS.json
  python3 scripts/build_v7.py configs/v7/v7_release_2b.json          # refuses while TODO fields remain
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import glob
import hashlib
import json
import math
import os
import re
import shutil
import sys
import zlib
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from decontam import N as N13, grams as grams13  # noqa: E402

TODO = "TODO"
RANK = {"PERM": 0, "ATTR": 1, "NC": 2, "SA": 3, "UNK": 4}
RELEASE_OK = {"PERM", "ATTR"}
POLICY = "policy:nodes"          # node rows a "drop" arm screens but leaves out
TOK = re.compile(r"[a-z0-9]+")
NOTES = ROOT / "notes/next_round/data_expansion"
DEFAULT_TMP = "/home/soray/.claude/jobs/3b5d797f/tmp/rel"


# ---------------------------------------------------------------------------------------------------------- config

def todo_paths(x, path: str = "$") -> list[str]:
    """Every place in a config whose value (or key) is or contains the string TODO."""
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            if TODO in str(k):
                out.append(f"{path}.{k} (key)")
            out += todo_paths(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out += todo_paths(v, f"{path}[{i}]")
    elif isinstance(x, str) and TODO in x:
        out.append(path)
    return out


def check_config(cfg: dict) -> list[str]:
    """Structural errors (empty list = fine)."""
    err = []
    for k in ("name", "tier", "out", "sources"):
        if k not in cfg:
            err.append(f"missing {k}")
    if cfg.get("tier") not in ("lab", "release"):
        err.append("tier must be lab or release")
    names = [s.get("name") for s in cfg.get("sources", [])]
    if len(set(names)) != len(names):
        err.append("source names must be unique")
    for s in cfg.get("sources", []):
        sel = s.get("select", {})
        if s.get("nodes", "keep") not in ("keep", "drop"):
            err.append(f"{s.get('name')}: nodes must be keep or drop")
        if s.get("label", "as_is") not in ("as_is", "gold", "program"):
            err.append(f"{s.get('name')}: label must be as_is, gold or program")
        if "weight" in sel and not (cfg.get("total_problems") or cfg.get("total_rows")):
            err.append(f"{s.get('name')}: select.weight needs total_problems or total_rows")
        if sel.get("fill") and (cfg.get("fill") or {}).get("source") != s.get("name"):
            err.append(f"{s.get('name')}: select.fill but the config's fill.source names another source")
        if not s.get("train"):
            err.append(f"{s.get('name')}: no train file")
    f = cfg.get("fill")
    if f and f.get("source") not in names:
        err.append("fill.source is not a source")
    if f and f.get("target") not in ("rows", "math_chars"):
        err.append("fill.target must be rows or math_chars")
    v = cfg.get("val", {"mode": "sources"})
    if v.get("mode") not in ("sources", "copy", "split"):
        err.append("val.mode must be sources, copy or split")
    return err


# ---------------------------------------------------------------------------------------------------------- helpers

def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lines(path):
    with open(path) as f:
        yield from f


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def hrank(*parts) -> str:
    """Deterministic rank key (never Python hash(), which is randomised per process)."""
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()


def norm_text(t: str) -> str:
    return " ".join(TOK.findall(t.lower()))


def row_texts(r: dict) -> list[str]:
    """All text fields of a row that the 13-gram gate screens (plan §2 row 2: prompt, options, node and teacher text)."""
    out = [r.get("prompt", "")]
    for key in ("subqs", "random_subqs"):
        for sq in r.get(key) or []:
            for k in ("question", "opposite"):
                if isinstance(sq.get(k), str):
                    out.append(sq[k])
    if isinstance(r.get("rationale"), str) and r["rationale"]:
        out.append(r["rationale"])
    return out + list(r.get("_screen_texts") or [])


def family_of(r: dict, key: str | None = None) -> str:
    if key and r.get(key) is not None:
        return str(r[key])
    for k in ("family", "group", "frame", "source"):
        if r.get(k) is not None:
            return str(r[k])
    return "?"


def most_restrictive(*cls: str) -> str:
    return max((c for c in cls if c), key=lambda c: RANK.get(c, 4))


class Licences:
    """v5 row classes: per_source.csv, replay_mc effective classes, tasksource audit effective classes."""

    def __init__(self):
        with open(ROOT / "data/license/per_source.csv") as f:
            self.csv = {r["source"]: r for r in csv.DictReader(f)}
        rs = json.loads((ROOT / "data/v5/replay_mc.summary.json").read_text())["by_source"]
        self.replay = {f"replay/{k}": v["effective_class"] for k, v in rs.items()}
        with open(NOTES / "tasksource_audit.csv") as f:
            rows = list(csv.DictReader(f))
        self.audit = {r["source"]: r["effective_class"] for r in rows}
        self.gen = {r["source"]: r["gen"] for r in rows if r.get("gen")}

    def v5(self, source: str) -> tuple[str, str]:
        """(class, flag)."""
        if source in self.replay:
            return self.replay[source], ""
        r = self.csv.get(source)
        cls, flag = (r["class"], r["flag"]) if r else ("UNK", "no per_source.csv row")
        if source.startswith("tasksource/") and source[len("tasksource/"):] in self.audit:
            cls = most_restrictive(cls, self.audit[source[len("tasksource/"):]])
        return cls, flag

    def gen_of(self, source: str) -> str:
        """The audit's proprietary-generator note of a tasksource source ("" if none; plan §6.1 item 2)."""
        return self.gen.get(source[len("tasksource/"):], "") if source.startswith("tasksource/") else ""


# ---------------------------------------------------------------------------------------------------------- gates

class Grams13:
    """scripts/decontam.py's rule in array form: evaluation 13-grams (decontam.grams) in >= boilerplate evaluation items
    are ignored; a row is contaminated if it shares a remaining 13-gram with an evaluation item. Kept as sorted
    (gram, file) pairs so the matching files are known (own-eval self-overlap) and memory stays ~100 MB."""

    def __init__(self, pattern: str, boilerplate: int = 20):
        self.paths = sorted(glob.glob(str(ROOT / pattern)))
        self.files = [Path(p).stem for p in self.paths]
        self.md5 = {Path(p).name: md5_file(Path(p)) for p in self.paths}
        hs, lens, fids = [], [], []
        for fi, p in enumerate(self.paths):
            for line in lines(p):
                if not line.strip():
                    continue
                g = grams13(json.loads(line).get("prompt", ""))
                if g:
                    hs.append(np.fromiter(g, dtype=np.int64, count=len(g)))
                    lens.append(len(g))
                    fids.append(fi)
        allh = np.concatenate(hs) if hs else np.zeros(0, np.int64)
        allf = np.repeat(np.array(fids, dtype=np.int16), lens)
        del hs
        uniq, counts = np.unique(allh, return_counts=True)
        self.n_grams = int(len(uniq))
        bad = uniq[counts >= boilerplate]
        del uniq, counts
        keep = ~np.isin(allh, bad)
        h, f = allh[keep], allf[keep]
        del allh, allf, keep
        o = np.lexsort((f, h))
        h, f = h[o], f[o]
        first = np.ones(len(h), bool)
        first[1:] = (h[1:] != h[:-1]) | (f[1:] != f[:-1])
        self.h, self.f = h[first], f[first]
        self.n_kept = int(len(np.unique(self.h)))

    def hits(self, texts: list[str]) -> set[str]:
        g = set()
        for t in texts:
            g |= grams13(t)
        if not g or not len(self.h):
            return set()
        a = np.fromiter(g, dtype=np.int64, count=len(g))
        lo = np.searchsorted(self.h, a, "left")
        hi = np.searchsorted(self.h, a, "right")
        m = hi > lo
        out = set()
        for x, y in zip(lo[m], hi[m]):
            out.update(self.files[i] for i in self.f[x:y].tolist())
        return out


def masked_view(gate, own: tuple[str, ...]):
    """A MaskedGate whose masked-n-gram references leave out the source's own evaluation files."""
    if not own:
        return gate
    ids = [i for i, n in enumerate(gate.names) if n.startswith(own)]
    if not ids:
        return gate
    v = copy.copy(gate)
    m = ~np.isin(gate.ref_o, ids)
    v.ref_h, v.ref_o = gate.ref_h[m], gate.ref_o[m]
    v.msets = [(n, s) for n, s in gate.msets if not n.startswith(own)]
    if len(v.msets) != len(gate.msets):     # rebuild the Jaccard index without them (positions change)
        idx = collections.defaultdict(list)
        for j, (_, s) in enumerate(v.msets):
            o = gate.order(s)
            for w in o[:len(o) - math.ceil(gate.jacc * len(o)) + 1]:
                idx[w].append(j)
        v.index = idx
    return v


class MinHash:
    """Deterministic MinHash over word shingles (crc32), for the cross-source near-duplicate check."""

    P = np.uint64(4294967311)

    def __init__(self, perms: int = 64, shingle: int = 5, seed: int = 1):
        rs = np.random.RandomState(seed)
        self.a = rs.randint(1, 2 ** 32 - 1, size=perms, dtype=np.uint64)
        self.b = rs.randint(0, 2 ** 32 - 1, size=perms, dtype=np.uint64)
        self.k, self.perms = shingle, perms

    def __call__(self, text: str) -> np.ndarray | None:
        w = TOK.findall(text.lower())
        if len(w) < self.k:
            return None
        sh = {zlib.crc32(" ".join(w[i:i + self.k]).encode()) for i in range(len(w) - self.k + 1)}
        x = np.fromiter(sh, dtype=np.uint64, count=len(sh))
        return ((self.a[:, None] * x[None, :] + self.b[:, None]) % self.P).min(axis=1)


# ---------------------------------------------------------------------------------------------------------- rows

class Meta:
    """The in-memory record of one written row."""
    __slots__ = ("src", "split", "pid", "item_id", "kind", "family", "chars", "lic", "lab_only", "flag", "exact",
                 "sig", "offset", "row_type", "n_subqs", "label_basis", "node_basis", "drop", "self_hits", "weight",
                 "rank", "policy", "gen")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def apply_label(r: dict, mode: str) -> bool:
    """Label policy; False = the row cannot be labelled this way (dropped, counted)."""
    labels, g = r.get("labels"), r.get("gold_label")
    if mode == "as_is":
        return isinstance(r.get("teacher"), dict)
    if not labels or g not in labels:
        return False
    onehot = {L: float(L == g) for L in labels}
    if mode == "gold":
        if r.get("label_basis") != "vote_share":
            r["teacher"] = onehot
        return True
    t = r.get("teacher")
    if t is None:
        r["teacher"] = onehot
        return True
    return all(abs(float(t.get(L, 0.0)) - v) < 1e-9 for L, v in onehot.items())


class Assembler:
    def __init__(self, cfg: dict, args):
        self.cfg, self.args = cfg, args
        self.seed = cfg.get("seed", 0)
        self.tmp = Path(args.tmp) / cfg["name"]
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.meta: list[Meta] = []
        self.sigs: list = []
        self.stats = collections.defaultdict(collections.Counter)
        self.inputs: dict[str, str] = {}
        self.selected_ids: set[str] = set()
        self.lic = Licences()
        self.mh = MinHash(**{k: v for k, v in {"perms": cfg.get("dedup", {}).get("perms", 64),
                                                "shingle": cfg.get("dedup", {}).get("shingle", 5)}.items()})
        self.g13 = None
        self.gate = None
        self.fill_order: list[str] = []      # problem keys of the fill source in rank order
        self.outfh: dict[tuple[int, str], object] = {}
        groups = [s.get("dedup_group", s["name"]) for s in cfg["sources"]]
        self.group = [groups.index(g) for g in groups]      # sources of one generator dedup as one source

    # ---------------------------------------------------------------- selection
    def n_for(self, s: dict):
        sel = s.get("select", {})
        if sel.get("fill"):
            return sel.get("candidates", "all")
        if "weight" in sel:
            tot = self.cfg.get("total_problems") or self.cfg.get("total_rows")
            return int(round(sel["weight"] * tot))
        n = sel.get("problems", sel.get("rows", "all"))
        if self.args.limit and (n == "all" or n > self.args.limit):
            return self.args.limit
        return n

    def problem_ids(self, s: dict, path: Path, pkey: str) -> list[tuple]:
        """[(rank key, problem id)] of the finals in a file that pass the source filters."""
        inc = re.compile(s["include_source"]) if s.get("include_source") else None
        exc = re.compile(s["exclude_source"]) if s.get("exclude_source") else None
        rank = s.get("select", {}).get("rank", "hash")
        where = s.get("where", {})
        out, seen = [], set()
        for i, line in enumerate(lines(path)):
            r = json.loads(line)
            if r.get("row_kind") == "node":
                continue
            if any(r.get(k) not in v for k, v in where.items()):
                self.stats[s["name"]]["filtered_by_where"] += 1
                continue
            src = r.get("source", "")
            if (inc and not inc.search(src)) or (exc and exc.search(src)):
                self.stats[s["name"]]["filtered_by_source_regex"] += 1
                continue
            pid = str(r.get(pkey, r["item_id"]))
            if pid in seen:
                continue
            seen.add(pid)
            key = (r["pool_rank"],) if rank == "pool_rank" else (i,) if rank == "file" else (
                hrank(self.seed, s.get("rank_key", s["name"]), pid),)
            out.append((key, pid))
        out.sort()
        return out

    def select(self, s: dict, path: Path, pkey: str) -> list[str]:
        ids = self.problem_ids(s, path, pkey)
        sel = s.get("select", {})
        ids = ids[sel.get("offset", 0):]
        if sel.get("skip_selected"):
            n0 = len(ids)
            ids = [x for x in ids if x[1] not in self.selected_ids]
            self.stats[s["name"]]["skipped_already_selected"] = n0 - len(ids)
        n = self.n_for(s)
        if n != "all":
            if len(ids) < n:
                self.stats[s["name"]]["supply_shortfall_problems"] = n - len(ids)
            ids = ids[:n]
        return [p for _, p in ids]

    def write_row(self, si: int, split: str, s: dict, r: dict, pid: str, kind: str, rank: int) -> None:
        name = s["name"]
        st = self.stats[name]
        if kind == "final" and split == "train" and s.get("nodes", "keep") == "drop":     # val keeps its subqs
            if r.get("subqs"):
                st["subqs_dropped"] += len(r["subqs"])
            # still screened, so a node-dropping arm keeps exactly the problems of its node-keeping twin
            r["_screen_texts"] = row_texts(r)[1:]
            r["subqs"], r["random_subqs"] = [], []
        if not apply_label(r, s.get("label", "as_is")):
            st[f"label_unusable:{split}"] += 1
            return
        lic = s.get("licence", {"from": "row"})
        flag, gen = "", ""
        if lic.get("from") == "v5":
            cls, flag = self.lic.v5(r.get("source", ""))
            gen = self.lic.gen_of(r.get("source", ""))
        elif lic.get("from") == "row":
            cls = r.get("licence_class") or "UNK"
        else:
            cls = lic["class"]
            if lic.get("text") and not r.get("licence"):
                r["licence"] = lic["text"]
        lab_only = bool(s.get("lab_only")) or cls not in RELEASE_OK or bool(flag)
        nodes_kept = s.get("nodes", "keep") == "keep" or split != "train"
        r["v7_source"], r["slice"] = name, s.get("slice", "?")
        r["licence_class"], r["lab_only"] = cls, lab_only
        if flag:
            r["licence_flag"] = flag
        if not r.get("label_basis"):
            t, g = r.get("teacher") or {}, r.get("gold_label")
            onehot = bool(t) and all(abs(float(v) - float(k == g)) < 1e-9 for k, v in t.items())
            r["label_basis"] = s.get("label_basis") or ("gold" if onehot else "teacher-soft")
        nb = r.get("node_basis") or s.get("node_basis", "none")
        if kind == "final":
            nb = nb if (nodes_kept and r.get("subqs")) else "none"
        r["node_basis"] = nb
        key = (si, split)
        if key not in self.outfh:
            self.outfh[key] = open(self.tmp / f"{si:02d}_{name}.{split}.jsonl", "w+b")
        fh = self.outfh[key]
        off = fh.tell()
        line = (json.dumps(r) + "\n").encode()
        fh.write(line)
        prompt = r.get("prompt", "")
        sig = self.mh(prompt)
        self.sigs.append(sig)
        self.meta.append(Meta(src=si, split=split, pid=pid, item_id=r["item_id"], kind=kind,
                              family=family_of(r, s.get("family_key")), chars=len(prompt), lic=cls,
                              lab_only=lab_only, flag=flag, exact=hrank(norm_text(prompt))[:16], sig=None,
                              offset=off, row_type=r.get("type", "choice"), n_subqs=len(r.get("subqs") or []),
                              label_basis=r["label_basis"], node_basis=nb, drop=None, self_hits=None,
                              weight=r.get("weight", 1.0), rank=rank, policy=bool(r.get("_policy_drop")), gen=gen))
        st[f"rows:{split}:{kind}" + (":screen_only" if r.get("_policy_drop") else "")] += 1

    def node_pick(self, s: dict, split: str, node_iter, wanted: set) -> dict[str, set]:
        """Node row ids to keep per problem (max_node_rows_per_problem by md5 rank; all when unset)."""
        cap = s.get("max_node_rows_per_problem")
        per = collections.defaultdict(list)
        for pid, iid in node_iter():
            if pid in wanted:
                per[pid].append(iid)
        out = {}
        for pid, ids in per.items():
            ids = sorted(ids, key=lambda x: hrank(self.seed, s["name"], split, x))
            out[pid] = set(ids[:cap] if cap else ids)
        return out

    def load(self, si: int, s: dict, split: str) -> None:
        """Select and write the rows of one source (train) or of its val spec."""
        pkey = s.get("problem_key", "item_id")
        link = s.get("node_link", "node_of")
        if split == "train":
            path = ROOT / s["train"]
            nodes_path = ROOT / s["node_rows"] if s.get("node_rows") else None
            pids = self.select(s, path, pkey)
        else:
            v = s["val"]
            path = ROOT / v["file"]
            nodes_path = ROOT / v["node_rows"] if v.get("node_rows") else None
            ids = self.problem_ids(dict(s, select={"rank": "hash"}), path, pkey)
            pids = self.val_sample(s, path, pkey, ids, v)
        for p in [path] + ([nodes_path] if nodes_path else []):
            self.inputs[rel(p)] = md5_file(p)
        order = {p: i for i, p in enumerate(pids)}
        wanted = set(pids)
        if split == "train":
            self.selected_ids |= wanted
        keep_nodes = s.get("nodes", "keep") == "keep" and (split == "train" or s["val"].get("node_rows_in_val"))
        # nodes "drop": the node rows are still read and screened (problem-level drops match the "keep" twin), then
        # left out ("policy:nodes")
        screen_nodes = keep_nodes or split == "train"

        def in_file_nodes():
            for line in lines(path):
                r = json.loads(line)
                if r.get("row_kind") == "node":
                    yield str(r.get(link, r.get("problem_id"))), r["item_id"]

        def file_nodes():
            for line in lines(nodes_path):
                r = json.loads(line)
                yield str(r.get(link)), r["item_id"]

        pick = {}
        if screen_nodes:
            pick = self.node_pick(s, split, in_file_nodes, wanted)
            if nodes_path:
                for k, v in self.node_pick(s, split, file_nodes, wanted).items():
                    pick.setdefault(k, set()).update(v)
        n_nodes = {k: len(v) for k, v in pick.items()}
        reweight = bool(s.get("max_node_rows_per_problem"))
        sources = [(path, False)] + ([(nodes_path, True)] if (nodes_path and screen_nodes) else [])
        for p, all_nodes in sources:
            for line in lines(p):
                r = json.loads(line)
                is_node = all_nodes or r.get("row_kind") == "node"
                pid = str(r.get(link) if is_node and r.get(link) is not None else
                          r.get("problem_id") if is_node else r.get(pkey, r["item_id"]))
                if pid not in wanted:
                    continue
                if is_node:
                    if r["item_id"] not in pick.get(pid, ()):
                        continue
                    if reweight:
                        r["weight"] = round(1.0 / n_nodes[pid], 6)
                    r.setdefault("row_kind", "node")
                    if not keep_nodes:
                        r["_policy_drop"] = True
                else:
                    src = r.get("source", "")
                    if (s.get("include_source") and not re.search(s["include_source"], src)) or \
                            (s.get("exclude_source") and re.search(s["exclude_source"], src)):
                        continue
                self.write_row(si, split, s, r, pid, "node" if is_node else "final", order[pid])

    def val_sample(self, s: dict, path: Path, pkey: str, ids: list, v: dict) -> list[str]:
        """Val problems of a source val file: per family at most max_per_family (md5 order), at most `problems`."""
        fam = {}
        for line in lines(path):
            r = json.loads(line)
            if r.get("row_kind") != "node":
                fam[str(r.get(pkey, r["item_id"]))] = family_of(r, s.get("family_key"))
        cap = v.get("max_per_family")
        n_fam = collections.Counter()
        out = []
        for _, pid in ids:
            f = fam.get(pid, "?")
            if cap and n_fam[f] >= cap:
                continue
            n_fam[f] += 1
            out.append(pid)
        if v.get("problems"):
            out = out[:v["problems"]]
        if self.args.limit:
            out = out[:self.args.limit]
        return out

    # ---------------------------------------------------------------- gates
    def decontaminate(self) -> None:
        srcs = self.cfg["sources"]
        need_masked = any("masked" in s.get("gates", ["13gram"]) for s in srcs)
        print(f"[v7] 13-gram index over {self.args.evals} ...", flush=True)
        self.g13 = Grams13(self.args.evals)
        print(f"[v7]   {len(self.g13.files)} eval files, {self.g13.n_grams} grams, {self.g13.n_kept} kept, "
              f"rss {rss_mb()} MB", flush=True)
        if need_masked:
            import build_breadth as bb
            self.gate = bb.MaskedGate()
            print(f"[v7]   masked gate ready, rss {rss_mb()} MB", flush=True)
        for (si, split), fh in sorted(self.outfh.items()):
            s = srcs[si]
            own = tuple(s.get("own_eval", []))
            gates = s.get("gates", ["13gram"])
            mv = masked_view(self.gate, own) if "masked" in gates else None
            idx = [i for i, m in enumerate(self.meta) if m.src == si and m.split == split]
            fh.flush()
            fh.seek(0)
            batch = []
            for i in idx:
                m = self.meta[i]
                fh.seek(m.offset)
                r = json.loads(fh.readline())
                if "13gram" in gates:
                    hits = self.g13.hits(row_texts(r))
                    bad = sorted(h for h in hits if not h.startswith(own)) if own else sorted(hits)
                    selfh = sorted(h for h in hits if own and h.startswith(own))
                    if bad:
                        m.drop = f"13gram:{bad[0]}"
                    if selfh:
                        m.self_hits = "13gram:" + ",".join(selfh)
                if mv is not None and m.drop is None:
                    batch.append({"item_id": str(i), "prompt": r.get("prompt", "")})
                    if len(batch) >= 2000:
                        self.masked(batch, mv, own)
                        batch = []
            if batch:
                self.masked(batch, mv, own)
            fh.seek(0, 2)
            dropped = collections.Counter(self.meta[i].drop for i in idx if self.meta[i].drop)
            print(f"[v7]   {s['name']}/{split}: {len(idx)} rows screened, {sum(dropped.values())} hit, "
                  f"rss {rss_mb()} MB", flush=True)

    def masked(self, batch: list[dict], mv, own) -> None:
        for k, why in mv(batch).items():
            self.meta[int(k)].drop = why
        if own and mv is not self.gate:         # template self-overlap: what the full gate would add
            rest = [b for b in batch if self.meta[int(b["item_id"])].drop is None]
            for k, why in self.gate(rest).items():
                m = self.meta[int(k)]
                m.self_hits = ((m.self_hits + ";") if m.self_hits else "") + why

    def propagate(self, reason_prefix: str | None = None) -> None:
        """A dropped row drops its whole problem (all rows of the same source, split and problem id)."""
        bad = {}
        for m in self.meta:
            if m.drop and (reason_prefix is None or m.drop.startswith(reason_prefix)):
                bad.setdefault((m.src, m.split, m.pid), m.drop)
        for m in self.meta:
            k = (m.src, m.split, m.pid)
            if m.drop is None and k in bad:
                m.drop = "problem:" + bad[k]

    def licence_filter(self) -> None:
        if self.cfg["tier"] != "release":
            return
        allow_gen = getattr(self.args, "allow_gen", False)
        for m in self.meta:
            if m.drop is None and (m.lab_only or m.lic not in RELEASE_OK or m.flag):
                m.drop = f"licence:{m.lic}{'/' + m.flag if m.flag else ''}"
            elif m.drop is None and m.gen and not allow_gen:
                m.drop = "licence:GEN"

    def dedup(self) -> None:
        d = self.cfg.get("dedup", {})
        thr, bands = d.get("near", .8), d.get("bands", 16)
        # node rows a "drop" arm leaves out (policy:nodes) take part exactly as in the "keep" twin
        live = [i for i, m in enumerate(self.meta) if m.drop is None or m.drop == POLICY]
        # exact: normalised prompt or item id, across sources (train), and val vs any train row
        first: dict[str, int] = {}
        for i in sorted(live, key=lambda i: (self.meta[i].split != "train", self.meta[i].src, i)):
            m = self.meta[i]
            for key in ("p:" + m.exact, "i:" + m.item_id):
                j = first.get(key)
                if j is None:
                    first[key] = i
                    continue
                o = self.meta[j]
                if m.split == "train" and self.group[o.src] == self.group[m.src]:
                    self.stats["_dedup"]["within_source_exact_kept"] += key.startswith("p:")
                    continue
                if m.split == "val" and o.split == "val":
                    continue
                m.drop = f"dup_exact:{self.cfg['sources'][o.src]['name']}"
                break
        self.propagate("dup_exact")
        live = [i for i, m in enumerate(self.meta) if (m.drop is None or m.drop == POLICY) and self.sigs[i] is not None]
        if not live:
            return
        sig = np.stack([self.sigs[i] for i in live])
        per = sig.shape[1] // bands
        prio = np.array([(self.meta[i].split != "train") * 1000 + self.meta[i].src for i in live])
        src = np.array([self.group[self.meta[i].src] for i in live])
        split_val = np.array([self.meta[i].split == "val" for i in live])
        dropped = np.zeros(len(live), bool)
        n_cmp = 0
        for b in range(bands):
            band = np.ascontiguousarray(sig[:, b * per:(b + 1) * per])
            _, inv = np.unique(band, axis=0, return_inverse=True)
            inv = inv.ravel()
            o = np.argsort(inv, kind="stable")
            cut = np.flatnonzero(np.diff(inv[o])) + 1
            for grp in np.split(o, cut):
                if len(grp) < 2 or len(set(src[grp].tolist())) < 2 and not split_val[grp].any():
                    continue
                grp = grp[np.argsort(prio[grp], kind="stable")]
                for k in range(1, len(grp)):
                    i = grp[k]
                    if dropped[i]:
                        continue
                    hi = grp[:k]
                    # cross-source only for train; val vs other sources' rows (same-source val is exact-only)
                    hi = hi[(src[hi] != src[i]) & ~split_val[hi] & ~dropped[hi]][:64]
                    if not len(hi):
                        continue
                    n_cmp += len(hi)
                    est = (sig[hi] == sig[i]).mean(axis=1)
                    if est.max() >= thr:
                        dropped[i] = True
                        j = live[int(hi[int(est.argmax())])]
                        self.meta[live[i]].drop = f"dup_near:{self.cfg['sources'][self.meta[j].src]['name']}"
        self.stats["_dedup"]["near_comparisons"] = n_cmp
        self.propagate("dup_near")

    # ---------------------------------------------------------------- fill, val split, caps
    def math_chars(self, split: str = "train") -> int:
        srcs = self.cfg["sources"]
        return sum(m.chars for m in self.meta if m.drop is None and m.split == split and srcs[m.src].get("math"))

    def fill(self) -> dict | None:
        f = self.cfg.get("fill")
        if not f:
            return None
        names = [s["name"] for s in self.cfg["sources"]]
        fi = names.index(f["source"])
        if f["target"] == "rows":
            target = f["rows"] if not self.args.limit else min(f["rows"], self.args.limit)
            have = sum(1 for m in self.meta if m.drop is None and m.split == "train" and m.src != fi)
            unit = "rows"
        else:
            ref = ROOT / f["reference"] / "summary.json"
            if not ref.exists():
                sys.exit(f"fill reference {ref} not built yet: build that config first")
            rs = json.loads(ref.read_text())
            target = rs["compute_chars"]["math_train"]
            have = sum(m.chars for m in self.meta if m.drop is None and m.split == "train" and m.src != fi
                       and self.cfg["sources"][m.src].get("math"))
            unit = "math_chars"
            self.inputs[rel(ref)] = md5_file(ref)
        rows = collections.defaultdict(list)
        for i, m in enumerate(self.meta):
            if m.src == fi and m.split == "train" and not (m.drop or "").startswith("policy:"):
                rows[m.rank].append(i)
        taken = 0
        for rk in sorted(rows):
            idx = rows[rk]
            if have >= target:
                for i in idx:
                    if self.meta[i].drop is None:
                        self.meta[i].drop = "fill:not_needed"
                continue
            if any(self.meta[i].drop for i in idx):
                continue
            have += len(idx) if unit == "rows" else sum(self.meta[i].chars for i in idx)
            taken += 1
        out = {"source": f["source"], "unit": unit, "target": target, "achieved": have,
               "ratio": round(have / max(target, 1), 4), "problems_taken": taken,
               "candidates": len(rows), "shortfall": max(target - have, 0)}
        if unit == "math_chars":
            out["reference"] = f["reference"]
            out["reference_md5"] = rs["md5"]
        if f.get("min_ratio") is not None:
            out["min_ratio"] = f["min_ratio"]
            out["below_min_ratio"] = out["ratio"] < f["min_ratio"]
        if out["shortfall"]:
            print(f"[v7] WARNING fill short: {out}", flush=True)
        return out

    def val_split(self) -> None:
        v = self.cfg.get("val", {"mode": "sources"})
        if v["mode"] != "split":
            return
        fam = collections.defaultdict(set)
        for m in self.meta:
            if m.drop is None and m.split == "train" and m.kind == "final":
                fam[m.family].add((m.src, m.pid))
        chosen = set()
        for f, probs in sorted(fam.items()):
            k = max(v.get("min_per_family", 1), math.ceil(v.get("frac", .02) * len(probs)))
            k = min(k, v.get("max_per_family", 10 ** 9), len(probs) - 1)
            chosen |= set(sorted(probs, key=lambda p: hrank(self.seed, "val", p[0], p[1]))[:max(k, 0)])
        for m in self.meta:
            if (m.src, m.pid) in chosen and m.split == "train":
                m.split = "val_from_train"

    def caps(self, slices: dict) -> list[str]:
        c = self.cfg.get("caps", {})
        out = []
        for sl, mx in c.get("slice_share_chars", {}).items():          # "M+L+P": the summed share
            share = round(sum(slices.get(x, {}).get("share_chars", 0) for x in sl.split("+")), 4)
            if share > mx:
                out.append(f"slice {sl} share of chars {share} > {mx}")
        if "gsm_share_of_math_problems" in c:
            srcs = self.cfg["sources"]
            probs = {(m.src, m.pid) for m in self.meta if m.drop is None and m.split == "train"
                     and srcs[m.src].get("math")}
            gsm = sum(srcs[s].get("gsm_derived", False) for s, _ in probs)
            share = gsm / max(len(probs), 1)
            if share > c["gsm_share_of_math_problems"]:
                out.append(f"GSM8K-derived share of math problems {share:.3f} > {c['gsm_share_of_math_problems']}")
        return out

    # ---------------------------------------------------------------- output
    def write(self, fill_info) -> dict:
        cfg, srcs = self.cfg, self.cfg["sources"]
        out = ROOT / (self.args.out or cfg["out"])
        out.mkdir(parents=True, exist_ok=True)
        for fh in self.outfh.values():
            fh.flush()
        readers = {k: open(self.tmp / f"{k[0]:02d}_{srcs[k[0]]['name']}.{k[1]}.jsonl", "rb") for k in self.outfh}
        vmode = cfg.get("val", {"mode": "sources"})
        files = {"train": out / "train.jsonl", "val": out / "val.jsonl"}
        for split, path in files.items():
            if split == "val" and vmode["mode"] == "copy":
                shutil.copyfile(ROOT / vmode["path"], path)
                self.inputs[vmode["path"]] = md5_file(ROOT / vmode["path"])
                continue
            want = ("train",) if split == "train" else ("val", "val_from_train")
            idx = [i for i, m in enumerate(self.meta) if m.drop is None and m.split in want]
            # order from (source, item id) only, so a set shared by two arms (e.g. the val of MS and MA) is byte-identical
            idx.sort(key=lambda i: (hrank(cfg.get("seed", 0), "order", srcs[self.meta[i].src]["name"],
                                          self.meta[i].item_id), self.meta[i].src, self.meta[i].offset))
            with open(path, "wb") as f:
                for i in idx:
                    m = self.meta[i]
                    rd = readers[(m.src, "train" if m.split == "val_from_train" else m.split)]
                    rd.seek(m.offset)
                    line = rd.readline()
                    r = json.loads(line)
                    r.pop("_screen_texts", None)
                    r.pop("_policy_drop", None)
                    r["decontam"] = ("clean: 13-gram (decontam.py rule, every data/eval file)" +
                                     (" + digit-masked 10-gram + math Jaccard>=.5" if "masked" in
                                      srcs[m.src].get("gates", ["13gram"]) else "") +
                                     (f"; own-eval template overlap {m.self_hits}" if m.self_hits else ""))
                    f.write((json.dumps(r) + "\n").encode())
        for rd in list(readers.values()) + list(self.outfh.values()):
            rd.close()
        return self.summary(out, files, fill_info)

    def summary(self, out: Path, files: dict, fill_info) -> dict:
        cfg, srcs = self.cfg, self.cfg["sources"]
        kept = [m for m in self.meta if m.drop is None and m.split == "train"]
        val = [m for m in self.meta if m.drop is None and m.split in ("val", "val_from_train")]
        tot_rows, tot_chars = len(kept), sum(m.chars for m in kept)
        per_source = {}
        for si, s in enumerate(srcs):
            ms = [m for m in self.meta if m.src == si]
            k = [m for m in ms if m.drop is None and m.split == "train"]
            fin = [m for m in k if m.kind == "final"]
            st = self.stats[s["name"]]
            drops = collections.Counter(m.drop.split(":")[0] if not m.drop.startswith("problem:") else
                                        "problem:" + m.drop.split(":")[1] for m in ms if m.drop and m.split == "train")
            drops_detail = collections.Counter(m.drop for m in ms if m.drop and m.split == "train"
                                               and not m.drop.startswith(("problem:", "fill:")))
            per_source[s["name"]] = {
                "slice": s.get("slice"), "train_file": s["train"], "node_rows_file": s.get("node_rows"),
                "selected_problems": len({m.pid for m in ms if m.split == "train"}),
                "kept_problems": len({m.pid for m in k}),
                "rows": len(k), "rows_by_kind": dict(collections.Counter(m.kind for m in k)),
                "rows_by_type": dict(collections.Counter(m.row_type for m in k)),
                "chars": sum(m.chars for m in k),
                "share_rows": round(len(k) / max(tot_rows, 1), 4),
                "share_chars": round(sum(m.chars for m in k) / max(tot_chars, 1), 4),
                "node_policy": s.get("nodes", "keep"), "label_policy": s.get("label", "as_is"),
                "node_coverage": {
                    "finals_with_subqs": round(sum(m.n_subqs > 0 for m in fin) / max(len(fin), 1), 4),
                    "mean_subqs_per_final": round(sum(m.n_subqs for m in fin) / max(len(fin), 1), 3),
                    "node_rows": sum(m.kind == "node" for m in k),
                    "node_rows_per_problem": round(sum(m.kind == "node" for m in k) / max(len(fin), 1), 3),
                    "node_row_weight_sum": round(sum(m.weight or 1.0 for m in k if m.kind == "node"), 2)},
                "licence_class": dict(collections.Counter(m.lic for m in k)),
                "lab_only_rows": sum(m.lab_only for m in k),
                "gen_rows": sum(bool(m.gen) for m in k),
                "label_basis": dict(collections.Counter(m.label_basis for m in k)),
                "node_basis": dict(collections.Counter(m.node_basis for m in k)),
                "dropped_rows": dict(drops), "dropped_by_reason": dict(drops_detail.most_common(30)),
                "own_eval_self_overlap_rows": sum(bool(m.self_hits) for m in k),
                "own_eval_self_overlap_by": dict(collections.Counter(
                    h for m in k if m.self_hits for h in re.findall(r"[a-z0-9_]+(?=[,;]|$)", m.self_hits.split(":", 1)[1])
                ).most_common(12)) if any(m.self_hits for m in k) else {},
                "val_rows": sum(1 for m in val if m.src == si),
                "kept_item_ids_md5": hashlib.md5("\n".join(sorted(m.item_id for m in k)).encode()).hexdigest(),
                "families": len({m.family for m in k}),
                **{k2: v for k2, v in st.items()},
            }
        slices = collections.defaultdict(lambda: {"rows": 0, "chars": 0})
        for m in kept:
            sl = srcs[m.src].get("slice", "?")
            slices[sl]["rows"] += 1
            slices[sl]["chars"] += m.chars
        for sl in slices.values():
            sl["share_rows"] = round(sl["rows"] / max(tot_rows, 1), 4)
            sl["share_chars"] = round(sl["chars"] / max(tot_chars, 1), 4)
        cap_viol = self.caps(slices)
        math_probs = {(m.src, m.pid) for m in kept if srcs[m.src].get("math")}
        gsm_probs = sum(srcs[s].get("gsm_derived", False) for s, _ in math_probs)
        lens = sorted(m.chars for m in kept) or [0]
        n_sub = sum(min(3, m.n_subqs) for m in kept)
        summ = {
            "builder": "scripts/build_v7.py", "builder_md5": md5_file(Path(__file__)), "config": cfg["name"],
            "tier": cfg["tier"],
            "description": cfg.get("description", ""), "plan": cfg.get("plan", ""),
            "args": {k: v for k, v in vars(self.args).items()}, "seed": cfg.get("seed", 0),
            "rows": {"train": len(kept), "val": len(val) if cfg.get("val", {}).get("mode") != "copy" else
                     sum(1 for _ in lines(files["val"]))},
            "rows_by_kind": dict(collections.Counter(m.kind for m in kept)),
            "rows_by_type": dict(collections.Counter(m.row_type for m in kept)),
            "slices": dict(slices),
            "licence_class": dict(collections.Counter(m.lic for m in kept)),
            "lab_only_rows": sum(m.lab_only for m in kept),
            "licence_gen_rows": sum(bool(m.gen) for m in kept),
            "label_basis": dict(collections.Counter(m.label_basis for m in kept)),
            "node_basis": dict(collections.Counter(m.node_basis for m in kept)),
            "node_coverage": {
                "finals": sum(m.kind == "final" for m in kept),
                "finals_with_subqs": sum(m.kind == "final" and m.n_subqs > 0 for m in kept),
                "subqs": sum(m.n_subqs for m in kept), "node_rows": sum(m.kind == "node" for m in kept)},
            "math": {"problems": len(math_probs), "gsm_derived_problems": gsm_probs,
                     "gsm_share_of_math_problems": round(gsm_probs / max(len(math_probs), 1), 4)},
            "compute_chars": {
                "train": tot_chars, "math_train": self.math_chars(),
                "note": "sum of prompt characters; with --grouped-aux K every row costs (1+K) views (missing views are "
                        "zero-weight copies of the final), so arms trained with the same K compare on this sum",
                "subq_views_at_k3": n_sub},
            "prompt_chars": {"mean": round(tot_chars / max(tot_rows, 1)), "median": lens[len(lens) // 2],
                             "p95": lens[int(.95 * (len(lens) - 1))], "max": lens[-1]},
            "fill": fill_info,
            "caps": {"config": cfg.get("caps", {}), "violations": cap_viol},
            "val": {"mode": cfg.get("val", {"mode": "sources"}).get("mode", "sources"),
                    "families": dict(collections.Counter(m.family for m in val if m.kind == "final").most_common(40)),
                    "val_dropped": dict(collections.Counter(m.drop.split(":")[0] for m in self.meta
                                                            if m.drop and m.split != "train"))},
            "dedup": {"config": cfg.get("dedup", {}), **dict(self.stats["_dedup"]),
                      "exact_dropped_rows": sum(1 for m in self.meta if m.drop and m.drop.startswith("dup_exact")),
                      "near_dropped_rows": sum(1 for m in self.meta if m.drop and m.drop.startswith("dup_near")),
                      "pairs": dict(collections.Counter(
                          f"{srcs[m.src]['name']}<-{m.drop.split(':')[1]}" for m in self.meta
                          if m.drop and m.drop.startswith("dup_")))},
            "decontam": {
                "eval_files": len(self.g13.files), "eval_13grams": self.g13.n_grams,
                "eval_13grams_kept": self.g13.n_kept, "boilerplate_items": 20, "n": N13,
                "masked_gate_references": getattr(self.gate, "ref_counts", None),
                "by_reason": dict(collections.Counter(m.drop for m in self.meta if m.drop and m.drop.startswith(
                    ("13gram", "masked", "jaccard"))).most_common(40)),
                "eval_md5": self.g13.md5},
            "per_source": per_source,
            "inputs_md5": self.inputs,
            "config_md5": hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
            "md5": {"train": md5_file(files["train"]), "val": md5_file(files["val"])},
        }
        (out / "summary.json").write_text(json.dumps(summ, indent=1))
        (out / "per_source.json").write_text(json.dumps(dict(collections.Counter(
            srcs[m.src]["name"] + "/" + m.family for m in kept).most_common()), indent=1))
        (out / "config.json").write_text(json.dumps(cfg, indent=1))
        return summ

    def run(self) -> dict:
        srcs = self.cfg["sources"]
        for si, s in enumerate(srcs):
            self.load(si, s, "train")
            print(f"[v7] {s['name']}: {dict(self.stats[s['name']])}, rss {rss_mb()} MB", flush=True)
        if self.cfg.get("val", {"mode": "sources"}).get("mode", "sources") == "sources":
            for si, s in enumerate(srcs):
                if s.get("val"):
                    self.load(si, s, "val")
        self.decontaminate()
        self.propagate()
        for m in self.meta:
            if m.policy and m.drop is None:
                m.drop = POLICY
        self.licence_filter()
        self.dedup()
        fill_info = self.fill()
        if fill_info and fill_info.get("below_min_ratio") and not getattr(self.args, "allow_short_fill", False):
            for fh in self.outfh.values():
                fh.close()
            sys.exit(f"fill below min_ratio {fill_info['min_ratio']} (refusing to write; --allow-short-fill builds "
                     f"it anyway): {fill_info}")
        self.val_split()
        summ = self.write(fill_info)
        if summ["caps"]["violations"] and self.cfg["tier"] == "release":
            sys.exit(f"cap violations (release refuses): {summ['caps']['violations']}")
        return summ


def rss_mb() -> int:
    try:
        with open("/proc/self/status") as f:
            return next(int(l.split()[1]) // 1024 for l in f if l.startswith("VmRSS"))
    except OSError:
        return -1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default="", help="output dir (default: the config's out)")
    ap.add_argument("--tmp", default=DEFAULT_TMP)
    ap.add_argument("--evals", default="data/eval/*.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: at most this many problems per source")
    ap.add_argument("--check-only", action="store_true", help="validate the config (TODO fields, structure) and stop")
    ap.add_argument("--allow-short-fill", action="store_true",
                    help="build even when the fill falls below the config's fill.min_ratio (marked in summary.json)")
    ap.add_argument("--allow-gen", action="store_true",
                    help="release: keep tasksource rows the audit marks as generated by a proprietary model (owner "
                         "decision, plan §6.1)")
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())
    todos = todo_paths(cfg)
    if todos:
        print(f"{args.config}: refusing to build, {len(todos)} TODO field(s) left:\n  " + "\n  ".join(todos),
              file=sys.stderr)
        return 2
    err = check_config(cfg)
    if err:
        print(f"{args.config}: invalid config:\n  " + "\n  ".join(err), file=sys.stderr)
        return 2
    if args.check_only:
        print(f"{args.config}: ok")
        return 0
    summ = Assembler(cfg, args).run()
    print(json.dumps({k: summ[k] for k in ("config", "rows", "rows_by_kind", "slices", "licence_class",
                                           "lab_only_rows", "math", "compute_chars", "fill", "caps", "md5")},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
