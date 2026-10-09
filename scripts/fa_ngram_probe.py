"""N-gram surface probe for failure-attribution step files (build_fa_public.py layout): does any word n-gram of the
rendered step texts predict the gold step beyond the position prior? CPU only, numpy only, no model.

Steps are parsed back from the rendered prompt ('Step i [Agent]: text' blocks, the text the model sees after the cut).
Group = fa origin/subset, with /benchmark appended for aftraj and agentracer (build_fa_public.debias_group); every number
is computed within a group (a model reads the framework off the log, so the fair prior is the group's position prior)
and pooled over groups by summing hits ("all").

Per group with >= --min-group train trajectories:
  position   conditional logit over a trajectory's steps, position features of build_fa_public.step_features
  structure  position + speaker change + one-hots of the group's --max-agents most frequent agent names (a step's role:
             tool output or agent reasoning; lexical cues that only mark the role are absorbed here)
  ngram      position + binary word n-grams of the step text (n = 1..3; lower-cased letter runs and digit runs, digit
             runs -> '#'; hashed to 2^20 buckets) that occur in >= --min-df trajectories of the fit set; L2 penalty
             picked from --lams by the within-train CV
  cv         both cross-fitted within train (--folds folds by task, stable hash of the normalized Task line)
  eval       both fit on all of the group's train trajectories, applied to the group's trajectories of every eval file
  phrases    single-phrase rules on top of the position prior: "the position-probe argmax among the steps that contain
             the phrase" (dir +) or "... that do not contain it" (dir -), against the position-probe argmax, on the
             trajectories where the phrase occurs in some but not all steps (cross-fitted position probabilities).
             gain = (rule hits - prior hits) / group size, in points of the group's accuracy.
             top: the in-sample top --top phrases (for reading); selected: per fold the best phrase on the other folds,
             scored on the held-out fold (selection-honest: with ~10^4 candidates the in-sample maximum is biased up);
             eval: the in-sample top phrases scored on each eval file (prior fit on train). The selected gain is
             also reported on top of the structure prior (phrases_over_structure).
Eval-file trajectories of groups without train data are listed under uncovered.

  python scripts/fa_ngram_probe.py --train data/student_fa1b/train.jsonl \\
      --eval val=data/student_fa1b/val.jsonl agtr=data/eval/fa1_agentracer_test.jsonl --out <json>
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cotdistill import fa_public as fp  # noqa: E402

WORD = re.compile(r"[^\W\d_]+|\d+")
NB = 20                                         # 2^20 hash buckets
GOLDEN = np.uint64(0x9E3779B97F4A7C15)
NPOS = 16


# ------------------------------------------------------------------------------------------------- items
def step_texts(item: dict) -> list[str]:
    """The rendered text of every step of a step-level item, from its prompt (headers from label_names, in order)."""
    names = item["label_names"]
    refs = [names[L] for L in item["labels"]]
    p = item["prompt"]
    end = p.rfind("\n\n" + fp.Q_STEP)
    log = p[:end] if end >= 0 else p
    heads, pos = [], 0
    for r in refs:
        h = "\n" + r + ": "
        k = log.find(h, pos)
        if k < 0:
            raise ValueError(f"{item['item_id']}: step header {r!r} not found")
        heads.append((k, k + len(h)))
        pos = k + len(h)
    return [log[c:(heads[i + 1][0] if i + 1 < len(heads) else len(log))] for i, (_, c) in enumerate(heads)]


def group_of(fa: dict) -> str:
    key = f"{fa.get('origin')}/{fa.get('subset')}"
    return key + f"/{fa.get('benchmark')}" if fa.get("origin") in ("aftraj", "agentracer") else key


def task_of(prompt: str) -> str:
    """The rendered Task block (up to the reference line or the first step), normalized: the fold key."""
    head = prompt.split("\n\nStep 0 [", 1)[0]
    head = head.split("\nTask: ", 1)[-1].split("\n" + fp.GT_PREFIX, 1)[0]
    return " ".join(head.lower().split())


def read_file(path: Path) -> list[dict]:
    """Trajectories of a step-level item file: {id, group, task, steps (texts), gold}."""
    out = []
    with open(path) as f:
        for line in f:
            it = json.loads(line)
            if not it.get("fa") or str(it.get("group", "")).endswith("/agent"):
                continue
            texts = step_texts(it)
            agents = [re.match(r"Step \d+ \[(.*)\]$", it["label_names"][L]).group(1) for L in it["labels"]]
            out.append({"id": it["item_id"], "group": group_of(it["fa"]), "task": task_of(it["prompt"]),
                        "steps": texts, "agents": agents, "gold": it["labels"].index(it["gold_label"])})
    return out


# ------------------------------------------------------------------------------------------------- features
class Hasher:
    """Word ids (a growing vocabulary) and hashed 1..3-gram bucket ids per step text."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def words(self, text: str) -> list[str]:
        return ["#" if w[0].isdigit() else w for w in WORD.findall(text.lower())]

    def buckets(self, text: str) -> np.ndarray:
        ids = np.array([self.vocab.setdefault(w, len(self.vocab) + 1) for w in self.words(text)], dtype=np.uint64)
        keys = [ids * np.uint64(4) + np.uint64(1)]
        if len(ids) > 1:
            keys.append(((ids[:-1] << np.uint64(21)) + ids[1:]) * np.uint64(4) + np.uint64(2))
        if len(ids) > 2:
            keys.append(((((ids[:-2] << np.uint64(21)) + ids[1:-1]) << np.uint64(21)) + ids[2:]) * np.uint64(4)
                        + np.uint64(3))
        k = np.concatenate(keys) if keys else np.zeros(0, dtype=np.uint64)
        return np.unique((k * GOLDEN) >> np.uint64(64 - NB)).astype(np.int32)

    def grams_of_bucket(self, texts: list[str], want: set, enough: int = 50) -> dict:
        """bucket -> Counter of the n-gram strings that hash to it (for reading the selected buckets; the scan stops
        once every wanted bucket has `enough` occurrences)."""
        out: dict = collections.defaultdict(collections.Counter)
        mask, gold = 2 ** 64 - 1, int(GOLDEN)
        left = set(want)
        for t in texts:
            if not left:
                break
            ws = self.words(t)
            ids = [self.vocab.get(w, 0) for w in ws]
            for n in (1, 2, 3):
                for i in range(len(ws) - n + 1):
                    key = 0
                    for x in ids[i:i + n]:
                        key = (key << 21) + x
                    b = ((((key * 4 + n) & mask) * gold) & mask) >> (64 - NB)
                    if b in want:
                        out[b][" ".join(ws[i:i + n])] += 1
                        if b in left and sum(out[b].values()) >= enough:
                            left.discard(b)
        return out


def pos_features(n: int) -> np.ndarray:
    """build_fa_public.step_features at level position (relative position, its square, first / last / second-to-last /
    step 1, index one-hots 0..9)."""
    rows = []
    for i in range(n):
        rel = i / max(1, n - 1)
        rows.append([rel, rel * rel, i == 0, i == n - 1, i == n - 2, i == 1] + [min(i, 9) == k for k in range(10)])
    return np.array(rows, dtype=np.float64)


def struct_features(agents: list[str], vocab: list[str]) -> np.ndarray:
    """Speaker change and agent-name one-hots (the group's most frequent train agents) per step."""
    rows = []
    for i, a in enumerate(agents):
        rows.append([a != (agents[i - 1] if i else None)] + [a == v for v in vocab])
    return np.array(rows, dtype=np.float64)


class Block:
    """A set of trajectories in flat step layout: dense features (position; with an agent vocabulary also speaker
    change and agent one-hots: the structure level), sparse n-gram feature ids (CSR), gold."""

    def __init__(self, trajs: list[dict], fmap: dict | None, agents: list[str] | None = None):
        self.n = len(trajs)
        sizes = np.array([len(t["steps"]) for t in trajs])
        self.start = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        self.sizes = sizes
        self.S = int(sizes.sum())
        self.traj = np.repeat(np.arange(self.n), sizes)
        self.Xd = np.concatenate([pos_features(int(k)) for k in sizes]) if self.n else np.zeros((0, NPOS))
        if agents is not None and self.n:
            self.Xd = np.hstack([self.Xd, np.concatenate([struct_features(t["agents"], agents) for t in trajs])])
        self.gold = np.array([t["gold"] for t in trajs], dtype=np.int64)
        self.gstep = self.start + self.gold
        if fmap is None:
            self.indptr = np.zeros(self.S + 1, dtype=np.int64)
            self.idx = np.zeros(0, dtype=np.int32)
        else:
            cols, ptr = [], [0]
            for t in trajs:
                for b in t["buckets"]:
                    c = fmap[b]
                    c = c[c >= 0]
                    cols.append(c)
                    ptr.append(ptr[-1] + len(c))
            self.indptr = np.array(ptr, dtype=np.int64)
            self.idx = np.concatenate(cols).astype(np.int32) if cols else np.zeros(0, dtype=np.int32)
        self.row = np.repeat(np.arange(self.S), np.diff(self.indptr))

    def subset(self, sel: np.ndarray) -> "Block":
        b = Block.__new__(Block)
        sel = np.asarray(sel)
        steps = np.concatenate([np.arange(self.start[t], self.start[t] + self.sizes[t]) for t in sel]) \
            if len(sel) else np.zeros(0, dtype=np.int64)
        b.n, b.sizes = len(sel), self.sizes[sel]
        b.start = np.concatenate([[0], np.cumsum(b.sizes)[:-1]]).astype(np.int64)
        b.S = int(b.sizes.sum())
        b.traj = np.repeat(np.arange(b.n), b.sizes)
        b.Xd = self.Xd[steps]
        b.gold = self.gold[sel]
        b.gstep = b.start + b.gold
        lens = np.diff(self.indptr)[steps]
        b.indptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        b.idx = np.concatenate([self.idx[self.indptr[s]:self.indptr[s + 1]] for s in steps]).astype(np.int32) \
            if len(steps) else np.zeros(0, dtype=np.int32)
        b.row = np.repeat(np.arange(b.S), lens)
        return b


# ------------------------------------------------------------------------------------------------- conditional logit
def scores(B: Block, w: np.ndarray) -> np.ndarray:
    D = B.Xd.shape[1]
    z = B.Xd @ w[:D]
    if len(B.idx):
        z = z + np.bincount(B.row, weights=w[D:][B.idx], minlength=B.S)
    return z


def probs(B: Block, w: np.ndarray) -> np.ndarray:
    z = scores(B, w)
    zmax = np.maximum.reduceat(z, B.start)
    e = np.exp(z - zmax[B.traj])
    return e / np.add.reduceat(e, B.start)[B.traj]


def loss_grad(B: Block, w: np.ndarray, lam_d: float, lam_s: float):
    p = probs(B, w)
    loss = -np.log(np.maximum(p[B.gstep], 1e-300)).mean()
    r = p.copy()
    r[B.gstep] -= 1.0
    r /= B.n
    D = B.Xd.shape[1]
    g = np.zeros_like(w)
    g[:D] = B.Xd.T @ r + lam_d * w[:D]
    if len(w) > D:
        g[D:] = (np.bincount(B.idx, weights=r[B.row], minlength=len(w) - D) if len(B.idx) else 0) + lam_s * w[D:]
    loss += lam_d / 2 * w[:D] @ w[:D] + lam_s / 2 * w[D:] @ w[D:]
    return loss, g


def fit(B: Block, n_feat: int, lam_s: float, lam_d: float = 1e-3, iters: int = 150, m: int = 10) -> np.ndarray:
    """L-BFGS (two-loop recursion, Armijo backtracking) on the mean conditional-logit NLL + L2."""
    w = np.zeros(B.Xd.shape[1] + n_feat)
    f, g = loss_grad(B, w, lam_d, lam_s)
    S, Y = [], []
    for _ in range(iters):
        q = g.copy()
        al = []
        for s, y in reversed(list(zip(S, Y))):
            a = (s @ q) / (y @ s)
            al.append(a)
            q -= a * y
        if S:
            q *= (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1])
        for (s, y), a in zip(zip(S, Y), reversed(al)):
            b = (y @ q) / (y @ s)
            q += (a - b) * s
        d = -q
        if g @ d >= 0:
            d, S, Y = -g, [], []
        t = 1.0
        while True:
            w2 = w + t * d
            f2, g2 = loss_grad(B, w2, lam_d, lam_s)
            if f2 <= f + 1e-4 * t * (g @ d) or t < 1e-8:
                break
            t /= 2
        s, y = w2 - w, g2 - g
        if s @ y > 1e-12:
            S.append(s)
            Y.append(y)
            if len(S) > m:
                S.pop(0)
                Y.pop(0)
        done = f - f2 < 1e-7
        w, f, g = w2, f2, g2
        if done:
            break
    return w


def hits(B: Block, w: np.ndarray) -> np.ndarray:
    """1 where the argmax step (first maximum) is the gold step, per trajectory."""
    z = scores(B, w)
    best = np.array([int(np.argmax(z[s:s + k])) for s, k in zip(B.start, B.sizes)], dtype=np.int64)
    return (best == B.gold).astype(np.int64)


# ------------------------------------------------------------------------------------------------- single phrases
def rank_layout(B: Block, p: np.ndarray):
    """Per step its rank in its trajectory by descending p (stable: the earlier step wins ties), per trajectory the
    gold step's rank."""
    rank = np.zeros(B.S, dtype=np.int64)
    for s, k in zip(B.start, B.sizes):
        order = np.argsort(-p[s:s + k], kind="stable")
        rank[s + order] = np.arange(k)
    return rank, rank[B.gstep]


def phrase_gains(B: Block, rank: np.ndarray, grank: np.ndarray, n_feat: int, mask: np.ndarray | None = None):
    """Per feature: hits gained over the prior argmax by the + rule (best-ranked step containing it) and the - rule
    (best-ranked step not containing it), summed over the trajectories (mask) where it occurs in some but not all
    steps; and the number of such trajectories."""
    if not len(B.idx):
        z = np.zeros(n_feat, dtype=np.int64)
        return z, z, z
    t = B.traj[B.row]
    keep = np.ones(len(t), dtype=bool) if mask is None else mask[t]
    f, t, r = B.idx[keep].astype(np.int64), t[keep], rank[B.row][keep]
    o = np.lexsort((r, t, f))
    f, t, r = f[o], t[o], r[o]
    new = np.ones(len(f), dtype=bool)
    new[1:] = (f[1:] != f[:-1]) | (t[1:] != t[:-1])
    bid = np.cumsum(new) - 1
    first = np.flatnonzero(new)
    cnt = np.diff(np.append(first, len(f)))
    bf, bt, rmin = f[first], t[first], r[first]
    pos_in = np.arange(len(f)) - first[bid]
    gpos = np.flatnonzero(r != pos_in)                        # mex of the block's ranks = first position with a gap
    mex = cnt.copy()
    np.minimum.at(mex, bid[gpos], pos_in[gpos])
    some = cnt < B.sizes[bt]
    base = (grank[bt] == 0).astype(np.int64)
    gp = ((rmin == grank[bt]).astype(np.int64) - base) * some
    gn = ((mex == grank[bt]).astype(np.int64) - base) * some
    return (np.bincount(bf, weights=gp, minlength=n_feat).astype(np.int64),
            np.bincount(bf, weights=gn, minlength=n_feat).astype(np.int64),
            np.bincount(bf, weights=some, minlength=n_feat).astype(np.int64))


# ------------------------------------------------------------------------------------------------- per group
def folds_of(trajs: list[dict], k: int) -> np.ndarray:
    return np.array([int(hashlib.md5(f"ngram-fold|{t['task']}".encode()).hexdigest()[:8], 16) % k for t in trajs])


def run_group(train: list[dict], evals: dict, H: Hasher, args) -> dict:
    n = len(train)
    fold = folds_of(train, args.folds)
    df = np.zeros(1 << NB, dtype=np.int32)
    for t in train:
        df[np.unique(np.concatenate(t["buckets"]))] += 1
    kept = np.flatnonzero(df >= args.min_df)
    fmap = np.full(1 << NB, -1, dtype=np.int64)
    fmap[kept] = np.arange(len(kept))
    F = len(kept)
    B = Block(train, fmap)
    P = Block(train, None)
    ag = [a for a, _ in collections.Counter(a for t in train for a in t["agents"]).most_common(args.max_agents)]
    Q = Block(train, None, ag)
    res: dict = {"n": n, "features": F, "nnz": int(len(B.idx)), "agents": ag}
    # cross-fitted probes
    acc = {"position": 0, "structure": 0}
    acc.update({f"ngram_lam{lam:g}": 0 for lam in args.lams})
    p_pos, p_str = np.zeros(B.S), np.zeros(B.S)
    for k in range(args.folds):
        te, tr = np.flatnonzero(fold == k), np.flatnonzero(fold != k)
        if not len(te) or not len(tr):
            continue
        for lev, X, pv in (("position", P, p_pos), ("structure", Q, p_str)):
            wk = fit(X.subset(tr), 0, 0.0)
            Xt = X.subset(te)
            acc[lev] += int(hits(Xt, wk).sum())
            pk = probs(Xt, wk)
            for j, t in enumerate(te):
                pv[B.start[t]:B.start[t] + B.sizes[t]] = pk[Xt.start[j]:Xt.start[j] + Xt.sizes[j]]
        Btr, Bte = B.subset(tr), B.subset(te)
        for lam in args.lams:
            acc[f"ngram_lam{lam:g}"] += int(hits(Bte, fit(Btr, F, lam)).sum())
    cv = {k: round(v / n, 4) for k, v in acc.items()}
    best_lam = max(args.lams, key=lambda lam: (cv[f"ngram_lam{lam:g}"], -lam))
    cv["ngram"] = cv[f"ngram_lam{best_lam:g}"]
    cv["hits_position"], cv["hits_ngram"] = acc["position"], acc[f"ngram_lam{best_lam:g}"]
    cv["hits_structure"] = acc["structure"]
    cv["lam"] = best_lam
    cv["gain"] = round(cv["ngram"] - cv["position"], 4)
    res["cv"] = cv
    # single phrases (cross-fitted prior)
    rank, grank = rank_layout(B, p_pos)
    gp, gn, cov = phrase_gains(B, rank, grank, F)
    best = np.maximum(gp, gn)
    top = np.argsort(-best, kind="stable")[:args.top]
    sel_total, picks = 0, []
    for k in range(args.folds):
        te = fold == k
        if not te.any() or te.all():
            continue
        a, b, _ = phrase_gains(B, rank, grank, F, ~te)
        j = int(np.argmax(np.maximum(a, b)))
        d = "+" if a[j] >= b[j] else "-"
        ha, hb, _ = phrase_gains(B, rank, grank, F, te)
        sel_total += int(ha[j] if d == "+" else hb[j])
        picks.append((int(kept[j]), d))
    names = H.grams_of_bucket([s for t in train for s in t["steps"]], {int(kept[j]) for j in top} | {b for b, _ in picks})

    def label(b):
        c = names.get(b)
        return c.most_common(1)[0][0] if c else f"#bucket{b}"
    # the same selection-honest single-phrase gain on top of the structure prior (speaker / agent identity)
    rank_s, grank_s = rank_layout(B, p_str)
    sel_s, picks_s = 0, []
    for k in range(args.folds):
        te = fold == k
        if not te.any() or te.all():
            continue
        a, b, _ = phrase_gains(B, rank_s, grank_s, F, ~te)
        j = int(np.argmax(np.maximum(a, b)))
        d = "+" if a[j] >= b[j] else "-"
        ha, hb, _ = phrase_gains(B, rank_s, grank_s, F, te)
        sel_s += int(ha[j] if d == "+" else hb[j])
        picks_s.append((int(kept[j]), d))
    names.update(H.grams_of_bucket([s for t in train for s in t["steps"]], {b for b, _ in picks_s} - set(names)))
    res["phrases_over_structure"] = {"selected_heldout_hits": sel_s, "selected_heldout_gain": round(sel_s / n, 4),
                                     "selected": [f"{label(b)} ({d})" for b, d in picks_s]}
    res["phrases"] = {
        "selected_heldout_hits": sel_total,
        "top": [{"phrase": label(int(kept[j])), "dir": "+" if gp[j] >= gn[j] else "-", "coverage": int(cov[j]),
                 "gain": round(int(best[j]) / n, 4)} for j in top if best[j] > 0],
        "selected_heldout_gain": round(sel_total / n, 4),
        "selected": [f"{label(b)} ({d})" for b, d in picks]}
    # fit on all of train, applied to the eval files
    wp = fit(P, 0, 0.0)
    wq = fit(Q, 0, 0.0)
    wn = fit(B, F, best_lam)
    res["eval"] = {}
    for name, ts in evals.items():
        if not ts:
            continue
        E, Ep, Eq = Block(ts, fmap), Block(ts, None), Block(ts, None, ag)
        e = {"n": len(ts), "position": round(float(hits(Ep, wp).mean()), 4),
             "structure": round(float(hits(Eq, wq).mean()), 4), "ngram": round(float(hits(E, wn).mean()), 4)}
        e["gain"] = round(e["ngram"] - e["position"], 4)
        r2, g2 = rank_layout(E, probs(Ep, wp))
        a, b, c = phrase_gains(E, r2, g2, F)
        e["top_phrases"] = [{"phrase": label(int(kept[j])), "dir": "+" if gp[j] >= gn[j] else "-",
                             "coverage": int(c[j]), "gain": round(int(a[j] if gp[j] >= gn[j] else b[j]) / len(ts), 4)}
                            for j in top[:5] if best[j] > 0]
        e["hits_position"], e["hits_ngram"] = int(hits(Ep, wp).sum()), int(hits(E, wn).sum())
        e["hits_structure"] = int(hits(Eq, wq).sum())
        res["eval"][name] = e
    return res


def analyse(args) -> dict:
    H = Hasher()

    def load(path):
        return read_file(Path(path) if Path(path).is_absolute() else ROOT / path)
    train = [t for p in args.train for t in load(p)]
    evals = {}
    for spec in args.eval:
        name, _, path = spec.partition("=")
        evals[name] = load(path)
    by = collections.defaultdict(list)
    for t in train:
        by[t["group"]].append(t)
    out: dict = {"args": vars(args), "groups": {}, "skipped": {}, "uncovered": {}}
    for g in sorted(by):
        if len(by[g]) < args.min_group:
            out["skipped"][g] = len(by[g])
            continue
        ev = {name: [t for t in ts if t["group"] == g] for name, ts in evals.items()}
        for t in by[g] + [t for ts in ev.values() for t in ts]:
            t["buckets"] = [H.buckets(s) for s in t["steps"]]
        out["groups"][g] = run_group(by[g], ev, H, args)
        for t in by[g] + [t for ts in ev.values() for t in ts]:
            del t["buckets"]
        r = out["groups"][g]
        print(g, r["n"], "cv", r["cv"], "sel", r["phrases"]["selected_heldout_gain"],
              "top", [(x["phrase"], x["dir"], x["gain"]) for x in r["phrases"]["top"][:3]], flush=True)
    covered = set(out["groups"])
    for name, ts in evals.items():
        unc = collections.Counter(t["group"] for t in ts if t["group"] not in covered)
        if unc:
            out["uncovered"][name] = dict(unc)
    # pooled over groups
    tot = {"train_cv": collections.Counter()}
    for g, r in out["groups"].items():
        tot["train_cv"]["n"] += r["n"]
        tot["train_cv"]["position"] += r["cv"]["hits_position"]
        tot["train_cv"]["structure"] += r["cv"]["hits_structure"]
        tot["train_cv"]["ngram"] += r["cv"]["hits_ngram"]
        tot["train_cv"]["phrase_sel"] += r["phrases"]["selected_heldout_hits"]
        tot["train_cv"]["phrase_sel_struct"] += r["phrases_over_structure"]["selected_heldout_hits"]
        for name, e in r["eval"].items():
            c = tot.setdefault(name, collections.Counter())
            c["n"] += e["n"]
            c["position"] += e["hits_position"]
            c["structure"] += e["hits_structure"]
            c["ngram"] += e["hits_ngram"]
    out["all"] = {k: {"n": c["n"], "position": round(c["position"] / max(1, c["n"]), 4),
                      "structure": round(c["structure"] / max(1, c["n"]), 4),
                      "ngram": round(c["ngram"] / max(1, c["n"]), 4),
                      "gain": round((c["ngram"] - c["position"]) / max(1, c["n"]), 4)}
                  | ({"phrase_selected_heldout_gain": round(c["phrase_sel"] / max(1, c["n"]), 4),
                      "phrase_selected_heldout_gain_over_structure": round(c["phrase_sel_struct"] / max(1, c["n"]), 4)}
                     if k == "train_cv" else {}) for k, c in tot.items()}
    return out


def report(res: dict) -> str:
    def pc(x):
        return "–" if x is None else f"{100 * x:.1f}"
    names = sorted({n for r in res["groups"].values() for n in r["eval"]})
    L = ["| group | n | cv position | cv structure | cv ngram | gain | best phrase, held-out (over structure) | "
         "top phrase (in-sample gain) | " + " | ".join(f"{n}: n, position / ngram" for n in names) + " |",
         "|" + "---|" * (8 + len(names))]
    for g, r in res["groups"].items():
        top = r["phrases"]["top"][0] if r["phrases"]["top"] else None
        cells = []
        for n in names:
            e = r["eval"].get(n)
            cells.append(f"{e['n']}, {pc(e['position'])} / {pc(e['ngram'])}" if e else "")
        L.append(f"| {g} | {r['n']} | {pc(r['cv']['position'])} | {pc(r['cv']['structure'])} | "
                 f"{pc(r['cv']['ngram'])} | {pc(r['cv']['gain'])} | {pc(r['phrases']['selected_heldout_gain'])} "
                 f"({pc(r['phrases_over_structure']['selected_heldout_gain'])}) | "
                 + (f"{top['phrase']} ({top['dir']}, {top['coverage']}): {pc(top['gain'])}" if top else "–") + " | "
                 + " | ".join(cells) + " |")
    a = res["all"]
    L.append("")
    L.append("all: " + "; ".join(f"{k} n {v['n']} position {pc(v['position'])} structure {pc(v['structure'])} ngram "
                                 f"{pc(v['ngram'])} gain {pc(v['gain'])}" for k, v in a.items()))
    t = a.get("train_cv", {})
    if t:
        L.append(f"selected single phrase, held-out gain over position {pc(t['phrase_selected_heldout_gain'])}, over "
                 f"structure {pc(t['phrase_selected_heldout_gain_over_structure'])} (train CV, pooled)")
    if res["uncovered"]:
        L.append("uncovered (no train group): " + json.dumps(res["uncovered"]))
    return "\n".join(L)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True, help="step-level train file(s)")
    ap.add_argument("--eval", nargs="*", default=[], help="name=path step-level eval files")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-df", type=int, default=20, help="n-grams in fewer train trajectories of a group are ignored")
    ap.add_argument("--min-group", type=int, default=40)
    ap.add_argument("--lams", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1])
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--max-agents", type=int, default=30, help="agent-name one-hots of the structure level")
    ap.add_argument("--out", default=None, help="json path (a .md report is written next to it)")
    args = ap.parse_args(argv)
    res = analyse(args)
    md = report(res)
    print(md)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        p.with_suffix(".md").write_text(md + "\n")
    return res


if __name__ == "__main__":
    main()
