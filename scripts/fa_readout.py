"""Failure-attribution readout (notes/next_round/failure_attribution/tier1_public.md): any set of FA run directories
(results/hpc/<x>/<run>/{metrics.json, preds_<eval>.jsonl}), scored per eval file against trivial baselines computed
from the eval file itself. CPU only, numpy only, no model.

Per FA eval file (an eval whose items have source fa_public / who_and_when / fa_tier0; the eval name -> file map comes
from metrics.json args.eval, else DEFAULT_EVAL; "val" without metrics.json is resolved by its item ids):
  runs        accuracy with an item-bootstrap 95% CI, Acc@3, MRR, NLL of the gold option; ranks are the stable
              descending order of probs (ties go to the earlier option, so rank 1 == the trainer's argmax, which takes
              the first maximum); accuracy is checked against metrics.json eval.<name>.acc when present.
  breakdown   per group key: fa.origin/fa.subset (Who&When Pro framework, AgenTracer / AFTraj domain), with
              fa.benchmark appended for aftraj and agentracer (injected / diagnosed provenance); else the item group.
              Keys that a run's training data dropped (debias.dropped in the summary.json next to its --train file,
              or --dropped) are flagged.
  rel. error  step items: (predicted - gold step) / (n - 1), binned, with the early / exact / late shares.
  baselines   step files: chance (mean 1/n, expected), always first step, always step 1, always last step, best
              constant index (fit on the file: oracle), best constant relative position round(r (n-1)) on a 0.05 grid
              (oracle), and two conditional-logit probes over a trajectory's steps fit on the train split (--train-step,
              default data/student_fa1/train.jsonl, --probe-n items): "position probe" (relative position, its square,
              first / last / second-to-last / step 1, index one-hots 0-9) and "surface probe" (position + log rendered
              length, length rank, longest / shortest step, speaker change; the features of build_fa_public.py
              step_features, with lengths of the rendered, already cut step text parsed from the prompt).
              Agent files: chance, always option A (first speaker), most frequent gold agent of --train-agent among the
              item's options, agent of the last step (option A when that is not an option), best constant option
              index (oracle).
  beats       arm accuracy minus the best trivial baseline of the file (the max over the baselines above except the
              probes and chance; picked on the file itself, so the comparison is conservative) and minus the surface
              probe, each with a paired item-bootstrap 95% CI; "yes" when the CI lies above 0.
Paired arm differences (--contrast, default GFPS-A2): runs paired by seed label (when an arm has a single run and the
  other a single run with another seed, they are paired and marked "unpaired seeds"); per item the mean correctness
  over the paired seeds, then an item bootstrap of the mean difference over items present in both.

Run name: <ARM>-<tag>-<model>-<data>-s<seed> (hpc/train_arm.sbatch), e.g. A2-fa1x6-Qwen3.5-0.8B-Base-student_fa1-s12.
Runs still training (no metrics.json or no final eval) are read from whatever preds files exist.

  python scripts/fa_readout.py 'results/hpc/fa0/*' --name fa0_tier0
  python scripts/fa_readout.py 'results/hpc/fa1/*' --name fa1 --contrast GFPS-A2
  -> results/fa_readout/<name>.md, <name>.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FA_SOURCES = {"fa_public", "who_and_when", "fa_tier0"}
DEFAULT_EVAL = {
    "ww_step": "data/eval/ww_alg_step.jsonl", "ww_stepgt": "data/eval/ww_alg_step_gt.jsonl",
    "ww_agent": "data/eval/ww_alg_agent.jsonl", "ww_hc": "data/eval/ww_hc_agent.jsonl",
    "fa0": "data/eval/fa0_test.jsonl", "fa0gt": "data/eval/fa0_test_gt.jsonl",
    "fa0ag": "data/eval/fa0_test_agent.jsonl",
    "wwpro": "data/eval/fa1_wwpro_test.jsonl", "wwprogt": "data/eval/fa1_wwpro_test_gt.jsonl",
    "wwproag": "data/eval/fa1_wwpro_test_agent.jsonl",
    "aftraj": "data/eval/fa1_aftraj_test.jsonl", "aftrajgt": "data/eval/fa1_aftraj_test_gt.jsonl",
    "aftrajag": "data/eval/fa1_aftraj_test_agent.jsonl",
    "agtr": "data/eval/fa1_agentracer_test.jsonl", "agtrgt": "data/eval/fa1_agentracer_test_gt.jsonl",
    "agtrag": "data/eval/fa1_agentracer_test_agent.jsonl",
    "te": "data/eval/fa1_traceelephant_test.jsonl", "tegt": "data/eval/fa1_traceelephant_test_gt.jsonl",
    "teag": "data/eval/fa1_traceelephant_test_agent.jsonl",
    "fa1v": "data/student_fa1/val.jsonl", "fa1va": "data/student_fa1/val_agent.jsonl",
}
VAL_BY_PREFIX = {"fa0-val": "data/student_fa0/val.jsonl", "fa1-val": "data/student_fa1/val.jsonl",
                 "fa1b-val": "data/student_fa1b/val.jsonl"}
RUN_RE = re.compile(r"^(?P<arm>[^-]+)-(?P<tag>[^-]+)-(?P<model>.+)-(?P<data>student_[^-]+)-s(?P<seed>\d+)$")
REL_BINS = [(-1.01, -0.5, "< -.5"), (-0.5, -0.2, "-.5..-.2"), (-0.2, -1e-9, "-.2..0"), (-1e-9, 1e-9, "0"),
            (1e-9, 0.2, "0..+.2"), (0.2, 0.5, "+.2..+.5"), (0.5, 1.01, "> +.5")]
TRIVIAL_STEP = ["first_step", "step_1", "last_step", "best_const_index", "best_const_rel"]
TRIVIAL_AGENT = ["option_A", "most_frequent_train_agent", "last_step_agent", "best_const_option"]


# ------------------------------------------------------------------------------------------------- items
def options(item: dict) -> list[str]:
    """Option texts in label order: label_names when present (a dict, or its repr in older files), else the
    'X) text' lines after the last '\\nOptions:\\n' of the prompt."""
    names = item.get("label_names")
    if isinstance(names, str):
        import ast
        names = ast.literal_eval(names)
    if names:
        return [names[L] for L in item["labels"]]
    tail = item["prompt"].rsplit("\nOptions:\n", 1)[1].split("\n")
    out = []
    for L, line in zip(item["labels"], tail):
        if not line.startswith(L + ") "):
            raise ValueError(f"{item['item_id']}: option line {line[:40]!r}")
        out.append(line[len(L) + 2:])
    return out


STEP_OPT = re.compile(r"^Step (\d+) \[(.*)\]$")
STEP_HEAD = re.compile(r"\nStep (\d+) \[([^\n]*?)\]: ")


def parse_steps(prompt: str, refs: list[str] | None = None) -> list[tuple[str, int]]:
    """(agent, rendered length) of every 'Step i [Agent]: content' block of a rendered log, in order. With refs (the
    step options 'Step i [Agent]') each header is searched exactly and in order; without, step i is the first
    '\\nStep i [..]: ' header after step i-1. Content runs to the next header or to the final '\\n\\nQuestion: '."""
    end = prompt.rfind("\n\nQuestion: ")
    log = prompt[:end] if end >= 0 else prompt
    heads: list[tuple[int, int, str]] = []          # (header start, content start, agent)
    pos = 0
    i = 0
    while True:
        if refs is not None:
            if i >= len(refs):
                break
            h = "\n" + refs[i] + ": "
            m = log.find(h, pos)
            if m < 0:
                raise ValueError(f"step header {refs[i]!r} not found")
            agent = STEP_OPT.match(refs[i]).group(2)
            heads.append((m, m + len(h), agent))
            pos = m + len(h)
        else:
            mm = None
            for mm in STEP_HEAD.finditer(log, pos):
                if int(mm.group(1)) == i:
                    break
            else:
                mm = None
            if mm is None:
                break
            heads.append((mm.start(), mm.end(), mm.group(2)))
            pos = mm.end()
        i += 1
    out = []
    for k, (s, c, a) in enumerate(heads):
        stop = heads[k + 1][0] if k + 1 < len(heads) else len(log)
        out.append((a, stop - c))
    return out


def item_info(item: dict) -> dict:
    """What the readout keeps of an eval / train item (the prompt is dropped)."""
    opts = options(item)
    labels = item["labels"]
    fa = item.get("fa") or {}
    if isinstance(fa, str):
        import ast
        fa = ast.literal_eval(fa)
    is_step = all(STEP_OPT.match(o) for o in opts)
    info = {"item_id": item["item_id"], "group": item.get("group"), "labels": labels,
            "gold": labels.index(item["gold_label"]) if item.get("gold_label") in labels else None,
            "kind": "step" if is_step else "agent", "options": None if is_step else opts}
    steps = parse_steps(item["prompt"], opts if is_step else None)
    info["agents"] = [a for a, _ in steps]
    info["lens"] = [n for _, n in steps]
    if is_step and len(steps) != len(opts):
        raise ValueError(f"{item['item_id']}: {len(steps)} steps for {len(opts)} options")
    if fa.get("origin"):
        key = f"{fa['origin']}/{fa.get('subset')}"
        if fa["origin"] in ("aftraj", "agentracer"):
            key += f"/{fa.get('benchmark')}"
    else:
        key = item.get("group") or "all"
    info["key"] = key
    return info


ITEM_ID = re.compile(r'^\{"item_id": "([^"\\]+)"')


def line_id(line: str) -> str:
    m = ITEM_ID.match(line)
    return m.group(1) if m else json.loads(line)["item_id"]


def read_items(path: Path, limit: int | None = None, pick: set | None = None) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip() or (pick is not None and line_id(line) not in pick):
                continue
            out.append(item_info(json.loads(line)))
            if limit and len(out) >= limit:
                break
    return out


def is_fa_file(path: Path) -> bool:
    with open(path) as f:
        line = f.readline()
    return bool(line.strip()) and json.loads(line).get("source") in FA_SOURCES


# ------------------------------------------------------------------------------------------------- probes
def features(agents: list[str], lens: list[int], level: str) -> np.ndarray:
    """build_fa_public.step_features on parsed (agent, rendered length) steps; level position or surface."""
    n = len(lens)
    rank = sorted(range(n), key=lambda i: lens[i])
    rk = {i: k / max(1, n - 1) for k, i in enumerate(rank)}
    rows = []
    for i in range(n):
        rel = i / max(1, n - 1)
        f = [rel, rel * rel, i == 0, i == n - 1, i == n - 2, i == 1] + [min(i, 9) == k for k in range(10)]
        if level == "surface":
            f += [math.log1p(lens[i]) / 8, rk[i], lens[i] == max(lens), lens[i] == min(lens),
                  agents[i] != (agents[i - 1] if i else None)]
        rows.append(f)
    return np.array(rows, dtype=float)


def tensor(infos: list[dict], level: str):
    m = max(len(x["lens"]) for x in infos)
    f = features(["a", "b"], [1, 2], level).shape[1]
    X = np.zeros((len(infos), m, f))
    mask = np.zeros((len(infos), m), dtype=bool)
    for j, x in enumerate(infos):
        n = len(x["lens"])
        X[j, :n] = features(x["agents"], x["lens"], level)
        mask[j, :n] = True
    return X, mask


def clogit_probs(X: np.ndarray, mask: np.ndarray, w: np.ndarray) -> np.ndarray:
    z = np.where(mask, X @ w, -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def fit_clogit(X: np.ndarray, mask: np.ndarray, y: np.ndarray, steps: int = 300, lr: float = 0.5,
               lam: float = 1e-3) -> np.ndarray:
    """Conditional logit over the steps of each trajectory (build_fa_public.fit_clogit, vectorized over padding)."""
    w = np.zeros(X.shape[2])
    rows = np.arange(len(y))
    for _ in range(steps):
        p = clogit_probs(X, mask, w)
        g = np.einsum("nm,nmf->f", p, X) - X[rows, y].sum(0)
        w -= lr * (g / len(y) + lam * w)
    return w


def stable(*parts) -> int:
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


class TrainPriors:
    """Probes fit on the step-level train split and gold-agent counts of the agent-level train split."""

    def __init__(self, train_step: Path | None, train_agent: Path | None, probe_n: int):
        self.weights: dict = {}
        self.n_fit = 0
        self.agent_counts: Counter = Counter()
        self.sources = {"train_step": str(train_step) if train_step else None,
                        "train_agent": str(train_agent) if train_agent else None}
        if train_step and Path(train_step).exists():
            with open(train_step) as f:
                ids = [line_id(line) for line in f if line.strip()]
            keep = set(sorted(ids, key=lambda i: stable("probe", i))[:probe_n])
            infos = [x for x in read_items(Path(train_step), pick=keep)
                     if x["kind"] == "step" and x["gold"] is not None]
            self.n_fit = len(infos)
            y = np.array([x["gold"] for x in infos])
            for level in ("position", "surface"):
                X, mask = tensor(infos, level)
                self.weights[level] = fit_clogit(X, mask, y)
                del X, mask
        if train_agent and Path(train_agent).exists():
            with open(train_agent) as f:
                for line in f:
                    it = json.loads(line)
                    names = it.get("label_names") or {}
                    if it.get("gold_label") in names:
                        self.agent_counts[names[it["gold_label"]]] += 1

    def probe_preds(self, infos: list[dict], level: str) -> np.ndarray | None:
        if level not in self.weights:
            return None
        X, mask = tensor(infos, level)
        return clogit_probs(X, mask, self.weights[level]).argmax(1)


# ------------------------------------------------------------------------------------------------- baselines
def baselines(infos: list[dict], pri: TrainPriors) -> dict:
    """Per-item correctness (0/1 arrays in infos order) of every trivial baseline; 'chance' is an expected value."""
    gold = np.array([x["gold"] for x in infos])
    if infos[0]["kind"] == "step":
        n = np.array([len(x["labels"]) for x in infos])
        out = {"chance": float(np.mean(1 / n)),
               "first_step": gold == 0, "step_1": gold == np.minimum(1, n - 1), "last_step": gold == n - 1}
        best = max(range(62), key=lambda c: (np.sum(gold == np.minimum(c, n - 1)), -c))
        out["best_const_index"] = gold == np.minimum(best, n - 1)
        rs = [round(0.05 * k, 2) for k in range(21)]
        rel = lambda r: np.floor(r * (n - 1) + 0.5).astype(int)          # noqa: E731  (round half up)
        r_best = max(rs, key=lambda r: (np.sum(gold == rel(r)), -r))
        out["best_const_rel"] = gold == rel(r_best)
        meta = {"best_const_index": best, "best_const_rel": r_best}
        for level, name in (("position", "position_probe"), ("surface", "surface_probe")):
            p = pri.probe_preds(infos, level)
            if p is not None:
                out[name] = gold == p
        trivial = TRIVIAL_STEP
    else:
        k = np.array([len(x["labels"]) for x in infos])
        out = {"chance": float(np.mean(1 / k)), "option_A": gold == 0}
        top = [a for a, _ in pri.agent_counts.most_common()]
        if top:
            pick = []
            for x in infos:
                opts = x["options"]
                pick.append(next((opts.index(a) for a in top if a in opts), 0))
            out["most_frequent_train_agent"] = gold == np.array(pick)
        last = np.array([x["options"].index(x["agents"][-1]) if x["agents"] and x["agents"][-1] in x["options"] else 0
                         for x in infos])
        out["last_step_agent"] = gold == last
        best = max(range(62), key=lambda c: (np.sum(gold == np.minimum(c, k - 1)), -c))
        out["best_const_option"] = gold == np.minimum(best, k - 1)
        meta = {"best_const_option": best}
        trivial = TRIVIAL_AGENT
    vec = {k: v.astype(float) for k, v in out.items() if not isinstance(v, float)}
    present = [t for t in trivial if t in vec]
    best_t = max(present, key=lambda t: (vec[t].mean(), -present.index(t)))
    return {"vectors": vec, "chance": out["chance"], "meta": meta, "best_trivial": best_t}


# ------------------------------------------------------------------------------------------------- statistics
def boot_ci(v: np.ndarray, draws: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(v) == 0:
        return (float("nan"), float("nan"))
    means = np.empty(draws)
    for s in range(0, draws, 500):
        b = min(500, draws - s)
        means[s:s + b] = v[rng.integers(0, len(v), (b, len(v)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rng_for(*key) -> np.random.Generator:
    return np.random.default_rng(stable(*key))


def ranks(probs: list[float], gold: int) -> int:
    """1-based rank of gold in the stable descending order of probs (ties: earlier option first)."""
    pg = probs[gold]
    return 1 + sum(1 for i, p in enumerate(probs) if p > pg or (p == pg and i < gold))


def rel_hist(pred: np.ndarray, gold: np.ndarray, n: np.ndarray) -> dict:
    e = (pred - gold) / np.maximum(1, n - 1)
    out = {lab: int(np.sum((e > lo) & (e <= hi))) for lo, hi, lab in REL_BINS}
    out["n"] = int(len(e))
    out["early"] = float(np.mean(e < 0)) if len(e) else float("nan")
    out["exact"] = float(np.mean(e == 0)) if len(e) else float("nan")
    out["late"] = float(np.mean(e > 0)) if len(e) else float("nan")
    out["mean_signed"] = float(np.mean(e)) if len(e) else float("nan")
    out["mean_abs_steps"] = float(np.mean(np.abs(pred - gold))) if len(e) else float("nan")
    return out


# ------------------------------------------------------------------------------------------------- runs
def parse_run(name: str) -> dict:
    m = RUN_RE.match(name)
    if m:
        return {**m.groupdict(), "seed": int(m.group("seed"))}
    s = re.search(r"-s(\d+)$", name)
    return {"arm": name.split("-")[0], "tag": "", "model": "", "data": "", "seed": int(s.group(1)) if s else None}


def eval_map(run: Path, metrics: dict | None) -> dict[str, str]:
    m = dict(DEFAULT_EVAL)
    for spec in ((metrics or {}).get("args") or {}).get("eval") or []:
        name, _, path = spec.partition("=")
        m[name] = path
    return m


def resolve(name: str, emap: dict, preds_first_id: str) -> Path | None:
    path = emap.get(name)
    if path is None and name == "val":
        for pre, p in VAL_BY_PREFIX.items():
            if preds_first_id.startswith(pre + "-"):
                path = p
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def dropped_for(metrics: dict | None, extra: list[str]) -> list[str]:
    out = list(extra)
    train = ((metrics or {}).get("args") or {}).get("train")
    if train:
        for t in str(train).split(","):
            s = ROOT / Path(t).parent / "summary.json"
            if s.exists():
                try:
                    out += json.loads(s.read_text()).get("debias", {}).get("dropped", [])
                except (ValueError, OSError):
                    pass
    return sorted(set(out))


def load_runs(patterns: list[str]) -> list[dict]:
    runs = []
    for d in sorted({p for pat in patterns for p in glob.glob(pat)}):
        d = Path(d)
        if not d.is_dir() or not list(d.glob("preds_*.jsonl")):
            continue
        mp = d / "metrics.json"
        metrics = None
        if mp.exists():
            try:
                metrics = json.loads(mp.read_text())
            except ValueError:
                metrics = None
        runs.append({"dir": str(d), "name": d.name, **parse_run(d.name), "metrics": metrics})
    return runs


def read_preds(path: Path) -> dict[str, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                out[p["item_id"]] = p
    return out


def score_run(preds: dict, infos: list[dict], draws: int, key: tuple) -> dict:
    """Correctness vector (infos order, NaN where the run has no prediction) and the summary metrics."""
    corr = np.full(len(infos), np.nan)
    rk = np.full(len(infos), np.nan)
    pred = np.full(len(infos), -1)
    nll = []
    for j, x in enumerate(infos):
        p = preds.get(x["item_id"])
        if p is None or x["gold"] is None:
            continue
        if p["labels"] != x["labels"] or p.get("gold_label") != x["labels"][x["gold"]]:
            raise ValueError(f"{x['item_id']}: preds and eval item disagree on labels / gold")
        pr = p["probs"]
        r = ranks(pr, x["gold"])
        rk[j] = r
        corr[j] = float(r == 1)
        pred[j] = max(range(len(pr)), key=pr.__getitem__)
        nll.append(-math.log(max(pr[x["gold"]], 1e-12)))
    ok = ~np.isnan(corr)
    c = corr[ok]
    res = {"n": int(ok.sum()), "acc": float(c.mean()) if len(c) else float("nan"),
           "ci": boot_ci(c, draws, rng_for("acc", *key)), "acc3": float(np.mean(rk[ok] <= 3)) if len(c) else None,
           "mrr": float(np.mean(1 / rk[ok])) if len(c) else None, "nll": float(np.mean(nll)) if nll else None}
    return {"summary": res, "corr": corr, "pred": pred}


def paired(a: np.ndarray, b: np.ndarray, draws: int, key: tuple) -> dict:
    ok = ~np.isnan(a) & ~np.isnan(b)
    d = a[ok] - b[ok]
    lo, hi = boot_ci(d, draws, rng_for("paired", *key))
    return {"n": int(ok.sum()), "diff": float(d.mean()) if len(d) else float("nan"), "ci": (lo, hi),
            "beats": bool(lo > 0), "worse": bool(hi < 0)}


# ------------------------------------------------------------------------------------------------- main analysis
def analyse(args) -> dict:
    runs = load_runs(args.runs)
    if not runs:
        raise SystemExit(f"no run directories with preds_*.jsonl under {args.runs}")
    pri = TrainPriors(Path(args.train_step) if args.train_step else None,
                      Path(args.train_agent) if args.train_agent else None, args.probe_n)
    files: dict[str, dict] = {}                     # eval file path -> {infos, names, runs}
    other: dict[str, dict] = defaultdict(dict)      # non-FA evals: run -> name -> acc (metrics.json)
    checks = []
    for r in runs:
        emap = eval_map(Path(r["dir"]), r["metrics"])
        r["dropped"] = dropped_for(r["metrics"], args.dropped.split(",") if args.dropped else [])
        final = (r["metrics"] or {}).get("eval") or {}
        r["complete"] = bool(final)
        for pf in sorted(Path(r["dir"]).glob("preds_*.jsonl")):
            name = pf.stem[len("preds_"):]
            if name.startswith("subq"):
                continue
            preds = read_preds(pf)
            if not preds:
                continue
            path = resolve(name, emap, next(iter(preds)))
            if path is None or not path.exists() or not is_fa_file(path):
                if name in final:
                    other[r["name"]][name] = final[name].get("acc")
                continue
            ent = files.setdefault(str(path), {"names": set(), "runs": {}})
            ent["names"].add(name)
            ent["runs"][r["name"]] = (preds, final.get(name))
    out_files = {}
    for path, ent in sorted(files.items(), key=lambda kv: sorted(kv[1]["names"])[0]):
        ids = set().union(*(set(p) for p, _ in ent["runs"].values()))
        infos = read_items(Path(path), pick=ids)
        infos = [x for x in infos if x["gold"] is not None]
        if not infos:
            continue
        rel = str(Path(path).relative_to(ROOT)) if Path(path).is_relative_to(ROOT) else path
        base = baselines(infos, pri)
        bres = {k: {"acc": float(v.mean()), "ci": boot_ci(v, args.boot, rng_for("base", rel, k))}
                for k, v in base["vectors"].items()}
        keys = sorted({x["key"] for x in infos})
        kidx = {k: np.array([x["key"] == k for x in infos]) for k in keys}
        gold = np.array([x["gold"] for x in infos])
        nopt = np.array([len(x["labels"]) for x in infos])
        fres = {"names": sorted(ent["names"]), "n": len(infos), "kind": infos[0]["kind"], "baselines": bres,
                "chance": base["chance"], "baseline_meta": base["meta"], "best_trivial": base["best_trivial"],
                "breakdown_keys": keys, "breakdown": {}, "runs": {}}
        for k in keys:
            m = kidx[k]
            fres["breakdown"][k] = {"n": int(m.sum()), "baselines": {b: float(v[m].mean())
                                                                      for b, v in base["vectors"].items()},
                                    "runs": {}}
        vecs = {}
        for rname, (preds, final) in sorted(ent["runs"].items()):
            sc = score_run(preds, infos, args.boot, (rel, rname))
            s = sc["summary"]
            vecs[rname] = sc["corr"]
            if final and final.get("n") == s["n"]:
                same = abs(final["acc"] - s["acc"]) < 1e-9
                checks.append({"run": rname, "file": rel, "metrics_acc": final["acc"], "readout_acc": s["acc"],
                               "ok": same})
            bt = base["best_trivial"]
            s["vs_best_trivial"] = {"baseline": bt, **paired(sc["corr"], base["vectors"][bt], args.boot,
                                                             (rel, rname, "bt"))}
            if "surface_probe" in base["vectors"]:
                s["vs_surface_probe"] = paired(sc["corr"], base["vectors"]["surface_probe"], args.boot,
                                               (rel, rname, "sp"))
            if fres["kind"] == "step":
                ok = sc["pred"] >= 0
                s["rel_error"] = rel_hist(sc["pred"][ok], gold[ok], nopt[ok])
            run = next(x for x in runs if x["name"] == rname)
            for k in keys:
                m = kidx[k] & ~np.isnan(sc["corr"])
                fres["breakdown"][k]["runs"][rname] = {"n": int(m.sum()), "acc": float(sc["corr"][m].mean())
                                                       if m.any() else None, "dropped_from_train":
                                                       k in run["dropped"]}
            fres["runs"][rname] = s
        fres["contrasts"] = contrasts(runs, vecs, args, rel)
        out_files[rel] = fres
    return {"args": vars(args), "runs": [{k: v for k, v in r.items() if k != "metrics"} for r in runs],
            "train_priors": {**pri.sources, "probe_items": pri.n_fit, "agent_names": len(pri.agent_counts)},
            "files": out_files, "other_evals": dict(other), "acc_checks": checks}


def contrasts(runs: list[dict], vecs: dict, args, rel: str) -> list[dict]:
    out = []
    for spec in args.contrast:
        a1, _, a2 = spec.partition("-")
        r1 = [r for r in runs if r["arm"] == a1 and r["name"] in vecs]
        r2 = [r for r in runs if r["arm"] == a2 and r["name"] in vecs]
        if not r1 or not r2:
            continue
        s1, s2 = {r["seed"]: r for r in r1}, {r["seed"]: r for r in r2}
        seeds = sorted(set(s1) & set(s2), key=str)
        note = "paired by seed"
        if seeds:
            pairs = [(s1[s], s2[s]) for s in seeds]
        elif len(r1) == 1 and len(r2) == 1:
            pairs, note = [(r1[0], r2[0])], "unpaired seeds"
        else:
            continue
        v1 = np.nanmean(np.stack([vecs[a["name"]] for a, _ in pairs]), 0)
        v2 = np.nanmean(np.stack([vecs[b["name"]] for _, b in pairs]), 0)
        res = paired(v1, v2, args.boot, (rel, spec))
        out.append({"contrast": spec, "pairs": [(a["name"], b["name"]) for a, b in pairs], "note": note, **res})
    return out


# ------------------------------------------------------------------------------------------------- report
def pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"


def ci(c) -> str:
    return f"[{pct(c[0])}, {pct(c[1])}]"


def sdiff(d) -> str:
    return f"{100 * d['diff']:+.1f} [{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}]"


BASE_LABEL = {"first_step": "always first step", "step_1": "always step 1", "last_step": "always last step",
              "best_const_index": "best constant index (oracle)",
              "best_const_rel": "best constant rel. position (oracle)",
              "position_probe": "position probe (train)", "surface_probe": "surface probe (train)",
              "option_A": "always option A (first speaker)", "most_frequent_train_agent": "most frequent agent (train)",
              "last_step_agent": "agent of the last step", "best_const_option": "best constant option (oracle)"}


def report(res: dict) -> str:
    L = [f"# FA readout: {res['args']['name']}", ""]
    L.append(f"Runs: {', '.join(r['name'] + ('' if r['complete'] else ' (no final eval yet)') for r in res['runs'])}")
    tp = res["train_priors"]
    L.append(f"Probes fit on {tp['probe_items']} items of {tp['train_step']}; agent prior from {tp['train_agent']} "
             f"({tp['agent_names']} agent names). Bootstrap: {res['args']['boot']} item resamples, 95% percentile CI. "
             "Oracle baselines are fit on the file itself; 'beats' = paired CI above 0.")
    bad = [c for c in res["acc_checks"] if not c["ok"]]
    L.append(f"Accuracy check against metrics.json: {len(res['acc_checks']) - len(bad)}/{len(res['acc_checks'])} "
             "equal" + (f" (MISMATCH: {bad})" if bad else ""))
    L.append("")
    for path, f in res["files"].items():
        L.append(f"## {', '.join(f['names'])} ({path}, n = {f['n']}, {f['kind']} level)")
        L.append("")
        L.append("| | n | acc [95% CI] | Acc@3 | MRR | − best trivial [CI] | − surface probe [CI] |")
        L.append("|---|---|---|---|---|---|---|")
        L.append(f"| chance (expected) | {f['n']} | {pct(f['chance'])} | | | | |")
        for b, v in f["baselines"].items():
            lab = BASE_LABEL.get(b, b)
            meta = f["baseline_meta"].get(b)
            if meta is not None:
                lab += f" = {meta}"
            star = " (best trivial)" if b == f["best_trivial"] else ""
            L.append(f"| {lab}{star} | {f['n']} | {pct(v['acc'])} {ci(v['ci'])} | | | | |")
        for rn, s in f["runs"].items():
            bt = s["vs_best_trivial"]
            sp = s.get("vs_surface_probe")
            L.append(f"| **{rn}** | {s['n']} | {pct(s['acc'])} {ci(s['ci'])} | {pct(s['acc3'])} | "
                     f"{s['mrr']:.3f} | {sdiff(bt)}{' beats' if bt['beats'] else ''} | "
                     f"{sdiff(sp) + (' beats' if sp['beats'] else '') if sp else '–'} |")
        for c in f["contrasts"]:
            L.append("")
            L.append(f"Paired {c['contrast']} ({c['note']}; n = {c['n']}): {sdiff(c)}")
        if len(f["breakdown_keys"]) > 1:
            rn = list(f["runs"])
            bl = [b for b in f["baselines"] if b in ("best_const_index", "best_const_rel", "position_probe",
                                                     "surface_probe", "most_frequent_train_agent", "last_step",
                                                     "last_step_agent")]
            L.append("")
            L.append("| group | n | " + " | ".join(BASE_LABEL.get(b, b).split(" (")[0] for b in bl) + " | "
                     + " | ".join(rn) + " |")
            L.append("|---|---|" + "---|" * (len(bl) + len(rn)))
            for k, v in f["breakdown"].items():
                cells = [pct(v["baselines"][b]) for b in bl]
                cells += [pct(v["runs"][r]["acc"]) + (" †" if v["runs"][r]["dropped_from_train"] else "")
                          for r in rn]
                L.append(f"| {k} | {v['n']} | " + " | ".join(cells) + " |")
            if any(v["runs"][r]["dropped_from_train"] for v in f["breakdown"].values() for r in rn):
                L.append("")
                L.append("† group dropped from this run's training data (length-leak debias).")
        if f["kind"] == "step" and f["runs"]:
            L.append("")
            L.append("Relative position error (pred − gold) / (n − 1):")
            L.append("")
            L.append("| run | " + " | ".join(b[2] for b in REL_BINS) + " | early | exact | late | mean signed | "
                     "mean abs (steps) |")
            L.append("|---|" + "---|" * (len(REL_BINS) + 5))
            for rn, s in f["runs"].items():
                h = s["rel_error"]
                L.append(f"| {rn} | " + " | ".join(str(h[b[2]]) for b in REL_BINS) +
                         f" | {pct(h['early'])} | {pct(h['exact'])} | {pct(h['late'])} | {h['mean_signed']:+.3f} | "
                         f"{h['mean_abs_steps']:.2f} |")
        L.append("")
    if res["other_evals"]:
        names = sorted({n for v in res["other_evals"].values() for n in v})
        L.append("## Other evals (metrics.json, accuracy)")
        L.append("")
        L.append("| run | " + " | ".join(names) + " |")
        L.append("|---|" + "---|" * len(names))
        for rn, v in sorted(res["other_evals"].items()):
            L.append(f"| {rn} | " + " | ".join(pct(v.get(n)) for n in names) + " |")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run directory globs")
    ap.add_argument("--name", default="fa")
    ap.add_argument("--out-dir", default="results/fa_readout")
    ap.add_argument("--train-step", default="data/student_fa1/train.jsonl")
    ap.add_argument("--train-agent", default="data/student_fa1/train_agent.jsonl")
    ap.add_argument("--probe-n", type=int, default=3000, help="train items the probes are fit on (stable sample)")
    ap.add_argument("--contrast", nargs="*", default=["GFPS-A2"], help="ARM1-ARM2 differences (paired bootstrap)")
    ap.add_argument("--dropped", default="", help="extra group keys to flag as dropped from training (comma list)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--quiet", action="store_true", help="do not print the markdown report")
    args = ap.parse_args(argv)
    for k in ("train_step", "train_agent"):
        v = getattr(args, k)
        if v and not Path(v).is_absolute():
            setattr(args, k, str(ROOT / v))
    res = analyse(args)
    out = Path(args.out_dir)
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(res, indent=1, default=lambda o: sorted(o) if isinstance(o, set)
                                                      else str(o)))
    md = report(res)
    (out / f"{args.name}.md").write_text(md + "\n")
    if not args.quiet:
        print(md)
    return res


if __name__ == "__main__":
    main()
