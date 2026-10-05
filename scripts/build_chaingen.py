"""Decision-chain data (testbed TB-C, src/cotdistill/chaingen.py) and its shortcut audit.

  data/student_chain/train.jsonl         training domains, depths 1-4 (--n-train per depth), not filtered on depth
                                         needed; subqs = the checks of both chains (truth-valued, on_path, node_depth,
                                         cf/cf_value), random_subqs = matched questions about the two reserved facts
                                         (easy lookups: chain-data G4 is a lookup control, the mirror arm is G3M2)
  data/student_chain/val.jsonl           depths 1-8 (--n-val per depth), full depth only (cert_depth == depth), same
                                         schema (train_arm.sbatch scores its sub-questions with --eval-subq)
  data/eval/chain_val.jsonl              the same items as val.jsonl (rollout / parent-sensitivity evaluations)
  data/eval/chain_test_d{1..8}.jsonl     training domains and renderer, new cases, --n-test per depth, full depth only
  data/eval/chain_id.jsonl               chain_test_d1..d4 in one file (pooled; group = chain/d<k> in the predictions)
  data/eval/chain_dx.jsonl               chain_test_d5..d8 in one file (depth extrapolation)
  data/eval/chain_ho.jsonl               held-out domains and the chain_prose renderer, depths 1-6, --n-ho per depth,
                                         full depth only
  data/student_chain/audit.json          shortcut audit (also printed)
  data/student_chain/summary.json        counts, balance, certificate depth, node baselines, prompt length, md5s, audit

Every evaluation item needs its whole chain: cert_depth (the levels in the shallowest proof of the top check's value)
equals its depth. Training items keep the natural mix (summary: cert_depth by depth).

Shortcut audit: per depth, logistic regressions on shallow features (hashed word uni/bigrams of the prompt; counts of
fails/not/and/or; the top check's operator, link polarity and satisfied predicate literals; satisfied literals over
all checks; the prototype heuristic bit) predict whether the top check passes. They are fit on a separate full-depth
audit sample and scored on chain_test_d{k}; with the side condition given for free, their accuracy on that bit is the
final accuracy of a shallow solver (an upper bound). Gate: <= .60 at depth >= 3 (the best L2 strength is taken on the
test file, which only makes the bound looser), the top-passes share within .45-.55 at every depth, and cert_depth ==
depth in every evaluation file. Also reported (not gated): "read k levels, then guess" solvers, and the
"level 1 + link polarities" solver that ignores every predicate above level 1 (exact on full-depth items by
construction; on the natural training mix it shows how often facts above level 1 decide the top).

  python scripts/build_chaingen.py
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import statistics as st
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # small matrices: many BLAS threads
    os.environ.setdefault(_v, "2")                                         # only oversubscribe the CPU
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cotdistill import chaingen as cg  # noqa: E402
from cotdistill import rulegen as rg  # noqa: E402

TRAIN_DEPTHS = (1, 2, 3, 4)
TEST_DEPTHS = tuple(range(1, 9))
HO_DEPTHS = tuple(range(1, 7))
HASH_DIM = 2048


def write(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    print(f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}: {len(rows)}  md5 {md5}")
    return md5


# ---------------------------------------------------------------- audit
def heuristic_top(case: cg.ChainCase, pv: dict) -> bool:
    """Prototype heuristic: the top check from its predicate literals only (the link to the chain ignored)."""
    top = case.by_id()[case.top]
    vals = [pv[l.ref] == l.want for l in top.lits if l.kind == "p"]
    return cg.gate(top.op, vals) if vals else True


def dense_features(case: cg.ChainCase, prompt: str) -> list[float]:
    pv = cg.pred_values(case)
    by = case.by_id()
    top = by[case.top]
    tp = [pv[l.ref] == l.want for l in top.lits if l.kind == "p"]
    tc = [l.want for l in top.lits if l.kind == "c"]
    all_p = [pv[l.ref] == l.want for c in case.checks for l in c.lits if l.kind == "p"]
    checks = prompt.split("\nDecision: ")[0]
    bools = [case.facts[f.name] for f in case.domain.facts if f.kind == "bool"]
    return [float(top.op == "and"), float(sum(tc)), float(len(tc) - sum(tc)), float(len(tp)), float(sum(tp)),
            float(heuristic_top(case, pv)), float(pv[case.side]), float(sum(all_p)), float(len(all_p) - sum(all_p)),
            float(checks.count(" fails")), float(checks.count(" not ")), float(checks.count(" or ")),
            float(checks.count(" and ")), float(checks.count("at least one of")), float(checks.count("all of")),
            float(sum(bools)) / max(1, len(bools)), float(case.line_order.index(case.top)) / len(case.line_order),
            float(len(case.checks))]


def bow(prompt: str) -> list[int]:
    w = prompt.lower().replace("\n", " ").split()
    grams = w + [a + " " + b for a, b in zip(w, w[1:])]
    return sorted({int(hashlib.md5(g.encode()).hexdigest()[:8], 16) % HASH_DIM for g in grams})


def matrix(cases: list[cg.ChainCase], use_bow: bool) -> tuple[np.ndarray, np.ndarray]:
    rows, ys = [], []
    for c in cases:
        prompt = cg.render(c)
        x = np.array(dense_features(c, prompt), dtype=np.float32)
        if use_bow:
            b = np.zeros(HASH_DIM, dtype=np.float32)
            b[bow(prompt)] = 1.0
            x = np.concatenate([x, b])
        rows.append(x)
        ys.append(float(c.evaluate()["c"][c.top]))
    return np.stack(rows), np.array(ys, dtype=np.float32)


def fit_logistic(X: np.ndarray, y: np.ndarray, lam: float, steps: int = 400, lr: float = 0.05) -> np.ndarray:
    """L2 logistic regression (bias unpenalised), full-batch Adam."""
    Xb = np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], 1)
    w = np.zeros(Xb.shape[1], dtype=np.float32)
    m, v = np.zeros_like(w), np.zeros_like(w)
    pen = np.ones_like(w)
    pen[-1] = 0.0
    for t in range(1, steps + 1):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) / len(y) + lam * pen * w
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        w -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return w


def accuracy(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    Xb = np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], 1)
    return float(((Xb @ w > 0) == (y > 0.5)).mean())


def links_only(case: cg.ChainCase) -> bool:
    """The top check from the exact level-1 path check and the check literals above it, every predicate literal above
    level 1 ignored (a fan-in-only gate)."""
    v = case.evaluate()["c"]
    cv: dict[str, bool] = {}
    for c in case.checks:
        if c.on_path:
            cv[c.cid] = v[c.cid] if c.level == 1 else cg.gate(c.op, [cv[l.ref] == l.want for l in c.lits
                                                                     if l.kind == "c"])
    return cv[case.top]


def depth_solvers(cases: list[cg.ChainCase]) -> dict:
    """Accuracy on the top bit of solvers that read the top's chain only k levels deep and guess beyond (a solver that
    proves the value whenever cert_depth <= k, else a coin: P(cert <= k) + P(cert > k) / 2), and of links_only."""
    cert = [cg.cert_depth(c) for c in cases]
    top = [c.evaluate()["c"][c.top] for c in cases]
    out = {f"read{k}_guess": round(sum(x <= k for x in cert) / len(cert) + 0.5 * sum(x > k for x in cert) / len(cert),
                                   3) for k in (2, 4)}
    out["links_only"] = round(st.mean(links_only(c) == t for c, t in zip(cases, top)), 3)
    return out


def node_baselines(cases: list[cg.ChainCase]) -> dict:
    """Node-question baselines: level-1 checks from the operator alone (OR passes, AND fails) and checks above level 1
    from the value of their link literal (the parent check as stated, with the polarity applied: what a view that
    states the parents' results gives away), by chain."""
    op1, link = collections.defaultdict(list), collections.defaultdict(list)
    for c in cases:
        cv = c.evaluate()["c"]
        for ch in c.checks:
            tag = "path" if ch.on_path else "mirror"
            if ch.level == 1:
                op1[f"{tag}/{ch.op}"].append(cv[ch.cid])
                op1[tag].append(cv[ch.cid] == (ch.op == "or"))
            else:
                lk = next(l for l in ch.lits if l.kind == "c" and l.ref == f"{ch.cid[0]}{ch.level - 1}")
                link[tag].append(cv[ch.cid] == (cv[lk.ref] == lk.want))
    return {"level1_passes_by_op": {k: round(st.mean(v), 3) for k, v in sorted(op1.items()) if "/" in k},
            "level1_op_prior_acc": {k: round(st.mean(v), 3) for k, v in sorted(op1.items()) if "/" not in k},
            "above1_link_copy_acc": {k: round(st.mean(v), 3) for k, v in sorted(link.items())}}


def audit_depth(fit: list[cg.ChainCase], test: list[cg.ChainCase]) -> dict:
    out: dict = {}
    for name, use_bow in (("lr_dense", False), ("lr_bow+dense", True)):
        Xf, yf = matrix(fit, use_bow)
        Xt, yt = matrix(test, use_bow)
        mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
        Xf, Xt = (Xf - mu) / sd, (Xt - mu) / sd
        best = None
        for lam in (1e-4, 1e-3, 1e-2, 1e-1):
            w = fit_logistic(Xf, yf, lam)
            r = (accuracy(w, Xt, yt), accuracy(w, Xf, yf), lam)
            best = r if best is None or r[0] > best[0] else best
        out[name] = {"test": round(best[0], 3), "fit": round(best[1], 3), "lambda": best[2]}
    heur, o1, eff = [], [], []
    for c in test:
        res = c.evaluate()
        guess = c.outs[0] if heuristic_top(c, res["p"]) else (c.outs[1] if res["p"][c.side] else c.outs[2])
        heur.append(guess == res["decision"])
        o1.append(res["decision"] == c.outs[0])
    letters = collections.Counter(cg.eval_record(c)["gold_label"] for c in test)
    out["heuristic_top_from_literals"] = round(st.mean(heur), 3)
    out["always_outcome1"] = round(st.mean(o1), 3)
    out["majority_letter"] = round(max(letters.values()) / len(test), 3)
    return out


# ---------------------------------------------------------------- main
def describe(cases: list[cg.ChainCase]) -> dict:
    recs = [cg.eval_record(c) for c in cases]
    nodes = [n for c in cases for n in cg.tree_nodes(c)]
    depth = cases[0].depth
    lv = collections.defaultdict(list)
    for n in nodes:
        lv[f"{'path' if n['on_path'] else 'mirror'}/L{n['node_depth']}"].append(n["truth"])
    return {"n": len(cases), "top_passes": round(st.mean(r["top_passes"] for r in recs), 3),
            "gold_letter": dict(sorted(collections.Counter(r["gold_label"] for r in recs).items())),
            "outcome_rank": dict(sorted(collections.Counter(c.outs.index(r["gold"]) + 1
                                                            for c, r in zip(cases, recs)).items())),
            "cert_depth": dict(sorted(collections.Counter(r["cert_depth"] for r in recs).items())),
            "full_depth": round(st.mean(r["cert_depth"] == depth for r in recs), 3),
            "piv_run": dict(sorted(collections.Counter(r["piv_run"] for r in recs).items())),
            "prompt_chars": round(st.mean(len(r["prompt"]) for r in recs)),
            "prompt_tok_est": round(st.mean(len(r["prompt"]) for r in recs) / 3.6),
            "node_truth_by_level": {k: round(st.mean(v), 3) for k, v in sorted(lv.items())},
            "domains": dict(sorted(collections.Counter(c.domain.name for c in cases).items()))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1000, help="training items per depth (depths 1-4)")
    ap.add_argument("--n-val", type=int, default=100, help="val / chain_val items per depth (depths 1-8)")
    ap.add_argument("--n-test", type=int, default=400, help="chain_test_d<k> items per depth")
    ap.add_argument("--n-ho", type=int, default=200, help="chain_ho items per depth (depths 1-6)")
    ap.add_argument("--n-audit", type=int, default=2000, help="audit fitting sample per depth")
    ap.add_argument("--q", type=float, default=0.8, help="pivotality knob")
    ap.add_argument("--fanin", type=float, default=0.3)
    ap.add_argument("--out", default=str(ROOT / "data"), help="data root (student_chain/ and eval/ below it)")
    args = ap.parse_args()
    data = Path(args.out)
    dom = rg.TRAIN_DOMAINS
    kw = {"q": args.q, "fanin": args.fanin}

    train = [c for d in TRAIN_DEPTHS for c in cg.generate(args.n_train, d, dom, seed=21, prefix="chtr", **kw)]
    val = [c for d in TEST_DEPTHS
           for c in cg.generate(args.n_val, d, dom, seed=22, prefix="chva", full_depth=True, **kw)]
    test = {d: cg.generate(args.n_test, d, dom, seed=23, prefix="chte", full_depth=True, **kw) for d in TEST_DEPTHS}
    ho = [c for d in HO_DEPTHS for c in cg.generate(args.n_ho, d, rg.HELDOUT_DOMAINS, seed=25, prefix="chho",
                                                    full_depth=True, renderer="chain_prose", **kw)]
    random.Random(0).shuffle(train)              # interleave depths and domains
    random.Random(1).shuffle(val)
    random.Random(2).shuffle(ho)

    md5 = {}
    rng = random.Random(0)
    md5["student_chain/train.jsonl"] = write(data / "student_chain/train.jsonl", [cg.train_record(c, rng) for c in train])
    val_rows = [cg.train_record(c, rng) for c in val]
    md5["student_chain/val.jsonl"] = write(data / "student_chain/val.jsonl", val_rows)
    md5["eval/chain_val.jsonl"] = write(data / "eval/chain_val.jsonl", val_rows)
    test_rows = {d: [cg.eval_record(c) for c in cs] for d, cs in test.items()}
    for d, rows in test_rows.items():
        md5[f"eval/chain_test_d{d}.jsonl"] = write(data / f"eval/chain_test_d{d}.jsonl", rows)
    md5["eval/chain_id.jsonl"] = write(data / "eval/chain_id.jsonl", [r for d in TRAIN_DEPTHS for r in test_rows[d]])
    md5["eval/chain_dx.jsonl"] = write(data / "eval/chain_dx.jsonl",
                                       [r for d in TEST_DEPTHS if d not in TRAIN_DEPTHS for r in test_rows[d]])
    ho_rows = [cg.eval_record(c) for c in ho]
    md5["eval/chain_ho.jsonl"] = write(data / "eval/chain_ho.jsonl", ho_rows)
    full = all(r["cert_depth"] == r["depth"] for r in val_rows + ho_rows + [r for v in test_rows.values() for r in v])

    train_prompts = {cg.render(c) for c in train}
    summary: dict = {"args": vars(args), "seeds": {"train": 21, "val": 22, "test": 23, "audit": 24, "ho": 25},
                     "md5": md5,
                     "train_prompt_overlap": {"val": sum(cg.render(c) in train_prompts for c in val),
                                              **{f"test_d{d}": sum(cg.render(c) in train_prompts for c in cs)
                                                 for d, cs in test.items()}},
                     "train": {f"d{d}": describe([c for c in train if c.depth == d]) for d in TRAIN_DEPTHS},
                     "val": {f"d{d}": describe([c for c in val if c.depth == d]) for d in TEST_DEPTHS},
                     "test": {f"d{d}": describe(cs) for d, cs in test.items()},
                     "ho": {f"d{d}": describe([c for c in ho if c.depth == d]) for d in HO_DEPTHS},
                     "node_baselines": {"train": node_baselines(train), "val": node_baselines(val)}}
    nodes = [cg.train_record(c, random.Random(0)) for c in train[:500]]
    summary["train_views"] = {"subqs_mean": round(st.mean(len(r["subqs"]) for r in nodes), 2),
                              "random_subqs_mean": round(st.mean(len(r["random_subqs"]) for r in nodes), 2),
                              "random_subqs_min": min(len(r["random_subqs"]) for r in nodes)}

    print("\nshortcut audit (top-check bit; with the side condition given it is the final accuracy):")
    audit, ok = {}, full
    for d in TEST_DEPTHS:
        fit = cg.generate(args.n_audit, d, dom, seed=24, prefix="chau", full_depth=True, **kw)
        a = audit_depth(fit, test[d])
        a["top_passes_test"] = summary["test"][f"d{d}"]["top_passes"]
        a["solvers_test"] = depth_solvers(test[d])
        nat = cg.generate(min(args.n_audit, 1000), d, dom, seed=26, prefix="chna", **kw)
        a["solvers_natural"] = depth_solvers(nat)
        a["cert_depth_natural"] = dict(sorted(collections.Counter(cg.cert_depth(c) for c in nat).items()))
        audit[f"d{d}"] = a
        lr_max = max(a["lr_dense"]["test"], a["lr_bow+dense"]["test"])
        bal = 0.45 <= a["top_passes_test"] <= 0.55
        gate = bal and (d < 3 or lr_max <= 0.60)
        ok &= gate
        print(f"  d{d}: LR dense {a['lr_dense']['test']:.3f} (fit {a['lr_dense']['fit']:.3f})  "
              f"LR bow+dense {a['lr_bow+dense']['test']:.3f} (fit {a['lr_bow+dense']['fit']:.3f})  "
              f"heuristic {a['heuristic_top_from_literals']:.3f}  outcome1 {a['always_outcome1']:.3f}  "
              f"maj.letter {a['majority_letter']:.3f}  top-pass {a['top_passes_test']:.3f}  "
              f"read2/read4 {a['solvers_test']['read2_guess']:.3f}/{a['solvers_test']['read4_guess']:.3f}  "
              f"links-only test/natural {a['solvers_test']['links_only']:.3f}/"
              f"{a['solvers_natural']['links_only']:.3f}  {'ok' if gate else 'FAIL'}", flush=True)
    audit["full_depth_eval"] = full
    audit["gate"] = "pass" if ok else "fail"
    print(f"  cert_depth == depth in every evaluation file: {full}")
    print(f"  gate (LR <= .60 at depth >= 3, top-pass share .45-.55 at every depth, full-depth evaluation): "
          f"{audit['gate']}")
    summary["audit"] = audit
    (data / "student_chain/audit.json").write_text(json.dumps(audit, indent=1))
    (data / "student_chain/summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("train_prompt_overlap", "train_views", "node_baselines")}, indent=1))
    for split in ("train", "val", "test", "ho"):
        for d, s in summary[split].items():
            print(f"{split} {d}: cert_depth {s['cert_depth']}  piv_run {s['piv_run']}  chars {s['prompt_chars']} "
                  f"(~{s['prompt_tok_est']} tok)  top-pass {s['top_passes']}  letters {s['gold_letter']}  "
                  f"outcome rank {s['outcome_rank']}")


if __name__ == "__main__":
    main()
