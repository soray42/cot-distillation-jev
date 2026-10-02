"""Layer-wise linear probes on readout hidden-state dumps (early-exit and depth-to-layer analysis).

Reads <run>/hidden_<set>.pt ([n, len(layers) + 1, d]: the readout-position state after each dumped layer, then the final
readout state; written with --dump-hidden/--dump-layers), in the item order of the evaluation file. For Knights & Knaves
sets it probes, at every dumped layer:
  decision   the gold option letter (one-vs-rest over the item's options)
  roles      the role (knight/knave) of the person at each of the first four positions of the gold assignment; the
             readout state alone does not show the options, so a decodable role means the state carries that
             intermediate judgment, not just the answer letter
Probes are ridge regressions in the dual form (n < d), on features standardised with the training mean and deviation,
with the penalty chosen by 5-fold cross-validation on the training set. They are fitted on the training sets and scored on
the test sets, and on the depth-graded set per puzzle size (number of inhabitants).

  python scripts/layer_probe.py runs/TF-ep2-...-s2 runs/T1-...-s2 --layers 4,8,12,16,20,24 --out runs/report_layer_probe.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (Path.cwd() / "data/eval").exists() and not (ROOT / "data/eval").exists():   # HPC: data next to the repo clone
    ROOT = Path.cwd()
SETS = {"kk": "data/eval/kk_heldout.jsonl", "diag": "data/eval/diag_kk.jsonl", "kkdeep": "data/eval/kk_deep.jsonl"}


def load_items(path: Path) -> list[dict]:
    out = []
    for line in open(path):
        it = json.loads(line)
        tail = it["prompt"].rpartition("\nOptions:\n")[2]
        opts = [m.group(2) for m in (re.match(r"^\(?([A-Z])\) (.*)$", ln) for ln in tail.splitlines()) if m]
        gold = it["labels"].index(it["gold_label"])
        roles = re.findall(r"(\w+): (knight|knave)", opts[gold])
        out.append({"gold": gold, "n_opts": len(opts), "roles": [r == "knight" for _, r in roles],
                    "depth": it.get("depth") or len(roles)})
    return out


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Dual ridge: W = X^T (X X^T + lam I)^-1 Y."""
    K = X @ X.T
    return X.T @ np.linalg.solve(K + lam * np.eye(len(X)), Y)


def choose_lam(X: np.ndarray, Y: np.ndarray, score, lams=(1e1, 1e2, 1e3, 1e4, 1e5)) -> float:
    idx = np.arange(len(X)) % 5
    best, best_s = lams[0], -1.0
    for lam in lams:
        s = np.mean([score(X[idx == f] @ ridge_fit(X[idx != f], Y[idx != f], lam), f) for f in range(5)])
        if s > best_s:
            best, best_s = lam, s
    return best


def standardise(tr: np.ndarray, *others: np.ndarray):
    mu, sd = tr.mean(0), tr.std(0) + 1e-4
    return [(a - mu) / sd for a in (tr, *others)]


def decision_probe(Xtr, itr, Xte_list, ite_list):
    k = max(i["n_opts"] for i in itr)
    Y = -np.ones((len(itr), k))
    for r, it in enumerate(itr):
        Y[r, it["gold"]] = 1.0
    mask_tr = np.array([[j < it["n_opts"] for j in range(k)] for it in itr])

    def acc(S, items, mask):
        S = np.where(mask, S, -np.inf)
        return float(np.mean(S.argmax(1) == np.array([it["gold"] for it in items])))
    folds = np.arange(len(itr)) % 5
    lam = choose_lam(Xtr, Y, lambda S, f: acc(S, [it for it, ff in zip(itr, folds) if ff == f], mask_tr[folds == f]))
    W = ridge_fit(Xtr, Y, lam)
    out = []
    for Xte, ite in zip(Xte_list, ite_list):
        kk = max(k, max(i["n_opts"] for i in ite))
        S = np.full((len(ite), kk), -np.inf)
        S[:, :k] = Xte @ W
        mask = np.array([[j < it["n_opts"] and j < k for j in range(kk)] for it in ite])
        out.append(np.where(mask, S, -np.inf).argmax(1) == np.array([it["gold"] for it in ite]))
    return out


def role_probe(Xtr, itr, Xte_list, ite_list, positions=4):
    res = [[] for _ in Xte_list]
    for p in range(positions):
        tr = [r for r, it in enumerate(itr) if len(it["roles"]) > p]
        y = np.array([1.0 if itr[r]["roles"][p] else -1.0 for r in tr])
        Xp = Xtr[tr]
        folds = np.arange(len(tr)) % 5
        lam = choose_lam(Xp, y[:, None], lambda S, f: float(np.mean((S[:, 0] > 0) == (y[folds == f] > 0))))
        W = ridge_fit(Xp, y[:, None], lam)
        for j, (Xte, ite) in enumerate(zip(Xte_list, ite_list)):
            te = [r for r, it in enumerate(ite) if len(it["roles"]) > p]
            pred = (Xte[te] @ W)[:, 0] > 0
            truth = np.array([ite[r]["roles"][p] for r in te])
            res[j].append((te, pred == truth))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--layers", default="4,8,12,16,20,24")
    ap.add_argument("--train", default="kk")
    ap.add_argument("--test", default="diag,kkdeep")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import torch
    layers = [int(x) for x in args.layers.split(",")] + ["final"]
    items = {s: load_items(ROOT / p) for s, p in SETS.items()}
    report = {}
    for r in args.runs:
        run = Path(r)
        H = {s: torch.load(run / f"hidden_{s}.pt", map_location="cpu").float().numpy()
             for s in [args.train] + args.test.split(",")}
        rows = {}
        for li, L in enumerate(layers):
            Xtr = H[args.train][:, li]
            tests = args.test.split(",")
            Xs = standardise(Xtr, *[H[t][:, li] for t in tests])
            dec = decision_probe(Xs[0], items[args.train], Xs[1:], [items[t] for t in tests])
            rol = role_probe(Xs[0], items[args.train], Xs[1:], [items[t] for t in tests])
            row = {}
            for t, d, rp in zip(tests, dec, rol):
                row[f"{t}_decision"] = round(float(d.mean()), 4)
                row[f"{t}_roles"] = round(float(np.mean(np.concatenate([ok for _, ok in rp]))), 4)
                if t == "kkdeep":
                    depth = np.array([it["depth"] for it in items[t]])
                    for dv in sorted(set(depth.tolist())):
                        row[f"kkdeep_decision_d{dv}"] = round(float(d[depth == dv].mean()), 4)
            rows[str(L)] = row
        report[run.name] = rows
        print(f"\n{run.name}")
        keys = list(next(iter(rows.values())).keys())
        main_keys = [k for k in keys if not k.startswith("kkdeep_decision_d")]
        print(f"{'layer':>6s} " + " ".join(f"{k:>16s}" for k in main_keys))
        for L, row in rows.items():
            print(f"{L:>6s} " + " ".join(f"{row[k]:16.3f}" for k in main_keys))
        dk = [k for k in keys if k.startswith("kkdeep_decision_d")]
        if dk:
            print("kkdeep decision by puzzle size: " + "  ".join(
                f"{k[18:]}: " + "/".join(f"{rows[str(L)][k]:.2f}" for L in layers) for k in dk))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
