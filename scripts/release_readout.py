"""Release-line readout by the rules registered in notes/prereg_release.md.

  table      accuracy and ECE per set for every run (ours and peers; preds_<set>.jsonl in each run directory)
  peers      paired bootstrap (over items) of our candidate minus each peer, per set; "ahead" needs a 95% interval
             above 0, and on JevBench also a margin of at least 4 tasks
  rlcd       RLCD criteria against stage 1:
             - TD yes/no accuracy
             - claims_ho negation consistency: mean P(Yes|claim) + P(Yes|negation) over the -p/-pn and -a/-an pairs
             - drops on BBH, MuSR and K&K diag
             (JevBench ECE after post-hoc calibration comes from runs/report_calibration.txt)
  exit       the early-exit layer L*: the shallowest layer certified by Learn-then-Test on v4heldout (disagreement with
             the full model <= .05 with probability >= .9, fixed-sequence testing from the deepest layer), next to the
             earlier >= 95%-agreement rule; then the accuracy change on each test set at L* and the latency ratio

  claim      per reasoning set: ahead of every comparator that ran (prereg_release addendum 2026-10-02 21:05 UTC)

  python scripts/release_readout.py --ours runs/TF-v4t-...-s0,runs/reason-sev2b \\
      --rlcd runs/A2-rlcd-model-student_rlcd-s0,runs/reason-rlcd --others A2-v4=runs/A2-v4-... TF-ep2=runs/TF-ep2-... \\
      --peers decider=runs/peer-decider,runs/reason-peer-decider strands=runs/peer-strands,runs/reason-peer-strands \\
      jeff=runs/jeff-native,runs/reason-jeff qwen2b=runs/reason-qwen2b-nothink \\
      --exit runs/early-exit-TF-v4t-.../early_exit.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if (Path.cwd() / "data/eval").exists() and not (ROOT / "data/eval").exists():
    ROOT = Path.cwd()
SETS = ["jevbench", "bbh", "musr", "td", "diag", "kk", "policy", "folio", "claims_ho", "tdq", "tdn", "v4heldout",
        "bbeh", "xkk", "proverqa", "zebra", "zebra_all", "logiqa2", "lsat_lr", "lsat_ar"]
REASONING = ["bbh", "musr", "diag", "bbeh", "xkk", "proverqa", "zebra", "logiqa2", "lsat_lr", "lsat_ar"]
MERGED = {"zebra_all": ["zebra", "zebra_ovl"]}      # registered sensitivity set: zebra plus its overlap-dropped items
EVAL_FILES = {"td": "data/eval/typed_decisions_test.jsonl", "claims_ho": "data/eval/claims_ho.jsonl"}


def preds(dirs: list[Path], name: str) -> dict[str, dict]:
    """A model's predictions on one set: from the first of its directories that has them (a run directory, then e.g.
    runs/reason-<name> for the external reasoning sets); a merged set joins its parts."""
    if name in MERGED:
        parts = [preds(dirs, n) for n in MERGED[name]]
        return {k: v for p in parts for k, v in p.items()} if all(parts) else {}
    for d in dirs:
        f = d / f"preds_{name}.jsonl"
        if f.exists():
            return {json.loads(l)["item_id"]: json.loads(l) for l in open(f)}
    return {}


def correct(p: dict) -> bool:
    return p["labels"][max(range(len(p["probs"])), key=p["probs"].__getitem__)] == p["gold_label"]


def ece(ps: list[dict]) -> float:
    bins = [[0, 0.0, 0.0] for _ in range(10)]
    for p in ps:
        k = max(range(len(p["probs"])), key=p["probs"].__getitem__)
        b = min(9, int(p["probs"][k] * 10))
        bins[b][0] += 1; bins[b][1] += p["probs"][k]; bins[b][2] += p["labels"][k] == p["gold_label"]
    return sum(abs(c - a) for _, c, a in bins) / max(1, len(ps))


def boot(a: list[int], b: list[int], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    m = sum(d) / len(d)
    reps = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return m, reps[int(0.025 * n)], reps[int(0.975 * n)]


def yes_prob(p: dict, item: dict) -> float:
    names = item.get("label_names") or {}
    yes = next(L for L, nm in names.items() if str(nm).lower() in ("true", "yes"))
    return p["probs"][p["labels"].index(yes)]


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(Binomial(n, p) <= k), summed in log space."""
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
             for i in range(k + 1)]
    m = max(terms)
    return math.exp(m) * sum(math.exp(t - m) for t in terms)


def ltt_layer(sel: dict, layers: list[str], alpha: float = 0.05, delta: float = 0.1):
    """Learn-then-Test (Angelopoulos et al. 2021, Thm 1 with the fixed-sequence procedure, Prop 3): loss = the
    tuned-lens answer at layer L differs from the full model's. Each layer tests H0 "disagreement rate > alpha" with
    the exact binomial p-value P(Bin(n, alpha) <= k), from the deepest evaluated layer to the shallowest, stopping at
    the first p > delta. With probability >= 1 - delta every certified layer disagrees with the full model on at
    most a fraction alpha of items drawn like the selection set. Returns (shallowest certified layer or None,
    [(layer, disagreements, p)] in testing order)."""
    n = sel.get("n") or 0
    out, best = [], None
    for L in sorted(layers, key=int, reverse=True):
        k = round((1 - sel["layers"][L]["tuned"]["agree_full"]) * n)
        pv = binom_cdf(k, n, alpha)
        out.append((L, k, pv))
        if pv > delta:
            break
        best = L
    return best, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--rlcd", default=None)
    ap.add_argument("--others", nargs="*", default=[])
    ap.add_argument("--peers", nargs="*", default=[])
    ap.add_argument("--exit", default=None)
    args = ap.parse_args()
    dirs = lambda v: [Path(x) for x in v.split(",")]      # noqa: E731  (a model's directories, comma-separated)
    runs = {"ours": dirs(args.ours)}
    if args.rlcd:
        runs["rlcd"] = dirs(args.rlcd)
    for spec in args.others + args.peers:
        k, v = spec.split("=", 1)
        runs[k] = dirs(v)
    P = {k: {s: preds(r, s) for s in SETS} for k, r in runs.items()}

    print("== table (accuracy / ECE)")
    print(f"{'run':10s}" + "".join(f"{s:>13s}" for s in SETS))
    for k in runs:
        cells = []
        for s in SETS:
            ps = list(P[k][s].values())
            cells.append(f"{sum(map(correct, ps)) / len(ps):.3f}/{ece(ps):.2f}" if ps else "-")
        print(f"{k:10s}" + "".join(f"{c:>13s}" for c in cells))

    for cand in [c for c in ("ours", "rlcd") if c in runs]:
        for spec in args.peers:
            peer = spec.split("=", 1)[0]
            print(f"\n== {cand} minus {peer} (paired bootstrap 95%)")
            for s in SETS:
                ids = [i for i in P[cand][s] if i in P[peer][s]]
                if not ids:
                    continue
                a = [int(correct(P[cand][s][i])) for i in ids]
                b = [int(correct(P[peer][s][i])) for i in ids]
                m, lo, hi = boot(a, b)
                ahead = lo > 0 and (s != "jevbench" or sum(a) - sum(b) >= 4)
                behind = hi < 0
                print(f"  {s:10s} n={len(ids):5d} diff {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  tasks {sum(a)} vs {sum(b)}"
                      f"  {'AHEAD' if ahead else ('BEHIND' if behind else '')}")

    peers = [spec.split("=", 1)[0] for spec in args.peers]
    if peers:
        print(f"\n== claim (prereg_release 21:05 UTC): ahead of every comparator that ran ({', '.join(peers)})")
        for cand in [c for c in ("ours", "rlcd") if c in runs]:
            for s in REASONING:
                ran = [p for p in peers if P[p][s]]
                if not P[cand][s] or not ran:
                    continue
                verdicts = []
                for peer in ran:
                    ids = [i for i in P[cand][s] if i in P[peer][s]]
                    a = [int(correct(P[cand][s][i])) for i in ids]
                    b = [int(correct(P[peer][s][i])) for i in ids]
                    _, lo, _ = boot(a, b)
                    verdicts.append(lo > 0 and (s != "jevbench" or sum(a) - sum(b) >= 4))
                missing = [p for p in peers if p not in ran]
                print(f"  {cand:5s} {s:9s} {'CLAIM' if all(verdicts) else 'no claim':9s} ahead of "
                      f"{sum(verdicts)}/{len(ran)}" + (f"  (not run: {', '.join(missing)})" if missing else ""))

    if "rlcd" in runs:
        print("\n== RLCD criteria (stage 2 vs stage 1)")
        td_items = {json.loads(l)["item_id"]: json.loads(l) for l in open(ROOT / EVAL_FILES["td"])}
        noul = [i for i, it in td_items.items() if it["type"] == "noul"]
        cl_items = {json.loads(l)["item_id"]: json.loads(l) for l in open(ROOT / EVAL_FILES["claims_ho"])}
        for k in ("ours", "rlcd"):
            td = [correct(P[k]["td"][i]) for i in noul if i in P[k]["td"]]
            sums = []
            for i, it in cl_items.items():
                m = re.match(r"^(.*)-(p|a)$", i)
                if m and f"{i}n" in cl_items and i in P[k]["claims_ho"] and f"{i}n" in P[k]["claims_ho"]:
                    sums.append(yes_prob(P[k]["claims_ho"][i], it) + yes_prob(P[k]["claims_ho"][f"{i}n"], cl_items[f"{i}n"]))
            line = f"  {k:5s} TD yes/no acc {sum(td) / max(1, len(td)):.3f} (>= .72)"
            if sums:
                line += f" | negation sum {sum(sums) / len(sums):.3f} (1 +- .1, n={len(sums)})"
            print(line)
        for s in ("bbh", "musr", "diag"):
            a = [correct(p) for p in P["ours"][s].values()]
            b = [correct(p) for p in P["rlcd"][s].values()]
            if a and b:
                d = sum(b) / len(b) - sum(a) / len(a)
                print(f"  {s:5s} stage2 - stage1 {d:+.3f} ({'ok' if d >= -0.01 else 'DROP > 1 pt'})")

    if args.exit and Path(args.exit).exists():
        ex = json.load(open(args.exit))
        sel = ex["sets"].get("v4heldout", {})
        layers = [k for k in sel.get("layers", {}) if k != "full" and "tuned" in sel["layers"][k]]
        rule95 = next((L for L in sorted(layers, key=int) if sel["layers"][L]["tuned"]["agree_full"] >= 0.95), None)
        lstar, cert = ltt_layer(sel, layers)
        print(f"\n== early exit (v4heldout, n={sel.get('n')}): L* = {lstar} by Learn-then-Test (alpha .05, delta .1, "
              f"fixed sequence from the deepest layer; prereg_release addendum 2026-10-02 21:34 UTC); the earlier "
              f">= .95 agreement rule gives {rule95}")
        for L, k, pv in cert:
            print(f"  layer {L:>2s}: {k} disagreements, p = {pv:.4f}{'  certified' if pv <= 0.1 else '  stop'}")
        if lstar:
            drops = []
            for s, v in ex["sets"].items():
                if s == "v4heldout":
                    continue
                full = v["layers"]["full"]["lens"]["acc"]
                at = v["layers"][lstar]["tuned"]["acc"]
                drops.append(full - at)
                print(f"  {s:10s} full {full:.3f}  L*{lstar} {at:.3f}  change {at - full:+.3f}  agree {v['layers'][lstar]['tuned']['agree_full']:.3f}")
            lat = ex.get("latency_s", {})
            nl = str(ex["n_layers"])
            if lstar in lat and nl in lat:
                print(f"  latency {lat[lstar] * 1000:.1f} ms vs {lat[nl] * 1000:.1f} ms (x{lat[nl] / lat[lstar]:.2f} faster)")
            ok = sum(drops) / len(drops) <= 0.01 and max(drops) <= 0.03
            print(f"  claim rule (mean drop <= 1 pt, none > 3 pts): {'MET' if ok else 'NOT MET'}")


if __name__ == "__main__":
    main()
