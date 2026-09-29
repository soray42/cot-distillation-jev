"""Summarize a teacher pilot run (teacher_cache/<run>/*.json).

Reports: teacher accuracy and saturation, trace agreement, trace-confidence AUROC vs a
length baseline, sub-question yield, quote anchoring, predicate matching, teacher accuracy
and calibration on verifiable sub-questions, and a node-certainty proxy (K5 pre-check).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def auroc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def fmt(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot1")
    args = ap.parse_args()
    d = ROOT / "teacher_cache" / args.run
    res = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
    calls = [json.loads(l) for l in open(d / "calls.jsonl")] if (d / "calls.jsonl").exists() else []
    if not res:
        sys.exit(f"no results in {d}")
    S: dict = {"n_items": len(res)}

    # ---- final answers ----
    correct, top1, agree, by_dom, by_depth = [], [], [], defaultdict(list), defaultdict(list)
    conf_ok, conf_bad, len_ok, len_bad = defaultdict(list), defaultdict(list), [], []
    for r in res:
        gold = r["item"]["gold_label"]
        answers = []
        for tr in r["traces"]:
            dist = tr["dist"]
            lab = max(dist["probs"], key=dist["probs"].get) if dist["probs"] else None
            answers.append(lab)
            ok = lab == gold
            correct.append(ok)
            if dist["probs"]:
                top1.append(max(dist["probs"].values()))
            for key in ("mean", "bottom10", "tail", "lowest"):
                (conf_ok if ok else conf_bad)[key].append(tr["conf"][key])
            (len_ok if ok else len_bad).append(-tr["conf"]["n"])      # shorter = more confident
            by_dom[r["item"]["domain"]].append(ok)
            by_depth[min(r["item"]["depth"], 6)].append(ok)
        if len(answers) > 1:
            agree.append(len(set(answers)) == 1)
    S["teacher_acc"] = sum(correct) / len(correct)
    S["acc_by_domain"] = {k: round(sum(v) / len(v), 3) for k, v in sorted(by_dom.items())}
    S["acc_by_depth"] = {k: (round(sum(v) / len(v), 3), len(v)) for k, v in sorted(by_depth.items())}
    S["ask_items"] = sum(r["item"]["gold"] == "ask" for r in res)
    S["median_top1"] = st.median(top1) if top1 else None
    S["share_top1_ge_0.999"] = sum(t >= 0.999 for t in top1) / len(top1) if top1 else None
    S["trace_agreement"] = sum(agree) / len(agree) if agree else None
    S["trace_conf_auroc_pooled"] = {k: auroc(conf_ok[k], conf_bad[k]) for k in conf_ok}
    S["length_baseline_auroc"] = auroc(len_ok, len_bad)

    # ---- sub-questions ----
    n_sq, revised, located, matched, stated_consistent = [], 0, 0, 0, []
    ver = []          # (p_yes_mean, truth, span_min_p, status)
    fresh = []        # (p_yes without CoT, truth)
    for r in res:
        sqs = r["subquestions"]
        n_sq.append(len(sqs))
        for sq in sqs:
            revised += sq.get("status") == "revised"
            located += sq.get("span") is not None
            ps = [a["p_yes"] for a in sq.get("answers", []) if a.get("p_yes") is not None]
            p = sum(ps) / len(ps) if ps else None
            if p is not None and sq.get("stated") in ("yes", "no"):
                stated_consistent.append((p > 0.5) == (sq["stated"] == "yes"))
            if sq.get("match", {}).get("pid"):
                matched += 1
                if sq.get("truth") is not None and p is not None:
                    mp = (sq.get("span_stats") or {}).get("min_p")
                    ver.append((p, sq["truth"], mp, sq.get("status")))
                    pf = (sq.get("answer_nocot") or {}).get("p_yes")
                    if pf is not None:
                        fresh.append((pf, sq["truth"]))
    tot = sum(n_sq)
    S["subq_per_item"] = {"mean": st.mean(n_sq), "median": st.median(n_sq), "zero_items": sum(x == 0 for x in n_sq)}
    S["subq_total"] = tot
    S["revised_share"] = revised / tot if tot else None
    S["quote_located_share"] = located / tot if tot else None
    S["matched_to_predicate_share"] = matched / tot if tot else None
    S["p_yes_consistent_with_stated_answer"] = sum(stated_consistent) / len(stated_consistent) if stated_consistent else None
    if ver:
        acc = [(p > 0.5) == t for p, t, _, _ in ver]
        S["verifiable_n"] = len(ver)
        S["teacher_subq_acc"] = sum(acc) / len(acc)
        S["teacher_subq_brier"] = st.mean([(p - (1.0 if t else 0.0)) ** 2 for p, t, _, _ in ver])
        S["subq_p_is_saturated_share"] = sum(p >= 0.99 or p <= 0.01 for p, _, _, _ in ver) / len(ver)
        conf = [max(p, 1 - p) for p, _, _, _ in ver]
        S["subq_conf_auroc"] = auroc([c for c, a in zip(conf, acc) if a], [c for c, a in zip(conf, acc) if not a])
        sp = [(mp, a) for (_, _, mp, _), a in zip(ver, acc) if mp is not None]
        S["node_proxy_minp_auroc(K5 pre-check)"] = auroc([m for m, a in sp if a], [m for m, a in sp if not a])
        if fresh:
            facc = [(p > 0.5) == t for p, t in fresh]
            S["fresh_subq_acc"] = sum(facc) / len(facc)
            S["fresh_subq_brier"] = st.mean([(p - (1.0 if t else 0.0)) ** 2 for p, t in fresh])
            S["fresh_subq_saturated_share"] = sum(p >= 0.99 or p <= 0.01 for p, _ in fresh) / len(fresh)
            fconf = [max(p, 1 - p) for p, _ in fresh]
            S["fresh_subq_conf_auroc"] = auroc([c for c, a in zip(fconf, facc) if a], [c for c, a in zip(fconf, facc) if not a])
        rev = [a for (_, _, _, s), a in zip(ver, acc) if s == "revised"]
        S["revised_verifiable_acc"] = (sum(rev) / len(rev), len(rev)) if rev else None

    # ---- cost ----
    usd = sum(c["usd"] for c in calls)
    S["calls"] = len(calls)
    S["usd_total"] = usd
    S["usd_per_item"] = usd / len(res)
    tags = Counter(c["tag"].split("/")[-1].rstrip("0123456789.t") for c in calls)
    S["calls_by_kind"] = dict(tags)
    S["fingerprints"] = dict(Counter(c.get("fingerprint") for c in calls))

    for k, v in S.items():
        print(f"{k:40s} {v if isinstance(v, (dict, list, tuple)) else fmt(v)}")
    (d / "summary.json").write_text(json.dumps(S, indent=1, default=str))


if __name__ == "__main__":
    main()
