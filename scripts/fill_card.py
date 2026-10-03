"""Fill the release model card's result placeholders from prediction files.

Every number comes from preds_<set>.jsonl files; nothing is typed by hand:
- the accuracy table (Sev-2b and each comparator, our harness) and the paired-bootstrap difference table
  (scripts/release_readout.py: same scoring, same bootstrap);
- the claim sentence, by the registered rule (prereg_release 21:05 UTC): a reasoning set is named when the candidate
  is ahead of every comparator that ran on it (95% interval above 0). JevBench public and TD are named only when the
  candidate is ahead of decider-2b and Strands, on JevBench also by at least 4 tasks;
- the per-type temperatures (scripts/fit_calibration.py, fitted on val, v4heldout and claims_ho of the candidate) and
  the calibration table;
- alpha, the arm text and the GPU time.
Owner decisions (repository, licence, author) come from arguments. Any placeholder left over is listed at the end;
scripts/hf_upload.py refuses a card that still has one.

  python scripts/fill_card.py --card notes/release/sev-2b-v5/README.md --out notes/release/sev-2b-v5/README.filled.md \\
      --cand runs/wise-M1-test/a0.8,runs/wise-M1-sel/a0.8 --arm M1 --alpha 0.8 --gpu-hours 5.1 \\
      --peers decider=dirs strands=dirs jeff=dirs base=dirs --repo <account>/Sev-2b --author "<name>"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fit_calibration import fit, load, metrics  # noqa: E402
from release_readout import EVAL_FILES, REASONING, boot, correct, preds, yes_prob  # noqa: E402

ROWS = {"BBH": "bbh", "MuSR": "musr", "BBEH": "bbeh", "K&K diag": "diag", "xkk (Xie et al.)": "xkk",
        "ProverQA": "proverqa", "ZebraLogic mc": "zebra", "LogiQA 2.0": "logiqa2", "LSAT-LR": "lsat_lr",
        "LSAT-AR": "lsat_ar", "JevBench public": "jevbench", "TD": "td",
        "ZebraLogic mc, with overlap items (sensitivity)": "zebra_all"}
PEER_NAMES = {"decider": "decider-2b v11", "strands": "Strands Decider 2B v19", "jeff": "Jeff-Qwen3.5-2B"}
ARMS = {
    "M1": ("TF-M1 (yes/no node questions)",
           'Each node is asked as a yes/no question ("Intermediate question: ..." with the options Yes and No in '
           "random order).", ""),
    "M2": ("TF-M2 (multiple-choice node questions)",
           'Each node is asked inside a multiple-choice view, "Which of these questions has the answer Yes?", over '
           'the node, up to three other nodes of the same tree and "None of them"; the target follows the teacher\'s '
           "confidence in each node.",
           "Node questions are multiple-choice views instead of single yes/no questions."),
}


def dirs(spec: str) -> list[Path]:
    return [Path(p) for p in spec.split(",") if p]


def scored(ds: list[Path], name: str) -> dict[str, int]:
    return {k: int(correct(v)) for k, v in preds(ds, name).items()}


def cal_rows(ds: list[Path], name: str) -> list[dict]:
    d = next((x for x in ds if (x / f"preds_{name}.jsonl").exists()), None)
    return load(d, name) if d else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cand", required=True, help="comma-separated directories of the released weights' predictions")
    ap.add_argument("--peers", nargs="+", required=True, help="decider=dirs strands=dirs jeff=dirs base=dirs")
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--alpha", required=True)
    ap.add_argument("--gpu-hours", required=True)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--author", default=None)
    ap.add_argument("--licence", default=None, help="e.g. cc-by-nc-4.0 (sets the front matter and both licence lines)")
    ap.add_argument("--name", default="Sev-2B", help="model name used in the claim sentence")
    ap.add_argument("--infer-check", default=None, help="sentence on the sev_infer.py reproduction check")
    args = ap.parse_args()
    s = Path(args.card).read_text()
    cand = dirs(args.cand)
    peers = {k: dirs(v) for k, v in (p.split("=", 1) for p in args.peers)}
    comps = [k for k in ("decider", "strands", "jeff") if k in peers]

    # accuracy and difference tables
    acc_cache, diff_cache = {}, {}
    for label, name in ROWS.items():
        c = scored(cand, name)
        row = [f"{sum(c.values()) / len(c):.3f}" if c else "not run"]
        drow = []
        for k in comps + (["base"] if "base" in peers else []):
            p = scored(peers[k], name)
            row.append(f"{sum(p.values()) / len(p):.3f}" if p else "not run")
            common = sorted(set(c) & set(p))
            if c and p and len(common) >= 0.95 * len(c):
                m, lo, hi = boot([c[i] for i in common], [p[i] for i in common])
                drow.append(f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
                diff_cache[(name, k)] = (m, lo, hi, len(common))
            else:
                drow.append("not run")
        acc_cache[name] = row
        pat = re.compile(rf"^\| {re.escape(label)} \| ([\d,]+) \|( RESULT \|)+$", re.M)
        s = pat.sub(lambda mt: f"| {label} | {mt.group(1)} | " + " | ".join(row) + " |", s, count=1)
        dpat = re.compile(rf"^\| {re.escape(label)} \|( RESULT \|)+$", re.M)
        s = dpat.sub(lambda mt: f"| {label} | " + " | ".join(drow) + " |", s, count=1)
    s = s.replace("\nZebraLogic mc by grid size (Sev-2b): RESULT.\n", "\n")

    # claim sentence
    named = []
    for name in REASONING:
        verdicts = [diff_cache[(name, k)][1] > 0 for k in comps if (name, k) in diff_cache]
        if verdicts and all(verdicts):
            named.append(name)
    for name in ("jevbench", "td"):
        need = [k for k in ("decider", "strands") if k in comps]
        ok = need and all((name, k) in diff_cache and diff_cache[(name, k)][1] > 0 for k in need)
        if ok and name == "jevbench":
            ok = all(diff_cache[(name, k)][0] * diff_cache[(name, k)][3] >= 4 for k in need)
        if ok:
            named.append(name)
    label_of = {v: k for k, v in ROWS.items()}
    def join(xs: list[str]) -> str:
        return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
    ran = join([PEER_NAMES[k] for k in comps])
    if named:
        claim = (f"**Claim.** On {join([label_of[n] for n in named])}, {args.name} is ahead of {ran} (paired bootstrap, "
                 "95% interval above 0 against each" + ("; on JevBench public also by at least 4 tasks" if "jevbench"
                                                          in named else "") + ").")
        notes = [t for n, t in (("diag", "K&K diag is in-family"), ("xkk", "xkk is near-distribution"),
                                ("lsat_ar", "LSAT-AR has n = 61")) if n in named]
        if notes:
            claim += " " + "; ".join(notes) + "."
    else:
        claim = f"**Claim.** {args.name} is not ahead of every comparator ({ran}) on any registered set."
    claim += " Single training seed; the intervals cover item sampling only."
    s = s.replace("CLAIM_SENTENCE", claim)

    # per-type temperatures, fitted on the candidate's selection-set predictions
    fit_rows = [r for n in ("val", "v4heldout", "claims_ho") for r in cal_rows(cand, n)]
    params, counts = {}, {}
    for t in ("choice", "score", "noul"):
        sub = [r for r in fit_rows if r["type"] == t]
        counts[t] = len(sub)
        if len(sub) >= 30:
            params[t] = fit(sub, with_bias=t == "noul")
    T = {t: f"{params[t][0]:.2f}" if t in params else "1.00" for t in ("choice", "score", "noul")}
    b = f"{params['noul'][1]:+.2f}" if "noul" in params else "+0.00"
    s = re.sub(r"^\| choice \| (.+?) \| [\d,]+ \| RESULT \| – \|$",
               lambda m: f"| choice | {m.group(1)} | {counts['choice']:,} | {T['choice']} | – |", s, count=1, flags=re.M)
    s = re.sub(r"^\| score \| (.+?) \| [\d,]+ \| RESULT \| – \|$",
               lambda m: f"| score | {m.group(1)} | {counts['score']:,} | {T['score']} | – |", s, count=1, flags=re.M)
    s = re.sub(r"^\| yes/no \| (.+?) \| [\d,]+ \| RESULT \| RESULT \|$",
               lambda m: f"| yes/no | {m.group(1)} | {counts['noul']:,} | {T['noul']} | {b} |", s, count=1, flags=re.M)
    s = s.replace('temperature=RESULT, yes_letter="B", yes_bias=RESULT', f'temperature={T["noul"]}, yes_letter="B", '
                  f'yes_bias={b}')
    s = s.replace("decide(model, tok, choice, temperature=RESULT)", f"decide(model, tok, choice, temperature={T['choice']})")
    s = s.replace("decide(model, tok, score, temperature=RESULT)", f"decide(model, tok, score, temperature={T['score']})")
    s = s.replace("The score T rests on 97 items.", f"The score T rests on {counts['score']} items.")

    # calibration table
    jb = cal_rows(cand, "jevbench")
    td = [r for r in cal_rows(cand, "td") if r["type"] == "noul"]
    s = s.replace("| JevBench public ECE | RESULT | RESULT |",
                  f"| JevBench public ECE | {metrics(jb, {})['ece']:.3f} | {metrics(jb, params)['ece']:.3f} |")
    s = s.replace("| TD yes/no accuracy | RESULT | RESULT |",
                  f"| TD yes/no accuracy | {metrics(td, {})['acc']:.3f} | {metrics(td, params)['acc']:.3f} |")
    cp = preds(cand, "claims_ho")
    items = {json.loads(l)["item_id"]: json.loads(l) for l in open(ROOT / EVAL_FILES["claims_ho"])}
    raw, cal = [], []
    Tn, bn = params.get("noul", (1.0, 0.0))
    import math

    def calp(p: dict, it: dict) -> float:
        names = it.get("label_names") or {}
        yes = next(L for L, nm in names.items() if str(nm).lower() in ("true", "yes"))
        z = [math.log(max(x, 1e-12)) / Tn for x in p["probs"]]
        z[p["labels"].index(yes)] += bn
        m = max(z)
        e = [math.exp(x - m) for x in z]
        return e[p["labels"].index(yes)] / sum(e)
    for i, it in items.items():
        if re.search(r"-(p|a)$", i) and f"{i}n" in items and i in cp and f"{i}n" in cp:
            raw.append(yes_prob(cp[i], it) + yes_prob(cp[f"{i}n"], items[f"{i}n"]))
            cal.append(calp(cp[i], it) + calp(cp[f"{i}n"], items[f"{i}n"]))
    if raw:
        s = s.replace("| claims_ho: mean P(Yes \\| claim) + P(Yes \\| negation) (ideal 1) | RESULT | RESULT",
                      f"| claims_ho: mean P(Yes \\| claim) + P(Yes \\| negation) (ideal 1) | {sum(raw) / len(raw):.3f} "
                      f"| {sum(cal) / len(cal):.3f}")

    # arm, alpha, GPU time
    name, node_format, recipe = ARMS[args.arm]
    s = s.replace("ARM_NAME", name).replace("NODE_FORMAT", node_format)
    s = s.replace(" ARM_RECIPE", (" " + recipe) if recipe else "")
    s = s.replace("alpha = RESULT", f"alpha = {args.alpha}")
    s = s.replace("GPU time: about RESULT h.", f"GPU time: about {args.gpu_hours} h.")

    # owner decisions
    if args.repo:
        s = s.replace("TODO-CONFIRM/Sev-2b", args.repo)
    if args.licence:
        s = re.sub(r"^license: .*$", f"license: {args.licence}", s, count=1, flags=re.M)
        s = s.replace("| Weights licence | CC BY-NC 4.0 (TODO-CONFIRM) |", "| Weights licence | CC BY-NC 4.0 |")
        s = s.replace("- Weights: CC BY-NC 4.0 (TODO-CONFIRM).", "- Weights: CC BY-NC 4.0.")
    if args.author:
        s = s.replace("title  = {Sev-2b: TODO-CONFIRM}", "title  = {Sev-2b: a one-pass decision model trained with Tree "
                      "Forcing}").replace("author = {TODO-CONFIRM}", f"author = {{{args.author}}}")
    if args.infer_check:
        s = re.sub(r"`sev_infer\.py` reproduces the evaluation probabilities: TODO-CONFIRM \(check on a few\n"
                   r"hundred evaluation items against the run's predictions\)\.", args.infer_check, s)
    Path(args.out).write_text(s)
    left = sorted(set(re.findall(r"TODO-CONFIRM[^\n]{0,40}|RESULT|ARM_\w+|NODE_FORMAT|CLAIM_SENTENCE", s)))
    print(f"written {args.out}; claim: {claim}")
    print(f"temperatures: {params} on {counts}")
    print(f"placeholders left ({len(left)}): {left}")


if __name__ == "__main__":
    main()
