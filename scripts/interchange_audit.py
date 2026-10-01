"""Interchange-intervention audit of a decision model on the rule data's third-outcome pairs (GPU job).

  --check   X0 sanity checks only:
            - an identity patch leaves the letter logits unchanged;
            - full replacement of the readout state copies the source answer more often at later layers;
            - DAS gradients reach the subspace and no model weight;
            - the time of one DAS epoch.
  default   for each layer and each rule-variable group (a:1 ... a:5; --groups), fit a rank-k DAS subspace on half
            of the base cases and report interchange-intervention accuracy (IIA) on the other half for das,
            random-subspace, norm-matched-noise and full-state exchanges, plus copy rates. Pairs are restricted to
            those whose base and source the unpatched model answers correctly (--all-pairs keeps every pair; then the
            target is still the program's counterfactual outcome).

  python scripts/interchange_audit.py --model runs/G3-rule-...-s31/model --out runs/audit-G3-s31 --layers 8,12,16,20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch  # noqa: E402

from cotdistill import interchange as ic  # noqa: E402
from cotdistill.student import evaluate, load_model  # noqa: E402


def read(path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def argmax(p: list[float]) -> int:
    return max(range(len(p)), key=p.__getitem__)


def build_pairs(args, model, tok) -> list[dict]:
    base = {r["item_id"]: r for r in read(ROOT / args.base)}
    src = {r["item_id"]: r for r in read(ROOT / args.src)}
    pairs = [p for p in read(ROOT / args.pairs) if p["type"] == "third" and p["base_id"] in base and p["other_id"] in src]
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    need = sorted({p["base_id"] for p in pairs}), sorted({p["other_id"] for p in pairs})
    _, bp = evaluate(model, tok, [base[i] for i in need[0]], args.max_len, args.bs, {})
    _, sp = evaluate(model, tok, [src[i] for i in need[1]], args.max_len, args.bs, {})
    bpred = {p["item_id"]: argmax(p["probs"]) for p in bp}
    spred = {p["item_id"]: argmax(p["probs"]) for p in sp}
    out = []
    for p in pairs:
        b, s = base[p["base_id"]], src[p["other_id"]]
        rec = {"pair_id": p["pair_id"], "base_id": p["base_id"], "var": p["var"], "rule_size": p["rule_size"],
               "base_prompt": b["prompt"], "source_prompt": s["prompt"], "labels": b["labels"],
               "target": b["label_order"].index(p["target_out"]),
               "base_pred": bpred[p["base_id"]], "source_pred": b["label_order"].index(s["label_order"][spred[p["other_id"]]]),
               "base_ok": bpred[p["base_id"]] == b["label_order"].index(b["gold"]),
               "source_ok": spred[p["other_id"]] == s["label_order"].index(s["gold"])}
        out.append(rec)
    return out


def split(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Half of the base cases fit the subspace, the other half test it (no base case in both)."""
    tr, te = [], []
    for p in pairs:
        (tr if int(hashlib.md5(p["base_id"].encode()).hexdigest(), 16) % 2 == 0 else te).append(p)
    return tr, te


def check(args, model, tok, pairs: list[dict], layers: list[int]) -> dict:
    out = {}
    sub = pairs[:16]
    cache: dict = {}
    dev = next(model.parameters()).device
    with torch.no_grad():
        b = ic._batch(tok, [p["base_prompt"] for p in sub], [p["labels"] for p in sub], args.max_len, dev)
        ref = ic._letter_logits(model, tok, b, [p["labels"] for p in sub], cache)
        z = ic.patched_logits(model, tok, [p["base_prompt"] for p in sub], [p["labels"] for p in sub], layers[0],
                              lambda h: h, args.max_len, cache)
    out["identity_max_abs_diff"] = max(float((a - c).abs().max()) for a, c in zip(ref, z))
    src = ic.site_states(model, tok, [p["source_prompt"] for p in pairs[:64]], [p["labels"] for p in pairs[:64]],
                         layers, args.max_len, args.bs)
    out["full_patch"] = {}
    for j, L in enumerate(layers):
        r = ic.iia(model, tok, pairs[:64], src[:, j], L, "full", max_len=args.max_len, bs=args.bs)
        out["full_patch"][L] = r
    for prm in model.parameters():               # as in fit_das: only the subspace may receive gradients
        prm.requires_grad_(False)
    t0 = time.time()
    s = ic.Subspace(src.shape[-1], 1).to(dev)
    zs = ic.patched_logits(model, tok, [p["base_prompt"] for p in pairs[:4]], [p["labels"] for p in pairs[:4]],
                           layers[-1], lambda hb: s.exchange(hb.float(), src[:4, -1].to(dev).float()), args.max_len, cache)
    loss = sum(torch.nn.functional.cross_entropy(z[None], torch.tensor([p["target"]], device=z.device))
               for z, p in zip(zs, pairs[:4]))
    loss.backward()
    out["das_grad_norm"] = float(sum(p.grad.norm() for p in s.parameters() if p.grad is not None))
    out["model_grads"] = sum(p.grad is not None for p in model.parameters())
    ic.fit_das(model, tok, pairs[:64], src[:64, -1], layers[-1], k=1, epochs=1, bs=args.bs, max_len=args.max_len)
    out["das_epoch_seconds_per_64_pairs"] = round(time.time() - t0, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default="data/rule/pairs_ho.jsonl")
    ap.add_argument("--base", default="data/eval/rule_ho.jsonl")
    ap.add_argument("--src", default="data/eval/rule_ho_src.jsonl")
    ap.add_argument("--layers", default="4,8,12,16,20")
    ap.add_argument("--groups", default="a:1,a:2,a:3,a:4,a:5")
    ap.add_argument("--min-rule-size", type=int, default=2, help="primary: rule variables of multi-condition rules")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--all-pairs", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import transformers
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    layers = [int(x) for x in args.layers.split(",")]
    pairs = build_pairs(args, model, tok)
    print(f"{len(pairs)} third-outcome pairs; base right {sum(p['base_ok'] for p in pairs)}, "
          f"both right {sum(p['base_ok'] and p['source_ok'] for p in pairs)}", flush=True)
    if args.check:
        res = check(args, model, tok, pairs, layers)
        (out / "x0_check.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return
    results = {"args": vars(args), "groups": {}}
    for g in args.groups.split(","):
        sel = [p for p in pairs if p["var"] == g and (p["rule_size"] or 0) >= args.min_rule_size
               and (args.all_pairs or (p["base_ok"] and p["source_ok"]))]
        tr, te = split(sel)
        if len(tr) < 20 or len(te) < 20:
            results["groups"][g] = {"skipped": f"train {len(tr)}, test {len(te)}"}
            continue
        src_tr = ic.site_states(model, tok, [p["source_prompt"] for p in tr], [p["labels"] for p in tr], layers,
                                args.max_len, args.bs)
        src_te = ic.site_states(model, tok, [p["source_prompt"] for p in te], [p["labels"] for p in te], layers,
                                args.max_len, args.bs)
        rows = {}
        for j, L in enumerate(layers):
            sub = ic.fit_das(model, tok, tr, src_tr[:, j], L, k=args.k, epochs=args.epochs, bs=args.bs,
                             max_len=args.max_len)
            rnd = ic.random_subspace(src_te.shape[-1], args.k, seed=L).to(next(model.parameters()).device)
            rows[L] = {m: ic.iia(model, tok, te, src_te[:, j], L, m, sub=s, max_len=args.max_len, bs=args.bs)
                       for m, s in (("das", sub), ("random", rnd), ("noise", sub), ("full", None))}
            print(g, L, json.dumps({m: round(v["iia"], 3) for m, v in rows[L].items()}), flush=True)
        results["groups"][g] = {"n_train": len(tr), "n_test": len(te), "layers": rows}
        (out / "audit.json").write_text(json.dumps(results, indent=1))
    (out / "audit.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
