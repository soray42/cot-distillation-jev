"""Parent-sensitivity evaluation (evaluation only): does a model use the stated results of a node's parents?

Every eligible tree node of held-out items (full_tree_plan: the node has parents, every parent has a confident result,
and depth-matched non-parent replacements exist) is asked in five yes/no views that differ only in the stated results:
  true    the parents' results            flip_true    the same, with ONE random parent's answer negated
  matched as many matched non-parents     flip_matched the same, with ONE random matched result negated
  plain   no stated results
The yes/no order is drawn once per node and shared by its views, so the views differ in the stated block only.
Target: program truth when known, else the teacher's CoT-conditioned answer (confident nodes only, as the stated
results are). LO(c) = log-odds toward the target in view c. Unit of analysis: items (node values averaged within an
item; 95% bootstrap over items).
  use       = LO(true) - LO(matched)          what the right results add over as many other results
  drop_par  = LO(true) - LO(flip_true)        how far one negated parent pulls the answer away from the target
  drop_np   = LO(matched) - LO(flip_matched)  the same for one negated non-parent (control)
  specific  = drop_par - drop_np;  S_par, S_np = the unsigned shifts
Negating a parent does not always change the node's truth (a conjunction that is already false stays false), so
drop_par is also reported by stratum: rule items, where the counterfactual truth is computed from item["rules"]
(cf_changes: the negation flips the node's truth, a model that uses its parents should show drop_par > 0;
cf_invariant: it does not, drop_par should stay near 0), and single-parent nodes (npar1). The plan uses the
post-audit matching for every set (tree_fix: no ancestors among the matched results), also for the older v4t data.
A model trained with true parents stated (TF) should show use > 0 and specific > 0; final-only training (A2) and the
base give the reference levels. Exploratory: not part of the registered prereg_T / prereg_M endpoints.

  python scripts/parent_sens_eval.py --model runs/TF-v4t-.../model --data v3f=data/student_v3f/val.jsonl \
      v4t=data/student_v4t/val.jsonl rule=data/student_rule/val.jsonl --out runs/ps-TF-v4t
  python scripts/parent_sens_eval.py --dry-run --data v3f=data/student_v3f/val.jsonl --out /tmp/ps   # views only
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.student import FACTS_HEADER, SUBQ_TEMPLATE, Example  # noqa: E402
from cotdistill.treeplan import _fact_line, _node_p, full_tree_plan  # noqa: E402

CONDS = ("true", "flip_true", "matched", "flip_matched", "plain")


def _flip(line: str) -> str:
    return line[:-3] + "No" if line.endswith("A: Yes") else line[:-2] + "Yes"


def _cf_rule(item: dict, sq: dict, by: dict, flipped: str) -> bool | None:
    """Rule items: the node's truth when parent `flipped` is negated (node "a:k" = every condition of rule k holds,
    parents "p:<predicate>"), else None."""
    var = sq.get("var") or ""
    if not var.startswith("a:") or "rules" not in item:
        return None
    try:
        conds = dict(item["rules"][int(var[2:]) - 1]["conds"])
    except (ValueError, IndexError, KeyError, TypeError):
        return None
    vals = {}
    for j in sq.get("depends_on") or []:
        pv = by[j].get("var") or ""
        if not pv.startswith("p:") or by[j].get("truth") is None:
            return None
        vals[pv[2:]] = bool(by[j]["truth"]) != (j == flipped)
    if set(vals) != set(conds):
        return None
    return all(vals[v] == conds[v] for v in conds)


def build_views(items: list[dict], *, cap: int, seed: int, limit: int = 0) -> tuple[list[Example], list[dict]]:
    """Five views per eligible confident node; meta rows carry the item, node, condition, target and strata."""
    rng = random.Random(seed)
    exs, meta = [], []
    for it in items[:limit or None]:
        by = {sq["id"]: sq for sq in it.get("subqs", []) if "id" in sq}
        for e in full_tree_plan(dict(it, tree_fix=True), cap=cap, rng=rng, subq_target="cot"):
            if not e["parents"] or not e["matched"]:
                continue
            sq = e["node"]
            p = _node_p(sq, "truth")
            if p is None or 0.3 < p < 0.7:
                continue
            fp, fm = rng.randrange(len(e["parents"])), rng.randrange(len(e["matched"]))
            par = [(_fact_line(by[j]), k == fp) for k, j in enumerate(e["parents"])]
            mat = [(_fact_line(by[j]), k == fm) for k, j in enumerate(e["matched"])]
            rng.shuffle(par)
            rng.shuffle(mat)
            facts = {"true": [x for x, _ in par], "flip_true": [_flip(x) if f else x for x, f in par],
                     "matched": [x for x, _ in mat], "flip_matched": [_flip(x) if f else x for x, f in mat],
                     "plain": []}
            cf = _cf_rule(it, sq, by, e["parents"][fp])
            yes_first = rng.random() < 0.5
            a, b = ("Yes", "No") if yes_first else ("No", "Yes")
            for c in CONDS:
                lines = facts[c]
                problem = it["prompt"] + (f"\n\n{FACTS_HEADER}\n" + "\n".join(lines) if lines else "")
                exs.append(Example(SUBQ_TEMPLATE.format(problem=problem, question=sq["question"], a=a, b=b),
                                   ["A", "B"], [0.5, 0.5], 1.0, "subq", it["item_id"]))
                meta.append({"item_id": it["item_id"], "source": it.get("source"), "node": sq["id"], "cond": c,
                             "target": p >= 0.5, "truth_known": sq.get("truth") is not None,
                             "n_par": len(e["parents"]), "cf_changes": None if cf is None else cf != (p >= 0.5),
                             "n_facts": len(lines), "yes_first": yes_first})
    return exs, meta


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def summarize(preds: list[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    """Item-level means of the per-node contrasts, with 95% bootstrap intervals over items; overall, by source and by
    stratum (npar1, cf_changes, cf_invariant)."""
    nodes: dict[tuple, dict] = {}
    for r in preds:
        n = nodes.setdefault((r["item_id"], r["node"]), {"item": r["item_id"], "src": r["source"], "t": r["target"],
                                                         "npar": r.get("n_par"), "cf": r.get("cf_changes")})
        n[r["cond"]] = _logit(r["p_yes"])
    rows: dict[str, list[dict]] = {}
    for n in nodes.values():
        sgn = 1.0 if n["t"] else -1.0
        lo = {c: sgn * n[c] for c in CONDS}               # log-odds toward the target
        d = {f"lo_{c}": lo[c] for c in ("true", "matched", "plain")}
        d.update({f"acc_{c}": float(lo[c] > 0) for c in ("true", "matched", "plain")})
        d["use"] = lo["true"] - lo["matched"]
        d["true_vs_plain"] = lo["true"] - lo["plain"]
        d["drop_par"] = lo["true"] - lo["flip_true"]
        d["drop_np"] = lo["matched"] - lo["flip_matched"]
        d["specific"] = d["drop_par"] - d["drop_np"]
        d["S_par"] = abs(n["true"] - n["flip_true"])
        d["S_np"] = abs(n["matched"] - n["flip_matched"])
        rows.setdefault(n["item"], []).append(dict(d, _src=n["src"], _npar=n["npar"], _cf=n["cf"]))
    keys = [k for k in next(iter(rows.values()))[0] if not k.startswith("_")] if rows else []

    def block(sel):
        per = {i: [x for x in ns if sel(x)] for i, ns in rows.items()}
        per = {i: ns for i, ns in per.items() if ns}
        ids = sorted(per)
        if not ids:
            return {}
        means = {i: {k: sum(x[k] for x in per[i]) / len(per[i]) for k in keys} for i in ids}

        def avg(sample):
            return {k: sum(means[i][k] for i in sample) / len(sample) for k in keys}
        est = avg(ids)
        rng = random.Random(seed)
        boots = [avg([rng.choice(ids) for _ in ids]) for _ in range(n_boot)]
        res = {"n_items": len(ids), "n_nodes": sum(len(per[i]) for i in ids)}
        for k in keys:
            bs = sorted(b[k] for b in boots)
            res[k] = [round(est[k], 4), round(bs[int(0.025 * n_boot)], 4), round(bs[int(0.975 * n_boot) - 1], 4)]
        return res

    out = {"all": block(lambda x: True), "npar1": block(lambda x: x["_npar"] == 1),
           "cf_changes": block(lambda x: x["_cf"] is True), "cf_invariant": block(lambda x: x["_cf"] is False)}
    for s in sorted({x["_src"] for ns in rows.values() for x in ns}, key=str):
        out[f"src/{s}"] = block(lambda x, s=s: x["_src"] == s)
    return {k: v for k, v in out.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--data", nargs="+", required=True, help="name=student val.jsonl (items with tree nodes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=10, help="at most this many nodes per item (as in training)")
    ap.add_argument("--limit", type=int, default=0, help="first N items per set (0 = all; smoke runs)")
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="build the views, write counts and samples, no model")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = []
    for kv in args.data:
        name, path = kv.split("=", 1)
        items = [json.loads(l) for l in open(path) if l.strip()]
        sets.append((name, *build_views(items, cap=args.cap, seed=args.seed, limit=args.limit)))
        print(name, "items", len(items), "views", len(sets[-1][1]), "nodes", len(sets[-1][1]) // len(CONDS),
              flush=True)
    if args.dry_run:
        with open(out / "ps_dry_run.txt", "w") as f:
            for name, exs, meta in sets:
                for e, m in list(zip(exs, meta))[:len(CONDS)]:
                    f.write(f"### {name} {m}\n{e.text}\n\n")
        print("dry run: samples in", out / "ps_dry_run.txt")
        return

    import torch
    import transformers
    from cotdistill.student import label_logits, load_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.model, torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    cache, results = {}, {"args": vars(args), "eval": {}}
    for name, exs, meta in sets:
        order = sorted(range(len(exs)), key=lambda i: len(exs[i].text))
        p_yes = [0.0] * len(exs)
        with torch.no_grad():
            for s in range(0, len(order), args.bs):
                idx = order[s:s + args.bs]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                    zs = label_logits(model, tok, [exs[i] for i in idx], args.max_len, cache)
                for i, z in zip(idx, zs):
                    pa = torch.softmax(z, -1)[0].item()
                    p_yes[i] = pa if meta[i]["yes_first"] else 1 - pa
        preds = [dict(m, p_yes=p) for m, p in zip(meta, p_yes)]
        with open(out / f"preds_ps_{name}.jsonl", "w") as f:
            for r in preds:
                f.write(json.dumps(r) + "\n")
        results["eval"][name] = summarize(preds, seed=args.seed) if preds else {}
        a = results["eval"][name].get("all", {})
        print(f"ps_{name}", json.dumps({k: a.get(k) for k in ("n_items", "n_nodes", "acc_true", "acc_matched",
                                                               "acc_plain", "use", "drop_par", "drop_np", "specific")}),
              flush=True)
    (out / "ps_metrics.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
