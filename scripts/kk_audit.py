"""Interchange-intervention audit of a decision model on Knights & Knaves (GPU job; design:
notes/next_round/causal_audit/kk_design.md). Reads data/eval/kk_audit_pairs.jsonl only (scripts/build_kk_audit_pairs.py
built and solver-checked it on CPU; the solver is not needed here).

Two versions. --preset v2 (default; notes/prereg_M.md, K&K addendum, post-hoc amendment of 2026-10-09) differs from the
registered v1 run (--preset v1, unchanged, described below) in:
  - DAS optimisation: per condition, layer and rank, one subspace per slot, all slots fitted together for at least
    --das-steps (500) Adam steps at --lr 1e-2 (v1: 3 epochs at 1e-3, 9-30 steps, flat loss), early stop on held-out
    train programs (--val-frac 0.2; patience 200, at most 1500 steps), best held-out parameters kept; the loss curve,
    initial / best held-out loss, relative drop and a below-chance flag of every fit go to audit.json ("fits").
  - Sites: readout only by default; layers from a screen over dense late candidates (13..20 of 24, 17..28 of 32): kept
    if the full exchange of the readout state copies the source answer on >= .5 of stmt pairs (v1: 4,8,...,20, where
    L4-12 were inert). Ranks k in {1, 4, 8} (--k).
  - Positive-control gate: stmt:source@last (target = the source answer) is fitted first; DAS IIA >= .50 on both-correct
    test pairs at some cell ("_gate" line). E3 (stmt:source@stmt, an inert site) is dropped from the defaults.
  - Scoring: test sets "both" (unpatched base and source right; primary) and "all" (secondary), each with the
    coincidence rate (unpatched base answer == target); inv pairs scored against the model's own unpatched answer.
  - Precision: --dtype fp32 = fp32 weights with autocast and TF32 off (v1's fp32 ran its matmuls under bf16 autocast:
    --dtype fp32-autocast). Unpatched predictions come from the same forward path as the patched ones.
  - Readout patches run as single-token decodes over cached prefixes (interchange.PrefixStore; --no-prefix-cache for
    the full-forward hook); every prefix is checked against the unpatched forward ("decode_fidelity").
  - Linear probes (role at name / statement tokens and per slot at the readout, layers 16/20) with control tasks.

Conditions (name = family/subset:target@position; --conds picks them):
  role:local@last     role(X), every role pair; target = base with only X flipped (answer component)
  role/mid:prop@last  role(X) with proof descendants; target = do(role(X)) propagated through the proof
  role/mid:local@last the same pairs, target = local flip (the competing hypothesis; fitted separately)
  role:z@last         spurious-target control: target = base with Z flipped, Z's role equal in base and source
  role/mid:zprop@last the plausibility-matched floor of prop: target = do(role(Z)) propagated (same Z), mid pairs
  role/mid:z@last     z fitted on the mid pairs (the matched floor of role/mid:local in the E2 difference-in-differences)
  stmt:source@stmt    content of X's own statement at the last token of X's statement line; target = source answer
  stmt:source@last    the same pairs at the readout
  role:local@stmt / role:local@name  secondary sites (last token of X's statement line / of X's name at its line start)
Each condition and layer: fit a rank-k DAS subspace on the train programs (md5(program) even) -- at the readout one
per slot (X's index in the person list; Z's for z/zprop), since one pooled rank-k exchange cannot pick out which
person to flip (see slot_key; --pool-slots restores pooling) -- and test on the others with
das, random-subspace, norm-matched-noise and full-state exchange, on two test sets: the jointly-correct test pairs
(keys das/random/noise/full) and all test pairs (key all_pairs; das_all_pairs = all_pairs.das, the set the registered
endpoints use). Both sets are scored from one patched pass over their union. Shares of patched answers landing on
base / source / local / prop / z options; DAS test IIA by n_people, how X is derived, subfam and chain depth. The inv
family (same-value sources: the solution changes, X's role does not) is scored with the subspaces of role:local and of
role/mid:prop (inv/mid pairs, whose options hold X's prop) at the same layer and site: the patched answer should stay
the base answer (a "move off base" direction would carry it to the most plausible option instead, often prop). At the stmt/name sites, pairs whose source and
base token prefixes are identical up to the site (the edited line comes after it) are dropped: the exchange is a no-op
there (n_noop in the results).

One line per model/layer/condition: "KKAUDIT {json}" on stdout and in <out>/summary.jsonl; everything in audit.json;
per-pair patched answers in <out>/preds.jsonl (a "pairs" line per condition, then one "preds" line per layer x method,
aligned with it) for paired bootstraps over programs and paired arm contrasts.

  --check   sanity checks only (identity patches at the readout and at the statement site, token positions, full-state
            exchange copy rates by layer, gradients reach the subspace only and the weights are bit-identical after a
            DAS fit, bf16 fidelity of a rank-1 exchange, peak GPU memory, timing and a GPU-time extrapolation).

--layers auto (v1 default) = every 4th layer from 4 to n_layers - 4 (the input of the layer after each full-attention
layer): 4,8,12,16,20 for the 24-layer 2B, 4,8,...,28 for the 32-layer 4B.

v2 outputs: preds.jsonl lines carry "k" (pairs lines: in_both / in_all), summary lines have "both" and "all" blocks
(per method: iia, coinc, copy and landing shares) and a "fit" block; "_gate" and "_probes" lines; audit.json also holds
the layer screen, the fits with their curves and the decode fidelity. --check (v2) adds the decode-vs-hook comparison
at every candidate layer, the fp32 forward check, the positive-control mini-fit (stmt:source@last, best-copy layer,
largest k, --check-fit-steps), bf16 exchange fidelity computed in bf16 whatever the model dtype, and measured timings.

  python scripts/kk_audit.py --model runs/TFL-.../model --name TFL-s12 --out runs/kkaudit_v2/TFL-s12
  python scripts/kk_audit.py --preset v1 --model runs/TFL-.../model --name TFL-s12 --out runs/kkaudit/TFL-s12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))                 # the code lives next to this script, wherever the data is
if (Path.cwd() / "data/eval").exists() and not (ROOT / "data/eval").exists():   # HPC: data next to the repo clone
    ROOT = Path.cwd()
import torch  # noqa: E402

from cotdistill import interchange as ic  # noqa: E402
from cotdistill.student import FINAL_TEMPLATE, evaluate, load_model, text_parts  # noqa: E402

DEFAULT_CONDS = ("role:local@last,role/mid:prop@last,role/mid:local@last,role:z@last,role/mid:z@last,"
                 "role/mid:zprop@last,stmt:source@stmt,stmt:source@last")
TARGET_FIELD = {"local": "t_local", "prop": "t_prop", "z": "t_z", "zprop": "t_zprop", "source": "source_gold",
                "base": "base_gold"}
OUTCOMES = ("base_gold", "source_gold", "t_local", "t_prop", "t_z", "t_zprop")
METHODS = ("das", "random", "noise", "full")
INV_METHODS = ("das", "random", "full")


def read(path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def md5(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def token_pos(tok, prompt: str, char: int, max_len: int) -> int:
    """Index of the token covering character `char` of the prompt in the collated sequence (FINAL_TEMPLATE puts the
    prompt first; collate keeps the last max_len tokens, so a truncated prompt is an error)."""
    text = FINAL_TEMPLATE.format(problem=prompt)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    assert ids == tok.encode(text, add_special_tokens=False), "offset tokenisation differs from collate"
    assert len(ids) <= max_len, f"prompt longer than max_len ({len(ids)})"
    hit = [t for t, (a, b) in enumerate(enc["offset_mapping"]) if a <= char < b]
    assert hit, f"no token covers character {char}"
    return hit[0]


def parse_cond(c: str) -> dict:
    fam, rest = c.split(":")
    tgt, site = rest.split("@")
    family, _, subset = fam.partition("/")
    assert family in ("role", "stmt", "inv") and tgt in TARGET_FIELD and site in ("last", "stmt", "name"), c
    return {"name": c, "family": family, "subset": subset or None, "target": tgt, "site": site}


def prepare(args, model, tok, raw: list[dict], pred: dict | None = None) -> list[dict]:
    """Unpatched predictions on every distinct prompt, then one record per pair with token positions. `pred` (prompt ->
    predicted option index) replaces the v1 prediction pass (student.evaluate, bf16 autocast) when given."""
    if pred is None:
        prompts = {}
        for p in raw:
            for k in ("base_prompt", "source_prompt"):
                prompts.setdefault(p[k], {"item_id": f"q{len(prompts)}", "prompt": p[k], "labels": p["labels"],
                                          "gold_label": p["labels"][p["base_gold" if k == "base_prompt"
                                                                          else "source_gold"]]})
        items = list(prompts.values())
        _, preds = evaluate(model, tok, items, args.max_len, args.bs, {})
        pred = {it["prompt"]: max(range(len(pr["probs"])), key=pr["probs"].__getitem__) for it, pr in zip(items, preds)}
    out = []
    for p in raw:
        r = dict(p)
        r["base_pred"], r["source_pred"] = pred[p["base_prompt"]], pred[p["source_prompt"]]
        r["base_ok"], r["source_ok"] = r["base_pred"] == p["base_gold"], r["source_pred"] == p["source_gold"]
        for key, pr, sp in (("base", p["base_prompt"], p["base_spans"]), ("source", p["source_prompt"], p["source_spans"])):
            r[f"{key}_tok"] = {"last": None, "stmt": token_pos(tok, pr, sp["stmt_end"], args.max_len),
                               "name": token_pos(tok, pr, sp["name_end"], args.max_len)}
        # under causal mixing the source state at a site differs from the base state only if the token prefixes up to
        # the site differ (role pairs whose edited line comes after X's line: identical, the exchange is a no-op)
        r["site_moves"] = {"last": True}
        for site in ("stmt", "name"):
            r["site_moves"][site] = (enc(tok, p["base_prompt"])[:r["base_tok"][site] + 1]
                                     != enc(tok, p["source_prompt"])[:r["source_tok"][site] + 1])
        out.append(r)
    return out


_ENC: dict = {}


def enc(tok, prompt: str) -> list[int]:
    if prompt not in _ENC:
        _ENC[prompt] = tok.encode(FINAL_TEMPLATE.format(problem=prompt), add_special_tokens=False)
    return _ENC[prompt]


def select(pairs: list[dict], cond: dict, jointly_correct: bool, noop: list | None = None) -> list[dict]:
    """Pairs of the condition's family/subset with its target defined (and jointly correct if asked), excluding pairs
    where the site cannot move (appended to `noop` if given)."""
    field = TARGET_FIELD[cond["target"]]
    out = []
    for p in pairs:
        if p["family"] != cond["family"] or (cond["subset"] and p["subfam"] != cond["subset"]):
            continue
        if p.get(field) is None or (jointly_correct and not (p["base_ok"] and p["source_ok"])):
            continue
        if not p["site_moves"][cond["site"]]:
            if noop is not None:
                noop.append(p["pair_id"])
            continue
        r = dict(p, target=p[field])
        if cond["site"] != "last":
            r["base_pos"] = p["base_tok"][cond["site"]]
        out.append(r)
    return out


def slot_key(cond: dict, args):
    """The variable index a subspace belongs to. At the readout a role(X) subspace must be specific to X: the exchange
    is h_b + c w with one scalar c per pair, so one pooled w cannot send the same base to a different person's flip
    depending on X. The slot (X's index in `people`, which orders both the statement lines and the option fields) is
    that index; z/zprop conditions use Z's slot. At the stmt/name sites the position already identifies X: pooled.
    --pool-slots pools everywhere (the pre-review behaviour)."""
    if args.pool_slots or cond["site"] != "last":
        return lambda p: 0
    f = "z_slot" if cond["target"] in ("z", "zprop") else "slot"
    return lambda p: p[f]


def split(pairs: list[dict], max_train: int, max_test: int, key=None) -> tuple[list[dict], list[dict]]:
    """Train on programs with an even md5, test on the others (no program on both sides); deterministic caps. With
    key (slot), the cap is filled round-robin over the key's values, each in md5(pair_id) order."""
    def take(half, cap):
        half = sorted(half, key=lambda p: md5(p["pair_id"]))
        if key is None:
            return half[:cap or None]
        groups: dict = {}
        for p in half:
            groups.setdefault(key(p), []).append(p)
        out, qs = [], [groups[k] for k in sorted(groups)]
        for r in range(max((len(q) for q in qs), default=0)):
            out += [q[r] for q in qs if r < len(q)]
        return out[:cap or None]
    return (take([p for p in pairs if md5(p["program"]) % 2 == 0], max_train),
            take([p for p in pairs if md5(p["program"]) % 2 == 1], max_test))


def src_states(args, model, tok, pairs, site: str, layers) -> torch.Tensor:
    pos = None if site == "last" else [p["source_tok"][site] for p in pairs]
    return ic.site_states(model, tok, [p["source_prompt"] for p in pairs], [p["labels"] for p in pairs], layers,
                          args.max_len, args.bs, pos=pos)


def strata(te: list[dict], preds: list[int]) -> dict:
    out: dict = {}
    for p, k in zip(te, preds):
        keys = [f"n{p['n_people']}", f"how/{p['how']}", f"subfam/{p['subfam']}", f"chain/{min(p['chain_depth'], 3)}",
                f"slot/{p.get('slot')}"]
        a = p.get("src_desc_agree")             # mid role pairs: share of X's descendants the source agrees with prop
        if a is not None:
            keys.append("desc_agree/" + ("none" if a == 0 else "all" if a == 1 else "some"))
        if p.get("prop_extra_viol") is not None and p["subfam"] == "mid":
            keys.append("prop_clean/" + ("yes" if p["prop_extra_viol"] == 0 else "no"))
        v = p.get("viol_base") or {}
        if v.get("prop") is not None and p["subfam"] == "mid":
            keys.append("viol_gap/" + ("prop<local" if v["prop"] < v["local"] else
                                       "prop=local" if v["prop"] == v["local"] else "prop>local"))
        for key in keys:
            h = out.setdefault(key, [0, 0])
            h[0] += k == p["target"]
            h[1] += 1
    return {k: {"iia": round(a / b, 4), "n": b} for k, (a, b) in sorted(out.items())}


def score(recs: list[dict], preds: list[int]) -> dict:
    """The statistics of interchange.iia from per-pair patched answers."""
    n = max(1, len(recs))
    out = {"iia": sum(k == p["target"] for p, k in zip(recs, preds)) / n,
           "copy_base": sum(k == p["base_pred"] for p, k in zip(recs, preds)) / n,
           "copy_source": sum(k == p["source_pred"] for p, k in zip(recs, preds)) / n, "n": len(recs)}
    for f in OUTCOMES:
        out[f"to_{f}"] = sum(p.get(f) is not None and k == p[f] for p, k in zip(recs, preds)) / n
    return out


def union(*sets: list[dict]) -> list[dict]:
    seen: dict = {}
    for s in sets:
        for p in s:
            seen.setdefault(p["pair_id"], p)
    return list(seen.values())


def rounded(d: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


PAIR_META = ("pair_id", "program", "n_people", "subfam", "how", "chain_depth", "n_desc", "target", "base_pred",
             "source_pred", "base_ok", "source_ok", "slot", "z_slot", "src_desc_agree", "viol_base",
             "prop_extra_viol") + OUTCOMES                  # option indices: to_* rates per pair


def write_preds(args, cname: str, ev: list[dict], sets: dict, preds: dict) -> None:
    """preds: {layer: {method: [pred per ev pair]}}; sets: {set name: set of pair ids}."""
    with open(Path(args.out) / "preds.jsonl", "a") as f:
        f.write(json.dumps({"kind": "pairs", "model": args.name, "cond": cname,
                            "pairs": [dict({k: p.get(k) for k in PAIR_META},
                                           **{f"in_{s}": p["pair_id"] in ids for s, ids in sets.items()})
                                      for p in ev]}) + "\n")
        for L, by in preds.items():
            for m, ks in by.items():
                f.write(json.dumps({"kind": "preds", "model": args.name, "cond": cname, "layer": L, "method": m,
                                    "preds": ks}) + "\n")


def emit(args, rec: dict) -> None:
    line = json.dumps(rec)
    print("KKAUDIT " + line, flush=True)
    with open(Path(args.out) / "summary.jsonl", "a") as f:
        f.write(line + "\n")


def run(args, model, tok, pairs: list[dict], layers: list[int]) -> dict:
    dev = next(model.parameters()).device
    results: dict = {"args": vars(args), "conds": {}}
    subspaces: dict = {}
    (Path(args.out) / "preds.jsonl").write_text("")

    def patched(ev, states, L, methods, subs, key):
        """{method: patched answers on ev} at layer L (one pass per method over the union of the test sets); subs maps
        each slot (key) to its subspace and every ev pair's slot has one."""
        out = {m: [None] * len(ev) for m in methods}
        rnd = ic.random_subspace(states.shape[-1], args.k, seed=1000 + L).to(dev)
        for g, sub in subs.items():
            idx = [i for i, p in enumerate(ev) if key(p) == g]
            if not idx:
                continue
            for m in methods:
                s = {"das": sub, "noise": sub, "full": None, "random": rnd}[m]
                ks = ic.iia(model, tok, [ev[i] for i in idx], states[idx], L, m, sub=s, max_len=args.max_len,
                            bs=args.bs, seed=args.seed, return_preds=True)["preds"]
                for i, k in zip(idx, ks):
                    out[m][i] = k
        return out

    def fit_slots(tr, states, L, key):
        subs = {}
        for g in sorted({key(p) for p in tr}):
            idx = [i for i, p in enumerate(tr) if key(p) == g]
            subs[g] = ic.fit_das(model, tok, [tr[i] for i in idx], states[idx], L, k=args.k, epochs=args.epochs,
                                 bs=args.bs, max_len=args.max_len, seed=args.seed)
        return subs

    def scored(ev, ids, preds, methods):
        keep = [i for i, p in enumerate(ev) if p["pair_id"] in ids]
        recs = [ev[i] for i in keep]
        return {m: score(recs, [preds[m][i] for i in keep]) for m in methods}, recs, keep

    for cname in args.conds.split(","):
        cond = parse_cond(cname)
        key = slot_key(cond, args)
        noop: list = []
        sel = select(pairs, cond, not args.all_pairs, noop)
        tr0 = split(sel, args.max_train, 0, key)[0]
        n_tr = {g: sum(key(p) == g for p in tr0) for g in sorted({key(p) for p in tr0})}
        slots = {g for g, n in n_tr.items() if n >= args.min_slot_pairs}      # slots with too few train pairs: skipped
        skipped_slots = {str(g): n for g, n in n_tr.items() if g not in slots}
        sel = [p for p in sel if key(p) in slots]
        mte = args.max_test_mid if cond["subset"] == "mid" else args.max_test
        tr, te = split(sel, args.max_train, mte, key)
        te_all = split([p for p in select(pairs, cond, False) if key(p) in slots], 0, mte, key)[1]
        if len(tr) < args.min_pairs or len(te) < args.min_pairs:
            results["conds"][cname] = {"skipped": f"train {len(tr)}, test {len(te)}", "n_noop": len(noop)}
            emit(args, {"model": args.name, "cond": cname, **results["conds"][cname]})
            continue
        t0 = time.time()
        ev = union(te_all, te)
        sets = {"test": {p["pair_id"] for p in te}, "all": {p["pair_id"] for p in te_all}}
        s_tr = src_states(args, model, tok, tr, cond["site"], layers)
        s_ev = src_states(args, model, tok, ev, cond["site"], layers)
        rows, preds = {}, {}
        for j, L in enumerate(layers):
            subspaces[(cname, L)] = fit_slots(tr, s_tr[:, j], L, key)
            preds[L] = patched(ev, s_ev[:, j], L, METHODS, subspaces[(cname, L)], key)
            row, recs, keep = scored(ev, sets["test"], preds[L], METHODS)
            row["das_strata"] = strata(recs, [preds[L]["das"][i] for i in keep])
            row["all_pairs"], recs, keep = scored(ev, sets["all"], preds[L], METHODS)
            row["das_all_pairs"] = row["all_pairs"]["das"]
            row["das_all_strata"] = strata(recs, [preds[L]["das"][i] for i in keep])
            rows[L] = row
            emit(args, {"model": args.name, "cond": cname, "layer": L, "k": args.k, "n_train": len(tr),
                        "n_test": len(te), "n_test_all": len(te_all), "n_noop": len(noop),
                        "n_slots": len(slots), "skipped_slots": skipped_slots,
                        **{m: rounded(row[m]) for m in METHODS},
                        "all_pairs": {m: rounded(v) for m, v in row["all_pairs"].items()},
                        "das_all_pairs": rounded(row["das_all_pairs"]),
                        "das_strata": row["das_strata"], "das_all_strata": row["das_all_strata"]})
        write_preds(args, cname, ev, sets, preds)
        results["conds"][cname] = {"n_train": len(tr), "n_test": len(te), "n_test_all": len(te_all),
                                   "n_noop": len(noop), "n_slots": len(slots), "skipped_slots": skipped_slots,
                                   "seconds": round(time.time() - t0, 1), "layers": rows}
        (Path(args.out) / "audit.json").write_text(json.dumps(results, indent=1))
        # inv (X's role unchanged in the source): the role(X) subspaces fitted to local (all role pairs) and to prop
        # (mid pairs; inv/mid pairs carry X's prop among their options, to_t_prop) should leave the base answer
        if cond["family"] == "role" and (cond["target"], cond["subset"]) in (("local", None), ("prop", "mid")) \
                and not args.no_inv:
            sub_ = cond["subset"]
            inv_cond = {"name": f"inv{'/' + sub_ if sub_ else ''}:base@{cond['site']}", "family": "inv",
                        "subset": sub_, "target": "base", "site": cond["site"]}
            inv = split([p for p in select(pairs, inv_cond, not args.all_pairs) if key(p) in slots], 0,
                        args.max_test, key)[1]
            inv_all = split([p for p in select(pairs, inv_cond, False) if key(p) in slots], 0, args.max_test, key)[1]
            if len(inv) >= args.min_pairs:
                ev = union(inv_all, inv)
                sets = {"test": {p["pair_id"] for p in inv}, "all": {p["pair_id"] for p in inv_all}}
                s_inv = src_states(args, model, tok, ev, cond["site"], layers)
                rows, preds = {}, {}
                for j, L in enumerate(layers):
                    preds[L] = patched(ev, s_inv[:, j], L, INV_METHODS, subspaces[(cname, L)], key)
                    rows[L], _, _ = scored(ev, sets["test"], preds[L], INV_METHODS)
                    rows[L]["all_pairs"], _, _ = scored(ev, sets["all"], preds[L], INV_METHODS)
                    emit(args, {"model": args.name, "cond": inv_cond["name"], "subspace_from": cname, "layer": L,
                                "n_test": len(inv), "n_test_all": len(inv_all),
                                **{m: rounded(rows[L][m]) for m in INV_METHODS},
                                "all_pairs": {m: rounded(v) for m, v in rows[L]["all_pairs"].items()}})
                write_preds(args, inv_cond["name"], ev, sets, preds)
                results["conds"][inv_cond["name"]] = {"n_test": len(inv), "n_test_all": len(inv_all),
                                                      "subspace_from": cname, "layers": rows}
        (Path(args.out) / "audit.json").write_text(json.dumps(results, indent=1))
    return results


def check(args, model, tok, pairs: list[dict], layers: list[int]) -> dict:
    out: dict = {}
    dev = next(model.parameters()).device
    cache: dict = {}
    sub = [p for p in pairs if p["family"] == "role"][:16] or pairs[:16]
    bp, lb = [p["base_prompt"] for p in sub], [p["labels"] for p in sub]
    with torch.no_grad():
        b = ic._batch(tok, bp, lb, args.max_len, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            ref = ic._letter_logits(model, tok, b, lb, cache)
        for site in ("last", "stmt"):
            pos = None if site == "last" else [p["base_tok"]["stmt"] for p in sub]
            for L in (layers[0], layers[-1]):
                z = ic.patched_logits(model, tok, bp, lb, L, lambda h: h, args.max_len, cache, pos=pos)
                out[f"identity_max_abs_diff/{site}/L{L}"] = max(float((a - c).abs().max()) for a, c in zip(ref, z))
    # token positions: the statement site must be the closing quote of X's line, the name site inside X's name
    ok_stmt = ok_name = 0
    for p in pairs[:64]:
        ids = tok.encode(FINAL_TEMPLATE.format(problem=p["base_prompt"]), add_special_tokens=False)
        ok_stmt += '"' in tok.decode([ids[p["base_tok"]["stmt"]]])
        ok_name += tok.decode([ids[p["base_tok"]["name"]]]).strip() in p["person"]
    out["stmt_site_is_quote"] = ok_stmt / min(64, len(pairs))
    out["name_site_in_name"] = ok_name / min(64, len(pairs))
    out["n_pairs"] = len(pairs)
    out["base_acc"] = sum(p["base_ok"] for p in pairs) / len(pairs)
    out["jointly_correct"] = {f: sum(p["base_ok"] and p["source_ok"] for p in pairs if p["family"] == f)
                              for f in ("role", "inv", "stmt")}
    # full-state exchange: copies of the source answer should rise with depth at the readout
    full = pairs[:64]
    t0 = time.time()
    src = ic.site_states(model, tok, [p["source_prompt"] for p in full], [p["labels"] for p in full], layers,
                         args.max_len, args.bs)
    t_fwd = time.time() - t0
    out["full_patch_last"] = {L: {k: round(v, 3) for k, v in
                                  ic.iia(model, tok, [dict(p, target=p["source_gold"]) for p in full], src[:, j], L,
                                         "full", max_len=args.max_len, bs=args.bs).items()}
                              for j, L in enumerate(layers)}
    st = [dict(p, target=p["source_gold"], base_pos=p["base_tok"]["stmt"]) for p in pairs if p["family"] == "stmt"][:64]
    if st:
        s2 = ic.site_states(model, tok, [p["source_prompt"] for p in st], [p["labels"] for p in st], layers,
                            args.max_len, args.bs, pos=[p["source_tok"]["stmt"] for p in st])
        out["full_patch_stmt"] = {L: {k: round(v, 3) for k, v in ic.iia(model, tok, st, s2[:, j], L, "full",
                                                                       max_len=args.max_len, bs=args.bs).items()}
                                  for j, L in enumerate(layers)}
    # bf16 fidelity of a rank-1 exchange: the patch is cast back to the residual dtype, so a small exchanged component
    # can be partly rounded away against large residual coordinates (realised / intended change, cosine)
    with torch.no_grad():
        base = ic.site_states(model, tok, [p["base_prompt"] for p in full], [p["labels"] for p in full], layers,
                              args.max_len, args.bs)
        rnd = ic.random_subspace(src.shape[-1], 1, seed=7)
        dt = next(model.parameters()).dtype
        out["exchange_fidelity"] = {}
        for j, L in enumerate(layers):
            hb, hs = base[:, j], src[:, j]
            want = rnd.exchange(hb, hs) - hb
            got = rnd.exchange(hb.to(dt), hs.to(dt)).to(dt).float() - hb.to(dt).float()
            live = want.norm(dim=-1) > 1e-6 * hb.norm(dim=-1)        # base == source state (e.g. embeddings): skip
            if not bool(live.any()):
                out["exchange_fidelity"][L] = None
                continue
            got, want = got[live], want[live]
            ratio = got.norm(dim=-1) / want.norm(dim=-1)
            cos = torch.nn.functional.cosine_similarity(got, want, dim=-1)
            out["exchange_fidelity"][L] = {"norm_ratio": round(float(ratio.median()), 4),
                                           "cos_median": round(float(cos.median()), 4),
                                           "cos_min": round(float(cos.min()), 4), "n": int(live.sum())}
    # gradients reach the subspace and no model weight; the weights are bit-identical after a DAS fit
    for prm in model.parameters():
        prm.requires_grad_(False)

    def checksum() -> list[float]:
        return [float(prm.detach().sum(dtype=torch.float64)) for prm in model.parameters()]
    ck0 = checksum()
    s = ic.Subspace(src.shape[-1], 1).to(dev)
    tg = [dict(p, target=p["t_local"] if p.get("t_local") is not None else p["source_gold"]) for p in full]
    zs = ic.patched_logits(model, tok, [p["base_prompt"] for p in tg[:4]], [p["labels"] for p in tg[:4]], layers[-1],
                           lambda hb: s.exchange(hb.float(), src[:4, -1].to(dev).float()), args.max_len, cache)
    loss = sum(torch.nn.functional.cross_entropy(z[None], torch.tensor([p["target"]], device=z.device))
               for z, p in zip(zs, tg[:4]))
    loss.backward()
    out["das_grad_norm"] = float(sum(p.grad.norm() for p in s.parameters() if p.grad is not None))
    out["model_grads"] = sum(p.grad is not None for p in model.parameters())

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()
    # DAS cost depends on the layer (backward runs from the logits down to the site): time the lowest and the highest
    # layer after a warm-up fit (kernel autotuning), and average; peak memory is the lowest-layer fit (longest backward)
    ic.fit_das(model, tok, tg[:16], src[:16, 0], layers[0], k=args.k, epochs=1, bs=args.bs, max_len=args.max_len,
               log=lambda *a: None)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_das = []
    for j in (0, len(layers) - 1):
        sync()
        t0 = time.time()
        ic.fit_das(model, tok, tg, src[:, j], layers[j], k=args.k, epochs=1, bs=args.bs, max_len=args.max_len)
        sync()
        t_das.append(time.time() - t0)
        if j == 0 and dev.type == "cuda":
            out["peak_gpu_gb_das_fit"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    out["weights_unchanged"] = checksum() == ck0
    sync()
    t0 = time.time()
    ic.iia(model, tok, tg, src[:, -1], layers[-1], "das", sub=s, max_len=args.max_len, bs=args.bs)
    sync()
    t_eval = time.time() - t0
    n = len(full)
    out["seconds_per_64"] = {"source_states": round(t_fwd * 64 / n, 2),
                             "das_epoch": round(sum(t_das) / len(t_das) * 64 / n, 2),
                             "das_epoch_by_layer": {layers[0]: round(t_das[0] * 64 / n, 2),
                                                    layers[-1]: round(t_das[-1] * 64 / n, 2)},
                             "patched_eval": round(t_eval * 64 / n, 2)}
    q = sum(p["base_ok"] and p["source_ok"] for p in pairs) / len(pairs)
    out["gpu_hours_estimate_default"] = estimate(args, out["seconds_per_64"], len(layers), q)
    return out


def estimate(args, sec: dict, n_layers: int, q: float = 0.5) -> float:
    """Hours for the run of one model with these args from the measured per-64-pair costs: per condition, source
    states for train + the evaluated test pairs, then per layer epochs x DAS + 4 patched evaluations (das, random,
    noise, full) over the union of the jointly-correct and the all-pairs test sets (about nte x (2 - q) pairs, q = the
    jointly-correct share; nte = --max-test-mid or about 900 for role/mid conditions), plus the inv scoring (3
    evaluations per layer: das, random, full; once for role:local and once for role/mid:prop). Per-slot subspaces at the readout split the same pairs, so the cost is unchanged. Prediction pass
    ignored (small). An upper bound when a condition has fewer pairs than the caps (role/mid, weak models)."""
    ntr = args.max_train or 800
    per = lambda n, s: n / 64 * s
    tot = 0.0
    for c in args.conds.split(","):
        mid = "/mid:" in c                      # role/mid conditions: test cap --max-test-mid (0: every mid test pair)
        nte = (args.max_test_mid or 900) if mid else (args.max_test or 800)
        nev = nte * (2 - q) if not args.all_pairs else nte
        tot += per(ntr + nev, sec["source_states"]) + n_layers * (args.epochs * per(ntr, sec["das_epoch"])
                                                                  + 4 * per(nev, sec["patched_eval"]))
        if c.split("@")[0] in ("role:local", "role/mid:prop") and not args.no_inv:
            nin = 400 if mid else (args.max_test or 800)            # inv/mid: about 340 test pairs
            tot += per(nin, sec["source_states"]) + n_layers * 3 * per(nin, sec["patched_eval"])
    return round(tot / 3600, 2)


# =====================================================================================================================
# Audit v2 (2026-10-09; notes/prereg_M.md, K&K addendum, "Post-hoc amendment 2026-10-09"). The v1 run (registered
# settings) could not detect a known variable: DAS loss flat after 9-30 Adam steps, readout sites at L4-12 inert, the
# stmt site inert, inv retention scored against gold. --preset v1 keeps that behaviour; v2 is the default.
# =====================================================================================================================

V2_CONDS = ("stmt:source@last,role:local@last,role:z@last,role/mid:prop@last,role/mid:local@last,role/mid:z@last,"
            "role/mid:zprop@last")
GATE_COND = "stmt:source@last"          # positive control: target = the source answer, which full exchange copies
GATE_MIN = 0.50
V2_INV_METHODS = ("das", "random", "full")
STMT_LINE = re.compile(r'^- ([A-Z][a-z]+) says: "([^"\n]+)"$', re.M)      # as build_kk_audit_pairs.STMT_LINE
PRESETS = {
    "v2": {"layers": "gate", "conds": V2_CONDS, "k": "1,4,8", "dtype": "fp32", "lr": 1e-2},
    "v1": {"layers": "auto", "conds": DEFAULT_CONDS, "k": "1", "dtype": "bf16", "lr": 1e-3},
}


def candidate_layers(n_layers: int) -> list[int]:
    """The dense late-layer candidates of the layer screen: 13..20 of 24 (2B), 17..28 of 32 (4B)."""
    return list(range(n_layers * 13 // 24, n_layers - 3))


def probe_layers_auto(n_layers: int) -> list[int]:
    """Probe layers: 16 and 20 of 24 (2B); the same depth fractions for other depths (21 and 27 of 32)."""
    return sorted({min(n_layers - 1, round(n_layers * 16 / 24)), min(n_layers - 1, round(n_layers * 20 / 24))})


def prepare_v2(args, model, tok, raw: list[dict], layers: list[int]):
    """Unpatched predictions and readout states at `layers` for every distinct prompt from one forward per prompt at the
    run's precision (the same path as every patched pass), then prepare()'s pair records with base_row / source_row
    (rows of the returned states [n_prompts, len(layers), d]) and the unpatched letter logits per prompt row."""
    prompts: dict = {}
    for p in raw:
        for k in ("base_prompt", "source_prompt"):
            prompts.setdefault(p[k], (len(prompts), p["labels"]))
    plist = list(prompts)
    t0 = time.time()
    logits, states = ic.readout_pass(model, tok, plist, [prompts[q][1] for q in plist], layers, args.max_len, args.bs)
    secs = time.time() - t0
    pred = {q: int(z.argmax()) for q, z in zip(plist, logits)}
    pairs = prepare(args, model, tok, raw, pred=pred)
    for r in pairs:
        r["base_row"], r["source_row"] = prompts[r["base_prompt"]][0], prompts[r["source_prompt"]][0]
    return pairs, states, logits, {"n_prompts": len(plist), "seconds": round(secs, 1)}


class Readout:
    """Readout-site patches as single-token decodes over stored prefixes (interchange.PrefixStore / decode_logits).
    Base and source readout states come from prepare_v2; every new prefix is checked against the unpatched forward
    (decoded logits at the store's lowest layer vs the full-forward logits): `fidelity`."""

    def __init__(self, args, model, tok, states, logits, lidx: dict, lo: int, dev):
        self.args, self.model, self.tok, self.states, self.logits, self.lidx = args, model, tok, states, logits, lidx
        self.dev = dev
        cdev = dev if dev.type == "cuda" and args.cache_device != "cpu" else torch.device("cpu")
        self.store = ic.PrefixStore(model, tok, lo, args.max_len, cdev, args.cache_gb * 2 ** 30)
        self.cache: dict = {}
        self.fidelity = {"n": 0, "max_abs_diff": 0.0, "argmax_mismatch": 0, "layer": lo}

    def ensure(self, recs: list[dict]) -> None:
        new = [p for p in recs if p["pair_id"] not in self.store.items]
        self.store.ensure([p["pair_id"] for p in recs], [p["base_prompt"] for p in recs])
        lo = self.store.lo
        for s in range(0, len(new), 64):
            ch = new[s:s + 64]
            with torch.no_grad():
                zs = self.logits_at(ch, lo, lambda hb, hs, sl=None: hb)
            for p, z in zip(ch, zs):
                ref = self.logits[p["base_row"]]
                self.fidelity["max_abs_diff"] = max(self.fidelity["max_abs_diff"], float((z.cpu() - ref).abs().max()))
                self.fidelity["argmax_mismatch"] += int(z.argmax()) != int(ref.argmax())
                self.fidelity["n"] += 1

    def hb(self, recs, L) -> torch.Tensor:
        return self.states[[p["base_row"] for p in recs], self.lidx[L]].to(self.dev)

    def hs(self, recs, L) -> torch.Tensor:
        return self.states[[p["source_row"] for p in recs], self.lidx[L]].to(self.dev)

    def logits_at(self, recs, L, hfun) -> list[torch.Tensor]:
        """hfun(hb, hs, sl) -> patched states of recs[sl] (sl = slice(None): all of them)."""
        h = hfun(self.hb(recs, L), self.hs(recs, L), slice(None))
        return ic.decode_logits(self.model, self.tok, self.store, [p["pair_id"] for p in recs], h, L,
                                [p["labels"] for p in recs], self.cache)


class Hook:
    """Patches by the forward pre-hook on the full forward (any site; the v1 mechanism, --no-prefix-cache). The source
    state at a non-readout site is computed per pair; at the readout it comes from prepare_v2."""

    def __init__(self, args, model, tok, states, lidx: dict, site: str, layers: list[int], dev):
        self.args, self.model, self.tok, self.states, self.lidx = args, model, tok, states, lidx
        self.site, self.layers, self.dev = site, layers, dev
        self.src: dict = {}
        self.cache: dict = {}

    def ensure(self, recs: list[dict]) -> None:
        if self.site == "last":
            return
        new = [p for p in recs if p["pair_id"] not in self.src]
        if new:
            st = src_states(self.args, self.model, self.tok, new, self.site, self.layers)
            for p, s in zip(new, st):
                self.src[p["pair_id"]] = s

    def hs(self, recs, L) -> torch.Tensor:
        if self.site == "last":
            return self.states[[p["source_row"] for p in recs], self.lidx[L]].to(self.dev)
        j = self.layers.index(L)
        return torch.stack([self.src[p["pair_id"]][j] for p in recs]).to(self.dev)

    def logits_at(self, recs, L, hfun) -> list[torch.Tensor]:
        """hfun(hb, hs, sl) -> patched states of recs[sl]; run in prompt batches of --bs."""
        out = []
        for s in range(0, len(recs), self.args.bs):
            sl = slice(s, s + self.args.bs)
            ch = recs[sl]
            hs = self.hs(ch, L)
            pos = None if self.site == "last" else [p["base_tok"][self.site] for p in ch]
            out += ic.patched_logits(self.model, self.tok, [p["base_prompt"] for p in ch], [p["labels"] for p in ch], L,
                                     lambda hb, hs=hs, sl=sl: hfun(hb.float(), hs.float(), sl), self.args.max_len,
                                     self.cache, pos=pos)
        return out


def is_val(p: dict, frac: float) -> bool:
    """Held-out fit split (early stopping): programs with md5("val:" + program) % round(1 / frac) == 0."""
    return frac > 0 and md5("val:" + p["program"]) % max(1, round(1 / frac)) == 0


def score2(recs: list[dict], preds: list[int]) -> dict:
    out = score(recs, preds)
    n = max(1, len(recs))
    out["coinc"] = sum(p["base_pred"] == p["target"] for p in recs) / n       # IIA of a patch that changes nothing
    return out


def screen_layers(args, model, tok, pairs, states, logits, lidx, cands, dev) -> tuple[list[int], dict]:
    """Full exchange of the readout state at each candidate layer on stmt pairs of the train half (base and source both
    right, different answers; the test half if the train half has fewer than 32): the share of patched answers equal
    to the source answer. Layers with share >= --screen-min are tested."""
    pool = [p for p in pairs if p["family"] == "stmt" and p["base_ok"] and p["source_ok"]
            and p["base_pred"] != p["source_pred"]]
    tr = [p for p in pool if md5(p["program"]) % 2 == 0]
    use = sorted(tr if len(tr) >= 32 else pool, key=lambda p: md5(p["pair_id"]))[:args.screen_pairs]
    if not use:
        return [], {"n": 0}
    rd = Readout(args, model, tok, states, logits, lidx, min(cands), dev)
    rd.ensure(use)
    copy = {}
    for L in cands:
        hit = 0
        for s in range(0, len(use), 64):
            ch = use[s:s + 64]
            with torch.no_grad():
                zs = rd.logits_at(ch, L, lambda hb, hs, sl: hs)
            hit += sum(int(z.argmax()) == p["source_pred"] for z, p in zip(zs, ch))
        copy[L] = round(hit / len(use), 4)
    sel = [L for L in cands if copy[L] >= args.screen_min]
    return sel, {"n": len(use), "copy_source_full": copy, "min": args.screen_min, "selected": sel,
                 "decode_fidelity": rd.fidelity}


def write_preds_v2(args, cname: str, ev: list[dict], sets: dict, preds: dict) -> None:
    """preds: {(layer, k): {method: [pred per ev pair]}}; sets: {name: pair ids} -> in_<name> flags."""
    with open(Path(args.out) / "preds.jsonl", "a") as f:
        f.write(json.dumps({"kind": "pairs", "version": "v2", "model": args.name, "cond": cname,
                            "pairs": [dict({k: p.get(k) for k in PAIR_META},
                                           **{f"in_{s}": p["pair_id"] in ids for s, ids in sets.items()})
                                      for p in ev]}) + "\n")
        for (L, k), by in preds.items():
            for m, ks in by.items():
                f.write(json.dumps({"kind": "preds", "version": "v2", "model": args.name, "cond": cname, "layer": L,
                                    "k": k, "method": m, "preds": ks}) + "\n")


def predict_all(args, be, ev: list[dict], layers, ks, subs: dict, key, methods, d: int, dev) -> dict:
    """{(L, k): {method: [patched answer per ev pair]}}: das / noise use each pair's slot subspace, random a random
    rank-k subspace (one per layer and k), full the source state (rank independent, repeated under every k)."""
    preds = {(L, k): {m: [None] * len(ev) for m in methods} for L in layers for k in ks}
    rnd = {(L, k): ic.random_subspace(d, k, seed=1000 + L + 100 * k).R.detach().to(dev) for L in layers for k in ks}
    gen = {(L, k): torch.Generator().manual_seed(args.seed + 10 * L + k) for L in layers for k in ks}
    for s in range(0, len(ev), args.eval_chunk):
        ch = ev[s:s + args.eval_chunk]
        be.ensure(ch)
        for L in layers:
            full = None
            for k in ks:
                with torch.no_grad():
                    S = subs[(L, k)]
                    R = {g: S[g].R.detach() for g in S}
                    Rrow = torch.stack([R[key(p)] for p in ch]).to(dev)
                    Rr = rnd[(L, k)][None].expand(len(ch), -1, -1)
                    fns = {"das": lambda hb, hs, sl: ic.exchange_rows(hb, hs, Rrow[sl]),
                           "random": lambda hb, hs, sl: ic.exchange_rows(hb, hs, Rr[sl]),
                           "noise": lambda hb, hs, sl, g=gen[(L, k)]: ic.noise_rows(hb, hs, Rrow[sl], g)}
                    for m in methods:
                        if m == "full":
                            if full is None:
                                full = [int(z.argmax()) for z in be.logits_at(ch, L, lambda hb, hs, sl: hs)]
                            out = full
                        else:
                            out = [int(z.argmax()) for z in be.logits_at(ch, L, fns[m])]
                        preds[(L, k)][m][s:s + len(ch)] = out
    return preds


def run_v2(args, model, tok, pairs: list[dict], layers: list[int], states, logits, lidx: dict, head: dict) -> dict:
    dev = text_parts(model)[1].weight.device
    d = states.shape[-1]
    ks = [int(x) for x in str(args.k).split(",")]
    out = Path(args.out)
    (out / "preds.jsonl").write_text("")
    results: dict = {"version": "v2", "args": vars(args), "head": head, "conds": {}, "fits": {}, "gate": None}
    readout = Readout(args, model, tok, states, logits, lidx, min(layers), dev) if args.prefix_cache else None
    subspaces: dict = {}

    def save():
        if readout:
            results["decode_fidelity"] = dict(readout.fidelity, bytes_per_prompt=round(readout.store.bytes_per_prompt()),
                                              n_built=readout.store.n_built)
        (out / "audit.json").write_text(json.dumps(results, indent=1))

    def scored(ev, preds, sets, L, k, methods):
        row = {}
        for sname, ids in sets.items():
            keep = [i for i, p in enumerate(ev) if p["pair_id"] in ids]
            recs = [ev[i] for i in keep]
            row[sname] = {m: rounded(score2(recs, [preds[(L, k)][m][i] for i in keep])) for m in methods}
        return row

    for cname in args.conds.split(","):
        cond = parse_cond(cname)
        key = slot_key(cond, args)
        be = readout if cond["site"] == "last" and readout else Hook(args, model, tok, states, lidx, cond["site"],
                                                                     layers, dev)
        noop: list = []
        sel = select(pairs, cond, not args.all_pairs, noop)
        tr0 = split(sel, args.max_train, 0, key)[0]
        n_tr = {g: sum(key(p) == g for p in tr0) for g in sorted({key(p) for p in tr0})}
        slots = {g for g, n in n_tr.items() if n >= args.min_slot_pairs}
        skipped = {str(g): n for g, n in n_tr.items() if g not in slots}
        sel = [p for p in sel if key(p) in slots]
        mte = args.max_test_mid if cond["subset"] == "mid" else args.max_test
        tr, te = split(sel, args.max_train, mte, key)
        te_all = split([p for p in select(pairs, cond, False) if key(p) in slots], 0, mte, key)[1]
        fit = [p for p in tr if not is_val(p, args.val_frac)]
        slots = {key(p) for p in fit}                    # a slot whose train pairs all fell in the held-out split
        val = [p for p in tr if is_val(p, args.val_frac) and key(p) in slots]
        te, te_all = [p for p in te if key(p) in slots], [p for p in te_all if key(p) in slots]
        info = {"n_train": len(tr), "n_fit": len(fit), "n_val": len(val), "n_test_both": len(te),
                "n_test_all": len(te_all), "n_noop": len(noop), "n_slots": len(slots), "skipped_slots": skipped}
        if len(fit) < args.min_pairs or len(te) < args.min_pairs:
            results["conds"][cname] = dict(info, skipped=f"fit {len(fit)}, test {len(te)}")
            emit(args, {"model": args.name, "version": "v2", "cond": cname, **results["conds"][cname]})
            save()
            continue
        t0 = time.time()
        be.ensure(fit + val)
        n_slots = len({key(p) for p in fit})
        for L in layers:
            for k in ks:
                def fwd(rows, train, R, L=L):
                    recs = [(fit if train else val)[i] for i in rows]
                    return be.logits_at(recs, L, lambda hb, hs, sl: ic.exchange_rows(hb, hs, R[sl]))
                subs, fi = ic.fit_slot_subspaces(
                    fwd, d, k, [key(p) for p in fit], [key(p) for p in val], [p["target"] for p in fit],
                    [p["target"] for p in val], [len(p["labels"]) for p in fit], steps_min=args.das_steps,
                    steps_max=max(args.das_steps, args.das_steps_max), patience=args.das_patience,
                    eval_every=args.das_eval_every, rows_per_slot=max(1, args.das_bs // max(1, n_slots)), lr=args.lr,
                    seed=args.seed + 1000 * L + k, dev=dev, fwd_chunk=0 if be is readout else args.bs)
                subspaces[(cname, L, k)] = subs
                results["fits"][f"{cname}|L{L}|k{k}"] = fi
                print(f"fit {cname} L{L} k{k}: {fi['steps']} steps ({fi['stop']}), loss {fi['init_loss']} -> "
                      f"{fi['best_loss']} @ {fi['best_step']} (chance {fi['chance_loss']})", flush=True)
        ev = union(te_all, te)
        sets = {"both": {p["pair_id"] for p in te}, "all": {p["pair_id"] for p in te_all}}
        preds = predict_all(args, be, ev, layers, ks, {(L, k): subspaces[(cname, L, k)] for L in layers for k in ks},
                            key, METHODS, d, dev)
        write_preds_v2(args, cname, ev, sets, preds)
        rows = {}
        for L in layers:
            for k in ks:
                row = scored(ev, preds, sets, L, k, METHODS)
                keep = [i for i, p in enumerate(ev) if p["pair_id"] in sets["both"]]
                row["das_strata_both"] = strata([ev[i] for i in keep], [preds[(L, k)]["das"][i] for i in keep])
                rows[f"L{L}/k{k}"] = row
                fi = results["fits"][f"{cname}|L{L}|k{k}"]
                emit(args, {"model": args.name, "version": "v2", "cond": cname, "layer": L, "k": k, **info,
                            "both": row["both"], "all": row["all"],
                            "fit": {x: fi.get(x) for x in ("steps", "stop", "best_step", "init_loss", "best_loss",
                                                           "rel_drop", "chance_loss", "below_chance", "converged")}})
        results["conds"][cname] = dict(info, seconds=round(time.time() - t0, 1), cells=rows)
        if cname == GATE_COND:
            cells = {f"L{L}/k{k}": rows[f"L{L}/k{k}"]["both"]["das"]["iia"] for L in layers for k in ks}
            by_k = {k: max(rows[f"L{L}/k{k}"]["both"]["das"]["iia"] for L in layers) for k in ks}
            results["gate"] = {"gate_cond": GATE_COND, "threshold": GATE_MIN, "set": "both", "iia": cells,
                               "best_by_k": by_k, "passed_k": {k: v >= GATE_MIN for k, v in by_k.items()},
                               "passed": any(v >= GATE_MIN for v in by_k.values())}
            emit(args, {"model": args.name, "version": "v2", "cond": "_gate", **results["gate"]})
        save()
        # inv pairs (X's role unchanged in the source) with the role:local and role/mid:prop subspaces: retention of the
        # model's own unpatched answer (base_pred), not of gold
        if cond["family"] == "role" and (cond["target"], cond["subset"]) in (("local", None), ("prop", "mid")) \
                and not args.no_inv:
            sub_ = cond["subset"]
            inv_cond = {"name": f"inv{'/' + sub_ if sub_ else ''}:base@{cond['site']}", "family": "inv",
                        "subset": sub_, "target": "base", "site": cond["site"]}
            inv = split([p for p in select(pairs, inv_cond, not args.all_pairs) if key(p) in slots], 0,
                        args.max_test, key)[1]
            inv_all = split([p for p in select(pairs, inv_cond, False) if key(p) in slots], 0, args.max_test, key)[1]
            if len(inv) >= args.min_pairs:
                ev = [dict(p, target=p["base_pred"]) for p in union(inv_all, inv)]   # target = own unpatched answer
                isets = {"both": {p["pair_id"] for p in inv}, "all": {p["pair_id"] for p in inv_all}}
                ipreds = predict_all(args, be, ev, layers, ks,
                                     {(L, k): subspaces[(cname, L, k)] for L in layers for k in ks}, key,
                                     V2_INV_METHODS, d, dev)
                write_preds_v2(args, inv_cond["name"], ev, isets, ipreds)
                irows = {}
                for L in layers:
                    for k in ks:
                        irows[f"L{L}/k{k}"] = scored(ev, ipreds, isets, L, k, V2_INV_METHODS)
                        emit(args, {"model": args.name, "version": "v2", "cond": inv_cond["name"], "subspace_from": cname,
                                    "layer": L, "k": k, "n_test_both": len(inv), "n_test_all": len(inv_all),
                                    "retention_target": "base_pred", **irows[f"L{L}/k{k}"]})
                results["conds"][inv_cond["name"]] = {"n_test_both": len(inv), "n_test_all": len(inv_all),
                                                      "subspace_from": cname, "retention_target": "base_pred",
                                                      "cells": irows}
        save()
    return results


# ---- linear probes (correlational screen) ---------------------------------------------------------------------------

def probe_records(pairs: list[dict], tok, max_len: int) -> list[dict]:
    """One base prompt per program (its role pair with the smallest md5(pair_id)): every person's name token and
    statement closing-quote token, base roles (options[base_gold]) and slots."""
    by: dict = {}
    for p in pairs:
        if p["family"] == "role" and (p["program"] not in by or md5(p["pair_id"]) < md5(by[p["program"]]["pair_id"])):
            by[p["program"]] = p
    out = []
    for prog in sorted(by):
        p = by[prog]
        lines = {m.group(1): m for m in STMT_LINE.finditer(p["base_prompt"])}
        if any(x not in lines for x in p["people"]):
            continue
        name = [token_pos(tok, p["base_prompt"], lines[x].start(1) + len(x) - 1, max_len) for x in p["people"]]
        stmt = [token_pos(tok, p["base_prompt"], lines[x].end() - 1, max_len) for x in p["people"]]
        out.append({"program": prog, "pair_id": p["pair_id"], "prompt": p["base_prompt"], "people": p["people"],
                    "roles": [int(bool(b)) for b in p["options"][p["base_gold"]]], "pos": name + stmt,
                    "base_row": p.get("base_row")})
    return out


def fit_logreg(X: torch.Tensor, y: torch.Tensor, lam: float, iters: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """L2-regularised logistic regression by L-BFGS (X standardised by the caller)."""
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + lam * (w @ w)
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), b.detach()


def probe(Xtr, ytr, gtr, Xte, yte, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)) -> dict:
    """Test accuracy of a logistic probe whose L2 weight is chosen on held-out train programs (gtr: program per row;
    md5("val:" + program) % 5 == 0 is the validation part); majority = the train majority label on the test rows."""
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    v = torch.tensor([md5("val:" + g) % 5 == 0 for g in gtr])
    best = (-1.0, lams[0])
    if v.any() and (~v).any():
        for lam in lams:
            w, b = fit_logreg(Xtr[~v], ytr[~v], lam)
            acc = float((((Xtr[v] @ w + b) > 0).float() == ytr[v]).float().mean())
            best = max(best, (acc, lam), key=lambda t: (t[0], t[1]))
    w, b = fit_logreg(Xtr, ytr, best[1])
    maj = float(ytr.mean() >= 0.5)
    return {"acc": round(float((((Xte @ w + b) > 0).float() == yte).float().mean()), 4), "lam": best[1],
            "majority": round(float((yte == maj).float().mean()), 4), "n_train": len(ytr), "n_test": len(yte)}


def control_label(name: str, seed: int = 0) -> int:
    """Control task (Hewitt & Liang 2019): a fixed random label per person name (word type)."""
    return md5(f"control{seed}:{name}") % 2


def run_probes(args, model, tok, pairs, states, lidx: dict, players: list[int]) -> dict:
    """role(person) probes at each person's name token and statement closing quote (pooled over persons), and per slot
    at the readout, at the probe layers; each next to its control task (labels = control_label of the person's name)
    and the majority baseline. Train programs: md5(program) even (as the DAS split), test: odd."""
    recs = probe_records(pairs, tok, args.max_len)
    t0 = time.time()
    st = ic.token_states(model, tok, [r["prompt"] for r in recs], [r["pos"] for r in recs], players, args.max_len,
                         args.bs)
    res: dict = {"layers": players, "n_programs": len(recs), "control": "random bit per person name (Hewitt & Liang)",
                 "name": {}, "stmt": {}, "readout": {}}
    tr = [md5(r["program"]) % 2 == 0 for r in recs]
    for j, L in enumerate(players):
        for site, off in (("name", 0), ("stmt", 1)):
            X = {True: [], False: []}
            Y = {True: [], False: []}
            C = {True: [], False: []}
            G = {True: [], False: []}
            for r, s, t in zip(recs, st, tr):
                n = len(r["people"])
                for i, x in enumerate(r["people"]):
                    X[t].append(s[off * n + i, j])
                    Y[t].append(r["roles"][i])
                    C[t].append(control_label(x))
                    G[t].append(r["program"])
            if not X[True] or not X[False]:
                continue
            Xtr, Xte = torch.stack(X[True]), torch.stack(X[False])
            task = probe(Xtr, torch.tensor(Y[True], dtype=torch.float), G[True], Xte, torch.tensor(Y[False],
                                                                                                    dtype=torch.float))
            ctl = probe(Xtr, torch.tensor(C[True], dtype=torch.float), G[True], Xte, torch.tensor(C[False],
                                                                                                   dtype=torch.float))
            res[site][L] = dict(task, control_acc=ctl["acc"], control_majority=ctl["majority"],
                                selectivity=round(task["acc"] - ctl["acc"], 4))
        if L in lidx and all(r["base_row"] is not None for r in recs):
            by = {}
            for slot in range(max(len(r["people"]) for r in recs)):
                rows = [(r, t) for r, t in zip(recs, tr) if len(r["people"]) > slot]
                a = [(states[r["base_row"], lidx[L]], r["roles"][slot], control_label(r["people"][slot]), r["program"], t)
                     for r, t in rows]
                trn, tst = [x for x in a if x[4]], [x for x in a if not x[4]]
                if len(trn) < 20 or len(tst) < 20:
                    continue
                Xtr, Xte = torch.stack([x[0] for x in trn]), torch.stack([x[0] for x in tst])
                f = lambda i, s: torch.tensor([x[i] for x in s], dtype=torch.float)
                task = probe(Xtr, f(1, trn), [x[3] for x in trn], Xte, f(1, tst))
                ctl = probe(Xtr, f(2, trn), [x[3] for x in trn], Xte, f(2, tst))
                by[slot] = dict(task, control_acc=ctl["acc"], selectivity=round(task["acc"] - ctl["acc"], 4))
            if by:
                res["readout"][L] = {"mean_acc": round(sum(v["acc"] for v in by.values()) / len(by), 4),
                                     "mean_control_acc": round(sum(v["control_acc"] for v in by.values()) / len(by), 4),
                                     "mean_majority": round(sum(v["majority"] for v in by.values()) / len(by), 4),
                                     "mean_selectivity": round(sum(v["selectivity"] for v in by.values()) / len(by), 4),
                                     "by_slot": by}
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ---- smoke checks and cost (v2) -------------------------------------------------------------------------------------

def check_v2(args, model, tok, pairs, states, logits, lidx, cands, dev, prep: dict) -> dict:
    out: dict = {"version": "v2", "candidates": cands}
    cache: dict = {}
    sub = [p for p in pairs if p["family"] == "role"][:16] or pairs[:16]
    bp, lb = [p["base_prompt"] for p in sub], [p["labels"] for p in sub]
    # identity patches (hook) at the readout and the statement site; precision of the forward
    with torch.no_grad():
        b = ic._batch(tok, bp, lb, args.max_len, dev)
        with ic.amp(dev):
            ref = ic._letter_logits(model, tok, b, lb, cache)
        for site in ("last", "stmt"):
            pos = None if site == "last" else [p["base_tok"]["stmt"] for p in sub]
            for L in (cands[0], cands[-1]):
                z = ic.patched_logits(model, tok, bp, lb, L, lambda h: h, args.max_len, cache, pos=pos)
                out[f"identity_max_abs_diff/{site}/L{L}"] = max(float((a - c).abs().max()) for a, c in zip(ref, z))
        seen = []
        lin = next(m for m in ic.decoder_layers(model)[cands[0]].modules() if isinstance(m, torch.nn.Linear))
        hk = lin.register_forward_hook(lambda m, i, o: seen.append(o.dtype))
        ic.patched_logits(model, tok, bp[:2], lb[:2], cands[0], lambda h: h, args.max_len, cache)
        hk.remove()
        out["matmul_output_dtype"] = str(seen[0]).replace("torch.", "")
    ok_stmt = ok_name = 0
    for p in pairs[:64]:
        ids = tok.encode(FINAL_TEMPLATE.format(problem=p["base_prompt"]), add_special_tokens=False)
        ok_stmt += '"' in tok.decode([ids[p["base_tok"]["stmt"]]])
        ok_name += tok.decode([ids[p["base_tok"]["name"]]]).strip() in p["person"]
    out["stmt_site_is_quote"] = ok_stmt / min(64, len(pairs))
    out["name_site_in_name"] = ok_name / min(64, len(pairs))
    out["n_pairs"] = len(pairs)
    out["base_acc"] = sum(p["base_ok"] for p in pairs) / len(pairs)
    out["jointly_correct"] = {f: sum(p["base_ok"] and p["source_ok"] for p in pairs if p["family"] == f)
                              for f in ("role", "inv", "stmt")}
    # layer screen (full exchange of the readout state, decode path)
    sel, scr = screen_layers(args, model, tok, pairs, states, logits, lidx, cands, dev)
    out["layer_screen"] = scr
    out["layers_selected"] = sel
    best = max(scr.get("copy_source_full", {cands[-1]: 0}).items(), key=lambda t: (t[1], t[0]))[0]
    out["best_copy_layer"] = best
    # decode path vs the hook-patched full forward, at every candidate layer
    rd = Readout(args, model, tok, states, logits, lidx, min(cands), dev)
    fp = pairs[:16]
    t0 = time.time()
    rd.ensure(fp)
    t_prefill = (time.time() - t0) / len(fp)
    kmax = max(int(x) for x in str(args.k).split(","))
    fid = {}
    for L in cands:
        R = ic.random_subspace(states.shape[-1], kmax, seed=7 + L).R.detach().to(dev)
        Rr = R[None].expand(len(fp), -1, -1)
        with torch.no_grad():
            row = {}
            fns = (("unpatched", lambda hb, hs, sl=None: hb),
                   ("random_k", lambda hb, hs, sl=None: ic.exchange_rows(hb, hs, Rr)),
                   ("full", lambda hb, hs, sl=None: hs))
            for nm, fn in fns:
                zd = rd.logits_at(fp, L, fn)
                hs = rd.hs(fp, L)
                zh = ic.patched_logits(model, tok, [p["base_prompt"] for p in fp], [p["labels"] for p in fp], L,
                                       lambda hb, f=fn, hs=hs: f(hb.float(), hs.float()), args.max_len, cache)
                row[nm] = round(max(float((a - c).abs().max()) for a, c in zip(zd, zh)), 6)
                row[nm + "_argmax_agree"] = sum(int(a.argmax()) == int(c.argmax()) for a, c in zip(zd, zh)) / len(fp)
            fid[L] = row
    out["decode_fidelity"] = fid
    out["decode_max_abs_diff"] = max(max(v["unpatched"], v["random_k"], v["full"]) for v in fid.values())
    out["decode_prefix_fidelity"] = rd.fidelity
    out["cache_bytes_per_prompt"] = round(rd.store.bytes_per_prompt())
    # bf16 fidelity of a rank-1 exchange: in the model's dtype and, explicitly, in bf16 whatever the model's dtype
    with torch.no_grad():
        rnd = ic.random_subspace(states.shape[-1], 1, seed=7)
        dt = next(model.parameters()).dtype
        q = pairs[:64]
        for nm, cast in (("exchange_fidelity", dt), ("exchange_fidelity_bf16", torch.bfloat16)):
            out[nm] = {}
            for L in cands:
                hb = states[[p["base_row"] for p in q], lidx[L]]
                hs = states[[p["source_row"] for p in q], lidx[L]]
                want = rnd.exchange(hb, hs) - hb
                got = rnd.exchange(hb.to(cast), hs.to(cast)).to(cast).float() - hb.to(cast).float()
                live = want.norm(dim=-1) > 1e-6 * hb.norm(dim=-1)
                if not bool(live.any()):
                    out[nm][L] = None
                    continue
                got, want = got[live], want[live]
                cos = torch.nn.functional.cosine_similarity(got, want, dim=-1)
                out[nm][L] = {"norm_ratio": round(float((got.norm(dim=-1) / want.norm(dim=-1)).median()), 4),
                              "cos_median": round(float(cos.median()), 4), "cos_min": round(float(cos.min()), 4),
                              "n": int(live.sum())}
    # gradients reach the subspaces only; weights bit-identical after the fits; positive-control mini-fit
    for prm in model.parameters():
        prm.requires_grad_(False)
    ck0 = [float(prm.detach().sum(dtype=torch.float64)) for prm in model.parameters()]
    st = [dict(p, target=p["source_gold"]) for p in pairs if p["family"] == "stmt" and p["base_ok"] and p["source_ok"]
          and p["base_pred"] != p["source_pred"]] or [dict(p, target=p["source_gold"]) for p in pairs
                                                     if p["family"] == "stmt"][:16]
    rd.ensure(st)
    s = ic.Subspace(states.shape[-1], 1).to(dev)
    zs = rd.logits_at(st[:4], best,
                      lambda hb, hs, sl=None: ic.exchange_rows(hb, hs, s.R[None].expand(len(st[:4]), -1, -1)))
    sum(torch.nn.functional.cross_entropy(z[None], torch.tensor([p["target"]], device=z.device))
        for z, p in zip(zs, st[:4])).backward()
    out["das_grad_norm"] = float(sum(p.grad.norm() for p in s.parameters() if p.grad is not None))
    out["model_grads"] = sum(p.grad is not None for p in model.parameters())

    def fwd(rows, train, R):
        return rd.logits_at([st[i] for i in rows], best, lambda hb, hs, sl=None: ic.exchange_rows(hb, hs, R))
    steps = args.check_fit_steps
    t0 = time.time()
    _, fi = ic.fit_slot_subspaces(fwd, states.shape[-1], kmax, [0] * len(st), [], [p["target"] for p in st], [],
                                  [len(p["labels"]) for p in st], steps_min=steps, steps_max=steps, patience=steps,
                                  eval_every=max(1, steps // 10), rows_per_slot=min(args.das_bs, len(st)), lr=args.lr,
                                  seed=args.seed, dev=dev)
    t_fit = time.time() - t0
    first = next(c["train_loss"] for c in fi["curve"] if c["train_loss"] is not None)
    last = fi["curve"][-1]["train_loss"]
    out["posctl_fit"] = {"cond": GATE_COND, "layer": best, "k": kmax, "n": len(st), "steps": steps,
                         "first_loss": first, "last_loss": last, "rel_drop": round((first - last) / first, 4),
                         "curve": fi["curve"]}
    out["weights_unchanged"] = [float(prm.detach().sum(dtype=torch.float64)) for prm in model.parameters()] == ck0
    # timing: decode DAS step (fwd + bwd, das_bs rows) at the lowest and highest candidate layer, eval decode, prefill
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    t_step = {}
    rows = [st[i % len(st)] for i in range(args.das_bs)]
    rd.ensure(rows)
    for L in (cands[0], cands[-1]):
        s = ic.Subspace(states.shape[-1], kmax).to(dev)
        sync()
        t0 = time.time()
        for _ in range(3):
            zs = rd.logits_at(rows, L,
                              lambda hb, hs, sl=None: ic.exchange_rows(hb, hs, s.R[None].expand(len(rows), -1, -1)))
            torch.stack([z.logsumexp(0) for z in zs]).sum().backward()
        sync()
        t_step[L] = (time.time() - t0) / 3
    sync()
    t0 = time.time()
    with torch.no_grad():
        rd.logits_at(rows, cands[0], lambda hb, hs, sl=None: hs)
    sync()
    t_dec = (time.time() - t0) / len(rows)
    if dev.type == "cuda":
        out["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    out["seconds"] = {"forward_per_prompt": round(prep["seconds"] / max(1, prep["n_prompts"]), 4),
                      "prefill_per_prompt": round(t_prefill, 4),
                      "das_step": {L: round(v, 4) for L, v in t_step.items()},
                      "decode_eval_per_row": round(t_dec, 5), "posctl_fit": round(t_fit, 1)}
    out["gpu_hours_estimate"] = estimate_v2(args, out["seconds"], len(sel or cands), len(str(args.k).split(",")),
                                            n_layers_total=len(ic.decoder_layers(model)))
    return out


def estimate_v2(args, sec: dict, n_sel: int, n_k: int, n_layers_total: int = 24) -> dict:
    """GPU hours of one model's v2 run from the --check timings (all 8313 pairs; nominal pair counts from the 2B v1 run:
    per condition about 600 fit+val pairs (mid conditions about 300) and 900 evaluated test pairs (mid about 1100)).
    expected = every fit stops at --das-steps; upper = every fit runs --das-steps-max. Cost terms: one forward per
    distinct prompt (16624), one prefix forward per evaluated or fitted pair, DAS steps (+ a validation decode every
    --das-eval-every steps), patched evaluation decodes (4 methods x k plus inv: 3 x k), probes (900 forwards)."""
    conds = args.conds.split(",")
    step = sum(sec["das_step"].values()) / len(sec["das_step"])
    fwd, pre, dec = sec["forward_per_prompt"], sec["prefill_per_prompt"], sec["decode_eval_per_row"]
    n_fit = n_ev = n_dec = 0.0
    for c in conds:
        mid = "/mid:" in c
        n_fit += 300 if mid else 600
        ev = 1100 if mid else 900
        n_ev += ev
        n_dec += n_sel * (3 * n_k + 1) * ev
        if c.split("@")[0] in ("role:local", "role/mid:prop"):
            n_ev += 400 if mid else 900
            n_dec += n_sel * (2 * n_k + 1) * (400 if mid else 900)
    fits = len(conds) * n_sel * n_k
    val_dec = 0.2 * 600 / args.das_eval_every * dec        # validation decodes per step, about 120 rows per evaluation
    base = 16624 * fwd + (n_fit + n_ev) * pre + n_dec * dec + 900 * fwd
    return {"expected": round((base + fits * args.das_steps * (step + val_dec)) / 3600, 2),
            "upper": round((base + fits * max(args.das_steps, args.das_steps_max) * (step + val_dec)) / 3600, 2),
            "fits": fits}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default=None, help="model name in the summary lines (default: the model path)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", choices=("v2", "v1"), default="v2",
                    help="v2 (default): settings of the 2026-10-09 amendment; v1: the registered 2026-10-06 run")
    ap.add_argument("--pairs", default="data/eval/kk_audit_pairs.jsonl")
    ap.add_argument("--layers", default=None,
                    help="v2: gate (default; candidates 13..20 of 24 / 17..28 of 32, kept if the full-exchange copy "
                         "rate >= --screen-min), gate:A-B (candidates A..B), or a comma list (no screen); "
                         "v1: auto = 4, 8, ..., n_layers - 4, or a comma list")
    ap.add_argument("--conds", default=None, help="default: the preset's conditions")
    ap.add_argument("--sizes", default="4-8", help="n_people range kept")
    ap.add_argument("--k", default=None, help="rank(s): v2 comma list (default 1,4,8), v1 one rank (default 1)")
    ap.add_argument("--epochs", type=int, default=3, help="v1 only: DAS epochs")
    ap.add_argument("--lr", type=float, default=None, help="DAS Adam learning rate (v2 default 1e-2, v1 1e-3)")
    ap.add_argument("--das-steps", type=int, default=500, help="v2: minimum DAS steps per subspace fit")
    ap.add_argument("--das-steps-max", type=int, default=1500, help="v2: maximum DAS steps")
    ap.add_argument("--das-patience", type=int, default=200,
                    help="v2: stop after this many steps without a new best held-out loss (once --das-steps are done)")
    ap.add_argument("--das-eval-every", type=int, default=25, help="v2: held-out loss every N steps (the curve)")
    ap.add_argument("--das-bs", type=int, default=32, help="v2: rows per DAS step, spread evenly over the slots")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="v2: share of train programs held out for early stopping (md5 split)")
    ap.add_argument("--screen-min", type=float, default=0.5, help="v2 layer screen: minimum full-exchange copy rate")
    ap.add_argument("--screen-pairs", type=int, default=128)
    ap.add_argument("--probe-layers", default="auto", help="v2 probes: auto = 16,20 of 24 (same fractions otherwise)")
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--no-prefix-cache", dest="prefix_cache", action="store_false",
                    help="v2: patch through the full forward (hook) instead of single-token decodes over cached "
                         "prefixes (degraded mode: about 100x slower)")
    ap.add_argument("--cache-gb", type=float, default=None,
                    help="v2: prefix-cache memory cap in GB (default: 60%% of the GPU memory free after loading the "
                         "model, minus 4 GB, at least 2; 4 on CPU)")
    ap.add_argument("--cache-device", choices=("auto", "cpu"), default="auto")
    ap.add_argument("--eval-chunk", type=int, default=64)
    ap.add_argument("--check-fit-steps", type=int, default=200, help="v2 --check: positive-control mini-fit steps")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--max-train", type=int, default=600)
    ap.add_argument("--max-test", type=int, default=600)
    ap.add_argument("--max-test-mid", type=int, default=0,
                    help="test cap of the role/mid and inv/mid conditions (0: none, so role/mid:prop, :local, :z and "
                         ":zprop are all scored on every mid test pair and the E2 difference-in-differences uses "
                         "every zprop pair)")
    ap.add_argument("--max-pairs", type=int, default=0, help="first N pairs per family (smoke tests)")
    ap.add_argument("--min-pairs", type=int, default=20)
    ap.add_argument("--min-slot-pairs", type=int, default=10,
                    help="per-slot subspaces at the readout: slots with fewer train pairs are skipped (reported)")
    ap.add_argument("--pool-slots", action="store_true",
                    help="one subspace per condition at every site (pre-review behaviour; smoke runs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-pairs", action="store_true", help="fit and test on all pairs, not only jointly correct")
    ap.add_argument("--no-inv", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dtype", choices=("bf16", "fp32", "fp32-autocast"), default=None,
                    help="v2 default fp32: fp32 weights, autocast and TF32 off (every product in fp32); fp32-autocast: "
                         "fp32 weights under bf16 autocast (what v1's --dtype fp32 ran); bf16: bf16 weights + autocast. "
                         "Under --preset v1, fp32 means fp32-autocast (v1 behaviour)")
    args = ap.parse_args(argv)
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.preset == "v1":
        args.k = int(args.k)
        if args.dtype == "fp32":
            args.dtype = "fp32-autocast"
    args.name = args.name or args.model
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import transformers
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float32 if dev.type != "cuda" or args.dtype != "bf16" else torch.bfloat16
    model = load_model(args.model, dt).to(dev)
    model.eval()
    prec = ic.set_precision(args.dtype)
    if args.cache_gb is None:
        args.cache_gb = (round(max(2.0, 0.6 * torch.cuda.mem_get_info()[0] / 2 ** 30 - 4), 1) if dev.type == "cuda"
                         else 4.0)
    kernels = ic.use_reference_kernels(model) if args.preset == "v2" else []
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    n_layers = len(ic.decoder_layers(model))
    screen = args.preset == "v2" and str(args.layers).startswith("gate")
    if args.layers == "auto":
        layers = list(range(4, n_layers - 3, 4))
    elif screen:
        if ":" in args.layers:
            a, b = (int(x) for x in args.layers.split(":")[1].split("-"))
            layers = list(range(a, b + 1))
        else:
            layers = candidate_layers(n_layers)
    else:
        layers = [int(x) for x in str(args.layers).split(",")]
    # hidden_states[n_layers] is the final-normed output, not a layer input: sites must be 0 <= L < n_layers
    assert layers and all(0 <= L < n_layers for L in layers), (layers, n_layers)
    path = Path(args.pairs) if Path(args.pairs).exists() else ROOT / args.pairs
    lo, hi = (int(x) for x in args.sizes.split("-"))
    raw = [p for p in read(path) if lo <= p["n_people"] <= hi]
    if args.max_pairs:
        seen: dict = {}
        raw = [p for p in raw if seen.setdefault(p["family"], []).append(1) or len(seen[p["family"]]) <= args.max_pairs]
    if args.preset == "v1":
        args.layers = ",".join(map(str, layers))
        pairs = prepare(args, model, tok, raw)
    else:
        players = [] if args.no_probes else (probe_layers_auto(n_layers) if args.probe_layers == "auto"
                                             else [int(x) for x in args.probe_layers.split(",")])
        collect = sorted(set(layers) | set(players))
        pairs, states, logits, prep = prepare_v2(args, model, tok, raw, collect)
        lidx = {L: j for j, L in enumerate(collect)}
    head = {"model": args.name, "version": args.preset, "job": os.environ.get("SLURM_JOB_ID"), "n_layers": n_layers,
            "pairs_file": args.pairs, "pairs_md5": hashlib.md5(path.read_bytes()).hexdigest(),
            "n": len(pairs), "base_acc": round(sum(p["base_ok"] for p in pairs) / len(pairs), 4),
            "source_acc": round(sum(p["source_ok"] for p in pairs) / len(pairs), 4),
            "jointly_correct": {f: sum(p["base_ok"] and p["source_ok"] for p in pairs if p["family"] == f)
                                for f in ("role", "inv", "stmt")},
            "dtype": args.dtype, **prec, "kernels": "reference" if kernels else "default", "kernels_replaced": kernels}
    if args.preset == "v1":
        head["layers"] = layers
        print(json.dumps(head), flush=True)
        if args.check:
            res = dict(head, **check(args, model, tok, pairs, layers))
            (out / "check.json").write_text(json.dumps(res, indent=1))
            print(json.dumps(res, indent=1))
            return
        emit(args, dict(head, cond="_head"))
        run(args, model, tok, pairs, layers)
        return
    head.update(prepare=prep, candidates=layers if screen else None, probe_layers=players,
                prefix_cache=args.prefix_cache)
    print(json.dumps(head), flush=True)
    if args.check:
        res = dict(head, **check_v2(args, model, tok, pairs, states, logits, lidx, layers, dev, prep))
        (out / "check.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return
    if screen:
        sel, scr = screen_layers(args, model, tok, pairs, states, logits, lidx, layers, dev)
        head["layer_screen"] = scr
        if not sel:
            (out / "audit.json").write_text(json.dumps({"version": "v2", "args": vars(args), "head": head}, indent=1))
            raise SystemExit(f"layer screen: no candidate layer has full-exchange copy >= {args.screen_min}: {scr}")
        layers = sel
    head["layers"] = layers
    args.layers = ",".join(map(str, layers))
    print("layers tested:", layers, flush=True)
    emit(args, dict(head, cond="_head"))
    if players:
        pr = run_probes(args, model, tok, pairs, states, lidx, players)
        head["probes"] = pr
        emit(args, {"model": args.name, "version": "v2", "cond": "_probes", **pr})
    run_v2(args, model, tok, pairs, layers, states, logits, lidx, head)


if __name__ == "__main__":
    main()
