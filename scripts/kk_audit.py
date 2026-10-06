"""Interchange-intervention audit of a decision model on Knights & Knaves (GPU job; design:
notes/next_round/causal_audit/kk_design.md). Reads data/eval/kk_audit_pairs.jsonl only (scripts/build_kk_audit_pairs.py
built and solver-checked it on CPU; the solver is not needed here).

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

--layers auto (default) = every 4th layer from 4 to n_layers - 4 (the input of the layer after each full-attention
layer): 4,8,12,16,20 for the 24-layer 2B, 4,8,...,28 for the 32-layer 4B.

  python scripts/kk_audit.py --model runs/TFL-.../model --name TFL-s31 --out runs/kkaudit/TFL-s31
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))                 # the code lives next to this script, wherever the data is
if (Path.cwd() / "data/eval").exists() and not (ROOT / "data/eval").exists():   # HPC: data next to the repo clone
    ROOT = Path.cwd()
import torch  # noqa: E402

from cotdistill import interchange as ic  # noqa: E402
from cotdistill.student import FINAL_TEMPLATE, evaluate, load_model  # noqa: E402

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


def prepare(args, model, tok, raw: list[dict]) -> list[dict]:
    """Unpatched predictions on every distinct prompt, then one record per pair with token positions."""
    prompts = {}
    for p in raw:
        for k in ("base_prompt", "source_prompt"):
            prompts.setdefault(p[k], {"item_id": f"q{len(prompts)}", "prompt": p[k], "labels": p["labels"],
                                      "gold_label": p["labels"][p["base_gold" if k == "base_prompt" else "source_gold"]]})
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default=None, help="model name in the summary lines (default: the model path)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default="data/eval/kk_audit_pairs.jsonl")
    ap.add_argument("--layers", default="auto", help="comma list, or auto: 4, 8, ..., n_layers - 4")
    ap.add_argument("--conds", default=DEFAULT_CONDS)
    ap.add_argument("--sizes", default="4-8", help="n_people range kept")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
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
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16",
                    help="model weights on GPU (fp32 when the smoke's exchange_fidelity shows bf16 rounding)")
    args = ap.parse_args()
    args.name = args.name or args.model
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import transformers
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32 if dev != "cuda" or args.dtype == "fp32" else torch.bfloat16
    model = load_model(args.model, dt).to(dev)
    model.eval()
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    n_layers = len(ic.decoder_layers(model))
    layers = (list(range(4, n_layers - 3, 4)) if args.layers == "auto" else [int(x) for x in args.layers.split(",")])
    # hidden_states[n_layers] is the final-normed output, not a layer input: sites must be 0 <= L < n_layers
    assert layers and all(0 <= L < n_layers for L in layers), (layers, n_layers)
    args.layers = ",".join(map(str, layers))
    path = Path(args.pairs) if Path(args.pairs).exists() else ROOT / args.pairs
    lo, hi = (int(x) for x in args.sizes.split("-"))
    raw = [p for p in read(path) if lo <= p["n_people"] <= hi]
    if args.max_pairs:
        seen: dict = {}
        raw = [p for p in raw if seen.setdefault(p["family"], []).append(1) or len(seen[p["family"]]) <= args.max_pairs]
    pairs = prepare(args, model, tok, raw)
    head = {"model": args.name, "job": os.environ.get("SLURM_JOB_ID"), "layers": layers, "n_layers": n_layers,
            "pairs_file": args.pairs, "pairs_md5": hashlib.md5(path.read_bytes()).hexdigest(),
            "n": len(pairs), "base_acc": round(sum(p["base_ok"] for p in pairs) / len(pairs), 4),
            "source_acc": round(sum(p["source_ok"] for p in pairs) / len(pairs), 4),
            "jointly_correct": {f: sum(p["base_ok"] and p["source_ok"] for p in pairs if p["family"] == f)
                                for f in ("role", "inv", "stmt")}}
    print(json.dumps(head), flush=True)
    if args.check:
        res = dict(head, **check(args, model, tok, pairs, layers))
        (out / "check.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return
    emit(args, dict(head, cond="_head"))
    run(args, model, tok, pairs, layers)


if __name__ == "__main__":
    main()
