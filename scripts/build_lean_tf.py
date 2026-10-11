"""Lean F1 pilot (notes/next_round/lean/feasibility.md section 5; generator src/cotdistill/leangen.py): congruence chains
over integers, every option and every tree node certified by the Lean 4 kernel (core only, no Mathlib).

Writes
  data/eval/lean_f1_d{1..8}.jsonl                 --eval-per-depth (25) final items per depth (format (i), 5 options)
  data/student_lean_tf_pilot/train.jsonl          depths 1..--max-depth (4), --train-per-depth (500) each, with trees
  data/student_lean_tf_pilot/val.jsonl            same depths, --val-per-depth (50) each, own seeds
  data/student_lean_tf_pilot/local_deep.jsonl     the eval items of depths > --max-depth with their trees (--eval-local)
  data/student_lean_tf_pilot/eval_trees.jsonl     every eval item with its tree (node-vs-final consistency)
  data/student_lean_tf_pilot/summary.json         counts, md5s, certificates (kinds, failures, timing, peak RSS),
                                                  brute-force agreement, audit, tell battery, toolchain
  data/student_lean_tf_pilot/tokens.json          final / GFP-S node view token lengths (Qwen3.5 tokenizer)
  data/lean_f1_certs/batch_*.lean (+ .out)        the certificate files and Lean's JSON messages (kept, re-checkable)

Per (split, depth) slot the plan is drawn from a balanced schedule: drift (half the items carry 1-2 Lean-true off-target
options), n_on (1-2 on-target refuted options), non_z (about 40%: one option about another variable), n_irrelevant
(1-3 distractor premises); rank targets of the key (length, premise overlap) are uniform per slot. A surplus of slots
is generated and certified; an item is kept only if every one of its required certificates is certified (no error,
`#print axioms` within {propext, Classical.choice, Quot.sound}); per (split, depth) the first N/2 drift and N/2
non-drift certified items in slot order are kept. The key's grind certificate is a second certifier (reported, not
required). Lean: the pinned toolchain binary, at most --procs (2) processes, `-j 1 -M --memory --json`.

Gates (exit 1 on failure): every kept item fully certified; the one-period brute force (leangen.bf_status) agrees with
every certified label (options, nodes from their parents only, necessity); the text audit (premises parse back,
statement_block cut, every local view decides its node from its text alone, no node equals the key). The tell battery
is reported (summary["tells"]); pure surface tells should sit near chance (.20 final, .50 node pairs).

  CUDA_VISIBLE_DEVICES= ~/.venvs/sev/bin/python scripts/build_lean_tf.py
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import math
import os
import random
import re
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill import leangen as G  # noqa: E402
from cotdistill.teacher import leaks_answer  # noqa: E402
from cotdistill.treeplan import (LEAN_F1, LEAN_F1_QUESTION, local_context, node_depths,  # noqa: E402
                                 problem_no_options, statement_block, view_leak)

TOKENIZER = "results/weights/T1-Qwen3.5-2B-Base-student_v3-s2/model/tokenizer.json"
LEAN = Path.home() / ".elan/toolchains/leanprover--lean4---v4.29.1/bin/lean"
REQUIRED = lambda role: role != "key/grind"   # noqa: E731


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def templates() -> dict:
    """SUBQ_TEMPLATE / FINAL_TEMPLATE of cotdistill.student, read from its source (importing it needs torch)."""
    tree = ast.parse((ROOT / "src/cotdistill/student.py").read_text())
    return {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
            if isinstance(n, ast.Assign) and len(n.targets) == 1
            and getattr(n.targets[0], "id", "") in ("SUBQ_TEMPLATE", "FINAL_TEMPLATE")}


# ------------------------------------------------------------------------------------------------------ plan
def schedule(n: int, rng: random.Random) -> list[dict]:
    """n item plans, balanced: drift alternates by position (half each, the first one drawn at random), and n_drift,
    n_on, non_z, n_irrelevant and the key's letter (gold_pos) are cycled through balanced lists, shuffled within each
    drift arm."""
    first = rng.random() < 0.5
    drift = [(i % 2 == 0) == first for i in range(n)]
    out = []
    for arm in (True, False):
        k = sum(d == arm for d in drift)
        cyc = lambda vals: [vals[i % len(vals)] for i in range(k)]   # noqa: E731
        cols = [cyc([1, 2]), cyc([1, 2, 1, 2, 2, 1]), cyc([True, False, True, False, False]), cyc([1, 2, 3]),
                cyc([0, 1, 2, 3, 4])]
        for c in cols:
            rng.shuffle(c)
        out.append(iter([{"drift": arm, "n_drift": a if arm else 0, "n_on": b, "non_z": c, "n_irrelevant": d,
                          "gold_pos": e} for a, b, c, d, e in zip(*cols)]))
    return [next(out[0] if d else out[1]) for d in drift]


def slots(args) -> list[dict]:
    """(split, kind, depth, per, path, group) strata; each generates ceil(per * (1 + surplus)) + 2 slots."""
    edir, out = ROOT / args.eval_dir, ROOT / args.out
    st = [{"split": f"lean_f1_d{d}", "kind": "eval", "depth": d, "per": args.eval_per_depth,
           "path": edir / f"lean_f1_d{d}.jsonl", "group": f"lean_f1/d{d}"} for d in range(1, args.eval_depths + 1)]
    for split, per in (("val", args.val_per_depth), ("train", args.train_per_depth)):
        st += [{"split": split, "kind": "train", "depth": d, "per": per, "path": out / f"{split}.jsonl", "group": None}
               for d in range(1, args.max_depth + 1)]
    return st


def generate(args, strata: list[dict]) -> tuple[list[dict], dict]:
    cands, seen, stats = [], {}, collections.defaultdict(collections.Counter)
    t0 = time.time()
    for s in strata:
        n = math.ceil(s["per"] * (1 + args.surplus)) + 2
        plans = schedule(n, random.Random(f"{args.seed}-{s['split']}-{s['depth']}-plan"))
        for j, plan in enumerate(plans):
            for dup in range(20):
                it, rej = G.build_item(s["depth"], (args.seed, s["split"], s["depth"], j, dup), plan)
                stats[f"{s['split']}/d{s['depth']}"].update({f"reject/{k}": v for k, v in rej.items()})
                if it is None:
                    raise RuntimeError(f"{s['split']} d{s['depth']} slot {j}: no item ({dict(rej)})")
                key = G.canonical(it)
                if key not in seen:
                    break
                stats[f"{s['split']}/d{s['depth']}"]["duplicates_regenerated"] += 1
            else:
                raise RuntimeError(f"{s['split']} d{s['depth']} slot {j}: duplicates")
            seen[key] = (s["split"], s["depth"], j)
            it["_stratum"], it["_slot"] = s, j
            cands.append(it)
        print(f"generated {s['split']} d{s['depth']}: {n} slots, {time.time() - t0:.0f}s, rss {rss_mb()} MB",
              flush=True)
    return cands, {k: dict(v) for k, v in stats.items()}


# ------------------------------------------------------------------------------------------------------ Lean
def write_cert_files(cands: list[dict], cdir: Path, per_file: int) -> list[tuple[Path, dict]]:
    cdir.mkdir(parents=True, exist_ok=True)
    for p in cdir.glob("batch_*"):
        p.unlink()
    entries, k = [], 0
    for it in cands:
        for c in it["certificates"]:
            c["name"] = f"c{k:06d}"
            entries.append((c["name"], c))
            k += 1
    files = []
    for b in range(0, len(entries), per_file):
        text, where = G.lean_file(entries[b:b + per_file])
        path = cdir / f"batch_{b // per_file:04d}.lean"
        path.write_text(text)
        files.append((path, where))
    return files


def run_lean(files: list[tuple[Path, dict]], lean: Path, procs: int, memory: int, timeout: int) -> tuple[dict, dict]:
    """Run `lean -j 1 -M memory --json` on every file, at most procs at once; per-declaration results and per-file
    timing / peak RSS (os.wait4 rusage of each child)."""
    res, timing, queue, running = {}, {}, list(files), {}
    env = {k: v for k, v in os.environ.items() if not k.startswith("LEAN")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    t_all = time.time()
    while queue or running:
        while queue and len(running) < procs:
            path, where = queue.pop(0)
            out = open(path.with_suffix(".out"), "w")
            p = subprocess.Popen([str(lean), "-j", "1", "-M", str(memory), "--json", path.name], cwd=path.parent,
                                 stdout=out, stderr=subprocess.STDOUT, env=env)
            running[p.pid] = (p, path, where, time.time(), out)
        time.sleep(0.05)
        for pid in list(running):
            p, path, where, t0, out = running[pid]
            got, status, ru = os.wait4(pid, os.WNOHANG)
            if got == 0:
                if time.time() - t0 > timeout:
                    os.kill(pid, signal.SIGKILL)
                continue
            p.returncode = os.waitstatus_to_exitcode(status)
            out.close()
            r, other = G.parse_lean_output(path.with_suffix(".out").read_text(), where)
            res.update(r)
            timing[path.name] = {"wall_s": round(time.time() - t0, 2), "user_s": round(ru.ru_utime, 2),
                                 "sys_s": round(ru.ru_stime, 2), "maxrss_mb": round(ru.ru_maxrss / 1024),
                                 "rc": p.returncode, "decls": len(set(where.values())), "other_lines": other[:5]}
            del running[pid]
            done = len(timing)
            if done % 20 == 0 or not (queue or running):
                print(f"lean {done}/{len(files)} files, {time.time() - t_all:.0f}s", flush=True)
    return res, timing


def empty_file_time(lean: Path, cdir: Path) -> float:
    p = cdir / "empty_probe.lean"
    p.write_text(G.LEAN_HEADER)
    t0 = time.time()
    subprocess.run([str(lean), "-j", "1", "--json", p.name], cwd=cdir, capture_output=True,
                   env={k: v for k, v in os.environ.items() if not k.startswith("LEAN")})
    p.unlink()
    return round(time.time() - t0, 2)


# ------------------------------------------------------------------------------------------------------ checks
def expected_status(role: str) -> str:
    return {"key/omega": "entailed", "key/grind": "entailed", "drift_true/omega": "entailed",
            "refuted/omega": "refuted", "independent/holds": "independent", "independent/fails": "independent",
            "node/yes": "entailed", "node/no": "refuted", "final/parents": "entailed",
            "necessary": "independent"}[role]


def brute_force(it: dict) -> list[tuple]:
    """Every label of the item re-decided by brute force over one period: [(role, target, expected, got)]."""
    hyps = [G.strip_meta(h) for h in it["program"]["hyps"]]
    key = it["program"]["key"]
    out = []
    for L, meta in it["option_meta"].items():
        claim = {"kind": "cong", "var": meta["var"], "m": meta["m"], "r": meta["r"]}
        want = {"key": "entailed", "drift_true": "entailed", "refuted": "refuted",
                "independent": "independent"}[it["option_kinds"][L]]
        out.append(("option/" + it["option_kinds"][L], L, want, G.bf_status(hyps, claim)))
    hyp = {h["id"]: G.strip_meta(h) for h in it["program"]["hyps"]}
    by = {n["id"]: n for n in it["subqs"]}
    for n in it["subqs"]:
        par = [G.strip_meta(hyp[n["id"].rstrip("x")])] if n["type"] != "derive" else \
              [by[d]["claim"] for d in n["depends_on"]]
        claim = n["claim"]
        want = "entailed" if n["truth"] else "refuted"
        out.append((f"node/{n['type']}", n["id"], want, G.bf_status(par, claim)))
        out.append((f"node_full/{n['type']}", n["id"], want, G.bf_status(hyps, claim)))   # true of the whole problem
    for hid in it["program"]["critical"]:
        rest = [G.strip_meta(h) for h in it["program"]["hyps"] if h["id"] != hid]
        out.append(("necessary", hid, "independent", G.bf_status(rest, key)))
    return out


def safe_brute_force(it: dict) -> list[tuple]:
    try:
        return brute_force(it)
    except (OverflowError, ValueError) as e:
        return [("brute_force_error", type(e).__name__, "decided", str(e)[:80])]


def audit_item(it: dict, rng: random.Random, a: collections.Counter, fails: list) -> None:
    iid = it["item_id"]
    fail = lambda msg: fails.append((iid, msg))  # noqa: E731
    prog = it["program"]
    hyps = prog["hyps"]
    body = G.body_text(hyps)
    block = G.PREAMBLE + "\n\n" + body
    q = G.QUESTION.format(v=prog["key"]["var"], m=prog["key"]["m"])
    try:
        if statement_block(it) != block or problem_no_options(it) != block + "\n\n" + q:
            fail("statement_block / problem_no_options cut differs")
    except ValueError as e:
        fail(f"statement_block: {e}")
    if not re.fullmatch(LEAN_F1_QUESTION, q) or LEAN_F1_QUESTION != G.FINAL_RE:
        fail("final question does not match LEAN_F1_QUESTION")
    if [G.strip_meta(h) for h in G.parse_body(body)] != [G.strip_meta(h) for h in hyps]:
        fail("premises do not parse back")
    if "Question:" in block:
        fail("statement block contains a question")
    for L, t in it["options"].items():
        o = G.parse_option(t)
        if o is None or (o["var"], o["m"], o["r"]) != tuple(it["option_meta"][L][k] for k in ("var", "m", "r")):
            fail(f"option {L} does not parse back")
    kinds = collections.Counter(it["option_kinds"].values())
    if kinds["key"] != 1 or it["option_kinds"][it["gold_label"]] != "key":
        fail("not exactly one key")
    if not kinds["refuted"] or not kinds["independent"]:
        a["final_missing_refuted_or_independent"] += 1
    key = (prog["key"]["var"], prog["key"]["m"], prog["key"]["r"])
    nodes = it.get("subqs") or []
    depth = node_depths(nodes)
    by = {n["id"]: n for n in nodes}
    for n in nodes:
        a[f"nodes/{n['type']}"] += 1
        a[f"yes/{n['type']}"] += n["truth"]
        if n["claim"]["kind"] == "cong" and G.claim_key(n["claim"]) == key:
            fail(f"{n['id']}: node asks the key")
        if depth[n["id"]] != n["node_depth"]:
            fail(f"{n['id']}: node_depth {n['node_depth']} != graph depth {depth[n['id']]}")
        par = [by[d] for d in n["depends_on"] if d in by]
        if len(par) != len(n["depends_on"]) or not all(x["truth"] for x in par):
            fail(f"{n['id']}: parent missing or not Yes")
        stated = [] if n.get("local_lines") else par
        if view_leak(it, n, stated):
            a["view_leak"] += 1
        a["leaks_answer_generic"] += leaks_answer(n["question"], it)
        ctx = local_context(it, n, stated, rng=rng)
        v = G.solve_view(ctx, n["question"])
        a[f"local_views/{n['type']}"] += 1
        if v is None or v != n["truth"]:
            fail(f"{n['id']}: local view does not decide it ({v} vs {n['truth']})")
        else:
            a[f"local_views_sufficient/{n['type']}"] += 1
        if n.get("local_lines") and n["local_lines"][0] not in body.split("\n"):
            fail(f"{n['id']}: leaf line is not a premise line")
        a[f"node_depth/{n['node_depth']}"] += 1
    opts = {(m["var"], m["m"], m["r"]) for m in it["option_meta"].values()}
    for n in nodes:
        if n["claim"]["kind"] == "cong" and G.claim_key(n["claim"]) in opts and not (n["type"] == "read" and n["truth"]):
            fail(f"{n['id']}: node claim is an option")


# ------------------------------------------------------------------------------------------------------ tells
def _pick(mask: list[bool], gold: int) -> float:
    """Expected accuracy of picking uniformly among the flagged options (none flagged: uniform over all)."""
    idx = [i for i, f in enumerate(mask) if f] or list(range(len(mask)))
    return (gold in idx) / len(idx)


def final_tells(items: list[dict]) -> dict:
    rows = collections.defaultdict(list)
    for it in items:
        L = list(it["options"])
        gold = L.index(it["gold_label"])
        meta = [it["option_meta"][x] for x in L]
        kinds = [it["option_kinds"][x] for x in L]
        texts = [it["options"][x] for x in L]
        ptoks = set(re.findall(r"[a-z]+|\d+", G.body_text(it["program"]["hyps"]).lower()))
        ov = [G.overlap(t, ptoks) for t in texts]
        ln = [len(t) for t in texts]
        ms = [m["m"] for m in meta]
        fc = [G.form_class(m) for m in meta]
        cnt = collections.Counter(fc)
        odd = [cnt[f] == 1 for f in fc] if len(cnt) == 2 and min(cnt.values()) == 1 else [True] * 5
        z, M = it["program"]["key"]["var"], it["program"]["key"]["m"]
        lean_true = [k in ("key", "drift_true") for k in kinds]
        r = {"longest": _pick([x == max(ln) for x in ln], gold), "shortest": _pick([x == min(ln) for x in ln], gold),
             "overlap_high": _pick([x == max(ov) for x in ov], gold), "overlap_low": _pick([x == min(ov) for x in ov], gold),
             "form_odd_one_out": _pick(odd, gold),
             "modulus_smallest": _pick([x == min(ms) for x in ms], gold),
             "modulus_largest": _pick([x == max(ms) for x in ms], gold),
             "remainder_zero": _pick([m["r"] == 0 for m in meta], gold),
             "mentions_asked_variable": _pick([m["var"] == z for m in meta], gold),
             "question_tracking_var_and_divisor": _pick([m["var"] == z and m["m"] == M for m in meta], gold),
             "truth_only_any_lean_true": _pick(lean_true, gold),
             "truth_and_variable": _pick([t and m["var"] == z for t, m in zip(lean_true, meta)], gold)}
        for name in SURFACE_EXTREMES:                       # elimination: pick among the options NOT flagged
            flags = {"longest": [x == max(ln) for x in ln], "shortest": [x == min(ln) for x in ln],
                     "overlap_high": [x == max(ov) for x in ov], "overlap_low": [x == min(ov) for x in ov],
                     "form_odd_one_out": [o and not all(odd) for o in odd],
                     "modulus_smallest": [x == min(ms) for x in ms], "modulus_largest": [x == max(ms) for x in ms],
                     "remainder_zero": [m["r"] == 0 for m in meta]}[name]
            r[f"avoid_{name}"] = _pick([not f for f in flags], gold)
        for p_ in range(5):
            r[f"position_{'ABCDE'[p_]}"] = float(gold == p_)
        for arm in ("all", "drift" if it["drift"] else "no_drift"):
            rows[arm].append(r)
    out = {}
    for arm, rs in rows.items():
        out[arm] = {"n": len(rs), **{k: round(sum(x[k] for x in rs) / len(rs), 3) for k in rs[0]}}
        surf = {k: v for k, v in out[arm].items() if k in SURFACE_EXTREMES or k.startswith("avoid_")
                or k.startswith("position_") or k == "mentions_asked_variable"}
        best = max(surf, key=surf.get)
        out[arm]["best_surface_oracle"] = [best, surf[best]]
    return out


# pure surface oracles (gate: the best of them, incl. "avoid" and position, within chance + .05 = .25); the
# question-tracking and truth-only oracles are reported, not gated (they measure the two skills the design separates)
SURFACE_EXTREMES = ("longest", "shortest", "overlap_high", "overlap_low", "form_odd_one_out", "modulus_smallest",
                    "modulus_largest", "remainder_zero")
SURFACE_GATE = 0.25


def node_tells(items: list[dict]) -> dict:
    """Yes/No twin pairs of the GFP-S nodes: which of the two is Yes, picked by surface features (chance .50), and
    single-feature stumps over all node questions (max(acc, 1 - acc))."""
    pairs = collections.defaultdict(list)
    stumps = collections.defaultdict(list)
    for it in items:
        ptoks = set(re.findall(r"[a-z]+|\d+", G.body_text(it["program"]["hyps"]).lower()))
        pnums = set(re.findall(r"\d+", G.body_text(it["program"]["hyps"])))
        by = {n["id"]: n for n in it["subqs"]}
        for n in it["subqs"]:
            t = n["type"]
            if n["claim"]["kind"] == "cong":
                stumps[f"{t}/residue_in_premises"].append((n["truth"], str(n["claim"]["r"]) in pnums))
                stumps[f"{t}/residue_zero"].append((n["truth"], n["claim"]["r"] == 0))
            else:
                stumps[f"{t}/has_minus"].append((n["truth"], " minus " in n["question"]))
                stumps[f"{t}/constant_in_premises"].append((n["truth"], str(abs(n["claim"]["c"])) in pnums))
            if not n["truth"]:
                continue
            tw = by[n["id"] + "x"]
            qs = [n["question"], tw["question"]]
            ov = [G.overlap(q, ptoks) for q in qs]
            ln = [len(q) for q in qs]
            for name, mask in (("longer", [x == max(ln) for x in ln]), ("shorter", [x == min(ln) for x in ln]),
                               ("overlap_high", [x == max(ov) for x in ov]), ("overlap_low", [x == min(ov) for x in ov])):
                for k in (t, "all"):
                    pairs[f"{k}/{name}"].append(_pick(mask, 0))
    out = {"pairs": {k: {"n": len(v), "acc": round(sum(v) / len(v), 3)} for k, v in sorted(pairs.items())},
           "stumps": {}}
    for k, v in sorted(stumps.items()):
        acc = sum(a == b for a, b in v) / len(v)
        out["stumps"][k] = {"n": len(v), "acc": round(max(acc, 1 - acc), 3)}
    return out


# ------------------------------------------------------------------------------------------------------ tokens
def token_report(paths: list[Path], max_len: int = 1536) -> dict:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / TOKENIZER))
    T = templates()

    def summ(xs):
        xs = sorted(xs)
        return {"n": len(xs), "mean": round(sum(xs) / len(xs), 1), "p50": xs[len(xs) // 2],
                "p99": xs[min(len(xs) - 1, int(0.99 * len(xs)))], "max": xs[-1],
                "share_le_max_len": round(sum(x <= max_len for x in xs) / len(xs), 4)} if xs else {}
    rep = {"tokenizer": TOKENIZER, "max_len": max_len,
           "views": "final: FINAL_TEMPLATE(prompt); GFP-S node: SUBQ_TEMPLATE(statement_block, node question), every "
                    "node (--tree-cap 0); no special tokens", "files": {}}
    for p in paths:
        fin, nod = [], []
        for line in open(p):
            it = json.loads(line)
            texts = [T["FINAL_TEMPLATE"].format(problem=it["prompt"])]
            if it.get("subqs"):
                ctx = statement_block(it)
                texts += [T["SUBQ_TEMPLATE"].format(problem=ctx, question=n["question"], a="Yes", b="No")
                          for n in it["subqs"]]
            lens = [len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False)]
            fin.append(lens[0])
            nod += lens[1:]
        rep["files"][rel(p)] = {"final_view": summ(fin), "gfps_node_view": summ(nod), "items": len(fin),
                                "tokens_final": sum(fin), "tokens_nodes": sum(nod), "views": len(nod)}
    return rep


# ------------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/student_lean_tf_pilot")
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--certs-dir", default="data/lean_f1_certs")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--eval-depths", type=int, default=8)
    ap.add_argument("--train-per-depth", type=int, default=500)
    ap.add_argument("--val-per-depth", type=int, default=50)
    ap.add_argument("--eval-per-depth", type=int, default=25)
    ap.add_argument("--surplus", type=float, default=0.15, help="extra slots per stratum (certificate failures)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-file", type=int, default=200)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--memory", type=int, default=1400, help="lean -M (MB)")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--lean", default=str(LEAN))
    ap.add_argument("--no-tokens", action="store_true")
    args = ap.parse_args(argv)
    if args.procs > 2:
        raise SystemExit("--procs: at most 2 Lean processes (7 GB laptop)")
    out, cdir, lean = ROOT / args.out, ROOT / args.certs_dir, Path(args.lean)
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / args.eval_dir).mkdir(parents=True, exist_ok=True)
    version = subprocess.run([str(lean), "--version"], capture_output=True, text=True).stdout.strip()
    t_start = time.time()
    strata = slots(args)
    cands, gen_stats = generate(args, strata)
    t_gen = time.time() - t_start
    files = write_cert_files(cands, cdir, args.per_file)
    n_certs = sum(len(it["certificates"]) for it in cands)
    print(f"{len(cands)} candidates, {n_certs} certificates in {len(files)} files", flush=True)
    t0 = time.time()
    res, timing = run_lean(files, lean, args.procs, args.memory, args.timeout)
    t_lean = time.time() - t0
    t_empty = empty_file_time(lean, cdir)
    # certification
    role_c = collections.defaultdict(collections.Counter)
    fail_ex = []
    for it in cands:
        it["_ok"] = True
        for c in it["certificates"]:
            r = res.get(c["name"])
            c["certified"] = G.certified(r)
            role_c[c["role"]]["certified" if c["certified"] else "failed"] += 1
            if not c["certified"]:
                if len(fail_ex) < 20:
                    fail_ex.append({"name": c["name"], "role": c["role"], "item_slot": [it["_stratum"]["split"],
                                    it["_stratum"]["depth"], it["_slot"]], "errors": (r or {}).get("errors", [])[:1],
                                    "axioms": (r or {}).get("axioms")})
                if REQUIRED(c["role"]):
                    it["_ok"] = False
    # selection
    kept, sel_stats = collections.defaultdict(list), {}
    for s in strata:
        pool = [it for it in cands if it["_stratum"] is s]
        ok = [it for it in pool if it["_ok"]]
        nd = (s["per"] + (s["depth"] % 2)) // 2 if s["kind"] == "eval" else s["per"] // 2
        take = [it for it in ok if it["drift"]][:nd] + [it for it in ok if not it["drift"]][:s["per"] - nd]
        if len(take) != s["per"]:
            raise SystemExit(f"{s['split']} d{s['depth']}: only {len(take)} certified items for {s['per']} "
                             f"(drift {sum(it['drift'] for it in ok)}, none {sum(not it['drift'] for it in ok)}); "
                             "raise --surplus")
        take.sort(key=lambda it: it["_slot"])
        sel_stats[f"{s['split']}/d{s['depth']}"] = {"slots": len(pool), "certified": len(ok), "kept": len(take),
                                                   "dropped_uncertified": len(pool) - len(ok)}
        kept[s["split"]] += [(s, it) for it in take]
        for it in take:
            it["_kept"] = True
    # brute force over every label of every candidate (kept: gate; dropped: diagnosis)
    bf = collections.Counter()
    bf_bad, dropped_true = [], collections.Counter()
    t0 = time.time()
    for it in cands:
        res_bf = safe_brute_force(it)
        tag = "kept" if it.get("_kept") else "dropped"
        for role, tgt, want, got in res_bf:
            bf[f"{tag}/{role}/{'agree' if want == got else 'disagree'}"] += 1
            if tag == "kept" and want != got and len(bf_bad) < 20:
                bf_bad.append([it["_stratum"]["split"], it["_slot"], role, tgt, want, got])
        if not it["_ok"]:
            for c in it["certificates"]:
                if not c["certified"] and REQUIRED(c["role"]):
                    dropped_true[c["role"]] += 1
    t_bf = time.time() - t0
    # outputs
    fails, audit = [], collections.Counter()
    summary = {"args": vars(args), "toolchain": {"lean": version, "binary": str(lean)}, "files": {}, "md5": {}}
    deep, all_eval = [], []
    finals = collections.defaultdict(list)
    for split in sorted(kept):
        group_by_path = collections.defaultdict(list)
        for s, it in kept[split]:
            group_by_path[s["path"]].append((s, it))
        for path, rows in sorted(group_by_path.items()):
            with open(path, "w") as f:
                counts = collections.Counter()
                for j, (s, it) in enumerate(rows):
                    d = s["depth"]
                    tag = {"train": "tr", "val": "va"}.get(split, "ev")
                    iid = f"lean_f1-{tag}-d{d}-{sum(1 for x, _ in rows[:j] if x['depth'] == d):05d}"
                    item = {k: v for k, v in it.items() if not k.startswith("_")}
                    rec = G.finalize(item, iid, s["kind"], s["group"])
                    arng = random.Random(f"{args.seed}-audit-{iid}")
                    audit_item(dict(rec, subqs=item["subqs"], local_preamble=G.PREAMBLE), arng, audit, fails)
                    if not all(c["certified"] for c in it["certificates"] if REQUIRED(c["role"])):
                        fails.append((iid, "kept item not fully certified"))
                    f.write(json.dumps(rec) + "\n")
                    counts[d] += 1
                    full = G.finalize(item, iid, "train")
                    full.update({k: rec[k] for k in ("group", "type", "label_order", "gold_probs") if k in rec})
                    finals[split].append(full)
                    if s["kind"] == "eval":
                        all_eval.append(full)
                        if d > args.max_depth:
                            deep.append(full)
            summary["files"][rel(path)] = {"items": sum(counts.values()), "depth": dict(sorted(counts.items()))}
            summary["md5"][rel(path)] = md5(path)
    for name, rows in (("local_deep.jsonl", deep), ("eval_trees.jsonl", all_eval)):
        with open(out / name, "w") as f:
            for it in rows:
                f.write(json.dumps(it) + "\n")
        summary["md5"][rel(out / name)] = md5(out / name)
        summary["files"][rel(out / name)] = {"items": len(rows)}
    # stats per split
    per_split = {}
    for split, rows in finals.items():
        nt, nd_, ok_ = collections.Counter(), collections.Counter(), collections.Counter()
        for it in rows:
            for n in it["subqs"]:
                nt[f"{n['type']}/{'yes' if n['truth'] else 'no'}"] += 1
                nd_[n["node_depth"]] += 1
            ok_.update(it["option_kinds"].values())
            ok_.update("drift_subtype_" + x for x in it["drift_subtypes"])
            ok_["drift_items"] += it["drift"]
            ok_[f"n_on_target_{it['n_on_target']}"] += 1
            ok_["non_z_items"] += any(m["var"] != it["program"]["key"]["var"] for m in it["option_meta"].values())
        n_nodes = sum(nt.values())
        per_split[split] = {"items": len(rows), "nodes": n_nodes, "nodes_per_item": round(n_nodes / len(rows), 2),
                            "node_types": dict(sorted(nt.items())), "node_depth": dict(sorted(nd_.items())),
                            "options": dict(sorted(ok_.items())),
                            "certificates": sum(len(it["certificates"]) for it in rows)}
    summary["splits"] = per_split
    eval_items = [it for s in finals if s.startswith("lean_f1_d") for it in finals[s]]
    summary["tells"] = {"final_eval": final_tells(eval_items), "final_train": final_tells(finals["train"]),
                        "final_val": final_tells(finals["val"]),
                        "nodes_train": node_tells(finals["train"]), "nodes_eval": node_tells(eval_items)}
    walls = [t["wall_s"] for t in timing.values()]
    cpu = sum(t["user_s"] + t["sys_s"] for t in timing.values())
    summary["certificates"] = {
        "candidates": len(cands), "certificates": n_certs, "files": len(files), "per_file": args.per_file,
        "by_role": {k: dict(v) for k, v in sorted(role_c.items())},
        "required_failed_by_role": dict(dropped_true), "failure_examples": fail_ex,
        "selection": sel_stats,
        "timing": {"generate_s": round(t_gen, 1), "lean_wall_s": round(t_lean, 1), "lean_cpu_s": round(cpu, 1),
                   "procs": args.procs, "cpu_s_per_certificate": round(cpu / max(1, n_certs), 4),
                   "wall_s_per_certificate": round(t_lean / max(1, n_certs), 4),
                   "empty_file_s": t_empty, "file_wall_s": {"min": min(walls), "max": max(walls),
                                                             "mean": round(sum(walls) / len(walls), 2)},
                   "peak_rss_mb": max(t["maxrss_mb"] for t in timing.values()),
                   "nonzero_rc_files": sum(t["rc"] != 0 for t in timing.values()),
                   "brute_force_s": round(t_bf, 1)},
        "per_file": timing}
    summary["brute_force"] = {"counts": dict(sorted(bf.items())), "kept_disagreements": bf_bad,
                              "kept_agree": sum(v for k, v in bf.items() if k.startswith("kept/") and k.endswith("/agree")),
                              "kept_disagree": sum(v for k, v in bf.items() if k.startswith("kept/")
                                                   and k.endswith("/disagree"))}
    summary["generation"] = gen_stats
    summary["audit"] = dict(sorted(audit.items()))
    summary["audit"]["failures"] = len(fails)
    summary["failure_examples"] = fails[:20]
    summary["fields"] = {
        "item": ["source", "prompt", "labels", "gold_label", "options", "option_kinds (key / refuted / independent / "
                 "drift_true)", "option_meta (var, m, r, target on/off, tag off_target + subtype a/b/c)", "depth",
                 "drift", "n_drift", "drift_subtypes", "n_on_target", "n_irrelevant", "steps", "targets", "plan",
                 "program (hyps, key, critical, final_modulus)", "formal (statement, hyps, options: Lean)",
                 "certificates (role, target, tactic, body, name, certified, witness)",
                 "teacher / subqs / has_tree / tree_source / local_preamble / tree_fix (train, val, local_deep, "
                 "eval_trees)", "group / type / label_order / gold_probs (eval)"],
        "subq": ["id", "type (read/parse/derive)", "kind", "level", "question", "opposite", "truth", "p_cot",
                 "p_fresh", "depends_on", "node_depth", "initial", "status", "local_lines (leaves)", "claim", "lean"]}
    if not args.no_tokens:
        tok = token_report([ROOT / args.out / "train.jsonl", ROOT / args.out / "val.jsonl"]
                           + [ROOT / args.eval_dir / f"lean_f1_d{d}.jsonl" for d in range(1, args.eval_depths + 1)])
        (out / "tokens.json").write_text(json.dumps(tok, indent=1))
        summary["tokens"] = {k: v["final_view"] for k, v in tok["files"].items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("splits", "brute_force", "audit")}, indent=1)[:6000])
    print(json.dumps(summary["certificates"]["timing"], indent=1))
    print(json.dumps(summary["tells"]["final_eval"], indent=1))
    print(f"total {time.time() - t_start:.0f}s, rss {rss_mb()} MB")
    pooled = final_tells(eval_items + finals["train"] + finals["val"])["all"]
    nt = [summary["tells"]["nodes_train"], summary["tells"]["nodes_eval"]]
    node_max = max(v["acc"] for t in nt for part in ("pairs", "stumps") for v in t[part].values())
    summary["tells"]["final_pooled"] = pooled
    summary["gates"] = {
        "all_kept_certified": not any(m == "kept item not fully certified" for _, m in fails),
        "brute_force_agrees": summary["brute_force"]["kept_disagree"] == 0,
        "text_audit": not fails,
        "final_surface_tells_le_.25": {"pass": pooled["best_surface_oracle"][1] <= SURFACE_GATE,
                                       "best": pooled["best_surface_oracle"],
                                       "eval_best": summary["tells"]["final_eval"]["all"]["best_surface_oracle"]},
        "node_tells_le_.55": {"pass": node_max <= 0.55, "max": node_max}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["gates"], indent=1))
    gate = not fails and summary["brute_force"]["kept_disagree"] == 0
    if not gate:
        print("GATE FAILED", fails[:10], bf_bad[:10])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
