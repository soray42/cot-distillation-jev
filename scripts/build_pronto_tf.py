"""Tree Forcing data for the PrOntoQA-OOD domain (source "pronto", src/cotdistill/pronto.py): a depth-extrapolation split
with exact proof trees, built like data/student_numgen_tf so the same arms run unchanged.

Writes (counts per hop k; answers True/False 50/50 and query polarity 50/50 within every k, crossed):
  data/student_pronto_tf/train.jsonl   hops 1..--max-hops (default 4), --train-per-hop each (2,500 -> 10,000)
  data/student_pronto_tf/val.jsonl     same hops, --val-per-hop each (100 -> 400)
  data/eval/pronto_d{1..8}.jsonl       --eval-per-hop each (400); d5-d8 beyond the training range
  data/eval/pronto_wide.jsonl          hops 1-4, 100 each, width 2: two extra chain pairs and side branches at p .5
                                       (about twice the sentences; in-range depth, out-of-range distractor width)
  data/student_pronto_tf/local_deep.jsonl  the first 100 items of pronto_d5..d8 with their trees (--eval-local probe)
  data/student_pronto_tf/summary.json  counts, md5s, node stats, hop histograms, audit, cue stumps, baselines, tokens
Train / val / local_deep items carry subqs (pronto.proof_tree), teacher (one-hot gold), local_preamble, tree_fix; eval
items carry group "pronto/d<k>", type "choice", label_order, gold_probs null (as data/eval/numgen_d*.jsonl). Every item
keeps its theory (pronto.make_problem), so trees can be rebuilt. No two items of any file share a logical problem
(pronto.canonical: the rule set, facts and query; templates and order ignored).

Audit (exit 1 on any failure), every item of every file:
  - the prompt is preamble + body + the final question + options; treeplan.statement_block / problem_no_options cut it
    there (FINAL_QUESTION["pronto"]); the body parses back (pronto.parse_sentence) into exactly the theory's rules/facts
  - deductive closure of the parsed body (pronto.closure): consistent; the query literal or its negation derived, with
    the gold answer; its shortest proof exactly k rule applications; E derives no concept of F's chains and v.v.
  - every node: Yes = its statement in the closure (memberships also at step = node_depth) or the stated rule; No =
    the negation in the closure or the opposite-polarity rule stated; parents all Yes; no node names the query entity
    together with the query property (pronto.node_leak); view_leak and teacher.leaks_answer counted (reported)
  - every derived node re-solved from its rendered local view (treeplan.local_context) TEXT alone; every leaf from its
    line alone
Reports: trivial baselines per eval file (majority, always True), single-feature stumps for the final answer
(final_cues: query negated, the query predicate's rule that comes first in the text, the one whose subject is mentioned
more often, a polarity-blind one-hop lookahead), a hashed bag-of-words logistic regression (entities / concepts / query
predicate normalised; fit on train and scored per file, and 5-fold held-out within train by hop), stumps for the node
targets in the GFPS (statements) view (node_cues), and prompt token lengths with the Qwen3.5 tokenizer.

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python3 scripts/build_pronto_tf.py
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill import pronto as P  # noqa: E402
from cotdistill.teacher import leaks_answer  # noqa: E402
from cotdistill.treeplan import (FINAL_QUESTION, LOCAL_HEADER, full_tree_plan, local_context, node_depths,  # noqa: E402
                                 problem_no_options, statement_block, view_leak)

TOKENIZER = "results/weights/T1-Qwen3.5-2B-Base-student_v3-s2/model/tokenizer.json"
LOCAL_DEEP_PER_DEPTH = 100
_FACTLINE = re.compile(r"^Q: (.+) A: (Yes|No)$")


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


def schedule(n: int, rng: random.Random) -> list[tuple[bool, bool]]:
    """(answer, query negated) for n items: the four combinations equally often (n % 4 spread at random)."""
    combos = [(a, q) for a in (True, False) for q in (False, True)]
    out = combos * (n // 4) + rng.sample(combos, n % 4)
    rng.shuffle(out)
    return out


# ------------------------------------------------------------------------------------------------------ audit
def solve_view(ctx: str, question: str):
    """Answer a node question from a rendered local view's text alone: leaf views (preamble + one sentence) and
    derived views (preamble + "Known results:" Q/A lines). True / False, or None if the text does not decide it."""
    pre, _, rest = ctx.partition("\n\n")
    if pre != P.LOCAL_PREAMBLE:
        return None
    facts, rules = [], []
    if rest.startswith(LOCAL_HEADER + "\n"):
        for line in rest[len(LOCAL_HEADER) + 1:].split("\n"):
            m = _FACTLINE.match(line)
            if not m or m.group(2) != "Yes":
                return None
            mem, rule = P.parse_member_q(m.group(1)), P.parse_rule_q(m.group(1))
            if mem:
                facts.append(mem)
            elif rule:
                rules.append(rule)
            else:
                return None
    else:
        p = P.parse_sentence(rest)
        if p is None:
            return None
        (facts if p[0] == "fact" else rules).append(p[1])
    mem, rule = P.parse_member_q(question), P.parse_rule_q(question)
    if rule:                          # a rule question: the stated rule, or its opposite polarity
        if rule in rules:
            return True
        if rule[:3] + (not rule[3],) in rules:
            return False
        return None
    if mem:
        try:
            cl = P.closure(facts, rules)
        except ValueError:
            return None
        if mem in cl:
            return not cl[mem][0]
    return None


def node_truth(n: dict, cl: dict, stated_rules: set):
    """(truth from the closure / stated rules, the step at which a membership is derived)."""
    mem, rule = P.parse_member_q(n["question"]), P.parse_rule_q(n["question"])
    if mem:
        if mem not in cl:
            return None, None
        return (not cl[mem][0]), cl[mem][1]
    if rule:
        if rule in stated_rules:
            return True, 0
        if rule[:3] + (not rule[3],) in stated_rules:
            return False, 0
    return None, None


def audit_item(it: dict, nodes: list[dict], rng: random.Random, a: collections.Counter, fails: list) -> None:
    th, iid = it["theory"], it["item_id"]
    k = th["hops"]
    fail = lambda msg: fails.append((iid, msg))  # noqa: E731
    body = P.body_text(th)
    block = P.LOCAL_PREAMBLE + "\n\n" + body
    try:
        if statement_block(it) != block or problem_no_options(it) != block + "\n\n" + P.QUESTION.format(
                query=P.query_text(th)):
            fail("statement_block / problem_no_options cut differs")
    except ValueError as e:
        fail(f"statement_block: {e}")
    if not re.fullmatch(FINAL_QUESTION[P.SOURCE], P.QUESTION.format(query=P.query_text(th))):
        fail("final question does not match FINAL_QUESTION")
    if P.query_text(th) in block or "Question:" in block or re.search(r"\d", block):
        fail("statement block states the query, a question or a digit")
    facts, rules = P.parse_body(body)
    if sorted(map(tuple, th["facts"])) != sorted(facts) or sorted(tuple(r[:4]) for r in th["rules"]) != sorted(rules):
        fail("body does not parse back into the theory")
    try:
        cl = P.closure(facts, rules)
    except ValueError as e:
        fail(f"inconsistent: {e}")
        return
    e, prop = th["entity"], th["property"]
    lit = cl.get((e, prop))
    if lit is None:
        fail("query undecided")
        return
    if (lit[0] == th["query_neg"]) != th["answer"] or it["gold_label"] != ("A" if th["answer"] else "B"):
        fail("answer differs from the closure")
    if lit[1] != k:
        fail(f"shortest proof {lit[1]} != hops {k}")
    a[f"proof_steps/{lit[1]}"] += 1
    (cmain, dmain) = th["pairs"][0]
    for ent, others in ((e, {c for _, b in th["pairs"] for c in b}), (th["other"], {c for a_, _ in th["pairs"]
                                                                                  for c in a_})):
        if any(cl.get((ent, c), (True,))[0] is False for c in others):
            fail(f"{ent} derives a concept of the other entity's chains")
    stated = set(rules)
    by = {n["id"]: n for n in nodes}
    depth = node_depths(nodes)
    for n in nodes:
        a[f"nodes/{n['type']}"] += 1
        a[f"yes/{n['type']}"] += n["truth"]
        if depth[n["id"]] != n["node_depth"]:
            fail(f"{n['id']}: node_depth {n['node_depth']} != {depth[n['id']]}")
        t, step = node_truth(n, cl, stated)
        if t is None or t != n["truth"]:
            fail(f"{n['id']}: truth {n['truth']} but closure says {t}")
        if n["type"] in ("read", "derive") and step != n["node_depth"]:
            fail(f"{n['id']}: derived at step {step}, node_depth {n['node_depth']}")
        if P.node_leak(n["question"], th):
            a["leak_node_question"] += 1
            fail(f"{n['id']}: names the query entity and property")
        par = [by[d] for d in n["depends_on"] if d in by]
        if len(par) != len(n["depends_on"]) or not all(x["truth"] for x in par):
            fail(f"{n['id']}: parent missing or not Yes")
        stated_par = [] if n.get("local_lines") else par
        if view_leak(it, n, stated_par):
            a["view_leak"] += 1
        a["leaks_answer_generic"] += leaks_answer(n["question"], it)
        ctx = local_context(it, n, stated_par, rng=rng)
        v = solve_view(ctx, n["question"])
        a[f"local_views/{n['type']}"] += 1
        if v is None or v != n["truth"]:
            fail(f"{n['id']}: local view does not decide it ({v} vs {n['truth']})")
        else:
            a[f"local_views_sufficient/{n['type']}"] += 1
        if n.get("local_lines") and (n["local_lines"][0] not in body):
            fail(f"{n['id']}: leaf line not a problem sentence")
        a[f"node_depth/{n['node_depth']}"] += 1
    for x in full_tree_plan(it, cap=0, rng=rng, subq_target="cot"):
        if x["node"]["type"] == "derive":
            a["tflm_derive"] += 1
            a["tflm_matched"] += bool(x["parents"])


# ------------------------------------------------------------------------------------------------------ cues
def final_cues(th: dict) -> dict:
    """Surface features of the final question (each predicts True iff the polarity of the property rule it picks
    matches the query's; None = no pick): the query's own "not"; the query predicate's rule that comes first in the
    text (the last one is its complement); the one whose subject concept is mentioned more often; the one whose
    subject is the object of some rule (either polarity) from a concept the query entity is stated to be (one-hop
    lookahead blind to "not"; picks only when exactly one does)."""
    prop, qneg = th["property"], th["query_neg"]
    prules = [r for r in th["rules"] if r[1] == prop]
    assert len(prules) == 2 and prules[0][3] != prules[1][3]
    agree = lambda r: r[3] == qneg  # noqa: E731     # predicting True iff this rule's polarity matches the query's
    text = P.body_text(th)
    cnt = [len(re.findall(rf"\b{r[0]}(?:es)?\b", text, re.I)) for r in prules]
    more = None if cnt[0] == cnt[1] else agree(prules[0] if cnt[0] > cnt[1] else prules[1])
    efacts = {c for e, c in th["facts"] if e == th["entity"]}
    look = [r for r in prules if any(x[0] in efacts and x[1] == r[0] for x in th["rules"])]
    return {"query_not_false": not qneg, "first_property_rule": agree(prules[0]), "more_mentioned_subject": more,
            "lookahead1_polarity_blind": agree(look[0]) if len(look) == 1 else None}


def norm_tokens(th: dict) -> list[str]:
    """Prompt tokens with names normalised (query entity, other entities, query property, concepts)."""
    ents = {x for f in th["facts"] for x in f[:1]}
    out = []
    for w in re.findall(r"[A-Za-z]+|[.:]", P.body_text(th) + " Q " + P.query_text(th)):
        lw = w.lower()
        if w == th["entity"]:
            out.append("QE")
        elif w in ents:
            out.append("ENT")
        elif lw == th["property"]:
            out.append("QP")
        elif lw.endswith("pus") or lw.endswith("puses"):
            out.append("CON")
        else:
            out.append(lw)
    return out


def hashed(tokens: list[str], dim: int) -> dict[int, float]:
    v: dict[int, float] = collections.Counter()
    for g in tokens + [a + " " + b for a, b in zip(tokens, tokens[1:])] + \
            [a + " " + b + " " + c for a, b, c in zip(tokens, tokens[1:], tokens[2:])]:
        v[int(hashlib.md5(g.encode()).hexdigest()[:8], 16) % dim] += 1.0
    return v


def logreg(train_x, train_y, tests: dict, dim: int, iters: int = 300, l2: float = 1e-3) -> dict:
    import numpy as np
    X = np.zeros((len(train_x), dim), dtype=np.float32)
    for i, v in enumerate(train_x):
        for j, c in v.items():
            X[i, j] = c
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    y = np.array(train_y, dtype=np.float32)
    w, b = np.zeros(dim, dtype=np.float32), 0.0
    m, s, mb, sb = np.zeros_like(w), np.zeros_like(w), 0.0, 0.0
    for t in range(1, iters + 1):                     # full-batch Adam on the L2-regularised logistic loss
        p = 1 / (1 + np.exp(-np.clip(X @ w + b, -30, 30)))
        g, gb = X.T @ (p - y) / len(y) + l2 * w, float((p - y).mean())
        m, s = .9 * m + .1 * g, .999 * s + .001 * g * g
        mb, sb = .9 * mb + .1 * gb, .999 * sb + .001 * gb * gb
        w -= .05 * (m / (1 - .9 ** t)) / (np.sqrt(s / (1 - .999 ** t)) + 1e-8)
        b -= .05 * (mb / (1 - .9 ** t)) / (np.sqrt(sb / (1 - .999 ** t)) + 1e-8)
    out = {"train": round(float((((X @ w + b) > 0) == (y > .5)).mean()), 3)}
    del X
    for name, (xs, ys) in tests.items():
        Z = np.zeros((len(xs), dim), dtype=np.float32)
        for i, v in enumerate(xs):
            for j, c in v.items():
                Z[i, j] = c
        out[name] = round(float(((((Z - mu) / sd) @ w + b > 0) == (np.array(ys) > .5)).mean()), 3)
    return out


def cv_by_hop(xs, ys, hops, dim: int, folds: int = 5, seed: int = 1) -> dict:
    idx = list(range(len(xs)))
    random.Random(seed).shuffle(idx)
    hit, tot = collections.Counter(), collections.Counter()
    for f in range(folds):
        te = idx[f * len(idx) // folds:(f + 1) * len(idx) // folds]
        tes = set(te)
        tr = [i for i in idx if i not in tes]
        tests = {k: ([xs[i] for i in te if hops[i] == k], [ys[i] for i in te if hops[i] == k]) for k in set(hops)}
        r = logreg([xs[i] for i in tr], [ys[i] for i in tr], tests, dim)
        for k in tests:
            hit[k] += r[k] * len(tests[k][1])
            tot[k] += len(tests[k][1])
    return {f"hop{k}": {"n": tot[k], "acc": round(hit[k] / tot[k], 3)} for k in sorted(tot)}


def node_cues(it: dict, nodes: list[dict], rows: list) -> None:
    """GFPS (statements view) features of the membership nodes: is the asked concept the object of a negative rule /
    of a positive rule whose subject is a concept the query entity is stated to be; which template and text half the
    positive rule with that object has. Parse nodes: does the question contain "not"."""
    th = it["theory"]
    rules = th["rules"]
    efacts = {c for e, c in th["facts"] if e == th["entity"]}
    for n in nodes:
        if n["type"] == "derive":
            c = n["concept"]
            pos = [(j, r) for j, r in enumerate(rules) if r[1] == c and not r[3]]
            rows.append(("member", n["node_depth"], n["truth"], {
                "obj_of_negative_rule": any(r[1] == c and r[3] for r in rules),
                "obj_of_positive_rule_from_entity_fact": any(r[0] in efacts for _, r in pos),
                "positive_rule_template0": bool(pos) and pos[0][1][4] == 0,
                "positive_rule_in_first_half": bool(pos) and pos[0][0] < len(rules) / 2}))
        elif n["type"] == "parse":
            rows.append((n["kind"], 0, n["truth"], {"question_has_not": " not " in n["question"]}))
            if n["kind"] in ("concept_rule", "twin_rule"):        # the same question form "Is every X (not) a Y?"
                rows.append(("concept_or_twin_rule", 0, n["truth"], {"question_has_not": " not " in n["question"]}))


def stump(rows: list[tuple[bool, object]]) -> dict | None:
    rows = [(t, f) for t, f in rows if f is not None]
    if not rows:
        return None
    acc = sum(t == f for t, f in rows) / len(rows)
    return {"n": len(rows), "acc": round(max(acc, 1 - acc), 3)}


# ------------------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/student_pronto_tf")
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--max-hops", type=int, default=4)
    ap.add_argument("--eval-hops", type=int, default=8)
    ap.add_argument("--train-per-hop", type=int, default=2500)
    ap.add_argument("--val-per-hop", type=int, default=100)
    ap.add_argument("--eval-per-hop", type=int, default=400)
    ap.add_argument("--wide-per-hop", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-tokens", action="store_true", help="skip the token-length report")
    args = ap.parse_args()
    out, edir = ROOT / args.out, ROOT / args.eval_dir
    out.mkdir(parents=True, exist_ok=True)
    edir.mkdir(parents=True, exist_ok=True)
    tok = None
    if not args.no_tokens:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(ROOT / TOKENIZER))
    # (split name, path, hops, per hop, kind, width)
    plan = [(f"pronto_d{k}", edir / f"pronto_d{k}.jsonl", [k], args.eval_per_hop, "eval", 0)
            for k in range(1, args.eval_hops + 1)]
    plan += [("pronto_wide", edir / "pronto_wide.jsonl", list(range(1, args.max_hops + 1)), args.wide_per_hop, "eval", 2),
             ("val", out / "val.jsonl", list(range(1, args.max_hops + 1)), args.val_per_hop, "train", 0),
             ("train", out / "train.jsonl", list(range(1, args.max_hops + 1)), args.train_per_hop, "train", 0)]
    seen: dict = {}
    fails: list = []
    audit = collections.Counter()
    summary: dict = {"args": vars(args), "files": {}, "md5": {}}
    fin_rows: dict = collections.defaultdict(list)       # split -> [(answer, cues)]
    bow: dict = collections.defaultdict(lambda: ([], []))
    ncue: list = []
    fin_hops: list = []
    tokens: dict = {}
    deep_items: dict = {}
    for split, path, hops, per, kind, width in plan:
        arng = random.Random(f"{args.seed}-audit-{split}")
        st = collections.Counter()
        hop_c, nodes_c, kind_c, nd_c, sent_c = (collections.Counter() for _ in range(5))
        ans_c = collections.Counter()
        toks = []
        with open(path, "w") as f:
            for k in hops:
                rng = random.Random(f"{args.seed}-{split}-{k}")
                for j, (ans, qneg) in enumerate(schedule(per, rng)):
                    for tries in range(100):
                        th = P.make_problem(k, ans, qneg, rng, width=width)
                        key = P.canonical(th)
                        if key not in seen:
                            break
                        st["duplicates_regenerated"] += 1
                    else:
                        raise RuntimeError(f"{split} k={k}: no new problem after 100 draws")
                    iid = {"train": f"pronto-tr-d{k}-{j:05d}", "val": f"pronto-va-d{k}-{j:05d}",
                           "pronto_wide": f"pronto-wide-d{k}-{j:05d}"}.get(split, f"pronto-d{k}-{j:05d}")
                    seen[key] = iid
                    nodes = P.proof_tree(th)
                    if kind == "train":
                        it = P.train_item(th, iid)
                    else:
                        it = P.eval_item(th, iid, f"pronto/{split.replace('pronto_', '')}" if width else f"pronto/d{k}")
                        if split in {f"pronto_d{d}" for d in range(5, 9)} and j < LOCAL_DEEP_PER_DEPTH:
                            deep = P.train_item(th, iid)
                            deep.update({x: it[x] for x in ("group", "type", "label_order", "gold_probs")})
                            deep_items.setdefault(split, []).append(deep)
                    audit_item(dict(it, subqs=nodes, local_preamble=P.LOCAL_PREAMBLE), nodes, arng, audit, fails)
                    f.write(json.dumps(it) + "\n")
                    hop_c[k] += 1
                    ans_c["true" if ans else "false"] += 1
                    ans_c["query_negated"] += qneg
                    sent_c[it["n_sentences"]] += 1
                    nodes_c[len(nodes)] += 1
                    for n in nodes:
                        kind_c[f"{n['type']}/{'yes' if n['truth'] else 'no'}"] += 1
                        nd_c[n["node_depth"]] += 1
                    fin_rows[split].append((ans, final_cues(th)))
                    if split != "val":
                        bow[split][0].append(hashed(norm_tokens(th), 4096))
                        bow[split][1].append(ans)
                        if split == "train":
                            fin_hops.append(k)
                    if split == "train":
                        node_cues(it, nodes, ncue)
                    if tok is not None:
                        toks.append(len(tok.encode(it["prompt"] + "\n\nAnswer:").ids))
        tot = sum(a * b for a, b in nodes_c.items())
        n_out = sum(hop_c.values())
        sents = sorted(x for x, c in sent_c.items() for _ in range(c))
        summary["files"][split] = {
            "path": rel(path), "items": n_out, "hops": dict(sorted(hop_c.items())),
            "answers": dict(ans_c), "sentences": {"min": sents[0], "median": sents[len(sents) // 2], "max": sents[-1]},
            "nodes": tot, "nodes_per_item": round(tot / max(1, n_out), 2), "node_types": dict(sorted(kind_c.items())),
            "node_depth": dict(sorted(nd_c.items())), "build": dict(st)}
        if toks:
            toks.sort()
            tokens[split] = {"max": toks[-1], "p99": toks[int(.99 * (len(toks) - 1))], "median": toks[len(toks) // 2],
                             "share_le_1536": round(sum(t <= 1536 for t in toks) / len(toks), 4)}
        print(split, json.dumps(summary["files"][split]), f"rss {rss_mb()} MB", flush=True)
        summary["md5"][rel(path)] = md5(path)
    with open(out / "local_deep.jsonl", "w") as f:
        for split in sorted(deep_items):
            for it in deep_items[split]:
                f.write(json.dumps(it) + "\n")
    summary["files"]["local_deep"] = {"path": rel(out / "local_deep.jsonl"),
                                      "items": {s: len(v) for s, v in sorted(deep_items.items())},
                                      "nodes": sum(len(it["subqs"]) for v in deep_items.values() for it in v)}
    summary["md5"][rel(out / "local_deep.jsonl")] = md5(out / "local_deep.jsonl")
    # baselines and cue stumps
    base = {}
    for split, rows in fin_rows.items():
        n = len(rows)
        tr = sum(a for a, _ in rows) / n
        base[split] = {"n": n, "majority": round(max(tr, 1 - tr), 3), "always_true": round(tr, 3),
                       **{c: stump([(a, cu[c]) for a, cu in rows]) for c in rows[0][1]}}
    summary["baselines_final"] = base
    tests = {s: v for s, v in bow.items() if s != "train"}
    summary["bow_logreg_final"] = {"per_file": logreg(bow["train"][0], bow["train"][1], tests, 4096),
                                   "cv5_train_by_hop": cv_by_hop(bow["train"][0], bow["train"][1], fin_hops, 4096),
                                   "features": "hashed 1-3-grams (4096) of the body + query, names normalised; "
                                               "per_file: fit on train (its train accuracy = fit), accuracy per file; "
                                               "cv5: 5-fold held-out accuracy within train, by hop"}
    nc: dict = {}
    for kind_, depth, truth, feats in ncue:
        for f_, v in feats.items():
            nc.setdefault(f"{kind_}/{f_}", []).append((truth, v))
            if kind_ == "member":
                nc.setdefault(f"{kind_}/d{depth}/{f_}", []).append((truth, v))
    summary["cues_gfps_nodes_train"] = {k: stump(v) for k, v in sorted(nc.items())}
    summary["tokens_final_prompt"] = tokens
    summary["audit"] = dict(sorted(audit.items()))
    summary["audit"]["failures"] = len(fails)
    summary["failure_examples"] = fails[:20]
    summary["fields"] = {"item": ["theory", "hops", "width", "answer", "query_neg", "rule_neg", "n_sentences",
                                  "local_preamble (train)", "tree_fix (train)", "tree_source (train)"],
                         "subq": ["id", "type (read/parse/derive)", "kind", "level", "question", "opposite", "truth",
                                  "p_cot", "p_fresh", "depends_on", "node_depth", "initial", "status", "local_lines",
                                  "concept (derive)"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("audit", "baselines_final", "bow_logreg_final", "cues_gfps_nodes_train",
                                              "tokens_final_prompt")}, indent=1))
    print(f"rss {rss_mb()} MB")
    if fails:
        print("AUDIT FAILED", fails[:10])
        sys.exit(1)


if __name__ == "__main__":
    main()
