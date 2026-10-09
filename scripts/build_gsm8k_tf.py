"""Tree Forcing data for a real-world reasoning domain: GSM8K, with trees read off its existing annotations (the
`socratic` sub-questions and the <<a*b=c>> calculator notes), so no new labelling is needed
(notes/next_round/tree_forcing/gsm8k_tf.md).

Finals: the multiple-choice finals of data/student_gsm8k_soc (scripts/build_gsm8k_socratic.py: GSM8K train, 5 numeric
options, decontaminated against GSM8K test / GSM-Plus / GSM-Symbolic / SVAMP / AQuA / GSM1k / data/eval) of problems
whose annotations parse and reach the gold (node_status "ok": every note's expression executes to its stated value,
every operand is an earlier result or a question number / constant, and some step yields the gold). Every final field
is byte-identical except the tree fields (CHANGED): source "gsm8k_tf", group "gsm8k_tf/s<n>", n_steps (= depth = the
number of calculator notes, as data/eval/gsm8k_mc*.jsonl count them), subqs, has_tree, tree_source.

Nodes (each a yes/no question with an exact truth; p_cot = truth, node_depth = the step's dependency level):
  per solution step j that is not the gold step: "s<j>v" "<sub-question> Is it <value>?" (Yes) and "s<j>n" the same
  with a wrong value (No), so every item, depth and step is exactly 50% Yes. The sub-question is the Socratic
  annotation's own text (verbatim; "?" appended when it has no end mark); "opposite" holds the reversed wording ("Is it
  different from <x>?"). The wrong value is above or below the value 50/50 and comes from a planted slip of the step
  (an operator of it or of an ancestor swapped, or a stated operand moved by 1, propagated through the graph; .5), another
  step's value (.25) or a perturbation keeping parity and roundness (.25), each among those available on that side;
  it is positive, has the value's integer-ness, at most 2 decimals, is within 10x of the value and is never the gold.
  depends_on = the "v" nodes of the step's parents (earlier steps whose result is an operand of its note, the
  certain edges of build_gsm8k_socratic.parse_problem; an operand that is both an earlier result and a unit constant /
  number word / question number gets no edge and the flag parents_uncertain / ambiguous_q). Only Yes nodes are parents.
Not built (counted in summary.json):
  - the gold step (the final question in disguise) and any step whose value equals the gold, and every step downstream
    of such a step (its probe would pin the gold), and steps after the gold step;
  - steps whose sub-question text is shared with another step of a different value (the Socratic annotation reuses a
    sub-question for a partial result, e.g. "How many floors does the Chrysler Building have?" for 11 and then 23), or
    equals the problem's final question sentence (normalised) while its value is not the gold: such a Yes node would
    be false as worded;
  - steps whose value is not positive or not exact at 2 decimals (the option formatter's precision).
Every node is re-verified from its own fields (audit_item): its expression is executed again and equals its value,
every parent's value is an operand of the expression, Yes iff the probe is the value, no probe is the gold.

Splits (problem level; a problem is never in two files):
  data/student_gsm8k_tf/train.jsonl     GSM8K train problems, all step counts
  data/student_gsm8k_tf/val.jsonl       the 300 held-out problems of student_gsm8k_soc/val + --n-val-extra (100) drawn
                                        from the train pool (seeded): about 400
  data/student_gsm8k_tf_le4/{train,val}.jsonl   the same items with n_steps <= 4 (extrapolation view)
  data/eval/gsm8k_s{2..8}.jsonl         data/eval/gsm8k_mc_v2.jsonl (QO-v2 options, GSM8K test) cut by step count,
                                        recomputed from the socratic test notes (checked against the items' depth);
  data/eval/gsm8k_s5p.jsonl             the same, >= 5 steps pooled (s6-s8 hold 57 / 21 / 9 problems)
Dedup: every train / val question vs every GSM8K-test-derived and GSM1k evaluation question (GSM8K test, gsm8k_mc,
gsm8k_mc_v2, gsm1k_mc, gsm_plus_mc, gsm_symbolic_mc): word 8-grams, an 8-gram shared by >= 5 test problems is
boilerplate; any other shared 8-gram drops the problem. Train problems sharing such an 8-gram with a val problem are
dropped from train.

Reports (summary.json): parse / drop counts, node stats, leak checks, single-feature cue stumps for the node targets in
the GFPQ view, the clean-cut share (treeplan.statement_block), dedup. Token lengths need the `tokenizers` package
(not in the system python): a second pass with --tokens writes data/student_gsm8k_tf/tokens.json.

  python3 scripts/build_gsm8k_tf.py                                    # pandas + numpy (system python3)
  /home/soray/.venvs/sev/bin/python scripts/build_gsm8k_tf.py --tokens  # tokenizers (stdlib otherwise)
"""
from __future__ import annotations

import argparse
import ast
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

from cotdistill.sources import _fmt  # noqa: E402
from cotdistill.treeplan import gsm8k_question_cut, problem_no_options, statement_block  # noqa: E402

SOURCE = "gsm8k_tf"
TREE_SOURCE = "gsm8k_socratic_tf"
CHANGED = {"source", "group", "subqs", "has_tree", "tree_source", "n_steps"}   # every other final field is kept
TOKENIZER = "results/weights/T1-Qwen3.5-2B-Base-student_v3-s2/model/tokenizer.json"
RAW = ROOT / "data/raw/bench_math/gsm8k/socratic"
ANN = re.compile(r"<<([^=<>]*)=([^<>]+)>>")
WTOK = re.compile(r"[a-z0-9]+")
NUMTOK = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?")
STEP_FILES = tuple(range(2, 9))
WRONG_W = {"slip": 0.5, "other": 0.25, "perturb": 0.25}
EVAL_TARGETS = ("gsm8k_mc", "gsm8k_mc_v2", "gsm1k_mc", "gsm_plus_mc", "gsm_symbolic_mc")


def soc():
    """scripts/build_gsm8k_socratic.py (parse_problem, run, SWAP, ...); imported lazily, it needs numpy."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_gsm8k_socratic as m
    return m


# ------------------------------------------------------------------------------------------------------ numbers
def is_int(v: float) -> bool:
    return abs(v - round(v)) < 1e-9


def exact2(v: float) -> bool:
    """Positive and printed exactly by the option formatter (_fmt: at most 2 decimals)."""
    return v > 0 and abs(float(_fmt(v)) - v) < 1e-9


def ok_wrong(w: float, v: float, gold: float) -> bool:
    return (exact2(w) and _fmt(w) != _fmt(v) and _fmt(w) != _fmt(gold) and (is_int(w) or not is_int(v))
            and max(w / v, v / w) <= 10)


def perturbs(v: float, up: bool) -> list[float]:
    """v +- d keeping parity and roundness (an integer with no trailing zero moves by 2, 4 or 10, a round one by
    {1, 2, 5, 10} x 10^k, k its trailing zeros up to 3); a non-integer moves by 1, 2, 5 or 10."""
    if is_int(v):
        n, k = int(round(v)), 0
        while n % 10 ** (k + 1) == 0 and k < 3:
            k += 1
        ds = [2, 4, 10] if k == 0 else [x * 10 ** k for x in (1, 2, 5, 10)]
    else:
        ds = [1, 2, 5, 10]
    return [v + d if up else v - d for d in ds]


def ask_text(sq: str) -> str:
    sq = " ".join(sq.split())
    return sq if sq[-1:] in (".", "?", "!") else sq + "?"


def node_question(sq: str, x: float) -> str:
    return f"{ask_text(sq)} Is it {_fmt(x)}?"


def node_opposite(sq: str, x: float) -> str:
    return f"{ask_text(sq)} Is it different from {_fmt(x)}?"


def norm(s: str) -> str:
    return " ".join(WTOK.findall(s.lower().replace("’", "'")))


def final_sentence(question: str) -> str:
    return re.split(r"(?<=[.!?])\s+", question.strip())[-1]


# ------------------------------------------------------------------------------------------------------ tree
def ancestors(steps: list[dict], j: int) -> set:
    seen, todo = set(), list(steps[j]["parents"])
    while todo:
        k = todo.pop()
        if k not in seen:
            seen.add(k)
            todo += steps[k]["parents"]
    return seen


def slip_values(p: dict, j: int) -> list[float]:
    """Planted slips of step j: swap one operator, or move one stated operand by 1, in j or one of its ancestors and
    re-run the graph (parents' results flow through the certain edges); [] when any step up to j has an uncertain
    operand (the propagation would be unsound)."""
    S = soc()
    steps = p["steps"]
    if any("parents_uncertain" in s["flags"] or "ambiguous_q" in s["flags"] for s in steps[:j + 1]):
        return []
    order = sorted(ancestors(steps, j) | {j})
    out = set()
    for i in order:
        tree = steps[i]["tree"]
        edits = [("swap", id(n)) for n in ast.walk(tree) if isinstance(n, ast.BinOp) and type(n.op) in S.SWAP]
        edits += [("bump", (id(c), d)) for c in S.constants(tree) if id(c) not in steps[i]["refs"] for d in (1, -1)]
        for kind, e in edits:
            vals = {}
            try:
                for k in order:
                    env = {cid: vals[src] for cid, src in steps[k]["refs"].items()}
                    vals[k] = S.run(steps[k]["tree"], env, swap=e if (k == i and kind == "swap") else None,
                                    bump=e if (k == i and kind == "bump") else None)
            except (ZeroDivisionError, OverflowError, ValueError, KeyError):
                continue
            out.add(round(vals[j], 6))
    return sorted(out)


def wrong_value(v: float, gold: float, slips: list, others: list, rng: random.Random) -> tuple[float, str] | None:
    up = rng.random() < 0.5
    for side in (up, not up):
        pools = {"slip": slips, "other": others, "perturb": perturbs(v, side)}
        pools = {k: sorted({w for w in xs if ok_wrong(w, v, gold) and (w > v) == side}) for k, xs in pools.items()}
        avail = [k for k in ("slip", "other", "perturb") if pools[k]]
        if avail:
            k = rng.choices(avail, weights=[WRONG_W[a] for a in avail])[0]
            return rng.choice(pools[k]), k
    return None


def tf_nodes(p: dict, rng: random.Random, st: collections.Counter) -> list[dict]:
    """The yes/no nodes of one parsed problem (build_gsm8k_socratic.parse_problem, status ok)."""
    steps, gs, gold = p["steps"], p["gold_step"], p["gold"]
    close = soc().close
    fin = norm(final_sentence(p["question"]))
    blocked = {gs} | {j for j, s in enumerate(steps) if close(s["value"], gold)}
    st["blocked_gold_valued"] += len(blocked) - 1
    for j in range(len(steps)):                                   # downstream of a blocked step: blocked too
        if j not in blocked and ancestors(steps, j) & blocked and j < gs:
            blocked.add(j)
            st["blocked_downstream_of_gold_valued"] += 1
    by_text = collections.defaultdict(set)
    for s in steps:
        by_text[norm(s["subq"])].add(round(s["value"], 6))
    gold_anc = ancestors(steps, gs)
    nodes, ids = [], {}
    for j, s in enumerate(steps):
        v = s["value"]
        if j in blocked:
            continue
        if j > gs:
            st["dropped_after_gold_step"] += 1
            continue
        if len(by_text[norm(s["subq"])]) > 1:
            st["dropped_subq_shared_other_value"] += 1
            continue
        if norm(s["subq"]) == fin:
            st["dropped_subq_is_final_question"] += 1
            continue
        if not exact2(v):
            st["dropped_value_not_exact2" if v > 0 else "dropped_value_not_positive"] += 1
            continue
        others = [x["value"] for k, x in enumerate(steps) if k != j]
        w = wrong_value(v, gold, slip_values(p, j), others, rng)
        if w is None:
            st["dropped_no_wrong_value"] += 1
            continue
        par = [f"s{k + 1}v" for k in s["parents"] if f"s{k + 1}v" in ids]
        st["parent_edges"] += len(s["parents"])
        st["parent_edges_unbuilt"] += len(s["parents"]) - len(par)
        meta = {"step": j + 1, "expr": s["expr"], "value": v, "flags": s["flags"], "gold_ancestor": j in gold_anc,
                "node_depth": s["level"]}
        for suffix, x, truth, kind in (("v", v, True, None), ("n", w[0], False, w[1])):
            n = {"id": f"s{j + 1}{suffix}", "question": node_question(s["subq"], x),
                 "opposite": node_opposite(s["subq"], x), "truth": truth, "p_cot": 1.0 if truth else 0.0,
                 "p_fresh": None, "type": "step", "depends_on": par, "initial": None, "status": "direct",
                 "probe": x, **meta}
            if kind:
                n["wrong"] = kind
                st[f"wrong_{kind}"] += 1
            nodes.append(n)
        ids[f"s{j + 1}v"] = j
    return nodes


def convert(final: dict, p: dict, rng: random.Random, st: collections.Counter) -> dict:
    out = dict(final)
    out["source"] = SOURCE
    out["n_steps"] = len(p["steps"])
    out["group"] = f"{SOURCE}/s{out['n_steps']}"
    out["subqs"] = tf_nodes(p, rng, st)
    out["has_tree"] = bool(out["subqs"])
    out["tree_source"] = TREE_SOURCE
    return out


# ------------------------------------------------------------------------------------------------------ audit
def gold_text(item: dict) -> str:
    return str(item["options"][item["labels"].index(item["gold_label"])])


def audit_item(item: dict) -> list[str]:
    """Problems of one built item, from its own fields only (empty: clean): the final, the GFPQ context, every node's
    arithmetic re-executed, parents' values among its operands, truth = (probe == value), Yes/No pairing, no probe or
    value equal to the gold, node_depth = 1 + the deepest parent."""
    S = soc()
    bad, iid = [], item.get("item_id")
    gold = gold_text(item)
    if item.get("source") != SOURCE or item.get("n_steps") != item.get("depth") or item.get("tree_fix") is not True:
        bad.append("item fields")
    try:
        ctx = problem_no_options(item)
        if ctx + "\nOptions:\n" != item["prompt"][:len(ctx) + 10] or "\n" in ctx:
            bad.append("GFPQ context is not the prompt before its options")
    except ValueError as e:
        bad.append(f"problem_no_options: {e}")
    by = {n["id"]: n for n in item["subqs"]}
    if len(by) != len(item["subqs"]):
        bad.append("duplicate node ids")
    for n in item["subqs"]:
        tree = S.parse_expr(n["expr"])
        try:
            got = S.run(tree) if tree is not None else None
        except (ValueError, ZeroDivisionError, OverflowError):
            got = None
        if got is None or not S.close(got, n["value"]) or not exact2(n["value"]):
            bad.append(f"{n['id']}: expression {n['expr']!r} does not give {n['value']}")
        if (n["truth"] is True) != (_fmt(n["probe"]) == _fmt(n["value"])) or n["p_cot"] != float(n["truth"]):
            bad.append(f"{n['id']}: truth is not (probe == value)")
        if n["id"].endswith("v") != n["truth"] or not n["question"].endswith(f" Is it {_fmt(n['probe'])}?"):
            bad.append(f"{n['id']}: id / wording")
        if _fmt(n["probe"]) == gold or _fmt(n["value"]) == gold:
            bad.append(f"{n['id']}: probe or value equals the gold {gold}")
        twin = by.get(n["id"][:-1] + ("n" if n["truth"] else "v"))
        if not twin or twin["value"] != n["value"] or twin["truth"] == n["truth"]:
            bad.append(f"{n['id']}: no Yes/No twin")
        consts = [abs(float(c.value)) for c in S.constants(tree)] if tree is not None else []
        depth = 0
        for d in n["depends_on"]:
            par = by.get(d)
            if not par or not par["truth"]:
                bad.append(f"{n['id']}: parent {d} missing or not Yes")
                continue
            if not any(S.matches(c, par["value"]) for c in consts):
                bad.append(f"{n['id']}: parent {d} value {par['value']} is not an operand of {n['expr']!r}")
            depth = max(depth, par["node_depth"] + 1)
        if n["node_depth"] < depth:
            bad.append(f"{n['id']}: node_depth {n['node_depth']} below its parents' + 1")
    return [f"{iid}: {b}" for b in bad]


def cue_features(item: dict) -> list[tuple]:
    """(truth, probe in the problem text, probe a multiple of 10, probe even, probe above every number of the
    problem, probe = a op b of two problem numbers, No probe above the value) per node, as seen in the GFPQ view."""
    q = item["prompt"].partition("\nOptions:")[0]
    nums = [float(x.replace(",", "")) for x in NUMTOK.findall(q)]
    ops = set()
    for i, a in enumerate(nums):
        for b in nums[i + 1:]:
            ops |= {round(x, 6) for x in (a + b, abs(a - b), a * b) + ((a / b,) if b else ()) + ((b / a,) if a else ())}
    rows = []
    for n in item["subqs"]:
        x = n["probe"]
        rows.append((n["truth"], any(abs(x - y) < 1e-9 for y in nums), is_int(x) and round(x) % 10 == 0,
                     is_int(x) and round(x) % 2 == 0, bool(nums) and x > max(nums), round(x, 6) in ops,
                     None if n["truth"] else x > n["value"]))
    return rows


def stump(rows: list[tuple], k: int) -> float | None:
    xs = [(r[0], r[k]) for r in rows if r[k] is not None]
    if not xs:
        return None
    acc = sum(t == f for t, f in xs) / len(xs)
    return round(max(acc, 1 - acc), 3)


# ------------------------------------------------------------------------------------------------------ dedup
def grams8(text: str) -> set:
    w = WTOK.findall(text.lower())
    return {" ".join(w[i:i + 8]) for i in range(len(w) - 7)}


def stem(prompt: str) -> str:
    s = prompt.partition("\nOptions:")[0]
    return s[len("Question: "):] if s.startswith("Question: ") else s


def eval_targets(test_questions: list[str], ours: set) -> dict[str, set]:
    """8-gram -> the set of test problems it occurs in (GSM8K test rows and every file derived from them by orig_id,
    GSM1k items by item_id), for the 8-grams in `ours` only (memory)."""
    owners = collections.defaultdict(set)
    for i, q in enumerate(test_questions):
        for g in grams8(q) & ours:
            owners[g].add(f"gsm8k-test-{i}")
    for name in EVAL_TARGETS:
        with open(ROOT / "data/eval" / f"{name}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                key = r.get("orig_id") or r["item_id"]
                if key.startswith("gsm8k-test-"):
                    key = "gsm8k-test-" + key.rsplit("-", 1)[1]
                for g in grams8(stem(r["prompt"])) & ours:
                    owners[g].add(key)
    return owners


def shared(text: str, owners: dict, boiler: int = 5) -> list[str]:
    return sorted(g for g in grams8(text) if g in owners and len(owners[g]) < boiler)


# ------------------------------------------------------------------------------------------------------ io
def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def write(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return md5(path)


def rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def stats(items: list[dict]) -> dict:
    nodes = [n for it in items for n in it["subqs"]]
    return {"items": len(items), "with_tree": sum(it["has_tree"] for it in items),
            "n_steps": dict(sorted(collections.Counter(it["n_steps"] for it in items).items())),
            "nodes": len(nodes), "nodes_per_item": round(len(nodes) / max(1, len(items)), 2),
            "yes_share": round(sum(n["truth"] for n in nodes) / max(1, len(nodes)), 4),
            "node_depth": dict(sorted(collections.Counter(n["node_depth"] for n in nodes).items())),
            "nodes_by_item_steps": dict(sorted(collections.Counter(it["n_steps"] for it in items
                                                                   for _ in it["subqs"]).items())),
            "gold_ancestor_share": round(sum(n["gold_ancestor"] for n in nodes) / max(1, len(nodes)), 3),
            "parents_uncertain_share": round(sum("parents_uncertain" in n["flags"] for n in nodes) / max(1, len(nodes)), 3),
            "clean_cut_share": round(sum(gsm8k_question_cut(stem(it["prompt"])) is not None for it in items)
                                     / max(1, len(items)), 3)}


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------------------------------------------ build
def build(args) -> None:
    import pandas as pd
    S = soc()
    out, out4, edir = ROOT / args.out, ROOT / args.out_le4, ROOT / args.eval_dir
    df = pd.read_parquet(RAW / "train-00000-of-00001.parquet")
    tq = list(pd.read_parquet(RAW / "test-00000-of-00001.parquet").question)
    ta = list(pd.read_parquet(RAW / "test-00000-of-00001.parquet").answer)
    st = collections.Counter()
    finals = {"train": [], "val": []}
    for split in ("train", "val"):
        for r in read_jsonl(ROOT / args.base / f"{split}.jsonl"):
            if r.get("row_kind") != "final":
                continue
            st[f"soc_{split}_finals"] += 1
            if r.get("node_status") != "ok":
                st[f"dropped_status_{r.get('node_status')}"] += 1
                continue
            finals[split].append(r)
    print("finals", {k: len(v) for k, v in finals.items()}, dict(st), f"rss {rss_mb()} MB", flush=True)
    items = {"train": [], "val": []}
    fails = []
    for split, rows in finals.items():
        for r in rows:
            i = int(r["problem_id"].rsplit("-", 1)[1])
            p = S.parse_problem(i, df.question[i], df.answer[i])
            if p["status"] != "ok" or f"Question: {p['question']}\nOptions:\n" != r["prompt"][:len(p["question"]) + 20]:
                fails.append(f"{r['item_id']}: raw row {i} does not reproduce the gsm8k_soc final")
                continue
            if _fmt(p["gold"]) != gold_text(r) or len(p["steps"]) != r["depth"] or len(ANN.findall(df.answer[i])) != r["depth"]:
                fails.append(f"{r['item_id']}: gold / step count differs from the raw solution")
                continue
            t = convert(r, p, random.Random(f"{args.seed}/{r['problem_id']}"), st)
            diff = {k for k in set(t) | set(r) if t.get(k) != r.get(k)}
            if not diff <= CHANGED:
                fails.append(f"{r['item_id']}: fields changed beyond the tree fields: {sorted(diff - CHANGED)}")
            items[split].append(t)
    print("converted", {k: len(v) for k, v in items.items()}, f"rss {rss_mb()} MB", flush=True)
    # ---- 8-gram dedup against every GSM8K-test-derived and GSM1k evaluation question
    ours = set().union(*(grams8(stem(it["prompt"])) for split in items for it in items[split]))
    owners = eval_targets(tq, ours)
    del ours
    dedup = {"targets": EVAL_TARGETS + ("gsm8k_test_raw",), "boilerplate_min_problems": 5, "dropped": {},
             "examples": [], "any_8gram_incl_boilerplate": {}}
    for split in ("train", "val"):
        keep = []
        for it in items[split]:
            g = shared(stem(it["prompt"]), owners)
            dedup["any_8gram_incl_boilerplate"][split] = dedup["any_8gram_incl_boilerplate"].get(split, 0) + bool(
                grams8(stem(it["prompt"])) & owners.keys())
            if g:
                dedup["dropped"][split] = dedup["dropped"].get(split, 0) + 1
                if len(dedup["examples"]) < 6:
                    dedup["examples"].append({"item_id": it["item_id"], "grams": g[:3],
                                              "test": sorted(set().union(*(owners[x] for x in g)))[:3]})
                continue
            keep.append(it)
        items[split] = keep
    del owners
    # ---- val: student_gsm8k_soc's held-out 300 + n_val_extra seeded from the train pool
    pool = sorted(items["train"], key=lambda r: int(r["problem_id"].rsplit("-", 1)[1]))
    extra = set(x["item_id"] for x in random.Random(f"{args.seed}-val").sample(pool, args.n_val_extra))
    val = items["val"] + [it for it in pool if it["item_id"] in extra]
    train = [it for it in items["train"] if it["item_id"] not in extra]
    vown = collections.defaultdict(set)
    for it in val:
        for g in grams8(stem(it["prompt"])):
            vown[g].add(it["item_id"])
    tr_owners = collections.defaultdict(set)
    for it in train:
        for g in grams8(stem(it["prompt"])):
            if g in vown:
                tr_owners[g].add(it["item_id"])
    boiler = {g for g in vown if len(vown[g] | tr_owners.get(g, set())) >= 5}
    near = {it["item_id"] for it in train if any(g in vown and g not in boiler for g in grams8(stem(it["prompt"])))}
    dedup["train_dropped_vs_val"] = len(near)
    train = [it for it in train if it["item_id"] not in near]
    assert not {it["problem_id"] for it in train} & {it["problem_id"] for it in val}
    # ---- audit
    for it in train + val:
        fails += audit_item(it)
    leak = collections.Counter()
    from cotdistill.teacher import leaks_answer
    for it in train + val:
        gold = gold_text(it)
        for n in it["subqs"]:
            leak["nodes"] += 1
            leak["yes_node_of_gold_step_or_gold_value"] += n["truth"] and _fmt(n["value"]) == gold
            leak["probe_equals_gold"] += _fmt(n["probe"]) == gold
            leak["gold_number_in_node_text"] += gold in NUMTOK.findall(n["question"].rpartition(" Is it ")[0])
            leak["teacher_leaks_answer"] += leaks_answer(n["question"], it)
    # ---- cues in the GFPQ view
    rows = [r for it in train for r in cue_features(it)]
    cues = {"n": len(rows), "probe_in_problem": stump(rows, 1), "probe_round10": stump(rows, 2),
            "probe_even": stump(rows, 3), "probe_above_all_problem_numbers": stump(rows, 4),
            "probe_one_op_of_problem_numbers": stump(rows, 5),
            "no_probes_above_value": round(sum(r[6] for r in rows if r[6] is not None)
                                           / max(1, sum(r[6] is not None for r in rows)), 3)}
    # ---- write
    summary = {"script": "scripts/build_gsm8k_tf.py", "args": vars(args), "md5": {}, "files": {}}
    le4 = {"train": [it for it in train if it["n_steps"] <= 4], "val": [it for it in val if it["n_steps"] <= 4]}
    for d, name, rows_ in ((out, "train", train), (out, "val", val), (out4, "train", le4["train"]),
                           (out4, "val", le4["val"])):
        key = f"{d.name}/{name}.jsonl"
        summary["md5"][key] = write(d / f"{name}.jsonl", rows_)
        summary["files"][key] = stats(rows_)
    summary["val_extra_from_train_pool"] = sorted(extra)
    # ---- per-step-count evaluation files from gsm8k_mc_v2 (step count recomputed from the socratic test notes)
    ev = read_jsonl(edir / "gsm8k_mc_v2.jsonl")
    steps_of = {f"gsm8k-test-{i}": max(1, len(ANN.findall(a))) for i, a in enumerate(ta)}
    by_s = collections.defaultdict(list)
    for r in ev:
        s = steps_of[r["orig_id"]]
        if s != r["depth"]:
            fails.append(f"{r['item_id']}: step count {s} != depth {r['depth']}")
        by_s[min(s, max(STEP_FILES))].append(r)
    summary["eval_files"] = {}
    for s in STEP_FILES:
        name = f"gsm8k_s{s}.jsonl"
        summary["md5"][f"eval/{name}"] = write(edir / name, by_s[s])
        summary["eval_files"][name] = len(by_s[s])
    s5p = [r for s in STEP_FILES if s >= 5 for r in by_s[s]]
    summary["md5"]["eval/gsm8k_s5p.jsonl"] = write(edir / "gsm8k_s5p.jsonl", s5p)
    summary["eval_files"]["gsm8k_s5p.jsonl"] = len(s5p)
    summary["eval_files"]["source"] = "data/eval/gsm8k_mc_v2.jsonl (md5 %s), items unchanged" % md5(edir / "gsm8k_mc_v2.jsonl")
    summary["build"] = dict(sorted(st.items()))
    summary["dedup"] = dedup
    summary["leaks"] = dict(leak)
    summary["cues_gfpq_view"] = cues
    summary["audit"] = {"failures": len(fails), "examples": fails[:20]}
    summary["fields"] = {"item": ["source gsm8k_tf", "group gsm8k_tf/s<n>", "n_steps", "subqs", "has_tree",
                                  "tree_source"],
                         "subq": ["id s<j>v / s<j>n", "question", "opposite", "truth", "p_cot", "p_fresh", "type step",
                                  "depends_on", "initial", "status", "probe", "wrong (No: slip / other / perturb)",
                                  "step", "expr", "value", "flags", "gold_ancestor", "node_depth"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("files", "eval_files", "build", "dedup", "leaks", "cues_gfpq_view",
                                              "audit")}, indent=1))
    print(f"rss {rss_mb()} MB")
    if fails:
        print("AUDIT FAILED", fails[:10])
        sys.exit(1)


# ------------------------------------------------------------------------------------------------------ tokens
def templates() -> dict:
    """SUBQ_TEMPLATE / FINAL_TEMPLATE of cotdistill.student, read from its source (importing it needs torch)."""
    tree = ast.parse((ROOT / "src/cotdistill/student.py").read_text())
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and getattr(n.targets[0], "id", "") in (
                "SUBQ_TEMPLATE", "FINAL_TEMPLATE"):
            out[n.targets[0].id] = ast.literal_eval(n.value)
    return out


def tokens(args) -> None:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / TOKENIZER))
    T = templates()
    n_tok = lambda s: len(tok.encode(s, add_special_tokens=False).ids)   # noqa: E731
    rep = {"tokenizer": TOKENIZER, "max_len": args.max_len, "files": {}}

    def summ(xs: list[int]) -> dict:
        xs = sorted(xs)
        if not xs:
            return {}
        return {"n": len(xs), "mean": round(sum(xs) / len(xs), 1), "p50": xs[len(xs) // 2],
                "p99": xs[min(len(xs) - 1, int(0.99 * len(xs)))], "max": xs[-1],
                "share_le_max_len": round(sum(x <= args.max_len for x in xs) / len(xs), 4)}
    for rel in (f"{args.out}/train.jsonl", f"{args.out}/val.jsonl", f"{args.out_le4}/train.jsonl"):
        fin, gfpq, gfp, per_item = [], [], [], []
        for it in read_jsonl(ROOT / rel):
            f = n_tok(T["FINAL_TEMPLATE"].format(problem=it["prompt"]))
            fin.append(f)
            ctx = problem_no_options(it)
            tot = f
            for n in it["subqs"]:
                a = n_tok(T["SUBQ_TEMPLATE"].format(problem=ctx, question=n["question"], a="Yes", b="No"))
                gfpq.append(a)
                gfp.append(n_tok(T["SUBQ_TEMPLATE"].format(problem=it["prompt"], question=n["question"], a="Yes",
                                                           b="No")))
                tot += a
            per_item.append(tot)
        rep["files"][rel] = {"final_view": summ(fin), "gfpq_node_view": summ(gfpq), "gfp_node_view": summ(gfp),
                             "epoch_tokens_A2": sum(fin), "epoch_tokens_GFPQ": sum(per_item),
                             "GFPQ_over_A2": round(sum(per_item) / max(1, sum(fin)), 2)}
    for s in list(STEP_FILES) + ["5p"]:
        rel = f"{args.eval_dir}/gsm8k_s{s}.jsonl"
        rep["files"][rel] = {"final_view": summ([n_tok(T["FINAL_TEMPLATE"].format(problem=r["prompt"]))
                                                 for r in read_jsonl(ROOT / rel)])}
    (ROOT / args.out / "tokens.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/student_gsm8k_soc")
    ap.add_argument("--out", default="data/student_gsm8k_tf")
    ap.add_argument("--out-le4", default="data/student_gsm8k_tf_le4")
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--n-val-extra", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens", action="store_true", help="only the token-length report (needs tokenizers)")
    ap.add_argument("--max-len", type=int, default=1536)
    args = ap.parse_args()
    if args.tokens:
        tokens(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
