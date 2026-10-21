"""GSM8K verification nodes for Tree Forcing (verification-tree pilot; arms GFPQVT / GFPQVTR in hpc/train_arm.sbatch,
student.build_full_tree --node-format value --node-render vt / vtr --tree-view problem).

Why: one-pass deciders estimate the answer and then match the options; when wrong they mostly pick the nearest
distractor or an intermediate result. A verification node supervises that matching step: it shows the problem WITH
its options and asks what one option is ("In this problem, what does option C (8) stand for?"), four choices:
  "The answer to the problem's final question"      the gold option
  "<Socratic sub-question>"                         an option equal to that solution step's value ("How many ants
                                                    does Beth see?")
  "A number written in the problem text"            an option equal to a number written in the problem
  "None of the quantities in the solution"          anything else that is no quantity of the solution
The fixed texts are sized to sit inside the sub-questions' length range (sub-questions: 26 / 33 / 40 / 48 / 57
characters at the 5th / 25th / 50th / 75th / 90th percentile; the fixed texts 36 / 38 / 42), so that the length rank
of the right choice can be balanced (with "The answer to the question", 26 characters, it was the shortest choice of
every gold node).
Automatic, no teacher: the Socratic sub-questions and calculator notes of GSM8K's own solutions
(build_gsm8k_socratic.parse_problem, as scripts/build_gsm8k_tf.py / build_gsm8k_ops.py parse them).

Rows: every row of data/student_gsm8k_tf_si/{train,val}.jsonl (yes/no nodes in subqs, value_nodes, worked_steps), byte
for byte, plus ONE new field `verify_nodes` (last). The yes/no, value-node, SI and slot arms read other fields, so they
train on these rows exactly as on their own data.

Descriptions of one problem (its choice pool):
  - A, N and G (the fixed texts above);
  - S_j for every solution step j that is not the gold step, whose value is not the gold, whose sub-question text is
    not shared with a step of another value and is not the problem's final question sentence (the rules under which
    build_gsm8k_tf keeps a yes/no node true as worded), whose value is positive and printable at 2 decimals, and whose
    sub-question shows no gold number the problem does not show; identical texts are one description.
What an option value v is (exactly one canonical description, else no node):
  gold                      v = the gold                                        -> A
  step                      v = the value of exactly one pool step's description -> S_j (two different: skipped)
  (skipped)                 v = the value of a step outside the pool              (no true description to offer)
  given                     v is written in the problem (digits, or a cardinal word: one .. twenty, thirty .. ninety,
                            hundred, thousand, million, dozen) and is a quantity of the solution (an operand of a
                            note)                                                -> G
  (skipped)                 v is written in the problem but the solution never uses it (G and N both true)
  (skipped)                 v is another quantity of the solution (a partial result inside a multi-operation note, a
                            constant operand such as 60 or 12)
  none                      anything else                                        -> N
A description other than the canonical one that is ALSO true of v (the gold or a step value written in the problem:
G) is never offered in that node, so every node has exactly one true choice.
Nodes per problem: the gold option, every step and given option, and --none-nodes (default 1) of the none options
(seeded draw; "all" keeps every option). Why not all five by default: in the training finals 65% of the options are
"none" (QO-v2 perturbations and one-step combinations of the stated numbers that the solution never computes), so
with every option asked "None of ..." would be the right choice of 2 nodes in 3 and always-"none" would score 65%;
with one, the four kinds are right in 37 / 37 / 19 / 8% of the nodes (answer / none / step / given; summary
type_prior), and the shuffled control can move every problem's descriptions among different options.
Choices (4 per node; a node with fewer than 3 false descriptions in its pool is not built): the canonical description,
A and N whenever they are false (so every node offers both, and the "answer" / "none" decision is never given away by
the choice set), the control target (below) when it is another description, and the rest from the false S_j and G,
picked so that the canonical description's character-length rank among the 4 is drawn uniformly (the rank tried first
from a seeded order, the next one when no pick realises it; as the value nodes' rank rule). The 4 are shuffled
(random.Random("<seed>/<problem>/<node>/vt/order")): answer position about uniform.
Control (rendering vtr, arm GFPQVTR: wrong content, the same stems and choices): the canonical descriptions of a
problem's built nodes permuted uniformly among those nodes (random.Random("<seed>/<problem>/vt/control"); a node keeps
its own by chance, about 1 in 2.7, as GFPQR's targets); vtr's answer is the index of the permuted description in the
SAME options as vt (the permuted description is always among them). When the permuted description is true of the
option without being its canonical one (G of a step value written in the problem), the node keeps its own (counted).
No gold leak beyond the options: the stem is the final question's prompt (problem + options) and the node question
names only the option's letter and value; the choices name sub-questions of the solution, never a computed value.

A node (verify_nodes entry): id v<letter>, type verify, option, value (the option text), vt_kind gold / step / given /
none, step (1-based, step nodes), option_kind (the final's option_kinds entry: qo2 partial_ab, perturb_gold, ...),
node_depth None, depends_on [], control_agree, renderings {vt, vtr: {question, options (4), answer, kinds (the kind of
each choice: answer / none / given / step:<j>)}} (vt and vtr share question and options).
Audit (audit_item, from the stored fields and the parsed solution): the option value is the prompt's option, the
canonical choice is true of it, every other choice false, 4 distinct one-line choices, A and N offered, the stem text,
vtr = vt but the answer, the control answers a permutation of the canonical descriptions (but the counted keeps).
Reports (summary.json): node and skip counts, kinds, answer positions, length ranks, the type prior (how often each
kind is right when offered), shortest / longest picks, a description-only adversary (conditional logit on each
choice's kind, length, length rank, position and step position, fitted on train nodes, scored on val), control
agreement; node_audit.txt shows 5 problems (gsm8k-train-660 first) with all their nodes.

  python3 scripts/build_gsm8k_vt.py                                   # pandas + numpy (system python3)
  /home/soray/.venvs/sev/bin/python scripts/build_gsm8k_vt.py --tokens  # tokens / epoch (tokenizers)
"""
from __future__ import annotations

import argparse
import ast
import collections
import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_gsm8k_tf as T  # noqa: E402

FIELD = "verify_nodes"
RENDERINGS = ("vt", "vtr")
DESC_ANSWER = "The answer to the problem's final question"
DESC_NONE = "None of the quantities in the solution"
DESC_GIVEN = "A number written in the problem text"
K = 4
CARDINALS = ("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
             "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand "
             "million dozen").split()
KIND_OF = {DESC_ANSWER: "answer", DESC_NONE: "none", DESC_GIVEN: "given"}


def step_desc(subq: str) -> str:
    return T.ask_text(subq)


def node_question(letter: str, value: str) -> str:
    return f"In this problem, what does option {letter} ({value}) stand for?"


# ------------------------------------------------------------------------------------------------------ problem
def given_numbers(q: str) -> set:
    """Numbers written in the problem: digit numbers (build_gsm8k_socratic.question_numbers' verbatim set) and
    cardinal number words."""
    S = T.soc()
    verb, _, _ = S.question_numbers(q)
    words = {float(S.WORDS[w]) for w in S.TOK.findall(q.lower()) if w in CARDINALS}
    return verb | words


def subtree_values(tree) -> list[float]:
    """Every value of a note: its constants (absolute) and the value of each binary operation."""
    S = T.soc()
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant):
            out.append(abs(float(n.value)))
        elif isinstance(n, ast.BinOp):
            try:
                out.append(float(S.run(n)))
            except (ZeroDivisionError, OverflowError, ValueError, KeyError):
                pass
    return out


def problem_info(p: dict) -> dict:
    """Pool descriptions and the value sets of one parsed problem (status ok)."""
    S = T.soc()
    steps, gs, gold = p["steps"], p["gold_step"], p["gold"]
    fin = T.norm(T.final_sentence(p["question"]))
    by_text = collections.defaultdict(set)
    for s in steps:
        by_text[T.norm(s["subq"])].add(round(s["value"], 6))
    qnums = set(T.NUMTOK.findall(p["question"]))
    gold_t = T._fmt(gold)
    pool, skipped = [], collections.Counter()     # pool: (desc, value, step index)
    seen = set()
    for j, s in enumerate(steps):
        v = s["value"]
        if j == gs:
            continue
        if S.close(v, gold):
            skipped["gold_valued"] += 1
        elif len(by_text[T.norm(s["subq"])]) > 1:
            skipped["subq_shared_other_value"] += 1
        elif T.norm(s["subq"]) == fin:
            skipped["subq_is_final_question"] += 1
        elif not T.exact2(v):
            skipped["value_not_exact2"] += 1
        elif gold_t in T.NUMTOK.findall(s["subq"]) and gold_t not in qnums:
            skipped["subq_shows_gold"] += 1
        elif step_desc(s["subq"]) in seen:
            skipped["same_text_same_value"] += 1
        else:
            seen.add(step_desc(s["subq"]))
            pool.append((step_desc(s["subq"]), v, j))
            continue
    quantities = [x for s in steps for x in [s["value"]] + subtree_values(s["tree"])]
    return {"pool": pool, "steps": [(s["value"], j) for j, s in enumerate(steps)], "gold": gold,
            "given": given_numbers(p["question"]), "quantities": quantities, "pool_skipped": skipped,
            "level": {j: s["level"] for j, s in enumerate(steps)}, "n_steps": len(steps)}


def classify(v: float, info: dict) -> tuple[str | None, str | None, set, int | None]:
    """(kind, canonical description, every true description, step index) of option value v; kind None = no node
    (the reason in the description slot)."""
    S = T.soc()
    close = lambda x: S.close(x, v)  # noqa: E731
    given = any(close(x) for x in info["given"])
    true = {DESC_GIVEN} if given else set()
    if close(info["gold"]):
        return "gold", DESC_ANSWER, true | {DESC_ANSWER}, None
    hits = [(d, j) for d, x, j in info["pool"] if close(x)]
    if len({d for d, _ in hits}) > 1:
        return None, "ambiguous_steps", true, None
    if hits:
        return "step", hits[0][0], true | {hits[0][0]}, hits[0][1]
    if any(close(x) for x, _ in info["steps"]):
        return None, "unnamed_step_value", true, None
    used = any(close(x) for x in info["quantities"])
    if given and not used:                   # written in the problem, never used: "given" and "none" both true
        return None, "given_unused", true, None
    if given:
        return "given", DESC_GIVEN, true, None
    if used:
        return None, "unnamed_solution_quantity", true, None
    return "none", DESC_NONE, {DESC_NONE}, None


def descriptions(info: dict) -> list[str]:
    return [DESC_ANSWER, DESC_NONE, DESC_GIVEN] + [d for d, _, _ in info["pool"]]


def length_rank(c: str, choices: list[str]) -> int:
    return sum(len(x) < len(c) for x in choices if x != c)


def pick_choices(c: str, false: list[str], forced: str | None, rng: random.Random) -> list[str]:
    """The 3 distractors of a node with canonical description c: A and N when false, the forced control target, the
    rest from the other false descriptions so that c's length rank is the first realisable one of a seeded order."""
    must = [d for d in (DESC_ANSWER, DESC_NONE) if d in false]
    if forced is not None and forced != c and forced not in must:
        must.append(forced)
    rest = [d for d in false if d not in must]
    m = K - 1 - len(must)
    combos = [list(x) for x in itertools.combinations(rest, m)] if m > 0 else [[]]
    by_rank = collections.defaultdict(list)
    for x in combos:
        by_rank[length_rank(c, [c] + must + x)].append(x)
    ranks = list(range(K))
    rng.shuffle(ranks)
    r = next(r for r in ranks if by_rank[r])
    return must + rng.choice(by_rank[r])


def verify_nodes(item: dict, p: dict, seed: int, none_nodes: int | None, st: collections.Counter) -> list[dict]:
    """The verification nodes of one training row (p: its parsed problem)."""
    pid = item["problem_id"]
    info = problem_info(p)
    st.update({f"pool_skipped_{k}": v for k, v in info["pool_skipped"].items()})
    st["pool_steps"] += len(info["pool"])
    allD = descriptions(info)
    cands, nones = [], []
    for L, o, ok in zip(item["labels"], item["options"], item["option_kinds"]):
        kind, desc, true, j = classify(float(o), info)
        st["options"] += 1
        if kind is None:
            st[f"skipped_{desc}"] += 1
            continue
        false = [d for d in allD if d not in true]
        if len(false) < K - 1:
            st[f"skipped_few_choices_{kind}"] += 1
            continue
        rec = {"letter": L, "value": str(o), "kind": kind, "desc": desc, "true": true, "false": false, "step": j,
               "option_kind": ok}
        (nones if kind == "none" else cands).append(rec)
    if none_nodes is not None and len(nones) > none_nodes:
        st["none_options_not_asked"] += len(nones) - none_nodes
        nones = random.Random(f"{seed}/{pid}/vt/none").sample(nones, none_nodes)
    nodes = sorted(cands + nones, key=lambda r: item["labels"].index(r["letter"]))
    if not nodes:
        return []
    perm = list(range(len(nodes)))
    random.Random(f"{seed}/{pid}/vt/control").shuffle(perm)
    out = []
    for i, r in enumerate(nodes):
        tgt = nodes[perm[i]]["desc"]
        if tgt != r["desc"] and tgt in r["true"]:          # true of this option without being its canonical one
            st["control_kept_true_noncanonical"] += 1
            tgt = r["desc"]
        nid = f"v{r['letter']}"
        rng = random.Random(f"{seed}/{pid}/{nid}/vt")
        dis = pick_choices(r["desc"], r["false"], tgt if tgt != r["desc"] else None, rng)
        opts = [r["desc"]] + dis
        random.Random(f"{seed}/{pid}/{nid}/vt/order").shuffle(opts)
        kinds = [KIND_OF.get(d) or "step:%d" % (next(j for dd, _, j in info["pool"] if dd == d) + 1) for d in opts]
        q = node_question(r["letter"], r["value"])
        a, ac = opts.index(r["desc"]), opts.index(tgt)
        out.append({"id": nid, "type": "verify", "option": r["letter"], "value": r["value"], "vt_kind": r["kind"],
                    "step": None if r["step"] is None else r["step"] + 1, "option_kind": r["option_kind"],
                    "node_depth": None, "depends_on": [], "control_agree": a == ac,
                    "renderings": {"vt": {"question": q, "options": opts, "answer": a, "kinds": kinds},
                                   "vtr": {"question": q, "options": opts, "answer": ac, "kinds": kinds}}})
        st[f"nodes_{r['kind']}"] += 1
        st["nodes"] += 1
        st["control_agree"] += a == ac
    return out


# ------------------------------------------------------------------------------------------------------ audit
def audit_item(item: dict, p: dict, seed: int) -> list[str]:
    """Problems of one built row (empty: clean), from its fields and its parsed problem; the control answers are
    re-derived from the seed (the permutation of the canonical descriptions over the stored nodes)."""
    bad = []
    info = problem_info(p)
    nodes = item.get(FIELD)
    if not isinstance(nodes, list):
        return [f"{item.get('item_id')}: no {FIELD} list"]
    opt = dict(zip(item["labels"], item["options"]))
    trues = []
    for n in nodes:
        nid = n["id"]
        if n["option"] not in opt or str(opt[n["option"]]) != n["value"] or nid != f"v{n['option']}":
            bad.append(f"{nid}: option / value / id")
            trues.append(set())
            continue
        kind, desc, true, _ = classify(float(n["value"]), info)
        trues.append(true)
        r, rc = n["renderings"]["vt"], n["renderings"]["vtr"]
        o, a = r["options"], r["answer"]
        if kind != n["vt_kind"] or len(o) != K or len(set(o)) != K or o[a] != desc or any("\n" in x for x in o):
            bad.append(f"{nid}: kind {kind} / options {o} / answer {a}")
            continue
        if any(x in true for i, x in enumerate(o) if i != a):
            bad.append(f"{nid}: a distractor is true of {n['value']}")
        if any(x not in descriptions(info) for x in o) or not {DESC_ANSWER, DESC_NONE} <= set(o):
            bad.append(f"{nid}: choices outside the pool or A / N missing")
        if r["question"] != node_question(n["option"], n["value"]) or rc["question"] != r["question"] \
                or rc["options"] != o or not 0 <= rc["answer"] < K or n["control_agree"] != (rc["answer"] == a):
            bad.append(f"{nid}: question / vtr")
    if not bad and nodes:
        perm = list(range(len(nodes)))
        random.Random(f"{seed}/{item['problem_id']}/vt/control").shuffle(perm)
        for i, n in enumerate(nodes):
            r = n["renderings"]["vt"]
            want = nodes[perm[i]]["renderings"]["vt"]
            want = want["options"][want["answer"]]
            if want != r["options"][r["answer"]] and want in trues[i]:
                want = r["options"][r["answer"]]
            if n["renderings"]["vtr"]["options"][n["renderings"]["vtr"]["answer"]] != want:
                bad.append(f"{n['id']}: control target is not the permuted description")
    return [f"{item.get('item_id')}: {b}" for b in bad]


# ------------------------------------------------------------------------------------------------------ reports
def node_stats(items: list[dict]) -> dict:
    nodes = [n for it in items for n in it[FIELD]]
    r = [n["renderings"]["vt"] for n in nodes]
    per = collections.Counter(len(it[FIELD]) for it in items)
    offered, right = collections.Counter(), collections.Counter()
    short = long_ = 0
    ranks = collections.Counter()
    for x in r:
        for i, k in enumerate(x["kinds"]):
            k = k.split(":")[0]
            offered[k] += 1
            right[k] += i == x["answer"]
        L = [len(o) for o in x["options"]]
        short += L[x["answer"]] == min(L) and L.count(min(L)) == 1
        long_ += L[x["answer"]] == max(L) and L.count(max(L)) == 1
        ranks[length_rank(x["options"][x["answer"]], x["options"])] += 1
    prior = {k: round(right[k] / offered[k], 4) for k in sorted(offered)}
    best = 0
    for x in r:                                   # pick the offered kind with the highest prior (first on ties)
        ks = [k.split(":")[0] for k in x["kinds"]]
        best += max(range(K), key=lambda i: prior[ks[i]]) == x["answer"]
    n = max(1, len(nodes))
    return {"items": len(items), "items_with_nodes": sum(bool(it[FIELD]) for it in items), "nodes": len(nodes),
            "nodes_per_item": round(len(nodes) / max(1, len(items)), 2),
            "nodes_per_item_hist": dict(sorted(per.items())),
            "vt_kind": dict(collections.Counter(nd["vt_kind"] for nd in nodes)),
            "vt_kind_share": {k: round(v / n, 4) for k, v in collections.Counter(nd["vt_kind"] for nd in nodes).items()},
            "option_kind_by_vt_kind": {f"{a}/{b}": c for (a, b), c in sorted(collections.Counter(
                (nd["vt_kind"], nd["option_kind"]) for nd in nodes).items())},
            "answer_position": dict(sorted(collections.Counter("ABCD"[x["answer"]] for x in r).items())),
            "control_answer_position": dict(sorted(collections.Counter(
                "ABCD"[nd["renderings"]["vtr"]["answer"]] for nd in nodes).items())),
            "length_rank_of_answer": dict(sorted(ranks.items())),
            "shortest_pick_acc": round(short / n, 4), "longest_pick_acc": round(long_ / n, 4),
            "type_prior": {"offered": dict(offered), "right_when_offered": prior,
                           "best_prior_pick_acc": round(best / n, 4)},
            "control_agree_share": round(sum(nd["control_agree"] for nd in nodes) / n, 4),
            "distinct_choice_texts": len({o for x in r for o in x["options"]})}


def choice_features(item: dict, node: dict) -> list[list[float]]:
    """Per-choice features of the description-only adversary (no problem, no options, no value)."""
    r = node["renderings"]["vt"]
    L = [len(o) for o in r["options"]]
    srt = sorted(range(K), key=lambda i: (L[i], i))
    steps = [int(k.split(":")[1]) for k in r["kinds"] if k.startswith("step:")]
    out = []
    for i, k in enumerate(r["kinds"]):
        kk = k.split(":")[0]
        f = [float(kk == x) for x in ("answer", "none", "given", "step")] + [L[i] / 50.0] + \
            [float(srt.index(i) == j) for j in range(K)] + [float(i == j) for j in range(K)]
        if kk == "step":
            s = int(k.split(":")[1])
            f += [s / max(1, item["n_steps"]), float(s == max(steps)), float(s == min(steps))]
        else:
            f += [0.0, 0.0, 0.0]
        out.append(f)
    return out


def adversary(train: list[dict], val: list[dict]) -> dict:
    import numpy as np
    import build_math_evals as M
    def xy(items):
        X, y = [], []
        for it in items:
            for n in it[FIELD]:
                X.append(np.array(choice_features(it, n)))
                y.append(n["renderings"]["vt"]["answer"])
        return X, y
    Xt, yt = xy(train)
    Xv, yv = xy(val)
    m = M.fit_logit(Xt, yt)
    return {"chance": 0.25, "train_acc": round(M.logit_acc(m, Xt, yt), 4), "val_acc": round(M.logit_acc(m, Xv, yv), 4),
            "n_train": len(yt), "n_val": len(yv)}


def show_problem(item: dict, answer: str) -> str:
    out = [f"=== {item['item_id']}  (gold {T.gold_text(item)}, {item['n_steps']} steps, {len(item[FIELD])} verification "
           "nodes)", item["prompt"], "", "GSM8K solution:"]
    out += ["  " + line for line in answer.strip().split("\n")]
    out.append("")
    for n in item[FIELD]:
        r, rc = n["renderings"]["vt"], n["renderings"]["vtr"]
        out.append(f"[{n['id']}] option {n['option']} = {n['value']}: {n['vt_kind']}"
                   f"{' (step %d)' % n['step'] if n['step'] else ''}, final option kind {n['option_kind']}")
        out.append(f"    {r['question']}")
        for i, (L, o) in enumerate(zip("ABCD", r["options"])):
            mark = ("*" if i == r["answer"] else " ") + ("c" if i == rc["answer"] else " ")
            out.append(f"   {mark} {L}) {o}   [{r['kinds'][i]}]")
    out.append("    (* = vt answer, c = vtr control answer)")
    return "\n".join(out) + "\n"


def pick_audit(items: list[dict]) -> list[dict]:
    """gsm8k-train-660, then the first train items with a step node, a given node, a 2-step problem and a control that
    keeps no node's own description."""
    by = {it["item_id"]: it for it in items}
    out = [by["gsm8k-train-660"]] if "gsm8k-train-660" in by else []
    tests = [lambda it: any(n["vt_kind"] == "step" for n in it[FIELD]),
             lambda it: any(n["vt_kind"] == "given" for n in it[FIELD]),
             lambda it: it["n_steps"] == 2 and len(it[FIELD]) >= 2,
             lambda it: len(it[FIELD]) >= 3 and not any(n["control_agree"] for n in it[FIELD])]
    for f in tests:
        for it in items:
            if it not in out and it[FIELD] and f(it):
                out.append(it)
                break
    return out


# ------------------------------------------------------------------------------------------------------ build
def dumps(r: dict) -> str:
    return json.dumps(r, ensure_ascii=False)


def none_arg(x: str) -> int | None:
    return None if x == "all" else int(x)


def build(args) -> None:
    import pandas as pd
    S = T.soc()
    src, out = ROOT / args.src, ROOT / args.out
    df = pd.read_parquet(T.RAW / "train-00000-of-00001.parquet")
    st = collections.Counter()
    fails, built, answers, parsed = [], {}, {}, {}
    nn = none_arg(args.none_nodes)
    for split in ("train", "val"):
        built[split] = []
        with open(src / f"{split}.jsonl") as f:
            lines = [x for x in f.read().split("\n") if x]           # not splitlines: U+2028 in texts
        for line in lines:
            r = json.loads(line)
            i = int(r["problem_id"].rsplit("-", 1)[1])
            p = S.parse_problem(i, df.question[i], df.answer[i])
            if p["status"] != "ok" or f"Question: {p['question']}\nOptions:\n" != r["prompt"][:len(p["question"]) + 20]:
                fails.append(f"{r['item_id']}: raw row {i} does not reproduce the prompt")
                continue
            if T._fmt(p["gold"]) != T.gold_text(r) or len(p["steps"]) != r["n_steps"]:
                fails.append(f"{r['item_id']}: gold / step count differs from the raw solution")
                continue
            o = dict(r)
            o[FIELD] = verify_nodes(r, p, args.seed, nn, st)
            if dumps({k: v for k, v in o.items() if k != FIELD}) != line:
                fails.append(f"{r['item_id']}: row differs from {args.src} beyond {FIELD}")
            built[split].append(o)
            answers[r["item_id"]] = df.answer[i]
            parsed[r["item_id"]] = p
    for it in built["train"] + built["val"]:
        fails += audit_item(it, parsed[it["item_id"]], args.seed)
    summary = {"script": "scripts/build_gsm8k_vt.py", "args": vars(args), "field": FIELD, "md5": {}, "files": {}}
    for split in ("train", "val"):
        key = f"{out.name}/{split}.jsonl"
        summary["md5"][key] = T.write(out / f"{split}.jsonl", built[split])
        summary["files"][key] = node_stats(built[split])
    summary["source_md5"] = {f"{args.src}/{s}.jsonl": T.md5(src / f"{s}.jsonl") for s in ("train", "val")}
    summary["build"] = dict(sorted(st.items()))
    summary["audit"] = {"failures": len(fails), "examples": fails[:20]}
    summary["description_only_adversary"] = adversary(built["train"], built["val"])
    picks = pick_audit(built["train"])
    (out / "node_audit.txt").write_text("\n".join(show_problem(it, answers[it["item_id"]]) for it in picks))
    summary["node_audit"] = [it["item_id"] for it in picks]
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("files", "build", "audit", "description_only_adversary")}, indent=1,
                     ensure_ascii=False))
    print(f"rss {T.rss_mb()} MB")
    if fails:
        print("AUDIT FAILED", fails[:10])
        sys.exit(1)


def templates() -> dict:
    """The prompt templates of cotdistill.student, read from its source (importing it needs torch)."""
    tree = ast.parse((ROOT / "src/cotdistill/student.py").read_text())
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and getattr(n.targets[0], "id", "") in (
                "SUBQ_TEMPLATE", "SUBQ_OPTS_TEMPLATE", "FINAL_TEMPLATE"):
            out[n.targets[0].id] = ast.literal_eval(n.value)
    return out


def tokens(args) -> None:
    """Tokens per epoch (every view once, the trainer's templates, no special tokens) of A2 (finals), GFPQ (finals +
    the yes/no views in the statements+question view) and GFPQVT (finals + the verification views in the problem
    view), on the Qwen3.5 tokenizer of build_gsm8k_tf.TOKENIZER."""
    from tokenizers import Tokenizer
    from cotdistill.treeplan import mc_options_block, problem_no_options
    tok = Tokenizer.from_file(str(ROOT / T.TOKENIZER))
    tp = templates()
    n_tok = lambda s: len(tok.encode(s, add_special_tokens=False).ids)   # noqa: E731
    rep = {"tokenizer": T.TOKENIZER, "files": {}}
    for split in ("train", "val"):
        fin = gfpq = vt = 0
        n_vt, lens = 0, []
        for it in T.read_jsonl(ROOT / args.out / f"{split}.jsonl"):
            fin += n_tok(tp["FINAL_TEMPLATE"].format(problem=it["prompt"]))
            ctx = problem_no_options(it)
            for n in it["subqs"]:
                gfpq += n_tok(tp["SUBQ_TEMPLATE"].format(problem=ctx, question=n["question"], a="Yes", b="No"))
            for n in it[FIELD]:
                r = n["renderings"]["vt"]
                x = n_tok(tp["SUBQ_OPTS_TEMPLATE"].format(problem=it["prompt"], question=r["question"],
                                                          opts=mc_options_block(r)))
                vt += x
                lens.append(x)
                n_vt += 1
        lens.sort()
        rep["files"][f"{args.out}/{split}.jsonl"] = {
            "epoch_tokens_A2": fin, "epoch_tokens_GFPQ": fin + gfpq, "epoch_tokens_GFPQVT": fin + vt,
            "GFPQ_over_A2": round((fin + gfpq) / max(1, fin), 2), "GFPQVT_over_A2": round((fin + vt) / max(1, fin), 2),
            "GFPQVT_over_GFPQ": round((fin + vt) / max(1, fin + gfpq), 2),
            "vt_view": {"n": n_vt, "mean": round(sum(lens) / max(1, n_vt), 1), "p50": lens[len(lens) // 2] if lens else 0,
                        "max": lens[-1] if lens else 0}}
    (ROOT / args.out / "tokens.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/student_gsm8k_tf_si")
    ap.add_argument("--out", default="data/student_gsm8k_tf_vt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--none-nodes", default="1", help="none options asked per problem (seeded draw), or 'all'")
    ap.add_argument("--tokens", action="store_true", help="only the tokens-per-epoch report (needs tokenizers)")
    args = ap.parse_args()
    if args.tokens:
        tokens(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
