"""Tree Forcing data for the numeric-chain domain (numgen): data/student_numgen_tf, a depth-extrapolation split with
exact, locally sufficient step trees (notes/next_round/tree_forcing/numgen_domain.md).

Why new nodes: the subqs of data/student_numgen are G3-style probes, not a proof tree. Their depends_on links only the
N1 node of each intermediate ancestor (stated quantities are never nodes), the operation and its constant are in no
node, and half of the N1 parents are "No" ("Is q2 equal to 51? No"), which pins nothing. A derived node's local view
(parents' results only) can therefore not determine its answer. This script rebuilds the tree from the program
(item["program"]["quantities"]: op, inputs, const, value, sentence), which numgen.py keeps exact.

Split: items with depth (number of operations) <= --max-depth (default 4) from data/student_numgen train/val; finals
(item_id, prompt, labels, gold_label, options, teacher, program, ...) are byte-identical to student_numgen, only
subqs / local fields / tree flags change. Evaluation: data/eval/numgen_d1..d8 and numgen_ood unchanged (d5-d8 are
beyond the training depth; d7-d8 beyond the original numgen range too).

Nodes per item (fields as data/student_v3sl, consumed by treeplan.local_context / student._local_views; every node a
yes/no question with program truth, p_cot = truth, node_depth from treeplan.node_depths):
  read    q with no inputs (a stated count, or "n dozen"): "{q}v" "Is <name> equal to <value>?" (Yes) and "{q}n" the
          same with a wrong number (No); local_lines = [the sentence that states it]
  parse   operation q (any op with a sentence): "{q}r" "Is <name> equal to <formula over its inputs' names and the
          constant>?" (Yes) and "{q}s" with a wrong formula (No: the inverse operation, or the constant +-1);
          local_lines = [the operation's sentence]
  derive  operation q: "{q}v" "Is <name> equal to <value>?" (Yes) and "{q}n" with a planted-slip or perturbed value (No,
          above or below the value 50/50); depends_on = the "v" nodes of its inputs + its own "r" node, so the local
          view states exactly the input values and the formula
Only "Yes" nodes are ever parents, so every stated result pins its quantity or formula. Not built (as numgen's own
nodes): value/check nodes of the answer quantity and of any operation whose value equals the gold (the final question
in disguise in problem views), and therefore of every operation downstream of such a quantity (its view would miss a
parent); their parse nodes are kept. Items get local_preamble = the frame's intro sentence (no numbers) and
tree_fix = True (post-audit matching in full_tree_plan for the TFLM control).

Sufficiency audit (exit 1 on any failure): every derived node's TFL view is rendered with treeplan.local_context and
re-solved from its TEXT alone (value facts and the formula fact parsed back, the formula evaluated on the stated
values), and the result must equal the node's value and decide its truth; every view must pass view_leak, contain
only the preamble and its own lines/results, and never state the gold of the item; read/parse leaves must find their
number/constant in their own line (numgen.stated_numbers). Also reported: TFLM matched coverage (full_tree_plan, cap 0)
and single-feature cue baselines for the yes/no targets.

  python3 scripts/build_numgen_tf.py                      # data/student_numgen_tf, max depth 4
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
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill import numgen as ng  # noqa: E402
from cotdistill.treeplan import (LOCAL_HEADER, full_tree_plan, local_context, node_depths, view_leak)  # noqa: E402

LOCAL_DEEP_PER_DEPTH = 100
CHANGED = {"subqs", "has_tree", "tree_source", "local_preamble", "tree_fix"}     # everything else stays byte-identical


# ------------------------------------------------------------------------------------------------- program access
class Q:
    """A quantity of an item's program, with the numgen.Qty attributes slips()/execute() read."""

    def __init__(self, d: dict):
        self.id, self.kind, self.value, self.stated = d["id"], d["kind"], d["value"], d["stated"]
        self.op, self.inputs, self.const = d["op"], list(d["inputs"]), d["const"]
        self.np, self.sentence = d["name"], d["sentence"]

    def fmt(self, x: int) -> str:
        return f"${x}" if self.kind == "money" else (f"{x} minutes" if self.kind == "minutes" else str(x))


class St:
    def __init__(self, program: dict):
        self.qs = [Q(d) for d in program["quantities"]]
        self._by = {q.id: q for q in self.qs}

    def by(self, i: str) -> Q:
        return self._by[i]


# --------------------------------------------------------------------------------------------------- formulas
def const_text(q: Q, c: int, cls: str) -> str:
    """Additive constants carry the target's unit ($, minutes); factors, divisors and percentages are bare."""
    return q.fmt(c) if cls in ("ADD", "SUB") else str(c)


def formula(q: Q, st: St, cls: str, c) -> str:
    names = [st.by(i).np for i in q.inputs]
    if cls == "ADD":
        return f"{names[0]} plus {const_text(q, c, cls)}"
    if cls == "SUB":
        return f"{names[0]} minus {const_text(q, c, cls)}"
    if cls == "MUL":
        return f"{names[0]} multiplied by {c}"
    if cls == "DIV":
        return f"{names[0]} divided by {c}"
    if cls == "PCT":
        return f"{c}% of {names[0]}"
    if cls == "PCTOFF":
        return f"{names[0]} reduced by {c}%"
    if cls == "SUM":
        return f"{names[0]} plus {names[1]}"
    if cls == "DIFF":
        return f"the difference between {names[0]} and {names[1]}"
    raise ValueError(cls)


SWAP = {"ADD": "SUB", "SUB": "ADD", "MUL": "DIV", "DIV": "MUL", "PCT": "PCTOFF", "PCTOFF": "PCT", "SUM": "DIFF"}


def wrong_formula(q: Q, st: St, rng: random.Random) -> tuple[str, str]:
    cls, c = ng.OPCLS[q.op], q.const
    if c is not None and rng.random() < 0.5:
        for _ in range(10):
            c2 = c + rng.choice([-1, 1])
            if c2 >= 1 and not (cls in ("PCT", "PCTOFF") and c2 >= 100) and not (cls in ("MUL", "DIV") and c2 == 1):
                return formula(q, st, cls, c2), "off1"
    return formula(q, st, SWAP[cls], c), "swap"


# ------------------------------------------------------------------------------------------------ wrong values
def wrong_value(v: int, slip_vals: list[int], rng: random.Random) -> tuple[int, str]:
    """A wrong number for a check node: above or below the value 50/50 (so "asked number is larger" carries no
    signal), a planted-slip value of that side with probability .6 when one exists, else a perturbation."""
    hi = rng.random() < 0.5
    for side in (hi, not hi):
        sl = [x for x in slip_vals if x != v and (x > v) == side]
        if sl and rng.random() < 0.6:
            return rng.choice(sl), "slip"
        d = rng.choice([1, 2] if v < 20 else [1, 2, 5, 10])
        cands = [v + d, v * 2] if side else [v - d] + ([v // 2] if v % 2 == 0 else [])
        cands = [x for x in cands if 0 < x <= ng.MAXV and x != v]
        if cands:
            return rng.choice(cands), "perturb"
    raise RuntimeError(f"no wrong value for {v}")


# ------------------------------------------------------------------------------------------------------- tree
def node(nid: str, typ: str, question: str, truth: bool, deps: list[str], value, probe=None, lines=None, **kw) -> dict:
    out = {"question": question, "p_cot": 1.0 if truth else 0.0, "p_fresh": None, "truth": truth, "type": typ,
           "depends_on": deps, "id": nid, "initial": None, "status": "direct", "value": value}
    if probe is not None:
        out["probe"] = probe
    if lines:
        out["local_lines"] = lines
    out.update(kw)
    return out


def preamble(item: dict) -> str:
    p = item["program"]
    f = ng.FRAME_BY_ID[p["frame"]]
    for poss in ("her", "his"):
        intro = f["intro"].format(N=p["name"], poss=poss)
        if item["prompt"].startswith("Question: " + intro + " "):
            return intro
    raise AssertionError(f"{item['item_id']}: intro not found")


def tf_tree(item: dict, rng: random.Random, stats: collections.Counter) -> list[dict]:
    st = St(item["program"])
    ans = st.by(item["program"]["answer"])
    gold = ans.value
    blocked: set = {ans.id}
    for q in st.qs:
        if not q.stated and q.op != "dozen" and q.inputs and q.value == gold and q.id != ans.id:
            blocked.add(q.id)
            stats["blocked_gold_valued_op"] += 1
    nodes = []
    have_v: set = set()
    for q in st.qs:
        leaf = not q.inputs
        if not leaf and q.sentence:                                   # parse leaves: the formula of the operation
            cls = ng.OPCLS[q.op]
            wf, wk = wrong_formula(q, st, rng)
            nodes.append(node(f"{q.id}r", "parse", f"Is {q.np} equal to {formula(q, st, cls, q.const)}?", True, [],
                              q.value, lines=[q.sentence], qid=q.id, op=q.op))
            nodes.append(node(f"{q.id}s", "parse", f"Is {q.np} equal to {wf}?", False, [], q.value,
                              lines=[q.sentence], qid=q.id, op=q.op, wrong=wk))
        elif not leaf:
            stats["implicit_op_" + q.op] += 1                         # total / hours: final question only
        if q.id in blocked:
            continue
        if leaf:
            assert q.sentence, q.id
            x, k = wrong_value(q.value, [], rng)
            nodes.append(node(f"{q.id}v", "read", f"Is {q.np} equal to {q.fmt(q.value)}?", True, [], q.value,
                              probe=q.value, lines=[q.sentence], qid=q.id, op=q.op or "stated"))
            nodes.append(node(f"{q.id}n", "read", f"Is {q.np} equal to {q.fmt(x)}?", False, [], q.value,
                              probe=x, lines=[q.sentence], qid=q.id, op=q.op or "stated", wrong=k))
            have_v.add(q.id)
            continue
        if not all(i in have_v for i in q.inputs) or not q.sentence:
            stats["dropped_downstream_of_blocked"] += 1
            continue
        deps = [f"{i}v" for i in q.inputs] + [f"{q.id}r"]
        x, k = wrong_value(q.value, ng.slips(st, q.id), rng)
        nodes.append(node(f"{q.id}v", "derive", f"Is {q.np} equal to {q.fmt(q.value)}?", True, deps, q.value,
                          probe=q.value, qid=q.id, op=q.op))
        nodes.append(node(f"{q.id}n", "derive", f"Is {q.np} equal to {q.fmt(x)}?", False, deps, q.value,
                          probe=x, qid=q.id, op=q.op, wrong=k))
        have_v.add(q.id)
    depth = node_depths(nodes)
    for n in nodes:
        n["node_depth"] = depth[n["id"]]
    return nodes


def convert(item: dict, rng: random.Random, stats: collections.Counter) -> dict:
    out = dict(item)
    out["subqs"] = tf_tree(item, rng, stats)
    out["local_preamble"] = preamble(item)
    out["tree_fix"] = True
    out["has_tree"] = bool(out["subqs"])
    out["tree_source"] = "numgen_program_tf"
    return out


# ------------------------------------------------------------------------------------------------------ audit
_FACT = re.compile(r"^Q: Is (.+) equal to (.+)\? A: (Yes|No)$")
_NUM = re.compile(r"^\$?(\d+)(?: minutes)?$")
_ASK = re.compile(r"^Is (.+) equal to (\$?\d+(?: minutes)?)\?$")
FORMS = [(re.compile(r"^(\d+)% of (.+)$"), "PCT", (2,), 1),
         (re.compile(r"^(.+) reduced by (\d+)%$"), "PCTOFF", (1,), 2),
         (re.compile(r"^(.+) multiplied by (\d+)$"), "MUL", (1,), 2),
         (re.compile(r"^(.+) divided by (\d+)$"), "DIV", (1,), 2),
         (re.compile(r"^(.+) plus (.+)$"), "PLUS", (1, 2), None),
         (re.compile(r"^(.+) minus (\$?\d+(?: minutes)?)$"), "SUB", (1,), 2)]


def solve_from_text(ctx: str, target_np: str) -> Fraction | None:
    """Re-solve a derived node from its rendered local view: parse the stated value facts and the one formula fact
    about target_np, evaluate the formula on the stated values. None if the text does not determine the value."""
    body = ctx.split(f"\n\n{LOCAL_HEADER}\n", 1)
    if len(body) != 2:
        return None
    vals, formula_rhs = {}, None
    for line in body[1].split("\n"):
        m = _FACT.match(line)
        if not m or m.group(3) != "Yes":
            return None
        lhs, rhs = m.group(1), m.group(2)
        n = _NUM.match(rhs)
        if n:
            vals[lhs] = Fraction(int(n.group(1)))
        elif lhs == target_np:
            formula_rhs = rhs
        else:
            return None
    if formula_rhs is None:
        return None

    def term(t: str):
        if t in vals:
            return vals[t]
        n = _NUM.match(t)
        return Fraction(int(n.group(1))) if n else None

    for rx, cls, name_groups, const_group in FORMS:
        m = rx.match(formula_rhs)
        if not m:
            continue
        if cls == "PLUS":
            a, b = term(m.group(1)), term(m.group(2))
            if a is None or b is None or m.group(1) not in vals:
                continue
            return a + b
        a = term(m.group(name_groups[0]))
        c = int(m.group(const_group).lstrip("$").split()[0])
        if a is None or m.group(name_groups[0]) not in vals:
            continue
        return ng.apply(cls, [a], c)
    return None


def number_of(probe_text: str) -> int:
    return int(_NUM.match(probe_text).group(1))


def audit_item(item: dict, rng: random.Random, a: collections.Counter, fails: list, cue: list) -> None:
    nodes = item["subqs"]
    by = {n["id"]: n for n in nodes}
    pre = item["local_preamble"]
    gold = item["options"][item["gold_label"]]
    if re.search(r"\d", pre):
        fails.append((item["item_id"], "preamble has a digit"))
    for n in nodes:
        a[f"nodes/{n['type']}"] += 1
        a[f"yes/{n['type']}"] += n["truth"]
        a[f"node_depth/{n['node_depth']}"] += 1
        bad = [d for d in n["depends_on"] if d not in by or not by[d]["truth"]]
        if bad:
            fails.append((item["item_id"], f"{n['id']}: parent missing or not Yes: {bad}"))
            continue
        if n.get("local_lines"):
            ctx = local_context(item, n, [], rng=rng)
            if ctx != pre + "\n\n" + n["local_lines"][0] or n["local_lines"][0] not in item["prompt"]:
                fails.append((item["item_id"], f"{n['id']}: leaf line not a prompt sentence"))
            nums = ng.stated_numbers(n["local_lines"][0])
            if n["type"] == "read" and n["truth"]:
                ok = n["value"] in nums or (n["op"] == "dozen" and n["value"] % 12 == 0 and n["value"] // 12 in nums)
                a["read_value_in_line"] += ok
                if not ok:
                    fails.append((item["item_id"], f"{n['id']}: stated value not in its line"))
            if n["type"] == "parse" and n["truth"]:
                c = next(iter(q["const"] for q in item["program"]["quantities"] if q["id"] == n["qid"]))
                ok = c is None or c in nums or (n["op"] == "split" and (c - 1 in nums or (c == 2 and " a friend " in
                                                                                     n["local_lines"][0])))
                a["parse_const_in_line"] += ok
                a["parse_nodes_true"] += 1
                if not ok:
                    a["parse_const_not_in_line"] += 1
                    fails.append((item["item_id"], f"{n['id']}: constant {c} not in its line"))
            stated = []
        else:
            stated = [by[d] for d in n["depends_on"]]
            ctx = local_context(item, n, stated, rng=rng)
            m = _ASK.match(n["question"])
            v = solve_from_text(ctx, m.group(1))
            a["derive_views"] += 1
            if v is None or v != n["value"]:
                fails.append((item["item_id"], f"{n['id']}: view does not determine the value ({v} vs {n['value']})"))
            elif (number_of(m.group(2)) == v) != n["truth"]:
                fails.append((item["item_id"], f"{n['id']}: re-solved truth differs"))
            else:
                a["derive_views_sufficient"] += 1
            for x in stated:
                if x["type"] != "parse" and x["value"] == gold:
                    a[f"stated_{x['type']}_value_equals_gold"] += 1
                    if x["type"] == "derive":
                        fails.append((item["item_id"], f"{n['id']}: a stated derived result equals the gold"))
            a["stated_lines"] += len(stated)
            # single-feature cues of the yes/no target in the local view
            ctxnums = [int(t) for t in re.findall(r"\d+", ctx.split(LOCAL_HEADER, 1)[1])]
            p = n["probe"]
            cue.append((n["truth"], p > n["value"] if not n["truth"] else None, p in ctxnums, p % 10 == 0, p % 2 == 0,
                        p > max(ctxnums)))
        extra = ctx.replace(pre, "", 1)
        others = [q["sentence"] for q in item["program"]["quantities"] if q["sentence"]] + item["program"]["noise"]
        if any(x in extra for x in others if x not in (n.get("local_lines") or [])):
            fails.append((item["item_id"], f"{n['id']}: another problem sentence in the view"))
        if view_leak(item, n, stated):
            fails.append((item["item_id"], f"{n['id']}: view_leak"))
    # TFLM coverage (matched non-parent results) under the post-audit plan, cap 0
    for e in full_tree_plan(item, cap=0, rng=rng, subq_target="cot"):
        if e["node"]["type"] != "derive":
            continue
        a["tflm_derive"] += 1
        if e["parents"]:
            a["tflm_matched"] += 1
            ms = [by[j] for j in e["matched"]]
            a["tflm_matched_lines"] += len(ms)
            a["tflm_matched_no_lines"] += sum(not x["truth"] for x in ms)
            a["tflm_matched_rule_of_same_q"] += any(x["type"] == "parse" and x["qid"] == e["node"]["qid"] for x in ms)
            sib = [x for x in ms if x["type"] == "derive" and x["qid"] == e["node"]["qid"]]
            a["tflm_matched_sibling_yes_sufficient"] += any(x["truth"] for x in sib)
            a["tflm_matched_sibling_no"] += any(not x["truth"] for x in sib)
        else:
            a["tflm_frozen"] += 1


# ------------------------------------------------------------------------------------------------------- main
def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/student_numgen")
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--out", default="data/student_numgen_tf")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--deep-depths", default="5,6,7,8", help="eval depths for local_deep.jsonl (local-step probe)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"args": vars(args), "files": {}, "md5": {}}
    fails: list = []
    cue: list = []
    audit = collections.Counter()
    for split, src, keep in [("train", f"{args.base}/train.jsonl", lambda r: r["depth"] <= args.max_depth),
                             ("val", f"{args.base}/val.jsonl", lambda r: r["depth"] <= args.max_depth)] + \
            [(f"local_deep_d{d}", f"{args.eval_dir}/numgen_d{d}.jsonl", None) for d in args.deep_depths.split(",")]:
        rng = random.Random(f"{args.seed}-{split}")
        arng = random.Random(f"{args.seed}-audit-{split}")
        st = collections.Counter()
        path = out / (f"{split}.jsonl" if split in ("train", "val") else "local_deep.jsonl")
        mode = "a" if split.startswith("local_deep") and not split.endswith(args.deep_depths.split(",")[0]) else "w"
        n_in = n_out = 0
        depth_c, nodes_c, kind_c, nd_c = (collections.Counter() for _ in range(4))
        with open(ROOT / src) as fin, open(path, mode) as fout:
            for line in fin:
                r = json.loads(line)
                n_in += 1
                if keep is not None and not keep(r):
                    continue
                if keep is None and n_out >= LOCAL_DEEP_PER_DEPTH:
                    break
                t = convert(r, rng, st)
                diff = {k for k in set(t) | set(r) if t.get(k) != r.get(k)}
                if not diff <= CHANGED:
                    fails.append((r["item_id"], f"fields changed beyond the tree fields: {sorted(diff - CHANGED)}"))
                audit_item(t, arng, audit, fails, cue)
                fout.write(json.dumps(t) + "\n")
                n_out += 1
                depth_c[r["depth"]] += 1
                nodes_c[len(t["subqs"])] += 1
                for n in t["subqs"]:
                    kind_c[f"{n['type']}/{'yes' if n['truth'] else 'no'}"] += 1
                    nd_c[n["node_depth"]] += 1
        tot = sum(k * v for k, v in nodes_c.items())
        summary["files"][split] = {"source": src, "read": n_in, "written": n_out, "depth": dict(sorted(depth_c.items())),
                                   "nodes": tot, "nodes_per_item": round(tot / max(1, n_out), 2),
                                   "node_types": dict(sorted(kind_c.items())), "node_depth": dict(sorted(nd_c.items())),
                                   "build": dict(st)}
        print(split, json.dumps(summary["files"][split]), f"rss {rss_mb()} MB", flush=True)
    for name in ("train.jsonl", "val.jsonl", "local_deep.jsonl"):
        summary["md5"][name] = md5(out / name)
    # cue baselines (derived local views): best single feature, and "asked larger" among No probes
    def stump(idx):
        rows = [(c[0], c[idx]) for c in cue if c[idx] is not None]
        if not rows:
            return None
        acc = sum(t == f for t, f in rows) / len(rows)
        return round(max(acc, 1 - acc), 3)
    no_higher = [c[1] for c in cue if c[1] is not None]
    summary["cues_derive_views"] = {"n": len(cue), "asked_in_context": stump(2), "asked_round10": stump(3),
                                    "asked_even": stump(4), "asked_above_all_context_numbers": stump(5),
                                    "no_probes_above_value": round(sum(no_higher) / max(1, len(no_higher)), 3)}
    summary["audit"] = dict(sorted(audit.items()))
    summary["audit"]["failures"] = len(fails)
    summary["failure_examples"] = fails[:20]
    summary["fields"] = {"item": ["local_preamble", "tree_fix", "tree_source"],
                         "subq": ["id", "type (read/parse/derive)", "question", "truth", "p_cot", "p_fresh",
                                  "depends_on", "node_depth", "initial", "status", "value", "probe", "local_lines",
                                  "qid", "op", "wrong"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("audit", "cues_derive_views")}, indent=1))
    print(f"rss {rss_mb()} MB")
    if fails:
        print("AUDIT FAILED", fails[:10])
        sys.exit(1)


if __name__ == "__main__":
    main()
