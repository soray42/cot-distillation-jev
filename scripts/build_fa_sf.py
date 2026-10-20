"""StepFinder's failure-attribution protocol (Zhu et al., KDD'26, arXiv 2606.03467; repo data/raw/stepfinder/repo,
README + main.py) as one-pass student data (notes/next_round/failure_attribution/sf_protocol.md). ONE MODEL PER
Who&When SUBSET: trained on StepFinder's regenerated trajectories of that subset only (data/raw/stepfinder/repo/data/
<Algorithm-Generated | Hand-Crafted>/train: 1,564 / 2,604), tested on the whole Who&When subset (126 / 58) without the
ground truth. Target = the earliest decisive error step (mistake_step, 0-based over history; every history entry is a
step, as StepFinder encodes them). StepFinder does not deduplicate its training questions against Who&When; neither
does the main variant here, but the overlap is measured and a stricter '_disj' variant drops it.

  cd <scratch> && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2 ~/.venvs/sev/bin/python -P -E \\
      ~/projects/jev/scripts/build_fa_sf.py [--subsets alg,hc] [--trees data/raw/stepfinder_trees/{subset}.jsonl]

Per subset s (alg, hc):
  data/student_fa_sf_<s>/train.jsonl, val.jsonl        step items in the fa1d format (fa_public.make_items: the log, the
                                                       step question, options "Step k [Agent]" under A-Z a-z 0-9, gold,
                                                       the label-derived yes/no nodes as subqs, half of the items with
                                                       the "Correct final answer (for reference)" line, --gt-p), source
                                                       fa_public, so GFPSP runs as in the fa1d runs
                        train_agent.jsonl, val_agent.jsonl   the agent-level items of the same trajectories (not used by
                                                       the planned runs: StepFinder supervises steps only)
                        summary.json                   counts, overlap, val groups, caps / cut counts, baselines, tokens
                                                       per epoch (A2, GFP-S, packed GFPSP), md5s
  data/student_fa_sf_<s>_disj/                         the same val file; train without every trajectory whose question
                                                       overlaps a Who&When question of EITHER subset (exact text, the
                                                       fa_public.OverlapIndex 8-gram criterion of the fa1 builds, or a
                                                       shared GAIA / AssistantBench id); its lines are a subset of the
                                                       main train file, byte for byte
  data/eval/fa_sf_ww_<s>.jsonl                         the Who&When subset as step items (no reference answer, nodes as
                                                       the fa1d test files); trajectories with more than 62 steps do not
                                                       fit the option alphabet and are left out (HC: 19 of 58, listed in
                                                       the summary); the readout scores the 58 with them counted wrong
With --trees (the MC reasoning trees of another agent, schema of src/cotdistill/fa_trees.py output_record: per line
{"traj_id": "alg/trajectory_0000", "subset", "teacher", "nodes": [{"id", "type", "question", "options", "answer",
"parents", "cite_step", "quote"}], "checks": {"pass": ...}}):
  data/student_fa_sf_<s>_t2/, data/student_fa_sf_<s>_disj_t2/   the same items with the label-derived nodes replaced
                                                       by the tree's multiple-choice nodes (treeplan.is_mc_node: options
                                                       + answer, asked like a final question by the full-tree arms);
                                                       trees that fail their own checks or this converter's (pointer
                                                       options exactly "Step k [Agent]" of the trajectory, ids, parents,
                                                       a DAG, 2-62 distinct options, no metadata leak, no copy of the
                                                       final question) are left out and the item keeps no nodes
  v3 (opt-in; without these flags every output above is byte-identical): --trees-addon <add-on lines with {subset},
  fa_trees.addon_record: the teacher's "clean" nodes, merged with the v2 tree by traj_id when their checks passed>
  and / or --pointer-render free (fa_trees.free_render at the item's step cap: no step number in any question or
  option, pointer options "<Agent>: “<excerpt>”", step references "the <Agent> step “<excerpt>”", nodes that cannot be
  rendered dropped) write data/student_fa_sf_<s>_t3/ and _disj_t3/ (--trees-suffix) with a summary.json; node types
  stay answer_step / provenance / content / conflict / agent (+ clean), so --tree-node-types selects them as before

Splits: val = about --val-frac of the subset's trajectories, whole question groups (fa_public.norm_task of the task;
StepFinder has about 18 trajectories per question), drawn by a seeded hash among the questions that overlap no Who&When
question, so val is the same for the main and the disjoint variant and no task is in both val and train. StepFinder's
main.py has no val split: it evaluates on the test set every epoch, keeps the best test-accuracy checkpoint and stops
early on it; the runs here select on this val split instead (sf_protocol.md).
Length policy (the fa1d one): every step of a trajectory is cut to the same per-trajectory cap (at most --cap-max
characters, head 2/3 + tail), the largest whose step prompt with the reference line fits --budget Qwen3.5 tokens
(build_fa_public.fit_caps), for train, val and test alike; the summary counts the items with at least one cut step.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import random
import re
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cotdistill import fa_public as fp  # noqa: E402
from cotdistill import fa_trees as ft  # noqa: E402
from cotdistill.treeplan import MC_SYMBOLS, mc_check, node_depths, statement_block  # noqa: E402


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_fa_public_for_sf", ROOT / "scripts/build_fa_public.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bfp = _load_builder()           # Tok, fit_caps, baselines, describe, pct, md5: the fa1d builder's own helpers

SUBSETS = {"alg": "Algorithm-Generated", "hc": "Hand-Crafted"}
UID_PREFIX = {"alg": "alg", "hc": "han"}            # fa_public.read_stepfinder: f"{sub[:3].lower()}-<file stem>"
PUBLISHED = {   # StepFinder (KDD'26) Table: mean +- sd over seeds, test-selected checkpoints (main.py)
    "alg": {"acc": 29.63, "acc_sd": 1.50, "acc1": 28.84, "acc2": 45.24, "acc3": 60.31, "s_per_traj": 0.61},
    "hc": {"acc": 22.99, "acc_sd": 2.15, "acc1": 21.26, "acc2": 25.29, "acc3": 30.46, "s_per_traj": 3.56}}
TREE_TYPES = ("answer_step", "provenance", "content", "conflict", "agent", "clean")   # clean: v3 add-on nodes
POINTER = re.compile(r"^Step (\d+) \[(.*)\]$")
STEP_WORD = re.compile(r"^\s*step\s*\d+", re.I)


# ------------------------------------------------------------------------------------------------- readers
def read_sf(raw: Path, subset: str, counts: collections.Counter) -> list[dict]:
    """StepFinder's regenerated trajectories of one subset as fa_public records (fa_public.read_stepfinder: every
    history entry a step, gold = mistake_step, gold agent = the agent of that step; flag agent_mismatch when the
    file's mistake_agent names another agent)."""
    pre = UID_PREFIX[subset] + "-"
    out = [r for r in fp.read_stepfinder(str(raw / "stepfinder/repo/data"), counts) if r["uid"].startswith(pre)]
    for r in out:
        r["split"] = "pool"
    return out


def read_ww(raw: Path, subset: str) -> list[dict]:
    """Who&When (data/raw/who_and_when/Who&When/<subset>/<n>.json) as fa_public records, in file-number order: every
    history entry a step (hand-crafted: history[0] is the human's task message), gold = mistake_step, the agent of a
    step = name, else role without its " (...)" suffix (build_whowhen_eval.agent_of); mistake_agent kept as given."""
    d = raw / "who_and_when/Who&When" / SUBSETS[subset]
    out = []
    for fn in sorted(d.glob("*.json"), key=lambda p: int(p.stem)):
        x = json.loads(fn.read_text())
        h = x["history"]
        k = int(x["mistake_step"])
        if not 0 <= k < len(h):
            raise ValueError(f"{fn}: mistake_step {k} outside {len(h)} steps")
        gold = fp.ww_agent(h[k])
        flags = [] if fp._anorm(gold) == fp._anorm(x["mistake_agent"]) else ["agent_mismatch"]
        r = fp._rec(f"ww-{subset}-{fn.stem}", "whowhen", subset, "whowhen", x["question"],
                    x.get("ground_truth", x.get("groundtruth")), [(fp.ww_agent(s), s.get("content", "")) for s in h],
                    k, gold, split="test", ids=fp.uuids(str(x.get("question_ID", "")) + " " + str(x["question"])),
                    flags=flags)
        r["mistake_agent"] = str(x["mistake_agent"])
        r["file"] = fn.name
        r["question_ID"] = str(x.get("question_ID", "")).lower()
        out.append(r)
    return out


# ------------------------------------------------------------------------------------------------- overlap
def overlap(sf: list[dict], ww: dict, df: int, thr: float) -> dict:
    """Per StepFinder record (uid): the Who&When subsets its question overlaps, by criterion: exact (equal
    fa_public.norm_task text), ngram (fa_public.OverlapIndex over the Who&When questions of both subsets, the fa1
    builds' 8-gram containment rule; catches StepFinder's questions that are a Who&When question plus an attachment
    path), id (a shared GAIA / AssistantBench question_ID among the record's uuids)."""
    exact = collections.defaultdict(set)
    for s, rs in ww.items():
        for r in rs:
            exact[fp.norm_task(r["task"])].add(s)
    idx = fp.OverlapIndex([(f"{s}/{r['file']}", r["task"]) for s, rs in ww.items() for r in rs], df=df, thr=thr)
    ids = collections.defaultdict(set)
    for s, rs in ww.items():
        for r in rs:
            if r["question_ID"]:
                ids[r["question_ID"]].add(s)
    out = {}
    for r in sf:
        ratio, who = idx.match(r["task"])
        out[r["uid"]] = {"exact": sorted(exact.get(fp.norm_task(r["task"]), set())),
                         "ngram": sorted({w.split("/")[0] for w in who}) if ratio >= thr else [],
                         "id": sorted({s for i in r["ids"] for s in ids.get(i, ())})}
    return out


def overlap_summary(sf: list[dict], ov: dict, subset: str) -> dict:
    """Distinct questions and trajectories per criterion, within the subset (the test set of this subset's model) and
    across (the other subset's test set)."""
    q = {r["uid"]: fp.norm_task(r["task"]) for r in sf}
    out = {"distinct_questions": len(set(q.values())), "trajectories": len(sf)}
    for crit in ("exact", "ngram", "id", "any"):
        for scope in ("same", "other", "either"):
            def hit(u):
                ss = set().union(*(ov[u][c] for c in ("exact", "ngram", "id"))) if crit == "any" else set(ov[u][crit])
                return bool(ss) if scope == "either" else (subset in ss if scope == "same" else bool(ss - {subset}))
            us = [r["uid"] for r in sf if hit(r["uid"])]
            out[f"{crit}/{scope}"] = {"questions": len({q[u] for u in us}), "trajectories": len(us)}
    return out


def any_overlap(ov: dict, uid: str) -> bool:
    return any(ov[uid][c] for c in ("exact", "ngram", "id"))


# ------------------------------------------------------------------------------------------------- splits
def val_groups(sf: list[dict], ov: dict, frac: float, seed: int, subset: str) -> set:
    """Question groups (norm_task) for val: whole groups among those with no Who&When overlap, in a seeded hash order,
    until the val trajectories reach round(frac x the subset's trajectories)."""
    groups = collections.defaultdict(list)
    for r in sf:
        groups[fp.norm_task(r["task"])].append(r)
    clean = [g for g, rs in groups.items() if not any(any_overlap(ov, r["uid"]) for r in rs)]
    want = round(frac * len(sf))
    chosen, n = set(), 0
    for g in sorted(clean, key=lambda g: fp.stable_hash(seed, "sfval", subset, g)):
        if n >= want:
            break
        chosen.add(g)
        n += len(groups[g])
    return chosen


# ------------------------------------------------------------------------------------------------- items
def cut_stats(recs: list[dict], caps: dict) -> dict:
    """How much the per-trajectory cap cuts: items with at least one cut step, cut steps, tasks over fa_public's
    TASK_CAP, caps at the minimum, mean cap."""
    if not recs:
        return {"n": 0}
    cut_items = cut_steps = steps = 0
    for r in recs:
        c = caps[r["uid"]]
        k = sum(len(s["text"].strip()) > c for s in r["steps"])
        cut_items += k > 0
        cut_steps += k
        steps += len(r["steps"])
    return {"n": len(recs), "items_with_a_cut_step": cut_items, "share_items_cut": round(cut_items / len(recs), 3),
            "cut_steps": cut_steps, "share_steps_cut": round(cut_steps / max(1, steps), 3),
            "tasks_over_task_cap": sum(len(r["task"].strip()) > fp.TASK_CAP for r in recs),
            "cap": bfp.pct([caps[r["uid"]] for r in recs])}


def item_rng(seed: int, uid: str) -> random.Random:
    return random.Random(f"fasf|{seed}|{uid}|items")


def build_items(recs: list[dict], caps: dict, subset: str, split: str, args, fails: list) -> tuple[list, list]:
    """Step and agent items of a split (fa_public.make_items with the fa1d per-item draws: the reference-line coin,
    then the label-derived nodes), in a seeded order; every step item passes fa_public.audit_item."""
    rs = sorted(recs, key=lambda r: fp.stable_hash(args.seed, subset, split, "order", r["uid"]))
    steps, agents = [], []
    for k, r in enumerate(rs):
        rng = item_rng(args.seed, r["uid"])
        with_gt = rng.random() < args.gt_p
        s, a = fp.make_items(r, f"fasf-{subset}-{split}-{k:05d}", f"fa_sf/{subset}", cap=caps[r["uid"]],
                             with_gt=with_gt, rng=rng)
        bad = fp.audit_item(s)
        if bad:
            fails.append((s["item_id"], bad[:3]))
        steps.append(s)
        if a:
            agents.append(a)
    return steps, agents


def build_test(recs: list[dict], caps: dict, subset: str, seed: int, fails: list) -> list[dict]:
    """The Who&When step items (no reference line; nodes drawn as the fa1d test files), file-number order, with the
    benchmark's own mistake_agent label and file name in fa."""
    out = []
    for r in recs:
        s, _ = fp.make_items(r, f"fasf-ww-{subset}-{r['file'][:-5]}", f"fa_sf/whowhen_{subset}", cap=caps[r["uid"]],
                             with_gt=False, rng=random.Random(f"fa1|{seed}|{r['uid']}|items"))
        s["fa"]["mistake_agent"] = r["mistake_agent"]
        s["fa"]["ww_file"] = r["file"]
        bad = fp.audit_item(s)
        if bad:
            fails.append((s["item_id"], bad[:3]))
        out.append(s)
    return out


def compare_ww_alg_step(items: list[dict], path: Path) -> dict:
    """data/eval/ww_alg_step.jsonl (scripts/build_whowhen_eval.py: fixed cap 3000 per step, A-Z) against the items
    built here: count, gold, option texts, prompts."""
    if not path.exists():
        return {"present": False}
    ref = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    by = {r["item_id"].split("-")[2]: r for r in ref}               # ww-alg-<n>-step
    same_gold = same_opts = same_prompt = 0
    for it in items:
        r = by.get(it["fa"]["ww_file"][:-5])
        if r is None:
            continue
        same_gold += r["gold_label"] == it["gold_label"] and r["labels"] == it["labels"]
        same_opts += r["prompt"].rsplit("\nOptions:\n", 1)[1] == it["prompt"].rsplit("\nOptions:\n", 1)[1]
        same_prompt += r["prompt"] == it["prompt"]
    return {"present": True, "n_ref": len(ref), "n_here": len(items), "same_gold_and_labels": same_gold,
            "same_options": same_opts, "same_prompt": same_prompt,
            "note": "ww_alg_step cuts every step at a fixed 3000 characters; the fa1d policy fits a per-trajectory cap "
                    "to the token budget, so prompts differ where a trajectory is cut harder"}


# ------------------------------------------------------------------------------------------------- tokens
def token_report(items: list[dict], tok, max_len: int, pack: int = 8) -> dict:
    """Tokens per epoch: A2 (final prompts + "\\n\\nAnswer:"), GFP-S (finals + one statements view per node) and the
    packed GFPSP estimate (finals + per pack the statement block once plus each node's question chunk, packs filled
    greedily up to `pack` nodes within max_len, as student.pack_views)."""
    finals, blocks, chunks = [], [], []
    for it in items:
        finals.append(it["prompt"] + "\n\nAnswer:")
        nodes = [sq for sq in it.get("subqs", []) if "id" in sq]
        if nodes:
            blocks.append(statement_block(it))
            ch = []
            for sq in nodes:
                opts = ("\n".join(f"{L}) {t}" for L, t in zip(MC_SYMBOLS, sq["options"])) if "options" in sq
                        else "A) Yes\nB) No")
                ch.append("\n\nIntermediate question: " + sq["question"] + "\nOptions:\n" + opts + "\nAnswer:")
            chunks.append(ch)
        else:
            blocks.append(None)
            chunks.append([])
    lf = tok.count(finals)
    lb = tok.count([b for b in blocks if b is not None])
    flat = [c for ch in chunks for c in ch]
    lc = iter(tok.count(flat) if flat else [])
    gfps = packed = 0
    bi = iter(lb)
    for b, ch in zip(blocks, chunks):
        if b is None:
            continue
        nb = next(bi)
        cs = [next(lc) for _ in ch]
        gfps += sum(nb + c for c in cs)
        size, k = 0, 0
        for c in cs:
            if k and k < pack and size + c <= max_len:
                size += c
                k += 1
                continue
            packed += size
            size, k = nb + c, 1
        packed += size
    n_nodes = sum(len(ch) for ch in chunks)
    return {"n": len(items), "final_tokens": bfp.pct(lf), "over_budget": sum(x > max_len for x in lf),
            "nodes_per_item": round(n_nodes / max(1, len(items)), 2), "tokens_per_epoch_a2": int(sum(lf)),
            "tokens_per_epoch_gfps": int(sum(lf) + gfps), "tokens_per_epoch_gfpsp_packed": int(sum(lf) + packed)}


# ------------------------------------------------------------------------------------------------- trees (_t2)
LEAK = re.compile(r"\b(is_mistake|mistake_step|mistake_agent|mistake_reason|ground[ _-]?truth)\b", re.I)
EXCERPT = re.compile(ft.OPEN_Q + "[^" + ft.CLOSE_Q + "]*" + ft.CLOSE_Q)    # a quoted log excerpt (fa_trees.free_render)


def convert_tree(line: dict, rec: dict, item: dict) -> tuple[list[dict], list[str], dict]:
    """One tree line -> the item's MC nodes, the reasons it is refused (empty: accepted) and per-node facts.
    Checks: the tree's own checks passed; 3-12 nodes with unique ids, known types, parents among the ids and a DAG;
    every node a valid MC node (treeplan.mc_check: 2-62 distinct one-line options, an answer index); a pointer option
    (one that starts with "Step <n>") is exactly "Step k [Agent]" of this trajectory (fa_public.step_ref, the item's
    option text), option_steps (when given) agree; a pointer node's options are all pointers; no question or option
    names a label field and no question holds "Answer:" (an index-free node: outside its quoted log excerpts, which
    are verbatim log text the prompt already shows, e.g. the step reference the Lead step “Answer: 3”); no node
    repeats the final question; cite_step within the trajectory."""
    bad = []
    n = len(rec["steps"])
    refs = [fp.step_ref(rec, k) for k in range(n)]
    nodes_in = line.get("nodes")
    if not isinstance(nodes_in, list) or not 1 <= len(nodes_in) <= 12:
        return [], ["node_count"], {}
    ids = [x.get("id") if isinstance(x, dict) else None for x in nodes_in]
    if not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
        return [], ["ids"], {}
    out, facts = [], {"pointer_nodes": 0, "quote_visible": 0, "quotes": 0}
    final_q = fp.norm_task(fp.Q_STEP.removeprefix("Question: "))
    for x in nodes_in:
        nid = x["id"]
        typ = x.get("type")
        if typ not in TREE_TYPES:
            bad.append(f"type:{nid}")
        q = " ".join(str(x.get("question") or "").split())
        opts = [" ".join(str(o).split()) for o in (x.get("options") or [])]
        node = {"id": nid, "type": typ, "question": q, "options": opts, "answer": x.get("answer"),
                "depends_on": list(x.get("parents") or []), "cite_step": x.get("cite_step"),
                "quote": str(x.get("quote") or ""), "status": "teacher"}
        if x.get("render") == "free":       # v3 index-free node (fa_trees.free_render): its steps by excerpt only
            node["render"] = "free"
            if x.get("option_steps") is not None:
                node["option_steps"] = list(x["option_steps"])
            if not ft.node_text_index_free(node):
                bad.append(f"free_index:{nid}")
        try:
            mc_check(node)
        except ValueError:
            bad.append(f"mc:{nid}")
            continue
        if any(p not in ids or p == nid for p in node["depends_on"]):
            bad.append(f"parents:{nid}")
        ptr = [POINTER.match(o) for o in opts]
        looks = [bool(STEP_WORD.match(o)) for o in opts]
        if any(looks):
            facts["pointer_nodes"] += 1
            if not all(looks):
                bad.append(f"pointer_mixed:{nid}")
            for o, m in zip(opts, ptr):
                if not m or int(m.group(1)) >= n or o != refs[int(m.group(1))]:
                    bad.append(f"pointer:{nid}")
                    break
            st = x.get("option_steps")
            if st is not None and [int(m.group(1)) if m else None for m in ptr] != list(st):
                bad.append(f"option_steps:{nid}")
            node["option_steps"] = [int(m.group(1)) if m else None for m in ptr]
        cs = node["cite_step"]
        if cs is not None and (not isinstance(cs, int) or isinstance(cs, bool) or not 0 <= cs < n):
            bad.append(f"cite_step:{nid}")
        q_own = EXCERPT.sub(" ", q) if node.get("render") == "free" else q       # the teacher's own wording
        if LEAK.search(q) or any(LEAK.search(o) for o in opts) or "Answer:" in q_own:
            bad.append(f"leak:{nid}")
        if fp.norm_task(q) == final_q:
            bad.append(f"final_question:{nid}")
        if node["quote"] and isinstance(cs, int) and 0 <= cs < n:
            facts["quotes"] += 1
            facts["quote_visible"] += node["quote"] in fp.cut(rec["steps"][cs]["text"], item["fa"]["step_cap"])
        out.append(node)
    if not bad:
        depth = node_depths(out)
        seen, stack = set(), set()

        def cyclic(i):
            if i in stack:
                return True
            if i in seen:
                return False
            seen.add(i)
            stack.add(i)
            r = any(cyclic(p) for p in next(nd for nd in out if nd["id"] == i)["depends_on"])
            stack.discard(i)
            return r
        if any(cyclic(nd["id"]) for nd in out):
            bad.append("cycle")
        for nd in out:
            nd["node_depth"] = depth[nd["id"]]
    return (out if not bad else []), bad, facts


def traj_of(rec: dict) -> dict:
    """A fa_public record in the shape fa_trees.free_render / merge_v3 read (the steps as the student sees them)."""
    return {"traj_id": rec["uid"], "subset": rec.get("subset", ""), "mistake_step": rec["dec"],
            "ground_truth": str(rec.get("answer") or ""),
            "steps": [{"k": i, "agent": s["agent"], "text": s["text"]} for i, s in enumerate(rec["steps"])]}


def tree_items(items: list[dict], recs: dict, trees: dict, keep_failed: bool, stats: dict,
               addons: dict | None = None, render: str = "index") -> list[dict]:
    """The items with their label-derived nodes replaced by the tree's MC nodes (no tree, or a refused one: no nodes,
    final question only). stats collects coverage, refusals, node types, option counts, answer positions.
    v3 (addons given or render "free"): each tree is first merged with its add-on line and / or rendered index-free
    at the item's step cap (fa_trees.merge_v3); stats["v3"] counts merges, clean nodes and dropped nodes."""
    out = []
    v3 = addons is not None or render != "index"
    for it in items:
        uid = it["fa"]["uid"]
        line = trees.get(uid)
        nodes, why = [], ["no_tree"]
        if line is not None:
            if not (line.get("checks") or {}).get("pass") and not keep_failed:
                why = ["tree_checks_failed"]
            else:
                nodes, why, facts = convert_tree(line, recs[uid], it)
                if v3 and not why:          # the v2 tree is accepted as it is; then merged / rendered and re-checked
                    line = ft.merge_v3(traj_of(recs[uid]), line, (addons or {}).get(uid), render=render,
                                       cap=it["fa"]["step_cap"], stats=stats["v3"])
                    nodes, why, facts = convert_tree(line, recs[uid], it)
                if not why:
                    stats["facts"].update(facts)
        if why:
            stats["refused"][why[0].split(":")[0]] += 1
        else:
            stats["items_with_tree"] += 1
            g = it["fa"]["decisive_step"]
            for nd in nodes:
                stats["types"][nd["type"]] += 1
                stats["n_options"][len(nd["options"])] += 1
                stats["answer_pos"][nd["answer"]] += 1
                stats["depth"][nd["node_depth"]] += 1
                if nd["type"] == "answer_step" and nd.get("option_steps"):
                    stats["answer_step_is_decisive"][nd["option_steps"][nd["answer"]] == g] += 1
        new = dict(it, subqs=nodes, has_tree=bool(nodes), tree_source="fa_sf_trees",
                   fa=dict(it["fa"], tree_teacher=(line or {}).get("teacher") if nodes else None))
        out.append(new)
    stats["items"] += len(items)
    return out


def read_trees(path: Path, subset: str) -> dict:
    """uid -> tree line (traj_id "<subset>/<file stem>"); a later line for the same trajectory replaces an earlier."""
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            sub, _, stem = str(x.get("traj_id", "")).partition("/")
            if sub != subset or not stem:
                raise SystemExit(f"{path}: traj_id {x.get('traj_id')!r} is not {subset}/<file stem>")
            out[f"{UID_PREFIX[subset]}-{stem}"] = x
    return out


def v3_setup(args, tstats: dict, subset: str | None = None) -> tuple[dict | None, str]:
    """The add-on lines (--trees-addon, {subset} filled in) and the output suffix (--trees-suffix, default _t2 for a
    v2 build, _t3 with --trees-addon or --pointer-render free); adds the v3 counters to tstats in v3 mode only."""
    addons = None
    v3 = bool(args.trees_addon) or args.pointer_render != "index"
    if args.trees_addon:
        apath = ROOT / args.trees_addon.format(subset=subset)
        addons = read_trees(apath, subset) if subset in SUBSETS else read_lines_by_tail(apath, subset)
        tstats["addon_file"] = str(apath.relative_to(ROOT)) if apath.is_relative_to(ROOT) else str(apath)
        tstats["addon_lines"] = len(addons)
    if v3:
        tstats["v3"] = collections.Counter()
        tstats["pointer_render"] = args.pointer_render
    return addons, args.trees_suffix or ("_t3" if v3 else "_t2")


def read_lines_by_tail(path: Path, subset: str) -> dict:
    """uid -> line for the aftraj / real files (traj_id "<subset>/<uid>"); a later line replaces an earlier."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                x = json.loads(line)
                sub, _, uid = str(x.get("traj_id", "")).partition("/")
                if sub != subset or not uid:
                    raise SystemExit(f"{path}: traj_id {x.get('traj_id')!r} is not {subset}/<uid>")
                out[uid] = x
    return out


def v3_summary(d: Path, args, trees: dict, files: dict) -> None:
    """summary.json of a v3 output directory: the build flags, the tree statistics and the md5s of its files."""
    mine = {p: n for p, n in files.items() if Path(p).parent == (d.relative_to(ROOT) if d.is_relative_to(ROOT) else d)}
    (d / "summary.json").write_text(json.dumps({"args": vars(args), "trees": trees, "md5": md5_all(mine)}, indent=1,
                                               ensure_ascii=False, default=str))


def add_v3_args(ap: argparse.ArgumentParser, pattern: bool = True) -> None:
    hint = " (a path with {subset})" if pattern else ""
    ap.add_argument("--trees-addon", default="", help="v3 add-on lines (fa_trees.addon_record) merged with the trees "
                    f"by traj_id{hint}")
    ap.add_argument("--pointer-render", choices=("index", "free"), default="index",
                    help="free: v3 index-free render of every tree node (fa_trees.free_render)")
    ap.add_argument("--trees-suffix", default="", help="output suffix of the tree variant (default _t2, or _t3 with "
                    "--trees-addon / --pointer-render free)")


# ------------------------------------------------------------------------------------------------- main
def write(path: Path, items: list[dict], files: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    files[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = len(items)


def md5_all(files: dict) -> dict:
    return {p: bfp.md5(ROOT / p if not Path(p).is_absolute() else Path(p)) for p in files}


def build_subset(subset: str, args, tok, ww: dict) -> dict:
    raw = ROOT / args.raw
    counts: collections.Counter = collections.Counter()
    sf = read_sf(raw, subset, counts)
    counts["sf/trajectories"] = len(sf)
    counts["sf/agent_mismatch_kept"] = sum("agent_mismatch" in r["flags"] for r in sf)
    counts["sf/gold_agent_not_an_agent_kept"] = sum(r.get("gold_agent") not in fp.agents_in_order(r) for r in sf)
    long_sf = [r for r in sf if len(r["steps"]) > fp.MAX_STEPS or len(r["steps"]) < 2]
    if long_sf:
        raise SystemExit(f"{subset}: {len(long_sf)} StepFinder trajectories do not fit 2-62 steps")
    norm = ROOT / args.compare_norm if args.compare_norm else None
    if norm is not None and norm.exists():    # the fa1 builds' normalized records: the same reader, the same records
        with open(norm) as f:
            old = {r["uid"]: r for r in map(json.loads, f) if r["uid"].startswith(UID_PREFIX[subset] + "-")}
        counts["sf/equal_to_fa1_norm"] = sum(old.get(r["uid"]) == r for r in sf)
    ov = overlap(sf, ww, args.boilerplate_df, args.overlap)
    vg = val_groups(sf, ov, args.val_frac, args.seed, subset)
    val = [r for r in sf if fp.norm_task(r["task"]) in vg]
    train = [r for r in sf if fp.norm_task(r["task"]) not in vg]
    assert not {fp.norm_task(r["task"]) for r in val} & {fp.norm_task(r["task"]) for r in train}
    assert not any(any_overlap(ov, r["uid"]) for r in val)
    test_all = ww[subset]
    test = [r for r in test_all if len(r["steps"]) <= fp.MAX_STEPS]
    excluded = [{"file": r["file"], "n_steps": len(r["steps"]), "decisive_step": r["dec"]} for r in test_all
                if len(r["steps"]) > fp.MAX_STEPS]
    print(f"{subset}: {len(sf)} trajectories ({len({fp.norm_task(r['task']) for r in sf})} questions); val "
          f"{len(val)} ({len(vg)} questions), train {len(train)}, disjoint train "
          f"{sum(not any_overlap(ov, r['uid']) for r in train)}; test {len(test)} of {len(test_all)}", flush=True)
    if args.dry:
        return {"overlap": overlap_summary(sf, ov, subset), "counts": dict(counts)}
    caps, cap_stats = {}, {}
    for name, rs in (("train", train), ("val", val), ("test", test)):
        st: dict = {}
        caps.update(bfp.fit_caps(rs, tok, args, st))
        cap_stats[name] = st | {"cut": cut_stats(rs, caps)}
    fails: list = []
    files: dict = {}
    tr_s, tr_a = build_items(train, caps, subset, "train", args, fails)
    va_s, va_a = build_items(val, caps, subset, "val", args, fails)
    te = build_test(test, caps, subset, args.seed, fails)
    out = ROOT / f"{args.out_prefix}_{subset}"
    dis = ROOT / f"{args.out_prefix}_{subset}_disj"
    write(out / "train.jsonl", tr_s, files)
    write(out / "val.jsonl", va_s, files)
    write(out / "train_agent.jsonl", tr_a, files)
    write(out / "val_agent.jsonl", va_a, files)
    keep = lambda its: [it for it in its if not any_overlap(ov, it["fa"]["uid"])]        # noqa: E731
    write(dis / "train.jsonl", keep(tr_s), files)
    write(dis / "val.jsonl", va_s, files)
    write(dis / "train_agent.jsonl", keep(tr_a), files)
    write(dis / "val_agent.jsonl", va_a, files)
    ev = ROOT / args.eval_dir
    write(ev / f"fa_sf_ww_{subset}.jsonl", te, files)
    by_uid = {r["uid"]: r for r in sf}
    trees = None
    if args.trees:
        tpath = ROOT / args.trees.format(subset=subset)
        if tpath.exists():
            tl = read_trees(tpath, subset)
            tstats = {"file": str(tpath.relative_to(ROOT)) if tpath.is_relative_to(ROOT) else str(tpath),
                      "lines": len(tl), "items": 0, "items_with_tree": 0, "refused": collections.Counter(),
                      "types": collections.Counter(), "n_options": collections.Counter(),
                      "answer_pos": collections.Counter(), "depth": collections.Counter(),
                      "answer_step_is_decisive": collections.Counter(), "facts": collections.Counter()}
            addons, suffix = v3_setup(args, tstats, subset)
            t_tr = tree_items(tr_s, by_uid, tl, args.trees_keep_failed, tstats, addons, args.pointer_render)
            t_va = tree_items(va_s, by_uid, tl, args.trees_keep_failed, tstats, addons, args.pointer_render)
            for it in t_tr + t_va:
                statement_block(it)                       # GFP-S view of every item builds
            write(ROOT / f"{args.out_prefix}_{subset}{suffix}" / "train.jsonl", t_tr, files)
            write(ROOT / f"{args.out_prefix}_{subset}{suffix}" / "val.jsonl", t_va, files)
            write(ROOT / f"{args.out_prefix}_{subset}_disj{suffix}" / "train.jsonl", keep(t_tr), files)
            write(ROOT / f"{args.out_prefix}_{subset}_disj{suffix}" / "val.jsonl", t_va, files)
            trees = {k: (dict(sorted(v.items(), key=lambda kv: str(kv[0])))
                         if isinstance(v, collections.Counter) else v) for k, v in tstats.items()}
            trees["tokens_train"] = token_report(t_tr, tok, args.budget)
            if "v3" in tstats:
                for d in (f"{args.out_prefix}_{subset}{suffix}", f"{args.out_prefix}_{subset}_disj{suffix}"):
                    v3_summary(ROOT / d, args, trees, files)
        else:
            trees = {"file": str(tpath), "missing": True}
            print(f"{subset}: no trees file {tpath}", flush=True)
    baseline_sets = {"train": train, "val": val, "test_fit": test, "test_all": test_all}
    summary = {
        "args": vars(args), "subset": subset, "who_and_when_dir": f"data/raw/who_and_when/Who&When/{SUBSETS[subset]}",
        "stepfinder_dir": f"data/raw/stepfinder/repo/data/{SUBSETS[subset]}/train", "counts": dict(sorted(counts.items())),
        "overlap": overlap_summary(sf, ov, subset),
        "overlap_note": "exact = equal normalized question text; ngram = fa_public.OverlapIndex (8-gram containment >= "
                        f"{args.overlap}, boilerplate df {args.boilerplate_df}); id = shared GAIA / AssistantBench "
                        "question_ID; same = this subset's Who&When test set, other = the other subset's",
        "val": {"questions": len(vg), "trajectories": len(val), "drawn_from": "questions without any Who&When overlap"},
        "train": {"main": len(train), "disjoint": sum(not any_overlap(ov, r["uid"]) for r in train)},
        "test": {"file": f"{args.eval_dir}/fa_sf_ww_{subset}.jsonl", "n": len(test), "of": len(test_all),
                 "excluded_over_62_steps": excluded,
                 "agent_label_mismatch": sum("agent_mismatch" in r["flags"] for r in test)},
        "caps": cap_stats,
        "stats": {k: bfp.describe(v) for k, v in baseline_sets.items()},
        "baselines": {k: bfp.baselines(v, train) for k, v in baseline_sets.items()},
        "tokens": {"train": token_report(tr_s, tok, args.budget), "train_disj": token_report(keep(tr_s), tok,
                                                                                               args.budget),
                   "val": token_report(va_s, tok, args.budget), "test": token_report(te, tok, args.budget),
                   "method": tok.method},
        "published_stepfinder": PUBLISHED[subset],
        "audit": {"failures": len(fails), "examples": fails[:10]},
        "files": files}
    if subset == "alg":
        summary["test"]["vs_ww_alg_step"] = compare_ww_alg_step(te, ev / "ww_alg_step.jsonl")
    if trees is not None:
        summary["trees"] = trees
    summary["md5"] = md5_all(files)
    for d in (out, dis):
        (d / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    if fails:
        raise SystemExit(f"AUDIT FAILED {subset}: {fails[:5]}")
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--subsets", default="alg,hc")
    ap.add_argument("--out-prefix", default="data/student_fa_sf")
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--val-frac", type=float, default=0.10, help="val share of a subset's trajectories (whole "
                    "question groups)")
    ap.add_argument("--gt-p", type=float, default=0.5, help="train / val items with the reference-answer line (fa1d)")
    ap.add_argument("--budget", type=int, default=6144, help="Qwen3.5 tokens of a step prompt + '\\n\\nAnswer:' (fa1d)")
    ap.add_argument("--cap-max", type=int, default=3000)
    ap.add_argument("--cap-min", type=int, default=80)
    ap.add_argument("--overlap", type=float, default=0.5, help="8-gram overlap ratio (fa1 builds)")
    ap.add_argument("--boilerplate-df", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default=bfp.TOKENIZER)
    ap.add_argument("--trees", default="", help="MC trees per subset, a path with {subset}, e.g. "
                    "data/raw/stepfinder_trees/{subset}.jsonl (writes the _t2 variants)")
    ap.add_argument("--trees-keep-failed", action="store_true", help="also convert trees whose own checks failed")
    add_v3_args(ap)
    ap.add_argument("--compare-norm", default="data/student_fa1/norm/stepfinder.jsonl",
                    help="normalized StepFinder records of the fa1 builds to compare with ('' = off)")
    ap.add_argument("--dry", action="store_true", help="splits and overlap only")
    args = ap.parse_args(argv)
    subsets = [s for s in args.subsets.split(",") if s]
    if not set(subsets) <= set(SUBSETS):
        raise SystemExit(f"--subsets {args.subsets}: alg, hc")
    tok = bfp.Tok(args.tokenizer)
    ww = {s: read_ww(ROOT / args.raw, s) for s in SUBSETS}
    res = {}
    for s in subsets:
        res[s] = build_subset(s, args, tok, ww)
        print(json.dumps({k: res[s].get(k) for k in ("overlap", "val", "train", "test", "audit", "md5")}, indent=1,
                         default=str), flush=True)
    print(f"rss {bfp.rss_mb()} MB")


if __name__ == "__main__":
    main()
