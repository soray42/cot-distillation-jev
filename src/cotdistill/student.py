"""One-pass decision student: LM-head letter readout, KL to soft targets, sub-question supervision.

Every question (the final decision and each sub-question) is its own sequence that repeats the
problem text; hybrid linear-attention models (Qwen3.5) ignore attention masks, so we do not pack
questions with a block-causal mask. Sequences are right-padded; the readout is the next-token
distribution at the last real token, restricted to the option-letter tokens. The opt-in --tree-pack (Pack) reads
several node questions of one problem in one plain causal pass instead: later questions see the earlier questions,
never an answer.
"""
from __future__ import annotations

import dataclasses
import math
import random
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from .metrics import calibration  # noqa: F401  (re-exported for scripts)
from .treeplan import (CTX_VIEWS, LOCAL_HEADER, MC_SYMBOLS, _descendants, _fact_line, _node_p,  # noqa: F401
                       ancestors, answer_component, depth_split, full_tree_plan, is_local_leaf, is_mc_node,
                       local_context, mc_check, mc_labels, mc_options_block, mix_statements, node_depths,
                       problem_no_options, scaffold_fade, scaffold_rng, statement_block, subq_keep, view_leak)

SUBQ_TEMPLATE = "{problem}\n\nIntermediate question: {question}\nOptions:\nA) {a}\nB) {b}\nAnswer:"
# a multiple-choice tree node (treeplan.is_mc_node): its own options, the layout of SUBQ_TEMPLATE
SUBQ_OPTS_TEMPLATE = "{problem}\n\nIntermediate question: {question}\nOptions:\n{opts}\nAnswer:"
FINAL_TEMPLATE = "{problem}\n\nAnswer:"
RATIONALE_PREFIX = "{problem}\n\nReasoning:"          # rationale-LM baseline: CoT text, then the answer
# --tree-view bare (value nodes): the node question with its own options and nothing before it
SUBQ_BARE_TEMPLATE = SUBQ_OPTS_TEMPLATE.replace("{problem}\n\n", "", 1)
# value nodes (scripts/build_gsm8k_ops.py): one node per atomic operation, in an item field of its own, asked as a
# multiple-choice node in one of its renderings (--node-format value / yesno+value, --node-render)
VALUE_FIELD = "value_nodes"
VALUE_RENDERINGS = ("op", "qop", "opsel", "vt", "vtr")
VALUE_FORMATS = ("value", "yesno+value")
# verification nodes (scripts/build_gsm8k_vt.py; arms GFPQVT / GFPQVTR): one node per option of the final question
# ("In this problem, what does option C (8) stand for?", four descriptions), in an item field of their own, asked
# through the value-node path in the problem view (the prompt WITH its options); rendering vt = the real answers, vtr =
# the same stems and choices with each problem's descriptions permuted among its nodes (wrong-content control)
VERIFY_FIELD = "verify_nodes"
VERIFY_RENDERINGS = ("vt", "vtr")


def value_field(render: str) -> str:
    """The item field that holds the nodes of a value-node rendering: VERIFY_FIELD for vt / vtr, else VALUE_FIELD."""
    return VERIFY_FIELD if render in VERIFY_RENDERINGS else VALUE_FIELD


@dataclass
class Example:
    text: str
    labels: list[str]            # option letters, e.g. ["A", "B", "C"]
    target: list[float]          # soft target over labels (sums to 1)
    weight: float
    kind: str                    # "final" | "subq"
    item_id: str
    gold: int | None = None      # index of the gold label (final questions, if known)
    continuation: str | None = None   # kind "lm": text scored token by token after `text`


def letter_token_ids(tokenizer, letters: list[str]) -> list[int]:
    """Token ids of ' A', ' B', ... (the token that follows 'Answer:'; under --chat-template the bare 'A', 'B', ...
    that opens the assistant turn); each must be a single token. A label whose spaced form is not one token but whose
    bare form is (Qwen splits ' 0' into ' ' + '0': the digit labels 0-9 of option lists beyond A-Z, a-z) is read as the
    bare token."""
    ids = []
    sp = "" if _chat(tokenizer) else " "      # --chat-template: the assistant turn opens with the bare letter
    for L in letters:
        t = tokenizer.encode(sp + L, add_special_tokens=False)
        if len(t) != 1 and sp:
            bare = tokenizer.encode(L, add_special_tokens=False)
            t = bare if len(bare) == 1 else t
        if len(t) != 1:
            raise ValueError(f"label '{sp}{L}' is not a single token: {t}")
        ids.append(t[0])
    return ids


# ---- --chat-template (opt-in; nothing below runs on the default path) ---------------------------------------------
CHAT_INSTRUCTION = "\n\nAnswer with the letter of the correct option only."     # as scripts/eval_llm_baseline.py
_NO_THINK_CLOSE = "\n\n</think>\n\n"
_CHAT: dict[int, tuple] = {}


def chat_render(proc, typed: bool = False, bos: str | None = None):
    """The --chat-template renderer of an Example text: the text without its closing "Answer:" cue, plus
    CHAT_INSTRUCTION, as one user turn of the model's chat template with thinking off, ending where the assistant's
    answer starts. For a final question this is exactly scripts/eval_llm_baseline.py's rendering of the prompt. A
    template that ignores enable_thinking but opens a think block gets it closed empty; a leading BOS string is cut
    (special_prefix puts the BOS token back). typed: content as typed parts (multimodal processors)."""
    def render(text: str) -> str:
        body = text
        for cue in ("\n\nAnswer:", "\nAnswer:"):
            if text.endswith(cue):
                body = text[:-len(cue)]
                break
        content = body + CHAT_INSTRUCTION
        out = proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": content}] if typed
                                         else content}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        if out.rstrip().endswith("<think>"):
            out = out.rstrip() + _NO_THINK_CLOSE
        return out[len(bos):] if bos and out.startswith(bos) else out
    return render


def set_chat_render(tokenizer, render) -> None:
    """--chat-template: every Example text this tokenizer encodes (text_ids: collate, micro-batch lengths, FSDP split)
    is rendered by render first, and option letters are read as bare tokens ("A") instead of " A". Keyed by the
    tokenizer object (kept, so its id cannot be reused), like special_prefix."""
    _CHAT[id(tokenizer)] = (tokenizer, render)


def _chat(tokenizer):
    hit = _CHAT.get(id(tokenizer))
    return hit[1] if hit is not None and hit[0] is tokenizer else None


def use_chat_template(tokenizer, model_path: str) -> str:
    """Set the chat renderer for --chat-template, the template loaded as scripts/eval_llm_baseline.py loads it (the
    model's processor, else the tokenizer). Returns a rendered example for the log."""
    import transformers
    try:
        proc = transformers.AutoProcessor.from_pretrained(model_path)
    except Exception:
        proc = tokenizer
    typed = isinstance(proc, transformers.ProcessorMixin)
    set_chat_render(tokenizer, chat_render(proc, typed, tokenizer.bos_token if special_prefix(tokenizer) else None))
    letter_token_ids(tokenizer, ["A", "B", "C", "D", "E"])          # bare letters must be single tokens
    return _chat(tokenizer)(FINAL_TEMPLATE.format(problem="Question: Is it?\nOptions:\nA) Yes\nB) No"))


def text_ids(tokenizer, text: str) -> list[int]:
    """Token ids of an Example text without special tokens (rendered as a chat turn under --chat-template)."""
    r = _chat(tokenizer)
    return tokenizer.encode(r(text) if r else text, add_special_tokens=False)


_PREFIX: dict[int, tuple] = {}


def special_prefix(tokenizer) -> list[int]:
    """Special tokens the tokenizer itself puts before every text: [<bos>] for Gemma, [] for Qwen.

    Our encodes use add_special_tokens=False so that left truncation keeps the question end; callers re-attach this
    prefix after truncation (Gemma degrades badly without its BOS). The tokenizer object is kept in the cache, so
    its id cannot be reused by another tokenizer."""
    hit = _PREFIX.get(id(tokenizer))
    if hit is None or hit[0] is not tokenizer:
        probe = "Answer:"
        with_sp = list(tokenizer.encode(probe, add_special_tokens=True))
        without = list(tokenizer.encode(probe, add_special_tokens=False))
        pre = with_sp[:len(with_sp) - len(without)] if with_sp[len(with_sp) - len(without):] == without else []
        hit = _PREFIX[id(tokenizer)] = (tokenizer, pre)
    return hit[1]


def encode_tail(tokenizer, text: str, max_len: int) -> list[int]:
    """Token ids of text, cut to its last tokens so that the special prefix plus the text fit max_len."""
    pre = special_prefix(tokenizer)
    ids = text_ids(tokenizer, text)
    return pre + ids[-max(1, max_len - len(pre)):]


def logit_softcap(model) -> float | None:
    """final_logit_softcapping of the (text) config: 30.0 for Gemma 4, None for Qwen. The HF forward applies
    s * tanh(z / s) to every logit, so a readout from h @ W.T must do the same to match the model's distribution."""
    cfg = model.config
    cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    return getattr(cfg, "final_logit_softcapping", None) or None


def head_logits(h: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None,
                softcap: float | None = None) -> torch.Tensor:
    """h @ w.T (+ bias) for rows w of the output embedding. Without a soft-cap: exactly the previous readout (in w's
    dtype, so Qwen numbers are unchanged). With one: in float32 outside autocast (bf16 spacing is 0.125-0.25 at
    |z| = 16-64, coarse for calibrated letter probabilities), then s * tanh(z / s)."""
    if not softcap:
        z = h.to(w.dtype) @ w.T
        return z + bias if bias is not None else z
    with torch.autocast(device_type=h.device.type, enabled=False):
        z = h.float() @ w.float().T
        if bias is not None:
            z = z + bias.float()
        return softcap * torch.tanh(z / softcap)


def _soft(p_yes: float, yes_first: bool, eps: float = 1e-4) -> list[float]:
    p = min(max(p_yes, eps), 1 - eps)
    return [p, 1 - p] if yes_first else [1 - p, p]


def mc_node_example(problem: str, sq: dict, weight: float, item_id) -> Example:
    """A multiple-choice tree node (treeplan.is_mc_node) asked like a final question: the context, the node question
    with its own options under the letters A, B, ... (SUBQ_OPTS_TEMPLATE), readout at "Answer:" over those letters, a
    one-hot target on its answer index (label smoothing, if any, is applied by kl_loss as for the final question).
    Draws nothing from any generator (the option order is the data's)."""
    labels = mc_labels(sq)
    return Example(SUBQ_OPTS_TEMPLATE.format(problem=problem, question=sq["question"], opts=mc_options_block(sq)),
                   labels, [float(i == sq["answer"]) for i in range(len(labels))], weight, "subq", item_id)


def value_mc_node(sq: dict, render: str) -> dict | None:
    """A value node (item field VALUE_FIELD; vt / vtr: a verification node of VERIFY_FIELD) as a multiple-choice node
    in one rendering: {id, question, options, answer, type value_<render>, node_depth, depends_on}; None when the node
    has no such rendering (qop / opsel exist only where the Socratic sub-question is true as worded; opsel only for a
    step's last operation)."""
    if render not in VALUE_RENDERINGS:
        raise ValueError(f"value node rendering {render!r}: one of {', '.join(VALUE_RENDERINGS)}")
    r = (sq.get("renderings") or {}).get(render)
    if r is None:
        return None
    mc = {"id": sq["id"], "question": r["question"], "options": r["options"], "answer": r["answer"],
          "type": f"value_{render}", "node_depth": sq.get("node_depth"), "depends_on": sq.get("depends_on") or []}
    mc_check(mc)
    return mc


def value_node_example(context: str | None, mc: dict, weight: float, item_id) -> Example:
    """A value node (value_mc_node) asked like a multiple-choice tree node (mc_node_example: SUBQ_OPTS_TEMPLATE,
    labels A, B, C, one-hot target on its answer); context None (--tree-view bare) asks the question alone
    (SUBQ_BARE_TEMPLATE: "Intermediate question: ...\\nOptions:\\nA) ...\\nAnswer:"). Draws nothing from any generator."""
    if context is not None:
        return mc_node_example(context, mc, weight, item_id)
    labels = mc_labels(mc)
    return Example(SUBQ_BARE_TEMPLATE.format(question=mc["question"], opts=mc_options_block(mc)), labels,
                   [float(i == mc["answer"]) for i in range(len(labels))], weight, "subq", item_id)


def _value_views(item: dict, *, view: str, rng: random.Random, cap: int, aux_total: float, node_render: str,
                 node_weight: float, stats: dict | None) -> list[Example]:
    """The value-node views of one problem (--node-format value / yesno+value): every node of item[value_field(
    node_render)] (VALUE_FIELD; VERIFY_FIELD for vt / vtr) that has the rendering node_render (at most cap, drawn from
    rng only when the problem has more), asked in the view's context (problem: the prompt with options; statements:
    statement_block; statements+question: problem_no_options; bare: nothing) with weight node_weight, else aux_total /
    number of views (the per-problem convention), times the item weight. vtr also counts vtr_agree in stats (asked
    nodes whose control answer is the real one)."""
    field = item.get(value_field(node_render)) or []
    nodes = [mc for mc in (value_mc_node(sq, node_render) for sq in field) if mc is not None]
    if stats is not None:
        stats["value_nodes_without_render"] = (stats.get("value_nodes_without_render", 0)
                                              + len(field) - len(nodes))
    if cap and len(nodes) > cap:
        nodes = rng.sample(nodes, cap)
    if not nodes:
        return []
    context = (None if view == "bare" else statement_block(item) if view == "statements" else problem_no_options(item)
               if view == "statements+question" else item["prompt"])
    w = item.get("weight", 1.0) * (node_weight or aux_total / len(nodes))
    if stats is not None:
        stats["value_views"] = stats.get("value_views", 0) + len(nodes)
        stats["items_with_value_views"] = stats.get("items_with_value_views", 0) + 1
        if node_render == "vtr":
            real = {sq["id"]: sq["renderings"]["vt"]["answer"] for sq in field}
            stats["vtr_agree"] = stats.get("vtr_agree", 0) + sum(real.get(mc["id"]) == mc["answer"] for mc in nodes)
    return [value_node_example(context, mc, w, item["item_id"]) for mc in nodes]


_OPT_LINE = re.compile(r"^([A-Za-z0-9])\) (.*)$")     # option symbols A-Z, a-z, 0-9 (lists of up to 62 options)


def permute_options(prompt: str, labels: list[str], target: list[float], gold: int | None,
                    rng: random.Random) -> tuple[str, list[float], int | None] | None:
    """Shuffle the lettered options of a prompt that ends in an 'Options:' block; the target mass and the gold
    index follow the option texts. "None of the above" stays last and ordered scales (options starting with a
    digit) are left alone. Returns None when the prompt does not have exactly the expected option lines."""
    head, sep, tail = prompt.rpartition("\nOptions:\n")
    if not sep:
        return None
    ms = [_OPT_LINE.match(line) for line in tail.split("\n")]
    if len(ms) != len(labels) or not all(ms) or [m.group(1) for m in ms] != labels:
        return None
    texts = [m.group(2) for m in ms]
    if all(t[:1].isdigit() for t in texts):
        return None
    free = [i for i, t in enumerate(texts) if not t.lower().startswith("none of the above")]
    pinned = [i for i in range(len(texts)) if i not in free]
    perm = rng.sample(free, len(free)) + pinned          # new position i shows old option perm[i]
    body = "\n".join(f"{L}) {texts[j]}" for L, j in zip(labels, perm))
    return head + sep + body, [target[j] for j in perm], (perm.index(gold) if gold is not None else None)


def build_examples(item: dict, *, final: str, subq: str, subq_target: str, lambda_sub: float,
                   rng: random.Random, rationale_lm: bool = False, lambda_lm: float = 1.0,
                   permute_final: float = 0.0, subq_k: int = 0, subq_weight: str = "split",
                   subq_frac: float = 0.0, subq_filter: str = "all") -> list[Example]:
    """Turn one student-data record into training examples for a given arm.

    final: "none" | "teacher" (teacher answer distribution) | "gold" (one-hot gold)
    subq: "none" | "cot" (teacher-extracted sub-questions) | "random" (matched generic questions) |
          "mix" (as many sub-questions as "cot", half CoT nodes and half matched controls)
    subq_target: "fresh" (teacher answer without CoT) | "cot" (with CoT) | "truth" (program truth,
                 falling back to the CoT answer, then the fresh one, when no truth is available) |
                 "commit" (the teacher's value-commitment confidence where the CoT settled the node)
    rationale_lm: add a token-level LM example on the teacher's reasoning and answer (DHRD-style baseline)
    permute_final: probability of showing the final question with its options in a new random order (the
                   target follows the option texts), so the student cannot lean on letter positions
    subq_k: if > 0, use at most this many (randomly drawn) sub-questions per item and epoch
    subq_weight: "split" (each of the n sub-questions weighs lambda_sub / n) | "each" (each weighs lambda_sub)
    subq_frac: if > 0, use a random ceil(frac * n) of the item's sub-questions per epoch
    subq_filter: "all" | "path" | "mirror": ask only the tree nodes subq_keep admits (decision chains; the matched
                 controls of "random" and "mix" are not filtered)
    Every example's weight is multiplied by item["weight"] (default 1; e.g. per-category balancing weights).
    """
    wi = item.get("weight", 1.0)
    ex: list[Example] = []
    labels = item["labels"]
    gold = labels.index(item["gold_label"]) if item.get("gold_label") in labels else None
    target = None
    if final == "teacher" and item.get("teacher"):
        t = [item["teacher"].get(L, 0.0) for L in labels]
        s = sum(t)
        if s > 0:
            target = [x / s for x in t]
    elif final == "gold" and gold is not None:
        target = [1.0 if i == gold else 0.0 for i in range(len(labels))]
    if target is not None:
        prompt, fgold = item["prompt"], gold
        if permute_final > 0 and rng.random() < permute_final:
            shuffled = permute_options(prompt, labels, target, gold, rng)
            if shuffled:
                prompt, target, fgold = shuffled
        ex.append(Example(FINAL_TEMPLATE.format(problem=prompt), labels, target, wi, "final", item["item_id"], fgold))
    if subq_filter != "all":
        item = dict(item, subqs=[sq for sq in item.get("subqs", []) if subq_keep(sq, subq_filter)])
    if subq == "mix":            # same number of sub-questions as "cot": half CoT nodes, half matched controls
        cot = [dict(sq, _kind="cot") for sq in item.get("subqs", [])]
        ctl = [dict(sq, _kind="random") for sq in item.get("random_subqs", [])]
        n = len(cot)
        k = (n + 1) // 2
        pool = rng.sample(cot, k) + rng.sample(ctl, min(n - k, len(ctl)))
    else:
        pool = {"cot": item.get("subqs", []), "random": item.get("random_subqs", [])}.get(subq, [])
    if any(is_mc_node(sq) for sq in pool):              # opt-in MC nodes: the full-tree arms read them
        raise ValueError(f"{item.get('item_id')}: multiple-choice tree nodes are read by the full-tree arms "
                         "(--tree-full) and evaluate_subq, not by --subq")
    usable = []
    for sq in pool:
        p = None
        if sq.get("_kind") == "random":      # controls in a mix keep their own (fresh) target
            p = sq.get("p_fresh")
        elif subq_target == "truth" and sq.get("truth") is not None:
            p = 1.0 if sq["truth"] else 0.0
        elif subq_target == "commit" and sq.get("p_commit") is not None:
            p = sq["p_commit"]
        elif subq_target in ("cot", "truth", "commit"):
            p = sq.get("p_cot")
        if p is None:
            p = sq.get("p_fresh")
        if p is not None:
            usable.append((sq["question"], p))
    if subq_k > 0 and len(usable) > subq_k:
        usable = rng.sample(usable, subq_k)
    if subq_frac > 0 and usable:
        usable = rng.sample(usable, max(1, math.ceil(subq_frac * len(usable))))
    w = wi * (lambda_sub if subq_weight == "each" else lambda_sub / max(1, len(usable)))
    for q, p in usable:
        yes_first = rng.random() < 0.5
        a, b = ("Yes", "No") if yes_first else ("No", "Yes")
        ex.append(Example(SUBQ_TEMPLATE.format(problem=item["prompt"], question=q, a=a, b=b), ["A", "B"],
                          _soft(p, yes_first), w, "subq", item["item_id"]))
    if rationale_lm and item.get("rationale") and item.get("teacher"):
        answer = max(item["teacher"], key=item["teacher"].get)
        ex.append(Example(RATIONALE_PREFIX.format(problem=item["prompt"]), labels, [], lambda_lm, "lm",
                          item["item_id"], continuation=f" {item['rationale'].strip()}\n\nAnswer: {answer}"))
    return ex


def build_stage_examples(item: dict, *, stage: int, n_stages: int, k: int, rng: random.Random,
                         subq_target: str = "cot", p_new: float = 0.5, weight: float = 1.0,
                         level_balanced: bool = False) -> list[Example]:
    """Depth-curriculum sub-questions for one item. Stage s (0-based, of n_stages) covers the item's tree up to
    level ceil((s+1)(M+1)/n_stages) - 1, where M is the item's deepest level, so every item reaches its own top in
    the last stage. Each of the k drawn nodes comes from the levels this stage adds with probability p_new, else
    from all levels covered so far (replay); stages that add no level replay only. level_balanced draws a covered
    level uniformly first (so the many depth-0 facts do not dominate replay); stage=n_stages-1 with p_new=0 is
    plain replay over the whole tree (used for node replay during the final stage)."""
    nodes = [sq for sq in item.get("subqs", []) if "id" in sq]
    if not nodes:
        return []
    d = node_depths(nodes)
    top = max(d.values())
    hi = math.ceil((stage + 1) * (top + 1) / n_stages) - 1
    lo = math.ceil(stage * (top + 1) / n_stages) - 1 if stage > 0 else -1      # levels <= lo were covered before
    covered = [sq for sq in nodes if d[sq["id"]] <= hi]
    new = [sq for sq in covered if d[sq["id"]] > lo]
    ex = []
    for _ in range(k):
        pool = new if new and rng.random() < p_new else covered
        if level_balanced:
            lv = rng.choice(sorted({d[x["id"]] for x in pool}))
            pool = [x for x in pool if d[x["id"]] == lv]
        sq = rng.choice(pool)
        p = sq.get("truth") if subq_target == "truth" and sq.get("truth") is not None else None
        p = (1.0 if p else 0.0) if p is not None else sq.get("p_cot" if subq_target != "fresh" else "p_fresh")
        if p is None:
            p = sq.get("p_fresh")
        if p is None:
            continue
        yes_first = rng.random() < 0.5
        a, b = ("Yes", "No") if yes_first else ("No", "Yes")
        ex.append(Example(SUBQ_TEMPLATE.format(problem=item["prompt"], question=sq["question"], a=a, b=b), ["A", "B"],
                          _soft(p, yes_first), weight * item.get("weight", 1.0), "subq", item["item_id"]))
    return ex


def build_transition_views(item: dict, *, k: int, rng: random.Random, shuffled: bool = False, keep_prob: float = 1.0,
                           weight: float = 0.5, subq_target: str = "cot", subq_filter: str = "all") -> list[Example]:
    """Tree-transition auxiliary views: k nodes, each asked with the results of its parent nodes stated in the prompt
    (the local step parents -> child of the teacher's dependency tree). Nodes with parents are drawn first.
    shuffled=True is the edge-rewiring control: the same node, label and number of stated results, but the stated
    results come from nodes of the same item that are neither the node's parents nor its descendants. keep_prob drops
    each stated result independently (scaffold withdrawal; 0 gives plain node questions). subq_filter restricts the
    asked nodes only (parents and the rewiring pool come from every node)."""
    nodes = [sq for sq in item.get("subqs", []) if "id" in sq]
    if not nodes:
        return []
    by = {sq["id"]: sq for sq in nodes}
    parents = {sq["id"]: [x for x in (sq.get("depends_on") or []) if x in by] for sq in nodes}
    children = {i: [j for j in by if i in parents[j]] for i in by}
    def descendants(i: str) -> set:
        out, todo = set(), list(children[i])
        while todo:
            j = todo.pop()
            if j not in out:
                out.add(j)
                todo += children[j]
        return out
    with_p = [sq for sq in nodes if parents[sq["id"]] and _node_p(sq, subq_target) is not None]
    without = [sq for sq in nodes if not parents[sq["id"]] and _node_p(sq, subq_target) is not None]
    if subq_filter != "all":
        with_p = [sq for sq in with_p if subq_keep(sq, subq_filter)]
        without = [sq for sq in without if subq_keep(sq, subq_filter)]
    pick = rng.sample(with_p, min(k, len(with_p)))
    pick += rng.sample(without, min(k - len(pick), len(without)))
    ex = []
    for sq in pick:
        given = [by[x] for x in parents[sq["id"]]]
        if shuffled and given:
            banned = {sq["id"], *parents[sq["id"]], *descendants(sq["id"])}
            pool = [x for x in nodes if x["id"] not in banned]
            given = rng.sample(pool, min(len(given), len(pool)))
        lines = [ln for ln in (_fact_line(x) for x in given) if ln and rng.random() < keep_prob]
        problem = item["prompt"] + (f"\n\n{FACTS_HEADER}\n" + "\n".join(lines) if lines else "")
        yes_first = rng.random() < 0.5
        a, b = ("Yes", "No") if yes_first else ("No", "Yes")
        ex.append(Example(SUBQ_TEMPLATE.format(problem=problem, question=sq["question"], a=a, b=b), ["A", "B"],
                          _soft(_node_p(sq, subq_target), yes_first), weight * item.get("weight", 1.0), "subq",
                          item["item_id"]))
    return ex


def _mc_node_view(item: dict, sq: dict, lines: list[str], tgt: str, rng: random.Random, weight: float,
                  m: int = 3, p_pair: float = 0.85) -> Example | None:
    """A tree node asked in multiple-choice form, with the same stated results as its yes/no view: the node, other
    confident nodes of the item and "None of them" as options ("Which of these questions has the answer Yes?"). A Yes
    node goes with m-1 confident No nodes; a No node with one confident Yes node (when the item has one) and m-2 No
    nodes (with probability p_pair; otherwise "None of them" is the answer, about 7% of views). The soft target is the
    build_mc_views one, P(i) proportional to p_i * prod_{j != i} (1 - p_j) and P(none)
    to prod_j (1 - p_j). None when the node has no target or the item has too few confident nodes."""
    p = _node_p(sq, tgt)
    rest = [x for x in item.get("subqs", []) if x is not sq and x.get("question") != sq.get("question")
            and _node_p(x, tgt) is not None]
    no = [x for x in rest if _node_p(x, tgt) <= 0.3]
    yes = [x for x in rest if _node_p(x, tgt) >= 0.7]
    if p is None:
        return None
    # a No node is asked next to a confident Yes node of the item when there is one, so "None of them" is not the
    # answer half of the time (a hedge-option prior); the node itself is still an option to judge
    extra = [rng.choice(yes)] if p < 0.5 and yes and rng.random() < p_pair else []
    if len(no) < m - 1 - len(extra):
        return None
    chosen = [sq] + extra + rng.sample(no, m - 1 - len(extra))
    rng.shuffle(chosen)
    ps = [min(max(_node_p(x, tgt), 1e-4), 1 - 1e-4) for x in chosen]
    none = math.prod(1 - q for q in ps)
    scores = [q * none / (1 - q) for q in ps] + [none]
    z = sum(scores)
    letters = [chr(65 + i) for i in range(m + 1)]
    opts = "\n".join(f"{L}) {t}" for L, t in zip(letters, [x["question"] for x in chosen] + [MC_NONE]))
    problem = item["prompt"] + (f"\n\n{FACTS_HEADER}\n" + "\n".join(lines) if lines else "")
    return Example(SUBQ_MC_TEMPLATE.format(problem=problem, opts=opts), letters, [x / z for x in scores], weight,
                   "subq", item["item_id"])


def build_full_tree(item: dict, *, mode: str, rng: random.Random, cap: int = 10, aux_total: float = 1.0,
                    final: str = "teacher", subq_target: str = "cot", permute_final: float = 0.0,
                    node_format: str = "yesno", eligible_only: bool = False, subq_filter: str = "all",
                    view: str = "problem", sources: set | None = None, node_weight: float = 0.0,
                    stats: dict | None = None, mix_p: float = 0.5, mix_seed: int = 0,
                    mix_epoch: int = 0, scaffold_p: float | None = None, depth_weight: float = 0.0,
                    node_types: set | frozenset | None = None, target_seed: int | None = None,
                    target_kind: str = "shuffled", placebo_bank: PlaceboBank | None = None,
                    node_render: str = "op") -> list[Example]:
    """One problem with ALL its tree nodes (up to `cap`) instead of k sampled ones: [final] + one view per node.
    mode "true": eligible nodes are asked with their parents' results stated; "matched": with the same number of
    depth-matched non-parent results (see full_tree_plan); "plain": every node asked without facts; "placebo": the
    plain views at zero weight; "control": the item's control questions (fresh targets), plainly. The auxiliary
    weight is normalised per problem (aux_total / number of views), so larger trees do not weigh more.
    node_format "mc" asks each node in multiple-choice form (_mc_node_view) where the item allows it.
    eligible_only keeps only the eligible views of the plan (TFE/TFME/GFE: every asked view differs between true and
    matched); the plan, cap included, is drawn first as before, so the asked nodes are the same in every mode. An item
    with no eligible view trains its final question only. With eligible_only, plain views draw from rng as if they
    stated the parents' results, so GFE stays in lockstep with TFE/TFME (same node subsample, same option order).
    subq_filter restricts the asked nodes (full_tree_plan); the control mode is not filtered.
    cap 0 asks every node. sources (a set of item sources, None = all): items of other sources train their final
    question only. node_weight > 0 gives every node view that weight (times the item weight) instead of the
    per-problem split aux_total / views. view "local" / "ancestors" (Tree Forcing): see _local_views; "problem" (the
    default) is the behaviour above. view "statements" (GFP-S, mode "plain" only): the problem view with the problem
    replaced by its statement block (treeplan.statement_block: no final question, no options); rng draws, nodes,
    targets and weights are exactly those of mode "plain" with the problem view (GFP). views "local+ctxparse" (TFLP)
    and "mix" (MIX), mode "true" only: TFL with some nodes asked in the statements view instead (see _local_views;
    mix_p, mix_seed and mix_epoch key MIX's per-node coin).
    GTF (grounded Tree Forcing) = GFP-S plus two options, both off by default:
    - scaffold_p (None = off; view "statements" only): this problem's probability p(t) (treeplan.scaffold_fade) that a
      derived node is asked with its parents' results stated under the statement block (_scaffold_lines); the coin
      and the line order come from the node's own generator (treeplan.scaffold_rng, keyed by mix_seed, mix_epoch,
      item and node), never from rng, so nodes, order, targets (yes/no order included) and weights stay GFP-S's;
    - depth_weight G (0 = off; every view and mode except control): the problem's node-view weight (aux_total, or
      node_weight x views) split in proportion to (1 + node depth)^G instead of evenly (_depths).
    node_types (None = every node; --tree-node-types, e.g. GFPSNC = GFP-S without the K&K "case" nodes; every view and
    mode except control): only the nodes whose "type" is listed are asked (full_tree_plan, before the cap; items of
    other sources are untouched). The weight convention is unchanged, so the kept views share the same per-problem
    total (aux_total / kept views each); depths, parents, matched pools and scaffold results still use every node.
    Not in rng lockstep with the unfiltered arm (fewer views draw fewer yes/no orders).
    Multiple-choice nodes (treeplan.is_mc_node: the node carries "options" and "answer"; opt-in data such as
    scripts/build_fa_sf.py --trees) are asked in the same context as a yes/no node of the view (stated results and
    GTF scaffold included) with their own options (mc_node_example: SUBQ_OPTS_TEMPLATE, labels A, B, ..., one-hot
    target on the answer, the per-problem weight) and draw nothing from rng; node_format "mc" leaves them as they are.
    The local views (_local_views) refuse them.
    target_seed (None = off, the data's targets; --tree-target shuffled = GFPSR / GFPQR, the random-label control):
    mode "plain" with views "problem" / "statements" / "statements+question" only (yes/no node format, no eligible_only,
    no scaffold: no view states or selects by a real result). Every asked node keeps its view, question, option order
    and weight, and the final question is untouched; only a node's target is replaced by its shuffled_targets
    one (the item's P(Yes) values permuted among its nodes, keyed by target_seed and the item; an MC node: a uniformly
    drawn option, one-hot). Nothing is drawn from rng, so the run is in rng lockstep with the real-target arm (same
    nodes, order, option orders and weights). stats counts shuffled_views, shuffled_agree (the random target's answer
    is the real one), shuffled_yes and real_yes (yes/no views with P(Yes) >= .5).
    Why not --aux-kind cot_randlabel (G3R) or a --subq-target value: cot_randlabel is the grouped loop's (build_group),
    draws from the training rng (breaking the lockstep) and redraws a uniform 50/50 label per view and epoch;
    --subq-target picks which real field is the target and also feeds the non-tree loops.
    target_kind "placebo" (with target_seed; --tree-target placebo = GFPSX / GFPQX, the placebo control): the same
    scope, but every asked node keeps its view, option order, weight and answer while its question is replaced by its
    placebo_questions one (a surface question about the problem text with a correct answer, keyed by target_seed and
    the item; placebo_bank supplies the "no" sentences of other problems); the target is the hard answer (with hard node
    targets, every view's target vector is GFPS's). Nothing is drawn from rng (rng lockstep with GFPS / GFPQ). An asked
    multiple-choice node raises ValueError. stats counts placebo_views, placebo_yes, real_yes, placebo_words /
    real_words (whitespace words of the placebo / node questions) and placebo_<template> per template. (Unrelated to
    mode "placebo", the zero-weight plain views.)
    node_format "value" (value nodes, scripts/build_gsm8k_ops.py; arms GFPQV / GFPQVQ / GFPQO / GFPVB): mode "plain",
    views problem / statements / statements+question / bare, no eligible_only, target_seed, scaffold, depth weight or
    node_types: the final, then every node of item[VALUE_FIELD] that has the rendering node_render ("op", "qop",
    "opsel"; value_mc_node) as a multiple-choice node in the view's context (bare: the node question alone,
    SUBQ_BARE_TEMPLATE), labels A..C, one-hot target on its answer, weight aux_total / number of value views (or
    node_weight) times the item weight (_value_views). The yes/no nodes in item["subqs"] are not asked. Nothing is
    drawn from rng (cap 0; a cap below the node count samples them).
    node_render "vt" / "vtr" (verification nodes, scripts/build_gsm8k_vt.py; arms GFPQVT / GFPQVTR; view "problem"
    only): the same, with the nodes of item[VERIFY_FIELD] (one per asked option of the final question, four
    descriptions as options, labels A..D) in the problem view (the prompt with its options); vtr = vt's stems and
    options with the control answer (the problem's descriptions permuted among its nodes); the value nodes in
    item[VALUE_FIELD] are not asked.
    node_format "yesno+value" (GFPQVM; views problem / statements / statements+question): exactly the yes/no views of
    node_format "yesno" (the same call: same nodes, option orders, targets and rng draws), followed by the value views
    of node_render. Weights (node_weight 0): with both formats present, every yes/no view weighs (aux_total / n_yesno)
    x 0.5 and every value view aux_total x 0.5 / n_value (times the item weight), so each format carries half of the
    problem's aux_total; a problem with views of one format only gives that format the whole aux_total. With
    node_weight > 0 every view weighs node_weight."""
    if node_format in VALUE_FORMATS:
        return _value_tree(item, mode=mode, rng=rng, cap=cap, aux_total=aux_total, final=final,
                           subq_target=subq_target, permute_final=permute_final, node_format=node_format,
                           eligible_only=eligible_only, subq_filter=subq_filter, view=view, sources=sources,
                           node_weight=node_weight, stats=stats, scaffold_p=scaffold_p, depth_weight=depth_weight,
                           node_types=node_types, target_seed=target_seed, node_render=node_render)
    if view == "bare":
        raise ValueError("--tree-view bare is implemented for --node-format value (value nodes) only")
    fin = [e for e in build_examples(item, final=final, subq="none", subq_target=subq_target, lambda_sub=0.0, rng=rng,
                                     permute_final=permute_final) if e.kind == "final"]
    if not fin:
        return []
    if sources is not None and item.get("source") not in sources:
        return fin
    if scaffold_p is not None and (view != "statements" or mode != "plain" or not 0.0 <= scaffold_p <= 1.0):
        raise ValueError(f"scaffold_p {scaffold_p}: a probability, for --tree-full plain --tree-view statements (GTF)")
    if depth_weight and mode == "control":
        raise ValueError("--tree-depth-weight needs tree nodes: control questions have no depth")
    if node_types is not None and mode == "control":
        raise ValueError("--tree-node-types needs tree nodes: control questions have no type")
    if target_seed is not None and target_kind not in ("shuffled", "placebo"):
        raise ValueError(f"target_kind {target_kind}: shuffled or placebo")
    if target_seed is not None and (mode != "plain" or view not in ("problem", "statements", "statements+question")
                                    or node_format != "yesno" or eligible_only or scaffold_p is not None):
        raise ValueError(f"--tree-target {target_kind} is implemented for --tree-full plain, views problem / statements "
                         "/ statements+question, yes/no node format, without --tree-eligible-only or --tree-scaffold")
    if view not in ("problem", "statements", "statements+question"):
        return fin + _local_views(item, mode=mode, view=view, rng=rng, cap=cap, aux_total=aux_total,
                                  subq_target=subq_target, node_format=node_format, eligible_only=eligible_only,
                                  subq_filter=subq_filter, node_weight=node_weight, stats=stats, mix_p=mix_p,
                                  mix_seed=mix_seed, mix_epoch=mix_epoch, depth_weight=depth_weight,
                                  node_types=node_types)
    if view in ("statements", "statements+question") and (mode != "plain" or eligible_only or node_format != "yesno"):
        raise ValueError("--tree-view statements is implemented for --tree-full plain (yes/no nodes, all nodes)")
    context = (statement_block(item) if view == "statements" else problem_no_options(item)
               if view == "statements+question" else item["prompt"])
    wi = item.get("weight", 1.0)
    views = []
    if eligible_only and mode == "control":
        raise ValueError("eligible_only needs tree nodes: control questions have no parents")
    by = {sq["id"]: sq for sq in item.get("subqs", []) if "id" in sq}
    if mode == "control":
        pool = [sq for sq in item.get("random_subqs", []) if sq.get("p_fresh") is not None]
        if len(pool) > cap:
            pool = rng.sample(pool, cap)
        views = [(sq, [], "fresh", 0) for sq in pool]
    else:
        plan = full_tree_plan(item, cap=cap, rng=rng, subq_target=subq_target, subq_filter=subq_filter,
                              node_types=node_types, stats=stats)
        if eligible_only:
            plan = [e for e in plan if e["parents"]]
        for e in plan:
            facts = {"true": e["parents"], "matched": e["matched"]}.get(mode, [])
            views.append((e["node"], [by[j] for j in facts], subq_target, len(e["parents"])))
    out = list(fin)
    w = 0.0 if mode == "placebo" else (node_weight or aux_total / max(1, len(views)))
    ws = [w] * len(views)
    if depth_weight and views:                              # GTF: same total per problem, split by depth
        total = 0.0 if mode == "placebo" else (node_weight * len(views) if node_weight else aux_total)
        depth = _depths(item)
        ws = depth_split(total, [depth[sq["id"]] for sq, *_ in views], depth_weight)
    if stats is not None:
        stats["items_with_views"] = stats.get("items_with_views", 0) + bool(views)
        stats["views"] = stats.get("views", 0) + len(views)
        if scaffold_p is not None:
            for key in ("scaffold_views", "scaffold_lines", "scaffold_eligible"):
                stats[key] = stats.get(key, 0)
    shuf = plc = None
    if target_seed is not None and target_kind == "placebo":    # GFPSX / GFPQX: the placebo questions, drawn off rng
        plc = placebo_questions(item, target_seed, placebo_bank, subq_target) if views else {}
        st = stats if stats is not None else {}
        for key in ("placebo_views", "placebo_yes", "real_yes", "placebo_words", "real_words",
                    *(f"placebo_{t}" for t in PLACEBO_TEMPLATES)):
            st[key] = st.get(key, 0)
    elif target_seed is not None:                           # GFPSR / GFPQR: the random targets, drawn off rng
        shuf = shuffled_targets(item, target_seed, subq_target)
        st = stats if stats is not None else {}
        for key in ("shuffled_views", "shuffled_agree", "shuffled_yes", "real_yes"):
            st[key] = st.get(key, 0)
    for (sq, facts, tgt, k), wk in zip(views, ws):
        lines = [_fact_line(x) for x in facts]
        rng.shuffle(lines)
        if eligible_only and not lines:                     # GFE: the draws of the k stated lines of TFE/TFME
            rng.shuffle([None] * k)
        mc = mode != "control" and is_mc_node(sq)         # a node with its own options (opt-in data)
        if node_format == "mc" and mode != "control" and not mc:   # multiple-choice node view, same stated results
            ex = _mc_node_view(item, sq, lines, tgt, rng, wk * wi)
            if ex is not None:
                out.append(ex)
                continue
        problem = context + (f"\n\n{FACTS_HEADER}\n" + "\n".join(lines) if lines else "")
        if scaffold_p is not None:                          # GTF (lines is empty in mode plain)
            sc = _scaffold_lines(item, sq, by, scaffold_p, mix_seed, mix_epoch, stats)
            if sc:
                problem = context + f"\n\n{LOCAL_HEADER}\n" + "\n".join(sc)
        if mc:                                              # its options and answer, as a final question; no draw
            if plc is not None:
                raise ValueError(f"--tree-target placebo: {item['item_id']} node {sq['id']} is multiple-choice")
            if shuf is not None:                            # GFPSR / GFPQR: a random option
                st["shuffled_views"] += 1
                st["shuffled_agree"] += shuf[sq["id"]] == sq["answer"]
                sq = dict(sq, answer=shuf[sq["id"]])
            out.append(mc_node_example(problem, sq, wk * wi, item["item_id"]))
            continue
        yes_first = rng.random() < 0.5
        a, b = ("Yes", "No") if yes_first else ("No", "Yes")
        p = _node_p(sq, tgt)
        question = sq["question"]
        if shuf is not None:                                # GFPSR / GFPQR: the permuted target, same view and draws
            st["shuffled_views"] += 1
            st["shuffled_agree"] += (shuf[sq["id"]] >= 0.5) == (p >= 0.5)
            st["shuffled_yes"] += shuf[sq["id"]] >= 0.5
            st["real_yes"] += p >= 0.5
            p = shuf[sq["id"]]
        elif plc is not None:                               # GFPSX / GFPQX: the placebo question, its correct answer
            q, yes, kind = plc[sq["id"]]
            st["placebo_views"] += 1
            st["placebo_yes"] += yes
            st["real_yes"] += p >= 0.5
            st["placebo_words"] += len(q.split())
            st["real_words"] += len(question.split())
            st[f"placebo_{kind}"] += 1
            question, p = q, float(yes)
        out.append(Example(SUBQ_TEMPLATE.format(problem=problem, question=question, a=a, b=b), ["A", "B"],
                           _soft(p, yes_first), wk * wi, "subq", item["item_id"]))
    return out


def _value_tree(item: dict, *, mode: str, rng: random.Random, cap: int, aux_total: float, final: str,
                subq_target: str, permute_final: float, node_format: str, eligible_only: bool, subq_filter: str,
                view: str, sources: set | None, node_weight: float, stats: dict | None, scaffold_p: float | None,
                depth_weight: float, node_types, target_seed: int | None, node_render: str) -> list[Example]:
    """build_full_tree for node_format "value" / "yesno+value" (see there)."""
    views = ("problem", "statements", "statements+question") + (("bare",) if node_format == "value" else ())
    if (mode != "plain" or view not in views or eligible_only or target_seed is not None or scaffold_p is not None
            or depth_weight or node_types is not None):
        raise ValueError(f"--node-format {node_format} is implemented for --tree-full plain, views {', '.join(views)}, "
                         "without --tree-eligible-only, --tree-target, --tree-scaffold, --tree-depth-weight or "
                         "--tree-node-types")
    if node_render not in VALUE_RENDERINGS:
        raise ValueError(f"--node-render {node_render!r}: one of {', '.join(VALUE_RENDERINGS)}")
    if node_render in VERIFY_RENDERINGS and view != "problem":
        raise ValueError(f"--node-render {node_render} (verification nodes ask what an option is): --tree-view problem "
                         "(the prompt with its options)")
    if node_format == "yesno+value":                  # the GFPQ group itself: same call, same draws
        out = build_full_tree(item, mode=mode, rng=rng, cap=cap, aux_total=aux_total, final=final,
                              subq_target=subq_target, permute_final=permute_final, node_format="yesno",
                              subq_filter=subq_filter, view=view, sources=sources, node_weight=node_weight,
                              stats=stats)
    else:
        out = [e for e in build_examples(item, final=final, subq="none", subq_target=subq_target, lambda_sub=0.0,
                                         rng=rng, permute_final=permute_final) if e.kind == "final"]
    if not out or (sources is not None and item.get("source") not in sources):
        return out
    yn = [e for e in out if e.kind == "subq"]
    half = node_format == "yesno+value" and bool(yn) and not node_weight
    vals = _value_views(item, view=view, rng=rng, cap=cap, aux_total=aux_total * 0.5 if half else aux_total,
                        node_render=node_render, node_weight=node_weight, stats=stats)
    if half and vals:                                 # each format carries half of the problem's aux_total (no value
        out = [e if e.kind != "subq" else dataclasses.replace(e, weight=e.weight * 0.5) for e in out]   # view: all)
    return out + vals


def shuffled_targets(item: dict, seed: int, subq_target: str = "cot") -> dict:
    """--tree-target shuffled (GFPSR / GFPQR: the random-label control of GFPS / GFPQ): a random target for every tree
    node of the item that has one, by node id. Yes/no nodes (a target under subq_target, _node_p): the item's P(Yes)
    values permuted among them, so the item keeps exactly its multiset of targets (its yes rate, and the soft values if
    any) while a node's target no longer depends on its question (a node keeps its own value by chance, about half the
    time on balanced trees). Multiple-choice nodes (is_mc_node): an option index drawn uniformly. Drawn over every
    node with a target (before the cap, --tree-node-types or --subq-filter select the asked ones), from generators of
    their own seeded with str keys (sha512, as treeplan.scaffold_rng): (seed, item) for the permutation, (seed, item,
    node) for an MC draw. The epoch is not in the key: every node keeps one random target over the whole run (a fixed
    random labelling, as consistent across epochs as the real targets, not noise redrawn each epoch), and the draw
    depends on nothing else (not the training rng, PYTHONHASHSEED, the cap or the order of the views)."""
    nodes = [sq for sq in item.get("subqs", []) if "id" in sq]
    yn = [sq for sq in nodes if not is_mc_node(sq) and _node_p(sq, subq_target) is not None]
    ps = [_node_p(sq, subq_target) for sq in yn]
    random.Random(f"tree-target|{seed}|{item['item_id']}").shuffle(ps)
    out = {sq["id"]: p for sq, p in zip(yn, ps)}
    for sq in nodes:
        if is_mc_node(sq):
            out[sq["id"]] = random.Random(f"tree-target|{seed}|{item['item_id']}|{sq['id']}").randrange(
                len(sq["options"]))
    return out


# ---- --tree-target placebo (GFPSX / GFPQX): surface questions about the problem text instead of the node questions ----
PLACEBO_TEMPLATES = ("name", "number", "word", "sentence")    # this order breaks ties and orders the fallbacks
PLACEBO_MIN_K = 3              # a sentence question quotes at least this many words (or the whole, shorter sentence)
PLACEBO_OVERLAP = 0.5          # a foreign sentence may share at most this share of its content tokens with the prompt
PLACEBO_FOREIGN = 24           # other problems drawn per problem for its "no" sentences and words
_PL_FIXED = {"name": 6, "number": 7, "word": 8}      # words of the fixed templates (a sentence: 7 + k)
_PL_STOP = frozenset("""a about above after again against all also am an and any are as at be because been before
being below between both but by can could did do does doing down during each either else every few for from further
had has have having he her here hers herself him himself his how i if in into is it its itself just let me more most
much must my myself neither no nor not now of off on only or other our ours ourselves out over own per same she should
so some such than that the their theirs them themselves then there these they this those through to too under until
up upon us very was we were what when where whether which while who whom whose why will with within without would yet
you your yours yourself""".split())
_PL_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
            "seventeen eighteen nineteen").split()
_PL_TENS = dict(zip(range(20, 100, 10), "twenty thirty forty fifty sixty seventy eighty ninety".split()))
_PL_ORD = ("zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth "
           "fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth").split()
_PL_NUMX = {1: "once single", 2: "twice double doubled half halves pair pairs", 3: "thrice triple tripled thirds",
            4: "quarter quarters", 12: "dozen dozens", 100: "hundred", 1000: "thousand", 10 ** 6: "million"}
_PL_NUMWORDS = frozenset(_PL_ONES + _PL_ORD + list(_PL_TENS.values()) + [w[:-1] + "ieth" for w in _PL_TENS.values()]
                         + " ".join(_PL_NUMX.values()).split() + ["billion", "percent", "hundredth", "thousandth"])
# Names with no other English reading (Will, Max, Grace, Rose, Mark, Bill, ... left out): numgen, K&K (sources.NAMES and
# scripts/build_kk_deep.py's EXTRA_NAMES), PrOntoQA's entities and frequent GSM8K names.
_PL_NAME_X = ("Quinn Rosa Sam Tina Umar Vera Yara John Mary Mike Peter Sarah Jerry Martha Harry Lisa Steve Janet Joe "
              "Kevin Carl Jill Ryan Michael Jim Jason Carla Jenny Ben Andrew Anna Jane Jimmy Billy Tony Julia Adam Emily "
              "Jake Susan Lucy Jacob Jeff Dave Betty Monica Kim Gerald Amanda Kyle Rick Rachel Nancy Andy Jose").split()
_PL_NOT_NAMES = frozenset("Will Max Kit Ivy Opal Hope Ray Pip Wren Sally Grace Rose Mark Bill May June Bo Mo Cy Ty "
                          "Val".split())
# "No" words when no other problem of the source has one (e.g. the K&K and Lean vocabularies are shared by all items).
_PL_WORD_POOL = ("anchor basket blanket bridge candle carpet castle cherry compass copper cotton desert engine feather "
                 "forest garden glacier hammer harbor helmet island jacket kettle ladder lantern lemon market meadow "
                 "mirror needle orchard paddle parade pencil pepper pillow planet pocket puzzle rabbit ribbon river "
                 "saddle saucer shovel silver spider stable tablet ticket timber tunnel valley violin wagon walnut "
                 "window winter").split()
_PL_NUM = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_PL_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_PL_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_PL_NAMES: list = []


def _pl_names() -> list[str]:
    """The name pool of the name questions, sorted (built once)."""
    if not _PL_NAMES:
        from . import numgen, pronto, sources
        pool = set(sources.NAMES) | {n for n, _ in numgen.NAMES} | set(pronto.ENTITIES) | set(_PL_NAME_X)
        _PL_NAMES.extend(sorted(pool - _PL_NOT_NAMES))
    return _PL_NAMES


def _pl_val(tok: str):
    """A number token's value ("1,200" -> 1200, "2.50" -> 2.5, 24.0 -> 24): what number questions compare."""
    v = float(str(tok).replace(",", ""))
    return int(v) if v == int(v) else v


def _pl_nums(text: str) -> set:
    return {_pl_val(x) for x in _PL_NUM.findall(text)}


def _pl_words(text: str) -> list[str]:
    """The word tokens of a text: letters only, a possessive 's dropped, other contractions skipped."""
    out = []
    for m in _PL_WORD.finditer(text.replace("’", "'")):
        w = m.group()
        if "'" in w:
            if not w.lower().endswith("'s"):
                continue
            w = w[:-2]
        out.append(w)
    return out


def _pl_variants(w: str) -> set:
    """w lower-cased with its plural / singular look-alikes (candle ~ candles, snarpus ~ snarpuses, city ~ cities)."""
    w = w.lower()
    v = {w, w + "s", w + "es"}
    if w.endswith("s"):
        v.add(w[:-1])
    if w.endswith("es"):
        v.add(w[:-2])
    if w.endswith("ies"):
        v.add(w[:-3] + "y")
    if w.endswith("y"):
        v.add(w[:-1] + "ies")
    return v


def _pl_spelled(n: int) -> list[str]:
    """Word forms of a whole number that a text may use instead of digits (four, fourth, quarter, sixty-one, ...)."""
    out = list(_PL_NUMX.get(n, "").split())
    if 0 <= n < 20:
        out += [_PL_ONES[n], _PL_ORD[n]]
    elif 20 <= n < 100:
        t = _PL_TENS[n - n % 10]
        out += [t, t[:-1] + "ieth"] if n % 10 == 0 else [f"{t}-{_PL_ONES[n % 10]}", f"{t} {_PL_ONES[n % 10]}",
                                                         f"{t}-{_PL_ORD[n % 10]}", f"{t} {_PL_ORD[n % 10]}"]
    return out


def _pl_sentences(text: str) -> list[str]:
    """The sentences of a text, each an exact substring of it: per line, a K&K "- " bullet dropped, split after . ! ?"""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("- "):
            line = line[2:]
        out += [s for s in _PL_SENT.split(line) if s.strip()]
    return out


def _pl_prefix(s: str, k: int) -> tuple[str, int]:
    """The first k words of sentence s, cut at the k-th word's last character (an exact substring of s), with "..."
    when words are left out; and the number of words quoted."""
    ws = list(re.finditer(r"\S+", s))
    k = min(k, len(ws))
    return s[:ws[k - 1].end()] + ("..." if k < len(ws) else ""), k


def placebo_text(item: dict) -> str:
    """The text a placebo question is about: the statement block (treeplan.statement_block, GFP-S's context), or for a
    gsm8k_tf item without a clean statement / question cut (about 60% of GSM8K) the whole problem text (GFPQ's context
    minus the "Question: " label); for a qasc_tf item (no statement block: the problem is its question) the question
    text. Inside every view's context (problem / statements / statements+question) wherever that view exists. Other
    sources without a statement block raise ValueError."""
    from .treeplan import GSM8K_TF, QASC_TF, _gsm8k_tf_problem, _qasc_tf_question
    try:
        return statement_block(item)
    except ValueError:
        from .treeplan import ENTAILMENTBANK_TF, _entailmentbank_tf_problem
        if item.get("source") == ENTAILMENTBANK_TF:      # entailmentbank_tf: no statement block, the ARC question
            return _entailmentbank_tf_problem(item)
        from .treeplan import STRATEGYQA_TF, _strategyqa_tf_question
        if item.get("source") == STRATEGYQA_TF:          # no statement block: the problem is its question alone
            return _strategyqa_tf_question(item)
        if item.get("source") == QASC_TF:
            return _qasc_tf_question(item)
        if item.get("source") != GSM8K_TF:
            raise
        return _gsm8k_tf_problem(item)


def placebo_tree_numbers(item: dict) -> set:
    """Every number of the item's reasoning tree and answer: in the node questions, the nodes' value / probe fields, the
    numgen program's quantities and the answer options. A number question never asks one of them; a tree number that
    the statement block does not state (a derived value, a wrong probe, an option) occurs in no placebo question."""
    out = set()
    for sq in item.get("subqs", []):
        if "id" not in sq:
            continue
        out |= _pl_nums(sq.get("question", ""))
        for key in ("value", "probe"):
            if isinstance(sq.get(key), (int, float)) and not isinstance(sq.get(key), bool):
                out.add(_pl_val(sq[key]))
    prog = item.get("program")
    for q in (prog.get("quantities") or []) if isinstance(prog, dict) else []:
        if isinstance(q, dict) and isinstance(q.get("value"), (int, float)) and not isinstance(q.get("value"), bool):
            out.add(_pl_val(q["value"]))
    out |= _pl_nums(item.get("prompt", "").partition("\nOptions:\n")[2])
    return out


class PlaceboBank:
    """--tree-target placebo: the placebo texts (placebo_text) of a set of problems by source, sorted by item id (not by
    file order): another problem of the same source is where a placebo "no" sentence (and a "no" word) comes from.
    train_student builds it from the training items of --tree-sources; items without a placebo text are left out."""

    def __init__(self, items):
        by: dict = {}
        for it in items:
            try:
                t = placebo_text(it)
            except ValueError:
                continue
            by.setdefault(it.get("source"), []).append((str(it["item_id"]), t))
        self.texts = {s: sorted(v) for s, v in by.items()}

    def draw(self, item: dict, text: str, rng: random.Random, n: int) -> list[str]:
        """Up to n texts of other problems of item's source (not the item, not the same text), drawn by rng."""
        pool = self.texts.get(item.get("source"), [])
        out = []
        for _ in range(n if len(pool) > 1 else 0):
            iid, t = pool[rng.randrange(len(pool))]
            if iid != str(item["item_id"]) and t != text:
                out.append(t)
        return out


def placebo_questions(item: dict, seed: int, bank: PlaceboBank | None = None, subq_target: str = "cot") -> dict:
    """--tree-target placebo (GFPSX / GFPQX: the placebo control of GFPS / GFPQ): for every yes/no tree node of the item
    with a target, by node id, (question, answer, template): a surface question about the problem's text (placebo_text)
    with a correct answer, in place of the node question. The answer is the node's own (P(Yes) >= .5), so a problem's
    yes rate, and with hard targets every view's target, is GFPS's.
    Templates:
    - name: "Is <Name> mentioned in the problem?": a name of the pool that the text has (case-sensitive whole word), or
      one the whole prompt does not have (any case, plural look-alikes too);
    - number: "Does the problem mention the number N?": a number token of the text, or a nearby whole number the whole
      prompt has neither as a number nor in words (four / fourth / quarter / sixty-one ...);
    - word: "Does the word '<w>' appear in the problem?": a lower-case content word of the text (not a stop word, number
      word or name; at least 3 letters), or one of another problem of the source (else of a fixed pool) that the whole
      prompt does not have in any case or plural form;
    - sentence: "Is the sentence '<first k words...>' part of the problem?": the first k words of a sentence of the text
      (an exact substring), or of a sentence of another problem of the source (bank) that is not a substring of the
      prompt, shares no capitalised word (names) and no single letter (variables) with the prompt, at most
      PLACEBO_OVERLAP of its content tokens with the prompt (no paraphrase of a premise: "Every X is Y" vs "Each X is
      Y"), and contains no tree number.
    Never about the tree: a name, number or word question never uses a token of any node question (any node, asked or
    not; words by plural look-alikes) nor a tree number (placebo_tree_numbers); a sentence question quotes this
    problem's own text or holds no tree number. Absent tokens are checked against the whole prompt (final question and
    options included), so the answer is right in every view (GFPQ shows the final question).
    Length: a node of L words gets a template whose length (name 6, number 7, word 8, sentence 7 + k words, k >= 3) is
    within max(2, L / 4) of L, at random among those that fit (else the nearest); a sentence question quotes
    k = L - 7 words where its sentence allows. Templates are picked among those the problem can ask with both answers,
    so the template does not tell the answer; a problem's questions are distinct where its supply allows.
    Drawn over every yes/no node with a target (before the cap or filters select the asked ones), in node order, from a
    generator of its own keyed by (seed, item) (sha512 of a str key, as shuffled_targets): nothing from the training
    rng, no epoch (each node keeps its placebo question over the run). With the same bank the questions depend on
    nothing else; without a bank (None) no "no" sentence exists and "no" words come from the fixed pool.
    Multiple-choice nodes get none (build_full_tree refuses them under placebo). ValueError if the item has no
    placebo_text."""
    text = placebo_text(item)
    prompt = item.get("prompt", "")
    nodes = [sq for sq in item.get("subqs", []) if "id" in sq]
    yn = [sq for sq in nodes if not is_mc_node(sq) and _node_p(sq, subq_target) is not None]
    if not yn:
        return {}
    rng = random.Random(f"tree-placebo|{seed}|{item['item_id']}")
    tree_nums = placebo_tree_numbers(item)
    node_words = {w.lower() for sq in nodes for w in _pl_words(sq.get("question", ""))}
    p_words = {w.lower() for w in _pl_words(prompt)}
    p_nums = _pl_nums(prompt)
    names = _pl_names()
    name_set = set(names)
    lower_names = {n.lower() for n in names}

    def in_tree(w):
        return bool(_pl_variants(w) & node_words)

    def in_prompt(w):
        return bool(_pl_variants(w) & p_words)

    def content(w):
        lw = w.lower()
        return len(w) >= 3 and w.isalpha() and lw not in _PL_STOP and lw not in _PL_NUMWORDS and lw not in lower_names

    def uniq(xs):
        return list(dict.fromkeys(xs))

    tw = _pl_words(text)
    yes_num, seen = [], set()
    for x in _PL_NUM.findall(text):                     # surface form as in the text, one per value, text order
        if _pl_val(x) not in tree_nums and _pl_val(x) not in seen:
            seen.add(_pl_val(x))
            yes_num.append(x)
    supply = {True: {"name": uniq(w for w in tw if w in name_set and not in_tree(w)),
                     "word": uniq(w for w in tw if w == w.lower() and content(w) and not in_tree(w))},
              False: {"name": [n for n in names if not in_prompt(n) and not in_tree(n)]}}
    supply[True]["number"] = yes_num
    own = _pl_sentences(text)

    def foreign_ok(q: str) -> bool:
        """A "no" sentence quote: absent, about other entities, not a paraphrase, no tree number."""
        if q in prompt or q in text:
            return False
        nums = _pl_nums(q)
        if nums & tree_nums:
            return False
        ws = _pl_words(q)
        for w in ws:
            if len(w) == 1 and w not in ("a", "A", "I") and w.lower() in p_words:
                return False                             # a variable of this problem (Lean)
            if w[0].isupper() and w.lower() not in _PL_STOP and in_prompt(w):
                return False                             # a name / proper noun of this problem
        toks = [w for w in ws if w.lower() not in _PL_STOP]
        if not toks and not nums:
            return False
        shared = sum(in_prompt(w) for w in toks) + len(nums & p_nums)
        return shared <= PLACEBO_OVERLAP * (len(toks) + len(nums))

    # Other problems of the source: PLACEBO_FOREIGN of them, more (up to 4x) until there is a "no" sentence per "no"
    # node (sources with few passing sentences, e.g. Lean, stop at the cap).
    need = sum(_node_p(sq, subq_target) < 0.5 for sq in yn)
    foreign_texts, foreign = [], []
    for j in range(4 * PLACEBO_FOREIGN if bank is not None else 0):
        if j >= PLACEBO_FOREIGN and len(foreign) >= need:
            break
        for t in bank.draw(item, text, rng, 1):
            foreign_texts.append(t)
            ss = _pl_sentences(t)
            for x in rng.sample(ss, min(4, len(ss))):
                if foreign_ok(x) and x not in foreign:
                    foreign.append(x)
    no_words = uniq(w for t in foreign_texts for w in _pl_words(t)
                    if w == w.lower() and content(w) and not in_tree(w) and not in_prompt(w))
    supply[False]["word"] = no_words + [w for w in _PL_WORD_POOL if w not in no_words and not in_tree(w)
                                        and not in_prompt(w)]
    supply[True]["sentence"], supply[False]["sentence"] = own, foreign
    ok = {y: {t: bool(supply[y].get(t)) or (t == "number" and not y) for t in PLACEBO_TEMPLATES} for y in (True, False)}
    both = [t for t in PLACEBO_TEMPLATES if ok[True][t] and ok[False][t]]

    def nwords(x):
        return len(x.split())

    smax = {y: max((nwords(x) for x in supply[y]["sentence"]), default=0) for y in (True, False)}
    used: set = set()                                   # questions asked so far
    taken: dict = {}                                    # (template, answer) -> the supply items used so far

    uses: dict = {}                                     # template -> nodes given it so far

    def room(t, y_ok):
        """Whether the template has room for one more node without looking at its answer: its uses stay under
        len(y_ok) x the smaller supply of the answers in y_ok (an absent number is always new)."""
        cap = min(float("inf") if (t == "number" and not y) else len(supply[y].get(t, ())) for y in y_ok)
        return uses.get(t, 0) < len(y_ok) * cap

    def pick(t, y, render):
        xs = supply[y][t]
        new = [x for x in xs if x not in taken.get((t, y), ())]
        x = (new or xs)[rng.randrange(len(new or xs))]
        taken.setdefault((t, y), set()).add(x)
        return render(x)

    def absent_number() -> str:
        low = prompt.lower()
        bases = sorted({int(abs(v)) for v in _pl_nums(text)} - {0}) or list(range(2, 13))

        def fits(c):
            return (c >= 1 and c not in p_nums and c not in tree_nums and f"Does the problem mention the number {c}?"
                    not in used and not any(re.search(rf"\b{re.escape(w)}\b", low) for w in _pl_spelled(c)))
        for _ in range(60):
            b = bases[rng.randrange(len(bases))]
            c = b + rng.choice((-1, 1)) * rng.randint(1, max(3, b // 10))
            if fits(c):
                return str(c)
        c = max(bases) + 1
        while not fits(c):
            c += 1
        return str(c)

    def make(t, yes, L):
        if t == "name":
            return pick(t, yes, lambda x: f"Is {x} mentioned in the problem?")
        if t == "number":
            if yes:
                return pick(t, yes, lambda x: f"Does the problem mention the number {x}?")
            return f"Does the problem mention the number {absent_number()}?"
        if t == "word":
            return pick(t, yes, lambda x: f"Does the word '{x}' appear in the problem?")
        k = max(PLACEBO_MIN_K, L - 7)                   # sentence: 7 + k words
        ss = supply[yes]["sentence"]
        tk = taken.setdefault((t, yes), set())
        fallback = None                                 # random order: unused sentences of k words or more first
        for x in sorted(rng.sample(ss, len(ss)), key=lambda x: (x in tk, nwords(x) < k)):
            q, _ = _pl_prefix(x, k)
            if not yes and not foreign_ok(q.removesuffix("...")):
                continue
            qq = f"Is the sentence '{q}' part of the problem?"
            if qq not in used:
                tk.add(x)
                return qq
            fallback = fallback or qq
        return fallback

    def span(t, y_ok):
        if t != "sentence":
            return _PL_FIXED[t], _PL_FIXED[t]
        hi = min(smax[y] for y in y_ok)
        return 7 + min(PLACEBO_MIN_K, hi), 7 + hi

    out = {}
    for sq in rng.sample(yn, len(yn)):                  # random order: which nodes meet a used-up supply is by chance,
        yes = _node_p(sq, subq_target) >= 0.5           # not by the node order (e.g. pronto's first node is always Yes)
        L = nwords(sq["question"])
        avail, y_ok = (both, (True, False)) if both else ([t for t in PLACEBO_TEMPLATES if ok[yes][t]], (yes,))

        def dist(t):
            lo, hi = span(t, y_ok)
            return max(lo - L, 0, L - hi)
        pool = ([t for t in avail if room(t, y_ok)]                      # templates with unused items first (by
                or [t for t in avail if t == "sentence"] or avail)       # counts, not this node's answer); then a
        tol = max(2.0, L / 4)                                            # sentence quoted at another length
        cands = [t for t in pool if dist(t) <= tol] or [t for t in pool if dist(t) == min(map(dist, pool))]
        first = cands[rng.randrange(len(cands))]
        q = None
        for t in [first] + sorted((t for t in PLACEBO_TEMPLATES if ok[yes][t] and t != first), key=dist):
            q = make(t, yes, L)
            if q is not None:
                break
        if q is None:
            raise ValueError(f"{item['item_id']}: no placebo question for node {sq['id']}")
        uses[t] = uses.get(t, 0) + 1
        used.add(q)
        out[sq["id"]] = (q, yes, t)
    return out


def _depths(item: dict) -> dict:
    """Node depth by id: the node_depth field when every node has one (v3s/v3sl/numgen), else node_depths()."""
    nodes = [sq for sq in item.get("subqs", []) if "id" in sq]
    return ({sq["id"]: sq["node_depth"] for sq in nodes} if all("node_depth" in sq for sq in nodes)
            else node_depths(nodes))


def _scaffold_lines(item: dict, sq: dict, by: dict, p: float, seed: int, epoch: int, stats: dict | None) -> list[str]:
    """GTF's stated results for one node view (empty: the plain GFP-S view): the "Q: ... A: Yes/No" lines of the
    node's parents (depends_on within the item; _fact_line: program truth, else a confident CoT answer), in an order
    drawn from the node's own generator, with probability p (its first draw). Always plain: leaves (parse / read
    nodes with local_lines, or no parent), nodes with an unconfident parent, and nodes whose stated results would give
    the final answer away (view_leak, the TF views' filter). Every case is counted in stats."""
    st = stats if stats is not None else {}
    bump = lambda key, n=1: st.__setitem__(key, st.get(key, 0) + n)
    par = [x for x in (sq.get("depends_on") or []) if x in by and x != sq["id"]]
    if is_local_leaf(sq) or not par:
        bump("scaffold_leaf")
        return []
    stated = [by[j] for j in par]
    if not all(_fact_line(x) for x in stated):
        bump("scaffold_unconfident")
        return []
    leak = view_leak(item, sq, stated)
    if leak:
        bump(f"scaffold_leak_{leak}")
        return []
    bump("scaffold_eligible")
    r = scaffold_rng(seed, epoch, item["item_id"], sq["id"])
    if not r.random() < p:
        return []
    lines = [_fact_line(x) for x in stated]
    r.shuffle(lines)
    bump("scaffold_views")
    bump("scaffold_lines", len(lines))
    return lines


def _local_views(item: dict, *, mode: str, view: str, rng: random.Random, cap: int, aux_total: float,
                 subq_target: str, node_format: str, eligible_only: bool, subq_filter: str, node_weight: float,
                 stats: dict | None, mix_p: float = 0.5, mix_seed: int = 0, mix_epoch: int = 0,
                 depth_weight: float = 0.0, node_types: set | frozenset | None = None) -> list[Example]:
    """Tree Forcing node views WITHOUT the original problem (local_context): the item's preamble, then
    - parse nodes (local_lines): the statement line(s) they read, in every mode;
    - other nodes, view "local": mode "true" = the results of the node's parents (its depends_on; every parent of a
      solver-tree node is sufficient, scripts/build_local_kk.py), mode "matched" = as many matched non-parent results
      (full_tree_plan's selection, post-audit rules) where the plan finds them, else the true parents (the view is
      then the same in both modes; counted as matched_frozen), so the two modes state the same number of results in
      every view and draw rng in lockstep;
    - other nodes, view "ancestors" (mode "true" only): the results of every transitive ancestor, shallow to deep;
    - other nodes, view "local" with mode "plain" (TFLN): nothing stated, the preamble only. The node set, the drops
      and the rng draws are those of mode "true" (the parents' results are selected and checked, and their line order
      drawn, but not shown), so TFLN is in lockstep with TFL: same nodes, order, weights and targets.
    - views "local+ctxparse" (TFLP) and "mix" (MIX), mode "true" only: TFL, except that some nodes are asked in the
      GFP-S statements view (treeplan.statement_block: the problem without its final question and options, nothing
      stated) instead of their local view. TFLP: the parse nodes. MIX: each node with probability mix_p, by its own
      coin (treeplan.mix_statements, keyed by mix_seed, mix_epoch, item and node), so every node has one view per
      epoch. Both keep the non-leaf nodes without parents that TFL drops, always in the statements view (counted as
      statements_no_parents; none in v3sl or numgen, where such a node would break the lockstep with TFL).
      A statements view still draws TFL's line-order shuffle (the TFLN trick), so both arms are in rng lockstep with
      TFL (their local views are TFL's, byte for byte) and have its nodes, order, weights and soft targets; nodes,
      order, weights and P(Yes) are GFP-S's too as long as TFL drops nothing (0 drops on v3sl and numgen).
      stated_lines counts the results shown, withheld_lines those of TFL's view not shown (MIX's derived statements
      views).
    Views that would state the final answer (view_leak) are dropped and counted; so are non-leaf nodes without
    parents or with a parent whose result is not confident (nothing sufficient to state). The node is then asked as
    in the problem-view arms (yes/no order drawn from rng, the same soft target). Weight: node_weight, else
    aux_total / views (the full-tree convention); depth_weight G splits the same total in proportion to
    (1 + node depth)^G. node_types restricts the asked nodes (full_tree_plan; see build_full_tree)."""
    if (mode not in ("true", "matched", "plain") or (view != "local" and mode != "true")
            or view not in ("local", "ancestors") + CTX_VIEWS):
        raise ValueError(f"--tree-view {view} is implemented for --tree-full true (and matched / plain with the local "
                         "view)")
    if eligible_only or node_format != "yesno":
        raise ValueError(f"--tree-view {view} does not combine with --tree-eligible-only / --node-format mc")
    if any(is_mc_node(sq) for sq in item.get("subqs", [])):
        raise ValueError(f"{item.get('item_id')}: multiple-choice tree nodes are read in the problem / statements "
                         f"views only, not in --tree-view {view}")
    ctx_view = view in CTX_VIEWS
    if ctx_view and not 0.0 <= mix_p <= 1.0:
        raise ValueError(f"mix_p {mix_p} is not a probability")
    block = statement_block(item) if ctx_view else None      # TFLP/MIX: raises for sources without a statement block
    if ctx_view and item.get("local_preamble") is None:
        raise ValueError(f"{item.get('item_id')}: no local_preamble (local views need e.g. data/student_v3sl)")
    st = stats if stats is not None else {}
    bump = lambda k, n=1: st.__setitem__(k, st.get(k, 0) + n)
    nodes = [sq for sq in item.get("subqs", []) if "id" in sq]
    by = {sq["id"]: sq for sq in nodes}
    depth = ({i: by[i]["node_depth"] for i in by} if all("node_depth" in sq for sq in nodes) else node_depths(nodes))
    anc = ancestors(nodes) if view == "ancestors" else None
    views, no_par = [], set()
    for e in full_tree_plan(item, cap=cap, rng=rng, subq_target=subq_target, subq_filter=subq_filter,
                            node_types=node_types, stats=st):
        sq = e["node"]
        if is_local_leaf(sq):
            stated = []
        else:
            par = [x for x in (sq.get("depends_on") or []) if x in by and x != sq["id"]]
            if not par and not ctx_view:
                bump("dropped_no_parents")
                continue
            if not par:                             # TFLP / MIX: no local view, asked in the statements view
                ids = []
                no_par.add(sq["id"])
            elif view == "ancestors":
                ids = sorted(anc[sq["id"]] - {sq["id"]})
            elif mode == "matched" and e["parents"]:
                ids = e["matched"]
            else:
                ids = par
                if mode == "matched":
                    bump("matched_frozen")
            stated = [by[j] for j in ids]
            if not all(_fact_line(x) for x in stated):
                bump("dropped_unconfident_parent")
                continue
        leak = view_leak(item, sq, stated)
        if leak:
            bump(f"dropped_leak_{leak}")
            continue
        views.append((sq, stated))
    bump("items_with_views", bool(views))
    bump("views", len(views))
    bump("leaf_views", sum(is_local_leaf(sq) for sq, _ in views))
    if mode == "plain":
        bump("withheld_lines", sum(len(x) for _, x in views))
    elif not ctx_view:
        bump("stated_lines", sum(len(x) for _, x in views))
    else:                                           # TFLP / MIX: counted per view below
        for k in ("stated_lines", "withheld_lines", "local_views", "statements_views", "statements_leaf_views"):
            bump(k, 0)
        bump("statements_no_parents", len(no_par))
    w = (node_weight or aux_total / max(1, len(views))) * item.get("weight", 1.0)
    ws = [w] * len(views)
    if depth_weight and views:                      # same total per problem, split by depth
        ws = [x * item.get("weight", 1.0) for x in depth_split(node_weight * len(views) if node_weight else aux_total,
                                                               [depth[sq["id"]] for sq, _ in views], depth_weight)]
    out = []
    for (sq, stated), w in zip(views, ws):
        if mode == "plain" and stated:              # TFLN: TFL's line-order draw, nothing shown
            rng.shuffle([None] * len(stated))
            stated = []
        if ctx_view and (sq["id"] in no_par or (is_local_leaf(sq) if view == "local+ctxparse" else
                                                mix_statements(mix_seed, mix_epoch, item["item_id"], sq["id"], mix_p))):
            rng.shuffle([None] * len(stated))       # TFLP / MIX statements view: TFL's line-order draw, nothing shown
            ctx = block
            bump("statements_views")
            bump("statements_leaf_views", is_local_leaf(sq))
            bump("withheld_lines", len(stated))
        else:
            ctx = local_context(item, sq, stated, depth=depth, rng=rng,
                                order="depth" if view == "ancestors" else "shuffle")
            if ctx_view:
                bump("local_views")
                bump("stated_lines", len(stated))
        yes_first = rng.random() < 0.5
        a, b = ("Yes", "No") if yes_first else ("No", "Yes")
        out.append(Example(SUBQ_TEMPLATE.format(problem=ctx, question=sq["question"], a=a, b=b), ["A", "B"],
                           _soft(_node_p(sq, subq_target), yes_first), w, "subq", item["item_id"]))
    return out


SUBQ_MC_TEMPLATE = "{problem}\n\nIntermediate question: Which of these questions has the answer Yes?\nOptions:\n{opts}\nAnswer:"
MC_NONE = "None of them"


def _mc_capacity(n_yes: int, n_no: int, m: int, k: int) -> int:
    """How many disjoint m-question views (at most one Yes each) the confident questions allow, up to k."""
    c = 0
    while c < k:
        if n_yes >= 1 and n_no >= m - 1:
            n_yes, n_no = n_yes - 1, n_no - (m - 1)
        elif n_no >= m:
            n_no -= m
        else:
            break
        c += 1
    return c


def mc_pool(item: dict, kind: str) -> tuple[list[dict], list[dict], str]:
    pool = item.get("subqs", []) if kind == "cot" else item.get("random_subqs", [])
    key = "p_cot" if kind == "cot" else "p_fresh"
    return ([sq for sq in pool if sq.get(key) is not None and sq[key] >= 0.7],
            [sq for sq in pool if sq.get(key) is not None and sq[key] <= 0.3], key)


def mc_limit(item: dict, k: int, m: int = 3) -> int:
    """Views per item for the matched multiple-choice arms: the smaller of the two pools' capacities, so the CoT and
    the control arm give every item the same number of weighted views."""
    caps = []
    for kind in ("cot", "random"):
        y, n, _ = mc_pool(item, kind)
        caps.append(_mc_capacity(len(y), len(n), m, k))
    return min(caps)


def build_mc_views(item: dict, *, k: int, rng: random.Random, kind: str = "cot", m: int = 3, weight: float = 0.5,
                   p_one: float = 0.75) -> list[Example]:
    """Multiple-choice auxiliary views: up to k views, each packing m of the item's yes/no questions (CoT nodes, or
    the control questions with their fresh targets) as "Which of these questions has the answer Yes?" with the
    questions as options plus a final "None of them". Only confident questions are used (p <= 0.3 or >= 0.7), at most
    one confident Yes per view: with probability p_one a view gets one Yes and m-1 No questions, otherwise m No
    questions (answer "None of them"). Views use disjoint questions. The soft target puts P(i) proportional to
    p_i * prod_{j != i} (1 - p_j) and P(none) to prod_j (1 - p_j), renormalised over these m + 1 events."""
    yes, no, key = mc_pool(item, kind)
    k = min(k, _mc_capacity(len(yes), len(no), m, k))
    rng.shuffle(yes)
    rng.shuffle(no)
    out = []
    for made in range(k):
        n_yes = 1 if (yes and rng.random() < p_one) else 0
        left = k - made - 1                          # keep the remaining views feasible
        if n_yes == 0 and (len(no) < m or _mc_capacity(len(yes), len(no) - m, m, left) < left):
            n_yes = 1
        if n_yes == 1 and (not yes or len(no) < m - 1 or _mc_capacity(len(yes) - 1, len(no) - m + 1, m, left) < left):
            n_yes = 0
        if len(no) < m - n_yes:
            break
        chosen = [yes.pop() for _ in range(n_yes)] + [no.pop() for _ in range(m - n_yes)]
        rng.shuffle(chosen)
        ps = [min(max(sq[key], 1e-4), 1 - 1e-4) for sq in chosen]
        none = math.prod(1 - p for p in ps)
        scores = [p * none / (1 - p) for p in ps] + [none]
        z = sum(scores)
        letters = [chr(65 + i) for i in range(m + 1)]
        opts = "\n".join(f"{L}) {t}" for L, t in zip(letters, [sq["question"] for sq in chosen] + [MC_NONE]))
        out.append(Example(SUBQ_MC_TEMPLATE.format(problem=item["prompt"], opts=opts), letters, [x / z for x in scores],
                           weight * item.get("weight", 1.0), "subq", item["item_id"]))
    return out


def build_group(item: dict, *, aux_kind: str, k: int, aux_weight: float, rng: random.Random, final: str = "teacher",
                subq_target: str = "cot", permute_final: float = 0.0, keep_prob: float = 1.0,
                aux_active: bool = True, aux_answer: str = "all", subq_filter: str = "all") -> list[Example]:
    """One problem for exposure-matched grouped training: [final] + exactly k auxiliary views. aux_kind "cot" uses
    CoT nodes, "random" the matched control questions (fresh targets), "placebo" the same CoT views with zero loss
    weight (same forward/backward work, no auxiliary signal), "tree" CoT nodes asked with their parents' results
    stated, "tree_shuf" the same with unrelated nodes' results stated (edge-rewiring control). Missing views are
    zero-weight copies of the final. aux_active=False keeps the views but gives them zero weight (placebo for this
    item), so per-family auxiliary supervision stays exposure-matched. "cot_randlabel": the "cot" views with each
    target redrawn uniformly (yes/no) per view and epoch (format-only control). aux_answer "exclude"/"only" keeps only
    the CoT nodes that are not / are answer components (K&K "Is X a knight?"; treeplan.answer_component). subq_filter
    "path"/"mirror" asks only that chain's nodes (decision chains; for tree/tree_shuf the stated results may still
    come from any node)."""
    if aux_answer != "all":
        item = dict(item, subqs=[sq for sq in item.get("subqs", []) if answer_component(sq, item) == (aux_answer == "only")])
    if subq_filter != "all" and aux_kind == "cot_mc":
        item = dict(item, subqs=[sq for sq in item.get("subqs", []) if subq_keep(sq, subq_filter)])
    if aux_kind in ("cot_mc", "random_mc"):
        fin = [e for e in build_examples(item, final=final, subq="none", subq_target=subq_target, lambda_sub=0.0,
                                         rng=rng, permute_final=permute_final) if e.kind == "final"]
        if not fin:
            return []
        aux = build_mc_views(item, k=mc_limit(item, k), rng=rng, kind="cot" if aux_kind == "cot_mc" else "random",
                             weight=aux_weight if aux_active else 0.0)
        return fin + aux + [dataclasses.replace(fin[0], weight=0.0) for _ in range(k - len(aux))]
    if aux_kind in ("tree", "tree_shuf"):
        fin = [e for e in build_examples(item, final=final, subq="none", subq_target=subq_target, lambda_sub=0.0,
                                         rng=rng, permute_final=permute_final) if e.kind == "final"]
        if not fin:
            return []
        aux = build_transition_views(item, k=k, rng=rng, shuffled=aux_kind == "tree_shuf", keep_prob=keep_prob,
                                     weight=aux_weight if aux_active else 0.0, subq_target=subq_target,
                                     subq_filter=subq_filter)
        return fin + aux + [dataclasses.replace(fin[0], weight=0.0) for _ in range(k - len(aux))]
    sub, tgt = ("random", "fresh") if aux_kind == "random" else ("cot", subq_target)
    views = build_examples(item, final=final, subq=sub, subq_target=tgt, lambda_sub=aux_weight, rng=rng,
                           permute_final=permute_final, subq_k=k, subq_weight="each", subq_filter=subq_filter)
    fin = [e for e in views if e.kind == "final"]
    if not fin:
        return []
    aux = [e for e in views if e.kind == "subq"][:k]
    if aux_kind == "cot_randlabel":
        aux = [dataclasses.replace(e, target=[1.0, 0.0] if rng.random() < 0.5 else [0.0, 1.0]) for e in aux]
    if aux_kind == "placebo" or not aux_active:
        aux = [dataclasses.replace(e, weight=0.0) for e in aux]
    return fin + aux + [dataclasses.replace(fin[0], weight=0.0) for _ in range(k - len(aux))]


FACTS_HEADER = "Known intermediate results:"


def fact_block(item: dict, min_depth: int = 0) -> str:
    """The item's tree nodes as stated results ("Q: ... A: Yes/No"), shallow to deep, keeping nodes of depth >=
    min_depth. The answer is the program truth when known, else the teacher's CoT-conditioned answer; nodes the
    teacher was unsure about (0.3 < p < 0.7) are left out. Returns "" when nothing is kept."""
    nodes = [sq for sq in item.get("subqs", []) if "id" in sq]
    if not nodes:
        return ""
    d = node_depths(nodes)
    lines = []
    for sq in sorted(nodes, key=lambda x: (d[x["id"]], x["id"])):
        if d[sq["id"]] < min_depth:
            continue
        p = (1.0 if sq["truth"] else 0.0) if sq.get("truth") is not None else sq.get("p_cot")
        if p is None or 0.3 < p < 0.7:
            continue
        lines.append(f"Q: {sq['question']} A: {'Yes' if p >= 0.5 else 'No'}")
    return f"\n\n{FACTS_HEADER}\n" + "\n".join(lines) if lines else ""


def build_fact_examples(item: dict, *, stage: int, n_stages: int, final: str = "teacher") -> list[Example]:
    """Fact-internalization curriculum: the final question with the item's tree nodes stated in the prompt. Stage s
    of n_stages drops the nodes shallower than ceil(s (M+1) / n_stages) (shallow facts are internalized first);
    stage n_stages (or any stage that drops everything) gives the plain final question. n_stages=0 keeps all."""
    nodes = [sq for sq in item.get("subqs", []) if "id" in sq]
    top = max(node_depths(nodes).values()) if nodes else 0
    cut = 0 if n_stages == 0 else math.ceil(stage * (top + 1) / n_stages)
    ex = build_examples(dict(item, prompt=item["prompt"] + fact_block(item, cut)), final=final, subq="none",
                        subq_target="cot", lambda_sub=0.0, rng=random.Random(0))
    return [e for e in ex if e.kind == "final"]


# ---- stepwise internalisation (--si-stages; arms SI / SIY / SIS; Deng et al. 2024, ICoT-SI) ------------------------
# The final question with the problem's worked steps written between the problem and its options; stage k of S removes
# the first ceil(k n / S) of the n steps (stage S: none left = the test format). Nothing here draws from the training
# rng: the group is build_full_tree's, only the final view's text changes (si_group).
CHAIN_HEADER = "Worked steps:"
CHAIN_FIELD = "worked_steps"          # GSM8K: scripts/build_gsm8k_si.py; numgen: from the item's program (numgen_chain)
SI_CHAINS = {"true": "text", "wrong": "wrong_text", "wrong-results": "wrong_result_text"}
SI_REMOVE = ("front", "back")


def _ng_expr(cls: str, a: str, b: str | None, c) -> str:
    return {"ADD": f"{a} + {c}", "SUB": f"{a} − {c}", "MUL": f"{a} × {c}", "DIV": f"{a} ÷ {c}",
            "PCT": f"{c}% of {a}", "PCTOFF": f"{100 - c if c is not None else c}% of {a}", "SUM": f"{a} + {b}",
            "DOZEN": f"{c} × 12"}[cls]


def numgen_chain(item: dict) -> list[dict] | None:
    """The worked steps of a numgen item (data/student_numgen*, cotdistill.numgen) from its program: one line per
    derived quantity in program order (each operation reads the latest quantity, so the order is the evaluation order;
    the answer quantity last), "a op c = v" with the operation class's symbol (numgen.OPCLS: + − × ÷, "c% of a",
    PCTOFF as "(100 - c)% of a", SUM "a + b", DOZEN "c × 12"), each line checked against numgen.apply. The fields of
    scripts/build_gsm8k_si.py: text, value, a_from / b_from (the earlier line an input quantity is; a stated input is
    written as its number), wrong (the wrong probe of the quantity's derive node "<q>n" when it has one, else a
    numgen.perturb of the value from random.Random("si-wrong|<item>|<q>"); never the value, positive, and for a line
    whose value is the answer's never one of the options), wrong_text (results and linked operands written wrongly)
    and wrong_result_text (results only). None when the item has no program or no derived quantity."""
    from .numgen import OPCLS, apply, perturb
    prog = item.get("program")
    if not prog:
        return None
    qs = {q["id"]: q for q in prog["quantities"]}
    probes = {sq["id"][:-1]: sq.get("probe") for sq in item.get("subqs") or []
              if sq.get("type") == "derive" and str(sq.get("id", "")).endswith("n")}
    opts = item.get("options")
    banned = set(opts.values() if isinstance(opts, dict) else opts or [])
    lines, at = [], {}
    for q in prog["quantities"]:
        if not q.get("op") or q.get("stated"):
            continue
        cls = OPCLS[q["op"]]
        ins = [qs[i] for i in q["inputs"]]
        got = apply(cls, [x["value"] for x in ins], q["const"])
        if got is None or got != q["value"]:
            raise ValueError(f"{item.get('item_id')}: {q['id']} {q['op']} gives {got}, the program says {q['value']}")
        frm = [at.get(x["id"]) for x in ins] + [None] * (2 - len(ins))
        vals = [str(x["value"]) for x in ins] + [None] * (2 - len(ins))
        v = q["value"]
        rng = random.Random(f"si-wrong|{item.get('item_id')}|{q['id']}")
        w = probes.get(q["id"])
        bad = banned if v == qs[prog["answer"]]["value"] else set()        # the answer's value: never an option
        tries = 0
        while w is None or w == v or w <= 0 or w in bad:
            w = perturb(v, rng) if tries < 50 else v + tries - 49
            tries += 1
        lines.append({"id": q["id"], "op": q["op"], "cls": cls, "a": vals[0], "b": vals[1], "c": q["const"],
                      "value": str(v), "a_from": frm[0], "b_from": frm[1], "wrong": str(w),
                      "text": f"{_ng_expr(cls, vals[0], vals[1], q['const'])} = {v}"})
        at[q["id"]] = len(lines) - 1
    for ln in lines:
        wa = lines[ln["a_from"]]["wrong"] if ln["a_from"] is not None else ln["a"]
        wb = lines[ln["b_from"]]["wrong"] if ln["b_from"] is not None else ln["b"]
        ln["wrong_text"] = f"{_ng_expr(ln['cls'], wa, wb, ln['c'])} = {ln['wrong']}"
        ln["wrong_result_text"] = f"{_ng_expr(ln['cls'], ln['a'], ln['b'], ln['c'])} = {ln['wrong']}"
    return lines or None


def worked_steps(item: dict) -> list[dict] | None:
    """The item's worked steps: its CHAIN_FIELD (GSM8K rows of data/student_gsm8k_tf_si*, possibly null), else the
    chain of its numgen program (numgen_chain), else None (the final question is asked without a chain)."""
    if CHAIN_FIELD in item:
        return item[CHAIN_FIELD] or None
    if item.get("source") == "numgen":
        return numgen_chain(item)
    return None


def si_removed(n: int, stage: int, n_stages: int, coin: float, smooth: float) -> int:
    """Steps removed at stage (0..n_stages) of a chain of n: ceil(stage x n / n_stages), one more when coin < smooth
    (removal smoothing), at most n."""
    m = -(-stage * n // n_stages)
    return min(n, m + (coin < smooth))


def si_rng(seed: int, stage: int, epoch: int, item_id) -> random.Random:
    """The removal-smoothing coin of one (seed, stage, pass, item), seeded through a str key (sha512, as
    treeplan.scaffold_rng): it depends on nothing else (not the training rng, the order or the other items)."""
    return random.Random(f"si|{seed}|{stage}|{epoch}|{item_id}")


def chain_block(texts: list[str], first: int = 1) -> str:
    """'\\n\\nWorked steps:\\nStep <first>: ...\\nStep <first + 1>: ...', or '' without lines."""
    if not texts:
        return ""
    return f"\n\n{CHAIN_HEADER}\n" + "\n".join(f"Step {first + i}: {t}" for i, t in enumerate(texts))


def insert_chain(text: str, block: str) -> str:
    """text with block inserted before its (last) '\\nOptions:\\n' line, i.e. between the problem and its options."""
    if not block:
        return text
    head, sep, tail = text.rpartition("\nOptions:\n")
    if not sep:
        raise ValueError("no options block to write the worked steps before")
    return head + block + sep + tail


def si_view(steps: list[dict], *, stage: int, n_stages: int, chain: str, coin: float, smooth: float,
            remove: str = "front") -> tuple[str, int]:
    """(the chain block of this stage, steps removed). remove "front" drops the first steps (the steps kept keep their
    numbers: stage 1 of 4 on six steps shows Step 3 .. Step 6); "back" drops the last ones (Step 1 .. Step n - m)."""
    n = len(steps)
    m = si_removed(n, stage, n_stages, coin, smooth)
    texts = [s[SI_CHAINS[chain]] for s in steps]
    if remove == "front":
        return chain_block(texts[m:], m + 1), m
    return chain_block(texts[:n - m], 1), m


def si_group(group: list[Example], item: dict, *, stage: int, n_stages: int, chain: str = "true", seed: int = 0,
             epoch: int = 0, smooth: float = 0.1, final_only: bool = False, remove: str = "front",
             stats: dict | None = None) -> list[Example]:
    """A full-tree group (build_full_tree's [final] + node views) at SI stage `stage` of n_stages: the final view's text
    gets the item's worked steps of this stage (si_view; worked_steps) before its options, the node views are kept
    as they are (SIY: GFPQ's views) or dropped (final_only: SI / SIS). chain "true" writes the steps, "wrong" the SIS
    control (every computed value wrong: results and the operands that are earlier results), "wrong-results" (only
    the results wrong). The smoothing coin comes from si_rng, never the training rng, so the views, targets, option
    orders and weights stay build_full_tree's. stats counts si_items (with a chain), si_lines, si_shown, si_smoothed
    (an extra step removed), si_all_removed (no step left) and si_items_without_chain."""
    if chain not in SI_CHAINS:
        raise ValueError(f"--si-chain {chain!r}: one of {', '.join(SI_CHAINS)}")
    if remove not in SI_REMOVE:
        raise ValueError(f"--si-remove {remove!r}: one of {', '.join(SI_REMOVE)}")
    if not 0 <= stage <= n_stages:
        raise ValueError(f"SI stage {stage} of {n_stages}")
    steps = worked_steps(item)
    st = stats if stats is not None else {}
    out = []
    block, m = "", 0
    if steps:
        coin = si_rng(seed, stage, epoch, item["item_id"]).random()
        block, m = si_view(steps, stage=stage, n_stages=n_stages, chain=chain, coin=coin, smooth=smooth, remove=remove)
        st["si_items"] = st.get("si_items", 0) + 1
        st["si_lines"] = st.get("si_lines", 0) + len(steps)
        st["si_shown"] = st.get("si_shown", 0) + len(steps) - m
        st["si_smoothed"] = st.get("si_smoothed", 0) + (m > -(-stage * len(steps) // n_stages))
        st["si_all_removed"] = st.get("si_all_removed", 0) + (m == len(steps))
    else:
        st["si_items_without_chain"] = st.get("si_items_without_chain", 0) + 1
    for e in group:
        if e.kind == "final":
            out.append(dataclasses.replace(e, text=insert_chain(e.text, block)) if block else e)
        elif not final_only:
            out.append(e)
    return out


# ---- step curriculum (--curriculum-steps; arm GFPQC; opt-in, nothing below runs on the default path) ----------------
CURRICULUM_MODES = ("cumulative", "exact")


def curriculum_stage(step: int, n_stages: int) -> int:
    """The curriculum node stage (1..n_stages) of a node at solution step `step` (GSM8K: the Socratic step index, the
    node field "step", 1-based): its step, with every step beyond n_stages folded into the last node stage, so no node
    is left out of the curriculum."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError(f"curriculum: node step {step!r}, want an integer >= 1")
    return min(step, n_stages)


def curriculum_item(item: dict, *, stage: int, n_stages: int, mode: str = "cumulative") -> dict:
    """item for curriculum node stage `stage` (1..n_stages): a shallow copy whose subqs are only the tree nodes (with an
    "id") of that stage (curriculum_stage) or earlier (mode "cumulative") / of that stage only (mode "exact"); every
    other field is the item's. build_full_tree on it asks exactly those nodes in the arm's view, with the per-problem
    weight split over them (the --tree-node-types convention); the caller drops the final view (node stages train no
    final question). Nodes without a step raise ValueError."""
    if mode not in CURRICULUM_MODES:
        raise ValueError(f"--curriculum-mode {mode!r}: one of {', '.join(CURRICULUM_MODES)}")
    if not 1 <= stage <= n_stages:
        raise ValueError(f"curriculum node stage {stage} of {n_stages}")
    keep = []
    for sq in item.get("subqs") or []:
        if "id" not in sq:
            continue
        if "step" not in sq:
            raise ValueError(f"curriculum: {item.get('item_id')} node {sq['id']} has no step")
        c = curriculum_stage(sq["step"], n_stages)
        if c == stage or mode == "cumulative" and c < stage:
            keep.append(sq)
    return dict(item, subqs=keep)


def text_parts(model):
    """(backbone returning last_hidden_state, output embedding) for causal or multimodal wrappers."""
    head = model.get_output_embeddings()
    inner = getattr(model, "model", model)
    body = getattr(inner, "language_model", None) or getattr(model, "language_model", None)
    if body is None:
        body = model.get_decoder() if hasattr(model, "get_decoder") else inner
    return body, head


# ---- --tree-pack (opt-in; nothing below runs on the default path) ------------------------------------------------------
PACK_CUE = "\n\nIntermediate question: "


@dataclass
class Pack:
    """--tree-pack: several GFP-S node views of one problem read in ONE causal pass. parts[0] is the first view's own
    text (statement block + its question, options and "Answer:" cue, exactly its unpacked text); each further part is
    a further view's question chunk (PACK_CUE + question + options + "Answer:"), so a later question sees the block and
    the earlier QUESTIONS, never an answer. nodes[k] is view k as built for GFP-S (labels, target, weight, kind,
    item_id; its own text is not read): the readout of part k is the next-token distribution at the last token of
    part k (its "Answer:" cue), restricted to the option letters, as for the unpacked view. Parts are tokenized one by
    one and concatenated (pack_ids), so a cue's last token is the unpacked one (no merge with the next chunk)."""
    parts: list[str]
    nodes: list[Example]
    item_id: str
    kind: str = "pack"

    @property
    def text(self) -> str:
        return "".join(self.parts)


def readouts(batch: list) -> list:
    """The examples whose letter logits label_logits returns, in order: each Pack's node views in its place (batch
    itself when it has no Pack)."""
    if not any(isinstance(e, Pack) for e in batch):
        return batch
    return [n for e in batch for n in (e.nodes if isinstance(e, Pack) else [e])]


def pack_ids(tokenizer, p: Pack) -> tuple[list[int], list[int]]:
    """A Pack's token ids (the special prefix, then each part tokenized on its own) and its readout positions (the last
    token of every part)."""
    ids, pos = list(special_prefix(tokenizer)), []
    for part in p.parts:
        ids += text_ids(tokenizer, part)
        pos.append(len(ids) - 1)
    return ids, pos


def pack_rng(seed: int, epoch: int, item_id) -> random.Random:
    """--tree-pack: the node order of one problem's packs in one epoch, from its own generator (str key through sha512,
    as treeplan.scaffold_rng), never the training rng, so a packed run draws GFP-S's nodes, option orders and targets."""
    return random.Random(f"pack|{seed}|{epoch}|{item_id}")


def pack_views(group: list[Example], tokenizer, block: str, *, max_len: int, max_nodes: int, rng: random.Random,
               stats: dict | None = None) -> list:
    """--tree-pack: a GFP-S full-tree group ([final] + node views, each view = block + PACK_CUE + its question) with its
    node views packed. The views are shuffled by rng (pack_rng) and filled greedily into Packs of at most max_nodes
    views and max_len tokens (special prefix included); every Pack re-reads the block. A Pack of one view stays the
    plain view, and so does a view whose own text exceeds max_len (collate cuts it as in GFP-S). Weights and targets
    are the views' own, so the update loss sums the same per-node terms as GFP-S; the other examples (the final
    question) come first, unchanged. stats counts pack_seqs, pack_nodes, pack_single and pack_long."""
    head = block + PACK_CUE
    nodes = [e for e in group if e.kind == "subq"]
    out = [e for e in group if e.kind != "subq"]
    for e in nodes:
        if not e.text.startswith(head):
            raise ValueError(f"{e.item_id}: --tree-pack needs GFP-S views (statement block + question)")
    order = nodes[:]
    rng.shuffle(order)
    st = stats if stats is not None else {}
    bump = lambda k, n=1: st.__setitem__(k, st.get(k, 0) + n)
    for k in ("pack_seqs", "pack_nodes", "pack_single", "pack_long"):
        bump(k, 0)
    n_pre = len(special_prefix(tokenizer))
    cur: list[Example] = []
    size = 0

    def flush():
        if len(cur) == 1:
            out.append(cur[0])
            bump("pack_single")
        elif cur:
            out.append(Pack([cur[0].text] + [e.text[len(block):] for e in cur[1:]], list(cur), cur[0].item_id))
            bump("pack_seqs")
            bump("pack_nodes", len(cur))
    for e in order:
        if cur and len(cur) < max_nodes:
            n = len(text_ids(tokenizer, e.text[len(block):]))
            if size + n <= max_len:
                cur.append(e)
                size += n
                continue
        flush()
        cur, size = [e], n_pre + len(text_ids(tokenizer, e.text))
        if size > max_len:                  # too long even alone: the plain view, cut by collate as in GFP-S
            out.append(e)
            bump("pack_long")
            cur, size = [], 0
    flush()
    return out


def seq_len(tokenizer, e, max_len: int) -> int:
    """Token length of an example as collate builds it (a Pack is never cut: pack_views keeps it within max_len)."""
    if isinstance(e, Pack):
        return len(pack_ids(tokenizer, e)[0])
    return min(max_len, len(special_prefix(tokenizer)) + len(text_ids(tokenizer, e.text)))


def collate(tokenizer, batch: list[Example], max_len: int, device) -> dict:
    if any(isinstance(e, Pack) for e in batch):     # --tree-pack: rows/pos index every readout (readouts(batch))
        return _collate_packs(tokenizer, batch, max_len, device)
    enc = [encode_tail(tokenizer, e.text, max_len) for e in batch]  # keep the question end (and a BOS if any)
    L = max(len(x) for x in enc)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = torch.full((len(enc), L), pad, dtype=torch.long)
    mask = torch.zeros((len(enc), L), dtype=torch.long)
    for i, x in enumerate(enc):
        ids[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device),
            "last": torch.tensor([len(x) - 1 for x in enc], device=device)}


def _collate_packs(tokenizer, batch: list, max_len: int, device) -> dict:
    """collate for a batch with Packs: the same right-padded rows (plain examples cut as encode_tail cuts them), plus
    rows/pos, the row and position of every readout in readouts(batch) order."""
    enc, rows, pos = [], [], []
    for i, e in enumerate(batch):
        if isinstance(e, Pack):
            x, ps = pack_ids(tokenizer, e)
            if len(x) > max_len:
                raise ValueError(f"{e.item_id}: a pack of {len(x)} tokens exceeds max_len {max_len}")
        else:
            x = encode_tail(tokenizer, e.text, max_len)
            ps = [len(x) - 1]
        enc.append(x)
        rows += [i] * len(ps)
        pos += ps
    L = max(len(x) for x in enc)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = torch.full((len(enc), L), pad, dtype=torch.long)
    mask = torch.zeros((len(enc), L), dtype=torch.long)
    for i, x in enumerate(enc):
        ids[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device),
            "last": torch.tensor([len(x) - 1 for x in enc], device=device),
            "rows": torch.tensor(rows, device=device), "pos": torch.tensor(pos, device=device)}


def token_budget_batches(lengths: list[int], budget: int, max_seqs: int = 16) -> list[list[int]]:
    """Split one update's sequences (token lengths as collate sees them) into micro-batches of indices, longest first:
    a micro-batch grows while (n + 1) x its longest length <= budget padded tokens and n < max_seqs; a sequence
    longer than the budget goes alone. Every index appears once. The update's gradient does not depend on the split
    (sum over sequences of w x CE / norm), so this only changes the number of forward passes and the padding."""
    out: list[list[int]] = []
    for i in sorted(range(len(lengths)), key=lambda j: -lengths[j]):
        if out and len(out[-1]) < max_seqs and (len(out[-1]) + 1) * lengths[out[-1][0]] <= budget:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def micro_batches(tokenizer, batch: list[Example], max_len: int, micro_bs: int, token_budget: int = 0,
                  max_seqs: int = 16, padded: list | None = None) -> list[list[Example]]:
    """One update's micro-batches: consecutive chunks of micro_bs, or with token_budget the length-sorted split of
    token_budget_batches (lengths counted as collate truncates them; padded, if a list, receives each micro-batch's
    padded token count)."""
    if not token_budget:
        return [batch[i:i + micro_bs] for i in range(0, len(batch), micro_bs)]
    n_pre = len(special_prefix(tokenizer))
    lens = [seq_len(tokenizer, e, max_len) if isinstance(e, Pack) else
            min(max_len, n_pre + len(text_ids(tokenizer, e.text))) for e in batch]
    groups = token_budget_batches(lens, token_budget, max_seqs)
    if padded is not None:
        padded.extend(len(b) * max(lens[j] for j in b) for b in groups)
    return [[batch[j] for j in b] for b in groups]


def label_logits(model, tokenizer, batch: list[Example], max_len: int, label_ids_cache: dict,
                 hidden_out: list | None = None, layers: list[int] | None = None) -> list[torch.Tensor]:
    """Logits over each example's option letters at the readout position (optionally also appends each
    example's readout hidden state, detached, to hidden_out; with layers, a [len(layers) + 1, hidden] stack of the
    readout position's states after those decoder layers plus the final readout state). A Pack (--tree-pack) gives one
    entry per node view, in readouts(batch) order."""
    if getattr(model, "cotd_readout", False):    # FSDP2 root (cotdistill.fsdp.Readout): this function runs in its forward
        return model.readout(tokenizer, batch, max_len, label_ids_cache, hidden_out, layers)
    body, head = text_parts(model)
    dev = head.weight.device
    b = collate(tokenizer, batch, max_len, dev)
    want_layers = bool(layers) and hidden_out is not None
    res = body(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=want_layers,
               use_cache=False)
    hidden = res.last_hidden_state
    rows, last = torch.arange(hidden.shape[0], device=dev), b["last"]
    if "pos" in b:                               # --tree-pack: every node view's "Answer:" cue of each pack
        rows, last, batch = b["rows"], b["pos"], readouts(batch)
    h = hidden[rows, last]
    if hidden_out is not None:
        if want_layers:
            per = [res.hidden_states[l][rows, last] for l in layers] + [h]
            hidden_out.extend(torch.stack(per, 1).detach().float().cpu())
        else:
            hidden_out.extend(h.detach().float().cpu())
    out, cap = [], logit_softcap(model)
    for i, e in enumerate(batch):
        key = tuple(e.labels)
        if key not in label_ids_cache:
            label_ids_cache[key] = torch.tensor(letter_token_ids(tokenizer, e.labels), device=dev)
        rows = label_ids_cache[key]
        bias = head.bias[rows] if getattr(head, "bias", None) is not None else None
        out.append(head_logits(h[i], head.weight[rows], bias, cap).float())
    return out


def _ce_sum(h: torch.Tensor, w: torch.Tensor, y: torch.Tensor, softcap: float | None = None) -> torch.Tensor:
    z = (h.to(w.dtype) @ w.T).float()
    if softcap:                      # Gemma: the model's own (capped) next-token distribution
        z = softcap * torch.tanh(z / softcap)
    return F.cross_entropy(z, y, reduction="sum")


def lm_loss(model, tokenizer, batch: list[Example], max_len: int, chunk: int = 512) -> torch.Tensor:
    """Sum over examples of weight x mean token cross-entropy on each continuation.

    The vocabulary projection runs in checkpointed chunks so full-vocabulary logits (248k for Qwen3.5)
    are never held for the whole sequence. Continuations longer than the budget keep their start and
    their last 64 tokens (where the answer is)."""
    body, head = text_parts(model)
    dev = head.weight.device
    seqs, starts = [], []
    for e in batch:
        pre = encode_tail(tokenizer, e.text, max_len // 2)
        cont = tokenizer.encode(e.continuation, add_special_tokens=False)
        room = max_len - len(pre)
        if len(cont) > room:
            cont = cont[:room - 64] + cont[-64:]
        seqs.append(pre + cont)
        starts.append(len(pre))
    L = max(len(x) for x in seqs)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, x in enumerate(seqs):
        ids[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    ids, mask = ids.to(dev), mask.to(dev)
    hidden = body(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    total = hidden.new_zeros((), dtype=torch.float32)
    cap = logit_softcap(model)
    for i, (x, st) in enumerate(zip(seqs, starts)):
        h, y = hidden[i, st - 1:len(x) - 1], ids[i, st:len(x)]
        ce = sum(torch.utils.checkpoint.checkpoint(_ce_sum, h[j:j + chunk], head.weight, y[j:j + chunk], cap,
                                                   use_reentrant=False) for j in range(0, len(y), chunk))
        total = total + batch[i].weight * ce / max(1, len(y))
    return total


def kl_loss(logits: list[torch.Tensor], batch: list[Example], smoothing: float = 0.0) -> torch.Tensor:
    """Weighted soft cross-entropy (= KL(target || p) + const). smoothing mixes each target with the uniform
    distribution over its options: t' = (1 - smoothing) t + smoothing / K (label smoothing; the teacher's
    targets are near one-hot)."""
    tot = 0.0
    for z, e in zip(logits, batch):
        t = torch.tensor(e.target, device=z.device)
        if smoothing:
            t = (1 - smoothing) * t + smoothing / t.numel()
        tot = tot + e.weight * -(t * F.log_softmax(z, -1)).sum()
    return tot / len(batch)


def brier_loss(logits: list[torch.Tensor], batch: list[Example]) -> torch.Tensor:
    """Weighted Brier score of the readout against the soft target (a proper scoring rule, minimised at p = t)."""
    tot = 0.0
    for z, e in zip(logits, batch):
        t = torch.tensor(e.target, device=z.device)
        tot = tot + e.weight * ((F.softmax(z, -1) - t) ** 2).sum()
    return tot / len(batch)


def load_model(path: str, dtype, attn_implementation: str | None = None):
    import transformers
    kw = {"attn_implementation": attn_implementation} if attn_implementation else {}
    cfg = transformers.AutoConfig.from_pretrained(path)
    if getattr(cfg, "model_type", None) == "gemma4_unified" and (cfg.vision_config is not None or
                                                                   cfg.audio_config is not None):
        # Gemma 4 12B "Unified" is encoder-free: images and audio enter through small linear embedders
        # (model.embed_vision / model.embed_audio, 52.4M params) beside the text decoder. With both sub-configs
        # removed, Gemma4UnifiedForConditionalGeneration builds the text decoder + tied head only (11.91B params),
        # loads it with the same keys and logits, skips the embedder weights as unexpected keys, and saves a
        # text-only checkpoint (vision_config/audio_config null) that loads back through this same path.
        # (Gemma4UnifiedForCausalLM.from_pretrained on this checkpoint finds no weights under its key prefix
        # model.* and silently returns a randomly initialised model; do not use it.)
        cfg.vision_config = None
        cfg.audio_config = None
        kw["config"] = cfg
        print("gemma4_unified: text-only load (vision/audio embedders dropped)", flush=True)
    try:
        m = transformers.AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, **kw)
    except Exception:
        m = transformers.AutoModelForImageTextToText.from_pretrained(path, torch_dtype=dtype, **kw)
    for n, p in m.named_parameters():
        if "visual" in n or "vision" in n:
            p.requires_grad = False
    return m


@torch.no_grad()
def evaluate(model, tok, items: list[dict], max_len: int, bs: int, cache: dict,
             hidden: list | None = None, layers: list[int] | None = None) -> tuple[dict, list[dict]]:
    """Final-question predictions for eval records (prompt, labels, gold_label; optional gold_probs, group).

    Items are batched by length to limit padding; predictions come back in input order. If `hidden` is a list,
    it receives each item's readout hidden state (float32, CPU), also in input order. The question text is
    final_text's: FINAL_TEMPLATE, with the latent slot line when set_slot_prompt registered one for tok."""
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    order = sorted(range(len(items)), key=lambda i: len(items[i]["prompt"]))
    probs_by, hid_by = {}, {}
    if getattr(model, "cotd_readout", False):    # FSDP2 root: the same batches, split across the ranks and gathered
        exs = [Example(final_text(tok, it["prompt"]), it["labels"], [0.0] * len(it["labels"]), 1.0,
                       "final", it["item_id"]) for it in items]
        zs, hs = model.batched(tok, exs, order, bs, max_len, cache, hidden is not None, layers)
        for i, z in enumerate(zs):
            probs_by[i] = torch.softmax(z, -1).tolist()
            if hidden is not None:
                hid_by[i] = hs[i]
    else:
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            batch = [Example(final_text(tok, items[i]["prompt"]), items[i]["labels"],
                             [0.0] * len(items[i]["labels"]), 1.0, "final", items[i]["item_id"]) for i in idx]
            hs = [] if hidden is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                zs = label_logits(model, tok, batch, max_len, cache, hidden_out=hs, layers=layers)
            for j, (i, z) in enumerate(zip(idx, zs)):
                probs_by[i] = torch.softmax(z, -1).tolist()
                if hs is not None:
                    hid_by[i] = hs[j]
    if hidden is not None:
        hidden.extend(hid_by[i] for i in range(len(items)))
    probs, gold, preds = [], [], []
    for i, it in enumerate(items):
        p = probs_by[i]
        preds.append({"item_id": it["item_id"], "group": it.get("group"), "labels": it["labels"], "probs": p,
                      "gold_label": it.get("gold_label"), "gold_probs": it.get("gold_probs")})
        if it.get("gold_label") in it["labels"]:
            probs.append(p)
            gold.append(it["labels"].index(it["gold_label"]))
    if was_training:
        model.train()
    return (calibration(probs, gold) if gold else {"n": 0}), preds


def subq_sets(spec: str) -> list[tuple[str, str]]:
    """--eval-subq value -> [(result name, path)]: a bare path is scored as "subq"; "name=path" entries (comma
    separated) as "subq_<name>"."""
    out = []
    for part in (x.strip() for x in spec.split(",")):
        if part:
            name, _, path = part.rpartition("=")
            out.append((f"subq_{name}" if name else "subq", path))
    return out


@torch.no_grad()
def evaluate_subq(model, tok, items: list[dict], max_len: int, bs: int, cache: dict,
                  seed: int = 0, value_render: str | None = "op") -> tuple[dict, list[dict]]:
    """Student accuracy on the sub-questions of held-out items (the mechanism check: does the one-pass
    student answer the intermediate judgements?). Target: program truth when known, else the teacher's
    CoT-conditioned answer. Also scores the matched control questions against the teacher's fresh answer.
    Broken down by node type and source; yes/no order randomised per question as in training.
    Multiple-choice nodes (treeplan.is_mc_node) are asked with their own options (mc_node_example, no rng draw) and
    scored by argmax == answer; their preds carry probs, pred, answer and rank instead of p_yes (p_yes is P(answer)),
    and they also count in the group cot/mc.
    Value nodes (item field VALUE_FIELD, scripts/build_gsm8k_ops.py) in the rendering value_render (None: not scored)
    are asked like MC nodes in the same context (the prompt with its options; no rng draw, so the yes/no draws are
    those of the items without value nodes), after the item's other questions, kind "value": groups all_value,
    value/<source>, value/type/value_<render>, value/mc and value/depth/<node_depth> (nodes with a node_depth);
    preds also carry node_id and render. Nodes without that rendering are skipped. value_render vt / vtr scores the
    verification nodes of VERIFY_FIELD instead (scripts/build_gsm8k_vt.py; vtr against the control answers, so
    train_student scores vt for the vtr arm), with the extra groups value/vt_kind/<gold|step|given|none>."""
    rng = random.Random(seed)
    exs, meta = [], []
    for it in items:
        for kind, pool in (("cot", it.get("subqs", [])), ("control", it.get("random_subqs", []))):
            for sq in pool:
                if kind == "cot" and is_mc_node(sq):           # own options, as a final question; no rng draw
                    exs.append(mc_node_example(it["prompt"], sq, 1.0, it["item_id"]))
                    meta.append({"item_id": it["item_id"], "source": it.get("source"), "kind": kind,
                                 "type": sq.get("type") or kind, "truth_known": True, "target": sq["answer"],
                                 "mc": len(sq["options"]), "question": sq["question"], "var": sq.get("var")})
                    continue
                if kind == "cot":
                    tgt = sq["truth"] if sq.get("truth") is not None else (
                        None if sq.get("p_cot") is None else sq["p_cot"] > 0.5)
                else:
                    tgt = None if sq.get("p_fresh") is None else sq["p_fresh"] > 0.5
                if tgt is None:
                    continue
                yes_first = rng.random() < 0.5
                a, b = ("Yes", "No") if yes_first else ("No", "Yes")
                exs.append(Example(SUBQ_TEMPLATE.format(problem=it["prompt"], question=sq["question"], a=a, b=b),
                                   ["A", "B"], [0.5, 0.5], 1.0, "subq", it["item_id"]))
                meta.append({"item_id": it["item_id"], "source": it.get("source"), "kind": kind,
                             "type": sq.get("type") or kind, "truth_known": sq.get("truth") is not None,
                             "target": bool(tgt), "yes_first": yes_first, "question": sq["question"],
                             "var": sq.get("var")})
                if kind == "cot" and "on_path" in sq and "node_depth" in sq:      # decision chains: chain x level,
                    tag = "path" if sq["on_path"] else "mirror"                  # chain x levels below the top,
                    meta[-1]["chain"] = f"{tag}/d{sq['node_depth']}"             # path nodes x item depth
                    if "depth" in it:
                        meta[-1]["chain_top"] = f"{tag}/top-{it['depth'] - sq['node_depth']}"
                        meta[-1]["chain_item"] = f"{tag}/item_d{it['depth']}"
        for sq in (it.get(value_field(value_render)) or []) if value_render else []:   # value nodes: own options
            mc = value_mc_node(sq, value_render)
            if mc is None:
                continue
            exs.append(mc_node_example(it["prompt"], mc, 1.0, it["item_id"]))
            meta.append({"item_id": it["item_id"], "source": it.get("source"), "kind": "value", "type": mc["type"],
                         "truth_known": True, "target": mc["answer"], "mc": len(mc["options"]),
                         "question": mc["question"], "var": None, "node_id": sq["id"], "render": value_render,
                         "depth": None if sq.get("node_depth") is None else f"value/depth/{sq.get('node_depth')}"})
            if sq.get("vt_kind"):
                meta[-1]["vt_kind"] = sq["vt_kind"]
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    order = sorted(range(len(exs)), key=lambda i: len(exs[i].text))
    p_yes = [0.0] * len(exs)
    mc_probs: dict[int, list[float]] = {}

    def keep(i: int, z) -> None:
        if "mc" in meta[i]:
            mc_probs[i] = torch.softmax(z.float(), -1).tolist()
            p_yes[i] = mc_probs[i][meta[i]["target"]]
            return
        pa = torch.softmax(z, -1)[0].item()
        p_yes[i] = pa if meta[i]["yes_first"] else 1 - pa
    if getattr(model, "cotd_readout", False):    # FSDP2 root: the same batches, split across the ranks and gathered
        for i, z in enumerate(model.batched(tok, exs, order, bs, max_len, cache)[0]):
            keep(i, z)
    else:
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                zs = label_logits(model, tok, [exs[i] for i in idx], max_len, cache)
            for i, z in zip(idx, zs):
                keep(i, z)
    if was_training:
        model.train()
    groups: dict[str, list[int]] = {}
    preds = []
    for i, (m, py) in enumerate(zip(meta, p_yes)):
        if "mc" in m:
            pr = mc_probs[i]
            pred = max(range(len(pr)), key=pr.__getitem__)           # the first maximum (the trainer's argmax)
            ok = int(pred == m["target"])
            rank = 1 + sum(1 for j, x in enumerate(pr) if x > pr[m["target"]] or (x == pr[m["target"]]
                                                                                 and j < m["target"]))
            preds.append(dict(m, p_yes=py, probs=pr, pred=pred, answer=m["target"], rank=rank, correct=ok))
            groups.setdefault("cot/mc" if m["kind"] != "value" else "value/mc", []).append(ok)
        else:
            ok = int((py > 0.5) == m["target"])
            preds.append(dict(m, p_yes=py, correct=ok))
        for g in ("all_" + m["kind"], f"{m['kind']}/{m['source']}", f"{m['kind']}/type/{m['type']}"):
            groups.setdefault(g, []).append(ok)
        if m["kind"] == "cot" and m["truth_known"] and "mc" not in m:
            groups.setdefault("cot/program_truth", []).append(ok)
        for g in ("chain", "chain_top", "chain_item"):
            if g in m:
                groups.setdefault(f"cot/{m[g]}", []).append(ok)
        if m["kind"] == "value" and m["depth"] is not None:
            groups.setdefault(m["depth"], []).append(ok)
        if m.get("vt_kind"):
            groups.setdefault(f"value/vt_kind/{m['vt_kind']}", []).append(ok)
    metrics = {g: {"n": len(v), "acc": sum(v) / len(v)} for g, v in sorted(groups.items())}
    return metrics, preds

def local_eval_examples(items: list[dict], *, view: str = "local", seed: int = 0) -> tuple[list[Example], list[dict]]:
    """The held-out local-step probe: every node of the items that carry local fields (K&K in data/student_v3sl),
    asked in its local view with its TRUE parents' results (view "local") or all ancestors' results ("ancestors"),
    no problem text; target = program truth. Leaf (parse) nodes show their statement line(s). Views that would state
    the final answer (view_leak) are skipped, as in training."""
    rng = random.Random(seed)
    exs, meta = [], []
    for it in items:
        if it.get("local_preamble") is None:
            continue
        nodes = [sq for sq in it.get("subqs", []) if "id" in sq]
        by = {sq["id"]: sq for sq in nodes}
        depth = ({i: by[i]["node_depth"] for i in by} if all("node_depth" in sq for sq in nodes) else node_depths(nodes))
        anc = ancestors(nodes) if view == "ancestors" else None
        for sq in nodes:
            tgt = sq["truth"] if sq.get("truth") is not None else (None if sq.get("p_cot") is None else sq["p_cot"] > 0.5)
            if tgt is None:
                continue
            if is_local_leaf(sq):
                stated = []
            else:
                par = [x for x in (sq.get("depends_on") or []) if x in by and x != sq["id"]]
                ids = sorted(anc[sq["id"]] - {sq["id"]}) if view == "ancestors" else par
                stated = [by[j] for j in ids]
                if not par or not all(_fact_line(x) for x in stated) or view_leak(it, sq, stated):
                    continue
            ctx = local_context(it, sq, stated, depth=depth, rng=rng, order="depth" if view == "ancestors" else "shuffle")
            yes_first = rng.random() < 0.5
            a, b = ("Yes", "No") if yes_first else ("No", "Yes")
            exs.append(Example(SUBQ_TEMPLATE.format(problem=ctx, question=sq["question"], a=a, b=b), ["A", "B"],
                               [0.5, 0.5], 1.0, "subq", it["item_id"]))
            meta.append({"item_id": it["item_id"], "id": sq["id"], "type": sq.get("type") or "node",
                         "depth": depth[sq["id"]], "n_stated": len(stated), "target": bool(tgt),
                         "trivial": None if is_local_leaf(sq) else sq.get("local_trivial"),
                         "yes_first": yes_first, "question": sq["question"]})
    return exs, meta


def _depth_bin(d: int) -> str:
    for lo, hi in ((0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16)):
        if lo <= d <= hi:
            return f"{lo}" if lo == hi else f"{lo}-{hi}"
    return "17+"


@torch.no_grad()
def evaluate_local(model, tok, items: list[dict], max_len: int, bs: int, cache: dict, *, view: str = "local",
                   seed: int = 0) -> tuple[dict, list[dict]]:
    """Accuracy on local_eval_examples (does the student apply each local operator from its parents' results alone?),
    overall and by node type, exact depth and depth bin, and for derived nodes split into those their own hypothesis
    decides (derived_trivial, data field local_trivial) and those that need the stated results (derived_nontrivial,
    the operator test proper). Yes/no order randomised per question."""
    exs, meta = local_eval_examples(items, view=view, seed=seed)
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    order = sorted(range(len(exs)), key=lambda i: len(exs[i].text))
    p_yes = [0.0] * len(exs)
    if getattr(model, "cotd_readout", False):    # FSDP2 root: the same batches, split across the ranks and gathered
        for i, z in enumerate(model.batched(tok, exs, order, bs, max_len, cache)[0]):
            pa = torch.softmax(z, -1)[0].item()
            p_yes[i] = pa if meta[i]["yes_first"] else 1 - pa
    else:
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                zs = label_logits(model, tok, [exs[i] for i in idx], max_len, cache)
            for i, z in zip(idx, zs):
                pa = torch.softmax(z, -1)[0].item()
                p_yes[i] = pa if meta[i]["yes_first"] else 1 - pa
    if was_training:
        model.train()
    groups: dict[str, list[int]] = {}
    preds = []
    for m, py in zip(meta, p_yes):
        ok = int((py > 0.5) == m["target"])
        preds.append(dict(m, p_yes=py, correct=ok))
        gs = ["all", f"type/{m['type']}", f"depth/{m['depth']}", f"depth_bin/{_depth_bin(m['depth'])}",
              f"type_depth_bin/{m['type']}/{_depth_bin(m['depth'])}"]
        if m["trivial"] is not None:     # derived nodes: decided by their own hypothesis, or needing stated results
            gs.append("derived_trivial" if m["trivial"] else "derived_nontrivial")
            if not m["trivial"]:
                gs.append(f"nontrivial_depth_bin/{_depth_bin(m['depth'])}")
        for g in gs:
            groups.setdefault(g, []).append(ok)
    metrics = {g: {"n": len(v), "acc": sum(v) / len(v)} for g, v in sorted(groups.items())}
    return metrics, preds


# ---- latent node slots (--latent-slots; arms LS / LSU / LSR / LSY; opt-in, nothing below runs on the default path) ---
# Tree Forcing v2, design (a) of notes/latent_primer_zh_2026-10-19.md: K x c existing reserved tokens in the final
# question's prompt; at training only, the final-layer states of slot j are fed as the first input embeddings of a short
# separate sequence through the same network and head, which must continue with the j-th worked step ("4 + 2 = 6").
# The deployed model is the same network read out the same way in one forward pass over the prompt with the slots.
# v2.1 (arms LSL / LSLU / LSLR / LSLY; --slot-readout lens, all opt-in): the slot state is supervised at the output of an
# intermediate decoder layer L (--slot-layer; later layers, the answer position's included, read it), through the
# model's own final norm + LM head ("lens": no second body pass, no new parameters), and the target is the step's
# VALUE only ("6"): position i of a slot of c tokens predicts the value's token i, then SLOT_END. Slots after a shorter
# chain can get the last value (--slot-fill), and every slot can have its own token (--slot-tokens distinct).
SLOT_TOKENS = ("<|box_start|>", "<|quad_start|>", "<|object_ref_start|>", "<|box_end|>", "<|quad_end|>",
               "<|object_ref_end|>")      # Qwen3.5's reserved grounding tokens: single ids, no special processing
# --slot-tokens distinct (auto): one token per slot from this pool, the swap token the first pool token left over.
# Qwen3.5 ids 248047-248052 (the grounding tokens) and 248060-248065 (code-completion tokens of the base model). Avoided:
# 248044-248046 (<|endoftext|> = eos / pad, chat turns), 248053-248057 (vision start / end / pad, image and video pads:
# config.vision_start_token_id / image_token_id / video_token_id, read by get_rope_index and the multimodal merge),
# 248058-248059 / 248066-248069 (tool calls, <think>: the chat template's), 248070-248076 (audio / tts).
SLOT_POOL = SLOT_TOKENS + ("<|fim_pad|>", "<|repo_name|>", "<|file_sep|>", "<|fim_prefix|>", "<|fim_middle|>",
                           "<|fim_suffix|>")
SLOT_PLACES = ("before-options", "after-options")
SLOT_TARGETS = ("tree", "none", "shuffled")
SLOT_OVERFLOW = ("merge", "truncate")
SLOT_READOUTS = ("decode", "lens", "tsf")  # decode: v2 (separate decoding pass, step text); lens: v2.1 (layer L, value);
                                          # tsf: teacher-state slots with forced reading (teacher_forward)
SLOT_FILLS = ("none", "last", "answer")
SLOT_TOKEN_MODES = ("same", "distinct")
SLOT_MERGE_SEP = "; "
SLOT_END = "\n"                           # closes every decoded step (the greedy readout stops there); the lens pad
FINAL_CUE = FINAL_TEMPLATE.replace("{problem}", "", 1)      # "\n\nAnswer:"


@dataclass(frozen=True)
class SlotSpec:
    """The slot line of a final question: k slots of c copies each of token (or, tokens non-empty, of tokens[j] for
    slot j: --slot-tokens distinct), placed before the options block or at the end of the prompt (place). layer: the
    decoder layer (0-based) whose OUTPUT holds the supervised slot states (None: the final, normed states, v2);
    readout / fill: how they are supervised (SLOT_READOUTS / SLOT_FILLS); head_grad: the lens slot loss also trains the
    LM head (= the tied input embedding on 2B / 4B; default off: the lens is a fixed readout, its gradient reaches the
    body and the slot states only). pad_weight (--slot-pad-weight, lens only): the weight of a slot's SLOT_END padding
    positions in its lens CE relative to the value's own tokens (1: the plain mean over the c positions; 0: the value's
    tokens only), the CE renormalised over the weights. overflow: the --slot-overflow of training (recorded so that
    eval_student / probe_slots rebuild the same targets). as_dict leaves the v2.1 fields out at their defaults, so a v2
    spec (and its latent_slots.json) is unchanged, and so is a v2.1 spec at pad_weight 1 and overflow merge."""
    k: int
    c: int
    token: str
    place: str = "before-options"
    tokens: tuple = ()
    layer: int | None = None
    readout: str = "decode"
    fill: str = "none"
    head_grad: bool = False
    pad_weight: float = 1.0
    overflow: str = "merge"

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        for key, default in (("tokens", ()), ("layer", None), ("readout", "decode"), ("fill", "none"),
                             ("head_grad", False), ("pad_weight", 1.0), ("overflow", "merge")):
            if d[key] == default:
                d.pop(key)
        if "tokens" in d:
            d["tokens"] = list(d["tokens"])
        return d

    def slot_tokens(self) -> tuple:
        """The token of every slot (k entries)."""
        return tuple(self.tokens) if self.tokens else (self.token,) * self.k


@dataclass
class SlotExample(Example):
    """A final view with latent slots (a subclass, so Example and every existing arm's tuples are unchanged).
    slot_targets[j] is slot j's step text or None (no loss); each supervised slot's decoding loss (mean token CE of
    its text + SLOT_END) weighs weight x slot_lambda, slot_lambda = lambda / supervised slots of the problem (the
    per-problem convention of the node views), so a weight-0 copy (FSDP pad) has no slot loss either."""
    slot_targets: tuple = ()
    slot_lambda: float = 0.0


def slot_token_id(tokenizer, token: str) -> int:
    """The id of a slot token: it must be one token on its own and in a run (token x 3 -> 3 ids) and not the pad,
    eos, bos or unk token."""
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"slot token {token!r} is {len(ids)} tokens, not one")
    if tokenizer.encode(token * 3, add_special_tokens=False) != ids * 3:
        raise ValueError(f"slot token {token!r}: a run of three does not encode to three copies of its id")
    bad = {tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.unk_token_id} - {None}
    if ids[0] in bad:
        raise ValueError(f"slot token {token!r} is the tokenizer's pad / eos / bos / unk token")
    return ids[0]


def choose_slot_token(tokenizer, requested: str = "auto", avoid: tuple = (), pool: tuple = SLOT_TOKENS) -> str:
    """requested itself (checked by slot_token_id), or for "auto" the first token of pool (SLOT_TOKENS; the swap token
    of distinct slot tokens: SLOT_POOL) that is a valid slot token for this tokenizer and not in avoid. A tokenizer
    without any (GPT-2) needs an explicit token."""
    if requested != "auto":
        slot_token_id(tokenizer, requested)
        return requested
    for t in pool:
        if t in avoid:
            continue
        try:
            slot_token_id(tokenizer, t)
            return t
        except ValueError:
            continue
    raise ValueError(f"no reserved token of {SLOT_TOKENS} is a single token of this tokenizer: pass a token "
                     "(e.g. --slot-token §)")


def choose_slot_tokens(tokenizer, k: int, requested: str = "auto", avoid: tuple = ()) -> tuple:
    """--slot-tokens distinct: k different slot tokens, requested as a comma list of k tokens (each checked by
    slot_token_id) or, for "auto", the first k valid tokens of SLOT_POOL not in avoid."""
    if requested != "auto":
        toks = tuple(t for t in requested.split(",") if t)
        if len(toks) != k or len(set(toks)) != k:
            raise ValueError(f"--slot-tokens distinct: --slot-token needs {k} different comma-separated tokens, got "
                             f"{requested!r}")
        for t in toks:
            slot_token_id(tokenizer, t)
        return toks
    out = []
    for t in SLOT_POOL:
        if t in avoid:
            continue
        try:
            slot_token_id(tokenizer, t)
        except ValueError:
            continue
        out.append(t)
        if len(out) == k:
            return tuple(out)
    raise ValueError(f"--slot-tokens distinct: {len(out)} usable tokens of {SLOT_POOL} for {k} slots: pass --slot-token "
                     "t1,t2,...")


def slot_line(spec: SlotSpec) -> str:
    """The slot line: "\\n" + k x c tokens (slot j's c copies of its token, j = 1..k)."""
    if spec.tokens:
        return "\n" + "".join(t * spec.c for t in spec.tokens)
    return "\n" + spec.token * (spec.k * spec.c)


def slot_prompt(prompt: str, spec: SlotSpec) -> str:
    """prompt with the slot line ("\\n" + k x c tokens): before-options puts it before the last "\\nOptions:\\n" line
    (between the problem and its options; a prompt without that block, e.g. BBEH's inline layouts, gets it at the
    end); after-options at the end of the prompt (after the options, right before the "Answer:" cue)."""
    line = slot_line(spec)
    if spec.place == "before-options":
        head, sep, tail = prompt.rpartition("\nOptions:\n")
        if sep:
            return head + line + sep + tail
    elif spec.place != "after-options":
        raise ValueError(f"--slot-place {spec.place!r}: one of {', '.join(SLOT_PLACES)}")
    return prompt + line


def slot_text(text: str, spec: SlotSpec) -> str:
    """A final question text (FINAL_TEMPLATE: prompt + FINAL_CUE) with the slot line in its prompt."""
    if not text.endswith(FINAL_CUE):
        raise ValueError("latent slots: a final question text must end with the 'Answer:' cue")
    return slot_prompt(text[:-len(FINAL_CUE)], spec) + FINAL_CUE


_SLOTS: dict[int, tuple] = {}


def set_slot_prompt(tokenizer, spec: SlotSpec | None) -> None:
    """Register (spec) or clear (None) the slot line of every final question evaluate builds with this tokenizer
    (final_text), and the slot id slot_forward / slot_decode look for. Keyed by the tokenizer object (kept), as
    set_chat_render. With distinct tokens (spec.tokens) every token is checked and the slot line must encode to slot
    j's id c times for j = 1..k (no merge between neighbouring tokens)."""
    if spec is None:
        _SLOTS.pop(id(tokenizer), None)
        return
    if spec.tokens:
        if len(spec.tokens) != spec.k or len(set(spec.tokens)) != spec.k:
            raise ValueError(f"latent slots: {spec.k} slots need {spec.k} different tokens, got {spec.tokens}")
        ids = tuple(slot_token_id(tokenizer, t) for t in spec.tokens)
        want = [i for i in ids for _ in range(spec.c)]
        got = tokenizer.encode(slot_line(spec)[1:], add_special_tokens=False)
        if got != want:
            raise ValueError(f"latent slots: the slot line {slot_line(spec)[1:]!r} encodes to {got}, not {want}")
        _SLOTS[id(tokenizer)] = (tokenizer, spec, ids[0], ids)
        return
    sid = slot_token_id(tokenizer, spec.token)
    _SLOTS[id(tokenizer)] = (tokenizer, spec, sid, (sid,) * spec.k)


def slot_spec(tokenizer) -> SlotSpec | None:
    hit = _SLOTS.get(id(tokenizer))
    return hit[1] if hit is not None and hit[0] is tokenizer else None


def _slot_reg(tokenizer) -> tuple[SlotSpec, int]:
    hit = _SLOTS.get(id(tokenizer))
    if hit is None or hit[0] is not tokenizer:
        raise ValueError("latent slots: no slot prompt registered for this tokenizer (set_slot_prompt)")
    return hit[1], hit[2]


def _slot_ids(tokenizer) -> tuple:
    """The id of every slot's token (k entries; all equal for a single slot token)."""
    _slot_reg(tokenizer)
    return _SLOTS[id(tokenizer)][3]


def final_text(tokenizer, prompt: str) -> str:
    """The final question text of an eval record: FINAL_TEMPLATE, with the registered slot line (set_slot_prompt) if
    any, i.e. exactly the training final view of a slot arm (slot_group)."""
    text = FINAL_TEMPLATE.format(problem=prompt)
    spec = slot_spec(tokenizer)
    return slot_text(text, spec) if spec is not None else text


def slot_targets(item: dict, k: int, overflow: str = "merge") -> list[str | None]:
    """Slot j's target (j = 0..k-1): the text of the item's j-th worked step (worked_steps: the atomic chain of
    data/student_gsm8k_tf_si*, "4 + 2 = 6", the answer's operation last; numgen: the program's chain), None for the
    slots after a shorter chain (no loss) and every slot of an item without a chain. A chain longer than k: overflow
    "merge" joins steps k..n into the last slot (SLOT_MERGE_SEP, so the answer-producing step stays in), "truncate"
    keeps the first k steps."""
    if overflow not in SLOT_OVERFLOW:
        raise ValueError(f"--slot-overflow {overflow!r}: one of {', '.join(SLOT_OVERFLOW)}")
    steps = worked_steps(item) or []
    texts = [s["text"] for s in steps]
    if len(texts) > k:
        texts = texts[:k - 1] + [SLOT_MERGE_SEP.join(texts[k - 1:])] if overflow == "merge" else texts[:k]
    return texts + [None] * (k - len(texts))


def slot_values(item: dict) -> list[str]:
    """The result of every worked step of the item (its "value" field, else what follows the step text's last '='),
    in chain order (the answer's operation last); [] without a chain."""
    out = []
    for s in worked_steps(item) or []:
        v = s.get("value")
        v = str(v).strip() if v is not None else _value(s.get("text"))
        if not v:
            raise ValueError(f"{item.get('item_id')}: worked step {s.get('id')} has no value")
        out.append(v)
    return out


def answer_text(item: dict) -> str | None:
    """The gold option's text (options as a list in label order or a dict by label), else None."""
    opts, g = item.get("options"), item.get("gold_label")
    if isinstance(opts, dict):
        v = opts.get(g)
    elif isinstance(opts, list) and g in item.get("labels", []) and len(opts) == len(item["labels"]):
        v = opts[item["labels"].index(g)]
    else:
        v = None
    return str(v).strip() if v is not None and str(v).strip() else None


def slot_value_targets(item: dict, k: int, overflow: str = "merge", fill: str = "none") -> list[str | None]:
    """v2.1 (--slot-readout lens): slot j's target is the VALUE of the item's j-th worked step ("6" for "4 + 2 = 6").
    A chain longer than k: overflow "merge" keeps steps 1..k-1 and puts the LAST step's value (the answer's) on slot
    k (values cannot be joined, so the merge's intent, the answer-producing step stays in, is kept); "truncate" keeps
    the first k. Slots after a shorter chain: fill "none" = no target, "last" = the chain's last value (the answer's
    step), "answer" = the gold option's text. An item without a chain has no target in every mode."""
    if overflow not in SLOT_OVERFLOW:
        raise ValueError(f"--slot-overflow {overflow!r}: one of {', '.join(SLOT_OVERFLOW)}")
    if fill not in SLOT_FILLS:
        raise ValueError(f"--slot-fill {fill!r}: one of {', '.join(SLOT_FILLS)}")
    vals = slot_values(item)
    if not vals:
        return [None] * k
    if len(vals) > k:
        vals = vals[:k - 1] + [vals[-1]] if overflow == "merge" else vals[:k]
    pad = {"none": None, "last": vals[-1], "answer": answer_text(item)}[fill]
    return vals + [pad] * (k - len(vals))


def spec_slot_targets(item: dict, spec: SlotSpec, overflow: str = "merge") -> list[str | None]:
    """The real slot targets of a spec: worked-step texts (decode) or values (lens, spec.fill)."""
    if spec.readout == "lens":
        return slot_value_targets(item, spec.k, overflow, spec.fill)
    return slot_targets(item, spec.k, overflow)


def shuffled_value_targets(items: list[dict], *, k: int, overflow: str = "merge", fill: str = "none",
                           seed: int = 0) -> dict:
    """LSLR, the wrong-content control of the lens arms: shuffled_slot_targets on the value targets (same supervised
    slots and count as LSL, fill included; every value drawn from other items' value targets, never one of the
    item's own values; keyed by seed and the item, fixed over the epochs)."""
    return _shuffle_slot_targets({it["item_id"]: slot_value_targets(it, k, overflow, fill) for it in items}, seed)


def slot_end_id(tokenizer) -> int:
    ids = tokenizer.encode(SLOT_END, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"latent slots: SLOT_END {SLOT_END!r} is {len(ids)} tokens, not one")
    return ids[0]


def lens_target_ids(tokenizer, value: str, c: int) -> tuple[list[int], bool]:
    """The lens targets of one slot (c positions): the value's tokens (encoded alone, no leading space: Qwen3.5 writes
    every digit and the decimal point as its own token, "20" -> "2", "0"), then SLOT_END up to c; a value of more than
    c tokens keeps its first c (fit False: it can never be read out exactly)."""
    ids = tokenizer.encode(value, add_special_tokens=False)
    end = slot_end_id(tokenizer)
    return (ids + [end] * (c - len(ids)))[:c], len(ids) <= c


def lens_position_weights(n_value: int, c: int, pad_weight: float) -> list[float]:
    """--slot-pad-weight W: the weights of one slot's c lens positions, 1 for each of the value's own tokens (the first
    min(n_value, c)) and W for each SLOT_END padding position after them, normalised to sum 1 (W = 1: 1/c each, the
    plain mean). Positions with no value token at all (the weight-0 FSDP stand-in, n_value 0) and W = 0: uniform 1/c,
    so the normaliser is never 0."""
    n = min(max(n_value, 0), c)
    raw = [1.0] * n + [float(pad_weight)] * (c - n)
    tot = sum(raw)
    return [x / tot for x in raw] if tot > 0 else [1.0 / c] * c


def shuffled_slot_targets(items: list[dict], *, k: int, overflow: str = "merge", seed: int = 0) -> dict:
    """LSR, the wrong-content control: item_id -> slot targets with the same supervised slots as slot_targets (same
    count and positions) whose texts are drawn from the slot targets of OTHER items (never one of the item's own
    texts), by a generator keyed by seed and the item (not the training rng; fixed over the epochs). The pool is every
    supervised target of items (in file order), so lengths and merged lines follow the real distribution."""
    return _shuffle_slot_targets({it["item_id"]: slot_targets(it, k, overflow) for it in items}, seed)


def _shuffle_slot_targets(real: dict, seed: int) -> dict:
    """shuffled_slot_targets on given real targets (item_id -> list, None = no loss): same supervised slots, each
    target drawn from the pool of every other item's targets, never one of the item's own."""
    pool = [(iid, t) for iid, ts in real.items() for t in ts if t is not None]
    out = {}
    for iid, ts in real.items():
        own = {t for t in ts if t is not None}
        foreign = [i for i, (src, t) in enumerate(pool) if src != iid and t not in own] if own else []
        if own and not foreign:
            raise ValueError(f"--slot-target shuffled: no foreign slot target for {iid}")
        rng = random.Random(f"slot-shuffle|{seed}|{iid}")
        out[iid] = [pool[foreign[rng.randrange(len(foreign))]][1] if t is not None else None for t in ts]
    return out


def slot_group(group: list[Example], item: dict, *, spec: SlotSpec, targets: list | None, lam: float = 1.0,
               final_only: bool = False, stats: dict | None = None, teacher=None) -> list[Example]:
    """A full-tree group (build_full_tree's [final] + node views) of a slot arm: the final view's text gets the slot line
    (slot_text) and becomes a SlotExample with targets (LS: slot_targets, LSR: shuffled_slot_targets) and slot_lambda =
    lam / supervised slots; targets None (LSU, the pause-token control) keeps a plain Example (no decoding pass). Node
    views stay as built (LSY: GFPQ's, without slots) or are dropped (final_only: LS / LSU / LSR). Draws nothing from any
    generator (rng lockstep with the arm's GFPQ). stats counts slot_items, slot_supervised (slots with a target),
    slot_items_without_chain and slot_items_merged (an overflow target). teacher (TSF, --slot-readout tsf: a
    TeacherPlan, the item's teacher view): the final view becomes a TeacherExample whose teacher_text / teacher_marks
    come from teacher_text_marks (targets: the plan's line per supervised slot, plan_slot_targets); default None: as
    before."""
    st = stats if stats is not None else {}
    bump = lambda key, n=1: st.__setitem__(key, st.get(key, 0) + n)
    out = []
    for e in group:
        if e.kind == "final":
            text = slot_text(e.text, spec)
            bump("slot_items")
            if targets is None:
                out.append(dataclasses.replace(e, text=text))
                continue
            tg = tuple(targets)
            if len(tg) != spec.k:
                raise ValueError(f"{item.get('item_id')}: {len(tg)} slot targets for {spec.k} slots")
            m = sum(t is not None for t in tg)
            bump("slot_supervised", m)
            bump("slot_items_without_chain", m == 0)
            bump("slot_items_merged", any(t is not None and SLOT_MERGE_SEP in t for t in tg))
            if teacher is not None:          # TSF: the teacher view of the same final view (no slot line)
                t_text, marks = teacher_text_marks(e.text, teacher)
                if len(marks) != spec.k or [mk is not None for mk in marks] != [t is not None for t in tg]:
                    raise ValueError(f"{item.get('item_id')}: the teacher plan's slots differ from the slot targets")
                bump("teacher_views", bool(t_text))
                out.append(TeacherExample(**dict(dataclasses.asdict(e), text=text), slot_targets=tg,
                                          slot_lambda=lam / m if m else 0.0, teacher_text=t_text,
                                          teacher_marks=marks))
                continue
            out.append(SlotExample(**dict(dataclasses.asdict(e), text=text), slot_targets=tg,
                                   slot_lambda=lam / m if m else 0.0))
        elif not final_only:
            out.append(e)
    return out


def _letters(model, tokenizer, head, h: torch.Tensor, batch: list, cache: dict) -> list[torch.Tensor]:
    """label_logits' letter readout of the readout states h (one row per example)."""
    out, cap, dev = [], logit_softcap(model), head.weight.device
    for i, e in enumerate(batch):
        key = tuple(e.labels)
        if key not in cache:
            cache[key] = torch.tensor(letter_token_ids(tokenizer, e.labels), device=dev)
        rows = cache[key]
        bias = head.bias[rows] if getattr(head, "bias", None) is not None else None
        out.append(head_logits(h[i], head.weight[rows], bias, cap).float())
    return out


def _slot_positions(ids: torch.Tensor, sid, spec: SlotSpec, item_id) -> torch.Tensor:
    """The k x c slot positions of one encoded row, slot by slot. sid: the slot id, or (distinct tokens) the tuple of
    the k slot ids, whose positions must come as slot 1's c copies, slot 2's, ..."""
    if isinstance(sid, tuple) and len(set(sid)) > 1:
        ref = torch.tensor(sid, device=ids.device)
        pos = torch.isin(ids, ref).nonzero().flatten()
        want = ref.repeat_interleave(spec.c)
        if pos.numel() != want.numel() or not bool((ids[pos] == want).all()):
            raise ValueError(f"{item_id}: the slot tokens in the encoded prompt are not the slot line's {spec.k} x "
                             f"{spec.c} (a slot token occurs in the text, or the prompt was cut)")
        return pos
    if isinstance(sid, tuple):
        sid = sid[0]
    pos = (ids == sid).nonzero().flatten()
    if pos.numel() != spec.k * spec.c:
        raise ValueError(f"{item_id}: {pos.numel()} slot tokens in the encoded prompt, want {spec.k} x {spec.c} (the "
                         "token occurs in the text, or the prompt was cut)")
    return pos


def _slot_mask(ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """Boolean mask of the slot positions of a batch of input ids."""
    sids = sorted(set(_slot_ids(tokenizer)))
    if len(sids) == 1:
        return ids == sids[0]
    return torch.isin(ids, torch.tensor(sids, device=ids.device))


def decoder_count(model) -> int:
    from .fsdp import decoder_layers
    root = model.model if getattr(model, "cotd_readout", False) else model
    return len(decoder_layers(text_parts(root)[0]))


def resolve_slot_layer(model, value) -> tuple[int | None, dict]:
    """--slot-layer -> (decoder layer index, info). "final" / "" / None: None (the final, normed states; v2). An int
    (negative from the end) is a 0-based decoder layer; a fraction f in (0, 1) ("0.67", "2/3") is layer round(f x n) - 1,
    i.e. the output after round(f x n) of the n layers: 0.67 -> 20 of 32 on Qwen3.5-4B / 9B (Gated DeltaNet layers 0-2,
    4-6, ..., full attention every 4th: 3, 7, ..., 31; after layer 20 come 11 layers, 3 of them full attention: 23, 27,
    31, which attend straight to the slot states), 15 of 24 on Qwen3.5-2B. Must be an intermediate layer (0..n-2):
    the final layer's slot states are read by no later position. info: n_layers, the layer and the full-attention
    layers after it (config.layer_types; every layer when the config has none)."""
    if value in (None, "", "final"):
        return None, {}
    n = decoder_count(model)
    s = str(value).strip()
    if "/" in s or "." in s:
        a, _, b = s.partition("/")
        f = float(a) / float(b) if b else float(a)
        if not 0 < f < 1:
            raise ValueError(f"--slot-layer {value}: a fraction in (0, 1)")
        li = int(round(f * n)) - 1
    else:
        li = int(s)
        li = li + n if li < 0 else li
    if not 0 <= li <= n - 2:
        raise ValueError(f"--slot-layer {value}: an intermediate decoder layer, 0..{n - 2} (of {n})")
    root = model.model if getattr(model, "cotd_readout", False) else model
    cfg = root.config.get_text_config() if hasattr(root.config, "get_text_config") else root.config
    types = list(getattr(cfg, "layer_types", None) or ["full_attention"] * n)[:n]
    full = [i for i in range(li + 1, n) if types[i] == "full_attention"]
    return li, {"n_layers": n, "layer": li, "layers_after": n - 1 - li, "full_attention_after": full}


class _SlotCapture:
    """The output of decoder layer li (0-based) at every position, NOT detached: a one-shot forward pre-hook on layer
    li + 1 keeps its hidden-state input (the output of layer li; with reentrant or non-reentrant gradient checkpointing
    the input of a checkpointed layer is the graph tensor itself, and the recompute in backward does not overwrite the
    capture). A context manager over one forward."""

    def __init__(self, body, li: int):
        from .fsdp import decoder_layers
        layers = decoder_layers(body)
        if not 0 <= li <= len(layers) - 2:
            raise ValueError(f"slot layer {li}: an intermediate decoder layer, 0..{len(layers) - 2}")
        self.nxt, self.h, self.handle = layers[li + 1], None, None

    def __enter__(self):
        def pre(mod, args, kwargs):
            if self.h is None:
                self.h = args[0] if args else kwargs["hidden_states"]
        self.handle = self.nxt.register_forward_pre_hook(pre, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        self.handle.remove()
        return False


def _final_norm(body):
    norm = getattr(body, "norm", None)
    if norm is None:
        raise ValueError("--slot-readout lens: the text decoder has no final norm (body.norm)")
    return norm


def _ce_rows(h: torch.Tensor, w: torch.Tensor, y: torch.Tensor, softcap: float | None = None) -> torch.Tensor:
    z = (h.to(w.dtype) @ w.T).float()
    if softcap:
        z = softcap * torch.tanh(z / softcap)
    return F.cross_entropy(z, y, reduction="none")


def _decode_inputs(model, tokenizer, vecs: list[torch.Tensor], conts: list[list[int]]):
    """The separate decoding sequences: [special prefix] + slot states (c vectors) + continuation tokens, as input
    embeddings of the same network (its own embedding module, so a scaled embedding stays scaled), right-padded."""
    emb = model.get_input_embeddings()
    pre = special_prefix(tokenizer)
    n, c, P = len(vecs), vecs[0].shape[0], len(pre)
    L = P + c + max(len(x) for x in conts)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    dev = emb.weight.device
    ids = torch.full((n, L), pad, dtype=torch.long, device=dev)
    mask = torch.zeros((n, L), dtype=torch.long, device=dev)
    for r, x in enumerate(conts):
        if P:
            ids[r, :P] = torch.tensor(pre, device=dev)
        if x:
            ids[r, P + c:P + c + len(x)] = torch.tensor(x, device=dev)
        mask[r, :P + c + len(x)] = 1
    e = emb(ids)
    v = torch.stack(vecs).to(e.dtype)
    return torch.cat([e[:, :P], v, e[:, P + c:]], dim=1), mask, P + c


def slot_forward(model, tokenizer, batch: list, max_len: int, cache: dict, _detach: bool = False,
                 _always: bool = False):
    """One training micro-batch of a slot arm: (letter logits as label_logits gives them, the summed slot decoding
    loss, stats). One main pass over the batch (collate); for every SlotExample, slot j's final-layer states (its c
    positions, found by the slot id) are NOT detached and start a separate sequence followed by the target text +
    SLOT_END, read by the same body and head (next-token CE on the text tokens, mean per slot), so the decoding loss
    reaches the main pass through those states. Loss = sum over supervised slots of weight x slot_lambda x CE. Examples
    without targets (node views, LSU) only give letter logits. Under FSDP (Readout root) both passes run inside the
    root's forward, and a micro-batch without a supervised slot still runs a decoding pass (one sequence at weight 0),
    so every rank makes the same FSDP collectives. stats: slot_n (supervised slots), slot_ce_sum (their unweighted
    CEs). _detach (tests only) cuts the slot states from the main pass."""
    if getattr(model, "cotd_readout", False):
        return model.run(slot_forward, tokenizer, batch, max_len, cache, _detach, True)
    spec, _ = _slot_reg(tokenizer)
    if spec.readout == "tsf":
        raise ValueError("--slot-readout tsf: the training pass is teacher_forward, not slot_forward")
    if spec.readout == "lens":
        return _slot_forward_lens(model, tokenizer, batch, max_len, cache, _detach, _always)
    sid = _slot_ids(tokenizer)
    body, head = text_parts(model)
    dev = head.weight.device
    b = collate(tokenizer, batch, max_len, dev)
    hidden = body(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).last_hidden_state
    rows = torch.arange(hidden.shape[0], device=dev)
    out = _letters(model, tokenizer, head, hidden[rows, b["last"]], batch, cache)
    vecs, conts, wts = [], [], []
    for i, e in enumerate(batch):
        tg = getattr(e, "slot_targets", ())
        if not any(t is not None for t in tg):
            continue
        pos = _slot_positions(b["input_ids"][i], sid, spec, e.item_id)
        for j, t in enumerate(tg):
            if t is None:
                continue
            v = hidden[i, pos[j * spec.c:(j + 1) * spec.c]]
            vecs.append(v.detach() if _detach else v)
            conts.append(tokenizer.encode(t + SLOT_END, add_special_tokens=False))
            wts.append(e.weight * e.slot_lambda)
    zero = hidden.new_zeros((), dtype=torch.float32)
    n_real = len(conts)
    if not vecs and not _always:
        return out, zero, {"slot_n": 0, "slot_ce_sum": 0.0}
    if not vecs:                             # FSDP: a weight-0 decoding sequence keeps the ranks' collectives equal
        vecs, conts, wts = [hidden[0, :spec.c]], [[tokenizer.eos_token_id if tokenizer.eos_token_id is not None
                                                    else 0] * 2], [0.0]
    embeds, mask, s0 = _decode_inputs(model, tokenizer, vecs, [x[:-1] for x in conts])
    dh = body(inputs_embeds=embeds, attention_mask=mask, use_cache=False).last_hidden_state
    hs = torch.cat([dh[r, s0 - 1:s0 - 1 + len(x)] for r, x in enumerate(conts)])
    ys = torch.tensor([t for x in conts for t in x], device=dev)
    wrow = torch.tensor([w / len(x) for w, x in zip(wts, conts) for _ in x], device=dev, dtype=torch.float32)
    cap = logit_softcap(model)
    ce = torch.cat([torch.utils.checkpoint.checkpoint(_ce_rows, hs[j:j + 512], head.weight, ys[j:j + 512], cap,
                                                      use_reentrant=False) for j in range(0, len(ys), 512)])
    with torch.no_grad():
        seg = torch.repeat_interleave(torch.arange(len(conts), device=dev),
                                      torch.tensor([len(x) for x in conts], device=dev))
        per = torch.zeros(len(conts), device=dev).index_add_(0, seg, ce.detach().float())
        per = per / torch.tensor([len(x) for x in conts], device=dev, dtype=torch.float32)
    return out, zero + (wrow * ce).sum(), {"slot_n": n_real, "slot_ce_sum": float(per[:n_real].sum())}


def _slot_forward_lens(model, tokenizer, batch: list, max_len: int, cache: dict, _detach: bool = False,
                       _always: bool = False):
    """slot_forward for --slot-readout lens (v2.1, arms LSL / LSLR / LSLY): ONE pass over the micro-batch (collate)
    with the output of decoder layer spec.layer captured (_SlotCapture, not detached), the letter logits as
    label_logits gives them, and for every supervised slot j of a SlotExample its c layer-L states through the model's
    own final norm and LM head (soft-capped like the model's logits): position i predicts token i of
    lens_target_ids (the value's tokens, then SLOT_END), the slot's CE = the mean over its c positions. Loss = sum over
    supervised slots of weight x slot_lambda x CE (slot_lambda = lambda / supervised slots of the problem; with
    spec.pad_weight != 1 the slot's CE is the weighted mean of lens_position_weights: value tokens 1, pads W). No second
    pass, no new parameter. The head weight is read detached (spec.head_grad False, the default): the slot loss trains
    the body (and the final norm), never the LM head / tied embedding; the letter CE reads it as before. Under FSDP (_always) a micro-batch without a supervised slot adds the lens CE of c states
    at weight 0 (the same graph on every rank). stats: slot_n, slot_ce_sum as slot_forward's."""
    spec, _ = _slot_reg(tokenizer)
    if spec.layer is None:
        raise ValueError("--slot-readout lens needs --slot-layer (an intermediate decoder layer)")
    sid = _slot_ids(tokenizer)
    body, head = text_parts(model)
    dev = head.weight.device
    b = collate(tokenizer, batch, max_len, dev)
    with _SlotCapture(body, spec.layer) as cap:
        hidden = body(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).last_hidden_state
    hl = cap.h
    rows = torch.arange(hidden.shape[0], device=dev)
    out = _letters(model, tokenizer, head, hidden[rows, b["last"]], batch, cache)
    vecs, ys, wts, nv = [], [], [], []
    for i, e in enumerate(batch):
        tg = getattr(e, "slot_targets", ())
        if not any(t is not None for t in tg):
            continue
        pos = _slot_positions(b["input_ids"][i], sid, spec, e.item_id)
        for j, t in enumerate(tg):
            if t is None:
                continue
            v = hl[i, pos[j * spec.c:(j + 1) * spec.c]]
            vecs.append(v.detach() if _detach else v)
            ys.append(lens_target_ids(tokenizer, t, spec.c)[0])
            wts.append(e.weight * e.slot_lambda)
            if spec.pad_weight != 1.0:       # --slot-pad-weight: the value's own token count
                nv.append(len(tokenizer.encode(t, add_special_tokens=False)))
    zero = hidden.new_zeros((), dtype=torch.float32)
    n_real = len(vecs)
    if not vecs and not _always:
        return out, zero, {"slot_n": 0, "slot_ce_sum": 0.0}
    if not vecs:                             # FSDP: a weight-0 lens term keeps the ranks' graphs equal
        vecs, ys, wts, nv = [hl[0, :spec.c]], [[slot_end_id(tokenizer)] * spec.c], [0.0], [0]
    hs = _final_norm(body)(torch.cat(vecs))
    yt = torch.tensor([t for y in ys for t in y], device=dev)
    if spec.pad_weight == 1.0:               # the plain mean over the c positions (v2.1 as before)
        wrow = torch.tensor([w / spec.c for w in wts for _ in range(spec.c)], device=dev, dtype=torch.float32)
        pw = None
    else:                                    # value tokens 1, SLOT_END pads W, renormalised per slot
        pw = [lens_position_weights(n, spec.c, spec.pad_weight) for n in nv]
        wrow = torch.tensor([w * q for w, ps in zip(wts, pw) for q in ps], device=dev, dtype=torch.float32)
    cap_ = logit_softcap(model)
    w = head.weight if spec.head_grad else head.weight.detach()          # the lens: a fixed readout by default
    ce = torch.cat([torch.utils.checkpoint.checkpoint(_ce_rows, hs[j:j + 512], w, yt[j:j + 512], cap_,
                                                      use_reentrant=False) for j in range(0, len(yt), 512)])
    with torch.no_grad():
        if pw is None:
            per = ce.detach().float().view(-1, spec.c).mean(-1)
        else:                                # the slot CE as trained: the weighted mean over its positions
            per = (ce.detach().float().view(-1, spec.c) * torch.tensor(pw, device=dev, dtype=torch.float32)).sum(-1)
    return out, zero + (wrow * ce).sum(), {"slot_n": n_real, "slot_ce_sum": float(per[:n_real].sum())}


def slot_lens_readout(model, tokenizer, items: list[dict], *, max_len: int, bs: int = 8,
                      overflow: str = "merge") -> list[dict]:
    """The lens readout (v2.1): for each item, one pass over its final question with the registered slot line
    (final_text); slot j's c states at the output of decoder layer spec.layer through the final norm + LM head,
    argmax per position, read up to the first SLOT_END. Rows: item_id, n_steps and per slot target (the real value
    target, spec_slot_targets: fill included; None = no target), read, exact (read == target), fit (the target's tokens
    fit the c positions), tok_ok (positions equal to lens_target_ids). Under FSDP every rank reads the same items
    inside the root's forward."""
    if getattr(model, "cotd_readout", False):
        return model.run(_slot_lens_readout, tokenizer, items, max_len, bs, overflow)
    return _slot_lens_readout(model, tokenizer, items, max_len, bs, overflow)


@torch.no_grad()
def _slot_lens_readout(model, tokenizer, items, max_len, bs, overflow) -> list[dict]:
    spec, _ = _slot_reg(tokenizer)
    if spec.layer is None:
        raise ValueError("lens readout: the slot spec has no layer")
    sid = _slot_ids(tokenizer)
    end = slot_end_id(tokenizer)
    was_training = model.training
    model.eval()
    body, head = text_parts(model)
    dev = head.weight.device
    norm, cap_ = _final_norm(body), logit_softcap(model)
    bias = head.bias if getattr(head, "bias", None) is not None else None
    rows = []
    for s in range(0, len(items), bs):
        its = items[s:s + bs]
        exs = [Example(final_text(tokenizer, it["prompt"]), it["labels"], [0.0] * len(it["labels"]), 1.0, "final",
                       it["item_id"]) for it in its]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            b = collate(tokenizer, exs, max_len, dev)
            with _SlotCapture(body, spec.layer) as cap:
                body(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False)
            pos = torch.stack([_slot_positions(b["input_ids"][i], sid, spec, e.item_id) for i, e in enumerate(exs)])
            h = cap.h[torch.arange(len(exs), device=dev)[:, None], pos]            # [n, k x c, d]
            pred = head_logits(norm(h), head.weight, bias, cap_).float().argmax(-1).view(len(exs), spec.k, spec.c)
        for i, it in enumerate(its):
            tg = spec_slot_targets(it, spec, overflow)
            slots = []
            for j in range(spec.k):
                p = pred[i, j].tolist()
                d = tokenizer.decode(p[:p.index(end)] if end in p else p).strip()
                t = tg[j]
                ids, fit = lens_target_ids(tokenizer, t, spec.c) if t is not None else (None, None)
                slots.append({"slot": j + 1, "target": t, "read": d, "exact": None if t is None else d == t,
                              "fit": fit, "tok_ok": None if t is None else sum(a == b for a, b in zip(p, ids)) / spec.c})
                if spec.pad_weight != 1.0:   # pads at W != 1: SLOT_END after the value is not (fully) trained, so the
                    n = min(len(tokenizer.encode(t, add_special_tokens=False)), spec.c) if t is not None else 0
                    slots[-1].update(value_read=tokenizer.decode(p[:n]).strip() if t is not None else None,  # value's
                                     value_ok=None if t is None else bool(fit) and p[:n] == ids[:n],         # own n
                                     first_ok=None if t is None else p[0] == ids[0])                         # positions
            rows.append({"item_id": it["item_id"], "n_steps": len(worked_steps(it) or []), "slots": slots})
    if was_training:
        model.train()
    return rows


def slot_readout_summary(rows: list[dict]) -> dict:
    """Exact value match (value), position accuracy (token) and the share of targets that fit c positions (fit) of
    slot_lens_readout rows, overall and per slot (slots with a target only); rows of a pad_weight != 1 spec add
    value_prefix (the value's own positions all right, whatever follows) and first (its first token right)."""
    out: dict = {}
    for key in ["all"] + sorted({f"slot{s['slot']}" for r in rows for s in r["slots"]}, key=lambda x: int(x[4:])):
        xs = [s for r in rows for s in r["slots"] if s["target"] is not None and (key == "all" or f"slot{s['slot']}" == key)]
        if xs:
            out[key] = {"n": len(xs), "value": sum(bool(s["exact"]) for s in xs) / len(xs),
                        "token": sum(s["tok_ok"] for s in xs) / len(xs), "fit": sum(bool(s["fit"]) for s in xs) / len(xs)}
            if "value_ok" in xs[0]:          # --slot-pad-weight != 1: the value's own tokens, whatever follows them
                out[key].update(value_prefix=sum(bool(s["value_ok"]) for s in xs) / len(xs),
                                first=sum(bool(s["first_ok"]) for s in xs) / len(xs))
    return out


def slot_layer_means(model, tokenizer, items: list[dict], *, layer: int, max_len: int, bs: int = 8) -> torch.Tensor:
    """The mean state of every slot position (slot j's copy i; [k x c, hidden], float32, CPU) at the output of
    decoder layer `layer`, over the final questions of items (the registered slot line): the mean-ablation's
    replacement. Under FSDP every rank runs the same items inside the root's forward."""
    if getattr(model, "cotd_readout", False):
        return model.run(_slot_layer_means, tokenizer, items, layer, max_len, bs)
    return _slot_layer_means(model, tokenizer, items, layer, max_len, bs)


@torch.no_grad()
def _slot_layer_means(model, tokenizer, items, layer, max_len, bs) -> torch.Tensor:
    spec, _ = _slot_reg(tokenizer)
    sid = _slot_ids(tokenizer)
    was_training = model.training
    model.eval()
    body, head = text_parts(model)
    dev = head.weight.device
    acc, n = None, 0
    for s in range(0, len(items), bs):
        exs = [Example(final_text(tokenizer, it["prompt"]), it["labels"], [0.0] * len(it["labels"]), 1.0, "final",
                       it["item_id"]) for it in items[s:s + bs]]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            b = collate(tokenizer, exs, max_len, dev)
            with _SlotCapture(body, layer) as cap:
                body(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False)
        for i, e in enumerate(exs):
            x = cap.h[i, _slot_positions(b["input_ids"][i], sid, spec, e.item_id)].float()
            acc = x if acc is None else acc + x
            n += 1
    if was_training:
        model.train()
    if not n:
        raise ValueError("slot mean-ablation: no items for the slot means")
    return (acc / n).cpu()


def _layer_index(n: int, layer) -> int:
    li = n // 2 if layer == "mid" else int(layer)
    return li + n if li < 0 else li


class _SlotMean:
    """Necessity test "mean" / "mean@<layer>": the output of decoder layer `layer` at the slot positions (found by the
    slot ids in the input ids, captured by a pre-hook on the text decoder) replaced by means ([k x c, hidden]: slot j's
    copy i gets its mean state, slot_layer_means), every other position unchanged. A context manager."""

    def __init__(self, model, tokenizer, layer: int, means: torch.Tensor):
        from .fsdp import decoder_layers
        root = model.model if getattr(model, "cotd_readout", False) else model
        self.body, _ = text_parts(root)
        layers = decoder_layers(self.body)
        if not 0 <= layer <= len(layers) - 2:
            raise ValueError(f"slot mean layer {layer}: an intermediate decoder layer, 0..{len(layers) - 2}")
        self.layer, self.tok, self.means = layers[layer], tokenizer, means
        self.kc = means.shape[0]
        self.ids, self.handles = None, []

    def __enter__(self):
        def pre(mod, args, kwargs):
            self.ids = kwargs.get("input_ids", args[0] if args else None)

        def post(mod, args, out):
            if self.ids is None:
                return out
            h = out[0] if isinstance(out, tuple) else out
            mask = _slot_mask(self.ids, self.tok)
            cnt = mask.sum(-1)
            if not bool(mask.any()):
                return out
            if not bool(((cnt == 0) | (cnt == self.kc)).all()):
                raise ValueError(f"slot mean-ablation: rows with {cnt.tolist()} slot tokens, want 0 or {self.kc}")
            h = h.clone()
            h[mask] = self.means.to(h.device, h.dtype).repeat(int((cnt > 0).sum()), 1)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        self.handles = [self.body.register_forward_pre_hook(pre, with_kwargs=True),
                        self.layer.register_forward_hook(post)]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


def _value(text: str | None) -> str | None:
    """The result of a step text: what follows its last '=' (stripped), else None."""
    if not text or "=" not in text:
        return None
    return text.rsplit("=", 1)[1].strip() or None


def slot_decode(model, tokenizer, items: list[dict], *, max_len: int, bs: int = 8, max_new: int = 24,
                overflow: str = "merge") -> list[dict]:
    """The "slot j decodes to 4 + 2 = 6" readout: for each item, the main pass over its final question with the
    registered slot line (final_text), then every slot's final-layer states start a separate sequence (as in training)
    decoded greedily with the same body and head, up to max_new tokens or SLOT_END. Rows: item_id, n_steps and per slot
    target (slot_targets; None past the chain), decoded, exact (decoded == target) and value_ok (same result after
    the last '='). Under FSDP every rank decodes the same items inside the root's forward."""
    if getattr(model, "cotd_readout", False):
        return model.run(_slot_decode, tokenizer, items, max_len, bs, max_new, overflow)
    return _slot_decode(model, tokenizer, items, max_len, bs, max_new, overflow)


@torch.no_grad()
def _slot_decode(model, tokenizer, items, max_len, bs, max_new, overflow) -> list[dict]:
    spec, _ = _slot_reg(tokenizer)
    sid = _slot_ids(tokenizer)
    was_training = model.training
    model.eval()
    body, head = text_parts(model)
    dev = head.weight.device
    emb = model.get_input_embeddings()
    cap = logit_softcap(model)
    bias = head.bias if getattr(head, "bias", None) is not None else None
    rows = []
    for s in range(0, len(items), bs):
        its = items[s:s + bs]
        exs = [Example(final_text(tokenizer, it["prompt"]), it["labels"], [0.0] * len(it["labels"]), 1.0, "final",
                       it["item_id"]) for it in its]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            b = collate(tokenizer, exs, max_len, dev)
            hidden = body(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                          use_cache=False).last_hidden_state
            vecs = []
            for i, e in enumerate(exs):
                pos = _slot_positions(b["input_ids"][i], sid, spec, e.item_id)
                vecs += [hidden[i, pos[j * spec.c:(j + 1) * spec.c]] for j in range(spec.k)]
            x, _, _ = _decode_inputs(model, tokenizer, vecs, [[]] * len(vecs))
            gen = [[] for _ in vecs]
            done = [False] * len(vecs)
            for _ in range(max_new):
                h = body(inputs_embeds=x, use_cache=False).last_hidden_state[:, -1]
                nxt = head_logits(h, head.weight, bias, cap).float().argmax(-1)
                for r, t in enumerate(nxt.tolist()):
                    if not done[r]:
                        gen[r].append(t)
                        done[r] = SLOT_END in tokenizer.decode(gen[r])
                if all(done):
                    break
                x = torch.cat([x, emb(nxt)[:, None].to(x.dtype)], dim=1)
        for i, it in enumerate(its):
            tg = slot_targets(it, spec.k, overflow)
            slots = []
            for j in range(spec.k):
                d = tokenizer.decode(gen[i * spec.k + j]).split(SLOT_END)[0].strip()
                t = tg[j]
                slots.append({"slot": j + 1, "target": t, "decoded": d, "exact": None if t is None else d == t,
                              "value_ok": None if t is None else _value(d) is not None and _value(d) == _value(t)})
            rows.append({"item_id": it["item_id"], "n_steps": len(worked_steps(it) or []), "slots": slots})
    if was_training:
        model.train()
    return rows


def slot_decode_summary(rows: list[dict]) -> dict:
    """Exact-match and value rates of slot_decode rows, overall and per slot (supervised slots only)."""
    out: dict = {}
    for key in ["all"] + sorted({f"slot{s['slot']}" for r in rows for s in r["slots"]}, key=lambda x: int(x[4:])):
        xs = [s for r in rows for s in r["slots"] if s["target"] is not None and (key == "all" or f"slot{s['slot']}" == key)]
        if xs:
            out[key] = {"n": len(xs), "exact": sum(bool(s["exact"]) for s in xs) / len(xs),
                        "value": sum(bool(s["value_ok"]) for s in xs) / len(xs)}
    return out


class _SlotNoise:
    """Necessity test "noise@<layer>:<sigma>": Gaussian noise of sigma x the state's RMS added to the output of decoder
    layer <layer> (an index into the decoder layers, "mid" = the middle one) at the slot positions only (found by the
    slot id in the input ids, captured by a pre-hook on the text decoder), from a fixed generator. A context manager;
    the hooks are removed on exit. (The final layer's slot states are not read by the readout, so the layer must be an
    intermediate one.)"""

    def __init__(self, model, tokenizer, layer, sigma: float, seed: int = 0):
        from .fsdp import decoder_layers
        root = model.model if getattr(model, "cotd_readout", False) else model
        self.body, _ = text_parts(root)
        self.layers = decoder_layers(self.body)
        n = len(self.layers)
        li = n // 2 if layer == "mid" else int(layer)
        li = li + n if li < 0 else li
        if not 0 <= li <= n - 2:
            raise ValueError(f"slot noise layer {layer}: an intermediate decoder layer, 0..{n - 2} (of {n})")
        self.layer = self.layers[li]
        self.li = li
        self.sid = _slot_reg(tokenizer)[1]
        self.tok = tokenizer
        self.sigma, self.seed = sigma, seed
        self.ids = None
        self.handles = []

    def __enter__(self):
        def pre(mod, args, kwargs):
            self.ids = kwargs.get("input_ids", args[0] if args else None)
        self.gen = None

        def post(mod, args, out):
            if self.ids is None:
                return out
            h = out[0] if isinstance(out, tuple) else out
            mask = _slot_mask(self.ids, self.tok)
            if not bool(mask.any()):
                return out
            if self.gen is None or self.gen.device != h.device:
                self.gen = torch.Generator(device=h.device).manual_seed(self.seed)
            sel = h[mask]
            rms = sel.float().pow(2).mean(-1, keepdim=True).sqrt()
            noise = torch.randn(sel.shape, generator=self.gen, device=h.device, dtype=torch.float32)
            h = h.clone()
            h[mask] = (sel.float() + self.sigma * rms * noise).to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        self.handles = [self.body.register_forward_pre_hook(pre, with_kwargs=True),
                        self.layer.register_forward_hook(post)]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


def parse_slot_necessity(spec: str) -> list[str]:
    """--slot-necessity value -> modes: "swap" (every slot token replaced by the swap token: same length), "drop" (the
    prompt without the slot line: the A2 format), "noise@<layer>:<sigma>" (_SlotNoise; <layer> an int or "mid"),
    "mean" / "mean@<layer>" (v2.1, _SlotMean: the slot states at the output of the slot layer, or of <layer>, replaced
    by their mean over training items, slot_layer_means), "force" / "forceswap" (TSF, --slot-readout tsf: the layer-L
    slot states replaced by the teacher view's vectors of the item's own worked steps / of another item's, tsf_eval)."""
    modes = [x.strip() for x in spec.split(",") if x.strip()]
    for m in modes:
        if m in ("swap", "drop", "mean", "force", "forceswap") or re.fullmatch(r"mean@(mid|-?\d+)", m):
            continue
        hit = re.fullmatch(r"noise@(mid|-?\d+):([0-9.]+(?:e-?\d+)?)", m)
        if not hit:
            raise ValueError(f"--slot-necessity {m!r}: swap, drop, mean[@<layer|mid>], noise@<layer|mid>:<sigma>, "
                             "force or forceswap")
    return modes


def slot_necessity(model, tok, items: list[dict], max_len: int, bs: int, cache: dict, modes: list[str], *,
                   swap_token: str, intact: dict, seed: int = 0, mean_items: list | None = None,
                   means: dict | None = None) -> dict:
    """The necessity test: evaluate's metrics and predictions on items with the slots perturbed (parse_slot_necessity
    modes), next to the intact metrics: {mode: (metrics + acc_intact / nll_intact / acc_drop = intact - perturbed,
    preds)}. Swap replaces every slot token (distinct ones too) by the one swap token. The mean modes need mean_items
    (training items: their slot means are computed once per layer and kept in means, a dict layer -> tensor). The
    registered slot prompt is restored afterwards. force / forceswap (TSF): tsf_eval, run once for both."""
    spec, _ = _slot_reg(tok)
    means = means if means is not None else {}
    out = {}
    forced = None
    try:
        for m in modes:
            if m in ("force", "forceswap"):
                set_slot_prompt(tok, spec)
                if not any(worked_steps(it) for it in items):    # nothing to force: the mode is left out
                    print(f"slot necessity {m}: no item with worked steps, skipped", flush=True)
                    continue
                if forced is None:
                    forced = tsf_eval(model, tok, items, max_len=max_len, bs=bs,
                                      modes=tuple(x for x in modes if x in ("force", "forceswap")))
                mm, pr = forced[m]
            elif m == "swap":
                set_slot_prompt(tok, dataclasses.replace(spec, token=swap_token, tokens=()))
                mm, pr = evaluate(model, tok, items, max_len, bs, cache)
            elif m == "drop":
                set_slot_prompt(tok, None)
                mm, pr = evaluate(model, tok, items, max_len, bs, cache)
            elif m.startswith("mean"):
                set_slot_prompt(tok, spec)
                if m == "mean":
                    if spec.layer is None:
                        raise ValueError("--slot-necessity mean: the slot spec has no --slot-layer (use mean@<layer>)")
                    li = spec.layer
                else:
                    li = _layer_index(decoder_count(model), m[len("mean@"):])
                if not mean_items:
                    raise ValueError(f"--slot-necessity {m}: no training items for the slot means")
                if li not in means:
                    means[li] = slot_layer_means(model, tok, mean_items, layer=li, max_len=max_len, bs=bs)
                with _SlotMean(model, tok, li, means[li]):
                    mm, pr = evaluate(model, tok, items, max_len, bs, cache)
                mm = dict(mm, mean_layer=li, mean_items=len(mean_items))
            else:
                layer, sigma = m[len("noise@"):].split(":")
                set_slot_prompt(tok, spec)
                with _SlotNoise(model, tok, layer, float(sigma), seed=seed):
                    mm, pr = evaluate(model, tok, items, max_len, bs, cache)
            set_slot_prompt(tok, spec)
            if "acc" in mm and "acc" in intact:
                mm = dict(mm, acc_intact=intact["acc"], acc_drop=intact["acc"] - mm["acc"],
                          nll_intact=intact.get("nll"))
            out[m] = (mm, pr)
    finally:
        set_slot_prompt(tok, spec)
    return out


def slot_mode_name(mode: str) -> str:
    """A file-name-safe name of a necessity mode ("noise@mid:1.0" -> "noise_mid_1.0")."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", mode)


SLOT_SIDECAR = "latent_slots.json"          # next to a saved slot-arm model: the slot line eval_student rebuilds


def slot_evals(model, tok, evals: dict, results: dict, out, *, modes: list[str], sets: list[str], swap_token: str | None,
               decode_n: int, decode_set: str, max_len: int, bs: int, cache: dict, seed: int = 0,
               overflow: str = "merge", write: bool = True, mean_items: list | None = None,
               intact_preds: dict | None = None) -> None:
    """After the plain evaluations of a slot arm (results["eval"][set] = the intact metrics): the necessity test
    (slot_necessity) on the named sets (all when empty; names not in evals are skipped), eval keys <set>@<mode>,
    results["slot_necessity"][set][mode] and preds_<set>@<mode>.jsonl; then the decoding readout (slot_decode) of the
    first decode_n items of decode_set (default / absent: the first set): slot_decode_<set>.jsonl and
    results["slot_decode"] (slot_decode_summary). A lens spec (v2.1) gets the lens readout instead, on every set of
    the comma list decode_set (default: the first set; the first decode_n items of each): slot_readout_<set>.jsonl,
    results["slot_readout"][set] (slot_readout_summary) and results["slot_pilot"] (the pilot's pass criteria: value
    readout >= 45% on the first readout set and an acc_drop >= 2 points in the mean (else swap) mode of the first
    necessity set). mean_items: the training items of the mean modes. write False: no files (FSDP ranks other than
    0). A TSF spec (--slot-readout tsf) has no slot readout: every necessity mode also gets its numbers on the items
    with a chain (tsf_subsets; intact_preds: name -> the intact predictions, else they are recomputed) and
    results["tsf_pilot"] the own-run part of gate G1 (mean drop >= 2, forceswap drop >= 3 points)."""
    import json
    from pathlib import Path
    out = Path(out)
    means: dict = {}
    spec0 = slot_spec(tok)
    tsf = spec0 is not None and spec0.readout == "tsf"
    for name in sets or list(evals):
        if name not in evals:
            print(f"slot necessity: no eval set {name!r}, skipped", flush=True)
            continue
        if not modes:
            break
        intact = results["eval"].get(name) or evaluate(model, tok, evals[name], max_len, bs, cache)[0]
        res = slot_necessity(model, tok, evals[name], max_len, bs, cache, modes, swap_token=swap_token or "",
                             intact=intact, seed=seed, mean_items=mean_items, means=means)
        if tsf:
            ip = (intact_preds or {}).get(name)
            if ip is None:
                ip = evaluate(model, tok, evals[name], max_len, bs, cache)[1]
        for mode, (mm, pr) in res.items():
            key = f"{name}@{mode}"
            results["eval"][key] = mm
            results.setdefault("slot_necessity", {}).setdefault(name, {})[mode] = {
                k: mm.get(k) for k in ("n", "acc", "acc_intact", "acc_drop", "nll", "nll_intact", "ece") +
                (("mean_layer", "mean_items") if "mean_layer" in mm else ()) + (("n_forced",) if "n_forced" in mm else ())}
            if tsf:                          # TSF: the chained items, and the items with an intermediate value as option
                results["slot_necessity"][name][mode].update(tsf_subsets(evals[name], pr, ip, spec0))
            if write:
                with open(out / f"preds_{name}@{slot_mode_name(mode)}.jsonl", "w") as f:
                    for p in pr:
                        f.write(json.dumps(p) + "\n")
            print(key, json.dumps(results["slot_necessity"][name][mode]), flush=True)
    spec = slot_spec(tok)
    if tsf:                                  # no slot readout; the own-run part of G1 on the first necessity set
        if results.get("slot_necessity"):
            nset, nm = next(iter(results["slot_necessity"].items()))
            pil = {"necessity_set": nset}
            for mode in ("mean", "swap", "drop", "force", "forceswap"):
                if mode in nm:
                    pil[f"{mode}_drop"] = nm[mode].get("acc_drop")
                    pil[f"{mode}_drop_chain"] = nm[mode].get("acc_drop_chain")
            if "mean_drop" in pil and "forceswap_drop" in pil:
                pil["g1_own"] = bool(pil["mean_drop"] is not None and pil["mean_drop"] >= 0.02 and
                                     pil["forceswap_drop"] is not None and pil["forceswap_drop"] >= 0.03)
            results["tsf_pilot"] = pil
            print("tsf pilot", json.dumps(pil), flush=True)
        return
    if spec is not None and spec.readout == "lens":
        if decode_n and evals:
            names = [x for x in decode_set.split(",") if x in evals] or [next(iter(evals))]
            for name in names:
                rows = slot_lens_readout(model, tok, evals[name][:decode_n], max_len=max_len, bs=bs, overflow=overflow)
                results.setdefault("slot_readout", {})[name] = {"items": len(rows), "layer": spec.layer,
                                                                **slot_readout_summary(rows)}
                if write:
                    with open(out / f"slot_readout_{name}.jsonl", "w") as f:
                        for r in rows:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"slot readout {name}", json.dumps(results["slot_readout"][name]), flush=True)
                for r in rows[:2]:
                    print("  ", r["item_id"], " | ".join(f"{s['slot']}: {s['read']!r} ({s['target']!r})"
                                                         for s in r["slots"]), flush=True)
        pil: dict = {}
        if results.get("slot_readout"):
            first = next(iter(results["slot_readout"].values()))
            pil["readout_set"] = next(iter(results["slot_readout"]))
            pil["readout_value"] = first.get("all", {}).get("value")
            if "value_prefix" in first.get("all", {}):   # --slot-pad-weight != 1: no trained SLOT_END after the value
                pil.update(readout_metric="value_prefix", readout_value=first["all"]["value_prefix"])
            pil["readout_ok"] = pil["readout_value"] is not None and pil["readout_value"] >= 0.45
        if results.get("slot_necessity"):
            nset, nm = next(iter(results["slot_necessity"].items()))
            mode = next((m for m in ("mean", "swap") if m in nm), None) or next(iter(nm))
            pil.update(necessity_set=nset, necessity_mode=mode, acc_drop=nm[mode].get("acc_drop"))
            pil["necessity_ok"] = pil["acc_drop"] is not None and pil["acc_drop"] >= 0.02
        if pil:
            pil["pass"] = bool(pil.get("readout_ok")) and bool(pil.get("necessity_ok"))
            results["slot_pilot"] = pil
            print("slot pilot", json.dumps(pil), flush=True)
        return
    if decode_n and evals:
        name = decode_set if decode_set in evals else next(iter(evals))
        rows = slot_decode(model, tok, evals[name][:decode_n], max_len=max_len, bs=bs, overflow=overflow)
        results["slot_decode"] = {"set": name, "items": len(rows), **slot_decode_summary(rows)}
        if write:
            with open(out / f"slot_decode_{name}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("slot decode", json.dumps(results["slot_decode"]), flush=True)
        for r in rows[:2]:
            print("  ", r["item_id"], " | ".join(f"{s['slot']}: {s['decoded']!r} ({s['target']!r})" for s in r["slots"]),
                  flush=True)


# ---- teacher-state slots (TSF) and answer-position self-distillation (PTAD) ------------------------------------------
# notes/latent_comeback_memo_2026-10-21.md: §3 TSF (arms TSF / TSFN / TSFR / TSFX / TSF6; --slot-readout tsf) and §4
# PTAD, the zero-latent CODI backup (PTAD / PTADR / PTADX; --answer-teacher). All opt-in: nothing below runs on any
# other arm's path. A teacher view through the same weights, the final question with the item's worked steps written
# before its options (SI stage 0: insert_chain + chain_block; no slot line), gives (i) regression targets, its residual
# states at the OUTPUT of decoder layers B (stop-grad): TSF at two positions of the line of slot j's worked step (slot
# token 1 <- the token that predicts the value's first token, token 2 <- the value's last token), PTAD at the
# "Answer:" position; loss smooth L1 (beta 1) of s / sigma vs t / sigma, sigma = the RMS of the teacher vector (the
# memo's small-batch variant of CODI's batch std), mean over hidden dims, layers (and the c tokens), weight x lambda /
# supervised slots per problem (TSF) or x lambda (PTAD); (ii) TSF only, forced reading: in a micro-batch whose coin
# (keyed by seed, update and the micro-batch's items: tsf_coin) falls below p, the student's layer-L slot states are
# REPLACED by the teacher's layer-L vectors before layers L + 1 .. n - 1 compute the letter CE; the align loss is taken
# on the states before the replacement, so one student pass + one teacher pass per micro-batch. The teacher view also
# trains the letter CE at weight teacher_ce x the example's weight. Deployed: LSL's prompt (TSF) / A2's (PTAD).
TEACHER_KINDS = ("steps", "shuffled", "placebo")
TEACHER_MODES = ("tsf", "ptad")


@dataclass
class TeacherConfig:
    """A TSF / PTAD run's teacher settings, registered per tokenizer (set_teacher) like the slot prompt. layers: the
    decoder layers (0-based) whose OUTPUT is aligned (TSF: the last one is the slot spec's layer L, where forcing
    happens); scale: the one-time rescale of the align loss (--teacher-grad-probe); teacher_ce: the teacher view's
    letter-CE weight (--teacher-ce-guard may lower it once). Both are read at every forward."""
    mode: str
    kind: str
    layers: tuple
    force_p: float = 0.0
    teacher_ce: float = 0.5
    seed: int = 0
    scale: float = 1.0
    smoothing: float = 0.0


_TEACHERS: dict[int, tuple] = {}


def set_teacher(tokenizer, cfg: TeacherConfig | None) -> None:
    """Register (cfg) or clear (None) the teacher settings teacher_forward reads for this tokenizer."""
    if cfg is None:
        _TEACHERS.pop(id(tokenizer), None)
        return
    if cfg.mode not in TEACHER_MODES or cfg.kind not in TEACHER_KINDS:
        raise ValueError(f"teacher: mode {cfg.mode!r} / kind {cfg.kind!r}")
    _TEACHERS[id(tokenizer)] = (tokenizer, cfg)


def teacher_config(tokenizer) -> TeacherConfig | None:
    hit = _TEACHERS.get(id(tokenizer))
    return hit[1] if hit is not None and hit[0] is tokenizer else None


@dataclass
class TeacherExample(SlotExample):
    """A final view with a teacher view (TSF / PTAD; a subclass, so Example / SlotExample are unchanged). teacher_text:
    the final question with worked-step lines before its options (the item's own: steps; another item's: shuffled;
    the problem's sentences: placebo), no slot line; "" = no teacher view (an item without a chain: no teacher CE, no
    alignment). teacher_marks (TSF): per slot None or (first, last), the characters of teacher_text that start / end the
    'value' of the slot's line (steps: what follows the last '='; placebo: the line's last character). slot_lambda:
    lambda / supervised slots (TSF), lambda (PTAD), 0 without a teacher view."""
    teacher_text: str = ""
    teacher_marks: tuple = ()


@dataclass(frozen=True)
class TeacherPlan:
    """The teacher view of one item: lines (the step lines written, empty = no teacher view), slot_lines (TSF: per slot
    the index of its line, None = unsupervised; PTAD: ()), rule ("value": the value after the line's last '=';
    "last": the line's last character, placebo lines)."""
    lines: tuple = ()
    slot_lines: tuple = ()
    rule: str = "value"


def parse_layer_list(spec: str, n: int) -> tuple:
    """--slot-align-layers / --answer-align-layers: "a-b" ranges and integers, comma separated (0-based decoder layers
    whose OUTPUT is meant; each must be read by a later layer: 0..n-2) -> sorted unique tuple."""
    out = set()
    for part in (x.strip() for x in str(spec).split(",")):
        if not part:
            continue
        a, sep, b = part.partition("-")
        try:
            got = range(int(a), int(b) + 1) if sep and a else [int(part)]
        except ValueError:
            raise ValueError(f"{spec!r}: 'a-b' ranges or integers")
        out.update(got)
    if not out:
        raise ValueError(f"{spec!r}: no layer")
    bad = [x for x in out if not 0 <= x <= n - 2]
    if bad:
        raise ValueError(f"{spec!r}: layers {sorted(bad)} outside 0..{n - 2} (intermediate layers of {n})")
    return tuple(sorted(out))


def slot_step_map(n: int, k: int, overflow: str = "merge") -> list:
    """Slot j's worked step (0-based index into a chain of n steps) or None, as slot_value_targets with fill none: a
    chain of n <= k steps fills slots 1..n; a longer one: merge = steps 1..k-1 and the LAST step on slot k (TSF6:
    the first five steps + the answer's), truncate = steps 1..k (TSF)."""
    if overflow not in SLOT_OVERFLOW:
        raise ValueError(f"--slot-overflow {overflow!r}: one of {', '.join(SLOT_OVERFLOW)}")
    if n <= k:
        return list(range(n)) + [None] * (k - n)
    return list(range(k - 1)) + [n - 1] if overflow == "merge" else list(range(k))


def placebo_lines(item: dict, n: int) -> list[str]:
    """TSFX / PTADX: n lines of text with no computation in place of the n worked steps: the sentences of the item's
    placebo text (placebo_text: the problem's statements), in order, cycled to n lines."""
    sents = [x.strip() for x in _pl_sentences(placebo_text(item)) if x.strip()]
    if not sents:
        raise ValueError(f"{item.get('item_id')}: no placebo sentence")
    return [sents[i % len(sents)] for i in range(n)]


def teacher_partners(items: list[dict], seed, need: dict | None = None) -> dict:
    """TSFR / PTADR (and the cross-item forced evaluation, forceswap): item_id -> the id of another item whose worked
    steps the item's teacher view shows. Candidates: items with a chain of at least need[item] steps (default 1, so
    every supervised slot of the item gets a line), other than the item and with other step lines; drawn by a
    generator keyed by seed and the item (never the training rng; fixed over the epochs)."""
    chains = {it["item_id"]: tuple(s["text"] for s in worked_steps(it) or []) for it in items}
    ids = sorted((i for i, c in chains.items() if c), key=str)
    by_min: dict = {}
    out = {}
    for iid in ids:
        m = max(1, int((need or {}).get(iid, 1)))
        if m not in by_min:
            by_min[m] = [j for j in ids if len(chains[j]) >= m]
        pool = by_min[m]
        rng = random.Random(f"teacher-partner|{seed}|{iid}")
        pick = None
        for _ in range(64):
            j = pool[rng.randrange(len(pool))] if pool else None
            if j is not None and j != iid and chains[j] != chains[iid]:
                pick = j
                break
        if pick is None:
            ok = [j for j in pool if j != iid and chains[j] != chains[iid]]
            if not ok:
                raise ValueError(f"teacher partners: no other item with >= {m} worked steps for {iid}")
            pick = ok[rng.randrange(len(ok))]
        out[iid] = pick
    return out


def teacher_plan(item: dict, kind: str, *, k: int = 0, overflow: str = "merge", partner: dict | None = None) -> TeacherPlan:
    """The teacher view of item: steps = its worked steps (slot j <- slot_step_map's step); shuffled = partner's steps
    (the item's supervised slots, each the partner's line at that slot by the same map; the partner has at least as
    many steps as the item has supervised slots); placebo = placebo_lines in place of its n steps (same slots, rule
    "last"). k = 0 (PTAD): no slots. An item without a chain gets the empty plan."""
    if kind not in TEACHER_KINDS:
        raise ValueError(f"teacher kind {kind!r}: one of {', '.join(TEACHER_KINDS)}")
    steps = worked_steps(item) or []
    if not steps:
        return TeacherPlan((), (None,) * k)
    own = slot_step_map(len(steps), k, overflow) if k else []
    if kind == "steps":
        return TeacherPlan(tuple(s["text"] for s in steps), tuple(own), "value")
    if kind == "placebo":
        return TeacherPlan(tuple(placebo_lines(item, len(steps))), tuple(own), "last")
    if partner is None:
        raise ValueError(f"{item.get('item_id')}: shuffled teacher view without a partner")
    lines = tuple(s["text"] for s in worked_steps(partner) or [])
    pm = slot_step_map(len(lines), k, overflow) if k else []
    smap = []
    for j, li in enumerate(own):
        if li is not None and pm[j] is None:
            raise ValueError(f"{item.get('item_id')}: partner {partner.get('item_id')} has no line for slot {j + 1}")
        smap.append(pm[j] if li is not None else None)
    if not lines:
        raise ValueError(f"{item.get('item_id')}: partner {partner.get('item_id')} has no worked steps")
    return TeacherPlan(lines, tuple(smap), "value")


def plan_slot_targets(plan: TeacherPlan, k: int) -> list:
    """The slot targets of a TSF item (SlotExample.slot_targets): its plan's line per supervised slot, else None."""
    if not plan.lines:
        return [None] * k
    return [plan.lines[li] if li is not None else None for li in plan.slot_lines]


def teacher_text_marks(text: str, plan: TeacherPlan) -> tuple[str, tuple]:
    """(teacher view text, marks) of a final view text (FINAL_TEMPLATE; no slot line): text with chain_block(lines)
    inserted before its options (insert_chain), and per slot of plan.slot_lines None or (first, last), the character
    indices of the value of the slot's line in it (rule "value": from the first non-space character after the line's
    last '=' to its last non-space character; "last": the line's last non-space character twice). ("", all None)
    for an empty plan."""
    if not plan.lines:
        return "", tuple(None for _ in plan.slot_lines)
    t = insert_chain(text, chain_block(list(plan.lines)))
    off = len(text.rpartition("\nOptions:\n")[0]) + len(f"\n\n{CHAIN_HEADER}\n")
    starts = []
    for i, ln in enumerate(plan.lines):
        pre = f"Step {i + 1}: "
        starts.append(off + len(pre))
        off += len(pre) + len(ln) + 1
    marks = []
    for li in plan.slot_lines:
        if li is None:
            marks.append(None)
            continue
        ln = plan.lines[li]
        end = len(ln.rstrip()) - 1
        if end < 0:
            raise ValueError("teacher view: an empty line")
        if plan.rule == "value":
            eq = ln.rfind("=")
            tail = ln[eq + 1:] if eq >= 0 else ""
            if not tail.strip():
                raise ValueError(f"teacher view: worked step {ln!r} has no value after '='")
            first = eq + 1 + len(tail) - len(tail.lstrip())
        else:
            first = end
        marks.append((starts[li] + first, starts[li] + end))
    return t, tuple(marks)


def _offsets(tokenizer, text: str) -> tuple[list[int], list]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    if ids != text_ids(tokenizer, text):
        raise ValueError("teacher view: the offset-mapped encoding differs from text_ids (a slow tokenizer, or a chat "
                         "template)")
    return ids, [tuple(o) for o in enc["offset_mapping"]]


def _token_of(offs: list, c: int) -> int:
    for t, (a, b) in enumerate(offs):
        if a <= c < b:
            return t
    raise ValueError(f"teacher view: no token covers character {c}")


def teacher_mark_positions(tokenizer, text: str, marks: tuple, max_len: int) -> list:
    """Per slot (marks order): None, or the two positions of the slot's teacher tokens in collate's row of text (the
    special prefix + text_ids cut at the head to max_len, as encode_tail): (the token BEFORE the token holding the
    value's first character = the position whose next token is the value's first; on Qwen3.5 "= 6" is ' =', ' ', '6',
    so it is the bare space, on GPT-2 ' =' before ' 6'), the token holding the value's last character); None when a
    position falls in the cut head."""
    ids, offs = _offsets(tokenizer, text)
    pre = len(special_prefix(tokenizer))
    cut = max(0, len(ids) - max(1, max_len - pre))
    out = []
    for mk in marks:
        if mk is None:
            out.append(None)
            continue
        t1, t2 = _token_of(offs, mk[0]) - 1, _token_of(offs, mk[1])
        out.append((pre + t1 - cut, pre + t2 - cut) if t1 >= 0 and t1 - cut >= 0 else None)
    return out


def tsf_coin(seed: int, step: int, batch: list) -> float:
    """The forced-reading coin of one micro-batch: keyed by the seed, the update and the micro-batch's item ids (sorted),
    never by its index, so --fsdp's re-ordered micro-batches (split_update) draw the single-process coins."""
    key = "|".join(sorted(str(e.item_id) for e in batch))
    return random.Random(f"tsf-force|{seed}|{step}|{key}").random()


class _LayerCapture:
    """The OUTPUT of decoder layers `layers` (0-based) at every position, not detached (one-shot forward pre-hooks on
    layer l + 1, as _SlotCapture: with or without gradient checkpointing the captured input of a checkpointed layer is
    the graph tensor itself). force = (li, rows, pos, vals): the output of layer li at (rows[r], pos[r]) replaced by
    vals[r] (a forward hook on layer li); its un-replaced output is the capture of li. While the context is open in
    training, layer li runs WITHOUT gradient checkpointing: a checkpointed layer recomputes its forward in backward,
    after the context (and its hook) is gone, which would drop the replacement from the gradient. A context manager
    over one forward."""

    def __init__(self, body, layers, force=None):
        from .fsdp import decoder_layers
        self.dl = decoder_layers(body)
        n = len(self.dl)
        for li in list(layers) + ([force[0]] if force is not None else []):
            if not 0 <= li <= n - 2:
                raise ValueError(f"layer {li}: an intermediate decoder layer, 0..{n - 2}")
        self.layers, self.force = sorted(set(layers)), force
        self.h, self.handles, self._ckpt = {}, [], None

    def __enter__(self):
        for li in self.layers:
            def pre(mod, args, kwargs, li=li):
                if li not in self.h:
                    self.h[li] = args[0] if args else kwargs["hidden_states"]
            self.handles.append(self.dl[li + 1].register_forward_pre_hook(pre, with_kwargs=True))
        if self.force is not None:
            li, rows, pos, vals = self.force
            lay = self.dl[li]
            if getattr(lay, "gradient_checkpointing", False):
                self._ckpt = lay
                lay.gradient_checkpointing = False

            def post(mod, args, out):
                h = out[0] if isinstance(out, tuple) else out
                if li not in self.h:
                    self.h[li] = h
                h = h.clone()
                h[rows, pos] = vals.to(h.device, h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            self.handles.append(lay.register_forward_hook(post))
        return self

    def __exit__(self, *exc):
        for hd in self.handles:
            hd.remove()
        self.handles = []
        if self._ckpt is not None:
            self._ckpt.gradient_checkpointing = True
            self._ckpt = None
        return False


def _align_rows(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """[N] alignment losses of student / teacher states [N, ..., d]: smooth L1 (beta 1) of s / sigma vs t / sigma with
    sigma the RMS of each teacher vector (detached), mean over the hidden dims, then over every other axis."""
    t = t.detach().float()
    s = s.float()
    sig = t.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    per = F.smooth_l1_loss(s / sig, t / sig, reduction="none", beta=1.0).mean(-1)
    return per.flatten(1).mean(1)


def teacher_forward(model, tokenizer, batch: list, max_len: int, cache: dict, step: int = 0, _always: bool = False):
    """One training micro-batch of a TSF / PTAD arm: (the student's letter logits as label_logits gives them, parts,
    stats); parts = {"teacher": the teacher views' letter CE summed over examples (teacher_ce x weight each),
    "align": scale x the summed weighted alignment loss}, so loss = (kl_loss x n + teacher + align) / norm. TSF also
    forces (tsf_coin(seed, step, batch) < force_p). Under FSDP both passes run inside the root's forward, and a
    micro-batch without a teacher view still runs a weight-0 teacher pass (the same collectives on every rank).
    stats: slot_n (aligned slots / answer positions), align_sum (their unweighted losses), teacher_n, teacher_ce_sum,
    forced (0 / 1), forced_n (examples forced), marks_cut (teacher positions lost to max_len)."""
    if getattr(model, "cotd_readout", False):
        return model.run(teacher_forward, tokenizer, batch, max_len, cache, step, True)
    cfg = teacher_config(tokenizer)
    if cfg is None:
        raise ValueError("teacher_forward: no teacher settings registered for this tokenizer (set_teacher)")
    return (_tsf_forward if cfg.mode == "tsf" else _ptad_forward)(model, tokenizer, batch, max_len, cache, step,
                                                                  _always, cfg)


def _teacher_pass(model, tokenizer, batch: list, max_len: int, cache: dict, cfg: TeacherConfig, layers, _always: bool):
    """The teacher views of a micro-batch in one pass: None (no view and not _always), else (letter-CE sum, the views'
    batch indices, the collated views, {layer: output, detached}, stats). A micro-batch without a view under FSDP runs
    batch[0]'s text at weight 0 instead (no stats, no targets)."""
    body, head = text_parts(model)
    dev = head.weight.device
    idx = [i for i, e in enumerate(batch) if isinstance(e, TeacherExample) and e.teacher_text]
    tex = [Example(batch[i].teacher_text, batch[i].labels, batch[i].target, batch[i].weight * cfg.teacher_ce,
                   "teacher", batch[i].item_id) for i in idx]
    if not tex and not _always:
        return None
    stand = not tex
    if stand:
        e0 = batch[0]
        tex = [Example(e0.text, e0.labels, e0.target, 0.0, "teacher", e0.item_id)]
    bt = collate(tokenizer, tex, max_len, dev)
    with _LayerCapture(body, layers) as cap:
        ht = body(input_ids=bt["input_ids"], attention_mask=bt["attention_mask"], use_cache=False).last_hidden_state
    zt = _letters(model, tokenizer, head, ht[torch.arange(len(tex), device=dev), bt["last"]], tex, cache)
    ce = kl_loss(zt, tex, cfg.smoothing) * len(tex)
    with torch.no_grad():
        ce_sum = 0.0 if stand else sum(float(kl_loss([z.detach().float()], [dataclasses.replace(e, weight=1.0)],
                                                     cfg.smoothing)) for z, e in zip(zt, tex))
    hs = {li: cap.h[li].detach() for li in layers}
    return ce, ([] if stand else idx), bt, hs, {"teacher_n": 0 if stand else len(tex), "teacher_ce_sum": ce_sum}


def _tsf_forward(model, tokenizer, batch, max_len, cache, step, _always, cfg):
    spec, _ = _slot_reg(tokenizer)
    sid = _slot_ids(tokenizer)
    body, head = text_parts(model)
    dev = head.weight.device
    layers = tuple(sorted(set(cfg.layers)))
    L = spec.layer
    if L is None or L not in layers or spec.c != 2:
        raise ValueError("TSF: the slot layer must be the last align layer, and c = 2 tokens per slot")
    li_L = layers.index(L)
    zero = torch.zeros((), device=dev, dtype=torch.float32)
    tp = _teacher_pass(model, tokenizer, batch, max_len, cache, cfg, layers, _always)
    tgt: dict = {}                           # batch index -> [(slot j, teacher states [layers, c, d])]
    cut = 0
    if tp is not None:
        ce_t, idx, _, hs, tst = tp
        for r, i in enumerate(idx):
            e = batch[i]
            for j, pp in enumerate(teacher_mark_positions(tokenizer, e.teacher_text, e.teacher_marks, max_len)):
                if e.teacher_marks[j] is None:
                    continue
                if pp is None:
                    cut += 1
                    continue
                ix = torch.tensor(pp, device=dev)
                tgt.setdefault(i, []).append((j, torch.stack([hs[li][r, ix] for li in layers])))
        del hs
    else:
        ce_t, tst = zero, {"teacher_n": 0, "teacher_ce_sum": 0.0}
    b = collate(tokenizer, batch, max_len, dev)
    rows, poss, tgts, wts = [], [], [], []
    for i, lst in tgt.items():
        e = batch[i]
        sp = _slot_positions(b["input_ids"][i], sid, spec, e.item_id)
        for j, t in lst:
            rows.append(i)
            poss.append(sp[j * spec.c:(j + 1) * spec.c])
            tgts.append(t)
            wts.append(e.weight * e.slot_lambda)
    forced = bool(rows) and cfg.force_p > 0 and tsf_coin(cfg.seed, step, batch) < cfg.force_p
    force = None
    if forced:
        force = (L, torch.tensor([r for r in rows for _ in range(spec.c)], device=dev), torch.cat(poss),
                 torch.cat([t[li_L] for t in tgts]))
    with _LayerCapture(body, layers, force) as cap:
        hidden = body(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).last_hidden_state
    out = _letters(model, tokenizer, head, hidden[torch.arange(hidden.shape[0], device=dev), b["last"]], batch, cache)
    align, asum = zero, 0.0
    if rows:
        S = torch.stack([torch.stack([cap.h[li][r, p] for li in layers]) for r, p in zip(rows, poss)])
        per = _align_rows(S, torch.stack(tgts))
        align = cfg.scale * (torch.tensor(wts, device=dev, dtype=torch.float32) * per).sum()
        asum = float(per.detach().sum())
    return out, {"teacher": ce_t, "align": align}, {
        "slot_n": len(rows), "align_sum": asum, "teacher_n": tst["teacher_n"], "teacher_ce_sum": tst["teacher_ce_sum"],
        "forced": int(forced), "forced_n": len(set(rows)) if forced else 0, "marks_cut": cut}


def _ptad_forward(model, tokenizer, batch, max_len, cache, step, _always, cfg):
    body, head = text_parts(model)
    dev = head.weight.device
    layers = tuple(sorted(set(cfg.layers)))
    zero = torch.zeros((), device=dev, dtype=torch.float32)
    tp = _teacher_pass(model, tokenizer, batch, max_len, cache, cfg, layers, _always)
    tgt: dict = {}                           # batch index -> teacher states at "Answer:" [layers, d]
    if tp is not None:
        ce_t, idx, bt, hs, tst = tp
        for r, i in enumerate(idx):
            if batch[i].slot_lambda:
                tgt[i] = torch.stack([hs[li][r, bt["last"][r]] for li in layers])
        del hs
    else:
        ce_t, tst = zero, {"teacher_n": 0, "teacher_ce_sum": 0.0}
    b = collate(tokenizer, batch, max_len, dev)
    with _LayerCapture(body, layers) as cap:
        hidden = body(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).last_hidden_state
    out = _letters(model, tokenizer, head, hidden[torch.arange(hidden.shape[0], device=dev), b["last"]], batch, cache)
    align, asum = zero, 0.0
    rows = sorted(tgt)
    if rows:
        S = torch.stack([torch.stack([cap.h[li][i, b["last"][i]] for li in layers]) for i in rows])
        per = _align_rows(S, torch.stack([tgt[i] for i in rows]))
        w = torch.tensor([batch[i].weight * batch[i].slot_lambda for i in rows], device=dev, dtype=torch.float32)
        align = cfg.scale * (w * per).sum()
        asum = float(per.detach().sum())
    return out, {"teacher": ce_t, "align": align}, {
        "slot_n": len(rows), "align_sum": asum, "teacher_n": tst["teacher_n"], "teacher_ce_sum": tst["teacher_ce_sum"],
        "forced": 0, "forced_n": 0, "marks_cut": 0}


def answer_group(group: list[Example], item: dict, *, plan: TeacherPlan | None, lam: float = 1.0,
                 stats: dict | None = None) -> list[Example]:
    """PTAD: a full-tree group reduced to its final view (the A2 prompt, no slots), as a TeacherExample whose teacher
    view is the final question with the plan's lines (teacher_text_marks) and slot_lambda = lam (0 and no teacher view
    for an item without a chain). The node views are dropped. Draws nothing from any generator."""
    st = stats if stats is not None else {}
    out = []
    for e in group:
        if e.kind != "final":
            continue
        t_text = teacher_text_marks(e.text, plan)[0] if plan is not None and plan.lines else ""
        st["teacher_views"] = st.get("teacher_views", 0) + bool(t_text)
        st["answer_items"] = st.get("answer_items", 0) + 1
        out.append(TeacherExample(**dataclasses.asdict(e), slot_targets=(), slot_lambda=lam if t_text else 0.0,
                                  teacher_text=t_text, teacher_marks=()))
    return out


# ---- TSF / PTAD evaluation: the early "use" detector, the forced necessity modes, held-out alignment -------------------
def _item_example(tokenizer, it: dict, text: str | None = None, kind: str = "final") -> Example:
    return Example(final_text(tokenizer, it["prompt"]) if text is None else text, it["labels"],
                   [0.0] * len(it["labels"]), 1.0, kind, it["item_id"])


def _metrics_preds(items: list[dict], probs_by: dict) -> tuple[dict, list[dict]]:
    """evaluate's (metrics, predictions) from per-index probabilities."""
    probs, gold, preds = [], [], []
    for i, it in enumerate(items):
        p = probs_by[i]
        preds.append({"item_id": it["item_id"], "group": it.get("group"), "labels": it["labels"], "probs": p,
                      "gold_label": it.get("gold_label"), "gold_probs": it.get("gold_probs")})
        if it.get("gold_label") in it["labels"]:
            probs.append(p)
            gold.append(it["labels"].index(it["gold_label"]))
    return (calibration(probs, gold) if gold else {"n": 0}), preds


def tsf_eval(model, tokenizer, items: list[dict], *, max_len: int, bs: int = 8, modes=("force", "forceswap"),
             align_layers=None) -> dict:
    """TSF evaluation on items with the registered slot line (one pass per mode; eval mode, no gradient). A teacher
    pass over every item with a chain (its own worked steps, FINAL_TEMPLATE without the slot line) gives the item's
    teacher vectors at its supervised slots (slot_step_map with the spec's k / overflow; positions as in training).
    Modes: "force" = the layer-L slot states of those slots replaced by the item's own vectors; "forceswap" = by those
    of another item (teacher_partners keyed "eval-swap", so every arm swaps the same pairs; the partner has a line for
    each of the item's supervised slots); items without a chain are run unchanged. Returns {mode: (metrics + n_forced,
    preds)}, "chained" (the ids with a chain) and, with align_layers, "align": the alignment loss (_align_rows, the
    training loss without weights) of the student's slot states at those layers (before any replacement) against the
    item's own teacher vectors, mean over supervised slots (n_slots; per slot j: align_slot<j>). Under FSDP every rank
    runs every item inside the root's forward."""
    if getattr(model, "cotd_readout", False):
        return model.run(_tsf_eval, tokenizer, items, max_len, bs, tuple(modes), tuple(align_layers or ()))
    return _tsf_eval(model, tokenizer, items, max_len, bs, tuple(modes), tuple(align_layers or ()))


@torch.no_grad()
def _tsf_eval(model, tokenizer, items, max_len, bs, modes, align_layers) -> dict:
    spec, _ = _slot_reg(tokenizer)
    if spec.layer is None or spec.c != 2:
        raise ValueError("forced evaluation: a TSF slot spec (an intermediate slot layer, c = 2)")
    for m in modes:
        if m not in ("force", "forceswap"):
            raise ValueError(f"tsf_eval mode {m!r}")
    sid = _slot_ids(tokenizer)
    L = spec.layer
    al = tuple(sorted(set(align_layers)))
    layers = tuple(sorted(set(al) | {L}))
    was_training = model.training
    model.eval()
    body, head = text_parts(model)
    dev = head.weight.device
    auto = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")
    plans = {}
    for it in items:
        pl = teacher_plan(it, "steps", k=spec.k, overflow=spec.overflow)
        if pl.lines:
            plans[it["item_id"]] = pl
    chained = [it for it in items if it["item_id"] in plans]
    V: dict = {}                             # item id -> {slot j: [layers, c, d] float32, CPU}
    order = sorted(range(len(chained)), key=lambda i: len(chained[i]["prompt"]))
    for s in range(0, len(order), bs):
        its = [chained[i] for i in order[s:s + bs]]
        tm = [teacher_text_marks(FINAL_TEMPLATE.format(problem=it["prompt"]), plans[it["item_id"]]) for it in its]
        exs = [_item_example(tokenizer, it, t, "teacher") for it, (t, _) in zip(its, tm)]
        with torch.autocast(**auto):
            bt = collate(tokenizer, exs, max_len, dev)
            with _LayerCapture(body, layers) as cap:
                body(input_ids=bt["input_ids"], attention_mask=bt["attention_mask"], use_cache=False)
        for r, (it, (t, mk)) in enumerate(zip(its, tm)):
            V[it["item_id"]] = {j: torch.stack([cap.h[li][r, torch.tensor(pp, device=dev)] for li in layers]).float().cpu()
                                for j, pp in enumerate(teacher_mark_positions(tokenizer, t, mk, max_len)) if pp is not None}
    partners = {}
    if "forceswap" in modes and V:
        partners = teacher_partners(chained, "eval-swap", {i: len(v) for i, v in V.items()})
    res: dict = {"chained": sorted((it["item_id"] for it in chained), key=str)}
    order_all = sorted(range(len(items)), key=lambda i: len(items[i]["prompt"]))
    li_L = layers.index(L)
    passes = list(modes) or (["align"] if al else [])
    if al and "force" not in passes and "align" not in passes:
        passes.append("align")
    aligned: list = []
    for mode in passes:
        probs_by, n_forced = {}, 0
        for s in range(0, len(order_all), bs):
            idx = order_all[s:s + bs]
            exs = [_item_example(tokenizer, items[i]) for i in idx]
            b = collate(tokenizer, exs, max_len, dev)
            rows, poss, vals, own = [], [], [], []
            for r, i in enumerate(idx):
                iid = items[i]["item_id"]
                if not V.get(iid):
                    continue
                src = V[partners[iid]] if mode == "forceswap" else V[iid]
                sp = _slot_positions(b["input_ids"][r], sid, spec, iid)
                n_forced += mode != "align"
                for j, t in V[iid].items():
                    own.append((r, sp[j * spec.c:(j + 1) * spec.c], t, j))
                    if j not in src:             # (a partner always has the item's slots; kept as a guard)
                        continue
                    rows += [r] * spec.c
                    poss.append(sp[j * spec.c:(j + 1) * spec.c])
                    vals.append(src[j][li_L])
            force = None
            if rows and mode != "align":
                force = (L, torch.tensor(rows, device=dev), torch.cat(poss), torch.cat(vals).to(dev))
            with torch.autocast(**auto):
                with _LayerCapture(body, al if mode in ("force", "align") else (), force) as cap:
                    hidden = body(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                                  use_cache=False).last_hidden_state
                zs = _letters(model, tokenizer, head, hidden[torch.arange(len(idx), device=dev), b["last"]], exs, {})
            for r, i in enumerate(idx):
                probs_by[i] = torch.softmax(zs[r], -1).tolist()
            if al and mode in ("force", "align") and own:
                ia = [layers.index(li) for li in al]
                S = torch.stack([torch.stack([cap.h[li][r, p] for li in al]) for r, p, _, _ in own]).float().cpu()
                T = torch.stack([t[ia] for _, _, t, _ in own])
                aligned += list(zip([j for _, _, _, j in own], _align_rows(S, T).tolist()))
        if mode != "align":
            mm, pr = _metrics_preds(items, probs_by)
            res[mode] = (dict(mm, n_forced=n_forced), pr)
        if al and mode in ("force", "align") and "align" not in res:
            a = {"n_slots": len(aligned), "align": sum(x for _, x in aligned) / len(aligned) if aligned else None,
                 "layers": list(al)}
            for j in sorted({j for j, _ in aligned}):
                xs = [x for jj, x in aligned if jj == j]
                a[f"align_slot{j + 1}"] = sum(xs) / len(xs)
            res["align"] = a
    if was_training:
        model.train()
    return res


def ptad_eval(model, tokenizer, items: list[dict], *, max_len: int, bs: int = 8, layers=()) -> dict:
    """PTAD's held-out alignment: for every item with a chain, the alignment loss (_align_rows, unweighted) of the
    student's "Answer:" states (the A2 prompt) at layers against the teacher view's (its own worked steps); {"n",
    "align"} (mean over items). Under FSDP every rank runs every item inside the root's forward."""
    if getattr(model, "cotd_readout", False):
        return model.run(_ptad_eval, tokenizer, items, max_len, bs, tuple(layers))
    return _ptad_eval(model, tokenizer, items, max_len, bs, tuple(layers))


@torch.no_grad()
def _ptad_eval(model, tokenizer, items, max_len, bs, layers) -> dict:
    layers = tuple(sorted(set(layers)))
    was_training = model.training
    model.eval()
    body, head = text_parts(model)
    dev = head.weight.device
    auto = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")
    chained = [it for it in items if worked_steps(it)]
    order = sorted(range(len(chained)), key=lambda i: len(chained[i]["prompt"]))
    losses = []
    for s in range(0, len(order), bs):
        its = [chained[i] for i in order[s:s + bs]]
        hs = []
        for teacher in (True, False):
            exs = [_item_example(tokenizer, it, teacher_text_marks(FINAL_TEMPLATE.format(problem=it["prompt"]),
                                                                   teacher_plan(it, "steps"))[0] if teacher
                                 else FINAL_TEMPLATE.format(problem=it["prompt"])) for it in its]
            with torch.autocast(**auto):
                b = collate(tokenizer, exs, max_len, dev)
                with _LayerCapture(body, layers) as cap:
                    body(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False)
            hs.append(torch.stack([torch.stack([cap.h[li][r, b["last"][r]] for li in layers])
                                   for r in range(len(its))]).float().cpu())
        losses += _align_rows(hs[1], hs[0]).tolist()
    if was_training:
        model.train()
    return {"n": len(losses), "align": sum(losses) / len(losses) if losses else None, "layers": list(layers)}


def _num(x) -> float | None:
    try:
        return float(str(x).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def option_has_intermediate(item: dict) -> bool:
    """Whether an option of the item equals one of its worked steps' values other than the last (the answer's): the
    'partial value' items of the memo's G1 split."""
    vals = slot_values(item)[:-1]
    want = {v for v in (_num(x) for x in vals) if v is not None}
    if not want:
        return False
    opts = []
    for line in item["prompt"].rpartition("\nOptions:\n")[2].split("\n"):
        m = _OPT_LINE.match(line.strip())
        if m:
            opts.append(_num(m.group(2)))
    return any(o is not None and any(abs(o - w) < 1e-9 for w in want) for o in opts)


def _acc_of(preds: list[dict], ids: set) -> tuple[int, float | None]:
    xs = [p for p in preds if p["item_id"] in ids and p.get("gold_label") in p["labels"]]
    if not xs:
        return 0, None
    ok = sum(max(range(len(p["probs"])), key=lambda i: p["probs"][i]) == p["labels"].index(p["gold_label"]) for p in xs)
    return len(xs), ok / len(xs)


def tsf_subsets(items: list[dict], preds: list[dict], intact: list[dict], spec: SlotSpec) -> dict:
    """A TSF necessity mode's accuracy on the items with a chain (the only ones a force mode changes) and on those with /
    without an intermediate value among the options (option_has_intermediate, chained items), next to the intact
    accuracy on the same items: n_chain, acc_chain, acc_intact_chain, acc_drop_chain, and the same for _optmid /
    _nooptmid."""
    chained = {it["item_id"] for it in items
               if teacher_plan(it, "steps", k=spec.k, overflow=spec.overflow).lines}
    mid = {it["item_id"] for it in items if it["item_id"] in chained and option_has_intermediate(it)}
    out = {}
    for tag, ids in (("chain", chained), ("optmid", mid), ("nooptmid", chained - mid)):
        n, a = _acc_of(preds, ids)
        _, b = _acc_of(intact, ids)
        out.update({f"n_{tag}": n, f"acc_{tag}": a, f"acc_intact_{tag}": b,
                    f"acc_drop_{tag}": None if a is None or b is None else b - a})
    return out
