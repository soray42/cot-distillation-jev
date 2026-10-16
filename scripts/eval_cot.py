"""Chain-of-thought upper bound of the base model: the post-trained chat model reasons step by step, then names an option
letter. Reported next to the single-pass students (one forward pass, no generated tokens) as accuracy vs inference
tokens.

Each item becomes one user turn: its prompt (question + lettered options, as in data/eval) plus INSTRUCTION ("Think
step by step ... 'Answer: X'"), rendered with the model's own chat template (loaded as scripts/eval_llm_baseline.py
loads it: the processor if there is one, else the tokenizer). Thinking is off by default (the Qwen3.5 template then
writes an empty <think></think> block); --enable-thinking passes enable_thinking=True (the Qwen3.5 template opens
"<think>\\n" and the model thinks before answering). Decoding is greedy (do_sample=False; the generation_config's
temperature / top_p / top_k are overridden with None), batched with left padding (items sorted by prompt length, longest
first), up to --max-new-tokens new tokens; a sequence ends at its first EOS (generation_config eos ids, the tokenizer's
eos and <|im_end|>). Qwen3.5's linear-attention layers zero padded positions during prefill (transformers
apply_mask_to_padding_states); hpc/eval_cot.sbatch's smoke compares bs 1 against a full batch on the same items.

Answer parsing (parse_answer), on the text after the last "</think>" (with --enable-thinking and no "</think>": no
answer, the model never left its thinking):
  1. explicit: the last "Answer: X" / "answer is X" / "Answer: (X)" / "**Answer:** X" or \\boxed{X} whose X is one of
     the item's labels (upper case, not followed by a letter or digit);
  2. otherwise the last standalone label letter on the last non-empty line ("So the correct option is D."), skipping
     "A"/"I" used as a word ("A farmer ...");
  3. otherwise unparsable: counted as wrong, probs uniform.

Outputs in --out (the training runs' layout, so the readout scripts apply):
  preds_<name>.jsonl  item_id, group, labels, probs (one-hot on the parsed letter; uniform if unparsable), gold_label,
                      gold_probs (as scripts/train_student.py writes them), then pred_label (null if unparsable), parsed,
                      parse_method, gen_tokens (generated tokens up to and including the EOS; --max-new-tokens if
                      truncated), truncated, prompt_tokens. NOTE: a uniform row's argmax is its first label; scorers
                      that take argmax(probs) must use `parsed` (or pred_label) to count unparsable rows as wrong.
  gens_<name>.jsonl   item_id and the generated text (special tokens kept, EOS cut), for audits.
  metrics.json        per set: n, acc (unparsable = wrong), unparsable_rate, truncated_rate, gen_tokens mean / median /
                      p90 / max / total, prompt_tokens_mean, wall_s, s_per_item (wall / n: batch-amortised GPU seconds),
                      gen_tok_per_s, decode_steps_per_s (batch decode steps per second = per-sequence decode speed),
                      parse_methods, per-group accuracy.
--limit N keeps the first N items of each set; --shard i/n (0-based) keeps items i, i+n, i+2n, ... (interleaved:
kk_deep is sorted by depth) and writes <file>.shard<i>of<n>.<ext>; --merge (no model, no GPU) combines the n shards of
every --eval set into the unsharded files, wall_s summed over the shards.

  python scripts/eval_cot.py --model models/Qwen3.5-4B --out runs/cot-Qwen3.5-4B \\
      --eval gsm8kv2=data/eval/gsm8k_mc_v2.jsonl gsm1k=data/eval/gsm1k_mc.jsonl [--shard 0/2] [--enable-thinking]
  python scripts/eval_cot.py --merge --shards 2 --out runs/cot-Qwen3.5-4B --eval gsm8kv2=... gsm1k=...
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

INSTRUCTION = ("\n\nThink step by step. Then give your final answer on the last line in the form \"Answer: X\", "
               "where X is the letter of the correct option.")
NO_THINK_CLOSE = "\n\n</think>\n\n"          # an empty think block, as Qwen's no-think mode renders it
THINK_END = "</think>"

_END = r"(?![A-Za-z0-9])"
ANSWER_RE = re.compile(r"(?i:\banswer\b)(?:\s+(?i:is))?\s*[:：]?\s*(?:(?i:option|choice)\s*)?[\*_\(\[\s]*([A-Z])" + _END)
BOXED_RE = re.compile(r"\\boxed\s*\{\s*(?:\\text(?:bf)?\s*\{\s*)?\(?([A-Z])" + _END + r"[^{}]*\}")
LETTER_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z])" + _END)


# ---- prompt -------------------------------------------------------------------------------------------------------
def render(proc, prompt: str, typed: bool, thinking: bool) -> str:
    """The item as one user turn ending where the assistant's turn starts. typed: content as typed parts (multimodal
    processor templates). Thinking off: a template that opens a think block anyway gets it closed empty (as
    scripts/eval_llm_baseline.py)."""
    text = prompt + INSTRUCTION
    content = [{"type": "text", "text": text}] if typed else text
    out = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=thinking)
    if not thinking and out.rstrip().endswith("<think>"):
        out = out.rstrip() + NO_THINK_CLOSE
    return out


def eos_ids(model, tok) -> list[int]:
    """generation_config eos ids, the tokenizer's eos and <|im_end|> (if a single known token)."""
    ids = set()
    ge = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    ids.update(ge if isinstance(ge, (list, tuple)) else [ge] if ge is not None else [])
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != getattr(tok, "unk_token_id", None):
        ids.add(im_end)
    return sorted(i for i in ids if i is not None)


# ---- parsing --------------------------------------------------------------------------------------------------------
def answer_region(text: str, thinking: bool) -> str | None:
    """The text after the last </think>; with thinking on and no </think>, None (no answer was given)."""
    if THINK_END in text:
        return text.rsplit(THINK_END, 1)[1]
    return None if thinking else text


def parse_answer(text: str, labels: list[str], thinking: bool = False) -> tuple[str | None, str]:
    """(letter or None, method): explicit (last Answer:/answer is/boxed match among labels), last_line (last standalone
    label letter on the last non-empty line), or none / no_think_end."""
    region = answer_region(text, thinking)
    if region is None:
        return None, "no_think_end"
    labs = set(labels)
    hits = [(m.start(), m.group(1)) for rx in (ANSWER_RE, BOXED_RE) for m in rx.finditer(region)
            if m.group(1) in labs]
    if hits:
        return max(hits)[1], "explicit"
    lines = [l for l in region.split("\n") if l.strip()]
    if lines:
        last = lines[-1]
        cands = [m for m in LETTER_RE.finditer(last) if m.group(1) in labs
                 and not (m.group(1) in "AI" and re.match(r" [a-z]", last[m.end():m.end() + 2]))]
        if cands:
            return cands[-1].group(1), "last_line"
    return None, "none"


# ---- generation -----------------------------------------------------------------------------------------------------
def count_generated(new: list[int], eos: set[int], max_new: int) -> tuple[int, bool, list[int]]:
    """(generated tokens incl. the first EOS, truncated, content ids before the EOS). Pads after the EOS are not
    counted; no EOS means the sequence ran to max_new."""
    for j, t in enumerate(new):
        if t in eos:
            return j + 1, False, new[:j]
    return max_new, True, new


def generate_all(model, tok, texts: list[str], bs: int, max_new: int, eos: list[int], log=print,
                 tag: str = "") -> tuple[list[dict], dict]:
    """Greedy generations for texts (input order) and the batch counters. Left padding; batches of bs sorted by prompt
    length, longest first (an out-of-memory error shows in the first batch)."""
    import torch
    dev = next(model.parameters()).device
    tok.padding_side = "left"
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    n_tok = [len(tok.encode(t, add_special_tokens=False)) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: (-n_tok[i], i))
    res: list = [None] * len(texts)
    eset = set(eos)
    t0, steps, useful, nb = time.time(), 0, 0, 0
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
        ids, mask = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
        with torch.inference_mode():
            out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new, do_sample=False,
                                 temperature=None, top_p=None, top_k=None, eos_token_id=eos, pad_token_id=pad)
        new = out[:, ids.shape[1]:].tolist()
        steps += out.shape[1] - ids.shape[1]
        for r, i in enumerate(idx):
            n, trunc, content = count_generated(new[r], eset, max_new)
            useful += n
            res[i] = {"gen_tokens": n, "truncated": trunc, "prompt_tokens": n_tok[i],
                      "text": tok.decode(content, skip_special_tokens=False)}
        nb += 1
        if nb % 10 == 0 or s + bs >= len(order):
            dt = time.time() - t0
            log(f"{tag}: {min(s + bs, len(order))}/{len(order)} items, {dt:.0f} s, {steps / max(dt, 1e-9):.1f} decode "
                f"steps/s, {useful / max(dt, 1e-9):.0f} generated tok/s", flush=True)
    return res, {"batches": nb, "decode_steps": steps, "wall_s": time.time() - t0}


# ---- records and metrics ------------------------------------------------------------------------------------------
def pred_row(it: dict, g: dict, thinking: bool) -> dict:
    labels = it["labels"]
    letter, how = parse_answer(g["text"], labels, thinking)
    probs = [1.0 if L == letter else 0.0 for L in labels] if letter else [1.0 / len(labels)] * len(labels)
    return {"item_id": it["item_id"], "group": it.get("group"), "labels": labels, "probs": probs,
            "gold_label": it.get("gold_label"), "gold_probs": it.get("gold_probs"), "pred_label": letter,
            "parsed": letter is not None, "parse_method": how, "gen_tokens": g["gen_tokens"],
            "truncated": g["truncated"], "prompt_tokens": g["prompt_tokens"]}


def p90(xs: list[float]) -> float:
    s = sorted(xs)
    return s[max(0, math.ceil(0.9 * len(s)) - 1)]


def set_metrics(preds: list[dict], wall_s: float, decode_steps: int | None = None) -> dict:
    n = len(preds)
    scored = [p for p in preds if p.get("gold_label") in p["labels"]]
    right = [p["parsed"] and p["pred_label"] == p["gold_label"] for p in scored]
    toks = [p["gen_tokens"] for p in preds]
    groups = defaultdict(list)
    for p, ok in zip(scored, right):
        groups[p.get("group") or "?"].append(ok)
    m = {"n": n, "n_scored": len(scored), "acc": sum(right) / len(scored) if scored else None,
         "n_correct": sum(right), "unparsable_rate": sum(not p["parsed"] for p in preds) / n,
         "truncated_rate": sum(p["truncated"] for p in preds) / n,
         "unparsable_truncated": sum((not p["parsed"]) and p["truncated"] for p in preds),
         "gen_tokens": {"mean": statistics.mean(toks), "median": statistics.median(toks), "p90": p90(toks),
                        "max": max(toks), "total": sum(toks)},
         "prompt_tokens_mean": statistics.mean(p["prompt_tokens"] for p in preds),
         "wall_s": round(wall_s, 1), "s_per_item": wall_s / n, "gen_tok_per_s": sum(toks) / max(wall_s, 1e-9),
         "parse_methods": dict(Counter(p["parse_method"] for p in preds))}
    if decode_steps is not None:
        m["decode_steps"] = decode_steps
        m["decode_steps_per_s"] = decode_steps / max(wall_s, 1e-9)
    m["groups"] = {k: {"n": len(v), "acc": sum(v) / len(v)} for k, v in sorted(groups.items())}
    return m


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def suffix(shard: tuple[int, int] | None) -> str:
    return f".shard{shard[0]}of{shard[1]}" if shard else ""


def parse_shard(s: str) -> tuple[int, int] | None:
    if not s:
        return None
    i, n = (int(x) for x in s.split("/"))
    if not 0 <= i < n:
        raise SystemExit(f"--shard {s}: need 0 <= i < n")
    return i, n


# ---- driver ---------------------------------------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="models/Qwen3.5-4B", help="local directory of the post-trained chat model")
    ap.add_argument("--eval", nargs="+", required=True, help="name=path.jsonl (data/eval format)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=16, help="sequences per generate call")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="enable_thinking=True in the chat template (Qwen3.5: the model thinks in <think> first)")
    ap.add_argument("--limit", type=int, default=0, help="first N items of each set (0 = all)")
    ap.add_argument("--shard", default="", help="i/n (0-based): items i, i+n, ... of each set; shard-suffixed files")
    ap.add_argument("--merge", action="store_true", help="combine the --shards shard files of every --eval set (no model)")
    ap.add_argument("--shards", type=int, default=0, help="--merge: number of shards n")
    return ap.parse_args(argv)


def load_sets(specs: list[str], limit: int) -> list[tuple[str, list[dict]]]:
    out = []
    for spec in specs:
        name, path = spec.split("=", 1)
        items = read_jsonl(path)
        out.append((name, items[:limit] if limit else items))
    return out


def run(args, model, proc, tok, typed: bool, log=print) -> dict:
    """Generate, parse and write every --eval set with a loaded model (tests pass fakes)."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard = parse_shard(args.shard)
    sfx = suffix(shard)
    eos = eos_ids(model, tok)
    results = {"args": vars(args), "eos_ids": eos, "eval": {}}
    if args.enable_thinking:
        probe = render(proc, "Question: x?", typed, True)
        if not probe.rstrip().endswith("<think>"):
            log("note: --enable-thinking given but the template did not open a think block; the parse reads the whole "
                "output only after a </think>", flush=True)
    shown = False
    for name, items in load_sets(args.eval, args.limit):
        if shard:
            items = items[shard[0]::shard[1]]
        if not items:
            continue
        t = time.time()
        texts = [render(proc, it["prompt"], typed, args.enable_thinking) for it in items]
        if not shown:
            log("first rendered prompt:", repr(texts[0]), flush=True)
        gens, cnt = generate_all(model, tok, texts, args.bs, args.max_new_tokens, eos, log=log, tag=name)
        if not shown:
            log("its generation:", repr(gens[0]["text"][-600:]), flush=True)
            shown = True
        preds = [pred_row(it, g, args.enable_thinking) for it, g in zip(items, gens)]
        m = set_metrics(preds, time.time() - t, cnt["decode_steps"])
        m["batches"] = cnt["batches"]
        results["eval"][name] = m
        write_jsonl(out / f"preds_{name}{sfx}.jsonl", preds)
        write_jsonl(out / f"gens_{name}{sfx}.jsonl", [{"item_id": p["item_id"], "pred_label": p["pred_label"],
                                                        "parse_method": p["parse_method"], "gen_tokens": p["gen_tokens"],
                                                        "truncated": p["truncated"], "text": g["text"]}
                                                       for p, g in zip(preds, gens)])
        log(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
                              if k not in ("groups", "gen_tokens")}), "gen_tokens", json.dumps(m["gen_tokens"]),
            flush=True)
        (out / f"metrics{sfx}.json").write_text(json.dumps(results, indent=1))
    return results


def merge(args, log=print) -> dict:
    """Unsharded preds/gens/metrics from the args.shards shard files of every --eval set (file order restored)."""
    out, n = Path(args.out), args.shards
    if n < 1:
        raise SystemExit("--merge needs --shards n")
    shard_metrics = []
    for i in range(n):
        p = out / f"metrics{suffix((i, n))}.json"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        shard_metrics.append(json.loads(p.read_text()))
    results = {"args": vars(args), "merged_from": n, "eos_ids": shard_metrics[0].get("eos_ids"), "eval": {},
               "shard_args": shard_metrics[0]["args"]}
    for name, items in load_sets(args.eval, shard_metrics[0]["args"].get("limit", 0)):
        pos = {it["item_id"]: k for k, it in enumerate(items)}
        preds, gens, wall, steps = [], [], 0.0, 0
        for i in range(n):
            pf, gf = out / f"preds_{name}{suffix((i, n))}.jsonl", out / f"gens_{name}{suffix((i, n))}.jsonl"
            if not pf.exists():
                raise SystemExit(f"missing {pf}")
            preds += read_jsonl(pf)
            gens += read_jsonl(gf)
            m = shard_metrics[i]["eval"].get(name)
            if m is None:
                raise SystemExit(f"shard {i}/{n} metrics have no set {name}")
            wall += m["wall_s"]
            steps += m.get("decode_steps", 0)
        if sorted(p["item_id"] for p in preds) != sorted(pos):
            raise SystemExit(f"{name}: the shards do not cover the set exactly once")
        preds.sort(key=lambda p: pos[p["item_id"]])
        gens.sort(key=lambda g: pos[g["item_id"]])
        m = set_metrics(preds, wall, steps)
        m["wall_s_is_sum_over_shards"] = True
        results["eval"][name] = m
        write_jsonl(out / f"preds_{name}.jsonl", preds)
        write_jsonl(out / f"gens_{name}.jsonl", gens)
        log(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
                              if k not in ("groups", "gen_tokens")}), flush=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    return results


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.merge:
        merge(args)
        return
    import torch
    import transformers
    from cotdistill.student import load_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:                                                # Qwen3.5 ships its chat template with the processor
        proc = transformers.AutoProcessor.from_pretrained(args.model)
    except Exception:
        proc = transformers.AutoTokenizer.from_pretrained(args.model)
    typed = isinstance(proc, transformers.ProcessorMixin)
    tok = proc.tokenizer if typed else proc
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.model, torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    print(f"model class {type(model).__name__}, template from {type(proc).__name__}, device {dev}, "
          f"thinking {'on' if args.enable_thinking else 'off'}, max_new_tokens {args.max_new_tokens}, bs {args.bs}",
          flush=True)
    # temperature / top_p / top_k are overridden per call; a repetition or presence penalty here would still apply
    print("generation_config (penalties, if any, change the greedy argmax):", model.generation_config, flush=True)
    run(args, model, proc, tok, typed)


if __name__ == "__main__":
    main()
