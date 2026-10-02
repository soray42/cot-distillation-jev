"""System-1 LLM baseline: a post-trained chat model answers our multiple-choice items directly, thinking off, one pass.

Registered in notes/prereg_release.md (addendum 2026-10-02 19:35 UTC) for Qwen3.5-2B and, later, Qwen3.5-4B.

Each item becomes one user message: its prompt plus "Answer with the letter of the correct option only." The message
is rendered with the model's own chat template, thinking off. The readout is the next-token distribution at the end of
the rendered prompt, which is the start of the assistant turn. Each option letter takes the bare ("A") and the spaced
(" A") token, where single tokens, combined by logsumexp; a softmax over the item's letters gives its probabilities.

The share of the full next-token distribution that falls on these letter tokens is recorded per item and per set
(letter_mass). A low value means the model wants to open its answer with something else, and then the letter readout
is not a fair test of it.

Sequences are right-padded and read at the last real token (cotdistill.student.collate). Qwen3.5's linear-attention
layers are not relied on to honour padding masks: with right padding, causality alone makes the readout exact.

Writes <out>/preds_<set>.jsonl (probabilities over our letters, in it["labels"] order) and <out>/metrics.json (per set:
calibration metrics, per-group accuracy, letter mass, truncation count), like scripts/eval_student.py, so
scripts/release_readout.py applies unchanged.

  python scripts/eval_llm_baseline.py --model models/Qwen3.5-2B --bs 4 --out runs/reason-qwen2b-nothink \\
      --eval xkk=data/eval/xkk.jsonl proverqa=data/eval/proverqa.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.metrics import calibration  # noqa: E402
from cotdistill.student import Example, collate, load_model, text_parts  # noqa: E402

INSTRUCTION = "\n\nAnswer with the letter of the correct option only."
NO_THINK_CLOSE = "\n\n</think>\n\n"          # an empty think block, as Qwen's no-think mode renders it


def render(proc, prompt: str, typed: bool) -> str:
    """The item as one user turn with thinking off, ending where the assistant's answer starts. With typed, the
    content is a list of typed parts (the form multimodal processor templates expect), else a plain string."""
    text = prompt + INSTRUCTION
    content = [{"type": "text", "text": text}] if typed else text
    out = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    # A template without the enable_thinking switch ignores it (Jinja drops unknown variables); if it opened a think
    # block anyway, close it empty.
    if out.rstrip().endswith("<think>"):
        out = out.rstrip() + NO_THINK_CLOSE
    return out


def letter_variants(tok, letters: list[str]) -> list[list[tuple[int, bool]]]:
    """Per letter, the single-token ids of "A" and " A", each with a flag for the spaced form."""
    out = []
    for L in letters:
        var = []
        for spaced, s in ((False, L), (True, " " + L)):
            t = tok.encode(s, add_special_tokens=False)
            if len(t) == 1 and t[0] not in (v for v, _ in var):
                var.append((t[0], spaced))
        if not var:
            raise ValueError(f"neither '{L}' nor ' {L}' is a single token")
        out.append(var)
    return out


@torch.no_grad()
def score(model, tok, texts: list[str], labels: list[list[str]], n_tok: list[int], max_len: int,
          bs: int) -> list[dict]:
    """Per item: letter probabilities (labels order), letter mass, its spaced-token part and the five most likely
    next tokens (id, probability). Batched by token length; results come back in input order."""
    body, head = text_parts(model)
    dev = head.weight.device
    order = sorted(range(len(texts)), key=n_tok.__getitem__)
    out, cache = [None] * len(texts), {}
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        batch = [Example(texts[i], labels[i], [0.0] * len(labels[i]), 1.0, "final", str(i)) for i in idx]
        b = collate(tok, batch, max_len, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            res = body(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False)
        hidden = res.last_hidden_state
        h = hidden[torch.arange(len(idx), device=dev), b["last"]].to(head.weight.dtype)
        logits = (h @ head.weight.T).float()
        if getattr(head, "bias", None) is not None:
            logits = logits + head.bias.float()
        logp = torch.log_softmax(logits, -1)            # full-vocabulary log-probabilities at the readout position
        for r, i in enumerate(idx):
            key = tuple(labels[i])
            if key not in cache:
                cache[key] = letter_variants(tok, labels[i])
            per_letter = torch.stack([torch.logsumexp(logp[r, [t for t, _ in var]], 0) for var in cache[key]])
            spaced = [t for var in cache[key] for t, sp in var if sp]
            top = logp[r].topk(5)
            out[i] = {"probs": torch.softmax(per_letter, -1).tolist(),
                      "letter_mass": per_letter.exp().sum().item(),
                      "spaced_mass": logp[r, spaced].exp().sum().item() if spaced else 0.0,
                      "top5": list(zip(top.indices.tolist(), top.values.exp().tolist()))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local directory of the post-trained chat model")
    ap.add_argument("--eval", nargs="+", required=True, help="name=path.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=6144, help="token cap; longer rendered prompts keep their end")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N items of each set")
    args = ap.parse_args()

    import transformers
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:                                                # Qwen3.5 ships its chat template with the processor
        proc = transformers.AutoProcessor.from_pretrained(args.model)
    except Exception:
        proc = transformers.AutoTokenizer.from_pretrained(args.model)
    typed = isinstance(proc, transformers.ProcessorMixin)
    tok = proc.tokenizer if typed else proc
    model = load_model(args.model, torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    print(f"model class {type(model).__name__}, template from {type(proc).__name__}, device {dev}", flush=True)

    results, shown = {"args": vars(args), "eval": {}}, False
    for spec in args.eval:
        name, path = spec.split("=", 1)
        items = [json.loads(l) for l in open(path) if l.strip()]
        if args.limit:
            items = items[:args.limit]
        t = time.time()
        texts = [render(proc, it["prompt"], typed) for it in items]
        if not shown:
            print("first rendered prompt:", repr(texts[0]), flush=True)
        n_tok = [len(tok.encode(x, add_special_tokens=False)) for x in texts]
        scored = score(model, tok, texts, [it["labels"] for it in items], n_tok, args.max_len, args.bs)
        if not shown:                                   # what the model wants to say first, for the job log
            print("its top-5 next tokens:", [(tok.decode([i]), round(q, 4)) for i, q in scored[0]["top5"]], flush=True)
            shown = True

        preds, probs, gold, groups = [], [], [], defaultdict(list)
        for it, sc in zip(items, scored):
            preds.append({"item_id": it["item_id"], "group": it.get("group"), "labels": it["labels"],
                          "probs": sc["probs"], "gold_label": it.get("gold_label"), "gold_probs": it.get("gold_probs"),
                          "letter_mass": sc["letter_mass"]})
            if it.get("gold_label") in it["labels"]:
                g = it["labels"].index(it["gold_label"])
                probs.append(sc["probs"])
                gold.append(g)
                groups[it.get("group") or "?"].append(max(range(len(sc["probs"])), key=sc["probs"].__getitem__) == g)
        m = calibration(probs, gold) if gold else {"n": 0}
        m["groups"] = {k: {"n": len(v), "acc": sum(v) / len(v)} for k, v in sorted(groups.items())}
        mass = sum(sc["letter_mass"] for sc in scored)
        m["letter_mass"] = mass / len(scored)
        m["spaced_share"] = sum(sc["spaced_mass"] for sc in scored) / max(mass, 1e-12)
        m["n_truncated"] = sum(n > args.max_len for n in n_tok)
        m["seconds"] = round(time.time() - t, 1)
        results["eval"][name] = m
        with open(out / f"preds_{name}.jsonl", "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        print(name, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items() if k != "groups"}),
              flush=True)
        (out / "metrics.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
