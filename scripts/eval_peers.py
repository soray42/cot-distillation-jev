"""Score open peer decision models on our evaluation sets, each through its own prompt layout and readout.

  decider  Mapika/decider-2b (v11, Apache-2.0): the prompt is built by its own decider/prompt.py (plain state-first layout,
           options in our order, no shuffling, wide labels allowed so no option is dropped); option-label logits from the
           LM head at the "Answer: (" slot; probabilities at its per-type temperature (decider_config.json).
  strands  StrandsAgents/strands-decider-2B-hobson-v19 (Apache-2.0): its own engine (strands_decider.infer.load_engine)
           with the pointer head and per-kind temperature from its config; yes/no items (label names true/false) are
           asked as NoulQuestion, every other item as ChoiceQuestion with our option texts as the option names. The
           checkpoint's base_model is pointed at the local Qwen3.5-2B-Base copy (--base).

Each of our items is split into a state (the prompt before its last "Question:" line), the question text and the
options. Predictions go to <out>/preds_<set>.jsonl in the format of scripts/train_student.py (probabilities over our
letters), and metrics to <out>/metrics.json, so the analysis scripts apply unchanged.

  python scripts/eval_peers.py --peer decider --model models/decider-2b-v11 --out runs/peer-decider \\
      --eval bbh=data/eval/bbh.jsonl musr=data/eval/musr.jsonl
  python scripts/eval_peers.py --peer strands --model models/strands-decider-2B-v19 --base models/Qwen3.5-2B-Base \\
      --strands-src third_party/strands-decider/src --out runs/peer-strands --eval ...
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def split_item(it: dict) -> tuple[str, str, list[str]]:
    """(state, question, options) of one of our evaluation items."""
    head, _, tail = it["prompt"].rpartition("\nOptions:\n")
    opts = []
    for ln in tail.splitlines():
        m = re.match(r"^\(?([A-Z])\) (.*)$", ln)            # "A) text" (ours) or "(A) text" (BBH)
        if m:
            opts.append(m.group(2))
    if len(opts) != len(it["labels"]):
        raise ValueError(f"{it['item_id']}: parsed {len(opts)} options, item has {len(it['labels'])} labels")
    lines = head.rstrip().split("\n")
    qi = max((i for i, ln in enumerate(lines) if ln.startswith("Question:")), default=None)
    if qi is None:
        return head.strip(), "Which option is correct?", opts
    question = " ".join([lines[qi][len("Question:"):].strip()] + [ln.strip() for ln in lines[qi + 1:] if ln.strip()])
    return "\n".join(lines[:qi]).strip(), question, opts


def yes_no_letters(it: dict) -> tuple[str, str] | None:
    """(letter of true/yes, letter of false/no) for a yes/no item, else None."""
    raw = it.get("label_names")
    if not isinstance(raw, dict):                      # some sets store option names as a list
        return None
    names = {L: str(n).lower() for L, n in raw.items()}
    t = [L for L, n in names.items() if n in ("true", "yes")]
    f = [L for L, n in names.items() if n in ("false", "no")]
    if it.get("type") == "noul" and len(t) == 1 and len(f) == 1 and len(it["labels"]) == 2:
        return t[0], f[0]
    return None


def calibration(probs: list[list[float]], gold: list[int]) -> dict:
    n = len(gold)
    acc = sum(max(range(len(p)), key=p.__getitem__) == g for p, g in zip(probs, gold)) / n
    nll = -sum(math.log(max(p[g], 1e-12)) for p, g in zip(probs, gold)) / n
    bins = [[0, 0.0, 0.0] for _ in range(10)]
    for p, g in zip(probs, gold):
        k = max(range(len(p)), key=p.__getitem__)
        b = min(9, int(p[k] * 10))
        bins[b][0] += 1; bins[b][1] += p[k]; bins[b][2] += k == g
    ece = sum(abs(c - a) for _, c, a in bins) / n
    return {"n": n, "acc": acc, "nll": nll, "ece": ece}


class DeciderPeer:
    def __init__(self, path: str, max_ctx_tokens: int):
        import torch
        import transformers
        sys.path.insert(0, path)                      # the model repo ships its own `decider` package
        from decider.prompt import MAX_OPTIONS, build, label_table
        from cotdistill.student import load_model, text_parts
        self.torch, self.build, self.MAX = torch, build, MAX_OPTIONS
        self.tok = transformers.AutoTokenizer.from_pretrained(path)
        self.labels = label_table(self.tok)[1]
        cfg = json.load(open(Path(path) / "decider_config.json"))
        self.T, self.T_by = cfg.get("temperature", 1.0), cfg.get("temperature_by_type", {})
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_model(path, torch.bfloat16 if self.dev == "cuda" else torch.float32).to(self.dev).eval()
        self.body, self.head = text_parts(self.model)
        self.max_ctx = max_ctx_tokens

    def score(self, items: list[dict], bs: int) -> list[list[float]]:
        torch = self.torch

        class NoShuffle:                              # keep our option order (letters map 1:1)
            def shuffle(self, x): pass
            def sample(self, xs, k): return xs[:k]
        built = []
        for it in items:
            state, q, opts = split_item(it)
            ex = SimpleNamespace(context=state, qs=[SimpleNamespace(text=q, options=opts, gold=0)])
            b = self.build(ex, self.tok, rng=NoShuffle(), max_options=self.MAX, max_ctx_tokens=self.max_ctx)
            assert b["perms"][0] == list(range(len(opts)))
            t = it.get("type") or "choice"
            built.append((b["ids"], b["slots"][0], len(opts), self.T_by.get(t, self.T)))
        out = [None] * len(items)
        order = sorted(range(len(items)), key=lambda i: len(built[i][0]))
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        with torch.inference_mode():
            for s in range(0, len(order), bs):
                idx = order[s:s + bs]
                L = max(len(built[i][0]) for i in idx)
                ids = torch.full((len(idx), L), pad, dtype=torch.long)
                mask = torch.zeros((len(idx), L), dtype=torch.long)
                for r, i in enumerate(idx):
                    x = built[i][0]
                    ids[r, :len(x)] = torch.tensor(x)
                    mask[r, :len(x)] = 1
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev == "cuda"):
                    h = self.body(input_ids=ids.to(self.dev), attention_mask=mask.to(self.dev)).last_hidden_state
                for r, i in enumerate(idx):
                    _, slot, n, T = built[i]
                    w = self.head.weight[torch.tensor(self.labels[:n], device=h.device)]
                    z = (h[r, slot].to(w.dtype) @ w.T).float() / T
                    out[i] = torch.softmax(z, -1).tolist()
        return out


class StrandsPeer:
    def __init__(self, path: str, base: str, src: str, work: Path):
        import torch
        sys.path.insert(0, src)
        from strands_decider.infer import load_engine
        from strands_decider.prompting import render_question, render_state
        from strands_decider.schema import ChoiceQuestion, NoulQuestion
        self.torch, self.rq, self.rs = torch, render_question, render_state
        self.Choice, self.Noul = ChoiceQuestion, NoulQuestion
        local = work / "strands_ckpt"                 # the checkpoint with base_model pointed at the local torso copy
        if local.exists():
            shutil.rmtree(local)
        shutil.copytree(path, local, ignore=shutil.ignore_patterns("eval", "training", ".cache"))
        for name in ("strands_decider_config.json", "hobson_config.json"):
            f = local / name
            if f.exists():
                cfg = json.load(open(f))
                cfg["base_model"] = str(Path(base).resolve())
                json.dump(cfg, open(f, "w"), indent=2)
        self.eng = load_engine(str(local), device="cuda" if torch.cuda.is_available() else "cpu", use_prefix_cache=False)

    def score(self, items: list[dict], bs: int) -> list[list[float]]:
        out = []
        with self.torch.inference_mode():
            for it in items:
                state, q, opts = split_item(it)
                yn = yes_no_letters(it)
                if yn:
                    rq = self.rq(self.Noul(instructions=q))
                else:
                    names, seen = [], {}
                    for o in opts:                        # option names must be unique keys
                        seen[o] = seen.get(o, 0) + 1
                        names.append(o if seen[o] == 1 else f"{o} ({seen[o]})")
                    rq = self.rq(self.Choice(instructions=q, criteria={n: None for n in names}))
                probs, _ = self.eng._slot_probs_batched(self.rs(state), [rq.text], [rq.n_slots], [rq.kind], rendered=[rq])
                row = probs[0, :rq.n_slots].float().tolist()
                if yn:
                    p_true = row[rq.slot_labels.index("true")]
                    p = {yn[0]: p_true, yn[1]: 1 - p_true}
                    out.append([p[L] for L in it["labels"]])
                else:
                    by = dict(zip(rq.slot_labels, row))
                    out.append([by[n] for n in names])
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--peer", choices=["decider", "strands"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="models/Qwen3.5-2B-Base")
    ap.add_argument("--strands-src", default="third_party/strands-decider/src")
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-ctx-tokens", type=int, default=1536, help="decider: its default context cap")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N items of each set")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    peer = (DeciderPeer(args.model, args.max_ctx_tokens) if args.peer == "decider"
            else StrandsPeer(args.model, args.base, args.strands_src, out))
    results = {"args": vars(args), "eval": {}}
    for spec in args.eval:
        name, path = spec.split("=", 1)
        items = [json.loads(l) for l in open(path) if l.strip()]
        if args.limit:
            items = items[:args.limit]
        t = time.time()
        probs = peer.score(items, args.bs)
        gold = [it["labels"].index(it["gold_label"]) for it in items]
        m = calibration(probs, gold)
        m["seconds"] = round(time.time() - t, 1)
        results["eval"][name] = m
        with open(out / f"preds_{name}.jsonl", "w") as f:
            for it, p in zip(items, probs):
                f.write(json.dumps({"item_id": it["item_id"], "group": it.get("group"), "labels": it["labels"],
                                    "probs": p, "gold_label": it["gold_label"]}) + "\n")
        print(name, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}), flush=True)
        (out / "metrics.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
