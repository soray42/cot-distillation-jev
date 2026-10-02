"""Early-exit evaluation of a one-pass decision model: how good is the decision read after only L of its layers, and
how much faster is that forward pass?

One forward per item collects the readout-position state after every layer (hidden_states, index L = after L decoder
layers; the last index is already final-normed). Then, per exit layer L:
  lens    logit lens: the model's own final norm and LM head applied to the layer-L state, letters only, no training
  tuned   tuned lens (Belrose et al. 2023): an affine map from the normed layer-L state to the final state, fitted by
          ridge regression on the fit sets' states alone (no labels), then the model's own LM head
For both: accuracy, agreement of the answer with the full model, ECE. Latency: a forward hook stops the pass after
layer L (batch size 1, median over --timing-items prompts), next to the full pass.

  python scripts/early_exit_eval.py --model runs/TF-v4t-.../model --out runs/early-exit-TF-v4t \\
      --fit val=data/student_v4t/val.jsonl v4heldout=data/eval/v4_heldout.jsonl --eval bbh=data/eval/bbh.jsonl ...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch  # noqa: E402

from cotdistill.student import FINAL_TEMPLATE, Example, collate, letter_token_ids, load_model, text_parts  # noqa: E402


def read(path: str, limit: int) -> list[dict]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r.get("gold_label") in r["labels"]]
    return rows[:limit] if limit else rows


@torch.no_grad()
def collect(model, tok, items: list[dict], max_len: int, bs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """([n, len(hidden_states), d] readout states, [n, d] the model's final readout state), float16 on the CPU, in item
    order. Whether the last hidden_states entry is already final-normed depends on the transformers version; main()
    checks it against the final state."""
    body, head = text_parts(model)
    dev = head.weight.device
    out, fin = [None] * len(items), [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: len(items[i]["prompt"]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        exs = [Example(FINAL_TEMPLATE.format(problem=items[i]["prompt"]), items[i]["labels"],
                       [0.0] * len(items[i]["labels"]), 1.0, "final", items[i]["item_id"]) for i in idx]
        b = collate(tok, exs, max_len, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            res = body(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=True)
        rows = torch.arange(len(idx), device=dev)
        st = torch.stack([h[rows, b["last"]] for h in res.hidden_states], 1).to(torch.float16).cpu()
        fs = res.last_hidden_state[rows, b["last"]].to(torch.float16).cpu()
        for j, i in enumerate(idx):
            out[i], fin[i] = st[j], fs[j]
    return torch.stack(out), torch.stack(fin)


def normed(body, h: torch.Tensor, L: int, n_layers: int, last_is_normed: bool) -> torch.Tensor:
    """The state as the LM head would see it: the final norm applied, unless it is the last entry and already normed."""
    return h if (L == n_layers and last_is_normed) else body.norm(h)


def letter_probs(head, tok, states: torch.Tensor, items: list[dict], cache: dict) -> list[list[float]]:
    out = []
    W = head.weight
    for h, it in zip(states, items):
        key = tuple(it["labels"])
        if key not in cache:
            cache[key] = torch.tensor(letter_token_ids(tok, it["labels"]), device=W.device)
        z = (h.to(W.device, W.dtype) @ W[cache[key]].T).float()
        out.append(torch.softmax(z, -1).tolist())
    return out


def scores(probs: list[list[float]], items: list[dict], ref: list[int] | None) -> dict:
    n = len(items)
    am = [max(range(len(p)), key=p.__getitem__) for p in probs]
    gold = [it["labels"].index(it["gold_label"]) for it in items]
    bins = [[0, 0.0, 0.0] for _ in range(10)]
    for p, a, g in zip(probs, am, gold):
        b = min(9, int(p[a] * 10))
        bins[b][0] += 1; bins[b][1] += p[a]; bins[b][2] += a == g
    out = {"acc": round(sum(a == g for a, g in zip(am, gold)) / n, 4),
           "ece": round(sum(abs(c - k) for _, c, k in bins) / n, 4)}
    if ref is not None:
        out["agree_full"] = round(sum(a == r for a, r in zip(am, ref)) / n, 4)
    return out


def tuned_map(X: torch.Tensor, Y: torch.Tensor, rel_lambda: float = 0.1):
    """Affine ridge map X -> Y in the dual form (n < d), fitted in float32 on the GPU."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    K = Xc @ Xc.T
    lam = rel_lambda * K.diagonal().mean()
    A = torch.linalg.solve(K + lam * torch.eye(len(X), device=X.device), Yc)
    W = Xc.T @ A
    return lambda Z: (Z - mx) @ W + my


@torch.no_grad()
def latency(model, tok, items: list[dict], L: int | None, max_len: int, reps: int = 3) -> float:
    """Median seconds per item at batch size 1 for a forward stopped after layer L (None = full pass)."""
    body, head = text_parts(model)
    dev = head.weight.device

    class Stop(Exception):
        pass

    def hook(m, a, o):
        raise Stop
    h = body.layers[L - 1].register_forward_hook(hook) if L is not None else None
    times = []
    try:
        for it in items:
            b = collate(tok, [Example(FINAL_TEMPLATE.format(problem=it["prompt"]), it["labels"], [0.0] * len(it["labels"]),
                                      1.0, "final", it["item_id"])], max_len, dev)
            best = math.inf
            for _ in range(reps):
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t = time.perf_counter()
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                        body(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
                except Stop:
                    pass
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                best = min(best, time.perf_counter() - t)
            times.append(best)
    finally:
        if h is not None:
            h.remove()
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit", nargs="+", required=True, help="name=path sets whose states fit the tuned lens (no labels used)")
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--layers", default="8,10,12,13,14,15,16,17,18,20,22")
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timing-items", type=int, default=40)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import transformers
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    body, head = text_parts(model)
    n_layers = len(body.layers)
    layers = [int(x) for x in args.layers.split(",")] + [n_layers]
    cache: dict = {}

    fit = [it for spec in args.fit for it in read(spec.split("=", 1)[1], args.limit)]
    Hfit, Ffit = collect(model, tok, fit, args.max_len, args.bs)
    if Hfit.shape[1] != n_layers + 1:
        raise SystemExit(f"hidden_states has {Hfit.shape[1]} entries for {n_layers} layers")
    last_is_normed = bool(torch.allclose(Hfit[:, n_layers].float(), Ffit.float(), atol=1e-2, rtol=1e-2))
    print(f"{n_layers} layers; last hidden_states entry already final-normed: {last_is_normed}", flush=True)
    with torch.no_grad():
        Yfit = Ffit.to(dev).float()
        maps = {L: tuned_map(normed(body, Hfit[:, L].to(dev).float(), L, n_layers, last_is_normed), Yfit)
                for L in layers if L < n_layers}
    del Hfit, Ffit
    report = {"args": vars(args), "n_layers": n_layers, "sets": {}, "latency_s": {}}
    for spec in args.eval:
        name, path = spec.split("=", 1)
        items = read(path, args.limit)
        H, F = collect(model, tok, items, args.max_len, args.bs)
        with torch.no_grad():
            full = letter_probs(head, tok, F.to(dev).float(), items, cache)
            ref = [max(range(len(p)), key=p.__getitem__) for p in full]
            rows = {"full": {"lens": scores(full, items, ref)}}
            for L in layers:
                hL = normed(body, H[:, L].to(dev).float(), L, n_layers, last_is_normed)
                row = {"lens": scores(letter_probs(head, tok, hL, items, cache), items, ref)}
                if L in maps:
                    row["tuned"] = scores(letter_probs(head, tok, maps[L](hL), items, cache), items, ref)
                rows[L] = row
        report["sets"][name] = {"n": len(items), "layers": rows}
        print(name, " ".join(f"L{L}: lens {r['lens']['acc']:.3f}" + (f" tuned {r['tuned']['acc']:.3f}/agree {r['tuned']['agree_full']:.2f}"
                                if 'tuned' in r else "") for L, r in rows.items()), flush=True)
        (out / "early_exit.json").write_text(json.dumps(report, indent=1))
    timing = read(args.eval[0].split("=", 1)[1], args.timing_items)
    for L in layers:
        report["latency_s"][L] = round(latency(model, tok, timing, None if L == n_layers else L, args.max_len), 5)
        print(f"latency L{L}: {report['latency_s'][L] * 1000:.1f} ms", flush=True)
    (out / "early_exit.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
