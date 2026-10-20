"""Evaluate a model (untrained base control or a saved student) on eval sets, final question only.

  python scripts/eval_student.py --model ~/cotd/models/Qwen3.5-2B-Base \
      --eval jevbench=data/eval/jevbench_public.jsonl kk=data/eval/kk_heldout.jsonl --out runs/base-2B

Writes metrics.json and preds_<set>.jsonl; scripts/score_preds.py gives per-group breakdowns.

--time (opt-in): forward-only seconds per item for a practicality table (e.g. StepFinder's seconds per trajectory):
every item of every set alone (batch 1, the readout forward of evaluate, no generation), after one warm-up item,
CUDA-synchronized wall time per item -> timing_<set>.json (mean, median, p90, max, total, prompt tokens, device).
The predictions and metrics are written as without the flag.

Latent-slot models (train_student.py --latent-slots, arms LS / LSU / LSR / LSY): a saved model directory holds
latent_slots.json, and every final question is then built with the same slot line as in training (final_text); the
--latent-slots / --slot-* flags set or override it. --slot-necessity / --slot-decode run the necessity test and the
decoding readout as at the end of training. v2.1 models (arms LSL / LSLU / LSLR / LSLY) keep their distinct tokens,
slot layer, lens readout and fill in latent_slots.json: --slot-decode N then gives the lens readout (slot_readout), and
the necessity mode mean needs --slot-mean-train (training items for the slot means). A model trained with
--slot-overflow truncate / --slot-pad-weight W != 1 records them there too (overflow, pad_weight), so --slot-decode
rebuilds its targets (truncate: the FIRST K steps) without --slot-overflow.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.student import (SLOT_POOL, SLOT_SIDECAR, SlotSpec, choose_slot_token, encode_tail,  # noqa: E402
                                evaluate, evaluate_subq, final_text, load_model, parse_slot_necessity, set_slot_prompt,
                                slot_evals, slot_spec, subq_sets, use_chat_template)


def time_items(model, tok, items: list[dict], max_len: int, cache: dict) -> dict:
    """--time: seconds per item, each item alone (batch 1), after one untimed warm-up item; CUDA synchronized."""
    import time
    import numpy as np
    dev = next(model.parameters()).device
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    evaluate(model, tok, items[:1], max_len, 1, cache)                 # warm-up (kernels, allocator, label ids)
    secs, ntok = [], []
    for it in items:
        sync()
        t = time.perf_counter()
        evaluate(model, tok, [it], max_len, 1, cache)
        sync()
        secs.append(time.perf_counter() - t)
        ntok.append(len(encode_tail(tok, final_text(tok, it["prompt"]), max_len)))
    a = np.array(secs)
    return {"n": len(secs), "s_mean": float(a.mean()), "s_median": float(np.median(a)),
            "s_p90": float(np.percentile(a, 90)), "s_max": float(a.max()), "s_total": float(a.sum()),
            "tokens_mean": float(np.mean(ntok)), "tokens_max": int(max(ntok)), "batch": 1, "max_len": max_len,
            "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
            "dtype": str(next(model.parameters()).dtype), "per_item_s": [round(x, 4) for x in secs]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", nargs="+", required=True, help="name=path.jsonl")
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-layers", default="", help="comma list of decoder layers whose readout states are saved "
                    "with --dump-hidden (shape [n, layers + 1, hidden]; the last entry is the final readout state)")
    ap.add_argument("--dump-hidden", default="", help="comma list of eval-set names whose readout hidden states "
                    "to save as hidden_<name>.pt (float16, preds order)")
    ap.add_argument("--eval-subq", default=None, help="student data file whose sub-questions to score, or "
                    "comma-separated name=path entries (predictions in preds_subq[_name].jsonl)")
    ap.add_argument("--chat-template", action="store_true",
                    help="read out every item as one user turn of the model's chat template, thinking off, at the "
                         "assistant turn's first token (bare letters), as train_student.py --chat-template; off: "
                         "unchanged")
    ap.add_argument("--time", action="store_true", help="also time every item alone (batch 1, forward only): "
                    "timing_<set>.json")
    ap.add_argument("--latent-slots", type=int, default=None,
                    help="slot line of every final question: K slots (default: the model's latent_slots.json, if any; "
                         "0 = none)")
    ap.add_argument("--slot-c", type=int, default=None, help="tokens per slot (default: latent_slots.json, else 1)")
    ap.add_argument("--slot-token", default=None, help="the slot token (default: latent_slots.json, else auto)")
    ap.add_argument("--slot-place", default=None, choices=["before-options", "after-options"],
                    help="slot line placement (default: latent_slots.json, else before-options)")
    ap.add_argument("--slot-overflow", default=None, choices=["merge", "truncate"],
                    help="--slot-decode: how the targets of chains longer than K are shown (as in training; default: "
                         "the overflow in latent_slots.json, else merge)")
    ap.add_argument("--slot-necessity", default="", help="necessity test modes (swap, drop, noise@<layer|mid>:<sigma>; "
                    "train_student.py --slot-necessity)")
    ap.add_argument("--slot-necessity-sets", default="", help="--slot-necessity: comma list of eval sets (default all)")
    ap.add_argument("--slot-swap-token", default="auto", help="--slot-necessity swap: the replacement token")
    ap.add_argument("--slot-decode", type=int, default=0, help="decode every slot of the first N items of "
                    "--slot-decode-set greedily: slot_decode_<set>.jsonl")
    ap.add_argument("--slot-decode-set", default="", help="--slot-decode: the eval set (default: the first --eval set; "
                    "lens models: a comma list)")
    ap.add_argument("--slot-mean-train", default="", help="--slot-necessity mean: training items (jsonl) whose slot "
                    "states give the means")
    ap.add_argument("--slot-mean-n", type=int, default=300, help="--slot-mean-train: the first N items")
    args = ap.parse_args()

    import transformers
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if args.chat_template:
        print("chat template: every item rendered as", repr(use_chat_template(tok, args.model)), flush=True)
    side = Path(args.model) / SLOT_SIDECAR
    sd = json.loads(side.read_text()) if side.exists() else {}
    k = args.latent_slots if args.latent_slots is not None else sd.get("k", 0)
    if args.slot_overflow is None:       # a v2.1 model trained with --slot-overflow truncate records it
        args.slot_overflow = sd.get("overflow", "merge")
    swap = None
    if k:
        if args.chat_template:
            raise SystemExit("latent slots do not combine with --chat-template")
        try:
            token = choose_slot_token(tok, args.slot_token or sd.get("token", "auto"))
            spec = SlotSpec(k, args.slot_c or sd.get("c", 1), token, args.slot_place or sd.get("place", "before-options"))
            if any(x in sd for x in ("tokens", "layer", "readout", "fill", "head_grad", "pad_weight", "overflow")):
                spec = SlotSpec(spec.k, spec.c, spec.token, spec.place, tokens=tuple(sd.get("tokens", ())),  # v2.1
                                layer=sd.get("layer"), readout=sd.get("readout", "decode"), fill=sd.get("fill", "none"),
                                head_grad=sd.get("head_grad", False), pad_weight=sd.get("pad_weight", 1.0),
                                overflow=sd.get("overflow", "merge"))
            set_slot_prompt(tok, spec)
            modes = parse_slot_necessity(args.slot_necessity)
            if "swap" in modes:
                swap = choose_slot_token(tok, args.slot_swap_token, avoid=spec.tokens or (token,),
                                         **({"pool": SLOT_POOL} if spec.tokens else {}))
            if "mean" in modes and spec.layer is None:
                raise ValueError("--slot-necessity mean: the model has no slot layer (use mean@<layer>)")
            if any(m.startswith("mean") for m in modes) and not args.slot_mean_train:
                raise ValueError("--slot-necessity mean: pass --slot-mean-train <train.jsonl>")
        except ValueError as e:
            raise SystemExit(f"latent slots: {e}")
        print(f"latent slots ({'from ' + str(side) if sd else 'flags'}): {spec}", flush=True)
    elif args.slot_necessity or args.slot_decode:
        raise SystemExit("--slot-necessity / --slot-decode need a slot model (latent_slots.json) or --latent-slots")
    model = load_model(args.model, torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
    results, cache = {"args": vars(args), "eval": {}}, {}
    evals = {}
    for kv in args.eval:
        name, path = kv.split("=", 1)
        items = [json.loads(l) for l in open(path) if l.strip()]
        evals[name] = items
        if k and items:
            if any(any(t in it["prompt"] for t in spec.slot_tokens()) or swap and swap in it["prompt"] for it in items):
                raise SystemExit(f"{name}: the slot (or swap) token occurs in a prompt; pick another --slot-token")
            print(name, "final question e.g.", repr(final_text(tok, items[0]["prompt"])), flush=True)
        hid = [] if name in args.dump_hidden.split(",") else None
        m, preds = evaluate(model, tok, items, args.max_len, args.bs, cache, hidden=hid,
                            layers=[int(x) for x in args.dump_layers.split(",") if x] or None)
        if hid is not None:
            torch.save(torch.stack(hid).half(), out / f"hidden_{name}.pt")
        results["eval"][name] = m
        with open(out / f"preds_{name}.jsonl", "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        print(name, json.dumps(m), flush=True)
        if args.time:
            tm = time_items(model, tok, items, args.max_len, cache)
            (out / f"timing_{name}.json").write_text(json.dumps(tm, indent=1))
            results.setdefault("timing", {})[name] = {k: v for k, v in tm.items() if k != "per_item_s"}
            print("timing", name, json.dumps(results["timing"][name]), flush=True)
    if k:
        results["latent_slots"] = slot_spec(tok).as_dict()
        slot_evals(model, tok, evals, results, out, modes=parse_slot_necessity(args.slot_necessity),
                   sets=[x for x in args.slot_necessity_sets.split(",") if x], swap_token=swap,
                   decode_n=args.slot_decode, decode_set=args.slot_decode_set, max_len=args.max_len, bs=args.bs,
                   cache=cache, overflow=args.slot_overflow,
                   mean_items=[json.loads(l) for l in open(args.slot_mean_train) if l.strip()][:args.slot_mean_n]
                   if args.slot_mean_train else None)
    for sname, spath in subq_sets(args.eval_subq or ""):
        items = [json.loads(l) for l in open(spath) if l.strip()]
        sm, spreds = evaluate_subq(model, tok, items, args.max_len, args.bs, cache)
        results["eval"][sname] = sm
        with open(out / f"preds_{sname}.jsonl", "w") as f:
            for p in spreds:
                f.write(json.dumps(p) + "\n")
        print(sname, json.dumps({k: round(v["acc"], 3) for k, v in sm.items()}), flush=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
