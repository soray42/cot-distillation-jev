"""Weight-space interpolation between the untrained base and a fine-tuned student (WiSE-FT, Wortsman et al. 2022):
theta(alpha) = theta_base + alpha * (theta_ft - theta_base), evaluated at each alpha on the final questions.

Answers whether abilities the fine-tune lost (GSM8K, BBH, reading-style reasoning sets that fall below the base) come
back while the trained gains (K&K, ProverQA, policy) stay. Nothing is trained and no weights are saved. Each alpha
writes <out>/a<alpha>/preds_<set>.jsonl and metrics.json in the format of scripts/eval_student.py.

  python scripts/wise_ft_eval.py --base models/Qwen3.5-2B-Base --ft runs/TF-v4t-.../model --alphas 0.5,0.7,0.85 \\
      --eval gsm8k=data/eval/gsm8k_mc.jsonl bbh=data/eval/bbh.jsonl --out runs/wise-TF-v4t
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cotdistill.student import evaluate, load_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--alphas", default="0.5,0.7,0.85")
    ap.add_argument("--eval", nargs="+", required=True, help="name=path.jsonl")
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import transformers
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = transformers.AutoTokenizer.from_pretrained(args.ft)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.ft, dtype).to(dev).eval()
    ft = {k: v.detach().clone() for k, v in model.state_dict().items()}
    base_sd = load_model(args.base, dtype).state_dict()
    missing = [k for k in ft if k not in base_sd or base_sd[k].shape != ft[k].shape]
    if missing:
        raise SystemExit(f"{len(missing)} tensors do not match between base and fine-tune, e.g. {missing[:3]}")
    base = {k: base_sd[k].to(dev) for k in ft}
    del base_sd
    sets = [(kv.split("=", 1)[0], [json.loads(l) for l in open(kv.split("=", 1)[1]) if l.strip()]) for kv in args.eval]
    for a in [float(x) for x in args.alphas.split(",")]:
        with torch.no_grad():
            model.load_state_dict({k: base[k] + a * (ft[k] - base[k]) if ft[k].is_floating_point() else ft[k]
                                   for k in ft})
        out = Path(args.out) / f"a{a:g}"
        out.mkdir(parents=True, exist_ok=True)
        results, cache = {"args": vars(args), "alpha": a, "eval": {}}, {}
        for name, items in sets:
            m, preds = evaluate(model, tok, items, args.max_len, args.bs, cache)
            results["eval"][name] = m
            with open(out / f"preds_{name}.jsonl", "w") as f:
                for p in preds:
                    f.write(json.dumps(p) + "\n")
            print(f"alpha {a:g} {name} acc {m.get('acc', float('nan')):.4f}", flush=True)
        (out / "metrics.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
