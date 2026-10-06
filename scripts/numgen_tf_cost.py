"""Per-epoch sequences / tokens / forward passes of the numgen Tree Forcing arms (data/student_numgen_tf) and an HPC
time estimate with the time model of notes/next_round/tree_forcing/impl.md §7 (s/epoch = 189 + 122 x M padded tokens +
0.154 x forward passes, x 0.946). Views are built by scripts/tree_forcing_cost.py's measure() (train_student.py's
builders, seed 0); this machine has neither torch nor tokenizers, so torch is stubbed (the view builders are pure
Python) and the Qwen3.5 tokenizer.json is applied with a small byte-level BPE (pre-tokenizer regex translated to `re`;
ASCII text, so \\p{L} / \\p{N} = letters / digits). CPU only, streams the data.

  python3 scripts/numgen_tf_cost.py --tokenizer <dir with tokenizer.json> --out data/student_numgen_tf/cost.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from unittest import mock

for m in ("torch", "torch.nn", "torch.nn.functional", "torch.utils", "torch.utils.checkpoint"):
    sys.modules.setdefault(m, mock.MagicMock())
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tree_forcing_cost as tfc  # noqa: E402

PRE = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|(?:[^\r\n\w]|_)?[^\W\d_]+|\d| ?(?:[^\s\w]|_)+[\r\n]*|\s*[\r\n]+"
                 r"|\s+(?!\S)|\s+")


def bytes_to_unicode() -> dict:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class BPE:
    def __init__(self, path: Path):
        t = json.loads(path.read_text())
        self.vocab = t["model"]["vocab"]
        merges = t["model"]["merges"]
        self.rank = {tuple(m.split(" ")) if isinstance(m, str) else tuple(m): i for i, m in enumerate(merges)}
        self.b2u = bytes_to_unicode()
        self.cache: dict = {}

    def word(self, w: str) -> int:
        if w in self.cache:
            return self.cache[w]
        parts = [self.b2u[b] for b in w.encode("utf-8")]
        while len(parts) > 1:
            pairs = [(self.rank.get((a, b), 1 << 30), i) for i, (a, b) in enumerate(zip(parts, parts[1:]))]
            r, i = min(pairs)
            if r == 1 << 30:
                break
            parts = parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2:]
        self.cache[w] = len(parts)
        return len(parts)

    def count(self, text: str) -> int:
        return sum(self.word(w) for w in PRE.findall(text))

    def encode_batch(self, texts, add_special_tokens=False):
        return [type("Enc", (), {"ids": range(self.count(t))})() for t in texts]


ARMS = {
    "A2": ("tree", dict(mode="plain", cap=0, view="problem", sources={"no-node-views"})),   # finals only
    "G3": ("group", {}),
    "TFL": ("tree", dict(mode="true", cap=0, view="local", sources={"numgen"})),
    "TFLM": ("tree", dict(mode="matched", cap=0, view="local", sources={"numgen"})),
    "TFA": ("tree", dict(mode="true", cap=0, view="ancestors", sources={"numgen"})),
    "GFP": ("tree", dict(mode="plain", cap=0, view="problem", sources={"numgen"})),
}


def hours(r: dict, epochs: float, fwd_key: str, pad_key: str) -> float:
    return round(epochs * 0.946 * (189 + 122 * r[pad_key] / 1e6 + 0.154 * r[fwd_key]) / 3600, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", default="data/student_numgen_tf")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--check-kk", action="store_true", help="also measure K&K TFL on v3sl (tokenizer check vs cost.json)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    tok = BPE(Path(args.tokenizer) / "tokenizer.json")
    res = {}
    todo = [(n, args.data, *ARMS[n]) for n in args.arms.split(",")]
    if args.check_kk:
        todo.append(("KK TFL (check)", "data/student_v3sl", "tree",
                     dict(mode="true", cap=0, view="local", sources={"knights_knaves"})))
    for name, data, kind, kw in todo:
        r = tfc.measure(tok, data, kind, kw, 1536, 2, [3072])
        r["est_h_mbs2"] = hours(r, args.epochs, "fwd_mbs", "pad_mbs")
        r["est_h_tb3072"] = hours(r, args.epochs, "fwd_tb3072", "pad_tb3072")
        res[name] = r
        print(name, json.dumps(r), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"epochs": args.epochs, "arms": res}, indent=1))


if __name__ == "__main__":
    main()
