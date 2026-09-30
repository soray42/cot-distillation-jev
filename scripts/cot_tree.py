"""Split a teacher CoT into step nodes and show the key tokens of each step from its reasoning logprobs.

A key token is one the teacher was unsure about (chosen p < --max-p) where a top-5 alternative says
something different (T vs F, knight vs knave, another number or word), not just other punctuation or
formatting. Steps opening with a backtracking cue (Wait, Actually, ...) are marked; extracted
sub-questions are attached to the step that holds their quote.

  python3 scripts/cot_tree.py teacher_cache/label_kk_v1/kk-104-00002.json [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cotdistill.teacher import _expand  # noqa: E402

BACKTRACK = re.compile(r"^\s*(wait|actually|hmm|but wait|however|hold on|let me re|let's re|double[- ]check|"
                       r"check|recheck|verify|oops|no,|contradiction)", re.I)
CONTRA = re.compile(r"contradict|impossible|inconsistent|not possible|fails|violat", re.I)


def _word(tok: str) -> str:
    return re.sub(r"[^0-9a-z¬∧∨↔→=≠<>]+", "", tok.lower())


def key_tokens(toks: list[dict], max_p: float) -> list[dict]:
    out = []
    for i, t in enumerate(toks):
        p = math.exp(t["logprob"])
        w = _word(t["token"])
        if p >= max_p or not w:
            continue
        alts = []
        for a in t["top_logprobs"]:
            aw = _word(a["token"])
            if aw and aw != w and not (aw.startswith(w) or w.startswith(aw)):
                alts.append((a["token"], math.exp(a["logprob"])))
        if alts:
            out.append({"i": i, "token": t["token"], "p": p, "alts": alts[:3], "alt_mass": sum(q for _, q in alts)})
    return out


def steps_of(toks: list[dict]) -> list[dict]:
    """Token ranges of steps: a newline ends a step; long lines also split at sentence ends."""
    steps, start, text = [], 0, ""
    for i, t in enumerate(toks):
        text += t["token"]
        end_line = "\n" in t["token"]
        end_sent = len(text) > 160 and re.search(r"[.!?]\s*$", t["token"]) is not None
        if end_line or end_sent or i == len(toks) - 1:
            if text.strip():
                steps.append({"start": start, "end": i + 1, "text": text.strip()})
            start, text = i + 1, ""
    return steps


def tree(res: dict, max_p: float = 0.9) -> dict:
    tr = res["traces"][0]
    toks = _expand(tr["reasoning_lp"])
    steps = steps_of(toks)
    keys = key_tokens(toks, max_p)
    for s in steps:
        s["keys"] = [k for k in keys if s["start"] <= k["i"] < s["end"]]
        s["min_p"] = min((math.exp(t["logprob"]) for t in toks[s["start"]:s["end"]]), default=1.0)
        s["backtrack"] = bool(BACKTRACK.match(s["text"]))
        s["contradiction"] = bool(CONTRA.search(s["text"]))
        s["subqs"] = []
    for sq in res.get("subquestions", []):
        span = sq.get("span")
        if not span:
            continue
        st = next((s for s in steps if s["start"] <= span[0] < s["end"]), None)
        if st is not None:
            p = [a["p_yes"] for a in sq.get("answers", []) if a.get("p_yes") is not None]
            st["subqs"].append({"q": sq["question"], "p_cot": p[0] if p else None, "truth": sq.get("truth"),
                                "status": sq.get("status"), "span_minp": (sq.get("span_stats") or {}).get("min_p")})
    d = tr.get("dist_outcome") or {}
    it = res["item"]
    return {"item_id": it["item_id"], "depth": it.get("depth"), "gold": it["gold"],
            "teacher": max(d, key=d.get) if d else None, "teacher_p": max(d.values()) if d else None,
            "n_tokens": len(toks), "steps": steps, "n_keys": len(keys)}


def show(t: dict, width: int = 110, only_interesting: bool = True) -> None:
    ok = "correct" if t["teacher"] == t["gold"] else "WRONG"
    print(f"=== {t['item_id']}  depth {t['depth']}  {t['n_tokens']} tokens  {len(t['steps'])} steps  "
          f"{t['n_keys']} key tokens  teacher {ok} (p={t['teacher_p']:.3f})")
    for k, s in enumerate(t["steps"]):
        if only_interesting and not (s["keys"] or s["backtrack"] or s["subqs"] or s["contradiction"]):
            continue
        flag = ("↩ " if s["backtrack"] else "") + ("✗ " if s["contradiction"] else "")
        print(f"[{k:3d}] {flag}min_p={s['min_p']:.2f}  {s['text'][:width]!r}")
        for kt in s["keys"][:3]:
            alts = ", ".join(f"{a!r} {q:.2f}" for a, q in kt["alts"])
            print(f"        key {kt['token']!r} p={kt['p']:.2f}  vs {alts}")
        for q in s["subqs"]:
            pc = f"{q['p_cot']:.2f}" if q["p_cot"] is not None else "-"
            print(f"        ▶ sub-q ({q['status']}) P(yes|CoT)={pc} truth={q['truth']} "
                  f"span_min_p={q['span_minp'] if q['span_minp'] is None else round(q['span_minp'], 2)}: {q['q'][:90]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--max-p", type=float, default=0.9)
    ap.add_argument("--all-steps", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    trees = [tree(json.loads(Path(f).read_text()), args.max_p) for f in args.files]
    for t in trees:
        show(t, only_interesting=not args.all_steps)
    if args.json:
        Path(args.json).write_text(json.dumps(trees))


if __name__ == "__main__":
    main()
