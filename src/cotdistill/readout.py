"""Read label distributions and confidence signals out of token logprobs.

All functions take the `logprobs.content` / `logprobs.reasoning_content` lists returned by
the API: [{"token", "logprob", "top_logprobs": [{"token", "logprob"}, ...]}, ...].
"""
from __future__ import annotations

import difflib
import math
import re

CENSORED = -9000.0  # anything at or below this is the API's -9999 sentinel


def _norm_label(tok: str) -> str:
    return tok.strip().strip("()[].:*").strip()


def label_position(tokens: list[dict], marker: str | None = "ANSWER:") -> int | None:
    """Index of the first non-space token after `marker` (or the first non-space token if None)."""
    text = ""
    for i, t in enumerate(tokens):
        tok = t["token"]
        if marker is None or marker in text:
            if _norm_label(tok):          # skip pure punctuation such as "(" or "**" before the letter
                return i
        elif marker in text + tok:
            rest = (text + tok).split(marker, 1)[1]
            if _norm_label(rest):     # marker and label fused into one token
                return i
        text += tok
    return None


def label_distribution(tokens: list[dict], labels: list[str], marker: str | None = "ANSWER:") -> dict:
    """Distribution over `labels` at the answer position.

    Returns {"probs": {label: p renormalized over found labels}, "raw": {label: p},
             "missing": [labels not in top-k], "bound": upper bound for each missing label,
             "mass": sum of raw label probs, "sampled": sampled label or None, "pos": index}.
    Missing labels are censored (0 <= p <= bound), never treated as zero by callers.
    """
    pos = label_position(tokens, marker)
    if pos is None:
        return {"probs": {}, "raw": {}, "missing": list(labels), "bound": None, "mass": 0.0, "sampled": None, "pos": None}
    top = tokens[pos].get("top_logprobs") or [{"token": tokens[pos]["token"], "logprob": tokens[pos]["logprob"]}]
    raw: dict[str, float] = {}
    for alt in top:
        t = alt["token"].split(marker)[-1] if marker and marker in alt["token"] else alt["token"]
        lab = _norm_label(t)
        if lab in labels and alt["logprob"] > CENSORED:
            raw[lab] = raw.get(lab, 0.0) + math.exp(alt["logprob"])
    valid = [a["logprob"] for a in top if a["logprob"] > CENSORED]
    bound = math.exp(min(valid)) if valid else None
    mass = sum(raw.values())
    probs = {k: v / mass for k, v in raw.items()} if mass > 0 else {}
    st = tokens[pos]["token"]
    sampled = _norm_label(st.split(marker)[-1] if marker and marker in st else st)
    return {"probs": probs, "raw": raw, "missing": [l for l in labels if l not in raw], "bound": bound,
            "mass": mass, "sampled": sampled if sampled in labels else None, "pos": pos}


def token_confidence(tok: dict) -> float:
    """DeepConf token confidence: -(1/k) * sum of top-k logprobs (higher = more peaked)."""
    top = [a["logprob"] for a in (tok.get("top_logprobs") or []) if a["logprob"] > CENSORED]
    if not top:
        return 0.0
    return -sum(top) / len(top)


def token_entropy(tok: dict) -> float:
    """Entropy of the renormalized top-k distribution (a lower bound on the true entropy)."""
    ps = [math.exp(a["logprob"]) for a in (tok.get("top_logprobs") or []) if a["logprob"] > CENSORED]
    s = sum(ps)
    if s <= 0:
        return 0.0
    return -sum(p / s * math.log(p / s) for p in ps if p > 0)


def trace_confidence(tokens: list[dict], window: int = 32, tail: int = 64) -> dict:
    """DeepConf-style trace scores: mean, lowest window, bottom-10% windows, tail window."""
    c = [token_confidence(t) for t in tokens]
    if not c:
        return {"mean": 0.0, "lowest": 0.0, "bottom10": 0.0, "tail": 0.0, "n": 0}
    w = max(1, min(window, len(c)))
    groups = [sum(c[i:i + w]) / w for i in range(0, len(c) - w + 1)]
    groups.sort()
    k = max(1, len(groups) // 10)
    t = c[-tail:]
    return {"mean": sum(c) / len(c), "lowest": groups[0], "bottom10": sum(groups[:k]) / k,
            "tail": sum(t) / len(t), "n": len(c)}


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def locate_span(tokens: list[dict], quote: str, min_ratio: float = 0.8) -> tuple[int, int] | None:
    """Token index range [start, end) whose text best matches `quote` (fuzzy)."""
    if not quote or not tokens:
        return None
    pieces = [t["token"] for t in tokens]
    text = "".join(pieces)
    starts, pos = [], 0
    for p in pieces:
        starts.append(pos); pos += len(p)
    q = _squash(quote)
    low = text.lower()
    idx = low.find(quote.strip().lower())
    if idx < 0:
        sm = difflib.SequenceMatcher(None, low, q, autojunk=False)
        m = sm.find_longest_match(0, len(low), 0, len(q))
        if m.size < min_ratio * len(q):
            return None
        idx = m.a - m.b
        length = len(q)
    else:
        length = len(quote.strip())
    a = max(0, idx); b = min(len(text), idx + length)
    s = max(i for i, st in enumerate(starts) if st <= a)
    e = max(i for i, st in enumerate(starts) if st < b) + 1
    return s, e


def span_stats(tokens: list[dict], span: tuple[int, int] | None) -> dict | None:
    """Confidence of the tokens inside a CoT span (a proxy for the teacher's node certainty)."""
    if span is None:
        return None
    seg = tokens[span[0]:span[1]]
    if not seg:
        return None
    ps = [math.exp(t["logprob"]) for t in seg]
    ents = [token_entropy(t) for t in seg]
    return {"min_p": min(ps), "mean_p": sum(ps) / len(ps), "max_entropy": max(ents),
            "mean_entropy": sum(ents) / len(ents), "n_tokens": len(seg)}


VALUE_POS = {"true", "t", "knight", "knights", "yes", "holds", "valid", "consistent", "satisfied", "proved",
             "entailed", "follows", "possible"}
VALUE_NEG = {"false", "f", "knave", "knaves", "no", "untrue", "invalid", "inconsistent",
             "contradiction", "impossible", "disproved", "uncertain", "unknown"}
# "not" / "¬" are left out: their meaning depends on the next token ("Bob knight" vs "Bob not knave" say
# the same thing), so they create false forks.


def value_class(tok: str) -> str | None:
    """"P" / "N" for tokens that state a truth value or role (true, knave, not, ...), else None."""
    w = re.sub(r"[^0-9a-z¬]+", "", tok.lower())
    return "P" if w in VALUE_POS else "N" if w in VALUE_NEG else None


def value_commitment(tokens: list[dict], span: tuple[int, int] | None) -> dict | None:
    """How firmly the teacher settled a value inside a CoT span.

    For each value token in the span (true/false, knight/knave, not, ...), the top-5 probability on
    its own class vs the opposite class; returns the least confident one: {"i", "token", "conf"} with
    conf = own / (own + opposite) in [0.5, 1]. Wording alternatives ("So" vs "Then") are ignored, which is
    what separates a judgement fork from phrasing noise. Mass outside the stored top-5 is unknown and
    counted as zero."""
    if span is None:
        return None
    best = None
    for i in range(span[0], min(span[1], len(tokens))):
        c = value_class(tokens[i]["token"])
        if c is None:
            continue
        own = opp = 0.0
        for a in tokens[i].get("top_logprobs") or [{"token": tokens[i]["token"], "logprob": tokens[i]["logprob"]}]:
            ac = value_class(a["token"])
            if a["logprob"] <= CENSORED or ac is None:
                continue
            if ac == c:
                own += math.exp(a["logprob"])
            else:
                opp += math.exp(a["logprob"])
        conf = own / (own + opp) if own + opp > 0 else 1.0
        if best is None or conf < best["conf"]:
            best = {"i": i, "token": tokens[i]["token"], "conf": conf}
    return best
