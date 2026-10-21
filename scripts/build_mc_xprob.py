"""Other-problem distractors for the GSM8K and GSM1K multiple-choice test sets (verification-tree pilot): the same
problems, question texts and golds as data/eval/gsm8k_mc_v2.jsonl and data/eval/gsm1k_mc.jsonl, but the 4 wrong
options are final answers of OTHER problems of the same file, so no option is built from the problem (QO-v2 of
scripts/build_math_evals.py builds one-step combinations of the stated numbers, partial_*, and perturbations,
perturb_*, which a model can learn to spot; StartLux 2610.17309, App. D.2, reports such cues). A distractor equals a
number of the problem only by chance: on GSM8K test 4.3% of them are values of the solution's atomic steps
(data/eval/gsm8k_mc_v2_steps.jsonl) and 4.8% numbers written in the problem, against 7.4% and 5.9% of the QO-v2
distractors.

  data/eval/gsm8k_mc_v2_xprob.jsonl   from data/eval/gsm8k_mc_v2.jsonl (1,317 items)
  data/eval/gsm1k_mc_xprob.jsonl      from data/eval/gsm1k_mc.jsonl (1,205 items)
  data/eval/mc_xprob.summary.json     md5s, construction counts and the options-only adversaries

Every field of an item is the original's (item_id, source, group, depth, level, mag, nq, orig_id, variant, labels,
label_order, gold_probs, type) except prompt (same "Question: <text>" line, new "Options:" block), gold_label,
label_names (o0 = the gold, o1..o4 the distractors in draw order, as in the originals), option_kinds ("gold" /
"other_problem"), recipe ("xprob_v1:<md5 of this construction>") and one new field xprob {src: letter -> item_id of
the problem whose answer the option is, window_ratio, gold_rank}.

Construction (every draw from random.Random(f"xprob/<seed>/<item_id>/<try>"); the letter and rank quotas from
random.Random(f"xprob/<seed>/<file>/...")):
- candidates: the other items of the file whose gold differs from this gold (as printed); drawn per ITEM (not per
  distinct value), so a value is drawn as often as it is an answer and the distractors have the golds' own value
  distribution (roundness, size, frequency); a gold that is an integer only gets integer distractors (all golds of
  both files are integers);
- an exchangeable window on the scale u = ln(x + SHIFT), SHIFT = 5: the window [u(g) - t w, u(g) + (1 - t) w] holds
  the gold at a uniform position t, width w = ln 2; the 4 distractors are distinct values drawn from the candidates
  inside it. For large values this is a ratio-2 window (every distractor within 0.5x-2x of the gold); for small
  values it is nearly linear (g = 3: a window such as 1..6 or 3..11), where a ratio-2 window holds too few answers. With fewer than 4
  distinct values inside, w grows by ln 2 and t is redrawn, up to the whole range ("else any"). Because the window's
  position is uniform and not centred on the gold, the 5 values are (up to the discreteness of the answers and the
  lower end of the range) an exchangeable draw: given the five numbers, each is about equally likely to be the gold.
  A gold-centred window would not be: the gold would be the one option whose [x/2, 2x] holds all others (the
  "window rule" in the audit). On the plain log scale (SHIFT 0) small golds stay identifiable (the windows must
  widen around them): the construction-aware adversary below reads .25 there (.33 on golds < 10) against .21 with
  SHIFT 5 (3 seeds each; SHIFT 10 reads .20 but puts only 73% of the items within 0.5x-2x, SHIFT 5 83%);
- rank quotas: the gold's rank among the 5 values (0 = smallest) has n/5 slots per rank; an item whose draw lands on
  a full rank redraws (up to MAX_TRIES; then the last draw is kept and counted as quota_overflow);
- letters: the gold letters are a seeded shuffle of A..E repeated (exactly balanced, +-1); the 4 distractors take the
  remaining letters in a seeded order.
Checks (audit_item): the question text and gold unchanged, 5 distinct values, no distractor equal to the gold, every
distractor the gold of its source item (another item), option_kinds / label_names consistent.

Options-only adversaries (no question text; summary "adversary"; chance .200):
- optonly: the features of the scratch adversary optonly.py (log size vs the option mean, nearest log distances,
  integrality, roundness, digit count, rank and position one-hots, near-cluster and multiple counts), conditional
  logit (build_math_evals.fit_logit) fitted on the GSM8K training finals (data/student_gsm8k_tf/train.jsonl: QO-v2,
  inter and slip options); scored on the original and the new files;
- optonly_xprob_train: the same model fitted on the training finals rebuilt with this construction (training golds,
  other training problems' answers; not written to disk), scored on the new files: an adversary that knows the
  construction;
- crossfit: the same features and build_math_evals.features(rich) fitted 5-fold within each new file;
- rules: largest / smallest / middle / hub / 1.5x magnitude oracle (build_math_evals.rule_accs) and the window rule.

  python3 scripts/build_mc_xprob.py          # numpy (system python3); a few seconds
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import inspect
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

LETTERS = ["A", "B", "C", "D", "E"]
SETS = (("gsm8k_mc_v2", "gsm8k_mc_v2_xprob"), ("gsm1k_mc", "gsm1k_mc_xprob"))
EVAL = ROOT / "data/eval"
W0 = math.log(2.0)                 # window width on the u scale: ratio 2 for large values
SHIFT = 5.0                        # u = ln(x + SHIFT)
MAX_TRIES = 60                      # redraws per item while its rank's quota is full
OPT = re.compile(r"^([A-E])\) (.+)$", re.M)


# ------------------------------------------------------------------------------------------------------ numbers
def num(s: str) -> float:
    return float(str(s).replace(",", "").replace("$", ""))


def fmt(v: float) -> str:
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.4f}".rstrip("0").rstrip(".")


def is_int(v: float) -> bool:
    return abs(v - round(v)) < 1e-9


def scale(v: float) -> float:
    """The window scale u = ln(v + SHIFT)."""
    return math.log(v + SHIFT)


def split_prompt(prompt: str) -> tuple[str, dict]:
    """(question text, {letter: option text}) of a "Question: <q>\\nOptions:\\nA) ..." prompt."""
    head, sep, opts = prompt.partition("\nOptions:\n")
    if not sep or not head.startswith("Question: "):
        raise ValueError("prompt is not 'Question: ...' + options")
    return head[len("Question: "):], dict(OPT.findall(opts))


def gold_of(item: dict) -> str:
    return split_prompt(item["prompt"])[1][item["gold_label"]]


# ------------------------------------------------------------------------------------------------------ construction
class Pool:
    """The golds of one file (per item) for window lookups."""

    def __init__(self, ids: list[str], values: list[float]):
        order = sorted(range(len(values)), key=lambda i: values[i])
        self.ids = [ids[i] for i in order]
        self.vals = [values[i] for i in order]
        self.logs = [scale(v) for v in self.vals]

    def inside(self, lo: float, hi: float) -> range:
        import bisect
        return range(bisect.bisect_left(self.logs, lo - 1e-12), bisect.bisect_right(self.logs, hi + 1e-12))


def draw(g: float, own_id: str, pool: Pool, rng: random.Random) -> tuple[list[tuple[str, float]], float] | None:
    """4 distractors (source item id, value) for gold g from the exchangeable window, and its width ratio on the u
    scale (2 unless it had to grow); None if the file has fewer than 4 other distinct values of g's integer-ness."""
    lg, gt = scale(g), fmt(g)
    w = W0
    span = pool.logs[-1] - pool.logs[0] + W0
    while True:
        t = rng.random()
        lo = lg - t * w
        r = pool.inside(lo, lo + w)
        cand = [k for k in r if pool.ids[k] != own_id and fmt(pool.vals[k]) != gt
                and (is_int(pool.vals[k]) or not is_int(g))]
        if len({fmt(pool.vals[k]) for k in cand}) >= 4:
            break
        if w > 2 * span + 2 * W0:
            return None
        w += W0
    out, seen = [], set()
    while len(out) < 4:
        k = rng.choice(cand)
        v = fmt(pool.vals[k])
        if v not in seen:
            seen.add(v)
            out.append((pool.ids[k], pool.vals[k]))
    return out, math.exp(w)


def quotas(n: int, rng: random.Random) -> list[int]:
    q = [n // 5] * 5
    extra = list(range(5))
    rng.shuffle(extra)
    for r in extra[:n % 5]:
        q[r] += 1
    return q


def build_options(problems: list[tuple[str, float]], seed: int, tag: str) -> dict[str, dict]:
    """item id -> {distractors [(src id, value)], rank, window_ratio, gold_letter, tries, overflow} for problems
    [(item id, gold value)] of one file."""
    ids, vals = [p[0] for p in problems], [p[1] for p in problems]
    pool = Pool(ids, vals)
    n = len(problems)
    q = quotas(n, random.Random(f"xprob/{seed}/{tag}/rank"))
    letters = (LETTERS * (n // 5 + 1))[:n]
    random.Random(f"xprob/{seed}/{tag}/letters").shuffle(letters)
    order = list(range(n))
    random.Random(f"xprob/{seed}/{tag}/order").shuffle(order)
    out = {}
    for i in order:
        iid, g = problems[i]
        got = None
        for k in range(MAX_TRIES):
            d = draw(g, iid, pool, random.Random(f"xprob/{seed}/{iid}/{k}"))
            if d is None:
                raise ValueError(f"{iid}: fewer than 4 other answers of its kind in the file")
            rank = sum(v < g for _, v in d[0])
            got = (d, rank, k)
            if q[rank] > 0:
                break
        (dis, ratio), rank, k = got
        overflow = q[rank] <= 0
        q[rank] -= 1
        out[iid] = {"distractors": dis, "rank": rank, "window_ratio": ratio, "gold_letter": letters[i],
                    "tries": k + 1, "overflow": overflow}
    return out


def recipe_md5() -> str:
    src = json.dumps({"W0": W0, "SHIFT": SHIFT, "MAX_TRIES": MAX_TRIES}) + "".join(
        inspect.getsource(f) for f in (fmt, is_int, scale, Pool, draw, quotas, build_options, convert))
    return hashlib.md5(src.encode()).hexdigest()


def convert(item: dict, plan: dict, md5: str, seed: int) -> dict:
    q, opts = split_prompt(item["prompt"])
    gold = opts[item["gold_label"]]
    gl = plan["gold_letter"]
    rest = [L for L in LETTERS if L != gl]
    random.Random(f"xprob/{seed}/{item['item_id']}/letters").shuffle(rest)
    by_letter = {gl: (gold, "o0", "gold", None)}
    for j, ((src, v), L) in enumerate(zip(plan["distractors"], rest)):
        by_letter[L] = (fmt(v), f"o{j + 1}", "other_problem", src)
    out = dict(item)
    out["prompt"] = f"Question: {q}\nOptions:\n" + "\n".join(f"{L}) {by_letter[L][0]}" for L in LETTERS)
    out["gold_label"] = gl
    out["label_names"] = {L: by_letter[L][1] for L in LETTERS}
    out["option_kinds"] = {L: by_letter[L][2] for L in LETTERS}
    out["recipe"] = f"xprob_v1:{md5}"
    out["xprob"] = {"src": {L: by_letter[L][3] for L in LETTERS if by_letter[L][3]},
                    "window_ratio": round(plan["window_ratio"], 6), "gold_rank": plan["rank"]}
    return out


def audit_item(new: dict, orig: dict, golds: dict) -> list[str]:
    """Problems of one built item (empty: clean). golds: item id -> gold text of the source file."""
    bad = []
    iid = new["item_id"]
    q0, o0 = split_prompt(orig["prompt"])
    q1, o1 = split_prompt(new["prompt"])
    if q1 != q0:
        bad.append("question text changed")
    if sorted(o1) != LETTERS or new["labels"] != LETTERS:
        bad.append("letters")
    g = o0[orig["gold_label"]]
    if o1.get(new["gold_label"]) != g:
        bad.append("gold changed")
    vals = [num(o1[L]) for L in LETTERS]
    if len({fmt(v) for v in vals}) != 5:
        bad.append("duplicate option values")
    for L in LETTERS:
        if L == new["gold_label"]:
            if new["option_kinds"][L] != "gold" or new["label_names"][L] != "o0":
                bad.append("gold kind / name")
            continue
        src = new["xprob"]["src"].get(L)
        if new["option_kinds"][L] != "other_problem" or src is None or src == iid or golds.get(src) != o1[L]:
            bad.append(f"option {L} is not the answer of another problem")
        if fmt(num(o1[L])) == fmt(num(g)):
            bad.append(f"option {L} equals the gold")
    keep = {k for k in orig if k not in ("prompt", "gold_label", "label_names", "option_kinds", "recipe")}
    if any(new.get(k) != orig.get(k) for k in keep) or set(new) != set(orig) | {"xprob"}:
        bad.append("other fields changed")
    return [f"{iid}: {b}" for b in bad]


# ------------------------------------------------------------------------------------------------------ adversaries
def optonly_features(v: list[float]) -> list[list[float]]:
    """Per-option features of the scratch adversary optonly.py (same definitions)."""
    lv = [math.log10(abs(x) + 1) for x in v]
    srt = sorted(range(5), key=lambda i: v[i])
    rank = [srt.index(i) for i in range(5)]
    mu = sum(lv) / 5
    F = []
    for i in range(5):
        d = sorted(abs(lv[i] - lv[j]) for j in range(5) if j != i)
        f = [lv[i] - mu, (lv[i] - mu) ** 2, d[0], d[1], sum(d) / 4, float(v[i] == int(v[i])), float(v[i] % 10 == 0),
             float(v[i] % 5 == 0), len(str(int(abs(v[i])))), float(v[i] < 0)] + [float(rank[i] == r) for r in range(5)] \
            + [float(i == p) for p in range(5)]
        f += [sum(1 for j in range(5) if j != i and abs(v[j] - v[i]) <= max(1, 0.1 * abs(v[i]))),
              sum(1 for j in range(5) if j != i and v[j] != 0 and abs(v[i] / v[j] - round(v[i] / v[j])) < 1e-9
                  and v[i] != v[j])]
        F.append(f)
    return F


def mag(v: float) -> str:
    """Magnitude stratum of a gold (as the eval files' mag field): m1 < 10 <= m2 < 100 <= m3 < 1000 <= m4."""
    return "m1" if v < 10 else "m2" if v < 100 else "m3" if v < 1000 else "m4"


def window_rule(sets: list[list[float]], y: list[int]) -> float:
    """Pick the option x whose [x/2, 2x] holds every other option (expected credit on ties; none: chance)."""
    tot = 0.0
    for vs, t in zip(sets, y):
        ok = [i for i, x in enumerate(vs) if all(x / 2 - 1e-9 <= w <= 2 * x + 1e-9 for w in vs)]
        tot += (t in ok) / len(ok) if ok else 0.2
    return tot / max(1, len(y))


def option_sets(items: list[dict]) -> tuple[list[list[float]], list[int]]:
    sets, y = [], []
    for it in items:
        o = split_prompt(it["prompt"])[1]
        if sorted(o) != LETTERS:
            continue
        try:
            sets.append([num(o[L]) for L in LETTERS])
        except ValueError:
            continue
        y.append(LETTERS.index(it["gold_label"]))
    return sets, y


def train_sets() -> tuple[list[list[float]], list[int], list[tuple[str, float]]]:
    """The GSM8K training finals' option sets (data/student_gsm8k_tf/train.jsonl) and their (id, gold) list."""
    sets, y, probs = [], [], []
    with open(ROOT / "data/student_gsm8k_tf/train.jsonl") as f:
        for line in f:
            r = json.loads(line)
            vs = [num(x) for x in r["options"]]
            sets.append(vs)
            gi = r["labels"].index(r["gold_label"])
            y.append(gi)
            probs.append((r["item_id"], vs[gi]))
    return sets, y, probs


def adversaries(new: dict[str, list[dict]], orig: dict[str, list[dict]], seed: int) -> dict:
    import numpy as np
    import build_math_evals as M
    res = {"chance": 0.2}
    trs, try_, tprobs = train_sets()
    feats = lambda S: [np.array(optonly_features(v)) for v in S]  # noqa: E731
    m_qo = M.fit_logit(feats(trs), try_)
    res["optonly_fit_train_acc"] = round(M.logit_acc(m_qo, feats(trs), try_), 4)
    plan = build_options(tprobs, seed, "train_audit")
    xs, xy = [], []
    for iid, g in tprobs:
        p = plan[iid]
        vals = [g] + [v for _, v in p["distractors"]]
        letters = [p["gold_letter"]] + [L for L in LETTERS if L != p["gold_letter"]]
        by = dict(zip(letters, vals))
        xs.append([by[L] for L in LETTERS])
        xy.append(LETTERS.index(p["gold_letter"]))
    m_x = M.fit_logit(feats(xs), xy)
    res["optonly_xprob_train_fit_acc"] = round(M.logit_acc(m_x, feats(xs), xy), 4)
    for name, items in list(orig.items()) + list(new.items()):
        S, Y = option_sets(items)
        r = {"n": len(Y), "optonly": round(M.logit_acc(m_qo, feats(S), Y), 4)}
        if name in new:
            r["optonly_xprob_train"] = round(M.logit_acc(m_x, feats(S), Y), 4)
            by = collections.defaultdict(list)
            for f, v, t in zip(feats(S), S, Y):
                by[mag(v[t])].append(M.logit_acc(m_x, [f], [t]))
            r["optonly_xprob_train_by_gold_size"] = {k: [len(a), round(sum(a) / len(a), 4)] for k, a in sorted(by.items())}
            r["crossfit_optonly"] = round(M.crossfit_acc(feats(S), Y), 4)
            r["crossfit_rich"] = round(M.crossfit_acc([M.features(v, True) for v in S], Y), 4)
            r.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in M.rule_accs(S, Y).items()})
            r["window_rule"] = round(window_rule(S, Y), 4)
            r["gold_rank"] = dict(sorted(collections.Counter(sorted(v).index(v[t]) for v, t in zip(S, Y)).items()))
            r["gold_letter"] = dict(sorted(collections.Counter(LETTERS[t] for t in Y).items()))
        res[name] = r
    return res


# ------------------------------------------------------------------------------------------------------ main
def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def within2(it: dict, plan: dict) -> bool:
    g = num(gold_of(it))
    return all(0.5 * g - 1e-9 <= v <= 2 * g + 1e-9 for _, v in plan[it["item_id"]]["distractors"])


def build_file(items: list[dict], tag: str, seed: int, md5_: str) -> tuple[list[dict], dict, list[str]]:
    probs = [(it["item_id"], num(gold_of(it))) for it in items]
    if len({p[0] for p in probs}) != len(probs):
        raise ValueError(f"{tag}: duplicate item ids")
    plan = build_options(probs, seed, tag)
    out = [convert(it, plan[it["item_id"]], md5_, seed) for it in items]
    golds = {it["item_id"]: gold_of(it) for it in items}
    fails = [b for a, o in zip(out, items) for b in audit_item(a, o, golds)]
    ratios = collections.Counter(round(p["window_ratio"]) for p in plan.values())
    st = {"items": len(out), "gold_rank": dict(sorted(collections.Counter(p["rank"] for p in plan.values()).items())),
          "gold_letter": dict(sorted(collections.Counter(p["gold_letter"] for p in plan.values()).items())),
          "window_ratio": {str(k): v for k, v in sorted(ratios.items())},
          "within_0.5x_2x": sum(within2(it, plan) for it in items),
          "within_0.5x_2x_by_gold_size": {k: [sum(1 for it in items if mag(num(gold_of(it))) == k),
                                              sum(within2(it, plan) for it in items if mag(num(gold_of(it))) == k)]
                                          for k in ("m1", "m2", "m3", "m4")},
          "quota_overflow": sum(p["overflow"] for p in plan.values()),
          "redraws": sum(p["tries"] - 1 for p in plan.values()),
          "distinct_distractor_values": len({fmt(v) for p in plan.values() for _, v in p["distractors"]}),
          "distinct_golds": len({fmt(v) for _, v in probs})}
    return out, st, fails


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-audit", action="store_true", help="skip the options-only adversaries")
    args = ap.parse_args()
    md5_ = recipe_md5()
    summary = {"script": "scripts/build_mc_xprob.py", "args": vars(args), "recipe_md5": md5_, "md5": {},
               "source_md5": {}, "files": {}}
    fails, new, orig = [], {}, {}
    for src, dst in SETS:
        items = read(EVAL / f"{src}.jsonl")
        out, st, f = build_file(items, src, args.seed, md5_)
        fails += f
        path = EVAL / f"{dst}.jsonl"
        with open(path, "w") as fh:
            for r in out:
                fh.write(json.dumps(r) + "\n")
        summary["md5"][f"data/eval/{dst}.jsonl"] = md5(path)
        summary["source_md5"][f"data/eval/{src}.jsonl"] = md5(EVAL / f"{src}.jsonl")
        summary["files"][f"data/eval/{dst}.jsonl"] = st
        new[dst], orig[src] = out, items
        print(f"data/eval/{dst}.jsonl: {len(out)} items, md5 {summary['md5'][f'data/eval/{dst}.jsonl']}", flush=True)
    summary["audit"] = {"failures": len(fails), "examples": fails[:20]}
    if not args.no_audit:
        summary["adversary"] = adversaries(new, orig, args.seed)
    (EVAL / "mc_xprob.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("files", "audit", "adversary") if k in summary}, indent=1))
    if fails:
        print("AUDIT FAILED", fails[:10])
        sys.exit(1)


if __name__ == "__main__":
    main()
