"""Data-efficiency curves: accuracy vs share of the training set, answer-only (A2) vs Tree Forcing (4B, LR 3e-6).

Reads the prediction dumps under results/hpc/l4b (argmax over the option letters, as every readout here):
  GSM8K  test + GSM1K (gsm8kv2, gsm1k); A2-c12lr3 vs GFPQ-c4lr3 on data/student_gsm8k_tf{_f10,_f25,_f50,}
  math   numgen depths 1-4 (in range); A2-c12lr3 vs GFPS-c4lr3 on data/student_numgen_tf{_f10,_f25,}
Lines are seed 12 (every fraction); the other seeds at 100% are drawn as small points; grey numbers = Tree Forcing
minus answer-only at each share; the dotted line = the share at which Tree Forcing reaches answer-only's 100% accuracy
(linear interpolation in log share).

  .venv-rg/bin/python scripts/plot_data_efficiency.py [--out figures/data_efficiency]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/hpc/l4b"
PALETTE = {"blue_main": "#0F4D92", "red_strong": "#B64342", "neutral": "#CFCECE"}
PANELS = [
    {"title": "GSM8K (test + GSM1K)", "data": "Qwen3.5-4B-student_gsm8k_tf", "a2": "A2-c12lr3", "tf": "GFPQ-c4lr3",
     "evals": ["gsm8kv2", "gsm1k"], "fracs": [10, 25, 50, 100], "note_x": 32, "gap_left": [25]},
    {"title": "Math chains (depth 1-4)", "data": "Qwen3.5-4B-student_numgen_tf", "a2": "A2-c12lr3", "tf": "GFPS-c4lr3",
     "evals": [f"ng_d{d}" for d in range(1, 5)], "fracs": [10, 25, 100]},
]


def acc(run: str, evals: list[str]) -> float | None:
    hits = []
    for k in evals:
        f = RUNS / run / f"preds_{k}.jsonl"
        if not f.exists():
            return None
        for p in map(json.loads, open(f)):
            hits.append(p["labels"][max(range(len(p["probs"])), key=p["probs"].__getitem__)] == p["gold_label"])
    return 100 * sum(hits) / len(hits) if hits else None


def run_name(arm: str, data: str, frac: int, seed: int) -> str:
    return f"{arm}-{data}{'' if frac == 100 else f'_f{frac}'}-s{seed}"


def share_needed(fracs: list[int], ys: list[float], target: float) -> float | None:
    """Smallest training share where the curve reaches target, interpolated linearly in log(share)."""
    for (f0, y0), (f1, y1) in zip(zip(fracs, ys), zip(fracs[1:], ys[1:])):
        if y0 >= target:
            return f0
        if y0 < target <= y1:
            t = (target - y0) / (y1 - y0)
            return math.exp(math.log(f0) + t * (math.log(f1) - math.log(f0)))
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="figures/data_efficiency")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 13, "axes.linewidth": 1.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    report = {}
    for ax, pn in zip(axes, PANELS):
        fr = pn["fracs"]
        a2 = [acc(run_name(pn["a2"], pn["data"], f, 12), pn["evals"]) for f in fr]
        tf = [acc(run_name(pn["tf"], pn["data"], f, 12), pn["evals"]) for f in fr]
        if None in a2 or None in tf:
            raise SystemExit(f"{pn['title']}: missing runs (A2 {a2}, TF {tf})")
        ax.plot(fr, a2, "-o", color=PALETTE["red_strong"], lw=2.4, ms=7, label="Answer only")
        ax.plot(fr, tf, "-o", color=PALETTE["blue_main"], lw=2.4, ms=7, label="Tree Forcing")
        for arm, col in ((pn["a2"], PALETTE["red_strong"]), (pn["tf"], PALETTE["blue_main"])):
            for s in (13, 14):
                y = acc(run_name(arm, pn["data"], 100, s), pn["evals"])
                if y is not None:
                    ax.plot([100], [y], "o", color=col, ms=4, alpha=0.45, zorder=1)
        ax.axhline(a2[-1], ls="--", lw=1.2, color=PALETTE["red_strong"], alpha=0.6)
        need = share_needed(fr, tf, a2[-1])
        if need is not None:
            ax.axvline(need, ymax=0.97, ls=":", lw=1.2, color=PALETTE["blue_main"], alpha=0.7)
            ax.annotate(f"Tree Forcing matches\nanswer-only @100%\nwith ~{need:.0f}% of the data",
                        xy=(need, a2[-1]),
                        xytext=(pn.get("note_x", need * 1.25), min(a2) + 0.12 * (a2[-1] - min(a2))),
                        fontsize=10.5, color=PALETTE["blue_main"],
                        arrowprops={"arrowstyle": "->", "color": PALETTE["blue_main"], "lw": 1.0})
        for x, ya, yt in zip(fr, a2, tf):            # the gap, between the two points, right of them
            ax.plot([x, x], [ya, yt], color=PALETTE["neutral"], lw=1.0, zorder=0)
            small = yt - ya < 1.5                       # a small gap: the number goes above the upper point
            left = x in pn.get("gap_left", [])          # keep the number clear of the note's arrow
            ax.annotate(f"{yt - ya:+.1f}", xy=(x, yt if small else (ya + yt) / 2),
                        xytext=(0, 9) if small else ((-7, 0) if left else (7, 0)), textcoords="offset points",
                        ha="center" if small else ("right" if left else "left"), va="bottom" if small else "center",
                        fontsize=10, color="#555555")
        ax.set_xscale("log")
        ax.set_xlim(8.3, 135)
        ax.set_xticks(fr)
        ax.set_xticklabels([f"{f}%" for f in fr])
        ax.minorticks_off()
        ax.set_xlabel("Share of the training set")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(pn["title"], fontsize=13)
        report[pn["title"]] = {"fracs": fr, "answer_only": [round(v, 1) for v in a2],
                               "tree_forcing": [round(v, 1) for v in tf],
                               "share_to_match_answer_only_full": round(need, 1) if need else None}
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    (out.with_suffix(".json")).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
