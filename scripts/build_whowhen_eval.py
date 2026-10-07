"""Who&When (Zhang et al. 2025, HF Kevin355/Who_and_When) as one-pass multiple-choice eval files for Sev-style models.

Each failed multi-agent trajectory becomes a prompt with the task, the numbered steps (agent + content) and a final
question; the options are the steps (step-level) or the agents (agent-level). Gold = the annotated decisive error
(mistake_step / mistake_agent: the earliest step whose correction would make the task succeed). Long step contents are
cut to their head and tail so a trajectory fits the eval context; the question sits at the end (scorers keep the tail).

  python scripts/build_whowhen_eval.py --src data/raw/who_and_when --out data/eval
  -> ww_alg_step.jsonl, ww_alg_step_gt.jsonl (task answer shown, the paper's "with ground truth"), ww_alg_agent.jsonl,
     ww_hc_agent.jsonl. Hand-crafted step-level (up to 130 steps) does not fit letter options and is not built.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import string

LETTERS = string.ascii_uppercase


def cut(text: str, cap: int) -> str:
    text = str(text).strip()
    if len(text) <= cap:
        return text
    head = cap * 2 // 3
    return text[:head] + " [...] " + text[-(cap - head):]


def agent_of(step: dict) -> str:
    """Hand-crafted roles look like 'Orchestrator (-> WebSurfer)'; algorithm-generated steps carry a name."""
    name = step.get("name") or step.get("role") or "unknown"
    return str(name).split(" (")[0].strip()


def norm(a: str) -> str:
    return a.lower().replace("_", "").replace(" ", "")


def render(x: dict, cap: int, with_gt: bool) -> str:
    q = str(x.get("question", "")).strip()
    lines = ["The log below comes from a multi-agent system that failed to solve its task.", "", f"Task: {cut(q, 1500)}"]
    gt = x.get("ground_truth", x.get("groundtruth"))
    if with_gt and gt is not None:
        lines.append(f"Correct final answer (for reference): {cut(gt, 300)}")
    lines.append("")
    for i, s in enumerate(x["history"]):
        lines.append(f"Step {i} [{agent_of(s)}]: {cut(s.get('content', ''), cap)}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/who_and_when")
    ap.add_argument("--out", default="data/eval")
    ap.add_argument("--cap-alg", type=int, default=3000, help="max characters per step, algorithm-generated")
    ap.add_argument("--cap-hc", type=int, default=800, help="max characters per step, hand-crafted")
    args = ap.parse_args()
    q_step = ("Question: Which step contains the decisive error, i.e. the earliest mistake whose correction would have "
              "let the system solve the task?")
    q_agent = "Question: Which agent made the decisive error that caused the failure?"
    stats = {}
    for sub, tag, cap in (("Algorithm-Generated", "alg", args.cap_alg), ("Hand-Crafted", "hc", args.cap_hc)):
        files = sorted(glob.glob(os.path.join(args.src, "Who&When", sub, "*.json")),
                       key=lambda f: int(os.path.basename(f)[:-5]))
        data = [(os.path.basename(f)[:-5], json.load(open(f))) for f in files]
        outs = {"agent": []} if tag == "hc" else {"step": [], "step_gt": [], "agent": []}
        skipped = 0
        for qid, x in data:
            body = render(x, cap, False)
            agents = []
            for s in x["history"]:
                a = agent_of(s)
                if a.lower() != "human" and a not in agents:
                    agents.append(a)
            gold_a = next((a for a in agents if norm(a) == norm(x["mistake_agent"])), None)
            if gold_a is None or len(agents) > len(LETTERS):
                skipped += 1
            else:
                labs = list(LETTERS[:len(agents)])
                outs["agent"].append({
                    "item_id": f"ww-{tag}-{qid}-agent", "source": "who_and_when", "group": f"ww/{tag}/agent",
                    "type": "choice", "prompt": body + "\n\n" + q_agent + "\nOptions:\n" +
                    "\n".join(f"{l}) {a}" for l, a in zip(labs, agents)),
                    "labels": labs, "label_order": labs, "gold_label": labs[agents.index(gold_a)],
                    "label_names": dict(zip(labs, agents)), "gold_probs": None})
            if tag == "alg":
                steps = x["history"]
                labs = list(LETTERS[:len(steps)])
                opts = "\n".join(f"{l}) Step {i} [{agent_of(s)}]" for i, (l, s) in enumerate(zip(labs, steps)))
                gold = labs[int(x["mistake_step"])]
                for key, gt in (("step", False), ("step_gt", True)):
                    outs[key].append({
                        "item_id": f"ww-{tag}-{qid}-{key}", "source": "who_and_when", "group": f"ww/{tag}/{key}",
                        "type": "choice", "prompt": render(x, cap, gt) + "\n\n" + q_step + "\nOptions:\n" + opts,
                        "labels": labs, "label_order": labs, "gold_label": gold, "gold_probs": None})
        for key, rows in outs.items():
            path = os.path.join(args.out, f"ww_{tag}_{key}.jsonl")
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_opt = [len(r["labels"]) for r in rows]
            stats[f"ww_{tag}_{key}"] = {"n": len(rows), "chance": round(sum(1 / k for k in n_opt) / len(rows), 4),
                                        "max_prompt_chars": max(len(r["prompt"]) for r in rows)}
        stats[f"{tag}_skipped_agent"] = skipped
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
