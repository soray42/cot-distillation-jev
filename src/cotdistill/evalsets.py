"""Held-out evaluation sets converted to the student's record format (evaluation only, never training).

Record: item_id, source, group, type (choice/noul/score), prompt (state, question and lettered options,
the same layout as the training prompts), labels (letters), gold_label, label_names (letter -> native
label), and gold_probs (letter -> probability) where the set ships a soft gold.

- jevbench: JevBench public tiers (MIT; original / easy / hard jsonl from the repo).
- typed_decisions: Typed Decisions test split (Apache-2.0); one record per (case, question), soft gold.
- bbeh: BIG-Bench Extra Hard (Apache-2.0) tasks with a closed answer set, as lettered choices. Hard for
  reasoning models too (DeepSeek R1 averages 37% on BBEH mini), so the teacher is not an oracle here.
"""
from __future__ import annotations

import json
import random
import re

NOUL_NAMES = {"yes": "Yes", "no": "No", "true": "Yes", "false": "No"}


def _options(qtype: str, labels: list[str], criteria) -> list[tuple[str, str]]:
    """(native label, option text) in the item's label order."""
    out = []
    for i, lab in enumerate(labels):
        if qtype == "noul":
            key = "true" if lab in ("yes", "true") else "false"
            desc = criteria.get(key, "") if isinstance(criteria, dict) else ""
            text = NOUL_NAMES.get(lab, lab)
        elif qtype == "score":
            desc = criteria[i] if isinstance(criteria, list) and i < len(criteria) else \
                (criteria.get(lab, "") if isinstance(criteria, dict) else "")
            text = lab
        else:
            desc = criteria.get(lab, "") if isinstance(criteria, dict) else ""
            text = lab
        out.append((lab, f"{text}: {desc}" if desc else text))
    return out


def render(state: str, question: dict, labels: list[str]) -> tuple[str, list[str], dict]:
    qtype = question.get("type", "choice")
    opts = _options(qtype, labels, question.get("criteria") or {})
    letters = [chr(65 + i) for i in range(len(opts))]
    opt_txt = "\n".join(f"{L}) {text}" for L, (_, text) in zip(letters, opts))
    prompt = f"{state}\n\nQuestion: {question.get('instructions', '').strip()}\nOptions:\n{opt_txt}"
    return prompt, letters, {L: lab for L, (lab, _) in zip(letters, opts)}


def _gold_letter(names: dict, gold) -> str | None:
    g = str(gold).lower() if isinstance(gold, bool) else str(gold)
    g = {"true": "yes", "false": "no"}.get(g, g)
    for L, lab in names.items():
        if lab == g or {"true": "yes", "false": "no"}.get(lab, lab) == g:
            return L
    return None


def jevbench(paths: dict[str, str]) -> list[dict]:
    """paths: tier -> jsonl path (e.g. {"original": ..., "easy": ..., "hard": ...})."""
    out = []
    for tier, path in paths.items():
        for line in open(path):
            if not line.strip():
                continue
            it = json.loads(line)
            labels = [str(x) for x in it["labels"]]
            prompt, letters, names = render(it["state"], it["question"], labels)
            out.append({"item_id": it["id"], "source": "jevbench", "group": f"{tier}/{it['family']}",
                        "type": it["question"].get("type"), "prompt": prompt, "labels": letters,
                        "gold_label": _gold_letter(names, it.get("expected")), "label_names": names,
                        "gold_probs": None, "pair": it.get("group")})
    return out


def typed_decisions(parquet_path: str) -> list[dict]:
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    out = []
    for _, row in df.iterrows():
        state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"])
        try:
            state_txt = json.dumps(json.loads(state), indent=1)
        except (TypeError, json.JSONDecodeError):
            state_txt = state
        qs = json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
        gold = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]
        for qname, q in qs.items():
            g = gold[qname]
            if q["type"] == "noul":
                labels = ["false", "true"]
            elif q["type"] == "score":
                labels = sorted(g["probabilities"], key=lambda x: float(x))
            else:
                labels = list(q["criteria"])
            prompt, letters, names = render(f"State:\n{state_txt}", q, labels)
            probs = {L: float(g["probabilities"].get(names[L], 0.0)) for L in letters}
            gl = next((L for L in letters if names[L] == str(g["label"])), None)
            out.append({"item_id": f"{row['id']}/{qname}", "source": "typed_decisions",
                        "group": f"{row['workflow']}/{qname}", "type": q["type"], "prompt": prompt,
                        "labels": letters, "gold_label": gl, "label_names": names, "gold_probs": probs})
    return out


BBEH_LETTER_TASKS = ["boolean_expressions", "disambiguation_qa", "geometric_shapes", "hyperbaton",
                     "movie_recommendation", "nycc", "shuffled_objects"]
BBEH_SET_TASKS = {"boardgame_qa": ["proved", "disproved", "unknown"],
                  "causal_understanding": ["Yes", "No", "Ambiguous"]}


def bbeh(raw_dir: str, n_per_task: int | None = None, seed: int = 0,
         tasks: list[str] | None = None) -> list[dict]:
    """BBEH tasks whose answer is one option letter or one value from a small closed set."""
    rng = random.Random(seed)
    out = []
    for task in tasks or BBEH_LETTER_TASKS + list(BBEH_SET_TASKS) + ["zebra_puzzles"]:
        exs = json.load(open(f"{raw_dir}/{task}.json"))["examples"]
        idx = list(range(len(exs)))
        rng.shuffle(idx)
        for i in idx[:n_per_task] if n_per_task else idx:
            text, target = exs[i]["input"].strip(), str(exs[i]["target"]).strip()
            if task in BBEH_LETTER_TASKS:
                letters = sorted(set(re.findall(r"^\(([A-Z])\)", text, re.M)))
                gold = target.strip("()")
                prompt, names = text, {L: L for L in letters}
            else:
                values = BBEH_SET_TASKS.get(task)
                if values is None:                          # zebra: "What position is ... at?"
                    m = re.match(r"There are (\d+) (?:people|houses)", text)
                    if not m:
                        continue
                    values = [str(k) for k in range(1, int(m.group(1)) + 1)]
                letters = [chr(65 + k) for k in range(len(values))]
                names = dict(zip(letters, values))
                gold = next((L for L, v in names.items() if v.lower() == target.lower()), None)
                prompt = text + "\n\nOptions:\n" + "\n".join(f"{L}) {v}" for L, v in names.items())
            if gold not in letters:
                continue
            out.append({"item_id": f"bbeh-{task}-{i:03d}", "source": "bbeh", "group": f"bbeh/{task}",
                        "type": "choice", "prompt": prompt, "labels": letters, "label_order": letters,
                        "gold_label": gold, "label_names": names, "gold_probs": None})
    return out
