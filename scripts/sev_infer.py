"""Minimal Sev-2b inference: one forward pass per decision, no generated tokens.

A decision is rendered as "<problem>\\n\\nAnswer:", where <problem> ends with its options block ("Options:\\nA) ...\\nB)
..."). The answer is read from the next-token logits at the last prompt token, restricted to the single tokens " A",
" B", ... of the problem's option letters, and normalised over those letters with a softmax. This is the readout used
in training and evaluation (src/cotdistill/student.py: FINAL_TEMPLATE, letter_token_ids, collate, label_logits,
evaluate):
- no special tokens are added and no chat template is used;
- a prompt longer than --max-len keeps its last tokens, so "Answer:" is never cut (evaluation used 6,144; training cut
  prompts to their last 1,536 tokens);
- the letter logits are the final hidden state at the last prompt token times the LM-head rows of the letter tokens;
- on a GPU the model is in bf16 and the forward pass and the letter matmul run under torch.autocast(bfloat16), as in
  evaluation; on the CPU everything is fp32, so probabilities differ slightly from the GPU ones.

One prompt per forward pass (batch size 1). To batch, right-pad and read each row at its last real token: the
linear-attention layers of Qwen3.5 ignore the attention mask, so left padding feeds the pad tokens into the state and
corrupts the readout.

Optional post-hoc calibration, as fitted by scripts/fit_calibration.py: the letter logits are divided by the temperature
of the question type; for a TD-style yes/no question, a bias is then added to the logit of the Yes letter.

  python scripts/sev_infer.py --model <repo-or-dir>                                  # built-in demo decision
  python scripts/sev_infer.py --model <repo-or-dir> --problem-file p.txt --temperature <T>
  python scripts/sev_infer.py --model <repo-or-dir> --problem-file p.txt --temperature <T> --yes-letter B --yes-bias <b>

From Python:

  from sev_infer import decide, load
  model, tok = load("<repo-or-dir>")
  probs = decide(model, tok, problem)            # {"A": p_A, "B": p_B, ...}
"""
from __future__ import annotations

import argparse
import json
import re

import torch
import transformers

FINAL_TEMPLATE = "{problem}\n\nAnswer:"
OPTION_LINE = re.compile(r"^\(?([A-Z])\) ", re.M)          # "A) ..." or "(A) ..."
DEMO = ("Policy: a company laptop may leave the office only with the manager's written approval and with full-disk "
        "encryption turned on. Dana's laptop is encrypted. Her manager approved the trip by phone.\n\n"
        "Question: Does the policy allow Dana to take the laptop out of the office?\n"
        "Options:\n"
        "A) No: The policy does not allow it.\n"
        "B) Yes: The policy allows it.")      # a TD-style yes/no decision, written for this example


def option_letters(problem: str) -> list[str]:
    """Letters of the option lines ("A) ..." or "(A) ...") that follow the last "Options:" line of the problem."""
    _, sep, block = problem.rpartition("Options:")
    letters = OPTION_LINE.findall(block) if sep else []
    if not letters:
        raise ValueError("no 'Options:' block with lines 'A) ...' found; pass the option letters explicitly")
    return letters


def letter_token_ids(tok, letters: list[str]) -> list[int]:
    """Token ids of ' A', ' B', ... (the token that follows 'Answer:'); each must be a single token."""
    ids = []
    for letter in letters:
        t = tok.encode(" " + letter, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"label ' {letter}' is not a single token: {t}")
        ids.append(t[0])
    return ids


def backbone_and_head(model):
    """(decoder that returns the final-normed last_hidden_state, LM head), for the causal and the multimodal classes."""
    head = model.get_output_embeddings()
    inner = getattr(model, "model", model)
    body = getattr(inner, "language_model", None) or getattr(model, "language_model", None)
    if body is None:
        body = model.get_decoder() if hasattr(model, "get_decoder") else inner
    return body, head


def load(path: str, device: str | None = None):
    """(model, tokenizer) from a Hugging Face repo id or a local directory; bf16 on GPU, fp32 on CPU, as in evaluation."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tok = transformers.AutoTokenizer.from_pretrained(path)
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    except ValueError:  # "Unrecognized configuration class": a checkpoint saved with the multimodal Qwen3.5 config
        model = transformers.AutoModelForImageTextToText.from_pretrained(path, dtype=dtype)
    return model.to(device).eval(), tok


@torch.no_grad()
def decide(model, tok, problem: str, *, letters: list[str] | None = None, temperature: float = 1.0,
           yes_letter: str | None = None, yes_bias: float = 0.0, max_len: int = 6144) -> dict[str, float]:
    """{letter: probability} for one decision. `letters` defaults to the problem's option letters; `temperature` and,
    for TD-style yes/no decisions only, `yes_letter` with `yes_bias` apply the post-hoc calibration of the model card."""
    letters = [L.strip() for L in letters] if letters else option_letters(problem)
    yes_letter = yes_letter.strip() if yes_letter else None
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if (yes_letter is not None or yes_bias) and yes_letter not in letters:
        raise ValueError("yes_letter must be one of the option letters (and is required with yes_bias)")
    body, head = backbone_and_head(model)
    dev = head.weight.device
    ids = tok.encode(FINAL_TEMPLATE.format(problem=problem), add_special_tokens=False)[-max_len:]
    x = torch.tensor([ids], device=dev)
    rows = torch.tensor(letter_token_ids(tok, letters), device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):    # as in evaluation
        h = body(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state[0, -1]
        w = head.weight[rows]
        z = h.to(w.dtype) @ w.T
        if getattr(head, "bias", None) is not None:
            z = z + head.bias[rows]
    z = z.float() / temperature
    if yes_letter is not None:
        z[letters.index(yes_letter)] += yes_bias
    return dict(zip(letters, torch.softmax(z, -1).tolist()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face repo id or local model directory")
    ap.add_argument("--problem-file", help="text file with one problem ending in its options block (default: a demo); "
                    "its trailing newlines are dropped, nothing else is changed")
    ap.add_argument("--letters", help="comma-separated option letters (default: read from the options block)")
    ap.add_argument("--temperature", type=float, default=1.0, help="per-type temperature from the model card")
    ap.add_argument("--yes-letter", help="letter of the Yes option of a TD-style yes/no question (for --yes-bias)")
    ap.add_argument("--yes-bias", type=float, default=0.0, help="yes/no bias from the model card, added after --temperature")
    ap.add_argument("--max-len", type=int, default=6144, help="longer prompts keep their last tokens (evaluation default)")
    ap.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    args = ap.parse_args()

    problem = open(args.problem_file).read().rstrip("\n") if args.problem_file else DEMO   # only the final newlines go
    model, tok = load(args.model, args.device)
    probs = decide(model, tok, problem, letters=args.letters.split(",") if args.letters else None,
                   temperature=args.temperature, yes_letter=args.yes_letter, yes_bias=args.yes_bias,
                   max_len=args.max_len)
    if not args.problem_file:
        print(problem + "\n")
    print(json.dumps({"answer": max(probs, key=probs.get), "probs": {k: round(v, 4) for k, v in probs.items()}}))


if __name__ == "__main__":
    main()
