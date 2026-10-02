"""Upload the Sev-2b release to a Hugging Face model repo: the model directory, the model card, the data card, the
inference example and the base model's licence, all in one commit.

What goes up:
- from --model-dir, only the top-level weight, config and tokenizer files (MODEL_FILES patterns); training leftovers
  such as preds_*.jsonl, hidden_*.pt, metrics.json or optimizer states are never uploaded;
- --card as README.md, --data-card as data_card.csv and scripts/sev_infer.py as sev_infer.py (the card's usage snippet
  imports it); the card replaces any README.md of the model directory;
- the base model's Apache-2.0 LICENSE as LICENSE-Qwen3.5-2B-Base, and its NOTICE as NOTICE-Qwen3.5-2B-Base if it has
  one (Apache-2.0 section 4(a) and 4(d): the weights are a modified Qwen3.5-2B-Base).

Checks before anything is sent (the dry run makes them too):
- the card and the data card must hold no placeholder (PLACEHOLDERS) unless --allow-placeholders, which is refused
  together with --public;
- the model directory needs config.json, at least one .safetensors file and the tokenizer files (tokenizer_config.json
  and tokenizer.json, or vocab.json and merges.txt);
- the base licence file must exist and be the Apache License text;
- the repo is created private unless --public. An existing public repo is refused without --public; an existing
  private repo stays private even with --public (create_repo does not change visibility; change it on the Hub).

No token is taken on the command line or written anywhere: huggingface_hub uses HF_TOKEN or the credentials of an
earlier `hf auth login`. notes/ is not in git, so copy the two card files to the login node or pass their paths.

  python scripts/hf_upload.py --repo <account>/Sev-2b --model-dir runs/TF-v4t-.../model --dry-run
  python scripts/hf_upload.py --repo <account>/Sev-2b --model-dir runs/TF-v4t-.../model
"""
from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD_DIR = ROOT / "notes/release/sev-2b"
BASE_DIRS = (ROOT / "models/Qwen3.5-2B-Base", ROOT.parent / "models/Qwen3.5-2B-Base")   # local clone; ~/cotd on HPC
PLACEHOLDERS = ("RESULT", "TODO-CONFIRM", "[CLAIM SENTENCE")
MODEL_FILES = ("config.json", "generation_config.json", "*.safetensors", "model.safetensors.index.json",
               "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json",
               "merges.txt", "chat_template.jinja", "chat_template.json", "preprocessor_config.json",
               "video_preprocessor_config.json")


def model_files(model_dir: Path) -> list[Path]:
    """Top-level files of the model directory that match MODEL_FILES; weights, config and tokenizer are required."""
    if not model_dir.is_dir():
        sys.exit(f"{model_dir}: not a directory")
    files = sorted(p for p in model_dir.iterdir() if p.is_file() and any(fnmatch.fnmatch(p.name, m) for m in MODEL_FILES))
    names = {p.name for p in files}
    if "config.json" not in names or not any(n.endswith(".safetensors") for n in names):
        sys.exit(f"{model_dir}: no config.json or no .safetensors weights")
    if "tokenizer_config.json" not in names or not ("tokenizer.json" in names or {"vocab.json", "merges.txt"} <= names):
        sys.exit(f"{model_dir}: no tokenizer (tokenizer_config.json and tokenizer.json, or vocab.json and merges.txt)")
    return files


def base_licence(path: Path | None) -> Path:
    """The base model's LICENSE file: --base-license, else LICENSE in the first existing BASE_DIRS entry."""
    cands = [path] if path else [d / n for d in BASE_DIRS for n in ("LICENSE", "LICENSE.txt", "LICENSE.md")]
    found = next((p for p in cands if p.is_file()), None)
    if found is None:
        sys.exit("no base-model LICENSE found (looked for " + ", ".join(map(str, cands)) + "); pass --base-license "
                 "<Qwen3.5-2B-Base>/LICENSE: Apache-2.0 4(a) requires a copy with the modified weights")
    if "Apache License" not in found.read_text(errors="replace")[:2000]:
        sys.exit(f"{found}: not the Apache License text")
    return found


def placeholders(path: Path) -> list[str]:
    """'file:line: text' for every line that still holds a placeholder (case-sensitive)."""
    return [f"{path.name}:{i}: {line.strip()[:120]}" for i, line in enumerate(path.read_text().splitlines(), 1)
            if any(m in line for m in PLACEHOLDERS)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="<account>/<name> on the Hugging Face Hub")
    ap.add_argument("--model-dir", required=True, type=Path, help="save_pretrained directory (weights and tokenizer)")
    ap.add_argument("--card", type=Path, default=CARD_DIR / "README.md")
    ap.add_argument("--data-card", type=Path, default=CARD_DIR / "data_card.csv")
    ap.add_argument("--example", type=Path, default=ROOT / "scripts/sev_infer.py", help="inference example the card uses")
    ap.add_argument("--base-license", type=Path, default=None,
                    help="LICENSE file of Qwen3.5-2B-Base (default: LICENSE in " + " or ".join(map(str, BASE_DIRS)) + ")")
    ap.add_argument("--base-notice", type=Path, default=None,
                    help="NOTICE file of Qwen3.5-2B-Base (default: NOTICE next to the base LICENSE, if there is one)")
    ap.add_argument("--public", action="store_true", help="create (or upload to) a public repo; default private")
    ap.add_argument("--allow-placeholders", action="store_true",
                    help="upload even if the cards hold placeholders (private repos only)")
    ap.add_argument("--message", default="Upload Sev-2b", help="commit message")
    ap.add_argument("--dry-run", action="store_true", help="list what would be uploaded and stop")
    args = ap.parse_args()

    if args.public and args.allow_placeholders:
        sys.exit("refusing: --public with --allow-placeholders; a public card must be complete")
    for p in (args.card, args.data_card, args.example):
        if not p.is_file():
            sys.exit(f"missing file: {p}")
    if not args.card.read_text().startswith("---\n"):
        sys.exit(f"{args.card}: no YAML front matter")
    files = model_files(args.model_dir)
    lic = base_licence(args.base_license)
    notice = args.base_notice or (lic.parent / "NOTICE")
    if args.base_notice and not notice.is_file():
        sys.exit(f"missing file: {notice}")
    extra = {"README.md": args.card, "data_card.csv": args.data_card, "sev_infer.py": args.example,
             "LICENSE-Qwen3.5-2B-Base": lic}
    if notice.is_file():
        extra["NOTICE-Qwen3.5-2B-Base"] = notice

    print(f"repo {args.repo} (requested: {'public' if args.public else 'private'})")
    for p in files:
        print(f"  {p.name:40s} {p.stat().st_size / 2**20:10.1f} MiB  from {p}")
    for name, p in extra.items():
        print(f"  {name:40s} {p.stat().st_size / 2**20:10.1f} MiB  from {p}")
    if not notice.is_file():
        print(f"  (no base NOTICE at {notice}; none uploaded)")
    found = placeholders(args.card) + placeholders(args.data_card)
    if found:
        print(f"{len(found)} placeholder line(s):\n  " + "\n  ".join(found))
        if not args.allow_placeholders:
            sys.exit("refusing: fill the placeholders or pass --allow-placeholders (private repos only)")
    if args.dry_run:
        print("dry run: nothing uploaded")
        return

    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    private = api.repo_info(args.repo, repo_type="model").private
    if not private and not args.public:
        sys.exit(f"refusing: {args.repo} already exists and is public; pass --public to upload to it")
    if private and args.public:
        print(f"note: {args.repo} already exists and is private; it stays private (change visibility on the Hub)")
    ops = [CommitOperationAdd(path_in_repo=p.name, path_or_fileobj=str(p)) for p in files]
    ops += [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(p)) for name, p in extra.items()]
    info = api.create_commit(repo_id=args.repo, repo_type="model", operations=ops, commit_message=args.message)
    print(f"uploaded {len(ops)} files in one commit to https://huggingface.co/{args.repo} "
          f"({'private' if private else 'public'}): {info.commit_url}")


if __name__ == "__main__":
    main()
