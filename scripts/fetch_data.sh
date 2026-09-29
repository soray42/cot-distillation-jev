#!/usr/bin/env bash
# Download the raw text data (a few MB) into data/raw/ and check hashes (as of 2026-09-30).
# Training source: JustLogic (MIT). Evaluation only: JevBench public tiers (MIT), Typed Decisions test (Apache-2.0).
set -euo pipefail
cd "$(dirname "$0")/.."
get() { mkdir -p "$(dirname "$2")"; [ -s "$2" ] || curl -sfL -o "$2" "$1"; }
HF=https://huggingface.co/datasets
get $HF/michaelchenkj/JustLogic/resolve/main/train_dataset.csv data/raw/justlogic/train_dataset.csv
get $HF/michaelchenkj/JustLogic/resolve/main/validate_dataset.csv data/raw/justlogic/validate_dataset.csv
JB=https://raw.githubusercontent.com/fstandhartinger/jevbench/main
for t in original easy hard; do get $JB/datasets/public/$t.jsonl data/raw/jevbench/$t.jsonl; done
get $HF/LocalLLaMA/typed-decisions/resolve/main/all/test-00000-of-00001.parquet data/raw/typed_decisions/test.parquet
BB=https://raw.githubusercontent.com/google-deepmind/bbeh/main/bbeh/benchmark_tasks   # Apache-2.0, evaluation only
for t in boolean_expressions disambiguation_qa geometric_shapes hyperbaton movie_recommendation nycc \
         shuffled_objects boardgame_qa causal_understanding zebra_puzzles; do
  get $BB/bbeh_$t/task.json data/raw/bbeh/$t.json
done
sha256sum -c <<'SUMS'
e2df49c6f8a8da22173afe6ea291e20de053983a3c2ac54b97b6419e6df26226  data/raw/justlogic/train_dataset.csv
6dd4d1e13ccd58d356e9cbda9c984ad6062e333630cd70364566d585b466ca96  data/raw/justlogic/validate_dataset.csv
5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180  data/raw/jevbench/original.jsonl
231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b  data/raw/jevbench/easy.jsonl
89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb  data/raw/jevbench/hard.jsonl
4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c  data/raw/typed_decisions/test.parquet
SUMS
