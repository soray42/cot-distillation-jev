#!/usr/bin/env bash
# From ~/cotd: packs the outputs of every finished 12B run (metrics, predictions, mid-training predictions, logs; no
# weights) into results_12b_<date>.tgz to send back. Small (tens of MB).
set -euo pipefail
cd "$HOME/cotd"
OUT=results_12b_$(date +%Y%m%d_%H%M).tgz
tar czf "$OUT" $(ls -d runs/*gemma-4-12B* 2>/dev/null | while read d; do ls "$d"/metrics.json "$d"/preds_*.jsonl "$d"/trajectory.json "$d"/mid 2>/dev/null; done) logs
ls -la "$OUT"
