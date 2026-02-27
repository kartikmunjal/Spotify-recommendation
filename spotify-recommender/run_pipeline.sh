#!/usr/bin/env bash
set -euo pipefail

python src/pipeline.py \
  --account-dir "../Spotify Account Data" \
  --extended-dir "../Spotify Extended Streaming History" \
  --tech-dir "../Spotify Technical Log Information" \
  --output-dir "./outputs"

# Optional visualization step (generates SVG charts + dashboard)
python src/visualize.py \
  --scored-samples "./outputs/scored_samples.csv" \
  --out-dir "./outputs/figures" || true
