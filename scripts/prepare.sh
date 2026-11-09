#!/usr/bin/env bash
# Preprocess images (crop + resize to 128x128), build the fold file, run EDA.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m bengali.data --size 128 --workers "${WORKERS:-8}"
uv run python -m bengali.splits --n-folds 5 --ood-frac 0.1 --seed 42   # deterministic; splits/folds.csv is committed
uv run python -m bengali.eda
