#!/usr/bin/env bash
# ~2 minute end-to-end check on a small subset (needs data/processed + splits/folds.csv).
set -euo pipefail
cd "$(dirname "$0")/.."
uv run pytest -q
uv run python -m bengali.train --config configs/smoke.yaml
