#!/usr/bin/env bash
# Full HW1 pipeline: all baselines on the same 5 folds + seed study + evaluation.
# Approximate wall-clock on Apple M4 Pro (MPS): ~2.8 h, dominated by B2.
# The main baseline (B2) and its seed study run right after the cheap baselines; the optional frozen-encoder
# probes (B3, B4) run last and are non-fatal, so a failure there cannot block B2 or the evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
uv run python -m bengali.baselines --config configs/constant.yaml          2>&1 | tee logs/b0.log    # B0, seconds
uv run python -m bengali.probe     --config configs/probe_pixels.yaml      2>&1 | tee logs/b1.log    # B1, ~2 min
uv run python -m bengali.train     --config configs/baseline.yaml          2>&1 | tee logs/b2.log    # B2, 5 folds x ~16 min
uv run python -m bengali.train     --config configs/baseline.yaml --fold 0 --seed 43 44 2>&1 | tee logs/b2_seeds.log  # RQ2, ~30 min
uv run python -m bengali.probe     --config configs/probe_dinov3.yaml      2>&1 | tee logs/b3.log \
  || echo "B3 (DINOv3 probe) failed, continuing"                                                 # ~12 min extraction + probes
uv run python -m bengali.probe     --config configs/probe_mobileclip2.yaml 2>&1 | tee logs/b4.log \
  || echo "B4 (MobileCLIP2 probe) failed, continuing"                                            # ~12 min extraction + probes
uv run python -m bengali.train     --config configs/dinov3_partial.yaml    2>&1 | tee logs/b5.log \
  || echo "B5 (DINOv3, last 4 blocks tuned) failed, continuing"                                  # fold 0 only, ~17 min
uv run python -m bengali.evaluate                                          2>&1 | tee logs/evaluate.log
