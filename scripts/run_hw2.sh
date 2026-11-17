#!/usr/bin/env bash
# Reproduce HW2 on the seven original B2 checkpoints or train fresh references.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
    cat <<'HELP'
Usage: bash scripts/run_hw2.sh [--rebuild-baseline | --analyze]
  No option          Use the seven local B2 checkpoints and their archived predictions.
  --rebuild-baseline Train B2 on the seven fold/seed pairs into a separate local directory.
  --analyze          Recompute tables, confidence intervals and figures from published predictions.
HELP
}

if [[ $# -gt 1 ]]; then usage; exit 2; fi
case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    ""|--rebuild-baseline|--analyze) ;;
    *) usage; exit 2 ;;
esac

result_dir=results/hw2
run_heads=(uv run --frozen python -u -m bengali.hw2_run --config configs/hw2.yaml)
if [[ ${1:-} == --rebuild-baseline ]]; then
    reference_dir=data/hw2-rebuilt/reference
    result_dir=data/hw2-rebuilt/results
    train_b2=(uv run --frozen python -m bengali.train --config configs/baseline.yaml)
    train_settings=(experiment=b2_hw2_rebuilt "output_dir=$reference_dir")
    "${train_b2[@]}" --fold 0 1 2 3 4 --seed 42 --set "${train_settings[@]}"
    "${train_b2[@]}" --fold 0 --seed 43 44 --set "${train_settings[@]}"
    run_heads+=(--set hw2.checkpoint_dir=checkpoints/b2_hw2_rebuilt "hw2.archive_dir=$reference_dir" "output_dir=$result_dir")
fi

if [[ ${1:-} != --analyze ]]; then
    "${run_heads[@]}"
fi
uv run --frozen python -m bengali.hw2_evaluate --results "$result_dir"
uv run --frozen python -m bengali.hw2_diagnostics --results "$result_dir"
uv run --frozen python -m bengali.hw2_image_bootstrap --results "$result_dir"
