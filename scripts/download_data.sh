#!/usr/bin/env bash
# Download the competition data (5.18 GB) into data/raw.
# Prerequisites: (1) accept the rules at https://www.kaggle.com/competitions/bengaliai-cv19/rules
#                (2) Kaggle API token in ~/.kaggle/kaggle.json (or KAGGLE_USERNAME/KAGGLE_KEY env vars)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
uv run kaggle competitions download -c bengaliai-cv19 -p data/raw
unzip -o -q data/raw/bengaliai-cv19.zip -d data/raw
rm -f data/raw/bengaliai-cv19.zip
ls -lh data/raw
