#!/usr/bin/env bash
# Tests, training (4 variants x 3 seeds, ~25 min on 2 cores), evaluation, direction analysis, plots.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pytest -q tests
python3 scripts/train.py --seeds 3
python3 scripts/evaluate.py
python3 scripts/direction.py
python3 scripts/plot.py
