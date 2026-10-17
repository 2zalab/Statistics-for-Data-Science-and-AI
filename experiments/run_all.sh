#!/usr/bin/env bash
# Run all experiments in sequence (about three hours on four cores).
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/exp1_controlled.py --workers 4
python experiments/exp2_learners.py --workers 4
python experiments/exp3_real_data.py
python experiments/analyze.py
