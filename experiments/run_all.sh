#!/usr/bin/env bash
# Run all experiments in sequence (about five hours on four cores).
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/exp1_controlled.py --workers 4
for set in rho robust; do python experiments/exp1b_robustness.py --set $set; done
python experiments/exp1b_robustness.py --set range --pilot
python experiments/exp1b_robustness.py --set range
python experiments/exp2_learners.py --workers 4
python experiments/exp3_real_data.py
python experiments/analyze.py
