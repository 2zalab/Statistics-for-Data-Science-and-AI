# Are Machine Learning Improvements Statistically Real?

Code, data and manuscript for the paper

> **Are Machine Learning Improvements Statistically Real? Dependence-Aware Inference for Reliable Model Comparison on Spatiotemporal Data**
> Isaac Touza (University of Maroua) and Baudoin Nguimeya Tsofack (University of Yaoundé I).
> Prepared for the *Special Issue on Statistics for Data Science and AI*, International Journal of Data Science and Analytics (Springer Nature).

When two machine learning models are compared on spatially or temporally
correlated data, standard tests (paired *t*-test on fold scores, bootstrap of
observations, corrected resampled *t*-test) can declare improvements that are
not real. This repository implements **DA-P**, a dependence-aware test for
the expected loss differential between two learning procedures evaluated by
blocked cross-validation (variogram-based test-sampling variance + training
variance + Satterthwaite reference, with a three-way decision rule), its kernel
variant DA-K, a space-time Hering-Genton test and the conventional procedures,
together with the simulation and case studies used to evaluate them.

## Repository layout

```
src/dacomp/            Python package
  splitting.py         random and spatio-temporally blocked K-fold partitions, buffers
  variogram.py         directional empirical variograms, exponential fits, practical ranges
  inference.py         paired t, corrected resampled t, i.i.d./block bootstrap,
                       mixed model, Diebold-Mariano-HAC, Hering-Genton,
                       DA-K (kernel) and DA-P (proposed, `da_test`)
  simulate.py          separable spatio-temporal Gaussian fields
  benchmark.py         runs all blocked-CV procedures on one set of loss differentials
experiments/
  exp1_controlled.py   Experiment 1: simulated loss-differential fields (exact truth)
  exp1b_robustness.py  training correlation, misspecification, range grid
  pilot/               pilot studies on separate seeds (design choices)
  exp2_learners.py     Experiment 2: XGBoost learners with exchangeable features
  exp3_real_data.py    Experiment 3: NYC weather, Melbourne pedestrians, Las Rosas maize
  analyze.py           tables (paper/tables) and figures (paper/figures)
data/raw/              public data sets (from Rdatasets, see below)
results/               raw simulation and case-study outputs (CSV)
paper/                 LaTeX manuscript (Springer Nature sn-jnl template)
tests/                 unit tests
```

## Quick start

```bash
pip install -e ".[test]"
pytest
```

Using DA-P on your own comparison only requires the per-observation
out-of-fold losses of the two models, the fold labels and the coordinates:

```python
import numpy as np
from dacomp import make_blocks, blocked_folds, da_test

blocks = make_blocks(coords, times, space_tile=50_000, time_block=24 * 14)
folds = blocked_folds(blocks, n_folds=5, rng=np.random.default_rng(0))
# ... fit models A and B on each training fold, predict the held-out fold ...
d = loss_A - loss_B                      # one value per observation
res = da_test(d, folds, coords_space=coords, times=times)              # rho0 = 1/K
res05 = da_test(d, folds, coords_space=coords, times=times, rho=0.5)  # sensitivity
print(res.estimate, res.ci_low, res.ci_high, res.pvalue, res.info)
# three-way rule: improvement only if both p-values < 0.05; indeterminate if only res.pvalue < 0.05
```

## Reproducing the paper

```bash
# pilots on separate seeds (design choices reported in the paper)
python experiments/exp1_controlled.py --pilot
python experiments/pilot/exp1_pilot_df.py
python experiments/pilot/pilot_variants2.py exp1
python experiments/pilot/pilot_variants2.py exp2
# main studies
python experiments/exp1_controlled.py           # Experiment 1
python experiments/exp1b_robustness.py --set rho
python experiments/exp1b_robustness.py --set robust
python experiments/exp1b_robustness.py --set range --pilot
python experiments/exp1b_robustness.py --set range
python experiments/exp2_learners.py             # Experiment 2
python experiments/exp3_real_data.py            # case studies
python experiments/analyze.py                   # tables and figures
cd paper && make                                # manuscript
```

All runs are seeded; re-running Experiment 1 reproduces the stored results exactly. On a 4-core machine the main studies take about five hours.

## Data

The three data sets are redistributed from the
[Rdatasets](https://vincentarelbundock.github.io/Rdatasets/) collection:

| File | Source R package | Description |
|---|---|---|
| `nyc_weather.csv` | `nycflights13` (CC0) | hourly weather at EWR, JFK and LGA airports, 2013 |
| `pedestrian.csv` | `tsibble` | hourly pedestrian counts at four Melbourne sensors, 2015-2016 |
| `lasrosas.corn.csv` | `agridat` | yield-monitor maize data, Las Rosas (Argentina), 1999 and 2001 |

## License

Code: MIT. Data sets retain the licenses of their original packages.
