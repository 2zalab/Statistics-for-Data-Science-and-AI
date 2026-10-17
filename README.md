# Are Machine Learning Improvements Statistically Real?

Code, data and manuscript for the paper

> **Are Machine Learning Improvements Statistically Real? Dependence-Aware Inference for Reliable Model Comparison on Spatiotemporal Data**
> Isaac Touza, University of Maroua.
> Prepared for the *Special Issue on Statistics for Data Science and AI*, International Journal of Data Science and Analytics (Springer Nature).

When two machine learning models are compared on spatially or temporally
correlated data, standard tests (paired *t*-test on fold scores, bootstrap of
observations, corrected resampled *t*-test) can declare improvements that are
not real. This repository implements **DA-HAC**, a dependence-aware test for
the expected loss differential between two models evaluated by blocked
cross-validation, and the simulation and case studies used to evaluate it.

## Repository layout

```
src/dacomp/            Python package
  splitting.py         random and spatio-temporally blocked K-fold partitions, buffers
  variogram.py         directional empirical variograms, exponential fits, practical ranges
  inference.py         paired t, corrected resampled t, i.i.d./block bootstrap,
                       mixed model, Diebold-Mariano-HAC, DA-HAC (proposed)
  simulate.py          separable spatio-temporal Gaussian fields
  benchmark.py         runs all blocked-CV procedures on one set of loss differentials
experiments/
  exp1_controlled.py   Experiment 1: simulated loss-differential fields (exact truth)
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

Using DA-HAC on your own comparison only requires the per-observation
out-of-fold losses of the two models, the fold labels and the coordinates:

```python
import numpy as np
from dacomp import make_blocks, blocked_folds, da_hac

blocks = make_blocks(coords, times, space_tile=50_000, time_block=24 * 14)
folds = blocked_folds(blocks, n_folds=5, rng=np.random.default_rng(0))
# ... fit models A and B on each training fold, predict the held-out fold ...
d = loss_A - loss_B                      # one value per observation
res = da_hac(d, folds, coords_space=coords, times=times)
print(res.estimate, res.ci_low, res.ci_high, res.pvalue)
```

## Reproducing the paper

```bash
python experiments/exp1_controlled.py --pilot   # bandwidth pilot (separate seeds)
python experiments/exp1_controlled.py           # Experiment 1
python experiments/exp2_learners.py             # Experiment 2
python experiments/exp3_real_data.py            # Experiment 3
python experiments/analyze.py                   # tables and figures
cd paper && make                                # manuscript
```

All runs are seeded. On a 4-core machine the experiments take about three hours.

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
