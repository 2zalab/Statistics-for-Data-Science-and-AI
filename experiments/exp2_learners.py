"""Experiment 2: gradient-boosting learners on simulated spatio-temporal data.

Covariates x1, x2, x3 and a latent residual w are independent spatio-temporal
Gaussian fields with a common dependence level. The response is

    y = g(x1) + (1 - gamma) g(x2) + x3 + w + noise,   g(x) = sin(2x) + x.

Model A is fitted on (x1, x3, coordinates, time) and model B on
(x2, x3, coordinates, time) with the same learner. Because x1 and x2 are
exchangeable, gamma = 0 gives an exact null (equal expected loss under any
validation scheme); gamma > 0 makes model A truly better.

Both models are compared with (i) random K-fold CV and conventional tests and
(ii) blocked K-fold CV and dependence-aware tests.

Usage: python experiments/exp2_learners.py [--workers 4]
"""

from __future__ import annotations

import argparse
import itertools
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from dacomp import (blocked_folds, corrected_resampled_t, iid_bootstrap, make_blocks,
                    paired_t_folds, random_folds, training_mask)
from dacomp.benchmark import blocked_cv_tests
from dacomp.simulate import DEPENDENCE_LEVELS, grid_coords, panel_index, spatial_cholesky, st_fields

SIDE, N_TIMES, K, R_REPEATS = 10, 20, 5, 5
TILE, TIME_BLOCK = 5, 4
ROOT = Path(__file__).resolve().parents[1]
_CHOL = {}


def learner(seed):
    return XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8,
                        n_jobs=1, tree_method="hist", random_state=seed)


def oof_squared_loss(x, y, folds, seed, masks=None):
    loss = np.empty(len(y))
    for k in np.unique(folds):
        test = folds == k
        train = ~test if masks is None else masks[k]
        model = learner(seed + int(k)).fit(x[train], y[train])
        loss[test] = (y[test] - model.predict(x[test])) ** 2
    return loss


def simulate(rng, level, gamma):
    a, phi = DEPENDENCE_LEVELS[level]
    coords = grid_coords(SIDE)
    if level not in _CHOL:
        _CHOL[level] = spatial_cholesky(coords, a)
    cs, t = panel_index(coords, N_TIMES)
    x1, x2, x3, w = st_fields(rng, _CHOL[level], len(coords), N_TIMES, phi, 4)
    g = lambda v: np.sin(2 * v) + v
    y = g(x1) + (1 - gamma) * g(x2) + x3 + w + rng.normal(0, 0.5, len(t))
    xa = np.column_stack([x1, x3, cs, t])
    xb = np.column_stack([x2, x3, cs, t])
    return cs, t, np.tile(np.arange(len(coords)), N_TIMES), xa, xb, y


def run_rep(job):
    level, gamma, rep, seed = job
    rng = np.random.default_rng([seed, list(DEPENDENCE_LEVELS).index(level), int(gamma * 1000), rep])
    cs, t, sites, xa, xb, y = simulate(rng, level, gamma)
    n = len(y)
    fit_seed = int(rng.integers(1 << 30))
    base = dict(level=level, gamma=gamma, rep=rep)
    rows = []

    # (i) random K-fold, repeated R times (first repetition = single K-fold run)
    split_means, d_first, folds_first = [], None, None
    for r in range(R_REPEATS):
        folds = random_folds(n, K, rng)
        d = oof_squared_loss(xa, y, folds, fit_seed + 100 * r) - oof_squared_loss(xb, y, folds, fit_seed + 100 * r + 50)
        split_means += [d[folds == k].mean() for k in range(K)]
        if r == 0:
            d_first, folds_first = d, folds
    res = [paired_t_folds(d_first, folds_first, method="paired_t"),
           iid_bootstrap(d_first, rng, method="iid_bootstrap"),
           corrected_resampled_t(split_means, n * (K - 1) / K, n / K, method="nadeau_bengio")]
    rows += [dict(base, scheme="random", **r_.as_dict()) for r_ in res]

    # (ii) spatio-temporally blocked K-fold with a one-unit buffer
    blocks = make_blocks(cs, t, TILE, TIME_BLOCK)
    folds = blocked_folds(blocks, K, rng)
    masks = {k: training_mask(folds == k, cs, t, 1.0, 1.0) for k in range(K)}
    d = oof_squared_loss(xa, y, folds, fit_seed + 7, masks) - oof_squared_loss(xb, y, folds, fit_seed + 57, masks)
    res = blocked_cv_tests(d, folds, blocks, cs, t, sites, rng, ablations=False,
                           max_lag_space=SIDE / 2, max_lag_time=N_TIMES / 2)
    rows += [dict(base, scheme="blocked", **r_.as_dict()) for r_ in res]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reps-null", type=int, default=300)
    ap.add_argument("--reps-alt", type=int, default=150)
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    args = ap.parse_args()
    seed = 20261018
    jobs = [(lvl, 0.0, r, seed) for lvl in DEPENDENCE_LEVELS for r in range(args.reps_null)]
    jobs += [(lvl, gm, r, seed) for lvl, gm in itertools.product(DEPENDENCE_LEVELS, args.gammas)
             for r in range(args.reps_alt)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        rows = list(itertools.chain.from_iterable(pool.imap_unordered(run_rep, jobs, chunksize=2)))
    out = ROOT / "results" / "exp2.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(jobs)} replicates in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
