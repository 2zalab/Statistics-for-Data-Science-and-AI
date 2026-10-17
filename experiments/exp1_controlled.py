"""Experiment 1: controlled loss-differential fields.

Prediction errors of two models are simulated directly as spatio-temporal
Gaussian fields, so that the true expected loss differential is known exactly:
Delta = sigma_A^2 - sigma_B^2 = -delta. Fold-level training effects with
between-fold correlation ``rho_eta`` mimic the variability induced by
refitting the models on overlapping training sets.

Usage: python experiments/exp1_controlled.py [--pilot] [--workers 4]
"""

from __future__ import annotations

import argparse
import itertools
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from dacomp import blocked_folds, make_blocks
from dacomp.benchmark import blocked_cv_tests
from dacomp.simulate import DEPENDENCE_LEVELS, grid_coords, panel_index, spatial_cholesky, st_fields

SIDE, N_TIMES, K = 12, 24, 5
TILE, TIME_BLOCK = 4, 6
SIGMA_ETA, RHO_ETA = 0.05, 0.25
ROOT = Path(__file__).resolve().parents[1]

_CHOL = {}


def run_rep(job):
    level, delta, rep, seed, kappas = job
    a, phi = DEPENDENCE_LEVELS[level]
    coords = grid_coords(SIDE)
    if level not in _CHOL:
        _CHOL[level] = spatial_cholesky(coords, a)
    rng = np.random.default_rng([seed, list(DEPENDENCE_LEVELS).index(level), int(delta * 1000), rep])
    cs, t = panel_index(coords, N_TIMES)
    sites = np.tile(np.arange(len(coords)), N_TIMES)
    eps, e_a, e_b = st_fields(rng, _CHOL[level], len(coords), N_TIMES, phi, 3)
    blocks = make_blocks(cs, t, TILE, TIME_BLOCK)
    folds = blocked_folds(blocks, K, rng)
    common = rng.standard_normal()
    eta = SIGMA_ETA * (np.sqrt(RHO_ETA) * common + np.sqrt(1 - RHO_ETA) * rng.standard_normal(K))
    d = (eps - e_a) ** 2 - (eps - np.sqrt(1 + delta) * e_b) ** 2 + eta[folds]
    res = blocked_cv_tests(d, folds, blocks, cs, t, sites, rng, kappas=kappas,
                           fixed_bandwidth=(TILE, TIME_BLOCK),
                           max_lag_space=SIDE / 2, max_lag_time=N_TIMES / 2, rhos=(0.0, 0.5))
    return [dict(level=level, delta=delta, rep=rep, truth=-delta, **r.as_dict()) for r in res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="bandwidth-multiplier pilot on separate seeds")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reps-null", type=int, default=1000)
    ap.add_argument("--reps-alt", type=int, default=300)
    args = ap.parse_args()

    if args.pilot:
        seed, kappas, out = 999_001, (0.5, 1.0, 1.5, 2.0), ROOT / "results" / "exp1_pilot.csv"
        jobs = [(lvl, 0.0, r, seed, kappas) for lvl in DEPENDENCE_LEVELS for r in range(300)]
    else:
        seed, kappas, out = 20261017, (1.0,), ROOT / "results" / "exp1.csv"
        jobs = [(lvl, 0.0, r, seed, kappas) for lvl in DEPENDENCE_LEVELS for r in range(args.reps_null)]
        jobs += [(lvl, dl, r, seed, kappas) for lvl, dl in itertools.product(DEPENDENCE_LEVELS, (0.1, 0.2, 0.4))
                 for r in range(args.reps_alt)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        rows = list(itertools.chain.from_iterable(pool.imap_unordered(run_rep, jobs, chunksize=4)))
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(jobs)} replicates in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
