"""Experiment 1b: sensitivity, robustness and calibration of the decision rule.

Same construction as Experiment 1 (simulated error fields with exact truth),
under three families of scenarios:

* ``rho``    true correlation of the fold effects rho_eta in {0, .25, .5, .75}
             with a larger training variance (sigma_eta = 0.1), no and moderate
             dependence; DA-HAC is run with assumed rho in {0, 1/K, 0.5}.
* ``robust`` moderate dependence with departures from the main design:
             anisotropy, Gaussian covariance, heteroscedastic errors,
             non-stationary dependence and irregularly located sites.
* ``range``  a grid of dependence ranges, to choose (pilot seeds) and then
             validate (main seeds) the range-ratio threshold of the
             three-way decision rule.

Usage: python experiments/exp1b_robustness.py --set rho|robust|range [--pilot]
"""

from __future__ import annotations

import argparse
import itertools
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from dacomp import blocked_folds, make_blocks
from dacomp.benchmark import blocked_cv_tests
from dacomp.simulate import grid_coords, spatial_cholesky, st_fields

SIDE, N_TIMES, K = 12, 24, 5
TILE, TIME_BLOCK = 4, 6
ROOT = Path(__file__).resolve().parents[1]

BASE = dict(a=2.0, phi=0.6, kind="exp", aniso=(1.0, 1.0), hetero=False, nonstat=False,
            irregular=False, sigma_eta=0.05, rho_eta=0.25)


def scenarios(which):
    if which == "rho":
        out = {}
        for lvl, (a, phi) in {"none": (0.0, 0.0), "moderate": (2.0, 0.6)}.items():
            for r in (0.0, 0.25, 0.5, 0.75):
                out[f"{lvl}_rho{r:g}"] = dict(BASE, a=a, phi=phi, sigma_eta=0.1, rho_eta=r)
        return out
    if which == "robust":
        return {"baseline": dict(BASE),
                "anisotropic": dict(BASE, aniso=(2.0, 0.5)),
                "gaussian_cov": dict(BASE, kind="gauss"),
                "heteroscedastic": dict(BASE, hetero=True),
                "nonstationary": dict(BASE, nonstat=True),
                "irregular_sites": dict(BASE, irregular=True)}
    if which == "range":
        return {f"a{a:g}": dict(BASE, a=a, phi=float(np.exp(-1.0 / a)))
                for a in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)}
    raise ValueError(which)


_CACHE = {}


def site_coords(sc):
    if sc["irregular"]:
        return np.random.default_rng(12345).uniform(0, SIDE - 1, (SIDE * SIDE, 2))
    return grid_coords(SIDE)


def fields(rng, sc, coords, n_fields):
    key = (sc["a"], sc["kind"], sc["aniso"], sc["irregular"], sc["nonstat"])
    if key not in _CACHE:
        if sc["nonstat"]:  # short range on the left of the domain, long range on the right
            _CACHE[key] = (spatial_cholesky(coords, 0.5), spatial_cholesky(coords, 4.0))
        else:
            _CACHE[key] = spatial_cholesky(coords, sc["a"], sc["kind"], sc["aniso"])
    s = len(coords)
    if sc["nonstat"]:
        c_short, c_long = _CACHE[key]
        w = np.clip(coords[:, 0] / (SIDE - 1), 0, 1)
        f1 = st_fields(rng, c_short, s, N_TIMES, sc["phi"], n_fields)
        f2 = st_fields(rng, c_long, s, N_TIMES, sc["phi"], n_fields)
        wt = np.tile(w, N_TIMES)
        f = np.sqrt(1 - wt) * f1 + np.sqrt(wt) * f2
    else:
        f = st_fields(rng, _CACHE[key], s, N_TIMES, sc["phi"], n_fields)
    if sc["hetero"]:  # error standard deviation varies smoothly over space (ratio about 4.5)
        sd = np.exp(0.75 * (coords[:, 1] / (SIDE - 1) - 0.5) * 2)
        f = f * np.tile(sd, N_TIMES)
    return f


def run_rep(job):
    which, name, sc, delta, rep, seed = job
    rng = np.random.default_rng([seed, zlib.crc32(name.encode()), int(delta * 1000), rep])
    coords = site_coords(sc)
    cs = np.tile(coords, (N_TIMES, 1))
    t = np.repeat(np.arange(N_TIMES, dtype=float), len(coords))
    sites = np.tile(np.arange(len(coords)), N_TIMES)
    eps, e_a, e_b = fields(rng, sc, coords, 3)
    blocks = make_blocks(cs, t, TILE, TIME_BLOCK)
    folds = blocked_folds(blocks, K, rng)
    r = sc["rho_eta"]
    eta = sc["sigma_eta"] * (np.sqrt(r) * rng.standard_normal() + np.sqrt(1 - r) * rng.standard_normal(K))
    d = (eps - e_a) ** 2 - (eps - np.sqrt(1 + delta) * e_b) ** 2 + eta[folds]
    truth = -delta * (np.mean(np.tile(np.exp(0.75 * (coords[:, 1] / (SIDE - 1) - 0.5) * 2), 1) ** 2)
                      if sc["hetero"] else 1.0)
    res = blocked_cv_tests(d, folds, blocks, cs, t, sites, rng, ablations=False,
                           max_lag_space=SIDE / 2, max_lag_time=N_TIMES / 2,
                           rhos=(0.0, 0.5) if which == "rho" else ())
    return [dict(set=which, scenario=name, delta=delta, rep=rep, truth=truth, **x.as_dict()) for x in res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["rho", "robust", "range"])
    ap.add_argument("--pilot", action="store_true", help="separate seeds (range threshold choice)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    sc = scenarios(args.set)
    seed = 777_001 if args.pilot else 20261019
    n_null = {"rho": 500, "robust": 500, "range": 300 if args.pilot else 500}[args.set]
    jobs = [(args.set, nm, s, 0.0, r, seed) for nm, s in sc.items() for r in range(n_null)]
    if args.set == "robust":
        jobs += [(args.set, nm, s, 0.2, r, seed) for nm, s in sc.items() for r in range(300)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        rows = list(itertools.chain.from_iterable(pool.imap_unordered(run_rep, jobs, chunksize=4)))
    out = ROOT / "results" / f"exp1b_{args.set}{'_pilot' if args.pilot else ''}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(jobs)} replicates in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
