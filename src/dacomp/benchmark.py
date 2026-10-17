"""Run the full battery of comparison procedures on one set of loss differentials."""

from __future__ import annotations

import numpy as np

from .inference import (block_bootstrap, da_hac, dm_hac, hering_genton, iid_bootstrap,
                        mixed_model, paired_t_folds)
from .variogram import dependence_ranges


def blocked_cv_tests(d, folds, blocks, coords_space, times, sites, rng, alpha=0.05,
                     kappas=(1.0,), ablations=True, fixed_bandwidth=None,
                     max_lag_space=None, max_lag_time=None, suffix="", rhos=()):
    """Procedures applied to loss differentials from a blocked K-fold partition.

    Returns a list of :class:`~dacomp.inference.TestResult`; method names get
    ``suffix`` appended.
    """
    res = [
        paired_t_folds(d, folds, alpha, method="paired_t"),
        iid_bootstrap(d, rng, alpha=alpha, method="iid_bootstrap"),
        block_bootstrap(d, blocks, rng, alpha=alpha, method="block_bootstrap"),
        mixed_model(d, blocks, alpha, method="mixed_model"),
    ]
    if times is not None and sites is not None:
        res.append(dm_hac(d, sites, times, alpha, method="dm_hac"))
    res.append(hering_genton(d, coords_space, times, alpha, max_lag_space, max_lag_time,
                             method="hering_genton"))

    _, f = np.unique(folds, return_inverse=True)
    u = d - (np.bincount(f, weights=d) / np.bincount(f))[f]
    r_s, r_t, fits = dependence_ranges(u, coords_space, times, max_lag_space, max_lag_time)
    ranges = (r_s, r_t)

    # Proposed procedure (DA-P) and its variants, sharing one parametric variance computation
    variants = [("da_p", True, "satterthwaite")]
    if ablations:
        variants += [("da_p_no_tau", False, "satterthwaite"), ("da_p_normal", True, "normal")]
    variants += [(f"da_p_rho{r:g}", True, "satterthwaite", r) for r in rhos]
    res += da_hac(d, folds, coords_space, times, alpha, ranges=ranges, fits=fits,
                  estimator="parametric", variants=variants)

    # Kernel (HAC) version DA-K, formerly the main procedure
    for kappa in kappas:
        name = "da_hac" if kappa == 1.0 else f"da_hac_k{kappa:g}"
        res.append(da_hac(d, folds, coords_space, times, alpha, kappa=kappa, ranges=ranges, method=name))
    if ablations and fixed_bandwidth is not None:
        res.append(da_hac(d, folds, coords_space, times, alpha, bandwidth=fixed_bandwidth,
                          method="hac_block_bw"))
    for r in res:
        r.method += suffix
    return res
