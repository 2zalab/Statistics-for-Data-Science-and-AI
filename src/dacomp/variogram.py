"""Directional empirical variograms of loss differentials and range estimation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

_FAR = 1e7  # scaling that keeps pairs from different strata out of a KD-tree query


def _subsample(n, max_points, rng):
    if n <= max_points:
        return np.arange(n)
    return np.sort(rng.choice(n, max_points, replace=False))


def empirical_variogram(u, coords_space=None, times=None, direction="space",
                        max_lag=None, n_bins=15, max_points=20000, seed=0):
    """Empirical semivariogram of ``u`` along one direction.

    ``direction='space'`` uses pairs observed at the same time (or all pairs if
    there is no time axis) binned by Euclidean distance; ``direction='time'``
    uses pairs at the same spatial location binned by integer time lag.

    Returns ``(lags, gamma, counts)``.
    """
    u = np.asarray(u, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(u)
    idx = _subsample(n, max_points, rng)
    u = u[idx]
    cs = None if coords_space is None else np.asarray(coords_space, float).reshape(n, -1)[idx]
    t = None if times is None else np.asarray(times, float)[idx]

    if direction == "space":
        if cs is None:
            raise ValueError("spatial variogram needs coordinates")
        z = cs if t is None else np.hstack([cs, t[:, None] * _FAR])
        if max_lag is None:
            max_lag = 0.5 * np.max(cs.max(0) - cs.min(0))
        pairs = cKDTree(z).query_pairs(r=max_lag, p=2, output_type="ndarray")
        h = np.linalg.norm(cs[pairs[:, 0]] - cs[pairs[:, 1]], axis=1)
        edges = np.linspace(0, max_lag, n_bins + 1)
    elif direction == "time":
        if t is None:
            raise ValueError("temporal variogram needs times")
        z = t[:, None] if cs is None else np.hstack([cs * _FAR, t[:, None]])
        if max_lag is None:
            max_lag = 0.25 * (t.max() - t.min())
        pairs = cKDTree(z).query_pairs(r=max_lag + 0.5, p=np.inf, output_type="ndarray")
        h = np.abs(t[pairs[:, 0]] - t[pairs[:, 1]])
        n_bins = int(min(n_bins, max(1, np.ceil(max_lag))))
        edges = np.linspace(0, max_lag, n_bins + 1)
        edges[0] = 0.5  # exclude zero lag (repeated times at one site are not expected)
    else:
        raise ValueError(direction)

    if len(pairs) == 0:
        return np.array([]), np.array([]), np.array([])
    sq = 0.5 * (u[pairs[:, 0]] - u[pairs[:, 1]]) ** 2
    which = np.digitize(h, edges) - 1
    ok = (which >= 0) & (which < len(edges) - 1)
    counts = np.bincount(which[ok], minlength=len(edges) - 1)
    sums = np.bincount(which[ok], weights=sq[ok], minlength=len(edges) - 1)
    hsum = np.bincount(which[ok], weights=h[ok], minlength=len(edges) - 1)
    keep = counts > 0
    return hsum[keep] / counts[keep], sums[keep] / counts[keep], counts[keep]


def fit_exponential_variogram(lags, gamma, counts, sill=None):
    """Weighted least-squares fit of ``c0 + c1 (1 - exp(-h / a))``.

    Weights follow Cressie (1985): ``N_h / gamma_model(h)^2``. Returns a dict
    with nugget ``c0``, partial sill ``c1``, scale ``a`` and practical range
    ``3a`` (distance at which correlation drops to about 5%). The practical
    range is set to zero when the structured part explains less than 5% of the
    sill.
    """
    lags, gamma, counts = map(np.asarray, (lags, gamma, counts))
    if len(lags) < 3 or np.all(gamma <= 0):
        return {"c0": float(np.mean(gamma)) if len(gamma) else 0.0, "c1": 0.0, "a": 0.0, "range": 0.0}
    top = float(np.average(gamma, weights=counts))
    if sill is None:
        sill = top
    hmax = float(lags.max())

    def resid(p):
        c0, c1, a = p
        model = c0 + c1 * (1 - np.exp(-lags / a))
        return np.sqrt(counts) * (gamma - model) / np.maximum(model, 1e-12 * top)

    starts = [(0.5 * sill, 0.5 * sill, hmax / 3), (0.1 * sill, 0.9 * sill, hmax / 10),
              (0.9 * sill, 0.1 * sill, hmax), (0.1 * sill, 0.9 * sill, hmax / 30)]
    best = None
    for s in starts:
        try:
            r = least_squares(resid, s, bounds=([0, 0, 1e-6 * hmax], [10 * top, 10 * top, 3 * hmax]))
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    c0, c1, a = best.x
    structured = c1 / max(c0 + c1, 1e-300)
    rng_ = 3 * a if structured >= 0.05 else 0.0
    return {"c0": float(c0), "c1": float(c1), "a": float(a), "range": float(rng_)}


def dependence_ranges(u, coords_space=None, times=None, max_lag_space=None, max_lag_time=None,
                      n_bins=15, max_points=20000, seed=0):
    """Practical ranges of dependence of ``u`` in space and in time.

    Returns ``(range_space, range_time, fits)`` where a missing axis has
    range ``None``.
    """
    fits = {}
    r_s = r_t = None
    if coords_space is not None:
        v = empirical_variogram(u, coords_space, times, "space", max_lag_space, n_bins, max_points, seed)
        fits["space"] = (v, fit_exponential_variogram(*v))
        r_s = fits["space"][1]["range"]
    if times is not None:
        v = empirical_variogram(u, coords_space, times, "time", max_lag_time, n_bins, max_points, seed)
        fits["time"] = (v, fit_exponential_variogram(*v))
        r_t = fits["time"][1]["range"]
    return r_s, r_t, fits
