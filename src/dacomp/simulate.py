"""Gaussian spatio-temporal fields on a regular grid.

Fields are separable: exponential correlation ``exp(-h / a)`` in space and a
stationary AR(1) process with coefficient ``phi`` in time, with unit marginal
variance. Observations are ordered time-major (index = t * S + s).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def grid_coords(side: int) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)


def spatial_cholesky(coords: np.ndarray, scale: float, kind: str = "exp", aniso=(1.0, 1.0)):
    """Cholesky factor of a spatial correlation matrix (None if ``scale == 0``).

    ``kind='exp'`` gives ``exp(-h / a)`` and ``kind='gauss'`` the smoother
    ``exp(-(h / a)^2)``; ``aniso`` stretches the coordinate axes, so that the
    scale along axis j is ``a * aniso[j]``.
    """
    if scale <= 0:
        return None
    z = coords / np.asarray(aniso, float)
    h = cdist(z, z) / scale
    c = np.exp(-h) if kind == "exp" else np.exp(-h ** 2)
    return np.linalg.cholesky(c + 1e-8 * np.eye(len(c)))


def st_fields(rng, chol, n_sites: int, n_times: int, phi: float, n_fields: int) -> np.ndarray:
    """Draw ``n_fields`` independent unit-variance fields, shape (n_fields, T * S)."""
    z = rng.standard_normal((n_fields, n_times, n_sites))
    e = z @ chol.T if chol is not None else z
    out = np.empty_like(e)
    out[:, 0] = e[:, 0]
    innov = np.sqrt(1 - phi ** 2)
    for t in range(1, n_times):
        out[:, t] = phi * out[:, t - 1] + innov * e[:, t]
    return out.reshape(n_fields, n_times * n_sites)


def panel_index(coords: np.ndarray, n_times: int):
    """Spatial coordinates and times of the time-major panel."""
    s = len(coords)
    return np.tile(coords, (n_times, 1)), np.repeat(np.arange(n_times, dtype=float), s)


# Dependence levels used throughout the simulations: spatial scale a (in grid
# units) and temporal AR(1) coefficient phi. Practical ranges are 3a in space
# and -3 / log(phi) in time.
DEPENDENCE_LEVELS = {
    "none": (0.0, 0.0),
    "weak": (1.0, 0.3),
    "moderate": (2.0, 0.6),
    "strong": (4.0, 0.8),
}
