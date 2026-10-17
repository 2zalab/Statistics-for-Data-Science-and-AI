"""Pilot (separate seeds): parametric-covariance version of the DA variance in
Experiments 1 and 2. Run from the repository root:
python experiments/pilot/pilot_variants2.py exp1|exp2
"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "experiments")
from dacomp import blocked_folds, hering_genton, make_blocks, paired_t_folds, training_mask  # noqa: E402
from dacomp.inference import hac_variance, parametric_variance  # noqa: E402
from dacomp.simulate import DEPENDENCE_LEVELS, grid_coords, panel_index, spatial_cholesky, st_fields  # noqa: E402
from dacomp.variogram import dependence_ranges  # noqa: E402

_CH = {}


def pval(est, v, df):
    ref = stats.t(df) if np.isfinite(df) else stats.norm()
    return 2 * ref.sf(abs(est) / np.sqrt(v))


def variants(d, folds, cs, t, side, T):
    n, K = len(d), len(np.unique(folds))
    fm = np.bincount(folds, weights=d) / np.bincount(folds)
    u = d - fm[folds]
    r_s, r_t, fits = dependence_ranges(u, cs, t, side / 2, T / 2)
    zeta = 1 / K + 1 / (K - 1)
    ext = [np.ptp(cs[:, 0]), np.ptp(cs[:, 1]), np.ptp(t)]
    out = {}
    for name, (vu, vk) in {"A": hac_variance(u, cs, t, r_s or 0, r_t or 0, groups=folds),
                           "P": parametric_variance(u, cs, t, fits, groups=folds)}.items():
        neff = np.clip(np.mean(u ** 2) / vu, 2, n)
        bw = [r_s, r_s, r_t]
        nuk = np.prod([max(1, 1.5 * L / b) for L, b in zip(ext, bw) if b]) if any(bw) else np.inf
        nu_u = min(neff - 1, nuk)
        tau_raw = fm.var(ddof=1) - vk.mean()
        tau = max(0.0, tau_raw)
        v = vu + zeta * tau
        den = vu ** 2 / nu_u + ((zeta * tau) ** 2 / (K - 1) if tau > 0 else 0)
        df = v ** 2 / den
        out[name] = pval(d.mean(), v, df)
        out[name + "E"] = pval(d.mean(), v, df if tau_raw > 0 else K - 1)
    out["HG"] = hering_genton(d, cs, t, 0.05, side / 2, T / 2).pvalue
    out["pt"] = paired_t_folds(d, folds).pvalue
    return out


def run1(job):
    level, rep = job
    a, phi = DEPENDENCE_LEVELS[level]
    coords = grid_coords(12)
    if level not in _CH:
        _CH[level] = spatial_cholesky(coords, a)
    rng = np.random.default_rng([555_001, list(DEPENDENCE_LEVELS).index(level), rep])
    cs, t = panel_index(coords, 24)
    eps, ea, eb = st_fields(rng, _CH[level], len(coords), 24, phi, 3)
    folds = blocked_folds(make_blocks(cs, t, 4, 6), 5, rng)
    eta = 0.05 * (np.sqrt(.25) * rng.standard_normal() + np.sqrt(.75) * rng.standard_normal(5))
    d = (eps - ea) ** 2 - (eps - eb) ** 2 + eta[folds]
    return dict(level=level, rep=rep, **variants(d, folds, cs, t, 12, 24))


def run2(job):
    import exp2_learners as e2
    level, rep = job
    rng = np.random.default_rng([555_002, list(DEPENDENCE_LEVELS).index(level), rep])
    cs, t, sites, xa, xb, y = e2.simulate(rng, level, 0.0)
    blocks = make_blocks(cs, t, e2.TILE, e2.TIME_BLOCK)
    folds = blocked_folds(blocks, e2.K, rng)
    masks = {k: training_mask(folds == k, cs, t, 1.0, 1.0) for k in range(e2.K)}
    s = int(rng.integers(1 << 30))
    d = e2.oof_squared_loss(xa, y, folds, s, masks) - e2.oof_squared_loss(xb, y, folds, s + 50, masks)
    return dict(level=level, rep=rep, **variants(d, folds, cs, t, e2.SIDE, e2.N_TIMES))


if __name__ == "__main__":
    which = sys.argv[1]
    reps = 300 if which == "exp1" else 120
    jobs = [(l, r) for l in DEPENDENCE_LEVELS for r in range(reps)]
    with Pool(4) as p:
        rows = p.map(run1 if which == "exp1" else run2, jobs, chunksize=2)
    df = pd.DataFrame(rows)
    df.to_csv(f"results/pilot_variants2_{which}.csv", index=False)
    cols = [c for c in df.columns if c not in ("level", "rep")]
    print(which)
    print((100 * (df[cols] < 0.05).groupby(df.level).mean()).round(1).loc[list(DEPENDENCE_LEVELS)].T)
