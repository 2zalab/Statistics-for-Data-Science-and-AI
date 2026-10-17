"""Pilot (separate seeds): candidate treatments of the training-variance boundary.

Run from the repository root: python experiments/pilot/pilot_variants.py
"""
import itertools
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "experiments")
import exp1_controlled as e  # noqa: E402
from dacomp import blocked_folds, hering_genton, make_blocks  # noqa: E402
from dacomp.inference import hac_variance  # noqa: E402
from dacomp.simulate import DEPENDENCE_LEVELS, grid_coords, panel_index, spatial_cholesky, st_fields  # noqa: E402
from dacomp.variogram import dependence_ranges  # noqa: E402

SEED = 555_001
_CH = {}


def pval(est, v, df):
    ref = stats.t(df) if np.isfinite(df) else stats.norm()
    return 2 * ref.sf(abs(est) / np.sqrt(v))


def run(job):
    level, rep = job
    a, phi = DEPENDENCE_LEVELS[level]
    coords = grid_coords(12)
    if level not in _CH:
        _CH[level] = spatial_cholesky(coords, a)
    rng = np.random.default_rng([SEED, list(DEPENDENCE_LEVELS).index(level), rep])
    cs, t = panel_index(coords, 24)
    eps, ea, eb = st_fields(rng, _CH[level], len(coords), 24, phi, 3)
    blocks = make_blocks(cs, t, 4, 6)
    folds = blocked_folds(blocks, 5, rng)
    eta = 0.05 * (np.sqrt(.25) * rng.standard_normal() + np.sqrt(.75) * rng.standard_normal(5))
    d = (eps - ea) ** 2 - (eps - eb) ** 2 + eta[folds]
    n, K = len(d), 5
    fm = np.bincount(folds, weights=d) / np.bincount(folds)
    u = d - fm[folds]
    r_s, r_t, _ = dependence_ranges(u, cs, t, 6, 12)
    zeta = 1 / K + 1 / (K - 1)
    out = {}
    # fold-centred HAC (current)
    vu, vk = hac_variance(u, cs, t, r_s or 0, r_t or 0, groups=folds)
    neff = np.clip(np.mean(u ** 2) / vu, 2, n)
    nuk = np.prod([max(1, 1.5 * 11 / r_s)] * 2 + [max(1, 1.5 * 23 / r_t)]) if (r_s and r_t) else np.inf
    nu_u = min(neff - 1, nuk)
    tau_raw = fm.var(ddof=1) - vk.mean()
    tau = max(0, tau_raw)

    def sat(vu_, vt_):
        den = vu_ ** 2 / nu_u + (vt_ ** 2 / (K - 1) if vt_ > 0 else 0)
        return (vu_ + vt_) ** 2 / den
    v = vu + zeta * tau
    out["A_current"] = pval(d.mean(), v, sat(vu, zeta * tau))
    # E: conservative df when truncated
    out["E_df_trunc"] = pval(d.mean(), v, sat(vu, zeta * tau) if tau_raw > 0 else K - 1)
    # B: globally centred HAC, no fold-centring; training term from globally centred within-fold sums
    g = d - d.mean()
    vg, vkg = hac_variance(g, cs, t, r_s or 0, r_t or 0, groups=folds)
    out["B_global"] = pval(d.mean(), vg, min(np.clip(np.mean(g ** 2) / vg, 2, n) - 1, nuk))
    # F: max of fold-centred DA-HAC and global HAC
    out["F_max"] = pval(d.mean(), max(v, vg), sat(vu, zeta * tau) if v >= vg else min(np.clip(np.mean(g ** 2) / vg, 2, n) - 1, nuk))
    # G: global HAC + training term (truncated)
    out["G_global_tau"] = pval(d.mean(), vg + zeta * tau, sat(vg, zeta * tau))
    out["HG"] = hering_genton(d, cs, t, 0.05, 6, 12).pvalue
    out["trunc"] = tau_raw <= 0
    return dict(level=level, rep=rep, **out)


if __name__ == "__main__":
    jobs = [(l, r) for l in DEPENDENCE_LEVELS for r in range(300)]
    with Pool(4) as p:
        rows = p.map(run, jobs, chunksize=4)
    df = pd.DataFrame(rows)
    df.to_csv("results/pilot_variants.csv", index=False)
    cols = [c for c in df.columns if c not in ("level", "rep", "trunc")]
    print((100 * (df[cols] < 0.05).groupby(df.level).mean()).round(1).loc[list(DEPENDENCE_LEVELS)].T)
