"""Inference procedures for comparing two predictive models.

Every procedure takes the per-observation out-of-fold loss differentials
``d_i = loss_A(i) - loss_B(i)`` (negative values favour model A) and
returns a :class:`TestResult` for the null hypothesis that the expected loss
differential is zero.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

from .variogram import dependence_ranges


@dataclass
class TestResult:
    method: str
    estimate: float
    se: float
    df: float
    pvalue: float
    ci_low: float
    ci_high: float
    info: dict = field(default_factory=dict)

    def as_dict(self):
        out = {k: getattr(self, k) for k in ("method", "estimate", "se", "df", "pvalue", "ci_low", "ci_high")}
        out.update({f"info_{k}": v for k, v in self.info.items() if np.isscalar(v)})
        return out


def _t_result(method, est, se, df, alpha, info=None):
    """Wald-type result with a Student-t (finite ``df``) or normal reference."""
    ref = stats.t(df) if np.isfinite(df) else stats.norm()
    q = ref.ppf(1 - alpha / 2)
    if se > 0 and np.isfinite(se):
        p = 2 * ref.sf(abs(est) / se)
    else:
        p = 1.0 if est == 0 else 0.0
    return TestResult(method, float(est), float(se), float(df), float(p),
                      float(est - q * se), float(est + q * se), info or {})


def fold_means(d, folds):
    labels, inv = np.unique(folds, return_inverse=True)
    return np.bincount(inv, weights=d) / np.bincount(inv)


# ---------------------------------------------------------------------------
# Conventional procedures
# ---------------------------------------------------------------------------

def paired_t_folds(d, folds, alpha=0.05, method="paired_t"):
    """Paired t-test on the K per-fold mean loss differences (df = K - 1)."""
    m = fold_means(d, folds)
    k = len(m)
    return _t_result(method, m.mean(), m.std(ddof=1) / np.sqrt(k), k - 1, alpha)


def corrected_resampled_t(split_means, n_train, n_test, alpha=0.05, method="nadeau_bengio"):
    """Corrected resampled t-test of Nadeau and Bengio (2003).

    ``split_means`` holds the mean loss differential of each of the J
    train/test splits (e.g. all folds of R repeated K-fold runs).
    """
    x = np.asarray(split_means, float)
    j = len(x)
    var = (1.0 / j + n_test / n_train) * x.var(ddof=1)
    return _t_result(method, x.mean(), np.sqrt(var), j - 1, alpha)


def _bootstrap_result(method, est, boot, alpha):
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    p = (1 + np.sum(np.abs(boot - est) >= abs(est))) / (len(boot) + 1)
    return TestResult(method, float(est), float(boot.std(ddof=1)), np.inf, float(p), float(lo), float(hi))


def iid_bootstrap(d, rng, n_boot=999, alpha=0.05, method="iid_bootstrap"):
    """Nonparametric bootstrap that resamples observations independently."""
    d = np.asarray(d, float)
    n = len(d)
    boot = np.empty(n_boot)
    chunk = max(1, int(2e7 // n))
    for s in range(0, n_boot, chunk):
        e = min(n_boot, s + chunk)
        idx = rng.integers(0, n, size=(e - s, n))
        boot[s:e] = d[idx].mean(axis=1)
    return _bootstrap_result(method, d.mean(), boot, alpha)


def block_bootstrap(d, blocks, rng, n_boot=999, alpha=0.05, method="block_bootstrap"):
    """Block (cluster) bootstrap resampling whole spatio-temporal blocks."""
    labels, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    sizes = np.bincount(inv).astype(float)
    g = len(labels)
    idx = rng.integers(0, g, size=(n_boot, g))
    boot = sums[idx].sum(1) / sizes[idx].sum(1)
    return _bootstrap_result(method, float(np.mean(d)), boot, alpha)


def mixed_model(d, groups, alpha=0.05, method="mixed_model"):
    """Random-intercept linear mixed model ``d_i = Delta + b_g(i) + e_i`` (REML).

    The standard error of the fixed intercept is referred to a t distribution
    with G - 1 degrees of freedom, G being the number of groups.
    """
    import statsmodels.api as sm

    d = np.asarray(d, float)
    _, g = np.unique(groups, return_inverse=True)
    n_groups = g.max() + 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = sm.MixedLM(d, np.ones((len(d), 1)), groups=g).fit(reml=True)
            est, se = float(fit.fe_params[0]), float(fit.bse_fe[0])
            vc = float(fit.cov_re.iloc[0, 0])
        except Exception:  # fall back to the cluster-robust (CR1) variance
            sums = np.bincount(g, weights=d)
            sizes = np.bincount(g)
            est = d.mean()
            r = sums - sizes * est
            se = np.sqrt(n_groups / (n_groups - 1) * np.sum(r ** 2)) / len(d)
            vc = np.nan
    if not np.isfinite(se):
        se = np.nan
    return _t_result(method, est, se, n_groups - 1, alpha, {"var_block": vc})


# ---------------------------------------------------------------------------
# Kernel (HAC) variance estimation
# ---------------------------------------------------------------------------

def _kernel_pairs(n, coords_space, times, b_space, b_time):
    """Pairs i < j with positive product-Bartlett weight, and their weights.

    The kernel is ``prod_axis (1 - |delta_axis| / b_axis)_+`` over the spatial
    axes (bandwidth ``b_space``) and time (bandwidth ``b_time``). A product of
    one-dimensional Bartlett kernels is positive semi-definite, so the
    resulting variance estimate is never negative. A zero bandwidth keeps only
    pairs with identical coordinates on that axis.
    """
    cols, bws = [], []
    if coords_space is not None:
        cs = np.asarray(coords_space, float).reshape(n, -1)
        for j in range(cs.shape[1]):
            cols.append(cs[:, j])
            bws.append(b_space or 0.0)
    if times is not None:
        cols.append(np.asarray(times, float))
        bws.append(b_time or 0.0)
    if not cols:
        return np.empty((0, 2), int), np.empty(0)
    z = np.column_stack(cols) / np.where(np.array(bws) > 0, bws, 1e-9)
    pairs = cKDTree(z).query_pairs(r=1.0, p=np.inf, output_type="ndarray")
    if len(pairs) == 0:
        return pairs, np.empty(0)
    w = np.prod(np.clip(1.0 - np.abs(z[pairs[:, 0]] - z[pairs[:, 1]]), 0.0, None), axis=1)
    keep = w > 0
    return pairs[keep], w[keep]


def hac_variance(u, coords_space=None, times=None, b_space=0.0, b_time=0.0, groups=None):
    """Kernel estimate of ``Var(mean(u))`` for a mean-zero dependent field ``u``.

    Returns the overall variance and, if ``groups`` is given, the variance of
    each group mean computed from within-group pairs only.
    """
    u = np.asarray(u, float)
    n = len(u)
    pairs, w = _kernel_pairs(n, coords_space, times, b_space, b_time)
    cross = w * u[pairs[:, 0]] * u[pairs[:, 1]] if len(pairs) else np.empty(0)
    total = (np.sum(u ** 2) + 2 * cross.sum()) / n ** 2
    if groups is None:
        return total, None
    _, g = np.unique(groups, return_inverse=True)
    ng = np.bincount(g)
    diag = np.bincount(g, weights=u ** 2)
    if len(pairs):
        same = g[pairs[:, 0]] == g[pairs[:, 1]]
        off = np.bincount(g[pairs[same, 0]], weights=cross[same], minlength=len(ng))
    else:
        off = np.zeros(len(ng))
    return total, (diag + 2 * off) / ng ** 2


def parametric_variance(u, coords_space=None, times=None, fits=None, groups=None, cutoff=1e-3):
    """Variance of mean(u) under a fitted separable exponential correlation.

    ``fits`` holds exponential variogram fits per direction (as returned by
    :func:`~dacomp.variogram.dependence_ranges`); the correlation between two
    observations is ``g_S(h) g_T(l)`` with ``g(x) = c1/(c0+c1) exp(-x/a)`` for
    ``x > 0`` and 1 at ``x = 0``. Correlations below ``cutoff`` are neglected.
    Returns the overall variance and, if ``groups`` is given, the variance of
    each group mean from within-group pairs only.
    """
    u = np.asarray(u, float)
    n = len(u)
    s2 = np.mean(u ** 2)
    fits = fits or {}

    def params(direction):
        if direction not in fits:
            return 0.0, 0.0
        f = fits[direction][1]
        sill = f["c0"] + f["c1"]
        return (f["c1"] / sill if sill > 0 else 0.0), f["a"]

    w_s, a_s = params("space")
    w_t, a_t = params("time")
    cut = np.log(1.0 / cutoff)
    cols, scales = [], []
    if coords_space is not None:
        cs = np.asarray(coords_space, float).reshape(n, -1)
        cols.append(cs)
        scales += [cut * a_s if (w_s > 0 and a_s > 0) else 1e-9] * cs.shape[1]
    if times is not None:
        t = np.asarray(times, float)
        cols.append(t[:, None])
        scales.append(cut * a_t if (w_t > 0 and a_t > 0) else 1e-9)
    _, gr = np.unique(groups, return_inverse=True) if groups is not None else (None, np.zeros(n, int))
    ng = np.bincount(gr)
    gsum, off = 0.0, np.zeros(len(ng))
    if cols:
        z = np.hstack(cols) / np.array(scales)
        tree = cKDTree(z)
        probe = tree.query_ball_point(z[:: max(1, n // 50)], r=1.0, p=np.inf, return_length=True)
        chunk = int(max(20, min(5000, 5e6 / max(probe.mean(), 1.0))))
        for start in range(0, n, chunk):  # chunked to bound memory on large data sets
            idx = np.arange(start, min(n, start + chunk))
            nb = tree.query_ball_point(z[idx], r=1.0, p=np.inf)
            lens = np.fromiter((len(x) for x in nb), int, len(nb))
            if lens.sum() == 0:
                continue
            i = np.repeat(idx, lens)
            j = np.concatenate([np.asarray(x, int) for x in nb])
            keep = j > i
            i, j = i[keep], j[keep]
            g = np.ones(len(i))
            if coords_space is not None:
                h = np.linalg.norm(cs[i] - cs[j], axis=1)
                g *= np.where(h > 0, w_s * np.exp(-h / max(a_s, 1e-12)), 1.0)
            if times is not None:
                lag = np.abs(t[i] - t[j])
                g *= np.where(lag > 0, w_t * np.exp(-lag / max(a_t, 1e-12)), 1.0)
            gsum += g.sum()
            same = gr[i] == gr[j]
            off += np.bincount(gr[i[same]], weights=g[same], minlength=len(ng))
    total = s2 * (n + 2 * gsum) / n ** 2
    if groups is None:
        return total, None
    return total, s2 * (ng + 2 * off) / ng ** 2


def dm_hac(d, sites, times, alpha=0.05, lag=None, method="dm_hac"):
    """Diebold-Mariano-type test with a Newey-West (Bartlett) variance in time.

    Temporal autocorrelation within each site is accounted for; correlation
    between sites is ignored, as when the DM test is applied to panel data.
    """
    d = np.asarray(d, float)
    t = np.asarray(times, float)
    n_times = len(np.unique(t))
    if lag is None:
        lag = int(np.floor(4 * (n_times / 100) ** (2 / 9)))
    _, site_idx = np.unique(sites, return_inverse=True)
    u = d - d.mean()
    v, _ = hac_variance(u, site_idx.astype(float), t, 0.0, lag + 1.0)
    return _t_result(method, d.mean(), np.sqrt(v), np.inf, alpha, {"lag": lag})


def da_test(d, folds, coords_space=None, times=None, alpha=0.05, estimator="parametric",
            max_lag_space=None, max_lag_time=None, fits=None, method="da_p", **kwargs):
    """Dependence-aware test (proposed); ``estimator`` selects the test-sampling variance.

    ``'parametric'`` (DA-P, default) uses the separable exponential correlation
    fitted to the directional variograms of the fold-centred differentials
    (:func:`parametric_variance`); ``'kernel'`` (DA-K) uses the product-Bartlett
    kernel with variogram-calibrated bandwidths (:func:`da_hac`). All other
    steps (training term, reference distribution, diagnostics) are shared.
    """
    return da_hac(d, folds, coords_space, times, alpha, max_lag_space=max_lag_space,
                  max_lag_time=max_lag_time, method=method, estimator=estimator, fits=fits, **kwargs)


def da_hac(d, folds, coords_space=None, times=None, alpha=0.05, kappa=1.0,
           bandwidth=None, training_term=True, reference="satterthwaite",
           max_lag_space=None, max_lag_time=None, ranges=None, method="da_hac",
           variants=None, rho=None, estimator="kernel", fits=None):
    """Dependence-aware test with variogram-calibrated test-sampling variance.

    With ``estimator='kernel'`` (DA-K) the test-sampling variance is a
    product-Bartlett kernel (HAC) estimate; with ``estimator='parametric'``
    (DA-P, see :func:`da_test`) it is computed from the fitted variogram model.

    Steps
    -----
    1. Fold-centre the loss differentials: ``u_i = d_i - dbar_{k(i)}``.
    2. Estimate practical ranges of dependence of ``u`` in space and time from
       directional empirical variograms (exponential model) and set the
       product-Bartlett bandwidths to ``kappa`` times these ranges, unless
       ``bandwidth=(b_space, b_time)`` is given (``ranges=(r_space, r_time)``
       supplies precomputed ranges).
    3. ``V_hac``: kernel estimate of the test-sampling variance of ``dbar``.
    4. ``tau2``: method-of-moments estimate of the variance of the fold-level
       (training-induced) effects, i.e. the excess of the between-fold variance
       of fold means over their kernel-estimated within-fold variance.
    5. ``V = V_hac + zeta tau2`` with
       ``zeta = (1/K + rho (1 - 1/K)) / (1 - rho)``, where ``rho`` is the
       assumed correlation between fold effects. The default ``rho = 1/K``
       gives ``zeta = 1/K + 1/(K-1)``, the Nadeau-Bengio inflation.
    6. Reference distribution: Student t with Welch-Satterthwaite degrees of
       freedom combining the HAC component, with
       ``nu_hac = min(n_eff - 1, prod_axis 3 L / (2 b))`` where
       ``n_eff = s_u^2 / V_hac`` and L is the extent of the domain along each
       axis, and the K - 1 df of ``tau2``. ``reference='satterthwaite_neff'``
       uses ``n_eff - 1`` only and ``reference='normal'`` a normal reference.

    ``info['range_ratio']`` (largest bandwidth relative to the extent of the
    domain along its axis) flags designs in which the dependence range is
    comparable to the domain; values above about 0.25 indicate that no
    procedure can be well calibrated from a single sample.

    ``variants``, a list of ``(name, training_term, reference)`` or
    ``(name, training_term, reference, rho)``, returns one result per variant
    while computing the kernel sums only once.
    """
    d = np.asarray(d, float)
    n = len(d)
    _, f = np.unique(folds, return_inverse=True)
    k = f.max() + 1
    fm = np.bincount(f, weights=d) / np.bincount(f)
    u = d - fm[f]

    if bandwidth is None:
        if ranges is None or (estimator == "parametric" and fits is None):
            r_s, r_t, fits = dependence_ranges(u, coords_space, times, max_lag_space, max_lag_time)
        else:
            r_s, r_t = ranges
        b_s = kappa * r_s if r_s is not None else None
        b_t = kappa * r_t if r_t is not None else None
    else:
        b_s, b_t = bandwidth
        r_s, r_t = b_s, b_t

    if estimator == "parametric":
        v_hac, v_fold = parametric_variance(u, coords_space, times, fits, groups=f)
    else:
        v_hac, v_fold = hac_variance(u, coords_space, times, b_s or 0.0, b_t or 0.0, groups=f)
    s2u = np.mean(u ** 2)
    n_eff = float(np.clip(s2u / max(v_hac, 1e-300), 2.0, n))
    tau2_raw = fm.var(ddof=1) - v_fold.mean() if k > 1 else 0.0
    tau2 = max(0.0, tau2_raw)

    # Degrees of freedom of a product-kernel variance estimate (Priestley):
    # Var(V)/V^2 ~ 2 prod_axis (b * int k^2) / L, with int k^2 = 2/3 for Bartlett,
    # so nu_kernel = prod_axis 3 L / (2 b) over axes with positive bandwidth.
    nu_kernel = np.inf
    axes = []
    if coords_space is not None and b_s:
        cs = np.asarray(coords_space, float).reshape(n, -1)
        axes += [(np.ptp(cs[:, j]), b_s) for j in range(cs.shape[1])]
    if times is not None and b_t:
        axes.append((np.ptp(np.asarray(times, float)), b_t))
    range_ratio = 0.0
    if axes:
        nu_kernel = float(np.prod([max(1.0, 1.5 * length / bw) for length, bw in axes]))
        range_ratio = float(max(bw / length for length, bw in axes if length > 0))
    info = {"range_space": r_s if r_s is not None else np.nan,
            "range_time": r_t if r_t is not None else np.nan,
            "b_space": b_s if b_s is not None else np.nan,
            "b_time": b_t if b_t is not None else np.nan,
            "v_hac": v_hac, "tau2": tau2, "tau2_raw": tau2_raw, "n_eff": n_eff,
            "nu_kernel": nu_kernel, "range_ratio": range_ratio}

    def finish(name, use_tau, ref, rho_=rho):
        r = 1.0 / k if rho_ is None else rho_
        zeta = (1.0 / k + r * (1 - 1.0 / k)) / (1 - r)
        v_train = zeta * tau2 if use_tau else 0.0
        v = v_hac + v_train
        if ref == "normal":
            df = np.inf
        else:
            nu_hac = n_eff - 1 if ref == "satterthwaite_neff" else min(n_eff - 1, nu_kernel)
            denom = v_hac ** 2 / nu_hac + (v_train ** 2 / (k - 1) if v_train > 0 else 0.0)
            df = v ** 2 / denom if denom > 0 else np.inf
        return _t_result(name, d.mean(), np.sqrt(v), df, alpha, dict(info))

    if variants is None:
        return finish(method, training_term, reference)
    return [finish(*v) for v in variants]


def three_way_decision(result, alpha=0.05, max_range_ratio=0.25):
    """Decision rule attached to DA-HAC.

    Returns ``'indeterminate'`` when the estimated dependence range is too large
    relative to the study domain (``info['range_ratio'] > max_range_ratio``),
    in which case the sample contains too few effectively independent regions
    for a calibrated comparison; otherwise ``'reject'`` or ``'not rejected'``.
    """
    if result.info.get("range_ratio", 0.0) > max_range_ratio:
        return "indeterminate"
    return "reject" if result.pvalue < alpha else "not rejected"


def hering_genton(d, coords_space=None, times=None, alpha=0.05, max_lag_space=None,
                  max_lag_time=None, method="hering_genton"):
    """Spatial prediction comparison test of Hering and Genton (2011), space-time version.

    The covariance of the loss differential is modelled parametrically: an
    exponential model with nugget is fitted to the empirical spatial and
    temporal semivariograms of the centred differentials and combined into a
    separable correlation ``g_S(h) g_T(l)``. The variance of the mean is
    ``n^-2 sum_ij s^2 g_S(h_ij) g_T(l_ij)`` and the statistic is referred to a
    normal distribution. Training-induced fold effects are not modelled.
    """
    d = np.asarray(d, float)
    n = len(d)
    u = d - d.mean()
    s2 = np.mean(u ** 2)
    _, _, fits = dependence_ranges(u, coords_space, times, max_lag_space, max_lag_time)

    def corr_params(direction):
        if direction not in fits:
            return 0.0, 0.0
        f = fits[direction][1]
        sill = f["c0"] + f["c1"]
        return (f["c1"] / sill if sill > 0 else 0.0), f["a"]

    w_s, a_s = corr_params("space")
    w_t, a_t = corr_params("time")
    v, _ = parametric_variance(u, coords_space, times, fits)
    return _t_result(method, d.mean(), np.sqrt(v), np.inf, alpha, {"lag": lag})


def da_test(d, folds, coords_space=None, times=None, alpha=0.05, estimator="parametric",
            max_lag_space=None, max_lag_time=None, fits=None, method="da_p", **kwargs):
    """Dependence-aware test (proposed); ``estimator`` selects the test-sampling variance.

    ``'parametric'`` (DA-P, default) uses the separable exponential correlation
    fitted to the directional variograms of the fold-centred differentials
    (:func:`parametric_variance`); ``'kernel'`` (DA-K) uses the product-Bartlett
    kernel with variogram-calibrated bandwidths (:func:`da_hac`). All other
    steps (training term, reference distribution, diagnostics) are shared.
    """
    return da_hac(d, folds, coords_space, times, alpha, max_lag_space=max_lag_space,
                  max_lag_time=max_lag_time, method=method, estimator=estimator, fits=fits, **kwargs)


def da_hac(d, folds, coords_space=None, times=None, alpha=0.05, kappa=1.0,
           bandwidth=None, training_term=True, reference="satterthwaite",
           max_lag_space=None, max_lag_time=None, ranges=None, method="da_hac",
           variants=None, rho=None, estimator="kernel", fits=None):
    """Dependence-aware test with variogram-calibrated test-sampling variance.

    With ``estimator='kernel'`` (DA-K) the test-sampling variance is a
    product-Bartlett kernel (HAC) estimate; with ``estimator='parametric'``
    (DA-P, see :func:`da_test`) it is computed from the fitted variogram model.

    Steps
    -----
    1. Fold-centre the loss differentials: ``u_i = d_i - dbar_{k(i)}``.
    2. Estimate practical ranges of dependence of ``u`` in space and time from
       directional empirical variograms (exponential model) and set the
       product-Bartlett bandwidths to ``kappa`` times these ranges, unless
       ``bandwidth=(b_space, b_time)`` is given (``ranges=(r_space, r_time)``
       supplies precomputed ranges).
    3. ``V_hac``: kernel estimate of the test-sampling variance of ``dbar``.
    4. ``tau2``: method-of-moments estimate of the variance of the fold-level
       (training-induced) effects, i.e. the excess of the between-fold variance
       of fold means over their kernel-estimated within-fold variance.
    5. ``V = V_hac + zeta tau2`` with
       ``zeta = (1/K + rho (1 - 1/K)) / (1 - rho)``, where ``rho`` is the
       assumed correlation between fold effects. The default ``rho = 1/K``
       gives ``zeta = 1/K + 1/(K-1)``, the Nadeau-Bengio inflation.
    6. Reference distribution: Student t with Welch-Satterthwaite degrees of
       freedom combining the HAC component, with
       ``nu_hac = min(n_eff - 1, prod_axis 3 L / (2 b))`` where
       ``n_eff = s_u^2 / V_hac`` and L is the extent of the domain along each
       axis, and the K - 1 df of ``tau2``. ``reference='satterthwaite_neff'``
       uses ``n_eff - 1`` only and ``reference='normal'`` a normal reference.

    ``info['range_ratio']`` (largest bandwidth relative to the extent of the
    domain along its axis) flags designs in which the dependence range is
    comparable to the domain; values above about 0.25 indicate that no
    procedure can be well calibrated from a single sample.

    ``variants``, a list of ``(name, training_term, reference)`` or
    ``(name, training_term, reference, rho)``, returns one result per variant
    while computing the kernel sums only once.
    """
    d = np.asarray(d, float)
    n = len(d)
    _, f = np.unique(folds, return_inverse=True)
    k = f.max() + 1
    fm = np.bincount(f, weights=d) / np.bincount(f)
    u = d - fm[f]

    if bandwidth is None:
        if ranges is None or (estimator == "parametric" and fits is None):
            r_s, r_t, fits = dependence_ranges(u, coords_space, times, max_lag_space, max_lag_time)
        else:
            r_s, r_t = ranges
        b_s = kappa * r_s if r_s is not None else None
        b_t = kappa * r_t if r_t is not None else None
    else:
        b_s, b_t = bandwidth
        r_s, r_t = b_s, b_t

    if estimator == "parametric":
        v_hac, v_fold = parametric_variance(u, coords_space, times, fits, groups=f)
    else:
        v_hac, v_fold = hac_variance(u, coords_space, times, b_s or 0.0, b_t or 0.0, groups=f)
    s2u = np.mean(u ** 2)
    n_eff = float(np.clip(s2u / max(v_hac, 1e-300), 2.0, n))
    tau2_raw = fm.var(ddof=1) - v_fold.mean() if k > 1 else 0.0
    tau2 = max(0.0, tau2_raw)

    # Degrees of freedom of a product-kernel variance estimate (Priestley):
    # Var(V)/V^2 ~ 2 prod_axis (b * int k^2) / L, with int k^2 = 2/3 for Bartlett,
    # so nu_kernel = prod_axis 3 L / (2 b) over axes with positive bandwidth.
    nu_kernel = np.inf
    axes = []
    if coords_space is not None and b_s:
        cs = np.asarray(coords_space, float).reshape(n, -1)
        axes += [(np.ptp(cs[:, j]), b_s) for j in range(cs.shape[1])]
    if times is not None and b_t:
        axes.append((np.ptp(np.asarray(times, float)), b_t))
    range_ratio = 0.0
    if axes:
        nu_kernel = float(np.prod([max(1.0, 1.5 * length / bw) for length, bw in axes]))
        range_ratio = float(max(bw / length for length, bw in axes if length > 0))
    info = {"range_space": r_s if r_s is not None else np.nan,
            "range_time": r_t if r_t is not None else np.nan,
            "b_space": b_s if b_s is not None else np.nan,
            "b_time": b_t if b_t is not None else np.nan,
            "v_hac": v_hac, "tau2": tau2, "tau2_raw": tau2_raw, "n_eff": n_eff,
            "nu_kernel": nu_kernel, "range_ratio": range_ratio}

    def finish(name, use_tau, ref, rho_=rho):
        r = 1.0 / k if rho_ is None else rho_
        zeta = (1.0 / k + r * (1 - 1.0 / k)) / (1 - r)
        v_train = zeta * tau2 if use_tau else 0.0
        v = v_hac + v_train
        if ref == "normal":
            df = np.inf
        else:
            nu_hac = n_eff - 1 if ref == "satterthwaite_neff" else min(n_eff - 1, nu_kernel)
            denom = v_hac ** 2 / nu_hac + (v_train ** 2 / (k - 1) if v_train > 0 else 0.0)
            df = v ** 2 / denom if denom > 0 else np.inf
        return _t_result(name, d.mean(), np.sqrt(v), df, alpha, dict(info))

    if variants is None:
        return finish(method, training_term, reference)
    return [finish(*v) for v in variants]


def three_way_decision(result, alpha=0.05, max_range_ratio=0.25):
    """Decision rule attached to DA-HAC.

    Returns ``'indeterminate'`` when the estimated dependence range is too large
    relative to the study domain (``info['range_ratio'] > max_range_ratio``),
    in which case the sample contains too few effectively independent regions
    for a calibrated comparison; otherwise ``'reject'`` or ``'not rejected'``.
    """
    if result.info.get("range_ratio", 0.0) > max_range_ratio:
        return "indeterminate"
    return "reject" if result.pvalue < alpha else "not rejected"


def hering_genton(d, coords_space=None, times=None, alpha=0.05, max_lag_space=None,
                  max_lag_time=None, method="hering_genton"):
    """Spatial prediction comparison test of Hering and Genton (2011), space-time version.

    The covariance of the loss differential is modelled parametrically: an
    exponential model with nugget is fitted to the empirical spatial and
    temporal semivariograms of the centred differentials and combined into a
    separable correlation ``g_S(h) g_T(l)``. The variance of the mean is
    ``n^-2 sum_ij s^2 g_S(h_ij) g_T(l_ij)`` and the statistic is referred to a
    normal distribution. Training-induced fold effects are not modelled.
    """
    d = np.asarray(d, float)
    n = len(d)
    u = d - d.mean()
    s2 = np.mean(u ** 2)
    _, _, fits = dependence_ranges(u, coords_space, times, max_lag_space, max_lag_time)

    def corr_params(direction):
        if direction not in fits:
            return 0.0, 0.0
        f = fits[direction][1]
        sill = f["c0"] + f["c1"]
        return (f["c1"] / sill if sill > 0 else 0.0), f["a"]

    w_s, a_s = corr_params("space")
    w_t, a_t = corr_params("time")
    cut = np.log(1e3)  # neglect correlations below 1e-3
    cols, scales = [], []
    if coords_space is not None:
        cs = np.asarray(coords_space, float).reshape(n, -1)
        cols.append(cs)
        scales += [cut * a_s if (w_s > 0 and a_s > 0) else 1e-9] * cs.shape[1]
    if times is not None:
        t = np.asarray(times, float)
        cols.append(t[:, None])
        scales.append(cut * a_t if (w_t > 0 and a_t > 0) else 1e-9)
    total = n * s2
    if cols:
        z = np.hstack(cols) / np.array(scales)
        pairs = cKDTree(z).query_pairs(r=1.0, p=np.inf, output_type="ndarray")
        if len(pairs):
            g = np.ones(len(pairs))
            if coords_space is not None:
                h = np.linalg.norm(cs[pairs[:, 0]] - cs[pairs[:, 1]], axis=1)
                g *= np.where(h > 0, w_s * np.exp(-h / max(a_s, 1e-12)), 1.0)
            if times is not None:
                lag = np.abs(t[pairs[:, 0]] - t[pairs[:, 1]])
                g *= np.where(lag > 0, w_t * np.exp(-lag / max(a_t, 1e-12)), 1.0)
            total += 2 * s2 * g.sum()
    v = max(total, 0.0) / n ** 2
    return _t_result(method, d.mean(), np.sqrt(v), np.inf, alpha,
                     {"range_space": 3 * a_s if w_s > 0 else 0.0, "range_time": 3 * a_t if w_t > 0 else 0.0})
