import numpy as np
import pytest

from dacomp import (blocked_folds, da_hac, dependence_ranges, hac_variance, make_blocks,
                    paired_t_folds, random_folds, training_mask)
from dacomp.inference import _kernel_pairs
from dacomp.simulate import grid_coords, panel_index, spatial_cholesky, st_fields


def test_hac_zero_bandwidth_is_iid_variance():
    rng = np.random.default_rng(0)
    u = rng.standard_normal(500)
    u -= u.mean()
    cs = rng.uniform(0, 10, (500, 2))
    v, _ = hac_variance(u, cs, None, 0.0, 0.0)
    assert v == pytest.approx(np.sum(u ** 2) / 500 ** 2)


def test_product_bartlett_kernel_is_psd():
    rng = np.random.default_rng(1)
    cs = rng.uniform(0, 5, (150, 2))
    t = rng.integers(0, 6, 150).astype(float)
    pairs, w = _kernel_pairs(150, cs, t, 2.0, 3.0)
    k = np.eye(150)
    k[pairs[:, 0], pairs[:, 1]] = w
    k[pairs[:, 1], pairs[:, 0]] = w
    assert np.linalg.eigvalsh(k).min() > -1e-10


def test_blocked_folds_keep_blocks_together_and_balance():
    rng = np.random.default_rng(2)
    cs, t = panel_index(grid_coords(12), 24)
    blocks = make_blocks(cs, t, space_tile=4, time_block=6)
    folds = blocked_folds(blocks, 5, rng)
    for b in np.unique(blocks):
        assert len(np.unique(folds[blocks == b])) == 1
    sizes = np.bincount(folds)
    assert sizes.max() - sizes.min() <= 16 * 6
    assert len(np.unique(random_folds(100, 5, rng))) == 5


def test_training_mask_buffer():
    cs, t = panel_index(grid_coords(6), 4)
    test = cs[:, 0] < 2
    m0 = training_mask(test, cs, t)
    m1 = training_mask(test, cs, t, buffer_space=1, buffer_time=10)
    assert np.array_equal(m0, ~test)
    assert not np.any(m1 & (cs[:, 0] == 2)) and np.all(m1[cs[:, 0] >= 3])


def test_variogram_recovers_ranges_roughly():
    rng = np.random.default_rng(3)
    coords = grid_coords(16)
    chol = spatial_cholesky(coords, 2.0)
    f = st_fields(rng, chol, len(coords), 40, 0.6, 1)[0]
    cs, t = panel_index(coords, 40)
    r_s, r_t, _ = dependence_ranges(f, cs, t, max_lag_space=10, max_lag_time=12)
    assert 3 < r_s < 10            # true practical range 6
    assert 3 < r_t < 10            # true practical range -3/log(0.6) = 5.9


def test_da_hac_reduces_to_t_like_test_without_dependence():
    rng = np.random.default_rng(4)
    cs, t = panel_index(grid_coords(10), 10)
    d = rng.standard_normal(len(t))
    folds = blocked_folds(make_blocks(cs, t, 5, 5), 5, rng)
    r = da_hac(d, folds, cs, t, bandwidth=(0.0, 0.0), training_term=False)
    assert r.se == pytest.approx(np.sqrt(np.mean((d - 0) ** 2) / len(d)), rel=0.1)
    assert paired_t_folds(d, folds).df == 4
