"""Random and spatio-temporally blocked K-fold partitions."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def random_folds(n: int, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """Assign ``n`` observations uniformly at random to ``n_folds`` balanced folds."""
    folds = np.arange(n) % n_folds
    rng.shuffle(folds)
    return folds


def make_blocks(coords_space=None, times=None, space_tile=None, time_block=None) -> np.ndarray:
    """Label each observation with a spatio-temporal block.

    Space is cut into square tiles of side ``space_tile`` (one tile per
    coordinate axis) and time into consecutive windows of length
    ``time_block``. Either component may be omitted.
    """
    keys = []
    if coords_space is not None and space_tile is not None:
        cs = np.asarray(coords_space, dtype=float)
        cs = cs.reshape(len(cs), -1)
        keys.append(np.floor((cs - cs.min(axis=0)) / space_tile).astype(int))
    if times is not None and time_block is not None:
        t = np.asarray(times, dtype=float)
        keys.append(np.floor((t - t.min()) / time_block).astype(int)[:, None])
    if not keys:
        raise ValueError("at least one of space or time blocking must be given")
    key = np.hstack(keys)
    _, labels = np.unique(key, axis=0, return_inverse=True)
    return labels.ravel()


def blocked_folds(blocks: np.ndarray, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """Assign whole blocks to folds, balancing the number of observations per fold.

    Blocks are visited in random order and each is given to the fold that
    currently holds the fewest observations.
    """
    labels, sizes = np.unique(blocks, return_counts=True)
    order = rng.permutation(len(labels))
    fold_of_block = np.empty(len(labels), dtype=int)
    load = np.zeros(n_folds)
    for b in order:
        k = int(np.argmin(load + rng.uniform(0, 1e-9, n_folds)))
        fold_of_block[b] = k
        load[k] += sizes[b]
    lookup = dict(zip(labels, fold_of_block))
    return np.array([lookup[b] for b in blocks])


def training_mask(test: np.ndarray, coords_space=None, times=None,
                  buffer_space: float = 0.0, buffer_time: float = 0.0) -> np.ndarray:
    """Training indicator for one fold, excluding a buffer around the test set.

    An observation is dropped from the training set if it lies within
    ``buffer_space`` (per spatial axis) *and* within ``buffer_time`` of some
    test observation. With zero buffers this is simply ``~test``.
    """
    train = ~test
    if buffer_space <= 0 and buffer_time <= 0:
        return train
    cols = []
    if coords_space is not None:
        cs = np.asarray(coords_space, dtype=float).reshape(len(test), -1)
        cols.append(cs / max(buffer_space, 1e-9))
    if times is not None:
        cols.append(np.asarray(times, dtype=float)[:, None] / max(buffer_time, 1e-9))
    z = np.hstack(cols)
    tree = cKDTree(z[test])
    dist, _ = tree.query(z[train], k=1, p=np.inf, distance_upper_bound=1.0 + 1e-9)
    near = np.isfinite(dist)
    idx = np.flatnonzero(train)
    train[idx[near]] = False
    return train
