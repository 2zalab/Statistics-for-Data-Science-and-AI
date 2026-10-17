"""Experiment 3: case studies on public spatio-temporal data sets.

* ``nyc_weather``  hourly weather at three New York airports (2013); binary
  classification of precipitation, log loss.
* ``pedestrian``   hourly pedestrian counts at four Melbourne sensors
  (2015-2016); regression of log(1 + count), squared loss.
* ``lasrosas``     yield-monitor maize data from Las Rosas, Argentina (1999,
  2001); spatial regression of yield, squared loss.

For each data set three pairwise comparisons are made (gradient boosting,
random forest, linear baseline) under random and blocked CV and, for the two
hourly data sets, under forward-chaining (rolling-origin) CV,
followed by a "seed null" diagnostic: gradient boosting compared with itself
under different random seeds, a comparison whose expected loss differential
is zero by construction.

Usage: python experiments/exp3_real_data.py [--seed-pairs 30]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from dacomp import (blocked_folds, corrected_resampled_t, iid_bootstrap, make_blocks,
                    paired_t_folds, random_folds, training_mask)
from dacomp.benchmark import blocked_cv_tests
from dacomp.variogram import dependence_ranges

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
K, R_REPEATS = 5, 5


# ---------------------------------------------------------------------------
# Data sets
# ---------------------------------------------------------------------------

def load_nyc_weather():
    w = pd.read_csv(RAW / "nyc_weather.csv", parse_dates=["time_hour"])
    w = w.dropna(subset=["temp", "dewp", "humid", "wind_speed"]).sort_values(["time_hour", "origin"])
    t = ((w.time_hour - w.time_hour.min()).dt.total_seconds() / 3600).to_numpy()
    x = pd.get_dummies(w[["temp", "dewp", "humid", "wind_dir", "wind_speed", "pressure",
                          "month", "hour", "origin"]], columns=["origin"], dtype=float)
    return dict(name="nyc_weather", task="classification", x=x.to_numpy(float),
                y=(w.precip > 0).astype(int).to_numpy(), times=t, coords=None,
                sites=w.origin.to_numpy(), time_block=24 * 14, space_tile=None,
                buffer_time=24.0, buffer_space=0.0, max_lag_time=24 * 7, max_lag_space=None)


def load_pedestrian():
    p = pd.read_csv(RAW / "pedestrian.csv", parse_dates=["Date_Time", "Date"])
    p = p.sort_values(["Date_Time", "Sensor"])
    t = ((p.Date_Time - p.Date_Time.min()).dt.total_seconds() / 3600).to_numpy()
    feats = pd.DataFrame({"hour": p.Time, "dow": p.Date.dt.dayofweek, "month": p.Date.dt.month,
                          "doy": p.Date.dt.dayofyear, "year": p.Date.dt.year, "sensor": p.Sensor})
    x = pd.get_dummies(feats, columns=["sensor"], dtype=float)
    return dict(name="pedestrian", task="regression", x=x.to_numpy(float),
                y=np.log1p(p.Count.to_numpy(float)), times=t, coords=None,
                sites=p.Sensor.to_numpy(), time_block=24 * 14, space_tile=None,
                buffer_time=24.0, buffer_space=0.0, max_lag_time=24 * 7, max_lag_space=None)


def load_lasrosas():
    c = pd.read_csv(RAW / "lasrosas.corn.csv")
    lat0, lon0 = c.lat.mean(), c.long.mean()
    xm = (c.long - lon0) * 111_320 * np.cos(np.radians(lat0))
    ym = (c.lat - lat0) * 110_540
    coords = np.column_stack([xm, ym])
    feats = pd.DataFrame({"nitro": c.nitro, "bv": c.bv, "year": c.year, "x": xm, "y": ym, "topo": c.topo})
    x = pd.get_dummies(feats, columns=["topo"], dtype=float)
    return dict(name="lasrosas", task="regression", x=x.to_numpy(float), y=c["yield"].to_numpy(float),
                times=None, coords=coords, sites=None, time_block=None, space_tile=75.0,
                buffer_time=0.0, buffer_space=10.0, max_lag_time=None, max_lag_space=200.0)


# ---------------------------------------------------------------------------
# Learners and losses
# ---------------------------------------------------------------------------

def make_learner(kind, task, seed):
    if kind == "xgb":
        cls = XGBClassifier if task == "classification" else XGBRegressor
        return cls(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                   colsample_bytree=0.8, n_jobs=4, tree_method="hist", random_state=seed)
    if kind == "rf":
        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        return cls(n_estimators=200, min_samples_leaf=5, max_features=0.5, n_jobs=4, random_state=seed)
    if kind == "linear":
        est = LogisticRegression(max_iter=2000) if task == "classification" else Ridge(alpha=1.0)
        return make_pipeline(SimpleImputer(), StandardScaler(), est)
    raise ValueError(kind)


def oof_loss(ds, kind, folds, seed, masks=None):
    """Out-of-fold losses; folds labelled -1 are never tested (forward chaining)."""
    x, y, task = ds["x"], ds["y"], ds["task"]
    loss = np.full(len(y), np.nan)
    for k in np.unique(folds[folds >= 0]):
        test = folds == k
        train = ~test if masks is None else masks[k]
        model = make_learner(kind, task, seed + int(k)).fit(x[train], y[train])
        if task == "classification":
            p = np.clip(model.predict_proba(x[test])[:, 1], 1e-6, 1 - 1e-6)
            loss[test] = -(y[test] * np.log(p) + (1 - y[test]) * np.log(1 - p))
        else:
            loss[test] = (y[test] - model.predict(x[test])) ** 2
    return loss


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def partitions(ds, rng):
    n = len(ds["y"])
    rand = [random_folds(n, K, rng) for _ in range(R_REPEATS)]
    blocks = make_blocks(ds["coords"], ds["times"], ds["space_tile"], ds["time_block"])
    folds = blocked_folds(blocks, K, rng)
    masks = {k: training_mask(folds == k, ds["coords"], ds["times"], ds["buffer_space"], ds["buffer_time"])
             for k in range(K)}
    return rand, blocks, folds, masks


def forward_partition(ds):
    """Forward chaining: K consecutive groups of time blocks; group k >= 1 is
    predicted from all observations before it (minus the buffer), group 0 is
    used for training only."""
    t = ds["times"]
    blocks = make_blocks(None, t, None, ds["time_block"])
    folds = (blocks * K) // (blocks.max() + 1)
    masks = {k: t < t[folds == k].min() - ds["buffer_time"] for k in range(1, K)}
    folds = np.where(folds == 0, -1, folds)
    return blocks, folds, masks


def compare_forward(ds, kind_a, kind_b, fparts, rng, seed_a=1, seed_b=2):
    blocks, folds, masks = fparts
    test = folds >= 0
    d = (oof_loss(ds, kind_a, folds, seed_a + 11, masks) - oof_loss(ds, kind_b, folds, seed_b + 11, masks))[test]
    res = blocked_cv_tests(d, folds[test], blocks[test], None, ds["times"][test], ds["sites"][test], rng,
                           ablations=False, max_lag_time=ds["max_lag_time"])
    return [dict(scheme="forward", **r_.as_dict()) for r_ in res]


def compare(ds, kind_a, kind_b, parts, rng, seed_a=1, seed_b=2, repeats=R_REPEATS, keep_d=False):
    rand, blocks, folds, masks = parts
    n = len(ds["y"])
    rows, split_means = [], []
    for r in range(repeats):
        d = oof_loss(ds, kind_a, rand[r], seed_a + 100 * r) - oof_loss(ds, kind_b, rand[r], seed_b + 100 * r)
        split_means += [d[rand[r] == k].mean() for k in range(K)]
        if r == 0:
            res = [paired_t_folds(d, rand[0], method="paired_t"),
                   iid_bootstrap(d, rng, method="iid_bootstrap")]
    if repeats > 1:
        res.append(corrected_resampled_t(split_means, n * (K - 1) / K, n / K, method="nadeau_bengio"))
    rows += [dict(scheme="random", **r_.as_dict()) for r_ in res]

    d = oof_loss(ds, kind_a, folds, seed_a + 7, masks) - oof_loss(ds, kind_b, folds, seed_b + 7, masks)
    res = blocked_cv_tests(d, folds, blocks, ds["coords"], ds["times"], ds["sites"], rng, ablations=False,
                           max_lag_space=ds["max_lag_space"], max_lag_time=ds["max_lag_time"])
    rows += [dict(scheme="blocked", **r_.as_dict()) for r_ in res]
    return (rows, d) if keep_d else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-pairs", type=int, default=30)
    ap.add_argument("--datasets", nargs="+", default=["nyc_weather", "pedestrian", "lasrosas"])
    args = ap.parse_args()
    loaders = {"nyc_weather": load_nyc_weather, "pedestrian": load_pedestrian, "lasrosas": load_lasrosas}
    out = ROOT / "results"
    for name in args.datasets:
        ds = loaders[name]()
        rng = np.random.default_rng([2026, len(name)])
        parts = partitions(ds, rng)
        t0 = time.time()
        rows, varios = [], []
        fparts = forward_partition(ds) if ds["coords"] is None else None
        for a, b in [("xgb", "rf"), ("xgb", "linear"), ("rf", "linear")]:
            res, d = compare(ds, a, b, parts, rng, keep_d=True)
            if fparts is not None:
                res += compare_forward(ds, a, b, fparts, rng)
            rows += [dict(dataset=name, comparison=f"{a}_vs_{b}", **r) for r in res]
            _, f = np.unique(parts[2], return_inverse=True)
            u = d - (np.bincount(f, weights=d) / np.bincount(f))[f]
            _, _, fits = dependence_ranges(u, ds["coords"], ds["times"], ds["max_lag_space"], ds["max_lag_time"])
            for direction, ((lags, gamma, counts), fit) in fits.items():
                varios += [dict(dataset=name, comparison=f"{a}_vs_{b}", direction=direction, lag=h, gamma=g,
                                count=c, **{f"fit_{k}": v for k, v in fit.items()})
                           for h, g, c in zip(lags, gamma, counts)]
            print(f"{name} {a} vs {b}: {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out / f"exp3_{name}.csv", index=False)
        pd.DataFrame(varios).to_csv(out / f"exp3_{name}_variogram.csv", index=False)

        null_rows = []
        for s in range(args.seed_pairs):
            res = compare(ds, "xgb", "xgb", parts, rng, seed_a=1000 + 2 * s, seed_b=1001 + 2 * s, repeats=1)
            null_rows += [dict(dataset=name, pair=s, **r) for r in res]
        pd.DataFrame(null_rows).to_csv(out / f"exp3_{name}_seednull.csv", index=False)
        print(f"{name} seed null: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
