"""Station model output statistics (MOS) on top of the forecast.

For each species a LightGBM model learns obs - forecast at the input stations from: the forecast of the six species at the
station, the t0 analysis of the six species, the station meteorology at t0 and at the valid time (16 each), local time
features (5), the lead, and station statics (urban, agricultural and water fractions, log population, elevation, column and
row position). Station identifiers are not used. Training year: hindcasts of one year, with February, June and October
held for early stopping. The station corrections are spread to every station and grid cell with a Gaussian-weighted mean
of the input-station corrections (L = 40 km) multiplied by a network-density taper."""
import numpy as np
import pandas as pd
import torch

from .constants import NX, NY, SPECIES
from .model import sample_points, to_cells

PARAMS = {"objective": "l2", "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 500, "feature_fraction": 0.8,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1}
ROUNDS, PATIENCE = 3000, 100
STOP_MONTHS = (2, 6, 10)
SPREAD_KM, DENSITY_N0, SELF_WEIGHT = 40.0, 5.0, 0.5
STATIC = ("static_urban", "static_agri", "static_water", "static_logpop", "static_dem")


@torch.no_grad()
def station_weather(region, batch):
    """Meteorology of the forecast at the stations [48, 16, S]."""
    cells = to_cells(batch["met"][0], region.rv, region.rw)
    return sample_points(cells, region.p4).numpy()


def station_statics(region):
    st = region.stations
    p4 = region.p4.numpy()
    return np.c_[st[list(STATIC)].to_numpy(np.float32), p4[:, 1] / NX, p4[:, 0] / NY].astype(np.float32)


def time_features(valid_utc, utc_offset):
    k = pd.DatetimeIndex(valid_utc) + pd.Timedelta(hours=utc_offset)
    h, d = k.hour.values, k.dayofyear.values
    return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * d / 365.25),
                     np.cos(2 * np.pi * d / 365.25), (k.dayofweek.values >= 5).astype(float)], 1).astype(np.float32)


def rows(P, A0, O, met, t0, statics, stations, utc_offset, leads=range(1, 49)):
    """Feature rows. P, O [case, 48, 6, S]; A0 [case, 6, S]; met [case, 48, 16, S]; t0 DatetimeIndex (UTC).
    Returns X, target (O - P) [n, 6] and keys (case, lead, station)."""
    W = P.shape[0]
    w, s = np.meshgrid(np.arange(W), np.nonzero(stations)[0], indexing="ij")
    w, s = w.ravel(), s.ravel()
    Xs, Ys, K = [], [], []
    for h in leads:
        valid = t0[w] + pd.Timedelta(hours=h)
        X = np.concatenate([P[w, h - 1, :, s], A0[w, :, s], met[w, 0, :, s], met[w, min(h, 47), :, s],
                            time_features(valid, utc_offset), np.full((len(w), 1), h, np.float32), statics[s]], 1)
        Xs.append(X)
        Ys.append(O[w, h - 1, :, s] - P[w, h - 1, :, s])
        K.append(np.c_[w, np.full(len(w), h), s])
    return np.concatenate(Xs).astype(np.float32), np.concatenate(Ys).astype(np.float32), np.concatenate(K)


def train(P, A0, O, met, t0, statics, held, utc_offset, threads=12, log=print):
    """One LightGBM model per species, trained at the input stations."""
    import lightgbm as lgb
    X, Y, K = rows(P, A0, O, met, t0, statics, ~held, utc_offset)
    month = (t0[K[:, 0]] + pd.Timedelta(hours=utc_offset)).month
    stop = np.isin(month, STOP_MONTHS)
    models = []
    for k, sp in enumerate(SPECIES):
        ok = np.isfinite(Y[:, k])
        ds = lgb.Dataset(X[ok & ~stop], Y[ok & ~stop, k])
        dv = lgb.Dataset(X[ok & stop], Y[ok & stop, k], reference=ds)
        bst = lgb.train(PARAMS | {"num_threads": threads}, ds, num_boost_round=ROUNDS, valid_sets=[dv],
                        callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        log(f"mos {sp}: {bst.best_iteration} trees")
        models.append(bst)
    return models


def spread_weights(station_km, held, targets_km=None):
    """[targets, inputs] weights of the input-station corrections: Gaussian (40 km) normalised with a self weight of 0.5
    and tapered where the network is sparse (n^2 / (n^2 + 5^2), n = sum of the Gaussian weights)."""
    targets_km = station_km if targets_km is None else targets_km
    inputs = station_km[~held]
    d = np.linalg.norm(targets_km[:, None].astype(np.float64) - inputs[None].astype(np.float64), axis=-1)
    g = np.exp(-0.5 * (d / SPREAD_KM) ** 2)
    n = g.sum(1, keepdims=True)
    return (n ** 2 / (n ** 2 + DENSITY_N0 ** 2)) * g / (n + SELF_WEIGHT)


def apply(models, P, A0, met, t0, statics, held, station_km, utc_offset, threads=12):
    """MOS forecasts: at each station from its own model ('station', every station) and mapped from the input stations to
    every station ('map', the product used for held-out stations)."""
    S = P.shape[-1]
    station = P.copy()
    O = np.full_like(P, np.nan)
    for h in range(1, 49):
        X, _, K = rows(P, A0, O, met, t0, statics, np.ones(S, bool), utc_offset, leads=[h])
        for k in range(len(SPECIES)):
            station[K[:, 0], h - 1, k, K[:, 2]] = np.maximum(X[:, k] + models[k].predict(X, num_threads=threads), 0)
    corr = station - P
    Wt = spread_weights(station_km, held)
    mapped = np.maximum(P + np.einsum("whki,si->whks", np.nan_to_num(corr[..., ~held]), Wt), 0)
    return station, mapped
