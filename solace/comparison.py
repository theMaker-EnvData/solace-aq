"""Comparisons with chemical transport model products at the stations.

CTM forecasts at the stations: bilinear interpolation of hourly fields (GEOS-CF, CAMS) or the nearest native cell (AQM),
hourly means of consecutive instantaneous values (GEOS-CF); observation-corrected forms max{0, f(h) + (c_a(t0) - f(t0)) exp(-h / 12)} with
c_a the t0 analysis at the station. Skill tables: pooled RMSE, R2 and bias on the station-hours valid for every compared
model (a model without the species is left out for that species), and RMSE(model) / RMSE(reference) - 1 with the 7-day
circular block bootstrap interval."""
import json

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

from .constants import SPECIES
from .evaluate import WINDOWS, block_weights

TAU_H = 12.0


def bilinear(lat, lon, field, slat, slon):
    """field [..., lat, lon] at station coordinates -> [..., S]."""
    la, lo = np.asarray(lat, float), np.asarray(lon, float)
    if la[0] > la[-1]:
        la, field = la[::-1], field[..., ::-1, :]
    fi, fj = np.interp(slat, la, np.arange(len(la))), np.interp(slon, lo, np.arange(len(lo)))
    i0 = np.clip(np.floor(fi).astype(int), 0, len(la) - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, len(lo) - 2)
    wi, wj = fi - i0, fj - j0
    return ((1 - wi) * (1 - wj) * field[..., i0, j0] + wi * (1 - wj) * field[..., i0 + 1, j0] + (1 - wi) * wj * field[..., i0, j0 + 1]
            + wi * wj * field[..., i0 + 1, j0 + 1])


def interval_means(point):
    """Instantaneous values at hourly steps [n + 1, ...] -> hourly means [n, ...] (row 0 is the hour ending at t0)."""
    point = np.asarray(point, np.float32)
    return 0.5 * (point[:-1] + point[1:])


def corrected(f, analysis_t0, tau=TAU_H):
    """f [49, 6, S] from lead 0; analysis_t0 [6, S] -> raw and observation-corrected forecasts for leads 1..48."""
    d0 = analysis_t0 - f[0]
    decay = np.exp(-np.arange(1, f.shape[0], dtype=np.float32) / tau)[:, None, None]
    return f[1:], np.maximum(0.0, f[1:] + np.where(np.isfinite(d0), d0, 0.0)[None] * decay)


def nearest_cells(x_grid, y_grid, crs_wkt, longitude, latitude):
    """Indices (i, j) of the native grid cell nearest to each station."""
    x, y = Transformer.from_crs(4326, CRS.from_wkt(crs_wkt), always_xy=True).transform(longitude, latitude)
    j = np.rint((x - x_grid[0]) / (x_grid[1] - x_grid[0])).astype(int)
    i = np.rint((y - y_grid[0]) / (y_grid[1] - y_grid[0])).astype(int)
    if (i < 0).any() or (j < 0).any() or (i >= len(y_grid)).any() or (j >= len(x_grid)).any():
        raise ValueError("station outside the product grid")
    return i, j


def fire_free(days, utc_offset, fires_json, crs, bounds, min_acres=10000):
    """Cases whose 48-h window does not overlap a large wildfire with its origin in the domain (NIFC WFIGS incidents):
    'no_fire_7d' excludes discovery to discovery + 7 days, 'no_fire_containment' discovery to containment + 1 day."""
    z = json.loads(fires_json.read_text())
    d = pd.DataFrame([{**f["attributes"], "lon": f["geometry"]["x"], "lat": f["geometry"]["y"]} for f in z["features"]])
    x, y = Transformer.from_crs(4326, CRS.from_user_input(crs), always_xy=True).transform(d.lon.values, d.lat.values)
    a, b, c, e = bounds
    d = d[(x >= a) & (x <= c) & (y >= b) & (y <= e) & (d.IncidentSize >= min_acres) & (d.IncidentTypeCategory == "WF")]
    start = pd.to_datetime(d.FireDiscoveryDateTime, unit="ms", utc=True).dt.normalize()
    contained = pd.to_datetime(d.ContainmentDateTime, unit="ms", utc=True)
    contained = contained.fillna(start + pd.Timedelta(days=14)).dt.normalize() + pd.Timedelta(days=2)
    t0 = days.tz_localize("UTC") - pd.Timedelta(hours=utc_offset)
    masks = {}
    for name, end in (("no_fire_7d", start + pd.Timedelta(days=8)), ("no_fire_containment", contained)):
        hit = np.zeros(len(days), bool)
        for s, en in zip(start, end):
            hit |= np.asarray((t0 + pd.Timedelta(hours=48) > s) & (t0 < en))
        masks[name] = ~hit
    return masks, d.IncidentName.tolist()


def skill(D, truth, stations, days, reference, W=None, rng=None, species=SPECIES, cases=None):
    """D: {model: [case, 48, 6, S]}; truth [case, 48, 6, S]; stations bool [S]; cases bool [case] (subset) or None;
    reference: a model name or a sequence of names, the first available for the species is used.
    Returns rows for each window, species and model."""
    W = block_weights(days, rng) if W is None else W
    cases = np.ones(len(days), bool) if cases is None else cases
    Wp = W[:, cases]
    rows = []
    for wn, sl in WINDOWS.items():
        for k, sp in enumerate(species):
            ms = [m for m in D if np.isfinite(D[m][:, :, k]).any()]
            obs = truth[cases][:, sl, k][..., stations]
            ok = np.isfinite(obs)
            for m in ms:
                ok &= np.isfinite(D[m][cases][:, sl, k][..., stations])
            n = ok.sum((1, 2)).astype("f8")
            sse = {m: (np.where(ok, D[m][cases][:, sl, k][..., stations] - obs, 0.0).astype("f8") ** 2).sum((1, 2)) for m in ms}
            var = np.nanvar(np.where(ok, obs, np.nan))
            ref = next(r for r in ((reference,) if isinstance(reference, str) else reference) if r in ms)
            rb = {m: np.sqrt(Wp @ sse[m] / (Wp @ n)) for m in ms}
            for m in ms:
                r = np.sqrt(sse[m].sum() / n.sum())
                d = 100 * (rb[m] / rb[ref] - 1)
                bias = float(np.where(ok, D[m][cases][:, sl, k][..., stations] - obs, 0).sum() / n.sum())
                rows.append(dict(window=wn, species=sp, model=m, cases=int(cases.sum()), stations=int(np.any(ok, (0, 1)).sum()),
                                 n=int(n.sum()), rmse=r, r2=1 - r ** 2 / var, bias=bias, ref=ref,
                                 vs_ref_pct=100 * (r / np.sqrt(sse[ref].sum() / n.sum()) - 1),
                                 vs_ref_lo=np.percentile(d, 2.5), vs_ref_hi=np.percentile(d, 97.5)))
    return rows
