"""Constants derived from the training years: loss scales, outer-input statistics, the surface / lowest-level ratios of
the outer fields and the boundary-gate thresholds."""
import numpy as np
import pandas as pd
import torch

from ..constants import HALO, NX, NY, OUTER_AUX, OUTER_INDEX, OUTER_MET
from ..model import to_halo

RING = 4  # cells of the inner edge ring used to compare analysis and outer field


def lead_scale(years_obs, keep, hmax, step=7):
    """Per-lead error scale E_h: station persistence RMSE |y(t + h) - y(t)| over the training years at the input stations.
    years_obs: list of [T, S] arrays of one species."""
    se, n = np.zeros(hmax), np.zeros(hmax)
    for Y in years_obs:
        Y = Y[:, keep]
        t0 = np.arange(0, len(Y) - hmax - 1, step)
        for h in range(1, hmax + 1):
            d = Y[t0 + h] - Y[t0]
            m = np.isfinite(d)
            se[h - 1] += (d[m] ** 2).sum()
            n[h - 1] += m.sum()
    return np.sqrt(se / n)


def outer_stats(outer_years):
    """Mean and spread of the outer meteorology and auxiliary fields over every 97th training hour."""
    a = np.concatenate([np.asarray(o[::97], np.float32) for o in outer_years])
    v = a[:, [OUTER_INDEX[k] for k in OUTER_MET + OUTER_AUX]]
    return torch.tensor(np.nanmean(v, (0, 2, 3))), torch.tensor(np.nanstd(v, (0, 2, 3)) + 1e-6)


def _inner(fields, grid):
    return to_halo(torch.as_tensor(fields), grid)[:, :, HALO:HALO + NY, HALO:HALO + NX]


def ring_mask():
    m = torch.zeros(NY, NX, dtype=torch.bool)
    m[:RING] = m[-RING:] = True
    m[:, :RING] = m[:, -RING:] = True
    return m


def ring_q(c, ref, sr):
    """Per-segment |log((analysis + 2) / (outer + 2))|, c and ref [B, 1, NY, NX] -> [B, 12]."""
    return torch.stack([torch.log((c[:, 0][:, sr == k].mean(1) + 2) / (ref[:, 0][:, sr == k].mean(1) + 2)).abs()
                        for k in range(12)], 1)


def window_times(year, idx):
    return pd.Timestamp(f"{year}-01-01", tz="UTC") + pd.to_timedelta(idx, "h")


@torch.no_grad()
def pm25_level_ratio(years, grid, chunk=64):
    """R[12, 4]: analysis / outer PM2.5 summed over the inner ring at the training windows, by month and 6-h UTC bin.
    years: list of (year, idx, pm25 analyses [n, NY, NX], outer cache)."""
    ring = ring_mask()
    sa, sc = np.zeros((12, 4)), np.zeros((12, 4))
    for year, idx, ana, outer in years:
        tt = window_times(year, idx)
        for b in range(0, len(idx), chunk):
            sl = slice(b, b + chunk)
            a = torch.as_tensor(np.asarray(ana[sl], np.float32))
            o = np.stack([np.asarray(outer[t][OUTER_INDEX["pm25"]], np.float32) for t in idx[sl]])[:, None]
            ch = _inner(o, grid)[:, 0]
            for k, t in enumerate(tt[sl]):
                m, hb = t.month - 1, t.hour // 6
                sa[m, hb] += float(a[k][ring].mean())
                sc[m, hb] += float(ch[k][ring].mean())
    return np.clip(sa / np.maximum(sc, 1e-6), 0.2, 2.0).astype(np.float32)


@torch.no_grad()
def pm25_gate(years, grid, R, sr, chunk=64):
    """95th and 99th percentiles of the per-segment t0 mismatch between the analysis and the level-corrected outer PM2.5."""
    Q = []
    for year, idx, ana, outer in years:
        tt = window_times(year, idx)
        for b in range(0, len(idx), chunk):
            sl = slice(b, b + chunk)
            a = torch.as_tensor(np.asarray(ana[sl], np.float32))[:, None]
            r = torch.as_tensor(R[tt[sl].month - 1, tt[sl].hour // 6])[:, None, None, None]
            o = np.stack([np.asarray(outer[t][OUTER_INDEX["pm25"]], np.float32) for t in idx[sl]])[:, None]
            Q.append(ring_q(a, _inner(o, grid) * r, sr).numpy())
    Q = np.concatenate(Q)
    return float(np.percentile(Q, 95)), float(np.percentile(Q, 99))


@torch.no_grad()
def species_level_gate(years, grid, sr, pm25_gate_q, chunk=64):
    """Level ratios [6, 12, 4] (clip 0.2-5; PM2.5 row 1, its ratio is applied separately) and gate thresholds of every
    species. years: list of (year, idx, analyses [6 x [n, NY, NX]], outer cache, gas cache)."""
    ring = sr >= 0
    sa, sc = np.zeros((6, 12, 4)), np.zeros((6, 12, 4))
    SA, SC, MM, HB = [], [], [], []
    for year, idx, anas, outer, gas in years:
        tt = window_times(year, idx)
        for b in range(0, len(idx), chunk):
            sl = np.arange(b, min(b + chunk, len(idx)))
            a = torch.as_tensor(np.stack([x[sl].astype(np.float32) for x in anas], 1))
            fld = np.stack([np.concatenate([np.asarray(outer[t], np.float32)[[0, 1]], np.asarray(gas[t], np.float32)])
                            for t in idx[sl]])
            ch = _inner(fld, grid)
            SA.append(torch.stack([a[:, :, sr == k].mean(2) for k in range(12)], 2).numpy())
            SC.append(torch.stack([ch[:, :, sr == k].mean(2) for k in range(12)], 2).numpy())
            MM.append(tt[sl].month.values - 1)
            HB.append(tt[sl].hour.values // 6)
            ra, rc = a[:, :, ring].mean(2).numpy(), ch[:, :, ring].mean(2).numpy()
            for k, (m, hb) in enumerate(zip(MM[-1], HB[-1])):
                sa[:, m, hb] += ra[k]
                sc[:, m, hb] += rc[k]
    SA, SC, MM, HB = np.concatenate(SA), np.concatenate(SC), np.concatenate(MM), np.concatenate(HB)
    R = np.clip(sa / np.maximum(sc, 1e-6), 0.2, 5.0)
    R[0] = 1.0
    q95, q99 = [pm25_gate_q[0]], [pm25_gate_q[1]]
    for k in range(1, 6):
        r = R[k, MM, HB][:, None]
        Q = np.abs(np.log((SA[:, k] + 2) / (SC[:, k] * r + 2)))
        q95.append(float(np.percentile(Q, 95)))
        q99.append(float(np.percentile(Q, 99)))
    return R.astype(np.float32), np.asarray(q95, np.float32), np.asarray(q99, np.float32)
