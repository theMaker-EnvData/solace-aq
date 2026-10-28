"""Station observations for training: spike quality control, the held-out split, the input network and its coverage."""
import numpy as np
from scipy.spatial import cKDTree

SPIKE_THRESHOLD = {0: 300.0, 1: 600.0}  # PM2.5, PM10 (ug m-3); gases rely on the providers' validity flags
COVERAGE = 0.75
ACTIVE_HOURS = 720


def grid_km(stations):
    """Station coordinates on the 1-km grid (row, column), as used for the split blocks and neighbour search."""
    return np.c_[stations.y_km.values - 0.5, stations.x_km.values - 0.5]


def qc_spikes(Y, stations, threshold):
    """Remove isolated spikes from one species [T, S]: a value above `threshold` that exceeds three times both the median
    of the station's own values within +-3 h and the median of its five nearest stations at that hour."""
    Y = Y.copy()
    ti, si = np.where(np.nan_to_num(Y) > threshold)
    if len(ti) == 0:
        return Y, 0
    neighbours = cKDTree(grid_km(stations).astype(np.float32)).query(grid_km(stations).astype(np.float32), k=6)[1][:, 1:]
    bad = []
    for t, j in zip(ti, si):
        own = np.r_[Y[max(0, t - 3):t, j], Y[t + 1:t + 4, j]]
        mo = np.nanmedian(own) if np.isfinite(own).any() else np.nan
        near = Y[t, neighbours[j]]
        mn = np.nanmedian(near) if np.isfinite(near).any() else np.nan
        if np.isfinite(mo) and np.isfinite(mn) and Y[t, j] > 3 * mo and Y[t, j] > 3 * mn:
            bad.append((t, j))
    for t, j in bad:
        Y[t, j] = np.nan
    return Y, len(bad)


def quality_control(values, stations):
    """values [T, 6, S] of one UTC year -> copy with PM spikes removed."""
    out = np.array(values, copy=True)
    for k, threshold in SPIKE_THRESHOLD.items():
        out[:, k], _ = qc_spikes(out[:, k], stations, threshold)
    return out


def held_split(stations, seed=11, block_km=30, fraction=0.12):
    """Hold out whole 30-km blocks of stations, in random block order, until 12 % of the stations are held."""
    gy, gx = grid_km(stations).T
    block = (gy // block_km) * 1000 + gx // block_km
    blocks = np.unique(block)
    np.random.default_rng(seed).shuffle(blocks)
    held, n = set(), 0
    for b in blocks:
        if n >= fraction * len(stations):
            break
        held.add(b)
        n += int((block == b).sum())
    return np.isin(block, list(held))


def eligibility(years):
    """[6, S]: the station reported the species at least once in the training years (arrays [T, 6, S])."""
    return np.isfinite(np.concatenate(years)).any(0)


def mask_ineligible(values, eligible, held):
    """Observations of input stations that never reported a species in the training years are not used."""
    out = np.array(values, copy=True)
    for k in range(out.shape[1]):
        out[:, k, ~eligible[k] & ~held] = np.nan
    return out


def active_network(years, window=ACTIVE_HOURS):
    """Per-year bool arrays [T, 6, S]: the station reported the species at least once in [t - 720 h, t)."""
    full = np.concatenate([np.isfinite(np.asarray(v)) for v in years])
    cs = np.concatenate([np.zeros((1,) + full.shape[1:], np.int32), np.cumsum(full, 0, dtype=np.int32)])
    t = np.arange(len(full))
    seen = (cs[t] - cs[np.maximum(t - window, 0)]) > 0
    out, pos = [], 0
    for v in years:
        out.append(seen[pos:pos + len(v)])
        pos += len(v)
    return out


def coverage(values, eligible, held, active=None):
    """Fraction [T, 6] of the input network reporting; the network is the eligible, not-held stations (and, when given,
    those active in the preceding 720 h)."""
    network = np.asarray(eligible, bool) & ~np.asarray(held, bool)[None]
    if active is None:
        return (np.isfinite(values) & network).sum(-1) / network.sum(-1)
    network = network & active
    return (np.isfinite(values) & network).sum(-1) / np.maximum(network.sum(-1), 1)
