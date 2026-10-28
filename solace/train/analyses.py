"""Gridded OI analyses for training (every 7 h at the training windows, every 6 h for the emission backgrounds) and the
strictly past 30-day mean analyses."""
import numpy as np
import torch

from .. import oi
from ..constants import NX, NY
from .observations import COVERAGE

WINDOW_STEP = 7  # hours between training windows
HORIZON_MARGIN = 24


def training_times(cov_frac):
    """Training window times (hour of year): every 7 h where every species has at least 75 % coverage."""
    idx = np.arange(0, len(cov_frac) - HORIZON_MARGIN, WINDOW_STEP)
    return idx[(cov_frac[idx] >= COVERAGE).all(1)]


@torch.no_grad()
def fields(values, cov_frac, stations, network, cov, chunk=32):
    """Analyses [T, NY, NX] (float16) of one species at the given hours; NaN where the species coverage is below 75 %.
    values [T, S]; cov_frac [T]; stations [S, 2] km; network bool [S]."""
    pts = torch.as_tensor(stations[network], dtype=torch.float32)
    queries = oi.grid_points()
    K, Kq, r = cov.k(pts, pts), cov.k(queries, pts), torch.exp(cov.logr)
    out = np.empty((len(values), NY, NX), np.float16)
    for b in range(0, len(values), chunk):
        z, vm, mean, sd = oi.standardise(torch.as_tensor(np.array(values[b:b + chunk][:, network], copy=True)))
        A = K[None] * (vm[:, :, None] * vm[:, None, :]) + torch.diag_embed(r * vm + 1 - vm + 1e-5)
        chol, info = torch.linalg.cholesky_ex(A)
        if bool((info > 0).any()):
            chol = torch.linalg.cholesky(A + 1e-3 * torch.eye(A.shape[-1]))
        alpha = torch.cholesky_solve(z[..., None], chol)[..., 0]
        res = (mean + sd * (Kq @ (alpha * vm).T).T).clamp(min=0).numpy().reshape(-1, NY, NX).astype(np.float16)
        res[cov_frac[b:b + chunk] < COVERAGE] = np.nan
        out[b:b + len(res)] = res
    return out


def rolling_means(times, analyses, window=oi.WINDOW_HOURS):
    """Entry j = mean of the valid analyses with times in [q_j - 720 h, q_j), q = times followed by one step after the
    last; frames that are entirely NaN are skipped. analyses [T, NY, NX] float16; returns float16 [T + 1, NY, NX]."""
    times = np.asarray(times, dtype=np.int64)
    query = np.r_[times, times[-1] + 6]
    out = np.empty((len(query),) + analyses.shape[1:], np.float16)
    acc = np.zeros(analyses.shape[1:], np.float64)
    count, left = 0, 0
    for j, t in enumerate(query):
        if j:
            f = np.asarray(analyses[j - 1], np.float32)
            if np.isfinite(f).all():
                acc += f
                count += 1
            elif np.isfinite(f).any():
                raise ValueError("partially invalid analysis")
        while left < j and times[left] < t - window:
            f = np.asarray(analyses[left], np.float32)
            if np.isfinite(f).all():
                acc -= f
                count -= 1
            left += 1
        out[j] = acc / count if count else np.nan
    return out


def background_at(times, means, t0):
    """30-day mean analysis for a forecast starting at hour t0 (same clock as times)."""
    return np.asarray(means[int(np.searchsorted(times, t0, side="left"))], np.float32)
