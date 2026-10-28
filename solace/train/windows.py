"""Training windows of one year: t0 analyses, observations along the forecast and every model input from the caches."""
import numpy as np
import pandas as pd
import torch

from ..constants import LAYERS12, OUTER_INDEX, SPINUP, SURFACE6
from ..inputs import time_features


class Year:
    """One UTC year of training data.
    obs [T, 6, S]; idx training times (hour of year); analyses: 6 arrays [len(idx), NY, NX]; caches: dict from
    caches.open_year; background: (times, 6 x means) of the 30-day mean analyses on the hour clock starting at `epoch`."""

    def __init__(self, year, obs, idx, analyses, caches, utc_offset, level_ratio, level6, background, epoch_year):
        self.year, self.obs, self.idx, self.anas, self.c = year, obs, np.asarray(idx), analyses, caches
        times = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h", tz="UTC")
        self.tf = time_features(times, utc_offset)
        self.level_ratio, self.level6 = level_ratio, level6
        self.bg_times, self.bg_means = background
        self.offset = int((np.datetime64(f"{year}-01-01") - np.datetime64(f"{epoch_year}-01-01")).astype(int)) * 24

    def ok_windows(self, horizon, spin=SPINUP):
        """Windows whose spin-up start and forecast end stay inside the year."""
        return np.where((self.idx >= spin) & (self.idx + horizon + 1 <= len(self.obs)))[0]

    def _scale(self, t):
        ts = pd.Timestamp(f"{self.year}-01-01", tz="UTC") + pd.to_timedelta(np.asarray(t).ravel(), "h")
        return ts.month.values - 1, ts.hour.values // 6

    def pm25_batch(self, ns, horizon):
        """Surface PM2.5 pre-training: t0 analysis, meteorology, observations and time features."""
        t0 = self.idx[ns]
        c = torch.as_tensor(self.anas[0][ns].astype(np.float32))[:, None]
        met = torch.as_tensor(np.stack([np.asarray(self.c["met"][t:t + horizon], np.float32) for t in t0]))
        obs = torch.as_tensor(np.stack([self.obs[t + 1:t + 1 + horizon, 0] for t in t0]))
        tf = torch.as_tensor(np.stack([self.tf[t:t + horizon] for t in t0]))
        return c, met, obs, tf

    def batch(self, ns, horizon, spin=SPINUP):
        """Full model batch (see Solace.forward) with the observations [B, H, 6, S] and the outer fields of every hour (the
        auxiliary target of the outer model; the forecast reads chemistry only at t0 - spin)."""
        t0 = self.idx[ns]
        tq = t0 - spin
        n = spin + horizon + 1
        c = torch.as_tensor(np.stack([a[ns].astype(np.float32) for a in self.anas], 1))
        met = torch.as_tensor(np.stack([np.asarray(self.c["met"][t:t + horizon], np.float32) for t in t0]))
        ext = torch.as_tensor(np.stack([np.asarray(self.c["extra"][t:t + horizon], np.float32) for t in t0]))
        obs = torch.as_tensor(np.stack([self.obs[t + 1:t + 1 + horizon] for t in t0]))
        tf = torch.as_tensor(np.stack([self.tf[t - spin:t + horizon] for t in t0]))
        frames = tq[:, None] + np.arange(n)[None]
        m, hb = self._scale(frames)
        o = np.zeros((len(t0), n, 34) + self.c["outer"].shape[-2:], np.float32)
        o[:, :, :18] = np.stack([np.asarray(self.c["outer"][t:t + n], np.float32) for t in tq])
        o[:, :, list(SURFACE6[2:])] = np.stack([np.asarray(self.c["gas"][t:t + n], np.float32) for t in tq])
        o[:, :, OUTER_INDEX["pm25"]] *= self.level_ratio[m, hb].reshape(frames.shape)[:, :, None, None]
        r = np.moveaxis(self.level6[:, m, hb].reshape(6, *frames.shape), 0, -1)[..., 1:]
        o[:, :, list(SURFACE6[1:])] *= r[..., None, None]
        layers = np.nan_to_num(np.stack([np.asarray(self.c["layers"][t:t + n], np.float32) for t in tq]))  # [B, n, 6, 2, ..]
        o[:, :, list(LAYERS12)] = layers.transpose(0, 1, 3, 2, 4, 5).reshape(len(t0), n, 12, *o.shape[-2:])
        gus = torch.as_tensor(np.stack([np.asarray(self.c["layers"][t], np.float32) for t in tq]))
        pos = np.searchsorted(self.bg_times, self.offset + t0, side="left")
        bg = torch.as_tensor(np.stack([np.asarray(mm[pos], np.float32) for mm in self.bg_means], 1))
        last_m = torch.where(torch.isfinite(bg), bg, c)
        return {"c": c, "met": met, "ext": ext, "o": torch.as_tensor(o), "gus": gus, "tf": tf, "last_m": last_m}, obs
