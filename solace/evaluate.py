"""Hindcasts from local standard midnight and their scores.

Scores are pooled RMSE and R2 over cases and stations, for today (leads 1-24) and tomorrow (25-48), at the held-out stations
(primary) or the input stations, on the station-hours valid for every compared forecast. Intervals come from a circular
moving-block bootstrap over calendar days (7-day blocks, 2000 resamples)."""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import forecast, mos
from .constants import HORIZON
from .model import load_weights, sample_points

WINDOWS = {"today": slice(0, 24), "tomorrow": slice(24, 48)}
BOOTSTRAP, BLOCK_DAYS = 2000, 7


def local_midnights(year, utc_offset):
    zone = dt.timezone(dt.timedelta(hours=utc_offset))
    first = dt.datetime(year, 1, 1, tzinfo=zone)
    return [(first + dt.timedelta(days=d)).astimezone(dt.timezone.utc) for d in range(364)]


def hindcast(region, source, observations, obs_start, year, out_dir, model=None, log=print, limit=0):
    """Forecasts for the first 364 local midnights of `year`. Writes <out_dir>/<local date>.npz with forecast [48, 6, S],
    persistence (t0 analysis at the stations), truth [48, 6, S], t0_utc and the station meteorology
    [48, 16, S] used by the MOS. Cases without 75 % input coverage are skipped."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model or load_weights(region.path / "weights.pt")
    skipped = []
    days = local_midnights(year, region.utc_offset)
    for t0 in days[:limit] if limit else days:
        day = (t0 + dt.timedelta(hours=region.utc_offset)).date().isoformat()
        dest = out_dir / f"{day}.npz"
        if dest.exists():
            continue
        try:
            batch = forecast.build_inputs(region, source, observations, obs_start, t0)
        except ValueError as e:
            skipped.append((day, str(e)))
            continue
        pred = forecast.run(region, batch, model)
        persistence = sample_points(batch["c"], region.p4)[0].numpy()
        h0 = forecast.hours_between(obs_start, t0)
        truth = np.asarray(observations[h0 + 1:h0 + 1 + HORIZON], np.float32)
        np.savez_compressed(dest, forecast=pred, persistence=persistence, truth=truth, t0_utc=t0.isoformat(),
                            station_met=mos.station_weather(region, batch).astype(np.float16))
        log(f"hindcast {day}")
    return skipped


def corrected_ctm(f, analysis_t0, tau=12.0):
    """Observation-corrected chemical transport model forecast max(0, f(h) + (c_a(t0) - f(t0)) exp(-h / tau)); f [49, ...]
    from lead 0."""
    h = np.arange(f.shape[0], dtype=np.float32).reshape((-1,) + (1,) * (f.ndim - 1))
    return np.maximum(0.0, f + (analysis_t0 - f[0]) * np.exp(-h / tau))


def load_cases(folder, names=("forecast", "persistence", "truth")):
    files = sorted(Path(folder).glob("????-??-??.npz"))
    out = {k: [] for k in names}
    for f in files:
        with np.load(f) as z:
            for k in names:
                v = z[k]
                out[k].append(np.broadcast_to(v, z["truth"].shape) if k == "persistence" else v)
    days = pd.to_datetime([f.stem for f in files])
    return days, {k: np.stack(v).astype(np.float32) for k, v in out.items()}


def block_weights(days, rng, n=BOOTSTRAP, block=BLOCK_DAYS):
    """Circular moving-block bootstrap on calendar days: [n, cases] counts."""
    doy = np.asarray((days - pd.Timestamp(days[0].year, 1, 1)).days)
    length = 366
    W = np.zeros((n, len(days)))
    k = int(np.ceil(length / block))
    for b in range(n):
        for s in rng.integers(0, length, k):
            W[b] += np.isin(doy, (s + np.arange(block)) % length)
    return W


def scores(models, truth, stations, days, seed=0):
    """models: {name: [case, 48, 6, S]}; truth [case, 48, 6, S]; stations bool [S]. Returns a table of RMSE and R2 with
    95 % intervals for each window and species, on the station-hours valid for every model."""
    rng = np.random.default_rng(seed)
    W = block_weights(days, rng)
    rows = []
    for wname, sl in WINDOWS.items():
        for k in range(truth.shape[2]):
            o = truth[:, sl, k][..., stations]
            ok = np.isfinite(o)
            for v in models.values():
                ok &= np.isfinite(v[:, sl, k][..., stations])
            n_case = ok.sum((1, 2))
            for name, v in models.items():
                p = v[:, sl, k][..., stations]
                se = np.where(ok, (p - o) ** 2, 0).sum((1, 2))
                rm = np.sqrt(se.sum() / n_case.sum())
                bs = np.sqrt((W @ se) / (W @ n_case))
                om = o[ok].mean()
                r2 = 1 - se.sum() / ((o[ok] - om) ** 2).sum()
                rows.append({"window": wname, "species": k, "model": name, "rmse": rm, "rmse_lo": np.percentile(bs, 2.5),
                             "rmse_hi": np.percentile(bs, 97.5), "r2": r2, "n": int(n_case.sum())})
    return pd.DataFrame(rows)


def rmse_ratio(a, b, truth, stations, days, window, species, seed=0):
    """RMSE(a) / RMSE(b) - 1 with its 95 % block-bootstrap interval."""
    rng = np.random.default_rng(seed)
    W = block_weights(days, rng)
    sl = WINDOWS[window]
    o = truth[:, sl, species][..., stations]
    pa, pb = a[:, sl, species][..., stations], b[:, sl, species][..., stations]
    ok = np.isfinite(o) & np.isfinite(pa) & np.isfinite(pb)
    sa = np.where(ok, (pa - o) ** 2, 0).sum((1, 2))
    sb = np.where(ok, (pb - o) ** 2, 0).sum((1, 2))
    point = np.sqrt(sa.sum() / sb.sum()) - 1
    bs = np.sqrt((W @ sa) / (W @ sb)) - 1
    return point, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


@torch.no_grad()
def variants(region, batch, states, switches=None):
    """Station forecasts of several weight sets on one batch: {name: [48, 6, S]} (see solace.exchange)."""
    from .model import Solace
    model = Solace().eval()
    out = {}
    for name, state in states.items():
        model.load_state_dict(state)
        model.switches_off = frozenset((switches or {}).get(name, ()))
        p, _ = model(batch, region)
        out[name] = p[0].numpy()
    return out
