"""One forecast: initial fields from the station observations up to t0, inputs from meteorology and GEOS-CF, 48-h rollout."""
import datetime as dt

import numpy as np
import torch

from . import inputs, oi
from .constants import HORIZON, NX, NY, SPECIES, SPINUP
from .model import load_weights


def hours_between(a, b):
    return int((b - a).total_seconds() // 3600)


def initial_fields(region, observations, obs_start, t0):
    """t0 analysis and 30-day mean analysis [6, NY, NX] from hourly observations [T, 6, S] (NaN missing) whose first row
    is obs_start (UTC). Rows must cover t0 - 720 h to t0. Held-out stations are never used."""
    h0 = hours_between(obs_start, t0)
    if h0 < oi.WINDOW_HOURS:
        raise ValueError("observations must start at least 720 h before t0")
    first = h0 - oi.WINDOW_HOURS
    six = np.array([h for h in range(first, h0) if (obs_start + dt.timedelta(hours=h)).hour % 6 == 0])
    hours = np.r_[six, h0]
    c = np.empty((len(SPECIES), NY, NX), np.float32)
    background = np.empty_like(c)
    for k, sp in enumerate(SPECIES):
        network = region.eligible[k] & ~region.held
        fields, valid = oi.analysis_fields(np.asarray(observations[hours, k], np.float32), region.station_km, network,
                                           region.covariance(sp))
        if not valid[-1]:
            raise ValueError(f"fewer than 75 % of the {sp} input stations report at t0")
        c[k] = fields[-1].astype(np.float32)
        b = oi.background(fields[:-1], six, h0).astype(np.float32)
        background[k] = np.where(np.isfinite(b), b, c[k])
    return c, background


def build_inputs(region, source, observations, obs_start, t0, horizon=HORIZON, spin=SPINUP):
    """source: an object with era5(start, count) and geoscf(start, count) as inputs.Archive."""
    c, background = initial_fields(region, observations, obs_start, t0)
    start = t0 - dt.timedelta(hours=spin)
    nodes, extra, outer_met = inputs.weather(*source.era5(start, spin + horizon + 1))
    chem, gas, layers = inputs.chemistry(*source.geoscf(start, 1), region.dust_ratio)
    return inputs.assemble(c, background, nodes, extra, outer_met, chem[0], gas[0], layers[0], t0, region, horizon, spin)


@torch.no_grad()
def run(region, batch, model=None, fields=False, horizon=HORIZON):
    """Station forecasts [H, 6, S] (and the surface fields [H, 6, NY, NX] with fields=True)."""
    model = model or load_weights(region.path / "weights.pt")
    out = [] if fields else None
    pred, _ = model(batch, region, horizon, fields=out)
    pred = pred[0].numpy()
    return (pred, torch.cat(out).numpy()) if fields else pred
