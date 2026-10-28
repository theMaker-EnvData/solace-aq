"""Operator experiments over a year of forecasts (module exchange, sensitivity, seed control, ...), one file per case:

operators         variant names
held_pred         [variant, 48, 6, held stations] float16
held_truth        [48, 6, held stations] float16
held_persistence  [6, held stations] float16 (t0 analysis)
all_sse, all_err  [variant, 48, 6] sums of squared and signed errors over every station with an observation
all_n             [48, 6] number of those observations
field mode adds RMS differences of the surface fields to the home model at leads 1, 6, 24, 48, overall and by distance to
the nearest input station."""
import datetime as dt
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from . import forecast
from .constants import NX, NY
from .evaluate import local_midnights
from .model import Solace, sample_points

FIELD_LEADS = (1, 6, 24, 48)
FIELD_DIST_KM = (0, 8, 16, 32, 64, np.inf)
WEEKLY = ("field", "prune", "dose")  # modes run on every 7th case


def distance_bins(region):
    rc = region.p4.numpy().astype(np.float64)[~region.held]
    yy, xx = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    km = 4.0 * cKDTree(rc).query(np.c_[yy.ravel(), xx.ravel()])[0].reshape(NY, NX)
    b = np.digitize(km, FIELD_DIST_KM[1:-1])
    return b, np.bincount(b.ravel(), minlength=len(FIELD_DIST_KM) - 1)


@torch.no_grad()
def run_case(region, batch, truth, variants, field=False, bins=None):
    held = region.held
    names = list(variants)
    ok = np.isfinite(truth)
    P = np.empty((len(names), 48, 6, int(held.sum())), np.float16)
    sse = np.zeros((len(names), 48, 6))
    err = np.zeros_like(sse)
    model = Solace().eval()
    extra, home_field = {}, None
    nb = len(FIELD_DIST_KM) - 1
    if field:
        frms = np.zeros((len(names), len(FIELD_LEADS), 6))
        frmsd = np.zeros((len(names), len(FIELD_LEADS), 6, nb))
    for n, name in enumerate(names):
        state, off = variants[name]
        model.load_state_dict(state)
        model.switches_off = frozenset(off)
        fields = [] if field else None
        p, _ = model(batch, region, fields=fields)
        p = p[0].numpy()
        if not np.isfinite(p).all():
            raise RuntimeError(f"non-finite forecast: {name}")
        P[n] = p[..., held]
        e = np.where(ok, p - truth, 0.0)
        sse[n] = (e.astype(np.float64) ** 2).sum(-1)
        err[n] = e.sum(-1)
        if field:
            f = np.stack([fields[h - 1][0].numpy().astype(np.float16) for h in FIELD_LEADS]).astype(np.float32)
            if home_field is None:
                home_field = f
                hf = f.astype(np.float64).reshape(len(FIELD_LEADS), 6, -1)
                extra["field_home_std_dist"] = np.stack([hf[..., bins.ravel() == b].std(-1) for b in range(nb)], -1)
                extra["field_home_mean_dist"] = np.stack([hf[..., bins.ravel() == b].mean(-1) for b in range(nb)], -1)
            d2 = (f - home_field).astype(np.float64) ** 2
            frms[n] = np.sqrt(d2.mean((-2, -1)))
            d2 = d2.reshape(len(FIELD_LEADS), 6, -1)
            for b in range(nb):
                m = bins.ravel() == b
                frmsd[n, :, :, b] = np.sqrt(d2[..., m].mean(-1)) if m.any() else np.nan
    if field:
        extra |= dict(field_rms_diff=frms, field_home_std=home_field.astype(np.float64).std((-2, -1)),
                      field_home_mean=home_field.astype(np.float64).mean((-2, -1)), field_leads=np.array(FIELD_LEADS),
                      field_rms_diff_dist=frmsd, field_dist_km=np.array(FIELD_DIST_KM))
    persistence = sample_points(batch["c"], region.p4)[0].numpy()
    return dict(operators=np.array(names), held_pred=P, held_truth=truth[..., held].astype(np.float16),
                held_persistence=persistence[..., held].astype(np.float16), all_sse=sse, all_err=err, all_n=ok.sum(-1), **extra)


def run_year(region, source, observations, obs_start, year, mode, variants, out_dir, limit=0, log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    days = local_midnights(year, region.utc_offset)
    days = days[::7] if mode in WEEKLY else days
    days = days[:limit] if limit else days
    bins, counts = distance_bins(region) if mode == "field" else (None, None)
    for t0 in days:
        day = (t0 + dt.timedelta(hours=region.utc_offset)).date().isoformat()
        dest = out_dir / f"{day}.npz"
        if dest.exists():
            continue
        try:
            batch = forecast.build_inputs(region, source, observations, obs_start, t0)
        except ValueError:
            continue
        h0 = forecast.hours_between(obs_start, t0)
        truth = np.asarray(observations[h0 + 1:h0 + 49], np.float32)
        rec = run_case(region, batch, truth, variants, field=mode == "field", bins=bins)
        if mode == "field":
            rec["field_dist_cells"] = counts
        np.savez_compressed(dest, t0_utc=t0.isoformat(), held_id=region.stations.station_id.values[region.held], **rec)
        log(f"{mode} {day}")
