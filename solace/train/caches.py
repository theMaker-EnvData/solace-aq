"""Yearly training caches (float16) built from the regional archive:
    met_<year>.npy     [T, 16, N]       inner meteorology at the nodes, model units
    extra_<year>.npy   [T, 4, N]        u700, v700, w850, w925
    outer_<year>.npy   [T, 18, 61, 81]  outer surface fields (6) and outer meteorology (12)
    gas_<year>.npy     [T, 4, 61, 81]   GEOS-CF surface O3, NO2, SO2, CO
    layers_<year>.npy  [T, 6, 2, 61, 81] GEOS-CF residual and upper layers of the six species
and the dust ratio that turns GEOS-CF fine dust into the coarse part of PM10 aloft."""
import datetime as dt
import json
import warnings

import numpy as np

from .. import inputs
from ..constants import MET, OUTER_NX, OUTER_NY

CHUNK = 48


def surface_means(archive, var, start, count):
    """Hourly surface values: mean of the two time-averaged GEOS-CF files around each hour; the first hour of a year
    uses its own file only."""
    if (start.month, start.day, start.hour) == (1, 1, 0):
        a = inputs.f16(archive.read("geoscf_surface", var, start, count)[:, 0])
        out = a.copy()
        out[1:] = 0.5 * (a[1:] + a[:-1])
        return out
    a = inputs.f16(archive.read("geoscf_surface", var, start - dt.timedelta(hours=1), count + 1)[:, 0])
    return 0.5 * (a[1:] + a[:-1])


def dust_ratio(archive, years, week=168):
    """Ratio of the coarse dust (PM10 minus PM2.5 dust bins) to the fine dust over the training years."""
    coarse_sum = fine_sum = 0.0
    for year in years:
        start = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
        hours = (dt.datetime(year + 1, 1, 1, tzinfo=dt.timezone.utc) - start).days * 24
        for offset in range(0, hours, week):
            t = start + dt.timedelta(hours=offset)
            n = min(week, hours - offset)
            coarse = 1.2 * (0.62 * surface_means(archive, "dst2", t, n) + surface_means(archive, "dst3", t, n)
                            + 0.87 * surface_means(archive, "dst4", t, n))
            fine = surface_means(archive, "pm25du_rh35_gcc", t, n)
            coarse_sum += float(np.nansum(coarse, dtype=np.float64))
            fine_sum += float(np.nansum(fine, dtype=np.float64))
    return coarse_sum / fine_sum


def hours_in(year):
    return (dt.date(year + 1, 1, 1) - dt.date(year, 1, 1)).days * 24


def chunk(archive, when, count, dust):
    """Cache rows for `count` hours from `when`."""
    nodes, extra, outer_met = inputs.weather(*archive.era5(when, count))
    sf = {v: surface_means(archive, v, when, count) for v in ("pm25_rh35_gcc", "o3", "no2", "so2", "co", "dst2", "dst3", "dst4")}
    pm = sf["pm25_rh35_gcc"]
    pm10 = pm + 1.2 * (0.62 * sf["dst2"] + sf["dst3"] + 0.87 * sf["dst4"])
    gas = np.stack([sf[k] for k in ("o3", "no2", "so2", "co")], 1)
    lg = np.log10(np.maximum(gas, 1e-3))
    surface = np.stack([pm, pm10, lg[:, 1], lg[:, 2], lg[:, 3], lg[:, 0]], 1)
    levels = None
    lay = {}
    for var in inputs.GEOSCF_PRESSURE:
        a = inputs.f16(archive.read("geoscf_pressure", var, when, count))
        levels = archive.levels["geoscf_pressure"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            residual = np.nanmean(a[:, [levels.index(v) for v in inputs.RESIDUAL_LEVELS]], 1)
            upper = np.nanmean(a[:, [levels.index(v) for v in inputs.UPPER_LEVELS]], 1)
        upper = np.nan_to_num(upper, nan=0.0)
        lay[var] = np.stack([np.where(np.isnan(residual), upper, residual), upper], 1)
    pm25 = lay["pm25_rh35_gcc"]
    layers = np.stack([pm25, pm25 + dust * lay["pm25du_rh35_gcc"], lay["o3"], lay["no2"], lay["so2"], lay["co"]], 1)
    rows = {"met": inputs.model_meteorology(nodes), "extra": extra, "outer": np.concatenate([surface, outer_met], 1),
            "gas": gas, "layers": layers}
    for k, v in rows.items():
        if not np.isfinite(v).all():
            raise ValueError(f"non-finite {k} at {when}")
    return {k: v.astype(np.float16) for k, v in rows.items()}


def build_year(archive, year, dust, folder, n_nodes):
    """Write the five caches of one UTC year (resumable in 48-h chunks)."""
    folder.mkdir(parents=True, exist_ok=True)
    hours = hours_in(year)
    shapes = {"met": (hours, len(MET), n_nodes), "extra": (hours, 4, n_nodes), "outer": (hours, 18, OUTER_NY, OUTER_NX),
              "gas": (hours, 4, OUTER_NY, OUTER_NX), "layers": (hours, 6, 2, OUTER_NY, OUTER_NX)}
    progress = folder / f"cache_{year}.progress.json"
    done = json.loads(progress.read_text())["hours"] if progress.exists() else 0
    if done >= hours:
        return
    arrays = {k: np.lib.format.open_memmap(folder / f"{k}_{year}.npy", mode="r+" if done else "w+", dtype=np.float16, shape=s)
              for k, s in shapes.items()}
    start = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
    for offset in range(done, hours, CHUNK):
        n = min(CHUNK, hours - offset)
        for k, v in chunk(archive, start + dt.timedelta(hours=offset), n, dust).items():
            arrays[k][offset:offset + n] = v
            arrays[k].flush()
        progress.write_text(json.dumps({"hours": offset + n, "total": hours}))


def open_year(folder, year):
    return {k: np.load(folder / f"{k}_{year}.npy", mmap_mode="r") for k in ("met", "extra", "outer", "gas", "layers")}
