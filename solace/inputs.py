"""Forecast inputs from ERA5 (or GFS) meteorology and GEOS-CF chemistry on the regional 0.25-degree outer grid.

Archive layout (one NetCDF file per variable and month, all hours of the month, 61 x 81 points):
    <archive>/ecmwf_arco/hourly/surface/YYYYMM/{t2m,d2m,u10,v10,blh,tp,ssrd,tcc}.nc
    <archive>/ecmwf_arco/hourly/pressure/YYYYMM/{t,r,u,v,w}.nc          levels 925, 850, 700 hPa
    <archive>/geoscf_surface/YYYYMM/{pm25_rh35_gcc,pm25du_rh35_gcc,o3,no2,so2,co,dst2,dst3,dst4}.nc
    <archive>/geoscf_pressure/YYYYMM/{pm25_rh35_gcc,pm25du_rh35_gcc,o3,no2,so2,co}.nc   levels 975-700 hPa (8)
Units expected in the archive: ERA5 as distributed (K, m s-1, m, J m-2, %, Pa s-1); GEOS-CF PM2.5 in ug m-3 as distributed,
gases and dust bins converted from the distributed mole fractions (mol mol-1) by x 1e9: gases in ppb, dust in ug kg-1 (the
GEOS-CF dust molecular weight equals that of air, so the mole fraction equals the mass fraction).
"""
import calendar
import datetime as dt
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

from .constants import (HORIZON, MET, OUTER_INDEX, OUTER_MET, OUTER_NX, OUTER_NY, SPECIES, SPINUP, SURFACE6, LAYERS12)

ERA5_SURFACE = ("t2m", "d2m", "u10", "v10", "blh", "tp", "ssrd", "tcc")
ERA5_PRESSURE = ("t", "r", "u", "v", "w")
GEOSCF_SURFACE = ("pm25_rh35_gcc", "pm25du_rh35_gcc", "o3", "no2", "so2", "co", "dst2", "dst3", "dst4")
GEOSCF_PRESSURE = ("pm25_rh35_gcc", "pm25du_rh35_gcc", "o3", "no2", "so2", "co")
RESIDUAL_LEVELS = (925, 900, 850)  # hPa, mean over the levels = residual layer
UPPER_LEVELS = (850, 800, 750, 700)  # upper layer


def f16(a):
    """Round to float16 precision (the precision at which the training inputs were stored)."""
    return np.asarray(a).astype(np.float16).astype(np.float32)


# --------------------------------------------------------------------------------------------------- archive
class Archive:
    """Reads hourly slices from the monthly NetCDF archive of one region."""

    def __init__(self, path, bbox, max_open=40):
        import netCDF4
        self._nc = netCDF4
        self.path, self.bbox, self.max_open = path, bbox, max_open
        self.opened = OrderedDict()
        self.levels = {}

    def _dataset(self, path):
        key = str(path)
        if key not in self.opened:
            self.opened[key] = self._nc.Dataset(key)
        self.opened.move_to_end(key)
        while len(self.opened) > self.max_open:
            self.opened.popitem(last=False)[1].close()
        return self.opened[key]

    def read(self, product, var, start, count):
        """[count, (level,) 61, 81] float32 from start (UTC datetime), rows north to south."""
        parts, time = [], start
        base = self.path / "ecmwf_arco/hourly" / product[5:] if product.startswith("era5_") else self.path / product
        while count:
            hour = (time.day - 1) * 24 + time.hour
            n = min(count, calendar.monthrange(time.year, time.month)[1] * 24 - hour)
            ds = self._dataset(base / time.strftime("%Y%m") / f"{var}.nc")
            a = np.ma.filled(ds[var][hour:hour + n], np.nan).astype("f4")
            lat = np.asarray(ds["latitude" if product.startswith("era5_") else "lat"][:])
            lon = np.asarray(ds["longitude" if product.startswith("era5_") else "lon"][:])
            w, s, e, north = self.bbox
            np.testing.assert_allclose(lon, np.linspace(w, e, OUTER_NX))
            np.testing.assert_allclose(np.sort(lat), np.linspace(s, north, OUTER_NY))
            if lat[0] < lat[-1]:
                a = a[..., ::-1, :]
            level_name = "level" if product.startswith("era5_") else "lev"
            if level_name in ds.variables:
                self.levels[product] = [int(v) for v in np.asarray(ds[level_name][:])]
            parts.append(a)
            time += dt.timedelta(hours=n)
            count -= n
        return np.concatenate(parts)

    def era5(self, start, count):
        surface = {k: self.read("era5_surface", k, start, count) for k in ERA5_SURFACE}
        pressure = {k: self.read("era5_pressure", k, start, count) for k in ERA5_PRESSURE}
        return surface, pressure, self.levels["era5_pressure"]

    def geoscf(self, start, count=1):
        """Surface fields from start - 1 h (count + 1 hours) and pressure-level fields from start (count hours)."""
        surface = {k: self.read("geoscf_surface", k, start - dt.timedelta(hours=1), count + 1)[:, 0] for k in GEOSCF_SURFACE}
        pressure = {k: self.read("geoscf_pressure", k, start, count) for k in GEOSCF_PRESSURE}
        return surface, pressure, self.levels["geoscf_pressure"]

    def close(self):
        for ds in self.opened.values():
            ds.close()
        self.opened.clear()


# --------------------------------------------------------------------------------------------------- meteorology
def weather(surface, pressure, levels):
    """ERA5 fields [T, 61, 81] / [T, 3, 61, 81] -> inner meteorology at the nodes [T, 16, N] (raw units), extra fields
    [T, 4, N] (u700, v700, w850, w925) and outer meteorology [T, 12, 61, 81]."""
    s, p = surface, pressure
    at = lambda name, level: p[name][:, levels.index(level)]  # noqa: E731
    tc, td = s["t2m"] - 273.15, s["d2m"] - 273.15
    rh = np.clip(100 * np.exp(17.625 * td / (243.04 + td) - 17.625 * tc / (243.04 + tc)), 0, 100)
    ws = np.hypot(s["u10"], s["v10"])
    data = {"u10": s["u10"], "v10": s["v10"], "pblh": np.maximum(s["blh"], 20), "ventilation_index": s["blh"] * ws,
            "t2m": s["t2m"], "rh2m": rh, "ssrd_1h": np.maximum(s["ssrd"], 0) / 3600, "tp_1h": np.maximum(s["tp"], 0),
            "tcc": np.clip(s["tcc"], 0, 1)}
    for level in (925, 850, 700):
        for name in ("u", "v", "t", "r"):
            data[("rh" if name == "r" else name) + str(level)] = at(name, level)
    t = len(s["t2m"])
    nodes = np.stack([data[k] for k in MET], 1).reshape(t, len(MET), -1)
    extra = np.stack([data["u700"], data["v700"], at("w", 850), at("w", 925)], 1).reshape(t, 4, -1)
    outer = dict(data, t2m=tc, tp_1h=data["tp_1h"] * 1000)
    out = np.stack([outer[k] for k in OUTER_MET], 1)
    if not (np.isfinite(nodes).all() and np.isfinite(extra).all() and np.isfinite(out).all()):
        raise ValueError("non-finite meteorology")
    return nodes, extra, out


def model_meteorology(nodes):
    """Raw node meteorology -> model units: log1p ventilation index, temperatures in degC, precipitation in mm."""
    a = nodes.astype(np.float32).copy()
    v = MET.index("ventilation_index")
    a[:, v] = np.log1p(np.maximum(a[:, v], 0))
    for k in ("t2m", "t925", "t850"):
        a[:, MET.index(k)] -= 273.15
    a[:, MET.index("tp_1h")] *= 1000.0
    return a


# --------------------------------------------------------------------------------------------------- chemistry
def chemistry(surface, pressure, levels, dust_ratio):
    """GEOS-CF -> outer surface fields [T, 6, 61, 81] (PM2.5, PM10, log10 NO2, SO2, CO, O3), surface gases [T, 4, 61, 81]
    (O3, NO2, SO2, CO) and layers [T, 6, 2, 61, 81] (residual, upper). Surface values are the mean of the two
    time-averaged files around each hour; the PM10 surface adds the coarse dust bins."""
    sf = {k: 0.5 * (f16(a[:-1]) + f16(a[1:])) for k, a in surface.items()}
    gas = np.stack([sf[k] for k in ("o3", "no2", "so2", "co")], 1)
    pm = sf["pm25_rh35_gcc"]
    pm10 = pm + 1.2 * (0.62 * sf["dst2"] + sf["dst3"] + 0.87 * sf["dst4"])
    lg = np.log10(np.maximum(gas, 1e-3))
    outer = f16(np.stack([pm, pm10, lg[:, 1], lg[:, 2], lg[:, 3], lg[:, 0]], 1))
    res_i = [levels.index(v) for v in RESIDUAL_LEVELS]
    up_i = [levels.index(v) for v in UPPER_LEVELS]
    lay = {}
    for k, a in pressure.items():
        a = f16(a)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            residual, upper = np.nanmean(a[:, res_i], 1), np.nanmean(a[:, up_i], 1)
        upper = np.nan_to_num(upper, nan=0)
        lay[k] = np.stack([np.where(np.isnan(residual), upper, residual), upper], 1)
    pm25 = lay["pm25_rh35_gcc"]
    layers = f16(np.stack([pm25, pm25 + dust_ratio * lay["pm25du_rh35_gcc"], lay["o3"], lay["no2"], lay["so2"], lay["co"]], 1))
    if not (np.isfinite(outer).all() and np.isfinite(gas).all() and np.isfinite(layers).all()):
        raise ValueError("non-finite chemistry")
    return outer, f16(gas), layers


# --------------------------------------------------------------------------------------------------- time
def time_features(times, utc_offset):
    """[T, 5]: UTC hour on the circle, day of year on the circle, weekend (Saturday/Sunday local)."""
    h = np.array([t.hour for t in times], np.float32)
    d = np.array([t.dayofyear for t in times], np.float32)
    weekend = np.array([(t + pd.Timedelta(hours=utc_offset)).dayofweek >= 5 for t in times], np.float32)
    return np.c_[np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * d / 365.25),
                 np.cos(2 * np.pi * d / 365.25), weekend].astype(np.float32)


def level_scaling(times, level_ratio, level6):
    """Surface / lowest-model-level ratio of each surface species by month and 6-h bin [T, 6]."""
    r = np.empty((len(times), len(SPECIES)), np.float32)
    for j, t in enumerate(times):
        r[j, 0] = level_ratio[t.month - 1, t.hour // 6]
        r[j, 1:] = level6[1:, t.month - 1, t.hour // 6]
    return r


# --------------------------------------------------------------------------------------------------- batch
def assemble(c, emission_field, nodes, extra, outer_met, outer_chem, gas, layers, t0, region, horizon=HORIZON, spin=SPINUP,
             future=None):
    """One forecast case.
    c, emission_field: t0 analysis and 30-day mean analysis [6, NY, NX];
    nodes, extra, outer_met: raw meteorology for the spin + horizon + 1 hours from t0 - spin;
    outer_chem, gas, layers: GEOS-CF at t0 - spin (frame 0 of `chemistry`);
    future: optional (outer_chem, gas, layers) for every hour, used only as the auxiliary training target of the
    outer model (the forecast itself reads chemistry only from frame zero)."""
    t = spin + horizon + 1
    times = pd.date_range(pd.Timestamp(t0) - pd.Timedelta(hours=spin), periods=t, freq="h")
    scale = level_scaling(times, region.level_ratio, region.level6)
    o = np.zeros((t, 34, OUTER_NY, OUTER_NX), np.float32)
    first = slice(0, 6)
    if future is None:
        o[:, first] = outer_chem[None]
        o[:, list(SURFACE6[2:])] = gas[None]
        o[:, list(LAYERS12)] = layers.transpose(1, 0, 2, 3).reshape(12, OUTER_NY, OUTER_NX)[None]
    else:
        fc, fg, fl = future
        o[:, first] = fc
        o[:, list(SURFACE6[2:])] = fg
        o[:, list(LAYERS12)] = fl.transpose(0, 2, 1, 3, 4).reshape(t, 12, OUTER_NY, OUTER_NX)
    o[:, 0] *= scale[:, 0, None, None]
    o[:, list(SURFACE6[1:])] *= scale[:, 1:, None, None]
    o[:, [OUTER_INDEX[k] for k in OUTER_MET]] = f16(outer_met)
    arrays = {"c": c, "met": f16(model_meteorology(nodes[spin:spin + horizon])), "ext": f16(extra[spin:spin + horizon]),
              "o": o, "gus": layers, "tf": time_features(times[:-1], region.utc_offset), "last_m": emission_field}
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("non-finite forecast input")
    return {k: torch.as_tensor(np.ascontiguousarray(v))[None] for k, v in arrays.items()}


class ArraySource:
    """In-memory inputs with the same interface as Archive (used for the example case)."""

    def __init__(self, start, era5_surface, era5_pressure, era5_levels, geoscf_start, geoscf_surface, geoscf_pressure, geoscf_levels):
        self.start, self.gstart = start, geoscf_start
        self.e5s, self.e5p, self.e5l = era5_surface, era5_pressure, list(era5_levels)
        self.gs, self.gp, self.gl = geoscf_surface, geoscf_pressure, list(geoscf_levels)

    def _rows(self, start, origin, count):
        i = int((start - origin).total_seconds() // 3600)
        if i < 0:
            raise ValueError("requested hours before the stored inputs")
        return slice(i, i + count)

    def era5(self, start, count):
        r = self._rows(start, self.start, count)
        return {k: v[r] for k, v in self.e5s.items()}, {k: v[r] for k, v in self.e5p.items()}, self.e5l

    def geoscf(self, start, count=1):
        rs = self._rows(start - dt.timedelta(hours=1), self.gstart, count + 1)
        rp = self._rows(start, self.gstart + dt.timedelta(hours=1), count)
        return {k: v[rs] for k, v in self.gs.items()}, {k: v[rp] for k, v in self.gp.items()}, self.gl

    def save(self, path, **extra):
        arrays = {f"era5_surface_{k}": v for k, v in self.e5s.items()} | {f"era5_pressure_{k}": v for k, v in self.e5p.items()}
        arrays |= {f"geoscf_surface_{k}": v for k, v in self.gs.items()} | {f"geoscf_pressure_{k}": v for k, v in self.gp.items()}
        np.savez_compressed(path, era5_start=str(self.start.isoformat()), geoscf_start=str(self.gstart.isoformat()),
                            era5_levels=np.asarray(self.e5l), geoscf_levels=np.asarray(self.gl), **arrays, **extra)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        get = lambda p: {k[len(p):]: z[k] for k in z.files if k.startswith(p)}  # noqa: E731
        return cls(dt.datetime.fromisoformat(str(z["era5_start"])), get("era5_surface_"), get("era5_pressure_"), z["era5_levels"].tolist(),
                   dt.datetime.fromisoformat(str(z["geoscf_start"])), get("geoscf_surface_"), get("geoscf_pressure_"),
                   z["geoscf_levels"].tolist()), z
