"""Regional preparation and training.

prepare(): station positions and held-out split, observation QC, input network, OI covariances, gridded analyses,
           30-day mean analyses, dust ratio, yearly input caches and the constants of constants.npz.
train():   PM2.5 pre-training, then the full model; writes weights.pt.

Region folder: region.json (crs, bounds, outer_bbox, utc_offset, active_network, train_years, selection_year) and
stations.csv (station_id, longitude, latitude). Observations: one array [T, 6, S] of hourly values (NaN missing) in
SPECIES order and stations.csv order, row 0 at `obs_start` (UTC 00 on 1 January of the first training year)."""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import exchange, geometry
from ..constants import OUTER_NX, OUTER_NY, SPECIES
from ..inputs import Archive, time_features
from ..region import load_region
from . import analyses, caches, covfit, derived, observations, pretrain
from . import train as full
from .windows import Year


def log_to(path):
    def log(message):
        line = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S} {message}"
        print(line, flush=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return log


def years_of(cfg):
    return list(cfg["train_years"]) + [cfg["selection_year"]]


def year_rows(year, epoch):
    a = int((dt.datetime(year, 1, 1) - dt.datetime(epoch, 1, 1)).total_seconds() // 3600)
    return slice(a, a + caches.hours_in(year))


def prepare(region_dir, archive_dir, observations_path, work, log=None, covariance_steps=None):
    region_dir, work = Path(region_dir), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    log = log or log_to(work / "prepare.log")
    cfg = json.loads((region_dir / "region.json").read_text(encoding="utf-8"))
    years, train_years, epoch = years_of(cfg), cfg["train_years"], years_of(cfg)[0]

    st = pd.read_csv(region_dir / "stations.csv", dtype={"station_id": str})
    crs = geometry.CRS.from_user_input(cfg["crs"])
    km, p4 = geometry.station_positions(st.longitude, st.latitude, cfg["bounds"], crs)
    st["y_km"], st["x_km"] = km[:, 0].astype(np.float64), km[:, 1].astype(np.float64)
    st["held"] = observations.held_split(st)
    geo = geometry.build(cfg["bounds"], crs, cfg["outer_bbox"])
    log(f"stations {len(st)}, held out {int(st.held.sum())}")

    full_obs = np.load(observations_path, mmap_mode="r")
    obs = {}
    for y in years:
        f = work / f"observations_{y}.npy"
        if not f.exists():
            np.save(f, observations.quality_control(np.asarray(full_obs[year_rows(y, epoch)], np.float32), st))
        obs[y] = np.load(f)
    held = st.held.to_numpy(bool)
    eligible = observations.eligibility([obs[y] for y in train_years])
    obs = {y: observations.mask_ineligible(v, eligible, held) for y, v in obs.items()}
    active = dict(zip(years, observations.active_network([obs[y] for y in years]))) if cfg.get("active_network") else None
    log(f"input stations by species {(eligible & ~held).sum(1).tolist()}")

    xy = st[["y_km", "x_km"]].to_numpy(np.float32)
    for k, sp in enumerate(SPECIES):
        out = region_dir / f"oi_{sp}.pt"
        if out.exists():
            continue
        el = eligible[k]
        h = held[el]
        Y = np.concatenate([obs[y][:, k, el] for y in years])
        if active is None:
            keep = np.isfinite(Y[:, ~h]).sum(1) >= observations.COVERAGE * (~h).sum()
        else:
            live = np.concatenate([active[y][:, k, el] for y in years])[:, ~h]
            keep = np.isfinite(Y[:, ~h]).sum(1) >= observations.COVERAGE * np.maximum(live.sum(1), 1)
        c = covfit.CONFIG.get(sp, covfit.DEFAULT)
        log(f"OI covariance {sp}: {int(keep.sum())} hours")
        cov = covfit.fit(Y[keep], xy[el], h, covariance_steps or c["steps"], c["batch"], cosine=c["cosine"], sparse=c["sparse"], log=log)
        torch.save(cov.state_dict(), out)

    adir = work / "analyses"
    adir.mkdir(exist_ok=True)
    six_all = {k: [] for k in range(6)}
    for y in years:
        cf = observations.coverage(obs[y], eligible, held, None if active is None else active[y])
        idx = analyses.training_times(cf)
        np.save(adir / f"times_{y}.npy", idx)
        six = np.arange(0, len(obs[y]), 6)
        for k, sp in enumerate(SPECIES):
            for tag, t in (("train", idx), ("six", six)):
                f = adir / f"{sp}_{y}_{tag}.npy"
                if not f.exists():
                    cov = covfit_load(region_dir, sp)
                    np.save(f, analyses.fields(obs[y][t, k], cf[t, k], xy, eligible[k] & ~held, cov))
            six_all[k].append(np.load(adir / f"{sp}_{y}_six.npy", mmap_mode="r"))
        log(f"analyses {y}: {len(idx)} training windows")
    rdir = work / "rolling"
    rdir.mkdir(exist_ok=True)
    times = np.arange(sum(len(obs[y]) for y in years), step=6)
    np.save(rdir / "times.npy", times)
    for k, sp in enumerate(SPECIES):
        if not (rdir / f"{sp}.npy").exists():
            np.save(rdir / f"{sp}.npy", analyses.rolling_means(times, np.concatenate([np.asarray(a) for a in six_all[k]])))

    archive = Archive(Path(archive_dir), cfg["outer_bbox"])
    dfile = work / "dust_ratio.json"
    if not dfile.exists():
        dfile.write_text(json.dumps({"dust_ratio": caches.dust_ratio(archive, train_years)}))
    dust = json.loads(dfile.read_text())["dust_ratio"]
    for y in years:
        caches.build_year(archive, y, dust, work / "cache", OUTER_NY * OUTER_NX)
        log(f"cache {y} complete")
    archive.close()

    grid = torch.as_tensor(geo["halo_grid"])
    sr = torch.as_tensor(geo["gate_sr"])
    train_data = []
    for y in train_years:
        idx = np.load(adir / f"times_{y}.npy")
        anas = [np.load(adir / f"{sp}_{y}_train.npy", mmap_mode="r") for sp in SPECIES]
        c = caches.open_year(work / "cache", y)
        train_data.append((y, idx, anas, c["outer"], c["gas"]))
    R = derived.pm25_level_ratio([(y, idx, anas[0], outer) for y, idx, anas, outer, gas in train_data], grid)
    q = derived.pm25_gate([(y, idx, anas[0], outer) for y, idx, anas, outer, gas in train_data], grid, R, sr)
    R6, q95, q99 = derived.species_level_gate(train_data, grid, sr, q)
    np.savez_compressed(region_dir / "constants.npz", **{k: geo[k] for k in ("rv", "rw", "halo_grid", "outer_inner_mask", "dx_km",
                                                                         "dy_km", "gate_sr", "gate_sp")},
                        p4=p4, gate6_q95=q95, gate6_q99=q99, level_ratio=R, level6=R6, dust_ratio=np.float64(dust),
                        eligible=eligible)
    st[["station_id", "longitude", "latitude", "y_km", "x_km", "held"]].to_csv(region_dir / "stations.csv", index=False)
    log("preparation complete")


def covfit_load(region_dir, sp):
    from ..oi import load_covariance
    return load_covariance(Path(region_dir) / f"oi_{sp}.pt")


def open_years(region, work, years, epoch):
    work = Path(work)
    held = region.held
    obs = {y: observations.mask_ineligible(np.load(work / f"observations_{y}.npy"), region.eligible, held) for y in years}
    times = np.load(work / "rolling/times.npy")
    means = [np.load(work / f"rolling/{sp}.npy", mmap_mode="r") for sp in SPECIES]
    out = {}
    for y in years:
        idx = np.load(work / f"analyses/times_{y}.npy")
        anas = [np.load(work / f"analyses/{sp}_{y}_train.npy", mmap_mode="r") for sp in SPECIES]
        out[y] = Year(y, obs[y], idx, anas, caches.open_year(work / "cache", y), region.utc_offset, region.level_ratio, region.level6,
                      (times, means), epoch)
    return obs, out


def train(region_dir, work, seed=0, log=None, pretrain_steps=pretrain.CONFIG["steps"], steps=full.CONFIG["steps"],
          selection_windows=full.CONFIG["selection_windows"], experiment_a=False, device=None):
    region_dir, work = Path(region_dir), Path(work)
    log = log or log_to(work / "train.log")
    region = load_region(region_dir)
    pins = exchange.PRUNE["prune:all"] if experiment_a else None
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"training on {device}")
    cfg = region.config
    years, train_years, sel_year = years_of(cfg), cfg["train_years"], cfg["selection_year"]
    obs, data = open_years(region, work, years, years[0])
    held = region.held
    keep = ~held

    pfile = work / "pretrain.pt"
    if not pfile.exists():
        Eh = derived.lead_scale([obs[y][:, 0] for y in train_years], keep, 24)
        Y = obs[sel_year][:, 0]
        live = None
        if cfg.get("active_network"):
            live = observations.active_network([obs[y] for y in years])[years.index(sel_year)]
        cf = observations.coverage(obs[sel_year], region.eligible, held, live)[:, 0]
        idx = np.arange(3, len(Y) - 24, 6)
        idx = idx[cf[idx] >= observations.COVERAGE]
        tf = time_features(pd.date_range(f"{sel_year}-01-01", f"{sel_year}-12-31 23:00", freq="h", tz="UTC"), region.utc_offset)
        sel = pretrain.Selection(Y, tf, data[sel_year].c["met"], idx, region.station_km, held, region.covariance("pm25"), Eh,
                                 region.p4, region.rv, region.rw)
        best = pretrain.train([data[y] for y in train_years], sel, held, Eh, region.p4, region.rv, region.rw, steps=pretrain_steps, seed=seed,
                              log=log, pins=pins, device=device)
        torch.save(best[1], pfile)
        log(f"pre-training selected step {best[2]} (score {best[0]:.4f})")

    Eh = np.stack([derived.lead_scale([obs[y][:, k] for y in train_years], keep, 48) for k in range(len(SPECIES))])
    mu, sd = derived.outer_stats([data[y].c["outer"] for y in train_years])
    torch.manual_seed(seed)
    model = full.initial_model(torch.load(pfile, weights_only=True), mu, sd)
    sel = full.Selection(data[sel_year], held, Eh, n=selection_windows)
    best = full.train(model, [data[y] for y in train_years], sel, region, held, Eh, steps=steps, seed=seed, log=log, pins=pins, device=device)
    torch.save(best[1], region_dir / "weights.pt")
    log(f"training selected step {best[2]} (score {best[0]:.4f})")
