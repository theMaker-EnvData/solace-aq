"""Command line: python -m solace <command> --help

  example    run the packaged Taiwan case and compare with the reference forecast
  forecast   one 48-h forecast from an archive and station observations
  hindcast   forecasts for every local midnight of a year
  scores     RMSE / R2 tables of hindcast folders
  mos-train  train the station MOS on one year of hindcasts
  mos-apply  apply a trained MOS to another year of hindcasts
  exchange   module-exchange, sensitivity, seed-control or experiment-A variants for a year of cases
  summary    tables of exchange runs (home advantage, module effects, intervals)
  prepare    regional preparation for training
  train      PM2.5 pre-training and full training"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def utc(s):
    t = dt.datetime.fromisoformat(s)
    return t.replace(tzinfo=dt.timezone.utc) if t.tzinfo is None else t.astimezone(dt.timezone.utc)


def cmd_example(a):
    from . import forecast, inputs
    from .model import sample_points
    from .region import load_region
    region = load_region("taiwan")
    source, z = inputs.ArraySource.load(Path(__file__).resolve().parents[1] / "examples/taiwan_2024-01-01/case.npz")
    t0, start = utc(str(z["t0"])), utc(str(z["obs_start"]))
    batch = forecast.build_inputs(region, source, z["observations"], start, t0)
    pred = forecast.run(region, batch)
    ref = z["reference_forecast"]
    truth = z["observations"][721:769]
    held = region.held
    print(f"max difference from the reference forecast: {np.abs(pred - ref).max():.4f}")
    persistence = np.broadcast_to(sample_points(batch["c"], region.p4)[0].numpy(), pred.shape)
    for k, sp in enumerate(("PM2.5", "PM10", "O3", "NO2", "SO2", "CO")):
        o = truth[:, k][:, held]
        ok = np.isfinite(o)
        r = lambda p: np.sqrt(np.mean((p[:, k][:, held][ok] - o[ok]) ** 2))  # noqa: E731
        print(f"{sp:6s} held-out stations, 48 h: RMSE forecast {r(pred):7.2f}   persistence {r(persistence):7.2f}")
    if a.out:
        np.savez_compressed(a.out, forecast=pred, station_id=region.stations.station_id.values)


def load_obs(a):
    return np.load(a.observations, mmap_mode="r"), utc(a.obs_start)


def cmd_forecast(a):
    from . import forecast, inputs
    from .region import load_region
    region = load_region(a.region)
    obs, start = load_obs(a)
    source = inputs.Archive(Path(a.archive), region.config["outer_bbox"])
    batch = forecast.build_inputs(region, source, obs, start, utc(a.t0))
    pred, fields = forecast.run(region, batch, fields=True)
    np.savez_compressed(a.out, forecast=pred, station_id=region.stations.station_id.values, t0_utc=utc(a.t0).isoformat(),
                        **({"surface_fields": fields.astype(np.float16)} if a.fields else {}))


def cmd_hindcast(a):
    from . import evaluate, inputs
    from .region import load_region
    region = load_region(a.region)
    obs, start = load_obs(a)
    source = inputs.Archive(Path(a.archive), region.config["outer_bbox"])
    skipped = evaluate.hindcast(region, source, obs, start, a.year, a.out, limit=a.limit)
    print(json.dumps({"skipped": skipped}, indent=1))


def cases(folder):
    files = sorted(Path(folder).glob("????-??-??.npz"))
    z = [np.load(f) for f in files]
    P = np.stack([x["forecast"] for x in z]).astype(np.float32)
    A0 = np.stack([x["persistence"] for x in z]).astype(np.float32)
    O = np.stack([x["truth"] for x in z]).astype(np.float32)
    met = np.stack([x["station_met"] for x in z]).astype(np.float32)
    t0 = pd.DatetimeIndex([pd.Timestamp(str(x["t0_utc"])) for x in z])
    return files, P, A0, O, met, t0


def cmd_scores(a):
    from . import evaluate
    from .constants import SPECIES
    from .region import load_region
    region = load_region(a.region)
    files, P, A0, O, met, t0 = cases(a.cases)
    models = {"forecast": P, "persistence": np.repeat(A0[:, None], 48, 1)}
    if a.mos:
        z = np.load(a.mos)
        models["forecast+MOS"] = z["mos_map"]
    stations = region.held if a.stations == "held" else ~region.held
    days = pd.to_datetime([f.stem for f in files])
    table = evaluate.scores(models, O, stations, days)
    table["species"] = [SPECIES[k] for k in table.species]
    table.to_csv(a.out, index=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def cmd_mos_train(a):
    from . import mos
    from .region import load_region
    region = load_region(a.region)
    _, P, A0, O, met, t0 = cases(a.cases)
    models = mos.train(P, A0, O, met, t0, mos.station_statics(region), region.held, region.utc_offset)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for sp, m in zip(("pm25", "pm10", "o3", "no2", "so2", "co"), models):
        m.save_model(str(Path(a.out) / f"mos_{sp}.txt"))


def cmd_mos_apply(a):
    import lightgbm as lgb
    from . import mos
    from .region import load_region
    region = load_region(a.region)
    files, P, A0, O, met, t0 = cases(a.cases)
    models = [lgb.Booster(model_file=str(Path(a.models) / f"mos_{sp}.txt")) for sp in ("pm25", "pm10", "o3", "no2", "so2", "co")]
    station, mapped = mos.apply(models, P, A0, met, t0, mos.station_statics(region), region.held, region.station_km, region.utc_offset)
    np.savez_compressed(a.out, mos_station=station, mos_map=mapped, days=np.array([f.stem for f in files]))


def cmd_exchange(a):
    import torch
    from . import exchange, experiments, inputs
    from .region import REGIONS, load_region
    region = load_region(a.region)
    obs, start = load_obs(a)
    source = inputs.Archive(Path(a.archive), region.config["outer_bbox"])
    home = torch.load(region.path / "weights.pt", weights_only=True)
    others = {r.name: torch.load(r / "weights.pt", weights_only=True) for r in sorted(REGIONS.iterdir())
              if r.name != region.name and (r / "weights.pt").exists()}
    other = torch.load(a.other, weights_only=True) if a.other else None
    variants = exchange.variants(a.mode, home, others, other)
    experiments.run_year(region, source, obs, start, a.year, a.mode, variants, a.out, limit=a.limit)


def cmd_summary(a):
    from . import summary
    folders = dict(x.split("=", 1) for x in a.folders)
    table = summary.matrix_table(folders) if a.kind == "matrix" else summary.effect_table(folders)
    table.to_csv(a.out, index=False)
    print(f"{len(table)} rows -> {a.out}")


def cmd_prepare(a):
    from .train import pipeline
    pipeline.prepare(a.region_dir, a.archive, a.observations, a.work)


def cmd_train(a):
    from .train import pipeline
    pipeline.train(a.region_dir, a.work, seed=a.seed, experiment_a=a.experiment_a)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m solace", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def obs_args(q):
        q.add_argument("--region", required=True, help="region name (regions/<name>) or folder")
        q.add_argument("--archive", required=True, help="regional ERA5 / GEOS-CF archive (see solace/inputs.py)")
        q.add_argument("--observations", required=True, help=".npy [hours, 6, stations] in stations.csv order, NaN missing")
        q.add_argument("--obs-start", required=True, help="UTC time of the first observation row, e.g. 2023-01-01T00")

    q = sub.add_parser("example"); q.add_argument("--out"); q.set_defaults(fn=cmd_example)
    q = sub.add_parser("forecast"); obs_args(q); q.add_argument("--t0", required=True, help="UTC start time")
    q.add_argument("--out", required=True); q.add_argument("--fields", action="store_true", help="also store hourly surface fields")
    q.set_defaults(fn=cmd_forecast)
    q = sub.add_parser("hindcast"); obs_args(q); q.add_argument("--year", type=int, required=True); q.add_argument("--out", required=True)
    q.add_argument("--limit", type=int, default=0, help="only the first N days")
    q.set_defaults(fn=cmd_hindcast)
    q = sub.add_parser("scores"); q.add_argument("--region", required=True); q.add_argument("--cases", required=True)
    q.add_argument("--mos", help="output of mos-apply for the same cases"); q.add_argument("--stations", choices=["held", "input"], default="held")
    q.add_argument("--out", required=True); q.set_defaults(fn=cmd_scores)
    q = sub.add_parser("mos-train"); q.add_argument("--region", required=True); q.add_argument("--cases", required=True)
    q.add_argument("--out", required=True); q.set_defaults(fn=cmd_mos_train)
    q = sub.add_parser("mos-apply"); q.add_argument("--region", required=True); q.add_argument("--cases", required=True)
    q.add_argument("--models", required=True); q.add_argument("--out", required=True); q.set_defaults(fn=cmd_mos_apply)
    q = sub.add_parser("exchange"); obs_args(q); q.add_argument("--year", type=int, required=True)
    q.add_argument("--mode", choices=["matrix", "core", "sensitivity", "seed", "seedfull", "prune", "dose", "fixa", "fixswap", "field"],
                   default="matrix")
    q.add_argument("--other", help="second model of the home region (seed, seedfull: another seed; fixa, fixswap: experiment A)")
    q.add_argument("--limit", type=int, default=0); q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_exchange)
    q = sub.add_parser("summary"); q.add_argument("kind", choices=["matrix", "effects"])
    q.add_argument("folders", nargs="+", help="home=folder of exchange files, one per region")
    q.add_argument("--out", required=True); q.set_defaults(fn=cmd_summary)
    q = sub.add_parser("prepare"); q.add_argument("--region-dir", required=True); q.add_argument("--archive", required=True)
    q.add_argument("--observations", required=True, help=".npy [hours, 6, stations] from 1 January 00 UTC of the first training year")
    q.add_argument("--work", required=True); q.set_defaults(fn=cmd_prepare)
    q = sub.add_parser("train"); q.add_argument("--region-dir", required=True); q.add_argument("--work", required=True)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--experiment-a", action="store_true", help="hold the surface and residual-layer wind blends at 1, the learned "
                   "diffusion at 0 and a_12 at 0 throughout training")
    q.set_defaults(fn=cmd_train)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
