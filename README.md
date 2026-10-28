# SOLACE

SOLACE is a station-trained gridded process model for regional air-quality forecasting. It forecasts PM2.5, PM10, O3,
NO2, SO2 and CO hourly for 48 h on a 4-km grid from three inputs: the hourly observations of a monitoring network up to
the forecast start, meteorology (ERA5 or GFS), and a chemical-transport hindcast (GEOS-CF) at 6 h before the start. No
emission inventory and no chemistry after the start time are used.

Every hour the model advects and diffuses the surface, residual and upper layers, applies a learned net change of each
species, adds an emission term whose spatial pattern is the 30-day mean of the station analyses, and exchanges mass
between the layers. An outer model on a 0.25° grid carries the boundary forward from the start time. The model has about
142,000 trained parameters and runs on a CPU.

This repository holds the model, training and analysis code and the trained weights of the five regions of the paper:

> TK Lee. A station-trained process model forecasts air quality without regional emission inventories.

The forecasts evaluated in the paper and the data behind its figures are archived at
[doi:10.5281/zenodo.23728107](https://doi.org/10.5281/zenodo.23728107).

## Relation to the paper's code

The paper's results were produced with the original research code. This package is a clean reimplementation of the
same model and procedures. It has been checked against the original code as follows (CPU against CPU unless noted):

| Part | Check | Result |
|---|---|---|
| Forecast model | 48-h rollout, five regions, same inputs and weights | relative difference ≤ 3 × 10⁻⁷ |
| Inputs | meteorology, outer fields, time features, four regions | identical |
| Initial fields (OI) | analysis and 30-day mean, four regions | within one float16 rounding step (forecasts ≤ 0.03 µg m⁻³ apart) |
| Hindcasts | Taiwan, first three days of 2024, `python -m solace hindcast` | forecast difference ≤ 0.03, identical truth |
| Training preparation | QC, held-out split, input network, training times, 30-day means, caches, dust ratio | identical |
| Training constants | loss scales, outer statistics, level ratios, boundary-gate thresholds | identical up to float32 rounding |
| OI covariance fit | 40 steps, PM2.5 (sparse layouts) and NO2 | parameters within 7 × 10⁻⁷ |
| PM2.5 pre-training | 6 steps from the same seed: window draws, weights, selection score | same draws, weights within 6 × 10⁻⁷ |
| Full training | 3 steps from the same initial weights: window draws, weights, selection score | same draws, weights within 4 × 10⁻⁹ |
| Module exchange and sensitivity | 55 operator variants on one Taiwan case | within float16 storage precision |
| MOS | original LightGBM models applied to the 2024 Israel hindcasts | identical station values, map within 5 × 10⁻⁵ |
| Experiment tables | home advantage, module effects, sensitivity, core transplant, seed control, Experiment A exchange, recomputed from the original per-case files | within 3 × 10⁻¹³ |
| Experiment A | pinned training, 6 + 3 steps against the original pinning mechanism | same draws, weights within 6 × 10⁻⁷ |
| Synthetic supervision | arms S and D, 3 steps from the same start | same draws, weights within 2 × 10⁻⁸ |
| Separability | error model and separation of 173,160 network-pair-window cases | within 4 × 10⁻⁹ |
| Recovery and field diagnostics | verdict tables, field differences, transport functions, source-sink and net-balance fields | within 5 × 10⁻⁷ |
| Attenuation | summary ratios from the original per-window files; one window recomputed (86 directions) | within 10⁻¹⁶; recomputed group medians equal to five digits |
| CTM and AQM comparisons | five-region skill table and the Southern California AQM table from the same forecasts | within 2 × 10⁻¹³ |

Training on a GPU is not bit-reproducible, so retraining gives weights that differ from the released ones by roughly the
spread between random seeds.

## Install

Python 3.10 or later.

```
pip install -r requirements.txt
python -m solace example
```

`example` runs the packaged Taiwan case (forecast from 1 January 2024, 00 local time). It prints the largest difference
from the forecast of the original code and the 48-h RMSE at the held-out stations against persistence.

`pytest` runs the same check as a test.

## Forecasting with your own data

A forecast needs:

1. **Region files**: `regions/<name>/` (provided for `korea`, `taiwan`, `europe` (Netherlands-Rhine), `middle_east`
   (Israel) and `us_west` (Southern California)).
2. **Observations**: a float array `[hours, 6, stations]` (`.npy`), species in the order pm25, pm10, o3, no2, so2, co
   (PM in µg m⁻³, gases in ppb), stations in the order of `regions/<name>/stations.csv`, NaN for missing values. The
   rows must cover at least 720 h before the start time.
3. **Meteorology and GEOS-CF**: an archive of monthly NetCDF files on the region's 0.25° outer grid. The layout is
   described at the top of `solace/inputs.py`.

```
python -m solace forecast --region taiwan --archive ARCHIVE --observations obs.npy --obs-start 2023-01-01T00 \
    --t0 2023-12-31T16 --out forecast.npz [--fields]
python -m solace hindcast --region taiwan --archive ARCHIVE --observations obs.npy --obs-start 2023-01-01T00 \
    --year 2024 --out cases/
python -m solace scores --region taiwan --cases cases/ --out scores.csv
```

Forecasts start at local standard midnight in the paper (UTC offsets in `regions/<name>/region.json`). The 75 % rule
applies: a forecast is made only if at least 75 % of each species' input stations report at the start time.

## MOS

```
python -m solace mos-train --region taiwan --cases cases_2022/ --out mos/
python -m solace mos-apply --region taiwan --cases cases_2024/ --models mos/ --out mos_2024.npz
python -m solace scores --region taiwan --cases cases_2024/ --mos mos_2024.npz --out scores.csv
```

The MOS learns the forecast error at the input stations for each species with LightGBM and spreads the corrections to all
stations with a 40-km Gaussian weight. Station static predictors are in `stations.csv` for the four regions whose MOS is
described in the paper.

## Module exchange and sensitivity

```
python -m solace exchange --region taiwan --archive ARCHIVE --observations obs.npy --obs-start 2023-01-01T00 \
    --year 2024 --mode matrix|core|sensitivity --out exchange/
```

`solace/exchange.py` defines the modules (transport, net change, emission, layer exchange, layer initialisation, outer
model), the knock-outs and the bounds. `solace/evaluate.py` has the block-bootstrap intervals used for the comparisons.

## Analyses of the paper

| Paper item | Code |
|---|---|
| Skill against persistence and CTMs, intervals | `evaluate.py` (hindcasts, block bootstrap), `comparison.py` (CTM extraction, observation-corrected CTMs, skill tables, wildfire subsets) |
| MOS | `mos.py` |
| Module exchange, home advantage, core and local transplants, seed control | `exchange.py` (variants), `experiments.py` (case files), `summary.py` (tables) |
| Sensitivity (knock-outs, bounds, dose) | `exchange.py`, `Solace.switches_off` |
| Experiment A (insensitive parts removed and retrained) | `python -m solace train --experiment-a`, `exchange.variants("fixa" / "fixswap")` |
| Learned fields and functions: source-sink fields, net balance, seed spread, transport functions, field differences | `fields.py` |
| Separability without training, recovery with sparse and dense supervision | `synthetic.py` (teacher, supervision for `train.train`, error model, separation, recovery diagnostics) |
| Attenuation from components to stations | `attenuation.py` |

`regions/<name>/variants/` holds the other trained weights used by these analyses: a second random seed (`seed1`, Taiwan
and Southern California), the Experiment A models (`experiment_a`, same two regions), and for Taiwan the start state and
final weights of the synthetic-supervision arms (`synthetic_start`, `synthetic_S`, `synthetic_D`, with `_raw` for the
non-averaged weights). `regions/<name>/static.npz` holds the 4-km land-cover fractions, log population and elevation
(WorldClim 2.1, C3S Land Cover 2022, WorldPop 2022) used for the land samples and the MOS predictors.

## Training a region

```
python -m solace prepare --region-dir regions/myregion --archive ARCHIVE --observations obs_2018_2022.npy --work work/
python -m solace train --region-dir regions/myregion --work work/
```

The region folder needs `region.json` (projection, inner bounds, outer box, UTC offset, training and selection years)
and `stations.csv` (`station_id`, `longitude`, `latitude`). The observations start at 1 January 00 UTC of the first
training year. `prepare` fits the OI covariances, computes the analyses, the 30-day means, the input caches and the
training constants. `train` runs the PM2.5 pre-training (4,000 steps) and the full training (2,000 steps, batch 4,
48-h windows) and writes `weights.pt`. Training uses the GPU when one is available (`--device`); the full training needs about 3 GB of GPU memory.

These steps reproduce the procedure of the Taiwan, Israel, Netherlands-Rhine and Southern California models (training
2018–2021, selection 2022 at the held-out stations). The Korea weights come from the original Korean run. That run used
the same model and optimiser but a different data preparation: training windows and analyses chosen by absolute station
counts instead of the 75 % input rule, its own meteorological node grid, the emission pattern averaged over the 30
calendar days before the start date, and checkpoint selection over leads up to 48 h.

## Licences

- Code: MIT (`LICENSE`).
- Weights and regional constants (`regions/*/weights.pt`, `constants.npz`, `oi_*.pt`): CC BY-NC 4.0
  (`LICENSE-weights`). Commercial use by separate agreement.
- The example case contains ERA5 (Copernicus Climate Change Service), GEOS-CF (NASA GMAO) and Taiwan Ministry of
  Environment observations (Open Government Data License, version 1.0). See `examples/README.md`.

## Citation

See `CITATION.cff`.
