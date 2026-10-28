# Example case: Taiwan, 1 January 2024

`taiwan_2024-01-01/case.npz` holds everything needed for the forecast that starts on 1 January 2024 at 00 local time
(31 December 2023, 16 UTC):

- `era5_surface_*`, `era5_pressure_*` (925, 850, 700 hPa): ERA5 hourly fields on the Taiwan outer grid (0.25°, 61 × 81)
  from 10 UTC on 31 December 2023 for 55 hours. Contains modified Copernicus Climate Change Service information (2026).
  Neither the European Commission nor ECMWF is responsible for any use of this information.
- `geoscf_surface_*`, `geoscf_pressure_*` (975–700 hPa): GEOS-CF hindcast fields around 10 UTC, from NASA's Global
  Modeling and Assimilation Office.
- `observations`: hourly station observations from 720 h before the start to 48 h after it, in the order of
  `regions/taiwan/stations.csv` (Ministry of Environment, Taiwan; Open Government Data License, version 1.0).
- `reference_forecast`: the station forecast of the original research code for this case.

`python -m solace example` runs the case.
