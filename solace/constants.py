"""Fixed dimensions and channel orders of SOLACE."""

SPECIES = ("pm25", "pm10", "o3", "no2", "so2", "co")

# Inner grid: 4-km cells, 175 rows (north to south) x 153 columns (west to east), and a halo of 24 cells (96 km), larger
# than the largest hourly displacement.
NY, NX = 175, 153
HALO = 24
NYP, NXP = NY + 2 * HALO, NX + 2 * HALO
CELL_KM = 4.0
CELLS_PER_HOUR = 3600.0 / (CELL_KM * 1000)  # m s-1 -> cells per hour

# Outer grid: 0.25 degree, 61 x 81.
OUTER_NY, OUTER_NX = 61, 81

# Meteorology at the nodes of the meteorological grid (regridded to the inner cells with fixed triangle weights).
MET = ("u10", "v10", "u925", "v925", "u850", "v850", "pblh", "ventilation_index", "t2m", "t925", "t850",
       "rh2m", "rh925", "ssrd_1h", "tp_1h", "tcc")
MET_INDEX = {k: i for i, k in enumerate(MET)}
EXTRA = ("u700", "v700", "w850", "w925")
EXTRA_INDEX = {k: i for i, k in enumerate(EXTRA)}

# Outer input tensor o[:, time, channel], 34 channels: surface PM2.5 and PM10 and log10 of the surface NO2, SO2, CO and
# O3 mixing ratios (the time-constant auxiliary inputs of the outer model), outer meteorology, the four surface gases,
# and the residual and upper layers of the six species.
OUTER_CHANNELS = (("pm25", "pm10", "log10_no2", "log10_so2", "log10_co", "log10_o3")
                  + ("u10", "v10", "u850", "v850", "u700", "v700", "pblh", "t2m", "rh2m", "tp_1h", "tcc", "ssrd_1h")
                  + ("o3", "no2", "so2", "co")
                  + tuple(f"{layer}_{sp}" for layer in ("residual", "upper") for sp in SPECIES))
OUTER_INDEX = {k: i for i, k in enumerate(OUTER_CHANNELS)}
OUTER_MET = ("u10", "v10", "u850", "v850", "u700", "v700", "pblh", "t2m", "rh2m", "tp_1h", "tcc", "ssrd_1h")
OUTER_AUX = ("pm10", "log10_no2", "log10_so2", "log10_co", "log10_o3")
SURFACE6 = tuple(OUTER_INDEX[k] for k in ("pm25", "pm10", "o3", "no2", "so2", "co"))  # surface species in SPECIES order
LAYERS12 = tuple(range(22, 34))  # residual (6) and upper (6) layers
N_OUTER = 34

HORIZON = 48  # forecast hours
SPINUP = 6  # outer-only hours before t0
