"""The packaged Taiwan case reproduces the forecast of the original research code."""
import datetime as dt
from pathlib import Path

import numpy as np

from solace import forecast, inputs
from solace.region import load_region

CASE = Path(__file__).resolve().parents[1] / "examples/taiwan_2024-01-01/case.npz"


def test_taiwan_case():
    region = load_region("taiwan")
    source, z = inputs.ArraySource.load(CASE)
    t0, obs_start = dt.datetime.fromisoformat(str(z["t0"])), dt.datetime.fromisoformat(str(z["obs_start"]))
    batch = forecast.build_inputs(region, source, z["observations"], obs_start, t0)
    pred = forecast.run(region, batch)
    ref = z["reference_forecast"]
    assert pred.shape == ref.shape == (48, 6, len(region.stations))
    # initial fields are stored at float16 precision; one-unit rounding differences move forecasts by < 0.1 ug m-3 or ppb
    assert np.abs(pred - ref).max() < 0.1
    assert np.abs(pred - ref).max() / np.abs(ref).max() < 1e-4
