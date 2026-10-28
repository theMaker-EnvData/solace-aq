"""Every packaged region loads, and its weights fit the model exactly."""
import pytest
import torch

from solace import exchange
from solace.model import Solace, load_weights
from solace.region import REGIONS, load_region

NAMES = sorted(p.name for p in REGIONS.iterdir() if (p / "weights.pt").exists())


@pytest.mark.parametrize("name", NAMES)
def test_region(name):
    region = load_region(name)
    model = load_weights(region.path / "weights.pt")
    n = len(region.stations)
    assert region.p4.shape == (n, 2) and region.eligible.shape == (6, n)
    assert sum(p.numel() for p in model.parameters()) > 140_000
    for sp in ("pm25", "pm10", "o3", "no2", "so2", "co"):
        assert set(region.covariance(sp).state_dict()) == {"logL", "logw", "logr"}


def test_exchange_variants():
    states = {n: torch.load(load_region(n).path / "weights.pt", weights_only=True) for n in NAMES}
    home = states.pop(NAMES[0])
    variants = exchange.matrix(home, states)
    assert len(variants) == 1 + len(states) * (1 + len(exchange.MODULES))
    model = Solace()
    for st in variants.values():
        model.load_state_dict(st)
    assert set(exchange.sensitivity(home)) >= set(exchange.KNOCKOUTS) | set(exchange.BOUNDS)
