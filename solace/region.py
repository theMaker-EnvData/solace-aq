"""Regional files (regions/<name>/): region.json (projection, inner bounds, outer box, UTC offset), constants.npz (grid
geometry, boundary-gate thresholds, level ratios, dust ratio, input network), stations.csv, oi_<species>.pt, weights.pt."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REGIONS = Path(__file__).resolve().parents[1] / "regions"


@dataclass
class Region:
    name: str
    path: Path
    config: dict
    utc_offset: int
    rv: torch.Tensor  # [NY*NX, 3] outer grid points of each inner cell
    rw: torch.Tensor  # [NY*NX, 3] interpolation weights
    halo_grid: torch.Tensor  # [NYP, NXP, 2] position of the padded inner cells in the outer grid
    outer_inner_mask: torch.Tensor  # [61, 81] outer points inside the inner domain
    dy_km: float
    dx_km: torch.Tensor  # [61, 81]
    p4: torch.Tensor  # [S, 2] station positions in cells (row, column)
    gate_sr: torch.Tensor  # [NY, NX] segment of the edge ring (-1 inside)
    gate_sp: torch.Tensor  # [NYP, NXP] segment of each padded cell
    gate6_q95: torch.Tensor  # [6]
    gate6_q99: torch.Tensor  # [6]
    level_ratio: np.ndarray  # [12, 4] PM2.5 surface / lowest-level ratio by month and 6-h UTC bin
    level6: np.ndarray  # [6, 12, 4] same for every species (row 0 unused)
    dust_ratio: float
    eligible: np.ndarray  # [6, S] input network by species
    stations: pd.DataFrame  # station_id, longitude, latitude, y_km, x_km, held

    @property
    def held(self):
        return self.stations.held.to_numpy(bool)

    @property
    def station_km(self):
        return self.stations[["y_km", "x_km"]].to_numpy(np.float32)

    def covariance(self, species):
        from .oi import load_covariance
        return load_covariance(self.path / f"oi_{species}.pt")

    def to(self, device):
        for k, v in vars(self).items():
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        return self


def load_region(name_or_path):
    path = Path(name_or_path)
    if not path.exists():
        path = REGIONS / str(name_or_path)
    cfg = json.loads((path / "region.json").read_text(encoding="utf-8"))
    z = np.load(path / "constants.npz")
    t = lambda k: torch.from_numpy(np.asarray(z[k]))  # noqa: E731
    stations = pd.read_csv(path / "stations.csv", dtype={"station_id": str})
    return Region(name=path.name, path=path, config=cfg, utc_offset=int(cfg["utc_offset"]), rv=t("rv"), rw=t("rw"),
                  halo_grid=t("halo_grid"), outer_inner_mask=t("outer_inner_mask"), dy_km=float(z["dy_km"]), dx_km=t("dx_km"),
                  p4=t("p4"), gate_sr=t("gate_sr"), gate_sp=t("gate_sp"), gate6_q95=t("gate6_q95"), gate6_q99=t("gate6_q99"),
                  level_ratio=z["level_ratio"], level6=z["level6"], dust_ratio=float(z["dust_ratio"]), eligible=z["eligible"],
                  stations=stations)
