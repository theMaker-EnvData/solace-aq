"""Regional geometry: the 4-km inner grid in a projected coordinate system, the 0.25-degree outer grid, the fixed
interpolation from meteorological nodes (outer grid points) to inner cells, the boundary band and the gate segments."""
import numpy as np
import torch
from pyproj import CRS, Transformer
from scipy.spatial import Delaunay

from .constants import CELL_KM, HALO, NX, NY, OUTER_NX, OUTER_NY

KM_PER_DEGREE = 111.2


def outer_grid(bbox):
    """Outer grid (north to south, west to east) of bbox = (west, south, east, north)."""
    w, s, e, n = bbox
    return np.linspace(n, s, OUTER_NY), np.linspace(w, e, OUTER_NX)


def inner_centres(bounds):
    """Projected coordinates (m) of the inner cell centres, rows north to south."""
    xmin, _, _, ymax = bounds
    cell = CELL_KM * 1000
    return np.meshgrid(xmin + (np.arange(NX) + 0.5) * cell, ymax - (np.arange(NY) + 0.5) * cell)


def node_weights(bounds, crs, bbox):
    """Each inner cell as a barycentric combination of three outer grid points (Delaunay triangles in the projection)."""
    lat, lon = outer_grid(bbox)
    lon2, lat2 = np.meshgrid(lon, lat)
    mx, my = Transformer.from_crs(4326, crs, always_xy=True).transform(lon2, lat2)
    xx, yy = inner_centres(bounds)
    tri = Delaunay(np.c_[mx.ravel(), my.ravel()])
    q = np.c_[xx.ravel(), yy.ravel()]
    si = tri.find_simplex(q)
    if (si < 0).any():
        raise ValueError("inner grid extends beyond the outer grid")
    uv = np.einsum("nij,nj->ni", tri.transform[si, :2], q - tri.transform[si, 2])
    return tri.simplices[si].astype(np.int64), np.c_[uv, 1 - uv.sum(1)].astype(np.float32), mx, my


def halo_grid(bounds, crs, bbox):
    """grid_sample coordinates of the padded inner cells in the outer grid [NYP, NXP, 2]."""
    xmin, _, _, ymax = bounds
    w, s, e, n = bbox
    cell = CELL_KM * 1000
    rr, cc = np.meshgrid(np.arange(-HALO, NY + HALO), np.arange(-HALO, NX + HALO), indexing="ij")
    lo, la = Transformer.from_crs(crs, 4326, always_xy=True).transform(xmin + (cc + 0.5) * cell, ymax - (rr + 0.5) * cell)
    return np.stack([2 * (lo - w) / (e - w) - 1, 2 * (la - n) / (s - n) - 1], -1).astype(np.float32)


def gate_segments():
    """Segment (0-11, three per side) of the four-cell inner ring [NY, NX] (-1 elsewhere) and of every padded cell
    [NYP, NXP] (segment of the nearest inner boundary point)."""
    def seg(yc, xc):
        d = np.stack([yc, NY - 1 - yc, xc, NX - 1 - xc])  # top, bottom, left, right
        side = d.argmin(0)
        along = np.where(side < 2, xc / NX, yc / NY)
        return side * 3 + np.minimum((along * 3).astype(int), 2)

    yy, xx = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    ring = (yy < 4) | (yy >= NY - 4) | (xx < 4) | (xx >= NX - 4)
    sr = np.where(ring, seg(yy, xx), -1)
    py, px = np.meshgrid(np.arange(NY + 2 * HALO), np.arange(NX + 2 * HALO), indexing="ij")
    return sr, seg(np.clip(py - HALO, 0, NY - 1), np.clip(px - HALO, 0, NX - 1))


def station_positions(longitude, latitude, bounds, crs):
    """Station coordinates in km from the north-west corner (y down, x right) and fractional cell positions (row, col)."""
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(np.asarray(longitude), np.asarray(latitude))
    xmin, _, _, ymax = bounds
    x_km, y_km = (x - xmin) / 1000, (ymax - y) / 1000
    return np.c_[y_km, x_km].astype(np.float32), np.c_[(y_km - 2) / 4, (x_km - 2) / 4].astype(np.float32)


def build(bounds, crs, bbox):
    """Geometry constants of a region (dict of arrays)."""
    crs = CRS.from_user_input(crs)
    rv, rw, mx, my = node_weights(bounds, crs, bbox)
    xmin, ymin, xmax, ymax = bounds
    lat, lon = outer_grid(bbox)
    sr, sp = gate_segments()
    return {"rv": rv, "rw": rw, "halo_grid": halo_grid(bounds, crs, bbox),
            "outer_inner_mask": (mx >= xmin) & (mx <= xmax) & (my >= ymin) & (my <= ymax),
            "dy_km": np.float32(abs(lat[1] - lat[0]) * KM_PER_DEGREE),
            "dx_km": (abs(lon[1] - lon[0]) * KM_PER_DEGREE * np.cos(np.deg2rad(np.meshgrid(lon, lat)[1]))).astype(np.float32),
            "gate_sr": sr, "gate_sp": sp}


def as_tensors(geo):
    return {k: torch.as_tensor(v) for k, v in geo.items()}
