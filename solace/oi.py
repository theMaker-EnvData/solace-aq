"""Initial fields from station observations: optimal interpolation with a learned isotropic covariance (two Matern-3/2
scales and a nugget), and the 30-day mean analysis that sets the spatial pattern of the emission term."""
import math

import numpy as np
import torch
import torch.nn as nn

from .constants import CELL_KM, NX, NY

S3 = math.sqrt(3.0)
COVERAGE = 0.75  # minimum fraction of the input network reporting for a valid analysis
WINDOW_HOURS = 30 * 24


class Covariance(nn.Module):
    """k(d) = w1 m(d / L1) + w2 m(d / L2), m(r) = (1 + sqrt3 r) exp(-sqrt3 r), nugget r (relative to unit variance)."""

    def __init__(self):
        super().__init__()
        self.logL = nn.Parameter(torch.log(torch.tensor([20.0, 120.0])))  # km, short and long
        self.logw = nn.Parameter(torch.log(torch.tensor([0.6, 0.4])))
        self.logr = nn.Parameter(torch.log(torch.tensor(0.1)))

    def k(self, xa, xb):
        """xa [P, 2], xb [Q, 2] (km) -> [P, Q]."""
        dr = xa[:, None, 0] - xb[None, :, 0]
        dc = xa[:, None, 1] - xb[None, :, 1]
        d = torch.sqrt(dr * dr + dc * dc + 1e-9)
        w = torch.exp(self.logw)
        rs, rl = d / torch.exp(self.logL[0]), d / torch.exp(self.logL[1])
        return w[0] * (1 + S3 * rs) * torch.exp(-S3 * rs) + w[1] * (1 + S3 * rl) * torch.exp(-S3 * rl)


def grid_points(device="cpu"):
    """Inner cell centres (km from the north-west corner, row then column) [NY * NX, 2]."""
    yy, xx = np.meshgrid(np.arange(NY) * CELL_KM + CELL_KM / 2, np.arange(NX) * CELL_KM + CELL_KM / 2, indexing="ij")
    return torch.tensor(np.c_[yy.ravel(), xx.ravel()], dtype=torch.float32, device=device)


def standardise(v):
    """Per-time mean and spread over the reporting stations; v [B, S] with NaN for missing."""
    valid = torch.isfinite(v)
    vm = valid.float()
    count = vm.sum(1, keepdim=True).clamp(min=1)
    y = torch.nan_to_num(v)
    mean = (y * vm).sum(1, keepdim=True) / count
    sd = torch.sqrt((((y - mean) * vm) ** 2).sum(1, keepdim=True) / count) + 1
    return (y - mean) / sd * vm, vm, mean, sd


@torch.no_grad()
def analyse(values, stations, cov, queries=None, chunk=8):
    """Analyses at the query points (default: every inner cell) [B, Q] from observations values [B, S] (NaN missing) at
    station coordinates [S, 2] km. Each time is standardised by its own station mean and spread."""
    queries = grid_points(stations.device) if queries is None else queries
    K = cov.k(stations, stations)
    Kq = cov.k(queries, stations)
    r = torch.exp(cov.logr)
    out = []
    for b in range(0, len(values), chunk):
        z, vm, mean, sd = standardise(values[b:b + chunk])
        A = K[None] * (vm[:, :, None] * vm[:, None, :]) + torch.diag_embed(r * vm + (1 - vm) + 1e-5)
        chol, info = torch.linalg.cholesky_ex(A)
        if bool((info > 0).any()):
            chol = torch.linalg.cholesky(A + 1e-3 * torch.eye(A.shape[-1], device=A.device))
        alpha = torch.cholesky_solve(z[..., None], chol)[..., 0]
        out.append((mean + sd * (Kq @ (alpha * vm).T).T).clamp(min=0))
    return torch.cat(out)


def analysis_fields(observations, stations, network, cov, chunk=8):
    """Gridded analyses [T, NY, NX] (float16 precision, NaN where fewer than 75 % of the input network report) from
    observations [T, S] of one species. network: bool [S], the input stations (eligible and not held out)."""
    values = observations[:, network]
    valid = np.isfinite(values).sum(1) >= COVERAGE * int(network.sum())
    fields = np.full((len(values), NY, NX), np.nan, np.float16)
    ids = np.flatnonzero(valid)
    if len(ids):
        pts = torch.as_tensor(stations[network], dtype=torch.float32)
        res = analyse(torch.as_tensor(values[ids], dtype=torch.float32), pts, cov, chunk=chunk)
        fields[ids] = res.numpy().reshape(-1, NY, NX).astype(np.float16)
    return fields, valid


def background(fields, times, t0):
    """Mean of the valid analyses with times in [t0 - 720 h, t0) (float16 precision); fields [T, NY, NX], times in hours."""
    times = np.asarray(times)
    choose = np.flatnonzero((times >= t0 - WINDOW_HOURS) & (times < t0))
    acc = np.zeros(fields.shape[1:], np.float64)
    n = 0
    for i in choose:
        f = np.asarray(fields[i], np.float32)
        if np.isfinite(f).all():
            acc += f
            n += 1
    return (acc / n).astype(np.float16) if n else np.full(fields.shape[1:], np.nan, np.float16)


def load_covariance(path):
    cov = Covariance()
    cov.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return cov
