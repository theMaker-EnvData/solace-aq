"""Fit the OI covariance of one species by leave-stations-out prediction at hourly station observations.

Each step draws `batch` hours. Inputs exclude the held-out stations, a rotating 5 % of the input stations (redrawn every
1000 steps) and a random 10 % per hour; the rotating and dropped stations are the targets. With `sparse`, each hour may
instead remove every input within 10, 25 or 50 km of a few target stations (training on sparser layouts)."""
import numpy as np
import torch

from ..oi import Covariance

SCALE = 30.0
SPARSE_RADII = (0, 10, 25, 50)
SPARSE_TARGETS = {10: 16, 25: 8, 50: 4}
CONFIG = {"pm25": {"steps": 5000, "batch": 16, "cosine": True, "sparse": True}}
DEFAULT = {"steps": 3000, "batch": 16, "cosine": False, "sparse": False}


def predict_at_stations(cov, xs, y, vin):
    vm = vin.float()
    n = vm.sum(1, keepdim=True).clamp(min=1)
    m = (y * vm).sum(1, keepdim=True) / n
    s = torch.sqrt((((y - m) * vm) ** 2).sum(1, keepdim=True) / n) + 1.0
    z = (y - m) / s * vm
    K = cov.k(xs, xs)[None].expand(y.shape[0], -1, -1)
    A = K * (vm[:, :, None] * vm[:, None, :]) + torch.diag_embed(torch.exp(cov.logr) * vm + (1 - vm) + 1e-5)
    chol, info = torch.linalg.cholesky_ex(A)
    if bool((info > 0).any()):
        chol = torch.linalg.cholesky(A + 1e-3 * torch.eye(A.shape[-1], device=A.device))
    alpha = torch.cholesky_solve(z.unsqueeze(-1), chol).squeeze(-1)
    return m + s * ((K * vm[:, None, :]) @ alpha.unsqueeze(-1)).squeeze(-1)


def sparse_masks(rng, valid, held, D):
    radius = int(rng.choice(SPARSE_RADII))
    if radius == 0:
        return None
    candidates = np.where(valid & ~held)[0]
    targets = rng.choice(candidates, min(SPARSE_TARGETS[radius], len(candidates)), replace=False)
    vin = valid & ~held & ~(D[targets] < radius).any(0)
    t = np.zeros_like(valid)
    t[targets] = True
    return vin, t


def fit(Y, xy, held, steps, batch, cosine=False, sparse=False, seed=0, log=None, every=500, device="cpu"):
    """Y [T, S] observations (NaN missing) of the stations at xy [S, 2] km; held bool [S]. Returns the fitted Covariance."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cov = Covariance().to(device)
    opt = torch.optim.Adam(cov.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=3e-5) if cosine else None
    xs = torch.as_tensor(xy, dtype=torch.float32, device=device)
    D = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1)) if sparse else None
    pool = np.where(~held)[0]
    hm = torch.as_tensor(held, device=device)
    losses = []
    for it in range(steps):
        if it % 1000 == 0:
            rot = np.zeros(len(held), bool)
            rot[rng.choice(pool, int(0.05 * len(pool)), replace=False)] = True
        idx = np.sort(rng.choice(len(Y), batch, replace=False))
        y = torch.as_tensor(np.nan_to_num(Y[idx]), device=device)
        valid = torch.as_tensor(np.isfinite(Y[idx]), device=device)
        drop = torch.as_tensor(rng.random((batch, len(held))) < 0.10, device=device)
        rm = torch.as_tensor(rot, device=device)
        vin = valid & ~hm & ~rm & ~drop
        tgt = valid & ~hm & (rm | drop)
        if sparse:
            vn, tn, va = vin.cpu().numpy(), tgt.cpu().numpy(), valid.cpu().numpy()
            for b in range(batch):
                mk = sparse_masks(rng, va[b], held, D)
                if mk is not None and mk[0].sum() >= 8:
                    vn[b], tn[b] = mk
            vin, tgt = torch.as_tensor(vn, device=device), torch.as_tensor(tn, device=device)
        pred = predict_at_stations(cov, xs, y, vin)
        if sparse:
            se = (((pred - y) / SCALE) ** 2) * tgt
            loss = (se.sum(1) / tgt.sum(1).clamp(min=1)).mean()
        else:
            loss = (((pred - y) / SCALE) ** 2)[tgt].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        losses.append(loss.item())
        if log is not None and (it % every == 0 or it == steps - 1):
            log(f"oi it {it} rmse {np.sqrt(np.mean(losses[-200:])) * SCALE:.2f} L {torch.exp(cov.logL).detach().cpu().numpy().round(1)} "
                f"w {torch.exp(cov.logw).detach().cpu().numpy().round(2)} r {torch.exp(cov.logr).item():.3f}")
    return cov.cpu()
