"""Pre-training of the surface transport and net change on PM2.5 alone (one layer, copy boundary), whose weights
initialise the transport and the net-change networks of every species."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import CELLS_PER_HOUR, MET_INDEX, NX, NY
from ..model import conv_mlp, sample_points, to_cells
from .schedule import EMA, pin, warmup_cosine

# Mean and spread of the eleven meteorological features over the Korean training period 2018-2022; kept fixed in every
# region so that the learned transport and net-change networks share one input scaling.
FEATURE_MEAN = (-0.1845199316740036, 0.7363400459289551, -0.8242803812026978, -1.4985933303833008, 0.7337563037872314,
                0.6516912579536438, 0.7211971879005432, 0.3570592999458313, 0.08371902257204056, 0.5727542638778687,
                0.49378523230552673)
FEATURE_STD = (1.0328117609024048, 0.16571839153766632, 0.6764652729034424, 0.7688255906105042, 0.1613403558731079,
               0.2343943566083908, 0.4355926513671875, 0.5057735443115234, 0.26996180415153503, 0.40600234270095825,
               0.3281630873680115)
LEADS = (1, 3, 6, 12, 24)
CONFIG = {"steps": 4000, "batch": 4, "warmup": 200, "lr": 2e-3, "ema": 0.999, "every": 250, "selection_windows": 96}


def curriculum(it, steps):
    f = it / steps
    return 3 if f < 0.15 else 6 if f < 0.35 else 12 if f < 0.6 else 24


class SurfaceStep(nn.Module):
    """One hour of surface PM2.5: advection by the blended wind, diffusion, net change. Same networks as Inner."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mu", torch.tensor(FEATURE_MEAN, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("sd", torch.tensor(FEATURE_STD, dtype=torch.float32).view(1, -1, 1, 1))
        self.blend = conv_mlp(3, 16, 1)
        self.kap = conv_mlp(2, 16, 1)
        self.k = conv_mlp(16, 32, 1)

    def forward(self, c, met, tf):
        g = lambda k: met[:, MET_INDEX[k]:MET_INDEX[k] + 1]  # noqa: E731
        raw = torch.cat([torch.log(g("pblh").clamp(min=20) / 500), g("ventilation_index") / 10, (g("t925") - g("t2m")) / 5,
                         (g("t850") - g("t2m")) / 5, g("rh2m") / 100, g("rh925") / 100, g("t2m") / 20, g("ssrd_1h") / 500,
                         torch.log1p(g("tp_1h").clamp(min=0)), g("tcc"), torch.sqrt(g("u10") ** 2 + g("v10") ** 2) / 10], 1)
        zf = (raw - self.mu) / self.sd
        lp, stab, ws10 = zf[:, 0:1], zf[:, 2:3], zf[:, 10:11]
        a = torch.sigmoid(self.blend(torch.cat([lp, stab, ws10], 1)))
        u = a * g("u10") + (1 - a) * (g("u925") + g("u850")) / 2
        v = a * g("v10") + (1 - a) * (g("v925") + g("v850")) / 2
        b = c.shape[0]
        yy = torch.arange(NY, device=c.device, dtype=torch.float32).view(1, NY, 1).expand(b, NY, NX)
        xx = torch.arange(NX, device=c.device, dtype=torch.float32).view(1, 1, NX).expand(b, NY, NX)
        grid = torch.stack([(xx - u[:, 0] * CELLS_PER_HOUR) / (NX - 1) * 2 - 1, (yy + v[:, 0] * CELLS_PER_HOUR) / (NY - 1) * 2 - 1], -1)
        c = F.grid_sample(c, grid, mode="bilinear", padding_mode="border", align_corners=True)
        kap = 0.25 * torch.sigmoid(self.kap(torch.cat([lp, ws10], 1)))
        cp = F.pad(c, (1, 1, 1, 1), mode="replicate")
        c = c + kap * (cp[..., :-2, 1:-1] + cp[..., 2:, 1:-1] + cp[..., 1:-1, :-2] + cp[..., 1:-1, 2:] - 4 * c)
        f = torch.cat([zf[:, :10], tf[:, :, None, None].expand(-1, -1, NY, NX), torch.log1p(c.clamp(min=0)) / 5], 1)
        return (c * torch.exp(0.3 * torch.tanh(self.k(f)))).clamp(min=0)


def rollout(model, c, met, tf, p4, rv, rw, hours):
    out = []
    for s in range(hours):
        c = model(c, to_cells(met[:, s], rv, rw), tf[:, s])
        out.append(sample_points(c, p4)[:, 0])
    return torch.stack(out, 1), c


class Selection:
    """Fixed selection windows of the selection year: OI analysis at t0 from the input stations, 24-h forecast, score =
    mean over leads of RMSE_h / E_h at the held-out stations."""

    def __init__(self, Y, tf, met, idx, stations_xy, held, cov, Eh, p4, rv, rw, n=96, batch=4):
        self.Y, self.tf, self.met, self.held, self.Eh, self.p4, self.rv, self.rw, self.batch = Y, tf, met, held, Eh, p4, rv, rw, batch
        self.idx = idx[::max(1, len(idx) // n)][:n]
        xs = torch.as_tensor(stations_xy, dtype=torch.float32)
        cells = torch.as_tensor(np.c_[np.repeat(np.arange(NY) * 4 + 2.0, NX), np.tile(np.arange(NX) * 4 + 2.0, NY)].astype(np.float32))
        with torch.no_grad():
            self.Kss, self.Kcs, self.r = cov.k(xs, xs), cov.k(cells, xs), float(torch.exp(cov.logr))

    def analysis(self, vals, valid):
        v0 = valid.float()
        y = torch.nan_to_num(vals)
        m = (y * v0).sum(1) / v0.sum(1).clamp(min=1)
        s = torch.sqrt((((y - m[:, None]) * v0) ** 2).sum(1) / v0.sum(1).clamp(min=1)) + 1.0
        z = (y - m[:, None]) / s[:, None] * v0
        K = self.Kss[None] * (v0[:, :, None] * v0[:, None, :]) + torch.diag_embed(self.r * v0 + (1 - v0) + 1e-5)
        a = torch.cholesky_solve(z.unsqueeze(-1), torch.linalg.cholesky(K)).squeeze(-1) * v0
        return (m.view(-1, 1, 1, 1) + s.view(-1, 1, 1, 1) * (a @ self.Kcs.T).view(-1, 1, NY, NX)).clamp(min=0)

    @torch.no_grad()
    def score(self, model):
        model.eval()
        keep = torch.as_tensor(~self.held)
        hj = np.where(self.held)[0]
        P, O = {h: [] for h in LEADS}, {h: [] for h in LEADS}
        for b in range(0, len(self.idx), self.batch):
            t0 = self.idx[b:b + self.batch]
            y0 = torch.as_tensor(self.Y[t0])
            c = self.analysis(y0, torch.isfinite(y0) & keep[None])
            met = torch.as_tensor(np.stack([np.asarray(self.met[t:t + 24], np.float32) for t in t0]))
            tf = torch.as_tensor(np.stack([self.tf[t:t + 24] for t in t0]))
            for s in range(24):
                c = model(c, to_cells(met[:, s], self.rv, self.rw), tf[:, s])
                if s + 1 in LEADS:
                    P[s + 1].append(sample_points(c, self.p4)[:, 0][:, hj].numpy())
                    O[s + 1].append(self.Y[t0 + s + 1][:, hj])
        model.train()
        return float(np.mean([rmse(np.concatenate(P[h]), np.concatenate(O[h])) / self.Eh[h - 1] for h in LEADS]))


def rmse(p, o):
    m = np.isfinite(p) & np.isfinite(o)
    return float(np.sqrt(((p[m] - o[m]) ** 2).mean()))


def train(years, selection, held, Eh, p4, rv, rw, steps=CONFIG["steps"], batch=CONFIG["batch"], seed=0, log=None, every=CONFIG["every"],
          pins=None):
    """years: list of windows.Year (training years). Returns the EMA weights with the best selection score.
    pins: {key: value} of weights held fixed (keys of the full model; the surface-transport ones apply here)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SurfaceStep()
    if pins:
        pin(model, {k[len("inner."):]: v for k, v in pins.items() if k.startswith(("inner.blend.", "inner.kap."))})
    ema = EMA(model, CONFIG["ema"])
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(CONFIG["warmup"], steps))
    hm = torch.as_tensor(held)
    w_lead = torch.as_tensor(1.0 / Eh ** 2, dtype=torch.float32)
    sizes = np.array([len(y.idx) for y in years])
    pr = sizes / sizes.sum()
    evaluator = SurfaceStep()
    best = (np.inf, None, -1)
    for it in range(steps):
        if it % every == 0 or it == steps - 1:
            evaluator.load_state_dict(ema.state())
            sc = selection.score(evaluator)
            if sc < best[0]:
                best = (sc, ema.state(), it)
            if log is not None:
                log(f"pretrain it {it} selection {sc:.4f} best_it {best[2]}")
        hours = curriculum(it, steps)
        src = years[rng.choice(len(years), p=pr)]
        ns = np.sort(rng.choice(len(src.idx), batch, replace=False))
        c, met, obs, tf = src.pm25_batch(ns, hours)
        pred, _ = rollout(model, c, met, tf, p4, rv, rw, hours)
        m = (torch.isfinite(obs) & ~hm[None, None]).float()
        mse_h = (((pred - torch.nan_to_num(obs)) ** 2) * m).sum((0, 2)) / m.sum((0, 2)).clamp(min=1)
        loss = (mse_h * w_lead[:hours]).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(model)
    return best
