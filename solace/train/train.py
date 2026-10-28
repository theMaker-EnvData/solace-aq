"""Training of the full model: six species, three layers, outer model, 48-h windows with a 6-h outer spin-up.

Loss = mean over species and leads of MSE_{h,s} / E_{h,s}^2 at the input stations (E: persistence RMSE of the training
years) + the outer model's log error outside the inner domain. Adam (networks 5e-4, scalar constants 1e-2), warm-up 100
steps then cosine decay, gradient clipping at 1, EMA 0.999. The EMA weights are scored every 250 steps on 300 windows of
the selection year at the held-out stations; the best score is kept."""
import copy

import numpy as np
import torch

from ..constants import HORIZON, SPECIES
from ..model import Solace
from .pretrain import rmse
from .schedule import EMA, pin, warmup_cosine

SELECTION_LEADS = (1, 3, 6, 12, 24)  # leads of the selection score (the loss uses every lead)
CONFIG = {"steps": 2000, "batch": 4, "warmup": 100, "lr_networks": 5e-4, "lr_constants": 1e-2, "ema": 0.999, "every": 250,
          "selection_windows": 300, "aux": 1.0}
GASES = ("o3", "no2", "so2", "co")  # start from transport only (zero net change)


def initial_model(pretrained, outer_mu, outer_sd):
    """Full model initialised from the PM2.5 pre-training: shared transport, and the PM2.5 net-change network copied to
    PM2.5 and PM10 (the gases start at zero net change)."""
    model = Solace()
    inner = model.inner
    with torch.no_grad():
        inner.mu.copy_(pretrained["mu"])
        inner.sd.copy_(pretrained["sd"])
    inner.blend.load_state_dict({k[6:]: v for k, v in pretrained.items() if k.startswith("blend.")})
    inner.kap.load_state_dict({k[4:]: v for k, v in pretrained.items() if k.startswith("kap.")})
    for sp, k in zip(SPECIES, inner.ks):
        k.load_state_dict({n[2:]: v for n, v in pretrained.items() if n.startswith("k.")})
        if sp in GASES:
            torch.nn.init.zeros_(k[-1].weight)
            torch.nn.init.zeros_(k[-1].bias)
    with torch.no_grad():
        model.outer.mu.copy_(outer_mu.view(1, -1, 1, 1))
        model.outer.sd.copy_(outer_sd.view(1, -1, 1, 1))
    return model


def optimizer(model):
    up = model.up
    constants = [p for n, p in up.named_parameters() if "blend1" not in n]
    networks = list(model.inner.parameters()) + list(model.outer.parameters()) + list(up.lay.blend1.parameters())
    return torch.optim.Adam([{"params": networks, "lr": CONFIG["lr_networks"]}, {"params": constants, "lr": CONFIG["lr_constants"]}])


def loss_fn(pred, obs, aux, held, w_lead):
    """pred, obs [B, H, 6, S]; w_lead [6, H]."""
    m = (torch.isfinite(obs) & ~held[None, None, None]).float()
    mse = (((pred - torch.nan_to_num(obs)) ** 2) * m).sum((0, 3)) / m.sum((0, 3)).clamp(min=1)  # [H, 6]
    return (mse.T * w_lead).mean() + CONFIG["aux"] * aux


class Selection:
    def __init__(self, year, held, Eh, n=CONFIG["selection_windows"], batch=4, horizon=HORIZON):
        ok = year.ok_windows(horizon)
        self.year, self.idx, self.held, self.Eh, self.batch, self.horizon = year, ok[::max(1, len(ok) // n)][:n], held, Eh, batch, horizon

    @torch.no_grad()
    def score(self, model, region):
        model.eval()
        P, O = {h: [] for h in SELECTION_LEADS}, {h: [] for h in SELECTION_LEADS}
        dev = region.p4.device
        for b in range(0, len(self.idx), self.batch):
            batch, obs = self.year.batch(self.idx[b:b + self.batch], self.horizon)
            p, _ = model({k: v.to(dev) for k, v in batch.items()}, region, self.horizon)
            p, ob = p.cpu().numpy(), obs.numpy()
            for h in SELECTION_LEADS:
                P[h].append(p[:, h - 1][:, :, self.held])
                O[h].append(ob[:, h - 1][:, :, self.held])
        model.train()
        P, O = {h: np.concatenate(v) for h, v in P.items()}, {h: np.concatenate(v) for h, v in O.items()}
        return float(np.mean([rmse(P[h][:, k], O[h][:, k]) / self.Eh[k, h - 1] for h in SELECTION_LEADS for k in range(len(SPECIES))]))


def train(model, years, selection, region, held, Eh, steps=CONFIG["steps"], batch=CONFIG["batch"], seed=0, log=None,
          every=CONFIG["every"], horizon=HORIZON, pins=None, frozen=(), points=None, targets=None, loss=None, final=None,
          device="cpu"):
    """years: training windows.Year list; Eh [6, 48]. Returns (score, EMA state, step) of the selected weights.
    pins: {key: value} of weights held fixed during training (Experiment A).
    Synthetic supervision (solace.synthetic): frozen, prefixes of weights not trained; points, the positions [P, 2] (cells)
    where the training rollout is scored; targets(inputs, obs) replaces the observations; loss(pred, targets, aux)
    replaces the loss. final: optional dict that receives the last EMA and raw weights. The returned and final weights are
    on the CPU."""
    rng = np.random.default_rng(seed)
    model.to(device)
    region = copy.copy(region).to(device)
    points = None if points is None else points.to(device)
    if pins:
        pin(model, pins)
    for n, p in model.named_parameters():
        if n.startswith(tuple(frozen)):
            p.requires_grad_(False)
    ema = EMA(model, CONFIG["ema"])
    opt = optimizer(model)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(CONFIG["warmup"], steps))
    evaluator = Solace().to(device)
    hm = torch.as_tensor(held, device=device)
    w_lead = torch.as_tensor(1.0 / Eh ** 2, dtype=torch.float32, device=device)
    sizes = np.array([len(y.idx) for y in years])
    pr = sizes / sizes.sum()
    best = (np.inf, None, -1)

    def draw():
        src = years[rng.choice(len(years), p=pr)]
        return src, np.sort(rng.choice(src.ok_windows(horizon), batch, replace=False))

    nxt = draw()
    for it in range(steps):
        if it % every == 0 or it == steps - 1:
            evaluator.load_state_dict(ema.state())
            sc = selection.score(evaluator, region)
            if sc < best[0]:
                best = (sc, {k: v.cpu() for k, v in ema.state().items()}, it)
            if log is not None:
                log(f"train it {it} selection {sc:.4f} best_it {best[2]}")
        src, ns = nxt
        nxt = draw()
        inputs, obs = src.batch(ns, horizon)
        inputs, obs = {k: v.to(device) for k, v in inputs.items()}, obs.to(device)
        if targets is not None:
            obs = targets(inputs, obs)
        pred, aux = model(inputs, region, horizon, points=points)
        value = loss(pred, obs, aux) if loss is not None else loss_fn(pred, obs, aux, hm, w_lead)
        opt.zero_grad()
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(model)
    if final is not None:
        final["ema"] = {k: v.cpu() for k, v in ema.state().items()}
        final["raw"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best
