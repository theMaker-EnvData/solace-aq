"""The SOLACE forecast model: an hourly step of transport, net change, emission and layer exchange on a 4-km grid of
three layers (surface, residual, upper), with an outer model that supplies the boundary.

    c_bar = T(c; B)                                      semi-Lagrangian advection with the boundary band B, then diffusion
    c'    = X( c_bar * exp(r(w, tau, c_bar)) + alpha M P(tau) g(h) )

r = 0.3 tanh f is the learned hourly log net-change rate of each species, alpha M P g the observation-based emission
(M: 30-day mean analysis, P: learned profile by local hour and day type, g: mixing-height dilution), X the exchange
between the surface, residual and upper layers. The outer model advances six species in three layers on the 0.25-degree
outer grid from meteorology alone.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from .constants import (CELLS_PER_HOUR, EXTRA_INDEX, HALO, LAYERS12, MET_INDEX, NX, NXP, NY, NYP, OUTER_AUX, OUTER_INDEX,
                        OUTER_MET, SPECIES, SURFACE6)
from .swin import GlobalAttentionBlock, SwinBlock

N_SP = len(SPECIES)
RATIO_EPS = 2.0


def conv_mlp(i, h, o):
    """Pointwise MLP (1x1 convolutions) with a zero-initialised output layer."""
    net = nn.Sequential(nn.Conv2d(i, h, 1), nn.GELU(), nn.Conv2d(h, h, 1), nn.GELU(), nn.Conv2d(h, o, 1))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


# --------------------------------------------------------------------------------------------------- grid operations
def to_cells(nodes, rv, rw):
    """Values at meteorological nodes [..., N] -> inner cells [..., NY, NX] with fixed triangle weights."""
    x = nodes[..., rv] * rw
    return x.sum(-1).reshape(*nodes.shape[:-1], NY, NX)


def to_halo(o, grid):
    """Outer field [B, C, 61, 81] -> padded inner grid [B, C, NYP, NXP] (bilinear)."""
    return F.grid_sample(o, grid[None].expand(o.shape[0], -1, -1, -1), mode="bilinear", padding_mode="border",
                         align_corners=True)


def sample_points(c, p4):
    """Fields [B, C, NY, NX] at fractional cell positions p4 [S, 2] (row, column) -> [B, C, S]."""
    g = torch.stack([p4[:, 1] / (NX - 1) * 2 - 1, p4[:, 0] / (NY - 1) * 2 - 1], -1)
    g = g.view(1, 1, -1, 2).expand(c.shape[0], -1, -1, -1)
    return F.grid_sample(c, g, mode="bilinear", padding_mode="border", align_corners=True)[:, :, 0]


def _departure_grid(u, v, batch, device):
    yy = torch.arange(NY, device=device, dtype=torch.float32).view(1, NY, 1).expand(batch, NY, NX) + HALO
    xx = torch.arange(NX, device=device, dtype=torch.float32).view(1, 1, NX).expand(batch, NY, NX) + HALO
    return torch.stack([(xx - u[:, 0] * CELLS_PER_HOUR) / (NXP - 1) * 2 - 1,
                        (yy + v[:, 0] * CELLS_PER_HOUR) / (NYP - 1) * 2 - 1], -1)


def _laplacian(c):
    cp = F.pad(c, (1, 1, 1, 1), mode="replicate")
    return cp[..., :-2, 1:-1] + cp[..., 2:, 1:-1] + cp[..., 1:-1, :-2] + cp[..., 1:-1, 2:] - 4 * c


def advect(c, u, v, kappa, halo):
    """Semi-Lagrangian advection of c [B, C, NY, NX] by u, v [B, 1, NY, NX] (m s-1) for one hour, reading the boundary
    band halo [B, C, NYP, NXP] where departure points leave the domain, followed by diffusion with coefficient kappa."""
    padded = halo.clone()
    padded[:, :, HALO:HALO + NY, HALO:HALO + NX] = c
    c = F.grid_sample(padded, _departure_grid(u, v, c.shape[0], c.device), mode="bilinear", padding_mode="border",
                      align_corners=True)
    return c + kappa * _laplacian(c)


# --------------------------------------------------------------------------------------------------- inner processes
class Inner(nn.Module):
    """Surface transport (learned wind blending and diffusion) and the per-species net change."""

    def __init__(self, n_features=11):
        super().__init__()
        self.register_buffer("mu", torch.zeros(1, n_features, 1, 1))
        self.register_buffer("sd", torch.ones(1, n_features, 1, 1))
        self.blend = conv_mlp(3, 16, 1)  # surface wind: weight of the 10-m wind against the 925-850-hPa mean
        self.kap = conv_mlp(2, 16, 1)  # diffusion coefficient
        self.ks = nn.ModuleList([conv_mlp(10 + 5 + 1, 32, 1) for _ in SPECIES])  # met (10), time (5), own concentration

    def features(self, met):
        g = lambda k: met[:, MET_INDEX[k]:MET_INDEX[k] + 1]  # noqa: E731
        raw = torch.cat([torch.log(g("pblh").clamp(min=20) / 500), g("ventilation_index") / 10, (g("t925") - g("t2m")) / 5,
                         (g("t850") - g("t2m")) / 5, g("rh2m") / 100, g("rh925") / 100, g("t2m") / 20, g("ssrd_1h") / 500,
                         torch.log1p(g("tp_1h").clamp(min=0)), g("tcc"), torch.sqrt(g("u10") ** 2 + g("v10") ** 2) / 10], 1)
        return (raw - self.mu) / self.sd

    def surface_wind(self, met, z):
        g = lambda k: met[:, MET_INDEX[k]:MET_INDEX[k] + 1]  # noqa: E731
        lp, stab, ws10 = z[:, 0:1], z[:, 2:3], z[:, 10:11]
        a = torch.sigmoid(self.blend(torch.cat([lp, stab, ws10], 1)))
        u = a * g("u10") + (1 - a) * (g("u925") + g("u850")) / 2
        v = a * g("v10") + (1 - a) * (g("v925") + g("v850")) / 2
        return u, v

    def diffusion(self, z):
        return 0.25 * torch.sigmoid(self.kap(torch.cat([z[:, 0:1], z[:, 10:11]], 1)))

    def forward(self, c, met, tf, halo, off=()):
        """c [B, 6, NY, NX] surface state; met [B, 16, NY, NX]; tf [B, 5] time features; halo [B, 6, NYP, NXP]."""
        z = self.features(met)
        if "advection" not in off:
            u, v = self.surface_wind(met, z)
            c = advect(c, u, v, self.diffusion(z), halo)
        if "net_change" in off:
            return c.clamp(min=0)
        base = torch.cat([z[:, :10], tf[:, :, None, None].expand(-1, -1, NY, NX)], 1)
        lc = torch.log1p(c.clamp(min=0)) / 5
        out = [c[:, s:s + 1] * torch.exp(0.3 * torch.tanh(k(torch.cat([base, lc[:, s:s + 1]], 1))))
               for s, k in enumerate(self.ks)]
        return torch.cat(out, 1).clamp(min=0)


class Emission(nn.Module):
    """Observation-based emission alpha_s M_s(x) P_s(local hour, day type) min(500 m / h, 10)."""

    def __init__(self):
        super().__init__()
        self.pm = nn.Parameter(torch.full((N_SP,), -6.0))  # alpha_s = 0.3 sigmoid(pm_s)
        self.q = nn.Parameter(torch.zeros(N_SP, 2, 24))  # log profile by weekday/weekend and local hour

    def strength(self):
        return 0.3 * torch.sigmoid(self.pm)

    def forward(self, met, tf, field, utc_offset):
        """met [B, 16, NY, NX]; tf [B, 5] with UTC hour on the circle and the local weekend flag; field M [B, 6, NY, NX]."""
        h = met[:, MET_INDEX["pblh"]:MET_INDEX["pblh"] + 1].clamp(min=50)
        hfac = (500.0 / h).clamp(max=10.0)
        hour = torch.round(torch.atan2(tf[:, 0], tf[:, 1]) * 24 / (2 * np.pi)).long() % 24
        local = (hour + utc_offset) % 24
        weekend = (tf[:, 4] > 0.5).long()
        profile = torch.exp(self.q[:, weekend, local]).T  # [B, 6]
        flux = self.strength()[None, :, None, None] * field * profile[:, :, None, None]
        return flux * hfac


class Layers(nn.Module):
    """Exchange constants between the layers, the residual-layer wind and the initial scaling of the upper layers."""

    BOUNDS = {"a_e": (0, 1), "a_d": (0, 1), "a_12": (0, 0.2), "a_w": (0, 2)}
    START = {"a_e": 0.02, "a_d": 0.9, "a_12": 0.01, "a_w": 0.1}

    def __init__(self):
        super().__init__()
        for k, v in self.START.items():
            lo, hi = self.BOUNDS[k]
            self.register_parameter(k, nn.Parameter(torch.logit(torch.tensor((v - lo) / (hi - lo)))))
        z = float(torch.logit(torch.tensor((1.0 - 0.2) / 1.8)))
        self.g1s = nn.Parameter(torch.full((N_SP,), z))  # residual / surface ratio at t0, 0.2-2
        self.g2s = nn.Parameter(torch.full((N_SP,), z))  # upper / surface ratio at t0, 0.2-2
        self.blend1 = conv_mlp(2, 16, 1)  # residual-layer wind: weight of 925 hPa against 850 hPa

    def val(self, k):
        if k in ("g1", "g2"):
            return (0.2 + 1.8 * torch.sigmoid(getattr(self, k + "s")))[None, :, None, None]
        lo, hi = self.BOUNDS[k]
        return lo + (hi - lo) * torch.sigmoid(getattr(self, k))


class Upper(nn.Module):
    """Layer constants, emission and the gains of the residual / upper boundary from the outer model."""

    def __init__(self):
        super().__init__()
        self.lay = Layers()
        self.emis = Emission()
        self.lgb = nn.Parameter(torch.full((2, N_SP), float(torch.logit(torch.tensor((1.0 - 0.2) / 2.8)))))

    def gb(self):
        return 0.2 + 2.8 * torch.sigmoid(self.lgb)  # [2, 6]


class Outer(nn.Module):
    """Outer model: six species in three layers on the outer grid, each layer advected with its own wind, then a
    multiplicative change from a window/global-attention trunk."""

    def __init__(self, dim=48, heads=4):
        super().__init__()
        n_aux = len(OUTER_MET) + len(OUTER_AUX)
        self.register_buffer("mu", torch.zeros(1, n_aux, 1, 1))
        self.register_buffer("sd", torch.ones(1, n_aux, 1, 1))
        self.stem = nn.Sequential(nn.Conv2d(18 + n_aux + 5, dim, 3, padding=1), nn.GELU())
        self.blocks = nn.ModuleList([SwinBlock(dim, 8, heads, 0), GlobalAttentionBlock(dim, heads, downsample_factor=4),
                                     SwinBlock(dim, 8, heads, 4), GlobalAttentionBlock(dim, heads, downsample_factor=4)])
        self.head = nn.Conv2d(dim, 18, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.wblend = nn.Parameter(torch.tensor(0.0))  # surface wind: weight of 10 m against 850 hPa

    def forward(self, o, met, aux, tf, dy_km, dx_km):
        """o [B, 18, 61, 81] (surface, residual, upper x 6 species); met [B, 12, 61, 81]; aux [B, 5, 61, 81]; tf [B, 5]."""
        g = lambda k: met[:, OUTER_MET.index(k):OUTER_MET.index(k) + 1]  # noqa: E731
        a = torch.sigmoid(self.wblend)
        b, _, h, w = o.shape
        yy = torch.arange(h, device=o.device, dtype=torch.float32).view(1, h, 1).expand(b, h, w)
        xx = torch.arange(w, device=o.device, dtype=torch.float32).view(1, 1, w).expand(b, h, w)

        def adv(x, u, v):
            ry = yy + v[:, 0] * 3.6 / dy_km
            rx = xx - u[:, 0] * 3.6 / dx_km[None]
            grid = torch.stack([rx / (w - 1) * 2 - 1, ry / (h - 1) * 2 - 1], -1)
            return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)

        o = torch.cat([adv(o[:, :6], a * g("u10") + (1 - a) * g("u850"), a * g("v10") + (1 - a) * g("v850")),
                       adv(o[:, 6:12], g("u850"), g("v850")),
                       adv(o[:, 12:], (g("u850") + g("u700")) / 2, (g("v850") + g("v700")) / 2)], 1)
        x = torch.cat([torch.log1p(o.clamp(min=0)) / 5, (torch.cat([met, aux], 1) - self.mu) / self.sd,
                       tf[:, :, None, None].expand(-1, -1, h, w)], 1)
        y = self.stem(x)
        for blk in self.blocks:
            y = blk(y)
        return (o * torch.exp(0.3 * torch.tanh(self.head(y)))).clamp(min=0)


# --------------------------------------------------------------------------------------------------- the model
class Solace(nn.Module):
    """switches_off: components removed from a trained model for sensitivity runs: "advection" (surface advection and
    diffusion), "net_change" (rate 1), "emission", "exchange" (no exchange between layers), "boundary_gate" (copy boundary
    at the surface, the outer model never used at the inner edge)."""
    SWITCHES = ("advection", "net_change", "emission", "exchange", "boundary_gate")

    def __init__(self):
        super().__init__()
        self.inner = Inner()
        self.outer = Outer()
        self.up = Upper()
        self.switches_off = frozenset()

    # ------------------------------------------------------------------ boundary gate
    @staticmethod
    def gate(c, outer_surface, region):
        """Weight of the outer model in the surface boundary band [B, 6, NYP, NXP], set at t0: 1 where the outer field
        agrees with the analysis on the inner edge, 0 where it disagrees strongly (quantiles from the training years)."""
        sr = region.gate_sr
        ring = [torch.nonzero((sr == k).flatten()).squeeze(1) for k in range(12)]
        ws = []
        for k in range(N_SP):
            ref = to_halo(outer_surface[:, k:k + 1], region.halo_grid) + RATIO_EPS
            inner_ref = ref[:, :, HALO:HALO + NY, HALO:HALO + NX] - RATIO_EPS
            cf, af = c[:, k:k + 1][:, 0].flatten(1), inner_ref[:, 0].flatten(1)
            q = torch.stack([torch.log((cf[:, i].mean(1) + 2) / (af[:, i].mean(1) + 2)).abs() for i in ring], 1)
            w = ((region.gate6_q99[k] - q) / (region.gate6_q99[k] - region.gate6_q95[k])).clamp(0, 1)
            ws.append(w[:, region.gate_sp])
        return torch.stack(ws, 1)

    # ------------------------------------------------------------------ one hour
    def step(self, c0, c1, c2, metc, extc, h_prev, halo0, h1, h2, tf, field, utc_offset):
        lay = self.up.lay
        g = lambda k: metc[:, MET_INDEX[k]:MET_INDEX[k] + 1]  # noqa: E731
        x = lambda k: extc[:, EXTRA_INDEX[k]:EXTRA_INDEX[k] + 1]  # noqa: E731
        h = g("pblh").clamp(min=50)
        off = self.switches_off
        c0 = self.inner(c0, metc, tf, halo0, off)
        if "emission" not in off:
            c0 = c0 + self.up.emis(metc, tf, field, utc_offset)
        stab = (g("t925") - g("t2m")) / 5
        ws = torch.sqrt(g("u925") ** 2 + g("v925") ** 2) / 10
        b = torch.sigmoid(lay.blend1(torch.cat([stab, ws], 1)))
        c1 = advect(c1, b * g("u925") + (1 - b) * g("u850"), b * g("v925") + (1 - b) * g("v850"), 0.05, h1)
        c2 = advect(c2, (g("u850") + x("u700")) / 2, (g("v850") + x("v700")) / 2, 0.05, h2)
        grow = torch.relu(1 - h_prev / h)
        fall = torch.relu(1 - h / h_prev)
        if "exchange" in off:
            grow = fall = grow * 0
        c0n = c0 + lay.val("a_e") * grow * (c1 - c0)
        c1n = c1 + lay.val("a_d") * fall * (c0 - c1)
        r = (lay.val("a_12") + lay.val("a_w") * torch.relu(x("w850"))).clamp(max=0.3) * (0.0 if "exchange" in off else 1.0)
        c1n, c2 = c1n + r * (c2 - c1n), c2 + r * (c1n - c2)
        return c0n.clamp(min=0), c1n.clamp(min=0), c2.clamp(min=0), h

    # ------------------------------------------------------------------ forecast
    def forward(self, batch, region, horizon=48, fields=None, points=None):
        """batch: c [B,6,NY,NX] t0 analysis; met [B,H,16,N]; ext [B,H,4,N]; o [B,K+H+1,34,61,81] outer inputs from t0-K;
        gus [B,6,2,61,81] residual/upper layers at t0-K; tf [B,K+H,5]; last_m [B,6,NY,NX] emission pattern.
        Returns station forecasts [B,H,6,S] and the auxiliary outer loss. fields: optional list receiving the surface
        field of every hour; points: positions [P, 2] (cells) to forecast instead of the stations."""
        c, met, ext, o, gus, tf = batch["c"], batch["met"], batch["ext"], batch["o"], batch["gus"], batch["tf"]
        field = batch["last_m"]
        grid, inner_mask, dy_km, dx_km = region.halo_grid, region.outer_inner_mask, region.dy_km, region.dx_km
        rv, rw, offset = region.rv, region.rw, region.utc_offset
        p4 = region.p4 if points is None else points
        spin = tf.shape[1] - horizon
        state_ch = torch.as_tensor(list(SURFACE6) + list(LAYERS12), device=c.device)
        met_ch = torch.as_tensor([OUTER_INDEX[k] for k in OUTER_MET], device=c.device)
        aux_ch = torch.as_tensor([OUTER_INDEX[k] for k in OUTER_AUX], device=c.device)
        outside = torch.nonzero((~inner_mask).flatten()).squeeze(1)
        O = torch.cat([o[:, 0, list(SURFACE6)], gus[:, :, 0], gus[:, :, 1]], 1)
        aux0 = o[:, 0, aux_ch]
        aux_loss = []

        def run(fn, *args):  # recompute each hour in the backward pass when training (memory)
            return cp.checkpoint(fn, *args, use_reentrant=False) if torch.is_grad_enabled() else fn(*args)

        def outer_step(O, s):
            O = run(self.outer, O, o[:, s, met_ch], aux0, tf[:, s], dy_km, dx_km)
            aux_loss.append(((torch.log1p(O) - torch.log1p(o[:, s + 1, state_ch])) ** 2).flatten(2)[:, :, outside].mean())
            return O

        for s in range(spin):  # the outer model alone carries the outer state from t0 - K to t0
            O = outer_step(O, s)
        wmap = self.gate(c, O[:, :N_SP], region) * (0.0 if "boundary_gate" in self.switches_off else 1.0)
        lay = self.up.lay
        c1, c2 = c * lay.val("g1"), c * lay.val("g2")
        h_prev = to_cells(met[:, 0], rv, rw)[:, MET_INDEX["pblh"]:MET_INDEX["pblh"] + 1].clamp(min=50)
        c0 = c
        preds = []
        for s in range(horizon):
            metc, extc = to_cells(met[:, s], rv, rw), to_cells(ext[:, s], rv, rw)
            rep0 = F.pad(c0, (HALO, HALO, HALO, HALO), mode="replicate")
            gb = self.up.gb()
            U1, U2 = O[:, 6:12] * gb[0][None, :, None, None], O[:, 12:18] * gb[1][None, :, None, None]
            halo0 = wmap * to_halo(O[:, :N_SP], grid) + (1 - wmap) * rep0
            c0, c1, c2, h_prev = run(self.step, c0, c1, c2, metc, extc, h_prev, halo0, to_halo(U1, grid), to_halo(U2, grid),
                                     tf[:, spin + s], field, offset)
            if fields is not None:
                fields.append(c0)
            preds.append(sample_points(c0, p4))
            O = outer_step(O, spin + s)
        return torch.stack(preds, 1), torch.stack(aux_loss).mean()


def load_weights(path, device="cpu"):
    """Model with weights from a release file or from an original training checkpoint (unused entries are dropped)."""
    state = torch.load(path, map_location=device, weights_only=True)
    model = Solace().to(device)
    unused = {"up.lam", "up.lgam", "up.lay.g1", "up.lay.g2", "up.emis.stat", "up.emis.cbar"}
    model.load_state_dict({k: v for k, v in state.items() if k not in unused}, strict=True)
    return model.eval()
