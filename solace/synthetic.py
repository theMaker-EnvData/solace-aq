"""Synthetic supervision: a model trained on the forecasts of a known teacher instead of observations.

Teacher: the Taiwan model with its surface and residual-layer wind-blending functions taken from the Korea model. The
student starts from a pre-training on real data (another random seed); the outer model and the diffusion are fixed to the
teacher's and the outer model's auxiliary loss is dropped. Arms differ only in where the student is supervised:
  S  the input stations, with their real missing-value pattern
  D  0.5 x the station loss + 0.5 x the loss on an 8-km land lattice of the inner grid (every second cell)

Separability (no training): how many error units apart two forecasts are for a given network, under an error model fitted
to real observation-minus-forecast residuals (separable spatial x temporal covariance)."""
import numpy as np
import pandas as pd
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist

from .constants import NX, NY, SPECIES
from .model import Solace

FROM_SECOND = ("inner.blend.", "up.lay.blend1.")  # taken from the second model in the teacher
FROZEN = ("outer.", "inner.kap.")


def teacher_state(base, second):
    return {k: (second[k].clone() if k.startswith(FROM_SECOND) else v) for k, v in base.items()}


def land_sample(water, offset=0, step=2):
    """Positions (row, column) of every `step`-th cell over land (water fraction < 0.5)."""
    yy, xx = np.meshgrid(np.arange(offset, NY, step), np.arange(offset, NX, step), indexing="ij")
    keep = water[yy, xx] < 0.5
    return np.c_[yy[keep], xx[keep]].astype(np.float32)


def supervision(region, teacher, arm, held, w_lead, water, device="cpu"):
    """(points, targets, loss) for solace.train.train (same device)."""
    import copy
    S = len(held)
    region = copy.copy(region).to(device)
    grid = torch.as_tensor(land_sample(water), device=device) if arm == "D" else torch.zeros(0, 2, device=device)
    points = torch.cat([region.p4, grid], 0)
    model = Solace().eval().to(device)
    model.load_state_dict(teacher)
    for p in model.parameters():
        p.requires_grad_(False)
    hm = torch.as_tensor(held, device=device)
    w_lead = w_lead.to(device)

    def targets(inputs, obs):
        with torch.no_grad():
            out, _ = model({k: v.clone() for k, v in inputs.items()}, region, obs.shape[1], points=points)
        st = torch.where(torch.isfinite(obs), out[..., :S].to(obs.dtype), torch.full_like(obs, float("nan")))
        return torch.cat([st, out[..., S:].to(obs.dtype)], -1) if arm == "D" else st

    def part(pred, obs, mask):
        mse = (((pred - torch.nan_to_num(obs)) ** 2) * mask).sum((0, 3)) / mask.sum((0, 3)).clamp(min=1)
        return mse, (mse.T * w_lead).mean()

    def loss(pred, obs, aux):
        mS = (torch.isfinite(obs[..., :S]) & ~hm[None, None, None]).float()
        _, lossS = part(pred[..., :S], obs[..., :S], mS)
        if arm != "D":
            return lossS
        valid = (mS.sum((0, 3)) > 0).float()
        mseG, _ = part(pred[..., S:], obs[..., S:], torch.isfinite(obs[..., S:]).float())
        return 0.5 * lossS + 0.5 * ((mseG * valid).T * w_lead).mean()

    return points, targets, loss


# ------------------------------------------------------------------------------------------------- separability
PAIRS = (("teacher", "seed0"), ("teacher", "seed0_blend1"), ("seed0", "seed1"))
SIZES, DRAWS, LATTICE = (4, 8, 15, 30), 20, (8, 4, 2, 1)
LEADS = np.arange(2, 49, 2)


def error_model(truth, forecast, held, xy):
    """Per species: residual spread by lead window, temporal e-folding time tau (h), spatial e-folding length L (km) and
    nugget, from residuals truth - forecast [windows, 48, 6, stations] at the input stations."""
    R = truth[..., ~held] - forecast[..., ~held]
    dist = cdist(xy[~held], xy[~held])
    rows = []
    for si, sp in enumerate(SPECIES):
        r = R[:, :, si].astype("f8")
        sig = [np.nanstd(r[:, :24]), np.nanstd(r[:, 24:])]
        rs = np.concatenate([r[:, :24] / sig[0], r[:, 24:] / sig[1]], 1)
        rs = rs - np.nanmean(rs, axis=(0, 1), keepdims=True)
        lags = np.arange(2, 25, 2)
        ac = []
        for lag in lags:
            a, b = rs[:, :-lag], rs[:, lag:]
            ok = np.isfinite(a) & np.isfinite(b)
            ac.append(np.sum(np.where(ok, a * b, 0)) / np.sqrt(np.sum(np.where(ok, a * a, 0)) * np.sum(np.where(ok, b * b, 0))))
        tau = curve_fit(lambda x, t: np.exp(-x / t), lags, np.array(ac), p0=[10], bounds=(.1, 500))[0][0]
        flat = rs.reshape(-1, rs.shape[-1])
        n = flat.shape[1]
        need = min(200, flat.shape[0] // 2)
        cd, cc = [], []
        for i in range(n):
            for j in range(i + 1, n):
                ok = np.isfinite(flat[:, i]) & np.isfinite(flat[:, j])
                if ok.sum() > need:
                    cd.append(dist[i, j])
                    cc.append(np.corrcoef(flat[ok, i], flat[ok, j])[0, 1])
        cd, cc = np.array(cd), np.array(cc)
        (L, nug), _ = curve_fit(lambda d, L, g: (1 - g) * np.exp(-d / L), cd, cc, p0=[50, .3], bounds=([1, 0], [2000, .99]))
        rows.append(dict(species=sp, sigma_today=sig[0], sigma_tomorrow=sig[1], tau_h=tau, L_km=L, nugget=nug,
                         pair_corr_lt10km=float(cc[cd < 10].mean()) if (cd < 10).any() else np.nan, pairs=len(cc)))
    return pd.DataFrame(rows).set_index("species")


def networks(station_xy_inputs, land_rc, rng, cell_km=4.0):
    """Station subsets (nested random draws), land cells drawn at random with the same counts, and land lattices."""
    xy_land = np.c_[land_rc[:, 1], -land_rc[:, 0]] * cell_km
    n_st = len(station_xy_inputs)
    nets = [("stations", n_st, 0, "st", np.arange(n_st))]
    for d in range(DRAWS):
        perm = rng.permutation(n_st)
        for n in SIZES:
            nets.append(("stations", n, d, "st", perm[:n]))
        permg = rng.permutation(len(land_rc))
        for n in SIZES + (n_st,):
            nets.append(("land random", n, d, "land", permg[:n]))
    for step in LATTICE:
        keep = np.nonzero((land_rc[:, 0] % step == 0) & (land_rc[:, 1] % step == 0))[0]
        nets.append((f"land lattice {4 * step} km", len(keep), 0, "land", keep))
    return nets, xy_land


def separation(em, nets, station, land, models, xy_st, xy_land, days):
    """Squared Mahalanobis separation Lambda of each model pair, network, species and window.
    station [windows, model, T, 6, input stations], land [windows, model, T, 6, land cells] at LEADS."""
    win = (LEADS > 24).astype(int)
    rows = []
    for si, sp in enumerate(SPECIES):
        e = em.loc[sp]
        sig = np.where(win == 0, e.sigma_today, e.sigma_tomorrow)
        Ct = cho_factor(np.exp(-np.abs(LEADS[:, None] - LEADS[None]) / e.tau_h))
        for kind, n, draw, src, idx in nets:
            xy = (xy_st if src == "st" else xy_land)[idx]
            d = cdist(xy, xy)
            Cs = cho_factor((1 - e.nugget) * np.exp(-d / e.L_km) + e.nugget * np.eye(len(idx)) + 1e-9 * np.eye(len(idx)))
            F = station if src == "st" else land
            for a, b in PAIRS:
                Dm = (F[:, models.index(a), :, si][..., idx] - F[:, models.index(b), :, si][..., idx]) / sig[None, :, None]
                lam = [float(np.sum(cho_solve(Cs, Dw.T) * cho_solve(Ct, Dw).T)) for Dw in Dm]
                for day, value in zip(days, lam):
                    rows.append(dict(network=kind, points=n, draw=draw, pair=f"{a}-{b}", species=sp, day=day, lam=value,
                                     snr_point=float(np.sqrt((Dm ** 2).mean()))))
    return pd.DataFrame(rows)


@torch.no_grad()
def forecasts(region, batch, states, land_rc, leads=tuple(LEADS)):
    """Forecasts of several weight sets for the separability analysis: at all stations [model, 48, 6, S] and on land
    cells [model, len(leads), 6, L] (surface fields kept in float32)."""
    model = Solace().eval()
    st, ld = [], []
    for state in states.values():
        model.load_state_dict(state)
        fields = []
        p, _ = model(batch, region, fields=fields)
        st.append(p[0].numpy().astype(np.float32))
        ld.append(np.stack([fields[h - 1][0].numpy() for h in leads])[..., land_rc[:, 0], land_rc[:, 1]].astype(np.float32))
    return np.stack(st), np.stack(ld)


# ------------------------------------------------------------------------------------------------- recovery diagnostics
@torch.no_grad()
def diagnose_case(region, batch, states, water, case):
    """One window: distances of every model's wind-blend functions to the teacher's on the teacher's own hourly inputs,
    and the RMSE of its free forecast to the teacher's (stations and an 8-km land sample offset by one cell from the D
    training sample), with the teacher's persistence error on the same points. states must start with 'teacher'."""
    from .constants import MET_INDEX
    gy, gx = (lambda p: (p[:, 0].astype(int), p[:, 1].astype(int)))(land_sample(water, offset=1))
    land = torch.as_tensor(water < .5)
    rec = {"blend": [], "blend1": [], "met": []}
    model = Solace().eval()
    out, func = {}, {}
    for name, st in states.items():
        model.load_state_dict(st)
        hooks = []
        if name == "teacher":
            hooks = [model.inner.blend.register_forward_pre_hook(lambda _, a: rec["blend"].append(a[0].clone())),
                     model.up.lay.blend1.register_forward_pre_hook(lambda _, a: rec["blend1"].append(a[0].clone())),
                     model.inner.register_forward_pre_hook(lambda _, a: rec["met"].append(a[1].clone()))]
        fields = []
        try:
            p, _ = model(batch, region, fields=fields)
        finally:
            for h in hooks:
                h.remove()
        f = np.stack([x[0].numpy() for x in fields])[:, :6][..., gy, gx].astype("f8")
        out[name] = (p[0].numpy().astype("f8"), f)
    xb, x1, mt = rec["blend"][-48:], rec["blend1"][-48:], rec["met"][-48:]
    g = lambda m, k: m[:, MET_INDEX[k]]  # noqa: E731
    Ds = torch.stack([torch.hypot(g(m, "u10") - (g(m, "u925") + g(m, "u850")) / 2, g(m, "v10") - (g(m, "v925") + g(m, "v850")) / 2) for m in mt])
    Dr = torch.stack([torch.hypot(g(m, "u925") - g(m, "u850"), g(m, "v925") - g(m, "v850")) for m in mt])
    for name, st in states.items():
        model.load_state_dict(st)
        func[name] = (torch.stack([torch.sigmoid(model.inner.blend(x))[:, 0] for x in xb]),
                      torch.stack([torch.sigmoid(model.up.lay.blend1(x))[:, 0] for x in x1]))
    A_t, B_t = func["teacher"]
    pt, ft = out["teacher"]
    rows = []
    for name in states:
        if name == "teacher":
            continue
        A, B = func[name]
        da, db = (A - A_t).abs(), (B - B_t).abs()
        row = dict(case=case, model=name)
        for suffix, sel in (("", None), ("_land", land)):
            pick = (lambda z: z) if sel is None else (lambda z, s=sel: z[..., s])
            a_, b_ = pick(da), pick(db)
            for key, M, Mt in (("blend", A, A_t), ("blend1", B, B_t)):
                m, mt_ = pick(M).double(), pick(Mt).double()
                row.update({f"{key}_sse{suffix}": float((m - mt_).pow(2).sum()), f"{key}_sse_one{suffix}": float((m - 1).pow(2).sum()),
                            f"{key}_n{suffix}": int(mt_.numel()), f"{key}_tsum{suffix}": float(mt_.sum()), f"{key}_tsq{suffix}": float(mt_.pow(2).sum())})
            row.update({f"blend_rms{suffix}": float(a_.pow(2).mean().sqrt()), f"blend_p95{suffix}": float(a_.flatten().kthvalue(int(.95 * a_.numel())).values),
                        f"blend1_rms{suffix}": float(b_.pow(2).mean().sqrt()), f"blend1_p95{suffix}": float(b_.flatten().kthvalue(int(.95 * b_.numel())).values),
                        f"wind_sfc_rms{suffix}": float(pick(da * Ds).pow(2).mean().sqrt()), f"wind_res_rms{suffix}": float(pick(db * Dr).pow(2).mean().sqrt())})
        p, f = out[name]
        for place, (P, Tt) in {"station": (p, pt), "grid": (f, ft)}.items():
            for w, sl in (("today", slice(0, 24)), ("tomorrow", slice(24, 48))):
                for si, sp in enumerate(SPECIES):
                    a, b = P[sl, si], Tt[sl, si]
                    ok = np.isfinite(a) & np.isfinite(b)
                    row[f"rmse_{place}_{w}_{sp}"] = float(np.sqrt(((a - b)[ok] ** 2).mean()))
                    pers = np.broadcast_to(Tt[0, si], b.shape)
                    row[f"pers_{place}_{w}_{sp}"] = float(np.sqrt(((pers - b)[ok] ** 2).mean()))
        rows.append(row)
    return rows


def verdicts(d):
    """Pooled distance d of each model's blends to the teacher's, against its start (d0) and the best constant:
    teacher-near if d <= 0.05 and d / d0 <= 0.1; beyond-constant if also d <= 0.5 x the teacher's spread."""
    ref = d[d.model == d.model.iloc[0]]
    rows = []
    for key in ("blend", "blend1"):
        for suffix, area in (("_land", "land"), ("", "all")):
            n = ref[f"{key}_n{suffix}"].sum()
            cstar = ref[f"{key}_tsum{suffix}"].sum() / n
            delta = np.sqrt(max(ref[f"{key}_tsq{suffix}"].sum() / n - cstar ** 2, 0.0))
            dist = {m: np.sqrt(x[f"{key}_sse{suffix}"].sum() / x[f"{key}_n{suffix}"].sum()) for m, x in d.groupby("model")}
            one = {m: np.sqrt(x[f"{key}_sse_one{suffix}"].sum() / x[f"{key}_n{suffix}"].sum()) for m, x in d.groupby("model")}
            for m in dist:
                start = m.split("_")[0] + "_start"
                d0 = dist.get(start, np.nan) if not m.endswith("_start") and m.split("_")[0] not in ("seed0", "const") else np.nan
                near = dist[m] <= .05 and (np.isfinite(d0) and d0 > 0 and dist[m] / d0 <= .1)
                beyond = near and delta > 1e-4 and dist[m] <= .5 * delta
                rows.append(dict(model=m, blend=key, area=area, d=dist[m], d0=d0, d_over_d0=dist[m] / d0 if np.isfinite(d0) and d0 > 0 else np.nan,
                                 c_star=cstar, delta_const=delta, d_over_delta=dist[m] / delta if delta > 0 else np.nan, d_one=one[m],
                                 teacher_near=near, beyond_constant=beyond if delta > 1e-4 else "not judged"))
    return pd.DataFrame(rows)


def normalised_error(d, place):
    """Pooled forecast error to the teacher relative to the teacher's persistence: sqrt(mean over species and windows of
    (RMSE / persistence)^2), per model."""
    cols = [c for c in d.columns if c.startswith(f"rmse_{place}_")]
    return {name: float(np.sqrt(np.mean([(x[c] ** 2).mean() / (x[c.replace("rmse_", "pers_")] ** 2).mean() for c in cols])))
            for name, x in d.groupby("model")}
