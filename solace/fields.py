"""Diagnostics of learned fields and functions (no training).

source_sink        48-h emission sum and mean surface net-change rate of a model on a case
net_balance        emission E, net change R = c (exp(r) - 1) and their sum B on the states of a reference model, for
                   other models applied to exactly the same inputs; compensation K = rms(dB) / sqrt(rms(dE)^2 + rms(dR)^2)
compare            pattern correlation, RMS difference / reference spatial std, domain-mean ratio
seed_spread        surface-field difference between two models by distance to the nearest input station, and their
                   held-station error and difference
transport_curves   learned wind blends and diffusion swept along one physical input, others at the normalisation mean
field_table        RMS field differences of the field experiment relative to the home field
advection_vs_cv    cost of switching advection off against the spatial variability of the home field"""
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from .constants import NX, NY, SPECIES
from .experiments import FIELD_DIST_KM, FIELD_LEADS
from .model import Solace


def _model(state):
    m = Solace().eval()
    m.load_state_dict(state)
    return m


@torch.no_grad()
def source_sink(state, region, batch):
    m = _model(state)
    E, K, hour = [], [], {}

    def on_k(s):
        def hook(_, __, out):
            hour[s] = (0.3 * torch.tanh(out[0, 0])).float()
            if len(hour) == 6:
                K.append(torch.stack([hour[i] for i in range(6)]))
                hour.clear()
        return hook
    hooks = [m.up.emis.register_forward_hook(lambda _, __, out: E.append(out[0, :6].float()))]
    hooks += [k.register_forward_hook(on_k(s)) for s, k in enumerate(m.inner.ks)]
    try:
        m(batch, region)
    finally:
        for h in hooks:
            h.remove()
    return {"emission": torch.stack(E[-48:]).sum(0).numpy(), "k": torch.stack(K[-48:]).mean(0).numpy()}


def compare(a, b):
    a, b = a.ravel(), b.ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    sd = a.std()
    return dict(corr=float(np.corrcoef(a, b)[0, 1]) if sd > 0 and b.std() > 0 else np.nan,
                nrms=float(np.sqrt(((a - b) ** 2).mean()) / sd) if sd > 0 else np.nan,
                mean_ratio=float(b.mean() / a.mean()) if a.mean() != 0 else np.nan)


@torch.no_grad()
def record_inputs(state, region, batch):
    """Per-hour inputs of the net-change networks and the emission term of a reference run."""
    m = _model(state)
    rec, hour = {"x": [], "emis": []}, {}

    def on_k(s):
        def hook(_, args):
            hour[s] = args[0].detach().clone()
            if len(hour) == 6:
                rec["x"].append([hour[i] for i in range(6)])
                hour.clear()
        return hook
    hooks = [k.register_forward_pre_hook(on_k(s)) for s, k in enumerate(m.inner.ks)]
    hooks.append(m.up.emis.register_forward_pre_hook(
        lambda _, args: rec["emis"].append(tuple(a.detach().clone() if torch.is_tensor(a) else a for a in args))))
    try:
        m(batch, region)
    finally:
        for h in hooks:
            h.remove()
    rec["x"], rec["emis"] = rec["x"][-48:], rec["emis"][-48:]
    return rec


@torch.no_grad()
def balance_terms(state, rec, lc_index=15):
    """48-h sums of emission, net change and their sum for one model on recorded inputs: {term: [6, NY, NX]}."""
    m = _model(state)
    R = E = 0
    for xs, args in zip(rec["x"], rec["emis"]):
        c = torch.cat([torch.expm1(5 * x[:, lc_index:lc_index + 1]) for x in xs], 1)
        r = torch.cat([0.3 * torch.tanh(k(x)) for k, x in zip(m.inner.ks, xs)], 1)
        R = R + c * torch.expm1(r)
        E = E + m.up.emis(*args)[:, :6]
    R, E = R[0].double().numpy(), E[0].double().numpy()
    return {"emission": E, "net_change": R, "balance": R + E}


def net_balance_rows(case, ref_terms, other, terms):
    rows = []
    rms = {q: np.sqrt(((terms[q] - ref_terms[q]) ** 2).mean((-2, -1))) for q in ref_terms}
    for q in ref_terms:
        for si, sp in enumerate(SPECIES):
            rows.append(dict(case=case, other=other, term=q, species=sp, **compare(ref_terms[q][si], terms[q][si]),
                             rms_vs_emission_std=rms[q][si] / ref_terms["emission"][si].std(),
                             compensation=rms["balance"][si] / np.hypot(rms["emission"][si], rms["net_change"][si])))
    return rows


@torch.no_grad()
def seed_spread(region, batch, truth, state_a, state_b, case):
    """Rows for leads 1, 6, 24, 48: RMS field difference / bin-mean concentration by distance bin; at the held-out stations
    the error of model a and the a-b difference, both relative to the mean observation."""
    held = region.held
    yy, xx = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    km = 4.0 * cKDTree(region.p4.numpy().astype(np.float64)[~held]).query(np.c_[yy.ravel(), xx.ravel()])[0]
    dbin = np.digitize(km, FIELD_DIST_KM[1:-1])
    nb = len(FIELD_DIST_KM) - 1
    L = np.array(FIELD_LEADS) - 1
    F, P = {}, {}
    for name, st in (("a", state_a), ("b", state_b)):
        fields = []
        p, _ = _model(st)(batch, region, fields=fields)
        P[name] = p[0].numpy()[..., held]
        F[name] = np.stack([fields[h - 1][0].numpy().astype(np.float16) for h in FIELD_LEADS]).astype(np.float64).reshape(len(FIELD_LEADS), 6, -1)
    d2 = (F["b"] - F["a"]) ** 2
    truth = truth[..., held]
    rows = []
    for li, lead in enumerate(FIELD_LEADS):
        for si, sp in enumerate(SPECIES):
            row = dict(case=case, lead=lead, species=sp)
            for b in range(nb):
                m = dbin == b
                row[f"field_d{b}"] = np.sqrt(d2[li, si, m].mean()) / F["a"][li, si, m].mean()
            obs = truth[L[li], si]
            ok = np.isfinite(obs)
            if ok.sum() >= 3:
                mo = obs[ok].mean()
                row["held_err"] = np.sqrt(((P["a"][L[li], si][ok] - obs[ok]) ** 2).mean()) / mo
                row["held_seed_diff"] = np.sqrt(((P["b"][L[li], si][ok] - P["a"][L[li], si][ok]) ** 2).mean()) / mo
            rows.append(row)
    return rows


AXES = {"pblh": np.geomspace(50, 3000, 60), "stability": np.linspace(-3, 5, 60), "wind10": np.linspace(0, 25, 60),
        "wind925": np.linspace(0, 30, 60)}


@torch.no_grad()
def transport_curves(states, common_mu):
    """Surface wind blend, diffusion (cells^2 h^-1) and residual-layer blend of each model along one input, every other
    normalised input at the common mean."""
    rows = []
    for label, state in states.items():
        m = _model(state)
        mu, sd = m.inner.mu.flatten().numpy(), m.inner.sd.flatten().numpy()
        for axis, grid in AXES.items():
            raw = np.broadcast_to(common_mu, (len(grid), len(common_mu))).copy()
            if axis == "pblh":
                raw[:, 0] = np.log(grid / 500)
            elif axis == "stability":
                raw[:, 2] = grid
            elif axis == "wind10":
                raw[:, 10] = grid / 10
            z = torch.tensor(((raw - mu) / sd)[:, :, None, None], dtype=torch.float32)
            blend = torch.sigmoid(m.inner.blend(z[:, [0, 2, 10]])).flatten().numpy()
            diff = (0.25 * torch.sigmoid(m.inner.kap(z[:, [0, 10]]))).flatten().numpy()
            up = np.zeros((len(grid), 2, 1, 1), "f4")
            up[:, 0, 0, 0] = raw[:, 2]
            up[:, 1, 0, 0] = grid / 10 if axis == "wind925" else 0.5
            resid = torch.sigmoid(m.up.lay.blend1(torch.tensor(up))).flatten().numpy()
            for j, x in enumerate(grid):
                rows.append(dict(model=label, axis=axis, input=float(x), surface_wind_blend=float(blend[j]),
                                 diffusion=float(diff[j]), residual_wind_blend=float(resid[j])))
    return pd.DataFrame(rows)


def field_table(folders, regions):
    """folders: {home: folder of field-mode experiment files}."""
    def part(name):
        head = name.split(":")[0]
        return name.split(":")[1] if head in regions else head
    rows = []
    for home, folder in folders.items():
        for f in sorted(folder.glob("????-??-??.npz")):
            with np.load(f, allow_pickle=True) as z:
                ops = list(z["operators"])
                frms, frmsd, std, leads = z["field_rms_diff"], z["field_rms_diff_dist"], z["field_home_std"], z["field_leads"]
                mean_d = z["field_home_mean_dist"]
                for oi, n in enumerate(ops):
                    if n == "home":
                        continue
                    for li, lead in enumerate(leads):
                        for si, sp in enumerate(SPECIES):
                            row = dict(home=home, case=f.stem, op=n, part=part(n), lead=int(lead), species=sp, rel_std=frms[oi, li, si] / std[li, si])
                            for b in range(mean_d.shape[-1]):
                                row[f"d{b}"] = frmsd[oi, li, si, b] / mean_d[li, si, b]
                            rows.append(row)
    return pd.DataFrame(rows)


DETAIL = ["blend:1", "kap:0", "kap:max", "resblend:0", "resblend:1", "a12:0", "a12:0.2"]


def advection_vs_cv(field_folders, sensitivity):
    """field_folders: {home: folder of field-mode files}; sensitivity: effect table of the sensitivity experiment."""
    cv = []
    for home, folder in field_folders.items():
        for f in sorted(folder.glob("????-??-??.npz")):
            with np.load(f, allow_pickle=True) as z:
                r = z["field_home_std"] / z["field_home_mean"]
            cv += [dict(home=home, species=sp, cv=float(np.median(r[:, si]))) for si, sp in enumerate(SPECIES)]
    cv = pd.DataFrame(cv).groupby(["home", "species"]).cv.median()
    s = sensitivity
    adv = s[s.op == "ko:adv"].pivot_table(index=["home", "species"], columns="window", values="d_pct")
    det = s[s.op.isin(DETAIL)].assign(a=lambda x: x.d_pct.abs()).groupby(["home", "species"]).a.max().rename("detail_max")
    d = pd.concat([cv, adv.add_prefix("adv_"), det], axis=1).reset_index()
    stats = {c: spearmanr(d.cv, d[c]) for c in ("adv_today", "adv_tomorrow", "detail_max")}
    return d, stats
