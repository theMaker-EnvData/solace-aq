"""Attenuation from components to stations (two ratios of Ju, 2026) for a trained model, without training.

Small perturbations are added to the outputs of the learned functions:
  transport  surface wind blend, diffusion and residual-layer blend, in their sigmoid output, pushed towards the interior
             of (0, 1) along nonnegative bases (constant, positive and negative parts of each input), size 0.02
  k          the hourly log rate 0.3 tanh f, bases: constant, mixing height, stability, temperature, solar radiation and
             the species' own concentration, size 0.01
  emission   relative change, bases: constant and two diurnal harmonics, size 0.05
plus five random unit combinations of the bases per group. For each direction and size (eps, eps / 2), sums of squares by
hour, layer and species of
  local  the direct effect of one perturbed step on the base trajectory state
  full   the state difference with the perturbation on for all 48 h
  stn    the surface difference at the input stations
r1 = ||full|| / ||local||, r2 = ||stn|| / ||full||; directions whose response is not linear (eps vs eps / 2 ratio outside
2 +- 15 %) are excluded."""
import numpy as np
import pandas as pd
import torch

EPS = {"blend": 0.02, "kap": 0.02, "blend1": 0.02, "k": 0.01, "emis": 0.05}
NB = {"blend": 7, "kap": 5, "blend1": 5, "k": 6, "emis": 3}
K_INPUTS = (0, 2, 6, 7)  # mixing height, stability t925 - t2m, t2m, solar radiation (+ own log concentration)
GROUP = {"blend": "transport", "kap": "transport", "blend1": "transport", "k": "k", "emis": "emission"}
N_RANDOM = 5
TOLERANCE = 0.15
SPECIES = ("pm25", "pm10", "o3", "no2", "so2", "co")


def basis(name, x, tf=None):
    if name in ("blend", "kap", "blend1"):
        return torch.cat([torch.ones_like(x[:, :1]), torch.relu(x), torch.relu(-x)], 1)
    if name == "k":
        return torch.cat([torch.ones_like(x[:, :1]), x[:, list(K_INPUTS)], x[:, -1:]], 1)
    one = torch.ones(tf.shape[0], 1, 1, 1, device=tf.device)
    return torch.cat([one, tf[:, 0].view(-1, 1, 1, 1), tf[:, 1].view(-1, 1, 1, 1)], 1)


def directions(rng):
    keys = [("blend", None), ("kap", None), ("blend1", None)] + [("k", s) for s in range(6)] + [("emis", s) for s in range(6)]
    out = []
    for name, s in keys:
        for j in range(NB[name]):
            c = np.zeros(NB[name], "f4")
            c[j] = 1
            out.append({"name": f'{name}{"" if s is None else s}:{j}', "group": GROUP[name], "coef": {(name, s): c}})
    for group in ("transport", "k", "emission"):
        members = [(n, s) for n, s in keys if GROUP[n] == group]
        for r in range(N_RANDOM):
            v = rng.standard_normal(sum(NB[n] for n, _ in members)).astype("f4")
            v /= np.linalg.norm(v)
            i, coef = 0, {}
            for n, s in members:
                coef[(n, s)] = v[i:i + NB[n]]
                i += NB[n]
            out.append({"name": f"{group}:random{r}", "group": group, "coef": coef})
    return out


class Perturbation:
    def __init__(self, model):
        self.on, self.eps, self.coef = False, 0.0, {}
        inner, up = model.inner, model.up
        inner.blend.register_forward_hook(self._sigmoid("blend"))
        inner.kap.register_forward_hook(self._sigmoid("kap"))
        up.lay.blend1.register_forward_hook(self._sigmoid("blend1"))
        for s, k in enumerate(inner.ks):
            k.register_forward_hook(self._rate(s))
        up.emis.register_forward_hook(self._emission)

    def _sigmoid(self, name):
        def hook(module, inp, out):
            c = self.coef.get((name, None)) if self.on else None
            if c is None:
                return None
            d = (basis(name, inp[0]) * torch.as_tensor(c).view(1, -1, 1, 1)).sum(1, keepdim=True) * self.eps
            a = torch.sigmoid(out)
            d = d.abs()
            return torch.logit((a + torch.where(a > 0.5, -d, d)).clamp(1e-6, 1 - 1e-6))
        return hook

    def _rate(self, s):
        def hook(module, inp, out):
            c = self.coef.get(("k", s)) if self.on else None
            if c is None:
                return None
            d = (basis("k", inp[0]) * torch.as_tensor(c).view(1, -1, 1, 1)).sum(1, keepdim=True) * self.eps
            return torch.atanh((0.3 * torch.tanh(out) + d).clamp(-0.2999, 0.2999) / 0.3)
        return hook

    def _emission(self, module, inp, out):
        if not self.on:
            return None
        new = out
        for s in range(6):
            c = self.coef.get(("emis", s))
            if c is None:
                continue
            d = (basis("emis", None, inp[1]) * torch.as_tensor(c).view(1, -1, 1, 1)).sum(1, keepdim=True) * self.eps
            new = torch.cat([new[:, :s], new[:, s:s + 1] * (1 + d), new[:, s + 1:]], 1)
        return new


class Stepper:
    """Wraps the hourly step: records the base trajectory, or measures the direct ('local') or propagated ('full') state
    differences against it."""

    def __init__(self, model):
        self.orig, self.mode, self.base, self.n, self.acc = model.step, "record", [], 0, None
        model.step = self

    def __call__(self, *args):
        out = self.orig(*args)
        i = self.n
        self.n += 1
        if self.mode == "record":
            self.base.append(tuple(t.detach().clone() for t in out[:3]))
            return out
        self.acc[i] = torch.stack([((out[j][0, :6] - self.base[i][j][0, :6]) ** 2).sum((-1, -2)) for j in range(3)]).double().numpy()
        if self.mode == "local":
            return (*self.base[i], out[3])
        return out

    def run(self, model, batch, region, mode):
        self.mode, self.n = mode, 0
        if mode == "record":
            self.base = []
        else:
            self.acc = np.zeros((48, 3, 6))
        with torch.inference_mode():
            return model(batch, region)[0][0]


def run_case(model, region, batch, dirs):
    """Sums of squares for every direction: full, local [dir, eps, 48, layer, species], stn [dir, eps, 48, species]."""
    pert, st = Perturbation(model), Stepper(model)
    inputs = ~region.held
    base = st.run(model, batch, region, "record")
    full = np.zeros((len(dirs), 2, 48, 3, 6))
    local = np.zeros_like(full)
    stn = np.zeros((len(dirs), 2, 48, 6))
    for di, d in enumerate(dirs):
        pert.coef = d["coef"]
        name = next(iter(d["coef"]))[0]
        for ei, f in enumerate((1.0, 0.5)):
            pert.eps, pert.on = EPS[name] * f, True
            st.run(model, batch, region, "local")
            local[di, ei] = st.acc
            pred = st.run(model, batch, region, "full")
            full[di, ei] = st.acc
            stn[di, ei] = ((pred[:, :6][..., inputs] - base[:, :6][..., inputs]) ** 2).sum(-1).double().numpy()
            pert.on = False
    return dict(names=np.array([d["name"] for d in dirs]), groups=np.array([d["group"] for d in dirs]),
                eps=np.array([EPS[next(iter(d["coef"]))[0]] for d in dirs]), full=full, local=local, stn=stn,
                n_cells=175 * 153, n_stations=int(inputs.sum()))


def table(files):
    rows = []
    for f in files:
        z = np.load(f)
        F, L, S = z["full"], z["local"], z["stn"]
        for d, (name, group) in enumerate(zip(z["names"], z["groups"])):
            for si, sp in enumerate(SPECIES):
                f0, f1 = F[d, 0, :, :, si].sum(), F[d, 1, :, :, si].sum()
                l0, l1 = L[d, 0, :, :, si].sum(), L[d, 1, :, :, si].sum()
                s0, fs = S[d, 0, :, si].sum(), F[d, 0, :, 0, si].sum()
                if l0 <= 0 or f0 <= 0:
                    continue
                rows.append(dict(window=f.stem, name=str(name), group=str(group), species=sp, random="random" in str(name),
                                 full=f0, local=l0, stn=s0, full_sfc=fs, r1=np.sqrt(f0 / l0), r2=np.sqrt(s0 / f0),
                                 r2s=np.sqrt(s0 / fs) if fs > 0 else np.nan,
                                 lin_full=np.sqrt(f0 / f1) if f1 > 0 else np.nan, lin_local=np.sqrt(l0 / l1) if l1 > 0 else np.nan,
                                 n_stations=int(z["n_stations"]), n_cells=int(z["n_cells"])))
    d = pd.DataFrame(rows)
    d["r1r2"] = d.r1 * d.r2
    rename = {f"k{s}": "k" for s in range(6)} | {f"emis{s}": "emission" for s in range(6)}
    d["component"] = np.where(d.random, d.group, d.name.str.split(":").str[0].replace(rename))
    d["linear"] = (d.lin_full.sub(2).abs() <= 2 * TOLERANCE) & (d.lin_local.sub(2).abs() <= 2 * TOLERANCE)
    return d


def summaries(d, seed=20261028):
    """Group medians, component medians (basis directions) and transport / k and transport / emission ratios with a
    window bootstrap (2000 draws)."""
    groups = d[d.linear].groupby(["group", "random"])[["r1", "r2", "r2s", "r1r2"]].median()
    components = d[d.linear & ~d.random].groupby("component")[["r1", "r2", "r2s", "r1r2"]].median()
    rng = np.random.default_rng(seed)
    win = np.array(sorted(d.window.unique()))
    out = []
    for rand in (False, True):
        x = d[(d.random == rand) & d.linear]
        for q in ("r1", "r2", "r1r2"):
            for other in ("k", "emission"):
                a = {w: g[g.group == "transport"][q].to_numpy() for w, g in x.groupby("window")}
                b = {w: g[g.group == other][q].to_numpy() for w, g in x.groupby("window")}

                def ratio(ws):
                    return np.median(np.concatenate([a[w] for w in ws])) / np.median(np.concatenate([b[w] for w in ws]))
                boot = [ratio(rng.choice(win, len(win))) for _ in range(2000)]
                out.append(dict(directions="random" if rand else "basis", quantity=q, contrast=f"transport/{other}", ratio=ratio(win),
                                lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5)))
    return groups, components, pd.DataFrame(out)
