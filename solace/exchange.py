"""Operator variants for the module-exchange and sensitivity experiments, as weight dictionaries (no retraining).

home                the regional model
full:<src>          another region's whole model
<src>:<module>      home model with one module from another region (all other weights stay home)
core:<src>          transport, exchange and layer initialisation from another region
local:<src>         emission, net change and outer model from another region
ko:<component>      a component switched off (Solace.switches_off)
<constant>:<value>  a learned function or constant pinned to a bound"""
import numpy as np
import torch

MODULES = {
    "transport": ("inner.blend.", "inner.kap.", "up.lay.blend1."),
    "k": ("inner.ks.",),
    "emission": ("up.emis.pm", "up.emis.q"),
    "emission_k": ("up.emis.pm", "up.emis.q", "inner.ks."),
    "exchange": ("up.lay.a_e", "up.lay.a_d", "up.lay.a_12", "up.lay.a_w"),
    "a_12": ("up.lay.a_12",),
    "layer_init": ("up.lay.g1s", "up.lay.g2s"),
    "outer": ("outer.",),
}
CORE = MODULES["transport"] + MODULES["exchange"] + MODULES["layer_init"]
LOCAL_PART = MODULES["emission_k"] + MODULES["outer"]
KNOCKOUTS = {"ko:adv": ("advection",), "ko:k": ("net_change",), "ko:emis": ("emission",), "ko:emis+k": ("emission", "net_change"),
             "ko:exch": ("exchange",), "ko:gate": ("boundary_gate",)}
LOGIT = 13.815510557964274  # logit(1 - 1e-6): a sigmoid-bounded value at its bound
BOUNDS = {"a12:0": {"up.lay.a_12": -LOGIT}, "a12:0.2": {"up.lay.a_12": LOGIT},
          "init:1": {"up.lay.g1s": -0.22314355131420976, "up.lay.g2s": -0.22314355131420976},  # 0.2 + 1.8 sigmoid(z) = 1
          "init:0.2": {"up.lay.g1s": -LOGIT, "up.lay.g2s": -LOGIT}, "init:2": {"up.lay.g1s": LOGIT, "up.lay.g2s": LOGIT}}
# learned transport functions pinned to an output bound: zero last layer, bias at +-LOGIT. Surface wind blend: 0 = 925/850
# hPa wind, 1 = 10-m wind; diffusion: 0 or 0.25 cell^2 h-1; residual-layer wind blend: 0 = 850 hPa, 1 = 925 hPa.
for _name, _layer, _sign in (("blend:0", "inner.blend.4", -1), ("blend:1", "inner.blend.4", 1), ("kap:0", "inner.kap.4", -1),
                             ("kap:max", "inner.kap.4", 1), ("resblend:0", "up.lay.blend1.4", -1), ("resblend:1", "up.lay.blend1.4", 1)):
    BOUNDS[_name] = {_layer + ".weight": 0.0, _layer + ".bias": _sign * LOGIT}
PRUNE = {"prune:all": BOUNDS["blend:1"] | BOUNDS["kap:0"] | BOUNDS["resblend:1"] | BOUNDS["a12:0"],
         "prune:transport": BOUNDS["blend:1"] | BOUNDS["kap:0"] | BOUNDS["resblend:1"]}


def swapped(home, foreign, prefixes):
    out = {k: v.clone() for k, v in home.items()}
    n = 0
    for k in out:
        if any(k == p or k.startswith(p) for p in prefixes):
            out[k] = foreign[k].clone()
            n += 1
    if n == 0:
        raise KeyError(prefixes)
    return out


def pinned(base, values):
    state = dict(base)
    for k, v in values.items():
        state[k] = torch.full_like(base[k], v)
    return state


def dosed(base, key, lo, hi, value, index=None):
    """A sigmoid-bounded constant lo + (hi - lo) sigmoid(z) set to `value` (one element with `index`)."""
    state = dict(base)
    t = base[key].clone()
    raw = float(np.log((value - lo) / (hi - value)))
    if index is None:
        t.fill_(raw)
    else:
        t.view(-1)[index] = raw
    state[key] = t
    return state


def matrix(home_state, others):
    """Variants of the exchange matrix: {name: state}. others: {region: state}."""
    out = {"home": home_state}
    for src, st in others.items():
        out["full:" + src] = st
        for m, prefixes in MODULES.items():
            out[f"{src}:{m}"] = swapped(home_state, st, prefixes)
    return out


def core(home_state, others):
    out = {"home": home_state}
    for src, st in others.items():
        out["core:" + src] = swapped(home_state, st, CORE)
        out["local:" + src] = swapped(home_state, st, LOCAL_PART)
    return out


def sensitivity(home_state):
    """{name: (state, switches_off)} for the knock-outs and the bounds."""
    out = {"home": (home_state, ())}
    out |= {name: (home_state, off) for name, off in KNOCKOUTS.items()}
    out |= {name: (pinned(home_state, v), ()) for name, v in BOUNDS.items()}
    return out


# layer constants moved across their range, everything else home: key -> (lower, upper bound, element or None, values)
DOSE = {"up.lay.g1s": (0.2, 2.0, 2, (0.25, 0.6, 1.0, 1.3, 1.6, 1.75, 1.85, 1.98)),
        "up.lay.a_e": (0.0, 1.0, None, (0.02, 0.1, 0.17, 0.28, 0.4, 0.6, 0.9)),
        "up.lay.a_d": (0.0, 1.0, None, (0.02, 0.1, 0.3, 0.44, 0.6, 0.9))}
MODES = ("matrix", "core", "sensitivity", "seed", "seedfull", "prune", "dose", "fixa", "fixswap", "field")


def variants(mode, home, others=None, other=None):
    """{name: (state, switches_off)} of one experiment.
    others: {region: state} (matrix, core, field); other: a second model of the home region (seed: another random seed;
    fixa, fixswap: the model retrained with the insensitive parts pinned, see PRUNE)."""
    plain = lambda d: {k: (v, ()) for k, v in d.items()}  # noqa: E731
    if mode == "matrix":
        return plain(matrix(home, others))
    if mode == "core":
        return plain(core(home, others))
    if mode == "sensitivity":
        return sensitivity(home)
    if mode == "prune":
        return plain({"home": home} | {k: pinned(home, v) for k, v in PRUNE.items()})
    if mode == "dose":
        out = {"home": home}
        for key, (lo, hi, idx, values) in DOSE.items():
            for v in values:
                out[f"{key}={v:g}"] = dosed(home, key, lo, hi, v, idx)
        return plain(out)
    if mode == "seedfull":
        return plain({"home": home, "full:seed1": other})
    if mode == "seed":
        out = {"home": home, "full:seed1": other}
        out |= {f"seed1:{m}": swapped(home, other, p) for m, p in MODULES.items()}
        out |= {f"seed1|seed0:{m}": swapped(other, home, p) for m, p in MODULES.items()}
        return plain(out)
    if mode == "fixa":
        return plain({"home": home, "prune:all": pinned(home, PRUNE["prune:all"]), "full:fixA": pinned(other, PRUNE["prune:all"])})
    if mode == "fixswap":
        fixed = pinned(other, PRUNE["prune:all"])
        out = {"home": home, "full:fixA": fixed}
        out |= {f"fixA:{m}": swapped(home, fixed, p) for m, p in MODULES.items()}
        out |= {f"fixA|home:{m}": swapped(fixed, home, p) for m, p in MODULES.items()}
        return plain(out)
    if mode == "field":
        out = {"home": home}
        for src, st in others.items():
            out[f"{src}:transport"] = swapped(home, st, MODULES["transport"])
            out[f"{src}:exchange"] = swapped(home, st, MODULES["exchange"])
            out["core:" + src] = swapped(home, st, CORE)
            out[f"{src}:emission_k"] = swapped(home, st, MODULES["emission_k"])
            out["full:" + src] = st
        return plain(out)
    raise ValueError(mode)
