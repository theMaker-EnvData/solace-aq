"""Tables of the operator experiments (files written by solace.experiments), held-out stations, pooled RMSE, today (leads
1-24) and tomorrow (25-48). Intervals: circular 7-day moving-block bootstrap on calendar days, 2000 resamples, 95 %.

matrix_table:  home advantage H = RMSE(full:src) / RMSE(home) - 1; module effect d = RMSE(src:module) / RMSE(home) - 1;
               share = (RMSE(src:module) - RMSE(home)) / (RMSE(full:src) - RMSE(home)) where the H interval excludes 0
effect_table:  d for every variant of any experiment"""
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import SPECIES
from .evaluate import WINDOWS, block_weights

SEED = 20261026


def load(folder, persistence=False, all_stations=False):
    files = sorted(Path(folder).glob("????-??-??.npz"))
    if not files:
        return None
    ops, sse, n, pers, alls, alln = None, [], [], [], [], []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            if ops is None:
                ops = [str(o) for o in z["operators"]]
            P = z["held_pred"].astype("f4")
            T = z["held_truth"].astype("f4")
            ok = np.isfinite(T)
            e2 = np.where(ok, (P - T) ** 2, 0.0)
            sse.append([e2[:, w].sum((1, 3)) for w in WINDOWS.values()])
            n.append([ok[w].sum((0, 2)) for w in WINDOWS.values()])
            if persistence:
                per = z["held_persistence"].astype("f4")
                pers.append([np.where(ok, (per[None] - T) ** 2, 0.0)[w].sum((0, 2)) for w in WINDOWS.values()])
            if all_stations:
                alls.append([z["all_sse"][:, w].sum(1) for w in WINDOWS.values()])
                alln.append([z["all_n"][w].sum(0) for w in WINDOWS.values()])
    return dict(ops=ops, days=pd.to_datetime([f.stem for f in files]), held_n=int(P.shape[-1]), sse=np.array(sse),
                n=np.array(n), pers=np.array(pers), all_sse=np.array(alls), all_n=np.array(alln))


def rmse(sse, n, W=None):
    """sse [case, window, op, species], n [case, window, species] -> [window, op, species] (or [b, ...] with weights)."""
    if W is None:
        return np.sqrt(sse.sum(0) / n.sum(0)[:, None, :])
    return np.sqrt(np.einsum("bc,cwos->bwos", W, sse) / np.einsum("bc,cws->bws", W, n)[:, :, None, :])


def matrix_table(folders):
    """folders: {home region: folder of matrix-mode files}."""
    rng = np.random.default_rng(SEED)
    rows = []
    for home, folder in folders.items():
        D = load(folder, persistence=True, all_stations=True)
        ops = D["ops"]
        W = block_weights(D["days"], rng)
        R, Rb = rmse(D["sse"], D["n"]), rmse(D["sse"], D["n"], W)
        A = np.sqrt(D["all_sse"].sum(0) / D["all_n"].sum(0)[:, None, :])
        Pr = np.sqrt(D["pers"].sum(0) / D["n"].sum(0))
        h = ops.index("home")
        for wi, wn in enumerate(WINDOWS):
            for si, sp in enumerate(SPECIES):
                base, bb = R[wi, h, si], Rb[:, wi, h, si]
                for oi, op in enumerate(ops):
                    if op == "home":
                        continue
                    src, mod = op.split(":")
                    full = src == "full"
                    if full:
                        src, mod = mod, "full"
                    fi = ops.index("full:" + src)
                    db = Rb[:, wi, oi, si] / bb - 1
                    gap, gapb = R[wi, fi, si] - base, Rb[:, wi, fi, si] - bb
                    Hlo, Hhi = np.percentile(Rb[:, wi, fi, si] / bb - 1, [2.5, 97.5])
                    row = dict(home=home, source=src, module=mod, window=wn, species=sp, cases=len(D["days"]), held_stations=D["held_n"],
                               rmse_home=base, rmse=R[wi, oi, si], rmse_persistence=Pr[wi, si], d_pct=100 * (R[wi, oi, si] / base - 1),
                               d_lo=100 * np.percentile(db, 2.5), d_hi=100 * np.percentile(db, 97.5),
                               H_pct=100 * (R[wi, fi, si] / base - 1), H_lo=100 * Hlo, H_hi=100 * Hhi,
                               all_d_pct=100 * (A[wi, oi, si] / A[wi, h, si] - 1))
                    if not full and (Hlo > 0 or Hhi < 0):
                        sb = (Rb[:, wi, oi, si] - bb) / gapb
                        row |= dict(share_pct=100 * (R[wi, oi, si] - base) / gap, share_lo=100 * np.percentile(sb, 2.5),
                                    share_hi=100 * np.percentile(sb, 97.5))
                    rows.append(row)
    return pd.DataFrame(rows)


def effect_table(folders):
    """folders: {home region: folder}; d = RMSE(variant) / RMSE(home) - 1 for every variant."""
    rng = np.random.default_rng(SEED)
    rows = []
    for home, folder in folders.items():
        D = load(folder)
        if D is None:
            continue
        ops = D["ops"]
        h = ops.index("home")
        W = block_weights(D["days"], rng)
        R, Rb = rmse(D["sse"], D["n"]), rmse(D["sse"], D["n"], W)
        for wi, wn in enumerate(WINDOWS):
            for si, sp in enumerate(SPECIES):
                for oi, op in enumerate(ops):
                    if oi == h:
                        continue
                    db = Rb[:, wi, oi, si] / Rb[:, wi, h, si] - 1
                    rows.append(dict(home=home, op=op, window=wn, species=sp, cases=len(D["days"]), rmse_home=R[wi, h, si],
                                     rmse=R[wi, oi, si], d_pct=100 * (R[wi, oi, si] / R[wi, h, si] - 1),
                                     d_lo=100 * np.percentile(db, 2.5), d_hi=100 * np.percentile(db, 97.5)))
    return pd.DataFrame(rows)
