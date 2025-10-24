
from __future__ import annotations
import numpy as np, pandas as pd
from typing import Dict
from .core import Universe, make_universe
from .eval import timeseries, decompose, choose_window

def gray(n:int)->int: return n ^ (n>>1)

def enum_candidate(n:int, N:int)->Dict:
    s=n
    c=s%N; s//=N
    rset=[2,3,4]; r=rset[s%len(rset)]; s//=len(rset)
    ptype=s%3; s//=3  # 0:x_cur, 1:x_prev, 2:B
    L=2*r+1; mask=gray(s) & ((1<<L)-1)
    return dict(c=c, r=r, ptype=ptype, mask=mask)

def apply_local_perturb(U:Universe, cand:Dict)->None:
    c=cand["c"]; r=cand["r"]; ptype=cand["ptype"]; mask=cand["mask"]
    for j in range(-r, r+1):
        bit=(mask>>(j+r))&1
        if bit==0: continue
        i=(c+j)%U.N
        if ptype==0: U.cells[i].x_cur ^= 1
        elif ptype==1: U.cells[i].x_prev ^= 1
        else: U.cells[i].m["B"] = (U.cells[i].m.get("B",0) ^ 1)

def measure_candidate(N:int, T:int, cand:Dict)->Dict:
    # Current core.make_universe signature: (N, mu, L?)
    U=make_universe(N=N)
    apply_local_perturb(U, cand)
    df=timeseries(U,T=T)
    sub=df[df["t"]>=1].copy()
    # Column names per eval.timeseries: H_x_block3, MI_x_a, budget_residual
    R=sub["budget_residual"].values.astype(float)
    dec=decompose(R, frac=0.3, use_B4B6=True)
    best_w, best_dec, _ = choose_window(R, lam=0.1)
    mi_gain=float(df["MI_x_a"].iloc[-1]-df["MI_x_a"].iloc[0])
    h3_gain=float(df["H_x_block3"].iloc[-1]-df["H_x_block3"].iloc[0])
    return dict(alias_share=dec["alias"], recon_err=dec["recon_err"],
                alias_win=best_dec["alias"], win_name=best_w,
                mi_gain=mi_gain, h3_gain=h3_gain)

def dovetail(N:int=64, layers:int=4, delta:int=48, num_cand:int=32)->pd.DataFrame:
    rows=[]
    for layer in range(1,layers+1):
        T=delta*layer
        for n in range(num_cand):
            cand=enum_candidate(n,N)
            met=measure_candidate(N,T,cand)
            rows.append(dict(layer=layer, T=T, cand_id=n, **cand, **met))
    return pd.DataFrame(rows)

def pareto_front(df:pd.DataFrame, keys_max, keys_min)->pd.DataFrame:
    keep=[]
    vals=df.reset_index(drop=True)
    for i in range(len(vals)):
        ai=vals.loc[i]; dominated=False
        for j in range(len(vals)):
            if i==j: continue
            aj=vals.loc[j]
            ge_max=all(aj[k]>=ai[k] for k in keys_max)
            le_min=all(aj[k]<=ai[k] for k in keys_min)
            strict=any(aj[k]>ai[k] for k in keys_max) or any(aj[k]<ai[k] for k in keys_min)
            if ge_max and le_min and strict: dominated=True; break
        if not dominated: keep.append(i)
    return vals.loc[keep].copy()
