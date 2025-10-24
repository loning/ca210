
from __future__ import annotations
import math, numpy as np, pandas as pd
from typing import Dict, Tuple, List
from .core import Universe, make_universe

def shannon_entropy_from_counts(counts: Dict[int,int]) -> float:
    total=sum(counts.values()); 
    if total<=0: return 0.0
    H=0.0
    for c in counts.values():
        if c>0:
            p=c/total; H -= p*math.log(p,2)
    return H

def block_counts(bits: List[int], k:int=3)->Dict[Tuple[int,...], int]:
    N=len(bits); cnt={}
    for i in range(N):
        block=tuple(bits[(i+j)%N] for j in range(k))
        cnt[block]=cnt.get(block,0)+1
    return cnt

def block_entropy(bits: List[int], k:int=3)->float:
    return shannon_entropy_from_counts(block_counts(bits,k))

def mutual_info_xy(x_bits: List[int], y_bits: List[int])->float:
    joint={(0,0):0,(0,1):0,(1,0):0,(1,1):0}; mx={0:0,1:0}; my={0:0,1:0}
    for a,b in zip(x_bits,y_bits):
        joint[(a&1,b&1)]+=1; mx[a&1]+=1; my[b&1]+=1
    Hx=shannon_entropy_from_counts(mx); Hy=shannon_entropy_from_counts(my)
    Hj=shannon_entropy_from_counts(joint)
    return Hx+Hy-Hj

def snapshot_bits(U: Universe)->Tuple[List[int],List[int]]:
    x=[c.x_cur&1 for c in U.cells]; a=[c.a&1 for c in U.cells]; return x,a

def holo_bits_written(U: Universe)->float:
    total=0
    for c in U.cells:
        eta,k,rho,u,dB=(c.h+(0,0,0,0,0))[:5]
        total += (eta&1)+(u&1)+(dB&1)
    return total/float(U.N)

def timeseries(U: Universe, T:int=96)->pd.DataFrame:
    rows=[]; x,a=snapshot_bits(U)
    rows.append(dict(t=0, H_x_block3=block_entropy(x,3),
                     H_a=shannon_entropy_from_counts({b:a.count(b) for b in (0,1)}),
                     MI_x_a=mutual_info_xy(x,a),
                     freedom_ratio=U.freedom_ratio(),
                     budget_residual=0.0))
    prev=rows[-1]
    for t in range(1,T+1):
        U.step(t-1); x,a=snapshot_bits(U)
        Hx3=block_entropy(x,3); Ha=shannon_entropy_from_counts({b:a.count(b) for b in (0,1)}); MI=mutual_info_xy(x,a)
        holo=holo_bits_written(U)
        dS_phys=Hx3-prev["H_x_block3"]; dS_cons=(-Ha)-(-prev["H_a"]); dI=MI-prev["MI_x_a"]
        resid=dS_phys+dS_cons+holo-dI
        rows.append(dict(t=t, H_x_block3=Hx3, H_a=Ha, MI_x_a=MI, freedom_ratio=U.freedom_ratio(), budget_residual=resid))
        prev=rows[-1]
    return pd.DataFrame(rows)

def bernoulli_poly(t: np.ndarray, order:int)->np.ndarray:
    if order==2: return t*t - t + 1.0/6.0
    if order==4: return t**4 - 2*t**3 + t**2 - 1.0/30.0
    if order==6: return t**6 - 3*t**5 + 2.5*t**4 - 0.5*t**2 + 1.0/42.0
    raise ValueError

def three_decompose(series: np.ndarray, frac:float=0.3, use_B6:bool=True):
    x=np.asarray(series,float); x=x-np.mean(x)
    X=np.fft.rfft(x); K=len(X); kc=max(1,int(frac*K))
    Xl=np.zeros_like(X); Xh=np.zeros_like(X); Xl[:kc]=X[:kc]; Xh[kc:]=X[kc:]
    low=np.fft.irfft(Xl, n=len(x)); alias=np.fft.irfft(Xh, n=len(x))
    t=(np.arange(len(x))+0.5)/len(x)
    Phi=np.vstack([bernoulli_poly(t,2), bernoulli_poly(t,4)]).T
    if use_B6: Phi=np.vstack([Phi.T, bernoulli_poly(t,6)]).T
    beta,*_=np.linalg.lstsq(Phi, low, rcond=None)
    bern=Phi@beta; trunc=(low-bern)
    E=lambda z: float(np.dot(z,z))
    eT=max(1e-12, E(x))
    shares=dict(alias=E(alias)/eT, bern=E(bern)/eT, trunc=E(trunc)/eT)
    recon=alias+bern+trunc; err=float(np.max(np.abs(recon-x)))
    return dict(**shares, recon_err=err, alias_series=alias, bern_series=bern, trunc_series=trunc, recon_series=recon)

# Compatibility wrapper to match documented API name/signature
def decompose(series: np.ndarray, frac: float = 0.3, use_B4B6: bool = True):
    """Decompose into alias/bern/trunc shares and return a dict with recon_err.

    This wraps three_decompose; `use_B4B6=True` enables B6 layer to match docs.
    """
    return three_decompose(series, frac=frac, use_B6=use_B4B6)

def normalize_mean1(h: np.ndarray)->np.ndarray:
    m=float(np.mean(h)) or 1.0; return h/m
def win_blackman(L:int):
    n=np.arange(L); return 0.42 - 0.5*np.cos(2*np.pi*n/L) + 0.08*np.cos(4*np.pi*n/L)
def win_hann(L:int):
    n=np.arange(L); return 0.5 - 0.5*np.cos(2*np.pi*n/L)
def win_hamming(L:int):
    n=np.arange(L); return 0.54 - 0.46*np.cos(2*np.pi*n/L)
def win_rect(L:int):
    return np.ones(L)
def win_exp(L:int, lam:float=12.0):
    n=np.arange(L); d=np.minimum(n, L-1-n)/(0.5*L); return np.exp(-lam*d)
def win_kaiser(L:int, beta:float=12.0):
    return np.kaiser(L, beta)

WINDOWS=[("blackman",lambda L:win_blackman(L)),("hann",lambda L:win_hann(L)),("hamming",lambda L:win_hamming(L)),
         ("rect",lambda L:win_rect(L)),("exp12",lambda L:win_exp(L,12.0)),("kaiser12",lambda L:win_kaiser(L,12.0))]

def penalty(h: np.ndarray)->float:
    L=len(h); num=float(np.sum(h)); denom=L*math.sqrt(float(np.mean(h*h)) or 1.0)
    return 1.0 - (num / (denom or 1.0))

def choose_window(series: np.ndarray, windows=WINDOWS, lam:float=0.1):
    best=None; rows=[]
    L=len(series)
    for name,fn in windows:
        h=normalize_mean1(fn(L)); y=series*h
        dec=three_decompose(y, use_B6=True, frac=0.3)
        J=dec["alias"] + lam*penalty(h)
        rows.append(dict(window=name, objective=J, alias_share=dec["alias"], recon_err=dec["recon_err"]))
        if best is None or J<best[0]: best=(J,name,dec)
    import pandas as pd
    table=pd.DataFrame(rows).sort_values("objective").reset_index(drop=True)
    return best[1], best[2], table

def s0_phase_invariance(N:int=96, mu:int=5)->Dict[str,float]:
    U=make_universe(N=N, mu=mu); x=[c.x_cur for c in U.cells]
    def HNeN2(bits):
        c0=bits.count(0); c1=len(bits)-c0; N=len(bits); p0=c0/max(1,N); p1=c1/max(1,N)
        H=0.0
        for p in (p0,p1):
            if p>0: H-=p*math.log(p,2)
        Ne=math.exp(H); N2=1.0/((p0*p0+p1*p1) or 1e-12)
        return H,Ne,N2
    H1,Ne1,N21=HNeN2(x); s=13; x2=x[s:]+x[:s]; H2,Ne2,N22=HNeN2(x2)
    return {"H_diff":abs(H1-H2),"Neff_diff":abs(Ne1-Ne2),"N2_diff":abs(N21-N22)}

def s8_difference_annihilation(series: np.ndarray, order:int=2)->Dict[str,float]:
    y=np.asarray(series,float); x=y.copy()
    for _ in range(order): x = x[1:]-x[:-1]
    E=lambda z: float(np.dot(z,z))
    return {"order":order, "energy_drop": (E(y)-E(x))/max(1e-12,E(y))}

def s11_orbit_spectrum_toy(N:int=64, mu:int=5, T:int=96)->Dict[str,float]:
    U=make_universe(N=N, mu=mu)
    def hann(L): n=np.arange(L); return 0.5 - 0.5*np.cos(2*np.pi*n/L)
    geom=[]; xs=[]
    for t in range(T):
        x=[c.x_cur for c in U.cells]; xs.append(x)
        cnt=sum(1 for i in range(U.N) if x[i]==0 and x[(i-1)%U.N]==1)
        geom.append(cnt); U.step(t)
    hs=hann(T); G=float(np.dot(geom,hs))
    power=0.0
    for x in xs:
        X=np.fft.rfft(np.array(x)-np.mean(x)); power += float((np.abs(X)**2).sum())
    S=float(power/len(xs))*float(np.sum(hs))
    return {"orbit_sum":G, "spectrum_sum":S, "diff":S-G}
