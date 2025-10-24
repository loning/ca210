
from __future__ import annotations
import numpy as np, pandas as pd, math
from typing import Dict, List
from .core import LawTrack, make_universe
from .eval import timeseries, choose_window

CFG_TABLE={
 "GOLD-id-soft": dict(lex_mode="GOLD", h_L_mode="id",   hard_consensus=False),
 "GOLD-id-hard": dict(lex_mode="GOLD", h_L_mode="id",   hard_consensus=True),
 "GOLD-zero":    dict(lex_mode="GOLD", h_L_mode="zero", hard_consensus=False),
 "FWO-id-soft":  dict(lex_mode="FWO",  h_L_mode="id",   hard_consensus=False),
 "FWO-id-hard":  dict(lex_mode="FWO",  h_L_mode="id",   hard_consensus=True),
 "FWO-zero":     dict(lex_mode="FWO",  h_L_mode="zero", hard_consensus=False),
}

def build_universe_from_cfg(name:str,N:int=64,mu:int=5):
    kw=CFG_TABLE[name]; chi=2 if (N%2==0) else 3
    L=LawTrack(chi=chi, h_L_mode=kw["h_L_mode"], hard_consensus=kw["hard_consensus"], lex_mode=kw["lex_mode"])
    return make_universe(N=N, mu=mu, L=L)

def get_timeseries_all(N:int=64, T:int=96)->pd.DataFrame:
    rows=[]
    for name in CFG_TABLE.keys():
        U=build_universe_from_cfg(name, N=N)
        df=timeseries(U, T=T); df.insert(0,"name",name); rows.append(df)
    return pd.concat(rows, ignore_index=True)

def calibrate_thresholds(rows: List[Dict], anchor:str="GOLD-id-soft")->Dict[str,float]:
    anchor_row=None
    for r in rows:
        if r["config"]==anchor: anchor_row=r; break
    if anchor_row is None: anchor_row=min(rows, key=lambda r:r["alias_share"])
    a0=anchor_row["alias_share"]; e0=anchor_row["recon_err"]
    Ts_alias=min(0.25, max(0.15, 1.25*a0)); Ts_err=min(0.10, max(0.05, 1.25*e0))
    Pass_alias=min(0.35, Ts_alias+0.10); Pass_err=min(0.20, Ts_err+0.10)
    return dict(anchor=anchor_row["config"], strong_alias=Ts_alias, strong_err=Ts_err, pass_alias=Pass_alias, pass_err=Pass_err)

def status_with_thresholds(alias_share:float, recon_err:float, thr:Dict[str,float])->str:
    if (alias_share<=thr["strong_alias"]) and (recon_err<thr["strong_err"]): return "STRONG"
    if (alias_share<=thr["pass_alias"])   and (recon_err<thr["pass_err"]):   return "PASS"
    return "FAIL"

def run_ci(output_dir:str="/mnt/data", N:int=64, T:int=96, plot: bool=False)->Dict:
    df_all=get_timeseries_all(N=N, T=T); csv_ts=f"{output_dir}/ca210_timeseries.csv"; df_all.to_csv(csv_ts, index=False)
    rows=[]
    for name in CFG_TABLE.keys():
        sub=df_all[(df_all["name"]==name) & (df_all["t"]>=1)].copy()
        R=sub["budget_residual"].values.astype(float)
        best_w, dec, _ = choose_window(R)
        rows.append(dict(config=name, best_window=best_w, alias_share=dec["alias"], bern_share=dec["bern"], trunc_share=dec["trunc"], recon_err=dec["recon_err"]))
    thr=calibrate_thresholds(rows, anchor="GOLD-id-soft")
    for r in rows: r["status"]=status_with_thresholds(r["alias_share"], r["recon_err"], thr)
    report=pd.DataFrame(rows).sort_values(["status","alias_share"]).reset_index(drop=True)
    csv_report=f"{output_dir}/ca210_S_compliance_report.csv"; report.to_csv(csv_report, index=False)
    fig_path=None
    if plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            plt.figure(figsize=(9,4)); xs=np.arange(len(report)); plt.bar(xs, report["alias_share"].values)
            plt.xticks(xs, report["config"].values, rotation=25, ha="right"); plt.ylabel("alias_share (best window + B6)")
            plt.title("S4/S6 alias share by config"); plt.tight_layout()
            fig_path=f"{output_dir}/S_alias_by_config.png"; plt.savefig(fig_path, dpi=160); plt.close()
        except Exception:
            fig_path=None
    # md
    from datetime import datetime
    md=f"{output_dir}/ca210_full_compliance_report.md"; now=datetime.utcnow().isoformat()+"Z"
    with open(md,"w",encoding="utf-8") as f:
        f.write("# CA210 - Compliance Report\n\n"); f.write(f"_Generated: {now}_\n\n")
        f.write("## Thresholds (auto-calibrated)\n"); f.write(f"- Anchor: `{thr['anchor']}`\n")
        f.write(f"- STRONG: alias <= {thr['strong_alias']:.3f}, err < {thr['strong_err']:.3f}\n")
        f.write(f"- PASS  : alias <= {thr['pass_alias']:.3f}, err < {thr['pass_err']:.3f}\n\n")
        f.write("## Results\n\n"); f.write(report.to_markdown(index=False)); f.write("\n\n")
        if fig_path:
            f.write(f"![alias by config]({fig_path})\n")
    out = {"timeseries_csv":csv_ts, "report_csv":csv_report, "report_md":md, "thresholds":thr}
    if fig_path:
        out["figure"] = fig_path
    return out
