#!/usr/bin/python3
# =============================================================================
# bespoke_1.py — Residual (idiosyncratic) momentum: does momentum on FACTOR-RESIDUALS beat raw price momentum?
#
# Blitz-Huij-Martens: strip each stock's market exposure, then run momentum on the RESIDUALS — the idiosyncratic,
# "bespoke" part. Hypothesis: residual momentum has a higher information ratio and a MILDER crash than raw momentum,
# because it removes the time-varying beta tilts that cause momentum to detonate (ties to backwards/bulwark). ~90
# names + SPY, monthly, causal, SIP daily total return.
#   RAW MOM      : 12-1 cumulative raw return, cross-sectional long-top / short-bottom tercile, dollar-neutral.
#   RESIDUAL MOM : same, but on residuals r_i − β_i·mkt (rolling-60d β), i.e. momentum in the idiosyncratic return.
# =============================================================================
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bespoke_common import panel, rets, stats, skew_of, beta
import numpy as np
UNIV=["AAPL","MSFT","AMZN","GOOGL","META","NVDA","AVGO","ORCL","CRM","ADBE","CSCO","INTC","AMD","QCOM","TXN","IBM","MU","AMAT","LRCX","ADI","NOW","INTU",
      "JPM","BAC","WFC","GS","MS","C","AXP","V","MA","USB","PNC","SCHW","BLK","BK","COF","JNJ","PFE","MRK","ABT","TMO","LLY","UNH","BMY","ABBV","AMGN","GILD","CVS","CI","MDT","DHR","ISRG",
      "XOM","CVX","COP","SLB","EOG","OXY","PSX","VLO","HD","WMT","COST","TGT","NKE","MCD","SBUX","PG","KO","PEP","DIS","CMCSA","LOW","TJX","EL","CL","KMB","BA","CAT","GE","HON","UNP","UPS","MMM","LMT","RTX","DE","T","VZ","NEE","DUK","SO"]
P, dates = panel(UNIV+["SPY"], minlen=600); syms=[s for s in UNIV if s in P]
M=np.column_stack([P[s] for s in syms]); dr=M[1:]/M[:-1]-1; nd,nn=dr.shape; spy=rets(P["SPY"])
# precompute rolling-60d residuals: resid[t,i] = dr[t,i] - beta_i(trailing)·spy[t]
resid=np.full((nd,nn),np.nan)
for t in range(60,nd):
    sv=np.var(spy[t-60:t])+1e-12
    b=np.array([np.cov(dr[t-60:t,j],spy[t-60:t])[0,1]/sv for j in range(nn)])
    resid[t]=dr[t]-b*spy[t]
cumres=np.cumsum(np.nan_to_num(resid),axis=0)                   # cumulative residual (for 12-1 residual momentum)
def book(mode):
    w=np.zeros(nn); out=np.full(nd,np.nan)
    for t in range(312,nd):
        if (t-312)%21==0:
            if mode=="raw": sig=M[t-21]/M[t-252]-1
            else: sig=cumres[t-21]-cumres[t-252]               # 12-1 residual momentum
            k=max(2,nn//3); o=np.argsort(sig); w=np.zeros(nn); w[o[-k:]]=1.0/k; w[o[:k]]=-1.0/k
        out[t]=w@dr[t]
    return out[312:]
raw=book("raw"); res=book("res"); spc=spy[312:]
print("="*96); print(f"BESPOKE — residual (idiosyncratic) momentum  ({dates[0]} → {dates[-1]}, {nn} names)"); print("="*96)
ra=stats(raw); rs=stats(res)
print(f"\n  book              {'Sharpe':>8}{'CAGR':>8}{'maxDD':>8}{'skew':>7}{'β/SPY':>7}")
print(f"  RAW momentum      {ra['sh']:>+8.2f}{ra['cagr']*100:>+7.1f}%{ra['dd']*100:>+7.0f}%{skew_of(raw):>+7.2f}{beta(raw,spc):>+7.2f}")
print(f"  RESIDUAL momentum {rs['sh']:>+8.2f}{rs['cagr']*100:>+7.1f}%{rs['dd']*100:>+7.0f}%{skew_of(res):>+7.2f}{beta(res,spc):>+7.2f}")
better_sh=rs['sh']>ra['sh']+0.1; milder=skew_of(res)>skew_of(raw)+0.1 or abs(beta(res,spc))<abs(beta(raw,spc))-0.05
print("\n"+"="*96); print("READ:")
print(f"  • SHARPE: residual {rs['sh']:+.2f} vs raw {ra['sh']:+.2f} — residualizing {'IMPROVES the information ratio' if better_sh else 'gives ~the same Sharpe here'}.")
print(f"  • CLEANER BOOK: residual momentum skew {skew_of(res):+.2f} vs raw {skew_of(raw):+.2f}, β/SPY {beta(res,spc):+.2f} vs {beta(raw,spc):+.2f} — {'milder crash / lower beta (the crash cure via residualization)' if milder else 'similar tail here'}.")
if better_sh and milder:
    v=("CONDITIONAL KEEPER: residual momentum beats raw on BOTH the information ratio and the crash profile — stripping the "
       "market exposure before ranking removes the beta tilts that detonate plain momentum (ties to backwards/bulwark/"
       "ballast: build the neutrality IN). A genuine construction upgrade; wire residual momentum as the default momentum "
       "signal in boom/beyond. bespoke_2 = multi-factor residualization (size/value too) + residual-vol scaling.")
elif milder:
    v=("CONDITIONAL: residual momentum doesn't clearly beat raw on Sharpe here, but it's a CLEANER book — lower beta and a "
       f"milder crash (skew {skew_of(res):+.2f} vs {skew_of(raw):+.2f}) — so same return, less market-driven tail. Worth it as the construction for a "
       "market-neutral momentum book (kin to ballast's finding: neutralize in construction, not post-hoc). bespoke_2 = multi-factor residualization.")
else:
    v=("NULL-ish (this window): on large caps 2016-26 residualizing momentum doesn't clearly improve Sharpe or the crash "
       "over raw price momentum — both are weak this decade (see bulwark). The documented residual-momentum edge is "
       "stronger on broad/small-cap universes with real factor residualization = bespoke_2.")
print(f"\n  VERDICT: {v}")
print("  (Residual = r − β·market on a rolling 60d β (market factor only); full Fama-French residualization + residual-vol scaling = bespoke_2. Really a construction v2 of backwards/beyond.)")
