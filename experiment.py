"""alpha-mirage — how easy is it to 'discover' a trading strategy that's pure luck?

We search a grid of moving-average crossover rules, pick the best by in-sample
Sharpe, then look at (a) its OUT-OF-SAMPLE Sharpe and (b) the Deflated Sharpe
Ratio (Bailey & Lopez de Prado 2014), which corrects for the number of trials.
Run it on a pure random walk (no edge exists, by construction) AND on a
market-like series. In both cases the in-sample winner looks great and is a mirage.
"""
from __future__ import annotations
import json, numpy as np
from scipy.stats import norm, skew, kurtosis

ANN = 252
def gen_prices(n, kind, seed):
    rng = np.random.default_rng(seed)
    if kind == "randomwalk":          # zero drift -> no edge possible
        r = rng.normal(0, 0.01, n)
    else:                              # market-like: tiny drift + vol clustering
        vol = 0.008*(1+0.5*np.abs(rng.standard_normal(n)).cumsum()/np.sqrt(np.arange(1,n+1)))
        r = rng.normal(0.0003, 0.01, n)
    return r

def ma_signal(prices_cum, fast, slow):
    # position from dual moving-average crossover (long/short), shifted to avoid lookahead
    p = prices_cum
    mf = np.convolve(p, np.ones(fast)/fast, mode="full")[:len(p)]
    ms = np.convolve(p, np.ones(slow)/slow, mode="full")[:len(p)]
    sig = np.where(mf > ms, 1.0, -1.0)
    return np.r_[0.0, sig[:-1]]       # shift: trade on yesterday's signal

def sharpe(r):
    s = r.std()
    return 0.0 if s == 0 else r.mean()/s*np.sqrt(ANN)

def search(kind="randomwalk", seed=1, n=1500, fasts=range(2,22,2), slows=range(20,121,5)):
    r = gen_prices(n, kind, seed)
    cum = np.cumsum(r)
    half = n//2
    grid = [(f,s) for f in fasts for s in slows if f < s]
    in_sr=[]; best=None
    for (f,s) in grid:
        pos = ma_signal(cum, f, s)
        strat = pos*r
        isr = sharpe(strat[:half])
        in_sr.append(isr)
        if best is None or isr > best[0]:
            best=(isr,f,s,strat)
    in_sr=np.array(in_sr)
    bi, f, s, strat = best
    oos = sharpe(strat[half:])                       # out-of-sample Sharpe of the winner
    # Deflated Sharpe Ratio
    K=len(grid); V=in_sr.var(ddof=1)/ANN             # variance of (per-period) SR across trials
    g=0.5772156649
    emax=np.sqrt(V)*((1-g)*norm.ppf(1-1/K)+g*norm.ppf(1-1/(K*np.e)))
    sr_is = bi/np.sqrt(ANN)                           # winner per-period SR (in-sample)
    rr = strat[:half]
    sk=float(skew(rr)); ku=float(kurtosis(rr,fisher=False))
    T=half
    dsr=float(norm.cdf((sr_is-emax)*np.sqrt(T-1)/np.sqrt(max(1e-9,1-sk*sr_is+((ku-1)/4)*sr_is**2))))
    return dict(kind=kind,K=K,best_fast=f,best_slow=s,
                in_sample_sharpe=round(float(bi),2),
                out_of_sample_sharpe=round(float(oos),2),
                deflated_sharpe=round(dsr,3),
                expected_max_sharpe_null=round(float(emax*np.sqrt(ANN)),2))

if __name__=="__main__":
    res={k:search(k,seed=7) for k in ("randomwalk","market")}
    print(json.dumps(res,indent=2))
    # aggregate over many seeds to show it's not a fluke
    agg={}
    for kind in ("randomwalk","market"):
        rows=[search(kind,seed=s) for s in range(30)]
        agg[kind]=dict(mean_in_sample=round(float(np.mean([r['in_sample_sharpe'] for r in rows])),2),
                       mean_out_of_sample=round(float(np.mean([r['out_of_sample_sharpe'] for r in rows])),2),
                       mean_deflated=round(float(np.mean([r['deflated_sharpe'] for r in rows])),3))
    print("AGGREGATE over 30 seeds:"); print(json.dumps(agg,indent=2))
    json.dump(dict(example=res,aggregate=agg),open("results.json","w"),indent=2)
