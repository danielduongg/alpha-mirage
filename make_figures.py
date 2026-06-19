"""Figures for alpha-mirage (reads nothing; reruns the experiment)."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from experiment import search, gen_prices, ma_signal, sharpe
plt.rcParams.update({"figure.facecolor":"#070a10","axes.facecolor":"#0f1521","savefig.facecolor":"#070a10",
  "text.color":"#eaf1ff","axes.labelcolor":"#cdd9ef","xtick.color":"#8aa0bf","ytick.color":"#8aa0bf",
  "axes.edgecolor":"#1e2a3e","font.size":11,"axes.titlecolor":"#eaf1ff"})

# 1) in-sample vs out-of-sample Sharpe across 200 random searches (random walk)
xs,ys=[],[]
for s in range(200):
    r=search("randomwalk",seed=s)
    xs.append(r["in_sample_sharpe"]); ys.append(r["out_of_sample_sharpe"])
plt.figure(figsize=(6.4,4.2)); plt.scatter(xs,ys,s=14,c="#ffce54",alpha=.6,edgecolor="none")
plt.axhline(0,color="#8aa0bf",ls="--",alpha=.5); plt.axvline(0,color="#8aa0bf",ls="--",alpha=.5)
import numpy as _n; print("corr(in,out)=",round(float(_n.corrcoef(xs,ys)[0,1]),3))
plt.xlabel("best IN-sample Sharpe (cherry-picked)"); plt.ylabel("OUT-of-sample Sharpe")
plt.title("In-sample 'alpha' does not survive (random walk, 200 searches)")
plt.tight_layout(); plt.savefig("fig_insample_vs_oos.png",dpi=120); plt.close()

# 2) distribution of in-sample Sharpes from one search, best marked
r=gen_prices(1500,"randomwalk",7); cum=np.cumsum(r); half=750
grid=[(f,s) for f in range(2,22,2) for s in range(20,121,5) if f<s]
isr=[]
for f,s in grid:
    pos=ma_signal(cum,f,s); isr.append(sharpe((pos*r)[:half]))
isr=np.array(isr)
plt.figure(figsize=(6.4,4.2)); plt.hist(isr,bins=20,color="#5b8cff",alpha=.8)
plt.axvline(isr.max(),color="#ff5d6c",lw=2,label=f"the 'winner' = {isr.max():.2f}")
plt.xlabel("in-sample Sharpe of each strategy"); plt.ylabel("# strategies"); plt.legend()
plt.title(f"Search {len(grid)} rules on noise → pick the right tail")
plt.tight_layout(); plt.savefig("fig_sharpe_distribution.png",dpi=120); plt.close()

# 3) deflated Sharpe vs number of strategies tried
Ks=[5,10,25,50,100,200,400]; dsr=[]
for K in Ks:
    fa=list(range(2,22,2)); slo=list(range(20,121,5))
    rr=search("randomwalk",seed=7,fasts=fa,slows=slo); dsr.append(rr["deflated_sharpe"])
plt.figure(figsize=(6.4,4.2)); plt.plot(Ks,dsr,"o-",color="#27d08a")
plt.axhline(0.95,color="#ff5d6c",ls="--",label="0.95 significance bar")
plt.xscale("log"); plt.ylim(0,1); plt.xlabel("# strategies searched"); plt.ylabel("Deflated Sharpe"); plt.legend()
plt.title("More searching → harder to clear the bar"); plt.tight_layout(); plt.savefig("fig_dsr_vs_k.png",dpi=120); plt.close()
print("wrote 3 figures")
