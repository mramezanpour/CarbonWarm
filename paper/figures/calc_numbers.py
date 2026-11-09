"""Numbers quoted in the paper.  Usage: python calc_numbers.py results/raw_runs.csv"""
import sys, pandas as pd, numpy as np
pd.set_option("display.width",250); pd.set_option("display.max_columns",40); pd.set_option("display.max_rows",300)
d=pd.read_csv(sys.argv[1])
m=d[d.exp=="main"]
T=2.776
def mci(x): x=np.asarray(x,float); return x.mean(), T*x.std(ddof=1)/np.sqrt(len(x))
print("== main means by policy (all cells)")
cols=["carbon_total_kg","energy_total_kWh","avg_intensity_g_per_kWh","slo_attain_pct","ttft_p50_s","ttft_p95_s","ttft_p99_s","cold_start_pct","deadline_miss_pct","edge_share_pct","decision_overhead_us","carbon_idle_kg","carbon_load_kg","carbon_serve_kg","carbon_cache_kg"]
print(m.groupby("policy")[cols].mean().round(3).T)
print("== SLO min/max by policy"); print(m.groupby("policy").slo_attain_pct.agg(["min","max"]).round(2))
print("== CW per trace/scenario")
print(m[m.policy=="CarbonWarm"].groupby(["trace","scenario"])[["carbon_total_kg","energy_total_kWh","slo_attain_pct","ttft_p95_s","cold_start_pct","edge_share_pct","deadline_miss_pct"]].mean().round(2))
print("== SLO by trace, policy"); print(m.groupby(["trace","policy"]).slo_attain_pct.mean().unstack().round(1))
print("== energy by trace, policy"); print(m.groupby(["trace","policy"]).energy_total_kWh.mean().unstack().round(2))
print("== intensity by scenario, policy"); print(m.groupby(["scenario","policy"]).avg_intensity_g_per_kWh.mean().unstack().round(0))
print("== edge share by scenario, policy"); print(m.groupby(["scenario","policy"]).edge_share_pct.mean().unstack().round(1))
p=m.pivot_table(index=["trace","scenario","seed"],columns="policy",values="carbon_total_kg")
rows=[]
for b in [c for c in p.columns if c!="CarbonWarm"]:
  for (tr,sc),g in p.groupby(level=[0,1]):
    r=100*(1-g["CarbonWarm"]/g[b]); mu,ci=mci(r); rows.append((b,tr,sc,mu,ci))
R=pd.DataFrame(rows,columns=["base","trace","sc","red","ci"])
print("== reductions"); print(R.pivot_table(index=["base"],columns=["trace","sc"],values="red").round(1))
print(R.groupby("base").red.agg(["min","max","mean"]).round(1)); print("ci max",R.ci.max().round(1))
print(R.groupby("sc").red.mean().round(1))
pe=m.pivot_table(index=["trace","scenario","seed"],columns="policy",values="energy_total_kWh")
for b in pe.columns:
  if b!="CarbonWarm": r=100*(1-pe.CarbonWarm/pe[b]); print("energy red vs",b, round(r.mean(),1), round(r.min(),1), round(r.max(),1))
# ablation
a=d[d.exp=="ablation"]; pa=a.pivot_table(index="seed",columns="policy",values="carbon_total_kg")
print("== ablation paired change % (mean of ratios) and ratio of means")
for c in pa.columns:
  if c!="CarbonWarm": r=100*(pa[c]/pa.CarbonWarm-1); print(c, round(r.mean(),1), "+-",round(T*r.std()/np.sqrt(5),1), "| ratio of means", round(100*(pa[c].mean()/pa.CarbonWarm.mean()-1),1))
print(a.groupby("policy")[["slo_attain_pct","energy_total_kWh","avg_intensity_g_per_kWh","cold_start_pct","edge_share_pct","carbon_idle_kg","carbon_load_kg","carbon_serve_kg","carbon_cache_kg","deadline_miss_pct"]].mean().round(3))
# clairvoyant
c=d[d.exp=="clairvoyant"]
print("== clairvoyant"); print(c[["seed","carbon_total_kg","carbon_idle_kg","carbon_load_kg","carbon_cache_kg","slo_attain_pct"]])
cw=a[a.policy=="CarbonWarm"].set_index("seed")
print("CW abl", cw[["carbon_total_kg","carbon_idle_kg","carbon_load_kg","carbon_cache_kg"]])
# sensitivity
s=d[d.exp=="sensitivity"]
print("== sensitivity means"); print(s.groupby(["param","value","policy"])[["carbon_total_kg","slo_attain_pct","ttft_p95_s","deadline_miss_pct","edge_share_pct"]].mean().round(3))
print("== timeline"); t=d[d.exp=="timeline"]; print(t[["policy","carbon_total_kg","slo_attain_pct"]])
print("== cold loads by tier main burstgpt mixed"); print(m[(m.trace=="burstgpt")&(m.scenario=="mixed")].groupby("policy")[["cold_start_pct","cold_req_from_dram","cold_req_from_ssd","cold_req_from_remote","loads_total","prewarm_loads"]].mean().round(1))
print(m.groupby("policy")[["cold_start_pct"]].agg(["min","max"]).round(1))
