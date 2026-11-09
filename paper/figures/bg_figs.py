"""Carbon-intensity and motivation figures.  Usage: python bg_figs.py results figs"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(OUT, "vector"), exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "grid.color": "#E3E3E3", "grid.linewidth": 0.6})
tl = pd.read_excel(os.path.join(RES, "CarbonWarm_results.xlsx"), "Timeline_Hourly")
r0 = tl[tl.policy == tl.policy.iloc[0]]
fig, ax = plt.subplots(figsize=(6.4, 2.1))
for col, lab, c in [("I_cloud-PL", "Poland", "#555555"), ("I_edge-DE", "Germany", "#0072B2"), ("I_cloud-US", "California (CAISO)", "#E69F00"), ("I_cloud-FR", "France", "#009E73")]:
    ax.step(r0.hour, r0[col], where="post", color=c, lw=1.3, label=lab)
ax.set_xlim(0, 48); ax.set_ylim(0, 850); ax.set_xlabel("Time (h)"); ax.set_ylabel("Carbon intensity (gCO$_2$/kWh)"); ax.grid()
ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
fig.savefig(os.path.join(OUT, "intensity.png")); fig.savefig(os.path.join(OUT, "vector", ("intensity.png").replace(".png", ".pdf"))); plt.close(fig)

raw = pd.read_csv(os.path.join(RES, "raw_runs.csv"))
m = raw[(raw.exp == "main") & (raw.scenario == "mixed")]
POL = ["Fixed-KA", "Breakeven-E", "ServerlessLLM", "Carbon-Routing"]
TR = [("burstgpt", "BurstGPT"), ("azurellm2024", "Azure LLM"), ("af2019", "Azure Functions")]
cats = [("carbon_serve_kg", "Serving", "#0072B2"), ("carbon_idle_kg", "Idle GPU", "#E69F00"), ("carbon_load_kg", "Loading", "#56B4E9"), ("carbon_cache_kg", "DRAM/SSD parking", "#BDBDBD")]
fig, ax = plt.subplots(figsize=(6.4, 2.5))
xs, labels = [], []
x = 0
for t, tn in TR:
    for p in POL:
        d = m[(m.trace == t) & (m.policy == p)][[c[0] for c in cats]].mean()
        tot = d.sum(); bot = 0
        for c, lab, col in cats:
            v = 100 * d[c] / tot
            ax.bar(x, v, 0.75, bottom=bot, color=col, label=lab if (t == "burstgpt" and p == POL[0]) else None, zorder=2); bot += v
        xs.append(x); labels.append(p); x += 1
    ax.text(x - 2.5, -42, tn, ha="center", fontsize=8.5)
    x += 0.8
ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
ax.set_ylabel("Share of carbon (%)"); ax.set_ylim(0, 100); ax.grid(axis="y", zorder=0)
ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
fig.savefig(os.path.join(OUT, "motivation.png")); fig.savefig(os.path.join(OUT, "vector", ("motivation.png").replace(".png", ".pdf"))); plt.close(fig)
print("ok")
