"""Full-boundary figures.  Usage: python fb_figs.py results/raw_runs_power_and_ablation_ext.csv figs"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
F, OUT = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(OUT, "vector"), exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "grid.color": "#E3E3E3", "grid.linewidth": 0.6,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 8})
C = {"Fixed-KA": "#9E9E9E", "Histogram": "#56B4E9", "ServerlessLLM": "#0072B2", "Breakeven-E": "#E69F00",
     "MultiState-PD": "#8C564B", "Carbon-Routing": "#009E73", "Carbon-Routing+CS": "#7FC97F", "Carbon-Routing+DRAM": "#1B5E20",
     "CarbonWarm": "#D55E00"}
POL = list(C)
POL = ["Fixed-KA", "Histogram", "Breakeven-E", "MultiState-PD", "ServerlessLLM", "Carbon-Routing", "Carbon-Routing+CS",
       "Carbon-Routing+DRAM", "CarbonWarm"]
d = pd.read_csv(F); m = d[d.exp == "main"]
for tot, base, fn, ymax in [("carbon_full_kg", "carbon_full_base_kg", "full_boundary", 56),
                            ("carbon_full_pmgpu_kg", "carbon_full_pmgpu_base_kg", "full_boundary_pm", 56)]:
    fb = m.groupby("policy")[[tot, base]].mean().loc[POL]
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    x = np.arange(len(POL))
    ax.bar(x, fb[base], 0.65, color="#D9D9D9", label="Bare GPU idle, host, PUE", zorder=2)
    ax.bar(x, fb[tot] - fb[base], 0.65, bottom=fb[base], color=[C[p] for p in POL], zorder=2)
    ax.bar([0], [0], color="#666666", label="Policy-dependent")
    for k, p in enumerate(POL):
        ax.text(k, fb[tot][p] + ymax * 0.01, f"{fb[tot][p]:.1f}", ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(POL, rotation=40, ha="right"); ax.set_ylabel("Carbon (kgCO$_2$e)")
    ax.set_ylim(0, ymax); ax.grid(axis="y", zorder=0)
    ax.legend(frameon=False, fontsize=6.3, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.savefig(os.path.join(OUT, fn + ".png")); fig.savefig(os.path.join(OUT, "vector", fn + ".pdf")); plt.close(fig)
print("ok")
