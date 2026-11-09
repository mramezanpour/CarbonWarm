"""GPU measurement figures.  Usage: python gpu_figs.py results/gpu figs"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
J, OUT = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(OUT, "vector"), exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "grid.color": "#E3E3E3", "grid.linewidth": 0.6})
d = json.load(open(os.path.join(J, "rtx3060_laptop_run2.json"))); card = d["calibration_card"]
MS = {"Qwen/Qwen2.5-0.5B-Instruct": ("Qwen2.5-0.5B", "o"), "Qwen/Qwen2.5-1.5B-Instruct": ("Qwen2.5-1.5B", "s")}
TC = {"DRAM": "#E69F00", "SSD": "#555555"}
fig, ax = plt.subplots(figsize=(3.1, 2.3))
for mk, (lab, mrk) in MS.items():
    md = d["models"][mk]
    for tier, key in [("DRAM", "dram_points"), ("SSD", "ssd_points")]:
        pts = md[key]
        ax.plot([p["size_GB"] for p in pts], [p["s"] for p in pts], mrk, ms=4, mfc="none" if "1.5B" in lab else TC[tier],
                color=TC[tier], label=f"{tier}, {lab}, RTX 3060")
xs = np.logspace(np.log10(0.02), np.log10(4), 50)
xd = xs[xs >= 0.15]
ax.plot(xd, card["t_init_s_DRAM"] + xd / card["bw_DRAM_GBs"], color=TC["DRAM"], lw=0.9)
ax.plot(xs, card["t_init_s_SSD"] + xs / card["bw_SSD_GBs"], color=TC["SSD"], lw=0.9)
T4 = json.load(open(os.path.join(J, "tesla_t4_run1.json")))
t4p = [p for m_ in T4["models"].values() for p in m_.get("dram_points", [])]
ax.plot([p["size_GB"] for p in t4p], [p["s"] for p in t4p], "^", ms=4, color="#0072B2", label="DRAM, T4")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Transferred weights (GB)"); ax.set_ylabel("Load time (s)")
ax.grid(); ax.legend(frameon=False, fontsize=6, loc="upper left"); ax.tick_params(labelsize=7)
fig.savefig(os.path.join(OUT, "gpu_load.png")); fig.savefig(os.path.join(OUT, "vector", ("gpu_load.png").replace(".png", ".pdf"))); plt.close(fig)

fig, ax = plt.subplots(figsize=(3.1, 2.3))
for mk, (lab, mrk) in MS.items():
    dec = d["models"][mk]["decode"]
    ax.plot([x["batch"] for x in dec], [x["J_per_token"] for x in dec], "-" + mrk, ms=4, lw=1.0, color="#0072B2" if "0.5" in lab else "#D55E00", label=lab + ", RTX 3060")
for mk, (lab, mrk) in MS.items():
    dec = T4["models"].get(mk, {}).get("decode", [])
    if dec:
        ax.plot([x["batch"] for x in dec], [x["J_per_token"] for x in dec], "--" + mrk, ms=4, lw=1.0, mfc="none",
                color="#0072B2" if "0.5" in lab else "#D55E00", label=lab + ", T4")
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks([1, 8, 32]); ax.set_xticklabels(["1", "8", "32"])
ax.set_xlabel("Batch size"); ax.set_ylabel("Energy per output token (J)"); ax.grid(); ax.legend(frameon=False, fontsize=6.5)
ax.tick_params(labelsize=7)
fig.savefig(os.path.join(OUT, "gpu_batch.png")); fig.savefig(os.path.join(OUT, "vector", ("gpu_batch.png").replace(".png", ".pdf"))); plt.close(fig)
print("ok")
