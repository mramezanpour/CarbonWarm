"""Architecture and warmth-ladder figures.  Usage: python design_figs.py figs"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = sys.argv[1]
os.makedirs(os.path.join(OUT, "vector"), exist_ok=True)
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8.5, "savefig.dpi": 600, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02})
INK, MUTE = "#1d2433", "#5b6475"
TIER = {"GPU": "#D55E00", "DRAM": "#F0C05A", "SSD": "#BDBDBD"}
ZONE = {"CAISO": "#E69F00", "DE": "#0072B2", "FR": "#009E73", "PL": "#555555"}


def box(ax, x, y, w, h, fc="white", ec=INK, lw=0.8, r=0.012, ls="-", z=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, a, b, ls="-", c=INK, lw=0.9, rad=0.0, ms=8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, lw=lw, ls=ls, color=c,
                                 connectionstyle=f"arc3,rad={rad}", zorder=5, shrinkA=0, shrinkB=0))


# ============================================================ architecture
fig, ax = plt.subplots(figsize=(6.8, 3.9))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# users
ax.text(0.075, 0.955, "User regions", ha="center", fontsize=9, color=INK)
regs = [("US-West", "CAISO"), ("Germany", "DE"), ("France", "FR"), ("Poland", "PL")]
uy = [0.80, 0.64, 0.48, 0.32]
for (name, z), y in zip(regs, uy):
    box(ax, 0.015, y, 0.12, 0.085, fc="#F4F6F9")
    ax.text(0.075, y + 0.0425, name, ha="center", va="center", fontsize=8)
ax.text(0.075, 0.24, "interactive and\ndeferrable requests", ha="center", va="top", fontsize=7, color=MUTE)

# controller
cx, cy, cw, ch = 0.175, 0.18, 0.31, 0.75
box(ax, cx, cy, cw, ch, fc="#FFF8F3", ec="#D55E00", lw=1.1)
ax.text(cx + cw / 2, cy + ch - 0.035, "CarbonWarm controller", ha="center", fontsize=9.5, color="#B4471A")
mods = [("Warmth-aware placement", "site $j$ for request $i$"),
        ("Carbon-aware deferral", "start time and site"),
        ("Residency ladder", "GPU → DRAM → SSD → remote"),
        ("SLO price controller", "dual variable $\\mu$"),
        ("Carbon forecast", "$\\hat{I}_j$ over 2 h")]
my = [0.745, 0.615, 0.485, 0.355, 0.225]
for (t1, t2), y in zip(mods, my):
    box(ax, cx + 0.02, y, cw - 0.04, 0.105, fc="white", ec="#C9A27E")
    ax.text(cx + cw / 2, y + 0.068, t1, ha="center", va="center", fontsize=8.3)
    ax.text(cx + cw / 2, y + 0.030, t2, ha="center", va="center", fontsize=7.2, color=MUTE)

for y in uy:
    arrow(ax, (0.135, y + 0.0425), (cx, 0.55 + (y - 0.56) * 0.35))

# sites
sx = 0.60
ax.text(sx + 0.17, 0.955, "Edge sites (2 × L4 each)", ha="center", fontsize=9)
ax.text(sx + 0.17, 0.495, "Cloud sites (4 GPUs each)", ha="center", fontsize=9)


def site(x, y, name, z, gpu):
    w, h = 0.088, 0.30
    box(ax, x, y, w, h, fc="white", ec=INK)
    ax.text(x + w / 2, y + h - 0.03, name, ha="center", fontsize=6.8)
    for k, (tier, yy) in enumerate([("GPU", 0.17), ("DRAM", 0.105), ("SSD", 0.04)]):
        box(ax, x + 0.008, y + yy, w - 0.016, 0.055, fc=TIER[tier], ec="none", r=0.006)
        ax.text(x + w / 2, y + yy + 0.0275, gpu if tier == "GPU" else tier, ha="center", va="center", fontsize=6.3,
                color="white" if tier == "GPU" else INK)
    ax.add_patch(plt.Rectangle((x, y - 0.035), w, 0.022, fc=ZONE[z], ec="none"))
    ax.text(x + w / 2, y - 0.024, z, ha="center", va="center", fontsize=6, color="white")


for k, (n, z) in enumerate([("edge-USW", "CAISO"), ("edge-DE", "DE"), ("edge-FR", "FR"), ("edge-PL", "PL")]):
    site(sx - 0.03 + k * 0.1, 0.60, n, z, "L4")
for k, (n, z, g) in enumerate([("cloud-US", "CAISO", "A100"), ("cloud-FR", "FR", "H100"), ("cloud-PL", "PL", "A100")]):
    site(sx - 0.03 + k * 0.1, 0.145, n, z, g)

arrow(ax, (cx + cw, 0.82), (sx - 0.035, 0.82))
ax.text(0.528, 0.835, "route", fontsize=7, color=INK, ha="center")
arrow(ax, (cx + cw, 0.27), (sx - 0.035, 0.27))
ax.text(0.528, 0.285, "route", fontsize=7, color=INK, ha="center")
arrow(ax, (cx + cw, 0.56), (sx - 0.035, 0.70), ls="--", c="#B4471A")
arrow(ax, (cx + cw, 0.52), (sx - 0.035, 0.38), ls="--", c="#B4471A")
ax.text(0.528, 0.54, "demote,\npre-load", fontsize=6.8, color="#B4471A", ha="center", va="center")

# remote store and grid
box(ax, 0.895, 0.145, 0.095, 0.30, fc="#EEF2F7", ec=INK)
ax.text(0.9425, 0.295, "Remote\nobject\nstore", ha="center", va="center", fontsize=7)
arrow(ax, (0.895, 0.25), (0.862, 0.25), lw=0.8)
ax.text(0.5, 0.06, "", fontsize=7)
box(ax, cx, 0.025, cw, 0.09, fc="#F2FAF4", ec="#1B7A3D")
ax.text(cx + cw / 2, 0.07, "Grid carbon intensity $I_j(t)$ of each zone", ha="center", va="center", fontsize=7.4)
arrow(ax, (cx + cw / 2, 0.115), (cx + cw / 2, 0.225), c="#1B7A3D")

fig.savefig(os.path.join(OUT, "architecture.png")); fig.savefig(os.path.join(OUT, "vector", ("architecture.png").replace(".png", ".pdf")))
plt.close(fig)

# ============================================================ warmth ladder (a)
fig, ax = plt.subplots(figsize=(3.3, 2.5))
tiers = ["GPU", "DRAM", "SSD", "Remote"]
hold = [26.3, 5.6, 0.08, 0.0]
load = [0.0, 1.8, 6.3, 14.3]
xs = np.arange(4)
lv = [3.0, 2.1, 1.25, 0.5]
for k in range(4):
    top = lv[k] + 0.55
    ax.add_patch(plt.Rectangle((k - 0.42, 0), 0.84, top, fc=["#D55E00", "#F0C05A", "#BDBDBD", "#EEF2F7"][k], ec=INK, lw=0.7))
    ax.text(k, top - 0.25, tiers[k], ha="center", va="center", fontsize=8.5, color="white" if k == 0 else INK)
    ax.text(k, top - 0.62, f"{hold[k]:g} W", ha="center", va="center", fontsize=7, color="white" if k == 0 else INK)
    ax.text(k, -0.25, f"{load[k]:g} s", ha="center", va="top", fontsize=7)
for k in range(3):
    x0, y0 = k + 0.25, lv[k] + 0.62
    x1, y1 = k + 1.0, lv[k + 1] + 0.62
    arrow(ax, (x0, y0), (x1, y1), c="#B4471A", rad=-0.45)
    ax.text(k + 0.72, lv[k] + 0.78, f"$\\Phi_{k + 1}$", fontsize=7.5, color="#B4471A", ha="left", va="bottom")
ax.annotate("", xy=(0.0, 4.25), xytext=(3.0, 4.25), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#0072B2"))
ax.text(1.5, 4.32, "load on demand", fontsize=7, color="#0072B2", ha="center", va="bottom")
ax.text(-0.75, -0.25, "reload", fontsize=7, va="top", ha="right", color=MUTE)
ax.text(-0.75, 3.55, "hold", fontsize=7, va="top", ha="right", color=MUTE)
ax.set_xlim(-1.05, 3.6); ax.set_ylim(-0.6, 4.7); ax.axis("off")
fig.savefig(os.path.join(OUT, "ladder.png")); fig.savefig(os.path.join(OUT, "vector", ("ladder.png").replace(".png", ".pdf")))
plt.close(fig)

# ============================================================ warmth ladder (b): threshold vs intensity
fig, ax = plt.subplots(figsize=(3.2, 2.5))
I = np.linspace(20, 820, 400)
E_load, dP = (26.3 + 80.0) * 1.8, 26.3 - 5.6
for mv, ls in [(0.0, ":"), (0.01, "--"), (0.1, "-")]:
    T = E_load / dP + mv / (dP * I / 3.6e6)
    ax.plot(I, T, ls, color="#D55E00", lw=1.2, label=f"$\\mu v$ = {mv:g} g")
for z, x in [("FR", 45), ("CAISO", 272), ("DE", 478), ("PL", 724)]:
    ax.axvline(x, color=ZONE[z], lw=0.6, alpha=0.8)
    ax.text(x + 8, 1000, z, fontsize=6.5, color=ZONE[z], rotation=90, va="top")
ax.set_yscale("log"); ax.set_ylim(5, 1200)
ax.set_xlabel("Carbon intensity $I_j$ (g/kWh)"); ax.set_ylabel("GPU idle time before demotion (s)")
ax.grid(color="#E3E3E3", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.86))
fig.savefig(os.path.join(OUT, "threshold.png")); fig.savefig(os.path.join(OUT, "vector", ("threshold.png").replace(".png", ".pdf")))
plt.close(fig)
print("ok")
