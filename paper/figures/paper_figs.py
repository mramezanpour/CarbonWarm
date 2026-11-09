"""Evaluation figures, one PNG and one PDF per panel.  Usage: python paper_figs.py results figs"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(OUT, "vector"), exist_ok=True)
os.makedirs(OUT, exist_ok=True)
XL = os.path.join(RES, "CarbonWarm_results.xlsx")
raw = pd.read_csv(os.path.join(RES, "raw_runs.csv"))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#E3E3E3", "grid.linewidth": 0.6,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
C = {"Fixed-KA": "#9E9E9E", "Histogram": "#56B4E9", "ServerlessLLM": "#0072B2", "Breakeven-E": "#E69F00",
     "MultiState-PD": "#8C564B", "Carbon-Routing": "#009E73", "Carbon-Routing+CS": "#7FC97F", "Carbon-Routing+DRAM": "#1B5E20",
     "CarbonWarm": "#D55E00", "Clairvoyant": "#222222", "CW w/o adaptive mu": "#CC79A7"}
POL = ["Fixed-KA", "Histogram", "Breakeven-E", "MultiState-PD", "ServerlessLLM", "Carbon-Routing", "Carbon-Routing+CS",
       "Carbon-Routing+DRAM", "CarbonWarm"]
NP = len(POL)
TR = {"burstgpt": "BurstGPT", "azurellm2024": "Azure LLM", "af2019": "Azure Functions"}
SC = ["mixed", "solar", "clean-flat", "dirty-flat"]
KG = "Carbon (kgCO$_2$e)"
T95 = 2.776


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png")); fig.savefig(os.path.join(OUT, "vector", (name + ".png").replace(".png", ".pdf")))
    plt.close(fig)


def mci(x):
    x = np.asarray(x, float)
    return x.mean(), T95 * x.std(ddof=1) / np.sqrt(len(x))


main = raw[raw.exp == "main"]

# ---------------------------------------------------------------- main carbon, one panel per trace
for tr in TR:
    fig, ax = plt.subplots(figsize=(6.6, 1.9))
    w = 0.095
    for k, p in enumerate(POL):
        mu, ci = zip(*[mci(main[(main.trace == tr) & (main.scenario == sc) & (main.policy == p)].carbon_total_kg) for sc in SC])
        x = np.arange(len(SC)) + (k - (NP - 1) / 2) * w
        ax.bar(x, mu, w, color=C[p], label=p, edgecolor="white", linewidth=0.3, zorder=2)
    ax.set_yscale("log"); ax.set_xticks(range(len(SC))); ax.set_xticklabels(SC)
    ax.set_ylabel(KG); ax.grid(axis="y", zorder=0)
    if tr == "burstgpt":
        ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.2, columnspacing=1.0)
    save(fig, f"main_{tr}")

# ---------------------------------------------------------------- reduction heatmap
p = main.pivot_table(index=["trace", "scenario", "seed"], columns="policy", values="carbon_total_kg")
bases = [b for b in POL if b != "CarbonWarm"]
cols = [(tr, sc) for tr in TR for sc in SC]
M = np.array([[100 * (1 - p.loc[(tr, sc)].CarbonWarm / p.loc[(tr, sc)][b]).mean() for tr, sc in cols] for b in bases])
fig, ax = plt.subplots(figsize=(6.6, 3.1))
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7.5, color="white" if M[i, j] > 60 else "black")
ax.set_yticks(range(len(bases))); ax.set_yticklabels(bases)
ax.set_xticks(range(len(cols))); ax.set_xticklabels([sc for _, sc in cols], rotation=40, ha="right", fontsize=7.5)
for k, tr in enumerate(TR):
    ax.text(4 * k + 1.5, -0.75, TR[tr], ha="center", va="bottom", fontsize=8.5)
    if k:
        ax.axvline(4 * k - 0.5, color="white", lw=2)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("Reduction (%)"); cb.outline.set_visible(False)
save(fig, "heatmap")

# ---------------------------------------------------------------- carbon / SLO trade-off (BurstGPT, mixed)
bm = main[(main.trace == "burstgpt") & (main.scenario == "mixed")]
sens = raw[raw.exp == "sensitivity"]
fig, ax = plt.subplots(figsize=(4.2, 3.0))
for pol in POL:
    d = bm[bm.policy == pol]
    (cx, ex), (cy, ey) = mci(d.slo_attain_pct), mci(d.carbon_total_kg)
    ax.errorbar(cx, cy, xerr=ex, yerr=[[min(ey, cy * 0.95)], [ey]], fmt="*" if pol == "CarbonWarm" else "o",
                ms=11 if pol == "CarbonWarm" else 6, color=C[pol], capsize=2, elinewidth=0.8, label=pol, zorder=3)
mu = sens[sens.param == "mu_g_per_violation"].groupby("value")[["slo_attain_pct", "carbon_total_kg"]].mean()
ax.plot(mu.slo_attain_pct, mu.carbon_total_kg, "-o", color=C["CW w/o adaptive mu"], ms=3, lw=1, label="CarbonWarm, fixed $\\mu$", zorder=2)
ax.axvline(95, ls="--", color="black", lw=0.8); ax.text(94.6, 3.0, "target 95%", rotation=90, ha="right", va="center", fontsize=7)
ax.set_yscale("log"); ax.set_xlabel("SLO attainment (%)"); ax.set_ylabel(KG); ax.grid(zorder=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=6.8)
ins = ax.inset_axes([0.05, 0.06, 0.42, 0.36])
for pol in ["CarbonWarm"]:
    d = bm[bm.policy == pol]
    ins.plot(d.slo_attain_pct.mean(), d.carbon_total_kg.mean(), "*", ms=9, color=C[pol])
ins.plot(mu.slo_attain_pct, mu.carbon_total_kg, "-o", color=C["CW w/o adaptive mu"], ms=3, lw=1)
for v, r in mu.iterrows():
    if v in (0.01, 0.1, 100.0):
        ins.annotate(f"{v:g}", (r.slo_attain_pct, r.carbon_total_kg), xytext=(-12 if v == 0.01 else (3 if v == 0.1 else 3), -9 if v != 100.0 else 3),
                     textcoords="offset points", fontsize=6, color=C["CW w/o adaptive mu"])
ins.set_xlim(98.6, 100.1); ins.set_ylim(0.268, 0.322); ins.tick_params(labelsize=6); ins.grid()
ins.set_title("zoom", fontsize=6, pad=2)
save(fig, "tradeoff")

# ---------------------------------------------------------------- TTFT CDF
q = pd.read_excel(XL, "TTFT_Quantiles")
fig, ax = plt.subplots(figsize=(3.6, 2.6))
for pol in POL:
    ax.plot(q[pol], q["quantile"], color=C[pol], lw=2.0 if pol == "CarbonWarm" else 1.1, label=pol)
ax.axvline(3, ls="--", color="black", lw=0.8); ax.text(3.3, 0.04, "SLO", fontsize=7)
ax.set_xscale("log"); ax.set_xlabel("TTFT (s)"); ax.set_ylabel("CDF"); ax.set_ylim(0, 1.01); ax.grid()
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=6.5)
save(fig, "ttft_cdf")

# ---------------------------------------------------------------- carbon breakdown (BurstGPT, mixed)
cats = [("carbon_serve_kg", "Serving", "#0072B2"), ("carbon_idle_kg", "Idle GPU", "#E69F00"),
        ("carbon_load_kg", "Loading", "#56B4E9"), ("carbon_cache_kg", "DRAM/SSD parking", "#BDBDBD")]
fig, ax = plt.subplots(figsize=(4.6, 2.7))
bot = np.zeros(len(POL))
for col, lab, c in cats:
    v = np.array([bm[bm.policy == pol][col].mean() for pol in POL])
    ax.bar(range(len(POL)), v, 0.65, bottom=bot, color=c, label=lab, zorder=2); bot += v
ax.set_xticks(range(len(POL))); ax.set_xticklabels(POL, rotation=30, ha="right")
ax.set_ylabel(KG); ax.grid(axis="y", zorder=0); ax.legend(frameon=False, fontsize=7)
save(fig, "breakdown")

# ---------------------------------------------------------------- cold starts by source tier (BurstGPT, mixed)
fig, ax = plt.subplots(figsize=(4.6, 2.7))
bot = np.zeros(len(POL))
for col, lab, c in [("cold_req_from_dram", "from DRAM", "#E69F00"), ("cold_req_from_ssd", "from SSD", "#9E9E9E"),
                    ("cold_req_from_remote", "from remote", "#444444")]:
    v = np.array([bm[bm.policy == pol][col].mean() / 1000 for pol in POL])
    ax.bar(range(len(POL)), v, 0.65, bottom=bot, color=c, label=lab, zorder=2); bot += v
for k, pol in enumerate(POL):
    ax.text(k, bot[k] + 0.5, f"{bm[bm.policy == pol].cold_start_pct.mean():.1f}%", ha="center", fontsize=6.3)
ax.set_xticks(range(len(POL))); ax.set_xticklabels(POL, rotation=30, ha="right")
ax.set_ylabel("Requests waiting for a load (×10$^3$)"); ax.grid(axis="y", zorder=0); ax.legend(frameon=False, fontsize=7, loc="upper left")
ax.set_ylim(0, bot.max() * 1.12)
save(fig, "cold_starts")

# ---------------------------------------------------------------- tier occupancy
occ = pd.read_excel(XL, "Tier_Occupancy")
ymax = (occ.gpu_GB + occ.dram_GB).max() * 1.05
for pol in ["ServerlessLLM", "CarbonWarm"]:
    d = occ[occ.policy == pol]
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.stackplot(d.t_h, d.gpu_GB, d.dram_GB, colors=["#D55E00", "#F0C05A"], labels=["GPU memory", "Host DRAM"], lw=0)
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Resident weights (GB)"); ax.set_ylim(0, ymax); ax.set_xlim(0, 48); ax.grid()
    if pol == "ServerlessLLM":
        ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.12))
    save(fig, f"occupancy_{pol}")

# ---------------------------------------------------------------- timeline (seed 0, BurstGPT, mixed)
tl = pd.read_excel(XL, "Timeline_Hourly")
tl["carbon_g"] = tl[["carbon_idle_g", "carbon_load_g", "carbon_serve_g", "carbon_dram_g", "carbon_ssd_g"]].sum(axis=1)
r0 = tl[tl.policy == "CarbonWarm"]
fig, ax = plt.subplots(figsize=(6.6, 1.4))
ax.bar(r0.hour, r0.requests / 1000, 0.9, color="#9E9E9E", zorder=2)
ax.set_ylabel("Requests/h (×10$^3$)"); ax.set_xlim(-0.5, 48); ax.grid(axis="y", zorder=0); ax.set_xlabel("Time (h)")
save(fig, "timeline_requests")
fig, ax = plt.subplots(figsize=(6.6, 1.8))
for pol in ["Fixed-KA", "ServerlessLLM", "Carbon-Routing", "CarbonWarm"]:
    d = tl[(tl.policy == pol) & (tl.hour >= 2)]
    ax.step(d.hour, d.carbon_g, where="post", color=C[pol], lw=1.8 if pol == "CarbonWarm" else 1.0, label=pol)
ax.set_yscale("log"); ax.set_ylabel("Carbon (gCO$_2$e/h)"); ax.set_xlim(-0.5, 48); ax.grid(); ax.set_xlabel("Time (h)")
ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
save(fig, "timeline_carbon")
fig, ax = plt.subplots(figsize=(6.6, 1.5))
for site, lab, c in [("I_cloud-US", "CAISO (US)", "#E69F00"), ("I_edge-DE", "Germany", "#0072B2"), ("I_cloud-FR", "France", "#009E73"), ("I_cloud-PL", "Poland", "#555555")]:
    ax.step(r0.hour, r0[site], where="post", color=c, lw=1.1, label=lab)
ax.set_ylabel("gCO$_2$/kWh"); ax.set_xlabel("Time (h)"); ax.set_xlim(-0.5, 48); ax.grid()
ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
save(fig, "timeline_intensity")

# ---------------------------------------------------------------- edge share (BurstGPT)
bg = main[main.trace == "burstgpt"]
fig, ax = plt.subplots(figsize=(4.6, 2.6))
w = 0.095
for k, pol in enumerate(POL):
    v = [bg[(bg.scenario == sc) & (bg.policy == pol)].edge_share_pct.mean() for sc in SC]
    ax.bar(np.arange(len(SC)) + (k - (NP - 1) / 2) * w, v, w, color=C[pol], label=pol, zorder=2)
    for j, vv in enumerate(v):
        if vv < 0.5:
            ax.text(j + (k - (NP - 1) / 2) * w, 1, "0", ha="center", va="bottom", fontsize=6, color=C[pol])
ax.set_xticks(range(len(SC))); ax.set_xticklabels(SC); ax.set_ylabel("Served at the edge (%)"); ax.set_ylim(0, 100)
ax.grid(axis="y", zorder=0); ax.legend(ncol=3, frameon=False, fontsize=6.2, loc="upper left")
save(fig, "edge_share")

# ---------------------------------------------------------------- ablation (paired, normalised to full CarbonWarm)
ab = raw[raw.exp == "ablation"]
pa = ab.pivot_table(index="seed", columns="policy", values="carbon_total_kg")
sl = ab.groupby("policy").slo_attain_pct.mean()
order = ["CarbonWarm", "CW w/o coord. placement", "CW energy-only", "CW w/o DRAM tier", "CW w/o adaptive mu",
         "CW w/o prediction", "CW w/o defer. forecast", "CW w/o deferral", "CW w/o res. forecast"]
lab = {"CarbonWarm": "Full CarbonWarm", "CW w/o coord. placement": "w/o coordinated placement", "CW energy-only": "energy-only",
       "CW w/o DRAM tier": "w/o DRAM tier", "CW w/o res. forecast": "w/o forecast in residency",
       "CW w/o defer. forecast": "w/o forecast in deferral", "CW w/o adaptive mu": "w/o adaptive $\\mu$",
       "CW w/o prediction": "w/o idle-gap prediction", "CW w/o deferral": "w/o deferral"}
fig, ax = plt.subplots(figsize=(4.6, 2.9))
for k, pol in enumerate(order):
    r = pa[pol] / pa.CarbonWarm
    m_, e_ = mci(r)
    ax.barh(k, m_, 0.6, color="#D55E00" if k == 0 else "#F4A582", zorder=2)
    ax.text(40, k, (f"{100 * (m_ - 1):+.1f}%" if k else "reference") + f"   {sl[pol]:.2f}%", va="center", fontsize=7, family="monospace")
ax.text(40, -1.0, "Δcarbon   SLO", fontsize=7, family="monospace", va="center")
ax.set_yticks(range(len(order))); ax.set_yticklabels([lab[o] for o in order]); ax.invert_yaxis()
ax.set_xscale("log"); ax.set_xlim(0.9, 300); ax.set_xlabel("Carbon relative to full CarbonWarm"); ax.grid(axis="x", zorder=0)
ax.set_ylim(len(order) - 0.5, -1.5)
save(fig, "ablation")

# ---------------------------------------------------------------- sensitivity (paired, normalised to CarbonWarm at the default)
SW = [("slo_ttft_s", "TTFT SLO (s)", 3.0), ("deferrable_frac", "Deferrable share", 0.2), ("edge_gpus", "GPUs per edge site", 2),
      ("target_rate_rps", "Load (requests/s)", 0.5), ("slo_target_pct", "Attainment target (%)", 95.0),
      ("carbon_forecast_noise", "Carbon-forecast error $\\sigma_{\\mathrm{c}}$", 0.0), ("prediction_noise", "Prediction error $\\sigma_{\\mathrm{p}}$", 0.0),
      ("p_ctx_scale", "Context-power scale", 1.0)]
SP = ["ServerlessLLM", "Carbon-Routing", "Carbon-Routing+DRAM", "CarbonWarm"]
for k, (par, xl, dflt) in enumerate(SW):
    d = sens[sens.param == par]
    ref = d[(d.policy == "CarbonWarm") & np.isclose(d.value, dflt)].set_index("seed").carbon_total_kg
    for metric in ["carbon", "slo"]:
        fig, ax = plt.subplots(figsize=(2.15, 1.75))
        for pol in SP:
            xs, ms, es = [], [], []
            for v, g in d[d.policy == pol].groupby("value"):
                g = g.set_index("seed")
                y = g.carbon_total_kg / ref if metric == "carbon" else g.slo_attain_pct
                m_, e_ = mci(y); xs.append(v); ms.append(m_); es.append(e_)
            ax.plot(xs, ms, color=C[pol], marker="o", ms=2.5, lw=1.0, label=pol)
        if metric == "carbon":
            ax.set_yscale("log"); ax.set_ylim(0.4, 10)
            ax.set_ylabel("Normalised carbon")
        else:
            ax.axhline(95, ls="--", color="black", lw=0.6); ax.set_ylim(70, 101); ax.set_ylabel("SLO attainment (%)")
        ax.set_xlabel(xl); ax.grid(); ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(frameon=False, fontsize=5.5, loc="upper right" if metric == "carbon" else "lower right")
        save(fig, f"sens_{metric}_{par}")

# ---------------------------------------------------------------- Clairvoyant comparison (residency carbon, BurstGPT mixed)
cv = pd.read_excel(XL, "Clairvoyant_Compare")
res = lambda d: (d.carbon_idle_kg + d.carbon_load_kg + d.carbon_cache_kg)
vals = {"Clairvoyant": cv.clairvoyant_residency_kg.mean(), "CarbonWarm": cv.ref_residency_kg.mean()}
for pol in ["Carbon-Routing", "Carbon-Routing+CS", "Carbon-Routing+DRAM", "ServerlessLLM", "Breakeven-E", "MultiState-PD", "Fixed-KA"]:
    vals[pol] = res(bm[bm.policy == pol]).mean()
fig, ax = plt.subplots(figsize=(4.6, 2.7))
ks = list(vals)
ax.bar(range(len(ks)), [vals[k] for k in ks], 0.65, color=[C[k] for k in ks], zorder=2)
for i, k in enumerate(ks):
    ax.text(i, vals[k] * 1.08, f"{vals[k] / vals['Clairvoyant']:.2f}×", ha="center", fontsize=7)
ax.set_yscale("log"); ax.set_ylim(0.05, 8); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, rotation=30, ha="right")
ax.set_ylabel("Residency carbon (kgCO$_2$e)"); ax.grid(axis="y", zorder=0)
save(fig, "clairvoyant")

# ---------------------------------------------------------------- decision overhead
fig, ax = plt.subplots(figsize=(4.6, 2.5))
v = [mci(main[main.policy == pol].decision_overhead_us) for pol in POL]
ax.bar(range(len(POL)), [x[0] for x in v], 0.65, color=[C[p_] for p_ in POL], zorder=2)
ax.set_xticks(range(len(POL))); ax.set_xticklabels(POL, rotation=30, ha="right"); ax.set_ylabel("Decision time (µs)"); ax.grid(axis="y", zorder=0)
save(fig, "overhead")

# ---------------------------------------------------------------- v4: scale and DRAM pressure (BurstGPT, mixed, 1 request/s)
STP = ["ServerlessLLM", "Carbon-Routing", "Carbon-Routing+DRAM", "MultiState-PD", "CarbonWarm"]
MK = {"ServerlessLLM": "s", "Carbon-Routing": "^", "Carbon-Routing+DRAM": "D", "MultiState-PD": "v", "CarbonWarm": "o"}
for exp, par, xl, logx in [("scale", "n_models", "Number of models", True), ("dram", "dram_scale", "DRAM capacity scale", True)]:
    d = raw[raw.exp == exp]
    for metric, yl in [("carbon_total_kg", KG), ("slo_attain_pct", "SLO attainment (%)")]:
        fig, ax = plt.subplots(figsize=(3.1, 2.1))
        for pol in STP:
            g = d[d.policy == pol].groupby("value")[metric].mean()
            ax.plot(g.index, g.values, marker=MK[pol], ms=3.5, lw=1.8 if pol == "CarbonWarm" else 1.0, color=C[pol], label=pol)
        if logx:
            ax.set_xscale("log")
        xs = sorted(d.value.unique()); ax.set_xticks(xs); ax.set_xticklabels([f"{v:g}" if exp == "scale" else f"1/{int(round(1 / v))}" if v < 1 else "1" for v in xs])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if metric == "slo_attain_pct":
            ax.axhline(95, ls="--", color="black", lw=0.6); ax.set_ylim(30, 101)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(); ax.tick_params(labelsize=7)
        if exp == "scale" and metric == "carbon_total_kg":
            ax.legend(frameon=False, fontsize=5.8, loc="upper left")
        save(fig, f"stress_{exp}_{metric.split('_')[0]}")

# ---------------------------------------------------------------- v4: SLO price trajectory (hourly aggregates)
mt = pd.read_excel(XL, "Mu_Trajectory")
mt["h"] = np.floor(mt.t_h).astype(int)
MC = {90: "#56B4E9", 95: "#D55E00", 99: "#222222"}
for col, yl, fn in [("mu", "SLO price $\\mu$ (g/violation)", "mu_price"), ("window_attainment_pct", "Attainment (%)", "mu_attain")]:
    fig, ax = plt.subplots(figsize=(3.1, 2.1))
    for tgt, g in mt.groupby("slo_target_pct"):
        if col == "mu":
            y = g.groupby("h").mu.apply(lambda v: float(np.exp(np.log(v).mean())))
        else:
            y = g.groupby("h")[col].mean()
        ax.step(y.index, y.values, where="post", lw=1.0, color=MC.get(int(tgt), "gray"), label=f"target {tgt:g}%")
        if col != "mu":
            ax.axhline(tgt, ls="--", lw=0.6, color=MC.get(int(tgt), "gray"))
    if col == "mu":
        ax.set_yscale("log"); ax.set_ylim(5e-4, 2)
        ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3)
    else:
        ax.set_ylim(80, 100.5)
    ax.set_xlabel("Time (h)"); ax.set_ylabel(yl); ax.set_xlim(0, 50); ax.grid(); ax.tick_params(labelsize=7)
    save(fig, fn)

# ---------------------------------------------------------------- v4: full accounting boundary (all main cells)
fb = main.groupby("policy")[["carbon_full_kg", "carbon_full_base_kg"]].mean().loc[POL]
fig, ax = plt.subplots(figsize=(4.6, 2.6))
x = np.arange(len(POL))
ax.bar(x, fb.carbon_full_base_kg, 0.65, color="#D9D9D9", label="Policy-independent (bare GPU idle, host, PUE)", zorder=2)
ax.bar(x, fb.carbon_full_kg - fb.carbon_full_base_kg, 0.65, bottom=fb.carbon_full_base_kg, color=[C[p_] for p_ in POL], zorder=2)
ax.bar([0], [0], color="#666666", label="Policy-dependent (holding, loading, serving, parking)")
for k, p_ in enumerate(POL):
    ax.text(k, fb.carbon_full_kg[p_] + 0.4, f"{fb.carbon_full_kg[p_]:.1f}", ha="center", fontsize=6.3)
ax.set_xticks(x); ax.set_xticklabels(POL, rotation=30, ha="right"); ax.set_ylabel(KG); ax.set_ylim(0, 56)
ax.grid(axis="y", zorder=0); ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, "full_boundary")

print("done", len(os.listdir(OUT)), "files")
