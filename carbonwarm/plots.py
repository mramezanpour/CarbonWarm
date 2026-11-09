"""Figures written by run_all.py."""
from __future__ import annotations

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .carbon import build_carbon  # noqa: E402
from .config import CARBON_SCENARIOS  # noqa: E402

# Okabe-Ito colour-blind-safe palette; the proposed method is the only saturated warm colour
COLORS = {
    "Fixed-KA": "#9E9E9E",
    "Histogram": "#56B4E9",
    "ServerlessLLM": "#0072B2",
    "Breakeven-E": "#E69F00",
    "Carbon-Routing": "#009E73",
    "Carbon-Routing+CS": "#66C2A5",
    "Carbon-Routing+DRAM": "#00543F",
    "MultiState-PD": "#8C6D31",
    "CarbonWarm": "#D55E00",
    "Clairvoyant": "#222222",
    "CW w/o adaptive mu": "#CC79A7",
}
CAT_COLORS = {"idle": "#E69F00", "load": "#56B4E9", "serve": "#0072B2", "cache": "#BDBDBD"}
CAT_LABELS = {"idle": "Idle GPU holding", "load": "Model loading", "serve": "Serving", "cache": "DRAM/SSD holding"}
TIER_COLORS = {"GPU": "#D55E00", "DRAM": "#E69F00", "SSD": "#9E9E9E"}
ZONE_COLORS = {"CAISO": "#E69F00", "DE": "#0072B2", "FR": "#009E73", "PL": "#555555"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.bbox": "tight", "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
W1, W2 = 3.5, 7.2  # single / double column width (inches)


def _err(mean, ci, floor_frac=0.9, upper_cap=None):
    """Asymmetric error bars that never cross zero (needed on log axes with few seeds)."""
    mean = np.asarray(mean, float)
    ci = np.nan_to_num(np.asarray(ci, float))
    lo = np.minimum(ci, floor_frac * np.abs(mean))
    hi = ci if upper_cap is None else np.minimum(ci, np.maximum(upper_cap - mean, 0))
    return np.vstack([lo, hi])


def color(p):
    if p in COLORS:
        return COLORS[p]
    if p.startswith("CW"):
        return "#F4A582"
    return "#777777"


class FigureSet:
    def __init__(self, out_dir):
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.index = []

    def save(self, fig, name, caption):
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(self.out, f"{name}.{ext}"))
        plt.close(fig)
        self.index.append(dict(figure=name, file_pdf=f"figures/{name}.pdf", file_png=f"figures/{name}.png",
                               caption=caption))


def _order(policies, preferred):
    return [p for p in preferred if p in policies] + [p for p in policies if p not in preferred]


def _short(p):
    return p.replace("CW w/o ", "w/o ").replace("CW energy-only", "energy-only")


# ---------------------------------------------------------------------------
def fig_carbon_intensity(fs, preset, data_dir):
    hours = int(min(preset["base"]["duration_s"] / 3600, 7 * 24))
    hours = max(hours, 48)
    from .experiments import carbon_offset_h
    series, src = build_carbon(CARBON_SCENARIOS["mixed"], hours, data_dir, 0.0, 0, carbon_offset_h(0))
    fig, ax = plt.subplots(figsize=(W2, 2.2))
    for z, s in series.items():
        ax.step(np.arange(hours), s.values[:hours], where="post", color=ZONE_COLORS.get(z, None), lw=1.2,
                label=f"{z} ({src.get(z, '?').split(':')[0]})")
    ax.set_xlabel("Hour of simulation (UTC; window of seed 0)")
    ax.set_ylabel("Carbon intensity\n(gCO$_2$/kWh)")
    ax.set_xlim(0, hours)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False)
    fs.save(fig, "fig01_carbon_intensity",
            "Hourly grid carbon intensity of the four zones hosting edge and cloud sites (scenario 'mixed').")


def fig_workload(fs, raw: Dict[str, dict]):
    tr = list(raw)[0]
    d = raw[tr]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2))
    T = d["duration"]
    bins = np.arange(0, T + 600, 600)
    cnt, _ = np.histogram(d["t"], bins)
    axs[0].plot(bins[:-1] / 3600, cnt / 10.0, color="#0072B2", lw=1)
    axs[0].set_xlabel("Time (h)")
    axs[0].set_ylabel("Requests / min")
    axs[0].set_title("(a) Arrival rate (10-min bins)")
    gaps = sorted(d["gaps"].items(), key=lambda kv: -len(kv[1]))[:6]
    for k, (name, g) in enumerate(gaps):
        if len(g) < 2:
            continue
        x = np.sort(g)
        axs[1].plot(np.maximum(x, 0.1), np.arange(1, len(x) + 1) / len(x), lw=1, label=name,
                    color=plt.cm.viridis(k / 6))
    axs[1].axvline(600, color="k", ls="--", lw=0.8)
    axs[1].text(600, 0.04, " 10 min", fontsize=7)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("Inter-arrival gap per model (s)")
    axs[1].set_ylabel("CDF")
    axs[1].set_title("(b) Idle gaps")
    axs[1].legend(fontsize=6, frameon=False, loc="upper left")
    for arr, lab, c in [(d["tin"], "input", "#0072B2"), (d["tout"], "output", "#D55E00")]:
        x = np.sort(arr)
        axs[2].plot(x, np.arange(1, len(x) + 1) / len(x), color=c, lw=1.2, label=lab)
    axs[2].set_xscale("log")
    axs[2].set_xlabel("Tokens per request")
    axs[2].set_ylabel("CDF")
    axs[2].set_title("(c) Token lengths")
    axs[2].legend(frameon=False)
    fig.tight_layout()
    fs.save(fig, "fig02_workload", f"Workload characteristics ({d['source']}): arrival rate, per-model idle "
            "gaps (dashed: common 10-min keep-alive), and token-length distributions.")


def _breakdown(ax, summ, policies, normalize=False):
    cats = ["serve", "idle", "load", "cache"]
    bottom = np.zeros(len(policies))
    vals = {c: np.array([summ.loc[p, f"carbon_{c}_kg_mean"] for p in policies]) for c in cats}
    tot = sum(vals.values())
    for c in cats:
        v = vals[c] / tot * 100 if normalize else vals[c]
        ax.bar(range(len(policies)), v, bottom=bottom, color=CAT_COLORS[c], label=CAT_LABELS[c],
               edgecolor="white", lw=0.5)
        bottom += v
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels([_short(p) for p in policies], rotation=30, ha="right")


def fig_motivation(fs, summ_main, traces, sc0):
    pols = [p for p in ["Fixed-KA", "ServerlessLLM"] if p in set(summ_main.policy)]
    if not pols:
        return
    fig, axs = plt.subplots(1, len(traces), figsize=(min(W2, 2.4 * len(traces) + 1), 2.4), squeeze=False)
    for ax, tr in zip(axs[0], traces):
        s = summ_main[(summ_main.trace == tr) & (summ_main.scenario == sc0)].set_index("policy")
        _breakdown(ax, s, pols, normalize=True)
        ax.set_title(tr)
        ax.set_ylim(0, 100)
    axs[0][0].set_ylabel("Share of operational carbon (%)")
    axs[0][-1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fs.save(fig, "fig03_motivation_breakdown", "Where the carbon of serverless LLM serving goes under "
            "common keep-alive policies: idle GPU holding and model loading versus serving.")


def fig_main_carbon(fs, summ_main, traces, scenarios, policies):
    n = len(traces)
    fig, axs = plt.subplots(n, 1, figsize=(W2, 2.0 * n + 0.4), squeeze=False)
    w = 0.8 / len(policies)
    for ax, tr in zip(axs[:, 0], traces):
        s = summ_main[summ_main.trace == tr]
        for k, p in enumerate(policies):
            sp = s[s.policy == p].set_index("scenario").reindex(scenarios)
            x = np.arange(len(scenarios)) + (k - (len(policies) - 1) / 2) * w
            ax.bar(x, sp["carbon_total_kg_mean"], w, yerr=_err(sp["carbon_total_kg_mean"], sp["carbon_total_kg_ci95"]),
                   color=color(p),
                   label=p, edgecolor="black" if p == "CarbonWarm" else "white", lw=0.6,
                   error_kw=dict(lw=0.6, capsize=1.5))
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios)
        ax.set_ylabel("Carbon (kgCO$_2$e)")
        ax.set_yscale("log")
        ax.set_title(f"Trace: {tr}")
    axs[0, 0].legend(ncol=len(policies), loc="upper center", bbox_to_anchor=(0.5, 1.35), frameon=False)
    fig.tight_layout()
    fs.save(fig, "fig04_main_carbon", "Total operational carbon per policy, trace and carbon scenario "
            "(mean ± 95% CI over seeds; log scale).")


def fig_tradeoff(fs, summ_main, summ_sens, tr0, sc0, policies, slo_target):
    s = summ_main[(summ_main.trace == tr0) & (summ_main.scenario == sc0)].set_index("policy")
    fig, ax = plt.subplots(figsize=(W1 + 0.6, 3.4))
    if summ_sens is not None and len(summ_sens):
        mu = summ_sens[(summ_sens.param == "mu_g_per_violation")].sort_values("value")
        if len(mu):
            ax.plot(mu["slo_attain_pct_mean"], mu["carbon_total_kg_mean"], "-o", ms=3, color=COLORS["CW w/o adaptive mu"],
                    lw=1, label="CW fixed $\\mu$ (sweep)", zorder=2)
            for k, (_, r) in enumerate(mu.iterrows()):
                ax.annotate(f"$\\mu$={r['value']:g}", (r["slo_attain_pct_mean"], r["carbon_total_kg_mean"]),
                            fontsize=6, xytext=(4, -9 if k % 2 else 4), textcoords="offset points",
                            color=COLORS["CW w/o adaptive mu"])
    for p in policies:
        if p not in s.index:
            continue
        r = s.loc[p]
        ax.errorbar(r["slo_attain_pct_mean"], r["carbon_total_kg_mean"],
                    xerr=_err([r["slo_attain_pct_mean"]], [r["slo_attain_pct_ci95"]], 1.0, 100.0),
                    yerr=_err([r["carbon_total_kg_mean"]], [r["carbon_total_kg_ci95"]]), fmt="*" if p == "CarbonWarm" else "o",
                    ms=10 if p == "CarbonWarm" else 5, color=color(p), label=p, lw=0.8, capsize=2, zorder=3)
    ax.axvline(slo_target, color="k", ls="--", lw=0.8)
    ax.set_xlim(right=100.5)
    ax.text(slo_target, 0.02, f"SLO target {slo_target:g}% ", fontsize=7, va="bottom", ha="right",
            transform=ax.get_xaxis_transform())
    ax.set_yscale("log")
    ax.set_xlabel("Interactive requests meeting TTFT SLO (%)")
    ax.set_ylabel("Carbon (kgCO$_2$e)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fs.save(fig, "fig05_carbon_slo_tradeoff", "Carbon versus SLO attainment (lower-right is better). "
            "The line is the Pareto curve of CarbonWarm with a fixed violation price μ.")


def fig_ttft_cdf(fs, q: pd.DataFrame, slo, policies):
    fig, ax = plt.subplots(figsize=(W1 + 0.6, 2.5))
    for p in policies:
        if p not in q:
            continue
        ax.plot(np.maximum(q[p], 1e-3), q["quantile"], color=color(p), lw=2 if p == "CarbonWarm" else 1.1, label=p)
    ax.axvline(slo, color="k", ls="--", lw=0.8)
    ax.text(slo, 0.03, f" SLO {slo:g}s", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Time to first token (s)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.01)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fs.save(fig, "fig06_ttft_cdf", "CDF of time-to-first-token of interactive requests (averaged quantiles over seeds).")


def fig_breakdown_policies(fs, summ_main, tr0, sc0, policies):
    s = summ_main[(summ_main.trace == tr0) & (summ_main.scenario == sc0)].set_index("policy")
    pols = [p for p in policies if p in s.index]
    fig, ax = plt.subplots(figsize=(W1 + 0.8, 2.6))
    _breakdown(ax, s, pols)
    ax.set_ylabel("Carbon (kgCO$_2$e)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    fs.save(fig, "fig07_carbon_breakdown", f"Carbon breakdown per policy ({tr0}, {sc0}).")


def fig_cold(fs, summ_main, raw_main, tr0, sc0, policies):
    d = raw_main[(raw_main.trace == tr0) & (raw_main.scenario == sc0)]
    g = d.groupby("policy")[["cold_req_from_dram", "cold_req_from_ssd", "cold_req_from_remote", "cold_start_pct"]].mean()
    pols = [p for p in policies if p in g.index]
    fig, ax = plt.subplots(figsize=(W1 + 0.6, 2.5))
    bottom = np.zeros(len(pols))
    for col, lab, c in [("cold_req_from_dram", "from DRAM", TIER_COLORS["DRAM"]),
                        ("cold_req_from_ssd", "from SSD", TIER_COLORS["SSD"]),
                        ("cold_req_from_remote", "from remote", "#555555")]:
        v = g.loc[pols, col].values
        ax.bar(range(len(pols)), v, bottom=bottom, color=c, label=lab, edgecolor="white", lw=0.5)
        bottom += v
    for k, p in enumerate(pols):
        ax.text(k, bottom[k], f"{g.loc[p, 'cold_start_pct']:.1f}%", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(pols)))
    ax.set_xticklabels([_short(p) for p in pols], rotation=30, ha="right")
    ax.set_ylabel("Requests that waited for a load")
    ax.legend(frameon=False, fontsize=6.5)
    fs.save(fig, "fig08_cold_starts", "Requests delayed by a cold start, by the tier the model was loaded from "
            "(labels: share of all requests).")


def fig_ablation(fs, summ_abl):
    s = summ_abl.set_index("policy")
    if "CarbonWarm" not in s.index:
        return
    full = s.loc["CarbonWarm", "carbon_total_kg_mean"]
    pols = ["CarbonWarm"] + [p for p in s.index if p != "CarbonWarm"]
    fig, ax = plt.subplots(figsize=(W1 + 0.8, 0.32 * len(pols) + 0.8))
    y = np.arange(len(pols))[::-1]
    v = s.loc[pols, "carbon_total_kg_mean"].values
    e = s.loc[pols, "carbon_total_kg_ci95"].values
    ax.barh(y, v, xerr=_err(v, e, 1.0), color=[COLORS["CarbonWarm"] if p == "CarbonWarm" else "#F4A582" for p in pols],
            edgecolor="white", error_kw=dict(lw=0.6, capsize=1.5))
    xmax = float(np.nanmax(v + np.nan_to_num(e))) * 1.05
    for yy, p, val in zip(y, pols, v):
        d = 100 * (val - full) / full
        slo = s.loc[p, "slo_attain_pct_mean"]
        lab = "full" if p == "CarbonWarm" else f"{d:+.1f}%"
        ax.text(xmax * 1.03, yy, f"{lab:>8s}   {slo:5.1f}%", va="center", fontsize=6.5, family="monospace")
    ax.text(xmax * 1.03, y[0] + 0.75, "Δcarbon    SLO", fontsize=6.5, family="monospace", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([_short(p) for p in pols])
    ax.set_xlabel("Carbon (kgCO$_2$e)")
    ax.set_xlim(0, xmax * 1.55)
    ax.axvline(xmax * 1.01, color="#BDBDBD", lw=0.6)
    fs.save(fig, "fig09_ablation", "Ablation: carbon when one component of CarbonWarm is removed "
            "(change relative to the full method; SLO attainment shown).")


SWEEP_LABELS = {
    "slo_ttft_s": "TTFT SLO (s)", "deferrable_frac": "Deferrable share", "edge_gpus": "GPUs per edge site",
    "carbon_forecast_noise": "Carbon-forecast error ($\\sigma$)", "prediction_noise": "Idle-gap prediction error ($\\sigma$)",
    "slo_target_pct": "SLO target (%)", "target_rate_rps": "Load (requests/s)",
    "p_ctx_scale": "Context power (x calibrated)",
}


def fig_sensitivity(fs, summ_sens, metric, name, ylabel, caption):
    params = [p for p in SWEEP_LABELS if p in set(summ_sens.param)]
    if not params:
        return
    nc = 4
    nr = int(np.ceil(len(params) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(W2, 2.1 * nr + 0.5), squeeze=False)
    for ax, prm in zip(axs.flat, params):
        d = summ_sens[summ_sens.param == prm]
        for p in ["ServerlessLLM", "Carbon-Routing", "Carbon-Routing+DRAM", "CarbonWarm"]:
            dd = d[d.policy == p].sort_values("value")
            if len(dd):
                cap = 100.0 if metric.endswith("pct") else None
                ax.errorbar(dd["value"], dd[f"{metric}_mean"],
                            yerr=_err(dd[f"{metric}_mean"], dd[f"{metric}_ci95"], 0.9, cap), color=color(p),
                            marker="o", ms=3, lw=1.6 if p == "CarbonWarm" else 1, capsize=1.5, label=p)
        ax.set_xlabel(SWEEP_LABELS[prm])
        if metric == "carbon_total_kg":
            ax.set_yscale("log")
    for ax in axs.flat[len(params):]:
        ax.axis("off")
    for r in range(nr):
        axs[r, 0].set_ylabel(ylabel)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, ncol=3, loc="upper center")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.1, hspace=0.6, wspace=0.38)
    fs.save(fig, name, caption)


def fig_timeline(fs, tl: Dict[str, dict], hours_max=72):
    if not tl:
        return
    any_ts = next(iter(tl.values()))
    H = min(len(any_ts["hourly_requests"]) - 1, hours_max)
    fig, axs = plt.subplots(3, 1, figsize=(W2, 4.6), sharex=True)
    x = np.arange(H)
    axs[0].bar(x, any_ts["hourly_requests"][:H], color="#BDBDBD", width=1.0, align="edge", edgecolor="white", lw=0.5)
    axs[0].set_ylabel("Requests / h")
    for p, ts in tl.items():
        tot = sum(ts["hourly_carbon_g"].values())[:H]
        axs[1].step(x, tot, where="post", color=color(p), lw=1.8 if p == "CarbonWarm" else 1, label=p)
    axs[1].set_ylabel("Carbon (gCO$_2$e / h)")
    axs[1].set_yscale("log")
    axs[1].legend(frameon=False, ncol=4, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.3))
    for name, v in any_ts["hourly_intensity"].items():
        if name.startswith("cloud"):
            axs[2].step(x, v[:H], where="post", lw=1, label=name)
    axs[2].set_ylabel("gCO$_2$/kWh")
    axs[2].set_xlabel("Hour")
    axs[2].legend(frameon=False, ncol=3, fontsize=6.5)
    fig.tight_layout()
    fs.save(fig, "fig12_timeline", "Hourly request volume, hourly emissions per policy and grid intensity of the cloud sites.")


def fig_tiers(fs, tl: Dict[str, dict]):
    pols = [p for p in ["ServerlessLLM", "CarbonWarm"] if p in tl and "samples" in tl[p]]
    if not pols:
        return
    fig, axs = plt.subplots(1, len(pols), figsize=(W2, 2.2), sharey=True, squeeze=False)
    for ax, p in zip(axs[0], pols):
        a = tl[p]["samples"]
        t = a[:, 0] / 3600
        ax.stackplot(t, a[:, 1], a[:, 2], colors=[TIER_COLORS["GPU"], TIER_COLORS["DRAM"]],
                     labels=["GPU (incl. KV reserve)", "Host DRAM"], alpha=0.9)
        ax.set_title(p)
        ax.set_xlabel("Hour")
    axs[0][0].set_ylabel("Model weights held (GB)")
    axs[0][-1].legend(frameon=False, fontsize=6.5)
    fs.save(fig, "fig13_tier_occupancy", "Memory tiers occupied by model weights over time (all sites).")


def fig_oracle(fs, raw_or, summ_main, tr0, sc0):
    if raw_or is None or not len(raw_or):
        return
    res_or = (raw_or["carbon_idle_kg"] + raw_or["carbon_load_kg"] + raw_or["carbon_cache_kg"]).mean()
    res_cw = raw_or["ref_residency_kg"].mean()
    bars = [("Clairvoyant", res_or, COLORS["Clairvoyant"]), ("CarbonWarm", res_cw, COLORS["CarbonWarm"])]
    s = summ_main[(summ_main.trace == tr0) & (summ_main.scenario == sc0)].set_index("policy")
    for p in ["Carbon-Routing", "ServerlessLLM", "Breakeven-E", "Fixed-KA"]:
        if p in s.index:
            bars.append((p, s.loc[p, ["carbon_idle_kg_mean", "carbon_load_kg_mean", "carbon_cache_kg_mean"]].sum(), color(p)))
    fig, ax = plt.subplots(figsize=(W1 + 0.4, 2.4))
    ax.bar(range(len(bars)), [b[1] for b in bars], color=[b[2] for b in bars])
    for k, b in enumerate(bars):
        ax.text(k, b[1], f"{b[1] / res_or:.2f}x" if res_or > 0 else "", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Residency carbon (kgCO$_2$e)\nidle + load + cache")
    fs.save(fig, "fig14_clairvoyant_compare", "Residency-related carbon relative to a future-aware heuristic that knows all "
            "future arrivals and the true carbon trace (labels: multiple of that heuristic). The heuristic is a "
            "reference point, not a lower bound.")


def fig_overhead(fs, summ_main, tr0, sc0, policies):
    s = summ_main[(summ_main.trace == tr0) & (summ_main.scenario == sc0)].set_index("policy")
    pols = [p for p in policies if p in s.index]
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    ax.bar(range(len(pols)), s.loc[pols, "decision_overhead_us_mean"], color=[color(p) for p in pols])
    ax.set_xticks(range(len(pols)))
    ax.set_xticklabels([_short(p) for p in pols], rotation=30, ha="right")
    ax.set_ylabel("Mean decision time (µs)")
    fs.save(fig, "fig15_overhead", "Scheduler decision overhead per decision (Python prototype).")


def fig_reduction(fs, red: pd.DataFrame):
    if red is None or not len(red):
        return
    red = red.copy()
    red["col"] = red["trace"] + "\n" + red["scenario"]
    piv = red.pivot_table(index="baseline", columns="col", values="reduction_pct_mean", sort=False)
    fig, ax = plt.subplots(figsize=(min(W2, 1.0 * piv.shape[1] + 2.0), 0.4 * piv.shape[0] + 1.0))
    vmax = np.nanmax(np.abs(piv.values)) if piv.size else 1
    im = ax.imshow(piv.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 * vmax else "black")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, fontsize=6.5)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("Carbon reduction of CarbonWarm (%)")
    fs.save(fig, "fig16_reduction_heatmap", "Carbon reduction of CarbonWarm relative to each baseline "
            "(paired by seed; positive = CarbonWarm emits less).")


def fig_edge_share(fs, summ_main, tr0, scenarios, policies):
    s = summ_main[summ_main.trace == tr0]
    fig, ax = plt.subplots(figsize=(W2, 2.2))
    w = 0.8 / len(policies)
    for k, p in enumerate(policies):
        sp = s[s.policy == p].set_index("scenario").reindex(scenarios)
        x = np.arange(len(scenarios)) + (k - (len(policies) - 1) / 2) * w
        ax.bar(x, sp["edge_share_pct_mean"], w, color=color(p), label=p)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Requests served at edge (%)")
    ax.legend(ncol=len(policies), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25), fontsize=6.5)
    fs.save(fig, "fig17_edge_share", f"Share of requests served at edge sites ({tr0}).")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
STRESS = ["ServerlessLLM", "Carbon-Routing", "Carbon-Routing+DRAM", "MultiState-PD", "CarbonWarm"]


def fig_stress(fs, summ, param, xlabel, name, caption, logx=False):
    if summ is None or not len(summ):
        return
    d = summ[summ.param == param]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.4))
    for p in STRESS:
        dd = d[d.policy == p].sort_values("value")
        if not len(dd):
            continue
        axs[0].plot(dd["value"], dd["carbon_total_kg_mean"], marker="o", ms=3, color=color(p),
                    lw=1.6 if p == "CarbonWarm" else 1, label=p)
        axs[1].plot(dd["value"], dd["slo_attain_pct_mean"], marker="o", ms=3, color=color(p),
                    lw=1.6 if p == "CarbonWarm" else 1)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("Carbon (kgCO$_2$e)")
    axs[1].set_ylabel("SLO attainment (%)")
    axs[1].axhline(95, ls="--", color="black", lw=0.7)
    for ax in axs:
        ax.set_xlabel(xlabel)
        if logx:
            ax.set_xscale("log")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, ncol=5, loc="upper center")
    fig.subplots_adjust(top=0.82, wspace=0.3)
    fs.save(fig, name, caption)


def fig_zipf(fs, red_zipf):
    if red_zipf is None or not len(red_zipf):
        return
    fig, ax = plt.subplots(figsize=(W2, 2.3))
    groups = list(red_zipf.groupby(["trace", "value"], sort=True))
    bases = [b for b in ["ServerlessLLM", "Carbon-Routing", "Carbon-Routing+DRAM"] if b in set(red_zipf.baseline)]
    w = 0.8 / max(len(bases), 1)
    for k, b in enumerate(bases):
        v = [g[g.baseline == b].reduction_pct_mean.mean() for _, g in groups]
        ax.bar(np.arange(len(groups)) + (k - (len(bases) - 1) / 2) * w, v, w, color=color(b), label=f"vs {b}")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{t}\nperm {int(v)}" for (t, v), _ in groups], fontsize=6.5)
    ax.set_ylabel("Carbon reduction of CarbonWarm (%)")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    fs.save(fig, "fig20_zipf_order", "Carbon reduction for five assignments of popularity ranks to models.")


def fig_mu(fs, mu_df):
    if mu_df is None or not len(mu_df):
        return
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.4))
    for tgt, g in mu_df.groupby("slo_target_pct"):
        axs[0].step(g.t_h, g.mu, where="post", lw=1, label=f"target {tgt:g}%")
        axs[1].plot(g.t_h, g.window_attainment_pct, lw=0.8)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("SLO price $\\mu$ (g/violation)")
    axs[1].set_ylabel("Attainment, last 400 requests (%)")
    for ax in axs:
        ax.set_xlabel("Time (h)")
    axs[0].legend(frameon=False, fontsize=7)
    fs.save(fig, "fig21_mu_trajectory", "Trajectory of the adaptive SLO price for three attainment targets.")


def fig_full_boundary(fs, red, red_full):
    if red is None or not len(red) or red_full is None or not len(red_full):
        return
    a = red.groupby("baseline").reduction_pct_mean.mean()
    b = red_full.groupby("baseline").reduction_pct_mean.mean()
    bases = [x for x in a.index if x in b.index]
    fig, ax = plt.subplots(figsize=(W2, 2.3))
    x = np.arange(len(bases))
    ax.bar(x - 0.2, a[bases], 0.4, color="#D55E00", label="policy-dependent carbon")
    ax.bar(x + 0.2, b[bases], 0.4, color="#F4A582", label="full boundary (bare idle, host, PUE)")
    ax.set_xticks(x)
    ax.set_xticklabels(bases, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("Mean carbon reduction (%)")
    ax.legend(frameon=False, fontsize=7)
    fs.save(fig, "fig22_full_boundary", "Carbon reduction of CarbonWarm with the policy-dependent and the full "
            "accounting boundary.")
