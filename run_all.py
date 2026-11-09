#!/usr/bin/env python3
"""Run the experiments, write raw_runs.csv, the Excel report and the figures.

python run_all.py --preset quick
python run_all.py --preset full --workers 8
python run_all.py --preset full --only main ablation
python run_all.py --preset full --only ablation_ext tinit
python run_all.py --plots-only --out results_full
"""
from __future__ import annotations

import argparse
import copy
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

from carbonwarm import experiments as ex
from carbonwarm import plots as pl
from carbonwarm.report import write_workbook


def analyse(results, preset, data_dir, out):
    ok = [r for r in results if "error" not in r]
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in ok])
    tr0, sc0 = preset["traces"][0], preset["scenarios"][0]
    sheets = {}
    S = {}
    by_main = ["trace", "scenario", "policy"]
    for exp, by in [("main", by_main), ("ablation", by_main), ("sensitivity", ["param", "value", "policy"]),
                    ("timeline", by_main)]:
        d = df[df.exp == exp] if len(df) else df
        S[exp] = ex.summarize(d, by) if len(d) else pd.DataFrame()
    red = ex.reductions(df) if len(df) and (df.exp == "main").any() else pd.DataFrame()
    q = ex.ttft_quantiles(ok, "main", tr0, sc0)
    orc = df[df.exp.isin(["clairvoyant", "oracle"])].copy() if len(df) else pd.DataFrame()
    if len(orc):
        orc["clairvoyant_residency_kg"] = orc["carbon_idle_kg"] + orc["carbon_load_kg"] + orc["carbon_cache_kg"]
        orc["cw_over_clairvoyant_residency"] = orc["ref_residency_kg"] / orc["clairvoyant_residency_kg"].replace(0, np.nan)
        orc["cw_over_clairvoyant_total"] = orc["ref_carbon_total_kg"] / orc["carbon_total_kg"]
    wstats, wraw = ex.workload_stats(preset, data_dir)

    # timeline
    tl = {r["policy"]: r["_ts"] for r in ok if r.get("exp") == "timeline" and "_ts" in r}
    hourly, occ = [], []
    for p, ts in tl.items():
        H = len(ts["hourly_requests"])
        d = {"policy": p, "hour": np.arange(H), "requests": ts["hourly_requests"]}
        for c, v in ts["hourly_carbon_g"].items():
            d[f"carbon_{c}_g"] = v[:H]
        for s, v in ts["hourly_intensity"].items():
            d[f"I_{s}"] = v[:H]
        hourly.append(pd.DataFrame(d))
        if "samples" in ts:
            a = ts["samples"]
            occ.append(pd.DataFrame({"policy": p, "t_h": a[:, 0] / 3600, "gpu_GB": a[:, 1], "dram_GB": a[:, 2],
                                     "ssd_GB": a[:, 3], "replicas": a[:, 4], "busy_replicas": a[:, 5]}))

    sheets["Main_Summary"] = S["main"]
    sheets["Main_Reduction"] = red
    sheets["Ablation_Summary"] = S["ablation"]
    sheets["Sensitivity"] = S["sensitivity"]
    sheets["Clairvoyant_Compare"] = orc
    sheets["TTFT_Quantiles"] = q
    sheets["Timeline_Hourly"] = pd.concat(hourly) if hourly else pd.DataFrame()
    sheets["Tier_Occupancy"] = pd.concat(occ) if occ else pd.DataFrame()
    sheets["Workload_Stats"] = wstats
    V4 = {}
    for exp, by in [("scale", ["param", "value", "policy"]), ("dram", ["param", "value", "policy"]),
                    ("zipf", ["trace", "value", "policy"]), ("mu_trace", ["value", "policy"])]:
        d = df[df.exp == exp] if len(df) and "exp" in df else pd.DataFrame()
        V4[exp] = ex.summarize(d, by) if len(d) else pd.DataFrame()
    has = lambda e: len(df) and (df.exp == e).any()
    red_full = ex.reductions(df, metric="carbon_full_kg") if has("main") and "carbon_full_kg" in df else pd.DataFrame()
    red_scale = ex.reductions(df, exp="scale", by=("value",)) if has("scale") else pd.DataFrame()
    red_dram = ex.reductions(df, exp="dram", by=("value",)) if has("dram") else pd.DataFrame()
    red_zipf = ex.reductions(df, exp="zipf", by=("trace", "value")) if has("zipf") else pd.DataFrame()
    mu_rows = []
    for r in ok:
        if r.get("exp") == "mu_trace" and "_ts" in r and "mu_trace" in r["_ts"]:
            a = r["_ts"]["mu_trace"]
            mu_rows.append(pd.DataFrame({"slo_target_pct": r["value"], "t_h": a[:, 0] / 3600.0, "mu": a[:, 1],
                                         "window_attainment_pct": 100.0 * a[:, 2]}))
    mu_df = pd.concat(mu_rows) if mu_rows else pd.DataFrame()
    sheets["Scale_Summary"] = V4["scale"]
    sheets["Scale_Reduction"] = red_scale
    sheets["DRAM_Pressure"] = V4["dram"]
    sheets["DRAM_Reduction"] = red_dram
    sheets["Zipf_Order"] = V4["zipf"]
    sheets["Zipf_Reduction"] = red_zipf
    sheets["Full_Boundary_Reduction"] = red_full
    sheets["Mu_Trajectory"] = mu_df
    sheets["Raw_Runs"] = df

    # figures
    fs = pl.FigureSet(os.path.join(out, "figures"))
    mainp = [p for p in ex.MAIN_POLICIES if len(S["main"]) and p in set(S["main"].policy)]
    steps = [
        ("carbon intensity", lambda: pl.fig_carbon_intensity(fs, preset, data_dir)),
        ("workload", lambda: pl.fig_workload(fs, wraw)),
        ("motivation", lambda: pl.fig_motivation(fs, S["main"], preset["traces"], sc0)),
        ("main carbon", lambda: pl.fig_main_carbon(fs, S["main"], preset["traces"], preset["scenarios"], mainp)),
        ("trade-off", lambda: pl.fig_tradeoff(fs, S["main"], S["sensitivity"], tr0, sc0, mainp,
                                              preset["base"].get("slo_target_pct", 95.0))),
        ("ttft cdf", lambda: pl.fig_ttft_cdf(fs, q, preset["base"].get("slo_ttft_s", 3.0), mainp)),
        ("breakdown", lambda: pl.fig_breakdown_policies(fs, S["main"], tr0, sc0, mainp)),
        ("cold starts", lambda: pl.fig_cold(fs, S["main"], df[df.exp == "main"], tr0, sc0, mainp)),
        ("ablation", lambda: pl.fig_ablation(fs, S["ablation"])),
        ("sensitivity carbon", lambda: pl.fig_sensitivity(fs, S["sensitivity"], "carbon_total_kg",
                                                          "fig10_sensitivity_carbon", "Carbon (kgCO$_2$e)",
                                                          "Sensitivity of total carbon to one parameter at a time.")),
        ("sensitivity slo", lambda: pl.fig_sensitivity(fs, S["sensitivity"], "slo_attain_pct",
                                                       "fig11_sensitivity_slo", "SLO attainment (%)",
                                                       "Sensitivity of SLO attainment to one parameter at a time.")),
        ("timeline", lambda: pl.fig_timeline(fs, tl)),
        ("tiers", lambda: pl.fig_tiers(fs, tl)),
        ("clairvoyant", lambda: pl.fig_oracle(fs, orc, S["main"], tr0, sc0)),
        ("overhead", lambda: pl.fig_overhead(fs, S["main"], tr0, sc0, mainp)),
        ("reduction", lambda: pl.fig_reduction(fs, red)),
        ("edge share", lambda: pl.fig_edge_share(fs, S["main"], tr0, preset["scenarios"], mainp)),
        ("scale", lambda: pl.fig_stress(fs, V4["scale"], "n_models", "Number of models", "fig18_scale",
                                        "Long tail of models at 1 request/s (BurstGPT, mixed).", logx=True)),
        ("dram", lambda: pl.fig_stress(fs, V4["dram"], "dram_scale", "DRAM capacity (fraction of default)",
                                       "fig19_dram_pressure", "DRAM under pressure, 100 models (BurstGPT, mixed).",
                                       logx=True)),
        ("zipf", lambda: pl.fig_zipf(fs, red_zipf)),
        ("mu trajectory", lambda: pl.fig_mu(fs, mu_df)),
        ("full boundary", lambda: pl.fig_full_boundary(fs, red, red_full)),
    ]
    for name, fn in steps:
        try:
            fn()
        except Exception as e:  # keep going; report at the end
            print(f"  ! figure '{name}' failed: {e!r}")
    xlsx = os.path.join(out, "CarbonWarm_results.xlsx")
    write_workbook(xlsx, sheets, preset, fs.index, fs.out)
    df.to_csv(os.path.join(out, "raw_runs.csv"), index=False)
    return S, red, orc, xlsx, fs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="quick", choices=list(ex.PRESETS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["main", "ablation", "sensitivity", "clairvoyant", "oracle", "timeline",
                             "scale", "dram", "zipf", "mu_trace", "ablation_ext", "tinit"])
    ap.add_argument("--seeds", type=int, default=None, help="override number of seeds")
    ap.add_argument("--duration-h", type=float, default=None, help="override simulated hours")
    ap.add_argument("--traces", nargs="*", default=None,
                    help="override traces, e.g. --traces burstgpt azurellm2024 af2019 synthetic")
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="override carbon scenarios: mixed solar clean-flat dirty-flat")
    ap.add_argument("--plots-only", action="store_true", help="re-use <out>/results.pkl")
    a = ap.parse_args()

    preset = copy.deepcopy(ex.PRESETS[a.preset])
    if a.seeds:
        preset["seeds"] = list(range(a.seeds))
        preset["oracle_seeds"] = list(range(a.seeds))
    if a.traces:
        preset["traces"] = a.traces
    if a.scenarios:
        preset["scenarios"] = a.scenarios
    if a.duration_h:
        preset["base"]["duration_s"] = a.duration_h * 3600
    out = a.out or f"results_{a.preset}"
    os.makedirs(out, exist_ok=True)
    pkl = os.path.join(out, "results.pkl")

    if a.plots_only:
        with open(pkl, "rb") as f:
            results, preset = pickle.load(f)
    else:
        jobs = ex.make_jobs(preset, a.data_dir, a.only)
        print(f"{len(jobs)} simulation runs, {a.workers} worker(s)")
        t0 = time.time()
        results = ex.run_all(jobs, a.workers)
        print(f"simulations done in {time.time() - t0:.0f}s")
        with open(pkl, "wb") as f:
            pickle.dump((results, preset), f)

    errors = [r for r in results if "error" in r]
    S, red, orc, xlsx, fs = analyse(results, preset, a.data_dir, out)

    pd.set_option("display.width", 200)
    if len(S["main"]):
        cols = ["trace", "scenario", "policy", "carbon_total_kg_mean", "carbon_total_kg_ci95", "slo_attain_pct_mean",
                "ttft_p95_s_mean", "cold_start_pct_mean", "edge_share_pct_mean"]
        print("\nMain results\n", S["main"][cols].round(3).to_string(index=False))
    if len(red):
        print("\nCarbon reduction of CarbonWarm (%)\n",
              red[["trace", "scenario", "baseline", "reduction_pct_mean", "reduction_pct_ci95"]].round(1).to_string(index=False))
    if len(S["ablation"]):
        print("\nAblation\n", S["ablation"][["policy", "carbon_total_kg_mean", "slo_attain_pct_mean"]].round(3).to_string(index=False))
    if len(orc):
        print(f"\nClairvoyant baseline: CarbonWarm residency carbon = {orc['cw_over_clairvoyant_residency'].mean():.2f}x "
              f"the clairvoyant heuristic (a reference point, not a lower bound)")
    print(f"\n{len(fs.index)} figures -> {fs.out}\nExcel -> {xlsx}")
    if errors:
        print(f"\n{len(errors)} run(s) FAILED (see log above)")
        sys.exit(1)


if __name__ == "__main__":
    main()
