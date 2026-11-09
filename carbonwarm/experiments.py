"""Experiment matrix, parallel execution and aggregation."""
from __future__ import annotations

from dataclasses import replace

import math
import time
import traceback
import warnings
from dataclasses import asdict
from multiprocessing import Pool
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import EnergyParams
from .config import SimParams
from .metrics import Q_GRID, run_metrics, timeseries
from .policies import ABLATIONS, ORACLE, POLICIES, make_policy
from .sim import Simulator
from .workload import build_workload

BASELINES = ["Fixed-KA", "Histogram", "ServerlessLLM", "Breakeven-E", "Carbon-Routing",
             "Carbon-Routing+CS", "Carbon-Routing+DRAM", "MultiState-PD"]
OURS = "CarbonWarm"
MAIN_POLICIES = BASELINES + [OURS]
SENS_POLICIES = ["ServerlessLLM", "Carbon-Routing", "Carbon-Routing+DRAM", OURS]
STRESS_POLICIES = ["ServerlessLLM", "Carbon-Routing", "Carbon-Routing+DRAM", "MultiState-PD", OURS]
ZIPF_POLICIES = ["ServerlessLLM", "Carbon-Routing", "Carbon-Routing+DRAM", OURS]
TIMELINE_POLICIES = ["Fixed-KA", "ServerlessLLM", "Carbon-Routing", OURS]

PRESETS: Dict[str, dict] = {
    # tiny run used to check that everything works end to end (minutes on a laptop)
    "quick": dict(
        base=dict(duration_s=6 * 3600, n_models=8, target_rate_rps=0.3, warmup_s=0.0),
        seeds=[0, 1],
        traces=["synthetic"],
        scenarios=["mixed", "solar"],
        sweeps={
            "slo_ttft_s": [2.0, 3.0, 5.0],
            "deferrable_frac": [0.0, 0.2, 0.4],
            "edge_gpus": [0, 2, 4],
            "carbon_forecast_noise": [0.0, 0.3],
            "prediction_noise": [0.0, 0.5],
            "slo_target_pct": [90.0, 95.0, 99.0],
            "target_rate_rps": [0.15, 0.3, 0.6],
            "p_ctx_scale": [0.5, 1.0, 1.5],
        },
        mu_sweep=[0.1, 1.0, 10.0],
        oracle_seeds=[0, 1],
        scale=dict(n_models=[8, 24], target_rate_rps=0.6),
        dram=dict(n_models=24, target_rate_rps=0.6, dram_scale=[1.0, 0.25]),
        zipf=dict(perms=[0, 1], traces=["synthetic"]),
        mu_targets=[90.0, 99.0],
    ),
    # paper configuration (hours of CPU time; use --workers)
    "full": dict(
        base=dict(duration_s=48 * 3600, n_models=12, target_rate_rps=0.5, warmup_s=2 * 3600),
        seeds=[0, 1, 2, 3, 4],
        traces=["burstgpt", "azurellm2024", "af2019"],
        scenarios=["mixed", "solar", "clean-flat", "dirty-flat"],
        sweeps={
            "slo_ttft_s": [1.5, 2.0, 3.0, 5.0, 8.0],
            "deferrable_frac": [0.0, 0.1, 0.2, 0.3, 0.5],
            "edge_gpus": [0, 1, 2, 4, 8],
            "carbon_forecast_noise": [0.0, 0.1, 0.2, 0.4, 0.8],
            "prediction_noise": [0.0, 0.25, 0.5, 1.0, 2.0],
            "slo_target_pct": [90.0, 95.0, 97.5, 99.0],
            "target_rate_rps": [0.1, 0.25, 0.5, 1.0, 2.0],
            "p_ctx_scale": [0.5, 0.75, 1.0, 1.25, 1.5],
        },
        mu_sweep=[0.01, 0.1, 1.0, 10.0, 100.0],
        oracle_seeds=[0, 1, 2, 3, 4],
        # stress tests (BurstGPT, mixed): long tail of models and higher load, DRAM under pressure,
        # different assignment of Zipf ranks to models, trajectory of the SLO price
        scale=dict(n_models=[12, 50, 100, 200], target_rate_rps=1.0),
        dram=dict(n_models=100, target_rate_rps=1.0, dram_scale=[1.0, 0.5, 0.25, 0.125]),
        zipf=dict(perms=[0, 1, 2, 3, 4], traces=["burstgpt", "azurellm2024"]),
        mu_targets=[90.0, 95.0, 99.0],
    ),
}

# where to start reading each real trace (s); BurstGPT's first ~9 days are very sparse
# Each seed uses a different 48-h window of the carbon year (day of year), so results are not tied to
# one holiday week: seed k -> CARBON_DAYS[k % len]. Spread over seasons (Jan, Mar, Jun, Aug, Nov).
CARBON_DAYS = [15, 88, 161, 234, 307]


def carbon_offset_h(seed: int) -> int:
    return int(CARBON_DAYS[int(seed) % len(CARBON_DAYS)] * 24)


TRACE_OFFSETS = {"burstgpt": 16 * 86400.0, "azurellm2024": 0.0, "af2019": 0.0, "synthetic": 0.0}


# ---------------------------------------------------------------------------
def make_jobs(preset: dict, data_dir: str, only=None) -> List[dict]:
    jobs = []
    tr0, sc0 = preset["traces"][0], preset["scenarios"][0]

    def add(exp, policy, trace, scenario, seed, param="", value=np.nan, **ov):
        ov.setdefault("trace_offset_s", TRACE_OFFSETS.get(trace, 0.0))
        ov.setdefault("carbon_offset_h", carbon_offset_h(seed))
        jobs.append(dict(exp=exp, policy=policy, trace=trace, scenario=scenario, seed=seed,
                         param=param, value=value, overrides=ov, data_dir=data_dir,
                         base=preset["base"]))

    want = (lambda e: True) if not only else (lambda e: e in only)
    if want("main"):
        for tr in preset["traces"]:
            for sc in preset["scenarios"]:
                for seed in preset["seeds"]:
                    for p in MAIN_POLICIES:
                        add("main", p, tr, sc, seed)
    if want("ablation"):
        for seed in preset["seeds"]:
            for p in [OURS] + list(ABLATIONS):
                add("ablation", p, tr0, sc0, seed)
    if want("sensitivity"):
        for param, values in preset["sweeps"].items():
            for v in values:
                for seed in preset["seeds"]:
                    for p in SENS_POLICIES:
                        add("sensitivity", p, tr0, sc0, seed, param, v, **{param: v})
        for v in preset["mu_sweep"]:  # Pareto curve of the fixed-mu variant
            for seed in preset["seeds"]:
                add("sensitivity", "CW w/o adaptive mu", tr0, sc0, seed, "mu_g_per_violation", v,
                    mu_g_per_violation=v)
    if only and "ablation_ext" in only:   # ablation where the minor components can matter
        for param, value, sc, ov in [("scenario", "solar", "solar", {}),
                                     ("slo_target_pct", 99.0, sc0, {"slo_target_pct": 99.0}),
                                     ("target_rate_rps", 0.1, sc0, {"target_rate_rps": 0.1})]:
            for seed in preset["seeds"]:
                for p in [OURS] + list(ABLATIONS):
                    add("ablation_ext", p, tr0, sc, seed, param, value if param != "scenario" else np.nan, **ov)
    if only and "tinit" in only:   # sensitivity to the runtime initialisation time t_init
        for v in [1.0, 2.0, 4.0, 7.0, 10.0]:
            for seed in preset["seeds"]:
                for p in SENS_POLICIES:
                    add("sensitivity", p, tr0, sc0, seed, "init_overhead_s", v, init_overhead_s=v)
    if want("clairvoyant") or want("oracle"):
        for seed in preset["oracle_seeds"]:
            add("clairvoyant", ORACLE, tr0, sc0, seed)
    if want("timeline"):
        for p in TIMELINE_POLICIES:
            add("timeline", p, tr0, sc0, 0, record_timeseries=True)
    if want("scale") and "scale" in preset:
        c = preset["scale"]
        for n in c["n_models"]:
            for seed in preset["seeds"]:
                for p in STRESS_POLICIES:
                    add("scale", p, tr0, sc0, seed, "n_models", n, n_models=int(n),
                        target_rate_rps=c["target_rate_rps"])
    if want("dram") and "dram" in preset:
        c = preset["dram"]
        for f in c["dram_scale"]:
            for seed in preset["seeds"]:
                for p in STRESS_POLICIES:
                    add("dram", p, tr0, sc0, seed, "dram_scale", f, n_models=int(c["n_models"]),
                        target_rate_rps=c["target_rate_rps"], dram_scale=f)
    if want("zipf") and "zipf" in preset:
        c = preset["zipf"]
        for tr in c["traces"]:
            for k in c["perms"]:
                for seed in preset["seeds"]:
                    for p in ZIPF_POLICIES:
                        add("zipf", p, tr, sc0, seed, "zipf_perm", k, zipf_perm=int(k))
    if want("mu_trace") and "mu_targets" in preset:
        for tgt in preset["mu_targets"]:
            add("mu_trace", OURS, tr0, sc0, 0, "slo_target_pct", tgt, slo_target_pct=tgt,
                record_timeseries=True)
    return jobs


def _params(job) -> SimParams:
    kw = dict(job["base"])
    kw.update(trace=job["trace"], scenario=job["scenario"], seed=job["seed"], data_dir=job["data_dir"])
    kw.update(job["overrides"])
    if "edge_gpus" in kw:
        kw["edge_gpus"] = int(kw["edge_gpus"])
    if "init_overhead_s" in kw:   # runtime initialisation time of every load (nested in EnergyParams)
        kw["energy"] = replace(kw.get("energy") or EnergyParams(), init_overhead_s=float(kw.pop("init_overhead_s")))
    return SimParams(**kw)


def run_job(job) -> dict:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp = _params(job)
            wl = build_workload(sp)
            if job["policy"] == ORACLE:
                ref = Simulator(sp, wl, make_policy(OURS)).run()
                replay = {"site": ref.r_site.copy(), "disp": ref.r_disp.copy()}
                ref_m = run_metrics(ref)
                sim = Simulator(sp, wl, make_policy(ORACLE, replay=replay)).run()
                m = run_metrics(sim)
                m["ref_carbon_total_kg"] = ref_m["carbon_total_kg"]
                m["ref_residency_kg"] = ref_m["carbon_idle_kg"] + ref_m["carbon_load_kg"] + ref_m["carbon_cache_kg"]
            else:
                sim = Simulator(sp, wl, make_policy(job["policy"])).run()
                m = run_metrics(sim)
            out = {k: job[k] for k in ["exp", "policy", "trace", "scenario", "seed", "param", "value"]}
            out["workload_source"] = wl.source
            out["carbon_sources"] = ";".join(f"{k}={v}" for k, v in sorted(sim.carbon_sources.items()))
            out.update({k: v for k, v in m.items() if not k.startswith("_")})
            out["_ttft_q"] = m["_ttft_q"]
            if job["exp"] in ("timeline", "mu_trace"):
                out["_ts"] = timeseries(sim)
            return out
    except Exception:
        return {"exp": job["exp"], "policy": job["policy"], "error": traceback.format_exc()}


def run_all(jobs: List[dict], workers: int = 1, log=print) -> List[dict]:
    t0 = time.time()
    res = []
    n = len(jobs)
    if workers <= 1:
        it = map(run_job, jobs)
        pool = None
    else:
        pool = Pool(workers)
        it = pool.imap_unordered(run_job, jobs, chunksize=1)
    for k, r in enumerate(it, 1):
        res.append(r)
        if "error" in r:
            log(f"[{k}/{n}] ERROR in {r['exp']}/{r['policy']}:\n{r['error']}")
        elif k % max(1, n // 20) == 0 or k == n:
            el = time.time() - t0
            log(f"[{k}/{n}] {el:6.0f}s elapsed, ~{el / k * (n - k):6.0f}s left")
    if pool is not None:
        pool.close()
        pool.join()
    return res


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}


def t95(df):
    if df <= 0:
        return np.nan
    keys = sorted(_T95)
    for k in keys:
        if df <= k:
            return _T95[k]
    return 1.96


KEY_METRICS = ["carbon_total_kg", "carbon_idle_kg", "carbon_load_kg", "carbon_serve_kg", "carbon_cache_kg",
               "energy_total_kWh", "avg_intensity_g_per_kWh", "slo_attain_pct", "ttft_p50_s", "ttft_p95_s",
               "ttft_p99_s", "cold_start_pct", "deadline_miss_pct", "edge_share_pct", "loads_total",
               "decision_overhead_us", "carbon_per_1k_req_g", "carbon_full_kg", "carbon_full_base_kg",
               "mu_mean", "mu_at_min_pct", "mu_at_max_pct"]


def summarize(df: pd.DataFrame, by: List[str], metrics=KEY_METRICS) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row["n_seeds"] = len(g)
        for mtr in metrics:
            if mtr not in g:
                continue
            x = g[mtr].astype(float).dropna()
            mu = x.mean() if len(x) else np.nan
            sd = x.std(ddof=1) if len(x) > 1 else 0.0
            row[f"{mtr}_mean"] = mu
            row[f"{mtr}_std"] = sd
            row[f"{mtr}_ci95"] = t95(len(x) - 1) * sd / math.sqrt(len(x)) if len(x) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def reductions(df: pd.DataFrame, ours=OURS, metric="carbon_total_kg", exp="main",
               by=("trace", "scenario")) -> pd.DataFrame:
    """Paired (same seed within a group) % reduction of `ours` vs every other policy."""
    rows = []
    d = df[df.exp == exp]
    by = list(by)
    for key, g in d.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        tr, sc = (key + ("", ""))[:2]
        o = g[g.policy == ours].set_index("seed")[metric]
        for p in g.policy.unique():
            if p == ours:
                continue
            b = g[g.policy == p].set_index("seed")[metric]
            common = o.index.intersection(b.index)
            red = 100.0 * (b[common] - o[common]) / b[common]
            n = len(red)
            rows.append(dict(**dict(zip(by, key)), baseline=p, metric=metric, n_seeds=n,
                             reduction_pct_mean=red.mean(),
                             reduction_pct_ci95=t95(n - 1) * red.std(ddof=1) / math.sqrt(n) if n > 1 else np.nan,
                             baseline_mean=b[common].mean(), ours_mean=o[common].mean()))
    return pd.DataFrame(rows)


def ttft_quantiles(results: List[dict], exp="main", trace=None, scenario=None) -> pd.DataFrame:
    acc: Dict[str, list] = {}
    for r in results:
        if r.get("exp") != exp or "error" in r:
            continue
        if trace and r["trace"] != trace or scenario and r["scenario"] != scenario:
            continue
        acc.setdefault(r["policy"], []).append(r["_ttft_q"])
    out = {"quantile": Q_GRID}
    for p, arrs in acc.items():
        out[p] = np.nanmean(np.vstack(arrs), axis=0)
    return pd.DataFrame(out)


def workload_stats(preset: dict, data_dir: str) -> (pd.DataFrame, dict):
    rows, raw = [], {}
    for tr in preset["traces"]:
        kw = dict(preset["base"])
        kw.update(trace=tr, scenario=preset["scenarios"][0], seed=0, data_dir=data_dir,
                  trace_offset_s=TRACE_OFFSETS.get(tr, 0.0), carbon_offset_h=carbon_offset_h(0))
        sp = SimParams(**kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wl = build_workload(sp)
        gaps = []
        per_model = {}
        for m in range(len(wl.models)):
            tm = wl.t[wl.model == m]
            g = np.diff(tm)
            per_model[wl.models[m].name] = g
            gaps.append(g)
            rows.append(dict(trace=tr, source=wl.source, model=wl.models[m].name, size_GB=wl.models[m].size_GB,
                             requests=len(tm), share_pct=100.0 * len(tm) / max(len(wl), 1),
                             gap_median_s=float(np.median(g)) if len(g) else np.nan,
                             gap_p90_s=float(np.percentile(g, 90)) if len(g) else np.nan,
                             frac_gaps_over_10min_pct=100.0 * float((g > 600).mean()) if len(g) else np.nan))
        raw[tr] = dict(t=wl.t, tin=wl.tin, tout=wl.tout, gaps=per_model, duration=sp.duration_s,
                       source=wl.source, cls=wl.cls)
    return pd.DataFrame(rows), raw
