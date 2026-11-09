"""Per-run metrics."""
from __future__ import annotations

import numpy as np

from .carbon import KWH_J
from .config import BARE_IDLE_W, HOST_BASE_W, PUE
from .sim import TIER_DRAM, TIER_REMOTE, TIER_SSD, Simulator

Q_GRID = np.linspace(0, 1, 201)


def run_metrics(sim: Simulator) -> dict:
    wl, sp = sim.wl, sim.sp
    G, E = sim.tot.G, sim.tot.E
    valid = wl.t >= sp.warmup_s
    inter = valid & (wl.cls == 0)
    defer = valid & (wl.cls == 1)
    served = ~np.isnan(sim.r_done)
    ttft = sim.r_ttft[inter & served & ~np.isnan(sim.r_ttft)]
    n_inter = int(inter.sum())
    # unserved interactive requests count as SLO violations
    slo_ok = int((ttft <= sp.slo_ttft_s).sum())
    late = defer & served & (sim.r_done > wl.deadline)
    unserved_def = defer & ~served
    waited = (sim.r_start - sim.r_disp) > 0.01          # actually waited for a model load
    cold = np.where(waited, sim.r_cold, 0)[valid]
    src = sim.r_src[valid]
    is_edge = np.array([s.is_edge for s in sim.sites])
    sites = sim.r_site[valid & served]
    tot_g = sum(G.values())
    tot_E = sum(E.values())
    n_valid = int(valid.sum())
    m = {
        "n_requests": n_valid,
        "n_interactive": n_inter,
        "n_deferrable": int(defer.sum()),
        "n_unserved": int((valid & ~served).sum()),
        "carbon_total_kg": tot_g / 1000.0,
        "carbon_idle_kg": G["idle"] / 1000.0,
        "carbon_load_kg": G["load"] / 1000.0,
        "carbon_serve_kg": G["serve"] / 1000.0,
        "carbon_cache_kg": (G["dram"] + G["ssd"]) / 1000.0,
        "carbon_per_1k_req_g": tot_g / max(n_valid, 1) * 1000.0,
        "energy_total_kWh": tot_E / KWH_J,
        "energy_idle_kWh": E["idle"] / KWH_J,
        "energy_load_kWh": E["load"] / KWH_J,
        "energy_serve_kWh": E["serve"] / KWH_J,
        "energy_cache_kWh": (E["dram"] + E["ssd"]) / KWH_J,
        "avg_intensity_g_per_kWh": tot_g / max(tot_E / KWH_J, 1e-12),
        "slo_attain_pct": 100.0 * slo_ok / max(n_inter, 1),
        "ttft_mean_s": float(np.mean(ttft)) if len(ttft) else np.nan,
        "ttft_p50_s": float(np.percentile(ttft, 50)) if len(ttft) else np.nan,
        "ttft_p95_s": float(np.percentile(ttft, 95)) if len(ttft) else np.nan,
        "ttft_p99_s": float(np.percentile(ttft, 99)) if len(ttft) else np.nan,
        "cold_start_pct": 100.0 * float((cold > 0).mean()) if n_valid else np.nan,
        "loads_total": int(sum(sim.n_loads.values())),
        "loads_from_dram": int(sim.n_loads[TIER_DRAM]),
        "loads_from_ssd": int(sim.n_loads[TIER_SSD]),
        "loads_from_remote": int(sim.n_loads[TIER_REMOTE]),
        "prewarm_loads": int(sim.n_prewarm),
        "cold_req_from_dram": int(((cold > 0) & (src == TIER_DRAM)).sum()),
        "cold_req_from_ssd": int(((cold > 0) & (src == TIER_SSD)).sum()),
        "cold_req_from_remote": int(((cold > 0) & (src == TIER_REMOTE)).sum()),
        "deadline_miss_pct": 100.0 * float((late | unserved_def).sum()) / max(int(defer.sum()), 1),
        "edge_share_pct": 100.0 * float(is_edge[sites].mean()) if len(sites) else np.nan,
        "decision_overhead_us": sim.decision_ns / max(sim.n_decisions, 1) / 1000.0,
        "sim_wall_s": sim.wall_s,
    }
    if sim.samples:
        a = np.array(sim.samples)
        m["avg_gpu_resident_GB"] = float(a[:, 1].mean())
        m["avg_dram_GB"] = float(a[:, 2].mean())
    m["_ttft_q"] = np.quantile(ttft, Q_GRID) if len(ttft) else np.full(len(Q_GRID), np.nan)
    m.update(full_boundary(sim))
    m.update(mu_stats(sim))
    return m


def full_boundary(sim: Simulator) -> dict:
    """Carbon with a full accounting boundary: policy-dependent carbon of each site plus the bare idle power
    of all its GPUs and its host base power, all multiplied by the site's PUE (config BARE_IDLE_W,
    HOST_BASE_W, PUE). The baseline terms are identical for all policies; they make relative savings smaller
    and show how the result depends on the boundary."""
    sp = sim.sp
    t0, t1 = sp.warmup_s, sim.acc_end
    full, base_total = 0.0, 0.0
    for s in sim.sites:
        p_base = s.spec.n_gpus * BARE_IDLE_W[s.spec.gpu_type] + HOST_BASE_W[s.spec.kind]
        g_base = s.carbon.grams_power(p_base, t0, t1)
        pue = PUE[s.spec.kind]
        full += pue * (sim.G_site[s.idx] + g_base)
        base_total += pue * g_base
    # power-managed fleet: GPUs without a resident model (and their share of the host) are switched off
    pm = sum(PUE[s.spec.kind] * (sim.G_site[s.idx] + sim.G_on[s.idx]) for s in sim.sites)
    pm_base = sum(PUE[s.spec.kind] * sim.G_on[s.idx] for s in sim.sites)
    # only the GPUs are power-managed; the hosts stay on (they keep checkpoints in DRAM)
    pmg_base = sum(PUE[s.spec.kind] * (sim.G_on_bare[s.idx] +
                                       s.carbon.grams_power(HOST_BASE_W[s.spec.kind], t0, t1)) for s in sim.sites)
    pmg = sum(PUE[s.spec.kind] * sim.G_site[s.idx] for s in sim.sites) + pmg_base
    return {"carbon_full_kg": full / 1000.0, "carbon_full_base_kg": base_total / 1000.0,
            "carbon_site_sum_kg": float(sim.G_site.sum()) / 1000.0,
            "carbon_full_pm_kg": pm / 1000.0, "carbon_full_pm_base_kg": pm_base / 1000.0,
            "carbon_full_pmgpu_kg": pmg / 1000.0, "carbon_full_pmgpu_base_kg": pmg_base / 1000.0}


def mu_stats(sim: Simulator) -> dict:
    tr = getattr(sim.policy, "mu_trace", None)
    if not getattr(sim.policy, "adaptive_mu", False) or not tr:
        return {}
    mu = np.array([x[1] for x in tr])
    return {"mu_mean": float(mu.mean()), "mu_final": float(mu[-1]),
            "mu_at_min_pct": 100.0 * float((mu <= 1.0001e-3).mean()),
            "mu_at_max_pct": 100.0 * float((mu >= 0.9999 * sim.sp.mu_max).mean())}


def timeseries(sim: Simulator) -> dict:
    out = {}
    if sim.ts_G is not None:
        H = sim.H
        out["hourly_carbon_g"] = {c: v.copy() for c, v in sim.ts_G.items()}
        out["hourly_intensity"] = {s.spec.name: np.array([s.carbon.I(h * 3600.0) for h in range(H)])
                                   for s in sim.sites}
        t = sim.wl.t
        out["hourly_requests"] = np.bincount((t // 3600).astype(int), minlength=H)[:H]
    if sim.samples:
        out["samples"] = np.array(sim.samples)
    tr = getattr(sim.policy, "mu_trace", None)
    if tr:
        out["mu_trace"] = np.array(tr)   # columns: time (s), mu (g/violation), attainment of the window
    return out
