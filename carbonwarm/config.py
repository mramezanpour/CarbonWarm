"""Simulator configuration: hardware, sites, models, scenarios and run parameters.

Context power (p_ctx_W) and bare idle power of the A100 and H100 follow "The Model Parking Tax"
(arXiv:2605.23918, Table 2); the L4 value is estimated. Tier bandwidths follow ServerlessLLM (OSDI'24);
prefill/decode speed and dynamic power follow ML.ENERGY and Vidur.
Units: s, W, J, GB, gCO2/kWh, gCO2.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GPUType:
    name: str
    mem_GB: float
    p_ctx_W: float          # idle power while a CUDA context / model is resident
    p_dyn_max_W: float      # extra power at a full batch
    prefill_tps_7b: float   # prefill tokens/s for a 7B model, batch 1
    step_s_7b: float        # decode step time for a 7B model at batch 1 (s)
    step_slope: float       # relative step-time increase per extra request in batch
    max_batch: int          # max concurrent requests per replica


GPU_TYPES: Dict[str, GPUType] = {
    # Edge accelerator (NVIDIA L4 class, 72 W TDP)
    # p_ctx: not measured in the Parking Tax study; estimated as 19 % of TDP, the ratio measured for the
    # GDDR6-based L40S (66.4 W of 350 W), i.e. 0.19 x 72 W = 13.7 W.
    "L4": GPUType("L4", 24.0, 13.7, 50.0, 3000.0, 0.045, 0.015, 16),
    # Cloud accelerators (p_ctx from Parking Tax, Table 2)
    "A100": GPUType("A100", 80.0, 26.3, 300.0, 9000.0, 0.018, 0.008, 64),
    "H100": GPUType("H100", 80.0, 49.9, 450.0, 18000.0, 0.011, 0.006, 64),
}

# Full accounting boundary (used only for the metric carbon_full_kg; the policies never see it):
# bare idle power of every GPU (powered on whether or not it holds a model), host/server base power per
# site and PUE per site kind. Bare idle: Parking Tax Table 2 (A100, H100); L4 estimated (~16 W idle).
BARE_IDLE_W: Dict[str, float] = {"L4": 16.0, "A100": 53.7, "H100": 71.8}
HOST_BASE_W: Dict[str, float] = {"edge": 150.0, "cloud": 400.0}   # CPUs, DRAM refresh, fans, NICs per site
PUE: Dict[str, float] = {"edge": 1.5, "cloud": 1.2}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params_B: float
    size_GB: float   # fp16 weights


MODEL_CATALOG: List[ModelSpec] = [
    ModelSpec("llama3-8b", 8.0, 16.0),
    ModelSpec("mistral-7b", 7.2, 14.5),
    ModelSpec("qwen2.5-7b", 7.6, 15.2),
    ModelSpec("phi3-mini", 3.8, 7.6),
    ModelSpec("llama3.2-3b", 3.2, 6.4),
    ModelSpec("gemma2-9b", 9.2, 18.4),
    ModelSpec("llama2-13b", 13.0, 26.0),
    ModelSpec("qwen2.5-14b", 14.8, 29.6),
    ModelSpec("phi3-medium", 14.0, 28.0),
    ModelSpec("mistral-nemo-12b", 12.2, 24.5),
    ModelSpec("qwen2.5-1.5b", 1.5, 3.1),
    ModelSpec("gemma2-2b", 2.6, 5.2),
    ModelSpec("llama2-7b", 6.7, 13.5),
    ModelSpec("codellama-13b", 13.0, 26.0),
    ModelSpec("yi-9b", 8.8, 17.6),
    ModelSpec("internlm2-7b", 7.7, 15.4),
]


def model_catalog(n: int) -> List[ModelSpec]:
    """First n models. Beyond the 16 base models, the long tail consists of fine-tuned variants of the base
    models (same size and speed, different weights), as hosted by serverless LLM platforms."""
    out = list(MODEL_CATALOG[: min(n, len(MODEL_CATALOG))])
    k = 0
    while len(out) < n:
        b = MODEL_CATALOG[k % len(MODEL_CATALOG)]
        out.append(ModelSpec(f"{b.name}-ft{k // len(MODEL_CATALOG) + 1}", b.params_B, b.size_GB))
        k += 1
    return out


@dataclass(frozen=True)
class SiteSpec:
    name: str
    kind: str              # "edge" | "cloud"
    grid: str              # carbon-intensity zone key
    gpu_type: str
    n_gpus: int
    dram_GB: float         # host-memory checkpoint cache capacity
    ssd_GB: float          # local SSD checkpoint cache capacity
    bw_dram_GBs: float     # DRAM -> GPU
    bw_ssd_GBs: float      # SSD -> GPU
    bw_remote_GBs: float   # remote store -> GPU (via SSD)


@dataclass(frozen=True)
class EnergyParams:
    init_overhead_s: float = 1.0     # engine start from a pre-initialised pool (ServerlessLLM-style)
    p_load_extra_W: float = 80.0     # extra GPU+host power while loading
    p_dram_W_per_GB: float = 0.35    # DRAM holding power
    p_ssd_W_per_GB: float = 0.005    # SSD holding power
    prefill_power_frac: float = 0.9  # fraction of p_dyn_max used during prefill


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
USER_REGIONS = ["US-W", "EU-DE", "EU-FR", "EU-PL"]
REGION_WEIGHTS = [0.30, 0.25, 0.25, 0.20]


def default_sites(edge_gpus: int = 2) -> List[SiteSpec]:
    edges = [
        SiteSpec("edge-USW", "edge", "CAISO", "L4", edge_gpus, 64, 512, 12, 2.0, 0.6),
        SiteSpec("edge-DE", "edge", "DE", "L4", edge_gpus, 64, 512, 12, 2.0, 0.6),
        SiteSpec("edge-FR", "edge", "FR", "L4", edge_gpus, 64, 512, 12, 2.0, 0.6),
        SiteSpec("edge-PL", "edge", "PL", "L4", edge_gpus, 64, 512, 12, 2.0, 0.6),
    ]
    clouds = [
        SiteSpec("cloud-US", "cloud", "CAISO", "A100", 4, 512, 4096, 20, 3.0, 1.2),
        SiteSpec("cloud-FR", "cloud", "FR", "H100", 4, 512, 4096, 20, 3.0, 1.2),
        SiteSpec("cloud-PL", "cloud", "PL", "A100", 4, 512, 4096, 20, 3.0, 1.2),
    ]
    if edge_gpus <= 0:
        edges = []
    return edges + clouds


# one-way-ish RTT (s) from a user region to each site
def default_rtt() -> Dict[Tuple[str, str], float]:
    rtt = {}
    home = {"US-W": "edge-USW", "EU-DE": "edge-DE", "EU-FR": "edge-FR", "EU-PL": "edge-PL"}
    for r in USER_REGIONS:
        for s in ["edge-USW", "edge-DE", "edge-FR", "edge-PL", "cloud-US", "cloud-FR", "cloud-PL"]:
            us_user, us_site = r.startswith("US"), s.endswith("US") or s.endswith("USW")
            if s == home[r]:
                v = 0.008
            elif s.startswith("edge"):
                v = 0.035 if us_user == us_site else 0.140
            else:
                v = 0.030 if us_user == us_site else 0.140
            rtt[(r, s)] = v
    return rtt


HOME_EDGE = {"US-W": "edge-USW", "EU-DE": "edge-DE", "EU-FR": "edge-FR", "EU-PL": "edge-PL"}

# Carbon "scenarios": which grid each zone key is mapped to.
CARBON_SCENARIOS: Dict[str, Dict[str, str]] = {
    "mixed": {"CAISO": "CAISO", "DE": "DE", "FR": "FR", "PL": "PL"},
    "solar": {"CAISO": "CAISO", "DE": "CAISO+9h", "FR": "CAISO+9h", "PL": "CAISO+9h"},
    "clean-flat": {"CAISO": "FR", "DE": "FR", "FR": "FR", "PL": "FR"},
    "dirty-flat": {"CAISO": "PL", "DE": "PL", "FR": "PL", "PL": "PL"},
}


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
@dataclass
class SimParams:
    duration_s: float = 4 * 3600
    warmup_s: float = 0.0               # metrics ignore requests before this time
    n_models: int = 8
    zipf_alpha: float = 1.0
    target_rate_rps: float = 0.25       # mean total request rate (for synthetic / af2019 scaling)
    deferrable_frac: float = 0.2        # share of requests that are deferrable
    defer_window_s: Tuple[float, float] = (3600.0, 4 * 3600.0)
    slo_ttft_s: float = 3.0
    slo_target_pct: float = 95.0        # required share of interactive requests within the TTFT SLO
    green_prewarm: bool = True          # CarbonWarm: pre-warm a greener site that is only cold
    edge_gpus: int = 2
    carbon_forecast_noise: float = 0.0  # multiplicative lognormal sigma on forecasts
    prediction_noise: float = 0.0       # multiplicative lognormal sigma on idle-gap predictions
    mu_g_per_violation: float = 1.0     # initial SLO dual variable (g CO2 charged per SLO violation)
    slo_margin_pct: float = 1.0         # CarbonWarm aims at slo_target_pct + margin internally (safety margin)
    mu_gain: float = 0.25               # step size of the online dual update of mu
    trust_lambda: float = 0.5           # learning-augmented trust parameter (Purohit et al.)
    forecast_horizon_h: int = 2
    record_timeseries: bool = False
    sample_period_s: float = 300.0
    energy: EnergyParams = field(default_factory=EnergyParams)
    trace: str = "synthetic"
    scenario: str = "mixed"
    seed: int = 0
    data_dir: str = "data"
    trace_offset_s: float = 0.0         # where to start in a real trace
    carbon_offset_h: int = 0            # where to start in the carbon CSV files (hours)
    max_tokens_in: int = 8192
    max_tokens_out: int = 4096
    p_ctx_scale: float = 1.0            # multiplies p_ctx of every GPU type (sensitivity study)
    dram_scale: float = 1.0             # multiplies the DRAM checkpoint-cache capacity of every site
    zipf_perm: int = 0                  # 0: catalogue order gets Zipf ranks 1..n; k>0: random permutation k
    record_mu: bool = False             # keep the full trajectory of the SLO price mu
    mu_max: float = 1.0                 # upper bound of the SLO price (anti-windup)
    prewarm_no_evict: bool = True       # green pre-loading only into free GPU memory (no evictions)
    capacity_aware: bool = True         # placement charges the displacement of other models

    def copy(self, **kw) -> "SimParams":
        return replace(self, **kw)
