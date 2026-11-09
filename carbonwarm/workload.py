"""Workload construction from public traces (or a synthetic generator).

Supported traces (files are looked up under <data_dir>/traces/):
  * "burstgpt"     BurstGPT_without_fails_1.csv (or any BurstGPT*.csv)       github.com/HPMLL/BurstGPT
  * "azurellm2024" AzureLLMInferenceTrace_code_1week.csv / _conv_1week.csv   github.com/Azure/AzurePublicDataset
  * "af2019"       invocations_per_function_md.anon.d01.csv ... d14           (Azure Functions 2019)
                   -> arrival pattern per function mapped to one model; token lengths are
                      sampled from BurstGPT if available
  * "synthetic"    bursty on/off generator (default when files are missing)

A single LLM service trace has very few distinct models, but residency only matters with
many models of different popularity. We therefore map requests to `n_models` models with a
Zipf popularity (by Session ID for BurstGPT so that a conversation stays on one model).
"""
from __future__ import annotations

import glob
import os
import warnings
import zlib
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import REGION_WEIGHTS, USER_REGIONS, ModelSpec, SimParams, model_catalog


@dataclass
class Workload:
    t: np.ndarray          # arrival time (s)
    model: np.ndarray      # model index
    tin: np.ndarray        # input tokens
    tout: np.ndarray       # output tokens
    region: np.ndarray     # user-region index
    cls: np.ndarray        # 0 = interactive, 1 = deferrable
    deadline: np.ndarray   # absolute deadline (deferrable only; inf otherwise)
    models: List[ModelSpec]
    source: str

    def __len__(self):
        return len(self.t)


def _zipf_weights(n: int, alpha: float, rng) -> np.ndarray:
    w = 1.0 / np.arange(1, n + 1) ** alpha
    w = w / w.sum()
    return w[rng.permutation(n)]


def _finish(t, tin, tout, is_api, sp: SimParams, rng, source, model=None, sessions=None) -> Workload:
    order = np.argsort(t, kind="stable")
    t = np.asarray(t, float)[order]
    tin = np.clip(np.asarray(tin, int)[order], 1, sp.max_tokens_in)
    tout = np.clip(np.asarray(tout, int)[order], 1, sp.max_tokens_out)
    is_api = np.asarray(is_api, bool)[order]
    n = len(t)
    models = model_catalog(sp.n_models)
    if model is None:
        w = _zipf_weights(sp.n_models, sp.zipf_alpha, rng)
        if sessions is not None:
            sess = np.asarray(sessions, dtype=object)[order]
            model = np.empty(n, dtype=int)
            cdf = np.cumsum(w)
            rnd = rng.random(n)
            for i in range(n):
                s = sess[i]
                if isinstance(s, str) or (s is not None and not (isinstance(s, float) and np.isnan(s))):
                    u = (zlib.crc32(str(s).encode()) % 1_000_003) / 1_000_003.0
                else:
                    u = rnd[i]
                model[i] = int(np.searchsorted(cdf, u, side="right"))
            model = np.minimum(model, sp.n_models - 1)
        else:
            model = rng.choice(sp.n_models, size=n, p=w)
    else:
        model = np.asarray(model, int)[order]
    if sp.zipf_perm > 0:  # which model receives which popularity rank (robustness study)
        perm = np.random.default_rng(1000 + int(sp.zipf_perm)).permutation(sp.n_models)
        model = perm[model]
    region = rng.choice(len(USER_REGIONS), size=n, p=REGION_WEIGHTS)
    # deferrable: drawn from API-type requests first, so that the overall share = deferrable_frac
    api_share = max(is_api.mean(), 1e-9) if n else 0.0
    p_api = min(1.0, sp.deferrable_frac / api_share) if api_share > 0 else 0.0
    cls = np.where(is_api & (rng.random(n) < p_api), 1, 0)
    if api_share * 1.0 < sp.deferrable_frac:  # not enough API requests: top up from the rest
        extra = (sp.deferrable_frac - cls.mean()) / max(1 - cls.mean(), 1e-9)
        cls = np.where((cls == 0) & (rng.random(n) < extra), 1, cls)
    lo, hi = sp.defer_window_s
    deadline = np.where(cls == 1, t + rng.uniform(lo, hi, n), np.inf)
    return Workload(t, model.astype(int), tin, tout, region.astype(int), cls.astype(int),
                    deadline, models, source)


def _rescale(n, rate, target, rng, jitter_s=60.0):
    """Bring a real trace to `target` requests/s.

    * rate > target : random thinning (each request kept with prob target/rate)
    * rate < target : each request is replicated floor(f) or ceil(f) times (f = target/rate);
      copies are shifted by U(0, jitter_s) seconds. Copies keep the original's session/model,
      so the real burstiness and per-model popularity of the trace are preserved.
    Returns (row indices with repetition, time shift per row).
    """
    idx = np.arange(n)
    if target is None or target <= 0 or rate <= 0 or abs(rate - target) / target < 1e-9:
        return idx, np.zeros(n)
    f = target / rate
    if f < 1.0:
        keep = idx[rng.random(n) < f]
        return keep, np.zeros(len(keep))
    copies = np.floor(f).astype(int) + (rng.random(n) < (f - np.floor(f))).astype(int)
    rows = np.repeat(idx, copies)
    first = np.ones(len(rows), bool)
    first[1:] = rows[1:] != rows[:-1]            # first copy of each request stays at its real time
    shift = np.where(first, 0.0, rng.uniform(0.0, jitter_s, len(rows)))
    return rows, shift


# ---------------------------------------------------------------------------
def synthetic(sp: SimParams, rng) -> Workload:
    """Bursty on/off (MMPP-like) arrivals per model, Zipf popularity, lognormal tokens."""
    n = sp.n_models
    w = _zipf_weights(n, sp.zipf_alpha, rng)
    T = sp.duration_s
    ts, ms = [], []
    for m in range(n):
        mean_rate = sp.target_rate_rps * w[m]
        on_mean = 15 * 60.0
        # unpopular models are idle for long stretches
        off_mean = on_mean * (1.0 / max(w[m] * n, 0.05)) * 1.5
        duty = on_mean / (on_mean + off_mean)
        on_rate = mean_rate / duty
        t = rng.exponential(off_mean) * rng.random()
        while t < T:
            on = rng.exponential(on_mean)
            k = rng.poisson(on_rate * on)
            if k:
                arr = t + np.sort(rng.random(k)) * on
                arr = arr[arr < T]
                ts.append(arr)
                ms.append(np.full(len(arr), m))
            t += on + rng.exponential(off_mean)
    t = np.concatenate(ts) if ts else np.zeros(0)
    model = np.concatenate(ms) if ms else np.zeros(0, int)
    N = len(t)
    tin = rng.lognormal(np.log(600), 1.0, N).astype(int)
    tout = rng.lognormal(np.log(220), 0.9, N).astype(int)
    is_api = rng.random(N) < 0.35
    return _finish(t, tin, tout, is_api, sp, rng, "synthetic", model=model)


def _tokens_pool(data_dir: str):
    f = _find(data_dir, ["BurstGPT_without_fails_1.csv", "BurstGPT*.csv"])
    if not f:
        return None
    df = pd.read_csv(f, usecols=["Request tokens", "Response tokens"])
    df = df[(df["Response tokens"] > 0) & (df["Request tokens"] > 0)]
    return df["Request tokens"].values, df["Response tokens"].values


def _find(data_dir, patterns) -> Optional[str]:
    for p in patterns:
        hits = sorted(glob.glob(os.path.join(data_dir, "traces", p)))
        if hits:
            return hits[0]
    return None


def burstgpt(sp: SimParams, rng) -> Optional[Workload]:
    f = _find(sp.data_dir, ["BurstGPT_without_fails_1.csv", "BurstGPT_without_fails_*.csv", "BurstGPT*.csv"])
    if not f:
        return None
    df = pd.read_csv(f)
    df = df[df["Response tokens"] > 0]
    t0 = df["Timestamp"].min() + sp.trace_offset_s
    df = df[(df["Timestamp"] >= t0) & (df["Timestamp"] < t0 + sp.duration_s)]
    t = df["Timestamp"].values - t0
    rows, shift = _rescale(len(t), len(t) / sp.duration_s, sp.target_rate_rps, rng)
    df, t = df.iloc[rows], t[rows] + shift
    inside = t < sp.duration_s
    df, t = df[inside], t[inside]
    is_api = df["Log Type"].astype(str).str.contains("API", case=False).values
    sess = df["Session ID"].values if "Session ID" in df.columns else None
    return _finish(t, df["Request tokens"].values, df["Response tokens"].values, is_api,
                   sp, rng, f"burstgpt:{os.path.basename(f)}", sessions=sess)


def _read_window(path, t0_off, dur, chunksize=1_000_000):
    """Stream a (possibly multi-GB) Azure LLM trace and keep only the simulated window."""
    parts, start = [], None
    for ch in pd.read_csv(path, chunksize=chunksize):
        ts = pd.to_datetime(ch["TIMESTAMP"], utc=True, format="ISO8601")
        if start is None:
            start = ts.iloc[0]
        sec = (ts - start).dt.total_seconds().values
        m = (sec >= t0_off) & (sec < t0_off + dur)
        if m.any():
            parts.append(pd.DataFrame({"t": sec[m] - t0_off, "tin": ch["ContextTokens"].values[m],
                                       "tout": ch["GeneratedTokens"].values[m]}))
        if sec[-1] >= t0_off + dur:
            break
    return pd.concat(parts, ignore_index=True) if parts else None


def azurellm2024(sp: SimParams, rng) -> Optional[Workload]:
    """Azure LLM inference traces (2024 one-week files preferred, 2023 one-day files accepted)."""
    parts = []
    for kind, pats in [("code", ["AzureLLMInferenceTrace_code_1week.csv", "AzureLLMInferenceTrace_code*.csv"]),
                       ("conv", ["AzureLLMInferenceTrace_conv_1week.csv", "AzureLLMInferenceTrace_conv*.csv"])]:
        f = _find(sp.data_dir, pats)
        if f:
            d = _read_window(f, sp.trace_offset_s, sp.duration_s)
            if d is not None:
                d["is_api"] = kind == "code"
                parts.append(d)
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    t = df["t"].values
    rows, shift = _rescale(len(t), len(t) / sp.duration_s, sp.target_rate_rps, rng)
    df = df.iloc[rows].copy()
    df["t"] = t[rows] + shift
    df = df[df["t"] < sp.duration_s]
    return _finish(df["t"].values, df["tin"].values, df["tout"].values, df["is_api"].values,
                   sp, rng, "azurellm2024")


def af2019(sp: SimParams, rng) -> Optional[Workload]:
    files = sorted(glob.glob(os.path.join(sp.data_dir, "traces", "invocations_per_function_md.anon.d*.csv")))
    if not files:
        return None
    days = int(np.ceil(sp.duration_s / 86400.0))
    d0 = pd.read_csv(files[0])
    minute_cols = [c for c in d0.columns if c.isdigit()]
    tot = d0[minute_cols].sum(axis=1)
    active = (d0[minute_cols] > 0).sum(axis=1)
    cand = d0[(tot >= 100) & (tot <= 20000) & (active >= 20)]
    if len(cand) < sp.n_models:
        cand = d0.assign(_tot=tot).nlargest(sp.n_models, "_tot")
    pick = cand.sample(sp.n_models, random_state=int(rng.integers(1 << 30)))
    keys = list(zip(pick["HashOwner"], pick["HashApp"], pick["HashFunction"]))
    counts = []
    for d in range(days):
        f = files[d % len(files)]
        dd = d0 if d % len(files) == 0 else pd.read_csv(f)
        dd = dd.set_index(["HashOwner", "HashApp", "HashFunction"])
        rows = [dd.loc[k, minute_cols].values.astype(float) if k in dd.index else np.zeros(1440) for k in keys]
        counts.append(np.vstack([r if r.ndim == 1 else r[0] for r in rows]))
    C = np.hstack(counts)[:, : int(np.ceil(sp.duration_s / 60))]
    scale = sp.target_rate_rps * 60.0 / max(C.sum(axis=0).mean(), 1e-9)
    ts, ms = [], []
    for m in range(sp.n_models):
        k = rng.poisson(C[m] * scale)
        for j in np.nonzero(k)[0]:
            ts.append(60.0 * j + rng.random(k[j]) * 60.0)
            ms.append(np.full(k[j], m))
    t = np.concatenate(ts)
    model = np.concatenate(ms)
    keep = t < sp.duration_s
    t, model = t[keep], model[keep]
    pool = _tokens_pool(sp.data_dir)
    if pool is not None:
        idx = rng.integers(0, len(pool[0]), len(t))
        tin, tout = pool[0][idx], pool[1][idx]
    else:
        tin = rng.lognormal(np.log(600), 1.0, len(t)).astype(int)
        tout = rng.lognormal(np.log(220), 0.9, len(t)).astype(int)
    is_api = rng.random(len(t)) < 0.35
    return _finish(t, tin, tout, is_api, sp, rng, "af2019", model=model)


LOADERS = {"burstgpt": burstgpt, "azurellm2024": azurellm2024, "af2019": af2019, "synthetic": synthetic}


def build_workload(sp: SimParams) -> Workload:
    rng = np.random.default_rng(sp.seed * 1000 + 17)
    fn = LOADERS.get(sp.trace)
    if fn is None:
        raise ValueError(f"unknown trace {sp.trace}")
    wl = fn(sp, rng)
    if wl is None:
        warnings.warn(f"trace files for '{sp.trace}' not found in {sp.data_dir}/traces; using synthetic")
        wl = synthetic(sp, rng)
        wl.source = f"synthetic(fallback for {sp.trace})"
    return wl
