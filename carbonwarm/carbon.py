"""Grid carbon-intensity time series (hourly, gCO2/kWh)."""
from __future__ import annotations

import glob
import os
import warnings
import zlib
from typing import Dict, Optional

import numpy as np
import pandas as pd

KWH_J = 3.6e6


class CarbonSeries:
    """Piecewise-constant hourly carbon intensity with O(1) integrals.

    t = 0 corresponds to the first hour of `values`.
    """

    def __init__(self, name: str, values: np.ndarray, forecast_noise: float = 0.0, seed: int = 0):
        v = np.asarray(values, dtype=float)
        if len(v) == 0:
            raise ValueError("empty carbon series")
        self.name = name
        self.values = np.maximum(v, 0.0)
        self.cum = np.concatenate([[0.0], np.cumsum(self.values) * 3600.0])
        rng = np.random.default_rng(seed + 7919)
        if forecast_noise > 0:
            self.noise = rng.lognormal(0.0, forecast_noise, size=len(v))
        else:
            self.noise = np.ones(len(v))
        self.mean = float(self.values.mean())

    # -- basic access -------------------------------------------------------
    def _h(self, t: float) -> int:
        h = int(t // 3600.0)
        return min(max(h, 0), len(self.values) - 1)

    def I(self, t: float) -> float:
        return float(self.values[self._h(t)])

    def F(self, t: float) -> float:
        """Integral of I from 0 to t, in (gCO2/kWh)*s."""
        if t <= 0:
            return 0.0
        h = int(t // 3600.0)
        if h >= len(self.values):
            return float(self.cum[-1] + self.values[-1] * (t - 3600.0 * len(self.values)))
        return float(self.cum[h] + self.values[h] * (t - 3600.0 * h))

    def integral(self, t1: float, t2: float) -> float:
        if t2 <= t1:
            return 0.0
        return self.F(t2) - self.F(t1)

    def grams_power(self, p_W: float, t1: float, t2: float) -> float:
        """Carbon (g) of drawing p_W watts over [t1, t2]."""
        return p_W * self.integral(t1, t2) / KWH_J

    def grams_energy(self, e_J: float, t1: float, t2: float) -> float:
        """Carbon (g) of e_J joules spread uniformly over [t1, t2]."""
        if t2 - t1 <= 1e-9:
            return e_J * self.I(t1) / KWH_J
        return e_J * (self.integral(t1, t2) / (t2 - t1)) / KWH_J

    # -- forecasts ---------------------------------------------------------
    def forecast(self, t: float, horizon_h: int = 2) -> float:
        """Mean forecast intensity over the next `horizon_h` hours (noisy if configured)."""
        h0 = self._h(t)
        h1 = min(h0 + max(horizon_h, 1), len(self.values))
        return float(np.mean(self.values[h0:h1] * self.noise[h0:h1]))

    def forecast_at(self, t: float) -> float:
        h = self._h(t)
        return float(self.values[h] * self.noise[h])

    def next_hour_boundary(self, t: float) -> float:
        return (int(t // 3600.0) + 1) * 3600.0

    def best_start(self, t1: float, t2: float) -> (float, float):
        """Start time (hour-aligned, or t1) in [t1, t2] with min forecast intensity."""
        if t2 <= t1:
            return t1, self.forecast_at(t1)
        h0, h1 = self._h(t1), self._h(t2)
        seg = self.values[h0:h1 + 1] * self.noise[h0:h1 + 1]
        k = int(np.argmin(seg))
        return (t1 if k == 0 else (h0 + k) * 3600.0), float(seg[k])


# ---------------------------------------------------------------------------
# Synthetic profiles (used when no real data is present, and for unit tests)
# ---------------------------------------------------------------------------
def synthetic_profile(zone: str, hours: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(zone.encode()) + seed)  # stable across processes
    h = np.arange(hours)
    if zone == "CAISO":
        local = (h - 8) % 24
        solar = np.clip(np.sin(np.pi * (local - 6) / 12.0), 0, None)
        day = h // 24
        sun = rng.uniform(0.55, 1.0, size=day.max() + 1)[day]
        v = 290 - 210 * solar * sun + rng.normal(0, 12, hours)
        return np.clip(v, 40, 450)
    if zone == "DE":
        z = np.zeros(hours)
        for i in range(1, hours):
            z[i] = 0.97 * z[i - 1] + rng.normal(0, 0.25)
        local = (h + 1) % 24
        v = 380 + 140 * z + 35 * np.sin(np.pi * (local - 10) / 12.0) + rng.normal(0, 10, hours)
        return np.clip(v, 110, 680)
    if zone == "FR":
        local = (h + 1) % 24
        v = 55 + 15 * np.sin(2 * np.pi * (local - 14) / 24.0) + rng.normal(0, 5, hours)
        return np.clip(v, 20, 130)
    if zone == "PL":
        local = (h + 1) % 24
        v = 680 + 40 * np.sin(2 * np.pi * (local - 15) / 24.0) + rng.normal(0, 18, hours)
        return np.clip(v, 500, 820)
    raise ValueError(f"no synthetic profile for zone {zone}")


def load_carbon_csv(path: str) -> pd.Series:
    """Load an hourly carbon-intensity CSV (Electricity Maps export or generic).

    Accepted layouts:
      * Electricity Maps: 'Datetime (UTC)', 'Carbon Intensity gCO₂eq/kWh (direct)' / '(LCA)'
      * generic: 'datetime', 'carbon_intensity'
    Direct (operational) intensity is preferred over life-cycle values.
    """
    df = pd.read_csv(path)
    cols = {c: c.lower() for c in df.columns}
    tcol = next((c for c, l in cols.items() if "datetime" in l), None) or \
        next((c for c, l in cols.items() if "time" in l or "date" in l), None)
    cands = [c for c, l in cols.items() if "carbon" in l or "intensity" in l]
    vcol = next((c for c in cands if "direct" in cols[c]), None) or \
        next((c for c in cands if "lca" in cols[c] or "life" in cols[c]), None) or \
        (cands[0] if cands else None)
    if tcol is None or vcol is None:
        raise ValueError(f"{path}: cannot find datetime / carbon-intensity columns in {list(df.columns)}")
    s = pd.Series(pd.to_numeric(df[vcol], errors="coerce").values,
                  index=pd.to_datetime(df[tcol], utc=True, errors="coerce"))
    s = s[~s.index.isna()].sort_index()
    s = s.resample("1h").mean().interpolate(limit_direction="both")
    return s


def build_carbon(scenario_map: Dict[str, str], hours: int, data_dir: str,
                 forecast_noise: float, seed: int, offset_h: int = 0) -> (Dict[str, CarbonSeries], Dict[str, str]):
    """Return {zone_key: CarbonSeries} plus {zone_key: source description}."""
    cache: Dict[str, np.ndarray] = {}
    sources: Dict[str, str] = {}

    def base(zone: str) -> np.ndarray:
        if zone in cache:
            return cache[zone]
        path = os.path.join(data_dir, "carbon", f"{zone}.csv")
        arr = None
        if os.path.exists(path):
            try:
                s = load_carbon_csv(path)
                full = np.asarray(s.values, dtype=float)
                off = int(offset_h) % len(full)          # wrap around if the file is shorter
                arr = np.roll(full, -off)
                if len(arr) < hours:  # tile if the file is shorter than the horizon
                    arr = np.tile(arr, int(np.ceil(hours / len(arr))))
                arr = arr[:hours]
                sources[zone] = f"file:{path}"
            except Exception as e:  # pragma: no cover
                warnings.warn(f"carbon file {path} unusable ({e}); using synthetic profile")
        if arr is None:
            arr = synthetic_profile(zone, hours, seed=0)
            sources[zone] = "synthetic"
        cache[zone] = np.asarray(arr, dtype=float)
        return cache[zone]

    out = {}
    for key, spec in scenario_map.items():
        if "+" in spec:
            z, sh = spec.split("+")
            shift = int(sh.rstrip("h"))
            vals = np.roll(base(z), shift)
        else:
            vals = base(spec)
        out[key] = CarbonSeries(f"{key}<-{spec}", vals, forecast_noise, seed)
    return out, sources
