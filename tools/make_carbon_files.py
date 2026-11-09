# -*- coding: utf-8 -*-
"""Build hourly carbon-intensity files data/carbon/{CAISO,DE,FR,PL}.csv (datetime, carbon_intensity).

DE, FR, PL: Energy-Charts API (Fraunhofer ISE), hourly public net generation by source.
CAISO: EIA-930 hourly net generation by source of balancing authority CISO; place the downloaded file(s)
in data/eia_raw/. Intensities use one set of direct-combustion emission factors for all zones.

Usage: python tools/make_carbon_files.py
"""
# ============================ SETTINGS ===================================
YEAR = 2024
EUROPE = ["DE", "FR", "PL"]          # countries to download from Energy-Charts
EIA_DIR = "data/eia_raw"             # where you put the EIA Excel / BALANCE csv files
EIA_BA = "CISO"                      # balancing authority code of California ISO
OUT_DIR = "data/carbon"
# =========================================================================

import glob
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Direct (combustion) emission factors, gCO2 per kWh of electricity.
# Sources: IPCC AR5 WG3 Annex III (median direct values) / EIA average heat rates.
EF = {
    "lignite": 1100.0, "brown coal": 1100.0,
    "hard coal": 950.0, "coal-derived gas": 950.0, "coal": 1000.0,
    "fossil gas": 400.0, "natural gas": 410.0, "gas": 400.0,
    "fossil oil": 780.0, "oil": 780.0, "petroleum": 780.0,
    "waste": 400.0, "others": 400.0, "other": 400.0,
}
ZERO = ["nuclear", "hydro", "wind", "solar", "biomass", "geothermal", "run of river", "water reservoir",
        "pumped storage generation", "renewable"]
SKIP = ["load", "residual", "cross border", "share", "consumption", "import", "export", "net"]


def factor(name: str):
    n = name.lower()
    if any(k in n for k in SKIP):
        return None                 # not a generation type
    if any(k in n for k in ZERO):
        return 0.0
    for k in sorted(EF, key=len, reverse=True):
        if k in n:
            return EF[k]
    return None


def save(series: pd.Series, zone: str):
    s = series.dropna()
    if len(s) == 0:
        print(f"  !! no valid hours for {zone}; {zone}.csv NOT written (check the input file and YEAR)")
        return
    s = s[(s > 0) & (s < 2000)]
    out = pd.DataFrame({"datetime": s.index.strftime("%Y-%m-%d %H:%M:%S"), "carbon_intensity": s.values.round(2)})
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{zone}.csv")
    out.to_csv(path, index=False)
    print(f"  -> {path}: {len(out)} hours, mean {s.mean():.0f} g/kWh, min {s.min():.0f}, max {s.max():.0f}")
    if len(out) < 8000:
        print("     WARNING: less than ~1 year of hours; check the input data")


# ---------------------------------------------------------------- Europe
def intensity_from_mix(df: pd.DataFrame, label: str) -> pd.Series:
    """df: index = UTC time, columns = generation per fuel type (MW). Returns gCO2/kWh."""
    used, num, den = [], 0.0, 0.0
    for col in df.columns:
        f = factor(col)
        if f is None:
            continue
        g = df[col].clip(lower=0).fillna(0)
        num = num + g * f
        den = den + g
        used.append(f"{col}({f:g})")
    print(f"  {label}: fuel types used -> {', '.join(used)}")
    return (num / den.replace(0, np.nan)).resample("1h").mean()


def energy_charts(country: str, year: int) -> pd.Series:
    frames = []
    for m in range(1, 13):
        start = f"{year}-{m:02d}-01T00:00Z"
        end = f"{year + (m == 12)}-{(m % 12) + 1:02d}-01T00:00Z"
        url = f"https://api.energy-charts.info/public_power?country={country.lower()}&start={start}&end={end}"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    js = json.load(r)
                break
            except Exception as e:
                print(f"    {country} {year}-{m:02d}: attempt {attempt + 1} failed ({e})")
                time.sleep(5)
        else:
            raise RuntimeError(f"cannot download {url}")
        idx = pd.to_datetime(js["unix_seconds"], unit="s", utc=True)
        frames.append(pd.DataFrame({p["name"]: p["data"] for p in js["production_types"]}, index=idx))
        print(f"    {country} {year}-{m:02d}: {len(idx)} rows")
        time.sleep(1)
    df = pd.concat(frames).apply(pd.to_numeric, errors="coerce")
    df = df[~df.index.duplicated()].sort_index()
    return intensity_from_mix(df, country)


# ---------------------------------------------------------------- EIA
def _find(cols, *words, exclude=()):
    for c in cols:
        lc = str(c).lower()
        if all(w in lc for w in words) and not any(x in lc for x in exclude):
            return c
    return None


# EIA fuel codes in the per-BA workbook (columns "NG: COL", "NG: NG", ...) -> emission factor (g/kWh).
# Same factors as used for Europe, so all four zones are production-based and comparable.
EIA_FUEL_EF = {"COL": 1000.0, "NG": 410.0, "OIL": 780.0, "OTH": 400.0,
               "NUC": 0.0, "GEO": 0.0, "WAT": 0.0, "PS": 0.0, "SUN": 0.0, "SNB": 0.0,
               "WND": 0.0, "WNB": 0.0, "BAT": 0.0, "OES": 0.0, "UES": 0.0}   # UNK (unknown) is ignored
LB_TO_G = 453.592


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def eia_from_excel(path: str) -> pd.Series:
    """Per-BA Excel workbook from the Grid Monitor (columns 'UTC time', 'NG: COL', 'NG: NG', ...).

    Carbon intensity is computed from the hourly generation mix, exactly like DE/FR/PL
    (production-based, direct emissions). EIA's own 'CO2 Emissions Intensity for Generated
    Electricity' column is in lbs/kWh; it is converted (x 453.592) and printed only as a check.
    """
    sheets = pd.read_excel(path, sheet_name=None)
    name, df = max(sheets.items(), key=lambda kv: kv[1].shape[0] * kv[1].shape[1])
    df.columns = [str(c).strip() for c in df.columns]
    print(f"  EIA Excel '{os.path.basename(path)}', sheet '{name}', {len(df)} rows")
    if len(df) == 0:
        raise SystemExit("    The Excel file has column names but no data rows. Download the full file from EIA again.")
    if "BA" in df.columns:
        df = df[df["BA"].astype(str).str.strip().isin([EIA_BA, ""]) | df["BA"].isna()]
    tcol = next((c for c in df.columns if c.lower() == "utc time"), None) or _find(df.columns, "utc")
    if tcol is None:
        raise SystemExit("    no 'UTC time' column found; columns are: " + ", ".join(df.columns))
    t = pd.to_datetime(df[tcol], utc=True, errors="coerce", format="mixed")
    if t.isna().mean() > 0.5:   # e.g. '01/01/2024 8 a.m.' style
        t = pd.to_datetime(df[tcol].astype(str).str.replace("a.m.", "AM").str.replace("p.m.", "PM"),
                           utc=True, errors="coerce", format="mixed")
    fuel_cols = {c: c.split(":", 1)[1].strip().upper() for c in df.columns
                 if c.upper().startswith("NG:") and c.split(":", 1)[1].strip().upper() in EIA_FUEL_EF}
    if not fuel_cols:
        raise SystemExit("    no 'NG: <fuel>' generation columns found; columns are: " + ", ".join(df.columns))
    num = den = 0.0
    for c, code in fuel_cols.items():
        g = _num(df[c]).clip(lower=0).fillna(0).values      # MWh in the hour; batteries/pumped storage charging < 0 ignored
        num = num + g * EIA_FUEL_EF[code]
        den = den + g
    print("    fuel columns used: " + ", ".join(f"{c}({EIA_FUEL_EF[k]:g})" for c, k in fuel_cols.items()))
    ci = pd.Series(num / np.where(den > 0, den, np.nan), index=t)
    keep = ~ci.index.isna() & (ci.index.year == YEAR)
    ci = ci[keep].sort_index()
    ci = ci[~ci.index.duplicated()].resample("1h").mean().interpolate(limit=6)

    # cross-check against EIA's own estimates (lbs/kWh -> g/kWh)
    for col in ["CO2 Emissions Intensity for Generated Electricity", "CO2 Emissions Intensity for Consumed Electricity"]:
        if col in df.columns:
            v = pd.Series(_num(df[col]).values * LB_TO_G, index=t)[keep]
            if v.notna().any():
                print(f"    check: EIA '{col}' x 453.592 -> mean {v.mean():.0f} g/kWh "
                      f"(this script: {ci.mean():.0f} g/kWh, generation-based)")
    return ci


def eia_from_balance(paths) -> pd.Series:
    """Six-month EIA930_BALANCE csv files: compute intensity from the fuel mix of one BA."""
    frames = []
    for p in sorted(paths):
        d = pd.read_csv(p, low_memory=False, thousands=",")
        bacol = _find(d.columns, "balancing authority")
        d = d[d[bacol].astype(str).str.strip() == EIA_BA]
        tcol = _find(d.columns, "utc time") or _find(d.columns, "utc")
        d.index = pd.to_datetime(d[tcol], utc=True, errors="coerce")
        gen = {}
        for c in d.columns:
            lc = c.lower()
            if "net generation" in lc and "from" in lc and "adjusted" not in lc and "imputed" not in lc:
                gen[c.split("from")[-1].strip()] = pd.to_numeric(d[c], errors="coerce")
        frames.append(pd.DataFrame(gen, index=d.index))
        print(f"  {os.path.basename(p)}: {len(d)} rows for {EIA_BA}")
    df = pd.concat(frames).sort_index()
    df = df[df.index.year == YEAR]
    return intensity_from_mix(df, "CAISO")


def main():
    os.chdir(ROOT)
    print(f"Building carbon files for {YEAR} in {os.path.abspath(OUT_DIR)}\n")
    for c in EUROPE:
        print(f"[{c}] downloading from Energy-Charts ...")
        try:
            save(energy_charts(c, YEAR), c)
        except Exception as e:
            print(f"  FAILED for {c}: {e}")
    print("\n[CAISO] reading EIA files from", os.path.abspath(EIA_DIR))
    xls = glob.glob(os.path.join(EIA_DIR, "*.xls*"))
    bal = [p for p in glob.glob(os.path.join(EIA_DIR, "*.csv")) if "balance" in os.path.basename(p).lower()]
    if xls:
        save(eia_from_excel(xls[0]), "CAISO")
    elif bal:
        save(eia_from_balance(bal), "CAISO")
    else:
        print("  no EIA file found. Download the CISO Excel file or the two EIA930_BALANCE six-month csv files\n"
              "  from https://www.eia.gov/electricity/gridmonitor/ (button DOWNLOAD DATA) into", EIA_DIR)


if __name__ == "__main__":
    main()
