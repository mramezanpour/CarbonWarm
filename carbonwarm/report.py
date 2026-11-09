"""Excel workbook with all aggregated results."""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .config import GPU_TYPES, SimParams, default_sites, model_catalog

HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(name="Arial", bold=True, color="FFFFFF", size=10)
BODY_FONT = Font(name="Arial", size=10)
TITLE_FONT = Font(name="Arial", bold=True, size=13)

README = [
    ("CarbonWarm simulation results", "title"),
    ("", ""),
    ("Sheets", "bold"),
    ("Main_Summary      mean / std / 95% CI (Student t over seeds) of every metric, per trace x carbon scenario x policy", ""),
    ("Main_Reduction    paired (same seed) % carbon reduction of CarbonWarm vs each baseline", ""),
    ("Ablation_Summary  CarbonWarm with one component removed", ""),
    ("Sensitivity       sweeps of one parameter at a time (param / value columns)", ""),
    ("Clairvoyant_Compare  future-aware (clairvoyant) residency heuristic vs CarbonWarm on the same placement", ""),
    ("                     (a reference point, NOT a lower bound: CarbonWarm can be below it)", ""),
    ("TTFT_Quantiles    averaged TTFT quantiles of interactive requests (for CDF plots)", ""),
    ("Timeline_Hourly   hourly emissions by category and policy, hourly grid intensity", ""),
    ("Tier_Occupancy    GB of weights on GPU / DRAM / SSD over time", ""),
    ("Workload_Stats    per-model request counts and idle-gap statistics", ""),
    ("Scale_Summary / Scale_Reduction   12-200 models at 1 request/s", ""),
    ("DRAM_Pressure / DRAM_Reduction    100 models, DRAM capacity scaled down to 1/8", ""),
    ("Zipf_Order / Zipf_Reduction       five different assignments of popularity ranks to models", ""),
    ("Full_Boundary_Reduction           reductions incl. bare-idle GPU power, host power and PUE", ""),
    ("Mu_Trajectory                     SLO price mu over time for three attainment targets", ""),
    ("Config            simulation parameters, hardware, sites, models (all calibration inputs)", ""),
    ("Raw_Runs          one row per simulation run with every metric", ""),
    ("Figures           list of figures with captions (PDF files are in figures/)", ""),
    ("", ""),
    ("Metric definitions", "bold"),
    ("carbon_*_kg       operational carbon (kgCO2e) = power x grid intensity integrated over time", ""),
    ("   idle           GPU baseline power while a model is resident and no request runs", ""),
    ("   load           GPU baseline + extra power while a model is being loaded (cold start)", ""),
    ("   serve          GPU baseline while serving + dynamic prefill/decode energy", ""),
    ("   cache          DRAM / SSD power for holding checkpoints", ""),
    ("slo_attain_pct    % of interactive requests with TTFT <= SLO", ""),
    ("cold_start_pct    % of requests that waited for a model load", ""),
    ("deadline_miss_pct % of deferrable requests finishing after their deadline", ""),
    ("edge_share_pct    % of requests served at an edge site", ""),
    ("decision_overhead_us  mean wall-clock time of one scheduling decision (Python)", ""),
    ("carbon_full_kg    PUE x (policy carbon + bare-idle power of all GPUs + host base power) per site", ""),
    ("mu_mean / mu_at_min_pct  mean SLO price and % of updates at the lower bound (constraint slack)", ""),
    ("", ""),
    ("IMPORTANT", "bold"),
    ("Numbers are only as good as the calibration in carbonwarm/config.py and the input data.", ""),
    ("Check the 'workload_source' and 'carbon_sources' columns of Raw_Runs: 'synthetic' means no real file was found.", ""),
]


def _style(ws, pct_cols=()):
    ws.freeze_panes = "A2"
    for c in ws[1]:
        c.fill, c.font = HEADER_FILL, HEADER_FONT
        c.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[1].height = 32
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.font = BODY_FONT
            if isinstance(c.value, float):
                c.number_format = "0.0000" if abs(c.value) < 10 else "0.00"
    for k, col in enumerate(ws.columns, 1):
        width = max(len(str(col[0].value or "")), *(len(f"{c.value:.4f}") if isinstance(c.value, float)
                                                    else len(str(c.value or "")) for c in col[1:200])) if ws.max_row > 1 \
            else len(str(col[0].value or ""))
        ws.column_dimensions[get_column_letter(k)].width = min(max(10, width + 2), 48)


def config_frames(preset: dict) -> Dict[str, pd.DataFrame]:
    sp = SimParams(**preset["base"])
    d = asdict(sp)
    en = d.pop("energy")
    rows = [("SimParams", k, str(v)) for k, v in d.items()] + [("EnergyParams", k, str(v)) for k, v in en.items()]
    rows += [("Preset", k, str(v)) for k, v in preset.items() if k != "base"]
    gpus = pd.DataFrame([asdict(g) for g in GPU_TYPES.values()])
    sites = pd.DataFrame([asdict(s) for s in default_sites(sp.edge_gpus)])
    models = pd.DataFrame([asdict(m) for m in model_catalog(sp.n_models)])
    return {"params": pd.DataFrame(rows, columns=["group", "parameter", "value"]), "gpus": gpus,
            "sites": sites, "models": models}


def write_workbook(path: str, sheets: Dict[str, pd.DataFrame], preset: dict, fig_index: List[dict],
                   fig_dir: str, embed_figures=True):
    cfg = config_frames(preset)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame({"A": [r[0] for r in README]}).to_excel(xw, sheet_name="README", index=False, header=False)
        for name, df in sheets.items():
            if df is None or not len(df):
                continue
            df.to_excel(xw, sheet_name=name[:31], index=False)
        # config: stacked tables in one sheet
        r = 0
        for title, df in cfg.items():
            pd.DataFrame({title: []}).to_excel(xw, sheet_name="Config", startrow=r, index=False)
            df.to_excel(xw, sheet_name="Config", startrow=r + 1, index=False)
            r += len(df) + 4
        pd.DataFrame(fig_index).to_excel(xw, sheet_name="Figures", index=False)
    wb = load_workbook(path)
    ws = wb["README"]
    for k, (txt, kind) in enumerate(README, 1):
        c = ws.cell(row=k, column=1)
        c.font = TITLE_FONT if kind == "title" else Font(name="Arial", bold=(kind == "bold"), size=10)
    ws.column_dimensions["A"].width = 120
    for name in wb.sheetnames:
        if name in ("README",):
            continue
        if name == "Config":
            for row in wb[name].iter_rows():
                for c in row:
                    c.font = BODY_FONT
            wb[name].column_dimensions["A"].width = 18
            wb[name].column_dimensions["B"].width = 28
            wb[name].column_dimensions["C"].width = 40
            continue
        _style(wb[name])
    if embed_figures:
        ws = wb["Figures"]
        row = len(fig_index) + 4
        for f in fig_index:
            png = os.path.join(fig_dir, f"{f['figure']}.png")
            if not os.path.exists(png):
                continue
            img = XLImage(png)
            scale = 640.0 / img.width
            img.width, img.height = int(img.width * scale), int(img.height * scale)
            ws.cell(row=row, column=1, value=f["figure"]).font = Font(name="Arial", bold=True)
            ws.add_image(img, f"A{row + 1}")
            row += int(img.height / 20) + 4
    wb.save(path)
