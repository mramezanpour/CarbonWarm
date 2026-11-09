# CarbonWarm

Carbon-aware multi-tier model residency and cloud–edge placement for serverless LLM inference.

This repository contains the trace-driven simulator, the experiment scripts, the GPU measurement tools, the raw results and the figure scripts of the paper *CarbonWarm: Carbon-Aware Multi-Tier Model Residency and Cloud–Edge Placement for Serverless LLM Inference*.

## Layout

| Path | Content |
|---|---|
| `carbonwarm/` | Simulator: sites and tiers (`sim.py`), policies (`policies.py`), workloads (`workload.py`), carbon series (`carbon.py`), configuration (`config.py`), metrics, experiments, report, plots |
| `run_all.py` | Runs the experiments and writes `raw_runs.csv`, an Excel report and figures |
| `selftest.py` | Correctness tests of the simulator |
| `tools/make_carbon_files.py` | Builds hourly carbon-intensity files from Energy-Charts and EIA-930 |
| `tools/check_gpu.py`, `tools/gpu_validation.py` | GPU measurements used to validate the cost model |
| `tools/gpu_validation_colab.ipynb` | The same measurements on Google Colab |
| `results/` | Raw per-run results and GPU measurements reported in the paper |
| `paper/figures/` | Scripts that produce the figures and numbers of the paper from `results/` |
| `data/` | Input traces and carbon files (not included, see `data/README.md`) |

## Installation

```
pip install -r requirements.txt
pip install -r requirements-gpu.txt      # only for the GPU measurements
```

## Usage

```
python selftest.py
python tools/make_carbon_files.py
python run_all.py --preset quick
python run_all.py --preset full --workers 8
python run_all.py --preset full --workers 8 --only ablation_ext tinit
```

Experiments: `main`, `ablation`, `ablation_ext`, `sensitivity`, `tinit`, `clairvoyant`, `timeline`, `scale`, `dram`, `zipf`, `mu_trace`.
Without input files the simulator falls back to a synthetic workload and synthetic carbon series; the column `workload_source` of `raw_runs.csv` records which data were used.

GPU measurements:

```
python tools/check_gpu.py
python tools/gpu_validation.py
```

Figures of the paper:

```
python paper/figures/design_figs.py figs
python paper/figures/bg_figs.py results figs
python paper/figures/paper_figs.py results figs
python paper/figures/fb_figs.py results/raw_runs_power_and_ablation_ext.csv figs
python paper/figures/gpu_figs.py results/gpu figs
python paper/figures/calc_numbers.py results/raw_runs.csv
```

## Results

| File | Runs |
|---|---|
| `results/raw_runs.csv` | 1802 runs: main, ablation, sensitivity, Clairvoyant, timeline, scale, DRAM pressure, popularity order, SLO price |
| `results/raw_runs_power_and_ablation_ext.csv` | 540 main runs with the power-managed boundaries and 135 extended ablation runs |
| `results/CarbonWarm_results.xlsx` | Aggregated tables of `raw_runs.csv` |
| `results/gpu/` | RTX 3060 Laptop GPU and Tesla T4 measurements, two runs each |

## Citation

```
H. Jangholian, M. Ramezanpour, CarbonWarm: Carbon-aware multi-tier model residency and cloud–edge placement
for serverless LLM inference, (under review), 2026.
```

## License

MIT
