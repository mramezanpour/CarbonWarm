# Input data

## data/traces/

| File | Source |
|---|---|
| `BurstGPT_without_fails_1.csv` | https://github.com/HPMLL/BurstGPT |
| `AzureLLMInferenceTrace_code_1week.csv`, `AzureLLMInferenceTrace_conv_1week.csv` | https://github.com/Azure/AzurePublicDataset (dataset-llm-2024) |
| `invocations_per_function_md.anon.d01.csv` … `d14.csv` | https://github.com/Azure/AzurePublicDataset (dataset-functions-2019) |

## data/carbon/

Hourly carbon intensity per zone: `CAISO.csv`, `DE.csv`, `FR.csv`, `PL.csv` with columns `datetime,carbon_intensity` (gCO2/kWh).
Build them with `python tools/make_carbon_files.py` (Energy-Charts API for DE, FR, PL; EIA-930 files of CISO in `data/eia_raw/` for CAISO).
Electricity Maps hourly CSV exports are also accepted.
