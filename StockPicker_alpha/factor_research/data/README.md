# Data Notes

## Primary path: real SAS pipeline

Place these files in the **project root** (`StockPicker_alpha/`):

| File | Role |
|------|------|
| `signals_raw_plus.sas7bdat` | Signals / predictors |
| `msf.sas7bdat` | Monthly stock returns |
| `factors_monthly.sas7bdat` | FF/Carhart factors + `rf` |

`data_adapter.load_real_research_bundle()` reuses:

- `part3_script.step1_tables_with_cal_ym` + `step3`–`step5` (panel)
- `return_prediction_model` quintile long-short + FF4 metrics

## Fallback: synthetic

If SAS files are missing, `load_synthetic_research_bundle()` produces a deterministic demo panel with Part 2 signal names (`momentum`, `BtM`, `ROA`, `ivol`).

## Cache

| Path | Content |
|------|---------|
| `data/cache/panel_essential.parquet` | Part 3 essential panel (real data only) |
| `data/cache/factor_ls_returns.parquet` | Monthly long-short return series |
| `runs/{timestamp}/` | Per-run manifest, metrics, agent findings, memo |
