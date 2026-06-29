# Agentic Factor Research Monitor

A **factor research flight recorder** built on top of the take-home pipeline (`part1`–`part3` + `return_prediction_model`). It answers the asset-management question: **should a researcher or PM trust this backtest?**

## Narrative (for recruiters)

Last take-home: SAS audit → signal selection → prediction panel with strict `t → t+1` timing → quintile long-short + **Carhart-4 alpha**.

This demo adds a **governance layer**: specialized agents audit data integrity, timing, alpha vs Sharpe, regime stability, and literature priors; a rule-based monitor flags leakage, turnover, FF4 alpha gaps, and composite dilution; output is a PM-ready research memo with Parquet artifacts.

> This is not “AI inventing alpha.” It is **AI supervising whether to trust the research process.**

## Architecture

```text
part1/2/3 + return_prediction_model.py
        ↓
   data_adapter.py  (real SAS or synthetic fallback)
        ↓
   monitor.py + agents/ (orchestrator)
        ↓
   cache.py → runs/{timestamp}/manifest.json + Parquet
        ↓
   Agentic_Factor_Research_Monitor.ipynb
```

### Agents

| Agent | Role |
|-------|------|
| `DataIntegrityAgent` | Part 1 style required-field & duplicate-key checks |
| `TimingAgent` | Validates `cal_ym_signal` < `cal_ym_fwd` |
| `AlphaDiagnosticsAgent` | Sharpe vs FF4 alpha significance |
| `RegimeAgent` | Cross-regime Sharpe dispersion |
| `LiteratureAgent` | Academic priors (momentum, value, ivol puzzle, etc.) |
| `MemoAgent` | Investment memo + research score |

## Files

| File | Purpose |
|------|---------|
| `data_adapter.py` | Loads real Part 3/4 bundle or synthetic fallback |
| `monitor.py` | Per-factor monitor + memo generation |
| `agents/` | Multi-agent orchestrator |
| `cache.py` | Parquet cache + run manifest |
| `run_flight_recorder.py` | CLI entry point |
| `Agentic_Factor_Research_Monitor.ipynb` | Notebook demo |

## Data requirements

Place SAS files in the **project root** (`StockPicker_alpha/`):

- `signals_raw_plus.sas7bdat`
- `msf.sas7bdat`
- `factors_monthly.sas7bdat`

If missing, the pipeline automatically uses **synthetic fallback** so the demo still runs.

## How to run

```bash
cd factor_research_demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# CLI (uses real SAS when available)
python run_flight_recorder.py

# Force synthetic only
python run_flight_recorder.py --synthetic-only

# Notebook
jupyter notebook Agentic_Factor_Research_Monitor.ipynb
```

Run all cells top-to-bottom. Set `PREFER_REAL = False` in the notebook to force synthetic data.

## Outputs

Each run writes to `runs/{timestamp}/`:

- `manifest.json` — git hash, data source, factor list
- `metrics.parquet` / `metrics.json`
- `agent_findings.parquet`
- `monitor_{factor}.parquet`
- `memo.md`

Cached panel: `data/cache/panel_essential.parquet` (when real data loaded).

## JD alignment

- **Factor research:** classic signals from Part 2 on real academic data
- **Data engineering:** Parquet cache, reproducible manifests
- **Portfolio & risk:** FF4 alpha diagnostics, not raw Sharpe alone
- **Agentic AI:** transparent multi-agent governance (rule-based, LLM-ready hooks)
