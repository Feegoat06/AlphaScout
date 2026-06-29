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


| Agent                   | Role                                                 |
| ----------------------- | ---------------------------------------------------- |
| `DataIntegrityAgent`    | Part 1 style required-field & duplicate-key checks   |
| `TimingAgent`           | Validates `cal_ym_signal` < `cal_ym_fwd`             |
| `AlphaDiagnosticsAgent` | Sharpe vs FF4 alpha significance                     |
| `RegimeAgent`           | Cross-regime Sharpe dispersion                       |
| `LiteratureAgent`       | Academic priors (momentum, value, ivol puzzle, etc.) |
| `MemoAgent`             | Investment memo + research score                     |


## Files


| File                                    | Purpose                                          |
| --------------------------------------- | ------------------------------------------------ |
| `data_adapter.py`                       | Loads real Part 3/4 bundle or synthetic fallback |
| `monitor.py`                            | Per-factor monitor + memo generation             |
| `agents/`                               | Multi-agent orchestrator                         |
| `cache.py`                              | Parquet cache + run manifest                     |
| `run_flight_recorder.py`                | CLI entry point                                  |
| `Agentic_Factor_Research_Monitor.ipynb` | Notebook demo                                    |


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

---

## LangChain embedded implementation plan

This section is a **roadmap** for embedding [LangChain](https://python.langchain.com/) into the governance layer. The current codebase is intentionally **rule-first**; LangChain should augment **narrative and retrieval**, not replace deterministic quant logic.

### Design principles

1. **Rules decide, LLMs explain** — `monitor.py` and hard-check agents (`DataIntegrity`, `Timing`, `AlphaDiagnostics`) stay deterministic. LangChain must not downgrade `High` severity flags or invent metrics.
2. **Structured I/O** — LLM inputs/outputs are JSON-serializable artifacts already written to `runs/{timestamp}/` (`metrics.json`, monitor tables, `agent_findings`).
3. **Graceful fallback** — if `OPENAI_API_KEY` (or chosen provider) is missing, fall back to today's template-based `LiteratureAgent` / `MemoAgent`.
4. **Audit trail** — extend `manifest.json` with `llm_model`, `prompt_version`, and content hashes.

### Target architecture

```text
part1/2/3 + return_prediction_model.py
        ↓
   data_adapter.py
        ↓
   monitor.py  (deterministic — unchanged)
        ↓
   agents/     (rule agents — unchanged)
        ↓
   agents/langchain/   (NEW — optional LLM layer)
        ├── chains.py          # LCEL chains (literature, memo polish)
        ├── prompts.py         # versioned prompt templates
        ├── schemas.py         # Pydantic output models
        ├── retriever.py       # RAG over literature + run artifacts
        └── governance_graph.py  # optional LangGraph orchestration
        ↓
   cache.py → runs/{timestamp}/ (+ memo_llm.md, llm_trace.json)
```

### What LangChain should do (and what it should not)


| Component                       | LangChain role                                               | Keep rule-based?       |
| ------------------------------- | ------------------------------------------------------------ | ---------------------- |
| `DataIntegrityAgent`            | No LLM                                                       | Yes                    |
| `TimingAgent`                   | No LLM                                                       | Yes                    |
| `AlphaDiagnosticsAgent`         | No LLM                                                       | Yes                    |
| `RegimeAgent`                   | Optional: natural-language regime summary                    | Yes (metrics)          |
| `LiteratureAgent`               | **RAG + synthesis** over curated papers / factor notes       | Fallback templates     |
| `MemoAgent`                     | **Narrative polish** from structured metrics + monitor flags | Fallback template memo |
| `monitor.py`                    | No LLM                                                       | Yes                    |
| New: `ExoticDataAgent` (future) | RAG over Edgar/news snippets                                 | N/A                    |


### Phased rollout

#### Phase 0 — Scaffolding (0.5 day)

**Goal:** Add optional LangChain without changing default CLI behavior.


| Task              | File(s)                                                                                     |
| ----------------- | ------------------------------------------------------------------------------------------- |
| Add optional deps | `requirements-llm.txt` (`langchain`, `langchain-openai`, `langchain-community`, `pydantic`) |
| Feature flag      | `FACTOR_GOV_LLM=0/1` env var; `--with-llm` CLI flag on `run_flight_recorder.py`             |
| Shared client     | `agents/langchain/client.py` — lazy-init `ChatOpenAI` (or configurable provider)            |
| Output schemas    | `agents/langchain/schemas.py` — `LiteratureNote`, `MemoDraft`, `AgentNarrative`             |


**Acceptance:** `python run_flight_recorder.py` behaves exactly as today when LLM is off.

#### Phase 1 — LiteratureAgent via LCEL (1 day)

**Goal:** Replace static `LITERATURE_NOTES` with retrieval-grounded summaries.

1. **Corpus** — add `data/literature/` with small curated markdown files (e.g. `momentum.md`, `value.md`, `ivol_puzzle.md`) citing Jegadeesh & Titman, Fama-French, Ang et al.
2. **Retriever** — `agents/langchain/retriever.py`:
  - `DirectoryLoader` + `RecursiveCharacterTextSplitter`
  - `Chroma` or in-memory `FAISS` vector store (persist under `data/cache/literature_index/`)
3. **Chain** — `agents/langchain/chains.py`:

```python
# Sketch — not yet implemented
literature_chain = (
    {"context": retriever | format_docs, "factor": RunnablePassthrough(), "hypothesis": RunnablePassthrough()}
    | literature_prompt
    | llm.with_structured_output(LiteratureNote)
)
```

1. **Integration** — `agents/literature.py` calls `run_literature_agent_llm(bundle)` when flag enabled; maps `LiteratureNote` → `AgentFinding`.
2. **Guardrail** — prompt instructs: *cite only retrieved context; if insufficient context, say "insufficient literature coverage".*

**Acceptance:** LLM literature findings reference corpus chunks; offline fallback still works.

#### Phase 2 — MemoAgent narrative layer (1 day)

**Goal:** Keep `choose_best_factor()` and monitor tables deterministic; use LLM only for PM-facing prose.

1. **Input bundle** (built in `agents/langchain/chains.py`):

```json
{
  "data_source": "real_sas_pipeline",
  "recommended_factor": "BtM",
  "metrics": { "sharpe": 0.46, "ff4_alpha_monthly": 0.0231, "...": "..." },
  "monitor_flags": [{"risk_flag": "Material drawdown", "severity": "Medium"}],
  "agent_findings": ["..."],
  "part2_hypothesis": "..."
}
```

1. **Chain** — structured output `MemoDraft` with fields: `executive_summary`, `bull_case`, `bear_case`, `next_steps` (list), `governance_takeaway`.
2. **Post-process** — `monitor.build_investment_memo()` remains source of truth for **numbers**; LLM output saved as `runs/{ts}/memo_llm.md` and cross-checked:
  - numeric fields in LLM text must match `metrics.json` (simple regex validation)
  - on mismatch → write `memo_llm.md` with warning banner + use template memo
3. **Integration** — `ResearchOrchestrator.run(..., use_llm=True)` appends LLM memo path to manifest.

**Acceptance:** PM memo reads naturally; no hallucinated Sharpe/alpha values.

#### Phase 3 — LangGraph governance graph (optional, 1–2 days)

**Goal:** Model agent workflow explicitly for demos and future extension.

```text
load_bundle → rule_agents → monitor → branch
                              ├─ (high severity) → llm_deep_review
                              └─ (else)          → llm_memo_polish → persist
```


| Node              | Type                                        |
| ----------------- | ------------------------------------------- |
| `load_bundle`     | Python                                      |
| `rule_agents`     | Python (existing agents)                    |
| `monitor`         | Python                                      |
| `llm_deep_review` | LangChain chain (summarize blocking issues) |
| `llm_memo_polish` | LangChain chain                             |
| `persist`         | Python (`cache.py`)                         |


Implement in `agents/langchain/governance_graph.py` using `StateGraph` with typed state:

```python
class GovernanceState(TypedDict):
    bundle: dict
    combined_findings: list
    monitor_reports: dict
    memo_template: str
    memo_llm: str | None
```

**Acceptance:** Graph run reproduces Phase 2 outputs with visible step trace in `llm_trace.json`.

#### Phase 4 — Exotic / alternative data hook (future)

Aligns with JD themes (news, Edgar) without blocking core delivery.

1. Drop files into `data/exotic/` (or fetch via scripted loader).
2. Reuse `retriever.py` with separate collection `exotic_index`.
3. New `ExoticDataAgent` — LangChain chain proposes *research questions*, not trading signals.
4. Pipe into `monitor.py` as additional `risk_flag` rows (severity capped by rules).

### Proposed new files


| Path                                   | Purpose                                     |
| -------------------------------------- | ------------------------------------------- |
| `requirements-llm.txt`                 | Optional LangChain stack                    |
| `agents/langchain/client.py`           | LLM provider factory                        |
| `agents/langchain/prompts.py`          | Versioned prompts (`PROMPT_VERSION = "v1"`) |
| `agents/langchain/schemas.py`          | Pydantic structured outputs                 |
| `agents/langchain/retriever.py`        | Literature / exotic RAG                     |
| `agents/langchain/chains.py`           | LCEL chains                                 |
| `agents/langchain/governance_graph.py` | Optional LangGraph workflow                 |
| `data/literature/*.md`                 | Curated factor paper notes                  |
| `runs/{ts}/memo_llm.md`                | LLM-polished memo                           |
| `runs/{ts}/llm_trace.json`             | Prompt hash, model, token usage             |


### Orchestrator integration sketch

```python
# agents/orchestrator.py (future)
class ResearchOrchestrator:
    def __init__(self, *, use_llm: bool = False):
        self.use_llm = use_llm

    def run(self, bundle: ResearchBundle) -> GovernanceResult:
        # 1. existing rule agents (unchanged)
        # 2. monitor (unchanged)
        # 3. if self.use_llm:
        #        result.memo_llm = run_memo_chain(bundle, monitor_reports, combined)
        #        result.literature = run_literature_chain(bundle)
```

### CLI / notebook changes

```bash
# Rule-only (default)
python run_flight_recorder.py

# Rule + LangChain narrative layer
FACTOR_GOV_LLM=1 python run_flight_recorder.py --with-llm

# Install LLM extras
python -m pip install -r requirements-llm.txt
```

Notebook: add cell `USE_LLM = False` mirroring CLI flag.

### Testing strategy


| Test                    | Type                                                           |
| ----------------------- | -------------------------------------------------------------- |
| Rule pipeline unchanged | Regression — `use_llm=False` byte-match on `metrics.json` keys |
| LLM off fallback        | No API key → template memo                                     |
| Structured output parse | Mock LLM returns valid `MemoDraft`                             |
| Numeric guardrail       | Inject wrong Sharpe in mock → fallback to template             |
| RAG grounding           | Literature answer must include source doc id from metadata     |


### Security and cost

- Read API keys from environment only (`OPENAI_API_KEY`); never commit secrets.
- Log `prompt_version` + hashed prompt inputs in `llm_trace.json`, not full SAS rows.
- Cap retrieved chunks (e.g. `k=4`, `max_tokens=800`) and use a small model for memo polish.
- Consider running LLM steps only when `high_severity_findings > 0` to reduce cost.

### Suggested implementation order

1. Phase 0 scaffolding + feature flag
2. Phase 1 LiteratureAgent RAG
3. Phase 2 MemoAgent polish + numeric guardrails
4. Phase 3 LangGraph (optional demo)
5. Phase 4 exotic data (JD stretch goal)

