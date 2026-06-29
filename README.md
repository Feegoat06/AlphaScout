# QuantTradeProjects

A Python toolkit for quantitative trading with two complementary stock-selection frameworks: **rule-based** screening and **alpha-based** factor research.

## Rule-based — `stockPicker_rulebased/`

A **rule engine** for live screening: interpretable technical and fundamental filters, with tiered priority outputs.

| Script | Market | Highlights |
|--------|--------|------------|
| `StockChooseScript_CN.py` | China A-shares | East Money fund-flow top pool + MA / volume / return filters; ROE and 7-day repeat-appearance tiers (Level 1–3) |
| `StockChooseScript_NA.py` | US equities | Custom ticker universe; trend, relative strength, VWAP, volume expansion; Level 1 (no ROE) / Level 2 (with ROE) |

Data sources: **yfinance** and East Money API. Results can be saved to local CSV/Excel history files.

## Alpha-based — `StockPicker_alpha/`

A **factor / alpha research pipeline** on monthly SAS panels (signals → forward returns → Fama–French / Carhart factors), evaluating long–short quintile performance for single and composite signals.

| Module | Role |
|--------|------|
| `part1_script.py` – `part3_script.py` | Data audit, signal selection, monthly prediction panel (strict `t → t+1` timing) |
| `return_prediction_model.py` | Quintile long–short backtests, Sharpe, CAPM / Carhart-4 alpha |
| `factor_research_demo/` | Factor research flight recorder: rule-based monitoring + agent governance layer, research memos and metric artifacts |

See each subdirectory’s `README.md` and `requirements.txt` for details.

---

## 中文

Python 量化交易工具集，包含两套互补的选股框架：**规则驱动（rule-based）** 与 **因子/Alpha 驱动（alpha-based）**。

### Rule-based — `stockPicker_rulebased/`

面向实盘筛选的**规则引擎**，用可解释的技术面与基本面条件过滤标的，并按优先级分层输出。

| 脚本 | 市场 | 要点 |
|------|------|------|
| `StockChooseScript_CN.py` | A 股 | 东方财富资金流 Top 池 + 均线/量能/收益等技术面；ROE 与 7 日重复出现次数分层（Level 1–3） |
| `StockChooseScript_NA.py` | 美股 | 自定义 ticker 池；趋势、相对强度、VWAP、量能扩张等规则；Level 1（不含 ROE）/ Level 2（含 ROE） |

数据源以 **yfinance**、东方财富 API 为主，结果可写入本地 CSV/Excel 历史记录。

### Alpha-based — `StockPicker_alpha/`

面向研究与回测的**因子/Alpha 管线**，基于 SAS 月度面板（信号 → 前瞻收益 → Fama-French/Carhart 因子），评估单因子与复合因子的多空组合表现。

| 模块 | 作用 |
|------|------|
| `part1_script.py` – `part3_script.py` | 数据审计、信号选取、月度预测面板（严格 `t → t+1` 时序） |
| `return_prediction_model.py` | 五分位多空回测、Sharpe、CAPM / Carhart-4 Alpha |
| `factor_research_demo/` | 因子研究「飞行记录仪」：规则化监控 + Agent 治理层，输出研究备忘录与指标产物 |

详见各子目录下的 `README.md` 与 `requirements.txt`。
