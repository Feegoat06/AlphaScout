"""Bridge Part 3/4 pipeline to the factor research flight recorder."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEMO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from part2_script import SELECTED_SIGNALS  # noqa: E402


@dataclass
class ResearchBundle:
    """Normalized inputs for monitor, agents, and notebook visualization."""

    data_source: str
    summary: pd.DataFrame
    metrics_df: pd.DataFrame
    factor_hypotheses: dict[str, str]
    factor_holdings: dict[str, dict[Any, set[str]]]
    regime_df: pd.DataFrame
    rolling_sharpe: pd.DataFrame
    q4_metrics: pd.DataFrame
    panel: pd.DataFrame | None = None
    tables: dict[str, pd.DataFrame] | None = None
    audits: dict[str, Any] = field(default_factory=dict)


def factor_hypotheses_from_part2() -> dict[str, str]:
    return {str(sig["column"]): str(sig["economic_idea"]) for sig in SELECTED_SIGNALS}


def _month_regime(date: pd.Timestamp) -> str:
    ts = pd.Timestamp(date)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    if ts < pd.Timestamp("2000-01-01"):
        return "pre_2000"
    if ts < pd.Timestamp("2010-01-01"):
        return "2000_2009"
    if ts < pd.Timestamp("2020-01-01"):
        return "2010_2019"
    return "recent"


def _regime_key_from_index(idx: object) -> str:
    ts = pd.Timestamp(idx)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return _month_regime(ts)


def _annualized_metrics(returns: pd.Series) -> dict[str, float]:
    r = returns.dropna()
    if r.empty:
        return {"cagr": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "hit_rate": np.nan}
    years = len(r) / 12
    total_return = (1 + r).prod() - 1
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
    sharpe = np.sqrt(12) * r.mean() / r.std() if r.std() > 0 else np.nan
    equity = (1 + r).cumprod()
    max_dd = float((equity / equity.cummax() - 1).min())
    return {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "hit_rate": float((r > 0).mean()),
    }


def _turnover_from_quintile_panel(qdf: pd.DataFrame) -> float:
    if qdf.empty:
        return float("nan")
    holdings: dict[Any, set[str]] = {}
    for month, grp in qdf.groupby("cal_ym_signal"):
        longs = grp.loc[grp["quintile"] == grp["quintile"].max(), "PERMNO"].astype(str)
        shorts = grp.loc[grp["quintile"] == 1, "PERMNO"].astype(str)
        holdings[month] = set(longs).union(set(shorts))
    dates = sorted(holdings.keys())
    values: list[float] = []
    for prev, curr in zip(dates[:-1], dates[1:]):
        previous, current = holdings[prev], holdings[curr]
        if not previous and not current:
            values.append(0.0)
        else:
            values.append(1 - len(previous.intersection(current)) / len(previous.union(current)))
    return float(np.mean(values)) if values else float("nan")


def _build_regime_and_rolling(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = summary.copy()
    summary.index = pd.to_datetime(summary.index)
    if getattr(summary.index, "tz", None) is not None:
        summary.index = summary.index.tz_localize(None)
    regime_rows: list[dict[str, Any]] = []
    for factor in summary.columns:
        factor_returns = summary[factor].dropna()
        for regime, r in factor_returns.groupby(_regime_key_from_index):
            row = _annualized_metrics(r)
            row["factor"] = factor
            row["regime"] = regime
            regime_rows.append(row)
    regime_df = pd.DataFrame(regime_rows).set_index(["factor", "regime"]).sort_index()
    rolling_sharpe = summary.rolling(24).mean() / summary.rolling(24).std() * np.sqrt(12)
    return regime_df, rolling_sharpe


def _metrics_df_from_q4(q4: pd.DataFrame, q_panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in q4.iterrows():
        signal = str(row["signal"])
        m = {
            "factor": signal,
            "n_months": int(row["n_months"]),
            "cagr": np.nan,
            "sharpe": float(row["sharpe_annualized"]),
            "max_drawdown": float(row["max_drawdown"]),
            "hit_rate": np.nan,
            "avg_turnover": _turnover_from_quintile_panel(q_panels.get(signal, pd.DataFrame())),
            "capm_alpha_monthly": float(row["capm_alpha_monthly"]),
            "capm_alpha_se": float(row["capm_alpha_se"]),
            "ff4_alpha_monthly": float(row["ff4_alpha_monthly"]),
            "ff4_alpha_se": float(row["ff4_alpha_se"]),
            "mean_monthly_ls": float(row["mean_monthly_ls"]),
            "annualized_vol_ls": float(row["annualized_vol_ls"]),
        }
        ls = q_panels.get(signal)
        if ls is not None and not ls.empty and "ls_ret_fwd_ew" in ls.columns:
            pass
        rows.append(m)
    out = pd.DataFrame(rows).set_index("factor")
    return out


def load_real_research_bundle() -> ResearchBundle:
    from return_prediction_model import (  # noqa: WPS433
        attach_part2_scores,
        build_composite_score_panel,
        assign_quintiles,
        gather_q4_metrics_rows,
        format_q4_metrics_dataframe,
        load_prediction_panel_from_part3,
        merge_quintile_panel_with_ret_fwd,
        monthly_equal_weighted_quintile_returns,
        monthly_long_short_ew_returns,
        quintile_panel_for_all_signals,
    )
    from part1_script import PATHS, q3_required_fields_rows, q4_duplicate_check_rows  # noqa: WPS433
    from part3_script import prediction_panel_essential, step1_tables_with_cal_ym  # noqa: WPS433

    for path in PATHS.values():
        if not path.exists():
            raise FileNotFoundError(path)

    tables = step1_tables_with_cal_ym()
    panel, _ = load_prediction_panel_from_part3(attach_factors_and_excess=True)
    scored = attach_part2_scores(panel)
    q_panels = quintile_panel_for_all_signals(scored)

    scored_c = build_composite_score_panel(scored)
    qdf_c = assign_quintiles(scored_c, "composite_score").copy()
    qdf_c.insert(0, "signal", "composite")
    q_panels["composite"] = qdf_c

    summary_cols: dict[str, pd.Series] = {}
    factor_holdings: dict[str, dict[Any, set[str]]] = {}
    for name, qdf in q_panels.items():
        merged = merge_quintile_panel_with_ret_fwd(qdf, scored)
        ew_long = monthly_equal_weighted_quintile_returns(merged)
        ls = monthly_long_short_ew_returns(ew_long)
        if ls.empty:
            continue
        idx = pd.to_datetime(ls["cal_ym_signal"].astype(str) + "-01")
        summary_cols[name] = pd.Series(ls["ls_ret_fwd_ew"].values, index=idx, name=name)
        holdings: dict[Any, set[str]] = {}
        for month, grp in qdf.groupby("cal_ym_signal"):
            longs = grp.loc[grp["quintile"] == grp["quintile"].max(), "PERMNO"].astype(str)
            shorts = grp.loc[grp["quintile"] == 1, "PERMNO"].astype(str)
            holdings[month] = set(longs).union(set(shorts))
        factor_holdings[name] = holdings

    summary = pd.DataFrame(summary_cols).sort_index()
    q4_rows = gather_q4_metrics_rows(q_panels, scored)
    q4 = format_q4_metrics_dataframe(q4_rows)
    metrics_df = _metrics_df_from_q4(q4, q_panels)

    for factor in summary.columns:
        m = _annualized_metrics(summary[factor])
        if factor in metrics_df.index:
            metrics_df.loc[factor, "cagr"] = m["cagr"]
            metrics_df.loc[factor, "hit_rate"] = m["hit_rate"]

    regime_df, rolling_sharpe = _build_regime_and_rolling(summary)

    hypotheses = factor_hypotheses_from_part2()
    hypotheses["composite"] = (
        "Equal-weight composite of winsorized, z-scored Part 2 signals; tests whether "
        "diversification across momentum, value, profitability, and low-ivol improves FF4 alpha."
    )

    cached = {name: tables.get(name) for name in PATHS}
    audits = {
        "required_fields": q3_required_fields_rows(cached).to_dict(orient="records"),
        "duplicate_keys": q4_duplicate_check_rows(cached).to_dict(orient="records"),
    }

    return ResearchBundle(
        data_source="real_sas_pipeline",
        summary=summary,
        metrics_df=metrics_df,
        factor_hypotheses=hypotheses,
        factor_holdings=factor_holdings,
        regime_df=regime_df,
        rolling_sharpe=rolling_sharpe,
        q4_metrics=q4,
        panel=prediction_panel_essential(panel),
        tables=tables,
        audits=audits,
    )


def load_synthetic_research_bundle(seed: int = 42) -> ResearchBundle:
    rng = np.random.default_rng(seed)
    tickers = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "JPM", "V", "MA", "UNH",
        "HD", "PG", "KO", "PEP", "COST", "WMT", "XOM", "CVX", "CAT", "BA",
        "GE", "IBM", "INTC", "CSCO", "ADBE", "CRM", "NFLX", "DIS", "MCD", "NKE",
    ]
    dates = pd.bdate_range("2017-01-03", "2026-05-29")
    n = len(tickers)
    static = pd.DataFrame(
        {
            "quality_exposure": rng.normal(0, 1, n),
            "value_exposure": rng.normal(0, 1, n),
            "risk_exposure": rng.uniform(0.75, 1.45, n),
        },
        index=tickers,
    )
    returns = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for i, date in enumerate(dates):
        market = rng.normal(0.0003, 0.01)
        idio = rng.normal(0, 0.009, n)
        returns.iloc[i] = market * static["risk_exposure"].to_numpy() + idio
    prices = 100 * (1 + returns).cumprod()
    monthly_prices = prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change()
    factor_scores = {
        "momentum": monthly_prices.pct_change(12).shift(1),
        "BtM": monthly_prices.pct_change(24).shift(1),
        "ivol": -returns.rolling(126).std().resample("ME").last().shift(1),
    }
    factor_scores["ROA"] = pd.DataFrame(
        static["quality_exposure"].to_numpy()[None, :] + rng.normal(0, 0.2, (len(monthly_prices), n)),
        index=monthly_prices.index,
        columns=tickers,
    )

    hypotheses = factor_hypotheses_from_part2()
    summary_cols: dict[str, pd.Series] = {}
    factor_holdings: dict[str, dict[Any, set[str]]] = {}

    next_month_returns = monthly_returns.shift(-1)
    for name, scores in factor_scores.items():
        rows = []
        holdings: dict[Any, set[str]] = {}
        for date in scores.index.intersection(next_month_returns.index):
            s = scores.loc[date].dropna()
            r = next_month_returns.loc[date].dropna()
            common = s.index.intersection(r.index)
            if len(common) < 10:
                continue
            n_bucket = max(3, int(np.floor(len(common) * 0.2)))
            ranked = s.loc[common].sort_values()
            long_names = ranked.tail(n_bucket).index
            short_names = ranked.head(n_bucket).index
            weight = pd.Series(0.0, index=common)
            weight.loc[long_names] = 1 / n_bucket
            weight.loc[short_names] = -1 / n_bucket
            rows.append((date, float((weight * r.loc[common]).sum())))
            holdings[date] = set(long_names).union(set(short_names))
        if not rows:
            continue
        idx, vals = zip(*rows)
        summary_cols[name] = pd.Series(vals, index=pd.to_datetime(list(idx)), name=name)
        factor_holdings[name] = holdings

    summary = pd.DataFrame(summary_cols).sort_index()
    metrics_rows = []
    for name in summary.columns:
        m = _annualized_metrics(summary[name])
        m["factor"] = name
        m["avg_turnover"] = _turnover_from_holdings(factor_holdings[name])
        m["ff4_alpha_monthly"] = np.nan
        m["ff4_alpha_se"] = np.nan
        m["capm_alpha_monthly"] = np.nan
        m["capm_alpha_se"] = np.nan
        m["n_months"] = int(summary[name].dropna().shape[0])
        m["mean_monthly_ls"] = float(summary[name].mean())
        m["annualized_vol_ls"] = float(summary[name].std() * np.sqrt(12))
        metrics_rows.append(m)
    metrics_df = pd.DataFrame(metrics_rows).set_index("factor")
    regime_df, rolling_sharpe = _build_regime_and_rolling(summary)
    q4 = metrics_df.reset_index().rename(columns={"factor": "signal"})
    q4["sharpe_annualized"] = q4["sharpe"]
    return ResearchBundle(
        data_source="synthetic_fallback",
        summary=summary,
        metrics_df=metrics_df,
        factor_hypotheses=hypotheses,
        factor_holdings=factor_holdings,
        regime_df=regime_df,
        rolling_sharpe=rolling_sharpe,
        q4_metrics=q4,
        audits={"note": "Synthetic bundle used because SAS files are unavailable."},
    )


def _turnover_from_holdings(holdings: dict[Any, set[str]]) -> float:
    dates = sorted(holdings.keys())
    values: list[float] = []
    for prev, curr in zip(dates[:-1], dates[1:]):
        previous, current = holdings[prev], holdings[curr]
        if not previous and not current:
            values.append(0.0)
        else:
            values.append(1 - len(previous.intersection(current)) / len(previous.union(current)))
    return float(np.mean(values)) if values else float("nan")


def load_research_bundle(*, prefer_real: bool = True) -> ResearchBundle:
    if prefer_real:
        try:
            return load_real_research_bundle()
        except FileNotFoundError:
            pass
    return load_synthetic_research_bundle()
