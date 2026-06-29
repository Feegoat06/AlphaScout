"""Rule-based factor research monitor with FF4 alpha and composite checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from data_adapter import ResearchBundle


def severity_rank(label: str) -> int:
    return {"Low": 1, "Medium": 2, "High": 3}.get(label, 0)


def _alpha_significant(alpha: float, se: float, *, z: float = 1.96) -> bool:
    if pd.isna(alpha) or pd.isna(se) or se <= 0:
        return False
    return abs(alpha) > z * se


def monitor_factor(
    factor: str,
    returns: pd.Series,
    metrics_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    rolling_sharpe: pd.DataFrame,
    factor_hypotheses: dict[str, str],
    *,
    q4_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    m = metrics_df.loc[factor]
    factor_regimes = regime_df.loc[factor] if factor in regime_df.index.get_level_values(0) else pd.DataFrame()
    rs = rolling_sharpe[factor].dropna() if factor in rolling_sharpe.columns else pd.Series(dtype=float)
    rows: list[dict[str, str]] = []

    hypothesis = factor_hypotheses.get(
        factor,
        "No economic hypothesis attached; document rationale before trusting the backtest.",
    )
    rows.append(
        {
            "risk_flag": "Hypothesis clarity",
            "evidence": hypothesis,
            "severity": "Low",
            "next_action": "Keep the economic rationale attached to every backtest variant.",
        }
    )

    rows.append(
        {
            "risk_flag": "Lookahead control",
            "evidence": (
                "Signals use formation month t; returns attach to t+1 via Part 3 panel "
                "(cal_ym_signal -> cal_ym_fwd)."
            ),
            "severity": "Low",
            "next_action": "In live data, verify point-in-time fundamentals and corporate-action handling.",
        }
    )

    turnover = float(m.get("avg_turnover", np.nan))
    if pd.notna(turnover) and turnover > 0.70:
        rows.append(
            {
                "risk_flag": "High turnover",
                "evidence": f"Average monthly holdings turnover is {turnover:.1%}.",
                "severity": "High",
                "next_action": "Add transaction costs, rebalance buffers, and capacity assumptions.",
            }
        )
    elif pd.notna(turnover) and turnover > 0.45:
        rows.append(
            {
                "risk_flag": "Moderate turnover",
                "evidence": f"Average monthly holdings turnover is {turnover:.1%}.",
                "severity": "Medium",
                "next_action": "Estimate transaction cost sensitivity before treating the alpha as tradable.",
            }
        )

    sharpe = float(m.get("sharpe", np.nan))
    if pd.notna(sharpe) and sharpe < 0.30:
        rows.append(
            {
                "risk_flag": "Weak full-sample signal",
                "evidence": f"Full-sample Sharpe is only {sharpe:.2f}.",
                "severity": "High",
                "next_action": "Reject or redesign the signal unless a stronger sub-universe rationale exists.",
            }
        )
    elif pd.notna(sharpe) and sharpe > 2.50:
        rows.append(
            {
                "risk_flag": "Suspiciously strong result",
                "evidence": (
                    f"Full-sample Sharpe is {sharpe:.2f}, unusually high for a simple public factor."
                ),
                "severity": "High",
                "next_action": "Audit data generation, timestamp alignment, survivorship assumptions, and cost model.",
            }
        )

    ff4_alpha = float(m.get("ff4_alpha_monthly", np.nan))
    ff4_se = float(m.get("ff4_alpha_se", np.nan))
    if pd.notna(sharpe) and sharpe > 0.50 and not _alpha_significant(ff4_alpha, ff4_se):
        rows.append(
            {
                "risk_flag": "Sharpe without FF4 alpha",
                "evidence": (
                    f"Sharpe {sharpe:.2f} but Carhart-4 alpha {ff4_alpha:.4f} "
                    f"(SE {ff4_se:.4f}) is not significant at 5%."
                ),
                "severity": "High",
                "next_action": "Check whether returns are style-factor exposure rather than incremental alpha.",
            }
        )
    elif _alpha_significant(ff4_alpha, ff4_se) and ff4_alpha < 0:
        rows.append(
            {
                "risk_flag": "Negative risk-adjusted alpha",
                "evidence": f"FF4 alpha is {ff4_alpha:.4f} (SE {ff4_se:.4f}) after adjusting for known factors.",
                "severity": "High",
                "next_action": "Do not allocate capital; investigate sign, universe, and implementation shortfall.",
            }
        )

    if not factor_regimes.empty and "recent" in factor_regimes.index:
        recent_sharpe = float(factor_regimes.loc["recent", "sharpe"])
        earlier = factor_regimes.drop(index="recent", errors="ignore")
        earlier_sharpe = float(earlier["sharpe"].mean()) if not earlier.empty else np.nan
        if pd.notna(recent_sharpe) and pd.notna(earlier_sharpe) and recent_sharpe < earlier_sharpe - 0.75:
            rows.append(
                {
                    "risk_flag": "Weak recent out-of-sample behavior",
                    "evidence": (
                        f"Recent Sharpe is {recent_sharpe:.2f} versus earlier average Sharpe of {earlier_sharpe:.2f}."
                    ),
                    "severity": "High",
                    "next_action": "Investigate crowding, regime dependence, and whether the signal needs conditioning.",
                }
            )

    if not factor_regimes.empty:
        regime_sharpes = factor_regimes["sharpe"].dropna()
        if len(regime_sharpes) >= 3 and regime_sharpes.min() < -0.25 and regime_sharpes.max() > 0.75:
            rows.append(
                {
                    "risk_flag": "Regime instability",
                    "evidence": (
                        f"Regime Sharpe ranges from {regime_sharpes.min():.2f} to {regime_sharpes.max():.2f}."
                    ),
                    "severity": "Medium",
                    "next_action": "Test macro-conditioned allocation or combine with complementary factors.",
                }
            )

    if len(rs) > 0 and (rs < 0).mean() > 0.35:
        rows.append(
            {
                "risk_flag": "Unstable rolling Sharpe",
                "evidence": f"{(rs < 0).mean():.1%} of 24-month rolling Sharpe observations are negative.",
                "severity": "Medium",
                "next_action": "Run walk-forward validation and compare against a simple equal-weight benchmark.",
            }
        )

    max_dd = float(m.get("max_drawdown", np.nan))
    if pd.notna(max_dd) and max_dd < -0.20:
        rows.append(
            {
                "risk_flag": "Material drawdown",
                "evidence": f"Max drawdown is {max_dd:.1%}.",
                "severity": "Medium",
                "next_action": "Review drawdown timing and whether risk controls would have reduced exposure.",
            }
        )

    if factor == "composite" and q4_metrics is not None and not q4_metrics.empty:
        singles = q4_metrics[q4_metrics["signal"] != "composite"]
        comp_row = q4_metrics[q4_metrics["signal"] == "composite"]
        if not singles.empty and not comp_row.empty:
            best = singles.sort_values("sharpe_annualized", ascending=False).iloc[0]
            comp_sharpe = float(comp_row.iloc[0]["sharpe_annualized"])
            best_sharpe = float(best["sharpe_annualized"])
            if comp_sharpe < best_sharpe - 0.15:
                rows.append(
                    {
                        "risk_flag": "Composite dilutes best single signal",
                        "evidence": (
                            f"Composite Sharpe {comp_sharpe:.2f} trails best single signal "
                            f"{best['signal']} at {best_sharpe:.2f}."
                        ),
                        "severity": "Medium",
                        "next_action": (
                            f"Prefer continuing research on {best['signal']} rather than equal-weight composite."
                        ),
                    }
                )

    report = pd.DataFrame(rows)
    report["severity_rank"] = report["severity"].map(severity_rank)
    return report.sort_values(["severity_rank", "risk_flag"], ascending=[False, True]).drop(
        columns="severity_rank"
    )


def build_monitor_reports(bundle: ResearchBundle) -> dict[str, pd.DataFrame]:
    reports: dict[str, pd.DataFrame] = {}
    for factor in bundle.summary.columns:
        reports[factor] = monitor_factor(
            factor,
            bundle.summary[factor],
            bundle.metrics_df,
            bundle.regime_df,
            bundle.rolling_sharpe,
            bundle.factor_hypotheses,
            q4_metrics=bundle.q4_metrics,
        )
    return reports


def choose_best_factor(
    metrics_df: pd.DataFrame,
    monitor_reports: dict[str, pd.DataFrame],
    *,
    exclude: set[str] | None = None,
) -> tuple[str, pd.Series]:
    exclude = exclude or set()
    candidates = [f for f in metrics_df.index if f not in exclude]
    score = metrics_df.loc[candidates, "sharpe"].fillna(-99) + 0.5 * metrics_df.loc[candidates, "hit_rate"].fillna(0)
    score = score - metrics_df.loc[candidates, "avg_turnover"].fillna(1)
    for factor in candidates:
        report = monitor_reports.get(factor, pd.DataFrame())
        if report.empty:
            continue
        score.loc[factor] -= 0.35 * (report["severity"] == "High").sum()
        score.loc[factor] -= 0.12 * (report["severity"] == "Medium").sum()
        if "ff4_alpha_monthly" in metrics_df.columns:
            alpha = metrics_df.loc[factor, "ff4_alpha_monthly"]
            se = metrics_df.loc[factor, "ff4_alpha_se"]
            if _alpha_significant(float(alpha), float(se)):
                score.loc[factor] += 0.25
    ranked = score.sort_values(ascending=False)
    return str(ranked.index[0]), ranked


def build_investment_memo(
    bundle: ResearchBundle,
    monitor_reports: dict[str, pd.DataFrame],
    *,
    exclude_composite_from_recommendation: bool = True,
) -> str:
    exclude = {"composite"} if exclude_composite_from_recommendation else set()
    best_factor, research_scores = choose_best_factor(bundle.metrics_df, monitor_reports, exclude=exclude)
    best_report = monitor_reports[best_factor]
    top_risks = best_report[best_report["severity"].isin(["High", "Medium"])]
    if top_risks.empty:
        top_risk_text = "The monitor did not find major blocking issues, but live-data validation is still required."
    else:
        top_risk_text = "; ".join(top_risks.head(3)["risk_flag"].tolist())

    m = bundle.metrics_df.loc[best_factor]
    hypothesis = bundle.factor_hypotheses.get(best_factor, "N/A")

    return f"""
# Investment Research Memo

## Data source
**{bundle.data_source}** — Part 3 panel + Part 4 quintile long-short when SAS files are present.

## Recommended factor for continued research
**{best_factor}** is the strongest candidate after combining performance, hit rate, turnover, FF4 alpha, and monitor penalties.

## Why it might work
{hypothesis}

## Evidence from the backtest
- Sharpe: {m.get('sharpe', float('nan')):.2f}
- Max drawdown: {m.get('max_drawdown', float('nan')):.2%}
- Hit rate: {m.get('hit_rate', float('nan')):.2%}
- Average monthly turnover: {m.get('avg_turnover', float('nan')):.2%}
- FF4 alpha (monthly): {m.get('ff4_alpha_monthly', float('nan')):.4f} (SE {m.get('ff4_alpha_se', float('nan')):.4f})

## What the monitor caught
{top_risk_text}

## What a PM or researcher should do next
1. Re-run the same workflow on point-in-time live data with survivorship-bias controls.
2. Add transaction costs and liquidity constraints before accepting the result as tradable.
3. Run walk-forward validation and compare against simpler baselines.
4. Test whether the factor should be combined with complementary factors rather than traded alone.

## Research governance takeaway
The value of the agentic layer is not that it magically finds alpha. Its value is that it makes the research process auditable: every attractive backtest is paired with explicit assumptions, risk flags, and next tests.
""".strip()
