from __future__ import annotations

import pandas as pd

from agents.base import AgentFinding, AgentReport
from data_adapter import ResearchBundle


def _alpha_significant(alpha: float, se: float) -> bool:
    if pd.isna(alpha) or pd.isna(se) or se <= 0:
        return False
    return abs(alpha) > 1.96 * se


def run_alpha_diagnostics_agent(bundle: ResearchBundle) -> AgentReport:
    findings: list[AgentFinding] = []
    mdf = bundle.metrics_df

    for factor in mdf.index:
        sharpe = float(mdf.loc[factor, "sharpe"]) if "sharpe" in mdf.columns else float("nan")
        alpha = float(mdf.loc[factor, "ff4_alpha_monthly"]) if "ff4_alpha_monthly" in mdf.columns else float("nan")
        se = float(mdf.loc[factor, "ff4_alpha_se"]) if "ff4_alpha_se" in mdf.columns else float("nan")

        if pd.isna(alpha):
            continue
        if pd.notna(sharpe) and sharpe > 0.5 and not _alpha_significant(alpha, se):
            findings.append(
                AgentFinding(
                    agent="AlphaDiagnosticsAgent",
                    risk_flag=f"{factor}: style exposure risk",
                    evidence=f"Sharpe {sharpe:.2f} without significant FF4 alpha ({alpha:.4f}, SE {se:.4f}).",
                    severity="High",
                    next_action="Decompose returns against FF factors before portfolio construction.",
                )
            )
        elif _alpha_significant(alpha, se) and alpha > 0:
            findings.append(
                AgentFinding(
                    agent="AlphaDiagnosticsAgent",
                    risk_flag=f"{factor}: positive FF4 alpha",
                    evidence=f"Monthly FF4 alpha {alpha:.4f} (SE {se:.4f}) is statistically meaningful.",
                    severity="Low",
                    next_action="Stress-test alpha stability across subperiods and after costs.",
                )
            )

    if not findings:
        findings.append(
            AgentFinding(
                agent="AlphaDiagnosticsAgent",
                risk_flag="No FF4 diagnostics available",
                evidence="FF4 alpha not computed (synthetic fallback or insufficient factor history).",
                severity="Medium",
                next_action="Run on real SAS pipeline with factor merge enabled.",
            )
        )

    return AgentReport(agent="AlphaDiagnosticsAgent", findings=findings)
