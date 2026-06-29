from __future__ import annotations

from agents.base import AgentFinding, AgentReport
from data_adapter import ResearchBundle


def run_regime_agent(bundle: ResearchBundle) -> AgentReport:
    findings: list[AgentFinding] = []
    regime_df = bundle.regime_df

    if regime_df.empty:
        return AgentReport(agent="RegimeAgent", findings=findings)

    for factor in bundle.summary.columns:
        if factor not in regime_df.index.get_level_values(0):
            continue
        sub = regime_df.loc[factor]
        sharpes = sub["sharpe"].dropna()
        if len(sharpes) < 2:
            continue
        spread = float(sharpes.max() - sharpes.min())
        if spread > 1.0:
            findings.append(
                AgentFinding(
                    agent="RegimeAgent",
                    risk_flag=f"{factor}: wide regime Sharpe dispersion",
                    evidence=f"Regime Sharpe spread {spread:.2f} ({sharpes.min():.2f} to {sharpes.max():.2f}).",
                    severity="Medium",
                    next_action="Condition signal on macro regime or reduce allocation in weak regimes.",
                )
            )

    if not findings:
        findings.append(
            AgentFinding(
                agent="RegimeAgent",
                risk_flag="Regime stability acceptable",
                evidence="No factor shows extreme cross-regime Sharpe dispersion in this sample.",
                severity="Low",
                next_action="Extend regime buckets when using longer live histories.",
            )
        )

    return AgentReport(agent="RegimeAgent", findings=findings)
