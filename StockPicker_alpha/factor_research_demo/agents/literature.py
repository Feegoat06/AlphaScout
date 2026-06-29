from __future__ import annotations

from agents.base import AgentFinding, AgentReport
from data_adapter import ResearchBundle

LITERATURE_NOTES: dict[str, str] = {
    "momentum": (
        "Jegadeesh & Titman (1993): momentum premia are well documented but crash-prone; "
        "monitor regime and turnover."
    ),
    "BtM": (
        "Fama & French (1992): value premia are classic but cyclical; verify economic link in current sample."
    ),
    "ROA": (
        "Novy-Marx (2013) profitability: quality/profitability factors can complement value; check overlap with BtM."
    ),
    "ivol": (
        "Ang et al. (2006) idiosyncratic volatility puzzle: higher ivol predicts lower returns; sign flip is intentional."
    ),
    "composite": (
        "Composite equal-weight blends diversify signal noise but can dilute the strongest single anomaly."
    ),
}


def run_literature_agent(bundle: ResearchBundle) -> AgentReport:
    findings: list[AgentFinding] = []

    for factor in bundle.summary.columns:
        note = LITERATURE_NOTES.get(factor)
        if not note:
            continue
        findings.append(
            AgentFinding(
                agent="LiteratureAgent",
                risk_flag=f"{factor}: academic prior",
                evidence=note,
                severity="Low",
                next_action="Compare backtest sign and magnitude against published anomaly baselines.",
            )
        )

    return AgentReport(agent="LiteratureAgent", findings=findings)
