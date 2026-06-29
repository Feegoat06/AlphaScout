from __future__ import annotations

from agents.base import AgentFinding, AgentReport
from data_adapter import ResearchBundle
from monitor import build_investment_memo, build_monitor_reports, choose_best_factor


def run_memo_agent(bundle: ResearchBundle) -> AgentReport:
    monitor_reports = build_monitor_reports(bundle)
    best_factor, scores = choose_best_factor(bundle.metrics_df, monitor_reports, exclude={"composite"})
    memo = build_investment_memo(bundle, monitor_reports)

    findings = [
        AgentFinding(
            agent="MemoAgent",
            risk_flag="Research recommendation",
            evidence=f"Top candidate after governance scoring: {best_factor} (score {scores.loc[best_factor]:.2f}).",
            severity="Low",
            next_action="Share memo with PM; run validation checklist before capital allocation.",
        ),
        AgentFinding(
            agent="MemoAgent",
            risk_flag="Investment memo generated",
            evidence=memo[:500] + ("..." if len(memo) > 500 else ""),
            severity="Low",
            next_action="Attach memo to research ticket with monitor tables and data manifest.",
        ),
    ]
    return AgentReport(agent="MemoAgent", findings=findings)
