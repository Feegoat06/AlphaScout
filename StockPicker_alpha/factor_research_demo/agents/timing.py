from __future__ import annotations

from agents.base import AgentFinding, AgentReport
from data_adapter import ResearchBundle


def run_timing_agent(bundle: ResearchBundle) -> AgentReport:
    findings: list[AgentFinding] = []

    if bundle.panel is not None and {"cal_ym_signal", "cal_ym_fwd"}.issubset(bundle.panel.columns):
        bad = bundle.panel[bundle.panel["cal_ym_signal"] >= bundle.panel["cal_ym_fwd"]]
        if len(bad) > 0:
            findings.append(
                AgentFinding(
                    agent="TimingAgent",
                    risk_flag="Potential lookahead in panel",
                    evidence=f"{len(bad)} rows have cal_ym_signal >= cal_ym_fwd.",
                    severity="High",
                    next_action="Re-check Part 3 forward-return merge; signal month must precede outcome month.",
                )
            )
        else:
            findings.append(
                AgentFinding(
                    agent="TimingAgent",
                    risk_flag="Forward-return timing validated",
                    evidence="All rows satisfy cal_ym_signal < cal_ym_fwd (signal t, return t+1).",
                    severity="Low",
                    next_action="Preserve this convention when adding new signals or alternative data.",
                )
            )
    else:
        findings.append(
            AgentFinding(
                agent="TimingAgent",
                risk_flag="Synthetic timing convention",
                evidence="Synthetic bundle uses month-end scores shifted before next-month returns.",
                severity="Low",
                next_action="Replace with Part 3 panel timing when SAS files are available.",
            )
        )

    return AgentReport(agent="TimingAgent", findings=findings)
