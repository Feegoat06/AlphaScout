from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from agents.alpha_diagnostics import run_alpha_diagnostics_agent
from agents.data_integrity import run_data_integrity_agent
from agents.literature import run_literature_agent
from agents.memo import run_memo_agent
from agents.regime import run_regime_agent
from agents.timing import run_timing_agent
from data_adapter import ResearchBundle, load_research_bundle
from monitor import build_investment_memo, build_monitor_reports


@dataclass
class GovernanceResult:
    bundle: ResearchBundle
    agent_reports: dict[str, pd.DataFrame] = field(default_factory=dict)
    factor_monitor_reports: dict[str, pd.DataFrame] = field(default_factory=dict)
    memo: str = ""
    combined_findings: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "data_source": self.bundle.data_source,
            "factors": list(self.bundle.summary.columns),
            "agent_count": len(self.agent_reports),
            "high_severity_findings": int(
                (self.combined_findings["severity"] == "High").sum()
                if not self.combined_findings.empty
                else 0
            ),
        }


class ResearchOrchestrator:
    """Coordinates specialized governance agents over a research bundle."""

    def run(self, bundle: ResearchBundle) -> GovernanceResult:
        agent_runs = [
            run_data_integrity_agent(bundle),
            run_timing_agent(bundle),
            run_alpha_diagnostics_agent(bundle),
            run_regime_agent(bundle),
            run_literature_agent(bundle),
            run_memo_agent(bundle),
        ]
        agent_reports = {r.agent: r.to_dataframe() for r in agent_runs}
        combined = pd.concat([r.to_dataframe() for r in agent_runs], ignore_index=True)
        factor_monitor = build_monitor_reports(bundle)
        memo = build_investment_memo(bundle, factor_monitor)
        return GovernanceResult(
            bundle=bundle,
            agent_reports=agent_reports,
            factor_monitor_reports=factor_monitor,
            memo=memo,
            combined_findings=combined,
        )


def run_research_governance(*, prefer_real: bool = True) -> GovernanceResult:
    bundle = load_research_bundle(prefer_real=prefer_real)
    return ResearchOrchestrator().run(bundle)
