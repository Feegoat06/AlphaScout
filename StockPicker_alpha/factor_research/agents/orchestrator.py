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


def _empty_llm_trace() -> dict[str, Any]:
    return {
        "prompt_version": None,
        "llm_model": None,
        "steps": [],
        "used_llm": False,
        "fallback_reasons": [],
    }


@dataclass
class GovernanceResult:
    bundle: ResearchBundle
    agent_reports: dict[str, pd.DataFrame] = field(default_factory=dict)
    factor_monitor_reports: dict[str, pd.DataFrame] = field(default_factory=dict)
    memo: str = ""
    combined_findings: pd.DataFrame = field(default_factory=pd.DataFrame)
    memo_llm: str | None = None
    llm_trace: dict[str, Any] = field(default_factory=_empty_llm_trace)
    deep_review: str | None = None
    use_llm: bool = False

    def to_manifest(self) -> dict[str, Any]:
        trace = self.llm_trace or {}
        return {
            "data_source": self.bundle.data_source,
            "factors": list(self.bundle.summary.columns),
            "agent_count": len(self.agent_reports),
            "high_severity_findings": int(
                (self.combined_findings["severity"] == "High").sum()
                if not self.combined_findings.empty
                else 0
            ),
            "use_llm": self.use_llm,
            "llm_model": trace.get("llm_model"),
            "prompt_version": trace.get("prompt_version"),
            "llm_used": bool(trace.get("used_llm")),
            "llm_fallback_reasons": list(trace.get("fallback_reasons") or []),
            "memo_llm": self.memo_llm is not None,
        }


class ResearchOrchestrator:
    """Coordinates specialized governance agents over a research bundle."""

    def __init__(self, *, use_llm: bool = False):
        self.use_llm = use_llm

    def run(self, bundle: ResearchBundle) -> GovernanceResult:
        trace = _empty_llm_trace()
        if self.use_llm:
            try:
                from agents.langchain.prompts import PROMPT_VERSION

                trace["prompt_version"] = PROMPT_VERSION
            except ImportError:
                trace["fallback_reasons"].append("langchain_prompts_unavailable")

        lit = run_literature_agent(bundle, use_llm=self.use_llm, trace=trace)
        agent_runs = [
            run_data_integrity_agent(bundle),
            run_timing_agent(bundle),
            run_alpha_diagnostics_agent(bundle),
            run_regime_agent(bundle),
            lit,
            run_memo_agent(bundle),
        ]
        agent_reports = {r.agent: r.to_dataframe() for r in agent_runs}
        combined = pd.concat([r.to_dataframe() for r in agent_runs], ignore_index=True)
        factor_monitor = build_monitor_reports(bundle)
        memo = build_investment_memo(bundle, factor_monitor)

        memo_llm: str | None = None
        deep_review: str | None = None
        if self.use_llm:
            try:
                from agents.langchain.chains import (
                    llm_layer_available,
                    run_deep_review_chain,
                    run_memo_chain,
                )

                if llm_layer_available(True):
                    high_n = int((combined["severity"] == "High").sum()) if not combined.empty else 0
                    if high_n > 0:
                        narrative = run_deep_review_chain(combined, trace=trace)
                        if narrative:
                            deep_review = narrative.body
                    memo_llm, _ = run_memo_chain(
                        bundle, factor_monitor, combined, memo, trace=trace
                    )
                else:
                    trace["fallback_reasons"].append("no_api_or_llm_deps")
            except Exception as exc:  # noqa: BLE001
                trace["fallback_reasons"].append(f"llm_layer_error:{type(exc).__name__}")

        return GovernanceResult(
            bundle=bundle,
            agent_reports=agent_reports,
            factor_monitor_reports=factor_monitor,
            memo=memo,
            combined_findings=combined,
            memo_llm=memo_llm,
            llm_trace=trace,
            deep_review=deep_review,
            use_llm=self.use_llm,
        )


def _result_from_graph_state(state: dict[str, Any], *, use_llm: bool) -> GovernanceResult:
    return GovernanceResult(
        bundle=state["bundle"],
        agent_reports=state.get("agent_reports") or {},
        factor_monitor_reports=state.get("monitor_reports") or {},
        memo=state.get("memo_template") or "",
        combined_findings=state.get("combined_findings", pd.DataFrame()),
        memo_llm=state.get("memo_llm"),
        llm_trace=state.get("llm_trace") or _empty_llm_trace(),
        deep_review=state.get("deep_review"),
        use_llm=use_llm,
    )


def run_research_governance(
    *,
    prefer_real: bool = True,
    use_llm: bool = False,
    use_graph: bool | None = None,
) -> GovernanceResult:
    """Run rule-first governance. Optional LangChain via use_llm / FACTOR_GOV_LLM.

    When use_llm and langgraph are available, defaults to the Phase-3 graph; set
    use_graph=False to force the linear orchestrator path.
    """
    if use_graph is None:
        use_graph = use_llm
    if use_graph and use_llm:
        try:
            from agents.langchain.governance_graph import run_governance_graph

            state = run_governance_graph(prefer_real=prefer_real, use_llm=use_llm)
            return _result_from_graph_state(state, use_llm=use_llm)
        except Exception:
            # Fall through to linear path (deps missing or graph error)
            pass

    bundle = load_research_bundle(prefer_real=prefer_real)
    return ResearchOrchestrator(use_llm=use_llm).run(bundle)
