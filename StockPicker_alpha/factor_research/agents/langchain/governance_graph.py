"""Optional LangGraph governance workflow (Phase 3)."""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd

from agents.alpha_diagnostics import run_alpha_diagnostics_agent
from agents.data_integrity import run_data_integrity_agent
from agents.langchain.chains import (
    _empty_trace,
    llm_layer_available,
    run_deep_review_chain,
    run_literature_agent_llm,
    run_memo_chain,
)
from agents.literature import run_literature_agent_template
from agents.memo import run_memo_agent
from agents.regime import run_regime_agent
from agents.timing import run_timing_agent
from data_adapter import ResearchBundle, load_research_bundle
from monitor import build_investment_memo, build_monitor_reports


class GovernanceState(TypedDict, total=False):
    prefer_real: bool
    use_llm: bool
    bundle: ResearchBundle
    agent_reports: dict[str, pd.DataFrame]
    combined_findings: pd.DataFrame
    monitor_reports: dict[str, pd.DataFrame]
    memo_template: str
    memo_llm: str | None
    llm_trace: dict[str, Any]
    deep_review: str | None
    step_log: list[str]


def _concat_reports(reports: list) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    agent_reports = {r.agent: r.to_dataframe() for r in reports}
    combined = pd.concat([r.to_dataframe() for r in reports], ignore_index=True)
    return agent_reports, combined


def build_governance_graph():
    """Compile StateGraph: load → rule agents → monitor → LLM branch → done."""
    from langgraph.graph import END, StateGraph

    def load_bundle(state: GovernanceState) -> GovernanceState:
        bundle = load_research_bundle(prefer_real=state.get("prefer_real", True))
        log = list(state.get("step_log") or [])
        log.append("load_bundle")
        return {**state, "bundle": bundle, "step_log": log}

    def rule_agents(state: GovernanceState) -> GovernanceState:
        bundle = state["bundle"]
        use_llm = bool(state.get("use_llm"))
        trace = state.get("llm_trace") or _empty_trace()
        lit = (
            run_literature_agent_llm(bundle, trace=trace)
            if llm_layer_available(use_llm)
            else run_literature_agent_template(bundle)
        )
        if use_llm and not llm_layer_available(use_llm):
            trace.setdefault("fallback_reasons", []).append("literature_no_api_or_deps")

        reports = [
            run_data_integrity_agent(bundle),
            run_timing_agent(bundle),
            run_alpha_diagnostics_agent(bundle),
            run_regime_agent(bundle),
            lit,
            run_memo_agent(bundle),
        ]
        agent_reports, combined = _concat_reports(reports)
        log = list(state.get("step_log") or [])
        log.append("rule_agents")
        return {
            **state,
            "agent_reports": agent_reports,
            "combined_findings": combined,
            "llm_trace": trace,
            "step_log": log,
        }

    def monitor_node(state: GovernanceState) -> GovernanceState:
        bundle = state["bundle"]
        factor_monitor = build_monitor_reports(bundle)
        memo = build_investment_memo(bundle, factor_monitor)
        log = list(state.get("step_log") or [])
        log.append("monitor")
        return {
            **state,
            "monitor_reports": factor_monitor,
            "memo_template": memo,
            "step_log": log,
        }

    def route_after_monitor(state: GovernanceState) -> str:
        if not llm_layer_available(bool(state.get("use_llm"))):
            return "persist"
        combined = state.get("combined_findings")
        high = 0
        if combined is not None and not combined.empty:
            high = int((combined["severity"] == "High").sum())
        if high > 0:
            return "llm_deep_review"
        return "llm_memo_polish"

    def llm_deep_review(state: GovernanceState) -> GovernanceState:
        trace = state.get("llm_trace") or _empty_trace()
        narrative = run_deep_review_chain(state["combined_findings"], trace=trace)
        log = list(state.get("step_log") or [])
        log.append("llm_deep_review")
        deep = narrative.body if narrative else None
        # After deep review, still polish memo
        return {**state, "deep_review": deep, "llm_trace": trace, "step_log": log}

    def llm_memo_polish(state: GovernanceState) -> GovernanceState:
        trace = state.get("llm_trace") or _empty_trace()
        try:
            memo_llm, _guardrail_fallback = run_memo_chain(
                state["bundle"],
                state["monitor_reports"],
                state["combined_findings"],
                state["memo_template"],
                trace=trace,
            )
        except Exception as exc:  # noqa: BLE001 — graceful fallback
            trace.setdefault("fallback_reasons", []).append(f"memo_error:{type(exc).__name__}")
            memo_llm = None
        log = list(state.get("step_log") or [])
        log.append("llm_memo_polish")
        return {**state, "memo_llm": memo_llm, "llm_trace": trace, "step_log": log}

    def persist(state: GovernanceState) -> GovernanceState:
        log = list(state.get("step_log") or [])
        log.append("persist")
        trace = state.get("llm_trace") or _empty_trace()
        trace["graph_steps"] = list(log)
        return {**state, "step_log": log, "llm_trace": trace}

    g = StateGraph(GovernanceState)
    g.add_node("load_bundle", load_bundle)
    g.add_node("rule_agents", rule_agents)
    g.add_node("monitor", monitor_node)
    g.add_node("llm_deep_review", llm_deep_review)
    g.add_node("llm_memo_polish", llm_memo_polish)
    g.add_node("persist", persist)

    g.set_entry_point("load_bundle")
    g.add_edge("load_bundle", "rule_agents")
    g.add_edge("rule_agents", "monitor")
    g.add_conditional_edges(
        "monitor",
        route_after_monitor,
        {
            "llm_deep_review": "llm_deep_review",
            "llm_memo_polish": "llm_memo_polish",
            "persist": "persist",
        },
    )
    g.add_edge("llm_deep_review", "llm_memo_polish")
    g.add_edge("llm_memo_polish", "persist")
    g.add_edge("persist", END)
    return g.compile()


def run_governance_graph(*, prefer_real: bool = True, use_llm: bool = False) -> GovernanceState:
    graph = build_governance_graph()
    initial: GovernanceState = {
        "prefer_real": prefer_real,
        "use_llm": use_llm,
        "llm_trace": _empty_trace(),
        "memo_llm": None,
        "deep_review": None,
        "step_log": [],
    }
    return graph.invoke(initial)
