"""LCEL chains for literature RAG and memo polish."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from agents.base import AgentFinding, AgentReport
from agents.langchain.client import DEFAULT_MODEL, get_chat_model, llm_enabled
from agents.langchain.guardrails import draft_model_to_markdown, validate_memo_numbers
from agents.langchain.prompts import (
    PROMPT_VERSION,
    deep_review_prompt,
    literature_prompt,
    memo_prompt,
)
from agents.langchain.retriever import format_docs, retrieve_for_factor, source_ids
from data_adapter import ResearchBundle
from monitor import choose_best_factor


def _hash_payload(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _empty_trace() -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "llm_model": None,
        "steps": [],
        "used_llm": False,
        "fallback_reasons": [],
    }


def _metrics_snapshot(bundle: ResearchBundle, factor: str) -> dict[str, float]:
    m = bundle.metrics_df.loc[factor]
    keys = ["sharpe", "max_drawdown", "hit_rate", "avg_turnover", "ff4_alpha_monthly", "ff4_alpha_se"]
    out: dict[str, float] = {}
    for k in keys:
        if k in m.index:
            try:
                out[k] = float(m.get(k))
            except (TypeError, ValueError):
                continue
    return out


def run_literature_agent_llm(bundle: ResearchBundle, *, trace: dict[str, Any] | None = None) -> AgentReport:
    """RAG + structured LiteratureNote → AgentFindings. Raises on hard failure for caller fallback."""
    from agents.langchain.schemas import LiteratureNote

    llm = get_chat_model()
    chain = literature_prompt() | llm.with_structured_output(LiteratureNote)
    findings: list[AgentFinding] = []
    step_records: list[dict[str, Any]] = []

    for factor in bundle.summary.columns:
        hypothesis = bundle.factor_hypotheses.get(factor, "")
        docs = retrieve_for_factor(str(factor), k=4)
        context = format_docs(docs)
        payload = {"factor": str(factor), "hypothesis": hypothesis, "context": context}
        note: LiteratureNote = chain.invoke(payload)
        sources = note.sources or source_ids(docs)
        evidence = note.summary
        if note.caveat:
            evidence = f"{evidence} | caveat: {note.caveat}"
        if sources:
            evidence = f"{evidence} | sources: {', '.join(sources)}"
        findings.append(
            AgentFinding(
                agent="LiteratureAgent",
                risk_flag=f"{factor}: academic prior (LLM+RAG)",
                evidence=evidence,
                severity="Low",
                next_action="Compare backtest sign and magnitude against published anomaly baselines.",
            )
        )
        step_records.append(
            {
                "step": "literature",
                "factor": str(factor),
                "input_hash": _hash_payload({"factor": factor, "hypothesis": hypothesis}),
                "sources": sources,
                "caveat": note.caveat,
            }
        )

    if trace is not None:
        trace["used_llm"] = True
        trace["llm_model"] = DEFAULT_MODEL
        trace["steps"].extend(step_records)

    return AgentReport(agent="LiteratureAgent", findings=findings)


def build_memo_input_bundle(
    bundle: ResearchBundle,
    monitor_reports: dict[str, pd.DataFrame],
    combined_findings: pd.DataFrame,
    *,
    recommended: str | None = None,
) -> dict[str, Any]:
    if recommended is None:
        recommended, _ = choose_best_factor(bundle.metrics_df, monitor_reports, exclude={"composite"})
    flags: list[dict[str, str]] = []
    report = monitor_reports.get(recommended, pd.DataFrame())
    if not report.empty:
        for _, row in report.iterrows():
            flags.append(
                {
                    "risk_flag": str(row.get("risk_flag", "")),
                    "severity": str(row.get("severity", "")),
                }
            )
    findings_brief: list[str] = []
    if combined_findings is not None and not combined_findings.empty:
        for _, row in combined_findings.head(20).iterrows():
            findings_brief.append(
                f"{row.get('agent')}: {row.get('risk_flag')} [{row.get('severity')}]"
            )
    metrics = _metrics_snapshot(bundle, recommended)
    return {
        "data_source": bundle.data_source,
        "recommended_factor": recommended,
        "metrics": metrics,
        "monitor_flags": flags,
        "agent_findings": findings_brief,
        "part2_hypothesis": bundle.factor_hypotheses.get(recommended, ""),
    }


def run_memo_chain(
    bundle: ResearchBundle,
    monitor_reports: dict[str, pd.DataFrame],
    combined_findings: pd.DataFrame,
    memo_template: str,
    *,
    trace: dict[str, Any] | None = None,
) -> tuple[str | None, bool]:
    """Return (memo_llm_markdown, used_template_due_to_guardrail).

    On numeric mismatch, returns warning-banner markdown and signals fallback to template as authoritative.
    """
    recommended, _ = choose_best_factor(bundle.metrics_df, monitor_reports, exclude={"composite"})
    metrics = _metrics_snapshot(bundle, recommended)
    input_bundle = build_memo_input_bundle(
        bundle, monitor_reports, combined_findings, recommended=recommended
    )
    from agents.langchain.schemas import MemoDraft

    llm = get_chat_model()
    chain = memo_prompt() | llm.with_structured_output(MemoDraft)
    invoke_payload = {
        "bundle_json": json.dumps(input_bundle, indent=2, default=str),
        "sharpe": f"{metrics.get('sharpe', float('nan')):.2f}",
        "max_drawdown": f"{metrics.get('max_drawdown', float('nan'))}",
        "hit_rate": f"{metrics.get('hit_rate', float('nan'))}",
        "avg_turnover": f"{metrics.get('avg_turnover', float('nan'))}",
        "ff4_alpha_monthly": f"{metrics.get('ff4_alpha_monthly', float('nan')):.4f}",
        "ff4_alpha_se": f"{metrics.get('ff4_alpha_se', float('nan')):.4f}",
        "memo_template": memo_template,
    }
    draft = chain.invoke(invoke_payload)
    memo_llm = draft_model_to_markdown(draft, recommended=recommended, metrics=metrics)
    mismatches = validate_memo_numbers(memo_llm, metrics)
    if trace is not None:
        trace["used_llm"] = True
        trace["llm_model"] = DEFAULT_MODEL
        trace["steps"].append(
            {
                "step": "memo_polish",
                "input_hash": _hash_payload(input_bundle),
                "mismatches": mismatches,
            }
        )

    if mismatches:
        banner = (
            "> **WARNING:** numeric guardrail failed — treating template memo as source of truth.\n>\n> "
            + "; ".join(mismatches)
            + "\n\n"
        )
        if trace is not None:
            trace.setdefault("fallback_reasons", []).append("numeric_guardrail")
        return banner + memo_llm, True

    return memo_llm, False


def run_deep_review_chain(
    combined_findings: pd.DataFrame,
    *,
    trace: dict[str, Any] | None = None,
) -> Any | None:
    if combined_findings is None or combined_findings.empty:
        return None
    high = combined_findings[combined_findings["severity"] == "High"]
    if high.empty:
        return None
    from agents.langchain.schemas import AgentNarrative

    llm = get_chat_model()
    chain = deep_review_prompt() | llm.with_structured_output(AgentNarrative)
    payload = high[["agent", "risk_flag", "evidence", "severity", "next_action"]].to_dict(orient="records")
    narrative = chain.invoke({"findings_json": json.dumps(payload, indent=2, default=str)})
    if trace is not None:
        trace["steps"].append(
            {
                "step": "llm_deep_review",
                "input_hash": _hash_payload(payload),
                "title": narrative.title,
            }
        )
    return narrative


def llm_layer_available(use_llm: bool) -> bool:
    if not use_llm:
        return False
    if not llm_enabled(use_llm=True):
        return False
    try:
        import langchain_openai  # noqa: F401
    except ImportError:
        return False
    return True
