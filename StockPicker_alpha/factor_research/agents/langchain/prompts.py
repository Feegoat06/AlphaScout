"""Versioned prompt templates for literature RAG and memo polish."""

from __future__ import annotations

from typing import Any

PROMPT_VERSION = "v1"

LITERATURE_SYSTEM = """You are a factor-research literature assistant for an asset-management governance layer.
Rules:
- Cite ONLY the retrieved context below. Do not invent papers or results.
- If context is insufficient for the factor, set caveat to "insufficient literature coverage"
  and keep the summary brief and cautious.
- Do not invent Sharpe, alpha, or backtest numbers.
- Output must match the structured schema."""

LITERATURE_HUMAN = """Factor: {factor}
Part-2 economic hypothesis: {hypothesis}

Retrieved literature context:
{context}

Produce a LiteratureNote grounded in the context."""

MEMO_SYSTEM = """You are polishing an investment research memo for a PM.
Rules:
- Rules decide; you only explain. Do not invent or alter metrics.
- Every numeric claim must match the provided metrics JSON exactly
  (use the same rounding: Sharpe 2 decimals, drawdown/hit/turnover as percent-ready floats,
  FF4 alpha and SE to 4 decimals as given).
- Do not downgrade High severity findings.
- Be concise and professional; no marketing fluff."""

MEMO_HUMAN = """Write a MemoDraft from this structured research bundle:

{bundle_json}

Recommended factor metrics to echo exactly:
- sharpe: {sharpe}
- max_drawdown: {max_drawdown}
- hit_rate: {hit_rate}
- avg_turnover: {avg_turnover}
- ff4_alpha_monthly: {ff4_alpha_monthly}
- ff4_alpha_se: {ff4_alpha_se}

Template memo (numbers are authoritative — align prose with these):
{memo_template}
"""

DEEP_REVIEW_SYSTEM = """Summarize blocking High-severity governance findings for a PM.
Do not invent new flags or change severity. Cite evidence briefly."""

DEEP_REVIEW_HUMAN = """High-severity findings (JSON):
{findings_json}

Produce an AgentNarrative titled "Deep governance review"."""


def _chat_prompt(system: str, human: str) -> Any:
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )


def literature_prompt() -> Any:
    return _chat_prompt(LITERATURE_SYSTEM, LITERATURE_HUMAN)


def memo_prompt() -> Any:
    return _chat_prompt(MEMO_SYSTEM, MEMO_HUMAN)


def deep_review_prompt() -> Any:
    return _chat_prompt(DEEP_REVIEW_SYSTEM, DEEP_REVIEW_HUMAN)
