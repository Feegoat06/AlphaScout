"""Numeric guardrails for LLM memo text (no LangChain dependency)."""

from __future__ import annotations

from typing import Any

import pandas as pd


def validate_memo_numbers(memo_text: str, metrics: dict[str, float]) -> list[str]:
    """Return mismatch messages; empty means key formatted tokens appear in text."""
    mismatches: list[str] = []
    checks: list[tuple[str, str]] = []
    if "sharpe" in metrics and pd.notna(metrics["sharpe"]):
        checks.append(("sharpe", f"{metrics['sharpe']:.2f}"))
    if "ff4_alpha_monthly" in metrics and pd.notna(metrics["ff4_alpha_monthly"]):
        checks.append(("ff4_alpha_monthly", f"{metrics['ff4_alpha_monthly']:.4f}"))
    if "ff4_alpha_se" in metrics and pd.notna(metrics["ff4_alpha_se"]):
        checks.append(("ff4_alpha_se", f"{metrics['ff4_alpha_se']:.4f}"))

    for name, expected in checks:
        if expected not in memo_text:
            mismatches.append(f"Missing or mismatched {name}: expected token '{expected}' in LLM memo")
    return mismatches


def memo_draft_to_markdown(
    *,
    executive_summary: str,
    bull_case: str,
    bear_case: str,
    next_steps: list[str],
    governance_takeaway: str,
    recommended: str,
    metrics: dict[str, float],
) -> str:
    steps = (
        "\n".join(f"{i}. {s}" for i, s in enumerate(next_steps, start=1))
        or "1. Re-validate on live PIT data."
    )
    sharpe = metrics.get("sharpe", float("nan"))
    alpha = metrics.get("ff4_alpha_monthly", float("nan"))
    se = metrics.get("ff4_alpha_se", float("nan"))
    return f"""# Investment Research Memo (LLM narrative)

## Executive summary
{executive_summary}

## Recommended factor
**{recommended}** — Sharpe {sharpe:.2f}; FF4 alpha (monthly) {alpha:.4f} (SE {se:.4f}).

## Bull case
{bull_case}

## Bear case
{bear_case}

## Next steps
{steps}

## Research governance takeaway
{governance_takeaway}
""".strip()


def draft_model_to_markdown(draft: Any, *, recommended: str, metrics: dict[str, float]) -> str:
    return memo_draft_to_markdown(
        executive_summary=draft.executive_summary,
        bull_case=draft.bull_case,
        bear_case=draft.bear_case,
        next_steps=list(draft.next_steps or []),
        governance_takeaway=draft.governance_takeaway,
        recommended=recommended,
        metrics=metrics,
    )
