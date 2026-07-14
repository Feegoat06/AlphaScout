"""Lightweight unit checks for numeric memo guardrails (no API key required)."""

from __future__ import annotations

import sys
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from agents.langchain.guardrails import memo_draft_to_markdown, validate_memo_numbers


def test_validate_memo_numbers_ok() -> None:
    metrics = {
        "sharpe": 0.46,
        "ff4_alpha_monthly": 0.0231,
        "ff4_alpha_se": 0.0100,
    }
    text = memo_draft_to_markdown(
        executive_summary="BtM looks reasonable.",
        bull_case="Cyclical value premium.",
        bear_case="Long droughts possible.",
        next_steps=["Validate PIT data"],
        governance_takeaway="Trust process over story.",
        recommended="BtM",
        metrics=metrics,
    )
    assert validate_memo_numbers(text, metrics) == []


def test_validate_memo_numbers_mismatch() -> None:
    metrics = {"sharpe": 0.46, "ff4_alpha_monthly": 0.0231, "ff4_alpha_se": 0.0100}
    bad = "# memo\nSharpe 9.99 and alpha 1.0000 (SE 0.0001)"
    mismatches = validate_memo_numbers(bad, metrics)
    assert mismatches


if __name__ == "__main__":
    test_validate_memo_numbers_ok()
    test_validate_memo_numbers_mismatch()
    print("ok")
