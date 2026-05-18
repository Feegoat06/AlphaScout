"""
Part 2 — Signal selection and economic sign.

Run: python part2_signals.py
"""

from __future__ import annotations

# Four signals — one per required category — with economic rationale (assignment Part 2).
SELECTED_SIGNALS: list[dict[str, object]] = [
    {
        "column": "momentum",
        "category": "Momentum or reversal",
        "economic_idea": (
            "Past relative strength over a recent horizon; captures gradual information "
            "diffusion / underreaction and related risk narratives for continuation."
        ),
        "higher_value_predicts": "higher near- to medium-term future returns (momentum premium).",
        "flip_sign_for_higher_means_expected_return": False,
        "flip_note": "Higher momentum already aligns with higher expected return; rank long high / short low without negating.",
    },
    {
        "column": "BtM",
        "category": "Value",
        "economic_idea": (
            "Book-to-market: high ratios indicate ’cheap’ equities vs. book fundamentals "
            "(value tilt), as in classical value-factor discussions."
        ),
        "higher_value_predicts": "higher average future returns (value premium).",
        "flip_sign_for_higher_means_expected_return": False,
        "flip_note": "High BtM is ‘value’ and typically commands higher subsequent returns — no −1 multiplier.",
    },
    {
        "column": "ROA",
        "category": "Profitability or quality",
        "economic_idea": (
            "Return on assets proxies for operating profitability / quality — firms "
            "with stronger realized profitability often earn persistent cross-sectional premia."
        ),
        "higher_value_predicts": "higher future returns on average (profitability / quality tilt).",
        "flip_sign_for_higher_means_expected_return": False,
        "flip_note": "More profitable firms map to stronger expected returns in standard anomaly framing — no −1 multiplier.",
    },
    {
        "column": "ivol",
        "category": "Risk, distress, or limits to arbitrage",
        "economic_idea": (
            "Idiosyncratic volatility measures stock-specific risk; linked to arbitrage limits, "
            "lottery-demand, and mispricing mechanisms in asset-pricing anomalies literature."
        ),
        "higher_value_predicts": "lower subsequent returns in the canonical U.S. idiosyncratic-volatility puzzle.",
        "flip_sign_for_higher_means_expected_return": True,
        "flip_note": (
            "Use −ivol (or multiply by −1 after standardization) so that a higher composite "
            "score consistently means higher expected return, matching other signals."
        ),
    },
]


def formatted_report() -> str:
    divider = "-" * 70
    blocks: list[str] = []
    for i, sig in enumerate(SELECTED_SIGNALS, start=1):
        flip = "Yes (\u2212signal or score \u2261 \u2212ivol)" if sig[
            "flip_sign_for_higher_means_expected_return"
        ] else "No"

        blocks.append(
            "\n".join(
                [
                    divider,
                    f"{i}. Signal: `{sig['column']}` \u2014 {sig['category']}",
                    divider,
                    f"(1) Economic idea:\n    {sig['economic_idea']}",
                    "",
                    "(2) Direction: A higher raw value should predict:",
                    f"    \u2192 {sig['higher_value_predicts']}",
                    "",
                    '(3) Multiply by \u22121 so "higher score" \u21d2 "higher expected return"?',
                    f"    {flip}",
                    f"    {sig['flip_note']}",
                ]
            )
        )

    intro_lines = [
        "=" * 80,
        "Part 2. Signal selection and economic sign",
        "=" * 80,
        "Chosen signals (four categories):",
        "  * Momentum or reversal \u2192 momentum",
        "  * Value \u2192 BtM",
        "  * Profitability or quality \u2192 ROA",
        "  * Risk / limits to arbitrage \u2192 ivol (\u2212sign when pooling with others)",
        "",
        "For each signal: economic idea; predicted direction; whether to flip sign.",
        "=" * 80,
    ]

    return "\n\n".join(["\n".join(intro_lines), "\n\n".join(blocks), ""])


def main() -> None:
    print(formatted_report())


if __name__ == "__main__":
    main()
