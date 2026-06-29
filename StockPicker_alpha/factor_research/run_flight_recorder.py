#!/usr/bin/env python3
"""Run the factor research flight recorder end-to-end."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parent
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from agents.orchestrator import run_research_governance  # noqa: E402
from cache import persist_research_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic Factor Research Flight Recorder")
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Skip SAS pipeline and use synthetic fallback data.",
    )
    args = parser.parse_args()

    result = run_research_governance(prefer_real=not args.synthetic_only)
    run_dir = persist_research_run(result)

    print(f"Data source: {result.bundle.data_source}")
    print(f"Factors: {', '.join(result.bundle.summary.columns)}")
    print(f"Run artifacts: {run_dir}")
    print()
    print(result.memo)
    print()
    print("=== Agent findings (summary) ===")
    if result.combined_findings.empty:
        print("(none)")
    else:
        print(result.combined_findings[["agent", "risk_flag", "severity"]].to_string(index=False))


if __name__ == "__main__":
    main()
