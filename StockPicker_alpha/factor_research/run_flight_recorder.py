#!/usr/bin/env python3
"""Run the factor research flight recorder end-to-end."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parent
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from agents.langchain.client import resolve_use_llm  # noqa: E402
from agents.orchestrator import run_research_governance  # noqa: E402
from cache import persist_research_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic Factor Research Flight Recorder")
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Skip SAS pipeline and use synthetic fallback data.",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Enable optional LangChain narrative layer (also: FACTOR_GOV_LLM=1).",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="With --with-llm, use linear orchestrator instead of LangGraph.",
    )
    args = parser.parse_args()

    use_llm = resolve_use_llm(cli_with_llm=args.with_llm)
    result = run_research_governance(
        prefer_real=not args.synthetic_only,
        use_llm=use_llm,
        use_graph=False if args.no_graph else None,
    )
    run_dir = persist_research_run(result)

    print(f"Data source: {result.bundle.data_source}")
    print(f"Factors: {', '.join(result.bundle.summary.columns)}")
    print(f"LLM layer: {'on' if use_llm else 'off'}")
    print(f"Run artifacts: {run_dir}")
    print()
    print(result.memo)
    if result.memo_llm:
        print()
        print("=== LLM memo (narrative) ===")
        print(result.memo_llm[:2000] + ("..." if len(result.memo_llm) > 2000 else ""))
    print()
    print("=== Agent findings (summary) ===")
    if result.combined_findings.empty:
        print("(none)")
    else:
        print(result.combined_findings[["agent", "risk_flag", "severity"]].to_string(index=False))


if __name__ == "__main__":
    main()
