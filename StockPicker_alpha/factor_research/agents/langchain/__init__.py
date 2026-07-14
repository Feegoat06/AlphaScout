"""Optional LangChain narrative / RAG layer for factor research governance.

Rules in monitor.py and hard-check agents remain the source of truth.
Enable via FACTOR_GOV_LLM=1 or --with-llm; falls back to templates without an API key.
"""

from __future__ import annotations

from agents.langchain.client import llm_enabled, resolve_use_llm

__all__ = ["llm_enabled", "resolve_use_llm"]
