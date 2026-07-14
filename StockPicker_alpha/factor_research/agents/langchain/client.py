"""Lazy LLM client factory. Keys come from the environment only."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any


DEFAULT_MODEL = os.environ.get("FACTOR_GOV_LLM_MODEL", "gpt-4o-mini")


def env_flag_enabled() -> bool:
    return os.environ.get("FACTOR_GOV_LLM", "0").strip().lower() in {"1", "true", "yes", "on"}


def resolve_use_llm(*, cli_with_llm: bool = False) -> bool:
    """True when CLI --with-llm or FACTOR_GOV_LLM=1."""
    return bool(cli_with_llm) or env_flag_enabled()


def llm_enabled(*, use_llm: bool = False) -> bool:
    """Ready to call an LLM: flag on and API key present."""
    if not use_llm:
        return False
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


@lru_cache(maxsize=1)
def get_chat_model(*, model: str | None = None, max_tokens: int = 800) -> Any:
    """Lazy-init ChatOpenAI. ImportError / missing key → raises for caller to fall back."""
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is not set")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or DEFAULT_MODEL,
        temperature=0,
        max_tokens=max_tokens,
    )


def get_embeddings(*, model: str = "text-embedding-3-small") -> Any:
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is not set")
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=model)


def reset_client_cache() -> None:
    get_chat_model.cache_clear()
