"""Pydantic structured outputs for the optional LangChain layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LiteratureNote(BaseModel):
    """RAG-grounded literature summary for one factor."""

    factor: str
    summary: str = Field(description="Concise literature synthesis grounded in retrieved context.")
    sources: list[str] = Field(
        default_factory=list,
        description="Doc ids / titles cited from retrieved context only.",
    )
    caveat: str = Field(
        default="",
        description="Key risk or open question; use 'insufficient literature coverage' if context is thin.",
    )


class MemoDraft(BaseModel):
    """PM-facing narrative; numbers must match the deterministic metrics bundle."""

    executive_summary: str
    bull_case: str
    bear_case: str
    next_steps: list[str] = Field(default_factory=list)
    governance_takeaway: str


class AgentNarrative(BaseModel):
    """Optional short natural-language summary (e.g. regime or deep review)."""

    title: str
    body: str
    severity_view: str = Field(
        default="Low",
        description="Does not override rule severity; informational only.",
    )
