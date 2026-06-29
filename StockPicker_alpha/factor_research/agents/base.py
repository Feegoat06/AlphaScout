from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class AgentFinding:
    agent: str
    risk_flag: str
    evidence: str
    severity: str
    next_action: str


@dataclass
class AgentReport:
    agent: str
    findings: list[AgentFinding] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.findings:
            return pd.DataFrame(columns=["agent", "risk_flag", "evidence", "severity", "next_action"])
        return pd.DataFrame([f.__dict__ for f in self.findings])
