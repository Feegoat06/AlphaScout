from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from agents.base import AgentFinding, AgentReport
from data_adapter import ResearchBundle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_data_integrity_agent(bundle: ResearchBundle) -> AgentReport:
    findings: list[AgentFinding] = []
    audits = bundle.audits or {}

    for row in audits.get("required_fields", []):
        if not row.get("required_present", True):
            findings.append(
                AgentFinding(
                    agent="DataIntegrityAgent",
                    risk_flag="Missing required columns",
                    evidence=f"{row.get('dataset')}: missing {row.get('missing_required')}",
                    severity="High",
                    next_action="Fix SAS inputs before any factor research proceeds.",
                )
            )

    for row in audits.get("duplicate_keys", []):
        dup = row.get("n_keys_with_duplicates", 0)
        if isinstance(dup, (int, float)) and dup > 0:
            findings.append(
                AgentFinding(
                    agent="DataIntegrityAgent",
                    risk_flag="Duplicate merge keys",
                    evidence=(
                        f"{row.get('dataset')}: {dup} duplicate keys "
                        f"({row.get('duplicate_check')})."
                    ),
                    severity="Medium",
                    next_action="Deduplicate or document aggregation rule before panel merge.",
                )
            )

    if bundle.panel is not None:
        na_ret = int(bundle.panel["ret_fwd"].isna().sum()) if "ret_fwd" in bundle.panel.columns else 0
        if na_ret > 0:
            findings.append(
                AgentFinding(
                    agent="DataIntegrityAgent",
                    risk_flag="Missing forward returns",
                    evidence=f"{na_ret} panel rows have NaN ret_fwd after Step 3 merge.",
                    severity="Medium",
                    next_action="Review CRSP coverage gaps and listwise deletion policy in estimation.",
                )
            )

    if not findings:
        findings.append(
            AgentFinding(
                agent="DataIntegrityAgent",
                risk_flag="Data integrity checks passed",
                evidence="Required fields present and no blocking duplicate-key issues detected.",
                severity="Low",
                next_action="Proceed to timing and factor diagnostics.",
            )
        )

    return AgentReport(agent="DataIntegrityAgent", findings=findings)
