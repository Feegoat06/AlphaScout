"""Parquet cache and reproducible research-run manifests."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from agents.orchestrator import GovernanceResult
from data_adapter import ResearchBundle

DEMO_ROOT = Path(__file__).resolve().parent
CACHE_DIR = DEMO_ROOT / "data" / "cache"
RUNS_DIR = DEMO_ROOT / "runs"


def _git_hash() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=DEMO_ROOT.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _to_parquet_or_csv(df: pd.DataFrame, parquet_path: Path) -> Path:
    try:
        df.to_parquet(parquet_path)
        return parquet_path
    except ImportError:
        csv_path = parquet_path.with_suffix(".csv")
        df.to_csv(csv_path)
        return csv_path


def write_panel_cache(bundle: ResearchBundle) -> Path | None:
    if bundle.panel is None:
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "panel_essential.parquet"
    return _to_parquet_or_csv(bundle.panel, path)


def write_ls_cache(bundle: ResearchBundle) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "factor_ls_returns.parquet"
    return _to_parquet_or_csv(bundle.summary, path)


def persist_research_run(result: GovernanceResult) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    write_ls_cache(result.bundle)
    panel_path = write_panel_cache(result.bundle)

    metrics_path = run_dir / "metrics.parquet"
    _to_parquet_or_csv(result.bundle.metrics_df.reset_index(), metrics_path)
    findings_path = run_dir / "agent_findings.parquet"
    _to_parquet_or_csv(result.combined_findings, findings_path)

    for factor, report in result.factor_monitor_reports.items():
        safe = factor.replace("/", "_")
        _to_parquet_or_csv(report, run_dir / f"monitor_{safe}.parquet")

    manifest: dict[str, Any] = {
        "timestamp_utc": ts,
        "git_hash": _git_hash(),
        **result.to_manifest(),
        "panel_cache": str(panel_path) if panel_path else None,
        "ls_cache": str(CACHE_DIR / "factor_ls_returns.parquet"),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "memo.md").write_text(result.memo, encoding="utf-8")
    metrics_json = result.bundle.metrics_df.reset_index().to_dict(orient="records")
    (run_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    return run_dir
