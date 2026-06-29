"""
Part 3 — Return-prediction dataset (built step-by-step).

STEP 1 (this script): convert every observation date into one consistent calendar-month ID.
Uses the same month bucketing idea as ``part1_script.ym_key`` (Period M -> 'YYYY-MM').

STEP 2: merge ``signals_raw_plus`` to CRSP-like ``msf`` on ``PERMNO`` + same-calendar ``cal_ym``
(diagnostic contemporaneous pairing).

STEP 3: attach **future** CRSP monthly return ``ret_fwd`` — signal formation month ``t`` maps to ``RET`` in ``t+1``.

STEP 4: merge Fama-French/Carhart **factor rows** onto the Step-3 panel by calendar month —
``cal_ym_fwd`` (the month ``ret_fwd`` is realised in).

STEP 5: **excess return** ``excess_ret = ret_fwd - rf`` (same calendar month as ``ret_fwd``).

STEP 6 (documentation): ``print_question6_documentation`` — short policy summary + audits.

Deliverable subset: ``prediction_panel_essential(final_panel)`` drops non-essential ``signals_raw_plus`` /
``msf`` fields; Steps 3–5 still return the full merged frame for reproducibility.

Run:
    python part3_script.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

PATHS = {
    "signals_raw_plus": ROOT / "signals_raw_plus.sas7bdat",
    "msf": ROOT / "msf.sas7bdat",
    "factors_monthly": ROOT / "factors_monthly.sas7bdat",
}

# Factor columns appended in Step 4 (monthly observables on ``factors_monthly``).
FACTOR_COLS_FOR_PANEL = ["date", "mktrf", "smb", "hml", "umd", "rf"]

# Narrow deliverable: keys, timing, outcomes, factors, default Part~2 predictors (if present).
PART3_DEFAULT_PREDICTORS: tuple[str, ...] = ("momentum", "BtM", "ROA", "ivol")

PART3_ESSENTIAL_CORE: tuple[str, ...] = (
    "PERMNO",
    "fdate",
    "cal_ym_signal",
    "DATE_fwd",
    "cal_ym_fwd",
    "ret_fwd",
    "factor_date",
    "mktrf",
    "smb",
    "hml",
    "umd",
    "rf",
    "excess_ret",
)


def prediction_panel_essential(
    panel: pd.DataFrame,
    *,
    predictors: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Keep only columns needed for return-prediction / risk adjustment (Part~3 spec).

    Returns a **copy** with core keys, timing, ``ret_fwd``, factors, ``excess_ret`` (when present),
    and any predictors that exist (default: Part~2-style names).
    """
    pred = tuple(predictors) if predictors is not None else PART3_DEFAULT_PREDICTORS
    want: list[str] = []
    for c in PART3_ESSENTIAL_CORE:
        if c in panel.columns:
            want.append(c)
    for p in pred:
        if p in panel.columns and p not in want:
            want.append(p)
    return panel[want].copy()


## Step 1
def load_sas(path: Path) -> pd.DataFrame:
    return pd.read_sas(path, format="sas7bdat", encoding="utf-8")


def add_calendar_month_id(df: pd.DataFrame, *, date_column: str, id_column: str = "cal_ym") -> pd.DataFrame:
    """
    Attach a timezone-naive **calendar-month** bucket as ``id_column``: ``YYYY-MM``.

    The same logical month gets the same id whether ``date_column`` is a month-end
    trading day (``fdate`` / ``DATE``) or month-start factor ``date`` (e.g. 1926-07-01
    maps to ``1926-07``).
    """
    out = df.copy()
    ts = pd.to_datetime(out[date_column], errors="coerce")
    out[id_column] = ts.dt.to_period("M").astype(str)
    return out


def step1_tables_with_cal_ym() -> dict[str, pd.DataFrame]:
    """Load the three SAS tables and attach ``cal_ym``."""

    configs: tuple[tuple[str, Path, str], ...] = (
        ("signals_raw_plus", PATHS["signals_raw_plus"], "fdate"),
        ("msf", PATHS["msf"], "DATE"),
        ("factors_monthly", PATHS["factors_monthly"], "date"),
    )

    loaded: dict[str, pd.DataFrame] = {}

    for name, path, dcol in configs:
        if not path.exists():
            raise FileNotFoundError(path)
        raw = load_sas(path)
        if dcol not in raw.columns:
            raise KeyError(f"{name}: missing `{dcol}`")

        loaded[name] = add_calendar_month_id(raw, date_column=dcol, id_column="cal_ym")

    return loaded

## This function is to show the transformation of the data into the calendar month id (YYYY-MM)
def print_step1_summary(tables: dict[str, pd.DataFrame]) -> None:
    rows = []
    for name, tbl in tables.items():
        ym = tbl["cal_ym"]
        bad_mask = ym.isna() | (ym.astype(str) == "NaT") | ym.astype(str).str.lower().isin({"nan"})
        rows.append(
            {
                "dataset": name,
                "n_rows": len(tbl),
                "cal_ym_min": ym.dropna().min() if ym.notna().any() else None,
                "cal_ym_max": ym.dropna().max() if ym.notna().any() else None,
                "n_bad_cal_ym": int(bad_mask.sum()),
            }
        )

    print("=== Step 1: consistent monthly id `cal_ym` (YYYY-MM) ===\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nSamples (native date vs cal_ym):")
    sig = tables["signals_raw_plus"]
    ms = tables["msf"]
    ff = tables["factors_monthly"]
    print("\nsignals_raw_plus:\n", sig[["PERMNO", "fdate", "cal_ym"]].head(3).to_string(index=False))
    print("\nmsf:\n", ms[["PERMNO", "DATE", "cal_ym"]].head(3).to_string(index=False))
    print("\nfactors_monthly:\n", ff[["date", "cal_ym", "mktrf", "rf"]].head(3).to_string(index=False))


## Step 2
def step2_merge_signals_to_msf(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Join signal panel to monthly CRSP-style returns using ``PERMNO`` + ``cal_ym``.

    This lines up the **signal month** and the CRSP observation whose ``DATE``
    falls in the **same calendar month** (same ``cal_ym``). Subsequent steps rename
    or relink ``RET`` to the **next** month's return when required for forecasting.
    """
    sig = tables["signals_raw_plus"].copy()
    ms = tables["msf"].copy()

    ms_keys = ms[["PERMNO", "cal_ym", "DATE", "RET"]].copy()

    ## auditting purpose: test if the code is working correctly
    before_sig, before_msf = len(sig), len(ms_keys)

    merged = sig.merge(
        ms_keys,
        on=["PERMNO", "cal_ym"],
        how="inner",
        validate="many_to_one",
    )

    merged["RET"] = pd.to_numeric(merged["RET"], errors="coerce")

    audit = {
        "signals_rows_before": before_sig,
        "msf_key_rows_before": len(ms_keys),
        "merged_rows_inner": len(merged),
        "signals_not_matched_approx": max(0, before_sig - len(merged)),
    }
    dup_msf = ms_keys.duplicated(subset=["PERMNO", "cal_ym"]).sum()
    audit["duplicate_permno_month_in_msf_mini"] = int(dup_msf)

    return merged, audit


def print_step2_summary(merged: pd.DataFrame, audit: dict[str, int]) -> None:
    print("\n=== Step 2: signals merged to CRSP ``msf`` on PERMNO + cal_ym ===\n")
    print(pd.Series(audit).to_string())
    print("\nMerged shape:", merged.shape)

    pred = [c for c in PART3_DEFAULT_PREDICTORS if c in merged.columns]
    cols_show = ["PERMNO", "fdate", "DATE", "cal_ym", "RET", *pred]

    print("\nPreview (essential columns only):\n")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 14)
    one_perm = merged["PERMNO"].iloc[0] if len(merged) else None
    demo = merged.loc[merged["PERMNO"] == one_perm].head(8) if one_perm is not None else merged.head(8)
    use = [c for c in cols_show if c in demo.columns]
    print(demo[use].to_string(index=False))
    print(f"\n(PERMNO == {one_perm}, up to 8 rows; merged has {len(merged)} rows × {merged.shape[1]} columns.)\n")


## Step 3 — prediction target: month t -> t+1
def step3_construct_forward_returns(
    tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Rows dated in signals' calendar month ``t`` attach ``RET`` from ``msf`` in calendar month ``t+1`` as ``ret_fwd``
    (+ ``DATE_fwd``). Drops signal rows with no next-month CRSP observation (``inner``).
    """
    sig = tables["signals_raw_plus"].copy()
    ms = tables["msf"][["PERMNO", "cal_ym", "DATE", "RET"]].copy()

    sig = sig.rename(columns={"cal_ym": "cal_ym_signal"})
    ym_sig_ts = pd.to_datetime(sig["cal_ym_signal"] + "-01", errors="coerce").dt.to_period("M")
    sig["cal_ym_fwd"] = (ym_sig_ts + 1).astype(str)

    ms = ms.rename(columns={"cal_ym": "cal_ym_fwd", "DATE": "DATE_fwd", "RET": "ret_fwd"})

    merged = sig.merge(ms, on=["PERMNO", "cal_ym_fwd"], how="inner", validate="many_to_one")
    merged["ret_fwd"] = pd.to_numeric(merged["ret_fwd"], errors="coerce")

    dup_msf = ms.duplicated(subset=["PERMNO", "cal_ym_fwd"]).sum()
    audit = {
        "signals_rows_before_merge": len(sig),
        "msf_rows_avail": len(ms),
        "panel_rows_fwd_ret": len(merged),
        "signals_dropped_no_fwd_msf": len(sig) - len(merged),
        "duplicate_permno_fwd_month_msf": int(dup_msf),
    }
    return merged, audit


def print_step3_summary(panel: pd.DataFrame, audit: dict[str, int]) -> None:
    print("\n=== Step 3: forward one-month returns (signal t -> RET in t+1) ===\n")
    print(pd.Series(audit).to_string())
    slim = prediction_panel_essential(panel)
    print(f"\nPanel shape: {panel.shape[0]} × {panel.shape[1]} (full) | essential preview: {slim.shape[1]} columns")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 14)
    print("\nPreview (essential columns):\n")
    one_perm = panel["PERMNO"].iloc[0] if len(panel) else None
    demo = slim.loc[slim["PERMNO"] == one_perm].head(8) if one_perm is not None else slim.head(8)
    print(demo.to_string(index=False))
    print(f"\n(PERMNO == {one_perm}, up to 8 rows; N = {len(panel)}.)\n")


## Step 4 — FF/Carhart factors by return month (no PERMNO on factor table)
def step4_merge_factor_returns_by_month(
    panel: pd.DataFrame, tables: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Attach one factor row per row of ``panel`` using ``cal_ym_fwd`` ≡ ``factors_monthly.cal_ym``.
    ``factor_date`` renames FF ``date`` to avoid clashes.
    """
    want = [*FACTOR_COLS_FOR_PANEL, "cal_ym"]
    ff = tables["factors_monthly"][want].copy()
    ff = ff.rename(columns={"date": "factor_date"})
    dup_ff = int(ff.duplicated(subset=["cal_ym"]).sum())

    merged = panel.merge(ff, left_on="cal_ym_fwd", right_on="cal_ym", how="left", validate="m:1")
    merged = merged.drop(columns=["cal_ym"])

    for c in ["mktrf", "smb", "hml", "umd", "rf"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    audit = {
        "panel_rows_before": len(panel),
        "panel_rows_after": len(merged),
        "dup_factor_cal_ym_rows": dup_ff,
        "rows_missing_rf_after_merge": int(merged["rf"].isna().sum()),
    }
    return merged, audit


def print_step4_summary(panel: pd.DataFrame, audit: dict[str, int]) -> None:
    print("\n=== Step 4: merge factor returns by month (aligned to ``cal_ym_fwd``) ===\n")
    print(pd.Series(audit).to_string())
    slim = prediction_panel_essential(panel)
    print(f"\nPanel shape: {panel.shape[0]} × {panel.shape[1]} (full) | essential preview: {slim.shape[1]} columns")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 13)
    one_perm = panel["PERMNO"].iloc[0] if len(panel) else None
    demo = slim.loc[slim["PERMNO"] == one_perm].head(6) if one_perm is not None else slim.head(6)
    print("\nPreview (essential columns):\n")
    print(demo.to_string(index=False))
    print(f"\n(PERMNO == {one_perm}, up to 6 rows; N = {len(panel)}.)\n")


## Step 5 — excess return on the forward holding month
def step5_construct_excess_returns(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    ``excess_ret = ret_fwd - rf`` with both series in the same units as provided in the SAS files
    (decimal per month, not percent).
    """
    out = panel.copy()
    r = pd.to_numeric(out["ret_fwd"], errors="coerce")
    rf = pd.to_numeric(out["rf"], errors="coerce")
    out["excess_ret"] = r - rf

    both = r.notna() & rf.notna()
    audit = {
        "panel_rows": len(out),
        "rows_with_both_ret_fwd_and_rf": int(both.sum()),
        "rows_with_excess_ret_na": int((~both).sum()),
    }
    return out, audit


def print_step5_summary(panel: pd.DataFrame, audit: dict[str, int]) -> None:
    print("\n=== Step 5: excess returns (ret_fwd - rf) ===\n")
    print(pd.Series(audit).to_string())
    slim = prediction_panel_essential(panel)
    print(f"\nPanel shape: {panel.shape[0]} × {panel.shape[1]} (full) | essential preview: {slim.shape[1]} columns")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 13)
    one_perm = panel["PERMNO"].iloc[0] if len(panel) else None
    demo = slim.loc[slim["PERMNO"] == one_perm].head(6) if one_perm is not None else slim.head(6)
    print("\nPreview (essential columns):\n")
    print(demo.to_string(index=False))
    print(f"\n(PERMNO == {one_perm}, up to 6 rows; N = {len(panel)}.)\n")


def print_question6_documentation(
    panel: pd.DataFrame, *, audits: dict[str, dict[str, object]]
) -> None:
    """Part 3 item 6: concise policy tied to pipeline audits (no look-ahead)."""
    n = len(panel)
    n_miss_ret = int(panel["ret_fwd"].isna().sum()) if "ret_fwd" in panel.columns else 0
    n_miss_rf = int(panel["rf"].isna().sum()) if "rf" in panel.columns else 0
    n_miss_exc = int(panel["excess_ret"].isna().sum()) if "excess_ret" in panel.columns else 0
    sig_here = [c for c in PART3_DEFAULT_PREDICTORS if c in panel.columns]
    na_per_sig = {c: int(panel[c].isna().sum()) for c in sig_here}

    s3 = audits.get("step3", {})
    print(
        """
=== Question 6 — Missing data, extremes, timing (summary) ===

Timing: signals at formation month ``t`` (``fdate``, ``cal_ym_signal``); outcomes and ``rf`` /
factors attach to ``cal_ym_fwd`` (= ``t+1``). No look-ahead beyond ``t`` on the signal side.

Missing returns: Step 3 inner-merge on (PERMNO, ``t+1``); rows without forward ``msf`` are dropped
(signals_dropped_no_fwd_msf = {dropped}). If ``RET`` exists but is NaN, ``ret_fwd`` stays NaN
(panel ret_fwd NA count ≈ {nret}).

Missing factors: Step 4 left-merge; ``rf`` can be NaN without a factor row (rf NA ≈ {nrf}).

Missing predictors: not listwise-dropped here; NA counts (demo columns) = {nas}. Handle in estimation
with rules that use only information ≤ ``t``.

Extremes: no winsorisation in Part~3 pipeline; raw ``ret_fwd`` / signals as in SAS.

``excess_ret``: requires both ``ret_fwd`` and ``rf``; NA count ≈ {nexc} of {n} rows.
""".format(
            dropped=s3.get("signals_dropped_no_fwd_msf", "n/a"),
            nret=n_miss_ret,
            nrf=n_miss_rf,
            nas=na_per_sig,
            nexc=n_miss_exc,
            n=n,
        ).strip()
    )
    print()


def main() -> None:
    tables = step1_tables_with_cal_ym()
    print_step1_summary(tables)

    merged, audit = step2_merge_signals_to_msf(tables)
    print_step2_summary(merged, audit)

    panel_fwd, audit_fwd = step3_construct_forward_returns(tables)
    print_step3_summary(panel_fwd, audit_fwd)

    panel_factor, audit_4 = step4_merge_factor_returns_by_month(panel_fwd, tables)
    print_step4_summary(panel_factor, audit_4)

    panel_final, audit_5 = step5_construct_excess_returns(panel_factor)
    print_step5_summary(panel_final, audit_5)

    qa6_audits = {
        "step3": audit_fwd,
        "step4": audit_4,
        "step5": audit_5,
    }
    print_question6_documentation(panel_final, audits=qa6_audits)

    panel_out = prediction_panel_essential(panel_final)
    print("=== Final panel — essential columns only (deliverable subset) ===\n")
    print(f"shape: {panel_out.shape[0]} × {panel_out.shape[1]}")
    print(f"columns: {list(panel_out.columns)}\n")
    print(panel_out.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
