"""Part 1 — Q1–Q5 (audit tables + merge-key guidance).

Run:
    python part1_script.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

PATHS = {
    "signals_raw_plus": ROOT / "signals_raw_plus.sas7bdat",
    "msf": ROOT / "msf.sas7bdat",
    "factors_monthly": ROOT / "factors_monthly.sas7bdat",
}

# Calendar date column used to describe coverage for each file.
DATE_FIELD = {
    "signals_raw_plus": "fdate",
    "msf": "DATE",
    "factors_monthly": "date",
}

# Q3 — minimal columns needed to align stocks over time and run excess-return / FF–Carhart style tests.
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "signals_raw_plus": ["PERMNO", "fdate"],
    "msf": ["PERMNO", "DATE", "RET"],
    "factors_monthly": ["date", "mktrf", "smb", "hml", "umd", "rf"],
}


def load_sas(path: Path) -> pd.DataFrame:
    return pd.read_sas(path, format="sas7bdat", encoding="utf-8")


def preview_column_order(dataset_name: str, df: pd.DataFrame, k: int) -> list[str]:
    """Put PERMNO and the canonical date (`DATE`, `fdate`, or `date`) first, then pad with remaining cols."""
    dcol = DATE_FIELD[dataset_name]
    ordered: list[str] = []
    for c in ("PERMNO", dcol):
        if c in df.columns:
            ordered.append(c)
    for c in df.columns:
        if c not in ordered:
            ordered.append(c)
        if len(ordered) >= k:
            break
    return ordered[: k]


def print_table_previews(*, n: int = 5, n_preview_cols: int = 5) -> None:
    """List every column name; sample rows always include the table's date/DATE column (see DATE_FIELD)."""

    k = max(1, min(n_preview_cols, 5))
    print("\n=== Preview: all column names + sample (includes DATE / fdate / date) ===\n")

    for name, path in PATHS.items():
        if not path.exists():
            print(f"{name}: (file missing) {path}\n")
            continue

        df = load_sas(path)
        n_cols = df.shape[1]

        print(f"--- {name} | shape {df.shape[0]} rows × {df.shape[1]} columns ---")
        print(f"Column names ({n_cols}):")
        name_block = ", ".join(map(str, df.columns))
        for line in textwrap.wrap(name_block, width=110, break_long_words=False):
            print(f"  {line}")

        pr_cols = preview_column_order(name, df, min(k, n_cols))
        subset = df[pr_cols].head(n)
        dlabel = DATE_FIELD[name]
        print(
            f"Sample: includes `{dlabel}`, {len(pr_cols)} columns, {min(n, len(df))} rows — {list(subset.columns)}"
        )
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", 24)
        print(subset.to_string(index=True))
        print()


def q3_required_fields_rows(cached: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    """Missing required columns ⇒ cannot claim a full prediction + risk-adjustment pipeline without fixing data."""
    rows: list[dict[str, object]] = []
    for name in PATHS:
        df = cached.get(name)
        required = REQUIRED_COLUMNS[name]

        if df is None:
            rows.append(
                {
                    "dataset": name,
                    "required_present": False,
                    "missing_required": "(file missing)",
                    "n_numeric_signal_cols": "—",
                }
            )
            continue

        missing = [c for c in required if c not in df.columns]
        base_ok = len(missing) == 0

        n_extra = "—"
        if name == "signals_raw_plus" and base_ok:
            key_cols = {"PERMNO", "fdate"}
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            n_extra = sum(1 for c in numeric_cols if c not in key_cols)

        rows.append(
            {
                "dataset": name,
                "required_present": base_ok,
                "missing_required": ", ".join(missing) if missing else "",
                "n_numeric_signal_cols": n_extra,
            }
        )

    return pd.DataFrame(rows)


def _calendar_month_str(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, errors="coerce").dt.to_period("M").astype(str)


def q4_duplicate_check_rows(cached: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    """Stock tables: duplicates on (PERMNO, calendar month). Factors: duplicates on month only."""

    rows: list[dict[str, object]] = []
    for name in PATHS:
        df = cached.get(name)
        dcol = DATE_FIELD[name]

        if df is None:
            rows.append(
                {
                    "dataset": name,
                    "duplicate_check": "",
                    "n_keys_with_duplicates": "",
                    "n_excess_rows_from_dup_keys": "",
                }
            )
            continue

        if name == "factors_monthly":
            mo = _calendar_month_str(df[dcol]).dropna()
            counts = mo.groupby(mo, observed=True).size()
            dup = counts[counts > 1]
            nz = int(len(dup))
            exc = int((dup - 1).sum()) if nz else 0
            rows.append(
                {
                    "dataset": name,
                    "duplicate_check": "calendar month from `date` (no PERMNO)",
                    "n_keys_with_duplicates": nz,
                    "n_excess_rows_from_dup_keys": exc,
                }
            )
            continue

        if "PERMNO" not in df.columns or dcol not in df.columns:
            rows.append(
                {
                    "dataset": name,
                    "duplicate_check": "PERMNO + calendar month",
                    "n_keys_with_duplicates": "n/a",
                    "n_excess_rows_from_dup_keys": "n/a",
                }
            )
            continue

        sub = df[["PERMNO", dcol]].dropna()
        sub = sub.assign(_ym=_calendar_month_str(sub[dcol]))
        grp = sub.groupby(["PERMNO", "_ym"], observed=True).size()
        dup = grp[grp > 1]
        nz = int(len(dup))
        exc = int((dup - 1).sum()) if nz else 0
        rows.append(
            {
                "dataset": name,
                "duplicate_check": "PERMNO + calendar month (" + dcol + ")",
                "n_keys_with_duplicates": nz,
                "n_excess_rows_from_dup_keys": exc,
            }
        )

    return pd.DataFrame(rows)


def print_calendar_month_stock_panel_audit(
    *,
    cached: dict[str, pd.DataFrame | None],
    ym: str = "1995-01",
    sample_n: int = 14,
    datasets: tuple[str, ...] = ("signals_raw_plus", "msf"),
) -> None:
    """
    Inspect one calendar-month bucket (default ``1995-01``) in PERMNO-level files.

    Shows row counts, breadth, and whether all rows anchor to one calendar day in the month
    (common for CRSP monthly ``DATE`` / signal ``fdate``).
    """

    print(f"\n=== Part 1 — snapshot: `{ym}` ({', '.join(datasets)}) ===\n")

    for name in datasets:
        df = cached.get(name)
        if df is None:
            print(f"--- {name}: (file missing) ---\n")
            continue

        if name not in DATE_FIELD:
            print(f"--- {name}: (no DATE_FIELD entry) ---\n")
            continue

        dcol = DATE_FIELD[name]
        if dcol not in df.columns or "PERMNO" not in df.columns:
            print(f"--- {name}: missing `{dcol}` or PERMNO ---\n")
            continue

        ts = pd.to_datetime(df[dcol], errors="coerce")
        cal_m = ts.dt.to_period("M").astype(str)
        sub = df.loc[cal_m == ym].copy()
        dates_in_sub = pd.to_datetime(sub[dcol], errors="coerce")

        print(f"--- {name} | `{dcol}` ---")
        print(f"  Rows in `{ym}`: {len(sub):,}")
        print(f"  Distinct PERMNO (in `{ym}`): {sub['PERMNO'].nunique(dropna=False):,}")
        if dates_in_sub.notna().any():
            dt_min = dates_in_sub.min()
            dt_max = dates_in_sub.max()
            n_distinct_days = int(dates_in_sub.dt.normalize().nunique())
            print(f"  `{dcol}` min / max within `{ym}`: {dt_min}  /  {dt_max}")
            print(f"  Distinct calendar dates in `{ym}`: {n_distinct_days}")
            vc = (
                dates_in_sub.dt.strftime("%Y-%m-%d")
                .dropna()
                .value_counts()
                .sort_index()
            )
            print("  Rows by calendar date:")
            for d, ct in vc.items():
                print(f"    {d}: {int(ct)}")
        else:
            print("  No non-missing dates in subset.")

        pr_cols = preview_column_order(name, sub, min(6, max(1, sub.shape[1]))) if sub.shape[1] else []
        if len(sub) and pr_cols:
            print(f"\n  Sample ({min(sample_n, len(sub))} rows, columns={pr_cols}):")
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", None)
            pd.set_option("display.max_colwidth", 26)
            print(sub[pr_cols].head(sample_n).to_string(index=False))
        print()


def print_q5_merge_key_guidance() -> None:
    """Part 1 Q5 — which variables are permanent identifiers vs poor merge keys."""

    text = """
=== Part 1, Q5: permanent identifiers vs merge keys ===

Use (primary chain linking stock panels across months):
  - PERMNO — CRSP permanent security identifier; aligns signals_raw_plus with msf.
  - Calendar month — derive YYYY-MM from fdate / DATE (same as cal_ym in Part 3);
    together with PERMNO, each row means "this stock in this calendar month".

  factors_monthly (no PERMNO):
  - Join on calendar month only (from date / year + month), after you decide which
    month's factors match the RETURN month used in forecasting (typically no look-ahead).

Avoid as principal merge keys:
  - Tickers / company names — not provided here as CRSP-grade keys and can change; prefer PERMNO.
  - CUSIP-only chains — auxiliary only; weaker than PERMNO for longitudinal CRSP-style merges.
  - Prices, spreads, VOL, RET, or other flows — not identifiers; wrong tool for aligning panels.
  - Merging PERMNO into factors as if rows were stock-month — blows up dimensions; attach
    monthly factors strictly by time dimension.

Detailed timing for signal versus return follows Backtest_Manual_v2.pdf; keys remain PERMNO + month IDs.
"""
    print(text.strip())


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    rows_q1: list[dict[str, object]] = []
    rows_q2: list[dict[str, object]] = []
    cached: dict[str, pd.DataFrame | None] = {}

    for name, path in PATHS.items():
        dcol = DATE_FIELD[name]

        if not path.exists():
            cached[name] = None
            rows_q1.append(
                {"dataset": name, "n_observations": None, "n_distinct_PERMNO": None}
            )
            rows_q2.append(
                {
                    "dataset": name,
                    "date_column": dcol,
                    "date_min": None,
                    "date_max": None,
                }
            )
            continue

        df = load_sas(path)
        cached[name] = df

        n_obs = len(df)
        if "PERMNO" in df.columns:
            n_perm = int(df["PERMNO"].nunique(dropna=False))
        else:
            n_perm = "—"
        rows_q1.append(
            {
                "dataset": name,
                "n_observations": n_obs,
                "n_distinct_PERMNO": n_perm,
            }
        )

        if dcol not in df.columns:
            ts = pd.Series(dtype="datetime64[ns]")
        else:
            ts = pd.to_datetime(df[dcol], errors="coerce")
        rows_q2.append(
            {
                "dataset": name,
                "date_column": dcol,
                "date_min": ts.min(),
                "date_max": ts.max(),
            }
        )

    print("=== Part 1, Q1: observations & distinct PERMNOs ===\n")
    print(pd.DataFrame(rows_q1).to_string(index=False))
    print("\n=== Part 1, Q2: date ranges ===\n")
    print(pd.DataFrame(rows_q2).to_string(index=False))

    print("\n=== Part 1, Q3: fields for return-prediction / risk adjustment ===")
    print("(Required checklist; third column counts numeric signals excluding PERMNO/fdate on the signal file only)")
    print()
    df_q3 = q3_required_fields_rows(cached)
    print(df_q3.to_string(index=False))

    print("\n=== Part 1, Q4: duplicate PERMNO–month (stock panels) / duplicate month (factors) ===")
    print(
        "(n_keys_with_duplicates: count of (PERMNO, month) keys with >1 row; "
        "n_excess_rows_from_dup_keys: rows beyond one per such key.)"
    )
    print()
    df_q4 = q4_duplicate_check_rows(cached)
    print(df_q4.to_string(index=False))

    print_q5_merge_key_guidance()

    for ym_audit in ("1995-01", "1995-02"):
        print_calendar_month_stock_panel_audit(cached=cached, ym=ym_audit)

    print_table_previews(n=5)


if __name__ == "__main__":
    main()
