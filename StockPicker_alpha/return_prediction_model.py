"""
Part 4 — Single-signal portfolio backtests (incremental).

Q1: Each formation month (``cal_ym_signal`` ≈ prior signal month ``t``), sort stocks into
quintiles (1 = lowest score, 5 = highest) using Part~2 scores (``ivol`` signed so higher = higher ex-ante return).

Q2: For each quintile each month, **next-month equal-weighted return** = cross-sectional mean of
``ret_fwd`` (Part~3: realised in ``cal_ym_fwd`` = ``t+1``) among names in that quintile; rows with
missing ``ret_fwd`` are dropped from the mean for that bucket.

Q3: **Long--short** each formation month: **long** quintile 5 (highest ``*_score``), **short** quintile 1
(lowest); portfolio return = EW return of Q5 minus EW return of Q1 (same ``ret_fwd`` horizon as Q2).

Q4: Summary stats on the monthly LS series (aligned ``cal_ym_fwd`` factors): mean monthly LS, annualised
vol, Sharpe (√12 mean/std on spread), CAPM \& Carhart-4 **monthly** alphas + homoskedastic SE, max drawdown
on ∏(1+r), mean EW leg counts.

Part 5: **Composite signal** — cross-sectional winsorize (1\%/99\%) each raw Part~2 column by ``cal_ym_signal``,
then monthly z-score, align signs (``flip`` from Part~2), equal-weight mean → ``composite_score``; repeat
quintile / LS / performance table and compare to single-signal Q4 rows.

**Reuse Part~3 primitives** wherever possible --- same merges and timing as
``step1_tables_with_cal_ym``, ``step3_construct_forward_returns``, and (for CAPM /
FF-Carhart) ``step4_merge_factor_returns_by_month`` + ``step5_construct_excess_returns``.
See ``load_prediction_panel_from_part3``.

Run:
    python part4_script.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part2_script import SELECTED_SIGNALS  # noqa: E402
from part3_script import (
    step1_tables_with_cal_ym,
    step3_construct_forward_returns,
    step4_merge_factor_returns_by_month,
    step5_construct_excess_returns,
)  # noqa: E402


N_QUINTILES = 5


def load_prediction_panel_from_part3(
    *,
    attach_factors_and_excess: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Panel built exactly like Part~3: Step~1 calendars, Step~3 ``ret_fwd`` on ``cal_ym_fwd``.
    Optionally chain Step~4 (factors by ``cal_ym_fwd``) and Step~5 (``excess_ret``).

    Returns ``(panel, tables)`` so callers can reuse the same ``tables`` dict for audits if needed.

    Quintile sorts use formation month ``cal_ym_signal``; returns and factor rows stay on outcome month ``cal_ym_fwd``.
    """
    tables = step1_tables_with_cal_ym()
    panel, _ = step3_construct_forward_returns(tables)
    if attach_factors_and_excess:
        panel, _ = step4_merge_factor_returns_by_month(panel, tables)
        panel, _ = step5_construct_excess_returns(panel)
    return panel, tables


def load_formation_panel() -> pd.DataFrame:
    """Same as ``load_prediction_panel_from_part3(..., attach_factors_and_excess=False)``."""
    panel, _ = load_prediction_panel_from_part3(attach_factors_and_excess=False)
    return panel


def cal_ym_signal_to_fwd_series(cal_ym_signal: pd.Series) -> pd.Series:
    """Map formation month ``YYYY-MM`` → outcome month holding ``ret_fwd`` (``t`` → ``t+1``)."""
    ts = pd.to_datetime(cal_ym_signal.astype(str) + "-01", errors="coerce").dt.to_period("M")
    return (ts + 1).astype(str)


def _ols_intercept_stderr(y: np.ndarray, X_rhs: np.ndarray) -> tuple[float, float, int]:
    """
    OLS: ``y ~ 1 + X_rhs``. Homoskedastic SE for intercept ; returns ``(alpha_monthly, se_alpha, n)``.
    ``X_rhs`` excludes constant (handled here).
    """
    yv = np.asarray(y, dtype=float)
    xv = np.asarray(X_rhs, dtype=float)
    if xv.ndim == 1:
        xv = xv.reshape(-1, 1)
    nobs, _ = xv.shape
    X = np.column_stack([np.ones(nobs), xv])
    k = X.shape[1]
    beta, _, rank, _ = np.linalg.lstsq(X, yv, rcond=None)
    if rank < k or nobs <= k:
        return float(np.nan), float(np.nan), int(nobs)
    resid = yv - X @ beta
    rss = float(resid.T @ resid)
    df_resid = max(nobs - k, 1)
    s2 = rss / df_resid
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
        se0 = float(np.sqrt(max(cov[0, 0], 0.0)))
    except np.linalg.LinAlgError:
        se0 = float("nan")
    return float(beta[0]), se0, int(nobs)


def max_drawdown_from_simple_returns(r: pd.Series) -> float:
    """Peak-to-trough drawdown on wealth ``∏(1+r)`` (monthly ``r`` as decimal)."""
    s = pd.to_numeric(r, errors="coerce").dropna()
    if len(s) < 2:
        return float("nan")
    w = (1.0 + s).cumprod()
    peak = w.cummax()
    dd = w / peak - 1.0
    return float(dd.min())


def factor_slice_by_fwd_month(panel_with_factors: pd.DataFrame) -> pd.DataFrame:
    """One row per ``cal_ym_fwd`` with FF/Carhart factors."""
    need = ["cal_ym_fwd", "mktrf", "smb", "hml", "umd"]
    for c in need:
        if c not in panel_with_factors.columns:
            raise KeyError(f"Panel missing `{c}` for Q4 (load with attach_factors_and_excess=True)")
    return (
        panel_with_factors[need]
        .drop_duplicates(subset=["cal_ym_fwd"])
        .sort_values("cal_ym_fwd")
        .reset_index(drop=True)
    )


def compute_ls_performance_metrics(
    ls: pd.DataFrame,
    factor_panel: pd.DataFrame,
) -> dict[str, float | int]:
    """
    All headline stats on the **intersection** of months with non-missing LS and complete factors
    (``cal_ym_fwd`` row), so means / vol / Sharpe / drawdown / alphas share one sample.

    Sharpe: ``√12 × mean(LS) / std(LS)`` (self-financing spread; no ``rf`` subtracted on ``LS``).
    Alphas: monthly OLS intercept; ``mktrf`` is already excess market (Part~3 factor table).
    """
    empty: dict[str, float | int] = {
        "n_months": 0,
        "mean_monthly_ls": np.nan,
        "annualized_vol_ls": np.nan,
        "sharpe_annualized": np.nan,
        "capm_alpha_monthly": np.nan,
        "capm_alpha_se": np.nan,
        "ff4_alpha_monthly": np.nan,
        "ff4_alpha_se": np.nan,
        "max_drawdown": np.nan,
        "avg_n_long": np.nan,
        "avg_n_short": np.nan,
    }
    if ls.empty:
        return empty

    ff = factor_slice_by_fwd_month(factor_panel)
    ls2 = ls.copy()
    ls2["cal_ym_fwd"] = cal_ym_signal_to_fwd_series(ls2["cal_ym_signal"])
    j = ls2.merge(ff, on="cal_ym_fwd", how="inner")
    j = j.dropna(subset=["ls_ret_fwd_ew", "mktrf", "smb", "hml", "umd"])
    if len(j) < 6:
        out = empty.copy()
        out["n_months"] = int(len(j))
        return out

    y = j["ls_ret_fwd_ew"].to_numpy(dtype=float)
    mkt = j["mktrf"].to_numpy(dtype=float)
    smb = j["smb"].to_numpy(dtype=float)
    hml = j["hml"].to_numpy(dtype=float)
    umd = j["umd"].to_numpy(dtype=float)

    mean_m = float(np.mean(y))
    std_m = float(np.std(y, ddof=1))
    ann_vol = std_m * np.sqrt(12.0)
    sharpe = (np.sqrt(12.0) * mean_m / std_m) if std_m > 1e-16 else float("nan")

    a_capm, se_capm, _ = _ols_intercept_stderr(y, mkt)
    X4 = np.column_stack([mkt, smb, hml, umd])
    a4, se4, _ = _ols_intercept_stderr(y, X4)

    return {
        "n_months": int(len(j)),
        "mean_monthly_ls": mean_m,
        "annualized_vol_ls": float(ann_vol),
        "sharpe_annualized": float(sharpe),
        "capm_alpha_monthly": a_capm,
        "capm_alpha_se": se_capm,
        "ff4_alpha_monthly": a4,
        "ff4_alpha_se": se4,
        "max_drawdown": max_drawdown_from_simple_returns(j["ls_ret_fwd_ew"]),
        "avg_n_long": float(j["n_long"].mean()),
        "avg_n_short": float(j["n_short"].mean()),
    }


def attach_part2_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """``{col}_score`` matches Part~2 economic sign (``ivol`` negated)."""
    out = panel.copy()
    for spec in SELECTED_SIGNALS:
        col = str(spec["column"])
        if col not in out.columns:
            raise KeyError(f"Panel missing signal column `{col}`")
        raw = pd.to_numeric(out[col], errors="coerce")
        flip = bool(spec["flip_sign_for_higher_means_expected_return"])
        out[f"{col}_score"] = -raw if flip else raw
    return out


def raw_selected_signal_columns() -> list[str]:
    return [str(s["column"]) for s in SELECTED_SIGNALS]


def winsorize_cross_section_by_month(
    df: pd.DataFrame,
    cols: list[str],
    *,
    group_col: str = "cal_ym_signal",
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """Clip each column within ``group_col`` to ``[lower_q, upper_q]`` cross-sectional quantiles."""

    def _clip(ser: pd.Series) -> pd.Series:
        x = pd.to_numeric(ser, errors="coerce")
        ok = x.notna()
        if int(ok.sum()) < 2:
            return x
        qs = x[ok].quantile([lower_q, upper_q])
        lo_v, hi_v = float(qs.iloc[0]), float(qs.iloc[1])
        if np.isnan(lo_v) or np.isnan(hi_v) or lo_v > hi_v:
            return x
        return x.clip(lo_v, hi_v)

    out = df.copy()
    for c in cols:
        if c not in out.columns:
            raise KeyError(f"winsorize: missing `{c}`")
        out[c] = out.groupby(group_col, observed=True)[c].transform(_clip)
    return out


def zscore_cross_section_by_month(
    df: pd.DataFrame,
    cols: list[str],
    *,
    group_col: str = "cal_ym_signal",
    min_n: int = 2,
    z_suffix: str = "_z_cs",
) -> pd.DataFrame:
    """Attach ``{{col}}{z_suffix}`` = cross-sectional z-score within ``group_col`` (sample ``std``, ``ddof=1``)."""

    def _z(ser: pd.Series) -> pd.Series:
        x = pd.to_numeric(ser, errors="coerce")
        ok = x.notna()
        if int(ok.sum()) < min_n:
            return pd.Series(np.nan, index=ser.index, dtype=float)
        mu = float(x[ok].mean())
        sd = float(x[ok].std(ddof=1))
        if sd < 1e-15:
            return pd.Series(np.nan, index=ser.index, dtype=float)
        return (x - mu) / sd

    out = df.copy()
    for c in cols:
        zn = f"{c}{z_suffix}"
        out[zn] = out.groupby(group_col, observed=True)[c].transform(_z)
    return out


def build_composite_score_panel(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Part~5 pipeline on **raw** Part~2 columns: winsor 1\%/99\% × month → z-score × month → sign align → mean.

    Requires all aligned z-components non-null for ``composite_score`` (``mean(..., skipna=False)``).
    """
    raw_cols = raw_selected_signal_columns()
    out = scored.copy()
    miss = [c for c in raw_cols if c not in out.columns]
    if miss:
        raise KeyError(f"composite: missing raw columns {miss}")

    work = out[["PERMNO", "cal_ym_signal", *raw_cols]].copy()
    for c in raw_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work = winsorize_cross_section_by_month(work, raw_cols, group_col="cal_ym_signal")
    work = zscore_cross_section_by_month(work, raw_cols, group_col="cal_ym_signal")

    zcols = [f"{c}_z_cs" for c in raw_cols]
    aligned_cols: list[str] = []
    for spec, zn in zip(SELECTED_SIGNALS, zcols):
        ac = f"__aligned_{spec['column']}"
        sign = -1.0 if bool(spec["flip_sign_for_higher_means_expected_return"]) else 1.0
        work[ac] = pd.to_numeric(work[zn], errors="coerce") * sign
        aligned_cols.append(ac)

    out["composite_score"] = work[aligned_cols].mean(axis=1, skipna=False)
    return out


def assign_quintiles(panel: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """
    Quintiles 1..5 within each ``cal_ym_signal`` from ascending ``score_col``:
    low score -> 1, high score -> 5 (via percentile ranks; ties use average rank).
    Rows with missing ``score_col`` are dropped (no sort that month).
    """
    need = ["PERMNO", "cal_ym_signal", score_col]
    df = panel[need].dropna(subset=[score_col]).copy()
    if df.empty:
        return df.assign(quintile=np.nan)

    pct = df.groupby("cal_ym_signal", observed=True)[score_col].rank(method="average", pct=True)
    q = np.clip(np.ceil(pct.to_numpy(dtype=float) * N_QUINTILES), 1, N_QUINTILES).astype(np.int8)
    df["quintile"] = q
    return df


def quintile_panel_for_all_signals(panel_scored: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per raw signal name, return long table (PERMNO, month, score used, quintile)."""
    out: dict[str, pd.DataFrame] = {}
    for spec in SELECTED_SIGNALS:
        col = str(spec["column"])
        score_col = f"{col}_score"
        qdf = assign_quintiles(panel_scored, score_col)
        qdf.insert(0, "signal", col)
        out[col] = qdf
    return out


def summarize_quintile_counts_per_month(qdf: pd.DataFrame, signal_name: str) -> pd.DataFrame:
    """Rows: month; columns: count in each quintile (+ n names that month)."""
    if qdf.empty:
        return pd.DataFrame()
    raw = (
        qdf.groupby(["cal_ym_signal", "quintile"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    raw = raw.reindex(columns=list(range(1, N_QUINTILES + 1)), fill_value=0)
    raw.columns = [f"n_q{k}" for k in range(1, N_QUINTILES + 1)]
    counts = raw.copy()
    counts["n_total"] = counts.sum(axis=1)
    counts.insert(0, "signal", signal_name)
    return counts.reset_index()


def merge_quintile_panel_with_ret_fwd(qdf: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Attach Part~3 ``ret_fwd`` (next calendar month) to each (``PERMNO``, ``cal_ym_signal``) quintile row."""
    keys = ["PERMNO", "cal_ym_signal"]
    for k in keys + ["ret_fwd"]:
        if k not in panel.columns:
            raise KeyError(f"Panel missing `{k}` for Q2 merge")
    base = qdf.merge(
        panel[keys + ["ret_fwd"]],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    return base


def monthly_equal_weighted_quintile_returns(merged: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (formation month, quintile): equal-weight portfolio return =
    ``mean(ret_fwd)`` within bucket. ``count`` ignores NaN ``ret_fwd`` (names without a usable
    forward return do not enter the EW mean).
    """
    if merged.empty:
        cols = ["cal_ym_signal", "quintile", "ret_fwd_ew", "n_stocks_in_mean"]
        return pd.DataFrame(columns=cols)
    grp = merged.groupby(["cal_ym_signal", "quintile"], observed=True)
    out = grp.agg(ret_fwd_ew=("ret_fwd", "mean"), n_stocks_in_mean=("ret_fwd", "count")).reset_index()
    return out


def pivot_ew_returns_wide(ew_long: pd.DataFrame) -> pd.DataFrame:
    """Wide sheet: ``r_q1_ew`` … ``r_q5_ew`` by formation month."""
    if ew_long.empty:
        return pd.DataFrame()
    w = ew_long.pivot(index="cal_ym_signal", columns="quintile", values="ret_fwd_ew")
    w = w.reindex(columns=list(range(1, N_QUINTILES + 1)))
    w.columns = [f"r_q{k}_ew" for k in range(1, N_QUINTILES + 1)]
    return w.sort_index().reset_index()


def monthly_long_short_ew_returns(ew_long: pd.DataFrame) -> pd.DataFrame:
    """
    Per ``cal_ym_signal``: **long** top quintile EW ``ret_fwd``, **short** bottom quintile EW ``ret_fwd``.
    ``ls_ret_fwd_ew = ret_fwd_ew(Q5) - ret_fwd_ew(Q1)`` (decimal/month; same horizon as Q2).

    Months missing either leg after Q2 aggregation are dropped.
    """
    cols = ["cal_ym_signal", "ls_ret_fwd_ew", "ret_q5_ew", "ret_q1_ew", "n_long", "n_short"]
    if ew_long.empty:
        return pd.DataFrame(columns=cols)
    q1 = ew_long.loc[ew_long["quintile"] == 1, ["cal_ym_signal", "ret_fwd_ew", "n_stocks_in_mean"]].rename(
        columns={"ret_fwd_ew": "ret_q1_ew", "n_stocks_in_mean": "n_short"}
    )
    q5 = ew_long.loc[ew_long["quintile"] == N_QUINTILES, ["cal_ym_signal", "ret_fwd_ew", "n_stocks_in_mean"]].rename(
        columns={"ret_fwd_ew": "ret_q5_ew", "n_stocks_in_mean": "n_long"}
    )
    out = q5.merge(q1, on="cal_ym_signal", how="inner")
    out["ls_ret_fwd_ew"] = out["ret_q5_ew"] - out["ret_q1_ew"]
    return out[cols].sort_values("cal_ym_signal").reset_index(drop=True)


Q4_ROW_ORDER = [
    "signal",
    "n_months",
    "mean_monthly_ls",
    "annualized_vol_ls",
    "sharpe_annualized",
    "capm_alpha_monthly",
    "capm_alpha_se",
    "ff4_alpha_monthly",
    "ff4_alpha_se",
    "max_drawdown",
    "avg_n_long",
    "avg_n_short",
]


def gather_q4_metrics_rows(
    q_panels: dict[str, pd.DataFrame],
    scored_panel: pd.DataFrame,
) -> list[dict[str, object]]:
    """One Part~4 Q4 metric dict per quintile-sort key in ``q_panels``."""
    rows: list[dict[str, object]] = []
    for name, qdf in q_panels.items():
        merged = merge_quintile_panel_with_ret_fwd(qdf, scored_panel)
        ew_long = monthly_equal_weighted_quintile_returns(merged)
        ls = monthly_long_short_ew_returns(ew_long)
        met = compute_ls_performance_metrics(ls, scored_panel)
        met["signal"] = name
        rows.append(met)
    return rows


def format_q4_metrics_dataframe(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Display-ready table with ordered columns."""
    if not rows:
        return pd.DataFrame(columns=Q4_ROW_ORDER)
    tab = pd.DataFrame(rows)[Q4_ROW_ORDER]
    disp = tab.copy()
    disp[Q4_ROW_ORDER[2:]] = disp[Q4_ROW_ORDER[2:]].round(6)
    disp["n_months"] = pd.to_numeric(disp["n_months"], errors="coerce").fillna(0).astype(int)
    return disp


def print_question1_report(q_panels: dict[str, pd.DataFrame]) -> None:
    print("\nPart 4 — Q1. Quintile assignments\n")

    for name, df in q_panels.items():
        score_col = f"{name}_score"
        print(name)
        if df.empty:
            print("(empty)\n")
            continue

        summ = summarize_quintile_counts_per_month(df, name)
        mq = pd.DataFrame([{f"q{k}": summ[f"n_q{k}"].mean() for k in range(1, N_QUINTILES + 1)}])
        print(f"n stock-months (non-missing score): {len(df)}")
        print("Mean stocks per quintile (across months):")
        print(mq.to_string(index=False))
        print("Counts by month (first 6 formation months):")
        print(summ.head(6).to_string(index=False))

        first_months = sorted(df["cal_ym_signal"].unique())[:3]
        demo = (
            df.loc[df["cal_ym_signal"].isin(first_months), ["cal_ym_signal", "PERMNO", score_col, "quintile"]]
            .sort_values(["cal_ym_signal", "quintile", "PERMNO"])
            .head(12)
        )
        print("Sample rows (first 3 months, up to 12 rows):")
        print(demo.to_string(index=False))
        print()


def print_question2_report(q_panels: dict[str, pd.DataFrame], scored_panel: pd.DataFrame) -> None:
    print("\nPart 4 — Q2. Equal-weight quintile returns: ret_fwd, decimal/month\n")

    for name, qdf in q_panels.items():
        print(name)
        merged = merge_quintile_panel_with_ret_fwd(qdf, scored_panel)
        ew_long = monthly_equal_weighted_quintile_returns(merged)
        if ew_long.empty:
            print("(no rows)\n")
            continue

        wide = pivot_ew_returns_wide(ew_long)
        print("By month × quintile (first 6 months):")
        print(wide.head(6).to_string(index=False, float_format=lambda x: f"{x: .6f}"))

        by_q = ew_long.groupby("quintile", observed=True)["ret_fwd_ew"].mean()
        print("Time-average EW return by quintile:")
        print(pd.DataFrame([by_q.to_dict()]).to_string(index=False))

        first_m = sorted(ew_long["cal_ym_signal"].unique())[0]
        demo = ew_long.loc[ew_long["cal_ym_signal"] == first_m].sort_values("quintile")
        print(f"Single month ({first_m}), long format:")
        print(demo.to_string(index=False, float_format=lambda x: f"{x: .6f}" if isinstance(x, float) else str(x)))
        print()


def print_question3_report(q_panels: dict[str, pd.DataFrame], scored_panel: pd.DataFrame) -> None:
    print("\nPart 4 — Q3. Long-short: Q5 minus Q1, decimal/month\n")

    for name, qdf in q_panels.items():
        print(name)
        merged = merge_quintile_panel_with_ret_fwd(qdf, scored_panel)
        ew_long = monthly_equal_weighted_quintile_returns(merged)
        ls = monthly_long_short_ew_returns(ew_long)
        if ls.empty:
            print("(no rows)\n")
            continue

        print("Monthly LS (first 8 months):")
        print(
            ls.head(8).to_string(
                index=False,
                float_format=lambda x: f"{x: .6f}" if isinstance(x, float) else str(x),
            )
        )
        summary = pd.DataFrame(
            {
                "n_months": [len(ls)],
                "mean_monthly_LS": [ls["ls_ret_fwd_ew"].mean()],
                "mean_n_long": [ls["n_long"].mean()],
                "mean_n_short": [ls["n_short"].mean()],
            }
        )
        print("Summary:")
        print(summary.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
        print()


def print_question4_report(q_panels: dict[str, pd.DataFrame], scored_panel: pd.DataFrame) -> None:
    print("\nPart 4 — Q4. Long-short performance\n")
    print(
        "Note: Sharpe = √12·μ/σ on LS spread; CAPM α: LS ~ 1 + mktrf; "
        "Carhart-4 α: + smb + hml + umd; homoskedastic SE; monthly decimals.\n"
    )

    rows = gather_q4_metrics_rows(q_panels, scored_panel)

    if not rows:
        print("(no signals)\n")
        return

    print(format_q4_metrics_dataframe(rows).to_string(index=False))
    print()


def print_part5_report(
    scored_panel: pd.DataFrame,
    q_panels_singles: dict[str, pd.DataFrame],
) -> None:
    print("\nPart 5 — Composite signal\n")
    print(
        "Note: raw SELECTED_SIGNALS columns → winsor 1%/99% (cross-section/month) "
        "→ z-score → sign flip (Part 2) → equal-weight mean; LS metrics as Part 4 Q4.\n"
    )

    scored_c = build_composite_score_panel(scored_panel)

    qdf_c = assign_quintiles(scored_c, "composite_score").copy()
    qdf_c.insert(0, "signal", "composite")

    qp_c = {"composite": qdf_c}

    merged_c = merge_quintile_panel_with_ret_fwd(qdf_c, scored_panel)
    ew_c = monthly_equal_weighted_quintile_returns(merged_c)

    print("Composite — EW quintile returns (first 6 months, decimal/month):")
    if ew_c.empty:
        print("(no rows)\n")
    else:
        print(pivot_ew_returns_wide(ew_c).head(6).to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    ls_c = monthly_long_short_ew_returns(ew_c)
    print("\nComposite — monthly LS (first 8 months):")
    if ls_c.empty:
        print("(no rows)\n")
    else:
        print(
            ls_c.head(8).to_string(
                index=False,
                float_format=lambda x: f"{x: .6f}" if isinstance(x, float) else str(x),
            )
        )
        print()

    singles_rows = gather_q4_metrics_rows(q_panels_singles, scored_panel)
    comp_rows = gather_q4_metrics_rows(qp_c, scored_panel)
    all_rows = singles_rows + comp_rows

    print("Single signals vs composite (same metrics as Part 4 Q4):")
    print(format_q4_metrics_dataframe(all_rows).to_string(index=False))
    print()


def main() -> None:
    panel, _ = load_prediction_panel_from_part3(attach_factors_and_excess=True)
    scored = attach_part2_scores(panel)
    q_panels = quintile_panel_for_all_signals(scored)
    print_question1_report(q_panels)
    print_question2_report(q_panels, scored)
    print_question3_report(q_panels, scored)
    print_question4_report(q_panels, scored)
    print_part5_report(scored, q_panels)


if __name__ == "__main__":
    main()
