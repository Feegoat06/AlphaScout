from collections import Counter
from datetime import datetime, timedelta
import time

import numpy as np
import pandas as pd
import yfinance as yf

# ------------------------------
# Basic config
# ------------------------------

# Custom ticker list (deduped from user input)
ticker_list = [
    "TSLA", "NVDA", "AAPL", "MU", "AVGO", "ASML", "AMD", "MSFT", "ORCL", "AMZN",
    "TSM", "CSCO", "META", "GOOGL", "QCOM", "ADBE", "INTC", "BABA",
    "RMBS", "STX", "SNDK", "WDC",
    "UMAC", "KTOS", "DPRO", "NOC", "JOBY", "ACHR", "RCAT",
    "LMT", "ASTS", "RKLB", "SPIR", "RDW", "SPCE",
    "IONQ", "QUBT", "RGTI", "IBM", "QBTS", "QMCO",
    "CRCL", "COIN", "PYPL", "V", "MPU", "SOFI", "HOOD",
    "LEU", "NNE", "SMR", "GEV", "OKLO", "CCJ", "VST", "CEG",
    "MSTR", "MARA", "RIOT", "FIGR", "BKKT", "IREN", "DFDV", "EQ", "CIFR", "GLXY",
    "CLSK", "HUT", "BTDR", "BLSH", "DJT", "WULF",
    "ANET", "APLD", "NBIS", "VRT", "LRCX", "AMAT", "PLTR", "AI", "CRM", "BBAI",
    "TEM", "HIMS", "ABSI", "CRSP", "BEAM",
    "LLY", "PFE", "AMGN", "NVO", "VKTX",
    "USAR", "NB", "MP", "CRML", "CLF", "UUUU", "UURAF",
    "JPM", "BAC", "GS", "CAT", "HON", "DE", "NEM", "FCX", "LIN", "XOM", "CVX", "COP",
    "DHI", "LEN", "O", "RTX", "GD", "LHX", "PL", "ISRG", "TER",
    "UBER", "AUR", "EOG", "BWXT", "FSLR", "ENPH", "SEDG", "NEE", "SO", "BE", "FLNC",
    "MEDP", "ABVX", "ABVC", "GALT", "CTMX", "GOSS", "JNJ", "ABBV", "MRK",
    "VRTX", "REGN", "PPL", "AEP",
    "SEI", "SYNA", "POET", "OSCR", "FLY", "FSLY", "RR", "BMNR", "SOUN"
]

# ------------------------------
# Strategy params
# ------------------------------
lookback_days = 120
min_roe = 10.0  # ROE > 10%
min_return_5d = 0.03  # 5d return > 3%
min_rs_window = 20
min_volume_increase = 1.2
price_high_ratio = 0.98

results_level1 = []
results_level2 = []


def get_history_df(ticker, start_date, end_date):
    """Fetch daily OHLCV data from yfinance."""
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(1)
    df.columns = [str(col).lower().replace("adj close", "adj_close") for col in df.columns]
    df = df.rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "adj_close": "adj_close",
            "volume": "volume",
        }
    )
    df = df.reset_index()
    date_col = None
    for col in ["Date", "Datetime", "index"]:
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        date_col = df.columns[0]
    df["time_key"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    return df.sort_values("time_key").reset_index(drop=True)


def calc_slope(series):
    """Linear regression slope for trend direction."""
    values = series.dropna().values
    if len(values) < 2:
        return None
    x = np.arange(len(values))
    return float(np.polyfit(x, values, 1)[0])


def calc_vwap_series(df):
    """Approximate daily VWAP using typical price."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    return typical_price


def fetch_roe(ticker):
    """Fetch ROE from yfinance info; returns percent."""
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return None
    roe = info.get("returnOnEquity")
    if roe is None:
        return None
    return float(roe * 100) if roe < 1 else float(roe)


def get_relative_strength(stock_df, spy_df, window):
    """RS = stock window return - SPY window return."""
    merged = pd.merge(
        stock_df[["time_key", "close"]],
        spy_df[["time_key", "close"]],
        on="time_key",
        how="inner",
        suffixes=("_stock", "_spy"),
    ).dropna(subset=["close_stock", "close_spy"])
    if len(merged) < window + 1:
        return None
    stock_ret = merged["close_stock"].iloc[-1] / merged["close_stock"].iloc[-(window + 1)] - 1
    spy_ret = merged["close_spy"].iloc[-1] / merged["close_spy"].iloc[-(window + 1)] - 1
    return float(stock_ret - spy_ret)


def count_volume_expansion_days(df):
    """Count volume or dollar-volume expansion days in last 7 sessions."""
    recent_7 = df.tail(7).copy()
    if len(recent_7) < 7:
        return 0

    base_20 = df.tail(27).head(20)
    if base_20.empty:
        base_20 = df.tail(20)

    avg_volume = base_20["volume"].mean()
    avg_dollar_volume = (base_20["close"] * base_20["volume"]).mean()

    recent_7["is_up"] = recent_7["close"] > recent_7["close"].shift(1)
    recent_7["vol_expand"] = recent_7["volume"] >= avg_volume * min_volume_increase
    recent_7["dollar_expand"] = (
        recent_7["close"] * recent_7["volume"] >= avg_dollar_volume * min_volume_increase
    )

    valid_days = recent_7[recent_7["is_up"] & (recent_7["vol_expand"] | recent_7["dollar_expand"])]
    return int(valid_days.shape[0])


def calc_price_location(df, window=60):
    """Avoid extreme highs."""
    window_df = df.tail(window)
    if window_df.empty:
        return None
    highest_price = window_df["high"].max()
    current_price = window_df["close"].iloc[-1]
    return float(current_price / highest_price) if highest_price > 0 else None


def main():
    results_level1.clear()
    results_level2.clear()
    unique_tickers = list(dict.fromkeys(ticker_list))
    dup_counts = {t: c for t, c in Counter(ticker_list).items() if c > 1}
    if dup_counts:
        dup_list = ", ".join(sorted(dup_counts.keys()))
        print(f"Warning: duplicate tickers in ticker_list: {dup_list}")

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    spy_df = get_history_df("SPY", start_date, end_date)
    if spy_df is None:
        print("Failed to fetch SPY data.")
        return

    for ticker in unique_tickers:
        try:
            df = get_history_df(ticker, start_date, end_date)
            if df is None or len(df) < lookback_days:
                continue

            # Moving averages
            df["ma5"] = df["close"].rolling(5).mean()
            df["ma10"] = df["close"].rolling(10).mean()
            df["ma20"] = df["close"].rolling(20).mean()
            df["ma7"] = df["close"].rolling(7).mean()
            df["ma14"] = df["close"].rolling(14).mean()
            df["ma21"] = df["close"].rolling(21).mean()

            trend_ok = (
                df["ma7"].iloc[-1] > df["ma14"].iloc[-1] > df["ma21"].iloc[-1]
                and (calc_slope(df["ma7"].tail(5)) or 0) > 0
                and (calc_slope(df["ma14"].tail(5)) or 0) > 0
                and (calc_slope(df["ma21"].tail(5)) or 0) > 0
            )

            short_ma_ok = df["ma5"].iloc[-1] > df["ma10"].iloc[-1] > df["ma20"].iloc[-1]

            close_above_ma5 = df["close"].tail(6) > df["ma5"].tail(6)
            close_above_ma5_ok = int(close_above_ma5.sum()) >= 5

            last_5d_return = df["close"].iloc[-1] / df["close"].iloc[-6] - 1

            rs_20d = get_relative_strength(df, spy_df, min_rs_window)

            vol_expand_days = count_volume_expansion_days(df)
            vol_expand_ok = vol_expand_days >= 2

            vwap_series = calc_vwap_series(df)
            vwap_slope = calc_slope(vwap_series.tail(5))
            vwap_ok = vwap_slope is not None and vwap_slope > 0

            price_ratio = calc_price_location(df, window=60)
            price_ok = price_ratio is not None and price_ratio < price_high_ratio

            roe = fetch_roe(ticker)
            roe_ok = roe is not None and roe > min_roe

            base_ok = (
                trend_ok
                and short_ma_ok
                and close_above_ma5_ok
                and last_5d_return > min_return_5d
                and rs_20d is not None
                and rs_20d > 0
                and vol_expand_ok
                and vwap_ok
                and price_ok
            )
            if base_ok:
                score = (
                    rs_20d * 100
                    + last_5d_return * 100
                    + vol_expand_days * 5
                    + (vwap_slope or 0) * 10
                )
                row = {
                    "Ticker": ticker,
                    "ROE": round(roe, 2) if roe is not None else None,
                    "5D_Return": round(last_5d_return, 3),
                    "RS_20D": round(rs_20d, 3),
                    "Vol_Expand_Days": vol_expand_days,
                    "VWAP_Slope": round(vwap_slope, 6) if vwap_slope is not None else None,
                    "Price/High": round(price_ratio, 3) if price_ratio is not None else None,
                    "Score": round(score, 2),
                }
                results_level1.append(row)

                if roe_ok:
                    results_level2.append(row)

        except Exception as exc:
            print(f"Failed on {ticker}: {exc}")
        finally:
            time.sleep(1.5)

    if not results_level1:
        print("No candidates found.")
        return

    df_level1 = pd.DataFrame(results_level1).sort_values("Score", ascending=False)
    print("Level 1 (all conditions except ROE):")
    print(df_level1.to_string(index=False))

    if results_level2:
        df_level2 = pd.DataFrame(results_level2).sort_values("Score", ascending=False)
        print("\nLevel 2 (all conditions including ROE):")
        print(df_level2.to_string(index=False))
    else:
        print("\nLevel 2 (all conditions including ROE): none")


if __name__ == "__main__":
    main()
