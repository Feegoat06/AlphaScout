import time
import random
from collections import Counter

import akshare as ak
import numpy as np
import pandas as pd
import requests

print("CN Stock Screener - Fund Flow + Technical Filters")

# ------------------------------
# Configuration
# ------------------------------
LOOKBACK_DAYS = 120
MIN_RETURN_5D = 0.03
MIN_RETURN_7D_CUM = 0.25
MIN_VOLUME_INCREASE = 1.2
MIN_ROE_PCT = 10.0
MAX_PRICE_RATIO = 0.90
CLOSE_ABOVE_MA5_DAYS = 15
MA_SLOPE_DAYS = 5
MIN_APPEARANCES_7D = 3
LAST_TRADE_DAYS = 7

EASTMONEY_URL = "https://push2.eastmoney.com/api/qt/clist/get"
EASTMONEY_UT = "8dec03ba335b81bf4ebdf7b29ec27d15"
FIELDS_STAT_1 = (
    "f12,f14,f2,f3,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f204,f205,f124,f1,f13"
)
FS_HSA = "m:0+t:6+f:!2,m:0+t:13+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,m:1+t:23+f:!2"

CACHE_PATH = "fund_flow_top40_history.csv"
CACHE_XLSX_PATH = "fund_flow_top40_history.xlsx"

COLUMN_MAP = {
    "\u65e5\u671f": "date",
    "\u5f00\u76d8": "open",
    "\u6536\u76d8": "close",
    "\u6700\u9ad8": "high",
    "\u6700\u4f4e": "low",
    "\u6210\u4ea4\u91cf": "volume",
}


def get_last_trade_dates(n):
    try:
        df = ak.tool_trade_date_hist_sina()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df[df["trade_date"] <= pd.Timestamp.today().normalize()]
        dates = df["trade_date"].tail(n).dt.strftime("%Y-%m-%d").tolist()
        if dates:
            return dates
    except Exception:
        pass
    return pd.bdate_range(end=pd.Timestamp.today(), periods=n).strftime("%Y-%m-%d").tolist()


def fetch_fund_flow_top40():
    params = {
        "pn": 1,
        "pz": 40,
        "po": 1,
        "np": 1,
        "fltt": 2,
        "invt": 2,
        "fid": "f62",
        "fs": FS_HSA,
        "fields": FIELDS_STAT_1,
        "ut": EASTMONEY_UT,
    }
    resp = requests.get(EASTMONEY_URL, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    items = payload.get("data", {}).get("diff", []) or []
    results = []
    for idx, item in enumerate(items, start=1):
        code = item.get("f12")
        name = item.get("f14")
        if code:
            code = str(code).zfill(6)
            results.append({"code": code, "name": name, "rank": idx})
    return results


def load_cache():
    try:
        df = pd.read_csv(CACHE_PATH, dtype={"code": str})
        df["code"] = df["code"].astype(str).str.zfill(6)
        return df
    except Exception:
        return pd.DataFrame(columns=["trade_date", "code", "name", "rank"])


def save_cache(df):
    df.to_csv(CACHE_PATH, index=False, encoding="utf-8-sig")
    write_excel_with_text_code(df, CACHE_XLSX_PATH)


def update_cache_for_date(trade_date):
    cache_df = load_cache()
    existing = cache_df[cache_df["trade_date"] == trade_date]
    if len(existing) >= 20:
        return cache_df
    rows = fetch_fund_flow_top40()
    if not rows:
        print(f"Warning: fund flow list is empty for {trade_date}.")
        return cache_df
    new_df = pd.DataFrame(rows)
    new_df["trade_date"] = trade_date
    merged = pd.concat([cache_df, new_df], ignore_index=True)
    save_cache(merged)
    return merged


def get_candidates_from_cache(trade_dates):
    cache_df = load_cache()
    available_dates = sorted(set(cache_df["trade_date"]).intersection(trade_dates))
    missing = [d for d in trade_dates if d not in set(cache_df["trade_date"])]
    if missing:
        print(
            "Warning: missing cached fund flow dates: "
            + ", ".join(missing)
            + ". Run this script on each trading day to build history."
        )
    recent = cache_df[cache_df["trade_date"].isin(available_dates)]
    counts = Counter(recent["code"].tolist())
    available_days = len(available_dates)
    if available_days == 0:
        return {}
    if available_days < LAST_TRADE_DAYS:
        threshold = int(available_days * 0.5) + 1
        print(
            f"Only {available_days} cached day(s). Using >50% rule: "
            f"appearances >= {threshold}."
        )
    else:
        threshold = MIN_APPEARANCES_7D
    return {code: cnt for code, cnt in counts.items() if cnt >= threshold}


def write_excel_with_text_code(df, path):
    try:
        import openpyxl  # noqa: F401
        from openpyxl.styles import numbers
    except Exception:
        print(f"Warning: openpyxl not available, skipped Excel output: {path}")
        return
    df_out = df.copy()
    for col in ["code", "Code"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(str).str.zfill(6)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="data")
        ws = writer.book["data"]
        header = [cell.value for cell in ws[1]]
        for col_name in ["code", "Code"]:
            if col_name in header:
                col_idx = header.index(col_name) + 1
                for row in ws.iter_rows(min_row=2, min_col=col_idx, max_col=col_idx):
                    for cell in row:
                        cell.number_format = numbers.FORMAT_TEXT


def parse_roe_value(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        value = value.replace("%", "").replace(",", "").strip()
    try:
        val = float(value)
    except Exception:
        return None
    if val <= 1.0:
        val *= 100.0
    return val


def get_latest_roe_pct(code):
    df = None
    try:
        df = ak.stock_financial_analysis_indicator(symbol=code)
    except Exception:
        try:
            df = ak.stock_financial_abstract(symbol=code)
        except Exception:
            return None
    if df is None or df.empty:
        return None
    roe_cols = [
        col
        for col in df.columns
        if "ROE" in col
        or "roe" in col.lower()
        or "\u51c0\u8d44\u4ea7\u6536\u76ca\u7387" in col
    ]
    if not roe_cols:
        return None
    series = df[roe_cols[0]].dropna()
    if series.empty:
        return None
    return parse_roe_value(series.iloc[-1])


def ma_divergence_up(df):
    if len(df) < MA_SLOPE_DAYS + 1:
        return False
    m7 = df["MA7"]
    m14 = df["MA14"]
    m21 = df["MA21"]
    if not (m7.iloc[-1] > m14.iloc[-1] > m21.iloc[-1]):
        return False
    if not (
        m7.iloc[-1] > m7.iloc[-1 - MA_SLOPE_DAYS]
        and m14.iloc[-1] > m14.iloc[-1 - MA_SLOPE_DAYS]
        and m21.iloc[-1] > m21.iloc[-1 - MA_SLOPE_DAYS]
    ):
        return False
    gap_now_7_14 = m7.iloc[-1] - m14.iloc[-1]
    gap_now_14_21 = m14.iloc[-1] - m21.iloc[-1]
    gap_prev_7_14 = m7.iloc[-1 - MA_SLOPE_DAYS] - m14.iloc[-1 - MA_SLOPE_DAYS]
    gap_prev_14_21 = m14.iloc[-1 - MA_SLOPE_DAYS] - m21.iloc[-1 - MA_SLOPE_DAYS]
    return gap_now_7_14 > gap_prev_7_14 and gap_now_14_21 > gap_prev_14_21


def close_above_ma5(df):
    if len(df) < CLOSE_ABOVE_MA5_DAYS:
        return False
    recent_close = df["close"].tail(CLOSE_ABOVE_MA5_DAYS)
    recent_ma5 = df["MA5"].tail(CLOSE_ABOVE_MA5_DAYS)
    return (recent_close > recent_ma5).all()


def screen_stock(code, appearances_7d):
    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date="20180101",
        end_date="20991231",
        adjust="qfq",
    )
    if df is None or df.empty:
        return None

    df = df.rename(columns=COLUMN_MAP)
    if "date" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < LOOKBACK_DAYS:
        return None

    for col in ["close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["MA5"] = df["close"].rolling(5).mean()
    df["MA7"] = df["close"].rolling(7).mean()
    df["MA10"] = df["close"].rolling(10).mean()
    df["MA14"] = df["close"].rolling(14).mean()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA21"] = df["close"].rolling(21).mean()
    df["return_5d"] = df["close"].pct_change(5)
    df["return_7d"] = df["close"].pct_change(7)

    volume_now = df["volume"].tail(5).mean()
    volume_prev = df["volume"].tail(15).head(10).mean()
    volume_ratio = volume_now / volume_prev if volume_prev else 0

    highest_price = df["close"].max()
    current_price = df["close"].iloc[-1]
    price_ratio = current_price / highest_price if highest_price else 0

    roe_pct = get_latest_roe_pct(code)

    p1 = ma_divergence_up(df) and close_above_ma5(df)
    p2 = df["return_7d"].iloc[-1] > MIN_RETURN_7D_CUM
    p3_checks = [
        roe_pct is not None and roe_pct >= MIN_ROE_PCT,
        df["MA5"].iloc[-1] > df["MA10"].iloc[-1] > df["MA20"].iloc[-1],
        df["return_5d"].iloc[-1] > MIN_RETURN_5D,
        volume_ratio >= MIN_VOLUME_INCREASE,
        price_ratio < MAX_PRICE_RATIO,
    ]
    p3 = all(p3_checks)

    level = None
    if p1:
        level = 1
    if p1 and p2:
        level = 2
    if p1 and p2 and p3:
        level = 3
    if level is None:
        return None

    return {
        "Code": code,
        "ROE_Pct": round(roe_pct, 2),
        "5D_Return": round(df["return_5d"].iloc[-1], 3),
        "7D_Return": round(df["return_7d"].iloc[-1], 3),
        "Volume_Ratio": round(volume_ratio, 2),
        "Price/High": round(price_ratio, 2),
        "Appearances_7D": appearances_7d,
        "Level": level,
    }


trade_dates = get_last_trade_dates(LAST_TRADE_DAYS)
latest_trade_date = trade_dates[-1]
cache_df = update_cache_for_date(latest_trade_date)
candidate_counts = get_candidates_from_cache(trade_dates)

print(f"Trade dates (last {LAST_TRADE_DAYS}): {', '.join(trade_dates)}")
print(f"Candidates after fund flow filter: {len(candidate_counts)}")

results = []
for idx, (code, cnt) in enumerate(candidate_counts.items(), start=1):
    try:
        print(f"[{idx}/{len(candidate_counts)}] Checking {code} ...")
        hit = screen_stock(code, cnt)
        if hit:
            print(f"Hit: {code}")
            results.append(hit)
        time.sleep(random.uniform(0.3, 0.6))
    except Exception as exc:
        print(f"Error for {code}: {exc}")
        time.sleep(1)

df_res = pd.DataFrame(results)
print("\nSelected stocks (Level 1/2/3):\n")
if df_res.empty:
    print("No stocks matched the rules.")
else:
    df_res = df_res.sort_values(["Level", "Appearances_7D"], ascending=[True, False])
    for level in [1, 2, 3]:
        df_level = df_res[df_res["Level"] == level]
        print(f"\nLevel {level}:\n")
        print(df_level if not df_level.empty else "None")

if not df_res.empty:
    df_res.to_csv("selected_cn.csv", index=False, encoding="utf-8-sig")
    write_excel_with_text_code(df_res, "selected_cn.xlsx")
    print("Saved to selected_cn.csv")

print("\nScan completed.")
