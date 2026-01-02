import akshare as ak
import pandas as pd
import numpy as np
import time
import random

print("🔥 CN Stock Screener — Scan First 50 A Stocks v3.1")

# ------------------------------
# 获取全部 A 股代码 & 取前50
# ------------------------------
stock_df = ak.stock_zh_a_spot_em()
ticker_list = stock_df["代码"].tolist()[:50]
print(f"📌 Loaded {len(ticker_list)} tickers.")

# ------------------------------
# 选股参数
# ------------------------------
lookback_days = 60
min_return_5d = 0.03
min_return_7d = 0.25
min_volume_increase = 1.2
results = []

# ------------------------------
# 扫描前 50 支股票
# ------------------------------
for idx, code in enumerate(ticker_list):
    try:
        print(f"\n[{idx+1}/{len(ticker_list)}] 🔍 Checking {code} ...")

        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date="20180101",
            end_date="20991231",
            adjust="qfq"
        )
        if df.empty:
            print("⚠️ 数据为空，跳过")
            continue

        # --- 统一列名 ---
        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
        })

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        if len(df) < lookback_days:
            print("⚠️ 数据不足 60 天，跳过")
            continue

        # --- 技术指标 ---
        df["MA5"]  = df["close"].rolling(5).mean()
        df["MA10"] = df["close"].rolling(10).mean()
        df["MA20"] = df["close"].rolling(20).mean()
        df["return_5d"] = df["close"].pct_change(5)
        df["return_7d"] = df["close"].pct_change(7)

        # --- 成交量 ---
        volume_now = df["volume"].tail(5).mean()
        volume_prev = df["volume"].tail(15).head(10).mean()
        volume_ratio = volume_now / volume_prev if volume_prev else 0

        # --- 价格位置 ---
        highest_price = df["close"].max()
        current_price = df["close"].iloc[-1]
        price_ratio = current_price / highest_price

        # --- 选股条件 ---
        if (
            df["MA5"].iloc[-1] > df["MA10"].iloc[-1] > df["MA20"].iloc[-1] and
            df["return_5d"].iloc[-1] > min_return_5d and
            df["return_7d"].iloc[-1] > min_return_7d and
            volume_ratio >= min_volume_increase and
            price_ratio < 0.90
        ):
            print(f"🎯 {code} HIT CONDITIONS!")
            results.append({
                "Code": code,
                "5D_Return": round(df['return_5d'].iloc[-1], 3),
                "7D_Return": round(df['return_7d'].iloc[-1], 3),
                "Volume_Ratio": round(volume_ratio, 2),
                "Price/High": round(price_ratio, 2)
            })

        # --- 防止限流 ---
        time.sleep(random.uniform(0.3, 0.6))

    except Exception as e:
        print(f"❌ Error for {code}: {e}")
        time.sleep(1)

# ------------------------------
# 输出结果
# ------------------------------
df_res = pd.DataFrame(results)
print("\n🎯 满足你策略的股票：\n")
print(df_res if not df_res.empty else "❌ 今天 50 支股票中没有满足条件的")

if not df_res.empty:
    df_res.to_csv("selected_top50_cn.csv", index=False, encoding="utf-8-sig")
    print("💾 已保存到 selected_top50_cn.csv")

print("\n🔥 Scan of top 50 completed.")
