import yfinance as yf
import pandas as pd
import numpy as np

# ------------------------------
# Stock List
# ------------------------------
ticker_list = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA",
    "TSLA","META","JPM","JNJ","V","WMT",
    "AMD","NFLX","BABA","DIS","BAC"
]


# ------------------------------
# 筛选参数
# ------------------------------
lookback_days = 60 # Condition: 最近 60 天用于判断位置
min_roe = 10 # Condition: ROE > 10%
min_return_5d = 0.03 # Condition: 5日涨幅 > 3%
min_return_7d = 0.25 # Condition: 7日涨幅 > 25%
min_volume_increase = 1.2 # Condition: 成交量至少放大 20%

results = [] # Init Result

for ticker in ticker_list:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # 获取 ROE
        roe = info.get("returnOnEquity")
        if roe is None or roe < min_roe/100:
            continue

        # 获取 K 线
        # Note for developer: df is a DataFrame, which is kinda like an excel file with multiple 
        # rows and columns
        df = stock.history(period="3mo")  # 足够覆盖60交易日
        if len(df) < lookback_days:
            continue

        # ------------------------------
        # 计算指标
        # ------------------------------
        df['MA5'] = df['Close'].rolling(5).mean() # df['Close'] is pre-defined
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()

        # 最近5日与7日收益
        last_5d_return = df['Close'].pct_change(5).iloc[-1] # pct_change() and iloc are built-in functions in pandas
        last_7d_return = (df['Close'].iloc[-1] - df['Close'].iloc[-7]) / df['Close'].iloc[-7]

        # 成交量放大
        volume_now = df['Volume'].tail(5).mean()
        volume_prev = df['Volume'].tail(15).head(10).mean()
        volume_ratio = volume_now / volume_prev if volume_prev > 0 else 0

        # 历史高位判断
        highest_price = df['Close'].max()
        current_price = df['Close'].iloc[-1]
        price_ratio = current_price / highest_price

        # ------------------------------
        # 满足条件判定（与你的策略一致）
        # ------------------------------
        if (
            df['MA5'].iloc[-1] > df['MA10'].iloc[-1] > df['MA20'].iloc[-1] and
            last_5d_return > min_return_5d and
            last_7d_return > min_return_7d and
            volume_ratio >= min_volume_increase and
            price_ratio < 0.90    # 不在历史高位
        ):
            results.append({
                "Ticker": ticker,
                "ROE": roe,
                "5D_Return": round(last_5d_return, 3),
                "7D_Return": round(last_7d_return, 3),
                "Volume_Ratio": round(volume_ratio, 2),
                "Price/High": round(price_ratio, 2)
            })
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")

# 输出结果
df_res = pd.DataFrame(results)
print("\n📌 满足你条件的股票候选：\n")
print(df_res if not df_res.empty else "❌ 当前股票池内没有满足条件的。")
