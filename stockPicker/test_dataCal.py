import akshare as ak
import pandas as pd

print("🔥 Testing Script: v1.0 - START 🔥")

code = "600519"   # 你可以换其他股票测试

try:
    print(f"\n📌 Step 1: 获取日线数据 stock_zh_a_hist({code}) ...")
    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date="20190101",
        end_date="20991231",
        adjust="qfq"
    )

    if df is None or df.empty:
        raise ValueError("❌ ERROR: 返回数据为空！")

    print("✔ 数据成功获取，前5行：")
    print(df.head())

    # ------------------------------
    # 列名检查 & 重命名
    # ------------------------------
    print("\n📌 Step 2: 检查列名并重命名为英文 ...")

    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
    })

    # 检查必备列是否存在
    required_cols = ["date", "open", "close", "high", "low", "volume"]
    for col in required_cols:
        assert col in df.columns, f"❌ 缺失列: {col}"

    print("✔ 列名正常：", df.columns.tolist())


    # ------------------------------
    # 数据类型检查
    # ------------------------------
    print("\n📌 Step 3: 检查 date 是否转换为 datetime ...")
    df["date"] = pd.to_datetime(df["date"])
    assert pd.api.types.is_datetime64_any_dtype(df["date"]), "❌ date 不是 datetime 类型"
    print("✔ date 转换正常")


    # ------------------------------
    # 技术指标测试
    # ------------------------------
    print("\n📌 Step 4: 测试技术指标计算 ...")

    df["MA5"] = df["close"].rolling(5).mean()
    df["MA20"] = df["close"].rolling(20).mean()

    assert df["MA5"].notna().sum() > 0, "❌ MA5 全为空"
    assert df["MA20"].notna().sum() > 0, "❌ MA20 全为空"

    print("✔ MA5 & MA20 正常计算")


    # ------------------------------
    # 涨幅计算测试
    # ------------------------------
    print("\n📌 Step 5: 测试5日涨幅计算 ...")

    df["return_5d"] = df["close"].pct_change(5)
    latest_return = df["return_5d"].iloc[-1]

    print(f"✔ 最近5日涨幅(return_5d)：{round(latest_return, 3)}")


    # ------------------------------
    # 成功结束
    # ------------------------------
    print("\n🎉 测试全部通过，没有发现问题！")

except Exception as e:
    print("\n❌ TEST FAILED:")
    print(e)

print("\n🔥 Testing Script: END 🔥")
