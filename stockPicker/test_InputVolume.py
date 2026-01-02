import akshare as ak

df_flow = ak.stock_individual_fund_flow_rank(indicator="今日")
print(df_flow.columns)
print(df_flow.head(3))
