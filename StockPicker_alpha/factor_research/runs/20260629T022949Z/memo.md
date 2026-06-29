# Investment Research Memo

## Data source
**real_sas_pipeline** — Part 3 panel + Part 4 quintile long-short when SAS files are present.

## Recommended factor for continued research
**BtM** is the strongest candidate after combining performance, hit rate, turnover, FF4 alpha, and monitor penalties.

## Why it might work
Book-to-market: high ratios indicate ’cheap’ equities vs. book fundamentals (value tilt), as in classical value-factor discussions.

## Evidence from the backtest
- Sharpe: 0.46
- Max drawdown: -84.70%
- Hit rate: 58.77%
- Average monthly turnover: 20.71%
- FF4 alpha (monthly): 0.0231 (SE 0.0091)

## What the monitor caught
Material drawdown

## What a PM or researcher should do next
1. Re-run the same workflow on point-in-time live data with survivorship-bias controls.
2. Add transaction costs and liquidity constraints before accepting the result as tradable.
3. Run walk-forward validation and compare against simpler baselines.
4. Test whether the factor should be combined with complementary factors rather than traded alone.

## Research governance takeaway
The value of the agentic layer is not that it magically finds alpha. Its value is that it makes the research process auditable: every attractive backtest is paired with explicit assumptions, risk flags, and next tests.