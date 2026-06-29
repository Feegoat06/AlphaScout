# Investment Research Memo

## Data source
**synthetic_fallback** — Part 3 panel + Part 4 quintile long-short when SAS files are present.

## Recommended factor for continued research
**momentum** is the strongest candidate after combining performance, hit rate, turnover, FF4 alpha, and monitor penalties.

## Why it might work
Past relative strength over a recent horizon; captures gradual information diffusion / underreaction and related risk narratives for continuation.

## Evidence from the backtest
- Sharpe: 0.39
- Max drawdown: -14.61%
- Hit rate: 61.62%
- Average monthly turnover: 34.85%
- FF4 alpha (monthly): nan (SE nan)

## What the monitor caught
Unstable rolling Sharpe

## What a PM or researcher should do next
1. Re-run the same workflow on point-in-time live data with survivorship-bias controls.
2. Add transaction costs and liquidity constraints before accepting the result as tradable.
3. Run walk-forward validation and compare against simpler baselines.
4. Test whether the factor should be combined with complementary factors rather than traded alone.

## Research governance takeaway
The value of the agentic layer is not that it magically finds alpha. Its value is that it makes the research process auditable: every attractive backtest is paired with explicit assumptions, risk flags, and next tests.