# Portfolio Optimization — Time Series Forecasting for Portfolio Management

**GMF Investments — Week 9 Challenge:** apply time series forecasting (ARIMA/SARIMA and LSTM) and
Modern Portfolio Theory to build, validate, and backtest a data-driven portfolio across TSLA, BND,
and SPY.

## Project Structure

```
portfolio-optimization/
├── .vscode/settings.json
├── .github/workflows/unittests.yml
├── data/
│   ├── raw/                  # (gitignored) raw yfinance downloads, if cached
│   └── processed/            # cleaned per-ticker CSVs (Task 1 output, gitignored)
├── notebooks/
│   ├── task1_eda.ipynb                    # Data extraction, cleaning, EDA, ADF test, risk metrics
│   ├── task2_modeling.ipynb               # ARIMA/SARIMA + LSTM, train/test, metric comparison
│   ├── task3_forecast.ipynb               # 12-month future forecast with confidence intervals
│   ├── task4_portfolio_optimization.ipynb # Efficient frontier, Max Sharpe / Min Vol portfolios
│   └── task5_backtesting.ipynb            # Strategy vs 60/40 SPY-BND benchmark backtest
├── src/
│   ├── data_loader.py            # yfinance fetch, cleaning, returns
│   ├── eda.py                    # plots, ADF test, rolling stats, outliers
│   ├── risk_metrics.py           # VaR, Sharpe, annualized return/vol, max drawdown
│   ├── arima_model.py            # auto_arima fit + forecast helpers
│   ├── lstm_model.py             # sequence prep, Keras LSTM, iterative multi-step forecast
│   ├── evaluation.py             # MAE / RMSE / MAPE
│   ├── portfolio_optimization.py # expected returns, covariance, efficient frontier (PyPortfolioOpt)
│   └── backtesting.py            # portfolio simulator (buy&hold / rebalance) + performance metrics
├── tests/                        # pytest unit tests for src/
├── results/
│   ├── plots/                    # generated figures (committed; regenerate via notebooks)
│   └── reports/                  # generated CSV summaries consumed across notebooks (committed)
├── INVESTMENT_MEMO.md            # final investment-committee report (Task 1-5 synthesis)
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the analysis

The five notebooks in `notebooks/` are meant to be run **in order** — later notebooks read CSV
outputs written by earlier ones (`data/processed/*.csv`, `results/reports/task2_model_comparison.csv`,
`results/reports/task3_future_forecast.csv`, `results/reports/task4_recommended_portfolio.csv`).

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace notebooks/task1_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/task2_modeling.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/task3_forecast.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/task4_portfolio_optimization.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/task5_backtesting.ipynb
```

Or open them in Jupyter/VS Code and run cell-by-cell.

## Tasks

1. **Preprocess and Explore the Data** — fetch TSLA/BND/SPY (2015-01-01 to 2026-06-30) via
   `yfinance`, clean/align to a business-day calendar, run EDA (price trends, daily returns,
   rolling volatility, outliers), test stationarity with the Augmented Dickey-Fuller test, and
   compute foundational risk metrics (VaR, Sharpe ratio).
2. **Build Time Series Forecasting Models** — chronological train/test split, fit an
   `auto_arima`-selected ARIMA/SARIMA model and a 2-layer LSTM on TSLA's closing price, and
   compare them on MAE / RMSE / MAPE.
3. **Forecast Future Market Trends** — use the better-performing model from Task 2 to project TSLA
   ~12 months forward with confidence intervals, and discuss trend, opportunity/risk, and forecast
   reliability over the horizon.
4. **Optimize Portfolio Based on Forecast** — combine the Task 3 TSLA return view with BND/SPY
   historical returns, build the covariance matrix, trace the efficient frontier with
   `PyPortfolioOpt`, and recommend the Maximum Sharpe Ratio portfolio.
5. **Strategy Backtesting** — simulate the Task 4 portfolio (buy & hold and monthly rebalance)
   over the final out-of-sample year and compare cumulative returns, total/annualized return,
   Sharpe ratio, and max drawdown against a static 60% SPY / 40% BND benchmark.

## Testing

```bash
pytest tests/ -v
```

## Notes and Limitations

- Per the Efficient Market Hypothesis, none of these models are treated as standalone price
  predictors — they feed into the broader portfolio-construction workflow (Tasks 3-5), consistent
  with how these techniques are actually used in practice.
- The Task 5 backtest is a single one-year historical path, run without transaction costs or taxes;
  see the notebook's conclusion section for a full discussion of its limitations.
- YFinance downloads may occasionally be rate-limited; re-running the fetch cell after a short wait
  usually resolves it.
