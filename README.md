# Time Series Forecasting - Tesla Stock Price Prediction

A comprehensive time series forecasting project implementing classical (ARIMA) and deep learning (LSTM) approaches for Tesla (TSLA) stock prices, including future forecasting, portfolio optimization, and strategy backtesting.

## Project Overview

This project demonstrates end-to-end time series analysis and forecasting, from data extraction and exploratory analysis to model implementation, future prediction, risk-adjusted portfolio optimization, and rigorous backtesting against market benchmarks.

## Features

- **Data Extraction**: Automated fetching of TSLA, BND, and SPY historical data using YFinance
- **Comprehensive EDA**: Statistical analysis, visualizations, and risk metrics (VaR, Sharpe Ratio)
- **Dual Modeling**: Implementation and performance comparison of ARIMA and LSTM models
- **Future Forecasting**: 6-month and 12-month ahead predictions with 95% confidence intervals
- **Portfolio Optimization**: Modern Portfolio Theory implementation using `PyPortfolioOpt` to identify Max Sharpe and Min Volatility portfolios
- **Strategy Backtesting**: Performance simulation against a 60/40 SPY/BND benchmark (Jan 2025 - Jan 2026)

## Project Structure

```
time-series-forecasting/
├── data/
│   ├── raw/                # Raw data from YFinance
│   └── processed/          # Cleaned and processed datasets
├── notebooks/
│   ├── eda.ipynb           # Exploratory Data Analysis
│   └── forecasting.ipynb   # Model implementation notebook
├── src/
│   └── data_loader.py      # Data extraction utilities
├── scripts/
│   ├── run_forecasting.py  # Model comparison and local evaluation
│   ├── generate_forecasts.py # Future price predictions (6-12m)
│   ├── portfolio_optimization.py # MPT and Efficient Frontier
│   └── backtest_strategy.py # Strategy performance simulation
├── outputs/                # Generated reports and visualizations
│   ├── forecasts/          # Forecast CSVs, plots, and trend analysis
│   ├── portfolio/          # Efficient Frontier and recommended allocations
│   └── backtesting/        # Backtest metrics and performance plots
├── model_comparison.csv     # Local evaluation metrics
├── discussion.md            # Model analysis and rationale
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/msultan001/time-series-forecasting.git
cd time-series-forecasting
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install pinned dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Extraction
```bash
python src/data_loader.py
```

### 2. Model Comparison (Task 1 & 2)
```bash
python scripts/run_forecasting.py
```

### 3. Future Forecasting (Task 3)
```bash
python scripts/generate_forecasts.py
```

### 4. Portfolio Optimization (Task 4)
```bash
python scripts/portfolio_optimization.py
```

### 5. Strategy Backtesting (Task 5)
```bash
python scripts/backtest_strategy.py
```

## Results Summary

### Model Performance (Historical Test)
| Model | MAE | RMSE | MAPE (%) |
|-------|-----|------|----------|
| ARIMA | 69.50 | 82.93 | 22.56 |
| **LSTM** | **20.30** | **25.36** | **5.61** |

### Portfolio Backtest (2025-2026)
| Metric | Optimized Portfolio | 60/40 Benchmark |
|--------|---------------------|-----------------|
| Total Return | **21.59%** | 15.07% |
| Sharpe Ratio | 0.605 | **0.893** |
| Max Drawdown | -26.03% | **-11.29%** |

## Task Breakdown

### Task 1 & 2: Data & Modeling ✅
- Extracts TSLA, BND, SPY data (2015-2026)
- Implements ARIMA and LSTM for historical evaluation
- LSTM achieved ~5.6% MAPE, capturing non-linear volatility better than ARIMA

### Task 3: Future Forecasting ✅
- Generates 6 and 12-month ahead forecasts from Jan 2026
- Includes 95% confidence bands for ARIMA predictions
- Outputs trend analysis identifying specific opportunities and risks

### Task 4: Portfolio Optimization ✅
- Uses `PyPortfolioOpt` for Efficient Frontier generation
- Identifies **Max Sharpe Ratio** and **Min Volatility** portfolios
- Provides clear allocation recommendations with metrics

### Task 5: Strategy Backtesting ✅
- Simulates recommended portfolio vs. 60/40 benchmark
- Calculates cumulative returns, Sharpe Ratio, and Max Drawdown
- Documents conclusive assessment of risk-adjusted performance

## Technical Details

- **ARIMA**: Statistical baseline using `pmdarima` for auto-parameter selection.
- **LSTM**: Multi-layer recurrent neural network with 60-day lookback for capturing long-term dependencies.
- **Optimization**: Modern Portfolio Theory (MPT) implementation for optimal capital allocation.

## Dependencies

- yfinance, pandas, numpy, matplotlib, seaborn, statsmodels, scikit-learn, pmdarima, tensorflow, pypfopt

## Author

Mohammed Sultan

## License

This project is for educational purposes.
