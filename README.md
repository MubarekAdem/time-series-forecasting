# Time Series Forecasting - Tesla Stock Price Prediction

A comprehensive time series forecasting project implementing both classical (ARIMA) and deep learning (LSTM) approaches to predict Tesla (TSLA) stock prices.

## Project Overview

This project demonstrates end-to-end time series analysis and forecasting, from data extraction and exploratory analysis to model implementation and evaluation. The goal is to predict future Tesla stock prices using historical data from January 2015 to January 2026.

## Features

- **Data Extraction**: Automated fetching of TSLA, BND, and SPY historical data using YFinance
- **Comprehensive EDA**: Statistical analysis, visualizations, and risk metrics
- **Dual Modeling Approach**: Implementation of both ARIMA and LSTM models
- **Performance Evaluation**: Rigorous comparison using MAE, RMSE, and MAPE metrics
- **Clean Architecture**: Modular codebase with separate branches for different tasks

## Project Structure

```
time-series-forecasting/
├── data/
│   ├── raw/              # Raw data from YFinance
│   └── processed/        # Cleaned and processed datasets
├── notebooks/
│   ├── eda.ipynb         # Exploratory Data Analysis
│   └── forecasting.ipynb # Model implementation notebook
├── src/
│   └── data_loader.py    # Data extraction utilities
├── scripts/
│   └── run_forecasting.py # Production forecasting script
├── model_comparison.csv   # Model performance metrics
├── discussion.md          # Model analysis and rationale
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

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Extraction
```bash
python src/data_loader.py
```

### Run Forecasting Models
```bash
python scripts/run_forecasting.py
```

### Explore Notebooks
```bash
jupyter notebook notebooks/eda.ipynb
jupyter notebook notebooks/forecasting.ipynb
```

## Results

| Model | MAE | RMSE | MAPE (%) |
|-------|-----|------|----------|
| ARIMA | 69.50 | 82.93 | 22.56 |
| **LSTM** | **20.30** | **25.36** | **5.61** |

**Key Findings:**
- LSTM significantly outperformed ARIMA across all metrics
- LSTM achieved ~5.6% MAPE, demonstrating strong predictive accuracy
- The model successfully captured non-linear patterns in Tesla's volatile stock movements

## Task Breakdown

### Task 1: Data Preprocessing and EDA (Branch: `task-1`)
- ✅ Extract TSLA, BND, SPY data (Jan 2015 - Jan 2026)
- ✅ Data cleaning and validation
- ✅ 3+ EDA visualizations (Closing Price, Daily Returns, Rolling Stats)
- ✅ ADF Stationarity Test
- ✅ Risk Metrics (VaR, Sharpe Ratio)

### Task 2: Forecasting Models (Branch: `task-2`)
- ✅ Chronological train/test split (2015-2024 / 2025-2026)
- ✅ ARIMA model with auto-parameter selection
- ✅ LSTM model with sequence generation
- ✅ Model evaluation and comparison

## Technical Details

### Models Implemented

**ARIMA (AutoRegressive Integrated Moving Average)**
- Classical statistical approach
- Auto-parameter selection using `pmdarima`
- Best order determined through AIC/BIC optimization

**LSTM (Long Short-Term Memory)**
- Deep learning approach for sequence prediction
- Architecture: 2 LSTM layers (50 units each) + Dense layers
- Sequence length: 60 days
- Training: 5 epochs, batch size 32

### Why LSTM Performed Better

1. **Non-linearity**: Captures complex patterns in volatile stock data
2. **Long-term dependencies**: Learns broader market trends
3. **Adaptability**: Better handles Tesla's high volatility

See [discussion.md](discussion.md) for detailed analysis.

## Future Improvements

- Hyperparameter tuning (grid search, Bayesian optimization)
- Ensemble methods combining ARIMA and LSTM
- External features (sentiment analysis, trading volume, market indicators)
- Transformer-based models (Temporal Fusion Transformers)
- Real-time prediction pipeline

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- pmdarima
- scikit-learn
- tensorflow/keras
- yfinance

## Repository Best Practices

- ✅ Clean commit history with descriptive messages
- ✅ Branch-based development (`task-1`, `task-2`)
- ✅ PEP8 compliant code
- ✅ Comprehensive documentation
- ✅ Modular, reusable code structure

## License

This project is for educational purposes.

## Author

Mohammed Sultan

## Acknowledgments

- Data source: Yahoo Finance (via YFinance API)
- Inspiration: Financial time series analysis and forecasting techniques
