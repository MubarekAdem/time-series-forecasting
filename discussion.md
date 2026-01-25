# Model Selection and Performance Discussion

## Model Comparison

| Model | MAE | RMSE | MAPE (%) |
|-------|-----|------|----------|
| ARIMA | 69.50 | 82.93 | 22.56 |
| LSTM | 20.30 | 25.36 | 5.61 |

## Rationale

### Performance Analysis
The LSTM (Long Short-Term Memory) model significantly outperformed the ARIMA model across all metrics. 
- **MAPE**: LSTM achieved a Mean Absolute Percentage Error of ~5.6%, compared to ~22.6% for ARIMA.
- **Accuracy**: The LSTM was able to capture the trends and price movements of Tesla stock much more accurately in the test period (2025-2026).

### Why LSTM Performed Better
1.  **Non-Linearity**: Financial time series data, especially for volatile stocks like TSLA, contain complex non-linear patterns. ARIMA is a linear model and struggles to capture these intricacies compared to deep learning models like LSTMs which are designed for non-linear sequence modeling.
2.  **Long-Term Dependencies**: LSTMs are capable of learning long-term dependencies in data sequences, which helps in understanding broader market trends beyond immediate past values.
3.  **Volatility**: Tesla stock is known for high volatility. ARIMA tends to revert to the mean or project simple linear trends, whereas LSTM can adapt better to fluctuating variances.

### Conclusion
For forecasting Tesla stock prices, the LSTM model provides a superior predictive capability compared to the baseline ARIMA model. Future improvements could include hyperparameter tuning, adding external features (sentiment, volume), or exploring Transformer-based architectures.
