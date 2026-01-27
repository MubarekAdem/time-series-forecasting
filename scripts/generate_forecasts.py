"""
Generate Future Forecasts with Confidence Intervals

This script generates 6-month and 12-month ahead forecasts for TSLA stock prices
using both ARIMA and LSTM models, with confidence intervals and trend analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_data():
    """Load processed TSLA data."""
    print("Loading data...")
    file_path = 'data/processed/TSLA_processed.csv'
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(inplace=True)
    return df

def train_arima_model(data):
    """Train ARIMA model and return fitted model."""
    print("\nTraining ARIMA model...")
    model_auto = auto_arima(
        data['Close'], 
        seasonal=False, 
        trace=False, 
        error_action='ignore', 
        suppress_warnings=True,
        stepwise=True
    )
    print(f"Best ARIMA order: {model_auto.order}")
    return model_auto

def generate_arima_forecasts(model, n_periods):
    """Generate ARIMA forecasts with confidence intervals."""
    print(f"Generating ARIMA forecast for {n_periods} periods...")
    forecast_result = model.predict(n_periods=n_periods, return_conf_int=True, alpha=0.05)
    
    if isinstance(forecast_result, tuple):
        predictions, conf_int = forecast_result
    else:
        predictions = forecast_result
        conf_int = None
    
    return predictions, conf_int

def train_lstm_model(data, seq_length=60):
    """Train LSTM model and return model and scaler."""
    print("\nTraining LSTM model...")
    
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length])
    X, y = np.array(X), np.array(y)
    
    # Build model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=32, epochs=5, verbose=0)
    
    return model, scaler, scaled_data, seq_length

def generate_lstm_forecasts(model, scaler, last_sequence, n_periods):
    """Generate LSTM forecasts."""
    print(f"Generating LSTM forecast for {n_periods} periods...")
    
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_periods):
        # Predict next value
        pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
        predictions.append(pred[0, 0])
        
        # Update sequence
        current_sequence = np.append(current_sequence[1:], pred[0, 0])
    
    # Inverse transform
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

def create_forecast_dates(start_date, n_periods, freq='B'):
    """Create future dates for forecasts (business days)."""
    return pd.bdate_range(start=start_date, periods=n_periods, freq=freq)

def plot_forecasts(historical_data, forecast_6m_arima, forecast_12m_arima, 
                   forecast_6m_lstm, forecast_12m_lstm, 
                   conf_int_6m, conf_int_12m, dates_6m, dates_12m):
    """Plot historical data with forecasts and confidence intervals."""
    print("\nCreating forecast visualization...")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 6-month forecast
    ax1 = axes[0]
    # Historical data (last 2 years)
    historical_data_recent = historical_data.last('730D')
    ax1.plot(historical_data_recent.index, historical_data_recent['Close'], 
             label='Historical Price', color='black', linewidth=2)
    
    # ARIMA forecast
    ax1.plot(dates_6m, forecast_6m_arima, label='ARIMA Forecast', 
             color='blue', linewidth=2, linestyle='--')
    
    # Confidence intervals
    if conf_int_6m is not None:
        ax1.fill_between(dates_6m, conf_int_6m[:, 0], conf_int_6m[:, 1], 
                         color='blue', alpha=0.2, label='95% Confidence Interval')
    
    # LSTM forecast
    ax1.plot(dates_6m, forecast_6m_lstm, label='LSTM Forecast', 
             color='green', linewidth=2, linestyle='--')
    
    ax1.set_title('6-Month Ahead Forecast (Feb 2026 - Jul 2026)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('TSLA Price ($)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 12-month forecast
    ax2 = axes[1]
    ax2.plot(historical_data_recent.index, historical_data_recent['Close'], 
             label='Historical Price', color='black', linewidth=2)
    
    # ARIMA forecast
    ax2.plot(dates_12m, forecast_12m_arima, label='ARIMA Forecast', 
             color='blue', linewidth=2, linestyle='--')
    
    # Confidence intervals
    if conf_int_12m is not None:
        ax2.fill_between(dates_12m, conf_int_12m[:, 0], conf_int_12m[:, 1], 
                         color='blue', alpha=0.2, label='95% Confidence Interval')
    
    # LSTM forecast
    ax2.plot(dates_12m, forecast_12m_lstm, label='LSTM Forecast', 
             color='green', linewidth=2, linestyle='--')
    
    ax2.set_title('12-Month Ahead Forecast (Feb 2026 - Jan 2027)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('TSLA Price ($)', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = 'outputs/forecasts/forecast_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved forecast plot to {output_path}")
    plt.close()

def analyze_trends(forecast_6m, forecast_12m, conf_int_6m, conf_int_12m, current_price):
    """Analyze forecast trends and identify opportunities and risks."""
    print("\nAnalyzing trends...")
    
    analysis = []
    analysis.append("# Forecast Trend Analysis\n")
    analysis.append(f"**Date Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    analysis.append(f"**Current TSLA Price:** ${current_price:.2f}\n\n")
    
    # 6-month analysis
    analysis.append("## 6-Month Forecast Analysis (Feb - Jul 2026)\n\n")
    
    # Ensure we use numpy arrays for calculation to avoid indexing issues
    forecast_6m = np.array(forecast_6m)
    forecast_12m = np.array(forecast_12m)
    if conf_int_6m is not None: conf_int_6m = np.array(conf_int_6m)
    if conf_int_12m is not None: conf_int_12m = np.array(conf_int_12m)
    
    forecast_6m_mean = np.mean(forecast_6m)
    forecast_6m_final = forecast_6m[-1]
    forecast_6m_change = ((forecast_6m_final - current_price) / current_price) * 100
    
    analysis.append(f"**Predicted Price Range:** ${forecast_6m.min():.2f} - ${forecast_6m.max():.2f}\n")
    analysis.append(f"**Average Predicted Price:** ${forecast_6m_mean:.2f}\n")
    analysis.append(f"**End of Period Price:** ${forecast_6m_final:.2f}\n")
    analysis.append(f"**Expected Change:** {forecast_6m_change:+.2f}%\n\n")
    
    if conf_int_6m is not None:
        analysis.append(f"**95% Confidence Interval (End):** ${conf_int_6m[-1, 0]:.2f} - ${conf_int_6m[-1, 1]:.2f}\n\n")
    
    # Trend direction
    if forecast_6m_change > 5:
        analysis.append("**Trend:** [Upward] - Strong growth expected\n\n")
        analysis.append("**Opportunities:**\n")
        analysis.append("- Potential for capital appreciation in the short term\n")
        analysis.append("- Momentum-based trading strategies may be favorable\n")
        analysis.append("- Consider increasing allocation if risk tolerance allows\n\n")
        analysis.append("**Risks:**\n")
        analysis.append("- High volatility may lead to sharp corrections\n")
        analysis.append("- Model may not account for unexpected negative events\n")
        analysis.append("- Overvaluation risk if growth is not sustained\n\n")
    elif forecast_6m_change < -5:
        analysis.append("**Trend:** [Downward] - Decline expected\n\n")
        analysis.append("**Opportunities:**\n")
        analysis.append("- Potential buying opportunity at lower prices\n")
        analysis.append("- Consider defensive strategies (hedging, reduced allocation)\n\n")
        analysis.append("**Risks:**\n")
        analysis.append("- Further downside potential if negative trend continues\n")
        analysis.append("- Market sentiment may deteriorate\n")
        analysis.append("- Consider stop-loss orders to limit downside\n\n")
    else:
        analysis.append("**Trend:** [Sideways] - Relatively stable\n\n")
        analysis.append("**Opportunities:**\n")
        analysis.append("- Range-bound trading strategies may be effective\n")
        analysis.append("- Lower volatility reduces downside risk\n\n")
        analysis.append("**Risks:**\n")
        analysis.append("- Limited upside potential in the short term\n")
        analysis.append("- Breakout in either direction remains possible\n\n")
    
    # 12-month analysis
    analysis.append("---\n\n")
    analysis.append("## 12-Month Forecast Analysis (Feb 2026 - Jan 2027)\n\n")
    
    forecast_12m_mean = np.mean(forecast_12m)
    forecast_12m_final = forecast_12m[-1]
    forecast_12m_change = ((forecast_12m_final - current_price) / current_price) * 100
    
    analysis.append(f"**Predicted Price Range:** ${forecast_12m.min():.2f} - ${forecast_12m.max():.2f}\n")
    analysis.append(f"**Average Predicted Price:** ${forecast_12m_mean:.2f}\n")
    analysis.append(f"**End of Period Price:** ${forecast_12m_final:.2f}\n")
    analysis.append(f"**Expected Change:** {forecast_12m_change:+.2f}%\n\n")
    
    if conf_int_12m is not None:
        analysis.append(f"**95% Confidence Interval (End):** ${conf_int_12m[-1, 0]:.2f} - ${conf_int_12m[-1, 1]:.2f}\n\n")
    
    # Long-term trend
    if forecast_12m_change > 10:
        analysis.append("**Long-Term Trend:** [Strong Growth] - Significant appreciation expected\n\n")
    elif forecast_12m_change > 0:
        analysis.append("**Long-Term Trend:** [Moderate Growth] - Positive outlook\n\n")
    elif forecast_12m_change > -10:
        analysis.append("**Long-Term Trend:** [Moderate Decline] - Cautious outlook\n\n")
    else:
        analysis.append("**Long-Term Trend:** [Strong Decline] - Bearish outlook\n\n")
    
    analysis.append("**Strategic Recommendations:**\n")
    if forecast_12m_change > 0:
        analysis.append("- Long-term investors may consider holding or accumulating\n")
        analysis.append("- Monitor for entry points during short-term pullbacks\n")
    else:
        analysis.append("- Consider reducing exposure or implementing hedging strategies\n")
        analysis.append("- Reassess investment thesis and fundamentals\n")
    
    analysis.append("\n---\n\n")
    analysis.append("## Risk Considerations\n\n")
    analysis.append("- Forecasts are based on historical patterns and may not capture regime changes\n")
    analysis.append("- Black swan events (regulatory changes, economic shocks) are not predicted\n")
    analysis.append("- Confidence intervals widen over time, reflecting increased uncertainty\n")
    analysis.append("- LSTM and ARIMA models may diverge, indicating model uncertainty\n")
    analysis.append("- Always use forecasts as **one input** among many in investment decisions\n")
    
    return ''.join(analysis)

def main():
    """Main execution function."""
    print("=" * 60)
    print("TSLA Stock Price Forecasting")
    print("=" * 60)
    
    # Load data
    df = load_data()
    current_price = df['Close'].iloc[-1]
    print(f"Current TSLA price: ${current_price:.2f}")
    print(f"Data range: {df.index.min()} to {df.index.max()}")
    
    # Train ARIMA model
    arima_model = train_arima_model(df)
    
    # Generate ARIMA forecasts
    forecast_6m_arima, conf_int_6m = generate_arima_forecasts(arima_model, 126)  # ~6 months business days
    forecast_12m_arima, conf_int_12m = generate_arima_forecasts(arima_model, 252)  # ~12 months business days
    
    # Train LSTM model
    lstm_model, scaler, scaled_data, seq_length = train_lstm_model(df)
    last_sequence = scaled_data[-seq_length:].flatten()
    
    # Generate LSTM forecasts
    forecast_6m_lstm = generate_lstm_forecasts(lstm_model, scaler, last_sequence, 126)
    forecast_12m_lstm = generate_lstm_forecasts(lstm_model, scaler, last_sequence, 252)
    
    # Create forecast dates
    start_date = df.index.max() + timedelta(days=1)
    dates_6m = create_forecast_dates(start_date, 126)
    dates_12m = create_forecast_dates(start_date, 252)
    
    # Save forecasts to CSV
    forecast_6m_df = pd.DataFrame({
        'Date': dates_6m,
        'ARIMA_Forecast': forecast_6m_arima,
        'LSTM_Forecast': forecast_6m_lstm,
        'Lower_CI': conf_int_6m[:, 0] if conf_int_6m is not None else np.nan,
        'Upper_CI': conf_int_6m[:, 1] if conf_int_6m is not None else np.nan
    })
    forecast_6m_df.to_csv('outputs/forecasts/forecast_6month.csv', index=False)
    print("\nSaved 6-month forecast to outputs/forecasts/forecast_6month.csv")
    
    forecast_12m_df = pd.DataFrame({
        'Date': dates_12m,
        'ARIMA_Forecast': forecast_12m_arima,
        'LSTM_Forecast': forecast_12m_lstm,
        'Lower_CI': conf_int_12m[:, 0] if conf_int_12m is not None else np.nan,
        'Upper_CI': conf_int_12m[:, 1] if conf_int_12m is not None else np.nan
    })
    forecast_12m_df.to_csv('outputs/forecasts/forecast_12month.csv', index=False)
    print("Saved 12-month forecast to outputs/forecasts/forecast_12month.csv")
    
    # Plot forecasts
    plot_forecasts(df, forecast_6m_arima, forecast_12m_arima, 
                   forecast_6m_lstm, forecast_12m_lstm,
                   conf_int_6m, conf_int_12m, dates_6m, dates_12m)
    
    # Analyze trends
    trend_analysis = analyze_trends(
        forecast_6m_arima, forecast_12m_arima, 
        conf_int_6m, conf_int_12m, current_price
    )
    
    with open('outputs/forecasts/trend_analysis.md', 'w') as f:
        f.write(trend_analysis)
    print("Saved trend analysis to outputs/forecasts/trend_analysis.md")
    
    print("\n" + "=" * 60)
    print("Forecasting complete!")
    print("=" * 60)
    print("\nOutputs:")
    print("  - outputs/forecasts/forecast_6month.csv")
    print("  - outputs/forecasts/forecast_12month.csv")
    print("  - outputs/forecasts/forecast_visualization.png")
    print("  - outputs/forecasts/trend_analysis.md")

if __name__ == "__main__":
    main()
