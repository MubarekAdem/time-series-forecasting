import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

def debug_lstm():
    file_path = 'data/processed/TSLA_processed.csv'
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df.dropna(inplace=True)
    
    train_data = df[df.index < '2025-01-01']
    test_data = df[df.index >= '2025-01-01']
    
    print("Train NaNs:", train_data['Close'].isnull().sum())
    
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_data[['Close']])
    scaled_test = scaler.transform(test_data[['Close']])
    
    print("Scaled Train NaNs:", np.isnan(scaled_train).sum())
    
    def create_sequences(data, seq_length=60):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    seq_length = 60
    X_train, y_train = create_sequences(scaled_train, seq_length)
    
    print("X_train shape:", X_train.shape)
    
    model_lstm = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    print("Training LSTM...")
    model_lstm.fit(X_train, y_train, batch_size=32, epochs=1, verbose=1)
    print("LSTM trained.")

if __name__ == "__main__":
    try:
        debug_lstm()
    except Exception as e:
        print("LSTM failed:", e)
