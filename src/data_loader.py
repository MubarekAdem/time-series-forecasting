import yfinance as yf
import pandas as pd
import os

def fetch_data(tickers, start_date, end_date, output_dir="data/raw"):
    """
    Fetches historical data for the given tickers and saves them to CSV files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    combined_data = {}
    
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            print(f"Warning: No data found for {ticker}")
            continue
            
        # Save individual ticker data
        file_path = os.path.join(output_dir, f"{ticker}.csv")
        data.to_csv(file_path)
        print(f"Saved {ticker} data to {file_path}")
        combined_data[ticker] = data

    return combined_data

if __name__ == "__main__":
    TICKERS = ["TSLA", "BND", "SPY"]
    START_DATE = "2015-01-01"
    END_DATE = "2026-01-15"
    
    fetch_data(TICKERS, START_DATE, END_DATE)
