# scripts/fetch_data.py

import yfinance as yf
import pandas as pd
import os

def fetch_historical_data(tickers, period="2y", interval="1d", save_path="data/"):
    os.makedirs(save_path, exist_ok=True)
    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        # Download data
        df = yf.download(ticker, period=period, interval=interval)
        df.dropna(inplace=True)
        
        # Check if columns are MultiIndex and handle appropriately
        if isinstance(df.columns, pd.MultiIndex):
            print(f"Detected MultiIndex columns for {ticker}")
            # Select columns by level to get 1D arrays
            open_col = df['Open'].iloc[:, 0] if isinstance(df['Open'], pd.DataFrame) else df['Open']
            high_col = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
            low_col = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']
            close_col = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
            volume_col = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
            
            # Create DataFrame with flattened columns
            clean_df = pd.DataFrame({
                'Open': open_col,
                'High': high_col,
                'Low': low_col,
                'Close': close_col,
                'Volume': volume_col
            }, index=df.index)
        else:
            # No MultiIndex, create DataFrame directly
            clean_df = pd.DataFrame({
                'Open': df['Open'],
                'High': df['High'],
                'Low': df['Low'],
                'Close': df['Close'],
                'Volume': df['Volume']
            }, index=df.index)
        
        # Save with proper column headers
        csv_path = os.path.join(save_path, f"{ticker}.csv")
        clean_df.to_csv(csv_path, index_label="Date")
        print(f"Saved {ticker} data to {csv_path}")

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'AMZN']
    fetch_historical_data(tickers)