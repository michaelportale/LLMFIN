# scripts/fetch_data.py

import yfinance as yf
import pandas as pd
import os

def fetch_historical_data(tickers, period="2y", interval="1d", save_path="data/"):
    os.makedirs(save_path, exist_ok=True)
    for ticker in tickers:
        df = yf.download(ticker, period=period, interval=interval)
        df.dropna(inplace=True)
        csv_path = os.path.join(save_path, f"{ticker}.csv")
        df.to_csv(csv_path, index=True)
        print(f"Saved {ticker} data to {csv_path}")

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'AMZN']
    fetch_historical_data(tickers)