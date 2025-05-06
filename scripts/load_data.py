# scripts/load_data.py

import os
import pandas as pd

def load_data(ticker, data_dir="data"):
    file_path = os.path.join(data_dir, f"{ticker}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No data file found for {ticker} at {file_path}")
    
    df = pd.read_csv(file_path)
    print("Columns in CSV:", df.columns)

    # Rename the first column to "Date" if it's unnamed
    if df.columns[0] not in ["Date", "date"]:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df