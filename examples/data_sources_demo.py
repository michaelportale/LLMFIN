#!/usr/bin/env python3
"""
Example script demonstrating the usage of various data sources and data quality verification.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
from dotenv import load_dotenv
import argparse
from datetime import datetime, timedelta

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our data sources
from scripts.data_sources import (
    YahooFinanceDataSource,
    AlphaVantageDataSource,
    QuandlDataSource,
    CryptoDataSource,
    ForexDataSource,
    AlphaVantageRealTime,
    CryptoRealTimeStream,
    DataStreamManager
)
from scripts.data_quality import DataQualityChecker, compare_data_sources, visualize_comparison


def load_api_keys():
    """
    Load API keys from environment variables.
    """
    load_dotenv()
    
    return {
        "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
        "quandl": os.getenv("QUANDL_API_KEY"),
        "cryptocompare": os.getenv("CRYPTOCOMPARE_API_KEY"),
        "fixer": os.getenv("FIXER_API_KEY"),
        "oanda": os.getenv("OANDA_API_KEY")
    }


def demo_stocks():
    """
    Demonstrate fetching stock data from different sources and compare them.
    """
    print("\n" + "="*80)
    print("STOCK DATA COMPARISON DEMO")
    print("="*80)
    
    # Load API keys
    api_keys = load_api_keys()
    
    # Initialize data sources
    yahoo_finance = YahooFinanceDataSource(save_path="data/yahoo/")
    
    if api_keys["alpha_vantage"]:
        alpha_vantage = AlphaVantageDataSource(api_key=api_keys["alpha_vantage"], save_path="data/alpha_vantage/")
    else:
        print("Alpha Vantage API key not found. Skipping Alpha Vantage demo.")
        alpha_vantage = None
    
    # Set up parameters
    symbols = ["AAPL", "MSFT", "GOOG"]
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Fetch data from Yahoo Finance
    print("\nFetching data from Yahoo Finance...")
    yahoo_data = yahoo_finance.fetch_historical_data(symbols, start_date=start_date, end_date=end_date)
    
    # Fetch data from Alpha Vantage if available
    alpha_vantage_data = {}
    if alpha_vantage:
        print("\nFetching data from Alpha Vantage...")
        alpha_vantage_data = alpha_vantage.fetch_historical_data(symbols, start_date=start_date, end_date=end_date)
    
    # Compare data sources
    if alpha_vantage and "AAPL" in yahoo_data and "AAPL" in alpha_vantage_data:
        print("\nComparing AAPL data from Yahoo Finance and Alpha Vantage...")
        compare_data_sources(
            yahoo_data["AAPL"], 
            alpha_vantage_data["AAPL"],
            source1_name="Yahoo Finance",
            source2_name="Alpha Vantage"
        )
        
        # Visualize the comparison
        visualize_comparison(
            yahoo_data["AAPL"], 
            alpha_vantage_data["AAPL"],
            source1_name="Yahoo Finance",
            source2_name="Alpha Vantage",
            output_file="data/comparison_AAPL.png"
        )
    
    # Check data quality
    print("\nChecking data quality for Yahoo Finance AAPL data...")
    checker = DataQualityChecker()
    quality_report = checker.generate_report(yahoo_data["AAPL"])
    
    # Visualize data quality
    checker.visualize_data_quality(yahoo_data["AAPL"], output_file="data/quality_AAPL.png")


def demo_crypto():
    """
    Demonstrate fetching cryptocurrency data.
    """
    print("\n" + "="*80)
    print("CRYPTOCURRENCY DATA DEMO")
    print("="*80)
    
    # Load API keys
    api_keys = load_api_keys()
    
    # Initialize data sources
    crypto_source = CryptoDataSource(
        api_key=api_keys.get("cryptocompare"),
        provider="coingecko",  # No API key needed for basic CoinGecko
        save_path="data/crypto/"
    )
    
    # Set up parameters
    symbols = ["BTC", "ETH", "XRP"]
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Fetch cryptocurrency data
    print("\nFetching cryptocurrency data...")
    crypto_data = crypto_source.fetch_historical_data(symbols, start_date=start_date, end_date=end_date)
    
    # Get market cap data
    print("\nFetching market cap data...")
    market_caps = crypto_source.get_market_cap_data(symbols)
    print(f"Market caps: {market_caps}")
    
    # Check data quality
    if "BTC" in crypto_data:
        print("\nChecking data quality for BTC data...")
        checker = DataQualityChecker()
        quality_report = checker.generate_report(crypto_data["BTC"])
        
        # Visualize data quality
        checker.visualize_data_quality(crypto_data["BTC"], output_file="data/quality_BTC.png")


def demo_forex():
    """
    Demonstrate fetching forex data.
    """
    print("\n" + "="*80)
    print("FOREX DATA DEMO")
    print("="*80)
    
    # Load API keys
    api_keys = load_api_keys()
    
    if not api_keys["alpha_vantage"]:
        print("Alpha Vantage API key not found. Skipping Forex demo.")
        return
    
    # Initialize data sources
    forex_source = ForexDataSource(
        api_key=api_keys["alpha_vantage"],
        provider="alpha_vantage",
        save_path="data/forex/"
    )
    
    # Set up parameters
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Fetch forex data
    print("\nFetching forex data...")
    forex_data = forex_source.fetch_historical_data(symbols, start_date=start_date, end_date=end_date)
    
    # Get latest rates
    print("\nGetting latest forex rates...")
    latest_rates = forex_source.get_latest_rates("EUR", ["USD", "GBP", "JPY"])
    print(f"Latest rates: {latest_rates}")
    
    # Check data quality
    if "EURUSD" in forex_data:
        print("\nChecking data quality for EURUSD data...")
        checker = DataQualityChecker()
        quality_report = checker.generate_report(forex_data["EURUSD"])
        
        # Visualize data quality
        checker.visualize_data_quality(forex_data["EURUSD"], output_file="data/quality_EURUSD.png")


def demo_real_time():
    """
    Demonstrate real-time data streaming.
    """
    print("\n" + "="*80)
    print("REAL-TIME DATA STREAMING DEMO")
    print("="*80)
    
    # Load API keys
    api_keys = load_api_keys()
    
    if not api_keys["alpha_vantage"]:
        print("Alpha Vantage API key not found. Skipping real-time demo.")
        return
    
    # Define callback function to handle incoming data
    def on_data(data):
        print(f"Received data: {data}")
    
    # Initialize data stream manager
    manager = DataStreamManager()
    
    # Add Alpha Vantage real-time stream (simulated with polling)
    alpha_vantage_stream = AlphaVantageRealTime(
        api_key=api_keys["alpha_vantage"],
        symbols=["AAPL", "MSFT"],
        interval=60,  # Poll every 60 seconds
        on_data=on_data
    )
    manager.add_stream("alpha_vantage", alpha_vantage_stream)
    
    # Add cryptocurrency real-time stream
    crypto_stream = CryptoRealTimeStream(
        symbols=["btcusdt", "ethusdt"],
        on_data=on_data
    )
    manager.add_stream("crypto", crypto_stream)
    
    # Start all streams
    print("\nStarting real-time data streams. Press Ctrl+C to stop.")
    try:
        manager.start_all()
        
        # Run for a short time as demo
        import time
        time.sleep(120)  # Run for 2 minutes
        
    except KeyboardInterrupt:
        print("\nStopping streams...")
    finally:
        manager.stop_all()
        print("All streams stopped.")


def main():
    """
    Main function to run the demos.
    """
    parser = argparse.ArgumentParser(description="Data sources demo script")
    parser.add_argument("--demo", choices=["stocks", "crypto", "forex", "real-time", "all"], 
                      default="all", help="Which demo to run")
    
    args = parser.parse_args()
    
    # Create data directories if they don't exist
    os.makedirs("data/yahoo", exist_ok=True)
    os.makedirs("data/alpha_vantage", exist_ok=True)
    os.makedirs("data/crypto", exist_ok=True)
    os.makedirs("data/forex", exist_ok=True)
    
    if args.demo in ["stocks", "all"]:
        demo_stocks()
    
    if args.demo in ["crypto", "all"]:
        demo_crypto()
    
    if args.demo in ["forex", "all"]:
        demo_forex()
    
    if args.demo in ["real-time", "all"]:
        demo_real_time()


if __name__ == "__main__":
    main() 