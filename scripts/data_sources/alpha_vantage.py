"""
Alpha Vantage data source implementation.
"""
import pandas as pd
import requests
from typing import List, Dict, Any, Optional
import os
import time
from datetime import datetime, timedelta
import json
from .base import DataSource


class AlphaVantageDataSource(DataSource):
    """
    Alpha Vantage data source implementation.
    Fetches financial data from the Alpha Vantage API.
    """
    
    def __init__(self, api_key: str, save_path: str = "data/"):
        """
        Initialize the Alpha Vantage data source.
        
        Args:
            api_key: Alpha Vantage API key
            save_path: Directory to save downloaded data
        """
        super().__init__(save_path)
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.request_count = 0
        self.last_request_time = None
        
        # Alpha Vantage has a limit of 5 API requests per minute for free tier
        self.rate_limit = 5
        self.rate_limit_period = 60  # seconds
        self.requests_this_period = 0
        self.period_start_time = datetime.now()
    
    def fetch_historical_data(self, 
                             symbols: List[str], 
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             period: Optional[str] = None,
                             interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical price data from Alpha Vantage.
        
        Args:
            symbols: List of ticker symbols to fetch
            start_date: Start date in YYYY-MM-DD format (not used directly with Alpha Vantage)
            end_date: End date in YYYY-MM-DD format (not used directly with Alpha Vantage)
            period: Time period as string (not used directly with Alpha Vantage)
            interval: Data interval ("1d" for daily, "60min" for hourly, etc.)
            
        Returns:
            Dictionary mapping symbols to their respective DataFrames
        """
        result = {}
        
        # Map interval to Alpha Vantage format
        av_interval = self._map_interval(interval)
        
        for symbol in symbols:
            print(f"Downloading data for {symbol} from Alpha Vantage...")
            
            # Apply rate limiting
            self._rate_limit()
            
            # Determine function based on interval
            if av_interval == "daily":
                function = "TIME_SERIES_DAILY"
                time_series_key = "Time Series (Daily)"
            elif av_interval == "weekly":
                function = "TIME_SERIES_WEEKLY"
                time_series_key = "Weekly Time Series"
            elif av_interval == "monthly":
                function = "TIME_SERIES_MONTHLY"
                time_series_key = "Monthly Time Series"
            else:
                function = "TIME_SERIES_INTRADAY"
                time_series_key = f"Time Series ({av_interval})"
            
            # Prepare request parameters
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": "full"
            }
            
            # Add interval parameter for intraday data
            if function == "TIME_SERIES_INTRADAY":
                params["interval"] = av_interval
            
            # Make API request
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            # Check for error messages
            if "Error Message" in data:
                print(f"Error fetching data for {symbol}: {data['Error Message']}")
                continue
            
            if "Note" in data:
                print(f"API limit reached: {data['Note']}")
                # Wait and try again later
                time.sleep(60)
                continue
            
            # Parse the response
            if time_series_key not in data:
                print(f"Unexpected response format for {symbol}: {data.keys()}")
                continue
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            
            # Rename columns to standard format
            column_mapping = {
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume"
            }
            df.rename(columns=column_mapping, inplace=True)
            
            # Convert data types
            for col in ["Open", "High", "Low", "Close"]:
                df[col] = pd.to_numeric(df[col])
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype(int)
            
            # Set index to datetime
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Filter by date range if provided
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            
            # Verify data quality
            quality_metrics = self.verify_data_quality(df)
            print(f"Data quality metrics for {symbol}: {quality_metrics}")
            
            # Save data
            csv_path = self.save_data(symbol, df)
            print(f"Saved {symbol} data to {csv_path}")
            
            result[symbol] = df
        
        return result
    
    def get_supported_assets(self) -> List[str]:
        """
        Get list of asset types supported by Alpha Vantage.
        
        Returns:
            List of supported asset types
        """
        return ["stocks", "etfs", "forex", "cryptocurrencies", "commodities", "economic_indicators"]
    
    def get_api_usage_info(self) -> Dict[str, Any]:
        """
        Get information about API usage limits and current status.
        
        Returns:
            Dictionary with API usage information
        """
        return {
            "provider": "Alpha Vantage",
            "requests_made": self.request_count,
            "last_request": self.last_request_time.isoformat() if self.last_request_time else None,
            "limits": {
                "free_tier": "5 API calls per minute, 500 per day",
                "premium": "Available with paid plans"
            }
        }
    
    def _rate_limit(self):
        """
        Apply rate limiting to comply with Alpha Vantage API limits.
        """
        now = datetime.now()
        
        # Check if we're in a new rate limit period
        if (now - self.period_start_time).total_seconds() > self.rate_limit_period:
            self.period_start_time = now
            self.requests_this_period = 0
        
        # Check if we've hit the rate limit
        if self.requests_this_period >= self.rate_limit:
            # Calculate time to wait
            elapsed = (now - self.period_start_time).total_seconds()
            wait_time = self.rate_limit_period - elapsed + 1  # Add 1 second buffer
            
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self.period_start_time = datetime.now()
                self.requests_this_period = 0
        
        self.requests_this_period += 1
        self.request_count += 1
        self.last_request_time = datetime.now()
    
    def _map_interval(self, interval: str) -> str:
        """
        Map standard interval format to Alpha Vantage format.
        
        Args:
            interval: Standard interval format (e.g. "1d", "1h", "15m")
            
        Returns:
            Alpha Vantage interval format
        """
        interval_map = {
            "1d": "daily",
            "1w": "weekly",
            "1mo": "monthly",
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "60min",
        }
        
        return interval_map.get(interval, "daily")
    
    def get_forex_data(self, from_symbol: str, to_symbol: str, interval: str = "1d") -> pd.DataFrame:
        """
        Fetch forex exchange rate data.
        
        Args:
            from_symbol: From currency symbol (e.g. "USD")
            to_symbol: To currency symbol (e.g. "EUR")
            interval: Data interval
            
        Returns:
            DataFrame with exchange rate data
        """
        # Apply rate limiting
        self._rate_limit()
        
        # Map interval to Alpha Vantage format
        av_interval = self._map_interval(interval)
        
        # Determine function based on interval
        if av_interval == "daily":
            function = "FX_DAILY"
            time_series_key = "Time Series FX (Daily)"
        elif av_interval == "weekly":
            function = "FX_WEEKLY"
            time_series_key = "Time Series FX (Weekly)"
        elif av_interval == "monthly":
            function = "FX_MONTHLY"
            time_series_key = "Time Series FX (Monthly)"
        else:
            function = "FX_INTRADAY"
            time_series_key = f"Time Series FX ({av_interval})"
        
        # Prepare request parameters
        params = {
            "function": function,
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "apikey": self.api_key,
            "outputsize": "full"
        }
        
        # Add interval parameter for intraday data
        if function == "FX_INTRADAY":
            params["interval"] = av_interval
        
        # Make API request
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        # Check for error messages
        if "Error Message" in data:
            raise ValueError(f"Error fetching forex data: {data['Error Message']}")
        
        if "Note" in data:
            print(f"API limit reached: {data['Note']}")
            # Wait and try again later
            time.sleep(60)
            return self.get_forex_data(from_symbol, to_symbol, interval)
        
        # Parse the response
        if time_series_key not in data:
            raise ValueError(f"Unexpected response format: {data.keys()}")
        
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient="index")
        
        # Rename columns to standard format
        column_mapping = {
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close"
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Add Volume column (not provided for forex)
        df["Volume"] = 0
        
        # Convert data types
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col])
        
        # Set index to datetime
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        # Save data
        symbol = f"{from_symbol}{to_symbol}"
        csv_path = self.save_data(symbol, df)
        print(f"Saved {symbol} forex data to {csv_path}")
        
        return df
    
    def get_crypto_data(self, symbol: str, market: str = "USD", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch cryptocurrency data.
        
        Args:
            symbol: Cryptocurrency symbol (e.g. "BTC")
            market: Market currency (e.g. "USD")
            interval: Data interval
            
        Returns:
            DataFrame with cryptocurrency data
        """
        # Apply rate limiting
        self._rate_limit()
        
        # Map interval to Alpha Vantage format
        av_interval = self._map_interval(interval)
        
        # Determine function based on interval
        if av_interval == "daily":
            function = "DIGITAL_CURRENCY_DAILY"
            time_series_key = "Time Series (Digital Currency Daily)"
        elif av_interval == "weekly":
            function = "DIGITAL_CURRENCY_WEEKLY"
            time_series_key = "Time Series (Digital Currency Weekly)"
        elif av_interval == "monthly":
            function = "DIGITAL_CURRENCY_MONTHLY"
            time_series_key = "Time Series (Digital Currency Monthly)"
        else:
            raise ValueError(f"Interval {interval} not supported for crypto data")
        
        # Prepare request parameters
        params = {
            "function": function,
            "symbol": symbol,
            "market": market,
            "apikey": self.api_key
        }
        
        # Make API request
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        # Check for error messages
        if "Error Message" in data:
            raise ValueError(f"Error fetching crypto data: {data['Error Message']}")
        
        if "Note" in data:
            print(f"API limit reached: {data['Note']}")
            # Wait and try again later
            time.sleep(60)
            return self.get_crypto_data(symbol, market, interval)
        
        # Parse the response
        if time_series_key not in data:
            raise ValueError(f"Unexpected response format: {data.keys()}")
        
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient="index")
        
        # Rename columns to standard format
        column_mapping = {
            f"1a. open ({market})": "Open",
            f"2a. high ({market})": "High",
            f"3a. low ({market})": "Low",
            f"4a. close ({market})": "Close",
            f"5. volume": "Volume"
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Keep only the columns we need
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        
        # Convert data types
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col])
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype(int)
        
        # Set index to datetime
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        # Save data
        crypto_symbol = f"{symbol}-{market}"
        csv_path = self.save_data(crypto_symbol, df)
        print(f"Saved {crypto_symbol} crypto data to {csv_path}")
        
        return df 