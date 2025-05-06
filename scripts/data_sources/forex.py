"""
Forex data source implementation.
"""
import pandas as pd
import requests
import json
from typing import List, Dict, Any, Optional
import os
import time
from datetime import datetime, timedelta
from .base import DataSource


class ForexDataSource(DataSource):
    """
    Forex data source implementation.
    Fetches forex data from various APIs.
    """
    
    def __init__(self, api_key: str = None, provider: str = "alpha_vantage", save_path: str = "data/"):
        """
        Initialize the forex data source.
        
        Args:
            api_key: API key (required for most providers)
            provider: Data provider ("alpha_vantage", "oanda", "fixer")
            save_path: Directory to save downloaded data
        """
        super().__init__(save_path)
        self.api_key = api_key
        self.provider = provider.lower()
        self.request_count = 0
        self.last_request_time = None
        
        # Set up provider-specific configurations
        if self.provider == "alpha_vantage":
            self.base_url = "https://www.alphavantage.co/query"
        elif self.provider == "oanda":
            self.base_url = "https://api-fxtrade.oanda.com/v3"
        elif self.provider == "fixer":
            self.base_url = "http://data.fixer.io/api"
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def fetch_historical_data(self, 
                             symbols: List[str], 
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             period: Optional[str] = None,
                             interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical forex data.
        
        Args:
            symbols: List of forex pairs (e.g. ["EURUSD", "GBPUSD"])
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            period: Time period as string (e.g. "1y" for 1 year)
            interval: Data interval (e.g. "1d" for daily, "1h" for hourly)
            
        Returns:
            Dictionary mapping symbols to their respective DataFrames
        """
        result = {}
        
        # Convert dates to timestamps if needed
        start_timestamp = None
        end_timestamp = None
        
        if start_date:
            start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        if end_date:
            end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        
        # If period is provided but not start_date, calculate start_date
        if not start_date and period:
            end_dt = datetime.now() if not end_date else datetime.strptime(end_date, "%Y-%m-%d")
            if period.endswith("d"):
                days = int(period[:-1])
                start_dt = end_dt - timedelta(days=days)
            elif period.endswith("w"):
                weeks = int(period[:-1])
                start_dt = end_dt - timedelta(weeks=weeks)
            elif period.endswith("m"):
                months = int(period[:-1])
                start_dt = end_dt - timedelta(days=months*30)
            elif period.endswith("y"):
                years = int(period[:-1])
                start_dt = end_dt - timedelta(days=years*365)
            else:
                raise ValueError(f"Unsupported period format: {period}")
            
            start_timestamp = int(start_dt.timestamp())
            start_date = start_dt.strftime("%Y-%m-%d")
        
        # Use provider-specific implementation
        for symbol in symbols:
            print(f"Downloading {symbol} data from {self.provider}...")
            
            try:
                if self.provider == "alpha_vantage":
                    df = self._fetch_from_alpha_vantage(symbol, interval)
                elif self.provider == "oanda":
                    df = self._fetch_from_oanda(symbol, start_date, end_date, interval)
                elif self.provider == "fixer":
                    df = self._fetch_from_fixer(symbol, start_date, end_date)
                
                if df is not None and not df.empty:
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
                else:
                    print(f"No data returned for {symbol}")
            
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
        
        return result
    
    def _fetch_from_alpha_vantage(self, symbol: str, interval: str) -> pd.DataFrame:
        """
        Fetch forex data from Alpha Vantage API.
        
        Args:
            symbol: Forex pair (e.g. "EURUSD")
            interval: Data interval
            
        Returns:
            DataFrame with OHLCV data
        """
        # Apply rate limiting
        self._rate_limit()
        
        # Check if API key is provided
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")
        
        # Split the symbol into from_currency and to_currency
        if len(symbol) != 6:
            raise ValueError(f"Invalid forex symbol format: {symbol}. Expected format: 'EURUSD'")
        
        from_currency = symbol[:3]
        to_currency = symbol[3:]
        
        # Map interval to Alpha Vantage format
        if interval == "1d":
            function = "FX_DAILY"
            time_series_key = "Time Series FX (Daily)"
        elif interval == "1w":
            function = "FX_WEEKLY"
            time_series_key = "Time Series FX (Weekly)"
        elif interval == "1m":
            function = "FX_MONTHLY"
            time_series_key = "Time Series FX (Monthly)"
        else:
            # For intraday data
            function = "FX_INTRADAY"
            # Map interval to Alpha Vantage format
            av_interval = interval
            if interval == "1h":
                av_interval = "60min"
            elif interval == "30m":
                av_interval = "30min"
            elif interval == "15m":
                av_interval = "15min"
            elif interval == "5m":
                av_interval = "5min"
            elif interval == "1m":
                av_interval = "1min"
            
            time_series_key = f"Time Series FX ({av_interval})"
        
        # Prepare API URL
        params = {
            "function": function,
            "from_symbol": from_currency,
            "to_symbol": to_currency,
            "apikey": self.api_key,
            "outputsize": "full"
        }
        
        # Add interval parameter for intraday data
        if function == "FX_INTRADAY":
            params["interval"] = av_interval
        
        # Make API request
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        # Check for errors
        if "Error Message" in data:
            print(f"Error from Alpha Vantage: {data['Error Message']}")
            return None
        
        if "Note" in data:
            print(f"API limit reached: {data['Note']}")
            return None
        
        # Parse the response
        if time_series_key not in data:
            print(f"Unexpected response format: {data.keys()}")
            return None
        
        time_series = data[time_series_key]
        
        # Create DataFrame
        df = pd.DataFrame.from_dict(time_series, orient="index")
        
        # Rename columns to standard format
        column_mapping = {
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close"
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Convert data types
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col])
        
        # Add a Volume column (forex doesn't typically have volume)
        df["Volume"] = 0
        
        # Set index to datetime
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        return df
    
    def _fetch_from_oanda(self, symbol: str, start_date: Optional[str], 
                         end_date: Optional[str], interval: str) -> pd.DataFrame:
        """
        Fetch forex data from OANDA API.
        
        Args:
            symbol: Forex pair (e.g. "EUR_USD")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval
            
        Returns:
            DataFrame with OHLCV data
        """
        # Apply rate limiting
        self._rate_limit()
        
        # Check if API key is provided
        if not self.api_key:
            raise ValueError("OANDA API key is required")
        
        # Format symbol for OANDA (replace / with _)
        symbol = symbol.replace("/", "_")
        if len(symbol) == 6:
            # Insert underscore if not present (e.g., "EURUSD" -> "EUR_USD")
            symbol = f"{symbol[:3]}_{symbol[3:]}"
        
        # Map interval to OANDA format
        if interval == "1d":
            granularity = "D"
        elif interval == "1h":
            granularity = "H1"
        elif interval == "30m":
            granularity = "M30"
        elif interval == "15m":
            granularity = "M15"
        elif interval == "5m":
            granularity = "M5"
        elif interval == "1m":
            granularity = "M1"
        else:
            granularity = "D"  # Default to daily
        
        # Prepare API URL
        url = f"{self.base_url}/instruments/{symbol}/candles"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        params = {
            "granularity": granularity,
            "price": "M"  # Midpoint candles
        }
        
        # Add date range parameters if provided
        if start_date:
            params["from"] = f"{start_date}T00:00:00Z"
        if end_date:
            params["to"] = f"{end_date}T23:59:59Z"
        
        # Make API request
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        
        # Check for errors
        if "errorMessage" in data:
            print(f"Error from OANDA: {data['errorMessage']}")
            return None
        
        # Parse the response
        if "candles" not in data:
            print(f"Unexpected response format: {data.keys()}")
            return None
        
        candles = data["candles"]
        
        # Create DataFrame
        records = []
        for candle in candles:
            if candle["complete"]:  # Only use complete candles
                records.append({
                    "time": candle["time"],
                    "Open": float(candle["mid"]["o"]),
                    "High": float(candle["mid"]["h"]),
                    "Low": float(candle["mid"]["l"]),
                    "Close": float(candle["mid"]["c"]),
                    "Volume": 0  # OANDA doesn't provide volume
                })
        
        df = pd.DataFrame(records)
        
        # Set index to datetime
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def _fetch_from_fixer(self, symbol: str, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """
        Fetch forex data from Fixer API.
        Note: Fixer only provides daily data.
        
        Args:
            symbol: Forex pair (e.g. "EURUSD")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with OHLCV data
        """
        # Apply rate limiting
        self._rate_limit()
        
        # Check if API key is provided
        if not self.api_key:
            raise ValueError("Fixer API key is required")
        
        # Split the symbol into base and quote currencies
        if len(symbol) != 6:
            raise ValueError(f"Invalid forex symbol format: {symbol}. Expected format: 'EURUSD'")
        
        base_currency = symbol[:3]
        quote_currency = symbol[3:]
        
        # Prepare date range
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if not start_date:
            # Default to 1 year of data
            start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)
            start_date = start_dt.strftime("%Y-%m-%d")
        
        # Create date range
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        date_range = [start_dt + timedelta(days=x) for x in range((end_dt - start_dt).days + 1)]
        
        # Collect data for each date
        records = []
        
        for date in date_range:
            date_str = date.strftime("%Y-%m-%d")
            
            # Prepare API URL
            url = f"{self.base_url}/{date_str}"
            
            params = {
                "access_key": self.api_key,
                "base": base_currency,
                "symbols": quote_currency
            }
            
            # Make API request
            response = requests.get(url, params=params)
            data = response.json()
            
            # Check for errors
            if not data.get("success", False):
                print(f"Error from Fixer for {date_str}: {data.get('error', {}).get('info', 'Unknown error')}")
                continue
            
            # Parse the response
            if "rates" not in data or quote_currency not in data["rates"]:
                print(f"Unexpected response format for {date_str}: {data.keys()}")
                continue
            
            rate = data["rates"][quote_currency]
            
            # Add to records
            records.append({
                "date": date_str,
                "rate": rate
            })
            
            # Fixer has strict rate limits, so add a small delay
            time.sleep(0.5)
        
        # Create DataFrame
        if not records:
            print(f"No data collected for {symbol}")
            return None
        
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        
        # Fixer only provides closing rates, so we'll use the same value for OHLC
        df["Close"] = df["rate"]
        df["Open"] = df["rate"]
        df["High"] = df["rate"]
        df["Low"] = df["rate"]
        df["Volume"] = 0
        
        # Drop the original rate column
        df.drop(columns=["rate"], inplace=True)
        
        return df
    
    def get_supported_assets(self) -> List[str]:
        """
        Get list of asset types supported by this data source.
        
        Returns:
            List of supported asset types
        """
        return ["forex"]
    
    def get_api_usage_info(self) -> Dict[str, Any]:
        """
        Get information about API usage limits and current status.
        
        Returns:
            Dictionary with API usage information
        """
        limits = {
            "alpha_vantage": {
                "free_tier": "5 API calls per minute, 500 per day",
                "premium": "Available with paid plans"
            },
            "oanda": {
                "free_tier": "None (requires account)",
                "premium": "Available with OANDA account"
            },
            "fixer": {
                "free_tier": "100 API calls per month",
                "premium": "Available with paid plans"
            }
        }
        
        return {
            "provider": self.provider,
            "requests_made": self.request_count,
            "last_request": self.last_request_time.isoformat() if self.last_request_time else None,
            "limits": limits.get(self.provider, "Unknown")
        }
    
    def _rate_limit(self):
        """
        Apply rate limiting to avoid hitting API limits.
        """
        now = datetime.now()
        
        if self.last_request_time:
            # Define delay based on provider
            if self.provider == "alpha_vantage":
                # Alpha Vantage: 5 calls per minute (free tier)
                min_delay = 12.0  # seconds
            elif self.provider == "oanda":
                # OANDA: No strict rate limit, but be nice
                min_delay = 0.5  # seconds
            elif self.provider == "fixer":
                # Fixer: Strict limits on free tier
                min_delay = 1.0  # seconds
            else:
                min_delay = 1.0  # Default
            
            elapsed = (now - self.last_request_time).total_seconds()
            if elapsed < min_delay:
                time.sleep(min_delay - elapsed)
        
        self.request_count += 1
        self.last_request_time = datetime.now()
    
    def get_latest_rates(self, base_currency: str, quote_currencies: List[str]) -> Dict[str, float]:
        """
        Get latest exchange rates for a base currency.
        
        Args:
            base_currency: Base currency code (e.g. "EUR")
            quote_currencies: List of quote currency codes (e.g. ["USD", "GBP"])
            
        Returns:
            Dictionary mapping currency pairs to their rates
        """
        result = {}
        
        if self.provider == "alpha_vantage":
            # Alpha Vantage doesn't have a dedicated endpoint for latest rates
            # We'll use the FX_DAILY endpoint and get the latest data point
            for quote_currency in quote_currencies:
                symbol = f"{base_currency}{quote_currency}"
                df = self._fetch_from_alpha_vantage(symbol, "1d")
                if df is not None and not df.empty:
                    latest_rate = df["Close"].iloc[-1]
                    result[symbol] = latest_rate
        
        elif self.provider == "oanda":
            # Format currencies for OANDA
            formatted_pairs = []
            for quote_currency in quote_currencies:
                formatted_pairs.append(f"{base_currency}_{quote_currency}")
            
            # Prepare API URL
            url = f"{self.base_url}/accounts/{{account_id}}/pricing"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            params = {
                "instruments": ",".join(formatted_pairs)
            }
            
            # Make API request
            # Note: This requires an OANDA account ID, which we don't have
            # This is a simplified example
            print("OANDA requires an account ID for latest rates")
            
        elif self.provider == "fixer":
            # Apply rate limiting
            self._rate_limit()
            
            # Prepare API URL
            url = f"{self.base_url}/latest"
            
            params = {
                "access_key": self.api_key,
                "base": base_currency,
                "symbols": ",".join(quote_currencies)
            }
            
            # Make API request
            response = requests.get(url, params=params)
            data = response.json()
            
            # Check for errors
            if not data.get("success", False):
                print(f"Error from Fixer: {data.get('error', {}).get('info', 'Unknown error')}")
                return result
            
            # Parse the response
            if "rates" not in data:
                print(f"Unexpected response format: {data.keys()}")
                return result
            
            # Add rates to result
            for quote_currency, rate in data["rates"].items():
                symbol = f"{base_currency}{quote_currency}"
                result[symbol] = rate
        
        return result
    
    def get_currency_list(self) -> List[Dict[str, str]]:
        """
        Get list of available currencies.
        
        Returns:
            List of dictionaries with currency information
        """
        if self.provider == "alpha_vantage":
            # Alpha Vantage doesn't have a dedicated endpoint for currency list
            # Return a predefined list of common currencies
            return [
                {"code": "USD", "name": "US Dollar"},
                {"code": "EUR", "name": "Euro"},
                {"code": "GBP", "name": "British Pound"},
                {"code": "JPY", "name": "Japanese Yen"},
                {"code": "AUD", "name": "Australian Dollar"},
                {"code": "CAD", "name": "Canadian Dollar"},
                {"code": "CHF", "name": "Swiss Franc"},
                {"code": "CNY", "name": "Chinese Yuan"},
                {"code": "NZD", "name": "New Zealand Dollar"},
                {"code": "SEK", "name": "Swedish Krona"}
            ]
        
        elif self.provider == "fixer":
            # Apply rate limiting
            self._rate_limit()
            
            # Prepare API URL
            url = f"{self.base_url}/symbols"
            
            params = {
                "access_key": self.api_key
            }
            
            # Make API request
            response = requests.get(url, params=params)
            data = response.json()
            
            # Check for errors
            if not data.get("success", False):
                print(f"Error from Fixer: {data.get('error', {}).get('info', 'Unknown error')}")
                return []
            
            # Parse the response
            if "symbols" not in data:
                print(f"Unexpected response format: {data.keys()}")
                return []
            
            # Convert to list of dictionaries
            currencies = []
            for code, name in data["symbols"].items():
                currencies.append({
                    "code": code,
                    "name": name
                })
            
            return currencies
        
        return [] 