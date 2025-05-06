"""
Cryptocurrency data source implementation.
"""
import pandas as pd
import requests
import json
from typing import List, Dict, Any, Optional
import os
import time
from datetime import datetime, timedelta
from .base import DataSource


class CryptoDataSource(DataSource):
    """
    Cryptocurrency data source implementation.
    Fetches cryptocurrency data from various APIs.
    """
    
    def __init__(self, api_key: str = None, provider: str = "coingecko", save_path: str = "data/"):
        """
        Initialize the cryptocurrency data source.
        
        Args:
            api_key: API key (required for some providers)
            provider: Data provider ("coingecko", "cryptocompare", "binance")
            save_path: Directory to save downloaded data
        """
        super().__init__(save_path)
        self.api_key = api_key
        self.provider = provider.lower()
        self.request_count = 0
        self.last_request_time = None
        
        # Set up provider-specific configurations
        if self.provider == "coingecko":
            self.base_url = "https://api.coingecko.com/api/v3"
        elif self.provider == "cryptocompare":
            self.base_url = "https://min-api.cryptocompare.com/data"
        elif self.provider == "binance":
            self.base_url = "https://api.binance.com/api/v3"
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def fetch_historical_data(self, 
                             symbols: List[str], 
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             period: Optional[str] = None,
                             interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical cryptocurrency price data.
        
        Args:
            symbols: List of cryptocurrency symbols (e.g. ["BTC", "ETH"])
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
        
        # Use provider-specific implementation
        for symbol in symbols:
            print(f"Downloading {symbol} data from {self.provider}...")
            
            try:
                if self.provider == "coingecko":
                    df = self._fetch_from_coingecko(symbol, start_timestamp, end_timestamp, interval)
                elif self.provider == "cryptocompare":
                    df = self._fetch_from_cryptocompare(symbol, start_timestamp, end_timestamp, interval)
                elif self.provider == "binance":
                    df = self._fetch_from_binance(symbol, start_timestamp, end_timestamp, interval)
                
                if df is not None and not df.empty:
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
    
    def _fetch_from_coingecko(self, symbol: str, start_timestamp: Optional[int], 
                             end_timestamp: Optional[int], interval: str) -> pd.DataFrame:
        """
        Fetch data from CoinGecko API.
        
        Args:
            symbol: Cryptocurrency symbol (e.g. "bitcoin", "ethereum")
            start_timestamp: Start timestamp
            end_timestamp: End timestamp
            interval: Data interval
            
        Returns:
            DataFrame with OHLCV data
        """
        # Apply rate limiting
        self._rate_limit()
        
        # Map common ticker symbols to CoinGecko IDs
        symbol_mapping = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "XRP": "ripple",
            "LTC": "litecoin",
            "BCH": "bitcoin-cash",
            "ADA": "cardano",
            "DOT": "polkadot",
            "LINK": "chainlink",
            "BNB": "binancecoin",
            "XLM": "stellar"
        }
        
        # Convert symbol to CoinGecko ID if needed
        coin_id = symbol_mapping.get(symbol.upper(), symbol.lower())
        
        # Map interval to CoinGecko format
        if interval == "1d":
            days = "max"  # CoinGecko uses "max" for daily data
        elif interval == "1h":
            days = 90  # CoinGecko limits hourly data to 90 days
        else:
            days = "max"  # Default to daily
        
        # Prepare API URL
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "daily" if interval == "1d" else interval
        }
        
        # Make API request
        response = requests.get(url, params=params)
        data = response.json()
        
        # Check for errors
        if "error" in data:
            print(f"Error from CoinGecko: {data['error']}")
            return None
        
        # Parse the response
        if "prices" not in data or "market_caps" not in data or "total_volumes" not in data:
            print(f"Unexpected response format: {data.keys()}")
            return None
        
        # Create DataFrame from price data
        df_prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        df_prices["timestamp"] = pd.to_datetime(df_prices["timestamp"], unit="ms")
        df_prices.set_index("timestamp", inplace=True)
        
        # Add volume data
        df_volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
        df_volumes["timestamp"] = pd.to_datetime(df_volumes["timestamp"], unit="ms")
        df_volumes.set_index("timestamp", inplace=True)
        
        # Combine price and volume data
        df = pd.DataFrame({
            "Close": df_prices["price"],
            "Volume": df_volumes["volume"]
        })
        
        # CoinGecko doesn't provide OHLC data directly, so we need to estimate
        # We'll use the close price for open, high, and low as an approximation
        df["Open"] = df["Close"]
        df["High"] = df["Close"]
        df["Low"] = df["Close"]
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Filter by date range if provided
        if start_timestamp:
            start_date = pd.to_datetime(start_timestamp, unit="s")
            df = df[df.index >= start_date]
        if end_timestamp:
            end_date = pd.to_datetime(end_timestamp, unit="s")
            df = df[df.index <= end_date]
        
        return df
    
    def _fetch_from_cryptocompare(self, symbol: str, start_timestamp: Optional[int], 
                                 end_timestamp: Optional[int], interval: str) -> pd.DataFrame:
        """
        Fetch data from CryptoCompare API.
        
        Args:
            symbol: Cryptocurrency symbol (e.g. "BTC", "ETH")
            start_timestamp: Start timestamp
            end_timestamp: End timestamp
            interval: Data interval
            
        Returns:
            DataFrame with OHLCV data
        """
        # Apply rate limiting
        self._rate_limit()
        
        # Map interval to CryptoCompare format
        if interval == "1d":
            endpoint = "histoday"
        elif interval == "1h":
            endpoint = "histohour"
        elif interval == "1m":
            endpoint = "histominute"
        else:
            endpoint = "histoday"  # Default to daily
        
        # Prepare API URL
        url = f"{self.base_url}/{endpoint}"
        
        params = {
            "fsym": symbol.upper(),
            "tsym": "USD",
            "limit": 2000,  # Maximum allowed by API
            "aggregate": 1
        }
        
        # Add API key if available
        if self.api_key:
            params["api_key"] = self.api_key
        
        # Add timestamp parameters if provided
        if start_timestamp:
            params["toTs"] = end_timestamp if end_timestamp else int(datetime.now().timestamp())
            
            # CryptoCompare requires a limit rather than a start timestamp
            # We'll estimate the number of data points needed
            if interval == "1d":
                seconds_per_unit = 86400
            elif interval == "1h":
                seconds_per_unit = 3600
            else:
                seconds_per_unit = 60
            
            time_diff = params["toTs"] - start_timestamp
            limit = min(2000, time_diff // seconds_per_unit)
            params["limit"] = limit
        
        # Make API request
        response = requests.get(url, params=params)
        data = response.json()
        
        # Check for errors
        if "Response" in data and data["Response"] == "Error":
            print(f"Error from CryptoCompare: {data['Message']}")
            return None
        
        # Parse the response
        if "Data" not in data:
            print(f"Unexpected response format: {data.keys()}")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(data["Data"])
        
        # Check if data is empty
        if df.empty or "time" not in df.columns:
            print(f"No data returned for {symbol}")
            return None
        
        # Convert timestamp to datetime
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        
        # Rename columns to standard format
        column_mapping = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volumefrom": "Volume"
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Select only the columns we need
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        
        # Sort by date
        df.sort_index(inplace=True)
        
        return df
    
    def _fetch_from_binance(self, symbol: str, start_timestamp: Optional[int], 
                           end_timestamp: Optional[int], interval: str) -> pd.DataFrame:
        """
        Fetch data from Binance API.
        
        Args:
            symbol: Cryptocurrency symbol (e.g. "BTCUSDT", "ETHUSDT")
            start_timestamp: Start timestamp in milliseconds
            end_timestamp: End timestamp in milliseconds
            interval: Data interval
            
        Returns:
            DataFrame with OHLCV data
        """
        # Apply rate limiting
        self._rate_limit()
        
        # Add USDT suffix if not present
        if not symbol.upper().endswith("USDT"):
            symbol = f"{symbol.upper()}USDT"
        
        # Map interval to Binance format
        interval_mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "2h": "2h",
            "4h": "4h",
            "6h": "6h",
            "8h": "8h",
            "12h": "12h",
            "1d": "1d",
            "3d": "3d",
            "1w": "1w",
            "1M": "1M"
        }
        
        binance_interval = interval_mapping.get(interval, "1d")
        
        # Prepare API URL
        url = f"{self.base_url}/klines"
        
        params = {
            "symbol": symbol,
            "interval": binance_interval,
            "limit": 1000  # Maximum allowed by API
        }
        
        # Convert timestamps from seconds to milliseconds if provided
        if start_timestamp:
            params["startTime"] = start_timestamp * 1000
        if end_timestamp:
            params["endTime"] = end_timestamp * 1000
        
        # Make API request
        response = requests.get(url, params=params)
        data = response.json()
        
        # Check for errors
        if isinstance(data, dict) and "code" in data:
            print(f"Error from Binance: {data['msg']}")
            return None
        
        # Create DataFrame
        columns = ["timestamp", "Open", "High", "Low", "Close", "Volume", 
                   "close_time", "quote_asset_volume", "number_of_trades", 
                   "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
        df = pd.DataFrame(data, columns=columns)
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        # Convert data types
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col])
        
        # Select only the columns we need
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        
        # Sort by date
        df.sort_index(inplace=True)
        
        return df
    
    def get_supported_assets(self) -> List[str]:
        """
        Get list of asset types supported by this data source.
        
        Returns:
            List of supported asset types
        """
        return ["cryptocurrencies"]
    
    def get_api_usage_info(self) -> Dict[str, Any]:
        """
        Get information about API usage limits and current status.
        
        Returns:
            Dictionary with API usage information
        """
        limits = {
            "coingecko": {
                "free_tier": "50 calls per minute",
                "premium": "Available with paid plans"
            },
            "cryptocompare": {
                "free_tier": "100,000 calls per month",
                "premium": "Available with paid plans"
            },
            "binance": {
                "free_tier": "1200 requests per minute",
                "premium": "N/A"
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
            if self.provider == "coingecko":
                # CoinGecko: 50 calls per minute (free tier)
                min_delay = 1.2  # seconds
            elif self.provider == "cryptocompare":
                # CryptoCompare: No strict rate limit, but be nice
                min_delay = 0.5  # seconds
            elif self.provider == "binance":
                # Binance: 1200 requests per minute
                min_delay = 0.05  # seconds
            else:
                min_delay = 1.0  # Default
            
            elapsed = (now - self.last_request_time).total_seconds()
            if elapsed < min_delay:
                time.sleep(min_delay - elapsed)
        
        self.request_count += 1
        self.last_request_time = datetime.now()
    
    def get_market_cap_data(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get market capitalization data for cryptocurrencies.
        
        Args:
            symbols: List of cryptocurrency symbols
            
        Returns:
            Dictionary mapping symbols to their market cap values
        """
        result = {}
        
        if self.provider == "coingecko":
            # Apply rate limiting
            self._rate_limit()
            
            # Map symbols to CoinGecko IDs if needed
            symbol_mapping = {
                "BTC": "bitcoin",
                "ETH": "ethereum",
                "XRP": "ripple",
                "LTC": "litecoin",
                "BCH": "bitcoin-cash",
                "ADA": "cardano",
                "DOT": "polkadot",
                "LINK": "chainlink",
                "BNB": "binancecoin",
                "XLM": "stellar"
            }
            
            # Convert symbols to comma-separated list of IDs
            ids = []
            for symbol in symbols:
                coin_id = symbol_mapping.get(symbol.upper(), symbol.lower())
                ids.append(coin_id)
            
            # Prepare API URL
            url = f"{self.base_url}/coins/markets"
            
            params = {
                "vs_currency": "usd",
                "ids": ",".join(ids),
                "order": "market_cap_desc",
                "per_page": 250,
                "page": 1
            }
            
            # Make API request
            response = requests.get(url, params=params)
            data = response.json()
            
            # Parse the response
            for coin in data:
                symbol = coin.get("symbol", "").upper()
                market_cap = coin.get("market_cap")
                if symbol and market_cap:
                    result[symbol] = market_cap
        
        elif self.provider == "cryptocompare":
            # Apply rate limiting
            self._rate_limit()
            
            for symbol in symbols:
                # Prepare API URL
                url = f"{self.base_url}/pricemultifull"
                
                params = {
                    "fsyms": symbol.upper(),
                    "tsyms": "USD"
                }
                
                # Add API key if available
                if self.api_key:
                    params["api_key"] = self.api_key
                
                # Make API request
                response = requests.get(url, params=params)
                data = response.json()
                
                # Parse the response
                if "RAW" in data and symbol.upper() in data["RAW"] and "USD" in data["RAW"][symbol.upper()]:
                    market_cap = data["RAW"][symbol.upper()]["USD"].get("MKTCAP")
                    if market_cap:
                        result[symbol.upper()] = market_cap
        
        return result
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get information about cryptocurrency exchanges.
        
        Returns:
            Dictionary with exchange information
        """
        if self.provider == "coingecko":
            # Apply rate limiting
            self._rate_limit()
            
            # Prepare API URL
            url = f"{self.base_url}/exchanges"
            
            # Make API request
            response = requests.get(url)
            data = response.json()
            
            # Parse the response
            exchanges = []
            for exchange in data:
                exchanges.append({
                    "id": exchange.get("id"),
                    "name": exchange.get("name"),
                    "year_established": exchange.get("year_established"),
                    "country": exchange.get("country"),
                    "url": exchange.get("url"),
                    "trust_score": exchange.get("trust_score"),
                    "trade_volume_24h_btc": exchange.get("trade_volume_24h_btc")
                })
            
            return {"exchanges": exchanges}
        
        elif self.provider == "cryptocompare":
            # Apply rate limiting
            self._rate_limit()
            
            # Prepare API URL
            url = f"{self.base_url}/exchanges/general"
            
            params = {}
            if self.api_key:
                params["api_key"] = self.api_key
            
            # Make API request
            response = requests.get(url, params=params)
            data = response.json()
            
            # Parse the response
            exchanges = []
            if "Data" in data:
                for exchange_id, exchange_data in data["Data"].items():
                    exchanges.append({
                        "id": exchange_id,
                        "name": exchange_data.get("Name"),
                        "country": exchange_data.get("Country"),
                        "url": exchange_data.get("AffiliateURL"),
                        "volume_usd_24h": exchange_data.get("TOTALVOLUME24H", {}).get("USD")
                    })
            
            return {"exchanges": exchanges}
        
        return {"exchanges": []} 