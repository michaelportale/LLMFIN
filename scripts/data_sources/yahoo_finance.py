"""
Yahoo Finance data source implementation.
"""
import yfinance as yf
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time
from .base import DataSource


class YahooFinanceDataSource(DataSource):
    """
    Yahoo Finance data source implementation.
    Uses the yfinance library to fetch data from Yahoo Finance.
    """
    
    def __init__(self, save_path: str = "data/"):
        """
        Initialize the Yahoo Finance data source.
        
        Args:
            save_path: Directory to save downloaded data
        """
        super().__init__(save_path)
        self.request_count = 0
        self.last_request_time = None
    
    def fetch_historical_data(self, 
                              symbols: List[str], 
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              period: Optional[str] = "2y",
                              interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical price data from Yahoo Finance.
        
        Args:
            symbols: List of ticker symbols to fetch
            start_date: Start date in YYYY-MM-DD format (if None, use period)
            end_date: End date in YYYY-MM-DD format (if None, use today)
            period: Time period as string (e.g. "2y" for 2 years) - used if start_date is None
            interval: Data interval (e.g. "1d" for daily, "1h" for hourly)
            
        Returns:
            Dictionary mapping symbols to their respective DataFrames
        """
        result = {}
        
        for symbol in symbols:
            print(f"Downloading data for {symbol} from Yahoo Finance...")
            
            # Implement rate limiting
            self._rate_limit()
            
            # Download data
            if start_date and end_date:
                df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
            else:
                df = yf.download(symbol, period=period, interval=interval)
            
            df.dropna(inplace=True)
            
            # Check if columns are MultiIndex and handle appropriately
            if isinstance(df.columns, pd.MultiIndex):
                print(f"Detected MultiIndex columns for {symbol}")
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
            
            # Verify data quality
            quality_metrics = self.verify_data_quality(clean_df)
            print(f"Data quality metrics for {symbol}: {quality_metrics}")
            
            # Save data
            csv_path = self.save_data(symbol, clean_df)
            print(f"Saved {symbol} data to {csv_path}")
            
            result[symbol] = clean_df
        
        return result
    
    def get_supported_assets(self) -> List[str]:
        """
        Get list of asset types supported by Yahoo Finance.
        
        Returns:
            List of supported asset types
        """
        return ["stocks", "etfs", "indices", "mutual_funds", "currencies", "cryptocurrencies"]
    
    def get_api_usage_info(self) -> Dict[str, Any]:
        """
        Get information about API usage limits and current status.
        Yahoo Finance doesn't have official API limits, but we track requests for rate limiting.
        
        Returns:
            Dictionary with API usage information
        """
        return {
            "provider": "Yahoo Finance",
            "requests_made": self.request_count,
            "last_request": self.last_request_time.isoformat() if self.last_request_time else None,
            "limits": "No official limits, but rate limiting is applied"
        }
    
    def _rate_limit(self):
        """
        Apply rate limiting to avoid being blocked by Yahoo Finance.
        """
        if self.last_request_time:
            # Ensure at least 0.5 seconds between requests
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < 0.5:
                time.sleep(0.5 - elapsed)
        
        self.request_count += 1
        self.last_request_time = datetime.now()
        
        # Add a small delay every 10 requests
        if self.request_count % 10 == 0:
            time.sleep(2) 