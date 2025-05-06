"""
Quandl data source implementation.
"""
import pandas as pd
import quandl
from typing import List, Dict, Any, Optional
import os
import time
from datetime import datetime, timedelta
from .base import DataSource


class QuandlDataSource(DataSource):
    """
    Quandl data source implementation.
    Fetches financial data from the Quandl API.
    """
    
    def __init__(self, api_key: str, save_path: str = "data/"):
        """
        Initialize the Quandl data source.
        
        Args:
            api_key: Quandl API key
            save_path: Directory to save downloaded data
        """
        super().__init__(save_path)
        self.api_key = api_key
        quandl.ApiConfig.api_key = api_key
        self.request_count = 0
        self.last_request_time = None
    
    def fetch_historical_data(self, 
                             symbols: List[str], 
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             period: Optional[str] = None,
                             interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical price data from Quandl.
        
        Args:
            symbols: List of Quandl dataset codes (e.g. ["WIKI/AAPL", "FRED/GDP"])
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            period: Time period as string (not used with Quandl)
            interval: Data interval (only "1d" supported for most Quandl datasets)
            
        Returns:
            Dictionary mapping symbols to their respective DataFrames
        """
        result = {}
        
        for symbol in symbols:
            print(f"Downloading data for {symbol} from Quandl...")
            
            # Apply rate limiting
            self._rate_limit()
            
            try:
                # Parse database and dataset code
                if "/" not in symbol:
                    print(f"Invalid Quandl symbol format for {symbol}. Expected format: 'DATABASE/DATASET'")
                    continue
                
                # Fetch data from Quandl
                params = {}
                if start_date:
                    params["start_date"] = start_date
                if end_date:
                    params["end_date"] = end_date
                
                df = quandl.get(symbol, **params)
                
                # Check if data was returned
                if df.empty:
                    print(f"No data returned for {symbol}")
                    continue
                
                # Rename columns to standard format if possible
                column_mapping = self._get_column_mapping(df.columns, symbol)
                
                # If we can't map to our standard format, skip this dataset
                if not column_mapping:
                    print(f"Couldn't map columns for {symbol} to standard OHLCV format")
                    continue
                
                # Rename columns
                df_mapped = df[list(column_mapping.keys())].rename(columns=column_mapping)
                
                # Add missing columns with default values
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_columns:
                    if col not in df_mapped.columns:
                        if col == 'Volume':
                            df_mapped[col] = 0
                        else:
                            # For price columns, use Close if available, otherwise skip
                            if 'Close' in df_mapped.columns:
                                df_mapped[col] = df_mapped['Close']
                            else:
                                print(f"Missing required column {col} for {symbol} and no substitute available")
                                continue
                
                # Ensure index is datetime
                if not isinstance(df_mapped.index, pd.DatetimeIndex):
                    df_mapped.index = pd.to_datetime(df_mapped.index)
                
                # Sort by date
                df_mapped.sort_index(inplace=True)
                
                # Verify data quality
                quality_metrics = self.verify_data_quality(df_mapped)
                print(f"Data quality metrics for {symbol}: {quality_metrics}")
                
                # Save data
                # Use the last part of the symbol as the filename (e.g., "WIKI/AAPL" -> "AAPL")
                filename = symbol.split("/")[-1]
                csv_path = self.save_data(filename, df_mapped)
                print(f"Saved {symbol} data to {csv_path}")
                
                result[symbol] = df_mapped
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        return result
    
    def get_supported_assets(self) -> List[str]:
        """
        Get list of asset types supported by Quandl.
        
        Returns:
            List of supported asset types
        """
        return ["stocks", "etfs", "futures", "commodities", "economic_indicators", "alternative_data"]
    
    def get_api_usage_info(self) -> Dict[str, Any]:
        """
        Get information about API usage limits and current status.
        
        Returns:
            Dictionary with API usage information
        """
        return {
            "provider": "Quandl",
            "requests_made": self.request_count,
            "last_request": self.last_request_time.isoformat() if self.last_request_time else None,
            "limits": {
                "free_tier": "50 calls per day",
                "premium": "Available with paid plans"
            }
        }
    
    def _rate_limit(self):
        """
        Apply rate limiting to avoid hitting Quandl's rate limits.
        """
        now = datetime.now()
        
        if self.last_request_time:
            # Ensure at least 1 second between requests
            elapsed = (now - self.last_request_time).total_seconds()
            if elapsed < 1:
                time.sleep(1 - elapsed)
        
        self.request_count += 1
        self.last_request_time = now
    
    def _get_column_mapping(self, columns: pd.Index, symbol: str) -> Dict[str, str]:
        """
        Get mapping from Quandl columns to standard OHLCV format.
        
        Args:
            columns: DataFrame columns
            symbol: Quandl symbol
            
        Returns:
            Dictionary mapping Quandl columns to standard columns
        """
        # Convert columns to strings
        columns = [str(col) for col in columns]
        
        # Common mappings for different Quandl databases
        if symbol.startswith("WIKI/"):
            # WIKI database (discontinued but used as example)
            mappings = {
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
                "Adj. Open": "Open",
                "Adj. High": "High",
                "Adj. Low": "Low",
                "Adj. Close": "Close",
                "Adj. Volume": "Volume"
            }
        elif symbol.startswith("EOD/"):
            # End of Day database
            mappings = {
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
                "Adjusted_close": "Close"
            }
        elif symbol.startswith("FRED/"):
            # FRED database (economic data)
            # Usually only has one column, which we'll map to Close
            mappings = {col: "Close" for col in columns}
        else:
            # Generic mapping based on common column names
            mappings = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "value": "Close",
                "price": "Close",
                "settle": "Close",
                "settlement_price": "Close"
            }
        
        # Try to find matches for each column
        result = {}
        for col in columns:
            col_lower = col.lower()
            
            # Direct match
            if col in mappings:
                result[col] = mappings[col]
                continue
            
            # Case-insensitive match
            for pattern, target in mappings.items():
                if pattern.lower() in col_lower:
                    result[col] = target
                    break
        
        # Check if we have at least one price column
        if not any(col in result.values() for col in ["Open", "High", "Low", "Close"]):
            # If we have only one column, map it to Close
            if len(columns) == 1:
                result[columns[0]] = "Close"
        
        return result
    
    def get_economic_data(self, indicator: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch economic indicator data from Quandl.
        
        Args:
            indicator: Economic indicator code (e.g. "FRED/GDP")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with economic indicator data
        """
        # Apply rate limiting
        self._rate_limit()
        
        try:
            # Prepare parameters
            params = {}
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date
            
            # Fetch data
            df = quandl.get(indicator, **params)
            
            # Check if data was returned
            if df.empty:
                print(f"No data returned for {indicator}")
                return pd.DataFrame()
            
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Sort by date
            df.sort_index(inplace=True)
            
            # Save data
            filename = indicator.replace("/", "_")
            
            # Create a DataFrame with standard format
            # Economic data typically has just one column, map it to Close
            if len(df.columns) == 1:
                std_df = pd.DataFrame({
                    "Close": df.iloc[:, 0],
                    "Open": df.iloc[:, 0],
                    "High": df.iloc[:, 0],
                    "Low": df.iloc[:, 0],
                    "Volume": 0
                }, index=df.index)
            else:
                # Try to map columns
                column_mapping = self._get_column_mapping(df.columns, indicator)
                if column_mapping:
                    std_df = df[list(column_mapping.keys())].rename(columns=column_mapping)
                    # Add missing columns
                    for col in ["Open", "High", "Low", "Close"]:
                        if col not in std_df.columns and "Close" in std_df.columns:
                            std_df[col] = std_df["Close"]
                    if "Volume" not in std_df.columns:
                        std_df["Volume"] = 0
                else:
                    # Can't map columns, just use the first column as Close
                    std_df = pd.DataFrame({
                        "Close": df.iloc[:, 0],
                        "Open": df.iloc[:, 0],
                        "High": df.iloc[:, 0],
                        "Low": df.iloc[:, 0],
                        "Volume": 0
                    }, index=df.index)
            
            # Save data
            csv_path = self.save_data(filename, std_df)
            print(f"Saved {indicator} data to {csv_path}")
            
            # Also save the original data for reference
            orig_path = os.path.join(self.save_path, f"{filename}_original.csv")
            df.to_csv(orig_path)
            
            return std_df
            
        except Exception as e:
            print(f"Error fetching economic data for {indicator}: {str(e)}")
            return pd.DataFrame()
    
    def search_datasets(self, query: str, per_page: int = 10, page: int = 1) -> List[Dict[str, Any]]:
        """
        Search for datasets in Quandl.
        
        Args:
            query: Search query
            per_page: Number of results per page
            page: Page number
            
        Returns:
            List of matching datasets
        """
        # Apply rate limiting
        self._rate_limit()
        
        try:
            # Quandl's search function returns a list of dictionaries
            results = quandl.search(query, per_page=per_page, page=page)
            
            # Format the results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "code": result.get("dataset", {}).get("dataset_code", ""),
                    "database_code": result.get("dataset", {}).get("database_code", ""),
                    "name": result.get("dataset", {}).get("name", ""),
                    "description": result.get("dataset", {}).get("description", ""),
                    "newest_available_date": result.get("dataset", {}).get("newest_available_date", ""),
                    "oldest_available_date": result.get("dataset", {}).get("oldest_available_date", ""),
                    "column_names": result.get("dataset", {}).get("column_names", []),
                    "frequency": result.get("dataset", {}).get("frequency", ""),
                    "full_code": f"{result.get('dataset', {}).get('database_code', '')}/{result.get('dataset', {}).get('dataset_code', '')}"
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching for datasets: {str(e)}")
            return [] 