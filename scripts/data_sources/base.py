"""
Base class for data sources.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any, Optional
import os


class DataSource(ABC):
    """
    Abstract base class for all data sources.
    All data source implementations should inherit from this class.
    """
    
    def __init__(self, save_path: str = "data/"):
        """
        Initialize the data source.
        
        Args:
            save_path: Directory to save downloaded data
        """
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
    
    @abstractmethod
    def fetch_historical_data(self, 
                              symbols: List[str], 
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              period: Optional[str] = None,
                              interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical price data for the given symbols.
        
        Args:
            symbols: List of ticker symbols to fetch
            start_date: Start date in YYYY-MM-DD format (if None, use period)
            end_date: End date in YYYY-MM-DD format (if None, use today)
            period: Time period as string (e.g. "2y" for 2 years) - used if start_date is None
            interval: Data interval (e.g. "1d" for daily, "1h" for hourly)
            
        Returns:
            Dictionary mapping symbols to their respective DataFrames
        """
        pass
    
    @abstractmethod
    def get_supported_assets(self) -> List[str]:
        """
        Get list of asset types supported by this data source.
        
        Returns:
            List of supported asset types (e.g. ["stocks", "etfs", "forex", "crypto"])
        """
        pass
    
    @abstractmethod
    def get_api_usage_info(self) -> Dict[str, Any]:
        """
        Get information about API usage limits and current status.
        
        Returns:
            Dictionary with API usage information
        """
        pass
    
    def save_data(self, symbol: str, data: pd.DataFrame) -> str:
        """
        Save data to CSV file.
        
        Args:
            symbol: Ticker symbol
            data: DataFrame with OHLCV data
            
        Returns:
            Path to saved file
        """
        # Ensure data has the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Data missing required column: {col}")
        
        # Save with proper column headers
        csv_path = os.path.join(self.save_path, f"{symbol}.csv")
        data.to_csv(csv_path, index_label="Date")
        return csv_path
    
    def verify_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Verify the quality of the data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            "row_count": len(data),
            "missing_values": data.isnull().sum().to_dict(),
            "has_gaps": False,
            "date_range": None
        }
        
        # Check for date gaps
        if len(data) > 1 and isinstance(data.index, pd.DatetimeIndex):
            date_diff = data.index.to_series().diff().dropna()
            unique_diffs = date_diff.unique()
            if len(unique_diffs) > 1:
                metrics["has_gaps"] = True
                metrics["gap_sizes"] = [str(diff) for diff in unique_diffs]
            
            metrics["date_range"] = {
                "start": data.index.min().strftime("%Y-%m-%d"),
                "end": data.index.max().strftime("%Y-%m-%d")
            }
        
        return metrics 