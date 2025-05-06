# Financial Data Sources

This part of the project provides a flexible and extensible system for fetching financial data from various providers including Yahoo Finance, Alpha Vantage, Quandl, and specialized sources for cryptocurrencies and forex.

## Features

- Multiple data source providers (Yahoo Finance, Alpha Vantage, Quandl, etc.)
- Support for different asset types (stocks, ETFs, crypto, forex)
- Real-time data streaming capabilities
- Data quality verification system
- Consistent API across all data sources

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set up API keys:

Create a `.env` file in the project root with your API keys:

```
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
QUANDL_API_KEY=your_quandl_key
CRYPTOCOMPARE_API_KEY=your_cryptocompare_key
FIXER_API_KEY=your_fixer_key
OANDA_API_KEY=your_oanda_key
```

Note: Most data sources require API keys for access. Free tiers are available for development and testing.

## Usage

### Basic Usage

Here's how to fetch historical data for stocks using Yahoo Finance:

```python
from scripts.data_sources import YahooFinanceDataSource

# Initialize the data source
yahoo_finance = YahooFinanceDataSource(save_path="data/yahoo/")

# Fetch historical data
symbols = ["AAPL", "MSFT", "GOOG"]
start_date = "2022-01-01"
end_date = "2023-01-01"

data = yahoo_finance.fetch_historical_data(
    symbols, 
    start_date=start_date, 
    end_date=end_date
)

# Access the data
apple_data = data["AAPL"]
print(apple_data.head())
```

### Available Data Sources

1. **Yahoo Finance** (`YahooFinanceDataSource`):
   - No API key required
   - Supports stocks, ETFs, indices, and some crypto

2. **Alpha Vantage** (`AlphaVantageDataSource`):
   - Requires API key
   - Supports stocks, forex, crypto, and economic indicators

3. **Quandl** (`QuandlDataSource`):
   - Requires API key
   - Supports stocks, commodities, and economic data

4. **Crypto** (`CryptoDataSource`):
   - Supports multiple providers (CoinGecko, CryptoCompare, Binance)
   - Some providers require API keys

5. **Forex** (`ForexDataSource`):
   - Supports multiple providers (Alpha Vantage, OANDA, Fixer)
   - All providers require API keys

### Real-Time Data Streaming

You can stream real-time data using the following classes:

```python
from scripts.data_sources import AlphaVantageRealTime, CryptoRealTimeStream, DataStreamManager

# Define callback to handle incoming data
def on_data(data):
    print(f"Received data: {data}")

# Initialize data stream manager
manager = DataStreamManager()

# Add Alpha Vantage real-time stream
alpha_vantage_stream = AlphaVantageRealTime(
    api_key="your_alpha_vantage_key",
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
manager.start_all()

# When done
manager.stop_all()
```

### Data Quality Verification

You can check the quality of your financial data using the `DataQualityChecker` class:

```python
from scripts.data_quality import DataQualityChecker

# Initialize checker
checker = DataQualityChecker()

# Generate a comprehensive quality report
quality_report = checker.generate_report(data_frame)

# Visualize data quality issues
checker.visualize_data_quality(data_frame, output_file="data_quality.png")
```

You can also compare data from different sources:

```python
from scripts.data_quality import compare_data_sources, visualize_comparison

# Compare two data sources
comparison = compare_data_sources(
    data_frame1, 
    data_frame2,
    source1_name="Yahoo Finance",
    source2_name="Alpha Vantage"
)

# Visualize the comparison
visualize_comparison(
    data_frame1, 
    data_frame2,
    source1_name="Yahoo Finance",
    source2_name="Alpha Vantage",
    output_file="comparison.png"
)
```

## Examples

Check the `examples/data_sources_demo.py` script for comprehensive examples of using all data sources:

```bash
# Run all demos
python examples/data_sources_demo.py

# Run specific demo
python examples/data_sources_demo.py --demo stocks
python examples/data_sources_demo.py --demo crypto
python examples/data_sources_demo.py --demo forex
python examples/data_sources_demo.py --demo real-time
```

## Extending

To add a new data source, create a new class that inherits from the `DataSource` base class:

```python
from scripts.data_sources.base import DataSource

class MyCustomDataSource(DataSource):
    def __init__(self, api_key: str, save_path: str = "data/"):
        super().__init__(save_path)
        self.api_key = api_key
        # Initialize your custom data source
        
    def fetch_historical_data(self, symbols, start_date=None, end_date=None, 
                             period=None, interval="1d"):
        # Implement fetching historical data
        pass
        
    def get_supported_assets(self):
        # Return list of supported asset types
        return ["stocks", "etfs"]
        
    def get_api_usage_info(self):
        # Return API usage information
        return {"provider": "My Custom Provider", "limits": "..."}
```

## API Reference

### DataSource

Base class for all data sources with the following methods:

- `fetch_historical_data(symbols, start_date=None, end_date=None, period=None, interval="1d")`: Fetch historical price data
- `get_supported_assets()`: Get list of supported asset types
- `get_api_usage_info()`: Get information about API usage limits
- `save_data(symbol, data)`: Save data to CSV file
- `verify_data_quality(data)`: Verify the quality of the data

### RealTimeDataStream

Base class for real-time data streams with the following methods:

- `start()`: Start the data stream
- `stop()`: Stop the data stream
- `_run()`: Run the data stream (implemented by subclasses)
- `_process_data(data)`: Process incoming data

### DataQualityChecker

Class for checking and verifying financial data quality:

- `check_completeness(df)`: Check data completeness
- `check_time_consistency(df)`: Check time series consistency
- `check_price_quality(df)`: Check price data quality
- `check_volume_quality(df)`: Check volume data quality
- `visualize_data_quality(df, output_file=None)`: Visualize data quality issues
- `generate_report(df)`: Generate a comprehensive data quality report 