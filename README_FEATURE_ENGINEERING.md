# Feature Engineering for Financial Data Analysis

This module provides comprehensive feature engineering capabilities for financial time series data, supporting technical indicators, automated feature importance analysis, fundamental data integration, and advanced visualization tools.

## Features

- **Technical Indicators**: Over 15 technical indicators including MACD, RSI, Bollinger Bands, Stochastic Oscillator, and more
- **Feature Importance Analysis**: Multiple methods for identifying important features for predictive modeling
- **Fundamental Data Integration**: Fetch, process, and integrate fundamental financial data with price data
- **Feature Visualization Tools**: Comprehensive visualization capabilities for financial data and features

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Technical Indicators

```python
from scripts.feature_engineering.technical_indicators import add_all_indicators
import pandas as pd

# Load your OHLCV data
df = pd.read_csv('data/AAPL.csv', index_col=0, parse_dates=True)

# Add all technical indicators
df_with_indicators = add_all_indicators(df)

# Or add specific indicators
from scripts.feature_engineering.technical_indicators import add_macd, add_rsi, add_bollinger_bands

df = add_macd(df)
df = add_rsi(df)
df = add_bollinger_bands(df)
```

### Feature Importance Analysis

```python
from scripts.feature_engineering.feature_importance import run_feature_importance_analysis
import pandas as pd

# Load your data with features
df = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)

# Run a comprehensive feature importance analysis
results = run_feature_importance_analysis(
    df,
    target_column='Return_5d',  # Target variable for prediction
    output_dir='data/feature_analysis',
    n_features=20
)

# Get top features
top_features = results['top_features']
print(f"Top features: {top_features}")
```

### Fundamental Data Integration

```python
from scripts.feature_engineering.fundamental_data import FundamentalDataFetcher
import pandas as pd

# Initialize the fundamental data fetcher
fetcher = FundamentalDataFetcher(save_path='data/fundamental/')

# Fetch fundamental data for a list of symbols
symbols = ['AAPL', 'MSFT', 'GOOG']
fundamental_data = fetcher.fetch_all_fundamental_data(symbols)

# Calculate financial ratios
from scripts.feature_engineering.fundamental_data import calculate_financial_ratios
ratios = calculate_financial_ratios(fundamental_data)

# Get sector and industry data
from scripts.feature_engineering.fundamental_data import get_sector_industry_data
sector_data = get_sector_industry_data(symbols)

# Add fundamental features to price data
from scripts.feature_engineering.fundamental_data import add_fundamental_features
price_data = pd.read_csv('data/AAPL.csv', index_col=0, parse_dates=True)
df_with_fundamentals = add_fundamental_features(price_data, fetcher, symbols=['AAPL'])
```

### Feature Visualization

```python
from scripts.feature_engineering.feature_visualization import visualize_features
import pandas as pd

# Load your data with features
df = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)

# Create comprehensive visualizations
visualize_features(
    df,
    target_column='Return_5d',
    top_n_features=10,
    output_dir='data/visualizations'
)

# Create an interactive dashboard
from scripts.feature_engineering.feature_visualization import create_feature_dashboard
create_feature_dashboard(df, target_column='Return_5d', output_file='data/dashboard.html')
```

## Example Script

Check out the `examples/feature_engineering_demo.py` script for a comprehensive demonstration of all feature engineering capabilities:

```bash
# Run all demos
python examples/feature_engineering_demo.py

# Run specific demo
python examples/feature_engineering_demo.py --demo technical
python examples/feature_engineering_demo.py --demo importance
python examples/feature_engineering_demo.py --demo fundamental
python examples/feature_engineering_demo.py --demo visualization
python examples/feature_engineering_demo.py --demo combined
```

## Technical Indicators

The following technical indicators are available:

- Simple Moving Averages (SMA)
- Exponential Moving Averages (EMA)
- Moving Average Convergence Divergence (MACD)
- Relative Strength Index (RSI)
- Bollinger Bands
- Stochastic Oscillator
- Average True Range (ATR)
- Fibonacci Retracement Levels
- Ichimoku Cloud
- On-Balance Volume (OBV)
- Accumulation/Distribution Line
- Average Directional Index (ADX)
- Volume Weighted Average Price (VWAP)
- Money Flow Index (MFI)
- Price returns (percentage and logarithmic)

## Feature Importance Methods

Multiple methods are used to calculate feature importance:

- **Correlation Importance**: Linear correlation between features and target
- **Mutual Information**: Information theory-based importance measure
- **Random Forest Importance**: Tree-based feature importance
- **Permutation Importance**: Importance based on impact of feature shuffling
- **Combined Importance**: Ensemble of multiple importance methods

## Fundamental Data Integration

The following fundamental data can be fetched and integrated:

- Financial statements (Income Statement, Balance Sheet, Cash Flow)
- Key financial metrics (P/E ratio, EPS, ROE, etc.)
- Earnings data and analyst estimates
- Sector and industry classification
- Financial ratios and calculated metrics

## Visualization Capabilities

The module includes the following visualization tools:

- Time series plots with technical indicators
- Feature distribution analysis
- Correlation heatmaps
- Feature-target correlation analysis
- Feature importance plots
- Seasonal decomposition
- Interactive dashboards

## Dependencies

- pandas, numpy: Data manipulation
- matplotlib, seaborn: Visualization
- scikit-learn: Feature importance algorithms
- statsmodels: Time series analysis
- plotly: Interactive visualizations
- yfinance: Financial data fetching
- and more (see requirements.txt)

## Contributing

To contribute to the feature engineering module:

1. Add new technical indicators in `technical_indicators.py`
2. Implement new feature importance methods in `feature_importance.py`
3. Add new fundamental data sources in `fundamental_data.py`
4. Create new visualization tools in `feature_visualization.py`
5. Update the demo script and README accordingly 