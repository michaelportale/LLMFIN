# Data Management System

This folder contains financial data used by the application. The system is designed to dynamically discover, fetch, and update stock data without hard-coding symbols.

## Directory Structure

```
data/
├── stocks/         # Stock price data (OHLCV)
├── sentiment/      # Sentiment data related to stocks
├── metadata/       # Metadata about each stock
└── update_history.yml  # History of data updates
```

## Tools for Dynamic Data Management

Several tools have been created to manage data dynamically:

### 1. Web-Based Stock Search (New!)

The easiest way to find and add any stock to your tracking list:

```
http://localhost:5000/stocks/search
```

This user-friendly interface allows you to:
- Search for any real stock by name, symbol, or description
- View detailed company information and metrics
- Add selected stocks to your tracking list with one click
- Immediately fetch historical data for added stocks

### 2. Command-Line Stock Search Tool

For power users and automation, use the command-line tool:

```bash
python scripts/stock_search.py "Tesla" --limit 10 --add --fetch
```

Options:
- `--limit`: Maximum number of results to return
- `--market`: Market to search in (US, ALL)
- `--config`: Path to configuration file
- `--details SYMBOL`: Get detailed information for a specific symbol
- `--add`: Add found symbols to custom list in config
- `--fetch`: Immediately fetch data for found symbols

### 3. Data Fetcher (`scripts/data_fetcher.py`)

Fetches stock data based on configuration:

```bash
python scripts/data_fetcher.py --config configs/data_fetcher_config.yml
```

### 4. Stock Discovery (`scripts/stock_discovery.py`)

Automatically discovers interesting stocks to track based on various criteria:

```bash
python scripts/stock_discovery.py --max-stocks 150
```

This will:
- Find high volume stocks
- Identify trending stocks
- Discover volatile stocks
- Find sector leaders
- Generate a configuration file with these stocks

### 5. Scheduled Updates (`scripts/scheduled_data_update.py`)

Set up regular data updates (can be scheduled with cron):

```bash
python scripts/scheduled_data_update.py --config configs/data_fetcher_config.yml
```

Example crontab entry for daily updates at 6pm:
```
0 18 * * * /path/to/venv/bin/python /path/to/scripts/scheduled_data_update.py --config /path/to/configs/data_fetcher_config.yml
```

## API Endpoints

The system includes REST API endpoints for integration with other applications:

- `GET /api/stocks/search?query=<search_term>&market=US&limit=20` - Search for stocks
- `GET /api/stocks/details/<symbol>` - Get detailed information about a stock
- `POST /api/stocks/add` - Add stocks to tracking list (with JSON body containing symbols array)
- `POST /api/stocks/fetch` - Fetch data for specified stocks (with JSON body containing symbols array)

## Configuration (`configs/data_fetcher_config.yml`)

The configuration file controls which data is fetched:

```yaml
base_data_dir: data
period: '5y'        # 5 years of data
interval: '1d'      # Daily data points
max_symbols: 200    # Limit to 200 symbols
threads: 15         # Parallel downloading

sources:
  sp500: true       # Fetch S&P 500 stocks
  nasdaq100: true   # Fetch NASDAQ 100 stocks
  dow30: false      # Skip Dow Jones stocks
  
  # Custom list of symbols
  custom_list:
    - TSLA
    - NVDA
    
  # Sectors to fetch
  sectors:
    - technology
    - healthcare
```

## Workflow for Data Management

1. **Initial Setup**: 
   - Use the web interface at `/stocks/search` to find stocks you're interested in
   - Or run stock discovery to find interesting stocks: `python scripts/stock_discovery.py`
   - Review and adjust the generated config file as needed

2. **Data Fetching**:
   - Fetch data using the config: `python scripts/data_fetcher.py --config configs/data_fetcher_config.yml`

3. **Regular Updates**:
   - Set up scheduled updates using cron
   - Periodically run discovery to refresh the list of interesting stocks

4. **Data Usage**:
   - The application will automatically use the data in this directory
   - All data is in CSV format for easy analysis

## Extending the System

To add more data sources:
1. Extend the `DataFetcher` class in `scripts/data_fetcher.py`
2. Add new methods to fetch different types of data
3. Update the configuration structure as needed 