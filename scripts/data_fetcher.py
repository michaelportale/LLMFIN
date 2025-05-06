import os
import yfinance as yf
import pandas as pd
from pathlib import Path
import logging
import argparse
from datetime import datetime
import yaml
import requests
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, config_path=None):
        """Initialize the data fetcher with a configuration."""
        self.config = self._load_config(config_path)
        
        # Ensure data directories exist
        self.base_dir = self.config.get('base_data_dir', 'data')
        self.stock_dir = os.path.join(self.base_dir, 'stocks')
        self.sentiment_dir = os.path.join(self.base_dir, 'sentiment')
        self.metadata_dir = os.path.join(self.base_dir, 'metadata')
        
        for directory in [self.base_dir, self.stock_dir, self.sentiment_dir, self.metadata_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path):
        """Load configuration from YAML file or use defaults."""
        default_config = {
            'base_data_dir': 'data',
            'period': '5y',
            'interval': '1d',
            'max_symbols': 100,
            'threads': 10,
            'sources': {
                'sp500': True,
                'nasdaq100': True,
                'dow30': True,
                'custom_list': [],
                'sectors': []
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        
        return default_config
    
    def get_symbols_from_index(self, index_name):
        """Get symbols from a market index."""
        symbols = []
        
        if index_name.lower() == 'sp500':
            try:
                table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
                symbols = table['Symbol'].tolist()
            except Exception as e:
                logger.error(f"Failed to fetch S&P 500 symbols: {e}")
        
        elif index_name.lower() == 'nasdaq100':
            try:
                table = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[1]
                symbols = table['Ticker'].tolist()
            except Exception as e:
                logger.error(f"Failed to fetch NASDAQ 100 symbols: {e}")
        
        elif index_name.lower() == 'dow30':
            try:
                table = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')[1]
                symbols = table['Symbol'].tolist()
            except Exception as e:
                logger.error(f"Failed to fetch Dow 30 symbols: {e}")
        
        return [sym.replace('.', '-') for sym in symbols]  # Fix symbols like BRK.B
    
    def get_symbols_from_sector(self, sector):
        """Get symbols for a specific sector using Yahoo Finance sector listings."""
        sectors_map = {
            'technology': 'technology',
            'healthcare': 'healthcare',
            'financial': 'financial-services',
            'consumer_cyclical': 'consumer-cyclical',
            'energy': 'energy',
            'utilities': 'utilities',
            'real_estate': 'real-estate',
            'communication_services': 'communication-services',
            'industrials': 'industrials',
            'basic_materials': 'basic-materials',
            'consumer_defensive': 'consumer-defensive'
        }
        
        if sector not in sectors_map:
            logger.warning(f"Unknown sector: {sector}. Available sectors: {list(sectors_map.keys())}")
            return []
            
        try:
            # Get top stocks from the sector
            url = f"https://finance.yahoo.com/sector/{sectors_map[sector]}"
            tables = pd.read_html(url)
            
            # Usually the first table contains the stocks
            if len(tables) > 0:
                df = tables[0]
                # The column name might vary, try common ones
                for col in ['Symbol', 'ticker', 'Ticker']:
                    if col in df.columns:
                        return df[col].tolist()[:50]  # Limit to top 50 per sector
            
            logger.warning(f"Could not find symbols for sector: {sector}")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch symbols for sector {sector}: {e}")
            return []
    
    def get_all_symbols(self):
        """Get all symbols based on configuration."""
        all_symbols = set()
        sources = self.config['sources']
        
        # Get symbols from indices
        if sources.get('sp500', False):
            all_symbols.update(self.get_symbols_from_index('sp500'))
        
        if sources.get('nasdaq100', False):
            all_symbols.update(self.get_symbols_from_index('nasdaq100'))
            
        if sources.get('dow30', False):
            all_symbols.update(self.get_symbols_from_index('dow30'))
        
        # Add custom symbols
        all_symbols.update(sources.get('custom_list', []))
        
        # Add sector-based symbols
        for sector in sources.get('sectors', []):
            all_symbols.update(self.get_symbols_from_sector(sector))
        
        # Limit the number of symbols if needed
        max_symbols = int(self.config.get('max_symbols', 100))
        symbols_list = list(all_symbols)
        if len(symbols_list) > max_symbols:
            symbols_list = symbols_list[:max_symbols]
            logger.warning(f"Limited to {max_symbols} symbols out of {len(all_symbols)} total")
        
        return symbols_list
    
    def fetch_stock_data(self, symbol):
        """Fetch data for a single stock symbol."""
        try:
            period = self.config.get('period', '5y')
            interval = self.config.get('interval', '1d')
            
            logger.info(f"Fetching {symbol} data with period={period}, interval={interval}")
            
            # Fetch the data
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return None
                
            # Save the data
            file_path = os.path.join(self.stock_dir, f"{symbol}.csv")
            df.to_csv(file_path)
            logger.info(f"Saved {symbol} data to {file_path}")
            
            # Get basic info for metadata
            try:
                info = stock.info
                metadata = {
                    'symbol': symbol,
                    'company_name': info.get('shortName', 'Unknown'),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'last_updated': datetime.now().isoformat(),
                    'data_start': df.index.min().isoformat() if not df.empty else None,
                    'data_end': df.index.max().isoformat() if not df.empty else None,
                    'data_points': len(df),
                    'fields': list(df.columns)
                }
                
                # Save metadata
                meta_path = os.path.join(self.metadata_dir, f"{symbol}_meta.yml")
                with open(meta_path, 'w') as f:
                    yaml.dump(metadata, f)
                
            except Exception as e:
                logger.error(f"Failed to fetch metadata for {symbol}: {e}")
            
            return symbol
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def fetch_all_stocks(self):
        """Fetch data for all configured symbols in parallel."""
        symbols = self.get_all_symbols()
        logger.info(f"Found {len(symbols)} symbols to fetch")
        
        # Track stocks we already have data for
        existing_stocks = {f.split('.')[0] for f in os.listdir(self.stock_dir) if f.endswith('.csv')}
        new_symbols = [s for s in symbols if s not in existing_stocks]
        update_symbols = [s for s in symbols if s in existing_stocks]
        
        logger.info(f"New symbols to fetch: {len(new_symbols)}")
        logger.info(f"Existing symbols to update: {len(update_symbols)}")
        
        # Fetch all symbols in parallel
        threads = min(int(self.config.get('threads', 10)), len(symbols))
        all_symbols = new_symbols + update_symbols
        
        with ThreadPoolExecutor(max_workers=threads) as executor:
            results = list(executor.map(self.fetch_stock_data, all_symbols))
        
        successful = [r for r in results if r is not None]
        logger.info(f"Successfully fetched data for {len(successful)}/{len(all_symbols)} symbols")
        
        # Create a summary file
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_symbols': len(symbols),
            'new_symbols': len(new_symbols),
            'updated_symbols': len(update_symbols),
            'successful_fetches': len(successful),
            'failed_fetches': len(all_symbols) - len(successful)
        }
        
        summary_path = os.path.join(self.base_dir, 'fetch_summary.yml')
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f)
            
        return successful

def main():
    parser = argparse.ArgumentParser(description='Fetch stock data based on configuration')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    args = parser.parse_args()
    
    fetcher = DataFetcher(args.config)
    fetcher.fetch_all_stocks()

if __name__ == "__main__":
    main() 