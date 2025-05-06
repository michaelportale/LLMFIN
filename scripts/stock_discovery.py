#!/usr/bin/env python3
"""
Stock discovery script to find interesting stocks to add to your tracking.
Generates custom_list suggestions based on criteria like trading volume, volatility, etc.
"""

import os
import sys
import logging
import argparse
import yaml
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class StockDiscovery:
    def __init__(self, output_file=None):
        """Initialize stock discovery tool."""
        self.output_file = output_file or os.path.join(project_root, 'configs', 'discovered_stocks.yml')
        
    def get_highest_volume_stocks(self, count=20):
        """Find stocks with highest trading volume."""
        logger.info(f"Finding {count} highest volume stocks...")
        try:
            # Get top US stocks by volume
            url = "https://finance.yahoo.com/most-active"
            tables = pd.read_html(url)
            
            if len(tables) > 0:
                df = tables[0]
                # Make sure we have the Symbol column
                for col in ['Symbol', 'Ticker', 'symbol']:
                    if col in df.columns:
                        symbols = df[col].tolist()[:count]
                        logger.info(f"Found {len(symbols)} high volume stocks")
                        return symbols
                        
            logger.warning("Could not find volume data in the expected format")
            return []
        except Exception as e:
            logger.error(f"Error fetching high volume stocks: {e}")
            return []
            
    def get_trending_stocks(self, count=20):
        """Find trending stocks based on price momentum."""
        logger.info(f"Finding {count} trending stocks...")
        try:
            # Get trending stocks from Yahoo Finance
            url = "https://finance.yahoo.com/trending-tickers"
            tables = pd.read_html(url)
            
            if len(tables) > 0:
                df = tables[0]
                for col in ['Symbol', 'Ticker', 'symbol']:
                    if col in df.columns:
                        symbols = df[col].tolist()[:count]
                        logger.info(f"Found {len(symbols)} trending stocks")
                        return symbols
                        
            logger.warning("Could not find trending data in the expected format")
            return []
        except Exception as e:
            logger.error(f"Error fetching trending stocks: {e}")
            return []
            
    def get_high_volatility_stocks(self, count=20):
        """Find stocks with high volatility."""
        logger.info(f"Finding {count} high volatility stocks...")
        try:
            # Stocks with high beta (volatility)
            url = "https://finance.yahoo.com/screener/predefined/high_volatility_growth_stocks"
            tables = pd.read_html(url)
            
            if len(tables) > 0:
                df = tables[0]
                for col in ['Symbol', 'Ticker', 'symbol']:
                    if col in df.columns:
                        symbols = df[col].tolist()[:count]
                        logger.info(f"Found {len(symbols)} high volatility stocks")
                        return symbols
                        
            logger.warning("Could not find volatility data in the expected format")
            return []
        except Exception as e:
            logger.error(f"Error fetching high volatility stocks: {e}")
            return []
    
    def get_sector_leaders(self, sectors=None, count_per_sector=5):
        """Find leading stocks in each sector."""
        if sectors is None:
            sectors = [
                'technology', 'healthcare', 'financial', 'energy',
                'consumer_cyclical', 'industrials', 'communication_services'
            ]
            
        logger.info(f"Finding {count_per_sector} leaders for each of {len(sectors)} sectors...")
        sector_leaders = {}
        
        # Mapping between our sector names and Yahoo Finance URLs
        sector_urls = {
            'technology': 'technology',
            'healthcare': 'healthcare',
            'financial': 'financial-services',
            'energy': 'energy',
            'consumer_cyclical': 'consumer-cyclical',
            'utilities': 'utilities',
            'real_estate': 'real-estate',
            'communication_services': 'communication-services',
            'industrials': 'industrials',
            'basic_materials': 'basic-materials',
            'consumer_defensive': 'consumer-defensive'
        }
        
        for sector in sectors:
            if sector not in sector_urls:
                logger.warning(f"Unknown sector: {sector}")
                continue
                
            url_sector = sector_urls[sector]
            try:
                url = f"https://finance.yahoo.com/sector/{url_sector}"
                tables = pd.read_html(url)
                
                if len(tables) > 0:
                    df = tables[0]
                    # The column name might vary
                    for col in ['Symbol', 'Ticker', 'symbol']:
                        if col in df.columns:
                            symbols = df[col].tolist()[:count_per_sector]
                            sector_leaders[sector] = symbols
                            logger.info(f"Found {len(symbols)} leaders for {sector}")
                            break
                    else:
                        logger.warning(f"Could not find symbol column for sector {sector}")
            except Exception as e:
                logger.error(f"Error fetching sector leaders for {sector}: {e}")
                
        return sector_leaders
    
    def analyze_stocks(self, symbols, period='3mo'):
        """Analyze a list of stocks and get relevant metrics."""
        logger.info(f"Analyzing {len(symbols)} stocks...")
        results = []
        
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period=period)
                
                if len(hist) < 20:  # Need enough data points
                    continue
                    
                # Calculate metrics
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
                
                avg_volume = hist['Volume'].mean()
                current_price = hist['Close'].iloc[-1]
                
                # Get company info
                info = stock.info
                market_cap = info.get('marketCap', 0)
                name = info.get('shortName', symbol)
                sector = info.get('sector', 'Unknown')
                
                results.append({
                    'symbol': symbol,
                    'name': name,
                    'sector': sector,
                    'price': current_price,
                    'market_cap': market_cap,
                    'volatility': volatility,
                    'avg_volume': avg_volume
                })
                
            except Exception as e:
                logger.warning(f"Error analyzing {symbol}: {e}")
        
        return results
    
    def find_interesting_stocks(self, max_stocks=100):
        """Find interesting stocks using multiple methods."""
        # Get stocks from different sources
        volume_stocks = self.get_highest_volume_stocks(count=30)
        trending_stocks = self.get_trending_stocks(count=30)
        volatile_stocks = self.get_high_volatility_stocks(count=30)
        
        sector_leaders_dict = self.get_sector_leaders(count_per_sector=10)
        sector_leaders = []
        for sector, symbols in sector_leaders_dict.items():
            sector_leaders.extend(symbols)
        
        # Combine and remove duplicates
        all_symbols = list(set(volume_stocks + trending_stocks + volatile_stocks + sector_leaders))
        logger.info(f"Found {len(all_symbols)} unique symbols across all discovery methods")
        
        # Limit to max_stocks
        if len(all_symbols) > max_stocks:
            logger.info(f"Limiting to {max_stocks} stocks for analysis")
            all_symbols = all_symbols[:max_stocks]
        
        # Analyze the stocks
        analyzed_stocks = self.analyze_stocks(all_symbols)
        
        # Sort by different criteria
        by_volume = sorted(analyzed_stocks, key=lambda x: x.get('avg_volume', 0), reverse=True)
        by_volatility = sorted(analyzed_stocks, key=lambda x: x.get('volatility', 0), reverse=True)
        by_market_cap = sorted(analyzed_stocks, key=lambda x: x.get('market_cap', 0), reverse=True)
        
        # Create output with categorized recommendations
        output = {
            'timestamp': datetime.now().isoformat(),
            'total_analyzed': len(analyzed_stocks),
            'categories': {
                'high_volume': [s['symbol'] for s in by_volume[:20]],
                'high_volatility': [s['symbol'] for s in by_volatility[:20]],
                'large_cap': [s['symbol'] for s in by_market_cap[:20]],
                'sector_leaders': sector_leaders_dict,
            },
            'all_discovered': [s['symbol'] for s in analyzed_stocks],
            'stock_details': {s['symbol']: s for s in analyzed_stocks}
        }
        
        # Save to file
        if self.output_file:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            with open(self.output_file, 'w') as f:
                yaml.dump(output, f)
            logger.info(f"Saved discovery results to {self.output_file}")
            
        return output
        
    def generate_config_update(self, discovery_output, existing_config_path=None):
        """Generate an updated config file with discovered stocks."""
        # Default config if none exists
        config = {
            'base_data_dir': 'data',
            'period': '5y',
            'interval': '1d',
            'max_symbols': 200,
            'threads': 15,
            'sources': {
                'sp500': False,
                'nasdaq100': False,
                'dow30': False,
                'custom_list': [],
                'sectors': []
            }
        }
        
        # Load existing config if available
        if existing_config_path and os.path.exists(existing_config_path):
            with open(existing_config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Update custom_list with discovered stocks
        custom_list = set(config['sources'].get('custom_list', []))
        
        # Add high volume stocks
        custom_list.update(discovery_output['categories']['high_volume'][:10])
        
        # Add high volatility stocks
        custom_list.update(discovery_output['categories']['high_volatility'][:10])
        
        # Add large cap stocks
        custom_list.update(discovery_output['categories']['large_cap'][:10])
        
        # Add some sector leaders from each sector
        for sector, symbols in discovery_output['categories']['sector_leaders'].items():
            custom_list.update(symbols[:5])
            
        # Update config
        config['sources']['custom_list'] = sorted(list(custom_list))
        
        # Generate filename for new config
        new_config_path = os.path.join(
            project_root, 
            'configs', 
            f'data_fetcher_discovered_{datetime.now().strftime("%Y%m%d")}.yml'
        )
        
        # Save new config
        os.makedirs(os.path.dirname(new_config_path), exist_ok=True)
        with open(new_config_path, 'w') as f:
            yaml.dump(config, f)
            
        logger.info(f"Generated updated config with {len(custom_list)} discovered stocks: {new_config_path}")
        return new_config_path

def main():
    parser = argparse.ArgumentParser(description='Discover interesting stocks to track')
    parser.add_argument('--output', type=str, help='Output file path for discovery results')
    parser.add_argument('--config', type=str, help='Existing config file to update')
    parser.add_argument('--max-stocks', type=int, default=100, help='Maximum number of stocks to analyze')
    args = parser.parse_args()
    
    discovery = StockDiscovery(output_file=args.output)
    discovery_output = discovery.find_interesting_stocks(max_stocks=args.max_stocks)
    
    new_config_path = discovery.generate_config_update(
        discovery_output, 
        existing_config_path=args.config
    )
    
    print(f"\nDiscovery complete!")
    print(f"Found {len(discovery_output['all_discovered'])} interesting stocks")
    print(f"Updated config saved to: {new_config_path}")
    print(f"Discovery details saved to: {discovery.output_file}")
    
if __name__ == "__main__":
    main() 