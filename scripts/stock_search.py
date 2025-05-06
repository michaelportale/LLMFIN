#!/usr/bin/env python3
"""
Stock search utility that allows users to find any real stock by name, symbol, or description.
This tool can be used to discover stocks not included in the automated discovery process.
"""

import os
import sys
import logging
import argparse
import yaml
import pandas as pd
import yfinance as yf
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import re
import requests
from datetime import datetime

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

class StockSearch:
    def __init__(self, config_path=None):
        """Initialize stock search utility."""
        self.config_path = config_path
        self.data_dir = os.path.join(project_root, 'data')
        self.default_market = "US"
        
        # Load existing config if available
        self.config = self._load_config(config_path)
        
        # Initialize Yahoo Finance API cache
        self.search_cache_file = os.path.join(self.data_dir, 'search_cache.json')
        self.search_cache = self._load_search_cache()
    
    def _load_config(self, config_path):
        """Load configuration file if it exists."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return None
        
    def _load_search_cache(self):
        """Load search cache from disk if it exists."""
        if os.path.exists(self.search_cache_file):
            try:
                with open(self.search_cache_file, 'r') as f:
                    cache = json.load(f)
                    # Check if cache is still valid (less than 1 day old)
                    if 'timestamp' in cache:
                        cache_time = datetime.fromisoformat(cache['timestamp'])
                        now = datetime.now()
                        if (now - cache_time).days < 1:
                            return cache
            except Exception as e:
                logger.warning(f"Failed to load search cache: {e}")
        
        # Return empty cache
        return {
            'timestamp': datetime.now().isoformat(),
            'searches': {}
        }
        
    def _save_search_cache(self):
        """Save search cache to disk."""
        try:
            self.search_cache['timestamp'] = datetime.now().isoformat()
            os.makedirs(os.path.dirname(self.search_cache_file), exist_ok=True)
            with open(self.search_cache_file, 'w') as f:
                json.dump(self.search_cache, f)
        except Exception as e:
            logger.error(f"Failed to save search cache: {e}")
    
    def search_stocks(self, query, limit=20, market=None):
        """
        Search for stocks by name, symbol, or description.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results to return
            market (str): Market to search in (default: US)
            
        Returns:
            list: List of matching stocks with metadata
        """
        market = market or self.default_market
        
        # Check cache first
        cache_key = f"{query}_{limit}_{market}"
        if cache_key in self.search_cache.get('searches', {}):
            logger.info(f"Using cached results for '{query}'")
            return self.search_cache['searches'][cache_key]
        
        logger.info(f"Searching for stocks matching '{query}'")
        
        # Method 1: YFinance Search API (most comprehensive)
        results = self._search_yahoo_finance(query, limit, market)
        
        # Method 2: Fall back to searching through indices if no results
        if not results:
            logger.info(f"No direct matches found, searching through indices...")
            results = self._search_through_indices(query, limit)
        
        # Cache results
        if 'searches' not in self.search_cache:
            self.search_cache['searches'] = {}
        self.search_cache['searches'][cache_key] = results
        self._save_search_cache()
        
        return results
    
    def _search_yahoo_finance(self, query, limit=20, market="US"):
        """Search using Yahoo Finance API."""
        try:
            query = query.strip()
            
            # Try Yahoo Finance API
            url = f"https://query2.finance.yahoo.com/v1/finance/search"
            params = {
                'q': query,
                'quotesCount': limit,
                'newsCount': 0,
                'enableFuzzyQuery': True,
                'enableEnhancedTrivialQuery': True
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'
            }
            
            r = requests.get(url, params=params, headers=headers)
            data = r.json()
            
            if 'quotes' not in data or not data['quotes']:
                return []
                
            results = []
            for quote in data['quotes']:
                # Filter by market if needed
                if market and market.upper() != "ALL":
                    exchange = quote.get('exchange', '')
                    if market.upper() == "US" and not any(ex in exchange for ex in ["NYSE", "NASDAQ", "AMEX", "BATS", "US"]):
                        continue
                
                symbol = quote.get('symbol', '')
                if not symbol:
                    continue
                    
                results.append({
                    'symbol': symbol,
                    'name': quote.get('shortname', quote.get('longname', '')),
                    'exchange': quote.get('exchange', ''),
                    'type': quote.get('quoteType', ''),
                    'score': quote.get('score', 0)
                })
            
            # Sort by score
            results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching Yahoo Finance: {e}")
            return []
    
    def _search_through_indices(self, query, limit=20):
        """Search through major indices for matching stocks."""
        try:
            from scripts.data_fetcher import DataFetcher
            
            # Create a temporary DataFetcher to get symbols
            fetcher = DataFetcher()
            
            # Get symbols from all major indices
            all_symbols = []
            all_symbols.extend(fetcher.get_symbols_from_index('sp500'))
            all_symbols.extend(fetcher.get_symbols_from_index('nasdaq100'))
            all_symbols.extend(fetcher.get_symbols_from_index('dow30'))
            
            # Remove duplicates
            all_symbols = list(set(all_symbols))
            
            # Match query against symbols or company info
            matches = []
            
            # Create regex pattern for matching
            query_parts = query.lower().split()
            query_pattern = re.compile('|'.join([re.escape(part) for part in query_parts]))
            
            # First pass: direct symbol matches
            direct_matches = [s for s in all_symbols if query.upper() in s]
            
            # Second pass: use yfinance to get company names and info
            with ThreadPoolExecutor(max_workers=10) as executor:
                symbol_chunks = [all_symbols[i:i+50] for i in range(0, len(all_symbols), 50)]
                
                for symbols in symbol_chunks:
                    futures = []
                    for symbol in symbols:
                        futures.append(executor.submit(self._check_symbol_match, symbol, query_pattern))
                    
                    for future in futures:
                        result = future.result()
                        if result:
                            matches.append(result)
                            if len(matches) >= limit:
                                break
                    
                    if len(matches) >= limit:
                        break
            
            # Combine direct matches and info matches
            combined = direct_matches + [m['symbol'] for m in matches if m['symbol'] not in direct_matches]
            combined = combined[:limit]
            
            # Get full info for each match
            results = []
            for symbol in combined:
                try:
                    stock = yf.Ticker(symbol)
                    info = stock.info
                    results.append({
                        'symbol': symbol,
                        'name': info.get('shortName', info.get('longName', '')),
                        'exchange': info.get('exchange', ''),
                        'sector': info.get('sector', ''),
                        'industry': info.get('industry', '')
                    })
                except Exception as e:
                    logger.warning(f"Error getting info for {symbol}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching through indices: {e}")
            return []
    
    def _check_symbol_match(self, symbol, query_pattern):
        """Check if a symbol matches the query based on company info."""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Check name
            name = info.get('shortName', info.get('longName', ''))
            if not name:
                return None
                
            name_lower = name.lower()
            
            # Check for match in name, sector, or industry
            if (query_pattern.search(name_lower) or 
                query_pattern.search(info.get('sector', '').lower()) or 
                query_pattern.search(info.get('industry', '').lower())):
                
                return {
                    'symbol': symbol,
                    'name': name,
                    'exchange': info.get('exchange', ''),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', '')
                }
            
            return None
            
        except Exception:
            return None
    
    def get_stock_details(self, symbol):
        """Get detailed information about a specific stock."""
        try:
            logger.info(f"Getting details for {symbol}")
            
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Basic info
            details = {
                'symbol': symbol,
                'name': info.get('shortName', info.get('longName', symbol)),
                'exchange': info.get('exchange', 'Unknown'),
                'currency': info.get('currency', 'USD'),
                'market_cap': info.get('marketCap', None),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'website': info.get('website', None),
                'description': info.get('longBusinessSummary', None),
                
                # Financial info
                'pe_ratio': info.get('trailingPE', None),
                'eps': info.get('trailingEps', None),
                'dividend_yield': info.get('dividendYield', None) * 100 if info.get('dividendYield') else None,
                'price': info.get('regularMarketPrice', None),
                'price_change': info.get('regularMarketChangePercent', None),
                '52w_low': info.get('fiftyTwoWeekLow', None),
                '52w_high': info.get('fiftyTwoWeekHigh', None),
                
                # Analyst data
                'recommendation': info.get('recommendationKey', None),
                'target_price': info.get('targetMeanPrice', None)
            }
            
            # Check if we have data for this stock
            stock_file = os.path.join(self.data_dir, 'stocks', f"{symbol}.csv")
            if os.path.exists(stock_file):
                details['has_data'] = True
                details['data_path'] = stock_file
            else:
                details['has_data'] = False
            
            return details
            
        except Exception as e:
            logger.error(f"Error getting details for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def add_to_custom_list(self, symbols, config_path=None):
        """Add symbols to custom list in config file."""
        if not config_path and self.config_path:
            config_path = self.config_path
            
        if not config_path:
            logger.error("No config file specified")
            return False
        
        try:
            # Load config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Ensure structure exists
            if 'sources' not in config:
                config['sources'] = {}
            if 'custom_list' not in config['sources']:
                config['sources']['custom_list'] = []
            
            # Add symbols
            custom_list = set(config['sources']['custom_list'])
            for symbol in symbols:
                custom_list.add(symbol)
            
            # Update config
            config['sources']['custom_list'] = sorted(list(custom_list))
            
            # Save config
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
                
            logger.info(f"Added {len(symbols)} symbols to custom list in {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding symbols to custom list: {e}")
            return False
    
    def fetch_selected_stocks(self, symbols):
        """Immediately fetch data for selected stocks."""
        try:
            from scripts.data_fetcher import DataFetcher
            
            # Generate a temporary config
            temp_config = {
                'base_data_dir': 'data',
                'period': '5y',
                'interval': '1d',
                'threads': min(len(symbols), 10),
                'sources': {
                    'sp500': False,
                    'nasdaq100': False,
                    'dow30': False,
                    'custom_list': symbols
                }
            }
            
            # Create temp config file
            temp_config_path = os.path.join(project_root, 'configs', 'temp_fetch_config.yml')
            with open(temp_config_path, 'w') as f:
                yaml.dump(temp_config, f)
            
            # Create fetcher and fetch data
            fetcher = DataFetcher(temp_config_path)
            successful = fetcher.fetch_all_stocks()
            
            if successful:
                logger.info(f"Successfully fetched data for {len(successful)}/{len(symbols)} symbols")
            else:
                logger.warning(f"Failed to fetch any data")
            
            return successful
            
        except Exception as e:
            logger.error(f"Error fetching selected stocks: {e}")
            return []

def main():
    """Command-line interface for stock search."""
    parser = argparse.ArgumentParser(description='Search for any real stock by name, symbol, or description')
    parser.add_argument('query', type=str, help='Search query')
    parser.add_argument('--limit', type=int, default=10, help='Maximum number of results to return')
    parser.add_argument('--market', type=str, default='US', help='Market to search in (US, ALL)')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--details', type=str, help='Get detailed information for a specific symbol')
    parser.add_argument('--add', action='store_true', help='Add found symbols to custom list in config')
    parser.add_argument('--fetch', action='store_true', help='Immediately fetch data for found symbols')
    args = parser.parse_args()
    
    searcher = StockSearch(args.config)
    
    if args.details:
        # Get details for a specific symbol
        details = searcher.get_stock_details(args.details)
        print(json.dumps(details, indent=2))
        return
    
    # Search for stocks
    results = searcher.search_stocks(args.query, args.limit, args.market)
    
    if not results:
        print("No matching stocks found.")
        return
    
    # Display results
    print(f"\nFound {len(results)} stocks matching '{args.query}':")
    print("-" * 80)
    
    for i, stock in enumerate(results):
        print(f"{i+1:2d}. {stock['symbol']:8s} - {stock['name']}")
        if 'exchange' in stock and stock['exchange']:
            print(f"    Exchange: {stock['exchange']}")
        if 'sector' in stock and stock['sector']:
            print(f"    Sector:   {stock['sector']}")
        print()
    
    # Ask which symbols to add/fetch
    if args.add or args.fetch:
        selected = input("Enter the numbers of the stocks to add/fetch (comma-separated, or 'all'): ").strip()
        
        selected_indices = []
        if selected.lower() == 'all':
            selected_indices = range(len(results))
        else:
            try:
                selected_indices = [int(x.strip()) - 1 for x in selected.split(',') if x.strip()]
            except ValueError:
                print("Invalid input. Please enter comma-separated numbers.")
                return
        
        selected_symbols = [results[i]['symbol'] for i in selected_indices if 0 <= i < len(results)]
        
        if not selected_symbols:
            print("No valid stocks selected.")
            return
        
        print(f"Selected: {', '.join(selected_symbols)}")
        
        # Add to custom list
        if args.add and args.config:
            if searcher.add_to_custom_list(selected_symbols, args.config):
                print(f"Added {len(selected_symbols)} symbols to custom list in {args.config}")
        
        # Fetch data
        if args.fetch:
            print(f"Fetching data for {len(selected_symbols)} symbols...")
            successful = searcher.fetch_selected_stocks(selected_symbols)
            print(f"Successfully fetched data for {len(successful)} symbols")

if __name__ == "__main__":
    main() 