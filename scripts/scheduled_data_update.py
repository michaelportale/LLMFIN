#!/usr/bin/env python3
"""
Scheduled data update script.
Run this script with cron or another scheduler to keep your data updated.

Example crontab entry for daily update at 6pm:
0 18 * * * /path/to/venv/bin/python /path/to/scripts/scheduled_data_update.py --config /path/to/configs/data_fetcher_config.yml
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import yaml
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Import the DataFetcher
from scripts.data_fetcher import DataFetcher

# Set up logging
log_dir = os.path.join(project_root, 'logs')
Path(log_dir).mkdir(parents=True, exist_ok=True)

log_file = os.path.join(log_dir, f'data_update_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def update_data(config_path):
    """Update all stock data based on configuration."""
    logger.info(f"Starting scheduled data update with config: {config_path}")
    
    try:
        # Create data fetcher instance
        fetcher = DataFetcher(config_path)
        
        # Fetch all stocks
        successful_symbols = fetcher.fetch_all_stocks()
        
        # Log summary
        logger.info(f"Data update completed. Successfully updated {len(successful_symbols)} symbols.")
        
        # Record success in update log
        update_log_path = os.path.join(project_root, 'data', 'update_history.yml')
        
        # Load existing history or create new
        if os.path.exists(update_log_path):
            with open(update_log_path, 'r') as f:
                history = yaml.safe_load(f) or {}
        else:
            history = {'updates': []}
            
        # Add this update to history
        history['updates'].append({
            'timestamp': datetime.now().isoformat(),
            'successful_symbols': len(successful_symbols),
            'config_used': config_path,
            'log_file': log_file
        })
        
        # Keep only last 30 updates in history
        if len(history['updates']) > 30:
            history['updates'] = history['updates'][-30:]
            
        # Save history
        with open(update_log_path, 'w') as f:
            yaml.dump(history, f)
            
        return True
        
    except Exception as e:
        logger.error(f"Error during scheduled update: {e}", exc_info=True)
        return False

def main():
    """Main entry point for scheduled data update."""
    parser = argparse.ArgumentParser(description='Scheduled data update for stock data')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file')
    args = parser.parse_args()
    
    # Run the update
    success = update_data(args.config)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 