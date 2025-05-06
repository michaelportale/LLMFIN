"""
API routes for stock search functionality.
Allows users to search for and add any stock to the tracking list.
"""

import os
import json
import logging
from flask import Blueprint, jsonify, request, current_app
from scripts.stock_search import StockSearch

# Set up logger
logger = logging.getLogger(__name__)

# Create blueprint
stock_search_bp = Blueprint('stock_search', __name__)

@stock_search_bp.route('/search', methods=['GET'])
def search_stocks():
    """
    Search for stocks by name, symbol, or description.
    
    Query parameters:
        query (str): Search query
        limit (int): Maximum number of results (default: 20)
        market (str): Market to search in (default: US)
    
    Returns:
        JSON response with matching stocks
    """
    query = request.args.get('query', '')
    limit = int(request.args.get('limit', 20))
    market = request.args.get('market', 'US')
    
    if not query:
        return jsonify({'error': 'Search query is required'}), 400
        
    try:
        # Get the config path
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'configs',
            'data_fetcher_config.yml'
        )
        
        searcher = StockSearch(config_path)
        results = searcher.search_stocks(query, limit, market)
        
        return jsonify({
            'results': results,
            'count': len(results),
            'query': query
        })
        
    except Exception as e:
        logger.error(f"Error searching stocks: {e}", exc_info=True)
        return jsonify({'error': f'Error searching stocks: {str(e)}'}), 500

@stock_search_bp.route('/details/<symbol>', methods=['GET'])
def get_stock_details(symbol):
    """
    Get detailed information about a specific stock.
    
    Parameters:
        symbol (str): Stock symbol
    
    Returns:
        JSON response with stock details
    """
    try:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'configs',
            'data_fetcher_config.yml'
        )
        
        searcher = StockSearch(config_path)
        details = searcher.get_stock_details(symbol)
        
        return jsonify(details)
        
    except Exception as e:
        logger.error(f"Error getting stock details: {e}", exc_info=True)
        return jsonify({'error': f'Error getting stock details: {str(e)}'}), 500

@stock_search_bp.route('/add', methods=['POST'])
def add_stocks():
    """
    Add stocks to the tracking list.
    
    Request body:
        symbols (list): List of stock symbols to add
    
    Returns:
        JSON response with success status
    """
    data = request.get_json()
    
    if not data or 'symbols' not in data or not isinstance(data['symbols'], list):
        return jsonify({'error': 'Invalid request. Expected "symbols" array in JSON body'}), 400
        
    symbols = data['symbols']
    
    if not symbols:
        return jsonify({'error': 'No symbols provided'}), 400
        
    try:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'configs',
            'data_fetcher_config.yml'
        )
        
        searcher = StockSearch(config_path)
        success = searcher.add_to_custom_list(symbols, config_path)
        
        if success:
            # Optionally fetch data for these symbols immediately
            fetch = request.args.get('fetch', 'false').lower() == 'true'
            fetched = []
            
            if fetch:
                fetched = searcher.fetch_selected_stocks(symbols)
            
            return jsonify({
                'success': True,
                'message': f'Added {len(symbols)} symbols to tracking list',
                'symbols': symbols,
                'fetched': fetched if fetch else []
            })
        else:
            return jsonify({'error': 'Failed to add symbols to tracking list'}), 500
            
    except Exception as e:
        logger.error(f"Error adding stocks: {e}", exc_info=True)
        return jsonify({'error': f'Error adding stocks: {str(e)}'}), 500

@stock_search_bp.route('/fetch', methods=['POST'])
def fetch_stocks():
    """
    Fetch data for specified stocks.
    
    Request body:
        symbols (list): List of stock symbols to fetch
    
    Returns:
        JSON response with success status
    """
    data = request.get_json()
    
    if not data or 'symbols' not in data or not isinstance(data['symbols'], list):
        return jsonify({'error': 'Invalid request. Expected "symbols" array in JSON body'}), 400
        
    symbols = data['symbols']
    
    if not symbols:
        return jsonify({'error': 'No symbols provided'}), 400
        
    try:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'configs',
            'data_fetcher_config.yml'
        )
        
        searcher = StockSearch(config_path)
        successful = searcher.fetch_selected_stocks(symbols)
        
        return jsonify({
            'success': True,
            'message': f'Fetched data for {len(successful)} out of {len(symbols)} symbols',
            'fetched': successful
        })
        
    except Exception as e:
        logger.error(f"Error fetching stocks: {e}", exc_info=True)
        return jsonify({'error': f'Error fetching stocks: {str(e)}'}), 500 