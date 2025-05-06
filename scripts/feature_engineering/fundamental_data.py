"""
Fundamental data integration for financial analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import requests
import os
import time
from datetime import datetime, timedelta
import yfinance as yf


class FundamentalDataFetcher:
    """
    Fetch fundamental financial data from various sources.
    """
    
    def __init__(self, api_key: Optional[str] = None, save_path: str = "data/fundamental/"):
        """
        Initialize the fundamental data fetcher.
        
        Args:
            api_key: API key for premium data sources
            save_path: Directory to save downloaded data
        """
        self.api_key = api_key
        self.save_path = save_path
        
        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
    
    def fetch_financials_yf(self, symbols: List[str], quarterly: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Fetch financial statement data from Yahoo Finance.
        
        Args:
            symbols: List of stock symbols
            quarterly: Whether to fetch quarterly statements (True) or annual statements (False)
            
        Returns:
            Dictionary mapping symbols to financial statement DataFrames
        """
        result = {}
        
        for symbol in symbols:
            print(f"Fetching financial statements for {symbol}...")
            
            try:
                # Create Ticker object
                ticker = yf.Ticker(symbol)
                
                # Fetch income statement, balance sheet, and cash flow statement
                if quarterly:
                    income_stmt = ticker.quarterly_income_stmt
                    balance_sheet = ticker.quarterly_balance_sheet
                    cash_flow = ticker.quarterly_cashflow
                else:
                    income_stmt = ticker.income_stmt
                    balance_sheet = ticker.balance_sheet
                    cash_flow = ticker.cashflow
                
                # Combine statements
                financials = pd.concat([income_stmt, balance_sheet, cash_flow])
                
                # Transpose for easier use
                financials = financials.T
                
                # Save data
                if not financials.empty:
                    result[symbol] = financials
                    
                    # Save to file
                    period_str = "quarterly" if quarterly else "annual"
                    file_path = os.path.join(self.save_path, f"{symbol}_financials_{period_str}.csv")
                    financials.to_csv(file_path)
                    print(f"Saved {symbol} financial statements to {file_path}")
                
                # Add a small delay to avoid hitting rate limits
                time.sleep(0.5)
            
            except Exception as e:
                print(f"Error fetching financial statements for {symbol}: {str(e)}")
        
        return result
    
    def fetch_key_metrics_yf(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch key financial metrics from Yahoo Finance.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            DataFrame with key metrics for all symbols
        """
        metrics = []
        
        for symbol in symbols:
            print(f"Fetching key metrics for {symbol}...")
            
            try:
                # Create Ticker object
                ticker = yf.Ticker(symbol)
                
                # Get info
                info = ticker.info
                
                # Extract key metrics
                metric_data = {
                    'symbol': symbol,
                    'date': datetime.now().strftime("%Y-%m-%d"),
                    'marketCap': info.get('marketCap'),
                    'enterpriseValue': info.get('enterpriseValue'),
                    'trailingPE': info.get('trailingPE'),
                    'forwardPE': info.get('forwardPE'),
                    'pegRatio': info.get('pegRatio'),
                    'priceToBook': info.get('priceToBook'),
                    'priceToSales': info.get('priceToSales'),
                    'enterpriseToRevenue': info.get('enterpriseToRevenue'),
                    'enterpriseToEbitda': info.get('enterpriseToEbitda'),
                    'beta': info.get('beta'),
                    'profitMargins': info.get('profitMargins'),
                    'operatingMargins': info.get('operatingMargins'),
                    'returnOnAssets': info.get('returnOnAssets'),
                    'returnOnEquity': info.get('returnOnEquity'),
                    'revenue': info.get('totalRevenue'),
                    'revenuePerShare': info.get('revenuePerShare'),
                    'revenueGrowth': info.get('revenueGrowth'),
                    'grossMargins': info.get('grossMargins'),
                    'ebitda': info.get('ebitda'),
                    'netIncomeToCommon': info.get('netIncomeToCommon'),
                    'earningsGrowth': info.get('earningsGrowth'),
                    'earningsQuarterlyGrowth': info.get('earningsQuarterlyGrowth'),
                    'totalCash': info.get('totalCash'),
                    'totalCashPerShare': info.get('totalCashPerShare'),
                    'totalDebt': info.get('totalDebt'),
                    'debtToEquity': info.get('debtToEquity'),
                    'currentRatio': info.get('currentRatio'),
                    'quickRatio': info.get('quickRatio'),
                    'bookValue': info.get('bookValue'),
                    'sharesOutstanding': info.get('sharesOutstanding'),
                    'dividendRate': info.get('dividendRate'),
                    'dividendYield': info.get('dividendYield'),
                    'payoutRatio': info.get('payoutRatio'),
                    'sector': info.get('sector'),
                    'industry': info.get('industry')
                }
                
                metrics.append(metric_data)
                
                # Add a small delay to avoid hitting rate limits
                time.sleep(0.5)
            
            except Exception as e:
                print(f"Error fetching key metrics for {symbol}: {str(e)}")
        
        # Create DataFrame
        if metrics:
            df = pd.DataFrame(metrics)
            
            # Save to file
            file_path = os.path.join(self.save_path, "key_metrics.csv")
            df.to_csv(file_path, index=False)
            print(f"Saved key metrics to {file_path}")
            
            return df
        
        return pd.DataFrame()
    
    def fetch_earnings_yf(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch earnings data from Yahoo Finance.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to earnings DataFrames
        """
        result = {}
        
        for symbol in symbols:
            print(f"Fetching earnings data for {symbol}...")
            
            try:
                # Create Ticker object
                ticker = yf.Ticker(symbol)
                
                # Get earnings
                earnings = ticker.earnings
                
                if not earnings.empty:
                    result[symbol] = earnings
                    
                    # Save to file
                    file_path = os.path.join(self.save_path, f"{symbol}_earnings.csv")
                    earnings.to_csv(file_path)
                    print(f"Saved {symbol} earnings to {file_path}")
                
                # Get earnings dates and expectations
                earnings_dates = ticker.earnings_dates
                
                if earnings_dates is not None and not earnings_dates.empty:
                    # Save to file
                    file_path = os.path.join(self.save_path, f"{symbol}_earnings_dates.csv")
                    earnings_dates.to_csv(file_path)
                    print(f"Saved {symbol} earnings dates to {file_path}")
                
                # Add a small delay to avoid hitting rate limits
                time.sleep(0.5)
            
            except Exception as e:
                print(f"Error fetching earnings data for {symbol}: {str(e)}")
        
        return result
    
    def fetch_analyst_recommendations(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch analyst recommendations from Yahoo Finance.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to analyst recommendation DataFrames
        """
        result = {}
        
        for symbol in symbols:
            print(f"Fetching analyst recommendations for {symbol}...")
            
            try:
                # Create Ticker object
                ticker = yf.Ticker(symbol)
                
                # Get recommendations
                recommendations = ticker.recommendations
                
                if recommendations is not None and not recommendations.empty:
                    result[symbol] = recommendations
                    
                    # Save to file
                    file_path = os.path.join(self.save_path, f"{symbol}_recommendations.csv")
                    recommendations.to_csv(file_path)
                    print(f"Saved {symbol} recommendations to {file_path}")
                
                # Add a small delay to avoid hitting rate limits
                time.sleep(0.5)
            
            except Exception as e:
                print(f"Error fetching analyst recommendations for {symbol}: {str(e)}")
        
        return result
    
    def fetch_all_fundamental_data(self, symbols: List[str], 
                                  include_quarterly: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch all available fundamental data.
        
        Args:
            symbols: List of stock symbols
            include_quarterly: Whether to include quarterly financial statements
            
        Returns:
            Dictionary with all fetched data
        """
        result = {}
        
        # Fetch key metrics
        metrics_df = self.fetch_key_metrics_yf(symbols)
        if not metrics_df.empty:
            result['key_metrics'] = metrics_df
        
        # Fetch annual financial statements
        financials = self.fetch_financials_yf(symbols, quarterly=False)
        if financials:
            result['annual_financials'] = financials
        
        # Fetch quarterly financial statements if requested
        if include_quarterly:
            quarterly_financials = self.fetch_financials_yf(symbols, quarterly=True)
            if quarterly_financials:
                result['quarterly_financials'] = quarterly_financials
        
        # Fetch earnings data
        earnings = self.fetch_earnings_yf(symbols)
        if earnings:
            result['earnings'] = earnings
        
        # Fetch analyst recommendations
        recommendations = self.fetch_analyst_recommendations(symbols)
        if recommendations:
            result['recommendations'] = recommendations
        
        return result


def calculate_financial_ratios(fundamental_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate financial ratios from fundamental data.
    
    Args:
        fundamental_data: Dictionary with fundamental data
        
    Returns:
        DataFrame with calculated financial ratios
    """
    # Extract key metrics
    if 'key_metrics' not in fundamental_data:
        print("Key metrics not found in fundamental data")
        return pd.DataFrame()
    
    key_metrics = fundamental_data['key_metrics']
    
    # Create DataFrame for ratios
    symbols = key_metrics['symbol'].unique()
    latest_date = key_metrics['date'].max()
    
    ratios_list = []
    
    for symbol in symbols:
        # Get latest metrics for the symbol
        symbol_metrics = key_metrics[
            (key_metrics['symbol'] == symbol) & 
            (key_metrics['date'] == latest_date)
        ].iloc[0]
        
        # Calculate additional ratios
        ratio_data = {
            'symbol': symbol,
            'date': latest_date
        }
        
        # Copy existing ratios
        for ratio in ['trailingPE', 'forwardPE', 'pegRatio', 'priceToBook', 
                     'priceToSales', 'debtToEquity', 'currentRatio', 'quickRatio']:
            ratio_data[ratio] = symbol_metrics.get(ratio)
        
        # Calculate additional ratios if data is available
        
        # Enterprise Value to EBITDA
        if symbol_metrics.get('enterpriseValue') and symbol_metrics.get('ebitda') and symbol_metrics.get('ebitda') != 0:
            ratio_data['evToEbitda'] = symbol_metrics.get('enterpriseValue') / symbol_metrics.get('ebitda')
        
        # Return on Invested Capital (approximation)
        if (symbol_metrics.get('netIncomeToCommon') and symbol_metrics.get('totalDebt') and 
            symbol_metrics.get('marketCap') and 
            (symbol_metrics.get('totalDebt') + symbol_metrics.get('marketCap')) != 0):
            ratio_data['roic'] = symbol_metrics.get('netIncomeToCommon') / (symbol_metrics.get('totalDebt') + symbol_metrics.get('marketCap'))
        
        # Cash Flow to Debt
        if symbol_metrics.get('cashflow') and symbol_metrics.get('totalDebt') and symbol_metrics.get('totalDebt') != 0:
            ratio_data['cashFlowToDebt'] = symbol_metrics.get('cashflow') / symbol_metrics.get('totalDebt')
        
        # Interest Coverage Ratio (approximation)
        if symbol_metrics.get('ebitda') and symbol_metrics.get('interestExpense') and symbol_metrics.get('interestExpense') != 0:
            ratio_data['interestCoverage'] = symbol_metrics.get('ebitda') / symbol_metrics.get('interestExpense')
        
        # Price to Cash Flow
        if symbol_metrics.get('marketCap') and symbol_metrics.get('cashflow') and symbol_metrics.get('cashflow') != 0:
            ratio_data['priceToCashFlow'] = symbol_metrics.get('marketCap') / symbol_metrics.get('cashflow')
        
        # Add to list
        ratios_list.append(ratio_data)
    
    # Create DataFrame
    if ratios_list:
        ratios_df = pd.DataFrame(ratios_list)
        
        # Save to file
        file_path = os.path.join(fundamental_data['key_metrics'].iloc[0]['save_path'] if 'save_path' in fundamental_data['key_metrics'].columns else "data/fundamental/", "financial_ratios.csv")
        ratios_df.to_csv(file_path, index=False)
        print(f"Saved financial ratios to {file_path}")
        
        return ratios_df
    
    return pd.DataFrame()


def combine_price_and_fundamental_data(price_data: Dict[str, pd.DataFrame], 
                                      fundamental_data: Dict[str, pd.DataFrame], 
                                      resample: str = 'Q') -> Dict[str, pd.DataFrame]:
    """
    Combine price data with fundamental data.
    
    Args:
        price_data: Dictionary mapping symbols to price DataFrames
        fundamental_data: Dictionary with fundamental data
        resample: Frequency to resample price data to ('Q' for quarterly, 'Y' for yearly)
        
    Returns:
        Dictionary mapping symbols to combined DataFrames
    """
    result = {}
    
    # Extract key metrics
    if 'key_metrics' not in fundamental_data:
        print("Key metrics not found in fundamental data")
        return result
    
    key_metrics = fundamental_data['key_metrics']
    
    # Process each symbol
    for symbol, price_df in price_data.items():
        print(f"Combining price and fundamental data for {symbol}...")
        
        try:
            # Resample price data
            price_resampled = price_df.resample(resample).last()
            
            # Get fundamental data for the symbol
            symbol_metrics = key_metrics[key_metrics['symbol'] == symbol]
            
            if symbol_metrics.empty:
                print(f"No fundamental data found for {symbol}")
                continue
            
            # Convert date columns to datetime
            symbol_metrics['date'] = pd.to_datetime(symbol_metrics['date'])
            symbol_metrics.set_index('date', inplace=True)
            
            # Align indices
            # Get all dates from both dataframes
            all_dates = price_resampled.index.union(symbol_metrics.index)
            
            # Reindex both dataframes with the union of dates
            price_aligned = price_resampled.reindex(all_dates)
            metrics_aligned = symbol_metrics.reindex(all_dates)
            
            # Forward fill fundamental data (fundamentals don't change as frequently)
            metrics_aligned = metrics_aligned.ffill()
            
            # Merge the dataframes
            combined = pd.concat([price_aligned, metrics_aligned.drop(columns=['symbol'], errors='ignore')], axis=1)
            
            # Clean up NaN values
            combined = combined.dropna(subset=['Close'])  # Ensure we have price data
            
            # Add to result
            result[symbol] = combined
            
            # Save to file
            file_path = os.path.join(symbol_metrics.iloc[0]['save_path'] if 'save_path' in symbol_metrics.columns else "data/fundamental/", f"{symbol}_combined.csv")
            combined.to_csv(file_path)
            print(f"Saved combined data for {symbol} to {file_path}")
        
        except Exception as e:
            print(f"Error combining data for {symbol}: {str(e)}")
    
    return result


def calculate_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate additional features from fundamental data.
    
    Args:
        df: DataFrame with combined price and fundamental data
        
    Returns:
        DataFrame with additional fundamental features
    """
    result = df.copy()
    
    # Calculate price-based ratios
    if all(col in result.columns for col in ['Close', 'marketCap', 'sharesOutstanding']):
        # Calculate implied price from market cap and shares outstanding
        result['impliedPrice'] = result['marketCap'] / result['sharesOutstanding']
        
        # Calculate price deviation from implied price
        result['priceDeviation'] = (result['Close'] - result['impliedPrice']) / result['impliedPrice']
    
    # Calculate momentum of fundamental metrics
    for col in ['trailingPE', 'priceToBook', 'priceToSales', 'returnOnEquity', 'returnOnAssets']:
        if col in result.columns:
            # Calculate 1-period and 4-period changes
            result[f'{col}_change_1'] = result[col].pct_change(1)
            result[f'{col}_change_4'] = result[col].pct_change(4)
    
    # Calculate earnings surprise
    if all(col in result.columns for col in ['epsActual', 'epsEstimate']):
        result['earningsSurprise'] = (result['epsActual'] - result['epsEstimate']) / result['epsEstimate']
    
    # Calculate PEG ratio if growth data is available
    if all(col in result.columns for col in ['trailingPE', 'earningsGrowth']) and 'earningsGrowth' in result.columns:
        result['pegRatio'] = result['trailingPE'] / (result['earningsGrowth'] * 100)
    
    # Calculate Earnings Yield
    if 'trailingPE' in result.columns:
        result['earningsYield'] = 1 / result['trailingPE']
    
    # Calculate Free Cash Flow Yield
    if all(col in result.columns for col in ['freeCashFlow', 'marketCap']) and 'marketCap' in result.columns and result['marketCap'].any():
        result['fcfYield'] = result['freeCashFlow'] / result['marketCap']
    
    # Calculate Altman Z-Score (bankruptcy risk)
    # Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    # X1 = Working Capital / Total Assets
    # X2 = Retained Earnings / Total Assets
    # X3 = EBIT / Total Assets
    # X4 = Market Value of Equity / Total Liabilities
    # X5 = Sales / Total Assets
    if all(col in result.columns for col in ['workingCapital', 'totalAssets', 'retainedEarnings', 
                                            'ebit', 'marketCap', 'totalLiabilities', 'revenue']):
        
        # Calculate components
        X1 = result['workingCapital'] / result['totalAssets']
        X2 = result['retainedEarnings'] / result['totalAssets']
        X3 = result['ebit'] / result['totalAssets']
        X4 = result['marketCap'] / result['totalLiabilities']
        X5 = result['revenue'] / result['totalAssets']
        
        # Calculate Z-Score
        result['altmanZScore'] = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        
        # Interpret Z-Score
        result['bankruptcyRisk'] = 'Unknown'
        result.loc[result['altmanZScore'] < 1.8, 'bankruptcyRisk'] = 'High'
        result.loc[(result['altmanZScore'] >= 1.8) & (result['altmanZScore'] < 3.0), 'bankruptcyRisk'] = 'Medium'
        result.loc[result['altmanZScore'] >= 3.0, 'bankruptcyRisk'] = 'Low'
    
    return result


def get_sector_industry_data(symbols: List[str]) -> pd.DataFrame:
    """
    Get sector and industry data for a list of symbols.
    
    Args:
        symbols: List of stock symbols
        
    Returns:
        DataFrame with sector and industry data
    """
    data = []
    
    for symbol in symbols:
        try:
            # Create Ticker object
            ticker = yf.Ticker(symbol)
            
            # Get info
            info = ticker.info
            
            # Extract sector and industry
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            data.append({
                'symbol': symbol,
                'sector': sector,
                'industry': industry
            })
            
            # Add a small delay
            time.sleep(0.5)
        
        except Exception as e:
            print(f"Error getting sector/industry for {symbol}: {str(e)}")
            data.append({
                'symbol': symbol,
                'sector': 'Unknown',
                'industry': 'Unknown'
            })
    
    # Create DataFrame
    return pd.DataFrame(data)


def add_sector_industry_features(df: pd.DataFrame, sector_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add sector and industry features to a DataFrame.
    
    Args:
        df: DataFrame with ticker data
        sector_data: DataFrame with sector and industry data
        
    Returns:
        DataFrame with added sector and industry features
    """
    # Check if 'symbol' column exists in df
    if 'symbol' not in df.columns:
        print("Symbol column not found in DataFrame")
        return df
    
    result = df.copy()
    
    # Merge sector data
    result = pd.merge(result, sector_data, on='symbol', how='left')
    
    # One-hot encode sector and industry
    if 'sector' in result.columns:
        sector_dummies = pd.get_dummies(result['sector'], prefix='sector')
        result = pd.concat([result, sector_dummies], axis=1)
    
    if 'industry' in result.columns:
        industry_dummies = pd.get_dummies(result['industry'], prefix='industry')
        result = pd.concat([result, industry_dummies], axis=1)
    
    return result


def add_fundamental_features(df: pd.DataFrame, 
                            fundamendal_data_fetcher: FundamentalDataFetcher,
                            symbols: List[str] = None) -> pd.DataFrame:
    """
    Add fundamental features to a DataFrame.
    
    Args:
        df: DataFrame with price data
        fundamendal_data_fetcher: FundamentalDataFetcher instance
        symbols: List of stock symbols (if None, try to extract from df)
        
    Returns:
        DataFrame with added fundamental features
    """
    # Extract symbols if not provided
    if symbols is None:
        if 'symbol' in df.columns:
            symbols = df['symbol'].unique().tolist()
        else:
            print("Symbols not provided and cannot be extracted from DataFrame")
            return df
    
    # Fetch fundamental data
    fundamental_data = fundamendal_data_fetcher.fetch_all_fundamental_data(symbols)
    
    # Prepare price data dictionary
    price_data = {}
    if 'symbol' in df.columns:
        for symbol in symbols:
            symbol_df = df[df['symbol'] == symbol].copy()
            if isinstance(symbol_df.index, pd.DatetimeIndex):
                price_data[symbol] = symbol_df
            else:
                # Try to convert index to datetime
                try:
                    if 'date' in symbol_df.columns:
                        symbol_df['date'] = pd.to_datetime(symbol_df['date'])
                        symbol_df.set_index('date', inplace=True)
                    else:
                        symbol_df.index = pd.to_datetime(symbol_df.index)
                    price_data[symbol] = symbol_df
                except:
                    print(f"Could not convert index to datetime for {symbol}")
                    continue
    else:
        # Assume df is a single symbol's data
        if isinstance(df.index, pd.DatetimeIndex):
            price_data[symbols[0]] = df
        else:
            # Try to convert index to datetime
            try:
                if 'date' in df.columns:
                    df_copy = df.copy()
                    df_copy['date'] = pd.to_datetime(df_copy['date'])
                    df_copy.set_index('date', inplace=True)
                else:
                    df_copy = df.copy()
                    df_copy.index = pd.to_datetime(df_copy.index)
                price_data[symbols[0]] = df_copy
            except:
                print("Could not convert index to datetime")
                return df
    
    # Combine price and fundamental data
    combined_data = combine_price_and_fundamental_data(price_data, fundamental_data)
    
    # Return combined data for the first symbol
    if combined_data and symbols[0] in combined_data:
        return combined_data[symbols[0]]
    
    return df 