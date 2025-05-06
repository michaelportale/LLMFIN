"""
Sector and industry analysis tools for financial data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import yfinance as yf
import os
from datetime import datetime, timedelta


def fetch_sector_data(tickers: List[str], 
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch sector and industry data for a list of tickers.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for price data (YYYY-MM-DD)
        end_date: End date for price data (YYYY-MM-DD)
        
    Returns:
        Dictionary with sector and industry data
    """
    # Initialize result dictionaries
    sector_info = {}
    price_data = {}
    ticker_sector_map = {}
    ticker_industry_map = {}
    
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
        
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    # Fetch data for each ticker
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract sector and industry information
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            # Update mappings
            ticker_sector_map[ticker] = sector
            ticker_industry_map[ticker] = industry
            
            # Initialize sector and industry in the result dictionary if needed
            if sector not in sector_info:
                sector_info[sector] = {'industries': {}, 'tickers': []}
                
            if industry not in sector_info[sector]['industries']:
                sector_info[sector]['industries'][industry] = []
            
            # Add ticker to appropriate lists
            sector_info[sector]['tickers'].append(ticker)
            sector_info[sector]['industries'][industry].append(ticker)
            
            # Fetch historical price data
            hist = stock.history(start=start_date, end=end_date)
            if not hist.empty:
                price_data[ticker] = hist
                
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
    
    # Calculate sector and industry returns
    sector_returns = calculate_sector_returns(price_data, ticker_sector_map)
    industry_returns = calculate_industry_returns(price_data, ticker_industry_map)
    
    return {
        'sector_info': sector_info,
        'price_data': price_data,
        'ticker_sector_map': ticker_sector_map,
        'ticker_industry_map': ticker_industry_map,
        'sector_returns': sector_returns,
        'industry_returns': industry_returns
    }


def calculate_sector_returns(price_data: Dict[str, pd.DataFrame], 
                           ticker_sector_map: Dict[str, str]) -> pd.DataFrame:
    """
    Calculate daily returns for each sector.
    
    Args:
        price_data: Dictionary with price data for each ticker
        ticker_sector_map: Mapping from tickers to sectors
        
    Returns:
        DataFrame with sector returns
    """
    # Extract closing prices
    close_prices = {}
    for ticker, data in price_data.items():
        if 'Close' in data.columns:
            close_prices[ticker] = data['Close']
    
    # Convert to DataFrame
    if close_prices:
        prices_df = pd.DataFrame(close_prices)
        
        # Calculate daily returns
        returns_df = prices_df.pct_change().dropna()
        
        # Group by sector
        sector_returns = {}
        for sector in set(ticker_sector_map.values()):
            sector_tickers = [t for t, s in ticker_sector_map.items() if s == sector and t in returns_df.columns]
            if sector_tickers:
                # Equal-weighted sector return
                sector_returns[sector] = returns_df[sector_tickers].mean(axis=1)
        
        return pd.DataFrame(sector_returns)
    
    return pd.DataFrame()


def calculate_industry_returns(price_data: Dict[str, pd.DataFrame], 
                             ticker_industry_map: Dict[str, str]) -> pd.DataFrame:
    """
    Calculate daily returns for each industry.
    
    Args:
        price_data: Dictionary with price data for each ticker
        ticker_industry_map: Mapping from tickers to industries
        
    Returns:
        DataFrame with industry returns
    """
    # Extract closing prices
    close_prices = {}
    for ticker, data in price_data.items():
        if 'Close' in data.columns:
            close_prices[ticker] = data['Close']
    
    # Convert to DataFrame
    if close_prices:
        prices_df = pd.DataFrame(close_prices)
        
        # Calculate daily returns
        returns_df = prices_df.pct_change().dropna()
        
        # Group by industry
        industry_returns = {}
        for industry in set(ticker_industry_map.values()):
            industry_tickers = [t for t, i in ticker_industry_map.items() if i == industry and t in returns_df.columns]
            if industry_tickers:
                # Equal-weighted industry return
                industry_returns[industry] = returns_df[industry_tickers].mean(axis=1)
        
        return pd.DataFrame(industry_returns)
    
    return pd.DataFrame()


def plot_sector_performance(sector_returns: pd.DataFrame,
                          period: str = 'all',
                          title: str = "Sector Performance",
                          figsize: Tuple[int, int] = (12, 8),
                          output_file: Optional[str] = None) -> None:
    """
    Plot sector performance over time.
    
    Args:
        sector_returns: DataFrame with sector returns
        period: Time period to plot ('all', 'ytd', '1m', '3m', '6m', '1y')
        title: Plot title
        figsize: Figure size
        output_file: Path to save the output file (if None, display the plot)
    """
    if sector_returns.empty:
        print("No sector returns data available")
        return
    
    # Filter data based on period
    if period != 'all':
        today = pd.Timestamp.now()
        if period == 'ytd':
            start_date = pd.Timestamp(year=today.year, month=1, day=1)
        elif period == '1m':
            start_date = today - pd.DateOffset(months=1)
        elif period == '3m':
            start_date = today - pd.DateOffset(months=3)
        elif period == '6m':
            start_date = today - pd.DateOffset(months=6)
        elif period == '1y':
            start_date = today - pd.DateOffset(years=1)
        else:
            start_date = sector_returns.index[0]
        
        sector_returns = sector_returns[sector_returns.index >= start_date]
    
    # Calculate cumulative returns
    cumulative_returns = (1 + sector_returns).cumprod() - 1
    
    # Create plot
    plt.figure(figsize=figsize)
    
    for sector in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[sector] * 100, label=sector)
    
    plt.title(f"{title} ({period.upper()})", fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add recent performance for each sector
    recent_returns = cumulative_returns.iloc[-1].sort_values(ascending=False) * 100
    
    # Create a table with recent performance
    table_text = ""
    for sector, ret in recent_returns.items():
        table_text += f"{sector}: {ret:.2f}%\n"
    
    plt.annotate(
        table_text, 
        xy=(0.02, 0.02), 
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        verticalalignment='bottom',
        fontsize=9
    )
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_sector_correlation(sector_returns: pd.DataFrame,
                          title: str = "Sector Return Correlation",
                          figsize: Tuple[int, int] = (10, 8),
                          output_file: Optional[str] = None) -> None:
    """
    Plot correlation matrix of sector returns.
    
    Args:
        sector_returns: DataFrame with sector returns
        title: Plot title
        figsize: Figure size
        output_file: Path to save the output file (if None, display the plot)
    """
    if sector_returns.empty:
        print("No sector returns data available")
        return
    
    # Calculate correlation matrix
    corr_matrix = sector_returns.corr()
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr_matrix, 
        mask=mask, 
        cmap=cmap, 
        vmax=1, 
        vmin=-1, 
        center=0,
        square=True, 
        linewidths=.5, 
        annot=True, 
        fmt=".2f",
        cbar_kws={"shrink": .8}
    )
    
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_sector_volatility(sector_returns: pd.DataFrame,
                         window: int = 20,
                         title: str = "Sector Volatility (Rolling 20-day)",
                         figsize: Tuple[int, int] = (12, 8),
                         output_file: Optional[str] = None) -> None:
    """
    Plot rolling volatility of sector returns.
    
    Args:
        sector_returns: DataFrame with sector returns
        window: Rolling window size
        title: Plot title
        figsize: Figure size
        output_file: Path to save the output file (if None, display the plot)
    """
    if sector_returns.empty:
        print("No sector returns data available")
        return
    
    # Calculate rolling volatility
    rolling_vol = sector_returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    # Create plot
    plt.figure(figsize=figsize)
    
    for sector in rolling_vol.columns:
        plt.plot(rolling_vol.index, rolling_vol[sector] * 100, label=sector)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Annualized Volatility (%)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add recent volatility for each sector
    recent_vol = rolling_vol.iloc[-1].sort_values(ascending=False) * 100
    
    # Create a table with recent volatility
    table_text = ""
    for sector, vol in recent_vol.items():
        table_text += f"{sector}: {vol:.2f}%\n"
    
    plt.annotate(
        table_text, 
        xy=(0.02, 0.02), 
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        verticalalignment='bottom',
        fontsize=9
    )
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def calculate_sector_metrics(sector_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate various performance metrics for each sector.
    
    Args:
        sector_returns: DataFrame with sector returns
        
    Returns:
        DataFrame with sector metrics
    """
    if sector_returns.empty:
        return pd.DataFrame()
    
    # Calculate metrics
    metrics = {}
    
    # Annualized return
    metrics['annualized_return'] = sector_returns.mean() * 252 * 100
    
    # Annualized volatility
    metrics['annualized_volatility'] = sector_returns.std() * np.sqrt(252) * 100
    
    # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
    metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['annualized_volatility']
    
    # Maximum drawdown
    cumulative_returns = (1 + sector_returns).cumprod()
    max_drawdowns = {}
    
    for sector in sector_returns.columns:
        # Calculate running maximum
        running_max = cumulative_returns[sector].cummax()
        
        # Calculate drawdown
        drawdown = (cumulative_returns[sector] - running_max) / running_max
        
        # Get maximum drawdown
        max_drawdowns[sector] = drawdown.min() * 100
    
    metrics['max_drawdown'] = pd.Series(max_drawdowns)
    
    # Percentage of positive days
    metrics['positive_days_pct'] = (sector_returns > 0).mean() * 100
    
    # Skewness
    metrics['skewness'] = sector_returns.skew()
    
    # Kurtosis
    metrics['kurtosis'] = sector_returns.kurtosis()
    
    # Value at Risk (95%)
    metrics['var_95'] = sector_returns.quantile(0.05) * 100
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    return metrics_df


def plot_sector_metrics(sector_metrics: pd.DataFrame,
                      title: str = "Sector Performance Metrics",
                      figsize: Tuple[int, int] = (15, 10),
                      output_file: Optional[str] = None) -> None:
    """
    Plot sector performance metrics.
    
    Args:
        sector_metrics: DataFrame with sector metrics
        title: Plot title
        figsize: Figure size
        output_file: Path to save the output file (if None, display the plot)
    """
    if sector_metrics.empty:
        print("No sector metrics available")
        return
    
    # Create plot with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Flatten axes for easier access
    axes = axes.flatten()
    
    # 1. Annualized Return vs Volatility
    axes[0].scatter(
        sector_metrics['annualized_volatility'],
        sector_metrics['annualized_return'],
        s=100
    )
    
    # Add sector labels
    for sector in sector_metrics.index:
        axes[0].annotate(
            sector,
            (sector_metrics.loc[sector, 'annualized_volatility'],
             sector_metrics.loc[sector, 'annualized_return']),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    axes[0].set_title('Return vs Volatility')
    axes[0].set_xlabel('Annualized Volatility (%)')
    axes[0].set_ylabel('Annualized Return (%)')
    axes[0].grid(True, alpha=0.3)
    
    # Add y=x line for reference
    min_val = min(sector_metrics['annualized_volatility'].min(), sector_metrics['annualized_return'].min())
    max_val = max(sector_metrics['annualized_volatility'].max(), sector_metrics['annualized_return'].max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    # 2. Sharpe Ratio
    sectors = sector_metrics.index
    sharpe_ratios = sector_metrics['sharpe_ratio'].sort_values(ascending=False)
    
    axes[1].barh(sharpe_ratios.index, sharpe_ratios.values)
    axes[1].set_title('Sharpe Ratio')
    axes[1].set_xlabel('Sharpe Ratio')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Maximum Drawdown
    max_drawdowns = sector_metrics['max_drawdown'].sort_values()
    
    axes[2].barh(max_drawdowns.index, max_drawdowns.values)
    axes[2].set_title('Maximum Drawdown')
    axes[2].set_xlabel('Maximum Drawdown (%)')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Positive Days Percentage
    positive_days = sector_metrics['positive_days_pct'].sort_values(ascending=False)
    
    axes[3].barh(positive_days.index, positive_days.values)
    axes[3].set_title('Percentage of Positive Days')
    axes[3].set_xlabel('Positive Days (%)')
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def analyze_sector_rotation(sector_returns: pd.DataFrame,
                          window: int = 126,  # About 6 months
                          top_n: int = 3) -> pd.DataFrame:
    """
    Analyze sector rotation by tracking top performing sectors over time.
    
    Args:
        sector_returns: DataFrame with sector returns
        window: Rolling window for performance calculation
        top_n: Number of top sectors to track
        
    Returns:
        DataFrame with top sectors for each time period
    """
    if sector_returns.empty:
        return pd.DataFrame()
    
    # Calculate rolling returns
    rolling_returns = (1 + sector_returns).rolling(window=window).apply(
        lambda x: (1 + x).prod() - 1, raw=True
    )
    
    # For each date, find the top N sectors
    top_sectors = pd.DataFrame(index=rolling_returns.index, columns=[f'rank_{i+1}' for i in range(top_n)])
    
    for date in rolling_returns.index:
        date_returns = rolling_returns.loc[date].dropna()
        
        if not date_returns.empty:
            # Sort sectors by return
            sorted_sectors = date_returns.sort_values(ascending=False)
            
            # Get top N sectors
            for i in range(min(top_n, len(sorted_sectors))):
                top_sectors.loc[date, f'rank_{i+1}'] = sorted_sectors.index[i]
    
    return top_sectors


def plot_sector_rotation(sector_rotation: pd.DataFrame,
                       title: str = "Sector Rotation Analysis",
                       figsize: Tuple[int, int] = (12, 8),
                       output_file: Optional[str] = None) -> None:
    """
    Visualize sector rotation over time.
    
    Args:
        sector_rotation: DataFrame with sector rotation data
        title: Plot title
        figsize: Figure size
        output_file: Path to save the output file (if None, display the plot)
    """
    if sector_rotation.empty:
        print("No sector rotation data available")
        return
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Get all unique sectors
    all_sectors = set()
    for col in sector_rotation.columns:
        all_sectors.update(sector_rotation[col].dropna().unique())
    
    # Assign a unique color to each sector
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_sectors)))
    sector_colors = {sector: colors[i] for i, sector in enumerate(sorted(all_sectors))}
    
    # Plot each rank as a separate line
    for col in sector_rotation.columns:
        # Create a numerical representation of sectors for plotting
        sector_values = pd.Series(index=sector_rotation.index, dtype=float)
        
        for date, sector in sector_rotation[col].items():
            if pd.notna(sector):
                # Use the position in all_sectors as the y-value
                sector_values[date] = float(sorted(all_sectors).index(sector))
        
        # Plot with scattered points colored by sector
        plt.plot(sector_values.index, sector_values, 'k-', alpha=0.3)
        
        for sector in all_sectors:
            # Find dates where this sector is in this rank
            mask = sector_rotation[col] == sector
            dates = sector_rotation.index[mask]
            
            if len(dates) > 0:
                values = [float(sorted(all_sectors).index(sector))] * len(dates)
                plt.scatter(dates, values, color=sector_colors[sector], label=sector, s=50)
    
    # Remove duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Customize y-axis tick labels
    plt.yticks(range(len(all_sectors)), sorted(all_sectors))
    
    plt.title(title, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sector', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    pass 