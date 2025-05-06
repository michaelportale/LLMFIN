"""
Performance attribution analysis for investment portfolios.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import os


def calculate_returns(prices: pd.DataFrame, method: str = 'arithmetic') -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Args:
        prices: DataFrame with price data
        method: Return calculation method ('arithmetic' or 'logarithmic')
        
    Returns:
        DataFrame with calculated returns
    """
    if method == 'arithmetic':
        returns = prices.pct_change()
    elif method == 'logarithmic':
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'arithmetic' or 'logarithmic'")
    
    return returns


def calculate_active_return(portfolio_return: pd.Series, 
                          benchmark_return: pd.Series) -> pd.Series:
    """
    Calculate active (excess) return of a portfolio relative to a benchmark.
    
    Args:
        portfolio_return: Series with portfolio returns
        benchmark_return: Series with benchmark returns
        
    Returns:
        Series with active returns
    """
    return portfolio_return - benchmark_return


def calculate_attribution_brinson(portfolio_weights: pd.DataFrame, 
                                benchmark_weights: pd.DataFrame,
                                asset_returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Calculate performance attribution using the Brinson model.
    
    Args:
        portfolio_weights: DataFrame with portfolio weights over time
        benchmark_weights: DataFrame with benchmark weights over time
        asset_returns: DataFrame with asset returns over time
        
    Returns:
        Dictionary with attribution components
    """
    # Ensure all DataFrames have the same assets (columns)
    common_assets = list(set(portfolio_weights.columns) & 
                       set(benchmark_weights.columns) & 
                       set(asset_returns.columns))
    
    if not common_assets:
        raise ValueError("No common assets found in portfolio, benchmark, and returns data")
    
    # Filter data to include only common assets
    portfolio_weights = portfolio_weights[common_assets]
    benchmark_weights = benchmark_weights[common_assets]
    asset_returns = asset_returns[common_assets]
    
    # Initialize attribution components
    allocation_effect = pd.DataFrame(0, index=portfolio_weights.index, columns=common_assets)
    selection_effect = pd.DataFrame(0, index=portfolio_weights.index, columns=common_assets)
    interaction_effect = pd.DataFrame(0, index=portfolio_weights.index, columns=common_assets)
    
    # Calculate attribution effects for each time period
    for date in portfolio_weights.index:
        if date in benchmark_weights.index and date in asset_returns.index:
            p_weights = portfolio_weights.loc[date]
            b_weights = benchmark_weights.loc[date]
            returns = asset_returns.loc[date]
            
            # Calculate allocation effect
            allocation_effect.loc[date] = (p_weights - b_weights) * \
                                        benchmark_weights.loc[date].sum() * \
                                        asset_returns.loc[date].mean()
            
            # Calculate selection effect
            selection_effect.loc[date] = b_weights * (returns - returns.mean())
            
            # Calculate interaction effect
            interaction_effect.loc[date] = (p_weights - b_weights) * (returns - returns.mean())
    
    # Calculate total effects
    total_allocation = allocation_effect.sum(axis=1)
    total_selection = selection_effect.sum(axis=1)
    total_interaction = interaction_effect.sum(axis=1)
    total_active_return = total_allocation + total_selection + total_interaction
    
    # Calculate cumulative effects
    cumulative_allocation = (1 + total_allocation).cumprod() - 1
    cumulative_selection = (1 + total_selection).cumprod() - 1
    cumulative_interaction = (1 + total_interaction).cumprod() - 1
    cumulative_active_return = (1 + total_active_return).cumprod() - 1
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'Allocation_Effect': total_allocation,
        'Selection_Effect': total_selection,
        'Interaction_Effect': total_interaction,
        'Total_Active_Return': total_active_return,
        'Cumulative_Allocation': cumulative_allocation,
        'Cumulative_Selection': cumulative_selection,
        'Cumulative_Interaction': cumulative_interaction,
        'Cumulative_Active_Return': cumulative_active_return
    })
    
    return {
        'allocation_effect': allocation_effect,
        'selection_effect': selection_effect,
        'interaction_effect': interaction_effect,
        'summary': summary
    }


def plot_attribution(attribution_data: Dict[str, pd.DataFrame],
                   title: str = "Performance Attribution Analysis",
                   figsize: Tuple[int, int] = (12, 8),
                   output_file: Optional[str] = None) -> None:
    """
    Plot performance attribution results.
    
    Args:
        attribution_data: Dictionary with attribution data
        title: Plot title
        figsize: Figure size
        output_file: Path to save the output file (if None, display the plot)
    """
    if 'summary' not in attribution_data:
        print("No summary data found in attribution data")
        return
    
    summary = attribution_data['summary']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 2]})
    
    # Plot 1: Stacked bar chart of attribution components over time
    components = ['Allocation_Effect', 'Selection_Effect', 'Interaction_Effect']
    summary[components].plot(kind='bar', stacked=True, ax=ax1)
    ax1.set_title('Attribution Components', fontsize=12)
    ax1.set_xlabel('')
    ax1.set_ylabel('Return Contribution', fontsize=10)
    ax1.legend(loc='best', fontsize=8)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Calculate totals for text labels
    total_allocation = summary['Allocation_Effect'].sum()
    total_selection = summary['Selection_Effect'].sum()
    total_interaction = summary['Interaction_Effect'].sum()
    total_active = summary['Total_Active_Return'].sum()
    
    # Create table of totals
    table_text = f"""
    Total Attribution:
    Allocation: {total_allocation:.4f}
    Selection: {total_selection:.4f}
    Interaction: {total_interaction:.4f}
    Active Return: {total_active:.4f}
    """
    
    ax1.annotate(
        table_text, 
        xy=(0.02, 0.02), 
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        verticalalignment='bottom',
        fontsize=8
    )
    
    # Plot 2: Cumulative attribution effects
    cumulative_components = ['Cumulative_Allocation', 'Cumulative_Selection', 
                           'Cumulative_Interaction', 'Cumulative_Active_Return']
    
    summary[cumulative_components].plot(ax=ax2)
    ax2.set_title('Cumulative Attribution Effects', fontsize=12)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Cumulative Return', fontsize=10)
    ax2.legend(loc='best', fontsize=8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def calculate_factor_attribution(portfolio_returns: pd.Series,
                               factor_returns: pd.DataFrame,
                               risk_free_rate: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Calculate performance attribution using factor analysis.
    
    Args:
        portfolio_returns: Series with portfolio returns
        factor_returns: DataFrame with factor returns
        risk_free_rate: Series with risk-free rate returns
        
    Returns:
        Dictionary with factor attribution results
    """
    # Create copy of the factor returns DataFrame
    factor_data = factor_returns.copy()
    
    # Add a constant factor for alpha calculation
    factor_data['Constant'] = 1.0
    
    # Handle risk-free rate
    if risk_free_rate is not None:
        # Convert portfolio and factor returns to excess returns
        excess_portfolio = portfolio_returns - risk_free_rate
        
        # Adjust factor returns to be excess returns (except for the constant)
        for factor in factor_data.columns:
            if factor != 'Constant':
                factor_data[factor] = factor_data[factor] - risk_free_rate
    else:
        excess_portfolio = portfolio_returns
    
    # Fit the regression model
    from sklearn.linear_model import LinearRegression
    
    X = factor_data.dropna()
    y = excess_portfolio.reindex(X.index).dropna()
    
    # Ensure X and y have the same index after handling NAs
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Get factor exposures (betas)
    factor_exposures = pd.Series(model.coef_, index=X.columns)
    
    # Calculate predicted returns
    predicted_returns = model.predict(X)
    predicted_series = pd.Series(predicted_returns, index=common_idx)
    
    # Calculate residual returns (alpha)
    residual_returns = y - predicted_series
    
    # Calculate factor contributions to return
    factor_contributions = pd.DataFrame(index=common_idx, columns=X.columns)
    
    for factor in X.columns:
        factor_contributions[factor] = X[factor] * factor_exposures[factor]
    
    # Calculate total contributions
    total_factor_contribution = factor_contributions.sum(axis=1)
    
    # Calculate metrics
    r_squared = model.score(X, y)
    tracking_error = np.std(residual_returns) * np.sqrt(252)  # Annualized
    information_ratio = (residual_returns.mean() * 252) / tracking_error  # Annualized
    
    return {
        'factor_exposures': factor_exposures,
        'factor_contributions': factor_contributions,
        'total_factor_contribution': total_factor_contribution,
        'residual_returns': residual_returns,
        'r_squared': r_squared,
        'tracking_error': tracking_error,
        'information_ratio': information_ratio
    }


def plot_factor_attribution(factor_attribution: Dict[str, Any],
                          title: str = "Factor Attribution Analysis",
                          figsize: Tuple[int, int] = (12, 10),
                          output_file: Optional[str] = None) -> None:
    """
    Plot factor attribution results.
    
    Args:
        factor_attribution: Dictionary with factor attribution data
        title: Plot title
        figsize: Figure size
        output_file: Path to save the output file (if None, display the plot)
    """
    if 'factor_exposures' not in factor_attribution:
        print("No factor exposures found in attribution data")
        return
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
    
    # Plot 1: Factor exposures (betas)
    exposures = factor_attribution['factor_exposures']
    exposures = exposures.sort_values(ascending=False)
    
    ax1.barh(exposures.index, exposures.values)
    ax1.set_title('Factor Exposures (Betas)', fontsize=12)
    ax1.set_xlabel('Exposure', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative factor contributions
    contributions = factor_attribution['factor_contributions']
    cumulative_contrib = contributions.cumsum()
    
    cumulative_contrib.plot(ax=ax2)
    ax2.set_title('Cumulative Factor Contributions', fontsize=12)
    ax2.set_xlabel('')
    ax2.set_ylabel('Contribution to Return', fontsize=10)
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Residual returns
    residuals = factor_attribution['residual_returns']
    cumulative_residuals = (1 + residuals).cumprod() - 1
    
    ax3.plot(residuals.index, residuals, 'o', markersize=2, label='Daily Residual')
    ax3.plot(residuals.index, residuals.rolling(window=20).mean(), 'r-', label='20-Day Moving Average')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('Residual Returns (Alpha)', fontsize=12)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('Residual Return', fontsize=10)
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Add text with metrics
    metrics_text = f"""
    R-squared: {factor_attribution['r_squared']:.4f}
    Tracking Error: {factor_attribution['tracking_error']:.4f}
    Information Ratio: {factor_attribution['information_ratio']:.4f}
    """
    
    ax3.annotate(
        metrics_text, 
        xy=(0.02, 0.02), 
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        verticalalignment='bottom',
        fontsize=8
    )
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def attribution_summary_report(attribution_data: Dict[str, Any],
                             factor_attribution: Optional[Dict[str, Any]] = None,
                             output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Generate a comprehensive performance attribution summary report.
    
    Args:
        attribution_data: Dictionary with Brinson attribution data
        factor_attribution: Dictionary with factor attribution data
        output_file: Path to save the report CSV file
        
    Returns:
        DataFrame with attribution summary
    """
    if 'summary' not in attribution_data:
        print("No summary data found in attribution data")
        return pd.DataFrame()
    
    # Extract Brinson model data
    summary = attribution_data['summary']
    
    # Create period-by-period attribution summary
    attribution_summary = pd.DataFrame({
        'Total_Active_Return': summary['Total_Active_Return'],
        'Allocation_Effect': summary['Allocation_Effect'],
        'Selection_Effect': summary['Selection_Effect'],
        'Interaction_Effect': summary['Interaction_Effect'],
        'Allocation_Pct': summary['Allocation_Effect'] / summary['Total_Active_Return'].where(summary['Total_Active_Return'] != 0, np.nan),
        'Selection_Pct': summary['Selection_Effect'] / summary['Total_Active_Return'].where(summary['Total_Active_Return'] != 0, np.nan),
        'Interaction_Pct': summary['Interaction_Effect'] / summary['Total_Active_Return'].where(summary['Total_Active_Return'] != 0, np.nan)
    })
    
    # Add factor attribution data if available
    if factor_attribution is not None and 'factor_contributions' in factor_attribution:
        factor_contrib = factor_attribution['factor_contributions']
        residuals = factor_attribution['residual_returns']
        
        # Add factor contributions
        for factor in factor_contrib.columns:
            attribution_summary[f'Factor_{factor}'] = factor_contrib[factor]
        
        # Add residual returns
        attribution_summary['Residual_Return'] = residuals
        
        # Add factor model metrics
        attribution_summary['R_Squared'] = factor_attribution['r_squared']
        attribution_summary['Tracking_Error'] = factor_attribution['tracking_error']
        attribution_summary['Information_Ratio'] = factor_attribution['information_ratio']
    
    # Add cumulative metrics
    attribution_summary['Cumulative_Active_Return'] = summary['Cumulative_Active_Return']
    
    if output_file:
        attribution_summary.to_csv(output_file)
        print(f"Attribution summary saved to {output_file}")
    
    return attribution_summary


def calculate_rolling_attribution(portfolio_weights: pd.DataFrame, 
                                benchmark_weights: pd.DataFrame,
                                asset_returns: pd.DataFrame,
                                window: int = 252) -> Dict[str, pd.DataFrame]:
    """
    Calculate rolling performance attribution over time.
    
    Args:
        portfolio_weights: DataFrame with portfolio weights over time
        benchmark_weights: DataFrame with benchmark weights over time
        asset_returns: DataFrame with asset returns over time
        window: Rolling window size
        
    Returns:
        Dictionary with rolling attribution components
    """
    # Calculate day-by-day attribution
    daily_attribution = calculate_attribution_brinson(
        portfolio_weights, benchmark_weights, asset_returns
    )
    
    # Get the summary data
    summary = daily_attribution['summary']
    
    # Calculate rolling components
    rolling_data = {}
    
    for component in ['Allocation_Effect', 'Selection_Effect', 'Interaction_Effect', 'Total_Active_Return']:
        rolling_data[f'Rolling_{component}'] = summary[component].rolling(window=window).sum()
    
    # Create summary DataFrame
    rolling_summary = pd.DataFrame(rolling_data)
    
    # Add contribution percentages
    for component in ['Allocation_Effect', 'Selection_Effect', 'Interaction_Effect']:
        rolling_component = f'Rolling_{component}'
        rolling_total = 'Rolling_Total_Active_Return'
        
        rolling_summary[f'{rolling_component}_Pct'] = \
            rolling_summary[rolling_component] / rolling_summary[rolling_total].where(rolling_summary[rolling_total] != 0, np.nan)
    
    return {
        'daily_attribution': daily_attribution,
        'rolling_summary': rolling_summary
    }


def plot_rolling_attribution(rolling_attribution: Dict[str, pd.DataFrame],
                           title: str = "Rolling Performance Attribution",
                           window: int = 252,
                           figsize: Tuple[int, int] = (12, 10),
                           output_file: Optional[str] = None) -> None:
    """
    Plot rolling performance attribution results.
    
    Args:
        rolling_attribution: Dictionary with rolling attribution data
        title: Plot title
        window: Rolling window size (for title display)
        figsize: Figure size
        output_file: Path to save the output file (if None, display the plot)
    """
    if 'rolling_summary' not in rolling_attribution:
        print("No rolling summary found in attribution data")
        return
    
    rolling_summary = rolling_attribution['rolling_summary']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot 1: Rolling attribution components
    components = ['Rolling_Allocation_Effect', 'Rolling_Selection_Effect', 'Rolling_Interaction_Effect']
    rolling_summary[components].plot(ax=ax1)
    ax1.set_title(f'Rolling {window}-Day Attribution Components', fontsize=12)
    ax1.set_xlabel('')
    ax1.set_ylabel('Return Contribution', fontsize=10)
    ax1.legend(loc='best', fontsize=8)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Contribution percentages
    pct_components = [c + '_Pct' for c in components]
    
    # Handle extreme values by clipping
    plotting_data = rolling_summary[pct_components].copy()
    plotting_data = plotting_data.clip(lower=-2, upper=2)  # Clip to reasonable range
    
    # Convert to stacked area chart
    plotting_data.plot(kind='area', stacked=True, ax=ax2)
    ax2.set_title('Attribution Component Percentages', fontsize=12)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Percentage of Active Return', fontsize=10)
    ax2.legend(loc='best', fontsize=8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Set y-axis limits for percentage plot
    ax2.set_ylim(-1.5, 1.5)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    pass 