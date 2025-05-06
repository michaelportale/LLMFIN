"""
Portfolio optimization algorithms using Modern Portfolio Theory.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from scipy.optimize import minimize
import scipy.stats as stats
import os


def calculate_portfolio_performance(weights: np.ndarray, 
                                   returns: np.ndarray, 
                                   cov_matrix: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate portfolio expected return, volatility and Sharpe ratio.
    
    Args:
        weights: Array of asset weights
        returns: Array of asset expected returns
        cov_matrix: Covariance matrix of asset returns
        
    Returns:
        Tuple of (expected return, volatility, Sharpe ratio)
    """
    portfolio_return = np.sum(returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Assuming risk-free rate of 0 for simplicity
    sharpe_ratio = portfolio_return / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio


def negative_sharpe_ratio(weights: np.ndarray, 
                         returns: np.ndarray, 
                         cov_matrix: np.ndarray) -> float:
    """
    Calculate negative Sharpe ratio (for minimization).
    
    Args:
        weights: Array of asset weights
        returns: Array of asset expected returns
        cov_matrix: Covariance matrix of asset returns
        
    Returns:
        Negative Sharpe ratio
    """
    portfolio_return, portfolio_volatility, sharpe_ratio = calculate_portfolio_performance(
        weights, returns, cov_matrix
    )
    return -sharpe_ratio


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Calculate portfolio volatility.
    
    Args:
        weights: Array of asset weights
        cov_matrix: Covariance matrix of asset returns
        
    Returns:
        Portfolio volatility
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def optimize_portfolio(returns: pd.DataFrame, 
                      optimization_goal: str = 'sharpe',
                      risk_free_rate: float = 0.0,
                      target_return: Optional[float] = None,
                      target_risk: Optional[float] = None,
                      constraints: Optional[List] = None) -> Dict[str, Any]:
    """
    Optimize portfolio weights according to Modern Portfolio Theory.
    
    Args:
        returns: DataFrame with asset returns
        optimization_goal: Optimization objective ('sharpe', 'min_risk', 'max_return', 'target_return', 'target_risk')
        risk_free_rate: Risk-free rate
        target_return: Target portfolio return (used if optimization_goal is 'target_return')
        target_risk: Target portfolio risk (used if optimization_goal is 'target_risk')
        constraints: Additional constraints for the optimization
        
    Returns:
        Dictionary with optimization results
    """
    # Calculate expected returns and covariance matrix
    expected_returns = returns.mean()
    cov_matrix = returns.cov()
    
    num_assets = len(expected_returns)
    args = (expected_returns.values, cov_matrix.values)
    
    # Set constraints
    if constraints is None:
        constraints = []
    
    # Initial weights: equal allocation
    initial_weights = np.array([1.0 / num_assets] * num_assets)
    
    # Weight bounds: each asset weight between 0% and 100%
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Weight sum to 1 constraint
    weight_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    constraints.append(weight_constraint)
    
    # Optimize based on the specified goal
    if optimization_goal == 'sharpe':
        # Maximize Sharpe ratio
        result = minimize(
            negative_sharpe_ratio, 
            initial_weights, 
            args=args, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        optimal_weights = result['x']
    
    elif optimization_goal == 'min_risk':
        # Minimize portfolio volatility
        result = minimize(
            portfolio_volatility, 
            initial_weights, 
            args=(cov_matrix.values,), 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        optimal_weights = result['x']
    
    elif optimization_goal == 'max_return':
        # Maximize expected return (will typically allocate 100% to the highest return asset)
        def negative_return(weights, returns, _):
            return -np.sum(weights * returns)
        
        result = minimize(
            negative_return, 
            initial_weights, 
            args=args, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        optimal_weights = result['x']
    
    elif optimization_goal == 'target_return':
        if target_return is None:
            raise ValueError("target_return must be specified when optimization_goal is 'target_return'")
        
        # Minimize risk while achieving target return
        return_constraint = {
            'type': 'eq', 
            'fun': lambda x: np.sum(x * expected_returns.values) - target_return
        }
        constraints.append(return_constraint)
        
        result = minimize(
            portfolio_volatility, 
            initial_weights, 
            args=(cov_matrix.values,), 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        optimal_weights = result['x']
    
    elif optimization_goal == 'target_risk':
        if target_risk is None:
            raise ValueError("target_risk must be specified when optimization_goal is 'target_risk'")
        
        # Maximize return while not exceeding target risk
        risk_constraint = {
            'type': 'eq', 
            'fun': lambda x: portfolio_volatility(x, cov_matrix.values) - target_risk
        }
        constraints.append(risk_constraint)
        
        def negative_return(weights, returns, _):
            return -np.sum(weights * returns)
        
        result = minimize(
            negative_return, 
            initial_weights, 
            args=args, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        optimal_weights = result['x']
    
    else:
        raise ValueError(f"Unknown optimization_goal: {optimization_goal}")
    
    # Calculate portfolio performance with optimized weights
    portfolio_return, portfolio_volatility_val, sharpe_ratio = calculate_portfolio_performance(
        optimal_weights, expected_returns.values, cov_matrix.values
    )
    
    # Create dictionary of asset weights
    asset_weights = {asset: weight for asset, weight in zip(expected_returns.index, optimal_weights)}
    
    return {
        'weights': asset_weights,
        'expected_return': portfolio_return,
        'volatility': portfolio_volatility_val,
        'sharpe_ratio': sharpe_ratio,
        'optimization_goal': optimization_goal,
        'success': result['success']
    }


def generate_efficient_frontier(returns: pd.DataFrame, 
                              n_points: int = 100,
                              risk_free_rate: float = 0.0,
                              include_assets: bool = True,
                              include_sharpe: bool = True,
                              include_monte_carlo: bool = True,
                              monte_carlo_n: int = 1000) -> Dict[str, Any]:
    """
    Generate the efficient frontier and related portfolio data.
    
    Args:
        returns: DataFrame with asset returns
        n_points: Number of points in the efficient frontier
        risk_free_rate: Risk-free rate
        include_assets: Whether to include individual assets in the plot
        include_sharpe: Whether to include the tangency portfolio with max Sharpe ratio
        include_monte_carlo: Whether to include random portfolios for comparison
        monte_carlo_n: Number of random portfolios to generate
        
    Returns:
        Dictionary with efficient frontier data
    """
    # Calculate expected returns and covariance matrix
    expected_returns = returns.mean()
    cov_matrix = returns.cov()
    
    num_assets = len(expected_returns)
    
    # Generate the efficient frontier
    target_returns = np.linspace(expected_returns.min(), expected_returns.max(), n_points)
    efficient_volatilities = []
    efficient_weights = []
    
    for target_return in target_returns:
        result = optimize_portfolio(
            returns, 
            optimization_goal='target_return',
            target_return=target_return
        )
        efficient_volatilities.append(result['volatility'])
        efficient_weights.append(result['weights'])
    
    # Find the minimum volatility portfolio
    min_vol_result = optimize_portfolio(returns, optimization_goal='min_risk')
    min_vol_return = min_vol_result['expected_return']
    min_vol_volatility = min_vol_result['volatility']
    
    # Find the maximum Sharpe ratio portfolio
    max_sharpe_result = optimize_portfolio(returns, optimization_goal='sharpe')
    max_sharpe_return = max_sharpe_result['expected_return']
    max_sharpe_volatility = max_sharpe_result['volatility']
    
    # Generate random portfolios for comparison
    monte_carlo_returns = []
    monte_carlo_volatilities = []
    monte_carlo_sharpe_ratios = []
    monte_carlo_weights = []
    
    if include_monte_carlo:
        for _ in range(monte_carlo_n):
            # Generate random weights
            weights = np.random.random(num_assets)
            weights = weights / np.sum(weights)
            
            # Calculate portfolio performance
            portfolio_return, portfolio_volatility_val, sharpe_ratio = calculate_portfolio_performance(
                weights, expected_returns.values, cov_matrix.values
            )
            
            monte_carlo_returns.append(portfolio_return)
            monte_carlo_volatilities.append(portfolio_volatility_val)
            monte_carlo_sharpe_ratios.append(sharpe_ratio)
            monte_carlo_weights.append(
                {asset: weight for asset, weight in zip(expected_returns.index, weights)}
            )
    
    # Extract individual asset data
    asset_returns = expected_returns.values
    asset_volatilities = np.sqrt(np.diag(cov_matrix.values))
    
    return {
        'efficient_returns': target_returns,
        'efficient_volatilities': efficient_volatilities,
        'efficient_weights': efficient_weights,
        'min_vol_return': min_vol_return,
        'min_vol_volatility': min_vol_volatility,
        'min_vol_weights': min_vol_result['weights'],
        'max_sharpe_return': max_sharpe_return,
        'max_sharpe_volatility': max_sharpe_volatility,
        'max_sharpe_weights': max_sharpe_result['weights'],
        'max_sharpe_ratio': max_sharpe_result['sharpe_ratio'],
        'monte_carlo_returns': monte_carlo_returns,
        'monte_carlo_volatilities': monte_carlo_volatilities,
        'monte_carlo_sharpe_ratios': monte_carlo_sharpe_ratios,
        'monte_carlo_weights': monte_carlo_weights,
        'asset_returns': asset_returns,
        'asset_volatilities': asset_volatilities,
        'asset_names': list(expected_returns.index),
        'risk_free_rate': risk_free_rate
    }


def plot_efficient_frontier(efficient_frontier_data: Dict[str, Any],
                          title: str = "Efficient Frontier",
                          figsize: Tuple[int, int] = (12, 8),
                          output_file: Optional[str] = None) -> None:
    """
    Plot the efficient frontier and related portfolio data.
    
    Args:
        efficient_frontier_data: Data from generate_efficient_frontier
        title: Plot title
        figsize: Figure size
        output_file: Path to save the output file (if None, display the plot)
    """
    plt.figure(figsize=figsize)
    
    # Plot the efficient frontier
    plt.plot(
        efficient_frontier_data['efficient_volatilities'],
        efficient_frontier_data['efficient_returns'],
        'b-', linewidth=3, label='Efficient Frontier'
    )
    
    # Plot the minimum volatility portfolio
    plt.plot(
        efficient_frontier_data['min_vol_volatility'],
        efficient_frontier_data['min_vol_return'],
        'g*', markersize=15, label='Minimum Volatility'
    )
    
    # Plot the maximum Sharpe ratio portfolio
    plt.plot(
        efficient_frontier_data['max_sharpe_volatility'],
        efficient_frontier_data['max_sharpe_return'],
        'r*', markersize=15, label='Maximum Sharpe Ratio'
    )
    
    # Plot individual assets
    plt.scatter(
        efficient_frontier_data['asset_volatilities'],
        efficient_frontier_data['asset_returns'],
        marker='o', s=100, color='black', label='Individual Assets'
    )
    
    # Add asset labels
    for i, asset in enumerate(efficient_frontier_data['asset_names']):
        plt.annotate(
            asset, 
            (efficient_frontier_data['asset_volatilities'][i], 
             efficient_frontier_data['asset_returns'][i]),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    # Plot random portfolios
    if 'monte_carlo_volatilities' in efficient_frontier_data and efficient_frontier_data['monte_carlo_volatilities']:
        plt.scatter(
            efficient_frontier_data['monte_carlo_volatilities'],
            efficient_frontier_data['monte_carlo_returns'],
            marker='.', color='lightgray', s=10, alpha=0.3, label='Random Portfolios'
        )
    
    # Plot the capital market line (CML)
    if 'risk_free_rate' in efficient_frontier_data:
        max_x = max(max(efficient_frontier_data['efficient_volatilities']), 
                   max(efficient_frontier_data['asset_volatilities'])) * 1.2
        
        # Capital Market Line: y = rf + slope * x
        slope = (efficient_frontier_data['max_sharpe_return'] - efficient_frontier_data['risk_free_rate']) / \
                efficient_frontier_data['max_sharpe_volatility']
        
        x_cml = np.array([0, max_x])
        y_cml = efficient_frontier_data['risk_free_rate'] + slope * x_cml
        
        plt.plot(x_cml, y_cml, 'y--', label='Capital Market Line')
    
    plt.title(title, fontsize=14)
    plt.xlabel('Volatility (Standard Deviation)', fontsize=12)
    plt.ylabel('Expected Return', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add annotations with key metrics
    annotation_text = f"""
    Min Volatility: Return = {efficient_frontier_data['min_vol_return']:.4f}, Volatility = {efficient_frontier_data['min_vol_volatility']:.4f}
    Max Sharpe: Return = {efficient_frontier_data['max_sharpe_return']:.4f}, Volatility = {efficient_frontier_data['max_sharpe_volatility']:.4f}, Sharpe = {efficient_frontier_data['max_sharpe_ratio']:.4f}
    """
    
    plt.annotate(
        annotation_text, 
        xy=(0.02, 0.02), 
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8)
    )
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_optimal_portfolio_weights(weights: Dict[str, float],
                                 title: str = "Optimal Portfolio Weights",
                                 figsize: Tuple[int, int] = (10, 6),
                                 output_file: Optional[str] = None) -> None:
    """
    Plot the optimal portfolio weights as a pie chart and bar chart.
    
    Args:
        weights: Dictionary of asset weights
        title: Plot title
        figsize: Figure size
        output_file: Path to save the output file (if None, display the plot)
    """
    # Sort weights by value
    sorted_weights = {k: v for k, v in sorted(weights.items(), key=lambda item: item[1], reverse=True)}
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Pie chart
    assets = list(sorted_weights.keys())
    values = list(sorted_weights.values())
    
    # Plot pie chart
    ax1.pie(
        values, 
        labels=assets, 
        autopct='%1.1f%%',
        startangle=90, 
        wedgeprops=dict(width=0.5)
    )
    ax1.set_title('Allocation Pie Chart')
    
    # Plot bar chart
    colors = plt.cm.viridis(np.linspace(0, 1, len(assets)))
    ax2.barh(
        assets,
        values,
        color=colors
    )
    
    # Add percentage labels on the bars
    for i, v in enumerate(values):
        ax2.text(v + 0.01, i, f"{v:.1%}", va='center')
    
    ax2.set_title('Allocation Bar Chart')
    ax2.set_xlim(0, max(values) * 1.2)
    ax2.invert_yaxis()  # Invert y-axis to match pie chart order
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def calculate_portfolio_statistics(weights: Dict[str, float],
                                 returns: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate various portfolio statistics.
    
    Args:
        weights: Dictionary of asset weights
        returns: DataFrame with asset returns
        
    Returns:
        Dictionary with portfolio statistics
    """
    # Convert weights to array
    assets = list(weights.keys())
    weights_array = np.array([weights[asset] for asset in assets])
    
    # Extract returns data for the assets in the portfolio
    portfolio_returns = returns[assets]
    
    # Calculate expected returns and covariance matrix
    expected_returns = portfolio_returns.mean()
    cov_matrix = portfolio_returns.cov()
    
    # Calculate portfolio performance
    weights_array = np.array([weights[asset] for asset in assets])
    portfolio_return, portfolio_volatility, sharpe_ratio = calculate_portfolio_performance(
        weights_array, expected_returns.values, cov_matrix.values
    )
    
    # Calculate historical portfolio returns
    historical_returns = portfolio_returns.dot(weights_array)
    
    # Calculate additional statistics
    negative_returns = historical_returns[historical_returns < 0]
    
    statistics = {
        'expected_annual_return': portfolio_return * 252,  # Annualized
        'annual_volatility': portfolio_volatility * np.sqrt(252),  # Annualized
        'sharpe_ratio': sharpe_ratio * np.sqrt(252),  # Annualized
        'max_drawdown': calculate_max_drawdown(historical_returns),
        'skewness': stats.skew(historical_returns),
        'kurtosis': stats.kurtosis(historical_returns),
        'var_95': -np.percentile(historical_returns, 5),  # 95% VaR
        'cvar_95': -negative_returns.mean() if len(negative_returns) > 0 else 0,  # 95% CVaR
        'positive_days_pct': (historical_returns > 0).mean(),
        'negative_days_pct': (historical_returns < 0).mean(),
        'best_return': historical_returns.max(),
        'worst_return': historical_returns.min(),
        'avg_positive_return': historical_returns[historical_returns > 0].mean() if any(historical_returns > 0) else 0,
        'avg_negative_return': historical_returns[historical_returns < 0].mean() if any(historical_returns < 0) else 0
    }
    
    return statistics


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate the maximum drawdown of a return series.
    
    Args:
        returns: Series of returns
        
    Returns:
        Maximum drawdown as a positive value
    """
    # Calculate wealth index (cumulative returns)
    wealth_index = (1 + returns).cumprod()
    
    # Calculate previous peaks
    previous_peaks = wealth_index.cummax()
    
    # Calculate drawdowns
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    
    # Return the maximum drawdown as a positive value
    return -drawdowns.min() if not drawdowns.empty else 0


def optimal_portfolio_with_constraints(returns: pd.DataFrame,
                                     min_weights: Optional[Dict[str, float]] = None,
                                     max_weights: Optional[Dict[str, float]] = None,
                                     risk_budget: Optional[Dict[str, float]] = None,
                                     target_return: Optional[float] = None,
                                     target_risk: Optional[float] = None,
                                     optimization_goal: str = 'sharpe') -> Dict[str, Any]:
    """
    Optimize portfolio with additional constraints.
    
    Args:
        returns: DataFrame with asset returns
        min_weights: Minimum weights for assets
        max_weights: Maximum weights for assets
        risk_budget: Target risk contribution for each asset
        target_return: Target portfolio return
        target_risk: Target portfolio risk
        optimization_goal: Optimization objective
        
    Returns:
        Dictionary with optimization results
    """
    assets = returns.columns
    num_assets = len(assets)
    
    # Set default min/max constraints if not provided
    if min_weights is None:
        min_weights = {asset: 0.0 for asset in assets}
    
    if max_weights is None:
        max_weights = {asset: 1.0 for asset in assets}
    
    # Create bounds
    bounds = []
    for asset in assets:
        min_w = min_weights.get(asset, 0.0)
        max_w = max_weights.get(asset, 1.0)
        bounds.append((min_w, max_w))
    
    # Create constraints
    constraints = []
    
    # Weights sum to 1
    constraints.append({
        'type': 'eq',
        'fun': lambda x: np.sum(x) - 1
    })
    
    # Risk budgeting constraint (if provided)
    if risk_budget is not None:
        cov_matrix = returns.cov().values
        
        def risk_budget_constraint(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            asset_contribs = []
            
            for i, asset in enumerate(assets):
                # Calculate marginal contribution to risk
                mcr = np.dot(cov_matrix[i], weights) / portfolio_vol
                
                # Risk contribution of asset i
                rc = weights[i] * mcr
                
                # Target risk contribution
                trc = risk_budget.get(asset, 1.0 / num_assets) * portfolio_vol
                
                asset_contribs.append((rc - trc) ** 2)
            
            return np.sum(asset_contribs)
        
        # Use different optimization approach for risk parity
        if optimization_goal == 'risk_budget':
            initial_weights = np.array([1.0 / num_assets] * num_assets)
            
            result = minimize(
                risk_budget_constraint, 
                initial_weights, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}],
                options={'ftol': 1e-9, 'disp': False}
            )
            
            optimal_weights = result['x']
            weights_dict = {asset: weight for asset, weight in zip(assets, optimal_weights)}
            
            # Calculate portfolio performance
            expected_returns_arr = returns.mean().values
            cov_matrix = returns.cov().values
            
            portfolio_return, portfolio_volatility_val, sharpe_ratio = calculate_portfolio_performance(
                optimal_weights, expected_returns_arr, cov_matrix
            )
            
            return {
                'weights': weights_dict,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility_val,
                'sharpe_ratio': sharpe_ratio,
                'optimization_goal': 'risk_budget',
                'success': result['success']
            }
    
    # Run standard optimization with constraints
    return optimize_portfolio(
        returns=returns,
        optimization_goal=optimization_goal,
        target_return=target_return,
        target_risk=target_risk,
        constraints=constraints
    )


if __name__ == "__main__":
    # Example usage
    pass 