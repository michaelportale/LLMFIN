"""
Feature visualization tools for financial data analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_time_series(df: pd.DataFrame, columns: List[str], 
                    title: str = "Time Series Plot",
                    figsize: Tuple[int, int] = (12, 6),
                    output_file: Optional[str] = None) -> None:
    """
    Plot time series data.
    
    Args:
        df: DataFrame with time series data
        columns: List of columns to plot
        title: Plot title
        figsize: Figure size
        output_file: Path to save the plot, or None to display
    """
    plt.figure(figsize=figsize)
    
    for column in columns:
        if column in df.columns:
            plt.plot(df.index, df[column], label=column)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_technical_indicators(df: pd.DataFrame, 
                            price_column: str = 'Close',
                            volume_column: str = 'Volume',
                            figsize: Tuple[int, int] = (12, 10),
                            output_file: Optional[str] = None) -> None:
    """
    Plot stock price with technical indicators.
    
    Args:
        df: DataFrame with price and technical indicator data
        price_column: Column name for price data
        volume_column: Column name for volume data
        figsize: Figure size
        output_file: Path to save the plot, or None to display
    """
    # Create figure and axis
    fig, axes = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot price
    ax1 = axes[0]
    ax1.plot(df.index, df[price_column], label=price_column)
    
    # Add moving averages if available
    for column in df.columns:
        if 'SMA_' in column or 'EMA_' in column:
            ax1.plot(df.index, df[column], label=column)
    
    # Add Bollinger Bands if available
    if all(col in df.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
        ax1.plot(df.index, df['BB_Upper'], 'r--', label='BB Upper')
        ax1.plot(df.index, df['BB_Middle'], 'g--', label='BB Middle')
        ax1.plot(df.index, df['BB_Lower'], 'b--', label='BB Lower')
    
    ax1.set_title('Price and Technical Indicators', fontsize=14)
    ax1.set_ylabel(price_column, fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Plot volume
    if volume_column in df.columns:
        ax2 = axes[1]
        ax2.bar(df.index, df[volume_column], width=1, color='blue', alpha=0.5)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)
    
    # Plot oscillators (if available)
    ax3 = axes[2]
    
    if 'RSI' in df.columns:
        ax3.plot(df.index, df['RSI'], label='RSI')
        # Add RSI levels
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        ax3.plot(df.index, df['MACD'], label='MACD')
        ax3.plot(df.index, df['MACD_Signal'], label='Signal')
        
        # Plot MACD histogram if available
        if 'MACD_Hist' in df.columns:
            ax3.bar(df.index, df['MACD_Hist'], width=1, color='green', 
                  alpha=0.5, label='MACD Hist')
    
    if 'Stoch_%K' in df.columns and 'Stoch_%D' in df.columns:
        ax3.plot(df.index, df['Stoch_%K'], label='%K')
        ax3.plot(df.index, df['Stoch_%D'], label='%D')
        
        # Add Stochastic levels
        ax3.axhline(y=80, color='r', linestyle='--', alpha=0.5)
        ax3.axhline(y=20, color='g', linestyle='--', alpha=0.5)
    
    ax3.set_ylabel('Oscillators', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_feature_distributions(df: pd.DataFrame, 
                             features: List[str],
                             n_cols: int = 3,
                             figsize: Tuple[int, int] = (15, 10),
                             output_file: Optional[str] = None) -> None:
    """
    Plot distributions of features.
    
    Args:
        df: DataFrame with feature data
        features: List of features to plot
        n_cols: Number of columns in the subplot grid
        figsize: Figure size
        output_file: Path to save the plot, or None to display
    """
    # Calculate number of rows needed
    n_rows = (len(features) + n_cols - 1) // n_cols
    
    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes for easy iteration
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Plot distributions
    for i, feature in enumerate(features):
        if i < len(axes) and feature in df.columns:
            # Create histogram and KDE
            sns.histplot(
                df[feature].dropna(),
                kde=True,
                ax=axes[i]
            )
            
            # Add mean and median lines
            mean_val = df[feature].mean()
            median_val = df[feature].median()
            
            axes[i].axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(median_val, color='g', linestyle='--', label=f'Median: {median_val:.2f}')
            
            axes[i].set_title(feature, fontsize=12)
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_feature_correlations(df: pd.DataFrame, 
                            target_column: Optional[str] = None,
                            top_n: int = 20,
                            figsize: Tuple[int, int] = (12, 10),
                            output_file: Optional[str] = None) -> None:
    """
    Plot feature correlations.
    
    Args:
        df: DataFrame with feature data
        target_column: Target column to highlight correlations with, or None
        top_n: Number of top correlated features to include
        figsize: Figure size
        output_file: Path to save the plot, or None to display
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Calculate correlation matrix
    corr = numeric_df.corr()
    
    # If target column is specified, sort features by correlation with target
    if target_column and target_column in corr.columns:
        target_corr = corr[target_column].abs().sort_values(ascending=False)
        
        # Select top N features (plus target)
        top_features = target_corr.head(top_n + 1).index.tolist()
        
        # Ensure target column is included
        if target_column not in top_features:
            top_features.append(target_column)
        
        # Filter correlation matrix
        corr = corr.loc[top_features, top_features]
    
    # Create heatmap
    plt.figure(figsize=figsize)
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr, 
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
    
    title = 'Feature Correlation Heatmap'
    if target_column:
        title += f' (Top {top_n} features correlated with {target_column})'
    
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_feature_importance(importance_df: pd.DataFrame,
                          feature_column: str = 'feature',
                          score_column: str = 'importance',
                          title: str = 'Feature Importance',
                          top_n: int = 20,
                          figsize: Tuple[int, int] = (10, 8),
                          output_file: Optional[str] = None) -> None:
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with feature importance scores
        feature_column: Column name for feature names
        score_column: Column name for importance scores
        title: Plot title
        top_n: Number of top features to include
        figsize: Figure size
        output_file: Path to save the plot, or None to display
    """
    # Check required columns
    if feature_column not in importance_df.columns or score_column not in importance_df.columns:
        raise ValueError(f"DataFrame must contain '{feature_column}' and '{score_column}' columns")
    
    # Sort by importance score and get top N
    sorted_df = importance_df.sort_values(score_column, ascending=False).head(top_n)
    
    # Create horizontal bar chart
    plt.figure(figsize=figsize)
    
    # Plot importance scores
    ax = plt.barh(sorted_df[feature_column], sorted_df[score_column])
    
    # Add values as text
    for i, (feature, score) in enumerate(zip(sorted_df[feature_column], sorted_df[score_column])):
        plt.text(score + 0.01, i, f'{score:.4f}', va='center')
    
    plt.title(title, fontsize=14)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_pairplot(df: pd.DataFrame, 
                 features: List[str], 
                 hue: Optional[str] = None,
                 figsize: Tuple[int, int] = (12, 12),
                 output_file: Optional[str] = None) -> None:
    """
    Create a pairplot of features.
    
    Args:
        df: DataFrame with feature data
        features: List of features to include
        hue: Column to use for coloring points
        figsize: Figure size
        output_file: Path to save the plot, or None to display
    """
    # Check that features exist in DataFrame
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Features not found in DataFrame: {missing_features}")
        features = [f for f in features if f in df.columns]
    
    if not features:
        print("No valid features to plot")
        return
    
    # Create pairplot
    plt.figure(figsize=figsize)
    
    g = sns.pairplot(
        df[features + ([hue] if hue else [])],
        hue=hue,
        diag_kind='kde',
        plot_kws={'alpha': 0.6},
        height=2.5
    )
    
    g.fig.suptitle('Feature Pairplot', y=1.02, fontsize=16)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def visualize_features(df: pd.DataFrame, 
                      target_column: Optional[str] = None,
                      top_n_features: int = 10,
                      output_dir: str = 'data/visualizations',
                      prefix: str = '') -> None:
    """
    Create a comprehensive visualization of features.
    
    Args:
        df: DataFrame with feature data
        target_column: Target column for supervised learning, or None
        top_n_features: Number of top features to include in some plots
        output_dir: Directory to save output plots
        prefix: Prefix for output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=['number'])
    
    # Get top correlated features if target is specified
    if target_column and target_column in numeric_df.columns:
        corr = numeric_df.corr()[target_column].abs().sort_values(ascending=False)
        top_features = corr.head(top_n_features + 1).index.tolist()
        
        # Ensure target is not in the list of features
        if target_column in top_features:
            top_features.remove(target_column)
    else:
        # Just use first N columns
        top_features = numeric_df.columns[:top_n_features].tolist()
    
    # Plot time series if index is datetime
    if isinstance(df.index, pd.DatetimeIndex):
        print("Plotting time series...")
        
        # Price and volume if available
        if all(col in df.columns for col in ['Close', 'Volume']):
            plot_technical_indicators(
                df,
                output_file=os.path.join(output_dir, f'{prefix}technical_indicators.png')
            )
        
        # Top features time series
        if target_column:
            plot_time_series(
                df,
                columns=[target_column] + top_features[:5],
                title=f'Top Features with {target_column}',
                output_file=os.path.join(output_dir, f'{prefix}top_features_time_series.png')
            )
    
    # Plot feature distributions
    print("Plotting feature distributions...")
    plot_feature_distributions(
        numeric_df,
        features=top_features,
        output_file=os.path.join(output_dir, f'{prefix}feature_distributions.png')
    )
    
    # Plot feature correlations
    print("Plotting feature correlations...")
    plot_feature_correlations(
        numeric_df,
        target_column=target_column,
        top_n=top_n_features,
        output_file=os.path.join(output_dir, f'{prefix}feature_correlations.png')
    )
    
    # Plot pairplot for top features
    print("Plotting feature pairplot...")
    if len(top_features) > 5:
        plot_features = top_features[:5]  # Limit to 5 for readability
    else:
        plot_features = top_features
    
    if target_column:
        plot_features = plot_features + [target_column]
    
    plot_pairplot(
        numeric_df,
        features=plot_features,
        hue=target_column if target_column and target_column in numeric_df.columns else None,
        output_file=os.path.join(output_dir, f'{prefix}feature_pairplot.png')
    )
    
    print(f"All visualizations saved to {output_dir}/")


def plot_seasonal_decomposition(df: pd.DataFrame, 
                              column: str = 'Close', 
                              period: int = 252,
                              figsize: Tuple[int, int] = (12, 10),
                              output_file: Optional[str] = None) -> None:
    """
    Plot seasonal decomposition of time series.
    
    Args:
        df: DataFrame with time series data
        column: Column to decompose
        period: Period for seasonal decomposition (252 for daily data = 1 trading year)
        figsize: Figure size
        output_file: Path to save the plot, or None to display
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Check if index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    # Perform seasonal decomposition
    result = seasonal_decompose(df[column].dropna(), model='multiplicative', period=period)
    
    # Create figure and axes
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    # Plot observed, trend, seasonal, and residual components
    result.observed.plot(ax=axes[0], title='Observed')
    result.trend.plot(ax=axes[1], title='Trend')
    result.seasonal.plot(ax=axes[2], title='Seasonal')
    result.resid.plot(ax=axes[3], title='Residual')
    
    # Add grid to each subplot
    for ax in axes:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def create_feature_dashboard(df: pd.DataFrame, 
                           target_column: Optional[str] = None,
                           output_file: str = 'data/visualizations/dashboard.html') -> None:
    """
    Create an interactive feature dashboard.
    
    Args:
        df: DataFrame with feature data
        target_column: Target column for supervised learning, or None
        output_file: Path to save the dashboard
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        print("Plotly is required for interactive dashboards. Install with: pip install plotly")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create dashboard with multiple subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Price Chart', 'Feature Correlation', 'Feature Distribution', 'Time Series'),
        specs=[
            [{'type': 'xy'}, {'type': 'xy'}],
            [{'type': 'xy'}, {'type': 'xy'}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # 1. Price Chart (if available)
    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add volume as bar chart
        if 'Volume' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color='rgba(0, 0, 255, 0.3)'
                ),
                row=1, col=1
            )
            
            # Create secondary y-axis for volume
            fig.update_layout(
                yaxis=dict(
                    title='Price',
                    side='left'
                ),
                yaxis2=dict(
                    title='Volume',
                    side='right',
                    overlaying='y'
                )
            )
    
    # 2. Feature Correlation
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    
    # If target column exists, sort by correlation with target
    if target_column and target_column in corr.columns:
        corr_with_target = corr[target_column].abs().sort_values(ascending=False).head(10)
        features = corr_with_target.index.tolist()
        
        # Ensure target column is not included
        if target_column in features:
            features.remove(target_column)
        
        # Create bar chart of correlations
        fig.add_trace(
            go.Bar(
                x=features,
                y=[corr.loc[feature, target_column] for feature in features],
                name=f'Correlation with {target_column}'
            ),
            row=1, col=2
        )
        
        # Update layout for correlation subplot
        fig.update_yaxes(title_text='Correlation', row=1, col=2)
        fig.update_xaxes(tickangle=45, row=1, col=2)
    
    # 3. Feature Distribution
    if target_column and target_column in numeric_df.columns:
        feature_to_plot = numeric_df.columns[0] if numeric_df.columns[0] != target_column else numeric_df.columns[1]
        
        if len(numeric_df.columns) > 1:
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=numeric_df[feature_to_plot],
                    name=feature_to_plot,
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # Update layout for distribution subplot
            fig.update_yaxes(title_text='Frequency', row=2, col=1)
            fig.update_xaxes(title_text=feature_to_plot, row=2, col=1)
    
    # 4. Time Series of key features
    if isinstance(df.index, pd.DatetimeIndex):
        # If target exists, plot it
        if target_column and target_column in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[target_column],
                    name=target_column
                ),
                row=2, col=2
            )
        
        # Add a technical indicator if available
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    name='RSI',
                    line=dict(color='orange')
                ),
                row=2, col=2
            )
            
            # Add RSI levels
            fig.add_shape(
                type='line',
                x0=df.index[0],
                x1=df.index[-1],
                y0=70,
                y1=70,
                line=dict(color='red', width=1, dash='dash'),
                row=2, col=2
            )
            
            fig.add_shape(
                type='line',
                x0=df.index[0],
                x1=df.index[-1],
                y0=30,
                y1=30,
                line=dict(color='green', width=1, dash='dash'),
                row=2, col=2
            )
        
        # Update layout for time series subplot
        fig.update_yaxes(title_text='Value', row=2, col=2)
        fig.update_xaxes(title_text='Date', row=2, col=2)
    
    # Update layout for entire figure
    fig.update_layout(
        title='Financial Feature Dashboard',
        height=800,
        width=1200,
        showlegend=True
    )
    
    # Save to HTML file
    pio.write_html(fig, file=output_file, auto_open=False)
    print(f"Interactive dashboard saved to {output_file}") 