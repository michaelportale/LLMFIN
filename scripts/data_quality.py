"""
Data quality verification utilities.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


class DataQualityChecker:
    """
    Class for checking and verifying financial data quality.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the data quality checker.
        
        Args:
            verbose: Whether to print detailed information
        """
        self.verbose = verbose
    
    def check_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data completeness.
        
        Args:
            df: DataFrame with financial data
            
        Returns:
            Dictionary with completeness metrics
        """
        # Check for missing values
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        
        # Check if all required columns are present
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        # Check for zero volume days (might be holidays or missing data)
        if 'Volume' in df.columns:
            zero_volume_days = (df['Volume'] == 0).sum()
        else:
            zero_volume_days = None
        
        result = {
            "row_count": len(df),
            "missing_values": missing_values.to_dict(),
            "missing_percentage": missing_percentage.to_dict(),
            "missing_columns": missing_columns,
            "zero_volume_days": zero_volume_days
        }
        
        if self.verbose:
            print("Data Completeness Check:")
            print(f"  Total rows: {len(df)}")
            print(f"  Missing values: {missing_values}")
            print(f"  Missing percentage: {missing_percentage}")
            if missing_columns:
                print(f"  Missing columns: {missing_columns}")
            if zero_volume_days:
                print(f"  Zero volume days: {zero_volume_days}")
        
        return result
    
    def check_time_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check time series consistency.
        
        Args:
            df: DataFrame with financial data
            
        Returns:
            Dictionary with time consistency metrics
        """
        # Check if index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if self.verbose:
                print("Error: DataFrame index is not DatetimeIndex")
            return {"is_datetime_index": False}
        
        # Calculate date diffs
        date_diffs = df.index.to_series().diff().dropna()
        
        # Count unique differences
        unique_diffs = date_diffs.unique()
        unique_diff_counts = date_diffs.value_counts().to_dict()
        
        # Check for gaps (more than expected diff)
        expected_diff = pd.Timedelta(days=1)  # Assuming daily data
        if len(unique_diffs) > 1:
            has_gaps = True
            gaps = [str(diff) for diff in unique_diffs if diff > expected_diff]
        else:
            has_gaps = False
            gaps = []
        
        # Check date range
        date_range = {
            "start": df.index.min().strftime("%Y-%m-%d"),
            "end": df.index.max().strftime("%Y-%m-%d"),
            "span_days": (df.index.max() - df.index.min()).days
        }
        
        # Check weekends and holidays (for daily data)
        is_weekday = df.index.dayofweek < 5  # 0-4 are Monday to Friday
        weekend_count = (~is_weekday).sum()
        
        result = {
            "is_datetime_index": True,
            "unique_diffs": [str(diff) for diff in unique_diffs],
            "unique_diff_counts": unique_diff_counts,
            "has_gaps": has_gaps,
            "gaps": gaps,
            "date_range": date_range,
            "weekend_count": weekend_count
        }
        
        if self.verbose:
            print("\nTime Consistency Check:")
            print(f"  Date range: {date_range['start']} to {date_range['end']} ({date_range['span_days']} days)")
            print(f"  Unique time differences: {[str(diff) for diff in unique_diffs]}")
            if has_gaps:
                print(f"  Found gaps: {gaps}")
            if weekend_count > 0:
                print(f"  Weekend data points: {weekend_count}")
        
        return result
    
    def check_price_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check price data quality.
        
        Args:
            df: DataFrame with financial data
            
        Returns:
            Dictionary with price quality metrics
        """
        # Check price columns
        price_columns = [col for col in ['Open', 'High', 'Low', 'Close'] if col in df.columns]
        
        if not price_columns:
            if self.verbose:
                print("Error: No price columns found in DataFrame")
            return {"has_price_columns": False}
        
        # Check for negative prices
        negative_prices = {col: (df[col] < 0).sum() for col in price_columns}
        
        # Check for price consistency (High >= Open >= Low, High >= Close >= Low)
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            invalid_high = (df['High'] < df[['Open', 'Close']].max(axis=1)).sum()
            invalid_low = (df['Low'] > df[['Open', 'Close']].min(axis=1)).sum()
        else:
            invalid_high = None
            invalid_low = None
        
        # Check for extreme outliers (using IQR method)
        outliers = {}
        for col in price_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            outliers[col] = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        # Check for unchanged prices (potential data issues)
        unchanged_prices = {col: (df[col].diff() == 0).sum() for col in price_columns}
        
        # Calculate extreme daily returns
        if 'Close' in df.columns:
            daily_returns = df['Close'].pct_change()
            extreme_returns = (daily_returns.abs() > 0.1).sum()  # >10% daily change
        else:
            extreme_returns = None
        
        result = {
            "has_price_columns": True,
            "negative_prices": negative_prices,
            "invalid_high": invalid_high,
            "invalid_low": invalid_low,
            "outliers": outliers,
            "unchanged_prices": unchanged_prices,
            "extreme_returns": extreme_returns
        }
        
        if self.verbose:
            print("\nPrice Quality Check:")
            if any(negative_prices.values()):
                print(f"  Negative prices: {negative_prices}")
            if invalid_high:
                print(f"  Invalid high prices: {invalid_high}")
            if invalid_low:
                print(f"  Invalid low prices: {invalid_low}")
            if any(outliers.values()):
                print(f"  Outliers detected: {outliers}")
            if any(unchanged_prices.values()):
                print(f"  Unchanged prices: {unchanged_prices}")
            if extreme_returns:
                print(f"  Extreme daily returns (>10%): {extreme_returns}")
        
        return result
    
    def check_volume_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check volume data quality.
        
        Args:
            df: DataFrame with financial data
            
        Returns:
            Dictionary with volume quality metrics
        """
        if 'Volume' not in df.columns:
            if self.verbose:
                print("Error: No Volume column found in DataFrame")
            return {"has_volume_column": False}
        
        # Check for negative volume
        negative_volume = (df['Volume'] < 0).sum()
        
        # Check for zero volume
        zero_volume = (df['Volume'] == 0).sum()
        
        # Check for extreme volume outliers
        Q1 = df['Volume'].quantile(0.25)
        Q3 = df['Volume'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 3 * IQR
        volume_outliers = (df['Volume'] > upper_bound).sum()
        
        # Check for sudden volume changes
        volume_change = df['Volume'].pct_change()
        extreme_volume_changes = (volume_change.abs() > 5).sum()  # >500% volume change
        
        result = {
            "has_volume_column": True,
            "negative_volume": negative_volume,
            "zero_volume": zero_volume,
            "volume_outliers": volume_outliers,
            "extreme_volume_changes": extreme_volume_changes
        }
        
        if self.verbose:
            print("\nVolume Quality Check:")
            if negative_volume > 0:
                print(f"  Negative volume found: {negative_volume}")
            if zero_volume > 0:
                print(f"  Zero volume days: {zero_volume}")
            if volume_outliers > 0:
                print(f"  Volume outliers: {volume_outliers}")
            if extreme_volume_changes > 0:
                print(f"  Extreme volume changes (>500%): {extreme_volume_changes}")
        
        return result
    
    def visualize_data_quality(self, df: pd.DataFrame, output_file: Optional[str] = None) -> None:
        """
        Visualize data quality issues.
        
        Args:
            df: DataFrame with financial data
            output_file: Optional file path to save the visualization
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            print("Error: DataFrame index is not DatetimeIndex")
            return
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 16))
        
        # Plot 1: Price data with outliers highlighted
        price_columns = [col for col in ['Open', 'High', 'Low', 'Close'] if col in df.columns]
        
        if price_columns:
            ax = axes[0]
            df[price_columns].plot(ax=ax)
            
            # Highlight outliers
            for col in price_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
                if not outliers.empty:
                    ax.scatter(outliers.index, outliers[col], color='red', s=50, label=f'{col} outliers')
            
            ax.set_title('Price Data with Outliers')
            ax.set_ylabel('Price')
            ax.legend()
        
        # Plot 2: Daily returns
        if 'Close' in df.columns:
            ax = axes[1]
            daily_returns = df['Close'].pct_change()
            daily_returns.plot(ax=ax)
            
            # Highlight extreme returns
            extreme_returns = daily_returns[daily_returns.abs() > 0.1]
            if not extreme_returns.empty:
                ax.scatter(extreme_returns.index, extreme_returns, color='red', s=50, label='Extreme returns (>10%)')
            
            ax.set_title('Daily Returns')
            ax.set_ylabel('Return')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.legend()
        
        # Plot 3: Volume
        if 'Volume' in df.columns:
            ax = axes[2]
            df['Volume'].plot(ax=ax)
            
            # Highlight volume outliers
            Q1 = df['Volume'].quantile(0.25)
            Q3 = df['Volume'].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 3 * IQR
            volume_outliers = df[df['Volume'] > upper_bound]
            
            if not volume_outliers.empty:
                ax.scatter(volume_outliers.index, volume_outliers['Volume'], color='red', s=50, label='Volume outliers')
            
            ax.set_title('Volume')
            ax.set_ylabel('Volume')
            ax.legend()
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            if self.verbose:
                print(f"Saved visualization to {output_file}")
        else:
            plt.show()
    
    def generate_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive data quality report.
        
        Args:
            df: DataFrame with financial data
            
        Returns:
            Dictionary with all quality metrics
        """
        # Run all checks
        completeness = self.check_completeness(df)
        time_consistency = self.check_time_consistency(df)
        price_quality = self.check_price_quality(df)
        volume_quality = self.check_volume_quality(df)
        
        # Calculate overall quality score (0-100)
        score = 100
        
        # Penalize for missing values
        if completeness["row_count"] > 0:
            for col, count in completeness["missing_values"].items():
                score -= (count / completeness["row_count"]) * 20
        
        # Penalize for missing columns
        score -= len(completeness["missing_columns"]) * 20
        
        # Penalize for time inconsistency
        if time_consistency.get("has_gaps", False):
            score -= 10
        
        # Penalize for price issues
        if price_quality.get("has_price_columns", False):
            # Penalize for negative prices
            for col, count in price_quality["negative_prices"].items():
                if count > 0:
                    score -= 20
                    break
            
            # Penalize for inconsistent OHLC
            if price_quality.get("invalid_high", 0) > 0 or price_quality.get("invalid_low", 0) > 0:
                score -= 15
            
            # Penalize for outliers (less severe)
            if sum(price_quality["outliers"].values()) > 0:
                score -= 5
        
        # Penalize for volume issues
        if volume_quality.get("has_volume_column", False):
            if volume_quality["negative_volume"] > 0:
                score -= 15
            
            # Zero volume is not always an issue, but many zeros might be
            if volume_quality["zero_volume"] > completeness["row_count"] * 0.1:
                score -= 10
        
        # Ensure score is between 0 and 100
        score = max(0, min(100, score))
        
        # Determine quality category
        if score >= 90:
            quality_category = "Excellent"
        elif score >= 75:
            quality_category = "Good"
        elif score >= 50:
            quality_category = "Fair"
        else:
            quality_category = "Poor"
        
        # Compile recommendations
        recommendations = []
        
        if len(completeness["missing_columns"]) > 0:
            recommendations.append(f"Add missing columns: {completeness['missing_columns']}")
        
        if sum(completeness["missing_values"].values()) > 0:
            recommendations.append("Interpolate or fill missing values")
        
        if time_consistency.get("has_gaps", False):
            recommendations.append("Fill time series gaps")
        
        if price_quality.get("has_price_columns", False):
            if sum(price_quality["negative_prices"].values()) > 0:
                recommendations.append("Fix negative prices")
            
            if price_quality.get("invalid_high", 0) > 0 or price_quality.get("invalid_low", 0) > 0:
                recommendations.append("Fix inconsistent OHLC values")
            
            if sum(price_quality["outliers"].values()) > 0:
                recommendations.append("Review price outliers")
        
        if volume_quality.get("has_volume_column", False):
            if volume_quality["negative_volume"] > 0:
                recommendations.append("Fix negative volume values")
        
        # Combine all results
        result = {
            "quality_score": score,
            "quality_category": quality_category,
            "recommendations": recommendations,
            "completeness": completeness,
            "time_consistency": time_consistency,
            "price_quality": price_quality,
            "volume_quality": volume_quality
        }
        
        if self.verbose:
            print("\n===============================")
            print(f"Data Quality Score: {score:.2f}/100 ({quality_category})")
            print("===============================")
            if recommendations:
                print("\nRecommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec}")
        
        return result


def check_multiple_datasets(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """
    Check quality of multiple datasets.
    
    Args:
        data_dict: Dictionary mapping dataset names to DataFrames
        
    Returns:
        Dictionary with quality reports for each dataset
    """
    checker = DataQualityChecker(verbose=False)
    results = {}
    
    for name, df in data_dict.items():
        print(f"Checking quality for {name}...")
        results[name] = checker.generate_report(df)
        
        # Print summary
        score = results[name]["quality_score"]
        category = results[name]["quality_category"]
        print(f"  Quality score: {score:.2f}/100 ({category})")
        
        if results[name]["recommendations"]:
            print("  Key recommendations:")
            for i, rec in enumerate(results[name]["recommendations"][:3], 1):
                print(f"    {i}. {rec}")
        
        print()
    
    return results


def compare_data_sources(df1: pd.DataFrame, df2: pd.DataFrame, source1_name: str = "Source 1", 
                        source2_name: str = "Source 2") -> Dict[str, Any]:
    """
    Compare data from two different sources.
    
    Args:
        df1: DataFrame from first source
        df2: DataFrame from second source
        source1_name: Name of first source
        source2_name: Name of second source
        
    Returns:
        Dictionary with comparison metrics
    """
    # Ensure both DataFrames have DateTime index
    if not isinstance(df1.index, pd.DatetimeIndex) or not isinstance(df2.index, pd.DatetimeIndex):
        print("Error: Both DataFrames must have DatetimeIndex")
        return {}
    
    # Align DataFrames on index
    df1_aligned, df2_aligned = df1.align(df2, join='inner')
    
    if df1_aligned.empty or df2_aligned.empty:
        print("Error: No overlapping dates between the two sources")
        return {
            "overlap_days": 0,
            "source1_only_days": len(df1),
            "source2_only_days": len(df2)
        }
    
    # Calculate overlap statistics
    overlap_days = len(df1_aligned)
    source1_only_days = len(df1) - overlap_days
    source2_only_days = len(df2) - overlap_days
    
    # Compare price columns
    price_columns = [col for col in ['Open', 'High', 'Low', 'Close'] 
                    if col in df1_aligned.columns and col in df2_aligned.columns]
    
    price_differences = {}
    for col in price_columns:
        absolute_diff = (df1_aligned[col] - df2_aligned[col]).abs()
        percentage_diff = ((df1_aligned[col] - df2_aligned[col]) / df2_aligned[col]).abs() * 100
        
        price_differences[col] = {
            "mean_abs_diff": absolute_diff.mean(),
            "max_abs_diff": absolute_diff.max(),
            "mean_pct_diff": percentage_diff.mean(),
            "max_pct_diff": percentage_diff.max(),
            "identical_values": (absolute_diff < 1e-6).sum(),
            "small_diff_values": ((absolute_diff >= 1e-6) & (percentage_diff < 1)).sum(),
            "large_diff_values": (percentage_diff >= 1).sum(),
        }
    
    # Compare volume if available
    volume_difference = None
    if 'Volume' in df1_aligned.columns and 'Volume' in df2_aligned.columns:
        absolute_diff = (df1_aligned['Volume'] - df2_aligned['Volume']).abs()
        percentage_diff = ((df1_aligned['Volume'] - df2_aligned['Volume']) / 
                          df2_aligned['Volume'].replace(0, 1)).abs() * 100
        
        volume_difference = {
            "mean_abs_diff": absolute_diff.mean(),
            "max_abs_diff": absolute_diff.max(),
            "mean_pct_diff": percentage_diff.mean(),
            "max_pct_diff": percentage_diff.max(),
            "identical_values": (absolute_diff < 1e-6).sum(),
            "small_diff_values": ((absolute_diff >= 1e-6) & (percentage_diff < 1)).sum(),
            "large_diff_values": (percentage_diff >= 1).sum(),
        }
    
    # Calculate correlation
    correlations = {}
    for col in price_columns + (['Volume'] if 'Volume' in df1_aligned.columns and 'Volume' in df2_aligned.columns else []):
        correlations[col] = df1_aligned[col].corr(df2_aligned[col])
    
    result = {
        "overlap_days": overlap_days,
        "source1_only_days": source1_only_days,
        "source2_only_days": source2_only_days,
        "price_differences": price_differences,
        "volume_difference": volume_difference,
        "correlations": correlations
    }
    
    # Print summary
    print(f"Comparison between {source1_name} and {source2_name}:")
    print(f"  Overlapping days: {overlap_days}")
    print(f"  Days only in {source1_name}: {source1_only_days}")
    print(f"  Days only in {source2_name}: {source2_only_days}")
    
    if price_differences:
        print("\nPrice differences:")
        for col, diff in price_differences.items():
            print(f"  {col}:")
            print(f"    Mean absolute difference: {diff['mean_abs_diff']:.6f}")
            print(f"    Mean percentage difference: {diff['mean_pct_diff']:.6f}%")
            print(f"    Identical values: {diff['identical_values']} ({diff['identical_values']/overlap_days*100:.2f}%)")
            print(f"    Large differences (>1%): {diff['large_diff_values']} ({diff['large_diff_values']/overlap_days*100:.2f}%)")
    
    if correlations:
        print("\nCorrelations:")
        for col, corr in correlations.items():
            print(f"  {col}: {corr:.6f}")
    
    return result


def visualize_comparison(df1: pd.DataFrame, df2: pd.DataFrame, source1_name: str = "Source 1", 
                        source2_name: str = "Source 2", output_file: Optional[str] = None) -> None:
    """
    Visualize comparison between two data sources.
    
    Args:
        df1: DataFrame from first source
        df2: DataFrame from second source
        source1_name: Name of first source
        source2_name: Name of second source
        output_file: Optional file path to save the visualization
    """
    # Ensure both DataFrames have DateTime index
    if not isinstance(df1.index, pd.DatetimeIndex) or not isinstance(df2.index, pd.DatetimeIndex):
        print("Error: Both DataFrames must have DatetimeIndex")
        return
    
    # Align DataFrames on index
    df1_aligned, df2_aligned = df1.align(df2, join='inner')
    
    if df1_aligned.empty or df2_aligned.empty:
        print("Error: No overlapping dates between the two sources")
        return
    
    # Create comparison plots
    columns_to_plot = [col for col in ['Close', 'Volume'] 
                      if col in df1_aligned.columns and col in df2_aligned.columns]
    
    if not columns_to_plot:
        print("Error: No common columns to plot")
        return
    
    # Create a figure with multiple subplots
    num_plots = len(columns_to_plot) + 1  # +1 for difference plot
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))
    
    if num_plots == 1:
        axes = [axes]
    
    # Set the style
    sns.set_style("whitegrid")
    
    # Plot each column
    for i, col in enumerate(columns_to_plot):
        ax = axes[i]
        
        # Plot both sources
        ax.plot(df1_aligned.index, df1_aligned[col], label=source1_name, alpha=0.7)
        ax.plot(df2_aligned.index, df2_aligned[col], label=source2_name, alpha=0.7)
        
        ax.set_title(f'{col} Comparison')
        ax.set_ylabel(col)
        ax.legend()
        
        # Format x-axis
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot the difference
    ax = axes[-1]
    for col in columns_to_plot:
        if col == 'Volume':
            continue  # Skip volume for difference plot (different scale)
        
        pct_diff = ((df1_aligned[col] - df2_aligned[col]) / df2_aligned[col]) * 100
        ax.plot(df1_aligned.index, pct_diff, label=f'{col} % Diff')
    
    ax.set_title('Percentage Difference')
    ax.set_ylabel('Difference (%)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend()
    
    # Format x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Saved visualization to {output_file}")
    else:
        plt.show() 