"""
Feature importance analysis for financial data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr, spearmanr


def calculate_correlation_importance(df: pd.DataFrame, target_column: str, 
                                    method: str = 'pearson') -> pd.DataFrame:
    """
    Calculate feature importance based on correlation with target.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        method: Correlation method ('pearson' or 'spearman')
        
    Returns:
        DataFrame with correlation importance
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    if target_column not in numeric_df.columns:
        raise ValueError(f"Target column '{target_column}' is not numeric")
    
    # Calculate correlations
    if method == 'pearson':
        corr_series = numeric_df.drop(columns=[target_column]).apply(
            lambda x: pearsonr(x.fillna(0), numeric_df[target_column].fillna(0))[0]
            if not (x.fillna(0) == x.fillna(0).iloc[0]).all() else 0
        )
    elif method == 'spearman':
        corr_series = numeric_df.drop(columns=[target_column]).apply(
            lambda x: spearmanr(x.fillna(0), numeric_df[target_column].fillna(0))[0]
            if not (x.fillna(0) == x.fillna(0).iloc[0]).all() else 0
        )
    else:
        raise ValueError(f"Invalid correlation method: {method}. Use 'pearson' or 'spearman'")
    
    # Get absolute correlations for ranking
    abs_corr = corr_series.abs()
    
    # Create result DataFrame
    result = pd.DataFrame({
        'feature': abs_corr.index,
        'correlation': corr_series.values,
        'abs_correlation': abs_corr.values
    })
    
    # Sort by absolute correlation
    result = result.sort_values('abs_correlation', ascending=False).reset_index(drop=True)
    
    return result


def calculate_mutual_information(df: pd.DataFrame, target_column: str, 
                               classification: bool = False) -> pd.DataFrame:
    """
    Calculate feature importance using mutual information.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        classification: Whether this is a classification problem
        
    Returns:
        DataFrame with mutual information importance
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    if target_column not in numeric_df.columns:
        raise ValueError(f"Target column '{target_column}' is not numeric")
    
    # Split data
    X = numeric_df.drop(columns=[target_column]).fillna(0)
    y = numeric_df[target_column].fillna(0)
    
    # Calculate mutual information
    if classification:
        mi_scores = mutual_info_classif(X, y)
    else:
        mi_scores = mutual_info_regression(X, y)
    
    # Create result DataFrame
    result = pd.DataFrame({
        'feature': X.columns,
        'mutual_information': mi_scores
    })
    
    # Sort by mutual information
    result = result.sort_values('mutual_information', ascending=False).reset_index(drop=True)
    
    return result


def calculate_random_forest_importance(df: pd.DataFrame, target_column: str, 
                                     classification: bool = False, 
                                     n_estimators: int = 100,
                                     random_state: int = 42) -> pd.DataFrame:
    """
    Calculate feature importance using Random Forest.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        classification: Whether this is a classification problem
        n_estimators: Number of trees in the forest
        random_state: Random state for reproducibility
        
    Returns:
        DataFrame with Random Forest importance
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    if target_column not in numeric_df.columns:
        raise ValueError(f"Target column '{target_column}' is not numeric")
    
    # Split data
    X = numeric_df.drop(columns=[target_column]).fillna(0)
    y = numeric_df[target_column].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Initialize and fit Random Forest
    if classification:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    
    model.fit(X_scaled, y)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create result DataFrame
    result = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    })
    
    # Sort by importance
    result = result.sort_values('importance', ascending=False).reset_index(drop=True)
    
    return result


def calculate_permutation_importance(df: pd.DataFrame, target_column: str,
                                   classification: bool = False,
                                   n_estimators: int = 100,
                                   n_repeats: int = 10,
                                   random_state: int = 42) -> pd.DataFrame:
    """
    Calculate feature importance using permutation importance.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        classification: Whether this is a classification problem
        n_estimators: Number of trees in the forest
        n_repeats: Number of times to permute each feature
        random_state: Random state for reproducibility
        
    Returns:
        DataFrame with permutation importance
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    if target_column not in numeric_df.columns:
        raise ValueError(f"Target column '{target_column}' is not numeric")
    
    # Split data
    X = numeric_df.drop(columns=[target_column]).fillna(0)
    y = numeric_df[target_column].fillna(0)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Initialize and fit Random Forest
    if classification:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    
    model.fit(X_train_scaled, y_train)
    
    # Calculate permutation importance
    perm_importance = permutation_importance(
        model, X_test_scaled, y_test, n_repeats=n_repeats, random_state=random_state
    )
    
    # Create result DataFrame
    result = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    })
    
    # Sort by importance
    result = result.sort_values('importance_mean', ascending=False).reset_index(drop=True)
    
    return result


def combined_feature_importance(df: pd.DataFrame, target_column: str,
                              classification: bool = False,
                              n_estimators: int = 100,
                              random_state: int = 42) -> pd.DataFrame:
    """
    Calculate feature importance using multiple methods and combine them.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        classification: Whether this is a classification problem
        n_estimators: Number of trees in the forest
        random_state: Random state for reproducibility
        
    Returns:
        DataFrame with combined importance scores
    """
    # Calculate importance using different methods
    corr_importance = calculate_correlation_importance(df, target_column)
    mi_importance = calculate_mutual_information(df, target_column, classification)
    rf_importance = calculate_random_forest_importance(
        df, target_column, classification, n_estimators, random_state
    )
    
    # Create a combined DataFrame
    features = list(set(
        list(corr_importance['feature']) + 
        list(mi_importance['feature']) + 
        list(rf_importance['feature'])
    ))
    
    combined = pd.DataFrame({'feature': features})
    
    # Add importance scores from different methods
    combined = combined.merge(
        corr_importance[['feature', 'abs_correlation']], 
        on='feature', how='left'
    ).fillna(0)
    
    combined = combined.merge(
        mi_importance[['feature', 'mutual_information']], 
        on='feature', how='left'
    ).fillna(0)
    
    combined = combined.merge(
        rf_importance[['feature', 'importance']], 
        on='feature', how='left'
    ).fillna(0)
    
    # Normalize each importance score (0-1 scale)
    for col in ['abs_correlation', 'mutual_information', 'importance']:
        if combined[col].max() > 0:
            combined[col] = combined[col] / combined[col].max()
    
    # Calculate combined score (average of all methods)
    combined['combined_score'] = combined[['abs_correlation', 'mutual_information', 'importance']].mean(axis=1)
    
    # Sort by combined score
    combined = combined.sort_values('combined_score', ascending=False).reset_index(drop=True)
    
    return combined


def plot_feature_importance(importance_df: pd.DataFrame, 
                          score_column: str = 'combined_score',
                          title: str = 'Feature Importance',
                          top_n: int = 20,
                          figsize: Tuple[int, int] = (10, 8),
                          output_file: Optional[str] = None) -> None:
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with feature importance
        score_column: Column to use for importance score
        title: Plot title
        top_n: Number of top features to plot
        figsize: Figure size
        output_file: Path to save the plot, or None to display
    """
    if 'feature' not in importance_df.columns or score_column not in importance_df.columns:
        raise ValueError(f"DataFrame must contain 'feature' and '{score_column}' columns")
    
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Create horizontal bar plot
    ax = sns.barplot(
        x=score_column, 
        y='feature', 
        data=top_features.sort_values(score_column),
        palette='viridis'
    )
    
    # Add title and labels
    plt.title(title, fontsize=14)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    
    # Add grid
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Add importance values as text
    for i, value in enumerate(top_features.sort_values(score_column)[score_column]):
        ax.text(value + 0.01, i, f'{value:.4f}', va='center')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def get_top_n_features(df: pd.DataFrame, target_column: str, n: int = 20,
                      classification: bool = False) -> List[str]:
    """
    Get the top N most important features.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        n: Number of top features to return
        classification: Whether this is a classification problem
        
    Returns:
        List of top feature names
    """
    # Calculate combined importance
    importance_df = combined_feature_importance(df, target_column, classification)
    
    # Return top N feature names
    return importance_df.head(n)['feature'].tolist()


def feature_correlation_heatmap(df: pd.DataFrame, 
                              features: Optional[List[str]] = None,
                              figsize: Tuple[int, int] = (12, 10),
                              output_file: Optional[str] = None) -> None:
    """
    Plot a correlation heatmap for features.
    
    Args:
        df: DataFrame with features
        features: List of features to include, or None for all
        figsize: Figure size
        output_file: Path to save the plot, or None to display
    """
    # Select features if provided
    if features:
        # Check that all features are in DataFrame
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")
        
        data = df[features]
    else:
        data = df
    
    # Calculate correlation matrix
    corr = data.corr()
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Create heatmap
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
    
    plt.title('Feature Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def feature_target_correlation_analysis(df: pd.DataFrame, target_column: str,
                                      top_n: int = 10,
                                      figsize: Tuple[int, int] = (15, 10),
                                      output_file: Optional[str] = None) -> None:
    """
    Analyze and visualize feature-target correlations.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        top_n: Number of top features to analyze
        figsize: Figure size
        output_file: Path to save the plot, or None to display
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Calculate correlation importance
    corr_importance = calculate_correlation_importance(df, target_column)
    
    # Get top features
    top_features = corr_importance.head(top_n)['feature'].tolist()
    
    # Create a subplot grid
    n_cols = 2
    n_rows = (len(top_features) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes for easy iteration
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Plot scatter plots for each feature
    for i, feature in enumerate(top_features):
        if i < len(axes):
            # Create scatter plot
            ax = axes[i]
            sns.scatterplot(
                x=feature, 
                y=target_column, 
                data=df,
                alpha=0.6, 
                ax=ax
            )
            
            # Add regression line
            sns.regplot(
                x=feature, 
                y=target_column, 
                data=df,
                scatter=False, 
                ax=ax,
                color='red'
            )
            
            # Add correlation coefficient to title
            corr = df[[feature, target_column]].corr().iloc[0, 1]
            ax.set_title(f'{feature} (corr: {corr:.4f})', fontsize=12)
            
            # Add grid
            ax.grid(linestyle='--', alpha=0.6)
    
    # Hide any unused subplots
    for i in range(len(top_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Feature-Target Correlation Analysis (Target: {target_column})', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def run_feature_importance_analysis(df: pd.DataFrame, target_column: str, 
                                  output_dir: str = 'data/feature_analysis',
                                  classification: bool = False,
                                  n_features: int = 20) -> Dict[str, Any]:
    """
    Run a complete feature importance analysis.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        output_dir: Directory to save output plots
        classification: Whether this is a classification problem
        n_features: Number of top features to analyze
        
    Returns:
        Dictionary with analysis results
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate importance using multiple methods
    print(f"Calculating feature importance for target: {target_column}")
    combined_importance = combined_feature_importance(df, target_column, classification)
    
    # Get top features
    top_features = combined_importance.head(n_features)['feature'].tolist()
    
    # Plot feature importance
    print("Plotting feature importance...")
    plot_feature_importance(
        combined_importance, 
        title=f'Feature Importance (Target: {target_column})',
        output_file=os.path.join(output_dir, 'feature_importance.png')
    )
    
    # Plot feature correlation heatmap
    print("Plotting feature correlation heatmap...")
    feature_correlation_heatmap(
        df[top_features + [target_column]], 
        output_file=os.path.join(output_dir, 'feature_correlation.png')
    )
    
    # Plot feature-target correlation analysis
    print("Plotting feature-target correlation analysis...")
    feature_target_correlation_analysis(
        df, 
        target_column,
        top_n=min(10, n_features),
        output_file=os.path.join(output_dir, 'feature_target_correlation.png')
    )
    
    # Save feature importance to CSV
    print("Saving feature importance to CSV...")
    combined_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    # Return results
    return {
        'importance_df': combined_importance,
        'top_features': top_features,
        'output_dir': output_dir
    } 