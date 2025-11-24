"""
Correlation Analysis Utility Functions

This module provides reusable functions for correlation analysis between
news sentiment and stock returns.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Optional, Tuple


def calculate_daily_returns(df: pd.DataFrame, 
                           price_col: str = 'close',
                           date_col: str = 'date') -> pd.DataFrame:
    """
    Calculate daily returns from stock price data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with stock price data
    price_col : str
        Name of price column
    date_col : str
        Name of date column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added return columns
    """
    df = df.copy()
    df = df.sort_values(date_col)
    
    # Calculate daily returns (percentage change)
    df['daily_return'] = df[price_col].pct_change() * 100
    
    # Calculate log returns (alternative method)
    df['log_return'] = np.log(df[price_col] / df[price_col].shift(1)) * 100
    
    return df


def merge_sentiment_returns(sentiment_df: pd.DataFrame,
                            returns_df: pd.DataFrame,
                            stock_col: str = 'stock',
                            date_col: str = 'date') -> pd.DataFrame:
    """
    Merge sentiment and returns data by stock and date.
    
    Parameters:
    -----------
    sentiment_df : pd.DataFrame
        DataFrame with sentiment data (must have stock and date columns)
    returns_df : pd.DataFrame
        DataFrame with returns data (must have stock and date columns)
    stock_col : str
        Name of stock column
    date_col : str
        Name of date column
    
    Returns:
    --------
    pd.DataFrame
        Merged DataFrame with sentiment and returns
    """
    # Ensure dates are datetime
    sentiment_df = sentiment_df.copy()
    returns_df = returns_df.copy()
    
    sentiment_df[date_col] = pd.to_datetime(sentiment_df[date_col])
    returns_df[date_col] = pd.to_datetime(returns_df[date_col])
    
    # Merge
    merged = pd.merge(
        sentiment_df,
        returns_df,
        on=[stock_col, date_col],
        how='inner'
    )
    
    return merged.sort_values([stock_col, date_col])


def calculate_correlation(sentiment: pd.Series,
                          returns: pd.Series,
                          method: str = 'pearson') -> Tuple[float, float]:
    """
    Calculate correlation between sentiment and returns.
    
    Parameters:
    -----------
    sentiment : pd.Series
        Sentiment scores
    returns : pd.Series
        Return values
    method : str
        'pearson' or 'spearman'
    
    Returns:
    --------
    Tuple[float, float]
        (correlation_coefficient, p_value)
    """
    # Remove NaN values
    clean_data = pd.DataFrame({
        'sentiment': sentiment,
        'returns': returns
    }).dropna()
    
    if len(clean_data) < 2:
        return np.nan, np.nan
    
    if method.lower() == 'pearson':
        corr, p_val = pearsonr(clean_data['sentiment'], clean_data['returns'])
    elif method.lower() == 'spearman':
        corr, p_val = spearmanr(clean_data['sentiment'], clean_data['returns'])
    else:
        raise ValueError(f"Method must be 'pearson' or 'spearman', got {method}")
    
    return corr, p_val


def analyze_correlation_by_stock(df: pd.DataFrame,
                                  stock_col: str = 'stock',
                                  sentiment_col: str = 'avg_sentiment',
                                  returns_col: str = 'daily_return',
                                  min_data_points: int = 10) -> pd.DataFrame:
    """
    Calculate correlation for each stock separately.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sentiment and returns data
    stock_col : str
        Name of stock column
    sentiment_col : str
        Name of sentiment column
    returns_col : str
        Name of returns column
    min_data_points : int
        Minimum data points required for correlation calculation
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with correlation results per stock
    """
    correlations = []
    
    for stock in df[stock_col].unique():
        stock_data = df[df[stock_col] == stock]
        
        if len(stock_data) >= min_data_points:
            corr_pearson, p_pearson = calculate_correlation(
                stock_data[sentiment_col],
                stock_data[returns_col],
                method='pearson'
            )
            
            corr_spearman, p_spearman = calculate_correlation(
                stock_data[sentiment_col],
                stock_data[returns_col],
                method='spearman'
            )
            
            correlations.append({
                'Stock': stock,
                'Pearson_Correlation': corr_pearson,
                'Pearson_P_Value': p_pearson,
                'Spearman_Correlation': corr_spearman,
                'Spearman_P_Value': p_spearman,
                'Data_Points': len(stock_data),
                'Significant': 'Yes' if p_pearson < 0.05 else 'No'
            })
    
    if correlations:
        return pd.DataFrame(correlations).sort_values('Pearson_Correlation', ascending=False)
    else:
        return pd.DataFrame()


def analyze_lag_correlation(df: pd.DataFrame,
                            stock_col: str = 'stock',
                            sentiment_col: str = 'avg_sentiment',
                            returns_col: str = 'daily_return',
                            lags: range = range(-2, 3)) -> pd.DataFrame:
    """
    Analyze correlation at different time lags.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sentiment and returns data
    stock_col : str
        Name of stock column
    sentiment_col : str
        Name of sentiment column
    returns_col : str
        Name of returns column
    lags : range
        Range of lags to test
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with correlation results at different lags
    """
    lag_correlations = []
    
    for lag in lags:
        lag_data = df.copy()
        lag_data = lag_data.sort_values([stock_col, 'date'])
        
        if lag != 0:
            # Shift returns by lag days
            lag_data['returns_lag'] = lag_data.groupby(stock_col)[returns_col].shift(-lag)
            lag_data_clean = lag_data.dropna(subset=[sentiment_col, 'returns_lag'])
            
            if len(lag_data_clean) > 10:
                corr, p_val = calculate_correlation(
                    lag_data_clean[sentiment_col],
                    lag_data_clean['returns_lag'],
                    method='pearson'
                )
                lag_correlations.append({
                    'Lag': lag,
                    'Correlation': corr,
                    'P_Value': p_val,
                    'Data_Points': len(lag_data_clean)
                })
        else:
            # Lag 0 (no shift)
            lag_data_clean = lag_data.dropna(subset=[sentiment_col, returns_col])
            if len(lag_data_clean) > 10:
                corr, p_val = calculate_correlation(
                    lag_data_clean[sentiment_col],
                    lag_data_clean[returns_col],
                    method='pearson'
                )
                lag_correlations.append({
                    'Lag': lag,
                    'Correlation': corr,
                    'P_Value': p_val,
                    'Data_Points': len(lag_data_clean)
                })
    
    if lag_correlations:
        return pd.DataFrame(lag_correlations).sort_values('Lag')
    else:
        return pd.DataFrame()

