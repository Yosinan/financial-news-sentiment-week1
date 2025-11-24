"""
Technical Analysis Utility Functions

This module provides reusable functions for technical analysis using TA-Lib
and financial metrics calculations.
"""

import pandas as pd
import numpy as np
import talib
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional


def download_stock_data(tickers: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
    """
    Download stock price data using yfinance.
    
    Parameters:
    -----------
    tickers : List[str]
        List of stock ticker symbols
    start_date : datetime
        Start date for data download
    end_date : datetime
        End date for data download
    
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary mapping ticker symbols to DataFrames with OHLCV data
    """
    stock_data = {}
    
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start_date, end=end_date)
            
            if not df.empty:
                # Standardize column names
                df.columns = [col.lower() for col in df.columns]
                df.index.name = 'date'
                df = df.reset_index()
                
                # Ensure required columns exist
                required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                if all(col in df.columns for col in required_cols):
                    stock_data[ticker] = df
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    
    return stock_data


def prepare_data_for_talib(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame for TA-Lib calculations.
    TA-Lib requires numpy arrays with specific data types.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLCV data
    
    Returns:
    --------
    pd.DataFrame
        Prepared DataFrame with date as index
    """
    df = df.copy()
    
    # Ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    
    # Sort by date
    df = df.sort_index()
    
    # Ensure required columns exist and are numeric
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove any rows with NaN in critical columns
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    
    return df


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate various technical indicators using TA-Lib.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLCV data (date as index)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added technical indicator columns
    """
    df = df.copy()
    
    # Convert to numpy arrays for TA-Lib
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    close = df['close'].values.astype(float)
    open_price = df['open'].values.astype(float)
    volume = df['volume'].values.astype(float)
    
    # Moving Averages
    df['SMA_20'] = talib.SMA(close, timeperiod=20)
    df['SMA_50'] = talib.SMA(close, timeperiod=50)
    df['SMA_200'] = talib.SMA(close, timeperiod=200)
    df['EMA_12'] = talib.EMA(close, timeperiod=12)
    df['EMA_26'] = talib.EMA(close, timeperiod=26)
    
    # RSI (Relative Strength Index)
    df['RSI'] = talib.RSI(close, timeperiod=14)
    
    # MACD (Moving Average Convergence Divergence)
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal
    df['MACD_hist'] = macd_hist
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['BB_upper'] = bb_upper
    df['BB_middle'] = bb_middle
    df['BB_lower'] = bb_lower
    
    # Stochastic Oscillator
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
    df['Stoch_K'] = slowk
    df['Stoch_D'] = slowd
    
    # Average True Range (ATR)
    df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
    
    # On Balance Volume (OBV)
    df['OBV'] = talib.OBV(close, volume)
    
    # Average Directional Index (ADX)
    df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
    
    return df


def calculate_financial_metrics(df: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict:
    """
    Calculate financial metrics including returns, volatility, Sharpe ratio, and drawdown.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price data
    risk_free_rate : float
        Annual risk-free rate (default: 0.02 for 2%)
    
    Returns:
    --------
    Dict
        Dictionary containing financial metrics
    """
    df = df.copy()
    
    # Daily returns
    df['daily_return'] = df['close'].pct_change()
    
    # Cumulative returns
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
    
    # Volatility (rolling 30-day, annualized)
    df['volatility_30d'] = df['daily_return'].rolling(window=30).std() * np.sqrt(252)
    
    # Sharpe ratio (annualized)
    excess_returns = df['daily_return'] - (risk_free_rate / 252)
    df['sharpe_ratio'] = (excess_returns.rolling(window=252).mean() / 
                         excess_returns.rolling(window=252).std()) * np.sqrt(252)
    
    # Maximum drawdown
    cumulative = (1 + df['daily_return']).cumprod()
    running_max = cumulative.expanding().max()
    df['drawdown'] = (cumulative - running_max) / running_max
    df['max_drawdown'] = df['drawdown'].expanding().min()
    
    metrics = {
        'total_return': df['cumulative_return'].iloc[-1] if len(df) > 0 else 0,
        'volatility': df['volatility_30d'].iloc[-1] if len(df) > 0 else 0,
        'sharpe_ratio': df['sharpe_ratio'].iloc[-1] if len(df) > 0 else 0,
        'max_drawdown': df['max_drawdown'].iloc[-1] if len(df) > 0 else 0,
        'avg_daily_return': df['daily_return'].mean(),
        'data': df
    }
    
    return metrics


def get_rsi_signal(rsi_value: float) -> str:
    """
    Interpret RSI value as trading signal.
    
    Parameters:
    -----------
    rsi_value : float
        RSI value
    
    Returns:
    --------
    str
        Signal: 'Overbought', 'Oversold', or 'Neutral'
    """
    if rsi_value > 70:
        return "Overbought"
    elif rsi_value < 30:
        return "Oversold"
    else:
        return "Neutral"


def get_macd_signal(macd: float, signal: float) -> str:
    """
    Interpret MACD crossover as trading signal.
    
    Parameters:
    -----------
    macd : float
        MACD line value
    signal : float
        MACD signal line value
    
    Returns:
    --------
    str
        Signal: 'Bullish' or 'Bearish'
    """
    if macd > signal:
        return "Bullish"
    else:
        return "Bearish"


def get_trend_signal(close: float, sma_50: float, sma_200: float) -> str:
    """
    Determine trend based on moving averages.
    
    Parameters:
    -----------
    close : float
        Current closing price
    sma_50 : float
        50-day SMA value
    sma_200 : float
        200-day SMA value
    
    Returns:
    --------
    str
        Trend: 'Uptrend', 'Downtrend', or 'Sideways'
    """
    if close > sma_50 > sma_200:
        return "Uptrend"
    elif close < sma_50 < sma_200:
        return "Downtrend"
    else:
        return "Sideways"

