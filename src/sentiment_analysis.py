"""
Sentiment Analysis Utility Functions

This module provides reusable functions for sentiment analysis on financial news.
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from typing import Tuple, Optional


def analyze_sentiment_textblob(text: str) -> Tuple[float, float]:
    """
    Analyze sentiment using TextBlob.
    
    Parameters:
    -----------
    text : str
        Text to analyze
    
    Returns:
    --------
    Tuple[float, float]
        (polarity, subjectivity) scores
        - Polarity: -1 (negative) to 1 (positive)
        - Subjectivity: 0 (objective) to 1 (subjective)
    """
    if pd.isna(text) or text == '':
        return 0.0, 0.0
    
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    return polarity, subjectivity


def classify_sentiment(polarity: float, threshold: float = 0.1) -> str:
    """
    Classify sentiment based on polarity score.
    
    Parameters:
    -----------
    polarity : float
        Sentiment polarity score
    threshold : float
        Threshold for positive/negative classification (default: 0.1)
    
    Returns:
    --------
    str
        'Positive', 'Negative', or 'Neutral'
    """
    if polarity > threshold:
        return 'Positive'
    elif polarity < -threshold:
        return 'Negative'
    else:
        return 'Neutral'


def aggregate_daily_sentiment(df: pd.DataFrame, 
                              stock_col: str = 'stock',
                              date_col: str = 'date',
                              sentiment_col: str = 'sentiment_polarity') -> pd.DataFrame:
    """
    Aggregate sentiment by stock and date.
    If multiple articles for same stock on same day, calculate average sentiment.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sentiment data
    stock_col : str
        Name of stock column
    date_col : str
        Name of date column
    sentiment_col : str
        Name of sentiment polarity column
    
    Returns:
    --------
    pd.DataFrame
        Aggregated daily sentiment by stock and date
    """
    # Normalize date to date only
    df = df.copy()
    df['date_only'] = pd.to_datetime(df[date_col]).dt.date
    
    # Aggregate
    daily_sentiment = df.groupby([stock_col, 'date_only']).agg({
        sentiment_col: ['mean', 'count'],
        'sentiment_subjectivity': 'mean' if 'sentiment_subjectivity' in df.columns else 'first'
    }).reset_index()
    
    # Flatten column names
    daily_sentiment.columns = [stock_col, 'date', 'avg_sentiment', 'article_count', 'avg_subjectivity']
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    
    return daily_sentiment


def apply_sentiment_analysis(df: pd.DataFrame, 
                            text_col: str = 'headline') -> pd.DataFrame:
    """
    Apply sentiment analysis to a DataFrame column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with text data
    text_col : str
        Name of text column to analyze
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added sentiment columns
    """
    df = df.copy()
    
    # Apply sentiment analysis
    sentiment_results = df[text_col].apply(analyze_sentiment_textblob)
    df['sentiment_polarity'] = [result[0] for result in sentiment_results]
    df['sentiment_subjectivity'] = [result[1] for result in sentiment_results]
    
    # Classify sentiment
    df['sentiment_label'] = df['sentiment_polarity'].apply(classify_sentiment)
    
    return df

