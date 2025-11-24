"""
EDA Utility Functions for Financial News Sentiment Analysis

This module provides reusable functions for exploratory data analysis
on financial news datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def download_nltk_data():
    """Download required NLTK data if not already present"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)


def preprocess_text(text, stop_words=None, lemmatizer=None):
    """
    Clean and preprocess text for analysis.
    
    Parameters:
    -----------
    text : str
        Text to preprocess
    stop_words : set, optional
        Set of stopwords to remove. If None, downloads default English stopwords.
    lemmatizer : WordNetLemmatizer, optional
        Lemmatizer instance. If None, creates a new one.
    
    Returns:
    --------
    list
        List of preprocessed tokens
    """
    if pd.isna(text):
        return []
    
    if stop_words is None:
        download_nltk_data()
        stop_words = set(stopwords.words('english'))
    
    if lemmatizer is None:
        download_nltk_data()
        lemmatizer = WordNetLemmatizer()
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and short words
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens


def extract_domain(publisher):
    """
    Extract domain from publisher field (if it's an email).
    
    Parameters:
    -----------
    publisher : str
        Publisher name or email address
    
    Returns:
    --------
    str
        Domain name or original publisher name
    """
    if pd.isna(publisher):
        return None
    publisher_str = str(publisher).lower()
    if '@' in publisher_str:
        return publisher_str.split('@')[1]
    return publisher_str


def calculate_headline_stats(df, headline_col='headline'):
    """
    Calculate headline length statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing headlines
    headline_col : str
        Name of the headline column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with headline statistics
    """
    df = df.copy()
    df['headline_length'] = df[headline_col].str.len()
    df['headline_word_count'] = df[headline_col].str.split().str.len()
    return df


def get_financial_keywords():
    """
    Get list of common financial keywords for analysis.
    
    Returns:
    --------
    list
        List of financial keywords
    """
    return [
        'price', 'target', 'stock', 'share', 'earnings', 'revenue', 'profit',
        'fda', 'approval', 'merger', 'acquisition', 'dividend', 'ipo',
        'analyst', 'rating', 'upgrade', 'downgrade', 'forecast', 'guidance',
        'quarter', 'annual', 'report', 'beat', 'miss', 'expectation'
    ]


def analyze_keyword_frequency(df, tokens_col='processed_tokens', keywords=None):
    """
    Analyze frequency of financial keywords in processed tokens.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with processed tokens
    tokens_col : str
        Name of the column containing processed tokens
    keywords : list, optional
        List of keywords to search for. If None, uses default financial keywords.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with keyword frequencies
    """
    if keywords is None:
        keywords = get_financial_keywords()
    
    keyword_counts = {}
    for keyword in keywords:
        count = sum(1 for tokens in df[tokens_col] if keyword in tokens)
        keyword_counts[keyword] = count
    
    keyword_df = pd.DataFrame(list(keyword_counts.items()), 
                            columns=['Keyword', 'Frequency'])
    keyword_df = keyword_df.sort_values('Frequency', ascending=False)
    
    return keyword_df


def detect_publication_spikes(daily_counts, z_threshold=2):
    """
    Detect days with unusually high publication frequency.
    
    Parameters:
    -----------
    daily_counts : pd.Series
        Series with daily article counts (indexed by date)
    z_threshold : float
        Z-score threshold for spike detection (default: 2)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with spike days and their z-scores
    """
    daily_freq_df = daily_counts.reset_index()
    daily_freq_df.columns = ['date', 'count']
    
    mean_daily = daily_counts.mean()
    std_daily = daily_counts.std()
    daily_freq_df['z_score'] = (daily_freq_df['count'] - mean_daily) / std_daily
    
    spike_days = daily_freq_df[daily_freq_df['z_score'] > z_threshold].sort_values(
        'count', ascending=False
    )
    
    return spike_days


def prepare_date_features(df, date_col='date'):
    """
    Extract date-related features from datetime column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with date column
    date_col : str
        Name of the date column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added date features
    """
    df = df.copy()
    
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
        
        # Extract date components
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['day_of_week'] = df[date_col].dt.day_name()
        df['hour'] = df[date_col].dt.hour
    
    return df

