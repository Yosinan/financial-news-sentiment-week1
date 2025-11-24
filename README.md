# Financial News Sentiment Analysis

Predicting Price Moves with News Sentiment - Week 1 Challenge

This project focuses on analyzing financial news data to discover correlations between news sentiment and stock market movements. The challenge enhances skills in Data Engineering (DE), Financial Analytics (FA), and Machine Learning Engineering (MLE).

## Business Objective

Nova Financial Solutions aims to enhance its predictive analytics capabilities to significantly boost financial forecasting accuracy and operational efficiency through advanced data analysis. The primary tasks are:

1. **Sentiment Analysis**: Perform sentiment analysis on news headlines to quantify tone and sentiment
2. **Correlation Analysis**: Establish statistical correlations between news sentiment and stock price movements

## Project Structure

```
financial-news-sentiment-week1/
├── .github/
│   └── workflows/
│       └── unittests.yml
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   └── __init__.py
├── notebooks/
│   ├── __init__.py
│   └── README.md
├── tests/
│   └── __init__.py
└── scripts/
    ├── __init__.py
    └── README.md
```

## Tasks Overview

### Task 1: Git and GitHub
- Setting up Python environment
- Git version control
- CI/CD setup
- EDA on financial news data

### Task 2: Quantitative Analysis
- Using PyNance and TA-Lib
- Technical indicators (MA, RSI, MACD)
- Financial metrics and visualizations

### Task 3: Correlation Analysis
- Sentiment analysis on headlines
- Correlation between sentiment and stock returns
- Statistical analysis

## Key Dates

- Challenge Introduction: 10:30 AM UTC, Wednesday, 19 Nov 2025
- Interim Submission: 8:00 PM UTC, Sunday, 23 Nov 2025
- Final Submission: 8:00 PM UTC, Tuesday, 25 Nov 2025

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## References

- [TA-Lib Python](https://github.com/ta-lib/ta-lib-python)
- [PyNance](https://github.com/mqandil/pynance)
- [TextBlob](https://textblob.readthedocs.io/en/dev/)

