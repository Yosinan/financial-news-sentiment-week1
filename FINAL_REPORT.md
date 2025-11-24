# Predicting Stock Movements with News Sentiment: A Data-Driven Analysis

*How financial news headlines shape stock market swings — and what the data tells us*

---

## Executive Summary

In an era where information moves markets in milliseconds, understanding the relationship between financial news sentiment and stock price movements has never been more critical. This comprehensive analysis dives deep into a large corpus of financial news data to discover correlations between news sentiment and stock market performance.

Through rigorous exploratory data analysis, quantitative technical analysis, and statistical correlation studies, we've uncovered fascinating insights about how news headlines influence stock prices. This report presents our methodology, findings, and actionable recommendations for leveraging news sentiment as a predictive tool in financial markets.

**Key Findings:**
- News sentiment shows measurable correlation with stock returns, though strength varies by stock
- Technical indicators provide complementary signals to sentiment analysis
- Temporal patterns in news publication reveal optimal trading windows
- Certain stocks exhibit stronger sentiment-return relationships than others

---

## 1. Introduction: The Power of Words in Financial Markets

Financial markets are driven by information. Every earnings report, analyst upgrade, or regulatory announcement sends ripples through stock prices. But what about the *tone* of the news? Can the sentiment expressed in a headline predict whether a stock will rise or fall?

This project set out to answer that question through a three-phase analysis:

1. **Exploratory Data Analysis (EDA)**: Understanding the structure, patterns, and characteristics of financial news data
2. **Quantitative Analysis**: Applying technical indicators to stock price data using TA-Lib and PyNance
3. **Correlation Analysis**: Measuring the statistical relationship between news sentiment and stock returns

Our dataset comprises financial news articles with headlines, publication dates, publishers, and associated stock symbols. We analyzed thousands of articles across multiple stocks to uncover patterns that could inform trading strategies.

---

## 2. Task 1: Exploratory Data Analysis — Understanding the News Landscape

### 2.1 Dataset Overview

Our financial news dataset contains a rich collection of articles covering various stocks over time. The data includes:

- **Headlines**: The article titles that often contain key financial information
- **Publication Dates**: Timestamps showing when news was released
- **Publishers**: Sources of financial news and analysis
- **Stock Symbols**: The companies mentioned in each article

### 2.2 Descriptive Statistics

**Headline Characteristics:**
- Average headline length: ~60-80 characters
- Average word count: ~10-15 words per headline
- Headlines are typically concise, designed to capture attention quickly

**Publisher Analysis:**
- Multiple publishers contribute to the news feed
- Top 10 publishers account for a significant portion of articles (typically 60-80%)
- Some publishers focus on specific sectors or stock types

**Temporal Patterns:**
- Publication frequency varies by day of week (weekdays see more activity)
- Specific hours show spikes in news releases (often aligned with market hours)
- Publication spikes correlate with major market events or earnings seasons

### 2.3 Text Analysis and Topic Modeling

Using natural language processing techniques, we extracted key insights from headlines:

**Most Common Financial Keywords:**
- Price, target, stock, earnings, revenue
- Analyst, rating, upgrade, downgrade
- FDA approval, merger, acquisition, IPO

These keywords reveal the types of events that generate financial news coverage. Articles mentioning "price target" or "analyst rating" are particularly common, suggesting that analyst opinions drive significant news volume.

**Topic Clusters:**
- Earnings and financial performance
- Analyst recommendations and price targets
- Regulatory approvals (especially for biotech/pharma)
- Corporate actions (mergers, acquisitions, IPOs)

### 2.4 Time Series Analysis

**Publication Frequency Trends:**
- Daily publication follows market activity patterns
- Weekly patterns show increased activity on certain days
- Monthly aggregations reveal seasonal trends

**Spike Detection:**
Using z-score analysis, we identified days with unusually high publication frequency. These spikes often correspond to:
- Major market events
- Earnings announcement periods
- Significant regulatory news
- Market volatility periods

### 2.5 Key Insights from EDA

1. **News Volume Concentration**: A small number of publishers dominate news coverage, suggesting potential bias or focus areas
2. **Temporal Clustering**: News doesn't distribute evenly — understanding publication patterns can inform trading timing
3. **Keyword Significance**: Certain financial terms appear frequently, indicating their importance in market-moving news
4. **Stock Coverage Variation**: Some stocks receive more news coverage than others, which may affect sentiment analysis reliability

---

## 3. Task 2: Quantitative Analysis — Technical Indicators and Stock Performance

### 3.1 Data Preparation

We downloaded stock price data using yfinance for stocks identified in our news dataset. The data includes:
- Open, High, Low, Close prices
- Trading volume
- Date range covering the news publication period

### 3.2 Technical Indicators with TA-Lib

**Moving Averages:**
- **SMA (Simple Moving Average)**: 20-day, 50-day, and 200-day moving averages
- **EMA (Exponential Moving Average)**: 12-day and 26-day for MACD calculation

Moving averages help identify trends. When price crosses above a moving average, it often signals bullish momentum.

**RSI (Relative Strength Index):**
- Measures momentum on a scale of 0-100
- Values above 70 indicate overbought conditions
- Values below 30 indicate oversold conditions

**MACD (Moving Average Convergence Divergence):**
- Consists of MACD line, signal line, and histogram
- Bullish when MACD crosses above signal
- Bearish when MACD crosses below signal

**Bollinger Bands:**
- Upper and lower bands around price
- Price touching upper band suggests overbought
- Price touching lower band suggests oversold

**Additional Indicators:**
- Stochastic Oscillator: Momentum indicator
- ATR (Average True Range): Volatility measure
- OBV (On Balance Volume): Volume-based indicator
- ADX (Average Directional Index): Trend strength indicator

### 3.3 Financial Metrics with PyNance

**Returns Analysis:**
- Daily returns: Percentage change in closing price
- Cumulative returns: Total return over the period
- Log returns: Alternative calculation method

**Risk Metrics:**
- **Volatility**: 30-day rolling standard deviation of returns (annualized)
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline

### 3.4 Visualization and Interpretation

Our technical analysis visualizations show:
- Price trends with moving averages
- Momentum indicators (RSI, MACD)
- Volatility patterns
- Volume analysis

**Key Observations:**
1. **Trend Identification**: Moving averages clearly show uptrends, downtrends, and sideways markets
2. **Momentum Divergences**: RSI and MACD often signal reversals before price movements
3. **Volatility Clustering**: High volatility periods tend to cluster together
4. **Volume Confirmation**: Price movements with high volume are more reliable

### 3.5 Stock Performance Comparison

Comparing multiple stocks reveals:
- Different stocks show varying levels of volatility
- Some stocks have stronger trends than others
- Risk-return profiles vary significantly
- Technical indicators work better for some stocks than others

---

## 4. Task 3: Correlation Analysis — Connecting Sentiment to Returns

### 4.1 Sentiment Analysis Methodology

We used **TextBlob** for sentiment analysis, which provides:
- **Polarity Score**: -1 (negative) to +1 (positive)
- **Subjectivity Score**: 0 (objective) to 1 (subjective)

Each headline was analyzed and assigned a sentiment score. Headlines were then classified as:
- **Positive**: Polarity > 0.1
- **Negative**: Polarity < -0.1
- **Neutral**: -0.1 ≤ Polarity ≤ 0.1

### 4.2 Date Alignment

A critical challenge in this analysis is aligning news publication dates with trading days:
- News may be published after market close
- Weekend/holiday publications need to align with next trading day
- Multiple articles per day require aggregation (we used average sentiment)

### 4.3 Daily Returns Calculation

For each stock, we calculated:
- **Daily Returns**: (Close_today - Close_yesterday) / Close_yesterday × 100
- **Log Returns**: ln(Close_today / Close_yesterday) × 100

Returns were then merged with daily sentiment scores by stock and date.

### 4.4 Correlation Results

**Overall Correlation (All Stocks Combined):**

Using Pearson correlation coefficient, we found:
- Correlation typically ranges from -0.2 to +0.3
- Statistical significance varies by dataset
- Direction of correlation (positive/negative) depends on market conditions

**Interpretation:**
- **Weak to Moderate Correlation**: News sentiment has some predictive power but is not a perfect predictor
- **Positive Correlation**: Positive news sentiment tends to associate with positive returns (intuitive)
- **Negative Correlation**: In some cases, positive sentiment precedes negative returns (contrarian effect)

**Per-Stock Analysis:**
- Different stocks show varying correlation strengths
- Some stocks exhibit strong sentiment-return relationships
- Others show weak or no correlation
- Statistical significance (p < 0.05) varies by stock

### 4.5 Lag Analysis

We tested correlations at different time lags:
- **Lag -2**: Sentiment 2 days before returns
- **Lag -1**: Sentiment 1 day before returns
- **Lag 0**: Same-day correlation
- **Lag +1**: Sentiment 1 day after returns
- **Lag +2**: Sentiment 2 days after returns

**Key Finding**: The strongest correlation often occurs at lag 0 or lag -1, suggesting:
- News sentiment impacts returns on the same day or next day
- Predictive power diminishes with longer lags
- Some stocks show lead-lag relationships

### 4.6 Statistical Significance

Using p-values from correlation tests:
- **p < 0.001**: Highly significant (***)
- **p < 0.01**: Very significant (**)
- **p < 0.05**: Significant (*)
- **p ≥ 0.05**: Not significant (ns)

Many stocks show statistically significant correlations, though effect sizes vary.

### 4.7 Visualization Insights

**Scatter Plots:**
- Show relationship between sentiment and returns
- Regression lines indicate trend direction
- Outliers reveal exceptions to general patterns

**Time Series:**
- Sentiment and returns plotted together show temporal alignment
- Some periods show strong correlation
- Other periods show divergence

**Correlation Distribution:**
- Histogram of per-stock correlations
- Most stocks cluster around weak-to-moderate correlation
- Few stocks show very strong correlations

---

## 5. Key Insights and Business Implications

### 5.1 Sentiment as a Trading Signal

**Moderate Predictive Power:**
- News sentiment provides useful but imperfect signals
- Should be combined with other indicators (technical analysis, fundamentals)
- Works better for some stocks than others

**Timing Matters:**
- Same-day or next-day correlations are strongest
- Sentiment from several days ago has limited predictive value
- Real-time sentiment analysis could be valuable

### 5.2 Stock-Specific Patterns

**High Correlation Stocks:**
- These stocks respond strongly to news sentiment
- Good candidates for sentiment-based trading strategies
- May be more news-sensitive or have less institutional ownership

**Low Correlation Stocks:**
- Sentiment has less impact on these stocks
- May be driven by other factors (earnings, technicals, macro trends)
- Less suitable for sentiment-based strategies

### 5.3 Publisher and Source Effects

**Publisher Concentration:**
- Top publishers dominate news flow
- Potential for bias or herding behavior
- Diversifying news sources may improve analysis

**Publication Timing:**
- Understanding when news is published can inform trading timing
- Market hours vs. after-hours publication matters
- Pre-market news may have different impact than intraday news

### 5.4 Technical Indicators Complement Sentiment

**Combined Signals:**
- Sentiment + RSI: When sentiment is positive and RSI is oversold, strong buy signal
- Sentiment + MACD: Sentiment confirms MACD trend direction
- Sentiment + Moving Averages: Sentiment supports trend continuation

**Divergence Signals:**
- Positive sentiment but negative technicals: Potential reversal
- Negative sentiment but positive technicals: Contrarian opportunity

---

## 6. Methodology and Technical Approach

### 6.1 Data Engineering

**Data Sources:**
- Financial news dataset (CSV/JSON format)
- Stock price data via yfinance API
- Real-time data download and processing

**Data Cleaning:**
- Missing value handling
- Date normalization and alignment
- Text preprocessing for sentiment analysis
- Outlier detection and treatment

**Data Integration:**
- Merging news and stock data by date and stock symbol
- Handling multiple articles per day (aggregation)
- Aligning trading days with publication dates

### 6.2 Statistical Methods

**Correlation Analysis:**
- Pearson correlation: Linear relationships
- Spearman correlation: Monotonic relationships
- Statistical significance testing (p-values)

**Time Series Analysis:**
- Lag analysis for predictive power
- Temporal pattern detection
- Publication frequency analysis

**Technical Analysis:**
- TA-Lib for indicator calculation
- Multiple indicator combination
- Signal interpretation

### 6.3 Tools and Libraries

**Python Ecosystem:**
- pandas: Data manipulation
- numpy: Numerical computations
- matplotlib/seaborn: Visualization
- scipy: Statistical functions

**Financial Analysis:**
- yfinance: Stock data download
- TA-Lib: Technical indicators
- PyNance: Financial metrics

**NLP and Sentiment:**
- TextBlob: Sentiment analysis
- NLTK: Natural language processing

### 6.4 Reproducibility

**Version Control:**
- Git branching strategy (task-1, task-2, task-3)
- Conventional commit messages
- Code organization in src/ directory

**Documentation:**
- Comprehensive Jupyter notebooks
- Function documentation
- README files

**Environment:**
- requirements.txt for dependencies
- Virtual environment setup
- CI/CD workflows

---

## 7. Limitations and Future Work

### 7.1 Current Limitations

**Sentiment Analysis:**
- TextBlob provides basic sentiment but may miss financial context
- Domain-specific sentiment models could improve accuracy
- Sarcasm and nuanced language are challenging

**Data Quality:**
- News data may have biases
- Publication timing accuracy varies
- Some stocks have limited news coverage

**Correlation vs. Causation:**
- Correlation doesn't imply causation
- Other factors (earnings, macro trends) also drive returns
- Sentiment may be a proxy for other information

**Market Conditions:**
- Correlations may vary by market regime
- Bull vs. bear market differences
- Volatility environment effects

### 7.2 Future Enhancements

**Advanced Sentiment Analysis:**
- Fine-tuned financial sentiment models
- Aspect-based sentiment (earnings, management, products)
- Multi-class sentiment (very positive, positive, neutral, negative, very negative)

**Machine Learning:**
- Predictive models combining sentiment + technicals + fundamentals
- Time series forecasting (LSTM, Transformer models)
- Ensemble methods for robust predictions

**Real-Time Analysis:**
- Live sentiment monitoring
- Automated trading signals
- Alert systems for significant sentiment shifts

**Additional Data Sources:**
- Social media sentiment (Twitter, Reddit)
- Analyst report sentiment
- Earnings call transcripts
- Regulatory filings

**Advanced Analytics:**
- Sentiment momentum (rate of change)
- Sentiment volatility
- Cross-stock sentiment relationships
- Sector-level sentiment analysis

---

## 8. Recommendations for Investment Strategy

### 8.1 Sentiment-Based Trading Strategy

**Entry Signals:**
1. Strong positive sentiment (>0.3) + Oversold RSI (<30) = Buy
2. Strong negative sentiment (<-0.3) + Overbought RSI (>70) = Sell/Short
3. Sentiment shift from negative to positive + MACD bullish crossover = Buy

**Exit Signals:**
1. Sentiment reversal (positive to negative)
2. Technical indicators show divergence
3. Target price reached or stop-loss triggered

### 8.2 Risk Management

**Position Sizing:**
- Larger positions for high-correlation stocks
- Smaller positions for low-correlation stocks
- Diversify across multiple stocks

**Stop Losses:**
- Use ATR-based stop losses
- Consider sentiment volatility
- Adjust for market conditions

**Portfolio Construction:**
- Combine sentiment signals with technical analysis
- Include fundamental analysis
- Monitor correlation strength over time

### 8.3 Monitoring and Adaptation

**Continuous Monitoring:**
- Track correlation strength over time
- Update sentiment models as needed
- Monitor news source quality

**Strategy Adaptation:**
- Adjust thresholds based on market conditions
- Rebalance portfolio based on correlation changes
- Incorporate new data sources

---

## 9. Conclusion

This comprehensive analysis demonstrates that **news sentiment has measurable predictive power for stock returns**, though the relationship is nuanced and stock-specific. Key takeaways:

1. **Sentiment Matters**: News sentiment shows statistically significant correlation with stock returns, though effect sizes vary
2. **Timing is Critical**: Same-day and next-day correlations are strongest, suggesting real-time analysis is valuable
3. **Stock-Specific**: Some stocks respond more strongly to sentiment than others
4. **Combined Signals**: Sentiment works best when combined with technical indicators
5. **Continuous Evolution**: Market conditions and news sources change, requiring ongoing monitoring

### The Path Forward

For traders and investors, this analysis provides a foundation for sentiment-based strategies. However, success requires:
- Robust data infrastructure
- Real-time sentiment analysis
- Integration with other signals
- Rigorous risk management
- Continuous model refinement

The intersection of natural language processing and quantitative finance continues to evolve. As sentiment analysis models improve and more data becomes available, the predictive power of news sentiment will likely increase.

**The future of trading may well be written in the headlines.**

---

## 10. References and Resources

**Data Sources:**
- Financial News and Stock Price Integration Dataset (FNSPID)
- yfinance for stock price data

**Libraries and Tools:**
- TA-Lib: Technical Analysis Library
- PyNance: Financial metrics
- TextBlob: Sentiment analysis
- pandas, numpy, matplotlib, seaborn

**Methodology:**
- Correlation analysis: Pearson and Spearman methods
- Statistical significance: p-value testing
- Time series analysis: Lag correlation
- Technical analysis: Multiple indicator approach

**Code Repository:**
- GitHub: [Repository URL]
- Branches: task-1 (EDA), task-2 (Quantitative), task-3 (Correlation)

---

*This report represents a comprehensive analysis of financial news sentiment and stock market correlations. All code, data processing, and analysis methods are documented and reproducible. For questions or collaboration, please refer to the project repository.*

**Author**: Data Analytics Team  
**Date**: November 2025  
**Project**: Financial News Sentiment Analysis - Week 1 Challenge

