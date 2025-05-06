# scripts/sentiment_analysis.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import requests
import json
from textblob import TextBlob
import re

# Cache directory for sentiment data
CACHE_DIR = "cache/sentiment"
os.makedirs(CACHE_DIR, exist_ok=True)

def clean_text(text):
    """Clean and normalize text for sentiment analysis"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    if not text or len(text) < 10:
        return 0.0
    
    cleaned_text = clean_text(text)
    
    if not cleaned_text or len(cleaned_text) < 10:
        return 0.0
        
    analysis = TextBlob(cleaned_text)
    
    # Return polarity score between -1 and 1
    return analysis.sentiment.polarity

def generate_simulated_news(ticker, date):
    """Generate simulated news for a given ticker and date"""
    # Use the date and ticker as a seed for reproducible randomness
    seed = hash(f"{date}-{ticker}") % (2**32)
    rng = np.random.default_rng(seed)
    
    # List of potential headlines (positive, neutral, negative)
    positive_headlines = [
        f"{ticker} Exceeds Quarterly Earnings Expectations",
        f"{ticker} Announces New Product Line",
        f"{ticker} Secures Major Partnership Deal",
        f"{ticker} Stock Upgraded by Analysts",
        f"{ticker} Expands into New Markets"
    ]
    
    neutral_headlines = [
        f"{ticker} Reports Mixed Quarterly Results",
        f"{ticker} Holds Annual Shareholder Meeting",
        f"{ticker} Announces Management Changes",
        f"{ticker} to Present at Industry Conference",
        f"{ticker} Files Annual Report"
    ]
    
    negative_headlines = [
        f"{ticker} Misses Earnings Targets",
        f"{ticker} Faces Regulatory Investigation",
        f"{ticker} Announces Layoffs",
        f"{ticker} Stock Downgraded by Analysts",
        f"{ticker} Delays Product Launch"
    ]
    
    # Decide sentiment tendency based on hash of date and ticker
    sentiment_bias = rng.normal(0, 0.5)
    
    # Generate 0-3 news items for the day
    num_news = rng.integers(0, 4)
    
    news_items = []
    for _ in range(num_news):
        if sentiment_bias > 0.2:
            # More likely positive news
            headline = rng.choice(positive_headlines)
            sentiment = rng.uniform(0.3, 1.0)
        elif sentiment_bias < -0.2:
            # More likely negative news
            headline = rng.choice(negative_headlines)
            sentiment = rng.uniform(-1.0, -0.3)
        else:
            # Neutral news
            headline = rng.choice(neutral_headlines)
            sentiment = rng.uniform(-0.3, 0.3)
        
        news_items.append({
            "headline": headline,
            "sentiment": sentiment
        })
    
    return news_items

def get_sentiment_from_news(news_items):
    """Calculate aggregate sentiment from news items"""
    if not news_items:
        return 0.0
    
    total_sentiment = sum(item["sentiment"] for item in news_items)
    avg_sentiment = total_sentiment / len(news_items)
    
    # Add some noise
    noise = np.random.normal(0, 0.1)
    final_sentiment = avg_sentiment + noise
    
    # Ensure it's between -1 and 1
    return max(min(final_sentiment, 1.0), -1.0)

def get_sentiment(date, ticker="AAPL", use_cache=True):
    """
    Get sentiment score for a given date and ticker
    Returns a value between -1 (negative) and 1 (positive)
    """
    # Format date if it's a datetime object
    if isinstance(date, datetime):
        date_str = date.strftime("%Y-%m-%d")
    else:
        date_str = date
    
    cache_file = os.path.join(CACHE_DIR, f"{ticker}_{date_str}.json")
    
    # Check cache first
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            sentiment_data = json.load(f)
        return sentiment_data["sentiment"]
    
    # Generate simulated news and sentiment
    news_items = generate_simulated_news(ticker, date_str)
    sentiment_score = get_sentiment_from_news(news_items)
    
    # Cache the result
    sentiment_data = {
        "ticker": ticker,
        "date": date_str,
        "sentiment": sentiment_score,
        "news": news_items
    }
    
    with open(cache_file, 'w') as f:
        json.dump(sentiment_data, f)
    
    return sentiment_score

def get_sentiment_for_period(ticker, start_date, end_date):
    """
    Get sentiment data for a period of time
    Returns a pandas Series with dates as index and sentiment as values
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    date_range = pd.date_range(start=start_date, end=end_date)
    sentiment_data = {}
    
    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        sentiment_data[date_str] = get_sentiment(date_str, ticker)
    
    return pd.Series(sentiment_data, name="Sentiment")

# Add this function to the bottom of your existing sentiment_analysis.py file

def get_dummy_sentiment(date, ticker="AAPL"):
    """
    Returns a random sentiment score between -1 and 1.
    This bypasses the get_sentiment function to avoid JSON loading errors.
    """
    # Use the date and ticker as a seed for reproducible randomness
    if isinstance(date, datetime):
        date_str = date.strftime("%Y-%m-%d")
    else:
        date_str = date
        
    seed = hash(f"{date_str}-{ticker}") % (2**32)
    np.random.seed(seed)
    
    # Generate a random sentiment with slight positive bias
    sentiment = np.random.normal(0.05, 0.3)
    
    # Ensure it's between -1 and 1
    return max(min(sentiment, 1.0), -1.0)

if __name__ == "__main__":
    # Test the sentiment function
    today = datetime.now().strftime("%Y-%m-%d")
    sentiment = get_sentiment(today, "AAPL")
    print(f"AAPL sentiment for {today}: {sentiment}")
    
    # Test period sentiment
    start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    end = today
    sentiment_series = get_sentiment_for_period("AAPL", start, end)
    print("\nSentiment over the last 10 days:")
    print(sentiment_series)