import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import datetime

# --- Functions ---

def get_stock_data(ticker, period="1y"):
    data = yf.Ticker(ticker).history(period=period)
    data = data.dropna()
    return data

def calculate_indicators(df):
    # Simple Moving Averages
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
    
    # RSI
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    
    # Volume spike: volume today vs average volume of last 20 days
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Spike'] = df['Volume'] > 1.5 * df['Volume_MA20']
    
    return df

def identify_market_cycle(df_index):
    # Bull if price > SMA 200, Bear otherwise
    sma_200 = SMAIndicator(df_index['Close'], window=200).sma_indicator()
    latest_close = df_index['Close'].iloc[-1]
    latest_sma200 = sma_200.iloc[-1]
    if latest_close > latest_sma200:
        return "Bull Market"
    else:
        return "Bear Market"

def fetch_geopolitical_headlines():
    # Using a simple news site RSS or Google news with keyword "geopolitics"
    url = 'https://news.google.com/search?q=geopolitics&hl=en-US&gl=US&ceid=US:en'
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')
    
    headlines = []
    for h in soup.select('a.DY5T1d'):
        headlines.append(h.text)
    return headlines[:10]

def analyze_sentiment(headlines):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    return avg_sentiment

def monte_carlo_simulation(S0, mu, sigma, days=30, sims=1000):
    dt = 1/252
    price_paths = np.zeros((days, sims))
    price_paths[0] = S0
    
    for t in range(1, days):
        rand = np.random.standard_normal(sims)
        price_paths[t] = price_paths[t-1] * np.exp((mu - 0.5 * sigma**2)*dt + sigma * np.sqrt(dt) * rand)
    return price_paths

# --- Streamlit UI ---

st.title("Advanced Stock Price Simulator with Market & Geopolitical Analysis")

ticker = st.text_input("Enter Stock Ticker (e.g. AAPL)", "AAPL")

if ticker:
    with st.spinner("Fetching data..."):
        # Fetch stock and market index data
        stock_df = get_stock_data(ticker)
        sp500_df = get_stock_data("^GSPC", period="2y")  # S&P500 as market proxy
        
        # Indicators
        stock_df = calculate_indicators(stock_df)
        
        # Market cycle
        market_cycle = identify_market_cycle(sp500_df)
        
        # Geopolitical headlines & sentiment
        headlines = fetch_geopolitical_headlines()
        geo_sentiment = analyze_sentiment(headlines)
        
        # Display key info
        st.subheader(f"Market Cycle: {market_cycle}")
        st.subheader(f"Geopolitical Sentiment Score: {geo_sentiment:.2f} (-1 negative, +1 positive)")
        st.write("Recent Geopolitical Headlines:")
        for h in headlines:
            st.write("- " + h)
        
        # Show latest price and indicators
        st.subheader(f"Latest price for {ticker}: ${stock_df['Close'][-1]:.2f}")
        st.line_chart(stock_df[['Close', 'SMA_50', 'SMA_200']])
        st.line_chart(stock_df['RSI'])
        st.bar_chart(stock_df['Volume'])
        
        # Calculate returns stats for Monte Carlo
        log_returns = np.log(stock_df['Close'] / stock_df['Close'].shift(1)).dropna()
        mu = log_returns.mean() * 252
        sigma = log_returns.std() * np.sqrt(252)
        
        S0 = stock_df['Close'][-1]
        
        # Adjust mu based on market cycle and geopolitical sentiment
        if market_cycle == "Bear Market":
            mu *= 0.9
        if geo_sentiment < -0.1:  # negative sentiment dampens expected returns
            mu *= 0.85
        
        # Monte Carlo simulation
        sims = monte_carlo_simulation(S0, mu, sigma)
        
        # Percentiles
        last_prices = sims[-1]
        p5 = np.percentile(last_prices, 5)
        p50 = np.percentile(last_prices, 50)
        p95 = np.percentile(last_prices, 95)
        
        st.subheader("Monte Carlo Price Prediction (Next 30 trading days)")
        st.write(f"5th percentile: ${p5:.2f} ({(p5 - S0)/S0 * 100:.2f}%)")
        st.write(f"Median: ${p50:.2f} ({(p50 - S0)/S0 * 100:.2f}%)")
        st.write(f"95th percentile: ${p95:.2f} ({(p95 - S0)/S0 * 100:.2f}%)")
