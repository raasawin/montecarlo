import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@st.cache_data(ttl=3600)
def get_stock_data(ticker, period="1y"):
    try:
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            st.error("No data found for this ticker.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_indicators(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD Calculation
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def plot_stock_with_indicators(df, ticker):
    st.subheader(f"{ticker} Price with Indicators")
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Price + MA20
    axs[0].plot(df.index, df['Close'], label='Close Price')
    axs[0].plot(df.index, df['MA20'], label='MA 20')
    axs[0].set_ylabel("Price ($)")
    axs[0].legend()
    axs[0].set_title(f"{ticker} Close Price & Moving Average")
    
    # RSI
    axs[1].plot(df.index, df['RSI'], label='RSI', color='purple')
    axs[1].axhline(70, color='red', linestyle='--')
    axs[1].axhline(30, color='green', linestyle='--')
    axs[1].set_ylabel("RSI")
    axs[1].legend()
    axs[1].set_title("Relative Strength Index")
    
    # MACD + Signal Line
    axs[2].plot(df.index, df['MACD'], label='MACD', color='blue')
    axs[2].plot(df.index, df['Signal_Line'], label='Signal Line', color='orange')
    axs[2].set_ylabel("MACD")
    axs[2].legend()
    axs[2].set_title("MACD Indicator")
    
    st.pyplot(fig)
    
    # Volume + Volume MA20
    st.subheader("Volume and Volume Moving Average")
    fig_vol, ax_vol = plt.subplots(figsize=(12,3))
    ax_vol.bar(df.index, df['Volume'], label='Volume')
    ax_vol.plot(df.index, df['Volume_MA20'], label='Volume MA 20', color='red')
    ax_vol.set_ylabel("Volume")
    ax_vol.legend()
    st.pyplot(fig_vol)

def main():
    st.title("Advanced Stock Analysis with Momentum & Volume Indicators")
    ticker = st.text_input("Enter stock ticker (e.g., AAPL)", value="AAPL")
    period = st.selectbox("Select data period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

    if ticker:
        df = get_stock_data(ticker, period)
        if df is not None:
            df = calculate_indicators(df)
            plot_stock_with_indicators(df, ticker)
        else:
            st.warning("Could not load stock data.")
        
if __name__ == "__main__":
    main()
