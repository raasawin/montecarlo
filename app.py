import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Stock Price Prediction with Technical Indicators")

ticker = st.text_input("Enter Stock Ticker (e.g. AAPL):", "AAPL")
period = "6mo"

@st.cache_data
def get_stock_data(ticker, period="6mo"):
    data = yf.Ticker(ticker).history(period=period)
    data['MA20'] = data['Close'].rolling(window=20).mean()
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Volume_MA20'] = data['Volume'].rolling(window=20).mean()
    return data.dropna()

def plot_indicators(df):
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # Price + MA20
    axs[0].plot(df.index, df['Close'], label='Close Price')
    axs[0].plot(df.index, df['MA20'], label='MA20')
    axs[0].set_title('Price and 20-day Moving Average')
    axs[0].legend()
    
    # RSI
    axs[1].plot(df.index, df['RSI'], label='RSI', color='orange')
    axs[1].axhline(70, color='red', linestyle='--')
    axs[1].axhline(30, color='green', linestyle='--')
    axs[1].set_title('Relative Strength Index (RSI)')
    
    # MACD and Signal Line
    axs[2].plot(df.index, df['MACD'], label='MACD', color='purple')
    axs[2].plot(df.index, df['Signal_Line'], label='Signal Line', color='blue')
    axs[2].set_title('MACD and Signal Line')
    axs[2].legend()
    
    # Volume + Volume MA20
    axs[3].bar(df.index, df['Volume'], label='Volume', color='grey')
    axs[3].plot(df.index, df['Volume_MA20'], label='Volume MA20', color='red')
    axs[3].set_title('Volume and 20-day Volume MA')
    axs[3].legend()
    
    plt.tight_layout()
    st.pyplot(fig)

def generate_signal_and_estimate(df):
    last = df.iloc[-1]
    score = 0

    # Price vs MA20
    if last['Close'] > last['MA20']:
        score += 1
    else:
        score -= 1

    # RSI
    if last['RSI'] < 30:
        score += 1
    elif last['RSI'] > 70:
        score -= 1

    # MACD crossover
    if df['MACD'].iloc[-2] < df['Signal_Line'].iloc[-2] and last['MACD'] > last['Signal_Line']:
        score += 1
    elif df['MACD'].iloc[-2] > df['Signal_Line'].iloc[-2] and last['MACD'] < last['Signal_Line']:
        score -= 1

    # Volume confirmation
    if last['Volume'] > last['Volume_MA20']:
        score += 0.5
    else:
        score -= 0.5

    # Interpretation
    if score >= 2:
        direction = "likely to rise"
    elif score <= -2:
        direction = "likely to fall"
    else:
        direction = "uncertain"

    # Estimate expected % change over next month (~20 trading days)
    returns = df['Close'].pct_change().tail(20)
    avg_return = returns.mean()
    expected_move = avg_return * 20 * 100  # in percentage

    return direction, expected_move

if ticker:
    try:
        df = get_stock_data(ticker, period)
        plot_indicators(df)
        direction, expected_pct = generate_signal_and_estimate(df)
        st.markdown(f"### Prediction: The stock price is **{direction}** over the next month.")
        st.markdown(f"Estimated expected price change: **{expected_pct:.2f}%**")
    except Exception as e:
        st.error(f"Error fetching data or calculating indicators: {e}")

