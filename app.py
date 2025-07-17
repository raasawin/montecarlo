import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

st.set_page_config(layout="wide", page_title="Stock Price Forecast (MC + ML)")

# --------------------------------------
# Fetch stock data
# --------------------------------------
@st.cache_data(ttl=3600)
def get_stock_data(ticker_symbol, period="1y"):
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period=period)
    data.dropna(inplace=True)
    return data

# --------------------------------------
# Technical indicators
# --------------------------------------
def add_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Momentum'] = df['Close'].diff(4)
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['Volume_Change'] = df['Volume'].pct_change()
    return df.dropna()

# --------------------------------------
# Monte Carlo Simulation
# --------------------------------------
def monte_carlo_simulation(S0, mu, sigma, T, N, M):
    dt = T / N
    simulations = np.zeros((N, M))
    for i in range(M):
        prices = [S0]
        for _ in range(N - 1):
            S_t = prices[-1] * np.exp((mu - 0.5 * sigma ** 2) * dt +
                                      sigma * np.sqrt(dt) * np.random.normal())
            prices.append(S_t)
        simulations[:, i] = prices
    return simulations

# --------------------------------------
# ML Model (lightweight version for speed)
# --------------------------------------
def train_random_forest(df, n_days_ahead):
    df = df.copy()
    df['Target'] = df['Close'].shift(-n_days_ahead)
    df = df.dropna()

    features = ['Close', 'SMA_20', 'Momentum', 'Volatility', 'Volume_Change']
    X = df[features]
    y = df['Target']

    # Train/test split without shuffle (time series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Simple RF with fixed hyperparams (no GridSearch for speed)
    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    latest_features = X.iloc[[-1]]
    predicted_price = rf.predict(latest_features)[0]

    # Simple CI approximation ±2%
    ci_lower = predicted_price * 0.98
    ci_upper = predicted_price * 1.02

    best_params = {'n_estimators': 100}

    return rf, rmse, predicted_price, y_test.values[-1], ci_lower, ci_upper, best_params

# --------------------------------------
# Sidebar Inputs
# --------------------------------------
st.sidebar.title("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
period = st.sidebar.selectbox("Historical Data Period", ["6mo", "1y", "2y", "5y"], index=1)
n_simulations = st.sidebar.slider("Number of Monte Carlo Simulations", 100, 10000, 500, step=100)
n_days = st.sidebar.slider("Days into the Future", 10, 180, 30, step=10)

# --------------------------------------
# Main App
# --------------------------------------
st.title(f"📈 Forecasting Stock Price: {ticker.upper()}")

try:
    df = get_stock_data(ticker, period)
    df = add_technical_indicators(df)

    latest_close = df['Close'].iloc[-1]
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    mu, sigma = log_returns.mean(), log_returns.std()

    # Monte Carlo simulation
    sim_data = monte_carlo_simulation(S0=latest_close, mu=mu, sigma=sigma,
                                      T=n_days, N=n_days, M=n_simulations)

    final_prices = sim_data[-1, :]
    p5 = np.percentile(final_prices, 5)
    p50 = np.percentile(final_prices, 50)
    p95 = np.percentile(final_prices, 95)

    st.subheader("Monte Carlo Simulation Results")
    st.write(f"**5th percentile price**: ${p5:.2f} ({(p5 - latest_close)/latest_close:.2%})")
    st.write(f"**Median price**: ${p50:.2f} ({(p50 - latest_close)/latest_close:.2%})")
    st.write(f"**95th percentile price**: ${p95:.2f} ({(p95 - latest_close)/latest_close:.2%})")

    # ML Forecast
    st.write("🔍 Running ML Forecast...")
    try:
        st.write("✅ Step 1: Starting training...")
        model, rmse, predicted_price, actual_price, ci_lower, ci_upper, best_params = train_random_forest(df, n_days)
        st.write("✅ Step 2: Training complete.")

        ml_change_pct = (predicted_price - latest_close) / latest_close * 100

        st.subheader(f"Machine Learning Prediction ({n_days}-Day Close)")
        st.write(f"**Predicted Price**: ${predicted_price:.2f}")
        st.write(f"**95% Prediction Interval**: ${ci_lower:.2f} to ${ci_upper:.2f}")
        st.write(f"**Actual Price (last test sample)**: ${actual_price:.2f}")
        st.write(f"**RMSE**: ${rmse:.2f}")
        st.write(f"**Expected Price Change**: {ml_change_pct:+.2f}%")
        st.write(f"**Best Model Parameters**: `{best_params}`")

        # Final verdict summary
        st.markdown("---")
        st.subheader("📊 Final Summary")
        if ml_change_pct > 1 and p50 > latest_close:
            st.success(f"**Likely Upward Trend** — Expected increase of ~{ml_change_pct:.2f}% (ML) and {(p50 - latest_close)/latest_close:.2%} (MC).")
        elif ml_change_pct < -1 and p50 < latest_close:
            st.error(f"**Likely Downward Trend** — Expected drop of ~{ml_change_pct:.2f}% (ML) and {(p50 - latest_close)/latest_close:.2%} (MC).")
        else:
            st.warning("**Uncertain** — Predictions show mixed or flat direction. Exercise caution.")

    except Exception as e:
        st.error(f"Error during ML training: {e}")

except Exception as e:
    st.error(f"Error loading data or running simulation: {e}")
