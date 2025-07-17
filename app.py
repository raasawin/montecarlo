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
# ML Model with tuning and progress bar
# --------------------------------------
def train_random_forest(df, n_days_ahead, use_gridsearch=False, use_bootstrap=False, bootstrap_iters=1000):
    df = df.copy()
    df['Target'] = df['Close'].shift(-n_days_ahead)
    df = df.dropna()

    features = ['Close', 'SMA_20', 'Momentum', 'Volatility', 'Volume_Change']
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    if use_gridsearch:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10]
        }
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(RandomForestRegressor(random_state=0),
                                   param_grid, cv=tscv,
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        st.info("🔍 Running GridSearchCV (this may take a while)...")
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        best_model = RandomForestRegressor(n_estimators=100, random_state=0)
        best_model.fit(X_train, y_train)
        best_params = {'n_estimators': 100}

    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    latest_features = X.iloc[[-1]]
    predicted_price = best_model.predict(latest_features)[0]

    # Bootstrapping with progress bar
    if use_bootstrap:
        st.info(f"🔁 Running Bootstrapping ({bootstrap_iters} iterations)...")
        boot_preds = []

        progress_text = st.empty()
        progress_bar = st.progress(0)

        for i in range(bootstrap_iters):
            X_resampled, y_resampled = resample(X_train, y_train)
            rf = RandomForestRegressor(**best_model.get_params())
            rf.fit(X_resampled, y_resampled)
            boot_preds.append(rf.predict(latest_features)[0])

            if i % 10 == 0 or i == bootstrap_iters - 1:
                percent_complete = (i + 1) / bootstrap_iters
                progress_bar.progress(percent_complete)
                progress_text.markdown(f"**Bootstrapping Progress:** {i+1}/{bootstrap_iters}")

        progress_bar.empty()
        progress_text.empty()

        ci_lower = np.percentile(boot_preds, 2.5)
        ci_upper = np.percentile(boot_preds, 97.5)
    else:
        ci_lower = predicted_price * 0.98
        ci_upper = predicted_price * 1.02

    return best_model, rmse, predicted_price, y_test.values[-1], ci_lower, ci_upper, best_params

# --------------------------------------
# Sidebar Inputs
# --------------------------------------
st.sidebar.title("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
period = st.sidebar.selectbox("Historical Data Period", ["6mo", "1y", "2y", "5y"], index=1)
n_simulations = st.sidebar.slider("Monte Carlo Simulations", 100, 10000, 500, step=100)
n_days = st.sidebar.slider("Days into the Future", 10, 180, 30, step=10)

use_gridsearch = st.sidebar.checkbox("Use GridSearchCV (slower, better tuning)", value=False)
use_bootstrap = st.sidebar.checkbox("Use Bootstrapping for Confidence Interval (very slow)", value=False)
bootstrap_iters = st.sidebar.slider("Bootstrapping Iterations", 100, 2000, 1000, step=100) if use_bootstrap else 0

if use_gridsearch and use_bootstrap:
    st.sidebar.warning("⚠️ GridSearch + Bootstrapping enabled — expect long wait times.")
elif use_gridsearch:
    st.sidebar.info("📊 GridSearchCV will tune model hyperparameters.")
elif use_bootstrap:
    st.sidebar.info("📈 Bootstrapping will generate 95% CI from resampled models.")
else:
    st.sidebar.info("⚡ Fast mode (no GridSearch or Bootstrapping).")

# --------------------------------------
# Main App Logic
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

    # Machine Learning Forecast
    st.write("🔍 Running ML Forecast...")
    try:
        model, rmse, predicted_price, actual_price, ci_lower, ci_upper, best_params = train_random_forest(
            df, n_days,
            use_gridsearch=use_gridsearch,
            use_bootstrap=use_bootstrap,
            bootstrap_iters=bootstrap_iters
        )

        ml_change_pct = (predicted_price - latest_close) / latest_close * 100

        st.subheader(f"Machine Learning Prediction ({n_days}-Day Close)")
        st.write(f"**Predicted Price**: ${predicted_price:.2f}")
        st.write(f"**95% Prediction Interval**: ${ci_lower:.2f} to ${ci_upper:.2f}")
        st.write(f"**Actual Price (last test sample)**: ${actual_price:.2f}")
        st.write(f"**RMSE**: ${rmse:.2f}")
        st.write(f"**Expected Price Change**: {ml_change_pct:+.2f}%")
        st.write(f"**Best Model Parameters**: `{best_params}`")

        st.markdown("---")
        st.subheader("📊 Final Summary")
        if ml_change_pct > 1 and p50 > latest_close:
            st.success(f"**Likely Upward Trend** — ML: ~{ml_change_pct:.2f}%, MC: {(p50 - latest_close)/latest_close:.2%}")
        elif ml_change_pct < -1 and p50 < latest_close:
            st.error(f"**Likely Downward Trend** — ML: ~{ml_change_pct:.2f}%, MC: {(p50 - latest_close)/latest_close:.2%}")
        else:
            st.warning("**Uncertain** — Mixed or flat predictions. Use caution.")

    except Exception as e:
        st.error(f"Error during ML training: {e}")

except Exception as e:
    st.error(f"Error loading data or running simulation: {e}")
