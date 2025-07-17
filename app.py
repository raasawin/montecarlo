import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

st.set_page_config(layout="wide", page_title="Stock Price Forecast (MC + ML)")

# --------------------------------------
# Fetch stock data (Fixed cache handling)
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
# ML Model with tuning and CI
# --------------------------------------
def train_random_forest(df, n_days_ahead, bootstrap_iters=1000):
    df = df.copy()
    df['Target'] = df['Close'].shift(-n_days_ahead)
    df = df.dropna()

    features = ['Close', 'SMA_20', 'Momentum', 'Volatility', 'Volume_Change']
    
    if len(df) < 50:
        raise ValueError(f"Too little data ({len(df)} rows) after shifting for prediction. Try a longer historical period or fewer forecast days.")

    X = df[features]
    y = df['Target']

    # TimeSeries split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    
    if len(X_test) == 0:
        raise ValueError("Test set is empty. Adjust your time range or forecast horizon.")

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    model = GridSearchCV(RandomForestRegressor(random_state=0), param_grid, cv=tscv, scoring='neg_mean_squared_error')
    model.fit(X_train, y_train)

    best_model = model.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    latest_features = X.iloc[[-1]]
    predicted_price = best_model.predict(latest_features)[0]

    boot_preds = []
    for _ in range(bootstrap_iters):
        X_resampled, y_resampled = resample(X_train, y_train)
        rf = RandomForestRegressor(**best_model.get_params())
        rf.fit(X_resampled, y_resampled)
        boot_preds.append(rf.predict(latest_features)[0])

    ci_lower = np.percentile(boot_preds, 2.5)
    ci_upper = np.percentile(boot_preds, 97.5)

    return best_model, rmse, predicted_price, y_test.values[-1], ci_lower, ci_upper, model.best_params_



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

    # Run Monte Carlo
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

    # Run ML model
    model, rmse, predicted_price, actual_price, ci_lower, ci_upper, best_params = train_random_forest(df, n_days)
    ml_change_pct = (predicted_price - latest_close) / latest_close * 100

    st.subheader(f"Machine Learning Prediction ({n_days}-Day Close)")
    st.write(f"**Predicted Price**: ${predicted_price:.2f}")
    st.write(f"**95% Prediction Interval**: ${ci_lower:.2f} to ${ci_upper:.2f}")
    st.write(f"**Actual Price (last test sample)**: ${actual_price:.2f}")
    st.write(f"**RMSE**: ${rmse:.2f}")
    st.write(f"**Expected Price Change**: {ml_change_pct:+.2f}%")
    st.write(f"**Best Model Parameters**: `{best_params}`")

    # Summary
    st.markdown("---")
    st.subheader("📊 Final Summary")
    if ml_change_pct > 1 and p50 > latest_close:
        st.success(f"**Likely Upward Trend** — Expected increase of ~{ml_change_pct:.2f}% (ML) and {(p50 - latest_close)/latest_close:.2%} (MC).")
    elif ml_change_pct < -1 and p50 < latest_close:
        st.error(f"**Likely Downward Trend** — Expected drop of ~{ml_change_pct:.2f}% (ML) and {(p50 - latest_close)/latest_close:.2%} (MC).")
    else:
        st.warning("**Uncertain** — Predictions show mixed or flat direction. Exercise caution.")

except Exception as e:
    st.error(f"Error loading data: {e}")

