import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

# Set layout
st.set_page_config(layout="wide", page_title="Stock Price Forecast (MC + ML)")

# ----------------------------------------
# Fetch S&P 500 tickers via CSV (not HTML)
@st.cache_data(ttl=86400)
def get_sp500_tickers():
    url = "https://datahub.io/core/s-and-p-500-companies-financials/r/constituents.csv"
    df = pd.read_csv(url)
    return df['Symbol'].dropna().unique().tolist()

# ----------------------------------------
# Fetch stock data
@st.cache_data(ttl=3600)
def get_stock_data(ticker_symbol, period="1y"):
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period=period)
    data.dropna(inplace=True)
    return data

# ----------------------------------------
# Technical indicators
def add_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Momentum'] = df['Close'].diff(4)
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['Volume_Change'] = df['Volume'].pct_change()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['BB_Upper'] = df['SMA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['SMA_20'] - 2 * df['Close'].rolling(window=20).std()
    df['Close_vs_SMA20'] = df['Close'] - df['SMA_20']
    df['Close_vs_SMA50'] = df['Close'] - df['SMA_50']
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_5d'] = df['Close'].pct_change(5)
    return df

# ----------------------------------------
# Monte Carlo simulation
def monte_carlo_simulation(S0, mu, sigma, T, N, M):
    dt = T / N
    simulations = np.zeros((N, M))
    for i in range(M):
        prices = [S0]
        for _ in range(N - 1):
            S_t = prices[-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.normal())
            prices.append(S_t)
        simulations[:, i] = prices
    return simulations

# ----------------------------------------
# ML model training
def train_random_forest(df, n_days_ahead, eps, bootstrap_iters=100, use_gridsearch=False, use_bootstrap=False, progress_bar=None):
    df['EPS'] = eps
    df['Target'] = df['Close'].shift(-n_days_ahead)
    df = df.dropna()

    features = [
        'Close', 'SMA_20', 'SMA_50', 'Momentum', 'Volatility', 'Volume_Change',
        'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower',
        'Close_vs_SMA20', 'Close_vs_SMA50', 'Return_1d', 'Return_5d'
    ]

    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    if len(X_train) < 20:
        raise ValueError("Not enough valid data to train the model.")

    if use_gridsearch:
        param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]}
        tscv = TimeSeriesSplit(n_splits=5)
        model = GridSearchCV(RandomForestRegressor(), param_grid, cv=tscv)
        model.fit(X_train, y_train)
        best_model = model.best_estimator_
    else:
        best_model = RandomForestRegressor(n_estimators=100)
        best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    latest_features = X.iloc[[-1]]
    predicted_price = best_model.predict(latest_features)[0]

    ci_lower, ci_upper = None, None
    if use_bootstrap and progress_bar:
        boot_preds = []
        for i in range(bootstrap_iters):
            X_resampled, y_resampled = resample(X_train, y_train)
            rf = RandomForestRegressor()
            rf.fit(X_resampled, y_resampled)
            boot_preds.append(rf.predict(latest_features)[0])
            if i % max(1, bootstrap_iters // 100) == 0:
                progress_bar.progress(min(i / bootstrap_iters, 1.0))
        ci_lower = np.percentile(boot_preds, 2.5)
        ci_upper = np.percentile(boot_preds, 97.5)

    return best_model, rmse, predicted_price, y_test.values[-1], ci_lower, ci_upper

# ----------------------------------------
# Sidebar
st.sidebar.title("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
period = st.sidebar.selectbox("Historical Data Period", ["6mo", "1y", "2y", "5y"], index=1)
n_simulations = st.sidebar.slider("Monte Carlo Simulations", 100, 10000, 500, step=100)
n_days = st.sidebar.slider("Days into the Future", 10, 180, 30, step=10)

use_gridsearch = st.sidebar.checkbox("Use GridSearchCV", value=False)
use_bootstrap = st.sidebar.checkbox("Use Bootstrapping for CI", value=False)

eps = st.sidebar.number_input("Enter EPS (Earnings Per Share)", min_value=0.01, value=5.0, step=0.01)

# ----------------------------------------
# Main Forecasting
st.title(f"📈 Forecasting Stock Price: {ticker.upper()}")

try:
    df = get_stock_data(ticker, period)
    df = add_technical_indicators(df)
    df.dropna(inplace=True)

    latest_close = df['Close'].iloc[-1]
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    mu, sigma = log_returns.mean(), log_returns.std()
    pe_ratio = latest_close / eps if eps > 0 else np.nan
    adjusted_mu = mu * (20 / pe_ratio) if pe_ratio > 0 else mu

    sim_data = monte_carlo_simulation(S0=latest_close, mu=adjusted_mu, sigma=sigma, T=n_days, N=n_days, M=n_simulations)
    final_prices = sim_data[-1, :]
    p5, p50, p95 = np.percentile(final_prices, [5, 50, 95])

    st.subheader("Monte Carlo Results")
    st.write(f"**5th Percentile**: ${p5:.2f}, **Median**: ${p50:.2f}, **95th Percentile**: ${p95:.2f}")

    progress_bar = st.progress(0)
    model, rmse, predicted_price, actual_price, ci_lower, ci_upper = train_random_forest(
        df, n_days, eps, bootstrap_iters=100, use_gridsearch=use_gridsearch, use_bootstrap=use_bootstrap, progress_bar=progress_bar
    )

    ml_pct = (predicted_price - latest_close) / latest_close * 100
    st.subheader("Machine Learning Forecast")
    st.write(f"**Predicted Price**: ${predicted_price:.2f} ({ml_pct:+.2f}%)")
    if ci_lower is not None:
        st.write(f"**95% CI**: ${ci_lower:.2f} to ${ci_upper:.2f}")
    st.write(f"**RMSE**: ${rmse:.2f}")

except Exception as e:
    st.error(f"Error: {e}")

# ----------------------------------------
# Scanner Functionality
st.sidebar.markdown("---")
scan_sp500 = st.sidebar.checkbox("📡 Run S&P 500 Scanner")

if scan_sp500:
    scan_period = st.sidebar.selectbox("Scanner Period", ["1y", "2y", "5y"], index=1)
    scan_days = st.sidebar.slider("Forecast Days (Scan)", 10, 90, 30, step=10)
    scan_sims = st.sidebar.slider("Simulations (Scan)", 100, 1000, 300, step=100)

    if st.sidebar.button("Start Scan"):
        sp500_tickers = get_sp500_tickers()
        results = []
        bar = st.progress(0)
        for i, tk in enumerate(sp500_tickers):
            try:
                df = get_stock_data(tk, scan_period)
                df = add_technical_indicators(df)
                df.dropna(inplace=True)
                latest = df['Close'].iloc[-1]
                ret = np.log(df['Close'] / df['Close'].shift(1)).dropna()
                mu, sigma = ret.mean(), ret.std()
                pe = latest / eps if eps > 0 else np.nan
                adj_mu = mu * (20 / pe) if pe > 0 else mu

                sim = monte_carlo_simulation(latest, adj_mu, sigma, scan_days, scan_days, scan_sims)
                mc_median = np.percentile(sim[-1], 50)
                mc_pct = (mc_median - latest) / latest * 100

                _, _, ml_pred, _, _, _ = train_random_forest(df, scan_days, eps)
                ml_pct = (ml_pred - latest) / latest * 100
                score = (ml_pct + mc_pct) / 2

                results.append({"Ticker": tk, "Latest Close": latest, "ML %": ml_pct, "MC %": mc_pct, "Score": score})
            except:
                continue
            bar.progress((i + 1) / len(sp500_tickers))

        df_scan = pd.DataFrame(results).sort_values("Score", ascending=False)
        st.subheader("📊 S&P 500 Scanner Results")
        st.dataframe(df_scan.reset_index(drop=True).head(20))
