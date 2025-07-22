import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

st.set_page_config(layout="wide", page_title="Stock Price Forecast (MC + ML + Trade Engine)")

# --------------------------
# Safe writable path for trade log
# --------------------------
log_dir = "/mnt/data/trade_logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "trade_log.csv")

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
# Fetch S&P 500 tickers from Wikipedia
# --------------------------------------
@st.cache_data(ttl=86400)
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    return df['Symbol'].tolist()

# --------------------------------------
# Technical indicators
# --------------------------------------
def add_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Momentum'] = df['Close'].diff(4)
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI_14'] = compute_rsi(df['Close'], 14)
    df['MACD'] = compute_macd(df['Close'])
    df['MACD_Signal'] = compute_macd_signal(df['Close'])
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, span_short=12, span_long=26):
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    return ema_short - ema_long

def compute_macd_signal(series, span_short=12, span_long=26, span_signal=9):
    macd = compute_macd(series, span_short, span_long)
    return macd.ewm(span=span_signal, adjust=False).mean()

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
# ML Model
# --------------------------------------
def train_random_forest(df, n_days_ahead, eps, bootstrap_iters=1000, use_gridsearch=False, use_bootstrap=False, progress_bar=None):
    df['EPS'] = eps
    df['Target'] = df['Close'].shift(-n_days_ahead)
    df = df.dropna()

    features = ['Close', 'SMA_20', 'Momentum', 'Volatility', 'Volume_Change', 'EPS',
                'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal']
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    if use_gridsearch:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10]
        }
        tscv = TimeSeriesSplit(n_splits=5)
        model = GridSearchCV(RandomForestRegressor(random_state=0), param_grid, cv=tscv, scoring='neg_mean_squared_error')
        model.fit(X_train, y_train)
        best_model = model.best_estimator_
        best_params = model.best_params_
    else:
        best_model = RandomForestRegressor(n_estimators=100, random_state=0)
        best_model.fit(X_train, y_train)
        best_params = {'n_estimators': 100, 'max_depth': None}

    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    latest_features = X.iloc[[-1]]
    predicted_price = best_model.predict(latest_features)[0]

    ci_lower, ci_upper = None, None
    if use_bootstrap and progress_bar:
        boot_preds = []
        for i in range(bootstrap_iters):
            X_resampled, y_resampled = resample(X_train, y_train)
            rf = RandomForestRegressor(**best_model.get_params())
            rf.fit(X_resampled, y_resampled)
            boot_preds.append(rf.predict(latest_features)[0])
            if i % max(1, bootstrap_iters // 100) == 0:
                progress_bar.progress(min(i / bootstrap_iters, 1.0))
        ci_lower = np.percentile(boot_preds, 2.5)
        ci_upper = np.percentile(boot_preds, 97.5)

    return best_model, rmse, predicted_price, y_test.values[-1], ci_lower, ci_upper, best_params

# --------------------------------------
# Position Sizing + Trade Logging
# --------------------------------------
def simulate_trade(latest_close, predicted_price, ci_lower, ci_upper, account_size, risk_pct):
    stop_price = ci_lower if ci_lower is not None else predicted_price * 0.95
    target_price = ci_upper if ci_upper is not None else predicted_price * 1.05
    risk_per_trade = account_size * (risk_pct / 100)
    stop_loss_per_share = max(latest_close - stop_price, 0.01)
    shares = int(risk_per_trade / stop_loss_per_share)
    entry_value = shares * latest_close
    return {
        'Entry Price': latest_close,
        'Stop Price': stop_price,
        'Target Price': target_price,
        'Shares': shares,
        'Entry Value': entry_value
    }

def log_trade(ticker, trade, predicted_price, actual_price, n_days):
    pnl = (actual_price - trade['Entry Price']) * trade['Shares']
    trade_record = {
        'Ticker': ticker,
        'Entry Price': trade['Entry Price'],
        'Predicted Price': predicted_price,
        'Actual Price': actual_price,
        'Stop Price': trade['Stop Price'],
        'Target Price': trade['Target Price'],
        'Shares': trade['Shares'],
        'Entry Value': trade['Entry Value'],
        'PnL': pnl,
        'Days Held': n_days
    }
    df_log = pd.DataFrame([trade_record])
    if os.path.exists(log_path):
        df_log.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df_log.to_csv(log_path, index=False)

# --------------------------------------
# UI: Sidebar
# --------------------------------------
st.sidebar.title("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
period = st.sidebar.selectbox("Historical Data Period", ["6mo", "1y", "2y", "5y"], index=1)
n_simulations = st.sidebar.slider("Monte Carlo Simulations", 100, 10000, 500, step=100)
n_days = st.sidebar.slider("Days into the Future", 10, 180, 30, step=10)
use_gridsearch = st.sidebar.checkbox("Use GridSearchCV (slower)", value=False)
use_bootstrap = st.sidebar.checkbox("Use Bootstrapping for CI", value=False)
use_manual_price = st.sidebar.checkbox("Use Manual Close Price")
manual_price = st.sidebar.number_input("Manual Close Price", min_value=0.0, value=150.0) if use_manual_price else None
eps = st.sidebar.number_input("EPS (Earnings Per Share)", min_value=0.01, value=5.0, step=0.01)
account_size = st.sidebar.number_input("Account Size ($)", min_value=1000.0, value=10000.0, step=100.0)
risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.1, 10.0, 1.0, step=0.1)

# --------------------------------------
# MAIN
# --------------------------------------
st.title(f"ð Forecasting Stock Price: {ticker.upper()}")
try:
    df = get_stock_data(ticker, period)
    df = add_technical_indicators(df)
    if use_manual_price:
        df.iloc[-1, df.columns.get_loc("Close")] = manual_price
        df = add_technical_indicators(df)
    df.dropna(inplace=True)
    latest_close = df['Close'].iloc[-1]
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    mu, sigma = log_returns.mean(), log_returns.std()
    pe_ratio = latest_close / eps if eps > 0 else np.nan
    baseline_pe = 20.0
    adjusted_mu = mu * (baseline_pe / pe_ratio) if pe_ratio > 0 else mu
    sim_data = monte_carlo_simulation(latest_close, adjusted_mu, sigma, n_days, n_days, n_simulations)
    final_prices = sim_data[-1, :]
    p5, p50, p95 = np.percentile(final_prices, [5, 50, 95])

    st.subheader("Monte Carlo Simulation Results")
    st.write(f"**5th Percentile**: ${p5:.2f}")
    st.write(f"**Median (P50)**: ${p50:.2f}")
    st.write(f"**95th Percentile**: ${p95:.2f}")

    progress_bar = st.progress(0)
    model, rmse, predicted_price, actual_price, ci_lower, ci_upper, best_params = train_random_forest(
        df, n_days, eps, 1000, use_gridsearch, use_bootstrap, progress_bar)

    ml_change_pct = (predicted_price - latest_close) / latest_close * 100
    st.subheader("Machine Learning Forecast")
    st.write(f"**Predicted Price ({n_days} days)**: ${predicted_price:.2f}")
    if ci_lower and ci_upper:
        st.write(f"**95% CI**: ${ci_lower:.2f} - ${ci_upper:.2f}")
    st.write(f"**RMSE**: ${rmse:.2f}")
    st.write(f"**Expected ML Change**: {ml_change_pct:+.2f}%")

    trade = simulate_trade(latest_close, predicted_price, ci_lower, ci_upper, account_size, risk_pct)
    st.subheader("Trade Plan (Simulated)")
    st.write(f"**Shares**: {trade['Shares']}, **Entry Value**: ${trade['Entry Value']:.2f}")
    st.write(f"**Stop-Loss**: ${trade['Stop Price']:.2f}, **Take-Profit**: ${trade['Target Price']:.2f}")

    log_trade(ticker, trade, predicted_price, actual_price, n_days)
    pnl_pct = (actual_price - latest_close) / latest_close * 100
    st.subheader("Backtest Result")
    st.write(f"**Return over {n_days} days**: {pnl_pct:+.2f}%")

except Exception as e:
    st.error(f"Error: {e}")
