import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

# --------------------------------------
# Cache S&P 500 tickers (only once)
@st.cache_data(ttl=86400)
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)
    sp500_df = table[0]
    return sp500_df['Symbol'].tolist()

# --------------------------------------
# Sidebar Inputs
st.sidebar.title("Settings")

ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
period = st.sidebar.selectbox("Historical Data Period", ["6mo", "1y", "2y", "5y"], index=1)
n_simulations = st.sidebar.slider("Monte Carlo Simulations", 100, 10000, 500, step=100)
n_days = st.sidebar.slider("Days into the Future", 10, 180, 30, step=10)

use_gridsearch = st.sidebar.checkbox("Use GridSearchCV (slower, better tuning)", value=False)
use_bootstrap = st.sidebar.checkbox("Use Bootstrapping for CI (slower)", value=False)
use_manual_price = st.sidebar.checkbox("Use Manual Close Price")

manual_price = None
if use_manual_price:
    manual_price = st.sidebar.number_input("Enter Latest Close Price", min_value=0.0, value=150.0, step=0.1)

eps = st.sidebar.number_input("Enter EPS (Earnings Per Share)", min_value=0.01, value=5.0, step=0.01)

# Scanner checkbox and additional scanner options
scan_sp500 = st.sidebar.checkbox("Run S&P 500 Scanner")
if scan_sp500:
    st.sidebar.write("⚠️ Scanner can take several minutes.")
    scan_period = st.sidebar.selectbox("Data Period for Scanner", ["1y", "2y", "5y"], index=1)
    scan_days = st.sidebar.slider("Forecast Days for Scanner", 10, 90, 30, step=10)
    scan_sims = st.sidebar.slider("Simulations for Scanner", 100, 1000, 300, step=100)

# --------------------------------------
# Fetch stock data
@st.cache_data(ttl=3600)
def get_stock_data(ticker_symbol, period="1y"):
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period=period)
    data.dropna(inplace=True)
    return data

# --------------------------------------
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

# --------------------------------------
# Monte Carlo Simulation
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
# ML Model training
def train_random_forest(df, n_days_ahead, eps, bootstrap_iters=1000, use_gridsearch=False, use_bootstrap=False, progress_bar=None):
    df['EPS'] = eps  # Add EPS as constant feature
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
        raise ValueError("Not enough valid data to train the model. Try selecting a longer period.")

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
# Scanner Function
def run_scanner(tickers, period, n_simulations, n_days, eps, use_gridsearch, use_bootstrap):
    results = []
    progress_bar = st.progress(0)
    total = len(tickers)
    for i, tk in enumerate(tickers):
        try:
            df = get_stock_data(tk, period)
            df = add_technical_indicators(df)
            df.dropna(inplace=True)
            latest_close = df['Close'].iloc[-1]
            log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            mu, sigma = log_returns.mean(), log_returns.std()
            pe_ratio = latest_close / eps if eps > 0 else np.nan
            baseline_pe = 20.0
            adjusted_mu = mu * (baseline_pe / pe_ratio) if pe_ratio > 0 else mu

            sim_data = monte_carlo_simulation(S0=latest_close, mu=adjusted_mu, sigma=sigma, T=n_days, N=n_days, M=n_simulations)
            final_prices = sim_data[-1, :]
            mc_p50 = np.percentile(final_prices, 50)

            model, rmse, predicted_price, actual_price, ci_lower, ci_upper, best_params = train_random_forest(
                df, n_days, eps,
                bootstrap_iters=100,
                use_gridsearch=use_gridsearch,
                use_bootstrap=use_bootstrap,
                progress_bar=None
            )

            ml_change_pct = (predicted_price - latest_close) / latest_close * 100
            mc_change_pct = (mc_p50 - latest_close) / latest_close * 100

            combined_score = (ml_change_pct + mc_change_pct) / 2

            results.append({
                "Ticker": tk,
                "Latest Close": latest_close,
                "ML % Change": ml_change_pct,
                "MC Median % Change": mc_change_pct,
                "Combined Score": combined_score,
                "RMSE":
