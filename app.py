import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

st.set_page_config(layout="wide", page_title="Stock Price Forecast (MC + ML)")

# --------------------------------------
# Create log directory (relative path)
# --------------------------------------
log_dir = "./log_trade"
try:
    os.makedirs(log_dir, exist_ok=True)
except Exception as e:
    st.error(f"Error creating directory: {e}")

# --------------------------------------
@st.cache_data(ttl=3600)
def get_stock_data(ticker_symbol, period="1y"):
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period=period)
    data.dropna(inplace=True)
    return data

@st.cache_data(ttl=86400)
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    return df['Symbol'].tolist()

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
    macd = ema_short - ema_long
    return macd

def compute_macd_signal(series, span_short=12, span_long=26, span_signal=9):
    macd = compute_macd(series, span_short, span_long)
    macd_signal = macd.ewm(span=span_signal, adjust=False).mean()
    return macd_signal

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

def train_random_forest(df, n_days_ahead, eps, bootstrap_iters=1000, use_gridsearch=False, use_bootstrap=False, progress_bar=None):
    df['EPS'] = eps
    df['Target'] = df['Close'].shift(-n_days_ahead)
    df = df.dropna()

    features = ['Close', 'SMA_20', 'Momentum', 'Volatility', 'Volume_Change', 'EPS',
                'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal']
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

# Renamed to avoid conflict with checkbox variable 'log_trades'
def log_trade_entry(ticker, entry_price, predicted_price, stop_loss, target_price, position_size):
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = os.path.join(log_dir, "trades.csv")
    entry = {
        "Time": date,
        "Ticker": ticker,
        "Entry Price": entry_price,
        "Predicted Price": predicted_price,
        "Stop Price": stop_loss,
        "Target Price": target_price,
        "Position Size": position_size
    }
    df_entry = pd.DataFrame([entry])
    if os.path.exists(filename):
        df_entry.to_csv(filename, mode='a', header=False, index=False)
    else:
        df_entry.to_csv(filename, index=False)

# --------------------------------------
# Sidebar Inputs
# --------------------------------------
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
capital = st.sidebar.number_input("Trading Capital ($)", min_value=100.0, value=10000.0, step=100.0)
risk_pct = st.sidebar.slider("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1) / 100.0
account_balance = st.sidebar.number_input("Account Balance ($)", min_value=100.0, value=10000.0, step=100.0)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
rr_ratio = st.sidebar.slider("Risk-Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.5)

log_trades = st.sidebar.checkbox("Log Trades & Backtest", value=True)

# Create log directory
if log_trades:
    try:
        os.makedirs("./log_trade", exist_ok=True)
    except Exception as e:
        st.error(f"Failed to create log directory: {e}")

# --------------------------------------
# Main Logic
# --------------------------------------
try:
    df = get_stock_data(ticker, period=period)
    if df.empty:
        st.error(f"No data found for ticker {ticker} with period {period}.")
        st.stop()

    df = add_technical_indicators(df)
    df.dropna(inplace=True)

    if use_manual_price and manual_price is not None:
        latest_close = manual_price
    else:
        latest_close = df['Close'][-1]

    # Calculate daily returns mean and std dev for MC
    daily_returns = df['Close'].pct_change().dropna()
    mu = daily_returns.mean()
    sigma = daily_returns.std()

    # Monte Carlo simulation
    sims = monte_carlo_simulation(S0=latest_close, mu=mu, sigma=sigma, T=n_days/252, N=n_days, M=n_simulations)
    p50 = np.percentile(sims[-1, :], 50)
    p5 = np.percentile(sims[-1, :], 5)
    p95 = np.percentile(sims[-1, :], 95)

    st.header(f"{ticker} Stock Price Forecast")
    st.write(f"Latest Close Price: ${latest_close:.2f}")
    st.write(f"Monte Carlo 50th Percentile Price in {n_days} days: ${p50:.2f}")
    st.write(f"Monte Carlo 5th Percentile Price in {n_days} days: ${p5:.2f}")
    st.write(f"Monte Carlo 95th Percentile Price in {n_days} days: ${p95:.2f}")
    mc_change_pct = (p50 - latest_close) / latest_close * 100
    st.write(f"Predicted % Change (Monte Carlo median): {mc_change_pct:.2f}%")



    progress_bar = st.progress(0) if use_bootstrap else None

    rf_model, rmse, predicted_price, actual_price, ci_lower, ci_upper, best_params = train_random_forest(
        df, n_days, eps,
        bootstrap_iters=100 if use_bootstrap else 0,
        use_gridsearch=use_gridsearch,
        use_bootstrap=use_bootstrap,
        progress_bar=progress_bar
    )

    st.subheader("Random Forest Model")
    st.write(f"RMSE: {rmse:.4f}")
    st.write(f"Predicted Price in {n_days} days: ${predicted_price:.2f}")
    ml_change_pct = (predicted_price - latest_close) / latest_close * 100
    st.write(f"Predicted % Change in {n_days} days: {ml_change_pct:.2f}%")
    if ci_lower and ci_upper:
        st.write(f"95% Confidence Interval: (${ci_lower:.2f}, ${ci_upper:.2f})")
    st.write(f"Best Params: {best_params}")

    # Calculate % change from latest close
    ml_change_pct = ((predicted_price - latest_close) / latest_close) * 100

    st.subheader("Trade Management")
    position_size = (capital * risk_pct) / (latest_close * risk_pct)
    stop_price = latest_close * (1 - risk_pct)
    target_price = latest_close * (1 + risk_pct * rr_ratio)

    st.write(f"Position Size (shares): {position_size:.2f}")
    st.write(f"Stop Loss Price: ${stop_price:.2f}")
    st.write(f"Take Profit Price: ${target_price:.2f}")

    # Trade signals and logging
    if ml_change_pct > 1 and p50 > latest_close:
        st.success(f"**Likely Upward Trend** — ML: ~{ml_change_pct:.2f}%, MC: {(p50 - latest_close)/latest_close:.2%}")
        if log_trades:
            log_trade_entry(ticker, latest_close, predicted_price, stop_price, target_price, position_size)
    elif ml_change_pct < -1 and p50 < latest_close:
        st.error(f"**Likely Downward Trend** — ML: ~{ml_change_pct:.2f}%, MC: {(p50 - latest_close)/latest_close:.2%}")
    else:
        st.warning("**Uncertain** — Mixed or flat predictions. Use caution.")

except Exception as e:
    st.error(f"Error: {e}")

# -----------------------------
# Backtest Log Viewer
# -----------------------------
if log_trades:
    try:
        log_path = os.path.join(log_dir, "trades.csv")
        if os.path.exists(log_path):
            log_df = pd.read_csv(log_path)
            st.subheader("🧾 Trade Log")
            st.dataframe(log_df.tail(10))

            if 'Exit Price' in log_df.columns:
                log_df['PL %'] = ((log_df['Exit Price'] - log_df['Entry Price']) / log_df['Entry Price']) * 100
                win_rate = (log_df['PL %'] > 0).mean() * 100
                avg_return = log_df['PL %'].mean()
                st.markdown(f"**Backtest Summary**")
                st.write(f"**Win Rate**: {win_rate:.2f}%")
                st.write(f"**Average Return per Trade**: {avg_return:.2f}%")
            else:
                st.info("Trade log has no 'Exit Price' column yet. P/L calculation not available.")
        else:
            st.info("No trade log found yet.")
    except Exception as e:
        st.warning(f"Could not load trade log: {e}")
