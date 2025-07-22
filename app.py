
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
log_dir = "./trade_logs"
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

def log_trade(ticker, entry_price, stop_loss, target_price, predicted_price, log_dir):
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = os.path.join(log_dir, f"{ticker}_log.csv")
    entry = {
        "Time": date,
        "Ticker": ticker,
        "Entry": entry_price,
        "Stop": stop_loss,
        "Target": target_price,
        "Prediction": predicted_price
    }
    df_entry = pd.DataFrame([entry])
    if os.path.exists(filename):
        df_entry.to_csv(filename, mode='a', header=False, index=False)
    else:
        df_entry.to_csv(filename, index=False)

# Sidebar and app logic would follow after this...
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
        os.makedirs("./trade_logs", exist_ok=True)
    except Exception as e:
        st.error(f"Failed to create log directory: {e}")
# --------------------------------------
# Main App Logic
# --------------------------------------
st.title(f"📈 Forecasting Stock Price: {ticker.upper()}")

try:
    df = get_stock_data(ticker, period)
    df = add_technical_indicators(df)

    if use_manual_price and manual_price is not None:
        df.iloc[-1, df.columns.get_loc("Close")] = manual_price
        df = add_technical_indicators(df)  # Recalculate indicators

    df.dropna(inplace=True)

    latest_close = df['Close'].iloc[-1]
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    mu, sigma = log_returns.mean(), log_returns.std()

    pe_ratio = latest_close / eps if eps > 0 else np.nan
    baseline_pe = 20.0
    adjusted_mu = mu * (baseline_pe / pe_ratio) if pe_ratio > 0 else mu

    sim_data = monte_carlo_simulation(S0=latest_close, mu=adjusted_mu, sigma=sigma, T=n_days, N=n_days, M=n_simulations)
    final_prices = sim_data[-1, :]
    p5 = np.percentile(final_prices, 5)
    p50 = np.percentile(final_prices, 50)
    p95 = np.percentile(final_prices, 95)

    st.subheader("Monte Carlo Simulation Results")
    st.write(f"**5th percentile price**: ${p5:.2f} ({(p5 - latest_close)/latest_close:.2%})")
    st.write(f"**Median price**: ${p50:.2f} ({(p50 - latest_close)/latest_close:.2%})")
    st.write(f"**95th percentile price**: ${p95:.2f} ({(p95 - latest_close)/latest_close:.2%})")

    progress_bar = st.progress(0)

    model, rmse, predicted_price, actual_price, ci_lower, ci_upper, best_params = train_random_forest(
        df, n_days, eps,
        bootstrap_iters=1000,
        use_gridsearch=use_gridsearch,
        use_bootstrap=use_bootstrap,
        progress_bar=progress_bar
    )

    ml_change_pct = (predicted_price - latest_close) / latest_close * 100

    st.subheader(f"Machine Learning Prediction ({n_days}-Day Close)")
    st.write(f"**Predicted Price**: ${predicted_price:.2f}")
    if ci_lower is not None and ci_upper is not None:
        st.write(f"**95% Prediction Interval**: ${ci_lower:.2f} to ${ci_upper:.2f}")
    st.write(f"**Actual Price (last test sample)**: ${actual_price:.2f}")
    st.write(f"**RMSE**: ${rmse:.2f}")
    st.write(f"**Expected Price Change**: {ml_change_pct:+.2f}%")
    st.write(f"**Best Model Parameters**: `{best_params}`")
    st.markdown("---")
    st.subheader("📊 Final Summary")

    position_size = capital * risk_pct / (latest_close * 0.05)  # Assume 5% stop range if CI unavailable
    stop_price = ci_lower if ci_lower is not None else p5
    target_price = ci_upper if ci_upper is not None else p95

    stop_loss_pct = (stop_price - latest_close) / latest_close if stop_price else -0.05
    take_profit_pct = (target_price - latest_close) / latest_close if target_price else 0.1

    position_size = max(0, position_size)
    stop_price = round(stop_price, 2) if stop_price else None
    target_price = round(target_price, 2) if target_price else None

    if ml_change_pct > 1 and p50 > latest_close:
        st.success(f"**Likely Upward Trend** — ML: ~{ml_change_pct:.2f}%, MC: {(p50 - latest_close)/latest_close:.2%}")
        if log_trade:
            log_trade(ticker, latest_close, predicted_price, stop_price, target_price, position_size)
    elif ml_change_pct < -1 and p50 < latest_close:
        st.error(f"**Likely Downward Trend** — ML: ~{ml_change_pct:.2f}%, MC: {(p50 - latest_close)/latest_close:.2%}")
    else:
        st.warning("**Uncertain** — Mixed or flat predictions. Use caution.")

    st.markdown("---")
    st.subheader("💼 Trade Management")
    st.write(f"**Capital**: ${capital:,.2f}")
    st.write(f"**Position Size Estimate**: {int(position_size)} shares")
    st.write(f"**Stop-Loss Price**: ${stop_price}")
    st.write(f"**Take-Profit Price**: ${target_price}")

    # Backtest Log Viewer
    if log_trade:
        try:
            log_df = pd.read_csv(os.path.join(log_dir, "trades.csv"))
            st.subheader("🧾 Trade Log")
            st.dataframe(log_df.tail(10))

            log_df['PL %'] = ((log_df['Exit Price'] - log_df['Entry Price']) / log_df['Entry Price']) * 100
            win_rate = (log_df['PL %'] > 0).mean() * 100
            avg_return = log_df['PL %'].mean()
            st.markdown(f"**Backtest Summary**")
            st.write(f"**Win Rate**: {win_rate:.2f}%")
            st.write(f"**Average Return per Trade**: {avg_return:.2f}%")

        except Exception as e:
            st.warning(f"Could not load trade log: {e}")

except Exception as e:
    st.error(f"Error loading data or running simulation: {e}")
