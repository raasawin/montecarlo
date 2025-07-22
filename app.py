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

# -------------------------
# New: Automated ML Backtest Function
# -------------------------
def ml_backtest(df, n_days_ahead, eps, capital, risk_pct, rr_ratio):
    df = df.copy()
    df['EPS'] = eps
    df['Target'] = df['Close'].shift(-n_days_ahead)
    df = add_technical_indicators(df)
    df.dropna(inplace=True)

    features = ['Close', 'SMA_20', 'Momentum', 'Volatility', 'Volume_Change', 'EPS',
                'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal']

    # Store trade results here
    trades = []

    # Walk-forward backtest with expanding train set:
    for i in range(100, len(df) - n_days_ahead):
        train_df = df.iloc[:i]
        test_df = df.iloc[i:i+1]
        
        X_train = train_df[features]
        y_train = train_df['Target']
        
        # Skip if not enough training data
        if len(X_train) < 20:
            continue
        
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)

        X_test = test_df[features]
        predicted_price = model.predict(X_test)[0]
        entry_price = test_df['Close'].values[0]
        true_future_price = df['Close'].iloc[i + n_days_ahead]  # Real future price at prediction horizon
        
        # Trading logic:
        # Only enter trade if predicted price > entry_price (expecting gain)
        if predicted_price > entry_price:
            # Stop loss 5% below entry
            stop_loss = entry_price * 0.95
            # Target price based on RR ratio
            target_price = entry_price + rr_ratio * (entry_price - stop_loss)
            
            # Position size shares based on risk_pct of capital
            position_size = int((capital * risk_pct) / (entry_price - stop_loss)) if (entry_price - stop_loss) != 0 else 0
            
            # Exit price: simulate selling at true_future_price
            exit_price = true_future_price
            
            pl_pct = 100 * (exit_price - entry_price) / entry_price
            
            trades.append({
                "Entry Time": train_df.index[-1].strftime('%Y-%m-%d'),
                "Entry Price": entry_price,
                "Predicted Price": predicted_price,
                "Exit Price": exit_price,
                "Stop Loss": stop_loss,
                "Target Price": target_price,
                "Position Size": position_size,
                "P/L %": pl_pct,
                "Trade Result": "Win" if pl_pct > 0 else "Loss"
            })

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    win_rate = trades_df[trades_df["P/L %"] > 0].shape[0] / len(trades_df) * 100
    avg_return = trades_df["P/L %"].mean()
    total_return = (trades_df["P/L %"] / 100 + 1).prod() - 1
    total_return_pct = total_return * 100

    return trades_df, win_rate, avg_return, total_return_pct

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

log_trades = st.sidebar.checkbox("Log Trades", value=True)

# --------------------------------------
# Main App Logic
# --------------------------------------
st.title("Stock Price Forecast with Monte Carlo & Random Forest ML")

# Fetch Data
try:
    df = get_stock_data(ticker, period)
except Exception as e:
    st.error(f"Failed to download data for {ticker}: {e}")
    st.stop()

df = add_technical_indicators(df)

# Display latest data
st.subheader(f"{ticker} Historical Price Data")
st.write(df.tail())

# Monte Carlo Simulation
st.subheader("Monte Carlo Price Simulation")

S0 = manual_price if use_manual_price and manual_price else df['Close'][-1]
mu = df['Close'].pct_change().mean()
sigma = df['Close'].pct_change().std()
T = n_days / 252  # trading days fraction
N = n_days
M = n_simulations

simulations = monte_carlo_simulation(S0, mu, sigma, T, N, M)

st.line_chart(simulations)

# ML Model Training & Prediction
st.subheader("Random Forest Regression Model")

progress_bar = st.progress(0) if use_bootstrap else None

try:
    model, rmse, predicted_price, latest_close, ci_lower, ci_upper, best_params = train_random_forest(
        df, n_days, eps, bootstrap_iters=500 if use_bootstrap else 0,
        use_gridsearch=use_gridsearch, use_bootstrap=use_bootstrap, progress_bar=progress_bar)
except Exception as e:
    st.error(f"Model training failed: {e}")
    st.stop()

st.write(f"RMSE on Test Set: {rmse:.2f}")
st.write(f"Latest Actual Close Price: ${latest_close:.2f}")
st.write(f"Predicted Price in {n_days} days: ${predicted_price:.2f}")
if ci_lower and ci_upper:
    st.write(f"95% Prediction Interval: ${ci_lower:.2f} - ${ci_upper:.2f}")
st.write(f"Best Model Parameters: {best_params}")

# Trade Logging Inputs
if log_trades:
    position_size = int((capital * risk_pct) / (S0 * 0.05)) if S0 * 0.05 != 0 else 0
    stop_loss = S0 * 0.95
    target_price = S0 + rr_ratio * (S0 - stop_loss)
    if st.button("Log Trade"):
        log_trade_entry(ticker, S0, predicted_price, stop_loss, target_price, position_size)
        st.success("Trade logged successfully.")

# --------------------------------------
# New Backtesting Section
# --------------------------------------
st.subheader("Backtest ML Trading Strategy")

if st.button("Run Backtest"):
    with st.spinner("Running backtest... this may take some time"):
        result = ml_backtest(df, n_days, eps, capital, risk_pct, rr_ratio)
    if result is None:
        st.warning("Not enough data to run backtest or no trades generated.")
    else:
        trades_df, win_rate, avg_return, total_return_pct = result
        st.write("Backtest Trades:")
        st.dataframe(trades_df)
        st.markdown(f"**Win Rate:** {win_rate:.2f}%")
        st.markdown(f"**Average Return per Trade:** {avg_return:.2f}%")
        st.markdown(f"**Total Cumulative Return:** {total_return_pct:.2f}%")

        # Summary plot
        st.line_chart((trades_df["P/L %"]/100 + 1).cumprod())

# --------------------------------------
# Display logged trades if any
# --------------------------------------
st.subheader("Logged Trades History")
trade_log_path = os.path.join(log_dir, "trades.csv")
if os.path.exists(trade_log_path):
    try:
        trade_log_df = pd.read_csv(trade_log_path)
        st.dataframe(trade_log_df)
    except Exception as e:
        st.error(f"Error reading trade log: {e}")
else:
    st.info("No trade logs found.")

