import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
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


# Automated backtesting function
def run_backtest(df, model, n_days_ahead, eps, risk_pct, rr_ratio):
    df = df.copy()
    df['EPS'] = eps
    features = ['Close', 'SMA_20', 'Momentum', 'Volatility', 'Volume_Change', 'EPS',
                'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal']

    trades = []
    position = None  # Track open position dict: {'entry_date', 'entry_price', 'stop_price', 'target_price', ...}

    for i in range(len(df) - n_days_ahead):
        row = df.iloc[i]
        future_row = df.iloc[i + n_days_ahead]

        # Prepare features for prediction at index i
        X_pred = row[features].values.reshape(1, -1)
        pred_price = model.predict(X_pred)[0]
        current_price = row['Close']

        pred_pct_change = (pred_price - current_price) / current_price

        # Define thresholds to trigger buy signals (adjust as needed)
        buy_threshold = 0.01  # e.g. predicted price 1% higher than current price triggers buy
        sell_threshold = -0.01  # predicted price 1% lower triggers no buy or consider short sell (not implemented here)

        # If no open position, check for buy signal
        if position is None and pred_pct_change > buy_threshold:
            position = {
                'entry_index': i,
                'entry_date': df.index[i],
                'entry_price': current_price,
                'stop_price': current_price * (1 - risk_pct),
                'target_price': current_price * (1 + risk_pct * rr_ratio),
                'exit_index': None,
                'exit_date': None,
                'exit_price': None,
                'pl_pct': None,
                'holding_period': None,
                'status': 'open'
            }
        # If position open, check exit conditions
        elif position is not None:
            # Check if target or stop price hit in any of the next n_days_ahead days or time expired
            window_df = df.iloc[i:i + n_days_ahead + 1]

            exit_price = None
            exit_index = None
            exit_date = None

            hit_target = window_df['Close'] >= position['target_price']
            hit_stop = window_df['Close'] <= position['stop_price']

            if hit_target.any():
                exit_index = hit_target.idxmax()
                exit_price = window_df.loc[exit_index, 'Close']
                exit_date = exit_index
            elif hit_stop.any():
                exit_index = hit_stop.idxmax()
                exit_price = window_df.loc[exit_index, 'Close']
                exit_date = exit_index
            else:
                # Exit after holding period if no stop or target hit
                exit_index = df.index[i + n_days_ahead]
                exit_price = df.loc[exit_index, 'Close']
                exit_date = exit_index

            # Save trade
            holding_period = (exit_date - position['entry_date']).days if isinstance(exit_date, pd.Timestamp) else n_days_ahead
            pl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100

            trades.append({
                'Entry Date': position['entry_date'],
                'Entry Price': position['entry_price'],
                'Exit Date': exit_date,
                'Exit Price': exit_price,
                'P/L %': round(pl_pct, 2),
                'Holding Period': holding_period
            })

            position = None  # reset position

    trades_df = pd.DataFrame(trades)

    # Performance metrics
    total_return = trades_df['P/L %'].sum()
    win_rate = (trades_df['P/L %'] > 0).mean()
    avg_return = trades_df['P/L %'].mean()
    max_drawdown = calculate_max_drawdown(trades_df['P/L %'])
    sharpe_ratio = calculate_sharpe_ratio(trades_df['P/L %'])

    return trades_df, total_return, win_rate, avg_return, max_drawdown, sharpe_ratio

def calculate_max_drawdown(returns):
    cum_returns = (1 + returns / 100).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_dd = drawdown.min() * 100
    return max_dd

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    mean_return = returns.mean()
    std_return = returns.std()
    if std_return == 0:
        return 0.0
    sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(252)  # annualized assuming daily returns
    return sharpe

# --------------------------------------
# Streamlit UI
# --------------------------------------
st.title("Stock Price Forecast with MC + ML + Auto Backtesting")

# Sidebar input
ticker_symbol = st.sidebar.text_input("Ticker Symbol", "AAPL").upper()
n_days_ahead = st.sidebar.number_input("Days Ahead to Predict", min_value=1, max_value=30, value=5)
eps = st.sidebar.number_input("EPS", min_value=0.0, step=0.01, value=4.0)
risk_pct = st.sidebar.slider("Risk % (Stop Loss)", min_value=0.01, max_value=0.10, value=0.02, step=0.01)
rr_ratio = st.sidebar.slider("Reward/Risk Ratio (Take Profit)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

if st.sidebar.button("Run Forecast and Backtest"):
    with st.spinner("Fetching data..."):
        df = get_stock_data(ticker_symbol, period="2y")
    if len(df) < 60:
        st.warning("Not enough historical data. Please try another ticker or longer period.")
    else:
        df = add_technical_indicators(df)
        df.dropna(inplace=True)

        st.subheader("Historical Data Sample")
        st.dataframe(df.tail(10))

        progress_bar = st.progress(0)
        try:
            model, rmse, predicted_price, actual_price, ci_lower, ci_upper, best_params = train_random_forest(
                df, n_days_ahead, eps, bootstrap_iters=1000, use_gridsearch=False, use_bootstrap=True, progress_bar=progress_bar
            )
        except Exception as e:
            st.error(f"Training error: {e}")
            st.stop()

        st.write(f"Random Forest Parameters: {best_params}")
        st.write(f"RMSE: {rmse:.4f}")
        st.write(f"Predicted Price {n_days_ahead} days ahead: ${predicted_price:.2f}")
        if ci_lower and ci_upper:
            st.write(f"95% Confidence Interval: (${ci_lower:.2f} - ${ci_upper:.2f})")
        st.write(f"Actual Price: ${actual_price:.2f}")

        # Monte Carlo Simulation
        st.subheader("Monte Carlo Simulation")
        S0 = df['Close'][-1]
        mu = np.log(df['Close'] / df['Close'].shift(1)).mean()
        sigma = np.log(df['Close'] / df['Close'].shift(1)).std()
        T = n_days_ahead
        N = n_days_ahead
        M = 1000

        simulations = monte_carlo_simulation(S0, mu, sigma, T, N, M)
        last_prices = simulations[-1, :]
        st.write(f"Median predicted price by MC: ${np.median(last_prices):.2f}")

        # Auto Backtest
        st.subheader("Automatic Backtest Based on Model Predictions")
        trades_df, total_return, win_rate, avg_return, max_drawdown, sharpe_ratio = run_backtest(df, model, n_days_ahead, eps, risk_pct, rr_ratio)

        if trades_df.empty:
            st.info("No trades were triggered based on the current model and thresholds.")
        else:
            st.write("Backtest Trades Summary")
            st.dataframe(trades_df)

            st.write(f"Total Return (sum of all trades): {total_return:.2f} %")
            st.write(f"Win Rate: {win_rate:.2%}")
            st.write(f"Average Return per Trade: {avg_return:.2f} %")
            st.write(f"Max Drawdown: {max_drawdown:.2f} %")
            st.write(f"Sharpe Ratio (annualized): {sharpe_ratio:.2f}")

        progress_bar.empty()
