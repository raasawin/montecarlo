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
# Technical indicators (ALL added)
# --------------------------------------
def add_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Momentum'] = df['Close'].diff(4)
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['Volume_Change'] = df['Volume'].pct_change()
    # Added indicators:
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
    df['EPS'] = eps  # Add EPS as a constant feature
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
st.sidebar.markdown("---")
st.sidebar.header("Risk Settings")

account_size = st.sidebar.number_input("Account Size ($)", value=10000, min_value=1000)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.1, 5.0, 1.0, step=0.1) / 100
use_take_profit = st.sidebar.checkbox("Use Take-Profit Target", value=True)

# ---------------------------
# Run S&P 500 Scanner Button
# ---------------------------
if st.sidebar.button("Run S&P 500 Scanner"):
    with st.spinner("Running scanner on S&P 500..."):
        sp500_tickers = get_sp500_tickers()
        results = []
        progress_bar = st.progress(0)
        total = len(sp500_tickers)

        for i, scan_ticker in enumerate(sp500_tickers):
            try:
                df_scan = get_stock_data(scan_ticker, period)
                df_scan = add_technical_indicators(df_scan)

                if df_scan.empty or len(df_scan) < 60:
                    continue

                latest_close = df_scan['Close'].iloc[-1]
                log_returns = np.log(df_scan['Close'] / df_scan['Close'].shift(1)).dropna()
                mu, sigma = log_returns.mean(), log_returns.std()

                pe_ratio = latest_close / eps if eps > 0 else np.nan
                baseline_pe = 20.0
                adjusted_mu = mu * (baseline_pe / pe_ratio) if pe_ratio > 0 else mu

                sim_data = monte_carlo_simulation(S0=latest_close, mu=adjusted_mu, sigma=sigma, T=n_days, N=n_days, M=n_simulations)
                final_prices = sim_data[-1, :]
                mc_p50 = np.percentile(final_prices, 50)
                mc_change_pct = (mc_p50 - latest_close) / latest_close * 100

                model_scan, _, predicted_price, _, _, _, _ = train_random_forest(
                    df_scan, n_days, eps,
                    bootstrap_iters=100,
                    use_gridsearch=False,
                    use_bootstrap=False,
                    progress_bar=None
                )
                ml_change_pct = (predicted_price - latest_close) / latest_close * 100

                results.append({
                    'Ticker': scan_ticker,
                    'ML % Increase': ml_change_pct,
                    'MC Median % Increase': mc_change_pct
                })
            except Exception:
                continue
            progress_bar.progress((i + 1) / total)

        if results:
            results_df = pd.DataFrame(results)
            results_df.sort_values(by='ML % Increase', ascending=False, inplace=True)
            st.subheader("S&P 500 Scanner Results (Ranked by ML % Increase)")
            st.dataframe(results_df.style.format({"ML % Increase": "{:.2f}%", "MC Median % Increase": "{:.2f}%"}))
        else:
            st.warning("No results to display.")

# --------------------------------------
# Main App Logic
# --------------------------------------
st.title(f"📈 Forecasting Stock Price: {ticker.upper()}")

try:
    df = get_stock_data(ticker, period)
    df = add_technical_indicators(df)

    if use_manual_price and manual_price is not None:
        df.iloc[-1, df.columns.get_loc("Close")] = manual_price
        df = add_technical_indicators(df)  # Recalculate indicators after price change

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
# --- Trade Plan Logic ---
stop_price = ci_lower if ci_lower is not None else p5
take_profit_price = predicted_price if use_take_profit else None

dollar_risk_per_share = max(1e-6, latest_close - stop_price)
shares = (risk_per_trade * account_size) / dollar_risk_per_share
shares = min(shares, account_size / latest_close)
max_loss = shares * dollar_risk_per_share

st.subheader("📏 Trade Plan")
st.write(f"**Entry**: ${latest_close:.2f}")
st.write(f"**Stop Loss**: ${stop_price:.2f}")
if use_take_profit:
    st.write(f"**Take Profit**: ${take_profit_price:.2f}")
st.write(f"**Shares to Buy**: {int(shares)}")
st.write(f"**Max Risk**: ${max_loss:.2f}")

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
    if ml_change_pct > 1 and p50 > latest_close:
        st.success(f"**Likely Upward Trend** — ML: ~{ml_change_pct:.2f}%, MC: {(p50 - latest_close)/latest_close:.2%}")
    elif ml_change_pct < -1 and p50 < latest_close:
        st.error(f"**Likely Downward Trend** — ML: ~{ml_change_pct:.2f}%, MC: {(p50 - latest_close)/latest_close:.2%}")
    else:
        st.warning("**Uncertain** — Mixed or flat predictions. Use caution.")
# --- Trade Log ---
trade_log = pd.DataFrame([{
    'Date': df.index[-1],
    'Ticker': ticker.upper(),
    'Entry': latest_close,
    'Stop': stop_price,
    'Target': take_profit_price,
    'Shares': int(shares),
    'Predicted': predicted_price,
    'Actual_Future': df['Close'].shift(-n_days).iloc[-1] if len(df) > n_days else None,
    'Model_Change_%': ml_change_pct,
    'MC_Change_%': (p50 - latest_close) / latest_close * 100
}])

st.subheader("📘 Trade Log (Latest)")
st.dataframe(trade_log)

except Exception as e:
    st.error(f"Error loading data or running simulation: {e}")
st.markdown("---")
st.subheader("🔁 Backtesting Simulation")

def backtest_model(df, eps, n_days, risk_pct, account_size, use_take_profit=True, lookback=200):
    logs = []
    equity = account_size

    for i in range(lookback, len(df) - n_days):
        try:
            df_bt = df.iloc[:i].copy()
            df_bt = add_technical_indicators(df_bt)
            df_bt.dropna(inplace=True)
            model, _, pred, _, _, _, _ = train_random_forest(df_bt, n_days, eps)

            entry = df_bt['Close'].iloc[-1]
            stop = entry * 0.95
            target = pred if use_take_profit else None
            future_close = df['Close'].iloc[i + n_days]

            dollar_risk = entry - stop
            shares = (risk_pct * equity) / dollar_risk
            shares = min(shares, equity / entry)

            if future_close <= stop:
                exit_price = stop
                outcome = "Stopped"
            elif target and future_close >= target:
                exit_price = target
                outcome = "Target Hit"
            else:
                exit_price = future_close
                outcome = "Held"

            pnl = shares * (exit_price - entry)
            equity += pnl

            logs.append({
                'Date': df.index[i],
                'Entry': entry,
                'Exit': exit_price,
                'Target': target,
                'Stop': stop,
                'Actual': future_close,
                'Shares': int(shares),
                'PnL': pnl,
                'Equity': equity,
                'Outcome': outcome
            })
        except:
            continue

    return pd.DataFrame(logs)

if st.checkbox("Run Backtest on This Ticker"):
    with st.spinner("Running backtest..."):
        bt_results = backtest_model(df.copy(), eps, n_days, risk_per_trade, account_size, use_take_profit)
        if not bt_results.empty:
            st.write(f"**Final Equity:** ${bt_results['Equity'].iloc[-1]:.2f}")
            st.write(f"**Total Return:** {(bt_results['Equity'].iloc[-1] - account_size)/account_size:.2%}")
            st.write(f"**Win Rate:** {(bt_results['PnL'] > 0).mean():.2%}")
            st.write(f"**Total Trades:** {len(bt_results)}")
            st.dataframe(bt_results.tail(10))
        else:
            st.warning("No valid trades generated during backtest.")
