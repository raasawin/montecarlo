import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

st.set_page_config(layout="wide", page_title="Stock Price Forecast (MC + ML)")
# -------------------------
# ML Walk-Forward Backtest
# -------------------------
def ml_backtest(df, n_days_ahead, eps, capital, risk_pct, rr_ratio):
    df = df.copy()
    df['EPS'] = eps
    df['Target'] = df['Close'].shift(-n_days_ahead)
    df = add_technical_indicators(df)
    df.dropna(inplace=True)

    features = ['Close', 'SMA_20', 'Momentum', 'Volatility', 'Volume_Change', 'EPS',
                'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal']

    trades = []

    for i in range(100, len(df) - n_days_ahead):
        train_df = df.iloc[:i]
        test_df = df.iloc[i:i+1]
        
        X_train = train_df[features]
        y_train = train_df['Target']
        
        if len(X_train) < 20:
            continue
        
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)

        X_test = test_df[features]
        predicted_price = model.predict(X_test)[0]
        entry_price = test_df['Close'].values[0]
        true_future_price = df['Close'].iloc[i + n_days_ahead]
        
        if predicted_price > entry_price:
            stop_loss = entry_price * 0.95
            target_price = entry_price + rr_ratio * (entry_price - stop_loss)
            risk_per_share = entry_price - stop_loss
            position_size = int((capital * risk_pct) / risk_per_share) if risk_per_share != 0 else 0
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
# Sidebar Inputs for Backtesting
capital = st.sidebar.number_input("Backtest Capital ($)", min_value=100.0, value=10000.0, step=100.0)
risk_pct = st.sidebar.slider("Backtest Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1) / 100.0
rr_ratio = st.sidebar.slider("Backtest Risk-Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.5)

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

except Exception as e:
    st.error(f"Error loading data or running simulation: {e}")

import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, r2_score

# Sidebar toggle for backtesting
run_backtest = st.sidebar.checkbox("Run Backtest on Test Set", value=False)

def backtest_model_performance(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    residuals = y_true - y_pred

    st.subheader("📊 Backtest Results")

    # Metrics summary
    metrics_df = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "R² Score"],
        "Value": [rmse, mae, r2]
    })
    st.table(metrics_df.style.format({"Value": "{:.4f}"}))

    # Predicted vs Actual line plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true, mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicted Price'))
    fig.update_layout(
        title="Actual vs Predicted Stock Price",
        xaxis_title="Test Sample Index",
        yaxis_title="Price",
        legend=dict(x=0, y=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Residuals histogram
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=residuals, nbinsx=30))
    fig2.update_layout(
        title="Prediction Residuals Histogram",
        xaxis_title="Residual (Actual - Predicted)",
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig2, use_container_width=True)

# --- Insert after your model training and prediction code ---

# Prepare test features and true values for backtesting
features = ['Close', 'SMA_20', 'Momentum', 'Volatility', 'Volume_Change', 'EPS',
            'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal']

df['EPS'] = eps
df['Target'] = df['Close'].shift(-n_days)
df.dropna(inplace=True)

X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Generate predictions on test set
y_pred = model.predict(X_test)

# Display backtest if toggled
if run_backtest:
    backtest_model_performance(y_test, y_pred)

# --------------------------------------
# ML Strategy Backtesting Section
# --------------------------------------
st.subheader("🔄 ML Strategy Backtest (Walk-Forward)")

if st.button("Run ML Backtest"):
    with st.spinner("Running ML backtest... please wait..."):
        result = ml_backtest(df, n_days, eps, capital, risk_pct, rr_ratio)

    if result is None:
        st.warning("No trades were generated during backtest (or not enough data).")
    else:
        trades_df, win_rate, avg_return, total_return_pct = result
        st.write("Trade History:")
        st.dataframe(trades_df)

        st.markdown(f"**Win Rate:** `{win_rate:.2f}%`")
        st.markdown(f"**Avg Return per Trade:** `{avg_return:.2f}%`")
        st.markdown(f"**Total Return (Cumulative):** `{total_return_pct:.2f}%`")

        # Equity curve
        st.line_chart((trades_df["P/L %"] / 100 + 1).cumprod())

