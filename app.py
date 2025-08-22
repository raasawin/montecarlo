# app.py
import os
import io
import math
import time
import json
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample
import plotly.graph_objects as go

# Optional stronger models
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

st.set_page_config(layout="wide", page_title="Stock Price Forecast (MC + ML, Pro)")

# --------------------------------------
# Utilities / Paths
# --------------------------------------
TRADE_LOG_DIR = "./trade_logs"
os.makedirs(TRADE_LOG_DIR, exist_ok=True)
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)
SP500_CSV = os.path.join(DATA_DIR, "sp500.csv")  # optional: store a curated list here

def now_str():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

# --------------------------------------
# Fetch stock data
# --------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(ticker_symbol, period="1y"):
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period=period, auto_adjust=False)
    data.dropna(inplace=True)
    return data

# --------------------------------------
# S&P 500 tickers
# --------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_tickers():
    if os.path.exists(SP500_CSV):
        df = pd.read_csv(SP500_CSV)
        col = next((c for c in df.columns if c.lower() in ("symbol", "ticker", "tickers")), df.columns[0])
        return sorted(list(df[col].astype(str).str.upper().unique()))
    else:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]
        return sorted(list(df['Symbol'].astype(str).str.upper().unique()))

# --------------------------------------
# Indicators
# --------------------------------------
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

def compute_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([(high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

# --------------------------------------
# Technical indicators (expanded feature set)
# --------------------------------------
def add_technical_indicators(df, n_days_ahead):
    df = df.copy()
    # Core MAs
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    # Momentum/Vol
    df['Momentum'] = df['Close'].diff(4)
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['Volume_Change'] = df['Volume'].pct_change()
    # Oscillators
    df['RSI_14'] = compute_rsi(df['Close'], 14)
    df['MACD'] = compute_macd(df['Close'])
    df['MACD_Signal'] = compute_macd_signal(df['Close'])
    # Extras
    df['Ret_1'] = df['Close'].pct_change(1)
    df['Ret_5'] = df['Close'].pct_change(5)
    df['Ret_20'] = df['Close'].pct_change(20)
    df['RollVol_20'] = df['Ret_1'].rolling(20).std()
    df['ATR_14'] = compute_atr(df, 14)

    # Shift engineered features forward by n_days to avoid leakage
    feat_cols = [c for c in df.columns if c not in ['Open','High','Low','Close','Volume','Dividends','Stock Splits']]
    for c in feat_cols:
        df[c] = df[c].shift(n_days_ahead)
    return df

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
# Model factory + training (RF / XGB / LGBM)
# --------------------------------------
MODEL_FEATURES = [
    'Close','SMA_20','Momentum','Volatility','Volume_Change','EPS',
    'SMA_50','EMA_20','RSI_14','MACD','MACD_Signal',
    'Ret_1','Ret_5','Ret_20','RollVol_20','ATR_14'
]

def get_model(model_choice: str):
    """Return an untrained model; gracefully fallback if lib missing."""
    if model_choice == "XGBoost":
        if xgb is None:
            st.warning("xgboost not installed. Falling back to RandomForest.")
            return RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
        return xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_SEED,
            tree_method="hist", n_jobs=-1, reg_lambda=1.0
        )
    if model_choice == "LightGBM":
        if lgb is None:
            st.warning("lightgbm not installed. Falling back to RandomForest.")
            return RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
        return lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.05, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_SEED, n_jobs=-1
        )
    # Default RF
    return RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)

def get_param_grid(model_choice: str):
    """Small, time-series-friendly grids."""
    if model_choice == "XGBoost" and xgb is not None:
        return {
            'n_estimators': [300, 600],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    if model_choice == "LightGBM" and lgb is not None:
        return {
            'n_estimators': [400, 800],
            'learning_rate': [0.03, 0.07],
            'num_leaves': [31, 63],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    # RandomForest
    return {
        'n_estimators': [300, 600],
        'max_depth': [None, 8, 16],
        'min_samples_leaf': [1, 3]
    }

def train_model(df, n_days_ahead, eps, model_choice="RandomForest",
                bootstrap_iters=1000, use_gridsearch=False, use_bootstrap=False, progress_bar=None):
    df = df.copy()
    df['EPS'] = eps
    df['Target'] = df['Close'].shift(-n_days_ahead)
    df = df.dropna()

    features = [f for f in MODEL_FEATURES if f in df.columns]
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    if len(X_train) < 60:
        raise ValueError("Not enough valid data to train the model. Try selecting a longer period.")

    best_params = None
    if use_gridsearch:
        tscv = TimeSeriesSplit(n_splits=5)
        base_model = get_model(model_choice)
        param_grid = get_param_grid(model_choice)
        model = GridSearchCV(base_model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
        model.fit(X_train, y_train)
        best_model = model.best_estimator_
        best_params = model.best_params_
    else:
        best_model = get_model(model_choice)
        best_model.fit(X_train, y_train)
        best_params = getattr(best_model, "get_params", lambda: {})()

    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    latest_features = X.iloc[[-1]]
    predicted_price = float(best_model.predict(latest_features)[0])

    ci_lower, ci_upper = None, None
    if use_bootstrap:
        boot_preds = []
        iters = int(bootstrap_iters)
        for i in range(iters):
            X_res, y_res = resample(X_train, y_train, random_state=RANDOM_SEED + i)
            # Recreate a fresh model with same params
            m = get_model(model_choice)
            try:
                m.set_params(**best_model.get_params())
            except Exception:
                pass
            m.fit(X_res, y_res)
            boot_preds.append(float(m.predict(latest_features)[0]))
            if progress_bar:
                progress_bar.progress(min((i + 1) / max(iters, 1), 1.0))
        ci_lower = float(np.percentile(boot_preds, 2.5))
        ci_upper = float(np.percentile(boot_preds, 97.5))

    return best_model, rmse, predicted_price, y_test.values[-1], ci_lower, ci_upper, best_params, y_test, y_pred

# --------------------------------------
# Walk-Forward Strategy Backtest (uses selected model)
# --------------------------------------
def ml_backtest_pro(df_raw, n_days, eps, initial_capital, risk_pct, rr_mult,
                    tx_cost_bps=2, slippage_bps=2, atr_mult_sl=1.5, model_choice="RandomForest"):
    df = add_technical_indicators(df_raw, n_days).copy()
    df['EPS'] = eps
    df['Target'] = df['Close'].shift(-n_days)
    df = df.dropna().copy()

    features = [f for f in MODEL_FEATURES if f in df.columns]

    equity = initial_capital
    trades = []

    for i in range(100, len(df) - n_days):
        train_df = df.iloc[:i]
        test_row = df.iloc[i]
        future_window = df_raw.iloc[i+1:i+1+n_days]

        X_train, y_train = train_df[features], train_df['Target']
        if len(X_train) < 60:
            continue

        model = get_model(model_choice)
        model.fit(X_train, y_train)

        X_test = pd.DataFrame([test_row[features]])
        pred = float(model.predict(X_test)[0])

        entry_price = float(df_raw['Close'].iloc[i])
        latest_atr = float(df_raw['Close'].iloc[i] * 0.0 + (df_raw['High'].iloc[i] - df_raw['Low'].iloc[i]))
        if 'ATR_14' in df.columns and not np.isnan(df['ATR_14'].iloc[i]):
            latest_atr = float(df['ATR_14'].iloc[i])

        # Long entry rule
        if pred > entry_price:
            stop_loss = entry_price - atr_mult_sl * latest_atr
            target = entry_price + rr_mult * (entry_price - stop_loss)

            enter_fill = entry_price * (1 + slippage_bps / 1e4)
            risk_per_share = max(1e-9, enter_fill - stop_loss)
            risk_amount = equity * risk_pct
            qty = math.floor(risk_amount / risk_per_share)
            if qty <= 0:
                continue

            high_next = float(future_window['High'].max()) if not future_window.empty else entry_price
            low_next = float(future_window['Low'].min()) if not future_window.empty else entry_price

            hit_stop = low_next <= stop_loss
            hit_target = high_next >= target

            exit_reason = "TimeExit"
            exit_price = float(df_raw['Close'].iloc[i + n_days])
            if hit_stop and not hit_target:
                exit_reason = "Stop"; exit_price = stop_loss
            elif hit_target and not hit_stop:
                exit_reason = "Target"; exit_price = target
            elif hit_stop and hit_target:
                exit_reason = "StopAndTarget_StopFirst"; exit_price = stop_loss

            exit_fill = exit_price * (1 - slippage_bps / 1e4)
            tc = (enter_fill + exit_fill) * (tx_cost_bps / 1e4)
            pnl = (exit_fill - enter_fill) * qty - tc
            pl_pct = pnl / equity * 100.0

            equity += pnl
            trades.append({
                "Entry Time": df_raw.index[i].strftime('%Y-%m-%d'),
                "Exit Time": df_raw.index[i + n_days].strftime('%Y-%m-%d'),
                "Entry Price": round(enter_fill, 4),
                "Exit Price": round(exit_fill, 4),
                "Stop Loss": round(stop_loss, 4),
                "Target Price": round(target, 4),
                "Qty": int(qty),
                "TxnCost($)": round(tc, 4),
                "P/L ($)": round(pnl, 4),
                "P/L % (on equity)": round(pl_pct, 4),
                "Exit Reason": exit_reason,
                "Equity After": round(equity, 2)
            })

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    wins = (trades_df["P/L ($)"] > 0).sum()
    win_rate = 100.0 * wins / len(trades_df)
    avg_return = trades_df["P/L % (on equity)"].mean()
    equity_curve = (trades_df["P/L ($)"]).cumsum() + initial_capital
    peak = equity_curve.cummax()
    drawdown = equity_curve - peak
    max_dd = drawdown.min()
    max_dd_pct = (max_dd / peak.max()) * 100.0

    daily_ret = trades_df["P/L ($)"] / equity_curve.shift(1).fillna(initial_capital)
    sharpe = (daily_ret.mean() / (daily_ret.std() + 1e-12)) * np.sqrt(252)
    downside = daily_ret[daily_ret < 0].std()
    sortino = (daily_ret.mean() / (downside + 1e-12)) * np.sqrt(252)

    stats = {
        "Win Rate %": win_rate,
        "Avg Trade Return % (on equity)": avg_return,
        "Total Trades": int(len(trades_df)),
        "Ending Equity $": round(float(equity_curve.iloc[-1]), 2),
        "Max Drawdown $": round(float(max_dd), 2),
        "Max Drawdown %": round(float(max_dd_pct), 2),
        "Sharpe (approx)": round(float(sharpe), 3),
        "Sortino (approx)": round(float(sortino), 3),
    }
    return trades_df, stats, equity_curve

# --------------------------------------
# Sidebar Inputs
# --------------------------------------
st.sidebar.title("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()
period = st.sidebar.selectbox("Historical Data Period", ["6mo", "1y", "2y", "5y"], index=1)
n_simulations = st.sidebar.slider("Monte Carlo Simulations", 100, 10000, 500, step=100)
n_days = st.sidebar.slider("Days into the Future", 10, 180, 30, step=10)

# NEW: model selector
model_choice = st.sidebar.selectbox("Select Model", ["RandomForest", "XGBoost", "LightGBM"])

use_gridsearch = st.sidebar.checkbox("Use GridSearchCV (slower, better tuning)", value=False)
use_bootstrap = st.sidebar.checkbox("Use Bootstrapping for CI (slower)", value=False)
use_manual_price = st.sidebar.checkbox("Use Manual Close Price")

manual_price = None
if use_manual_price:
    manual_price = st.sidebar.number_input("Enter Latest Close Price", min_value=0.0, value=150.0, step=0.1)

eps = st.sidebar.number_input("Enter EPS (Earnings Per Share)", min_value=0.01, value=5.0, step=0.01)
capital = st.sidebar.number_input("Backtest Capital ($)", min_value=100.0, value=10000.0, step=100.0)
risk_pct = st.sidebar.slider("Backtest Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1) / 100.0
rr_ratio = st.sidebar.slider("Backtest Risk-Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.5)

tx_cost_bps = st.sidebar.slider("Transaction Cost (bps per side)", 0, 20, 2, 1)
slippage_bps = st.sidebar.slider("Slippage (bps per side)", 0, 30, 2, 1)
atr_mult_sl = st.sidebar.slider("ATR Stop Multiplier", 0.5, 4.0, 1.5, 0.1)

# --------------------------------------
# Run S&P 500 Scanner (parallel)
# --------------------------------------
if st.sidebar.button("Run S&P 500 Scanner"):
    with st.spinner("Running scanner on S&P 500..."):
        sp500_tickers = get_sp500_tickers()
        results = []
        total = len(sp500_tickers)

        def scan_one(sym):
            try:
                df_scan = get_stock_data(sym, period)
                if df_scan.empty or len(df_scan) < 80:
                    return None
                df_scan_ind = add_technical_indicators(df_scan, n_days)
                df_scan_ind.dropna(inplace=True)

                latest_close = df_scan['Close'].iloc[-1]
                log_returns = np.log(df_scan['Close'] / df_scan['Close'].shift(1)).dropna()
                mu, sigma = log_returns.mean(), log_returns.std()

                pe_ratio = latest_close / eps if eps > 0 else np.nan
                baseline_pe = 20.0
                adjusted_mu = mu * (baseline_pe / pe_ratio) if pe_ratio > 0 else mu

                sim_data = monte_carlo_simulation(S0=latest_close, mu=adjusted_mu, sigma=sigma,
                                                  T=n_days, N=n_days, M=500)
                final_prices = sim_data[-1, :]
                mc_p50 = np.percentile(final_prices, 50)
                mc_change_pct = (mc_p50 - latest_close) / latest_close * 100

                df_tmp = df_scan_ind.copy()
                df_tmp['EPS'] = eps
                df_tmp['Target'] = df_tmp['Close'].shift(-n_days)
                df_tmp.dropna(inplace=True)
                features_scan = [f for f in MODEL_FEATURES if f in df_tmp.columns]
                X = df_tmp[features_scan]
                y = df_tmp['Target']
                if len(X) < 60:
                    return None
                X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

                model_scan = get_model(model_choice)
                model_scan.fit(X_train, y_train)
                pred = float(model_scan.predict(X.iloc[[-1]])[0])
                ml_change_pct = (pred - latest_close) / latest_close * 100

                return {'Ticker': sym, 'ML % Increase': ml_change_pct, 'MC Median % Increase': mc_change_pct}
            except Exception:
                return None

        progress = st.progress(0)
        completed = 0
        with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as ex:
            futures = {ex.submit(scan_one, t): t for t in sp500_tickers}
            for fut in as_completed(futures):
                out = fut.result()
                if out:
                    results.append(out)
                completed += 1
                progress.progress(completed / total)
        progress.progress(1.0)

        if results:
            results_df = pd.DataFrame(results).dropna()
            if not results_df.empty:
                results_df.sort_values(by='ML % Increase', ascending=False, inplace=True)
                st.subheader(f"S&P 500 Scanner Results — Model: {model_choice} (Ranked by ML % Increase)")
                st.dataframe(results_df.style.format({"ML % Increase": "{:.2f}%", "MC Median % Increase": "{:.2f}%"}))
                csv_bytes = results_df.to_csv(index=False).encode()
                st.download_button("Download Scanner CSV", data=csv_bytes, file_name=f"scanner_{now_str()}.csv", mime="text/csv")
            else:
                st.warning("Scanner produced no usable rows.")
        else:
            st.warning("No results to display.")

# --------------------------------------
# Main App Logic
# --------------------------------------
st.title(f"📈 Forecasting Stock Price: {ticker}")

try:
    df = get_stock_data(ticker, period)
    if use_manual_price and manual_price is not None:
        df = df.copy()
        df.loc[df.index[-1], "Close"] = manual_price

    df_ind = add_technical_indicators(df, n_days)
    df_ind.dropna(inplace=True)

    latest_close = df['Close'].iloc[-1]
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    mu, sigma = log_returns.mean(), log_returns.std()

    pe_ratio = latest_close / eps if eps > 0 else np.nan
    baseline_pe = 20.0
    adjusted_mu = mu * (baseline_pe / pe_ratio) if pe_ratio > 0 else mu

    sim_data = monte_carlo_simulation(S0=latest_close, mu=adjusted_mu, sigma=sigma, T=n_days, N=n_days, M=n_simulations)
    final_prices = sim_data[-1, :]
    p5 = np.percentile(final_prices, 5); p50 = np.percentile(final_prices, 50); p95 = np.percentile(final_prices, 95)

    st.subheader("Monte Carlo Simulation Results")
    st.write(f"**5th percentile price**: ${p5:.2f} ({(p5 - latest_close)/latest_close:.2%})")
    st.write(f"**Median price**: ${p50:.2f} ({(p50 - latest_close)/latest_close:.2%})")
    st.write(f"**95th percentile price**: ${p95:.2f} ({(p95 - latest_close)/latest_close:.2%})")

    progress_bar = st.progress(0) if use_bootstrap else None

    model, rmse, predicted_price, actual_price, ci_lower, ci_upper, best_params, y_test, y_pred = train_model(
        df_ind, n_days, eps,
        model_choice=model_choice,
        bootstrap_iters=1000,
        use_gridsearch=use_gridsearch,
        use_bootstrap=use_bootstrap,
        progress_bar=progress_bar
    )

    ml_change_pct = (predicted_price - latest_close) / latest_close * 100

    st.subheader(f"Machine Learning Prediction ({n_days}-Day Close) — Model: {model_choice}")
    st.write(f"**Predicted Price**: ${predicted_price:.2f}")
    if ci_lower is not None and ci_upper is not None:
        st.write(f"**95% Prediction Interval**: ${ci_lower:.2f} to ${ci_upper:.2f}")
    st.write(f"**Actual Price (last test sample)**: ${actual_price:.2f}")
    st.write(f"**Test RMSE**: ${rmse:.2f}")
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

# --------------------------------------
# Backtest on Test Set (diagnostics)
# --------------------------------------
st.sidebar.markdown("---")
run_backtest = st.sidebar.checkbox("Show Test-Set Diagnostics", value=False)

def backtest_model_performance(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    residuals = y_true - y_pred

    st.subheader("📊 Test-Set Diagnostics")
    metrics_df = pd.DataFrame({"Metric": ["RMSE", "MAE", "R² Score"], "Value": [rmse, mae, r2]})
    st.table(metrics_df.style.format({"Value": "{:.4f}"}))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true, mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicted Price'))
    fig.update_layout(title="Actual vs Predicted (Test Window)", xaxis_title="Index", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=residuals, nbinsx=30))
    fig2.update_layout(title="Residuals Histogram", xaxis_title="Residual (Actual - Predicted)", yaxis_title="Frequency")
    st.plotly_chart(fig2, use_container_width=True)

if run_backtest and 'y_test' in locals():
    backtest_model_performance(pd.Series(y_test).reset_index(drop=True), pd.Series(y_pred).reset_index(drop=True))

# --------------------------------------
# ML Strategy Backtesting Section (Pro)
# --------------------------------------
st.markdown("---")
st.subheader("🔄 ML Strategy Backtest (Walk-Forward, Pro)")

if st.button("Run ML Backtest (Pro)"):
    try:
        with st.spinner("Running ML backtest with risk controls..."):
            result = ml_backtest_pro(
                df_raw=df, n_days=n_days, eps=eps, initial_capital=capital, risk_pct=risk_pct,
                rr_mult=rr_ratio, tx_cost_bps=tx_cost_bps, slippage_bps=slippage_bps, atr_mult_sl=atr_mult_sl,
                model_choice=model_choice
            )
        if result is None:
            st.warning("No trades were generated during backtest (or not enough data).")
        else:
            trades_df, stats, equity_curve = result
            st.write("Trade History:")
            st.dataframe(trades_df)

            st.markdown("**Performance Summary**")
            stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
            st.table(stats_df)

            eq = equity_curve.reset_index(drop=True)
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(y=eq, mode='lines', name='Equity'))
            fig_eq.update_layout(title=f"Equity Curve — Model: {model_choice}", xaxis_title="Trade #", yaxis_title="$")
            st.plotly_chart(fig_eq, use_container_width=True)

            dd = eq - eq.cummax()
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(y=dd, mode='lines', name='Drawdown'))
            fig_dd.update_layout(title="Drawdown ($)", xaxis_title="Trade #", yaxis_title="$")
            st.plotly_chart(fig_dd, use_container_width=True)

            csv_bytes = trades_df.to_csv(index=False).encode()
            st.download_button("Download Trades CSV", data=csv_bytes,
                               file_name=f"trades_{ticker}_{now_str()}.csv", mime="text/csv")

            log_path = os.path.join(TRADE_LOG_DIR, f"{ticker}_{now_str()}.csv")
            trades_df.to_csv(log_path, index=False)
            st.info(f"Trade log saved to `{log_path}`")
    except Exception as e:
        st.error(f"Backtest failed: {e}")

