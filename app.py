

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample
import plotly.graph_objects as go
from pathlib import Path
import datetime as dt
from xgboost import XGBRegressor

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
# Load S&P 500 tickers from CSV
# --------------------------------------
@st.cache_data(ttl=86400)
def get_sp500_tickers_from_csv(csv_like) -> list[str]:
    if isinstance(csv_like, pd.DataFrame):
        df = csv_like.copy()
    else:
        df = pd.read_csv(csv_like)
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False).str.strip()
    return df['Symbol'].dropna().unique().tolist()

def load_sp500_list(sp500_choice, uploaded_sp500, local_sp500_path):
    if sp500_choice == "Upload CSV":
        if uploaded_sp500 is None:
            st.warning("Upload a CSV to run the scanner.")
            return []
        return get_sp500_tickers_from_csv(uploaded_sp500)
    else:
        try:
            return get_sp500_tickers_from_csv(local_sp500_path)
        except Exception as e:
            st.error(f"Failed to read S&P 500 CSV at '{local_sp500_path}': {e}")
            return []

# --------------------------------------
# Technical indicators
# --------------------------------------
def add_technical_indicators(df):
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Momentum'] = df['Close'].diff(4)
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI_14'] = compute_rsi(df['Close'], 14)
    df['MACD'] = compute_macd(df['Close'])
    df['MACD_Signal'] = compute_macd_signal(df['Close'])
    # --- ATR(14)
    high = df['High']; low = df['Low']; close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()
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
def train_ml_model(df, n_days_ahead, eps, model_choice="RandomForest",
                   bootstrap_iters=1000, use_gridsearch=False, use_bootstrap=False, progress_bar=None):
    df['EPS'] = eps
    df['Target'] = df['Close'].shift(-n_days_ahead)
    df = df.dropna()

    features = ['Close', 'SMA_20', 'Momentum', 'Volatility', 'Volume_Change', 'EPS',
                'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal']
    X = df[features]; y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    if len(X_train) < 20:
        raise ValueError("Not enough valid data to train the model. Try selecting a longer period.")

    if model_choice == "RandomForest":
        if use_gridsearch:
            param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]}
            tscv = TimeSeriesSplit(n_splits=5)
            model = GridSearchCV(RandomForestRegressor(random_state=0),
                                 param_grid, cv=tscv, scoring='neg_mean_squared_error')
            model.fit(X_train, y_train)
            best_model = model.best_estimator_
            best_params = model.best_params_
        else:
            best_model = RandomForestRegressor(n_estimators=100, random_state=0)
            st.write("Training Random Forest...")
            best_model.fit(X_train, y_train)
            st.write("Training complete!")
            best_params = {'n_estimators': 100, 'max_depth': None}
    else:  # XGBoost
        n_estimators = 50 if len(X_train) < 200 else 300
        best_model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=0,
            objective="reg:squarederror",
            verbosity=0,
            n_jobs=1  # force single-thread
        )
        st.write("Training XGBoost, please wait...")
        best_model.fit(X_train, y_train)
        st.write("Training complete!")
        best_params = best_model.get_params()
    
    
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    latest_features = X.iloc[[-1]]
    predicted_price = best_model.predict(latest_features)[0]

    ci_lower, ci_upper = None, None
    if use_bootstrap and progress_bar:
        boot_preds = []
        for i in range(bootstrap_iters):
            X_res, y_res = resample(X_train, y_train)
            rf = RandomForestRegressor(**best_model.get_params())
            rf.fit(X_res, y_res)
            boot_preds.append(rf.predict(latest_features)[0])
            if i % max(1, bootstrap_iters // 100) == 0:
                progress_bar.progress(min(i / bootstrap_iters, 1.0))
        ci_lower = np.percentile(boot_preds, 2.5)
        ci_upper = np.percentile(boot_preds, 97.5)

    return best_model, rmse, predicted_price, y_test.values[-1], ci_lower, ci_upper, best_params

# --------------------------------------
# Trade logging helper
# --------------------------------------
def log_trade(row: dict, folder="./trade_logs", filename="trades.csv"):
    Path(folder).mkdir(parents=True, exist_ok=True)
    path = Path(folder) / filename
    df_row = pd.DataFrame([row])
    if path.exists():
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, index=False)
    return str(path)

# --------------------------------------
# Sidebar Inputs
# --------------------------------------
st.sidebar.title("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
period = st.sidebar.selectbox("Historical Data Period", ["6mo", "1y", "2y", "5y"], index=1)
n_simulations = st.sidebar.slider("Monte Carlo Simulations", 100, 10000, 500, step=100)
n_days = st.sidebar.slider("Days into the Future", 10, 180, 30, step=10)

model_choice = st.sidebar.selectbox("ML Model", ["RandomForest", "XGBoost"], index=0)
use_gridsearch = st.sidebar.checkbox("Use GridSearchCV (slower, better tuning)", value=False)
use_bootstrap = st.sidebar.checkbox("Use Bootstrapping for CI (slower)", value=False)
use_manual_price = st.sidebar.checkbox("Use Manual Close Price")
manual_price = st.sidebar.number_input("Enter Latest Close Price", min_value=0.0, value=150.0, step=0.1) if use_manual_price else None
eps = st.sidebar.number_input("Enter EPS (Earnings Per Share)", min_value=0.01, value=5.0, step=0.01)

# --- Position sizing
st.sidebar.markdown("### Position Sizing")
account_size = st.sidebar.number_input("Account size ($)", value=10000.0, min_value=0.0, step=100.0)
risk_pct = st.sidebar.slider("Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
atr_mult = st.sidebar.number_input("Stop distance (ATR ×)", min_value=0.1, value=1.5, step=0.1)
tp_rr = st.sidebar.number_input("Take-profit (R multiple)", min_value=0.5, value=2.0, step=0.1)

# --- S&P 500 source
st.sidebar.markdown("### S&P 500 Source")
sp500_choice = st.sidebar.selectbox("Ticker list source", ["Upload CSV", "Local file path"], index=0)
uploaded_sp500 = st.sidebar.file_uploader("Upload a CSV with 'Symbol'", type=["csv"]) if sp500_choice == "Upload CSV" else None
local_sp500_path = st.sidebar.text_input("Local CSV path", "./data/sp500.csv") if sp500_choice == "Local file path" else None

rank_mode = st.sidebar.selectbox("Scanner ranking mode",
    ["ML % Increase", "MC Median % Increase", "Blend (avg of both)"], index=0)

# ---------------------------
# Main App Logic
# ---------------------------
st.title(f"📈 Forecasting Stock Price: {ticker.upper()}")

model = None  # <<< Initialize model to prevent backtest errors

try:
    df = get_stock_data(ticker, period)
    df = add_technical_indicators(df)
    if use_manual_price and manual_price is not None:
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
    st.write(f"**5th percentile price**: ${p5:.2f} ({(p5 - latest_close)/latest_close:.2%})")
    st.write(f"**Median price**: ${p50:.2f} ({(p50 - latest_close)/latest_close:.2%})")
    st.write(f"**95th percentile price**: ${p95:.2f} ({(p95 - latest_close)/latest_close:.2%})")

    progress_bar = st.progress(0)
    model, rmse, predicted_price, actual_price, ci_lower, ci_upper, best_params = train_ml_model(
        df, n_days, eps, model_choice=model_choice,
        bootstrap_iters=1000, use_gridsearch=use_gridsearch, use_bootstrap=use_bootstrap,
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
        trade_side = "LONG"
    elif ml_change_pct < -1 and p50 < latest_close:
        st.error(f"**Likely Downward Trend** — ML: ~{ml_change_pct:.2f}%, MC: {(p50 - latest_close)/latest_close:.2%}")
        trade_side = "SHORT"
    else:
        st.warning("**Uncertain** — Mixed or flat predictions. Use caution.")
        trade_side = "FLAT"

    # --- Position sizing
    atr = float(df['ATR_14'].iloc[-1]) if 'ATR_14' in df.columns else np.nan
    entry = float(latest_close)
    if np.isnan(atr) or atr == 0 or trade_side == "FLAT":
        sl, tp, shares, risk_dollars = np.nan, np.nan, 0, 0.0
    else:
        stop_dist = atr_mult * atr
        if trade_side == "LONG":
            sl = entry - stop_dist
            risk_per_share = entry - sl
            tp = entry + tp_rr * risk_per_share
        else:  # SHORT
            sl = entry + stop_dist
            risk_per_share = sl - entry
            tp = entry - tp_rr * risk_per_share
        risk_dollars = account_size * (risk_pct / 100.0)
        shares = int(risk_dollars // risk_per_share) if risk_per_share > 0 else 0

    st.markdown("### 🧮 Position Sizing")
    st.write(f"**Side**: {trade_side} | **Entry**: ${entry:,.2f} | "
             f"**Stop**: {('—' if np.isnan(sl) else f'${sl:,.2f}')} | "
             f"**Take Profit**: {('—' if np.isnan(tp) else f'${tp:,.2f}')} | "
             f"**Shares**: {shares}")
    st.write(f"**Account Risk**: ${risk_dollars:,.2f}")

    # --- Trade logging button
    if trade_side in ("LONG", "SHORT") and shares > 0 and not np.isnan(sl) and not np.isnan(tp):
        if st.button("📝 Log Trade"):
            trade = {
                "timestamp": dt.datetime.utcnow().isoformat(),
                "ticker": ticker.upper(),
                "side": trade_side,
                "entry": entry,
                "stop": sl,
                "take_profit": tp,
                "shares": shares,
                "account_size": account_size,
                "risk_pct": risk_pct,
                "atr_mult": atr_mult,
                "tp_rr": tp_rr,
                "ml_change_pct": ml_change_pct,
                "mc_median_change_pct": (p50 - latest_close) / latest_close * 100,
                "predicted_price": predicted_price,
                "rmse": rmse,
                "n_days": n_days,
                "use_gridsearch": use_gridsearch,
                "use_bootstrap": use_bootstrap
            }
            path = log_trade(trade)
            st.success(f"Trade logged to {path}")

except Exception as e:
    st.error(f"Error loading data or running simulation: {e}")
    model = None  # Ensure variable exists for backtest

# --------------------------------------
# Backtest block
# --------------------------------------
run_backtest = st.sidebar.checkbox("Run Backtest on Test Set", value=False)

def backtest_model_performance(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    residuals = y_true - y_pred

    st.subheader("📊 Backtest Results")

    metrics_df = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "R² Score"],
        "Value": [rmse, mae, r2]
    })
    st.dataframe(metrics_df.style.format({"Value": "{:.4f}"}))

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

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=residuals, nbinsx=30))
    fig2.update_layout(
        title="Prediction Residuals Histogram",
        xaxis_title="Residual (Actual - Predicted)",
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig2, use_container_width=True)