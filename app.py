import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample
import plotly.graph_objects as go
from pathlib import Path
import datetime as dt
from xgboost import XGBRegressor
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

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
    # ATR(14)
    high = df['High']; low = df['Low']; close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()
    return df

@st.cache_data(ttl=3600)
def prepare_stock_df(ticker, period, manual_price=None):
    df = get_stock_data(ticker, period)
    if manual_price is not None:
        df.iloc[-1, df.columns.get_loc("Close")] = manual_price
    df = add_technical_indicators(df)
    df.dropna(inplace=True)
    return df

# --------------------------------------
# Monte Carlo Simulation (vectorized)
# --------------------------------------
def monte_carlo_simulation(S0, mu, sigma, T, N, M):
    """
    Returns an array of shape (N, M) where row 0 ~ S0, and row N-1 is final price.
    """
    dt = T / N
    # Generate (N-1) steps, then prepend a row of zeros for S0
    rand = np.random.normal(size=(N - 1, M))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand
    log_paths = np.cumsum(increments, axis=0)
    log_paths = np.vstack([np.zeros((1, M)), log_paths])  # include t0
    paths = S0 * np.exp(log_paths)
    return paths

# --------------------------------------
# ML Model
# --------------------------------------
def train_ml_model(df, n_days_ahead, eps, model_choice="RandomForest",
                   bootstrap_iters=1000, use_gridsearch=False, use_bootstrap=False, progress_bar=None):
    df = df.copy()
    df['EPS'] = eps
    df['Target'] = df['Close'].shift(-n_days_ahead)

    # --- Drop any rows with NaNs in features or target
    features = ['Close', 'SMA_20', 'Momentum', 'Volatility', 'Volume_Change', 'EPS',
                'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal']
    df = df.dropna(subset=features + ['Target'])

    if len(df) < 20:
        raise ValueError("Not enough valid data to train the model. Try selecting a longer period.")

    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    if len(X_train) < 20:
        raise ValueError("Not enough valid training data after splitting. Try selecting a longer period.")

    # --- Model selection
    if model_choice == "RandomForest":
        best_model = RandomForestRegressor(n_estimators=100, random_state=0)
        best_model.fit(X_train, y_train)
        best_params = {'n_estimators': 100, 'max_depth': None}
    else:
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
            n_jobs=1
        )
        best_model.fit(X_train, y_train)
        best_params = best_model.get_params()

    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    latest_features = X.iloc[[-1]]
    predicted_price = best_model.predict(latest_features)[0]

    # --- Bootstrap CI
    ci_lower, ci_upper = None, None
    if use_bootstrap and progress_bar is not None:
        boot_preds = []
        for i in range(bootstrap_iters):
            X_res, y_res = resample(X_train, y_train)
            rf = RandomForestRegressor(**best_model.get_params()) if model_choice == "RandomForest" else XGBRegressor(**best_model.get_params())
            rf.fit(X_res, y_res)
            boot_preds.append(rf.predict(latest_features)[0])
            if i % max(1, bootstrap_iters // 100) == 0:
                progress_bar.progress(min((i + 1) / bootstrap_iters, 1.0))
        ci_lower = np.percentile(boot_preds, 2.5)
        ci_upper = np.percentile(boot_preds, 97.5)

    return best_model, rmse, predicted_price, y_test.values, ci_lower, ci_upper, best_params, X_test

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
use_gridsearch = st.sidebar.checkbox("Use GridSearchCV (slower, better tuning)", value=False)  # kept for compatibility
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

run_backtest = st.sidebar.checkbox("Run Backtest on Test Set", value=False)

# ---------------------------
# Main App Logic
# ---------------------------
st.title(f"📈 Forecasting Stock Price: {ticker.upper()}")

model = None
X_test = None
y_test_values = None
trade_side = "FLAT"
shares = 0
sl = np.nan
tp = np.nan
risk_dollars = 0.0
ml_change_pct = np.nan
p50 = np.nan
latest_close = np.nan
predicted_price = np.nan
rmse = np.nan

try:
    # Data + indicators (cached)
    df = prepare_stock_df(ticker, period, manual_price if use_manual_price else None)
    latest_close = float(df['Close'].iloc[-1])

    # Monte Carlo setup (vectorized)
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    mu, sigma = log_returns.mean(), log_returns.std()

    # Optional P/E adjustment (preserved)
    pe_ratio = latest_close / eps if eps > 0 else np.nan
    baseline_pe = 20.0
    adjusted_mu = mu * (baseline_pe / pe_ratio) if pe_ratio and pe_ratio > 0 else mu

    sim_data = monte_carlo_simulation(latest_close, adjusted_mu, sigma, n_days, n_days, n_simulations)
    final_prices = sim_data[-1, :]
    p5, p50, p95 = np.percentile(final_prices, [5, 50, 95])

    st.subheader("Monte Carlo Simulation Results")
    st.write(f"**5th percentile price**: ${p5:.2f} ({(p5 - latest_close)/latest_close:.2%})")
    st.write(f"**Median price**: ${p50:.2f} ({(p50 - latest_close)/latest_close:.2%})")
    st.write(f"**95th percentile price**: ${p95:.2f} ({(p95 - latest_close)/latest_close:.2%})")

    # --- Train ML Model
    progress_bar = st.progress(0)
    model, rmse, predicted_price, y_test_values, ci_lower, ci_upper, best_params, X_test = train_ml_model(
        df, n_days, eps, model_choice=model_choice,
        bootstrap_iters=1000, use_gridsearch=use_gridsearch, use_bootstrap=use_bootstrap,
        progress_bar=progress_bar
    )
    progress_bar.empty()

    actual_price = y_test_values[-1] if len(y_test_values) else np.nan
    ml_change_pct = (predicted_price - latest_close) / latest_close * 100 if latest_close else np.nan

    st.subheader(f"Machine Learning Prediction ({n_days}-Day Close)")
    st.write(f"**Predicted Price**: ${predicted_price:.2f}")
    if ci_lower is not None and ci_upper is not None:
        st.write(f"**95% Prediction Interval**: ${ci_lower:.2f} to ${ci_upper:.2f}")
    st.write(f"**Actual Price (last test sample)**: ${actual_price:.2f}")
    st.write(f"**RMSE**: ${rmse:.2f}")
    st.write(f"**Expected Price Change**: {ml_change_pct:+.2f}%")
    st.write(f"**Best Model Parameters**: `{best_params}`")

    # --- Summary Trend
    st.markdown("---")
    st.subheader("📊 Final Summary")
    if (ml_change_pct > 1) and (p50 > latest_close):
        st.success(f"**Likely Upward Trend** — ML: ~{ml_change_pct:.2f}%, MC: {(p50 - latest_close)/latest_close:.2%}")
        trade_side = "LONG"
    elif (ml_change_pct < -1) and (p50 < latest_close):
        st.error(f"**Likely Downward Trend** — ML: ~{ml_change_pct:.2f}%, MC: {(p50 - latest_close)/latest_close:.2%}")
        trade_side = "SHORT"
    else:
        st.warning("**Uncertain** — Mixed or flat predictions. Use caution.")
        trade_side = "FLAT"

    # --- Position Sizing
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

except Exception as e:
    st.error(f"Error loading data or running simulation: {e}")
    st.caption(traceback.format_exc())

# --- Trade logging button (moved outside try/except; preserved functionality)
if 'trade_side' in locals() and trade_side in ("LONG", "SHORT") and shares > 0 and not np.isnan(sl) and not np.isnan(tp):
    if st.button("📝 Log Trade"):
        try:
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
                "mc_median_change_pct": (p50 - latest_close) / latest_close * 100 if latest_close else np.nan,
                "predicted_price": predicted_price,
                "rmse": rmse,
                "n_days": n_days,
                "use_gridsearch": use_gridsearch,
                "use_bootstrap": use_bootstrap
            }
            path = log_trade(trade)
            st.success(f"Trade logged to {path}")
        except Exception as e:
            st.error(f"Failed to log trade: {e}")

# ---------------------------
# Backtest block
# ---------------------------
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

# -------- Account Growth Simulation (new) --------
def simulate_account_growth(df_full, X_test, y_test, model,
                            starting_equity, risk_pct, atr_mult, tp_rr):
    """
    Simulates account balance across the test set.
    Entry price: current close at the index of each test sample.
    Exit price: actual target (n-days-ahead close) from y_test.
    Sizing: risk $ = balance * risk_pct%; risk per share = ATR*atr_mult.
    Direction: LONG if model predict > entry else SHORT.
    """
    if X_test is None or len(X_test) == 0:
        return [starting_equity]

    equity = starting_equity
    curve = [equity]

    # Ensure we align with df rows by index
    preds = model.predict(X_test)
    for i, idx in enumerate(X_test.index):
        try:
            entry = float(df_full.loc[idx, 'Close'])
            atr = float(df_full.loc[idx, 'ATR_14']) if 'ATR_14' in df_full.columns else np.nan
            if np.isnan(atr) or atr == 0:
                curve.append(equity)
                continue

            stop_dist = atr_mult * atr
            risk_per_share = stop_dist
            risk_dollars = equity * (risk_pct / 100.0)
            shares = int(risk_dollars // risk_per_share) if risk_per_share > 0 else 0
            if shares <= 0:
                curve.append(equity)
                continue

            pred = preds[i]
            actual_exit = y_test[i]
            side = "LONG" if pred > entry else "SHORT"

            if side == "LONG":
                pnl = (actual_exit - entry) * shares
            else:
                pnl = (entry - actual_exit) * shares

            equity += pnl
            curve.append(equity)
        except Exception:
            curve.append(equity)
            continue

    return curve

# ---------------------------
# Backtest Execution
# ---------------------------
if run_backtest and (model is not None) and (X_test is not None) and (y_test_values is not None):
    if len(y_test_values) > 0:
        y_pred_test = model.predict(X_test)
        backtest_model_performance(y_test_values, y_pred_test)

        st.subheader("💰 Account Growth Simulation")
        equity_curve = simulate_account_growth(df, X_test, y_test_values, model,
                                               account_size, risk_pct, atr_mult, tp_rr)
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(y=equity_curve, mode='lines', name='Equity'))
        fig_eq.update_layout(
            title="Account Balance Over Test Set",
            xaxis_title="Trade Index",
            yaxis_title="Account Balance ($)"
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100 if len(equity_curve) > 1 else 0.0
        st.write(f"**Simulated Total Return (Test Period)**: {total_return:.2f}%")
    else:
        st.warning("Not enough data for backtest. Try a longer period or different ticker.")

# ---------------------------
# Parallel S&P 500 Scanner (optimized)
# ---------------------------
def scan_single_ticker(t, period, n_days, eps, model_choice, rank_mode):
    try:
        df_t = prepare_stock_df(t, period)
        df_t['EPS'] = eps
        latest_close_t = float(df_t['Close'].iloc[-1])

        # Monte Carlo (vectorized)
        log_returns_t = np.log(df_t['Close'] / df_t['Close'].shift(1)).dropna()
        mu_t, sigma_t = log_returns_t.mean(), log_returns_t.std()
        sim_t = monte_carlo_simulation(latest_close_t, mu_t, sigma_t, n_days, n_days, 100)
        p50_t = np.percentile(sim_t[-1, :], 50)
        mc_change_pct = (p50_t - latest_close_t) / latest_close_t * 100

        # ML
        try:
            model_t, _, predicted_price_t, _, _, _, _, _ = train_ml_model(
                df_t, n_days, eps, model_choice=model_choice,
                use_gridsearch=False, use_bootstrap=False, progress_bar=None
            )
            ml_change_pct = (predicted_price_t - latest_close_t) / latest_close_t * 100
        except Exception:
            ml_change_pct = np.nan

        # Rank score
        if rank_mode == "ML % Increase":
            score = ml_change_pct
        elif rank_mode == "MC Median % Increase":
            score = mc_change_pct
        else:
            score = np.nanmean([ml_change_pct, mc_change_pct])

        return {
            "Ticker": t,
            "Latest Close": latest_close_t,
            "ML % Change": ml_change_pct,
            "MC % Change": mc_change_pct,
            "Score": score
        }
    except Exception:
        return None

def run_sp500_scanner(tickers, period, n_days, eps, model_choice, rank_mode):
    results = []
    progress = st.progress(0.0)
    max_workers = min(16, max(1, len(tickers)))
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scan_single_ticker, t, period, n_days, eps, model_choice, rank_mode): t for t in tickers}
        for f in as_completed(futures):
            res = f.result()
            if res:
                results.append(res)
            completed += 1
            progress.progress(min(completed / len(tickers), 1.0))
    progress.empty()
    if results:
        return pd.DataFrame(results).sort_values("Score", ascending=False)
    return pd.DataFrame(columns=["Ticker", "Latest Close", "ML % Change", "MC % Change", "Score"])

if st.sidebar.button("🚀 Run S&P 500 Scanner"):
    tickers = load_sp500_list(sp500_choice, uploaded_sp500, local_sp500_path)
    if not tickers:
        st.warning("No tickers found to scan.")
    else:
        st.info(f"Scanning {len(tickers)} tickers in parallel... This may still take a bit.")
        df_results = run_sp500_scanner(tickers, period, n_days, eps, model_choice, rank_mode)
        if not df_results.empty:
            st.subheader("S&P 500 Scan Results")
            st.dataframe(df_results)

            # Optional quick viz of top 10 by score
            top10 = df_results.head(10)
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=top10['Ticker'], y=top10['Score'], name='Score'))
            fig_bar.update_layout(title="Top 10 by Score", xaxis_title="Ticker", yaxis_title="Score")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("No valid results from scan.")