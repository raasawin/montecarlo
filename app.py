import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
import datetime as dt
from pathlib import Path
import traceback

st.set_page_config(layout="wide", page_title="Pro-Grade Stock Forecast & Scanner")

# -----------------------------------------------------------------------------
# 1. PROFESSIONAL DATA & FEATURE ENGINEERING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_sp500_tickers():
    """Download S&P 500 list dynamically."""
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        return tickers
    except:
        return ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "AMD"] # Fallback

def add_pro_features(df):
    """
    Generates stationary, professional-grade features.
    No raw prices are used as inputs to the ML model.
    """
    data = df.copy()
    
    # Target: Log Returns (Stationary)
    data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # 1. Volatility (Annualized)
    data['Vol_20'] = data['Log_Ret'].rolling(20).std() * np.sqrt(252)
    
    # 2. Distance from Moving Averages (Normalized)
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['Dist_SMA_50'] = (data['Close'] / data['SMA_50']) - 1
    
    # 3. RSI (Normalized to 0-1)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI_Norm'] = data['RSI'] / 100.0
    
    # 4. Relative Volume
    data['Vol_Rel'] = data['Volume'] / data['Volume'].rolling(20).mean()
    
    # 5. Lagged Returns (Autoregression)
    data['Ret_Lag1'] = data['Log_Ret'].shift(1)
    data['Ret_Lag2'] = data['Log_Ret'].shift(2)
    data['Ret_Lag5'] = data['Log_Ret'].shift(5)
    
    # ATR for Risk Management (Absolute value, needed for sizing, not ML)
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift(1)).abs()
    low_close = (data['Low'] - data['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR_14'] = tr.rolling(14).mean()

    return data.dropna()

@st.cache_data(ttl=3600)
def get_stock_data(ticker, period="2y"):
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period)
        if len(df) < 200: return None
        return add_pro_features(df)
    except:
        return None

# -----------------------------------------------------------------------------
# 2. MATH ENGINES (MC & ML)
# -----------------------------------------------------------------------------
def monte_carlo_bootstrap(df, n_sims, n_days, current_price):
    """Bootstraps from historical log returns (Fat Tail aware)."""
    hist_returns = df['Log_Ret'].values
    if len(hist_returns) < 50: return None
    
    # Randomly sample historical returns
    random_returns = np.random.choice(hist_returns, size=(n_days, n_sims))
    log_paths = np.cumsum(random_returns, axis=0)
    price_paths = current_price * np.exp(log_paths)
    return price_paths

def train_xgb_model(df, forecast_days):
    """
    Trains XGBoost using Walk-Forward logic.
    Predicts Cumulative Log Return for the forecast horizon.
    """
    data = df.copy()
    # Target: Future Cumulative Log Return
    data['Target'] = data['Log_Ret'].rolling(forecast_days).sum().shift(-forecast_days)
    data = data.dropna()
    
    if len(data) < 100: return None, None, None, None
    
    feature_cols = ['Vol_20', 'Dist_SMA_50', 'RSI_Norm', 'Vol_Rel', 'Ret_Lag1', 'Ret_Lag2', 'Ret_Lag5']
    X = data[feature_cols]
    y = data['Target']
    
    # Strict Walk-Forward Split (No shuffling)
    split_idx = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, n_jobs=1, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Current Prediction
    last_row = df[feature_cols].iloc[[-1]]
    last_row_scaled = scaler.transform(last_row)
    pred_log_ret = model.predict(last_row_scaled)[0]
    
    # Convert log return to %
    pred_pct = np.exp(pred_log_ret) - 1
    
    # Evaluate RMSE on Test Set (in % terms)
    preds_test = model.predict(X_test_scaled)
    rmse_log = np.sqrt(mean_squared_error(y_test, preds_test))
    
    return model, pred_pct, rmse_log, feature_cols

# -----------------------------------------------------------------------------
# 3. SCANNER LOGIC
# -----------------------------------------------------------------------------
def scan_ticker(ticker, n_days):
    """Analyzes a single ticker for the scanner."""
    try:
        df = get_stock_data(ticker, period="2y")
        if df is None: return None
        
        current_price = df['Close'].iloc[-1]
        
        # 1. Run ML
        model, ml_pred_pct, rmse, _ = train_xgb_model(df, n_days)
        if model is None: return None
        
        # 2. Run MC (Lighter version for speed)
        paths = monte_carlo_bootstrap(df, n_sims=200, n_days=n_days, current_price=current_price)
        p50 = np.percentile(paths[-1, :], 50)
        mc_pred_pct = (p50 / current_price) - 1
        
        # 3. Score: Confluence of ML and MC
        # High score if both agree on direction and magnitude
        score = (ml_pred_pct + mc_pred_pct) / 2
        
        return {
            "Ticker": ticker,
            "Price": current_price,
            "ML Forecast %": ml_pred_pct * 100,
            "MC Forecast %": mc_pred_pct * 100,
            "RMSE (Uncertainty)": rmse,
            "Score": score
        }
    except:
        return None

# -----------------------------------------------------------------------------
# 4. UI & MAIN APP
# -----------------------------------------------------------------------------
st.sidebar.title("Settings")
mode = st.sidebar.radio("Mode", ["Single Stock Analysis", "Market Scanner"])

if mode == "Single Stock Analysis":
    ticker = st.sidebar.text_input("Ticker", "NVDA").upper()
    forecast_days = st.sidebar.slider("Forecast Days", 5, 60, 20)
    sim_count = st.sidebar.slider("MC Simulations", 500, 5000, 1000)
    
    # Position Sizing Inputs
    st.sidebar.markdown("---")
    st.sidebar.subheader("Risk Management")
    account_size = st.sidebar.number_input("Account Balance ($)", 10000.0)
    risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 1.0)
    atr_mult = st.sidebar.number_input("Stop Loss (ATR Multiple)", 1.5)

    df = get_stock_data(ticker)
    
    if df is not None:
        current_price = df['Close'].iloc[-1]
        st.title(f"Analysis: {ticker} (${current_price:.2f})")
        
        # --- 1. ML Analysis ---
        model, ml_pred, rmse, feats = train_xgb_model(df, forecast_days)
        target_price_ml = current_price * (1 + ml_pred)
        
        # --- 2. MC Analysis ---
        mc_paths = monte_carlo_bootstrap(df, sim_count, forecast_days, current_price)
        p05 = np.percentile(mc_paths[-1, :], 5)
        p50 = np.percentile(mc_paths[-1, :], 50)
        p95 = np.percentile(mc_paths[-1, :], 95)
        mc_pred = (p50 / current_price) - 1
        
        # Display Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ML Forecast", f"{ml_pred*100:.2f}%", f"${target_price_ml:.2f}")
        col2.metric("MC Median", f"{mc_pred*100:.2f}%", f"${p50:.2f}")
        col3.metric("MC Bull (95%)", f"${p95:.2f}")
        col4.metric("MC Bear (5%)", f"${p05:.2f}")
        
        # Charts
        tab1, tab2 = st.tabs(["Monte Carlo Cloud", "Feature Importance"])
        
        with tab1:
            fig_mc = go.Figure()
            # Downsample paths for rendering speed
            subset = mc_paths[:, :100]
            for i in range(subset.shape[1]):
                fig_mc.add_trace(go.Scatter(y=subset[:, i], mode='lines', line=dict(color='gray', width=0.5), opacity=0.3, showlegend=False))
            fig_mc.add_trace(go.Scatter(x=[forecast_days-1], y=[target_price_ml], mode='markers', marker=dict(color='red', size=15, symbol='star'), name='ML Target'))
            fig_mc.update_layout(title="Monte Carlo Paths vs ML Target (Red Star)", height=500)
            st.plotly_chart(fig_mc, use_container_width=True)
            
        with tab2:
            importance = pd.DataFrame({'Feature': feats, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
            st.bar_chart(importance.set_index('Feature'))

        # --- 3. Position Sizing & Trade Log ---
        st.markdown("---")
        st.subheader("🛠️ Trade Setup & Logger")
        
        atr = df['ATR_14'].iloc[-1]
        stop_dist = atr * atr_mult
        
        # Determine direction based on ML
        direction = "LONG" if ml_pred > 0 else "SHORT"
        
        if direction == "LONG":
            stop_price = current_price - stop_dist
            risk_per_share = current_price - stop_price
        else:
            stop_price = current_price + stop_dist
            risk_per_share = stop_price - current_price
            
        risk_amount = account_size * (risk_pct / 100)
        shares = int(risk_amount // risk_per_share) if risk_per_share > 0 else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.info(f"**Direction:** {direction}")
        c2.info(f"**Entry:** ${current_price:.2f}")
        c3.info(f"**Stop Loss:** ${stop_price:.2f}")
        c4.success(f"**Size:** {shares} shares")
        
        # Logging
        if st.button("📝 Log Trade to CSV"):
            log_entry = {
                "Date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Ticker": ticker,
                "Direction": direction,
                "Entry": current_price,
                "Stop": stop_price,
                "Target_ML": target_price_ml,
                "Shares": shares,
                "ML_Conf": ml_pred,
                "RMSE": rmse
            }
            log_path = Path("trade_log.csv")
            log_df = pd.DataFrame([log_entry])
            log_df.to_csv(log_path, mode='a', header=not log_path.exists(), index=False)
            st.success("Trade saved to trade_log.csv")

    else:
        st.error("Could not fetch data. Check ticker.")

elif mode == "Market Scanner":
    st.title("🚀 Pro-Grade S&P 500 Scanner")
    st.markdown("Scans the S&P 500 using the XGBoost + Monte Carlo engine. This searches for **Confluence** (where Math and History agree).")
    
    scan_days = st.slider("Forecast Horizon for Scan", 10, 60, 20)
    
    if st.button("Start Scan (This takes time)"):
        tickers = get_sp500_tickers()
        # For demo speed, limit to first 50, remove [:50] for full scan
        tickers_to_scan = tickers[:50] 
        
        results = []
        progress = st.progress(0)
        status = st.empty()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(scan_ticker, t, scan_days): t for t in tickers_to_scan}
            completed = 0
            
            for future in as_completed(futures):
                res = future.result()
                if res:
                    results.append(res)
                completed += 1
                progress.progress(completed / len(tickers_to_scan))
                status.text(f"Scanned {completed}/{len(tickers_to_scan)} tickers...")
                
        if results:
            df_res = pd.DataFrame(results)
            # Filter for confluence: ML and MC both positive or both negative
            df_res['Confluence'] = np.sign(df_res['ML Forecast %']) == np.sign(df_res['MC Forecast %'])
            
            st.success("Scan Complete!")
            
            st.subheader("🏆 Top Bullish Opportunities")
            bulls = df_res[(df_res['ML Forecast %'] > 0) & (df_res['Confluence'])].sort_values('Score', ascending=False).head(10)
            st.dataframe(bulls.style.format({"Price": "${:.2f}", "ML Forecast %": "{:.2f}%", "MC Forecast %": "{:.2f}%", "Score": "{:.4f}"}))
            
            st.subheader("🐻 Top Bearish Opportunities")
            bears = df_res[(df_res['ML Forecast %'] < 0) & (df_res['Confluence'])].sort_values('Score', ascending=True).head(10)
            st.dataframe(bears.style.format({"Price": "${:.2f}", "ML Forecast %": "{:.2f}%", "MC Forecast %": "{:.2f}%", "Score": "{:.4f}"}))
        else:
            st.warning("No results found.")
