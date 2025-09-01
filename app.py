import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.utils import resample
from xgboost import XGBRegressor
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------------------------
# Utility functions
# --------------------------------------
def get_stock_data(ticker, period="1y"):
    df = yf.download(ticker, period=period, progress=False)
    df.dropna(inplace=True)
    return df

def add_technical_indicators(df):
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["Momentum"] = df["Close"].diff(4)
    df["Volatility"] = df["Close"].rolling(window=20).std()
    df["Volume_Change"] = df["Volume"].pct_change()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df

def monte_carlo_simulation(last_price, mu, sigma, T, n_steps, n_simulations):
    dt = T / n_steps
    prices = np.zeros((n_steps, n_simulations))
    prices[0] = last_price
    for t in range(1, n_steps):
        rand = np.random.standard_normal(n_simulations)
        prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand)
    return prices

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
            n_jobs=1
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

    return (
        best_model,
        rmse,
        predicted_price,
        y_test.values[-1],
        ci_lower,
        ci_upper,
        best_params,
        y_test.values.ravel(),   # ensure 1D
        np.array(y_pred).ravel() # ensure 1D
    )

# --------------------------------------
# Backtest function
# --------------------------------------
def backtest_model_performance(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_true, label="Actual")
    ax.plot(y_pred, label="Predicted")
    ax.set_title("Backtest: Actual vs Predicted")
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual Scatter")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    residuals = y_true - y_pred
    ax.hist(residuals, bins=30, alpha=0.7)
    ax.set_title("Residuals Distribution")
    st.pyplot(fig)

# --------------------------------------
# Sidebar inputs
# --------------------------------------
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", "AAPL")
n_days = st.sidebar.slider("Days ahead", 1, 30, 5)
n_simulations = st.sidebar.slider("Monte Carlo Simulations", 100, 2000, 500, step=100)
eps = st.sidebar.number_input("Earnings per Share (EPS)", value=1.0)
model_choice = st.sidebar.radio("ML Model", ["RandomForest", "XGBoost"])
use_gridsearch = st.sidebar.checkbox("Use GridSearchCV (RandomForest only)", value=False)
use_bootstrap = st.sidebar.checkbox("Bootstrap Confidence Interval", value=False)
rank_mode = st.sidebar.radio("Ranking Mode", ["ML % Increase", "MC Median % Increase", "Blend"])
sp500_choice = st.sidebar.radio("S&P 500 Source", ["Local CSV", "Upload CSV"])
uploaded_sp500 = st.sidebar.file_uploader("Upload S&P 500 CSV", type=["csv"])
local_sp500_path = "sp500.csv"
run_backtest = st.sidebar.checkbox("Run Backtest on Test Set", value=False)
scan_sp500 = st.sidebar.button("🔍 Scan S&P 500")

# --------------------------------------
# Main App
# --------------------------------------
st.title(f"📈 Forecasting Stock Price: {ticker.upper()}")

model = None
y_true_all, y_pred_all = None, None

try:
    df = get_stock_data(ticker, "1y")
    df = add_technical_indicators(df)
    df.dropna(inplace=True)

    latest_close = df['Close'].iloc[-1]
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    mu, sigma = log_returns.mean(), log_returns.std()

    sim = monte_carlo_simulation(latest_close, mu, sigma, n_days, n_days, n_simulations)
    median_price = np.median(sim[-1, :])

    st.subheader("Monte Carlo Simulation")
    fig, ax = plt.subplots()
    ax.plot(sim[:, :10])
    st.pyplot(fig)

    progress_bar = st.progress(0)
    model, rmse, predicted_price, actual_price, ci_lower, ci_upper, best_params, y_true_all, y_pred_all = train_ml_model(
        df, n_days, eps, model_choice=model_choice,
        bootstrap_iters=1000, use_gridsearch=use_gridsearch, use_bootstrap=use_bootstrap,
        progress_bar=progress_bar
    )

    st.subheader("Machine Learning Forecast")
    st.write(f"Predicted Price: {predicted_price:.2f}")
    if ci_lower and ci_upper:
        st.write(f"95% CI: {ci_lower:.2f} - {ci_upper:.2f}")
    st.write(f"RMSE: {rmse:.2f}")

except Exception as e:
    st.error(f"Error loading data or running simulation: {e}")
    model = None
    y_true_all, y_pred_all = None, None

# --------------------------------------
# Backtest block
# --------------------------------------
if run_backtest and model is not None and y_true_all is not None and y_pred_all is not None:
    backtest_model_performance(y_true_all, y_pred_all)

# --------------------------------------
# S&P 500 Scanner
# --------------------------------------
def load_sp500_list(choice, uploaded, local_path):
    if choice == "Upload CSV" and uploaded is not None:
        return pd.read_csv(uploaded)["Symbol"].tolist()
    elif choice == "Local CSV":
        try:
            return pd.read_csv(local_path)["Symbol"].tolist()
        except Exception:
            st.warning("Local CSV not found or invalid.")
            return []
    return []

if scan_sp500:
    st.header("🔍 S&P 500 Scanner Results")
    tickers = load_sp500_list(sp500_choice, uploaded_sp500, local_sp500_path)
    if not tickers:
        st.warning("No tickers to scan. Upload or provide a local CSV.")
    else:
        st.info(f"Scanning {len(tickers)} tickers (this may take a while)...")

        def process_ticker(ticker_sym):
            try:
                df_t = get_stock_data(ticker_sym, "1y")
                df_t = add_technical_indicators(df_t)
                df_t.dropna(inplace=True)
                if len(df_t) < 80:
                    return None

                latest_close_t = df_t['Close'].iloc[-1]
                log_returns_t = np.log(df_t['Close'] / df_t['Close'].shift(1)).dropna()
                mu_t, sigma_t = log_returns_t.mean(), log_returns_t.std()

                sim_t = monte_carlo_simulation(latest_close_t, mu_t, sigma_t, n_days, n_days, max(200, min(1000, n_simulations//2)))
                mc_median = float(np.median(sim_t[-1, :]))
                mc_change_pct = (mc_median - latest_close_t) / latest_close_t * 100.0

                try:
                    _, _, pred_t, _, _, _, _, _, _ = train_ml_model(df_t, n_days, eps,
                                                                    model_choice=model_choice,
                                                                    bootstrap_iters=100, use_gridsearch=False, use_bootstrap=False,
                                                                    progress_bar=None)
                    ml_change_pct_t = (pred_t - latest_close_t) / latest_close_t * 100.0
                except Exception:
                    ml_change_pct_t = np.nan

                if rank_mode == "ML % Increase":
                    score = ml_change_pct_t if not np.isnan(ml_change_pct_t) else -np.inf
                elif rank_mode == "MC Median % Increase":
                    score = mc_change_pct
                else:
                    if np.isnan(ml_change_pct_t):
                        score = mc_change_pct
                    else:
                        score = 0.5 * (ml_change_pct_t + mc_change_pct)

                return {
                    "Ticker": ticker_sym,
                    "ML % Increase": ml_change_pct_t,
                    "MC Median % Increase": mc_change_pct,
                    "Score": score
                }
            except Exception:
                return None

        placeholder = st.empty()
        progress_bar = st.progress(0)
        results = []
        max_workers = min(16, max(4, (len(tickers)//10) + 1))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_ticker, t): t for t in tickers}
            total = len(futures)
            completed = 0
            for fut in as_completed(futures):
                completed += 1
                progress_bar.progress(int(completed / total * 100))
                res = fut.result()
                if res is not None:
                    results.append(res)

        progress_bar.empty()

        if not results:
            st.warning("No valid results returned from the scanner.")
        else:
            df_results = pd.DataFrame(results)
            df_results = df_results.replace([np.inf, -np.inf], np.nan)
            df_results = df_results.sort_values("Score", ascending=False).reset_index(drop=True)

            st.subheader("Top 10 Scanner Results")
            st.dataframe(df_results.head(10).style.format({
                "ML % Increase": "{:+.2f}%",
                "MC Median % Increase": "{:+.2f}%",
                "Score": "{:+.2f}%"
            }))

            csv = df_results.to_csv(index=False)
            st.download_button("⬇️ Download full results (CSV)", csv, file_name="sp500_scan_results.csv", mime="text/csv")

            best = df_results.iloc[0]
            st.markdown(f"**Top pick:** {best['Ticker']} — Score: {best['Score']:+.2f}% | "
                        f"ML: {best['ML % Increase']:+.2f}% | MC: {best['MC Median % Increase']:+.2f}%")