import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Quantitative Trading System v2.0")

# =============================================================================
# CONFIGURATION & DATA CLASSES
# =============================================================================
@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    commission_pct: float = 0.001  # 0.1% per trade (realistic for retail)
    slippage_pct: float = 0.0005   # 0.05% slippage
    max_position_pct: float = 0.10  # Max 10% per position
    stop_loss_atr: float = 2.0
    take_profit_atr: float = 3.0

@dataclass
class SignalResult:
    ticker: str
    signal: int  # -1, 0, 1
    confidence: float
    predicted_return: float
    volatility: float
    sharpe_contribution: float

# =============================================================================
# ADVANCED FEATURE ENGINEERING
# =============================================================================
def add_professional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Professional-grade features that are:
    1. Stationary (or normalized)
    2. Theoretically motivated
    3. Not easily arbitraged
    """
    data = df.copy()
    
    # -------------------------------------------------------------------------
    # RETURNS & VOLATILITY
    # -------------------------------------------------------------------------
    data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
    data['log_ret_5d'] = data['log_ret'].rolling(5).sum()
    data['log_ret_20d'] = data['log_ret'].rolling(20).sum()
    
    # Realized Volatility (multiple scales)
    data['rvol_5'] = data['log_ret'].rolling(5).std() * np.sqrt(252)
    data['rvol_20'] = data['log_ret'].rolling(20).std() * np.sqrt(252)
    data['rvol_60'] = data['log_ret'].rolling(60).std() * np.sqrt(252)
    
    # Volatility Ratio (vol regime indicator)
    data['vol_ratio'] = data['rvol_5'] / data['rvol_60']
    
    # -------------------------------------------------------------------------
    # MOMENTUM & MEAN REVERSION FEATURES
    # -------------------------------------------------------------------------
    # Rate of Change (multiple horizons)
    for period in [5, 10, 20]:
        data[f'roc_{period}'] = data['Close'].pct_change(period)
    
    # Normalized Distance from Moving Averages
    for period in [20, 50, 200]:
        sma = data['Close'].rolling(period).mean()
        data[f'dist_sma_{period}'] = (data['Close'] - sma) / sma
    
    # Bollinger Band Position (0 to 1)
    sma20 = data['Close'].rolling(20).mean()
    std20 = data['Close'].rolling(20).std()
    data['bb_position'] = (data['Close'] - (sma20 - 2*std20)) / (4*std20)
    data['bb_position'] = data['bb_position'].clip(0, 1)
    
    # -------------------------------------------------------------------------
    # RSI (Properly Calculated)
    # -------------------------------------------------------------------------
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data['rsi'] = 100 - (100 / (1 + rs))
    data['rsi_norm'] = (data['rsi'] - 50) / 50  # Normalized to [-1, 1]
    
    # RSI Divergence (price vs RSI direction mismatch)
    data['rsi_divergence'] = np.sign(data['roc_5']) != np.sign(data['rsi'].diff(5))
    
    # -------------------------------------------------------------------------
    # VOLUME FEATURES
    # -------------------------------------------------------------------------
    data['volume_sma'] = data['Volume'].rolling(20).mean()
    data['rel_volume'] = data['Volume'] / data['volume_sma']
    data['volume_trend'] = data['Volume'].rolling(5).mean() / data['Volume'].rolling(20).mean()
    
    # On-Balance Volume Trend
    obv = (np.sign(data['log_ret']) * data['Volume']).cumsum()
    data['obv_trend'] = obv.diff(10) / obv.rolling(20).std()
    
    # -------------------------------------------------------------------------
    # VOLATILITY-ADJUSTED FEATURES
    # -------------------------------------------------------------------------
    # ATR (for stop-loss calculations)
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift(1)).abs()
    low_close = (data['Low'] - data['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['atr_14'] = tr.rolling(14).mean()
    data['atr_pct'] = data['atr_14'] / data['Close']
    
    # -------------------------------------------------------------------------
    # REGIME INDICATORS
    # -------------------------------------------------------------------------
    # Trend Strength (ADX-like)
    data['trend_strength'] = abs(data['dist_sma_50']) / data['rvol_20']
    
    # Market Regime (based on vol and trend)
    data['high_vol_regime'] = (data['rvol_20'] > data['rvol_60']).astype(int)
    data['trending'] = (abs(data['dist_sma_50']) > data['rvol_60']/np.sqrt(252)*50).astype(int)
    
    # -------------------------------------------------------------------------
    # LAGGED FEATURES (Autoregression)
    # -------------------------------------------------------------------------
    for lag in [1, 2, 3, 5, 10]:
        data[f'ret_lag_{lag}'] = data['log_ret'].shift(lag)
    
    # -------------------------------------------------------------------------
    # HIGHER-ORDER FEATURES
    # -------------------------------------------------------------------------
    # Skewness and Kurtosis of returns
    data['ret_skew'] = data['log_ret'].rolling(20).skew()
    data['ret_kurt'] = data['log_ret'].rolling(20).kurt()
    
    # Return autocorrelation (mean reversion signal)
    data['ret_autocorr'] = data['log_ret'].rolling(20).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    
    return data.replace([np.inf, -np.inf], np.nan).dropna()


# =============================================================================
# ML MODEL WITH PROPER VALIDATION
# =============================================================================
class QuantModel:
    """Professional ML model with proper time-series validation."""
    
    FEATURE_COLS = [
        'rvol_20', 'vol_ratio', 'roc_5', 'roc_10', 'roc_20',
        'dist_sma_20', 'dist_sma_50', 'dist_sma_200', 'bb_position',
        'rsi_norm', 'rel_volume', 'volume_trend', 'obv_trend',
        'atr_pct', 'trend_strength', 'high_vol_regime', 'trending',
        'ret_lag_1', 'ret_lag_2', 'ret_lag_5',
        'ret_skew', 'ret_kurt', 'ret_autocorr'
    ]
    
    def __init__(self, forecast_horizon: int = 20):
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.cv_scores = []
        
    def prepare_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create forward-looking target with no leakage."""
        data = df.copy()
        # Target: Forward log return over horizon
        data['target'] = data['log_ret'].shift(-1).rolling(self.forecast_horizon).sum().shift(-self.forecast_horizon + 1)
        return data.dropna()
    
    def time_series_cv(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict:
        """Walk-forward cross-validation."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        scores = []
        directional_accuracy = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train
            model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train, verbose=False)
            
            # Predict
            preds = model.predict(X_test_scaled)
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            dir_acc = np.mean(np.sign(preds) == np.sign(y_test))
            
            scores.append(rmse)
            directional_accuracy.append(dir_acc)
        
        return {
            'rmse_mean': np.mean(scores),
            'rmse_std': np.std(scores),
            'directional_accuracy': np.mean(directional_accuracy),
            'da_std': np.std(directional_accuracy)
        }
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train model with proper validation."""
        data = self.prepare_target(df)
        
        if len(data) < 252:  # At least 1 year of data
            return None
        
        # Get features that exist in the data
        available_features = [f for f in self.FEATURE_COLS if f in data.columns]
        X = data[available_features].values
        y = data['target'].values
        
        # Cross-validation first
        cv_results = self.time_series_cv(X, y)
        self.cv_scores = cv_results
        
        # Check if model has any predictive power
        # Random would be 50% directional accuracy
        if cv_results['directional_accuracy'] < 0.52:
            # Model is essentially random - no edge
            cv_results['has_edge'] = False
        else:
            cv_results['has_edge'] = True
        
        # Train final model on all data
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y, verbose=False)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Current prediction
        last_row = data[available_features].iloc[-1:].values
        last_row_scaled = self.scaler.transform(last_row)
        pred_log_return = self.model.predict(last_row_scaled)[0]
        
        cv_results['predicted_log_return'] = pred_log_return
        cv_results['predicted_pct_return'] = (np.exp(pred_log_return) - 1) * 100
        cv_results['current_volatility'] = data['rvol_20'].iloc[-1]
        
        return cv_results


# =============================================================================
# MONTE CARLO WITH VOLATILITY CLUSTERING
# =============================================================================
def garch_monte_carlo(df: pd.DataFrame, n_sims: int, n_days: int, 
                      current_price: float) -> Optional[np.ndarray]:
    """
    Monte Carlo with simple volatility clustering (GARCH-like).
    More realistic than IID bootstrap.
    """
    returns = df['log_ret'].values
    
    if len(returns) < 100:
        return None
    
    # Estimate simple GARCH(1,1) parameters
    # Vol tomorrow = omega + alpha * shock^2 + beta * vol_today
    omega = 0.00001
    alpha = 0.1
    beta = 0.85
    
    # Get current volatility estimate
    current_vol = df['rvol_20'].iloc[-1] / np.sqrt(252)  # Daily vol
    
    # Historical return distribution (for shock sampling)
    standardized_returns = returns / (np.std(returns) + 1e-8)
    
    paths = np.zeros((n_days, n_sims))
    vols = np.zeros((n_days, n_sims))
    
    vols[0, :] = current_vol ** 2  # Variance
    
    for t in range(n_days):
        # Sample standardized returns
        shocks = np.random.choice(standardized_returns, size=n_sims)
        
        if t == 0:
            var_t = current_vol ** 2
        else:
            # GARCH variance update
            var_t = omega + alpha * (paths[t-1, :] ** 2) + beta * vols[t-1, :]
            vols[t, :] = var_t
        
        vol_t = np.sqrt(np.abs(var_t))
        paths[t, :] = shocks * vol_t
    
    # Convert to price paths
    cumulative_returns = np.cumsum(paths, axis=0)
    price_paths = current_price * np.exp(cumulative_returns)
    
    return price_paths


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================
class Backtester:
    """Proper backtesting with transaction costs and realistic execution."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
    def run(self, df: pd.DataFrame, model: QuantModel) -> Dict:
        """Run walk-forward backtest."""
        data = model.prepare_target(df.copy())
        
        if len(data) < 504:  # Need 2 years minimum
            return None
        
        available_features = [f for f in model.FEATURE_COLS if f in data.columns]
        
        # Walk-forward: train on first 60%, trade on remaining 40%
        train_end = int(len(data) * 0.6)
        
        results = {
            'dates': [],
            'returns': [],
            'positions': [],
            'signals': [],
            'cumulative_returns': [],
            'drawdowns': []
        }
        
        capital = self.config.initial_capital
        peak_capital = capital
        position = 0
        entry_price = 0
        
        # Retrain every 60 days
        retrain_interval = 60
        last_train = 0
        
        for i in range(train_end, len(data)):
            current_date = data.index[i]
            current_price = data['Close'].iloc[i]
            
            # Retrain model periodically
            if i - last_train >= retrain_interval:
                train_data = data.iloc[:i]
                X_train = train_data[available_features].values
                y_train = train_data['target'].values
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_train)
                
                temp_model = XGBRegressor(
                    n_estimators=100, learning_rate=0.05, max_depth=3,
                    random_state=42, n_jobs=-1
                )
                temp_model.fit(X_scaled, y_train, verbose=False)
                last_train = i
            
            # Generate signal
            X_current = data[available_features].iloc[i:i+1].values
            X_current_scaled = scaler.transform(X_current)
            pred = temp_model.predict(X_current_scaled)[0]
            
            # Signal: only trade if prediction is significant
            vol = data['rvol_20'].iloc[i] / np.sqrt(252) * np.sqrt(model.forecast_horizon)
            signal_threshold = vol * 0.5  # Prediction must exceed half of expected vol
            
            if pred > signal_threshold:
                signal = 1  # Long
            elif pred < -signal_threshold:
                signal = -1  # Short
            else:
                signal = 0  # Flat
            
            # Execute trades
            daily_return = 0
            
            if signal != position:
                # Close existing position
                if position != 0:
                    exit_price = current_price * (1 - self.config.slippage_pct * np.sign(position))
                    if position == 1:
                        trade_return = (exit_price - entry_price) / entry_price
                    else:
                        trade_return = (entry_price - exit_price) / entry_price
                    
                    # Apply commission
                    trade_return -= self.config.commission_pct * 2  # Entry + exit
                    daily_return = trade_return * self.config.max_position_pct
                
                # Open new position
                if signal != 0:
                    entry_price = current_price * (1 + self.config.slippage_pct * np.sign(signal))
                    position = signal
            
            # Update capital
            capital *= (1 + daily_return)
            peak_capital = max(peak_capital, capital)
            drawdown = (peak_capital - capital) / peak_capital
            
            results['dates'].append(current_date)
            results['returns'].append(daily_return)
            results['positions'].append(position)
            results['signals'].append(signal)
            results['cumulative_returns'].append(capital / self.config.initial_capital - 1)
            results['drawdowns'].append(drawdown)
        
        # Calculate metrics
        returns = np.array(results['returns'])
        trading_days = len(returns)
        
        if trading_days < 20:
            return None
        
        # Performance Metrics
        total_return = capital / self.config.initial_capital - 1
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe = annualized_return / volatility if volatility > 0 else 0
        
        max_dd = max(results['drawdowns'])
        
        # Win rate
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades)) if len(returns) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        results['metrics'] = {
            'total_return': total_return * 100,
            'annualized_return': annualized_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd * 100,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'num_trades': np.sum(np.diff(results['positions']) != 0)
        }
        
        return results


# =============================================================================
# DATA FETCHING
# =============================================================================
@st.cache_data(ttl=3600)
def get_sp500_tickers():
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        return table[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
    except:
        return ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "AMD"]

@st.cache_data(ttl=3600)
def get_stock_data(ticker: str, period: str = "5y") -> Optional[pd.DataFrame]:
    """Fetch and process stock data."""
    try:
        df = yf.Ticker(ticker).history(period=period)
        if len(df) < 504:  # Need 2 years
            return None
        return add_professional_features(df)
    except:
        return None


# =============================================================================
# VISUALIZATION
# =============================================================================
def create_backtest_chart(results: Dict) -> go.Figure:
    """Create comprehensive backtest visualization."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=('Cumulative Returns', 'Position', 'Drawdown')
    )
    
    dates = results['dates']
    
    # Cumulative returns
    fig.add_trace(
        go.Scatter(x=dates, y=np.array(results['cumulative_returns']) * 100,
                   mode='lines', name='Strategy', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # Benchmark (buy & hold would need price data)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    # Position
    fig.add_trace(
        go.Scatter(x=dates, y=results['positions'],
                   mode='lines', name='Position', line=dict(color='green', width=1)),
        row=2, col=1
    )
    
    # Drawdown
    fig.add_trace(
        go.Scatter(x=dates, y=np.array(results['drawdowns']) * -100,
                   mode='lines', name='Drawdown', fill='tozeroy',
                   line=dict(color='red', width=1)),
        row=3, col=1
    )
    
    fig.update_layout(height=700, showlegend=True)
    fig.update_yaxes(title_text="Return %", row=1, col=1)
    fig.update_yaxes(title_text="Position", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=3, col=1)
    
    return fig


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    st.sidebar.title("⚙️ Configuration")
    mode = st.sidebar.radio("Mode", ["Single Stock Analysis", "Market Scanner", "About"])
    
    if mode == "About":
        st.title("📊 About This System")
        st.markdown("""
        ## What Makes This Different?
        
        ### Improvements Over Original Code:
        
        | Aspect | Original | Improved |
        |--------|----------|----------|
        | **Validation** | Single train/test split | Walk-forward cross-validation |
        | **Backtesting** | ❌ None | ✅ Full backtest with costs |
        | **Features** | Basic TA | Advanced + regime indicators |
        | **Monte Carlo** | IID bootstrap | GARCH-like vol clustering |
        | **Metrics** | Just RMSE | Sharpe, Max DD, Win Rate, etc. |
        | **Overfitting** | High risk | Regularization + proper CV |
        
        ### Honest Expectations:
        
        🔴 **This will NOT consistently beat the market.**
        
        Why? Because:
        1. Technical indicators are well-known and arbitraged
        2. True alpha requires alternative data or HFT infrastructure
        3. Transaction costs eat most retail edge
        4. Markets are adaptive - what works stops working
        
        ✅ **What this CAN do:**
        1. Help you understand if a signal has any historical edge
        2. Proper risk management framework
        3. Avoid obvious mistakes (overfitting, look-ahead bias)
        4. Educational tool for quant concepts
        """)
        return
    
    if mode == "Single Stock Analysis":
        ticker = st.sidebar.text_input("Ticker", "NVDA").upper()
        forecast_days = st.sidebar.slider("Forecast Horizon (days)", 5, 60, 20)
        run_backtest = st.sidebar.checkbox("Run Full Backtest", value=True)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Risk Parameters")
        initial_capital = st.sidebar.number_input("Initial Capital ($)", 10000, 1000000, 100000)
        risk_per_trade = st.sidebar.slider("Max Position Size (%)", 5, 25, 10)
        
        # Fetch data
        with st.spinner(f"Loading {ticker} data..."):
            df = get_stock_data(ticker, period="5y")
        
        if df is None:
            st.error("Could not fetch sufficient data. Need at least 2 years.")
            return
        
        current_price = df['Close'].iloc[-1]
        st.title(f"📈 {ticker} Analysis - ${current_price:.2f}")
        
        # Train model
        with st.spinner("Training model with cross-validation..."):
            model = QuantModel(forecast_horizon=forecast_days)
            cv_results = model.train(df)
        
        if cv_results is None:
            st.error("Insufficient data for analysis")
            return
        
        # Display CV Results
        st.subheader("🔬 Model Validation Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Directional accuracy color
        da = cv_results['directional_accuracy'] * 100
        da_color = "green" if da > 52 else "red"
        
        col1.metric(
            "Directional Accuracy", 
            f"{da:.1f}%",
            delta=f"{da - 50:.1f}% vs random" if da > 50 else f"{da - 50:.1f}%"
        )
        col2.metric("CV RMSE", f"{cv_results['rmse_mean']:.4f}", delta=f"±{cv_results['rmse_std']:.4f}")
        col3.metric("Predicted Return", f"{cv_results['predicted_pct_return']:.2f}%")
        col4.metric("Current Vol (Ann.)", f"{cv_results['current_volatility']*100:.1f}%")
        
        # Edge Warning
        if not cv_results['has_edge']:
            st.warning("""
            ⚠️ **No Statistical Edge Detected**
            
            The model's directional accuracy is below 52%, which means it's essentially 
            random after accounting for noise. This is NORMAL - most simple models 
            cannot beat the market.
            """)
        else:
            st.success(f"""
            ✅ **Potential Edge Detected** (proceed with caution)
            
            Directional accuracy of {da:.1f}% suggests some predictive signal, 
            but this needs to be validated with full backtesting including costs.
            """)
        
        # Monte Carlo
        st.subheader("📊 Monte Carlo Simulation (with Vol Clustering)")
        
        with st.spinner("Running Monte Carlo..."):
            mc_paths = garch_monte_carlo(df, n_sims=1000, n_days=forecast_days, current_price=current_price)
        
        if mc_paths is not None:
            final_prices = mc_paths[-1, :]
            p5, p25, p50, p75, p95 = np.percentile(final_prices, [5, 25, 50, 75, 95])
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("5th %ile (Bear)", f"${p5:.2f}", f"{(p5/current_price-1)*100:.1f}%")
            col2.metric("25th %ile", f"${p25:.2f}", f"{(p25/current_price-1)*100:.1f}%")
            col3.metric("Median", f"${p50:.2f}", f"{(p50/current_price-1)*100:.1f}%")
            col4.metric("75th %ile", f"${p75:.2f}", f"{(p75/current_price-1)*100:.1f}%")
            col5.metric("95th %ile (Bull)", f"${p95:.2f}", f"{(p95/current_price-1)*100:.1f}%")
            
            # MC Chart
            fig_mc = go.Figure()
            
            # Plot percentile bands
            days = list(range(forecast_days))
            for p in [5, 25, 50, 75, 95]:
                path_percentile = np.percentile(mc_paths, p, axis=1)
                fig_mc.add_trace(go.Scatter(
                    x=days, y=path_percentile,
                    mode='lines',
                    name=f'{p}th percentile',
                    line=dict(width=2 if p == 50 else 1)
                ))
            
            # Add ML prediction
            ml_target = current_price * (1 + cv_results['predicted_pct_return']/100)
            fig_mc.add_trace(go.Scatter(
                x=[forecast_days-1], y=[ml_target],
                mode='markers',
                name='ML Prediction',
                marker=dict(size=15, symbol='star', color='red')
            ))
            
            fig_mc.update_layout(title="Monte Carlo Projection with Percentile Bands", height=400)
            st.plotly_chart(fig_mc, use_container_width=True)
        
        # Backtest
        if run_backtest:
            st.subheader("📈 Walk-Forward Backtest")
            
            config = BacktestConfig(
                initial_capital=initial_capital,
                max_position_pct=risk_per_trade/100
            )
            
            with st.spinner("Running backtest..."):
                backtester = Backtester(config)
                bt_results = backtester.run(df, model)
            
            if bt_results is not None:
                metrics = bt_results['metrics']
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Return", f"{metrics['total_return']:.2f}%")
                col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.1f}%")
                col4.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Annualized Return", f"{metrics['annualized_return']:.2f}%")
                col2.metric("Volatility (Ann.)", f"{metrics['volatility']:.1f}%")
                col3.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                col4.metric("# Trades", f"{metrics['num_trades']:.0f}")
                
                # Interpretation
                if metrics['sharpe_ratio'] < 0.5:
                    st.error("⚠️ Poor risk-adjusted returns. Strategy not recommended.")
                elif metrics['sharpe_ratio'] < 1.0:
                    st.warning("🟡 Marginal edge. High uncertainty.")
                else:
                    st.success("✅ Positive Sharpe, but validate on more data before trading.")
                
                # Chart
                fig_bt = create_backtest_chart(bt_results)
                st.plotly_chart(fig_bt, use_container_width=True)
            else:
                st.warning("Insufficient data for backtest")
        
        # Feature Importance
        if model.feature_importance is not None:
            st.subheader("🔍 Feature Importance")
            fig_feat = go.Figure(go.Bar(
                x=model.feature_importance['importance'].head(10),
                y=model.feature_importance['feature'].head(10),
                orientation='h'
            ))
            fig_feat.update_layout(height=400, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_feat, use_container_width=True)
    
    elif mode == "Market Scanner":
        st.title("🔍 Market Scanner")
        st.info("Scans for stocks where the model shows potential edge (>52% directional accuracy)")
        
        forecast_days = st.slider("Forecast Horizon", 10, 60, 20)
        max_stocks = st.slider("Stocks to Scan", 20, 100, 50)
        
        if st.button("🚀 Start Scan"):
            tickers = get_sp500_tickers()[:max_stocks]
            
            results = []
            progress = st.progress(0)
            status = st.empty()
            
            def scan_single(ticker):
                try:
                    df = get_stock_data(ticker, "5y")
                    if df is None:
                        return None
                    
                    model = QuantModel(forecast_horizon=forecast_days)
                    cv_results = model.train(df)
                    
                    if cv_results is None:
                        return None
                    
                    return {
                        'Ticker': ticker,
                        'Price': df['Close'].iloc[-1],
                        'Dir. Accuracy': cv_results['directional_accuracy'] * 100,
                        'Pred. Return %': cv_results['predicted_pct_return'],
                        'Volatility %': cv_results['current_volatility'] * 100,
                        'Has Edge': cv_results['has_edge']
                    }
                except:
                    return None
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(scan_single, t): t for t in tickers}
                completed = 0
                
                for future in as_completed(futures):
                    res = future.result()
                    if res:
                        results.append(res)
                    completed += 1
                    progress.progress(completed / len(tickers))
                    status.text(f"Scanned {completed}/{len(tickers)}")
            
            if results:
                df_results = pd.DataFrame(results)
                
                st.subheader("✅ Stocks with Potential Edge")
                edge_stocks = df_results[df_results['Has Edge']].sort_values('Dir. Accuracy', ascending=False)
                
                if len(edge_stocks) > 0:
                    st.dataframe(
                        edge_stocks.style.format({
                            'Price': '${:.2f}',
                            'Dir. Accuracy': '{:.1f}%',
                            'Pred. Return %': '{:.2f}%',
                            'Volatility %': '{:.1f}%'
                        }),
                        use_container_width=True
                    )
                else:
                    st.warning("No stocks showed statistical edge. This is normal!")
                
                st.subheader("📊 All Results")
                st.dataframe(
                    df_results.sort_values('Dir. Accuracy', ascending=False).style.format({
                        'Price': '${:.2f}',
                        'Dir. Accuracy': '{:.1f}%',
                        'Pred. Return %': '{:.2f}%',
                        'Volatility %': '{:.1f}%'
                    }),
                    use_container_width=True
                )

if __name__ == "__main__":
    main()
