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
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Quantitative Trading System v3.1")

# =============================================================================
# CONFIGURATION WITH PRESETS
# =============================================================================
@dataclass
class TradingConfig:
    initial_capital: float = 100000.0
    commission_pct: float = 0.001      # 0.1% per trade
    slippage_pct: float = 0.0005       # 0.05% slippage
    max_position_pct: float = 0.10     # Max 10% of portfolio per trade
    
@dataclass  
class ModelConfig:
    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.05
    cv_folds: int = 5
    min_data_points: int = 504  # 2 years - ORIGINAL VALUE
    
    @classmethod
    def fast(cls):
        """Fast config for quick scans - less reliable but functional"""
        return cls(n_estimators=50, cv_folds=3, min_data_points=252)
    
    @classmethod
    def balanced(cls):
        """Balanced speed/accuracy - good middle ground"""
        return cls(n_estimators=75, cv_folds=4, min_data_points=378)
    
    @classmethod
    def full(cls):
        """Full power - most reliable, same as original"""
        return cls(n_estimators=100, cv_folds=5, min_data_points=504)

# =============================================================================
# COMPREHENSIVE TICKER DATABASE (FIXES THE SCANNER ISSUE)
# =============================================================================
SP500_TICKERS = [
    # Top 200 S&P 500 by market cap - hardcoded fallback
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "UNH", "XOM",
    "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY",
    "PEP", "KO", "AVGO", "COST", "MCD", "WMT", "CSCO", "TMO", "ACN", "ABT",
    "DHR", "BAC", "CRM", "CMCSA", "PFE", "ADBE", "NKE", "DIS", "VZ", "NFLX",
    "INTC", "WFC", "TXN", "PM", "NEE", "RTX", "BMY", "UNP", "QCOM", "UPS",
    "COP", "ORCL", "AMD", "MS", "HON", "LOW", "SPGI", "CAT", "IBM", "BA",
    "GS", "SBUX", "AMGN", "ELV", "DE", "INTU", "GE", "BLK", "AMAT", "GILD",
    "AXP", "PLD", "MDLZ", "LMT", "CVS", "ADI", "NOW", "TJX", "ISRG", "SYK",
    "REGN", "ADP", "VRTX", "BKNG", "MMC", "TMUS", "MO", "LRCX", "C", "ZTS",
    "CI", "SCHW", "CB", "ETN", "SO", "EOG", "BSX", "BDX", "DUK", "CME",
    "PGR", "NOC", "SLB", "MU", "ITW", "SNPS", "FI", "CL", "CSX", "CDNS",
    "HUM", "WM", "FCX", "AON", "ICE", "FDX", "MCK", "SHW", "ORLY", "MCO",
    "EMR", "GD", "PH", "KLAC", "PNC", "NXPI", "PSX", "TGT", "MAR", "NSC",
    "APD", "USB", "ROP", "AZO", "MSI", "CARR", "TDG", "PCAR", "AJG", "ECL",
    "OXY", "TT", "MCHP", "ADSK", "CTAS", "SRE", "MPC", "AEP", "CCI", "HCA",
    "FTNT", "TEL", "AFL", "TFC", "PAYX", "WELL", "KMB", "PSA", "DXCM", "GIS",
    "D", "VLO", "F", "MSCI", "MNST", "JCI", "AMP", "PEG", "A", "SPG",
    "KDP", "GM", "O", "CMG", "STZ", "NEM", "DHI", "HES", "ROST", "IDXX",
    "BIIB", "YUM", "CTSH", "DOW", "IQV", "ALL", "AIG", "LHX", "CHTR", "BK",
    "AME", "CPRT", "CMI", "EXC", "HAL", "KHC", "EA", "MRNA", "PRU", "OTIS"
]

NASDAQ100_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "AVGO", "COST", "ASML",
    "AMD", "ADBE", "NFLX", "PEP", "CSCO", "TMUS", "CMCSA", "INTC", "INTU", "AMGN",
    "TXN", "QCOM", "HON", "AMAT", "BKNG", "SBUX", "ISRG", "MDLZ", "ADP", "GILD",
    "VRTX", "ADI", "REGN", "PANW", "MU", "SNPS", "KLAC", "CDNS", "LRCX", "PYPL",
    "CSX", "MELI", "ORLY", "CRWD", "MAR", "CTAS", "MNST", "NXPI", "MCHP", "PCAR",
    "FTNT", "AEP", "KDP", "ADSK", "CPRT", "ROST", "DXCM", "AZN", "PAYX", "KHC",
    "IDXX", "CTSH", "CHTR", "MRNA", "EA", "BIIB", "ODFL", "EXC", "XEL", "GEHC",
    "ON", "CSGP", "FANG", "VRSK", "FAST", "DDOG", "ANSS", "ZS", "CDW", "TEAM",
    "GFS", "ILMN", "DLTR", "WBD", "BKR", "CEG", "ALGN", "ENPH", "WBA", "SIRI",
    "LCID", "JD", "PDD", "RIVN", "ZM", "ROKU", "COIN", "HOOD", "ABNB", "DASH"
]

POPULAR_TICKERS = [
    # Mega Cap Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # Semiconductors
    "AMD", "INTC", "AVGO", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC", "MRVL",
    # Software/Cloud
    "CRM", "ADBE", "NOW", "ORCL", "CSCO", "INTU", "PANW", "CRWD", "ZS", "SNOW",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V", "MA", "PYPL",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "BMY", "AMGN", "GILD",
    # Consumer
    "WMT", "COST", "HD", "LOW", "TGT", "NKE", "SBUX", "MCD", "KO", "PEP",
    # Industrial
    "CAT", "DE", "HON", "UNP", "BA", "LMT", "RTX", "GE",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLV", "XLI", "XLP",
    # High Growth/Speculative
    "COIN", "SQ", "SHOP", "PLTR", "UBER", "ABNB", "RIVN", "LCID", "SOFI", "RBLX"
]

MEGA_CAP_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "LLY"
]

ETF_TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLV", "XLI", 
    "XLP", "XLY", "XLB", "XLU", "XLRE", "VOO", "VTI", "VEA", "VWO",
    "BND", "TLT", "GLD", "SLV", "USO", "VNQ", "ARKK", "ARKG", "ARKW",
    "SMH", "XBI", "KRE", "XRT", "ITB", "XHB", "JETS", "HACK"
]

def get_ticker_list(source: str) -> List[str]:
    """Get ticker list from various sources with robust fallbacks."""
    
    if source == "S&P 500":
        # Try Wikipedia first for most up-to-date list
        try:
            table = pd.read_html(
                'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                timeout=10
            )
            tickers = table[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
            if len(tickers) > 400:
                return tickers
        except Exception:
            pass
        # Fallback to hardcoded list
        return SP500_TICKERS
    
    elif source == "Nasdaq 100":
        try:
            table = pd.read_html(
                'https://en.wikipedia.org/wiki/Nasdaq-100',
                timeout=10
            )
            for t in table:
                if 'Ticker' in t.columns:
                    tickers = t['Ticker'].tolist()
                    if len(tickers) > 90:
                        return tickers
                if 'Symbol' in t.columns:
                    tickers = t['Symbol'].tolist()
                    if len(tickers) > 90:
                        return tickers
        except Exception:
            pass
        return NASDAQ100_TICKERS
    
    elif source == "Popular Stocks":
        return POPULAR_TICKERS
    
    elif source == "Mega Caps Only":
        return MEGA_CAP_TICKERS
    
    elif source == "ETFs Only":
        return ETF_TICKERS
    
    elif source == "All Combined":
        # Combine all lists and deduplicate
        all_tickers = list(set(SP500_TICKERS + NASDAQ100_TICKERS + POPULAR_TICKERS + ETF_TICKERS))
        return sorted(all_tickers)
    
    return POPULAR_TICKERS

# =============================================================================
# PROFESSIONAL FEATURE ENGINEERING (UNCHANGED FROM ORIGINAL)
# =============================================================================
def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Vectorized RSI calculation."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_macd(prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD indicator."""
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete professional feature set - optimized for speed.
    No slow .apply() operations.
    """
    data = df.copy()
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']
    
    # =========================================================================
    # RETURNS & VOLATILITY
    # =========================================================================
    data['log_ret'] = np.log(close / close.shift(1))
    data['log_ret_2d'] = data['log_ret'].rolling(2).sum()
    data['log_ret_5d'] = data['log_ret'].rolling(5).sum()
    data['log_ret_10d'] = data['log_ret'].rolling(10).sum()
    data['log_ret_20d'] = data['log_ret'].rolling(20).sum()
    
    # Realized Volatility (annualized)
    data['rvol_5'] = data['log_ret'].rolling(5).std() * np.sqrt(252)
    data['rvol_10'] = data['log_ret'].rolling(10).std() * np.sqrt(252)
    data['rvol_20'] = data['log_ret'].rolling(20).std() * np.sqrt(252)
    data['rvol_60'] = data['log_ret'].rolling(60).std() * np.sqrt(252)
    
    # Volatility ratios (regime detection)
    data['vol_ratio_5_20'] = data['rvol_5'] / (data['rvol_20'] + 1e-10)
    data['vol_ratio_20_60'] = data['rvol_20'] / (data['rvol_60'] + 1e-10)
    
    # Volatility trend
    data['vol_change'] = data['rvol_20'].pct_change(5)
    
    # =========================================================================
    # PRICE MOMENTUM & MEAN REVERSION
    # =========================================================================
    # Rate of change
    for period in [5, 10, 20, 60]:
        data[f'roc_{period}'] = close.pct_change(period)
    
    # Moving averages & distance
    for period in [10, 20, 50, 100, 200]:
        sma = close.rolling(period).mean()
        data[f'sma_{period}'] = sma
        data[f'dist_sma_{period}'] = (close - sma) / (sma + 1e-10)
    
    # EMA
    for period in [12, 26, 50]:
        data[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
    
    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    data['bb_upper'] = sma20 + 2 * std20
    data['bb_lower'] = sma20 - 2 * std20
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / (sma20 + 1e-10)
    data['bb_position'] = (close - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'] + 1e-10)
    data['bb_position'] = data['bb_position'].clip(0, 1)
    
    # =========================================================================
    # OSCILLATORS
    # =========================================================================
    # RSI
    data['rsi_14'] = compute_rsi(close, 14)
    data['rsi_norm'] = (data['rsi_14'] - 50) / 50  # Normalized [-1, 1]
    
    # RSI overbought/oversold
    data['rsi_ob'] = (data['rsi_14'] > 70).astype(float)
    data['rsi_os'] = (data['rsi_14'] < 30).astype(float)
    
    # Stochastic
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    data['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
    data['stoch_d'] = data['stoch_k'].rolling(3).mean()
    data['stoch_norm'] = (data['stoch_k'] - 50) / 50
    
    # MACD
    macd_line, signal_line, histogram = compute_macd(close)
    data['macd'] = macd_line
    data['macd_signal'] = signal_line
    data['macd_hist'] = histogram
    data['macd_hist_norm'] = histogram / (close + 1e-10)  # Normalized by price
    
    # =========================================================================
    # VOLATILITY INDICATORS
    # =========================================================================
    data['atr_14'] = compute_atr(high, low, close, 14)
    data['atr_pct'] = data['atr_14'] / (close + 1e-10)
    
    # Keltner Channels
    ema20 = close.ewm(span=20, adjust=False).mean()
    data['kc_upper'] = ema20 + 2 * data['atr_14']
    data['kc_lower'] = ema20 - 2 * data['atr_14']
    data['kc_position'] = (close - data['kc_lower']) / (data['kc_upper'] - data['kc_lower'] + 1e-10)
    
    # =========================================================================
    # VOLUME FEATURES
    # =========================================================================
    data['volume_sma_20'] = volume.rolling(20).mean()
    data['rel_volume'] = volume / (data['volume_sma_20'] + 1e-10)
    data['volume_trend'] = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1e-10)
    
    # Volume-Price Trend
    data['vpt'] = (data['log_ret'] * volume).cumsum()
    data['vpt_sma'] = data['vpt'].rolling(20).mean()
    data['vpt_signal'] = (data['vpt'] > data['vpt_sma']).astype(float)
    
    # On-Balance Volume (normalized)
    obv = (np.sign(data['log_ret']) * volume).cumsum()
    data['obv_pct_change'] = obv.pct_change(10)
    
    # =========================================================================
    # HIGHER-ORDER STATISTICS (FAST - no .apply())
    # =========================================================================
    # Rolling skewness approximation using moments
    rolling_mean = data['log_ret'].rolling(20).mean()
    rolling_std = data['log_ret'].rolling(20).std()
    
    # Skewness proxy: (mean - median) / std
    rolling_median = data['log_ret'].rolling(20).median()
    data['ret_skew_proxy'] = (rolling_mean - rolling_median) / (rolling_std + 1e-10)
    
    # Kurtosis proxy: (max - min) / std
    rolling_max = data['log_ret'].rolling(20).max()
    rolling_min = data['log_ret'].rolling(20).min()
    data['ret_range_norm'] = (rolling_max - rolling_min) / (rolling_std + 1e-10)
    
    # =========================================================================
    # TREND INDICATORS
    # =========================================================================
    # ADX approximation
    data['trend_strength'] = abs(data['dist_sma_50']) * 100
    
    # Price position in range
    data['high_20'] = high.rolling(20).max()
    data['low_20'] = low.rolling(20).min()
    data['price_position'] = (close - data['low_20']) / (data['high_20'] - data['low_20'] + 1e-10)
    
    # Trend direction
    data['uptrend'] = (close > data['sma_50']).astype(float)
    data['downtrend'] = (close < data['sma_50']).astype(float)
    
    # Golden/Death cross
    data['golden_cross'] = ((data['sma_50'] > data['sma_200']) & 
                            (data['sma_50'].shift(1) <= data['sma_200'].shift(1))).astype(float)
    data['death_cross'] = ((data['sma_50'] < data['sma_200']) & 
                           (data['sma_50'].shift(1) >= data['sma_200'].shift(1))).astype(float)
    
    # =========================================================================
    # REGIME INDICATORS
    # =========================================================================
    data['high_vol_regime'] = (data['rvol_20'] > data['rvol_60']).astype(float)
    data['low_vol_regime'] = (data['rvol_20'] < data['rvol_60'] * 0.8).astype(float)
    data['trending_regime'] = (abs(data['dist_sma_50']) > 0.05).astype(float)
    data['mean_revert_regime'] = (abs(data['dist_sma_20']) > abs(data['dist_sma_50'])).astype(float)
    
    # =========================================================================
    # LAGGED FEATURES
    # =========================================================================
    for lag in [1, 2, 3, 5, 10]:
        data[f'ret_lag_{lag}'] = data['log_ret'].shift(lag)
        data[f'vol_lag_{lag}'] = data['rvol_20'].shift(lag)
    
    # =========================================================================
    # INTERACTION FEATURES
    # =========================================================================
    data['momentum_vol_adj'] = data['roc_20'] / (data['rvol_20'] + 1e-10)  # Sharpe-like
    data['rsi_vol_interaction'] = data['rsi_norm'] * data['vol_ratio_5_20']
    data['volume_momentum'] = data['rel_volume'] * data['roc_5']
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    # Drop intermediate columns not needed for ML
    cols_to_drop = ['sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
                    'ema_12', 'ema_26', 'ema_50', 'bb_upper', 'bb_lower',
                    'kc_upper', 'kc_lower', 'volume_sma_20', 'high_20', 'low_20',
                    'vpt', 'vpt_sma', 'macd', 'macd_signal']
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns], errors='ignore')
    
    # Handle infinities and NaN
    data = data.replace([np.inf, -np.inf], 0)
    data = data.dropna()
    
    return data

# =============================================================================
# PROFESSIONAL ML MODEL (UNCHANGED FROM ORIGINAL)
# =============================================================================
class ProfessionalModel:
    """Production-grade ML model with full validation."""
    
    # Core features for ML (selected for predictive power)
    FEATURE_COLS = [
        # Volatility
        'rvol_20', 'vol_ratio_5_20', 'vol_ratio_20_60', 'vol_change', 'atr_pct',
        # Momentum
        'roc_5', 'roc_10', 'roc_20', 'roc_60',
        'dist_sma_20', 'dist_sma_50', 'dist_sma_100', 'dist_sma_200',
        # Oscillators
        'rsi_norm', 'stoch_norm', 'macd_hist_norm', 'bb_position', 'kc_position',
        # Volume
        'rel_volume', 'volume_trend', 'obv_pct_change', 'vpt_signal',
        # Regime
        'high_vol_regime', 'trending_regime', 'mean_revert_regime',
        # Lagged
        'ret_lag_1', 'ret_lag_2', 'ret_lag_5',
        # Higher-order
        'ret_skew_proxy', 'ret_range_norm',
        # Interaction
        'momentum_vol_adj', 'rsi_vol_interaction', 'volume_momentum',
        # Trend
        'trend_strength', 'price_position', 'uptrend'
    ]
    
    def __init__(self, forecast_horizon: int = 20, config: ModelConfig = None):
        self.forecast_horizon = forecast_horizon
        self.config = config or ModelConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.cv_results = {}
        self.available_features = []
        
    def _get_available_features(self, df: pd.DataFrame) -> list:
        """Get features that exist in the dataframe."""
        return [f for f in self.FEATURE_COLS if f in df.columns]
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with target variable."""
        data = df.copy()
        # Target: Forward cumulative log return
        data['target'] = data['log_ret'].rolling(self.forecast_horizon).sum().shift(-self.forecast_horizon)
        return data.dropna()
    
    def cross_validate(self, df: pd.DataFrame, progress_callback=None) -> Optional[Dict]:
        """Walk-forward cross-validation."""
        data = self._prepare_data(df)
        
        if len(data) < self.config.min_data_points:
            return None
        
        self.available_features = self._get_available_features(data)
        
        if len(self.available_features) < 10:
            return None
            
        X = data[self.available_features].values
        y = data['target'].values
        
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        rmse_scores = []
        directional_accuracy = []
        fold_predictions = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            if progress_callback:
                progress_callback(f"CV Fold {fold + 1}/{self.config.cv_folds}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            # Train
            model = XGBRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            model.fit(X_train_s, y_train)
            
            # Predict
            preds = model.predict(X_test_s)
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            dir_acc = np.mean(np.sign(preds) == np.sign(y_test))
            
            rmse_scores.append(rmse)
            directional_accuracy.append(dir_acc)
            fold_predictions.append({'actual': y_test, 'predicted': preds})
        
        # Aggregate results
        avg_dir_acc = np.mean(directional_accuracy)
        
        return {
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'rmse_scores': rmse_scores,
            'directional_accuracy': avg_dir_acc,
            'da_std': np.std(directional_accuracy),
            'da_scores': directional_accuracy,
            'fold_predictions': fold_predictions,
            'has_edge': avg_dir_acc > 0.52,
            'confidence': min(1.0, (avg_dir_acc - 0.50) / 0.10)  # 0-1 scale
        }
    
    def train(self, df: pd.DataFrame, progress_callback=None) -> Optional[Dict]:
        """Train final model and return results."""
        
        # First, cross-validate
        if progress_callback:
            progress_callback("Running cross-validation...")
        
        cv_results = self.cross_validate(df, progress_callback)
        
        if cv_results is None:
            return None
        
        self.cv_results = cv_results
        
        # Train final model on all data
        if progress_callback:
            progress_callback("Training final model...")
        
        data = self._prepare_data(df)
        X = data[self.available_features].values
        y = data['target'].values
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Train
        self.model = XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        self.model.fit(X_scaled, y)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.available_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Current prediction
        last_row = data[self.available_features].iloc[-1:].values
        last_scaled = self.scaler.transform(last_row)
        pred_log_return = self.model.predict(last_scaled)[0]
        pred_pct = (np.exp(pred_log_return) - 1) * 100
        
        # Add to results
        cv_results['predicted_log_return'] = pred_log_return
        cv_results['predicted_pct_return'] = pred_pct
        cv_results['current_volatility'] = data['rvol_20'].iloc[-1]
        cv_results['current_price'] = data['Close'].iloc[-1]
        cv_results['signal'] = 1 if pred_log_return > 0 else -1 if pred_log_return < 0 else 0
        
        return cv_results

# =============================================================================
# GARCH-LIKE MONTE CARLO (UNCHANGED FROM ORIGINAL)
# =============================================================================
def garch_monte_carlo(df: pd.DataFrame, n_sims: int, n_days: int, 
                      current_price: float, progress_callback=None) -> Optional[Dict]:
    """
    Monte Carlo with volatility clustering (more realistic).
    Uses a simplified GARCH(1,1) process.
    """
    returns = df['log_ret'].dropna().values
    
    if len(returns) < 100:
        return None
    
    if progress_callback:
        progress_callback("Running Monte Carlo simulation...")
    
    # GARCH parameters (simplified estimation)
    omega = 0.00001  # Long-run variance constant
    alpha = 0.10     # Shock impact
    beta = 0.85      # Persistence
    
    # Current variance estimate
    current_var = df['rvol_20'].iloc[-1] ** 2 / 252  # Daily variance
    
    # Standardized returns for bootstrapping
    std_returns = returns / (np.std(returns) + 1e-10)
    
    # Simulate paths
    paths = np.zeros((n_days + 1, n_sims))
    paths[0, :] = current_price
    
    variances = np.zeros((n_days, n_sims))
    variances[0, :] = current_var
    
    for t in range(n_days):
        # Sample standardized shocks
        shocks = np.random.choice(std_returns, size=n_sims)
        
        # Current volatility
        vol_t = np.sqrt(variances[t, :])
        
        # Returns
        daily_returns = shocks * vol_t
        
        # Update prices
        paths[t + 1, :] = paths[t, :] * np.exp(daily_returns)
        
        # Update variance (GARCH)
        if t < n_days - 1:
            variances[t + 1, :] = omega + alpha * (daily_returns ** 2) + beta * variances[t, :]
    
    # Calculate statistics
    final_prices = paths[-1, :]
    
    percentiles = {}
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        percentiles[f'p{p}'] = np.percentile(final_prices, p)
    
    # Path percentiles over time
    path_percentiles = {}
    for p in [5, 25, 50, 75, 95]:
        path_percentiles[f'p{p}'] = np.percentile(paths, p, axis=1)
    
    return {
        'paths': paths,
        'final_prices': final_prices,
        'percentiles': percentiles,
        'path_percentiles': path_percentiles,
        'expected_return': (np.mean(final_prices) / current_price - 1) * 100,
        'expected_vol': np.std(final_prices) / current_price * 100,
        'prob_profit': np.mean(final_prices > current_price) * 100,
        'var_95': (current_price - percentiles['p5']) / current_price * 100,
        'cvar_95': (current_price - np.mean(final_prices[final_prices <= percentiles['p5']])) / current_price * 100
    }

# =============================================================================
# PROFESSIONAL BACKTESTER (UNCHANGED FROM ORIGINAL)
# =============================================================================
class WalkForwardBacktester:
    """Walk-forward backtest with periodic retraining."""
    
    def __init__(self, config: TradingConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        
    def run(self, df: pd.DataFrame, forecast_horizon: int, 
            retrain_frequency: int = 60, progress_callback=None) -> Optional[Dict]:
        """
        Run walk-forward backtest.
        
        Args:
            df: Price data with features
            forecast_horizon: Days to predict
            retrain_frequency: Retrain model every N days
        """
        # Prepare data
        data = df.copy()
        data['target'] = data['log_ret'].rolling(forecast_horizon).sum().shift(-forecast_horizon)
        data = data.dropna()
        
        if len(data) < self.model_config.min_data_points:
            return None
        
        # Get features
        feature_cols = [f for f in ProfessionalModel.FEATURE_COLS if f in data.columns]
        
        # Split: 60% train, 40% test (walk-forward)
        train_end = int(len(data) * 0.6)
        
        if train_end < 252:  # Need at least 1 year for training
            return None
        
        # Initialize tracking
        equity = [self.config.initial_capital]
        returns_list = []
        positions = []
        signals = []
        dates = []
        trades = []
        
        current_capital = self.config.initial_capital
        peak_capital = current_capital
        position = 0
        entry_price = 0
        entry_date = None
        
        # Model and scaler
        model = None
        scaler = StandardScaler()
        last_train_idx = 0
        
        for i in range(train_end, len(data)):
            current_date = data.index[i]
            current_price = data['Close'].iloc[i]
            
            if progress_callback and i % 50 == 0:
                progress_callback(f"Backtesting... {i - train_end}/{len(data) - train_end} days")
            
            # Retrain model periodically
            if model is None or (i - last_train_idx) >= retrain_frequency:
                # Use all data up to current point for training
                train_data = data.iloc[:i]
                X_train = train_data[feature_cols].values
                y_train = train_data['target'].values
                
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                
                model = XGBRegressor(
                    n_estimators=self.model_config.n_estimators,
                    max_depth=self.model_config.max_depth,
                    learning_rate=self.model_config.learning_rate,
                    min_child_weight=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )
                model.fit(X_train_s, y_train)
                last_train_idx = i
            
            # Generate signal
            X_current = data[feature_cols].iloc[i:i+1].values
            X_current_s = scaler.transform(X_current)
            pred = model.predict(X_current_s)[0]
            
            # Volatility-adjusted threshold
            current_vol = data['rvol_20'].iloc[i] / np.sqrt(252) * np.sqrt(forecast_horizon)
            signal_threshold = current_vol * 0.5
            
            # Determine signal
            if pred > signal_threshold:
                signal = 1  # Long
            elif pred < -signal_threshold:
                signal = -1  # Short
            else:
                signal = 0  # Flat
            
            signals.append(signal)
            
            # Execute trades
            daily_pnl = 0
            
            if signal != position:
                # Close existing position
                if position != 0:
                    # Apply slippage
                    exit_price = current_price * (1 - self.config.slippage_pct * np.sign(position))
                    
                    # Calculate P&L
                    if position == 1:
                        trade_return = (exit_price - entry_price) / entry_price
                    else:
                        trade_return = (entry_price - exit_price) / entry_price
                    
                    # Subtract commission
                    trade_return -= self.config.commission_pct * 2  # Entry + exit
                    
                    # Apply to portfolio
                    position_size = self.config.max_position_pct
                    daily_pnl = trade_return * position_size * current_capital
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'direction': 'LONG' if position == 1 else 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': trade_return * 100,
                        'pnl': daily_pnl
                    })
                
                # Open new position
                if signal != 0:
                    entry_price = current_price * (1 + self.config.slippage_pct * np.sign(signal))
                    entry_date = current_date
                    position = signal
                else:
                    position = 0
            
            # Update capital
            current_capital += daily_pnl
            peak_capital = max(peak_capital, current_capital)
            
            # Track
            equity.append(current_capital)
            returns_list.append(daily_pnl / (equity[-2] if equity[-2] > 0 else 1))
            positions.append(position)
            dates.append(current_date)
        
        # Calculate metrics
        returns_arr = np.array(returns_list)
        equity_arr = np.array(equity[1:])  # Skip initial
        
        # Drawdown
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (running_max - equity_arr) / running_max
        
        # Performance metrics
        total_days = len(returns_arr)
        total_return = (current_capital / self.config.initial_capital - 1) * 100
        ann_return = ((1 + total_return / 100) ** (252 / max(total_days, 1)) - 1) * 100
        
        ann_vol = np.std(returns_arr) * np.sqrt(252) * 100
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        max_dd = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0
        
        # Trade statistics
        if trades:
            trade_returns = [t['return'] for t in trades]
            winning_trades = [t for t in trades if t['return'] > 0]
            losing_trades = [t for t in trades if t['return'] <= 0]
            
            win_rate = len(winning_trades) / len(trades) * 100
            avg_win = np.mean([t['return'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['return'] for t in losing_trades]) if losing_trades else 0
            
            total_wins = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
            total_losses = sum([t['pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(total_wins / total_losses) if total_losses != 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Calmar ratio
        calmar = ann_return / max_dd if max_dd > 0 else 0
        
        # Sortino ratio (downside deviation)
        negative_returns = returns_arr[returns_arr < 0]
        downside_std = np.std(negative_returns) * np.sqrt(252) * 100 if len(negative_returns) > 0 else 1
        sortino = ann_return / downside_std
        
        return {
            'dates': dates,
            'equity': equity[1:],
            'returns': returns_list,
            'positions': positions,
            'drawdowns': drawdowns.tolist(),
            'trades': trades,
            'metrics': {
                'total_return': total_return,
                'annualized_return': ann_return,
                'annualized_volatility': ann_vol,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar,
                'max_drawdown': max_dd,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'num_trades': len(trades),
                'trading_days': total_days
            }
        }

# =============================================================================
# DATA FETCHING
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(ticker: str, period: str = "5y") -> Optional[pd.DataFrame]:
    """Fetch and process stock data."""
    try:
        df = yf.Ticker(ticker).history(period=period)
        if len(df) < 200:
            return None
        return add_all_features(df)
    except Exception as e:
        return None

# =============================================================================
# VISUALIZATION (COMPLETE - UNCHANGED FROM ORIGINAL)
# =============================================================================
def create_analysis_dashboard(df: pd.DataFrame, cv_results: Dict, mc_results: Dict, 
                              bt_results: Optional[Dict], model: ProfessionalModel) -> None:
    """Create comprehensive analysis dashboard."""
    
    current_price = cv_results['current_price']
    ticker = st.session_state.get('ticker', 'Stock')
    
    # Header metrics
    st.subheader("📊 Model Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    da = cv_results['directional_accuracy'] * 100
    col1.metric(
        "Directional Accuracy",
        f"{da:.1f}%",
        f"{da - 50:.1f}% vs random",
        delta_color="normal" if da > 50 else "inverse"
    )
    
    col2.metric("CV RMSE", f"{cv_results['rmse_mean']:.4f}", f"±{cv_results['rmse_std']:.4f}")
    
    pred_ret = cv_results['predicted_pct_return']
    col3.metric(
        "Predicted Return",
        f"{pred_ret:+.2f}%",
        "LONG" if pred_ret > 0 else "SHORT" if pred_ret < 0 else "FLAT"
    )
    
    col4.metric("Ann. Volatility", f"{cv_results['current_volatility'] * 100:.1f}%")
    
    confidence = cv_results['confidence'] * 100
    col5.metric("Model Confidence", f"{confidence:.0f}%")
    
    # Edge detection
    if cv_results['has_edge']:
        st.success(f"✅ **Potential Edge Detected** - {da:.1f}% directional accuracy (>{52}% threshold)")
    else:
        st.warning(f"⚠️ **No Significant Edge** - {da:.1f}% directional accuracy ≈ random (50%)")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Monte Carlo", "📊 Backtest", "🔍 Features", "📋 Details"])
    
    with tab1:
        if mc_results:
            st.subheader("Monte Carlo Simulation (GARCH Volatility)")
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Expected Return", f"{mc_results['expected_return']:.2f}%")
            col2.metric("Prob. of Profit", f"{mc_results['prob_profit']:.1f}%")
            col3.metric("VaR (95%)", f"{mc_results['var_95']:.2f}%")
            col4.metric("CVaR (95%)", f"{mc_results['cvar_95']:.2f}%")
            
            # Percentile table
            st.markdown("**Price Percentiles at Horizon:**")
            pct = mc_results['percentiles']
            pct_df = pd.DataFrame({
                'Percentile': ['1%', '5%', '10%', '25%', '50%', '75%', '90%', '95%', '99%'],
                'Price': [pct['p1'], pct['p5'], pct['p10'], pct['p25'], pct['p50'], 
                         pct['p75'], pct['p90'], pct['p95'], pct['p99']],
                'Return': [(p / current_price - 1) * 100 for p in 
                          [pct['p1'], pct['p5'], pct['p10'], pct['p25'], pct['p50'],
                           pct['p75'], pct['p90'], pct['p95'], pct['p99']]]
            })
            st.dataframe(pct_df.style.format({'Price': '${:.2f}', 'Return': '{:+.2f}%'}),
                        use_container_width=True)
            
            # Path chart
            fig = go.Figure()
            
            path_pct = mc_results['path_percentiles']
            colors = {'p5': 'red', 'p25': 'orange', 'p50': 'blue', 'p75': 'green', 'p95': 'darkgreen'}
            
            for p, color in colors.items():
                fig.add_trace(go.Scatter(
                    y=path_pct[p],
                    mode='lines',
                    name=p.replace('p', '') + 'th %ile',
                    line=dict(color=color, width=2 if p == 'p50' else 1)
                ))
            
            # Add ML prediction
            ml_target = current_price * (1 + cv_results['predicted_pct_return'] / 100)
            fig.add_trace(go.Scatter(
                x=[len(path_pct['p50']) - 1],
                y=[ml_target],
                mode='markers',
                name='ML Target',
                marker=dict(size=15, symbol='star', color='purple')
            ))
            
            fig.update_layout(
                title='Price Projection with Percentile Bands',
                xaxis_title='Days',
                yaxis_title='Price ($)',
                height=450,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if bt_results:
            st.subheader("Walk-Forward Backtest Results")
            
            m = bt_results['metrics']
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", f"{m['total_return']:.2f}%")
            col2.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}")
            col3.metric("Max Drawdown", f"{m['max_drawdown']:.1f}%")
            col4.metric("Win Rate", f"{m['win_rate']:.1f}%")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Ann. Return", f"{m['annualized_return']:.2f}%")
            col2.metric("Sortino Ratio", f"{m['sortino_ratio']:.2f}")
            col3.metric("Profit Factor", f"{m['profit_factor']:.2f}")
            col4.metric("# Trades", f"{m['num_trades']}")
            
            # Interpretation
            if m['sharpe_ratio'] >= 1.0:
                st.success("✅ Good risk-adjusted returns (Sharpe ≥ 1.0)")
            elif m['sharpe_ratio'] >= 0.5:
                st.warning("🟡 Marginal performance (0.5 ≤ Sharpe < 1.0)")
            else:
                st.error("❌ Poor risk-adjusted returns (Sharpe < 0.5)")
            
            # Equity curve
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.7, 0.3],
                subplot_titles=('Equity Curve', 'Drawdown')
            )
            
            fig.add_trace(
                go.Scatter(x=bt_results['dates'], y=bt_results['equity'],
                          mode='lines', name='Equity', line=dict(color='blue', width=2)),
                row=1, col=1
            )
            
            fig.add_hline(y=100000, line_dash='dash', line_color='gray', row=1, col=1)
            
            fig.add_trace(
                go.Scatter(x=bt_results['dates'], y=np.array(bt_results['drawdowns']) * -100,
                          mode='lines', fill='tozeroy', name='Drawdown',
                          line=dict(color='red', width=1)),
                row=2, col=1
            )
            
            fig.update_layout(height=500, showlegend=True)
            fig.update_yaxes(title_text='$', row=1, col=1)
            fig.update_yaxes(title_text='DD %', row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade list
            if bt_results['trades']:
                with st.expander(f"📋 Trade Log ({len(bt_results['trades'])} trades)"):
                    trade_df = pd.DataFrame(bt_results['trades'])
                    st.dataframe(trade_df.style.format({
                        'entry_price': '${:.2f}',
                        'exit_price': '${:.2f}',
                        'return': '{:+.2f}%',
                        'pnl': '${:,.2f}'
                    }), use_container_width=True)
        else:
            st.info("Run backtest to see results")
    
    with tab3:
        st.subheader("Feature Importance")
        
        if model.feature_importance is not None:
            top_n = 15
            top_features = model.feature_importance.head(top_n)
            
            fig = go.Figure(go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker_color='steelblue'
            ))
            
            fig.update_layout(
                title=f'Top {top_n} Features by Importance',
                xaxis_title='Importance',
                yaxis_title='Feature',
                height=450,
                yaxis=dict(autorange='reversed')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Full table
            with st.expander("📊 All Feature Importances"):
                st.dataframe(model.feature_importance.style.format({'importance': '{:.4f}'}),
                            use_container_width=True)
    
    with tab4:
        st.subheader("Cross-Validation Details")
        
        # CV fold results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**RMSE by Fold:**")
            fold_df = pd.DataFrame({
                'Fold': [f'Fold {i+1}' for i in range(len(cv_results['rmse_scores']))],
                'RMSE': cv_results['rmse_scores'],
                'Dir. Accuracy': [f"{da*100:.1f}%" for da in cv_results['da_scores']]
            })
            st.dataframe(fold_df, use_container_width=True)
        
        with col2:
            st.markdown("**Feature Count:**")
            st.write(f"Total features used: {len(model.available_features)}")
            st.write(f"Data points: {len(df)}")
            st.write(f"Training period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    st.sidebar.title("⚙️ Configuration")
    
    mode = st.sidebar.radio("Mode", ["📈 Single Stock", "🔍 Scanner", "ℹ️ About"])
    
    if mode == "ℹ️ About":
        st.title("📊 Quantitative Trading System v3.1")
        
        st.markdown("""
        ## Features
        
        ### 🔬 Professional ML Pipeline
        - **38+ Features**: Volatility, momentum, volume, regime indicators
        - **Walk-Forward CV**: 5-fold time-series cross-validation
        - **XGBoost**: With regularization to prevent overfitting
        
        ### 📈 Monte Carlo Simulation
        - **GARCH Volatility**: Accounts for volatility clustering
        - **Full Distribution**: Percentile bands, VaR, CVaR
        
        ### 📊 Backtesting
        - **Walk-Forward**: Periodic model retraining
        - **Realistic Costs**: Commission + slippage
        - **Professional Metrics**: Sharpe, Sortino, Calmar, Profit Factor
        
        ---
        
        ## Scanner Reliability Modes
        
        | Mode | Min Data | CV Folds | Trees | Reliability |
        |------|----------|----------|-------|-------------|
        | ⚡ Fast | 1 year | 3 | 50 | ⚠️ Lower |
        | ⚖️ Balanced | 1.5 years | 4 | 75 | 🟡 Medium |
        | 🎯 Full | 2 years | 5 | 100 | ✅ Highest |
        
        ---
        
        ## Interpretation Guide
        
        | Metric | Poor | Marginal | Good |
        |--------|------|----------|------|
        | Dir. Accuracy | <52% | 52-55% | >55% |
        | Sharpe Ratio | <0.5 | 0.5-1.0 | >1.0 |
        | Max Drawdown | >30% | 15-30% | <15% |
        | Win Rate | <45% | 45-55% | >55% |
        
        ---
        
        ## ⚠️ Important Disclaimers
        
        1. **Past performance ≠ future results**
        2. Most models show NO edge (this is expected)
        3. Transaction costs eat most retail edges
        4. Use for education/research, not blind trading
        """)
        return
    
    if mode == "📈 Single Stock":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Stock Settings")
        
        ticker = st.sidebar.text_input("Ticker Symbol", "NVDA").upper()
        st.session_state['ticker'] = ticker
        
        forecast_days = st.sidebar.slider("Forecast Horizon (days)", 5, 60, 20)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Analysis Options")
        
        run_backtest = st.sidebar.checkbox("Run Backtest", value=True)
        mc_sims = st.sidebar.slider("Monte Carlo Simulations", 500, 5000, 1000)
        
        # Advanced settings
        with st.sidebar.expander("⚙️ Advanced"):
            cv_folds = st.slider("CV Folds", 3, 7, 5)
            n_estimators = st.slider("XGB Estimators", 50, 200, 100)
            retrain_freq = st.slider("Backtest Retrain Frequency (days)", 20, 120, 60)
        
        # Main area
        status_container = st.empty()
        progress_container = st.empty()
        
        # Fetch data
        status_container.info(f"📥 Fetching {ticker} data (5 years)...")
        df = get_stock_data(ticker, period="5y")
        
        if df is None:
            status_container.error(f"❌ Could not fetch data for {ticker}. Check the symbol.")
            return
        
        current_price = df['Close'].iloc[-1]
        status_container.empty()
        
        st.title(f"📈 {ticker} - ${current_price:.2f}")
        st.caption(f"Data: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')} ({len(df)} trading days)")
        
        # Train model
        def update_status(msg):
            progress_container.text(msg)
        
        status_container.info("🔄 Training model with cross-validation...")
        
        model_config = ModelConfig(n_estimators=n_estimators, cv_folds=cv_folds)
        model = ProfessionalModel(forecast_horizon=forecast_days, config=model_config)
        cv_results = model.train(df, progress_callback=update_status)
        
        status_container.empty()
        progress_container.empty()
        
        if cv_results is None:
            st.error("❌ Insufficient data for analysis (need at least 2 years)")
            return
        
        # Monte Carlo
        status_container.info("🎲 Running Monte Carlo simulation...")
        mc_results = garch_monte_carlo(df, mc_sims, forecast_days, current_price, update_status)
        status_container.empty()
        
        # Backtest
        bt_results = None
        if run_backtest:
            status_container.info("📊 Running walk-forward backtest...")
            
            trading_config = TradingConfig()
            backtester = WalkForwardBacktester(trading_config, model_config)
            bt_results = backtester.run(df, forecast_days, retrain_frequency=retrain_freq, 
                                       progress_callback=update_status)
            status_container.empty()
        
        progress_container.empty()
        
        # Display dashboard
        create_analysis_dashboard(df, cv_results, mc_results, bt_results, model)
    
    # =========================================================================
    # FIXED SCANNER MODE
    # =========================================================================
    elif mode == "🔍 Scanner":
        st.title("🔍 Market Scanner")
        st.markdown("Scan multiple stocks for potential trading opportunities.")
        
        # Ticker source selection
        st.sidebar.markdown("---")
        st.sidebar.subheader("Ticker Source")
        
        ticker_source = st.sidebar.selectbox(
            "Select Universe",
            ["S&P 500", "Nasdaq 100", "Popular Stocks", "Mega Caps Only", "ETFs Only", "All Combined", "Custom List"]
        )
        
        if ticker_source == "Custom List":
            custom_input = st.sidebar.text_area(
                "Enter tickers (comma or newline separated)",
                "AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AMD, JPM, V"
            )
            all_tickers = [t.strip().upper() for t in custom_input.replace('\n', ',').split(',') if t.strip()]
        else:
            all_tickers = get_ticker_list(ticker_source)
        
        st.sidebar.info(f"📊 {len(all_tickers)} tickers available from {ticker_source}")
        
        # Scan settings
        st.sidebar.markdown("---")
        st.sidebar.subheader("Scan Settings")
        
        max_stocks = st.sidebar.slider(
            "Stocks to Scan", 
            10, 
            min(len(all_tickers), 300), 
            min(50, len(all_tickers))
        )
        forecast_days = st.sidebar.slider("Forecast Horizon", 10, 60, 20)
        min_accuracy = st.sidebar.slider("Min Dir. Accuracy %", 50, 60, 52)
        
        # RELIABILITY SETTING
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚖️ Speed vs Reliability")
        
        reliability_mode = st.sidebar.select_slider(
            "Choose Mode",
            options=["⚡ Fast", "⚖️ Balanced", "🎯 Full"],
            value="⚖️ Balanced",
            help="Fast=quick but less reliable, Full=slower but most reliable"
        )
        
        # Set config based on mode
        if reliability_mode == "⚡ Fast":
            scan_config = ModelConfig.fast()
            data_period = "2y"
            st.sidebar.warning("⚠️ Fast mode: Results may be less reliable")
        elif reliability_mode == "⚖️ Balanced":
            scan_config = ModelConfig.balanced()
            data_period = "3y"
            st.sidebar.info("🟡 Balanced: Good speed/accuracy trade-off")
        else:  # Full
            scan_config = ModelConfig.full()
            data_period = "5y"
            st.sidebar.success("✅ Full: Most reliable results (slower)")
        
        # Show current config
        with st.sidebar.expander("📋 Current Config Details"):
            st.write(f"**Min data points:** {scan_config.min_data_points} days")
            st.write(f"**CV folds:** {scan_config.cv_folds}")
            st.write(f"**XGB estimators:** {scan_config.n_estimators}")
            st.write(f"**Data period:** {data_period}")
        
        # Parallel processing option
        use_parallel = st.sidebar.checkbox("Use Parallel Processing", value=True)
        if use_parallel:
            n_workers = st.sidebar.slider("Worker Threads", 2, 8, 4)
        
        # Start scan button
        if st.button("🚀 Start Scan", type="primary", use_container_width=True):
            tickers = all_tickers[:max_stocks]
            
            st.info(f"Scanning {len(tickers)} stocks from **{ticker_source}** using **{reliability_mode}** mode...")
            
            results = []
            failed_tickers = []
            
            progress_bar = st.progress(0)
            status = st.empty()
            live_results = st.empty()
            
            def scan_single_ticker(ticker: str) -> Dict:
                """Scan a single ticker and return results."""
                try:
                    df = get_stock_data(ticker, data_period)
                    
                    if df is None:
                        return {'ticker': ticker, 'success': False, 'error': 'No data available'}
                    
                    if len(df) < scan_config.min_data_points:
                        return {
                            'ticker': ticker, 
                            'success': False, 
                            'error': f'Insufficient data: {len(df)}/{scan_config.min_data_points} days'
                        }
                    
                    model = ProfessionalModel(
                        forecast_horizon=forecast_days,
                        config=scan_config
                    )
                    cv = model.train(df)
                    
                    if cv is None:
                        return {'ticker': ticker, 'success': False, 'error': 'Model training failed'}
                    
                    return {
                        'success': True,
                        'ticker': ticker,
                        'data': {
                            'Ticker': ticker,
                            'Price': cv['current_price'],
                            'Dir. Accuracy': cv['directional_accuracy'] * 100,
                            'Pred. Return': cv['predicted_pct_return'],
                            'Volatility': cv['current_volatility'] * 100,
                            'Signal': '🟢 LONG' if cv['signal'] == 1 else '🔴 SHORT' if cv['signal'] == -1 else '⚪ FLAT',
                            'Edge': '✅' if cv['has_edge'] else '❌',
                            'Confidence': cv['confidence'] * 100
                        }
                    }
                except Exception as e:
                    return {'ticker': ticker, 'success': False, 'error': str(e)[:50]}
            
            # Execute scan
            if use_parallel:
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    futures = {executor.submit(scan_single_ticker, t): t for t in tickers}
                    
                    for i, future in enumerate(as_completed(futures)):
                        ticker = futures[future]
                        
                        try:
                            result = future.result(timeout=120)
                            
                            if result.get('success'):
                                results.append(result['data'])
                            else:
                                failed_tickers.append({
                                    'Ticker': result['ticker'],
                                    'Reason': result.get('error', 'Unknown')
                                })
                        except Exception as e:
                            failed_tickers.append({'Ticker': ticker, 'Reason': str(e)[:50]})
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(tickers))
                        status.text(f"Scanned {i+1}/{len(tickers)} | ✅ {len(results)} | ❌ {len(failed_tickers)}")
                        
                        # Update live results table
                        if results and (i + 1) % 3 == 0:
                            temp_df = pd.DataFrame(results).sort_values('Dir. Accuracy', ascending=False)
                            live_results.dataframe(temp_df.head(10), use_container_width=True)
            else:
                # Sequential processing
                for i, ticker in enumerate(tickers):
                    status.text(f"Scanning {ticker}... ({i+1}/{len(tickers)})")
                    
                    result = scan_single_ticker(ticker)
                    
                    if result.get('success'):
                        results.append(result['data'])
                    else:
                        failed_tickers.append({
                            'Ticker': result['ticker'],
                            'Reason': result.get('error', 'Unknown')
                        })
                    
                    progress_bar.progress((i + 1) / len(tickers))
                    
                    # Update live results
                    if results and (i + 1) % 5 == 0:
                        temp_df = pd.DataFrame(results).sort_values('Dir. Accuracy', ascending=False)
                        live_results.dataframe(temp_df.head(10), use_container_width=True)
            
            # Clear progress indicators
            status.empty()
            live_results.empty()
            progress_bar.empty()
            
            # Display final results
            if results:
                df_results = pd.DataFrame(results)
                
                # Filter for edge stocks
                edge_stocks = df_results[df_results['Dir. Accuracy'] >= min_accuracy].sort_values(
                    'Dir. Accuracy', ascending=False
                )
                
                # Edge stocks section
                st.subheader(f"✅ Stocks with Potential Edge (≥{min_accuracy}% accuracy)")
                
                if len(edge_stocks) > 0:
                    st.dataframe(
                        edge_stocks.style.format({
                            'Price': '${:.2f}',
                            'Dir. Accuracy': '{:.1f}%',
                            'Pred. Return': '{:+.2f}%',
                            'Volatility': '{:.1f}%',
                            'Confidence': '{:.0f}%'
                        }).background_gradient(subset=['Dir. Accuracy'], cmap='RdYlGn', vmin=50, vmax=60),
                        use_container_width=True
                    )
                    
                    # Quick stats for edge stocks
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        longs = len(edge_stocks[edge_stocks['Signal'] == '🟢 LONG'])
                        st.metric("Long Signals", longs)
                    with col2:
                        shorts = len(edge_stocks[edge_stocks['Signal'] == '🔴 SHORT'])
                        st.metric("Short Signals", shorts)
                    with col3:
                        avg_acc = edge_stocks['Dir. Accuracy'].mean()
                        st.metric("Avg Accuracy", f"{avg_acc:.1f}%")
                else:
                    st.warning(f"No stocks found with ≥{min_accuracy}% directional accuracy")
                
                # All results section
                st.subheader("📊 All Scanned Results")
                st.dataframe(
                    df_results.sort_values('Dir. Accuracy', ascending=False).style.format({
                        'Price': '${:.2f}',
                        'Dir. Accuracy': '{:.1f}%',
                        'Pred. Return': '{:+.2f}%',
                        'Volatility': '{:.1f}%',
                        'Confidence': '{:.0f}%'
                    }),
                    use_container_width=True
                )
                
                # Summary statistics
                reliability_emoji = {"⚡ Fast": "⚠️", "⚖️ Balanced": "🟡", "🎯 Full": "✅"}[reliability_mode]
                
                st.success(f"""
                **Scan Complete** {reliability_emoji} ({reliability_mode} mode)
                
                - **Attempted:** {len(tickers)} stocks
                - **Successful:** {len(results)} stocks
                - **Failed:** {len(failed_tickers)} stocks
                - **With edge (≥{min_accuracy}%):** {len(edge_stocks)} stocks
                - **Average accuracy:** {df_results['Dir. Accuracy'].mean():.1f}%
                - **Best stock:** {df_results.iloc[df_results['Dir. Accuracy'].argmax()]['Ticker']} ({df_results['Dir. Accuracy'].max():.1f}%)
                """)
                
            else:
                st.error("❌ No valid results from scan. Try a different ticker source or reliability mode.")
            
            # Show failed tickers
            if failed_tickers:
                with st.expander(f"⚠️ Failed Tickers ({len(failed_tickers)})"):
                    failed_df = pd.DataFrame(failed_tickers)
                    st.dataframe(failed_df, use_container_width=True)
                    
                    # Group by failure reason
                    st.markdown("**Failure Reasons:**")
                    if 'Reason' in failed_df.columns:
                        # Simplify reasons for grouping
                        failed_df['Reason_Simple'] = failed_df['Reason'].apply(
                            lambda x: 'Insufficient data' if 'Insufficient' in str(x) or 'data' in str(x).lower()
                            else 'No data' if 'No data' in str(x) 
                            else 'Training failed' if 'Training' in str(x) or 'failed' in str(x)
                            else 'Other'
                        )
                        reason_counts = failed_df['Reason_Simple'].value_counts()
                        for reason, count in reason_counts.items():
                            st.write(f"- {reason}: {count} tickers")

if __name__ == "__main__":
    main()
