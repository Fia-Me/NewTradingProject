"""
Feature engineering module for creating trading features.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import warnings
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Handles feature engineering for trading strategies."""
    
    def __init__(self, feature_params: Optional[Dict] = None):
        """
        Initialize the feature engineer.
        
        Args:
            feature_params: Dictionary of feature parameters
        """
        self.feature_params = feature_params or {}
        self.feature_names = []
        logger.info(f"Initialized FeatureEngineer with params: {self.feature_params}")
        
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for the trading model."""
        try:
            # Suppress numerical warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                
                # Each method gets the original data
                temporal = self._create_temporal_features(data)
                volatility = self._create_volatility_features(data)
                momentum = self._create_momentum_features(data)
                mean_reversion = self._create_mean_reversion_features(data)
                microstructure = self._create_market_microstructure_features(data)
                
                # Combine all features
                features = pd.concat(
                    [temporal, volatility, momentum, mean_reversion, microstructure],
                    axis=1
                )
                
                # Fill NaN values with 0
                features = features.fillna(0)
                
                # Log if any NaNs remain
                if features.isna().any().any():
                    logger.warning("NaN values remain in features after filling.")
                
                # Log feature creation summary
                logger.info(f"Generated features shape: {features.shape}")
                logger.info(f"Feature columns: {list(features.columns)}")
                
                # Log basic statistics without using describe()
                stats = {
                    'mean': features.mean(),
                    'std': features.std(),
                    'min': features.min(),
                    'max': features.max()
                }
                logger.info("Feature statistics summary:")
                for stat_name, stat_values in stats.items():
                    logger.info(f"{stat_name}: {stat_values.to_dict()}")
                
                return features
                
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
    def _create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features."""
        features = pd.DataFrame(index=data.index)
        
        # Calculate returns
        features['returns'] = data['close'].pct_change()
        features['returns_ma'] = features['returns'].rolling(window=20).mean()
        
        # Calculate price moving average
        features['price_ma'] = data['close'].rolling(window=20).mean()
        
        # Calculate volatility
        features['volatility'] = data['close'].pct_change().rolling(window=20).std()
        features['volatility_ma'] = features['volatility'].rolling(window=20).mean()
        
        return features
        
    def _create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create volatility features."""
        features = pd.DataFrame(index=data.index)
        
        # Calculate realized volatility
        windows = self.feature_params.get('volatility_features', {}).get('realized_vol_window', 20)
        features['realized_vol'] = data['close'].pct_change().rolling(window=windows).std()
        
        # Calculate Parkinson volatility
        features['parkinson_vol'] = np.sqrt(
            (1.0 / (4.0 * np.log(2.0))) * 
            ((np.log(data['high'] / data['low'])) ** 2)
        ).rolling(window=windows).mean()
        
        # Calculate Garman-Klass volatility
        features['garman_klass_vol'] = np.sqrt(
            0.5 * np.log(data['high'] / data['low']) ** 2 -
            (2 * np.log(2) - 1) * np.log(data['close'] / data['open']) ** 2
        ).rolling(window=windows).mean()
        
        return features
        
    def _create_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create momentum features."""
        # Create a copy of the data to avoid modifying the original
        df = data.copy()
        
        # If the DataFrame has a MultiIndex, we need to handle it properly
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten the MultiIndex columns
            df.columns = df.columns.get_level_values(0)
        
        features = pd.DataFrame(index=df.index)
        
        # Calculate returns
        features['returns'] = df['close'].pct_change()
        
        # Calculate volatility
        features['volatility'] = features['returns'].rolling(window=20).std()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        
        # Calculate momentum
        features['momentum'] = df['close'].pct_change(periods=10)
        
        # Calculate ADX
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        # Directional Movement
        high_diff = df['high'].diff()
        low_diff = df['low'].diff()
        
        # Plus Directional Movement
        plus_dm = pd.Series(0.0, index=df.index)
        plus_dm[high_diff > 0] = high_diff[high_diff > 0]
        
        # Minus Directional Movement
        minus_dm = pd.Series(0.0, index=df.index)
        minus_dm[low_diff < 0] = -low_diff[low_diff < 0]
        
        # Smoothed True Range
        tr14 = true_range.rolling(window=14).sum()
        
        # Smoothed Plus Directional Movement
        plus_dm14 = plus_dm.rolling(window=14).sum()
        
        # Smoothed Minus Directional Movement
        minus_dm14 = minus_dm.rolling(window=14).sum()
        
        # Plus Directional Indicator
        plus_di14 = 100 * (plus_dm14 / tr14)
        
        # Minus Directional Indicator
        minus_di14 = 100 * (minus_dm14 / tr14)
        
        # Directional Index
        dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
        
        # Average Directional Index
        features['adx'] = dx.rolling(window=14).mean()
        
        # Calculate Bollinger Bands
        bb_window = 20
        bb_std = 2
        features['bb_middle'] = df['close'].rolling(window=bb_window).mean()
        bb_std_dev = df['close'].rolling(window=bb_window).std()
        features['bb_upper'] = features['bb_middle'] + (bb_std_dev * bb_std)
        features['bb_lower'] = features['bb_middle'] - (bb_std_dev * bb_std)
        
        # Calculate ATR
        features['atr'] = true_range.rolling(window=14).mean()
        
        # Calculate OBV
        features['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Calculate VWAP
        features['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Fill NaN values with 0
        features = features.fillna(0)
        
        return features
        
    def _create_mean_reversion_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create mean reversion features."""
        features = pd.DataFrame(index=data.index)
        
        # Calculate z-score
        zscore_window = self.feature_params.get('mean_reversion_features', {}).get('zscore_window', 20)
        rolling_mean = data['close'].rolling(window=zscore_window).mean()
        rolling_std = data['close'].rolling(window=zscore_window).std()
        features['zscore'] = (data['close'] - rolling_mean) / rolling_std
        
        # Calculate Hurst exponent
        hurst_window = self.feature_params.get('mean_reversion_features', {}).get('hurst_window', 100)
        features['hurst'] = data['close'].rolling(window=hurst_window).apply(
            lambda x: self._calculate_hurst(x)
        )
        
        # Calculate half-life
        half_life_window = self.feature_params.get('mean_reversion_features', {}).get('half_life_window', 20)
        features['half_life'] = data['close'].rolling(window=half_life_window).apply(
            lambda x: self._calculate_half_life(x)
        )
        
        return features
        
    def _create_market_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features."""
        features = pd.DataFrame(index=data.index)
        
        # Calculate volume moving averages
        volume_ma_windows = self.feature_params.get('market_microstructure', {}).get('volume_ma_windows', [5, 10, 20])
        for window in volume_ma_windows:
            features[f'volume_ma_{window}'] = data['volume'].rolling(window=window).mean()
            
        # Calculate VWAP
        vwap_window = self.feature_params.get('market_microstructure', {}).get('vwap_window', 20)
        features['vwap'] = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).rolling(
            window=vwap_window
        ).sum() / data['volume'].rolling(window=vwap_window).sum()
        
        # Calculate price-volume correlation
        features['price_volume_corr'] = data['close'].rolling(window=20).corr(data['volume'])
        
        return features
        
    def _calculate_hurst(self, prices: pd.Series) -> float:
        """Calculate Hurst exponent."""
        if len(prices) < 2:
            return np.nan
            
        # Calculate returns
        returns = np.log(prices / prices.shift(1))
        returns = returns.dropna()
        
        if len(returns) < 2:
            return np.nan
            
        # Calculate variance of returns
        variance = returns.var()
        
        # Calculate range of cumulative returns
        cumulative_returns = returns.cumsum()
        range_returns = cumulative_returns.max() - cumulative_returns.min()
        
        # Calculate Hurst exponent
        hurst = 0.5 * np.log(range_returns / np.sqrt(variance)) / np.log(len(returns))
        
        return hurst
        
    def _calculate_half_life(self, prices: pd.Series) -> float:
        """Calculate half-life of mean reversion."""
        if len(prices) < 2:
            return np.nan
            
        # Calculate price changes
        price_changes = prices.diff().dropna()
        price_levels = prices.shift(1).dropna()
        
        if len(price_changes) < 2:
            return np.nan
            
        # Calculate half-life
        try:
            half_life = -np.log(2) / np.polyfit(price_levels, price_changes, 1)[0]
            return half_life
        except:
            return np.nan
            
    def create_cross_sectional_features(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Create cross-sectional features across multiple symbols.
        
        Args:
            data_dict: Dictionary of DataFrames with market data
            
        Returns:
            Dictionary of DataFrames with cross-sectional features
        """
        features_dict = {}
        
        # Calculate cross-sectional statistics
        for symbol, data in data_dict.items():
            features = pd.DataFrame(index=data.index)
            
            # Calculate relative strength
            other_returns = pd.concat([
                df['close'].pct_change()
                for sym, df in data_dict.items()
                if sym != symbol
            ], axis=1).mean(axis=1)
            
            features['relative_strength'] = data['close'].pct_change() - other_returns
            
            # Calculate relative volume
            other_volume = pd.concat([
                df['volume']
                for sym, df in data_dict.items()
                if sym != symbol
            ], axis=1).mean(axis=1)
            
            features['relative_volume'] = data['volume'] / other_volume
            
            features_dict[symbol] = features
            
        return features_dict
        
    def create_market_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create market regime features.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with market regime features
        """
        features = pd.DataFrame(index=data.index)
        
        # Calculate volatility regime
        volatility = data['close'].pct_change().rolling(window=20).std()
        features['volatility_regime'] = pd.qcut(volatility, q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        # Calculate trend regime
        returns = data['close'].pct_change()
        trend = returns.rolling(window=50).mean()
        features['trend_regime'] = pd.qcut(trend, q=5, labels=['strong_down', 'down', 'neutral', 'up', 'strong_up'])
        
        # Calculate market regime
        features['market_regime'] = features['volatility_regime'].astype(str) + '_' + features['trend_regime'].astype(str)
        
        return features 