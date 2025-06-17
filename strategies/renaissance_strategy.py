"""
Trading strategy implementing Renaissance Technologies-inspired approach.
"""
from typing import Dict, List, Optional, Union
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from models.ensemble import EnsembleModel
from features.engineer import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RenaissanceStrategy:
    """Trading strategy inspired by Renaissance Technologies."""
    
    def __init__(
        self,
        feature_engineer: FeatureEngineer,
        ensemble_model: EnsembleModel,
        lookback_period: int = 252,  # 1 year
        prediction_horizon: int = 5,  # 5 days
        signal_threshold: float = 0.02,  # 2% threshold
        position_size_limit: float = 0.02  # 2% of portfolio
    ):
        self.feature_engineer = feature_engineer
        self.ensemble_model = ensemble_model
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.signal_threshold = signal_threshold
        self.position_size_limit = position_size_limit
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized RenaissanceStrategy with parameters:")
        self.logger.info(f"Lookback period: {lookback_period}")
        self.logger.info(f"Prediction horizon: {prediction_horizon}")
        self.logger.info(f"Signal threshold: {signal_threshold}")
        self.logger.info(f"Position size limit: {position_size_limit}")
        
        # Performance tracking
        self.trades = []
        self.positions = {}
        self.performance_metrics = {}
        
    def train(self, data: pd.DataFrame) -> None:
        """
        Train the strategy on historical data.
        
        Args:
            data: DataFrame with market data
        """
        self.logger.info("Starting strategy training...")
        self.logger.info(f"Training data shape: {data.shape}")
        
        # Create features
        features = self.feature_engineer.create_features(data)
        self.logger.info(f"Generated features shape: {features.shape}")
        self.logger.info(f"Feature columns: {features.columns.tolist()}")
        
        # Prepare target
        target = self._prepare_target(data)
        self.logger.info(f"Target shape: {target.shape}")
        self.logger.info(f"Target statistics: {target.describe()}")
        
        # Train ensemble model
        self.logger.info("Training ensemble model...")
        self.ensemble_model.train(features.values, target.values)
        self.logger.info("Ensemble model training completed")
        
    def _prepare_target(self, data: pd.DataFrame) -> pd.Series:
        """
        Prepare target variable for training.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Series of target values
        """
        self.logger.info("Preparing target variable...")
        
        # Calculate future returns
        future_returns = data['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        self.logger.info(f"Future returns shape: {future_returns.shape}")
        self.logger.info(f"Future returns statistics: {future_returns.describe()}")
        
        # Create binary classification target
        target = (future_returns > self.signal_threshold).astype(int)
        self.logger.info(f"Target distribution: {target.value_counts(normalize=True)}")
        
        return target
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Generate trading signals based on model predictions.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary mapping timestamps to signals
        """
        self.logger.info("Generating trading signals...")
        self.logger.info(f"Input data shape: {data.shape}")
        
        # Create features
        features = self.feature_engineer.create_features(data)
        self.logger.info(f"Generated features shape: {features.shape}")
        
        # Make predictions
        predictions = self.ensemble_model.predict(features.values)
        self.logger.info(f"Predictions shape: {predictions.shape}")
        self.logger.info(f"Prediction statistics: {pd.Series(predictions).describe()}")
        
        # Generate signals
        signals = {}
        for symbol in data.columns.get_level_values(0).unique():
            symbol_data = data[symbol]
            symbol_pred = predictions[symbol]
            
            signal = self._generate_signal(symbol_pred, symbol_data)
            signals[symbol] = signal
            
            self.logger.info(f"Generated signal for {symbol}: {signal}")
            
        # Log signal distribution
        signal_dist = pd.Series(signals).value_counts()
        self.logger.info(f"Signal distribution: {signal_dist}")
        
        return signals
        
    def _generate_signal(self, prediction: float, data: pd.Series) -> float:
        """
        Generate trading signal from prediction.
        
        Args:
            prediction: Model prediction
            data: Current market data
            
        Returns:
            Trading signal (-1 to 1)
        """
        self.logger.info(f"Generating signal for prediction: {prediction}")
        
        # Calculate additional metrics
        volatility = data['close'].pct_change().std()
        trend = data['close'].pct_change(20).mean()
        
        self.logger.info(f"Current volatility: {volatility}")
        self.logger.info(f"Current trend: {trend}")
        
        # Generate signal based on prediction and market conditions
        if abs(prediction) < self.signal_threshold:
            signal = 0
            self.logger.info("Signal below threshold, no trade")
        else:
            signal = np.sign(prediction)
            self.logger.info(f"Generated signal: {signal}")
            
        return signal
        
    def update_positions(
        self,
        signals: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Update current positions based on signals.
        
        Args:
            signals: Dictionary of trading signals
            current_prices: Dictionary of current prices
            
        Returns:
            Dictionary of position updates
        """
        self.logger.info("Updating positions...")
        self.logger.info(f"Current signals: {signals}")
        self.logger.info(f"Current prices: {current_prices}")
        
        # Update positions
        for symbol, signal in signals.items():
            if signal != 0:
                price = current_prices[symbol]
                size = signal * self.position_size_limit
                
                self.logger.info(f"Updating position for {symbol}:")
                self.logger.info(f"Signal: {signal}")
                self.logger.info(f"Price: {price}")
                self.logger.info(f"Size: {size}")
                
                self.positions[symbol] = {
                    'size': size,
                    'price': price,
                    'timestamp': pd.Timestamp.now()
                }
                
        return self.positions
        
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate strategy performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        self.logger.info("Calculating performance metrics...")
        
        if not self.trades:
            self.logger.warning("No trades to calculate metrics")
            return {}
            
        # Calculate metrics
        returns = pd.Series([t['return'] for t in self.trades])
        self.logger.info(f"Trade returns statistics: {returns.describe()}")
        
        metrics = {
            'total_return': returns.sum(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'win_rate': (returns > 0).mean(),
            'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()),
            'max_drawdown': (returns.cumsum() - returns.cumsum().cummax()).min()
        }
        
        self.logger.info("Performance metrics:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value}")
            
        return metrics
        
    def analyze_feature_importance(self) -> pd.DataFrame:
        """
        Analyze feature importance.
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise RuntimeError("Strategy must be trained before analyzing features")
            
        # Get feature importance from ensemble model
        importance = self.ensemble_model.models['xgb'].feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_engineer.feature_names,
            'importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
        
    def generate_trade_report(self) -> Dict[str, Union[float, int]]:
        """
        Generate trade report.
        
        Returns:
            Dictionary of trade statistics
        """
        if not self.trades:
            return {}
            
        # Calculate trade statistics
        num_trades = len(self.trades)
        avg_trade_size = pd.Series([t['size'] for t in self.trades]).abs().mean()
        max_trade_size = pd.Series([t['size'] for t in self.trades]).abs().max()
        
        # Calculate prediction statistics
        prediction_accuracy = (pd.Series([t['return'] for t in self.trades]) > 0).mean()
        avg_prediction = pd.Series([t['return'] for t in self.trades]).mean()
        prediction_volatility = pd.Series([t['return'] for t in self.trades]).std()
        
        return {
            'num_trades': num_trades,
            'avg_trade_size': avg_trade_size,
            'max_trade_size': max_trade_size,
            'prediction_accuracy': prediction_accuracy,
            'avg_prediction': avg_prediction,
            'prediction_volatility': prediction_volatility
        }
        
    def optimize_parameters(
        self,
        data: pd.DataFrame,
        param_grid: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Optimize strategy parameters.
        
        Args:
            data: DataFrame with market data
            param_grid: Dictionary of parameter grids
            
        Returns:
            Dictionary of optimal parameters
        """
        best_params = {}
        best_sharpe = -np.inf
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        for params in param_combinations:
            # Update parameters
            self.signal_threshold = params['signal_threshold']
            self.position_size_limit = params['position_size_limit']
            
            # Train strategy
            self.train(data)
            
            # Calculate performance
            metrics = self.calculate_performance_metrics()
            
            if metrics['win_rate'] * metrics['avg_signal'] > best_sharpe:
                best_sharpe = metrics['win_rate'] * metrics['avg_signal']
                best_params = params
                
        return best_params
        
    def _generate_param_combinations(
        self,
        param_grid: Dict[str, List[float]]
    ) -> List[Dict[str, float]]:
        """
        Generate parameter combinations.
        
        Args:
            param_grid: Dictionary of parameter grids
            
        Returns:
            List of parameter combinations
        """
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))
            
        return combinations 