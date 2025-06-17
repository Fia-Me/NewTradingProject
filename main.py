"""
Main script for running the Renaissance Technologies-inspired trading system.
"""
import logging
import yfinance as yf
import yaml
from features.engineer import FeatureEngineer
from models.ensemble import EnsembleModel
from backtesting.framework import BacktestFramework
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Load configuration
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Collect data for AAPL
        symbol = "AAPL"
        data = yf.download(symbol, start="2023-01-01", end="2024-01-01")
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Convert column names to lowercase and ensure no duplicates
        data = data.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        })
        
        # Remove any duplicate columns
        data = data.loc[:, ~data.columns.duplicated()]
        
        logger.info(f"Collected data for {symbol}")
        logger.info(f"Data columns: {data.columns.tolist()}")

        # Define required features
        required_features = [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'volatility', 'rsi', 'macd', 'macd_signal',
            'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'obv',
            'vwap', 'momentum', 'roc', 'williams_r', 'cci',
            'stoch_k', 'stoch_d', 'adx', 'mfi', 'trix',
            'aroon_up', 'aroon_down', 'ppo', 'roc_ma', 'mfi_ma',
            'rsi_ma', 'macd_ma', 'bb_width', 'bb_pct', 'atr_ratio',
            'volume_ma', 'price_ma', 'volatility_ma', 'momentum_ma',
            'returns_ma', 'vwap_ma'
        ]

        # Engineer features with configuration
        feature_engineer = FeatureEngineer(feature_params=config['feature_engineering'])
        features = feature_engineer.create_features(data)
        
        if features is None or features.empty:
            logger.error("Feature engineering failed to produce valid features.")
            return
            
        logger.info(f"Engineered features for {symbol}: {features.shape}")
        logger.info(f"Feature columns: {features.columns.tolist()}")

        # Initialize ensemble model with the correct number of features
        ensemble_model = EnsembleModel(input_size=len(required_features))

        # Train the model
        logger.info("Starting model training...")
        
        # Ensure all required features are present and in the correct order
        for feature in required_features:
            if feature not in features.columns:
                logger.warning(f"Missing required feature: {feature}")
                features.loc[:, feature] = 0.0
        
        # Select only the required features in the correct order
        X = features[required_features].copy()
        
        # Calculate 5-day future returns as target
        y = data['close'].pct_change(periods=5).shift(-5)
        
        # Align indices and handle NaN values
        X, y = X.align(y, join='inner', axis=0)
        y = y.dropna()
        X = X.loc[y.index]
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        
        # Convert to numpy arrays
        X = X.values
        y = y.values
        
        # Validate data before training
        if np.isnan(X).any() or np.isnan(y).any():
            logger.error("NaN values found in training data")
            return
            
        if np.isinf(X).any() or np.isinf(y).any():
            logger.error("Infinite values found in training data")
            return
            
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Target statistics - Mean: {np.mean(y):.4f}, Std: {np.std(y):.4f}")
        ensemble_model.train(X, y)
        logger.info("Model training completed successfully")

        # Prepare data for backtest
        backtest_data = data.copy()
        backtest_features = features.copy()
        
        # Ensure all required features are present in backtest data
        for feature in required_features:
            if feature not in backtest_features.columns:
                backtest_features.loc[:, feature] = 0.0
        
        # Select only the required features for backtesting
        backtest_features = backtest_features[required_features].copy()
        
        # Handle any NaN values in backtest data
        backtest_features = backtest_features.fillna(0)
        
        # Combine original data with features, ensuring no duplicate columns
        backtest_data = pd.concat([backtest_data, backtest_features], axis=1)
        backtest_data = backtest_data.loc[:, ~backtest_data.columns.duplicated()]
        
        logger.info(f"Backtest data columns: {backtest_data.columns.tolist()}")
        
        # Run backtest
        logger.info("Starting backtest...")
        backtest = BacktestFramework(
            initial_capital=100000,
            transaction_cost=0.001,
            slippage=0.001,
            risk_free_rate=0.02
        )
        results = backtest.run_backtest(backtest_data, ensemble_model)
        logger.info("Backtest completed")
        logger.info(f"Backtest results: {results}")
        
        # Plot results
        logger.info("Plotting backtest results...")
        backtest.plot_results()

    except Exception as e:
        logger.error(f"Error running trading system: {str(e)}")
        raise

if __name__ == "__main__":
    main() 