"""
Test script to verify the installation and basic functionality of the trading system.
"""
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from features.engineer import FeatureEngineer
from models.ensemble import EnsembleModel
from risk.manager import RiskManager
from backtesting.framework import BacktestFramework
from utils.config_loader import ConfigLoader
from models.ising import IsingModel
import traceback
import sys
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_setup.log')
    ]
)
logger = logging.getLogger(__name__)

def test_data_collection():
    """Test data collection using yfinance."""
    logger.info("Testing data collection...")
    
    # Download sample data
    symbol = "AAPL"
    data = yf.download(symbol, start="2023-01-01", end="2024-01-01")
    
    assert not data.empty, "Data collection failed"
    logger.info(f"Successfully collected data for {symbol}")
    
    # Standardize column names to lowercase
    data = data.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Adj Close': 'adj_close',
        'Volume': 'volume'
    })
    
    return data

def test_feature_engineering(data):
    """Test feature engineering process."""
    logger.info("Testing feature engineering...")
    try:
        # Initialize config loader and get feature parameters
        config_loader = ConfigLoader()
        feature_params = config_loader.get_feature_params('all')
        
        # Initialize feature engineer with parameters
        feature_engineer = FeatureEngineer(feature_params=feature_params)
        
        # Create features
        features = feature_engineer.create_features(data)
        
        # Verify features
        required_features = [
            'returns', 'volatility', 'rsi', 'macd', 'momentum', 'adx',
            'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'obv', 'vwap'
        ]
        
        # Check for required features
        missing_features = [f for f in required_features if f not in features.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            raise ValueError(f"Missing required features: {missing_features}")
            
        # Check for NaN values
        nan_features = features.columns[features.isna().any()].tolist()
        if nan_features:
            logger.warning(f"Features with NaN values: {nan_features}")
            raise ValueError(f"NaN values found in features: {nan_features}")
            
        # Log feature statistics
        logger.info("Feature statistics:")
        logger.info(features[required_features].describe())
        
        logger.info("Feature engineering test passed successfully")
        return features
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def test_ising_model(data):
    """Test Ising model functionality."""
    logger.info("Testing Ising model...")
    try:
        # Create returns DataFrame with proper index
        returns = pd.DataFrame(index=data.index)
        returns['close'] = data['close'].pct_change()
        returns = returns.dropna()
        
        logger.info(f"Returns DataFrame shape: {returns.shape}")
        logger.info(f"Returns DataFrame columns: {returns.columns.tolist()}")
        logger.info(f"Returns DataFrame index type: {type(returns.index)}")
        logger.info(f"Returns DataFrame head:\n{returns.head()}")
        
        # Initialize Ising model
        ising_model = IsingModel(
            n_assets=1,
            temperature=1.0,
            interaction_strength=0.5,
            external_field=0.1
        )
        logger.info("Ising model initialized successfully")
        
        # Fit interactions
        logger.info("Fitting interactions...")
        ising_model.fit_interactions(returns, window=20)
        logger.info("Interactions fitted successfully")
        
        # Calculate market regime
        logger.info("Calculating market regime...")
        regime = ising_model.calculate_market_regime(returns, window=20)
        logger.info(f"Market regime calculated: {regime}")
        
        # Generate signals
        logger.info("Generating signals...")
        logger.info(f"Input returns shape: {returns.shape}")
        signals = ising_model.generate_signals(returns, window=20)
        logger.info(f"Signals generated. Shape: {signals.shape if hasattr(signals, 'shape') else 'N/A'}")
        logger.info(f"Signal type: {type(signals)}")
        logger.info(f"First few signals: {signals[:5] if hasattr(signals, '__getitem__') else signals}")
        
        # Validate signals
        if isinstance(signals, np.ndarray):
            logger.info(f"Signal statistics - Mean: {np.mean(signals):.4f}, Std: {np.std(signals):.4f}")
            logger.info(f"Signal distribution - Min: {np.min(signals):.4f}, Max: {np.max(signals):.4f}")
        
        return signals
        
    except Exception as e:
        logger.error(f"Ising model failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def test_model_training(features, data):
    """Test model training and prediction."""
    try:
        logger.info("Starting model training test...")
        logger.info(f"Input features shape: {features.shape}")
        logger.info(f"Input data shape: {data.shape}")
        
        # Validate input data
        if features is None or data is None:
            logger.error("Input data is None")
            raise ValueError("Input data is None")
            
        if features.empty or data.empty:
            logger.error("Input data is empty")
            raise ValueError("Input data is empty")
            
        # Log feature statistics
        logger.info("Feature statistics:")
        logger.info(features.describe())
        
        # Initialize model with correct input size
        input_size = len(features.columns)
        logger.info(f"Initializing EnsembleModel with input_size={input_size}")
        model = EnsembleModel(input_size=input_size)
        
        # Prepare target variable (5-day future returns)
        logger.info("Preparing target variable...")
        target = data['close'].pct_change(periods=5).shift(-5)
        logger.info(f"Target shape: {target.shape}")
        logger.info(f"Target statistics: mean={float(target.mean()):.4f}, std={float(target.std()):.4f}")
        
        # Align features and target
        logger.info("Aligning features and target...")
        features, target = features.align(target, join='inner', axis=0)
        logger.info(f"Aligned features shape: {features.shape}")
        logger.info(f"Aligned target shape: {target.shape}")
        
        # Handle NaN values
        logger.info("Handling NaN values...")
        features = features.fillna(0)
        target = target.fillna(0)
        
        # Convert to numpy arrays
        logger.info("Converting to numpy arrays...")
        X = features.values
        y = target.values
        
        # Validate data before training
        logger.info("Validating data before training...")
        if np.isnan(X).any() or np.isnan(y).any():
            logger.error("NaN values found in training data")
            raise ValueError("NaN values found in training data")
            
        if np.isinf(X).any() or np.isinf(y).any():
            logger.error("Infinite values found in training data")
            raise ValueError("Infinite values found in training data")
            
        # Train model
        logger.info("Starting model training...")
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        model.train(X, y)
        logger.info("Model training completed")
        
        # Test prediction
        logger.info("Testing model prediction...")
        test_data = features.iloc[-20:].values  # Use last 20 days for testing
        logger.info(f"Test data shape: {test_data.shape}")
        
        try:
            prediction = model.predict(test_data)
            logger.info(f"Prediction shape: {prediction.shape if hasattr(prediction, 'shape') else 'N/A'}")
            logger.info(f"Prediction type: {type(prediction)}")
            logger.info(f"Prediction value: {prediction}")
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
        # Test signal generation
        logger.info("Testing signal generation...")
        try:
            signal = model.generate_signals(features.iloc[-1])
            logger.info(f"Generated signal: {signal}")
        except Exception as e:
            logger.error(f"Error during signal generation: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
        logger.info("Model training test completed successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error in test_model_training: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def test_risk_management():
    """Test risk management functionality."""
    logger.info("Testing risk management...")
    try:
        risk_manager = RiskManager(
            max_position_size=0.15,  # 15% of portfolio
            max_sector_exposure=0.30,  # 30% of portfolio
            max_leverage=1.5,  # 150% leverage
            max_correlation=0.7,  # 70% correlation limit
            max_daily_loss=0.02,  # 2% daily loss limit
            position_holding_limit=10  # 10 days max holding
        )
        logger.info("Risk management initialized successfully")
        return risk_manager
    except Exception as e:
        logger.error(f"Risk management failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def test_backtesting(data, features, model):
    """Test backtesting framework."""
    logger.info("Testing backtesting framework...")
    
    # Initialize backtesting framework
    backtest = BacktestFramework(
        initial_capital=100000,
        transaction_cost=0.001,
        slippage=0.0005,
        risk_free_rate=0.02
    )
    
    # Combine data and features for backtesting
    backtest_data = pd.concat([data, features], axis=1)
    backtest_data = backtest_data.loc[:, ~backtest_data.columns.duplicated()]

    # If columns are MultiIndex, flatten them
    if isinstance(backtest_data.columns, pd.MultiIndex):
        backtest_data.columns = ['_'.join(map(str, col)).strip('_') for col in backtest_data.columns.values]

    # Utility to robustly recover columns
    def find_column(cols, col):
        for c in cols:
            if isinstance(c, str) and c.startswith(col):
                return c
            if isinstance(c, tuple) and len(c) > 0 and isinstance(c[0], str) and c[0].startswith(col):
                return c
        return None

    # Ensure required columns are present and correctly named
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in backtest_data.columns:
            candidate = find_column(backtest_data.columns, col)
            if candidate is not None:
                backtest_data[col] = backtest_data[candidate]
            else:
                raise ValueError(f"Required column '{col}' is missing from backtest_data.")

    # Run backtest
    results = backtest.run_backtest(
        data=backtest_data,
        strategy=model
    )
    
    assert results is not None, "Backtesting failed"
    logger.info("Successfully completed backtest")
    return results

def main():
    """Run all tests."""
    try:
        logger.info("Starting test suite...")
        
        # Test data collection
        logger.info("Testing data collection...")
        try:
            data = test_data_collection()
            logger.info(f"Data collection successful. Shape: {data.shape}")
        except Exception as e:
            logger.error(f"Data collection failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Test feature engineering
        logger.info("Testing feature engineering...")
        try:
            features = test_feature_engineering(data)
            logger.info(f"Feature engineering successful. Features created: {len(features.columns)}")
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Test Ising model
        logger.info("Testing Ising model...")
        try:
            signals = test_ising_model(data)
            logger.info(f"Ising model successful. Signal shape: {signals.shape}")
        except Exception as e:
            logger.error(f"Ising model failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Test model training
        logger.info("Testing model training...")
        try:
            model = test_model_training(features, data)
            logger.info("Model training successful")
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Test risk management
        logger.info("Testing risk management...")
        try:
            risk_manager = test_risk_management()
            logger.info("Risk management initialization successful")
        except Exception as e:
            logger.error(f"Risk management failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Test backtesting
        logger.info("Testing backtesting...")
        try:
            results = test_backtesting(data, features, model)
            logger.info(f"Backtesting successful. Results: {results}")
        except Exception as e:
            logger.error(f"Backtesting failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 