"""
Configuration loader for the trading system.
"""
import os
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Handles loading and validation of configuration settings."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.load_config()  # Load config immediately
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration settings
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"Configuration file not found: {self.config_path}, using default settings")
                self.config = self._get_default_config()
                return self.config
                
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            self._validate_config()
            logger.info("Configuration loaded successfully")
            
            return self.config
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {str(e)}")
            self.config = self._get_default_config()
            return self.config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            self.config = self._get_default_config()
            return self.config
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration settings."""
        return {
            'feature_engineering': {
                'temporal_features': {
                    'windows': [5, 10, 20, 50, 100, 200],
                    'include_returns': True,
                    'include_volatility': True,
                    'include_momentum': True,
                    'include_mean_reversion': True
                },
                'volatility_features': {
                    'garch_window': 20,
                    'realized_vol_window': 20,
                    'parkinson_vol_window': 20,
                    'garman_klass_vol_window': 20
                },
                'momentum_features': {
                    'rsi_periods': [14, 21, 50],
                    'macd_params': {
                        'fast_period': 12,
                        'slow_period': 26,
                        'signal_period': 9
                    },
                    'bollinger_bands': {
                        'window': 20,
                        'num_std': 2
                    }
                },
                'mean_reversion_features': {
                    'zscore_window': 20,
                    'hurst_window': 100,
                    'half_life_window': 20
                },
                'market_microstructure': {
                    'volume_ma_windows': [5, 10, 20],
                    'vwap_window': 20,
                    'order_imbalance_window': 20
                }
            }
        }
        
    def _validate_config(self) -> None:
        """
        Validate configuration settings.
        
        Raises:
            ValueError: If required settings are missing or invalid
        """
        required_sections = [
            'data_collection',
            'feature_engineering',
            'model',
            'strategy',
            'risk_management',
            'backtesting',
            'performance_metrics',
            'logging'
        ]
        
        # Check required sections
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
                
        # Validate data collection settings
        if not self.config['data_collection'].get('symbols'):
            raise ValueError("No trading symbols specified")
            
        # Validate model parameters
        model_params = self.config['model']['ensemble']
        if not all(key in model_params for key in ['lstm', 'transformer', 'xgboost']):
            raise ValueError("Missing required model configurations")
            
        # Validate risk management parameters
        risk_params = self.config['risk_management']
        if not all(key in risk_params for key in ['position_limits', 'risk_limits', 'correlation']):
            raise ValueError("Missing required risk management configurations")
            
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a specific configuration section.
        
        Args:
            section: Name of the configuration section
            
        Returns:
            Dictionary containing section settings
            
        Raises:
            KeyError: If section doesn't exist
        """
        if section not in self.config:
            raise KeyError(f"Configuration section not found: {section}")
            
        return self.config[section]
        
    def update_config(self, section: str, key: str, value: Any) -> None:
        """
        Update a specific configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: New value
            
        Raises:
            KeyError: If section or key doesn't exist
        """
        if section not in self.config:
            raise KeyError(f"Configuration section not found: {section}")
            
        if key not in self.config[section]:
            raise KeyError(f"Configuration key not found: {section}.{key}")
            
        self.config[section][key] = value
        logger.info(f"Updated configuration: {section}.{key} = {value}")
        
    def save_config(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            path: Optional path to save configuration
        """
        save_path = path or self.config_path
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Configuration saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            raise
            
    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """
        Get parameters for a specific model type.
        
        Args:
            model_type: Type of model ('lstm', 'transformer', or 'xgboost')
            
        Returns:
            Dictionary of model parameters
            
        Raises:
            KeyError: If model type doesn't exist
        """
        if model_type not in self.config['model']['ensemble']:
            raise KeyError(f"Model type not found: {model_type}")
            
        return self.config['model']['ensemble'][model_type]
        
    def get_feature_params(self, feature_type: str) -> Dict[str, Any]:
        """
        Get parameters for a specific feature type.
        
        Args:
            feature_type: Type of feature ('all' for all features)
            
        Returns:
            Dictionary of feature parameters
            
        Raises:
            KeyError: If feature type doesn't exist
        """
        if feature_type == 'all':
            return self.config.get('feature_engineering', {})
            
        if 'feature_engineering' not in self.config:
            logger.warning("Feature engineering section not found in config, using default settings")
            return self._get_default_config()['feature_engineering']
            
        if feature_type not in self.config['feature_engineering']:
            logger.warning(f"Feature type {feature_type} not found, using default settings")
            return self._get_default_config()['feature_engineering'].get(feature_type, {})
            
        return self.config['feature_engineering'][feature_type]
        
    def get_risk_params(self) -> Dict[str, Any]:
        """
        Get risk management parameters.
        
        Returns:
            Dictionary of risk parameters
        """
        return self.config['risk_management']
        
    def get_strategy_params(self) -> Dict[str, Any]:
        """
        Get strategy parameters.
        
        Returns:
            Dictionary of strategy parameters
        """
        return self.config['strategy']['renaissance'] 