"""
Ensemble model architecture implementing non-linear modeling framework.
"""
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from transformers import AutoModel, AutoTokenizer
import optuna
from optuna import create_study, Trial
from .ising import IsingModel
from scipy import stats
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """Dataset for time series data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            X: Feature matrix with shape (n_samples, sequence_length, n_features)
            y: Target values with shape (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
        logger.info(f"Dataset initialized with X shape: {self.X.shape}, y shape: {self.y.shape}")
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (features, target)
        """
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """LSTM model for time series prediction."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input has shape (batch_size, sequence_length, input_size)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        return out
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions with the LSTM model."""
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x).to(next(self.parameters()).device)
            return self.forward(x)

class TransformerModel(nn.Module):
    """Transformer model for time series prediction."""
    
    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2
    ):
        super().__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input has shape (batch_size, sequence_length, input_size)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        # Project input to d_model dimensions
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create attention mask (optional)
        mask = None
        
        # Pass through transformer
        x = self.transformer_encoder(x, mask)
        
        # Take the last sequence output
        x = x[:, -1, :]
        
        # Apply dropout and project to output size
        x = self.dropout(x)
        x = self.output_proj(x)
        
        return x
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions with the Transformer model."""
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x).to(next(self.parameters()).device)
            return self.forward(x)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class EnsembleModel:
    """Ensemble model combining LSTM, Transformer, and XGBoost."""
    
    def __init__(
        self,
        input_size: int,
        sequence_length: int = 20,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.device = device
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing EnsembleModel with parameters:")
        self.logger.info(f"Input size: {input_size}")
        self.logger.info(f"Sequence length: {sequence_length}")
        self.logger.info(f"Device: {device}")
        
        # Initialize feature scaler
        self.feature_scaler = StandardScaler()
        
        # Initialize model weights (equal weights initially)
        self.model_weights = {
            'lstm': 0.33,
            'transformer': 0.33,
            'xgboost': 0.34
        }
        
        # Initialize models
        self.lstm = LSTMModel(
            input_size=input_size,  # Use original input_size for sequence models
            hidden_size=64,
            num_layers=2,
            output_size=1,
            dropout=0.2
        ).to(device)
        
        self.transformer = TransformerModel(
            input_size=input_size,  # Use original input_size for sequence models
            d_model=64,
            nhead=4,
            num_layers=2,
            output_size=1,
            dropout=0.1
        ).to(device)
        
        self.xgboost = xgb.XGBRegressor(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0
        )
        
        # Initialize Ising model
        try:
            from utils.config_loader import ConfigLoader
            config_loader = ConfigLoader()
            ising_params = config_loader.get_model_params('ising')
            self.ising_model = IsingModel(
                n_assets=1,  # or more, depending on your use case
                temperature=ising_params.get('temperature', 1.0),
                interaction_strength=ising_params.get('interaction_strength', 0.5),
                external_field=ising_params.get('external_field', 0.1)
            )
            self.logger.info(f"IsingModel initialized with params: {ising_params}")
        except Exception as e:
            self.logger.warning(f"Could not load Ising model parameters from config, using defaults. Error: {str(e)}")
            self.ising_model = IsingModel(
                n_assets=1,
                temperature=1.0,
                interaction_strength=0.5,
                external_field=0.1
            )
        
        self.logger.info("Models initialized successfully")
        
        # =============================
        # Load additional configuration for signal generation
        # =============================
        try:
            # Attempt to fetch strategy specific parameters for thresholds
            strategy_params = config_loader.get_strategy_params()
            self.signal_threshold = strategy_params.get('signal_threshold', 0.005)
            self.logger.info(f"Signal threshold set to: {self.signal_threshold}")
        except Exception as e:
            # Fallback to a hard-coded default if config section is missing
            self.logger.warning(f"Could not load strategy parameters from config. Using default signal_threshold. Error: {str(e)}")
            self.signal_threshold = 0.005  # 0.5 % as reasonable default

        # Signal and regime weights determine how different model components
        # and market factors are combined when producing the final trade signal.
        # They are optional in the config – provide sensible defaults if absent.
        self.signal_weights = ising_params.get(
            'signal_weights',
            {'ensemble': 0.7, 'ising': 0.3}
        )
        # Ensure the weights sum to 1 to avoid unintended scaling
        total_sw = sum(self.signal_weights.values())
        if total_sw == 0:
            # Avoid division by zero – reset to defaults
            self.logger.warning("Signal weights in config sum to 0. Resetting to default (ensemble=0.7, ising=0.3)")
            self.signal_weights = {'ensemble': 0.7, 'ising': 0.3}
        else:
            # Normalise so they always sum to 1
            self.signal_weights = {k: v / total_sw for k, v in self.signal_weights.items()}
        self.logger.info(f"Signal weights: {self.signal_weights}")

        # Regime weights used inside _calculate_market_factors()
        # Provide defaults covering all keys referenced in that method.
        default_regime_weights = {
            'volatility': 0.25,
            'trend': 0.25,
            'momentum': 0.25,
            'regime': 0.25
        }
        self.regime_weights = ising_params.get('regime_weights', default_regime_weights)
        # Ensure all required keys exist – fall back to default values where missing
        for k, v in default_regime_weights.items():
            if k not in self.regime_weights:
                self.logger.warning(f"Regime weight for '{k}' missing – defaulting to {v}")
                self.regime_weights[k] = v
        # Normalise regime weights as well
        total_rw = sum(self.regime_weights.values())
        if total_rw == 0:
            self.logger.warning("Regime weights sum to 0 – resetting to equal weights")
            self.regime_weights = default_regime_weights
            total_rw = sum(self.regime_weights.values())
        self.regime_weights = {k: v / total_rw for k, v in self.regime_weights.items()}
        self.logger.info(f"Regime weights: {self.regime_weights}")
        
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare data for training and validation."""
        logger.info("Starting data preparation...")
        logger.info(f"Initial X shape: {X.shape}, y shape: {y.shape}")
        
        # Prepare sequence features
        X_seq = self._prepare_sequence_features(X)
        logger.info(f"Sequence features shape: {X_seq.shape}")
        logger.info(f"Sequence features contains NaN: {np.isnan(X_seq).any()}")
        logger.info(f"Sequence features contains inf: {np.isinf(X_seq).any()}")
        
        # Prepare flat features
        X_flat = self._prepare_flat_features(X)
        logger.info(f"Flat features shape: {X_flat.shape}")
        logger.info(f"Flat features contains NaN: {np.isnan(X_flat).any()}")
        logger.info(f"Flat features contains inf: {np.isinf(X_flat).any()}")
        
        # Generate meta features
        X_meta = self._generate_meta_features(X)
        logger.info(f"Meta features shape: {X_meta.shape}")
        logger.info(f"Meta features contains NaN: {np.isnan(X_meta).any()}")
        logger.info(f"Meta features contains inf: {np.isinf(X_meta).any()}")
        
        # Adjust flat and meta features to match sequence length
        X_flat = X_flat[self.sequence_length-1:]
        X_meta = X_meta[self.sequence_length-1:]
        y = y[self.sequence_length-1:]
        
        logger.info(f"Adjusted shapes - X_flat: {X_flat.shape}, X_meta: {X_meta.shape}, y: {y.shape}")
        logger.info(f"Adjusted y contains NaN: {np.isnan(y).any()}")
        logger.info(f"Adjusted y contains inf: {np.isinf(y).any()}")
        
        # Combine features for XGBoost
        X_combined = np.hstack([X_flat, X_meta])
        logger.info(f"Combined features shape: {X_combined.shape}")
        logger.info(f"Combined features contains NaN: {np.isnan(X_combined).any()}")
        logger.info(f"Combined features contains inf: {np.isinf(X_combined).any()}")
        
        # Split data
        train_size = int(0.8 * len(X_seq))  # Use X_seq length for splitting
        X_train_seq = X_seq[:train_size]
        X_val_seq = X_seq[train_size:]
        y_train = y[:train_size]
        y_val = y[train_size:]
        
        logger.info(f"Final shapes - X_train_seq: {X_train_seq.shape}, y_train: {y_train.shape}")
        logger.info(f"Final shapes - X_val_seq: {X_val_seq.shape}, y_val: {y_val.shape}")
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train_seq, y_train)
        val_dataset = TimeSeriesDataset(X_val_seq, y_val)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        return train_loader, val_loader
        
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100
    ) -> Dict[str, float]:
        """Train the ensemble model."""
        self.logger.info("Starting model training...")
        
        # Fit feature scaler
        self.logger.info("Fitting feature scaler...")
        self.feature_scaler.fit(X)
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(X, y)
        
        # Train deep learning models
        self.logger.info("Training LSTM model...")
        self._train_deep_model(self.lstm, train_loader, val_loader, epochs)
        
        self.logger.info("Training Transformer model...")
        self._train_deep_model(self.transformer, train_loader, val_loader, epochs)
        
        # Train XGBoost model
        self.logger.info("Training XGBoost model...")
        # Prepare flat features
        X_flat = self._prepare_flat_features(X)
        # Generate meta features
        X_meta = self._generate_meta_features(X)
        # Combine features
        X_combined = np.hstack([X_flat, X_meta])
        # Adjust target for sequence length
        y_xgb = y[self.sequence_length-1:]
        # Ensure features and target have same length
        min_length = min(len(X_combined), len(y_xgb))
        X_combined = X_combined[:min_length]
        y_xgb = y_xgb[:min_length]
        
        self.logger.info(f"XGBoost training data shapes - X: {X_combined.shape}, y: {y_xgb.shape}")
        self.xgboost.fit(X_combined, y_xgb)
        
        self.logger.info("Model training completed")
        
        # Return training metrics
        return self.evaluate(X, y)
        
    def _train_deep_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int
    ) -> None:
        """Train a deep learning model."""
        self.logger.info(f"Training {model.__class__.__name__}...")
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                    val_loss += loss.item()
                    
            # Log progress
            self.logger.info(f"Epoch {epoch+1}/{epochs}:")
            self.logger.info(f"Training loss: {train_loss/len(train_loader):.4f}")
            self.logger.info(f"Validation loss: {val_loss/len(val_loader):.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info("Early stopping triggered")
                    break
                    
        self.logger.info(f"{model.__class__.__name__} training completed")
        
    def _prepare_flat_features(self, X: np.ndarray) -> np.ndarray:
        """Prepare features for XGBoost model."""
        try:
            # Reshape to 2D array
            n_samples = X.shape[0]
            n_features = X.shape[1] * X.shape[2] if len(X.shape) == 3 else X.shape[1]
            X_flat = X.reshape(n_samples, n_features)
            
            # Add meta-features
            X_meta = self._generate_meta_features(X_flat)
            
            self.logger.info(f"Prepared flat features shape: {X_meta.shape}")
            return X_meta
            
        except Exception as e:
            self.logger.error(f"Error preparing flat features: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
        
    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features from input data."""
        try:
            # Convert to numpy array if it's a pandas DataFrame
            if isinstance(X, pd.DataFrame):
                X = X.values
            
            # Calculate basic statistics
            mean = np.mean(X, axis=1, keepdims=True)
            std = np.std(X, axis=1, keepdims=True)
            min_val = np.min(X, axis=1, keepdims=True)
            max_val = np.max(X, axis=1, keepdims=True)
            
            # Calculate additional features
            range_val = max_val - min_val
            zscore = (X - mean) / (std + 1e-8)  # Add small epsilon to avoid division by zero
            
            # Combine all features
            meta_features = np.hstack([
                mean, std, min_val, max_val, range_val,
                np.mean(zscore, axis=1, keepdims=True),
                np.std(zscore, axis=1, keepdims=True)
            ])
            
            return meta_features
            
        except Exception as e:
            self.logger.error(f"Error generating meta-features: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Return zeros as fallback
            return np.zeros((X.shape[0], 7))
        
    def _prepare_sequence_features(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare features for sequence models (LSTM and Transformer).
        
        Args:
            X: Input features array
            
        Returns:
            Array of sequence features with shape (n_samples, sequence_length, input_size)
        """
        try:
            # Ensure input is 2D
            if len(X.shape) == 3:
                n_samples, seq_len, n_features = X.shape
                X = X.reshape(n_samples * seq_len, n_features)
            
            # Initialize scaler if not exists
            if self.feature_scaler is None:
                from sklearn.preprocessing import StandardScaler
                self.feature_scaler = StandardScaler()
                self.feature_scaler.fit(X)
            
            # Scale features
            X_scaled = self.feature_scaler.transform(X)
            
            # Reshape back to 3D if needed
            if len(X.shape) == 3:
                X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)
            
            # Create sequences using rolling window
            sequences = []
            for i in range(len(X_scaled) - self.sequence_length + 1):
                sequence = X_scaled[i:i + self.sequence_length]
                sequences.append(sequence)
            
            # If we don't have enough data for a full sequence, pad with the last value
            if len(sequences) == 0:
                last_sequence = np.tile(X_scaled[-1:], (self.sequence_length, 1))
                sequences.append(last_sequence)
            
            # Convert to numpy array and ensure correct shape
            sequences = np.array(sequences)
            logger.info(f"Prepared sequence features shape: {sequences.shape}")
            
            return sequences
            
        except Exception as e:
            logger.error(f"Error preparing sequence features: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        self.logger.info("Evaluating model performance...")
        
        # Prepare features for each model
        X_seq = self._prepare_sequence_features(X)
        X_flat = self._prepare_flat_features(X)
        X_meta = self._generate_meta_features(X)
        
        # Combine features for XGBoost
        X_combined = np.hstack([X_flat, X_meta])
        
        # Adjust target for sequence length
        y_adjusted = y[self.sequence_length-1:]
        
        # Ensure features and target have same length
        min_length = min(len(X_combined), len(y_adjusted))
        X_combined = X_combined[:min_length]
        y_adjusted = y_adjusted[:min_length]
        
        # Validate data
        if np.isnan(X_combined).any() or np.isinf(X_combined).any():
            self.logger.warning("Features contain NaN or inf values, replacing with 0")
            X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)
            
        if np.isnan(y_adjusted).any() or np.isinf(y_adjusted).any():
            self.logger.warning("Target contains NaN or inf values, replacing with 0")
            y_adjusted = np.nan_to_num(y_adjusted, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get predictions from each model
        lstm_pred = self.lstm.predict(torch.FloatTensor(X_seq).to(self.device))
        transformer_pred = self.transformer.predict(torch.FloatTensor(X_seq).to(self.device))
        xgb_pred = self.xgboost.predict(X_combined)
        
        # Convert predictions to numpy arrays and ensure correct shape
        lstm_pred = lstm_pred.cpu().numpy().reshape(-1, 1)
        transformer_pred = transformer_pred.cpu().numpy().reshape(-1, 1)
        xgb_pred = xgb_pred.reshape(-1, 1)
        
        # Ensure all predictions have same length
        min_len = min(len(lstm_pred), len(transformer_pred), len(xgb_pred), len(y_adjusted))
        lstm_pred = lstm_pred[:min_len]
        transformer_pred = transformer_pred[:min_len]
        xgb_pred = xgb_pred[:min_len]
        y_adjusted = y_adjusted[:min_len]
        
        # Log shapes for debugging
        self.logger.info(f"Shapes - y_adjusted: {y_adjusted.shape}, lstm_pred: {lstm_pred.shape}, "
                        f"transformer_pred: {transformer_pred.shape}, xgb_pred: {xgb_pred.shape}")
        
        # Calculate ensemble prediction using model weights
        ensemble_pred = (
            self.model_weights['lstm'] * lstm_pred +
            self.model_weights['transformer'] * transformer_pred +
            self.model_weights['xgboost'] * xgb_pred
        )
        
        # Ensure ensemble_pred has same shape as y_adjusted
        ensemble_pred = ensemble_pred.reshape(-1, 1)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_adjusted, ensemble_pred),
            'mae': mean_absolute_error(y_adjusted, ensemble_pred),
            'r2': r2_score(y_adjusted, ensemble_pred)
        }
        
        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics
        
    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'lstm_hidden_size': trial.suggest_int('lstm_hidden_size', 32, 128),
                'lstm_num_layers': trial.suggest_int('lstm_num_layers', 1, 3),
                'transformer_d_model': trial.suggest_int('transformer_d_model', 32, 128),
                'transformer_nhead': trial.suggest_int('transformer_nhead', 2, 8),
                'xgb_n_estimators': trial.suggest_int('xgb_n_estimators', 50, 200),
                'xgb_max_depth': trial.suggest_int('xgb_max_depth', 3, 8)
            }
            
            # Update model architectures
            self.lstm = LSTMModel(
                input_size=self.input_size,
                hidden_size=params['lstm_hidden_size'],
                num_layers=params['lstm_num_layers'],
                output_size=1
            ).to(self.device)
            
            self.transformer = TransformerModel(
                input_size=self.input_size,
                d_model=params['transformer_d_model'],
                nhead=params['transformer_nhead'],
                num_layers=2,
                output_size=1
            ).to(self.device)
            
            self.xgboost = xgb.XGBRegressor(
                n_estimators=params['xgb_n_estimators'],
                max_depth=params['xgb_max_depth'],
                learning_rate=0.1
            )
            
            # Train and evaluate
            metrics = self.train(X, y, epochs=50)
            return metrics['mse']
            
        # Create and run study
        study = create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params 

    def _to_scalar(self, val):
        """Utility to convert a pandas Series, numpy array, or scalar to a float."""
        import numpy as np
        import pandas as pd
        if isinstance(val, (pd.Series, np.ndarray)):
            if len(val) == 1:
                return float(val.item())
            elif len(val) > 1:
                self.logger.warning(f"Expected scalar, got multiple values: {val}. Using the first value.")
                if isinstance(val, pd.Series):
                    return float(val.iloc[0])
                else:
                    return float(val[0])
            else:
                self.logger.warning("Empty Series/array encountered, returning 0.")
                return 0.0
        return float(val)

    def generate_signals(self, data: Union[pd.Series, pd.DataFrame]) -> int:
        """Generate trading signals."""
        try:
            self.logger.info(f"Generating signals with input data shape: {data.shape}")
            self.logger.info(f"Input data columns: {data.columns.tolist()}")
            
            # Prepare features
            X = self._prepare_flat_features(data.values)
            self.logger.info(f"Prepared features shape: {X.shape}")
            
            # Get ensemble prediction
            try:
                ensemble_pred = self.predict(X)
                self.logger.info(f"Raw ensemble prediction shape: {ensemble_pred.shape}")
                self.logger.info(f"Raw ensemble prediction values: {ensemble_pred}")
            except Exception as e:
                self.logger.error(f"Error making ensemble prediction: {str(e)}")
                self.logger.error(traceback.format_exc())
                ensemble_pred = np.zeros(len(data))
            
            # Get Ising prediction
            try:
                ising_pred = self.ising_model.generate_signals(data)
                self.logger.info(f"Ising prediction shape: {ising_pred.shape}")
                self.logger.info(f"Ising prediction values: {ising_pred}")
            except Exception as e:
                self.logger.error(f"Error getting Ising prediction: {str(e)}")
                self.logger.error(traceback.format_exc())
                ising_pred = np.zeros(len(data))
            
            # Combine predictions using configurable weights. Use .get to avoid KeyErrors.
            combined_pred = (
                self.signal_weights.get('ensemble', 0.5) * ensemble_pred +
                self.signal_weights.get('ising', 0.5) * ising_pred
            )
            self.logger.info(f"Combined prediction shape: {combined_pred.shape}")
            self.logger.info(f"Combined prediction values: {combined_pred}")
            
            # Calculate market factors
            try:
                market_factors = self._calculate_market_factors(data)
                self.logger.info(f"Market factors: {market_factors}")
            except Exception as e:
                self.logger.error(f"Error calculating market factors: {str(e)}")
                self.logger.error(traceback.format_exc())
                market_factors = 1.0
            
            # Calculate threshold
            base_threshold = self.signal_threshold
            final_threshold = base_threshold * market_factors
            self.logger.info(f"Base threshold: {base_threshold}")
            self.logger.info(f"Final threshold: {final_threshold}")
            
            # Convert latest prediction to a scalar for safe comparison
            try:
                latest_pred = float(np.squeeze(combined_pred[-1]))
            except Exception:
                self.logger.warning("Could not convert latest combined prediction to float – defaulting to 0.0")
                latest_pred = 0.0

            signal = 0
            if latest_pred > final_threshold:
                signal = 1
            elif latest_pred < -final_threshold:
                signal = -1
                
            self.logger.info(f"Generated signal: {signal}")
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            self.logger.error(traceback.format_exc())
            return 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the ensemble model."""
        try:
            self.logger.info("Making predictions with ensemble model...")
            
            # Convert to numpy array if it's a pandas DataFrame
            if isinstance(X, pd.DataFrame):
                X = X.values
            
            # Prepare sequence features
            X_seq = self._prepare_sequence_features(X)
            self.logger.info(f"Prepared sequence features shape: {X_seq.shape}")
            
            # Generate meta-features
            X_meta = self._generate_meta_features(X)
            
            # Make predictions with each model
            lstm_pred = self.lstm.predict(torch.FloatTensor(X_seq).to(self.device))
            transformer_pred = self.transformer.predict(torch.FloatTensor(X_seq).to(self.device))
            xgb_pred = self.xgboost.predict(X_meta)
            
            # Combine predictions
            ensemble_pred = (
                self.model_weights['lstm'] * lstm_pred +
                self.model_weights['transformer'] * transformer_pred +
                self.model_weights['xgboost'] * xgb_pred
            )
            
            return ensemble_pred
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            self.logger.error(traceback.format_exc())
            return np.zeros(len(X))

    def _calculate_market_factors(self, data: pd.DataFrame) -> float:
        """Calculate market factors that affect signal thresholds."""
        try:
            # Calculate volatility
            returns = data['returns'] if 'returns' in data.columns else data['close'].pct_change()
            volatility = returns.std()
            
            # Calculate trend strength using ADX
            adx = data['adx'].iloc[-1] if 'adx' in data.columns else 0
            
            # Calculate momentum using RSI
            rsi = data['rsi'].iloc[-1] if 'rsi' in data.columns else 50
            
            # Calculate market regime
            regime = 1.0
            if 'regime' in data.columns:
                regime = data['regime'].iloc[-1]
            
            # Combine factors
            volatility_factor = 1.0 / (1.0 + volatility)  # Reduce signals in high volatility
            trend_factor = adx / 100.0 if adx > 0 else 0.5  # Reduce signals in weak trends
            momentum_factor = 1.0 - abs(rsi - 50) / 50.0  # Reduce signals in extreme RSI
            regime_factor = regime if isinstance(regime, (int, float)) else 1.0
            
            # Calculate final factor
            market_factor = (
                self.regime_weights['volatility'] * volatility_factor +
                self.regime_weights['trend'] * trend_factor +
                self.regime_weights['momentum'] * momentum_factor +
                self.regime_weights['regime'] * regime_factor
            )
            
            self.logger.info(f"Market factors calculation:")
            self.logger.info(f"  Volatility: {volatility:.4f}")
            self.logger.info(f"  ADX: {adx:.2f}")
            self.logger.info(f"  RSI: {rsi:.2f}")
            self.logger.info(f"  Regime: {regime}")
            self.logger.info(f"  Final market factor: {market_factor:.4f}")
            
            return market_factor
            
        except Exception as e:
            self.logger.error(f"Error calculating market factors: {str(e)}")
            self.logger.error(traceback.format_exc())
            return 1.0  # Return neutral factor on error 