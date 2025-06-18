"""
Ising Model implementation for market regime detection and signal generation.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import logging
from scipy import stats
from sklearn.cluster import KMeans
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IsingModel:
    def __init__(
        self,
        n_assets: int = 1,
        temperature: float = 1.0,
        interaction_strength: float = 0.5,
        external_field: float = 0.1
    ):
        """
        Initialize the Ising model.
        
        Args:
            n_assets: Number of assets to model
            temperature: System temperature
            interaction_strength: Strength of interactions between assets
            external_field: External field strength
        """
        self.n_assets = n_assets
        self.temperature = temperature
        self.interaction_strength = interaction_strength
        self.external_field = external_field
        self.interaction_matrix = np.zeros((n_assets, n_assets))
        self.regime_history = []
        self.regime_metrics = {}
        logger.info(f"Initialized IsingModel with n_assets={n_assets}, temperature={temperature}, "
                   f"interaction_strength={interaction_strength}, external_field={external_field}")
        
    def calculate_energy(self, state: np.ndarray) -> float:
        """Calculate the energy of a given state."""
        if len(state) != self.n_assets:
            raise ValueError(f"State size {len(state)} does not match n_assets {self.n_assets}")
            
        energy = 0.0
        
        # For single asset case
        if self.n_assets == 1:
            # Ensure state is a scalar for single asset
            if isinstance(state, np.ndarray):
                if len(state.shape) > 1:
                    # Multi-dimensional array, flatten and take first element
                    state_val = float(state.flatten()[0]) if state.size > 0 else 0.0
                else:
                    # 1D array, take first element safely
                    state_val = float(state.item()) if state.size == 1 else float(state[0])
            else:
                state_val = float(state)
            energy = -self.external_field * state_val
            return energy
            
        # For multiple assets case
        for i in range(self.n_assets):
            energy -= self.external_field * state[i]
            for j in range(i + 1, self.n_assets):
                energy += self.interaction_matrix[i, j] * state[i] * state[j]
                
        return energy
        
    def metropolis_step(self, state: np.ndarray) -> np.ndarray:
        """Perform one step of the Metropolis algorithm."""
        new_state = state.copy()
        for i in range(self.n_assets):
            # Calculate energy difference
            old_energy = self.calculate_energy(new_state)
            new_state[i] *= -1
            new_energy = self.calculate_energy(new_state)
            delta_energy = new_energy - old_energy
            
            # Metropolis acceptance criterion
            if delta_energy > 0 and np.random.random() > np.exp(-delta_energy / self.temperature):
                new_state[i] *= -1
                
        return new_state
        
    def simulate(self, n_steps: int = 1000) -> np.ndarray:
        """Simulate the Ising model for n_steps."""
        state = np.random.choice([-1, 1], size=self.n_assets)
        for _ in range(n_steps):
            state = self.metropolis_step(state)
        return state
        
    def fit_interactions(
        self,
        returns: pd.DataFrame,
        window: int = 20
    ) -> None:
        """Fit interaction matrix using historical returns."""
        # Calculate correlation matrix
        corr_matrix = returns.rolling(window=window).corr()
        
        # Convert correlations to interaction strengths
        self.interaction_matrix = np.tanh(corr_matrix.fillna(0).values)
        
        # Normalize interaction strengths
        self.interaction_matrix = self.interaction_matrix * self.interaction_strength
        
    def calculate_market_regime(
        self,
        returns: pd.DataFrame,
        window: int = 20
    ) -> Dict[str, float]:
        """
        Calculate sophisticated market regime indicators.
        
        Args:
            returns: DataFrame of asset returns
            window: Rolling window size
            
        Returns:
            Dictionary of regime indicators
        """
        # Calculate basic regime metrics
        volatility = returns.std(axis=1).rolling(window=window).mean()
        momentum = returns.mean(axis=1).rolling(window=window).mean()
        correlation = returns.rolling(window=window).corr().mean(axis=1)
        
        # Calculate Ising model metrics
        magnetization = self._calculate_magnetization(returns)
        susceptibility = self._calculate_susceptibility(returns)
        energy = self._calculate_system_energy(returns)
        
        # Calculate regime clusters
        regime_clusters = self._identify_regime_clusters(returns)
        
        # Calculate regime stability
        stability = self._calculate_regime_stability(returns)
        
        # Calculate regime transitions
        transitions = self._detect_regime_transitions(returns)
        
        # Calculate regime characteristics
        characteristics = self._calculate_regime_characteristics(returns)
        
        # Combine all metrics
        regime_metrics = {
            'volatility': volatility.iloc[-1],
            'momentum': momentum.iloc[-1],
            'correlation': correlation.iloc[-1],
            'magnetization': magnetization,
            'susceptibility': susceptibility,
            'energy': energy,
            'regime_cluster': regime_clusters,
            'stability': stability,
            'transition_probability': transitions['probability'],
            'regime_duration': transitions['duration'],
            'regime_characteristics': characteristics
        }
        
        # Store regime history
        self.regime_history.append(regime_metrics)
        self.regime_metrics = regime_metrics
        
        return regime_metrics
        
    def _extract_returns_for_assets(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Extract returns for the configured number of assets consistently.
        
        Args:
            returns: DataFrame of returns (could have more columns than n_assets)
            
        Returns:
            Array of returns with shape matching n_assets
        """
        try:
            # If returns is a Series, convert to DataFrame
            if isinstance(returns, pd.Series):
                returns = returns.to_frame()
            
            # Get the most recent return values
            latest_returns = returns.iloc[-1].values
            
            # Handle the case where we have more features than assets
            if len(latest_returns) > self.n_assets:
                if self.n_assets == 1:
                    # For single asset, try to find 'close' or 'returns' column, or use first column
                    if 'returns' in returns.columns:
                        latest_returns = returns['returns'].iloc[-1:].values
                    elif 'close' in returns.columns:
                        latest_returns = returns['close'].pct_change().iloc[-1:].values
                    else:
                        latest_returns = latest_returns[:1]  # Use first feature
                else:
                    latest_returns = latest_returns[:self.n_assets]
                    
                logger.info(f"Extracted {len(latest_returns)} features for {self.n_assets} assets")
            
            # Ensure we have exactly n_assets values
            if len(latest_returns) < self.n_assets:
                # Pad with zeros if needed
                padding = np.zeros(self.n_assets - len(latest_returns))
                latest_returns = np.concatenate([latest_returns, padding])
                logger.warning(f"Padded returns from {len(latest_returns)} to {self.n_assets} values")
            
            return latest_returns[:self.n_assets]
            
        except Exception as e:
            logger.error(f"Error extracting returns for assets: {str(e)}")
            # Return safe fallback
            return np.zeros(self.n_assets)

    def _calculate_magnetization(self, returns: pd.DataFrame) -> float:
        """Calculate magnetization (average market state)."""
        try:
            # Use consistent asset extraction
            asset_returns = self._extract_returns_for_assets(returns)
            state = np.sign(asset_returns)
            return float(np.mean(state))
        except Exception as e:
            logger.warning(f"Error calculating magnetization: {str(e)}")
            return 0.0
        
    def _calculate_susceptibility(self, returns: pd.DataFrame) -> float:
        """Calculate susceptibility (market sensitivity to changes)."""
        try:
            # Use consistent asset extraction
            asset_returns = self._extract_returns_for_assets(returns)
            state = np.sign(asset_returns)
            return float(np.var(state) / self.temperature)
        except Exception as e:
            logger.warning(f"Error calculating susceptibility: {str(e)}")
            return 0.0
        
    def _calculate_system_energy(self, returns: pd.DataFrame) -> float:
        """Calculate total system energy."""
        try:
            # Use consistent asset extraction
            asset_returns = self._extract_returns_for_assets(returns)
            state = np.sign(asset_returns)
            return float(self.calculate_energy(state))
        except Exception as e:
            logger.warning(f"Error calculating system energy: {str(e)}")
            return 0.0
        
    def _identify_regime_clusters(
        self,
        returns: pd.DataFrame,
        n_clusters: int = 3
    ) -> int:
        """Identify market regime clusters using K-means."""
        try:
            # Input validation
            if returns.empty:
                raise ValueError("Input returns DataFrame is empty")
            
            # Calculate features with proper shape handling
            features_list = []
            
            # Mean returns
            mean_returns = returns.mean(axis=1)
            if len(mean_returns) == 0:
                logger.warning("Empty mean returns, using default value")
                mean_returns = pd.Series([0.0])
            features_list.append(mean_returns.fillna(0).values.reshape(-1, 1))
            
            # Standard deviation of returns  
            std_returns = returns.std(axis=1)
            if len(std_returns) == 0:
                logger.warning("Empty std returns, using default value")
                std_returns = pd.Series([0.0])
            features_list.append(std_returns.fillna(0).values.reshape(-1, 1))
            
            # Calculate rolling correlation with proper handling
            try:
                # Use only the window size available
                available_window = min(20, len(returns))
                if available_window < 2:
                    logger.warning("Insufficient data for correlation calculation")
                    corr_mean = pd.Series([0.0])
                else:
                    corr = returns.rolling(window=available_window, min_periods=1).corr()
                    if corr.isna().any().any():
                        logger.warning("NaN values found in correlation matrix, filling with 0")
                        corr = corr.fillna(0)
                    
                    # Calculate mean correlation, handling potential issues
                    corr_mean = corr.mean(axis=1)
                    if corr_mean.empty:
                        corr_mean = pd.Series([0.0])
                        
                features_list.append(corr_mean.fillna(0).values.reshape(-1, 1))
                
            except Exception as e:
                logger.warning(f"Error calculating correlation: {str(e)}, using default")
                corr_mean = pd.Series([0.0] * len(mean_returns))
                features_list.append(corr_mean.values.reshape(-1, 1))
            
            # Ensure all features have the same length
            min_length = min(len(f) for f in features_list)
            if min_length == 0:
                logger.warning("All features have zero length, using defaults")
                return 0
                
            # Trim all features to same length
            features_list = [f[:min_length] for f in features_list]
            
            # Stack features horizontally
            try:
                features = np.hstack(features_list)
                logger.info(f"Stacked features shape: {features.shape}")
            except Exception as e:
                logger.error(f"Error stacking features: {str(e)}")
                logger.error(f"Feature shapes: {[f.shape for f in features_list]}")
                # Return default cluster
                return 0
            
            # Handle edge cases
            if features.size == 0:
                logger.warning("Empty features array, returning default cluster")
                return 0
            
            if len(features) < n_clusters:
                logger.warning(f"Not enough samples ({len(features)}) for {n_clusters} clusters")
                n_clusters = max(1, len(features))
            
            # Handle single sample case
            if len(features) == 1:
                return 0
            
            # Ensure no NaN or inf values
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Perform clustering with error handling
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(features)
                return int(clusters[-1])
            except Exception as e:
                logger.warning(f"Clustering failed: {str(e)}, returning default cluster")
                return 0
                
        except Exception as e:
            logger.error(f"Error in regime cluster identification: {str(e)}")
            logger.error(traceback.format_exc())
            return 0
        
    def _calculate_regime_stability(self, returns: pd.DataFrame) -> float:
        """Calculate the stability of the current market regime."""
        try:
            # Calculate regime indicators
            volatility = returns['close'].rolling(window=20).std()
            trend = returns['close'].rolling(window=50).mean()
            
            # Create regime series (1 for high volatility/trend, 0 for low)
            regime_series = pd.Series(0, index=returns.index)
            regime_series[(volatility > volatility.quantile(0.7)) | 
                         (trend.abs() > trend.abs().quantile(0.7))] = 1
            
            # Calculate autocorrelation
            if len(regime_series) > 1:
                return float(regime_series.autocorr())
            return 0.0
        except Exception as e:
            logger.warning(f"Error calculating regime stability: {str(e)}")
            return 0.0
        
    def _detect_regime_transitions(
        self,
        returns: pd.DataFrame
    ) -> Dict[str, float]:
        """Detect regime transitions and calculate transition probabilities."""
        if len(self.regime_history) < 2:
            return {'probability': 0.0, 'duration': 0}
            
        # Calculate transition probability
        current_regime = self.regime_history[-1]['regime_cluster']
        previous_regime = self.regime_history[-2]['regime_cluster']
        transition_prob = 1.0 if current_regime != previous_regime else 0.0
        
        # Calculate regime duration
        regime_duration = 1
        for i in range(len(self.regime_history)-2, -1, -1):
            if self.regime_history[i]['regime_cluster'] == current_regime:
                regime_duration += 1
            else:
                break
                
        return {
            'probability': transition_prob,
            'duration': regime_duration
        }
        
    def _calculate_regime_characteristics(
        self,
        returns: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate detailed regime characteristics."""
        # Calculate return distribution characteristics
        returns_series = returns.mean(axis=1)
        skewness = stats.skew(returns_series)
        kurtosis = stats.kurtosis(returns_series)
        
        # Calculate market efficiency
        hurst = self._calculate_hurst_exponent(returns_series)
        
        # Calculate regime strength
        regime_strength = np.abs(self._calculate_magnetization(returns))
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'hurst_exponent': hurst,
            'regime_strength': regime_strength
        }
        
    def _calculate_hurst_exponent(self, returns: pd.Series) -> float:
        """Calculate Hurst exponent for market efficiency."""
        try:
            # Ensure we have enough data points
            if len(returns) < 10:
                logger.warning("Not enough data points for Hurst exponent calculation")
                return 0.5  # Return neutral value
                
            # Calculate lags and tau with safety checks
            lags = range(2, min(100, len(returns) // 2))
            tau = []
            
            for lag in lags:
                # Calculate standard deviation of differences
                diff = np.subtract(returns[lag:], returns[:-lag])
                if len(diff) > 0:
                    std = np.std(diff)
                    if std > 0:  # Avoid log(0)
                        tau.append(std)
                    else:
                        tau.append(1e-10)  # Small positive number
                else:
                    tau.append(1e-10)
            
            if len(tau) < 2:
                logger.warning("Insufficient tau values for Hurst exponent calculation")
                return 0.5
                
            # Convert to numpy arrays and take logs
            lags = np.array(lags)
            tau = np.array(tau)
            
            # Add small epsilon to avoid log(0)
            lags = np.log(lags + 1e-10)
            tau = np.log(tau + 1e-10)
            
            # Calculate Hurst exponent
            reg = np.polyfit(lags, tau, 1)
            hurst = reg[0]
            
            # Ensure Hurst exponent is in reasonable range
            hurst = np.clip(hurst, 0.1, 0.9)
            
            return float(hurst)
            
        except Exception as e:
            logger.warning(f"Error calculating Hurst exponent: {str(e)}")
            return 0.5  # Return neutral value on error
        
    def generate_signals(
        self,
        returns: pd.DataFrame,
        window: int = 20
    ) -> np.ndarray:
        """
        Generate trading signals using the Ising model.
        
        Args:
            returns: DataFrame of asset returns
            window: Rolling window size for calculations
            
        Returns:
            Array of trading signals (-1, 0, 1)
        """
        try:
            logger.info(f"Generating signals with window size {window}")
            logger.info(f"Input returns shape: {returns.shape}")
            
            # Ensure returns is a DataFrame
            if isinstance(returns, pd.Series):
                returns = returns.to_frame()
            
            # Calculate market regime
            regime = self.calculate_market_regime(returns, window)
            
            # Use consistent asset extraction
            latest_returns = self._extract_returns_for_assets(returns)
            
            # Convert returns to binary states (-1 or 1)
            state = np.sign(latest_returns)
            
            # Calculate energy of current state - ensure it returns a scalar
            energy = self.calculate_energy(state)
            
            # Generate signal based on energy and regime metrics
            signal = np.zeros_like(state)
            
            # Use scalar comparisons for energy
            if float(energy) < -0.5:  # Strong negative energy indicates potential reversal
                signal = -state
            elif float(energy) > 0.5:  # Strong positive energy indicates trend continuation
                signal = state
            else:  # Otherwise, use regime metrics to determine signal
                if regime['volatility'] > 0.02:  # High volatility
                    signal = -state  # Mean reversion
                elif regime['momentum'] > 0.001:  # Strong momentum
                    signal = state  # Trend following
                else:
                    signal = np.zeros_like(state)  # No clear signal
            
            # Ensure signal is returned as a scalar for single asset case
            if self.n_assets == 1:
                # Handle various signal types more robustly
                if isinstance(signal, np.ndarray):
                    if signal.size == 1:
                        signal_val = float(signal.item())
                    elif signal.size > 1:
                        signal_val = float(signal.flatten()[0])
                    else:
                        signal_val = 0.0
                else:
                    signal_val = float(signal)
                return np.array([signal_val])
            
            logger.info(f"Generated signal: {signal}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            logger.error(traceback.format_exc())
            return np.zeros(self.n_assets)  # Return neutral signal on error 