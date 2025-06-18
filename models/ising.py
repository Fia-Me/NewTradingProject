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
            energy = -self.external_field * state[0]
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
        
    def _calculate_magnetization(self, returns: pd.DataFrame) -> float:
        """Calculate magnetization (average market state)."""
        # Use only the most recent return value
        state = np.sign(returns.iloc[-1].values)
        return np.mean(state)
        
    def _calculate_susceptibility(self, returns: pd.DataFrame) -> float:
        """Calculate susceptibility (market sensitivity to changes)."""
        # Use only the most recent return value
        magnetization = self._calculate_magnetization(returns)
        return np.var(np.sign(returns.iloc[-1].values)) / self.temperature
        
    def _calculate_system_energy(self, returns: pd.DataFrame) -> float:
        """Calculate total system energy."""
        # Use only the most recent return value
        state = np.sign(returns.iloc[-1].values)
        return self.calculate_energy(state)
        
    def _identify_regime_clusters(
        self,
        returns: pd.DataFrame,
        n_clusters: int = 3
    ) -> int:
        """Identify market regime clusters using K-means."""
        # Input validation
        if returns.empty:
            raise ValueError("Input returns DataFrame is empty")
            
        # Calculate features with proper NaN handling
        features = []
        
        # Mean returns
        mean_returns = returns.mean(axis=1)
        logger.debug(f"Mean returns NaN count: {mean_returns.isna().sum()}")
        features.append(mean_returns)
        
        # Standard deviation of returns
        std_returns = returns.std(axis=1)
        logger.debug(f"Std returns NaN count: {std_returns.isna().sum()}")
        features.append(std_returns)
        
        # Rolling correlation with proper NaN handling
        corr = returns.rolling(window=20, min_periods=1).corr()
        if corr.isna().any().any():
            logger.warning("NaN values found in correlation matrix")
            # Fill NaN values in correlation matrix
            corr = corr.fillna(0)  # Fill with 0 for no correlation
        corr_mean = corr.mean(axis=1)
        logger.debug(f"Correlation NaN count: {corr_mean.isna().sum()}")
        features.append(corr_mean)
        
        # Stack features
        features = np.column_stack(features)
        
        # Convert to DataFrame for easier handling
        features_df = pd.DataFrame(features)
        
        # Handle NaN values using newer methods
        features_df = features_df.ffill().bfill()  # Forward fill then backward fill
        
        # Additional check for any remaining NaN values
        if features_df.isna().any().any():
            logger.warning("NaN values remain after cleaning, filling with 0")
            features_df = features_df.fillna(0)
            
        # Convert back to numpy array
        features = features_df.values
        
        # Verify no NaN values remain
        if np.isnan(features).any():
            raise ValueError("NaN values remain in features after cleaning")
            
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        return clusters[-1]
        
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
            returns: DataFrame of asset returns or features
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
            
            # Extract only the returns column if it exists, otherwise use the first column
            if 'returns' in returns.columns:
                returns_data = returns[['returns']].copy()
                logger.info("Using 'returns' column for Ising model calculations")
            else:
                # Use the first column or create returns from close if available
                if 'close' in returns.columns:
                    returns_data = pd.DataFrame(index=returns.index)
                    returns_data['returns'] = returns['close'].pct_change()
                    logger.info("Created returns from 'close' column for Ising model calculations")
                else:
                    # Just use the first column and assume it's returns-like
                    first_col = returns.columns[0]
                    returns_data = returns[[first_col]].copy()
                    returns_data.columns = ['returns']
                    logger.info(f"Using first column '{first_col}' as returns for Ising model calculations")
            
            # Calculate market regime using only returns data
            regime = self.calculate_market_regime(returns_data, window)
            
            # Extract the most recent return value - now should be size 1
            latest_returns = returns_data.iloc[-1].values
            
            # Verify size matches n_assets
            if len(latest_returns) != self.n_assets:
                logger.warning(f"Latest returns size {len(latest_returns)} doesn't match n_assets {self.n_assets}, taking first {self.n_assets} values")
                latest_returns = latest_returns[:self.n_assets]
            
            # Convert returns to binary states (-1 or 1)
            state = np.sign(latest_returns)
            
            # Handle zero returns (set to 1 by default)
            state[state == 0] = 1
            
            # Calculate energy of current state
            energy = self.calculate_energy(state)
            
            # Generate signal based on energy and regime metrics
            signal = np.zeros_like(state)
            
            # Strong negative energy indicates potential reversal
            if energy < -0.5:
                signal = -state
            # Strong positive energy indicates trend continuation
            elif energy > 0.5:
                signal = state
            # Otherwise, use regime metrics to determine signal
            else:
                if regime['volatility'] > 0.02:  # High volatility
                    signal = -state  # Mean reversion
                elif regime['momentum'] > 0.001:  # Strong momentum
                    signal = state  # Trend following
                else:
                    signal = np.zeros_like(state)  # No clear signal
            
            logger.info(f"Generated signal: {signal}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            logger.error(traceback.format_exc())
            return np.zeros(self.n_assets)  # Return neutral signal on error 