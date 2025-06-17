"""
Backtesting framework for evaluating trading strategies.
"""
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BacktestFramework:
    """Framework for backtesting trading strategies."""
    
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        transaction_cost: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,  # 0.05% slippage
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
        position_size_limit: float = 0.15  # 15% of portfolio
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        self.position_size_limit = position_size_limit
        
        # Initialize portfolio tracking
        self.portfolio_value = initial_capital
        self.positions = {}
        self.trades = []
        self.daily_performance = []
        
        # Initialize feature engineer
        from features.engineer import FeatureEngineer
        from utils.config_loader import ConfigLoader
        config_loader = ConfigLoader()
        feature_params = config_loader.get_feature_params('all')
        self.feature_engineer = FeatureEngineer(feature_params=feature_params)
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized BacktestFramework with parameters:")
        self.logger.info(f"Initial capital: {initial_capital}")
        self.logger.info(f"Transaction cost: {transaction_cost}")
        self.logger.info(f"Slippage: {slippage}")
        self.logger.info(f"Risk-free rate: {risk_free_rate}")
        self.logger.info(f"Position size limit: {position_size_limit}")
        
        # Initialize metrics
        self.current_capital = initial_capital
        self.current_price = 0.0  # Initialize current_price
        self.position = 0.0  # Initialize position
        self.timestamp = None  # Initialize timestamp
        self.current_position = 0  # Track current position for trade execution
        
    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy: Any,
        initial_capital: float = 100000.0,
        window_size: int = 20  # Added window_size parameter for Ising model and feature engineering
    ) -> Dict[str, Any]:
        """Run backtest on historical data."""
        try:
            self.logger.info("Starting backtest...")
            
            # Initialize portfolio
            self.portfolio_value = initial_capital
            self.positions = {}
            self.trades = []
            self.daily_performance = []
            self.current_position = 0  # Reset at start
            
            # Ensure data has all required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Data missing required columns: {required_columns}")
            
            # Calculate technical indicators if not present
            if 'returns' not in data.columns:
                data['returns'] = data['close'].pct_change()
            if 'volatility' not in data.columns:
                data['volatility'] = data['returns'].rolling(window=20).std()
            if 'rsi' not in data.columns:
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                data['rsi'] = 100 - (100 / (1 + rs))
            if 'macd' not in data.columns:
                exp1 = data['close'].ewm(span=12, adjust=False).mean()
                exp2 = data['close'].ewm(span=26, adjust=False).mean()
                data['macd'] = exp1 - exp2
                data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
            if 'momentum' not in data.columns:
                data['momentum'] = data['close'].pct_change(periods=10)
            if 'adx' not in data.columns:
                # Calculate True Range
                tr = pd.DataFrame()
                tr['h-l'] = data['high'] - data['low']
                tr['h-pc'] = abs(data['high'] - data['close'].shift(1))
                tr['l-pc'] = abs(data['low'] - data['close'].shift(1))
                tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
                
                # Calculate Directional Movement
                data['+dm'] = (data['high'] - data['high'].shift(1)).clip(lower=0)
                data['-dm'] = (data['low'].shift(1) - data['low']).clip(lower=0)
                
                # Calculate smoothed averages
                tr14 = tr['tr'].rolling(window=14).mean()
                plus_dm14 = data['+dm'].rolling(window=14).mean()
                minus_dm14 = data['-dm'].rolling(window=14).mean()
                
                # Calculate Directional Indicators
                data['+di'] = 100 * (plus_dm14 / tr14)
                data['-di'] = 100 * (minus_dm14 / tr14)
                
                # Calculate ADX
                dx = 100 * abs(data['+di'] - data['-di']) / (data['+di'] + data['-di'])
                data['adx'] = dx.rolling(window=14).mean()
            
            # Run backtest
            # Only start after enough data for Ising model window
            timestamps = list(data.index)
            for i in range(window_size - 1, len(timestamps)):
                timestamp = timestamps[i]
                try:
                    # Prepare features using only data up to current timestamp
                    features = self._prepare_features(data, timestamp)
                    # Pass rolling window of features to generate_signals
                    features_window = features.tail(window_size) if len(features) >= window_size else features
                    self.logger.info(f"[DEBUG] Backtest loop iteration {i}:")
                    self.logger.info(f"  Timestamp: {timestamp}")
                    self.logger.info(f"  Features window shape: {features_window.shape}")
                    self.logger.info(f"  Current position: {self.current_position}")
                    self.logger.info(f"  Portfolio value: {self.portfolio_value:.2f}")
                    
                    signal = strategy.generate_signals(features_window)
                    self.logger.info(f"  Generated signal: {signal}")
                    
                    # Execute trades
                    self.logger.info(f"  Attempting to execute trade with signal {signal}")
                    self._execute_trades(signal, data.loc[timestamp], timestamp)
                    self.logger.info(f"  Trade execution complete")
                    self.logger.info(f"  New position: {self.current_position}")
                    self.logger.info(f"  New portfolio value: {self.portfolio_value:.2f}")
                    
                    # Record daily performance
                    self._record_daily_performance()
                except Exception as e:
                    self.logger.error(f"Error processing timestamp {timestamp}: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    continue
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics()
            
            self.logger.info("Backtest completed successfully")
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {str(e)}")
            raise
            
    def _prepare_features(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
        """Prepare features for prediction."""
        try:
            # Get the data up to the current timestamp
            current_data = data.loc[:timestamp].copy()
            # Use feature engineer to create features
            features = self.feature_engineer.create_features(current_data)
            
            # Ensure all required features are present
            required_features = [
                'returns', 'volatility', 'rsi', 'macd', 'momentum', 'adx'
            ]
            
            # Check for missing features
            missing_features = [f for f in required_features if f not in features.columns]
            if missing_features:
                self.logger.warning(f"Missing features: {missing_features}")
                self.logger.info("Available features:")
                for idx in features.columns:
                    self.logger.info(f"- {idx}: {features[idx].iloc[0]}")
                raise ValueError(f"Missing required features: {missing_features}")
                
            # Return the full window of features instead of just the latest row
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
            
    def _execute_trades(self, signal: int, row: pd.Series, timestamp: pd.Timestamp) -> None:
        """
        Execute trades based on signals.
        
        Args:
            signal: Trading signal (-1, 0, 1)
            row: Current market data row
            timestamp: Current timestamp
        """
        try:
            # Get current price
            price = row['close']
            
            # Calculate position size based on available capital
            position_size = self.portfolio_value * self.position_size_limit
            
            # Execute trades based on signal
            if signal > 0:  # Long signal
                if self.current_position <= 0:  # No position or short position
                    # Close any existing short position
                    if self.current_position < 0:
                        self._close_position(price, timestamp, 'short')
                    
                    # Open long position
                    self._open_position(price, timestamp, 'long')
                    
            elif signal < 0:  # Short signal
                if self.current_position >= 0:  # No position or long position
                    # Close any existing long position
                    if self.current_position > 0:
                        self._close_position(price, timestamp, 'long')
                    
                    # Open short position
                    self._open_position(price, timestamp, 'short')
                    
            # Record daily performance
            self._record_daily_performance()
            
        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}")
            raise
            
    def _open_position(self, price: float, timestamp: pd.Timestamp, position_type: str) -> None:
        """Open a new position."""
        try:
            # Calculate position size based on risk parameters
            base_position_size = 0.15  # 15% of capital from config
            position_size = base_position_size * self.current_capital
            
            # Adjust position size based on market conditions
            volatility = self._calculate_volatility()
            rsi = self._calculate_rsi()
            adx = self._calculate_adx()
            
            # Reduce position size in high volatility or extreme RSI
            if volatility > 0.02:  # High volatility
                position_size *= 0.8
            if rsi > 70 or rsi < 30:  # Extreme RSI
                position_size *= 0.8
            if adx < 25:  # Weak trend
                position_size *= 0.8
            
            # Ensure minimum position size
            position_size = max(position_size, 0.05 * self.current_capital)
            
            # Cap position size
            position_size = min(position_size, 0.15 * self.current_capital)
            
            # Record the trade
            trade = {
                'entry_date': timestamp,
                'entry_price': price,
                'position_type': position_type,
                'size': position_size,
                'volatility': volatility,
                'rsi': rsi,
                'adx': adx
            }
            self.trades.append(trade)
            
            # Update positions
            self.positions['current'] = {
                'type': position_type,
                'size': position_size,
                'entry_price': price,
                'entry_date': timestamp
            }
            
            # Update capital
            self.current_capital -= position_size
            # Update current_position
            self.current_position = position_size if position_type == 'long' else -position_size
            
        except Exception as e:
            self.logger.error(f"Error opening position: {str(e)}")
            
    def _close_position(self, price: float, timestamp: pd.Timestamp, position_type: str) -> None:
        """Close an existing position."""
        try:
            if not self.positions:
                return
                
            position = self.positions['current']
            
            # Calculate P&L
            if position_type == 'long':
                pnl = (price - position['entry_price']) * position['size']
            else:  # short
                pnl = (position['entry_price'] - price) * position['size']
            
            # Update capital
            self.current_capital += position['size'] + pnl
            
            # Record trade completion
            self.trades[-1].update({
                'exit_date': timestamp,
                'exit_price': price,
                'pnl': pnl,
                'holding_period': (timestamp - position['entry_date']).days
            })
            
            # Clear position
            self.positions = {}
            self.current_position = 0
            
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")

    def _record_daily_performance(self) -> None:
        """Record daily performance metrics."""
        # Calculate current portfolio value
        portfolio_value = float(self.current_capital)
        if self.position != 0:
            portfolio_value += self.position * self.current_price
        
        # Calculate daily return relative to previous day
        if self.daily_performance:
            prev_value = self.daily_performance[-1]['portfolio_value']
            daily_return = (portfolio_value - prev_value) / prev_value
        else:
            daily_return = (portfolio_value - self.initial_capital) / self.initial_capital
        
        # Record performance
        self.daily_performance.append({
            'timestamp': self.timestamp,
            'portfolio_value': portfolio_value,
            'return': daily_return,
            'position': float(self.position),
            'price': float(self.current_price),
            'capital': float(self.current_capital)
        })
        
        # Update portfolio value
        self.portfolio_value = portfolio_value
        
        logger.info(f"Daily Performance - Portfolio Value: {portfolio_value:.2f}, Return: {daily_return:.4%}, Position: {self.position:.2f}")

    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from backtest results.
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.daily_performance:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'num_trades': 0,
                'win_rate': 0.0
            }
        
        # Convert to DataFrame for easier calculation
        perf_df = pd.DataFrame(self.daily_performance)
        
        # Calculate metrics
        total_return = (perf_df['portfolio_value'].iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Calculate returns and Sharpe ratio
        returns = perf_df['return'].astype(float)  # Ensure returns are float
        if len(returns) > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate drawdown
        portfolio_values = perf_df['portfolio_value'].astype(float)  # Ensure values are float
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Calculate trade statistics
        num_trades = len([t for t in self.trades if t['type'] == 'close'])
        winning_trades = len([t for t in self.trades if t['type'] == 'close' and t.get('pnl', 0) > 0])
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        
        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'num_trades': num_trades,
            'win_rate': float(win_rate),
            'trades': self.trades,
            'daily_performance': self.daily_performance
        }
        
    def plot_results(self) -> None:
        """Plot backtest results with trade markers."""
        if not self.daily_performance:
            logger.warning("No performance data to plot")
            return
            
        # Convert daily performance to DataFrame
        perf_df = pd.DataFrame(self.daily_performance)
        perf_df.set_index('timestamp', inplace=True)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        perf_df['portfolio_value'].plot(ax=ax1, label='Portfolio Value', color='blue')
        ax1.set_title('Equity Curve with Trades')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)
        
        # Plot trades
        for trade in self.trades:
            if trade['type'] == 'open':
                color = 'green' if trade['position_type'] == 'long' else 'red'
                marker = '^' if trade['position_type'] == 'long' else 'v'
                ax1.scatter(trade['timestamp'], trade['price'] * trade['size'], 
                          color=color, marker=marker, s=100, label=f"{trade['position_type'].upper()} Entry")
            else:  # close
                color = 'red' if trade['position_type'] == 'long' else 'green'
                marker = 'v' if trade['position_type'] == 'long' else '^'
                ax1.scatter(trade['timestamp'], trade['price'] * trade['size'], 
                          color=color, marker=marker, s=100, label=f"{trade['position_type'].upper()} Exit")
        
        # Plot drawdown
        rolling_max = perf_df['portfolio_value'].expanding().max()
        drawdowns = (perf_df['portfolio_value'] - rolling_max) / rolling_max
        drawdowns.plot(ax=ax2, label='Drawdown', color='red')
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True)
        
        # Add trade statistics
        stats_text = (
            f"Total Return: {self._calculate_performance_metrics()['total_return']:.2%}\n"
            f"Sharpe Ratio: {self._calculate_performance_metrics()['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {self._calculate_performance_metrics()['max_drawdown']:.2%}\n"
            f"Number of Trades: {self._calculate_performance_metrics()['num_trades']}\n"
            f"Win Rate: {self._calculate_performance_metrics()['win_rate']:.2%}"
        )
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
    def run_walk_forward_analysis(
        self,
        data: pd.DataFrame,
        strategy,
        window_size: int = 252,  # 1 year
        step_size: int = 63,  # 3 months
        min_periods: int = 126  # 6 months
    ) -> Dict[str, List[float]]:
        """
        Run walk-forward analysis.
        
        Args:
            data: DataFrame with market data
            strategy: Strategy object to test
            window_size: Size of training window
            step_size: Size of step between windows
            min_periods: Minimum number of periods required
            
        Returns:
            Dictionary of performance metrics for each window
        """
        results = {
            'returns': [],
            'sharpe_ratios': [],
            'max_drawdowns': [],
            'win_rates': []
        }
        
        for i in range(0, len(data) - window_size, step_size):
            # Split data into training and testing
            train_data = data.iloc[i:i + window_size]
            test_data = data.iloc[i + window_size:i + window_size + step_size]
            
            if len(train_data) < min_periods:
                continue
                
            # Train strategy
            strategy.train(train_data)
            
            # Run backtest on test data
            metrics = self.run_backtest(test_data, strategy)
            
            # Store results
            results['returns'].append(metrics['annual_return'])
            results['sharpe_ratios'].append(metrics['sharpe_ratio'])
            results['max_drawdowns'].append(metrics['max_drawdown'])
            results['win_rates'].append(metrics['win_rate'])
            
        return results
        
    def run_monte_carlo_simulation(
        self,
        returns: pd.Series,
        num_simulations: int = 1000,
        time_horizon: int = 252
    ) -> Dict[str, np.ndarray]:
        """
        Run Monte Carlo simulation.
        
        Args:
            returns: Series of returns
            num_simulations: Number of simulations
            time_horizon: Time horizon in days
            
        Returns:
            Dictionary of simulation results
        """
        # Calculate parameters
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate simulations
        simulations = np.random.normal(
            mean_return,
            std_return,
            (num_simulations, time_horizon)
        )
        
        # Calculate cumulative returns
        cumulative_returns = (1 + simulations).cumprod(axis=1)
        
        # Calculate statistics
        percentiles = np.percentile(cumulative_returns, [5, 25, 50, 75, 95], axis=0)
        
        return {
            'simulations': cumulative_returns,
            'percentiles': percentiles
        }
        
    def run_stress_test(
        self,
        data: pd.DataFrame,
        strategy,
        stress_scenarios: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Run stress test.
        
        Args:
            data: DataFrame with market data
            strategy: Strategy object to test
            stress_scenarios: Dictionary of stress scenarios
            
        Returns:
            Dictionary of stress test results
        """
        results = {}
        
        # Run backtest on original data
        base_metrics = self.run_backtest(data, strategy)
        results['base_case'] = base_metrics
        
        # Run stress scenarios
        for scenario, shock in stress_scenarios.items():
            # Apply shock to data
            stressed_data = data.copy()
            stressed_data['close'] *= (1 + shock)
            
            # Run backtest on stressed data
            stress_metrics = self.run_backtest(stressed_data, strategy)
            results[scenario] = stress_metrics
            
        return results
        
    def calculate_statistical_significance(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate statistical significance of strategy performance.
        
        Args:
            strategy_returns: Series of strategy returns
            benchmark_returns: Series of benchmark returns
            
        Returns:
            Dictionary of statistical tests results
        """
        # Calculate excess returns
        excess_returns = strategy_returns - benchmark_returns
        
        # Perform t-test
        t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
        
        # Calculate information ratio
        ir = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        # Calculate tracking error
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'information_ratio': ir,
            'tracking_error': tracking_error
        } 