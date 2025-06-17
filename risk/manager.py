"""
Risk management module implementing portfolio and trade-level risk controls.
"""
from typing import Dict, List, Optional, Union
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RiskManager:
    """Manages risk for the trading system."""
    
    def __init__(
        self,
        max_position_size: float = 0.15,  # 15% of portfolio (from config)
        max_sector_exposure: float = 0.30,  # 30% of portfolio (from config)
        max_leverage: float = 1.5,  # 150% leverage (from config)
        max_correlation: float = 0.7,  # 70% correlation limit (from config)
        max_daily_loss: float = 0.02,  # 2% daily loss limit (from config)
        position_holding_limit: int = 10  # 10 days max holding (from config)
    ):
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.max_leverage = max_leverage
        self.max_correlation = max_correlation
        self.max_daily_loss = max_daily_loss
        self.position_holding_limit = position_holding_limit
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized RiskManager with parameters:")
        self.logger.info(f"Max position size: {max_position_size}")
        self.logger.info(f"Max sector exposure: {max_sector_exposure}")
        self.logger.info(f"Max leverage: {max_leverage}")
        self.logger.info(f"Max correlation: {max_correlation}")
        self.logger.info(f"Max daily loss: {max_daily_loss}")
        self.logger.info(f"Position holding limit: {position_holding_limit}")
        
        # Initialize state
        self.portfolio_value = 0.0
        self.positions = {}
        self.sector_exposures = {}
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
    def check_position_size(
        self,
        symbol: str,
        size: float,
        price: float
    ) -> bool:
        """
        Check if position size is within limits.
        
        Args:
            symbol: Trading symbol
            size: Position size
            price: Current price
            
        Returns:
            Boolean indicating if position size is acceptable
        """
        position_value = abs(size * price)
        position_pct = position_value / self.portfolio_value if self.portfolio_value > 0 else 0
        
        self.logger.info(f"Checking position size for {symbol}:")
        self.logger.info(f"Position value: ${position_value:,.2f}")
        self.logger.info(f"Position percentage: {position_pct:.2%}")
        self.logger.info(f"Max allowed: {self.max_position_size:.2%}")
        
        is_valid = position_pct <= self.max_position_size
        if not is_valid:
            self.logger.warning(f"Position size {position_pct:.2%} exceeds limit {self.max_position_size:.2%}")
            
        return is_valid
        
    def check_sector_exposure(
        self,
        symbol: str,
        sector: str,
        size: float,
        price: float
    ) -> bool:
        """
        Check if sector exposure is within limits.
        
        Args:
            symbol: Trading symbol
            sector: Sector name
            size: Position size
            price: Current price
            
        Returns:
            Boolean indicating if sector exposure is acceptable
        """
        position_value = abs(size * price)
        
        # Update sector exposure
        if sector not in self.sector_exposures:
            self.sector_exposures[sector] = 0.0
        self.sector_exposures[sector] += position_value
        
        # Calculate sector percentage
        sector_pct = self.sector_exposures[sector] / self.portfolio_value if self.portfolio_value > 0 else 0
        
        self.logger.info(f"Checking sector exposure for {symbol} ({sector}):")
        self.logger.info(f"Sector exposure: ${self.sector_exposures[sector]:,.2f}")
        self.logger.info(f"Sector percentage: {sector_pct:.2%}")
        self.logger.info(f"Max allowed: {self.max_sector_exposure:.2%}")
        
        is_valid = sector_pct <= self.max_sector_exposure
        if not is_valid:
            self.logger.warning(f"Sector exposure {sector_pct:.2%} exceeds limit {self.max_sector_exposure:.2%}")
            
        return is_valid
        
    def check_leverage(self, total_exposure: float) -> bool:
        """
        Check if total leverage is within limits.
        
        Args:
            total_exposure: Total portfolio exposure
            
        Returns:
            Boolean indicating if leverage is acceptable
        """
        leverage = total_exposure / self.portfolio_value if self.portfolio_value > 0 else 0
        
        self.logger.info("Checking leverage:")
        self.logger.info(f"Total exposure: ${total_exposure:,.2f}")
        self.logger.info(f"Portfolio value: ${self.portfolio_value:,.2f}")
        self.logger.info(f"Current leverage: {leverage:.2f}x")
        self.logger.info(f"Max allowed: {self.max_leverage:.2f}x")
        
        is_valid = leverage <= self.max_leverage
        if not is_valid:
            self.logger.warning(f"Leverage {leverage:.2f}x exceeds limit {self.max_leverage:.2f}x")
            
        return is_valid
        
    def check_correlation(
        self,
        symbol: str,
        returns: pd.Series,
        existing_returns: Dict[str, pd.Series]
    ) -> bool:
        """
        Check if correlation with existing positions is within limits.
        
        Args:
            symbol: Trading symbol
            returns: Returns series for the symbol
            existing_returns: Dictionary of returns series for existing positions
            
        Returns:
            Boolean indicating if correlation is acceptable
        """
        self.logger.info(f"Checking correlation for {symbol}")
        
        for existing_symbol, existing_returns_series in existing_returns.items():
            correlation = returns.corr(existing_returns_series)
            
            self.logger.info(f"Correlation with {existing_symbol}: {correlation:.4f}")
            self.logger.info(f"Max allowed: {self.max_correlation:.4f}")
            
            if abs(correlation) > self.max_correlation:
                self.logger.warning(
                    f"Correlation {correlation:.4f} with {existing_symbol} "
                    f"exceeds limit {self.max_correlation:.4f}"
                )
                return False
                
        return True
        
    def check_daily_loss(self, pnl: float) -> bool:
        """
        Check if daily loss is within limits.
        
        Args:
            pnl: Daily profit/loss
            
        Returns:
            Boolean indicating if daily loss is acceptable
        """
        daily_loss_pct = abs(min(0, pnl)) / self.portfolio_value if self.portfolio_value > 0 else 0
        
        self.logger.info("Checking daily loss:")
        self.logger.info(f"Daily PnL: ${pnl:,.2f}")
        self.logger.info(f"Daily loss percentage: {daily_loss_pct:.2%}")
        self.logger.info(f"Max allowed: {self.max_daily_loss:.2%}")
        
        is_valid = daily_loss_pct <= self.max_daily_loss
        if not is_valid:
            self.logger.warning(f"Daily loss {daily_loss_pct:.2%} exceeds limit {self.max_daily_loss:.2%}")
            
        return is_valid
        
    def check_position_holding(
        self,
        symbol: str,
        entry_date: datetime
    ) -> bool:
        """
        Check if position holding period is within limits.
        
        Args:
            symbol: Trading symbol
            entry_date: Position entry date
            
        Returns:
            Boolean indicating if holding period is acceptable
        """
        holding_days = (datetime.now() - entry_date).days
        
        self.logger.info(f"Checking position holding for {symbol}:")
        self.logger.info(f"Entry date: {entry_date}")
        self.logger.info(f"Holding days: {holding_days}")
        self.logger.info(f"Max allowed: {self.position_holding_limit}")
        
        is_valid = holding_days <= self.position_holding_limit
        if not is_valid:
            self.logger.warning(f"Holding period {holding_days} days exceeds limit {self.position_holding_limit}")
            
        return is_valid
        
    def calculate_position_risk(
        self,
        symbol: str,
        size: float,
        price: float,
        volatility: float
    ) -> Dict[str, float]:
        """
        Calculate position-level risk metrics.
        
        Args:
            symbol: Trading symbol
            size: Position size
            price: Current price
            volatility: Price volatility
            
        Returns:
            Dictionary of risk metrics
        """
        position_value = abs(size * price)
        
        # Calculate Value at Risk (VaR)
        var_95 = position_value * volatility * 1.645
        var_99 = position_value * volatility * 2.326
        
        # Calculate Expected Shortfall (ES)
        es_95 = position_value * volatility * 2.063
        es_99 = position_value * volatility * 2.665
        
        risk_metrics = {
            'position_value': position_value,
            'var_95': var_95,
            'var_99': var_99,
            'es_95': es_95,
            'es_99': es_99
        }
        
        self.logger.info(f"Calculating risk metrics for {symbol}:")
        for metric, value in risk_metrics.items():
            self.logger.info(f"{metric}: ${value:,.2f}")
            
        return risk_metrics
        
    def calculate_portfolio_risk(
        self,
        positions: Dict[str, Dict[str, float]],
        returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate portfolio-level risk metrics.
        
        Args:
            positions: Dictionary of position information
            returns: DataFrame of returns for all positions
            
        Returns:
            Dictionary of portfolio risk metrics
        """
        self.logger.info("Calculating portfolio risk metrics...")
        
        # Calculate portfolio value
        portfolio_value = sum(pos['size'] * pos['price'] for pos in positions.values())
        
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0.0, index=returns.index)
        for symbol, pos in positions.items():
            if symbol in returns.columns:
                portfolio_returns += pos['size'] * returns[symbol]
                
        # Calculate risk metrics
        volatility = portfolio_returns.std() * np.sqrt(252)
        var_95 = portfolio_value * volatility * 1.645
        var_99 = portfolio_value * volatility * 2.326
        expected_shortfall = portfolio_value * volatility * 2.063
        sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        
        risk_metrics = {
            'portfolio_value': portfolio_value,
            'volatility': volatility,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall': expected_shortfall,
            'sharpe_ratio': sharpe_ratio
        }
        
        self.logger.info("Portfolio risk metrics:")
        for metric, value in risk_metrics.items():
            if metric == 'sharpe_ratio':
                self.logger.info(f"{metric}: {value:.4f}")
            else:
                self.logger.info(f"{metric}: ${value:,.2f}")
                
        return risk_metrics
        
    def calculate_stress_test(
        self,
        positions: Dict[str, Dict[str, float]],
        historical_returns: pd.DataFrame,
        stress_scenarios: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate stress test results.
        
        Args:
            positions: Dictionary of position information
            historical_returns: DataFrame of historical returns
            stress_scenarios: Dictionary of stress scenario returns
            
        Returns:
            Dictionary of stress test results
        """
        self.logger.info("Running stress tests...")
        
        # Calculate base portfolio value
        base_value = sum(pos['size'] * pos['price'] for pos in positions.values())
        self.logger.info(f"Base portfolio value: ${base_value:,.2f}")
        
        # Calculate stress test results
        stress_results = {}
        for scenario, shock in stress_scenarios.items():
            stressed_value = base_value * (1 + shock)
            loss = base_value - stressed_value
            
            self.logger.info(f"Stress scenario: {scenario}")
            self.logger.info(f"Shock: {shock:.2%}")
            self.logger.info(f"Stressed value: ${stressed_value:,.2f}")
            self.logger.info(f"Loss: ${loss:,.2f}")
            
            stress_results[scenario] = {
                'stressed_value': stressed_value,
                'loss': loss,
                'loss_pct': loss / base_value
            }
            
        return stress_results
        
    def monitor_risk_limits(self) -> Dict[str, bool]:
        """
        Monitor all risk limits.
        
        Returns:
            Dictionary of risk limit status
        """
        self.logger.info("Monitoring risk limits...")
        
        # Calculate total exposure
        total_exposure = sum(abs(pos['size'] * pos['price']) for pos in self.positions.values())
        
        # Check all risk limits
        risk_checks = {
            'leverage': self.check_leverage(total_exposure),
            'daily_loss': self.check_daily_loss(self.daily_pnl)
        }
        
        # Check position-specific limits
        for symbol, position in self.positions.items():
            risk_checks[f'position_size_{symbol}'] = self.check_position_size(
                symbol,
                position['size'],
                position['price']
            )
            
            if 'sector' in position:
                risk_checks[f'sector_exposure_{symbol}'] = self.check_sector_exposure(
                    symbol,
                    position['sector'],
                    position['size'],
                    position['price']
                )
                
            if 'entry_date' in position:
                risk_checks[f'holding_period_{symbol}'] = self.check_position_holding(
                    symbol,
                    position['entry_date']
                )
                
        # Log results
        self.logger.info("Risk limit check results:")
        for check, result in risk_checks.items():
            self.logger.info(f"{check}: {'PASS' if result else 'FAIL'}")
            
        return risk_checks
        
    def update_portfolio_value(self, new_value: float) -> None:
        """
        Update portfolio value.
        
        Args:
            new_value: New portfolio value
        """
        self.logger.info(f"Updating portfolio value: ${new_value:,.2f}")
        self.portfolio_value = new_value
        
    def update_position(
        self,
        symbol: str,
        size: float,
        price: float,
        sector: str,
        entry_date: datetime
    ) -> None:
        """
        Update position information.
        
        Args:
            symbol: Trading symbol
            size: Position size
            price: Current price
            sector: Sector name
            entry_date: Position entry date
        """
        self.logger.info(f"Updating position for {symbol}:")
        self.logger.info(f"Size: {size}")
        self.logger.info(f"Price: ${price:,.2f}")
        self.logger.info(f"Sector: {sector}")
        self.logger.info(f"Entry date: {entry_date}")
        
        self.positions[symbol] = {
            'size': size,
            'price': price,
            'sector': sector,
            'entry_date': entry_date
        }
        
    def reset_daily_metrics(self) -> None:
        """Reset daily risk metrics."""
        self.logger.info("Resetting daily metrics")
        self.daily_pnl = 0.0 