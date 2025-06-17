"""
Data collection module for market data from multiple sources.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCollector:
    """Handles data collection from multiple sources."""
    
    def __init__(self, data_source="yfinance"):
        self.data_source = data_source

    def fetch_ohlcv(self, symbol, start_date, end_date, interval="1d"):
        if self.data_source == "yfinance":
            return self._fetch_yfinance_ohlcv(symbol, start_date, end_date, interval)
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")

    def _fetch_yfinance_ohlcv(self, symbol, start_date, end_date, interval):
        try:
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
            return data
        except Exception as e:
            logger.error(f"Error fetching data from yfinance: {str(e)}")
            raise
            
    def fetch_order_book(
        self,
        symbol: str,
        depth: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch order book data.
        
        Args:
            symbol: Trading symbol
            depth: Order book depth
            
        Returns:
            Dictionary with bid and ask DataFrames
        """
        if self.data_source != "moomoo":
            logger.warning("Order book data only available from Moomoo")
            return None
            
        try:
            data = self.moomoo.get_order_book(symbol=symbol, depth=depth)
            
            # Convert to DataFrames
            bids = pd.DataFrame(data['bids'], columns=['price', 'volume'])
            asks = pd.DataFrame(data['asks'], columns=['price', 'volume'])
            
            return {
                'bids': bids,
                'asks': asks
            }
            
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {str(e)}")
            raise
            
    def fetch_market_depth(
        self,
        symbol: str,
        depth: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch market depth data.
        
        Args:
            symbol: Trading symbol
            depth: Market depth
            
        Returns:
            Dictionary with bid and ask DataFrames
        """
        if self.data_source != "moomoo":
            logger.warning("Market depth data only available from Moomoo")
            return None
            
        try:
            data = self.moomoo.get_market_depth(symbol=symbol, depth=depth)
            
            # Convert to DataFrames
            bids = pd.DataFrame(data['bids'], columns=['price', 'volume'])
            asks = pd.DataFrame(data['asks'], columns=['price', 'volume'])
            
            return {
                'bids': bids,
                'asks': asks
            }
            
        except Exception as e:
            logger.error(f"Error fetching market depth for {symbol}: {str(e)}")
            raise
            
    def fetch_tick_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch tick data.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with tick data
        """
        if self.data_source != "moomoo":
            logger.warning("Tick data only available from Moomoo")
            return None
            
        try:
            data = self.moomoo.get_tick_data(
                symbol=symbol,
                start_time=start_date,
                end_time=end_date
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching tick data for {symbol}: {str(e)}")
            raise
            
    def fetch_options_chain(
        self,
        symbol: str,
        expiration_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch options chain data.
        
        Args:
            symbol: Trading symbol
            expiration_date: Option expiration date
            
        Returns:
            DataFrame with options chain data
        """
        if self.data_source == "yfinance":
            try:
                ticker = yf.Ticker(symbol)
                options = ticker.options
                
                if not options:
                    logger.warning(f"No options data available for {symbol}")
                    return None
                    
                if expiration_date:
                    # Find closest expiration date
                    exp_dates = pd.to_datetime(options)
                    closest_exp = min(exp_dates, key=lambda x: abs(x - expiration_date))
                    options_data = ticker.option_chain(closest_exp.strftime('%Y-%m-%d'))
                else:
                    # Use nearest expiration
                    options_data = ticker.option_chain(options[0])
                    
                # Combine calls and puts
                calls = options_data.calls
                puts = options_data.puts
                
                calls['type'] = 'call'
                puts['type'] = 'put'
                
                return pd.concat([calls, puts])
                
            except Exception as e:
                logger.error(f"Error fetching options chain for {symbol}: {str(e)}")
                raise
        else:
            logger.warning("Options chain data only available from yfinance")
            return None
            
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the quality of collected data.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if data is None or data.empty:
            logger.error("Empty DataFrame")
            return False
            
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values found:\n{missing_values}")
            return False
            
        # Check for duplicate indices
        if data.index.duplicated().any():
            logger.warning("Duplicate timestamps found")
            return False
            
        # Check for invalid prices
        if (data['close'] <= 0).any():
            logger.warning("Invalid close prices found")
            return False
            
        # Check for sufficient data points
        if len(data) < 30:  # Minimum required for meaningful analysis
            logger.warning("Insufficient data points")
            return False
            
        return True
        
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading symbols.
        
        Returns:
            List of available symbols
        """
        if self.data_source == "moomoo":
            try:
                return self.moomoo.get_symbols()
            except Exception as e:
                logger.error(f"Error fetching available symbols: {str(e)}")
                return []
        else:  # yfinance
            # Return list of common symbols
            return [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META",
                "TSLA", "NVDA", "JPM", "V", "WMT",
                "JNJ", "PG", "MA", "UNH", "HD",
                "BAC", "XOM", "DIS", "NFLX", "PYPL"
            ] 