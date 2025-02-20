# src/backtesting/strategies/institutional.py

from .base import BaseStrategy, Trade
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class EnhancedInstitutionalStrategy(BaseStrategy):
    """
    Enhanced institutional trading strategy that combines:
    - VPIN-based institutional flow detection
    - Market impact prediction
    - Adaptive position sizing
    - Dynamic exit rules
    """

    def __init__(
        self,
        vpin_threshold: float = 0.7,
        min_holding_time: int = 5,  # minutes
        max_holding_time: int = 15,  # minutes
        stop_loss: float = 0.01,  # Tighter 1% stop loss
        take_profit: float = 0.005,  # 0.5% target return
        vol_window: int = 100,  # window for volatility calculation
    ):
        self.vpin_threshold = vpin_threshold
        self.min_holding_time = min_holding_time
        self.max_holding_time = max_holding_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.vol_window = vol_window
        self._vol_ma = None  # Cache for volatility moving average
        self.trades = []  # Store completed trades
        self.active_trade = None  # Store current active trade

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on institutional flow.

        Args:
            data: Market data with features

        Returns:
            Series of trading signals (1: buy, -1: sell, 0: no trade)
        """
        signals = pd.Series(0, index=data.index)

        # Calculate entry conditions
        vpin_condition = data["vpin"] > self.vpin_threshold
        volatility_condition = (
            data["daily_volatility"] < data["daily_volatility"].mean()
        )
        impact_condition = (
            data.get("market_impact_pred", pd.Series(0, index=data.index)) < 0.01
        )

        # Generate signals
        entry_condition = vpin_condition & volatility_condition & impact_condition
        signals[entry_condition] = 1

        # Update volatility MA
        self._vol_ma = data["daily_volatility"].rolling(self.vol_window).mean()

        return signals

    def should_exit(self, data: pd.Series, trade: Trade) -> bool:
        """
        Determine if position should be exited.

        Args:
            data: Current market data
            trade: Active trade to evaluate

        Returns:
            True if position should be exited
        """
        if not trade or not trade.entry_time:
            return False

        # Calculate holding time
        current_time = pd.to_datetime(data.name)
        holding_time = (current_time - trade.entry_time).total_seconds() / 60

        # Calculate returns
        returns = (data["close"] - trade.entry_price) / trade.entry_price

        # Exit conditions
        stop_loss_hit = returns < -self.stop_loss
        take_profit_hit = returns > self.take_profit
        max_time_exceeded = holding_time > self.max_holding_time
        min_time_met = holding_time >= self.min_holding_time

        # VPIN-based exit
        vpin_exit = data["vpin"] < self.vpin_threshold and min_time_met

        # Volatility-based exit (using cached MA)
        vol_exit = (
            self._vol_ma is not None
            and data["daily_volatility"] > self._vol_ma.iloc[-1]
            and min_time_met
        )

        return (
            stop_loss_hit
            or take_profit_hit
            or max_time_exceeded
            or (min_time_met and (vpin_exit or vol_exit))
        )

    def size_position(self, signal: float, data: pd.Series) -> float:
        """
        Calculate position size based on market conditions.

        Args:
            signal: Trading signal
            data: Current market data

        Returns:
            Position size as fraction of portfolio
        """
        if abs(signal) == 0:
            return 0.0

        # Ultra-conservative base size
        base_size = 0.1  # Start with 10% max position

        # Adjust for VPIN with conservative scaling
        vpin_factor = min(0.5 * data["vpin"] / self.vpin_threshold, 0.8)

        # Conservative volatility adjustment
        vol_factor = 1.0
        if "daily_volatility" in data:
            vol_factor = 1.0 / (1.0 + 5.0 * data["daily_volatility"])

        # Conservative impact adjustment
        impact_factor = 1.0
        if "market_impact_pred" in data:
            impact_factor = 1.0 / (1.0 + 5.0 * data["market_impact_pred"])

        # Volume-based adjustment
        volume_factor = 1.0
        if "volume" in data and "avg_volume" in data:
            # Significantly reduce size when volume is low
            volume_factor = min(0.8, np.power(data["volume"] / data["avg_volume"], 0.3))

        # Calculate final size with all adjustments
        size = base_size * vpin_factor * vol_factor * impact_factor * volume_factor

        # Ultra-conservative maximum position size
        return min(size, 0.1)  # Cap at 10% of portfolio

    def calculate_regime(self, data: pd.DataFrame) -> str:
        """
        Determine current market regime.

        Args:
            data: Market data with features

        Returns:
            String identifying current regime
        """
        # Calculate regime indicators
        vpin_high = data["vpin"] > self.vpin_threshold
        vol_high = data["daily_volatility"] > data["daily_volatility"].mean()
        trend_up = data["close"].pct_change(20) > 0

        # Determine regime
        if vpin_high and not vol_high:
            return "institutional_flow"
        elif vpin_high and vol_high:
            return "high_impact"
        elif trend_up:
            return "trending"
        else:
            return "neutral"

    def update_parameters(self, performance_metrics: Dict) -> None:
        """
        Update strategy parameters based on performance.

        Args:
            performance_metrics: Dictionary of performance metrics
        """
        # Adjust VPIN threshold
        if performance_metrics["win_rate"] < 0.68:
            self.vpin_threshold = min(self.vpin_threshold + 0.05, 0.9)
        elif performance_metrics["sharpe_ratio"] < 2.5:
            self.vpin_threshold = min(self.vpin_threshold + 0.02, 0.9)

        # Adjust holding times
        avg_duration = performance_metrics.get("avg_trade_duration", 0)
        if avg_duration < 10:
            self.min_holding_time = max(5, self.min_holding_time - 1)
        elif avg_duration > 30:
            self.max_holding_time = min(40, self.max_holding_time + 2)

        # Adjust take profit based on average return
        if performance_metrics["avg_trade_return"] < 0.0075:
            self.take_profit = min(self.take_profit * 1.1, 0.01)

        logger.info(
            f"Updated parameters: VPIN={self.vpin_threshold:.2f}, "
            f"Hold={self.min_holding_time}-{self.max_holding_time}min, "
            f"TP={self.take_profit:.2%}"
        )

    def on_trade_completed(self, trade: Trade) -> None:
        """
        Handle completed trade.

        Args:
            trade: Completed trade to process
        """
        if trade:
            self.trades.append(trade)
            if trade == self.active_trade:
                self.active_trade = None

    def on_trade_opened(self, trade: Trade) -> None:
        """
        Handle new trade.

        Args:
            trade: New trade to process
        """
        if trade:
            self.active_trade = trade

    def get_trade_history(self) -> pd.DataFrame:
        """
        Get trade history as DataFrame.

        Returns:
            DataFrame with trade history
        """
        if not self.trades:
            return pd.DataFrame()

        trade_data = []
        for trade in self.trades:
            trade_data.append(
                {
                    "entry_time": trade.entry_time,
                    "exit_time": trade.exit_time,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "quantity": trade.quantity,
                    "pnl": trade.pnl,
                    "holding_time": trade.holding_time,
                    "vpin_entry": trade.vpin_entry,
                    "regime": trade.regime,
                }
            )

        return pd.DataFrame(trade_data)

    def run_monte_carlo(
        self, n_simulations: int = 1000, confidence_level: float = 0.95
    ) -> Dict:
        """
        Run Monte Carlo simulation on trade history.

        Args:
            n_simulations: Number of simulations to run
            confidence_level: Confidence level for VaR/ES calculations

        Returns:
            Dict with simulation results
        """
        if not self.trades:
            return {
                "expected_return": 0.0,
                "var": 0.0,
                "es": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
            }

        # Get trade returns
        returns = [trade.pnl / trade.initial_value for trade in self.trades]

        # Run simulations
        sim_returns = []
        for _ in range(n_simulations):
            # Resample returns with replacement
            sim_path = np.random.choice(returns, size=len(returns))
            sim_returns.append(np.sum(sim_path))

        sim_returns = np.array(sim_returns)

        # Calculate metrics
        expected_return = np.mean(sim_returns)
        var = np.percentile(sim_returns, (1 - confidence_level) * 100)
        es = np.mean(sim_returns[sim_returns < var])

        # Calculate max drawdown
        cumsum = np.cumsum(sim_returns)
        running_max = np.maximum.accumulate(cumsum)
        drawdowns = running_max - cumsum
        max_drawdown = np.max(drawdowns)

        # Calculate Sharpe ratio (annualized)
        sharpe = np.sqrt(252) * expected_return / np.std(sim_returns)

        return {
            "expected_return": float(expected_return),
            "var": float(var),
            "es": float(es),
            "max_drawdown": float(max_drawdown),
            "sharpe_ratio": float(sharpe),
        }
