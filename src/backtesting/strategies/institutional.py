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
        max_holding_time: int = 30,  # minutes
        stop_loss: float = 0.02,  # 2% stop loss
        take_profit: float = 0.015,  # 1.5% target return
        vol_window: int = 100,  # window for volatility calculation
        trend_window: int = 20,  # window for trend calculation
    ):
        self.vpin_threshold = vpin_threshold
        self.min_holding_time = min_holding_time
        self.max_holding_time = max_holding_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.vol_window = vol_window
        self.trend_window = trend_window
        self._vol_ma = None  # Cache for volatility moving average
        self.trades = []  # Store completed trades
        self.active_trade = None  # Store current active trade
        self.max_position_size = 0.25  # Maximum position size (25% of portfolio)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on institutional flow.

        Args:
            data: Market data with features

        Returns:
            Series of trading signals (1: buy, -1: sell, 0: no trade)
        """
        signals = pd.Series(0, index=data.index)

        # Calculate trend using multiple timeframes
        ma_fast = data["close"].rolling(10).mean()
        ma_slow = data["close"].rolling(30).mean()
        trend = (ma_fast > ma_slow).astype(int)

        # VPIN conditions
        vpin_rising = data["vpin"].diff(5) > 0
        vpin_high = data["vpin"] > self.vpin_threshold

        # Volume conditions
        volume_ma = data["volume"].rolling(20).mean()
        vol_surge = data["volume"] > volume_ma * 1.5

        # Volatility conditions
        vol_ma = data["daily_volatility"].rolling(self.vol_window).mean()
        low_vol = data["daily_volatility"] < vol_ma

        # Market impact conditions
        low_impact = (
            data.get("market_impact_pred", pd.Series(0, index=data.index)) < 0.01
        )

        # Generate long signals
        long_condition = (
            vpin_high & vpin_rising & vol_surge & (trend == 1) & low_vol & low_impact
        )

        # Generate short signals
        short_condition = (
            vpin_high & vpin_rising & vol_surge & (trend == 0) & low_vol & low_impact
        )

        signals[long_condition] = 1
        signals[short_condition] = -1

        # Update volatility MA
        self._vol_ma = vol_ma

        return signals

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

        # Base size (25% max position)
        base_size = self.max_position_size

        # VPIN factor
        vpin_factor = np.clip(data["vpin"] / self.vpin_threshold, 0.5, 2.0)

        # Trend strength adjustment
        # Calculate returns using log returns instead of pct_change
        trend_strength = abs(data["log_returns"] / data["daily_volatility"])
        trend_factor = np.clip(trend_strength, 0.5, 2.0)

        # Volume-based adjustment
        volume_ma = (
            data["volume"].rolling(20).mean().iloc[-1]
            if isinstance(data["volume"], pd.Series)
            else data["volume"]
        )
        volume_ratio = data["volume"] / volume_ma
        volume_factor = np.clip(volume_ratio, 0.5, 2.0)

        # Volatility adjustment
        vol_factor = 1.0
        if "daily_volatility" in data:
            vol_factor = 1.0 / (1.0 + 3.0 * data["daily_volatility"])

        # Market impact adjustment
        impact_factor = 1.0
        if "market_impact_pred" in data:
            impact_factor = 1.0 / (1.0 + 3.0 * data["market_impact_pred"])

        # Calculate final size with all adjustments
        size = (
            base_size
            * vpin_factor
            * trend_factor
            * volume_factor
            * vol_factor
            * impact_factor
        )

        # Cap at maximum position size
        return min(size, self.max_position_size)

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

        # Dynamic stop loss based on volatility
        dynamic_stop = max(self.stop_loss, 2.0 * data["daily_volatility"])

        # Dynamic take profit based on VPIN
        dynamic_profit = max(
            self.take_profit, self.take_profit * data["vpin"] / self.vpin_threshold
        )

        # Exit conditions
        stop_loss_hit = returns < -dynamic_stop
        take_profit_hit = returns > dynamic_profit
        max_time_exceeded = holding_time > self.max_holding_time
        min_time_met = holding_time >= self.min_holding_time

        # Trend reversal exit (using log returns instead of pct_change)
        trend_reversal = (
            data["log_returns"] * np.sign(trade.quantity) < 0 and min_time_met
        )

        # VPIN-based exit
        vpin_exit = data["vpin"] < self.vpin_threshold * 0.8 and min_time_met

        # Volatility spike exit
        vol_exit = (
            self._vol_ma is not None
            and data["daily_volatility"] > self._vol_ma.iloc[-1] * 1.5
            and min_time_met
        )

        return (
            stop_loss_hit
            or take_profit_hit
            or max_time_exceeded
            or (min_time_met and (trend_reversal or vpin_exit or vol_exit))
        )

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
        # Adjust VPIN threshold based on win rate
        if performance_metrics["win_rate"] < 0.65:
            self.vpin_threshold = min(self.vpin_threshold + 0.05, 0.9)

        # Adjust position size based on Sharpe ratio
        if performance_metrics["sharpe_ratio"] < 1.5:
            self.max_position_size *= 0.9  # Reduce by 10%
        elif performance_metrics["sharpe_ratio"] > 2.5:
            self.max_position_size = min(
                self.max_position_size * 1.1, 0.25
            )  # Increase by 10%

        # Adjust holding times based on average return
        avg_return = performance_metrics.get("avg_trade_return", 0)
        if avg_return < 0:
            self.max_holding_time = max(10, self.max_holding_time - 5)
        elif avg_return > self.take_profit:
            self.max_holding_time = min(45, self.max_holding_time + 5)

        # Adjust take profit based on volatility
        if "avg_volatility" in performance_metrics:
            self.take_profit = max(0.01, 2.0 * performance_metrics["avg_volatility"])

        logger.info(
            f"Updated parameters: VPIN={self.vpin_threshold:.2f}, "
            f"Max Size={self.max_position_size:.2%}, "
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
