"""
Core backtesting engine that orchestrates strategy execution, portfolio management,
and performance tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from ..strategies.base import BaseStrategy, Trade
from .portfolio import Portfolio
from .metrics import BacktestMetrics
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class BacktestEngine:
    """
    Core backtesting engine that manages:
    - Strategy execution
    - Portfolio updates
    - Performance tracking
    - Risk management
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 1_000_000,
        commission_rate: float = 0.001,
        slippage_model: Optional[str] = "fixed",
        slippage_rate: float = 0.0001,
    ):
        self.strategy = strategy
        self.portfolio = Portfolio(initial_capital)
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model
        self.slippage_rate = slippage_rate
        self.trades: List[Trade] = []
        self.metrics = BacktestMetrics()

    def calculate_transaction_costs(
        self, price: float, quantity: float
    ) -> Tuple[float, float, float, float]:
        """
        Calculate transaction costs including commission, slippage, and market impact.

        Returns:
            Tuple of (total_costs, commission, slippage, temp_impact)
        """
        trade_value = price * abs(quantity)
        portfolio_value = self.portfolio.current_value
        trade_size_ratio = trade_value / portfolio_value

        # Commission (typically 0.1% of trade value)
        commission = trade_value * self.commission_rate

        # More conservative slippage model
        if self.slippage_model == "fixed":
            slippage = trade_value * self.slippage_rate * 1.5  # Increased base slippage
        elif self.slippage_model == "volume_based":
            # More conservative volume-based slippage
            volume_factor = np.power(abs(quantity) / 10000, 0.4)  # Gentler scaling
            slippage = trade_value * self.slippage_rate * min(0.75, volume_factor * 1.5)
        else:
            slippage = 0.0

        # More conservative market impact model
        size_factor = np.power(abs(quantity) / 10000, 0.4) * np.exp(
            -trade_size_ratio * 1.5
        )
        temp_impact = trade_value * self.slippage_rate * min(0.4, size_factor * 1.5)

        total_costs = commission + slippage + temp_impact
        return total_costs, commission, slippage, temp_impact

    def execute_trade(
        self, timestamp: datetime, data: pd.Series, signal: float
    ) -> Optional[Trade]:
        """Execute trade based on signal and current market data."""
        if signal == 0:
            return None

        # More conservative position sizing
        size = (
            self.strategy.size_position(signal, data) * 0.75
        )  # Reduce base position size
        quantity = size * self.portfolio.current_value / data["close"]

        # Calculate all transaction costs
        total_costs, commission, slippage, temp_impact = (
            self.calculate_transaction_costs(data["close"], quantity)
        )

        # More conservative permanent impact
        perm_impact = temp_impact * 0.4  # Increased from 0.3 to 0.4

        # Calculate trade metrics
        trade_value = data["close"] * abs(quantity)
        trade_size_ratio = trade_value / self.portfolio.current_value

        # More conservative cost thresholds
        # Base threshold increased to 0.75% for small trades
        # Maximum threshold reduced to 2.0% for large trades
        cost_threshold = 0.0075 + 0.0125 / (1 + np.exp(-3 * (trade_size_ratio - 0.5)))

        # More conservative volume-based adjustment
        if "volume" in data and "avg_volume" in data:
            volume_ratio = data["volume"] / data["avg_volume"]
            # Require higher volume for full threshold
            volume_factor = min(1.25, np.power(volume_ratio, 0.4))
            # More conservative in low volume
            cost_threshold *= volume_factor

            # Additional check for very low volume
            if volume_ratio < 0.5:
                cost_threshold *= 0.75

        if total_costs > self.portfolio.current_value * cost_threshold:
            logger.warning(f"Transaction costs too high at {timestamp}")
            return None

        # Create and record trade
        trade = Trade(
            entry_time=timestamp,
            entry_price=data["close"],
            quantity=quantity,
            initial_value=self.portfolio.current_value,
            vpin_entry=data.get("vpin", 0),
            regime=data.get("regime", "unknown"),
            temp_impact=temp_impact,
            perm_impact=perm_impact,
            commission=commission,
            slippage=slippage,
            expected_entry_price=data["close"],
            filled_size=quantity,
        )

        # Update portfolio and strategy
        self.portfolio.update(trade, total_costs)
        self.strategy.on_trade_opened(trade)
        return trade

    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """
        Run backtest on historical data.

        Args:
            data: Market data with features

        Returns:
            Dict with backtest results
        """
        logger.info("Starting backtest...")

        # Initialize tracking variables
        portfolio_values = pd.Series(index=data.index)
        active_trade = None

        # Generate trading signals
        signals = self.strategy.generate_signals(data)

        # Simulate trading
        for timestamp, row in data.iterrows():
            # Update portfolio value
            if active_trade:
                # Check for exit
                if self.strategy.should_exit(row, active_trade):
                    active_trade.exit_time = timestamp
                    active_trade.exit_price = row["close"]
                    active_trade.pnl = (
                        active_trade.exit_price - active_trade.entry_price
                    ) * active_trade.quantity
                    active_trade.holding_time = (
                        active_trade.exit_time - active_trade.entry_time
                    ).total_seconds() / 60

                    self.strategy.on_trade_completed(active_trade)
                    active_trade = None

            # Process new signals
            if not active_trade and timestamp in signals.index:
                signal = signals[timestamp]
                if signal != 0:
                    active_trade = self.execute_trade(timestamp, row, signal)

            # Record portfolio value
            portfolio_values[timestamp] = self.portfolio.current_value

        # Calculate final metrics
        results = {
            "trades": self.strategy.trades,
            "portfolio_values": portfolio_values,
            "metrics": self.metrics.calculate_metrics(
                self.strategy.trades, portfolio_values
            ),
            "trade_history": self.strategy.get_trade_history(),
            "monte_carlo": self.strategy.run_monte_carlo()
            if self.strategy.trades
            else None,
        }

        logger.info("Backtest completed")
        return results

    def update_strategy(self, performance_metrics: Dict) -> None:
        """Update strategy parameters based on performance."""
        self.strategy.update_parameters(performance_metrics)
