"""
Portfolio management component for tracking positions and value.
"""

from typing import Dict, Optional
from ..strategies.base import Trade
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class Portfolio:
    """
    Portfolio management system that tracks:
    - Current positions
    - Cash balance
    - Portfolio value
    - Transaction costs
    """

    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.current_value = initial_capital
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.active_position: Optional[Dict] = None

    def update(self, trade: Trade, transaction_costs: float) -> None:
        """
        Update portfolio state with new trade.

        Args:
            trade: Trade to process
            transaction_costs: Combined commission and slippage
        """
        # Entry
        if not trade.exit_time:
            trade_value = trade.quantity * trade.entry_price
            self.cash -= trade_value + transaction_costs
            self.current_value = self.cash + trade_value
            self.total_commission += transaction_costs

            self.active_position = {
                "quantity": trade.quantity,
                "entry_price": trade.entry_price,
                "value": trade_value,
            }

            logger.debug(
                f"Entered position: {trade.quantity:.0f} units at "
                f"${trade.entry_price:.2f}, costs=${transaction_costs:.2f}"
            )

        # Exit
        else:
            if self.active_position:
                exit_value = trade.quantity * trade.exit_price
                self.cash += exit_value - transaction_costs
                self.current_value = self.cash
                self.total_commission += transaction_costs
                self.active_position = None

                logger.debug(
                    f"Exited position: {trade.quantity:.0f} units at "
                    f"${trade.exit_price:.2f}, costs=${transaction_costs:.2f}"
                )

    def get_position_value(self, current_price: float) -> float:
        """Calculate current value of active position."""
        if self.active_position:
            return self.active_position["quantity"] * current_price
        return 0.0

    def get_metrics(self) -> Dict[str, float]:
        """Get current portfolio metrics."""
        return {
            "total_value": self.current_value,
            "cash": self.cash,
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "returns": (self.current_value / self.initial_capital) - 1,
        }
