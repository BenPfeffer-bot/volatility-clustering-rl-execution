"""Trade class for backtesting system."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Trade:
    """Represents a trade in the backtesting system."""

    # Required fields (non-default arguments)
    id: str
    symbol: str
    direction: int  # 1 for long, -1 for short
    entry_time: datetime
    entry_price: float
    expected_entry_price: float
    size: float

    # Optional fields (default arguments)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    expected_exit_price: Optional[float] = None
    filled_size: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    temp_impact: float = 0.0
    perm_impact: float = 0.0
    pnl: Optional[float] = None
    return_pct: Optional[float] = None

    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if self.exit_price is not None and self.entry_price is not None:
            # Calculate PnL
            raw_pnl = (self.exit_price - self.entry_price) * self.size
            self.pnl = raw_pnl * self.direction - self.commission

            # Calculate return percentage
            self.return_pct = (
                (self.exit_price - self.entry_price) / self.entry_price
            ) * self.direction
