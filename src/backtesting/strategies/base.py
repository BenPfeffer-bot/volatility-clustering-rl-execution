# src/backtesting/strategies/base.py

"""
Base strategy interface for all trading strategies.
Defines core Trade dataclass and BaseStrategy abstract class.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List
import pandas as pd
from abc import ABC, abstractmethod


@dataclass
class Trade:
    """
    Represents a single trade with entry/exit details.
    """

    entry_time: datetime
    entry_price: float
    quantity: float
    initial_value: float
    vpin_entry: float
    regime: str
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    holding_time: Optional[float] = None  # in minutes
    temp_impact: Optional[float] = None  # temporary market impact
    perm_impact: Optional[float] = None  # permanent market impact
    commission: Optional[float] = None  # commission costs
    slippage: Optional[float] = None  # slippage costs
    expected_entry_price: Optional[float] = None  # expected entry price before impact
    expected_exit_price: Optional[float] = None  # expected exit price before impact
    filled_size: Optional[float] = None  # actual filled quantity

    def __post_init__(self):
        """Convert timestamps to datetime objects if they are strings."""
        if isinstance(self.entry_time, str):
            self.entry_time = datetime.fromisoformat(self.entry_time)
        if isinstance(self.exit_time, str):
            self.exit_time = datetime.fromisoformat(self.exit_time)

    def to_dict(self) -> Dict:
        """Convert trade to dictionary for JSON serialization."""
        return {
            "entry_time": self.entry_time.isoformat()
            if isinstance(self.entry_time, (datetime, pd.Timestamp))
            else self.entry_time,
            "exit_time": self.exit_time.isoformat()
            if isinstance(self.exit_time, (datetime, pd.Timestamp))
            else self.exit_time,
            "entry_price": float(self.entry_price),
            "exit_price": float(self.exit_price)
            if self.exit_price is not None
            else None,
            "quantity": float(self.quantity),
            "initial_value": float(self.initial_value),
            "pnl": float(self.pnl) if self.pnl is not None else None,
            "holding_time": float(self.holding_time)
            if self.holding_time is not None
            else None,
            "vpin_entry": float(self.vpin_entry),
            "regime": str(self.regime),
            "temp_impact": float(self.temp_impact)
            if self.temp_impact is not None
            else None,
            "perm_impact": float(self.perm_impact)
            if self.perm_impact is not None
            else None,
            "commission": float(self.commission)
            if self.commission is not None
            else None,
            "slippage": float(self.slippage) if self.slippage is not None else None,
            "expected_entry_price": float(self.expected_entry_price)
            if self.expected_entry_price is not None
            else None,
            "expected_exit_price": float(self.expected_exit_price)
            if self.expected_exit_price is not None
            else None,
            "filled_size": float(self.filled_size)
            if self.filled_size is not None
            else None,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Trade":
        """Create trade from dictionary."""
        # Convert timestamp strings to datetime objects
        entry_time = data.get("entry_time")
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)

        exit_time = data.get("exit_time")
        if isinstance(exit_time, str) and exit_time:
            exit_time = datetime.fromisoformat(exit_time)

        return cls(
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=float(data["entry_price"]),
            exit_price=float(data["exit_price"])
            if data.get("exit_price") is not None
            else None,
            quantity=float(data["quantity"]),
            initial_value=float(data["initial_value"]),
            pnl=float(data["pnl"]) if data.get("pnl") is not None else None,
            holding_time=float(data["holding_time"])
            if data.get("holding_time") is not None
            else None,
            vpin_entry=float(data["vpin_entry"]),
            regime=str(data["regime"]),
            temp_impact=float(data["temp_impact"])
            if data.get("temp_impact") is not None
            else None,
            perm_impact=float(data["perm_impact"])
            if data.get("perm_impact") is not None
            else None,
            commission=float(data["commission"])
            if data.get("commission") is not None
            else None,
            slippage=float(data["slippage"])
            if data.get("slippage") is not None
            else None,
            expected_entry_price=float(data["expected_entry_price"])
            if data.get("expected_entry_price") is not None
            else None,
            expected_exit_price=float(data["expected_exit_price"])
            if data.get("expected_exit_price") is not None
            else None,
            filled_size=float(data["filled_size"])
            if data.get("filled_size") is not None
            else None,
        )


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Enforces implementation of core trading methods.
    """

    def __init__(self):
        self.trades = []  # Store completed trades
        self.active_trade = None  # Store current active trade

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from market data.

        Args:
            data: Market data with features

        Returns:
            Series of trading signals (1: buy, -1: sell, 0: no trade)
        """
        pass

    @abstractmethod
    def should_exit(self, data: pd.Series, trade: Trade) -> bool:
        """
        Determine if current position should be exited.

        Args:
            data: Current market data
            trade: Active trade to evaluate

        Returns:
            True if position should be exited
        """
        pass

    @abstractmethod
    def size_position(self, signal: float, data: pd.Series) -> float:
        """
        Calculate position size based on signal and market conditions.

        Args:
            signal: Trading signal
            data: Current market data

        Returns:
            Position size as fraction of portfolio
        """
        pass

    @abstractmethod
    def calculate_regime(self, data: pd.DataFrame) -> str:
        """
        Determine current market regime.

        Args:
            data: Market data with features

        Returns:
            String identifying current regime
        """
        pass

    @abstractmethod
    def update_parameters(self, performance_metrics: Dict) -> None:
        """
        Update strategy parameters based on performance.

        Args:
            performance_metrics: Dictionary of performance metrics
        """
        pass

    def on_trade_opened(self, trade: Trade) -> None:
        """
        Handle trade opening event.

        Args:
            trade: New trade that was opened
        """
        self.active_trade = trade

    def on_trade_completed(self, trade: Trade) -> None:
        """
        Handle trade completion event.

        Args:
            trade: Trade that was completed
        """
        if trade == self.active_trade:
            self.active_trade = None
        self.trades.append(trade)

    def get_trade_history(self) -> List[Trade]:
        """Get list of completed trades."""
        return self.trades
