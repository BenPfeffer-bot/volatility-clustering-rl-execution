"""
Base execution model interface for implementing trading execution strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import pandas as pd
from datetime import datetime
from ..strategies.base import Trade


class BaseExecutionModel(ABC):
    """
    Abstract base class for execution models.
    Defines interface for implementing custom execution logic.
    """

    @abstractmethod
    def calculate_execution_price(
        self,
        trade: Trade,
        market_data: pd.Series,
        market_impact: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float]:
        """
        Calculate execution price including market impact.

        Args:
            trade: Trade to execute
            market_data: Current market data
            market_impact: Optional pre-calculated market impact

        Returns:
            Tuple of (execution_price, transaction_costs)
        """
        pass

    @abstractmethod
    def should_split_order(
        self,
        quantity: float,
        market_data: pd.Series,
        max_participation: float = 0.1,
    ) -> bool:
        """
        Determine if order should be split into smaller chunks.

        Args:
            quantity: Order quantity
            market_data: Current market data
            max_participation: Maximum market participation rate

        Returns:
            True if order should be split
        """
        pass

    @abstractmethod
    def get_execution_schedule(
        self,
        trade: Trade,
        market_data: pd.DataFrame,
        duration: int = 30,  # minutes
    ) -> pd.Series:
        """
        Generate execution schedule for order.

        Args:
            trade: Trade to execute
            market_data: Market data for scheduling
            duration: Target execution duration in minutes

        Returns:
            Series of scheduled quantities
        """
        pass

    @abstractmethod
    def estimate_transaction_costs(
        self,
        price: float,
        quantity: float,
        market_data: pd.Series,
    ) -> Dict[str, float]:
        """
        Estimate transaction costs for trade.

        Args:
            price: Execution price
            quantity: Order quantity
            market_data: Current market data

        Returns:
            Dict of cost components
        """
        pass
