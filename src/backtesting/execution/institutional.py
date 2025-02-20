"""
Institutional execution model optimized for large orders.
Implements VPIN-aware execution with market impact minimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from .base import BaseExecutionModel
from .impact import MarketImpactModel
from ..strategies.base import Trade
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class InstitutionalExecutionModel(BaseExecutionModel):
    """
    Execution model for institutional orders that:
    - Minimizes market impact
    - Adapts to VPIN signals
    - Implements optimal execution scheduling
    """

    def __init__(
        self,
        impact_model: Optional[MarketImpactModel] = None,
        min_participation: float = 0.01,  # 1% min participation
        max_participation: float = 0.1,  # 10% max participation
        vpin_threshold: float = 0.7,  # VPIN threshold for aggressive execution
    ):
        self.impact_model = impact_model or MarketImpactModel()
        self.min_participation = min_participation
        self.max_participation = max_participation
        self.vpin_threshold = vpin_threshold

    def calculate_execution_price(
        self,
        trade: Trade,
        market_data: pd.Series,
        market_impact: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float]:
        """Calculate execution price with market impact."""
        base_price = market_data["close"]

        # Get market impact if not provided
        if not market_impact:
            market_impact = self.impact_model.estimate_total_impact(
                base_price,
                trade.quantity,
                market_data.to_dict(),
            )

        # Apply impact to price
        impact_adj = market_impact["total_impact"]
        exec_price = base_price * (1 + impact_adj)

        # Calculate transaction costs
        costs = self.estimate_transaction_costs(exec_price, trade.quantity, market_data)
        total_costs = sum(costs.values())

        return exec_price, total_costs

    def should_split_order(
        self,
        quantity: float,
        market_data: pd.Series,
        max_participation: float = 0.1,
    ) -> bool:
        """Determine if order should be split based on size."""
        if market_data.get("volume", 0) == 0:
            return True

        participation = quantity / market_data["volume"]
        return participation > max_participation

    def get_execution_schedule(
        self,
        trade: Trade,
        market_data: pd.DataFrame,
        duration: int = 30,  # minutes
    ) -> pd.Series:
        """Generate optimal execution schedule."""
        # Calculate time points
        start_time = trade.entry_time
        end_time = start_time + timedelta(minutes=duration)
        schedule_index = pd.date_range(
            start_time,
            end_time,
            freq="1min",
            inclusive="both",
        )

        # Base schedule using VWAP profile
        if "volume" in market_data.columns:
            volume_profile = (
                market_data["volume"].rolling(window=5, min_periods=1).mean()
            )
            schedule = volume_profile / volume_profile.sum() * trade.quantity
        else:
            # Fallback to simple linear schedule
            schedule = pd.Series(
                trade.quantity / len(schedule_index),
                index=schedule_index,
            )

        # Adjust for VPIN
        if "vpin" in market_data.columns:
            vpin_profile = market_data["vpin"].rolling(window=5, min_periods=1).mean()
            vpin_adj = np.where(
                vpin_profile > self.vpin_threshold,
                1.2,  # More aggressive when VPIN is high
                0.8,  # More passive when VPIN is low
            )
            schedule *= vpin_adj

        # Ensure schedule sums to total quantity
        schedule = schedule * (trade.quantity / schedule.sum())
        return schedule

    def estimate_transaction_costs(
        self,
        price: float,
        quantity: float,
        market_data: pd.Series,
    ) -> Dict[str, float]:
        """Estimate detailed transaction costs."""
        # Commission (fixed rate)
        commission = price * quantity * 0.001  # 0.1% commission

        # Spread cost
        spread = market_data.get("spread", price * 0.0001)  # Default 1bps spread
        spread_cost = spread * quantity / 2

        # Market impact cost
        impact = self.impact_model.estimate_total_impact(
            price, quantity, market_data.to_dict()
        )
        impact_cost = impact["impact_cost"]

        # Delay cost (opportunity cost)
        delay_cost = price * quantity * 0.0001  # Assume 1bps delay cost

        return {
            "commission": commission,
            "spread_cost": spread_cost,
            "impact_cost": impact_cost,
            "delay_cost": delay_cost,
        }

    def optimize_execution(
        self,
        trade: Trade,
        market_data: pd.DataFrame,
        constraints: Optional[Dict] = None,
    ) -> Dict:
        """
        Optimize execution parameters.

        Args:
            trade: Trade to optimize
            market_data: Market data for optimization
            constraints: Optional execution constraints

        Returns:
            Dict with optimal execution parameters
        """
        constraints = constraints or {}
        max_impact = constraints.get("max_impact", 0.01)
        target_duration = constraints.get("target_duration", 30)

        # Get optimal size from impact model
        optimal_size = self.impact_model.get_optimal_size(
            trade.entry_price,
            trade.quantity,
            market_data.iloc[0].to_dict(),
            max_impact,
        )

        # Calculate number of splits needed
        n_splits = max(1, int(np.ceil(trade.quantity / optimal_size)))

        # Generate execution schedule
        schedule = self.get_execution_schedule(trade, market_data, target_duration)

        # Estimate total costs
        total_costs = sum(
            self.estimate_transaction_costs(
                trade.entry_price,
                trade.quantity / n_splits,
                market_data.iloc[0],
            ).values()
        )

        return {
            "optimal_size": optimal_size,
            "n_splits": n_splits,
            "schedule": schedule,
            "estimated_costs": total_costs,
        }
