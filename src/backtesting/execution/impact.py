"""
Market impact model for estimating price impact of institutional trades.
Implements temporary and permanent impact calculations based on order size and market conditions.
"""

import numpy as np
from typing import Dict, Optional
from datetime import datetime
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class MarketImpactModel:
    """
    Market impact model that estimates:
    - Temporary impact (spread + volatility)
    - Permanent impact (information leakage)
    - VPIN-based impact adjustments
    """

    def __init__(
        self,
        spread_factor: float = 0.0001,  # 1bps base spread impact
        vol_factor: float = 0.1,  # Volatility scaling factor
        size_factor: float = 0.1,  # Order size impact factor
        vpin_factor: float = 0.2,  # VPIN impact scaling
    ):
        self.spread_factor = spread_factor
        self.vol_factor = vol_factor
        self.size_factor = size_factor
        self.vpin_factor = vpin_factor

    def calculate_temporary_impact(
        self,
        price: float,
        quantity: float,
        volatility: float,
        volume: float,
        spread: Optional[float] = None,
    ) -> float:
        """
        Calculate temporary price impact.

        Args:
            price: Current price
            quantity: Order quantity
            volatility: Current volatility
            volume: Market volume
            spread: Optional bid-ask spread

        Returns:
            Estimated temporary impact as fraction of price
        """
        # Base spread impact
        spread_impact = self.spread_factor
        if spread is not None:
            spread_impact = spread / 2

        # Volatility impact
        vol_impact = volatility * self.vol_factor

        # Size impact using square root model
        participation = quantity / volume if volume > 0 else 0
        size_impact = self.size_factor * np.sqrt(participation)

        # Combine impacts
        total_impact = spread_impact + vol_impact + size_impact
        return total_impact

    def calculate_permanent_impact(
        self,
        price: float,
        quantity: float,
        volume: float,
        vpin: float,
        regime: str,
    ) -> float:
        """
        Calculate permanent price impact.

        Args:
            price: Current price
            quantity: Order quantity
            volume: Market volume
            vpin: Current VPIN
            regime: Market regime

        Returns:
            Estimated permanent impact as fraction of price
        """
        # Base impact from order size
        participation = quantity / volume if volume > 0 else 0
        base_impact = self.size_factor * participation

        # VPIN adjustment
        vpin_impact = self.vpin_factor * (vpin - 0.5)  # Center around 0.5

        # Regime adjustment
        regime_factors = {
            "institutional_flow": 1.5,  # Higher impact in institutional flow
            "high_impact": 2.0,  # Highest impact
            "trending": 1.2,  # Moderate impact
            "neutral": 1.0,  # Base impact
        }
        regime_mult = regime_factors.get(regime, 1.0)

        # Calculate total permanent impact
        total_impact = (base_impact + vpin_impact) * regime_mult
        return max(0, total_impact)  # Ensure non-negative

    def estimate_total_impact(
        self,
        price: float,
        quantity: float,
        market_data: Dict,
    ) -> Dict[str, float]:
        """
        Estimate total price impact of a trade.

        Args:
            price: Current price
            quantity: Order quantity
            market_data: Dict with market metrics

        Returns:
            Dict with impact estimates
        """
        # Extract market data
        volatility = market_data.get("daily_volatility", 0.01)
        volume = market_data.get(
            "volume", quantity * 10
        )  # Fallback to 10% participation
        spread = market_data.get("spread", None)
        vpin = market_data.get("vpin", 0.5)
        regime = market_data.get("regime", "neutral")

        # Calculate impacts
        temp_impact = self.calculate_temporary_impact(
            price, quantity, volatility, volume, spread
        )
        perm_impact = self.calculate_permanent_impact(
            price, quantity, volume, vpin, regime
        )

        # Total impact estimate
        total_impact = temp_impact + perm_impact

        return {
            "temporary_impact": temp_impact,
            "permanent_impact": perm_impact,
            "total_impact": total_impact,
            "impact_cost": total_impact * price * quantity,
        }

    def get_optimal_size(
        self,
        price: float,
        max_quantity: float,
        market_data: Dict,
        max_impact: float = 0.01,  # Max 1% price impact
    ) -> float:
        """
        Calculate optimal order size to minimize impact.

        Args:
            price: Current price
            max_quantity: Maximum order size
            market_data: Dict with market metrics
            max_impact: Maximum acceptable impact

        Returns:
            Optimal order quantity
        """
        # Binary search for optimal size
        left, right = 0, max_quantity
        optimal_size = 0

        while left <= right:
            mid = (left + right) / 2
            impact = self.estimate_total_impact(price, mid, market_data)["total_impact"]

            if impact <= max_impact:
                optimal_size = mid
                left = mid + 1
            else:
                right = mid - 1

        return optimal_size
