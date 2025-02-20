"""
Risk management system for monitoring and controlling trading risks.
Implements position sizing, exposure limits, and drawdown controls.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from ..strategies.base import Trade
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class RiskManager:
    """
    Risk management system that enforces:
    - Maximum position sizes
    - Exposure limits
    - Drawdown controls
    - Risk factor monitoring
    """

    def __init__(
        self,
        max_position_size: float = 0.1,  # Max 10% of portfolio in single position
        max_drawdown: float = 0.04,  # Max 4% drawdown target
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
        vol_target: float = 0.15,  # 15% annualized volatility target
    ):
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.risk_free_rate = risk_free_rate
        self.vol_target = vol_target
        self.current_drawdown = 0.0
        self.peak_value = 0.0
        self.risk_metrics: Dict[str, float] = {}

    def calculate_position_size(
        self,
        portfolio_value: float,
        volatility: float,
        vpin: float,
        regime: str,
    ) -> float:
        """
        Calculate risk-adjusted position size.

        Args:
            portfolio_value: Current portfolio value
            volatility: Current market volatility
            vpin: Current VPIN value
            regime: Current market regime

        Returns:
            Position size as fraction of portfolio
        """
        # Base size from volatility targeting
        vol_scalar = self.vol_target / (volatility * np.sqrt(252))
        base_size = min(self.max_position_size, vol_scalar)

        # Adjust for market regime
        regime_scalars = {
            "institutional_flow": 1.2,
            "high_impact": 0.7,
            "trending": 1.0,
            "neutral": 0.8,
        }
        regime_adj = regime_scalars.get(regime, 0.8)

        # Adjust for VPIN
        vpin_scalar = np.clip(vpin, 0.5, 1.5)

        # Calculate final size
        position_size = base_size * regime_adj * vpin_scalar

        # Apply drawdown control
        if self.current_drawdown > self.max_drawdown * 0.75:
            position_size *= 0.5  # Reduce size when near drawdown limit

        return min(position_size, self.max_position_size)

    def update_risk_metrics(
        self,
        portfolio_value: float,
        trades: List[Trade],
        returns: pd.Series,
    ) -> Dict[str, float]:
        """
        Update risk metrics based on current portfolio state.

        Args:
            portfolio_value: Current portfolio value
            trades: List of completed trades
            returns: Series of portfolio returns

        Returns:
            Dictionary of risk metrics
        """
        # Update peak value and drawdown
        self.peak_value = max(self.peak_value, portfolio_value)
        self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value

        # Calculate risk metrics
        if len(returns) > 1:
            vol = returns.std() * np.sqrt(252)
            sharpe = (returns.mean() * 252 - self.risk_free_rate) / (
                returns.std() * np.sqrt(252)
            )
            sortino = (returns.mean() * 252 - self.risk_free_rate) / (
                returns[returns < 0].std() * np.sqrt(252)
            )
        else:
            vol = sharpe = sortino = 0.0

        # Calculate win rate and profit factor
        if trades:
            winning_trades = sum(1 for t in trades if t.pnl > 0)
            win_rate = winning_trades / len(trades)

            gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
            profit_factor = (
                gross_profit / gross_loss if gross_loss != 0 else float("inf")
            )
        else:
            win_rate = profit_factor = 0.0

        self.risk_metrics = {
            "current_drawdown": self.current_drawdown,
            "volatility": vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }

        return self.risk_metrics

    def check_risk_limits(self) -> Tuple[bool, Optional[str]]:
        """
        Check if any risk limits are breached.

        Returns:
            Tuple of (is_breached, reason)
        """
        if self.current_drawdown > self.max_drawdown:
            return True, f"Max drawdown breached: {self.current_drawdown:.1%}"

        if self.risk_metrics.get("volatility", 0) > self.vol_target * 1.5:
            return True, "Volatility target exceeded"

        if self.risk_metrics.get("win_rate", 1) < 0.4:
            return True, "Win rate below critical level"

        return False, None

    def generate_risk_report(self) -> str:
        """Generate detailed risk report."""
        report = []
        report.append("Risk Management Report")
        report.append("=" * 50)

        report.append("\nRisk Metrics:")
        for metric, value in self.risk_metrics.items():
            report.append(f"{metric}: {value:.2%}")

        report.append("\nRisk Limits:")
        report.append(f"Max Position Size: {self.max_position_size:.1%}")
        report.append(f"Max Drawdown: {self.max_drawdown:.1%}")
        report.append(f"Volatility Target: {self.vol_target:.1%}")

        is_breached, reason = self.check_risk_limits()
        report.append("\nRisk Status:")
        report.append(f"Limits Breached: {'Yes' if is_breached else 'No'}")
        if reason:
            report.append(f"Reason: {reason}")

        return "\n".join(report)
