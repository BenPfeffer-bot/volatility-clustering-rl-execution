"""
Position optimization module for determining optimal position sizes and timing.
Implements Kelly criterion and risk-adjusted position sizing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from ..strategies.base import Trade
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class PositionOptimizer:
    """
    Position optimization system that determines:
    - Optimal position sizes using Kelly criterion
    - Entry/exit timing based on market conditions
    - Risk-adjusted position scaling
    """

    def __init__(
        self,
        max_position_size: float = 0.1,  # Max 10% position size
        kelly_fraction: float = 0.5,  # Half-Kelly for conservative sizing
        min_win_rate: float = 0.55,  # Minimum required win rate
        vol_target: float = 0.15,  # Annual volatility target
    ):
        self.max_position_size = max_position_size
        self.kelly_fraction = kelly_fraction
        self.min_win_rate = min_win_rate
        self.vol_target = vol_target
        self.position_history: Dict[str, pd.Series] = {}

    def calculate_kelly_fraction(
        self,
        win_rate: float,
        win_loss_ratio: float,
        risk_free_rate: float = 0.02,
    ) -> float:
        """
        Calculate optimal Kelly fraction.

        Args:
            win_rate: Probability of winning trade
            win_loss_ratio: Ratio of average win to average loss
            risk_free_rate: Annual risk-free rate

        Returns:
            Optimal position size as fraction of portfolio
        """
        if win_rate < self.min_win_rate:
            return 0.0

        # Classic Kelly formula
        q = 1 - win_rate
        kelly = win_rate - (q / win_loss_ratio)

        # Adjust for risk-free rate
        kelly = kelly * (1 - risk_free_rate)

        # Apply fraction for conservative sizing
        kelly = kelly * self.kelly_fraction

        return min(kelly, self.max_position_size)

    def calculate_position_size(
        self,
        win_rate: float,
        volatility: float,
        market_data: pd.Series,
        regime: str,
    ) -> float:
        """
        Calculate optimal position size considering multiple factors.

        Args:
            win_rate: Historical win rate
            volatility: Current market volatility
            market_data: Current market conditions
            regime: Current market regime

        Returns:
            Optimal position size as fraction of portfolio
        """
        # Base size from Kelly criterion
        if "avg_win" in market_data and "avg_loss" in market_data:
            win_loss_ratio = abs(market_data["avg_win"] / market_data["avg_loss"])
        else:
            win_loss_ratio = 1.5  # Default assumption

        base_size = self.calculate_kelly_fraction(win_rate, win_loss_ratio)

        # Volatility adjustment
        vol_scalar = self.vol_target / (volatility * np.sqrt(252))
        vol_adj_size = base_size * vol_scalar

        # Regime-based adjustment
        regime_factors = {
            "institutional_flow": 1.2,  # Increase size
            "high_impact": 0.7,  # Reduce size
            "trending": 1.0,  # Normal size
            "neutral": 0.8,  # Slightly reduced
        }
        regime_mult = regime_factors.get(regime, 0.8)

        # Market condition adjustments
        vpin = market_data.get("vpin", 0.5)
        vpin_adj = np.clip(vpin / 0.5, 0.5, 1.5)

        # Final position size
        position_size = vol_adj_size * regime_mult * vpin_adj
        return min(position_size, self.max_position_size)

    def optimize_entry_timing(
        self,
        signal: float,
        market_data: pd.DataFrame,
        lookback: int = 20,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Optimize trade entry timing.

        Args:
            signal: Trading signal (-1, 0, 1)
            market_data: Recent market data
            lookback: Lookback period for analysis

        Returns:
            Tuple of (should_enter, timing_metrics)
        """
        if abs(signal) == 0:
            return False, {}

        # Calculate timing metrics
        volatility = market_data["daily_volatility"].rolling(lookback).mean().iloc[-1]
        vpin = market_data["vpin"].rolling(lookback).mean().iloc[-1]

        # Volume profile
        volume = market_data["volume"].rolling(lookback).mean()
        vol_percentile = (
            volume.iloc[-1] / volume.quantile(0.9) if len(volume) > lookback else 0.5
        )

        # Trend strength
        returns = market_data["close"].pct_change()
        trend_strength = abs(returns.rolling(lookback).mean().iloc[-1])

        # Timing score components
        timing_metrics = {
            "volatility_score": 1 - (volatility / volatility.max()),
            "vpin_score": vpin if signal > 0 else (1 - vpin),
            "volume_score": vol_percentile,
            "trend_score": trend_strength,
        }

        # Composite timing score
        timing_score = np.mean(list(timing_metrics.values()))
        timing_metrics["composite_score"] = timing_score

        # Decision threshold
        should_enter = timing_score > 0.6

        return should_enter, timing_metrics

    def optimize_exit_timing(
        self,
        trade: Trade,
        market_data: pd.Series,
        profit_target: float = 0.0075,  # 0.75% target return
        stop_loss: float = 0.02,  # 2% stop loss
    ) -> Tuple[bool, str]:
        """
        Optimize trade exit timing.

        Args:
            trade: Active trade
            market_data: Current market data
            profit_target: Target return for exit
            stop_loss: Stop loss level

        Returns:
            Tuple of (should_exit, reason)
        """
        if not trade or not trade.entry_time:
            return False, ""

        # Calculate current return
        current_return = (market_data["close"] - trade.entry_price) / trade.entry_price

        # Basic profit/loss exits
        if current_return <= -stop_loss:
            return True, "stop_loss"
        if current_return >= profit_target:
            return True, "profit_target"

        # VPIN-based exit
        if market_data["vpin"] < 0.5 and current_return > 0:
            return True, "vpin_reversal"

        # Volatility-based exit
        if (
            market_data["daily_volatility"]
            > market_data["daily_volatility"].rolling(20).mean().iloc[-1]
        ):
            return True, "volatility_spike"

        return False, ""

    def update_position_history(
        self,
        trade: Trade,
        market_data: pd.Series,
    ) -> None:
        """Update position history for analysis."""
        if not trade.exit_time:
            return

        # Record position metrics
        metrics = pd.Series(
            {
                "size": trade.quantity,
                "return": trade.pnl / trade.initial_value if trade.pnl else 0,
                "duration": trade.holding_time if trade.holding_time else 0,
                "vpin": market_data.get("vpin", 0),
                "volatility": market_data.get("daily_volatility", 0),
            }
        )

        # Store in history
        self.position_history[trade.entry_time.strftime("%Y%m%d_%H%M%S")] = metrics

    def generate_position_report(self) -> str:
        """Generate position optimization report."""
        if not self.position_history:
            return "No position history available"

        # Convert history to DataFrame
        history_df = pd.DataFrame(self.position_history).T

        report = []
        report.append("Position Optimization Report")
        report.append("=" * 50)

        # Size analysis
        report.append("\nPosition Sizing Analysis:")
        report.append(f"Average Position Size: {history_df['size'].mean():.2f}")
        report.append(
            f"Size Range: {history_df['size'].min():.2f} - {history_df['size'].max():.2f}"
        )

        # Return analysis
        report.append("\nReturn Analysis:")
        report.append(f"Average Return: {history_df['return'].mean():.2%}")
        report.append(f"Return Std Dev: {history_df['return'].std():.2%}")
        report.append(f"Win Rate: {(history_df['return'] > 0).mean():.2%}")

        # Timing analysis
        report.append("\nTiming Analysis:")
        report.append(f"Average Duration: {history_df['duration'].mean():.1f} minutes")
        report.append(f"VPIN at Entry (avg): {history_df['vpin'].mean():.3f}")
        report.append(
            f"Volatility at Entry (avg): {history_df['volatility'].mean():.3f}"
        )

        return "\n".join(report)
