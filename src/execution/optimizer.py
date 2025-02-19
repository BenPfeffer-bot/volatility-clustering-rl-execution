"""
Trade Execution Optimizer with Adaptive Risk Management.

This module combines market impact predictions with Bayesian position sizing
to optimize trade execution while managing risk.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.utils.log_utils import setup_logging
from src.execution.position_sizer import BayesianPositionSizer, PositionConfig

logger = setup_logging(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for trade execution optimization."""

    ticker: str
    position_config: PositionConfig
    min_vpin_signal: float = 0.7  # Minimum VPIN for trade entry
    max_adverse_impact: float = 0.02  # Maximum acceptable adverse market impact
    min_hurst_threshold: float = 0.55  # Minimum Hurst exponent for trend following
    max_positions: int = 3  # Maximum number of concurrent positions
    regime_window: int = 50  # Window for regime detection
    volatility_threshold: float = 0.02  # High volatility threshold


class ExecutionOptimizer:
    """
    Optimizes trade execution using market impact predictions and Bayesian position sizing.

    Key features:
    1. Dynamic position sizing
    2. Adaptive risk management
    3. Market impact-aware execution
    4. Multi-timeframe optimization
    """

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.position_sizer = BayesianPositionSizer(config.position_config)

        # Track open positions and performance
        self.open_positions: List[Dict] = []
        self.completed_trades: List[Dict] = []
        self.current_risk: float = 0.0

        # Market regime tracking
        self.volatility_history: List[float] = []
        self.vpin_history: List[float] = []
        self.current_regime: str = "normal"  # normal, high_vol, or low_vol

    def _update_market_regime(self, features: Dict[str, float]):
        """Update market regime based on recent market conditions."""
        # Update history
        self.volatility_history.append(features["daily_volatility"])
        self.vpin_history.append(features["vpin"])

        # Keep history within window
        if len(self.volatility_history) > self.config.regime_window:
            self.volatility_history.pop(0)
            self.vpin_history.pop(0)

        # Calculate regime metrics
        avg_vol = np.mean(self.volatility_history)
        vol_std = np.std(self.volatility_history)
        avg_vpin = np.mean(self.vpin_history)

        # Determine regime
        if avg_vol > self.config.volatility_threshold:
            if avg_vpin > 0.7:  # High volatility with strong institutional activity
                self.current_regime = "institutional"
            else:  # High volatility without clear institutional activity
                self.current_regime = "high_vol"
        elif vol_std < 0.005:  # Very stable volatility
            self.current_regime = "low_vol"
        else:
            self.current_regime = "normal"

        logger.info(f"Current market regime: {self.current_regime}")

    def _adjust_for_regime(
        self, position_size: float, features: Dict[str, float]
    ) -> float:
        """Adjust position size based on current market regime."""
        regime_adjustments = {
            "high_vol": 0.5,  # Reduce size in high volatility
            "low_vol": 1.2,  # Increase size in low volatility
            "institutional": 1.5,  # Increase size during institutional activity
            "normal": 1.0,  # No adjustment in normal conditions
        }

        # Apply regime-specific adjustment
        position_size *= regime_adjustments[self.current_regime]

        # Additional adjustments for specific conditions
        if self.current_regime == "high_vol":
            # Tighter risk controls in high volatility
            self.config.position_config.risk_limit *= 0.8
        elif self.current_regime == "institutional":
            # More aggressive during clear institutional activity
            if features["hurst"] > 0.65:  # Strong trend persistence
                position_size *= 1.2

        return position_size

    def process_market_update(
        self,
        timestamp: pd.Timestamp,
        price: float,
        features: Dict[str, float],
        market_impact_pred: float,
    ) -> Optional[Dict]:
        """
        Process market update and generate trading decisions.

        Args:
            timestamp: Current timestamp
            price: Current price
            features: Dictionary of current feature values
            market_impact_pred: Predicted market impact

        Returns:
            Optional[Dict]: Trading decision if any
        """
        # Update market regime
        self._update_market_regime(features)

        # Update risk metrics for open positions
        self._update_positions(price, features)

        # Check if we can take new positions
        if len(self.open_positions) >= self.config.max_positions:
            return None

        # Generate trading decision
        decision = self._generate_trading_decision(
            timestamp, price, features, market_impact_pred
        )

        if decision:
            # Adjust position size for current regime
            decision["size"] = self._adjust_for_regime(decision["size"], features)

        return decision

    def _generate_trading_decision(
        self,
        timestamp: pd.Timestamp,
        price: float,
        features: Dict[str, float],
        market_impact_pred: float,
    ) -> Optional[Dict]:
        """
        Generate trading decision based on current market conditions.

        Args:
            timestamp: Current timestamp
            price: Current price
            features: Dictionary of current feature values
            market_impact_pred: Predicted market impact

        Returns:
            Optional[Dict]: Trading decision if conditions are met
        """
        # Check entry conditions
        if not self._check_entry_conditions(features, market_impact_pred):
            return None

        # Calculate position size
        position_size = self.position_sizer.calculate_position_size(
            features["vpin"], features["daily_volatility"], market_impact_pred
        )

        # Calculate risk levels
        stop_loss, take_profit = self.position_sizer.calculate_risk_levels(
            price, features["daily_volatility"], features["vpin"]
        )

        # Determine trade direction based on features
        direction = self._determine_trade_direction(features, market_impact_pred)
        if direction == 0:
            return None

        return {
            "timestamp": timestamp,
            "type": "MARKET",
            "direction": direction,
            "size": position_size,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "features": features.copy(),
            "market_impact_pred": market_impact_pred,
        }

    def _check_entry_conditions(
        self, features: Dict[str, float], market_impact_pred: float
    ) -> bool:
        """Check if entry conditions are met."""
        # Base conditions
        if not super()._check_entry_conditions(features, market_impact_pred):
            return False

        # Regime-specific conditions
        if self.current_regime == "high_vol":
            # More stringent conditions in high volatility
            if features["hurst"] < 0.6:  # Require stronger trend persistence
                return False
            if abs(market_impact_pred) > 0.015:  # Lower impact threshold
                return False
        elif self.current_regime == "low_vol":
            # More relaxed conditions in low volatility
            if features["vpin"] > 0.6:  # Lower VPIN threshold
                return True

        return True

    def _determine_trade_direction(
        self, features: Dict[str, float], market_impact_pred: float
    ) -> int:
        """
        Determine trade direction (1: Long, -1: Short, 0: No trade).

        Uses a combination of:
        1. VPIN for order flow imbalance
        2. Market impact prediction
        3. Volume persistence (Hurst exponent)
        """
        # Strong institutional buying
        if (
            features["vpin"] > 0.7
            and market_impact_pred > 0
            and features["hurst"] > 0.6
        ):
            return 1

        # Strong institutional selling
        if (
            features["vpin"] > 0.7
            and market_impact_pred < 0
            and features["hurst"] > 0.6
        ):
            return -1

        return 0

    def _update_positions(self, price: float, features: Dict[str, float]):
        """Update open positions and risk metrics."""
        remaining_positions = []

        for position in self.open_positions:
            # Check stop loss and take profit
            if self._check_exit_conditions(position, price, features):
                # Close position
                exit_price = price  # In reality, would include slippage
                pnl = (
                    (exit_price - position["price"])
                    * position["direction"]
                    * position["size"]
                )

                # Record trade result
                trade_result = {
                    "entry_time": position["timestamp"],
                    "exit_time": pd.Timestamp.now(),
                    "direction": position["direction"],
                    "size": position["size"],
                    "entry_price": position["price"],
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "vpin_beta": position["features"]["vpin"],
                    "vol_beta": position["features"]["daily_volatility"],
                }

                self.completed_trades.append(trade_result)
                self.position_sizer.update_model(trade_result)

                # Update risk
                self.current_risk -= self.config.position_config.risk_limit
            else:
                remaining_positions.append(position)

        self.open_positions = remaining_positions

    def _check_exit_conditions(
        self, position: Dict, price: float, features: Dict[str, float]
    ) -> bool:
        """Check if position should be exited."""
        # Stop loss hit
        if (position["direction"] == 1 and price <= position["stop_loss"]) or (
            position["direction"] == -1 and price >= position["stop_loss"]
        ):
            return True

        # Take profit hit
        if (position["direction"] == 1 and price >= position["take_profit"]) or (
            position["direction"] == -1 and price <= position["take_profit"]
        ):
            return True

        # VPIN reversal
        if (position["direction"] == 1 and features["vpin"] < 0.3) or (
            position["direction"] == -1 and features["vpin"] > 0.7
        ):
            return True

        # Hurst exponent breakdown
        if features["hurst"] < 0.4:  # Strong mean reversion
            return True

        return False

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics for the strategy.

        Returns:
            Dict with performance metrics including:
            - Sharpe ratio
            - Win rate
            - Average trade return
            - Maximum drawdown
            - Risk-adjusted return
            - Average holding time
        """
        if not self.completed_trades:
            return {}

        # Calculate returns
        returns = [trade["pnl"] for trade in self.completed_trades]

        # Calculate holding times
        holding_times = [
            (trade["exit_time"] - trade["entry_time"]).total_seconds()
            / 60  # Convert to minutes
            for trade in self.completed_trades
        ]

        # Basic metrics
        total_trades = len(returns)
        winning_trades = sum(1 for r in returns if r > 0)

        metrics = {
            "total_trades": total_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "avg_return": np.mean(returns) if returns else 0,
            "return_std": np.std(returns) if len(returns) > 1 else 0,
            "avg_holding_time": np.mean(holding_times) if holding_times else 0,
            "min_holding_time": np.min(holding_times) if holding_times else 0,
            "max_holding_time": np.max(holding_times) if holding_times else 0,
        }

        # Add holding time distribution
        if holding_times:
            metrics["pct_trades_in_target_range"] = sum(
                1 for t in holding_times if 10 <= t <= 30
            ) / len(holding_times)

        # Flag if holding times are outside target range
        if metrics["avg_holding_time"] < 10 or metrics["avg_holding_time"] > 30:
            logger.warning(
                f"Average holding time ({metrics['avg_holding_time']:.1f} min) "
                "outside target range (10-30 min)"
            )

        # Calculate Sharpe ratio (assuming risk-free rate = 0)
        if metrics["return_std"] > 0:
            metrics["sharpe_ratio"] = (
                metrics["avg_return"] / metrics["return_std"] * np.sqrt(252)
            )  # Annualized

        # Maximum drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        metrics["max_drawdown"] = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # Risk-adjusted metrics
        metrics["calmar_ratio"] = (
            abs(metrics["avg_return"] / metrics["max_drawdown"])
            if metrics["max_drawdown"] > 0
            else 0
        )

        # Regime-specific performance
        regime_returns = {}
        for trade in self.completed_trades:
            regime = trade.get("regime", "normal")
            if regime not in regime_returns:
                regime_returns[regime] = []
            regime_returns[regime].append(trade["pnl"])

        for regime, rets in regime_returns.items():
            metrics[f"{regime}_avg_return"] = np.mean(rets)
            metrics[f"{regime}_sharpe"] = (
                np.mean(rets) / np.std(rets) * np.sqrt(252)
                if len(rets) > 1 and np.std(rets) > 0
                else 0
            )

        return metrics

    def get_risk_report(self) -> Dict[str, float]:
        """
        Generate risk management report.

        Returns:
            Dict with risk metrics including:
            - Current risk exposure
            - VaR (Value at Risk)
            - Expected Shortfall
            - Position concentration
            - Regime risk indicators
        """
        risk_report = {
            "current_risk": self.current_risk,
            "open_positions": len(self.open_positions),
            "risk_capacity": 0.1 - self.current_risk,  # Remaining risk capacity
        }

        if self.completed_trades:
            returns = [trade["pnl"] for trade in self.completed_trades]

            # Calculate VaR (95% confidence)
            risk_report["var_95"] = np.percentile(returns, 5)

            # Expected Shortfall (CVaR)
            risk_report["expected_shortfall"] = np.mean(
                [r for r in returns if r <= risk_report["var_95"]]
            )

            # Position concentration
            position_sizes = [pos["size"] for pos in self.open_positions]
            risk_report["position_concentration"] = (
                np.std(position_sizes) / np.mean(position_sizes)
                if position_sizes and np.mean(position_sizes) > 0
                else 0
            )

        # Regime risk indicators
        risk_report["current_regime"] = self.current_regime
        risk_report["regime_risk_multiplier"] = {
            "high_vol": 1.5,
            "low_vol": 0.8,
            "institutional": 1.2,
            "normal": 1.0,
        }[self.current_regime]

        # Market condition indicators
        if self.volatility_history:
            risk_report["volatility_trend"] = (
                np.mean(self.volatility_history[-10:])
                / np.mean(self.volatility_history[-30:-10])
                if len(self.volatility_history) >= 30
                else 1.0
            )

        if self.vpin_history:
            risk_report["vpin_trend"] = (
                np.mean(self.vpin_history[-10:]) / np.mean(self.vpin_history[-30:-10])
                if len(self.vpin_history) >= 30
                else 1.0
            )

        return risk_report

    def log_performance_update(self):
        """Log current performance and risk metrics."""
        metrics = self.get_performance_metrics()
        risk_report = self.get_risk_report()

        logger.info("=== Performance Update ===")
        logger.info(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
        logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"Avg Holding Time: {metrics.get('avg_holding_time', 0):.1f} min")
        logger.info(
            f"Trades in Target Range: {metrics.get('pct_trades_in_target_range', 0):.1%}"
        )

        logger.info("=== Risk Report ===")
        logger.info(f"Current Regime: {risk_report['current_regime']}")
        logger.info(f"Risk Exposure: {risk_report['current_risk']:.2%}")
        logger.info(f"VaR (95%): {risk_report.get('var_95', 0):.2%}")
        logger.info(
            f"Position Concentration: {risk_report.get('position_concentration', 0):.2f}"
        )
