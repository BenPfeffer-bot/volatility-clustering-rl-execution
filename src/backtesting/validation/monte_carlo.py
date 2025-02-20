"""
Monte Carlo simulation framework for strategy validation.
Implements trade sequence randomization and scenario analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from ..strategies.base import Trade
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class MonteCarloSimulator:
    """
    Monte Carlo simulation framework that:
    - Randomizes trade sequences
    - Analyzes worst-case scenarios
    - Evaluates strategy robustness
    """

    def __init__(
        self,
        n_simulations: int = 10000,
        confidence_level: float = 0.95,
        initial_capital: float = 1_000_000,
    ):
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.initial_capital = initial_capital
        self.simulation_results: List[Dict] = []

    def run_simulations(self, trades: List[Trade]) -> Dict:
        """
        Run Monte Carlo simulations.

        Args:
            trades: List of historical trades

        Returns:
            Dict with simulation results
        """
        logger.info(f"Running {self.n_simulations} Monte Carlo simulations...")

        # Extract trade returns
        returns = [trade.pnl / trade.initial_value for trade in trades]
        durations = [trade.holding_time for trade in trades if trade.holding_time]

        # Initialize results storage
        equity_curves = []
        drawdowns = []
        sharpe_ratios = []
        max_consecutive_losses = []

        # Run simulations
        for i in range(self.n_simulations):
            # Randomize trade sequence
            shuffled_returns = np.random.choice(returns, size=len(returns))

            # Generate equity curve
            equity_curve = self._generate_equity_curve(shuffled_returns)
            equity_curves.append(equity_curve)

            # Calculate metrics for this simulation
            metrics = self._calculate_simulation_metrics(equity_curve, shuffled_returns)

            drawdowns.append(metrics["max_drawdown"])
            sharpe_ratios.append(metrics["sharpe_ratio"])
            max_consecutive_losses.append(
                self._calculate_max_consecutive_losses(shuffled_returns)
            )

            if (i + 1) % 1000 == 0:
                logger.info(f"Completed {i + 1} simulations")

        # Analyze results
        analysis = self._analyze_simulation_results(
            equity_curves, drawdowns, sharpe_ratios, max_consecutive_losses
        )

        return {
            "equity_curves": equity_curves,
            "metrics": analysis,
            "confidence_intervals": self._calculate_confidence_intervals(
                equity_curves, drawdowns, sharpe_ratios
            ),
        }

    def _generate_equity_curve(self, returns: np.ndarray) -> np.ndarray:
        """
        Generate equity curve from returns.

        Args:
            returns: Array of trade returns

        Returns:
            Array representing equity curve
        """
        # Convert returns to cumulative returns
        cumulative_returns = np.cumprod(1 + returns)

        # Generate equity curve
        equity_curve = self.initial_capital * cumulative_returns
        return equity_curve

    def _calculate_simulation_metrics(
        self,
        equity_curve: np.ndarray,
        returns: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate metrics for a single simulation.

        Args:
            equity_curve: Equity curve array
            returns: Array of returns

        Returns:
            Dict of metrics
        """
        if len(equity_curve) < 2 or len(returns) < 2:
            return {
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "total_return": 0.0,
                "volatility": 0.0,
            }

        # Calculate drawdown with validation
        try:
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - peak) / peak
            max_drawdown = abs(np.nanmin(drawdown))
        except (ValueError, ZeroDivisionError):
            max_drawdown = 0.0

        # Calculate Sharpe ratio with validation
        try:
            returns_mean = np.nanmean(returns)
            returns_std = np.nanstd(returns)
            sharpe = (
                np.sqrt(252) * returns_mean / returns_std if returns_std > 0 else 0.0
            )
        except (ValueError, ZeroDivisionError):
            sharpe = 0.0

        # Calculate other metrics with validation
        try:
            total_return = (
                (equity_curve[-1] / equity_curve[0]) - 1
                if len(equity_curve) > 1
                else 0.0
            )
            volatility = np.nanstd(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
        except (ValueError, ZeroDivisionError, IndexError):
            total_return = volatility = 0.0

        metrics = {
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "total_return": total_return,
            "volatility": volatility,
        }

        # Replace any NaN or infinite values with 0.0
        return {k: 0.0 if np.isnan(v) or np.isinf(v) else v for k, v in metrics.items()}

    def _calculate_max_consecutive_losses(self, returns: np.ndarray) -> int:
        """
        Calculate maximum consecutive losing trades.

        Args:
            returns: Array of trade returns

        Returns:
            Maximum number of consecutive losses
        """
        # Convert returns to binary win/loss
        wins = returns > 0

        # Calculate consecutive losses
        consecutive_losses = 0
        max_consecutive = 0

        for win in wins:
            if not win:
                consecutive_losses += 1
                max_consecutive = max(max_consecutive, consecutive_losses)
            else:
                consecutive_losses = 0

        return max_consecutive

    def _analyze_simulation_results(
        self,
        equity_curves: List[np.ndarray],
        drawdowns: List[float],
        sharpe_ratios: List[float],
        max_consecutive_losses: List[int],
    ) -> Dict:
        """
        Analyze results from all simulations.

        Args:
            equity_curves: List of equity curves
            drawdowns: List of maximum drawdowns
            sharpe_ratios: List of Sharpe ratios
            max_consecutive_losses: List of maximum consecutive losses

        Returns:
            Dict with analysis results
        """
        if (
            not equity_curves
            or not drawdowns
            or not sharpe_ratios
            or not max_consecutive_losses
        ):
            return {
                "worst_case": {
                    "drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                    "consecutive_losses": 0,
                },
                "probabilities": {
                    "drawdown_above_10pct": 0.0,
                    "sharpe_below_1": 0.0,
                    "consecutive_losses_above_5": 0.0,
                },
                "averages": {
                    "drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                    "consecutive_losses": 0,
                },
            }

        # Convert lists to arrays and handle NaN/inf values
        drawdowns = np.array(
            [0.0 if np.isnan(x) or np.isinf(x) else x for x in drawdowns]
        )
        sharpe_ratios = np.array(
            [0.0 if np.isnan(x) or np.isinf(x) else x for x in sharpe_ratios]
        )
        max_consecutive_losses = np.array(max_consecutive_losses)

        # Calculate worst-case metrics with validation
        worst_drawdown = np.percentile(drawdowns, 99) if len(drawdowns) > 0 else 0.0
        worst_sharpe = (
            np.percentile(sharpe_ratios, 1) if len(sharpe_ratios) > 0 else 0.0
        )
        worst_consecutive_losses = (
            np.max(max_consecutive_losses) if len(max_consecutive_losses) > 0 else 0
        )

        # Calculate probabilities with validation
        prob_drawdown_above_10 = (
            np.mean(drawdowns > 0.10) if len(drawdowns) > 0 else 0.0
        )
        prob_sharpe_below_1 = (
            np.mean(sharpe_ratios < 1.0) if len(sharpe_ratios) > 0 else 0.0
        )
        prob_consecutive_losses_above_5 = (
            np.mean(max_consecutive_losses > 5)
            if len(max_consecutive_losses) > 0
            else 0.0
        )

        return {
            "worst_case": {
                "drawdown": worst_drawdown,
                "sharpe_ratio": worst_sharpe,
                "consecutive_losses": worst_consecutive_losses,
            },
            "probabilities": {
                "drawdown_above_10pct": prob_drawdown_above_10,
                "sharpe_below_1": prob_sharpe_below_1,
                "consecutive_losses_above_5": prob_consecutive_losses_above_5,
            },
            "averages": {
                "drawdown": np.nanmean(drawdowns),
                "sharpe_ratio": np.nanmean(sharpe_ratios),
                "consecutive_losses": np.nanmean(max_consecutive_losses),
            },
        }

    def _calculate_confidence_intervals(
        self,
        equity_curves: List[np.ndarray],
        drawdowns: List[float],
        sharpe_ratios: List[float],
    ) -> Dict:
        """
        Calculate confidence intervals for key metrics.

        Args:
            equity_curves: List of equity curves
            drawdowns: List of maximum drawdowns
            sharpe_ratios: List of Sharpe ratios

        Returns:
            Dict with confidence intervals
        """
        alpha = 1 - self.confidence_level

        # Calculate confidence intervals
        drawdown_ci = np.percentile(drawdowns, [alpha * 100, (1 - alpha) * 100])
        sharpe_ci = np.percentile(sharpe_ratios, [alpha * 100, (1 - alpha) * 100])

        # Calculate final equity confidence interval
        final_equities = [curve[-1] for curve in equity_curves]
        equity_ci = np.percentile(final_equities, [alpha * 100, (1 - alpha) * 100])

        return {
            "drawdown": {
                "lower": drawdown_ci[0],
                "upper": drawdown_ci[1],
            },
            "sharpe_ratio": {
                "lower": sharpe_ci[0],
                "upper": sharpe_ci[1],
            },
            "final_equity": {
                "lower": equity_ci[0],
                "upper": equity_ci[1],
            },
        }

    def generate_report(self) -> str:
        """Generate detailed Monte Carlo analysis report."""
        if not self.simulation_results:
            return "No simulation results available"

        report = []
        report.append("Monte Carlo Simulation Report")
        report.append("=" * 50)

        # Get latest results
        latest = self.simulation_results[-1]
        metrics = latest["metrics"]
        ci = latest["confidence_intervals"]

        # Worst-case scenario analysis
        report.append("\nWorst-Case Scenario Analysis:")
        report.append(
            f"Maximum Drawdown (99th): {metrics['worst_case']['drawdown']:.2%}"
        )
        report.append(
            f"Minimum Sharpe (1st): {metrics['worst_case']['sharpe_ratio']:.2f}"
        )
        report.append(
            f"Max Consecutive Losses: {metrics['worst_case']['consecutive_losses']}"
        )

        # Risk probabilities
        report.append("\nRisk Probabilities:")
        report.append(
            f"P(Drawdown > 10%): {metrics['probabilities']['drawdown_above_10pct']:.2%}"
        )
        report.append(
            f"P(Sharpe < 1): {metrics['probabilities']['sharpe_below_1']:.2%}"
        )
        report.append(
            f"P(Consecutive Losses > 5): "
            f"{metrics['probabilities']['consecutive_losses_above_5']:.2%}"
        )

        # Confidence intervals
        report.append(f"\n{self.confidence_level:.0%} Confidence Intervals:")
        report.append(
            f"Drawdown: [{ci['drawdown']['lower']:.2%}, {ci['drawdown']['upper']:.2%}]"
        )
        report.append(
            f"Sharpe: [{ci['sharpe_ratio']['lower']:.2f}, "
            f"{ci['sharpe_ratio']['upper']:.2f}]"
        )
        report.append(
            f"Final Equity: [{ci['final_equity']['lower']:,.0f}, "
            f"{ci['final_equity']['upper']:,.0f}]"
        )

        return "\n".join(report)
