"""
Transaction cost sensitivity analysis framework.
Evaluates strategy performance under different cost assumptions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from ..strategies.base import BaseStrategy
from ..core.engine import BacktestEngine
from ..performance.metrics import PerformanceMetrics
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class CostSensitivityAnalyzer:
    """
    Transaction cost sensitivity analysis framework that:
    - Tests different slippage models
    - Evaluates commission impact
    - Analyzes market impact assumptions
    """

    def __init__(
        self,
        base_commission: float = 0.001,  # 0.1% base commission
        base_slippage: float = 0.0001,  # 1bps base slippage
    ):
        self.base_commission = base_commission
        self.base_slippage = base_slippage
        self.analysis_results: List[Dict] = []

    def run_analysis(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        commission_range: Optional[List[float]] = None,
        slippage_range: Optional[List[float]] = None,
        impact_models: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run sensitivity analysis across different cost parameters.

        Args:
            strategy: Strategy to test
            data: Market data
            commission_range: List of commission rates to test
            slippage_range: List of slippage rates to test
            impact_models: List of impact models to test

        Returns:
            Dict with analysis results
        """
        logger.info("Starting transaction cost sensitivity analysis...")

        # Set default ranges if not provided
        commission_range = commission_range or [
            0.0005,  # 0.05%
            0.001,  # 0.10%
            0.002,  # 0.20%
            0.003,  # 0.30%
        ]

        slippage_range = slippage_range or [
            0.0001,  # 1bps
            0.0002,  # 2bps
            0.0005,  # 5bps
            0.001,  # 10bps
        ]

        impact_models = impact_models or [
            "fixed",
            "volume_based",
            "adaptive",
        ]

        # Initialize results storage
        results = {
            "commission_sensitivity": self._analyze_commission_sensitivity(
                strategy, data, commission_range
            ),
            "slippage_sensitivity": self._analyze_slippage_sensitivity(
                strategy, data, slippage_range
            ),
            "impact_model_comparison": self._compare_impact_models(
                strategy, data, impact_models
            ),
        }

        # Store results
        self.analysis_results.append(results)

        return results

    def _analyze_commission_sensitivity(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        commission_range: List[float],
    ) -> Dict:
        """
        Analyze sensitivity to different commission rates.

        Args:
            strategy: Strategy to test
            data: Market data
            commission_range: List of commission rates to test

        Returns:
            Dict with commission sensitivity analysis
        """
        results = []

        for commission in commission_range:
            # Run backtest with current commission rate
            engine = BacktestEngine(
                strategy=strategy,
                commission_rate=commission,
                slippage_rate=self.base_slippage,
            )
            backtest_results = engine.run_backtest(data)

            # Calculate metrics
            metrics = PerformanceMetrics(
                backtest_results["trades"],
                backtest_results["portfolio_values"],
            )
            performance = metrics.calculate_all_metrics()

            results.append(
                {
                    "commission_rate": commission,
                    "metrics": performance,
                }
            )

        # Calculate impact on key metrics
        commission_impact = self._calculate_cost_impact(results)

        return {
            "results": results,
            "impact_analysis": commission_impact,
        }

    def _analyze_slippage_sensitivity(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        slippage_range: List[float],
    ) -> Dict:
        """
        Analyze sensitivity to different slippage rates.

        Args:
            strategy: Strategy to test
            data: Market data
            slippage_range: List of slippage rates to test

        Returns:
            Dict with slippage sensitivity analysis
        """
        results = []

        for slippage in slippage_range:
            # Run backtest with current slippage rate
            engine = BacktestEngine(
                strategy=strategy,
                commission_rate=self.base_commission,
                slippage_rate=slippage,
            )
            backtest_results = engine.run_backtest(data)

            # Calculate metrics
            metrics = PerformanceMetrics(
                backtest_results["trades"],
                backtest_results["portfolio_values"],
            )
            performance = metrics.calculate_all_metrics()

            results.append(
                {
                    "slippage_rate": slippage,
                    "metrics": performance,
                }
            )

        # Calculate impact on key metrics
        slippage_impact = self._calculate_cost_impact(results)

        return {
            "results": results,
            "impact_analysis": slippage_impact,
        }

    def _compare_impact_models(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        impact_models: List[str],
    ) -> Dict:
        """
        Compare different market impact models.

        Args:
            strategy: Strategy to test
            data: Market data
            impact_models: List of impact models to test

        Returns:
            Dict with impact model comparison
        """
        results = []

        for model in impact_models:
            # Run backtest with current impact model
            engine = BacktestEngine(
                strategy=strategy,
                commission_rate=self.base_commission,
                slippage_rate=self.base_slippage,
                slippage_model=model,
            )
            backtest_results = engine.run_backtest(data)

            # Calculate metrics
            metrics = PerformanceMetrics(
                backtest_results["trades"],
                backtest_results["portfolio_values"],
            )
            performance = metrics.calculate_all_metrics()

            results.append(
                {
                    "impact_model": model,
                    "metrics": performance,
                }
            )

        return {
            "results": results,
            "model_comparison": self._compare_model_performance(results),
        }

    def _calculate_cost_impact(self, results: List[Dict]) -> Dict:
        """
        Calculate impact of cost changes on key metrics.

        Args:
            results: List of backtest results

        Returns:
            Dict with impact analysis
        """
        # Extract metrics for analysis
        sharpe_ratios = [r["metrics"]["sharpe_ratio"] for r in results]
        returns = [r["metrics"]["total_return"] for r in results]
        win_rates = [r["metrics"]["win_rate"] for r in results]

        # Calculate impact statistics
        impact = {
            "sharpe_ratio": {
                "max_impact": max(sharpe_ratios) - min(sharpe_ratios),
                "pct_change": (min(sharpe_ratios) / max(sharpe_ratios) - 1)
                if max(sharpe_ratios) != 0
                else 0,
            },
            "total_return": {
                "max_impact": max(returns) - min(returns),
                "pct_change": (min(returns) / max(returns) - 1)
                if max(returns) != 0
                else 0,
            },
            "win_rate": {
                "max_impact": max(win_rates) - min(win_rates),
                "pct_change": (min(win_rates) / max(win_rates) - 1)
                if max(win_rates) != 0
                else 0,
            },
        }

        return impact

    def _compare_model_performance(self, results: List[Dict]) -> Dict:
        """
        Compare performance across different impact models.

        Args:
            results: List of backtest results

        Returns:
            Dict with model comparison
        """
        # Calculate relative performance
        base_model = results[0]  # Use first model as baseline
        relative_performance = []

        for result in results[1:]:
            relative = {
                "model": result["impact_model"],
                "sharpe_ratio_diff": (
                    result["metrics"]["sharpe_ratio"]
                    - base_model["metrics"]["sharpe_ratio"]
                ),
                "return_diff": (
                    result["metrics"]["total_return"]
                    - base_model["metrics"]["total_return"]
                ),
                "win_rate_diff": (
                    result["metrics"]["win_rate"] - base_model["metrics"]["win_rate"]
                ),
            }
            relative_performance.append(relative)

        return {
            "base_model": base_model["impact_model"],
            "relative_performance": relative_performance,
        }

    def generate_report(self) -> str:
        """Generate detailed cost sensitivity analysis report."""
        if not self.analysis_results:
            return "No analysis results available"

        report = []
        report.append("Transaction Cost Sensitivity Analysis Report")
        report.append("=" * 50)

        # Get latest results
        latest = self.analysis_results[-1]

        # Commission sensitivity
        report.append("\nCommission Rate Sensitivity:")
        commission_impact = latest["commission_sensitivity"]["impact_analysis"]
        report.append(
            f"Sharpe Ratio Impact: {commission_impact['sharpe_ratio']['max_impact']:.2f} "
            f"({commission_impact['sharpe_ratio']['pct_change']:.1%})"
        )
        report.append(
            f"Return Impact: {commission_impact['total_return']['max_impact']:.2%} "
            f"({commission_impact['total_return']['pct_change']:.1%})"
        )
        report.append(
            f"Win Rate Impact: {commission_impact['win_rate']['max_impact']:.2%} "
            f"({commission_impact['win_rate']['pct_change']:.1%})"
        )

        # Slippage sensitivity
        report.append("\nSlippage Rate Sensitivity:")
        slippage_impact = latest["slippage_sensitivity"]["impact_analysis"]
        report.append(
            f"Sharpe Ratio Impact: {slippage_impact['sharpe_ratio']['max_impact']:.2f} "
            f"({slippage_impact['sharpe_ratio']['pct_change']:.1%})"
        )
        report.append(
            f"Return Impact: {slippage_impact['total_return']['max_impact']:.2%} "
            f"({slippage_impact['total_return']['pct_change']:.1%})"
        )
        report.append(
            f"Win Rate Impact: {slippage_impact['win_rate']['max_impact']:.2%} "
            f"({slippage_impact['win_rate']['pct_change']:.1%})"
        )

        # Impact model comparison
        report.append("\nImpact Model Comparison:")
        model_comparison = latest["impact_model_comparison"]["model_comparison"]
        report.append(f"Base Model: {model_comparison['base_model']}")
        for perf in model_comparison["relative_performance"]:
            report.append(f"\n{perf['model']} vs Base Model:")
            report.append(f"Sharpe Ratio Diff: {perf['sharpe_ratio_diff']:.2f}")
            report.append(f"Return Diff: {perf['return_diff']:.2%}")
            report.append(f"Win Rate Diff: {perf['win_rate_diff']:.2%}")

        return "\n".join(report)
