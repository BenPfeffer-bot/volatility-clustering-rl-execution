"""
Walk-forward optimization framework for strategy validation.
Implements rolling window optimization to prevent overfitting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from ..strategies.base import BaseStrategy
from ..core.engine import BacktestEngine
from ..performance.metrics import PerformanceMetrics
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class WalkForwardOptimizer:
    """
    Walk-forward optimization framework that:
    - Uses rolling windows for training and testing
    - Prevents overfitting through out-of-sample validation
    - Tracks performance stability across time periods
    """

    def __init__(
        self,
        train_window: int = 90,  # 3 months training
        test_window: int = 30,  # 1 month testing
        min_sharpe: float = 1.5,
        min_win_rate: float = 0.65,
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.min_sharpe = min_sharpe
        self.min_win_rate = min_win_rate
        self.results_history: List[Dict] = []

    def run_wfo(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        parameter_ranges: Dict[str, Tuple[float, float]],
    ) -> Dict:
        """
        Run walk-forward optimization.

        Args:
            strategy: Strategy to optimize
            data: Market data with features
            parameter_ranges: Dict of parameter names to (min, max) ranges

        Returns:
            Dict with optimization results
        """
        logger.info("Starting walk-forward optimization...")

        # Initialize results tracking
        all_window_results = []
        performance_stability = []

        # Create rolling windows
        start_dates = pd.date_range(
            data.index[0], data.index[-self.test_window], freq=f"{self.test_window}D"
        )

        # Process each window
        for start_date in start_dates:
            # Define window boundaries
            train_start = start_date
            train_end = start_date + timedelta(days=self.train_window)
            test_end = train_end + timedelta(days=self.test_window)

            # Get window data
            train_data = data[train_start:train_end]
            test_data = data[train_end:test_end]

            # Optimize on training data
            best_params = self._optimize_window(strategy, train_data, parameter_ranges)

            # Test on out-of-sample data
            train_results = self._evaluate_parameters(strategy, train_data, best_params)
            test_results = self._evaluate_parameters(strategy, test_data, best_params)

            # Store results
            window_result = {
                "window_start": start_date.isoformat(),
                "train_metrics": train_results["train_metrics"],
                "test_metrics": test_results["test_metrics"],
                "parameters": best_params,
            }

            # Extract key metrics for convenience
            window_result.update(
                {
                    "train_sharpe": train_results["train_metrics"]["sharpe_ratio"],
                    "test_sharpe": test_results["test_metrics"]["sharpe_ratio"],
                    "train_win_rate": train_results["train_metrics"]["win_rate"],
                    "test_win_rate": test_results["test_metrics"]["win_rate"],
                }
            )

            all_window_results.append(window_result)

            # Calculate stability metrics
            if len(all_window_results) > 1:
                stability = self._calculate_stability(all_window_results[-2:])
                performance_stability.append(stability)

            logger.info(f"Completed window starting {start_date}")

        # Analyze results
        analysis = self._analyze_wfo_results(all_window_results, performance_stability)

        return {
            "window_results": all_window_results,
            "stability_metrics": performance_stability,
            "analysis": analysis,
        }

    def _optimize_window(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        parameter_ranges: Dict[str, Tuple[float, float]],
    ) -> Dict[str, float]:
        """
        Optimize strategy parameters for a single window.

        Args:
            strategy: Strategy to optimize
            data: Training data
            parameter_ranges: Parameter ranges to test

        Returns:
            Dict of optimized parameters
        """
        from ..optimization.bayesian import BayesianOptimizer

        # Initialize optimizer
        optimizer = BayesianOptimizer(param_bounds=parameter_ranges, n_iterations=50)

        # Define evaluation function
        def evaluate_params(params):
            # Update strategy parameters
            for name, value in params.items():
                setattr(strategy, name, value)

            # Run backtest
            engine = BacktestEngine(strategy=strategy)
            results = engine.run_backtest(data)

            # Calculate metrics
            metrics = PerformanceMetrics(results["trades"], results["portfolio_values"])
            return metrics.calculate_all_metrics()

        # Run optimization
        best_params, _ = optimizer.optimize(evaluate_params)
        return best_params

    def _evaluate_parameters(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        parameters: Dict[str, float],
    ) -> Dict:
        """
        Evaluate parameter set on test data.

        Args:
            strategy: Strategy to evaluate
            data: Test data
            parameters: Parameters to test

        Returns:
            Dict of performance metrics
        """
        # Update strategy parameters
        for name, value in parameters.items():
            setattr(strategy, name, value)

        # Run backtest
        engine = BacktestEngine(strategy=strategy)
        results = engine.run_backtest(data)

        # Calculate metrics
        metrics = PerformanceMetrics(results["trades"], results["portfolio_values"])
        all_metrics = metrics.calculate_all_metrics()

        return {
            "train_metrics": all_metrics,  # Store metrics under train_metrics key
            "test_metrics": all_metrics,  # Store metrics under test_metrics key
            "trades": results["trades"],
            "portfolio_values": results["portfolio_values"],
        }

    def _calculate_stability(
        self,
        window_results: List[Dict],
    ) -> Dict[str, float]:
        """
        Calculate stability metrics between consecutive windows.

        Args:
            window_results: List of window results

        Returns:
            Dict of stability metrics
        """
        if len(window_results) < 2:
            return {}

        prev, curr = window_results[-2:]

        # Calculate stability metrics
        sharpe_stability = abs(curr["test_sharpe"] - prev["test_sharpe"]) / max(
            abs(prev["test_sharpe"]), 1e-6
        )
        win_rate_stability = abs(curr["test_win_rate"] - prev["test_win_rate"])
        param_stability = np.mean(
            [
                abs(curr["parameters"][p] - prev["parameters"][p])
                / abs(prev["parameters"][p])
                for p in curr["parameters"]
            ]
        )

        return {
            "sharpe_stability": sharpe_stability,
            "win_rate_stability": win_rate_stability,
            "parameter_stability": param_stability,
        }

    def _analyze_wfo_results(
        self,
        window_results: List[Dict],
        stability_metrics: List[Dict],
    ) -> Dict:
        """
        Analyze walk-forward optimization results.

        Args:
            window_results: Results from each window
            stability_metrics: Stability metrics between windows

        Returns:
            Dict with analysis results
        """
        # Calculate average metrics
        avg_train_sharpe = np.mean([r["train_sharpe"] for r in window_results])
        avg_test_sharpe = np.mean([r["test_sharpe"] for r in window_results])
        avg_train_win_rate = np.mean([r["train_win_rate"] for r in window_results])
        avg_test_win_rate = np.mean([r["test_win_rate"] for r in window_results])

        # Calculate stability averages
        if stability_metrics:
            avg_sharpe_stability = np.mean(
                [m["sharpe_stability"] for m in stability_metrics]
            )
            avg_win_rate_stability = np.mean(
                [m["win_rate_stability"] for m in stability_metrics]
            )
            avg_param_stability = np.mean(
                [m["parameter_stability"] for m in stability_metrics]
            )
        else:
            avg_sharpe_stability = avg_win_rate_stability = avg_param_stability = 0.0

        # Check for overfitting
        is_overfit = (
            avg_train_sharpe > 1.5 * avg_test_sharpe
            or avg_train_win_rate > 1.3 * avg_test_win_rate
        )

        # Convert boolean values to strings
        passes_validation = (
            avg_test_sharpe >= self.min_sharpe
            and avg_test_win_rate >= self.min_win_rate
            and not is_overfit
        )

        return {
            "avg_metrics": {
                "train_sharpe": float(avg_train_sharpe),
                "test_sharpe": float(avg_test_sharpe),
                "train_win_rate": float(avg_train_win_rate),
                "test_win_rate": float(avg_test_win_rate),
            },
            "stability": {
                "sharpe_stability": float(avg_sharpe_stability),
                "win_rate_stability": float(avg_win_rate_stability),
                "parameter_stability": float(avg_param_stability),
            },
            "is_overfit": str(is_overfit),
            "passes_validation": str(passes_validation),
        }

    def generate_report(self) -> str:
        """Generate detailed WFO analysis report."""
        if not self.results_history:
            return "No WFO results available"

        report = []
        report.append("Walk-Forward Optimization Report")
        report.append("=" * 50)

        # Add summary statistics
        latest_results = self.results_history[-1]
        report.append("\nLatest Results:")
        report.append(f"Training Sharpe: {latest_results['train_sharpe']:.2f}")
        report.append(f"Test Sharpe: {latest_results['test_sharpe']:.2f}")
        report.append(f"Training Win Rate: {latest_results['train_win_rate']:.2%}")
        report.append(f"Test Win Rate: {latest_results['test_win_rate']:.2%}")

        # Add stability metrics
        report.append("\nStability Metrics:")
        stability = latest_results.get("stability", {})
        report.append(f"Sharpe Stability: {stability.get('sharpe_stability', 0):.2f}")
        report.append(
            f"Win Rate Stability: {stability.get('win_rate_stability', 0):.2f}"
        )
        report.append(
            f"Parameter Stability: {stability.get('parameter_stability', 0):.2f}"
        )

        # Add validation status
        report.append("\nValidation Status:")
        report.append(
            f"Passes Validation: {'Yes' if latest_results.get('passes_validation') else 'No'}"
        )
        report.append(
            f"Overfitting Detected: {'Yes' if latest_results.get('is_overfit') else 'No'}"
        )

        return "\n".join(report)
