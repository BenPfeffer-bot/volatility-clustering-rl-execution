"""
Out-of-sample testing framework for strategy validation.
Implements strict separation of training and testing data to validate strategy robustness.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from ..strategies.base import BaseStrategy
from ..core.engine import BacktestEngine
from ..performance.metrics import PerformanceMetrics
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class OutOfSampleTester:
    """
    Out-of-sample testing framework that:
    - Enforces strict data separation
    - Validates strategy robustness
    - Detects overfitting
    - Ensures consistent performance
    """

    def __init__(
        self,
        train_size: float = 0.75,  # 75% for training
        min_sharpe_ratio: float = 2.0,
        min_win_rate: float = 0.65,
        max_drawdown: float = 0.04,
        min_trades: int = 100,  # Minimum trades for statistical significance
    ):
        self.train_size = train_size
        self.min_sharpe_ratio = min_sharpe_ratio
        self.min_win_rate = min_win_rate
        self.max_drawdown = max_drawdown
        self.min_trades = min_trades
        self.test_results: List[Dict] = []

    def run_oos_test(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        parameter_ranges: Optional[Dict] = None,
    ) -> Dict:
        """
        Run out-of-sample test.

        Args:
            strategy: Strategy to test
            data: Market data with features
            parameter_ranges: Optional parameter ranges for optimization

        Returns:
            Dict with test results
        """
        logger.info("Starting out-of-sample testing...")

        # Split data into training and testing sets
        train_data, test_data = self._split_data(data)
        logger.info(
            f"Data split - Training: {len(train_data)} samples, Testing: {len(test_data)} samples"
        )

        # Train and optimize on training data
        if parameter_ranges:
            best_params = self._optimize_parameters(
                strategy, train_data, parameter_ranges
            )
            self._apply_parameters(strategy, best_params)

        # Run backtest on training data
        train_results = self._run_backtest(strategy, train_data)
        logger.info("Completed training phase backtest")

        # Run backtest on test data
        test_results = self._run_backtest(strategy, test_data)
        logger.info("Completed testing phase backtest")

        # Analyze results
        analysis = self._analyze_results(train_results, test_results)
        logger.info("Completed performance analysis")

        # Store results
        results = {
            "train_results": train_results,
            "test_results": test_results,
            "analysis": analysis,
        }
        self.test_results.append(results)

        return results

    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.

        Args:
            data: Full dataset

        Returns:
            Tuple of (training_data, testing_data)
        """
        split_idx = int(len(data) * self.train_size)
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()
        return train_data, test_data

    def _optimize_parameters(
        self,
        strategy: BaseStrategy,
        train_data: pd.DataFrame,
        parameter_ranges: Dict,
    ) -> Dict[str, float]:
        """
        Optimize strategy parameters on training data.

        Args:
            strategy: Strategy to optimize
            train_data: Training data
            parameter_ranges: Parameter ranges to test

        Returns:
            Dict of optimized parameters
        """
        from ..optimization.bayesian import BayesianOptimizer

        optimizer = BayesianOptimizer(param_bounds=parameter_ranges)

        def evaluate_params(params):
            # Update strategy parameters
            for name, value in params.items():
                setattr(strategy, name, value)

            # Run backtest
            engine = BacktestEngine(strategy=strategy)
            results = engine.run_backtest(train_data)

            # Calculate metrics
            metrics = PerformanceMetrics(
                results["trades"],
                results["portfolio_values"],
            )
            return metrics.calculate_all_metrics()

        best_params, _ = optimizer.optimize(evaluate_params)
        return best_params

    def _apply_parameters(
        self, strategy: BaseStrategy, parameters: Dict[str, float]
    ) -> None:
        """
        Apply parameters to strategy.

        Args:
            strategy: Strategy to update
            parameters: Parameters to apply
        """
        for name, value in parameters.items():
            setattr(strategy, name, value)

    def _run_backtest(self, strategy: BaseStrategy, data: pd.DataFrame) -> Dict:
        """
        Run backtest on dataset.

        Args:
            strategy: Strategy to test
            data: Data to test on

        Returns:
            Dict with backtest results
        """
        engine = BacktestEngine(strategy=strategy)
        results = engine.run_backtest(data)

        # Convert any Timestamp objects in results to ISO format strings
        if "trades" in results:
            for trade in results["trades"]:
                if hasattr(trade, "entry_time"):
                    trade.entry_time = (
                        trade.entry_time.isoformat()
                        if isinstance(trade.entry_time, pd.Timestamp)
                        else trade.entry_time
                    )
                if hasattr(trade, "exit_time"):
                    trade.exit_time = (
                        trade.exit_time.isoformat()
                        if isinstance(trade.exit_time, pd.Timestamp)
                        else trade.exit_time
                    )

        metrics = PerformanceMetrics(
            results["trades"],
            results["portfolio_values"],
        )
        performance = metrics.calculate_all_metrics()

        return {
            "trades": results["trades"],
            "portfolio_values": {
                k.isoformat() if isinstance(k, pd.Timestamp) else str(k): v
                for k, v in results["portfolio_values"].items()
            },
            "metrics": performance,
        }

    def _analyze_results(self, train_results: Dict, test_results: Dict) -> Dict:
        """
        Analyze training and testing results.

        Args:
            train_results: Training phase results
            test_results: Testing phase results

        Returns:
            Dict with analysis results
        """
        train_metrics = train_results["metrics"]
        test_metrics = test_results["metrics"]

        # Calculate performance degradation
        degradation = {
            "sharpe_ratio": (
                test_metrics["sharpe_ratio"] / train_metrics["sharpe_ratio"] - 1
                if train_metrics["sharpe_ratio"] != 0
                else 0
            ),
            "win_rate": (
                test_metrics["win_rate"] / train_metrics["win_rate"] - 1
                if train_metrics["win_rate"] != 0
                else 0
            ),
            "avg_return": (
                test_metrics["avg_trade_return"] / train_metrics["avg_trade_return"] - 1
                if train_metrics["avg_trade_return"] != 0
                else 0
            ),
        }

        # Check if strategy passes validation criteria
        passes_validation = (
            test_metrics["sharpe_ratio"] >= self.min_sharpe_ratio
            and test_metrics["win_rate"] >= self.min_win_rate
            and test_metrics["max_drawdown"] <= self.max_drawdown
            and len(test_results["trades"]) >= self.min_trades
            and all(d > -0.3 for d in degradation.values())  # Max 30% degradation
        )

        # Check for overfitting
        is_overfit = (
            train_metrics["sharpe_ratio"] > 1.5 * test_metrics["sharpe_ratio"]
            or train_metrics["win_rate"] > 1.3 * test_metrics["win_rate"]
            or abs(degradation["avg_return"]) > 0.4
        )

        return {
            "degradation": {k: float(v) for k, v in degradation.items()},
            "passes_validation": str(passes_validation),
            "is_overfit": str(is_overfit),
            "train_metrics": {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in train_metrics.items()
            },
            "test_metrics": {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in test_metrics.items()
            },
        }

    def generate_report(self) -> str:
        """Generate detailed out-of-sample analysis report."""
        if not self.test_results:
            return "No test results available"

        report = []
        report.append("Out-of-Sample Testing Report")
        report.append("=" * 50)

        # Get latest results
        latest = self.test_results[-1]
        analysis = latest["analysis"]

        # Training phase results
        report.append("\nTraining Phase Results:")
        train_metrics = analysis["train_metrics"]
        report.append(f"Sharpe Ratio: {train_metrics['sharpe_ratio']:.2f}")
        report.append(f"Win Rate: {train_metrics['win_rate']:.2%}")
        report.append(f"Average Return: {train_metrics['avg_trade_return']:.2%}")
        report.append(f"Max Drawdown: {train_metrics['max_drawdown']:.2%}")
        report.append(f"Number of Trades: {train_metrics['total_trades']}")

        # Testing phase results
        report.append("\nTesting Phase Results:")
        test_metrics = analysis["test_metrics"]
        report.append(f"Sharpe Ratio: {test_metrics['sharpe_ratio']:.2f}")
        report.append(f"Win Rate: {test_metrics['win_rate']:.2%}")
        report.append(f"Average Return: {test_metrics['avg_trade_return']:.2%}")
        report.append(f"Max Drawdown: {test_metrics['max_drawdown']:.2%}")
        report.append(f"Number of Trades: {test_metrics['total_trades']}")

        # Performance degradation
        report.append("\nPerformance Degradation:")
        degradation = analysis["degradation"]
        report.append(f"Sharpe Ratio: {degradation['sharpe_ratio']:.1%}")
        report.append(f"Win Rate: {degradation['win_rate']:.1%}")
        report.append(f"Average Return: {degradation['avg_return']:.1%}")

        # Validation status
        report.append("\nValidation Status:")
        report.append(
            f"Passes Validation: {'Yes' if analysis['passes_validation'] else 'No'}"
        )
        report.append(
            f"Overfitting Detected: {'Yes' if analysis['is_overfit'] else 'No'}"
        )

        if not analysis["passes_validation"]:
            report.append("\nFailure Reasons:")
            if test_metrics["sharpe_ratio"] < self.min_sharpe_ratio:
                report.append(f"- Sharpe ratio below minimum ({self.min_sharpe_ratio})")
            if test_metrics["win_rate"] < self.min_win_rate:
                report.append(f"- Win rate below minimum ({self.min_win_rate:.0%})")
            if test_metrics["max_drawdown"] > self.max_drawdown:
                report.append(f"- Max drawdown above limit ({self.max_drawdown:.0%})")
            if len(latest["test_results"]["trades"]) < self.min_trades:
                report.append(f"- Insufficient trades ({self.min_trades} required)")
            if any(d <= -0.3 for d in degradation.values()):
                report.append("- Excessive performance degradation (>30%)")

        return "\n".join(report)

    def calculate_oos_ratio(self, train_metrics: Dict, test_metrics: Dict) -> float:
        """Calculate out-of-sample performance ratio."""
        try:
            # Extract key metrics with validation
            train_sharpe = float(train_metrics.get("sharpe_ratio", 0.0))
            test_sharpe = float(test_metrics.get("sharpe_ratio", 0.0))
            train_return = float(train_metrics.get("total_return", 0.0))
            test_return = float(test_metrics.get("total_return", 0.0))

            # Handle NaN/Inf values
            if (
                np.isnan(train_sharpe)
                or np.isinf(train_sharpe)
                or np.isnan(test_sharpe)
                or np.isinf(test_sharpe)
            ):
                logger.warning("Invalid Sharpe ratios detected in OOS calculation")
                return 0.0

            if (
                np.isnan(train_return)
                or np.isinf(train_return)
                or np.isnan(test_return)
                or np.isinf(test_return)
            ):
                logger.warning("Invalid returns detected in OOS calculation")
                return 0.0

            # Calculate performance ratio with validation
            if abs(train_sharpe) < 1e-6 or abs(train_return) < 1e-6:
                logger.warning(
                    "Training metrics too close to zero for meaningful OOS ratio"
                )
                return 0.0

            # Calculate composite OOS ratio
            sharpe_ratio = test_sharpe / train_sharpe if train_sharpe != 0 else 0.0
            return_ratio = test_return / train_return if train_return != 0 else 0.0

            # Weighted average of Sharpe and return ratios
            oos_ratio = 0.7 * sharpe_ratio + 0.3 * return_ratio

            # Validate final ratio
            if np.isnan(oos_ratio) or np.isinf(oos_ratio):
                return 0.0

            # Cap extreme values
            return max(min(oos_ratio, 2.0), 0.0)

        except Exception as e:
            logger.error(f"Error calculating OOS ratio: {e}")
            return 0.0
