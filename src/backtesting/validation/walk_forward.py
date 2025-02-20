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
from src.models.tcn_impact import MarketImpactPredictor
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class WalkForwardOptimizer:
    """
    Walk-forward optimization framework that:
    - Uses rolling windows for training and testing
    - Prevents overfitting through out-of-sample validation
    - Tracks performance stability across time periods
    - Retrains TCN model for each window
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
        self.tcn_model = MarketImpactPredictor()

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

            # Retrain TCN model on training data
            logger.info(f"Retraining TCN model for window starting {start_date}")
            self.tcn_model.train(train_df=train_data, epochs=25)

            # Update market impact predictions
            train_data["market_impact_pred"] = self.tcn_model.predict(train_data)
            test_data["market_impact_pred"] = self.tcn_model.predict(test_data)

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
                "tcn_performance": {
                    "train_mse": self.tcn_model.evaluate(train_data),
                    "test_mse": self.tcn_model.evaluate(test_data),
                },
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
        if data.empty:
            logger.warning("Empty data provided for window optimization")
            return {
                name: (low + high) / 2 for name, (low, high) in parameter_ranges.items()
            }

        from ..optimization.bayesian import BayesianOptimizer

        # Initialize optimizer with improved kernel parameters
        optimizer = BayesianOptimizer(
            param_bounds=parameter_ranges,
            n_iterations=50,
            exploration_weight=0.1,
            kernel_params={
                "length_scale": 1.0,
                "length_scale_bounds": (0.01, 100.0),
                "constant_value": 1.0,
                "constant_value_bounds": (0.01, 100.0),
                "noise": 1e-4,
            },
        )

        # Define evaluation function with validation
        def evaluate_params(params):
            try:
                # Update strategy parameters
                for name, value in params.items():
                    setattr(strategy, name, value)

                # Run backtest
                engine = BacktestEngine(strategy=strategy)
                results = engine.run_backtest(data)

                if not results or not results.get("trades"):
                    return {
                        "sharpe_ratio": 0.0,
                        "win_rate": 0.0,
                        "avg_trade_return": 0.0,
                        "max_drawdown": 1.0,
                    }

                # Calculate metrics with validation
                metrics = PerformanceMetrics(
                    results["trades"], results["portfolio_values"]
                )
                all_metrics = metrics.calculate_all_metrics()

                # Handle NaN/Inf values
                for key in [
                    "sharpe_ratio",
                    "win_rate",
                    "avg_trade_return",
                    "max_drawdown",
                ]:
                    if (
                        key not in all_metrics
                        or np.isnan(all_metrics[key])
                        or np.isinf(all_metrics[key])
                    ):
                        all_metrics[key] = 0.0 if key != "max_drawdown" else 1.0

                return all_metrics

            except Exception as e:
                logger.warning(f"Error in parameter evaluation: {e}")
                return {
                    "sharpe_ratio": 0.0,
                    "win_rate": 0.0,
                    "avg_trade_return": 0.0,
                    "max_drawdown": 1.0,
                }

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
        if data.empty:
            logger.warning("Empty data provided for parameter evaluation")
            return {
                "train_metrics": {"sharpe_ratio": 0.0, "win_rate": 0.0},
                "test_metrics": {"sharpe_ratio": 0.0, "win_rate": 0.0},
                "trades": [],
                "portfolio_values": pd.Series(dtype=float),
            }

        try:
            # Update strategy parameters
            for name, value in parameters.items():
                setattr(strategy, name, value)

            # Run backtest
            engine = BacktestEngine(strategy=strategy)
            results = engine.run_backtest(data)

            if not results or not results.get("trades"):
                return {
                    "train_metrics": {"sharpe_ratio": 0.0, "win_rate": 0.0},
                    "test_metrics": {"sharpe_ratio": 0.0, "win_rate": 0.0},
                    "trades": [],
                    "portfolio_values": pd.Series(dtype=float),
                }

            # Calculate metrics with validation
            metrics = PerformanceMetrics(results["trades"], results["portfolio_values"])
            all_metrics = metrics.calculate_all_metrics()

            # Handle NaN/Inf values
            for key in all_metrics:
                if np.isnan(all_metrics[key]) or np.isinf(all_metrics[key]):
                    all_metrics[key] = 0.0

            return {
                "train_metrics": all_metrics,
                "test_metrics": all_metrics,
                "trades": results["trades"],
                "portfolio_values": results["portfolio_values"],
            }

        except Exception as e:
            logger.warning(f"Error in parameter evaluation: {e}")
            return {
                "train_metrics": {"sharpe_ratio": 0.0, "win_rate": 0.0},
                "test_metrics": {"sharpe_ratio": 0.0, "win_rate": 0.0},
                "trades": [],
                "portfolio_values": pd.Series(dtype=float),
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
            return {
                "sharpe_stability": 0.0,
                "win_rate_stability": 0.0,
                "parameter_stability": 0.0,
            }

        try:
            prev, curr = window_results[-2:]

            # Calculate stability metrics with validation
            prev_sharpe = float(prev.get("test_sharpe", 0.0))
            curr_sharpe = float(curr.get("test_sharpe", 0.0))
            if np.isnan(prev_sharpe) or np.isinf(prev_sharpe):
                prev_sharpe = 0.0
            if np.isnan(curr_sharpe) or np.isinf(curr_sharpe):
                curr_sharpe = 0.0

            sharpe_stability = abs(curr_sharpe - prev_sharpe) / max(
                abs(prev_sharpe), 1e-6
            )

            prev_win_rate = float(prev.get("test_win_rate", 0.0))
            curr_win_rate = float(curr.get("test_win_rate", 0.0))
            if np.isnan(prev_win_rate) or np.isinf(prev_win_rate):
                prev_win_rate = 0.0
            if np.isnan(curr_win_rate) or np.isinf(curr_win_rate):
                curr_win_rate = 0.0

            win_rate_stability = abs(curr_win_rate - prev_win_rate)

            # Parameter stability with validation
            param_changes = []
            for p in curr.get("parameters", {}):
                prev_val = float(prev.get("parameters", {}).get(p, 0.0))
                curr_val = float(curr.get("parameters", {}).get(p, 0.0))
                if np.isnan(prev_val) or np.isinf(prev_val):
                    prev_val = 0.0
                if np.isnan(curr_val) or np.isinf(curr_val):
                    curr_val = 0.0
                if abs(prev_val) > 1e-6:
                    param_changes.append(abs(curr_val - prev_val) / abs(prev_val))

            param_stability = np.mean(param_changes) if param_changes else 0.0

            return {
                "sharpe_stability": float(sharpe_stability),
                "win_rate_stability": float(win_rate_stability),
                "parameter_stability": float(param_stability),
            }

        except Exception as e:
            logger.warning(f"Error calculating stability metrics: {e}")
            return {
                "sharpe_stability": 0.0,
                "win_rate_stability": 0.0,
                "parameter_stability": 0.0,
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
