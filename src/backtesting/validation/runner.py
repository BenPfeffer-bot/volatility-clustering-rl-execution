"""
Validation runner that orchestrates all validation steps.
Implements comprehensive strategy validation pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from ..strategies.base import BaseStrategy
from .walk_forward import WalkForwardOptimizer
from .monte_carlo import MonteCarloSimulator
from .cost_sensitivity import CostSensitivityAnalyzer
from .oos import OutOfSampleTester
from src.config.paths import OUTPUT_DIR, BACKTESTS_DIR
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class ValidationRunner:
    """
    Validation runner that:
    - Orchestrates all validation steps
    - Generates comprehensive reports
    - Tracks validation history
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        output_dir: Optional[Path] = None,
    ):
        self.strategy = strategy
        self.data = data
        self.output_dir = output_dir or BACKTESTS_DIR / "validation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validation_history: List[Dict] = []

    def run_validation(
        self,
        parameter_ranges: Optional[Dict] = None,
        skip_steps: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run complete validation pipeline.

        Args:
            parameter_ranges: Dict of parameter ranges for optimization
            skip_steps: List of validation steps to skip

        Returns:
            Dict with validation results
        """
        logger.info("Starting validation pipeline...")
        skip_steps = skip_steps or []
        results = {}

        # 1. Out-of-Sample Testing
        if "oos" not in skip_steps:
            logger.info("Running out-of-sample testing...")
            oos = OutOfSampleTester()
            oos_results = oos.run_oos_test(
                self.strategy,
                self.data,
                parameter_ranges or self._get_default_parameter_ranges(),
            )
            results["oos"] = oos_results
            self._save_results("oos", oos_results)

        # 2. Walk-Forward Optimization
        if "wfo" not in skip_steps:
            logger.info("Running walk-forward optimization...")
            wfo = WalkForwardOptimizer()
            wfo_results = wfo.run_wfo(
                self.strategy,
                self.data,
                parameter_ranges or self._get_default_parameter_ranges(),
            )
            results["walk_forward"] = wfo_results
            self._save_results("walk_forward", wfo_results)

        # 3. Monte Carlo Simulation
        if "monte_carlo" not in skip_steps:
            logger.info("Running Monte Carlo simulations...")
            mc = MonteCarloSimulator()
            mc_results = mc.run_simulations(self.strategy.trades)
            results["monte_carlo"] = mc_results
            self._save_results("monte_carlo", mc_results)

        # 4. Transaction Cost Sensitivity
        if "cost_sensitivity" not in skip_steps:
            logger.info("Running transaction cost sensitivity analysis...")
            cost_analyzer = CostSensitivityAnalyzer()
            cost_results = cost_analyzer.run_analysis(self.strategy, self.data)
            results["cost_sensitivity"] = cost_results
            self._save_results("cost_sensitivity", cost_results)

        # Store validation results
        self.validation_history.append(
            {
                "timestamp": datetime.now(),
                "results": results,
            }
        )

        # Generate and save report
        report = self.generate_validation_report(results)
        self._save_report(report)

        return results

    def _get_default_parameter_ranges(self) -> Dict:
        """Get default parameter ranges for optimization."""
        return {
            "vpin_threshold": (0.5, 0.9),
            "min_holding_time": (5, 20),
            "max_holding_time": (20, 60),
            "stop_loss": (0.01, 0.03),
            "take_profit": (0.005, 0.015),
        }

    def _save_results(self, step: str, results: Dict) -> None:
        """
        Save validation results to file.

        Args:
            step: Validation step name
            results: Results to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = self.output_dir / f"{step}_{timestamp}.json"

        import json

        with open(file_path, "w") as f:
            # Convert numpy types to Python types
            results_json = self._convert_to_json_serializable(results)
            json.dump(results_json, f, indent=2)

        logger.info(f"Saved {step} results to {file_path}")

    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {
                str(k): self._convert_to_json_serializable(v) for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(x) for x in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, bool):
            return bool(obj)  # Explicitly convert to Python bool
        elif hasattr(
            obj, "to_dict"
        ):  # Handle Trade objects and other custom classes with to_dict method
            return obj.to_dict()
        return obj

    def _save_report(self, report: str) -> None:
        """
        Save validation report to file.

        Args:
            report: Report text to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = self.output_dir / f"validation_report_{timestamp}.txt"

        with open(file_path, "w") as f:
            f.write(report)

        logger.info(f"Saved validation report to {file_path}")

    def generate_validation_report(self, results: Dict) -> str:
        """
        Generate comprehensive validation report.

        Args:
            results: Validation results

        Returns:
            Formatted report string
        """
        report = []
        report.append("Strategy Validation Report")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now()}\n")

        # Out-of-Sample Testing Results
        if "oos" in results:
            oos = results["oos"]
            report.append("Out-of-Sample Testing")
            report.append("-" * 30)

            analysis = oos["analysis"]
            train_metrics = analysis["train_metrics"]
            test_metrics = analysis["test_metrics"]

            report.append("\nTraining Phase:")
            report.append(f"Sharpe Ratio: {train_metrics['sharpe_ratio']:.2f}")
            report.append(f"Win Rate: {train_metrics['win_rate']:.2%}")
            report.append(f"Average Return: {train_metrics['avg_trade_return']:.2%}")

            report.append("\nTesting Phase:")
            report.append(f"Sharpe Ratio: {test_metrics['sharpe_ratio']:.2f}")
            report.append(f"Win Rate: {test_metrics['win_rate']:.2%}")
            report.append(f"Average Return: {test_metrics['avg_trade_return']:.2%}")

            report.append("\nPerformance Degradation:")
            degradation = analysis["degradation"]
            report.append(f"Sharpe Ratio: {degradation['sharpe_ratio']:.1%}")
            report.append(f"Win Rate: {degradation['win_rate']:.1%}")
            report.append(f"Average Return: {degradation['avg_return']:.1%}")
            report.append("")

        # Walk-Forward Optimization Results
        if "walk_forward" in results:
            wfo = results["walk_forward"]
            report.append("Walk-Forward Optimization")
            report.append("-" * 30)
            report.append(f"Number of Windows: {len(wfo['window_results'])}")

            # Calculate average metrics
            avg_metrics = {
                "train_sharpe": np.mean(
                    [w["train_sharpe"] for w in wfo["window_results"]]
                ),
                "test_sharpe": np.mean(
                    [w["test_sharpe"] for w in wfo["window_results"]]
                ),
                "train_win_rate": np.mean(
                    [w["train_win_rate"] for w in wfo["window_results"]]
                ),
                "test_win_rate": np.mean(
                    [w["test_win_rate"] for w in wfo["window_results"]]
                ),
            }

            report.append("\nAverage Metrics:")
            report.append(f"Training Sharpe: {avg_metrics['train_sharpe']:.2f}")
            report.append(f"Test Sharpe: {avg_metrics['test_sharpe']:.2f}")
            report.append(f"Training Win Rate: {avg_metrics['train_win_rate']:.2%}")
            report.append(f"Test Win Rate: {avg_metrics['test_win_rate']:.2%}")

            # Add stability metrics
            stability = wfo["analysis"]["stability"]
            report.append("\nStability Metrics:")
            report.append(
                f"Parameter Stability: {stability['parameter_stability']:.2f}"
            )
            report.append(f"Performance Stability: {stability['sharpe_stability']:.2f}")
            report.append("")

        # Monte Carlo Simulation Results
        if "monte_carlo" in results:
            mc = results["monte_carlo"]
            report.append("Monte Carlo Simulation")
            report.append("-" * 30)

            metrics = mc["metrics"]
            report.append("\nWorst-Case Analysis:")
            report.append(
                f"99th Percentile Drawdown: {metrics['worst_case']['drawdown']:.2%}"
            )
            report.append(
                f"1st Percentile Sharpe: {metrics['worst_case']['sharpe_ratio']:.2f}"
            )
            report.append(
                f"Max Consecutive Losses: {metrics['worst_case']['consecutive_losses']}"
            )

            report.append("\nProbability Analysis:")
            probs = metrics["probabilities"]
            report.append(f"P(Drawdown > 10%): {probs['drawdown_above_10pct']:.2%}")
            report.append(f"P(Sharpe < 1): {probs['sharpe_below_1']:.2%}")
            report.append(
                f"P(Consecutive Losses > 5): {probs['consecutive_losses_above_5']:.2%}"
            )
            report.append("")

        # Transaction Cost Sensitivity Results
        if "cost_sensitivity" in results:
            cost = results["cost_sensitivity"]
            report.append("Transaction Cost Sensitivity")
            report.append("-" * 30)

            # Commission impact
            commission = cost["commission_sensitivity"]["impact_analysis"]
            report.append("\nCommission Rate Impact:")
            report.append(
                f"Sharpe Ratio: {commission['sharpe_ratio']['max_impact']:.2f} "
                f"({commission['sharpe_ratio']['pct_change']:.1%})"
            )
            report.append(
                f"Total Return: {commission['total_return']['max_impact']:.2%} "
                f"({commission['total_return']['pct_change']:.1%})"
            )

            # Slippage impact
            slippage = cost["slippage_sensitivity"]["impact_analysis"]
            report.append("\nSlippage Rate Impact:")
            report.append(
                f"Sharpe Ratio: {slippage['sharpe_ratio']['max_impact']:.2f} "
                f"({slippage['sharpe_ratio']['pct_change']:.1%})"
            )
            report.append(
                f"Total Return: {slippage['total_return']['max_impact']:.2%} "
                f"({slippage['total_return']['pct_change']:.1%})"
            )

        # Overall validation summary
        report.append("\nValidation Summary")
        report.append("-" * 30)

        # Check if strategy passes all validation criteria
        passes_validation = (
            results.get("oos", {}).get("analysis", {}).get("passes_validation", False)
            and results.get("walk_forward", {})
            .get("analysis", {})
            .get("passes_validation", False)
            and results.get("monte_carlo", {})
            .get("metrics", {})
            .get("probabilities", {})
            .get("sharpe_below_1", 1.0)
            < 0.2
            and results.get("cost_sensitivity", {})
            .get("commission_sensitivity", {})
            .get("impact_analysis", {})
            .get("total_return", {})
            .get("pct_change", -1.0)
            > -0.3
        )

        report.append(f"Passes Validation: {'Yes' if passes_validation else 'No'}")

        if not passes_validation:
            report.append("\nFailure Reasons:")
            if "oos" in results and not results["oos"]["analysis"].get(
                "passes_validation"
            ):
                report.append("- Failed out-of-sample validation")
            if "walk_forward" in results and not results["walk_forward"][
                "analysis"
            ].get("passes_validation"):
                report.append("- Failed walk-forward optimization criteria")
            if (
                "monte_carlo" in results
                and results["monte_carlo"]["metrics"]["probabilities"].get(
                    "sharpe_below_1", 1.0
                )
                >= 0.2
            ):
                report.append(
                    "- High probability of poor performance in Monte Carlo simulation"
                )
            if (
                "cost_sensitivity" in results
                and results["cost_sensitivity"]["commission_sensitivity"][
                    "impact_analysis"
                ]["total_return"]["pct_change"]
                <= -0.3
            ):
                report.append("- Excessive sensitivity to transaction costs")

        return "\n".join(report)
