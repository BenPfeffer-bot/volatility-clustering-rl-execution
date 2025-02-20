"""
Test script for strategy validation components.
Tests walk-forward optimization, Monte Carlo simulation, and cost sensitivity analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.config.paths import PROCESSED_DIR, OUTPUT_DIR
from src.backtesting.strategies.institutional import EnhancedInstitutionalStrategy
from src.backtesting.validation.runner import ValidationRunner
from src.backtesting.validation.walk_forward import WalkForwardOptimizer
from src.backtesting.validation.monte_carlo import MonteCarloSimulator
from src.backtesting.validation.cost_sensitivity import CostSensitivityAnalyzer
from src.backtesting.validation.oos import OutOfSampleTester
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


def load_test_data(ticker: str = "AAPL") -> pd.DataFrame:
    """Load test data for validation."""
    file_path = PROCESSED_DIR / f"{ticker}_features.csv"
    if not file_path.exists():
        logger.info("Test data not found. Generating synthetic data...")
        from src.scripts.generate_test_data import generate_test_data

        data = generate_test_data()
        data.to_csv(file_path)
        logger.info(f"Generated test data saved to {file_path}")
    else:
        data = pd.read_csv(file_path)
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data.set_index("timestamp", inplace=True)
        logger.info(f"Loaded test data from {file_path}")

    return data


def test_out_of_sample(
    strategy: EnhancedInstitutionalStrategy, data: pd.DataFrame
) -> None:
    """Test out-of-sample validation."""
    logger.info("\nTesting Out-of-Sample Validation...")

    # Initialize tester
    oos = OutOfSampleTester()

    # Define parameter ranges
    parameter_ranges = {
        "vpin_threshold": (0.5, 0.9),
        "min_holding_time": (5, 20),
        "max_holding_time": (20, 60),
        "stop_loss": (0.01, 0.03),
        "take_profit": (0.005, 0.015),
    }

    # Run OOS test
    results = oos.run_oos_test(strategy, data, parameter_ranges)

    # Log results
    logger.info("\nOOS Results:")
    analysis = results["analysis"]
    logger.info("Training Phase:")
    logger.info(f"Sharpe Ratio: {analysis['train_metrics']['sharpe_ratio']:.2f}")
    logger.info(f"Win Rate: {analysis['train_metrics']['win_rate']:.2%}")
    logger.info("Testing Phase:")
    logger.info(f"Sharpe Ratio: {analysis['test_metrics']['sharpe_ratio']:.2f}")
    logger.info(f"Win Rate: {analysis['test_metrics']['win_rate']:.2%}")
    logger.info("Performance Degradation:")
    logger.info(f"Sharpe Ratio: {analysis['degradation']['sharpe_ratio']:.1%}")
    logger.info(f"Win Rate: {analysis['degradation']['win_rate']:.1%}")


def test_walk_forward_optimization(
    strategy: EnhancedInstitutionalStrategy, data: pd.DataFrame
) -> None:
    """Test walk-forward optimization."""
    logger.info("\nTesting Walk-Forward Optimization...")

    # Initialize optimizer
    wfo = WalkForwardOptimizer()

    # Define parameter ranges
    parameter_ranges = {
        "vpin_threshold": (0.5, 0.9),
        "min_holding_time": (5, 20),
        "max_holding_time": (20, 60),
        "stop_loss": (0.01, 0.03),
        "take_profit": (0.005, 0.015),
    }

    # Run optimization
    results = wfo.run_wfo(strategy, data, parameter_ranges)

    # Log results
    logger.info("\nWFO Results:")
    logger.info(f"Number of Windows: {len(results['window_results'])}")
    logger.info(
        f"Average Test Sharpe: {np.mean([w['test_sharpe'] for w in results['window_results']]):.2f}"
    )
    logger.info(
        f"Average Test Win Rate: {np.mean([w['test_win_rate'] for w in results['window_results']]):.2%}"
    )


def test_monte_carlo_simulation(strategy: EnhancedInstitutionalStrategy) -> None:
    """Test Monte Carlo simulation."""
    logger.info("\nTesting Monte Carlo Simulation...")

    # Initialize simulator
    mc = MonteCarloSimulator()

    # Run simulations
    results = mc.run_simulations(strategy.trades)

    # Log results
    logger.info("\nMonte Carlo Results:")
    metrics = results["metrics"]
    logger.info(f"99th Percentile Drawdown: {metrics['worst_case']['drawdown']:.2%}")
    logger.info(f"1st Percentile Sharpe: {metrics['worst_case']['sharpe_ratio']:.2f}")
    logger.info(
        f"P(Drawdown > 10%): {metrics['probabilities']['drawdown_above_10pct']:.2%}"
    )


def test_cost_sensitivity(
    strategy: EnhancedInstitutionalStrategy, data: pd.DataFrame
) -> None:
    """Test transaction cost sensitivity analysis."""
    logger.info("\nTesting Cost Sensitivity Analysis...")

    # Initialize analyzer
    analyzer = CostSensitivityAnalyzer()

    # Run analysis
    results = analyzer.run_analysis(strategy, data)

    # Log results
    logger.info("\nCost Sensitivity Results:")
    commission = results["commission_sensitivity"]["impact_analysis"]
    logger.info(
        f"Commission Impact on Sharpe: {commission['sharpe_ratio']['max_impact']:.2f} "
        f"({commission['sharpe_ratio']['pct_change']:.1%})"
    )
    logger.info(
        f"Commission Impact on Return: {commission['total_return']['max_impact']:.2%} "
        f"({commission['total_return']['pct_change']:.1%})"
    )


def test_validation_runner(
    strategy: EnhancedInstitutionalStrategy, data: pd.DataFrame
) -> None:
    """Test complete validation pipeline."""
    logger.info("\nTesting Validation Runner...")

    # Initialize runner
    runner = ValidationRunner(strategy, data)

    # Run validation
    results = runner.run_validation()

    # Log results
    logger.info("\nValidation Results:")
    if "oos" in results:
        logger.info("✓ Out-of-Sample Testing completed")
    if "walk_forward" in results:
        logger.info("✓ Walk-Forward Optimization completed")
    if "monte_carlo" in results:
        logger.info("✓ Monte Carlo Simulation completed")
    if "cost_sensitivity" in results:
        logger.info("✓ Cost Sensitivity Analysis completed")


def main():
    """Main test execution."""
    logger.info("Starting validation system tests...")

    try:
        # Load test data
        data = load_test_data()

        # Initialize strategy
        strategy = EnhancedInstitutionalStrategy()

        # Run individual component tests
        test_out_of_sample(strategy, data)
        test_walk_forward_optimization(strategy, data)
        test_monte_carlo_simulation(strategy)
        test_cost_sensitivity(strategy, data)

        # Test complete validation pipeline
        test_validation_runner(strategy, data)

        logger.info("\nAll validation tests completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
