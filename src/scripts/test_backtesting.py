"""
Test script to verify all Part 5 backtesting components.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.config.paths import PROCESSED_DIR, OUTPUT_DIR
from src.backtesting.strategies.institutional import EnhancedInstitutionalStrategy
from src.backtesting.performance.comparison import StrategyComparison
from src.backtesting.performance.metrics import PerformanceMetrics
from src.backtesting.analysis.drawdown import DrawdownMonitor
from src.backtesting.analysis.duration import TradeDurationAnalyzer
from src.backtesting.analysis.visualization import BacktestVisualizer
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


def load_test_data(ticker: str = "AAPL") -> pd.DataFrame:
    """Load processed feature data for testing."""
    file_path = PROCESSED_DIR / f"{ticker}_features.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"No processed data found for {ticker}")

    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df


def test_strategy_execution():
    """Test the enhanced institutional strategy execution."""
    logger.info("Testing strategy execution...")

    # Load test data
    data = load_test_data()

    # Initialize strategy
    strategy = EnhancedInstitutionalStrategy()

    # Generate signals
    signals = strategy.generate_signals(data)
    logger.info(f"Generated {len(signals[signals != 0])} trading signals")

    return strategy, signals


def test_performance_metrics(trades, portfolio_values):
    """Test performance metrics calculation."""
    logger.info("Testing performance metrics...")

    metrics = PerformanceMetrics(trades, portfolio_values)
    results = metrics.calculate_all_metrics()

    logger.info("Performance Metrics:")
    for metric, value in results.items():
        logger.info(f"{metric}: {value}")

    return results


def test_drawdown_monitoring(portfolio_values):
    """Test drawdown monitoring system."""
    logger.info("Testing drawdown monitoring...")

    monitor = DrawdownMonitor()

    # Update monitor with portfolio values
    alerts = []
    for timestamp, value in portfolio_values.items():
        alert = monitor.update(timestamp, value)
        if alert:
            alerts.append(alert)

    stats = monitor.get_drawdown_statistics()
    logger.info("Drawdown Statistics:")
    for stat, value in stats.items():
        logger.info(f"{stat}: {value}")

    return monitor, alerts


def test_duration_analysis(trades):
    """Test trade duration analysis."""
    logger.info("Testing duration analysis...")

    analyzer = TradeDurationAnalyzer()
    for trade in trades:
        analyzer.add_trade(trade)

    analysis = analyzer.analyze_durations()
    suggestions = analyzer.get_optimization_suggestions()

    logger.info("Duration Analysis:")
    for metric, value in analysis.items():
        logger.info(f"{metric}: {value}")

    logger.info("Optimization Suggestions:")
    for suggestion in suggestions:
        logger.info(f"- {suggestion}")

    return analyzer


def test_visualization(portfolio_values, trades, drawdowns):
    """Test visualization components."""
    logger.info("Testing visualization components...")

    viz = BacktestVisualizer(OUTPUT_DIR / "test_results")
    report_dir = viz.generate_performance_report(
        portfolio_values=portfolio_values, trades=trades, drawdowns=drawdowns
    )

    logger.info(f"Generated visualization report at {report_dir}")
    return viz


def test_strategy_comparison(data):
    """Test strategy comparison framework."""
    logger.info("Testing strategy comparison...")

    # Create strategies for comparison
    strategies = {
        "enhanced_institutional": EnhancedInstitutionalStrategy(),
    }

    comparison = StrategyComparison()
    results = comparison.compare_strategies(strategies, data)

    report = comparison.generate_comparison_report(results)
    logger.info("\nStrategy Comparison Report:")
    logger.info(report)

    return comparison, results


def main():
    """Main test execution."""
    logger.info("Starting backtesting system tests...")

    try:
        # Load test data
        data = load_test_data()

        # Test strategy execution
        strategy, signals = test_strategy_execution()

        # Run backtest
        comparison = StrategyComparison()
        trades, portfolio_values = comparison._run_backtest(strategy, data)

        # Calculate drawdowns
        drawdowns = pd.Series(index=portfolio_values.index)
        peak = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - peak) / peak

        # Test all components
        metrics_results = test_performance_metrics(trades, portfolio_values)
        monitor, alerts = test_drawdown_monitoring(portfolio_values)
        duration_analyzer = test_duration_analysis(trades)
        visualizer = test_visualization(portfolio_values, trades, drawdowns)
        comparison_results = test_strategy_comparison(data)

        logger.info("All tests completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
