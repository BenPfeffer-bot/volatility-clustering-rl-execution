"""
Test script for the backtesting system.
Tests all major components including strategy execution, metrics calculation,
and performance analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List
import traceback
import logging

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.config.paths import PROCESSED_DIR, OUTPUT_DIR
from src.backtesting.strategies.institutional import EnhancedInstitutionalStrategy
from src.backtesting.core.engine import BacktestEngine
from src.backtesting.core.metrics import BacktestMetrics
from src.backtesting.analysis.performance import PerformanceAnalyzer
from src.backtesting.analysis.drawdown import DrawdownMonitor
from src.backtesting.analysis.duration import TradeDurationAnalyzer
from src.backtesting.execution.impact import MarketImpactModel
from src.backtesting.optimization.position import PositionOptimizer
from src.backtesting.optimization.regime import RegimeDetector
from src.utils.log_utils import setup_logging
from src.backtesting.core.trade import Trade

logger = setup_logging(__name__)


def load_test_data(ticker: str = "AAPL") -> pd.DataFrame:
    """Load test data for backtesting."""
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


def test_strategy_execution(data: pd.DataFrame) -> tuple:
    """Test strategy execution and signal generation."""
    logger.info("\nTesting Strategy Execution...")

    # Initialize strategy
    strategy = EnhancedInstitutionalStrategy(
        vpin_threshold=0.7,
        min_holding_time=10,
        max_holding_time=30,
    )

    # Generate signals
    signals = strategy.generate_signals(data)
    n_signals = len(signals[signals != 0])
    logger.info(f"Generated {n_signals} trading signals")

    # Test position sizing
    size = strategy.size_position(1, data.iloc[0])
    logger.info(f"Test position size: {size:.2%}")

    return strategy, signals


def test_backtest_engine(
    strategy: EnhancedInstitutionalStrategy,
    data: pd.DataFrame,
) -> dict:
    """Test backtest engine execution."""
    logger.info("\nTesting Backtest Engine...")

    # Initialize engine
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=1_000_000,
        commission_rate=0.001,
    )

    # Run backtest
    results = engine.run_backtest(data)
    logger.info(f"Completed backtest with {len(results['trades'])} trades")

    return results


def test_metrics_calculation(
    trades: list,
    portfolio_values: pd.Series,
) -> BacktestMetrics:
    """Test performance metrics calculation."""
    logger.info("\nTesting Metrics Calculation...")

    metrics = BacktestMetrics()
    results = metrics.calculate_metrics(trades, portfolio_values)

    logger.info("Key Metrics:")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"Win Rate: {results['win_rate']:.2%}")
    logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")

    return metrics


def test_performance_analysis(
    trades: List[Trade],
    portfolio_values: pd.Series,
    market_data: pd.DataFrame,
    logger: logging.Logger,
) -> None:
    """Test performance analysis functionality."""
    logger.info("\nTesting Performance Analysis...")
    try:
        # Create analyzer
        analyzer = PerformanceAnalyzer()

        # Run analysis
        analysis_results = analyzer.analyze_performance(
            trades, portfolio_values, market_data
        )

        # Generate report
        report = analyzer.generate_performance_report(analysis_results)

        logger.info("Performance Analysis Results:")
        logger.info(report)

    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        logger.error("Stack trace:")
        traceback.print_exc()


def test_risk_monitoring(portfolio_values: pd.Series) -> None:
    """Test risk monitoring system."""
    logger.info("\nTesting Risk Monitoring...")

    monitor = DrawdownMonitor(max_drawdown_threshold=0.04)

    # Update monitor with portfolio values
    alerts = []
    for timestamp, value in portfolio_values.items():
        alert = monitor.update(timestamp, value)
        if alert:
            alerts.append(alert)

    stats = monitor.get_drawdown_statistics()
    logger.info("\nRisk Statistics:")
    for stat, value in stats.items():
        logger.info(f"{stat}: {value:.2%}")

    if alerts:
        logger.info(f"\nGenerated {len(alerts)} risk alerts")


def test_trade_duration_analysis(trades: list) -> None:
    """Test trade duration analysis."""
    logger.info("\nTesting Trade Duration Analysis...")

    analyzer = TradeDurationAnalyzer()
    for trade in trades:
        analyzer.add_trade(trade)

    analysis = analyzer.analyze_durations()
    logger.info("\nDuration Analysis:")
    logger.info(f"Average Duration: {analysis['avg_duration']:.1f} minutes")
    logger.info(f"Within Target (10-30 min): {analysis['pct_within_target']:.2%}")


def test_market_impact_model(data: pd.DataFrame) -> None:
    """Test market impact model."""
    logger.info("\nTesting Market Impact Model...")

    model = MarketImpactModel()

    # Test impact calculation
    test_trade = {
        "price": data["close"].iloc[0],
        "quantity": 1000,
        "market_data": {
            "daily_volatility": data["daily_volatility"].iloc[0],
            "volume": data["volume"].iloc[0],
            "vpin": data["vpin"].iloc[0],
            "regime": data["regime"].iloc[0],
        },
    }

    impact = model.estimate_total_impact(
        test_trade["price"],
        test_trade["quantity"],
        test_trade["market_data"],
    )

    logger.info("\nImpact Estimates:")
    for component, value in impact.items():
        logger.info(f"{component}: {value:.4%}")


def test_position_optimization(data: pd.DataFrame) -> None:
    """Test position optimization."""
    logger.info("\nTesting Position Optimization...")

    optimizer = PositionOptimizer()

    # Test position sizing
    size = optimizer.calculate_position_size(
        win_rate=0.7,
        volatility=data["daily_volatility"].iloc[0],
        market_data=data.iloc[0],
        regime=data["regime"].iloc[0],
    )

    logger.info(f"Optimal position size: {size:.2%}")


def test_regime_detection(data: pd.DataFrame) -> None:
    """Test regime detection."""
    logger.info("\nTesting Regime Detection...")

    detector = RegimeDetector()

    # Detect regimes
    regimes = detector.detect_regimes(data)
    flow_regimes = detector.classify_flow_regime(data)
    trend_regimes = detector.detect_trend_regime(data)

    regime_counts = regimes.value_counts()
    logger.info("\nRegime Distribution:")
    for regime, count in regime_counts.items():
        logger.info(f"{regime}: {count} periods ({count / len(regimes):.2%})")


def main():
    """Main test execution."""
    logger.info("Starting backtesting system tests...")

    try:
        # Load test data
        data = load_test_data()

        # Run all tests
        strategy, signals = test_strategy_execution(data)
        backtest_results = test_backtest_engine(strategy, data)
        metrics = test_metrics_calculation(
            backtest_results["trades"],
            backtest_results["portfolio_values"],
        )

        test_performance_analysis(
            backtest_results["trades"],
            backtest_results["portfolio_values"],
            data,
            logger,
        )

        test_risk_monitoring(backtest_results["portfolio_values"])
        test_trade_duration_analysis(backtest_results["trades"])
        test_market_impact_model(data)
        test_position_optimization(data)
        test_regime_detection(data)

        logger.info("\nAll tests completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
