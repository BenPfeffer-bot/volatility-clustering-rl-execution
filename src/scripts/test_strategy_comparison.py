"""
Test script for comparing different trading strategies.
Compares the Enhanced Institutional Strategy against benchmark strategies.
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
from src.backtesting.performance.comparison import StrategyComparison
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class VWAPStrategy(EnhancedInstitutionalStrategy):
    """Simple VWAP execution strategy for comparison."""

    def __init__(self):
        super().__init__(vpin_threshold=0.5)
        self.vwap_window = 30

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on VWAP crossover."""
        vwap = (data["close"] * data["volume"]).rolling(self.vwap_window).sum() / data[
            "volume"
        ].rolling(self.vwap_window).sum()

        signals = pd.Series(0, index=data.index)
        signals[data["close"] > vwap * 1.001] = 1  # Buy when price > VWAP + 0.1%
        signals[data["close"] < vwap * 0.999] = -1  # Sell when price < VWAP - 0.1%

        return signals


class MeanReversionStrategy(EnhancedInstitutionalStrategy):
    """Simple mean reversion strategy for comparison."""

    def __init__(self):
        super().__init__(vpin_threshold=0.5)
        self.ma_window = 20
        self.std_window = 20
        self.entry_z = 2.0

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on mean reversion."""
        ma = data["close"].rolling(self.ma_window).mean()
        std = data["close"].rolling(self.std_window).std()
        z_score = (data["close"] - ma) / std

        signals = pd.Series(0, index=data.index)
        signals[z_score > self.entry_z] = -1  # Sell when overbought
        signals[z_score < -self.entry_z] = 1  # Buy when oversold

        return signals


def load_test_data(ticker: str = "AAPL") -> pd.DataFrame:
    """Load test data for strategy comparison."""
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


def compare_strategies(data: pd.DataFrame) -> None:
    """Compare different trading strategies."""
    logger.info("Starting strategy comparison...")

    # Initialize strategies
    strategies = {
        "Enhanced Institutional": EnhancedInstitutionalStrategy(),
        "VWAP": VWAPStrategy(),
        "Mean Reversion": MeanReversionStrategy(),
    }

    # Initialize comparison framework
    comparison = StrategyComparison()

    # Run comparison
    results = comparison.compare_strategies(strategies, data)

    # Generate comparison report
    report = comparison.generate_comparison_report(results)
    logger.info("\nStrategy Comparison Report:")
    logger.info(report)

    # Generate plots
    output_dir = OUTPUT_DIR / "strategy_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.plot_strategy_comparison(
        results, save_path=str(output_dir / "strategy_comparison.png")
    )


def analyze_regime_performance(data: pd.DataFrame, results: dict) -> None:
    """Analyze strategy performance across different market regimes."""
    logger.info("\nAnalyzing regime-specific performance...")

    regimes = data["regime"].unique()

    for strategy_name, result in results.items():
        trades = result["trades"]
        logger.info(f"\n{strategy_name} Performance by Regime:")

        for regime in regimes:
            regime_trades = [t for t in trades if t.regime == regime]
            if regime_trades:
                win_rate = sum(1 for t in regime_trades if t.pnl > 0) / len(
                    regime_trades
                )
                avg_return = np.mean([t.pnl / t.initial_value for t in regime_trades])

                logger.info(f"{regime}:")
                logger.info(f"  Win Rate: {win_rate:.2%}")
                logger.info(f"  Avg Return: {avg_return:.2%}")
                logger.info(f"  Trade Count: {len(regime_trades)}")


def analyze_vpin_sensitivity(data: pd.DataFrame, results: dict) -> None:
    """Analyze strategy performance sensitivity to VPIN levels."""
    logger.info("\nAnalyzing VPIN sensitivity...")

    vpin_ranges = [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)]

    for strategy_name, result in results.items():
        trades = result["trades"]
        logger.info(f"\n{strategy_name} Performance by VPIN Range:")

        for vpin_min, vpin_max in vpin_ranges:
            range_trades = [t for t in trades if vpin_min <= t.vpin_entry < vpin_max]

            if range_trades:
                win_rate = sum(1 for t in range_trades if t.pnl > 0) / len(range_trades)
                avg_return = np.mean([t.pnl / t.initial_value for t in range_trades])

                logger.info(f"VPIN [{vpin_min:.1f}, {vpin_max:.1f}):")
                logger.info(f"  Win Rate: {win_rate:.2%}")
                logger.info(f"  Avg Return: {avg_return:.2%}")
                logger.info(f"  Trade Count: {len(range_trades)}")


def main():
    """Main execution."""
    logger.info("Starting strategy comparison tests...")

    try:
        # Load test data
        data = load_test_data()

        # Compare strategies
        compare_strategies(data)

        # Additional analysis
        strategies = {
            "Enhanced Institutional": EnhancedInstitutionalStrategy(),
            "VWAP": VWAPStrategy(),
            "Mean Reversion": MeanReversionStrategy(),
        }
        comparison = StrategyComparison()
        results = comparison.compare_strategies(strategies, data)

        analyze_regime_performance(data, results)
        analyze_vpin_sensitivity(data, results)

        logger.info("\nAll comparison tests completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
