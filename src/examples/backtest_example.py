"""
Example script demonstrating how to use the institutional order flow trading strategy.

This example:
1. Processes features for AAPL
2. Trains the market impact model
3. Runs a backtest
4. Analyzes the results
"""

import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.data.process_features import FeatureEngineering
from src.models.tcn_impact import MarketImpactPredictor
from src.scripts.train_tcn import train_model
from src.scripts.run_backtest import run_backtest
from src.utils.log_utils import setup_logging
from argparse import Namespace

logger = setup_logging(__name__)


def main():
    """Run complete example pipeline."""
    # Configuration
    ticker = "AAPL"
    # Update based on data availability
    start_date = datetime(2023, 3, 1)
    end_date = datetime(2025, 2, 14)

    try:
        # 1. Process features
        logger.info("Step 1: Processing features...")
        fe = FeatureEngineering(ticker)
        fe.process_features()
        fe.save_features()

        # 2. Train market impact model
        logger.info("Step 2: Training market impact model...")
        model = train_model(ticker)

        # 3. Run backtest
        logger.info("Step 3: Running backtest...")
        args = Namespace(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_capital=1_000_000,
            transaction_cost=0.001,
            base_position=100_000,
        )

        results = run_backtest(args)

        # 4. Print summary
        logger.info("\n=== Strategy Performance ===")
        logger.info(f"Total Return: {results['total_return']:.2%}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Win Rate: {results['win_rate']:.2%}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"\nDetailed report available at: {results['report_dir']}")

        return 0

    except Exception as e:
        logger.error(f"Error in example: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
