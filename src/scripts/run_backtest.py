"""
Entrypoint script for running the institutional order flow trading strategy backtest.

Usage:
    python -m src.scripts.run_backtest --ticker AAPL --start_date 2022-01-01 --end_date 2023-01-01
"""

import os
import sys
import argparse
from datetime import datetime
import pandas as pd
from pathlib import Path

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.config.paths import PROCESSED_DIR, MODELS_DIR, BACKTESTS_DIR
from src.config.settings import DJ_TITANS_50_TICKER
from src.models.tcn_impact import MarketImpactPredictor
from src.execution.optimizer import ExecutionOptimizer, ExecutionConfig
from src.execution.position_sizer import PositionConfig
from src.backtesting.engine import Backtester, BacktestConfig
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run institutional order flow strategy backtest"
    )

    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker to backtest",
    )

    parser.add_argument(
        "--start_date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        required=True,
        help="Start date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end_date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        required=True,
        help="End date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--initial_capital",
        type=float,
        default=1_000_000,
        help="Initial capital for backtest",
    )

    parser.add_argument(
        "--transaction_cost",
        type=float,
        default=0.001,
        help="Transaction cost in basis points",
    )

    parser.add_argument(
        "--base_position",
        type=float,
        default=100_000,
        help="Base position size",
    )

    return parser.parse_args()


def load_market_impact_model(ticker: str) -> MarketImpactPredictor:
    """Load trained market impact model."""
    model_path = MODELS_DIR / f"tcn_{ticker}.pt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found for {ticker}. Please run train_tcn.py first."
        )

    model = MarketImpactPredictor()
    model.load_model(str(model_path))
    logger.info(f"Loaded market impact model from {model_path}")

    return model


def load_feature_data(
    ticker: str, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """Load and prepare feature data for backtesting."""
    feature_path = PROCESSED_DIR / f"{ticker}_features.csv"

    if not feature_path.exists():
        raise FileNotFoundError(
            f"No feature data found for {ticker}. Please run process_features.py first."
        )

    # Load data
    df = pd.read_csv(feature_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Filter date range
    mask = (df["timestamp"] >= pd.Timestamp(start_date)) & (
        df["timestamp"] <= pd.Timestamp(end_date)
    )
    df = df[mask].copy()

    if len(df) == 0:
        raise ValueError(
            f"No data found for {ticker} between {start_date} and {end_date}"
        )

    logger.info(f"Loaded {len(df)} rows of feature data for {ticker}")
    return df


def setup_configs(args) -> tuple:
    """Setup configuration objects for backtest."""
    # Backtest configuration
    backtest_config = BacktestConfig(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        transaction_cost=args.transaction_cost,
    )

    # Position sizing configuration
    position_config = PositionConfig(
        base_position=args.base_position,
        min_position=args.base_position * 0.1,  # 10% of base
        max_position=args.base_position * 2.0,  # 200% of base
        vpin_threshold=0.7,
        vol_lookback=21,
        risk_limit=0.02,
    )

    # Execution configuration
    execution_config = ExecutionConfig(
        ticker=args.ticker,
        position_config=position_config,
        min_vpin_signal=0.7,
        max_adverse_impact=0.02,
        min_hurst_threshold=0.55,
        max_positions=3,
        regime_window=50,
        volatility_threshold=0.02,
    )

    return backtest_config, execution_config


def run_backtest(args):
    """Run backtest with specified configuration."""
    logger.info(f"Starting backtest for {args.ticker}")
    logger.info(f"Period: {args.start_date.date()} to {args.end_date.date()}")

    try:
        # Load market impact model
        model = load_market_impact_model(args.ticker)

        # Load feature data
        data = load_feature_data(args.ticker, args.start_date, args.end_date)

        # Setup configurations
        backtest_config, execution_config = setup_configs(args)

        # Initialize backtester
        backtester = Backtester(
            config=backtest_config,
            execution_config=execution_config,
            market_impact_model=model,
        )

        # Run backtest
        results = backtester.run(data)

        # Log summary results
        logger.info("Backtest completed successfully")
        logger.info(f"Total Return: {results['total_return']:.2%}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Win Rate: {results['win_rate']:.2%}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Performance report available at: {results['report_dir']}")

        return results

    except Exception as e:
        logger.error(f"Error during backtest: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()

    # Validate ticker
    if args.ticker not in DJ_TITANS_50_TICKER:
        raise ValueError(
            f"Invalid ticker: {args.ticker}. Must be one of {DJ_TITANS_50_TICKER}"
        )

    # Run backtest
    results = run_backtest(args)

    # Return success
    return 0


if __name__ == "__main__":
    sys.exit(main())
