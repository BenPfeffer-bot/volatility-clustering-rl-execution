"""
Run backtests using real market data from processed feature files.
Implements institutional order flow impact trading strategy with comprehensive validation.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import torch

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.backtesting.strategies.institutional import EnhancedInstitutionalStrategy
from src.backtesting.experiments import ExperimentManager
from src.config.paths import PROCESSED_DIR, BACKTESTS_DIR
from src.utils.log_utils import setup_logging
from src.models.tcn_impact import MarketImpactTCN, MarketImpactPredictor
from src.data.process_features import FeatureEngineering
from src.backtesting.optimization.bayesian import BayesianOptimizer
from src.backtesting.optimization.position import PositionOptimizer

logger = setup_logging(__name__, level=logging.INFO)


def get_date_range(interval: str = "1M") -> tuple[str, str]:
    """
    Get start and end dates based on reference interval.

    Args:
        interval: Reference interval string (1D, 1W, 1M, 3M, 1Y)
                 D=Day, W=Week, M=Month, Y=Year

    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format
    """
    end_date = datetime.now()

    interval_map = {
        "1D": timedelta(days=1),
        "1W": timedelta(weeks=1),
        "1M": timedelta(days=30),
        "3M": timedelta(days=90),
        "1Y": timedelta(days=365),
    }

    if interval not in interval_map:
        raise ValueError(
            f"Invalid interval. Must be one of {list(interval_map.keys())}"
        )

    start_date = end_date - interval_map[interval]

    return (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))


def load_stock_data(
    symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load processed feature data for a given stock symbol within specified date range.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with processed features
    """
    file_path = PROCESSED_DIR / f"{symbol}_features.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"No data file found for {symbol}")

    logger.info(f"Loading data for {symbol}...")

    # Initialize feature engineering
    feature_eng = FeatureEngineering(symbol)
    data = feature_eng.load_data(symbol)

    # Process all features
    data = feature_eng.process_features()

    # Validate features
    feature_eng.validate_processed_features()

    # Set timestamp as index
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data.set_index("timestamp", inplace=True)

    # Filter by date range if specified
    if start_date:
        data = data[data.index >= pd.Timestamp(start_date)]
    if end_date:
        data = data[data.index <= pd.Timestamp(end_date)]

    return data


def run_backtest_on_symbol(
    symbol: str,
    initial_capital: float = 1_000_000,
    output_dir: Path = BACKTESTS_DIR / "real_data",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: Optional[str] = None,
    validation_config: Optional[Dict] = None,
) -> Dict:
    """
    Run backtest for a single symbol with enhanced features and validation.
    """
    # If interval is provided, use it to set date range
    if interval:
        start_date, end_date = get_date_range(interval)

    # Load and process data
    data = load_stock_data(symbol, start_date, end_date)

    # Initialize TCN model
    tcn_model = MarketImpactPredictor(
        sequence_length=30, learning_rate=0.0005, batch_size=128
    )

    # Initialize Bayesian optimizer for position sizing
    bayesian_optimizer = BayesianOptimizer(
        param_bounds={
            "vpin_threshold": (0.5, 0.9),
            "position_size": (0.1, 0.5),
            "stop_loss": (0.005, 0.02),
            "take_profit": (0.003, 0.01),
        },
        n_iterations=50,
        exploration_weight=0.1,
    )

    # Initialize Position Optimizer with Kelly Criterion
    position_optimizer = PositionOptimizer(
        max_position_size=0.1, kelly_fraction=0.5, min_win_rate=0.55, vol_target=0.15
    )

    # Initialize strategy with enhanced components
    strategy = EnhancedInstitutionalStrategy(
        vpin_threshold=0.7,
        min_holding_time=5,
        max_holding_time=15,
        stop_loss=0.01,
        take_profit=0.005,
    )

    # Initialize experiment manager with comprehensive validation
    experiment_mgr = ExperimentManager(
        initial_capital=initial_capital,
        output_dir=output_dir / symbol,
        max_drawdown_limit=0.04,
    )

    # Run comprehensive analysis
    logger.info(f"Running backtest for {symbol}...")
    results = experiment_mgr.run_comprehensive_analysis(
        strategy=strategy,
        data=data,
        parameter_ranges=bayesian_optimizer.param_bounds,
    )

    # Generate and save detailed report
    report = experiment_mgr.generate_detailed_report()
    report_path = output_dir / symbol / "detailed_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)

    return results


def run_multi_symbol_backtest(
    symbols: List[str] = None,
    initial_capital: float = 10_000_000,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Run backtests on multiple symbols with comprehensive validation.
    """
    # If interval is provided, use it to set date range
    if interval:
        start_date, end_date = get_date_range(interval)

    # If no symbols specified, get all available symbols from processed dir
    if symbols is None:
        symbols = [f.stem.split("_")[0] for f in PROCESSED_DIR.glob("*_features.csv")]

    # Calculate per-symbol capital
    per_symbol_capital = initial_capital / len(symbols)

    # Create output directory with timestamp
    output_dir = (
        BACKTESTS_DIR / f"real_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define validation configuration
    validation_config = {
        "wfo": {
            "train_window": "3M",
            "test_window": "1M",
            "min_sharpe": 1.5,
            "min_win_rate": 0.65,
        },
        "oos": {"train_size": 0.8, "min_performance_ratio": 0.7},
        "monte_carlo": {"n_simulations": 10000, "confidence_level": 0.95},
        "cost_sensitivity": {
            "commission_range": np.linspace(0.0001, 0.001, 10),
            "slippage_range": np.linspace(0.0001, 0.001, 10),
        },
        "performance_targets": {
            "min_sharpe": 2.5,
            "min_win_rate": 0.68,
            "max_drawdown": 0.04,
            "min_trades": 100,
        },
    }

    # Run backtests with enhanced validation
    results = {}
    for symbol in symbols:
        try:
            symbol_results = run_backtest_on_symbol(
                symbol=symbol,
                initial_capital=per_symbol_capital,
                output_dir=output_dir,
                start_date=start_date,
                end_date=end_date,
                validation_config=validation_config,
            )
            results[symbol] = symbol_results

            # Log progress with enhanced metrics
            logger.info(f"Completed backtest for {symbol}")

            # Extract key metrics
            perf_metrics = symbol_results.get("performance", {}).get(
                "performance_metrics", {}
            )
            validation_metrics = symbol_results.get("validation", {})

            logger.info(f"{symbol} Results:")
            logger.info(f"  Total Return: {perf_metrics.get('total_return', 0):.2%}")
            logger.info(f"  Sharpe Ratio: {perf_metrics.get('sharpe_ratio', 0):.2f}")
            logger.info(f"  Win Rate: {perf_metrics.get('win_rate', 0):.2%}")
            logger.info(
                f"  OOS Performance Ratio: {validation_metrics.get('oos_performance_ratio', 0):.2f}"
            )
            logger.info(
                f"  MC VaR (95%): {validation_metrics.get('monte_carlo_var', 0):.2%}"
            )

        except Exception as e:
            logger.error(f"Error running backtest for {symbol}: {str(e)}")
            continue

    # Generate enhanced summary report
    generate_summary_report(results, output_dir, validation_config)

    return results


def generate_summary_report(
    results: Dict[str, Dict], output_dir: Path, validation_config: Dict
) -> None:
    """Generate summary report comparing performance across symbols."""
    summary_data = []

    for symbol, result in results.items():
        perf_metrics = result.get("performance", {}).get("performance_metrics", {})
        risk_metrics = result.get("risk_monitoring", {})
        validation_metrics = result.get("validation", {})

        summary_data.append(
            {
                "Symbol": symbol,
                "Total Return": perf_metrics.get("total_return", 0),
                "Sharpe Ratio": perf_metrics.get("sharpe_ratio", 0),
                "Win Rate": perf_metrics.get("win_rate", 0),
                "Volatility": perf_metrics.get("volatility", 0),
                "Max Drawdown": risk_metrics.get("max_drawdown", 0)
                if isinstance(risk_metrics, dict)
                else 0,
                "Total Trades": perf_metrics.get("total_trades", 0),
                "OOS Performance Ratio": validation_metrics.get(
                    "oos_performance_ratio", 0
                ),
                "MC VaR (95%)": validation_metrics.get("monte_carlo_var", 0),
            }
        )

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_df.sort_values("Sharpe Ratio", ascending=False, inplace=True)

    # Save summary
    summary_path = output_dir / "backtest_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Generate text report
    report = ["Multi-Symbol Backtest Summary", "=" * 50, ""]

    # Overall statistics
    report.extend(
        [
            "Overall Statistics:",
            f"Total Symbols Tested: {len(results)}",
            f"Average Sharpe Ratio: {summary_df['Sharpe Ratio'].mean():.2f}",
            f"Average Win Rate: {summary_df['Win Rate'].mean():.2%}",
            f"Average Return: {summary_df['Total Return'].mean():.2%}",
            "",
            "Top 5 Performing Symbols (by Sharpe):",
            "-----------------------------------",
        ]
    )

    # Add top 5 symbols
    for _, row in summary_df.head().iterrows():
        report.append(
            f"{row['Symbol']}:"
            f" Sharpe={row['Sharpe Ratio']:.2f},"
            f" Return={row['Total Return']:.2%},"
            f" Win Rate={row['Win Rate']:.2%}"
        )

    # Save text report
    report_path = output_dir / "backtest_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report))


if __name__ == "__main__":
    # Run backtest on all available symbols with reference interval
    results = run_multi_symbol_backtest(
        ["AAPL"],  # Test with AAPL first
        interval="1W",  # Use last 1 month of data
    )
    logger.info(
        "Completed all backtests. Check the output directory for detailed results."
    )
