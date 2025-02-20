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
import json

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.backtesting.strategies.institutional import EnhancedInstitutionalStrategy
from src.backtesting.experiments import ExperimentManager
from src.config.paths import PROCESSED_DIR, BACKTESTS_DIR, MODEL_WEIGHTS_DIR
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


def save_model_checkpoint(
    model: MarketImpactPredictor,
    symbol: str,
    window_start: str,
    metrics: Dict,
    output_dir: Path,
) -> None:
    """
    Save model checkpoint with performance metrics.

    Args:
        model: Trained model instance
        symbol: Stock symbol
        window_start: Start date of the window
        metrics: Performance metrics
        output_dir: Output directory
    """
    # Create model directory
    model_dir = output_dir / "models" / symbol
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model_path = model_dir / f"tcn_{window_start}.pt"
    model.save_model(str(model_path))

    # Save metrics
    metrics_path = model_dir / f"metrics_{window_start}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Saved model checkpoint for {symbol} at {window_start}")


def load_model_checkpoint(
    symbol: str,
    window_start: str,
    output_dir: Path,
) -> tuple[Optional[MarketImpactPredictor], Optional[Dict]]:
    """
    Load model checkpoint and metrics.

    Args:
        symbol: Stock symbol
        window_start: Start date of the window
        output_dir: Output directory

    Returns:
        Tuple of (model, metrics) if found, else (None, None)
    """
    model_dir = output_dir / "models" / symbol
    model_path = model_dir / f"tcn_{window_start}.pt"
    metrics_path = model_dir / f"metrics_{window_start}.json"

    if not model_path.exists() or not metrics_path.exists():
        return None, None

    try:
        # Load model
        model = MarketImpactPredictor()
        model.load_model(str(model_path))

        # Load metrics
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        return model, metrics
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        return None, None


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

    # Initialize strategy with enhanced parameters
    strategy = EnhancedInstitutionalStrategy(
        vpin_threshold=0.7,
        min_holding_time=5,
        max_holding_time=30,
        stop_loss=0.02,
        take_profit=0.015,
        vol_window=100,
        trend_window=20,
    )

    # Initialize experiment manager with comprehensive validation
    experiment_mgr = ExperimentManager(
        initial_capital=initial_capital,
        output_dir=output_dir / symbol,
        max_drawdown_limit=0.05,
    )

    # Track model performance across windows
    model_performance_history = []

    # Run walk-forward optimization with model persistence
    logger.info(f"Running walk-forward optimization for {symbol}...")

    # Convert string time windows to days
    train_window_days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365}[
        validation_config["wfo"]["train_window"]
    ]

    test_window_days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365}[
        validation_config["wfo"]["test_window"]
    ]

    # Create rolling windows using days
    window_size = pd.Timedelta(days=train_window_days)
    test_size = pd.Timedelta(days=test_window_days)

    start_dates = pd.date_range(
        data.index[0], data.index[-1] - test_size, freq=test_size
    )

    for window_start in start_dates:
        window_end = window_start + window_size
        test_end = window_end + test_size

        # Get window data
        train_data = data[window_start:window_end]
        test_data = data[window_end:test_end]

        # Try to load existing model checkpoint
        saved_model, saved_metrics = load_model_checkpoint(
            symbol,
            window_start.strftime("%Y%m%d"),
            output_dir,
        )

        if saved_model is not None:
            logger.info(f"Loading saved model for window {window_start}")
            tcn_model = saved_model
            model_performance_history.append(saved_metrics)
        else:
            # Train new model
            logger.info(f"Training new model for window {window_start}")
            tcn_model.train(train_df=train_data, epochs=25)

            # Evaluate model
            train_mse = tcn_model.evaluate(train_data)
            test_mse = tcn_model.evaluate(test_data)

            # Save checkpoint
            metrics = {
                "window_start": window_start.strftime("%Y%m%d"),
                "train_mse": train_mse,
                "test_mse": test_mse,
                "timestamp": datetime.now().isoformat(),
            }
            save_model_checkpoint(
                tcn_model, symbol, window_start.strftime("%Y%m%d"), metrics, output_dir
            )
            model_performance_history.append(metrics)

        # Update market impact predictions
        train_data["market_impact_pred"] = tcn_model.predict(train_data)
        test_data["market_impact_pred"] = tcn_model.predict(test_data)

    # Run comprehensive analysis
    logger.info(f"Running comprehensive analysis for {symbol}...")
    results = experiment_mgr.run_comprehensive_analysis(
        strategy=strategy,
        data=data,
        parameter_ranges=bayesian_optimizer.param_bounds,
    )

    # Add model performance history to results
    results["model_performance"] = {
        "history": model_performance_history,
        "avg_train_mse": np.mean([m["train_mse"] for m in model_performance_history]),
        "avg_test_mse": np.mean([m["test_mse"] for m in model_performance_history]),
        "mse_stability": np.std([m["test_mse"] for m in model_performance_history]),
    }

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
            "train_window": "3M",  # Use 3 months for training
            "test_window": "1M",  # Test on 1 month
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
            "min_sharpe": 2.0,  # Slightly lower target for initial testing
            "min_win_rate": 0.65,  # Adjusted from 0.68
            "max_drawdown": 0.05,  # Slightly relaxed from 0.04
            "min_trades": 50,  # Ensure sufficient trades for statistical significance
        },
    }

    # Track overall model performance
    overall_model_performance = {}

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

            # Track model performance
            overall_model_performance[symbol] = symbol_results["model_performance"]

            # Log progress with enhanced metrics
            logger.info(f"Completed backtest for {symbol}")

            # Extract key metrics
            perf_metrics = symbol_results.get("performance", {}).get(
                "performance_metrics", {}
            )
            validation_metrics = symbol_results.get("validation", {})
            model_metrics = symbol_results["model_performance"]

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
            logger.info(f"  Avg Model MSE: {model_metrics['avg_test_mse']:.6f}")

        except Exception as e:
            logger.error(f"Error running backtest for {symbol}: {str(e)}")
            continue

    # Add model performance to results
    results["overall_model_performance"] = overall_model_performance

    # Generate enhanced summary report
    generate_summary_report(results, output_dir, validation_config)

    return results


def generate_summary_report(results, output_dir, validation_config):
    """Generate a summary report of backtest results across all symbols."""
    summary_data = []
    for symbol, result in results.items():
        if symbol == "overall_model_performance":
            continue

        # Extract metrics from the correct nested structure
        perf_metrics = result.get("performance", {}).get("performance_metrics", {})
        validation_metrics = result.get("validation", {})
        model_metrics = result.get("model_performance", {})

        summary_data.append(
            {
                "Symbol": symbol,
                "Total_Return": perf_metrics.get("total_return", 0) * 100,
                "Sharpe_Ratio": perf_metrics.get("sharpe_ratio", 0),
                "Win_Rate": perf_metrics.get("win_rate", 0) * 100,
                "OOS_Performance": validation_metrics.get("oos_performance_ratio", 0),
                "VaR_95": validation_metrics.get("monte_carlo_var", 0) * 100,
                "MSE_Stability": model_metrics.get("mse_stability", 0),
            }
        )

    summary_df = pd.DataFrame(summary_data)

    # Handle empty or all-NaN cases
    if summary_df.empty:
        logger.warning("No valid results to generate summary report")
        return

    # Replace NaN with 0 for numerical comparisons
    summary_df = summary_df.fillna(0)

    # Find best performing metrics with validation
    best_return_symbol = (
        summary_df.loc[summary_df["Total_Return"].idxmax(), "Symbol"]
        if not summary_df["Total_Return"].empty
        else "None"
    )
    best_sharpe_symbol = (
        summary_df.loc[summary_df["Sharpe_Ratio"].idxmax(), "Symbol"]
        if not summary_df["Sharpe_Ratio"].empty
        else "None"
    )
    most_stable_symbol = (
        summary_df.loc[summary_df["MSE_Stability"].idxmin(), "Symbol"]
        if not summary_df["MSE_Stability"].empty
        else "None"
    )

    # Generate summary text
    summary_text = [
        "Backtest Summary Report",
        "=====================",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Symbols Tested: {', '.join(symbol for symbol in results.keys() if symbol != 'overall_model_performance')}",
        "",
        "Best Performers:",
        f"Best Return: {best_return_symbol} ({summary_df.loc[summary_df['Total_Return'].idxmax(), 'Total_Return']:.2f}%)",
        f"Best Sharpe: {best_sharpe_symbol} ({summary_df.loc[summary_df['Sharpe_Ratio'].idxmax(), 'Sharpe_Ratio']:.2f})",
        f"Most Stable Model: {most_stable_symbol}",
        "",
        "Average Performance Metrics:",
        f"Avg Return: {summary_df['Total_Return'].mean():.2f}%",
        f"Avg Sharpe: {summary_df['Sharpe_Ratio'].mean():.2f}",
        f"Avg Win Rate: {summary_df['Win_Rate'].mean():.2f}%",
        f"Avg OOS Performance: {summary_df['OOS_Performance'].mean():.2f}",
        "",
        "Risk Metrics:",
        f"Avg VaR (95%): {summary_df['VaR_95'].mean():.2f}%",
        f"Worst Return: {summary_df['Total_Return'].min():.2f}%",
        "",
        "Detailed Results:",
        summary_df.to_string(),
    ]

    # Save summary report
    summary_path = os.path.join(output_dir, "summary_report.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_text))

    logger.info(f"Summary report generated at {summary_path}")


if __name__ == "__main__":
    # Run backtest on selected symbols with longer timeframe
    results = run_multi_symbol_backtest(
        symbols=["CSCO", "DD", "GE", "GSK", "HSBC", "IBM", "MCD", "SHEL", "VIXY"],
        initial_capital=10_000_000,  # 10M initial capital
        interval="1M",  # Use last 1 month of data for initial testing
    )
    logger.info(
        "Completed all backtests. Check the output directory for detailed results."
    )
